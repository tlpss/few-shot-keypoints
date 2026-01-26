""" Code taken from https://arxiv.org/pdf/2306.03881 and slightly modified"""

import os
import gc
import random
import json
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import PILToTensor
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers import DDIMScheduler
from diffusers import StableDiffusionPipeline

from few_shot_keypoints.featurizers.base import BaseFeaturizer
from few_shot_keypoints.featurizers.registry import FeaturizerRegistry

class MyUNet2DConditionModel(UNet2DConditionModel):
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        up_ft_indices,
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None):
        r"""
        Args:
            sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
            encoder_hidden_states (`torch.FloatTensor`): (batch, sequence_length, feature_dim) encoder hidden states
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttnProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).
        """
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            # logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)

        emb = self.time_embedding(t_emb, timestep_cond)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        # 2. pre-process
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
            )

        # 5. up
        up_ft = {}
        for i, upsample_block in enumerate(self.up_blocks):

            if i > np.max(up_ft_indices):
                break

            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample, temb=emb, res_hidden_states_tuple=res_samples, upsample_size=upsample_size
                )

            if i in up_ft_indices:
                up_ft[i] = sample.detach()

        output = {}
        output['up_ft'] = up_ft
        return output

class OneStepSDPipeline(StableDiffusionPipeline):
    @torch.no_grad()
    def __call__(
        self,
        img_tensor,
        t,
        up_ft_indices,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None
    ):

        #TODO: make deterministic for reproducibility
        device = self._execution_device
        latents = self.vae.encode(img_tensor).latent_dist.sample() * self.vae.config.scaling_factor
        t = torch.tensor(t, dtype=torch.long, device=device)
        noise = torch.randn_like(latents).to(device)
        latents_noisy = self.scheduler.add_noise(latents, noise, t)
        unet_output = self.unet(latents_noisy,
                               t,
                               up_ft_indices,
                               encoder_hidden_states=prompt_embeds,
                               cross_attention_kwargs=cross_attention_kwargs)
        return unet_output


class SDFeaturizer(BaseFeaturizer):
    """ 
    This is taken from the paper: https://arxiv.org/pdf/2306.03881 

    with some changes:
        - I use the base version of SD2.1, which was finetuned on 512x512 images but not on 768x768. This is to align better with DinoV2 large (518x518)
        - i set the ensemble_size to 1, which should reduce PCK by 2 percentage points according to the paper but reduces the inference time from 330ms to 60ms on a 4090.


    """
    def __init__(self, sd_id='sd2-community/stable-diffusion-2-1',image_resize_size=None,device='cuda',
     default_t=261, default_up_ft_index=1, default_ensemble_size=1):
        """
        image_resize_size: tuple of int, (height, width) that the image is resized to before being passed to the model. If None, the image is not resized.
        """
        unet = MyUNet2DConditionModel.from_pretrained(sd_id, subfolder="unet")
        onestep_pipe = OneStepSDPipeline.from_pretrained(sd_id, unet=unet, safety_checker=None)
        onestep_pipe.vae.decoder = None
        onestep_pipe.scheduler = DDIMScheduler.from_pretrained(sd_id, subfolder="scheduler")
        gc.collect()
        onestep_pipe = onestep_pipe.to(device)
        onestep_pipe.enable_attention_slicing()
        onestep_pipe.enable_xformers_memory_efficient_attention()
        self.pipe = onestep_pipe

        self.default_t = default_t
        self.default_up_ft_index = default_up_ft_index
        self.default_ensemble_size = default_ensemble_size
        self.image_resize_size = image_resize_size
        self.device = self.pipe.device

    @torch.no_grad()
    def extract_features(self,
                img_tensor, 
                prompt = "",
                t = None,
                up_ft_index=None,
                ensemble_size=None):
        """
            img_tensor: [1,c,h,w]; (0,1) range, unnormalized.


            t= noise timestep, earlier will focus more on semantics, later more on details. 
            up_ft_index= index of the upsampling block to use for feature extraction.
            ensemble_size= number of noisy samples to use for feature extraction. 
        """
        assert len(img_tensor.shape) == 4 # 1,c,h,w
        assert img_tensor.shape[0] == 1


        t = self.default_t if t is None else t
        up_ft_index = self.default_up_ft_index if up_ft_index is None else up_ft_index
        ensemble_size = self.default_ensemble_size if ensemble_size is None else ensemble_size

        # copy tensor 
        resized_img_tensor = img_tensor.clone()
        
        if self.image_resize_size is not None:
            resized_img_tensor = F.interpolate(resized_img_tensor, size=self.image_resize_size, mode='bilinear', align_corners=False)
        # normalize to [-1,1], because that is how SD was trained 
        # cf https://github.com/huggingface/diffusers/blob/bc55b631fdf3d0961c27ec548b1155d1fccf0424/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_img2img.py#L115
        resized_img_tensor = resized_img_tensor * 2 - 1

        resized_img_tensor = resized_img_tensor.repeat(ensemble_size, 1, 1, 1).to(self.device)   # ensem, c, h, w

        prompt_embeds,neg_prompt_embeds = self.pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False) # [1, 77, dim]

        prompt_embeds = prompt_embeds.repeat(ensemble_size, 1, 1)
        unet_ft_all = self.pipe(
            img_tensor=resized_img_tensor,
            t=t,
            up_ft_indices=[up_ft_index],
            prompt_embeds=prompt_embeds)
        unet_ft = unet_ft_all['up_ft'][up_ft_index] # ensem, c, H//patch_size, W//patch_size
        unet_ft = unet_ft.mean(0, keepdim=True) # 1,c,H//patch_size,W//patch_size
        
        
        # upsample to original image size (latents are /8 and might be resized before inference)
        unet_ft = F.interpolate(unet_ft, size=(img_tensor.shape[2], img_tensor.shape[3]), mode='bilinear', align_corners=False) # 1,D,h,w

        return unet_ft

# 2.1 base models (512x512)
@FeaturizerRegistry.register("dift-sd2.1b-e1")
class SDFeaturizerSD21baseE1(SDFeaturizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.default_ensemble_size = 1

@FeaturizerRegistry.register("dift-sd2.1b-e2")
class SDFeaturizerSD21baseE2(SDFeaturizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.default_ensemble_size = 2

@FeaturizerRegistry.register("dift-sd2.1b-e4")
class SDFeaturizerSD21baseE4(SDFeaturizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.default_ensemble_size = 4

@FeaturizerRegistry.register("dift-sd2.1b-e8")
class SDFeaturizerSD21baseE8(SDFeaturizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.default_ensemble_size = 8


# 2.1 base with explicit resizes to the native 512x512 input size
@FeaturizerRegistry.register("dift-sd2.1b-e1-r512")
class SDFeaturizerSD21baseE1R512(SDFeaturizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.default_ensemble_size = 1
        self.image_resize_size = (512, 512)

@FeaturizerRegistry.register("dift-sd2.1b-e2-r512")
class SDFeaturizerSD21baseE2R512(SDFeaturizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.default_ensemble_size = 2
        self.image_resize_size = (512, 512)

@FeaturizerRegistry.register("dift-sd2.1b-e4-r512")
class SDFeaturizerSD21baseE4R512(SDFeaturizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.default_ensemble_size = 4
        self.image_resize_size = (512, 512)

@FeaturizerRegistry.register("dift-sd2.1b-e8-r512")
class SDFeaturizerSD21baseE8R512(SDFeaturizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.default_ensemble_size = 8
        self.image_resize_size = (512, 512)

# sd2.1 finetuned models, which have native 768x768 input size
@FeaturizerRegistry.register("dift-sd2.1-e1")
class SDFeaturizerSD21E1(SDFeaturizer):
    def __init__(self, **kwargs):
        super().__init__(sd_id='sd2-community/stable-diffusion-2-1', **kwargs)
        self.default_ensemble_size = 1

@FeaturizerRegistry.register("dift-sd2.1-e2")
class SDFeaturizerSD21E2(SDFeaturizer):
    def __init__(self, **kwargs):
        super().__init__(sd_id='sd2-community/stable-diffusion-2-1', **kwargs)
        self.default_ensemble_size = 2

@FeaturizerRegistry.register("dift-sd2.1-e4")
class SDFeaturizerSD21E4(SDFeaturizer):
    def __init__(self, **kwargs):
        super().__init__(sd_id='sd2-community/stable-diffusion-2-1', **kwargs)
        self.default_ensemble_size = 4

@FeaturizerRegistry.register("dift-sd2.1-e8")
class SDFeaturizerSD21E8(SDFeaturizer):
    def __init__(self, **kwargs):
        super().__init__(sd_id='sd2-community/stable-diffusion-2-1', **kwargs)
        self.default_ensemble_size = 8

# same but with explicit resizes to the native 768x768 input size
@FeaturizerRegistry.register("dift-sd2.1-e1-r768")
class SDFeaturizerSD21E1R768(SDFeaturizer):
    def __init__(self, **kwargs):
        super().__init__(sd_id='sd2-community/stable-diffusion-2-1', **kwargs)
        self.default_ensemble_size = 1
        self.image_resize_size = (768, 768)

@FeaturizerRegistry.register("dift-sd2.1-e2-r768")
class SDFeaturizerSD21E2R768(SDFeaturizer):
    def __init__(self, **kwargs):
        super().__init__(sd_id='sd2-community/stable-diffusion-2-1', **kwargs)
        self.default_ensemble_size = 2
        self.image_resize_size = (768, 768)

@FeaturizerRegistry.register("dift-sd2.1-e4-r768")
class SDFeaturizerSD21E4R768(SDFeaturizer):
    def __init__(self, **kwargs):
        super().__init__(sd_id='sd2-community/stable-diffusion-2-1', **kwargs)
        self.default_ensemble_size = 4
        self.image_resize_size = (768, 768)

@FeaturizerRegistry.register("dift-sd2.1-e8-r768")
class SDFeaturizerSD21E8R768(SDFeaturizer):
    def __init__(self, **kwargs):
        super().__init__(sd_id='sd2-community/stable-diffusion-2-1', **kwargs)
        self.default_ensemble_size = 8
        self.image_resize_size = (768, 768)

# 2.1 models with explicit resizes to 512x512 input
@FeaturizerRegistry.register("dift-sd2.1-e1-r512")
class SDFeaturizerSD21E1R512(SDFeaturizer):
    def __init__(self, **kwargs):
        super().__init__(sd_id='sd2-community/stable-diffusion-2-1', **kwargs)
        self.default_ensemble_size = 1
        self.image_resize_size = (512, 512)

@FeaturizerRegistry.register("dift-sd2.1-e2-r512")
class SDFeaturizerSD21E2R512(SDFeaturizer):
    def __init__(self, **kwargs):
        super().__init__(sd_id='sd2-community/stable-diffusion-2-1', **kwargs)
        self.default_ensemble_size = 2
        self.image_resize_size = (512, 512)

@FeaturizerRegistry.register("dift-sd2.1-e4-r512")
class SDFeaturizerSD21E4R512(SDFeaturizer):
    def __init__(self, **kwargs):
        super().__init__(sd_id='sd2-community/stable-diffusion-2-1', **kwargs)
        self.default_ensemble_size = 4
        self.image_resize_size = (512, 512)

@FeaturizerRegistry.register("dift-sd2.1-e8-r512")
class SDFeaturizerSD21E8R512(SDFeaturizer):
    def __init__(self, **kwargs):
        super().__init__(sd_id='sd2-community/stable-diffusion-2-1', **kwargs)
        self.default_ensemble_size = 8
        self.image_resize_size = (512, 512)

if __name__ == "__main__":
    tensor = torch.randn(1, 3, 640, 480)
    featurizer = SDFeaturizer()
    ft = featurizer.extract_features(tensor, "a photo of a cat")
    print(ft.shape)