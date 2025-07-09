from PIL import Image
from transformers import AutoImageProcessor, AutoModel, BitImageProcessor
import torch
from typing import Callable
import requests
import numpy as np
from few_shot_keypoints.featurizers.base import BaseFeaturizer
from torchvision.transforms.functional import normalize

def concatentation_aggregator(hidden_states: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat(hidden_states, dim=-1)


class ViTFeaturizer(BaseFeaturizer):
    def __init__(self, hf_model_name: str = "facebook/dinov2-small", layers: list[int] = [11], 
    layer_aggregator: Callable[[list[torch.Tensor]], torch.Tensor] = concatentation_aggregator,
    device: str = 'cuda'):
        self.model = AutoModel.from_pretrained(hf_model_name)
        self.model.eval()
        self.model.to(device)
        # self.processor = AutoImageProcessor.from_pretrained(hf_model_name)
        # https://huggingface.co/facebook/dinov2-base/discussions/9 -> default resolution is 224,224 for Dinov2!
        #TODO: want to make this bigger?
        # self.processor = BitImageProcessor(size=(self.model.config.image_size, self.model.config.image_size),do_center_crop=False,do_rescale=True,do_normalize=True)
        self.layers = layers
        self.layer_aggregator = layer_aggregator
        self.device = device




    def extract_features(self, image: torch.Tensor, **kwargs):
        assert len(image.shape) == 4 # [BATCH_SIZE, 3, IMG_HEIGHT, IMG_WIDTH]
        image = image.to(self.device)

        # find closest multiple of patch size, but do not force size or aspect ratio, this is responsability of the user.
        patch_size = self.model.config.patch_size
        new_height = image.shape[2] // patch_size * patch_size
        new_width = image.shape[3] // patch_size * patch_size
        patch_image = torch.nn.functional.interpolate(image, size=(new_height, new_width), mode="bilinear", align_corners=False)

        # normalize to imagenet mean and std
        # cf https://github.com/facebookresearch/dinov2/blob/main/dinov2/data/transforms.py#L63
        patch_image = normalize(patch_image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        features = self.forward(patch_image) # B,H',W',D
        # reshape to [B,D,H',W']
        features = features.permute(0,3,1,2)
        # upsample to the original image size
        upsampled_features = torch.nn.functional.interpolate(features, size=(image.shape[2], image.shape[3]), mode="bilinear", align_corners=False)
        return upsampled_features


    def forward(self, images: torch.Tensor):
        assert len(images.shape) == 4 # [BATCH_SIZE, 3, IMG_HEIGHT, IMG_WIDTH]

        # model uses embedding inerpolation to deal with different image sizes!
        outputs = self.model(pixel_values=images, output_hidden_states=True)
        hidden_states = outputs.hidden_states # [N_ATTN_LAYERS, BATCH_SIZE, N_PATCHES, HIDDEN_SIZE]
        # drop class token and reshape to 2D
        hidden_states = [hidden_states[i][:,1:,:] for i in range(len(hidden_states))] # [N_ATTN_LAYERS, BATCH_SIZE, N_PATCHES, HIDDEN_SIZE]
        hidden_states = self.arrange_tokens_in_grid(hidden_states,width=images.shape[3],height=images.shape[2]) # [N_ATTN_LAYERS, BATCH_SIZE, N_PATCHES_HEIGHT, N_PATCHES_WIDTH, HIDDEN_SIZE]
        features =  self.layer_aggregator([hidden_states[i] for i in self.layers])  # B,H',W',D
        return features


    def arrange_tokens_in_grid(self, hidden_states: list[torch.Tensor], width: int, height: int) -> list[torch.Tensor]:
        # hidden_states: [N_ATTN_LAYERS, BATCH_SIZE, N_PATCHES, HIDDEN_SIZE]
        # return: [N_ATTN_LAYERS, BATCH_SIZE, N_PATCHES_HEIGHT, N_PATCHES_WIDTH, HIDDEN_SIZE]
        return [hidden_states[i].reshape(hidden_states[i].shape[0], height // self.model.config.patch_size, width // self.model.config.patch_size, hidden_states[i].shape[2]) for i in range(len(hidden_states))]

if __name__ == "__main__":
    featurizer = ViTFeaturizer(
        hf_model_name="facebook/dinov2-small",
        layers=[4,11],
        layer_aggregator=concatentation_aggregator,
        device="cuda"
    )
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)
    image = image.resize((518,620))
    image = torch.tensor(np.array(image)).permute(2,0,1)
    image = image / 255.0
    image = image.unsqueeze(0)
    image = image.to(featurizer.device)
    print(image.shape)
    # convert to torch tensor
    print(featurizer.extract_features(image).shape)