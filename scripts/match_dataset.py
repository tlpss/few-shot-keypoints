import os
import json
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from tqdm import trange
from dataclasses import dataclass
import draccus

from few_shot_keypoints.dataset_matching import populate_matcher_w_random_references, run_coco_dataset_inference
from few_shot_keypoints.dataset_object_crop_matching import populate_matcher_w_random_references as populate_matcher_w_random_references_crop
from few_shot_keypoints.dataset_object_crop_matching import run_coco_dataset_inference as run_coco_dataset_inference_crop
from few_shot_keypoints.dataset_mask_matching import MASK_MODES
from few_shot_keypoints.dataset_mask_matching import populate_matcher_w_random_references as populate_matcher_w_random_references_mask
from few_shot_keypoints.dataset_mask_matching import run_coco_dataset_inference as run_coco_dataset_inference_mask
from few_shot_keypoints.config import ObjectsRepresentationConfig
from few_shot_keypoints.config_references import assert_config_matches_dataset, extract_reference_vectors_from_config
from few_shot_keypoints.datasets.coco_dataset import TorchCOCOKeypointsDataset
from few_shot_keypoints.datasets.transforms import RESIZE_TRANSFORM, revert_resize_transform, MAX_LENGTH_RESIZE_AND_PAD_TRANSFORM, revert_max_length_resize_and_pad_transform
from few_shot_keypoints.featurizers.registry import FeaturizerRegistry
from few_shot_keypoints.matcher import KeypointFeatureMatcher

@dataclass
class Config:
    train_dataset_path: str = "/home/tlips/Code/few-shot-keypoints/data/SPair-71k/SPAIR_coco_train_train.json"
    test_dataset_path: str = "/home/tlips/Code/few-shot-keypoints/data/SPair-71k/SPAIR_coco_train_test.json"
    seed: int = 2025
    featurizer: str = "dino" # or "dift"
    transform : str = "resize" # or "resize_max_and_pad"
    output_base_dir: str = "results/SPAIR-support-sets"
    dataset_name: Optional[str] = None
    crop_before_matching: bool = False
    mask_before_matching: bool = False
    crop_target_size: tuple = (256, 256)
    margin_scale: float = 0.1
    mask_mode: str = "pixels" # how to use the object mask, cf. few_shot_keypoints.dataset_mask_matching. only used if mask_before_matching.
    # number of matches to extract for each keypoint channel. COCO only allows a single keypoint per channel, so the
    # additional matches are stored in the optional `keypoint_topk_matches` field of the results.
    # None -> 1, or the per-keypoint top_ks of the reference config if one is used.
    top_k: Optional[int] = None
    min_distance_between_topk_matches: int = 50 # only relevant for top_k > 1
    # optional reference keypoint config directory (cf. few_shot_keypoints.config) to take the reference vectors from,
    # instead of sampling them from the ground truth annotations of random train dataset images.
    reference_config_dir: Optional[str] = None
    reference_config_object: Optional[str] = None # which object of the reference config to use, if it holds several.

    def __post_init__(self):
        if self.crop_before_matching and self.mask_before_matching:
            raise ValueError("Cannot use both crop_before_matching and mask_before_matching")
        if self.mask_mode not in MASK_MODES:
            raise ValueError(f"Invalid mask_mode: {self.mask_mode}, should be one of {MASK_MODES}")
        if (self.crop_before_matching or self.mask_before_matching) and self.transform != "none":
            # the albumentations transforms resize the image, keypoints and mask, but pass the bbox through
            # unchanged, so the bbox no longer matches the transformed image.
            raise ValueError("crop_before_matching and mask_before_matching require transform='none'")


def create_matcher(reference_vectors, top_k_matches: Optional[list], config: Config) -> KeypointFeatureMatcher:
    """Create the matcher for the given reference vectors. The top_k of the config takes precedence over the
    per-keypoint top_ks of a reference config, which in turn take precedence over the default of a single match."""
    if config.top_k is not None:
        top_k_matches = [config.top_k] * len(reference_vectors)
    elif top_k_matches is None:
        top_k_matches = [1] * len(reference_vectors)
    return KeypointFeatureMatcher(
        reference_vectors,
        top_k_matches=top_k_matches,
        min_distance_between_topk_matches=config.min_distance_between_topk_matches,
        device='cuda:0',
    )

#@draccus.wrap()
def match_dataset(config: Config):

    if config.transform == "resize":
        transform  = RESIZE_TRANSFORM
        transform_reverter = revert_resize_transform
    elif config.transform == "resize_max_and_pad":
        transform  = MAX_LENGTH_RESIZE_AND_PAD_TRANSFORM
        transform_reverter = revert_max_length_resize_and_pad_transform
    elif config.transform == "none":
        transform = None
        transform_reverter = None
    else:
        raise ValueError(f"Invalid transform: {config.transform}")

    # load train dataset
    train_dataset = TorchCOCOKeypointsDataset(config.train_dataset_path, transform=transform)
    # load test dataset 
    test_dataset = TorchCOCOKeypointsDataset(config.test_dataset_path, transform=transform)

    if config.dataset_name is not None:
        name = config.dataset_name
    else:
        name = train_dataset.parsed_coco.categories[0].name
    # the seed determines the reference vectors, unless they are taken from a reference config.
    reference_tag = str(config.seed) if config.reference_config_dir is None else f"config-{Path(config.reference_config_dir).name}"
    result_name = f"{config.transform}_{reference_tag}"
    if config.mask_before_matching and config.mask_mode != "pixels":
        result_name += f"_maskmode-{config.mask_mode}"
    if config.top_k is not None and config.top_k != 1:
        result_name += f"_topk-{config.top_k}"
    filename = Path(config.output_base_dir) / f"{config.featurizer}" / name / f"{result_name}_results.json"
    if filename.exists():
        print(f"Results already exist for {filename}")
        return

    # create matcher
    if config.featurizer in FeaturizerRegistry.list():
        featurizer = FeaturizerRegistry.create(config.featurizer, device='cuda:0')
    else:
        raise ValueError(f"Invalid featurizer: {config.featurizer}")

    # take the reference vectors from a reference keypoint config, if one is given.
    reference_config_object = None
    if config.reference_config_dir is not None:
        reference_config = ObjectsRepresentationConfig.load_from_dir(config.reference_config_dir)
        reference_config_object = reference_config.get_object(config.reference_config_object)
        assert_config_matches_dataset(reference_config_object, train_dataset)

    top_ks = None
    if config.mask_before_matching:
        # Use mask-based matching (requires segmentation masks in dataset)
        if reference_config_object is not None:
            references, top_ks = extract_reference_vectors_from_config(
                reference_config_object,
                featurizer,
                crop=True,
                mask_pixels=config.mask_mode in ("pixels", "both"),
                crop_target_size=config.crop_target_size,
                margin_scale=config.margin_scale,
            )
        else:
            references = populate_matcher_w_random_references_mask(
                train_dataset, 
                featurizer, 
                crop_target_size=config.crop_target_size,
                margin_scale=config.margin_scale,
                seed=config.seed,
                mask_mode=config.mask_mode,
            )
        matcher = create_matcher(references, top_ks, config)
        coco_results = run_coco_dataset_inference_mask(
            test_dataset, 
            matcher, 
            featurizer,
            crop_target_size=config.crop_target_size,
            margin_scale=config.margin_scale,
            mask_mode=config.mask_mode,
        )
    elif config.crop_before_matching:
        if reference_config_object is not None:
            references, top_ks = extract_reference_vectors_from_config(reference_config_object, featurizer, crop=True)
        else:
            references = populate_matcher_w_random_references_crop(train_dataset, featurizer, seed=config.seed)
        matcher = create_matcher(references, top_ks, config)
        coco_results = run_coco_dataset_inference_crop(test_dataset, matcher, featurizer)
    else:
        if reference_config_object is not None:
            references, top_ks = extract_reference_vectors_from_config(reference_config_object, featurizer, transform=transform)
        else:
            # populate matcher with random reference images
            references = populate_matcher_w_random_references(train_dataset, featurizer, seed=config.seed)
        matcher = create_matcher(references, top_ks, config)
        coco_results = run_coco_dataset_inference(test_dataset, matcher, featurizer, transform_reverter=transform_reverter)
    os.makedirs(filename.parent, exist_ok=True)
    with open(filename, "w") as f:
        f.write(coco_results.model_dump_json(indent=4))

    # clear VRAM 
    del featurizer
    import gc
    gc.collect()
    with torch.no_grad():
        # clear cache
        torch.cuda.empty_cache()

if __name__ == "__main__":
    print(FeaturizerRegistry.list())
    config = Config()
    # config.train_dataset_path = "/home/tlips/Code/few-shot-keypoints/data/aRTF/tshirts-train_resized_512x256/tshirts-train.json"
    # config.test_dataset_path = "/home/tlips/Code/few-shot-keypoints/data/aRTF/tshirts-test_resized_512x256/tshirts-test.json"
    config.output_base_dir = "results/aRTF-support-sets"
    config.featurizer = "dinov3-s"
    config.seed = 2025
    config.transform = "resize"
    match_dataset(config)