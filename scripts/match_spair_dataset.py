import os
import json
from pathlib import Path
import numpy as np
from tqdm import trange
from dataclasses import dataclass
import draccus

from few_shot_keypoints.dataset_matching import populate_matcher_w_random_references, run_coco_dataset_inference
from few_shot_keypoints.datasets.coco_dataset import TorchCOCOKeypointsDataset
from few_shot_keypoints.datasets.transforms import RESIZE_TRANSFORM, revert_resize_transform, MAX_LENGTH_RESIZE_AND_PAD_TRANSFORM, revert_max_length_resize_and_pad_transform
from few_shot_keypoints.featurizers.ViT_featurizer import ViTFeaturizer
from few_shot_keypoints.featurizers.dift_featurizer import SDFeaturizer
from few_shot_keypoints.matcher import KeypointFeatureMatcher, KeypointListMatcher

@dataclass
class Config:
    train_dataset_path: str = "/home/tlips/Code/few-shot-keypoints/data/SPair-71k/SPAIR_coco_train_train.json"
    test_dataset_path: str = "/home/tlips/Code/few-shot-keypoints/data/SPair-71k/SPAIR_coco_train_test.json"
    N_support_images: int = 1
    seed: int = 2025
    featurizer: str = "dino" # or "dift"
    transform : str = "resize" # or "resize_max_and_pad"
    output_base_dir: str = "results/SPAIR-support-sets"

@draccus.wrap()
def match_dataset(config: Config):

    if config.transform == "resize":
        transform  = RESIZE_TRANSFORM
        transform_reverter = revert_resize_transform
    elif config.transform == "resize_max_and_pad":
        transform  = MAX_LENGTH_RESIZE_AND_PAD_TRANSFORM
        transform_reverter = revert_max_length_resize_and_pad_transform
    else:
        raise ValueError(f"Invalid transform: {config.transform}")

    # load train dataset
    train_dataset = TorchCOCOKeypointsDataset(config.train_dataset_path, transform=transform)
    # load test dataset 
    test_dataset = TorchCOCOKeypointsDataset(config.test_dataset_path, transform=transform)
    # create matcher
    if config.featurizer == "dino":
        featurizer = ViTFeaturizer(device='cuda', hf_model_name="facebook/dinov2-small")
    elif config.featurizer == "dift":
        featurizer = SDFeaturizer(device='cuda')
    else:
        raise ValueError(f"Invalid featurizer: {config.featurizer}")

    keypoint_types = train_dataset.parsed_coco.categories[0].keypoints
    matchers = [KeypointFeatureMatcher(featurizer) for _ in keypoint_types]
    matcher = KeypointListMatcher(keypoint_channels=keypoint_types, matchers=matchers)

    # populate matcher with random reference images
    populate_matcher_w_random_references(train_dataset, matcher, N=config.N_support_images, seed=config.seed)


    coco_results = run_coco_dataset_inference(test_dataset, matcher, transform_reverter=transform_reverter)
    category = train_dataset.parsed_coco.categories[0].name
    filename = Path(config.output_base_dir) / f"{config.featurizer}" /category / f"{config.N_support_images}" / f"{config.transform}_{config.seed}_results.json"
    os.makedirs(filename.parent, exist_ok=True)
    with open(filename, "w") as f:
        f.write(coco_results.model_dump_json(indent=4))


if __name__ == "__main__":
    match_dataset()