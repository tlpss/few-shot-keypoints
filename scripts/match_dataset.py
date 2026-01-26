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
    filename = Path(config.output_base_dir) / f"{config.featurizer}" / name / f"{config.transform}_{config.seed}_results.json"
    if filename.exists():
        print(f"Results already exist for {filename}")
        return

    # create matcher
    if config.featurizer in FeaturizerRegistry.list():
        featurizer = FeaturizerRegistry.create(config.featurizer, device='cuda:0')
    else:
        raise ValueError(f"Invalid featurizer: {config.featurizer}")


    # populate matcher with random reference images
    references = populate_matcher_w_random_references(train_dataset, featurizer, seed=config.seed)
    matcher = KeypointFeatureMatcher(references, device='cuda:0')

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