from few_shot_keypoints.dataset_matching import run_coco_dataset_inference
from few_shot_keypoints.featurizers.ViT_featurizer import ViTFeaturizer
from few_shot_keypoints.datasets.coco_dataset import TorchCOCOKeypointsDataset
from airo_dataset_tools.data_parsers.coco import CocoKeypointsDataset as CocoParser
from few_shot_keypoints.matcher import KeypointFeatureMatcher, KeypointListMatcher
import albumentations as A
from albumentations.pytorch import ToTensorV2
from few_shot_keypoints.datasets.transforms import RESIZE_TRANSFORM, revert_resize_transform
import random
import json
import torch

random.seed(2025)
coco_json_path = "/home/tlips/Code/few-shot-keypoints/data/SPair-71k/SPAIR_coco_tvmonitor_test.json"

# enable memory profiling
torch.cuda.memory._record_memory_history()
with open(coco_json_path, "r") as f:
    coco_dataset = CocoParser(**json.load(f))
target_path ="test.json"
feature_extractor = ViTFeaturizer(device='cuda', hf_model_name="facebook/dinov2-small")
keypoint_types = coco_dataset.categories[0].keypoints
matchers = [KeypointFeatureMatcher(feature_extractor) for _ in keypoint_types]
matcher = KeypointListMatcher(keypoint_channels=keypoint_types, matchers=matchers)
coco_dataset = TorchCOCOKeypointsDataset(json_dataset_path=coco_json_path,transforms=RESIZE_TRANSFORM)

# add first two random images to the matcher
for i in range(len(keypoint_types)):
    N = 5
    while not len(matchers[i].reference_images) == N:
        idx = random.randint(0, len(coco_dataset)-1)
        image = coco_dataset[idx]["image"]
        image = image.unsqueeze(0)
        if len(coco_dataset[idx]["keypoints"][i]) > 0:
            matchers[i].add_reference_image(image, coco_dataset[idx]["keypoints"][i][0])
        else:
            print(f"no keypoints for {keypoint_types[i]}")
            pass

coco_results_dataset = run_coco_dataset_inference(coco_dataset, matcher,transform_reverter=revert_resize_transform)

torch.cuda.memory._dump_snapshot("cuda-memory.pickle")