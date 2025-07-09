# load coco dataset 
# take keypoint prediction method.
# for each image in the dataset, run the keypoint prediction method.
# load coco image and category id from the dataset
# create new annotation with the predicted keypoints
# save results in a new coco dataset

import random
from typing import Callable, List
from few_shot_keypoints.datasets.coco_dataset import TorchCOCOKeypointsDataset
from airo_dataset_tools.data_parsers.coco import CocoInstanceAnnotation, CocoKeypointsDataset
from tqdm import tqdm
from few_shot_keypoints.datasets.data_parsers import CocoKeypointsResultAnnotation, CocoKeypointsResultDataset
from few_shot_keypoints.matcher import KeypointListMatcher


def run_coco_dataset_inference(coco_dataset: TorchCOCOKeypointsDataset, keypoint_matcher: KeypointListMatcher, transform_reverter: Callable) -> CocoKeypointsResultDataset:

    coco_results_annotations = []
 
    for i in tqdm(range(len(coco_dataset))):
        datapoint = coco_dataset[i]
        image = datapoint["image"]
        image = image.unsqueeze(0)
        results = keypoint_matcher.get_keypoints(image)
        keypoints = []
        visibilities =[]
        for result in results:
            if result.u is not None and result.v is not None:
                keypoints.append((result.u, result.v))
                visibilities.append(2)
            else:
                keypoints.append((0,0))
                visibilities.append(0)

        keypoints = transform_reverter(keypoints, datapoint["original_image_size"], image.shape[2:])
        flattened_keypoints =[]
        for kp,vis in zip(keypoints, visibilities):
            flattened_keypoints.append(kp[0])
            flattened_keypoints.append(kp[1])
            flattened_keypoints.append(vis)
       
        coco_image_id = datapoint["coco_image_id"]
        coco_category_id = datapoint["coco_category_id"]
        coco_results_annotations.append(CocoKeypointsResultAnnotation(
            id=i,
            image_id=coco_image_id,
            category_id=coco_category_id,
            bbox=datapoint["bbox"],
            keypoints=flattened_keypoints,
            score=sum([result.score for result in results])/len(results), 
            keypoint_scores=[result.score for result in results]
        ))
        print(f"image {coco_image_id} predicted keypoints: {keypoints}, ground truth keypoints: {datapoint['keypoints']}")

    coco_results_dataset = CocoKeypointsResultDataset(coco_results_annotations)

    return coco_results_dataset

if __name__ == "__main__":
    from few_shot_keypoints.featurizers.ViT_featurizer import ViTFeaturizer
    from few_shot_keypoints.datasets.coco_dataset import TorchCOCOKeypointsDataset
    from airo_dataset_tools.data_parsers.coco import CocoKeypointsDataset as CocoParser
    from few_shot_keypoints.matcher import KeypointFeatureMatcher, KeypointListMatcher
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from few_shot_keypoints.datasets.transforms import RESIZE_TRANSFORM, revert_resize_transform
    import json
    coco_json_path = "/home/tlips/Code/few-shot-keypoints/data/SPair-71k/SPAIR_coco_bird_test.json"

    with open(coco_json_path, "r") as f:
        coco_dataset = CocoParser(**json.load(f))
    target_path ="test.json"
    feature_extractor = ViTFeaturizer(device='cuda')
    keypoint_types = coco_dataset.categories[0].keypoints
    matchers = [KeypointFeatureMatcher(feature_extractor) for _ in keypoint_types]
    matcher = KeypointListMatcher(keypoint_channels=keypoint_types, matchers=matchers)
    coco_dataset = TorchCOCOKeypointsDataset(json_dataset_path=coco_json_path,transforms=RESIZE_TRANSFORM)

    # add first two random images to the matcher
    for i in range(len(keypoint_types)):
        N = 1
        while not len(matchers[i].reference_images) == N:
            idx = random.randint(0, len(coco_dataset)-1)
            image = coco_dataset[idx]["image"]
            image = image.unsqueeze(0)
            print(coco_dataset[idx]["keypoints"][i])
            if len(coco_dataset[idx]["keypoints"][i]) > 0:
                matchers[i].add_reference_image(image, coco_dataset[idx]["keypoints"][i][0])
            else:
                print(f"no keypoints for {keypoint_types[i]}")

    coco_results_dataset = run_coco_dataset_inference(coco_dataset, matcher,transform_reverter=revert_resize_transform)
    with open(target_path, "w") as f:
        f.write(coco_results_dataset.model_dump_json(indent=4))

    