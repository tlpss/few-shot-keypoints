# load coco dataset 
# take keypoint prediction method.
# for each image in the dataset, run the keypoint prediction method.
# load coco image and category id from the dataset
# create new annotation with the predicted keypoints
# save results in a new coco dataset

import random
from typing import Callable, List
from few_shot_keypoints.datasets.coco_dataset import TorchCOCOKeypointsDataset
from tqdm import tqdm
from few_shot_keypoints.datasets.data_parsers import CocoKeypointsResultAnnotation, CocoKeypointsResultDataset
from few_shot_keypoints.matcher import KeypointFeatureMatcher
from few_shot_keypoints.featurizers.base import BaseFeaturizer
import torch

def run_coco_dataset_inference(
    coco_dataset: TorchCOCOKeypointsDataset,
    keypoint_matcher: KeypointFeatureMatcher,
    feature_extractor: BaseFeaturizer,
    transform_reverter: Callable,
) -> CocoKeypointsResultDataset:
    """
    Run inference on a COCO dataset using a keypoint matcher and a feature extractor.

    the transform_reverter is used to revert the transform applied to the images in the dataset before storing the keypoint results.
    This allows for easy comparison with the ground truth keypoints. 
    """

    coco_results_annotations = []

    for i in tqdm(range(len(coco_dataset))):
        datapoint = coco_dataset[i]
        image = datapoint["image"]
        image = image.unsqueeze(0)

        features = feature_extractor.extract_features(image)
        results = keypoint_matcher.get_best_matches_from_image_features(features)

        keypoints = []
        visibilities = []
        scores = []
        for result_list in results:
            if len(result_list) > 0 and result_list[0].u is not None and result_list[0].v is not None:
                assert len(result_list) == 1, "Expected only 1 match for each keypoint category."
                # COCO can only have 1 keypoint per category. 
                result = result_list[0]
                keypoints.append((result.u, result.v))
                visibilities.append(2)
                scores.append(result.score)
            else:
                keypoints.append((0, 0))
                visibilities.append(0)

        if transform_reverter is not None:
            keypoints = transform_reverter(
                keypoints, datapoint["original_image_size"], image.shape[2:]
            )
        flattened_keypoints = []
        for kp, vis in zip(keypoints, visibilities):
            flattened_keypoints.append(kp[0])
            flattened_keypoints.append(kp[1])
            flattened_keypoints.append(vis)

        coco_image_id = datapoint["coco_image_id"]
        coco_category_id = datapoint["coco_category_id"]
        coco_results_annotations.append(
            CocoKeypointsResultAnnotation(
                id=i,
                image_id=coco_image_id,
                category_id=coco_category_id,
                bbox=datapoint["bbox"],
                keypoints=flattened_keypoints,
                score=sum(scores) / len(scores),
                keypoint_scores=scores,
            )
        )
        # print(f"image {coco_image_id} predicted keypoints: {keypoints}, ground truth keypoints: {datapoint['keypoints']}")

    coco_results_dataset = CocoKeypointsResultDataset(coco_results_annotations)

    return coco_results_dataset


def populate_matcher_w_random_references(
    coco_dataset: TorchCOCOKeypointsDataset,
    feature_extractor: BaseFeaturizer,
    seed: int = 2025,
):
    """
    Populate each keypoint matcher with N random reference images that contain
    at least one annotation for that keypoint.
    """
    reference_vectors = [None] * len(coco_dataset.parsed_coco.categories[0].keypoints)
    # sample random images, until we find a reference vector for each keypoint type
    rng = random.Random(seed)
    while any(rv is None for rv in reference_vectors):
        idx = rng.randint(0, len(coco_dataset) - 1)
        print(f"sampling random image {idx}")
        image = coco_dataset[idx]["image"]
        image = image.unsqueeze(0)
        features = feature_extractor.extract_features(image)
        keypoints = coco_dataset[idx]["keypoints"]
        for i in range(len(reference_vectors)):
            if reference_vectors[i] is None and len(keypoints[i]) > 0:
                u,v = keypoints[i][0]
                u,v = int(u), int(v)
                reference_vectors[i] = features[0,:,v,u].clone()
        n_found = sum(rv is not None for rv in reference_vectors)
        print(f"found {n_found} reference vectors")
    return torch.stack(reference_vectors)

if __name__ == "__main__":
    from few_shot_keypoints.featurizers.ViT_featurizer import ViTFeaturizer
    from few_shot_keypoints.featurizers.dift_featurizer import SDFeaturizer
    from few_shot_keypoints.datasets.coco_dataset import TorchCOCOKeypointsDataset
    from airo_dataset_tools.data_parsers.coco import CocoKeypointsDataset as CocoParser
    from few_shot_keypoints.matcher import KeypointFeatureMatcher
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from few_shot_keypoints.featurizers.dino_vit_paper.featurizer import ViTPaperFeaturizer
    from few_shot_keypoints.datasets.transforms import RESIZE_TRANSFORM, revert_resize_transform
    import json
    from few_shot_keypoints.paths import DSD_SHOE_TEST_JSON

    coco_json_path = "/home/tlips/Code/few-shot-keypoints/data/SPair-71k/SPAIR_coco_bicycle_test.json"
    coco_json_path = DSD_SHOE_TEST_JSON

    with open(coco_json_path, "r") as f:
        coco_dataset = CocoParser(**json.load(f))

    target_path ="test.json"
    feature_extractor = ViTFeaturizer(device='cuda:0', hf_model_name="facebook/dinov2-small")
    feature_extractor = ViTPaperFeaturizer(device='cuda:0', model_type='dino_vits8', stride=4, layer=11, facet='key', use_bin_features=False)
    # feature_extractor = SDFeaturizer(device='cuda')

    keypoint_types = coco_dataset.categories[0].keypoints

    image_size = 224
    from few_shot_keypoints.datasets.augmentations import MultiChannelKeypointsCompose
    import cv2
    transform = MultiChannelKeypointsCompose([A.Resize(image_size,image_size,interpolation=cv2.INTER_CUBIC)])
    coco_dataset = TorchCOCOKeypointsDataset(json_dataset_path=coco_json_path,transform=transform)

    reference_vectors = populate_matcher_w_random_references(coco_dataset, feature_extractor, seed=2029)
    matcher = KeypointFeatureMatcher(reference_vectors, device='cuda:0')

    coco_results_dataset = run_coco_dataset_inference(coco_dataset, matcher, feature_extractor, transform_reverter=revert_resize_transform)
    with open(target_path, "w") as f:
        f.write(coco_results_dataset.model_dump_json(indent=4))

    