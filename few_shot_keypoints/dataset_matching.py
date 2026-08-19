# load coco dataset 
# take keypoint prediction method.
# for each image in the dataset, run the keypoint prediction method.
# load coco image and category id from the dataset
# create new annotation with the predicted keypoints
# save results in a new coco dataset

import random
from typing import Callable, List, Optional, Tuple
from few_shot_keypoints.datasets.coco_dataset import TorchCOCOKeypointsDataset
from tqdm import tqdm
from few_shot_keypoints.datasets.data_parsers import CocoKeypointsResultAnnotation, CocoKeypointsResultDataset
from few_shot_keypoints.matcher import KeypointFeatureMatcher, MatchingResult
from few_shot_keypoints.featurizers.base import BaseFeaturizer
import torch


def matches_to_coco_keypoints(
    results: List[List[MatchingResult]],
    point_mapper: Optional[Callable[[float, float], Tuple[float, float]]] = None,
) -> Tuple[List[float], List[float], List[List[List[float]]]]:
    """Convert the matches of the KeypointFeatureMatcher to the COCO keypoint result fields.

    Args:
        results: for each keypoint channel, the matches ordered from best to worst
            (a channel can have more than one match if the matcher was created with top_k > 1).
        point_mapper: optional callable to map a matched (u,v) back to the original image frame.

    Returns:
        (flattened_keypoints, scores, topk_matches):
        - flattened_keypoints: [u,v,visibility] triplets of the *best* match of each channel, as COCO
          only allows a single keypoint per channel. (0,0,0) if a channel has no match.
        - scores: the score of the best match of each channel that has a match.
        - topk_matches: for each channel, all its matches as [u,v,score] triplets.
    """
    flattened_keypoints = []
    scores = []
    topk_matches = []

    for channel_results in results:
        if not channel_results or channel_results[0].u is None or channel_results[0].v is None:
            flattened_keypoints.extend([0, 0, 0])
            topk_matches.append([])
            continue

        channel_matches = []
        for match in channel_results:
            u, v = match.u, match.v
            if point_mapper is not None:
                u, v = point_mapper(u, v)
            channel_matches.append([u, v, match.score])

        u, v, score = channel_matches[0]
        flattened_keypoints.extend([u, v, 2])  # 2 = visible
        scores.append(score)
        topk_matches.append(channel_matches)

    return flattened_keypoints, scores, topk_matches


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

        # revert the dataset transform to get the keypoints back in the original image frame.
        point_mapper = None
        if transform_reverter is not None:
            def point_mapper(u, v):
                return transform_reverter([(u, v)], datapoint["original_image_size"], image.shape[2:])[0]

        flattened_keypoints, scores, topk_matches = matches_to_coco_keypoints(results, point_mapper)

        coco_image_id = datapoint["coco_image_id"]
        coco_category_id = datapoint["coco_category_id"]
        coco_results_annotations.append(
            CocoKeypointsResultAnnotation(
                id=i,
                image_id=coco_image_id,
                category_id=coco_category_id,
                bbox=datapoint["bbox"],
                keypoints=flattened_keypoints,
                score=sum(scores) / len(scores) if scores else 0.0,
                keypoint_scores=scores,
                keypoint_topk_matches=topk_matches,
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
    from few_shot_keypoints.featurizers.ViT_featurizer import ViTFeaturizer, DinoV3LargeFeaturizer
    from few_shot_keypoints.featurizers.dift_featurizer import SDFeaturizer
    from few_shot_keypoints.datasets.coco_dataset import TorchCOCOKeypointsDataset
    from airo_dataset_tools.data_parsers.coco import CocoKeypointsDataset as CocoParser
    from few_shot_keypoints.matcher import KeypointFeatureMatcher
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from few_shot_keypoints.datasets.transforms import RESIZE_TRANSFORM, revert_resize_transform
    import json
    from few_shot_keypoints.paths import KIL_MUGS_V2_INITIAL_JSON

    coco_json_path = KIL_MUGS_V2_INITIAL_JSON

    with open(coco_json_path, "r") as f:
        coco_dataset = CocoParser(**json.load(f))

    target_path ="test.json"
    feature_extractor = DinoV3LargeFeaturizer(device='cuda:0')
    # feature_extractor = SDFeaturizer(device='cuda')

    keypoint_types = coco_dataset.categories[0].keypoints

    image_size = 512
    from few_shot_keypoints.datasets.augmentations import MultiChannelKeypointsCompose
    import cv2
    transform = MultiChannelKeypointsCompose([A.Resize(image_size,image_size,interpolation=cv2.INTER_CUBIC)])
    coco_dataset = TorchCOCOKeypointsDataset(json_dataset_path=coco_json_path,transform=transform)

    reference_vectors = populate_matcher_w_random_references(coco_dataset, feature_extractor, seed=2029)
    matcher = KeypointFeatureMatcher(reference_vectors, device='cuda:0')

    coco_results_dataset = run_coco_dataset_inference(coco_dataset, matcher, feature_extractor, transform_reverter=revert_resize_transform)
    with open(target_path, "w") as f:
        f.write(coco_results_dataset.model_dump_json(indent=4))

    