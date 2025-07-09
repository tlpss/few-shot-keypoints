# load coco dataset 
# take keypoint prediction method.
# for each image in the dataset, run the keypoint prediction method.
# load coco image and category id from the dataset
# create new annotation with the predicted keypoints
# save results in a new coco dataset

from typing import Callable, List
from few_shot_keypoints.datasets.coco_dataset import COCOKeypointsDataset
from airo_dataset_tools.data_parsers.coco import CocoInstanceAnnotation
from tqdm import tqdm
from few_shot_keypoints.datasets.data_parsers import CocoKeypointsResultAnnotation, CocoKeypointsResultDataset
from few_shot_keypoints.matcher import KeypointListMatcher


def run_coco_dataset_inference(coco_dataset: COCOKeypointsDataset, keypoint_matcher: KeypointListMatcher, coco_results_output_path: str, keypoint_channel_configuration: List[str], transform_reverter: Callable):

    coco_results_annotations = []
    coco_images = coco_dataset.parsed_coco.images
    coco_category = coco_dataset.parsed_coco.categories[0]
    coco_category.keypoints = keypoint_channel_configuration

    for i in tqdm(range(len(coco_dataset))):
        datapoint = coco_dataset[i]
        image = datapoint["image"]
        image = image.unsqueeze(0)
        results = keypoint_matcher.get_keypoints(image)
        keypoints = []
        visibilities =[]
        for result in results:
            if result.u is not None and result.v is not None:
                keypoints.append(result.u)
                keypoints.append(result.v)
                visibilities.append(2)
            else:
                keypoints.append(0)
                keypoints.append(0)
                visibilities.append(0)

        keypoints = transform_reverter(keypoints, datapoint["original_image_size"], image.shape[2:])
        flattened_keypoints =[]
        for u,v,vis in zip(keypoints[0::2], keypoints[1::2], visibilities):
            flattened_keypoints.append(u)
            flattened_keypoints.append(v)
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

    coco_results_dataset = CocoKeypointsResultDataset(
        images=coco_images,
        annotations=coco_results_annotations,
        categories=[coco_category]
    )

    with open(coco_results_output_path, "w") as f:
        f.write(coco_results_dataset.model_dump_json(indent=4))

if __name__ == "__main__":
    from few_shot_keypoints.featurizers.ViT_featurizer import ViTFeaturizer
    from few_shot_keypoints.datasets.coco_dataset import COCOKeypointsDataset
    from few_shot_keypoints.matcher import KeypointFeatureMatcher, KeypointListMatcher
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from few_shot_keypoints.datasets.transforms import RESIZE_TRANSFORM, revert_resize_transform

    coco_json_path = "/home/tlips/Code/few-shot-keypoints/data/SPair-71k/SPAIR_coco_bird_test.json"
    target_path ="test.json"
    feature_extractor = ViTFeaturizer()
    matcher = KeypointListMatcher(keypoint_channels=["kp0,kp1"], matchers=[KeypointFeatureMatcher(feature_extractor)])
    coco_dataset = COCOKeypointsDataset(json_dataset_path=coco_json_path, keypoint_channel_configuration=["kp0"],transforms=RESIZE_TRANSFORM)

    # add first two random images to the matcher
    for i in range(4):
        image = coco_dataset[i]["image"]
        keypoint = coco_dataset[i]["keypoints"][0]
        if len(keypoint) > 0:
            image = image.unsqueeze(0)
            matcher.matchers[0].add_reference_image(image, keypoint[0])

    run_coco_dataset_inference(coco_dataset, matcher, target_path, keypoint_channel_configuration=["kp0"], transform_reverter=revert_resize_transform)

    