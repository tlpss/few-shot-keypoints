from dataclasses import dataclass
import json
import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.utils.coco as fouc
import os

from few_shot_keypoints.datasets.coco_dataset import TorchCOCOKeypointsDataset
from few_shot_keypoints.datasets.data_parsers import CocoKeypointsResultDataset

@dataclass
class Config:
    images_dir: str
    labels_path: str
    results_path: str
    dataset_name: str = "coco-keypoints-gt"


def main(config: Config):
    # Import ground truth dataset


    # Load the dataset
    dataset = fo.Dataset.from_dir(
        data_path=config.images_dir,
        dataset_type=fo.types.COCODetectionDataset,
        labels_path=config.labels_path,
        label_types=["detections"],
        include_id=True,
        include_annotation_id=True,
    )

    # now iterate over all images (using the coco ID) 

    print(dataset.info)

    coco_image_id_to_sample = {sample["coco_id"]: sample for sample in dataset}

    # now load the dataset. For each image -> add all keypoints. 
    torch_dataset = TorchCOCOKeypointsDataset(config.labels_path, transform=None)

    category = torch_dataset.parsed_coco.categories[0]
    category_keypoint_names = category.keypoints
    for i in range(len(torch_dataset)):
        item = torch_dataset[i]
        img = item["image"]
        keypoints = item["keypoints"]
        H,W = img.shape[1:]
        fo_keypoints = []
        for i, category_keypoints in enumerate(keypoints):
            category_keypoints = [[u/W, v/H] for u,v in category_keypoints]
            fo_keypoints.append(fo.Keypoint(points=category_keypoints,label = category_keypoint_names[i]))
        coco_image_id = item["coco_image_id"]
        coco_sample = coco_image_id_to_sample[coco_image_id]
        coco_sample["keypoints"] = fo.Keypoints(keypoints=fo_keypoints)
        coco_sample.save()


    # add the predictions.
    predictions = json.load(open(config.results_path))
    predictions = CocoKeypointsResultDataset.model_validate(predictions)
    for prediction in predictions.root:
        img_id = prediction.image_id
        coco_sample = coco_image_id_to_sample[img_id]
        keypoints = prediction.keypoints
        keypoint_scores = prediction.keypoint_scores
        fo_keypoints = []
        H,W = coco_sample.metadata["height"], coco_sample.metadata["width"]
        for i in range(len(keypoints)//3):
            u,v,vis = keypoints[i*3:(i+1)*3]
            if vis == 0:
                continue
            u = u / W
            v = v /H
            confidence = keypoint_scores[i]
            fo_keypoints.append(fo.Keypoint(points=[[u,v]],label = category_keypoint_names[i], confidence=[confidence], name=category_keypoint_names[i]))
        coco_sample["predictions"] = fo.Keypoints(keypoints=fo_keypoints)
        coco_sample.save()



    session = fo.launch_app(dataset)
    session.wait()
    
  


if __name__ == "__main__":
    # Example usage: fill in your paths here
    config = Config(
        images_dir="/home/tlips/Code/few-shot-keypoints/data/aRTF/tshirts-test_resized_512x256",
        labels_path="/home/tlips/Code/few-shot-keypoints/data/aRTF/tshirts-test_resized_512x256/tshirts-test.json",
        results_path="/home/tlips/Code/few-shot-keypoints/results/aRTF-support-sets/dino/tshirt/1/resize_2025_results.json",
        dataset_name="coco-keypoints-gt"
    )

    # config = Config(
    #     labels_path="/home/tlips/Code/few-shot-keypoints/data/SPair-71k/SPAIR_coco_tvmonitor_test.json",
    #     results_path="/home/tlips/Code/few-shot-keypoints/results/SPAIR-support-sets/dino/tvmonitor/1/resize_2026_results.json",
    #     images_dir="/home/tlips/Code/few-shot-keypoints/data/SPair-71k"
    # )

    # config = Config(
    #     labels_path="/home/tlips/Code/few-shot-keypoints/data/SPair-71k/SPAIR_coco_aeroplane_test.json",
    #     results_path="/home/tlips/Code/few-shot-keypoints/results/SPAIR-support-sets/dino/aeroplane/1/resize_2026_results.json",
    #     images_dir="/home/tlips/Code/few-shot-keypoints/data/SPair-71k"
    # )

    main(config)
