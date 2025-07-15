from dataclasses import dataclass
import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.utils.coco as fouc
import os

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
        label_types=["keypoints"],
        include_id=True,
        include_annotation_id=True,
    )

    print(dataset.info)
    
    # add the predictions
    fouc.add_coco_labels(
        dataset,
        "predictions",
        config.results_path,
        categories = dataset.info["categories"],
        coco_id_field="coco_id",
        label_type="keypoints",

    )
    # Launch the FiftyOne App

    # Launch FiftyOne app
    print("Launching FiftyOne...")
    session = fo.launch_app(dataset)
    session.wait()


if __name__ == "__main__":
    # Example usage: fill in your paths here
    # config = Config(
    #     images_dir="/home/tlips/Code/few-shot-keypoints/data/aRTF/tshirts-test_resized_512x256",
    #     labels_path="/home/tlips/Code/few-shot-keypoints/data/aRTF/tshirts-test_resized_512x256/tshirts-test.json",
    #     results_path="/home/tlips/Code/few-shot-keypoints/results/aRTF-support-sets/dino/tshirt/1/resize_2025_results.json",
    #     dataset_name="coco-keypoints-gt"
    # )

    config = Config(
        labels_path="/home/tlips/Code/few-shot-keypoints/data/SPair-71k/SPAIR_coco_tvmonitor_test.json",
        results_path="/home/tlips/Code/few-shot-keypoints/results/SPAIR-support-sets/dino/tvmonitor/1/resize_2026_results.json",
        images_dir="/home/tlips/Code/few-shot-keypoints/data/SPair-71k"
    )

    main(config)
