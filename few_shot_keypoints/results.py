import dataclasses
from typing import List, Optional, Tuple
from airo_dataset_tools.data_parsers.coco import CocoInstanceAnnotation, CocoKeypointsDataset
from few_shot_keypoints.datasets.data_parsers import CocoKeypointsResultDataset
from few_shot_keypoints.matcher import KeypointListMatcher
import numpy as np

@dataclasses.dataclass
class PredictedAndGTKeypoint:

    """

    gt & pred: TP or FP or FN, depending on confidence/score (low confidence is actually no prediction and hence FNand distance tresholds (TP/FP)
    gt & no pred: FN
    no gt & pred: FP
    no gt & no pred: TN, these are not considered explicitly.
    """
    keypoint_category: str # name of the keypoint.
    image_id: int
    predicted_keypoint: Optional[Tuple[int,int]]
    gt_keypoint: Optional[Tuple[int,int]]
    gt_visible: bool
    keypoint_score: Optional[float]

    # these are used for relative metrics such as PCK or normalized distances
    bbox: Optional[Tuple[int,int,int,int]] # bbox of the gt instance (u,v,w,h)
    size: Optional[int] # size of instance in pixels A




def get_gt_and_predicted_keypoints(coco_dataset: CocoKeypointsDataset, coco_results_dataset: CocoKeypointsResultDataset):
    
    # # check if the category matches the category of the results.
    src_categories = coco_dataset.categories
    # dst_categories = coco_results_dataset.categories

    # assert len(src_categories) == len(dst_categories) == 1, "Only one category is supported for now."
    # assert src_categories[0].name == dst_categories[0].name, "The categories of the source and destination datasets must match."
    # assert src_categories[0].keypoints == dst_categories[0].keypoints, "The keypoints of the source and destination datasets must match."


    # # check if the images are the same
    # src_images = coco_dataset.parsed_coco.images
    # dst_images = coco_results_dataset.images
    # assert len(src_images) == len(dst_images), "The number of images in the source and destination datasets must match."
    # for src_image, dst_image in zip(src_images, dst_images):
    #     assert src_image.id == dst_image.id, "The images must have the same id."
    #     assert src_image.file_name == dst_image.file_name, "The images must have the same file name."
    #     assert src_image.width == dst_image.width, "The images must have the same width."


    gt_image_id_to_annotations = {}
    # there is only one per category.
    for annotation in coco_dataset.annotations:
        if annotation.image_id in gt_image_id_to_annotations:
            raise ValueError(f"Multiple annotations for image {annotation.image_id} in the source dataset.")
        gt_image_id_to_annotations[annotation.image_id] = annotation
    
    pred_image_id_to_annotations = {}
    for annotation in coco_results_dataset.root:
        if annotation.image_id in pred_image_id_to_annotations:
            raise ValueError(f"Multiple annotations for image {annotation.image_id} in the destination dataset.")
        pred_image_id_to_annotations[annotation.image_id] = annotation


    matched_predictions = []
    for image_id, pred_annotation in pred_image_id_to_annotations.items():
        gt_annotation = gt_image_id_to_annotations[image_id]
        for kp_type_idx, kp_type in enumerate(src_categories[0].keypoints):
            gt_keypoint = gt_annotation.keypoints[kp_type_idx*3:kp_type_idx*3+3]
            pred_keypoint = pred_annotation.keypoints[kp_type_idx*3:kp_type_idx*3+3]

            u_pred, v_pred, vis_pred = pred_keypoint
            score_pred = pred_annotation.keypoint_scores[kp_type_idx] if pred_annotation.keypoint_scores is not None else pred_annotation.score
            u_gt, v_gt, vis_gt = gt_keypoint

            pred_kp = (u_pred,v_pred)
            gt_kp = (u_gt,v_gt)

            if vis_gt == 0:
                gt_kp = None
            if vis_pred == 0:
                pred_kp = None

            matched_prediction = PredictedAndGTKeypoint(keypoint_category=kp_type, image_id=image_id, predicted_keypoint=pred_kp, gt_keypoint=gt_kp, gt_visible=vis_gt==2, keypoint_score=score_pred, bbox=gt_annotation.bbox, size=gt_annotation.area)
            matched_predictions.append(matched_prediction)
    
    return matched_predictions

    # for each result

    # look for GT instance
    # if no GT, add (None, pred, score) for each kp in result (FP)

    # for each type in annotation category:
    # if gt and pred, add (gt, pred, score)
    # if gt and no pred, add (gt, None, score) (FN) (this cannot happen in theory, there is always a pred, though it can have very low confidence)
    # if no gt and pred, add (None, pred, score) (FP)
    # if no gt and no pred, add (None, None, score), don't do anything?
    
def calculate_point_PCK(matched_predictions: List[PredictedAndGTKeypoint], threshold: float = 0.1):
    """
    PCK is the percentage of keypoints that are within a certain threshold of the ground truth.
    point PCK is the average of the PCK for all predictions in the dataset.
    """
    n_correct = 0
    n_total = 0
    for prediction in matched_predictions:
        if prediction.gt_visible:
            n_total += 1
            if prediction.predicted_keypoint is not None:
                dist = np.linalg.norm(np.array(prediction.predicted_keypoint) - np.array(prediction.gt_keypoint))
                bbox_dims = np.array(prediction.bbox[2:])
                if dist / max(bbox_dims) <= threshold:
                    n_correct += 1
    return n_correct / n_total

def calculate_image_PCK(matched_predictions: List[PredictedAndGTKeypoint], threshold: float = 0.1):
    """
    PCK is the percentage of keypoints that are within a certain threshold of the ground truth.
    image PCK is the average of the PCK of each image.

    """
    # group predictions by image_id
    predictions_by_image_id = {}
    for prediction in matched_predictions:
        if prediction.image_id not in predictions_by_image_id:
            predictions_by_image_id[prediction.image_id] = []
        predictions_by_image_id[prediction.image_id].append(prediction)
    
    # for each image, calculate the PCK
    pcks = []
    for image_id, predictions in predictions_by_image_id.items():
        pcks.append(calculate_point_PCK(predictions, threshold))
    return np.mean(pcks)


def get_keypoint_distances(matched_predictions: List[PredictedAndGTKeypoint]):
    """
    Get the distances of all predictions in the dataset.
    """
    distances = []
    for prediction in matched_predictions:
        if prediction.gt_visible:
            if prediction.predicted_keypoint is not None:
                distances.append(np.linalg.norm(np.array(prediction.predicted_keypoint) - np.array(prediction.gt_keypoint)))
    return distances

def calculate_average_keypoint_distance(matched_predictions: List[PredictedAndGTKeypoint]):
    """
    Average keypoint distance is the average distance of all predictions in the dataset.
    """
    distances = get_keypoint_distances(matched_predictions)
    return np.mean(distances)

def calculate_median_keypoint_distance(matched_predictions: List[PredictedAndGTKeypoint]):
    """
    Median keypoint distance is the median distance of all predictions in the dataset.
    """
    distances = get_keypoint_distances(matched_predictions)
    return np.median(distances)


def filter_predictions_by_keypoint_category(matched_predictions: List[PredictedAndGTKeypoint], keypoint_category: str):
    """
    Filter predictions by keypoint category.
    """
    return [prediction for prediction in matched_predictions if prediction.keypoint_category == keypoint_category]


if __name__ == "__main__":
    from few_shot_keypoints.datasets.data_parsers import CocoKeypointsResultDataset
    from airo_dataset_tools.data_parsers.coco import CocoKeypointsDataset
    from few_shot_keypoints.matcher import KeypointListMatcher
    import json

    coco_json = "/home/tlips/Code/few-shot-keypoints/data/SPair-71k/SPAIR_coco_tvmonitor_test.json"
    coco_results_json = "/home/tlips/Code/few-shot-keypoints/test.json"

    with open(coco_results_json, "r") as f:
        coco_results_dataset = CocoKeypointsResultDataset(json.load(f))
    
    with open(coco_json, "r") as f:
        coco_dataset = CocoKeypointsDataset(**json.load(f))

    matched_predictions = get_gt_and_predicted_keypoints(coco_dataset, coco_results_dataset)
    print(matched_predictions)
    print(len(matched_predictions))
    print(matched_predictions[0])
    print(f"point PCK: {calculate_point_PCK(matched_predictions)}")
    print(f"image PCK: {calculate_image_PCK(matched_predictions)}")
    print(f"average keypoint distance: {calculate_average_keypoint_distance(matched_predictions)}")
    print(f"median keypoint distance: {calculate_median_keypoint_distance(matched_predictions)}")