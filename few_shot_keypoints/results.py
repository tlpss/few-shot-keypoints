import dataclasses
from functools import partial
from typing import Callable, List, Optional, Tuple
from airo_dataset_tools.data_parsers.coco import CocoInstanceAnnotation, CocoKeypointsDataset
from few_shot_keypoints.datasets.data_parsers import CocoKeypointsResultDataset
import numpy as np
import sklearn


"""
predictions can be matched to the GT keypoints as follows:

for each prediction, calculate a metric for all GT predictions of that category.
if closest one is within threshold -> matched. This is TP.
else: if not -> FP.
in the end, if any of the GT keypoints remain, they are FNs.

TNs are not considered explicitly for detection metrics.
"""

@dataclasses.dataclass
class MatchedPredictionKeypoint:
    """ a predicted keypoint that has been matched to its GT keypoint.
    Can be either TP or FP ,depending on the metric and threshold used for classification."""

    keypoint_category: str # name of the keypoint.
    image_id: int
    predicted_keypoint: Optional[Tuple[int,int]]
    keypoint_score: Optional[float] # also called 'confidence'

    gt_keypoint: Optional[Tuple[int,int]] # the **matched** gt keypoint. If this is None, this is a FP.
    gt_visible: bool # used to optionally calculate metrics only for visible keypoints.

    # these are used for relative metrics such as PCK or normalized distances
    bbox: Optional[Tuple[int,int,int,int]] # bbox of the gt instance (u,v,w,h)
    size: Optional[int] # size of instance in pixels A



def match_keypoints(coco_dataset: CocoKeypointsDataset, coco_results_dataset: CocoKeypointsResultDataset):
    """ 

    matches the predictions to the GT keypoints.
    assuming exactly one annotation per image for both GT and predictions. but some of the KP could be invisible/ occluded.
    so there can be no FNs, only FPs and TPs.
    """
    
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
        gt_annotation = gt_image_id_to_annotations[image_id] # this could be non-existent! 
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

            is_gt_visible = vis_gt > 1.5 # 2 is visible, 1 is occluded, 0 is not in view.
            matched_prediction = MatchedPredictionKeypoint(keypoint_category=kp_type, image_id=image_id, predicted_keypoint=pred_kp, gt_keypoint=gt_kp, gt_visible=is_gt_visible, keypoint_score=score_pred, bbox=gt_annotation.bbox, size=gt_annotation.area)
            matched_predictions.append(matched_prediction)
    
    return matched_predictions

    
def calculate_point_PCK(matched_predictions: List[MatchedPredictionKeypoint], threshold: float = 0.1, visible_only: bool = False):
    """
    PCK is the percentage of keypoints that are within a certain threshold of the ground truth.
    point PCK is the average of the PCK for all predictions in the dataset.
    """
    n_correct = 0
    n_total = 1e-10
    for prediction in matched_predictions:
        if prediction.gt_keypoint is None:
            # this is a FP
            #n_total += 1
            continue
        if visible_only and prediction.gt_visible is False:
            # these would be FPs because there is no GT keypoint here.
            #n_total += 1
            continue
        n_total += 1
        if prediction.predicted_keypoint is not None: #otherwise, this would be a FN.
            dist = np.linalg.norm(np.array(prediction.predicted_keypoint) - np.array(prediction.gt_keypoint))
            bbox_dims = np.array(prediction.bbox[2:])
            if dist / max(bbox_dims) <= threshold:
                n_correct += 1
        else:
            print("FNs are not measured by PCK!")
    return n_correct / n_total

def calculate_image_PCK(matched_predictions: List[MatchedPredictionKeypoint], threshold: float = 0.1, visible_only: bool = False):
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
        pcks.append(calculate_point_PCK(predictions, threshold, visible_only))
    return np.mean(pcks)


def get_keypoint_distances(matched_predictions: List[MatchedPredictionKeypoint],visible_only: bool = False):
    """
    Get the distances of all predictions in the dataset.
    """
    distances = []
    for prediction in matched_predictions:
        if prediction.gt_keypoint is None:
            # this is a FP
            continue
        if visible_only and prediction.gt_visible is False:
            # these would be FPs because there is no GT keypoint here.
            continue
        if prediction.predicted_keypoint is not None:
            distances.append(np.linalg.norm(np.array(prediction.predicted_keypoint) - np.array(prediction.gt_keypoint)))
    return distances


def calculate_mAP(matched_predictions: List[MatchedPredictionKeypoint], tp_classifier: Callable[[MatchedPredictionKeypoint], bool], total_gt_keypoints: int = None):
    """
    Calculate the mAP of the predictions averaged over the different categories.
    """
    # get all keypoint categories
    aps = []
    keypoint_categories = set([prediction.keypoint_category for prediction in matched_predictions])
    for keypoint_category in keypoint_categories:
        local_matched_predictions = filter_predictions_by_keypoint_category(matched_predictions, keypoint_category)
        if len(local_matched_predictions) == 0:
            print(f"No predictions for keypoint category {keypoint_category}")
            continue
        local_total_gt_keypoints = len([prediction for prediction in local_matched_predictions if prediction.gt_keypoint is not None])
        local_tp_predictions = [prediction for prediction in local_matched_predictions if tp_classifier(prediction)]
        local_fp_predictions = [prediction for prediction in local_matched_predictions if not tp_classifier(prediction)]    
        local_prediction_scores = [kp.keypoint_score for kp in local_tp_predictions] + [kp.keypoint_score for kp in local_fp_predictions]
        local_prediction_labels = [1] * len(local_tp_predictions) + [0] * len(local_fp_predictions)
        # the AP calculation in sklearn is made for binary classifications, in which the number of predctions == number of GTs.
        # for detections, this is not the case, so we need to 'rescale' the recall values.
        # R = TP / (TP + FN). And TP + FN was != total_gt_keypoints (but rather )
        local_ap = sklearn.metrics.average_precision_score(local_prediction_labels, local_prediction_scores) 
        
        if total_gt_keypoints is not None:
            local_ap = local_ap* total_gt_keypoints / local_total_gt_keypoints
        aps.append(local_ap)
    return np.mean(aps)

def is_TP_by_fraction_of_max_bbox_size(x: MatchedPredictionKeypoint, alpha: float = 0.1):
    """ calculating mAP with this metric is the APK, as defined in 
    https://escholarship.org/content/qt7sk1s10g/qt7sk1s10g_noSplash_a8d7d492292a22ca3c20c0c99cbd9d1f.pdf
    """
    if x.gt_keypoint is None:
        return False
    distance = np.linalg.norm(np.array(x.predicted_keypoint) - np.array(x.gt_keypoint))
    threshold = alpha * max(x.bbox[2:])
    return distance / threshold < 1.0

def is_TP_by_distance(x: MatchedPredictionKeypoint, px_threshold: float = 20):
    if x.gt_keypoint is None:
        return False
    distance = np.linalg.norm(np.array(x.predicted_keypoint) - np.array(x.gt_keypoint))
    return distance < px_threshold



def calculate_average_keypoint_distance(matched_predictions: List[MatchedPredictionKeypoint], visible_only: bool = False):
    """
    Average keypoint distance is the average distance of all predictions in the dataset.
    """
    distances = get_keypoint_distances(matched_predictions, visible_only)
    return np.mean(distances)

def calculate_median_keypoint_distance(matched_predictions: List[MatchedPredictionKeypoint], visible_only: bool = False):
    """
    Median keypoint distance is the median distance of all predictions in the dataset.
    """
    distances = get_keypoint_distances(matched_predictions, visible_only)
    return np.median(distances)

def filter_predictions_by_keypoint_category(matched_predictions: List[MatchedPredictionKeypoint], keypoint_category: str):
    """
    Filter predictions by keypoint category.
    """
    return [prediction for prediction in matched_predictions if prediction.keypoint_category == keypoint_category]


def calculate_confidence_correlation(matched_predictions: List[MatchedPredictionKeypoint]):
    """
    Calculate the correlation between the confidence and the distance of the predictions.
    """
    distances = get_keypoint_distances(matched_predictions)
    confidences = [prediction.keypoint_score for prediction in matched_predictions if (prediction.predicted_keypoint is not None and prediction.gt_visible)]
    return distances, confidences

if __name__ == "__main__":
    from few_shot_keypoints.datasets.data_parsers import CocoKeypointsResultDataset
    from airo_dataset_tools.data_parsers.coco import CocoKeypointsDataset
    import json
    from few_shot_keypoints.paths import DSD_MUGS_TEST_JSON, DSD_SHOE_TEST_JSON, KIL_MUGS_V2_INITIAL_JSON

    coco_results_json = "/home/tlips/Code/few-shot-keypoints/test_crop_matching.json"
    # coco_results_json = "/home/tlips/Code/few-shot-keypoints/results/SPAIR-support-sets/dino/aeroplane/1/resize_2028_results.json"
    # coco_json = "/home/tlips/Code/few-shot-keypoints/data/aRTF/tshirts-test_resized_512x256/tshirts-test.json"
    # coco_results_json = "/home/tlips/Code/few-shot-keypoints/results/aRTF-support-sets/dift/tshirt/1/resize_2025_results.json"
    coco_json = KIL_MUGS_V2_INITIAL_JSON
    with open(coco_results_json, "r") as f:
        coco_results_dataset = CocoKeypointsResultDataset(json.load(f))
    
    with open(coco_json, "r") as f:
        coco_dataset = CocoKeypointsDataset(**json.load(f))

    matched_predictions = match_keypoints(coco_dataset, coco_results_dataset)
    print(len(matched_predictions))
    print(matched_predictions[0])
    # print(f"point PCK: {calculate_point_PCK(matched_predictions)}")
    # print(f"image PCK: {calculate_image_PCK(matched_predictions)}")
    # print(f"image PCK visible only: {calculate_image_PCK(matched_predictions, visible_only=True)}")
    print(f"average keypoint distance: {calculate_average_keypoint_distance(matched_predictions)}")
    print(f"average keypoint distance visible only: {calculate_average_keypoint_distance(matched_predictions, visible_only=True)}")
    print(f"median keypoint distance: {calculate_median_keypoint_distance(matched_predictions)}")
    print(f"median keypoint distance visible only: {calculate_median_keypoint_distance(matched_predictions, visible_only=True)}")





    # print(f"mAP bbox_alpha: {calculate_mAP(matched_predictions, is_TP_by_fraction_of_max_bbox_size)}")
    # print(f"mAP L2: {calculate_mAP(matched_predictions, partial(is_TP_by_distance, px_threshold=10))}")


