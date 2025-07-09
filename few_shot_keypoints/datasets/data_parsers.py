from typing import List, Optional
from airo_dataset_tools.data_parsers.coco import CocoInstanceAnnotation, CocoKeypointsDataset

class CocoKeypointsResultAnnotation(CocoInstanceAnnotation):
    keypoints: List[int]
    score: float # overall confidence, required by the COCO format.
    keypoint_scores: Optional[List[float]] = None # additional, optional field to store the confidence of the keypoints separately.


class CocoKeypointsResultDataset(CocoKeypointsDataset):
    annotations: List[CocoKeypointsResultAnnotation]
    