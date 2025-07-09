from typing import List, Optional
from airo_dataset_tools.data_parsers.coco import CocoInstanceAnnotation, CocoKeypointsDataset
from pydantic import RootModel

class CocoKeypointsResultAnnotation(CocoInstanceAnnotation):
    keypoints: List[int]
    score: float # overall confidence, required by the COCO format.
    keypoint_scores: Optional[List[float]] = None # additional, optional field to store the confidence of the keypoints separately.


class CocoKeypointsResultDataset(RootModel[List[CocoKeypointsResultAnnotation]]):
    pass 


if __name__ == "__main__":

    dummy = [ {
        "id": 1,
        "image_id": 1,
        "category_id": 1,
        "bbox": [1,2,3,4],
        "score": 0.9,
        "keypoints": [1,2,3,4,5,6,7,8,9,10,11,12],
        "keypoint_scores": [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
    },
    {
        "id": 2,
        "image_id": 1,
        "category_id": 1,
        "bbox": [1,2,3,4],
        "score": 0.9,
        "keypoints": [1,2,3,4,5,6,7,8,9,10,11,12],
        "keypoint_scores": [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
    }]

    dummy_dataset = CocoKeypointsResultDataset(dummy)
    print(dummy_dataset)