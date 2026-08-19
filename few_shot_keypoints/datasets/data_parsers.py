from typing import List, Optional
from airo_dataset_tools.data_parsers.coco import CocoInstanceAnnotation, CocoKeypointsDataset
from pydantic import RootModel

class CocoKeypointsResultAnnotation(CocoInstanceAnnotation):
    keypoints: List[float]
    score: float # overall confidence, required by the COCO format.
    keypoint_scores: Optional[List[float]] = None # additional, optional field to store the confidence of the keypoints separately.
    # additional, optional field to store all top-k matches of each keypoint channel, as [u,v,score] triplets,
    # ordered from best to worst match. COCO only allows a single keypoint per category, so the `keypoints` field
    # always contains the best match of each channel. Only relevant if the matcher was configured with top_k > 1.
    keypoint_topk_matches: Optional[List[List[List[float]]]] = None
    category_id: int
    image_id: int

class CocoKeypointsResultDataset(RootModel[List[CocoKeypointsResultAnnotation]]):
    def __getitem__(self, index):
        return self.root[index]


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
    print(dummy_dataset[0])