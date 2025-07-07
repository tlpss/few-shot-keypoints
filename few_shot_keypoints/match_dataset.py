import json
import os
from typing import List, Tuple
import numpy as np
from few_shot_keypoints.datasets.coco_dataset import TorchCOCOKeypointsDataset
from few_shot_keypoints.featurizers.dift_featurizer import SDFeaturizer
from few_shot_keypoints.matcher import KeypointFeatureMatcher, MultiQueryKeypointFeatureMatcher
from few_shot_keypoints.featurizers.ViT_featurizer import ViTFeaturizer
from few_shot_keypoints.featurizers.combined_featurizer import CombinedFeaturizer
from tqdm import trange
from dataclasses import dataclass
import draccus

def get_dataset_predictions(dataset: TorchCOCOKeypointsDataset, matcher: KeypointFeatureMatcher, keypoint_index: int):
    ground_truth_keypoints = []
    predictions = []
    for i in trange(len(dataset)):
        img, keypoints = dataset[i]
        img = img.unsqueeze(0)
        keypoint = keypoints[keypoint_index]
        if len(keypoint) == 1:
            u,v = matcher.get_best_match(img)
            predictions.append((u,v))
            ground_truth_keypoints.append(keypoint[0])
        elif len(keypoint) == 0:
            pass 
            # ignore invisible keypoints
        else:
            pass 
            # ignore multiple instances for now.
    return ground_truth_keypoints, predictions
        

def compute_L2_distances(ground_truth_keypoints: List[Tuple[int,int]], predictions: List[Tuple[int,int]]) -> np.ndarray:
    distances = []
    for gt, pred in zip(ground_truth_keypoints, predictions):
        distances.append(np.linalg.norm(np.array(gt) - np.array(pred)))
    return np.array(distances)

def compute_L2_distance_statistics(ground_truth_keypoints: List[Tuple[int,int]], predictions: List[Tuple[int,int]]):
    distances = compute_L2_distances(ground_truth_keypoints, predictions)
    return {
        "mean": np.mean(distances),
        "std": np.std(distances),
        "median": np.median(distances),
    }

def get_statistics(dataset, dataset_indices: list, keypoint_index: int):
    query_images = [dataset[idx][0].unsqueeze(0) for idx in dataset_indices]
    query_keypoints = [dataset[idx][1][keypoint_index][0] for idx in dataset_indices]

    #matcher = MultiQueryKeypointFeatureMatcher(SDFeaturizer())
    matcher = MultiQueryKeypointFeatureMatcher(ViTFeaturizer("facebook/dinov2-small", [-1]))
    #matcher = MultiQueryKeypointFeatureMatcher(CombinedFeaturizer([SDFeaturizer(), ViTFeaturizer("facebook/dinov2-small", [11])]))
    for img, kp in zip(query_images, query_keypoints):
        matcher.add_reference_image(img, kp)
    ground_truth_keypoints, predictions = get_dataset_predictions(dataset, matcher, keypoint_index)
    stats_dict = compute_L2_distance_statistics(ground_truth_keypoints, predictions)
    # Add all reference image paths and keypoints to the stats_dict
    stats_dict["query_image_paths"] = [dataset.dataset[idx][0] for idx in dataset_indices] 
    stats_dict["query_keypoints"] = [dataset[idx][1][keypoint_index][0] for idx in dataset_indices]
    return stats_dict

@dataclass
class Config:
    dataset_category: str = "train_test"
    N_query_sets: int = 4
    N_query_images: int = 1

@draccus.wrap()
def main(config: Config):
    dataset = TorchCOCOKeypointsDataset(f"/home/tlips/Code/droid/data/SPair-71k/SPAIR_coco_{config.dataset_category}.json")
    np.random.seed(2025)
    print(f" running for {len(dataset[0][1])} keypoints")
    file_name = f"results/SPAIR-query-sets/dino_{config.dataset_category}_#q{config.N_query_images}_statistics.json"

    for keypoint_index in trange(len(dataset[0][1])):
        print(f"keypoint index: {keypoint_index}")

        # get nmuber of images that have visible keypoints for this keypoint
        n_visible_keypoints = 0
        for i in range(len(dataset)):
            if len(dataset[i][1][keypoint_index]) == 1:
                n_visible_keypoints += 1
        print(f"number of images with visible keypoints for keypoint {keypoint_index}: {n_visible_keypoints}")

        # find number of runs to do
        n_runs = config.N_query_sets
        if os.path.exists(file_name):
            statistics = json.load(open(file_name))
            if str(keypoint_index) in statistics:
                n_runs = n_runs - len(statistics[str(keypoint_index)])
        print(f" doing {n_runs} runs for keypoint {keypoint_index}")
 
        # do the runs
        for _ in range(n_runs):

            if os.path.exists(file_name):
                statistics = json.load(open(file_name))
            else:
                statistics = {}
            if str(keypoint_index) not in statistics:
                statistics[str(keypoint_index)] = {}
            statistics[str(keypoint_index)]["n_visible_keypoints"] = n_visible_keypoints

            dataset_indices = None
            if n_visible_keypoints == 0:
                print("no images with visible keypoints for this keypoint type")
                continue
            while dataset_indices is None:
                candidate_indices = np.random.choice(len(dataset), size=config.N_query_images, replace=False)
                # check if kp is visible and not already in statistics
                #print([dataset[idx][1][keypoint_index] for idx in candidate_indices])
                if all(len(dataset[idx][1][keypoint_index]) == 1 for idx in candidate_indices) and all(str(idx) not in statistics[str(keypoint_index)] for idx in candidate_indices):
                    dataset_indices = candidate_indices

            statistics[str(keypoint_index)][str(tuple(dataset_indices))] = get_statistics(dataset, dataset_indices, keypoint_index)
            statistics[str(keypoint_index)][str(tuple(dataset_indices))]["image_paths"] = [dataset.dataset[idx][0] for idx in dataset_indices]
            with open(file_name, "w") as f:
                json.dump(statistics, f, indent=4)
            
if __name__ == "__main__":

    main()
    