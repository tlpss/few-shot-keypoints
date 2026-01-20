from functools import partial
import glob
import os
import pandas as pd
data_dir = "results/SPAIR-correspondences" 
from few_shot_keypoints.datasets.coco_dataset import TorchCOCOKeypointsDataset, CocoKeypointsDataset
from few_shot_keypoints.datasets.data_parsers import CocoKeypointsResultDataset
from few_shot_keypoints.results import calculate_mAP, is_TP_by_fraction_of_max_bbox_size, match_keypoints, calculate_image_PCK, calculate_median_keypoint_distance, calculate_average_keypoint_distance
import json 
#  featurizer / category / N_support_images / transform_seed_results.json
# create dataframe with columns: featurizer, category, N_support_images, results_json_path

d = []
jsons = glob.glob(f"{data_dir}/**/*.json", recursive=True)
for abs_path in jsons:
    # get relative to data_dir
    rel_path = os.path.relpath(abs_path, data_dir)
    print(rel_path)
    featurizer = rel_path.split("/")[0]
    category = rel_path.split("/")[1]
    N_support_images = rel_path.split("/")[2]
  
    results_json_path = abs_path
    d.append({"featurizer": featurizer, "category": category, "N_support_images": N_support_images, "results_json_path": results_json_path})

# add test dataset path to each dict
for row in d:
    test_dataset_path = f"/home/tlips/Code/few-shot-keypoints/data/SPair-71k/SPAIR_coco_{row['category']}_test.json"
    row["test_dataset_path"] = test_dataset_path

# calculate metrics.
# for each row, load the results_json_path and calculate the metrics
for row in d:
    with open(row["results_json_path"], "r") as f:
        results = json.load(f)
    coco_results_dataset = CocoKeypointsResultDataset(results)
    with open(row["test_dataset_path"], "r") as f:  
        test_dataset = json.load(f)
    coco_dataset = CocoKeypointsDataset(**test_dataset)
    matched_predictions = match_keypoints(coco_dataset, coco_results_dataset)
    image_PCK = calculate_image_PCK(matched_predictions)
    median_keypoint_distance = calculate_median_keypoint_distance(matched_predictions)
    average_keypoint_distance = calculate_average_keypoint_distance(matched_predictions)
    row["image_PCK"] = image_PCK
    row["image_PCK_visible_only"] = calculate_image_PCK(matched_predictions, visible_only=True)
    row["mAP_bbox_alpha"] = calculate_mAP(matched_predictions, partial(is_TP_by_fraction_of_max_bbox_size, alpha=0.1))
    row["median_keypoint_distance"] = median_keypoint_distance
    row["average_keypoint_distance"] = average_keypoint_distance
# save to csv
df = pd.DataFrame(d)
print(df)

# save to csv
df.to_csv("results/SPAIR-correspondences/SPAIR-correspondences.csv", index=False)

