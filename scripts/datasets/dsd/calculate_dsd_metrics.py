"""Calculate metrics for DSD matching results."""

import glob
import json
import os
from functools import partial

import pandas as pd

from few_shot_keypoints.datasets.data_parsers import CocoKeypointsResultDataset
from airo_dataset_tools.data_parsers.coco import CocoKeypointsDataset
from few_shot_keypoints.paths import (
    DSD_SHOE_TEST_JSON, DSD_MUGS_TEST_JSON
)
from few_shot_keypoints.results import (
    match_keypoints,
    calculate_average_keypoint_distance,
    calculate_median_keypoint_distance,
    calculate_mAP,
    is_TP_by_fraction_of_max_bbox_size,
    is_TP_by_distance,
    calculate_image_PCK,
    calculate_point_PCK,
)

# Map category names to their test JSON paths
# The category name corresponds to the directory name used by `match_dataset`,
# which for DSD defaults to the COCO category name (e.g. "shoe", "mug").
CATEGORY_TEST_PATHS = {
    "shoe": str(DSD_SHOE_TEST_JSON),
    "mug": str(DSD_MUGS_TEST_JSON),
}

def main():
    data_dir = "results/DSD"
    
    results = []
    jsons = glob.glob(f"{data_dir}/**/*.json", recursive=True)
    
    for abs_path in jsons:
        rel_path = os.path.relpath(abs_path, data_dir)
        parts = rel_path.split("/")
        
        # Structure: featurizer/category/processing_seed_results.json
        featurizer = parts[0]
        category = parts[1]
        processing, seed = parts[2].split("_")[0:2]
        
        print(f"\nProcessing: {rel_path}")
        
        # Get test dataset path based on category
        test_dataset_path = CATEGORY_TEST_PATHS.get(category)
        if test_dataset_path is None:
            print(f"  Unknown category: {category}, skipping...")
            continue
        
        # Load results and ground truth
        with open(abs_path, "r") as f:
            coco_results = CocoKeypointsResultDataset(json.load(f))
        
        with open(test_dataset_path, "r") as f:
            coco_dataset = CocoKeypointsDataset(**json.load(f))
        
        # Match predictions to ground truth
        matched_predictions = match_keypoints(coco_dataset, coco_results)
        
        # Calculate metrics
        row = {
            "featurizer": featurizer,
            "processing": processing,
            "seed": seed,
            "category": category,
            "results_path": abs_path,
            # Average keypoint distance
            "avg_distance_all": calculate_average_keypoint_distance(matched_predictions, visible_only=False),
            "avg_distance_visible": calculate_average_keypoint_distance(matched_predictions, visible_only=True),
            # Median keypoint distance
            "median_distance_all": calculate_median_keypoint_distance(matched_predictions, visible_only=False),
            "median_distance_visible": calculate_median_keypoint_distance(matched_predictions, visible_only=True),
            
            # PCK
            "image_PCK": calculate_image_PCK(matched_predictions),
            "image_PCK_visible": calculate_image_PCK(matched_predictions, visible_only=True),
            "point_PCK": calculate_point_PCK(matched_predictions),
            "point_PCK_visible": calculate_point_PCK(matched_predictions, visible_only=True),

            # mAP l2 distance
            "mAP_l2_distance": calculate_mAP(matched_predictions, partial(is_TP_by_distance, px_threshold=10)),
            "mAP_bbox_alpha": calculate_mAP(matched_predictions, partial(is_TP_by_fraction_of_max_bbox_size, alpha=0.1)),
        }
        results.append(row)
        
        # Print metrics
        print(f"  Avg distance (all):     {row['avg_distance_all']:.2f} px")
        print(f"  Avg distance (visible): {row['avg_distance_visible']:.2f} px")
        # print(f"  Median distance (all):     {row['median_distance_all']:.2f} px")
        # print(f"  Median distance (visible): {row['median_distance_visible']:.2f} px")
        # print(f"  mAP@10px:                {row['mAP_l2_distance']:.4f}")
        print(f" Image PCK:                {row['image_PCK']:.4f}")
        print(f" Image PCK visible:        {row['image_PCK_visible']:.4f}")
        # print(f" Point PCK:                {row['point_PCK']:.4f}")
        # print(f" Point PCK visible:        {row['point_PCK_visible']:.4f}")
    
    # Create DataFrame and save
    df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    print(df.to_string(index=False))
    
    # Save to CSV
    output_path = f"{data_dir}/metrics.csv"
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()

