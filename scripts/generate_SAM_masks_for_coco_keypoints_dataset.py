#!/usr/bin/env python3
"""
Generate SAM3 masks for a COCO keypoints dataset and create a new COCO dataset with segmentation masks.
Uses the keypoints as point prompts to segment the object.

SAM3 Tracker API reference: https://huggingface.co/facebook/sam3
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Sam3TrackerProcessor, Sam3TrackerModel

from few_shot_keypoints.datasets.coco_dataset import TorchCOCOKeypointsDataset
from airo_dataset_tools.segmentation_mask_converter import BinarySegmentationMask


@dataclass
class Config:
    """Configuration for SAM3 mask generation."""
    dataset: str  # Path to COCO keypoints JSON file
    output: str   # Output path for new COCO JSON with masks
    device: str = "cuda"  # Device to run on


def extract_visible_keypoints(keypoints: list) -> list:
    """
    Extract visible keypoints from the nested keypoint structure.
    keypoints: List of channels, each channel is a list of keypoints [[u, v], ...]
    Returns: list of [u, v] coordinates for visible keypoints
    """
    visible_points = []
    for channel in keypoints:
        for kp in channel:
            if len(kp) >= 2:
                visible_points.append([kp[0], kp[1]])
    return visible_points


def generate_sam_masks(config: Config):
    """Generate SAM3 masks for a COCO keypoints dataset."""
    
    output_path = Path(config.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load original COCO JSON
    with open(config.dataset, "r") as f:
        coco_data = json.load(f)
    
    # Load dataset (without transforms to get original coordinates)
    dataset = TorchCOCOKeypointsDataset(config.dataset, transform=None)
    
    # Load SAM3 model
    print("Loading SAM3 model...")
    device = config.device if torch.cuda.is_available() else "cpu"
    model = Sam3TrackerModel.from_pretrained("facebook/sam3").to(device)
    processor = Sam3TrackerProcessor.from_pretrained("facebook/sam3")
    print(f"SAM3 loaded on {device}")

    # Track which annotations we've processed
    processed_count = 0
    skipped_count = 0

    # Iterate over all annotations
    for idx in tqdm(range(len(dataset)), desc="Generating masks"):
        sample = dataset[idx]
        
        # Get image path and load as PIL
        image_path = dataset.dataset_dir_path / dataset.dataset[idx][0]
        image = Image.open(image_path).convert("RGB")
        
        # Extract visible keypoints as point prompts
        keypoints = sample["original_keypoints"]
        visible_points = extract_visible_keypoints(keypoints)
        
        if len(visible_points) == 0:
            print(f"Warning: No visible keypoints for image {idx}, skipping")
            skipped_count += 1
            continue
        
        # Format points for SAM3: [[[[x1, y1], [x2, y2], ...]]]
        # 4 dimensions: (batch, object, points_per_object, coordinates)
        input_points = [[visible_points]]
        # All points are positive (foreground)
        input_labels = [[[1] * len(visible_points)]]
        
        # Process with SAM3
        inputs = processor(
            images=image, 
            input_points=input_points, 
            input_labels=input_labels, 
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            # multimask_output=False returns single best mask per object
            outputs = model(**inputs, multimask_output=False)
        
        # Post-process masks - returns list of masks per image
        masks = processor.post_process_masks(
            outputs.pred_masks.cpu(), 
            inputs["original_sizes"]
        )[0]
        
        # Shape is (num_objects, num_masks, H, W), take first object's first mask
        best_mask = masks[0, 0].numpy().astype(np.uint8)
        
        # Use BinarySegmentationMask for conversion to COCO format
        seg_mask = BinarySegmentationMask(best_mask)
        
        # Find the corresponding annotation in the original COCO data
        # The dataset stores (image_path, keypoints, bbox, image_id, category_id)
        image_id = dataset.dataset[idx][3]
        
        # Find annotation for this image
        for ann in coco_data["annotations"]:
            if ann["image_id"] == image_id:
                # Add segmentation in RLE format, area, and bbox using BinarySegmentationMask
                ann["segmentation"] = seg_mask.as_compressed_rle
                ann["area"] = seg_mask.area
                ann["bbox"] = list(seg_mask.bbox)
                processed_count += 1
                break
    
    # Save the updated COCO dataset
    with open(output_path, "w") as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"\nDone!")
    print(f"Processed: {processed_count} annotations")
    print(f"Skipped (no keypoints): {skipped_count} annotations")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    from few_shot_keypoints.paths import DSD_SHOE_TRAIN_JSON, DSD_SHOE_TEST_JSON
    config = Config(
        dataset=DSD_SHOE_TEST_JSON,
        output=DSD_SHOE_TEST_JSON.with_suffix(".with_masks.json"),
        device="cuda",
    )
    generate_sam_masks(config)
