"""
dataset matching using object crops.

contains duplicate code from the dataset_matching.py.. should refactor them to have cropping as optional transform.
"""
import random
from typing import Callable, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import torch
import numpy as np
import cv2
from skimage import io

from few_shot_keypoints.datasets.coco_dataset import TorchCOCOKeypointsDataset
from few_shot_keypoints.datasets.data_parsers import CocoKeypointsResultAnnotation, CocoKeypointsResultDataset
from few_shot_keypoints.matcher import KeypointFeatureMatcher
from few_shot_keypoints.featurizers.base import BaseFeaturizer

# --- Refactored Geometry Logic ---

@dataclass
class CropTransform:
    """
    Encapsulates the geometry logic for cropping, padding, and resizing 
    regions of interest, ensuring coordinate transforms remain consistent.
    """
    crop_bbox: Tuple[int, int, int, int]  # (x, y, w, h) in original image
    padding: Tuple[int, int, int, int]    # (top, bottom, left, right) added to crop
    target_size: Tuple[int, int]          # (height, width) of final result
    scale: Tuple[float, float]            # (scale_x, scale_y) mapping padded crop -> target

    @classmethod
    def from_bbox(
        cls, 
        image_shape: Tuple[int, ...], 
        bbox: List[float], 
        target_size: Tuple[int, int], 
        margin_scale: float = 0.0
    ):
        """
        Factory: Calculates crop coordinates with MARGIN, determines necessary padding 
        to fit the result into target_size without distortion.
        
        Args:
            image_shape: (H, W, C) of original image
            bbox: [x, y, w, h]
            target_size: (height, width)
            margin_scale: Fraction of bbox size to add as context (e.g. 0.2 = 20% wider/taller)
        """
        img_h, img_w = image_shape[:2]
        tgt_h, tgt_w = target_size
        
        # 1. Parse Bbox
        x, y, w, h = map(float, bbox)
        
        # 2. Apply Margin
        if margin_scale > 0:
            x_margin = w * margin_scale
            y_margin = h * margin_scale
            
            x -= x_margin / 2
            y -= y_margin / 2
            w += x_margin
            h += y_margin
            
        # 3. Clip to Image Boundaries (Integer conversion happens here)
        x, y, w, h = int(x), int(y), int(w), int(h)
        
        # Handle left/top edges going negative
        if x < 0:
            w += x # shrink width
            x = 0
        if y < 0:
            h += y # shrink height
            y = 0
            
        # Handle right/bottom edges going out of bounds
        w = min(w, img_w - x)
        h = min(h, img_h - y)
        
        if w <= 0 or h <= 0:
            # Fallback for degenerate bboxes: just return a valid 1x1 crop to avoid crashing
            # or raise specific error depending on preference
            raise ValueError(f"Invalid crop dimensions after margin/clipping: {w}x{h}")

        # 4. Calculate Padding to match Target Aspect Ratio
        target_aspect = tgt_w / tgt_h
        current_aspect = w / h
        
        pad_t, pad_b, pad_l, pad_r = 0, 0, 0, 0
        
        if current_aspect > target_aspect: 
            # Image is wider than target: Pad height (top/bottom)
            new_h = int(w / target_aspect)
            diff = new_h - h
            pad_t = diff // 2
            pad_b = diff - pad_t
        else: 
            # Image is taller than target: Pad width (left/right)
            new_w = int(h * target_aspect)
            diff = new_w - w
            pad_l = diff // 2
            pad_r = diff - pad_l

        # 5. Calculate Scale based on the PADDED dimensions
        padded_w = w + pad_l + pad_r
        padded_h = h + pad_t + pad_b
        
        scale_x = tgt_w / padded_w
        scale_y = tgt_h / padded_h

        return cls(
            crop_bbox=(x, y, w, h),
            padding=(pad_t, pad_b, pad_l, pad_r),
            target_size=(tgt_h, tgt_w),
            scale=(scale_x, scale_y)
        )

    def apply_to_image(self, image: np.ndarray) -> np.ndarray:
        """
        Applies the crop -> pad -> resize pipeline.
        Uses BORDER_REPLICATE (last pixel) for padding.
        """
        x, y, w, h = self.crop_bbox
        pt, pb, pl, pr = self.padding
        th, tw = self.target_size

        # 1. Crop
        crop = image[y:y+h, x:x+w]
        
        # 2. Pad (Replicate last pixel instead of black)
        padded = cv2.copyMakeBorder(crop, pt, pb, pl, pr, cv2.BORDER_CONSTANT)
        
        # 3. Resize
        return cv2.resize(padded, (tw, th), interpolation=cv2.INTER_CUBIC)

    def to_original_point(self, u_crop: float, v_crop: float) -> Tuple[float, float]:
        """
        Transform point from Crop Space (Target Size) -> Original Image Space.
        """
        pt, _, pl, _ = self.padding
        cx, cy, _, _ = self.crop_bbox
        sx, sy = self.scale

        # Operation: (Point / Scale) - Pad + CropOffset
        u_orig = (u_crop / sx) - pl + cx
        v_orig = (v_crop / sy) - pt + cy
        return u_orig, v_orig

    def to_crop_point(self, u_orig: float, v_orig: float) -> Tuple[float, float]:
        """
        Transform point from Original Image Space -> Crop Space (Target Size).
        """
        pt, _, pl, _ = self.padding
        cx, cy, _, _ = self.crop_bbox
        sx, sy = self.scale

        # Operation: (Point - CropOffset + Pad) * Scale
        u_crop = (u_orig - cx + pl) * sx
        v_crop = (v_orig - cy + pt) * sy
        return u_crop, v_crop


# --- Main Logic ---

def run_coco_dataset_inference(
    coco_dataset: TorchCOCOKeypointsDataset,
    keypoint_matcher: KeypointFeatureMatcher,
    feature_extractor: BaseFeaturizer,
    crop_target_size: Tuple[int, int] = (256, 256),
    margin_scale: float = 0.1,  # Default margin 10%
) -> CocoKeypointsResultDataset:
    
    coco_results_annotations = []

    for i in tqdm(range(len(coco_dataset))):
        datapoint = coco_dataset[i]
        
        # 1. Load Data
        image_path = coco_dataset.dataset_dir_path / coco_dataset.dataset[i][0]
        original_image = io.imread(image_path)
        if original_image.shape[2] == 4:
            original_image = original_image[..., :3]
        
        # 2. Initialize Transform with Margin
        try:
            transform = CropTransform.from_bbox(
                original_image.shape, 
                datapoint["original_bbox"], 
                crop_target_size,
                margin_scale=margin_scale
            )
        except ValueError as e:
            print(f"Skipping image {i} due to error: {e}")
            continue
            
        # 3. Process Image
        processed_image = transform.apply_to_image(original_image)

        # cv2.imwrite(f"cropped_image.png", processed_image)
        # input("Press Enter to continue...")
        
        # 4. Extract Features
        # HWC -> CHW, Normalize
        image_tensor = torch.from_numpy(processed_image).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0)
        
        features = feature_extractor.extract_features(image_tensor)
        results = keypoint_matcher.get_best_matches_from_image_features(features)

        # 5. Process Results & Transform coordinates back
        flattened_keypoints = []
        scores = []
        
        for result_list in results:
            if result_list and result_list[0].u is not None:
                assert len(result_list) == 1, "Expected only 1 match per category."
                res = result_list[0]
                
                # Transform back to original coordinates
                u_orig, v_orig = transform.to_original_point(res.u, res.v)
                
                flattened_keypoints.extend([u_orig, v_orig, 2]) # 2 = visible
                scores.append(res.score)
            else:
                flattened_keypoints.extend([0, 0, 0])

        coco_results_annotations.append(
            CocoKeypointsResultAnnotation(
                id=i,
                image_id=datapoint["coco_image_id"],
                category_id=datapoint["coco_category_id"],
                bbox=datapoint["original_bbox"],
                keypoints=flattened_keypoints,
                score=sum(scores) / len(scores) if scores else 0.0,
                keypoint_scores=scores,
            )
        )

    return CocoKeypointsResultDataset(coco_results_annotations)


def populate_matcher_w_random_references(
    coco_dataset: TorchCOCOKeypointsDataset,
    feature_extractor: BaseFeaturizer,
    crop_target_size: Tuple[int, int] = (256, 256),
    margin_scale: float = 0.1,  # Default margin 10%
    seed: int = 2025,
):
    """
    Populate each keypoint matcher with N random reference images.
    Uses the same CropTransform logic with margins.
    """
    reference_vectors = [None] * len(coco_dataset.parsed_coco.categories[0].keypoints)
    rng = random.Random(seed)
    
    while any(rv is None for rv in reference_vectors):
        idx = rng.randint(0, len(coco_dataset) - 1)
        
        # Load Data
        image_path = coco_dataset.dataset_dir_path / coco_dataset.dataset[idx][0]
        original_image = io.imread(image_path)
        if original_image.shape[2] == 4:
            original_image = original_image[..., :3]
        
        bbox = coco_dataset.dataset[idx][2]
        keypoints = coco_dataset.dataset[idx][1]
        
        # Initialize Transform with Margin
        try:
            transform = CropTransform.from_bbox(
                original_image.shape, 
                bbox, 
                crop_target_size,
                margin_scale=margin_scale
            )
        except ValueError:
            continue
            
        # Process Image & Features
        processed_image = transform.apply_to_image(original_image)
        image_tensor = torch.from_numpy(processed_image).permute(2, 0, 1).float() / 255.0
        features = feature_extractor.extract_features(image_tensor.unsqueeze(0))
        
        # Extract Vectors for known keypoints in this image
        for i, kp_list in enumerate(keypoints):
            if reference_vectors[i] is None and len(kp_list) > 0:
                u_orig, v_orig = kp_list[0]
                
                # Transform Ground Truth to Crop Space
                u_crop, v_crop = transform.to_crop_point(u_orig, v_orig)
                u_crop, v_crop = int(u_crop), int(v_crop)

                # Validate bounds
                if 0 <= u_crop < crop_target_size[1] and 0 <= v_crop < crop_target_size[0]:
                    print(f"Found reference for keypoint {i} in image {idx}")
                    reference_vectors[i] = features[0, :, v_crop, u_crop].clone()
        
    return torch.stack(reference_vectors)


if __name__ == "__main__":
    from few_shot_keypoints.featurizers.ViT_featurizer import DinoV3LargeFeaturizer
    from few_shot_keypoints.datasets.coco_dataset import TorchCOCOKeypointsDataset
    from airo_dataset_tools.data_parsers.coco import CocoKeypointsDataset as CocoParser
    from few_shot_keypoints.matcher import KeypointFeatureMatcher
    import json
    from few_shot_keypoints.paths import KIL_MUGS_V2_INITIAL_JSON

    coco_json_path = KIL_MUGS_V2_INITIAL_JSON

    with open(coco_json_path, "r") as f:
        coco_dataset_parser = CocoParser(**json.load(f))

    target_path = "test_crop_matching.json"
    
    # Initialize feature extractor
    feature_extractor = DinoV3LargeFeaturizer(device='cuda:0')
    
    # Load dataset WITHOUT transforms - we handle geometry manually
    crop_target_size = (512, 512)
    coco_dataset = TorchCOCOKeypointsDataset(json_dataset_path=coco_json_path, transform=None)

    # Configuration for margins
    MARGIN_SCALE = 0.2  # 20% margin around the bbox

    # Populate matcher
    reference_vectors = populate_matcher_w_random_references(
        coco_dataset, 
        feature_extractor, 
        crop_target_size=crop_target_size, 
        margin_scale=MARGIN_SCALE,
        seed=2029
    )
    matcher = KeypointFeatureMatcher(reference_vectors, device='cuda:0')

    # Run inference
    coco_results_dataset = run_coco_dataset_inference(
        coco_dataset, 
        matcher, 
        feature_extractor, 
        crop_target_size=crop_target_size,
        margin_scale=MARGIN_SCALE
    )
    
    with open(target_path, "w") as f:
        f.write(coco_results_dataset.model_dump_json(indent=4))
    
    print(f"Results saved to {target_path}")