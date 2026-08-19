"""
Dataset matching using object masks.

Uses the segmentation mask from the COCO annotation to limit the matching to the object, in one of two ways
(cf. the `mask_mode` argument):

- "pixels": black out the background of the image before passing it to the featurizer.
    Hypothesis: this can help to remove background influence, without altering the object apparent size etc..
    Note that this does alter the features of the object itself as well, as the patches on the object border
    now contain black pixels.

- "similarities": leave the image untouched and instead restrict the argmax of the cosine similarity maps to
    the object mask. This is what the KIL representation extractor does at inference time
    (representation_extractor/extractor.py), where the mask comes from a detector + SAM instead of from the
    annotations. The features are not altered, only the search region is.

- "both": black out the background *and* restrict the search region.

In all cases the image is cropped to the (masked) object bbox afterwards, cf. dataset_object_crop_matching.py.
"""
import random
from typing import Tuple

import numpy as np
import torch
from tqdm import tqdm

from few_shot_keypoints.datasets.coco_dataset import TorchCOCOKeypointsDataset
from few_shot_keypoints.datasets.data_parsers import CocoKeypointsResultAnnotation, CocoKeypointsResultDataset
from few_shot_keypoints.dataset_matching import matches_to_coco_keypoints
from few_shot_keypoints.dataset_object_crop_matching import CropTransform
from few_shot_keypoints.matcher import KeypointFeatureMatcher
from few_shot_keypoints.featurizers.base import BaseFeaturizer
import cv2

MASK_MODES = ("pixels", "similarities", "both")

# the mask is dilated to make sure the entire object is covered by the mask.
# more dilation is needed when masking the pixels, to avoid that patches on the object border contain black pixels.
# 2 iterations is what the KIL representation extractor uses to mask the similarity maps.
DEFAULT_PIXEL_MASK_DILATION_ITERATIONS = 4
DEFAULT_SIMILARITY_MASK_DILATION_ITERATIONS = 2


def dilate_mask(mask: np.ndarray, iterations: int = DEFAULT_SIMILARITY_MASK_DILATION_ITERATIONS) -> np.ndarray:
    """Dilate a binary (H,W) mask with a 5x5 kernel, to make sure the entire object is covered."""
    if iterations == 0:
        return mask
    return cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=iterations)


def apply_mask_to_image(
    image: np.ndarray, mask: np.ndarray, dilation_iterations: int = DEFAULT_PIXEL_MASK_DILATION_ITERATIONS
) -> np.ndarray:
    """
    Black out pixels where mask == 0.

    Args:
        image: (H, W, C) numpy array.
        mask: (H, W) binary mask, 1 = foreground, 0 = background.
        dilation_iterations: how many times to dilate the mask before applying it.

    Returns:
        Masked image with background set to black.
    """
    # apply dilation to the mask to ensure patches on the edge of the object do not contain black pixels.
    mask = dilate_mask(mask, dilation_iterations)
    return image * mask[:, :, np.newaxis]


def get_similarity_mask(
    mask: np.ndarray,
    transform: CropTransform,
    device: torch.device,
    dilation_iterations: int = DEFAULT_SIMILARITY_MASK_DILATION_ITERATIONS,
) -> torch.Tensor:
    """Bring the object mask into the frame of the (cropped) image, as a boolean tensor that can be passed
    to the KeypointFeatureMatcher to restrict the matches to the object."""
    cropped_mask = transform.apply_to_mask(dilate_mask(mask, dilation_iterations))
    return torch.from_numpy(cropped_mask.astype(bool)).to(device)



# --- Main Logic ---

def run_coco_dataset_inference(
    coco_dataset: TorchCOCOKeypointsDataset,
    keypoint_matcher: KeypointFeatureMatcher,
    feature_extractor: BaseFeaturizer,
    crop_target_size: Tuple[int, int] = (256, 256),
    margin_scale: float = 0.1,
    mask_mode: str = "pixels",
    pixel_mask_dilation_iterations: int = DEFAULT_PIXEL_MASK_DILATION_ITERATIONS,
    similarity_mask_dilation_iterations: int = DEFAULT_SIMILARITY_MASK_DILATION_ITERATIONS,
) -> CocoKeypointsResultDataset:
    """
    Run inference on a COCO dataset using the object mask (cf. module docstring for the `mask_mode` options),
    combined with bbox cropping.
    """
    assert mask_mode in MASK_MODES, f"invalid mask_mode {mask_mode}, should be one of {MASK_MODES}"
    mask_pixels = mask_mode in ("pixels", "both")
    mask_similarities = mask_mode in ("similarities", "both")

    coco_results_annotations = []

    for i in tqdm(range(len(coco_dataset))):
        datapoint = coco_dataset[i]

        # 1. Load Data
        image = datapoint["image"]
        bbox = datapoint["bbox"]
        mask = datapoint["mask"]

        image = image.permute(1, 2, 0).numpy()  # CHW -> HWC

        # 2. Apply segmentation mask (black out background)
        if mask_pixels:
            image = apply_mask_to_image(image, mask, dilation_iterations=pixel_mask_dilation_iterations)

        # 3. Initialize CropTransform with Margin
        try:
            transform = CropTransform.from_bbox(
                image.shape,
                bbox,
                crop_target_size,
                margin_scale=margin_scale,
            )
        except ValueError as e:
            print(f"Skipping image {i} due to error: {e}")
            continue

        # 4. Process Image (crop + pad + resize)
        processed_image = transform.apply_to_image(image)

        # 5. Bring the mask into the frame of the crop, to restrict the matches to the object.
        similarity_mask = None
        if mask_similarities:
            similarity_mask = get_similarity_mask(
                mask, transform, keypoint_matcher.device, similarity_mask_dilation_iterations
            )
            if not similarity_mask.any():
                print(f"Empty mask for image {i}, matching on the entire crop instead")
                similarity_mask = None

        # 6. Extract Features
        image_tensor = torch.from_numpy(processed_image).permute(2, 0, 1)  # HWC -> CHW
        image_tensor = image_tensor.unsqueeze(0)

        features = feature_extractor.extract_features(image_tensor)
        results = keypoint_matcher.get_best_matches_from_image_features(features, mask=similarity_mask)

        # 7. Process Results & Transform coordinates back
        flattened_keypoints, scores, topk_matches = matches_to_coco_keypoints(
            results, point_mapper=transform.to_original_point
        )

        coco_results_annotations.append(
            CocoKeypointsResultAnnotation(
                id=i,
                image_id=datapoint["coco_image_id"],
                category_id=datapoint["coco_category_id"],
                bbox=datapoint["original_bbox"],
                keypoints=flattened_keypoints,
                score=sum(scores) / len(scores) if scores else 0.0,
                keypoint_scores=scores,
                keypoint_topk_matches=topk_matches,
            )
        )

    return CocoKeypointsResultDataset(coco_results_annotations)


def populate_matcher_w_random_references(
    coco_dataset: TorchCOCOKeypointsDataset,
    feature_extractor: BaseFeaturizer,
    crop_target_size: Tuple[int, int] = (256, 256),
    margin_scale: float = 0.1,
    seed: int = 2025,
    mask_mode: str = "pixels",
    pixel_mask_dilation_iterations: int = DEFAULT_PIXEL_MASK_DILATION_ITERATIONS,
):
    """
    Populate each keypoint matcher with N random reference images.
    Uses mask-based background removal combined with CropTransform.

    The reference images are preprocessed in the same way as the images at inference time, so the pixels are only
    blacked out for the mask modes that do so. The similarity masking itself is not relevant here, as the
    reference vectors are taken at the (annotated) keypoints, which are on the object.
    """
    assert mask_mode in MASK_MODES, f"invalid mask_mode {mask_mode}, should be one of {MASK_MODES}"
    mask_pixels = mask_mode in ("pixels", "both")

    reference_vectors = [None] * len(coco_dataset.parsed_coco.categories[0].keypoints)
    rng = random.Random(seed)

    while any(rv is None for rv in reference_vectors):
        idx = rng.randint(0, len(coco_dataset) - 1)

        # Load Data
        print(f"sampling random image {idx}")
        datapoint = coco_dataset[idx]
        image = datapoint["image"]
        bbox = datapoint["bbox"]
        keypoints = datapoint["keypoints"]
        mask = datapoint["mask"]
        image = np.array(image.permute(1, 2, 0))  # CHW -> HWC
        print(keypoints)

        # Apply segmentation mask (black out background)
        if mask_pixels:
            image = apply_mask_to_image(image, mask, dilation_iterations=pixel_mask_dilation_iterations)
        # # convert to uint8
        # image = (image * 255).astype(np.uint8)
        # import cv2
        # cv2.imwrite(f"cropped_image.png", image)
        # input("Press Enter to continue...")

        # Initialize CropTransform with Margin
        try:
            transform = CropTransform.from_bbox(
                image.shape,
                bbox,
                crop_target_size,
                margin_scale=margin_scale,
            )
        except ValueError as e:
            print(f"Skipping image {idx} due to error: {e}")
            continue

        # Process Image & Features
        processed_image = transform.apply_to_image(image)
        image_tensor = torch.from_numpy(processed_image).permute(2, 0, 1)  # HWC -> CHW
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

    target_path = "test_mask_matching.json"

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
        seed=2029,
    )
    matcher = KeypointFeatureMatcher(reference_vectors, device='cuda:0')

    # Run inference
    coco_results_dataset = run_coco_dataset_inference(
        coco_dataset,
        matcher,
        feature_extractor,
        crop_target_size=crop_target_size,
        margin_scale=MARGIN_SCALE,
    )

    with open(target_path, "w") as f:
        f.write(coco_results_dataset.model_dump_json(indent=4))

    print(f"Results saved to {target_path}")
