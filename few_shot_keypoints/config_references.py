"""Create matcher reference vectors from a reference keypoint config (cf. few_shot_keypoints.config).

This is the evaluation counterpart of what the KIL representation extractor does in
`BaseKeypointMatcher.__init__`: for each configured keypoint, extract the features of its (annotated)
reference image and take the feature vector at the annotated keypoint location.

Use this instead of the `populate_matcher_w_random_references` functions of the matching pipelines to
evaluate a specific set of reference keypoints (e.g. the config that is used on the robot) instead of
reference keypoints that are sampled from the ground truth annotations of a random dataset image.

The reference images are preprocessed in the same way as the images at inference time (dataset transform,
background masking, bbox cropping), so that the reference vectors and the image features are comparable.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch

from few_shot_keypoints.config import ObjectKeypointRepresentationConfig
from few_shot_keypoints.dataset_mask_matching import DEFAULT_PIXEL_MASK_DILATION_ITERATIONS, apply_mask_to_image
from few_shot_keypoints.dataset_object_crop_matching import CropTransform
from few_shot_keypoints.datasets.augmentations import MultiChannelKeypointsCompose
from few_shot_keypoints.datasets.coco_dataset import TorchCOCOKeypointsDataset
from few_shot_keypoints.featurizers.base import BaseFeaturizer


def _to_float_image(image: np.ndarray) -> np.ndarray:
    """convert a reference image to a (H,W,3) float image in range [0,1], as expected by the featurizers."""
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"expected a (H,W,3) RGB reference image, got shape {image.shape}")
    image = image[..., :3]  # drop alpha channel if present
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0
    return image


def extract_reference_vectors_from_config(
    object_config: ObjectKeypointRepresentationConfig,
    feature_extractor: BaseFeaturizer,
    transform: Optional[MultiChannelKeypointsCompose] = None,
    crop: bool = False,
    mask_pixels: bool = False,
    crop_target_size: Tuple[int, int] = (256, 256),
    margin_scale: float = 0.1,
    pixel_mask_dilation_iterations: int = DEFAULT_PIXEL_MASK_DILATION_ITERATIONS,
) -> Tuple[torch.Tensor, List[int]]:
    """Extract a reference feature vector for each keypoint of the object config.

    Args:
        object_config: the config of a single object, cf. `ObjectsRepresentationConfig.get_object`.
        feature_extractor: the featurizer to extract the reference vectors with.
        transform: the geometric transform that is applied to the dataset images (e.g. RESIZE_TRANSFORM).
        crop: crop the reference image to the object bbox of the config, as in dataset_object_crop_matching.py.
        mask_pixels: black out the background of the reference image using the object mask of the config,
            as in dataset_mask_matching.py with mask_mode="pixels".
        crop_target_size: (height, width) of the crop, only used if crop is True.
        margin_scale: bbox margin of the crop, only used if crop is True.
        pixel_mask_dilation_iterations: mask dilation, only used if mask_pixels is True.

    Returns:
        (reference_vectors, top_k_matches): a (N,D) tensor with a reference vector for each of the N configured
        keypoints, and the number of matches to extract for each of them, both ordered as the config.
    """
    if object_config.keypoint_config is None or len(object_config.keypoint_config) == 0:
        raise ValueError(f"object {object_config.object_name} has no keypoints in its config")
    if crop and transform is not None:
        # the albumentations transforms do not transform the bbox (only image, keypoints and mask), so the
        # bbox of the config would no longer match the transformed reference image.
        raise ValueError("cropping the reference images is only supported without a dataset transform")
    if crop and object_config.object_bbox is None:
        raise ValueError(
            f"object {object_config.object_name} has no bbox in its config, which is needed to crop its reference images."
        )
    if mask_pixels and object_config.object_mask is None:
        raise ValueError(
            f"object {object_config.object_name} has no mask in its config, which is needed to mask its reference images."
        )

    reference_vectors = []
    top_k_matches = []

    for i, keypoint_config in enumerate(object_config.keypoint_config):
        image = keypoint_config.reference_img
        mask = object_config.object_mask
        u, v = float(keypoint_config.keypoint[0]), float(keypoint_config.keypoint[1])

        if mask is not None and mask.shape[:2] != image.shape[:2]:
            raise ValueError(
                f"the object mask {mask.shape[:2]} does not match the reference image of keypoint {i} "
                f"{image.shape[:2]}. The object mask is only valid for reference images of the same frame."
            )

        # 1. apply the same geometric transform as the one applied to the dataset images.
        if transform is not None:
            transformed = transform(image=image, keypoints=[[(u, v)]], mask=mask if mask is not None else np.ones(image.shape[:2], dtype=np.uint8))
            image, mask = transformed["image"], transformed["mask"]
            if len(transformed["keypoints"][0]) == 0:
                raise ValueError(f"keypoint {i} of object {object_config.object_name} is no longer visible after the transform")
            u, v = transformed["keypoints"][0][0]

        image = _to_float_image(image)

        # 2. black out the background.
        if mask_pixels:
            image = apply_mask_to_image(image, mask, dilation_iterations=pixel_mask_dilation_iterations)

        # 3. crop to the object bbox.
        if crop:
            crop_transform = CropTransform.from_bbox(
                image.shape, object_config.object_bbox_xywh, crop_target_size, margin_scale=margin_scale
            )
            image = crop_transform.apply_to_image(image)
            u, v = crop_transform.to_crop_point(u, v)

        # 4. extract the feature vector at the keypoint.
        u, v = int(u), int(v)
        if not (0 <= u < image.shape[1] and 0 <= v < image.shape[0]):
            raise ValueError(
                f"keypoint {i} of object {object_config.object_name} is at ({u},{v}), which is outside of its "
                f"preprocessed reference image {image.shape[:2]}"
            )
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)  # HWC -> 1,C,H,W
        features = feature_extractor.extract_features(image_tensor)
        reference_vectors.append(features[0, :, v, u].clone())
        top_k_matches.append(int(keypoint_config.keypoint_top_ks))

    print(f"extracted {len(reference_vectors)} reference vectors from the config of object {object_config.object_name}")
    return torch.stack(reference_vectors), top_k_matches


def assert_config_matches_dataset(
    object_config: ObjectKeypointRepresentationConfig, coco_dataset: TorchCOCOKeypointsDataset
) -> None:
    """Check that the keypoints of the config match the keypoint channels of the dataset.

    The matching results are stored per keypoint channel of the dataset, so the keypoints of the config have to
    be in the same order as the keypoint channels of the dataset. The order can only be verified if the config
    stores the keypoint names (configs created in the KIL repo do not).
    """
    dataset_keypoint_names = coco_dataset.keypoint_channel_configuration
    n_config_keypoints = len(object_config.keypoint_config or [])
    if n_config_keypoints != len(dataset_keypoint_names):
        raise ValueError(
            f"the config of object {object_config.object_name} has {n_config_keypoints} keypoints, but the dataset "
            f"has {len(dataset_keypoint_names)} keypoint channels: {dataset_keypoint_names}"
        )
    if object_config.keypoint_names is None:
        print(
            f"WARNING: the config of object {object_config.object_name} does not contain keypoint names, so the "
            f"keypoints are assumed to be in the same order as the dataset channels: {dataset_keypoint_names}"
        )
        return
    if list(object_config.keypoint_names) != list(dataset_keypoint_names):
        raise ValueError(
            f"the keypoints of the config {object_config.keypoint_names} do not match the keypoint channels of the "
            f"dataset {dataset_keypoint_names}"
        )
