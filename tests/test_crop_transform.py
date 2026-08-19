"""Regression tests for the crop/pad/resize geometry of CropTransform.

Plain asserts, so this runs both under pytest and as `python tests/test_crop_transform.py`.
"""

import cv2
import numpy as np

from few_shot_keypoints.dataset_object_crop_matching import IMAGE_PADDING_BORDER_MODE, CropTransform

TARGET = (256, 256)


def test_point_roundtrip_through_crop_space():
    transform = CropTransform.from_bbox((480, 640, 3), [100, 50, 200, 120], TARGET, margin_scale=0.1)
    for u, v in [(100.0, 50.0), (300.0, 170.0), (200.0, 110.0)]:
        u_crop, v_crop = transform.to_crop_point(u, v)
        u_back, v_back = transform.to_original_point(u_crop, v_crop)
        assert abs(u_back - u) < 1e-6 and abs(v_back - v) < 1e-6, f"({u},{v}) -> ({u_back},{v_back})"


def test_bbox_corners_land_inside_the_crop():
    """A keypoint anywhere in the bbox must map into the target frame, for both bbox aspect ratios."""
    for bbox in ([100, 50, 200, 120], [100, 50, 120, 200]):
        transform = CropTransform.from_bbox((480, 640, 3), bbox, TARGET, margin_scale=0.1)
        x, y, w, h = bbox
        for u, v in [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]:
            u_crop, v_crop = transform.to_crop_point(u, v)
            assert -1 <= u_crop <= TARGET[1] + 1, f"bbox {bbox}: u {u} -> {u_crop} outside crop"
            assert -1 <= v_crop <= TARGET[0] + 1, f"bbox {bbox}: v {v} -> {v_crop} outside crop"


def test_bbox_at_the_image_border_is_clipped_not_wrapped():
    transform = CropTransform.from_bbox((480, 640, 3), [0, 0, 60, 40], TARGET, margin_scale=0.2)
    x, y, w, h = transform.crop_bbox
    assert x >= 0 and y >= 0 and w > 0 and h > 0, f"degenerate crop bbox {transform.crop_bbox}"

    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    assert transform.apply_to_image(image).shape == (*TARGET, 3)


def test_image_padding_matches_the_declared_border_mode():
    """`apply_to_image` used to pad with black while documenting BORDER_REPLICATE. Whichever mode is configured,
    the pixels it produces must actually match it - that mismatch is a silent accuracy knob otherwise."""
    image = np.full((480, 640, 3), 200, dtype=np.uint8)
    # a wide bbox needs top/bottom padding to reach the square target
    transform = CropTransform.from_bbox(image.shape, [100, 200, 300, 100], TARGET, margin_scale=0.0)
    pad_t, pad_b, _, _ = transform.padding
    assert pad_t > 0 and pad_b > 0, "expected top/bottom padding for a wide bbox"

    processed = transform.apply_to_image(image)
    if IMAGE_PADDING_BORDER_MODE == cv2.BORDER_REPLICATE:
        assert processed.min() == 200, f"replicate mode produced pixels of value {processed.min()}"
    elif IMAGE_PADDING_BORDER_MODE == cv2.BORDER_CONSTANT:
        assert processed.min() == 0, "constant mode did not produce black padding"
        assert (processed[TARGET[0] // 2, TARGET[1] // 2] == 200).all(), "the crop itself was altered"
    else:
        raise AssertionError(f"untested border mode {IMAGE_PADDING_BORDER_MODE}")


def test_mask_padding_stays_zero_and_binary():
    mask = np.ones((480, 640), dtype=np.uint8)
    transform = CropTransform.from_bbox((480, 640, 3), [100, 200, 300, 100], TARGET, margin_scale=0.0)

    processed = transform.apply_to_mask(mask)
    assert processed.shape == TARGET
    assert set(np.unique(processed)).issubset({0, 1}), "mask is no longer binary after crop/pad/resize"
    assert processed[0, 0] == 0, "mask padding should be background"


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
