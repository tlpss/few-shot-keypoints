"""Create a reference keypoint config (cf. few_shot_keypoints.config), which can be used to evaluate a specific
set of reference keypoints with scripts/match_dataset.py (`--reference_config_dir`).

This is the counterpart of `create_config.ipynb` in the KIL repo, which picks a frame from a set of demonstration
videos, detects the object (SAM3) and its mask (MobileSAM), and then clicks the keypoints on that frame.
Here the reference keypoints come either from the ground truth annotations of a COCO keypoints dataset
(mode "from-coco"), or from clicking them on an image (mode "annotate"), which is the KIL workflow.

The config format is identical to the KIL one, so the resulting config directory can be used on the robot as well
(and vice versa: a config created in the KIL repo can be evaluated here).

All keypoints of an object share a single reference frame, which has to contain all keypoint channels of the
dataset, in the same order.

Examples:
    # take the keypoints of a single (fully annotated) dataset image as reference keypoints
    python scripts/create_config.py --config_dir data/keypoint-configs/1mug-v2 --mode from-coco \
        --dataset_path data/kil/1mug-v2-initial-frames/annotations.with_masks.json

    # or click them yourself on an image, and segment the object with SAM3 using the clicks as point prompts
    python scripts/create_config.py --config_dir data/keypoint-configs/1mug-clicked --mode annotate \
        --image_path <image.png> --object_name mug --sam_mask True \
        --keypoint_names '[handle-top, handle-middle, handle-bottom, rim-front, rim-left, center-left, bottom-left]'
"""

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import draccus
import numpy as np
import torch
from skimage import io

from few_shot_keypoints.config import (
    KeypointRepresentationConfig,
    ObjectKeypointRepresentationConfig,
    ObjectsRepresentationConfig,
)
from few_shot_keypoints.datasets.coco_dataset import TorchCOCOKeypointsDataset

MODES = ("from-coco", "annotate")


@dataclass
class Config:
    config_dir: str = "keypoint-config"  # directory to write the config to
    mode: str = "from-coco"  # cf. MODES
    object_name: Optional[str] = None  # defaults to the COCO category name in "from-coco" mode
    keypoint_names: Optional[List[str]] = None  # taken from the dataset in "from-coco" mode
    top_k: int = 1  # number of matches to extract for each keypoint at inference time
    append: bool = False  # add the object to an existing config in config_dir, instead of creating a new config
    overwrite_object: bool = False  # replace the object if it is already in the config

    # mode "from-coco"
    dataset_path: Optional[str] = None
    image_index: Optional[int] = None  # dataset index of the reference frame, defaults to the first fully annotated one
    coco_image_id: Optional[int] = None  # alternative to image_index

    # mode "annotate"
    image_path: Optional[str] = None
    mask_path: Optional[str] = None  # optional binary mask of the object in the reference image
    sam_mask: bool = False  # segment the object with SAM3, using the annotated keypoints as point prompts
    device: str = "cuda:0"

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"invalid mode {self.mode}, should be one of {MODES}")
        if self.mode == "from-coco" and self.dataset_path is None:
            raise ValueError("mode 'from-coco' requires a dataset_path")
        if self.mode == "annotate":
            if self.image_path is None:
                raise ValueError("mode 'annotate' requires an image_path")
            if self.object_name is None:
                raise ValueError("mode 'annotate' requires an object_name")


def bbox_from_mask(mask: np.ndarray) -> Optional[List[float]]:
    """(x1,y1,x2,y2) bbox around all foreground pixels of a binary mask."""
    rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
    if not rows.any():
        return None
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return [float(x1), float(y1), float(x2 + 1), float(y2 + 1)]


def select_dataset_index(dataset: TorchCOCOKeypointsDataset, config: Config) -> int:
    """Determine which dataset image to use as reference frame.

    All keypoint channels have to be annotated on that single frame, as the object mask and bbox of the config are
    only valid for one frame.
    """
    n_channels = len(dataset.keypoint_channel_configuration)

    if config.coco_image_id is not None:
        indices = [i for i in range(len(dataset)) if dataset.dataset[i][3] == config.coco_image_id]
        if not indices:
            raise ValueError(f"no image with coco_image_id {config.coco_image_id} in {config.dataset_path}")
        return indices[0]

    if config.image_index is not None:
        return config.image_index

    # take the first image that has all keypoint channels annotated.
    best_index, best_n_visible = None, -1
    for i in range(len(dataset)):
        n_visible = sum(len(kps) > 0 for kps in dataset[i]["keypoints"])
        if n_visible == n_channels:
            return i
        if n_visible > best_n_visible:
            best_index, best_n_visible = i, n_visible

    raise ValueError(
        f"no image in {config.dataset_path} has all {n_channels} keypoint channels annotated "
        f"(best is index {best_index} with {best_n_visible} channels). Annotate the reference keypoints on a single "
        f"frame with mode 'annotate' instead, or pass an explicit image_index."
    )


def create_object_config_from_coco(config: Config) -> ObjectKeypointRepresentationConfig:
    """Create the config of a single object from the ground truth annotations of one dataset image."""
    dataset = TorchCOCOKeypointsDataset(config.dataset_path, transform=None)
    keypoint_names = config.keypoint_names or dataset.keypoint_channel_configuration
    object_name = config.object_name or dataset.parsed_coco.categories[0].name

    index = select_dataset_index(dataset, config)
    datapoint = dataset[index]
    print(f"using dataset index {index} (coco image id {datapoint['coco_image_id']}) as reference frame")

    keypoints = datapoint["keypoints"]
    missing = [name for name, kps in zip(keypoint_names, keypoints) if len(kps) == 0]
    if missing:
        raise ValueError(f"keypoint channels {missing} are not annotated on dataset index {index}")

    image = (datapoint["image"].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)  # CHW float -> HWC uint8

    mask = datapoint["mask"].astype(np.uint8)
    if mask.all():
        # the dataset has no segmentation for this annotation, cf. TorchCOCOKeypointsDataset.
        print("WARNING: no segmentation mask in the dataset for this image, the config will not contain a mask.")
        mask = None

    x, y, w, h = datapoint["bbox"]
    bbox = [float(x), float(y), float(x + w), float(y + h)]

    return build_object_config(
        object_name=object_name,
        image=image,
        # the annotations can hold multiple keypoints per channel, take the first one as reference.
        keypoints=[(float(kps[0][0]), float(kps[0][1])) for kps in keypoints],
        keypoint_names=list(keypoint_names),
        top_k=config.top_k,
        mask=mask,
        bbox=bbox,
    )


def annotate_keypoints_on_image(
    image: np.ndarray, keypoint_names: Optional[List[str]] = None
) -> List[Tuple[float, float]]:
    """Click the keypoints on the image, in the order of `keypoint_names` (or in the keypoint channel order of the
    dataset that the config will be evaluated on). Close the window when done."""
    import matplotlib.pyplot as plt

    colors = ["red", "lime", "blue", "yellow", "magenta", "orange", "brown", "pink", "gray", "black"]
    keypoints: List[Tuple[float, float]] = []
    n_expected = len(keypoint_names) if keypoint_names is not None else None

    def next_title() -> str:
        if n_expected is not None and len(keypoints) >= n_expected:
            return f"all {n_expected} keypoints annotated, close the window to continue"
        if keypoint_names is not None:
            return f"click keypoint {len(keypoints)}: {keypoint_names[len(keypoints)]}"
        return f"click keypoint {len(keypoints)} (close the window when done)"

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image)
    ax.set_title(next_title())

    def onclick(event):
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        if n_expected is not None and len(keypoints) >= n_expected:
            print(f"already annotated {n_expected} keypoints, ignoring click")
            return
        u, v = float(event.xdata), float(event.ydata)
        name = keypoint_names[len(keypoints)] if keypoint_names is not None else str(len(keypoints))
        print(f"keypoint {len(keypoints)} ({name}) at ({u:.1f}, {v:.1f})")
        ax.scatter(u, v, c=colors[len(keypoints) % len(colors)], s=50)
        keypoints.append((u, v))
        ax.set_title(next_title())
        fig.canvas.draw()

    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()  # blocks until the window is closed

    if not keypoints:
        raise ValueError("no keypoints were annotated")
    if n_expected is not None and len(keypoints) != n_expected:
        raise ValueError(f"annotated {len(keypoints)} keypoints, expected {n_expected}: {keypoint_names}")
    return keypoints


def sam3_mask_from_points(image: np.ndarray, points: List[Tuple[float, float]], device: str) -> np.ndarray:
    """Segment the object with SAM3, using the annotated keypoints as (positive) point prompts.
    Same approach as scripts/generate_SAM_masks_for_coco_keypoints_dataset.py."""
    from PIL import Image
    from transformers import Sam3TrackerModel, Sam3TrackerProcessor

    print("loading SAM3...")
    model = Sam3TrackerModel.from_pretrained("facebook/sam3").to(device)
    processor = Sam3TrackerProcessor.from_pretrained("facebook/sam3")

    # 4 dimensions: (batch, object, points_per_object, coordinates)
    input_points = [[[[float(u), float(v)] for u, v in points]]]
    input_labels = [[[1] * len(points)]]
    inputs = processor(
        images=Image.fromarray(image), input_points=input_points, input_labels=input_labels, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)
    masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
    return masks[0, 0].numpy().astype(np.uint8)


def create_object_config_from_image(config: Config) -> ObjectKeypointRepresentationConfig:
    """Create the config of a single object by clicking its keypoints on an image."""
    image = io.imread(config.image_path)
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[..., :3]

    keypoints = annotate_keypoints_on_image(image, config.keypoint_names)

    mask = None
    if config.mask_path is not None:
        mask = io.imread(config.mask_path)
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = (mask > (127 if mask.max() > 1 else 0)).astype(np.uint8)
    elif config.sam_mask:
        mask = sam3_mask_from_points(image, keypoints, config.device)

    bbox = bbox_from_mask(mask) if mask is not None else None
    if mask is None:
        print(
            "WARNING: the config will not contain a mask or bbox, so it can only be used for matching on the entire "
            "image (pass a mask_path or --sam_mask True to also support the crop/mask matching pipelines)."
        )

    return build_object_config(
        object_name=config.object_name,
        image=image,
        keypoints=keypoints,
        keypoint_names=config.keypoint_names,
        top_k=config.top_k,
        mask=mask,
        bbox=bbox,
    )


def build_object_config(
    object_name: str,
    image: np.ndarray,
    keypoints: List[Tuple[float, float]],
    keypoint_names: Optional[List[str]],
    top_k: int,
    mask: Optional[np.ndarray] = None,
    bbox: Optional[List[float]] = None,
) -> ObjectKeypointRepresentationConfig:
    keypoint_configs = [
        KeypointRepresentationConfig(reference_img=image, keypoint=(u, v), keypoint_top_ks=top_k)
        for u, v in keypoints
    ]
    return ObjectKeypointRepresentationConfig(
        object_name=object_name,
        reference_img=image,
        object_bbox=bbox,
        object_mask=mask,
        keypoint_config=keypoint_configs,
        keypoint_names=list(keypoint_names) if keypoint_names is not None else None,
    )


@draccus.wrap()
def create_config(config: Config):
    if config.mode == "from-coco":
        object_config = create_object_config_from_coco(config)
    else:
        object_config = create_object_config_from_image(config)

    # load the existing config to add this object to, if requested.
    if config.append and os.path.exists(os.path.join(config.config_dir, "config.json")):
        objects_config = ObjectsRepresentationConfig.load_from_dir(config.config_dir)
    else:
        objects_config = ObjectsRepresentationConfig(objects=[])

    existing_names = [obj.object_name for obj in objects_config.objects]
    if object_config.object_name in existing_names:
        if not config.overwrite_object:
            raise ValueError(
                f"object {object_config.object_name} is already in the config in {config.config_dir}, "
                f"pass --overwrite_object True to replace it."
            )
        objects_config.objects = [obj for obj in objects_config.objects if obj.object_name != object_config.object_name]
    objects_config.objects.append(object_config)

    os.makedirs(config.config_dir, exist_ok=True)
    objects_config.save_to_dir(config.config_dir)

    print(f"\nsaved config with objects {[obj.object_name for obj in objects_config.objects]} to {config.config_dir}")
    print(f"object {object_config.object_name}:")
    print(f"  keypoint names: {object_config.keypoint_names}")
    print(f"  keypoints (u,v): {[kp.keypoint for kp in object_config.keypoint_config]}")
    print(f"  top_k: {[kp.keypoint_top_ks for kp in object_config.keypoint_config]}")
    print(f"  bbox (x1,y1,x2,y2): {object_config.object_bbox}")
    print(f"  mask: {'yes' if object_config.object_mask is not None else 'no'}")
    print(
        "\nthe keypoints have to be in the same order as the keypoint channels of the dataset this config is "
        "evaluated on, cf. few_shot_keypoints.config_references.assert_config_matches_dataset"
    )


if __name__ == "__main__":
    create_config()
