"""Reference keypoint configurations.

Port of `representation-extractor/representation_extractor/config.py` from the
keypoint-imitation-learning (KIL) repo. The on-disk format is kept identical, so that config
directories can be moved between both repos:

    <config_dir>/config.json
    <config_dir>/img/obj_<i>_<object_name>.png       # object reference image
    <config_dir>/img/obj_<i>_<object_name>_mask.png  # object mask (optional)
    <config_dir>/img/obj_<i>_kp_<j>.png              # reference image of keypoint j

These configs hold the reference keypoints (as annotated on a reference image) that are used to
create the reference feature vectors for the matcher, cf. `few_shot_keypoints.config_references`.

Two deliberate deviations from the KIL version, both backwards compatible with KIL configs:
 - `keypoint_names` is stored (optional), so that the keypoint order of the config can be verified
    against the keypoint channel order of the dataset it is evaluated on. KIL ignores this field.
 - masks are binarized to {0,1} when loading. KIL writes binary masks as 0/255 pngs, but never reads
    them back, so it does not need to undo the scaling.
"""

from dataclasses import dataclass
import numpy as np
from typing import Optional, List, Tuple
import os
import json


@dataclass
class BaseRepresentationConfig:
    pass


@dataclass
class KeypointRepresentationConfig(BaseRepresentationConfig):
    reference_img: np.ndarray
    keypoint: Tuple[float]  # (u,v) float tuple
    keypoint_top_ks: int  # how many matches to extract


@dataclass
class ObjectKeypointRepresentationConfig(BaseRepresentationConfig):
    object_name: str
    reference_img: np.ndarray  # (H,W,3)
    object_bbox: Optional[List[float]] = None  # (4,): x1, y1, x2, y2
    object_mask: Optional[np.ndarray] = None  # (H,W) binary mask
    detector_confidence_threshold: Optional[float] = None  # confidence threshold for the detector
    keypoint_config: Optional[List[KeypointRepresentationConfig]] = None  # optional keypoint config for the object
    keypoint_names: Optional[List[str]] = None  # names of the keypoints, in the same order as keypoint_config

    def __post_init__(self):
        if self.keypoint_names is not None and self.keypoint_config is not None:
            assert len(self.keypoint_names) == len(
                self.keypoint_config
            ), f"got {len(self.keypoint_names)} keypoint names for {len(self.keypoint_config)} keypoints"

    @property
    def num_keypoints(self) -> int:
        if self.keypoint_config is None:
            return 0
        return sum([kp.keypoint_top_ks for kp in self.keypoint_config])

    @property
    def object_bbox_xywh(self) -> Optional[List[float]]:
        """the object bbox in the (x,y,w,h) COCO convention, as used by the dataset matching pipelines."""
        if self.object_bbox is None:
            return None
        x1, y1, x2, y2 = self.object_bbox
        return [x1, y1, x2 - x1, y2 - y1]


@dataclass
class ObjectsRepresentationConfig(BaseRepresentationConfig):
    objects: List[ObjectKeypointRepresentationConfig]

    def get_object(self, object_name: Optional[str] = None) -> ObjectKeypointRepresentationConfig:
        """Get the config of a single object.

        If no name is given, the config is expected to contain exactly one object.
        """
        if object_name is None:
            if len(self.objects) != 1:
                raise ValueError(
                    f"config contains {len(self.objects)} objects "
                    f"({[o.object_name for o in self.objects]}), specify which one to use."
                )
            return self.objects[0]
        for obj in self.objects:
            if obj.object_name == object_name:
                return obj
        raise ValueError(f"object {object_name} not in config, available objects: {[o.object_name for o in self.objects]}")

    def save_to_dir(self, parent_dir: str, json_filename: str = "config.json", image_format: str = "png"):
        """
        Save this config to a JSON file and write all reference images to disk
        in an 'img' subdirectory. The JSON will reference images by relative paths.

        Args:
            parent_dir: Parent directory where to save the config and images.
            json_filename: Name of the JSON file to create (default: "config.json").
            image_format: Image format/extension for saved images (e.g., "png").
        """
        json_path = os.path.join(parent_dir, json_filename)
        images_dir = os.path.join(parent_dir, "img")
        os.makedirs(images_dir, exist_ok=True)

        def _imwrite(path: str, array: np.ndarray):
            # Write HxWxC uint8 image
            img = array

            # Validate image format: either (H,W,3) RGB or (H,W) grayscale
            if img.ndim == 2:
                # Grayscale image (H, W)
                pass
            elif img.ndim == 3 and img.shape[-1] == 3:
                # RGB image (H, W, 3)
                pass
            else:
                raise ValueError(f"Image must be either (H,W) grayscale or (H,W,3) RGB, got shape {img.shape}")

            if img.dtype != np.uint8:
                # Handle both 0-255 and 0-1 ranges
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = np.clip(img, 0, 255).astype(np.uint8)
            try:
                import imageio.v3 as iio  # type: ignore

                iio.imwrite(path, img)
            except Exception:
                try:
                    from PIL import Image  # type: ignore

                    Image.fromarray(img).save(path)
                except Exception as e:
                    raise RuntimeError(f"Failed to save image at {path}: {e}")

        serializable = {"objects": []}
        for obj_idx, obj in enumerate(self.objects):
            obj_entry = {
                "object_name": obj.object_name,
                "object_bbox": obj.object_bbox,
                "object_mask": None,  # Will be set below if mask exists
                "keypoint_names": obj.keypoint_names,
            }

            # Save object's reference image
            obj_img_filename = f"obj_{obj_idx}_{obj.object_name}." + image_format
            obj_img_rel_path = os.path.join("img", obj_img_filename)
            obj_img_abs_path = os.path.join(images_dir, obj_img_filename)
            _imwrite(obj_img_abs_path, obj.reference_img)
            obj_entry["reference_img_path"] = obj_img_rel_path

            # Save object's mask if it exists
            if obj.object_mask is not None:
                obj_mask_filename = f"obj_{obj_idx}_{obj.object_name}_mask." + image_format
                obj_mask_rel_path = os.path.join("img", obj_mask_filename)
                obj_mask_abs_path = os.path.join(images_dir, obj_mask_filename)
                _imwrite(obj_mask_abs_path, obj.object_mask)
                obj_entry["object_mask_path"] = obj_mask_rel_path

            # Keypoints
            if obj.keypoint_config is not None:
                kp_entries = []
                for kp_idx, kp in enumerate(obj.keypoint_config):
                    kp_img_filename = f"obj_{obj_idx}_kp_{kp_idx}." + image_format
                    kp_img_rel_path = os.path.join("img", kp_img_filename)
                    kp_img_abs_path = os.path.join(images_dir, kp_img_filename)
                    _imwrite(kp_img_abs_path, kp.reference_img)
                    kp_entries.append(
                        {
                            "keypoint": list(kp.keypoint),
                            "keypoint_top_ks": kp.keypoint_top_ks,
                            "reference_img_path": kp_img_rel_path,
                        }
                    )
                obj_entry["keypoint_config"] = kp_entries
            else:
                obj_entry["keypoint_config"] = None

            serializable["objects"].append(obj_entry)

        with open(json_path, "w") as f:
            json.dump(serializable, f, indent=2)

    @classmethod
    def load_from_dir(cls, parent_dir: str, json_filename: str = "config.json"):
        """
        Load config from a JSON file that references images stored in an 'img' subdirectory.
        The JSON file should be in the parent directory, with images in the 'img' subdirectory.

        Args:
            parent_dir: Parent directory containing the config JSON and img subdirectory.
            json_filename: Name of the JSON file to load (default: "config.json").
        """
        json_path = os.path.join(parent_dir, json_filename)
        json_dir = parent_dir

        def _imread(path: str) -> np.ndarray:
            try:
                import imageio.v3 as iio  # type: ignore

                img = iio.imread(path)
            except Exception:
                try:
                    from PIL import Image  # type: ignore

                    img = np.array(Image.open(path))
                except Exception as e:
                    raise RuntimeError(f"Failed to read image at {path}: {e}")
            # Don't convert grayscale to RGB - preserve original format
            return img

        with open(json_path, "r") as f:
            d = json.load(f)

        objects = []
        for obj in d.get("objects", []):
            obj_img_path = os.path.join(json_dir, obj["reference_img_path"])
            reference_img = _imread(obj_img_path)

            # Load object mask if it exists
            object_mask = None
            if "object_mask_path" in obj:
                obj_mask_path = os.path.join(json_dir, obj["object_mask_path"])
                object_mask = _imread(obj_mask_path)
                # masks are stored as 0/255 pngs, binarize them again.
                object_mask = (object_mask > 127).astype(np.uint8) if object_mask.max() > 1 else object_mask.astype(np.uint8)

            keypoint_cfg = None
            if obj.get("keypoint_config") is not None:
                keypoint_cfg = []
                for kp in obj["keypoint_config"]:
                    kp_img_path = os.path.join(json_dir, kp["reference_img_path"])
                    kp_img = _imread(kp_img_path)
                    keypoint_cfg.append(
                        KeypointRepresentationConfig(
                            reference_img=kp_img,
                            keypoint=tuple(kp["keypoint"]),
                            keypoint_top_ks=int(kp["keypoint_top_ks"]),
                        )
                    )

            objects.append(
                ObjectKeypointRepresentationConfig(
                    object_name=obj["object_name"],
                    reference_img=reference_img,
                    object_bbox=obj.get("object_bbox"),
                    object_mask=object_mask,
                    keypoint_config=keypoint_cfg,
                    keypoint_names=obj.get("keypoint_names"),
                )
            )

        return cls(objects=objects)


if __name__ == "__main__":
    # roundtrip test
    import shutil

    image = np.random.randint(0, 255, (64, 64, 3)).astype(np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:50, 10:50] = 1

    config = ObjectsRepresentationConfig(
        objects=[
            ObjectKeypointRepresentationConfig(
                object_name="mug",
                reference_img=image,
                object_mask=mask,
                object_bbox=[10, 10, 50, 50],
                keypoint_names=["handle-top", "rim-front"],
                keypoint_config=[
                    KeypointRepresentationConfig(reference_img=image, keypoint=(12, 13), keypoint_top_ks=1),
                    KeypointRepresentationConfig(reference_img=image, keypoint=(30, 31), keypoint_top_ks=3),
                ],
            )
        ]
    )

    config.save_to_dir("test_config")
    loaded_config = ObjectsRepresentationConfig.load_from_dir("test_config")
    loaded_object = loaded_config.get_object()
    assert loaded_object.object_name == "mug"
    assert loaded_object.keypoint_names == ["handle-top", "rim-front"]
    assert loaded_object.object_bbox == [10, 10, 50, 50]
    assert loaded_object.object_bbox_xywh == [10, 10, 40, 40]
    assert np.array_equal(loaded_object.object_mask, mask), "mask did not survive the roundtrip"
    assert np.array_equal(loaded_object.reference_img, image), "reference image did not survive the roundtrip"
    assert [kp.keypoint for kp in loaded_object.keypoint_config] == [(12, 13), (30, 31)]
    assert [kp.keypoint_top_ks for kp in loaded_object.keypoint_config] == [1, 3]
    assert loaded_object.num_keypoints == 4
    print(loaded_config.get_object())
    print("roundtrip OK")

    shutil.rmtree("test_config")
