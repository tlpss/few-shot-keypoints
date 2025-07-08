import argparse
import json
import math
import typing
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torchvision.transforms import ToTensor

from airo_dataset_tools.data_parsers.coco import CocoImage, CocoKeypointCategory, CocoKeypointsDataset as CocoKeypointsDatasetParser
from skimage import io
from few_shot_keypoints.datasets.augmentations import MultiChannelKeypointsCompose


class COCOKeypointsDataset:
    """Pytorch Dataset for COCO-formatted Keypoint dataset

    cf. https://cocodataset.org/#format-data for more information.
    
     We expect each json annotation to have a bbox,  the keypoints and num_keypoints fields. 
     Each category should also have keypoints. For more information on the required fields and data types, have a look at the COCO parser in `coco_parser.py`.
    The image paths in the JSON should be relative to the directory in which the JSON is located.


    This parser also adds some constraints to the dataset beyond what is allowed in the COCO format.
     - There can be only one category per dataset 
     - There can be only one annotation per image.
    so there can be at most one keypoint per image for each keypoint category.

    You can specify a keypoint channel configuration, which will be used to filter the keypoints.
    If no configuration is specified, all keypoints will be used.
    If a configuration is specified, only the keypoints in the configuration will be used.
  
    

    The dataset builds an index during the init call that maps from each image_id to a list of all keypoints of all semantic types in the dataset.
    """



    def __init__(
        self,
        json_dataset_path: str,
        detect_only_visible_keypoints: bool = True,
        keypoint_channel_configuration: Optional[List[str]] = None,
        transform: Optional[MultiChannelKeypointsCompose] = None,
        **kwargs,
    ):

        self.image_to_tensor_transform = ToTensor()
        self.dataset_json_path = Path(json_dataset_path)
        self.dataset_dir_path = self.dataset_json_path.parent  # assume paths in JSON are relative to this directory!

        self.keypoint_channel_configuration = keypoint_channel_configuration
        self.detect_only_visible_keypoints = detect_only_visible_keypoints

        print(f"{detect_only_visible_keypoints=}")

        self.transform = transform
        self.dataset = self.prepare_dataset()  # idx: (image, list(keypoints/channel))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Returns:
            dict with keys:
                "image": 3xHxW tensor
                "keypoints": List(c x list( list of K_i keypoints )), K_i is either 0 or 1
                "original_image_size": (H,W)
                "bbox": (x,y,w,h)

            e.g. for 2 heatmap channels with respectively 1, and 0 keypoints, the keypoints list will be formatted as
            [[[u11,v11]],[]] 
            and the original_image_size will be (H,W)
        """
        if torch.is_tensor(index):
            index = index.tolist()
        index = int(index)

        image_path = self.dataset_dir_path / self.dataset[index][0]
        image = io.imread(image_path)
        # remove a-channel if needed
        if image.shape[2] == 4:
            image = image[..., :3]

        orig_keypoints = self.dataset[index][1]
        orig_bbox = self.dataset[index][2]

        original_image_size = image.shape[:2]
        if self.transform:
            transformed = self.transform(image=image, keypoints=orig_keypoints, bbox=orig_bbox)
            image, keypoints, bbox = transformed["image"], transformed["keypoints"], transformed["bbox"]
        else:
            keypoints = orig_keypoints
            bbox = orig_bbox

        # convert all keypoints to integers values.
        # COCO keypoints can be floats if they specify the exact location of the keypoint (e.g. from CVAT)
        # even though COCO format specifies zero-indexed integers (i.e. every keypoint in the [0,1]x [0.1] pixel box becomes (0,0)
        # we convert them to ints here, as the heatmap generation will add a 0.5 offset to the keypoint location to center it in the pixel
        # the distance metrics also operate on integer values.

        # so basically from here on every keypoint is an int that represents the pixel-box in which the keypoint is located.
        keypoints = [
            [[math.floor(keypoint[0]), math.floor(keypoint[1])] for keypoint in channel_keypoints]
            for channel_keypoints in keypoints
        ]
        image = self.image_to_tensor_transform(image)
        return {
            "image": image,
            "keypoints": keypoints,
            "original_keypoints": orig_keypoints,
            "original_image_size": original_image_size,
            "bbox": bbox,
            "original_bbox": orig_bbox,
        }

    def prepare_dataset(self):  # noqa: C901
        """Prepares the dataset to map from COCO to (img, [keypoints for each channel])

        Returns:
            [img_path, [list of keypoints for each channel]]
        """
        with open(self.dataset_json_path, "r") as file:
            data = json.load(file)
            parsed_coco = CocoKeypointsDatasetParser(**data)

            img_dict: typing.Dict[int, CocoImage] = {}
            for img in parsed_coco.images:
                img_dict[img.id] = img

            category_dict: typing.Dict[int, CocoKeypointCategory] = {}
            for category in parsed_coco.categories:
                category_dict[category.id] = category

            # keypoint configuration: simply enumerate all keypoints
            assert len(parsed_coco.categories) == 1, "Only one category supported"

            
            available_keypoints = [keypoint for keypoint in parsed_coco.categories[0].keypoints]
            if self.keypoint_channel_configuration is None:
                self.keypoint_channel_configuration = [keypoint for keypoint in available_keypoints]
            else:
                # check if all keypoints are in the configuration
                for keypoint in self.keypoint_channel_configuration:
                    assert keypoint in available_keypoints, f"Keypoint {keypoint} not found in available keypoints: {available_keypoints}"

            print(f"{self.keypoint_channel_configuration=}")

            # iterate over all annotations and create a dict {img_id: {semantic_type : [keypoints]}}
            # make sure that each images has at most one annotation per semantic type
            annotation_dict = defaultdict(list)  # {img_id: [annotations]}
            for annotation in parsed_coco.annotations:
                # add all keypoints from this annotation to the corresponding image in the dict
                annotation_dict[annotation.image_id].append(annotation)

            # iterate over each image and all it's annotations
            # filter the visible keypoints
            # and group them by channel
            dataset = []
            for img_id, annotations in annotation_dict.items():
                assert len(annotations) <= 1, "Each image should have at most one annotation per category"
                annotation = annotations[0]
                bbox = annotation.bbox
                keypoints = annotation.keypoints
                keypoints = self.split_list_in_keypoints(keypoints)
                category = category_dict[annotation.category_id]
                img_channels_keypoints = [[] for _ in range(len(self.keypoint_channel_configuration))]
                for semantic_type, keypoint in zip(category.keypoints, keypoints):
                    if self.is_keypoint_visible(keypoint) and self.is_keypoint_in_image(keypoint, (img_dict[img_id].height, img_dict[img_id].width)):
                        channel_idx = self.get_keypoint_channel_index(semantic_type)
                        if channel_idx > -1:
                            img_channels_keypoints[channel_idx].append(keypoint[:2])
                dataset.append([img_dict[img_id].file_name, img_channels_keypoints, bbox])

            return dataset

    def get_keypoint_channel_index(self, semantic_type: str) -> int:
        """
        given a semantic type, get it's channel according to the channel configuration.
        Returns -1 if the semantic type couldn't be found.
        """

        for i, types_in_channel in enumerate(self.keypoint_channel_configuration):
            if semantic_type in types_in_channel:
                return i
        return -1

    def is_keypoint_visible(self, keypoint: list) -> bool:
        """
        Args:
            keypoint (list): [u,v,flag]

        Returns:
            bool: True if current keypoint is considered visible according to the dataset configuration, else False
        """
        if self.detect_only_visible_keypoints:
            # filter out occluded keypoints with flag 1.0
            return keypoint[2] > 1.5
        else:
            # filter out non-labeled keypoints with flag 0.0
            return keypoint[2] > 0.5

    def is_keypoint_in_image(self, keypoint: list, image_size: tuple) -> bool:
        """
        Args:
            keypoint (list): [u,v,flag]
            image_size (tuple): (width, height)
        """
        # kp is (u,v), size is (H,W)
        return 0 <= keypoint[0] < image_size[1] and 0 <= keypoint[1] < image_size[0]

    @staticmethod
    def split_list_in_keypoints(list_to_split: list) -> list:
        """
        splits list [u1,v1,f1,u2,v2,f2,...] to [[u,v,f],..]
        """
        n = 3
        output = [list_to_split[i : i + n] for i in range(0, len(list_to_split), n)]
        return output



if __name__ == "__main__":
    data_path = "/home/tlips/Code/droid/data/SPair-71k/SPAIR_coco_aeroplane_test.json"
    from few_shot_keypoints.datasets.augmentations import MultiChannelKeypointsCompose,  A
    from few_shot_keypoints.datasets.transforms import revert_max_length_resize_and_pad_transform
    import cv2

    res = 500
    transform = MultiChannelKeypointsCompose([
        A.LongestMaxSize(max_size=res, always_apply=True, interpolation=cv2.INTER_NEAREST),
        A.PadIfNeeded(min_height=res, min_width=res, border_mode=cv2.BORDER_CONSTANT, value=0, always_apply=True),
    ])
    dataset = COCOKeypointsDataset(data_path, transform=transform)
    for k, v in dataset[0].items():
        print(f"{k=}: {v=}")
    
    dataset_wo_transform = COCOKeypointsDataset(data_path, transform=None)
    
    keypoints_orig = dataset_wo_transform[0]["keypoints"]
    keypoints_trans = dataset[0]["keypoints"]
    keypoints_trans_orig = revert_max_length_resize_and_pad_transform(keypoints_trans, dataset_wo_transform[0]["original_image_size"], (res, res))

    print(f"{keypoints_orig=}, {keypoints_trans=}, {keypoints_trans_orig=}")

    for d in dataset:
        kp_trans = d["keypoints"]
        kp_trans_orig = revert_max_length_resize_and_pad_transform(kp_trans, d["original_image_size"], (res, res))
        kp_orig = d["original_keypoints"]
        assert kp_orig==kp_trans_orig, f"{kp_orig=}, {kp_trans_orig=}"

    print("all good")