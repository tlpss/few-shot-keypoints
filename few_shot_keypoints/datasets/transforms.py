from albumentations.pytorch import ToTensorV2
import albumentations as A
import cv2
from few_shot_keypoints.datasets.augmentations import MultiChannelKeypointsCompose


IMAGE_SIZE = 512
RESIZE_TRANSFORM = MultiChannelKeypointsCompose([A.Resize(IMAGE_SIZE,IMAGE_SIZE)]) # A.Normalize(mean=(0,0,0), std=(1,1,1),max_pixel_value=255),ToTensorV2()])
MAX_LENGTH_RESIZE_AND_PAD_TRANSFORM = MultiChannelKeypointsCompose([A.LongestMaxSize(max_size=IMAGE_SIZE), A.PadIfNeeded(min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT, value=0)]) # A.Normalize(mean=(0,0,0), std=(1,1,1),max_pixel_value=255),ToTensorV2()])


def revert_max_length_resize_and_pad_transform(keypoints, original_image_size, new_image_size):
    """ Revert geometric transform in which images and keypoints have been resized to fit largest dimension and then padded to square.
    This function reverts the augmentation to get the original keypoints.

    img -> resize -> pad -> new_image
    """
    largest_dim = max(original_image_size)
    scale = largest_dim / new_image_size[0]
    intermediate_size_before_padding = (original_image_size[0] * scale, original_image_size[1] * scale)
    # https://github.com/albumentations-team/albumentations/blob/66212d77a44927a29d6a0e81621d3c27afbd929c/albumentations/augmentations/geometric/functional.py#L3384C26-L3384C62
    padding_u= max(0, int(new_image_size[1] - intermediate_size_before_padding[1]) // 2)
    padding_v = max(0, int(new_image_size[0] - intermediate_size_before_padding[0]) // 2)
    # apply inverse scale to keypoints
    new_kp = []
    for kp in keypoints:
            new_kp.append([round((kp[0] - padding_u) * scale), round((kp[1] - padding_v) * scale)])
    return new_kp


def revert_resize_transform(keypoints, original_image_size, new_image_size):
    """ Revert geometric transform in which images and keypoints have been resized to fit largest dimension.
    This function reverts the augmentation to get the original keypoints.
    """
    scale_v = original_image_size[0] / new_image_size[0]
    scale_u = original_image_size[1] / new_image_size[1]
    new_kp = []
    for kp in keypoints:
        new_kp.append([round(kp[0] * scale_u, 2), round(kp[1] * scale_v, 2)])
    return new_kp

if __name__ == "__main__":
    keypoints = [[10, 20], [30, 40]]
    original_image_size = (100, 100)
    new_image_size = (80, 80)
    new_kp = revert_max_length_resize_and_pad_transform(keypoints, original_image_size, new_image_size)
    print(f"{new_kp=}")
    new_kp = revert_resize_transform(keypoints, original_image_size, new_image_size)
    print(f"{new_kp=}")