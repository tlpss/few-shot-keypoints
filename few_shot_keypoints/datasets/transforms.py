from albumentations import Resize


def revert_max_length_resize_and_pad_transform(keypoints, original_image_size, new_image_size):
    """ Revert geometric transform in which images and keypoints have been resized to fit largest dimension and then padded to square.
    This function reverts the augmentation to get the original keypoints.

    img -> resize -> pad -> new_image
    """
    largest_dim = max(original_image_size)
    scale = largest_dim / new_image_size[0]
    intermediate_size_before_padding = (original_image_size[0] * scale, original_image_size[1] * scale)
    padding_u= (new_image_size[1] - intermediate_size_before_padding[1]) // 2
    padding_v = (new_image_size[0] - intermediate_size_before_padding[0]) // 2
    # apply inverse scale to keypoints
    #TODO:.
    new_kp = []
    for channel in keypoints:
        new_kp.append([])
        for kp in channel:
            new_kp[-1].append([round((kp[0] - padding_u) / scale), round((kp[1] - padding_v) / scale)])
    return new_kp


if __name__ == "__main__":
    keypoints = [[[10, 20]], [[30, 40]]]
    original_image_size = (100, 100)
    new_image_size = (50, 50)
    new_kp = revert_max_length_resize_and_pad_transform(keypoints, original_image_size, new_image_size)
    print(f"{new_kp=}")