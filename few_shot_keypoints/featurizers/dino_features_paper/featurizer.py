from few_shot_keypoints.featurizers.base import BaseFeaturizer
from few_shot_keypoints.featurizers.dino_features_paper.extractor_skil import ViTExtractor
from few_shot_keypoints.featurizers.registry import FeaturizerRegistry
import torch
from torchvision import transforms

class ViTPaperFeaturizer(BaseFeaturizer):
    """ ViT Featurizer using the features from this paper: 

     - reducing stride to increase spatial resolution.
     - use transformer key values instead of token values as features
     - combine neighboring tokens into a single feature for more spatial awareness.
    """
    def __init__(self, model_type: str = 'dino_vits8', stride: int = 4, device: str = 'cuda:0', layer: int = 11,
     facet: str = 'key', use_bin_features: bool = False):
        self.extractor = ViTExtractor(model_type, stride, device=device)
        self.layer = layer
        self.facet = facet
        self.use_bin_features = use_bin_features
        self.device = device

        self.normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    @torch.no_grad()
    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        assert len(image.shape) == 4 # [BATCH_SIZE, 3, IMG_HEIGHT, IMG_WIDTH]
        assert image.shape[1] == 3
        assert image.shape[0] == 1
        # check if in range 0-1
        assert torch.all(image >= 0) and torch.all(image <= 1), "image is not in range 0-1"
        assert torch.max(image) >= 1/255.0, "highest value in image is less than 1/255.0, might have rescaled it twice?"


        # normalize, but no resize here. this is responsibility of the user.
        image = self.normalizer(image)

        # resize to nearest feasible size, given stride and patch size
        # compute number of tokens taking stride into account
        n_patches_w = 1 + (image.shape[3] - self.extractor.p) // self.extractor.stride[1]
        n_patches_h = 1 + (image.shape[2] - self.extractor.p) // self.extractor.stride[0]

        image_w = (n_patches_w - 1) * self.extractor.stride[1] + self.extractor.p
        image_h = (n_patches_h - 1) * self.extractor.stride[0] + self.extractor.p
        # print(f"image_w: {image_w}, image_h: {image_h}")

        resized_image = torch.nn.functional.interpolate(image, size=(image_h, image_w), mode='bilinear', align_corners=False)


        descriptors = self.extractor.extract_descriptors(resized_image.to(self.device), self.layer, self.facet, self.use_bin_features)
        # create 2d grid of descriptors, taking stride and patch size into account
        descriptors = descriptors.reshape(1,n_patches_h, n_patches_w, descriptors.shape[-1])
        # upsample to original image size
        # reshape to [B,D,H',W']
        upsampled_descriptors = descriptors.permute(0,3,1,2)
        upsampled_descriptors = torch.nn.functional.interpolate(upsampled_descriptors, size=(image.shape[2], image.shape[3]), mode='bilinear', align_corners=False)
        return upsampled_descriptors

@FeaturizerRegistry.register("dinov2-s-paper")
class Dinov2sPaperFeaturizer(ViTPaperFeaturizer):
    def __init__(self, device: str = 'cuda:0'):
        # dinov2 variant with settings as in the paper
        super().__init__(model_type='dinov2_vits14', stride=7, device=device, layer=9, facet='key', use_bin_features=True)

@FeaturizerRegistry.register("dino-paper")
class DinoPaperFeaturizer(ViTPaperFeaturizer):
    def __init__(self, device: str = 'cuda:0'):
        # dinov1 - same as in the paper but had to disable bin_features to fit in memory..
        super().__init__(model_type='dino_vits8', stride=4, device=device, layer=9, facet='key', use_bin_features=False)

@FeaturizerRegistry.register("dinov2-s-paper-hf-equivalent")
class Dinov2sPaperHfEquivalentFeaturizer(ViTPaperFeaturizer):
    def __init__(self, device: str = 'cuda:0'):
        # dinov2 variant with settings that match HF Dinov2S model.
        super().__init__(model_type='dinov2_vits14', stride=14, device=device, layer=11, facet='token', use_bin_features=False)


if __name__ == "__main__":
    # settings similar to my HF model.
    featurizer = ViTPaperFeaturizer(model_type='dinov2_vits14', stride=7, device='cuda:0', layer=11, facet='token', use_bin_features=False)
    image = torch.rand(1, 3, 512, 512)
    descriptors = featurizer.extract_features(image)
    print(descriptors.shape)