from few_shot_keypoints.featurizers.ViT_featurizer import ViTFeaturizer
from few_shot_keypoints.featurizers.dift_featurizer import SDFeaturizer
from few_shot_keypoints.featurizers.combined_featurizer import CombinedFeaturizer
import torch
import time

def test_inference_speed(featurizer: ViTFeaturizer | SDFeaturizer | CombinedFeaturizer):
    image = torch.randn(1, 3, 224, 224)
    start_time = time.time()
    for i in range(100):
        featurizer.extract_features(image)
    end_time = time.time()
    return (end_time - start_time) / 100

if __name__ == "__main__":
    featurizer = ViTFeaturizer("facebook/dinov2-large", [11])
    print(f"ViTFeaturizer: {test_inference_speed(featurizer)}")
    featurizer = SDFeaturizer()
    print(f"SDFeaturizer: {test_inference_speed(featurizer)}")
    featurizer = CombinedFeaturizer([SDFeaturizer(), ViTFeaturizer("facebook/dinov2-small", [11])])
    print(f"CombinedFeaturizer: {test_inference_speed(featurizer)}")





