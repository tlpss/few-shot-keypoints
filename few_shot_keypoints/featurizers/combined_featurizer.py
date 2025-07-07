import torch

class CombinedFeaturizer:
    def __init__(self, featurizers):
        self.featurizers = featurizers

    def extract_features(self, image: torch.Tensor):
        features = []
        for featurizer in self.featurizers:
            features.append(featurizer.extract_features(image))
        return torch.cat(features, dim=1)

