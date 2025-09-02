from few_shot_keypoints.featurizers.registry import FeaturizerRegistry
from few_shot_keypoints.featurizers import ViT_featurizer # noqa
from few_shot_keypoints.featurizers import dift_featurizer # noqa



if __name__ == "__main__":
    print(FeaturizerRegistry.list())
