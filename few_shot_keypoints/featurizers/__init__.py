# Import all featurizer modules to populate the registry
import few_shot_keypoints.featurizers.ViT_featurizer 
import few_shot_keypoints.featurizers.dift_featurizer
import few_shot_keypoints.featurizers.dino_features_paper.featurizer


if __name__ == "__main__":
    from few_shot_keypoints.featurizers.registry import FeaturizerRegistry
    print(FeaturizerRegistry.list())
