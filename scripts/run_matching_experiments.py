# add scripts to path
import sys
import os
# add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from few_shot_keypoints.featurizers.registry import FeaturizerRegistry
from scripts.match_dataset import match_dataset, Config as MatchDatasetConfig
from dataclasses import dataclass, field

@dataclass
class Config:
    categories: list[str] = field(default_factory=lambda: ["train", "aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow","dog","horse","motorbike","person","pottedplant","sheep","train","tvmonitor"])
    # how many support sets (= seeds) to run per (category, featurizer). Each support set uses a single reference
    # image per keypoint channel: the matcher holds one reference vector per channel, so there is no N_support_images
    # dimension to sweep (the parameter this script used to pass does not exist on MatchDatasetConfig).
    N_support_sets: int = 5
    featurizers: list[str] = field(default_factory=lambda: ["dinov2-s"])
    transform : str = "resize"
    output_base_dir: str = "results/SPAIR-correspondences"

def run_matching_experiments(config: Config):
    # check all datasets exist
    for category in config.categories:
        train_dataset_path = f"/home/tlips/Code/few-shot-keypoints/data/SPair-71k/SPAIR_coco_{category}_train.json"
        test_dataset_path = f"/home/tlips/Code/few-shot-keypoints/data/SPair-71k/SPAIR_coco_{category}_test.json"
        if not os.path.exists(train_dataset_path):
            raise FileNotFoundError(f"Train dataset not found: {train_dataset_path}")
        if not os.path.exists(test_dataset_path):
            raise FileNotFoundError(f"Test dataset not found: {test_dataset_path}")

    for category in config.categories:
        train_dataset_path = f"/home/tlips/Code/few-shot-keypoints/data/SPair-71k/SPAIR_coco_{category}_train.json"
        test_dataset_path = f"/home/tlips/Code/few-shot-keypoints/data/SPair-71k/SPAIR_coco_{category}_test.json"
        for featurizer in config.featurizers:
            for i in range(config.N_support_sets):
                print(f"Running experiment for category {category}, featurizer {featurizer}, seed {2025 + i}")
                experiment_config = MatchDatasetConfig(
                    train_dataset_path=train_dataset_path,
                    test_dataset_path=test_dataset_path,
                    seed=2025 + i,
                    featurizer=featurizer,
                    transform=config.transform,
                    output_base_dir=config.output_base_dir
                )
                match_dataset(experiment_config)


if __name__ == "__main__":
    config = Config()
    config.featurizers = FeaturizerRegistry.list()
    run_matching_experiments(config)