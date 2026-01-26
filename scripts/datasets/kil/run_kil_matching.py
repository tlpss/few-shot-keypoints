"""Run matching experiments for DSD shoes and DSD mugs datasets."""
# add scripts to path
import sys
import os
import pathlib
# add parent directory to path
scripts_dir = pathlib.Path(__file__).parent.parent.parent.parent
sys.path.append(str(scripts_dir))


from few_shot_keypoints.paths import (
    KIL_SHOE_V2_INITIAL_JSON,
)
from scripts.match_dataset import Config, match_dataset
from few_shot_keypoints.featurizers.registry import FeaturizerRegistry

# Datasets to run
DATASETS = {
    "shoe-v2-initial-frames": {
        "train": str(KIL_SHOE_V2_INITIAL_JSON),
        "test": str(KIL_SHOE_V2_INITIAL_JSON),
    },
}

# Featurizers to test
FEATURIZERS = [
    "dinov2-s",
    "dinov2-b",
    "radiov2-b",
    "radiov2-h",
    "dift-sd2.1-e1",
    "dift-sd2.1-e4",
    "dift-sd2.1-e4-r512",
    "dift-sd2.1-e4-r768",
    "dift-sd2.1b-e4",
    "dift-sd2.1b-e4-r512",
    "dift-sd2.1-e8",
    "dift-sd2.1-e8-r512",
    "dift-sd2.1-e8-r768",
    "dift-sd2.1b-e8",
    "dift-sd2.1b-e8-r512",
    
]

for featurizer in FEATURIZERS:
    assert featurizer in FeaturizerRegistry.list(), f"Featurizer {featurizer} not found in registry"

def main():
    for dataset_name, paths in DATASETS.items():
        for featurizer in FEATURIZERS:
            print(f"\n{'='*60}")
            print(f"Running: {dataset_name} with {featurizer}")
            print(f"{'='*60}")
            
            config = Config(
                train_dataset_path=paths["train"],
                test_dataset_path=paths["test"],
                output_base_dir="results/KIL",
                featurizer=featurizer,
                seed=2025,
                transform="none",
                dataset_name=dataset_name,
            )
            
            try:
                match_dataset(config)
            except Exception as e:
                print(f"Error running {dataset_name} with {featurizer}: {e}")
                # print stack trace
                import traceback
                traceback.print_exc()
                continue

if __name__ == "__main__":
    main()
