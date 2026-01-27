"""Run matching experiments for DSD shoes and DSD mugs datasets."""
# add scripts to path
import sys
import os
# add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from few_shot_keypoints.paths import (
    DSD_SHOE_TRAIN_JSON, DSD_SHOE_TEST_JSON,
    DSD_MUGS_TRAIN_JSON, DSD_MUGS_TEST_JSON
)
from scripts.match_dataset import Config, match_dataset

# Datasets to run
DATASETS = {
    "dsd-shoes": {
        "train": str(DSD_SHOE_TRAIN_JSON),
        "test": str(DSD_SHOE_TEST_JSON),
    },
    "dsd-mugs": {
        "train": str(DSD_MUGS_TRAIN_JSON),
        "test": str(DSD_MUGS_TEST_JSON),
    },
}

# Featurizers to test
FEATURIZERS = [
    "dinov2-s",
    "dinov2-s-paper",
    "dino-paper",
    "dinov2-s-paper-hf-equivalent",
    "dinov2-b",
    "radiov2-b",
    "dift-sd2.1-e1",
]

def main():
    # Run multiple seeds, similar to KIL experiments
    for seed in [2027, 2028]:
        for dataset_name, paths in DATASETS.items():
            for featurizer in FEATURIZERS:
                print(f"\n{'='*60}")
                print(f"Running: {dataset_name} with {featurizer} (seed={seed})")
                print(f"{'='*60}")
                
                config = Config(
                    train_dataset_path=paths["train"],
                    test_dataset_path=paths["test"],
                    output_base_dir="results/DSD",
                    featurizer=featurizer,
                    seed=seed,
                    transform="resize",
                )
                
                try:
                    match_dataset(config)
                except Exception as e:
                    print(f"Error running {dataset_name} with {featurizer} (seed={seed}): {e}")
                    # print stack trace
                    import traceback
                    traceback.print_exc()
                    continue

if __name__ == "__main__":
    main()
