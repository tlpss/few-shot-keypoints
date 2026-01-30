"""Run matching experiments for DSD shoes and DSD mugs datasets."""
# add scripts to path
import sys
import os
import pathlib
# add parent directory to path
scripts_dir = pathlib.Path(__file__).parent.parent.parent.parent
sys.path.append(str(scripts_dir))
from few_shot_keypoints.featurizers.registry import FeaturizerRegistry

from scripts.datasets.kil.run_kil_matching import DATASETS
from scripts.match_dataset import Config, match_dataset

# Featurizers to test
FEATURIZERS = [

    "dinov3-b",
    "dinov3-l",

    "radiov2-b",
    "radiov2-l",


    "dift-sd2.1-e1-r768",
    "dift-sd2.1-e2-r768",
    "dift-sd2.1-e4-r768",

    
]

for featurizer in FEATURIZERS:
    assert featurizer in FeaturizerRegistry.list(), f"Featurizer {featurizer} not found in registry"
    print(f"Featurizer {featurizer} found in registry")

def main():
    for seed in [2025,2026,2027, 2028]:
        for dataset_name, paths in DATASETS.items():
            for featurizer in FEATURIZERS:
                print(f"\n{'='*60}")
                print(f"Running: {dataset_name} with {featurizer}")
                print(f"{'='*60}")
                
                config = Config(
                    train_dataset_path=paths["train"],
                    test_dataset_path=paths["test"],
                    output_base_dir="results/KIL_crop",
                    featurizer=featurizer,
                    seed=seed,
                    transform="none",
                    dataset_name=dataset_name,
                    crop_before_matching=True,
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
