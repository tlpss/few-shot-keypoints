from pathlib import Path

file_path = Path(__file__)

DATA_DIR = file_path.parents[1] / "data"

aRTF_DIR = DATA_DIR / "aRTF"
SPAIR_DIR = DATA_DIR / "SPair-71k"
DSD_DIR  = DATA_DIR / "dsd"

DSD_SHOE_DIR = DSD_DIR / "dsd-shoes-real_resized_512x512"
DSD_SHOE_TRAIN_JSON = DSD_SHOE_DIR / "annotations_train.json"
DSD_SHOE_TEST_JSON = DSD_SHOE_DIR / "annotations_val.json"

DSD_MUGS_DIR = DSD_DIR / "lab-mugs_resized_512x512"
DSD_MUGS_TRAIN_JSON = DSD_MUGS_DIR / "lab-mugs_train.json"
DSD_MUGS_TEST_JSON = DSD_MUGS_DIR / "lab-mugs_val.json"
DSD_MUGS_JSON = DSD_MUGS_DIR / "lab-mugs.json"
DSD_MUGS_HUMAN_EVAL_JSON = DSD_MUGS_DIR / "human-eval-victor.json"

if __name__ == "__main__":
    print(f"{DATA_DIR=}")
    
    # dynamically check if all global variables that are Path objects exist and are valid paths
    for var in list(globals().values()):
        if isinstance(var, Path):
            assert var.exists(), f"{var=} does not exist"



