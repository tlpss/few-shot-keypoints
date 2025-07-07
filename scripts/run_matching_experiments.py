from few_shot_keypoints.match_dataset import Config,main
import os

datasets = ["train_train", "bicycle_train","tvmonitor_train", "car_train","aeroplane_train"]

# assert all datasets exist
for dataset in datasets:
    assert os.path.exists(f"/home/tlips/Code/droid/data/SPair-71k/SPAIR_coco_{dataset}.json")

n_query_range = [1,2,3,4,5]
n_query_sets = 10

for dataset in datasets:
    for n_query in n_query_range:
        config = Config(dataset_category=dataset, N_query_images=n_query, N_query_sets=n_query_sets)
        print(f"running {dataset} with {n_query} query images and {n_query_sets} query sets")
        main(config)