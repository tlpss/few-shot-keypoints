
import json 


json_train_path = "/home/tlips/Code/few-shot-keypoints/data/aRTF/tshirts-train_resized_512x256/tshirts-train.json"
json_test_path = "/home/tlips/Code/few-shot-keypoints/data/aRTF/tshirts-test_resized_512x256/tshirts-test.json"
json_validation_path = "/home/tlips/Code/few-shot-keypoints/data/aRTF/tshirts-val_resized_512x256/tshirts-val.json"

json_train_path = "/home/tlips/Code/few-shot-keypoints/data/SPair-71k/SPAIR_coco_aeroplane_train.json"
json_test_path = "/home/tlips/Code/few-shot-keypoints/data/SPair-71k/SPAIR_coco_aeroplane_test.json"
json_validation_path = json_test_path



# create channel config by extracting all keypoint categories from the json_train_path
with open(json_train_path, "r") as f:
    data = json.load(f)
    keypoint_categories = [item for item in data["categories"][0]["keypoints"]]
    channel_config = ":".join(keypoint_categories)

command = f"""
uv run keypoint-detection train
 --augment_train 
 --keypoint_channel_configuration {channel_config}
 --accelerator gpu 
 --ap_epoch_freq 40 
 --backbone_type DinoV2Up 
 --devices 1 
 --early_stopping_relative_threshold -1 
 --json_dataset_path {json_train_path}  
 --json_test_dataset_path {json_test_path} 
 --json_validation_dataset_path {json_validation_path}
 --max_epochs 160
 --maximal_gt_keypoint_pixel_distances "8 16"
 --minimal_keypoint_extraction_pixel_distance 8
 --precision 16
 --seed 2024
 --heatmap_sigma 8
 --learning_rate 0.0001
 --batch_size 4
 --wandb_project few-shot-keypoints
 --wandb_name {json_train_path.split("/")[-1].split(".")[0]}
"""

print("running command")
print(command)

# remove all linebreaks from command
command = command.replace("\n", "")





import os 

os.system(command)