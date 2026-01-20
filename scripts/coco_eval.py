from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
from airo_dataset_tools.data_parsers.coco import CocoKeypointsDataset
import json

# Your annotation and result files
annFile = "/home/tlips/Code/few-shot-keypoints/data/SPair-71k/SPAIR_coco_aeroplane_test.json"
resFile = "/home/tlips/Code/few-shot-keypoints/test.json"
# annFile = "/home/tlips/Code/few-shot-keypoints/data/SPair-71k/SPAIR_coco_tvmonitor_test.json"
# resFile = "/home/tlips/Code/few-shot-keypoints/test.json"
with open(annFile, "r") as f:
    coco_dataset = CocoKeypointsDataset(**json.load(f))
n_keypoints = len(coco_dataset.categories[0].keypoints)

# Load datasets
cocoGt = COCO(annFile)
cocoDt = cocoGt.loadRes(resFile)

# Create COCOeval object for keypoints
cocoEval = COCOeval(cocoGt, cocoDt, iouType='keypoints')

# Set default sigmas as the coco human eval sigmas

sigmas = cocoEval.params.kpt_oks_sigmas
avg_sigma = np.mean(sigmas)
custom_sigmas = np.array([avg_sigma] * n_keypoints)
cocoEval.params.kpt_oks_sigmas = custom_sigmas

# Now run the evaluation
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

