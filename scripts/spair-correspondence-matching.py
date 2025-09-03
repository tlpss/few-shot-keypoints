""" taken from https://github.com/Tsingularity/dift/blob/main/eval_spair.py
and slightly adapted to this codebase: do not precompute maps for all images, but compute them on the fly. slower, but easier.

"""

import argparse
import torch
from torch.nn import functional as F
import torchvision.transforms.functional as TF
from tqdm import tqdm
import numpy as np
from few_shot_keypoints.featurizers.ViT_featurizer import ViTFeaturizer
from few_shot_keypoints.featurizers.dift_featurizer import SDFeaturizer
import os
import json
from PIL import Image
import torch.nn as nn
from few_shot_keypoints.featurizers import FeaturizerRegistry
import time

def preprocess_image(img, img_size):
    img = torch.from_numpy(np.array(img)).permute(2,0,1).unsqueeze(0)
    img = TF.resize(img, img_size)
    img = img /255.0
    return img

def main(args):
    for arg in vars(args):
        value = getattr(args,arg)
        if value is not None:
            print('%s: %s' % (str(arg),str(value)))

    torch.cuda.set_device(0)

    dataset_path = args.dataset_path
    test_path = 'PairAnnotation/test'
    json_list = os.listdir(os.path.join(dataset_path, test_path))
    all_cats = os.listdir(os.path.join(dataset_path, 'JPEGImages'))
    cat2json = {}

    for cat in all_cats:
        cat_list = []
        for i in json_list:
            if cat in i:
                cat_list.append(i)
        cat2json[cat] = cat_list

    # get test image path for all cats
    cat2img = {}
    for cat in all_cats:
        cat2img[cat] = []
        cat_list = cat2json[cat]
        for json_path in cat_list:
            with open(os.path.join(dataset_path, test_path, json_path)) as temp_f:
                data = json.load(temp_f)
                temp_f.close()
            src_imname = data['src_imname']
            trg_imname = data['trg_imname']
            if src_imname not in cat2img[cat]:
                cat2img[cat].append(src_imname)
            if trg_imname not in cat2img[cat]:
                cat2img[cat].append(trg_imname)

 
    featurizer = FeaturizerRegistry.create(args.featurizer)
    print(featurizer)



    # print("saving all test images' features...")
    # os.makedirs(args.save_path, exist_ok=True)
    # for cat in tqdm(all_cats):
    #     output_dict = {}
    #     image_list = cat2img[cat]
    #     for image_path in tqdm(image_list[:2]):
    #         img = Image.open(os.path.join(dataset_path, 'JPEGImages', cat, image_path))
    #         img = preprocess_image(img, args.img_size)
    #         feat  =dift.extract_features(img,
    #                                             prompt=f"a photo of a {cat}",
    #                                             t=args.t,
    #                                             up_ft_index=args.up_ft_index,
    #                                             ensemble_size=args.ensemble_size).cpu()
    #         # to float 16
    #         output_dict[image_path] = feat.to(torch.float16)

    #     torch.save(output_dict, os.path.join(args.save_path, f'{cat}.pth'))

    total_pck = []
    all_correct = 0
    all_total = 0

    for cat in tqdm(all_cats, desc="SPAIR"):
        cat_list = cat2json[cat]
        # output_dict = torch.load(os.path.join(args.save_path, f'{cat}.pth'))

        cat_pck = []
        cat_correct = 0
        cat_total = 0

        if args.num_pairs_per_category == -1:
            iterable = tqdm(cat_list)
        else:
            iterable = tqdm(cat_list[:args.num_pairs_per_category])

        for json_path in iterable:
            iterable.set_description(f"Processing {cat}, running PCK: {cat_correct/(cat_total+1e-6):.2f}")

            with open(os.path.join(dataset_path, test_path, json_path)) as temp_f:
                data = json.load(temp_f)

            src_img_size = data['src_imsize'][:2][::-1]
            trg_img_size = data['trg_imsize'][:2][::-1]

            src_img = Image.open(os.path.join(dataset_path, 'JPEGImages', cat, data['src_imname']))
            trg_img = Image.open(os.path.join(dataset_path, 'JPEGImages', cat, data['trg_imname']))
            src_img = preprocess_image(src_img, args.img_size)
            trg_img = preprocess_image(trg_img, args.img_size)

            # src_ft = output_dict[data['src_imname']]
            # trg_ft = output_dict[data['trg_imname']]

            # instead of precomputing all 
            before_inference = time.time()
            src_ft = featurizer.extract_features(src_img,
                                            prompt=f"a photo of a {cat}")
            trg_ft = featurizer.extract_features(trg_img,
                                            prompt=f"a photo of a {cat}")
            after_inference = time.time()
            #print(f"Inference time: {after_inference - before_inference}")
            # resize both to the original image size, as is done in the original code.
            #src_ft = nn.Upsample(size=src_img_size, mode='bilinear')(src_ft)
            #trg_ft = nn.Upsample(size=trg_img_size, mode='bilinear')(trg_ft)
            src_ft = TF.resize(src_ft, src_img_size)
            trg_ft = TF.resize(trg_ft, trg_img_size)

            h = trg_ft.shape[-2]
            w = trg_ft.shape[-1]

            trg_bndbox = data['trg_bndbox']
            threshold = max(trg_bndbox[3] - trg_bndbox[1], trg_bndbox[2] - trg_bndbox[0])

            total = 0
            correct = 0

            num_channel = src_ft.size(1)
            trg_vec = trg_ft.view(num_channel, -1).transpose(0, 1) # HW, C
            trg_vec = F.normalize(trg_vec) # HW, c

            for idx in range(len(data['src_kps'])):
                total += 1
                cat_total += 1
                all_total += 1
                src_point = data['src_kps'][idx]
                trg_point = data['trg_kps'][idx]

                src_vec = src_ft[0, :, src_point[1], src_point[0]].view(1, num_channel) # 1, C
                src_vec = F.normalize(src_vec).transpose(0, 1) # c, 1
                cos_map = torch.mm(trg_vec, src_vec).view(h, w)# H, W            
                # max_yx = np.unravel_index(cos_map.argmax(), cos_map.shape)
                max_yx = torch.argmax(cos_map)
                max_yx = torch.div(max_yx, cos_map.shape[1], rounding_mode='floor'), max_yx % cos_map.shape[1]

                dist = ((max_yx[1] - trg_point[0]) ** 2 + (max_yx[0] - trg_point[1]) ** 2) ** 0.5
                if (dist / threshold) <= 0.1:
                    correct += 1
                    cat_correct += 1
                    all_correct += 1

            cat_pck.append(correct / total)

            after_matching = time.time()
            #print(f"Matching time: {after_matching - after_inference}")

        total_pck.extend(cat_pck)
 
        print(f'{cat} per image PCK@0.1: {np.mean(cat_pck) * 100:.2f}')
        print(f'{cat} per point PCK@0.1: {cat_correct / cat_total * 100:.2f}')
    print(f'All per image PCK@0.1: {np.mean(total_pck) * 100:.2f}')
    print(f'All per point PCK@0.1: {all_correct / all_total * 100:.2f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SPair-71k Evaluation Script')
    parser.add_argument('--dataset_path', type=str, default='./data/SPair-71k/', help='path to spair dataset')
    parser.add_argument('--save_path', type=str, default='/storage/tlips/scratch_spair_ft/', help='path to save features')
    # parser.add_argument('--dift_model', choices=['sd', 'adm'], default='sd', help="which dift version to use")
    parser.add_argument('--img_size', nargs='+', type=int, default=[768, 768],
                        help='''in the order of [width, height], resize input image
                            to [w, h] before fed into  model, if set to 0, will
                            stick to the original input size. by default is 768x768.''')
    parser.add_argument('--num_pairs_per_category', default=-1, type=int, help='number of pairs to process per category')
    parser.add_argument('--featurizer', type=str, default='dinov3-l', help='featurizer to use')
    args = parser.parse_args()
    main(args)