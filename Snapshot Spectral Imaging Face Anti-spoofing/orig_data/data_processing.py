import os
import shutil
from tqdm import tqdm
import numpy as np
import cv2
import random

orig_dir = "./data"
extend_dir = "./data_extend"

def get_img_paths(orig_dir, extend_dir):
    image_extensions = ['.jpg', '.png', '.jpeg', '.gif', '.bmp']
    load_img_list = []
    save_img_list = []

    if os.path.exists(extend_dir):
        shutil.rmtree(extend_dir)

    for dirpath, dirnames, files in os.walk(orig_dir):
        structure = os.path.join(extend_dir, os.path.relpath(dirpath, orig_dir))
        if not os.path.isdir(structure):
            os.makedirs(structure)

        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                orig_path = os.path.join(dirpath, file)
                load_img_list.append(orig_path)
                save_img_list.append(orig_path.replace(orig_dir, extend_dir))

    return load_img_list, save_img_list

if os.path.exists(orig_dir):
    load_img_list, save_img_list = get_img_paths(orig_dir, extend_dir)

    for n in tqdm(range(len(load_img_list))):
        load_path = load_img_list[n]
        save_path = save_img_list[n]

        img = cv2.imread(load_path)
        H, W, C = img.shape
        cls = load_path.replace("\\", "/").split("/")[-2]

        cv2.imwrite(save_path, img)

        L = max(H, W)
        l = min(H, W)

        if cls == "0":
            choice = random.choice([0 , 1])
            if choice:
                img_downsample = cv2.resize(img, (l // 2, l // 2), cv2.INTER_CUBIC)
                save_downsample = save_path.replace(".png", "_ds2x.png")
                cv2.imwrite(save_downsample, img_downsample)
                
