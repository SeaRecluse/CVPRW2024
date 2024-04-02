import os
import shutil
from tqdm import tqdm

load_train_label = "./HySpeFAS_trainval/train.txt"
load_val_label = "./HySpeFAS_trainval/val_gt.txt"
load_dir = "./HySpeFAS_trainval/images"

os.makedirs("./data", exist_ok="True")
os.makedirs("./data/train", exist_ok="True")
os.makedirs("./data/train/0", exist_ok="True")
os.makedirs("./data/train/1", exist_ok="True")

os.makedirs("./data/dev", exist_ok="True")
os.makedirs("./data/dev/0", exist_ok="True")
os.makedirs("./data/dev/1", exist_ok="True")

with open(load_train_label, "r") as f:
    lines = f.readlines()
    for line in tqdm(lines):
        line = line.replace("\n", "")
        if line:
            info = line.replace("mat", "png").split(" ")
            img_name = info[0]
            cls = info[-1]
            if cls == "0":
                shutil.copyfile(os.path.join(load_dir, img_name), os.path.join("./data/train/0", img_name))
            else:
                shutil.copyfile(os.path.join(load_dir, img_name), os.path.join("./data/train/1", img_name))

with open(load_val_label, "r") as f:
    lines = f.readlines()
    for line in tqdm(lines):
        line = line.replace("\n", "")
        if line:
            info = line.replace("mat", "png").split(" ")
            img_name = info[0]
            cls = info[-1]
            if cls == "0":
                shutil.copyfile(os.path.join(load_dir, img_name), os.path.join("./data/dev/0", img_name))
            else:
                shutil.copyfile(os.path.join(load_dir, img_name), os.path.join("./data/dev/1", img_name))

