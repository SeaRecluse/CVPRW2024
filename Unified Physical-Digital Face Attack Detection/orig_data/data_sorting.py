import os
import glob
import shutil
from tqdm import tqdm

# 步骤 1: 查找所有的label.txt文件
label_files = glob.glob('./*/*label.txt')

# 步骤 3: 创建output文件夹
output_dir = 'data'
os.makedirs(output_dir, exist_ok=True)

for label_file in label_files:
    # 获取dev_label.txt文件所在的目录
    base_dir = os.path.dirname(label_file)
    
    # 读取txt文件并处理每行
    with open(label_file, 'r') as file:
        for line in tqdm(file):
            # 移除空白字符并分割路径和类别
            parts = line.strip().split(' ')
            image_path = parts[0]  # 图片路径，如 'p1/dev/000001.jpg'
            category = parts[1]    # 类别，'0' 或 '1'

            # 构造目标文件夹和文件路径
            # 如 'data/p1/dev/1/' 和 'data/p1/dev/1/000001.jpg'
            target_dir = os.path.join(output_dir, os.path.dirname(image_path), category)
            target_file = os.path.join(target_dir, os.path.basename(image_path))

            # 创建目标文件夹
            os.makedirs(target_dir, exist_ok=True)

            # 复制图片到目标文件夹
            shutil.copyfile(image_path, target_file)
            # print(f"Copied {image_path} to {target_file}")
