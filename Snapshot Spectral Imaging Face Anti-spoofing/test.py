import os
#import models.model
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import timm
from tqdm import tqdm
import argparse
import numpy as np

#device = torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class_nums = 2
INPUT_SIZE = 224
INPUT_CHNS = 3
#===================================================================
def model_load(model_path, data_type):
    model = timm.create_model('convnextv2_base.fcmae_ft_in22k_in1k',pretrained=False, num_classes=class_nums)
    checkpoint = torch.load(os.path.join(model_path,data_type,"checkpoint-best.pth"), map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device = device)
    model.eval()
    return model
#===================================================================
trans_test = transforms.Compose([
            transforms.Resize([INPUT_SIZE, INPUT_SIZE], interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.2254, 0.225])
            ])

softmax = nn.Softmax(dim=1)

def as_num(x):
     y='{:.6f}'.format(x)
     return y

def getPred(img, model):
    input_img = Image.fromarray(img)
    input_tensor = trans_test(input_img).to(device = device)
    input_tensor = input_tensor.view(1, INPUT_CHNS, INPUT_SIZE, INPUT_SIZE)

    return softmax(model(input_tensor)).detach().cpu().numpy()[0]

def checkPath(path):
    if not os.path.exists(path):
        print("{} is not exists!".format(path))

def clearRes(save_path):
    if os.path.exists(save_path):
        os.remove(save_path)

def get_score(img_path, model, use_tta=True):

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    output = getPred(img, model)
    if use_tta:
        img_flip = cv2.flip(img, 1)
        output_flip = getPred(img_flip, model)
        score = (output[1] + output_flip[1]) / 2
    else:
        score = output[1]

    return score


def get_img_paths(load_dir):
    image_extensions = ['.jpg', '.png', '.jpeg', '.gif', '.bmp']
    load_img_list = []

    for dirpath, _, files in os.walk(load_dir):

        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                orig_path = os.path.join(dirpath, file)
                load_img_list.append(orig_path)


    return load_img_list


def get_dim_idx(score_list, th):
    """
    在分数列表中找到第一个大于阈值的位置。
    :param score_list: 分数列表
    :param th: 阈值
    :return: 索引位置
    """
    for idx, score in enumerate(score_list):
        if score > th:
            return idx
    return len(score_list)  # 如果没有找到，则返回最后一个索引

def accuracy_detailed(roc_n_list, roc_p_list, fpr_list=[1e-10, 1e-4, 1e-3, 1e-2]):
    """
    计算详细的分类准确性指标。
    :param roc_n_list: 负样本的分数列表
    :param roc_p_list: 正样本的分数列表
    :param fpr_list: 假正率阈值列表
    :return: (roc_info字符串, roc_dict字典)
    """
    roc_n_list.sort(reverse=True)
    roc_p_list.sort()
    n_nums = len(roc_n_list)
    p_nums = len(roc_p_list)
    roc_info = ""
    roc_dict = {}

    #============================== TN TP ACC ========================
    tn, fn, tp, fp = 0, 0, 0, 0
    for i in range(n_nums):
        if  roc_n_list[i] <= 0.5:
            break
        fn = i + 1
    tn = n_nums - fn
  
    for i in range(p_nums):
        if roc_p_list[i] > 0.5:
            break
        fp = i + 1
    tp = p_nums - fp

    roc_dict.update({
        "TN"    : tn / len(roc_n_list),
        "TP"    : tp / len(roc_p_list),
        "ACC"   : (tp + tn) / (len(roc_n_list) + len(roc_p_list))
    })
    roc_info += "   TN: {}% TP: {}% ACC: {}%\n".format(
            round(roc_dict["TN"] * 100, 5),
            round(roc_dict["TP"] * 100, 5),
            round(roc_dict["ACC"] * 100, 5)
        )

    #============================== TPR FPR ==========================
    roc_dict.update({
        "FPR_list"    : [],
        "TPR_list"    : [],
        "TH_list"     : []
    })
    for fpr in fpr_list:
        n_idx = (int)(n_nums * fpr)
        th = roc_n_list[n_idx]
        p_idx = get_dim_idx(roc_p_list, th)
        tpr = 1 - p_idx / p_nums

        roc_dict["FPR_list"].append(fpr)
        roc_dict["TPR_list"].append(tpr)
        roc_dict["TH_list"].append(th)
        roc_info += "   FPR: {}% TPR: {}% TH: {}\n".format(
                round(fpr * 100, 5),
                round(tpr * 100, 5),
                th
            )
    
    #============================== ACER =============================
    apcer = 0
    bpcer = 0
    acer = 0
    acer_th = 0
    for n_idx in tqdm(range(n_nums)):
        frr = (n_idx - 1) / n_nums
        far = get_dim_idx(roc_p_list, roc_n_list[n_idx - 1]) / p_nums
        
        apcer = far
        bpcer = frr
        acer_th = roc_n_list[n_idx]
        if bpcer >= apcer:
            break
    acer = (apcer + bpcer) / 2

    roc_dict.update({
        "APCER"    : apcer,
        "BPCER"    : bpcer,
        "ACER"     : acer,
        "ACER_TH"  : acer_th
    })
    roc_info += "   APCER: {}% BPCER: {}% ACER: {}% TH: {}\n".format(
            round(apcer * 100, 5),
            round(bpcer * 100, 5),
            round(acer * 100, 5),
            round(acer_th, 5)
        )

    return roc_info, roc_dict


def is_triangle(a, b, c):
    if a + b > c and a + c > b and b + c > a:
        return True
    else:
        return False
    
def calculate_sides(a, b, c):
    BD, CD = b, c
    if is_triangle(a, b, c):
        cosB = (a ** 2 + c ** 2 - b ** 2) / (2 * a * c)
        cosC = (a ** 2 + b ** 2 - c ** 2) / (2 * a * b)

        BD = c * cosB
        CD = b * cosC
    
    return BD, CD


def get_balance_mean(score_array):
    scores_sorted = np.sort(score_array)
    all_sum = np.sum(scores_sorted)
    all_nums = len(scores_sorted)
    idx = 0
    left_sum = 0
    for n in range(1, all_nums - 1):
        score = scores_sorted[n]
        left_sum += score
        right_sum = all_sum - left_sum
        if n * score - left_sum >= right_sum - (all_nums - n) * score:
            idx = n - 1
            break

    mean = scores_sorted[idx]
    s = (mean - np.min(score_array)) / (np.max(score_array) -mean)
    return mean, s

def get_balance_threshold(p_scores, n_scores, p_border, n_border):
    if p_border >  n_border:
        p_slice = p_scores
        n_slice = n_scores
        th_list = [(p_border - (p_border - n_border) / n) for n in range(1, 100)]
    else:
        n_slice= p_scores[p_scores < n_border]
        p_slice = n_scores[n_scores > p_border]
        th_list = np.unique(np.concatenate((p_slice, n_slice)))
    
    idx = 0
    for th in th_list:
        left_sum = 0
        right_sum = 0
        for n_score in n_slice:
            left_sum += (th - n_score)
        for p_score in p_slice:
            right_sum += (p_score - th)
        left_sum /= n_slice.size
        right_sum /= p_slice.size
        if left_sum >= right_sum:
            break
        idx += 1
    if idx == len(th_list):
        return p_border
    return th_list[idx]
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--data_val_path', type=str, required=True, default="./orig_data/", help='orig data path')
    parser.add_argument('--data_test_path', type=str, required=True, default="./orig_data/", help='orig data path')
    parser.add_argument('--model_path', type=str, required=True, default="./runs-convnextv2-SSI", help='model save path')
    parser.add_argument('--save_path', type=str, required=False, default="./result.txt", help='txt save path')
    parser.add_argument('--data_type', type=str, required=False, default="extend-resize-gauss", help='test data type')

    args = parser.parse_args()

    data_val_path = args.data_val_path
    checkPath(data_val_path)

    data_test_path = args.data_test_path
    checkPath(data_test_path)

    model_path = args.model_path
    checkPath(model_path)

    save_path = args.save_path
    clearRes(save_path) 

    model = model_load(model_path, args.data_type)
    
    load_val_list = get_img_paths(data_val_path)

    p_scores = []
    n_scores = []

    for img_path in tqdm(load_val_list):
        score = get_score(img_path, model)
        cls = img_path.replace("\\", "/").split("/")[-2]
        if cls == "1":
            p_scores.append(score)
        else:
            n_scores.append(score)
    
    roc_info, roc_dict = accuracy_detailed(n_scores, p_scores)
    print(roc_info)
    th_train = roc_dict["ACER_TH"]

    p_scores = np.array(p_scores)
    n_scores = np.array(n_scores)
    p_mean, p_s = get_balance_mean(p_scores)
    n_mean, n_s = get_balance_mean(n_scores)

    p_radius = np.max(np.abs(p_scores - p_mean))
    n_radius = np.max(np.abs(n_scores - n_mean))

    th_p_border = p_mean - p_radius
    print("P border\n   FPR: {}% TPR: 100.0% TH: {}".format(
        round(np.sum(n_scores > th_p_border) / n_scores.size * 100, 3),
        th_p_border
    ))
    th_n_border = n_mean + n_radius
    print("n border\n   FPR: 0.0% TPR: {}% TH: {}".format(
        round(np.sum(p_scores > th_n_border) / p_scores.size * 100, 3),
        th_n_border
    ))
    
    bd, cd = calculate_sides(p_mean - n_mean, p_radius, n_radius)
    th_cross = ((n_mean + bd) + (p_mean - cd)) / 2
    print("cross\n   FPR: {}% TPR: {}% TH: {}".format(
        round(np.sum(n_scores > th_cross) / n_scores.size * 100, 3),
        round(np.sum(p_scores > th_cross) / p_scores.size * 100, 3),
        th_cross
    ))

    th_balance = get_balance_threshold(p_scores, n_scores, p_mean - p_radius, n_mean + n_radius)
    print("balance\n   FPR: {}% TPR: {}% TH: {}".format(
        round(np.sum(n_scores > th_balance) / n_scores.size * 100, 3),
        round(np.sum(p_scores >= th_balance) / p_scores.size * 100, 3),
        th_balance
    ))

    with open(save_path, "w") as f:
        load_test_list = get_img_paths(data_test_path)
        load_test_list.sort()
        for img_path in tqdm(load_test_list):
            score = get_score(img_path, model)
            score = 1 if score > th_balance else 0
            f.write("{} {}\n".format(img_path, score))
        
