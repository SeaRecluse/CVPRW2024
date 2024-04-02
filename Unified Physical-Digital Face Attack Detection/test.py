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

#device = torch.device("cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class_nums = 2
INPUT_SIZE = 384
INPUT_CHNS = 3
#===================================================================

def model_load(model_path, data_type):
    model = timm.create_model('convnextv2_base.fcmae_ft_in22k_in1k_384',pretrained=False, num_classes=class_nums)
    checkpoint = torch.load(os.path.join(model_path,data_type,"checkpoint-best.pth"), map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device = device)
    model.eval()
    return model
#===================================================================
trans_test = transforms.Compose([
            transforms.Resize(INPUT_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.2254, 0.225])
            ])

softmax = nn.Softmax(dim=1)

def as_num(x):
     y='{:.6f}'.format(x)
     return y

def getPred(img, data_type,model):
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

def writeRes(load_path, data_type, dim_txt, model, save_path, use_tta=True):
    line_list = []
    with open(os.path.join(load_path, data_type, dim_txt), "r") as f:
        line_list = f.readlines() 

    save_txt = open(save_path, "a", newline = "\n")
    for n in tqdm(range(len(line_list))):
        per_line = line_list[n].replace("\n", "")
        if per_line:
            img = cv2.imread(load_path + per_line)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            output = getPred(img, data_type, model)
            if use_tta:
                img_flip = cv2.flip(img, 1)
                output_flip = getPred(img_flip, data_type, model)
                out = as_num((output[1] + output_flip[1]) / 2)
            else:
                out = as_num(output[1])

            print(per_line + " " + out )
            save_txt.write(per_line + " " + out + "\n")
    save_txt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--data_path', type=str, required=True, default="./orig_data/", help='orig data path')
    parser.add_argument('--model_path', type=str, required=True, default="./runs-convnextv2-b", help='model save path')
    parser.add_argument('--save_path', type=str, required=True, default="./test-submit.txt", help='txt save path')
    parser.add_argument('--data_type', type=str, required=True, default="p1,p2.1,p2.2", help='test data type')

    args = parser.parse_args()

    data_path = args.data_path
    checkPath(data_path)

    model_path = args.model_path
    checkPath(model_path)

    save_path = args.save_path
    clearRes(save_path) 

    dev_txt = "dev.txt"
    test_txt = "test.txt"
    data_type_list = args.data_type.replace(" ", "").split(",")
    for data_type in data_type_list: 
        model = model_load(model_path, data_type)
        writeRes(data_path, data_type, dev_txt, model, save_path)
        writeRes(data_path, data_type, test_txt, model, save_path)

