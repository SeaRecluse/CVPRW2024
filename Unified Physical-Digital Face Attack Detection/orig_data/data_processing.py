import datetime
import os
from PIL import Image
import cv2
from detect.scrfd import SCRFD
img_size = 384
from tqdm import tqdm

def imgRecover(img, bad_size=256):
    H, W, _ = img.shape
    if H == bad_size and W == bad_size:
        new_height = H + H % 9
        new_width = int(new_height / 9 * 16)
        img = cv2.resize(img, (new_width, new_height), cv2.INTER_CUBIC)
    return img

def checkCrop(img, area = 1000 * 700):
    H, W, _ = img.shape
    if H * W > area:
        return True
    
    if H % 9 == 0 and W % 16 == 0:
        if H // 9 == W // 16:
            return True
    
    return False

def imageCrop(data_dir, data_race):
    detector = SCRFD(model_file=os.path.join(os.getcwd(), "detect/onnx_10_kps.pth"))
    detector.prepare(-1)

    for folder in data_race:
        folder_path = os.path.join(data_dir, folder)

        for type_folder in os.listdir(folder_path):
            type_folder_path = os.path.join(folder_path, type_folder)
       
            # 遍历0和1文件夹
            for class_folder in os.listdir(type_folder_path):
                class_folder_path = os.path.join(type_folder_path, class_folder)

                for per_file in tqdm(sorted(os.listdir(class_folder_path))):
                    img = cv2.imread(os.path.join(class_folder_path,per_file))
                    img = imgRecover(img)

                    if checkCrop(img):
                        for _ in range(1):
                            ta = datetime.datetime.now()
                            bboxes, kpss = detector.detect(img, 0.5, input_size = (640, 640))
                            #bboxes, kpss = detector.detect(img, 0.5)
                            tb = datetime.datetime.now()
                            # print('all cost:', (tb-ta).total_seconds()*1000)
                        # print(os.path.join(class_folder_path,per_file), bboxes.shape)
                        # if kpss is not None:
                        #     print(kpss.shape)
                        if bboxes.shape[0]:
                            bbox = bboxes[0]
                            # print(bbox)
                            x1,y1,x2,y2,score = bbox.astype(int)
                            crop_img =img[y1:y2,x1:x2]

                            filename = class_folder_path+ "/detect_crop_" + per_file
                            # print('output:', filename)
                            cv2.imwrite(filename, crop_img)
                            area_x1 = int(1/4 * img.shape[1])
                            area_x2 = int(3/4 * img.shape[1])
                            area_y1 = int(1/5 * img.shape[0])
                            area_y2 = int(4/5 * img.shape[0])                   
                            img =img[area_y1:area_y2,area_x1:area_x2]
                            filename = class_folder_path+ "/area_crop_" + per_file
                            # print('output:', filename)
                            cv2.imwrite(filename, img)

def fakeFaceExtend(data_dir, data_race):

    for folder in data_race:
        folder_path = os.path.join(data_dir, folder)

        for type_folder in os.listdir(folder_path):
            if type_folder == "train":
                type_folder_path = os.path.join(folder_path, type_folder)
        
                # 遍历0和1文件夹
                for class_folder in os.listdir(type_folder_path):
                    class_folder_path = os.path.join(type_folder_path, class_folder)
                    if class_folder == "1":
                        for per_file in tqdm(sorted(os.listdir(class_folder_path))):
                            image = Image.open(os.path.join(class_folder_path,per_file))

                            for scale_factor in [2, 4, 8]:  # 缩小比例因子
                                # 等比例缩小图片
                                scaled_image = image.resize((image.width // scale_factor, image.height // scale_factor))

                                # 保存原始等比例缩小的图片
                                output_path_scaled = os.path.join(class_folder_path, f'scaled_{scale_factor}x_{per_file}')
                                # print(output_path_scaled)
                                scaled_image.save(output_path_scaled)

                                # 计算填充大小
                                padding_left = (img_size - scaled_image.width) // 2
                                padding_top = (img_size - scaled_image.height) // 2
                                padding_right = img_size - scaled_image.width - padding_left
                                padding_bottom = img_size - scaled_image.height - padding_top

                                # 在四周添加黑色边框以填充到img_size
                                padded_image = Image.new('RGB', (img_size, img_size), color=(0, 0, 0))
                                padded_image.paste(scaled_image, (padding_left, padding_top))

                                # 保存填充后的图片
                                output_path_padded = os.path.join(class_folder_path, f'padded_{scale_factor}x_{per_file}')
                                # print(output_path_padded)
                                padded_image.save(output_path_padded)

def realFaceExtend(data_dir, data_race):

    for folder in data_race:
        folder_path = os.path.join(data_dir, folder)

        for type_folder in os.listdir(folder_path):
            if type_folder == "train":
                type_folder_path = os.path.join(folder_path, type_folder)
        
                # 遍历0和1文件夹
                for class_folder in os.listdir(type_folder_path):
                    class_folder_path = os.path.join(type_folder_path, class_folder)
                    if class_folder == "0":
                        for per_file in tqdm(sorted(os.listdir(class_folder_path))):
                            image = Image.open(os.path.join(class_folder_path,per_file))

                            for scale_factor in [2]:  # 缩小比例因子
                                # 等比例缩小图片
                                scaled_image = image.resize((image.width // scale_factor, image.height // scale_factor))

                                # 保存原始等比例缩小的图片
                                output_path_scaled = os.path.join(class_folder_path, f'scaled_{scale_factor}x_{per_file}')
                                # print(output_path_scaled)
                                scaled_image.save(output_path_scaled)

                                # 计算填充大小
                                padding_left = (img_size - scaled_image.width) // 2
                                padding_top = (img_size - scaled_image.height) // 2
                                padding_right = img_size - scaled_image.width - padding_left
                                padding_bottom = img_size - scaled_image.height - padding_top

                                # 在四周添加黑色边框以填充到img_size
                                padded_image = Image.new('RGB', (img_size, img_size), color=(0, 0, 0))
                                padded_image.paste(scaled_image, (padding_left, padding_top))

                                # 保存填充后的图片
                                output_path_padded = os.path.join(class_folder_path, f'padded_{scale_factor}x_{per_file}')
                                # print(output_path_padded)
                                padded_image.save(output_path_padded)


if __name__ == '__main__':

    imageCrop(data_dir = "data", data_race =["p1", "p2.1", "p2.2"])
    realFaceExtend(data_dir = "data", data_race =[ "p2.1"])   
    fakeFaceExtend(data_dir = "data", data_race =[ "p2.2"]) 
               