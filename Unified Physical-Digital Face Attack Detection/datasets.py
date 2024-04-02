import os
from torchvision import datasets, transforms
import torch
import random
from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform
from sampler import MultiScaleImageFolder


def build_dataset(is_train, args):
    root = os.path.join(args.data_path, 'train' if is_train else 'dev')

    if is_train: 
        transform= build_transform(is_train, args)
        dataset = datasets.ImageFolder(root, transform=transform)
    
    else:
        transform= build_transform(is_train, args)
        dataset = datasets.ImageFolder(root, transform=transform)
    nb_classes = args.nb_classes
    assert len(dataset.class_to_idx) == nb_classes
    
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    return dataset, nb_classes

def split_dataset(full_dataset, train_part, args):
    # 设置随机数种子以获得可重复的结果
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 创建一个打乱的索引列表
    indices = list(range(len(full_dataset)))
    random.shuffle(indices)

    split_point = int(train_part * len(full_dataset))

    train_indices = indices[:split_point]
    val_indices = indices[split_point:]

    # 计算训练集和验证集的大小
    train_subset = torch.utils.data.Subset(full_dataset, train_indices)
    val_subset = torch.utils.data.Subset(full_dataset, val_indices)

    return train_subset, val_subset

def build_transform(is_train, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        #This should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter if args.color_jitter > 0 else None,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        if "p2.1" in args.data_path:
            transform.transforms.insert(1,transforms.GaussianBlur(kernel_size=(5, 9)))
        return transform

  
    t = []

    t.append(
        # To maintain same ratio w.r.t. 224 images
        transforms.Resize(args.input_size, interpolation=transforms.InterpolationMode.BICUBIC))
    if "p2.1" in args.data_path:
        t.append(transforms.GaussianBlur(kernel_size=(5, 9)))            
    t.append(transforms.CenterCrop(args.input_size))


    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))


    return transforms.Compose(t)
