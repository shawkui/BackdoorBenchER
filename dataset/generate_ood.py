import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
from PIL import ImageFilter, Image
from concurrent.futures import ThreadPoolExecutor
import torch, torchvision
from torchvision import transforms


# Adjusting the system path to include the current directory at the beginning.
sys.path.insert(0, "./")

from utils.aggregate_block.dataset_and_transform_generate import dataset_and_transform_generate, get_num_classes, get_input_shape
from utils.aggregate_block.fix_random import fix_random


'''
Load data from
    {args.dataset_path}/{args.dataset}/reserved
    
Generate OOD data into 
    {args.dataset_path}/{args.dataset}/reserved_ood
'''

def main():
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser.add_argument('--dataset', type=str, default='cifar10_split_5_seed_0', help='which dataset to use')
    parser.add_argument('--dataset_path', type=str, default='data')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--ood_type', type=str, default='brightness', help='which type of OOD to use. ood_type can be brightness | contrast | color_jitter | gaussian_blur | rotation')
    parser.add_argument('--input_height', type=int, default=32)
    parser.add_argument('--input_width', type=int, default=32)
    parser.add_argument('--input_channel', type=int, default=3)
    parser.add_argument('--random_seed', type=int, default=0, help='random_seed')
    args = parser.parse_args()

    # Get dataset information
    args.img_size = (args.input_height, args.input_width, args.input_channel)
    args.dataset_path = f"{args.dataset_path}/{args.dataset}"
    args.num_classes = get_num_classes(args.dataset)

    # Fix random seed
    fix_random(int(args.random_seed))

    # Generate datasets
    reserved_dataset = args.dataset
    if "_split_" in reserved_dataset:
        from torchvision.datasets import ImageFolder
        def is_valid_file(path):
            try:
                img = Image.open(path)
                img.verify()
                img.close()
                return True
            except:
                return False
        reserved_dataset_without_transform = ImageFolder(
            root=f"{args.dataset_path}/reserved",
            is_valid_file=is_valid_file,
        )
        class_to_idx = reserved_dataset_without_transform.class_to_idx
        labels_to_class = {class_to_idx[key]:key for key in class_to_idx.keys()}
    else:
        raise TypeError("Unknown dataset. Only split dataset is supported.")

    # define the OOD dataset path
    args.dataset_ood_path = f"{args.dataset_path}/reserved_{args.ood_type}"
    # Create directory structure
    if not os.path.exists(args.dataset_ood_path):
        os.makedirs(args.dataset_ood_path)
        for i in range(args.num_classes):
            os.makedirs(f"{args.dataset_ood_path}/{labels_to_class[i]}")

    # Apply transformations
    if args.ood_type == 'brightness':
        brightness = transforms.ColorJitter(brightness=0.5)
        transform = transforms.Compose([
            brightness,
        ])

    elif args.ood_type == 'contrast':
        contrast = transforms.ColorJitter(contrast=0.5)
        transform = transforms.Compose([
            contrast,
        ])

    elif args.ood_type == 'color_jitter':
        transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        ])

    elif args.ood_type == 'gaussian_blur':
        # Assuming you want to apply a Gaussian blur with a kernel size of 3 and a sigma between 0.1 and 2.0
        transform = transforms.Compose([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        ])
    elif args.ood_type == 'rotation':
        transform = transforms.Compose([
            transforms.RandomRotation(30),
        ])

    elif args.ood_type == 'noise':
        transform = transforms.Compose([
            transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.1),
        ])

    else:
        raise ValueError(f"Unknown OOD type: {args.ood_type}")

    for i, (img, label) in enumerate(tqdm(reserved_dataset_without_transform)):
        # path to sample i
        ori_file_path = reserved_dataset_without_transform.imgs[i][0]
        # name of sample i
        ori_file_name = ori_file_path.split('/')[-1]
        # class name of sample i， should be the same as labels_to_class{label}
        ori_class_name = ori_file_path.split('/')[-2]

        img = transforms.ToTensor()(img)
        img_transformed = transform(img)
        img_transformed = transforms.ToPILImage()(img_transformed)
        img_transformed.save(f"{args.dataset_ood_path}/{labels_to_class[label]}/{ori_file_name}")

    # write meta data and indices to files
    with open(f"{args.dataset_ood_path}/meta.txt", 'w') as f:
        f.writelines([
            f"num_classes: {args.num_classes}\n",
            f"input_height: {args.input_height}\n",
            f"input_width: {args.input_width}\n",
            f"input_channel: {args.input_channel}\n",
            f"random_seed: {args.random_seed}\n",
            f"ood_type: {args.ood_type}\n",
        ])
    print("Done")
    

if __name__ == "__main__":
    main()