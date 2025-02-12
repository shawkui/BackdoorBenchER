import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
from PIL import ImageFilter, Image
from concurrent.futures import ThreadPoolExecutor
import torch, torchvision
from torchvision import transforms
import glob
import shutil
import random
from math import ceil, log10

# Adjusting the system path to include the current directory at the beginning.
sys.path.insert(0, "./")

from utils.aggregate_block.fix_random import fix_random


'''
Load data from
    {args.dataset_path}/{args.dataset}/reserved
    
Generate OOD data into 
    {args.dataset_path}/{args.dataset}/reserved_ood
'''


def get_padding_width(max_value):
    if max_value == 0:
        return 1
    if ceil(log10(max_value)) == log10(max_value):        
        return ceil(log10(max_value + 1)) 
    else:
        return ceil(log10(max_value))


def save_images(dataset, indices, base_path, split_type, args):
    """Save images from the dataset to the specified path with the given split type."""
    for idx in indices:
        img, label = dataset[idx]
        img.save(f"{base_path}/{split_type}/{label}/{idx}.png")


def save_image(dataset, idx, base_path, split_type):
    """Save a single image from the dataset to the specified path with the given split type."""
    img, label = dataset[idx]
    img.save(f"{base_path}/{split_type}/{label}/{idx}.png")
    
    
def save_images_parallel(dataset, indices, base_path, split_type, args, num_workers=8):
    """Save images from the dataset to the specified path with the given split type using parallel processing."""
    print(f'Svaing {split_type} data')
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(executor.map(lambda idx: save_image(dataset, idx, base_path, split_type), indices), total=len(indices)))


def write_indices_to_file(indices, dataset, file_path, split_type, args):
    """Write index, label, and split type to a text file."""
    with open(file_path, 'w') as f:
        for idx in indices:
            _, label = dataset[idx]
            f.write(f"{idx} {label} {split_type}\n")

def main():
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser.add_argument('--dataset', type=str, default='cifar10_split_5_seed_0', help='which dataset to use')
    parser.add_argument('--dataset_path', type=str, default='data')
    parser.add_argument('--ood_type', type=str, default='imagenet', help='which type of OOD to use.')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--random_seed', type=int, default=0, help='random_seed')
    parser.add_argument('--num_samples_per_class', type=int, default=1000)
    args = parser.parse_args()

    # Get dataset information
    args.dataset_path = f"{args.dataset_path}/{args.dataset}"
    cinic_directory = "./data/cinic-10"

    classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    # Fix random seed
    fix_random(int(args.random_seed))

    # define the OOD dataset path
    args.dataset_ood_path = f"{args.dataset_path}/reserved_{args.ood_type}"

    label_format =  f"{{:0{get_padding_width(args.num_classes-1)}d}}"
    file_format =   f"{{:0{get_padding_width(50000-1)}d}}"

    args.label_format = label_format
    args.file_format = file_format

    
    # Create directory structure
    if not os.path.exists(args.dataset_ood_path):
        os.makedirs(args.dataset_ood_path)
        for i in range(args.num_classes):
            os.makedirs(f"{args.dataset_ood_path}/{i}")

    # Extract data    
    idx = 0
    for i, class_i in enumerate(classes):
        class_path =  f'{cinic_directory}/train/{class_i}'
        filenames_i = glob.glob('{}/*.png'.format(class_path))
        filenames_i = [file_i for file_i in filenames_i if 'cifar10' not in file_i]
        random.shuffle(filenames_i)
        filenames_i_select = filenames_i[:args.num_samples_per_class]
        for file_i in filenames_i_select:
            shutil.copy(file_i, f'{ args.dataset_ood_path}/{args.label_format.format(i)}/{args.file_format[idx]}.png')
            print(f'{file_i} => {args.dataset_ood_path}/{args.label_format.format(i)}/{args.file_format[idx]}.png')
            idx+=1
            
    print("Done")
    

if __name__ == "__main__":
    main()