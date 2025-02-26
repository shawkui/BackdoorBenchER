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
    
    # Create directory structure
    if not os.path.exists(args.dataset_ood_path):
        os.makedirs(args.dataset_ood_path)
        for i in range(args.num_classes):
            os.makedirs(f"{args.dataset_ood_path}/{str(i).zfill(5)}")

    # Extract data    
    idx = 0
    for i, class_i in enumerate(classes):
        class_path =  f'{cinic_directory}/train/{class_i}'
        filenames_i = glob.glob('{}/*.png'.format(class_path))
        filenames_i = [file_i for file_i in filenames_i if 'cifar10' not in file_i]
        random.shuffle(filenames_i)
        filenames_i_select = filenames_i[:args.num_samples_per_class]
        for file_i in filenames_i_select:
            shutil.copy(file_i, f'{ args.dataset_ood_path}/{str(i).zfill(5)}/{str(idx).zfill(5)}.png')
            print(f'{file_i} => {args.dataset_ood_path}/{str(i).zfill(5)}/{str(idx).zfill(5)}.png')
            idx+=1
            
    print("Done")
    

if __name__ == "__main__":
    main()