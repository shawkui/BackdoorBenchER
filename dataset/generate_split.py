import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from math import ceil, log10

# Adjusting the system path to include the current directory at the beginning.
sys.path.insert(0, "./")

from utils.aggregate_block.dataset_and_transform_generate import dataset_and_transform_generate, get_num_classes, get_input_shape
from utils.aggregate_block.fix_random import fix_random

'''
Load data from
    {args.dataset_path}/{args.dataset}
    
Split data into 
    {args.dataset_path}/{args.dataset}_split_{int(args.split_ratio*100)}/train
    {args.dataset_path}/{args.dataset}_split_{int(args.split_ratio*100)}/test
    {args.dataset_path}/{args.dataset}_split_{int(args.split_ratio*100)}/reserved
'''


def get_padding_width(max_value):
    if max_value == 0:
        return 1
    if ceil(log10(max_value)) == log10(max_value):        
        return ceil(log10(max_value + 1))
    else:
        return ceil(log10(max_value))
    
def create_directory_structure(base_path, split_ratio, args):
    """Create necessary directories for the dataset split."""
    paths = [base_path, f"{base_path}/train", f"{base_path}/test", f"{base_path}/reserved"]
    for i in range(args.num_classes):
        paths.append(f"{base_path}/train/{args.label_format.format(i)}")
        paths.append(f"{base_path}/test/{args.label_format.format(i)}")
        paths.append(f"{base_path}/reserved/{args.label_format.format(i)}")
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)

def save_images(dataset, indices, base_path, split_type, args):
    """Save images from the dataset to the specified path with the given split type."""
    for i, idx in enumerate(indices):
        img, label = dataset[idx]
        img.save(f"{base_path}/{split_type}/{args.label_format.format(label)}/{args.file_format.format(i)}.png")


def save_image(dataset, i, idx, base_path, split_type,args):
    """Save a single image from the dataset to the specified path with the given split type."""
    img, label = dataset[idx]
    img.save(f"{base_path}/{split_type}/{args.label_format.format(label)}/{args.file_format.format(i)}.png")
    
    
def save_images_parallel(dataset, indices, base_path, split_type, args, num_workers=8):
    """Save images from the dataset to the specified path with the given split type using parallel processing."""
    print(f'Svaing {split_type} data')
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(executor.map(lambda i: save_image(dataset, i, indices[i], base_path, split_type, args), list(range(len(indices)))), total=len(indices)))


def write_indices_to_file(indices, dataset, file_path, split_type, args):
    """Write index, label, and split type to a text file."""
    with open(file_path, 'w') as f:
        for i, idx in enumerate(indices):
            _, label = dataset[idx]
            f.write(f"{i} {idx} {label} {split_type}\n")

def main():
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser.add_argument('--dataset', type=str, default='cifar10', help='which dataset to use')
    parser.add_argument('--dataset_path', type=str, default='data')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--split_ratio', type=float, default=0.05)
    parser.add_argument('--random_seed', type=int, default=0, help='random_seed')
    args = parser.parse_args()

    # Get dataset information
    args.num_classes = get_num_classes(args.dataset)
    input_shape = get_input_shape(args.dataset)
    args.input_height, args.input_width, args.input_channel = input_shape
    args.img_size = (args.input_height, args.input_width, args.input_channel)
    args.dataset_path = f"{args.dataset_path}/{args.dataset}"
    
    # Fix random seed
    fix_random(int(args.random_seed))

    # Generate datasets
    train_dataset, _, _, test_dataset, _, _ = dataset_and_transform_generate(args)

    # -1 as python start from 0
    label_format =  f"{{:0{get_padding_width(args.num_classes-1)}d}}"
    file_format =   f"{{:0{get_padding_width(len(train_dataset)-1)}d}}"
    
    args.label_format = label_format
    args.file_format = file_format

    # Define the split path
    args.dataset_path_split = f"{args.dataset_path}_split_{int(args.split_ratio*100)}_seed_{args.random_seed}"

    # Create directory structure
    create_directory_structure(args.dataset_path_split, int(args.split_ratio * 100), args)

    # Split the training dataset
    print("Splitting the data")

    # check if the split is already done
    if os.path.exists(f"{args.dataset_path_split}/train_idx.txt") and os.path.exists(f"{args.dataset_path_split}/reserved_idx.txt") and os.path.exists(f"{args.dataset_path_split}/test_idx.txt"):
        # load the indices
        train_idx = []
        with open(f"{args.dataset_path_split}/train_idx.txt", 'r') as f:
            for line in f:
                _, idx, _, _ = line.strip().split()
                train_idx.append(int(idx))
        reserved_idx = []
        with open(f"{args.dataset_path_split}/reserved_idx.txt", 'r') as f:
            for line in f:
                _, idx, _, _ = line.strip().split()
                reserved_idx.append(int(idx))
        test_idx = []
        with open(f"{args.dataset_path_split}/test_idx.txt", 'r') as f:
            for line in f:
                _, idx, _, _ = line.strip().split()
                test_idx.append(int(idx))
        print("Loading the indices")
    else:        
        class_all = [label for _, label in train_dataset]
        class_indices = [np.array([idx for idx, label in enumerate(class_all) if label == c]) for c in range(args.num_classes)]
        train_idx, reserved_idx = [], []
        for indices in class_indices:
            np.random.shuffle(indices)
            split_point = int(len(indices) * args.split_ratio)
            reserved_idx.extend(indices[:split_point].tolist())
            train_idx.extend(indices[split_point:].tolist())
    
    # We hope to keep index for all datasets, so, we rename all samples with new index
    # reserved_idx.sort()
    # train_idx.sort()

    # Save the data
    print("Saving the data")
    parallel = True
    if parallel:
        save_images_parallel(train_dataset, train_idx, args.dataset_path_split, "train", args, num_workers=8)
        save_images_parallel(train_dataset, reserved_idx, args.dataset_path_split, "reserved", args, num_workers=8)
        save_images_parallel(test_dataset, range(len(test_dataset)), args.dataset_path_split, "test", args, num_workers=8)
    else:
        save_images(train_dataset, train_idx, args.dataset_path_split, "train", args)
        save_images(train_dataset, reserved_idx, args.dataset_path_split, "reserved", args)
        save_images(test_dataset, range(len(test_dataset)), args.dataset_path_split, "test", args)


    # Write meta data and indices to files
    with open(f"{args.dataset_path_split}/meta.txt", 'w') as f:
        f.writelines([
            f"num_classes: {args.num_classes}\n",
            f"input_height: {args.input_height}\n",
            f"input_width: {args.input_width}\n",
            f"input_channel: {args.input_channel}\n",
            f"split_ratio: {args.split_ratio}\n",
            f"random_seed: {args.random_seed}\n",
            f"train_size: {len(train_idx)}\n",
            f"reserved_size: {len(reserved_idx)}\n",
            f"test_size: {len(test_dataset)}\n"
        ])

    write_indices_to_file(train_idx, train_dataset, f"{args.dataset_path_split}/train_idx.txt", "train", args)
    write_indices_to_file(reserved_idx, train_dataset, f"{args.dataset_path_split}/reserved_idx.txt", "reserved", args)
    write_indices_to_file(range(len(test_dataset)), test_dataset, f"{args.dataset_path_split}/test_idx.txt", "test", args)

    print("Data saved")

if __name__ == "__main__":
    main()