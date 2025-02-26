'''
This script provides a way to generate synthetic data as auxiliary data for backdoor purification.
This script is based on the official implementation of MMGeneration, which is a powerful tool for generative models.
Specifically, this script uses the conditional generation function to generate synthetic data, supporting various datasets.
To run this script, you need to properly install MMGeneration (https://github.com/open-mmlab/mmgeneration?tab=readme-ov-file) and download the pre-trained model and configuration file (if not provided in config/mmgen folder).
'''

import os
import sys
import argparse
import torch, torchvision
import categories
from concurrent.futures import ThreadPoolExecutor
from torchvision import transforms

import numpy as np
from tqdm import tqdm

from PIL import ImageFilter, Image
from mmgen.apis import init_model, sample_unconditional_model, sample_conditional_model

# Adjusting the system path to include the current directory at the beginning.
sys.path.insert(0, "./")

from utils.aggregate_block.dataset_and_transform_generate import dataset_and_transform_generate, get_num_classes, get_input_shape
from utils.aggregate_block.fix_random import fix_random

def label_mapping_tiny():
    # step 1: Tiny Label to wnid
    with open('./dataset/tiny_wnids.txt') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    sorted_lines = sorted(lines)
    # For ImageFolder, the labels are sorted in alphabetical order by the folder name.
    label_wnid = {i: wnid for i, wnid in enumerate(sorted_lines)}
        
    # step 2: wnid to words
    with open('./dataset/words.txt') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    wnid_words = {}
    for line in lines:
        wnid, words = line.split('\t')
        wnid_words[wnid] = words
        
    # tiny label to words
    label_words = {label: wnid_words[label_wnid[label]] for label in range(200)}
    
    # step 3: words to imagenet label
    words_imagenet = {words: i for i, words in enumerate(categories.IMAGENET_CATEGORIES)}
    
    # tiny label to imagenet label, tiny label => wnid => words => imagenet label
    label_imagenet = {label: words_imagenet[label_words[label]] for label in range(200)}
    
    # write a file to save the mapping, by tiny lable, wnid, words, imagenet label
    with open('./dataset/tiny2imagenet_map.txt', 'w') as f:
        for label in range(200):
            f.write(f"{label}\t{label_wnid[label]}\t{label_words[label]}\t{label_imagenet[label]}\n")

    return label_imagenet

def main():
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser.add_argument('--dataset', type=str, default='tiny_split_5_seed_0', help='which dataset to use')
    parser.add_argument('--dataset_path', type=str, default='data')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--random_seed', type=int, default=0, help='random_seed')
    parser.add_argument('--config_file', type=str, default='./config/mmgen/biggan/biggan_ajbrock-sn_imagenet1k_128x128_b32x8_1500k.py')
    parser.add_argument('--ckpt', type=str, default='./ckpt/biggan_imagenet1k_128x128_b32x8_best_fid_iter_1232000_20211111_122548-5315b13d.pth')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='device')
    args = parser.parse_args()
    
    # Get dataset information
    args.num_classes = get_num_classes(args.dataset)
    input_shape = get_input_shape(args.dataset)
    args.input_height, args.input_width, args.input_channel = input_shape
    args.img_size = (args.input_height, args.input_width, args.input_channel)
    args.dataset_path = f"{args.dataset_path}/{args.dataset}/reserved_synthetic"

    print(f"Generating synthetic data for {args.dataset} dataset.")
    # first, build label mapping from the intended dataset to the lables of the generative model. For most cases, it's a i-to-i mapping if the dataset for training the generative model is the same as the intended dataset.
    if 'cifar10' in args.dataset:
        label_mapping = {i: i for i in range(10)}
    if 'cifar100' in args.dataset:
        label_mapping = {i: i for i in range(100)}
    if 'gtsrb' in args.dataset:
        label_mapping = {i: i for i in range(43)}
    if 'tiny' in args.dataset:
        label_mapping = label_mapping_tiny()

    config_file = args.config_file
    checkpoint_file = args.ckpt
    device = args.device
    
    model = init_model(config_file, checkpoint_file, device=device)

    save_path = args.dataset_path

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    for label, label_map in label_mapping.items():
        label_dir = os.path.join(save_path, str(label).zfill(5))
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        
        fake_imgs = sample_conditional_model(model=model, num_samples=50, num_batches=4, label=label_map)
        print(fake_imgs.shape)
        print(fake_imgs.max())
        print(fake_imgs.min())
        for idx, img in enumerate(fake_imgs):
            img = img.cpu().numpy().transpose(1, 2, 0)  # By default, the image is in the shape of (C, H, W), so we need to transpose it to (H, W, C).
            img = (img + 1) / 2 * 255  # By default, the pixel values are in the range of [-1, 1] by normalization with mean and std 127.5, so we need to convert it to [0, 255].
            img = img[:, :, ::-1] # By default, the image is in the BGR format, so we need to convert it to RGB.
            img = img.astype('uint8')
            img = Image.fromarray(img)
            if args.input_height != img.size[1] or args.input_width != img.size[0]:
                img = img.resize((args.input_width, args.input_height), Image.BICUBIC) # rescale the image to the original size
    
            img_path = os.path.join(os.path.join(label_dir), f'{idx}.png')
            img.save(img_path)

    print("Synthetic data generation and saving completed.")


if __name__ == "__main__":
    main()