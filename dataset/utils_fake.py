import os

import torchvision.transforms as transforms
import random
import torch
import numpy as np
class xy_iter(torch.utils.data.dataset.Dataset):
    def __init__(self,
             x,
             y,
             transform
         ):
        assert len(x) == len(y)
        self.data = x
        self.targets = y
        self.transform = transform
    def __getitem__(self, item):
        img = self.data[item]
        label = self.targets[item]
        if self.transform is not None:
            img = self.transform(img)
        return img, label, 0, 0, 0
    def __len__(self):
        return len(self.targets)

def get_cifake(args, result):        
    transforms_list = []
    transforms_list.append(transforms.Resize((args.input_height, args.input_width)))
    transforms_list.append(transforms.ToTensor())
    transforms_list.append(get_dataset_normalization(args.dataset))
    tran = transforms.Compose(transforms_list)
    
    cifake_path =  '/home/wsk/Research/BackdoorBench/data/cifake/train/FAKE'
    image_path_list = []
    image_list = []
    label_list = []
    import PIL
    for root, dirs, files in os.walk(cifake_path):
        for file in files:
            if file.endswith(".jpg"):
                # print(os.path.join(root, file))
                image_path_list.append(os.path.join(root, file))
                image_list.append(PIL.Image.open(os.path.join(root, file)))
                label = 0
                for i in range(2, 11):
                    if file.endswith(f'({i}).jpg'):
                        label = i-1
                        break
                label_list.append(label)
    # subset 2500 images from 50000 images
    sample_list = random.sample(range(len(image_list)), int(2500*args.mixed_ratio))
    image_list = [image_list[i] for i in sample_list]
    label_list = [label_list[i] for i in sample_list]
    print(f'Create dataset with \n \t Dataset: CIFAKE \n \t Number of samples: {len(image_list)}')
    
    clean_dataset = result['clean_train']
    clean_dataset.wrap_img_transform = None
    sample_list_clean = random.sample(range(len(clean_dataset)), int(2500*(1-args.mixed_ratio)))
    image_list_clean = [clean_dataset[i][0] for i in sample_list_clean]
    label_list_clean = [clean_dataset[i][1] for i in sample_list_clean]
    
    mix_image_list = image_list + image_list_clean
    mix_label_list = label_list + label_list_clean

    # create dataset
    data_set_o = xy_iter(mix_image_list, mix_label_list, tran)
    
    return data_set_o

def get_cifar5m(args, result, part0_path = '/home/wsk/Research/BackdoorBenchER/data/cifar5m/cifar5m_part0.npz'):        
    transforms_list = []
    transforms_list.append(transforms.Resize((args.input_height, args.input_width)))
    transforms_list.append(transforms.ToTensor())
    transforms_list.append(get_dataset_normalization(args.dataset))
    tran = transforms.Compose(transforms_list)

    
    image_path_list = []
    image_list = []
    label_list = []
    part0_data = np.load(part0_path)
    images = part0_data['X']
    labels = part0_data['Y']
    # sample 250 images per class
    for i in range(10):
        image_list.append(images[labels==i][:int(250*args.mixed_ratio)])
        label_list.append(labels[labels==i][:int(250*args.mixed_ratio)])
        
    image_list = np.concatenate(image_list, axis=0)
    label_list = np.concatenate(label_list, axis=0)
    # turn np.ndarray to PIL.Image
    import PIL
    image_list = [PIL.Image.fromarray(image_list[i]) for i in range(len(image_list))]
    label_list = [label_list[i] for i in range(len(label_list))]
    print(f'Create dataset with \n \t Dataset: CIFAKE \n \t Number of samples: {len(image_list)}')
    
    clean_dataset = result['clean_train']
    clean_dataset.wrap_img_transform = None
    sample_list_clean = random.sample(range(len(clean_dataset)), int(2500*(1-args.mixed_ratio)))
    image_list_clean = [clean_dataset[i][0] for i in sample_list_clean]
    label_list_clean = [clean_dataset[i][1] for i in sample_list_clean]
    
    mix_image_list = image_list + image_list_clean
    mix_label_list = label_list + label_list_clean

    # create dataset
    data_set_o = xy_iter(mix_image_list, mix_label_list, tran)
    
    return data_set_o