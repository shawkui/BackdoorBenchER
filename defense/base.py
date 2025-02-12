import os,sys
import numpy as np
import torch
import os,sys
import numpy as np
from PIL import ImageFilter, Image
import collections
import logging
import copy

sys.path.append('../')
sys.path.append(os.getcwd())

from utils.choose_index import choose_by_class_flex
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2


# Image Folder dataset
from torchvision.datasets import ImageFolder
def is_valid_file(path):
    try:
        img = Image.open(path)
        img.verify()
        img.close()
        return True
    except:
        return False


# List dataset
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
        return img, label
    def __len__(self):
        return len(self.targets)



class defense(object):


    def __init__(self,):
        # TODO:yaml config log(测试两个防御方法同时使用会不会冲突)
        print(1)

    def add_arguments(parser):
        # TODO:当后续的防御方法没有复写这个方法的时候，该防御方法需要重写该方法以实现给参数的功能
        print('You need to rewrite this method for passing parameters')
    
    def set_result(self):
        # TODO:当后续的防御方法没有复写这个方法的时候，该防御方法需要重写该方法以读取攻击的结果
        print('You need to rewrite this method to load the attack result')
        
    def set_trainer(self):
        # TODO:当后续的防御方法没有复写这个方法的时候，该防御方法可以重写该方法以实现整合训练模块的功能
        print('If you want to use standard trainer module, please rewrite this method')
    
    def set_logger(self):
        # TODO:当后续的防御方法没有复写这个方法的时候，该防御方法可以重写该方法以实现存储log的功能
        print('If you want to use standard logger, please rewrite this method')

    def denoising(self):
        # TODO:当后续的防御方法没有复写这个方法的时候，就是该防御方法没有此项功能
        print('this method does not have this function')

    def mitigation(self):
        # TODO:当后续的防御方法没有复写这个方法的时候，就是该防御方法没有此项功能
        print('this method does not have this function')

    def inhibition(self):
        # TODO:当后续的防御方法没有复写这个方法的时候，就是该防御方法没有此项功能
        print('this method does not have this function')
    
    def defense(self):
        # TODO:当后续的防御方法没有复写这个方法的时候，就是该防御方法没有此项功能
        print('this method does not have this function')
    
    def detect(self):
        # TODO:当后续的防御方法没有复写这个方法的时候，就是该防御方法没有此项功能
        print('this method does not have this function')
        
    def get_reserved_data(self, train_tran, reserved_type, ratios = None):
        '''
        A standard method for fetching reserved dataset. 
        Load data from {self.args.dataset_path}/{reserved_type} and subset it.
        Note: self.args.dataset_path is overwrite in __init__ to dataset_path/dataset
        '''
        
        if ratios is None:
            # ratios allows select different ratios for different classes
            ratios = [self.args.ratio for _ in range(self.args.num_classes)]

        # Load dataset
        if reserved_type == 'train':
            # in case some dataset is not well splitted
            reserved_dataset_without_transform = self.result['clean_train'].wrapped_dataset

        elif '_vp' in reserved_type:
            images_all = np.load(f"./{self.args.dataset_path}/{reserved_type}/{self.args.result_file}/images.npy")
            labels_all = np.load(f"./{self.args.dataset_path}/{reserved_type}/{self.args.result_file}/labels.npy")
            # Note: float tensor will not be scaled by ToTensor Transform.
            tensor_images = torch.tensor(images_all).permute(0, 3, 1, 2).float()
            tensor_labels = torch.tensor(labels_all)
            reserved_dataset_without_transform = xy_iter(x = tensor_images, y = tensor_labels, transform = None)

        else:
            reserved_dataset_without_transform = ImageFolder(
                root=f"./{self.args.dataset_path}/{reserved_type}",
                is_valid_file=is_valid_file,
            )

        clean_dataset = prepro_cls_DatasetBD_v2(reserved_dataset_without_transform)
        
        # Set ratios
        # change ratios: new ratio * all_reserved = old ratio * all_train, so that the numer of selected samples are equal to reserved_type==train case
        len_train = len(self.result['clean_train'])
        len_reserved = len(clean_dataset)
        for i in range(len(ratios)):
            ratios[i] = ratios[i] * len_train/len_reserved


        # Random subset
        if self.args.index == None:
            ran_idx = choose_by_class_flex(ratios, clean_dataset)
        else:
            ran_idx = np.loadtxt(self.args.index, dtype=int)

        log_index = self.args.log + 'index.txt'
        np.savetxt(log_index, ran_idx, fmt='%d')
        clean_dataset.subset(ran_idx)

        data_set_without_tran = clean_dataset
        data_set_o = copy.deepcopy(self.result['clean_train'])
        data_set_o.wrapped_dataset = data_set_without_tran
        
        # if '_vp' in  reserved_type:
        #     train_tran = self.result["clean_test"].wrap_img_transform 
        data_set_o.wrap_img_transform = train_tran

        # checking
        y_list = []
        for img, label, *other_info in data_set_o:
            y_list.append(label)
        y_list = np.array(y_list)
        logging.info(f'Resvered dataset type <{reserved_type}>\n Sampled {len(data_set_o)} samples from dataset <{self.args.dataset_path}/{self.args.dataset}/{reserved_type}> \n Distribution (label: num_samples):\n')
        logging.info(collections.Counter(y_list))

        return data_set_o
            

