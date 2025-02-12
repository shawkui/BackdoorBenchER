import random
import sys, argparse, yaml
import numpy as np
from typing import  List
import collections


sys.path.append('../')

def choose_index(args, data_all_length) :
    # choose clean data according to index
    if args.index == None:
        ran_idx = random.sample(range(data_all_length),int(data_all_length*args.ratio))
    else:
        ran_idx = np.loadtxt(args.index, dtype=int)
    return ran_idx

def choose_by_class(args,bn_train_dataset):
    by_class: List[List[int]] = [[] for _ in range(args.num_classes)]
    length = len(bn_train_dataset)
    for img, label, original_index, _,_ in bn_train_dataset:
        by_class[label].append(original_index)
    ran_idx_all = []
    for class_ in range(args.num_classes):
        ran_idx = np.random.choice(by_class[class_],int(length*args.ratio/args.num_classes),replace=False)
        ran_idx_all += ran_idx.tolist()
    return ran_idx_all

def choose_by_class_flex(ratios, bn_train_dataset):
    y_list = []
    ori_idx_list = []
    for img, label, original_index, _,_ in bn_train_dataset:
        y_list.append(label)
        ori_idx_list.append(original_index)
    y_list = np.array(y_list)
    ori_idx_list = np.array(ori_idx_list)

    labels = np.unique(y_list)
    ran_idx_all = []
    for class_ in labels:
        ran_idx = np.random.choice(ori_idx_list[y_list==class_], int(ratios[class_]*np.sum(y_list==class_)),replace=False)
        ran_idx_all += ran_idx.tolist()
        if int(ratios[class_]*np.sum(y_list==class_))==0:
            print(f'Ratio for Lable {class_} is too low. No sample is selected!')
            
    
    return ran_idx_all
