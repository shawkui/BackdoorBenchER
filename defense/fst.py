'''
This is the implementation of the paper "Towards Stable Backdoor Purification through Feature Shift Tuning" (https://arxiv.org/abs/2310.01875) in PyTorch.

To align with other methods, some process, (e.g, preprocessing, scheduler) may be different from the official implementation (https://github.com/AISafety-HKUST/Backdoor_Safety_Tuning).
'''


import argparse
import os,sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('../')
sys.path.append(os.getcwd())

from pprint import  pformat
import yaml
import logging
import time
import random
from defense.base import defense

from utils.aggregate_block.train_settings_generate import argparser_opt_scheduler
from utils.trainer_cls import Metric_Aggregator, PureCleanModelTrainer, general_plot_for_epoch, given_dataloader_test
from utils.choose_index import choose_index
from utils.aggregate_block.fix_random import fix_random
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.log_assist import get_git_info
from utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform, get_dataset_normalization, get_dataset_denormalization
from utils.save_load_attack import load_attack_result, save_defense_result
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2

from itertools import repeat
import torchvision.transforms as transforms

from torch import Tensor

class fst(defense):

    def __init__(self,args):
        with open(args.yaml_path, 'r') as f:
            defaults = yaml.safe_load(f)

        defaults.update({k:v for k,v in args.__dict__.items() if v is not None})

        args.__dict__ = defaults

        args.terminal_info = sys.argv

        args.num_classes = get_num_classes(args.dataset)
        args.input_height, args.input_width, args.input_channel = get_input_shape(args.dataset)
        args.img_size = (args.input_height, args.input_width, args.input_channel)
        args.dataset_path = f"{args.dataset_path}/{args.dataset}"

        self.args = args

        if 'result_file' in args.__dict__ :
            if args.result_file is not None:
                self.set_result(args.result_file)

    def add_arguments(parser):
        parser.add_argument('--device', type=str, help='cuda, cpu')
        parser.add_argument("-pm","--pin_memory", type=lambda x: str(x) in ['True', 'true', '1'], help = "dataloader pin_memory")
        parser.add_argument("-nb","--non_blocking", type=lambda x: str(x) in ['True', 'true', '1'], help = ".to(), set the non_blocking = ?")
        parser.add_argument("-pf", '--prefetch', type=lambda x: str(x) in ['True', 'true', '1'], help='use prefetch')
        parser.add_argument('--amp', type=lambda x: str(x) in ['True','true','1'])

        parser.add_argument('--checkpoint_load', type=str, help='the location of load model')
        parser.add_argument('--checkpoint_save', type=str, help='the location of checkpoint where model is saved')
        parser.add_argument('--log', type=str, help='the location of log')
        parser.add_argument("--dataset_path", type=str, help='the location of data')
        parser.add_argument('--dataset', type=str, help='mnist, cifar10, cifar100, gtrsb, tiny') 
        parser.add_argument('--result_file', type=str, help='the location of result')
    
        parser.add_argument('--epochs', type=int)
        parser.add_argument('--batch_size', type=int)
        parser.add_argument("--num_workers", type=float)
        parser.add_argument('--lr', type=float)
        parser.add_argument('--lr_scheduler', type=str, help='the scheduler of lr')
        parser.add_argument('--steplr_stepsize', type=int)
        parser.add_argument('--steplr_gamma', type=float)
        parser.add_argument('--steplr_milestones', type=list)
        parser.add_argument('--model', type=str, help='resnet18')
        
        parser.add_argument('--client_optimizer', type=int)
        parser.add_argument('--sgd_momentum', type=float)
        parser.add_argument('--wd', type=float, help='weight decay of sgd')
        parser.add_argument('--frequency_save', type=int,
                        help=' frequency_save, 0 is never')

        parser.add_argument('--random_seed', type=int, help='random seed')
        parser.add_argument('--yaml_path', type=str, default="./config/defense/fst/config.yaml", help='the path of yaml')

        
        ###### fst defense parameter ######
        # defense setting
        parser.add_argument('--ratio', type=float, help='the ratio of clean data loader')
        parser.add_argument('--index', type=str, help='index of clean data')

        # hyper params
        parser.add_argument('--optim', type=str, default='Adam', help='type of outer loop optimizer utilized')

        ## optimization setting

        parser.add_argument('--reserved_type', type=str, help='reserved data will be loaded from dataset_path/dataset/reserved_type')


    def set_result(self, result_file):
        self.args.yaml_name = os.path.basename(args.yaml_path)[:-5]
        attack_file = 'record/' + result_file
        save_path = 'record/' + result_file + f'/defense/fst/{self.args.yaml_name}/'
        if not (os.path.exists(save_path)):
            os.makedirs(save_path)
        # assert(os.path.exists(save_path))    
        self.args.save_path = save_path
        if self.args.checkpoint_save is None:
            self.args.checkpoint_save = save_path + 'checkpoint/'
            if not (os.path.exists(self.args.checkpoint_save)):
                os.makedirs(self.args.checkpoint_save) 
        if self.args.log is None:
            self.args.log = save_path + 'log/'
            if not (os.path.exists(self.args.log)):
                os.makedirs(self.args.log)  
        self.result = load_attack_result(attack_file + '/attack_result.pt')


    def set_logger(self):
        args = self.args
        logFormatter = logging.Formatter(
            fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S',
        )
        logger = logging.getLogger()

        fileHandler = logging.FileHandler(args.log + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
        fileHandler.setFormatter(logFormatter)
        logger.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        logger.addHandler(consoleHandler)

        logger.setLevel(logging.INFO)
        logging.info(pformat(args.__dict__))

        try:
            logging.info(pformat(get_git_info()))
        except:
            logging.info('Getting git info fails.')
    

    def mitigation(self):
        fix_random(self.args.random_seed)

        # initialize models
        model = generate_cls_model(self.args.model,self.args.num_classes)
        model.load_state_dict(self.result['model'])

        if "," in self.args.device:
            model = torch.nn.DataParallel(model, device_ids=[int(i) for i in self.args.device[5:].split(",")])
            self.args.device = f'cuda:{model.device_ids[0]}'
            model.to(self.args.device)
        else:
            model.to(self.args.device)

        # a. get some clean data
        logging.info("Fetch some samples from clean train dataset.")

        train_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = False)

        # clean_dataset = prepro_cls_DatasetBD_v2(self.result['clean_train'].wrapped_dataset)
        # data_all_length = len(clean_dataset)
        # ran_idx = choose_index(self.args, data_all_length)
        # log_index = self.args.log + 'index.txt'
        # np.savetxt(log_index, ran_idx, fmt='%d')

        # clean_dataset.subset(ran_idx)

        # data_set_without_tran = clean_dataset
        # data_set_o = self.result['clean_train']
        # data_set_o.wrapped_dataset = data_set_without_tran
        # data_set_o.wrap_img_transform = train_tran
        data_set_o = self.get_reserved_data(train_tran, reserved_type = self.args.reserved_type, ratios = None)
        
        data_loader = torch.utils.data.DataLoader(data_set_o, batch_size=self.args.batch_size, num_workers=self.args.num_workers, shuffle=True, pin_memory=args.pin_memory)
        trainloader = data_loader
        
        ## set testing dataset
        test_tran = get_transform(self.args.dataset, *([self.args.input_height,self.args.input_width]) , train = False)
        data_bd_testset = self.result['bd_test']
        data_bd_testset.wrap_img_transform = test_tran
        data_bd_loader = torch.utils.data.DataLoader(data_bd_testset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False, shuffle=True,pin_memory=args.pin_memory)

        data_clean_testset = self.result['clean_test']
        data_clean_testset.wrap_img_transform = test_tran
        data_clean_loader = torch.utils.data.DataLoader(data_clean_testset, batch_size=self.args.batch_size, num_workers=self.args.num_workers,drop_last=False, shuffle=True,pin_memory=args.pin_memory)

        clean_test_loss_list = []
        bd_test_loss_list = []
        ra_test_loss_list = []
        test_acc_list = []
        test_asr_list = []
        test_ra_list = []

        # b. Defense
        logging.info("=> Conducting Defence..")
        model.eval()

        clean_test_loss_avg_over_batch, \
                    bd_test_loss_avg_over_batch, \
                    ra_test_loss_avg_over_batch, \
                    test_acc, \
                    test_asr, \
                    test_ra = self.eval_step(
                        model,
                        data_clean_loader,
                        data_bd_loader,
                        args,
                    )        

        logging.info('Initial State: clean test loss: {:.4f}, bd test loss: {:.4f}, ra test loss: {:.4f}, test acc: {:.4f}, test asr: {:.4f}, test ra: {:.4f}'.format(clean_test_loss_avg_over_batch, bd_test_loss_avg_over_batch, ra_test_loss_avg_over_batch, test_acc, test_asr, test_ra))


        normalization = get_dataset_normalization(args.dataset)
        denormalization = get_dataset_denormalization(normalization)
        
        agg = Metric_Aggregator()
        old_linear_weight = model.linear.weight.data.clone().detach()
        old_linear_norm = torch.norm(old_linear_weight)
        
        std = 1.0 / np.sqrt(model.linear.weight.data.size(-1))
        model.linear.weight.data.uniform_(-std, std)
        model.linear.weight.data = model.linear.weight.data * old_linear_norm / torch.norm(model.linear.weight.data)

        optimizer, scheduler = argparser_opt_scheduler(model, self.args)
        
        for round in range(args.epochs):
            model.train()
            for images, labels, original_index, poison_indicator, original_targets in trainloader:
                images = images.to(args.device)
                labels = labels.to(args.device)
                
                logits = model(images)
                loss_clean = F.cross_entropy(logits, labels)
                
                loss_weight = torch.sum( model.linear.weight * old_linear_weight)
                
                loss = loss_clean + args.alpha * loss_weight
                print(f'loss_clean: {loss_clean}, loss_weight: {loss_weight}, loss: {loss}')
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # project linear weight back
                model.linear.weight.data = model.linear.weight.data * old_linear_norm / torch.norm(model.linear.weight.data)
            scheduler.step()
            model.eval()

            clean_test_loss_avg_over_batch, \
            bd_test_loss_avg_over_batch, \
            ra_test_loss_avg_over_batch, \
            test_acc, \
            test_asr, \
            test_ra = self.eval_step(
                model,
                data_clean_loader,
                data_bd_loader,
                args,
            )

            agg({
                "epoch": round,

                "clean_test_loss_avg_over_batch": clean_test_loss_avg_over_batch,
                "bd_test_loss_avg_over_batch": bd_test_loss_avg_over_batch,
                "ra_test_loss_avg_over_batch": ra_test_loss_avg_over_batch,
                "test_acc": test_acc,
                "test_asr": test_asr,
                "test_ra": test_ra,
            })


            clean_test_loss_list.append(clean_test_loss_avg_over_batch)
            bd_test_loss_list.append(bd_test_loss_avg_over_batch)
            ra_test_loss_list.append(ra_test_loss_avg_over_batch)
            test_acc_list.append(test_acc)
            test_asr_list.append(test_asr)
            test_ra_list.append(test_ra)

            general_plot_for_epoch(
                {
                    "Test C-Acc": test_acc_list,
                    "Test ASR": test_asr_list,
                    "Test RA": test_ra_list,
                },
                save_path=f"{args.save_path}fst_acc_like_metric_plots.png",
                ylabel="percentage",
            )

            general_plot_for_epoch(
                {
                    "Test Clean Loss": clean_test_loss_list,
                    "Test Backdoor Loss": bd_test_loss_list,
                    "Test RA Loss": ra_test_loss_list,
                },
                save_path=f"{args.save_path}fst_loss_metric_plots.png",
                ylabel="percentage",
            )

            agg.to_dataframe().to_csv(f"{args.save_path}fst_df.csv")
        agg.summary().to_csv(f"{args.save_path}fst_df_summary.csv")
        # scheduler.step()
        result = {}
        result['model'] = model
        save_defense_result(
            model_name=args.model,
            num_classes=args.num_classes,
            model=model.cpu().state_dict(),
            save_path=args.save_path,
        )
        return result

    def eval_step(
            self,
            netC,
            clean_test_dataloader,
            bd_test_dataloader,
            args,
    ):
        clean_metrics, clean_epoch_predict_list, clean_epoch_label_list = given_dataloader_test(
            netC,
            clean_test_dataloader,
            criterion=torch.nn.CrossEntropyLoss(),
            non_blocking=args.non_blocking,
            device=self.args.device,
            verbose=0,
        )
        clean_test_loss_avg_over_batch = clean_metrics['test_loss_avg_over_batch']
        test_acc = clean_metrics['test_acc']
        bd_metrics, bd_epoch_predict_list, bd_epoch_label_list = given_dataloader_test(
            netC,
            bd_test_dataloader,
            criterion=torch.nn.CrossEntropyLoss(),
            non_blocking=args.non_blocking,
            device=self.args.device,
            verbose=0,
        )
        bd_test_loss_avg_over_batch = bd_metrics['test_loss_avg_over_batch']
        test_asr = bd_metrics['test_acc']

        bd_test_dataloader.dataset.wrapped_dataset.getitem_all_switch = True  # change to return the original label instead
        ra_metrics, ra_epoch_predict_list, ra_epoch_label_list = given_dataloader_test(
            netC,
            bd_test_dataloader,
            criterion=torch.nn.CrossEntropyLoss(),
            non_blocking=args.non_blocking,
            device=self.args.device,
            verbose=0,
        )
        ra_test_loss_avg_over_batch = ra_metrics['test_loss_avg_over_batch']
        test_ra = ra_metrics['test_acc']
        bd_test_dataloader.dataset.wrapped_dataset.getitem_all_switch = False  # switch back

        return clean_test_loss_avg_over_batch, \
                bd_test_loss_avg_over_batch, \
                ra_test_loss_avg_over_batch, \
                test_acc, \
                test_asr, \
                test_ra


    def defense(self,result_file):
        self.set_result(result_file)
        self.set_logger()
        result = self.mitigation()
        return result


    def eval_attack(self, netC, net_ref, clean_test_dataloader, pert, args = None):  
        total_success = 0
        total_success_ref = 0
        total_success_common = 0
        total_success_shared = 0
        
        total_samples = 0
        for images, labels, *other_info in clean_test_dataloader:
            images = images.to(self.args.device)
            labels = labels.to(self.args.device)
            pert_image = self.get_perturbed_image(images=images, pert=pert)
            outputs = netC(pert_image)
            outputs_ref = net_ref(pert_image)
            _, predicted = torch.max(outputs.data, 1)
            _, predicted_ref = torch.max(outputs_ref.data, 1)
            total_success += (predicted != labels).sum().item()
            total_success_ref += (predicted_ref != labels).sum().item()
            total_success_common += (torch.logical_and(predicted != labels, predicted_ref != labels)).sum().item()
            total_success_shared += (torch.logical_and(predicted != labels, predicted_ref == predicted)).sum().item()
            total_samples += labels.size(0)
        
        return total_success/total_samples, total_success_ref/total_samples, total_success_common/total_samples, total_success_shared/total_samples
    
    def eval_binary(self, netC, net_ref, bd_test_dataloader, pert, args = None):  
        total_success = 0
        total_success_ref = 0
        total_success_common = 0
        total_success_shared = 0
        
        total_samples = 0
        for images, labels, *other_info in bd_test_dataloader:
            images = images.to(self.args.device)
            labels = labels.to(self.args.device)
            pert_image = self.get_perturbed_image(images=images, pert=pert)
            outputs = netC(pert_image)
            outputs_ref = net_ref(pert_image)
            _, predicted = torch.max(outputs.data, 1)
            _, predicted_ref = torch.max(outputs_ref.data, 1)
            total_success += (predicted != labels).sum().item()
            total_success_ref += (predicted_ref != labels).sum().item()
            total_success_common += (torch.logical_and(predicted != labels, predicted_ref != labels)).sum().item()
            total_success_shared += (torch.logical_and(predicted != labels, predicted_ref == predicted)).sum().item()
            total_samples += labels.size(0)
        
        return total_success/total_samples, total_success_ref/total_samples, total_success_common/total_samples, total_success_shared/total_samples

    def defense(self,result_file):
        self.set_result(result_file)
        self.set_logger()
        result = self.mitigation()
        return result
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=sys.argv[0])
    fst.add_arguments(parser)
    args = parser.parse_args()
    fst_method = fst(args)
    if "result_file" not in args.__dict__:
        args.result_file = 'defense_test_badnet'
    elif args.result_file is None:
        args.result_file = 'defense_test_badnet'
    result = fst_method.defense(args.result_file)