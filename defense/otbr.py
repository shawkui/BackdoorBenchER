'''
This code is from the official implementation (https://github.com/linweiii/OTBR) of the paper "Fusing Pruned and Backdoored Models: Optimal Transport-based Data-free Backdoor Mitigation" AAAI 2025.

@inproceedings{lin2024fusing,
  title={Fusing Pruned and Backdoored Models: Optimal Transport-based Data-free Backdoor Mitigation},
  author={Lin, Weilin and Liu, Li and Li, Jianze and Xiong, Hui},
  journal={AAAI},
  year={2025}
}

'''


import argparse
import copy
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.append('../')
sys.path.append(os.getcwd())

import ot
from utils.aggregate_block.train_settings_generate import argparser_opt_scheduler
from utils.choose_index import choose_index
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2
from utils.save_load_attack import load_attack_result, save_defense_result
from utils.aggregate_block.dataset_and_transform_generate import get_input_shape, get_num_classes, get_transform
from utils.log_assist import get_git_info
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.aggregate_block.fix_random import fix_random
from utils.trainer_cls import Metric_Aggregator, PureCleanModelTrainer, general_plot_for_epoch
from utils.aggregate_block.train_settings_generate import argparser_criterion
from collections import OrderedDict
from torch.utils.data import DataLoader, RandomSampler, random_split
from defense.base import defense
import time
import logging
import yaml
from pprint import pformat

method_name = 'otbr'
gamma_list = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]


def pruning(net, top_num, changed_values_neuron):
    state_dict = net.state_dict()
    slct_layer = []
    for layer_name, nidx, value in changed_values_neuron[:top_num]:
        state_dict[layer_name][int(nidx)] = 0.0
        if layer_name not in slct_layer:
            slct_layer.append(layer_name)
    net.load_state_dict(state_dict)
    return slct_layer


@torch.no_grad()
def ot_fusion(args, net_bd, net_pruned, slct_layer, target_layers_all, changed_values_neuron_ori):
    device = args.device
    weight_bd, name_layer, weight_pruned = [], [], []
    start_fusion = False
    for (name1, param_bd), (name2, param_pruned) in zip(net_bd.named_parameters(), net_pruned.named_parameters()):
        assert name1 == name2
        if name1 not in target_layers_all:
            continue
        if name1 in slct_layer:
            start_fusion = True
        if start_fusion:
            weight_bd.append(param_bd.detach())
            name_layer.append(name1)
            weight_pruned.append(param_pruned.detach())
    state_dict = net_bd.state_dict()
    ot_map_prev, prob2_prev = None, None
    for i, (layer_bd, layer_pruned) in enumerate(zip(weight_bd, weight_pruned)):
        is_conv = True if len(layer_bd.shape) == 4 else False
        nwc_layer = [tmp[2]
                     for tmp in changed_values_neuron_ori if tmp[0] == name_layer[i]]

        if i == 0:
            prob2_prev = np.ones(layer_pruned.shape[1]) / layer_pruned.shape[1]
            ot_map_prev = np.diag(prob2_prev) @ np.eye(layer_pruned.shape[1])

        w1, w2 = layer_pruned, layer_bd
        w1 = w1.flatten(2) if is_conv else w1.flatten(1)

        ot_map_prev = ot_map_prev @ np.diag(1/prob2_prev)
        ot_map_prev = torch.tensor(
            ot_map_prev, device=device, dtype=torch.float32)
        w1_transformed = torch.bmm(w1.permute(2, 0, 1), ot_map_prev.unsqueeze(0).repeat(
            w1.shape[2], 1, 1)).permute(1, 2, 0) if is_conv else w1 @ ot_map_prev
        prob1, prob2 = to_probalility(w1,  prob_type='uniform'), to_probalility(
            w2, prob_type='nwc', value_list=nwc_layer)

        cost = torch.cdist(w1.flatten(1), w2.flatten(1), p=2)
        ot_map = ot.emd(prob1, prob2, cost.detach().cpu().numpy())
        ot_map_t = np.diag(1/prob2) @ ot_map.transpose()
        ot_map_t = torch.tensor(ot_map_t, device=device, dtype=torch.float32)
        w1_final = torch.matmul(ot_map_t, w1_transformed.flatten(1)).view(
            ot_map_t.shape[0], w1_transformed.shape[1], w2.shape[2], w2.shape[3])
        state_dict[name_layer[i]] = args.lambda_ * \
            w1_final + (1 - args.lambda_) * w2
        ot_map_prev, prob2_prev = ot_map, prob2
    net_bd.load_state_dict(state_dict)


def to_probalility(weight_, prob_type='uniform', value_list=None):
    if prob_type == 'uniform':
        prob = np.ones(len(weight_)) / len(weight_)
    elif prob_type == 'random':
        prob = torch.rand(weight_.shape[0])
        prob = prob / prob.sum()
        prob = prob.cpu().numpy()
    elif prob_type == 'nwc':
        prob = np.array(value_list)
        prob = prob / prob.sum()
    return prob


def get_layerName_from_type(model, layer_type):
    if layer_type == 'conv':
        instance_name = nn.Conv2d
    elif layer_type == 'bn':
        instance_name = nn.BatchNorm2d
    else:
        raise SystemError('NO valid layer_type match!')
    layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, instance_name) and 'shortcut' not in name:
            layer_names.append(name+'.weight')
    return layer_names

def read_data(file_name):
    tempt = pd.read_csv(file_name, sep='\s+', skiprows=1, header=None)
    layer = tempt.iloc[:, 1]
    idx = tempt.iloc[:, 2]
    value = tempt.iloc[:, 3]
    values = list(zip(layer, idx, value))
    return values


class otbr(defense):

    def __init__(self, args):
        with open(args.yaml_path, 'r') as f:
            defaults = yaml.safe_load(f)

        defaults.update(
            {k: v for k, v in args.__dict__.items() if v is not None})

        args.__dict__ = defaults

        args.terminal_info = sys.argv

        args.num_classes = get_num_classes(args.dataset)
        args.input_height, args.input_width, args.input_channel = get_input_shape(
            args.dataset)
        args.img_size = (args.input_height, args.input_width,
                         args.input_channel)
        args.dataset_path = f"{args.dataset_path}/{args.dataset}"

        self.args = args

        self.policyLoss = nn.CrossEntropyLoss(reduction='none')

        if 'result_file' in args.__dict__:
            if args.result_file is not None:
                self.set_result(args.result_file)

    def add_arguments(parser):
        parser.add_argument('--device', type=str, help='cuda, cpu')
        parser.add_argument("-pm", "--pin_memory", type=lambda x: str(x)
                            in ['True', 'true', '1'], help="dataloader pin_memory")
        parser.add_argument("-nb", "--non_blocking", type=lambda x: str(x)
                            in ['True', 'true', '1'], help=".to(), set the non_blocking = ?")
        parser.add_argument("-pf", '--prefetch', type=lambda x: str(x)
                            in ['True', 'true', '1'], help='use prefetch')
        parser.add_argument('--amp', type=lambda x: str(x)
                            in ['True', 'true', '1'])

        parser.add_argument('--checkpoint_load', type=str,
                            help='the location of load model')
        parser.add_argument('--checkpoint_save', type=str,
                            help='the location of checkpoint where model is saved')
        parser.add_argument('--log', type=str, help='the location of log')
        parser.add_argument("--dataset_path", type=str,
                            help='the location of data')
        parser.add_argument('--dataset', type=str,
                            help='mnist, cifar10, cifar100, gtrsb, tiny')
        parser.add_argument('--result_file', type=str,
                            help='the location of result')

        parser.add_argument('--epochs', type=int)
        parser.add_argument('--batch_size', type=int)
        parser.add_argument("--num_workers", type=float)
        parser.add_argument('--lr_un', type=float)
        parser.add_argument('--lr_scheduler', type=str,
                            help='the scheduler of lr')
        parser.add_argument('--steplr_stepsize', type=int)
        parser.add_argument('--steplr_gamma', type=float)
        parser.add_argument('--steplr_milestones', type=list)
        parser.add_argument('--model', type=str,
                            help='the network for defense')

        parser.add_argument('--client_optimizer', type=int)
        parser.add_argument('--sgd_momentum', type=float)
        parser.add_argument('--wd', type=float, help='weight decay of sgd')
        parser.add_argument('--frequency_save', type=int,
                            help=' frequency_save, 0 is never')

        parser.add_argument('--random_seed', type=int, help='random seed')
        parser.add_argument('--yaml_path', type=str,
                            default="./config/defense/otbr/config.yaml", help='the path of yaml')
        parser.add_argument('--layer_type', type=str,
                            help='the type of layer for reinitialization')

        parser.add_argument(
            '--I_', type=int, help='the number of iterative steps')
        parser.add_argument('--lambda_', type=float,
                            help='the coefficient for merge ratio')

    def set_result(self, result_file):
        attack_file = 'record/' + result_file
        save_path = 'record/' + result_file + f'/defense/{method_name}/'
        if not (os.path.exists(save_path)):
            os.makedirs(save_path)
        # assert(os.path.exists(save_path))
        self.args.save_path = save_path
        if self.args.checkpoint_save is None:
            self.args.checkpoint_save = save_path + f'checkpoint/'
            if not (os.path.exists(self.args.checkpoint_save)):
                os.makedirs(self.args.checkpoint_save)
        if self.args.log is None:
            self.args.log = save_path + 'log/'
            if not (os.path.exists(self.args.log)):
                os.makedirs(self.args.log)
        self.result = load_attack_result(attack_file + '/attack_result.pt')

    def set_trainer(self, model):
        self.trainer = PureCleanModelTrainer(
            model,
        )

    def set_logger(self):
        args = self.args
        logFormatter = logging.Formatter(
            fmt='%(asctime)s [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S',
        )
        logger = logging.getLogger()

        fileHandler = logging.FileHandler(
            args.log + '/' + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + '.log')
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

    def set_devices(self):
        self.device = self.args.device

    def train_unlearn_random(self, args, model, criterion, optimizer, data_loader):
        model.train()
        for i in tqdm(range(args.I_)):
            rand_x = torch.rand(
                [args.batch_size, 3, 32, 32], device=args.device)
            optimizer.zero_grad()
            output = model(rand_x)
            G_ = output.shape[-1]-1
            random_y = torch.randint(
                0, G_, [args.batch_size], device=args.device).squeeze()
            loss = criterion(output, random_y)
            (-loss).backward()
            optimizer.step()

    def test(self, args, model, criterion, data_loader):
        model.eval()
        total_correct = 0
        total_loss = 0.0
        with torch.no_grad():
            for i, (images, labels, *add_info) in enumerate(data_loader):
                images, labels = images.to(args.device), labels.to(args.device)
                output = model(images)
                total_loss += criterion(output, labels).item()
                pred = output.data.max(1)[1]
                total_correct += pred.eq(labels.data.view_as(pred)).sum()
        loss = total_loss / len(data_loader)
        acc = float(total_correct) / len(data_loader.dataset)
        return loss, acc

    def mitigation(self):
        self.set_devices()
        fix_random(self.args.random_seed)

        test_tran = get_transform(
            self.args.dataset, *([self.args.input_height, self.args.input_width]), train=False)
        data_bd_testset = self.result['bd_test']
        data_bd_testset.wrap_img_transform = test_tran
        data_bd_loader = torch.utils.data.DataLoader(
            data_bd_testset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, drop_last=False, pin_memory=args.pin_memory)

        data_clean_testset = self.result['clean_test']
        data_clean_testset.wrap_img_transform = test_tran
        data_clean_loader = torch.utils.data.DataLoader(
            data_clean_testset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, drop_last=False, pin_memory=args.pin_memory)

        test_dataloader_dict = {}
        test_dataloader_dict["clean_test_dataloader"] = data_clean_loader
        test_dataloader_dict["bd_test_dataloader"] = data_bd_loader

        criterion = argparser_criterion(args)

        model_ori = generate_cls_model(self.args.model, self.args.num_classes)
        model_ori.load_state_dict(self.result['model'])
        model_ori = model_ori.to(args.device)

        model = copy.deepcopy(model_ori)
        parameters_o = list(model_ori.named_parameters())

        target_layers = get_layerName_from_type(model_ori, args.layer_type)
        params_o = {'names': [n for n, v in parameters_o if n in target_layers],
                    'params': [v for n, v in parameters_o if n in target_layers]}

        _, test_acc_load = self.test(args, model, criterion, data_clean_loader)
        _, test_asr_load = self.test(args, model, criterion, data_bd_loader)
        logging.info(
            f"Test loaded model: acc_{test_acc_load}, asr_{test_asr_load}")

        logging.info("Unlearning...")
        unlearn_optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr_un, momentum=0.9)
        unlearning_time = 0
        start_time = time.time()
        self.train_unlearn_random(
            args, model, criterion, unlearn_optimizer, data_clean_loader)
        end_time = time.time()
        unlearning_time += end_time - start_time

        _, test_acc = self.test(args, model, criterion, data_clean_loader)
        _, test_asr = self.test(args, model, criterion, data_bd_loader)
        logging.info(f"Test unlearned model: acc_{test_acc}, asr_{test_asr}")
        logging.info('-'*50)

        parameters_u = list(model.named_parameters())
        params_u = {'names': [n for n, v in parameters_u if n in target_layers],
                    'params': [v for n, v in parameters_u if n in target_layers]}

        changed_values_neuron = []
        count = 0
        for layer_i in range(len(params_u['params'])):
            name_i = params_u['names'][layer_i]
            changed_params_i = params_u['params'][layer_i] - \
                params_o['params'][layer_i]
            changed_weight_i = changed_params_i.view(
                changed_params_i.shape[0], -1).abs()
            changed_neuron_i = changed_weight_i.sum(dim=-1)
            for idx in range(changed_neuron_i.size(0)):
                changed_values_neuron.append('{} \t {} \t {} \t {:.4f} \n'.format(
                    count, name_i, idx, changed_neuron_i[idx].item()))
                count += 1
        with open(os.path.join(args.checkpoint_save, f'nwc.txt'), "w") as f:
            f.write('No \t Layer_Name \t Neuron_Idx \t Score \n')
            f.writelines(changed_values_neuron)

        # ==================
        max2min = True
        changed_values_neuron_ori = read_data(
            args.checkpoint_save + f'nwc.txt')
        changed_values_neuron = sorted(
            changed_values_neuron_ori, key=lambda x: float(x[2]), reverse=max2min)

        agg = Metric_Aggregator()
        logging.info("OT-Fusion...")

        total_num = len(changed_values_neuron)

        for ratio in gamma_list:
            top_num = int(total_num*ratio)

            model_prune = copy.deepcopy(model_ori)
            slct_layer = pruning(model_prune, top_num, changed_values_neuron)
            _, test_acc_prune = self.test(
                args, model_prune, criterion, data_clean_loader)
            _, test_asr_prune = self.test(
                args, model_prune, criterion, data_bd_loader)

            model_copy = copy.deepcopy(model_ori)

            ot_fusion(args, model_copy, model_prune, slct_layer,
                      target_layers, changed_values_neuron_ori)

            _, test_acc = self.test(
                args, model_copy, criterion, data_clean_loader)
            _, test_asr = self.test(
                args, model_copy, criterion, data_bd_loader)
            logging.info(
                f"Test reinitialized model: acc_{test_acc}, asr_{test_asr}")

            agg({
                "gamma": ratio,
                "prune_num": top_num,
                "test_acc_pruned": test_acc_prune,
                "test_asr_pruned": test_asr_prune,
                "test_acc_fused": test_acc,
                "test_asr_fused": test_asr,
            })
            agg.to_dataframe().to_csv(os.path.join(args.checkpoint_save,
                                                   f'result_df_lambda{args.lambda_}_I{args.I_}.csv'))

    def defense(self, result_file):
        self.set_result(result_file)
        self.set_logger()
        result = self.mitigation()
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=sys.argv[0])
    otbr.add_arguments(parser)
    args = parser.parse_args()
    method = otbr(args)
    if "result_file" not in args.__dict__:
        args.result_file = 'defense_test_badnet'
    elif args.result_file is None:
        args.result_file = 'defense_test_badnet'
    result = method.defense(args.result_file)
