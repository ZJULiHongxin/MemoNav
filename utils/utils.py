# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from collections import namedtuple
from pathlib import Path

import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

def setup_logger(final_output_dir, rank, phase):
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_rank{}.log'.format(phase, time_str, rank)
    final_log_file = os.path.join(final_output_dir, log_file)
    head = '%(asctime)-15s %(message)s'
    # logging.basicConfig(format=head)
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger, time_str

def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = os.path.basename(os.path.dirname(cfg_name))

    final_output_dir = root_output_dir / dataset / cfg.EXP_NAME # ../output/MHP/MHP_v1

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    cfg_name = os.path.basename(cfg_name).split('.')[0]
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset /cfg.EXP_NAME / (cfg_name + '_' + time_str)

    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)


def get_optimizer(cfg, model):
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            [{'params': filter(lambda p: p.requires_grad, model.parameters()), 'initial_lr': cfg.TRAIN.LR}],
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            [{'params': filter(lambda p: p.requires_grad, model.parameters()), 'initial_lr': cfg.TRAIN.LR}],
            lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WD
        )
    elif cfg.TRAIN.OPTIMIZER.lower() == 'adamw':
        optimizer = optim.AdamW(
            [{'params': filter(lambda p: p.requires_grad, model.parameters()), 'initial_lr': cfg.TRAIN.LR}],
            lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WD
        )

    return optimizer

# https://github.com/yangjianxin1/GPT2-chitchat/issues/26
def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(current_step):
        if current_step <= num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.5, float(num_training_steps - current_step))  / float(max(1, num_training_steps - num_warmup_steps))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth.tar'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best and 'state_dict' in states:
        # torch.save(states['best_state_dict'],
        #            os.path.join(output_dir, 'model_best.pth'))
        torch.save(states['state_dict'],
        os.path.join(output_dir, 'model_best.pth.tar'))


def get_model_summary(model, input_tensors: list, item_length=26, verbose=False):
    """
    :param model:
    :param input_tensors:
    :param item_length:
    :return:
    """

    summary = []

    ModuleDetails = namedtuple(
        "Layer", ["name", "input_size", "output_size", "num_parameters", "multiply_adds"])
    hooks = []
    layer_instances = {}

    def add_hooks(module):

        def hook(module, input, output):
            class_name = str(module.__class__.__name__)

            instance_index = 1
            if class_name not in layer_instances:
                layer_instances[class_name] = instance_index
            else:
                instance_index = layer_instances[class_name] + 1
                layer_instances[class_name] = instance_index

            layer_name = class_name + "_" + str(instance_index)

            params = 0

            if class_name.find("Conv") != -1 or class_name.find("BatchNorm") != -1 or \
               class_name.find("Linear") != -1:
                for param_ in module.parameters():
                    params += param_.view(-1).size(0)

            flops = "Not Available"

            if class_name.lower().find("att") != -1:
                print("=====", class_name.lower(), hasattr(module, "weight"))
            # FLOPs computation available for Conv, Att and Linear modules
            if (class_name.find("Conv") != -1 or class_name.lower().find("att") != -1) and hasattr(module, "weight"):
                flops = (
                    torch.prod(
                        torch.LongTensor(list(module.weight.data.size()))) *
                    torch.prod(
                        torch.LongTensor(list(output.size())[2:]))).item()
            elif isinstance(module, nn.Linear):
                flops = (torch.prod(torch.LongTensor(list(output.size()))) \
                         * input[0].size(1)).item()

            if isinstance(input[0], list) or isinstance(input[0], tuple):
                input = input[0]
            if isinstance(input[0], dict):
                input = list(input[0].values())
            if isinstance(output, list) or isinstance(output, tuple):
                output = output[0]

            summary.append(
                ModuleDetails(
                    name=layer_name,
                    input_size=list(input[0].size()),
                    output_size=list(output.size()),
                    num_parameters=params,
                    multiply_adds=flops)
            )

        if not isinstance(module, nn.ModuleList) \
           and not isinstance(module, nn.Sequential) \
           and module != model:
            hooks.append(module.register_forward_hook(hook))

    model.eval()
    model.apply(add_hooks)

    space_len = item_length

    returns = model(*input_tensors)
    for hook in hooks:
        hook.remove()

    details = ''
    if verbose:
        details = "Model Summary" + \
            os.linesep + \
            "Name{}Input Size{}Output Size{}Parameters{}Multiply Adds (Flops){}".format(
                ' ' * (space_len - len("Name")),
                ' ' * (space_len - len("Input Size")),
                ' ' * (space_len - len("Output Size")),
                ' ' * (space_len - len("Parameters")),
                ' ' * (space_len - len("Multiply Adds (Flops)"))) \
                + os.linesep + '-' * space_len * 5 + os.linesep

    params_sum = 0
    flops_sum = 0
    for layer in summary:
        params_sum += layer.num_parameters
        if layer.multiply_adds != "Not Available":
            flops_sum += layer.multiply_adds
        if verbose:
            details += "{}{}{}{}{}{}{}{}{}{}".format(
                layer.name,
                ' ' * (space_len - len(layer.name)),
                layer.input_size,
                ' ' * (space_len - len(str(layer.input_size))),
                layer.output_size,
                ' ' * (space_len - len(str(layer.output_size))),
                layer.num_parameters,
                ' ' * (space_len - len(str(layer.num_parameters))),
                layer.multiply_adds,
                ' ' * (space_len - len(str(layer.multiply_adds)))) \
                + os.linesep + '-' * space_len * 5 + os.linesep

    details += os.linesep \
        + "Total Parameters: {:,}".format(params_sum) \
        + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Total Multiply Adds (For Convolution and Linear Layers only): {:,} GFLOPs".format(flops_sum/(1024**3)) \
        + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Number of Layers" + os.linesep
    for layer in layer_instances:
        details += "{} : {} layers   ".format(layer, layer_instances[layer])

    return details, flops_sum/(1024**3), returns


random_pose = torch.tensor([[[103.4235, 97.0037, 1089.6506],
[144.6203, 89.9559, 1056.7954],
[165.2254, 79.8157, 1025.5192],
[182.7628, 105.1650, 1003.4246],
[207.1768, 124.6632, 984.7589],
[86.1868, 87.4663, 999.1976],
[92.6364, 102.3894, 960.2798],
[90.4995, 113.3083, 936.2986],
[85.6754, 125.2476, 913.6019],
[70.6435, 113.3644, 1003.9622],
[137.5725, 118.0449, 979.6186],
[155.6357, 103.4445, 1005.4708],
[144.2760, 96.9013, 1019.0532],
[90.2591, 135.2774, 1015.3162],
[164.5405, 130.5495, 1003.0537],
[162.7032, 112.9808, 1025.8381],
[152.6994, 105.9358, 1037.3362],
[116.1028, 151.9360, 1029.3408],
[170.1566, 144.7706, 1018.1564],
[169.4300, 127.5041, 1034.0352],
[151.8128, 122.5810, 1044.4679],
[154.1367, 193.3621, 1089.6506],
[185.7619, 185.0684, 1056.7954],
[201.5798, 173.1357, 1025.5192],
[215.0426, 202.9662, 1003.4246],
[233.7843, 225.9111, 984.7589],
[140.9047, 182.1387, 999.1976],
[145.8558, 199.6999, 960.2798],
[144.2154, 212.5490, 936.2986],
[140.5121, 226.5990, 913.6019],
[128.9726, 212.6150, 1003.9622],
[180.3516, 218.1230, 979.6186],
[194.2181, 200.9415, 1005.4708],
[185.4977, 193.2416, 1019.0532],
[144.0309, 238.4017, 1015.3162],
[201.0540, 232.8381, 1003.0537],
[199.6436, 212.1636, 1025.8381],
[191.9640, 203.8732, 1037.3362],
[163.8701, 258.0052, 1029.3408],
[205.3653, 249.5732, 1018.1564],
[204.8075, 229.2543, 1034.0352],
[191.2834, 223.4609, 1044.4679]]])