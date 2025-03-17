# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import importlib
import torch.nn as nn
import functools

from .mammoth_backbone import *
from .utils.adab2n import AdaB2N


def get_all_backbone_modules():
    return [model.split('.')[0] for model in os.listdir('backbones')
            if not model.find('__') > -1 and 'py' in model and not 'PNN' in model and not 'mammoth_backbone' in model]


names = {}
for backbone in get_all_backbone_modules():
    backbone_module = importlib.import_module('backbones.' + backbone)
    module_backbones = backbone_module.get_all_backbones()
    for backbone_name in module_backbones:
        names[backbone_name] = getattr(backbone_module, backbone_name)


def get_all_backbones():
    return list(names.keys())


def get_norm_layer(args, n_tasks, n_classes_per_task):
    if args.norm_layer == 'batch_norm':
        return nn.BatchNorm2d
    elif args.norm_layer == 'instance_norm':
        return nn.InstanceNorm2d
    elif args.norm_layer == 'adab2n':
        return functools.partial(AdaB2N,
                                 num_tasks=n_tasks,
                                 num_classes_per_task=n_classes_per_task,
                                 kappa=args.adab2n_kappa)
    else:
        raise ValueError("Invalid norm layer")


def get_backbone(backbone_name: str, n_classes: int, n_tasks: int, n_classes_per_task: int, args):
    norm_layer_fn = get_norm_layer(args, n_tasks, n_classes_per_task)
    return names[backbone_name](n_classes, args.model_width, args.pretrained, norm_layer=norm_layer_fn)
