# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

from argparse import ArgumentParser
from datasets import NAMES as DATASET_NAMES
from models import get_all_models
from backbones import get_all_backbones


def add_experiment_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the models.
    :param parser: the parser instance
    """
    parser.add_argument('--dataset', type=str, required=True, choices=DATASET_NAMES,
                        help='Which dataset to perform experiments on.')
    parser.add_argument('--num_workers', default=4, type=int, help='number of processes to use for data loading')
    parser.add_argument('--half_classes_in_first_task', action='store_true', help='use half of data for first expirience')
    parser.add_argument('--img_size', type=int, default=None)
    parser.add_argument('--additional_augmentations', action='store_true', help='use additioanl augmentations for dataset')
    parser.add_argument('--split_domains', action='store_true', help='weather to split data according to domains in DomainNet dataset')

    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())
    parser.add_argument('--backbone', default=None, type=str, help='backbone to use during training', choices=get_all_backbones())
    parser.add_argument('--model_width', type=float, default=1.0)
    parser.add_argument('--pretrained', action='store_true', help='use pretrained model')
    parser.add_argument('--norm_layer', default='batch_norm', choices=('batch_norm', 'instance_norm', 'adab2n'), help='normalization layer used in model')
    parser.add_argument('--adab2n_kappa', default=0.5, type=float, help='adab2n kappa hyperparameter')
    parser.add_argument('--adab2n_lambd', default=1.0, type=float, help='adab2n lambd hyperparameter')

    parser.add_argument('--lr', type=float, required=True,
                        help='Learning rate.')

    parser.add_argument('--optim_wd', type=float, default=0.,
                        help='optimizer weight decay.')
    parser.add_argument('--optim_mom', type=float, default=0.,
                        help='optimizer momentum.')
    parser.add_argument('--optim_nesterov', type=int, default=0,
                        help='optimizer nesterov momentum.')

    parser.add_argument('--n_epochs', type=int,
                        help='Batch size.')
    parser.add_argument('--batch_size', type=int,
                        help='Batch size.')

    parser.add_argument('--distributed', type=str, default='no', choices=['no', 'dp', 'ddp'])
    parser.add_argument('--device', type=str, default='cuda:0')


def add_management_args(parser: ArgumentParser) -> None:
    parser.add_argument('--seed', type=int, default=None,
                        help='The random seed.')
    parser.add_argument('--notes', type=str, default=None,
                        help='Notes for this run.')

    parser.add_argument('--non_verbose', default=0, choices=[0, 1], type=int, help='Make progress bars non verbose')
    parser.add_argument('--disable_log', default=0, choices=[0, 1], type=int, help='Enable csv logging')
    parser.add_argument('--logging_level', default=30, type=int, choices=[0, 10, 20, 30, 40, 50],
                        help='logging level: 0 - noset, 10 - debug, 20 - info, 30 - warning, 40 - error, 50 - critical')

    parser.add_argument('--validation', default=0, choices=[0, 1], type=int,
                        help='Test on the validation set')
    parser.add_argument('--ignore_other_metrics', action='store_true', help='disable additional metrics')
    parser.add_argument('--debug', action='store_true', help='Run only a few forward steps per epoch')

    parser.add_argument('--experiment_name', type=str, default='Default')
    parser.add_argument('--parent_run_id', default=None, type=str, help='mlflow parent run id, used for creating nested run in mlflow logger')
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--n_tasks', type=int, default=None)
    parser.add_argument('--n_repeats', type=int, default=2, help='Number of tasks that will be repeated')


def add_rehearsal_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the rehearsal-based methods
    :param parser: the parser instance
    """
    parser.add_argument('--buffer_size', type=int, required=True,
                        help='The size of the memory buffer.')
    parser.add_argument('--minibatch_size', type=int,
                        help='The batch size of the memory buffer.')
