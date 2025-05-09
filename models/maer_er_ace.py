# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from datasets import get_dataset


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Memorisation-aware Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--buffer_policy', choices=['min', 'max', 'middle'], required=True, help='how to choose what to put into buffer')
    return parser


class MaerErACE(ContinualModel):
    NAME = 'maer_er_ace'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super().__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device, mode='balanced_reservoir')
        self.seen_so_far = torch.tensor([]).long().to(self.device)
        self.num_classes = get_dataset(args).N_TASKS * get_dataset(args).N_CLASSES_PER_TASK

        self.iteration_counter = 0
        self.trained_order = []
        self.trained_order_set = set()
        self.trained_iteration = []
        self.t = 0
        self.requires_indexes = True

    def begin_task(self, dataset):
        self.iteration_counter = 0
        self.trained_order = []
        self.trained_order_set = set()
        self.trained_iteration = []

    def observe(self, inputs, labels, not_aug_inputs, dataset_indexes):
        real_batch_size = inputs.shape[0]

        present = labels.unique()
        self.seen_so_far = torch.cat([self.seen_so_far, present]).unique()

        logits = self.net(inputs)
        with torch.no_grad():
            y_pred = logits.argmax(dim=1)
            batch_idxs = torch.argwhere(y_pred[:real_batch_size] == labels[:real_batch_size]).flatten()
            idxs = dataset_indexes[batch_idxs]
            for i in idxs:
                i = i.item()
                if i not in self.trained_order_set:
                    self.trained_order_set.add(i)
                    self.trained_order.append(i)
                    self.trained_iteration.append(self.iteration_counter)

            batch_idxs = torch.argwhere(y_pred[:real_batch_size] != labels[:real_batch_size]).flatten()
            idxs = dataset_indexes[batch_idxs]
            for i in idxs:
                i = i.item()
                if i in self.trained_order_set:
                    self.trained_order_set.remove(i)
                    index = self.trained_order.index(i)
                    self.trained_order.remove(i)
                    self.trained_iteration.pop(index)

        mask = torch.zeros_like(logits)
        mask[:, present] = 1

        self.opt.zero_grad()
        if self.seen_so_far.max() < (self.num_classes - 1):
            mask[:, self.seen_so_far.max():] = 1

        if self.t > 0:
            logits = logits.masked_fill(mask == 0, torch.finfo(logits.dtype).min)

        loss = self.loss(logits, labels)
        loss_re = torch.tensor(0.)

        if self.t > 0:
            # sample from buffer
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            loss_re = self.loss(self.net(buf_inputs), buf_labels)

        loss += loss_re

        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs, labels=labels[:real_batch_size])
        self.iteration_counter += 1

        return loss.item()

    def end_task(self, dataset):
        print()
        self.t += 1
        train_dataset = dataset.train_loader.dataset
        dataset_labels = [train_dataset[i][1] for i in self.trained_order]

        data_size = self.buffer.buffer_size // self.t
        class_data_size = data_size // dataset.N_CLASSES_PER_TASK + 1
        rest_size = data_size % dataset.N_CLASSES_PER_TASK

        task_classes = np.unique(dataset_labels).tolist()
        class_trained_order = {label: np.array(self.trained_order)[np.array(dataset_labels) == label] for label in task_classes}

        current_task_indexes = []
        for i, label in enumerate(self.buffer.labels):
            label = label.item()
            if label >= dataset.i - dataset.N_CLASSES_PER_TASK and label < dataset.i:
                current_task_indexes.append(i)

        if self.args.buffer_policy == 'max':
            selected_samples_idxs = []
            for i, label in enumerate(task_classes):
                size = class_data_size if i < rest_size else class_data_size - 1
                class_idxs = class_trained_order[label][-size:]
                selected_samples_idxs.extend(class_idxs)
        elif self.args.buffer_policy == 'middle':
            selected_samples_idxs = []
            for i, label in enumerate(task_classes):
                size = class_data_size if i < rest_size else class_data_size - 1
                class_idxs = class_trained_order[label]
                half_idx = len(class_idxs) // 2
                select_size = size // 2
                class_idxs = class_idxs[half_idx-select_size:half_idx+select_size]
                selected_samples_idxs.extend(class_idxs)
        elif self.args.buffer_policy == 'min':
            selected_samples_idxs = []
            for i, label in enumerate(task_classes):
                size = class_data_size if i < rest_size else class_data_size - 1
                class_idxs = class_trained_order[label][:size]
                selected_samples_idxs.extend(class_idxs)

        added_labels = []
        for buffer_idx, dataset_idx in zip(current_task_indexes, selected_samples_idxs):
            _, label, not_aug_img, _ = train_dataset[dataset_idx]
            self.buffer.examples[buffer_idx] = not_aug_img
            self.buffer.labels[buffer_idx] = label
            added_labels.append(label.item())

        print()
        print('labels added to the buffer')
        print(np.unique(added_labels, return_counts=True))
        print()
        print('all labels in the buffer:')
        print(torch.unique(self.buffer.labels, return_counts=True))
