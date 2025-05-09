# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from torch.nn import functional as F
from torch.utils.data import DataLoader


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Memorisation-aware Experience Replay for DarkExperience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--buffer_policy', choices=['min', 'max', 'middle'], required=True, help='how to choose what to put into buffer')
    parser.add_argument('--alpha', type=float, required=True, help='Penalty weight.')
    parser.add_argument('--beta', type=float, required=True, help='Penalty weight.')
    return parser


class MaerDerpp(ContinualModel):
    NAME = 'maer_derpp'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super().__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)

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

        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)

        if not self.buffer.is_empty():
            buf_inputs, _, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            loss += self.args.alpha * F.mse_loss(buf_outputs, buf_logits)

            buf_inputs, buf_labels, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            loss += self.args.beta * self.loss(buf_outputs, buf_labels)

        with torch.no_grad():
            y_pred = outputs.argmax(dim=1)
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

        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs, labels=labels[:real_batch_size], logits=outputs.data)
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
        # print('current_task_indexes len = ', len(current_task_indexes))
        # print('buffer len = ', len(self.buffer))

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

        assert len(selected_samples_idxs) <= data_size, f'selected size not excede data_size, got {len(selected_samples_idxs)} and {data_size}'

        logits = []
        tmp_loader = DataLoader(train_dataset, self.args.batch_size)
        self.net.eval()
        with torch.no_grad():
            for input, _, _, _ in tmp_loader:
                output = self.net(input.to(self.args.device))
                logits.append(output.data)
        logits = torch.cat(logits)

        # added_labels = []
        # while len(self.buffer) < self.buffer.buffer_size:
        #     dataset_idx = selected_samples_idxs[0]
        #     _, label, not_aug_img, _ = train_dataset[dataset_idx]
        #     not_aug_img = torch.unsqueeze(not_aug_img, 0)
        #     label = torch.Tensor([label.item()])
        #     logit = logits[dataset_idx].unsqueeze(0)

        #     self.buffer.add_data(examples=not_aug_img, labels=label, logits=logit)
        #     selected_samples_idxs.pop(0)
        #     added_labels.append(label.item())

        # for dataset_idx in selected_samples_idxs:
        #     _, label, not_aug_img, _ = train_dataset[dataset_idx]

        #     _, buffer_counts = torch.unique(self.buffer.labels, return_counts=True)
        #     max_label = torch.argmax(buffer_counts).item()
        #     buffer_class_idxs = torch.argwhere(self.buffer.labels == max_label).flatten()
        #     buffer_idx = buffer_class_idxs[torch.randint(len(buffer_class_idxs), (1,)).item()]

        #     self.buffer.examples[buffer_idx] = not_aug_img
        #     self.buffer.labels[buffer_idx] = label
        #     self.buffer.logits[buffer_idx] = logits[dataset_idx]
        #     added_labels.append(label.item())

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
