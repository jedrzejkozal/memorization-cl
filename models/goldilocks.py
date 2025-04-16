# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.utils.data
import numpy as np
import torch
import torch.utils

from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Memorisation-aware Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)

    parser.add_argument('--goldilocks_q', type=float, default=0.4)
    parser.add_argument('--goldilocks_s', type=float, default=0.6)

    return parser


class Goldilocks(ContinualModel):
    NAME = 'goldilocks'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super().__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)

    def begin_task(self, dataset):
        self.classification_matrix = np.zeros((self.args.n_epochs, dataset.N_CLASSES_PER_TASK))

    def observe(self, inputs, labels, not_aug_inputs, dataset_indexes):
        real_batch_size = inputs.shape[0]

        self.opt.zero_grad()
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs, labels=labels[:real_batch_size])
        self.iteration_counter += 1

        return loss.item()

    def end_epoch(self, dataset, epoch):
        train_dataset = dataset.train_loader.dataset
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)

        status = self.net.training
        self.net.eval()
        with torch.no_grad():
            for i, (_, label, not_aug_input) in enumerate(train_loader):
                not_aug_input = not_aug_input.to(self.device)
                label = label.to(self.device)
                outputs = self.net(not_aug_input)
                y_pred = outputs.argmax(dim=1)
                for l, y_p in zip(label, y_pred):
                    if l == y_p:
                        self.classification_matrix[epoch, i] = 1

        self.net.train(status)

    def end_task(self, dataset):
        train_dataset = dataset.train_loader.dataset

        current_task_indexes = []
        for i, label in enumerate(self.buffer.labels):
            label = label.item()
            if label >= dataset.i - dataset.N_CLASSES_PER_TASK and label < dataset.i:
                current_task_indexes.append(i)

        learning_speed = np.mean(self.classification_matrix, axis=0)
        max_idx = int(self.args.goldilocks_s * len(train_dataset))
        learning_speed = learning_speed[:-max_idx]
        min_idx = int(self.args.goldilocks_q * len(train_dataset))
        min_idx = max(len(current_task_indexes), min_idx)
        learning_speed = learning_speed[:min_idx]

        selected_samples_idxs = np.arange(min_idx, min_idx + len(current_task_indexes), 1, dtype=int)
        if len(learning_speed) > len(current_task_indexes):
            selected_samples_idxs = np.random.choice(selected_samples_idxs, size=len(current_task_indexes), replace=False)

        added_labels = []
        for buffer_idx, dataset_idx in zip(current_task_indexes, selected_samples_idxs):
            _, label, not_aug_img, _ = train_dataset[dataset_idx]
            self.buffer.examples[buffer_idx] = not_aug_img
            self.buffer.labels[buffer_idx] = label
            added_labels.append(label.item())

        print('labels added to the buffer')
        print(np.unique(added_labels, return_counts=True))
