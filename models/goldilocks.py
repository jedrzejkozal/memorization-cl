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
        self.t = 0

    def begin_task(self, dataset):
        self.classification_matrix = np.zeros((self.args.n_epochs, len(dataset.train_loader.dataset)))

    def observe(self, inputs, labels, not_aug_inputs):
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

        # self.buffer.add_data(examples=not_aug_inputs, labels=labels[:real_batch_size])

        return loss.item()

    def end_epoch(self, dataset, epoch):
        train_loader = dataset.train_loader

        status = self.net.training
        self.net.eval()
        with torch.no_grad():
            for _, label, not_aug_input, dataset_indexes in train_loader:
                not_aug_input = not_aug_input.to(self.device)
                label = label.to(self.device)
                outputs = self.net(not_aug_input)
                y_pred = outputs.argmax(dim=1)
                for i, l, y_p in zip(dataset_indexes, label, y_pred):
                    if l == y_p:
                        self.classification_matrix[epoch, i.item()] = 1

        self.net.train(status)

    def end_task(self, dataset):
        self.t += 1
        train_dataset = dataset.train_loader.dataset

        buffer_size = len(self.buffer)
        data_size = self.buffer.buffer_size // self.t

        learning_speed = np.mean(self.classification_matrix, axis=0, keepdims=False)
        indexes = np.argsort(learning_speed)
        max_idx = int(self.args.goldilocks_s * len(learning_speed))
        indexes = indexes[:-max_idx]
        min_idx = int(self.args.goldilocks_q * len(learning_speed))
        min_idx = max(data_size, min_idx)
        indexes = indexes[:min_idx]

        if len(learning_speed) > data_size:
            indexes = np.random.choice(indexes, size=data_size, replace=False)

        added_labels = []
        for dataset_idx in indexes:
            _, label, not_aug_img, _ = train_dataset[dataset_idx]
            if len(self.buffer) < self.buffer.buffer_size:
                self.buffer.add_data(examples=torch.unsqueeze(not_aug_img, 0), labels=torch.Tensor([label.item()]))
            else:
                buffer_idx = np.random.choice(buffer_size, size=1).item()
                self.buffer.examples[buffer_idx] = not_aug_img
                self.buffer.labels[buffer_idx] = label
            added_labels.append(label.item())

        print('labels added to the buffer')
        print(np.unique(added_labels, return_counts=True))
