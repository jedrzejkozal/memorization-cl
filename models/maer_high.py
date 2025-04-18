# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from utils.buffer_maer import BufferMaer


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Memorisation-aware Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--buffer_policy', choices=['min', 'max', 'middle', '0.1min', '0.9max', 'randomized'], required=True, help='how to choose what to put into buffer')
    parser.add_argument('--longtail_fraction', type=float, default=0.1, help='faction of buffer used for storing long-tail')
    return parser


class MaerHigh(ContinualModel):
    NAME = 'maer_high'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super().__init__(backbone, loss, args, transform)
        self.buffer = BufferMaer(self.args.buffer_size, self.device, mode='balanced_reservoir')

        self.iteration_counter = 0
        self.trained_order = []
        self.trained_order_set = set()
        self.trained_iteration = []
        self.t = 0

    def begin_task(self, dataset):
        self.iteration_counter = 0
        self.trained_order = []
        self.trained_order_set = set()
        self.trained_iteration = []

    def observe(self, inputs, labels, not_aug_inputs, dataset_indexes):
        real_batch_size = inputs.shape[0]

        self.opt.zero_grad()
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))

        outputs = self.net(inputs)

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

        # print(len(self.trained_order))

        loss = self.loss(outputs, labels)
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
        elif self.args.buffer_policy == '0.1min':
            selected_samples_idxs = self.trained_order[int(0.1*len(self.trained_order)):][:len(current_task_indexes)]
        elif self.args.buffer_policy == '0.9max':
            selected_samples_idxs = self.trained_order[:int(0.9*len(self.trained_order))][-len(current_task_indexes):]
        elif self.args.buffer_policy == 'randomized':
            max_idx = int(0.9 * len(self.trained_order))  # remove max indicies, as they are probably wrong labels
            trained_order = self.trained_order[:max_idx]
            trained_iteration = self.trained_iteration[:max_idx]
            selection_prob = np.array(trained_iteration) / np.sum(trained_iteration)
            selected_samples_idxs = np.random.choice(trained_order, size=len(current_task_indexes), replace=False, p=selection_prob)

        # assert len(selected_samples_idxs) == len(current_task_indexes), f'should be equal got {len(selected_samples_idxs)} and {len(current_task_indexes)}'

        added_labels = []
        for buffer_idx, dataset_idx in zip(current_task_indexes, selected_samples_idxs):
            _, label, not_aug_img, _ = train_dataset[dataset_idx]
            self.buffer.examples[buffer_idx] = not_aug_img
            self.buffer.labels[buffer_idx] = label
            added_labels.append(label.item())

        # update longtail
        class_data_size = int(self.args.buffer_size * self.args.longtail_fraction / dataset.N_CLASSES)
        assert class_data_size > 0, 'the buffer size and longtail fraction should match'
        # print('class_data_size = ', class_data_size)
        # policy max
        selected_samples_idxs = []
        for i, label in enumerate(task_classes):
            size = class_data_size if i < rest_size else class_data_size - 1
            class_idxs = class_trained_order[label][-size:]
            selected_samples_idxs.extend(class_idxs)

        lt_images = []
        lt_labels = []
        for dataset_idx in selected_samples_idxs:
            _, label, not_aug_img, _ = train_dataset[dataset_idx]
            lt_images.append(not_aug_img)
            lt_labels.append(label)
        lt_images = torch.stack(lt_images).to(self.args.device)
        lt_labels = torch.Tensor(lt_labels).to(self.args.device)

        print('longtail labels')
        print(torch.unique(lt_labels, return_counts=True))

        self.buffer.add_longtail(lt_images, lt_labels)

        print()
        print('labels added to the buffer')
        print(np.unique(added_labels, return_counts=True))
        print()
        print('all labels in the buffer:')
        print(torch.unique(self.buffer.labels, return_counts=True))
