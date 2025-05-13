# Implementation of PBCS Zhou et al. 2022 https://proceedings.mlr.press/v162/zhou22h.html
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import torch.nn as nn
import numpy as np
from copy import deepcopy
from datasets.utils.additional_augmentations import CIFAR10Policy

from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Bilevel Coreset Selection via Regularization')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--buffer_policy', choices=['balanced_reservoir', 'reservoir'], default='reservoir', help='policy for selecting samples stored into buffer')

    parser.add_argument('--num_iterations', default=500, type=int, help='number of iteration to refine the coreset')
    parser.add_argument('--inner_loop_iterations', default=100, type=int, help='number of epochs used for training auxilary model')
    return parser


class SimpleCNN(nn.Module):
    def __init__(self, n_classes, larger_input=False):
        super(SimpleCNN, self).__init__()
        self.larger_input = larger_input
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        output_size = 64 * 16 * 16 if larger_input else 64 * 8 * 8
        self.fc1 = nn.Linear(output_size, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        if self.larger_input:
            x = x.view(-1, 64 * 16 * 16)
        else:
            x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class RainbowMemory:
    def __init__(self, memory_size, num_augs, device="cpu", batch_size=128):
        self.memory_size = memory_size
        self.num_classes_so_far = 0
        self.augmentations = CIFAR10Policy()  # RandAugment()
        self.num_augs = num_augs
        self.device = device
        self.classes = []
        self.batch_size = batch_size

    def select_coreset(self, model, task_dataset, current_buffer_dataset):
        new_classes = np.unique(task_dataset.targets).tolist()
        self.classes.extend([c for c in new_classes if c not in self.classes])
        combined_dataset = ConcatDataset([task_dataset, current_buffer_dataset])
        targets = np.concatenate([np.array(d.targets) for d in combined_dataset.datasets])
        combined_dataloader = DataLoader(combined_dataset, batch_size=self.batch_size, shuffle=False)
        u = self.calculate_u(model, combined_dataloader)
        k_c = int(self.memory_size / len(self.classes))
        new_buffer_x, new_buffer_y = [], []
        for c in self.classes:
            D_c_idxs = np.where(targets == c)[0]
            if len(D_c_idxs) == 0:
                continue
            D_x_idxs_sorted = sorted(D_c_idxs, key=lambda idx: u[idx])
            x_selected_c, y_selected_c = [], []

            for j in range(min(k_c, len(D_c_idxs))):
                i = D_x_idxs_sorted[int(j * len(D_c_idxs) / k_c)]
                x, y = combined_dataset[i]
                x_selected_c.append(x)
                y_selected_c.append(c)
            new_buffer_x.extend(x_selected_c)
            new_buffer_y.extend(y_selected_c)
        return TensorDataset(torch.stack(new_buffer_x), torch.tensor(new_buffer_y))

    def calculate_u(self, model, dataloader):
        dataset_len = len(dataloader.dataset)
        u_matrix = np.zeros((dataset_len, len(self.classes)))

        model.eval()
        global_idx = 0

        for x_batch, _ in dataloader:
            x_batch = x_batch.to(self.device)

            for _ in range(self.num_augs):
                aug_x = self.augmentations(x_batch)
                with torch.no_grad():
                    predictions = model(aug_x).argmax(dim=1).cpu().numpy()

                for i, pred in enumerate(predictions):
                    if global_idx + i < dataset_len:
                        u_matrix[global_idx + i, pred] += 1

            global_idx += x_batch.size(0)

        u_matrix = 1 - u_matrix / self.num_augs
        return np.max(u_matrix, axis=1)


class RainbowMemory(ContinualModel):
    NAME = 'rainbow_memory'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super().__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device, mode=args.buffer_policy)
        self.t = 0

    def observe(self, inputs, labels, not_aug_inputs):
        real_batch_size = inputs.shape[0]

        self.opt.zero_grad()
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels[:real_batch_size])

        return loss.item()

    def end_task(self, dataset):
        device = self.args.device
        num_tasks = self.args.n_tasks if self.args.n_tasks is not None else dataset.N_TASKS
        task_id = self.t
        buffer_size = self.args.buffer_size

        # X_task, y_task = [], []
        # for x, y, _, _ in dataset.train_loader:
        #     X_task.append(x)
        #     y_task.append(y)
        # X_task = torch.cat(X_task)
        # y_task = torch.cat(y_task)

        model = SimpleCNN(dataset.N_CLASSES, larger_input=dataset.NAME == 'seq-tinyimg').to(device)

        selector = RainbowMemory(self.args.buffer_size, 5, device=device)
        selected_indices, _ = selector.select_coreset(model, dataset.train_loader.dataset, K=buffer_size // (task_id + 1))

        added_labels = []

        train_dataset = dataset.train_loader.dataset

        if self.t == 0:
            for buffer_idx, dataset_idx in enumerate(selected_indices):
                _, label, not_aug_img, _ = train_dataset[dataset_idx]

                self.buffer.examples[buffer_idx] = not_aug_img
                self.buffer.labels[buffer_idx] = label
                added_labels.append(label.item())
        else:
            for dataset_idx in selected_indices:
                _, label, not_aug_img, _ = train_dataset[dataset_idx]

                buffer_classes, buffer_counts = torch.unique(self.buffer.labels, return_counts=True)
                max_idx = torch.argmax(buffer_counts).item()
                max_label = buffer_classes[max_idx]
                buffer_class_idxs = torch.argwhere(self.buffer.labels == max_label).flatten()
                buffer_idx = buffer_class_idxs[torch.randint(len(buffer_class_idxs), (1,)).item()]

                self.buffer.examples[buffer_idx] = not_aug_img
                self.buffer.labels[buffer_idx] = label
                added_labels.append(label.item())

        print()
        print('labels added to the buffer')
        u1, u2 = np.unique(added_labels, return_counts=True)
        print(u1.tolist(), u2.tolist())
        print()
        print('all labels in the buffer:')
        u1, u2 = torch.unique(self.buffer.labels, return_counts=True)
        print(u1.tolist(), u2.tolist())

        self.t += 1
