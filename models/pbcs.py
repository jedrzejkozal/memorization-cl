# Implementation of PBCS Zhou et al. 2022 https://proceedings.mlr.press/v162/zhou22h.html
import torch
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import math
import numpy as np
from copy import deepcopy


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
    parser.add_argument('--grad_clip', default=1.0, type=float, help='gradient clipping to avoid NaN values in loss in inner_loop training')
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


class PBCS:
    def __init__(self, inner_loop_iterations=100, num_iterations=500, batch_size=128, device="cpu", eta=0.5, grad_clip=1.0):
        self.inner_loop_iterations = inner_loop_iterations
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.eta = eta
        self.device = device
        self.grad_clip = grad_clip

    def select_coreset(self, model, dataset, K):
        self.model = deepcopy(model).to(self.device)
        self.dataset = dataset
        self.K = K
        self.s = torch.nn.Parameter(torch.ones(len(self.dataset)) * self.K / len(self.dataset)).to(self.device)
        for _ in range(self.num_iterations):
            x, y, coreset_indexs = self.sample_minibatch(self.K, self.s)
            self.m = torch.zeros(len(self.dataset), dtype=torch.float).to(self.device)
            self.m[coreset_indexs] = 1
            self.inner_loop(x, y)
            self.outer_loop(self.batch_size)
        x, y, coreset_indexs = self.sample_minibatch(self.K, self.s)
        return coreset_indexs, (x, y)

    def inner_loop(self, x, y):
        optimizer = optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)
        self.model.train()
        for _ in range(self.inner_loop_iterations):
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            output = self.model(x)
            loss = F.cross_entropy(output, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            optimizer.step()

    def outer_loop(self, batch_size):
        x, y, _ = self.sample_minibatch(batch_size)
        x, y = x.to(self.device), y.to(self.device)
        loss = F.cross_entropy(self.model(x), y)
        # print('loss = ', loss)
        is_nan = torch.stack([torch.isnan(p).any() for p in self.model.parameters()]).any()
        # print('is_nan = ', is_nan)
        s_updated = self.s - self.eta * loss * self.calculate_gradient_p()
        self.s = self.project_onto_C(s_updated)

    def project_onto_C(self, s):
        """
        Projects s onto the set {x | 0 <= x_i <= 1, sum(x_i) = K}
        """
        with torch.no_grad():

            def objective_function(v):
                return torch.sum(torch.clamp(s - v, 0.0, 1.0)) - self.K

            def solve_for_v(tol=1e-6, max_iter=1000):
                v_min = (s - 1).min().item()
                v_max = s.max().item()
                for _ in range(max_iter):
                    v_mid = (v_min + v_max) / 2.0
                    sum_val = objective_function(v_mid)
                    if abs(sum_val.item()) < tol:
                        return v_mid
                    if sum_val > 0:
                        v_min = v_mid
                    else:
                        v_max = v_mid
                return (v_min + v_max) / 2.0
            v = solve_for_v()
            return torch.clamp(s - v, 0.0, 1.0)

    def calculate_gradient_p(self, eps=1e-8):
        gradient = (self.m / (self.s + eps)) - ((1 - self.m) / (1 - self.s + eps))
        # print('gradient = ', gradient)
        return gradient

    def sample_minibatch(self, batch_size, s=None):
        if s is None:
            s = torch.ones(len(self.dataset))
        # print(s)
        selected_indices = torch.multinomial(s, batch_size, replacement=False)
        x, y, _, _ = zip(*[self.dataset[i] for i in selected_indices])
        return torch.stack(x), torch.tensor(y), selected_indices


class Pbcs(ContinualModel):
    NAME = 'pbcs'
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

        selector = PBCS(device=device, num_iterations=self.args.num_iterations, inner_loop_iterations=self.args.inner_loop_iterations, grad_clip=self.args.grad_clip)
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
