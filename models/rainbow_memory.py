# Implementation of PBCS Zhou et al. 2022 https://proceedings.mlr.press/v162/zhou22h.html
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import torch.nn as nn
import numpy as np
from torchvision.transforms import ToPILImage

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


class RainbowMemorySelector:
    def __init__(self, memory_size, num_augs, augmentations, device="cpu", batch_size=128):
        self.memory_size = memory_size
        self.num_classes_so_far = 0
        self.augmentations = augmentations
        self.num_augs = num_augs
        self.device = device
        self.classes = []
        self.batch_size = batch_size
        self.to_pil = ToPILImage()

    def select_coreset(self, model, task_dataset, current_buffer_dataset):
        new_classes = np.unique(task_dataset.dataset.targets).tolist()
        self.classes.extend([c for c in new_classes if c not in self.classes])
        combined_dataset = ConcatDataset([task_dataset, current_buffer_dataset])
        targets = np.concatenate([np.array(d.dataset.targets) if type(d) == DatasetWrapper else np.array(d.targets.cpu()) for d in combined_dataset.datasets])
        combined_dataloader = DataLoader(combined_dataset, batch_size=self.batch_size, shuffle=False)
        u = self.calculate_u(model, combined_dataloader)
        k_c = int(self.memory_size / len(self.classes))
        rest = self.memory_size % len(self.classes)
        new_buffer_x, new_buffer_y = [], []
        for c in self.classes:
            D_c_idxs = np.where(targets == c)[0]
            if len(D_c_idxs) == 0:
                continue
            D_x_idxs_sorted = sorted(D_c_idxs, key=lambda idx: u[idx])
            x_selected_c, y_selected_c = [], []

            size = k_c + 1 if c < rest else k_c
            for j in range(min(size, len(D_c_idxs))):
                i = D_x_idxs_sorted[int(j * len(D_c_idxs) / size)]
                x, y = combined_dataset[i]
                x_selected_c.append(x)
                y_selected_c.append(y)
            new_buffer_x.extend(x_selected_c)
            new_buffer_y.extend(y_selected_c)
        # return TensorDataset(torch.stack(new_buffer_x), torch.tensor(new_buffer_y))
        return torch.stack(new_buffer_x), torch.tensor(new_buffer_y)

    def calculate_u(self, model, dataloader):
        dataset_len = len(dataloader.dataset)
        n_classes = len(self.classes)
        u_matrix = np.zeros((dataset_len, len(self.classes)))

        model.eval()
        global_idx = 0

        for x_batch, _ in dataloader:
            x_batch = x_batch.to(self.device)

            for _ in range(self.num_augs):
                # aug_x = self.augmentations(x_batch)
                aug_x = torch.stack([self.augmentations(self.to_pil(x)) for x in x_batch]).to(self.device)
                with torch.no_grad():
                    y_pred = model(aug_x)
                    y_pred = y_pred[:, :n_classes]
                    predictions = y_pred.argmax(dim=1).cpu().numpy()

                for i, pred in enumerate(predictions):
                    if global_idx + i < dataset_len:
                        u_matrix[global_idx + i, pred] += 1

            global_idx += x_batch.size(0)

        u_matrix = 1 - u_matrix / self.num_augs
        return np.max(u_matrix, axis=1)


class DatasetWrapper:
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, i):
        _, y, not_aug, _ = self.dataset[i]
        return not_aug, y

    def __len__(self):
        return len(self.dataset)


class RainbowMemory(ContinualModel):
    NAME = 'rainbow_memory'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super().__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device, mode=args.buffer_policy)
        self.t = 0
        self.selector = None

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

        if self.t == 0:
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

        if self.selector is None:
            augmentations = dataset.train_transform
            self.selector = RainbowMemorySelector(self.args.buffer_size, 5, augmentations, device=device)

        buf_inputs, buf_labels = self.buffer.get_data(len(self.buffer), transform=None)

        buffer_dataset = TensorDataset(buf_inputs.cpu(), buf_labels.cpu())
        buffer_dataset.targets = buf_labels.cpu()
        with torch.no_grad():
            x, y = self.selector.select_coreset(self.net, DatasetWrapper(dataset.train_loader.dataset), buffer_dataset)
        assert len(x) == self.args.buffer_size
        assert len(y) == self.args.buffer_size
        self.buffer.examples = x.to(self.device)
        self.buffer.labels = y.to(self.device)

        # added_labels = []

        # train_dataset = dataset.train_loader.dataset

        # if self.t == 0:
        #     for buffer_idx, dataset_idx in enumerate(selected_indices):
        #         _, label, not_aug_img, _ = train_dataset[dataset_idx]

        #         self.buffer.examples[buffer_idx] = not_aug_img
        #         self.buffer.labels[buffer_idx] = label
        #         added_labels.append(label.item())
        # else:
        #     for dataset_idx in selected_indices:
        #         _, label, not_aug_img, _ = train_dataset[dataset_idx]

        #         buffer_classes, buffer_counts = torch.unique(self.buffer.labels, return_counts=True)
        #         max_idx = torch.argmax(buffer_counts).item()
        #         max_label = buffer_classes[max_idx]
        #         buffer_class_idxs = torch.argwhere(self.buffer.labels == max_label).flatten()
        #         buffer_idx = buffer_class_idxs[torch.randint(len(buffer_class_idxs), (1,)).item()]

        #         self.buffer.examples[buffer_idx] = not_aug_img
        #         self.buffer.labels[buffer_idx] = label
        #         added_labels.append(label.item())

        # print()
        # print('labels added to the buffer')
        # u1, u2 = np.unique(added_labels, return_counts=True)
        # print(u1.tolist(), u2.tolist())
        print()
        print('all labels in the buffer:')
        u1, u2 = torch.unique(self.buffer.labels, return_counts=True)
        print(u1.tolist(), u2.tolist())

        self.t += 1
