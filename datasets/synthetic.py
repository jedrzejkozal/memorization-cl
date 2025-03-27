import matplotlib.pyplot as plt
import torch
import torch.utils.data
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch.utils.data import DataLoader
from datasets.utils.continual_benchmark import ContinualBenchmark
from sklearn.model_selection import train_test_split


class SequentialSynthetic(ContinualBenchmark):

    NAME = 'seq-synthetic'
    SETTING = 'class-il'
    N_CLASSES = 50
    N_TASKS = 10
    N_CLASSES_PER_TASK = N_CLASSES // N_TASKS
    IMG_SIZE = None

    class Quadractic(nn.Module):
        def forward(self, x):
            return x * x

    def __init__(self, args):
        super().__init__(args)
        self.d_z = 256
        d_z = self.d_z
        nonlinearities_list = [nn.Linear(d_z, d_z),
                               self.Quadractic(),
                               nn.Linear(d_z, d_z),
                               nn.ReLU(),
                               nn.Linear(d_z, d_z),
                               nn.Sigmoid(),
                               nn.Linear(d_z, d_z),
                               self.Quadractic(),
                               nn.Linear(d_z, d_z),
                               nn.Tanh(),
                               nn.Linear(d_z, d_z),
                               self.Quadractic(),
                               nn.Linear(d_z, d_z),
                               nn.ReLU(),
                               nn.Linear(d_z, d_z),
                               nn.Sigmoid(),
                               nn.Linear(d_z, d_z),
                               ]

        self.nonlinearities = nn.Sequential(*nonlinearities_list)
        self.nonlinearities = self.nonlinearities.requires_grad_(False)

        all_samples, all_labels, all_lt = self.generate_latents()
        with torch.no_grad():
            all_samples = self.nonlinearities(all_samples)

        all_samples_train, all_samples_test, all_labels_train, all_labels_test, all_lt_train, all_lt_test = train_test_split(
            all_samples, all_labels, all_lt, test_size=0.2, stratify=np.column_stack((all_labels, all_lt)), random_state=self.args.seed)

        self.train_samples = all_samples_train
        self.train_labels = all_labels_train
        self.train_lt = all_lt_train
        self.test_samples = all_samples_test
        self.test_labels = all_labels_test
        self.test_lt = all_lt_test

    def generate_latents(self):
        n = 5000  # number of samples

        n_fg = 100  # fine grained
        n_cg = 150  # corse grained
        n_lt = 6  # long tail

        long_tail_prob = 0.02

        all_samples = list()
        all_labels = list()
        all_lt = list()

        samples_per_class = n // SequentialSynthetic.N_CLASSES
        corse_per_class = n_cg // SequentialSynthetic.N_CLASSES

        for c in range(SequentialSynthetic.N_CLASSES):
            class_lt = list(range(250, 256))
            class_cg = list(range(100+c*corse_per_class, 100+(c+1)*corse_per_class))
            class_fg = np.random.choice(n_fg, 10, replace=False)
            class_fg += 0

            for _ in range(samples_per_class):
                created_sample = torch.zeros([self.d_z])
                selected_cg = np.random.choice(class_cg, corse_per_class-1, replace=False)
                for idx in selected_cg:
                    idx = idx.item()
                    created_sample[idx] = 1

                selected_fg = np.random.choice(class_fg, size=int(0.8 * len(class_fg)), replace=False)
                for idx in selected_fg:
                    idx = idx.item()
                    created_sample[idx] = 1

                selected_lt = np.random.choice(class_lt, size=int(0.8 * len(class_lt)), replace=False)
                used_lt = False
                for idx in selected_lt:
                    idx = idx.item()
                    if np.random.rand() < long_tail_prob:
                        created_sample[idx] = 1
                        torch.randn(self.d_z) * 10
                        used_lt = True
                all_samples.append(created_sample)
                all_labels.append(c)
                all_lt.append(used_lt)

        all_samples_latents = torch.stack(all_samples)

        random_mapping = torch.cat([torch.randn((self.d_z-n_lt, self.d_z)), torch.randn(n_lt, self.d_z)*2])
        all_samples_latents = all_samples_latents @ random_mapping

        all_labels = torch.tensor(all_labels, dtype=torch.long)
        all_lt = torch.tensor(all_lt, dtype=torch.bool)

        return all_samples_latents, all_labels, all_lt

    def get_data_loaders(self):
        train_samples, train_labels = self.get_subset(self.train_samples, self.train_labels)
        test_samples, test_labels = self.get_subset(self.test_samples, self.test_labels)
        longtail_mask, _ = self.get_subset(self.train_lt, self.train_labels)
        longtail_samples, longtail_labels = train_samples[longtail_mask], train_labels[longtail_mask]

        train_dataset = torch.utils.data.TensorDataset(train_samples, train_labels, train_samples)
        test_dataset = torch.utils.data.TensorDataset(test_samples, test_labels)
        longtail_dataset = torch.utils.data.TensorDataset(longtail_samples, longtail_labels)

        train_loader = DataLoader(train_dataset,
                                  batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)
        test_loader = DataLoader(test_dataset,
                                 batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)
        longtail_loader = DataLoader(longtail_dataset,
                                     batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)
        self.train_loader = train_loader
        self.test_loaders.append(test_loader)
        self.longtail_loaders.append(longtail_loader)

        self.i += self.N_CLASSES_PER_TASK

        return train_loader, test_loader

    def get_subset(self, all_samples, all_labels):
        mask = (all_labels >= self.i) & (all_labels < self.i + self.N_CLASSES_PER_TASK)
        task_samples = all_samples[mask]
        task_labels = all_labels[mask]
        return task_samples, task_labels

    def get_backbone(self):
        hidden_dim = 256
        model = nn.Sequential(
            nn.Linear(self.d_z, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, SequentialSynthetic.N_CLASSES)
        )
        return model

    @staticmethod
    def get_transform():
        return None

    @staticmethod
    def get_normalization_transform():
        return None

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_denormalization_transform():
        return None

    @staticmethod
    def get_scheduler(model, args) -> torch.optim.lr_scheduler:
        model.opt = torch.optim.Adam(model.net.parameters(), lr=args.lr, weight_decay=args.optim_wd)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(model.opt, [150, 180], gamma=0.1, verbose=False)
        return scheduler

    @staticmethod
    def get_batch_size() -> int:
        return 64

    @staticmethod
    def get_minibatch_size() -> int:
        return SequentialSynthetic.get_batch_size()

    @staticmethod
    def get_epochs():
        return 200
