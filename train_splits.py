import pathlib
import torch
import torch.nn as nn
import numpy as np
import argparse

import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR100, CIFAR10
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split


from backbones.resnet import resnet18, resnet50
from utils.conf import base_path_dataset as base_path


def main():
    args = parse_args()
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])
    if args.dataset_name == 'cifar100':
        train_dataset = CIFAR100(root=base_path() + 'CIFAR100', train=True, download=True, transform=train_transform)
    elif args.dataset_name == 'cifar10':
        train_dataset = CIFAR10(root=base_path() + 'CIFAR10', train=True, download=True, transform=train_transform)

    if args.class_range:
        range_begin, range_end = args.class_range.split(',')
        range_begin, range_end = int(range_begin), int(range_end)

        labels = np.array(train_dataset.targets)
        mask = np.logical_and(labels >= range_begin, labels < range_end)
        indicies = np.argwhere(mask).flatten()
        labels = labels[mask]
        train_dataset = Subset(train_dataset, indicies)
    else:
        labels = train_dataset.targets

    if args.dataset_size != 1.0:
        dataset_indicies = list(range(len(labels)))
        _, selected_indicies = train_test_split(dataset_indicies, test_size=args.dataset_size, random_state=42, stratify=labels)
        train_dataset = Subset(train_dataset, selected_indicies)
        labels = np.array(labels)[selected_indicies]

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    for fold_idx, (train_idx, _) in enumerate(skf.split(np.array(list(range(len(labels)))), labels)):
        train_subset = Subset(train_dataset, train_idx)
        for repeat_idx in range(args.n_repeats):
            net = train(train_subset, args.model_name)
            torch.save(net.state_dict(), args.weights_dir / f'resnet_cifar100_fold_{fold_idx}_{repeat_idx}.pth')


def parse_args():
    parser = argparse.ArgumentParser(description='Train CIFAR100 with ResNet18')
    parser.add_argument('--dataset_name', type=str, choices=['cifar10', 'cifar100'], required=True, help='dataset used for training')
    parser.add_argument('--model_name', type=str, choices=['resnet18', 'resnet50'], default='resnet18', help='what model should be used')
    parser.add_argument('--weights_dir', type=pathlib.Path, required=True, help='path where trained weights will be stored')

    parser.add_argument('--n_folds', type=int, default=10, help='Number of folds for cross-validation')
    parser.add_argument('--n_repeats', type=int, default=5, help='The number of repeats required')
    parser.add_argument('--class_range', type=str, default=None, help='class range used for training')
    parser.add_argument('--dataset_size', type=float, default=1.0, help='fraction of data used in program')
    args = parser.parse_args()
    assert 0.0 < args.dataset_size <= 1.0, 'dataset_size should be fraction in (0.0, 1.0] interval'
    return args


def train(train_subset, model_name='resnet18'):
    n_epochs = 50
    batch_size = 32
    device = 'cuda:0'

    if model_name == 'resnet18':
        net = resnet18(n_classes=100)
    elif model_name == 'resnet50':
        net = resnet50(n_classes=100)
    else:
        raise ValueError("Invalid model_name")
    net.to(device)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=16)

    opt = torch.optim.SGD(net.parameters(), lr=0.1, weight_decay=0.0, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, [35, 45], gamma=0.1, verbose=False)

    loss_fn = nn.CrossEntropyLoss()

    net.train()

    for epoch in tqdm(range(n_epochs)):
        for inputs, labels in train_loader:
            opt.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = net(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            opt.step()

        scheduler.step()

    return net


if __name__ == '__main__':
    main()
