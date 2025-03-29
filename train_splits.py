import torch
import torch.nn as nn
import numpy as np
import argparse

import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR100
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold


from backbones.resnet import resnet18
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
    train_dataset = CIFAR100(root=base_path() + 'CIFAR100', train=True, transform=train_transform)
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

    n_folds = 10
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_idx = args.fold_idx

    for i, (train_idx, val_idx) in enumerate(skf.split(np.array(list(range(len(labels)))), labels)):
        if i == fold_idx:
            subset_indcies = train_idx
            break

    train_subset = Subset(train_dataset, subset_indcies)
    for reapeat_num in range(args.n_repeats):
        train(train_subset, fold_idx, reapeat_num)


def parse_args():
    parser = argparse.ArgumentParser(description='Train CIFAR100 with ResNet18')
    parser.add_argument('--fold_idx', type=int, required=True, help='Fold index for cross-validation')
    parser.add_argument('--n_repeats', type=int, default=5, help='The number of repeats required')
    parser.add_argument('--class_range', type=str, default=None, help='class range used for training')
    args = parser.parse_args()
    return args


def train(train_subset, fold_idx, repeat_idx):
    n_epochs = 50
    batch_size = 32
    device = 'cuda:0'

    net = resnet18(n_classes=100)
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

    torch.save(net.state_dict(), f'resnet_cifar100_fold_{fold_idx}_{repeat_idx}.pth')


def compute_acc(model, dataloader, device):
    model.eval()
    correct, total = 0.0, 0.0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, pred = torch.max(outputs.data, 1)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]

    acc = correct / total * 100
    return acc


if __name__ == '__main__':
    main()
