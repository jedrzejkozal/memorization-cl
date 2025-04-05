import torch
import torch.nn as nn
import numpy as np

import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR100
from tqdm import tqdm

from backbones.resnet import resnet18
from utils.conf import base_path_dataset as base_path


def main():
    memorisation_file_path = '../leave-one-out/memorisation.txt'

    filename = 'idx_score.npy'

    computed_probs = {}
    with open(memorisation_file_path, 'r') as f:
        for line in f.readlines():
            idx, prob = line.split(':')
            idx, prob = int(idx), float(prob)
            computed_probs[idx] = prob

    # idxs = np.load(filename)
    idxs = np.random.permutation(50000)
    for i in idxs:
        if i in computed_probs:
            continue
        print(f'\ncomputing index {i}')
        y_pred = train(i)
        print(y_pred)

        with open(memorisation_file_path, 'a+') as f:
            f.write(f'{i}: {y_pred}\n')


def train(exclude_idx):
    n_epochs = 50
    batch_size = 32
    device = 'cuda:0'

    net = resnet18(n_classes=100)
    net.to(device)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])

    train_dataset = CIFAR100(root=base_path() + 'CIFAR100', train=True, transform=train_transform)
    subset_indcies = list(range(len(train_dataset)))
    subset_indcies.remove(exclude_idx)
    train_subset = torch.utils.data.Subset(train_dataset, subset_indcies)

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

    net.eval()
    with torch.no_grad():
        sample, label = train_dataset[exclude_idx]
        sample = torch.unsqueeze(sample, 0)
        sample = sample.to(device)
        y_pred = net(sample)
        y_pred = torch.softmax(y_pred, dim=1)
        y_pred = y_pred[:, label].item()
    return y_pred


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
