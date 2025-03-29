import torch
import torch.utils.data
import torch.nn as nn
import argparse
import numpy as np

from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR100

from backbones.resnet import resnet18
from utils.conf import base_path_dataset as base_path
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm


def main():
    args = parse_args()
    device = 'cuda:0'
    net = resnet18(n_classes=100)

    # get original probs
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])
    train_dataset = CIFAR100(root=base_path() + 'CIFAR100', train=True, transform=test_transform)
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

    set_range = np.array(list(range(len(labels))))
    in_set_probs = torch.zeros([len(train_dataset)])
    out_set_probs = torch.zeros([len(train_dataset)])
    in_counts = torch.zeros([len(train_dataset)])
    out_counts = torch.zeros([len(train_dataset)])

    for fold_idx, (train_idx, val_idx) in tqdm(enumerate(skf.split(set_range, labels))):
        train_subset = Subset(train_dataset, train_idx)
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=False, num_workers=16)
        val_subset = Subset(train_dataset, val_idx)
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=10)

        for repeat_num in range(args.n_repeats):
            net.load_state_dict(torch.load(f'resnet_cifar100_fold_{fold_idx}_{repeat_num}.pth'))
            net.to(device)
            net.eval()
            in_probs = eval(train_loader, net, device)
            for i, prob in zip(train_idx, in_probs):
                in_set_probs[i] += prob
                in_counts[i] += 1
            out_probs = eval(val_loader, net, device)
            for i, prob in zip(val_idx, out_probs):
                out_set_probs[i] += prob
                out_counts[i] += 1
            # break
    print(in_set_probs)
    print(in_counts)
    print(out_set_probs)
    print(out_counts)

    in_set_probs = in_set_probs / in_counts
    out_set_probs = out_set_probs / out_counts
    memorisation_scores = in_set_probs - out_set_probs
    print('memorisation_scores:')
    print(memorisation_scores)

    np.save('memorsation_scores.npy', memorisation_scores.numpy())


def parse_args():
    parser = argparse.ArgumentParser(description='Train CIFAR100 with ResNet18')
    parser.add_argument('--n_repeats', type=int, default=5, help='The number of repeats required')
    parser.add_argument('--class_range', type=str, default=None, help='class range used for training')
    args = parser.parse_args()
    return args


def eval(loader, net, device):
    probs = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            y_pred = net.forward(inputs)
            y_pred = torch.softmax(y_pred, dim=1)
            y_pred = y_pred.to('cpu')
            for i, label in enumerate(targets):
                prob = y_pred[i, label.item()].item()
                probs.append(prob)
    # print(probs[:10])
    return probs


if __name__ == '__main__':
    main()
