import pathlib
import torch
import torch.utils.data
import torch.nn as nn
import argparse
import numpy as np

from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR100, CIFAR10

from backbones.resnet import resnet18, resnet34, resnet50, resnet101
from train_splits import get_model
from utils.conf import base_path_dataset as base_path
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm
from datasets.seq_tinyimagenet import TestTinyImagenet


def main():
    args = parse_args()

    if args.dataset_name == 'cifar100':
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761)),
        ])
        train_dataset = CIFAR100(root=base_path() + 'CIFAR100', train=True, transform=test_transform)
        n_classes = 100
    elif args.dataset_name == 'cifar10':
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2615)),
        ])
        train_dataset = CIFAR10(root=base_path() + 'CIFAR10', train=True, transform=test_transform)
        n_classes = 10
    elif args.dataset_name == 'tinyimagenet':
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4480, 0.3975),
                                 (0.2770, 0.2691, 0.2821)),
        ])
        train_dataset = TestTinyImagenet(base_path() + 'TINYIMG', train=True, download=True, transform=test_transform)
        n_classes = 200

    if args.class_range:
        range_begin, range_end = args.class_range.split(',')
        range_begin, range_end = int(range_begin), int(range_end)

        labels = np.array(train_dataset.targets)
        mask = np.logical_and(labels >= range_begin, labels < range_end)
        indicies = np.argwhere(mask).flatten()
        labels = labels[mask]
        train_dataset = Subset(train_dataset, indicies)

        n_classes = range_end
    else:
        labels = train_dataset.targets

    if args.dataset_size != 1.0:
        dataset_indicies = list(range(len(labels)))
        _, selected_indicies = train_test_split(dataset_indicies, test_size=args.dataset_size, random_state=42, stratify=labels)
        train_dataset = Subset(train_dataset, selected_indicies)
        labels = np.array(labels)[selected_indicies]

    net = get_model(args.model_name, args.model_width, n_classes)

    set_range = np.array(list(range(len(labels))))
    in_set_probs = torch.zeros([len(train_dataset)])
    out_set_probs = torch.zeros([len(train_dataset)])
    in_counts = torch.zeros([len(train_dataset)])
    out_counts = torch.zeros([len(train_dataset)])

    for repeat_idx in tqdm(range(args.n_repeats)):
        train_idx, val_idx = train_test_split(set_range, test_size=0.5, random_state=repeat_idx, stratify=labels)

        train_subset = Subset(train_dataset, train_idx)
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=False, num_workers=args.num_workers)
        val_subset = Subset(train_dataset, val_idx)
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=args.num_workers)

        net.load_state_dict(torch.load(args.weights_dir / f'resnet_cifar100_repeat_{repeat_idx}.pth'))
        net.to(args.device)
        net.eval()
        in_probs = eval(train_loader, net, args.device)
        for i, prob in zip(train_idx, in_probs):
            in_set_probs[i] += prob
            in_counts[i] += 1
        out_probs = eval(val_loader, net, args.device)
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

    np.save(args.out_filename, memorisation_scores.numpy())


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate and average memorisation scores')
    parser.add_argument('--dataset_name', type=str, choices=['cifar10', 'cifar100', 'tinyimagenet'], required=True)
    parser.add_argument('--model_name', type=str, choices=['resnet18', 'resnet34', 'resnet50', 'resnet101'], default='resnet18', help='what model should be used')
    parser.add_argument('--model_width', type=float, default=1.0, help='width multiplier of model')
    parser.add_argument('--weights_dir', type=pathlib.Path, required=True, help='path where trained weights will be stored')
    parser.add_argument('--out_filename', type=str, default='memorsation_scores.npy', help='name of the .npy file that will be saved')

    parser.add_argument('--n_repeats', type=int, default=250, help='The number of repeats required')
    parser.add_argument('--class_range', type=str, default=None, help='class range used for training')
    parser.add_argument('--dataset_size', type=float, default=1.0, help='fraction of data used in program')
    parser.add_argument('--device', type=str, default='cuda:0', help='device used for training')
    parser.add_argument('--num_workers', type=int, default=10, help='number of workers used in dataloader')
    args = parser.parse_args()
    assert 0.0 < args.dataset_size <= 1.0, 'dataset_size should be fraction in (0.0, 1.0] interval'
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
