import pathlib
import torch
import torch.nn as nn
import numpy as np
import argparse
import multiprocessing

from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR100, CIFAR10
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split
from datasets.seq_tinyimagenet import TestTinyImagenet


from backbones.resnet import resnet18, resnet34, resnet50, resnet101
from utils.conf import base_path_dataset as base_path


def main():
    args = parse_args()
    if args.use_multiprocessing:
        torch.multiprocessing.set_start_method('spawn')
        args.num_workers = 0

    if args.dataset_name == 'cifar100':
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
        train_dataset = CIFAR100(root=base_path() + 'CIFAR100', train=True, download=True, transform=train_transform)
        test_dataset = CIFAR100(root=base_path() + 'CIFAR100', train=False, download=True, transform=test_transform)
        n_classes = 100
    elif args.dataset_name == 'cifar10':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2615)),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2615)),
        ])
        train_dataset = CIFAR10(root=base_path() + 'CIFAR10', train=True, download=True, transform=train_transform)
        test_dataset = CIFAR10(root=base_path() + 'CIFAR10', train=False, download=True, transform=test_transform)
        n_classes = 10
    elif args.dataset_name == 'tinyimagenet':
        train_transform = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4480, 0.3975),
                                 (0.2770, 0.2691, 0.2821)),
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4480, 0.3975),
                                 (0.2770, 0.2691, 0.2821)),
        ])
        train_dataset = TestTinyImagenet(base_path() + 'TINYIMG', train=True, download=True, transform=train_transform)
        test_dataset = TestTinyImagenet(base_path() + 'TINYIMG', train=False, download=True, transform=test_transform)
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

    workload = list(range(args.n_repeats))
    if args.use_multiprocessing:
        n_gpus = torch.cuda.device_count()
        chunk_size = len(workload)//n_gpus
        args_list = []
        for worker_id in range(n_gpus):
            workload_chunk = workload[worker_id*chunk_size:(worker_id+1)*chunk_size]
            if worker_id == n_gpus-1:
                workload_chunk = workload[worker_id*chunk_size:]
            args_list.append((args, f'cuda:{worker_id}', train_dataset, test_dataset, n_classes, labels, workload_chunk))
        with multiprocessing.Pool(processes=n_gpus) as pool:
            pool.starmap(train_networks, args_list)
    else:
        train_networks(args, args.device, train_dataset, test_dataset, n_classes, labels, workload)


def parse_args():
    parser = argparse.ArgumentParser(description='Train CIFAR100 with ResNet18')
    parser.add_argument('--dataset_name', type=str, choices=['cifar10', 'cifar100', 'tinyimagenet'], required=True, help='dataset used for training')
    parser.add_argument('--model_name', type=str, choices=['resnet18', 'resnet34', 'resnet50', 'resnet101'], default='resnet18', help='what model should be used')
    parser.add_argument('--model_width', type=float, default=1.0, help='width multiplier of model')
    parser.add_argument('--weights_dir', type=pathlib.Path, required=True, help='path where trained weights will be stored')

    parser.add_argument('--n_repeats', type=int, default=250, help='The number of repeats required')
    parser.add_argument('--class_range', type=str, default=None, help='class range used for training')
    parser.add_argument('--dataset_size', type=float, default=1.0, help='fraction of data used in program')

    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight decay')
    parser.add_argument('--device', type=str, default='cuda:0', help='device used for training')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers used in dataloader')
    parser.add_argument('--use_multiprocessing', action='store_true', help='use mutliprocessing to split the networks training across all avaliable machines')
    args = parser.parse_args()
    assert 0.0 < args.dataset_size <= 1.0, 'dataset_size should be fraction in (0.0, 1.0] interval'
    return args


def train_networks(args, device, train_dataset, test_dataset, n_classes, labels, workload):
    for repeat_idx in tqdm(workload):
        dataset_indicies = np.array(list(range(len(labels))))
        train_idx, _ = train_test_split(dataset_indicies, test_size=0.5, random_state=repeat_idx, stratify=labels)
        train_subset = Subset(train_dataset, train_idx)
        net = train(train_subset, test_dataset, args.model_name, args.model_width, device, args.num_workers, n_classes,
                    args.batch_size, args.lr, args.weight_decay, args.n_epochs)
        # test_acc = eval_model(net, test_dataset, device=device)
        # print('test acc = ', test_acc)
        # exit()
        torch.save(net.state_dict(), args.weights_dir / f'resnet_cifar100_repeat_{repeat_idx}.pth')


def train(train_subset, test_dataset, model_name='resnet18', model_width=1.0, device='cuda:0',
          num_workers=10, n_classes=100, batch_size=32, lr=0.1, weight_decay=1e-6, n_epochs=50):
    net = get_model(model_name, model_width, n_classes)
    net.to(device)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    opt = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, [int(0.7*n_epochs), int(0.9*n_epochs)], gamma=0.1, verbose=False)

    loss_fn = nn.CrossEntropyLoss()

    for epoch in tqdm(range(n_epochs)):
        net.train()
        for inputs, labels in train_loader:
            opt.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = net(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            opt.step()

        scheduler.step()
        # print(' epoch {}/{} test_acc = {:.2f}'.format(epoch, n_epochs, eval_model(net, test_dataset, device)))

    return net


def eval_model(net, test_dataset, device, num_workers=10):
    batch_size = 32

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    net.eval()
    correct, total = 0.0, 0.0
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, pred = torch.max(outputs.data, 1)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]

    acc = correct / total * 100
    return acc


def get_model(model_name, model_width, n_classes):
    if model_name == 'resnet18':
        net = resnet18(n_classes=n_classes, width=model_width)
    elif model_name == 'resnet34':
        net = resnet34(n_classes=n_classes, width=model_width)
    elif model_name == 'resnet50':
        net = resnet50(n_classes=n_classes, width=model_width)
    elif model_name == 'resnet101':
        net = resnet101(n_classes=n_classes, width=model_width)
    else:
        raise ValueError("Invalid model_name")
    return net


if __name__ == '__main__':
    main()
