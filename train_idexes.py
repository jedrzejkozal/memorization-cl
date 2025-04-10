import torch
import torch.nn as nn
import numpy as np

import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR100
from tqdm import tqdm

from backbones.resnet import resnet18
from utils.conf import base_path_dataset as base_path


class CIFAR100WithIndexes(CIFAR100):
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        return img, label, index


def main():
    # train()
    # train_forgetting()

    results = []
    for buffer_size in [500, 1000, 2000, 5000, 10000, 20000]:
        print('buffer size ', buffer_size)
        fraction = buffer_size / 50000
        # cutout = 1 - fraction
        # train_subset(cutout)
        test_acc = train_subset(buffer_size)
        results.append(test_acc)
    print(results)


def train():
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

    train_dataset = CIFAR100WithIndexes(root=base_path() + 'CIFAR100', train=True, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    test_dataset = CIFAR100(root=base_path() + 'CIFAR100', train=False, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16)

    opt = torch.optim.SGD(net.parameters(), lr=0.1, weight_decay=0.0, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, [35, 45], gamma=0.1, verbose=False)
    loss_fn = nn.CrossEntropyLoss()

    iteration_counter = 0
    trained_order = []
    trained_order_set = set()
    trained_iteration = []

    for epoch in tqdm(range(n_epochs)):
        net.train()
        for inputs, labels, dataset_indexes in train_loader:
            opt.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = net(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            opt.step()

            with torch.no_grad():
                y_pred = outputs.argmax(dim=1)
                batch_idxs = torch.argwhere(y_pred == labels).flatten()
                idxs = dataset_indexes[batch_idxs]
                for i in idxs:
                    i = i.item()
                    if i not in trained_order_set:
                        trained_order_set.add(i)
                        trained_order.append(i)
                        trained_iteration.append(iteration_counter)
            iteration_counter += 1

            # print(len(trained_order_set))
        scheduler.step()

        test_acc = compute_acc(net, test_loader, device)
        print(f'epoch {epoch}/{n_epochs} test acc: {test_acc}')

    trained_order = np.array(trained_order)
    trained_iteration = np.array(trained_iteration)
    np.save('trained_order_f.npy', trained_order)
    np.save('trained_iteration_f.npy', trained_iteration)
    torch.save(net.state_dict(), f'resnet_order.pth')


def train_forgetting():
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

    train_dataset = CIFAR100WithIndexes(root=base_path() + 'CIFAR100', train=True, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    test_dataset = CIFAR100(root=base_path() + 'CIFAR100', train=False, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16)

    opt = torch.optim.SGD(net.parameters(), lr=0.1, weight_decay=0.0, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, [35, 45], gamma=0.1, verbose=False)
    loss_fn = nn.CrossEntropyLoss()

    iteration_counter = 0
    trained_order = []
    trained_order_set = set()
    trained_iteration = []

    for epoch in tqdm(range(n_epochs)):
        net.train()
        for inputs, labels, dataset_indexes in train_loader:
            opt.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = net(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            opt.step()

            with torch.no_grad():
                y_pred = outputs.argmax(dim=1)
                batch_idxs = torch.argwhere(y_pred == labels).flatten()
                idxs = dataset_indexes[batch_idxs]
                for i in idxs:
                    i = i.item()
                    if i not in trained_order_set:
                        trained_order_set.add(i)
                        trained_order.append(i)
                        trained_iteration.append(iteration_counter)

                batch_idxs = torch.argwhere(y_pred != labels).flatten()
                idxs = dataset_indexes[batch_idxs]
                for i in idxs:
                    i = i.item()
                    if i in trained_order_set:
                        trained_order_set.remove(i)
                        index = trained_order.index(i)
                        trained_order.remove(i)
                        trained_iteration.pop(index)
                        # trained_iteration.append(iteration_counter)

            iteration_counter += 1

            # print(len(trained_order_set))
        scheduler.step()

        test_acc = compute_acc(net, test_loader, device)
        print(f'epoch {epoch}/{n_epochs} test acc: {test_acc}')

    trained_order = np.array(trained_order)
    trained_iteration = np.array(trained_iteration)
    np.save('trained_order.npy', trained_order)
    np.save('trained_iteration.npy', trained_iteration)


def train_subset(dataset_size):
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
    test_dataset = CIFAR100(root=base_path() + 'CIFAR100', train=False, transform=test_transform)

    opt = torch.optim.SGD(net.parameters(), lr=0.1, weight_decay=0.0, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, [35, 45], gamma=0.1, verbose=False)
    loss_fn = nn.CrossEntropyLoss()

    trained_order = np.load('trained_order.npy')
    trained_iteration = np.load('trained_iteration.npy')

    trained_iter = np.zeros(len(train_dataset))
    for i, iter in zip(trained_order, trained_iteration):
        trained_iter[i] = iter

    # subset_limit = int(data_cutout * len(trained_order))

    # select middle
    # half_idx = len(trained_order) // 2
    # trained_subset = Subset(train_dataset, trained_order[half_idx-dataset_size//2:half_idx+dataset_size//2])

    # select lowest and highest
    # subset_indexes = np.concatenate([trained_order[int(0.1*len(trained_order)):][:dataset_size // 2], trained_order[:int(0.8*len(trained_order))][-dataset_size // 2:]])
    # trained_subset = Subset(train_dataset, subset_indexes)

    # select lowest and middle
    subset_indexes = np.concatenate([trained_order[int(0.1*len(trained_order)):][:dataset_size // 2], trained_order[:int(0.6*len(trained_order))][-dataset_size // 2:]])
    trained_subset = Subset(train_dataset, subset_indexes)

    # select random
    # random_indexes = np.random.choice(np.arange(len(train_dataset)), size=dataset_size)
    # trained_subset = Subset(train_dataset, random_indexes)

    print('subset size = ', len(trained_subset))
    assert len(trained_subset) == dataset_size
    train_loader = DataLoader(trained_subset, batch_size=batch_size, shuffle=True, num_workers=16)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16)

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

        test_acc = compute_acc(net, test_loader, device)
        print(f'epoch {epoch}/{n_epochs} test acc: {test_acc}')

    return test_acc


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
