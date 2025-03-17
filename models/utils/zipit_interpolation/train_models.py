import numpy as np
import uuid
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from torch.optim import SGD, lr_scheduler
from tqdm import tqdm
from models.resnets import *


def main():
    device = 'cuda'
    lr = 0.4
    epochs = 50

    model = resnet20(num_classes=100)
    model.to(device)
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    loss_fn = nn.CrossEntropyLoss()

    dataset_path = '/usr/share/mammoth_datasets/CIFAR100'
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761))
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761))
    ])

    train_dataset = torchvision.datasets.CIFAR100(dataset_path, train=True,  transform=train_transforms)
    test_dataset = torchvision.datasets.CIFAR100(dataset_path, train=False, transform=test_transforms)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=10)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=10)

    ne_iters = len(train_loader)
    warmup = np.interp(np.arange(1+5*ne_iters), [0, 5*ne_iters], [1e-6, 1])
    ni = (epochs-5)*ne_iters
    xx = np.arange(ni)/ni
    cosine = (np.cos(np.pi*xx) + 1)/2
    lr_schedule = np.concatenate([warmup, cosine])
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_schedule.__getitem__)
    # scheduler = lr_scheduler.MultiStepLR(optimizer, [30, 40, 45], 0.1)

    for _ in tqdm(range(epochs)):
        model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs.cuda())
            loss = loss_fn(outputs, labels.cuda())
            loss.backward()
            optimizer.step()
            scheduler.step()
        print('Acc=%.2f%%' % (evaluate(model, loader=test_loader)/100))

    sd = model.state_dict()
    torch.save(sd, './checkpoints/batchnorm_resnet20x%d_e%d_%s.pt' % (1, 50, str(uuid.uuid4())))


def evaluate(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs.cuda())
            pred = outputs.argmax(dim=1)
            correct += (labels.cuda() == pred).sum().item()
    return correct


if __name__ == '__main__':
    main()
