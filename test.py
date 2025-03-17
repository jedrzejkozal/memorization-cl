import torch
from models.utils.zipit_interpolation.interpolate import interpolate1


@torch.no_grad()
def evaluate(network, test_loaders, device):
    status = network.training
    network.eval()
    network.to(device)
    accs = []
    for test_loader in test_loaders:
        correct, total = 0.0, 0.0
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = network(inputs)

            _, pred = torch.max(outputs.data, 1)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]

        accs.append(correct / total * 100)

    network.train(status)
    return accs


if __name__ == '__main__':
    import torchvision.transforms as transforms
    import torchvision

    dataset_path = '/usr/share/mammoth_datasets/CIFAR100'

    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761))
    ])
    train_dataset = torchvision.datasets.CIFAR100(dataset_path, train=True,  transform=train_transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=10)

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761))
    ])
    test_dataset = torchvision.datasets.CIFAR100(dataset_path, train=False, transform=test_transforms)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=10)

    # interpolate('cuda', train_loader)

    from models.utils.zipit_interpolation.models.resnets import resnet20
    model1 = resnet20(num_classes=100)
    model1.load_state_dict(torch.load('models/utils/zipit_interpolation/checkpoints/50_50/resnet20x1_v0.pt'))
    model2 = resnet20(num_classes=100)
    model2.load_state_dict(torch.load('models/utils/zipit_interpolation/checkpoints/50_51/resnet20x1_v0.pt'))
    # merged_model = interpolate1(model1, model2, train_loader, 'cuda')
    merged_model = interpolate1(model2, model1, train_loader, 'cuda')

    model1_acc = evaluate(model1, [test_loader], 'cuda')
    print('Model1 accuracy:', model1_acc)
    model2_acc = evaluate(model2, [test_loader], 'cuda')
    print('Model2 accuracy:', model2_acc)
    merged_acc = evaluate(merged_model, [test_loader], 'cuda')
    print('Merged model accuracy:', merged_acc)
