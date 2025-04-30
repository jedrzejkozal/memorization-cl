import torch
import torch.utils
import torch.utils.data

from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from backbones.resnet import resnet18


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='GCORESET SELECTION VIA REDUCIBLE LOSS IN CONTINUAL LEARNING \
                            https://openreview.net/pdf?id=mAztx8QO3B')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)

    parser.add_argument('--t_out', default=10, type=int, help='number of auxilary model training during coreset selection')

    parser.add_argument('--buffer_policy', choices=['balanced_reservoir', 'reservoir'], default='balanced_reservoir', help='policy for selecting samples stored into buffer')
    return parser


class TensorDatasetAug(torch.utils.data.TensorDataset):
    def __init__(self, transform, *tensors):
        super().__init__(*tensors)
        self.transform = transform

    def __call__(self, idx):
        img, label = super().__call__(idx)
        img = self.transform(img)
        return img, label


class CSReLPrv(ContinualModel):
    NAME = 'csrel_prv'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super().__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device, mode=args.buffer_policy)
        self.t = 0

    def observe(self, inputs, labels, not_aug_inputs):
        real_batch_size = inputs.shape[0]

        self.opt.zero_grad()
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()

        # self.buffer.add_data(examples=not_aug_inputs,
        #                      labels=labels[:real_batch_size])

        if not hasattr(self.buffer, 'examples'):
            self.buffer.init_tensors(examples=not_aug_inputs, labels=labels[:real_batch_size], logits=None, task_labels=None)

        return loss.item()

    @torch.no_grad()
    def end_task(self, dataset):
        self.t += 1

        import matplotlib.pyplot as plt

        # Move the tensor to CPU and detach it from the computation graph
        examples = self.buffer.examples.cpu().detach()
        print(examples)

        # Separate the channels
        channels = ['Red', 'Green', 'Blue']
        for i in range(examples.shape[1]):  # Assuming shape is (N, C, H, W)
            channel_data = examples[:, i, :, :].flatten().numpy()
            plt.hist(channel_data, bins=50, alpha=0.7, color=channels[i], label=f'{channels[i]} Channel')

        plt.title('Histogram of Pixel Values in Buffer Examples')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.legend()

        task_size = self.args.buffer_size // self.t
        coreset_inputs = []
        coreset_labels = []

        buf_inputs, buf_labels = self.buffer.get_all_data()
        for j in range(self.t-1):
            j_mask = torch.logical_and(buf_labels >= j*dataset.N_CLASSES_PER_TASK, buf_labels < (j+1)*dataset.N_CLASSES_PER_TASK)
            task_inputs = buf_inputs[j_mask]
            task_labels = buf_labels[j_mask]
            size = task_size + 1 if j < self.args.buffer_size % self.t else task_size
            S_data, S_labels = self.coreset_selection(task_inputs.cpu(), task_labels.cpu(), size, dataset.N_CLASSES, dataset.test_transform, concat_holdout=False)
            coreset_inputs.append(S_data.cpu())
            coreset_labels.append(S_labels.cpu())

        train_inputs, train_labels = [], []
        for data in dataset.train_loader:
            lab, not_aug_inp = data[1], data[2]
            train_inputs.append(not_aug_inp)
            train_labels.append(lab)
        train_inputs = torch.cat(train_inputs)
        train_labels = torch.cat(train_labels)
        S_data, S_labels = self.coreset_selection(train_inputs, train_labels, task_size, dataset.N_CLASSES, dataset.test_transform)
        coreset_inputs.append(S_data.cpu())
        coreset_labels.append(S_labels.cpu())

        coreset_inputs = torch.cat(coreset_inputs)
        coreset_labels = torch.cat(coreset_labels)
        assert len(coreset_inputs) == self.args.buffer_size

        self.buffer.examples = coreset_inputs.to(self.args.device)
        self.buffer.labels = coreset_labels.to(self.args.device)
        print()
        print(coreset_labels)
        print(torch.unique(self.buffer.labels, return_counts=True))

        import matplotlib.pyplot as plt

        # Move the tensor to CPU and detach it from the computation graph
        examples = self.buffer.examples.cpu().detach()
        print(examples)

        plt.figure()
        # Separate the channels
        channels = ['Red', 'Green', 'Blue']
        for i in range(examples.shape[1]):  # Assuming shape is (N, C, H, W)
            channel_data = examples[:, i, :, :].flatten().numpy()
            plt.hist(channel_data, bins=50, alpha=0.7, color=channels[i], label=f'{channels[i]} Channel')

        plt.title('Histogram of Pixel Values in Buffer Examples')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()

        # debug
        # print()
        # images = self.buffer.examples[-144:]
        # if self.t > 0:
        #     import matplotlib.pyplot as plt

        #     _, axes = plt.subplots(12, 12, figsize=(9, 9))
        #     for i, ax in enumerate(axes.flat):
        #         ax.imshow(images[i].permute(1, 2, 0).cpu().numpy())
        #         ax.axis('off')
        #     plt.tight_layout()
        #     plt.show()

    def coreset_selection(self, inputs, labels, select_size, n_classes, test_transforms, concat_holdout=True):
        """algorithm 3 from the paper"""
        buf_inputs, buf_labels = self.buffer.get_all_data()
        buf_inputs, buf_labels = buf_inputs.cpu(), buf_labels.cpu()
        if concat_holdout:
            buf_inputs = torch.cat([inputs, buf_inputs], dim=0)
            buf_labels = torch.cat([labels, buf_labels], dim=0)
        if self.t == 1:
            buf_inputs = inputs
            buf_labels = labels

        holdout_model = self.train_model(buf_inputs, buf_labels, n_classes)
        holdout_loss = self.evalute_loss(holdout_model, inputs, labels, test_transforms)

        select_step = select_size // self.args.t_out
        coreset_inputs = []
        coreset_labels = []
        used_indexes = []
        rest = 0
        for k in range(self.args.t_out):
            size = select_step + 1 if k < select_size % self.args.t_out else select_step
            if rest >= 0:
                size += rest
            if len(coreset_inputs) > 0:
                coreset_model = self.train_model(torch.cat(coreset_inputs), torch.cat(coreset_labels), n_classes)
                coreset_loss = self.evalute_loss(coreset_model, inputs, labels, test_transforms)
            else:
                coreset_loss = torch.zeros_like(holdout_loss)

            ReL = coreset_loss - holdout_loss
            for i in used_indexes:
                ReL[i] = -torch.inf

            if len(ReL) - len(used_indexes) > size:
                _, topk_idxs = torch.topk(ReL, k=size, sorted=False)
            else:
                topk_idxs = list(range(0, len(ReL)))
                for i in used_indexes:
                    try:
                        topk_idxs.remove(i)
                    except ValueError:
                        continue
                topk_idxs = torch.Tensor(topk_idxs).to(torch.long)  # torch.arange(0, len(ReL), dtype=torch.long)

            # topk_idxs = torch.randperm(len(inputs))[:size]  # TODO: debug
            # print(topk_idxs)
            used_indexes.extend(topk_idxs.tolist())

            # print()
            # print('top idxs = ', topk_idxs)

            coreset_inputs.append(inputs[topk_idxs])
            coreset_labels.append(labels[topk_idxs])
            if len(topk_idxs) < size:
                rest = size - len(topk_idxs)
            else:
                rest = 0

            # print(f'k = {k}, corset subset size = {len(topk_idxs)}')

        # print('rest after selection loop = ', rest)

        coreset_inputs = torch.cat(coreset_inputs)
        coreset_labels = torch.cat(coreset_labels)
        assert len(coreset_inputs) == select_size, f'selected coreset size = {len(coreset_inputs)}, should be size = {select_size}'
        assert len(coreset_inputs) == len(coreset_labels)
        return coreset_inputs, coreset_labels

    def train_model(self, buf_inputs, buf_labels, n_classes):
        torch.set_grad_enabled(True)

        dataset = TensorDatasetAug(self.transform, buf_inputs.cpu(), buf_labels.cpu())
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
        model = resnet18(n_classes=n_classes)
        model.train()
        model.to(self.args.device)
        opt = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        epochs = 1  # 20

        for _ in range(epochs):
            for input, labels in dataloader:
                input = input.to(self.args.device)
                labels = labels.to(self.args.device)
                opt.zero_grad()

                y_pred = model(input)
                loss = self.loss(y_pred, labels)
                loss.backward()
                opt.step()

        print()
        print('loss = ', loss.item())
        model.eval()
        torch.set_grad_enabled(False)
        return model

    def evalute_loss(self, model, inputs, labels, test_transforms, batch_size=32):
        model.eval()
        losses = []
        import torchvision.transforms as T
        test_transforms = T.Compose([T.ToPILImage(), test_transforms])
        transf_inputs = torch.stack(list(test_transforms(inp) for inp in inputs))
        with torch.no_grad():
            for i in range(0, len(transf_inputs), batch_size):
                batch_inputs = transf_inputs[i:i + batch_size]
                batch_labels = labels[i:i + batch_size]
                batch_inputs = batch_inputs.to(self.args.device)
                batch_labels = batch_labels.to(self.args.device)
                outputs = model(batch_inputs)
                batch_losses = self.loss(outputs, batch_labels, reduction='none')
                losses.extend(batch_losses.cpu())
        losses = torch.Tensor(losses)
        return losses
