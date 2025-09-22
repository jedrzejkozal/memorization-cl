from utils.buffer_odrered import OdereredBuffer
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from models.utils.continual_model import ContinualModel
import torch
import numpy as np
import collections


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via Experience Replay with herding.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    return parser


class Herding(ContinualModel):
    NAME = 'herding'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super().__init__(backbone, loss, args, transform)
        self.buffer = OdereredBuffer(self.args.buffer_size, self.device)
        self.t = 0

    def observe(self, inputs, labels, not_aug_inputs):
        real_batch_size = inputs.shape[0]

        self.opt.zero_grad()
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))

        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)
        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels[:real_batch_size],
                             order=torch.zeros_like(labels))

        return loss.item()

    def end_task(self, dataset):
        print()
        self.t += 1
        train_dataset = dataset.train_loader.dataset
        dataset_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
        task_classes = np.unique(dataset_labels).tolist()

        data_size = self.buffer.buffer_size // self.t
        class_data_size = data_size // dataset.N_CLASSES_PER_TASK + 1
        rest_size = data_size % dataset.N_CLASSES_PER_TASK

        current_task_indexes = []
        for i, label in enumerate(self.buffer.labels):
            label = label.item()
            if label >= dataset.i - dataset.N_CLASSES_PER_TASK and label < dataset.i:
                current_task_indexes.append(i)
        # print('current_task_indexes len = ', len(current_task_indexes))
        # print('buffer len = ', len(self.buffer))

        # hearding
        fvs = []
        labels = []
        with torch.no_grad():
            for inputs, targets, _, _ in torch.utils.data.DataLoader(dataset.train_loader.dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=4):
                inputs = inputs.to(self.device)
                features = self.net.forward(inputs, returnt='features')
                fvs.append(features.to('cpu'))
                labels.append(targets)

        fvs = torch.cat(fvs, dim=0)
        labels = torch.cat(labels, dim=0)

        class_fvs = collections.defaultdict(list)
        for fv, label in zip(fvs, labels):
            class_fvs[label.item()].append(fv)
        class_fvs = {label: torch.stack(fv) for label, fv in class_fvs.items()}

        icarl_ranks = compute_icarl_ranks(class_fvs)

        selected_samples_idxs = []
        for i, label in enumerate(task_classes):
            size = class_data_size if i < rest_size else class_data_size - 1
            class_selected_idxs = torch.argsort(icarl_ranks[label])[:size]  # class_trained_order[label][-size:]
            class_idxs = torch.argwhere(labels == label)
            selected_samples_idxs.extend(class_idxs[class_selected_idxs])

        # assert len(selected_samples_idxs) == len(current_task_indexes), f'should be equal got {len(selected_samples_idxs)} and {len(current_task_indexes)}'

        added_labels = []
        for order, (buffer_idx, dataset_idx) in enumerate(zip(current_task_indexes, selected_samples_idxs)):
            _, label, not_aug_img, _ = train_dataset[dataset_idx]
            self.buffer.examples[buffer_idx] = not_aug_img
            self.buffer.labels[buffer_idx] = label
            self.buffer.order[buffer_idx] = order
            added_labels.append(label.item())

        print()
        print('labels added to the buffer')
        print(np.unique(added_labels, return_counts=True))
        print()
        print('all labels in the buffer:')
        print(torch.unique(self.buffer.labels, return_counts=True))


def compute_icarl_ranks(class_fvs):
    class_means = compute_means(class_fvs)

    icarl_ranks = {}
    with torch.no_grad():
        for label, fv in class_fvs.items():
            mean_vector = class_means[label]

            current_avrg = []
            selected_idxs = []
            for k in range(len(fv)):
                if k == 0:
                    scores = torch.norm(mean_vector - fv, p=2, dim=1)
                else:
                    scores = torch.norm(mean_vector - 1/(k+1) * (fv + torch.sum(torch.stack(current_avrg), dim=0)), p=2, dim=1)
                idxs_k = torch.argsort(scores)
                for idx_k in idxs_k:
                    if idx_k.item() not in selected_idxs:
                        current_avrg.append(fv[idx_k])
                        selected_idxs.append(idx_k.item())
                        break
            assert list(sorted(selected_idxs)) == list(range(len(selected_idxs)))

            icarl_class_ranks = [0 for _ in range(len(fv))]
            for i, idx_k in enumerate(selected_idxs):
                icarl_class_ranks[idx_k] = i
            icarl_ranks[label] = torch.Tensor(icarl_class_ranks)
    return icarl_ranks


def compute_means(class_fvs):
    class_means = {}
    with torch.no_grad():
        for label, fv in class_fvs.items():
            mean = torch.mean(fv, dim=0)
            class_means[label] = mean
    return class_means
