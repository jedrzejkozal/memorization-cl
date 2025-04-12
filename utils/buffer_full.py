from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


def balanced_reservoir_sampling(num_seen_examples: int, buffer_size: int, labels: torch.Tensor) -> int:
    """
    Balanced reservoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :return: the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        classes, counts = torch.unique(labels, return_counts=True)
        i = torch.argmax(counts).item()
        l = classes[i]
        idx = torch.argwhere(labels == l).flatten()
        rand_idx = np.random.randint(0, len(idx))
        rand = idx[rand_idx]
        return rand
    else:
        return -1


class FullBuffer:
    """buffer with full access to previous data"""

    def __init__(self, buffer_size, device, n_tasks=None, mode='reservoir', policy='random'):
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0
        self.attributes = ['examples', 'labels', 'logits', 'task_labels', 'distances']
        assert policy in ['random', 'grasp', 'memorisation']
        self.policy = policy
        self.uses_logits = False
        self.batch_idx = 0

    def to(self, device):
        self.device = device
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                setattr(self, attr_str, getattr(self, attr_str).to(device))
        return self

    def __len__(self):
        return min(self.num_seen_examples, self.buffer_size)

    def init_tensors(self, examples: torch.Tensor, labels: torch.Tensor,
                     logits: torch.Tensor, task_labels: torch.Tensor, distances: torch.Tensor) -> None:
        """
        Initializes just the required tensors.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        """
        for attr_str in self.attributes:
            attr = eval(attr_str)
            if attr is not None and not hasattr(self, attr_str):
                typ = torch.int64 if attr_str.endswith('labels') else torch.float32
                setattr(self, attr_str, torch.zeros((self.buffer_size,
                        *attr.shape[1:]), dtype=typ, device=self.device))

    def add_data(self, examples, labels=None, logits=None, task_labels=None):
        if logits is not None:
            self.uses_logits = True

    def update_buffer(self, dataset, net, minibatch_size, n_epochs):
        with torch.no_grad():
            all_examples = []
            all_labels = []
            all_logits = []
            all_features = []

            status = net.training
            net.eval()
            for data in dataset.train_loader:
                not_aug_inputs, labels = data[2], data[1]
                not_aug_inputs = not_aug_inputs.to(self.device)
                labels = labels.to(self.device)
                outputs, features = net.forward(not_aug_inputs, returnt='all')

                all_examples.append(not_aug_inputs)
                all_labels.append(labels)
                all_logits.append(outputs.data)
                all_features.append(features)
            net.train(status)

            all_examples = torch.cat(all_examples, dim=0)
            all_labels = torch.cat(all_labels)
            all_logits = torch.cat(all_logits, dim=0)
            all_features = torch.cat(all_features, dim=0)
            assert all_examples.shape[0] == all_labels.shape[0] == all_logits.shape[0] == all_features.shape[
                0], f'shapes should be equal, got : {all_examples.shape[0]} {all_labels.shape[0]} {all_logits.shape[0]} {all_features.shape[0]}'

            distances = None
            if self.policy == 'grasp':
                distances = torch.zeros([len(all_examples)], dtype=torch.float32)
                for label in torch.unique(all_labels):
                    label = label.item()
                    idx = torch.argwhere(labels == label).flatten()
                    class_features = all_features[idx]
                    class_mean = torch.mean(class_features, dim=1)
                    distances[idx] = 0.5 * (1 - torch.nn.functional.cosine_similarity(class_features, class_mean.unsqueeze(0)))

            self.add_all_data(examples=all_examples, labels=all_labels, logits=all_logits, distances=distances)
            print('\nupdated buffer size = ', len(self))

            self.update_policy(len(dataset.train_loader) * minibatch_size * n_epochs)

    def add_all_data(self, examples, labels=None, logits=None, task_labels=None, distances=None):
        """
        Adds the data to the memory buffer according to the reservoir strategy.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        :return:
        """
        if not hasattr(self, 'examples'):
            self.init_tensors(examples, labels, logits, task_labels, distances)

        for i in range(examples.shape[0]):
            index = balanced_reservoir_sampling(self.num_seen_examples, self.buffer_size, self.labels)
            self.num_seen_examples += 1
            if index >= 0:
                self.examples[index] = examples[i].to(self.device)
                if labels is not None:
                    self.labels[index] = labels[i].to(self.device)
                if logits is not None:
                    self.logits[index] = logits[i].to(self.device)
                if task_labels is not None:
                    self.task_labels[index] = task_labels[i].to(self.device)
                if distances is not None:
                    self.distances[index] = distances[i].to(self.device)

    def update_policy(self, policy_size):
        if self.policy == 'random':
            current_size = len(self)
            print('current_size = ', current_size)
            self.sample_order = np.random.choice(current_size, size=policy_size, replace=True if policy_size >= current_size else False)
            print(self.sample_order)
        elif self.policy == 'grasp':
            self.sample_order = []
        else:
            raise ValueError('Invalid policy')

        self.batch_idx = 0

    def get_data(self, size: int, transform: nn.Module = None, return_index=False) -> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        if size > min(self.num_seen_examples, self.examples.shape[0]):
            size = min(self.num_seen_examples, self.examples.shape[0])

        choice = self.sample_order[self.batch_idx*size:(self.batch_idx+1)*size]
        self.batch_idx += 1
        if transform is None:
            def transform(x): return x
        ret_tuple = (torch.stack([transform(ee.cpu()) for ee in self.examples[choice]]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                if attr_str == 'logits' and not self.uses_logits:
                    continue
                if attr_str == 'distances':
                    continue
                attr = getattr(self, attr_str)
                ret_tuple += (attr[choice],)

        if not return_index:
            return ret_tuple
        else:
            return (torch.tensor(choice).to(self.device), ) + ret_tuple

    def get_data_by_index(self, indexes, transform: nn.Module = None) -> Tuple:
        """
        Returns the data by the given index.
        :param index: the index of the item
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        if transform is None:
            def transform(x): return x
        ret_tuple = (torch.stack([transform(ee.cpu())
                                  for ee in self.examples[indexes]]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str).to(self.device)
                ret_tuple += (attr[indexes],)
        return ret_tuple

    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples == 0:
            return True
        else:
            return False

    def get_all_data(self, transform: nn.Module = None) -> Tuple:
        """
        Return all the items in the memory buffer.
        :param transform: the transformation to be applied (data augmentation)
        :return: a tuple with all the items in the memory buffer
        """
        if transform is None:
            def transform(x): return x
        ret_tuple = (torch.stack([transform(ee.cpu())
                                  for ee in self.examples]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr,)
        return ret_tuple

    def empty(self) -> None:
        """
        Set all the tensors to None.
        """
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.num_seen_examples = 0
