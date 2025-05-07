import torch
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import math
import numpy as np

from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Bilevel Coreset Selection via Regularization')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--buffer_policy', choices=['balanced_reservoir', 'reservoir'], default='reservoir', help='policy for selecting samples stored into buffer')
    return parser


class SimpleCNN(nn.Module):
    def __init__(self, n_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class Training():
    def __init__(self, proxy_model, beta, device, lr_proxy_model, lr_weights):
        self.proxy_model = proxy_model
        self.lr_p = lr_proxy_model
        self.lr_w = lr_weights
        self.optimizer_theta_p_model = torch.optim.SGD(self.proxy_model.parameters(), lr=self.lr_p)
        self.weight_optimizer = None
        self.device = device
        self.eta = 0.5
        self.beta = beta
        self.buffer = []
        self.identity = []

    def init_proxy_model(self):
        for m in self.proxy_model.modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                torch.nn.init.xavier_uniform_(m.weight)

    def train_inner(self, data_S, target_S, sample_weights, inner_epchos):
        loss = math.inf
        for _ in range(inner_epchos):
            self.optimizer_theta_p_model = torch.optim.SGD(self.proxy_model.parameters(), lr=self.lr_p)
            self.proxy_model.train()
            self.optimizer_theta_p_model.zero_grad()
            data = data_S.to(self.device).type(torch.float)
            target = target_S.to(self.device).type(torch.long)
            sample_weights = sample_weights.to(self.device).type(torch.float).detach()
            output = self.proxy_model(data)
            loss = torch.mean(sample_weights * F.cross_entropy(output, target, reduction='none'))
            loss.backward()
            self.optimizer_theta_p_model.step()
            self.proxy_model.zero_grad()
        return loss

    def train_outer(self, data, target, task_id, data_weights, topk, ref_x=None, ref_y=None):
        data = data.to(self.device)
        target = target.to(self.device).type(torch.long)
        sample_weights = data_weights.to(self.device)
        X_S = data[:].to(self.device)
        y_S = target[:].to(self.device).type(torch.long)
        return self.update_sample_weights(data, target, task_id, X_S, y_S, sample_weights, topk, beta=self.beta, ref_x=ref_x, ref_y=ref_y)

    def update_sample_weights(self, input_train, target_train, task_id, input_selected, target_selected,  sample_weights, topk, beta, epsilon=1e-3, ref_x=None, ref_y=None):
        z = torch.normal(0, 1, size=[topk]).to(self.device)
        loss_outer = F.cross_entropy(self.proxy_model(input_train), target_train, reduction='none')
        topk_weights, ind = sample_weights.topk(topk)
        loss_outer_avg = torch.mean(loss_outer) - beta*(topk_weights + epsilon*z).sum()
        if ref_x != None:
            loss_buff = []
            for i in range(task_id-1):
                loss_buff += F.cross_entropy(self.proxy_model(ref_x[i].to(self.device), i+1), ref_y[i].to(self.device), reduction='none')
            loss_buff_avg = torch.mean(torch.Tensor(loss_buff))
            alpha = 0.1
            loss_outer_avg += alpha * loss_buff_avg
        d_theta = torch.autograd.grad(loss_outer_avg, self.proxy_model.parameters())
        v_0 = d_theta
        loss_inner = torch.mean(F.softmax(sample_weights, dim=-1) * F.cross_entropy(
            self.proxy_model(input_selected), target_selected, reduction='none'))
        grads_theta = torch.autograd.grad(loss_inner, self.proxy_model.parameters(), create_graph=True)
        G_theta = []
        for p, g in zip(self.proxy_model.parameters(), grads_theta):
            if g == None:
                G_theta.append(None)
            else:
                G_theta.append(p-self.lr_p*g)
        v_Q = v_0
        for _ in range(3):
            v_new = torch.autograd.grad(G_theta, self.proxy_model.parameters(), grad_outputs=v_0, retain_graph=True)
            v_0 = [i.detach() for i in v_new]
            for i in range(len(v_0)):
                v_Q[i].add_(v_0[i].detach())

        jacobian = -torch.autograd.grad(grads_theta, sample_weights, grad_outputs=v_Q)[0]
        with torch.no_grad():
            sample_weights -= self.lr_w * jacobian

        return sample_weights, jacobian, loss_outer


class BCSR:
    """"
    Coreset selection basede on bilevel optimzation

    Args:
        proxy_model: model for coreset selection
        lr_proxy_model: learning rare for proxy_model
        beta: balance the loss and regularizer
        out_dim: input dimension
        max_outer_it: outer loops for bilevel optimizaiton
        max_inner_it: inner loops for bilevel optimizaiton
        weight_lr: step size for updating samlple weights
        candidate_batch_size: number of coreset candidates
    """

    def __init__(self, proxy_model, lr_proxy_model,  beta, out_dim=10, max_outer_it=50, max_inner_it=1, weight_lr=1e-1,
                 candidate_batch_size=600, logging_period=1000, device='cpu'):
        self.out_dim = out_dim
        self.max_outer_it = max_outer_it
        self.max_inner_it = max_inner_it
        self.weight_lr = weight_lr
        self.candidate_batch_size = candidate_batch_size
        self.logging_period = logging_period
        self.nystrom_batch = None
        self.nystrom_normalization = None
        self.param_size = []
        self.seed = 0
        self.lr_proxy_model = lr_proxy_model
        self.training_model_op = Training(proxy_model, beta, device, lr_proxy_model, lr_weights=self.weight_lr)
        for p in self.training_model_op.proxy_model.parameters():
            self.param_size.append(p.size())

    def outer_loss(self, X, y, task_id, topk, ref_x=None, ref_y=None):
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        n = X.shape[0]
        coreset_weights = 1.0 / n * torch.ones([n], dtype=torch.float, requires_grad=True)

        _, _, outer_loss = self.training_model_op.train_outer(X, y, task_id, coreset_weights, topk, ref_x,
                                                              ref_y)
        return outer_loss

    def projection_onto_simplex(self, v, b=1):
        v = v.cpu().detach().numpy()
        n_features = v.shape[0]
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - b
        ind = np.arange(n_features) + 1
        cond = u - cssv / ind > 0
        rho = ind[cond][-1]
        theta = cssv[cond][-1] / float(rho)
        w = np.maximum(v - theta, 0)
        w = torch.from_numpy(w).cuda()
        w.requires_grad = True
        return w

    def select_coreset(self, model, X, y, task_id,  topk, out_loss=None, ref_x=None, ref_y=None):
        np.random.seed(self.seed)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        n = X.shape[0]
        self.training_model_op.proxy_model.load_state_dict(model.state_dict())
        # initialize sample weights
        coreset_weights = 1.0/n*torch.ones([n], dtype=torch.float, requires_grad=True)
        # project sample weights onto simplex
        coreset_weights = self.projection_onto_simplex(coreset_weights)

        self.training_model_op.lr_p = self.lr_proxy_model
        self.training_model_op.lr_w = self.weight_lr

        # solve the bilevel problem
        for i in range(self.max_outer_it):
            inner_loss = self.training_model_op.train_inner(X, y, coreset_weights, self.max_inner_it)
            coreset_weights, _, outer_loss = self.training_model_op.train_outer(X, y, task_id, coreset_weights, topk, ref_x, ref_y)
            coreset_weights = self.projection_onto_simplex(coreset_weights)
            total_loss = torch.mean(outer_loss).item()
        print('inner loss:{:.3f}, outer loss:{:.3f}'.format(inner_loss, total_loss))
        if out_loss != None and n == 50:
            out_loss.append(total_loss)

        return torch.multinomial(coreset_weights, topk, replacement=False), out_loss


class Bcsr(ContinualModel):
    NAME = 'bcsr'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super().__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device, mode=args.buffer_policy)
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
                             labels=labels[:real_batch_size])

        return loss.item()

    def end_task(self, dataset):
        device = self.args.device
        num_tasks = self.args.n_tasks if self.args.n_tasks is not None else dataset.N_TASKS
        task_id = self.t
        buffer_size = self.args.buffer_size

        X_task, y_task = [], []
        for x, y, _, _ in dataset.train_loader:
            X_task.append(x)
            y_task.append(y)
        X_task = torch.cat(X_task)
        y_task = torch.cat(y_task)

        model = SimpleCNN(dataset.N_CLASSES).to(device)

        selector = BCSR(model, lr_proxy_model=0.01, beta=1.0, device=device)
        selected_indices, _ = selector.select_coreset(model, X_task.to(device), y_task.to(device), task_id, topk=buffer_size // (task_id + 1))

        added_labels = []

        train_dataset = dataset.train_loader.dataset

        if self.t == 0:
            for buffer_idx, dataset_idx in enumerate(selected_indices):
                _, label, not_aug_img, _ = train_dataset[dataset_idx]

                self.buffer.examples[buffer_idx] = not_aug_img
                self.buffer.labels[buffer_idx] = label
                added_labels.append(label.item())
        else:
            for dataset_idx in selected_indices:
                _, label, not_aug_img, _ = train_dataset[dataset_idx]

                buffer_classes, buffer_counts = torch.unique(self.buffer.labels, return_counts=True)
                max_idx = torch.argmax(buffer_counts).item()
                max_label = buffer_classes[max_idx]
                buffer_class_idxs = torch.argwhere(self.buffer.labels == max_label).flatten()
                buffer_idx = buffer_class_idxs[torch.randint(len(buffer_class_idxs), (1,)).item()]

                self.buffer.examples[buffer_idx] = not_aug_img
                self.buffer.labels[buffer_idx] = label
                added_labels.append(label.item())

        print()
        print('labels added to the buffer')
        u1, u2 = np.unique(added_labels, return_counts=True)
        print(u1.tolist(), u2.tolist())
        print()
        print('all labels in the buffer:')
        u1, u2 = torch.unique(self.buffer.labels, return_counts=True)
        print(u1.tolist(), u2.tolist())

        self.t += 1
