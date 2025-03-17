import torch
import torch.utils.data
import torch.nn.functional as F
import torch.optim as optim
import copy

from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
# from models.utils.weight_interpolation import *
from models.utils.zipit_interpolation.interpolate import interpolate1 as interpolate
# from models.utils.sharpness_aware_optim import SAM
# from models.utils.weight_interpolation_mobilenet import *


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual Learning with Weight Interpolation https://arxiv.org/abs/2404.04002')
    parser.add_argument('--interpolation_alpha', type=float, default=0.5, help='interpolation alpha')
    parser.add_argument('--permuation_epochs', type=int, default=1)
    parser.add_argument('--batchnorm_epochs', type=int, default=1)

    parser.add_argument('--sam_rho', default=0.05, type=float, help='rho hyperparameter for SAM optimizer')
    parser.add_argument('--sam_adaptive', action='store_true', help='use adaptive SAM optimizer')

    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    return parser


class Clewipp(ContinualModel):
    NAME = 'clewipp'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super().__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.old_model = None
        self.interpolation_alpha = args.interpolation_alpha
        self.clipgrad = 1.0

        self.first_task = True
        self.t = 0

    def begin_task(self, dataset):
        self.steps_per_epoch = len(dataset.train_loader)

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
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clipgrad)
        self.opt.step()
        # self.opt.first_step(zero_grad=True)

        # outputs = self.net(inputs)
        # loss = self.loss(outputs, labels)
        # loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clipgrad)
        # self.opt.second_step(zero_grad=True)

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels[:real_batch_size])

        return loss.item()

    def end_task(self, dataset):
        if self.first_task:
            self.first_task = False
            self.old_model = self.deepcopy_model(self.net)
            self.t += 1
            return

        buffer_dataloder = self.get_buffer_dataloder()
        # self.interpolation_plot(dataset, buffer_dataloder)

        # old_classifier = copy.deepcopy(self.old_model.classifier)
        # new_classifier = copy.deepcopy(self.net.classifier)

        # old_activations_X, old_activations_y = self.get_activations(self.old_model, buffer_dataloder)
        # new_activations_X, new_activations_y = self.get_activations(self.net, buffer_dataloder)

        self.old_model = interpolate(self.net, self.old_model, buffer_dataloder, self.device, alpha=self.interpolation_alpha,
                                     permuation_epochs=self.args.permuation_epochs, batchnorm_epochs=self.args.batchnorm_epochs)
        self.net = self.deepcopy_model(self.old_model)

        # interpolation_activations_X, interpolation_activations_y = self.get_activations(self.net, buffer_dataloder)

        # previous_classes = list(range(self.t * dataset.N_CLASSES_PER_TASK))
        # current_classes = list(range(self.t * dataset.N_CLASSES_PER_TASK, (self.t + 1) * dataset.N_CLASSES_PER_TASK))
        # print()
        # print('previous_classes = ', previous_classes)
        # print('current_classes = ', current_classes)

        # print()
        # print('old_classifier.weight.shape = ', old_classifier.weight.shape)
        # with torch.no_grad():
        #     # translation_vectors = []
        #     for label in previous_classes:
        #         idx = torch.argwhere(old_activations_y == label).squeeze()
        #         class_old_act = old_activations_X[idx]
        #         idx = torch.argwhere(interpolation_activations_y == label).squeeze()
        #         class_int_act = interpolation_activations_X[idx]
        #         translation_vec = class_int_act.mean(dim=0) - class_old_act.mean(dim=0)
        #         # translation_vectors.append(translation_vec)
        #         # print('translation_vec.shape = ', translation_vec.shape)
        #         print(f'transaltion vec magnitute label {label} = ', torch.norm(translation_vec))
        #         self.net.classifier.weight[label] = old_classifier.weight[label] - 1.0 * translation_vec

        #         # covarance_old = torch.cov(class_old_act.T)
        #         # # covarance_new = torch.cov(new_activations_X.T)
        #         # covarance_int = torch.cov(class_int_act.T)

        #         # print('covarance_old = ')
        #         # print(covarance_old)
        #         # print('main_diag = ', torch.diag(covarance_old))
        #         # print('covariance old off diag sum = ', torch.sum(covarance_old) - torch.sum(torch.diag(covarance_old)))
        #         # # print('covarance_new = ')
        #         # # print(covarance_new)
        #         # # print('main_diag = ', torch.diag(covarance_new))
        #         # # print('covariance new off diag sum = ', torch.sum(covarance_new) - torch.sum(torch.diag(covarance_new)))
        #         # print('covarance_int = ')
        #         # print(covarance_int)
        #         # print('main_diag = ', torch.diag(covarance_int))
        #         # print('covariance int off diag sum = ', torch.sum(covarance_int) - torch.sum(torch.diag(covarance_int)))

        #     for label in current_classes:
        #         idx = torch.argwhere(new_activations_y == label).squeeze()
        #         class_new_act = new_activations_X[idx]
        #         idx = torch.argwhere(interpolation_activations_y == label).squeeze()
        #         class_int_act = interpolation_activations_X[idx]
        #         translation_vec = class_int_act.mean(dim=0) - class_new_act.mean(dim=0)
        #         print(f'transaltion vec magnitute label {label} = ', torch.norm(translation_vec))
        #         # translation_vectors.append(translation_vec)
        #         self.net.classifier.weight[label] = new_classifier.weight[label] - 1.0 * translation_vec

        #         # # covarance_old = torch.cov(class_old_act.T)
        #         # covarance_new = torch.cov(class_new_act.T)
        #         # covarance_int = torch.cov(class_int_act.T)

        #         # # print('covarance_old = ')
        #         # # print(covarance_old)
        #         # # print('main_diag = ', torch.diag(covarance_old))
        #         # # print('covariance old off diag sum = ', torch.sum(covarance_old) - torch.sum(torch.diag(covarance_old)))
        #         # print('covarance_new = ')
        #         # print(covarance_new)
        #         # print('main_diag = ', torch.diag(covarance_new))
        #         # print('covariance new off diag sum = ', torch.sum(covarance_new) - torch.sum(torch.diag(covarance_new)))
        #         # print('covarance_int = ')
        #         # print(covarance_int)
        #         # print('main_diag = ', torch.diag(covarance_int))
        #         # print('covariance int off diag sum = ', torch.sum(covarance_int) - torch.sum(torch.diag(covarance_int)))

        #     self.old_model = self.deepcopy_model(self.net)

        # C_vec = torch.ones(self.net.classifier.weight.shape[0], requires_grad=True, device=self.device)
        # C_opt = torch.optim.Adam([C_vec, self.net.classifier.bias], lr=0.01)

        # for _ in range(100):
        #     C_opt.zero_grad()
        #     # outputs = self.net.classifier(interpolation_activations_X)
        #     outputs = F.linear(interpolation_activations_X, torch.diag(C_vec) @ self.net.classifier.weight, self.net.classifier.bias)
        #     loss = F.cross_entropy(outputs, interpolation_activations_y)
        #     loss += 0.005 * C_vec.pow(2).sum()
        #     loss.backward()
        #     C_opt.step()

        # print('C_vec = ', C_vec)

        # with torch.no_grad():
        #     self.net.classifier.weight.data = torch.diag(C_vec) @ self.net.classifier.weight.data

        print('test')
        self.t += 1

    def get_buffer_dataloder(self, batch_size=32):
        if batch_size is None:
            batch_size = len(self.buffer)
        buf_inputs, buf_labels = self.buffer.get_data(len(self.buffer), transform=self.transform)
        buffer_dataset = torch.utils.data.TensorDataset(buf_inputs, buf_labels)
        buffer_dataloder = torch.utils.data.DataLoader(buffer_dataset, batch_size=batch_size, num_workers=0)
        return buffer_dataloder

    def get_activations(self, model, dataloader):
        status = model.training
        model.eval()
        model.to(self.device)
        activations = []
        gt = []
        with torch.no_grad():
            for data in dataloader:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs, returnt='features')
                activations.append(outputs)
                gt.append(labels)
        activations = torch.cat(activations, dim=0)
        labels = torch.cat(gt, dim=0)
        model.train(status)
        return activations, labels

    @staticmethod
    def deepcopy_model(model):
        model_copy = copy.deepcopy(model)
        # model_copy.load_state_dict(model.state_dict())
        return model_copy

    def get_scheduler(self):
        self.opt = torch.optim.SGD(self.net.parameters(), lr=self.args.lr, weight_decay=self.args.optim_wd, momentum=self.args.optim_mom)
        # self.opt = SAM(self.net.parameters(), torch.optim.SGD, rho=self.args.sam_rho, adaptive=self.args.sam_adaptive,
        #                lr=self.args.lr, weight_decay=self.args.optim_wd, momentum=self.args.optim_mom)
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.opt, milestones=[35, 45], gamma=0.1)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.opt, max_lr=5.0, epochs=self.args.n_epochs, steps_per_epoch=self.steps_per_epoch,
                                                       div_factor=10, final_div_factor=1000, pct_start=0.3)
        return self.scheduler

    def interpolation_plot(self, dataset, buffer_dataloader):
        alpha_grid = np.arange(0, 1.001, 0.02)
        interpolation_accs = []
        for alpha in alpha_grid:
            net = self.deepcopy_model(self.net)
            old = self.deepcopy_model(self.old_model)
            new_model = interpolate(net, old, buffer_dataloader, self.device, alpha=alpha)
            acc = evaluate(new_model, dataset, self.device)
            interpolation_accs.append(acc)
        print('interpolation accuracies:')
        print(interpolation_accs)
        print('old model accs:')
        accs = evaluate(self.old_model, dataset, self.device)
        print(accs)
        print('new model accs:')
        accs = evaluate(self.net, dataset, self.device)
        print(accs)


@torch.no_grad()
def evaluate(network: ContinualModel, dataset, device, last=False):
    status = network.training
    network.eval()
    network.to(device)
    accs = []
    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
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
