# based on original implementation https://github.com/grypesc/SEED

import collections
import copy
import torch
import torch.nn as nn
import torchvision.transforms
import torch.utils.data
from torch.distributions import MultivariateNormal

from models.utils.continual_model import ContinualModel
from models.utils.gmm import GaussianMixture
from backbones.resnet import resnet18, resnet50
from backbones.resnet32 import resnet32

from utils.args import add_management_args, add_experiment_args, ArgumentParser


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Selection of Experts for Ensemble Diversification, https://arxiv.org/pdf/2401.10191v1.pdf')
    add_management_args(parser)
    add_experiment_args(parser)

    parser.add_argument('--max_experts', help='Maximum number of experts', type=int, default=5)
    parser.add_argument('--gmms', help='Number of gaussian models in the mixture', type=int, default=1)
    parser.add_argument('--shared', help='Number of shared blocks', type=int, default=0)
    parser.add_argument('--initialization_strategy', help='How to initialize experts weight', type=str, choices=["first", "random"],
                        default="first")
    parser.add_argument('--ftepochs', help='Number of epochs for finetuning an expert', type=int, default=100)
    parser.add_argument('--ftwd', help='Weight decay for finetuning', type=float, default=0)
    parser.add_argument('--use_multivariate', help='Use multivariate distribution', action='store_true', default=True)
    parser.add_argument('--use_nmc', help='Use nearest mean classifier instead of bayes', action='store_true', default=False)
    parser.add_argument('--alpha', help='relative weight of kd loss', type=float, default=0.99)
    parser.add_argument('--tau', help='softmax temperature', type=float, default=3.0)
    parser.add_argument('--compensate_drifts', help='Drift compensation using MLP feature adaptation', action='store_true', default=False)
    parser.add_argument('--clipping', default=1, type=float, required=False, help='Clip gradient norm (default=%(default)s)')

    return parser


class ExtractorEnsemble(nn.Module):

    def __init__(self, taskcla, network_type, device):
        super().__init__()
        self.model = None
        self.network_type = network_type
        if network_type == "resnet18":
            self.bb_fun = resnet18
            self.num_features = 512
        # elif network_type == "resnet34":
        #     self.bb_fun = resnet34
        elif network_type == "resnet50":
            self.bb_fun = resnet50
        elif network_type == "resnet32":
            self.bb_fun = resnet32
            self.num_features = 64
        # elif network_type == "resnet20":
        #     self.num_features = 24
        #     self.bb_fun = resnet20
        else:
            raise RuntimeError("Network not supported")

        self.bbs = nn.ModuleList([])
        self.head = nn.Identity()

        # Uncomment to load a model, set 6 to number of experts that's in .pth, comment backbone training
        # self.bbs = nn.ModuleList([copy.deepcopy(bb) for _ in range(min(len(taskcla), 6))])
        # for bb in self.bbs:
        #     bb.classifier = nn.Identity()
        # state_dict = torch.load("seb-resnet32.pth")
        # self.load_state_dict(state_dict, strict=True)

        self.task_offset = [0]
        self.taskcla = taskcla
        self.device = device

    def forward(self, x):
        # semi_features = self.bbs[0].calculate_semi_features(x)
        features = [bb.forward(x) for bb in self.bbs]
        return torch.stack(features, dim=1)


class SEED(ContinualModel):
    NAME = 'seed'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super().__init__(backbone, loss, args, transform)

        self.t = 0

        network_type = 'resnet32'
        if args.backbone is not None:
            network_type = args.backbone
        # TODO fix this
        if self.args.n_tasks == 10:
            taskcla = [(0, 10), (1, 10), (2, 10), (3, 10), (4, 10), (5, 10), (6, 10), (7, 10), (8, 10), (9, 10)]
        elif self.args.n_tasks == 20:
            taskcla = [(i, 5) for i in range(20)]
        elif self.args.n_tasks == 50:
            taskcla = [(i, 2) for i in range(50)]
        elif self.args.n_tasks == 12:
            taskcla = [(i, 25) for i in range(self.args.n_tasks)]
        else:
            raise ValueError("n tasks not supported")

        self.net = ExtractorEnsemble(taskcla, network_type, self.device)

        self.max_experts: int = args.max_experts
        self.net.bbs = self.net.bbs[:args.max_experts]
        self.gmms: int = args.gmms
        self.alpha: float = args.alpha
        self.tau: float = args.tau
        self.use_multivariate = args.use_multivariate
        self.net.to(self.device)
        self.experts_distributions = []
        self.shared_layers = []
        if args.shared > 0:
            self.shared_layers = ["conv1.weight", "bn1.weight", "bn1.bias", "layer1"]
            if args.shared > 1:
                self.shared_layers.append("layer2")
                if args.shared > 2:
                    self.shared_layers.append("layer3")
                    if args.shared > 3:
                        self.shared_layers.append("layer4")

        self.initialization_strategy = args.initialization_strategy
        self.clipgrad = args.clipping

    def get_epochs(self):
        if self.t < self.max_experts:
            return self.args.n_epochs
        else:
            return self.args.ftepochs

    def begin_task(self, dataset):
        if self.t < self.max_experts:
            if self.initialization_strategy == "random" or self.t == 0:
                self.net.bbs.append(self.net.bb_fun(self.net.taskcla[self.t][1]))
            else:
                self.net.bbs.append(copy.deepcopy(self.net.bbs[0]))
            self.current_net = self.net.bbs[self.t]
            self.current_net.classifier = nn.Linear(self.net.num_features, self.net.taskcla[self.t][1])
            if self.t == 0:
                for param in self.current_net.parameters():
                    param.requires_grad = True
            else:
                for name, param in self.current_net.named_parameters():
                    param.requires_grad = True
                    for layer_not_to_train in self.shared_layers:
                        if layer_not_to_train in name:
                            self.current_net.get_parameter(name).data = self.net.bbs[0].get_parameter(name).data
                            param.requires_grad = False

            print(f'The expert has {sum(p.numel() for p in self.current_net.parameters() if p.requires_grad):,} trainable parameters')
            print(f'The expert has {sum(p.numel() for p in self.current_net.parameters() if not p.requires_grad):,} shared parameters\n')
            self.current_net.to(self.device)
            self.current_net.train()

            self._update_optimizer(self.t, self.args.optim_wd, [60, 120, 160])
        else:
            bb_to_finetune = self._choose_backbone_to_finetune(self.t, dataset.train_loader, dataset.test_loaders[-1])
            self.bb_to_finetune = bb_to_finetune
            print(f"Finetuning backbone {bb_to_finetune} on task {self.t}:")

            self.old_model = copy.deepcopy(self.net.bbs[bb_to_finetune])
            for name, param in self.old_model.named_parameters():
                param.requires_grad = False
            self.old_model.eval()

            self.current_net = self.net.bbs[bb_to_finetune]
            for name, param in self.current_net.named_parameters():
                param.requires_grad = True
                for layer_not_to_train in self.shared_layers:
                    if layer_not_to_train in name:
                        param.requires_grad = False
            self.current_net.classifier = nn.Linear(self.net.num_features, self.net.taskcla[self.t][1])
            self.current_net.to(self.device)
            self.current_net.train()
            for m in self.current_net.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

            self._update_optimizer(bb_to_finetune, self.args.ftwd, [30, 60, 80])

    def _update_optimizer(self, model_idx, weight_decay, milestones):
        """Returns the optimizer"""
        self.opt = torch.optim.SGD(self.net.bbs[model_idx].parameters(), lr=self.args.lr, weight_decay=weight_decay, momentum=0.9)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.opt, milestones=milestones, gamma=0.1)

    @torch.no_grad()
    def _choose_backbone_to_finetune(self, t, trn_loader, val_loader):
        self.create_distributions(t, trn_loader, val_loader)
        expert_overlap = torch.zeros(self.max_experts, device=self.device)
        for bb_num in range(self.max_experts):
            classes_in_t = self.net.taskcla[t][1]
            new_distributions = self.experts_distributions[bb_num][-classes_in_t:]
            kl_matrix = torch.zeros((len(new_distributions), len(new_distributions)), device=self.device)
            for o, old_gauss_ in enumerate(new_distributions):
                old_gauss = MultivariateNormal(old_gauss_.mu.data[0][0], covariance_matrix=old_gauss_.var.data[0][0])
                for n, new_gauss_ in enumerate(new_distributions):
                    new_gauss = MultivariateNormal(new_gauss_.mu.data[0][0], covariance_matrix=new_gauss_.var.data[0][0])
                    kl_matrix[n, o] = torch.distributions.kl_divergence(new_gauss, old_gauss)
            expert_overlap[bb_num] = torch.mean(kl_matrix)
            self.experts_distributions[bb_num] = self.experts_distributions[bb_num][:-classes_in_t]
        print(f"Expert overlap:{expert_overlap}")
        bb_to_finetune = torch.argmax(expert_overlap)
        self.net.task_offset = self.net.task_offset[:-1]
        return int(bb_to_finetune)

    @torch.no_grad()
    def create_distributions(self, t, trn_loader, val_loader):
        """ Create distributions for task t"""
        self.net.eval()
        classes = self.net.taskcla[t][1]
        self.net.task_offset.append(self.net.task_offset[-1] + classes)
        transforms = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(), val_loader.dataset.transform])

        class_indexes = collections.defaultdict(list)
        for i, (_, label, _) in enumerate(trn_loader.dataset):
            class_indexes[label].append(i)

        for bb_num in range(min(self.max_experts, t+1)):
            eps = 1e-8
            model = self.net.bbs[bb_num]
            for c in range(classes):
                c = c + self.net.task_offset[t]
                ds = copy.deepcopy(trn_loader.dataset)
                ds = torch.utils.data.Subset(ds, class_indexes[c])
                if len(ds) == 0:
                    raise ValueError(f"Dataset len for class {c} is equal to 0")

                loader = torch.utils.data.DataLoader(ds, batch_size=self.args.batch_size, num_workers=trn_loader.num_workers, shuffle=False)
                from_ = 0
                class_features = torch.full((2 * len(ds), self.net.num_features), fill_value=-999999999.0, device=self.net.device)
                for _, _, not_aug_images in loader:
                    images = torch.stack([transforms(img) for img in not_aug_images])
                    bsz = images.shape[0]
                    images = images.to(self.device)
                    features = model(images)
                    class_features[from_: from_+bsz] = features
                    features = model(torch.flip(images, dims=(3,)))
                    class_features[from_+bsz: from_+2*bsz] = features
                    from_ += 2*bsz
                    if self.args.debug:
                        break

                # Calculate distributions
                cov_type = "full" if self.use_multivariate else "diag"
                is_ok = False
                while not is_ok:
                    try:
                        gmm = GaussianMixture(self.gmms, class_features.shape[1], covariance_type=cov_type, eps=eps).to(self.device)
                        gmm.fit(class_features, delta=1e-3, n_iter=100)
                    except RuntimeError:
                        if eps == float('inf'):
                            raise ValueError('Float overflow during gnn fitting')

                        eps = 10 * eps
                        print(f"WARNING: Covariance matrix is singular. Increasing eps to: {eps:.7f} but this may hurt results")
                    else:
                        is_ok = True

                if len(gmm.mu.data.shape) == 2:
                    gmm.mu.data = gmm.mu.data.unsqueeze(1)
                self.experts_distributions[bb_num].append(gmm)

    def observe(self, inputs, labels, not_aug_inputs):
        labels -= self.net.task_offset[self.t]
        self.opt.zero_grad()

        outputs, features = self.current_net(inputs, returnt='all')
        loss = self.loss(outputs, labels)

        if self.t >= self.max_experts:
            # finetune backbone
            with torch.no_grad():
                old_features = self.old_model(inputs)  # resnet with fc as identity returns features by default
            kd_loss = nn.functional.mse_loss(features, old_features)
            loss = (1 - self.alpha) * loss + self.alpha * kd_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.current_net.parameters(), self.clipgrad)
        self.opt.step()

        return loss.item()

    def end_task(self, dataset):
        if self.t < self.max_experts:
            self.current_net.classifier = nn.Identity()
            self.net.bbs[self.t] = self.current_net
            self.experts_distributions.append([])
        else:
            self.current_net.classifier = nn.Identity()
            self.net.bbs[self.bb_to_finetune] = self.current_net

        print(f"Creating distributions for task {self.t}:")
        self.create_distributions(self.t, dataset.train_loader, dataset.test_loaders[-1])

        self.t += 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes a forward pass.
        :param x: batch of inputs
        :param task_label: some models require the task label
        :return: the result of the computation
        """
        features = self.net(x)
        conf = self.predict_class_bayes(features)
        return conf

    @torch.no_grad()
    def predict_class_bayes(self, features):
        log_probs = torch.full(
            (features.shape[0],
             len(self.experts_distributions),
             len(self.experts_distributions[0])),
            fill_value=-1e8, device=features.device)
        mask = torch.full_like(log_probs, fill_value=False, dtype=torch.bool)
        for bb_num, _ in enumerate(self.experts_distributions):
            for c, class_gmm in enumerate(self.experts_distributions[bb_num]):
                c += self.net.task_offset[bb_num]
                log_probs[:, bb_num, c] = class_gmm.score_samples(features[:, bb_num])
                mask[:, bb_num, c] = True

        # Task-Agnostic
        log_probs = self.softmax_temperature(log_probs, dim=2, tau=self.tau)
        confidences = torch.sum(log_probs, dim=1) / torch.sum(mask, dim=1)
        return confidences

    @staticmethod
    def softmax_temperature(x, dim, tau=1.0):
        return torch.softmax(x / tau, dim=dim)

    def get_scheduler(self):
        return self.scheduler
