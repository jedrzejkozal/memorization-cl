import torch

from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='GRADIENT CORESET BASED REPLAY BUFFER SELECTION FOR CONTINUAL LEARNING.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    # parser.add_argument('--rehersal_policy', choices=['random', 'grasp', 'grasp_modified', 'memorisation', 'memorisation_modified'], required=True,
    #                     help='policy for selecting what samples should be retrived from the buffer')
    parser.add_argument('--buffer_policy', choices=['balanced_reservoir', 'reservoir'], default='balanced_reservoir', help='policy for selecting samples stored into buffer')
    return parser


class Gcr(ContinualModel):
    NAME = 'gcr'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super().__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device, mode=args.buffer_policy)
        self.weights = torch.ones(size=[self.args.buffer_size]).to(self.args.device)
        self.requires_indexes = True

    def begin_task(self, dataset):
        self.weights = torch.cat((self.weights, torch.ones(size=[len(dataset.train_loader.dataset)]).to(self.args.device)))

    def observe(self, inputs, labels, not_aug_inputs, dataset_indexes):
        real_batch_size = inputs.shape[0]
        batch_weights = self.weights[dataset_indexes]  # TODO add buffer size to indexes after first task

        self.opt.zero_grad()
        if not self.buffer.is_empty():
            buf_indexes, buf_inputs, buf_labels = self.buffer.get_data(self.args.minibatch_size, transform=self.transform, return_index=True)
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))
            buf_weights = self.weights[buf_indexes]
            batch_weights = torch.cat((batch_weights, buf_weights))

        outputs = self.net(inputs)
        loss = self.weighted_loss(outputs, labels, batch_weights)
        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels[:real_batch_size])

        return loss.item()

    def weighted_loss(self, outputs, labels, batch_weights):
        loss = self.loss(outputs, labels, reduction='none')
        loss = loss * batch_weights
        loss = torch.mean(loss)
        return loss

    def end_task(self, dataset):
        buf_inputs, buf_labels = self.buffer.get_all_data()

        train_dataset = dataset.train_loader.dataset
        dataset_inputs, dataset_labels = [], []
        for _, labels, not_aug_inputs, _ in train_dataset:
            dataset_inputs.append(not_aug_inputs)
            dataset_labels.append(labels)
        dataset_inputs = torch.stack(dataset_inputs).to(self.args.device)
        dataset_labels = torch.Tensor(dataset_labels).to(self.args.device).to(buf_labels.dtype)

        inputs = torch.cat([buf_inputs, dataset_inputs], dim=0)
        labels = torch.cat([buf_labels, dataset_labels], dim=0)

        self.opt.zero_grad()
        (buffer_inputs, buffer_labels), self.weights = self.grad_approx(inputs, labels, self.weights, lamb=0.0001, K=len(buf_inputs))
        self.opt.zero_grad()
        self.buffer.examples = buffer_inputs
        self.buffer.labels = buffer_labels

    def grad_approx(self, data_inputs, data_labels, Dw, lamb, K, epsilon=1e-6):
        """
        Implements the GradApprox algorithm for selecting a subset of data points
        that approximates the gradient of the entire dataset.

        Args:
            D (list of tuples): The dataset, where each tuple contains (xi, yi, zi).
            Dw (torch.Tensor): Existing weights for each data point in D.
            theta (torch.nn.Module): The model parameters.
            lamb (float): Regularization parameter.
            K (int): Budget for the subset size.
            epsilon (float): Tolerance for the approximation error.

        Returns:
            tuple: A tuple containing the selected subset X (list of tuples) and
                the corresponding weights Xw (torch.Tensor).
        """
        # 1. Initialize LabelCount(D) - Number of classes.
        Y = len(torch.unique(data_labels))

        # 2. Partition dataset D based on labels: D = {Dy}
        # 3. Partition dataset weights Dw based on the labels of data samples: W = {WDy}
        Dy_data_list = []
        Dy_labels_list = []
        WDy_list = []
        for y in range(Y):
            idx_y = [i for i, yi in enumerate(data_labels) if yi == y]
            idx_y = torch.Tensor(idx_y).to(self.args.device).to(torch.long)
            Dy_data_list.append(data_inputs[idx_y])  # [data_inputs[i] for i in idx_y])
            Dy_labels_list.append(data_labels[idx_y])  # [data_labels[i] for i in idx_y])

            WDy = Dw[idx_y]  # [Dw[i] for i in idx_y]
            WDy_list.append(WDy)

        # 4. Initialize Replay Buffer and Replay Buffer weights
        X_data = []
        X_labels = []
        Xw = torch.empty(0)

        # 5. Iterate over each class
        for y in range(Y):
            # 6. Initialize PerClass budget, subset, and weights
            Dy_data = Dy_data_list[y]
            Dy_labels = Dy_labels_list[y]
            WDy = WDy_list[y]
            ky = K // Y  # Integer division to ensure sum(ky) <= K
            Xy_data = []
            Xy_labels = []
            Wxy = torch.zeros(size=[ky])

            grad_Dy = self.compute_gradient(Dy_data, Dy_labels, WDy)  # tensor of size [grad_size]
            # print()
            # print(grad_Dy.shape) # shape = 11220132
            grad_X = self.compute_buffer_gradiens(Dy_data, Dy_labels)  # Xy_data, Xy_labels)  # tensor of size [curr Xy_data size x grad_size]

            # 7. Calculate residuals
            L_sub, residuals = self.calculate_residuals(grad_Dy, len(Dy_data), Xy_data, Xy_labels, Wxy, lamb)

            # 8. While subset size is less than the budget and approximation error is above tolerance
            while len(Xy_data) < ky and torch.norm(L_sub) >= epsilon:
                # 9. Find out maximum residual element
                max_residual_idx = torch.argmax(torch.abs(residuals))
                e_data, e_labels = Dy_data[max_residual_idx], Dy_labels[max_residual_idx]

                # 10. Update PerClass subset
                Xy_data.append(e_data)
                Xy_labels.append(e_labels)

                # 11. Calculate updated weights
                Wxy = self.calculate_updated_weights(Dy_data, Dy_labels, WDy, Xy_data, Xy_labels, Wxy, lamb)

                # 12. Calculate residuals
                L_sub, residuals = self.calculate_residuals(Dy_data, Dy_labels, WDy, Xy_data, Xy_labels, Wxy, lamb)

            # 13. Update subset and subset weights
            X_data.extend(Xy_data)
            X_labels.extend(Xy_labels)
            if len(Wxy) > 0:  # checking to avoid errors if Wxy is empty.
                Xw = torch.cat((Xw, Wxy))

        return X_data, X_labels, Xw

    def compute_gradient(self, Dy_data, Dy_labels, WDy):
        """compute gradient for full dataset, that will be target for estimation"""
        self.net.train()
        output = self.net(Dy_data)
        loss = self.weighted_loss(output, Dy_labels, WDy)
        loss.backward()
        grad_flat = torch.cat([param.grad.view(-1) for param in self.net.parameters()])
        self.net.zero_grad()
        grad_flat = grad_flat.detach()
        return grad_flat

    def compute_buffer_gradiens(self, Xy_data, Xy_labels):
        buffer_gradients = list()

        for i, (xi, yi) in enumerate(zip(Xy_data, Xy_labels)):
            xi = xi.clone().detach().requires_grad_(False)
            sample_grad = self.calculate_sample_gradient(xi, yi)
            buffer_gradients.append(sample_grad.cpu())

        buffer_gradients = torch.cat(buffer_gradients)
        print()
        print(buffer_gradients.shape)
        exit()
        return buffer_gradients

    def calculate_residuals(self, dataset_grad, num_dy, Xy_data, Xy_labels, Wxy, lamb):
        """Calculates the residuals for the GradApprox algorithm."""
        L_sub = torch.zeros(num_dy)
        residuals = torch.zeros(num_dy)

        for i in range(num_dy):
            # total_grad = self.calculate_gradient(Dy_data, Dy_labels, WDy)
            # if len(Xy_data) > 0:
            #     subset_grad = self.calculate_gradient(Xy_data, Xy_labels, Wxy)
            # else:
            #     subset_grad = torch.zeros_like(total_grad)

            L_sub[i] = torch.norm(dataset_grad - subset_grad, p=2) ** 2 + lamb * torch.norm(Wxy, p=2)
            residuals[i] = 2 * Wxy[i] * torch.norm(subset_grad) ** 2 + 2 * lamb * Wxy[i]
        return L_sub, residuals

    def calculate_gradient(self, input_data, labels_data, weights):
        """
        Calculates the gradient of the replay loss with respect to the model parameters.

        Args:
            data (list of tuples): The data points to calculate the gradient for.
            weights (torch.Tensor): The weights for each data point.
            theta (torch.nn.Module): The model parameters.

        Returns:
            torch.Tensor: The gradient of the replay loss.
        """
        total_grad = None
        for i, (xi, yi) in enumerate(zip(input_data, labels_data)):
            xi = xi.clone().detach().requires_grad_(False)
            sample_grad = self.calculate_sample_gradient(xi, yi)
            sample_grad = weights[i] * sample_grad

            if total_grad is None:
                total_grad = sample_grad
            else:
                total_grad += sample_grad

        return total_grad / len(input_data)

    def calculate_sample_gradient(self, x, y):
        """Calculates gradient for one sample"""
        self.net.train()
        output = self.net(x.unsqueeze(0))
        loss = self.loss(output, y.unsqueeze(0))
        loss.backward()
        grad_flat = torch.cat([param.grad.view(-1) for param in self.net.parameters()])
        self.net.zero_grad()
        grad_flat = grad_flat.detach()
        return grad_flat

    def calculate_updated_weights(self, Dy, WDy, Xy, Wxy, lamb):
        """Calculates the updated weights for the GradApprox algorithm."""
        num_xy = len(Xy)
        Wxy_updated = torch.zeros(num_xy)

        # loop through the samples in Xy.
        for i in range(num_xy):
            xi, _, _ = Xy[i]
            xi = torch.tensor(xi, requires_grad=False).float()
            Wxy_updated[i] = 1

        return Wxy_updated
