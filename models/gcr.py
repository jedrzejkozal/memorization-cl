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
    parser.add_argument('--buffer_policy', choices=['balanced_reservoir', 'reservoir'], default='reservoir', help='policy for selecting samples stored into buffer')
    return parser


class Gcr(ContinualModel):
    NAME = 'gcr'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super().__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device, mode=args.buffer_policy)

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
        train_dataset = dataset.train_loader.dataset


def grad_approx(D, Dw, theta, lamb, K, epsilon):
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
    labels = set(yi for _, yi, _ in D)
    Y = len(labels)

    # 2. Partition dataset D based on labels: D = {Dy}
    # 3. Partition dataset weights Dw based on the labels of data samples: W = {WDy}
    Dy_list = []
    WDy_list = []
    for y in range(Y):
        idx_y = [i for i, (_, yi, _) in D if yi == y]
        Dy = [D[i] for i in idx_y]
        Dy_list.append(Dy)

        WDy = [Dw[i] for i in idx_y]
        WDy_list.append(WDy)

    # 4. Initialize Replay Buffer and Replay Buffer weights
    X = []
    Xw = torch.empty(0)

    # 5. Iterate over each class
    for y in range(Y):
        # 6. Initialize PerClass budget, subset, and weights
        Dy = Dy_list[y]
        WDy = WDy_list[y]
        ky = K // Y  # Integer division to ensure sum(ky) <= K
        Xy = []
        Wxy = torch.empty(0)

        # 7. Calculate residuals
        residuals = calculate_residuals(Dy, WDy, Xy, Wxy, theta, lamb)

        # 8. While subset size is less than the budget and approximation error is above tolerance
        while len(Xy) < ky and torch.norm(residuals) >= epsilon:
            # 9. Find out maximum residual element
            max_residual_idx = torch.argmax(torch.abs(residuals))
            e = Dy[max_residual_idx]

            # 10. Update PerClass subset
            Xy.append(e)

            # 11. Calculate updated weights
            Wxy = calculate_updated_weights(Dy, WDy, Xy, Wxy, theta, lamb)

            # 12. Calculate residuals
            residuals = calculate_residuals(Dy, WDy, Xy, Wxy, theta, lamb)

        # 13. Update subset and subset weights
        X.extend(Xy)
        if len(Wxy) > 0:  # checking to avoid errors if Wxy is empty.
            Xw = torch.cat((Xw, Wxy))

    return X, Xw


def calculate_residuals(Dy, WDy, Xy, Wxy, theta, lamb):
    """Calculates the residuals for the GradApprox algorithm."""
    if not Dy:  # returns an empty tensor if Dy is empty
        return torch.empty(0)

    num_dy = len(Dy)
    residuals = torch.zeros(num_dy)

    for j in range(num_dy):
        dj, _, _ = Dy[j]  # get data point xi from data point d = (xi,yi,zi)

        # convert xi to a tensor (assuming it is a numpy array).
        # dj = torch.tensor(dj, requires_grad=False).float()
        total_grad = calculate_gradient(Dy, WDy, theta)
        subset_grad = calculate_gradient(Xy, Wxy, theta)

        # residuals for data point j.
        residuals[j] = (torch.norm(WDy[j] * calculate_sample_gradient(dj, theta) - subset_grad))
    return residuals


def calculate_gradient(data, weights, theta):
    """
    Calculates the gradient of the replay loss with respect to the model parameters.

    Args:
        data (list of tuples): The data points to calculate the gradient for.
        weights (torch.Tensor): The weights for each data point.
        theta (torch.nn.Module): The model parameters.

    Returns:
        torch.Tensor: The gradient of the replay loss.
    """
    if not data:  # Returns zero gradients if data is empty.
        return torch.zeros_like(next(theta.parameters()))

    total_grad = None  # accumulate all gradients.
    for i, (xi, yi, zi) in enumerate(data):
        xi = torch.tensor(xi, requires_grad=False).float()
        sample_grad = calculate_sample_gradient(xi, theta)
        if weights is not None:
            sample_grad = weights[i] * sample_grad  # Apply weight.

        if total_grad is None:
            total_grad = sample_grad
        else:
            total_grad += sample_grad

    return total_grad / len(data)


def calculate_sample_gradient(x, theta):
    """Calculates gradient for one sample"""
    # set x to require gradients.
    x = x.requires_grad_(True)

    # set model to train, to enable grad calc.
    theta.train()
    output = theta(x)

    # calculate a dummy loss - CHANGE THIS to the actual loss function
    # criterion = torch.nn.CrossEntropyLoss() # or similar
    # loss = criterion(output, torch.tensor([y]))  # y is a single label

    # example of L2 Loss:
    loss = torch.sum(output**2)

    # compute gradients
    loss.backward()

    # get gradients - extract the gradients from the parameters
    # grads = [param.grad.clone().detach() for param in theta.parameters()]
    # return grads
    # flatten gradients:
    grad_flat = torch.cat([param.grad.view(-1) for param in theta.parameters()])

    # IMPORTANT - set gradients to zero before the next iteration
    theta.zero_grad()

    # detach so that the sample grad calculation is not retained in memory.
    return grad_flat.detach()


def calculate_updated_weights(Dy, WDy, Xy, Wxy, theta, lamb):
    """Calculates the updated weights for the GradApprox algorithm."""
    num_xy = len(Xy)
    Wxy_updated = torch.zeros(num_xy)

    # loop through the samples in Xy.
    for i in range(num_xy):
        xi, _, _ = Xy[i]
        xi = torch.tensor(xi, requires_grad=False).float()
        Wxy_updated[i] = 1

    return Wxy_updated
