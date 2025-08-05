import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import collections
# import copy

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR100
# from tqdm import tqdm

from backbones.resnet import resnet18
from utils.conf import base_path_dataset as base_path


def main():
    # device = 'cuda:0'
    # net = resnet18(n_classes=100)
    # net.load_state_dict(torch.load('trained_weights/old_method/whole_dataset/resnet_cifar100.pth'))
    # net.to(device)

    # fvs, labels = extract_features(net, device)

    # class_fvs = collections.defaultdict(list)
    # for fv, label in zip(fvs, labels):
    #     class_fvs[label.item()].append(fv)
    # class_fvs = {label: torch.stack(fv) for label, fv in class_fvs.items()}

    # l2_distances = compute_l2_dist(fvs, labels, class_fvs)
    # l2_plot(l2_distances)

    # cos_distances = compute_cos_dist(fvs, labels, class_fvs)
    # cos_plot(cos_distances)

    # mahalanobis_distances = compute_mahalanobis_dist(fvs, labels, class_fvs)
    # mahalanobis_plot(mahalanobis_distances)

    # mahalanobis_norm_distances = compute_mahalanobis_norm_dist(fvs, labels, class_fvs)
    # mahalanobis_norm_plot(mahalanobis_norm_distances)

    # icarl_ranks_list = compute_icarl_ranks(labels, class_fvs)
    # icarl_plot(icarl_ranks_list)

    lass_plot()

    cw_plot()

    # feldman_plot()

    # training_iter_plot()
    # training_iter_vs_feldman_plot()

    plt.show()


def extract_features(net, device):
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])

    train_dataset = CIFAR100(root=base_path() + 'CIFAR100', train=True, transform=test_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=0)
    net.eval()

    fvs = []
    labels = []
    with torch.no_grad():
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            features = net.forward(inputs, returnt='features')
            fvs.append(features.to('cpu'))
            labels.append(targets)
            torch.cuda.empty_cache()

    fvs = torch.cat(fvs, dim=0)
    labels = torch.cat(labels, dim=0)
    return fvs, labels


def compute_l2_dist(fvs, labels, class_fvs):
    class_means = compute_means(class_fvs)

    l2_distances = []
    with torch.no_grad():
        for fv, label in zip(fvs, labels):
            mean_vector = class_means[label.item()]

            l2_dist = torch.dist(fv, mean_vector, p=2)
            l2_distances.append(l2_dist.item())

    l2_distances = torch.tensor(l2_distances)
    return l2_distances


def compute_cos_dist(fvs, labels, class_fvs):
    class_means = compute_means(class_fvs)

    cos_distances = []
    with torch.no_grad():
        for fv, label in zip(fvs, labels):
            mean_vector = class_means[label.item()]

            cos_dist = 1 - torch.cosine_similarity(fv, mean_vector, dim=0)
            cos_distances.append(cos_dist.item())

    cos_distances = torch.tensor(cos_distances)
    return cos_distances


def compute_mahalanobis_dist(fvs, labels, class_fvs):
    class_means = compute_means(class_fvs)
    class_covariances = compute_cov(class_fvs)

    mahalanobis_distances = []
    eps = 1e-6
    with torch.no_grad():
        for fv, label in zip(fvs, labels):
            mean_vector = class_means[label.item()]
            cov_matrix = class_covariances[label.item()]
            cov_matrix_ = cov_matrix + eps * torch.eye(cov_matrix.shape[0])
            inv_cov_matrix = torch.inverse(cov_matrix_)

            delta = fv - mean_vector
            distance = torch.sqrt(torch.dot(delta, inv_cov_matrix @ delta))
            mahalanobis_distances.append(distance.item())

    mahalanobis_distances = torch.tensor(mahalanobis_distances)
    return mahalanobis_distances


def compute_mahalanobis_norm_dist(fvs, labels, class_fvs):
    class_means = compute_means(class_fvs)
    class_covariances = compute_cov(class_fvs)

    mahalanobis_distances = []
    eps = 1e-6
    with torch.no_grad():
        for fv, label in zip(fvs, labels):
            mean_vector = class_means[label.item()]
            cov_matrix = class_covariances[label.item()]
            cov_matrix_ = cov_matrix + eps * torch.eye(cov_matrix.shape[0])
            inv_cov_matrix = torch.inverse(cov_matrix_)

            delta = fv - mean_vector
            distance = torch.sqrt(torch.dot(delta, inv_cov_matrix @ delta))
            mahalanobis_distances.append(distance.item())

    mahalanobis_distances = torch.tensor(mahalanobis_distances)

    class_distances = collections.defaultdict(list)
    for dist, label in zip(mahalanobis_distances, labels):
        class_distances[label.item()].append(dist)

    class_max_dist = []
    for label in sorted(class_distances.keys()):
        class_max_dist.append(max(class_distances[label]).item())

    mahalanobis_normalized_distances = []
    for dist, label in zip(mahalanobis_distances, labels):
        score = dist / class_max_dist[label.item()]
        mahalanobis_normalized_distances.append(score.item())
    mahalanobis_normalized_distances = torch.tensor(mahalanobis_normalized_distances)

    return mahalanobis_normalized_distances


def compute_icarl_ranks(labels, class_fvs):
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
            icarl_ranks[label] = icarl_class_ranks

    icarl_ranks_list = []
    icarl_iters = {label: iter(icarl_ranks[label]) for label in icarl_ranks}
    with torch.no_grad():
        for label in labels:
            rank = next(icarl_iters[label.item()])
            icarl_ranks_list.append(rank)
    return icarl_ranks_list


def compute_means(class_fvs):
    class_means = {}
    with torch.no_grad():
        for label, fv in class_fvs.items():
            mean = torch.mean(fv, dim=0)
            class_means[label] = mean
    return class_means


def compute_cov(class_fvs):
    class_covariances = {}
    with torch.no_grad():
        for label, fv in class_fvs.items():
            cov = torch.cov(fv.T)
            class_covariances[label] = cov
    return class_covariances


def l2_plot(l2_distances):
    mem_scores_old, mem_scores_old_values_correct = read_memorization_scres()
    mem_scores_old_values_correct = mem_scores_old_values_correct[88:]

    l2_dist_selected = []
    for i in mem_scores_old:
        l2_dist_selected.append(l2_distances[i])
    l2_dist_selected = l2_dist_selected[88:]

    plot_correlations(mem_scores_old_values_correct, l2_dist_selected, 'Original Memorization Scores', 'Euclidean Distances')


def cos_plot(cos_distances):
    mem_scores_old, mem_scores_old_values_correct = read_memorization_scres()
    mem_scores_old_values_correct = mem_scores_old_values_correct[88:]

    cos_dist_selected = []
    for i in mem_scores_old:
        cos_dist_selected.append(cos_distances[i])
    cos_dist_selected = cos_dist_selected[88:]

    plot_correlations(mem_scores_old_values_correct, cos_dist_selected, 'Original Memorization Scores', 'Cosine Distances')


def mahalanobis_plot(mahalanobis_distances):
    mem_scores_old, mem_scores_old_values_correct = read_memorization_scres()
    mem_scores_old_values_correct = mem_scores_old_values_correct[88:]

    mahalanobis_dist_selected = []
    for i in mem_scores_old:
        mahalanobis_dist_selected.append(mahalanobis_distances[i])
    mahalanobis_dist_selected = mahalanobis_dist_selected[88:]

    plot_correlations(mem_scores_old_values_correct, mahalanobis_dist_selected, 'Original Memorization Scores', 'Mahalanobis Distances')


def mahalanobis_norm_plot(mahalanobis_norm_distances):
    mem_scores_old, mem_scores_old_values_correct = read_memorization_scres()
    mem_scores_old_values_correct = mem_scores_old_values_correct[88:]

    mahalanobis_norm_dist_selected = []
    for i in mem_scores_old:
        mahalanobis_norm_dist_selected.append(mahalanobis_norm_distances[i])
    mahalanobis_norm_dist_selected = mahalanobis_norm_dist_selected[88:]

    plot_correlations(mem_scores_old_values_correct, mahalanobis_norm_dist_selected, 'Original Memorization Scores', 'Normalized Mahalanobis Distances')


def icarl_plot(icarl_ranks):
    mem_scores_old, mem_scores_old_values_correct = read_memorization_scres()
    mem_scores_old_values_correct = mem_scores_old_values_correct[88:]

    icarl_ranks_selected = []
    for i in mem_scores_old:
        icarl_ranks_selected.append(icarl_ranks[i])
    icarl_ranks_selected = icarl_ranks_selected[88:]

    plot_correlations(mem_scores_old_values_correct, icarl_ranks_selected, 'Original Memorization Scores', 'iCaRL Ranks')


def lass_plot():
    mem_scores_old, mem_scores_old_values_correct = read_memorization_scres()
    mem_scores_old_values_correct = mem_scores_old_values_correct[88:]

    diffs_LASS = np.load('LASS_dsitance.npy')
    diffs_selected = []
    for i in mem_scores_old:
        diffs_selected.append(diffs_LASS[i])
    diffs_selected = diffs_selected[88:]
    diffs_selected = np.array(diffs_selected)

    plot_correlations(mem_scores_old_values_correct, diffs_selected, 'Original Memorization Scores', 'LASS Distances')


def cw_plot():
    mem_scores_old, mem_scores_old_values_correct = read_memorization_scres()
    mem_scores_old_values_correct = mem_scores_old_values_correct[88:]

    diffs_carlini = np.load('cw_attack.npy')
    diffs_selected = []
    for i in mem_scores_old:
        diffs_selected.append(diffs_carlini[i])
    diffs_selected = diffs_selected[88:]
    diffs_selected = np.array(diffs_selected)

    plot_correlations(mem_scores_old_values_correct, diffs_selected, 'Original Memorization Scores', 'Carlini-Wagner Distances')


def feldman_plot():
    mem_scores_old, mem_scores_old_values_correct = read_memorization_scres()
    mem_scores_old_values_correct = mem_scores_old_values_correct[88:]

    memorisation_scores_whole = np.load('datasets/memorsation_scores_cifar100.npy')
    memorisation_scores_whole_selected = []
    for i in mem_scores_old:
        memorisation_scores_whole_selected.append(memorisation_scores_whole[i])
    memorisation_scores_whole_selected = memorisation_scores_whole_selected[88:]

    plot_correlations(mem_scores_old_values_correct, memorisation_scores_whole_selected, 'Original Memorization Scores', 'Feldman Estimates')


def training_iter_plot():
    mem_scores_old, mem_scores_old_values_correct = read_memorization_scres()
    mem_scores_old_values_correct = mem_scores_old_values_correct[88:]

    trained_order = np.load('trained_order.npy')
    trained_iteration = np.load('trained_iteration.npy')
    # print('trained_order = ', trained_order)
    # print('trained_iteration = ', trained_iteration)
    # print(trained_order.shape)
    # print(trained_iteration.shape)

    trained_iter = np.zeros(50000)
    for i, iter in zip(trained_order, trained_iteration):
        trained_iter[i] = iter

    trained_iter_selected = []
    for i in mem_scores_old:
        trained_iter_selected.append(trained_iter[i])
    trained_iter_selected = trained_iter_selected[88:]
    # print(trained_iter_selected)

    plot_correlations(mem_scores_old_values_correct, trained_iter_selected, 'Original Memorization Scores', 'Training Iterations')


def training_iter_vs_feldman_plot():
    mem_scores_old, _ = read_memorization_scres()

    trained_order = np.load('trained_order.npy')
    trained_iteration = np.load('trained_iteration.npy')

    trained_iter = np.zeros(50000)
    for i, iter in zip(trained_order, trained_iteration):
        trained_iter[i] = iter

    trained_iter_selected = []
    for i in mem_scores_old:
        trained_iter_selected.append(trained_iter[i])
    trained_iter_selected = trained_iter_selected[88:]

    memorisation_scores_whole = np.load('datasets/memorsation_scores_cifar100.npy')
    memorisation_scores_whole_selected = []
    for i in mem_scores_old:
        memorisation_scores_whole_selected.append(memorisation_scores_whole[i])
    memorisation_scores_whole_selected = memorisation_scores_whole_selected[88:]

    plot_correlations(memorisation_scores_whole_selected, trained_iter_selected, 'Feldman Estimator', 'Training Iterations')


def read_memorization_scres():
    memorisation_file_path = 'leave-one-out/memorisation.txt'
    mem_scores_old = {}
    with open(memorisation_file_path, 'r') as f:
        for line in f.readlines():
            idx, prob = line.split(':')
            idx, prob = int(idx), float(prob)
            mem_scores_old[idx] = prob

    leave_one_out_probs = np.load('leave-one-out/orginal_probs.npy')
    leave_one_out_probs_selected = []
    for i in mem_scores_old:
        leave_one_out_probs_selected.append(leave_one_out_probs[i])

    mem_scores_old_values = list(mem_scores_old.values())
    mem_scores_old_values_correct = leave_one_out_probs_selected - np.array(mem_scores_old_values)

    return mem_scores_old, mem_scores_old_values_correct


def plot_correlations(data_x, data_y, x_label, y_label):
    correlation = np.corrcoef(data_x, data_y)[0, 1]
    print()
    print(f'Corelations {x_label} vs {y_label}')
    print("Correlation Coefficient:", correlation)

    # Plot mem_scores_old_values against icarl_ranks_selected
    plt.figure(figsize=(8, 6))
    plt.scatter(data_x, data_y, alpha=0.7)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # plt.title('Scatter Plot of orginal mem scores vs icarl ranks')
    plt.grid(True)


if __name__ == '__main__':
    main()
