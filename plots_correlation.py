import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, kendalltau
import torch
import torch.nn as nn
import collections

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR100

from backbones.resnet import resnet18
from utils.conf import base_path_dataset as base_path


def main():
    df = preprocess_data()

    regression_plot(
        x_data=df["mem_original"],
        y_data=df["l2"],
        x_label="Original Memorization Scores",
        y_label="Euclidean Distances",
        plot_filename="figures/original_mem_scores_vs_l2.pdf",
    )

    regression_plot(
        x_data=df["mem_original"],
        y_data=df["cosine"],
        x_label="Original Memorization Scores",
        y_label="Cosine Distances",
        plot_filename="figures/original_mem_scores_vs_cosine.pdf",
    )

    regression_plot(
        x_data=df["mem_original"],
        y_data=df["mahalanobis"],
        x_label="Original Memorization Scores",
        y_label="Mahalanobis Distances",
        plot_filename="figures/original_mem_scores_vs_mahalanobis.pdf",
    )

    regression_plot(
        x_data=df["mem_original"],
        y_data=df["mahalanobis_norm"],
        x_label="Original Memorization Scores",
        y_label="Normalized Mahalanobis Distances",
        plot_filename="figures/original_mem_scores_vs_mahalanobis_norm.pdf",
    )

    regression_plot(
        x_data=df["mem_original"],
        y_data=df["icarl_rank"],
        x_label="Original Memorization Scores",
        y_label="iCaRL Ranks",
        plot_filename="figures/original_mem_scores_vs_icarl.pdf",
    )

    regression_plot(
        x_data=df["mem_original"],
        y_data=df["lass"],
        x_label="Original Memorization Scores",
        y_label="LASS Distances",
        plot_filename="figures/original_mem_scores_vs_lass.pdf",
    )

    regression_plot(
        x_data=df["mem_original"],
        y_data=df["carlini_wagner"],
        x_label="Original Memorization Scores",
        y_label="Carlini-Wagner Distances",
        plot_filename="figures/original_mem_scores_vs_cw.pdf",
    )

    regression_plot(
        x_data=df["mem_original"],
        y_data=df["mem_feldman"],
        x_label="Original Memorization Scores",
        y_label="Feldman Estimates",
        plot_filename="figures/original_mem_scores_vs_feldman.pdf",
    )

    regression_plot(
        x_data=df["mem_original"],
        y_data=df["trained_iter"],
        x_label="Original Memorization Scores",
        y_label="Training Iterations",
        plot_filename="figures/original_mem_scores_vs_training_iter.pdf",
    )

    regression_plot(
        x_data=df["mem_feldman"],
        y_data=df["trained_iter"],
        x_label="Feldman Estimates",
        y_label="Training Iterations",
        plot_filename="figures/feldman_vs_training_iter.pdf",
    )


def preprocess_data():
    device = 'cuda:0'
    net = resnet18(n_classes=100)
    net.load_state_dict(torch.load('trained_weights/old_method/whole_dataset/resnet_cifar100.pth'))
    net.to(device)

    fvs, labels = extract_features(net, device)
    class_fvs = collections.defaultdict(list)
    for fv, label in zip(fvs, labels):
        class_fvs[label.item()].append(fv)
    class_fvs = {label: torch.stack(fv) for label, fv in class_fvs.items()}

    df = pd.read_csv("leave-one-out/memorisation.txt", sep=":", names=["idx", "p_full"])

    # Leave-one-out probabilities
    p_leave = np.load("leave-one-out/orginal_probs.npy")
    df["p_leave_one_out"] = df["idx"].map(dict(enumerate(p_leave)))

    # Original memorization scores
    df["mem_original"] = df["p_leave_one_out"] - df["p_full"]

    # Euclidean distances
    l2_distances = compute_l2_dist(fvs, labels, class_fvs)
    df["l2"] = df["idx"].map(dict(enumerate(l2_distances.tolist())))

    # Cosine distances
    cos_distances = compute_cos_dist(fvs, labels, class_fvs)
    df["cosine"] = df["idx"].map(dict(enumerate(cos_distances.tolist())))

    # Mahalanobis distances
    mahalanobis_distances = compute_mahalanobis_dist(fvs, labels, class_fvs)
    df["mahalanobis"] = df["idx"].map(dict(enumerate(mahalanobis_distances.tolist())))

    # Normalized Mahalanobis distances
    mahalanobis_norm_distances = compute_mahalanobis_norm_dist(fvs, labels, class_fvs)
    df["mahalanobis_norm"] = df["idx"].map(dict(enumerate(mahalanobis_norm_distances.tolist())))

    # iCaRL ranks
    icarl_ranks_list = compute_icarl_ranks(labels, class_fvs)
    df["icarl_rank"] = df["idx"].map(dict(enumerate(icarl_ranks_list)))

    # Lass distances
    lass_distances: np.array = np.load('LASS_dsitance.npy')
    df["lass"] = df["idx"].map(dict(enumerate(lass_distances.tolist())))

    # Carlini Wagner distances
    carlini_wagner = np.load('cw_attack.npy')
    df["carlini_wagner"] = df["idx"].map(dict(enumerate(carlini_wagner.tolist())))

    # Feldman estimates
    mem_feldman = np.load("datasets/memorsation_scores_cifar100.npy")
    df["mem_feldman"] = df["idx"].map(dict(enumerate(mem_feldman.tolist())))

    # Trained iter estimates
    trained_order = np.load("trained_order.npy")
    trained_iteration = np.load("trained_iteration.npy")
    df["trained_iter"] = df["idx"].map(dict(zip(trained_order.tolist(), trained_iteration.tolist())))

    return df[88:]


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


def regression_plot(
    x_data: pd.Series,
    y_data: pd.Series,
    x_label: str = "",
    y_label: str = "",
    x_lim: tuple = None,
    y_lim: tuple = None,
    plot_filename: str = "",
):
    pearson, _ = pearsonr(x_data, y_data)
    spearman, _ = spearmanr(x_data, y_data)
    kendall, _ = kendalltau(x_data, y_data)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 6))
    sns.regplot(
        x=x_data,
        y=y_data,
        scatter_kws={"s": 15, "alpha": 0.75},
    )

    plt.xlabel(x_label, fontsize=16)
    plt.xticks(fontsize=14)
    plt.ylabel(y_label, fontsize=16)
    plt.yticks(fontsize=14)
    if x_lim is not None:
        plt.xlim(x_lim)
    if y_lim is not None:
        plt.ylim(y_lim)

    legends = [
        Line2D([], [], linestyle="", label=f"Pearson, r = {pearson:.3f}"),
        Line2D([], [], linestyle="", label=f"Spearman, ρ = {spearman:.3f}"),
        Line2D([], [], linestyle="", label=f"Kendall, τ = {kendall:.3f}"),
    ]

    plt.legend(
        handles=legends,
        title="Correlation Coefficients",
        title_fontsize=14,
        fontsize=14,
        loc="lower right",
        handletextpad=-3,
    )

    plt.tight_layout()
    if plot_filename != "" and len(plot_filename) > 0:
        plt.savefig(plot_filename, format="pdf", dpi=300)


if __name__ == '__main__':
    main()
