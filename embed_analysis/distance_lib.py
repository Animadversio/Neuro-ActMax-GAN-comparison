
import torch
import math
import numpy as np
import pickle as pkl
from os.path import join
import matplotlib.pyplot as plt
# compute mahalanobis distance
def mahalanobis_sqdist(x, mean, eigvec, eigval, eigfloor=1e-5, device="cuda"):
    x = x.to(device) - mean[None, :].to(device)
    # rotate
    rot_x = x @ eigvec
    # scale
    return (rot_x ** 2 / torch.clamp(eigval[None, :].to(device), min=eigfloor)).sum(dim=1)


# plot functions to collect activation / distace according to gens. 
def compute_mean_var(x, gen_vec, var="std"):
    meanvec = []
    errvec = []
    for gen in range(gen_vec.min(), gen_vec.max()+1):
        meanvec.append(x[gen_vec==gen].mean())
        if var == "std":
            errvec.append(x[gen_vec==gen].std())
        elif var == "var":
            errvec.append(x[gen_vec==gen].var())
        elif var == "sem":
            errvec.append(x[gen_vec==gen].std() / np.sqrt((gen_vec==gen).sum()))
    return torch.tensor(meanvec), torch.tensor(errvec)


def plot_shaded_errorbar(x, y, color="b", label="", var="std", **kwargs):
    meanvec, errvec = compute_mean_var(y, x, var=var)
    plt.plot(np.unique(x), meanvec, color=color, label=label, **kwargs)
    plt.fill_between(np.unique(x), meanvec-errvec, meanvec+errvec, color=color, alpha=0.3)

def gaussian_nll_with_eig(x, mean, eigvals, eigvecs, eigfloor=1E-3, device="cuda"):
    """
    Calculate the Gaussian negative log likelihood of x given the mean, eigenvalues, and eigenvectors.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
        mean (torch.Tensor): Mean tensor of shape (input_dim,).
        eigvals (torch.Tensor): Eigenvalues tensor of shape (input_dim,).
        eigvecs (torch.Tensor): Eigenvectors tensor of shape (input_dim, input_dim).

    Returns:
        torch.Tensor: Gaussian negative log likelihood tensor of shape (batch_size,).
    """
    input_dim = x.size(1)
    diff = x - mean
    eigvals_clamped = torch.clamp(eigvals.to(device), min=eigfloor)
    mahalanobis = torch.matmul(diff, eigvecs)
    mahalanobis = mahalanobis**2 / eigvals_clamped
    mahalanobis = torch.sum(mahalanobis, dim=1)
    log_det = torch.sum(torch.log(eigvals_clamped))
    nll = 0.5 * (input_dim * math.log(2 * math.pi) + log_det + mahalanobis)
    return nll


def dist2k_nearest_neighbor_w_index(probe_embed, ref_embed, k_list, device="cuda"):
    """
    Calculate the Euclidean distance between a probe embedding and the k nearest neighbors in a reference embedding.

    Args:
        probe_embed (torch.Tensor): Embedding tensor of shape (probe_size, embedding_dim).
        ref_embed (torch.Tensor): Reference embedding tensor of shape (ref_size, embedding_dim).
        k_list (list): List of integers specifying the number of nearest neighbors to consider.
        device (str): Device to perform the calculations on (default is "cuda").

    Returns:
        dict: Dictionary containing the distances to the k nearest neighbors for each probe embedding.
    """
    probe_embed = probe_embed.to(device)
    ref_embed = ref_embed.to(device)
    dists = torch.cdist(probe_embed, ref_embed,
                            p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')
    max_k = max(k_list)
    k_nearest, indices = torch.topk(dists, max_k, largest=False, dim=-1)
    distances = {}
    for k in k_list:
        distances[k] = k_nearest[:, k-1]
    return distances, indices


def dist2k_nearest_neighbor_mahalanobis_w_index(probe_embed, ref_embed, k_list, eigvecs, eigvals, eigfloor=1E-3, device="cuda"):
    """
    Calculate the Euclidean distance between a probe embedding and the k nearest neighbors in a reference embedding.

    Args:
        probe_embed (torch.Tensor): Embedding tensor of shape (probe_size, embedding_dim).
        ref_embed (torch.Tensor): Reference embedding tensor of shape (ref_size, embedding_dim).
        k_list (list): List of integers specifying the number of nearest neighbors to consider.
        device (str): Device to perform the calculations on (default is "cuda").

    Returns:
        dict: Dictionary containing the distances to the k nearest neighbors for each probe embedding.
    """
    probe_embed = probe_embed.to(device)
    ref_embed = ref_embed.to(device)
    scaling = torch.sqrt(torch.clamp(eigvals.to(device), min=eigfloor))
    probe_embed_rot = (probe_embed @ eigvecs) / scaling
    ref_embed_rot = (ref_embed @ eigvecs) / scaling
    dists = torch.cdist(probe_embed_rot, ref_embed_rot,
                            p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')
    max_k = max(k_list)
    k_nearest, indices = torch.topk(dists, max_k, largest=False, dim=-1)
    distances = {}
    for k in k_list:
        distances[k] = k_nearest[:, k-1]
    return distances, indices


def dist2k_nearest_neighbor_cosine_w_index(probe_embed, ref_embed, k_list, device="cuda"):
    """
    Calculate the 1 - Cosine Similarity distance between a probe embedding and the k nearest neighbors in a reference embedding.

    Args:
        probe_embed (torch.Tensor): Embedding tensor of shape (probe_size, embedding_dim).
        ref_embed (torch.Tensor): Reference embedding tensor of shape (ref_size, embedding_dim).
        k_list (list): List of integers specifying the number of nearest neighbors to consider.
        device (str): Device to perform the calculations on (default is "cuda").

    Returns:
        dict: Dictionary containing the distances to the k nearest neighbors for each probe embedding.
    """
    probe_embed = probe_embed.to(device)
    ref_embed = ref_embed.to(device)
    similarity = torch.matmul(probe_embed, ref_embed.T)
    similarity = similarity / torch.norm(probe_embed, dim=1)[:, None]
    similarity = similarity / torch.norm(ref_embed, dim=1)[None, :]
    dists = 1 - similarity
    max_k = max(k_list)
    k_nearest, indices = torch.topk(dists, max_k, largest=False, dim=-1)
    distances = {}
    for k in k_list:
        distances[k] = k_nearest[:, k-1]
    return distances, indices


GANembed_root = "/n/holylabs/LABS/kempner_fellows/Users/binxuwang/Projects/GAN_img_embedding"
GANembed_dir = join(GANembed_root, "dinov2_vitb14")
Evol_embed_dir = '/n/holylabs/LABS/kempner_fellows/Users/binxuwang/Projects/BigGAN_img_embedding/dinov2_vitb14'

def load_compute_eigen(dataset_str):
    if dataset_str == "imagenet_valid":
        embed = pkl.load(open(join(GANembed_dir, dataset_str+"_embedding.pkl"), "rb"))
    else:
        _, embed = pkl.load(open(join(GANembed_dir, dataset_str+"_embedding.pkl"), "rb"))
    embed = embed.cuda()
    cov = torch.cov(embed.T, )
    data_mean = embed.mean(dim=0)
    data_eigvals, data_eigvecs = torch.linalg.eigh(cov)
    data_eigvals = torch.flip(data_eigvals, dims=[0])
    data_eigvecs = torch.flip(data_eigvecs, dims=[1])
    return (data_mean, data_eigvals, data_eigvecs)

def load_dataset_embed(dataset_str):
    if dataset_str == "imagenet_valid":
        embed = pkl.load(open(join(GANembed_dir, dataset_str+"_embedding.pkl"), "rb"))
    else:
        _, embed = pkl.load(open(join(GANembed_dir, dataset_str+"_embedding.pkl"), "rb"))
    return embed

def compute_eigen(embed):
    embed = embed.cuda()
    cov = torch.cov(embed.T, )
    data_mean = embed.mean(dim=0)
    data_eigvals, data_eigvecs = torch.linalg.eigh(cov)
    data_eigvals = torch.flip(data_eigvals, dims=[0])
    data_eigvecs = torch.flip(data_eigvecs, dims=[1])
    return (data_mean, data_eigvals, data_eigvecs)