import numpy as np
import pandas as pd
import pyro.distributions as dist
from scipy import stats
import torch
import seaborn as sns

import matplotlib.pyplot as plt
from scipy.stats import beta, pareto
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns

import copy
import json

import argparse

import sys
import os
# Set the parent directory
parent_dir = "../../"
sys.path.insert(0, parent_dir)
import new_model as mobster_mv
from utils.plot_functions import *
from utils.BoundedPareto import BoundedPareto
from utils.create_beta_pareto_dataset import *

def convert_to_list(item):
    """
    Recursively converts all NumPy arrays and PyTorch tensors in a dictionary or list
    to Python lists.
    """
    if isinstance(item, dict):
        return {key: convert_to_list(value) for key, value in item.items()}
    elif isinstance(item, list):
        return [convert_to_list(element) for element in item]
    elif isinstance(item, np.ndarray):  # Check if it's a NumPy array
        return item.tolist()
    elif isinstance(item, torch.Tensor):  # Check if it's a PyTorch tensor
        return item.detach().cpu().tolist()  # Detach from computation graph and convert to list
    else:
        return item

def create_folder(N,K,D,purity,coverage):
    
    folder_path = f'results/p_{str(purity).replace(".", "")}_cov_{coverage}/D_{D}/nmi'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    folder_path = f'results/p_{str(purity).replace(".", "")}_cov_{coverage}/D_{D}/ari'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    folder_path = f'results/p_{str(purity).replace(".", "")}_cov_{coverage}/D_{D}/csv'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    
    folder_path = f'plots/p_{str(purity).replace(".", "")}_cov_{coverage}/D_{D}/inference'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    folder_path = f'plots/p_{str(purity).replace(".", "")}_cov_{coverage}/D_{D}/real'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    folder_path = f'plots/p_{str(purity).replace(".", "")}_cov_{coverage}/D_{D}/real_marginals'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    folder_path = f'plots/p_{str(purity).replace(".", "")}_cov_{coverage}/D_{D}/inference_marginals'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    folder_path = f'plots/p_{str(purity).replace(".", "")}_cov_{coverage}/D_{D}/model_selection'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    folder_path = f'plots/p_{str(purity).replace(".", "")}_cov_{coverage}/D_{D}/betas_paretos'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    folder_path = f'plots/p_{str(purity).replace(".", "")}_cov_{coverage}/D_{D}/loss_lks'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    folder_path = f'plots/p_{str(purity).replace(".", "")}_cov_{coverage}/D_{D}/responsib_deltas'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    folder_path = f'saved_objects/p_{str(purity).replace(".", "")}_cov_{coverage}/D_{D}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def plot_final(mb, N, K, idx_real, purity, coverage):
    NV = mb.NV
    DP = mb.DP

    unique_labels = np.unique(mb.params["cluster_assignments"].detach().numpy())
    labels = mb.params["cluster_assignments"].detach().numpy()
    # cmap = cm.get_cmap('tab20')
    # color_mapping = {label: cmap(i) for i, label in enumerate(unique_labels)}
    # cmap = cm.get_cmap('tab20')
    plt.figure()
    plt.xlim([0,1])
    plt.ylim([0,1])

    delta = mb.params["delta_param"]
    pred_dist = []
    for k in range(mb.K):
        dist_d = []
        for d in range(mb.NV.shape[1]):
            delta_kd = delta[k, d]
            maxx = torch.argmax(delta_kd)
            if maxx == 0: # Pareto
                dist_d.append(0)
                # pred_dist[k,d] = 0
            else:
                dist_d.append(1)
                # pred_dist[k,d] = 1
        pred_dist.append(dist_d)
        
    D = NV.shape[1]
    pairs = np.triu_indices(D, k=1)  # Generate all unique pairs of samples (i, j)
    vaf = NV/DP    
    num_pairs = len(pairs[0])  # Number of unique pairs
    ncols = min(3, num_pairs)
    nrows = (num_pairs + ncols - 1) // ncols  # Calculate the number of rows

    fig_width_per_plot = 5
    fig_width = ncols * fig_width_per_plot
    fig_height = 5 * nrows

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))
    if num_pairs == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    idx = 0
    for i, j in zip(*pairs):
        ax = axes[idx]  # Select the appropriate subplot
        x = vaf[:, i].numpy()
        y = vaf[:, j].numpy()

        for c, cluster in enumerate(np.unique(labels)):
            mask = (labels == cluster)  # Mask for current cluster
            ax.scatter(x[mask], 
                        y[mask],
                        label=f'{cluster.astype("int")} {pred_dist[c]}')
        ax.legend(loc='best')
        ax.set_title(f'Sample {i+1} vs Sample {j+1}')
        ax.set_xlabel(f'Sample {i+1}')
        ax.set_ylabel(f'Sample {j+1}')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        idx += 1
    plt.suptitle(f'Inference with N = {N} and {K} clusters (i = {idx_real}) \n ')
    plt.show()
    plt.savefig(f'plots/p_{str(purity).replace(".", "")}_cov_{coverage}/D_{D}/inference/N_{N}_K_{K}_D_{mb.NV.shape[1]}_inference_{idx_real}.png')
    plt.close()

def plot_final_marginals(mb, N, K, D, idx_real, purity, coverage):
    
    delta = mb.params["delta_param"]  # K x D x 2
    phi_beta = mb.params["phi_beta_param"]
    if torch.is_tensor(phi_beta):
        phi_beta = phi_beta.detach().numpy()
    else:
        phi_beta = np.array(phi_beta)
    
    kappa_beta = mb.params["k_beta_param"]
    if torch.is_tensor(kappa_beta):
        kappa_beta = kappa_beta.detach().numpy()
    else:
        kappa_beta = np.array(kappa_beta)

    alpha = mb.params["alpha_pareto_param"]
    if torch.is_tensor(alpha):
        alpha = alpha.detach().numpy()
    else:
        alpha = np.array(alpha)
    
    weights = mb.params["weights_param"]
    if torch.is_tensor(weights):
        weights = weights.detach().numpy()
    else:
        weights = np.array(weights)
        
    labels = mb.params['cluster_assignments']
    if torch.is_tensor(labels):
        labels = labels.detach().numpy()
    else:
        labels = np.array(labels)
    if mb.K == 1:
        fig, axes = plt.subplots(mb.K, mb.NV.shape[1], figsize=(16, 4))
    else:
        fig, axes = plt.subplots(mb.K, mb.NV.shape[1], figsize=(16, mb.K*3))
    if mb.K == 1:
        axes = ax = np.array([axes])  # add an extra dimension to make it 2D
    plt.suptitle(f'Marginals with K={mb.K}, seed={mb.seed}',fontsize=14)
    x = np.linspace(0.001, 1, 1000)

    unique_labels = np.unique(labels)
    cmap = cm.get_cmap('tab20')
    color_mapping = {label: cmap(i) for i, label in enumerate(unique_labels)}
    for k in range(mb.K):
        for d in range(mb.NV.shape[1]):
            delta_kd = delta[k, d]
            maxx = torch.argmax(delta_kd)
            if maxx == 1:
                # plot beta
                a = phi_beta[k,d] * kappa_beta[k,d]
                b = (1-phi_beta[k,d]) * kappa_beta[k,d]
                pdf = beta.pdf(x, a, b)# * weights[k]
                axes[k,d].plot(x, pdf, linewidth=1.5, label='Beta', color='r')
                axes[k,d].legend()
            elif maxx == 0:
                # plot pareto
                pdf = pareto.pdf(x, alpha[k,d], scale=mb.pareto_L) #* weights[k]
                axes[k,d].plot(x, pdf, linewidth=1.5, label='Pareto', color='g')
                axes[k,d].legend()
            else:
                # private
                pdf = beta.pdf(x, mb.a_beta_zeros, mb.b_beta_zeros) # delta_approx
                axes[k,d].plot(x, pdf, linewidth=1.5, label='Zeros', color='b')
                axes[k,d].legend()

            if torch.is_tensor(mb.NV):
                data = mb.NV[:,d].numpy()/mb.DP[:,d].numpy()
            else:
                data = np.array(mb.NV[:,d])/np.array(mb.DP[:,d])
            # for i in np.unique(labels):
            if k in unique_labels:
                # axes[k, d].hist(data[labels == k],  density=True, bins=30, alpha=1, color=color_mapping[k])
                axes[k,d].hist(data[labels == k], density=True, bins=30, alpha=0.5)
            else:
                # Plot an empty histogram because we know there are no points in that k
                axes[k, d].hist([], density=True, bins=30, alpha=1)
            axes[k,d].set_title(f'Sample {d+1} - Cluster {k}')
            axes[k,d].grid(True, color='gray', linestyle='-', linewidth=0.2)
            axes[k,d].set_xlim([-0.01,0.8])
            plt.tight_layout()
    plt.show()
    plt.savefig(f'plots/p_{str(purity).replace(".", "")}_cov_{coverage}/D_{D}/inference_marginals/N_{N}_K_{K}_D_{mb.NV.shape[1]}_inference_{idx_real}.png')
    plt.close()


def plot_scatter_real(NV, DP, N, K, D, type_labels_cluster, cluster_labels, idx_real, purity, coverage):
    pairs = np.triu_indices(D, k=1)  # Generate all unique pairs of samples (i, j)
    vaf = NV/DP    
    num_pairs = len(pairs[0])  # Number of unique pairs
    ncols = min(3, num_pairs)
    nrows = (num_pairs + ncols - 1) // ncols  # Calculate the number of rows

    fig_width_per_plot = 5
    fig_width = ncols * fig_width_per_plot
    fig_height = 5 * nrows

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))
    if num_pairs == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    idx = 0
    for i, j in zip(*pairs):
        ax = axes[idx]  # Select the appropriate subplot
        x = vaf[:, i].numpy()
        y = vaf[:, j].numpy()

        for c, cluster in enumerate(np.unique(cluster_labels)):
            mask = (cluster_labels == cluster)  # Mask for current cluster
            ax.scatter(x[mask], 
                        y[mask],
                        label=f'{cluster.astype("int")} {type_labels_cluster[c].tolist()}')
        ax.legend(loc='best')
        ax.set_title(f'Sample {i+1} vs Sample {j+1}')
        ax.set_xlabel(f'Sample {i+1}')
        ax.set_ylabel(f'Sample {j+1}')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        idx += 1
    plt.suptitle(f'Orignal data with N = {N} and {K} clusters (i = {idx_real})')
    plt.show()
    plt.savefig(f'plots/p_{str(purity).replace(".", "")}_cov_{coverage}/D_{D}/real/N_{N}_K_{K}_D_{D}_real_{idx_real}.png')
    plt.close()


def plot_marginals_real(NV, DP, N, K, D, type_labels_cluster, cluster_labels, phi_beta, kappa_beta, alpha, idx, purity, coverage):
    vaf = NV/DP
    copy_vaf = torch.clone(vaf)
    # Replace zeros with a large value that will not be considered as minimum (i.e. 1)
    masked_tensor = copy_vaf.masked_fill(vaf == 0, float(1.))

    # Find the minimum value for each column excluding zeros
    min_values, _ = torch.min(masked_tensor, dim=0)
    min_values = min_values.repeat(K, 1)
    pareto_L = torch.min(min_values)
    if K == 1:
        fig, axes = plt.subplots(K, NV.shape[1], figsize=(16, 4))
    else:
        fig, axes = plt.subplots(K, NV.shape[1], figsize=(16, K*3))
    if K == 1:
        axes = ax = np.array([axes])  # add an extra dimension to make it 2D
    plt.suptitle(f'Marginals with N = {N} and {K} clusters (i = {idx}) \n ')
    x = np.linspace(0.001, 1, 1000)
    for k in range(K):
        for d in range(D):
            maxx = type_labels_cluster[k, d]
            if maxx == 1:
                # plot beta
                a = phi_beta[k,d] * kappa_beta[k,d]
                b = (1-phi_beta[k,d]) * kappa_beta[k,d]
                pdf = beta.pdf(x, a, b)
                axes[k,d].plot(x, pdf, linewidth=1.5, label='Beta', color='r')
                axes[k,d].legend()
            elif maxx == 0:
                #plot pareto
                pdf = pareto.pdf(x, alpha[k,d], scale=pareto_L)
                axes[k,d].plot(x, pdf, linewidth=1.5, label='Pareto', color='g')
                axes[k,d].legend()
            else:
                # private
                a_beta_zeros = torch.tensor(1e-3)
                b_beta_zeros = torch.tensor(1e3)
                pdf = beta.pdf(x, a_beta_zeros, b_beta_zeros) # delta_approx
                axes[k,d].plot(x, pdf, linewidth=1.5, label='Zeros', color='b')
                axes[k,d].legend()
            
            if torch.is_tensor(NV):
                data = NV[:,d].numpy()/DP[:,d].numpy()
            else:
                data = np.array(NV[:,d])/np.array(DP[:,d])
            # data = mb.NV[:,d].numpy()/mb.DP[:,d].numpy()
            # for i in np.unique(labels):
            axes[k,d].hist(data[cluster_labels == k], density=True, bins=30, alpha=0.5)#, color=cmap(i))
            axes[k,d].set_title(f'Sample {d+1} - Cluster {k}')
            axes[k,d].set_xlim([-0.01,1])
            plt.tight_layout()
    
    plt.show()
    plt.savefig(f'./plots/p_{str(purity).replace(".", "")}_cov_{coverage}/D_{D}/real_marginals/N_{N}_K_{K}_D_{D}_real_{idx}.png')
    plt.close()



def retrieve_info(mb, N, D):
    pred_cluster_labels = mb.params['cluster_assignments']
    delta = mb.params["delta_param"]
    phi = mb.params["phi_beta_param"]
    kappa = mb.params["k_beta_param"]
    alpha = mb.params["alpha_pareto_param"]
    pred_type_labels_data = torch.zeros((mb.NV.shape[0],mb.NV.shape[1]))
    
    phi_param_data = torch.zeros((N,D))
    kappa_param_data = torch.zeros((N,D))
    alpha_param_data = torch.zeros((N,D))

    for i, k in enumerate(np.unique(pred_cluster_labels)):
        mask = (pred_cluster_labels == k)  # Mask for current cluster
        for d in range(mb.NV.shape[1]):
            delta_kd = delta[k, d]
            maxx = torch.argmax(delta_kd)
            if maxx == 0: # Pareto
                pred_type_labels_data[mask,d] = 0
                phi_param_data[mask,d] = -1
                kappa_param_data[mask,d] = -1
                alpha_param_data[mask,d] = torch.round(alpha[k, d] * 1000) / 1000
                # pred_param_list_data[mask][d] = [None, None, alpha[k,d]]
            elif maxx == 1:
                pred_type_labels_data[mask,d] = 1
                phi_param_data[mask,d] = torch.round(phi[k, d] * 1000) / 1000
                kappa_param_data[mask,d] = torch.round(kappa[k, d] * 1000) / 1000
                alpha_param_data[mask,d] = -1
                # pred_param_list_data[mask][d] = [phi[k,d], kappa[k,d], None]
        
    return pred_cluster_labels, pred_type_labels_data, phi_param_data, kappa_param_data, alpha_param_data


def plot_deltas_gen(mb,  N, K, D, idx, savefig = False, data_folder = None):
    deltas = mb.params["delta_param"].detach().numpy()
    if deltas.shape[0] == 1:
        fig, ax = plt.subplots(nrows=deltas.shape[0], ncols=1, figsize=(6, 1.5))  # Custom size for 1 plot
        ax = [ax]  # add an extra dimension to make it 2D
    else:
        fig, ax = plt.subplots(nrows=deltas.shape[0], ncols=1, figsize=(6, mb.K*1))
    
    plt.suptitle(f'Delta with K={mb.K}, seed={mb.seed} (idx = {idx})', fontsize=14)
    fig.tight_layout() 
    for k in range(deltas.shape[0]):
        sns.heatmap(deltas[k], ax=ax[k], vmin=0, vmax=1, cmap="crest")
        # ax[k].set(xlabel="Distributions (0=Pareto, 1=Beta)", ylabel="Sample")
        # ax[k].set(xlabel="Distributions", ylabel="Sample")
        # ax[k].set_yticklabels([1, 2])
        num_rows = deltas[k].shape[0]
        ax[k].set_yticks([i + 0.5 for i in range(num_rows)])  # Center ticks in the middle of each row
        ax[k].set_yticklabels([str(i + 1) for i in range(num_rows)], rotation=0)  # Explicitly set rotation to 0

        # Set x-tick labels
        ax[k].set_xticklabels(["Pareto", "Beta", "Zeros"], rotation=0)

        # Setting x and y labels for the subplot
        ax[k].set(xlabel="", ylabel="Sample")
        if k == (deltas.shape[0] - 1):
            ax[k].set(xlabel="Distributions")
        ax[k].set_title(f'Cluster {k}', fontsize=14)
    seed = mb.seed
    if savefig:
        folder = data_folder + f'deltas_N_{N}_K_{K}_D_{D}_{idx}.png'
        plt.savefig(folder)
    plt.show()
    plt.close()


def plot_responsib_gen(mb,  N, K, D, idx, savefig = False, data_folder = None):
    
    if torch.is_tensor(mb.params['responsib']):
        resp = mb.params['responsib'].detach().numpy()
    else:
        resp = np.array(mb.params['responsib'])
    
    fig, ax = plt.subplots(nrows=1, ncols=1)
    plt.suptitle(f'Responsibilities with K={mb.K}, seed={mb.seed} (idx = {idx})', fontsize = 14)
    fig.tight_layout()
    sns.heatmap(resp, ax=ax, vmin=0, vmax=1, cmap="crest")
    seed = mb.seed
    if savefig:
        folder = data_folder + f'responsib_N_{N}_K_{K}_D_{D}_{idx}.png'
        plt.savefig(folder)
    plt.show()
    plt.close()


def plot_paretos_gen(mb, N, K, D, idx, savefig = False, data_folder = None):
    check = False
    check = "probs_pareto_param" in mb.params.keys()
    if check:
        probs_pareto = mb.params["probs_pareto_param"]

    if torch.is_tensor(mb.params['alpha_pareto_param']):
        alpha_pareto = mb.params["alpha_pareto_param"].detach().numpy()
    else:
        alpha_pareto = np.array(mb.params["alpha_pareto_param"])

    if alpha_pareto.shape[0] == 1:
        fig, ax = plt.subplots(nrows=alpha_pareto.shape[0], ncols=alpha_pareto.shape[1], figsize = (7,3))
        ax = np.array([ax])
    else:
        fig, ax = plt.subplots(nrows=alpha_pareto.shape[0], ncols=alpha_pareto.shape[1], figsize = (18,mb.K*1))      
    plt.suptitle(f'Pareto with K={mb.K}, seed={mb.seed} (idx = {idx})', fontsize=14)
    fig.tight_layout()
    x = np.arange(0,0.5,0.001)
    for k in range(alpha_pareto.shape[0]):
        for d in range(alpha_pareto.shape[1]):
            pdf = pareto.pdf(x, alpha_pareto[k,d], scale=0.001)
            ax[k,d].plot(x, pdf, 'r-', lw=1)
            if check:
                ax[k,d].set_title(f'Sample {d+1} Cluster {k} - alpha {round(float(alpha_pareto[k,d]), ndigits=2)}, p {round(float(probs_pareto[k,d]), ndigits=2)}', fontsize=10)
            else:
                ax[k,d].set_title(f'Sample {d+1} Cluster {k} - alpha {round(float(alpha_pareto[k,d]), ndigits=2)}', fontsize=10)
            # ax[k,d].set_title(f'Cluster {k} Dimension {d} - alpha {round(float(alpha_pareto[k,d]), ndigits=2)}")
    seed = mb.seed
    if savefig:
        folder = data_folder + f'paretos_N_{N}_K_{K}_D_{D}_{idx}.png'
        plt.savefig(folder)
    plt.show()
    plt.close()


def plot_betas_gen(mb, N, K, D, idx,savefig = False, data_folder = None):
    phi_beta = mb.params["phi_beta_param"].detach().numpy()
    kappa_beta = mb.params["k_beta_param"].detach().numpy()
    if phi_beta.shape[0] == 1:
        fig, ax = plt.subplots(nrows=phi_beta.shape[0], ncols=phi_beta.shape[1], figsize = (7,3))
        ax = np.array([ax])
    else:
        fig, ax = plt.subplots(nrows=phi_beta.shape[0], ncols=phi_beta.shape[1], figsize = (18,mb.K*1))   
    plt.suptitle(f'Beta with K={mb.K}, seed={mb.seed} (idx = {idx})', fontsize=14)
    fig.tight_layout()
    x = np.arange(0,1,0.001)
    for k in range(phi_beta.shape[0]):
        for d in range(phi_beta.shape[1]):
            a = phi_beta[k,d]*kappa_beta[k,d]
            b = (1-phi_beta[k,d])*kappa_beta[k,d]
            pdf = beta.pdf(x, a, b)
            ax[k,d].plot(x, pdf, 'r-', lw=1)
            ax[k,d].set_title(f'Sample {d+1} Cluster {k} - phi {round(float(phi_beta[k,d]), ndigits=2)}, kappa {round(float(kappa_beta[k,d]), ndigits=2)}', fontsize=10)
    seed = mb.seed
    
    if savefig:
        folder = data_folder + f'betas_N_{N}_K_{K}_D_{D}_{idx}.png'
        plt.savefig(folder)
    plt.show()
    plt.close()


def plot_loss_lks_gen(mb, N, K, D, idx, savefig = True, data_folder = None):
    # dist_pi, dist_pi_euc = mb.compute_mixing_distances(mb.pi_list)
    # dist_delta, dist_delta_euc = mb.compute_mixing_distances(mb.delta_list)
    # dist_alpha,dist_alpha_euc = mb.compute_mixing_distances(mb.alpha_list)
    # dist_phi,dist_phi_euc = mb.compute_mixing_distances(mb.phi_list)

    dist = mb.compute_mixing_distances(mb.params_stop_list)

    _, ax = plt.subplots(2, 2, figsize=(15, 15))
    ax[0,0].plot(mb.losses)
    ax[0,0].set_title(f'Loss (K = {mb.K}, seed = {mb.seed})')
    ax[0,0].grid(True, color='gray', linestyle='-', linewidth=0.2)

    ax[0,1].plot(mb.lks)
    ax[0,1].set_title(f'Likelihood (K = {mb.K}, seed = {mb.seed})')
    ax[0,1].grid(True, color='gray', linestyle='-', linewidth=0.2)

    keys = list(mb.params_stop_list.keys())
    for key in keys:
        dist_rel = dist[key]["max_relative_distances"]
        dist_euc = dist[key]["euclidean_distances"]

        # Plot max relative distances
        ax[1, 0].plot(dist_rel, label=key)

        # Plot Euclidean distances
        ax[1, 1].plot(dist_euc, label=key)

    ax[1, 0].set_title("Max relative dist between consecutive iterations")
    ax[1, 0].grid(True, color='gray', linestyle='-', linewidth=0.2)
    ax[1, 0].axhline(y=mb.par_threshold, color='red', linestyle='--', linewidth=0.8, label=f'Threshold')
    ax[1, 0].legend()

    ax[1, 1].set_title("Relative euclidean dist between consecutive iterations")
    ax[1, 1].grid(True, color='gray', linestyle='-', linewidth=0.2)
    ax[1, 1].legend()  
    
    if savefig:
        folder = data_folder + f'lks_loss_N_{N}_K_{K}_D_{D}_{idx}_seed_{mb.seed}.png'
        plt.savefig(folder)
    plt.show()
    plt.close()


def plot_model_selection_gen(mb_list, N, K_list, K, seed_list, D, idx, savefig = True, data_folder = None):
    lk_list = []
    bic_list = []
    icl_list = []
    for i, k in enumerate(K_list):
        # start_idx = i * len(seed_list)
        # end_idx = start_idx + len(seed_list)

        # elements_for_K = mb_list[start_idx:end_idx]
        # # values_for_specific_key = [d.final_dict["bic"] for d in elements_for_K]
        # values_for_specific_key = [d.final_dict["bic"] for d in elements_for_K] # Given a specific K, select the seed with the lowest bic_sampling_p
        # min_idx = values_for_specific_key.index(min(values_for_specific_key))
        # min_idx = start_idx + min_idx
        
        lk = mb_list[i].final_dict["final_likelihood"].detach().numpy()
        bic = mb_list[i].final_dict["bic"].detach().numpy()
        icl = mb_list[i].final_dict["icl"].detach().numpy()
        
        lk_list.append(lk)
        bic_list.append(bic)
        icl_list.append(icl)

    plt.figure(figsize=(20, 6))

    # Subplot 1: Likelihood over K
    plt.subplot(1, 3, 1)
    plt.title("Likelihood over K")
    plt.xlabel("K")
    plt.ylabel("Likelihood")
    plt.plot(K_list, lk_list, label="Likelihood", color='black')
    # Scatter all points (other than the minimum one) in black
    plt.scatter(K_list, lk_list, color='black', zorder=5)  
    # Find the index of the minimum value and highlight it in red
    min_index_lk = np.argmax(lk_list)
    plt.scatter(K_list[min_index_lk], lk_list[min_index_lk], color='black', zorder=10)  
    plt.legend()
    plt.grid(True, color='gray', linestyle='-', linewidth=0.2)  

    # Subplot 2: BIC over K
    plt.subplot(1, 3, 2)
    plt.title("BIC over K")
    plt.xlabel("K")
    plt.ylabel("BIC")
    plt.plot(K_list, bic_list, label="BIC", color='black')
    # Scatter all points (other than the minimum one) in black
    plt.scatter(K_list, bic_list, color='black', zorder=5)  
    # Find the index of the minimum value and highlight it in red
    min_index_bic = np.argmin(bic_list)
    plt.scatter(K_list[min_index_bic], bic_list[min_index_bic], color='r', zorder=10)  
    plt.legend()
    plt.grid(True, color='gray', linestyle='-', linewidth=0.2)

    # Subplot 3: ICL over K
    plt.subplot(1, 3, 3)
    plt.title("ICL over K")
    plt.xlabel("K")
    plt.ylabel("ICL")
    plt.plot(K_list, icl_list, label="ICL", color='black')
    # Scatter all points (other than the minimum one) in black
    plt.scatter(K_list, icl_list, color='black', zorder=5)  
    # Find the index of the minimum value and highlight it in red
    min_index_icl = np.argmin(icl_list)
    plt.scatter(K_list[min_index_icl], icl_list[min_index_icl], color='r', zorder=10)  
    plt.legend()
    plt.grid(True, color='gray', linestyle='-', linewidth=0.2)  

    plt.tight_layout()
    if savefig:
        folder = data_folder + f'metrics_over_K_N_{N}_K_{K}_D_{D}_{idx}.png'
        plt.savefig(folder)
    plt.show()
    plt.close()









