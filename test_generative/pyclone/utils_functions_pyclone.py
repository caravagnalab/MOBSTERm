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
from utils.plot_functions import *
from utils.BoundedPareto import BoundedPareto
from utils.create_beta_pareto_dataset import *

def create_folder(N,K,D,purity,coverage):
    folder_path = f'data/p_{str(purity).replace(".", "")}_cov_{coverage}/D_{D}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    folder_path = f'results/p_{str(purity).replace(".", "")}_cov_{coverage}/D_{D}/fit_files'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    folder_path = f'results/p_{str(purity).replace(".", "")}_cov_{coverage}/D_{D}/best_fit_files'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    folder_path = f'results/p_{str(purity).replace(".", "")}_cov_{coverage}/D_{D}/csv'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    folder_path = f'results/p_{str(purity).replace(".", "")}_cov_{coverage}/D_{D}/nmi'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    folder_path = f'results/p_{str(purity).replace(".", "")}_cov_{coverage}/D_{D}/ari'
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



def plot_scatter_final(NV, DP, D, fitted_table, N, K, idx_real, purity, coverage):
    columns = [f"S_{i}" for i in range(D)]

    D = NV.shape[1]  # Number of sets or columns
    pairs = np.triu_indices(D, k=1)  # Generate all unique pairs of sets (i, j)
    vaf = NV / DP  # Variant Allele Fraction
    num_pairs = len(pairs[0])  # Number of unique pairs of scatter plots
    ncols = min(3, num_pairs)  # Max 3 plots per row
    nrows = (num_pairs + ncols - 1) // ncols  # Calculate number of rows needed

    fig_width_per_plot = 5
    fig_width = ncols * fig_width_per_plot
    fig_height = 5 * nrows

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))

    # Flatten axes to make iteration easier
    axes = axes.flatten() if num_pairs > 1 else [axes] 

    for ax, pair in zip(axes, zip(*pairs)):
        x_col, y_col = pair  # Unpack the pair of indices
        sns.scatterplot(data=fitted_table, x=f"S_{x_col}", y=f"S_{y_col}", hue='cluster_id', palette='Set2', ax=ax, s=20)
        ax.set_title(f'Set {x_col} vs {y_col}')
        ax.set_xlabel(f"Set {x_col}")
        ax.set_ylabel(f"Set {y_col}")

    for ax in axes[len(pairs[0]):]:
        ax.axis('off')

    plt.tight_layout()
    plt.suptitle(f'Inference with N = {N} and {K} clusters (i = {idx_real}) \n ')
    plt.show()
    plt.savefig(f'plots/p_{str(purity).replace(".", "")}_cov_{coverage}/D_{D}/inference/N_{N}_K_{K}_D_{NV.shape[1]}_inference_{idx_real}.png')
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
            else:
                # private
                pred_type_labels_data[mask,d] = 2
                phi_param_data[mask,d] = -1
                kappa_param_data[mask,d] = -1
                alpha_param_data[mask,d] = -1

    return pred_cluster_labels, pred_type_labels_data, phi_param_data, kappa_param_data, alpha_param_data



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
                        label=f'{cluster.astype("int")} {type_labels_cluster[c].tolist()}', s = 10)
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








