import numpy as np
import pyro
import pyro.distributions as dist

import torch

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import binom, beta, pareto
from utils.BoundedPareto import BoundedPareto
import seaborn as sns


def plot_deltas(mb, savefig = False, data_folder = None):
    deltas = mb.params["delta_param"].detach().numpy()
    if deltas.shape[0] == 1:
        fig, ax = plt.subplots(nrows=deltas.shape[0], ncols=1, figsize=(6, 1.5))  # Custom size for 1 plot
        ax = [ax]  # add an extra dimension to make it 2D
    else:
        fig, ax = plt.subplots(nrows=deltas.shape[0], ncols=1, figsize=(6, mb.K*1))
    
    plt.suptitle(f"Delta with K={mb.K}, seed={mb.seed}", fontsize=14)
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
        ax[k].set_xticklabels(["Pareto", "Beta"], rotation=0)

        # Setting x and y labels for the subplot
        ax[k].set(xlabel="", ylabel="Sample")
        if k == (deltas.shape[0] - 1):
            ax[k].set(xlabel="Distributions")
        ax[k].set_title(f"Cluster {k}", fontsize=14)
    seed = mb.seed
    if savefig:
        plt.savefig(f"plots/{data_folder}/deltas_K_{mb.K}_seed_{seed}.png")
    # plt.show()
    # plt.close()

def plot_responsib(mb, which, savefig = False, data_folder = None):
    # resp = mb.params["responsib"].detach().numpy()
    if which == 'integr':
        resp = mb.params['responsib'].detach().numpy()
    else:
        resp = mb.params['responsib_sampling_p'].detach().numpy()

    fig, ax = plt.subplots(nrows=1, ncols=1)
    plt.suptitle(f"Responsibilities with K={mb.K}, seed={mb.seed} ({which})", fontsize = 14)
    fig.tight_layout()
    sns.heatmap(resp, ax=ax, vmin=0, vmax=1, cmap="crest")
    seed = mb.seed
    if savefig:
        plt.savefig(f"plots/{data_folder}/responsibilities_K_{mb.K}_seed_{seed}_{which}.png")
    # plt.show()
    # plt.close()

def plot_paretos(mb, savefig = False, data_folder = None):
    check = False
    check = "probs_pareto_param" in mb.params.keys()
    if check:
        probs_pareto = mb.params["probs_pareto_param"]
    alpha_pareto = mb.params["alpha_pareto_param"].detach().numpy()
    if alpha_pareto.shape[0] == 1:
        fig, ax = plt.subplots(nrows=alpha_pareto.shape[0], ncols=alpha_pareto.shape[1], figsize = (7,3))
        ax = np.array([ax])
    else:
        fig, ax = plt.subplots(nrows=alpha_pareto.shape[0], ncols=alpha_pareto.shape[1], figsize = (10,mb.K*1))      
    plt.suptitle(f"Pareto with K={mb.K}, seed={mb.seed}", fontsize=14)
    fig.tight_layout()
    x = np.arange(0,0.5,0.001)
    for k in range(alpha_pareto.shape[0]):
        for d in range(alpha_pareto.shape[1]):
            pdf = pareto.pdf(x, alpha_pareto[k,d], scale=0.001)
            ax[k,d].plot(x, pdf, 'r-', lw=1)
            if check:
                ax[k,d].set_title(f"Sample {d+1} Cluster {k} - alpha {round(float(alpha_pareto[k,d]), ndigits=2)}, p {round(float(probs_pareto[k,d]), ndigits=2)}", fontsize=10)
            else:
                ax[k,d].set_title(f"Sample {d+1} Cluster {k} - alpha {round(float(alpha_pareto[k,d]), ndigits=2)}", fontsize=10)
            # ax[k,d].set_title(f"Cluster {k} Dimension {d} - alpha {round(float(alpha_pareto[k,d]), ndigits=2)}")
    seed = mb.seed
    if savefig:
        plt.savefig(f"plots/{data_folder}/paretos_K_{mb.K}_seed_{seed}.png")
    # plt.show()
    # plt.close()

def plot_betas(mb, savefig = False, data_folder = None):
    phi_beta = mb.params["phi_beta_param"].detach().numpy()
    kappa_beta = mb.params["k_beta_param"].detach().numpy()
    if phi_beta.shape[0] == 1:
        fig, ax = plt.subplots(nrows=phi_beta.shape[0], ncols=phi_beta.shape[1], figsize = (7,3))
        ax = np.array([ax])
    else:
        fig, ax = plt.subplots(nrows=phi_beta.shape[0], ncols=phi_beta.shape[1], figsize = (10,mb.K*1))   
    plt.suptitle(f"Beta with K={mb.K}, seed={mb.seed}", fontsize=14)
    fig.tight_layout()
    x = np.arange(0,1,0.001)
    for k in range(phi_beta.shape[0]):
        for d in range(phi_beta.shape[1]):
            a = phi_beta[k,d]*kappa_beta[k,d]
            b = (1-phi_beta[k,d])*kappa_beta[k,d]
            pdf = beta.pdf(x, a, b)
            ax[k,d].plot(x, pdf, 'r-', lw=1)
            ax[k,d].set_title(f"Sample {d+1} Cluster {k} - phi {round(float(phi_beta[k,d]), ndigits=2)}, kappa {round(float(kappa_beta[k,d]), ndigits=2)}", fontsize=10)
    seed = mb.seed
    
    if savefig:
        plt.savefig(f"plots/{data_folder}/betas_K_{mb.K}_seed_{seed}.png")
    # plt.show()
    # plt.close()

def plot_marginals(mb, which = 'integr', savefig = False, data_folder = None):
    delta = mb.params["delta_param"]  # K x D x 2
    phi_beta = mb.params["phi_beta_param"].detach().numpy()
    kappa_beta = mb.params["k_beta_param"].detach().numpy()
    alpha = mb.params["alpha_pareto_param"].detach().numpy()
    weights = mb.params["weights_param"].detach().numpy()
    if which == 'integr':
        labels = mb.params['cluster_assignments'].detach().numpy()
    else:
        labels = mb.params['cluster_assignments_sampling_p'].detach().numpy()

    # For each sample I want to plot all the clusters separately.
    # For each cluster, we need to plot the density corresponding to the beta or the pareto based on the value of delta
    # For each cluster, we want to plot the histogram of the data assigned to that cluster
    if mb.K == 1:
        fig, axes = plt.subplots(mb.K, mb.NV.shape[1], figsize=(16, 4))
    else:
        fig, axes = plt.subplots(mb.K, mb.NV.shape[1], figsize=(16, mb.K*3))
    if mb.K == 1:
        axes = ax = np.array([axes])  # add an extra dimension to make it 2D
    plt.suptitle(f"Marginals with K={mb.K}, seed={mb.seed} ({which})",fontsize=14)
    x = np.linspace(0.001, 1, 1000)
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
            else:
                #plot pareto
                pdf = pareto.pdf(x, alpha[k,d], scale=mb.pareto_L) #* weights[k]
                axes[k,d].plot(x, pdf, linewidth=1.5, label='Pareto', color='g')
                axes[k,d].legend()
            data = mb.NV[:,d].numpy()/mb.DP[:,d].numpy()
            # for i in np.unique(labels):
            axes[k,d].hist(data[labels == k], density=True, bins=30, alpha=0.5)#, color=cmap(i))
            axes[k,d].set_title(f"Sample {d+1} - Cluster {k}")
            axes[k,d].set_ylim([0,25])
            axes[k,d].set_xlim([0,1])
            plt.tight_layout()
    if savefig:
        plt.savefig(f"plots/{data_folder}/marginals_K_{mb.K}_seed_{mb.seed}_{which}.png")
    # plt.show()
    # plt.close()