import numpy as np
import pyro
import pyro.distributions as dist

import torch

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import binom, beta, pareto
from BoundedPareto import BoundedPareto
import seaborn as sns


def plot_deltas(mb):
    deltas = mb.params["delta_param"].detach().numpy()
    fig, ax = plt.subplots(nrows=deltas.shape[0], ncols=1)
    fig.tight_layout()
    for k in range(deltas.shape[0]):
        sns.heatmap(deltas[k], ax=ax[k], vmin=0, vmax=1, cmap="crest")
        ax[k].set(xlabel="Distributions (0=Pareto, 1=Beta)", ylabel="Dimensions")
        ax[k].set_title(f"Cluster {k}")

def plot_responsib(mb):
    resp = mb.params["responsib"].detach().numpy()
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.tight_layout()
    sns.heatmap(resp, ax=ax, vmin=0, vmax=1, cmap="crest")

def plot_paretos(mb):
    alpha_pareto = mb.params["alpha_pareto_param"].detach().numpy()
    fig, ax = plt.subplots(nrows=alpha_pareto.shape[0], ncols=alpha_pareto.shape[1])
    fig.tight_layout()
    x = np.arange(0,0.5,0.001)
    for k in range(alpha_pareto.shape[0]):
        for d in range(alpha_pareto.shape[1]):
            pdf = pareto.pdf(x, alpha_pareto[k,d], scale=0.001)
            ax[k,d].plot(x, pdf, 'r-', lw=1)
            ax[k,d].set_title(f"Cluster {k} Dimension {d} - alpha {round(float(alpha_pareto[k,d]), ndigits=2)}")

def plot_betas(mb):
    phi_beta = mb.params["phi_beta_param"].detach().numpy()
    kappa_beta = mb.params["k_beta_param"].detach().numpy()

    fig, ax = plt.subplots(nrows=phi_beta.shape[0], ncols=phi_beta.shape[1])
    fig.tight_layout()
    x = np.arange(0,1,0.001)
    for k in range(phi_beta.shape[0]):
        for d in range(phi_beta.shape[1]):
            a = phi_beta[k,d]*kappa_beta[k,d]
            b = (1-phi_beta[k,d])*kappa_beta[k,d]
            pdf = beta.pdf(x, a, b)
            ax[k,d].plot(x, pdf, 'r-', lw=1)
            ax[k,d].set_title(f"Cluster {k} Dimension {d}")

def plot_marginals(mb):
    delta = mb.params["delta_param"]  # K x D x 2
    phi_beta = mb.params["phi_beta_param"].detach().numpy()
    kappa_beta = mb.params["k_beta_param"].detach().numpy()
    alpha = mb.params["alpha_pareto_param"].detach().numpy()
    weights = mb.params["weights_param"].detach().numpy()
    labels = mb.params['cluster_assignments'].detach().numpy()

    # For each dimension, for each cluster, we need to plot the density corresponding to the beta or the pareto based on the value of delta
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    x = np.linspace(0.001, 1, 1000)
    for d in range(mb.NV.shape[1]):
        for k in range(mb.K):
            delta_kd = delta[k, d]
            maxx = torch.argmax(delta_kd)
            if maxx == 1:
                # plot beta
                a = phi_beta[k,d] * kappa_beta[k,d]
                b = (1-phi_beta[k,d]) * kappa_beta[k,d]
                pdf = beta.pdf(x, a, b) * weights[k]
                axes[d].plot(x, pdf, linewidth=1.5, label='Beta', color='r')
            else:
                #plot pareto
                pdf = pareto.pdf(x, alpha[k,d], scale=0.01) * weights[k]
                axes[d].plot(x, pdf, linewidth=1.5, label='Pareto', color='g')
        axes[d].legend()
        data = mb.NV[:,d].numpy()/mb.DP[:,d].numpy()
        # cmap = plt.get_cmap('viridis', np.unique(labels))
        for i in np.unique(labels):
            axes[d].hist(data[labels == i], density=True, bins=30, alpha=0.5)#, color=cmap(i))

        # axes[d].hist(data[labels == 0], density=True, bins=30, alpha=0.3, color='violet')
        # axes[d].hist(data[labels == 1], density=True, bins=30, alpha=0.3, color='yellow')
        axes[d].set_title(f"Dimension {d+1}")
        axes[d].set_ylim([0,100])
        axes[d].set_xlim([0,1])
        plt.tight_layout()