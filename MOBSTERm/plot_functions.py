import numpy as np
import pyro
import pyro.distributions as dist

import torch

import matplotlib.pyplot as plt
from scipy.stats import beta, pareto
from .BoundedPareto import BoundedPareto
import seaborn as sns
import matplotlib.cm as cm
import pandas as pd
from itertools import combinations

colors = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#cccc33",  
    "#a65628", "#f781bf", "#999999", "#000000",  # First 10 colors (Set1)  
    "#46f0f0", "#f032e6", "#bcf60c", "#fabed4", "#008080", "#e6beff",  
    "#9a6324", "#fffac8", "#800000", "#aaffc3", "#808000", "#ffd8b1",  
    "#000075", "#808080", "#d3a6f3", "#ff9cdd", "#73d7b0"  
    ]

def plot_deltas(mb, savefig = False, data_folder = None):
    deltas = mb.params["delta_param"].detach().numpy()
    K = deltas.shape[0]
    if deltas.shape[0] == 1:
        fig, ax = plt.subplots(nrows=deltas.shape[0], ncols=1, figsize=(5, 1.5))
        ax = [ax] 
    else:
        fig, ax = plt.subplots(nrows=deltas.shape[0], ncols=1, figsize=(5, K * 0.6))

    plt.suptitle(f"Delta with K={mb.K}, seed={mb.seed}", fontsize=12, y=0.98)

    fig.subplots_adjust(top=0.93, hspace=0.2, right=0.8)

    # Define a shared color scale
    norm = plt.Normalize(vmin=0, vmax=1)
    cmap = sns.color_palette("crest", as_cmap=True)

    for k in range(deltas.shape[0]):
        sns.heatmap(deltas[k], ax=ax[k], vmin=0, vmax=1, cmap=cmap, cbar=False)

        num_rows = deltas[k].shape[0]
        ax[k].set_yticks([i + 0.5 for i in range(num_rows)])
        ax[k].set_yticklabels([str(i + 1) for i in range(num_rows)], rotation=0)

        ax[k].set(xlabel="", ylabel="Sample")

        if k == (deltas.shape[0] - 1):
            ax[k].set_xticklabels(["ParetoBinomial", "BetaBinomial", "Dirac"], rotation=0)
            ax[k].set(xlabel="Distributions")
        else:
            ax[k].set_xticklabels([])
            ax[k].tick_params(axis='x', which='both', bottom=False, top=False)

        # Add cluster label on the right side, slightly to the left to make space for colorbar
        ax[k].text(
            ax[k].get_xlim()[0] - 0.6,  # Position slightly outside the heatmap
            num_rows / 2,  # Center vertically
            f"C{k}",
            fontsize=12,
            va="center",
            ha="left"
        )

    cbar_ax = fig.add_axes([0.85, 0.2, 0.02, 0.6])  # [left, bottom, width, height]
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, cax=cbar_ax)
    seed = mb.seed
    if savefig:
        plt.savefig(f"plots/{data_folder}/deltas_K_{mb.K}_seed_{seed}.png")
    plt.show()
    plt.close()

def plot_responsib(mb, savefig = False, data_folder = None):
    if torch.is_tensor(mb.params['responsib']):
        resp = mb.params['responsib'].detach().numpy()
    else:
        resp = np.array(mb.params['responsib'])
    
    fig, ax = plt.subplots(nrows=1, ncols=1)
    plt.suptitle(f"Responsibilities with K={mb.K}, seed={mb.seed}", fontsize = 14)
    fig.tight_layout()
    sns.heatmap(resp, ax=ax, vmin=0, vmax=1, cmap="crest")
    seed = mb.seed
    if savefig:
        plt.savefig(f"plots/{data_folder}/responsibilities_K_{mb.K}_seed_{seed}.png")
    plt.show()
    plt.close()

def plot_paretos(mb, savefig = False, data_folder = None):
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
        fig, ax = plt.subplots(nrows=alpha_pareto.shape[0], ncols=alpha_pareto.shape[1], figsize = (18,mb.final_K*1))      
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
    seed = mb.seed
    if savefig:
        plt.savefig(f"plots/{data_folder}/paretos_K_{mb.K}_seed_{seed}.png")
    plt.show()
    plt.close()

def plot_betas(mb, savefig = False, data_folder = None):
    phi_beta = mb.params["phi_beta_param"].detach().numpy()
    kappa_beta = mb.params["k_beta_param"].detach().numpy()
    if phi_beta.shape[0] == 1:
        fig, ax = plt.subplots(nrows=phi_beta.shape[0], ncols=phi_beta.shape[1], figsize = (7,3))
        ax = np.array([ax])
    else:
        fig, ax = plt.subplots(nrows=phi_beta.shape[0], ncols=phi_beta.shape[1], figsize = (18,mb.final_K*1))   
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
    plt.show()
    plt.close()

def plot_marginals_single_nd(mb, savefig = False, data_folder = None):
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
        
    labels = mb.cluster_assignments
    if torch.is_tensor(labels):
        labels = labels.detach().numpy()
    else:
        labels = np.array(labels)

    # For each sample I want to plot all the clusters separately.
    # For each cluster, we need to plot the density corresponding to the beta or the pareto based on the value of delta
    # For each cluster, we want to plot the histogram of the data assigned to that cluster
    if mb.final_K == 1:
        fig, axes = plt.subplots(mb.final_K, mb.NV.shape[1], figsize=(10, 4))
    else:
        fig, axes = plt.subplots(mb.final_K, mb.NV.shape[1], figsize=(10, mb.final_K*3))
    if mb.final_K == 1:
        axes = ax = np.array([axes])  # add an extra dimension to make it 2D
    plt.suptitle(f"Marginals with K={mb.K}, seed={mb.seed}",fontsize=14)
    x = np.linspace(0.001, 1, 1000)

    unique_labels = np.unique(labels)

    # color_mapping = colors[:len(unique_labels)]
    if mb.final_K == mb.K:
        color_mapping = colors
    else:
        color_mapping = colors[:len(unique_labels)]
    # cmap = cm.get_cmap('tab20')
    # color_mapping = {label: cmap(i) for i, label in enumerate(unique_labels)}
    for k in range(mb.final_K):
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
                axes[k,d].plot(x, pdf, linewidth=1.5, label='Dirac', color='b')
                axes[k,d].legend()

            if torch.is_tensor(mb.NV):
                data = mb.NV[:,d].numpy()/mb.DP[:,d].numpy()
            else:
                data = np.array(mb.NV[:,d])/np.array(mb.DP[:,d])
            # for i in np.unique(labels):
            if k in unique_labels:
                if maxx == 2:
                    n_bins = 50
                else:
                    n_bins = min(int(np.ceil(np.sqrt(len(data[labels == k])))),30)
                axes[k, d].hist(data[labels == k],  density=True, bins=n_bins, color=color_mapping[k],alpha=1, edgecolor='white')
            else:
                # Plot an empty histogram because we know there are no points in that k
                axes[k, d].hist([], density=True, bins=30, alpha=1)
            axes[k,d].set_title(f"Sample {d+1} - Cluster {k}")
            axes[k,d].grid(True, color='gray', linestyle='-', linewidth=0.2)
            # axes[k,d].set_ylim([0,25])
            axes[k,d].set_xlim([-0.01,0.7])
            plt.tight_layout()
    if savefig:
        plt.savefig(f"plots/{data_folder}/marginals_K_{mb.K}_seed_{mb.seed}.png")
    plt.show()
    plt.close()


def plot_marginals_single_1d(mb, savefig = False, data_folder = None):
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
        
    labels = mb.cluster_assignments
    if torch.is_tensor(labels):
        labels = labels.detach().numpy()
    else:
        labels = np.array(labels)

    # For each sample I want to plot all the clusters separately.
    # For each cluster, we need to plot the density corresponding to the beta or the pareto based on the value of delta
    # For each cluster, we want to plot the histogram of the data assigned to that cluster
    if mb.final_K == 1:
        fig, axes = plt.subplots(mb.final_K, 1, figsize=(10, 4))
    else:
        fig, axes = plt.subplots(mb.final_K, 1, figsize=(10, mb.final_K*3))
    if mb.final_K == 1:
        axes = ax = np.array([axes])  # add an extra dimension to make it 2D
    plt.suptitle(f"Marginals with K={mb.K}, seed={mb.seed}",fontsize=14)
    x = np.linspace(0.001, 1, 1000)

    unique_labels = np.unique(labels)

    # color_mapping = colors[:len(unique_labels)]
    if mb.final_K == mb.K:
        color_mapping = colors
    else:
        color_mapping = colors[:len(unique_labels)]
    # cmap = cm.get_cmap('tab20')
    # color_mapping = {label: cmap(i) for i, label in enumerate(unique_labels)}
    for k in range(mb.final_K):
        delta_kd = delta[k]
        maxx = torch.argmax(delta_kd)
        if maxx == 1:
            # plot beta
            a = phi_beta[k] * kappa_beta[k]
            b = (1-phi_beta[k]) * kappa_beta[k]
            pdf = beta.pdf(x, a, b)# * weights[k]
            axes[k].plot(x, pdf, linewidth=1.5, label='Beta', color='r')
            axes[k].legend()
        elif maxx == 0:
            # plot pareto
            pdf = pareto.pdf(x, alpha[k], scale=mb.pareto_L) #* weights[k]
            axes[k].plot(x, pdf, linewidth=1.5, label='Pareto', color='g')
            axes[k].legend()
        else:
            # private
            pdf = beta.pdf(x, mb.a_beta_zeros, mb.b_beta_zeros) # delta_approx
            axes[k].plot(x, pdf, linewidth=1.5, label='Dirac', color='b')
            axes[k].legend()

        if torch.is_tensor(mb.NV):
            data = mb.NV[:].numpy()/mb.DP[:].numpy()
        else:
            data = np.array(mb.NV[:])/np.array(mb.DP[:])
        # for i in np.unique(labels):
        if k in unique_labels:
            if maxx == 2:
                n_bins = 50
            else:
                n_bins = min(int(np.ceil(np.sqrt(len(data[labels == k])))),30)
            axes[k].hist(data[labels == k],  density=True, bins=n_bins, color=color_mapping[k],alpha=1, edgecolor='white')
        else:
            # Plot an empty histogram because we know there are no points in that k
            axes[k].hist([], density=True, bins=30, alpha=1)
        axes[k].set_title(f"Sample {1} - Cluster {k}")
        axes[k].grid(True, color='gray', linestyle='-', linewidth=0.2)
        # axes[k,d].set_ylim([0,25])
        axes[k].set_xlim([-0.01,0.7])
        plt.tight_layout()
    if savefig:
        plt.savefig(f"plots/{data_folder}/marginals_K_{mb.K}_seed_{mb.seed}.png")
    plt.show()
    plt.close()


def plot_marginals_single(mb):
    D = mb.NV.shape[1]
    if D == 1:
        plot_marginals_single_1d(mb)
    else:
        plot_marginals_single_nd(mb)
       

def plot_mixing_proportions(mb, savefig=False, data_folder=None):
    weights = mb.params["weights_param"]
    if torch.is_tensor(weights):
        weights = weights.detach().numpy()
    else:
        weights = np.array(weights)
        
    labels = mb.cluster_assignments
    if not torch.is_tensor(labels):
        labels = torch.tensor(labels)

    num_clusters = weights.shape[0]
    unique_labels = np.unique(labels)

    # cmap = cm.get_cmap('tab20')
    # color_mapping = {label: cmap(i) for i, label in enumerate(unique_labels)}
    
    color_mapping = colors[:len(unique_labels)]
    if mb.final_K == mb.K:
        color_mapping = colors
    else:
        color_mapping = colors[:len(unique_labels)]

    # Plot 1: Mixing Proportions
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    bars1 = []
    for i in range(num_clusters):
        if i in unique_labels:
            bar = plt.bar(i, weights[i], color=color_mapping[i])
        else:
            bar = plt.bar(i, weights[i], color='gray')
        bars1.append(bar[0])  # Store the bar for legend

    plt.title('Mixing proportions')
    plt.xlabel('Cluster')
    plt.ylabel('Mixing proportion')
    plt.xticks(range(num_clusters))
    
    legend_labels = [f"Cluster {i}: {weights[i]:.3f}" for i in range(num_clusters)]
    plt.legend(bars1, legend_labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

    # Plot 2: Number of Points per Cluster
    plt.subplot(1, 2, 2)
    num_points_per_cluster = torch.bincount(labels)
    
    bars2 = []
    for i in range(len(num_points_per_cluster)):
        if i in unique_labels:
            bar = plt.bar(i, num_points_per_cluster[i].numpy(), color=color_mapping[i])
        else:
            bar = plt.bar(i, num_points_per_cluster[i].numpy(), color='gray')
        bars2.append(bar[0])

    plt.title('Number of points per cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    plt.xticks(range(len(num_points_per_cluster)))

    # Create legends for each bar in Points per Cluster
    legend_labels_points = [f"Cluster {i}: {num_points_per_cluster[i]}" for i in range(len(num_points_per_cluster))]
    plt.legend(bars2, legend_labels_points, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

    plt.tight_layout()

    if savefig:
        plt.savefig(f"plots/{data_folder}/mixing_proportions_{mb.K}_seed_{mb.seed}.png")
    else:
        plt.show()
    plt.close()

def plot_marginals_inference_nd(mb, D):
    pairs = np.triu_indices(D, k=1)  # Generate all unique pairs of samples (i, j)
    vaf = (mb.NV / mb.DP)#/purity
    
    columns=[f"Sample {d+1}" for d in range(D)]
    df = pd.DataFrame(vaf.numpy(), columns=columns)
    mutation_ids = [f"M{i}" for i in range(mb.NV.shape[0])]
    labels = mb.cluster_assignments.detach().numpy()
    df['Label'] = labels
    df['mutation_id'] = mutation_ids

    fig, axes = plt.subplots(1, D, figsize=(4*D, 4))
    axes = axes.flatten()

    unique_labels = sorted(df['Label'].unique())  # Ensures fixed order

    palette = colors[:len(unique_labels)]
    if mb.final_K == mb.K:
        palette = colors
    else:
        palette = colors[:len(unique_labels)]
    
    color_dict = dict(zip(unique_labels, palette))

    # Plot histograms for each sample (only values > 0)
    for ax, col in zip(axes, columns):
        label_sizes = df.groupby('Label')[col].count()
        sorted_labels = label_sizes.sort_values().index

        sns.histplot(
            data=df[df[col] > 0], x=col, hue='Label', 
            palette=color_dict,  # Use fixed colors
            hue_order=sorted_labels,  # Ensure labels are in order from smallest to largest
            ax=ax, bins=100, multiple='layer', alpha=0.7, edgecolor='white'
        )
        ax.set_title(f"{col}")
        ax.set_xlabel(f"{col}")
        ax.set_ylabel("Count")
        # ax.set_xlim([0,1])
        ax.grid(True, alpha=0.3)
    
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    
    plt.tight_layout()
    if mb.savefig:
        plt.savefig(f"plots/{mb.data_folder}/inference_marginals_K_{mb.K}_seed_{mb.seed}.png")
    plt.show()
    plt.close()

def plot_marginals_inference_1d(mb, D):
    vaf = (mb.NV / mb.DP)

    columns=[f"Sample {d+1}" for d in range(D)]
    df = pd.DataFrame(vaf.numpy(), columns=columns)
    mutation_ids = [f"M{i}" for i in range(mb.NV.shape[0])]
    labels = mb.cluster_assignments.detach().numpy()
    df['Label'] = labels
    df['Label'] = df['Label'].astype(str)
    df['mutation_id'] = mutation_ids

    plt.figure()
    unique_labels = sorted(df['Label'].unique())  # Ensures fixed order

    palette = colors[:len(unique_labels)]
    
    if mb.final_K == mb.K:
        palette = colors
    else:
        palette = colors[:len(unique_labels)]
    
    color_dict = dict(zip(unique_labels, palette))

    sns.histplot(
        data=df,x = 'Sample 1', hue='Label', 
        palette=color_dict,
        bins=100, multiple='layer', alpha=0.7, edgecolor='white'
    )
    
    plt.ylabel("Count")
    # ax.set_xlim([0,1])
    plt.grid(True, alpha=0.3)
    
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    
    plt.tight_layout()
    if mb.savefig:
        plt.savefig(f"plots/{mb.data_folder}/inference_marginals_K_{mb.K}_seed_{mb.seed}.png")
    plt.show()
    plt.close()

def plot_marginals_inference(mb):
    D = mb.NV.shape[1]
    if D == 1:
        plot_marginals_inference_1d(mb, D)
    else:
        plot_marginals_inference_nd(mb, D)
        
def plot_scatter_inference(mb):
    """
    Plot the results.
    """
    D = mb.NV.shape[1]
    pairs = np.triu_indices(D, k=1)
    vaf = (mb.NV / mb.DP)
    
    columns=[f"Sample {d+1}" for d in range(D)]
    df = pd.DataFrame(vaf.numpy(), columns=columns)
    mutation_ids = [f"M{i}" for i in range(mb.NV.shape[0])]
    labels = mb.cluster_assignments.detach().numpy()
    df['Cluster'] = labels
    df['mutation_id'] = mutation_ids
    # print(df)
    unique_labels = df['Cluster'].unique()  # Ensures fixed order

    pairs = list(combinations(columns, 2))  # Unique pairs of samples
    
    palette = colors[:len(unique_labels)]
    if mb.final_K == mb.K:
        palette = colors
    else:
        palette = colors[:len(unique_labels)]

    if len(pairs) == 1:
        # If there is only one pair, plot it in a single figure
        x_col, y_col = pairs[0]
        plt.figure(figsize=(5, 5))
        ax = sns.scatterplot(data=df, x=x_col, y=y_col, hue='Cluster', palette=palette, s=20, alpha = 0.7, edgecolor='none') # 'tab20'
        ax.grid(True,linewidth=0.4, color='grey', alpha=0.7)
        plt.title(f'{x_col} vs {y_col}')
        ax.legend(title=None)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
    else:
        # General case for more than 1 pair
        num_pairs = len(pairs)
        ncols = min(3, num_pairs)
        nrows = (num_pairs + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
        axes = axes.flatten()

        for ax, (x_col, y_col) in zip(axes, pairs):
            sns.scatterplot(data=df, x=x_col, y=y_col, hue='Cluster', palette=palette, ax=ax, alpha = 0.7, s=20, edgecolor='none') # 'tab20'
            ax.grid(True, linewidth=0.4, color='grey', alpha=0.7)
            ax.set_title(f'{x_col} vs {y_col}')
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_xlim([0,1])
            ax.set_ylim([0,1])
            ax.legend(title=None)  

        # Turn off extra axes
        for ax in axes[len(pairs):]:
            ax.axis('off')

        plt.tight_layout()
    if mb.savefig:
        plt.savefig(f"plots/{mb.data_folder}/inference_K_{mb.K}_seed_{mb.seed}.png")

    plt.show()
    plt.close()