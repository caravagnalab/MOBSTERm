import numpy as np

from torch import tensor, is_tensor, bincount

from scipy.stats import beta, pareto
from .BoundedPareto import BoundedPareto
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from itertools import combinations

from .aux_functions import *

def plot_deltas(mb):
    """
    Plots delta tests

    Args:
        mb (dict): The result of `MOBSTERM.fit()` or its 'best_fit' value.

    Returns:
        matplotlib.figure.Figure: A figure representing the delta tests.
    """

    best_fit = extract_best_fit_data(mb)
    deltas = best_fit['model_parameters']['delta_param']
    K = deltas.shape[0]

    fig, ax = build_figure(K, 1, figsizes=[(8, 1.5), (8, K * 0.6)])

    n_comps = best_fit['n_components']
    seed = best_fit['seed']
    fig.suptitle(f"Delta with K={n_comps}, seed={seed}", fontsize=12, y=0.98)

    fig.subplots_adjust(top=0.93, hspace=0.2, right=0.8)

    # Define a shared color scale
    norm = plt.Normalize(vmin=0, vmax=1)
    cmap = sns.color_palette("crest", as_cmap=True)

    samples = best_fit['sample_names']

    for k in range(deltas.shape[0]):
        sns.heatmap(deltas[k], ax=ax[k], vmin=0, vmax=1, cmap=cmap,
                    cbar=False)

        num_rows = deltas[k].shape[0]
        ax[k].set_yticks([i + 0.5 for i in range(num_rows)])
        ax[k].set_yticklabels(samples, rotation=0)

        ax[k].set_xlabel("")
        ax[k].set_ylabel(f"C{k}", labelpad=15, va='center', rotation=0)

        if k == (deltas.shape[0] - 1):
            ax[k].set_xticklabels(['ParetoBinomial', 'BetaBinomial',
                                   'Dirac'], rotation=0)
            ax[k].set_xlabel("Distributions")
        else:
            ax[k].set_xticklabels([])
            ax[k].tick_params(axis='x', which='both', bottom=False,
                              top=False)

        pos = ax[k].get_position()
        new_pos = [pos.x0 + 0.05, pos.y0, pos.width - 0.05, pos.height]
        ax[k].set_position(new_pos)


    cbar_ax = fig.add_axes([0.85, 0.2, 0.02, 0.6])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, cax=cbar_ax)

    plt.close()

    return fig

def plot_responsib(mb):
    """
    Plots responsibilities

    Args:
        mb (dict): The result of `MOBSTERM.fit()` or its 'best_fit' value.

    Returns:
        matplotlib.figure.Figure: A figure representing the responsibilities.
    """

    best_fit = extract_best_fit_data(mb)
    resp = best_fit['model_parameters']['responsib']
    if not is_tensor(resp):
        resp = np.array(resp)

    fig, ax = plt.subplots(nrows=1, ncols=1)
    n_comps = best_fit['n_components']
    seed = best_fit['seed']
    fig.suptitle(f"Responsibilities with K={n_comps}, seed={seed}",
                 fontsize = 14)
    fig.tight_layout()
    sns.heatmap(resp, ax=ax, vmin=0, vmax=1, cmap="crest")
    plt.close()

    return fig

def get_paretos_title(samples, k, d, alpha_pareto, probs_pareto=None):
    alpha_value = round(float(alpha_pareto[k,d]), ndigits=2)
    title = f"{samples[d]} Cluster {k} - alpha {alpha_value}"
    if probs_pareto is None:
        return title

    probs_value = round(float(probs_pareto[k,d]), ndigits=2)
    return f"{title}, p {probs_value}"

def plot_paretos(mb):
    """
    Plots pareto parameters

    Args:
        mb (dict): The result of `MOBSTERM.fit()` or its 'best_fit' value.

    Returns:
        matplotlib.figure.Figure: A figure representing the pareto parameters.
    """

    best_fit = extract_best_fit_data(mb)

    if "probs_pareto_param" in best_fit['model_parameters']:
        probs_pareto = best_fit['model_parameters']['probs_pareto_param']
    else:
        probs_pareto = None

    alpha_pareto = best_fit['model_parameters']['alpha_pareto_param']
    if not is_tensor(alpha_pareto):
        alpha_pareto = np.array(alpha_pareto)

    fig, ax = build_figure(alpha_pareto.shape[0], alpha_pareto.shape[1],
                           figsizes=[(7, 3),
                                     (18, best_fit['used_components'])])

    n_comps = best_fit['n_components']
    seed = best_fit['seed']
    fig.suptitle(f"Pareto with K={n_comps}, seed={seed}", fontsize=14)
    fig.tight_layout()
    x = np.arange(0,0.5,0.001)

    samples = best_fit['sample_names']
    for k in range(alpha_pareto.shape[0]):
        for d in range(alpha_pareto.shape[1]):
            pdf = pareto.pdf(x, alpha_pareto[k,d], scale=0.001)

            if alpha_pareto.shape[1] == 1:
                curr_ax = ax[k]
            else:
                curr_ax = ax[k,d]

            curr_ax.plot(x, pdf, 'r-', lw=1)
            title = get_paretos_title(samples, k, d, alpha_pareto,
                                      probs_pareto)

            curr_ax.set_title(title, fontsize=10)
    plt.close()

    return fig

def get_betas_title(samples, k, d, phi_beta, kappa_beta):
    phi_value = round(float(phi_beta[k,d]), ndigits=2)
    kappa_value = round(float(kappa_beta[k,d]), ndigits=2)
    return f"{samples[d]} Cluster {k} - phi {phi_value}, kappa {kappa_value}"

def plot_betas(mb):
    """
    Plots beta parameters

    Args:
        mb (dict): The result of `MOBSTERM.fit()` or its 'best_fit' value.

    Returns:
        matplotlib.figure.Figure: A figure representing the beta parameters.
    """

    best_fit = extract_best_fit_data(mb)
    phi_beta = best_fit['model_parameters']['phi_beta_param']
    kappa_beta = best_fit['model_parameters']['k_beta_param']

    fig, ax = build_figure(phi_beta.shape[0], phi_beta.shape[1],
                           figsizes=[(7, 3),
                                     (18, best_fit['used_components'])])

    n_comps = best_fit['n_components']
    seed = best_fit['seed']
    fig.suptitle(f"Beta with K={n_comps}, seed={seed}", fontsize=14)
    fig.tight_layout()
    x = np.arange(0,1,0.001)

    samples = best_fit['sample_names']
    for k in range(phi_beta.shape[0]):
        for d in range(phi_beta.shape[1]):
            a = phi_beta[k,d]*kappa_beta[k,d]
            b = (1-phi_beta[k,d])*kappa_beta[k,d]
            pdf = beta.pdf(x, a, b)

            if phi_beta.shape[1] == 1:
                curr_ax = ax[k]
            else:
                curr_ax = ax[k,d]

            curr_ax.plot(x, pdf, 'r-', lw=1)

            title = get_betas_title(samples, k, d, phi_beta, kappa_beta)
            curr_ax.set_title(title, fontsize=10)
    plt.close()

    return fig

def get_color_mapping(best_fit, unique_labels):
    if best_fit['used_components'] == best_fit['n_components']:
        return colors

    return colors[:len(unique_labels)]

def plot_cluster_marginals(mb, plot_null_vaf_values=False):
    """
    Plots cluster marginals

    Args:
        mb (dict): The result of `MOBSTERM.fit()` or its 'best_fit' value.
        plot_null_vaf_values (bool): A Boolean parameter to plot null VAF
                values (default: False).

    Returns:
        matplotlib.figure.Figure: A figure representing the cluster marginals.
    """

    best_fit = extract_best_fit_data(mb)
    a_beta_zeros = tensor(1e-4)
    b_beta_zeros = tensor(1e4)

    delta = best_fit['model_parameters']['delta_param']
    phi_beta = best_fit['model_parameters']['phi_beta_param']
    kappa_beta = best_fit['model_parameters']['k_beta_param']
    alpha = best_fit['model_parameters']['alpha_pareto_param']

    NV = best_fit['NV']
    DP = best_fit['DP']

    labels = best_fit['cluster_id']

    samples = best_fit['sample_names']

    fig, ax = build_figure(best_fit['used_components'], len(samples),
                           figsizes=[(10, 4),
                                     (10, best_fit['used_components']*3)])

    n_comps = best_fit['n_components']
    seed = best_fit['seed']
    fig.suptitle(f"Marginals with K={n_comps}, seed={seed}",fontsize=14)

    x = np.linspace(0.001, 1, 1000)

    unique_labels = np.unique(labels)

    color_mapping = get_color_mapping(best_fit, unique_labels)

    if len(samples) == 1:
        data = NV[:]/DP[:]

    for k in range(best_fit['used_components']):
        for d in range(len(samples)):

            if len(samples) == 1:
                curr_ax = ax[k]
            else:
                curr_ax = ax[k,d]
                data = NV[:,d]/DP[:,d]
                data = data[labels == k]

            if not plot_null_vaf_values:
                data[data > 0]

            maxx = np.argmax(delta[k, d])
            if maxx == 1:
                # plot beta
                a = phi_beta[k,d] * kappa_beta[k,d]
                b = (1-phi_beta[k,d]) * kappa_beta[k,d]
                pdf = beta.pdf(x, a, b)# * weights[k]
                curr_ax.plot(x, pdf, linewidth=1.5, label='Beta', color='r')
                curr_ax.legend()
            elif maxx == 0:
                # plot pareto
                scale = best_fit['model_parameters']['scale_pareto']

                pdf = pareto.pdf(x, alpha[k,d], scale=scale)
                curr_ax.plot(x, pdf, linewidth=1.5, label='Pareto', color='g')
                curr_ax.legend()
            else:
                # private
                pdf = beta.pdf(x, a_beta_zeros, b_beta_zeros) # delta_approx
                curr_ax.plot(x, pdf, linewidth=1.5, label='Dirac', color='b')
                curr_ax.legend()

            if k in unique_labels:
                if maxx == 2:
                    n_bins = 50
                else:
                    n_bins = int(np.ceil(np.sqrt(len(data))))
                    if n_bins < 30:
                        n_bins = 30
                curr_ax.hist(data, density=True, bins=n_bins,
                             color=color_mapping[k], alpha=1,
                             edgecolor='white')
            else:
                # Plot an empty histogram because we know there are
                # no points in that k
                curr_ax.hist([], density=True, bins=30, alpha=1)
            curr_ax.set_title(f"{samples[d]} - Cluster {k}")
            curr_ax.grid(True, color='gray', linestyle='-', linewidth=0.2)
            curr_ax.set_xlim([-0.01,0.7])
    fig.tight_layout()
    plt.close()

    return fig

def plot_mixing_proportions(mb):
    """
    Plots mixing proportions

    Args:
        mb (dict): The result of `MOBSTERM.fit()` or its 'best_fit' value.

    Returns:
        matplotlib.figure.Figure: A figure representing the mixing proportions.
    """

    best_fit = extract_best_fit_data(mb)
    weights = best_fit['model_parameters']['weights_param']

    labels = best_fit['cluster_id']
    if not is_tensor(labels):
        labels = tensor(labels)

    num_clusters = weights.shape[0]
    unique_labels = np.unique(labels)

    color_mapping = get_color_mapping(best_fit, unique_labels)

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(10, 5))

    # Plot 1: Mixing Proportions
    bars1 = []
    for i in range(num_clusters):
        if i in unique_labels:
            bar = axl.bar(i, weights[i], color=color_mapping[i])
        else:
            bar = axl.bar(i, weights[i], color='gray')
        bars1.append(bar[0])  # Store the bar for legend

    axl.set_title('Mixing proportions')
    axl.set_xlabel('Cluster')
    axl.set_ylabel('Mixing proportion')
    axl.set_xticks(range(num_clusters))

    legend_labels = [f"Cluster {i}: {weights[i]:.3f}"
                     for i in range(num_clusters)]
    axl.legend(bars1, legend_labels, loc='upper center',
               bbox_to_anchor=(0.5, -0.15), ncol=2)

    # Plot 2: Number of Points per Cluster
    num_points_per_cluster = bincount(labels)

    bars2 = []
    for i in range(len(num_points_per_cluster)):
        if i in unique_labels:
            bar = axr.bar(i, num_points_per_cluster[i].numpy(),
                          color=color_mapping[i])
        else:
            bar = axr.bar(i, num_points_per_cluster[i].numpy(),
                          color='gray')
        bars2.append(bar[0])

    axr.set_title('Number of points per cluster')
    axr.set_xlabel('Cluster')
    axr.set_ylabel('Count')
    axr.set_xticks(range(len(num_points_per_cluster)))

    # Create legends for each bar in Points per Cluster
    legend_labels_points = [f"Cluster {i}: {num_points_per_cluster[i]}"
                            for i in range(len(num_points_per_cluster))]
    axr.legend(bars2, legend_labels_points, loc='upper center',
               bbox_to_anchor=(0.5, -0.15), ncol=2)

    fig.tight_layout()

    plt.close()

    return fig

def plot_marginals_inference(mb, plot_null_vaf_values=False):
    """
    Plots marginal inference

    Args:
        mb (dict): The result of `MOBSTERM.fit()` or its 'best_fit' value.
        plot_null_vaf_values (bool): A Boolean parameter to plot null VAF
                values (default: True).

    Returns:
        matplotlib.figure.Figure: A figure representing the marginal inference.
    """

    best_fit = extract_best_fit_data(mb)

    NV = best_fit['NV']
    vaf = (NV / best_fit['DP'])

    samples = best_fit['sample_names']
    df = pd.DataFrame(vaf, columns=samples)
    mutation_ids = [f"M{i}" for i in range(NV.shape[0])]
    df['Label'] = [f"Cluster {x}" for x in best_fit['cluster_id']]
    df['mutation_id'] = mutation_ids

    fig, axes = plt.subplots(1, len(samples), figsize=(4*len(samples), 4))
    if len(samples)==1:
        axes = [axes]
    else:
        axes = axes.flatten()

    unique_labels = sorted(df['Label'].unique())  # Ensures fixed order

    color_mapping = get_color_mapping(best_fit, unique_labels)

    palette = dict(zip(unique_labels, color_mapping))

    # Plot histograms for each sample (only values > 0)
    for ax, col in zip(axes, samples):
        label_sizes = df.groupby('Label')[col].count()
        sorted_labels = label_sizes.sort_values().index

        sns.histplot(
            data=(df if plot_null_vaf_values else df[df[col] > 0]),
            x=col, hue='Label', palette=palette,
            hue_order=sorted_labels,  # Ensure labels are in order from
                                      # smallest to largest
            ax=ax, bins=100, multiple='layer', alpha=0.7, edgecolor='white'
        )
        ax.set_title(f"{col}")
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)
        ax.get_legend().set_title("")

    fig.subplots_adjust(wspace=0.3, hspace=0.3)

    fig.tight_layout()
    plt.close()

    return fig

def plot_scatter_inference(mb):
    """
    Plots scatter inference

    Args:
        mb (dict): The result of `MOBSTERM.fit()` or its 'best_fit' value.

    Returns:
        matplotlib.figure.Figure: A figure representing the scatter inference.

    Raises:
        ValueError: If the `MOBSTERm.fit()` considered only one sample.
    """

    best_fit = extract_best_fit_data(mb)
    samples = best_fit['sample_names']
    if len(samples) == 1:
        raise ValueError('`plot_scatter_inference()` requires two '
                         + 'samples at least')

    NV = best_fit['NV']
    vaf = (NV / best_fit['DP'])

    df = pd.DataFrame(vaf, columns=samples)
    mutation_ids = [f"M{i}" for i in range(NV.shape[0])]
    df['Cluster'] = [f'Cluster {x}' for x in best_fit['cluster_id']]
    df['mutation_id'] = mutation_ids

    unique_labels = df['Cluster'].unique()  # Ensures fixed order

    pairs = list(combinations(samples, 2))  # Unique pairs of samples

    color_mapping = colors[:len(unique_labels)]

    # General case for more than 1 pair
    num_pairs = len(pairs)
    ncols = min(3, num_pairs)
    nrows = (num_pairs + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    if nrows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for ax, (x_col, y_col) in zip(axes, pairs):
        sns.scatterplot(data=df, x=x_col, y=y_col, hue='Cluster',
                        palette=color_mapping, ax=ax, alpha = 0.7, s=20,
                        edgecolor='none')
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

    fig.tight_layout()

    plt.close()

    return fig
