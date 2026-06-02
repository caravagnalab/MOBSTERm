from numpy import array as np_array
from matplotlib.pyplot import subplots

colors = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#cccc33",
    "#a65628", "#f781bf", "#999999", "#000000",  # First 10 colors (Set1)
    "#46f0f0", "#f032e6", "#bcf60c", "#fabed4", "#008080", "#e6beff",
    "#9a6324", "#fffac8", "#800000", "#aaffc3", "#808000", "#ffd8b1",
    "#000075", "#808080", "#d3a6f3", "#ff9cdd", "#73d7b0"
    ]

def extract_best_fit_data(mb):
    """
    Extracts the 'best_fit' value if present

    Args:
        mb (dict): The result of `MOBSTERM.fit()` or its 'best_fit' value.

    Returns:
        `mb['best_fit']` if `mb` has 'best_fit' is among `mb`'s keys;
        `mb` otherways.
    """
    if isinstance(mb, dict):
        if 'best_fit' in mb:
            return mb['best_fit']

    return mb

def build_figure(nrows, ncols, figsizes):
    if nrows == 1:
        fig, ax = subplots(nrows=nrows, ncols=ncols,
                           figsize=figsizes[0])
        ax = np_array([ax])

        return fig, ax

    return subplots(nrows=nrows, ncols=ncols,
                    figsize=figsizes[1])

def get_paretos_title(samples, k, d, alpha_pareto, probs_pareto=None):
    alpha_value = round(float(alpha_pareto[k,d]), ndigits=2)
    title = f"{samples[d]} Cluster {k} - alpha {alpha_value}"
    if probs_pareto is None:
        return title

    probs_value = round(float(probs_pareto[k,d]), ndigits=2)
    return f"{title}, p {probs_value}"

def get_betas_title(samples, k, d, phi_beta, kappa_beta):
    phi_value = round(float(phi_beta[k,d]), ndigits=2)
    kappa_value = round(float(kappa_beta[k,d]), ndigits=2)
    return f"{samples[d]} Cluster {k} - phi {phi_value}, kappa {kappa_value}"

def get_color_mapping(best_fit, unique_labels):
    if best_fit['used_components'] == best_fit['n_components']:
        return colors

    return colors[:len(unique_labels)]

def build_mutation_ids(mutation_df):
    """
    Builds a mutation engine column for the mutations in a Pandas dataframe

    Args:
        mutation_df(pandas.Dataframe) A dataframe containing a set of
            mutations. The dataframe must have the columns 'chr', 'from',
            'ref', and 'alt'.

    Returns:
        pandas.Dataframe: A column containing a unique identifier for each
        mutation in `mutation_df`.
    """
    for column_name in ['chr', 'from', 'ref', 'alt']:
        if column_name not in mutation_df:
            raise ValueError(f'The input file miss the "{column_name}" column')

    return (mutation_df['chr'].astype(str)
            + '_' + mutation_df['from'].astype(str)
            + '_' + mutation_df['ref'].astype(str)
            + '_' + mutation_df['alt'].astype(str))