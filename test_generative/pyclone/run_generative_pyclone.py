import numpy as np
import pandas as pd
import pyro.distributions as dist
from scipy import stats
import torch
import subprocess as sb

from itertools import combinations

import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns

import copy
import json

import argparse
from natsort import natsorted
import re

import sys
import os
from utils_functions_pyclone import *

# Set the parent directory
parent_dir = "../../"
sys.path.insert(0, parent_dir)
from utils.create_beta_pareto_dataset import *


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run with parameters --N --K --D --p")
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--D", type=int, required=True)
    parser.add_argument("--p", type=float, default=1.) # purity
    parser.add_argument("--cov", type=int, default=100) # coverage

    num_iter = 2000
    num_dataset = 15

    args = parser.parse_args()
    N = args.N  # number of mutations
    K = args.K  # number of clusters
    D = args.D  # number of samples
    # purity = args.p
    purity=[args.p]*D
    coverage = args.cov

    create_folder(N,K,D,purity[0],coverage)
    

    nmi_list = []
    ari_list = []
    nmi_list_init = []
    ari_list_init = []
    conf_matrix_list = []
    seed = 0
    for idx in range(num_dataset):
        seed1 = seed+idx+K+N
        pyro.set_rng_seed(seed1)
        torch.manual_seed(seed1)
        np.random.seed(seed1)
        
        NV, DP, cluster_labels, type_labels_data, type_labels_cluster, phi_param_data, kappa_param_data, alpha_param_data, phi_param_cluster, kappa_param_cluster, alpha_param_cluster  = generate_data_new_model_final(N, K, D, purity, coverage, seed1)

        # purity=purity[0]
        plot_scatter_real(NV, DP, N, K, D, type_labels_cluster, cluster_labels, idx, purity[0], coverage)  
        plot_marginals_real(NV, DP, N, K, D, type_labels_cluster, cluster_labels, phi_param_cluster, kappa_param_cluster, alpha_param_cluster, idx, purity[0], coverage)
        
        # Generate mutation IDs
        mutation_ids = [f"M{i}" for i in range(N)]

        # Create D DataFrames
        dataframes = []
        for d in range(D):
            df = pd.DataFrame({
                "alt_counts": NV[:, d].numpy().astype(int),
                "ref_counts": DP[:, d].numpy().astype(int),
                "mutation_id": mutation_ids,
                "sample_id": f"S_{d}",
                "major_cn": 1,
                "minor_cn": 1,
                "normal_cn": 2,
                "tumour_content": purity[0],
                "VAF": NV[:, d].numpy()/DP[:, d].numpy()
            })
            dataframes.append(df)
        data = pd.concat(dataframes, axis=0, ignore_index=True)
        data_path = f'data/p_{str(purity[0]).replace(".", "")}_cov_{coverage}/D_{D}/N_{N}_K_{K}_D_{D}_real_{idx}.tsv'
        data.to_csv(data_path,  sep='\t', index=False)

        fit_path = f'results/p_{str(purity[0]).replace(".", "")}_cov_{coverage}/D_{D}/fit_files/N_{N}_K_{K}_D_{D}_fit_{idx}.h5'
        best_fit_path = f'results/p_{str(purity[0]).replace(".", "")}_cov_{coverage}/D_{D}/fit_files/N_{N}_K_{K}_D_{D}_best_fit_{idx}.h5'
        if K == 4:
            max_K = 10
        else:
            max_K = 2*K 
        sb.call('pyclone-vi fit -i '+data_path+' -o '+fit_path+' -c '+  str(max_K) + ' -d beta-binomial -r 10', shell = True)
        sb.call('pyclone-vi write-results-file -i '+fit_path+' -o '+best_fit_path, shell = True)

        fit = pd.read_csv(best_fit_path, sep='\t')
        data['sample_id'] = data['sample_id'].astype(str)
        fit['sample_id'] = fit['sample_id'].astype(str)

        final_df = data.merge(fit, on=['mutation_id', 'sample_id'], how='left')
        fitted_data = pd.merge(data, fit, how = 'outer', on=['mutation_id','sample_id'])
        table_to_print = fitted_data.pivot_table(index = ['mutation_id','cluster_id'], columns = 'sample_id', values = 'VAF', aggfunc = 'first').reset_index()
        
        plot_scatter_final(NV, DP, D, table_to_print, N, K, idx, purity[0], coverage)

        result = (
            fitted_data.pivot_table(
                index=['mutation_id', 'cluster_id'],  # Grouping columns
                columns='sample_id',  # Columns to pivot on
                values=['alt_counts', 'ref_counts', 'VAF', 'cellular_prevalence'],  # Values to pivot
                aggfunc='first'  # In case of duplicates
            )
            .reset_index()  # Reset index for a flat DataFrame
        )

        # Flatten the MultiIndex columns
        result.columns = ['_'.join(filter(None, col)).strip('_') for col in result.columns]
        # result.sort_values(by='mutation_id').reset_index(drop=True)
        result_sorted = result.reindex(natsorted(result.index, key=lambda x: result.loc[x, 'mutation_id'])).reset_index(drop=True)
        
        column_prefixes = ['alt_counts', 'ref_counts', 'cellular_prevalence', 'VAF']
        columns_to_unify = {
            prefix: [col for col in result_sorted.columns if re.match(f"{prefix}_S_\\d+", col)]
            for prefix in column_prefixes
        }

        for new_col, cols in columns_to_unify.items():
            result_sorted[new_col] = result_sorted[cols].values.tolist()

        result_sorted = result_sorted.drop(columns=[col for cols in columns_to_unify.values() for col in cols])
        result_sorted = result_sorted.drop(columns=['VAF', 'mutation_id'])
        result_sorted.rename(columns={"alt_counts": "NV", 
        "ref_counts": "DP", "cluster_id": "Pred_cluster",
        "cellular_prevalence": "Pred_phi"}, inplace=True)
        
        # NOW THIS FINAL RESULT SORTED IS THE ONE I CAN USE TO SAVE THE CSV
        # I need to add the true cluster assignment, the true center phi and the true distribution type
        
        # ------------------ #
        
        # column_names = ['NV', 'DP', 'True_cluster', 'Pred_cluster', 
        #     'True_distribution', 'Pred_distribution', 'True_phi', 'Pred_phi', 
        #     'True_kappa', 'Pred_kappa', 'True_alpha', 'Pred_alpha']
        # df = pd.DataFrame(columns=column_names)

        # df['NV'] = [[round(val, 3) for val in row.tolist()] for row in NV]
        # df['DP'] = [[round(val, 3) for val in row.tolist()] for row in DP]
        result_sorted['True_cluster'] = cluster_labels.tolist()  # No rounding needed, they are integer labels
        result_sorted['True_distribution'] = [[round(val, 3) for val in row.tolist()] for row in type_labels_data]
        result_sorted['True_phi'] = [[round(val, 3) for val in row.tolist()] for row in phi_param_data]
        result_sorted['True_kappa'] = [[round(val, 3) for val in row.tolist()] for row in kappa_param_data]
        result_sorted['True_alpha'] = [[round(val, 3) for val in row.tolist()] for row in alpha_param_data]
        
        csv_filename = f'./results/p_{str(purity[0]).replace('.', '')}_cov_{coverage}/D_{D}/csv/N_{N}_K_{K}_D_{D}_df_{idx}.csv'
        result_sorted.to_csv(csv_filename, index=False)


        # Measure NMI
        pred_cluster_labels = np.array(result_sorted['Pred_cluster'])
        true_labels = np.array(cluster_labels)
        predicted_labels = pred_cluster_labels

        nmi = normalized_mutual_info_score(true_labels, predicted_labels)
        nmi_list.append(nmi)

        ari = adjusted_rand_score(true_labels, predicted_labels)
        ari_list.append(ari)


    filename = f"results/p_{str(purity[0]).replace('.', '')}_cov_{coverage}/D_{D}/nmi/nmi_N_{N}_K_{K}_D_{D}.txt"
    with open(filename, "w") as file:
        for item in nmi_list:
            file.write(f"{item}\n")  # Writing each item on a new line

    filename = f"results/p_{str(purity[0]).replace('.', '')}_cov_{coverage}/D_{D}/ari/ari_N_{N}_K_{K}_D_{D}.txt"
    with open(filename, "w") as file:
        for item in ari_list:
            file.write(f"{item}\n")  # Writing each item on a new line