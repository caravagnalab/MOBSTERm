import numpy as np
import pandas as pd
import pyro.distributions as dist
from scipy import stats
import torch
import seaborn as sns
import pyro

import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns

import copy
import json

import argparse

import sys
import os
from utils_functions import *
import model_mobster_gmm as mobster_mv

# Set the parent directory
parent_dir = "../../"
sys.path.insert(0, parent_dir)
# import new_model as mobster_mv
from utils.BoundedPareto import BoundedPareto
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
    purity = args.p
    coverage = args.cov

    create_folder(N,K,D,purity,coverage)
    

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
        
        # Sample mixing proportions for clusters and multiply by N to obtain the number of data in each cluster
        pi = sample_mixing_prop(K, min_value=0.05) * N
        # print(pi/N)
        # print(pi)
        # pi = dist.Dirichlet(torch.ones(K)).sample() * N  # Number of data in each cluster
        pi = np.round(pi.numpy()).astype('int')

        # Adjust proportions to ensure they sum to N
        # print("np.sum(pi)", np.sum(pi))
        if np.sum(pi) < N:
            diff = N - np.sum(pi)
            pi[-1] += diff
        elif np.sum(pi) > N:
            diff = np.sum(pi) - N
            pi[-1] -= diff
        # print("np.sum(pi)", np.sum(pi))
        NV, DP, cluster_labels, type_labels_data, type_labels_cluster, phi_param_data, kappa_param_data, alpha_param_data, phi_param_cluster, kappa_param_cluster, alpha_param_cluster  = generate_data_new_model_final(N, K, pi, D, purity, coverage)

        plot_scatter_real(NV, DP, N, K, D, type_labels_cluster, cluster_labels, idx, purity, coverage)  
        plot_marginals_real(NV, DP, N, K, D, type_labels_cluster, cluster_labels, phi_param_cluster, kappa_param_cluster, alpha_param_cluster, idx, purity, coverage)
        
        # Run the model
        if K != 3:
            K_list = [K - 2, K - 1, K, K + 1, K + 2, K + 3]
        else:
            K_list = [K - 1, K, K + 1, K + 2, K + 3]
        
        seed_list = [40,41]
        mb_list, best_K, best_seed = mobster_mv.fit(NV, DP, num_iter = num_iter, K = K_list, seed = seed_list, lr = 0.01, purity = purity)
        
        mb = mb_list[K_list.index(best_K)]
        
        pred_cluster_labels, pred_type_labels_data, pred_phi_param_data, pred_kappa_param_data, pred_alpha_param_data = retrieve_info(mb, N, D)
        
        plot_initialization(mb, N, K, idx, purity, coverage)
        plot_final(mb, N, K, idx, purity, coverage)
        plot_final_marginals(mb, N, K, D, idx, purity, coverage)
        # plt.savefig(f"plots/p_{str(purity).replace('.', '')}_cov_{coverage}/D_{D}/real_marginals/N_{N}_K_{K}_D_{D}_real_{idx}.png")

        general_folder = f"plots/p_{str(purity).replace('.', '')}_cov_{coverage}/D_{D}/"
        # for m in range(len(mb_list)):
        data_folder = general_folder + "responsib_deltas/"
        plot_deltas_gen(mb, N, K, D, idx, savefig = True, data_folder = data_folder)
        plot_responsib_gen(mb, N, K, D, idx, savefig = True, data_folder = data_folder)
        
        data_folder = general_folder + "betas_paretos/"
        plot_paretos_gen(mb, N, K, D, idx, savefig = True, data_folder = data_folder)
        plot_betas_gen(mb, N, K, D, idx, savefig = True, data_folder = data_folder)

        data_folder = general_folder + "model_selection/"
        plot_model_selection_gen(mb_list, N, K_list, K, seed_list, D, idx, savefig = True, data_folder = data_folder)

        data_folder = general_folder + "loss_lks/"
        plot_loss_lks_gen(mb, N, K, D, idx, savefig = True, data_folder = data_folder)

        
        column_names = ['NV', 'DP', 'True_cluster', 'Pred_cluster', 
            'True_distribution', 'Pred_distribution', 'True_phi', 'Pred_phi', 
            'True_kappa', 'Pred_kappa', 'True_alpha', 'Pred_alpha']
        df = pd.DataFrame(columns=column_names)

        df['NV'] = [[round(val, 3) for val in row.tolist()] for row in NV]
        df['DP'] = [[round(val, 3) for val in row.tolist()] for row in DP]
        df['True_cluster'] = cluster_labels.tolist()  # No rounding needed, they are integer labels
        df['Pred_cluster'] = pred_cluster_labels.tolist()
        df['True_distribution'] = [[round(val, 3) for val in row.tolist()] for row in type_labels_data]
        df['Pred_distribution'] = [[round(val, 3) for val in row.tolist()] for row in pred_type_labels_data]
        df['True_phi'] = [[round(val, 3) for val in row.tolist()] for row in phi_param_data]
        df['Pred_phi'] = [[round(val, 3) for val in row.tolist()] for row in pred_phi_param_data]
        df['True_kappa'] = [[round(val, 3) for val in row.tolist()] for row in kappa_param_data]
        df['Pred_kappa'] = [[round(val, 3) for val in row.tolist()] for row in pred_kappa_param_data]
        df['True_alpha'] = [[round(val, 3) for val in row.tolist()] for row in alpha_param_data]
        df['Pred_alpha'] = [[round(val, 3) for val in row.tolist()] for row in pred_alpha_param_data]

        csv_filename = f'./results/p_{str(purity).replace('.', '')}_cov_{coverage}/D_{D}/csv/N_{N}_K_{K}_D_{D}_df_{idx}.csv'
        df.to_csv(csv_filename, index=False)

        dict_copy = copy.copy(mb.__dict__)
        dict_copy = convert_to_list(dict_copy)
        
        filename = f'saved_objects/p_{str(purity).replace('.', '')}_cov_{coverage}/D_{D}/N_{N}_K_{K}_D_{D}_pred_{idx}.txt'
        with open(filename, 'w') as f:
            f.write(json.dumps(dict_copy) + '\n')
            
        pred_cluster_labels_init, pred_type_labels_data_init, pred_phi_param_data_init, pred_alpha_param_data_init = retrieve_info_init(mb, N, D)
        column_names = ['NV', 'DP', 'True_cluster', 'Pred_cluster', 
            'True_distribution', 'Pred_distribution', 'True_phi', 'Pred_phi', 
            'True_alpha', 'Pred_alpha']
        df = pd.DataFrame(columns=column_names)

        df['NV'] = [[round(val, 3) for val in row.tolist()] for row in NV]
        df['DP'] = [[round(val, 3) for val in row.tolist()] for row in DP]
        df['True_cluster'] = cluster_labels.tolist()  # No rounding needed, they are integer labels
        df['Pred_cluster'] = pred_cluster_labels_init.tolist()
        df['True_distribution'] = [[round(val, 3) for val in row.tolist()] for row in type_labels_data]
        df['Pred_distribution'] = [[round(val, 3) for val in row.tolist()] for row in pred_type_labels_data_init]
        df['True_phi'] = [[round(val, 3) for val in row.tolist()] for row in phi_param_data]
        df['Pred_phi'] = [[round(val, 3) for val in row.tolist()] for row in pred_phi_param_data_init]
        df['True_alpha'] = [[round(val, 3) for val in row.tolist()] for row in alpha_param_data]
        df['Pred_alpha'] = [[round(val, 3) for val in row.tolist()] for row in pred_alpha_param_data_init]

        csv_filename = f'./results/p_{str(purity).replace('.', '')}_cov_{coverage}/D_{D}/init_csv/N_{N}_K_{K}_D_{D}_df_{idx}.csv'
        df.to_csv(csv_filename, index=False)

        # Measure NMI
        true_labels = cluster_labels
        predicted_labels = pred_cluster_labels

        nmi = normalized_mutual_info_score(true_labels, predicted_labels)
        nmi_list.append(nmi)

        ari = adjusted_rand_score(true_labels, predicted_labels)
        ari_list.append(ari)

        # Measure NMI init
        predicted_labels_init = pred_cluster_labels_init

        nmi_init = normalized_mutual_info_score(true_labels, predicted_labels_init)
        nmi_list_init.append(nmi_init)

        ari_init = adjusted_rand_score(true_labels, predicted_labels_init)
        ari_list_init.append(ari_init)

    filename = f"results/p_{str(purity).replace('.', '')}_cov_{coverage}/D_{D}/nmi/nmi_N_{N}_K_{K}_D_{D}.txt"
    with open(filename, "w") as file:
        for item in nmi_list:
            file.write(f"{item}\n")  # Writing each item on a new line

    filename = f"results/p_{str(purity).replace('.', '')}_cov_{coverage}/D_{D}/ari/ari_N_{N}_K_{K}_D_{D}.txt"
    with open(filename, "w") as file:
        for item in ari_list:
            file.write(f"{item}\n")  # Writing each item on a new line

    filename = f"results/p_{str(purity).replace('.', '')}_cov_{coverage}/D_{D}/init_nmi/nmi_N_{N}_K_{K}_D_{D}.txt"
    with open(filename, "w") as file:
        for item in nmi_list_init:
            file.write(f"{item}\n")  # Writing each item on a new line

    filename = f"results/p_{str(purity).replace('.', '')}_cov_{coverage}/D_{D}/init_ari/ari_N_{N}_K_{K}_D_{D}.txt"
    with open(filename, "w") as file:
        for item in ari_list_init:
            file.write(f"{item}\n")  # Writing each item on a new line

        """
        # Compute accuracy on predicted distributions (Pareto or Beta)
        true_distributions = np.array([[0. if elem == 'P' else 1. for elem in sublist] for sublist in unique_types])
        delta = mb.params["delta_param"]
        pred_distribtions = np.zeros((mb.K, mb.NV.shape[1]))
        for k in range(mb.K):
            for d in range(mb.NV.shape[1]):
                delta_kd = delta[k, d]
                maxx = torch.argmax(delta_kd)
                if maxx == 0: # Pareto
                    pred_distribtions[k,d] = 0
                else:
                    pred_distribtions[k,d] = 1
        accuracy = accuracy_score(true_labels, predicted_labels)
        acc_list.append(accuracy)
        
        filename = f"results/acc/ari_N_{N}_sub_{K1}"
        with open(filename, "w") as file:
            for item in acc_list:
                file.write(f"{item}\n")  # Writing each item on a new line

        # Confusion matrix
        conf_matrix = confusion_matrix(true_distributions.flatten(), pred_distribtions.flatten(), labels=[0, 1])
        conf_matrix_list.append(conf_matrix)
        filename = f"results/conf_matrix/ari_N_{N}_sub_{K1}"
        with open(filename, "w") as file:
            for item in conf_matrix_list:
                file.write(f"{item}\n")  # Writing each item on a new line
        """