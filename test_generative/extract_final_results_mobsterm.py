from sklearn.metrics import normalized_mutual_info_score, accuracy_score
import numpy as np
import torch
import pandas as pd
import ast
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os
import re
import seaborn as sns
import pickle


def compute_pairwise_nmi(true_labels, predicted_labels, threshold=0.5):
    unique_true = np.unique(np.array(true_labels, dtype = 'int'))
    unique_pred = np.unique(predicted_labels)
    # Initialize mapping and NMI matrix
    nmi_matrix = np.zeros((len(unique_true), len(unique_pred)))
    label_mapping = {}
    
    # Construct pairwise NMI matrix
    # Rows: true labels, Columns: predicted labels
    for i, true_label in enumerate(unique_true):
        for j, pred_label in enumerate(unique_pred):
            true_mask = (true_labels == true_label).astype(int)
            pred_mask = (predicted_labels == pred_label).astype(int)
            nmi_matrix[i, j] = normalized_mutual_info_score(true_mask, pred_mask)
    
    # Select the best matching predicted label for each true label
    for i, true_label in enumerate(unique_true):
        best_pred_index = np.argmax(nmi_matrix[i, :])  # Index of the best match (i.e. column)
        best_nmi = nmi_matrix[i, best_pred_index]

        if best_nmi >= threshold:
            label_mapping[true_label] = unique_pred[best_pred_index]

    return label_mapping, nmi_matrix

if __name__ == "__main__":
    # 4*9*15 = 540 - 3*15 = 495 (no D=2 and K = 15) files for each combination of purity and coverage
# => 540 * 6

    os.chdir("subclonal_deconvolution_mv/test_generative/")
    D_values = [2,3,4]
    K_values = [15,4,6,8]
    N_values = [5000, 10000, 15000]
    for purity in [0.7,0.9,1.0]:
        for coverage in [70,100]:
            print('PURITY: ', purity, 'COVERAGE:', coverage)
            general_folder_gmm = f"./mobsterm_and_mobster/results/p_{str(purity).replace('.', '')}_cov_{coverage}/"
            true_phi_list = []
            true_kappa_list = []
            true_alpha_list = []

            pred_phi_list = []
            pred_kappa_list = []
            pred_alpha_list = []

            # Regular expression pattern to match the file names and extract N, K, D, and df values
            pattern = re.compile(r'N_(\d+)_K_(\d+)_D_(\d+)_df_(\d+)\.csv')
            cluster_dist_total = []

            accuracy_gmm = {}
            accuracy_beta = {}
            accuracy_pareto = {}
            accuracy_dirac = {}

            mae_phi = {}
            mae_kappa = {}
            mae_alpha = {}

            acc_data_points_gmm = {}
            accuracy_data_points_beta = {}
            accuracy_data_points_pareto = {}
            accuracy_data_points_dirac = {}

            for N in N_values:
                for K in K_values:
                    for D in D_values:
                        if not (K == 15 and D == 2):  # invalid combination
                            key = f"N_{N}_K_{K}_D_{D}"
                            accuracy_gmm[key] = []
                            accuracy_beta[key] = []
                            accuracy_pareto[key] = []
                            accuracy_dirac[key] = []

                            mae_phi[key] = []
                            mae_kappa[key] = []
                            mae_alpha[key] = []

                            acc_data_points_gmm[key] = []
                            accuracy_data_points_beta[key] = []
                            accuracy_data_points_pareto[key] = []
                            accuracy_data_points_dirac[key] = []

            # accuracy_gmm = {f"N_{N}_K_{K}_D_{D}": [] for N in N_values for K in K_values for D in D_values}
            # accuracy_beta = {f"N_{N}_K_{K}_D_{D}": [] for N in N_values for K in K_values for D in D_values}
            # accuracy_pareto = {f"N_{N}_K_{K}_D_{D}": [] for N in N_values for K in K_values for D in D_values}
            # accuracy_dirac = {f"N_{N}_K_{K}_D_{D}": [] for N in N_values for K in K_values for D in D_values}

            # mae_phi = {f"N_{N}_K_{K}_D_{D}": [] for N in N_values for K in K_values for D in D_values}
            # mae_kappa = {f"N_{N}_K_{K}_D_{D}": [] for N in N_values for K in K_values for D in D_values}
            # mae_alpha = {f"N_{N}_K_{K}_D_{D}": [] for N in N_values for K in K_values for D in D_values}


            # acc_data_points_gmm = {f"N_{N}_K_{K}_D_{D}": [] for N in N_values for K in K_values for D in D_values}
            # accuracy_data_points_beta = {f"N_{N}_K_{K}_D_{D}": [] for N in N_values for K in K_values for D in D_values}
            # accuracy_data_points_pareto = {f"N_{N}_K_{K}_D_{D}": [] for N in N_values for K in K_values for D in D_values}
            # accuracy_data_points_dirac = {f"N_{N}_K_{K}_D_{D}": [] for N in N_values for K in K_values for D in D_values}


            true_phi_dic = []
            true_kappa_dic = []
            true_alpha_dic = []
            pred_phi_dic = []
            pred_kappa_dic = []
            pred_alpha_dic = []

            true_n_K = []
            pred_n_K = []

            idx = 0
            for dim in D_values:
                directory = general_folder_gmm + f"D_{dim}/csv"
                for filename in os.listdir(directory):
                    # Check if the file matches the pattern
                    match = pattern.match(filename)
                    if match:
                        # Extract N, K, D, and df values from the file name
                        N, K, D, df = map(int, match.groups())
                        if not (K == 15 and D == 2): 
                        # if idx < 15:
                            print('idx: ', idx, 'N ', N, 'K ', K, 'D ', D, 'idx ',df)
                            idx+=1
                            file_path = os.path.join(directory, filename)
                            df_data = pd.read_csv(file_path)
                            if 'True_cluster' in df_data.columns and 'Pred_cluster' in df_data.columns:
                                true_labels = df_data['True_cluster'].tolist()
                                pred_labels = df_data['Pred_cluster'].tolist()

                                true_n_K.append(len(np.unique(true_labels)))
                                pred_n_K.append(len(np.unique(pred_labels)))
                                

                                nmi_threshold = 0.4 # 0.5
                                label_mapping, nmi_matrix = compute_pairwise_nmi(true_labels, pred_labels, threshold=nmi_threshold)
                                true_labels_match = list(label_mapping.keys())
                                pred_labels_match = list(label_mapping.values())

                                # Extract true and predicted distributions from the csv
                                true_dist = df_data['True_distribution'].apply(ast.literal_eval)
                                true_dist = torch.tensor(true_dist, dtype=torch.int)
                                pred_dist = df_data['Pred_distribution'].apply(ast.literal_eval)
                                pred_dist = torch.tensor(pred_dist, dtype=torch.int)
                                """
                                COMPUTE VALUES WITH MATCHED CLUSTERS
                                """
                                first_occurrence_indices_true = [true_labels.index(label) for label in true_labels_match]
                                true_dist_match = true_dist[first_occurrence_indices_true].ravel()
                                
                                first_occurrence_indices_pred = [pred_labels.index(label) for label in pred_labels_match]
                                pred_dist_match = pred_dist[first_occurrence_indices_pred].ravel()
                                
                                true_phi = torch.tensor(df_data['True_phi'].apply(ast.literal_eval), dtype=torch.float)
                                pred_phi = torch.tensor(df_data['Pred_phi'].apply(ast.literal_eval), dtype=torch.float)
                                true_phi = true_phi[first_occurrence_indices_true].ravel()
                                pred_phi = pred_phi[first_occurrence_indices_pred].ravel()
                                true_alpha = torch.tensor(df_data['True_alpha'].apply(ast.literal_eval), dtype=torch.float)
                                pred_alpha = torch.tensor(df_data['Pred_alpha'].apply(ast.literal_eval), dtype=torch.float)
                                true_alpha = true_alpha[first_occurrence_indices_true].ravel()
                                pred_alpha = pred_alpha[first_occurrence_indices_pred].ravel()
                                true_kappa = torch.tensor(df_data['True_kappa'].apply(ast.literal_eval), dtype=torch.float)
                                pred_kappa = torch.tensor(df_data['Pred_kappa'].apply(ast.literal_eval), dtype=torch.float)
                                true_kappa = true_kappa[first_occurrence_indices_true].ravel()
                                pred_kappa = pred_kappa[first_occurrence_indices_pred].ravel()
                                diff_phi = []
                                diff_alpha = []
                                diff_kappa = []
                                for p in range(len(pred_dist_match)):
                                    if pred_dist_match[p] == 1 and true_dist_match[p] == 1: # beta
                                        diff_phi.append(np.abs(true_phi[p].item() - pred_phi[p].item()))
                                        diff_kappa.append(np.abs(true_kappa[p].item() - pred_kappa[p].item()))
                                        true_phi_dic.append(true_phi[p].item())
                                        true_kappa_dic.append(true_kappa[p].item())
                                        pred_phi_dic.append(pred_phi[p].item())
                                        pred_kappa_dic.append(pred_kappa[p].item())
                                    if pred_dist_match[p] == 0 and true_dist_match[p] == 0: # pareto
                                        diff_alpha.append(np.abs(true_alpha[p].item() - pred_alpha[p].item()))
                                        true_alpha_dic.append(true_alpha[p].item())
                                        pred_alpha_dic.append(pred_alpha[p].item())
                                
                                if len(diff_phi) > 0:
                                    mae_phi[f"N_{N}_K_{K}_D_{D}"].append(np.mean(diff_phi))
                                if len(diff_kappa) > 0:
                                    mae_kappa[f"N_{N}_K_{K}_D_{D}"].append(np.mean(diff_kappa))
                                if len(diff_alpha) > 0:
                                    mae_alpha[f"N_{N}_K_{K}_D_{D}"].append(np.mean(diff_alpha))
                                
                                # Flatten the arrays
                                # 1: beta, 0: Pareto, 2: Dirac
                                accuracy = accuracy_score(true_dist_match, pred_dist_match)
                                accuracy_gmm[f"N_{N}_K_{K}_D_{D}"].append(accuracy)
                                
                                idx_beta = np.where(true_dist_match == 1)[0]
                                if len(idx_beta) > 0:
                                    acc_beta = accuracy_score(true_dist_match[idx_beta], pred_dist_match[idx_beta])
                                    accuracy_beta[f"N_{N}_K_{K}_D_{D}"].append(acc_beta)
                                
                                idx_pareto = np.where(true_dist_match == 0)[0]
                                if len(idx_pareto) > 0:
                                    acc_pareto = accuracy_score(true_dist_match[idx_pareto], pred_dist_match[idx_pareto])
                                    accuracy_pareto[f"N_{N}_K_{K}_D_{D}"].append(acc_pareto)
                                
                                idx_dirac = np.where(true_dist_match == 2)[0]
                                if len(idx_dirac) > 0:
                                    acc_dirac = accuracy_score(true_dist_match[idx_dirac], pred_dist_match[idx_dirac])
                                    accuracy_dirac[f"N_{N}_K_{K}_D_{D}"].append(acc_dirac)
                                """
                                COMPUTE VALUES PER SINGLE MUTATION
                                """
                                true_dist = true_dist.ravel()
                                pred_dist = pred_dist.ravel()
                                
                                accuracy_data_points = accuracy_score(true_dist, pred_dist)
                                # accuracy_data_points = np.mean(true_dist[:, 0] == pred_dist[:, 0])
                                acc_data_points_gmm[f"N_{N}_K_{K}_D_{D}"].append(accuracy_data_points)

                                idx_beta = np.where(true_dist == 1)[0]
                                if len(idx_beta) > 0:
                                    acc_data_points_beta = accuracy_score(true_dist[idx_beta], pred_dist[idx_beta])
                                    accuracy_data_points_beta[f"N_{N}_K_{K}_D_{D}"].append(acc_data_points_beta)
                                
                                idx_pareto = np.where(true_dist == 0)[0]
                                if len(idx_pareto) > 0:
                                    acc_data_points_pareto = accuracy_score(true_dist[idx_pareto], pred_dist[idx_pareto])
                                    accuracy_data_points_pareto[f"N_{N}_K_{K}_D_{D}"].append(acc_data_points_pareto)
                                
                                idx_dirac = np.where(true_dist == 2)[0]
                                if len(idx_dirac) > 0:
                                    acc_data_points_dirac = accuracy_score(true_dist[idx_dirac], pred_dist[idx_dirac])
                                    accuracy_data_points_dirac[f"N_{N}_K_{K}_D_{D}"].append(acc_data_points_dirac)
            variable_names = [
                "accuracy_data_points_beta", "accuracy_data_points_pareto", "accuracy_data_points_dirac", "acc_data_points_gmm",
                "accuracy_beta", "accuracy_pareto", "accuracy_dirac", "accuracy_gmm",
                "mae_phi", "mae_kappa", "mae_alpha", 'true_n_K', 'pred_n_K',
                "true_phi_dic", "true_kappa_dic", "true_alpha_dic", "pred_phi_dic", "pred_kappa_dic", "pred_alpha_dic"
            ]
            folder_path = general_folder_gmm + 'finals_new/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            # Save each variable in a separate file
            for name in variable_names:
                with open(general_folder_gmm + f"finals_new/{name}_w_15.pkl", "wb") as f:
                    pickle.dump(globals()[name], f)