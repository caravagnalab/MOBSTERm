import model_mobster_mv_orfeo_generative as mobster_mv 
import numpy as np
import pandas as pd
import pyro.distributions as dist
from scipy import stats
import os
import torch
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns

from utils.plot_functions import *
from utils.BoundedPareto import BoundedPareto
from utils.create_beta_pareto_dataset import *


seed = 123
pyro.set_rng_seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# Pareto-Binomial
def pareto_binomial(N, alpha, L, H, depth):
    p = BoundedPareto(scale=L, alpha=alpha, upper_limit=H).sample((N,))
    return dist.Binomial(total_count=depth, probs=p).sample()

# Beta-Binomial
def beta_binomial(N, phi, kappa, depth):
    a = phi*kappa
    b = (1-phi)*kappa
    p = dist.Beta(a, b).sample((N,))
    return dist.Binomial(total_count=depth, probs=p).sample()


def generate_data(N, K, pi):
    NV = []
    labels = []
    type_labels = []
    param_list = []
    # depth = torch.randint(80, 150, (N,))  # Random depth
    depth = torch.tensor(120).repeat((N,))

    for k in range(K):
        # Pareto-Binomial for one cluster in each dimension
        if k == 0:
            alpha = torch.tensor(1.0)  # Pareto shape parameter
            pareto_L = torch.tensor(0.01)  # scale Pareto
            pareto_H = torch.tensor(0.5)  # Upper bound Pareto
            NV.append(pareto_binomial(pi[k], alpha, pareto_L, pareto_H, depth[:pi[k]]))
            labels.extend([k] * pi[k])
            type_labels.extend(['P'] * pi[k])
            param_list.extend([1.] * pi[k])
        elif k == 1:
            # Beta-Binomial in 0.5
            kappa = 200.
            phi = 0.5
            NV.append(beta_binomial(pi[k], phi, kappa, depth[pi[k-1]:(pi[k-1]+pi[k])]))
            labels.extend([k] * pi[k])
            type_labels.extend(['B'] * pi[k])
            param_list.extend([phi] * pi[k])
        else:
            # Beta-Binomial in [0.15, 0.45]
            kappa = 200.
            phi_L, phi_H = 0.15, 0.4
            phi = dist.Uniform(phi_L, phi_H).sample()
            NV.append(beta_binomial(pi[k], phi, kappa, depth[pi[k-1]:(pi[k-1]+pi[k])]))
            labels.extend([k] * pi[k])
            type_labels.extend(['B'] * pi[k])
            param_list.extend([round(phi.item(), 3)] * pi[k])
    
    NV = torch.cat(NV)
    labels = torch.tensor(labels)
    return NV, depth, labels, type_labels, param_list

N_list = [300,1000,5000]  # number of samples
K1_list = [2,3,4]    # number of clusters in dim 1
K2_list = [2,3,4]  # number of clusters in dim 2

folder_path = "results"
# Create the directory if it does not exist
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print(f"Folder '{folder_path}' created!")
else:
    print(f"Folder '{folder_path}' already exists.")

folder_path = "results/nmi"
# Create the directory if it does not exist
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print(f"Folder '{folder_path}' created!")
else:
    print(f"Folder '{folder_path}' already exists.")

folder_path = "results/ari"
# Create the directory if it does not exist
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print(f"Folder '{folder_path}' created!")
else:
    print(f"Folder '{folder_path}' already exists.")

folder_path = "results/acc"
# Create the directory if it does not exist
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print(f"Folder '{folder_path}' created!")
else:
    print(f"Folder '{folder_path}' already exists.")

folder_path = "results/conf_matrix"
# Create the directory if it does not exist
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print(f"Folder '{folder_path}' created!")
else:
    print(f"Folder '{folder_path}' already exists.")

for n in range(len(N_list)):
    N = N_list[n]
    for k in range(len(K1_list)):
        K1 = K1_list[k]
        K2 = K2_list[k]
        nmi_list = []
        ari_list = []
        acc_list = []
        conf_matrix_list = []
        for i in range(20):
            pi_x = dist.Dirichlet(torch.ones(K1)).sample() * N # number of data in each cluster for dim 1
            pi_x = np.round(pi_x.numpy()).astype('int')
            if(np.sum(pi_x) < N):
                pi_x[-1] = pi_x[-1]+1
            if(np.sum(pi_x) > N):
                pi_x[-1] = pi_x[-1]-1


            pi_y = dist.Dirichlet(torch.ones(K2)).sample() * N # number of data in each cluster for dim 2
            pi_y = np.round(pi_y.numpy()).astype('int')
            if(np.sum(pi_y) < N):
                pi_y[-1] = pi_y[-1]+1
            if(np.sum(pi_y) > N):
                pi_y[-1] = pi_y[-1]-1

            NV_x, DP_x, labels_x, type_labels_x, param_list_x = generate_data(N[n], K1, pi_x)
            NV_y, DP_y, labels_y, type_labels_y, param_list_y = generate_data(N[n], K2, pi_y)

            NV = torch.stack((NV_x, NV_y), dim = 0).T
            # print(NV.shape)

            DP = torch.stack((DP_x, DP_y), dim = 0).T
            
            # create labels for the found 2d clusters
            combined_labels = torch.stack((labels_x, labels_y), dim=1)
            unique_combinations, cluster_ids = torch.unique(combined_labels, return_inverse=True, dim=0)

            cluster_counts = torch.bincount(cluster_ids)
            true_K = len(cluster_counts)
            # Find true distribution (Pareto P or Beta B) for the clusters
            unique_types = []
            for combo in unique_combinations:
                x_index = (labels_x == combo[0]).nonzero(as_tuple=True)[0][0].item()  # Find index in labels_x
                y_index = (labels_y == combo[1]).nonzero(as_tuple=True)[0][0].item()  # Find index in labels_y
                unique_types.append([type_labels_x[x_index], type_labels_y[y_index]])
            
            # Run the model
            K_list = [true_K - 1, true_K, true_K + 1]
            seed_list = [40,41,42]
            mb, best_K, best_seed = mobster_mv.fit(NV, DP, num_iter = 2000, K = K_list, seed = seed_list, lr = 0.005)

            # Measure NMI
            true_labels = cluster_ids
            predicted_labels = mb.params["cluster_assignments"]

            nmi = normalized_mutual_info_score(true_labels, predicted_labels)
            nmi_list.append(nmi)
            filename = f"results/nmi/nmi_N_{N}_sub_{K1}"
            with open(filename, "w") as file:
                for item in nmi_list:
                    file.write(f"{item}\n")  # Writing each item on a new line
            ari = adjusted_rand_score(true_labels, predicted_labels)
            ari_list.append(ari)
            filename = f"results/ari/ari_N_{N}_sub_{K1}"
            with open(filename, "w") as file:
                for item in ari_list:
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

    # print(DP.shape)
# plt.scatter(NV[:,0].numpy()/DP[:,0].numpy(), NV[:,1].numpy()/DP[:,1].numpy())