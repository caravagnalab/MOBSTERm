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
import seaborn as sns

import copy
import json

import argparse

import sys
import os
import ast
import time
# from utils_functions_mobsterm import *

import model_mobster_parallel as model_mobster_mv
from plot_functions_parallel import *

# Set the parent directory
# parent_dir = "../../"
# sys.path.insert(0, parent_dir)

from BoundedPareto import BoundedPareto
# from utils.create_beta_pareto_dataset import *


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run with parameters --N --K --D --p")
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--D", type=int, required=True)
    parser.add_argument("--p", type=float, default=1.) # purity
    parser.add_argument("--cov", type=int, default=100) # coverage

    num_iter = 2000
    num_dataset = 15

    # num_iter = 5
    # num_dataset = 2

    args = parser.parse_args()
    N = args.N  # number of mutations
    K = args.K  # number of clusters
    D = args.D  # number of samples
    purity = [args.p]*D
    coverage = args.cov

    # create_folder(N,K,D,purity[0],coverage)
    # dir = "~/scratch/tesimagistrale/subclonal_deconvolution_mv/test_generative/mobsterm_and_mobster/parallel_model"
    # sys.path.insert(0, dir)

    time_list = []
    
    seed = 0
    
    for idx in range(num_dataset):
        seed1 = seed+idx+K+N
        pyro.set_rng_seed(seed1)
        torch.manual_seed(seed1)
        np.random.seed(seed1)
        
        # NV, DP, cluster_labels, type_labels_data, type_labels_cluster, phi_param_data, kappa_param_data, alpha_param_data, phi_param_cluster, kappa_param_cluster, alpha_param_cluster  = generate_data_new_model_final(N, K, D, purity, coverage, seed1)

        # plot_scatter_real(NV, DP, N, K, D, type_labels_cluster, cluster_labels, idx, purity[0], coverage)  
        # plot_marginals_real(NV, DP, N, K, D, type_labels_cluster, cluster_labels, phi_param_cluster, kappa_param_cluster, alpha_param_cluster, idx, purity[0], coverage)
        # plot_marginals_all_real(NV, DP, N, K, D, type_labels_cluster, cluster_labels, phi_param_cluster, kappa_param_cluster, alpha_param_cluster, idx, purity[0], coverage)
        
        # --- Load CSV ---
        p_str = str(purity[0]).replace(".", "")
        print('PURITY: ', p_str)
        # if (purity == 1):
        f = f"~/scratch/tesimagistrale/subclonal_deconvolution_mv/test_generative/mobsterm_and_mobster/results/p_{p_str}_cov_{coverage}/D_{D}/csv/N_{N}_K_{K}_D_{D}_df_{idx}.csv"
        # else:
        #     f = f"~/scratch/tesimagistrale/subclonal_deconvolution_mv/test_generative/mobsterm_and_mobster/results/p_{p_str}_cov_{coverage}/D_{D}/csv/N_{N}_K_{K}_D_{D}_df_{idx}.csv"
        
        df = pd.read_csv(f)

        def parse_column(series):
            """
            Parse a column that may contain:
            - interval strings like "[27.0, 37.0]"
            - scalar floats/ints like 0.0 or 0
            Returns a 2D tensor (N, 2) for intervals, or 1D tensor (N,) for scalars.
            """
            first_val = series.iloc[0]

            # Check if the values are interval strings
            if isinstance(first_val, str) and first_val.startswith("["):
                parsed = series.apply(lambda x: ast.literal_eval(x))
                return torch.tensor(parsed.tolist(), dtype=torch.float32)
            else:
                return torch.tensor(series.tolist(), dtype=torch.float32)

        tensors = {}
        for col in df.columns:
            tensors[col] = parse_column(df[col])
            print(f"{col}: shape={tensors[col].shape}, dtype={tensors[col].dtype}")

        NV = tensors["NV"]                           # shape (N, 2)
        DP = tensors["DP"]                           # shape (N, 2)

        # Run the model
        if K != 3:
            K_list = [K - 2, K - 1, K, K + 1, K + 2, K + 3]
        else:
            K_list = [K - 1, K, K + 1, K + 2, K + 3]
        
        # seed_list = [40,41]
        seed_list = [40,41,42]
        
        start_time = time.time()

        mb = model_mobster_mv.fit(
            NV=NV, DP=DP, mut_id=None,
            num_iter=num_iter, K=K_list,
            seed_list=seed_list, lr=0.01,
            purity=purity,
            n_jobs=-1  # to use all available CPU
        )

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Time taken for the fit function: {elapsed_time:.2f} seconds")

        time_list.append(elapsed_time)

        save_folder = f"p_{p_str}_cov_{coverage}/D_{D}/K_{K}/N_{N}"

        save_results(mb, idx, save_folder = save_folder)
        save_results(mb['best_fit'], idx, save_folder = f"{save_folder}/best")
        # plot_bic_icl(mb, idx, save_folder = save_folder)

        plot_scatter_inference(mb['best_fit'], idx, save_folder= f"{save_folder}/best")
        plot_marginals_inference(mb['best_fit'], idx, save_folder= f"{save_folder}/best")
        # plot_marginals_single(mb['best_fit'], idx, save_folder= f"{save_folder}/best")
        # plot_deltas(mb['best_fit'], idx, save_folder= f"{save_folder}/best")
        # plot_responsib(mb['best_fit'], idx, save_folder= f"{save_folder}/best")
        # plot_mixing_proportions(mb['best_fit'], idx, save_folder= f"{save_folder}/best")
        # plot_loss_lks_dist(mb['best_fit'], idx, save_folder= f"{save_folder}/best")

        # save_folder = f'p_{p_str}_cov_{coverage}/D_{D}/K_{K}/N_{N}'
        
        # save_results(mb, save_folder = save_folder)
   
    
    # time_folder = '../results/times_csv/mobsterm'
    
    # filename = f"{time_folder}/p_{str(purity[0]).replace('.', '')}_cov_{coverage}/D_{D}/N_{N}_K_{K}_D_{D}.csv"
    # if not os.path.exists(f"{time_folder}/p_{str(purity[0]).replace('.', '')}_cov_{coverage}/D_{D}/"):
    #         os.makedirs(f"{time_folder}/p_{str(purity[0]).replace('.', '')}_cov_{coverage}/D_{D}/")
    
    # with open(filename, "w") as file:
    #     for item in time_list:
    #         file.write(f"{item}\n")  # Writing each item on a new line   