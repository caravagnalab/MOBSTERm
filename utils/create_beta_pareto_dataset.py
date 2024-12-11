from utils.BoundedPareto import BoundedPareto
import numpy as np
import pyro
import pyro.distributions as dist
import scipy.stats as stats

import torch

# ---------------#

def euclidean_distance(a, b):
    return torch.dist(a, b)

def pareto_binomial(N, alpha, L, H, depth):
    p = BoundedPareto(scale=L, alpha=alpha, upper_limit=H).sample((N,))
    bin = dist.Binomial(total_count=depth, probs=p).sample()
    min_bin = torch.ceil(L * depth)
    bin = torch.max(bin, min_bin)
    return bin

def beta_binomial(N, phi, kappa, depth):
    a = phi * kappa
    b = (1 - phi) * kappa
    p = dist.Beta(a, b).sample((N,))
    return dist.Binomial(total_count=depth, probs=p).sample()


def generate_data_new_model(N, K, pi, D, purity, coverage):
    NV = torch.zeros((N, D))
    threshold=0.12
    cluster_labels = torch.zeros(N)  # one-dimensional labels, one per data
    type_labels_data = torch.zeros((N, D))  # D-dimensional labels, one per data
    type_labels_cluster = torch.zeros((K, D))  # D-dimensional label, one per cluster
    phi_param_data = torch.zeros((N, D))
    kappa_param_data = torch.zeros((N, D))
    alpha_param_data = torch.zeros((N, D))
    phi_param_cluster = torch.zeros((K, D))
    kappa_param_cluster = torch.zeros((K, D))
    alpha_param_cluster = torch.zeros((K, D))
    max_vaf = purity/2
    probs_pareto = 0.08

    # variance_negbin = 1000
    # mean_negbin = coverage
    # probs_negbin = mean_negbin/variance_negbin
    # n_negbin = mean_negbin**2/(variance_negbin-mean_negbin)
    # neg_bin = stats.nbinom.rvs(n_negbin, probs_negbin, size=N*D)
    # depth = torch.tensor(neg_bin).reshape([N,D])
    depth = dist.Poisson(coverage).sample([N,D])

    sampled_phi_list = []

    # Always have a Beta-Binomial component with phi=max_vaf in all dimensions
    k = 0
    for d in range(D):
        p = max_vaf
        kappa = dist.Uniform(90, 350).sample()
        NV[:pi[k], d] = beta_binomial(pi[k], p, kappa, depth[:pi[k],d])
        type_labels_data[:pi[k], d] = torch.tensor(1)  # beta
        type_labels_cluster[k, d] = torch.tensor(1)  # beta
        phi_param_data[:pi[k], d] = p
        kappa_param_data[:pi[k], d] = round(kappa.item(), 3)
        alpha_param_data[:pi[k], d] = -1
        phi_param_cluster[k, d] = p
        kappa_param_cluster[k, d] = round(kappa.item(), 3)
        alpha_param_cluster[k, d] = -1
    cluster_labels[:pi[k]] = k  # cluster k
    sampled_phi_list.append(torch.tensor([p] * D))

    # Always have a Pareto-Binomial component in all dimensions
    k = 1
    pareto_L = torch.tensor(0.05)  # Scale Pareto
    pareto_H = torch.tensor(max_vaf)  # Upper bound Pareto
    init_idx = np.sum(pi[:k])
    end_idx = init_idx + pi[k]
    for d in range(D):
        alpha = dist.Uniform(0.8, 1.5).sample()  # Pareto shape parameter
        NV[init_idx:end_idx, d] = pareto_binomial(pi[k], alpha, pareto_L, pareto_H, depth[init_idx:end_idx, d])
        type_labels_data[init_idx:end_idx, d] = torch.tensor(0)  # pareto
        type_labels_cluster[k, d] = torch.tensor(0)  # pareto
        phi_param_data[init_idx:end_idx, d] = -1
        kappa_param_data[init_idx:end_idx, d] = -1
        alpha_param_data[init_idx:end_idx, d] = round(alpha.item(), 3)
        phi_param_cluster[k, d] = -1
        kappa_param_cluster[k, d] = -1
        alpha_param_cluster[k, d] = round(alpha.item(), 3)
    cluster_labels[init_idx:end_idx] = k  # cluster k
    sampled_phi_list.append(torch.tensor([probs_pareto] * D))

    # Randomly sample from Beta-Binomial, Pareto-Binomial or Zeros for additional components
    for k in range(2, K):
        init_idx = np.sum(pi[:k])
        end_idx = init_idx + pi[k]
        pareto_count = 0
        zeros_count = 0
        cluster_labels[init_idx:end_idx] = k  # cluster k
        while True:
            curr_sampled_phi = []
            for d in range(D):
                choose_dist = torch.randint(1, 4, (1,)).item() # randomly sample a value between 1, 2 or 3
                if choose_dist == 1:
                    phi, kappa = dist.Uniform(0.15, 0.5).sample(), dist.Uniform(90, 350).sample()
                    NV[init_idx:end_idx, d] = beta_binomial(pi[k], phi, kappa, depth[init_idx:end_idx, d])
                    type_labels_data[init_idx:end_idx, d] = torch.tensor(1)  # beta
                    type_labels_cluster[k, d] = torch.tensor(1)  # beta
                    phi_param_data[init_idx:end_idx, d] = round(phi.item(), 3)
                    kappa_param_data[init_idx:end_idx, d] = round(kappa.item(), 3)
                    alpha_param_data[init_idx:end_idx, d] = -1
                    phi_param_cluster[k, d] = round(phi.item(), 3)
                    kappa_param_cluster[k, d] = round(kappa.item(), 3)
                    alpha_param_cluster[k, d] = -1
                    curr_sampled_phi.append(phi)
                elif choose_dist == 2: # Pareto-Binomial for this dimension
                    if pareto_count >= (D-1): 
                        # if the number of pareto dimensions are already D-1 (all but 1), then sample either a beta or zeros
                        if torch.rand(1).item() < 0.5 and zeros_count < (D-1): # zeros
                            phi = 0
                            type_labels_cluster[k, d] = torch.tensor(2)  # zeros
                            NV[init_idx:end_idx, d] = phi
                            phi_param_data[init_idx:end_idx, d] = -1
                            kappa_param_data[init_idx:end_idx, d] = -1
                            alpha_param_data[init_idx:end_idx, d] = -1
                            phi_param_cluster[k, d] = -1
                            kappa_param_cluster[k, d] = -1
                            alpha_param_cluster[k, d] = -1
                            zeros_count += 1
                            curr_sampled_phi.append(phi)                            
                        else: # beta
                            phi, kappa = dist.Uniform(0.15, 0.5).sample(), dist.Uniform(90, 350).sample()
                            NV[init_idx:end_idx, d] = beta_binomial(pi[k], phi, kappa, depth[init_idx:end_idx, d])
                            type_labels_data[init_idx:end_idx, d] = torch.tensor(1)  # beta
                            type_labels_cluster[k, d] = torch.tensor(1)  # beta
                            phi_param_data[init_idx:end_idx, d] = round(phi.item(), 3)
                            kappa_param_data[init_idx:end_idx, d] = round(kappa.item(), 3)
                            alpha_param_data[init_idx:end_idx, d] = -1
                            phi_param_cluster[k, d] = round(phi.item(), 3)
                            kappa_param_cluster[k, d] = round(kappa.item(), 3)
                            alpha_param_cluster[k, d] = -1                            
                            curr_sampled_phi.append(phi)
                            
                    else: # pareto
                        alpha = dist.Uniform(0.8, 1.5).sample()
                        NV[init_idx:end_idx, d] = pareto_binomial(pi[k], alpha, pareto_L, pareto_H, depth[init_idx:end_idx, d])
                        type_labels_data[init_idx:end_idx, d] = torch.tensor(0)  # pareto
                        type_labels_cluster[k, d] = torch.tensor(0)  # pareto
                        phi_param_data[init_idx:end_idx, d] = -1
                        kappa_param_data[init_idx:end_idx, d] = -1
                        alpha_param_data[init_idx:end_idx, d] = round(alpha.item(), 3)
                        phi_param_cluster[k, d] = -1
                        kappa_param_cluster[k, d] = -1
                        alpha_param_cluster[k, d] = round(alpha.item(), 3)
                        pareto_count += 1
                        curr_sampled_phi.append(probs_pareto)
                elif choose_dist == 3: # Zeros for this dimension
                    if zeros_count >= (D-1): 
                        # if the number of zeros dimensions are already D-1 (all but 1), then sample either a beta or a pareto
                        if torch.rand(1).item() < 0.5 and pareto_count < (D-1):  # zeros
                            alpha = dist.Uniform(0.8, 1.5).sample()
                            NV[init_idx:end_idx, d] = pareto_binomial(pi[k], alpha, pareto_L, pareto_H, depth[init_idx:end_idx, d])
                            type_labels_data[init_idx:end_idx, d] = torch.tensor(0)  # pareto
                            type_labels_cluster[k, d] = torch.tensor(0)  # pareto
                            phi_param_data[init_idx:end_idx, d] = -1
                            kappa_param_data[init_idx:end_idx, d] = -1
                            alpha_param_data[init_idx:end_idx, d] = round(alpha.item(), 3)
                            phi_param_cluster[k, d] = -1
                            kappa_param_cluster[k, d] = -1
                            alpha_param_cluster[k, d] = round(alpha.item(), 3)
                            pareto_count += 1
                            curr_sampled_phi.append(probs_pareto)
                        else: # beta
                            phi, kappa = dist.Uniform(0.15, 0.5).sample(), dist.Uniform(90, 350).sample()
                            NV[init_idx:end_idx, d] = beta_binomial(pi[k], phi, kappa, depth[init_idx:end_idx, d])
                            type_labels_data[init_idx:end_idx, d] = torch.tensor(1)  # beta
                            type_labels_cluster[k, d] = torch.tensor(1)  # beta
                            phi_param_data[init_idx:end_idx, d] = round(phi.item(), 3)
                            kappa_param_data[init_idx:end_idx, d] = round(kappa.item(), 3)
                            alpha_param_data[init_idx:end_idx, d] = -1
                            phi_param_cluster[k, d] = round(phi.item(), 3)
                            kappa_param_cluster[k, d] = round(kappa.item(), 3)
                            alpha_param_cluster[k, d] = -1
                            curr_sampled_phi.append(phi)
                    else:
                        phi = 0
                        type_labels_cluster[k, d] = torch.tensor(2)  # zeros
                        NV[init_idx:end_idx, d] = phi
                        phi_param_data[init_idx:end_idx, d] = -1
                        kappa_param_data[init_idx:end_idx, d] = -1
                        alpha_param_data[init_idx:end_idx, d] = -1
                        phi_param_cluster[k, d] = -1
                        kappa_param_cluster[k, d] = -1
                        alpha_param_cluster[k, d] = -1
                        zeros_count += 1
                        curr_sampled_phi.append(phi)

            
            # Convert curr_sampled_phi to a tensor
            curr_sampled_phi_tensor = torch.tensor(curr_sampled_phi)
            
            # Check if curr_sampled_phi list has a euclidean distance < threshold from all the already present element in sampled_phi_list:
            # if yes, add it to sampled_phi_list and go to the next iteration of k, otherwise repeat this loop over d
            
            # Check if the Euclidean distance is below the threshold for any sampled_phi in sampled_phi_list
            if all(euclidean_distance(curr_sampled_phi_tensor, phi) >= threshold for phi in sampled_phi_list):
                # If no element in sampled_phi_list is too close, add to sampled_phi_list and break the loop
                sampled_phi_list.append(curr_sampled_phi_tensor)
                break  # Move to the next cluster
    return NV, depth, cluster_labels, type_labels_data, type_labels_cluster, phi_param_data, kappa_param_data, alpha_param_data, phi_param_cluster, kappa_param_cluster, alpha_param_cluster



def generate_data(N, K, pi, D):
    NV = torch.zeros((N, D))
    threshold=0.12
    cluster_labels = torch.zeros(N)  # one-dimensional labels, one per data
    type_labels_data = torch.zeros((N, D))  # D-dimensional labels, one per data
    type_labels_cluster = torch.zeros((K, D))  # D-dimensional label, one per cluster
    phi_param_data = torch.zeros((N, D))
    kappa_param_data = torch.zeros((N, D))
    alpha_param_data = torch.zeros((N, D))

    depth = torch.tensor(120).repeat((N,))  # Fixed depth
    sampled_phi_list = []

    # Always have a Beta-Binomial component with phi=0.5 in all dimensions
    k = 0
    for d in range(D):
        p = 0.5
        kappa = dist.Uniform(90, 350).sample()
        NV[:pi[k], d] = beta_binomial(pi[k], p, kappa, depth[:pi[k]])
        type_labels_data[:pi[k], d] = torch.tensor(1)  # beta
        type_labels_cluster[k, d] = torch.tensor(1)  # beta
        phi_param_data[:pi[k], d] = p
        kappa_param_data[:pi[k], d] = round(kappa.item(), 3)
        alpha_param_data[:pi[k], d] = float('nan')
    cluster_labels[:pi[k]] = k  # cluster k
    sampled_phi_list.append(torch.tensor([p] * D))

    k = 1
    pareto_L = torch.tensor(0.01)  # Scale Pareto
    pareto_H = torch.tensor(0.5)  # Upper bound Pareto
    init_idx = np.sum(pi[:k])
    end_idx = init_idx + pi[k]
    for d in range(D):
        alpha = dist.Uniform(0.8, 1.5).sample()  # Pareto shape parameter
        NV[init_idx:end_idx, d] = pareto_binomial(pi[k], alpha, pareto_L, pareto_H, depth[init_idx:end_idx])
        type_labels_data[init_idx:end_idx, d] = torch.tensor(0)  # pareto
        type_labels_cluster[k, d] = torch.tensor(0)  # pareto
        phi_param_data[init_idx:end_idx, d] = float('nan')
        kappa_param_data[init_idx:end_idx, d] = float('nan')
        alpha_param_data[init_idx:end_idx, d] = round(alpha.item(), 3)
    cluster_labels[init_idx:end_idx] = k  # cluster k
    sampled_phi_list.append(torch.tensor([0.03] * D))

    # Randomly sample from Beta-Binomial or Pareto-Binomial for additional components
    for k in range(2, K):
        init_idx = np.sum(pi[:k])
        end_idx = init_idx + pi[k]
        pareto_count = 0
        cluster_labels[init_idx:end_idx] = k  # cluster k
        while True:
            curr_sampled_phi = []
            for d in range(D):
                if torch.rand(1).item() < 0.5:  # 50% chance for Beta-Binomial
                    phi, kappa = dist.Uniform(0.15, 0.5).sample(), dist.Uniform(90, 350).sample()
                    NV[init_idx:end_idx, d] = beta_binomial(pi[k], phi, kappa, depth[init_idx:end_idx])
                    type_labels_data[init_idx:end_idx, d] = torch.tensor(1)  # beta
                    type_labels_cluster[k, d] = torch.tensor(1)  # beta
                    phi_param_data[init_idx:end_idx, d] = round(phi.item(), 3)
                    kappa_param_data[init_idx:end_idx, d] = round(kappa.item(), 3)
                    alpha_param_data[init_idx:end_idx, d] = float('nan')
                    curr_sampled_phi.append(phi)
                else:
                    if pareto_count >= 1:
                        phi, kappa = dist.Uniform(0.15, 0.5).sample(), dist.Uniform(90, 350).sample()
                        NV[init_idx:end_idx, d] = beta_binomial(pi[k], phi, kappa, depth[init_idx:end_idx])
                        type_labels_data[init_idx:end_idx, d] = torch.tensor(1)  # beta
                        type_labels_cluster[k, d] = torch.tensor(1)  # beta
                        phi_param_data[init_idx:end_idx, d] = round(phi.item(), 3)
                        kappa_param_data[init_idx:end_idx, d] = round(kappa.item(), 3)
                        alpha_param_data[init_idx:end_idx, d] = float('nan')
                        curr_sampled_phi.append(phi)
                    else:
                        alpha = dist.Uniform(0.8, 1.5).sample()
                        NV[init_idx:end_idx, d] = pareto_binomial(pi[k], alpha, pareto_L, pareto_H, depth[init_idx:end_idx])
                        type_labels_data[init_idx:end_idx, d] = torch.tensor(0)  # pareto
                        type_labels_cluster[k, d] = torch.tensor(0)  # pareto
                        phi_param_data[init_idx:end_idx, d] = float('nan')
                        kappa_param_data[init_idx:end_idx, d] = float('nan')
                        alpha_param_data[init_idx:end_idx, d] = round(alpha.item(), 3)
                        pareto_count += 1
                        curr_sampled_phi.append(0.03)
            
            # Convert curr_sampled_phi to a tensor
            curr_sampled_phi_tensor = torch.tensor(curr_sampled_phi)
            

            # Check if curr_sampled_phi list has a euclidean distance < threshold from all the already present element in sampled_phi_list:
            # if yes, add it to sampled_phi_list and go to the next iteration of k, otherwise repeat this loop over d
            
            # Check if the Euclidean distance is below the threshold for any sampled_phi in sampled_phi_list
            if all(euclidean_distance(curr_sampled_phi_tensor, phi) >= threshold for phi in sampled_phi_list):
                # If no element in sampled_phi_list is too close, add to sampled_phi_list and break the loop
                sampled_phi_list.append(curr_sampled_phi_tensor)
                break  # Move to the next cluster
            

    depth = torch.tensor(120).repeat((N, D))  # Fixed depth
    return NV, depth, cluster_labels, type_labels_data, type_labels_cluster, phi_param_data, kappa_param_data, alpha_param_data

# ---------------#
def pareto_binomial_component(alpha=2, L=0.05, H=0.5, phi_beta = 0.5, k_beta = 0.5, n=100, N=1000, exchanged = False, seed = 123):
    """
    Create pareto-binomial component. 
    Default:
        x-axis is a Pareto-Binomial
        y-axis is a Beta-Binomial
    If exchanged == True:
        x-axis is a Beta-Binomial
        y-axis is a Pareto-Binomial
    """
    pyro.set_rng_seed(seed)
    d1 = torch.ones([N, 2]) # component 1

    # x-axis component 1
    p_p = BoundedPareto(scale=L, alpha = alpha, upper_limit = H).sample([N]).float()
    d1[:, 0] = dist.Binomial(total_count=n, probs=p_p).sample()#.squeeze(-1)

    a = phi_beta*k_beta
    b = (1-phi_beta)*k_beta
    p_p = dist.Beta(a, b).sample([N]).float()
    d1[:, 1] = dist.Binomial(total_count=n, probs=p_p).sample()#.squeeze(-1)
    
    DP = torch.ones([N, 2]) * n
    if exchanged == True:
        indices = torch.tensor([1,0])
        d1 = d1[:, indices]

    return d1, DP


def beta_binomial_component(phi_beta_x = 0.5, k_beta_x = 0.5, phi_beta_y = 0.5, k_beta_y= 0.5, n=100, N=1000, seed=123):
    """
    Create Beta-Binomial component:
    x-axis is a Beta-Binomial
    y-axis is a Beta-Binomial
    """
    pyro.set_rng_seed(seed)
    d2 = torch.ones([N, 2])

    a_x = phi_beta_x*k_beta_x
    b_x = (1-phi_beta_x)*k_beta_x
    a_y = phi_beta_y*k_beta_y
    b_y = (1-phi_beta_y)*k_beta_y
    # for i in range(N):
    p_x = dist.Beta(a_x, b_x).sample([N]).float()
    d2[:, 0] = dist.Binomial(total_count=n, probs=p_x).sample().squeeze(-1)
    p_y = dist.Beta(a_y, b_y).sample([N]).float()
    d2[:, 1] = dist.Binomial(total_count=n, probs=p_y).sample().squeeze(-1)


    # x-axis component 2
    # p_x = dist.Beta(a_x, b_x).sample()
    # d2[:, 0] = dist.Binomial(total_count=n, probs=p_x).sample([N]).squeeze(-1)
    
    # # # y-axis component 2
    # p_y = dist.Beta(a_y, b_y).sample()
    # d2[:, 1] = dist.Binomial(total_count=n, probs=p_y).sample([N]).squeeze(-1)

    DP = torch.ones([N, 2]) * n

    return d2, DP
    
def only_pareto_binomial_component(alpha_x=2, L_x=0.05, H_x=0.5, alpha_y=2, L_y=0.05, H_y=0.5, n=100, N=1000, seed = 123):
    """
    Create pareto-pareto component. 
    Default:
        x-axis is a Pareto-Binomial
        y-axis is a Pareto-Binomial
    """
    pyro.set_rng_seed(seed)
    d1 = torch.ones([N, 2]) # component 1
    
    # x-axis component 1
    # for i in range(N):
    p_p = BoundedPareto(scale=L_x, alpha = alpha_x, upper_limit = H_x).sample([N]).float()
    d1[:, 0] = dist.Binomial(total_count=n, probs=p_p).sample().squeeze(-1)
    # p_p = BoundedPareto(scale=L, alpha = alpha, upper_limit = H).sample().float()
    # d1[:, 0] = dist.Binomial(total_count=n, probs=p_p).sample([N]).squeeze(-1)


    # for i in range(N):
    p_p = BoundedPareto(scale=L_y, alpha = alpha_y, upper_limit = H_y).sample(([N])).float()
    d1[:, 1] = dist.Binomial(total_count=n, probs=p_p).sample().squeeze(-1)

    DP = torch.ones([N, 2]) * n

    return d1, DP


def pareto_binomial_component2(alpha=2, L=0.05, H=0.5, p=0.5, n=100, N=1000, exchanged = False, seed = 123):
    """
    Create pareto-binomial component. 
    Default:
        x-axis is a Pareto-Binomial
        y-axis is a Beta-Binomial
    If exchanged == True:
        x-axis is a Beta-Binomial
        y-axis is a Pareto-Binomial
    """
    pyro.set_rng_seed(seed)
    d1 = torch.ones([N, 2]) # component 1
    
    # x-axis component 1
    # for i in range(N):
    p_p = BoundedPareto(scale=L, alpha = alpha, upper_limit = H).sample([N]).float()
    d1[:, 0] = dist.Binomial(total_count=n, probs=p_p).sample().squeeze(-1)
    # p_p = BoundedPareto(scale=L, alpha = alpha, upper_limit = H).sample().float()
    # d1[:, 0] = dist.Binomial(total_count=n, probs=p_p).sample([N]).squeeze(-1)

    d1[:, 1] = dist.Binomial(total_count=n, probs=p).sample([N]).squeeze(-1)
    DP = torch.ones([N, 2]) * n
    if exchanged == True:
        indices = torch.tensor([1,0])
        d1 = d1[:, indices]

    return d1, DP


def beta_binomial_component2(p_x = 0.5, p_y= 0.5, n=100, N=1000, seed=123):
    """
    Create Beta-Binomial component:
    x-axis is a Beta-Binomial
    y-axis is a Beta-Binomial
    """
    pyro.set_rng_seed(seed)
    d2 = torch.ones([N, 2])
    
    # x-axis component 2
    d2[:, 0] = dist.Binomial(total_count=n, probs=p_x).sample([N]).squeeze(-1)
    
    # y-axis component 2
    d2[:, 1] = dist.Binomial(total_count=n, probs=p_y).sample([N]).squeeze(-1)

    DP = torch.ones([N, 2]) * n

    return d2, DP
