from utils.BoundedPareto import BoundedPareto
import numpy as np
import pyro
import pyro.distributions as dist
import scipy.stats as stats

import torch

def euclidean_distance(a, b):
    return torch.dist(a, b)


def sample_mixing_prop(K, min_value=0.05):
    while True: # loop until valid sample
        sample = dist.Dirichlet(torch.ones(K)).sample()
        if (sample > min_value).all():
            return sample

def find_mixing_proportions(K, N):
    # Sample mixing proportions for clusters and multiply by N to obtain the number of data in each cluster
    pi = sample_mixing_prop(K, min_value=0.008) * N
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
    
    return pi

def pareto_binomial(N, alpha, L, H, depth):
    p = BoundedPareto(scale=L, alpha=alpha, upper_limit=H).sample((N,))
    bin = dist.Binomial(total_count=depth, probs=p).sample()
    min_bin = torch.ceil(L * depth)
    max_bin = torch.ceil(H * depth)
    # bin = torch.max(bin, min_bin)
    while torch.any(bin > max_bin):
        mask = bin > max_bin
        bin[mask] = dist.Binomial(total_count=depth[mask], probs=p[mask]).sample()
    while torch.any(bin < min_bin):
        mask = bin < min_bin
        bin[mask] = dist.Binomial(total_count=depth[mask], probs=p[mask]).sample()
        
    return bin

# Define the Beta-Binomial function
def beta_binomial(N, phi, kappa, depth, L):
    a = phi * kappa
    b = (1 - phi) * kappa
    p = dist.Beta(a, b).sample((N,))
    bin = dist.Binomial(total_count=depth, probs=p).sample()
    min_bin = torch.ceil(L * depth)
    while torch.any(bin < min_bin):
        mask = bin < min_bin
        bin[mask] = dist.Binomial(total_count=depth[mask], probs=p[mask]).sample()
    return bin

def sample_kappa():
    return dist.Uniform(150, 350).sample()

def sample_alpha():
    return dist.Uniform(0.8, 1.5).sample()  # Pareto shape parameter

def sample_phi(min_phi, max_vaf):
    return dist.Uniform(min_phi, max_vaf).sample()

def set_pareto_parameters(k, d, alpha, init_idx, end_idx, type_labels_data, type_labels_cluster, phi_param_data, kappa_param_data, 
                        alpha_param_data, phi_param_cluster, kappa_param_cluster, alpha_param_cluster):
    
    type_labels_data[init_idx:end_idx, d] = 0
    type_labels_cluster[k, d] = 0

    phi_param_data[init_idx:end_idx, d] = -1
    kappa_param_data[init_idx:end_idx, d] = -1
    alpha_param_data[init_idx:end_idx, d] = round(alpha.item(), 3)

    phi_param_cluster[k, d] = -1
    kappa_param_cluster[k, d] = -1
    alpha_param_cluster[k, d] = round(alpha.item(), 3)


def set_beta_parameters(k, d, phi, kappa, init_idx, end_idx, type_labels_data, type_labels_cluster, phi_param_data, 
                        kappa_param_data, alpha_param_data, phi_param_cluster, kappa_param_cluster, alpha_param_cluster):
    
    type_labels_data[init_idx:end_idx, d] = 1 # distribution type 1 for each mutation
    type_labels_cluster[k, d] = 1 # distribution type 1 for each cluster

    phi_param_data[init_idx:end_idx, d] = round(phi.item(), 3)  # phi value for each mutation
    kappa_param_data[init_idx:end_idx, d] = round(kappa.item(), 3) # kappa value for each mutation
    alpha_param_data[init_idx:end_idx, d] = -1 # -1 value for each mutation

    phi_param_cluster[k, d] = round(phi.item(), 3)
    kappa_param_cluster[k, d] = round(kappa.item(), 3)
    alpha_param_cluster[k, d] = -1

def set_zero_parameters(k, d, phi, init_idx, end_idx, type_labels_data, type_labels_cluster, NV, phi_param_data, kappa_param_data, alpha_param_data,
                        phi_param_cluster, kappa_param_cluster, alpha_param_cluster):
    
    type_labels_cluster[k, d] = 2  # distribution type 2 for each mutation
    type_labels_data[init_idx:end_idx, d] = 2  # distribution type 2 for each cluster

    NV[init_idx:end_idx, d] = phi

    phi_param_data[init_idx:end_idx, d] = -1
    kappa_param_data[init_idx:end_idx, d] = -1
    alpha_param_data[init_idx:end_idx, d] = -1

    phi_param_cluster[k, d] = -1
    kappa_param_cluster[k, d] = -1
    alpha_param_cluster[k, d] = -1



def generate_data_new_model_final(N, K, D, purity, coverage, seed):

    pi = find_mixing_proportions(K, N)

    NV = torch.zeros((N, D))
    threshold=0.1
    cluster_labels = torch.zeros(N)  # list to save the true cluster for each mutation
    type_labels_data = torch.zeros((N, D))  # list to save the true distribution type for each dimension of each mutation
    type_labels_cluster = torch.zeros((K, D))  # list to save the true distribution type for each dimension of each cluster
    
    phi_param_data = torch.zeros((N, D)) # list to save the true phi param for each mutation (-1 if the distribution is not beta)
    kappa_param_data = torch.zeros((N, D)) # list to save the true kappa param for each mutation (-1 if the distribution is not beta)
    alpha_param_data = torch.zeros((N, D)) # list to save the true alpha param for each mutation (-1 if the distribution is not pareto)
    
    phi_param_cluster = torch.zeros((K, D))  # list to save the true phi param for each cluster (-1 if the distribution is not beta)
    kappa_param_cluster = torch.zeros((K, D)) # list to save the true kappa param for each cluster (-1 if the distribution is not beta)
    alpha_param_cluster = torch.zeros((K, D)) # list to save the true alpha param for each cluster (-1 if the distribution is not pareto)
    
    max_vaf = purity[0]/2
    min_phi = 0.08
    probs_pareto = 0.04
    pareto_L = torch.tensor(0.03)  # Scale Pareto
    pareto_H = torch.tensor(max_vaf)  # Upper bound Pareto
    depth = dist.Poisson(coverage).sample([N,D])

    sampled_phi_list = []

    # Always have a Beta-Binomial component with phi=max_vaf in all dimensions
    k = 0
    init_idx = 0
    end_idx = pi[k]
    for d in range(D):
        phi = torch.tensor(max_vaf)
        kappa = sample_kappa()
        NV[init_idx:end_idx, d] = beta_binomial(pi[k], phi, kappa, depth[:pi[k],d], pareto_L)

        set_beta_parameters(k, d, phi, kappa, init_idx, end_idx, type_labels_data, type_labels_cluster, phi_param_data, kappa_param_data, alpha_param_data, phi_param_cluster, kappa_param_cluster, alpha_param_cluster)

    cluster_labels[:pi[k]] = k  # cluster k
    sampled_phi_list.append(torch.tensor([phi] * D))

    # Always have a Pareto-Binomial component in all dimensions
    k = 1
    init_idx = np.sum(pi[:k])
    end_idx = init_idx + pi[k]
    for d in range(D):
        alpha = sample_alpha()
        NV[init_idx:end_idx, d] = pareto_binomial(pi[k], alpha, pareto_L, pareto_H, depth[init_idx:end_idx, d])

        set_pareto_parameters(k, d, alpha, init_idx, end_idx, type_labels_data, type_labels_cluster, phi_param_data, kappa_param_data, alpha_param_data, phi_param_cluster, kappa_param_cluster, alpha_param_cluster)
    
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
            pyro.set_rng_seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            curr_sampled_phi = []
            for d in range(D):
                choose_dist = torch.randint(1, 4, (1,)).item() # randomly sample a value between 1, 2 or 3
                if choose_dist == 1:
                    phi, kappa = sample_phi(min_phi, max_vaf), sample_kappa()
                    NV[init_idx:end_idx, d] = beta_binomial(pi[k], phi, kappa, depth[init_idx:end_idx, d],pareto_L)

                    set_beta_parameters(k, d, phi, kappa, init_idx, end_idx, type_labels_data, type_labels_cluster, phi_param_data, kappa_param_data, alpha_param_data, phi_param_cluster, kappa_param_cluster, alpha_param_cluster)

                    curr_sampled_phi.append(phi)

                elif choose_dist == 2: # Pareto-Binomial for this dimension
                    if pareto_count >= (D-1): 
                        # if the number of pareto dimensions are already D-1 (all but 1), then sample either a beta or zeros
                        if torch.rand(1).item() < 0.5 and zeros_count < (D-1): # zeros
                            phi = 0

                            set_zero_parameters(k, d, phi, init_idx, end_idx, type_labels_data, type_labels_cluster, NV, phi_param_data, kappa_param_data, alpha_param_data, phi_param_cluster, kappa_param_cluster, alpha_param_cluster)
                            
                            zeros_count += 1
                            curr_sampled_phi.append(phi)                            
                        else: # beta
                            phi, kappa = sample_phi(min_phi, max_vaf), sample_kappa()
                            NV[init_idx:end_idx, d] = beta_binomial(pi[k], phi, kappa, depth[init_idx:end_idx, d],pareto_L)

                            set_beta_parameters(k, d, phi, kappa, init_idx, end_idx, type_labels_data, type_labels_cluster, phi_param_data, kappa_param_data, alpha_param_data, phi_param_cluster, kappa_param_cluster, alpha_param_cluster)

                            curr_sampled_phi.append(phi)
                            
                    else: # pareto
                        alpha = sample_alpha()
                        NV[init_idx:end_idx, d] = pareto_binomial(pi[k], alpha, pareto_L, pareto_H, depth[init_idx:end_idx, d])

                        set_pareto_parameters(k, d, alpha, init_idx, end_idx, type_labels_data, type_labels_cluster, phi_param_data, kappa_param_data, alpha_param_data, phi_param_cluster, kappa_param_cluster, alpha_param_cluster)

                        pareto_count += 1
                        curr_sampled_phi.append(probs_pareto)

                elif choose_dist == 3: # Zeros for this dimension
                    if zeros_count >= (D-1): 
                        # if the number of zeros dimensions are already D-1 (all but 1), then sample either a beta or a pareto
                        if torch.rand(1).item() < 0.5 and pareto_count < (D-1):  # zeros
                            alpha = sample_alpha()
                            NV[init_idx:end_idx, d] = pareto_binomial(pi[k], alpha, pareto_L, pareto_H, depth[init_idx:end_idx, d])

                            set_pareto_parameters(k, d, alpha, init_idx, end_idx, type_labels_data, type_labels_cluster, phi_param_data, kappa_param_data, alpha_param_data, phi_param_cluster, kappa_param_cluster, alpha_param_cluster)
                            
                            pareto_count += 1
                            curr_sampled_phi.append(probs_pareto)
                            
                        else: # beta
                            phi, kappa = sample_phi(min_phi, max_vaf), sample_kappa()
                            NV[init_idx:end_idx, d] = beta_binomial(pi[k], phi, kappa, depth[init_idx:end_idx, d],pareto_L)

                            set_beta_parameters(k, d, phi, kappa, init_idx, end_idx, type_labels_data, type_labels_cluster, phi_param_data, kappa_param_data, alpha_param_data, phi_param_cluster, kappa_param_cluster, alpha_param_cluster)

                            curr_sampled_phi.append(phi)
                    else:
                        phi = 0

                        set_zero_parameters(k, d, phi, init_idx, end_idx, type_labels_data, type_labels_cluster, NV, phi_param_data, kappa_param_data, alpha_param_data, phi_param_cluster, kappa_param_cluster, alpha_param_cluster)
                        zeros_count += 1
                        curr_sampled_phi.append(pareto_L - threshold)
            
            # Convert curr_sampled_phi to a tensor
            curr_sampled_phi_tensor = torch.tensor(curr_sampled_phi)
            
            # Check if curr_sampled_phi list has a euclidean distance < threshold from all the already present element in sampled_phi_list:
            # if yes, add it to sampled_phi_list and go to the next iteration of k, otherwise repeat this loop over d
            
            # Check if the Euclidean distance is below the threshold for any sampled_phi in sampled_phi_list
            if all(euclidean_distance(curr_sampled_phi_tensor, phi) >= threshold for phi in sampled_phi_list):
                # If no element in sampled_phi_list is too close, add to sampled_phi_list and break the loop
                sampled_phi_list.append(curr_sampled_phi_tensor)
                break  # Move to the next cluster
            else:
                seed+=1
    return NV, depth, cluster_labels, type_labels_data, type_labels_cluster, phi_param_data, kappa_param_data, alpha_param_data, phi_param_cluster, kappa_param_cluster, alpha_param_cluster


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
    
    min_bin = torch.tensor(np.ceil(L * n))
    d1[:, 0] = torch.max(d1[:, 0], min_bin)

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
