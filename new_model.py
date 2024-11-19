import numpy as np
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import copy
import json

import torch
from torch.distributions import constraints
from pyro.infer.autoguide import AutoDelta
from matplotlib.backends.backend_pdf import PdfPages

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from utils.BoundedPareto import BoundedPareto

from collections import defaultdict
from pandas.core.common import flatten
from utils.plot_functions import *


def convert_to_list(item):
    """
    Recursively converts all NumPy arrays and PyTorch tensors in a dictionary or list
    to Python lists.
    """
    if isinstance(item, dict):
        return {key: convert_to_list(value) for key, value in item.items()}
    elif isinstance(item, list):
        return [convert_to_list(element) for element in item]
    elif isinstance(item, np.ndarray):  # Check if it's a NumPy array
        return item.tolist()
    elif isinstance(item, torch.Tensor):  # Check if it's a PyTorch tensor
        return item.detach().cpu().tolist()  # Detach from computation graph and convert to list
    else:
        return item


def fit(NV = None, DP = None, num_iter = 2000, K = [], tail=1, truncated_pareto = True, 
        purity=1, seed=[1,2,3], lr = 0.001, savefig = False, data_folder = None):
    """
    Function to run the inference with different values of K
    """
    min_bic = torch.tensor(float('inf'))
    best_K = torch.tensor(float('inf'))
    best_total_seed = torch.tensor(float('inf'))
    mb_list = []
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        print("No GPU available. Training will run on CPU.")

    for curr_k in K:
        j = 0
        curr_mb = []
        list_to_save = []
        min_bic_seed = torch.tensor(float('inf'))
        if curr_k != 0:
            # Fai ciclo con 4/5 seed diversi e prendi bic minore
            for curr_seed in seed:
                print(f"RUN WITH K = {curr_k} AND SEED = {curr_seed}")
                curr_mb.append(mobster_MV(NV, DP, K = curr_k, seed = curr_seed, savefig = savefig, data_folder = data_folder))
                curr_mb[j].run_inference(num_iter, lr)
                dict = copy.copy(curr_mb[j].__dict__)
                for key, value in dict.items():
                    if isinstance(value, torch.Tensor):  # Check if the value is a tensor
                        dict[key] = convert_to_list(value)#.detach().cpu().numpy()  # Convert tensor to lists
                list_to_save.append(dict)

                if curr_mb[j].final_dict['bic'] <= min_bic_seed:
                    min_bic_seed = curr_mb[j].final_dict['bic']
                    mb_best_seed = curr_mb[j]
                    
                j+=1
            if savefig:
                filename = f'saved_objects/{data_folder}/K_{curr_k}.txt'
                with open(filename, 'w') as f:
                    for dictionary in list_to_save:
                        # Recursively convert NumPy arrays to lists
                        dict_copy = convert_to_list(dictionary)
                        # Write the JSON representation of each dictionary to the file
                        f.write(json.dumps(dict_copy) + '\n')  # Write as a JSON string
            
            plot_marginals_new(mb_best_seed, savefig = savefig, data_folder = data_folder)
            # plot_marginals_alltogether(mb_best_seed, savefig = savefig, data_folder = data_folder)
            plot_deltas_new(mb_best_seed, savefig = savefig, data_folder = data_folder)
            plot_paretos(mb_best_seed, savefig = savefig, data_folder = data_folder)
            plot_betas(mb_best_seed, savefig = savefig, data_folder = data_folder)
            plot_responsib(mb_best_seed, savefig = savefig, data_folder = data_folder)
            
            mb_list.append(mb_best_seed)
            if mb_best_seed.final_dict['bic'] <= min_bic:
                min_bic = mb_best_seed.final_dict['bic']
                best_K = mb_best_seed.K
                best_total_seed = mb_best_seed.seed
    print(f"Selected number of clusters is {best_K} with seed {best_total_seed}")
    
    return mb_list, best_K, best_total_seed
    # return best_K, best_total_seed


class mobster_MV():
    def __init__(self, NV = None, DP = None, K = 1, tail=1, truncated_pareto = True, purity=1, seed=2, savefig = False, data_folder = None):
        """
        Parameters:
        
            NV : numpy array
                A numpy array containing the NV for each sample -> NV : [NV_s1, NV_s2, ..., NV_sn]
            DP : numpy array
                A numpy array containing the DP for each sample -> DP : [DP_s1, DP_s2, ..., DP_sn]
            K : int
                Number of clonal/subclonal clusters
            tail: int
                1 if inference is to perform with Pareto tail, 0 otherwise
            truncated_pareto: bool
                True if the pareto needs to be truncated at the mean of the lowest clonal cluster
            purity: float
                Previously estimated purity of the tumor
        """   

        if NV is not None:
            self.NV = torch.tensor(NV) if not isinstance(NV, torch.Tensor) else NV
        if DP is not None:
            self.DP = torch.tensor(DP) if not isinstance(DP, torch.Tensor) else DP
        self.K = K
        self.tail = tail
        self.truncated_pareto = truncated_pareto
        self.purity = purity
        self.seed = seed
        self.savefig = savefig
        self.data_folder = data_folder

        pyro.clear_param_store()
        pyro.set_rng_seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if NV is not None:
            self.set_prior_parameters()

    def compute_kmeans_centers(self):
        best_inertia = float('inf')
        best_centers = None
        best_labels = None
        # Implement loop to choose the seed which produces a result with the lowest inertia
        
        for seed in range(1, 10):
            kmeans = KMeans(n_clusters=self.K, random_state=seed, n_init=2).fit((self.NV/self.DP).numpy())
            best_cluster = kmeans.labels_.copy()
            centers = torch.tensor(kmeans.cluster_centers_)
            # Compute inertia (the lower the better)
            inertia = kmeans.inertia_
            
            # Update best results if current inertia is lower
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers.clone()
                best_labels = kmeans.labels_.copy()
        # -----------------Gaussian noise------------------#
        
        self.kmeans_centers_no_noise = best_centers.clone()
        # print("Kmeans centers:", self.kmeans_centers_no_noise)
        # self.kmeans_centers_no_noise[self.kmeans_centers_no_noise <= 0] = torch.min(self.min_vaf) # also used for init delta
        self.kmeans_centers_no_noise[self.kmeans_centers_no_noise <= 0] = 1e-10 # it could be 0 because now we are considering the private mutations 
        self.kmeans_centers_no_noise[self.kmeans_centers_no_noise >= 1] = 1 - 1e-10
        
        mean = 0
        std_dev = 0.005
        D = self.NV.shape[1]
        gaussian_noise = dist.Normal(mean, std_dev).sample([self.K, D])

        # Add gaussian noise to found centers
        best_centers = best_centers + gaussian_noise  
        # -----------------Gaussian noise------------------#
        
        # Clip probabilities in [min_vaf, 0.999]
        best_centers[best_centers <= torch.min(self.phi_beta_L)] = torch.min(self.phi_beta_L) # used as initial value of phi_beta
        # best_centers[best_centers <= 0] = 1 - 0.999
        best_centers[best_centers >= 1] = 0.999
        self.kmeans_centers = best_centers
        # self.kmeans_centers = self.kmeans_centers_no_noise
        # print(self.kmeans_centers)

        cluster_sizes = np.bincount(best_labels.astype(int), minlength=self.K)
        weights_kmeans = cluster_sizes.reshape(-1)/np.sum(cluster_sizes.reshape(-1))
        self.init_weights = torch.tensor(weights_kmeans)
        # print("self.init_weights: ", self.init_weights)
        """
        # INITIALIZE KAPPAS
        centroids = kmeans.cluster_centers_
        variances = []
        for i in range(self.K):
            vaf = self.NV/self.DP
            points_in_cluster = vaf[best_labels == i]
            
            # Get the centroid of the current cluster
            centroid = centroids[i]

            # Calculate the squared distances of points to the centroid
            distances = np.linalg.norm(points_in_cluster - centroid, axis=1) ** 2

            # Compute the variance (mean of squared distances)
            cluster_variance = np.mean(distances)
            variances.append(cluster_variance)

        kappas = [1 / variance if variance > 0 else np.inf for variance in variances]
        self.init_kappas = torch.tensor(np.tile(kappas, (self.NV.shape[1], 1)).T)
        """
        """
        # Print kmeans result
        plt.figure()
        sc = plt.scatter(self.NV[:,0]/self.DP[:,0], self.NV[:,1]/self.DP[:,1], c = best_labels)
        legend1 = plt.legend(*sc.legend_elements(), loc="lower right")
        plt.gca().add_artist(legend1)
        plt.title(f"Kmeans init (K = {self.K}, seed = {self.seed})")
        plt.xlim([0,1])
        plt.ylim([0,1])
        if self.savefig:
            plt.savefig(f"plots/{self.data_folder}/kmeans_K_{self.K}_seed_{self.seed}.png")
        plt.show()
        # print("kmeans_centers: ", self.kmeans_centers)
        plt.close()
        """
        

    def initialize_delta(self, phi_beta, k_beta, alpha):
        a_beta = self.get_a_beta(phi_beta, k_beta)
        b_beta = self.get_b_beta(phi_beta, k_beta)
        beta_lk = dist.Beta(a_beta, b_beta).log_prob(self.kmeans_centers_no_noise)
        # Note that I had to put 1 as upper bound of BoundedPareto because kmeans centers can also be bigger than 0.5 (due to my clip)
        # Otherwise the likelihood is infinite
        pareto_lk = BoundedPareto(self.pareto_L, alpha, 1).log_prob(self.kmeans_centers_no_noise)

        zeros_lk = dist.Beta(self.a_beta_zeros, self.b_beta_zeros).log_prob(self.kmeans_centers_no_noise)

        # print(self.kmeans_centers_no_noise)
        # print("Pareto lk", pareto_lk)
        # print("Beta lk", beta_lk)
        # print("Exp lk", exp_lk)
        # kmeans_centers: KxD
        K = self.K
        D = self.NV.shape[1]
        init_delta = torch.zeros((K,D,3))
        for i in range(K):
            for j in range(D):
                values = torch.tensor([pareto_lk[i,j].item(), beta_lk[i,j].item(), zeros_lk[i,j].item()])
                # print(values)
                sorted_indices = torch.argsort(values, descending=True) # list of indices ex. 0,2,1
                # print(sorted_indices)

                init_delta[i,j,sorted_indices[0]] = 0.6 # this can be either pareto, beta or exp
                init_delta[i,j,sorted_indices[1]] = 0.3
                init_delta[i,j,sorted_indices[2]] = 0.1
                # if(beta_lk[i,j] > pareto_lk[i,j]):
                #     init_delta[i,j,0] = 0.4 # pareto
                #     init_delta[i,j,1] = 0.5 # beta
                #     init_delta[i,j,2] = 0.1 # beta
                # else:
                #     init_delta[i,j,0] = 0.5 # pareto
                #     init_delta[i,j,1] = 0.4 # beta
                #     init_delta[i,j,2] = 0.1 # beta
        print("init_delta: ", init_delta)
        return init_delta


    def cluster_initialization(self):
        self.compute_kmeans_centers()
        self.init_delta = self.initialize_delta(self.kmeans_centers, self.k_beta_init, self.alpha_pareto_mean)


    def log_sum_exp(self, args):
        """
        Compute the log-sum-exp for each data point, i.e. the log-likelihood for each data point.
        log(p(x_i | theta)) = log(exp(a_1), ..., exp(a_K))
        where: a_k = log(pi_k) + sum^D log(Bin(x_{id} | DP_{id}, p_{dk})) 
        This function returns a N dimensional vector, where each entry corresponds to the log-likelihood of each data point.
        """
        if len(args.shape) == 1:
            args = args.unsqueeze(0)
        c = torch.amax(args, dim=0)
        return c + torch.log(torch.sum(torch.exp(args - c), axis=0)) # sum over the rows (different clusters), so obtain a single likelihood for each data


    def beta_lk(self, d, a_beta, b_beta):
        """
        Compute beta-binomial likelihood for a single dimension of a single cluster.
        """
        betabin = dist.BetaBinomial(a_beta, b_beta, total_count=self.DP[:,d]).log_prob(self.NV[:,d])
        return betabin # simply does log(weights) + log(density)
    
    def zeros_lk(self, d):
        """
        Compute beta-binomial likelihood for a single dimension of a single cluster, for values at 0.
        """
        dir = dist.BetaBinomial(self.a_beta_zeros, self.b_beta_zeros, total_count=self.DP[:,d]).log_prob(self.NV[:,d])
        return dir # simply does log(weights) + log(density)

    def pareto_binomial_pmf(self, NV, DP, alpha):
        integration_points=5000
        # Generate integration points across all rows at once
        t = torch.linspace(self.pareto_L, self.pareto_H, integration_points).unsqueeze(0)  # Shape (1, integration_points)
        NV_expanded = NV.unsqueeze(-1)  # Shape (NV.shape[0], 1)
        DP_expanded = DP.unsqueeze(-1)  # Shape (NV.shape[0], 1)
        binom_vals = dist.Binomial(total_count=DP_expanded, probs=t).log_prob(NV_expanded).exp()
        pareto_vals = BoundedPareto(self.pareto_L, alpha, self.pareto_H).log_prob(t).exp()  # Shape (1, integration_points)
        integrand = binom_vals * pareto_vals

        pmf_x = torch.trapz(integrand, t, dim=-1).log()  # Shape (NV.shape[0], NV.shape[1])

        return pmf_x.tolist()  # Convert the result to a list


    def pareto_lk_integr(self, d, alpha):
        paretobin = torch.tensor(self.pareto_binomial_pmf(NV=self.NV[:, d], DP=self.DP[:, d], alpha=alpha))
        return paretobin # tensor of len N (if D = 1, only N)

    
    def log_beta_par_mix_inference(self, probs_pareto, delta, alpha, a_beta, b_beta):
        # delta -> D x 2
        delta_pareto = torch.log(delta[:, 0]) + dist.Binomial(total_count=self.DP, probs = probs_pareto).log_prob(self.NV)  # NxD tensor
        delta_beta = torch.log(delta[:, 1]) + dist.BetaBinomial(a_beta, b_beta, total_count=self.DP).log_prob(self.NV) # N x D tensor
        delta_zeros = torch.log(delta[:, 2]) + dist.BetaBinomial(self.a_beta_zeros, self.b_beta_zeros, total_count=self.DP).log_prob(self.NV)
        
        # zeros_NV = np.where(self.NV == 0.)
        # print("Pareto", delta_pareto[zeros_NV])
        # print("Beta", delta_beta[zeros_NV])
        # print("Expon", delta_expon[zeros_NV])

        # Stack creates a 2 x N x D tensor to apply log sum exp on first dimension => it sums the delta_pareto and delta_beta 
        return self.log_sum_exp(torch.stack((delta_pareto, delta_beta, delta_zeros), dim=0)) # N

    
    def log_beta_par_mix_posteriors(self, delta, alpha, a_beta, b_beta):
        # delta -> D x 3
        log_lik = []
        for d in range(delta.shape[0]): # range over samples
            if ((delta[d, 0] > delta[d, 1]) and (delta[d, 0] > delta[d, 2])): # pareto
                # log_lik.append(torch.log(delta[d, 0]) + self.pareto_lk(d, alpha[d])) # N tensor
                log_lik.append(self.pareto_lk_integr(d, alpha[d])) # N tensor                
            elif ((delta[d, 1] > delta[d, 0]) and (delta[d, 1] > delta[d, 2])): # beta
                # log_lik.append(torch.log(delta[d, 1]) + self.beta_lk(d, a_beta[d], b_beta[d])) # N tensor
                log_lik.append(self.beta_lk(d, a_beta[d], b_beta[d])) # N tensor
            else:
                zeros_lk = self.zeros_lk(d)
                log_lik.append(zeros_lk)
        stack_tensors = torch.stack(log_lik, dim=1)
        # Non mi serve log sum exp perchè non devo più sommare sui delta_j
        return stack_tensors # N x D


    def m_total_lk(self, probs_pareto, alpha, a_beta, b_beta, weights, delta, which = 'inference'):
        lk = torch.ones(self.K, len(self.NV)) # matrix with K rows and as many columns as the number of data
        
        for k in range(self.K):
            if which == 'inference':
                lk[k, :] = torch.log(weights[k]) + self.log_beta_par_mix_inference(probs_pareto[k, :], delta[k, :, :], alpha[k, :], a_beta[k, :], b_beta[k, :]).sum(axis=1) # sums over the data dimensions (columns)
            else:
                lk[k, :] = torch.log(weights[k]) + self.log_beta_par_mix_posteriors(delta[k, :, :], alpha[k, :], a_beta[k, :], b_beta[k, :]).sum(axis=1) # sums over the data dimensions (columns)                                                                                                       # put on each column of lk a different data; rows are the clusters

        return lk


    def compute_min_vaf(self):
        """
        Function to compute the minimum vaf value (different from 0) found in 
        each dimension
        """
        vaf = self.NV/self.DP
        copy_vaf = torch.clone(vaf)
        # Replace zeros with a large value that will not be considered as minimum (i.e. 1)
        masked_tensor = copy_vaf.masked_fill(vaf == 0, float(1.))

        # Find the minimum value for each column excluding zeros
        min_values, _ = torch.min(masked_tensor, dim=0)
        min_values = min_values.repeat(self.K, 1)
        min_vaf = torch.min(min_values)
        print("Minimum detected VAF:", min_vaf)
        return min_vaf

    def set_prior_parameters(self):
        self.max_vaf = torch.tensor(0.55) # for a 1:1 karyotype
        self.min_vaf = self.compute_min_vaf()

        # phi_beta
        self.phi_beta_L = torch.tensor(0.15)
        self.phi_beta_H = 0.5
        # self.phi_beta_H = self.max_vaf

        # k_beta
        self.k_beta_L = torch.tensor(90.)
        self.k_beta_init = torch.tensor(200.) # which will be 90+200
        # self.k_beta_std = torch.tensor(100.)
        self.k_beta_mean = torch.tensor(200.)
        self.k_beta_std = torch.tensor(0.01)

        # alpha_pareto
        self.alpha_pareto_mean = torch.tensor(1.)
        self.alpha_pareto_std = torch.tensor(0.1)
        self.alpha_factor = torch.tensor(1.)
        self.alpha_pareto_init = torch.tensor(1.5)
        self.min_alpha = torch.tensor(0.5)
        self.max_alpha = torch.tensor(4.)

        # Bounded pareto
        self.pareto_L = self.min_vaf
        print("self.pareto_L", self.pareto_L)
        self.pareto_H = self.max_vaf
        self.probs_pareto_init = torch.tensor(0.09)

        # Approximated Dirac delta (very narrow Beta around 0)
        self.a_beta_zeros = torch.tensor(1e-3)
        self.b_beta_zeros = torch.tensor(1e3)

        self.temperature = 0.2


    def model(self):
        """
        Define the model.
        """
        NV = self.NV
        K = self.K
        D = NV.shape[1] # number of dimensions (samples)

        weights = pyro.sample("weights", dist.Dirichlet(torch.ones(K)))

        with pyro.plate("plate_dims", D):
            with pyro.plate("plate_probs", K):
                # Prior for the Beta-Pareto weights
                # delta is a K x D x 2 torch tensor (K: num layers, D: rows per layer, 2: columns per layer)
                delta = pyro.sample("delta", dist.Dirichlet(torch.ones(3)))

                w = pyro.sample("w", dist.RelaxedOneHotCategorical(torch.tensor([self.temperature]), probs=delta))
                
                phi_beta = pyro.sample("phi_beta", dist.Uniform(self.phi_beta_L, self.phi_beta_H)) # 0.5 because we are considering a 1:1 karyotype
                k_beta = pyro.sample("k_beta", dist.LogNormal(torch.log(self.k_beta_mean), self.k_beta_std))
                
                a_beta = self.get_a_beta(phi_beta, k_beta)
                b_beta = self.get_b_beta(phi_beta, k_beta)

                alpha = pyro.sample("alpha_pareto", dist.LogNormal(torch.log(self.alpha_pareto_mean), self.alpha_pareto_std))
                probs_pareto = pyro.sample("probs_pareto", BoundedPareto(self.pareto_L, alpha, self.pareto_H)) # probs_pareto is a K x D tensor
                # print(probs_pareto)
        # Data generation
        with pyro.plate("plate_data", len(NV)):
            # .sum() sums over the data because we have a log-likelihood
            pyro.factor("lik", self.log_sum_exp(self.m_total_lk(probs_pareto, self.alpha_factor*alpha, a_beta, b_beta, weights, w, which = 'inference')).sum())


    def get_a_beta(self, phi, kappa):
        return phi * (kappa + self.k_beta_L)


    def get_b_beta(self, phi, kappa):
        return (1-phi) * (kappa + self.k_beta_L)


    def init_fn(self, site):
        site_name = site["name"]
        param = None
        K, D = self.K, self.NV.shape[1]
        if site_name == "weights":
            param = self.init_weights
        if site_name == "delta":
            param = self.init_delta
        if site_name == "phi_beta":
            param = self.kmeans_centers
        if site_name == "k_beta":
            param = torch.ones((K,D))*self.k_beta_init
        if site_name == "alpha_pareto":
            param = torch.ones((K,D))*self.alpha_pareto_init
        if site_name == "probs_pareto":
            param = torch.ones((K,D))*self.probs_pareto_init
        # if site_name == "w":
        #     param = self.init_delta
        #     param[param<0.5] = 1e-10
        #     param[param>=0.5] = 1. - 1e-10
        return param


    def autoguide(self):
        return AutoDelta(poutine.block(self.model,
                                       expose=["weights","phi_beta","delta", "k_beta", 
                                               "alpha_pareto", "probs_pareto", "w"]),
                        init_loc_fn=self.init_fn)


    def guide(self):
        """
        Define the guide for the model.
        """
        NV = self.NV
        K = self.K
        D = NV.shape[1] # number of dimensions (samples)
        weights_param = pyro.param("weights_param", lambda: self.init_weights, constraint=constraints.simplex)
        pyro.sample("weights", dist.Delta(weights_param).to_event(1))

        alpha_param = pyro.param("alpha_pareto_param", lambda: torch.ones((K,D))*self.alpha_pareto_init, constraint=constraints.interval(self.min_alpha, self.max_alpha))        

        phi_beta_param = pyro.param("phi_beta_param", lambda: self.kmeans_centers, constraint=constraints.interval(self.phi_beta_L, self.phi_beta_H))
        # phi_beta_param = pyro.param("phi_beta_param", lambda: torch.ones((K,D))*0.2, constraint=constraints.interval(self.phi_beta_L, self.phi_beta_H))
        k_beta_param = pyro.param("k_beta_param", lambda: torch.ones((K,D))*self.k_beta_init, constraint=constraints.positive)

        probs_pareto_param = pyro.param("probs_pareto_param", lambda: torch.ones((K,D))*self.probs_pareto_init, constraint=constraints.interval(self.pareto_L, self.pareto_H))
        
        # probs_pareto_param = pyro.param("probs_pareto_param", lambda: torch.ones((K,D)*self.probs_pareto_init, constraint=constraints.interval(self.pareto_L, 0.15))
        
        delta_param = pyro.param("delta_param", lambda: self.init_delta, constraint=constraints.simplex)
        # delta_param = pyro.param("delta_param", lambda: dist.Dirichlet(torch.ones(3)).sample([K, D]).reshape(K, D, 3), constraint=constraints.simplex)
        # print(delta_param)
        # w_param = pyro.param(
        #     "w_param",
        #     lambda: torch.where(self.init_delta < 0.5, torch.tensor(1e-10), torch.tensor(1 - 1e-10)),
        #     constraint=constraints.simplex
        # )
        w_param = pyro.param("w_param", lambda: dist.Dirichlet(torch.ones(3)).sample([K, D]).reshape(K, D, 3), constraint=constraints.simplex)

        
        with pyro.plate("plate_dims", D):
            with pyro.plate("plate_probs", K):
                alpha = pyro.sample("alpha_pareto", dist.Delta(alpha_param)) # here because we need to have K x D samples
                
                pyro.sample("phi_beta", dist.Delta(phi_beta_param))
                pyro.sample("k_beta", dist.Delta(k_beta_param))

                # pyro.sample("probs_pareto", BoundedPareto(self.pareto_L, alpha, self.pareto_H))
                pyro.sample("probs_pareto", dist.Delta(probs_pareto_param))
                pyro.sample("delta", dist.Delta(delta_param).to_event(1))
                
                pyro.sample("w", dist.Delta(w_param).to_event(1))

                # pyro.sample("w", dist.RelaxedOneHotCategorical(torch.tensor([self.temperature]), probs = delta_param))


    def get_parameters(self):
        """
        Extract the learned parameters.
        """
        param_names = pyro.get_param_store()
        params = {nms: pyro.param(nms) for nms in param_names}

        new_keys = dict()
        for key in params.keys():
            if key.split(".")[0] == "AutoDelta":
                new_keys[key] = key.replace("AutoDelta.","") + "_param"
            else:
                new_keys[key] = key

        return dict((new_keys[key], value) for (key, value) in params.items())


    def flatten_params(self, pars):
        pars = list(flatten([value.detach().tolist() for key, value in pars.items()]))
        
        return(np.array(pars))


    def stopping_criteria(self, old_par, new_par, check_conv):#, e=0.01):
        old = self.flatten_params(old_par)
        new = self.flatten_params(new_par)
        diff_mix = np.abs(old - new)
        if np.all(diff_mix < old*0.1):
            return check_conv + 1 
        return 0


    # Model selection
    def compute_BIC(self, params, final_lk):
        # final_lk: K x N
        n_params = self.calculate_number_of_params(params)
        print("n_params: ", n_params)
        n = torch.tensor(self.NV.shape[0])
        print("n: ", n)
        lk = self.log_sum_exp(final_lk).sum()
        print("lk: ", lk)
        return torch.log(n) * n_params - torch.tensor(2.) * lk


    def compute_entropy(self, params):
        posts = params["responsib"] # K x N
        entropy = 0
        for k in range(posts.shape[0]):
            posts_k = posts[k,:] # len N
            log_post_k = torch.log(posts_k + 0.000001) # len N
            post_k_entr = posts_k * log_post_k  # len N (multiplication element-wise)
            post_k_entr = torch.sum(post_k_entr) # sum over N
            entropy += post_k_entr
        entropy = -1 * entropy
        return entropy


    def compute_ICL(self, params, bic):
        entropy = self.compute_entropy(params)
        return bic + entropy


    def run_inference(self, num_iter = 2000, lr = 0.001):
        pyro.clear_param_store()
        pyro.set_rng_seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.cluster_initialization()
        # NV, DP = self.NV, self.DP
        initial_temperature = torch.tensor(1)
        # final_temperature = 0.01
        # decr_rate = 0.99 
        # self.temperature = initial_temperature

        # svi = pyro.infer.SVI(self.model, self.autoguide(), pyro.optim.Adam({"lr": lr}), pyro.infer.TraceGraph_ELBO())
        svi = pyro.infer.SVI(self.model, self.guide, pyro.optim.Adam({"lr": lr}), pyro.infer.TraceGraph_ELBO())

        svi.step()
        gradient_norms = defaultdict(list)
        for name, value in pyro.get_param_store().named_parameters():
            value.register_hook(
                lambda g, name=name: gradient_norms[name].append(g.norm().item())
            )
        
        self.losses = []
        self.lks = []
        i = 0
        min_iter = 100
        check_conv = 0
        old_par = self.get_parameters() # Save current values of the parameters in old_params

        

        for i in range(num_iter):
            # curr_temperature = max(final_temperature, initial_temperature * (decr_rate ** i))
            # if i != 0:
            #     curr_temperature = initial_temperature/torch.log(torch.tensor(i + 0.1))
            #     self.temperature = torch.tensor([curr_temperature])
            # else:
            #     self.temperature = initial_temperature

            loss = svi.step()
            self.losses.append(loss)

            # Save likelihood values
            params = self.get_parameters()
            # if i == 0:
            #     print("delta_param", params["delta_param"])
            
            a_beta = self.get_a_beta(params["phi_beta_param"], params["k_beta_param"])
            b_beta = self.get_b_beta(params["phi_beta_param"], params["k_beta_param"])
            probs_pareto = params["probs_pareto_param"] if "probs_pareto_param" in params.keys() else BoundedPareto(self.pareto_L, params["alpha_pareto_param"], self.pareto_H).sample()
            lks = self.log_sum_exp(self.m_total_lk(probs_pareto,
                                                   params["alpha_pareto_param"], a_beta, b_beta, 
                                                   params["weights_param"], params["delta_param"], which = 'inference')).sum()

            self.lks.append(lks.detach().numpy())

            new_par = params.copy()
            check_conv = self.stopping_criteria(old_par, new_par, check_conv)
            
            # If convergence is reached (i.e. changes in parameters are small for min_iter iterations), stop the loop
            if check_conv == min_iter:
                break
            if i % 200 == 0:
                print("Iteration {}: Loss = {}".format(i, loss))
                # print("temperature", self.temperature)
                # print("delta_param", params['delta_param'])
                # print("w_param", params['w_param'])

        self.params = self.get_parameters()

        self.plot_loss_lks()
        self.plot_grad_norms(gradient_norms)
        
        final_lk = self.compute_posteriors()
        print("Inference lk: ", self.lks[-1])
        print("Final lk (integr): ", self.log_sum_exp(final_lk).sum())

        print("INTEGR: ")
        bic = self.compute_BIC(self.params, final_lk)
        print(f"bic: {bic} \n")
        icl = self.compute_ICL(self.params, bic)

        self.plot()
        self.params['k_beta_param'] = self.params['k_beta_param'] + self.k_beta_L
        
        self.final_dict = {
        "model_parameters" : self.params,
        "bic": bic,
        "icl": icl,

        # "final_likelihood": self.lks[-1], # during training, so no difference between integr and sampling_p
        "final_likelihood": self.log_sum_exp(final_lk).sum(), 
        "final_loss": self.losses[-1] # during training, so no difference between integr and sampling_p
        }


    def compute_final_lk(self, which):
        """
        Compute likelihood with learnt parameters
        """
        alpha = self.params["alpha_pareto_param"] * self.alpha_factor
        # delta = self.params["delta_param"]  # K x D x 2
        delta = self.params["w_param"]  # K x D x 2
        
        phi_beta = self.params["phi_beta_param"]
        k_beta = self.params["k_beta_param"]
        a_beta = self.get_a_beta(phi_beta, k_beta)
        b_beta = self.get_b_beta(phi_beta, k_beta)
        weights = self.params["weights_param"]
        
        lks = self.m_total_lk(probs_pareto = None, alpha=alpha, a_beta=a_beta, b_beta=b_beta, 
                               weights=weights, delta=delta, which = which) 

        return lks

    def compute_posteriors(self):
        # self.NV[self.zero_NV_idx] = torch.tensor(0, dtype=self.NV.dtype)
        lks = self.compute_final_lk(which = 'posteriors') # K x N
        res = torch.zeros(self.K, len(self.NV)) # K x N
        norm_fact = self.log_sum_exp(lks) # sums over the different cluster -> array of size 1 x len(NV)
        for k in range(len(res)): # iterate over the clusters
            lks_k = lks[k] # take row k -> array of size len(NV)
            res[k] = torch.exp(lks_k - norm_fact)
        self.params["responsib"] = res # qui non dovrebbe cambiare niente per le private perchè i punti sono già sommati sulle dimensioni
        self.params["cluster_assignments"] = torch.argmax(self.params["responsib"], dim = 0) # vector of dimension
        # self.NV[self.zero_NV_idx] = torch.tensor(0, dtype=self.NV.dtype)
        
        return lks


    def plot_loss_lks(self):
        _, ax = plt.subplots(1, 2, figsize=(15, 5))
        ax[0].plot(self.losses)
        ax[0].set_title(f"Loss (K = {self.K}, seed = {self.seed})")

        ax[1].plot(self.lks)
        ax[1].set_title(f"Likelihood (K = {self.K}, seed = {self.seed})")
        
        if self.savefig:
            plt.savefig(f"plots/{self.data_folder}/likelihood_K_{self.K}_seed_{self.seed}.png")
        plt.show()
        plt.close()


    def plot_grad_norms(self, gradient_norms):
        plt.figure(figsize=(10, 4), dpi=100).set_facecolor("white")
        for name, grad_norms in gradient_norms.items():
            plt.plot(grad_norms, label=name)
        plt.xlabel("iters")
        plt.ylabel("gradient norm")
        plt.yscale("log")
        plt.legend(loc="best")
        plt.title(f"Gradient norms during SVI (K = {self.K}, seed = {self.seed})")
        if self.savefig:
            plt.savefig(f"plots/{self.data_folder}/gradient_norms_K_{self.K}_seed_{self.seed}.png")
        plt.show()
        plt.close()


    def plot(self):
        """
        PLOT I HAVE AT THE MOMENT
        Plot the results.
        """
        D = self.NV.shape[1]
        pairs = np.triu_indices(D, k=1)  # Generate all unique pairs of samples (i, j)
        vaf = self.NV/self.DP  

        weights = self.params["weights_param"].detach().numpy()
        print("Weights: ", weights)
        unique_labels = np.unique(self.params["cluster_assignments"].detach().numpy())
        labels = self.params["cluster_assignments"].detach().numpy()
        cmap = cm.get_cmap('tab20')
        color_mapping = {label: cmap(i) for i, label in enumerate(unique_labels)}
        colors = [color_mapping[label] for label in labels]

        idx = 0
        for i, j in zip(*pairs):
            x = vaf[:, i].numpy()
            y = vaf[:, j].numpy()

            sc = plt.scatter(x, y, c=colors, cmap = 'Set3') # tab10
            handles = [plt.Line2D([0], [0], marker='o', color='w', label=f"{label} (\u03C0 = {weights[label]:.3f})",
                        markerfacecolor=color_mapping[label], markersize=10) 
            for label in unique_labels]
            plt.legend(handles=handles)
        
            plt.title(f"Inference Sample {i+1} vs Sample {j+1} (K = {self.K}, seed {self.seed})")        
            plt.xlabel(f"Sample {i+1}")
            plt.ylabel(f"Sample {j+1}")
            plt.xlim([0,1])
            plt.ylim([0,1])
            if self.savefig:
                plt.savefig(f"plots/{self.data_folder}/inference_K_{self.K}_seed_{self.seed}_{idx}.png")
                idx += 1
            plt.show()
            plt.close()
        
    def calculate_number_of_params(self, params):
        keys = ["phi_beta_param", "k_beta_param", "alpha_pareto_param", "delta_param", "weights_param", "probs_pareto_param", "w_param"]
        total_params = 0
        for key, param in params.items():
            if key in keys:
                param_size = np.prod(param.shape)  # Calculate the total number of elements
                total_params += param_size
        return total_params
    