import numpy as np
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import copy
import json
import time
import matplotlib



import torch
from torch.distributions import constraints
from pyro.infer.autoguide import AutoDelta
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.mixture import GaussianMixture
import pandas as pd
from itertools import combinations

# import matplotlib.pyplot as plt
# plt.rcParams['font.family'] = 'Arial' # Set Arial as the font
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


def fit(NV = None, DP = None, num_iter = 2000, K = [], purity=None, seed=[123,1234], par_threshold = 0.005, loss_threshold = 0.01, lr = 0.01, savefig = False, data_folder = None):
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
        print("No GPU available.")

    for curr_k in K:
        j = 0
        curr_mb = []
        list_to_save = []
        min_bic_seed = torch.tensor(float('inf'))
        if curr_k != 0:
            # Fai ciclo con 4/5 seed diversi e prendi bic minore
            for curr_seed in seed:
                print(f"RUN WITH K = {curr_k} AND SEED = {curr_seed}")
                start_time = time.time()
                curr_mb.append(mobster_MV(NV, DP, K = curr_k, purity = purity, 
                                            seed = curr_seed, par_threshold = par_threshold,
                                            loss_threshold = loss_threshold, savefig = savefig, 
                                            data_folder = data_folder))
                curr_mb[j].run_inference(num_iter, lr)
                plot_scatter_inference(curr_mb[j])
                plot_marginals_inference(curr_mb[j])
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Time taken for K = {curr_k} and seed = {curr_seed}: {elapsed_time:.3f} seconds")

                dict = copy.copy(curr_mb[j].__dict__)
                for key, value in dict.items():
                    if isinstance(value, torch.Tensor):  # Check if the value is a tensor
                        dict[key] = convert_to_list(value)#.detach().cpu().numpy()  # Convert tensor to lists
                list_to_save.append(dict)

                
                if curr_mb[j].final_dict['icl'] <= min_bic_seed:
                    min_bic_seed = curr_mb[j].final_dict['icl']
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
            
            plot_marginals(mb_best_seed, savefig = savefig, data_folder = data_folder)
            # plot_marginals_alltogether(mb_best_seed, savefig = savefig, data_folder = data_folder)
            plot_deltas(mb_best_seed, savefig = savefig, data_folder = data_folder)
            plot_paretos(mb_best_seed, savefig = savefig, data_folder = data_folder)
            plot_betas(mb_best_seed, savefig = savefig, data_folder = data_folder)
            plot_responsib(mb_best_seed, savefig = savefig, data_folder = data_folder)
            plot_mixing_proportions(mb_best_seed, savefig = savefig, data_folder = data_folder)
            
            mb_list.append(mb_best_seed)
            if mb_best_seed.final_dict['icl'] <= min_bic:
                min_bic = mb_best_seed.final_dict['icl']
                best_K = mb_best_seed.K
                best_total_seed = mb_best_seed.seed
    print(f"Selected number of clusters is {best_K} with seed {best_total_seed}")
    
    return mb_list, best_K, best_total_seed
    # return best_K, best_total_seed


class mobster_MV():
    def __init__(self, NV = None, DP = None, K = 1, purity=None, seed=[123,1234], 
                    par_threshold = 0.005, loss_threshold = 0.01, savefig = False, data_folder = None):
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

        if NV is not None and DP is not None:
            if NV.ndim == 1:
                NV = NV.unsqueeze(-1)
                DP = DP.unsqueeze(-1)
            vaf = NV.numpy()/DP.numpy()
            cond = np.where(np.all(((vaf == 0) | (vaf >= 0.03)), axis=1))[0]
            NV = NV[cond,:]
            self.NV = torch.tensor(NV) if not isinstance(NV, torch.Tensor) else NV
            DP = DP[cond,:]
            self.DP = torch.tensor(DP) if not isinstance(DP, torch.Tensor) else DP
            
        self.K = K
        if  NV is not None and purity is not None:
            self.purity = torch.tensor(purity)
        elif NV is not None:
            self.purity = torch.ones(NV.shape[1])
        self.seed = seed
        self.par_threshold = par_threshold
        self.loss_threshold = loss_threshold
        self.savefig = savefig
        self.data_folder = data_folder

        pyro.clear_param_store()
        pyro.set_rng_seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if NV is not None:
            self.set_prior_parameters()
        
        # vaf = NV.numpy()/DP.numpy()
        # self.idx = np.where((vaf[:,0] >= 0.03) & (vaf[:,0] < 0.2) & (vaf[:,1] == 0))[0]
    

    def compute_kmeans_centers(self):
        best_inertia = float('inf')
        best_bic = float('inf')
        self.best_centers = None
        best_labels = None
        best_weights = None
        best_cov = None
        
        """
        # Loop to choose the seed which produces a result with the lowest inertia
        for seed in range(1, 50):
            kmeans = KMeans(n_clusters=self.K, random_state=seed, max_iter=1).fit((self.NV/self.DP).numpy())
            centers = torch.tensor(kmeans.cluster_centers_)
            # Compute inertia (the lower the better)
            inertia = kmeans.inertia_
            
            # Update best results if current inertia is lower
            if inertia < best_inertia:
                best_inertia = inertia
                self.best_centers = centers.clone()
                best_labels = kmeans.labels_.copy()
        cluster_sizes = np.bincount(best_labels.astype(int), minlength=self.K)
        weights_kmeans = cluster_sizes.reshape(-1)/np.sum(cluster_sizes.reshape(-1))
        self.init_weights = torch.tensor(weights_kmeans)
        """
        for seed in range(1, 15):
            gmm = GaussianMixture(n_components=self.K, covariance_type='full', random_state=seed).fit((self.NV/self.DP).numpy())
            bic = gmm.bic((self.NV/self.DP).numpy())
            # Update best results if current bic is lower
            if bic < best_bic:
                best_bic = bic
                best_labels = gmm.predict((self.NV/self.DP).numpy())
                best_cov = gmm.covariances_
                best_weights = gmm.weights_
                self.best_centers = torch.tensor(gmm.means_)
        
        self.init_weights = torch.tensor(best_weights)
        
        # -----------------Gaussian noise------------------#
        
        self.kmeans_labels = torch.tensor(best_labels).clone()
        self.kmeans_centers_no_noise = self.best_centers.clone()
        # cov = torch.tensor(np.abs(np.array([np.diagonal(cov) for cov in gmm.covariances_])))
        # # print(torch.tensor(best_cov).shape)
        # print(cov)
        # self.init_kappas = ((self.kmeans_centers_no_noise*(1-self.kmeans_centers_no_noise)/cov) -1)-self.k_beta_L
        # print(self.init_kappas)

        # self.kmeans_centers_no_noise[self.kmeans_centers_no_noise <= 0] = torch.min(self.min_vaf) # also used for init delta
        self.kmeans_centers_no_noise[self.kmeans_centers_no_noise <= 0] = 1e-10 # it could be 0 because now we are considering the private mutations 
        # self.kmeans_centers_no_noise[self.kmeans_centers_no_noise >= self.max_vaf] = self.max_vaf - 1e-5
        
        self.kmeans_centers_no_noise = torch.minimum(self.kmeans_centers_no_noise, self.max_vaf.unsqueeze(0) - 1e-5)
        

    def noise_kmeans(self):
        # Add gaussian noise to found centers
        self.best_centers = self.best_centers + self.gaussian_noise  
        self.best_centers = torch.maximum(self.best_centers, self.phi_beta_L.unsqueeze(0) + 1e-5)
        self.best_centers = torch.minimum(self.best_centers, self.max_vaf.unsqueeze(0) - 1e-5)
        self.kmeans_centers = self.best_centers
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
        """"""
        

    def initialize_delta(self):
        phi_beta = self.kmeans_centers
        a_beta = self.get_a_beta(phi_beta, self.k_beta_init)
        b_beta = self.get_b_beta(phi_beta, self.k_beta_init)
        beta_lk = dist.Beta(a_beta, b_beta).log_prob(self.kmeans_centers_no_noise)
        # Note that I had to put 1 as upper bound of BoundedPareto because kmeans centers can also be bigger than 0.5 (due to my clip)
        # Otherwise the likelihood is infinite
        pareto_lk = BoundedPareto(self.pareto_L, self.alpha_pareto_mean, self.pareto_H).log_prob(self.kmeans_centers_no_noise)

        zeros_lk = dist.Beta(self.a_beta_zeros, self.b_beta_zeros).log_prob(self.kmeans_centers_no_noise)
        # kmeans_centers: KxD
        K = self.K
        D = self.NV.shape[1]
        init_delta = torch.zeros((K,D,3))
        self.dirichlet_conc = torch.zeros((K,D,3))
        
        for i in range(K):
            for j in range(D):
                """
                if self.kmeans_centers_no_noise[i,j] < 1e-3: # dirac
                    init_delta[i,j] = torch.tensor([0.3, 0.3, 0.6])
                    self.dirichlet_conc[i,j] = torch.tensor([1, 1, 1000])
                elif self.kmeans_centers_no_noise[i,j] < self.phi_beta_L: # pareto
                    init_delta[i,j] = torch.tensor([0.6, 0.3, 0.3])
                    self.dirichlet_conc[i,j] = torch.tensor([1000, 1, 1])
                else:
                    init_delta[i,j] = torch.tensor([0.3, 0.6, 0.3])
                    self.dirichlet_conc[i,j] = torch.tensor([1, 1000, 1])
            
                """
                values = torch.tensor([pareto_lk[i,j].item(), beta_lk[i,j].item(), zeros_lk[i,j].item()])
                # print(values)
                sorted_indices = torch.argsort(values, descending=True) # list of indices ex. 0,2,1
                # print(sorted_indices)
                
                init_delta[i,j,sorted_indices[0]] = 0.5 # this can be either pareto, beta or private
                init_delta[i,j,sorted_indices[1]] = 0.3
                init_delta[i,j,sorted_indices[2]] = 0.2
                """
                init_delta[i,j,sorted_indices[0]] = 0.7 # this can be either pareto, beta or private
                init_delta[i,j,sorted_indices[1]] = 0.2
                init_delta[i,j,sorted_indices[2]] = 0.1
                """
                self.dirichlet_conc[i,j,sorted_indices[0]] = 10. # this can be either pareto, beta or private
                self.dirichlet_conc[i,j,sorted_indices[1]] = 1.
                self.dirichlet_conc[i,j,sorted_indices[2]] = 1.
                
        # print("Init delta: ", init_delta)
        return init_delta


    def cluster_initialization(self, compute_kmeans = True, final = False):
        if compute_kmeans == True:
            self.compute_kmeans_centers()
        else:
            self.noise_kmeans()
            self.init_delta = self.initialize_delta()
            if final == True:
                self.plot_initial_assignments()

    def plot_initial_assignments(self):
        # Plot kmeans result scatter
        plt.figure()
        unique_labels = np.unique(self.kmeans_labels)
        cmap = cm.get_cmap('tab20')
        color_mapping = {label: cmap(i) for i, label in enumerate(unique_labels)}
        colors = [color_mapping[label] for label in self.kmeans_labels.detach().numpy()]

        # sc = plt.scatter(self.NV[:,0]/self.DP[:,0], self.NV[:,1]/self.DP[:,1], c = self.kmeans_labels, s = 20)
        # legend1 = plt.legend(*sc.legend_elements(), loc="lower right")
        # plt.gca().add_artist(legend1)

        sc = plt.scatter(self.NV[:,0]/self.DP[:,0], self.NV[:,1]/self.DP[:,1], c=colors, s=20)  # tab20
        handles = [plt.Line2D([0], [0], marker='o', color='w', label=f"{label}",
                            markerfacecolor=color_mapping[label], markersize=10) 
                for label in unique_labels]
        plt.legend(handles=handles)

        plt.scatter(self.kmeans_centers_no_noise[:, 0], 
                    self.kmeans_centers_no_noise[:, 1], color='red', marker='x', s=25)
        plt.title(f"GMM init (K = {self.K}, seed = {self.seed})")
        plt.xlim([0,1])
        plt.ylim([0,1])
        if self.savefig:
            plt.savefig(f"plots/{self.data_folder}/kmeans_K_{self.K}_seed_{self.seed}.png")
        plt.show()
        # print("kmeans_centers: ", self.kmeans_centers_no_noise)
        plt.close()

        if self.K == 1:
            fig, axes = plt.subplots(self.K, self.NV.shape[1], figsize=(16, 4))
        else:
            fig, axes = plt.subplots(self.K, self.NV.shape[1], figsize=(16, self.K*3))
        if self.K == 1:
            axes = ax = np.array([axes])  # add an extra dimension to make it 2D
        plt.suptitle(f"GMM marginals with K={self.K}, seed={self.seed}",fontsize=14)
        x = np.linspace(0.001, 1, 1000)

        cmap = cm.get_cmap('tab20')
        color_mapping = {label: cmap(i) for i, label in enumerate(unique_labels)}
        phi = self.kmeans_centers
        for k in range(self.K):
            for d in range(self.NV.shape[1]):
                delta_kd = self.init_delta[k, d]
                maxx = torch.argmax(delta_kd)
                if maxx == 1:
                    # plot beta
                    a = phi[k,d] * self.k_beta_init
                    b = (1-phi[k,d]) * self.k_beta_init
                    pdf = beta.pdf(x, a, b)# * weights[k]
                    axes[k,d].plot(x, pdf, linewidth=1.5, label='Beta', color='r')
                    axes[k,d].legend()
                elif maxx == 0:
                    # plot pareto
                    pdf = pareto.pdf(x, self.alpha_pareto_init, scale=self.pareto_L) #* weights[k]
                    axes[k,d].plot(x, pdf, linewidth=1.5, label='Pareto', color='g')
                    axes[k,d].legend()
                else:
                    # private
                    pdf = beta.pdf(x, self.a_beta_zeros, self.b_beta_zeros) # delta_approx
                    axes[k,d].plot(x, pdf, linewidth=1.5, label='Zeros', color='b')
                    axes[k,d].legend()

                if torch.is_tensor(self.NV):
                    data = self.NV[:,d].numpy()/self.DP[:,d].numpy()
                else:
                    data = np.array(self.NV[:,d])/np.array(self.DP[:,d])
                # for i in np.unique(labels):
                if k in unique_labels:
                    axes[k, d].hist(data[self.kmeans_labels == k],  density=True, bins=30, alpha=1, color=color_mapping[k])
                else:
                    # Plot an empty histogram because we know there are no points in that k
                    axes[k, d].hist([], density=True, bins=30, alpha=1)
                axes[k,d].set_title(f"Sample {d+1} - Cluster {k}")
                axes[k,d].grid(True, color='gray', linestyle='-', linewidth=0.2)
                # axes[k,d].set_ylim([0,25])
                axes[k,d].set_xlim([-0.01,0.8])
                plt.tight_layout()
        if self.savefig:
            plt.savefig(f"plots/{self.data_folder}/kmeans_marginals_K_{self.K}_seed_{self.seed}.png")
        plt.show()
        plt.close()

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

    def pareto_binomial_pmf(self, NV, DP, alpha, H):
        integration_points=500
        # Generate integration points across all rows at once
        t = torch.linspace(self.pareto_L, H, integration_points).unsqueeze(0)  # Shape (1, integration_points)
        NV_expanded = NV.unsqueeze(-1)  # Shape (NV.shape[0], 1)
        DP_expanded = DP.unsqueeze(-1)  # Shape (NV.shape[0], 1)
        binom_vals = dist.Binomial(total_count=DP_expanded, probs=t).log_prob(NV_expanded).exp()
        pareto_vals = BoundedPareto(self.pareto_L, alpha, H).log_prob(t).exp()  # Shape (1, integration_points)
        integrand = binom_vals * pareto_vals

        pmf_x = torch.trapz(integrand, t, dim=-1).log()  # Shape (NV.shape[0], NV.shape[1])

        return pmf_x.tolist()  # Convert the result to a list

    def pareto_lk_integr(self, d, alpha):
        paretobin = torch.tensor(self.pareto_binomial_pmf(NV=self.NV[:, d], DP=self.DP[:, d], alpha=alpha, H = self.pareto_H[d]))
        return paretobin # tensor of len N (if D = 1, only N)
    
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
    
    def log_beta_par_mix_inference_old(self, probs_pareto, delta, alpha, a_beta, b_beta):
        # delta -> (D, 3)
        # a_beta, b_beta, probs_pareto -> [D]
        delta_pareto = torch.log(delta[:, 0]) + dist.Binomial(total_count=self.DP, probs = probs_pareto).log_prob(self.NV)  # N x D tensor
        delta_beta = torch.log(delta[:, 1]) + dist.BetaBinomial(a_beta, b_beta, total_count=self.DP).log_prob(self.NV) # N x D tensor
        delta_zeros = torch.log(delta[:, 2]) + dist.BetaBinomial(self.a_beta_zeros, self.b_beta_zeros, total_count=self.DP).log_prob(self.NV) # N x D tensor
        
        assert delta_pareto.shape == (self.NV.shape[0],self.NV.shape[1])
        assert delta_beta.shape == (self.NV.shape[0],self.NV.shape[1])
        assert delta_zeros.shape == (self.NV.shape[0],self.NV.shape[1])
        
        # print("Pareto", delta_pareto[self.idx,:].sum())
        # print("Beta", delta_beta[self.idx,:].sum())
        # print("Dirac", delta_zeros[self.idx,:].sum())

        # Stack creates a 3 x N x D tensor to apply log sum exp on first dimension => it sums the delta_pareto, delta_beta and delta_zeros
        log = self.log_sum_exp(torch.stack((delta_pareto, delta_beta, delta_zeros), dim=0))
        assert log.shape == (self.NV.shape[0],self.NV.shape[1])
        
        return  log # N x D
    
    def log_beta_par_mix_inference(self, probs_pareto, delta, alpha, a_beta, b_beta):
        # delta -> (K, D, 3)
        # a_beta, b_beta, probs_pareto -> (K, D)

        expanded_DP = self.DP.unsqueeze(0)  # Shape: 1 x N x D
        expanded_NV = self.NV.unsqueeze(0)  # Shape: 1 x N x D
        """
        delta_ = torch.zeros((self.K, self.NV.shape[1],3))
        # delta[:, :, 0] (K, D) => I want it to be K x 1 x D => unsqueeze
        for k in range(self.K):
            for d in range(self.NV.shape[1]):
                if ((delta[k, d, 0] > delta[k, d, 1]) and (delta[k, d, 0] > delta[k, d, 2])): # pareto
                    delta_[k, d, 0] = 1 - 1e-10
                    delta_[k, d, 1] = (1 - delta_[k, d, 0])/2
                    delta_[k, d, 2] = (1 - delta_[k, d, 0])/2
                elif ((delta[k,d, 1] > delta[k,d, 0]) and (delta[k,d, 1] > delta[k,d, 2])): # beta
                    delta_[k, d, 1] = 1 - 1e-10
                    delta_[k, d, 0] = (1 - delta_[k, d, 0])/2
                    delta_[k, d, 2] = (1 - delta_[k, d, 0])/2
                else:
                    delta_[k, d, 2] = 1 - 1e-10
                    delta_[k, d, 1] = (1 - delta_[k, d, 0])/2
                    delta_[k, d, 0] = (1 - delta_[k, d, 0])/2
        """
        # delta_ = dist.RelaxedOneHotCategorical(torch.tensor([self.temperature]), probs=delta).sample()
        delta_ =  torch.softmax(delta / self.temperature, dim=-1)
        # print(delta_)
        delta_pareto = torch.log(delta_[:, :, 0].unsqueeze(1)) + dist.Binomial(total_count=expanded_DP, probs = probs_pareto.unsqueeze(1)).log_prob(expanded_NV)  # K x N x D tensor
        delta_beta = torch.log(delta_[:, :, 1].unsqueeze(1)) + dist.BetaBinomial(a_beta.unsqueeze(1), b_beta.unsqueeze(1), total_count=expanded_DP).log_prob(expanded_NV) # K x N x D tensor
        delta_zeros = torch.log(delta_[:, :, 2].unsqueeze(1)) + dist.BetaBinomial(self.a_beta_zeros, self.b_beta_zeros, total_count=expanded_DP).log_prob(expanded_NV) # K x N x D tensor
       
        
        """
        delta_pareto = torch.log(delta[:, :, 0].unsqueeze(1)) + dist.Binomial(total_count=expanded_DP, probs = probs_pareto.unsqueeze(1)).log_prob(expanded_NV)  # K x N x D tensor
        delta_beta = torch.log(delta[:, :, 1].unsqueeze(1)) + dist.BetaBinomial(a_beta.unsqueeze(1), b_beta.unsqueeze(1), total_count=expanded_DP).log_prob(expanded_NV) # K x N x D tensor
        delta_zeros = torch.log(delta[:, :, 2].unsqueeze(1)) + dist.BetaBinomial(self.a_beta_zeros, self.b_beta_zeros, total_count=expanded_DP).log_prob(expanded_NV) # K x N x D tensor
        """
        """
        print("Pareto", delta_pareto[4,self.idx,:].sum(axis = 0))
        print("Beta", delta_beta[4,self.idx,:].sum(axis = 0))
        print("Dirac", delta_zeros[4,self.idx,:].sum(axis = 0))
        """

        # Stack creates a 3 x K x N x D tensor to apply log sum exp on first dimension => it sums the delta_pareto, delta_beta and delta_zeros 
        stacked = torch.stack((delta_pareto, delta_beta, delta_zeros), dim=0) # 3 x K x N x D tensor
        log = self.log_sum_exp(stacked) # apply log sum exp on first dimension => K x N x D tensor
        
        assert log.shape == (self.K, self.NV.shape[0],self.NV.shape[1])
        
        return  log # K x N x D
    
    def m_total_lk(self, probs_pareto, alpha, a_beta, b_beta, weights, delta, which = 'inference'):
        lk = torch.ones(self.K, len(self.NV)) # matrix with K rows and as many columns as the number of data
        
        start_time_tot = time.time()
        if which == 'inference':
            start_time = time.time()
            log_sum = self.log_beta_par_mix_inference(probs_pareto, delta, alpha, a_beta, b_beta).sum(axis=2) # K x N x D, then sum over the data dimensions (columns) => K x N tensor
            end_time = time.time()
            elapsed_time = end_time - start_time
            # print(f"Time taken for log_beta_par_mix_inference: {elapsed_time:.2f} seconds")
            log_weights = torch.log(weights).unsqueeze(1)  # Shape: (K, 1)
            lk = log_weights + log_sum  # Broadcasting: (K, 1) + (K, N) → (K, N)
        else:
            for k in range(self.K):
                lk[k, :] = torch.log(weights[k]) + self.log_beta_par_mix_posteriors(delta[k, :, :], alpha[k, :], a_beta[k, :], b_beta[k, :]).sum(axis=1) # sums over the data dimensions (columns)                                                                                                       # put on each column of lk a different data; rows are the clusters
        
        end_time_tot = time.time()
        elapsed_time_tot = end_time_tot - start_time_tot
        # print(f"Time taken tot {elapsed_time_tot:.2f} seconds")

        assert lk.shape == (self.K, len(self.NV))
        """
        for k in range(self.K):
            if which == 'inference':
                lk[k, :] = torch.log(weights[k]) + self.log_beta_par_mix_inference_old(probs_pareto[k, :], delta[k, :, :], alpha[k, :], a_beta[k, :], b_beta[k, :]).sum(axis=1) # sums over the data dimensions (columns)
            else:
                lk[k, :] = torch.log(weights[k]) + self.log_beta_par_mix_posteriors(delta[k, :, :], alpha[k, :], a_beta[k, :], b_beta[k, :]).sum(axis=1) # sums over the data dimensions (columns)                                                                                                       # put on each column of lk a different data; rows are the clusters
        """
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
        self.max_vaf = self.purity/2 # for a 1:1 karyotype, tensor D x 1
        self.min_vaf = self.compute_min_vaf()

        # phi_beta
        self.phi_beta_L = torch.ones(self.NV.shape[1])*0.1
        # self.phi_beta_L = self.min_vaf
        self.phi_beta_H = self.max_vaf

        # k_beta
        self.k_beta_L = torch.tensor(50.)
        # self.k_beta_L = torch.tensor(0.)
        self.k_beta_init = torch.tensor(150.) # which will be 90+200
        self.k_beta_mean = torch.tensor(150.)
        self.k_beta_std = torch.tensor(0.01)

        # alpha_pareto
        # self.alpha_pareto_mean = torch.tensor(1.)
        # self.alpha_pareto_std = torch.tensor(0.1)
        self.alpha_pareto_mean = torch.tensor(1.8)
        self.alpha_pareto_std = torch.tensor(0.5)
        # self.alpha_pareto_mean = torch.tensor(2.7)
        # self.alpha_pareto_std = torch.tensor(1.)
        
        self.alpha_pareto_init = torch.tensor(1.5)
        # self.alpha_pareto_init = torch.tensor(2.5)
        self.min_alpha = torch.tensor(0.5)
        self.max_alpha = torch.tensor(4.)

        # Bounded pareto
        self.pareto_L = self.min_vaf
        self.pareto_H = self.max_vaf
        self.probs_pareto_init = torch.tensor(0.09)

        # Approximated Dirac delta (very narrow Beta around 0)
        self.a_beta_zeros = torch.tensor(1e-4)
        self.b_beta_zeros = torch.tensor(1e4)

        self.temperature = 0.1

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
                # delta is a K x D x 3 torch tensor (K: num layers, D: rows per layer, 3: columns per layer)
                
                # print(dist.Dirichlet(self.dirichlet_conc).sample())
                delta = pyro.sample("delta", dist.Dirichlet(self.dirichlet_conc)) # self.dirichlet_conc: K x D x 3
            
                # delta = pyro.sample("delta", dist.Dirichlet(torch.ones(3)))
                # print("Delta", delta[4,:,:])
                
                # w = pyro.sample("w", dist.RelaxedOneHotCategorical(torch.tensor([self.temperature]), probs=delta))
                
                phi_beta = pyro.sample("phi_beta", dist.Uniform(self.phi_beta_L, self.phi_beta_H)) # 0.5 because we are considering a 1:1 karyotype
                k_beta = pyro.sample("k_beta", dist.LogNormal(torch.log(self.k_beta_mean), self.k_beta_std))
                
                a_beta = self.get_a_beta(phi_beta, k_beta)
                b_beta = self.get_b_beta(phi_beta, k_beta)

                alpha = pyro.sample("alpha_pareto", dist.LogNormal(torch.log(self.alpha_pareto_mean), self.alpha_pareto_std))
                probs_pareto = pyro.sample("probs_pareto", BoundedPareto(self.pareto_L, alpha, self.pareto_H)) # probs_pareto is a K x D tensor

        # Data generation
        with pyro.plate("plate_data", len(NV)):
            # .sum() sums over the data because we have a log-likelihood
            pyro.factor("lik", self.log_sum_exp(self.m_total_lk(probs_pareto, alpha, a_beta, b_beta, weights, delta, which = 'inference')).sum())


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
        # k_beta_param = pyro.param("k_beta_param", lambda:self.init_kappas, constraint=constraints.positive)

        probs_pareto_param = pyro.param("probs_pareto_param", lambda: torch.ones((K,D))*self.probs_pareto_init, constraint=constraints.interval(self.pareto_L, self.pareto_H))
        
        # probs_pareto_param = pyro.param("probs_pareto_param", lambda: torch.ones((K,D)*self.probs_pareto_init, constraint=constraints.interval(self.pareto_L, 0.15))
        
        delta_param = pyro.param("delta_param", lambda: self.init_delta, constraint=constraints.simplex)
        # print(delta_param)
        # delta_param = pyro.param("delta_param", lambda: dist.Dirichlet(torch.ones(3)).sample([K, D]).reshape(K, D, 3), constraint=constraints.simplex)
       
        # w_param = pyro.param("w_param", lambda: self.init_delta, constraint=constraints.simplex)

        
        with pyro.plate("plate_dims", D):
            with pyro.plate("plate_probs", K):
                alpha = pyro.sample("alpha_pareto", dist.Delta(alpha_param)) # here because we need to have K x D samples
                
                pyro.sample("phi_beta", dist.Delta(phi_beta_param))
                pyro.sample("k_beta", dist.Delta(k_beta_param))

                # pyro.sample("probs_pareto", BoundedPareto(self.pareto_L, alpha, self.pareto_H))
                pyro.sample("probs_pareto", dist.Delta(probs_pareto_param))
                pyro.sample("delta", dist.Delta(delta_param).to_event(1))
                
                # pyro.sample("w", dist.Delta(w_param).to_event(1))

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
        '''
        The function adds +1 for each consecutive iteration where 
        all parameter values change by less than the specified threshold 
        compared to the previous iteration (relative difference).
        '''
        old = self.flatten_params(old_par)
        new = self.flatten_params(new_par)
        diff_mix = np.abs(old - new) / np.abs(old)
        if np.all(diff_mix < self.par_threshold):
            return check_conv + 1 
        return 0

    def get_parameters_stopping(self, params):
        par = {'phi_beta_param': params["phi_beta_param"],
               'k_beta_param': params["k_beta_param"],
                'alpha_pareto_param': params['alpha_pareto_param'],
                'delta_param': params['delta_param']} 
        return par
    
    def loss_convergence(self):
        old = self.losses[-2]
        curr = self.losses[-1]
        diff = np.abs(old - curr) / np.abs(old)
        if diff < self.loss_threshold:
            return True
        else:
            return False

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

        elbo = pyro.infer.TraceGraph_ELBO()
        svi = pyro.infer.SVI(self.model, self.guide, pyro.optim.Adam({"lr": lr}), elbo)
        min_loss = torch.tensor(float('inf'))
        best_noise = torch.zeros([self.K, self.NV.shape[1]])
        # for loop to sample different noise and see which gives the smaller loss
        self.cluster_initialization(compute_kmeans = True)
        for i in range(10):
            pyro.clear_param_store()
            mean = 0
            std_dev = 0.005
            D = self.NV.shape[1]
            self.gaussian_noise = dist.Normal(mean, std_dev).sample([self.K, D])
            # self.gaussian_noise = torch.zeros([self.K, D])
            # print("Noise: ", self.gaussian_noise)
            self.cluster_initialization(compute_kmeans = False)
            # print(self.kmeans_centers)
            loss = svi.step()
            if loss < min_loss:
                best_noise = self.gaussian_noise.clone()
        
        pyro.clear_param_store()
        self.gaussian_noise = best_noise
        self.cluster_initialization(compute_kmeans = False, final = True)
        svi.step()
        gradient_norms = defaultdict(list)
        for name, value in pyro.get_param_store().named_parameters():
            value.register_hook(
                lambda g, name=name: gradient_norms[name].append(g.norm().item())
            )
        
        self.losses = []
        self.lks = []
        self.params_stop_list = {}
        i = 0
        conv_iter = 200
        check_conv = 0
        params = self.get_parameters()
        min_iter = 200
        
        # vaf = self.NV/self.DP
        # self.central_ix = np.where((vaf[:,0] > 0.3) & (vaf[:,1] > 0.3))
        for i in range(num_iter):
            # if i >= min_iter: # posso togliere l'if e prendere qui i parametri da aggiungere alla lista di quelli da printare 
            old_par = self.get_parameters_stopping(params) # Save current values of the parameters in old_params
            for key in old_par:
                if key in self.params_stop_list:
                    self.params_stop_list[key].append(old_par[key])
                else:
                    self.params_stop_list[key] = [old_par[key]]

            loss = svi.step()
            self.losses.append(loss)
            
            params = self.get_parameters()
            
            a_beta = self.get_a_beta(params["phi_beta_param"], params["k_beta_param"])
            b_beta = self.get_b_beta(params["phi_beta_param"], params["k_beta_param"])
            probs_pareto = params["probs_pareto_param"] if "probs_pareto_param" in params.keys() else BoundedPareto(self.pareto_L, params["alpha_pareto_param"], self.pareto_H).sample()
            
            lks = self.log_sum_exp(self.m_total_lk(probs_pareto,
                                                   params["alpha_pareto_param"], a_beta, b_beta, 
                                                   params["weights_param"], params["delta_param"], which = 'inference')).sum()

            # print(params['delta_param'])
            self.lks.append(lks.detach().numpy())
            
            """"""
            if i >= min_iter:
                new_par = self.get_parameters_stopping(params)
                check_conv = self.stopping_criteria(old_par, new_par, check_conv)
                conv_loss = self.loss_convergence()
                # If convergence is reached (i.e. changes in parameters are small for min_iter iterations), stop the loop
                if check_conv == conv_iter and conv_loss == True:
                    break

            # print("Iteration {}: Loss = {}".format(i, loss))
            if i % 50 == 0:
                print("Iteration {}: Loss = {}".format(i, loss))
                """
                self.params = self.get_parameters()
                self.compute_posteriors(which = 'inference')
                # self.compute_posteriors()
                self.plot()
                plot_marginals(self)
                """

        self.params = self.get_parameters()

        self.plot_loss_lks_dist()
        self.plot_grad_norms(gradient_norms)
        
        final_lk = self.compute_posteriors()
        print("Inference lk: ", self.lks[-1])
        print("Final lk (integr): ", self.log_sum_exp(final_lk).sum())

        bic = self.compute_BIC(self.params, final_lk)
        print(f"bic: {bic} \n")
        icl = self.compute_ICL(self.params, bic)

        # self.plot()
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
        alpha = self.params["alpha_pareto_param"] 
        delta = self.params["delta_param"]  # K x D x 2
        # delta = self.params["w_param"]  # K x D x 2
        
        phi_beta = self.params["phi_beta_param"]
        k_beta = self.params["k_beta_param"]
        a_beta = self.get_a_beta(phi_beta, k_beta)
        b_beta = self.get_b_beta(phi_beta, k_beta)
        weights = self.params["weights_param"]
        if which == 'inference':
            probs_pareto = self.params["probs_pareto_param"] if "probs_pareto_param" in self.params.keys() else BoundedPareto(self.pareto_L, self.params["alpha_pareto_param"], self.pareto_H).sample()
        else:
            probs_pareto = None
        lks = self.m_total_lk(probs_pareto = probs_pareto, alpha=alpha, a_beta=a_beta, b_beta=b_beta, 
                               weights=weights, delta=delta, which = which) 

        return lks

    def compute_posteriors(self, which = 'posteriors'):
        
        lks = self.compute_final_lk(which = which) # K x N
        res = torch.zeros(self.K, len(self.NV)) # K x N
        norm_fact = self.log_sum_exp(lks) # sums over the different cluster -> array of size 1 x len(NV)
        for k in range(len(res)): # iterate over the clusters
            lks_k = lks[k] # take row k -> array of size len(NV)
            res[k] = torch.exp(lks_k - norm_fact)
        self.params["responsib"] = res # qui non dovrebbe cambiare niente per le private perchè i punti sono già sommati sulle dimensioni
        self.params["cluster_assignments"] = torch.argmax(self.params["responsib"], dim = 0) # vector of dimension
        """"""
        # Find empty components and remove them
        counts = torch.bincount(self.params["cluster_assignments"], minlength=self.K)  # Shape: (K,)
        empty_components = torch.where(counts == 0)[0]  # Indices of empty components
        

        keys = ["phi_beta_param", "k_beta_param", "alpha_pareto_param", 
                "delta_param", "weights_param", "probs_pareto_param", "w_param"]
        
        if len(empty_components) > 0:
            
            # Remove empty components from responsibilities
            mask = torch.ones(self.K, dtype=torch.bool)
            mask[empty_components] = False
            self.params["responsib"] = self.params["responsib"][mask]
            
            # Recompute cluster assignments with updated responsibilities
            self.params["cluster_assignments"] = torch.argmax(self.params["responsib"], dim=0)
            for key, _ in self.params.items():
                if key in keys and isinstance(self.params[key], torch.Tensor) and self.params[key].shape[0] == self.K:
                    self.params[key] = self.params[key][mask]  # Keep only non-empty rows

            # Update the number of components
            self.final_K = self.params["responsib"].shape[0]
            """
            self.final_K = self.K
            """
        else:
            self.final_K = self.K
        
        return lks

    def compute_euclidean_distance(self, t1, t2):
        # Flatten tensors to vectors
        t1_flat = t1.flatten()
        t2_flat = t2.flatten()
        return torch.norm(t1_flat - t2_flat).item()

    def compute_max_relative_distance(self, old, new):
        old_flat = old.flatten()
        new_flat = new.flatten()
        # Compute relative distances element-wise
        diff_mix = torch.abs(new_flat - old_flat) / (torch.abs(old_flat))
        # Return the maximum relative distance
        return torch.max(diff_mix).item()
    
    def compute_mixing_distances(self, dictionary):
        results = {}
        for key, vector in dictionary.items():
            distances = []
            distances_euc = []
            for i in range(1, len(vector)):
                # Compute the maximum relative distance for consecutive parameter vectors
                dist = self.compute_max_relative_distance(vector[i - 1], vector[i])
                distances.append(dist)
                dist_euc = self.compute_euclidean_distance(vector[i - 1], vector[i])
                distances_euc.append(dist_euc)
            # Store results for this key
            results[key] = {
                "max_relative_distances": distances,
                "euclidean_distances": distances_euc,
            }
        return results

    def plot_loss_lks_dist(self):
        # dist_pi, dist_pi_euc = self.compute_mixing_distances(self.pi_list)
        # dist_delta, dist_delta_euc = self.compute_mixing_distances(self.delta_list)
        # dist_alpha,dist_alpha_euc = self.compute_mixing_distances(self.alpha_list)
        # dist_phi,dist_phi_euc = self.compute_mixing_distances(self.phi_list)

        dist = self.compute_mixing_distances(self.params_stop_list)

        _, ax = plt.subplots(2, 2, figsize=(15, 15))
        ax[0,0].plot(self.losses)
        ax[0,0].set_title(f"Loss (K = {self.K}, seed = {self.seed})")
        ax[0,0].grid(True, color='gray', linestyle='-', linewidth=0.2)

        ax[0,1].plot(self.lks)
        ax[0,1].set_title(f"Likelihood (K = {self.K}, seed = {self.seed})")
        ax[0,1].grid(True, color='gray', linestyle='-', linewidth=0.2)

        keys = list(self.params_stop_list.keys())
        for key in keys:
            dist_rel = dist[key]["max_relative_distances"]
            dist_euc = dist[key]["euclidean_distances"]

            # Plot max relative distances
            ax[1, 0].plot(dist_rel, label=key)

            # Plot Euclidean distances
            ax[1, 1].plot(dist_euc, label=key)

        ax[1, 0].set_title("Max relative dist between consecutive iterations")
        ax[1, 0].grid(True, color='gray', linestyle='-', linewidth=0.2)
        ax[1, 0].axhline(y=self.par_threshold, color='red', linestyle='--', linewidth=0.8, label=f'Threshold')
        ax[1, 0].legend()

        ax[1, 1].set_title("Euclidean dist between consecutive iterations")
        ax[1, 1].grid(True, color='gray', linestyle='-', linewidth=0.2)
        ax[1, 1].legend()

        # ax[1,0].plot(dist_pi, label="pi")
        # ax[1,0].plot(dist_delta, label="delta")
        # ax[1,0].plot(dist_alpha, label="alpha")
        # ax[1,0].plot(dist_phi, label="phi")
        # ax[1,0].set_title("Max relative dist between consecutive iterations")  
        # ax[1,0].grid(True, color='gray', linestyle='-', linewidth=0.2)  
        # ax[1,0].legend()

        # ax[1,1].plot(dist_pi_euc, label="pi")
        # ax[1,1].plot(dist_delta_euc, label="delta")
        # ax[1,1].plot(dist_alpha_euc, label="alpha")
        # ax[1,1].plot(dist_phi_euc, label="phi")
        # ax[1,1].set_title("Euclidean dist between consecutive iterations")  
        # ax[1,1].grid(True, color='gray', linestyle='-', linewidth=0.2)  
        # ax[1,1].legend()   
        
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

    def plot_old(self):
        """
        Plot the results.
        """
        D = self.NV.shape[1]
        pairs = np.triu_indices(D, k=1)  # Generate all unique pairs of samples (i, j)
        vaf = self.NV / self.DP

        weights = self.params["weights_param"].detach().numpy()
        unique_labels = np.unique(self.params["cluster_assignments"].detach().numpy())
        labels = self.params["cluster_assignments"].detach().numpy()
        cmap = cm.get_cmap('tab20')
        color_mapping = {label: cmap(i) for i, label in enumerate(unique_labels)}
        colors = [color_mapping[label] for label in labels]

        num_pairs = len(pairs[0])  # Number of unique pairs
        # ncols = 3
        ncols = min(3, num_pairs)
        nrows = (num_pairs + ncols - 1) // ncols  # Calculate the number of rows

        fig_width_per_plot = 5
        fig_width = ncols * fig_width_per_plot
        fig_height = 5 * nrows

        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))
        if num_pairs == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        idx = 0
        for i, j in zip(*pairs):
            ax = axes[idx]  # Select the appropriate subplot
            x = vaf[:, i].numpy()
            y = vaf[:, j].numpy()

            sc = ax.scatter(x, y, c=colors,s=15)  # tab20
            handles = [plt.Line2D([0], [0], marker='o', color='w', label=f"{label} (\u03C0 = {weights[label]:.3f})",
                                markerfacecolor=color_mapping[label], markersize=10) 
                    for label in unique_labels]
            ax.legend(handles=handles)

            ax.set_title(f"Sample {i+1} vs Sample {j+1} (K = {self.K}, seed {self.seed})")
            ax.set_xlabel(f"Sample {i+1}")
            ax.set_ylabel(f"Sample {j+1}")
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])

            idx += 1

        # Hide any unused subplots (if there are fewer than nrows * ncols pairs)
        for i in range(idx, len(axes)):
            axes[i].axis('off')

        # Adjust layout to avoid overlap
        plt.tight_layout()

        if self.savefig:
            plt.savefig(f"plots/{self.data_folder}/inference_K_{self.K}_seed_{self.seed}.png")

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
    