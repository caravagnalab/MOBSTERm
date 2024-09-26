import numpy as np
import pyro
import pyro.distributions as dist

import torch
import torch.nn as nn
from torch.distributions import constraints

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import beta, pareto
from scipy.integrate import simpson
from sklearn.cluster import KMeans
from utils.BoundedPareto import BoundedPareto

from collections import defaultdict
from pandas.core.common import flatten
import copy

def fit(NV = None, DP = None, num_iter = 2000, K = [], tail=1, truncated_pareto = True, 
        purity=1, seed=[1,2,3], lr = 0.001):
    """
    Function to run the inference with different values of K
    """
    min_bic = torch.tensor(float('inf'))
    min_bic_seed = torch.tensor(float('inf'))
    mb_list = []
    curr_mb = []
    best_seed = 0
    j = 0
    for curr_k in K:
        if curr_k != 0:
            # Fai ciclo con 4/5 seed diversi e prendi bic minore
            for curr_seed in seed:
                print(f"RUN WITH K = {curr_k} AND SEED = {curr_seed}")
                curr_mb.append(mobster_MV(NV, DP, K = curr_k, seed = curr_seed))
                curr_mb[j].run_inference(num_iter, lr)
                if curr_mb[j].final_dict['bic'] <= min_bic_seed:
                    min_bic_seed = curr_mb[j].final_dict['bic']
                    mb_best_seed = curr_mb[j]
                    best_seed = curr_seed
                j+=1
            mb_list.append(mb_best_seed)
    i = 0
    final_index = 0
    for obj in mb_list:
        if obj.final_dict['bic'] <= min_bic:
            final_index = i
    final_k = K[final_index]
    print(f"Selected number of clusters is {final_k} with seed {best_seed}")
    final_mb = mb_list[final_index]
    final_mb.plot()
    return final_mb, mb_list

class mobster_MV():
    def __init__(self, NV = None, DP = None, K = 1, tail=1, truncated_pareto = True, purity=1, seed=2):
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

        self.NV = torch.tensor(NV) if not isinstance(NV, torch.Tensor) else NV
        self.DP = torch.tensor(DP) if not isinstance(DP, torch.Tensor) else DP
        self.K = K
        self.tail = tail
        self.truncated_pareto = truncated_pareto
        self.purity = purity
        self.seed = seed

        pyro.clear_param_store()
        pyro.set_rng_seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.set_prior_parameters()
        print("NV = 0 before:", torch.sum(self.NV == 0))
        # self.zero_NV_idx = (self.NV/self.DP < 0.01)
        # self.zero_NV_idx = (self.NV == 0.)
        # self.NV[self.zero_NV_idx] = torch.where(torch.round(DP[self.zero_NV_idx] * 0.01).to(NV.dtype) < 1, 
        #                                    torch.tensor(1, dtype=NV.dtype), torch.round(DP[self.zero_NV_idx] * 0.01).to(NV.dtype))
        print("NV = 0 after:", torch.sum(self.NV == 0))
        # print("VAF zero_idx: ", self.NV[self.zero_NV_idx]/self.DP[self.zero_NV_idx])

    
    def compute_kmeans_centers(self):
        best_inertia = float('inf')
        best_centers = None
        best_labels = None
        # Implement loop to choose the seed which produces a result with the lowest inertia
        
        for seed in range(1, 16):
            kmeans = KMeans(n_clusters=self.K, random_state=seed, n_init=2).fit((self.NV/self.DP).numpy())
            # best_cluster = kmeans.labels_.copy()
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
        self.kmeans_centers_no_noise[self.kmeans_centers_no_noise <= 0] = torch.min(self.min_vaf)
        self.kmeans_centers_no_noise[self.kmeans_centers_no_noise >= 1] = 0.999
        
        mean = 0
        std_dev = 0.005
        D = self.NV.shape[1]
        gaussian_noise = dist.Normal(mean, std_dev).sample([self.K, D])

        # Add gaussian noise to found centers
        best_centers = best_centers + gaussian_noise  
        # -----------------Gaussian noise------------------#
        
        # Clip probabilities in [min_vaf, 0.999]
        best_centers[best_centers <= 0] = torch.min(self.phi_beta_L)
        # best_centers[best_centers <= 0] = 1 - 0.999
        best_centers[best_centers >= 1] = 0.999
        self.kmeans_centers = best_centers

        cluster_sizes = np.bincount(best_labels.astype(int), minlength=self.K)
        weights_kmeans = cluster_sizes.reshape(-1)/np.sum(cluster_sizes.reshape(-1))
        self.init_weights = torch.tensor(weights_kmeans)
        # print("self.init_weights: ", self.init_weights)

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
        # Print kmeans result
        sc = plt.scatter(self.NV[:,0]/self.DP[:,0], self.NV[:,1]/self.DP[:,1], c = best_cluster)
        plt.xlim([0,1])
        plt.ylim([0,1])
        print("kmeans_centers: ", self.kmeans_centers)
        """
    
    def initialize_delta(self, phi_beta, k_beta, alpha):
        a_beta = phi_beta * k_beta
        b_beta = (1-phi_beta) * k_beta
        beta_lk = dist.Beta(a_beta, b_beta).log_prob(self.kmeans_centers_no_noise)
        # Note that I had to put 1 as upper bound of BoundedPareto because kmeans centers can also be bigger than 0.5 (due to my clip)
        # Otherwise the likelihood is infinite
        pareto_lk = BoundedPareto(self.pareto_L, alpha, 1).log_prob(self.kmeans_centers_no_noise)
        # print("Beta: ", beta_lk)
        # print("Pareto: ", pareto_lk)
        # kmeans_centers: KxD
        K = self.K
        D = self.NV.shape[1]
        init_delta = torch.zeros((K,D,2))
        for i in range(K):
            for j in range(D):
                if(beta_lk[i,j] > pareto_lk[i,j]):
                    init_delta[i,j,0] = 0.4 # pareto
                    init_delta[i,j,1] = 0.6 # beta
                else:
                    init_delta[i,j,0] = 0.6 # pareto
                    init_delta[i,j,1] = 0.4 # beta
        # print("init_delta: ", init_delta)
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

    def pareto_lk(self, d, alpha):
        LINSPACE = 10000
        x = torch.linspace(self.pareto_L, self.pareto_H, LINSPACE)
        y_1 = BoundedPareto(self.pareto_L, alpha, self.pareto_H).log_prob(x).exp()
        y_2 = dist.Binomial(probs = x.repeat([self.NV.shape[0], 1]).reshape([LINSPACE,-1]), total_count=self.DP[:,d]).log_prob(self.NV[:,d]).exp()
        paretobin = torch.trapz(y_1.reshape([LINSPACE, 1]) * y_2, x =  x, dim = 0).log()
        
        # p = BoundedPareto(self.pareto_L, alpha, self.pareto_H).sample()
        # paretobin2 = dist.Binomial(probs=p, total_count=self.DP[:,d]).log_prob(self.NV[:,d])
        return paretobin # tensor of len N (if D = 1, only N)
    
    def log_beta_par_mix_inference(self, probs_pareto, delta, alpha, a_beta, b_beta):
        # delta -> D x 2
        delta_pareto = torch.log(delta[:, 0]) + dist.Binomial(total_count=self.DP, probs = probs_pareto).log_prob(self.NV)  # NxD tensor
        delta_beta = torch.log(delta[:, 1]) + dist.BetaBinomial(a_beta, b_beta, total_count=self.DP).log_prob(self.NV) # N x D tensor
        
        # Stack ctreates a 2 x N x D tensor to apply log sum exp on first dimension => it sums the delta_pareto and delta_beta 
        return self.log_sum_exp(torch.stack((delta_pareto, delta_beta), dim=0)) # N
    
    def log_beta_par_mix_posteriors(self, delta, alpha, a_beta, b_beta):
        # delta -> D x 2
        # delta_pareto = torch.log(delta[:, 0]) + self.pareto_lk(alpha)   # NxD tensor
        # delta_beta = torch.log(delta[:, 1]) + self.beta_lk(a_beta, b_beta) # N x D tensor
        
        # return self.log_sum_exp(torch.stack((delta_pareto, delta_beta), dim=0)) # N x D
        log_lik = []
        for d in range(delta.shape[0]): # range over samples
            if (delta[d, 0] > delta[d, 1]): # pareto
                # log_lik.append(torch.log(delta[d, 0]) + self.pareto_lk(d, alpha[d])) # N tensor
                log_lik.append(self.pareto_lk(d, alpha[d])) # N tensor
            else:
                # log_lik.append(torch.log(delta[d, 1]) + self.beta_lk(d, a_beta[d], b_beta[d])) # N tensor
                log_lik.append(self.beta_lk(d, a_beta[d], b_beta[d])) # N tensor
        
        stack_tensors = torch.stack((log_lik[0], log_lik[1]), dim=1) # combine together the tensors of dim 0 and 1 to create a N x D tensor
        # Non mi serve log sum exp perchè non devo più sommare sui delta_j
        # print("stack_tensors", stack_tensors.shape)
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
        return min_values

    def set_prior_parameters(self):

        self.max_vaf = torch.tensor(0.55) # for a 1:1 karyotype
        self.min_vaf = torch.tensor(0.01)

        # phi_beta
        self.phi_beta_L = torch.tensor(0.1)
        # self.phi_beta_L = torch.tensor(0.01)
        self.phi_beta_H = self.max_vaf

        # k_beta
        self.k_beta_mean = torch.tensor(200.)
        self.k_beta_init = torch.tensor(200.)
        # self.k_beta_mean = torch.tensor(self.init_kappas)
        # self.k_beta_init = torch.tensor(self.init_kappas)
        self.k_beta_std = torch.tensor(0.01)

        # alpha_pareto
        self.alpha_pareto_mean = torch.tensor(1.1)
        self.alpha_pareto_std = torch.tensor(0.005)
        self.alpha_factor = torch.tensor(2.)
        self.alpha_pareto_init = torch.tensor(1.1)
        self.min_alpha = torch.tensor(0.5)
        self.max_alpha = torch.tensor(4.)

        # Bounded pareto
        # self.pareto_L = torch.tensor(0.01)
        self.pareto_L = torch.tensor(0.005)
        # self.pareto_L = self.compute_min_vaf()
        self.pareto_H = self.max_vaf

        self.temperature = 0.3


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
                delta = pyro.sample("delta", dist.Dirichlet(torch.ones(2)))
                # delta = torch.softmax(delta, dim=-1)
                # delta = delta / delta.sum(dim=-1, keepdim=True)
                
                # w = pyro.sample("w", dist.OneHotCategorical(probs = delta))
                w = pyro.sample("w", dist.RelaxedOneHotCategorical(torch.tensor([self.temperature]), probs=delta))
                
                phi_beta = pyro.sample("phi_beta", dist.Uniform(self.phi_beta_L, self.phi_beta_H)) # 0.5 because we are considering a 1:1 karyotype
                k_beta = pyro.sample("k_beta", dist.LogNormal(torch.log(self.k_beta_mean), self.k_beta_std))
                # k_beta = pyro.sample("k_beta", dist.LogNormal(torch.log(self.init_kappas), self.k_beta_std))
                
                a_beta = phi_beta * k_beta
                b_beta = (1-phi_beta) * k_beta

                alpha = pyro.sample("alpha_pareto", dist.LogNormal(torch.log(self.alpha_pareto_mean), self.alpha_pareto_std))
                probs_pareto = pyro.sample("probs_pareto", BoundedPareto(self.pareto_L, alpha, self.pareto_H)) # probs_pareto is a K x D tensor
        
        # Data generation
        with pyro.plate("plate_data", len(NV)):
            # .sum() sums over the data because we have a log-likelihood
            pyro.factor("lik", self.log_sum_exp(self.m_total_lk(probs_pareto, self.alpha_factor*alpha, a_beta, b_beta, weights, w, which = 'inference')).sum())


    def guide(self):
        """
        Define the guide for the model.
        """
        NV = self.NV
        K = self.K
        D = NV.shape[1] # number of dimensions (samples)
        # weights_param = pyro.param("weights_param", lambda: dist.Dirichlet(torch.ones(K)).sample(), constraint=constraints.simplex)
        weights_param = pyro.param("weights_param", lambda: self.init_weights, constraint=constraints.simplex)
        pyro.sample("weights", dist.Delta(weights_param).to_event(1))

        alpha_param = pyro.param("alpha_pareto_param", lambda: torch.ones((K,D))*self.alpha_pareto_init, constraint=constraints.interval(self.min_alpha, self.max_alpha))        

        phi_beta_param = pyro.param("phi_beta_param", lambda: self.kmeans_centers, constraint=constraints.interval(self.phi_beta_L, self.phi_beta_H))
        k_beta_param = pyro.param("k_beta_param", lambda: torch.ones((K,D))*self.k_beta_init, constraint=constraints.greater_than(150.))
        # k_beta_param = pyro.param("k_beta_param", lambda: torch.ones((K,D))*self.k_beta_init, constraint=constraints.positive)
        # k_beta_param = pyro.param("k_beta_param", lambda: self.init_kappas, constraint=constraints.positive)

        delta_param = pyro.param("delta_param", lambda: self.init_delta, constraint=constraints.simplex)
        # delta_param = torch.softmax(delta_param, dim=-1)
        
        # delta_param = pyro.param("delta_param", lambda: dist.Dirichlet(torch.ones(2)).sample([K, D]).reshape(K, D, 2), constraint=constraints.simplex)
        # delta_param = delta_param / delta_param.sum(dim=-1, keepdim=True)
        # print("delta_param", delta_param)

        with pyro.plate("plate_dims", D):
            with pyro.plate("plate_probs", K):

                alpha = pyro.sample("alpha_pareto", dist.Delta(alpha_param)) # here because we need to have K x D samples
                
                pyro.sample("phi_beta", dist.Delta(phi_beta_param))
                pyro.sample("k_beta", dist.Delta(k_beta_param))

                pyro.sample("probs_pareto", BoundedPareto(self.pareto_L, alpha, self.pareto_H))
                pyro.sample("delta", dist.Delta(delta_param).to_event(1))

                # pyro.sample("w", dist.OneHotCategorical(probs = delta_param))
                pyro.sample("w", dist.RelaxedOneHotCategorical(torch.tensor([self.temperature]), probs = delta_param))


    def get_parameters(self):
        """
        Extract the learned parameters.
        """
        param_names = pyro.get_param_store()
        params = {nms: pyro.param(nms) for nms in param_names}
        
        return params
    
    
    def flatten_params(self, pars):
        # pars = list(pars.values().detach().tolist())
        pars = list(flatten([value.detach().tolist() for key, value in pars.items()]))
        # print("pars: ", pars)
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
        n_params = calculate_number_of_params(params)
        n = self.NV.shape[0]
        # lk = self.log_sum_exp(self.compute_final_lk()).sum()
        lk = self.log_sum_exp(final_lk).sum()
        return np.log(n) * n_params - 2 * lk

    def run_inference(self, num_iter = 2000, lr = 0.001):
        pyro.clear_param_store()
        pyro.set_rng_seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.cluster_initialization()
        # NV, DP = self.NV, self.DP
        svi = pyro.infer.SVI(self.model, self.guide, pyro.optim.Adam({"lr": lr}), pyro.infer.TraceGraph_ELBO())
        
        svi.step()
        gradient_norms = defaultdict(list)
        for name, value in pyro.get_param_store().named_parameters():
            if name in ["alpha_pareto_param", "delta_param", "weights_param", "w_param"]:
                value.register_hook(
                    lambda g, name=name: gradient_norms[name].append(g.norm().item())
                )
        
        self.losses = []
        self.lks = []
        i = 0
        min_iter = 100
        check_conv = 0
        old_par = self.get_parameters() # Save current values of the parameters in old_params
        old_par.pop('weights_param')

        initial_temperature = 1.0
        final_temperature = 0.01
        decr_rate = 0.99 

        for i in range(num_iter):
            # curr_temperature = max(final_temperature, initial_temperature * (decr_rate ** i))
            # self.temperature = torch.tensor([curr_temperature])
            
            loss = svi.step()
            self.losses.append(loss)

            # Save likelihood values
            params = self.get_parameters()
            if i == 0:
                print("delta_param", params["delta_param"])
            
            a_beta = params["phi_beta_param"] * params["k_beta_param"]
            b_beta = (1-params["phi_beta_param"]) * params["k_beta_param"]
            probs_pareto = BoundedPareto(self.pareto_L, params["alpha_pareto_param"], self.pareto_H).sample()
            lks = self.log_sum_exp(self.m_total_lk(probs_pareto,
                                                   params["alpha_pareto_param"], a_beta, b_beta, 
                                                   params["weights_param"], params["delta_param"], which = 'inference')).sum()
            self.lks.append(lks.detach().numpy())

            new_par = params.copy()
            new_par.pop('weights_param')
            check_conv = self.stopping_criteria(old_par, new_par, check_conv)
            
            # If convergence is reached (i.e. changes in parameters are small for min_iter iterations), stop the loop
            # if check_conv == min_iter:
            #     break
            if i % 200 == 0:
                print("Iteration {}: Loss = {}".format(i, loss))
            
        self.params = self.get_parameters()
        
        self.plot_loss_lks()
        self.plot_grad_norms(gradient_norms)
        
            # for i in range(10):
        # pyro.set_rng_seed(self.seed+i)
        # torch.manual_seed(self.seed+i)
        # np.random.seed(self.seed+i)
        final_lk = self.compute_posteriors()
        bic = self.compute_BIC(self.params, final_lk)
        print("bic: ", bic)
        self.plot()
        self.final_dict = {
        "model_parameters" : self.params,
        "bic": bic,
        "final_likelihood": self.lks[-1],
        "final_loss": self.losses[-1]
        }
    
    def compute_final_lk(self):
        """
        Compute likelihood with learnt parameters
        """
        alpha = self.params["alpha_pareto_param"] * self.alpha_factor
        delta = self.params["delta_param"]  # K x D x 2
        
        phi_beta = self.params["phi_beta_param"]
        k_beta = self.params["k_beta_param"]
        a_beta = phi_beta * k_beta
        b_beta = (1-phi_beta) * k_beta
        weights = self.params["weights_param"]
        
        lks = self.m_total_lk(probs_pareto = None, alpha=alpha, a_beta=a_beta, b_beta=b_beta, 
                               weights=weights, delta=delta, which = 'posteriors') 

        return lks

    def compute_posteriors(self):
        # self.NV[self.zero_NV_idx] = torch.tensor(0, dtype=self.NV.dtype)
        lks = self.compute_final_lk() # K x N
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
        ax[0].set_title("Loss")
        ax[1].plot(self.lks)
        ax[1].set_title("Likelihood")
        plt.savefig(f"plots/likelihood_K_{self.K}_seed_{self.seed}.png")
        plt.close()

    def plot_grad_norms(self, gradient_norms):
        plt.figure(figsize=(10, 4), dpi=100).set_facecolor("white")
        for name, grad_norms in gradient_norms.items():
            plt.plot(grad_norms, label=name)
        plt.xlabel("iters")
        plt.ylabel("gradient norm")
        plt.yscale("log")
        plt.legend(loc="best")
        plt.title("Gradient norms during SVI")
        plt.savefig(f"plots/gradient_norms_K_{self.K}_seed_{self.seed}.png")
        plt.close()
        
    def plot(self):
        """
        PLOT I HAVE AT THE MOMENT
        Plot the results.
        """
        NV_S1 = self.NV[:,0]
        NV_S2 = self.NV[:,1]

        DP_S1 = self.DP[:,0]
        DP_S2 = self.DP[:,1]
        plt.xlim([0,1])
        plt.ylim([0,1])
        sc = plt.scatter(NV_S1/DP_S1, NV_S2/DP_S2, c = self.params["cluster_assignments"])
        legend1 = plt.legend(*sc.legend_elements(), loc="lower right")

        plt.title(f"Final inference with K = {self.K} and seed {self.seed}")
        plt.gca().add_artist(legend1)
        plt.xlabel('Set7_55')
        plt.ylabel('Set7_57')
        plt.savefig(f"plots/inference_K_{self.K}_seed_{self.seed}.png")
        plt.close()
        
def calculate_number_of_params(params):
    total_params = 0
    for key, param in params.items():
        param_size = np.prod(param.shape)  # Calculate the total number of elements
        total_params += param_size
    return total_params
    