import numpy as np
import pyro
import pyro.distributions as dist

import torch
from torch.distributions import constraints

import matplotlib.pyplot as plt
from scipy.stats import binom, beta, pareto
from sklearn.cluster import KMeans
from BoundedPareto import BoundedPareto

class mobster_MV():
    def __init__(self, NV = None, DP = None, K=1, tail=1, truncated_pareto = True, purity=1, seed=2):
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

        # if(NV is not None):
        self.NV = torch.tensor(NV) if not isinstance(NV, torch.Tensor) else NV
        # if(DP is not None):
        self.DP = torch.tensor(DP) if not isinstance(DP, torch.Tensor) else DP
        self.K = K
        self.tail = tail
        self.truncated_pareto = truncated_pareto
        self.purity = purity
        self.seed = seed
    
    def compute_kmeans_centers(self, seed):
        kmeans = KMeans(n_clusters=self.K, random_state=seed, n_init="auto").fit((self.NV/self.DP).numpy())
        cluster = kmeans.labels_
        centers = torch.tensor(kmeans.cluster_centers_)

        # mean = 0
        # std_dev = 0.05
        # D = self.NV.shape[1]
        # gaussian_noise = dist.Normal(mean, std_dev).sample([self.K, D])

        # # Add gaussian noise to found centers
        # centers = centers + gaussian_noise

        # # Clip probabilities in [0, 1]
        # # self.kmeans_centers = torch.clip(centers, 0, 1)
        # centers[centers <= 0] = 0.01
        # centers[centers >= 1] = 0.09
        self.kmeans_centers = centers
        # print("ckmeans_centers: ", self.kmeans_centers)

    def m_binomial_lk(self, probs, DP, weights, K, NV):
        """
        Compute multidimensional binomial likelihood.
        This function returns a K x N matrix. Each entry contains the sum between the log-weight of the cluster and the log-prob of the data.
        Assume independence between the D samples (i.e., sums the contributions):
            log(pi_k) + sum^D log(Bin(NV_i | p_k, DP_i))
        """
        lk = torch.ones(K, len(NV)) # matrix with K rows and as many columns as the number of data
        if K == 1:
            return torch.log(weights) + dist.Binomial(total_count=DP, probs = probs).log_prob(NV).sum(axis=1) # simply does log(weights) + log(density)
        for k in range(K):
            lk[k, :] = torch.log(weights[k]) + dist.Binomial(total_count=DP, probs=probs[k, :]).log_prob(NV).sum(axis=1) # sums over the data dimensions (columns)
                                                                                                                    # put on each column of lk a different data; rows are the clusters
        return lk
    
    def log_sum_exp(self, args):
        """
        Compute the log-sum-exp for each data point, i.e. the log-likelihood for each data point.
        log(p(x_i | theta)) = log(exp(a_1), ..., exp(a_K))
        where: a_k = log(pi_k) + sum^D log(Bin(x_{id} | DP_{id}, p_{dk})) 
        This function returns a N dimensional vector, where each entry corresponds to the log-likelihood of each data point.
        """
        c = torch.amax(args, dim=0)
        return c + torch.log(torch.sum(torch.exp(args - c), axis=0)) # sum over the rows (different clusters), so obtain a single likelihood for each data

    def log_beta_par_mix(self, probs_beta, probs_pareto, delta, alpha, a_beta, b_beta):
        # relaxed one hot
        delta_pareto = torch.log(delta[:, 0]) + BoundedPareto(0.001, alpha, 1).log_prob(probs_pareto) + dist.Binomial(total_count=self.DP, probs = probs_pareto).log_prob(self.NV)  # 1x2 tensor
        delta_beta = torch.log(delta[:, 1]) + dist.Beta(a_beta, b_beta).log_prob(probs_beta) + dist.Binomial(total_count=self.DP, probs = probs_beta).log_prob(self.NV) # 1x2 tensor
        
        return self.log_sum_exp(torch.stack((delta_pareto, delta_beta), dim=0)) # creates a 2x2 tensor with torch.stack because log_sum_exp has dim=0

    def m_total_lk(self, probs_beta, probs_pareto, alpha, a_beta, b_beta, weights, delta):
        lk = torch.ones(self.K, len(self.NV)) # matrix with K rows and as many columns as the number of data
        if self.K == 1:
            return torch.log(weights) + self.log_beta_par_mix(probs_beta, probs_pareto, delta[0, :, :], alpha, a_beta, b_beta).sum(axis=1) # simply does log(weights) + log(density)
        for k in range(self.K):
            # print("k: ", k)
            # print("a_beta: ", a_beta)
            # print("b_beta: ", b_beta)
            # print(alpha)
            lk[k, :] = torch.log(weights[k]) + self.log_beta_par_mix(probs_beta[k, :], probs_pareto[k, :], delta[k, :, :], alpha[k, :], a_beta[k, :], b_beta[k, :]).sum(axis=1) # sums over the data dimensions (columns)
                                                                                                                    # put on each column of lk a different data; rows are the clusters                                                                                             
        
        return lk

    
    def model(self):
        """
        Define the model.
        """
        NV, DP = self.NV, self.DP
        K = self.K
        D = NV.shape[1] # number of dimensions (samples)

        # Prior for the mixing weights
        weights = pyro.sample("weights", dist.Dirichlet(torch.ones(K)))

        with pyro.plate("plate_dims", D):
            with pyro.plate("plate_probs", K):
                # Prior for the Beta-Pareto weights
                delta = pyro.sample("delta", dist.Dirichlet(torch.ones(2))) # delta is a K x D x 2 torch tensor (K: num layers, D: rows per layer, 2: columns per layer)
                
                # a_beta = pyro.sample("a_beta", dist.Gamma(0.5, 0.5))
                phi_beta = pyro.sample("phi_beta", dist.Beta(1, 1))
                k_beta = pyro.sample("k_beta", dist.Normal(100, 0.5))
                a_beta = phi_beta*k_beta
                b_beta = (1-phi_beta)*k_beta


                # assume Beta prior for the success probabilities
                probs_beta = pyro.sample("probs_beta", dist.Beta(a_beta, b_beta)) # probs_beta is a K x D tensor
                # print("probs_beta: ", probs_beta)
            
                alpha = pyro.sample("alpha_pareto", dist.LogNormal(0, 100)) # alpha is a K x D tensor
                probs_pareto = pyro.sample("probs_pareto", BoundedPareto(0.001, alpha, 1)) # probs_pareto is a K x D tensor

        # Data generation
        with pyro.plate("plate_data", len(NV)):
            pyro.factor("lik", self.log_sum_exp(self.m_total_lk(probs_beta, probs_pareto, alpha, a_beta, b_beta, weights, delta)).sum()) # .sum() sums over the data because we have a log-likelihood  

    def guide(self):
        """
        Define the guide for the model.
        """
        NV, DP = self.NV, self.DP
        K = self.K
        D = NV.shape[1] # number of dimensions (samples)
        weights_param = pyro.param("weights_param", lambda: dist.Dirichlet(torch.ones(K)).sample(), constraint=constraints.simplex)
        pyro.sample("weights", dist.Delta(weights_param).to_event(1))

        delta_param = pyro.param("delta_param", lambda: dist.Dirichlet(torch.ones(K, D, 2)).sample(), constraint=constraints.simplex)
    
        alpha_param = pyro.param("alpha_param", torch.ones((K,D)), constraint=constraints.positive) # Use 0.8 as starting value
        # a_beta_param = pyro.param("a_beta_param", torch.ones((K,D))*5, constraint=constraints.greater_than(lower_bound=1.0))
        phi_beta_param = pyro.param("phi_beta_param", self.kmeans_centers, constraint=constraints.interval(0.2, 1.))
        k_beta_param = pyro.param("k_beta_param", torch.ones((K,D))*100, constraint=constraints.positive)


        probs_beta_param = pyro.param("probs_beta_param", self.kmeans_centers, constraint=constraints.interval(0.,1.))
        probs_pareto_param = pyro.param("probs_pareto_param", self.kmeans_centers, constraint=constraints.interval(0.,1.))
        
        print("Beta: ", probs_beta_param)
        print("Pareto: ", probs_pareto_param)
        with pyro.plate("plate_dims", D):
            with pyro.plate("plate_probs", K):
                pyro.sample("alpha_pareto", dist.Delta(alpha_param)) # here because we need to have K x D samples

                # pyro.sample("a_beta", dist.Delta(a_beta_param))
                pyro.sample("phi_beta", dist.Delta(phi_beta_param))
                pyro.sample("k_beta", dist.Delta(k_beta_param))
                
                pyro.sample("probs_beta", dist.Delta(probs_beta_param))
                pyro.sample("probs_pareto", dist.Delta(probs_pareto_param))
                pyro.sample("delta", dist.Delta(delta_param).to_event(1)) # not sure
        

    def get_parameters(self):
        """
        Extract the learned parameters.
        """
        param_store = pyro.get_param_store()
        params = {}
        params["probs_beta"] = param_store["probs_beta_param"].clone().detach()
        params["probs_pareto"] = param_store["probs_pareto_param"].clone().detach()
        params["weights"] = param_store["weights_param"].clone().detach()
        params["delta"] = param_store["delta_param"].clone().detach()
        params["alpha_pareto"] = param_store["alpha_param"].clone().detach()
        params["phi_beta"] = param_store["phi_beta_param"].clone().detach()
        params["k_beta"] = param_store["k_beta_param"].clone().detach()
        # params["a_beta"] = param_store["a_beta_param"].clone().detach()
        # params["b_beta"] = param_store["b_beta_param"].clone().detach()

        return params


    def fit(self, num_iter = 2000, lr = 0.001):
        pyro.clear_param_store()
        pyro.set_rng_seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        NV, DP = self.NV, self.DP
        K = self.K
        self.compute_kmeans_centers(self.seed)
        svi = pyro.infer.SVI(self.model, self.guide, pyro.optim.Adam({"lr": lr}), pyro.infer.TraceGraph_ELBO())
        for i in range(num_iter):
            loss = svi.step()
            if i % 200 == 0:
                print("Iteration {}: Loss = {}".format(i, loss))
        
        self.params = self.get_parameters()
        self.compute_posteriors()
        self.plot()
    
    def compute_posteriors(self):
        """
        Compute posterior assignment probabilities (i.e., the responsibilities) given the learned parameters.
        """
        # lks : K x N
        lks = self.m_binomial_lk(probs=self.params['probs_beta'], DP = self.DP, weights=self.params['weights'], K = self.K, NV = self.NV) # Compute log-likelihood for each data in each cluster
        # res : K x N
        res = torch.zeros(self.K, len(self.NV))
        norm_fact = self.log_sum_exp(lks) # sums over the different cluster -> array of size len(NV)
        for k in range(len(res)): # iterate over the clusters
            lks_k = lks[k] # take row k -> array of size len(NV)
            res[k] = torch.exp(lks_k - norm_fact)
        self.params["responsib"] = res
        # For each data point find the cluster assignment (si può creare anche una funzione a parte)
        self.params["cluster_assignments"] = torch.argmax(self.params["responsib"], dim = 0) # vector of dimension

        
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
        sc = plt.scatter(NV_S1/DP_S1, NV_S2/DP_S2, c = self.params["cluster_assignments"], label = "Original data")
        plt.scatter(self.params['probs_beta'][:, 0], self.params['probs_beta'][:, 1], c = 'r')
        plt.legend(*sc.legend_elements())
        plt.show()
        
    

