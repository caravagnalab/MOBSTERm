import numpy as np
import pyro
import pyro.distributions as dist

import torch
from torch.distributions import constraints

import matplotlib.pyplot as plt
from scipy.stats import binom, beta, pareto

class mobster_MV():
    def __init__(self, NV, DP, K=1, tail=1, truncated_pareto = True, purity=1):
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

        self.NV = np.array(NV)
        self.DP = np.array(DP)
        self.K = K
        self.tail = tail
        self.truncated_pareto = truncated_pareto
        self.purity = purity
    

    def m_binomial_lk(self, probs, DP, weights, K, NV):
        """
        Compute multidimensional binomial likelihood.
        """
        lk = torch.ones(K, len(NV)) # matrix with K rows and as many columns as the number of data
        if K == 1:
            return torch.log(weights) + dist.Binomial(total_count=DP, probs = probs).log_prob(NV).sum(axis=1) # simply does log(weights) + log(density)
        for k in range(K):
            lk[k, :] = torch.log(weights[k]) + dist.Binomial(total_count=DP, probs=probs[k, :]).log_prob(NV).sum(axis=1) # sums over the data dimensions (columns)
                                                                                                                    # put on each column of lk a different data; rows are the clusters
        return lk

    def log_sum_exp(self, args):
        c = torch.amax(args, dim=0)
        return c + torch.log(torch.sum(torch.exp(args - c), axis=0)) # sum over the rows (different clusters), so obtain a single likelihood for each data

    def model(self):
        """
        Define the model.
        """
        NV, DP = self.NV, self.DP
        K = self.K
        D = NV.shape[1] # number of dimensions (samples)

        # Prior for the mixing weights
        weights = pyro.sample("weights", dist.Dirichlet(torch.ones(K)))

        # Prior for success probabilities (each probability has 2 dimensions) for each component
        with pyro.plate("plate_probs", K):
            probs = pyro.sample("probs", dist.Beta(1, 1).expand([D]).to_event(1)) # assume Beta prior for the success probabilities
            # probs = pyro.sample("probs", dist.Beta(torch.ones(K, d), torch.ones(K, d)))

        # Data generation
        with pyro.plate("plate_data", len(NV)):
            pyro.factor("lik", self.log_sum_exp(self.m_binomial_lk(probs, DP, weights, K, NV)).sum()) # .sum() sums over the data because we have a log-likelihood


    def guide(self):
        """
        Define the guide for the model.
        """
        NV, DP = self.NV, self.DP
        K = self.K
        D = NV.shape[1] # number of dimensions (samples)

        weights_param = pyro.param("weights_param", lambda: dist.Dirichlet(torch.ones(K)).sample(), constraint=constraints.simplex)
        pyro.sample("weights", dist.Delta(weights_param).to_event(1))

        probs_param = pyro.param("probs_param", dist.Beta(torch.ones(K, D), torch.ones(K,D)).sample(), constraint=constraints.interval(0.,1.))
        print(probs_param)
        
        # Probability of success for each component
        with pyro.plate("plate_probs", K):
            pyro.sample("probs", dist.Delta(probs_param).to_event(1))   

    def get_parameters(self):
        """
        Extract the learned parameters.
        """
        param_store = pyro.get_param_store()
        params = {}
        params["probs_bin "] = param_store["probs_param"].clone().detach()
        params["weights"] = param_store["weights_param"].clone().detach()

        return params


    def fit(self, num_iter = 2000, lr = 0.001):
        pyro.clear_param_store()
        NV, DP = self.NV, self.DP
        K = self.K
        svi = pyro.infer.SVI(self.model(), self.guide(), pyro.optim.Adam({"lr": lr}), pyro.infer.TraceGraph_ELBO())
        for i in range(num_iter):
            loss = svi.step(NV, DP, K=K)
            if i % 100 == 0:
                print("Iteration {}: Loss = {}".format(i, loss))
        
        self.params = self.get_parameters()

    def plot(self):
        """
        PLOT I HAVE AT THE MOMENT 
        Plot the results.
        """
        NV_S1 = self.NV[:,0]
        NV_S2 = self.NV[:,1]

        DP_S1 = self.DP[:,0]
        DP_S2 = self.DP[:,1]
        plt.scatter(NV_S1/DP_S1, NV_S2/DP_S2, c = 'b', label = "Original data")

        # Plot samples from fitted densities
        x = np.linspace(0, 150, 1000)
        n = 130

        f1 = torch.ones([1000, 2])
        f1[:,0] = dist.Binomial(total_count=n, probs=self.params["probs_bin"][0,0]).sample([1000]).squeeze(-1)
        f1[:,1] = dist.Binomial(total_count=n, probs=self.params["probs_bin"][0,1]).sample([1000]).squeeze(-1)
        plt.scatter(f1[:,0]/n, f1[:,1]/n, c = 'r', label = "Samples from fitted beta components")

        f2 = torch.ones([1000, 2])
        f2[:,0] = dist.Binomial(total_count=n, probs=self.params["probs_bin"][1,0]).sample([1000]).squeeze(-1)
        f2[:,1] = dist.Binomial(total_count=n, probs=self.params["probs_bin"][1,1]).sample([1000]).squeeze(-1)
        plt.scatter(f2[:,0]/n, f2[:,1]/n, c = 'r')

        plt.title('2D binomial mixture model')
        plt.xlabel('VAF')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()
        
    def compute_posteriors(self):
        """
        Compute posterior assignment probabilities (i.e., the responsibilities) given the learned parameters.
        """
        # lks : K x N 
        lks = self.m_binomial_lk(probs=self.params['probs_bin'], DP = self.DP, weights=self.params['weights'], K = self.K, NV = self.NV) # Compute log-likelihood for each data in each cluster
        # res : K x N
        res = torch.zeros(self.K, len(self.NV))
        norm_fact = self.log_sum_exp(lks) # sums over the different cluster -> array of size len(NV)
        for k in range(len(self.res)): # iterate over the clusters
            lks_k = lks[k] # take row k -> array of size len(NV)
            res[k] = torch.exp(lks_k - norm_fact)
        self.params["responsib"] = res

        

