import numpy as np
import pyro
import pyro.distributions as dist

import torch
from torch.distributions import constraints

import matplotlib.pyplot as plt
from scipy.stats import binom, beta, pareto
from sklearn.cluster import KMeans

class beta_MV():
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

        if(NV is not None):
            self.NV = torch.tensor(NV) if not isinstance(NV, torch.Tensor) else NV
        if(DP is not None):
            self.DP = torch.tensor(DP) if not isinstance(DP, torch.Tensor) else DP
        self.K = K
        self.tail = tail
        self.truncated_pareto = truncated_pareto
        self.purity = purity
        self.seed = seed
    

    def create_dataset(self, n_components = 2, n_samples = torch.tensor([1000, 2000]), total_counts = torch.tensor([100, 150]), probs = torch.tensor([[.1, .3], [.5, .6]]), seed = 1):
        """
        Function to create a multidimensional binomial mixture dataset.
        """
        pyro.set_rng_seed(seed)
        self.NV = torch.ones([n_samples[0], 2])
        self.NV[:,0] = dist.Binomial(total_count=total_counts[0], probs=probs[0,0]).sample([n_samples[0]]).squeeze(-1) # S1 for component 1
        self.NV[:,1] = dist.Binomial(total_count=total_counts[0], probs=probs[0,1]).sample([n_samples[0]]).squeeze(-1) # S2 for component 1
        self.DP = torch.ones([n_samples[0], 2]) * total_counts[0]
        for i in range(1,n_components):
            d = torch.ones([n_samples[i], 2])
            d[:,0] = dist.Binomial(total_count=total_counts[i], probs=probs[i,0]).sample([n_samples[i]]).squeeze(-1) # S1 for component i
            d[:,1] = dist.Binomial(total_count=total_counts[i], probs=probs[i,1]).sample([n_samples[i]]).squeeze(-1) # S2 for component i
            c = torch.ones([n_samples[i], 2]) * total_counts[i]
            self.NV = torch.concat((self.NV,d))
            self.DP = torch.concat((self.DP,c))
        # plt.xlim([0,1])
        # plt.ylim([0,1])
        plt.scatter(self.NV[:,0]/self.DP[:,0], self.NV[:,1]/self.DP[:,1])
        plt.title("VAF 2d spectrum")
        
    def compute_kmeans_centers(self, seed):
        kmeans = KMeans(n_clusters=self.K, random_state=seed, n_init="auto").fit((self.NV/self.DP).numpy())
        cluster = kmeans.labels_
        self.kmeans_centers = torch.tensor(kmeans.cluster_centers_)

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
        with pyro.plate("plate_dims", D):
        # probs = pyro.sample("probs", dist.Beta(1, 1).expand([D]).to_event(1)) # assume Beta prior for the success probabilities   
            with pyro.plate("plate_probs", K):    
                probs = pyro.sample("probs", dist.Beta(1, 1)) # assume Beta prior for the success probabilities
        
        # Data generation
        with pyro.plate("plate_data", len(NV)):
            # .sum() sums over the data because we have a log-likelihood (sums over the log-likelihood of each data)
            pyro.factor("lik", self.log_sum_exp(self.m_binomial_lk(probs, DP, weights, K, NV)).sum())
            

    def guide(self):
        """
        Define the guide for the model.
        """
        NV, DP = self.NV, self.DP
        K = self.K
        D = NV.shape[1] # number of dimensions (samples)

        weights_param = pyro.param("weights_param", lambda: dist.Dirichlet(torch.ones(K)).sample(), constraint=constraints.simplex)
        pyro.sample("weights", dist.Delta(weights_param).to_event(1))

        probs_param = pyro.param("probs_param", self.kmeans_centers, constraint=constraints.interval(0.,1.))
        print(probs_param)
        
        # Probability of success for each component
        with pyro.plate("plate_dims", D):
            with pyro.plate("plate_probs", K):
                pyro.sample("probs", dist.Delta(probs_param))

    def get_parameters(self):
        """
        Extract the learned parameters.
        """
        param_store = pyro.get_param_store()
        params = {}
        params["probs_bin"] = param_store["probs_param"].clone().detach()
        params["weights"] = param_store["weights_param"].clone().detach()

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
        self.plot()

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

        f = torch.ones([1000, 2])
        f[:,0] = dist.Binomial(total_count=n, probs=self.params["probs_bin"][0,0]).sample([1000]).squeeze(-1)
        f[:,1] = dist.Binomial(total_count=n, probs=self.params["probs_bin"][0,1]).sample([1000]).squeeze(-1)
        plt.scatter(f[:,0]/n, f[:,1]/n, c = 'r', label = "Samples from fitted beta components")

        for i in range(1, self.K):
            f = torch.ones([1000, 2])
            f[:,0] = dist.Binomial(total_count=n, probs=self.params["probs_bin"][i,0]).sample([1000]).squeeze(-1)
            f[:,1] = dist.Binomial(total_count=n, probs=self.params["probs_bin"][i,1]).sample([1000]).squeeze(-1)
            plt.scatter(f[:,0]/n, f[:,1]/n, c = 'r')

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
        # For each data point find the cluster assignment (si può creare anche una funzione a parte)
        self.params["cluster_assignments"] = torch.argmax(self.params["cluster_probs"], dim = 0) # vector of dimension

    

        

