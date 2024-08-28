import numpy as np
import pyro
import pyro.distributions as dist

import torch
from torch.distributions import constraints

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import binom, beta, pareto
from sklearn.cluster import KMeans
from BoundedPareto import BoundedPareto

from collections import defaultdict
from pandas.core.common import flatten

"""
Cosa ho cambiato qui:
    - prior over alpha
    - initial value of phi_beta
"""

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

        self.set_prior_parameters()

    
    def compute_kmeans_centers(self):
        # Implement loop to choose the seed which produces a result with the lowest inertia
        kmeans = KMeans(n_clusters=self.K, random_state=self.seed, n_init="auto").fit((self.NV/self.DP).numpy())
        cluster = kmeans.labels_
        centers = torch.tensor(kmeans.cluster_centers_)

        # -----------------Gaussian noise------------------#
        """
        mean = 0
        std_dev = 0.005
        D = self.NV.shape[1]
        gaussian_noise = torch.abs(dist.Normal(mean, std_dev).sample([self.K, D]))

        # Add gaussian noise to found centers
        centers = centers + gaussian_noise

        # Clip probabilities in [0, 1]
        # self.kmeans_centers = torch.clip(centers, 0, 1)
        """
        # -----------------Gaussian noise------------------#
        
        centers[centers <= 0] = 0.001
        centers[centers >= 1] = 0.999
        self.kmeans_centers = centers
        print("kmeans_centers: ", self.kmeans_centers)

        

    def cluster_initialization(self):
        
        self.compute_kmeans_centers()
        self.init_delta = self.initialize_delta(self.kmeans_centers, self.k_beta_init, self.alpha_pareto_mean)



    def beta_lk(self, probs_beta, a_beta, b_beta, weights):
        """
        Compute beta-binomial likelihood for a single dimension of a single cluster.
        """
        return torch.log(weights) + dist.Beta(a_beta, b_beta).log_prob(probs_beta) + dist.Binomial(total_count=self.DP, probs = probs_beta).log_prob(self.NV) # simply does log(weights) + log(density)


    def pareto_lk(self, probs_pareto, alpha, weights):
        return torch.log(weights) + BoundedPareto(0.01, alpha, 0.55).log_prob(probs_pareto) + dist.Binomial(total_count=self.DP, probs = probs_pareto).log_prob(self.NV) # simply does log(weights) + log(density)

    def initialize_delta(self, phi_beta, k_beta, alpha):
        a_beta = phi_beta * k_beta
        b_beta = (1-phi_beta) * k_beta
        beta_lk = dist.Beta(a_beta, b_beta).log_prob(self.kmeans_centers)# + dist.Binomial(total_count=self.DP, probs = self.kmeans_centers).log_prob(self.NV)
        pareto_lk = BoundedPareto(0.01, alpha, 0.55).log_prob(self.kmeans_centers) #+ dist.Binomial(total_count=self.DP, probs = self.kmeans_centers).log_prob(self.NV)
        print("Beta: ", beta_lk)
        print("Pareto: ", pareto_lk)
        # kmeans_centers: KxD
        K = self.K
        D = self.NV.shape[1]
        init_delta = torch.zeros((K,D,2))
        for i in range(K):
            for j in range(D):
                if(beta_lk[i,j] > pareto_lk[i,j]):
                    init_delta[i,j,0] = 0.1 # pareto
                    init_delta[i,j,1] = 0.9 # beta
                else:
                    init_delta[i,j,0] = 0.9 # pareto
                    init_delta[i,j,1] = 0.1 # beta
        return init_delta



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
        # delta -> D x 2

        # ---------------------Relaxed one hot------------------------------ #
        """
        maskk = torch.zeros_like(delta)

        mask_min = delta != torch.max(delta, dim=1, keepdim=True).values
        mask_max = delta == torch.max(delta, dim=1, keepdim=True).values
        maskk[mask_min] = 1e-10
        maskk[mask_max] = 1. - 1e-10

        delta_pareto = torch.log(maskk[:, 0]) + BoundedPareto(0.001, alpha, 1).log_prob(probs_pareto) + dist.Binomial(total_count=self.DP, probs = probs_pareto).log_prob(self.NV)  # 1x2 tensor
        delta_beta = torch.log(maskk[:, 1]) + dist.Beta(a_beta, b_beta).log_prob(probs_beta) + dist.Binomial(total_count=self.DP, probs = probs_beta).log_prob(self.NV) # 1x2 tensor
        """
        # ---------------------Relaxed one hot------------------------------ #

        delta_pareto = torch.log(delta[:, 0]) + BoundedPareto(self.pareto_L, alpha, self.pareto_H).log_prob(probs_pareto) + dist.Binomial(total_count=self.DP, probs = probs_pareto).log_prob(self.NV)  # 1x2 tensor
        delta_beta = torch.log(delta[:, 1]) + dist.Beta(a_beta, b_beta).log_prob(probs_beta) + dist.Binomial(total_count=self.DP, probs = probs_beta).log_prob(self.NV) # 1x2 tensor

        return self.log_sum_exp(torch.stack((delta_pareto, delta_beta), dim=0)) # creates a 2x2 tensor with torch.stack because log_sum_exp has dim=0

    def m_total_lk(self, probs_beta, probs_pareto, alpha, a_beta, b_beta, weights, delta):
        lk = torch.ones(self.K, len(self.NV)) # matrix with K rows and as many columns as the number of data
        if self.K == 1:
            return torch.log(weights) + self.log_beta_par_mix(probs_beta, probs_pareto, delta[0, :, :], alpha, a_beta, b_beta).sum(axis=1) # simply does log(weights) + log(density)
        for k in range(self.K):
            lk[k, :] = torch.log(weights[k]) + self.log_beta_par_mix(probs_beta[k, :], probs_pareto[k, :], delta[k, :, :], alpha[k, :], a_beta[k, :], b_beta[k, :]).sum(axis=1) # sums over the data dimensions (columns)
                                                                                                                    # put on each column of lk a different data; rows are the clusters                                                                                             
        
        return lk

    def set_prior_parameters(self):

        self.max_vaf = 0.51 # for a 1:1 karyotype

        # phi_beta
        self.phi_beta_L = 0.
        self.phi_beta_H = self.max_vaf
        # self.phi_beta_H = 1.

        # k_beta
        self.k_beta_mean = 100
        self.k_beta_std = 0.5
        self.k_beta_init = 5

        # alpha_pareto (normal)
        self.alpha_pareto_mean = 2
        self.alpha_pareto_std = 0.0005
        # alpha_pareto (log-normal)
        # self.alpha_pareto_mean = 0.7
        # self.alpha_pareto_std = 0.005
        self.alpha_pareto_init = 2.

        # Bounded pareto
        self.pareto_L = 0.01
        self.pareto_H = self.max_vaf


    def model(self):
        """
        Define the model.
        """
        NV = self.NV
        K = self.K
        D = NV.shape[1] # number of dimensions (samples)

        weights = pyro.sample("weights", dist.Dirichlet(torch.ones(K)))
        phi_beta_L = self.phi_beta_L
        phi_beta_H = self.phi_beta_H

        k_beta_mean = self.k_beta_mean
        k_beta_std = self.k_beta_std

        alpha_pareto_mean = self.alpha_pareto_mean
        alpha_pareto_std = self.alpha_pareto_std

        pareto_L = self.pareto_L
        pareto_H = self.pareto_H

        with pyro.plate("plate_dims", D):
            with pyro.plate("plate_probs", K):
                # Prior for the Beta-Pareto weights
                 # delta is a K x D x 2 torch tensor (K: num layers, D: rows per layer, 2: columns per layer)
                delta = pyro.sample("delta", dist.Dirichlet(torch.ones(2)))

                # phi_beta = pyro.sample("phi_beta", dist.Beta(2., 2.)) # 3., 2.
                phi_beta = pyro.sample("phi_beta", dist.Uniform(phi_beta_L, phi_beta_H)) # 0.5 because we are considering a 1:1 karyotype
                k_beta = pyro.sample("k_beta", dist.Normal(k_beta_mean, k_beta_std))

                a_beta = phi_beta * k_beta
                b_beta = (1-phi_beta) * k_beta

                # assume Beta prior for the success probabilities
                probs_beta = pyro.sample("probs_beta", dist.Beta(a_beta, b_beta)) # probs_beta is a K x D tensor

                # alpha_mu = pyro.sample("alpha_mu", dist.Uniform(0.5,1.))
                # alpha = pyro.sample("alpha_pareto", dist.LogNormal(alpha_mu, 0.3)) # alpha is a K x D tensor
                # ------------------------Learn alpha------------------------- #
                alpha = pyro.sample("alpha_pareto", dist.Normal(alpha_pareto_mean, alpha_pareto_std)) # alpha is a K x D tensor
                # alpha = pyro.sample("alpha_pareto", dist.LogNormal(alpha_pareto_mean, alpha_pareto_std))


                
                # alpha = pyro.sample("alpha_pareto", dist.Gamma(5., 5.)) # alpha is a K x D tensor
                # alpha = torch.ones((K,D))*2.                
                # ------------------------Learn alpha------------------------- #
                probs_pareto = pyro.sample("probs_pareto", BoundedPareto(pareto_L, alpha, pareto_H)) # probs_pareto is a K x D tensor

        # Data generation
        with pyro.plate("plate_data", len(NV)):
            # .sum() sums over the data because we have a log-likelihood
            pyro.factor("lik", self.log_sum_exp(self.m_total_lk(probs_beta, probs_pareto, alpha, a_beta, b_beta, weights, delta)).sum())


    def guide(self):
        """
        Define the guide for the model.
        """
        NV = self.NV
        K = self.K
        D = NV.shape[1] # number of dimensions (samples)
        weights_param = pyro.param("weights_param", lambda: dist.Dirichlet(torch.ones(K)).sample(), constraint=constraints.simplex)
        pyro.sample("weights", dist.Delta(weights_param).to_event(1))

        # ------------------------Learn alpha------------------------- #
        alpha_param = pyro.param("alpha_param", lambda: torch.ones((K,D))*self.alpha_pareto_init, constraint=constraints.positive)
        print("Alpha: ", alpha_param)
        # ------------------------Learn alpha------------------------- #
        

        # alpha_mu_param = pyro.param("alpha_mu_param", lambda: torch.ones((K,D))*0.1, constraint=constraints.interval(0.5, 1.)) 
        # alpha_param = pyro.param("alpha_param", lambda: torch.ones((K,D))*2, constraint=constraints.positive)


        phi_beta_param = pyro.param("phi_beta_param", lambda: self.kmeans_centers, constraint=constraints.interval(0., self.max_vaf))
        k_beta_param = pyro.param("k_beta_param", lambda: torch.ones((K,D))*self.k_beta_init, constraint=constraints.positive)

        probs_beta_param = pyro.param("probs_beta_param", lambda: self.kmeans_centers, constraint=constraints.interval(0., self.max_vaf))
        probs_pareto_param = pyro.param("probs_pareto_param", lambda: self.kmeans_centers, constraint=constraints.interval(0., self.max_vaf))
        
        delta_param = pyro.param("delta_param", lambda: self.init_delta, constraint=constraints.simplex)
        # delta_param = pyro.param("delta_param", lambda: dist.Dirichlet(torch.ones(2)).sample([K, D]).reshape(K, D, 2), constraint=constraints.simplex)

        with pyro.plate("plate_dims", D):
            with pyro.plate("plate_probs", K):
                # pyro.sample("alpha_mu", dist.Delta(alpha_mu_param))
                
                # ------------------------Learn alpha------------------------- #
                pyro.sample("alpha_pareto", dist.Delta(alpha_param)) # here because we need to have K x D samples
                # ------------------------Learn alpha------------------------- #

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
        # ------------------------Learn alpha------------------------- #
        params["alpha_pareto"] = param_store["alpha_param"].clone().detach()
        # params["alpha_pareto"] = torch.ones((self.K,self.NV.shape[1]))*2.
        # ------------------------Learn alpha------------------------- #
        params["phi_beta"] = param_store["phi_beta_param"].clone().detach()
        params["k_beta"] = param_store["k_beta_param"].clone().detach()

        """
        Per farlo più automatico:
        param_names = pyro.get_param_store()
        params = {nms: pyro.param(nms) for nms in param_names}
        """

        return params
    
    
    def flatten_params(self, pars):
        # pars = list(pars.values().detach().tolist())
        pars = list(flatten([value.detach().tolist() for key, value in pars.items()]))
        # print("pars: ", pars)
        return(np.array(pars))

    
    def stopping_criteria(self, old_par, new_par):#, e=0.01):
        old = self.flatten_params(old_par)
        new = self.flatten_params(new_par)
        diff_mix = np.abs(old - new)# / np.abs(old)
        # if np.all(diff_mix < e):
        if np.all(diff_mix < old*0.1):
            return True
        return False

    def fit(self, num_iter = 2000, lr = 0.001):
        pyro.clear_param_store()
        pyro.set_rng_seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        NV, DP = self.NV, self.DP
        K = self.K
        self.cluster_initialization()
        svi = pyro.infer.SVI(self.model, self.guide, pyro.optim.Adam({"lr": lr}), pyro.infer.TraceGraph_ELBO())
        
        svi.step()
        gradient_norms = defaultdict(list)
        for name, value in pyro.get_param_store().named_parameters():
            if name in ["probs_beta_param", "probs_pareto_param", "alpha_param", "delta_param"]:
                value.register_hook(
                    lambda g, name=name: gradient_norms[name].append(g.norm().item())
                )
        
        self.losses = []
        self.lks = []
        i = 0
        min_iter = 100
        check_conv = False
        old_par = self.get_parameters() # Save current values of the parameters in old_params

        for i in range(num_iter):
            loss = svi.step()
            self.losses.append(loss)

            # Save likelihood values
            params = self.get_parameters()
            a_beta = params["phi_beta"] * params["k_beta"]
            b_beta = (1-params["phi_beta"]) * params["k_beta"]
            lks = self.log_sum_exp(self.m_total_lk(params["probs_beta"], params["probs_pareto"], 
                                                   params["alpha_pareto"], a_beta, b_beta, 
                                                   params["weights"], params["delta"])).sum()
            self.lks.append(lks)

            new_par = params.copy()
            check_conv = self.stopping_criteria(old_par, new_par)
            # If the minimum number of steps and convergence are reached, stop the loop
            if i > min_iter and check_conv:
                break
            
            if i % 200 == 0:
                print("Iteration {}: Loss = {}".format(i, loss))
            # ----------------------Check parameter convergence---------------------#
            """"""
            if i == 0:
                for name, value in pyro.get_param_store().items():
                    print(name, pyro.param(name))
            alpha = params["alpha_pareto"].numpy()
            weights = params["weights"].numpy()
            phi_beta = params["phi_beta"].numpy()
            k_beta = params["k_beta"].numpy()
            if i % 400 == 0:
                if i != 0:
                    print("probs_beta", params["probs_beta"].numpy())

                _, axes = plt.subplots(1, 2, figsize=(10, 5))
                x = np.linspace(0.001, 1, 1000)
                for d in range(self.NV.shape[1]):
                    for k in range(self.K):
                        delta_kd = params["delta"][k, d]
                        maxx = torch.argmax(delta_kd)
                        if maxx == 1:
                            # plot beta
                            a = phi_beta[k,d] * k_beta[k,d]
                            b = (1-phi_beta[k,d]) * k_beta[k,d]
                            pdf = beta.pdf(x, a, b) * weights[k]
                            axes[d].plot(x, pdf, linewidth=1.5, label='Beta', color='y')
                        else:
                            #plot pareto
                            pdf = pareto.pdf(x, alpha[k,d], scale=0.01) * weights[k]
                            axes[d].plot(x, pdf, linewidth=1.5, label='Pareto', color='g')
                        
                    axes[d].legend()
                    axes[d].hist(self.NV[:,d].numpy()/self.DP[:,d].numpy(), density=True, bins = 50)
                    axes[d].set_title(f"Dimension {d+1}")
                    axes[d].set_ylim([0,30])
                    plt.tight_layout()
                plt.show()
            # ----------------------End check parameter convergence---------------------#

        self.params = self.get_parameters()
        self.plot_loss_lks()
        self.plot_grad_norms(gradient_norms)
        self.compute_posteriors()
        self.plot()


    def get_probs(self):
        probs_beta = self.params["probs_beta"]  # K x D
        probs_pareto = self.params["probs_pareto"]  # K x D
        delta = self.params["delta"]  # K x D x 2

        probs = torch.ones(self.K, self.NV.shape[1])
        for k in range(self.K):
            for d in range(self.NV.shape[1]):
                delta_kd = delta[k, d]
                maxx = torch.argmax(delta_kd)
                probs[k,d] = probs_beta[k,d] if maxx==1 else probs_pareto[k,d]

        return probs


    def compute_posteriors(self):
        """
        Compute posterior assignment probabilities (i.e., the responsibilities) given the learned parameters.
        """
        # lks : K x N
        probs = self.get_probs()
        lks = self.m_binomial_lk(probs=probs, DP = self.DP, weights=self.params['weights'], K = self.K, NV = self.NV) # Compute log-likelihood for each data in each cluster
        # lks = self.m_binomial_lk(probs=self.params['probs_beta'], DP = self.DP, weights=self.params['weights'], K = self.K, NV = self.NV) # Compute log-likelihood for each data in each cluster
        # res : K x N
        res = torch.zeros(self.K, len(self.NV))
        norm_fact = self.log_sum_exp(lks) # sums over the different cluster -> array of size len(NV)
        for k in range(len(res)): # iterate over the clusters
            lks_k = lks[k] # take row k -> array of size len(NV)
            res[k] = torch.exp(lks_k - norm_fact)
        self.params["responsib"] = res
        # For each data point find the cluster assignment (si può creare anche una funzione a parte)
        self.params["cluster_assignments"] = torch.argmax(self.params["responsib"], dim = 0) # vector of dimension


    
    def plot_loss_lks(self):
        _, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].plot(self.losses)
        ax[0].set_title("Loss")
        ax[1].plot(self.lks)
        ax[1].set_title("Likelihood")
        plt.show()

    def plot_grad_norms(self, gradient_norms):
        plt.figure(figsize=(10, 4), dpi=100).set_facecolor("white")
        for name, grad_norms in gradient_norms.items():
            plt.plot(grad_norms, label=name)
        plt.xlabel("iters")
        plt.ylabel("gradient norm")
        plt.yscale("log")
        plt.legend(loc="best")
        plt.title("Gradient norms during SVI")
        plt.show()
        
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

        probs = self.get_probs()
        plt.scatter(self.params['probs_beta'][:, 0], self.params['probs_beta'][:, 1], c = 'g', label="Beta")
        plt.scatter(self.params['probs_pareto'][:, 0], self.params['probs_pareto'][:, 1], c = 'darkorange')
        plt.scatter(probs[:, 0], probs[:, 1], c = 'r', marker="x")

        red_patch = mpatches.Patch(color='r', label='Final probs')
        green_patch = mpatches.Patch(color='g', label='Beta')
        blue_patch = mpatches.Patch(color='darkorange', label='Pareto')

        plt.title("Final inference")
        plt.legend(handles=[red_patch, green_patch, blue_patch])
        plt.gca().add_artist(legend1)
        plt.show()
        
        
    

    

