import numpy as np
import pyro
import pyro.distributions as dist

import torch
from torch.distributions import constraints

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import beta, pareto
from scipy.integrate import simpson
from sklearn.cluster import KMeans
from utils.BoundedPareto import BoundedPareto

from collections import defaultdict
from pandas.core.common import flatten

def fit(NV = None, DP = None, num_iter = 2000, K = [], tail=1, truncated_pareto = True, 
        purity=1, seed=2, lr = 0.001):
    """
    Function to run the inference with different values of K
    """
    min_bic = torch.tensor(float('inf'))
    mb_list = []
    curr_dict = {}
    for curr_k in K:
        if curr_k != 0:
            print("RUN WITH K = ", curr_k)
            mb = mobster_MV(NV, DP, K = curr_k, seed = seed)
            mb.run_inference(num_iter, lr)
            mb_list.append(mb)
    i = 0
    final_index = 0
    for obj in mb_list:
        if obj.final_dict['bic'] <= min_bic:
            final_index = i
    final_k = K[final_index]
    print("Selected number of clusters is: ", final_k)
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

        self.set_prior_parameters()

    
    def compute_kmeans_centers(self):
        best_inertia = float('inf')
        best_centers = None
        best_cluster = None
        # Implement loop to choose the seed which produces a result with the lowest inertia
        
        for seed in range(1, 16):
            kmeans = KMeans(n_clusters=self.K, random_state=seed, n_init="auto").fit((self.NV/self.DP).numpy())
            best_cluster = kmeans.labels_.copy()
            centers = torch.tensor(kmeans.cluster_centers_)
            # Compute inertia (the lower the better)
            inertia = kmeans.inertia_
            
            # Update best results if current inertia is lower
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers.clone()
        
        # -----------------Gaussian noise------------------#
        
        self.kmeans_centers_no_noise = best_centers.clone()
        self.kmeans_centers_no_noise[self.kmeans_centers_no_noise <= 0] = torch.min(self.min_vaf)
        self.kmeans_centers_no_noise[self.kmeans_centers_no_noise >= 1] = 0.999
        
        mean = 0
        std_dev = 0.05
        D = self.NV.shape[1]
        gaussian_noise = dist.Normal(mean, std_dev).sample([self.K, D])

        # Add gaussian noise to found centers
        best_centers = best_centers + gaussian_noise  
        # -----------------Gaussian noise------------------#
        
        # Clip probabilities in [min_vaf, 0.999]
        best_centers[best_centers <= 0] = torch.min(self.min_vaf)
        best_centers[best_centers >= 1] = 0.999
        self.kmeans_centers = best_centers
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
        pareto_lk = BoundedPareto(self.pareto_L, alpha, 1).log_prob(self.kmeans_centers_no_noise) #+ dist.Binomial(total_count=self.DP, probs = self.kmeans_centers).log_prob(self.NV)
        print("Beta: ", beta_lk)
        print("Pareto: ", pareto_lk)
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
       
    
    def log_beta_par_mix(self, probs_pareto, delta, alpha, a_beta, b_beta):
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
        delta_beta = torch.log(maskk[:, 1]) + dist.BetaBinomial(a_beta, b_beta, total_count=self.DP).log_prob(self.NV) # 1x2 tensor
        """
        # ---------------------Relaxed one hot------------------------------ #
        
        delta_pareto = torch.log(delta[:, 0]) + dist.Binomial(total_count=self.DP, probs = probs_pareto).log_prob(self.NV)  # 1x2 tensor
        delta_beta = torch.log(delta[:, 1]) + dist.BetaBinomial(a_beta, b_beta, total_count=self.DP).log_prob(self.NV) # 1x2 tensor
        
        return self.log_sum_exp(torch.stack((delta_pareto, delta_beta), dim=0)) # creates a 2x2 tensor with torch.stack because log_sum_exp has dim=0

    def m_total_lk(self, probs_pareto, alpha, a_beta, b_beta, weights, delta):
        lk = torch.ones(self.K, len(self.NV)) # matrix with K rows and as many columns as the number of data
        for k in range(self.K):
            lk[k, :] = torch.log(weights[k]) + self.log_beta_par_mix(probs_pareto[k, :], delta[k, :, :], alpha[k, :], a_beta[k, :], b_beta[k, :]).sum(axis=1) # sums over the data dimensions (columns)
                                                                                                                    # put on each column of lk a different data; rows are the clusters
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
        self.phi_beta_L = self.min_vaf
        self.phi_beta_H = self.max_vaf

        # k_beta
        self.k_beta_init = torch.tensor(100.)
        self.k_beta_mean = torch.tensor(100.)
        self.k_beta_std = torch.tensor(0.001)

        # alpha_pareto
        self.alpha_pareto_mean = torch.tensor(1.)
        self.alpha_pareto_std = torch.tensor(0.005)
        self.alpha_pareto_init = torch.tensor(1.)
        self.min_alpha = torch.tensor(0.5)
        self.max_alpha = torch.tensor(4.)

        # Bounded pareto
        self.pareto_L = torch.tensor(0.01)
        # self.pareto_L = self.compute_min_vaf()
        self.pareto_H = self.max_vaf


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

                phi_beta = pyro.sample("phi_beta", dist.Uniform(self.phi_beta_L, self.phi_beta_H)) # 0.5 because we are considering a 1:1 karyotype
                k_beta = pyro.sample("k_beta", dist.LogNormal(torch.log(self.k_beta_mean), self.k_beta_std))
                
                a_beta = phi_beta * k_beta
                b_beta = (1-phi_beta) * k_beta

                alpha = pyro.sample("alpha_pareto", dist.LogNormal(torch.log(self.alpha_pareto_mean), self.alpha_pareto_std))
                probs_pareto = pyro.sample("probs_pareto", BoundedPareto(self.pareto_L, alpha, self.pareto_H)) # probs_pareto is a K x D tensor

        # Data generation
        with pyro.plate("plate_data", len(NV)):
            # .sum() sums over the data because we have a log-likelihood
            pyro.factor("lik", self.log_sum_exp(self.m_total_lk(probs_pareto, alpha, a_beta, b_beta, weights, delta)).sum())


    def guide(self):
        """
        Define the guide for the model.
        """
        NV = self.NV
        K = self.K
        D = NV.shape[1] # number of dimensions (samples)
        weights_param = pyro.param("weights_param", lambda: dist.Dirichlet(torch.ones(K)).sample(), constraint=constraints.simplex)
        pyro.sample("weights", dist.Delta(weights_param).to_event(1))

        alpha_param = pyro.param("alpha_pareto_param", lambda: torch.ones((K,D))*self.alpha_pareto_init, constraint=constraints.interval(self.min_alpha, self.max_alpha))        

        phi_beta_param = pyro.param("phi_beta_param", lambda: self.kmeans_centers, constraint=constraints.interval(self.phi_beta_L, self.phi_beta_H))
        k_beta_param = pyro.param("k_beta_param", lambda: torch.ones((K,D))*self.k_beta_init, constraint=constraints.positive)

        delta_param = pyro.param("delta_param", lambda: self.init_delta, constraint=constraints.simplex)
        
        with pyro.plate("plate_dims", D):
            with pyro.plate("plate_probs", K):

                alpha = pyro.sample("alpha_pareto", dist.Delta(alpha_param)) # here because we need to have K x D samples
                
                pyro.sample("phi_beta", dist.Delta(phi_beta_param))
                pyro.sample("k_beta", dist.Delta(k_beta_param))

                pyro.sample("probs_pareto", BoundedPareto(self.pareto_L, alpha, self.pareto_H))
                pyro.sample("delta", dist.Delta(delta_param).to_event(1)) # not sure


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
    def compute_BIC(self, params):
        n_params = calculate_number_of_params(params)
        n = self.NV.shape[0]
        lk = self.log_sum_exp(self.compute_final_lk()).sum()
        return np.log(n) * n_params - 2 * lk

    def run_inference(self, num_iter = 2000, lr = 0.001):
        pyro.clear_param_store()
        pyro.set_rng_seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        NV, DP = self.NV, self.DP
        self.cluster_initialization()
        svi = pyro.infer.SVI(self.model, self.guide, pyro.optim.Adam({"lr": lr}), pyro.infer.TraceGraph_ELBO())
        
        svi.step()
        gradient_norms = defaultdict(list)
        for name, value in pyro.get_param_store().named_parameters():
            if name in ["alpha_pareto_param", "delta_param"]:
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

        for i in range(num_iter):
            loss = svi.step()
            self.losses.append(loss)

            # Save likelihood values
            params = self.get_parameters()
            a_beta = params["phi_beta_param"] * params["k_beta_param"]
            b_beta = (1-params["phi_beta_param"]) * params["k_beta_param"]
            probs_pareto = BoundedPareto(self.pareto_L, params["alpha_pareto_param"], self.pareto_H).sample()
            lks = self.log_sum_exp(self.m_total_lk(probs_pareto,
                                                   params["alpha_pareto_param"], a_beta, b_beta, 
                                                   params["weights_param"], params["delta_param"])).sum()
            self.lks.append(lks.detach().numpy())

            new_par = params.copy()
            new_par.pop('weights_param')
            check_conv = self.stopping_criteria(old_par, new_par, check_conv)
            
            # If convergence is reached (i.e. changes in parameters are small for min_iter iterations), stop the loop
            if check_conv == min_iter:
                break
            
            if i % 200 == 0:
                print("Iteration {}: Loss = {}".format(i, loss))
            # ----------------------Check parameter convergence---------------------#
            if i == 0:
                for name, value in pyro.get_param_store().items():
                    print(name, pyro.param(name))
            alpha = params["alpha_pareto_param"].detach().numpy()
            weights = params["weights_param"].detach().numpy()
            phi_beta = params["phi_beta_param"].detach().numpy()
            k_beta = params["k_beta_param"].detach().numpy()
            # if i % 400 == 0:
            if i == num_iter-1:
                print("Iteration {}: Loss = {}".format(i, loss))
                if i != 0:
                    print("phi_beta", params["phi_beta_param"].detach().numpy())
                    # print("delta", params["delta_param"].detach().numpy())

                _, axes = plt.subplots(1, 2, figsize=(10, 5))
                x = np.linspace(0.001, 1, 1000)
                for d in range(self.NV.shape[1]):
                    for k in range(self.K):
                        delta_kd = params["delta_param"][k, d]
                        maxx = torch.argmax(delta_kd)
                        if maxx == 1:
                            # plot beta
                            a = phi_beta[k,d] * k_beta[k,d]
                            b = (1-phi_beta[k,d]) * k_beta[k,d]
                            pdf = beta.pdf(x, a, b) * weights[k]
                            axes[d].plot(x, pdf, linewidth=1.5, label='Beta', color='y')
                        else:
                            #plot pareto
                            pdf = pareto.pdf(x, alpha[k,d], scale=self.pareto_L) * weights[k]
                            axes[d].plot(x, pdf, linewidth=1.5, label='Pareto', color='g')
                        
                    axes[d].legend()
                    axes[d].hist(self.NV[:,d].detach().numpy()/self.DP[:,d].detach().numpy(), density=True, bins = 50)
                    axes[d].set_title(f"Sample {d+1}")
                    axes[d].set_ylim([0.,30.])
                    axes[d].set_xlim([0.,1.])
                    plt.tight_layout()
                plt.show()
            # ----------------------End check parameter convergence---------------------#

        self.params = self.get_parameters()
        self.plot_loss_lks()
        self.plot_grad_norms(gradient_norms)
        self.compute_posteriors()
        bic = self.compute_BIC(self.params)
        print("bic: ", bic)
        self.plot()
        self.final_dict = {
        "model_parameters" : self.params,
        "bic": bic,
        "final_likelihood": self.lks[-1],
        "final_loss": self.losses[-1]
        }


    def get_probs(self):
        delta = self.params["delta_param"]  # K x D x 2
        phi_beta = self.params["phi_beta_param"]
        k_beta = self.params["k_beta_param"]
        a_beta = phi_beta * k_beta
        b_beta = (1-phi_beta) * k_beta
        alpha = self.params["alpha_pareto_param"]

        probs = torch.ones(self.K, self.NV.shape[1])
        for k in range(self.K):
            for d in range(self.NV.shape[1]):
                delta_kd = delta[k, d]
                maxx = torch.argmax(delta_kd)
                if maxx == 1: # beta
                    probs[k,d] = dist.Beta(a_beta[k,d], b_beta[k,d]).sample()
                else: # pareto
                    probs[k,d] = BoundedPareto(self.pareto_L,alpha[k,d], self.pareto_H).sample()
        return probs

    def pareto_lk(self, alpha):
        LINSPACE = 8000
        x = torch.linspace(self.pareto_L, self.max_vaf, LINSPACE) # sampled "probability" values (possibili valori discreti sul dominio di integrazione)
        y_1 = torch.stack([
            BoundedPareto(self.pareto_L, alpha[0], self.pareto_H).log_prob(x).exp(),  # dim 0
            BoundedPareto(self.pareto_L, alpha[1], self.pareto_H).log_prob(x).exp()   # dim 1
        ], dim=1)
        y_2 = torch.stack([
            dist.Binomial(probs=x.repeat([self.NV.shape[0], 1]).reshape([LINSPACE, -1]), total_count=self.DP[:, 0]).log_prob(self.NV[:, 0]).exp(),
            dist.Binomial(probs=x.repeat([self.NV.shape[0], 1]).reshape([LINSPACE, -1]), total_count=self.DP[:, 1]).log_prob(self.NV[:, 1]).exp()
        ], dim=2)
        # pareto = simpson((y_1.reshape([LINSPACE, 1,2]) * y_2).numpy(), x=x.numpy(), axis=0)
        # ParetoBin = torch.tensor(pareto).log()
        ParetoBin = torch.trapz(y_1.reshape([LINSPACE, 1, 2]) * y_2, x=x, dim=0).log()
        return ParetoBin
    
    def beta_lk(self, a_beta, b_beta):
        """
        Compute beta-binomial likelihood for a single dimension of a single cluster.
        """
        # LINSPACE = 8000
        # x = torch.linspace(self.min_vaf, self.max_vaf, LINSPACE) # sampled "probability" values (possibili valori discreti sul dominio di integrazione)
        # y_1 = torch.stack([
        #     dist.Beta(a_beta[0], b_beta[0]).log_prob(x).exp(),  # dim 0
        #     dist.Beta(a_beta[1], b_beta[1]).log_prob(x).exp()   # dim 1
        # ], dim=1)
        # y_2 = torch.stack([
        #     dist.Binomial(probs=x.repeat([self.NV.shape[0], 1]).reshape([LINSPACE, -1]), total_count=self.DP[:, 0]).log_prob(self.NV[:, 0]).exp(),
        #     dist.Binomial(probs=x.repeat([self.NV.shape[0], 1]).reshape([LINSPACE, -1]), total_count=self.DP[:, 1]).log_prob(self.NV[:, 1]).exp()
        # ], dim=2)
        # # pareto = simpson((y_1.reshape([LINSPACE, 1,2]) * y_2).numpy(), x=x.numpy(), axis=0)
        # # ParetoBin = torch.tensor(pareto).log()
        # return torch.trapz(y_1.reshape([LINSPACE, 1, 2]) * y_2, x=x, dim=0).log()
        return dist.BetaBinomial(a_beta, b_beta, total_count=self.DP).log_prob(self.NV) # simply does log(weights) + log(density)

    
    def log_beta_par_mix2(self, delta, alpha, a_beta, b_beta):
        # relaxed one hot
        # delta -> D x 2
        delta_pareto = torch.log(delta[:, 0]) + self.pareto_lk(alpha)  # 1x2 tensor
        delta_beta = torch.log(delta[:, 1]) + self.beta_lk(a_beta, b_beta) # 1x2 tensor
        
        return self.log_sum_exp(torch.stack((delta_pareto, delta_beta), dim=0)) # creates a 2x2 tensor with torch.stack because log_sum_exp has dim=0

    def m_total_lk2(self, alpha, a_beta, b_beta, weights, delta):
        lk = torch.ones(self.K, len(self.NV)) # matrix with K rows and as many columns as the number of data
        for k in range(self.K):
            lk[k, :] = torch.log(weights[k]) + self.log_beta_par_mix2(delta[k, :, :], alpha[k, :], a_beta[k, :], b_beta[k, :]).sum(axis=1) # sums over the data dimensions (columns)
                                                                                                                    # put on each column of lk a different data; rows are the clusters
        return lk
    
    def compute_final_lk(self):
        """
        Compute likelihood with learnt parameters
        """
        alpha = self.params["alpha_pareto_param"]
        delta = self.params["delta_param"]  # K x D x 2
        phi_beta = self.params["phi_beta_param"]
        k_beta = self.params["k_beta_param"]
        a_beta = phi_beta * k_beta
        b_beta = (1-phi_beta) * k_beta
        weights = self.params["weights_param"]
        
        lks = self.m_total_lk2(alpha, a_beta, b_beta, weights, delta) # K x N
        return lks


    def compute_posteriors(self):
        lks = self.compute_final_lk() # K x N
        res = torch.zeros(self.K, len(self.NV)) # K x N
        norm_fact = self.log_sum_exp(lks) # sums over the different cluster -> array of size 1 x len(NV)
        for k in range(len(res)): # iterate over the clusters
            lks_k = lks[k] # take row k -> array of size len(NV)
            res[k] = torch.exp(lks_k - norm_fact)
        self.params["responsib"] = res
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
        plt.scatter(probs[:, 0].detach().numpy(), probs[:, 1].detach().numpy(), c = 'r', marker="x")

        # red_patch = mpatches.Patch(color='r', label='Final probs')
        # green_patch = mpatches.Patch(color='g', label='Beta')
        # blue_patch = mpatches.Patch(color='darkorange', label='Pareto')

        plt.title(f"Final inference with K = {self.K}")
        # plt.legend(handles=[red_patch, green_patch, blue_patch])
        plt.gca().add_artist(legend1)
        plt.show()
        
def calculate_number_of_params(params):
    total_params = 0
    for key, param in params.items():
        param_size = np.prod(param.shape)  # Calculate the total number of elements
        total_params += param_size
    return total_params
    