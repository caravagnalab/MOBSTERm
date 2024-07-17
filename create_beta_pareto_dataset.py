from BoundedPareto import BoundedPareto
import numpy as np
import pyro
import pyro.distributions as dist

import torch

def pareto_binomial_component(alpha=2, L=0.05, H=0.5, p=0.5, n=100, N=1000, exchanged = False, seed = 123):
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
    for i in range(N):
        p_p = BoundedPareto(scale=L, alpha = alpha, upper_limit = H).sample().float()
        d1[i, 0] = dist.Binomial(total_count=n, probs=p_p).sample().squeeze(-1)
    # p_p = BoundedPareto(scale=L, alpha = alpha, upper_limit = H).sample().float()
    # d1[:, 0] = dist.Binomial(total_count=n, probs=p_p).sample([N]).squeeze(-1)

    d1[:, 1] = dist.Binomial(total_count=n, probs=p).sample([N]).squeeze(-1)
    DP = torch.ones([N, 2]) * n
    if exchanged == True:
        indices = torch.tensor([1,0])
        d1 = d1[:, indices]

    return d1, DP


def beta_binomial_component(p_x = 0.5, p_y= 0.5, n=100, N=1000, seed=123):
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

def pareto_binomial_component2(alpha=2, L=0.05, H=0.5, a= 0.5, b= 0.5, n=100, N=1000, exchanged = False, seed = 123):
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
    for i in range(N):
        p_p = BoundedPareto(scale=L, alpha = alpha, upper_limit = H).sample().float()
        d1[i, 0] = dist.Binomial(total_count=n, probs=p_p).sample().squeeze(-1)
    # p_p = BoundedPareto(scale=L, alpha = alpha, upper_limit = H).sample().float()
    # d1[:, 0] = dist.Binomial(total_count=n, probs=p_p).sample([N]).squeeze(-1)


    # for i in range(N):
    #     p_p = dist.Beta(a, b).sample().float()
    #     d1[i, 1] = dist.Binomial(total_count=n, probs=p_p).sample().squeeze(-1)

    p = dist.Beta(a, b).sample()
    d1[:, 1] = dist.Binomial(total_count=n, probs=p).sample([N]).squeeze(-1)
    DP = torch.ones([N, 2]) * n
    if exchanged == True:
        indices = torch.tensor([1,0])
        d1 = d1[:, indices]

    return d1, DP


def beta_binomial_component2(a_x = 0.5, b_x = 0.5, a_y= 0.5, b_y= 0.5, n=100, N=1000, seed=123):
    """
    Create Beta-Binomial component:
    x-axis is a Beta-Binomial
    y-axis is a Beta-Binomial
    """
    pyro.set_rng_seed(seed)
    d2 = torch.ones([N, 2])

    # for i in range(N):
    #     p_x = dist.Beta(a_x, b_x).sample().float()
    #     d2[i, 0] = dist.Binomial(total_count=n, probs=p_x).sample().squeeze(-1)
    #     p_y = dist.Beta(a_y, b_y).sample().float()
    #     d2[i, 1] = dist.Binomial(total_count=n, probs=p_y).sample().squeeze(-1)


    # x-axis component 2
    p_x = dist.Beta(a_x, b_x).sample()
    d2[:, 0] = dist.Binomial(total_count=n, probs=p_x).sample([N]).squeeze(-1)
    
    # # y-axis component 2
    p_y = dist.Beta(a_y, b_y).sample()
    d2[:, 1] = dist.Binomial(total_count=n, probs=p_y).sample([N]).squeeze(-1)

    DP = torch.ones([N, 2]) * n

    return d2, DP


def pareto_binomial_component3(alpha=2, L=0.05, H=0.5, a= 0.5, b= 0.5, n=100, N=1000, exchanged = False, seed = 123):
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
    for i in range(N):
        p_p = BoundedPareto(scale=L, alpha = alpha, upper_limit = H).sample().float()
        d1[i, 0] = dist.Binomial(total_count=n, probs=p_p).sample().squeeze(-1)
    # p_p = BoundedPareto(scale=L, alpha = alpha, upper_limit = H).sample().float()
    # d1[:, 0] = dist.Binomial(total_count=n, probs=p_p).sample([N]).squeeze(-1)


    for i in range(N):
        p_p = dist.Beta(a, b).sample().float()
        d1[i, 1] = dist.Binomial(total_count=n, probs=p_p).sample().squeeze(-1)

    # p = dist.Beta(a, b).sample()
    # d1[:, 1] = dist.Binomial(total_count=n, probs=p).sample([N]).squeeze(-1)
    DP = torch.ones([N, 2]) * n
    if exchanged == True:
        indices = torch.tensor([1,0])
        d1 = d1[:, indices]

    return d1, DP


def beta_binomial_component3(a_x = 0.5, b_x = 0.5, a_y= 0.5, b_y= 0.5, n=100, N=1000, seed=123):
    """
    Create Beta-Binomial component:
    x-axis is a Beta-Binomial
    y-axis is a Beta-Binomial
    """
    pyro.set_rng_seed(seed)
    d2 = torch.ones([N, 2])

    for i in range(N):
        p_x = dist.Beta(a_x, b_x).sample().float()
        d2[i, 0] = dist.Binomial(total_count=n, probs=p_x).sample().squeeze(-1)
        p_y = dist.Beta(a_y, b_y).sample().float()
        d2[i, 1] = dist.Binomial(total_count=n, probs=p_y).sample().squeeze(-1)


    # x-axis component 2
    # p_x = dist.Beta(a_x, b_x).sample()
    # d2[:, 0] = dist.Binomial(total_count=n, probs=p_x).sample([N]).squeeze(-1)
    
    # # # y-axis component 2
    # p_y = dist.Beta(a_y, b_y).sample()
    # d2[:, 1] = dist.Binomial(total_count=n, probs=p_y).sample([N]).squeeze(-1)

    DP = torch.ones([N, 2]) * n

    return d2, DP
    