import model_mobster_mv_orfeo as model_mobster_mv
import numpy as np
import pandas as pd
import pyro.distributions as dist
from scipy import stats

import torch
import seaborn as sns

import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.metrics.cluster import normalized_mutual_info_score

from utils.plot_functions import *
from utils.BoundedPareto import BoundedPareto
from utils.create_beta_pareto_dataset import *


data = pd.read_csv("./real_data/Set7_mutations.csv")

NV1 = torch.tensor(data['Set7_55.NV'].to_numpy())
DP1 = torch.tensor(data['Set7_55.DP'].to_numpy())

NV2 = torch.tensor(data['Set7_57.NV'].to_numpy())
DP2 = torch.tensor(data['Set7_57.DP'].to_numpy())

NV = torch.stack((NV1,NV2),dim=1)
DP = torch.stack((DP1,DP2),dim=1)

print("Original NV data shape", NV.shape)
print("Original DP data shape",DP.shape)

# Plot the dataset
plt.xlim([0,1])
plt.ylim([0,1])
plt.scatter(NV[:,0]/DP[:,0], NV[:,1]/DP[:,1])

plt.xlabel('Set7_55')
plt.ylabel('Set7_57')
plt.title('Set7_55 vs Set7_57')
plt.savefig('plots/original_data.png') 
plt.close()

final_mb, mb_list = model_mobster_mv.fit(NV, DP, num_iter = 3000, K = [5], seed = [40], lr = 0.005)

plot_marginals(final_mb)
plot_deltas(final_mb)
plot_paretos(final_mb)
plot_betas(final_mb)

print("FINAL PARAMETERS: ", final_mb.params)