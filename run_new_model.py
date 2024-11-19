import new_model as model_mobster_mv
import numpy as np
import pandas as pd
import pyro.distributions as dist
from scipy import stats
import pickle
import json
import ast

import torch
import seaborn as sns

import matplotlib.pyplot as plt

from utils.plot_functions import *
from utils.BoundedPareto import BoundedPareto
from utils.create_beta_pareto_dataset import *

import os

"""
N1 = 500
N2 = 500
N3 = 500
N4 = 500
N5 = 500
N6 = 500
seed = 123
# Component 1
phi_beta_x = 0.5
k_beta_x = 300
phi_beta_y = 0.5
k_beta_y= 300
n1=100
NV1, DP1 = beta_binomial_component(phi_beta_x = phi_beta_x, k_beta_x = k_beta_x, phi_beta_y = phi_beta_y, k_beta_y= k_beta_y, n=n1, N=N1, seed=seed)


# Component 2
phi_beta_x = 0.5
k_beta_x = 300
phi_beta_y = 1e-10
k_beta_y= 300
n2=100
NV2, DP2 = beta_binomial_component(phi_beta_x = phi_beta_x, k_beta_x = k_beta_x, phi_beta_y = phi_beta_y, k_beta_y= k_beta_y, n=n2, N=N2, seed=seed)
NV2[:,1] = torch.tensor(0, dtype=NV2.dtype)
NV = torch.concat((NV1,NV2))
DP = torch.concat((DP1,DP2))

# Component 3
phi_beta_x = 0.2
k_beta_x = 300
phi_beta_y = 1e-10
k_beta_y= 300
n3=100
NV3, DP3 = beta_binomial_component(phi_beta_x = phi_beta_x, k_beta_x = k_beta_x, phi_beta_y = phi_beta_y, k_beta_y= k_beta_y, n=n3, N=N3, seed=seed)
NV3[:,1] = torch.tensor(0, dtype=NV3.dtype)
NV = torch.concat((NV,NV3))
DP = torch.concat((DP,DP3))


# Component 4
phi_beta_x = 1e-10
k_beta_x = 300
phi_beta_y = 0.4
k_beta_y= 150
n4=100
NV4, DP4 = beta_binomial_component(phi_beta_x = phi_beta_x, k_beta_x = k_beta_x, phi_beta_y = phi_beta_y, k_beta_y= k_beta_y, n=n4, N=N4, seed=seed)
NV4[:,0] = torch.tensor(0, dtype=NV4.dtype)
NV = torch.concat((NV,NV4))
DP = torch.concat((DP,DP4))


# Component 5
L_pareto = 0.01
H_pareto = 0.5
alpha_pareto = 1.5 # x-axis
phi_beta = 1e-10 # y-axis
k_beta = 300
n5=100
NV5, DP5 = pareto_binomial_component(alpha=alpha_pareto, L=L_pareto, H=H_pareto, phi_beta = phi_beta, k_beta = k_beta, n=n5, N=N5,exchanged = False, seed = seed)
NV5[:,1] = torch.tensor(0, dtype=NV1.dtype)
NV5[np.where(NV5[:,0] == 0), 0] = torch.tensor(1, dtype=NV1.dtype)
NV = torch.concat((NV,NV5))
DP = torch.concat((DP,DP5))
print(NV.shape)
print(DP.shape)


# Component 6
alpha_pareto = 1.5
phi_beta = 1e-10
k_beta = 300
n6=100
NV6, DP6 = pareto_binomial_component(alpha=alpha_pareto, L=L_pareto, H=H_pareto, phi_beta = phi_beta, k_beta = k_beta, n=n6, N=N6, exchanged = True, seed = seed)
NV6[:,0] = torch.tensor(0, dtype=NV6.dtype)
NV6[np.where(NV6[:,1] == 0), 1] = torch.tensor(1, dtype=NV1.dtype)
NV = torch.concat((NV,NV6))
DP = torch.concat((DP,DP6))
"""

# NV_r = pd.read_csv("./data/new_sim/NV_lesson.csv")
# DP_r = pd.read_csv("./data/new_sim/DP_lesson.csv")

# NV = torch.tensor(NV_r.to_numpy())
# DP = torch.tensor(DP_r.to_numpy())
# idx = [1,2]
# NV = NV[:,idx]
# DP = DP[:,idx]

data = pd.read_csv("./data/real_data/Set7_mutations.csv")

NV1 = torch.tensor(data['Set7_55.NV'].to_numpy())
DP1 = torch.tensor(data['Set7_55.DP'].to_numpy())

NV2 = torch.tensor(data['Set7_57.NV'].to_numpy())
DP2 = torch.tensor(data['Set7_57.DP'].to_numpy())

NV = torch.stack((NV1,NV2),dim=1)
DP = torch.stack((DP1,DP2),dim=1)

NV3 = torch.tensor(data['Set7_59.NV'].to_numpy()).view(NV1.shape[0], 1)
DP3 = torch.tensor(data['Set7_59.DP'].to_numpy()).view(NV1.shape[0], 1)

NV = torch.cat((NV,NV3),dim=1)
DP = torch.cat((DP,DP3),dim=1)

NV4 = torch.tensor(data['Set7_62.NV'].to_numpy()).view(NV1.shape[0], 1)
DP4 = torch.tensor(data['Set7_62.DP'].to_numpy()).view(NV1.shape[0], 1)

NV = torch.cat((NV,NV4),dim=1)
DP = torch.cat((DP,DP4),dim=1)
""""""
# ylim = 3000

ylim = 1200
data_folder = 'real_data_high_K_3'

folder_path = f"plots/{data_folder}"
# Create the directory if it does not exist
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print(f"Folder '{folder_path}' created!")
else:
    print(f"Folder '{folder_path}' already exists.")

folder_path = f"plots/{data_folder}/final_analysis"
# Create the directory if it does not exist
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print(f"Folder '{folder_path}' created!")
else:
    print(f"Folder '{folder_path}' already exists.")


folder_path = f"saved_objects/{data_folder}"
# Create the directory if it does not exist
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print(f"Folder '{folder_path}' created!")
else:
    print(f"Folder '{folder_path}' already exists.")

print("Original NV data shape", NV.shape)
print("Original DP data shape", DP.shape)

D = NV.shape[1]
pairs = np.triu_indices(D, k=1)  # Generate all unique pairs of samples (i, j)
vaf = NV/DP    

for i, j in zip(*pairs):
    x = vaf[:, i].numpy()
    y = vaf[:, j].numpy()

    plt.scatter(x, y, alpha=0.7)
    plt.title(f"Sample {i+1} vs Sample {j+1}")
    plt.xlabel(f"Sample {i+1}")
    plt.ylabel(f"Sample {j+1}")
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.show()
    plt.savefig(f'plots/{data_folder}/orig_data_{i+1}_vs_{j+1}.png')
    plt.close()

fig, axes = plt.subplots(1, D, figsize=(5*D, 4))
plt.suptitle("Marginals")

for i in range(D):
    x = vaf[:, i].numpy()
    axes[i].hist(x, bins = 80)    
    
    axes[i].set_xlabel(f"Sample {i+1}")
    axes[i].set_xlim([0,1])
    axes[i].set_ylim([0,ylim])

plt.show()
plt.savefig(f'plots/{data_folder}/marginals.png')
plt.close()

save = True
seed_list = [40,41,42]
# K_list = [11,12,13,14,15,16,17]
# K_list = [11,12,13,14]
# K_list = [15,16,17]
# K_list = [18,19,20,21]
K_list = [20,21]
# K_list = [4,5,6,7,8]
_, best_K, best_seed =  model_mobster_mv.fit(NV, DP, num_iter = 3500, K = K_list, seed = seed_list, lr = 0.01, savefig = save, data_folder = data_folder)

# Restore saved objects
loaded_list = []
for k in K_list:
    filename = f'saved_objects/{data_folder}/K_{k}.txt'
    with open(filename, 'r') as f:
        for line in f:
            # Use json.loads to safely convert the JSON string back to a dictionary
            loaded_dict = json.loads(line.strip())
            
            # Convert any lists back to NumPy arrays
            for key, value in loaded_dict.items():
                if isinstance(value, list):  # Check if the value is a list
                    loaded_dict[key] = np.array(value)  # Convert the list back to a NumPy array
            
            loaded_list.append(loaded_dict)

# Create a list of all saved objects
mb_list = []
for i in range(len(seed_list)*len(K_list)):
    K = loaded_list[i]['K']
    seed = loaded_list[i]['seed']
    mb_list.append(model_mobster_mv.mobster_MV())
    mb_list[i].__dict__ = loaded_list[i]
    print(f'K = {mb_list[i].K}, seed = {mb_list[i].seed}')

#  Create a dataframe with all the results
def highlight_min(data):
    attr = 'background-color: #ffffcc'
    result = pd.DataFrame('', index=data.index, columns=data.columns)
    for col in data.columns:
        if col != "final_likelihood" and col != "final_loss":
            min_value = data[col].min()
            result[col] = [attr if v == min_value else '' for v in data[col]]
    return result

keys_of_interest = ["bic", "icl", "final_likelihood", "final_loss"]
data_for_df = {}


for i in range(len(mb_list)):
    row_name = f"K = {mb_list[i].K}, seed = {mb_list[i].seed}"
    data_for_df[row_name] = [round(mb_list[i].final_dict[key], 2) for key in keys_of_interest]

df = pd.DataFrame(data_for_df, index=keys_of_interest).T  # Transpose to make rows as object names
print(df)
fig, ax = plt.subplots(figsize=(14, 4))

ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
ax.set_frame_on(False)
table = ax.table(cellText=df.values,
                 colLabels=df.columns,
                 rowLabels=df.index,
                 cellLoc='center',
                 loc='center')

highlight = highlight_min(df)
for (i, j), val in np.ndenumerate(df.values):
    if highlight.iloc[i, j] != '':
        table[(i + 1, j)].set_facecolor('lightgreen')

table.scale(1, 2)
plt.subplots_adjust(left=0.2, bottom=0.2)
plt.savefig(f"plots/{data_folder}/final_analysis/final_values.png", bbox_inches='tight', dpi=300)
plt.close()

# Plot bic and likelihood over different K (take the best seed for each K)

lk_list = []
bic_list = []
icl_list = []
for i, k in enumerate(K_list):
    start_idx = i * len(seed_list)
    end_idx = start_idx + len(seed_list)

    elements_for_K = mb_list[start_idx:end_idx]
    # values_for_specific_key = [d.final_dict["bic"] for d in elements_for_K]
    values_for_specific_key = [d.final_dict["bic"] for d in elements_for_K] # Given a specific K, select the seed with the lowest bic_sampling_p
    min_idx = values_for_specific_key.index(min(values_for_specific_key))
    min_idx = start_idx + min_idx
    
    lk = mb_list[min_idx].final_dict["final_likelihood"]
    bic = mb_list[min_idx].final_dict["bic"]
    icl = mb_list[min_idx].final_dict["icl"]
    print(f"k = {k}, seed = {seed_list[min_idx-start_idx]}: lk = {lk}, bic = {bic}")
    # print(f"k = {k}, seed = {seed_list[min_idx-start_idx]}: lk sampling p = {lk_sampling}, bic sampling = {bic_sampling}")
    lk_list.append(lk)
    bic_list.append(bic)
    icl_list.append(icl)

plt.figure()
plt.title("Likelihood over K")
plt.xlabel("K")
plt.ylabel("Likelihood")
plt.plot(K_list, lk_list)
plt.savefig(f"plots/{data_folder}/final_analysis/likelihood_over_K.png")
plt.close()

plt.figure()
plt.title("BIC over K")
plt.xlabel("K")
plt.ylabel("BIC")
plt.plot(K_list, bic_list)
plt.savefig(f"plots/{data_folder}/final_analysis/bic_over_K.png")
plt.close()

plt.figure()
plt.title("ICL over K")
plt.xlabel("K")
plt.ylabel("ICL")
plt.plot(K_list, icl_list)
plt.savefig(f"plots/{data_folder}/final_analysis/icl_over_K.png")
plt.close()

