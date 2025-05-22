# import new_model_integration as model_mobster_mv
# import new_model as model_mobster_mv
import model_mobster as model_mobster_mv
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
import time

data_folder = 'paper_new/set7_new_test'

"""
data = pd.read_csv("./data/real_data/Set7_mutations.csv")

sets = [55, 57, 59, 62]
s_number = 7
purity = [0.88, 0.88, 0.88, 0.8]

data = pd.read_csv("./data/real_data/Set6_mutations.csv")

sets = [42, 44, 45, 46, 47, 48]
s_number = 6
purity = [0.66, 0.72, 0.80, 0.80, 0.80, 0.80]


NV_list = []
DP_list = []

for s in sets:
    NV = torch.tensor(data[f'Set{s_number}_{s}.NV'].to_numpy())
    DP = torch.tensor(data[f'Set{s_number}_{s}.DP'].to_numpy())
    
    NV_list.append(NV.view(-1, 1))  # Ensure correct shape
    DP_list.append(DP.view(-1, 1))  # Ensure correct shape

NV = torch.cat(NV_list, dim=1)
DP = torch.cat(DP_list, dim=1)
K_list = [15,16,17,18,19,20,21,22,23,24]
"""
"""
data = pd.read_csv("./data/hitchhikers/new_hitchhikers.csv")
sets = ['Sample.A', 'Sample.B']

NV_list = []
DP_list = []

for s in sets:
    print(s)
    NV = torch.tensor(data[f'{s}.NV'].to_numpy())
    DP = torch.tensor(data[f'{s}.DP'].to_numpy())
    
    NV_list.append(NV.view(-1, 1))  # Ensure correct shape
    DP_list.append(DP.view(-1, 1))  # Ensure correct shape

NV = torch.cat(NV_list, dim=1)
DP = torch.cat(DP_list, dim=1)
purity = [1,1]
print(NV.shape, DP.shape)
K_list = [4,5,6,7,8,9,10,11]
"""
""""""

data = pd.read_csv("./data/real_data/Set7_mutations.csv")
columns_to_check = ["Set7_55.NV", "Set7_57.NV", "Set7_59.NV"]
data = data[~(data[columns_to_check] == 0).all(axis=1)]

# data.to_csv("./data/real_data/Set7_55_57_59.csv", index=False)

sets = [55, 57, 59]
s_number = 7

NV_list = []
DP_list = []

for s in sets:
    NV = torch.tensor(data[f'Set{s_number}_{s}.NV'].to_numpy())
    DP = torch.tensor(data[f'Set{s_number}_{s}.DP'].to_numpy())
    
    NV_list.append(NV.view(-1, 1)) 
    DP_list.append(DP.view(-1, 1))

NV = torch.cat(NV_list, dim=1)
DP = torch.cat(DP_list, dim=1)
vaf = NV/DP
# cond = np.where((vaf[:,0] == 0) & (vaf[:,1] == 0) & (vaf[:,2] == 0))[0]
# NV = np.delete(NV, cond, axis=0)
# DP = np.delete(DP, cond, axis=0)
purity = [0.88, 0.88, 0.88]
K_list = [16,17,18,19,20,21,22,23]

"""
data = pd.read_csv("./data/gbm_B7R7.csv")

sets = ['primary', 'relapse']

NV_list = []
DP_list = []

for s in sets:
    NV = torch.tensor(data[f'NV_{s}'].to_numpy())
    DP = torch.tensor(data[f'DP_{s}'].to_numpy())
    
    NV_list.append(NV.view(-1, 1))  # Ensure correct shape
    DP_list.append(DP.view(-1, 1))  # Ensure correct shape
NV = torch.cat(NV_list, dim=1)
DP = torch.cat(DP_list, dim=1)
K_list = [4,5,6,7,8,9,10,11]

"""
"""

NV = pd.read_csv("./data/hitchhikers/NV_bigger2.csv")
DP = pd.read_csv("./data/hitchhikers/DP_bigger2.csv")

NV = torch.tensor(NV.values)
DP = torch.tensor(DP.values)
K_list = [4,5,6,7,8,9,10,11]
"""

"""
data = pd.read_csv("./data/joint_table.csv")

sets = ['Pre', 'RL']

NV_list = []
DP_list = []

for s in sets:
    NV = torch.tensor(data[f'NV.{s}'].to_numpy())
    DP = torch.tensor(data[f'DP.{s}'].to_numpy())
    
    NV_list.append(NV.view(-1, 1))  # Ensure correct shape
    DP_list.append(DP.view(-1, 1))  # Ensure correct shape
NV = torch.cat(NV_list, dim=1)
DP = torch.cat(DP_list, dim=1)
K_list = [6,7,8,9,10,11,12,13]
purity = [1,1]
"""
"""
NV = pd.read_csv("./data/admixing/NV_admixing4.csv")
DP = pd.read_csv("./data/admixing/DP_admixing4.csv")

NV = torch.tensor(NV.values)
DP = torch.tensor(DP.values)
K_list = [7,8,9,10,11,12]
purity = [1,1]
"""
"""
# data = pd.read_csv('/Users/elenarivaroli/orfeo_remote/scratch/tesimagistrale/subclonal_deconvolution_mv_examples/hitchhiker_mirage_longitudinal_bigger/hitchhikers_bigger.csv')
# data = pd.read_csv('/Users/elenarivaroli/orfeo_remote/scratch/tesimagistrale/subclonal_deconvolution_mv_examples/admixing4/admixing4.csv')
# data = pd.read_csv('./data/hitchhikers/hitchhikers_bigger.csv')
data = pd.read_csv('../subclonal_deconvolution_mv_examples/hitchhiker_mirage_final2/hitchhikers_final2.csv')
sets = ['Sample.A', 'Sample.B']
# sets = ['Sample.B', 'Sample.A']

NV_list = []
DP_list = []

for s in sets:
    print(s)
    NV = torch.tensor(data[f'{s}.NV'].to_numpy())
    DP = torch.tensor(data[f'{s}.DP'].to_numpy())
    
    NV_list.append(NV.view(-1, 1))  # Ensure correct shape
    DP_list.append(DP.view(-1, 1))  # Ensure correct shape

NV = torch.cat(NV_list, dim=1)
DP = torch.cat(DP_list, dim=1)
purity = [1,1]
K_list = [4,5,6,7,8,9,10,11,12]
"""
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

    ax.scatter(x, y, alpha=0.7, s = 20)

    ax.set_title(f"Sample {i+1} vs Sample {j+1}")
    ax.set_xlabel(f"Sample {i+1}")
    ax.set_ylabel(f"Sample {j+1}")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    idx += 1

plt.show()
plt.savefig(f'plots/{data_folder}/orig_data.png')
plt.close()

fig, axes = plt.subplots(1, D, figsize=(5*D, 4))
plt.suptitle("Marginals")
n_bins = int(np.ceil(np.sqrt(len(vaf))))
for i in range(D):
    x = vaf[:, i].numpy()
    x = x[x > 0]
    axes[i].hist(x, bins = n_bins)    
    
    axes[i].set_xlabel(f"Sample {i+1}")
    axes[i].set_xlim([0,1])

plt.show()
plt.savefig(f'plots/{data_folder}/marginals.png')
plt.close()

save = True
seed_list = [41,42]
# K_list = [15,16,17,18,19,20,21,22]
# K_list = [4,5,6,7,8,9,10,11]
# Record the start time
start_time = time.time()

_, best_K, best_seed = model_mobster_mv.fit(NV, DP, num_iter=2000, K=K_list, seed=seed_list, lr=0.01, savefig=save, data_folder=data_folder, purity = purity)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Time taken for the fit function: {elapsed_time:.2f} seconds")


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
    mb_list.append(model_mobster_mv.mobster_MV(seed = seed))
    mb_list[i].__dict__ = loaded_list[i]
    # print(f'K = {mb_list[i].K}, seed = {mb_list[i].seed}')

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
    values_for_specific_key = [d.final_dict["icl"] for d in elements_for_K] # Given a specific K, select the seed with the lowest bic_sampling_p
    min_idx = values_for_specific_key.index(min(values_for_specific_key))
    min_idx = start_idx + min_idx
    
    lk = mb_list[min_idx].final_dict["final_likelihood"]
    bic = mb_list[min_idx].final_dict["bic"]
    icl = mb_list[min_idx].final_dict["icl"]
    print(f"k = {k}, seed = {seed_list[min_idx-start_idx]}: lk = {lk}, icl = {icl}")
    # print(f"k = {k}, seed = {seed_list[min_idx-start_idx]}: lk sampling p = {lk_sampling}, bic sampling = {bic_sampling}")
    lk_list.append(lk)
    bic_list.append(bic)
    icl_list.append(icl)

plt.figure(figsize=(20, 6))

# Subplot 1: Likelihood over K
plt.subplot(1, 3, 1)
plt.title("Likelihood over K")
plt.xlabel("K")
plt.ylabel("Likelihood")
plt.plot(K_list, lk_list, label="Likelihood", color='black')
# Scatter all points (other than the minimum one) in black
plt.scatter(K_list, lk_list, color='black', zorder=5)  
# Find the index of the minimum value and highlight it in red
min_index_lk = np.argmax(lk_list)
plt.scatter(K_list[min_index_lk], lk_list[min_index_lk], color='black', zorder=10)  
plt.legend()
plt.grid(True, color='gray', linestyle='-', linewidth=0.2)  

# Subplot 2: BIC over K
plt.subplot(1, 3, 2)
plt.title("BIC over K")
plt.xlabel("K")
plt.ylabel("BIC")
plt.plot(K_list, bic_list, label="BIC", color='black')
# Scatter all points (other than the minimum one) in black
plt.scatter(K_list, bic_list, color='black', zorder=5)  
# Find the index of the minimum value and highlight it in red
min_index_bic = np.argmin(bic_list)
plt.scatter(K_list[min_index_bic], bic_list[min_index_bic], color='r', zorder=10)  
plt.legend()
plt.grid(True, color='gray', linestyle='-', linewidth=0.2)

# Subplot 3: ICL over K
plt.subplot(1, 3, 3)
plt.title("ICL over K")
plt.xlabel("K")
plt.ylabel("ICL")
plt.plot(K_list, icl_list, label="ICL", color='black')
# Scatter all points (other than the minimum one) in black
plt.scatter(K_list, icl_list, color='black', zorder=5)  
# Find the index of the minimum value and highlight it in red
min_index_icl = np.argmin(icl_list)
plt.scatter(K_list[min_index_icl], icl_list[min_index_icl], color='r', zorder=10)  
plt.legend()
plt.grid(True, color='gray', linestyle='-', linewidth=0.2)  

plt.tight_layout()

plt.savefig(f"plots/{data_folder}/final_analysis/metrics_over_K.png")
plt.close()

