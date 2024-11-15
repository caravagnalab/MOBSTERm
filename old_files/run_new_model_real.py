import new_model_real as model_mobster_mv
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
ylim = 3000
data_folder = 'real_data_new_model_hlr2'

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

# Plot the dataset
plt.figure()
plt.xlim([0,1])
plt.ylim([0,1])
plt.scatter(NV[:,0]/DP[:,0], NV[:,1]/DP[:,1])

plt.xlabel('Set7_55')
plt.ylabel('Set7_57')
plt.title('Set7_55 vs Set7_57')
plt.title("Original data")
plt.savefig(f'plots/{data_folder}/original_data1.png')
plt.close()

plt.figure()
plt.xlim([0,1])
plt.ylim([0,1])
plt.scatter(NV[:,0]/DP[:,0], NV[:,2]/DP[:,2])

plt.xlabel('Set7_55')
plt.ylabel('Set7_59')
plt.title('Set7_55 vs Set7_59')
plt.title("Original data")
plt.savefig(f'plots/{data_folder}/original_data2.png')
plt.close()

plt.figure()
plt.xlim([0,1])
plt.ylim([0,1])
plt.scatter(NV[:,0]/DP[:,0], NV[:,3]/DP[:,3])

plt.xlabel('Set7_55')
plt.ylabel('Set7_62')
plt.title('Set7_55 vs Set7_62')
plt.title("Original data")
plt.savefig(f'plots/{data_folder}/original_data3.png')
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
plt.suptitle("Marginals")
axes[0].set_xlim([0,1])
axes[0].set_ylim([0,ylim])
axes[0].hist(NV[:,0].numpy()/DP[:,0].numpy(), bins = 50)
# axes[0].set_title("Sample 1")
axes[0].set_title("Set7_55")

axes[1].set_xlim([0,1])
axes[1].set_ylim([0,ylim])
axes[1].hist(NV[:,1].numpy()/DP[:,1].numpy(), bins = 50)
# axes[1].set_title("Sample 2")
axes[1].set_title("Set7_57")
plt.savefig(f'plots/{data_folder}/marginals.png')
plt.close()



save = True
seed_list = [40,41,42]
K_list = [9,10,11,12,13,14]
# K_list = [5,6,7,8,9]
best_K, best_seed =  model_mobster_mv.fit(NV, DP, num_iter = 3000, K = K_list, seed = seed_list, lr = 0.01, savefig = save, data_folder = data_folder)

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

