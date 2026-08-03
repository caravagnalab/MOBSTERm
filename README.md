# MOBSTERm
  `MOBSTERm` is a Python package implementing a Bayesian model to perform multivariate subclonal deconvolution, allowing to detect neutral, private and selected clonal mutations in multi-region and longitudinal cancer datasets. The package also provides also a command-line interface.

## To install

### From PyPI

​```{bash}
pip install MOBSTERm
​```

### From source
1. Clone the repository and enter the directory:
   
​```{bash}
git clone https://github.com/caravagnalab/MOBSTERm.git
cd MOBSTERm
​```

2. Install the package and its dependencies:
   
​```{bash}
pip install .
​```

3. Verify the installation:

​```{bash}
MOBSTERm -h
​```

## Running MOBSTERm
There are two ways to use `MOBSTERm`: through the command-line interface, or directly in Python via the `fit` function.

### Command-Line Interface

The package installs a command-line tool named `MOBSTERm`. This tool takes as input a CSV file reporting, for each relevant mutation and for each sample,  the variant allele count (*NV*) and the total depth (*DP*) and clusterizes the mutations. The resulting clusters are saved in CSV files.

For an example of usage, download the file [`test_data.csv`](https://raw.githubusercontent.com/albertocasagrande/MOBSTERm/refs/heads/main/data/test_data.csv) from the GitHub repository, and, in the download directory, use the command. 

```{bash}
 MOBSTERm test_data.csv deconvolution.csv -c "6,7,8"
```

This command will produce the CSV file `deconvolution.csv` reporting the cluster identifier of each mutation.

Notice that only the columns "`mutation_id`", "`<Sample name>.NV`", and "`<Sample name>.DP`" are used during by `MOBSTERm`. The other columns are ignored. 

To get a list of the options, use the option `-h`. 

```{bash}
MOBSTERm -h
```

#### Command-line options

Run `MOBSTERm -h` for the full list of options. The most commonly used ones are:

- `-c`, `--cluster-list` (default=`2,3,4`): List of cluster numbers to consider.

- `num_iter` (`int`, default=`2000`): Maximum number of SVI iterations for model fitting.

- `seed_list` (`list` of `int`, default=`[123,1234]`): List of random seeds for reproducibility.

- `-s`, `--samples` (default=`None`): Names of the samples to be processed.

- `-p`, `--purity` (default=`1,1,...`): Purity of the sample(s).

- `-k`, `--karyotypes` (default=`1:1,1:1,...`): Karyotype of the sample(s).

-----

### Using the `fit` function

Besides the command-line interface, `MOBSTERm` can be used directly in Python by calling the `fit` function. See the notebook [`test.ipynb`](test.ipynb) for an example of usage.

#### Input data

`MOBSTERm.fit` requires the following input:

- `NV` (`numpy.ndarray`): Variant allele count for each mutation and sample (shape: `[num_mutations, num_samples]`).

- `DP` (`numpy.ndarray`): Total read depth for each mutation and sample (shape: `[num_mutations, num_samples]`).

- `mut_id` (`list` of `str`, default=`None`): Identifiers for each mutation.

- `num_iter` (`int`, default=`2000`): Maximum number of SVI iterations for model fitting.

- `K` (`list` of `int`, default=`[]`): Number of clonal/subclonal clusters to consider (e.g., `[2,3,4]`).

- `purity` (`list` of `float`, default=`None`): Previously estimated purity of the tumor sample(s), one per sample. If `None`, purity is set to `1.` for every sample.

- `kr` (`list` of `str`, default=`None`): Copy-number state of the sample(s), one per sample, in the form `'major_allele:minor_allele'` (e.g., `['1:1','2:1']`). If `None`, defaults to `'1:1'` for every sample.

- `seed_list` (`list` of `int`, default=`[123,1234]`): List of random seeds for reproducibility.

- `par_threshold` (`float`, default=`0.005`): Tolerance for parameter convergence. As ELBO oscillations are common in gradient-based VI, we monitor the convergence of all the parameters in the model; inference stops when `abs(new-old) / abs(old) < par_threshold` for 200 consecutive iterations, for all the parameters.

- `loss_threshold` (`float`, default=`0.01`): Tolerance for loss convergence. As ELBO oscillations are common in gradient-based VI we monitor the convergence of the loss in the model; inference stops when `abs(new_loss-old_loss) / abs(old_loss) < loss_threshold` for 200 consecutive iterations.

- `lr` (`float`, default=`0.01`): Learning rate for optimization.

- `savefig` (`bool`, default=`False`): If `True`, saves output figures to `data_folder`.

- `data_folder` (`str`, default=`None`): Path to the directory where results or figures should be saved.

- `sample_names` (`list` of `str`, default=`None`): Names of the samples. If `None`, default names `sample1, sample2, ...` are used.

- `quiet` (`bool`, default=`False`): If `True`, suppresses progress logs and output messages.

- `num_of_threads` (`int`, default=`1`): Number of parallel threads to use during computation. Use `-1` to use all available CPUs.

#### Notes
- If `sample_names`, `purity` and `kr` are provided, their lengths must match the number of samples (`NV.shape[1]`).

- `NV` and `DP` must have the same shape (`N x D`, where `N` is the number of mutations and `D` is the number of samples).

-----

#### Copyright and contacts

- Elena Rivaroli, Cancer Data Science (CDS) Laboratory.

[![CaravagnaLab GitHub](https://img.shields.io/badge/CDS%20Lab%20Github-caravagnalab-seagreen.svg)](https://github.com/caravagnalab/)

  

  

  

  
