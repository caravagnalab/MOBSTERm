# MOBSTERm
`MOBSTERm` is a package that implements a Bayesian model to perform multivariate subclonal deconvolution, allowing to detect neutral, private and selected clonal mutations in multi-region and longitudinal cancer datasets.

### To install
`pip install MOBSTERm`

or 

`git clone https://github.com/caravagnalab/MOBSTERm.git`

### Input data
MOBSTERm requires the following input:

- `NV` (`torch.Tensor`): Number of variant reads for each mutation (shape: `[num_mutations, num_samples]`).

- `DP` (`torch.Tensor`): Total read depth for each mutation (shape: `[num_mutations, num_samples]`).

- `mut_id` (`list` or `array`): Unique identifiers for each mutation.

- `num_iter` (`int`, default=`2000`): Number of SVI iterations for model fitting.

- `K` (`list`, default=`[]`): List of cluster numbers to consider (e.g., `[2,3,4]`).

- `purity` (`list`, default=`[1.,1., ...]`): Purity of the sample(s). It has to be one per sample.

- `kr` (`list`, default=`[1:1,1:1, 1:1, ...]`): Karyotype of the sample(s). It has to be one per sample (e.g., `[1:1,2:1]`).

- `seed_list` (`list`, default=`[123,1234]`): List of random seeds for reproducibility.

- `par_threshold` (`float`, default=`0.005`): Tolerance for parameter convergence. As ELBO oscillations are common in gradient based VI, we will monitor the convergence of all the parameters in the model, the inference stops when (abs(new-old) / abs(old)) < par_threshold for 200 consecutive iterations, for all the parameters.

- `loss_threshold` (`float`, default=`0.01`): Tolerance for loss convergence. As ELBO oscillations are common in gradient based VI we will monitor the convergence of the loss in the model, the inference stops when (abs(new_loss-old_loss) / abs(old_loss)) < loss_threshold for 200 consecutive iterations.

- `lr` (`float`, default=0.01): Learning rate for optimization.

- `sample_names` (`list`, default=None): Names of the samples. If None, default names `sample1, sample2, ...` are used.

#### Notes
- If `sample_names`, `purity` and `kr` are provided, their lengths must match the number of samples (`NV.shape[1]`).

- The function assumes NV and DP have the same shape (`N x D`, where `N` is the number of mutations and `D` is the number of samples).
