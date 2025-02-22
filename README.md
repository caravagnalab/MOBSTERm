# MOBSTERm
`MOBSTERm` is a package that implements a Bayesian model to perform multivariate subclonal deconvolution, allowing to detect neutral, private and selected clonal mutations in multi-region and longitudinal cancer datasets.

### To install
`pip install MOBSTERm`

or 

`git clone https://github.com/caravagnalab/MOBSTERm.git`

### Input data
MOBSTERm requires two tensors as input:
- `NV` - number of variant reads that cover a given somatic mutation.
- `DP` - depth at the corresponding locus.
 
Both tensors have a shape of `N x D`, where `N` is the number of mutations and `D` is the number of samples.
