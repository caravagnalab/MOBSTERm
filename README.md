- # MOBSTERm
  `MOBSTERm` is a Python package implementing a Bayesian model to perform multivariate subclonal deconvolution, allowing to detect neutral, private and selected clonal mutations in multi-region and longitudinal cancer datasets. The package also provides also a command-line interface.

  ### To install
  `pip install MOBSTERm`

  or 

  `pip install git+https://github.com/caravagnalab/MOBSTERm.git`

  

  #### Command-Line Interface

  The package installs a command-line tool named `MOBSTERm`. This tool takes as input a CSV file reporting, for each relevant mutation and for each sample,  the variant allele count (*NV*) and the total depth (*DP*) and clusterizes the mutations. The resulting clusters are saved in CSV files.

  

  For an example of usage, download the file [`test_data.csv`](https://raw.githubusercontent.com/albertocasagrande/MOBSTERm/refs/heads/main/data/test_data.csv) from the GitHub repository, and, in the download directory, use the command. 

  ```{bash}
   MOBSTERm test_data.csv deconvolution.csv
  ```

  This command will produce the CSV file `deconvolution.csv` reporting the cluster identifier of each mutation.

  Notice that only the columns "`mutation_id`", "`<Sample name>.NV`", and "`<Sample name>.DP`" are used during by `MOBSTERm`. The other columns are ignored. 

  To get a list of the options, use the option `-h`. 

  ```{bash}
  MOBSTERm -h
  ```

-----

#### Copyright and contacts

- Elena Rivaroli, Cancer Data Science (CDS) Laboratory.

[![CaravagnaLab GitHub](https://img.shields.io/badge/CDS%20Lab%20Github-caravagnalab-seagreen.svg)](https://github.com/caravagnalab/)

  

  

  

  
