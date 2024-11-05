#!/bin/bash
#SBATCH --job-name=real_all
#SBATCH --account=cdslab
#SBATCH --no-requeue
#SBATCH -N1
#SBATCH -n4
#SBATCH --cpus-per-task=9
#SBATCH -p EPYC
#SBATCH --time=24:00:00
#SBATCH --mem=200gb
#SBATCH --output=./real_all_samples.out

python3 run_mobster_orfeo_real.py

