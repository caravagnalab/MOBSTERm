#!/bin/bash
#SBATCH --job-name=set
#SBATCH --account=cdslab
#SBATCH --no-requeue
#SBATCH -N1
#SBATCH -n4
#SBATCH --cpus-per-task=8
#SBATCH -p EPYC
#SBATCH --time=24:00:00
#SBATCH --mem=200gb
#SBATCH --output=./out_files/set7.out

python3 run_new_model.py

