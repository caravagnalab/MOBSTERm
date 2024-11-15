#!/bin/bash
#SBATCH --job-name=new_model
#SBATCH --account=cdslab
#SBATCH --no-requeue
#SBATCH -N1
#SBATCH -n4
#SBATCH --cpus-per-task=9
#SBATCH -p EPYC
#SBATCH --time=24:00:00
#SBATCH --mem=200gb
#SBATCH --output=./new_model2.out

python3 run_new_model.py

