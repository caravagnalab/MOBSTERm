#!/bin/bash
#SBATCH --job-name=new_model
#SBATCH --account=cdslab
#SBATCH --no-requeue
#SBATCH -N1
#SBATCH -n1
#SBATCH -c24
#SBATCH -p GPU
#SBATCH --gpus=1
#SBATCH --time=18:00:00
#SBATCH --mem=200gb
#SBATCH --output=./new_model.out

python3 run_new_model.py

