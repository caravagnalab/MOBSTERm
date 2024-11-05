#!/bin/bash
#SBATCH --job-name=races_data
#SBATCH --account=cdslab
#SBATCH --no-requeue
#SBATCH -N1
#SBATCH -n1
#SBATCH -c24
#SBATCH -p GPU
#SBATCH --gpus=1
#SBATCH --time=18:00:00
#SBATCH --mem=200gb
#SBATCH --output=./races_longitudinal.out

python3 run_mobster_orfeo_RACES.py

