#!/bin/bash
#SBATCH --job-name=fit
#SBATCH --account=cdslab
#SBATCH --no-requeue
#SBATCH --cpus-per-task=12
#SBATCH -p THIN
#SBATCH --time=24:00:00
#SBATCH --mem=100gb
#SBATCH --output=./out_files/set7_delta_.out

python3 run_new_model.py

