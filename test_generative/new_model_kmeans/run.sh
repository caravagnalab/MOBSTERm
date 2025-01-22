#!/bin/bash
#SBATCH --job-name=test_gen
#SBATCH --account=cdslab
#SBATCH --no-requeue
#SBATCH -N1
#SBATCH -n4
#SBATCH --cpus-per-task=8
#SBATCH -p EPYC
#SBATCH --time=24:00:00
#SBATCH --mem=200gb
#SBATCH --output=./out_files/test_gen_new.out

python3 run_generative.py --N 1000 --K 3 --D 3

