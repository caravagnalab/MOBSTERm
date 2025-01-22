#!/bin/bash
#SBATCH --job-name=test_gen
#SBATCH --account=cdslab
#SBATCH --no-requeue
#SBATCH --cpus-per-task=12
#SBATCH -p THIN
#SBATCH --time=24:00:00
#SBATCH --mem=50gb
#SBATCH --output=./out_files/test_%a.out
#SBATCH --array=23


echo $SLURM_ARRAY_TASK_ID

N=$(awk -F',' "NR==${SLURM_ARRAY_TASK_ID} { print \$1; exit }" params_config.txt)
K=$(awk -F',' "NR==${SLURM_ARRAY_TASK_ID} { print \$2; exit }" params_config.txt)
D=$(awk -F',' "NR==${SLURM_ARRAY_TASK_ID} { print \$3; exit }" params_config.txt)
purity=$(awk -F',' "NR==${SLURM_ARRAY_TASK_ID} { print \$4; exit }" params_config.txt)
coverage=$(awk -F',' "NR==${SLURM_ARRAY_TASK_ID} { print \$5; exit }" params_config.txt)

echo $N
echo $K
echo $D
echo $purity
echo $coverage

python3 run_generative.py --N $N --K $K --D $D --p $purity --cov $coverage