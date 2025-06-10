#!/bin/bash

#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 4:00:00

#SBATCH --mem=32G

#SBATCH --gres=gpu:rtxa5000:1

## Clear software
. ~/.bashrc
module purge
module load gcc

## - tell numpy,scipy that it can parallelize on the number of requested cores:
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd {{PROJECT_ROOT}}
. .venv/bin/activate

{{CMD}}
