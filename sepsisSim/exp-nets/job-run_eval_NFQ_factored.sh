#!/bin/bash

#SBATCH --job-name=run_eval_NFQ_factored
#SBATCH --output=./slurm_output/slurm-%j.out
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=8GB
#SBATCH -p standard
#SBATCH --time=4:00:00
#execute code

python run_eval-NFQ_factored.py --dir_data=$1;
