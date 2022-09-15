#!/bin/bash

#SBATCH --job-name=data-prep
#SBATCH --output=./slurm_output/slurm-%j.out
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=30GB
#SBATCH -p standard
#SBATCH --time=6:00:00
#execute code

jupyter nbconvert --ExecutePreprocessor.timeout=-1 --to notebook --execute --inplace $1;
