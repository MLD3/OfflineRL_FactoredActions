#!/bin/bash

#SBATCH --job-name=neuralFQI
#SBATCH --output=./slurm_output/slurm-%j.out
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4GB
#SBATCH -p standard
#SBATCH --time=24:00:00
#execute code


N=$1
name=$2
run=$3
layers=$4
units=$5
lr=$6
dir_in=$7
dir_out=$8

python run-NFQ.py --input_dir="../datagen/$dir_in/" --output_dir="./output/N=$N,run$run/$dir_out/" --N=$N --run=$run --num_hidden_layers=$layers --num_hidden_units=$units --learning_rate=$lr;
