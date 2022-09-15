#!/bin/bash

sbatch job.sh 'datagen-features-N=1e5-eps_1.ipynb';
sbatch job.sh 'datagen-features-N=1e5-eps_0_5.ipynb';
sbatch job.sh 'datagen-features-N=1e5-eps_0_1.ipynb';
sbatch job.sh 'datagen-features-N=1e5-suboptimal.ipynb';
sbatch job.sh 'datagen-features-N=1e5-suboptimal99.ipynb';
