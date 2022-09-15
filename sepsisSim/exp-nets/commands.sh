#!/bin/bash


################################################################################


#### Eps=0.1, N=50-10000
# FQI
for run in {0..9}; do for N in 50 100 200 500 1000 2000 5000 10000;
do sbatch job-NFQ.sh $N _ $run 1 1000 '1e-3' 'eps_0_1-100k' 'eps_0_1';
done; done

# FQI_factored
for run in {0..9}; do for N in 50 100 200 500 1000 2000 5000 10000;
do sbatch job-NFQ_factored.sh $N _ $run 1 1000 '1e-3' 'eps_0_1-100k' 'eps_0_1';
done; done


#### Eps=0.1, N=15000-50000

# FQI
for run in {0..9}; do for N in 15000 20000 25000 30000 35000 40000 45000 50000;
do sbatch job-NFQ.sh $N _ $run 1 1000 '1e-3' 'eps_0_1-100k' 'eps_0_1';
done; done

# FQI_factored
for run in {0..9}; do for N in 15000 20000 25000 30000 35000 40000 45000 50000;
do sbatch job-NFQ_factored.sh $N _ $run 1 1000 '1e-3' 'eps_0_1-100k' 'eps_0_1';
done; done


#### Eps=0.1, Eval
sbatch job-run_eval_NFQ.sh 'eps_0_1';
sbatch job-run_eval_NFQ_factored.sh 'eps_0_1';


################################################################################


#### Eps=0.5, N=50-10000

# FQI
for run in {0..9}; do for N in 50 100 200 500 1000 2000 5000 10000;
do sbatch job-NFQ.sh $N _ $run 1 1000 '1e-3' 'eps_0_5-100k' 'eps_0_5';
done; done

# FQI_factored
for run in {0..9}; do for N in 50 100 200 500 1000 2000 5000 10000;
do sbatch job-NFQ_factored.sh $N _ $run 1 1000 '1e-3' 'eps_0_5-100k' 'eps_0_5';
done; done


#### Eps=0.5, N=15000-50000

# FQI
for run in {0..9}; do for N in 15000 20000 25000 30000 35000 40000 45000 50000;
do sbatch job-NFQ.sh $N _ $run 1 1000 '1e-3' 'eps_0_5-100k' 'eps_0_5';
done; done

# FQI_factored
for run in {0..9}; do for N in 15000 20000 25000 30000 35000 40000 45000 50000;
do sbatch job-NFQ_factored.sh $N _ $run 1 1000 '1e-3' 'eps_0_5-100k' 'eps_0_5';
done; done


#### Eps=0.5, Eval
sbatch job-run_eval_NFQ.sh 'eps_0_5';
sbatch job-run_eval_NFQ_factored.sh 'eps_0_5';


################################################################################


#### Unif Eps=1, N=50-10000

# FQI
for run in {0..9}; do for N in 50 100 200 500 1000 2000 5000 10000;
do sbatch job-NFQ.sh $N _ $run 1 1000 '1e-3' 'eps_1-100k' 'eps_1';
done; done

# FQI_factored
for run in {0..9}; do for N in 50 100 200 500 1000 2000 5000 10000;
do sbatch job-NFQ_factored.sh $N _ $run 1 1000 '1e-3' 'eps_1-100k' 'eps_1';
done; done

#### Unif Eps=1, N=15000-50000

# FQI
for run in {0..9}; do for N in 15000 20000 25000 30000 35000 40000 45000 50000;
do sbatch job-NFQ.sh $N _ $run 1 1000 '1e-3' 'eps_1-100k' 'eps_1';
done; done

# FQI_factored
for run in {0..9}; do for N in 15000 20000 25000 30000 35000 40000 45000 50000;
do sbatch job-NFQ_factored.sh $N _ $run 1 1000 '1e-3' 'eps_1-100k' 'eps_1';
done; done


#### Unif Eps=1, Eval
sbatch job-run_eval_NFQ.sh 'eps_1';
sbatch job-run_eval_NFQ_factored.sh 'eps_1';


################################################################################


#### Suboptimal Eps=8/7, N=50-10000
# FQI
for run in {0..9}; do for N in 50 100 200 500 1000 2000 5000 10000;
do sbatch job-NFQ.sh $N _ $run 1 1000 '1e-3' 'suboptimal-100k' 'suboptimal';
done; done

# FQI_factored
for run in {0..9}; do for N in 50 100 200 500 1000 2000 5000 10000;
do sbatch job-NFQ_factored.sh $N _ $run 1 1000 '1e-3' 'suboptimal-100k' 'suboptimal';
done; done


#### Suboptimal Eps=8/7, N=15000-50000

# FQI
for run in {0..9}; do for N in 15000 20000 25000 30000 35000 40000 45000 50000;
do sbatch job-NFQ.sh $N _ $run 1 1000 '1e-3' 'suboptimal-100k' 'suboptimal';
done; done

# FQI_factored
for run in {0..9}; do for N in 15000 20000 25000 30000 35000 40000 45000 50000;
do sbatch job-NFQ_factored.sh $N _ $run 1 1000 '1e-3' 'suboptimal-100k' 'suboptimal';
done; done


#### Suboptimal Eps=8/7, Eval
sbatch job-run_eval_NFQ.sh 'suboptimal';
sbatch job-run_eval_NFQ_factored.sh 'suboptimal';


################################################################################


#### Suboptimal99 Eps=8/7*0.99, N=50-10000
# FQI
for run in {0..9}; do for N in 50 100 200 500 1000 2000 5000 10000;
do sbatch job-NFQ.sh $N _ $run 1 1000 '1e-3' 'suboptimal99-100k' 'suboptimal99';
done; done

# FQI_factored
for run in {0..9}; do for N in 50 100 200 500 1000 2000 5000 10000;
do sbatch job-NFQ_factored.sh $N _ $run 1 1000 '1e-3' 'suboptimal99-100k' 'suboptimal99';
done; done


#### Suboptimal99 Eps=8/7*0.99, N=15000-50000

# FQI
for run in {0..9}; do for N in 15000 20000 25000 30000 35000 40000 45000 50000;
do sbatch job-NFQ.sh $N _ $run 1 1000 '1e-3' 'suboptimal99-100k' 'suboptimal99';
done; done

# FQI_factored
for run in {0..9}; do for N in 15000 20000 25000 30000 35000 40000 45000 50000;
do sbatch job-NFQ_factored.sh $N _ $run 1 1000 '1e-3' 'suboptimal99-100k' 'suboptimal99';
done; done


#### Suboptimal Eps=8/7, Eval
sbatch job-run_eval_NFQ.sh 'suboptimal99';
sbatch job-run_eval_NFQ_factored.sh 'suboptimal99';


################################################################################

exit;


#### Example of running the python scripts directly

export run=0
mkdir -p "./output/run$run/unif-10k/"

python run-NFQ.py --input_dir='../datagen/unif-100k/' --output_dir="./output/run$run/unif-10k/" --N=10000 --run=$run --num_hidden_layers=1 --num_hidden_units=100 --learning_rate='1e-3'
python run-NFQ_factored.py --input_dir='../datagen/unif-100k/' --output_dir="./output/run$run/unif-10k/" --N=100 --run=$run --num_hidden_layers=1 --num_hidden_units=100 --learning_rate='1e-3'


python run-NFQE-clipped-keras-split-k.py --input_dir='../datagen/unif-100k/' --output_dir="./output/run$run/unif-10k/" --N=10000 --split='va' --run=$run --num_hidden_layers=1 --num_hidden_units=100 --learning_rate='1e-3' --model_k=1
python run-WIS-AM-models.py --input_dir='../datagen/unif-100k/' --output_dir="./output/run$run/unif-10k/" --N=10000 --split='va' --run=$run
python run-OPE.py --input_dir='../datagen/unif-100k/' --output_dir="./output/run$run/unif-10k/" --N=10000 --split='va' --run=$run
