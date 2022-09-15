from test_tube import Experiment, HyperOptArgumentParser, SlurmCluster
import time

import pytorch_lightning as pl
from torch.utils.data import DataLoader

import argparse
import numpy as np
from data import EpisodicBuffer, SASRBuffer
from data import add_data_specific_args, remap_rewards
from model import BCQf
from utils import MyTrainingEpochLoop

# Problem-specific hyperparameters
state_dim = 64
num_actions = 25
num_subactions = 10
horizon = 20

def main(args, cluster):
    # wait for Slurm file writes to propagate
    time.sleep(15)
    
    # Seed & Logging
    pl.seed_everything(args.seed)
    logger = pl.loggers.TestTubeLogger("logs", name="mimic_dBCQf")
    
    # Load training and validation data
    train_buffer = SASRBuffer(state_dim, num_actions)
    train_buffer.load('../data/episodes+encoded_state+knn_pibs_factored/train_data.pt')
    val_episodes = EpisodicBuffer(state_dim, num_actions, horizon)
    val_episodes.load('../data/episodes+encoded_state+knn_pibs_factored/val_data.pt')

    # Remap reward from [0,-1,1] to [R_immed, R_death, R_disch]
    train_buffer.reward = remap_rewards(train_buffer.reward, args)
    val_episodes.reward = remap_rewards(val_episodes.reward, args)
    Rmin = float(np.min(train_buffer.reward.cpu().numpy()))
    Rmax = float(np.max(train_buffer.reward.cpu().numpy()))
    
    # Data loaders for train and val
    train_buffer_loader = DataLoader(train_buffer, batch_size=100, shuffle=True)
    val_episodes_loader = DataLoader(val_episodes, batch_size=len(val_episodes), shuffle=False)

    # Create model
    policy = BCQf(
        state_dim=state_dim,
        num_actions=num_subactions,
        Rmin=Rmin,
        Rmax=Rmax,
        **vars(args),
    )
    
    # Create trainer with custom validation logic
    trainer = pl.Trainer(
        gpus=0,
        logger=logger,
        val_check_interval=args.eval_frequency,
        callbacks=[
            pl.callbacks.ModelCheckpoint(save_top_k=-1, filename='{step}'),
            pl.callbacks.ProgressBar(100),
        ],
    )
    my_epoch_loop = MyTrainingEpochLoop(min_steps=0, max_steps=args.max_steps)
    my_epoch_loop.connect(batch_loop=trainer.fit_loop.epoch_loop.batch_loop, val_loop=trainer.fit_loop.epoch_loop.val_loop)
    trainer.fit_loop.connect(epoch_loop=my_epoch_loop)
    
    # Run trainer
    trainer.fit(
        policy,
        train_buffer_loader,
        val_episodes_loader,
    )

def optimize_on_cluster(hyperparams):
    # enable cluster training
    cluster = SlurmCluster(
        hyperparam_optimizer=hyperparams,
        log_path='./slurm/',
    )

    # email for cluster coms
    cluster.notify_job_status(email='tangsp@umich.edu', on_done=False, on_fail=False)

    # configure cluster
    cluster.per_experiment_nb_nodes = 1
    cluster.job_time = '2:00:00'
    cluster.memory_mb_per_node = 0

    cluster.add_slurm_cmd(cmd='partition', value='standard', comment='')
    cluster.add_slurm_cmd(cmd='mem', value='4GB', comment='')
    cluster.add_slurm_cmd(cmd='cpus-per-task', value='1', comment='')

    # run hopt
    # creates and submits jobs to slurm
    cluster.optimize_parallel_cluster_cpu(
        main,
        nb_trials=None,
        job_name='mimic_sepsis_dBCQf_job',
    )

if __name__ ==  '__main__':
    # Hyperparameter search grid
    parser = HyperOptArgumentParser(strategy='grid_search')
    parser = add_data_specific_args(parser)
    parser.add_argument('--max_steps', type=int, default=10_000)  # Max number of training steps
    parser.add_argument('--eval_frequency', type=int, default=100) # Check validation every N training steps
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--eval_discount", type=float, default=1.0)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument('--target_value_clipping', default=False, action=argparse.BooleanOptionalAction)
    parser.opt_list("--seed", type=int, tunable=True, options=[0,1,2,3,4])
    parser.opt_list("--threshold", type=float, tunable=True, options=[0.0, 0.01, 0.05, 0.1, 0.3, 0.5, 0.75, 0.9999])
    hparams = parser.parse_args()
    
    optimize_on_cluster(hparams)
