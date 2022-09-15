import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dir_data', type=str)
args = parser.parse_args()
print(args)

import os
os.makedirs('./results/{}/'.format(args.dir_data), exist_ok=True)

import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
import itertools

PROB_DIAB = 0.2
NSTEPS = 20     # max episode length

# Ground truth MDP model
MDP_parameters = joblib.load('../data/MDP_parameters.joblib')
P_ = MDP_parameters['transition_matrix_absorbing'] # (A, S, S_next)
R_ = MDP_parameters['reward_matrix_absorbing_SA'] # (S, A)
nS, nA = R_.shape
γ = gamma = 0.99

# unif rand isd, mixture of diabetic state
isd = joblib.load('../data/prior_initial_state_absorbing.joblib')

# Optimal value function and optimal return
V_star = joblib.load('../data/V_π_star_PE.joblib')
J_star = V_star @ isd

# Make features for all states
X_ALL_states = []
for arrays in itertools.product(
    [[1,0], [0,1]], # Diabetic
    [[1,0,0], [0,1,0], [0,0,1]], # Heart Rate
    [[1,0,0], [0,1,0], [0,0,1]], # SysBP
    [[1,0], [0,1]], # Percent O2
    [[1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1]], # Glucose
    [[1,0], [0,1]], # Treat: AbX
    [[1,0], [0,1]], # Treat: Vaso
    [[1,0], [0,1]], # Treat: Vent
):
    X_ALL_states.append(np.concatenate(arrays))

X_ALL_states = np.array(X_ALL_states)
X_ALL_states.shape

import tensorflow as tf
from tensorflow import keras
from tf_utils import select_output_d, select_output
from OPE_utils_keras import *

print('Load FQI models')
Ns = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000]
runs = list(range(10))
nl = 1
nh = 1000
lr = 1e-3
k_list = list(range(50))

keys_list = list(itertools.product(Ns, runs, k_list))
len(keys_list)

print('Ground-truth performance')
true_value_dict = {}
for N, run, k in tqdm(keys_list):
    try:
        save_dir = './output/N={},run{}/{}/NFQ-clipped-keras.models.nl={},nh={},lr={}/'.format(N, run, args.dir_data, nl, nh, lr)
        hidden_net = keras.models.load_model('{}/iter={}.hidden_net'.format(save_dir, k), compile=False)
        Q_pred = hidden_net.predict(X_ALL_states)
        π_pred = convert_to_policy_table(Q_pred)
        true_value = isd @ policy_eval_analytic(P_.transpose((1,0,2)), R_, π_pred, gamma)
        true_value_dict[N, run, k] = true_value
        joblib.dump(true_value_dict, './results/{}/eval_NFQ.joblib'.format(args.dir_data))
    except:
        continue

joblib.dump(true_value_dict, './results/{}/eval_NFQ.joblib'.format(args.dir_data))
