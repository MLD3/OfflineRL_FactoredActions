import numpy as np
import pandas as pd
import itertools
import copy
from tqdm import tqdm
import random as python_random

import joblib
from joblib import Parallel, delayed

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--N', type=int)
parser.add_argument('--run', type=int)
parser.add_argument('--num_hidden_layers', type=int, default=1)
parser.add_argument('--num_hidden_units', type=int, default=1000)
parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--max_iterations', type=int, default=50)
args = parser.parse_args()
print(args)

run_idx_length = 10_000

gamma = 0.99
nS, nA = 1442, 8
d = 21
num_epoch = args.max_iterations

NSTEPS = 20
PROB_DIAB = 0.2
DISCOUNT = 1
USE_BOOSTRAP=True
N_BOOTSTRAP = 100

def load_sparse_features(fname):
    feat_dict = joblib.load('{}/{}'.format(args.input_dir, fname))
    INDS_init, X, A, X_next, R = feat_dict['inds_init'], feat_dict['X'], feat_dict['A'], feat_dict['X_next'], feat_dict['R']
    return INDS_init, X.toarray(), A, X_next.toarray(), R

print('Loading data ... ', end='')
if args.N <= 10000:
    loads = [1]
elif args.N <= 20000:
    loads = [1,3]
elif args.N <= 30000:
    loads = [1,3,5]
elif args.N <= 40000:
    loads = [1,3,5,7]
elif args.N <= 50000:
    loads = [1,3,5,7,9]
else:
    raise NotImplementedError

X, A, X_next, R = [], [], [], []
for i, idx in enumerate(loads):
#     df_features_tr = pd.read_csv('{}/{}-features.csv'.format(args.input_dir, idx))
    trINDS_init, trX, trA, trX_next, trR = load_sparse_features('{}-21d-feature-matrices.sparse.joblib'.format(idx))
    first_ind = trINDS_init[args.run*run_idx_length]
    last_ind = trINDS_init[args.run*run_idx_length+(run_idx_length if i+1<len(loads) else args.N-i*run_idx_length)]
    iX, iA, iX_next, iR = trX[first_ind:last_ind], trA[first_ind:last_ind], trX_next[first_ind:last_ind], trR[first_ind:last_ind]
    X.append(iX)
    A.append(iA)
    X_next.append(iX_next)
    R.append(iR)

X = np.vstack(X)
A = np.concatenate(A)
R = np.concatenate(R)
X_next = np.vstack(X_next)

print('DONE')
print()

# Factored action space specification
nAs = np.array([2,2,2])
dA = len(nAs)
assert nA == np.product(nAs)
dA1 = dA + 1

def convert_factored_action(a, nAj_all):
    # nAj_all: cardinality of sub-action spaces
    subactions = []
    for j in range(len(nAj_all)):
        _A_j = nAj_all[j]
        a_j = a % _A_j
        subactions.append(a_j)
        a = a // _A_j
    return subactions

A_fac = np.hstack([np.ones((len(A),1)), np.array(convert_factored_action(A, nAs)).T])


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

import tensorflow as tf
from tensorflow import keras

def init_networks():
    # Inputs
    state_input = keras.Input(shape=(d), name='state_input')
    action_input = keras.Input(shape=(dA1), name='action_input')

    # Layers
    hidden_layers = keras.Sequential([
        *[keras.layers.Dense(args.num_hidden_units, activation="relu") for _ in range(args.num_hidden_layers)],
        keras.layers.Dense(dA1),
    ], name='hidden_layers')

    action_combination_layer = keras.layers.Dot(axes=1, name='action_combination')

    # Outputs
    hidden_output = hidden_layers(state_input)
    Q_output = action_combination_layer([hidden_output, action_input])

    # Models
    hidden_net = keras.Model(inputs=[state_input], outputs=[hidden_output], name='hidden_net')
    Q_net = keras.Model(inputs=[state_input, action_input], outputs=[Q_output], name='Q_net')
    Q_net.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=keras.losses.MeanSquaredError(),
        metrics=keras.metrics.MeanSquaredError(),
    )
    return hidden_net, Q_net

fit_args = dict(
    batch_size=64, 
    validation_split=0.1, 
    epochs=100, 
    callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=10, restore_best_weights=True)]
)


print('FQI')

import pathlib
save_dir = '{}/NFQ_factored_v2-clipped-keras.models.nl={},nh={},lr={}/'.format(args.output_dir, args.num_hidden_layers, args.num_hidden_units, args.learning_rate)
pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)

try:
    Q_net = keras.models.load_model('{}/iter={}.Q_net'.format(save_dir, num_epoch))
except:
    print("\033[1m\033[31mFQI iteration", 0, "\033[0m\033[0m")
    np.random.seed(0)
    python_random.seed(0)
    tf.random.set_seed(0)
    
    hidden_net, Q_net = init_networks()
    Q_net.fit([X, A_fac], np.zeros_like(R), **fit_args)
    Q_net.save('{}/iter={}.Q_net'.format(save_dir, 0))
    hidden_net.save('{}/iter={}.hidden_net'.format(save_dir, 0))

    for k in range(num_epoch):
        print("\033[1m\033[31mFQI iteration", k+1, "\033[0m\033[0m")
        y = R + gamma * np.clip(hidden_net.predict(X_next), 0, np.inf).sum(axis=1)
        y = np.clip(y, -1, 1)
        tf.random.set_seed(0)
        hidden_net, Q_net = init_networks()
        Q_net.fit([X, A_fac], y, **fit_args)
        Q_net.save('{}/iter={}.Q_net'.format(save_dir, k+1))
        hidden_net.save('{}/iter={}.hidden_net'.format(save_dir, k+1))
