import numpy as np
import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl

def add_data_specific_args(parent_parser):
    parser = parent_parser.add_argument_group("reward")
    parser.add_argument('--R_death', type=float, default=0.)
    parser.add_argument('--R_disch', type=float, default=100.)
    parser.add_argument('--R_immed', type=float, default=0.)
    return parent_parser

def remap_rewards(R, args):
    R = np.select([R == 0, R == -1, R == 1], [args.R_immed, args.R_death, args.R_disch,], R)
    return torch.tensor(R)

class EpisodicBuffer(Dataset):
    def __init__(self, state_dim, num_actions, horizon, buffer_size=0):
        self.max_size = int(buffer_size)
        self.horizon = horizon
        self.state = torch.zeros((self.max_size, horizon, state_dim))
        self.action = torch.zeros((self.max_size, horizon, 1), dtype=torch.long)
        self.subaction = torch.zeros((self.max_size, horizon, 2), dtype=torch.long)
        self.subactionvec = torch.zeros((self.max_size, horizon, 10))
        self.reward = torch.zeros((self.max_size, horizon, 1))
        self.not_done = torch.zeros((self.max_size, horizon, 1))
        self.subpibs = torch.zeros((self.max_size, horizon, 10))
        self.estm_subpibs = torch.zeros((self.max_size, horizon, 10))
    
    def __len__(self):
        return len(self.state)
    
    def __getitem__(self, idx):
        return (
            self.state[idx],
            self.action[idx],
            self.subaction[idx],
            self.subactionvec[idx],
            self.reward[idx],
            self.not_done[idx],
            self.subpibs[idx],
            self.estm_subpibs[idx],
        )
    
    def load(self, filename):
        data = torch.load(filename)
        self.state = data['statevecs'][:, :-1, :]
        self.action = data['actions'][:, 1:].unsqueeze(-1)  # Need to offset by 1 so that we predict actions that have not yet occurred
        self.subaction = data['subactions'][:, 1:, :]  # Need to offset by 1 so that we predict actions that have not yet occurred
        self.subactionvec = data['subactionvecs'][:, 1:, :]  # Need to offset by 1 so that we predict actions that have not yet occurred
        self.reward = data['rewards'][:, 1:].unsqueeze(-1)  # Need to offset by 1
        self.not_done = data['notdones'][:, 1:].unsqueeze(-1)
        self.subpibs = data['subpibs'][:, :-1, :]
        self.estm_subpibs = data['estm_subpibs'][:, :-1, :]
        print(f"Episodic Buffer loaded with {len(self)} episides.")


class SASRBuffer(object):
    def __init__(self, state_dim, num_actions, buffer_size=0):
        self.max_size = int(buffer_size)
        self.state = torch.zeros((self.max_size, state_dim))
        self.action = torch.zeros((self.max_size, num_actions), dtype=torch.long)
        self.subaction = torch.zeros((self.max_size, 2), dtype=torch.long)
        self.subactionvec = torch.zeros((self.max_size, 10), dtype=torch.long)
        self.next_state = torch.zeros((self.max_size, state_dim))
        self.reward = torch.zeros((self.max_size, 1))
        self.not_done = torch.zeros((self.max_size, 1))
        self.subpibs = torch.zeros((self.max_size, 10))

    def __len__(self):
        return len(self.state)
    
    def __getitem__(self, idx):
        return (
            self.state[idx],
            self.action[idx],
            self.subaction[idx],
            self.subactionvec[idx],
            self.next_state[idx],
            self.reward[idx],
            self.not_done[idx],
            self.subpibs[idx],
        )

    def load(self, filename):
        data = torch.load(filename)
        state, action, subaction, subactionvec, reward, not_done, subpibs, next_state = [], [], [], [], [], [], [], []
        for i in range(len(data['statevecs'])):
            lng = data['lengths'][i]
            state.append(data['statevecs'][i, :lng-1, :])
            action.append(data['actions'][i, 1:lng])  # Need to offset by 1 so that we predict actions that have not yet occurred
            subaction.append(data['subactions'][i, 1:lng, :])  # Need to offset by 1 so that we predict actions that have not yet occurred
            subactionvec.append(data['subactionvecs'][i, 1:lng, :])  # Need to offset by 1 so that we predict actions that have not yet occurred
            reward.append(data['rewards'][i, 1:lng])  # Need to offset by 1
            not_done.append(data['notdones'][i, 1:lng])
            subpibs.append(data['subpibs'][i, :lng-1, :])
            next_state.append(data['statevecs'][i, 1:lng, :])
        self.state = torch.cat(state)
        self.action = torch.cat(action).unsqueeze(1)
        self.subaction = torch.cat(subaction)
        self.subactionvec = torch.cat(subactionvec)
        self.reward = torch.cat(reward).unsqueeze(1)
        self.not_done = torch.cat(not_done).unsqueeze(1)
        self.subpibs = torch.cat(subpibs)
        self.next_state = torch.cat(next_state)
        print(f"Replay Buffer loaded with {len(self)} transitions.")
