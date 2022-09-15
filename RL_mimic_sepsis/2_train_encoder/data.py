import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl

train_data_file = '../data/episodes/train_set.pt'
val_data_file = '../data/episodes/val_set.pt'
test_data_file = '../data/episodes/test_set.pt'

minibatch_size = 128
observation_dim = 33
context_dim = 5
num_actions = 25

def load_datasets():
    train_data = torch.load(train_data_file)
    train_dataset = TensorDataset(
        train_data['demographics'], 
        train_data['observations'], 
        train_data['actionvecs'], 
        train_data['lengths'], 
    )

    val_data = torch.load(val_data_file)
    val_dataset = TensorDataset(
        val_data['demographics'], 
        val_data['observations'], 
        val_data['actionvecs'], 
        val_data['lengths'], 
    )

    test_data = torch.load(test_data_file)
    test_dataset = TensorDataset(
        test_data['demographics'], 
        test_data['observations'], 
        test_data['subactionvecs'], 
        test_data['lengths'], 
    )
    
    return train_dataset, val_dataset, test_dataset

class MIMIC3SepsisDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.train, self.val, self.test = load_datasets()
        self.observation_dim = observation_dim
        self.context_dim = context_dim
        self.num_actions = num_actions

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=minibatch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=minibatch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=minibatch_size, shuffle=False)
