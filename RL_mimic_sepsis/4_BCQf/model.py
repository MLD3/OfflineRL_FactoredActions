import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import copy
import numpy as np
import argparse

all_subactions_vec = torch.tensor([
    #  Vaso        IV
    [1,0,0,0,0, 1,0,0,0,0],
    [0,1,0,0,0, 1,0,0,0,0],
    [0,0,1,0,0, 1,0,0,0,0],
    [0,0,0,1,0, 1,0,0,0,0],
    [0,0,0,0,1, 1,0,0,0,0],
    
    [1,0,0,0,0, 0,1,0,0,0],
    [0,1,0,0,0, 0,1,0,0,0],
    [0,0,1,0,0, 0,1,0,0,0],
    [0,0,0,1,0, 0,1,0,0,0],
    [0,0,0,0,1, 0,1,0,0,0],
    
    [1,0,0,0,0, 0,0,1,0,0],
    [0,1,0,0,0, 0,0,1,0,0],
    [0,0,1,0,0, 0,0,1,0,0],
    [0,0,0,1,0, 0,0,1,0,0],
    [0,0,0,0,1, 0,0,1,0,0],
    
    [1,0,0,0,0, 0,0,0,1,0],
    [0,1,0,0,0, 0,0,0,1,0],
    [0,0,1,0,0, 0,0,0,1,0],
    [0,0,0,1,0, 0,0,0,1,0],
    [0,0,0,0,1, 0,0,0,1,0],
    
    [1,0,0,0,0, 0,0,0,0,1],
    [0,1,0,0,0, 0,0,0,0,1],
    [0,0,1,0,0, 0,0,0,0,1],
    [0,0,0,1,0, 0,0,0,0,1],
    [0,0,0,0,1, 0,0,0,0,1.],
])

class BCQf_Net(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.q = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10), # vaso + iv
        )
        self.πb = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10), # vaso + iv
        )
    
    def forward(self, x):
        q_values = self.q(x)
        p_logits = self.πb(x)
        return q_values, F.log_softmax(p_logits, dim=-1), p_logits


class BCQf(pl.LightningModule):
    def __init__(
        self,
        *,
        state_dim,
        num_actions,
        hidden_dim,
        lr,
        weight_decay,
        threshold,
        discount,
        eval_discount,
        polyak_target_update=True,
        target_update_frequency=1,
        tau=0.005,
        target_value_clipping=False,
        Rmin=None,
        Rmax=None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self.Q = BCQf_Net(state_dim, num_actions, hidden_dim)
        self.Q_target = copy.deepcopy(self.Q)
        self.discount = discount
        self.num_actions = num_actions
        
        # Freeze target network so we don't accidentally train it
        for param in self.Q_target.parameters():
            param.requires_grad = False
        
        # Target update rule
        self.maybe_update_target = self.polyak_target_update if polyak_target_update else self.copy_target_update
        self.target_update_frequency = target_update_frequency
        self.tau = tau
        
        # Threshold for "unlikely" actions
        self.threshold = threshold
        
        # Discount for validation WIS OPE
        self.eval_discount = eval_discount or self.discount
        
        # Range of rewards/values for clipping
        self.hparams.vmin = self.hparams.Rmin / (1 - self.hparams.discount)
        self.hparams.vmax = self.hparams.Rmax / (1 - self.hparams.discount)
        
        # Number of training iterations
        self.iterations = 0
        
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("BCQ")
        parser.add_argument("--lr", type=float, default=3e-4)
        parser.add_argument("--weight_decay", type=float, default=1e-3)
        parser.add_argument("--threshold", type=float, default=0.3)
        parser.add_argument("--discount", type=float, default=0.99)
        parser.add_argument("--eval_discount", type=float, default=None)
        parser.add_argument("--hidden_dim", type=int, default=128)
        parser.add_argument('--target_value_clipping', default=False, action=argparse.BooleanOptionalAction)
        return parent_parser
    
    def forward(self, state):
        return self.Q(state)
    
    def configure_optimizers(self):
        self.all_subactions_vec = all_subactions_vec.to(self.device)
        self.Q_optimizer = torch.optim.Adam(self.Q.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        return self.Q_optimizer
    
    def training_step(self, batch, batch_idx):
        state, action, subaction, subactionvec, next_state, reward, notdone, subpibs = batch
        
        # Compute the target Q value
        with torch.no_grad():
            q, _, i = self.Q(next_state)
            imt = F.log_softmax(i.reshape(-1, 2, 5), dim=-1).exp()
            imt = (imt / imt.max(axis=-1, keepdim=True).values > self.threshold).float()
            
            # Factored action remapping
            q = q @ self.all_subactions_vec.T
            imt = torch.einsum('bi,bj->bji', imt[:,0,:], imt[:,1,:]).reshape(-1, 25) # both sub-action satisfy filter
            
            # Use large negative number to mask actions from argmax
            next_action = (imt * q + (1 - imt) * torch.finfo().min).argmax(axis=1, keepdim=True)

            q, _, _ = self.Q_target(next_state)
            q = q @ self.all_subactions_vec.T
            target_Q = reward + notdone * self.discount * q.gather(1, next_action).reshape(-1, 1)
            
            if self.hparams.target_value_clipping:
                target_Q = torch.clamp(target_Q, self.hparams.vmin, self.hparams.vmax)

        # Get current Q estimate
        current_Q, _, i = self.Q(state)
        imt = F.log_softmax(i.reshape(-1, 2, 5), dim=-1)
        current_Q = torch.bmm(current_Q.unsqueeze(1), subactionvec.unsqueeze(2)).squeeze(1)

        # Compute Q loss
        q_loss = F.smooth_l1_loss(current_Q, target_Q)
        i_loss = F.nll_loss(imt[:,0,:], subaction[:,0]) + F.nll_loss(imt[:,1,:], subaction[:,1])
        
        Q_loss = q_loss + i_loss + 1e-2 * i.pow(2).mean()

        # Optimize the Q
        self.Q_optimizer.zero_grad()
        self.manual_backward(Q_loss)
        self.Q_optimizer.step()

        # Update target network by polyak or full copy every X iterations.
        self.iterations += 1
        self.maybe_update_target()


    def polyak_target_update(self):
        for param, target_param in zip(self.Q.parameters(), self.Q_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def copy_target_update(self):
        if self.iterations % self.target_update_frequency == 0:
            self.Q_target.load_state_dict(self.Q.state_dict())


    def validation_step(self, batch, batch_idx):
        qvalues = self.offline_q_evaluation(batch)
        valid_wis, valid_ess = self.offline_evaluation(batch, weighted=True)
        self.log('iteration', int(self.iterations), prog_bar=True, logger=True)
        self.log('val_qvalues', qvalues, prog_bar=True, logger=True)
        self.log('val_wis', valid_wis, prog_bar=True, logger=True)
        self.log('val_ess', valid_ess, prog_bar=True, logger=True)
        return {
            'iteration': self.iterations,
            'val_qvalues': qvalues,
            'val_wis': valid_wis,
            'val_ess': valid_ess,
        }
    

    def offline_q_evaluation(self, eval_buffer):
        states, _, _, _, _, _, _, _ = eval_buffer
        states = states[:, 0, :]  # Only consider the initial states

        # Predict Q-values and Imitation probabilities
        q, _, i = self.Q(states)
        imt = F.log_softmax(i.reshape(-1, 2, 5), dim=-1).exp()
        imt = (imt / imt.max(axis=-1, keepdim=True).values > self.threshold).float()

        # Factored action remapping
        q = q @ self.all_subactions_vec.T
        imt = torch.einsum('bi,bj->bji', (imt[:,0,:], imt[:,1,:])).reshape(-1, 25)

        # Use large negative number to mask actions from argmax
        values = (imt * q + (1. - imt) * torch.finfo().min).max(axis=1).values
        return values.mean().item()


    def offline_evaluation(self, eval_buffer, weighted=True, eps=0.01):
        states, actions, subactions, subactionvecs, rewards, not_dones, subpibs, estm_subpibs = eval_buffer
        rewards = rewards[:, :, 0].cpu().numpy()
        n, horizon, _ = states.shape
        discounted_rewards = rewards * (self.eval_discount ** np.arange(horizon))

        ir = np.ones((n, horizon))
        for idx in range(n):
            lng = (not_dones[idx, :, 0].sum() + 1).item()  # all but the final transition has notdone==1

            # Predict Q-values and Imitation probabilities
            q, _, i = self.Q(states[idx])
            imt = F.log_softmax(i.reshape(-1, 2, 5), dim=-1).exp()
            imt = (imt / imt.max(axis=-1, keepdim=True).values > self.threshold).float()

            # Factored action remapping
            q = q @ self.all_subactions_vec.T
            imt = torch.einsum('bi,bj->bji', (imt[:,0,:], imt[:,1,:])).reshape(-1, 25)

            # Use large negative number to mask actions from argmax
            a_id = (imt * q + (1. - imt) * torch.finfo().min).argmax(axis=1).cpu().numpy()
            pie_soft = np.zeros((horizon, 25))
            estm_pibs = np.einsum('bi,bj->bji', estm_subpibs[idx][:,:5].cpu().numpy(), estm_subpibs[idx][:,5:].cpu().numpy()).reshape((-1, 25))
            pie_soft += eps * estm_pibs # Soften using training behavior policy
            pie_soft[range(horizon), a_id] += (1.0 - eps)

            # Compute importance sampling ratios
            a_obs = actions[idx, :, 0]
            ir[idx, :lng] = pie_soft[range(lng), a_obs[:lng].cpu().numpy()] / \
                (subpibs[idx, range(lng), a_obs[:lng] % 5].cpu().numpy() * subpibs[idx, range(lng), 5+a_obs[:lng] // 5].cpu().numpy())
            ir[idx, lng:] = 1  # Mask out the padded timesteps

        weights = np.clip(ir.cumprod(axis=1), 0, 1e3)
        if weighted:
            weights_norm = weights.sum(axis=0)
        else:
            weights_norm = weights.shape[0]
        weights /= weights_norm

        ess = (weights[:,-1].sum()) ** 2 / (((weights[:,-1]) ** 2).sum())
        estm = (weights[:,-1] * discounted_rewards.sum(axis=-1)).sum()

        return estm, ess
