import torch

import numpy as np

from geomloss import SamplesLoss

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

from eval import rollout_mdp
from genx.environments.lift import CompLiftEnv
from .util import get_PPO_prob_dist

class RobotDataset(Dataset):
    '''
        Dataset collected from policy rollouts (wrapped as a torch.utils.data.Dataset)
    '''
    def __init__(self, mdp):
        # Sample rollout info from MDP policy (PPO)
        self.policy = mdp

        task_success = 0
        while task_success == 0:
            stats = rollout_mdp(mdp, eps=1, verbose=False, render=False)
            task_success = stats['rollout_0']['is_success']
        # TODO: add 'time' to stats['rollout_0']['data']?
        self.sample = torch.tensor(stats['rollout_0']['data'], dtype=torch.float32, requires_grad=True)

    def __len__(self):
        return self.sample.size()[1]
    
    def __getitem__(self, idx):
        return self.sample[:,idx]

def rlxBisimLoss(f, g, mdp_1, mdp_2, samples_1, device=torch.device('cuda:0')):
    '''
        Relaxed bisimulation loss 
        - 2-norm for rewards
        - Wasserstein *-distance for action distributions

        :param f: function f: S_1 -> S_2
        :param g: function g: A_1 -> A_2
        :param mdp_1: input MDP (e.g. compositional RL policy for lift task)
        :param mdp_2: compositional structure G
        :param samples_1: training samples from mdp_1 rollouts
    '''
    losses = torch.empty(len(samples_1), dtype=torch.float32)

    for i in range(len(samples_1)):
        # The first six columns of samples (stats['rollout_0']['data']) is the subtask observations
        # The next four columns of samples is the action vector (see _process_action in genx/environments/lift.py)
        # The last column is the reward
        state_1 = samples_1[:,0:6]
        
        # Save the means and standard deviations of the action distributions for each sampled subtask obs
        action_dists_1 = [
            get_PPO_prob_dist(mdp_1.agent[mdp_1.tasks[0]], obs.detach().numpy()) \
            for obs in state_1
        ]
        action_dists_2 = [
            get_PPO_prob_dist(mdp_2.agent[mdp_2.tasks[0]], f(obs.cuda()).cpu().detach().numpy()) \
            for obs in state_1
        ]

        # TODO: rewrite reward function with param state and world state? 
        # (obtainable from state_1 since it is subtask observation)
        # rwd_function = CompLiftEnv._evaluate_task() 
        def reach_rwd_function(obs):
            # TODO: move somewhere else? preferably into genx lift environment (changed from np to torch)
            reach_dist = torch.linalg.vector_norm(obs[3:] - obs[:3])
            task_reward = 1 - torch.tanh(10.0 * reach_dist)
            if reach_dist < 0.01:
                task_reward = 2
            return task_reward
        
        rewards_1 = samples_1[:,-1].to(device)
        rewards_2 = torch.tensor(
            [reach_rwd_function(f(obs.cuda())) for obs in state_1], 
            requires_grad=True,
            device=device,
        )

        rwd_dif = torch.sub(rewards_1, rewards_2)

        # Estimating 1-Wasserstein distance using debiased sinkhorn divergences (see geomloss)
        was_1_loss = SamplesLoss("sinkhorn", p=1)

        num_samples = 100 # TODO: change to at least 16 * max variance for confidence interval of 95%

        action_samples_1 = torch.empty((len(action_dists_1), num_samples, 3), device=device)
        for j, distribution in enumerate(action_dists_1):
            samples = distribution.rsample((num_samples,))
            action_samples_1[j] = torch.squeeze(samples)

        action_samples_2 = torch.empty((len(action_dists_2), num_samples, 3), device=device)
        for j, distribution in enumerate(action_dists_2):
            samples = distribution.rsample((num_samples,))
            action_samples_2[j] = torch.squeeze(samples)

        was_1_dist = was_1_loss(g(action_samples_1), action_samples_2)
        # was_2_dist = torch.sqrt((mu_1 - mu_2).pow(2) +  (sigma_1 - sigma_2).pow(2)) # TODO

        # Normalize reward loss components
        rwd_dif = torch.div(rwd_dif, 2) # max reward for reach is 2. TODO: generalize

        loss = torch.add(rwd_dif, was_1_dist)

        # Take maximum over sampled states
        losses[i] = torch.max(loss)
    
    return torch.mean(losses)

class MLP(nn.Module):
    def __init__(self, in_features: int, out_features: int, hidden_layer_dims: [int] = [7,7]):
        super().__init__()
        assert len(hidden_layer_dims) == 2, "There is 1 hidden layer, so len(hidden_layer_dims) == 2!"
        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden_layer_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_layer_dims[0], hidden_layer_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_layer_dims[1], out_features)
        )

    def forward(self, x):
        return self.layers(x)