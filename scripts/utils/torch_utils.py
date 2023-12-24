import torch

import numpy as np

from geomloss import SamplesLoss

from torch.utils.data import Dataset, DataLoader

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
            stats = rollout_mdp(mdp, eps=1, render=False)
            task_success = stats['rollout_0']['is_success']
        self.sample = torch.tensor(stats['rollout_0']['data'], dtype=torch.float32, requires_grad=True)

    def __len__(self):
        return self.sample.size()[1]
    
    def __getitem__(self, idx):
        return self.sample[:,idx]

def rlxBisimLoss(f_star, g, mdp_1, mdp_2, samples_1, samples_2):
    '''
        Relaxed bisimulation loss 
        - 2-norm for rewards
        - Wasserstein *-distance for action distributions

        :param f_star: 
        :param g:
        :param mdp_1: input MDP (e.g. compositional RL policy for lift task)
        :param mdp_2: compositional structure G
        :param samples_1: 
        :param samples_2:   
    '''
    assert len(samples_1) == len(samples_2), "Lengths of training data should be the same!"
    losses = torch.empty(len(samples_1), dtype=torch.float32)

    for i in range(len(samples_1)):
        # The first six entities of samples (stats['rollout_0']['data']) is the subtask observations
        # The next seven entities of samples is the action vector (7D vector of applied torques)
        # The last entity is the reward
        state_1 = samples_1[0:6,:]
        state_2 = samples_2[0:6,:]

        print('dist_1', get_PPO_prob_dist(mdp_1.agent[mdp_1.tasks[0]], state_1))

        # Save the means and standard deviations of the action distributions for each sampled subtask obs
        action_dists_1 = [get_PPO_prob_dist(mdp_1.agent[mdp_1.tasks[0]], obs) for obs in state_1]
        action_dists_2 = [get_PPO_prob_dist(mdp_2.agent[mdp_2.tasks[0]], obs) for obs in state_2]

        # rwd_function = CompLiftEnv._evaluate_task()
        rewards_1 = samples_1[-1,:]
        rewards_2 = samples_2[-1,:]

        # TODO: try torch.linalg.vector_norm(rwd_fcn(state_1) - rwd_fcn(g(state_1))
        rwd_dif = torch.linalg.vector_norm(rewards_1-rewards_2)

        # TODO: what is the wasserstein dist for normal distributions?
        was_dist = SamplesLoss("sinkhorn", p=1)
        
        # losses = -(rwd_dif + was_dist(f_star(action_dists_1(s)), action_dists_2(g(s))))
        losses[i] = -(rwd_dif)
    
    return torch.mean(losses)
