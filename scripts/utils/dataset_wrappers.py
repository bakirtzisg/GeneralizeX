import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset

from eval import rollout_mdp
from utils.wrapper import MDP

class RobotDatasetLive(Dataset):
    '''
        Dataset collected from policy rollouts performed live (wrapped as a torch.utils.data.Dataset)
    '''
    def __init__(self, mdp: MDP):
        # Sample rollout info from MDP policy (PPO)
        self.policy = mdp

        task_success = 0
        while task_success == 0:
            stats = rollout_mdp(mdp, eps=1, verbose=False, render=False)
            task_success = stats['rollout_0']['is_success']
        self.task = mdp.env.unwrapped.current_task
        if mdp.baseline:
            self.sample = torch.tensor(stats['rollout_0']['data'], dtype=torch.float32, requires_grad=True)
        else:
            self.sample = torch.tensor(stats['rollout_0']['data'][f'{self.task}'], dtype=torch.float32, requires_grad=True)

    def __len__(self):
        return self.sample.size()[1]
    
    def __getitem__(self, idx):
        return self.sample[:,idx], self.task
    
class RobotDataset(Dataset):
    def __init__(self, mdp: MDP, file: str):
        self.policy = mdp
        self.data = np.load(file, allow_pickle=True).item()
        self.rollout_idx = np.random.randint(0, len(self.data.keys())-1)
        task = self.policy.env.unwrapped.current_task

        if self.policy.baseline:
            self.sample = torch.tensor(self.data[f'rollout_{self.rollout_idx}']['data'], dtype=torch.float32, requires_grad=True)
        else:
            self.sample = torch.tensor(
                self.data[f'rollout_{self.rollout_idx}']['data'][f'{task}'], dtype=torch.float32, requires_grad=True)
            
        self.task = self.data[f'rollout_{self.rollout_idx}']['subtask']

    def __len__(self):
        return self.sample.size()[1]

    def __getitem__(self, idx):
        return self.sample[:,idx], self.task[idx]