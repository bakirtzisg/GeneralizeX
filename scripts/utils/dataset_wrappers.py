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
        print('tasks', mdp.tasks)
        self.stats = rollout_mdp(mdp, eps=1, verbose=False, render=False, required_success=True)
        if self.policy.baseline:
            self.len = self.stats['rollout_0']['data'].shape[1]
        else:
            self.lengths = []
            for task in mdp.tasks:
                self.lengths.append(self.stats['rollout_0']['data'][f'{task}'].shape[1])
            self.len = sum(self.lengths)

        print('length', self.lengths)

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        # TODO: add subtask to stats['rollout_0']['data']! (why is task always reach?)        
        if self.policy.baseline:
            task = self.policy.tasks
            sample = torch.tensor(self.stats['rollout_0']['data'], dtype=torch.float32, requires_grad=True)
        else:
            # TODO: improve code...
            if self.lengths[0] + self.lengths[1] > idx >= self.lengths[0]:
                idx = idx % self.lengths[0]
                task = self.policy.tasks[1]
            elif self.lengths[0] + self.lengths[1] + self.lengths[2] > idx >= self.lengths[0] + self.lengths[1]:
                idx = idx % (self.lengths[0] + self.lengths[1])
                task = self.policy.tasks[2]
            else:
                task = self.policy.tasks[0]
            sample = torch.tensor(self.stats['rollout_0']['data'][f'{task}'], dtype=torch.float32, requires_grad=True)
        return sample[:,idx], task
    
class RobotDataset(Dataset):
    def __init__(self, mdp: MDP, file: str):
        self.policy = self.policy.env.unwrapped.current_task
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