import os
import glob
import random

import torch
import numpy as np

from stable_baselines3 import SAC, PPO
from stable_baselines3.common.policies import obs_as_tensor 
# from wrapper import MDP

def find_file(dir, prefix=''):
    file = glob.glob(os.path.join(dir, prefix))
    assert len(file) == 1, f'Found {len(file)} files in directory {os.path.join(dir, prefix)}!'

    return file[0]

def load_agents(dir, policy, baseline, tasks, prefix='', tb_path='', device=None):
    agents = {}
    if baseline:
        TASK_MODEL_FILE = find_file(dir, 'baseline_final*.zip')
        if policy == 'SAC':
            agents['baseline'] = SAC.load(TASK_MODEL_FILE)
        elif policy == 'PPO':
            agents['baseline'] = PPO.load(TASK_MODEL_FILE)
        else:
            raise RuntimeError('Invalid policy')
        print(f'Loaded baseline policy...')
    else:
        for task in tasks:
            TASK_MODEL_FILE = find_file(dir, f'{task}/{prefix}.zip')
            if policy == 'SAC':
                agents[task] = SAC.load(TASK_MODEL_FILE)
            elif policy == 'PPO':
                agents[task] = PPO.load(TASK_MODEL_FILE)
            else:
                raise RuntimeError('Invalid policy')
            print(f'Loaded {task} {policy} policy...')
    return agents

def get_PPO_prob_dist(agent, obs):
    assert isinstance(agent, PPO)
    if not isinstance(obs, torch.Tensor):
        obs = agent.policy.obs_to_tensor(obs)[0]
    dist = agent.policy.get_distribution(obs)

    assert isinstance(dist.distribution, torch.distributions.normal.Normal)
    # probs = dist.log_prob
    # probs_np = probs.detach().numpy()

    return dist.distribution.loc, dist.distribution.scale