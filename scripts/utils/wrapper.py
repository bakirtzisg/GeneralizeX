import os
import torch
import gymnasium as gym
import numpy as np
import torch.nn as nn

from typing import Dict
from time import strftime
from utils.util import load_agents
from utils.models import MLP

class MDP():
    def __init__(self, env: str, dir: str, policy: str, baseline_mode: bool = False, tasks='all', prefix: str = ''):
        self.env_name = env
        self.env = gym.make(self.env_name) 
        self.dir = dir
        self.policy = policy
        self.tasks = self.env.unwrapped.tasks if 'all' in tasks else tasks
        self.baseline = baseline_mode
        self.agent = load_agents(dir=self.dir, 
                                 policy=self.policy, 
                                 baseline=self.baseline, 
                                 tasks=self.tasks,
                                 prefix=prefix)
        if self.baseline:
            self.state_space_dim = {'baseline': self.env.unwrapped.observation_space.shape[0]}
            self.action_space_dim = {'baseline': self.env.unwrapped.action_space.shape[0]}
        else:
            self.state_space_dim = {f'{task}': self.env.unwrapped.observation_spaces[f'{task}'].shape[0] for task in self.tasks}
            self.action_space_dim = {f'{task}': self.env.unwrapped.action_spaces[f'{task}'].shape[0] for task in self.tasks}
        # self.horizon = self.env._max_episode_steps

        if isinstance(env, str):
            assert env.lower() in dir.lower(), 'Sanity check to make sure policy matches environment'
        
        assert all(task in self.env.unwrapped.tasks for task in self.tasks), 'Invalid tasks flag!'

class SequentialMDP():
    def __init__(self, mdps: MDP, subprocesses: MDP):
        pass

class MapsWrapper():
    def __init__(self, M: MDP, G: MDP, map_type: str, maps_dir: str, opt_params: Dict):
        self.f = {}
        self.g = {}

        self.f_optimizer = {}
        self.g_optimizer = {}

        self.maps_dir = maps_dir

        # Initialize maps (f,g) - parameterized by subtasks in the compositional structure G
        # Since G shares when to transition with M, this means that f depends on the structure of G
        # TODO: the two cases here are straight forward; how to generalize to arbitrary cases/dimensions?
        if map_type == 'linear':
            if M.baseline:
                self.f = {f'{task}': nn.Linear(M.state_space_dim, G.state_space_dim[f'{task}']) for task in G.tasks}
                self.g = {f'{task}': nn.Linear(M.action_space_dim, G.action_space_dim[f'{task}']) for task in G.tasks}
            elif len(M.tasks) == len(G.tasks) == 1:
                self.f[M.tasks[0]] = nn.Linear(M.state_space_dim[M.tasks[0]], G.state_space_dim[G.tasks[0]])
                self.g[M.tasks[0]] = nn.Linear(M.action_space_dim[M.tasks[0]], G.action_space_dim[G.tasks[0]]) 
        elif map_type == 'mlp':
            if M.baseline:
                self.f = {f'{task}': MLP(M.state_space_dim['baseline'], G.state_space_dim[f'{task}']) for task in G.tasks}
                self.g = {f'{task}': MLP(M.action_space_dim['baseline'], G.action_space_dim[f'{task}']) for task in G.tasks}
            elif len(M.tasks) == len(G.tasks) == 1:
                self.f[M.tasks[0]] = MLP(M.state_space_dim[M.tasks[0]], G.state_space_dim[G.tasks[0]])
                self.g[M.tasks[0]] = MLP(M.action_space_dim[M.tasks[0]], G.action_space_dim[G.tasks[0]])

        # Initialize optimizers and starting epochs - if continue training, load previously saved optimizer state dictionaries.
        if self.maps_dir is not None:
            self.save_path = self.maps_dir
            print(f'--- Loading {map_type} maps ---')

            self.f_checkpoint, self.g_checkpoint = self.load_state_dict(maps_dir=os.path.join(maps_dir))

            for task in G.tasks:
                self.f_optimizer[f'{task}'] = torch.optim.SGD(self.f[f'{task}'].parameters(), lr=opt_params['learning_rate'])
                self.g_optimizer[f'{task}'] = torch.optim.SGD(self.g[f'{task}'].parameters(), lr=opt_params['learning_rate'])
                self.f_optimizer[f'{task}'].load_state_dict(self.f_checkpoint[f'{task}']['optimizer_state_dict'])
                self.g_optimizer[f'{task}'].load_state_dict(self.g_checkpoint[f'{task}']['optimizer_state_dict'])

            self.f_epoch = self.f_checkpoint['epoch']
            self.g_epoch = self.g_checkpoint['epoch']
        else:
            self.save_path = os.path.join(os.path.curdir, f'results/maps/{map_type}/{strftime("%Y%m%d-%H%M%S")}-id-{np.random.randint(10000)}')

            print(f'--- Training {map_type} maps ---')

            for task in G.tasks:
                self.f_optimizer[f'{task}'] = torch.optim.SGD(self.f[f'{task}'].parameters(), lr=opt_params['learning_rate'])
                self.g_optimizer[f'{task}'] = torch.optim.SGD(self.g[f'{task}'].parameters(), lr=opt_params['learning_rate'])

            self.f_epoch = 0
            self.g_epoch = 0

        # Subtask training epoch
        self.epoch = {f'{task}': 0 for task in self.G.tasks}

    def train(self):
        # Set maps (f,g) to training mode
        for _, f_task in self.f.items(): 
            f_task.train()
        for _, g_task in self.g.items():
            g_task.train()
        
    def to(self, device=None):
        # Send maps (f,g) to torch device
        for _, f_task in self.f.items():
            f_task.to(device=device)
        for _, g_task in self.g.items():
            g_task.to(device=device)

    def eval(self):
        # Set maps (f,g) to eval mode
        for _, f_task in self.f.items(): 
            f_task.eval()
        for _, g_task in self.g.items():
            g_task.eval()

    def load_state_dict(self, maps_dir):
        # Return and load the state dict (from previous trained maps)
        f_checkpoint = torch.load(os.path.join(maps_dir, 'f.pt'))
        g_checkpoint = torch.load(os.path.join(maps_dir, 'g.pt'))
        self.f.load_state_dict(f_checkpoint['model_state_dict'])
        self.g.load_state_dict(g_checkpoint['model_state_dict'])
        
        return f_checkpoint, g_checkpoint
    
    def state_dict(self):
        # Return the torch state dictionaries for the maps (f,g) and optimizers (f_optimizer, g_optimizer)
        f_state_dict = {f'{task}': self.f[f'{task}'].state_dict() for task in self.G.tasks}
        g_state_dict = {f'{task}': self.g[f'{task}'].state_dict() for task in self.G.tasks}

        f_optimizer_state_dict = {f'{task}': self.f_optimizer[f'{task}'].state_dict() for task in self.G.tasks}
        g_optimizer_state_dict = {f'{task}': self.g_optimizer[f'{task}'].state_dict() for task in self.G.tasks}

        return f_state_dict, g_state_dict, f_optimizer_state_dict, g_optimizer_state_dict
    
    def save(self):
        # Save map and optimizer state dictionaries using torch.save()
        f_state_dict, g_state_dict, f_optimizer_state_dict, g_optimizer_state_dict = self.state_dict()

        torch.save({
                'epoch': {self.f_epoch + self.epoch[f'{task}'] for task in self.G.tasks},
                'model_state_dict': f_state_dict,
                'optimizer_state_dict': f_optimizer_state_dict,
               }, os.path.join(self.save_path, 'f.pt'))
    
        torch.save({
                'epoch': {self.g_epoch +self.epoch[f'{task}'] for task in self.G.tasks},
                'model_state_dict': g_state_dict,
                'optimizer_state_dict': g_optimizer_state_dict,
               }, os.path.join(self.save_path, 'g.pt'))