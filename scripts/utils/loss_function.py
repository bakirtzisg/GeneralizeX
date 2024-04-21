import torch
import torch.nn as nn
from typing import Dict

from geomloss import SamplesLoss
from utils.wrapper import MDP
from utils.util import get_PPO_prob_dist

# TODO: implement Amy Zhang's bisimulation loss

def epsilonRlxBisimLoss(f: nn.Module, g: nn.Module, 
                 mdp_1: MDP, mdp_2: MDP, 
                 task: str, samples_1: torch.Tensor,
                 device=torch.device('cuda:0')) -> torch.Tensor:
    '''
        Epsilon-relaxed bisimulation loss 
        - 2-norm for rewards
        - Wasserstein *-distance for action distributions

        :param f: function f: S_1 -> S_2
        :param g: function g: A_1 -> A_2
        :param mdp_1: input MDP (e.g. compositional RL policy for lift task)
        :param mdp_2: compositional structure G
        :param samples_1: training samples from mdp_1 rollouts
    '''
    assert(task in mdp_1.tasks)
    losses = torch.empty(len(samples_1), dtype=torch.float32)

    for i in range(len(samples_1)):
        # The first "state_space_dim" columns of samples are the subtask observations
        # The next four columns of samples is the action vector (see _process_action in genx/environments/lift.py)
        # The last column is the reward
        if mdp_1.baseline:
            state_space_dim = mdp_1.state_space_dim['baseline'] 
            action_space_dim = mdp_1.action_space_dim['baseline']
        else:
            state_space_dim = mdp_1.state_space_dim[f'{task}']
            action_space_dim = mdp_1.action_space_dim[f'{task}']
        
        state_1 = samples_1[:,0:state_space_dim]
        action_1 = samples_1[:,state_space_dim:state_space_dim+action_space_dim]
        
        # Save the means and standard deviations of the action distributions for each sampled subtask obs
        # TODO: generalize to other tasks
        agent_1 = mdp_1.agent['baseline'] if mdp_1.baseline else mdp_1.agent[task]
        action_dists_1 = [
            get_PPO_prob_dist(agent_1, obs.detach().numpy()) \
            for obs in state_1
        ]
        action_dists_2 = [
            get_PPO_prob_dist(mdp_2.agent[task], f(obs.cuda()).cpu().detach().numpy()) \
            for obs in state_1
        ]

        # TODO: rewrite reward function with param state and world state? 
        # (obtainable from state_1 since it is subtask observation)
        # rwd_function = CompLiftEnv._evaluate_task() 
        def lift_rwd_function(obs, subtask):
        # TODO: move somewhere else? preferably into genx lift environment (changed from np to torch)
            if subtask == 'reach':
                hand_pos = obs[0:3]
                cube_pos = obs[3:6]
                reach_dist = torch.linalg.norm(cube_pos - hand_pos)
                task_reward = 1 - torch.tanh(10.0 * reach_dist)
                if reach_dist < 0.01:
                    task_reward = 2.0
            elif subtask == 'grasp':
                # task_reward = 1 - torch.tanh(10.0 * reward_criteria['reach_dist'])
                # if reward_criteria['reach_dist'] < 0.01:
                #     task_reward = 2.0
                # TODO
                task_reward = 0.0
            elif subtask == 'lift':
                cube_z_pos = obs[1]
                task_reward = 0.0
                table_height = mdp_1.env.unwrapped._env.model.mujoco_arena.table_offset[2]
                lift_height = table_height + 0.05
                if cube_z_pos > lift_height: 
                    task_reward = 2.25
            else:
                raise RuntimeError("Invalid subtask for compositional lift environment")
            return task_reward
        
        rewards_1 = samples_1[:,-1].to(device)
        rewards_2 = torch.tensor(
            [lift_rwd_function(f(obs.cuda()), task) for obs in state_1], 
            requires_grad=True,
            device=device,
        )

        rwd_dif = torch.sub(rewards_1, rewards_2)

        # Estimating 1-Wasserstein distance using debiased sinkhorn divergences (see geomloss)
        was_1_loss = SamplesLoss("sinkhorn", p=1)

        num_samples = 100 # TODO: change to at least 16 * max variance for confidence interval of 95%

        action_samples_1 = torch.empty((len(action_dists_1), num_samples, action_space_dim), device=device)
        for j, distribution in enumerate(action_dists_1):
            samples = distribution.rsample((num_samples,))
            action_samples_1[j] = torch.squeeze(samples)

        action_samples_2 = torch.empty((len(action_dists_2), num_samples, mdp_2.action_space_dim[f'{task}']), device=device)
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