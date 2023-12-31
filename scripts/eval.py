import os
import glob
import genx

import numpy as np
import gymnasium as gym 

from argparse import ArgumentParser
from stable_baselines3 import SAC

from utils.util import *
from utils.wrapper import MDP

def rollout(env, agents, baseline_mode, eps=1, tasks=None, verbose=True, render=True):
    assert all([task in env.unwrapped.tasks for task in tasks]), \
        f'Invalid tasks {tasks} for {env} environment'
    env.unwrapped.tasks = tasks # TODO: check that order is preserved 

    stats = {}
    done = False
    e = 0
    success = 0
    subtask_obs_buffer = []
    q_buffer = []
    action_buffer = []
    reward_buffer = []
    env.reset()

    while e < eps:
        if done:
            stats[f'rollout_{e}'] = {}
            if info['task_success']: 
                if verbose: print("Episode success")
                stats[f'rollout_{e}']['is_success'] = 1
                success += 1
            else:
                if verbose: print('Episode failed')
                print(f'Subtask {env.unwrapped.current_task} success:', info['is_success'])
                stats[f'rollout_{e}']['is_success'] = 0
            
            stats[f'rollout_{e}']['obs'] = np.array(q_buffer)
            stats[f'rollout_{e}']['subtask_obs'] = np.array(subtask_obs_buffer)
            stats[f'rollout_{e}']['action'] = np.array(action_buffer)
            stats[f'rollout_{e}']['reward'] = np.array(reward_buffer)
            
            # data = concatenate the obs, actions, and rewards by timestep
            # np.shape = ((obs,action,reward),timesteps)
            stats[f'rollout_{e}']['data'] = np.vstack((np.array(subtask_obs_buffer).T,
                                                       np.array(action_buffer).T,
                                                       np.array(reward_buffer).T,
                                                     ))     
            # reset buffers
            q_buffer = []
            subtask_obs_buffer = []
            action_buffer = []
            reward_buffer = []

            env.unwrapped.fresh_reset = True
            obs, info = env.reset()
            if len(tasks) == 1: env.unwrapped.current_task = tasks[0]
            e = e + 1
        
        # Debug: Check init_qpos is correct (TODO: move somewhere else)
        # env.unwrapped._env.robots[0].init_qpos = env.unwrapped._robot_init_qpos[env.unwrapped.current_task]
        # env.unwrapped._env.robots[0].reset(deterministic=True)
        # print(env.unwrapped._env.robots[0].init_qpos)
        
        # Force reset if current_task should not be evaluated
        if env.unwrapped.current_task not in tasks:
            done = True
            continue

        current_policy = agents['baseline'] if baseline_mode else agents[env.unwrapped.current_task]

        # Since obs depends on subtask, retrieve obs again in case subtask has changed
        obs = env.unwrapped._get_obs()
      
        action, _states = current_policy.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        done = terminated or truncated

        # Log robot joint positions 
        observation = info['observation']
        q_sin = observation['robot0_joint_pos_sin']
        q_cos = observation['robot0_joint_pos_cos']
        q = np.arctan2(q_sin, q_cos)

        q_buffer.append(q) 
        subtask_obs_buffer.append(info['current_task_obs'])
        action_buffer.append(info['current_task_action'])
        reward_buffer.append(reward)

        if render:
            env.render()

    stats['success_rate'] = (success / eps) * 100

    env.close()
    return stats

def rollout_mdp(M, eps=1, verbose=True, render=True):
    ''' rollout using MDP wrapper '''
    assert isinstance(M, MDP)
    stats = rollout(M.env, M.agent, M.baseline, eps=eps, tasks=M.tasks, verbose=verbose, render=render)
    return stats

if __name__ == '__main__':
    # Parser
    parser = ArgumentParser()
    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('--dir', type=str, required=True)
    parser.add_argument('--policy', type=str, required=True)
    parser.add_argument('--eps', type=int, default=1)
    parser.add_argument('--tasks', type=str, nargs='*', default='all')
    parser.add_argument('--prefix', type=str, default='*')
    args = parser.parse_args()

    BASELINE_MODE = 'baseline' in args.env.lower()
    # Test MDP wrapper
    M = MDP(env=args.env, 
            dir=args.dir, 
            policy=args.policy, 
            baseline_mode=BASELINE_MODE, 
            tasks=args.tasks,
            prefix=args.prefix,
        )

    # Evaluate
    info = rollout_mdp(M, eps=args.eps)
    print(info['rollout_0']['obs'])
    # print(info['rollout_0']['action'])
    # print(info['rollout_0']['reward'])
    # print(np.shape(info['rollout_0']['data']))
    # print(get_PPO_prob_dist(M.agent['reach'], info['rollout_0']['subtask_obs'][0]))
    print(f'Success rate: {info["success_rate"]}')