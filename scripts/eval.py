import os
import glob
import genx

import numpy as np
import gymnasium as gym 

from argparse import ArgumentParser
from stable_baselines3 import SAC

from utils.util import *
from utils.wrapper import MDP

def rollout(env, agents, baseline_mode, eval_eps=10, tasks=None, render=True):
    # If only evaluating one subtask (instead of the entire composed task)
    if len(tasks) == 1:
        env.unwrapped.training_mode = True
        env.unwrapped.current_task = tasks[0]

    stats = {}
    done = False
    eps = 0
    success = 0
    q_buffer = []
    action_buffer = []
    reward_buffer = []
    env.reset()

    while eps < eval_eps:
        if done:
            stats[f'rollout_{eps}'] = {}
            if info['task_success']: 
                print("Episode success")
                stats[f'rollout_{eps}']['is_success'] = 1
                success += 1
            else:
                print('Episode failed')
                print(f'Subtask {env.unwrapped.current_task} success:', info['is_success'])
                stats[f'rollout_{eps}']['is_success'] = 0
            
            stats[f'rollout_{eps}']['obs'] = np.array(q_buffer)
            stats[f'rollout_{eps}']['action'] = np.array(action_buffer)
            stats[f'rollout_{eps}']['reward'] = np.array(reward_buffer)
            
            # data = concatenate the obs, actions, and rewards by timestep
            # np.shape = ((obs,action,reward),timesteps)
            stats[f'rollout_{eps}']['data'] = np.vstack((np.array(q_buffer).T,
                                                         np.array(action_buffer).T,
                                                         np.array(reward_buffer).T,
                                                         )) 
            
            # reset buffers
            q_buffer = []
            action_buffer = []
            reward_buffer = []

            env.unwrapped.fresh_reset = True if env.unwrapped.current_task == tasks[-1] else False
            # print('Current task?', env.unwrapped.current_task)
            # print('Fresh reset?', env.unwrapped.fresh_reset)
            obs, info = env.reset()
            eps = eps + 1
        
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
        # print(observation)

        q_buffer.append(q) 
        action_buffer.append(info['current_task_action'])
        reward_buffer.append(reward)

        if render:
            env.render()

    stats['success_rate'] = (success / eval_eps) * 100

    env.close()
    return stats

def rollout_mdp(M, eval_eps=1, render=True):
    ''' rollout using MDP wrapper '''
    assert isinstance(M, MDP)
    stats = rollout(M.env, M.agent, M.baseline, eval_eps=eval_eps, tasks=M.tasks, render=render)
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
    info = rollout_mdp(M, eval_eps=args.eps)
    print(info['rollout_0']['obs'])
    # print(info['rollout_0']['action'])
    # print(info['rollout_0']['reward'])
    # print(np.shape(info['rollout_0']['data']))
    print(f'Success rate: {info["success_rate"]}')