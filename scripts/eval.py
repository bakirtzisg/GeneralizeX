import os
import glob
import genx

import numpy as np
import gymnasium as gym 

from argparse import ArgumentParser
from stable_baselines3 import SAC

from utils.util import *
from utils.parser import CustomParser

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
                stats[f'rollout_{eps}']['is_success'] = 0
            
            stats[f'rollout_{eps}']['obs'] = np.array(q_buffer)
            stats[f'rollout_{eps}']['action'] =np.array(action_buffer)
            
            # reset buffers
            q_buffer = []
            action_buffer = []

            env.unwrapped.fresh_reset = False if len(tasks) == 1 else True
            obs, info = env.reset()
            eps = eps + 1
        
        # Debug: Check init_qpos is correct (TODO: move somewhere else)
        # env.unwrapped._env.robots[0].init_qpos = env.unwrapped._robot_init_qpos[env.unwrapped.current_task]
        # env.unwrapped._env.robots[0].reset(deterministic=True)
        # print(env.unwrapped._env.robots[0].init_qpos)
        
        
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
        # print(q)

        q_buffer.append(q) 
        action_buffer.append(info['action'])

        if render:
            env.render()

    stats['success_rate'] = (success / eval_eps) * 100

    env.close()
    return stats

if __name__ == '__main__':
    # Parser
    parser = CustomParser()
    args = parser.parse_args()
    # Constants
    ENV_NAME = args.env
    MODEL_DIR = args.dir
    EVAL_eps = 10
    TASKS_TO_CONSIDER = args.tasks
    BASELINE_MODE = 'baseline' in ENV_NAME.lower()

    assert ENV_NAME.lower() in MODEL_DIR.lower(), 'Sanity check to make sure policy matches environment'

    # Initialize environment and load task policies
    env = gym.make(ENV_NAME)

    # Specify which subset of tasks to consider
    TASKS = env.unwrapped.tasks if TASKS_TO_CONSIDER == 'all' else TASKS_TO_CONSIDER

    # Check that TASKS is a subset of the environment tasks
    assert all(task in env.unwrapped.tasks for task in TASKS), 'Invalid --tasks flag!'

    # Load policies
    agents = load_sac_policy(MODEL_DIR, BASELINE_MODE, TASKS)

    # Evaluate
    info = rollout(env, agents, baseline_mode=False, eval_eps=1, tasks=TASKS)
    print(info['rollout_0']['obs'])
    print(info['rollout_0']['action'])
    print(f'Success rate: {info["success_rate"]}')