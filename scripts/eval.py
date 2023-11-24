import os
import glob
import genx

import numpy as np
import gymnasium as gym 

from argparse import ArgumentParser
from stable_baselines3 import SAC

from utils.util import *
from utils.parser import CustomParser

if __name__ == '__main__':
    # Parser
    parser = CustomParser()
    args = parser.parse_args()
    # Constants
    ENV_NAME = args.env
    MODEL_DIR = args.dir
    EVAL_EPS = 10
    TASKS_TO_CONSIDER = args.tasks
    BASELINE_MODE = 'baseline' in ENV_NAME.lower()

    assert ENV_NAME.lower() in MODEL_DIR.lower(), 'Sanity check to make sure policy matches environment'

    # Initialize environment and load task policies
    env = gym.make(ENV_NAME)

    # Specify which subset of tasks to consider
    TASKS = env.unwrapped.tasks if TASKS_TO_CONSIDER == 'all' else TASKS_TO_CONSIDER
    ONLY_ONE_TASK = len(TASKS) == 1

    # Check that TASKS is a subset of the environment tasks
    assert all(task in env.unwrapped.tasks for task in TASKS), 'Invalid --tasks flag!'

    # If only evaluating one subtask (instead of the entire composed task)
    if ONLY_ONE_TASK:
        env.unwrapped.training_mode = True
        env.unwrapped.current_task = TASKS[0]

    # Load policies
    agents = load_sac_policy(MODEL_DIR, BASELINE_MODE, TASKS)

    done = False
    EPS = 0
    success = 0
    previous_task = env.unwrapped.current_task
    env.reset()
    # Evaluate
    while EPS < EVAL_EPS:
        if done:
            EPS = EPS + 1
            if info['task_success']: 
                print("Episode success")
                success += 1
            else:
                print('Episode failed')
            env.unwrapped.fresh_reset = False if ONLY_ONE_TASK else True
            obs, info = env.reset()
        
        # print(env.unwrapped.current_task, env.unwrapped.training_mode)

        # Check init_qpos is correct
        # env.unwrapped._env.robots[0].init_qpos = env.unwrapped._robot_init_qpos[env.unwrapped.current_task]
        # env.unwrapped._env.robots[0].reset(deterministic=True)
        # print(env.unwrapped._env.robots[0].init_qpos)
        
        
        current_policy = agents['baseline'] if BASELINE_MODE else agents[env.unwrapped.current_task]

        # Since obs depends on subtask, retrieve obs again in case subtask has changed
        obs = env.unwrapped._get_obs()
      
        action, _states = current_policy.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        done = terminated or truncated

        # Get current joint positions 
        # observation = info['observation']
        # q_sin = observation['robot0_joint_pos_sin']
        # q_cos = observation['robot0_joint_pos_cos']
        # q = np.arctan2(q_sin, q_cos)
        # print(q)
    
        env.render()

    print(f'Success rate = {(success / EVAL_EPS) * 100}')