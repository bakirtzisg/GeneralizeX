import os
import glob
import genx

import numpy as np
import gymnasium as gym 

from argparse import ArgumentParser
from stable_baselines3 import SAC

from utils.util import find_file

if __name__ == '__main__':
    # Parser
    parser = ArgumentParser()
    parser.add_argument('--env', type=str, default='BaselineCompPickPlaceCan-Panda')
    parser.add_argument('--dir', type=str, default='')
    parser.add_argument('--tasks', type=str, nargs='*', default='all')
    args = parser.parse_args()

    # Constants
    ENV_NAME = args.env
    MODEL_DIR = args.dir
    EVAL_EPS = 20
    TASKS_TO_CONSIDER = args.tasks
    BASELINE_MODE = 'baseline' in ENV_NAME.lower()

    # Initialize environment and load task policies
    env = gym.make(ENV_NAME)
    # Specify which subset of tasks to consider
    TASKS = env.unwrapped.tasks if TASKS_TO_CONSIDER == 'all' else TASKS_TO_CONSIDER
    ONLY_ONE_TASK = len(TASKS) == 1
    # Check that TASKS is a subset of the environment tasks
    assert all(task in env.unwrapped.tasks for task in TASKS), 'Invalid --tasks flag!'

    # Load policies
    agents = {}
    if BASELINE_MODE:
        TASK_MODEL_FILE = find_file(MODEL_DIR, 'baseline*.zip')
        agents['baseline'] = SAC.load(TASK_MODEL_FILE)
    else:
        for task in TASKS:
            TASK_MODEL_FILE = find_file(MODEL_DIR, f'{task}*.zip')
            agents[task] = SAC.load(TASK_MODEL_FILE)

    done = True
    EPS = 0
    # Evaluate
    while EPS < EVAL_EPS:
        if done:
            EPS = EPS + 1
            env.unwrapped.fresh_reset = False if ONLY_ONE_TASK else True
            obs, info = env.reset()
        
        current_task = TASKS[0] if ONLY_ONE_TASK else env.unwrapped.current_task
        print(current_task)
        current_policy = agents['baseline'] if BASELINE_MODE else agents[current_task] 
        action, _states = current_policy.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        done = terminated or truncated
        env.render()