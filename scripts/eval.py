import os
import glob
import genx

import numpy as np
import gymnasium as gym 

from argparse import ArgumentParser
from stable_baselines3 import SAC

if __name__ == '__main__':
    # Parser
    parser = ArgumentParser()
    parser.add_argument('--env', type=str, default='BaselineCompPickPlaceCan-Panda')
    parser.add_argument('--path', type=str, default='')
    parser.add_argument('--tasks', type=str, nargs='*', default='all')
    args = parser.parse_args()

    # Constants
    ENV_NAME = args.env
    MODEL_PATH = args.path
    EVAL_EPS = 20
    TASKS_TO_CONSIDER = args.tasks

    # Initialize environment and load task policies
    env = gym.make(ENV_NAME)
    # Specify which subset of tasks to consider
    TASKS = env.unwrapped.tasks if TASKS_TO_CONSIDER == 'all' else TASKS_TO_CONSIDER
    # Check that TASKS is a subset of the environment tasks
    assert all(task in env.unwrapped.tasks for task in TASKS), 'Invalid --tasks flag!'

    sac_agents = {}

    for task in TASKS:
        # Pattern-match model path
        TASK_MODEL_PATH = glob.glob(os.path.join(MODEL_PATH, f'{task}*.zip'))
        # Verify that only one model was pattern-matched
        print(TASK_MODEL_PATH)
        assert len(TASK_MODEL_PATH) == 1, 'Found multiple task policies!'

        sac_agents[task] = SAC.load(TASK_MODEL_PATH[0])

    done = True
    EPS = 0
    # Evaluate
    while EPS < EVAL_EPS:
        if done:
            EPS = EPS + 1
            obs, info = env.reset()
        obs = env.unwrapped._get_obs()
        current_task = TASKS[0] if len(TASKS) == 1 else env.unwrapped.current_task
        action, _states = sac_agents[current_task].predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        env.render()