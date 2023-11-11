import os
import genx

import numpy as np
import gymnasium as gym 

from argparse import ArgumentParser
from stable_baselines3 import SAC

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--env', type=str, default='BaselineCompPickPlaceCan-v1')
    parser.add_argument('--path', type=str, default='')
    parser.add_argument('--suffix', type=str, default='')
    args = parser.parse_args()

    ENV_NAME = args.env
    MODEL_PATH = args.path

    env = gym.make(ENV_NAME)
    sac_agents = {}
    for task in env.unwrapped.tasks:
        # FILE_NAME = f'sac_{task}{args.suffix}'
        if task == 'move' or task == 'place': # TODO: remove
            continue
        FILE_NAME = f'{task}_{args.suffix}'
        TASK_MODEL_PATH = os.path.join(MODEL_PATH, FILE_NAME)
        sac_agents[task] = SAC.load(TASK_MODEL_PATH)

    done = True

    while True:
        if done:
            obs, info = env.reset()
        obs = env.unwrapped._get_obs()
        current_task = env.unwrapped.current_task
        action, _states = sac_agents[current_task].predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        env.render()