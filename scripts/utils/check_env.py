import genx

import gymnasium as gym

from argparse import ArgumentParser
from stable_baselines3.common.env_checker import check_env

"""
    Gym Environment Checker
    - https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html
"""

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--env', type=str, default='CompLift-IIWA')
    args = parser.parse_args()

    ENV_NAME = args.env

    env = gym.make(ENV_NAME)
    for task in env.unwrapped.tasks:
        env.unwrapped.current_task = task
        env.reset()
        check_env(env)
        print(f'Valid environment for task {task}')