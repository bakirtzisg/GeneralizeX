from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

import genx
import gymnasium as gym
import numpy as np

from time import strftime
from argparse import ArgumentParser

if __name__ == '__main__':
    # Parser
    parser = ArgumentParser()
    parser.add_argument('--env_name', type=str, default='BaselineCompPickPlaceCan-v1')
    parser.add_argument('--gamma', type=float, default=0.96)
    parser.add_argument('--epochs', type=int, default=int(5e5))
    parser.add_argument('--eval_freq', type=int, default=int(1e4))
    parser.add_argument('--eval_ep', type=int, default=20)
    parser.add_argument('--save_freq',type=int,default=int(5e4))
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--path', type=str, default='')
    args = parser.parse_args()

    # Training parameters
    ENV_NAME = args.env_name
    DISCOUNT_RATE = args.gamma
    TRAINING_STEPS = args.epochs
    EVAL_FREQUENCY = args.eval_freq
    EVAL_EPISODES = args.eval_ep
    SAVE_FREQUENCY = args.save_freq
    LOAD_MODEL = args.load

    # Model and log paths
    MODEL_PATH = args.path if LOAD_MODEL else f'./experiments/{ENV_NAME}/{strftime("%Y%m%d-%H%M%S")}-id-{np.random.randint(10000)}/'
    TENSORBOARD_PATH = MODEL_PATH + f'tb/'

    # Create environment
    env = gym.make(ENV_NAME)

    agents = {}
    tasks = env.unwrapped.tasks

    # Initialize SAC policy for all tasks
    for task in tasks:
        env.unwrapped.current_task = task
        if LOAD_MODEL:
            agents[task] = SAC.load(MODEL_PATH+f'{task}/', tensorboard_log=TENSORBOARD_PATH+f'{task}/')
        else:
            agents[task] = SAC('MlpPolicy', env, verbose=1, gamma=DISCOUNT_RATE,
                                tensorboard_log=TENSORBOARD_PATH+f'{task}/')

    # Train all task policies sequentially
    for task in tasks:
        # TODO: change
        print(f'{task}')
        if task == 'reach' or task == 'lift':
            continue

        eval_env = Monitor(gym.make(ENV_NAME))
        eval_env.unwrapped.current_task = task

        eval_callback = EvalCallback(eval_env, 
                                    eval_freq=EVAL_FREQUENCY,
                                    n_eval_episodes=EVAL_EPISODES,
                                    deterministic=True,
                                    log_path=TENSORBOARD_PATH,
                                    render=False)
        
        auto_save_callback = CheckpointCallback(save_freq=SAVE_FREQUENCY,
                                                save_path=MODEL_PATH,
                                                name_prefix=f'{task}',
                                                save_replay_buffer=True)
        
        agents[task].learn(total_timesteps=TRAINING_STEPS,
                    callback=[eval_callback, auto_save_callback])
        
        agents[task].save(MODEL_PATH)