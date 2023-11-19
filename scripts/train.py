import os
import glob
import genx
import gymnasium as gym
import numpy as np

from time import strftime
from argparse import ArgumentParser

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor, ResultsWriter
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, StopTrainingOnNoModelImprovement, StopTrainingOnRewardThreshold

from utils.util import find_file

if __name__ == '__main__':
    # Parser
    parser = ArgumentParser()
    # Model parameters
    parser.add_argument('--env', type=str, default='BaselineCompPickPlaceCan-Panda')
    parser.add_argument('--dir', type=str, default='')
    parser.add_argument('--model_prefix', type=str, default='')
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--resume_training', action='store_true')
    # Training parameters
    parser.add_argument('--gamma', type=float, default=0.96)
    parser.add_argument('--epochs', type=int, default=int(5e5))
    parser.add_argument('--eval_freq', type=int, default=int(1e4))
    parser.add_argument('--eval_ep', type=int, default=20)
    parser.add_argument('--save_freq',type=int,default=int(5e4))

    args = parser.parse_args()

    # Training parameters
    ENV_NAME = args.env
    DISCOUNT_RATE = args.gamma
    TRAINING_STEPS = args.epochs
    EVAL_FREQUENCY = args.eval_freq
    EVAL_EPISODES = args.eval_ep
    SAVE_FREQUENCY = args.save_freq
    RESUME_TRAINING = args.resume_training
    BASELINE_MODE = args.baseline

    # Model and log paths
    MODEL_PREFIX = args.model_prefix
    MODEL_PATH = args.dir if RESUME_TRAINING else f'./experiments/{ENV_NAME}/{strftime("%Y%m%d-%H%M%S")}-id-{np.random.randint(10000)}/'
    TENSORBOARD_PATH = os.path.join(MODEL_PATH, 'tb/')
    LOG_PATH = os.path.join(MODEL_PATH, 'log/')

    # Initialize environment
    env = gym.make(ENV_NAME)
    
    agents = {} 
    if BASELINE_MODE:
        if RESUME_TRAINING:
            BASELINE_MODEL_PATH = find_file(MODEL_PATH, f'{MODEL_PREFIX}*.zip')
            agents['baseline'] = SAC.load(BASELINE_MODEL_PATH,
                                          tensorboard_log=os.path.join(TENSORBOARD_PATH))
            agents['baseline'].set_env(env)
        else:
            agents['baseline'] = SAC('MlpPolicy', env, verbose=1, gamma=DISCOUNT_RATE,
                                     tensorboard_log=TENSORBOARD_PATH)
    else: # Compositional mode
        tasks = env.unwrapped.tasks
        # Initialize SAC policy for all tasks
        for task in tasks:
            env.unwrapped.current_task = task
            if RESUME_TRAINING:
                # Pattern-match task policy path
                TASK_MODEL_PATH = find_file(MODEL_PATH, f'{task}*.zip')
                # Load previously trained task policy
                agents[task] = SAC.load(TASK_MODEL_PATH,                      
                                        tensorboard_log=os.path.join(TENSORBOARD_PATH, f'{task}/'))
                agents[task].set_env(env)
            else:
                # Initialize SAC policy
                agents[task] = SAC('MlpPolicy', env, verbose=1, gamma=DISCOUNT_RATE,
                                    tensorboard_log=os.path.join(TENSORBOARD_PATH, f'{task}/'))
    
    # Initialize evaluation environment (with custom logs info_keywords)
    info_kws = ('current_task', 'current_action', 'current_task_obs', 'episode_success')
    eval_env = Monitor(gym.make(ENV_NAME), 
                       LOG_PATH,
                       info_keywords=info_kws)
    
    # Train subtask policies sequentially
    for task, agent in agents.items():
        print(f'TRAINING {task.upper()}')
        if task == 'reach': continue
        # Setup current task of environment
        if not BASELINE_MODE:
            env.unwrapped.current_task = task
            eval_env.unwrapped.current_task = task
        
        # Stop training on no model improvement callback
        # stop_training_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=5e4,
        #                                                           min_evals=100,
        #                                                           verbose=1)

        # Stop training on reward threshold callback
        # 10 is the task completion reward - e.g. see reward function for lift
        stop_training_callback = StopTrainingOnRewardThreshold(reward_threshold=10, verbose=1)

        # Evaluation callback
        eval_callback = EvalCallback(eval_env, 
                                    eval_freq=EVAL_FREQUENCY,
                                    n_eval_episodes=EVAL_EPISODES,
                                    callback_on_new_best=stop_training_callback,
                                    deterministic=True,
                                    log_path=TENSORBOARD_PATH,
                                    # best_model_save_path=LOG_PATH,
                                    verbose=1,
                                    render=False)
        
        # Periodically save model callback
        auto_save_callback = CheckpointCallback(save_freq=SAVE_FREQUENCY,
                                                save_path=MODEL_PATH,
                                                name_prefix=f'{task}',
                                                save_replay_buffer=True,
                                                verbose=2)
        
        callbacks = [eval_callback, auto_save_callback]
        agent.learn(total_timesteps=TRAINING_STEPS,
                    callback=callbacks,
                    reset_num_timesteps=False)
        # Save final model
        agent.save(os.path.join(MODEL_PATH, f'{task}_final.zip'))