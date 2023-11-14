import os
import glob
import genx
import gymnasium as gym
import numpy as np

from time import strftime
from argparse import ArgumentParser

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, StopTrainingOnNoModelImprovement

from utils.callback import LoggerCallback
from utils.util import get_policy

if __name__ == '__main__':
    # Parser
    parser = ArgumentParser()
    parser.add_argument('--env_name', type=str, default='BaselineCompPickPlaceCan-Panda')
    parser.add_argument('--gamma', type=float, default=0.96)
    parser.add_argument('--epochs', type=int, default=int(5e5))
    parser.add_argument('--eval_freq', type=int, default=int(1e4))
    parser.add_argument('--eval_ep', type=int, default=20)
    parser.add_argument('--save_freq',type=int,default=int(5e4))
    parser.add_argument('--resume_training', action='store_true')
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--path', type=str, default='')
    args = parser.parse_args()

    # Training parameters
    ENV_NAME = args.env_name
    DISCOUNT_RATE = args.gamma
    TRAINING_STEPS = args.epochs
    EVAL_FREQUENCY = args.eval_freq
    EVAL_EPISODES = args.eval_ep
    SAVE_FREQUENCY = args.save_freq
    RESUME_TRAINING = args.resume_training
    BASELINE_MODE = args.baseline or 'baseline' in ENV_NAME.lower()

    # Model and log paths
    MODEL_PATH = args.path if RESUME_TRAINING else f'./experiments/{ENV_NAME}/{strftime("%Y%m%d-%H%M%S")}-id-{np.random.randint(10000)}/'
    TENSORBOARD_PATH = os.path.join(MODEL_PATH, 'tb/')
    LOG_PATH = os.path.join(MODEL_PATH, 'log/')

    # Initialize environment
    env = gym.make(ENV_NAME)
    # Initialize evaluation environment
    eval_env = Monitor(gym.make(ENV_NAME))
    # Stop training on no model improvement callback
    stop_training_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=2e4,
                                                            min_evals=int(5e4),
                                                            verbose=1)
    # Evaluation callback
    eval_callback = EvalCallback(eval_env, 
                                eval_freq=EVAL_FREQUENCY,
                                n_eval_episodes=EVAL_EPISODES,
                                callback_on_new_best=stop_training_callback,
                                deterministic=True,
                                log_path=TENSORBOARD_PATH,
                                best_model_save_path=LOG_PATH,
                                verbose=1,
                                render=False)
    # Log to tensorboard callback
    # logger_callback = LoggerCallback()

    if BASELINE_MODE:
        if RESUME_TRAINING:
            BASELINE_MODEL_PATH = get_policy(MODEL_PATH)
            agent = SAC.load(BASELINE_MODEL_PATH,
                             tensorboard_log=os.path.join(TENSORBOARD_PATH))
            agent.set_env(env)
        else:
            agent = SAC('MlpPolicy', env, verbose=1, gamma=DISCOUNT_RATE,
                        tensorboard_log=TENSORBOARD_PATH)
        
        print('TRAINING BASELINE POLICY')
        # Periodically save model callback
        auto_save_callback = CheckpointCallback(save_freq=SAVE_FREQUENCY,
                                                save_path=MODEL_PATH,
                                                save_replay_buffer=True,
                                                verbose=2)
        callbacks = [eval_callback, auto_save_callback]
        agent.learn(total_timesteps=TRAINING_STEPS,
                    callback=callbacks,
                    reset_num_timesteps=False)
        # Save final model
        agent.save(os.path.join(MODEL_PATH, f'_final.zip'))
    else: # Compositional mode
        agents = {}
        tasks = env.unwrapped.tasks
        # Initialize SAC policy for all tasks + train all task policies sequentially
        for task in tasks:
            env.unwrapped.current_task = task
            if RESUME_TRAINING:
                # Pattern-match task policy path
                TASK_MODEL_PATH = get_policy(MODEL_PATH, f'{task}')
                # Load previously trained task policy
                agents[task] = SAC.load(TASK_MODEL_PATH,                      
                                        tensorboard_log=os.path.join(TENSORBOARD_PATH, f'{task}/'))
                agents[task].set_env(env)
            else:
                # Initialize SAC policy
                agents[task] = SAC('MlpPolicy', env, verbose=1, gamma=DISCOUNT_RATE,
                                    tensorboard_log=os.path.join(TENSORBOARD_PATH, f'{task}/'))
            
            print(f'TRAINING COMPOSITIONAL POLICY: SUBTASK {task.upper()}')
            # Set current task in evaluation environment
            eval_env.unwrapped.current_task = task
            # Periodically save model callback
            auto_save_callback = CheckpointCallback(save_freq=SAVE_FREQUENCY,
                                                    save_path=MODEL_PATH,
                                                    name_prefix=f'{task}',
                                                    save_replay_buffer=True,
                                                    verbose=2)
            callbacks = [eval_callback, auto_save_callback]
            # Train model
            agents[task].learn(total_timesteps=TRAINING_STEPS,
                                callback=callbacks,
                                reset_num_timesteps=False)
            # Save final model
            agents[task].save(os.path.join(MODEL_PATH, f'{task}_final.zip'))