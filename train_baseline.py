import gymnasium as gym 
import genx
import numpy as np
import time
import argparse
from stable_baselines3 import SAC, PPO


TOTAL_TRAINING_STEPS = 500000
LOG_INTERVAL = 4
EVAL_INTERVAL = 2000
NUM_EVAL_EPISODES = 20


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='BaselineCompPickPlaceCan-v1')
    parser.add_argument('--policy', type=str, default='SAC')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    experiment_path = f'./experiments/{args.env_name}/{time.strftime("%Y%m%d-%H%M%S")}-id-{np.random.randint(10000)}/'
    env = gym.make(args.env_name)

    policy_params = {
        'policy': 'MlpPolicy',
        'env': env,
        'verbose': 0,
        'gamma': 0.96,
        'tensorboard_log': experiment_path + f'tb/',
    }
    # TODO: PPO doesn't work yet
    agent = PPO(**policy_params) if args.policy == 'PPO' else SAC(**policy_params)

    eval_env = gym.make(args.env_name)
    def evaluate():
        successes = np.zeros(NUM_EVAL_EPISODES, dtype=bool)
        episode_rewards = np.zeros(NUM_EVAL_EPISODES)
        episode_lengths = np.zeros(NUM_EVAL_EPISODES)
        for i in range(NUM_EVAL_EPISODES):
            obs, info = eval_env.reset()
            done = False
            length = 0
            while not done:
                action, _states = agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
                episode_rewards[i] += info['task_reward']
                length += 1
            if terminated and 'episode_success' in info and info['episode_success']:
                successes[i] = True
            episode_lengths[i] = length
        return successes, episode_rewards, episode_lengths
    eval_results = {}

    total_timesteps, callback = agent._setup_learn(
        total_timesteps=TOTAL_TRAINING_STEPS,
        callback=None,
        reset_num_timesteps=True,
        tb_log_name=args.policy,
        progress_bar=False,
    )

    print(f'Evaluating at step 0...')
    successes, episode_rewards, episode_lengths = evaluate()
    eval_results[0] = {
        'successes': successes,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths
    }
    print(eval_results[0])
    # https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/off_policy_algorithm.html
    for step in range(TOTAL_TRAINING_STEPS):
        agent.policy.set_training_mode(False)
        actions, buffer_actions = agent._sample_action(agent.learning_starts, agent.action_noise, agent.env.num_envs)

        new_obs, rewards, dones, infos = agent.env.step(actions)

        agent.num_timesteps += agent.env.num_envs

        # Retrieve reward and episode length if using Monitor wrapper
        agent._update_info_buffer(infos, dones)

        # Store data in replay buffer (normalized action and unnormalized observation)
        agent._store_transition(agent.replay_buffer, buffer_actions, new_obs, rewards, dones, infos)

        agent._update_current_progress_remaining(agent.num_timesteps, agent._total_timesteps)

        for idx, done in enumerate(dones):
            if done:
                agent._episode_num += 1

                # Log training infos
                if LOG_INTERVAL is not None and agent._episode_num % LOG_INTERVAL == 0:
                    agent._dump_logs()

        agent.train(batch_size=agent.batch_size, gradient_steps=1)

        if (step + 1) % EVAL_INTERVAL == 0:
            print(f'Evaluating at step {step}...')
            successes, episode_rewards, episode_lengths = evaluate()
            eval_results[step + 1] = {
                'successes': successes,
                'episode_rewards': episode_rewards,
                'episode_lengths': episode_lengths
            }
            print(eval_results[step + 1])

    agent.save(experiment_path + f'agent')
    np.save(experiment_path + 'eval_results.npy', eval_results)
