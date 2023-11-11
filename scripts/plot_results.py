import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from argparse import ArgumentParser

sns.set(font_scale=1.5)

PLOT_METRIC = 'Success Rate'
# PLOT_METRIC = 'Mean Episode Reward'
ENV_NAME = 'PickPlaceCan'
TITLE = 'Pick and Place a Can'

experiments = [f'BaselineComp{ENV_NAME}-v1', f'Comp{ENV_NAME}-v1']
labels = ['Baseline', 'Ours']
# experiments = [f'Comp{ENV_NAME}-v1']
# labels = ['Ours']


def compute_metrics(file_path):
    eval_results = np.load(file_path, allow_pickle=True)
    steps = list(eval_results.keys())
    steps.sort()
    success_rates = np.empty(len(steps))
    mean_episode_rewards = np.empty(len(steps))
    for i, step in enumerate(steps):
        # TODO: success rates are not logged - what does it mean to succeed?
        pass
        # success_rates[i] = np.mean(eval_results[step]['successes'])
        # mean_episode_rewards[i] = np.mean(eval_results[step]['episode_rewards'])
    return steps, success_rates, mean_episode_rewards


def compute_mean_std(metric_arrays):
    metric_arrays = np.array(metric_arrays)
    mean = np.mean(metric_arrays, axis=0)
    std = np.std(metric_arrays, axis=0)
    return mean, std

def plot(savefig=False):
    for i in range(len(experiments)):
        experiment = experiments[i]
        label = labels[i]
        base_path = f'./results/{experiment}'
        folders = os.listdir(base_path)
        success_rates = []
        mean_episode_rewards = []
        for folder in folders:
            file_path = f'{base_path}/{folder}/eval_results.npy'
            steps, success_rate, mean_episode_reward = compute_metrics(file_path)
            success_rates.append(success_rate)
            mean_episode_rewards.append(mean_episode_reward)
        
        if PLOT_METRIC == 'Success Rate':
            mean, std = compute_mean_std(success_rates)
            plt.plot(steps, mean, label=label)
            plt.fill_between(steps, (mean - std).clip(0, 1), (mean + std).clip(0, 1), alpha=0.2)
        elif PLOT_METRIC == 'Mean Episode Reward':
            mean, std = compute_mean_std(mean_episode_rewards)
            plt.plot(steps, mean, label=label)
            plt.fill_between(steps, mean - std, mean + std, alpha=0.2)

    plt.legend()
    plt.xlabel('Environment Steps')
    plt.ylabel(PLOT_METRIC)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.title(TITLE)
    plt.tight_layout()

    if savefig: plt.savefig(f'{ENV_NAME}-{PLOT_METRIC}.pdf')

    plt.show()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--path', type=str, default='')
    parser.add_argument('-s', '--save', action='store_true')
    args = parser.parse_args()

    compute_metrics(args.path)

    plot(savefig=args.save)