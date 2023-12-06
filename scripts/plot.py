import os
import genx

import gymnasium as gym
import numpy as np

from argparse import ArgumentParser
from stable_baselines3 import SAC

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sklearn.manifold import TSNE

from utils.util import find_file
from eval import rollout, load_sac_policy

def plot_tsne(info):
    # data should be state space
    rollout_obs = info

    # create tsne dataset using rollout_obs
    data_tsne = TSNE(n_components=2, 
                     perplexity=25,
                     learning_rate=50,
                     metric='euclidean',
                ).fit_transform(rollout_obs)
    
    # Plot tsne scatter plot
    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    scat = plt.scatter(data_tsne[:, 0], data_tsne[:, 1])
    plt.title("TSNE")
    plt.xlabel("TSNE Principal Component 1")
    plt.ylabel("TSNE Principal Component 2")
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(scat, cax=cax, orientation='vertical')

    plt.show()

if __name__ == '__main__':
    # Parser
    parser = ArgumentParser()
    parser.add_argument('--envs', type=str, nargs='*', required=True)
    parser.add_argument('--dirs', type=str, nargs='*', required=True)
    parser.add_argument('--tasks', type=str, nargs='*', default='all')
    args = parser.parse_args()
    # Constants
    ENV_NAMES = args.envs
    MODEL_DIRS = args.dirs
    EVAL_EPS = 2
    TASKS_TO_CONSIDER = args.tasks

    assert len(ENV_NAMES) == len(MODEL_DIRS)
    NUM_POLICIES = len(ENV_NAMES)

    for i in range(NUM_POLICIES):
        assert ENV_NAMES[i].lower() in MODEL_DIRS[i].lower(), 'Sanity check to make sure policy matches environment'

    # Initialize environment and load task policies
    envs = [gym.make(env) for env in ENV_NAMES]

    # Load policies
    agents = []
    for i in range(NUM_POLICIES):
        baseline = 'baseline' in ENV_NAMES[i].lower()
        tasks = envs[i].unwrapped.tasks
        agents.append(load_sac_policy(MODEL_DIRS[i], baseline, tasks))

    # Test 1: Joint/action spaces across robots for reach (TODO)
    infos = []
    for i in range(NUM_POLICIES):
        baseline = 'baseline' in ENV_NAMES[i].lower()
        tasks = envs[i].unwrapped.tasks
        info = rollout(envs[i], agents[i], baseline_mode=baseline, eval_eps=EVAL_EPS, tasks=tasks)
        print(f'Success rate: {info["success_rate"]}')
        infos.append(info)

    # Create plots
    print(np.shape(infos[0]['rollout_0']['obs']))
    print(np.shape(infos[0]['rollout_1']['obs']))

    plot_tsne(infos[0]['rollout_0']['obs'])
    plot_tsne(infos[0]['rollout_1']['obs'])