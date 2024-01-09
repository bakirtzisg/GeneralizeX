import os
import numpy as np

from argparse import ArgumentParser
from eval import rollout_mdp
from utils.wrapper import MDP
'''
python scripts/generate_dataset.py --env=CompLift-IIWA --dir=experiments/PPO/CompLift-IIWA/20231219-145654-id-7627/models --policy=PPO --save_path=results/maps/datasets --eps=1000 --tasks reach
'''

def generate_dataset(mdp: MDP, num_samples: int, save_path: str):
    '''
        Generate {num_samples} training dataset from evaluation rollouts of {mdp}

        :param mdp: MDP used to gather dataset
        :param num_samples: number of samples to collect 
        :param save_path: directory to save dataset
    '''
    stats = rollout_mdp(mdp, eps=num_samples, required_success=True, verbose=False, render=False)
    save_name = os.path.join(save_path, f'{mdp.env_name}_{mdp.policy}_{num_samples}_rollouts.npy')
    np.save(save_name, stats)

    data = np.load(save_name, allow_pickle=True).item()

    rnd_idx = np.random.randint(0, num_samples)
    rnd_rollout_data = data[f'rollout_{rnd_idx}']['data']
    print(f'Run {rnd_idx} data shape: {np.shape(rnd_rollout_data)}') # For testing

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--env', type=str, required=True)                 # input gym env name (e.g. CompLift-IIWA)
    parser.add_argument('--policy', type=str, required=True)              # input policy name (e.g. PPO)
    parser.add_argument('--tasks', type=str, nargs='*', default='all')    # input tasks
    parser.add_argument('--dir', type=str, required=True)                 # input MDP dir
    parser.add_argument('--save_path', type=str, default='')              # num epochs for map training
    parser.add_argument('--eps', type=int, default=1)
    args = parser.parse_args()

    mdp = MDP(env=args.env, 
              dir=args.dir, 
              policy=args.policy,
              tasks=args.tasks,
              baseline_mode=('baseline' in args.env.lower()),
              prefix='best_model',
             )
    
    generate_dataset(mdp, args.eps, args.save_path)