from argparse import ArgumentParser
from scripts.learn_maps import evaluate, test_maps

from utils.wrapper import MDP

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_env', type=str, required=True)
    parser.add_argument('--output_env', type=str, required=True)
    parser.add_argument('--input_policy', type=str, required=True)
    parser.add_argument('--output_policy', type=str, required=True)
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--maps_dir', type=str, required=True)

    args = parser.parse_args()

    input_mdp = MDP(env=args.input_env, 
                    dir=args.input_dir, 
                    policy=args.input_policy,
                    baseline_mode=True,
                )
    output_mdp = MDP(env=args.output_env, 
                     dir=args.output_dir, 
                     policy=args.output_policy,
                    #  tasks=['reach'],
                    #  tasks=['reach','grasp'],
                     prefix='best_model',
                )

    epsilon = evaluate(input_mdp, output_mdp, f, g)

    