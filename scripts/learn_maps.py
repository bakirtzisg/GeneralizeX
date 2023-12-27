'''
    Learn maps (f,g) where f: S -> S' and g: A -> A'

    Input
    - a metric MDP (one specific compositional RL policy?)
    - a set of compositional structures G (set of compositional RL policies)

    python scripts/learn_maps.py --input_env=CompLift-IIWA --input_policy=PPO --input_dir=experiments/PPO/CompLift-IIWA/20231219-145654-id-7627/models --output_env=CompLift-Panda --output_policy=PPO --output_dir=experiments/PPO/CompLift-Panda/20231222-172458-id-1179/models

'''
import numpy as np
import random 

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.neural_network import MLPRegressor

from argparse import ArgumentParser
from utils.torch_utils import rlxBisimLoss

from utils.wrapper import MDP
from utils.torch_utils import RobotDataset

def learn_linear_maps(M, G):
    print('--- Training linear maps ---')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # TODO: optimize together (multi-objective optimization) or separately?
    # TODO: generalize to not only reach. Think about input/output dims for f and g
    f = nn.Linear(M.state_space_dim['reach'][0], G.state_space_dim['reach'][0]) 
    g = nn.Linear(M.state_space_dim['reach'][0], G.state_space_dim['reach'][0]) 

    f.to(device)
    g.to(device)

    learning_rate = 1e-3
    f_optimizer = torch.optim.SGD(f.parameters(), lr=learning_rate)
    g_optimizer = torch.optim.SGD(g.parameters(), lr=learning_rate)

    num_epochs = 1

    num_batches = 10
    batch_size = 15
    num_examples = num_batches * batch_size

    for epoch in range(num_epochs):
        cumulative_loss = 0
        f.train()
        g.train()
        M_training_data = DataLoader(RobotDataset(M), batch_size=batch_size)
        G_training_data = DataLoader(RobotDataset(G), batch_size=batch_size)

        for i, (M_data, G_data) in enumerate(zip(M_training_data, G_training_data)):
            ''' 
            TODO
             - [x] try getting action distribution from PPO
             - [x] finish batch sampling
             - [x] finish reward function
             - [ ] test linear fitting
             - [ ] try neural network fitting
             
             lower priority
             - [ ] seperate script to generate dataset rather than generating dataset on the fly
             - (faster way to generate dataset ^)
            '''

            # print('M data transposed shape', torch.transpose(M_data, 0, 1).size())
            M_data.to(device)
            G_data.to(device)

            loss_fn = rlxBisimLoss
            loss = loss_fn(f, 
                           g, 
                           mdp_1=M, 
                           mdp_2=G, 
                           samples_1=M_data, 
                           samples_2=G_data,
                    )
            # Remember optimizing for f and g

            f_optimizer.zero_grad()
            g_optimizer.zero_grad()

            loss.backward()
            
            f_optimizer.step()
            g_optimizer.step()

            cumulative_loss += loss.item()
        print("Epoch %s, loss: %s" % (epoch, cumulative_loss / num_examples))

        # evaluation
        # f.eval()
        # g.eval()
    
    return f, g

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_env', type=str, required=True)
    parser.add_argument('--output_env', type=str, required=True)
    parser.add_argument('--input_policy', type=str, required=True)
    parser.add_argument('--output_policy', type=str, required=True)
    parser.add_argument('--input_dir', type=str, required=True) # input MDP dir
    parser.add_argument('--output_dir', type=str, required=True) # output compositional structure dir

    args = parser.parse_args()

    # should have f,g for each subtask (mdp)
    input_mdp = MDP(env=args.input_env, 
                    dir=args.input_dir, 
                    policy=args.input_policy,
                    tasks=['reach'],
                    prefix='best_model',
                )
    output_mdp = MDP(env=args.output_env, 
                     dir=args.output_dir, 
                     policy=args.output_policy,
                     tasks=['reach'],
                    #  tasks=['reach','grasp'],
                     prefix='best_model',
                )

    f, g = learn_linear_maps(input_mdp, output_mdp)

    print(f)