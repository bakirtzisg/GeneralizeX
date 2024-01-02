'''
    Domain-specific generalization -- learn maps (f,g) where f: S -> S' and g: A -> A'

    Input
    - a metric MDP (a specific compositional RL policy)
    - a set of compositional structures G (set of compositional RL policies)

    Output
    - (f,g,epsilon)

    python scripts/learn_maps.py --input_env=CompLift-IIWA --input_policy=PPO --input_dir=experiments/PPO/CompLift-IIWA/20231219-145654-id-7627/models --output_env=CompLift-Panda --output_policy=PPO --output_dir=experiments/PPO/CompLift-Panda/20231222-172458-id-1179/models --epochs=200

'''
import os
from time import strftime

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.neural_network import MLPRegressor

from argparse import ArgumentParser
from utils.torch_utils import rlxBisimLoss

from utils.wrapper import MDP
from utils.torch_utils import RobotDataset, MLP

def learn_maps(M: MDP, G: MDP, map_type: str = 'linear', epochs: int = 100):
    '''
        :param M: input MDP
        :param G: output MDP
        :param map_type: specify map parameterization (e.g. linear, mlp)
        :param epochs: number of training epochs
    '''
    save_path = os.path.join(os.path.curdir, f'results/maps/{strftime("%Y%m%d-%H%M%S")}-id-{np.random.randint(10000)}')
    log_path = os.path.join(save_path, 'log')
    if not os.path.exists(log_path): os.makedirs(log_path)
    writer = SummaryWriter(log_path)

    print(f'--- Training {map_type} maps ---')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # TODO: optimize together (multi-objective optimization) or separately?
    # TODO: generalize to not only reach. Think about input/output dims for f and g
    if map_type == 'linear':
        f = nn.Linear(M.state_space_dim['reach'][0], G.state_space_dim['reach'][0]) 
        g = nn.Linear(M.action_space_dim['reach'][0], G.action_space_dim['reach'][0]) 
    if map_type == 'mlp':
        # TODO
        f = MLP()
        g = MLP()

    f.to(device)
    g.to(device)

    # hyperparameters
    learning_rate = 1e-3
    num_batches = 10
    batch_size = 15
    num_examples = num_batches * batch_size

    # optimizer (stochastic gradient descent)
    f_optimizer = torch.optim.SGD(f.parameters(), lr=learning_rate)
    g_optimizer = torch.optim.SGD(g.parameters(), lr=learning_rate)

    # Set f and g to training mode
    f.train()
    g.train()

    for epoch in range(epochs):
        cumulative_loss = 0
        M_training_data = DataLoader(RobotDataset(M), batch_size=batch_size)
        G_training_data = DataLoader(RobotDataset(G), batch_size=batch_size)

        # Batch training
        for i, (M_data, G_data) in enumerate(zip(M_training_data, G_training_data)):
            ''' 
            TODO
             - [x] try getting action distribution from PPO
             - [x] finish batch sampling
             - [x] finish reward function
             - [x] test linear fitting - loss barely decreases... (1/2)
             - [ ] try neural network fitting
             
             lower priority
             - [ ] seperate script to generate dataset rather than generating dataset on the fly
             - (faster way to generate dataset ^)
            '''
            M_data.to(device)
            G_data.to(device)

            # Relaxed bisimulation loss
            loss_fn = rlxBisimLoss
            loss = loss_fn(f, 
                           g, 
                           mdp_1=M, 
                           mdp_2=G, 
                           samples_1=M_data, 
                           samples_2=G_data,
                           device=device,
                    )
            # Remember optimizing for f and g

            f_optimizer.zero_grad()
            g_optimizer.zero_grad()

            loss.backward()
            
            f_optimizer.step()
            g_optimizer.step()

            cumulative_loss += loss.item()
        
        print("Epoch %s, loss: %s" % (epoch, cumulative_loss / num_examples))
        writer.add_scalar("Loss/train", cumulative_loss / num_examples, epoch)

    # Save models
    torch.save({
                'epoch': epochs,
                'model_state_dict': f.state_dict(),
                'optimizer_state_dict': f_optimizer.state_dict(),
               }, os.path.join(save_path, 'f.pt'))
    
    torch.save({
                'epoch': epochs,
                'model_state_dict': g.state_dict(),
                'optimizer_state_dict': g_optimizer.state_dict(),
               }, os.path.join(save_path, 'g.pt'))
    
    # Log to tensorboard
    writer.flush()
    writer.close()

    return f, g

def evaluate(f: torch.nn.Module, g: torch.nn.Module):
    '''
        TODO
        :param f: trained map f: S_1 -> S_2
        :param g: trained map g: A_1 -> A_2
    '''
    print(f'Evaluating f and g')
    epsilon = 0
    # Set maps to evaluation mode
    f.eval()
    g.eval()

    return epsilon

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_env', type=str, required=True)     # input gym env name (e.g. CompLift-IIWA)
    parser.add_argument('--output_env', type=str, required=True)    # output gym env name (e.g. CompLift-Panda)
    parser.add_argument('--input_policy', type=str, required=True)  # input policy name (e.g. PPO)
    parser.add_argument('--output_policy', type=str, required=True) # output policy name (e.g. PPO)
    parser.add_argument('--input_dir', type=str, required=True)     # input MDP dir
    parser.add_argument('--output_dir', type=str, required=True)    # output compositional structure dir
    parser.add_argument('--epochs', type=int, default=100)          # num epochs for map training

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

    f, g = learn_maps(input_mdp, output_mdp, map_type='linear', epochs=args.epochs)

    epsilon = evaluate(f,g)