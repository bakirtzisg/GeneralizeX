'''
    Domain-specific generalization -- learn maps (f,g) where f: S -> S' and g: A -> A'

    Input
    - a metric MDP (a specific compositional RL policy)
    - a set of compositional structures G (set of compositional RL policies)

    Output
    - (f,g,epsilon)

    python scripts/learn_maps.py --input_env=CompLift-IIWA --input_policy=PPO --input_dir=experiments/PPO/CompLift-IIWA/20231219-145654-id-7627/models --output_env=CompLift-Panda --output_policy=PPO --output_dir=experiments/PPO/CompLift-Panda/20231222-172458-id-1179/models --epochs=200 --type=linear

'''
import os
from time import strftime

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from argparse import ArgumentParser
from utils.torch_utils import rlxBisimLoss

from utils.wrapper import MDP
from utils.torch_utils import RobotDataset, MLP

# torch device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def learn_maps(M: MDP, G: MDP, map_type: str = 'linear', epochs: int = 100, maps_dir=None):
    '''
        :param M: input MDP
        :param G: output MDP
        :param map_type: specify map parameterization (e.g. linear, mlp)
        :param epochs: number of training epochs
        :param maps_dir: directory of previously trained maps (f,g)
    '''
    # hyperparameters
    learning_rate = 1e-3
    batch_size = 15

    # TODO: generalize to not only reach. Think about input/output dims for f and g
    # when M transitions, G should too? (f,g) based on subtask? parameterize (f,g) with task (f,g,t)?
    if map_type == 'linear':
        f = nn.Linear(M.state_space_dim['reach'], G.state_space_dim['reach']) 
        g = nn.Linear(M.action_space_dim['reach'], G.action_space_dim['reach']) 
    elif map_type == 'mlp':
        f = MLP(M.state_space_dim['reach'], G.state_space_dim['reach'])
        g = MLP(M.action_space_dim['reach'], G.action_space_dim['reach'])

    if maps_dir is not None:
        save_path = maps_dir
        print(f'--- Loading {map_type} maps ---')

        f_checkpoint = torch.load(os.path.join(maps_dir, 'f.pt'))
        g_checkpoint = torch.load(os.path.join(maps_dir, 'g.pt'))
        f.load_state_dict(f_checkpoint['model_state_dict'])
        g.load_state_dict(g_checkpoint['model_state_dict'])

        # optimizer (stochastic gradient descent)
        f_optimizer = torch.optim.SGD(f.parameters(), lr=learning_rate)
        g_optimizer = torch.optim.SGD(g.parameters(), lr=learning_rate)
        f_optimizer.load_state_dict(f_checkpoint['optimizer_state_dict'])
        g_optimizer.load_state_dict(g_checkpoint['optimizer_state_dict'])

        f_epoch = f_checkpoint['epoch']
        g_epoch = g_checkpoint['epoch']
        
    else:
        save_path = os.path.join(os.path.curdir, f'results/maps/{map_type}/{strftime("%Y%m%d-%H%M%S")}-id-{np.random.randint(10000)}')

        print(f'--- Training {map_type} maps ---')

        f_optimizer = torch.optim.SGD(f.parameters(), lr=learning_rate)
        g_optimizer = torch.optim.SGD(g.parameters(), lr=learning_rate)

        f_epoch = 0
        g_epoch = 0
    
    # send f,g to device (GPU)
    f.to(device)
    g.to(device)

    # if skipping training
    if epochs == 0: return f, g
    
    # setup tensorboard
    log_path = os.path.join(save_path, 'log')
    if not os.path.exists(log_path): os.makedirs(log_path)
    writer = SummaryWriter(log_path)

    # Set f and g to training mode
    f.train()
    g.train()

    # TODO: optimize together (multi-objective optimization) or separately?
    for epoch in range(epochs):
        cumulative_loss = 0
        M_training_data = DataLoader(RobotDataset(M), batch_size=batch_size, shuffle=True)

        # Batch training
        for M_data in M_training_data:
            ''' 
            TODO
             - [ ] seperate script to generate dataset rather than generating dataset on the fly
             - (faster way to generate dataset ^)
            '''
            f_optimizer.zero_grad()
            g_optimizer.zero_grad()

            M_data.to(device)

            # Relaxed bisimulation loss
            loss_fn = rlxBisimLoss
            loss = loss_fn(f, 
                           g, 
                           mdp_1=M, 
                           mdp_2=G, 
                           samples_1=M_data, 
                           device=device,
                    )

            loss.backward()
            
            f_optimizer.step()
            g_optimizer.step()

            cumulative_loss += loss.item()
        
        print("Epoch %s, loss: %s" % (f_epoch + epoch, cumulative_loss / batch_size))
        writer.add_scalar("Loss/train - f", cumulative_loss / batch_size, f_epoch + epoch)
        writer.add_scalar("Loss/train - g", cumulative_loss / batch_size, g_epoch + epoch)

    # Save models
    torch.save({
                'epoch': f_epoch + epochs,
                'model_state_dict': f.state_dict(),
                'optimizer_state_dict': f_optimizer.state_dict(),
               }, os.path.join(save_path, 'f.pt'))
    
    torch.save({
                'epoch': g_epoch + epochs,
                'model_state_dict': g.state_dict(),
                'optimizer_state_dict': g_optimizer.state_dict(),
               }, os.path.join(save_path, 'g.pt'))
    
    # Log to tensorboard
    writer.flush()
    writer.close()

    return f, g

def evaluate(M: MDP, G: MDP, f, g):
    '''
        Evaluate maps f and g on evaluation set

        :param M: input MDP
        :param G: compositional structure
        :param f: trained map f: S_1 -> S_2
        :param g: trained map g: A_1 -> A_2
    '''
    print(f'Evaluating f and g')
    losses = []
    epsilon = 0

    # Set maps to evaluation mode
    f.eval()
    g.eval()

    test_set = DataLoader(RobotDataset(M))
    
    with torch.no_grad():
        for data in test_set:
            loss_fn = rlxBisimLoss
            loss = loss_fn(f=f,
                        g=g,
                        mdp_1=M,
                        mdp_2=G,
                        samples_1=data,
                        device=device,
            )

            loss = loss.detach().numpy().item() # this is a singleton
            if loss > epsilon: epsilon = loss

            losses.append(loss)

    print('losses:', losses)
    print('epsilon:', epsilon)

    return epsilon

def test_maps(M: MDP, G: MDP, f, g):
    '''
        TODO
        :param M: input MDP
        :param G: compositional structure
        :param f: trained map f: S_1 -> S_2
        :param g: trained map g: A_1 -> A_2
    '''
    '''
    - get rollout from MDP1: history of states, actions
    - apply (f,g) to states and actions respectively
    - when MDP2 is in (s_2), check when f(s_1) = s_2 and apply the corresponding action a_1 MDP1 will take at s_1 and apply g(a_1) to MDP2
    '''

    M_data = DataLoader(RobotDataset(M), batch_size=M.horizon)
    # get states and actions
    # transform states and actions using (f,g)

    epsilon = 0
    for data in M_data:
        M_states = data[:,0:6]
        M_actions = data[:,6:10]
        M_rwds = data[:,-1]
        
        G_states = f(M_states)
        G_actions = g(M_actions)

        e = rlxBisimLoss(f, g, M, G, data, device) # TODO: for this sample

        if e > epsilon: epsilon = e

    # Evaluation loop for G
    done = False
    observation, info = G.env.reset()
    for t in range(G.horizon):
        if done:
            G.env.reset()
        # this assumes that the action distributions are continuous wrt to state space. Not a good assumption
        action = G_actions[closest_state_to_observation_in_G_states] # interpolate??
        observation, reward, terminated, truncated, info = G.env.step(action)
        done = terminated or truncated
        G.env.render()

    print('Completed Task:', info['task_success'])
    
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
    parser.add_argument('--type', type=str, required=True)          # map parameterization for (f,g)
    parser.add_argument('--maps_dir', type=str)
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

    f, g = learn_maps(input_mdp, 
                      output_mdp, 
                      map_type=args.type, 
                      epochs=args.epochs, 
                      maps_dir=args.maps_dir)

    epsilon = evaluate(input_mdp, output_mdp, f, g)