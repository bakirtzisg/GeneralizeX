'''
    Domain-specific generalization -- learn maps (f,g) where f: S -> S' and g: A -> A'

    Input
    - a metric MDP (a specific compositional RL policy)
    - a set of compositional structures G (set of compositional RL policies)

    Output
    - (f,g,epsilon)

    CompLift-IIWA to CompLift-Panda:
    python scripts/learn_maps.py --input_env=CompLift-IIWA --input_policy=PPO --input_dir=experiments/PPO/CompLift-IIWA/20231219-145654-id-7627/models --input_tasks reach --output_env=CompLift-Panda --output_policy=PPO --output_dir=experiments/PPO/CompLift-Panda/20231222-172458-id-1179/models --output_tasks reach --epochs=1000 --type=linear

    BaselineLift-Panda to CompLift-Panda:
    python scripts/learn_maps.py --input_env=BaselineLift-Panda --input_policy=PPO --input_dir=experiments/PPO/BaselineLift-Panda/20240108-084034-id-8993/models --input_tasks all --output_env=CompLift-Panda --output_policy=PPO --output_dir=experiments/PPO/CompLift-Panda/20240111-104724-id-1351/models --output_tasks all --epochs=10 --type=mlp --train_data_path=results/maps/datasets/BaselineLift-Panda_PPO_1000_20240122-113005_rollouts.npy
'''
import os

import numpy as np

from time import strftime
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.wrapper import MDP, MapsWrapper
from utils.loss_function import rlxBisimLoss
from utils.dataset_wrappers import RobotDataset, RobotDatasetLive

# torch device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def learn_maps(M: MDP, G: MDP, 
               data_path=None, map_type: str = 'linear', 
               epochs: int = 100, maps_dir=None) -> MapsWrapper:
    '''
        Learn maps (f: S -> S', g: A -> A') to test domain-agnostic generalization

        :param M: input MDP
        :param G: output MDP
        :param map_type: specify map parameterization (e.g. linear, mlp)
        :param epochs: number of training epochs
        :param maps_dir: directory of previously trained maps (f,g)
    '''
    # hyperparameters
    opt_params = {'learning_rate': 1e-3}
    batch_size = 1

    # training parameters
    log_every_epochs = 10
    

    maps = MapsWrapper(M, G, map_type=map_type, maps_dir=maps_dir, opt_params=opt_params)

    # if skipping training
    if epochs == 0: return maps
    
    # setup tensorboard
    log_path = os.path.join(maps.save_path, 'log')
    if not os.path.exists(log_path): os.makedirs(log_path)
    writer = SummaryWriter(log_path)

    # Set f and g to training mode + send to device (GPU)
    maps.train()
    maps.to(device)

    maps.step = {f'{task}': 0 for task in G.tasks}
    for epoch in range(epochs):
        cumulative_loss = {f'{task}': 0 for task in G.tasks}
        if data_path is None:
            # Collect data live
            # M_training_data = DataLoader(RobotDatasetLive(M), batch_size=batch_size, shuffle=True)
            M_training_data = DataLoader(RobotDatasetLive(M), batch_size=batch_size)
        else:
            M_training_data = DataLoader(RobotDataset(M, data_path), batch_size=batch_size)
        # Batch training
        for M_data, M_task in M_training_data:
            current_task = M_task[0]
            f_subtask = maps.f[current_task]
            g_subtask = maps.g[current_task]

            f_subtask_optimizer = maps.f_optimizer[current_task]
            g_subtask_optimizer = maps.g_optimizer[current_task]
            # TODO: look at current state (with history?), 
            #       decide which task is being performed, 
            #       train corresponding (f,g) map
            # TODO: train on entire trajectory (maybe with dropout), and use G to determine when to transition?
            f_subtask_optimizer.zero_grad()
            g_subtask_optimizer.zero_grad()
            
            M_data.to(device)

            # Relaxed bisimulation loss
            loss_fn = rlxBisimLoss
            loss = loss_fn(f_subtask, 
                           g_subtask, 
                           mdp_1=M, 
                           mdp_2=G,
                           task=current_task, 
                           samples_1=M_data, 
                           device=device,
                    )

            loss.backward()
            
            f_subtask_optimizer.step()
            g_subtask_optimizer.step()

            cumulative_loss[current_task] += loss.item()
            maps.step[current_task] += 1

        step_loss = [v / (batch_size * len(M_training_data)) for v in cumulative_loss.values()]
        print(f"Epoch {maps.f_epoch + epoch}, losses: {step_loss}")
        
        if epoch % log_every_epochs == 0 or epoch == epochs - 1:
            for task in G.tasks:
                writer.add_scalar(f"f loss/train - {task}", cumulative_loss[f'{task}'] / (batch_size * len(M_training_data)), maps.f_epoch + step[f'{task}'])
                writer.add_scalar(f"g loss/train - {task}", cumulative_loss[f'{task}'] / (batch_size * len(M_training_data)), maps.g_epoch + step[f'{task}'])

    # Save models
    maps.save()
    
    # Log to tensorboard
    writer.flush()
    writer.close()

    return maps

def evaluate(M: MDP, G: MDP, f, g, data_path: str = None):
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

    # TODO: for single task - generalize to multi-task 
    f = f[M.tasks[0]] if len(M.tasks) == len(G.tasks) == 1 else None
    g = g[M.tasks[0]] if len(M.tasks) == len(G.tasks) == 1 else None

    # Set maps to evaluation mode
    f.eval()
    g.eval()

    if data_path is None:
        test_set = DataLoader(RobotDatasetLive(M))
    else:
        test_set = DataLoader(RobotDataset(M, data_path))
    
    with torch.no_grad():
        for data, task in test_set:
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

    M_data = DataLoader(RobotDatasetLive(M), batch_size=M.horizon)
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
    parser.add_argument('--input_env', type=str, required=True)                 # input gym env name (e.g. CompLift-IIWA)
    parser.add_argument('--output_env', type=str, required=True)                # output gym env name (e.g. CompLift-Panda)
    parser.add_argument('--input_policy', type=str, required=True)              # input policy name (e.g. PPO)
    parser.add_argument('--output_policy', type=str, required=True)             # output policy name (e.g. PPO)
    parser.add_argument('--input_tasks', type=str, nargs='*', required=True)    # input tasks
    parser.add_argument('--output_tasks', type=str, nargs='*', required=True)   # output tasks
    parser.add_argument('--input_dir', type=str, required=True)                 # input MDP dir
    parser.add_argument('--output_dir', type=str, required=True)                # output compositional structure dir
    parser.add_argument('--epochs', type=int, default=100)                      # num epochs for map training
    parser.add_argument('--type', type=str, required=True)                      # map parameterization for (f,g)
    parser.add_argument('--train_data_path', type=str)                          # training dataset path
    parser.add_argument('--eval_data_path', type=str)                           # evaluation dataset path
    parser.add_argument('--maps_dir', type=str)                                 # directory to save trained maps
    args = parser.parse_args()

    # should have f,g for each subtask (mdp)?
    input_mdp = MDP(env=args.input_env, 
                    dir=args.input_dir, 
                    policy=args.input_policy,
                    tasks=args.input_tasks,
                    baseline_mode=('baseline' in args.input_env.lower()),
                    prefix='best_model',
                )
    output_mdp = MDP(env=args.output_env, 
                     dir=args.output_dir, 
                     policy=args.output_policy,
                     tasks=args.output_tasks,
                     baseline_mode=('baseline' in args.output_env.lower()),
                     prefix='best_model',
                )
   
    train_data_path = args.train_data_path if args.train_data_path is not None else None

    maps = learn_maps(input_mdp, 
                      output_mdp, 
                      data_path=train_data_path,
                      map_type=args.type, 
                      epochs=args.epochs, 
                      maps_dir=args.maps_dir)

    # epsilon = evaluate(input_mdp, output_mdp, f, g, data_path=args.eval_data_path)