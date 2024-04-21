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
import ml_collections.config_dict
import numpy as np
import matplotlib.pyplot as plt

from time import strftime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import ml_collections

from config import create_maps_config

from utils.wrapper import MDP, MapsWrapper
from utils.loss_function import epsilonRlxBisimLoss
from utils.dataset_wrappers import RobotDataset, RobotDatasetLive

# torch device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def learn_maps(cfg: ml_collections.ConfigDict, M: MDP, G: MDP) -> MapsWrapper:
    '''
        Learn maps (f: S -> S', g: A -> A') to test domain-agnostic generalization

        :param M: input MDP
        :param G: output MDP
        :param map_type: specify map parameterization (e.g. linear, mlp)
        :param epochs: number of training epochs
        :param maps_dir: directory of previously trained maps (f,g)
    '''
    maps = MapsWrapper(M, G, map_type=cfg.train.map_type, maps_dir=cfg.train.maps_path, opt_params=cfg.opt)

    # if skipping training
    if cfg.train.epochs == 0: return maps
    
    # setup tensorboard
    log_path = os.path.join(maps.save_path, 'log')
    if not os.path.exists(log_path): os.makedirs(log_path)
    writer = SummaryWriter(log_path)

    # Set f and g to training mode + send to device (GPU)
    maps.train()
    maps.to(device)

    maps.step = {f'{task}': 0 for task in G.tasks}
    for epoch in range(cfg.train.epochs):
        cumulative_loss = {f'{task}': 0 for task in G.tasks}
        if cfg.train.train_data_path is None:
            # Collect data live
            training_data = DataLoader(RobotDatasetLive(M), batch_size=cfg.train.batch_size, shuffle=False)
        else:
            training_data = DataLoader(RobotDataset(M, cfg.train.train_data_path), batch_size=cfg.train.batch_size, shuffle=False)
        # Batch training
        for data, task in training_data:
            # print(task[0])
            current_task = task[0]
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
            
            data.to(device)

            # Relaxed bisimulation loss
            loss_fn = epsilonRlxBisimLoss
            loss = loss_fn(f_subtask, 
                           g_subtask, 
                           mdp_1=M, 
                           mdp_2=G,
                           task=current_task, 
                           samples_1=data, 
                           device=device,
                    )

            loss.backward()
            
            f_subtask_optimizer.step()
            g_subtask_optimizer.step()

            cumulative_loss[current_task] += loss.item()
            maps.epoch[current_task] += 1

        step_loss = [v / (cfg.train.batch_size * len(training_data)) for v in cumulative_loss.values()]
        print(f"Epoch {maps.f_epoch + epoch}, losses: {step_loss}")

        is_last_epoch = epoch == cfg.train.epochs - 1
        
        if epoch % cfg.train.log_every_epochs == 0 or is_last_epoch:
            for task in G.tasks:
                writer.add_scalar(f"f loss/train - {task}", cumulative_loss[f'{task}'] / (cfg.train.batch_size * len(training_data)), maps.f_epoch + maps.epoch[f'{task}'])
                writer.add_scalar(f"g loss/train - {task}", cumulative_loss[f'{task}'] / (cfg.train.batch_size * len(training_data)), maps.g_epoch + maps.epoch[f'{task}'])
            writer.flush()

        if epoch % cfg.train.eval_every_epochs == 0 or is_last_epoch:
            epsilon = evaluate(M, G, maps)
            for task in G.tasks:
                writer.add_scalar(f"Validation Epsilon - {task}", epsilon[f'{task}'], epoch)
            writer.flush()

    # Save models
    maps.save()
    
    # Log to tensorboard
    writer.flush()
    writer.close()

    return maps

def evaluate(M: MDP, G: MDP, maps: MapsWrapper, data_path: str = None):
    '''
        Evaluate maps f and g on evaluation set

        :param M: input MDP
        :param G: compositional structure
        :param f: trained map f: S_1 -> S_2
        :param g: trained map g: A_1 -> A_2
    '''
    print(f'Evaluating f and g')
    traj_data = {f'{task}': [] for task in G.tasks}
    losses = {f'{task}': [] for task in G.tasks}
    epsilon = {f'{task}': 0 for task in G.tasks}


    # Set maps to evaluation mode
    maps.to(device)
    maps.eval()

    if data_path is None:
        test_set = DataLoader(RobotDatasetLive(M), shuffle=False)
    else:
        test_set = DataLoader(RobotDataset(M, data_path), shuffle=False)
    
    task = []
    with torch.no_grad():
        for data, tasks in test_set:
            # print(tasks)
            current_task = tasks[0]
            f = maps.f[current_task]
            g = maps.g[current_task]
            loss_fn = epsilonRlxBisimLoss
            loss = loss_fn(f=f,
                        g=g,
                        mdp_1=M,
                        mdp_2=G,
                        task=current_task,
                        samples_1=data,
                        device=device,
            )

            loss = loss.detach().numpy().item() # this is a singleton
            if loss > epsilon[current_task]: epsilon[current_task] = loss

            losses[current_task].append(loss)

            if traj_data[current_task] == []: 
                traj_data[current_task] = data
            else:
                traj_data[current_task] = torch.vstack((traj_data[current_task], data))
            task.append(current_task)

    """
        TODO: print plots of s versus f(s) (where f: S1 -> S2), and a versus g(a) (where g: A1 -> A2)
        What is f(s) and g(a) suppose to be?
    """
    for task in G.tasks:
        traj_data[task] = traj_data[task].detach()
    
    fig, (ax_1, ax_2) = plt.subplots(2, 1)
    lengths = [len(traj_data[task]) for task in G.tasks]

    # state_labels = ['hand x', 'hand y', 'hand z']
    # ax_1.plot(np.arange(lengths[0]), traj_data['reach'][:,0:3], label=state_labels)
    # ax_1.plot(np.arange(lengths[0]), maps.f['reach'](traj_data['reach'][:,0:6].cuda()).detach().cpu()[:,0:3], label=['f hand x', 'f hand y', 'f hand z'])
    # ax_1.legend()
    # action_labels = ['x', 'y', 'z', 'g']
    # ax_2.plot(np.arange(lengths[0]), traj_data['reach'][:,6:10], label=action_labels)
    # ax_2.plot(t, maps.g['reach'](traj_data[:,6:10].cuda()).detach().cpu(), label=['g x', 'g y', 'g z', 'g g'])
    # ax_2.legend()
    # plt.show()

    print('losses:', losses.items())
    print('epsilon:', epsilon.items())

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

    data = DataLoader(RobotDatasetLive(M), batch_size=M.horizon)
    # get states and actions
    # transform states and actions using (f,g)

    epsilon = 0
    for data in data:
        M_states = data[:,0:6]
        M_actions = data[:,6:10]
        M_rwds = data[:,-1]
        
        G_states = f(M_states)
        G_actions = g(M_actions)

        e = epsilonRlxBisimLoss(f, g, M, G, data, device) # TODO: for this sample

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
    cfg = create_maps_config()
    mdp_cfg = cfg.mdp
    # should have f,g for each subtask (mdp)?
    input_mdp = MDP(env=mdp_cfg.input_env, 
                    dir=mdp_cfg.input_path, 
                    policy=mdp_cfg.input_policy,
                    tasks=mdp_cfg.input_tasks,
                    baseline_mode=('baseline' in mdp_cfg.input_env.lower()),
                    prefix='best_model',
                )
    output_mdp = MDP(env=mdp_cfg.output_env, 
                     dir=mdp_cfg.output_path, 
                     policy=mdp_cfg.output_policy,
                     tasks=mdp_cfg.output_tasks,
                     baseline_mode=('baseline' in mdp_cfg.output_env.lower()),
                     prefix='best_model',
                )
   
    train_data_path = cfg.train.train_data_path

    maps = learn_maps(cfg, input_mdp, output_mdp)

    epsilon = evaluate(input_mdp, output_mdp, maps, data_path=cfg.train.eval_data_path)