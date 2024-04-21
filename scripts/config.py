import os
import json
from time import strftime
from ml_collections import config_dict
import ml_collections.config_dict
    
def create_maps_config():
    mdp_cfg = config_dict.ConfigDict({
        'input_env': 'CompLift-IIWA',
        'output_env': 'CompLift-Panda',
        'input_policy': 'PPO',
        'output_policy': 'PPO',
        'input_tasks': ['reach', 'grasp', 'lift'],
        'output_tasks': ['reach', 'grasp', 'lift'],
        'input_path': 'experiments/PPO/CompLift-IIWA/20231219-145654-id-7627/models',
        'output_path': 'experiments/PPO/CompLift-Panda/20231222-172458-id-1179/models',
    })
    train_cfg = config_dict.ConfigDict({
        'maps_path': None,
        'train_data_path': None,
        'eval_data_path': None,
        'batch_size': 1,
        'epochs': int(100),
        'eval_every_epochs': 5,
        'log_every_epochs': 10,
        'map_type': 'mlp',
    })
    opt_cfg = config_dict.ConfigDict({
        'learning_rate': 1e-3,
    })
    cfg = config_dict.ConfigDict({
        'mdp': mdp_cfg,
        'train': train_cfg,
        'opt': opt_cfg,
    })

    return cfg

# def train_cfg():
#     cfg = config_dict.FrozenConfigDict({
#         'env': 'CompLift-IIWA',
#         'policy': 'SAC',
#         'discount_rate': 0.96,
#         'epochs': int(2.5e5),
#         'eval_freq': int(1e4),
#         'eval_ep': 20,
#         'save_freq': int(5e3),
#         'resume_training': False,
#         'skip_tasks': [],
#         'success_thres': 0.95,
#         'model_prefix': '',
#         'dir': '',
#     })

#     return cfg

def train_cfg():
    cfg = config_dict.FrozenConfigDict({
        'env': 'CompPickPlaceCan-IIWA',
        'policy': 'SAC',
        'discount_rate': 0.96,
        'epochs': int(2.5e5),
        'eval_freq': int(1e4),
        'eval_ep': 20,
        'save_freq': int(5e3),
        'resume_training': False,
        'skip_tasks': [],
        'success_thres': 0.95,
        'model_prefix': '',
        'dir': '',
        'device': 'cpu', # or 'cuda' for GPU
    })

    return cfg

# def eval_cfg():
#     cfg = config_dict.FrozenConfigDict({
#         'env': 'CompLift-IIWA',
#         'dir': 'experiments/PPO/CompLift-IIWA/20231219-145654-id-7627/models',
#         'policy': 'PPO',
#         'eps': 2,
#         'tasks': 'all',
#         'prefix': '*',
#     })

#     return cfg

def eval_cfg(): # Test
    cfg = config_dict.FrozenConfigDict({
        'env': 'CompPickPlaceCan-IIWA',
        'dir': 'experiments/SAC/CompPickPlaceCan-IIWA/20240419-234354-id-6073/models',
        'policy': 'SAC',
        'eps': 10,
        'tasks': ['reach', 'grasp', 'lift'],
        'prefix': '*',
    })

    return cfg

def save_cfg(cfg):
    cfg_js = cfg.to_json_best_effort()
    save_path = 'scripts/config'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    cfg_file = os.path.join(save_path, f'maps_cfg_{strftime("%H%M%S")}.js')
    with open(cfg_file, "w") as outfile:
        json.dump(cfg_js, outfile)

if __name__ == '__main__':
    cfg = create_maps_config()
    save_cfg(cfg)
    
