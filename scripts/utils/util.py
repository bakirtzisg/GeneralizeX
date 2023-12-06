import os
import glob

from stable_baselines3 import SAC

def find_file(dir, prefix=''):
    file = glob.glob(os.path.join(dir, prefix))
    assert len(file) == 1, f'Found {len(file)} files!'

    return file[0]

def load_sac_policy(dir, baseline, tasks):
    agents = {}
    if baseline:
        TASK_MODEL_FILE = find_file(dir, 'baseline_final*.zip')
        agents['baseline'] = SAC.load(TASK_MODEL_FILE)
    else:
        for task in tasks:
            TASK_MODEL_FILE = find_file(dir, f'{task}*.zip')
            agents[task] = SAC.load(TASK_MODEL_FILE)

    return agents