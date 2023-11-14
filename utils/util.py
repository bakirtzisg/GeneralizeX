import os
import glob

def get_policy(model_path, suffix=''):
    task_model_path = glob.glob(os.path.join(model_path, f'{suffix}*.zip'))
    assert len(task_model_path) == 1, 'Found multiple task models!'

    return task_model_path[0]
