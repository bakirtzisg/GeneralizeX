import os
import glob

def find_file(dir, prefix=''):
    file = glob.glob(os.path.join(dir, prefix))
    assert len(file) == 1, f'Found {len(file)} files!'

    return file[0]
