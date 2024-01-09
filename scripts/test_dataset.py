import numpy as np
from argparse import ArgumentParser

def test_dataset(data_path: str):
    data = np.load(data_path, allow_pickle=True).item()
    # num_samples = # TODO parse data_path to get num_samples
    # rnd_idx = np.random.randint(0, num_samples)
    rnd_idx = 563
    rnd_rollout_data = data[f'rollout_{rnd_idx}']
    print(f'Run {rnd_idx} keys: {rnd_rollout_data.keys()}') # For testing

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)

    args = parser.parse_args()

    test_dataset(args.data_path)