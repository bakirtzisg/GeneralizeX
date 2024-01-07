# GeneralizeX

## Setup
```
python3 -m venv env
source env/bin/activate
pip install -e .
```
Setup robosuite private macro file:
```
python $PWD/env/lib/python3.9/site-packages/robosuite/scripts/setup_macros.py
```

## Run
### Compositional RL Training
There is the option to either train the baseline policy or the compositional policy. To train the baseline policy, run with the flag `--baseline`.
```
python scripts/train.py
```

Flags (see `scripts\train.py` for more flags)
- `--env`: name of gymnasium environment
- `--dir`: directory for saved models (When resuming training, this is the directory with your trained models).
- `--model_prefix`: optional prefix for saved models
- `--baseline`: flag to train baseline policy. If this flag is not provided then by default train the compositional policy.
- `--resume_training`: flag to load trained models and continue training. If this flag is not provided then by default start training a new policy.

Example: train baseline lift policy (IIWA arm)
```
source shell_scripts/train.sh
```

### Evaluation
```
python scripts/eval.py --env=ENV_NAME --dir=PATH_TO_MODEL
```
Example
```
source shell_scripts/eval.sh
```
## Experiments

### Algorithm 1
E.g. Domain-specific generalization between reach policies on two different robot arms.
```
    python scripts/learn_maps.py --input_env=CompLift-IIWA --input_policy=PPO --input_dir=experiments/PPO/CompLift-IIWA/20231219-145654-id-7627/models --output_env=CompLift-Panda --output_policy=PPO --output_dir=experiments/PPO/CompLift-Panda/20231222-172458-id-1179/models --epochs=1000 --type=linear
``` 
