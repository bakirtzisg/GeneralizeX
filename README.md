# GeneralizeX

## Setup
```
python3 -m venv env
source env/bin/activate
pip install -e .
```

robosuite private macro file:
```
python $PWD/env/lib/python3.9/site-packages/robosuite/scripts/setup_macros.py
```

To resolve PyTorch 2.1 ImportError (may not be necessary):
```
source shell_scripts/fix_torch.sh
```

## Run
### Training
There is the option to either train the baseline policy or the compositional policy. To train the baseline policy, run with the flag `--baseline`.
```
python scripts/train.py
```

Flags
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
````