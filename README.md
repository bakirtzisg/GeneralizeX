# GeneralizationX

## Setup
```
python3 -m venv env
source env/bin/activate
python3 -m pip install .
```

To resolve PyTorch 2.1 ImportError:
```
source shell_scripts/setup.sh
```

robosuite private macro file:
```
python $PWD/env/lib/python3.9/site-packages/robosuite/scripts/setup_macros.py
```

## Run
Train baseline lift policy (IIWA arm):
```
source shell_scripts/train.sh
```
Evaluate policy:
```
source shell_scripts/eval.sh
```