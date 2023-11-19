# Train baseline controller for lift environment (IIWA)
python3 scripts/train.py --env_name=BaselineCompLift-IIWA --epochs=250000

# Continue training example
python scripts/train.py --dir=experiments/BaselineCompLift-IIWA/20231114-090225-id-1024/ --env_name=BaselineCompLift-IIWA --epochs=100000 --resume_training --baseline --model_prefix=rl_model_250000

# Train compositional RL controller for lift environment (IIWA robot)
 python scripts/train.py --env=CompLift-IIWA --epochs=250000 --eval_freq=5000