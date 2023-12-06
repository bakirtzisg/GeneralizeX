# Train baseline controller for lift environment (IIWA)
python scripts/train.py --env=BaselineCompLift-IIWA --epochs=250000 --baseline
# Train compositional RL controller for lift environment (IIWA robot)
python scripts/train.py --env=CompLift-IIWA
# Continue training example
python scripts/train.py --env=BaselineCompLift-IIWA --dir=experiments/BaselineCompLift-IIWA/20231114-090225-id-1024/  --epochs=100000 --resume_training --baseline --model_prefix=rl_model_250000
# Only train reach policy example (for compositional RL training). To do so, we skip training for the other subtasks grasp and lift:
python scripts/train.py --env=CompLift-IIWA --skip_tasks grasp lift