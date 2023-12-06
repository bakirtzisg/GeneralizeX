# Test IIWA baseline lift policy
python scripts/eval.py --env=BaselineCompPickPlaceCan-IIWA --dir=results/BaselineCompPickPlaneCan-IIWA 
# Test IIWA compositional RL lift policy
python scripts/eval.py --env=CompLift-IIWA --dir=results/CompLift-IIWA
# Test Panda compositional RL lift policy
python scripts/eval.py --env=CompLift-Panda --dir=results/CompLift-Panda
# Test IIWA reach RL policy by itself (reach is a subtask of lift)
python scripts/eval.py --env=CompLift-Panda --dir=results/CompLift-Panda --tasks reach