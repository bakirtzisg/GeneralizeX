python scripts/eval.py --path=results/BaselineCompPickPlaceCan-v1/best

# Test IIWA lift policy (by itself)
python3 scripts/eval.py --path=results/BaselineCompPickPlaneCan-IIWA --env=BaselineCompPickPlaceCan-IIWA --task lift
