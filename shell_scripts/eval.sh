python scripts/eval.py --dir=results/BaselineCompPickPlaceCan-v1/best

# Test IIWA lift policy (by itself)
python3 scripts/eval.py --dir=results/BaselineCompPickPlaneCan-IIWA --env=BaselineCompPickPlaceCan-IIWA --task lift
