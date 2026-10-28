#!/bin/zsh
set -eu

BASE=/Users/huali/Desktop/RideResponse/rideshare_v26r_revision7
PYTHON=/Users/huali/Desktop/RideResponse/.venv/bin/python
PREFLIGHT=/Users/huali/Desktop/RideResponse/artifacts/revision7_pretraining_20261018
FULL=/Users/huali/Desktop/RideResponse/artifacts/revision7_full_20000_20261018

cd "$BASE"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
export MPLCONFIGDIR="$FULL/mpl_cache"
unset PYTHONPATH

/bin/zsh "$BASE/scripts/start_revision7_pretraining.command"
"$PYTHON" -u "$BASE/scripts/validate_revision7_pretraining.py" \
  --results "$PREFLIGHT/audit/results.json" \
  --output "$PREFLIGHT/training_eligibility.json"

exec /usr/bin/caffeinate -i "$PYTHON" -u "$BASE/scripts/run_v26r_multicity.py" \
  --output "$FULL" \
  --days 20000 \
  --update-days 1 \
  --rollout-transitions 30 \
  --lead-weight 0.6 \
  --learner-revision 4 \
  --ppo-rollout-episodes 2 \
  --city-calibration-root "$PREFLIGHT/audit/primary_calibration" \
  --figures
