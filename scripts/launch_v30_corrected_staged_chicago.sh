#!/usr/bin/env bash
# Run independently in Terminal; all learning/evaluation output goes to files.
set -euo pipefail
cd /Users/huali/Desktop/RideResponse
export PATH="/Users/huali/opt/anaconda3/bin:$PATH"
OUT="artifacts/v30/chicago_staged_seed300101"
SMOKE="artifacts/v30/chicago_smoke_seed300097"
# A unique directory prevents accidental duplicate/replacement runs.
mkdir "$OUT"
export MPLCONFIGDIR="$PWD/$OUT/matplotlib"
mkdir -p "$MPLCONFIGDIR"
printf '%s\n' "$$" > "$OUT/launcher_pid.txt"
date -u +%FT%TZ > "$OUT/started_at.txt"
finish() {
  result=$?
  printf '%s\n' "$result" > "$OUT/exit_status.txt"
  date -u +%FT%TZ > "$OUT/finished_at.txt"
  if [[ "$result" -eq 0 ]]; then
    printf '%s\n' complete > "$OUT/status.txt"
  else
    printf '%s\n' failed > "$OUT/status.txt"
  fi
}
trap finish EXIT
shasum -a 256 Core.py PPOAgent.py platform_mdp.py pricing_models.py \
  digital_twin/*.py scripts/run_v30_smoke_chicago.sh \
  scripts/launch_v30_corrected_staged_chicago.sh \
  data/processed/chicago/calibration.json data/processed/chicago/trip_bank.csv \
  artifacts/v28_thresholds/chicago/native_uncapped_primary.jsonl \
  > "$OUT/source_and_data.sha256"
printf '%s\n' corrected_smoke > "$OUT/status.txt"
printf 'Corrected smoke started. Logs: %s/smoke.log\n' "$OUT"
V30_SEED=300097 V30_OUT="$SMOKE" bash scripts/run_v30_smoke_chicago.sh \
  > "$OUT/smoke.log" 2>&1
# Verify the observed failure directly: validation must not erase any part
# of the four 15-day terminal windows. CSV values are daily block averages.
python - "$SMOKE/chicago_training_diagnostics.csv" <<'PY'
import csv
import math
import sys
with open(sys.argv[1], newline="") as stream:
    rows = list(csv.DictReader(stream))
windows = [(i + 1, float(row.get("reward_terminal_observations") or 0))
           for i, row in enumerate(rows)
           if float(row.get("reward_terminal_observations") or 0) > 0]
expected = [(15, 15.0), (30, 15.0), (45, 15.0), (60, 15.0)]
if len(rows) != 60 or windows != expected:
    raise RuntimeError(f"Incomplete terminal histories: {windows}; expected {expected}")
for row in rows:
    for key in ("avg_reward", "avg_profit_per_request", "avg_profit_advantage_per_request"):
        if not math.isfinite(float(row[key])):
            raise RuntimeError(f"Nonfinite smoke metric: {key}")
print("Corrected smoke: all 60 training days preserved in terminal credit.")
PY
# Refuse to silently train a different revision than the verified smoke.
shasum -a 256 -c "$OUT/source_and_data.sha256" > "$OUT/source_verification.log"
printf '%s\n' staged_training_then_evaluation > "$OUT/status.txt"
printf 'Starting 1,200-day staged training and evaluation. Log: %s/run.log\n' "$OUT"
V30_SEED=300101 V30_OUT="$OUT" bash scripts/run_v30_smoke_chicago.sh \
  --training_stage_market_fraction 0.20 \
  --training_stage_static_fraction 0.20 \
  --training_stage_dynamic_fraction 0.60 \
  --training_dynamic_episode_days 120 \
  --train_timesteps 1200 \
  --train_steps_per_day 10 \
  --train_customers 500 \
  --firm1_action_interval_steps 10 \
  --firm2_action_interval_days 10 \
  --ppo_update_interval_days 5 \
  --ppo_min_rollout_transitions 24 \
  --ppo_batch_size 24 \
  --ppo_update_epochs 4 \
  --eval_timesteps 400 \
  --eval_customers 500 \
  --checkpoint_validation_horizon 400 \
  --checkpoint_validation_interval_days 240 \
  --checkpoint_validation_customers 500 \
  --checkpoint_validation_openings 3 \
  --benchmark_suite \
  --baseline_suite "static_grid,anchor_hold" \
  --static_baseline_multipliers "1.0,1.25,1.5,1.75,2.0,3.0,4.0" \
  --profiles_log_limit 10000 \
  > "$OUT/run.log" 2>&1
printf 'Staged training and evaluation completed. Artifacts: %s\n' "$OUT"
