#!/usr/bin/env bash
# One native-Terminal job: pilot -> fresh full budget -> benchmark report.
set -euo pipefail
cd /Users/huali/Desktop/RideResponse
export PATH="/Users/huali/opt/anaconda3/bin:$PATH"
SEED="${V30_SEED:-300127}"
OUT="${V30_PIPELINE_OUT:-artifacts/v30/chicago_corrected_pipeline_seed${SEED}}"
mkdir "$OUT" # never overwrite an earlier experiment
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
shasum -a 256 ./*.py digital_twin/*.py \
  scripts/run_v30_smoke_chicago.sh scripts/run_v30_full_chicago.sh \
  scripts/launch_v30_corrected_pipeline.sh scripts/report_v30_profit_dominance.py \
  data/processed/chicago/calibration.json data/processed/chicago/trip_bank.csv \
  artifacts/v28_thresholds/chicago/native_uncapped_primary.jsonl \
  > "$OUT/source_and_data.sha256"
tar -czf "$OUT/source_snapshot.tgz" ./*.py digital_twin/*.py \
  scripts/run_v30_smoke_chicago.sh scripts/run_v30_full_chicago.sh \
  scripts/launch_v30_corrected_pipeline.sh scripts/report_v30_profit_dominance.py
python -m pip freeze > "$OUT/python_packages.txt"
printf '%s\n' staged_training_and_validation > "$OUT/status.txt"
printf 'Starting 1,200-day corrected staged pilot. Log: %s/staged.log\n' "$OUT"
V30_SEED="$SEED" V30_OUT="$OUT/staged" bash scripts/run_v30_smoke_chicago.sh \
  --training_stage_market_fraction 0.20 \
  --training_stage_static_fraction 0.20 \
  --training_stage_dynamic_fraction 0.60 \
  --training_dynamic_episode_days 120 \
  --train_timesteps 1200 --train_steps_per_day 10 --train_customers 500 \
  --firm1_action_interval_steps 10 --firm2_action_interval_days 10 \
  --ppo_update_interval_days 5 --ppo_min_rollout_transitions 24 \
  --ppo_batch_size 24 --ppo_update_epochs 4 \
  --eval_timesteps 400 --eval_customers 500 \
  --checkpoint_validation_horizon 400 \
  --checkpoint_validation_interval_days 240 \
  --checkpoint_validation_customers 500 --checkpoint_validation_openings 3 \
  --eval_policy_mode sample_raw --profiles_log_limit 10000 \
  > "$OUT/staged.log" 2>&1
date -u +%FT%TZ > "$OUT/staged_finished_at.txt"
# Provenance check only: profit performance does not gate the next phase.
shasum -a 256 -c "$OUT/source_and_data.sha256" > "$OUT/source_verification.log"
printf '%s\n' full_budget_training_and_evaluation > "$OUT/status.txt"
printf 'Starting fresh 6,000-day full-budget training and benchmarks. Log: %s/full.log\n' "$OUT"
# Fresh actor/optimizer: do not reload the pilot or an older differential critic.
# This script opens final-test profiles only for the full run's final evaluation.
V30_SEED="$SEED" V30_OUT="$OUT/full" bash scripts/run_v30_full_chicago.sh \
  > "$OUT/full.log" 2>&1
printf '%s\n' summarizing > "$OUT/status.txt"
python scripts/report_v30_profit_dominance.py "$OUT/full/chicago_run_config.json" \
  --output "$OUT/profit_dominance.json" > "$OUT/profit_dominance.txt"
printf 'Complete. Profit-dominance report: %s/profit_dominance.txt\n' "$OUT"
