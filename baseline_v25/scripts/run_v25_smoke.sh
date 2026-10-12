#!/usr/bin/env bash
# v25 baseline smoke: same recipe as artifacts/three_city_full_20260825 smoke,
# run from the frozen v25 code in baseline_v25/. Takes a few minutes.
set -euo pipefail
ROOT=/Users/huali/Desktop/RideResponse
BASE=$ROOT/baseline_v25
PY=${PY:-/Users/huali/opt/anaconda3/envs/rideshareDT/bin/python3.11}
OUT=$BASE/runs/smoke_boston_seed2601
mkdir -p "$OUT"
cd "$BASE"
"$PY" -u Core.py --market Boston --firm1_mode RL --firm2_mode adaptive_best_response_aggressive \
  --run_experiment --credit_assignment_v25 --deterministic_experiment_seed --seed 2601 --experiment_seed 2601 \
  --policy_objective profit_50_competitiveness_50 --pool 100 \
  --threshold_profile_source cached --threshold_cache_path "$ROOT/artifacts/three_city_full_20260825/smoke/profiles.jsonl" \
  --calibration_json "$BASE/data/processed/boston/calibration.json" \
  --train_timesteps 40 --train_customers 20 --train_steps_per_day 4 \
  --firm1_action_interval_steps 5 --firm2_action_interval_steps 5 \
  --ppo_update_interval_days 5 --ppo_min_rollout_transitions 8 --ppo_batch_size 8 --ppo_update_epochs 2 \
  --training_stage_market_fraction 0.10 --training_stage_static_fraction 0.10 --training_stage_dynamic_fraction 0.80 \
  --training_dynamic_episode_days 5 \
  --checkpoint_validation_interval_days 20 --checkpoint_validation_horizon 20 --checkpoint_validation_customers 20 --checkpoint_validation_openings 1 \
  --eval_timesteps 20 --eval_customers 20 --benchmark_suite \
  --report_prefix "$OUT/run" --trained_model_out "$OUT/policy.pt" --trained_model_id v25-baseline-smoke-boston \
  --model_registry_dir "$OUT/registry" 2>&1 | tee "$OUT/run.log"
echo "exit=${PIPESTATUS[0]}" | tee "$OUT/exit_status.txt"
