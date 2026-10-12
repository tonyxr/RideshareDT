#!/usr/bin/env bash
# Full v25 training run with the exact three_city_full_20260825 recipe.
# Usage: scripts/run_v25_city.sh <nyc|boston|chicago> <objective> <seed>
#   objective: profit_maximization | competitiveness | profit_50_competitiveness_50 | ...
set -euo pipefail
CITY=${1:?city}; OBJ=${2:-profit_maximization}; SEED=${3:-4500}
case "$CITY" in nyc) MARKET="New York City";; boston) MARKET="Boston";; chicago) MARKET="Chicago";; *) echo "bad city"; exit 2;; esac
ROOT=/Users/huali/Desktop/RideResponse
BASE=$ROOT/baseline_v25
PY=${PY:-/Users/huali/opt/anaconda3/envs/rideshareDT/bin/python3.11}
OUT=$BASE/runs/${CITY}_${OBJ}_seed${SEED}
mkdir -p "$OUT"
cd "$BASE"
"$PY" -u Core.py --market "$MARKET" --choice_mode cognitive --firm1_mode RL --firm2_mode adaptive_best_response_aggressive \
  --run_experiment --credit_assignment_v25 --deterministic_experiment_seed --deterministic_torch \
  --seed "$SEED" --experiment_seed "$SEED" --policy_objective "$OBJ" --pool 20000 \
  --threshold_profile_source cached --threshold_cache_path "$ROOT/artifacts/three_city_full_20260825/raw/thresholds/$CITY/primary.jsonl" \
  --calibration_json "$BASE/data/processed/$CITY/calibration.json" \
  --train_timesteps 9150 --train_customers 100 --train_steps_per_day 4 \
  --firm1_action_interval_steps 5 --firm2_action_interval_steps 5 \
  --ppo_update_interval_days 10 --ppo_min_rollout_transitions 732 --ppo_batch_size 48 --ppo_update_epochs 8 \
  --training_stage_market_fraction 0.05 --training_stage_static_fraction 0.05 --training_stage_dynamic_fraction 0.90 \
  --training_dynamic_episode_days 15 --training_opening_tariff_randomization 0.15 \
  --checkpoint_validation_interval_days 915 --checkpoint_validation_horizon 300 --checkpoint_validation_customers 100 --checkpoint_validation_openings 2 \
  --eval_timesteps 1000 --eval_customers 200 --eval_policy_mode argmax_raw --require_competitive_backtest --benchmark_suite \
  --report_prefix "$OUT/$CITY" --trained_model_out "$OUT/policy.pt" --trained_model_id "v25-baseline-$CITY-$OBJ-seed$SEED" \
  --model_registry_dir "$OUT/registry" 2>&1 | tee "$OUT/run.log"
echo "exit=${PIPESTATUS[0]}" | tee "$OUT/exit_status.txt"
