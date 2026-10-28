#!/usr/bin/env bash
# Evaluate a trained v26r policy (and the Gate-1 comparators) on one scripted scenario with CRN tapes (Mac, real torch).
# Usage: scripts/run_v26r_scenario_eval.sh <city> <policy.pt> <scenario> <out.json> [openings] [horizon_steps] [regime|full]
#   The 7th argument must match the training configuration ("full" adds the persistence channels and the memory rival).
set -euo pipefail
CITY=${1:?}; MODEL=${2:?}; SCEN=${3:?}; OUT=${4:?}; OPEN=${5:-20}; HOR=${6:-150}; CHANNELS=${7:-full}
source "$(dirname "$0")/v26r_full_flags.sh"
if [[ "$CHANNELS" == "full" ]]; then EXTRA="$V26R_FULL_FLAGS --policy_sweep_rivals $V26R_FULL_RIVAL"; RIVAL=$V26R_FULL_RIVAL; else EXTRA="--spillover_price_aware"; RIVAL=adaptive_best_response_aggressive; fi
case "$CITY" in nyc) MARKET="New York City";; boston) MARKET="Boston";; chicago) MARKET="Chicago";; *) echo "bad city"; exit 2;; esac
ROOT=/Users/huali/Desktop/RideResponse; BASE=$ROOT/rideshare_v26r; PY=${PY:-$ROOT/.venv/bin/python}
cd "$BASE"
"$PY" -u Core.py --market "$MARKET" --choice_mode cognitive --firm1_mode RL --firm2_mode $RIVAL \
  --run_experiment --credit_assignment_v25 --v26 --deterministic_experiment_seed --seed 4500 --experiment_seed 4500 \
  --policy_objective profit_maximization --pool 20000 --threshold_profile_source cached \
  --threshold_cache_path "$ROOT/artifacts/v26_confirmatory/thresholds/$CITY/llm_numeric_primary.jsonl" \
  --calibration_json "$ROOT/data/processed/$CITY/calibration.json" --gpt_threshold_mode llm_numeric \
  --train_timesteps 0 --train_customers 100 --train_steps_per_day 6 --eval_customers 100 \
  --firm1_action_interval_steps 6 --firm2_action_interval_steps 1 \
  --regime_market --regime_features --shock_scenario "scenarios/$SCEN.json" \
  --tariff_targets "0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.8,2.0" \
  --trained_model_in "$MODEL" --eval_only \
  --policy_sweep "rl_actor;static:1.3;static:1.4;static:1.5;static:1.6" \
  --policy_sweep_openings "$OPEN" --eval_timesteps "$HOR" --policy_sweep_out "$OUT" --report_prefix runs/sweep_tmp/run $EXTRA
