#!/usr/bin/env bash
# v26r RL training on the Mac (real torch).  Same recipe as the v26 confirmatory
# runs, plus the regime market, daily decisions, extended tariff grid, public
# regime signals and recurrent categorical PPO (no external exploration mixture).
# Usage: scripts/run_v26r_train.sh <nyc|boston|chicago> <seed> <smoke|full> <regime|full> <city_calibration.json>
#   4th arg "regime" = v26r regime market only;
#   "full" (default) = regime market + the three persistence channels
#   (supply v2 with fare-based pay, loyalty v2, memory rival) + persistence
#   proxies in the frame + 60-day episodes.
set -euo pipefail
CITY=${1:?city}; SEED=${2:-4500}; MODE=${3:-full}; CHANNELS=${4:-full}
case "$CITY" in nyc) MARKET="New York City";; boston) MARKET="Boston";; chicago) MARKET="Chicago";; *) echo "bad city"; exit 2;; esac
ROOT=/Users/huali/Desktop/RideResponse
BASE=$ROOT/rideshare_v26r_revision5
PY=${PY:-$ROOT/.venv/bin/python}
CALIBRATION=${5:?provide the city_calibration.json from the completed revision-5 audit as argument 5}
# Budget is in DECISIONS: one decision per day, 180 decisions per PPO update.
#   smoke  = 1,200 days  ->   ~7 updates (plumbing check only; the policy barely leaves the warm-start prior)
#   full   = 24,000 days -> ~130 updates (the v26 runs had 68 updates at a 4-hour cadence; daily decisions need 6x the days)
if [[ "$MODE" == "smoke" ]]; then DAYS=1200; EVAL=120; VAL_INT=600; VAL_HOR=60; POOL=2000; CUST=50; else DAYS=${DAYS:-24000}; EVAL=1200; VAL_INT=2400; VAL_HOR=600; POOL=20000; CUST=100; fi
OUT=$BASE/runs/train/${CITY}_seed${SEED}_${MODE}_${CHANNELS}
if [[ "$CHANNELS" == "full" ]]; then
  # loyalty pool ~ one day of returning requests (CUST x 6 steps x 0.6)
  POOL_RIDERS=$(( CUST * 6 * 6 / 10 ))
  CHANNEL_FLAGS="--supply_model v2 --driver_pay_basis fare --loyalty_model v2 --loyalty_pool_size $POOL_RIDERS \
    --persistence_features --training_opponent_modes adaptive_best_response_memory,adaptive_best_response_aggressive \
    --training_dynamic_episode_days 60 --ppo_gae_lambda 0.90"
  RIVAL=adaptive_best_response_memory; WARM=auto
else
  CHANNEL_FLAGS="--training_dynamic_episode_days 15"
  RIVAL=adaptive_best_response_aggressive; WARM=auto
fi
mkdir -p "$OUT"
cd "$BASE"
"$PY" -u Core.py --market "$MARKET" --choice_mode cognitive --firm1_mode RL --firm2_mode $RIVAL \
  --run_experiment --credit_assignment_v25 --v26 --deterministic_experiment_seed --deterministic_torch \
  --learning_corrections --learner_revision 4 --stage_specific_rewards --opponent_profit_lead_weight 0.6 \
  --city_calibration_artifact "$CALIBRATION" \
  --seed "$SEED" --experiment_seed "$SEED" --policy_objective profit_maximization --pool $POOL \
  --threshold_profile_source cached --threshold_cache_path "$ROOT/artifacts/v26_confirmatory/thresholds/$CITY/llm_numeric_primary.jsonl" \
  --calibration_json "$ROOT/data/processed/$CITY/calibration.json" --gpt_threshold_mode llm_numeric \
  --train_timesteps $DAYS --train_customers $CUST --train_steps_per_day 6 \
  --firm1_action_interval_steps 6 --firm2_action_interval_steps 6 \
  --ppo_update_interval_days 5 --ppo_min_rollout_transitions 180 --ppo_batch_size 64 --ppo_update_epochs 8 \
  --training_stage_market_fraction 0.05 --training_stage_static_fraction 0.05 --training_stage_dynamic_fraction 0.90 \
  --training_opening_tariff_randomization 0.15 \
  --checkpoint_validation_interval_days $VAL_INT --checkpoint_validation_horizon $VAL_HOR --checkpoint_validation_customers $CUST --checkpoint_validation_openings 2 \
  --eval_timesteps $EVAL --eval_customers $CUST --eval_policy_mode argmax_raw --benchmark_suite \
  --static_baseline_multipliers "0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.8,2.0" \
  --regime_market --shocks calibrated --shock_intensity 1.0 --regime_features \
  --tariff_targets "0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.8,2.0" \
  --spillover_price_aware --rl_warm_start_target $WARM $CHANNEL_FLAGS \
  --report_prefix "$OUT/$CITY" --trained_model_out "$OUT/policy.pt" --trained_model_id "v26r-$CITY-seed$SEED-$MODE" \
  --model_registry_dir "$OUT/registry" 2>&1 | tee "$OUT/run.log"
echo "exit=${PIPESTATUS[0]}" | tee "$OUT/exit_status.txt"
