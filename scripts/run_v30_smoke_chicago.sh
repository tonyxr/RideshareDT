#!/usr/bin/env bash
set -euo pipefail

cd /Users/huali/Desktop/RideResponse
SEED="${V30_SEED:-300031}"
OUT="${V30_OUT:-artifacts/v30/chicago_smoke_seed${SEED}}"
mkdir -p "$OUT"

# Small end-to-end contract test. It is diagnostic and leaves the final test
# partition sealed; no gate, fallback or CPI controller alters PPO training.
exec python -u Core.py \
  --run_experiment \
  --market Chicago \
  --firm1_mode RL \
  --firm2_mode adaptive_best_response_aggressive \
  --v30 \
  --terminal_profit_dominance_weight 1.0 \
  --manipulated_price_coefficients 5 \
  --training_curriculum environment_then_opponent \
  --training_stage_market_fraction 0.50 \
  --training_stage_static_fraction 0.25 \
  --training_stage_dynamic_fraction 0.25 \
  --training_dynamic_episode_days 15 \
  --state_frame_stack 4 \
  --no-persistent_recurrence \
  --calibration_json data/processed/chicago/calibration.json \
  --trip_bank_path data/processed/chicago/trip_bank.csv \
  --threshold_profile_source cached \
  --threshold_cache_path artifacts/v28_thresholds/chicago/native_uncapped_primary.jsonl \
  --shocks calibrated \
  --train_timesteps 60 \
  --train_steps_per_day 4 \
  --train_customers 200 \
  --firm1_action_interval_steps 4 \
  --firm2_action_interval_days 4 \
  --ppo_update_interval_days 4 \
  --ppo_min_rollout_transitions 8 \
  --ppo_batch_size 8 \
  --ppo_update_epochs 2 \
  --eval_timesteps 40 \
  --eval_customers 200 \
  --checkpoint_validation_horizon 40 \
  --checkpoint_validation_interval_days 20 \
  --checkpoint_validation_customers 200 \
  --checkpoint_validation_openings 1 \
  --baseline_suite "" \
  --ood_perturbed_variants 0 \
  --unrestricted_policy_experiment \
  --defer_final_test \
  --deterministic_experiment_seed \
  --experiment_seed "$SEED" \
  --report_prefix "$OUT/chicago" \
  --trained_model_out "$OUT/policy.pt" \
  --model_registry_dir "" \
  --profiles_out "$OUT/profiles.csv" \
  --profiles_log_limit 5000 \
  "$@"
