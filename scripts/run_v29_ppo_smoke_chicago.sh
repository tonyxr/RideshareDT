#!/usr/bin/env bash
set -euo pipefail

cd /Users/huali/Desktop/RideResponse
OUT="artifacts/v29_ppo/chicago_smoke_seed160067"
mkdir -p "$OUT"

# Short diagnostic of the unrestricted original-PPO learner.  This run keeps
# the final actor exactly as trained; economic checks only report diagnostics.
exec python -u Core.py \
  --run_experiment \
  --market Chicago \
  --firm1_mode RL \
  --firm2_mode adaptive_best_response_aggressive \
  --manipulated_price_coefficients 5 \
  --policy_objective profit_with_dominance \
  --training_curriculum environment_then_opponent \
  --training_stage_market_fraction 0.25 \
  --training_stage_static_fraction 0.25 \
  --training_stage_dynamic_fraction 0.50 \
  --training_dynamic_episode_days 120 \
  --independent_coefficient_actions \
  --compact_foresight_observation \
  --state_frame_stack 4 \
  --no-persistent_recurrence \
  --v28 \
  --calibration_json data/processed/chicago/calibration.json \
  --trip_bank_path data/processed/chicago/trip_bank.csv \
  --threshold_profile_source cached \
  --threshold_cache_path artifacts/v28_thresholds/chicago/native_uncapped_primary.jsonl \
  --shocks calibrated \
  --train_timesteps 240 \
  --train_steps_per_day 10 \
  --train_customers 250 \
  --firm1_action_interval_steps 10 \
  --firm2_action_interval_days 10 \
  --ppo_update_interval_days 5 \
  --ppo_min_rollout_transitions 24 \
  --ppo_batch_size 24 \
  --ppo_update_epochs 4 \
  --eval_timesteps 200 \
  --eval_customers 250 \
  --checkpoint_validation_horizon 200 \
  --checkpoint_validation_interval_days 120 \
  --checkpoint_validation_customers 250 \
  --checkpoint_validation_openings 2 \
  --baseline_suite "" \
  --ood_perturbed_variants 0 \
  --unrestricted_policy_experiment \
  --defer_final_test \
  --deterministic_experiment_seed \
  --experiment_seed 160067 \
  --report_prefix "$OUT/chicago" \
  --trained_model_out "$OUT/policy.pt" \
  --model_registry_dir "" \
  --profiles_out "$OUT/profiles.csv" \
  --profiles_log_limit 25000
