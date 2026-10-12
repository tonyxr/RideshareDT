#!/usr/bin/env bash
set -euo pipefail

cd /Users/huali/Desktop/RideResponse
OUT="artifacts/v29_ppo/chicago_readiness_seed140041"
mkdir -p "$OUT"

# Diagnostic rehearsal of the unrestricted original PPO stack. Economic
# audits are reported but never alter, roll back, quarantine, or stop PPO.
exec python -u Core.py \
  --run_experiment \
  --market Chicago \
  --firm1_mode RL \
  --firm2_mode adaptive_best_response_aggressive \
  --v29 \
  --manipulated_price_coefficients 5 \
  --policy_objective profit_with_dominance \
  --training_curriculum environment_then_opponent \
  --training_stage_market_fraction 0.20 \
  --training_stage_static_fraction 0.20 \
  --training_stage_dynamic_fraction 0.60 \
  --training_dynamic_episode_days 120 \
  --independent_coefficient_actions \
  --compact_foresight_observation \
  --state_frame_stack 4 \
  --no-persistent_recurrence \
  --v28 \
  --rider_memory \
  --minimum_tariff_multiplier 1.0 \
  --calibration_json data/processed/chicago/calibration.json \
  --trip_bank_path data/processed/chicago/trip_bank.csv \
  --threshold_profile_source cached \
  --threshold_cache_path artifacts/v28_thresholds/chicago/native_uncapped_primary.jsonl \
  --shocks calibrated \
  --train_timesteps 1200 \
  --train_steps_per_day 10 \
  --train_customers 1000 \
  --firm1_action_interval_steps 10 \
  --firm2_action_interval_days 10 \
  --ppo_update_interval_days 5 \
  --ppo_min_rollout_transitions 24 \
  --ppo_batch_size 24 \
  --ppo_update_epochs 4 \
  --eval_timesteps 1000 \
  --eval_customers 1000 \
  --checkpoint_validation_horizon 1000 \
  --checkpoint_validation_interval_days 240 \
  --checkpoint_validation_customers 1000 \
  --checkpoint_validation_openings 3 \
  --benchmark_suite \
  --baseline_suite "static_grid" \
  --static_baseline_multipliers "1.0,1.1,1.2,1.3,1.4,1.5,1.75,2.0,3.0,4.0" \
  --ood_perturbed_variants 0 \
  --unrestricted_policy_experiment \
  --defer_final_test \
  --deterministic_experiment_seed \
  --experiment_seed 140041 \
  --report_prefix "$OUT/chicago" \
  --trained_model_out "$OUT/policy.pt" \
  --model_registry_dir "" \
  --profiles_out "$OUT/profiles.csv" \
  --profiles_log_limit 100000
