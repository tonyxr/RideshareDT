#!/usr/bin/env bash
set -euo pipefail

cd /Users/huali/Desktop/RideResponse
SEED="${V30_SEED:-300071}"
OUT="${V30_OUT:-artifacts/v30/chicago_full_seed${SEED}}"
mkdir -p "$OUT"

# Full staged PPO job, intended for an independent Terminal process.
# Performance diagnostics never gate training or replace the final actor.
exec python -u Core.py \
  --run_experiment \
  --market Chicago \
  --firm1_mode RL \
  --firm2_mode adaptive_best_response_aggressive \
  --v30 \
  --terminal_profit_dominance_weight 1.0 \
  --manipulated_price_coefficients 5 \
  --training_curriculum environment_then_opponent \
  --training_stage_market_fraction 0.20 \
  --training_stage_static_fraction 0.20 \
  --training_stage_dynamic_fraction 0.60 \
  --training_dynamic_episode_days 120 \
  --state_frame_stack 4 \
  --no-persistent_recurrence \
  --calibration_json data/processed/chicago/calibration.json \
  --trip_bank_path data/processed/chicago/trip_bank.csv \
  --threshold_profile_source cached \
  --threshold_cache_path artifacts/v28_thresholds/chicago/native_uncapped_primary.jsonl \
  --shocks calibrated \
  --train_timesteps 6000 \
  --train_steps_per_day 10 \
  --train_customers 2000 \
  --firm1_action_interval_steps 10 \
  --firm2_action_interval_days 10 \
  --ppo_update_interval_days 5 \
  --ppo_min_rollout_transitions 24 \
  --ppo_batch_size 24 \
  --ppo_update_epochs 4 \
  --eval_timesteps 1000 \
  --eval_customers 1000 \
  --checkpoint_validation_horizon 1000 \
  --checkpoint_validation_interval_days 300 \
  --checkpoint_validation_customers 1000 \
  --checkpoint_validation_openings 5 \
  --benchmark_suite \
  --baseline_suite "static_grid,myopic_greedy,anchor_hold" \
  --static_baseline_multipliers "1.0,1.1,1.2,1.3,1.4,1.5,1.75,2.0,3.0,4.0" \
  --ood_perturbed_variants 3 \
  --unrestricted_policy_experiment \
  --eval_policy_mode sample_raw \
  --deterministic_experiment_seed \
  --experiment_seed "$SEED" \
  --report_prefix "$OUT/chicago" \
  --trained_model_out "$OUT/policy.pt" \
  --trained_model_id "chicago-v30-ppo-seed${SEED}" \
  --model_registry_dir "" \
  --profiles_out "$OUT/profiles.csv" \
  --profiles_log_limit 200000 \
  "$@"
