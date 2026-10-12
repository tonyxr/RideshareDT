#!/usr/bin/env bash
# One-shot verification of the v27 market engine and shocks on a machine with
# the project's Python environment.  Run scripts/v26_verify.sh first (it builds
# the v26 city calibrations and trip banks this script uses).  Writes
# everything to artifacts/v27_verify/.
#
#   cd ~/Desktop/RideResponse && bash scripts/v27_verify.sh
#
# Steps: (1) v27 unit tests, (2) a short end-to-end v27 RL smoke (Boston,
# calibrated shocks, public shock features), (3) shock diagnostics per city
# (value of adapting to each shock type vs the best static tariff, with a
# placebo).  Nothing here calls the OpenAI API.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
OUT=artifacts/v27_verify
mkdir -p "$OUT"
export MPLCONFIGDIR="$OUT/matplotlib"
mkdir -p "$MPLCONFIGDIR"
unset OPENAI_API_KEY
if [ ! -x "$PY" ]; then
  echo "Python interpreter is not executable: $PY" >&2
  exit 2
fi
echo "python: $($PY -c 'import sys, numpy, torch; print(sys.version.split()[0], numpy.__version__, torch.__version__)')" | tee "$OUT/env.log"

echo "[1/3] v27 unit tests" | tee -a "$OUT/env.log"
$PY -m unittest -v test_v27_market_engine test_v27_core_integration > "$OUT/unittest.log" 2>&1
echo "unittest_exit=0" | tee -a "$OUT/unittest.log" "$OUT/env.log"

echo "[2/3] v27 RL smoke (Boston, calibrated shocks)" | tee -a "$OUT/env.log"
CACHE=artifacts/three_city_full_20260825/raw/thresholds/boston/primary.jsonl
$PY Core.py --market Boston --choice_mode cognitive --firm1_mode RL \
  --firm2_mode adaptive_best_response_aggressive --run_experiment \
  --credit_assignment_v25 --v27 --shocks calibrated --deterministic_experiment_seed \
  --seed 2701 --experiment_seed 2701 --policy_objective profit_maximization \
  --pool 400 --threshold_profile_source cached --threshold_cache_path "$CACHE" \
  --calibration_json data/processed/boston/calibration.json \
  --train_timesteps 40 --train_customers 30 --train_steps_per_day 6 \
  --firm1_action_interval_steps 1 --firm2_action_interval_steps 1 \
  --ppo_update_interval_days 5 --ppo_min_rollout_transitions 32 \
  --ppo_batch_size 16 --ppo_update_epochs 2 \
  --training_stage_market_fraction 0.10 --training_stage_static_fraction 0.10 \
  --training_stage_dynamic_fraction 0.80 --training_dynamic_episode_days 4 \
  --checkpoint_validation_interval_days 20 --checkpoint_validation_horizon 30 \
  --checkpoint_validation_customers 30 --checkpoint_validation_openings 1 \
  --eval_timesteps 30 --eval_customers 30 --benchmark_suite \
  --static_baseline_multipliers 0.8,1.0,1.2 --ood_perturbed_variants 1 \
  --report_prefix "$OUT/smoke" --trained_model_out "$OUT/smoke_policy.pt" \
  --trained_model_id v27-verify-boston --model_registry_dir "$OUT/registry" \
  > "$OUT/smoke.log" 2>&1
echo "smoke_exit=0" | tee -a "$OUT/smoke.log" "$OUT/env.log"

echo "[3/3] shock diagnostics (static vs switching vs oracle schedule, placebo)" | tee -a "$OUT/env.log"
for city in boston nyc chicago; do
  if [ ! -f "data/processed/$city/calibration.json" ]; then
    echo "skip $city: no calibration (run scripts/v26_verify.sh)" | tee -a "$OUT/env.log"
    continue
  fi
  $PY scripts/v27_shock_diagnostics.py --city "$city" \
      --out "$OUT/shock_diagnostics_$city.json" > "$OUT/shock_diagnostics_$city.log" 2>&1
  echo "diagnostics_${city}_exit=0" | tee -a "$OUT/env.log"
done
echo "done; see $OUT/" | tee -a "$OUT/env.log"
