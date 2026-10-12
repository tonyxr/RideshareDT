#!/usr/bin/env bash
# One-shot verification of the v26 changes on a machine with the project's
# Python environment (PyTorch, pyarrow).  Writes everything to
# artifacts/v26_verify/ so the logs can be inspected afterwards.
#
#   cd ~/Desktop/RideResponse && bash scripts/v26_verify.sh
#
# Steps: (1) full unit-test suite, (2) v25 calibration backup + v26 city
# preprocessing with trip banks, (3) a short end-to-end v26 RL smoke run with
# held-out benchmarks, OOD rivals, baselines, and the trajectory observed-fare
# diagnostic, (4) simulator diagnostics per city (static landscape,
# elasticities, promotion impulse responses by channel).  Nothing here calls the OpenAI API.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
OUT=artifacts/v26_verify
mkdir -p "$OUT"
export MPLCONFIGDIR="$OUT/matplotlib"
mkdir -p "$MPLCONFIGDIR"
# Verification is offline by contract.  Do not let an ambient credential turn
# unit tests with generated threshold profiles into external API calls.
unset OPENAI_API_KEY
if [ ! -x "$PY" ]; then
  echo "Python interpreter is not executable: $PY" >&2
  exit 2
fi
echo "python: $($PY -c 'import sys, numpy, pandas, torch, pyarrow, matplotlib, kagglehub; print(sys.version.split()[0], torch.__version__)')" | tee "$OUT/env.log"

echo "[1/4] unit tests" | tee -a "$OUT/env.log"
$PY -m unittest discover -v > "$OUT/unittest.log" 2>&1
echo "unittest_exit=0" | tee -a "$OUT/unittest.log" "$OUT/env.log"

echo "[2/4] v26 preprocessing (keeps a copy of the v25 calibrations)" | tee -a "$OUT/env.log"
for city in nyc boston chicago; do
  mkdir -p "data/processed/$city/v25_backup"
  # Only a schema-1 (v25) calibration is backed up; a file already rewritten
  # by the v26 prepare step must never be mistaken for the v25 original.
  if [ ! -f "data/processed/$city/v25_backup/calibration.json" ] && \
     $PY -c "import json,sys; sys.exit(0 if json.load(open('data/processed/$city/calibration.json')).get('schema_version', 1) == 1 else 1)"; then
    cp "data/processed/$city/calibration.json" "data/processed/$city/v25_backup/calibration.json"
  fi
done
$PY scripts/prepare_city_data.py --city all --sample-rows 5000 \
    --calibration-ride-rows 20000 --trip-bank-rows 50000 > "$OUT/prepare.log" 2>&1
echo "prepare_exit=0" | tee -a "$OUT/prepare.log" "$OUT/env.log"

echo "[3/4] v26 RL smoke (Boston)" | tee -a "$OUT/env.log"
CACHE=artifacts/three_city_full_20260825/raw/thresholds/boston/primary.jsonl
$PY Core.py --market Boston --choice_mode cognitive --firm1_mode RL \
  --firm2_mode adaptive_best_response_aggressive --run_experiment \
  --credit_assignment_v25 --v26 --deterministic_experiment_seed \
  --seed 2601 --experiment_seed 2601 --policy_objective profit_maximization \
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
  --trained_model_id v26-verify-boston --model_registry_dir "$OUT/registry" \
  --compare_with_dataset --dataset_root data/raw/rides/boston_kaggle \
  --dataset_city Boston --weather_root data/raw/weather/noaa \
  --comparison_out "$OUT/smoke_observed_price_comparison.csv" \
  --comparison_plot_prefix "$OUT/smoke_observed_price" \
  --comparison_limit 50 --dataset_preview_rows 0 > "$OUT/smoke.log" 2>&1
echo "smoke_exit=0" | tee -a "$OUT/smoke.log" "$OUT/env.log"
echo "[4/4] simulator diagnostics (landscape, elasticities, promotion impulse)" | tee -a "$OUT/env.log"
for city in nyc boston chicago; do
  $PY scripts/v26_market_diagnostics.py --city "$city" --seeds 2 --customers 300 \
      --out "$OUT/diagnostics_$city.json" > "$OUT/diagnostics_$city.log" 2>&1
  echo "diagnostics_${city}_exit=0" | tee -a "$OUT/env.log"
done
echo "done; see $OUT/" | tee -a "$OUT/env.log"
