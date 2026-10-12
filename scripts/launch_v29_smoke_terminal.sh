#!/usr/bin/env bash
set -u

cd /Users/huali/Desktop/RideResponse
OUT="artifacts/v29_ppo/chicago_smoke_seed160067"
mkdir -p "$OUT"
date -u +%FT%TZ > "$OUT/started_at.txt"
echo "$$" > "$OUT/launcher_pid.txt"

bash scripts/run_v29_ppo_smoke_chicago.sh > "$OUT/run.log" 2>&1
status=$?
echo "$status" > "$OUT/exit_status.txt"
date -u +%FT%TZ > "$OUT/finished_at.txt"

if [[ -f "$OUT/chicago_training_metrics.csv" ]]; then
  python scripts/analyze_v29_ppo_diagnostics.py \
    --run-dir "$OUT" \
    --output-prefix "$OUT/model_diagnostic_report" >> "$OUT/run.log" 2>&1 || true
fi

exit "$status"
