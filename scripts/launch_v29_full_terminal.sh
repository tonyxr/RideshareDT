#!/usr/bin/env bash
set -uo pipefail

cd /Users/huali/Desktop/RideResponse
OUT="artifacts/v29_ppo/chicago_full_seed150053"
mkdir -p "$OUT"
export MPLCONFIGDIR="/tmp/v29-full-mpl"

printf '%s\n' "$$" > "$OUT/terminal_launcher.pid"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$OUT/started_utc.txt"

bash scripts/run_v29_ppo_full_chicago.sh > "$OUT/run.log" 2>&1
status=$?
printf '%s\n' "$status" > "$OUT/exit_status.txt"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "$OUT/finished_utc.txt"

# Reporting is observational and never changes the trained policy or status.
python scripts/analyze_v29_ppo_diagnostics.py "$OUT" >> "$OUT/run.log" 2>&1 || true
exit "$status"
