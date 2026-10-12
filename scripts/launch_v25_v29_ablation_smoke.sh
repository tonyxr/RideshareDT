#!/usr/bin/env bash
set -u

cd /Users/huali/Desktop/RideResponse || exit 2
OUT="artifacts/v25_v29_ablation/smoke"
mkdir -p "$OUT"
date -u +%FT%TZ > "$OUT/started_at.txt"
echo "$$" > "$OUT/launcher_pid.txt"

python -u scripts/v25_v29_ablation_ladder.py \
  --budget smoke \
  --seeds 17101 \
  --resume \
  > "$OUT/ladder.log" 2>&1
status=$?

if [[ "$status" -eq 0 ]]; then
  python scripts/analyze_v25_v29_ablation.py "$OUT" \
    >> "$OUT/ladder.log" 2>&1
  status=$?
fi

echo "$status" > "$OUT/exit_status.txt"
date -u +%FT%TZ > "$OUT/finished_at.txt"
exit "$status"
