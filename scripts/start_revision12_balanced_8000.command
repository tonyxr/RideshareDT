#!/bin/zsh
cd /Users/huali/Desktop/RideResponse/rideshare_v26r_revision12 || exit 1
run_dir="runs/revision12_long_oracle_balanced_8000_$(date +%Y%m%d_%H%M%S)"
launcher_log="${run_dir}.launcher.log"
nohup /Users/huali/Desktop/RideResponse/.venv/bin/python -u \
  scripts/run_revision12_long_only.py 8000 "$run_dir" \
  > "$launcher_log" 2>&1 &
echo "Revision 12 balanced 8,000-day run started (PID $!)."
echo "Status: ${run_dir}/status.json"
echo "Log: ${run_dir}/training.log"
