#!/bin/zsh
cd /Users/huali/Desktop/RideResponse/rideshare_v26r_revision11 || exit 1
run_dir="runs/revision11_long_oracle_balanced_8000_20261020"
launcher_log="${run_dir}.launcher.log"
nohup /Users/huali/Desktop/RideResponse/.venv/bin/python -u \
  scripts/run_revision11_long_only.py 8000 "$run_dir" \
  > "$launcher_log" 2>&1 &
echo "Revision 11 balanced 8,000-day run started (PID $!)."
echo "Status: ${run_dir}/status.json"
echo "Log: ${run_dir}/training.log"
