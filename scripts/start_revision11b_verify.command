#!/bin/zsh
cd /Users/huali/Desktop/RideResponse/rideshare_v26r_revision11 || exit 1
# Stop the superseded revision-11 (no branch credit) runs only.
pkill -f "run_revision11_verify.py" ; pkill -f "revision11_verify_8000" ; sleep 2
nohup /Users/huali/Desktop/RideResponse/.venv/bin/python -u scripts/run_revision11b_verify.py 3000 > runs/revision11b_verify_3000.launcher.log 2>&1 &
echo "Revision 11b verification started (PID $!). Log: runs/revision11b_verify_3000.launcher.log"
