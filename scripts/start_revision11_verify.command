#!/bin/zsh
cd /Users/huali/Desktop/RideResponse/rideshare_v26r_revision11 || exit 1
nohup /Users/huali/Desktop/RideResponse/.venv/bin/python -u scripts/run_revision11_verify.py 8000 > runs/revision11_verify_8000.launcher.log 2>&1 &
echo "Revision 11 verification started (PID $!). Log: runs/revision11_verify_8000.launcher.log"
echo "You can close this window."
