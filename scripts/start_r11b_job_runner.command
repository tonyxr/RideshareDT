#!/bin/zsh
cd /Users/huali/Desktop/RideResponse/rideshare_v26r_revision11 || exit 1
rm -f runs/jobq/STOP
nohup zsh scripts/r11b_job_loop.zsh > runs/jobq/runner.out 2>&1 &
echo "Revision 11b job runner started (PID $!). You can close this window."
