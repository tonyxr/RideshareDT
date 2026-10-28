#!/bin/zsh
# Runs job scripts dropped into runs/jobq/*.sh (each in the background), then files them under done/.
# Create runs/jobq/STOP to end the loop.
B=/Users/huali/Desktop/RideResponse/rideshare_v26r_revision11
Q=$B/runs/jobq
mkdir -p $Q/running $Q/done
echo "$(date) runner started pid $$" >> $Q/runner.log
while [ ! -f $Q/STOP ]; do
  date +%s > $Q/heartbeat
  for f in $Q/*.sh(N); do
    b=${f:t}
    mv $f $Q/running/$b
    echo "$(date) start $b" >> $Q/runner.log
    ( cd $B && zsh $Q/running/$b > $Q/running/$b.log 2>&1; echo "exit=$?" >> $Q/running/$b.log; mv $Q/running/$b $Q/running/$b.log $Q/done/; echo "$(date) done $b" >> $Q/runner.log ) &
  done
  sleep 5
done
echo "$(date) runner stopped" >> $Q/runner.log
