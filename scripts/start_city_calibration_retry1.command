#!/bin/zsh
# User-approved retry; original failed artifacts remain intact.
cd /Users/huali/Desktop/RideResponse/rideshare_v26r_revision5 || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
/Users/huali/Desktop/RideResponse/.venv/bin/python -u /Users/huali/Desktop/RideResponse/rideshare_v26r_revision5/scripts/audit_city_calibration.py --cities nyc boston chicago --output /Users/huali/Desktop/RideResponse/artifacts/revision5_calibration_20261017_retry1 > /Users/huali/Desktop/RideResponse/artifacts/revision5_calibration_20261017_retry1.log 2>&1
audit_exit=$?
printf '%s\n' "$audit_exit" > /Users/huali/Desktop/RideResponse/artifacts/revision5_calibration_20261017_retry1.exit_status.txt
exit "$audit_exit"
