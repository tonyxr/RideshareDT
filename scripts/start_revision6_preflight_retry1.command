#!/bin/zsh
set -euC
cd /Users/huali/Desktop/RideResponse/rideshare_v26r_revision6
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
/Users/huali/Desktop/RideResponse/.venv/bin/python -u scripts/run_revision6_preflight.py --output /Users/huali/Desktop/RideResponse/artifacts/revision6_preflight_20261018_retry1 > /Users/huali/Desktop/RideResponse/artifacts/revision6_preflight_20261018_retry1.log 2>&1
