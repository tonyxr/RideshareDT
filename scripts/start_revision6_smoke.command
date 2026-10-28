#!/bin/zsh
set -eu
cd /Users/huali/Desktop/RideResponse/rideshare_v26r_revision6
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
export MPLCONFIGDIR=/Users/huali/Desktop/RideResponse/rideshare_v26r_revision6/runs/tests/mplconfig
/Users/huali/Desktop/RideResponse/.venv/bin/python -u scripts/run_revision6_smoke.py --output /Users/huali/Desktop/RideResponse/artifacts/revision6_smoke_20261018 > /Users/huali/Desktop/RideResponse/artifacts/revision6_smoke_20261018.log 2>&1
