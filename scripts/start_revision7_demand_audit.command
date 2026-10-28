#!/bin/zsh
set -eu
cd /Users/huali/Desktop/RideResponse/rideshare_v26r_revision7
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
set -o noclobber
/Users/huali/Desktop/RideResponse/.venv/bin/python -u scripts/verify_joint_demand.py --output /Users/huali/Desktop/RideResponse/artifacts/revision7_demand_20261018 > /Users/huali/Desktop/RideResponse/artifacts/revision7_demand_20261018.log 2>&1
