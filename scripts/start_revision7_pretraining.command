#!/bin/zsh
set -eu

BASE=/Users/huali/Desktop/RideResponse/rideshare_v26r_revision7
PYTHON=/Users/huali/Desktop/RideResponse/.venv/bin/python
OUT=/Users/huali/Desktop/RideResponse/artifacts/revision7_pretraining_20261018
DEMAND=/Users/huali/Desktop/RideResponse/artifacts/revision7_demand_20261018/audit

cd "$BASE"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLBACKEND=Agg
export V26R_IDENTITY_OUTPUT="$OUT/identity/boston"
export V26R_EXPANDED_IDENTITY_OUTPUT="$OUT/identity/expanded"

mkdir "$OUT"
print 'testing: complete maintained regression, integration and flags-off identity suite' > "$OUT/status.txt"
set +e
"$PYTHON" -u -c 'import sys; sys.path.append("/Users/huali/opt/anaconda3/lib/python3.9/site-packages"); import pytest; raise SystemExit(pytest.main(["-q","tests","--basetemp=/Users/huali/Desktop/RideResponse/artifacts/revision7_pretraining_20261018/fixtures","--junitxml=/Users/huali/Desktop/RideResponse/artifacts/revision7_pretraining_20261018/tests.xml"]))' > "$OUT/tests.log" 2>&1
TEST_CODE=$?
set -e
print "$TEST_CODE" > "$OUT/test_exit_status.txt"
if [[ "$TEST_CODE" != 0 ]]; then
  print 'failed: tests did not pass; simulator audit not started' > "$OUT/status.txt"
  print 1 > "$OUT/exit_status.txt"
  exit 1
fi

"$PYTHON" -u -c 'from pathlib import Path; from scripts.run_revision6_preflight import require_passing_report; import json; p=Path("/Users/huali/Desktop/RideResponse/artifacts/revision7_pretraining_20261018/tests.xml"); Path("/Users/huali/Desktop/RideResponse/artifacts/revision7_pretraining_20261018/test_result.json").write_text(json.dumps(require_passing_report(p), indent=2)+"\n")'
print 'tests passed; running evaluation-only supply, warm-start, Gate 0/1 and rival impulse audit' > "$OUT/status.txt"
set +e
"$PYTHON" -u scripts/audit_revision7_pretraining.py \
  --demand-root "$DEMAND" \
  --output "$OUT/audit" \
  --cities nyc boston chicago \
  --openings 10 \
  --sensitivity-openings 3 \
  --horizon-days 40 > "$OUT/audit.log" 2>&1
AUDIT_CODE=$?
set -e
print "$AUDIT_CODE" > "$OUT/audit_exit_status.txt"
if [[ "$AUDIT_CODE" != 0 ]]; then
  print 'failed: simulator audit evidence preserved; no retry or training' > "$OUT/status.txt"
  print 1 > "$OUT/exit_status.txt"
  exit 1
fi

print 'completed: tests and evaluation-only pre-training audit; no RL training' > "$OUT/status.txt"
print 0 > "$OUT/exit_status.txt"
