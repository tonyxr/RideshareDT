#!/bin/zsh
set -u

ROOT=/Users/huali/Desktop/RideResponse
CODE=$ROOT/rideshare_v26r_revision9
PYTHON=$ROOT/.venv/bin/python
PREFLIGHT=$ROOT/artifacts/revision9_smoke_preflight_20261019_retry1
RUN=$ROOT/artifacts/revision9_chicago_smoke_2400_20261019_retry1

mkdir -p "$PREFLIGHT/fixtures" "$PREFLIGHT/identity/boston" \
  "$PREFLIGHT/identity/expanded" "$RUN"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
export V26R_IDENTITY_OUTPUT="$PREFLIGHT/identity/boston"
export V26R_EXPANDED_IDENTITY_OUTPUT="$PREFLIGHT/identity/expanded"

cd "$CODE"
print 'preflight: running maintained regression, integration, and flags-off identity suite' \
  > "$PREFLIGHT/status.txt"
"$PYTHON" -u -c 'import sys; sys.path.append("/Users/huali/opt/anaconda3/lib/python3.9/site-packages"); import pytest; raise SystemExit(pytest.main(["-q", "tests", "--basetemp=/Users/huali/Desktop/RideResponse/artifacts/revision9_smoke_preflight_20261019_retry1/fixtures", "--junitxml=/Users/huali/Desktop/RideResponse/artifacts/revision9_smoke_preflight_20261019_retry1/tests.xml"]))' \
  > "$PREFLIGHT/tests.log" 2>&1
test_status=$?
print "$test_status" > "$PREFLIGHT/test_exit_status.txt"
if (( test_status != 0 )); then
  print 'failed: preflight tests; training not started' > "$PREFLIGHT/status.txt"
  exit "$test_status"
fi

print 'passed: all preflight tests and flags-off identities; launching smoke training' \
  > "$PREFLIGHT/status.txt"
"$PYTHON" -u "$CODE/scripts/run_v26r_multicity.py" \
  --output "$RUN" \
  --city chicago \
  --seed 261901 \
  --days 2400 \
  --update-days 5 \
  --rollout-transitions 180 \
  --lead-weight 0.6 \
  --learner-revision 4 \
  --ppo-rollout-episodes 2 \
  --city-calibration-root "$ROOT/artifacts/revision7_direct_calibration_20261018" \
  --training-scenario-root "$ROOT/rideshare_v26r_revision8/config/revision8_curriculum" \
  --figures \
  --skip-corners \
  > "$RUN/terminal.log" 2>&1
run_status=$?
print "$run_status" > "$RUN/terminal_exit_status.txt"
if (( run_status != 0 )); then
  print 'failed: smoke training or evaluation; artifacts retained, no automatic retry' \
    > "$RUN/terminal_status.txt"
  exit "$run_status"
fi
print 'complete: smoke training and standard benchmark evaluation' \
  > "$RUN/terminal_status.txt"
