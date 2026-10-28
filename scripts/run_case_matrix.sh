#!/usr/bin/env bash
# Run the whole corner-case matrix for one city (Mac: real torch optional; VM: numpy stub, call repeatedly).
# Usage: scripts/run_case_matrix.sh <City> <slug> [openings=10] [policy.pt] [channels=full]
#   Validation tapes (selection) and evaluation tapes (reporting) are both run; pass a policy to add rl_actor.
set -euo pipefail
CITY=${1:?City}; SLUG=${2:?slug}; OPEN=${3:-10}; POLICY=${4:-}; CH=${5:-full}
cd "$(dirname "$0")/.."
EXTRA=(); [[ -n "$POLICY" ]] && EXTRA=(--policy "$POLICY")
python3 scripts/v26r_case_eval.py "$CITY" "$SLUG" all --seed-set both --openings "$OPEN" --budget-seconds 1000000 --chunk 4 --channels "$CH" "${EXTRA[@]}"
python3 scripts/v26r_case_matrix.py "$SLUG"
echo "matrix: runs/cases/$SLUG/matrix/matrix.csv"
