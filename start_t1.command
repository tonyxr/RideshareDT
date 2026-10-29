#!/bin/zsh
set -euo pipefail

cd "${0:A:h}"
PYTHON="../.venv/bin/python"
OUTPUT="runs/v17_t1"

if command -v caffeinate >/dev/null 2>&1; then
  exec caffeinate -dimsu "$PYTHON" -u scripts/v17_market_checks.py --output "$OUTPUT"
fi
exec "$PYTHON" -u scripts/v17_market_checks.py --output "$OUTPUT"
