# baseline_v25 — frozen main model for the paper (restored 2026-10-09)

Code: exact copy of _backup_pre_v26_20261005 (SHA-256 matches
artifacts/three_city_full_20260825/experiment_manifest.json for all 10 tracked files).
Calibration: v25 calibration.json per city (from data/processed/<city>/v25_backup).
Thresholds: read from artifacts/three_city_full_20260825/raw/thresholds/<city>/primary.jsonl.

Run (on the Mac, rideshareDT conda env):
  bash scripts/run_v25_smoke.sh                       # few minutes, Boston
  bash scripts/run_v25_city.sh chicago profit_maximization 4500   # full v25 recipe

Outputs go to baseline_v25/runs/. Do not edit code here; extensions are added
one at a time behind default-off flags and must pass the v25 regression gate
(see the project doc "v25_baseline_and_contributions_2026-10-09").
