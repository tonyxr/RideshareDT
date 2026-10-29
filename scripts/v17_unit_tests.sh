#!/usr/bin/env bash
# CRISP-Ride V17 execution-manual T0 suite. Runs in registered order and
# stops at the first failed gate so later tests cannot hide the root failure.
set -euo pipefail

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-$BASE/../.venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg

run_gate() {
  local gate="$1"
  shift
  echo "[$gate] $*"
  "$PYTHON_BIN" -m pytest -q "$@"
}

cd "$BASE"

run_gate T0.1 tests/test_v17_flags_off_identity.py
run_gate T0.2 tests/test_v17_realistic_fare.py
run_gate T0.3 tests/test_v17_supply_v3.py
run_gate T0.4 tests/test_v17_shortage_willingness.py
run_gate T0.5 \
  tests/test_joint_demand.py::test_relative_units_refit_same_elasticity_target_per_city \
  tests/test_joint_demand_integration.py::test_relative_units_reject_legacy_beta_and_accept_matching_refit
run_gate T0.6 tests/test_v17_two_lever_action.py
run_gate T0.7 tests/test_v17_competitor_mapping.py
run_gate T0.8 tests/test_v17_exogenous_tape.py
run_gate T0.9 tests/test_v17_one_day_deviation.py
run_gate T0.10 \
  tests/test_v17_value_model.py::test_t0_10_recovers_two_regime_argmax_from_300_noisy_rows
run_gate T0.11 \
  tests/test_v17_value_fit.py::test_value_fit_resume_reproduces_exact_next_update
run_gate T0.12 \
  tests/test_v17_value_fit.py::test_t0_12_replay_labels_use_the_current_stage_objective
run_gate T0.13 tests/test_v17_eval_protocol.py
run_gate T0.14 \
  tests/test_v17_two_lever_action.py::test_t0_14_v2_grid_has_36_cells_and_v1_mapping_is_unchanged \
  tests/test_v17_two_lever_action.py::test_t0_14_v2_core_value_surface_and_artifact_are_36_cells \
  tests/test_v17_config.py::test_t0_14_config_switch_accepts_only_the_registered_six_level_grid

# User-approved V17 extensions that must remain wired into the same release.
run_gate V17.extensions tests/test_v17_progressive_competition.py
run_gate V17.config tests/test_v17_config.py

echo "V17 T0 PASS"
