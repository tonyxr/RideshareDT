#!/usr/bin/env python3
"""Registered CRISP-Ride V17 evaluation protocol and preflight entry point.

This first M7 work package records the immutable Phase 4 contract.  It does
not launch evaluation.  Execution backends are added only after the V17
market and learner interfaces exist and pass T0--T4.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from v17_objective import COMPETITOR_PROFIT_WEIGHT, OBJECTIVE_NAME


PROTOCOL_VERSION = "v1"
CASES = (
    "C0_calm",
    "A1_shortage_severe",
    "A2_shortage_moderate_null",
    "A3_storm_transit",
    "S1_promotion_bait",
    "S6_shortage_during_promotion",
)
TRAINING_EXPOSED_COMPETITORS = (
    "adaptive_best_response_memory",
    "profit_best_response_myopic",
    "profit_best_response_forward",
    "profit_best_response_anticipating",
    "profit_opt_mpc_grid",
)
HELD_OUT_COMPETITORS = (
    "pi_price_gap",
    "region_supply_demand",
    "queue_service_threshold",
)
PERTURBED_HELD_OUT_COMPETITORS = (
    "pi_price_gap@perturbed0",
    "queue_service_threshold@perturbed0",
)
COMPETITORS = (
    *TRAINING_EXPOSED_COMPETITORS,
    *HELD_OUT_COMPETITORS,
    *PERTURBED_HELD_OUT_COMPETITORS,
)
POLICY_CLASSES = (
    "rl_seed_101",
    "rl_seed_202",
    "rl_seed_303",
    "frozen_seed_101",
    "frozen_seed_202",
    "frozen_seed_303",
    "published_card",
    "validation_best_fixed_cell",
    "validation_own_profit_best_fixed_cell",
    "validation_tuned_surge_rule",
    "validation_regime_switch_table",
)


def protocol() -> dict[str, Any]:
    """Return the manual-registered contract without runtime-derived values."""
    return {
        "protocol_version": PROTOCOL_VERSION,
        "entry_point": "scripts/eval_protocol_v1.py",
        "held_out_tapes": {
            "required": True,
            "must_differ_from": ["training", "T2_validation"],
            "common_random_numbers_across_policies": True,
        },
        "ranking_objective": {
            "name": OBJECTIVE_NAME,
            "own_profit_weight": 1.0,
            "competitor_profit_weight": -COMPETITOR_PROFIT_WEIGHT,
            "always_report_components": [
                "own_profit",
                "competitor_profit",
            ],
        },
        "competitor_groups": {
            "training_exposed": list(TRAINING_EXPOSED_COMPETITORS),
            "held_out_canonical": list(HELD_OUT_COMPETITORS),
            "held_out_perturbed": list(PERTURBED_HELD_OUT_COMPETITORS),
        },
        "step_A": {
            "role": "comparator_selection_validation_only",
            "selection_objective": OBJECTIVE_NAME,
            "selectors": [
                "best_fixed_cell",
                "tuned_surge_rule",
                "regime_switch_table",
            ],
            "reference_comparator": "own_profit_best_fixed_cell",
            "report_columns": [
                "final_objective",
                "own_profit",
                "competitor_profit",
                "profit_lead",
            ],
            "cities_full": ["boston", "chicago"],
            "nyc_reuses_T2_cells": True,
            "nyc_runs_rule_only": True,
            "cases": ["C0_calm", "A1_shortage_severe"],
            "competitors": 6,
            "openings": 3,
            "fixed_cells": 36,
            "reactive_rule_candidates": 24,
            "fixed_rollouts_per_full_city": 1296,
            "rule_rollouts_per_city": 864,
        },
        "step_B": {
            "role": "main_held_out_evaluation",
            "cities": ["nyc", "boston", "chicago"],
            "horizon_days": 40,
            "event_after_day": 10,
            "cases": list(CASES),
            "competitors": list(COMPETITORS),
            "openings_per_case_competitor": 5,
            "policies": list(POLICY_CLASSES),
            "rollouts_per_city": 3300,
            "paired_by_opening": True,
            "holm_across_cases": True,
            "headline_pools_training_seeds": True,
            "appendix_reports_seeds_separately": True,
        },
        "step_C": {
            "role": "surge_cap_counterfactual",
            "caps": [1.0, 1.2, 1.5],
            "declared_event_days_only": True,
            "cases": ["A1_shortage_severe", "A3_storm_transit"],
            "competitors": 10,
            "openings": 5,
            "seeds_by_city": {"nyc": 3, "boston": 1, "chicago": 1},
            "rollouts_total": 1500,
            "outputs": [
                "platform_profit",
                "driver_earnings_per_online_hour",
                "riders_served",
                "consumer_surplus_by_income_band",
                "unfulfilled_demand",
            ],
        },
        "step_D": {
            "role": "data_validation_without_policy_simulation",
            "checks": [
                "surge_frequency_vs_boston_lyft_2018",
                "supply_elasticity_vs_nyc_tlc_2019_2024",
                "held_out_kaggle_fare_reality_gap_by_city",
            ],
        },
        "results_contract": {
            "rank_by": OBJECTIVE_NAME,
            "always_report": [
                "final_objective",
                "own_profit",
                "competitor_profit",
                "profit_lead",
            ],
            "cost_of_competing_for_lead": (
                "rl_own_profit_minus_own_profit_best_fixed_cell"
            ),
        },
    }


def canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def protocol_hash(value: dict[str, Any] | None = None) -> str:
    payload = canonical_json(value or protocol()).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate(value: dict[str, Any] | None = None) -> None:
    spec = value or protocol()
    a = spec["step_A"]
    b = spec["step_B"]
    c = spec["step_C"]
    if spec["entry_point"] != "scripts/eval_protocol_v1.py":
        raise ValueError("M7 requires eval_protocol_v1.py as the sole entry point")
    groups = spec["competitor_groups"]
    if tuple(groups["training_exposed"]) != TRAINING_EXPOSED_COMPETITORS:
        raise ValueError("training-exposed competitor group drifted")
    if tuple(groups["held_out_canonical"]) != HELD_OUT_COMPETITORS:
        raise ValueError("canonical held-out competitor group drifted")
    if tuple(groups["held_out_perturbed"]) != PERTURBED_HELD_OUT_COMPETITORS:
        raise ValueError("perturbed held-out competitor group drifted")
    if len(set(COMPETITORS)) != 10:
        raise ValueError("competitor groups must be disjoint and contain ten entries")
    if tuple(b["competitors"]) != COMPETITORS:
        raise ValueError("Step B competitor order drifted")
    objective = spec["ranking_objective"]
    if (
        objective["name"] != OBJECTIVE_NAME
        or float(objective["own_profit_weight"]) != 1.0
        or float(objective["competitor_profit_weight"])
        != -COMPETITOR_PROFIT_WEIGHT
        or objective["always_report_components"]
        != ["own_profit", "competitor_profit"]
    ):
        raise ValueError("final policy-ranking objective drifted")
    if (
        a["selection_objective"] != OBJECTIVE_NAME
        or a["selectors"]
        != ["best_fixed_cell", "tuned_surge_rule", "regime_switch_table"]
        or a["reference_comparator"] != "own_profit_best_fixed_cell"
        or spec["results_contract"]["rank_by"] != OBJECTIVE_NAME
        or spec["results_contract"]["always_report"]
        != ["final_objective", "own_profit", "competitor_profit", "profit_lead"]
    ):
        raise ValueError("F3 selector or results contract drifted")
    if (
        a["fixed_cells"] * a["competitors"] * len(a["cases"]) * a["openings"]
        != a["fixed_rollouts_per_full_city"]
        or a["fixed_rollouts_per_full_city"] != 1296
    ):
        raise ValueError("Step A fixed-cell rollout count drifted")
    if a["reactive_rule_candidates"] * a["competitors"] * len(a["cases"]) * a["openings"] != 864:
        raise ValueError("Step A rule rollout count drifted")
    expected_b = (
        len(b["cases"])
        * len(b["competitors"])
        * b["openings_per_case_competitor"]
        * len(b["policies"])
    )
    if expected_b != b["rollouts_per_city"] or expected_b != 3300:
        raise ValueError("Step B rollout count drifted")
    expected_c = (
        sum(c["seeds_by_city"].values())
        * len(c["caps"])
        * len(c["cases"])
        * c["competitors"]
        * c["openings"]
    )
    if expected_c != c["rollouts_total"] or expected_c != 1500:
        raise ValueError("Step C rollout count drifted")
    if not spec["held_out_tapes"]["common_random_numbers_across_policies"]:
        raise ValueError("common random numbers are mandatory")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--describe", action="store_true")
    mode.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec = protocol()
    validate(spec)
    result = {
        "state": "contract_valid",
        "execution_enabled": False,
        "protocol_hash": protocol_hash(spec),
        "protocol": spec,
    }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
