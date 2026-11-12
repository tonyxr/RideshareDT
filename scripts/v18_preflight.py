#!/usr/bin/env python3
"""Plan, record and freeze the seven registered V18 preflight decisions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.v18_config import BASE, DEFAULT_CONFIG, HORIZON_CANDIDATES, load_config, validate_config
from scripts.v18_runtime import atomic_json, utc_now


STEPS: dict[str, dict[str, Any]] = {
    "P1": {"name": "tests_and_legacy_identity", "hours": [1, 2],
           "decision": "stop_on_any_mismatch",
           "runs": "tests/test_v18.py plus NYC C0/A1 identity, 1 rival x 1 opening"},
    "P2": {"name": "calibration_and_effect_table", "hours": [3, 4],
           "decision": "fitted_supply",
           "runs": "pay OLS; fleet/pickup fit at N=500; reference earnings; T1.1-T1.3; nested effects"},
    "P3": {"name": "scale_check", "hours": [4, 6], "decision": "N_star",
           "runs": "N in 50,100,150,250,500; NYC/Chicago; C0/A1; 5 cards; 3 rivals x 3 openings"},
    "P4": {"name": "timing", "hours": [2, 2],
           "decision": "ppo_clock_days_and_mc_clock_days",
           "runs": "20 days agent H5/H10, PPO and mc_return plus 40-day eval at N* and 500"},
    "P5": {"name": "threshold_sources", "hours": [3, 4],
           "decision": "threshold_outcome",
           "runs": "36 cards x C0/A1 x NYC/Chicago x four threshold sources"},
    "P6": {"name": "calm_gap_probe", "hours": [3, 3],
           "decision": "H_star_and_value_head",
           "runs": "legacy agent/PPO; probe K=2; horizons 5/10; 3 rivals x 2 openings"},
    "P7": {"name": "rival_proxy_bias", "hours": [2, 2],
           "decision": "branch_planner",
           "runs": "P6 states; four profit rivals; proxy versus full planner"},
}


def report_path(step: str) -> Path:
    return BASE / f"runs/v18/preflight/{step}/report.json"


def validate_report(step: str, report: Mapping[str, Any]) -> None:
    if report.get("step") != step or report.get("state") != "complete":
        raise ValueError(f"{step} report must be complete and identify its step")
    decision = dict(report.get("decision") or {})
    if step == "P1" and decision.get("identity_pass") is not True:
        raise ValueError("P1 identity did not pass")
    if step == "P2" and not isinstance(decision.get("fitted_supply"), dict):
        raise ValueError("P2 must record fitted_supply by city")
    if step == "P3" and int(decision.get("N_star", -1)) not in (100, 150, 250):
        raise ValueError("P3 N_star must be 100, 150 or 250")
    if step == "P4" and not all(int(decision.get(key, 0)) > 0 for key in ("ppo_clock_days", "mc_clock_days")):
        raise ValueError("P4 must record positive clock-matched days")
    if step == "P5" and not decision.get("threshold_outcome"):
        raise ValueError("P5 must record the threshold outcome")
    if step == "P6":
        if int(decision.get("H_star", -1)) not in HORIZON_CANDIDATES:
            raise ValueError("P6 H_star is invalid")
        if decision.get("value_head") not in {"concave2d", "concave2d_residual"}:
            raise ValueError("P6 value_head is invalid")
    if step == "P7":
        planner = dict(decision.get("branch_planner") or {})
        if (int(planner.get("sample_cap", 0)), int(planner.get("horizon_days", 0))) not in {(8, 1), (16, 2)}:
            raise ValueError("P7 branch planner must be proxy 8/1 or expanded 16/2")


def freeze(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    decisions: dict[str, Any] = {}
    for step in STEPS:
        report = json.loads(report_path(step).read_text(encoding="utf-8"))
        validate_report(step, report)
        decisions.update(dict(report.get("decision") or {}))
    required = {
        "N_star", "H_star", "threshold_outcome", "branch_planner",
        "ppo_clock_days", "mc_clock_days", "value_head", "fitted_supply",
    }
    missing = sorted(required - set(decisions))
    if missing:
        raise ValueError("preflight decisions missing: " + ", ".join(missing))
    config["decisions"] = {key: decisions[key] for key in sorted(required)}
    config["status"] = "frozen"
    config["population"]["customers"] = int(decisions["N_star"])
    config["credit"]["horizon_days"] = int(decisions["H_star"])
    config["credit"].update({
        "branch_profit_planner_sample_cap": int(decisions["branch_planner"]["sample_cap"]),
        "branch_profit_planner_horizon_days": int(decisions["branch_planner"]["horizon_days"]),
    })
    config["learner"]["value_head"] = decisions["value_head"]
    config["training"]["ppo_clock_days"] = int(decisions["ppo_clock_days"])
    config["training"]["mc_clock_days"] = int(decisions["mc_clock_days"])
    temporary = config_path.with_name(f".{config_path.name}.tmp.{os.getpid()}")
    temporary.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    os.replace(temporary, config_path)
    validate_config(load_config(config_path), require_frozen=True)
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", choices=tuple(STEPS))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--record-report", type=Path)
    parser.add_argument("--run-unit-tests", action="store_true")
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        frozen = freeze(args.config)
        print(json.dumps(frozen["decisions"], indent=2, sort_keys=True))
        return 0
    if args.step is None:
        raise ValueError("--step is required unless --freeze is used")
    directory = report_path(args.step).parent
    directory.mkdir(parents=True, exist_ok=True)
    plan = {"step": args.step, "state": "planned", "updated_utc": utc_now(), **STEPS[args.step]}
    atomic_json(directory / "plan.json", plan)
    if args.run_unit_tests:
        if args.step != "P1":
            raise ValueError("--run-unit-tests is only valid for P1")
        subprocess.run(
            [str(BASE.parent / ".venv/bin/python"), "-m", "pytest", "-q", "tests/test_v18.py"],
            cwd=BASE, check=True,
        )
    if args.record_report:
        report = json.loads(args.record_report.read_text(encoding="utf-8"))
        validate_report(args.step, report)
        atomic_json(report_path(args.step), report)
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
