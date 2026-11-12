#!/usr/bin/env python3
"""Plan or run the ten registered V18 evaluation suites."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.v18_config import BASE, eval_args, load_config, validate_config
from scripts.v18_runtime import Job, atomic_json, run_jobs


PYTHON = (BASE.parent / ".venv/bin/python").resolve()
SCENARIOS = {
    "C0": BASE / "scenarios/calm.json",
    "A1": BASE / "scenarios/v17_t2_shortage_severe.json",
    "A3": BASE / "scenarios/cases/storm_transit_late.json",
    "S1": BASE / "scenarios/cases/promotion_bait.json",
    "S6": BASE / "scenarios/cases/shortage_during_promotion.json",
    "i080": BASE / "scenarios/cases/v17_shortage_i080.json",
    "i065": BASE / "scenarios/cases/v17_shortage_i065.json",
    "i050": BASE / "scenarios/cases/v17_shortage_i050.json",
    "i035": BASE / "scenarios/v18/shortage_i035.json",
    "natural_year_2023": BASE / "scenarios/v18/natural_year_2023.json",
}
SUITES: dict[str, dict[str, Any]] = {
    "E1": {"name": "main", "cases": ("C0", "A1", "A3", "S1", "S6"), "cities": ("nyc", "boston", "chicago"), "openings": 3},
    "E2": {"name": "severity_ladder", "cases": ("i080", "i065", "i050", "i035"), "cities": ("nyc", "boston", "chicago"), "openings": 2},
    "E3": {"name": "second_seed", "cases": ("C0", "A1"), "cities": ("nyc", "boston", "chicago"), "openings": 2, "seed": 88411},
    "E4": {"name": "learning_comparison", "cases": ("C0", "A1", "A3", "S6"), "cities": ("nyc",), "openings": 3, "credit_probe": True},
    "E5": {"name": "natural_year", "cases": ("natural_year_2023",), "cities": ("nyc", "boston", "chicago"), "openings": 2, "request_log": True},
    "E6": {"name": "scale_confirmation", "cases": ("C0", "A1"), "cities": ("nyc", "chicago"), "openings": 2, "customers": 500},
    "E7": {"name": "threshold_sources", "cases": ("C0", "A1"), "cities": ("nyc", "chicago"), "openings": 2, "threshold_sources": ("constant_mean", "shuffled", "rubric")},
    "E8": {"name": "driver_mechanism_map", "cases": ("C0", "A1"), "cities": ("nyc",), "openings": 2, "points": 11},
    "E9": {"name": "left_out_market", "cases": ("manual_part_D",), "cities": ("nyc", "chicago"), "variants": 7},
    "E10": {"name": "dataset", "cases": ("C0", "A1", "A3"), "cities": ("nyc", "boston", "chicago"), "openings": 1, "request_log": True},
}


def reference_specs(selected: Mapping[str, Any]) -> list[str]:
    keys = (
        "published_card", "best_fixed", "dated_two_lever",
        "count_rule_0step", "count_rule_1step", "reactive_surge",
    )
    specs: list[str] = []
    for key in keys:
        value = dict(selected.get(key) or {})
        if "spec" in value:
            specs.append(str(value["spec"]))
    if len(specs) != 6:
        raise ValueError("selection product does not contain all six references")
    return specs


def _set(values: Sequence[str], flag: str, value: Any) -> list[str]:
    result = list(values)
    while flag in result:
        index = result.index(flag)
        del result[index:index + 2]
    result.extend((flag, str(value)))
    return result


def _drop(values: Sequence[str], flags: Sequence[str]) -> list[str]:
    drop = set(flags)
    result: list[str] = []
    index = 0
    while index < len(values):
        if values[index] in drop:
            index += 2
        else:
            result.append(values[index])
            index += 1
    return result


def policy_runs(suite: str, city: str) -> list[str]:
    main = [f"{city}_s{seed}" for seed in (101, 202, 303)]
    if suite in {"E1", "E2", "E3", "E5"}:
        return main + (["nyc_s404", "nyc_s505"] if city == "nyc" else [])
    if suite == "E4":
        arms = ("", "_ppo", "_ppo_clock", "_mc_clock", "_nocrn",
                "_no_driver_obs", "_own_profit")
        return [f"nyc_s{seed}{arm}" for seed in (101, 202) for arm in arms] + ["nyc_s101_halt"]
    if suite in {"E6", "E7", "E8"}:
        return main
    if suite == "E10":
        return [f"{city}_s101"]
    return []


def build_jobs(config: dict, *, suite: str, root: Path) -> list[Job]:
    definition = SUITES[suite]
    jobs: list[Job] = []
    for city in definition["cities"]:
        selected_path = BASE / f"runs/v18/select/{city}/selected.json"
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
        specs = reference_specs(selected)
        for case in definition["cases"]:
            if case == "manual_part_D":
                continue
            leaf = (root / suite / city / case / "references").resolve()
            values = eval_args(
                config, city=city, case=case, specs=specs, suite=suite,
                output=leaf, seed=int(definition.get("seed", 77311)),
                scenario=SCENARIOS[case],
            )
            values = _set(values, "--policy_sweep_openings", definition["openings"])
            if "customers" in definition:
                values = _set(values, "--eval_customers", definition["customers"])
            if definition.get("credit_probe"):
                values = _set(values, "--v17_eval_credit_probe", 5)
                values = _set(values, "--v17_eval_credit_horizons", "5,10")
                values = _set(values, "--v17_eval_credit_rival_planner", "proxy")
                values = _set(values, "--v17_eval_credit_probe_out", leaf / "credit_probe.jsonl")
            command = (str(PYTHON), "-u", str(BASE / "Core.py"), *values)
            jobs.append(Job(
                name=f"{suite}/{city}/{case}/references", command=tuple(command),
                output=leaf / "raw.json", log=leaf / "run.log",
                manifest=leaf / "manifest.json",
            ))
            for run_name in policy_runs(suite, city):
                run_leaf = (root / suite / city / case / run_name).resolve()
                policy_values = eval_args(
                    config, city=city, case=case, specs=("rl_actor",),
                    suite=suite, output=run_leaf,
                    seed=int(definition.get("seed", 77311)),
                    scenario=SCENARIOS[case],
                )
                policy_values = _set(
                    policy_values, "--policy_sweep_openings",
                    definition["openings"],
                )
                if "customers" in definition:
                    policy_values = _set(
                        policy_values, "--eval_customers",
                        definition["customers"],
                    )
                if "_ppo" in run_name:
                    policy_values = _drop(policy_values, (
                        "--v16_branch_mode", "--v16_branch_horizon_days",
                        "--v16_branch_design", "--v16_audit_every",
                        "--v16_store_cap", "--v16_value_head",
                        "--v17_move_radius", "--v17_temperature_mode",
                        "--v17_tau_rel_start", "--v17_tau_rel_end",
                    ))
                    policy_values = _set(
                        policy_values, "--v16_policy_mode", "ppo"
                    )
                if "_mc_clock" in run_name:
                    policy_values = _set(
                        policy_values, "--v17_credit_source", "mc_return"
                    )
                if "_nocrn" in run_name:
                    policy_values = _set(
                        policy_values, "--v17_branch_noise", "independent"
                    )
                if definition.get("credit_probe"):
                    policy_values = _set(
                        policy_values, "--v17_eval_credit_probe", 5
                    )
                    policy_values = _set(
                        policy_values, "--v17_eval_credit_horizons", "5,10"
                    )
                    policy_values = _set(
                        policy_values, "--v17_eval_credit_rival_planner", "proxy"
                    )
                    policy_values = _set(
                        policy_values, "--v17_eval_credit_probe_out",
                        run_leaf / "credit_probe.jsonl",
                    )
                policy_values.extend((
                    "--trained_model_in",
                    str((BASE / "runs/v18/train" / run_name / "policy.pt").resolve()),
                    "--eval_only",
                ))
                policy_command = (
                    str(PYTHON), "-u", str(BASE / "Core.py"), *policy_values
                )
                jobs.append(Job(
                    name=f"{suite}/{city}/{case}/{run_name}",
                    command=tuple(policy_command),
                    output=run_leaf / "raw.json", log=run_leaf / "run.log",
                    manifest=run_leaf / "manifest.json",
                ))
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=tuple(SUITES), required=True)
    parser.add_argument("--config", type=Path, default=BASE / "configs/v18_production.yaml")
    parser.add_argument("--root", type=Path, default=BASE / "runs/v18/eval")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    definition = SUITES[args.suite]
    plan = {"suite": args.suite, **definition}
    atomic_json(args.root / args.suite / "plan.json", plan)
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    config = load_config(args.config)
    validate_config(config, require_frozen=True)
    planned = build_jobs(config, suite=args.suite, root=args.root)
    result = run_jobs(planned, cwd=BASE, workers=args.workers)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
