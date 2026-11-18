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
from scripts.v18_train import RUNS


PYTHON = (BASE.parent / ".venv/bin/python").absolute()
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
    "E1": {"name": "main", "tier": 1, "cases": ("C0", "A1", "A3", "S1", "S6"), "cities": ("nyc", "boston", "chicago"), "openings": 3, "rivals": 10},
    "E2": {"name": "severity_ladder", "tier": 1, "cases": ("i080", "i065", "i050", "i035"), "cities": ("nyc", "boston", "chicago"), "openings": 1, "rivals": 10},
    "E4": {
        "name": "learning_comparison", "tier": 1,
        "cases": ("C0", "A1", "A3", "S6"), "cities": ("nyc",),
        "openings": 3, "rivals": 10,
        "output_subdir": "E4_no_probe",
        "legacy_probe_output": "E4",
        "row_sources": {
            "nyc_s101": "E1_recalibrated_supply",
            "nyc_s202": "E1_recalibrated_supply",
            "wave_b": "E4_no_probe",
        },
    },
    "E5": {"name": "natural_year", "tier": 1, "cases": ("natural_year_2023",), "cities": ("nyc", "boston", "chicago"), "openings": 2, "rivals": 3, "request_log": True},
    "E10": {"name": "dataset", "tier": 1, "cases": ("C0", "A1", "A3"), "cities": ("nyc", "boston", "chicago"), "openings": 1, "rivals": 10, "request_log": True},
    "E6": {"name": "scale_confirmation", "tier": 2, "cases": ("C0", "A1"), "cities": ("nyc", "chicago"), "openings": 1, "rivals": 10, "customers": 500},
    "E7": {"name": "threshold_sources", "tier": 2, "cases": ("C0", "A1"), "cities": ("nyc", "chicago"), "openings": 2, "rivals": 3, "threshold_sources": ("constant_mean", "shuffled", "rubric")},
    "E3": {"name": "second_seed", "tier": 2, "cases": ("C0", "A1"), "cities": ("nyc", "boston", "chicago"), "openings": 1, "rivals": 10, "seed": 88411},
    "T8": {
        "name": "credit_probe_diagnostic", "tier": 3,
        "cases": ("C0", "A1"), "cities": ("nyc",),
        "openings": 1, "rivals": 3, "credit_probe": True,
        "diagnostic_only": True, "trajectories_used": False,
    },
}
R3 = ("adaptive_best_response_memory", "profit_best_response_myopic", "profit_opt_mpc_grid")
R5 = ("adaptive_best_response_memory", "profit_best_response_myopic", "profit_best_response_forward", "profit_opt_mpc_grid", "pi_price_gap")


def reference_specs(selected: Mapping[str, Any]) -> dict[str, str]:
    keys = (
        "published_card", "best_fixed", "dated_two_lever",
        "count_rule_0step", "count_rule_1step", "reactive_surge",
    )
    specs: dict[str, str] = {}
    for key in keys:
        value = dict(selected.get(key) or {})
        if "spec" in value:
            specs[key] = str(value["spec"])
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
        return [
            "nyc_s101", "nyc_s202", "nyc_s101_ppo_clock",
            "nyc_s202_ppo_clock", "nyc_s101_mc_clock", "nyc_s202_mc_clock",
            "nyc_s101_ppo", "nyc_s101_nocrn",
            "nyc_s101_no_driver_obs", "nyc_s101_own_profit",
        ]
    if suite in {"E6", "E7"}:
        return main
    if suite == "E10":
        return [f"{city}_s101"]
    if suite == "T8":
        return ["nyc_s101"]
    return []


def build_jobs(config: dict, *, suite: str, root: Path) -> list[Job]:
    definition = SUITES[suite]
    jobs: list[Job] = []
    for city in definition["cities"]:
        selected_path = BASE / f"runs/v18/select/{city}/selected.json"
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
        references = reference_specs(selected)
        if suite in {"E5"}:
            reference_names = ("published_card", "best_fixed", "dated_two_lever", "count_rule_1step", "reactive_surge")
        elif suite in {"E10", "E6", "E7"}:
            reference_names = ("published_card", "best_fixed", "dated_two_lever", "count_rule_1step")
        elif suite in {"E4", "T8"}:
            reference_names = ()
        else:
            reference_names = tuple(references)
        rivals = tuple(config["evaluation"]["rivals"][: int(definition["rivals"])])
        if int(definition["rivals"]) == 3:
            rivals = R3
        elif int(definition["rivals"]) == 5:
            rivals = R5
        for case in definition["cases"]:
            threshold_sources = definition.get("threshold_sources", ("as_loaded",))
            for threshold_source in threshold_sources:
              case_key = case if threshold_source == "as_loaded" else f"{case}_{threshold_source}"
              for reference_name in reference_names:
                leaf = (root / suite / city / case_key / reference_name).resolve()
                values = eval_args(
                    config, city=city, case=case, specs=(references[reference_name],), suite=suite,
                    output=leaf, seed=int(definition.get("seed", 77311)),
                    scenario=SCENARIOS[case], customers=definition.get("customers"),
                )
                values = _set(values, "--policy_sweep_openings", definition["openings"])
                values = _set(values, "--policy_sweep_rivals", ",".join(rivals))
                if threshold_source != "as_loaded":
                    values = _set(values, "--threshold_ablation", threshold_source)
                command = (str(PYTHON), "-u", str(BASE / "Core.py"), *values)
                jobs.append(Job(
                    name=f"{suite}/{city}/{case_key}/{reference_name}", command=tuple(command),
                    output=leaf / "raw.json", log=leaf / "run.log",
                    manifest=leaf / "manifest.json", tier=int(definition["tier"]),
                ))
            for run_name in policy_runs(suite, city):
              for threshold_source in threshold_sources:
                case_key = case if threshold_source == "as_loaded" else f"{case}_{threshold_source}"
                if suite == "E4" and run_name in {"nyc_s101", "nyc_s202"}:
                    output_suite = "E1_recalibrated_supply"
                    command_suite = "E1"
                else:
                    output_suite = str(definition.get("output_subdir", suite))
                    command_suite = suite
                run_leaf = (
                    root / output_suite / city / case_key / run_name
                ).resolve()
                spec = next(item for item in RUNS if item.name == run_name)
                policy_values = eval_args(
                    config, city=city, case=case, specs=("rl_actor",),
                    suite=command_suite, output=run_leaf,
                    seed=int(definition.get("seed", 77311)),
                    scenario=SCENARIOS[case], kind=spec.arm,
                    policy=BASE / "runs/v18/train" / run_name / "policy.pt",
                    customers=definition.get("customers"),
                )
                policy_values = _set(
                    policy_values, "--policy_sweep_openings",
                    definition["openings"],
                )
                policy_values = _set(policy_values, "--policy_sweep_rivals", ",".join(rivals))
                if threshold_source != "as_loaded":
                    policy_values = _set(policy_values, "--threshold_ablation", threshold_source)
                if suite == "E4" and run_name in {"nyc_s101", "nyc_s202"}:
                    if "--recalibrate_supply_on_eval" in policy_values:
                        raise ValueError(
                            "duplicate --recalibrate_supply_on_eval"
                        )
                    policy_values.append("--recalibrate_supply_on_eval")
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
                policy_command = (
                    str(PYTHON), "-u", str(BASE / "Core.py"), *policy_values
                )
                jobs.append(Job(
                    name=f"{suite}/{city}/{case_key}/{run_name}",
                    command=tuple(policy_command),
                    output=run_leaf / "raw.json", log=run_leaf / "run.log",
                    manifest=run_leaf / "manifest.json", tier=int(definition["tier"]),
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
    plan_root = args.root / str(definition.get("output_subdir", args.suite))
    atomic_json(plan_root / "plan.json", plan)
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
