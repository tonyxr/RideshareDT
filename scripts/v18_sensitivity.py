#!/usr/bin/env python3
"""Run the Last Revision S0--S3 realism arms; all outputs are report-only."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.v18_config import eval_args, load_config
from scripts.v18_market import market_flags
from scripts.v18_runtime import Job, atomic_json, run_jobs
from scripts.v18_select import run_selection
from scripts.v18_train import RUNS

PYTHON = (REPO.parent / ".venv/bin/python").absolute()
ARMS = {"S0": "production", "S1": "s1", "S2": "s2", "S3": "s3"}
SCENARIOS = {"C0": REPO / "scenarios/calm.json", "A1": REPO / "scenarios/v17_t2_shortage_severe.json"}


def frozen_at_500(config: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(config)
    value["status"] = "frozen"
    value["decisions"].update({"N_star": 500, "ppo_clock_days": 1200,
                                "mc_clock_days": 1200, "workers": 1,
                                "projection": {"source": "S", "wave_a_hours": 0}})
    return value


def evaluation_jobs(config: dict[str, Any], *, arm: str, selected: dict[str, Any], root: Path) -> list[Job]:
    variant = ARMS[arm]
    references = {
        key: selected[key]["spec"]
        for key in ("published_card", "best_fixed", "dated_two_lever", "count_rule_1step")
    }
    result: list[Job] = []
    for case, scenario in SCENARIOS.items():
        for name, spec in references.items():
            leaf = root / arm / "eval" / case / name
            args = eval_args(config, city="nyc", case=case, specs=(spec,), suite="S",
                             output=leaf, scenario=scenario, customers=500)
            args, _ = market_flags(args, variant=variant, city="nyc", customers=500)
            result.append(Job(f"S/{arm}/{case}/{name}",
                              (str(PYTHON), "-u", str(REPO / "Core.py"), *args),
                              leaf / "raw.json", leaf / "run.log", leaf / "manifest.json", tier=2))
        for run in (item for item in RUNS if item.city == "nyc" and item.seed in (101, 202, 303) and item.arm == "main"):
            leaf = root / arm / "eval" / case / run.name
            args = eval_args(config, city="nyc", case=case, specs=("rl_actor",), suite="S",
                             output=leaf, scenario=scenario, kind="main",
                             policy=REPO / "runs/v18/train" / run.name / "policy.pt", customers=500)
            args, _ = market_flags(args, variant=variant, city="nyc", customers=500)
            result.append(Job(f"S/{arm}/{case}/{run.name}",
                              (str(PYTHON), "-u", str(REPO / "Core.py"), *args),
                              leaf / "raw.json", leaf / "run.log", leaf / "manifest.json", tier=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO / "runs/v18/sensitivity")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = frozen_at_500(load_config())
    plan = {"arms": ARMS, "selection_seed": 77311, "customers": 500, "report_only": True}
    atomic_json(args.root / "plan.json", plan)
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return 0
    states: dict[str, str] = {}
    for arm, variant in ARMS.items():
        selected = run_selection(config, city="nyc", output=args.root / arm / "selection",
                                 workers=min(3, args.workers), variant=variant,
                                 customers=500, seed=77311)
        jobs = evaluation_jobs(config, arm=arm, selected=selected, root=args.root)
        states.update(run_jobs(jobs, cwd=REPO, workers=args.workers))
    atomic_json(args.root / "report.json", {"state": "complete", "report_only": True,
                                             "jobs": states})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
