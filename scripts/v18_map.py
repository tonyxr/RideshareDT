#!/usr/bin/env python3
"""Build and run the report-only E8 eta/omega driver-mechanism map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.v17_robust_common import set_option
from scripts.v18_config import eval_args, load_config, validate_config
from scripts.v18_eval import R3, SCENARIOS, reference_specs
from scripts.v18_runtime import Job, atomic_json, run_jobs

PYTHON = (REPO.parent / ".venv/bin/python").absolute()
ETAS = (0.3, 0.6, 1.0)
OMEGAS = (0.25, 0.5, 0.75)


def jobs(config: dict, root: Path) -> list[Job]:
    validate_config(config, require_frozen=True)
    selected = json.loads((REPO / "runs/v18/select/nyc/selected.json").read_text())
    refs = reference_specs(selected)
    policies = {key: refs[key] for key in ("published_card", "best_fixed", "dated_two_lever", "count_rule_1step")}
    policies.update({f"nyc_s{seed}": "rl_actor" for seed in (101, 202, 303)})
    result: list[Job] = []
    for eta in ETAS:
        for omega in OMEGAS:
            for case in ("C0", "A1"):
                for name, spec in policies.items():
                    leaf = root / f"eta{eta}_omega{omega}" / case / name
                    learned = name.startswith("nyc_s")
                    values = eval_args(config, city="nyc", case=case, specs=(spec,), suite="E8",
                        output=leaf, scenario=SCENARIOS[case], kind="main" if learned else None,
                        policy=REPO / "runs/v18/train" / name / "policy.pt" if learned else None)
                    values = set_option(values, "--supply_eta_v3", eta)
                    values = set_option(values, "--shortage_willingness_share", omega)
                    values = set_option(values, "--policy_sweep_rivals", ",".join(R3))
                    values = set_option(values, "--policy_sweep_openings", 2)
                    result.append(Job(f"E8/{eta}/{omega}/{case}/{name}",
                        (str(PYTHON), "-u", str(REPO / "Core.py"), *values),
                        leaf / "raw.json", leaf / "run.log", leaf / "manifest.json", tier=3))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO / "runs/v18/eval/E8")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    planned = jobs(load_config(), args.root)
    atomic_json(args.root / "plan.json", {"count": len(planned), "report_only": True})
    if args.execute:
        states = run_jobs(planned, cwd=REPO, workers=args.workers)
        atomic_json(args.root / "report.json", {"state": "complete", "jobs": states})
    else:
        print(json.dumps({"count": len(planned)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
