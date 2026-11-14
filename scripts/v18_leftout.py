#!/usr/bin/env python3
"""Run the Last Revision E9 left-out-market selections and loss evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.v17_robust_select import LEFTOUT_VARIANTS
from scripts.v18_config import eval_args, load_config, validate_config
from scripts.v18_eval import SCENARIOS, reference_specs
from scripts.v18_runtime import Job, atomic_json, run_jobs
from scripts.v18_select import run_selection

PYTHON = (REPO.parent / ".venv/bin/python").resolve()
VARIANTS = ("drivers_fixed", "no_loyalty", "static_rival", "no_spillover")


def eval_jobs(config: dict, *, variant: str, selected: dict, root: Path) -> list[Job]:
    production = json.loads((REPO / "runs/v18/select/nyc/selected.json").read_text())
    production_specs = reference_specs(production)
    specs = {
        "variant_best_fixed": selected["best_fixed"]["spec"],
        "variant_dated": selected["dated_two_lever"]["spec"],
        **{f"production_{key}": value for key, value in production_specs.items()},
    }
    result: list[Job] = []
    for case in ("C0", "A1"):
        for name, spec in specs.items():
            leaf = root / variant / "eval" / case / name
            values = eval_args(config, city="nyc", case=case, specs=(spec,), suite="E9",
                               output=leaf, scenario=SCENARIOS[case])
            result.append(Job(f"E9/{variant}/{case}/{name}",
                (str(PYTHON), "-u", str(REPO / "Core.py"), *values),
                leaf / "raw.json", leaf / "run.log", leaf / "manifest.json", tier=3))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO / "runs/v18/eval/E9")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = load_config(); validate_config(config, require_frozen=True)
    atomic_json(args.root / "plan.json", {"variants": VARIANTS, "report_only": True})
    if not args.execute:
        print(json.dumps({"variants": VARIANTS}, indent=2)); return 0
    states: dict[str, str] = {}
    for variant in VARIANTS:
        settings = LEFTOUT_VARIANTS[variant]
        openings = 3 if variant == "static_rival" else 1
        selected = run_selection(config, city="nyc", output=args.root / variant / "selection",
            workers=1, overrides=settings["overrides"], selection_rivals=settings["rivals"],
            selection_openings=openings, stage3=False)
        states.update(run_jobs(eval_jobs(config, variant=variant, selected=selected, root=args.root),
                               cwd=REPO, workers=args.workers))
    atomic_json(args.root / "report.json", {"state": "complete", "report_only": True, "jobs": states})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
