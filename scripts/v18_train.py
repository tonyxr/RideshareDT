#!/usr/bin/env python3
"""Plan or run the registered V18 Wave A/B training queue."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.v18_config import BASE, load_config, train_args, validate_config
from scripts.v18_runtime import Job, atomic_json, run_jobs


PYTHON = (BASE.parent / ".venv/bin/python").resolve()


@dataclass(frozen=True)
class RunSpec:
    name: str
    city: str
    seed: int
    arm: str
    wave: str


RUNS = (
    *(RunSpec(f"{city}_s{seed}", city, seed, "main", "A")
      for city in ("nyc", "chicago", "boston") for seed in (101, 202, 303)),
    *(RunSpec(f"nyc_s{seed}_ppo_clock", "nyc", seed, "ppo_clock", "B") for seed in (101, 202)),
    *(RunSpec(f"nyc_s{seed}_mc_clock", "nyc", seed, "mc_clock", "B") for seed in (101, 202)),
    RunSpec("nyc_s404", "nyc", 404, "main", "B"),
    RunSpec("nyc_s505", "nyc", 505, "main", "B"),
    RunSpec("nyc_s101_nocrn", "nyc", 101, "nocrn", "B"),
    RunSpec("nyc_s101_no_driver_obs", "nyc", 101, "no_driver_obs", "B"),
    RunSpec("nyc_s101_own_profit", "nyc", 101, "own_profit", "B"),
    RunSpec("nyc_s101_ppo", "nyc", 101, "ppo", "B"),
)


def jobs(config: dict, *, waves: set[str], root: Path) -> list[Job]:
    result: list[Job] = []
    for spec in RUNS:
        if spec.wave not in waves:
            continue
        run_dir = (root / spec.name).resolve()
        command = (
            str(PYTHON), "-u", str(BASE / "Core.py"),
            *train_args(
                config, city=spec.city, seed=spec.seed, arm=spec.arm,
                output=run_dir,
            ),
            "--training_resume_state", str(run_dir / "resume/training_state.pt"),
            "--training_resume_every_days", "1",
        )
        result.append(Job(
            name=spec.name, command=tuple(command),
            output=run_dir / "policy.pt", log=run_dir / "run.log",
            manifest=run_dir / "manifest.json",
        ))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=BASE / "configs/v18_production.yaml")
    parser.add_argument("--wave", choices=("A", "B", "all"), default="A")
    parser.add_argument("--workers", type=int, default=9)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--root", type=Path, default=BASE / "runs/v18/train")
    args = parser.parse_args()
    config = load_config(args.config)
    validate_config(config, require_frozen=True)
    waves = {"A", "B"} if args.wave == "all" else {args.wave}
    planned = jobs(config, waves=waves, root=args.root)
    plan_path = BASE / "runs/v18/train/plan.json"
    atomic_json(plan_path, {
        "waves": sorted(waves), "count": len(planned),
        "jobs": [{"name": job.name, "command": list(job.command)} for job in planned],
    })
    result = run_jobs(
        planned, cwd=BASE, workers=args.workers, dry_run=not args.execute
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
