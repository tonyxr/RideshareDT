#!/usr/bin/env python3
"""Run the declared V18 V3 event/calm twins without changing model science."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping


BASE = Path(__file__).resolve().parents[2]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from scripts.v18_config import eval_args, load_config, validate_config  # noqa: E402
from scripts.v18_eval import R5, _set, reference_specs  # noqa: E402
from scripts.v18_runtime import Job, atomic_json, run_jobs  # noqa: E402
from scripts.v18_train import RUNS  # noqa: E402


PYTHON = (BASE.parent / ".venv/bin/python").absolute()
SCENARIO_ROOT = BASE / "scenarios/validation"
OUTPUT_ROOT = BASE / "runs/v18/realdata/v3_events/sim"

EVENT_PAIRS: Mapping[str, tuple[tuple[str, str, str], ...]] = {
    "nyc": (
        (
            "E1_flood_20230929",
            "nyc_e1_flood_20230929.json",
            "twin_nyc_e1_flood_20230929.json",
        ),
        (
            "E2_thanksgiving_2023",
            "e2_thanksgiving_2023.json",
            "twin_e2_thanksgiving_2023.json",
        ),
        (
            "E3_christmas_2023",
            "e3_christmas_2023.json",
            "twin_e3_christmas_2023.json",
        ),
        (
            "E4_snow_20240213",
            "nyc_e4_snow_20240213.json",
            "twin_nyc_e4_snow_20240213.json",
        ),
        (
            "E4_rain_20231218",
            "nyc_e4_rain_20231218.json",
            "twin_nyc_e4_rain_20231218.json",
        ),
    ),
    "chicago": (
        (
            "E2_thanksgiving_2023",
            "e2_thanksgiving_2023.json",
            "twin_e2_thanksgiving_2023.json",
        ),
        (
            "E3_christmas_2023",
            "e3_christmas_2023.json",
            "twin_e3_christmas_2023.json",
        ),
        (
            "E4_snow_20240112",
            "chicago_e4_snow_20240112.json",
            "twin_chicago_e4_snow_20240112.json",
        ),
        (
            "E4_rain_20230917",
            "chicago_e4_rain_20230917.json",
            "twin_chicago_e4_rain_20230917.json",
        ),
    ),
}

REFERENCE_NAMES = (
    "published_card", "best_fixed", "dated_two_lever", "count_rule_1step",
)
AGENT_NAMES: Mapping[str, tuple[str, ...]] = {
    city: tuple(f"{city}_s{seed}" for seed in (101, 202, 303))
    for city in ("nyc", "chicago")
}


def _load_scenario(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_pairs() -> None:
    """Assert the predeclared event/twin calendar contract before planning."""
    for pairs in EVENT_PAIRS.values():
        for _, event_name, twin_name in pairs:
            event_path = SCENARIO_ROOT / event_name
            twin_path = SCENARIO_ROOT / twin_name
            event = _load_scenario(event_path)
            twin = _load_scenario(twin_path)
            if event.get("start_date") != twin.get("start_date"):
                raise ValueError(f"V3 pair calendar mismatch: {event_name}")
            if not event.get("events"):
                raise ValueError(f"V3 event file has no event: {event_name}")
            if twin.get("events"):
                raise ValueError(f"V3 twin is not calm: {twin_name}")


def build_jobs(
    config: Mapping[str, Any], *, root: Path = OUTPUT_ROOT,
) -> list[Job]:
    """One Core process per city, scenario side, and policy (R5 x 2 inside)."""
    validate_config(config, require_frozen=True)
    validate_pairs()
    jobs: list[Job] = []
    for city, pairs in EVENT_PAIRS.items():
        selected = json.loads(
            (BASE / f"runs/v18/select/{city}/selected.json").read_text(
                encoding="utf-8"
            )
        )
        references = reference_specs(selected)
        for event_id, event_name, twin_name in pairs:
            for side, scenario_name in (
                ("event", event_name), ("twin", twin_name)
            ):
                scenario = SCENARIO_ROOT / scenario_name
                for policy_name in (*REFERENCE_NAMES, *AGENT_NAMES[city]):
                    leaf = (
                        root / city / event_id / side / policy_name
                    ).resolve()
                    if policy_name in references:
                        values = eval_args(
                            config, city=city, case="V3",
                            specs=(references[policy_name],), suite="V3",
                            output=leaf, seed=77_311, scenario=scenario,
                        )
                    else:
                        run = next(
                            item for item in RUNS if item.name == policy_name
                        )
                        values = eval_args(
                            config, city=city, case="V3",
                            specs=("rl_actor",), suite="V3", output=leaf,
                            seed=77_311, scenario=scenario, kind=run.arm,
                            policy=(
                                BASE / "runs/v18/train" / policy_name
                                / "policy.pt"
                            ),
                        )
                    values = _set(values, "--policy_sweep_rivals", ",".join(R5))
                    values = _set(values, "--policy_sweep_openings", 2)
                    values = _set(values, "--policy_sweep_horizon", 40 * 6)
                    values = _set(
                        values, "--policy_sweep_request_log",
                        leaf / "requests",
                    )
                    command = (
                        str(PYTHON), "-u", str(BASE / "Core.py"), *values
                    )
                    jobs.append(Job(
                        name=f"V3/{city}/{event_id}/{side}/{policy_name}",
                        command=tuple(command), output=leaf / "raw.json",
                        log=leaf / "run.log", manifest=leaf / "manifest.json",
                        tier=1,
                    ))
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=BASE / "configs/v18_production.yaml",
    )
    parser.add_argument("--root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    jobs = build_jobs(config, root=args.root)
    plan = {
        "schema": "v18-v3-event-twins-v1",
        "state": "planned" if not args.execute else "running",
        "jobs": len(jobs),
        "rollouts": len(jobs) * len(R5) * 2,
        "workers": int(args.workers),
        "seed": 77_311,
        "horizon_days": 40,
        "rivals": list(R5),
        "openings": 2,
        "scenario_hazards": "off",
        "request_log": True,
        "commands": [list(job.command) for job in jobs],
    }
    report_root = BASE / "runs/v18/realdata/v3_events"
    atomic_json(report_root / "plan.json", plan)
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    states = run_jobs(jobs, cwd=BASE, workers=args.workers)
    report = {
        **plan,
        "state": "complete",
        "counts": {
            state: sum(value == state for value in states.values())
            for state in ("complete", "cached", "failed", "skipped")
        },
        "states": states,
    }
    atomic_json(report_root / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
