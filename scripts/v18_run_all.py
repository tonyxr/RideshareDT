#!/usr/bin/env python3
"""One Last Revision controller: checks, P3/P4, freeze, and the queue."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.v18_config import load_config, validate_config
from scripts.v18_eval import SUITES, build_jobs as evaluation_jobs
from scripts.v18_runtime import Job, append_progress, atomic_json, run_job, run_jobs
from scripts.v18_train import jobs as training_jobs


PYTHON = (REPO.parent / ".venv/bin/python").absolute()
CONTROLLER = REPO / "runs/v18/controller"
DROP_ORDER = (
    "T8", "E8", "E9", "E3", "S3", "S1", "E7/chicago", "E6/chicago",
    "nyc_s505", "nyc_s404", "E4/mc_clock_curves",
)
NEVER_DROP = ("P3", "P4", "selection", "wave_A", "E1", "E2", "E4/ladder", "E5", "E10", "V1", "V2", "V3", "V4", "V5", "V6", "V7")


def _status(**updates: Any) -> None:
    path = CONTROLLER / "status.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        value = {}
    value.update(updates)
    atomic_json(path, value)


def _phase_job(name: str, command: Sequence[str], output: Path) -> Job:
    leaf = CONTROLLER / "phases" / name
    return Job(
        name=name, command=tuple(map(str, command)), output=output,
        log=leaf / "run.log", manifest=leaf / "manifest.json", tier=0,
    )


def _run_phase(job: Job) -> str:
    _status(current_phase=job.name, state="running")
    state = run_job(
        job, cwd=REPO, progress=CONTROLLER / "progress.jsonl",
        max_attempts=3,
    )
    _status(current_phase=job.name, state=state)
    return state


def freeze(config_path: Path, p3_path: Path, p4_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    p3 = json.loads(p3_path.read_text(encoding="utf-8"))
    p4 = json.loads(p4_path.read_text(encoding="utf-8"))
    decisions = dict(config["decisions"])
    decisions.update({
        "N_star": int(p4.get("N_star", p3["N_star"])),
        "ppo_clock_days": int(p4["ppo_clock_days"]),
        "mc_clock_days": int(p4["mc_clock_days"]),
        "workers": int(p4["workers"]),
        "projection": {
            "wave_a_hours": float(p4["projected_wave_a_hours"]),
            "source": str(p4_path),
        },
    })
    config["decisions"] = decisions
    config["status"] = "frozen"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    validate_config(config, require_frozen=True)
    append_progress(
        CONTROLLER / "progress.jsonl", job="freeze", state="complete",
        attempt=1, seconds=0,
    )
    atomic_json(CONTROLLER / "freeze.json", {
        "state": "complete", "config": str(config_path),
        "decisions": decisions,
    })
    return config


def _selection_jobs(config: dict[str, Any]) -> list[Job]:
    return [
        Job(
            name=f"selection/{city}",
            command=(
                str(PYTHON), "-u", str(REPO / "scripts/v18_select.py"),
                "--city", city, "--run", "--workers", "1",
            ),
            output=REPO / f"runs/v18/select/{city}/selected.json",
            log=CONTROLLER / f"selection_{city}.log",
            manifest=CONTROLLER / f"selection_{city}_manifest.json",
            tier=0,
        )
        for city in ("nyc", "boston", "chicago")
    ]


def _queue(config: dict[str, Any]) -> dict[str, str]:
    workers = int(config["decisions"]["workers"])
    wave_a = training_jobs(
        config, waves={"A"}, root=REPO / "runs/v18/train"
    )
    _status(current_phase="wave_A", workers=workers)
    results = run_jobs(wave_a, cwd=REPO, workers=workers)

    selection = _selection_jobs(config)
    _status(current_phase="selection")
    results.update(run_jobs(selection, cwd=REPO, workers=min(workers, 3)))

    wave_b = training_jobs(
        config, waves={"B"}, root=REPO / "runs/v18/train"
    )
    wave_b_names = {job.name for job in wave_b}
    evaluations: list[Job] = []
    for suite in ("E1", "E2", "E4", "E5", "E10", "E6", "E7", "E3", "T8"):
        for job in evaluation_jobs(
            config, suite=suite, root=REPO / "runs/v18/eval"
        ):
            run_name = job.name.rsplit("/", 1)[-1]
            requires = (run_name,) if run_name in wave_b_names else ()
            evaluations.append(replace(job, requires=requires))
    _status(
        current_phase="wave_B_and_evaluations", workers=workers,
        queued_jobs=len(wave_b) + len(evaluations), drop_order=list(DROP_ORDER),
    )
    results.update(run_jobs((*wave_b, *evaluations), cwd=REPO, workers=workers))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO / "configs/v18_production.yaml")
    parser.add_argument("--workers", type=int, default=4, help="P3 workers before P4 fixes W")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--skip-z", action="store_true")
    args = parser.parse_args()
    if args.plan_only:
        print(json.dumps({
            "stages": ["Z1-Z4", "P3", "P4", "freeze", "wave_A", "selection", "wave_B_and_evaluations"],
            "drop_order": DROP_ORDER, "never_drop": NEVER_DROP,
        }, indent=2))
        return 0
    if not args.skip_z:
        z = _phase_job(
            "Z_checks",
            (str(PYTHON), "-u", str(REPO / "scripts/v18_z_checks.py"), "--execute"),
            CONTROLLER / "Z_checks_report.json",
        )
        if _run_phase(z) not in {"complete", "cached"}:
            return 1
    p3_report = REPO / "runs/v18/P3_scale/report.json"
    p3 = _phase_job(
        "P3", (str(PYTHON), "-u", str(REPO / "scripts/v18_scale.py"),
               "--execute", "--workers", str(args.workers)), p3_report,
    )
    if _run_phase(p3) not in {"complete", "cached"}:
        atomic_json(p3_report, {"step": "P3", "state": "failed", "N_star": 150,
                                "fallback": "registered none-pass value; queue continues"})
    p4_report = REPO / "runs/v18/P4_timing/report.json"
    p4 = _phase_job(
        "P4", (str(PYTHON), "-u", str(REPO / "scripts/v18_timing.py"), "--execute"),
        p4_report,
    )
    if _run_phase(p4) not in {"complete", "cached"}:
        atomic_json(p4_report, {"step": "P4", "state": "failed", "N_star": int(json.loads(p3_report.read_text())["N_star"]),
            "ppo_clock_days": 1200, "mc_clock_days": 1200, "workers": 1,
            "projected_wave_a_hours": 0, "fallback": "safe registered clocks and one worker"})
    config = freeze(args.config, p3_report, p4_report)
    results = _queue(config)
    workers = int(config["decisions"]["workers"])
    for name, script in (("S", "v18_sensitivity.py"), ("E9", "v18_leftout.py"), ("E8", "v18_map.py")):
        phase = _phase_job(
            name,
            (str(PYTHON), "-u", str(REPO / "scripts" / script), "--execute", "--workers", str(workers)),
            REPO / ("runs/v18/sensitivity/report.json" if name == "S" else f"runs/v18/eval/{name}/report.json"),
        )
        results[name] = _run_phase(phase)
    counts = {
        state: sum(value == state for value in results.values())
        for state in ("complete", "cached", "failed", "skipped")
    }
    atomic_json(CONTROLLER / "queue_report.json", {
        "state": "complete", "counts": counts, "jobs": results,
    })
    _status(current_phase="queue_complete", state="complete", counts=counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
