#!/usr/bin/env python3
"""Last Revision P4 timing, memory, worker, and clock calibration."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.v18_config import load_config, train_args, validate_config
from scripts.v18_runtime import Job, atomic_json, run_jobs


PYTHON = (REPO.parent / ".venv/bin/python").absolute()
SPECS = (
    ("agent_nyc_s101", "nyc", 101, "main"),
    ("agent_nyc_s202", "nyc", 202, "main"),
    ("agent_nyc_s303", "nyc", 303, "main"),
    ("agent_chicago_s101", "chicago", 101, "main"),
    ("agent_boston_s101", "boston", 101, "main"),
    ("ppo_nyc_s101", "nyc", 101, "ppo_clock"),
    ("ppo_nyc_s202", "nyc", 202, "ppo_clock"),
    ("mc_nyc_s101", "nyc", 101, "mc_clock"),
    ("mc_nyc_s202", "nyc", 202, "mc_clock"),
)


def _timing_config(config: dict[str, Any], n_star: int) -> dict[str, Any]:
    value = copy.deepcopy(config)
    value["decisions"]["N_star"] = int(n_star)
    return value


def jobs(config: dict[str, Any], *, n_star: int, root: Path) -> list[Job]:
    timing_config = _timing_config(config, n_star)
    result: list[Job] = []
    for name, city, seed, arm in SPECS:
        leaf = (root / f"N{n_star}" / name).resolve()
        command = (
            "/usr/bin/time", "-l", str(PYTHON), "-u", str(REPO / "Core.py"),
            *train_args(
                timing_config, city=city, seed=seed, arm=arm, output=leaf,
                days_override=60, require_frozen=False,
            ),
        )
        result.append(Job(
            name=f"P4/N{n_star}/{name}", command=tuple(command),
            output=leaf / "policy.pt", log=leaf / "run.log",
            manifest=leaf / "manifest.json", tier=0,
        ))
    return result


def _metrics(log: Path) -> tuple[float, int]:
    text = log.read_text(encoding="utf-8", errors="replace")
    wall_patterns = (
        r"(?m)^\s*([0-9.]+)\s+real\b",
        r"(?m)^\s*real\s+([0-9.]+)\b",
    )
    wall = None
    for pattern in wall_patterns:
        match = re.search(pattern, text)
        if match:
            wall = float(match.group(1))
    memory_matches = re.findall(
        r"(?m)^\s*([0-9]+)\s+maximum resident set size\b", text
    )
    if wall is None or not memory_matches:
        raise ValueError(f"cannot parse /usr/bin/time -l metrics from {log}")
    return wall, max(map(int, memory_matches))


def _round_clock(value: float) -> int:
    return max(1200, int(math.ceil(value / 30.0) * 30))


def _physical_resources() -> tuple[int, int]:
    try:
        cores = int(subprocess.check_output(
            ["sysctl", "-n", "hw.physicalcpu"], text=True
        ).strip())
    except Exception:
        cores = max(1, (os.cpu_count() or 2) // 2)
    try:
        memory = int(subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"], text=True
        ).strip())
    except Exception:
        memory = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    return cores, memory


def analyze(planned: list[Job], *, n_star: int) -> dict[str, Any]:
    rows = {}
    for job in planned:
        wall, peak = _metrics(job.log)
        rows[job.name] = {"wall_seconds": wall, "peak_memory_bytes": peak}
    groups = {
        "agent": [row for name, row in rows.items() if "/agent_" in name],
        "ppo": [row for name, row in rows.items() if "/ppo_" in name],
        "mc": [row for name, row in rows.items() if "/mc_" in name],
    }
    medians = {
        key: statistics.median(row["wall_seconds"] for row in group)
        for key, group in groups.items()
    }
    largest = max(row["peak_memory_bytes"] for row in rows.values())
    cores, memory = _physical_resources()
    workers = min(10, cores, max(1, math.floor(0.8 * memory / largest)))
    ppo_days = _round_clock(1200 * medians["agent"] / medians["ppo"])
    mc_days = _round_clock(1200 * medians["agent"] / medians["mc"])
    wave_a_hours = 20 * medians["agent"] / 3600.0 * max(1.9 / workers, 1.0)
    return {
        "step": "P4", "state": "complete", "N_star": int(n_star),
        "jobs": rows, "median_wall_seconds": medians,
        "largest_peak_memory_bytes": largest,
        "physical_cores": cores, "physical_memory_bytes": memory,
        "workers": workers, "ppo_clock_days": ppo_days,
        "mc_clock_days": mc_days,
        "projected_wave_a_hours": wave_a_hours,
        "requires_N_step_down": bool(wave_a_hours > 30 and n_star > 100),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO / "configs/v18_production.yaml")
    parser.add_argument("--p3", type=Path, default=REPO / "runs/v18/P3_scale/report.json")
    parser.add_argument("--root", type=Path, default=REPO / "runs/v18/P4_timing")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    validate_config(config)
    p3 = json.loads(args.p3.read_text(encoding="utf-8"))
    n_star = int(p3["N_star"])
    all_reports = []
    while True:
        planned = jobs(config, n_star=n_star, root=args.root)
        atomic_json(args.root / f"N{n_star}/plan.json", {
            "N_star": n_star,
            "jobs": [{"name": job.name, "command": list(job.command)} for job in planned],
        })
        if not args.execute:
            print(json.dumps({"N_star": n_star, "jobs": len(planned)}, indent=2))
            return 0
        states = run_jobs(planned, cwd=REPO, workers=9)
        if any(state not in {"complete", "cached"} for state in states.values()):
            raise RuntimeError("P4 has failed jobs; inspect manifests")
        report = analyze(planned, n_star=n_star)
        atomic_json(args.root / f"N{n_star}/report.json", report)
        all_reports.append(report)
        if not report["requires_N_step_down"]:
            break
        lower = [value for value in (100, 150, 250) if value < n_star]
        n_star = max(lower) if lower else 100
        if len(all_reports) >= 2:
            break
    final = dict(all_reports[-1])
    final["runs"] = all_reports
    if len(all_reports) > 1:
        final["N_step_down_reason"] = "projected Wave A wall time exceeded 30 hours"
    atomic_json(args.root / "report.json", final)
    print(json.dumps(final, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
