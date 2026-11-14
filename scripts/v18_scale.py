#!/usr/bin/env python3
"""Last Revision P3 scale check and N* selection."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import v17_t2_sweep as t2
from scripts.v17_config import load_config as load_v17_config
from scripts.v17_robust_common import set_option
from scripts.v18_config import load_config, validate_config
from scripts.v18_market import market_flags
from scripts.v18_runtime import Job, atomic_json, run_jobs
from v17_surge_decay import scenario_windows, surge_decay_spec


PYTHON = (REPO.parent / ".venv/bin/python").absolute()
SIZES = (50, 100, 150, 250, 500)
CITIES = ("nyc", "chicago")
CASES = {
    "C0": REPO / "scenarios/calm.json",
    "A1": REPO / "scenarios/v17_t2_shortage_severe.json",
}
RIVALS = (
    "adaptive_best_response_memory",
    "profit_best_response_myopic",
    "profit_opt_mpc_grid",
)


def policy_specs() -> dict[str, str]:
    normal = (0.8, 2.0)
    windows = scenario_windows(json.loads(CASES["A1"].read_text(encoding="utf-8")))
    return {
        "published": t2.cell_spec(1.0, 1.0),
        "v17_best_fixed": t2.cell_spec(*normal),
        "fixed_1.25": t2.cell_spec(1.25, 1.25),
        "fixed_1.5": t2.cell_spec(1.5, 1.5),
        "fixed_2.0": t2.cell_spec(2.0, 2.0),
        "fixed_2.5": t2.cell_spec(2.5, 2.5),
        "dated_1.5": surge_decay_spec(normal, (1.5, 1.5), windows, 0, 0),
        "dated_2.0": surge_decay_spec(normal, (2.0, 2.0), windows, 0, 0),
    }


def _command(*, city: str, case: str, customers: int, output: Path) -> list[str]:
    values = t2.sweep_command(
        load_v17_config(), city=city, case="C0_calm", rival=RIVALS[0],
        output=output.parent, openings=3, customers=customers,
        pool=20_000, days=40,
    )
    values, _ = market_flags(
        values, variant="production", city=city, customers=customers,
    )
    for flag, value in (
        ("--scenario_hazards", "off"),
        ("--seed", 20_231), ("--experiment_seed", 20_231),
        ("--policy_sweep_seed", 20_231),
        ("--shock_scenario", CASES[case].resolve()),
        ("--policy_sweep", ";".join(policy_specs().values())),
        ("--policy_sweep_rivals", ",".join(RIVALS)),
        ("--policy_sweep_openings", 3),
        ("--policy_sweep_horizon", 40 * 6),
        ("--policy_sweep_out", output.resolve()),
        ("--report_prefix", output.parent / "diagnostics"),
    ):
        values = set_option(values, flag, value)
    return values


def jobs(root: Path) -> list[Job]:
    result: list[Job] = []
    for city in CITIES:
        for case in CASES:
            for customers in SIZES:
                leaf = root / city / case / f"N{customers}"
                raw = leaf / "raw.json"
                result.append(Job(
                    name=f"P3/{city}/{case}/N{customers}",
                    command=tuple(_command(
                        city=city, case=case, customers=customers, output=raw,
                    )),
                    output=raw, log=leaf / "run.log",
                    manifest=leaf / "manifest.json", tier=0,
                ))
    return result


def _mean_objectives(payload: Mapping[str, Any]) -> dict[str, float]:
    names = {spec: name for name, spec in policy_specs().items()}
    return {
        names[spec]: sum(float(row["objective"]) for row in rows.values()) / len(rows)
        for spec, rows in payload["results"].items()
    }


def _kendall(scores: Mapping[str, float], reference: Mapping[str, float]) -> float:
    names = tuple(sorted(scores))
    concordant = discordant = 0
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            a = math.copysign(1.0, scores[left] - scores[right]) if scores[left] != scores[right] else 0.0
            b = math.copysign(1.0, reference[left] - reference[right]) if reference[left] != reference[right] else 0.0
            if not a or not b:
                continue
            if a == b:
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    return 1.0 if total == 0 else (concordant - discordant) / total


def _g(scores: Mapping[str, float]) -> float:
    return max(scores["dated_1.5"], scores["dated_2.0"]) - scores["v17_best_fixed"]


def _paired_driver_report(payload: Mapping[str, Any]) -> dict[str, float]:
    specs = policy_specs()
    pub = {
        (str(row["rollout"]), int(row["step"])): float(row["active_drivers_firm1"])
        for row in payload["traces"][specs["published"]]
        if 10 <= int(row["step"]) // 6 + 1 <= 15
    }
    diffs = [
        float(row["active_drivers_firm1"]) - pub[(str(row["rollout"]), int(row["step"]))]
        for row in payload["traces"][specs["fixed_1.5"]]
        if (str(row["rollout"]), int(row["step"])) in pub
    ]
    mean = sum(diffs) / len(diffs)
    variance = sum((value - mean) ** 2 for value in diffs) / max(1, len(diffs) - 1)
    se = math.sqrt(variance / len(diffs))
    return {"mean": mean, "t": mean / se if se else math.inf, "pairs": len(diffs)}


def analyze(root: Path) -> dict[str, Any]:
    scores: dict[str, dict[str, dict[int, dict[str, float]]]] = {}
    for city in CITIES:
        scores[city] = {}
        for case in CASES:
            scores[city][case] = {}
            for customers in SIZES:
                payload = json.loads((root / city / case / f"N{customers}/raw.json").read_text(encoding="utf-8"))
                scores[city][case][customers] = _mean_objectives(payload)
    diagnostics: dict[str, Any] = {}
    selected = None
    for customers in (100, 150, 250):
        rows = {}
        passes = True
        for city in CITIES:
            rows[city] = {}
            for case in CASES:
                tau = _kendall(scores[city][case][customers], scores[city][case][500])
                rows[city][case] = {"kendall_tau": tau}
                passes &= tau >= 0.8
            gap = abs(_g(scores[city]["A1"][customers]) - _g(scores[city]["A1"][500]))
            rows[city]["G_gap"] = gap
            passes &= gap <= 0.05
        diagnostics[str(customers)] = {"cities": rows, "pass": bool(passes)}
        if passes and selected is None:
            selected = customers
    if selected is None:
        selected = 150
    s1 = {}
    for city in CITIES:
        payload = json.loads((root / city / "A1" / f"N{selected}/raw.json").read_text(encoding="utf-8"))
        s1[city] = _paired_driver_report(payload)
    return {
        "step": "P3", "state": "complete", "N_star": selected,
        "selection_rule": diagnostics, "S-1_report_only": s1,
        "scores": scores,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO / "configs/v18_production.yaml")
    parser.add_argument("--root", type=Path, default=REPO / "runs/v18/P3_scale")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    validate_config(config)
    planned = jobs(args.root)
    atomic_json(args.root / "plan.json", {
        "jobs": [{"name": job.name, "command": list(job.command)} for job in planned],
        "count": len(planned),
    })
    if not args.execute:
        print(json.dumps({"jobs": len(planned)}, indent=2))
        return 0
    states = run_jobs(planned, cwd=REPO, workers=args.workers)
    if any(state not in {"complete", "cached"} for state in states.values()):
        raise RuntimeError("P3 has failed jobs; inspect manifests")
    report = analyze(args.root)
    atomic_json(args.root / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
