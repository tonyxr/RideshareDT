#!/usr/bin/env python3
"""Execute Last Revision code checks Z1--Z4 before production work."""

from __future__ import annotations

import argparse
import copy
import gzip
import json
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.v17_robust_common import set_option
from scripts.v18_config import eval_args, load_config, train_args
from scripts.v18_market import market_flags
from scripts.v18_runtime import Job, atomic_json, run_jobs
from scripts.v18_select import cell_spec, reactive_candidates
from v17_surge_decay import scenario_windows, surge_decay_spec

PYTHON = (REPO.parent / ".venv/bin/python").absolute()
ROOT = REPO / "runs/v18/controller/Z_checks"
PUBLISHED = cell_spec(1.0, 1.0)


def frozen(n: int = 100) -> dict:
    value = copy.deepcopy(load_config())
    value["status"] = "frozen"
    value["decisions"].update({"N_star": n, "ppo_clock_days": 1200,
        "mc_clock_days": 1200, "workers": 4,
        "projection": {"wave_a_hours": 0, "source": "Z3"}})
    return value


def z1() -> dict:
    started = time.monotonic()
    process = subprocess.run((str(PYTHON), "-m", "pytest", "-q", "tests/test_v18.py",
        "tests/test_v18_final.py"), cwd=REPO, text=True, capture_output=True)
    return {"state": "pass" if process.returncode == 0 else "fail", "returncode": process.returncode,
            "seconds": time.monotonic() - started, "output": process.stdout + process.stderr}


def z2() -> dict:
    source = REPO / "runs/v18/preflight/P2_fix/effects/v17_model/C0_calm/adaptive_best_response_memory/manifest.json"
    command = json.loads(source.read_text())["identity"]["command"]
    values, _ = market_flags(command[3:], variant="production", city="nyc", customers=500)
    leaf = ROOT / "Z2"
    raw = leaf / "raw.json"
    values = set_option(values, "--policy_sweep_out", raw)
    values = set_option(values, "--report_prefix", leaf / "diagnostics")
    job = Job("Z2", (str(PYTHON), "-u", str(REPO / "Core.py"), *values), raw,
              leaf / "run.log", leaf / "manifest.json", tier=0)
    state = run_jobs((job,), cwd=REPO, workers=1)["Z2"]
    objective = None
    if state in {"complete", "cached"}:
        product = json.loads(raw.read_text())
        objective = float(product["results"][PUBLISHED]["adaptive_best_response_memory__opening_1"]["objective"])
    expected = 0.3187048000915823
    return {"state": "pass" if objective == expected else "fail", "job_state": state,
            "objective": objective, "expected": expected, "source": str(source)}


def _reference_specs() -> tuple[str, ...]:
    shortage = json.loads((REPO / "scenarios/v17_t2_shortage_severe.json").read_text())
    dated = surge_decay_spec((0.8, 2.0), (1.5, 1.5), scenario_windows(shortage), 0, 0)
    count = "count_rule:" + json.dumps({"normal": [0.8, 2.0], "event": [1.5, 1.5],
        "norm": [50.0] * 6, "theta": 0.7, "mode": "step", "theta_lo": 0.5,
        "move_limit": 1}, sort_keys=True, separators=(",", ":"))
    return PUBLISHED, dated, count, reactive_candidates((0.8, 2.0))[0]


def z3(workers: int) -> dict:
    config = frozen(100); config["learner"]["policy_snapshot_every_days"] = 10
    arms = ("main", "ppo", "mc_clock", "nocrn", "no_driver_obs", "own_profit")
    train: list[Job] = []
    for arm in arms:
        leaf = ROOT / "Z3/train" / arm
        values = train_args(config, city="nyc", seed=101, arm=arm, output=leaf,
                            days_override=20, require_frozen=True)
        train.append(Job(f"Z3/train/{arm}", (str(PYTHON), "-u", str(REPO / "Core.py"), *values),
                         leaf / "policy.pt", leaf / "run.log", leaf / "manifest.json", tier=0))
    states = run_jobs(train, cwd=REPO, workers=workers)
    eval_jobs: list[Job] = []
    for arm in arms:
        leaf = ROOT / "Z3/eval" / arm
        values = eval_args(config, city="nyc", case="A1", specs=("rl_actor",), suite="Z3",
            output=leaf, scenario=REPO / "scenarios/v17_t2_shortage_severe.json", kind=arm,
            policy=ROOT / "Z3/train" / arm / "policy.pt")
        values = set_option(values, "--policy_sweep_rivals", "adaptive_best_response_memory")
        values = set_option(values, "--policy_sweep_openings", 1)
        values = set_option(values, "--policy_sweep_horizon", 30)
        eval_jobs.append(Job(f"Z3/eval/{arm}", (str(PYTHON), "-u", str(REPO / "Core.py"), *values),
            leaf / "raw.json", leaf / "run.log", leaf / "manifest.json", requires=(f"Z3/train/{arm}",), tier=0))
    snapshot_leaf = ROOT / "Z3/eval/main_snapshot_day10"
    snapshot_values = eval_args(config, city="nyc", case="A1", specs=("rl_actor",), suite="Z3",
        output=snapshot_leaf, scenario=REPO / "scenarios/v17_t2_shortage_severe.json", kind="main",
        policy=ROOT / "Z3/train/main/snapshots/policy.day10.pt")
    snapshot_values = set_option(snapshot_values, "--policy_sweep_rivals", "adaptive_best_response_memory")
    snapshot_values = set_option(snapshot_values, "--policy_sweep_openings", 1)
    snapshot_values = set_option(snapshot_values, "--policy_sweep_horizon", 30)
    eval_jobs.append(Job("Z3/eval/main_snapshot_day10",
        (str(PYTHON), "-u", str(REPO / "Core.py"), *snapshot_values), snapshot_leaf / "raw.json",
        snapshot_leaf / "run.log", snapshot_leaf / "manifest.json", tier=0))
    for index, spec in enumerate(_reference_specs()):
        leaf = ROOT / "Z3/reference" / str(index)
        values = eval_args(config, city="nyc", case="A1", specs=(spec,), suite="Z3",
            output=leaf, scenario=REPO / "scenarios/v17_t2_shortage_severe.json")
        values = set_option(values, "--policy_sweep_rivals", "adaptive_best_response_memory")
        values = set_option(values, "--policy_sweep_openings", 1)
        values = set_option(values, "--policy_sweep_horizon", 30)
        eval_jobs.append(Job(f"Z3/reference/{index}", (str(PYTHON), "-u", str(REPO / "Core.py"), *values),
            leaf / "raw.json", leaf / "run.log", leaf / "manifest.json", tier=0))
    # Training dependencies completed in the preceding batch, so remove those
    # external names before submitting the evaluation batch.
    eval_jobs = [Job(j.name, j.command, j.output, j.log, j.manifest, (), j.tier) for j in eval_jobs]
    eval_states = run_jobs(eval_jobs, cwd=REPO, workers=workers)
    states.update(eval_states)
    passed = all(state in {"complete", "cached"} for state in states.values())
    return {"state": "pass" if passed else "fail", "jobs": states}


def _logged_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    for file in sorted(path.rglob("*.jsonl.gz")):
        with gzip.open(file, "rt", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def z4(workers: int) -> dict:
    config = frozen(100)
    jobs: list[Job] = []
    for logged in (False, True):
        leaf = ROOT / "Z4" / ("logged" if logged else "plain")
        values = eval_args(config, city="nyc", case="A3", specs=(PUBLISHED,),
            suite="E10" if logged else "Z4", output=leaf,
            scenario=REPO / "scenarios/v17_t2_storm_transit_late.json")
        values = set_option(values, "--policy_sweep_rivals", "adaptive_best_response_memory")
        values = set_option(values, "--policy_sweep_openings", 1)
        jobs.append(Job(f"Z4/{'logged' if logged else 'plain'}",
            (str(PYTHON), "-u", str(REPO / "Core.py"), *values), leaf / "raw.json",
            leaf / "run.log", leaf / "manifest.json", tier=0))
    states = run_jobs(jobs, cwd=REPO, workers=min(2, workers))
    comparison = {"results_match": False, "traces_match": False,
                  "daily_trips_match": False, "daily_fare_match": False}
    if all(state in {"complete", "cached"} for state in states.values()):
        plain = json.loads((ROOT / "Z4/plain/raw.json").read_text())
        logged = json.loads((ROOT / "Z4/logged/raw.json").read_text())
        comparison["results_match"] = plain.get("results") == logged.get("results")
        comparison["traces_match"] = plain.get("traces") == logged.get("traces")
        rows = _logged_rows(ROOT / "Z4/logged/requests")
        comparison["logged_rows"] = len(rows)
        # The two exact product comparisons are the authoritative identity;
        # these log-side indicators explicitly record field availability.
        comparison["daily_trips_match"] = bool(rows) and all("RequestStep" in row for row in rows)
        comparison["daily_fare_match"] = bool(rows) and all("ReferenceFare" in row for row in rows)
    return {"state": "report", "jobs": states, **comparison}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"checks": ["Z1", "Z2", "Z3", "Z4"]}, indent=2)); return 0
    report = {"Z1": z1()}
    if report["Z1"]["state"] == "pass": report["Z2"] = z2()
    if report.get("Z2", {}).get("state") == "pass": report["Z3"] = z3(args.workers)
    if report.get("Z3", {}).get("state") == "pass": report["Z4"] = z4(args.workers)
    report["state"] = "pass" if all(report.get(key, {}).get("state") == "pass" for key in ("Z1", "Z2", "Z3")) else "fail"
    atomic_json(REPO / "runs/v18/controller/Z_checks_report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["state"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
