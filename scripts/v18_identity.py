#!/usr/bin/env python3
"""Exact P1 identity gate for every V18 flag at its legacy default."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence


BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from diagnostic_logging import atomic_json  # noqa: E402
from scripts import v17_results_eval as registered  # noqa: E402
from scripts.v17_config import canonical_hash, load_config, validate_config  # noqa: E402
from scripts.v17_results_train import RUN_BY_NAME  # noqa: E402
from scripts.v17_robust_common import robust_sources  # noqa: E402
from scripts.v17_robust_eval import make_job, run_robust_job  # noqa: E402


RIVAL = "adaptive_best_response_memory"
CASES = ("C0_calm", "A1_shortage_severe")
PUBLISHED = registered.cell_spec(1.0, 1.0)
DEFAULT_SWITCHES = {
    "v3_daily_stock": False,
    "supply_eta_within": None,
    "active_driver_feature_reference": "fixed",
    "driver_pay_rule": "share",
    "scenario_hazards": "natural",
    "recalibrate_supply_on_eval": False,
}
DEFAULT_V26R = {
    "v17_credit_source": "branches",
    "v17_branch_noise": "crn",
    "v17_eval_credit_probe": 0,
    "policy_snapshot_every_days": 0,
    "policy_sweep_request_log": False,
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return registered._sha256_file(path)


def _jobs(config: Mapping[str, Any], output: Path):
    run = RUN_BY_NAME["nyc_s101"]
    policy = BASE / "runs/v17_results/nyc_s101/policy.pt"
    status = BASE / "runs/v17_results/nyc_s101/status.json"
    jobs = []
    for case in CASES:
        scenario = registered.CASES[case]
        jobs.append(make_job(
            config,
            suite="v18_p1_identity",
            point="default_flags",
            city="nyc",
            case=case,
            scenario=scenario,
            labels=("published_card",),
            specs=(PUBLISHED,),
            output_root=output,
            rivals=(RIVAL,),
            openings=1,
            input_hashes={
                "reference": _sha256(
                    BASE / f"runs/v17_results_eval/nyc/{case}/published_card.json"
                ),
            },
        ))
        jobs.append(make_job(
            config,
            suite="v18_p1_identity",
            point="default_flags",
            city="nyc",
            case=case,
            scenario=scenario,
            labels=("nyc_s101",),
            specs=("rl_actor",),
            output_root=output,
            run=run,
            policy=policy,
            rivals=(RIVAL,),
            openings=1,
            input_hashes={
                "policy.pt": _sha256(policy),
                "status.json": _sha256(status),
                "reference": _sha256(
                    BASE / f"runs/v17_results_eval/nyc/{case}/nyc_s101.json"
                ),
            },
        ))
    return tuple(jobs)


def _first_difference(left: Any, right: Any, path: str = "$") -> dict[str, Any] | None:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            return {
                "path": path,
                "actual_keys": sorted(map(str, left)),
                "expected_keys": sorted(map(str, right)),
            }
        for key in sorted(left):
            found = _first_difference(left[key], right[key], f"{path}.{key}")
            if found is not None:
                return found
        return None
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return {"path": path, "actual_length": len(left), "expected_length": len(right)}
        for index, (actual, expected) in enumerate(zip(left, right)):
            found = _first_difference(actual, expected, f"{path}[{index}]")
            if found is not None:
                return found
        return None
    if left != right:
        difference = None
        if (
            isinstance(left, (int, float))
            and not isinstance(left, bool)
            and isinstance(right, (int, float))
            and not isinstance(right, bool)
            and math.isfinite(float(left))
            and math.isfinite(float(right))
        ):
            difference = abs(float(left) - float(right))
        return {
            "path": path,
            "actual": left,
            "expected": right,
            "absolute_difference": difference,
        }
    return None


def _check_defaults(payload: Mapping[str, Any], label: str) -> None:
    switches = dict(payload.get("model_switches") or {})
    v26r = dict(payload.get("v26r") or {})
    for key, expected in DEFAULT_SWITCHES.items():
        if switches.get(key) != expected:
            raise ValueError(
                f"{label}: default switch {key}={switches.get(key)!r}, expected {expected!r}"
            )
    for key, expected in DEFAULT_V26R.items():
        if v26r.get(key) != expected:
            raise ValueError(
                f"{label}: default v26r switch {key}={v26r.get(key)!r}, expected {expected!r}"
            )


def _compare(job) -> dict[str, Any]:
    actual = _load(job.outputs[0])
    _check_defaults(actual, job.name)
    case = job.case
    is_rl = job.kind == "rl"
    reference = _load(
        BASE / f"runs/v17_results_eval/nyc/{case}"
        / ("nyc_s101.json" if is_rl else "published_card.json")
    )
    spec = "rl_actor" if is_rl else PUBLISHED
    rollout = f"{RIVAL}__opening_1"
    expected = {
        "traces": [
            row for row in reference["traces"][spec]
            if row.get("rollout") == rollout
        ],
        "results": {rollout: reference["results"][spec][rollout]},
    }
    observed = {
        "traces": list(actual["traces"][spec]),
        "results": dict(actual["results"][spec]),
    }
    mismatch = _first_difference(observed, expected)
    return {
        "job": job.name,
        "case": case,
        "policy": "nyc_s101" if is_rl else "published_card",
        "rival": RIVAL,
        "openings": 1,
        "horizon_days": 40,
        "trace_rows": len(observed["traces"]),
        "maximum_absolute_difference": 0.0 if mismatch is None else mismatch.get(
            "absolute_difference"
        ),
        "first_difference": mismatch,
        "pass": mismatch is None,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=BASE / "runs/v18/preflight/P1/identity",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    config = load_config()
    validate_config(config)
    output = args.output.resolve()
    jobs = _jobs(config, output)
    plan = {
        "schema_version": "v18-p1-identity-v1",
        "state": "planned",
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=BASE, text=True
        ).strip(),
        "jobs": len(jobs),
        "workers": min(max(1, int(args.workers)), len(jobs)),
        "city": "nyc",
        "cases": list(CASES),
        "policies": ["published_card", "nyc_s101"],
        "rivals": [RIVAL],
        "openings": 1,
        "horizon_days": 40,
        "commands": [list(job.command) for job in jobs],
    }
    if args.plan_only:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "status.json"
    atomic_json(status_path, {
        **plan,
        "state": "running",
        "pass": False,
        "pid": os.getpid(),
    })
    sources = robust_sources(tuple(registered.CASES.values()))
    config_hash = canonical_hash(config)
    completed = 0
    try:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=plan["workers"]
        ) as executor:
            futures = {
                executor.submit(
                    run_robust_job,
                    job,
                    config_hash=config_hash,
                    sources=sources,
                    overrides={},
                ): job
                for job in jobs
            }
            for future in concurrent.futures.as_completed(futures):
                future.result()
                completed += 1
                atomic_json(status_path, {
                    **plan,
                    "state": "running",
                    "pass": False,
                    "pid": os.getpid(),
                    "completed_jobs": completed,
                    "last_completed": futures[future].name,
                })
        checks = [_compare(job) for job in jobs]
        passed = all(check["pass"] for check in checks)
        status = {
            **plan,
            "state": "completed" if passed else "failed",
            "pass": passed,
            "pid": os.getpid(),
            "completed_jobs": len(jobs),
            "tolerance": 0.0,
            "checks": checks,
        }
        atomic_json(status_path, status)
        if not passed:
            raise SystemExit("V18 P1 identity mismatch")
        print(json.dumps(status, indent=2, sort_keys=True))
        return 0
    except BaseException as exc:
        current = _load(status_path) if status_path.is_file() else plan
        if current.get("state") != "failed":
            atomic_json(status_path, {
                **current,
                "state": "failed",
                "pass": False,
                "completed_jobs": completed,
                "error": repr(exc),
            })
        raise


if __name__ == "__main__":
    raise SystemExit(main())
