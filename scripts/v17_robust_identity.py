#!/usr/bin/env python3
"""Required real-product identity check for the V17 robustness defaults."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from diagnostic_logging import atomic_json  # noqa: E402
from scripts import v17_results_eval as registered  # noqa: E402
from scripts.v17_config import (  # noqa: E402
    canonical_hash,
    check_model,
    load_config,
    validate_config,
)
from scripts.v17_results_train import RUN_BY_NAME  # noqa: E402
from scripts.v17_robust_common import robust_sources, set_option  # noqa: E402
from scripts.v17_robust_eval import make_job, run_robust_job  # noqa: E402


METRICS = (
    "own_profit", "rival_profit", "own_share", "rival_share", "fulfillment"
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _compare_products(
    actual: Mapping[str, Any], expected: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    differences: list[dict[str, Any]] = []
    actual_results = dict(actual.get("results", {}) or {})
    expected_results = dict(expected.get("results", {}) or {})
    if set(actual_results) != set(expected_results):
        raise ValueError(
            f"{label}: policy keys differ: "
            f"{sorted(actual_results)} != {sorted(expected_results)}"
        )
    maximum = 0.0
    compared = 0
    for spec in sorted(actual_results):
        left_rows = dict(actual_results[spec] or {})
        right_rows = dict(expected_results[spec] or {})
        if set(left_rows) != set(right_rows):
            raise ValueError(f"{label}/{spec}: rollout keys differ")
        for rollout in sorted(left_rows):
            for metric in METRICS:
                left = float(left_rows[rollout][metric])
                right = float(right_rows[rollout][metric])
                if not (math.isfinite(left) and math.isfinite(right)):
                    raise ValueError(f"{label}/{spec}/{rollout}/{metric}: non-finite")
                delta = abs(left - right)
                maximum = max(maximum, delta)
                compared += 1
                if delta > 1e-12:
                    differences.append({
                        "spec": spec,
                        "rollout": rollout,
                        "metric": metric,
                        "actual": left,
                        "expected": right,
                        "absolute_difference": delta,
                    })
    return {
        "label": label,
        "values_compared": compared,
        "maximum_absolute_difference": maximum,
        "differences": differences[:100],
        "pass": not differences,
    }


def _identity_jobs(
    config: Mapping[str, Any], output: Path, *, explicit: bool,
    comparator_root: Path | None = None,
    runs_root: Path | None = None,
):
    comparator_root = (
        BASE / "runs/v17_results_comparators"
        if comparator_root is None else comparator_root
    )
    selected_path = comparator_root / "nyc/selected.json"
    selected = _load(selected_path)
    comparator_specs = registered.comparator_specs(
        selected, "A1_shortage_severe"
    )
    comparator = make_job(
        config,
        suite="identity",
        point="explicit_defaults" if explicit else "implicit_defaults",
        city="nyc",
        case="A1_shortage_severe",
        scenario=registered.CASES["A1_shortage_severe"],
        labels=tuple(comparator_specs),
        specs=tuple(comparator_specs.values()),
        output_root=output,
        input_hashes={"selected.json": registered._sha256_file(selected_path)},
    )
    run = RUN_BY_NAME["nyc_s101"]
    runs_root = BASE / "runs/v17_results" if runs_root is None else runs_root
    run_dir = runs_root / run.name
    status_path = run_dir / "status.json"
    policy = run_dir / "policy.pt"
    rl = make_job(
        config,
        suite="identity",
        point="explicit_defaults" if explicit else "implicit_defaults",
        city="nyc",
        case="C0_calm",
        scenario=registered.CASES["C0_calm"],
        labels=(run.name,),
        specs=("rl_actor",),
        output_root=output,
        run=run,
        policy=policy,
        input_hashes={
            "policy.pt": registered._sha256_file(policy),
            "status.json": registered._sha256_file(status_path),
        },
    )
    if explicit:
        for job in (comparator, rl):
            command = set_option(job.command, "--supply_eta_v3", 0.6)
            command = set_option(command, "--memory_rival_retaliation", "on")
            object.__setattr__(job, "command", tuple(command))
    return comparator, rl


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=BASE / "runs/v17_robust/_identity"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--legacy-model", action="store_true")
    parser.add_argument(
        "--archive-root", type=Path,
        default=BASE / "runs/archive_v17_pre_correction_20261110",
    )
    args = parser.parse_args(argv)
    config = load_config()
    validate_config(config)
    output = args.output.resolve()
    if args.legacy_model:
        archive = args.archive_root.resolve()
        jobs = _identity_jobs(
            config,
            output,
            explicit=False,
            comparator_root=archive / "v17_results_comparators",
            runs_root=archive / "v17_results",
        )
        expected_model = {
            "acceptance_shock_wage": "legacy",
            "rate_upper_multiple": 2.0,
        }
        for job in jobs:
            command = set_option(
                job.command, "--acceptance_shock_wage", "legacy"
            )
            command = set_option(command, "--rate_upper_multiple", 2.0)
            object.__setattr__(job, "command", tuple(command))
        if args.plan_only:
            print(json.dumps({
                "jobs": [job.name for job in jobs],
                "commands": [list(job.command) for job in jobs],
                "references": str(archive / "v17_results_eval"),
            }, indent=2, sort_keys=True))
            return 0
        sources = robust_sources(tuple(registered.CASES.values()))
        config_hash = canonical_hash(config)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(max(1, int(args.workers)), len(jobs))
        ) as executor:
            futures = [
                executor.submit(
                    run_robust_job,
                    job,
                    config_hash=config_hash,
                    sources=sources,
                    overrides={},
                    model_expected=expected_model,
                )
                for job in jobs
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()
        expected_comparator = _load(
            archive / "v17_results_eval/_combined/nyc/A1_shortage_severe/comparators.json"
        )
        expected_rl = _load(
            archive / "v17_results_eval/nyc/C0_calm/nyc_s101.json"
        )
        actual = (_load(jobs[0].raw), _load(jobs[1].raw))
        for payload in actual:
            check_model(payload, config, expected=expected_model)
        checks = [
            _compare_products(
                actual[0], expected_comparator,
                label="legacy/nyc/A1/comparators",
            ),
            _compare_products(
                actual[1], expected_rl,
                label="legacy/nyc/C0/nyc_s101",
            ),
        ]
        passed = all(
            row["pass"] and row["maximum_absolute_difference"] <= 1e-12
            for row in checks
        )
        status = {
            "schema_version": "v17-corrected-legacy-identity-v1",
            "state": "completed" if passed else "failed",
            "pass": passed,
            "tolerance": 1e-12,
            "archive_root": str(archive),
            "model_switches": expected_model,
            "checks": checks,
        }
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(output / "status.json", status)
        if not passed:
            raise SystemExit("legacy-model identity check failed")
        print(json.dumps(status, indent=2, sort_keys=True))
        return 0

    implicit = _identity_jobs(config, output, explicit=False)
    explicit = _identity_jobs(config, output, explicit=True)
    all_jobs = (*implicit, *explicit)
    if args.plan_only:
        print(json.dumps({
            "jobs": [job.name for job in all_jobs],
            "commands": [list(job.command) for job in all_jobs],
        }, indent=2, sort_keys=True))
        return 0
    sources = robust_sources(tuple(registered.CASES.values()))
    config_hash = canonical_hash(config)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(max(1, int(args.workers)), len(all_jobs))
    ) as executor:
        futures = [
            executor.submit(
                run_robust_job,
                job,
                config_hash=config_hash,
                sources=sources,
                overrides={},
            )
            for job in all_jobs
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    expected_comparator = _load(
        BASE / "runs/v17_results_eval/_combined/nyc/A1_shortage_severe/comparators.json"
    )
    expected_rl = _load(
        BASE / "runs/v17_results_eval/nyc/C0_calm/nyc_s101.json"
    )
    checks = []
    for label, jobs in (("implicit", implicit), ("explicit", explicit)):
        checks.append(_compare_products(
            _load(jobs[0].raw), expected_comparator,
            label=f"{label}/nyc/A1/comparators",
        ))
        checks.append(_compare_products(
            _load(jobs[1].raw), expected_rl,
            label=f"{label}/nyc/C0/nyc_s101",
        ))
    status = {
        "schema_version": "v17-robust-identity-v1",
        "state": "completed" if all(row["pass"] for row in checks) else "failed",
        "pass": all(row["pass"] for row in checks),
        "tolerance": 1e-12,
        "checks": checks,
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "status.json", status)
    if not status["pass"]:
        raise SystemExit("robustness identity check failed")
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
