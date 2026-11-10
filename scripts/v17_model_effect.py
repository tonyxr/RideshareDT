#!/usr/bin/env python3
"""Preflight table isolating the two corrected V17 model switches."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
from pathlib import Path
from statistics import mean
import sys
from typing import Any, Mapping, Sequence


BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from diagnostic_logging import atomic_json  # noqa: E402
from scripts import v17_t2_sweep as t2  # noqa: E402
from scripts.v17_config import canonical_hash, load_config, validate_config  # noqa: E402
from scripts.v17_robust_common import set_option  # noqa: E402


RIVAL = "adaptive_best_response_memory"
CASES = ("C0_calm", "A1_shortage_severe")
CELLS = ((1.0, 1.0), (0.8, 2.0), (2.0, 2.0), (2.5, 2.5))
SPECS = tuple(t2.cell_spec(*cell) for cell in CELLS)
MODEL_ARMS = {
    "legacy": {
        "acceptance_shock_wage": "legacy",
        "rate_upper_multiple": 2.0,
    },
    "corrected": {
        "acceptance_shock_wage": "declared",
        "rate_upper_multiple": 2.5,
    },
}


def _paths(output: Path, arm: str, case: str) -> tuple[Path, Path, Path]:
    return (
        output / "raw" / arm / f"{case}.json",
        output / "manifests" / arm / f"{case}.json",
        output / "logs" / arm / f"{case}.log",
    )


def effect_command(
    config: Mapping[str, Any], *, city: str, arm: str, case: str,
    output: Path, openings: int = 3, customers: int = 50, days: int = 28,
) -> list[str]:
    raw, _, _ = _paths(output, arm, case)
    command = t2.sweep_command(
        config,
        case=case,
        rival=RIVAL,
        output=output,
        openings=openings,
        customers=customers,
        pool=2000,
        city=city,
        days=days,
    )
    command = set_option(command, "--policy_sweep", ";".join(SPECS))
    command = set_option(command, "--policy_sweep_out", raw.resolve())
    switches = MODEL_ARMS[arm]
    command = set_option(
        command, "--acceptance_shock_wage",
        switches["acceptance_shock_wage"],
    )
    command = set_option(
        command, "--rate_upper_multiple",
        switches["rate_upper_multiple"],
    )
    command = set_option(
        command, "--report_prefix",
        output / "diagnostics" / arm / case,
    )
    return command


def _summarize(
    payload: Mapping[str, Any], *, arm: str, case: str,
) -> list[dict[str, Any]]:
    rows = []
    for cell, spec in zip(CELLS, SPECS):
        results = tuple(dict(payload["results"][spec]).values())
        traces = [
            row for row in payload["traces"][spec]
            if 10 <= int(row["step"]) // t2.STEPS_PER_DAY <= 15
        ]
        if not results or not traces:
            raise ValueError(f"effect product has incomplete coverage: {arm}/{case}/{cell}")
        row = {
            "arm": arm,
            "case": case,
            "cell": f"{cell[0]:g},{cell[1]:g}",
            "objective": mean(float(item["objective"]) for item in results),
            "own_profit": mean(float(item["own_profit"]) for item in results),
            "rival_profit": mean(float(item["rival_profit"]) for item in results),
            "fulfillment_days_10_15": mean(float(item["fulfillment"]) for item in traces),
            "active_drivers_firm1_days_10_15": mean(
                float(item["active_drivers_firm1"]) for item in traces
            ),
            "firm1_level_days_10_15": mean(float(item["firm1_level"]) for item in traces),
        }
        if not all(
            math.isfinite(float(value))
            for key, value in row.items()
            if key not in {"arm", "case", "cell"}
        ):
            raise ValueError(f"effect table contains non-finite values: {row}")
        rows.append(row)
    return rows


def _maximum_difference(
    rows: Sequence[Mapping[str, Any]], *, case: str, cells: Sequence[str],
) -> float:
    by_key = {(str(row["arm"]), str(row["case"]), str(row["cell"])): row for row in rows}
    maximum = 0.0
    for cell in cells:
        legacy = by_key[("legacy", case, cell)]
        corrected = by_key[("corrected", case, cell)]
        for key in legacy:
            if key not in {"arm", "case", "cell"}:
                maximum = max(maximum, abs(float(legacy[key]) - float(corrected[key])))
    return maximum


def _manual_gate(checks: Mapping[str, float]) -> dict[str, Any]:
    """Separate the manual's stop condition from reported expectations.

    Section 4.6 says to stop only when corrected published-fare service
    does not exceed legacy service.  Calm identity and cap behavior remain
    useful diagnostics, but section 12 says other scientific outcomes are
    results rather than workflow gates.
    """
    stop_conditions = {
        "corrected_published_service_improves": (
            float(checks["corrected_published_service"])
            > float(checks["legacy_published_service"])
        ),
    }
    diagnostic_expectations = {
        "calm_first_three_identical": (
            float(checks["calm_first_three_maximum_difference"]) <= 1e-12
        ),
        "legacy_2_5_equals_2_0": (
            float(checks["legacy_2_5_vs_2_0_maximum_difference"]) <= 1e-12
        ),
        "corrected_2_5_differs_from_2_0": (
            float(checks["corrected_2_5_vs_2_0_maximum_difference"]) > 1e-12
        ),
    }
    return {
        "pass": all(stop_conditions.values()),
        "stop_conditions": stop_conditions,
        "diagnostic_expectations": diagnostic_expectations,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", choices=t2.CITY_SLUGS, default="nyc")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--output", type=Path,
        default=BASE / "runs/v17_preflight/model_effect",
    )
    parser.add_argument("--openings", type=int, default=3)
    parser.add_argument("--customers", type=int, default=50)
    parser.add_argument("--days", type=int, default=28)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    config = load_config()
    validate_config(config)
    output = args.output.resolve()
    jobs = []
    for arm in MODEL_ARMS:
        for case in CASES:
            raw, manifest, log = _paths(output, arm, case)
            command = effect_command(
                config,
                city=args.city,
                arm=arm,
                case=case,
                output=output,
                openings=int(args.openings),
                customers=int(args.customers),
                days=int(args.days),
            )
            jobs.append((arm, case, command, raw, manifest, log))
    if args.plan_only:
        print(json.dumps({
            "jobs": [f"{arm}/{case}" for arm, case, *_ in jobs],
            "commands": [command for _, _, command, *_ in jobs],
        }, indent=2, sort_keys=True))
        return 0

    sources = dict(t2.source_identity())
    sources["scripts/v17_model_effect.py"] = t2._sha256_file(Path(__file__))
    config_hash = canonical_hash(config)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(max(1, int(args.workers)), len(jobs))
    ) as executor:
        futures = {
            executor.submit(
                t2.run_job,
                command,
                raw=raw,
                manifest=manifest,
                log=log,
                config_hash=config_hash,
                sources=sources,
                force=bool(args.force),
                model_expected=MODEL_ARMS[arm],
            ): (arm, case)
            for arm, case, command, raw, manifest, log in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            future.result()

    rows = []
    for arm, case, _, raw, _, _ in jobs:
        rows.extend(_summarize(json.loads(raw.read_text()), arm=arm, case=case))
    output.mkdir(parents=True, exist_ok=True)
    fieldnames = tuple(rows[0])
    with (output / "model_effect.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    first_three = tuple(f"{d:g},{t:g}" for d, t in CELLS[:3])
    calm_maximum = _maximum_difference(rows, case="C0_calm", cells=first_three)
    by_key = {(row["arm"], row["case"], row["cell"]): row for row in rows}
    published = "0.8,2"
    corrected_service = float(
        by_key[("corrected", "A1_shortage_severe", published)]["fulfillment_days_10_15"]
    )
    legacy_service = float(
        by_key[("legacy", "A1_shortage_severe", published)]["fulfillment_days_10_15"]
    )
    legacy_cap_difference = _maximum_difference(
        rows, case="A1_shortage_severe", cells=("2,2", "2.5,2.5")
    )
    # The helper above compares arms, so compare the two cap cells explicitly.
    legacy_cap_difference = max(
        abs(float(by_key[("legacy", "A1_shortage_severe", "2,2")][key])
            - float(by_key[("legacy", "A1_shortage_severe", "2.5,2.5")][key]))
        for key in fieldnames if key not in {"arm", "case", "cell"}
    )
    corrected_cap_difference = max(
        abs(float(by_key[("corrected", "A1_shortage_severe", "2,2")][key])
            - float(by_key[("corrected", "A1_shortage_severe", "2.5,2.5")][key]))
        for key in fieldnames if key not in {"arm", "case", "cell"}
    )
    checks = {
        "calm_first_three_maximum_difference": calm_maximum,
        "corrected_published_service": corrected_service,
        "legacy_published_service": legacy_service,
        "legacy_2_5_vs_2_0_maximum_difference": legacy_cap_difference,
        "corrected_2_5_vs_2_0_maximum_difference": corrected_cap_difference,
    }
    gate = _manual_gate(checks)
    passed = bool(gate["pass"])
    status = {
        "schema_version": "v17-model-effect-v1",
        "state": "completed" if passed else "failed",
        "pass": passed,
        "checks": checks,
        "stop_conditions": gate["stop_conditions"],
        "diagnostic_expectations": gate["diagnostic_expectations"],
        "model_effect_csv": str(output / "model_effect.csv"),
    }
    atomic_json(output / "status.json", status)
    print(json.dumps(status, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit("corrected model effect gate failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
