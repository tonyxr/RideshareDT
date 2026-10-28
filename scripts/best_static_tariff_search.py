#!/usr/bin/env python3
"""Revision 14 validation-only coordinate search for a fixed differentiated tariff."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics as st
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parent
sys.path.insert(0, str(HERE))
from v26r_case_eval import run_specs, spec_key  # noqa: E402
from v26r_cases import BURNIN_DAYS, CASES, STATIC_GRID  # noqa: E402


COEFFICIENTS = (
    "base_fare",
    "booking_fee",
    "per_minute",
    "per_mile",
    "airport_fee",
)
CASES_TO_SCORE = ("A1_shortage_severe", "C0_calm")


def _write_tariff(path: Path, tariff: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({key: float(tariff[key]) for key in COEFFICIENTS}, indent=2)
        + "\n"
    )


def _candidate_path(root: Path, tariff: dict[str, float]) -> Path:
    canonical = json.dumps(tariff, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    path = root / "candidates" / f"tariff_{digest}.json"
    if not path.exists():
        _write_tariff(path, tariff)
    return path.resolve()


def _score_payload(payload: dict[str, Any], case_name: str) -> list[float]:
    spd = int(payload["steps_per_day"])
    burnin = int(CASES[case_name].get("burnin", BURNIN_DAYS))
    by_opening: dict[str, list[float]] = {}
    for row in payload["traces"]:
        if int(row["step"]) // spd < burnin:
            continue
        by_opening.setdefault(str(row["rollout"]), []).append(float(row["reward"]))
    if len(by_opening) != 6 or any(not values for values in by_opening.values()):
        raise ValueError(
            f"expected six complete validation openings for {case_name}, got "
            f"{len(by_opening)}"
        )
    return [st.mean(values) for values in by_opening.values()]


def evaluate_candidate(
    tariff: dict[str, float],
    *,
    root: Path,
    policy: Path,
) -> dict[str, Any]:
    tariff_path = _candidate_path(root, tariff)
    spec = f"static_tariff:@{tariff_path}"
    scores: list[float] = []
    case_scores: dict[str, float] = {}
    for case_name in CASES_TO_SCORE:
        out = root / "validation" / case_name
        out.mkdir(parents=True, exist_ok=True)
        payload_path = out / f"{spec_key(spec)}.json"
        if not payload_path.exists():
            run_specs(
                "New York City",
                "nyc",
                case_name,
                [spec],
                "val",
                6,
                "full",
                str(policy),
                str(out),
                customers=50,
                horizon_days=40,
                rivals=["adaptive_best_response_memory"],
                pool=2000,
            )
        opening_scores = _score_payload(json.loads(payload_path.read_text()), case_name)
        case_scores[case_name] = st.mean(opening_scores)
        scores.extend(opening_scores)
    objective = st.mean(scores)
    se = st.stdev(scores) / math.sqrt(len(scores)) if len(scores) > 1 else 0.0
    return {
        **{key: float(tariff[key]) for key in COEFFICIENTS},
        "objective": objective,
        "standard_error": se,
        "n_opening_cases": len(scores),
        **{f"objective_{case}": value for case, value in case_scores.items()},
        "candidate_file": str(tariff_path),
    }


def _anchors() -> dict[str, float]:
    calibration = json.loads((ROOT / "data/processed/nyc/calibration.json").read_text())
    tariff = calibration["calibration"]["tariff"]
    return {key: float(tariff[key]) for key in COEFFICIENTS}


def frozen_rl_tariff(diagnostics: Path) -> dict[str, float]:
    with diagnostics.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty evaluation diagnostics: {diagnostics}")
    last = rows[-1]
    anchors = _anchors()
    return {
        key: float(last[f"firm1_{key}"]) / anchors[key]
        for key in COEFFICIENTS
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=BASE / "runs" / "revision14_verify" / "static_tariff_search",
    )
    args = parser.parse_args()
    policy = args.policy.resolve()
    diagnostics = args.diagnostics.resolve()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    command = {
        "policy": str(policy),
        "diagnostics": str(diagnostics),
        "grid": list(STATIC_GRID),
        "coefficient_order": list(COEFFICIENTS),
        "sweeps": 2,
        "cases": list(CASES_TO_SCORE),
        "seed_set": "val",
        "openings": 6,
        "customers": 50,
        "horizon_days": 40,
        "rival": "adaptive_best_response_memory",
    }
    (root / "command.json").write_text(json.dumps(command, indent=2) + "\n")

    current = {key: 1.2 for key in COEFFICIENTS}
    records: list[dict[str, Any]] = []
    cache: dict[tuple[float, ...], dict[str, Any]] = {}
    for sweep in range(1, 3):
        for coefficient in COEFFICIENTS:
            candidates: list[dict[str, Any]] = []
            for target in STATIC_GRID:
                candidate = dict(current)
                candidate[coefficient] = float(target)
                key = tuple(candidate[name] for name in COEFFICIENTS)
                if key not in cache:
                    cache[key] = evaluate_candidate(candidate, root=root, policy=policy)
                row = dict(cache[key], sweep=sweep, coordinate=coefficient, target=float(target))
                records.append(row)
                candidates.append(row)
                print(
                    f"[static-search] sweep={sweep} {coefficient}={target:g} "
                    f"objective={row['objective']:.6f}",
                    flush=True,
                )
            # First grid value wins an exact tie, providing deterministic selection.
            winner = max(candidates, key=lambda row: row["objective"])
            current[coefficient] = float(winner[coefficient])

    _write_tariff(root / "best_static_tariff.json", current)
    frozen = frozen_rl_tariff(diagnostics)
    _write_tariff(root / "frozen_rl_tariff.json", frozen)
    fields = [
        "sweep",
        "coordinate",
        "target",
        *COEFFICIENTS,
        "objective",
        "standard_error",
        "n_opening_cases",
        *(f"objective_{case}" for case in CASES_TO_SCORE),
        "candidate_file",
    ]
    with (root / "candidate_scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "best_static_tariff": current,
        "best_static_objective": evaluate_candidate(current, root=root, policy=policy),
        "frozen_rl_tariff": frozen,
        "unique_candidates": len(cache),
        "evaluations": len(records),
    }
    (root / "search_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
