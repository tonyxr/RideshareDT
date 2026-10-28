#!/usr/bin/env python3
"""Revision 15 validation-only search for the fixed tariff structure S and reference level.

Step 1: coordinate search (two sweeps, five coefficients, the 13-point static grid,
start at a uniform 1.2x tariff) over fixed per-coefficient tariffs.
Step 2: factor the winner T into level(T) * S, where S is normalized to level 1 with
the typical-ride exposure weights.
Step 3: score S * L for every lattice level L within 0.3 of level(T); the best L is the
reference level L_ref.

Score: per validation opening, the mean over post-burn-in steps of
0.7 * own profit per request + 0.3 * (own - rival profit per request), on the severe
shortage case (A1) and the calm case (C0), six validation openings each, against the
memory rival. The training reward column is not used, so the search does not depend
on any policy's reward contract.

The contract policy is used only so Core restores the city, supply and demand contract
of the training runs. Every evaluated spec is a static tariff, so the actor is never
queried.
"""
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
sys.path.insert(0, str(BASE))
from v26r_case_eval import run_specs, spec_key  # noqa: E402
from v26r_cases import BURNIN_DAYS, CASES, STATIC_GRID  # noqa: E402
from pricing_models import (  # noqa: E402
    TARIFF_LEVEL_GRID_STEP,
    TARIFF_STRUCTURE_FORMAT,
    snap_tariff_level,
    tariff_level_of,
    validate_tariff_structure,
)

COEFFICIENTS = ("base_fare", "per_minute", "per_mile", "booking_fee", "airport_fee")
CASES_TO_SCORE = ("A1_shortage_severe", "C0_calm")
OWN_WEIGHT = 0.7
LEAD_WEIGHT = 0.3
OPENINGS = 6
LEVEL_SEARCH_RADIUS = 0.3
COMPARATOR_RADIUS = 0.2


def anchors() -> dict[str, float]:
    calibration = json.loads((ROOT / "data/processed/nyc/calibration.json").read_text())
    tariff = calibration["calibration"]["tariff"]
    return {key: float(tariff[key]) for key in COEFFICIENTS}


def parse_bounds(text: str) -> tuple[float, float]:
    lower, upper = (snap_tariff_level(float(part)) for part in text.split(","))
    if not 0.0 < lower < upper:
        raise ValueError(f"invalid level bounds {text!r}")
    return lower, upper


def lattice(lower: float, upper: float) -> list[float]:
    count = int(round((upper - lower) / TARIFF_LEVEL_GRID_STEP))
    return [round(lower + index * TARIFF_LEVEL_GRID_STEP, 6) for index in range(count + 1)]


def tariff_key(tariff: dict[str, float]) -> tuple[float, ...]:
    return tuple(round(float(tariff[key]), 12) for key in COEFFICIENTS)


def write_tariff(path: Path, tariff: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({key: float(tariff[key]) for key in COEFFICIENTS}, indent=2) + "\n"
    )


def tariff_file(root: Path, tariff: dict[str, float]) -> Path:
    canonical = json.dumps(
        {key: round(float(tariff[key]), 12) for key in COEFFICIENTS},
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    path = root / "candidates" / f"tariff_{digest}.json"
    if not path.exists():
        write_tariff(path, tariff)
    return path.resolve()


def score_payload(payload: dict[str, Any], case_name: str) -> list[float]:
    spd = int(payload["steps_per_day"])
    burnin = int(CASES[case_name].get("burnin", BURNIN_DAYS))
    by_opening: dict[str, list[float]] = {}
    for row in payload["traces"]:
        if int(row["step"]) // spd < burnin:
            continue
        own = float(row["profit_per_request"])
        rival = float(row["rival_profit_per_request"])
        by_opening.setdefault(str(row["rollout"]), []).append(
            OWN_WEIGHT * own + LEAD_WEIGHT * (own - rival)
        )
    if len(by_opening) != OPENINGS or any(not values for values in by_opening.values()):
        raise ValueError(
            f"expected {OPENINGS} complete validation openings for {case_name}, "
            f"got {len(by_opening)}"
        )
    return [st.mean(by_opening[name]) for name in sorted(by_opening)]


def evaluate_many(
    tariffs: list[dict[str, float]], *, root: Path, policy: Path
) -> list[dict[str, Any]]:
    specs = [f"static_tariff:@{tariff_file(root, tariff)}" for tariff in tariffs]
    for case_name in CASES_TO_SCORE:
        out = root / "validation" / case_name
        out.mkdir(parents=True, exist_ok=True)
        missing = [spec for spec in specs if not (out / f"{spec_key(spec)}.json").exists()]
        if missing:
            run_specs(
                "New York City",
                "nyc",
                case_name,
                missing,
                "val",
                OPENINGS,
                "full",
                str(policy),
                str(out),
                customers=50,
                horizon_days=40,
                rivals=["adaptive_best_response_memory"],
                pool=2000,
            )
    results = []
    for tariff, spec in zip(tariffs, specs):
        scores: list[float] = []
        case_scores: dict[str, float] = {}
        for case_name in CASES_TO_SCORE:
            path = root / "validation" / case_name / f"{spec_key(spec)}.json"
            opening_scores = score_payload(json.loads(path.read_text()), case_name)
            case_scores[case_name] = st.mean(opening_scores)
            scores.extend(opening_scores)
        results.append({
            **{key: float(tariff[key]) for key in COEFFICIENTS},
            "objective": st.mean(scores),
            "standard_error": st.stdev(scores) / math.sqrt(len(scores)),
            "n_opening_cases": len(scores),
            **{f"objective_{case}": value for case, value in case_scores.items()},
            "spec": spec,
        })
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract-policy", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--level-bounds", default="0.6,2.0")
    args = parser.parse_args()
    policy = args.contract_policy.resolve()
    if not policy.is_file():
        raise FileNotFoundError(f"contract policy not found: {policy}")
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lower, upper = parse_bounds(args.level_bounds)
    city_anchors = anchors()
    policy_sha = hashlib.sha256(policy.read_bytes()).hexdigest()
    (root / "command.json").write_text(json.dumps({
        "contract_policy": str(policy),
        "contract_policy_sha256": policy_sha,
        "grid": list(STATIC_GRID),
        "coefficient_order": list(COEFFICIENTS),
        "sweeps": 2,
        "cases": list(CASES_TO_SCORE),
        "seed_set": "val",
        "openings": OPENINGS,
        "customers": 50,
        "horizon_days": 40,
        "rival": "adaptive_best_response_memory",
        "objective": f"{OWN_WEIGHT} own profit + {LEAD_WEIGHT} profit lead per request",
        "level_bounds": [lower, upper],
    }, indent=2) + "\n")

    # Step 1: coordinate search over fixed per-coefficient tariffs.
    current = {key: 1.2 for key in COEFFICIENTS}
    cache: dict[tuple[float, ...], dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    for sweep in (1, 2):
        for coefficient in COEFFICIENTS:
            candidates = [dict(current, **{coefficient: float(target)}) for target in STATIC_GRID]
            pending = [c for c in candidates if tariff_key(c) not in cache]
            for result in evaluate_many(pending, root=root, policy=policy):
                cache[tariff_key(result)] = result
            rows = []
            for candidate, target in zip(candidates, STATIC_GRID):
                row = dict(cache[tariff_key(candidate)], sweep=sweep, coordinate=coefficient, target=float(target))
                rows.append(row)
                records.append(row)
                print(f"[r15-structure] sweep={sweep} {coefficient}={target:g} objective={row['objective']:.6f}", flush=True)
            # max() keeps the first grid value on an exact tie.
            winner = max(rows, key=lambda row: row["objective"])
            current[coefficient] = float(winner[coefficient])
    best_tariff = dict(current)
    best_result = cache[tariff_key(best_tariff)]

    # Step 2: factor the winner into level * structure.
    raw_level = tariff_level_of(best_tariff, city_anchors)
    structure = {key: float(best_tariff[key]) / raw_level for key in COEFFICIENTS}
    if abs(tariff_level_of(structure, city_anchors) - 1.0) > 1e-9:
        raise RuntimeError("structure normalization failed")
    center = snap_tariff_level(round(raw_level / TARIFF_LEVEL_GRID_STEP) * TARIFF_LEVEL_GRID_STEP)

    # Step 3: line search over the level lattice around the winner's level.
    levels = [
        level for level in lattice(lower, upper)
        if abs(level - center) <= LEVEL_SEARCH_RADIUS + 1e-9
    ]
    if not levels:
        raise RuntimeError(f"no lattice level within {LEVEL_SEARCH_RADIUS} of {center}")
    level_tariffs = [{key: structure[key] * level for key in COEFFICIENTS} for level in levels]
    level_results = evaluate_many(level_tariffs, root=root, policy=policy)
    level_rows = [dict(result, level=level) for level, result in zip(levels, level_results)]
    for row in level_rows:
        print(f"[r15-structure] level={row['level']:.1f} objective={row['objective']:.6f}", flush=True)
    reference = max(level_rows, key=lambda row: (row["objective"], -row["level"]))["level"]

    payload = {
        "format": TARIFF_STRUCTURE_FORMAT,
        "structure": structure,
        "anchors": city_anchors,
        "reference_level": reference,
        "source": {
            "best_static_tariff": best_tariff,
            "best_static_objective": best_result["objective"],
            "best_static_standard_error": best_result["standard_error"],
            "best_static_level": raw_level,
            "level_line_search": [
                {"level": row["level"], "objective": row["objective"], "standard_error": row["standard_error"]}
                for row in level_rows
            ],
            "objective": f"{OWN_WEIGHT} own profit + {LEAD_WEIGHT} profit lead per request",
            "cases": list(CASES_TO_SCORE),
            "seed_set": "val",
            "openings": OPENINGS,
            "contract_policy_sha256": policy_sha,
        },
    }
    validated = validate_tariff_structure(payload, list(COEFFICIENTS), (lower, upper))
    (root / "tariff_structure.json").write_text(json.dumps(validated, indent=2, sort_keys=True) + "\n")

    # Evaluation comparators: S * L for every lattice level within 0.2 of L_ref.
    comparators = {}
    for level in lattice(lower, upper):
        if abs(level - reference) <= COMPARATOR_RADIUS + 1e-9:
            path = root / "level_tariffs" / f"level_{level:.1f}.json"
            write_tariff(path, {key: structure[key] * level for key in COEFFICIENTS})
            comparators[f"{level:.1f}"] = str(path.resolve())

    fields = ["sweep", "coordinate", "target", *COEFFICIENTS, "objective", "standard_error",
              "n_opening_cases", *(f"objective_{case}" for case in CASES_TO_SCORE), "spec"]
    with (root / "candidate_scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "tariff_structure": str((root / "tariff_structure.json").resolve()),
        "reference_level": reference,
        "best_static_tariff": best_tariff,
        "best_static_level": raw_level,
        "structure": structure,
        "level_line_search": payload["source"]["level_line_search"],
        "comparators": comparators,
        "unique_candidates": len(cache) + len(level_rows),
    }
    (root / "search_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
