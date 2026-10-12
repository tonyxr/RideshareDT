#!/usr/bin/env python3
"""Multi-seed robustness backtest for the three-city preprocessing contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from city_data_pipeline import (  # noqa: E402
    DataValidationError,
    NOAAWeatherIndex,
    default_source_paths,
    discover_ride_files,
    iter_tabular_rows,
    normalize_ride_record,
    resolve_city_spec,
)


DEFAULT_SEEDS = (4500, 99173, 27183, 65537)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _moments(values: List[float]) -> Dict[str, float]:
    return {
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.pstdev(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _sample_city(
    city: str,
    data_root: Path,
    rows: int,
    seed: int,
    weather: NOAAWeatherIndex,
    files: List[Path],
) -> Dict[str, Any]:
    spec = resolve_city_spec(city)
    prices: List[float] = []
    distances: List[float] = []
    durations: List[float] = []
    source_ids: List[str] = []
    weather_joined = 0
    day_counts: Counter = Counter()
    hour_counts: Counter = Counter()
    rejections: Counter = Counter()
    processed = 0
    for file_index, path in enumerate(files):
        file_seed = int(seed) + file_index * 1000003
        for raw in iter_tabular_rows(
            path,
            sample_seed=file_seed,
            max_rows_hint=max(1, rows - len(prices)),
        ):
            processed += 1
            try:
                record = normalize_ride_record(spec.key, raw, weather_index=weather)
            except DataValidationError as exc:
                rejections[exc.code] += 1
                continue
            prices.append(record.actual_price)
            distances.append(record.distance_miles)
            if record.duration_minutes is not None:
                durations.append(record.duration_minutes)
            source_ids.append(record.source_id)
            day_counts[int(record.day_of_week)] += 1
            hour_counts[int(record.hour)] += 1
            weather_joined += int(record.weather_source == "noaa_global_hourly")
            if len(prices) >= rows:
                break
        if len(prices) >= rows:
            break
    signature = hashlib.sha256("\n".join(source_ids).encode("utf-8")).hexdigest()
    return {
        "seed": int(seed),
        "processed": processed,
        "accepted": len(prices),
        "rejections": dict(rejections),
        "sample_signature": signature,
        "unique_source_id_rate": len(set(source_ids)) / len(source_ids) if source_ids else 0.0,
        "weather_join_rate": weather_joined / len(prices) if prices else 0.0,
        "duration_coverage": len(durations) / len(prices) if prices else 0.0,
        "maximum_day_share": (
            max(day_counts.values()) / len(prices) if prices else 0.0
        ),
        "maximum_hour_share": (
            max(hour_counts.values()) / len(prices) if prices else 0.0
        ),
        "price": _moments(prices) if prices else {},
        "distance": _moments(distances) if distances else {},
        "duration": _moments(durations) if durations else {},
    }


def backtest_city(city: str, data_root: Path, rows: int, seeds: List[int]) -> Dict[str, Any]:
    spec = resolve_city_spec(city)
    paths = default_source_paths(spec.key, data_root)
    weather = NOAAWeatherIndex.from_directory(spec.key, paths["weather"])
    files = discover_ride_files(spec.key, paths["rides"])
    runs = [
        _sample_city(spec.key, data_root, rows, seed, weather, files)
        for seed in seeds
    ]
    failures: List[str] = []
    for run in runs:
        if run["accepted"] != rows:
            failures.append(f"seed={run['seed']} accepted {run['accepted']} != {rows}")
        if run["weather_join_rate"] < 0.98:
            failures.append(f"seed={run['seed']} weather join {run['weather_join_rate']:.3f} < 0.98")
        if run["unique_source_id_rate"] < 0.999:
            failures.append(
                f"seed={run['seed']} unique source-id rate {run['unique_source_id_rate']:.3f} < 0.999"
            )
        if run["maximum_day_share"] > 0.35:
            failures.append(
                f"seed={run['seed']} maximum day share {run['maximum_day_share']:.3f} > 0.35"
            )
        if run["maximum_hour_share"] > 0.20:
            failures.append(
                f"seed={run['seed']} maximum hour share {run['maximum_hour_share']:.3f} > 0.20"
            )
        for field in ("price", "distance"):
            if not run[field] or not all(math.isfinite(float(value)) for value in run[field].values()):
                failures.append(f"seed={run['seed']} non-finite {field} moments")
            elif run[field]["min"] <= 0.0:
                failures.append(f"seed={run['seed']} non-positive {field}")
        coverage = float(run["duration_coverage"])
        if spec.key == "boston" and coverage > 0.01:
            failures.append(f"Boston quote source unexpectedly reports duration coverage={coverage:.3f}")
        if spec.key != "boston" and coverage < 0.98:
            failures.append(f"seed={run['seed']} duration coverage {coverage:.3f} < 0.98")
    signatures = {run["sample_signature"] for run in runs}
    if len(signatures) != len(runs):
        failures.append("different dataset seeds produced identical source-id samples")
    price_means = [float(run["price"]["mean"]) for run in runs if run["price"]]
    distance_means = [float(run["distance"]["mean"]) for run in runs if run["distance"]]
    return {
        "city": spec.market_name,
        "ride_semantics": spec.ride_semantics,
        "price_semantics": spec.price_semantics,
        "passed": not failures,
        "failures": failures,
        "runs": runs,
        "across_seed": {
            "price_mean_mean": statistics.fmean(price_means),
            "price_mean_std": statistics.pstdev(price_means),
            "distance_mean_mean": statistics.fmean(distance_means),
            "distance_mean_std": statistics.pstdev(distance_means),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "data" / "raw"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "data" / "processed" / "backtest_report.json"))
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="comma-separated independent dataset seeds",
    )
    args = parser.parse_args()
    seeds = [int(item.strip()) for item in str(args.seeds).split(",") if item.strip()]
    if len(seeds) < 2:
        parser.error("at least two seeds are required")
    data_root = Path(args.data_root).expanduser().resolve()
    reports = []
    for city in ("nyc", "boston", "chicago"):
        print(f"[backtest] city={city}", flush=True)
        report = backtest_city(city, data_root, max(1, int(args.rows)), seeds)
        reports.append(report)
        print(
            f"[backtest] passed={report['passed']} "
            f"price_mean={report['across_seed']['price_mean_mean']:.3f} "
            f"price_seed_sd={report['across_seed']['price_mean_std']:.3f}",
            flush=True,
        )
    payload = {
        "schema_version": 1,
        "passed": all(report["passed"] for report in reports),
        "rows_per_seed": int(args.rows),
        "seeds": seeds,
        "cities": reports,
    }
    _atomic_json(Path(args.output).expanduser().resolve(), payload)
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
