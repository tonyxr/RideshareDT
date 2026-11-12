#!/usr/bin/env python3
"""Plan the C3 fleet grid and select fleet/pickup calibration results."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


BASE = Path(__file__).resolve().parents[1]
FLEET_GRID = tuple(round(0.6 + 0.1 * index, 1) for index in range(15))
NYC_TRIPS_PER_VEHICLE_HOUR = 645_549 / (55_701 * 6.5)


def select_fit(
    rows: Sequence[Mapping[str, Any]], *, target_trips_per_hour: float,
    target_median_wait: float,
) -> dict[str, Any]:
    finite = [row for row in rows if all(math.isfinite(float(row[key])) for key in (
        "fleet_scale", "trips_per_online_driver_hour", "median_wait_minutes"
    ))]
    if not finite:
        raise ValueError("fleet refit has no finite measurements")
    fleet = min(
        finite,
        key=lambda row: (
            abs(float(row["trips_per_online_driver_hour"]) - target_trips_per_hour),
            float(row["fleet_scale"]),
        ),
    )
    same_scale = [row for row in finite if float(row["fleet_scale"]) == float(fleet["fleet_scale"])]
    pickup = min(
        same_scale,
        key=lambda row: abs(float(row["median_wait_minutes"]) - target_median_wait),
    )
    return {
        "schema_version": "v18-supply-calibration-v1",
        "source": "TLC Monthly Data Reports and HVFHV request-to-pickup records",
        "target_trips_per_vehicle_hour": float(target_trips_per_hour),
        "target_median_wait_minutes": float(target_median_wait),
        "fleet_scale": float(fleet["fleet_scale"]),
        "base_pickup_minutes": float(pickup["base_pickup_minutes"]),
        "calm_fulfillment": float(pickup["fulfillment"]),
        "selected_measurement": dict(pickup),
        "measurements": [dict(row) for row in rows],
    }


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", choices=("nyc", "boston", "chicago"), default="nyc")
    parser.add_argument("--measurements", type=Path)
    parser.add_argument("--target-trips-per-hour", type=float, default=NYC_TRIPS_PER_VEHICLE_HOUR)
    parser.add_argument("--target-median-wait", type=float)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--emit-plan", action="store_true")
    args = parser.parse_args()
    output = args.output or BASE / f"calibration_artifacts/v18/{args.city}/supply.json"
    if args.emit_plan:
        payload = {
            "city": args.city, "customers": 500, "fleet_grid": FLEET_GRID,
            "pay_rule": "linear", "v3_daily_stock": True,
            "reference_earnings": "measured-calm-then-fixed",
            "target_trips_per_vehicle_hour": args.target_trips_per_hour,
            "pickup_fit": "match TLC median request-to-pickup wait",
        }
    else:
        if args.measurements is None or args.target_median_wait is None:
            raise ValueError("selection needs --measurements and --target-median-wait")
        rows = json.loads(args.measurements.read_text(encoding="utf-8"))
        payload = select_fit(
            rows, target_trips_per_hour=args.target_trips_per_hour,
            target_median_wait=args.target_median_wait,
        )
        if args.city != "nyc":
            payload["source"] = "transferred_from_nyc_unless_city_vehicle_target_available"
    atomic_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
