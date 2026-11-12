#!/usr/bin/env python3
"""Fit or transfer the registered V18 linear driver-pay rule."""

from __future__ import annotations

import argparse
import csv
from datetime import date, datetime
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


BASE = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = BASE.parent / "data/processed/nyc/trip_bank.csv"
NYC_RATE = {"per_mile": 2.50, "per_minute": 0.50}
CITY_RATE = {
    "nyc": NYC_RATE,
    "boston": {"per_mile": 2.80, "per_minute": 0.28},
    "chicago": {"per_mile": 2.25, "per_minute": 0.39},
}
ALIASES = {
    "driver_pay": ("driver_pay", "DriverPay", "driver_pay_amount"),
    "distance_miles": ("distance_miles", "TripDistance", "trip_miles"),
    "duration_minutes": ("duration_minutes", "DurationMinutes", "trip_minutes"),
    "actual_price": ("actual_price", "Price", "base_passenger_fare", "fare"),
    "pickup_date": ("pickup_date", "pickup_datetime", "PickupDateTime", "date"),
}


def _value(row: Mapping[str, str], name: str) -> str:
    for key in ALIASES[name]:
        if key in row and str(row[key]).strip():
            return str(row[key]).strip()
    raise KeyError(f"missing {name}; accepted columns: {ALIASES[name]}")


def _date(value: str) -> date:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).date()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            try:
                parsed = {
                    "driver_pay": float(_value(row, "driver_pay")),
                    "distance_miles": float(_value(row, "distance_miles")),
                    "duration_minutes": float(_value(row, "duration_minutes")),
                    "actual_price": float(_value(row, "actual_price")),
                    "pickup_date": _date(_value(row, "pickup_date")),
                }
            except (KeyError, TypeError, ValueError):
                continue
            if parsed["driver_pay"] > 0 and all(
                np.isfinite(parsed[key])
                for key in ("driver_pay", "distance_miles", "duration_minutes", "actual_price")
            ):
                rows.append(parsed)
    if not rows:
        raise ValueError(f"no usable driver-pay rows in {path}")
    return rows


def fit(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    cutoff = date(2023, 6, 30)
    train = [row for row in rows if row["pickup_date"] <= cutoff]
    test = [
        row for row in rows
        if date(2023, 7, 1) <= row["pickup_date"] <= date(2023, 8, 31)
    ]
    if len(train) < 5 or len(test) < 2:
        raise ValueError("pay fit needs at least five training and two July-August rows")
    design = np.asarray([
        [1.0, row["distance_miles"], row["duration_minutes"], row["actual_price"]]
        for row in train
    ], dtype=float)
    target = np.asarray([row["driver_pay"] for row in train], dtype=float)
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    test_design = np.asarray([
        [1.0, row["distance_miles"], row["duration_minutes"], row["actual_price"]]
        for row in test
    ], dtype=float)
    test_target = np.asarray([row["driver_pay"] for row in test], dtype=float)
    prediction = test_design @ coefficients
    denominator = float(np.sum((test_target - np.mean(test_target)) ** 2))
    r2 = 1.0 - float(np.sum((test_target - prediction) ** 2)) / max(denominator, 1e-12)
    return {
        "schema_version": "v18-driver-pay-v1",
        "rule": "linear",
        "source": "nyc_trip_bank_ols",
        "fit_through": "2023-06-30",
        "test_period": ["2023-07-01", "2023-08-31"],
        "n_train": len(train), "n_test": len(test),
        "intercept": float(coefficients[0]),
        "per_mile": float(coefficients[1]),
        "per_minute": float(coefficients[2]),
        "fare_coef": float(coefficients[3]),
        "test_r2": float(r2),
    }


def transfer(nyc: Mapping[str, Any], city: str) -> dict[str, Any]:
    if city == "nyc":
        return dict(nyc)
    return {
        "schema_version": "v18-driver-pay-v1", "rule": "linear",
        "source": "transferred_from_nyc", "transfer_city": city,
        "intercept": 0.0,
        "per_mile": float(nyc["per_mile"]) * CITY_RATE[city]["per_mile"] / NYC_RATE["per_mile"],
        "per_minute": float(nyc["per_minute"]) * CITY_RATE[city]["per_minute"] / NYC_RATE["per_minute"],
        "fare_coef": float(nyc["fare_coef"]),
        "nyc_source": dict(nyc),
    }


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", choices=tuple(CITY_RATE), default="nyc")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--nyc-rule", type=Path, default=BASE / "calibration_artifacts/v18/nyc/pay_rule.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or BASE / f"calibration_artifacts/v18/{args.city}/pay_rule.json"
    if args.city == "nyc":
        result = fit(load_rows(args.input))
    else:
        result = transfer(json.loads(args.nyc_rule.read_text(encoding="utf-8")), args.city)
    atomic_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
