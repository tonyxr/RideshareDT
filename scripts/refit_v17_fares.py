#!/usr/bin/env python3
"""Refit V17 variable card rates after the realistic-fare decomposition.

The V17 formula stops multiplying fixed base/booking charges by contextual
multipliers.  This script rescales only per-mile and per-minute anchors so the
mean card fare on the declared 80% fit partition matches the 15d card.  It
copies city calibrations into this tree; source calibrations remain untouched.
The final 20% of each trip bank is reserved for T1 validation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent
sys.path.insert(0, str(BASE))

from MarketInteraction import MarketInteraction, RideContext  # noqa: E402
from diagnostic_logging import atomic_json  # noqa: E402

CITIES = {
    "nyc": "New York City",
    "boston": "Boston",
    "chicago": "Chicago",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def context(row: Mapping[str, str]) -> tuple[float, float, RideContext]:
    distance = float(row["distance_miles"])
    duration = (
        float(row["duration_minutes"])
        if str(row.get("duration_minutes", "")).strip()
        else max(5.0, 60.0 * distance / 22.0)
    )
    return distance, duration, RideContext(
        day_of_week=int(row["day_of_week"]),
        weather=str(row["weather"]),
        hour=int(row["hour"]),
        airport=str(row["airport"]).strip().lower() in {"1", "true", "yes"},
        service=str(row["service"]),
    )


def quote_mean(
    city: str,
    tariff: Mapping[str, Any],
    rows: Iterable[Mapping[str, str]],
    *,
    model: str,
) -> float:
    market = MarketInteraction(city, seed=17)
    market.register_calibrated_tariff(dict(tariff))
    market.configure_fare_model(model)
    quotes = [market.quote_price(*context(row)) for row in rows]
    return float(np.mean(np.asarray(quotes, dtype=float)))


def scaled_tariff(tariff: Mapping[str, Any], scale: float) -> dict[str, Any]:
    value = json.loads(json.dumps(tariff))
    value["per_mile"] = float(tariff["per_mile"]) * float(scale)
    value["per_minute"] = float(tariff["per_minute"]) * float(scale)
    return value


def fit_scale(
    city: str,
    tariff: Mapping[str, Any],
    rows: list[Mapping[str, str]],
) -> tuple[float, float, float]:
    target = quote_mean(city, tariff, rows, model="legacy")
    low, high = 0.10, 5.0
    for _ in range(48):
        middle = (low + high) / 2.0
        value = quote_mean(
            city, scaled_tariff(tariff, middle), rows, model="realistic"
        )
        if value < target:
            low = middle
        else:
            high = middle
    scale = (low + high) / 2.0
    achieved = quote_mean(
        city, scaled_tariff(tariff, scale), rows, model="realistic"
    )
    return scale, target, achieved


def refit_city(slug: str, output: Path) -> dict[str, Any]:
    source = ROOT / "data/processed" / slug / "calibration.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    calibration = dict(payload["calibration"])
    trip_bank = source.parent / str(calibration["trip_bank"]["path"])
    with trip_bank.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    split = int(len(rows) * 0.80)
    fit_rows, heldout_rows = rows[:split], rows[split:]
    original = dict(calibration["tariff"])
    scale, fit_target, fit_achieved = fit_scale(
        CITIES[slug], original, fit_rows
    )
    refitted = scaled_tariff(original, scale)
    heldout_legacy = quote_mean(
        CITIES[slug], original, heldout_rows, model="legacy"
    )
    heldout_realistic = quote_mean(
        CITIES[slug], refitted, heldout_rows, model="realistic"
    )
    relative_gap = heldout_realistic / heldout_legacy - 1.0
    calibration["tariff"] = refitted
    # This file lives four directories below the shared data directory.
    calibration["trip_bank"] = dict(calibration["trip_bank"])
    calibration["trip_bank"]["path"] = (
        f"../../../../data/processed/{slug}/trip_bank.csv"
    )
    audit = {
        "contract": "v17-realistic-fare-variable-rate-refit-v1",
        "source_calibration": str(source.resolve()),
        "source_calibration_sha256": sha256(source),
        "source_trip_bank_sha256": sha256(trip_bank),
        "partition": {
            "order": "trip_bank_file_order",
            "fit_rows": len(fit_rows),
            "heldout_rows": len(heldout_rows),
            "fit_fraction": 0.80,
        },
        "method": "common scalar on per-mile and per-minute anchors",
        "variable_rate_scale": scale,
        "fit_v15d_card_mean": fit_target,
        "fit_v17_card_mean": fit_achieved,
        "heldout_v15d_card_mean": heldout_legacy,
        "heldout_v17_card_mean": heldout_realistic,
        "heldout_relative_gap": relative_gap,
    }
    payload["calibration"] = calibration
    payload["v17_fare_refit"] = audit
    city_out = output / slug
    city_out.mkdir(parents=True, exist_ok=False)
    atomic_json(city_out / "calibration.json", payload)
    atomic_json(city_out / "fare_refit.json", audit)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=BASE / "calibration_artifacts/v17",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {slug: refit_city(slug, output) for slug in CITIES}
    atomic_json(output / "report.json", report)
    for slug, result in report.items():
        print(
            f"{slug}: variable scale={result['variable_rate_scale']:.6f}; "
            f"held-out gap={result['heldout_relative_gap']:.3%}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
