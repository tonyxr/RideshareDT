#!/usr/bin/env python3
"""Validate public source bundles and materialize bounded canonical samples."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from city_market_calibration import market_dynamics_calibration  # noqa: E402
from city_data_pipeline import (  # noqa: E402
    CANONICAL_RIDE_COLUMNS,
    systematic_csv_sampling,
    default_source_paths,
    derive_city_calibration,
    discover_ride_files,
    iter_normalized_rides,
    load_city_catalog,
    resolve_city_spec,
    validate_city_bundle,
)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CANONICAL_RIDE_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    os.replace(temporary, path)
    return count


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_manifest(city: str, data_root: Path, include_hashes: bool) -> List[Dict[str, Any]]:
    spec = resolve_city_spec(city)
    paths = default_source_paths(spec.key, data_root)
    sources = discover_ride_files(spec.key, paths["rides"])
    sources.extend(sorted(paths["weather"].glob(spec.weather_glob)))
    sources.extend((paths["pums_person"], paths["pums_household"], paths["puma_geography"]))
    result = []
    for path in sources:
        stat = path.stat()
        item: Dict[str, Any] = {
            "path": str(path.resolve()),
            "size_bytes": int(stat.st_size),
            "modified_ns": int(stat.st_mtime_ns),
        }
        if include_hashes:
            item["sha256"] = _sha256(path)
        result.append(item)
    return result


def prepare_city(
    city: str,
    data_root: Path,
    output_root: Path,
    sample_rows: int,
    calibration_rows: int,
    sample_seed: int,
    include_hashes: bool,
    trip_bank_rows: int = 0,
) -> Dict[str, Any]:
    spec = resolve_city_spec(city)
    validation = validate_city_bundle(spec.key, data_root, sample_rows=min(sample_rows, 250))
    calibration = derive_city_calibration(
        spec.key,
        data_root,
        ride_sample_rows=calibration_rows,
        sample_seed=sample_seed,
    )
    paths = default_source_paths(spec.key, data_root)
    rejections: Dict[str, int] = {}
    records = iter_normalized_rides(
        spec.key,
        paths["rides"],
        weather_root=paths["weather"],
        max_rows=sample_rows,
        sample_seed=sample_seed,
        rejection_counts=rejections,
    )
    city_output = output_root / spec.key
    sample_path = city_output / "rides_normalized_sample.csv"
    sample_count = _atomic_csv(sample_path, (record.to_dict() for record in records))
    if trip_bank_rows > 0:
        # v26: a larger whole-file sample becomes the operational trip bank and
        # the source of data-calibrated tariff anchors, weather demand
        # intensity, city price level, and (when observed) driver pay share.
        bank_rejections: Dict[str, int] = {}
        bank_records = [
            record.to_dict()
            for record in iter_normalized_rides(
                spec.key,
                paths["rides"],
                weather_root=paths["weather"],
                max_rows=trip_bank_rows,
                sample_seed=sample_seed + 26,
                rejection_counts=bank_rejections,
            )
        ]
        bank_path = city_output / "trip_bank.csv"
        _atomic_csv(bank_path, bank_records)
        identifiable = calibration["quality"]["identifiable_market_fields"]
        duration_coverage = sum(
            1 for row in bank_records if row.get("duration_minutes") not in (None, "")
        ) / max(1, len(bank_records))
        v26 = market_dynamics_calibration(
            bank_records,
            weather_probs=calibration["calibration"]["market"]["weather_probs"],
            duration_identified=duration_coverage >= 0.5,
            service_identified=bool(identifiable.get("premium_service_probability")),
            airport_identified=bool(identifiable.get("airport_probability")),
            outside_option_base_fare=(spec.market or {}).get("transit_fare"),
        )
        cal = calibration["calibration"]
        cal["tariff"] = v26["tariff"]
        cal["market"]["weather_demand_multiplier"] = v26["weather_demand_multiplier"]
        cal["choice"] = {**cal.get("choice", {}), **v26["choice"]}
        cal["supply"] = v26["supply"]
        cal["trip_bank"] = {
            "path": "trip_bank.csv",
            "rows": len(bank_records),
            "rejections": bank_rejections,
            "duration_identified": duration_coverage >= 0.5,
            "duration_coverage": duration_coverage,
            "airport_identified": bool(identifiable.get("airport_probability")),
            "service_identified": bool(identifiable.get("premium_service_probability")),
            "sampling": "uniform_random_whole_source",
        }
        calibration["schema_version"] = 2
    calibration_path = city_output / "calibration.json"
    _atomic_json(calibration_path, calibration)
    report = {
        "schema_version": 1,
        "city": spec.market_name,
        "validation": validation,
        "normalization": {
            "sample_path": str(sample_path.resolve()),
            "sample_rows": sample_count,
            "rejections": rejections,
            "sample_seed": int(sample_seed),
        },
        "calibration_path": str(calibration_path.resolve()),
        "sources": _source_manifest(spec.key, data_root, include_hashes=include_hashes),
    }
    _atomic_json(city_output / "preprocessing_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--city",
        default="all",
        help="nyc, boston, chicago, or all",
    )
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "data" / "raw"))
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "data" / "processed"))
    parser.add_argument("--sample-rows", type=int, default=5000)
    parser.add_argument("--calibration-ride-rows", type=int, default=20000)
    parser.add_argument("--sample-seed", type=int, default=4500)
    parser.add_argument(
        "--trip-bank-rows",
        type=int,
        default=50000,
        help="Rows in the v26 empirical trip bank (0 disables v26 calibration).",
    )
    parser.add_argument(
        "--legacy-csv-sampling",
        action="store_true",
        help="Use the v25 first-N-rows CSV sampler instead of the whole-file uniform random sample.",
    )
    parser.add_argument(
        "--hash-sources",
        action="store_true",
        help="SHA-256 every source file; slower for the 11 GiB NYC Parquet file.",
    )
    args = parser.parse_args()

    systematic_csv_sampling(not args.legacy_csv_sampling)
    catalog = {
        key: spec for key, spec in load_city_catalog().items()
        if str(spec.status) == "available"
    }
    cities = list(catalog) if str(args.city).strip().lower() == "all" else [resolve_city_spec(args.city).key]
    reports = []
    for city in cities:
        print(f"[prepare] city={city}", flush=True)
        report = prepare_city(
            city,
            Path(args.data_root).expanduser().resolve(),
            Path(args.output_root).expanduser().resolve(),
            sample_rows=max(1, int(args.sample_rows)),
            calibration_rows=max(1, int(args.calibration_ride_rows)),
            sample_seed=int(args.sample_seed),
            include_hashes=bool(args.hash_sources),
            trip_bank_rows=max(0, int(args.trip_bank_rows)),
        )
        reports.append(report)
        print(
            f"[prepare] accepted={report['normalization']['sample_rows']} "
            f"semantics={report['validation']['ride_semantics']} "
            f"weather_join={report['validation']['sample_weather_join_rate']:.3f}",
            flush=True,
        )
    _atomic_json(Path(args.output_root).expanduser().resolve() / "preprocessing_summary.json", reports)


if __name__ == "__main__":
    main()

