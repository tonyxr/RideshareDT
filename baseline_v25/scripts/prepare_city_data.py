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

from city_data_pipeline import (  # noqa: E402
    CANONICAL_RIDE_COLUMNS,
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
        "--hash-sources",
        action="store_true",
        help="SHA-256 every source file; slower for the 11 GiB NYC Parquet file.",
    )
    args = parser.parse_args()

    catalog = load_city_catalog()
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

