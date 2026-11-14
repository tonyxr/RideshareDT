#!/usr/bin/env python3
"""Download the Last Revision D1 inputs without computing validation metrics.

The downloader is restart-safe: large files use ``.part`` files and HTTP range
requests, small Chicago responses are written atomically, and the final manifest
is assembled only from files that exist on disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


REPO = Path(__file__).resolve().parents[2]
DATA = REPO.parent / "data"
RAW_VALIDATION = DATA / "raw" / "validation"
RUN_ROOT = REPO / "runs" / "v18" / "realdata"
LOG_PATH = RUN_ROOT / "d1_downloads.log"
USER_AGENT = "CRISP-Ride-v18-validation/1.0 (research download)"
CHICAGO_DATASET = "n26f-ihde"
CHICAGO_API = f"https://data.cityofchicago.org/resource/{CHICAGO_DATASET}.json"
CHICAGO_META = f"https://data.cityofchicago.org/api/views/{CHICAGO_DATASET}"
CHICAGO_COLUMNS = {
    "trip_id",
    "trip_start_timestamp",
    "trip_seconds",
    "trip_miles",
    "fare",
    "additional_charges",
    "pickup_community_area",
    "dropoff_community_area",
}
PRINT_LOCK = threading.Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _log(message: str) -> None:
    line = f"[{_utc_now()}] {message}"
    with PRINT_LOCK:
        print(line, flush=True)
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def _request(url: str, *, headers: dict[str, str] | None = None) -> urllib.request.Request:
    merged = {"User-Agent": USER_AGENT}
    if headers:
        merged.update(headers)
    return urllib.request.Request(url, headers=merged)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def _download(url: str, destination: Path, *, retries: int = 3) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    if destination.exists() and destination.stat().st_size > 0:
        _log(f"reuse {destination} ({destination.stat().st_size} bytes)")
    else:
        for attempt in range(1, retries + 1):
            try:
                offset = partial.stat().st_size if partial.exists() else 0
                headers = {"Range": f"bytes={offset}-"} if offset else None
                with urllib.request.urlopen(_request(url, headers=headers), timeout=120) as response:
                    append = offset > 0 and getattr(response, "status", None) == 206
                    mode = "ab" if append else "wb"
                    if offset and not append:
                        offset = 0
                    _log(f"download {destination.name} attempt={attempt} offset={offset}")
                    with partial.open(mode) as handle:
                        while True:
                            chunk = response.read(1024 * 1024)
                            if not chunk:
                                break
                            handle.write(chunk)
                partial.replace(destination)
                break
            except (OSError, urllib.error.URLError) as exc:
                _log(f"retry {destination.name} attempt={attempt}: {exc}")
                if attempt == retries:
                    raise
                time.sleep(min(30, 2**attempt))
    return {
        "url": url,
        "path": str(destination),
        "downloaded_utc": _utc_now(),
        "bytes": destination.stat().st_size,
        "sha256": _sha256(destination),
    }


def _static_downloads() -> list[tuple[str, Path]]:
    tasks: list[tuple[str, Path]] = []
    for year, month in ((2023, 8), (2023, 9), (2023, 10), (2023, 11), (2023, 12), (2024, 1), (2024, 2)):
        name = f"fhvhv_tripdata_{year:04d}-{month:02d}.parquet"
        tasks.append(
            (
                f"https://d37ci6vzurychx.cloudfront.net/trip-data/{name}",
                RAW_VALIDATION / "nyc_tlc" / name,
            )
        )
    tasks.extend(
        [
            (
                "https://www.nyc.gov/assets/tlc/downloads/csv/data_reports_monthly.csv",
                RAW_VALIDATION / "nyc_tlc" / "data_reports_monthly.csv",
            ),
            (
                "https://www.ncei.noaa.gov/data/global-hourly/access/2024/72503014732.csv",
                DATA / "raw" / "weather" / "noaa" / "nyc_laguardia_2024.csv",
            ),
            (
                "https://www.ncei.noaa.gov/data/global-hourly/access/2023/72530094846.csv",
                DATA / "raw" / "weather" / "noaa" / "chicago_ohare_2023.csv",
            ),
            (
                "https://www.ncei.noaa.gov/data/global-hourly/access/2024/72530094846.csv",
                DATA / "raw" / "weather" / "noaa" / "chicago_ohare_2024.csv",
            ),
            (
                "https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/USW00014732.csv",
                RAW_VALIDATION / "ghcnd" / "USW00014732.csv",
            ),
            (
                "https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/USW00094846.csv",
                RAW_VALIDATION / "ghcnd" / "USW00094846.csv",
            ),
        ]
    )
    return tasks


def _json_get(url: str, *, retries: int = 3) -> Any:
    token = os.environ.get("SOCRATA_APP_TOKEN")
    headers = {"X-App-Token": token} if token else None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(_request(url, headers=headers), timeout=120) as response:
                return json.load(response)
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            _log(f"JSON retry attempt={attempt} url={url}: {exc}")
            if attempt == retries:
                raise
            time.sleep(min(30, 2**attempt))


def _soql(params: dict[str, str]) -> Any:
    return _json_get(f"{CHICAGO_API}?{urllib.parse.urlencode(params)}")


def _days(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def _chicago_downloads() -> list[dict[str, Any]]:
    root = RAW_VALIDATION / "chicago_tnp"
    root.mkdir(parents=True, exist_ok=True)
    metadata = _json_get(CHICAGO_META)
    columns = {column.get("fieldName") for column in metadata.get("columns", [])}
    missing = sorted(CHICAGO_COLUMNS - columns)
    if missing:
        raise RuntimeError(f"Chicago dataset is missing registered columns: {missing}")
    _write_json(root / "metadata.json", metadata)

    start = date(2023, 9, 1)
    end = date(2024, 2, 29)
    start_text = start.isoformat() + "T00:00:00.000"
    after_text = (end + timedelta(days=1)).isoformat() + "T00:00:00.000"
    window = f"trip_start_timestamp >= '{start_text}' AND trip_start_timestamp < '{after_text}'"
    aggregate_specs = {
        "daily.json": {
            "$select": (
                "date_trunc_ymd(trip_start_timestamp) as day,count(*) as trip_count,"
                "avg(trip_miles) as mean_trip_miles,avg(trip_seconds) as mean_trip_seconds,"
                "avg(fare) as mean_fare,avg(additional_charges) as mean_additional_charges"
            ),
            "$where": window,
            "$group": "date_trunc_ymd(trip_start_timestamp)",
            "$order": "day",
            "$limit": "50000",
        },
        "daily_airport.json": {
            "$select": "date_trunc_ymd(trip_start_timestamp) as day,count(*) as trip_count",
            "$where": (
                f"({window}) AND (pickup_community_area in (56,76) "
                "OR dropoff_community_area in (56,76))"
            ),
            "$group": "date_trunc_ymd(trip_start_timestamp)",
            "$order": "day",
            "$limit": "50000",
        },
        "hourly.json": {
            "$select": "date_trunc_ymdh(trip_start_timestamp) as hour,count(*) as trip_count",
            "$where": window,
            "$group": "date_trunc_ymdh(trip_start_timestamp)",
            "$order": "hour",
            "$limit": "50000",
        },
    }
    products: list[dict[str, Any]] = []
    for name, params in aggregate_specs.items():
        path = root / name
        if not path.exists():
            _write_json(path, _soql(params))
        products.append({"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)})
        _log(f"Chicago aggregate complete: {name}")

    sample_root = root / "samples"
    for day in _days(start, end):
        path = sample_root / f"{day.isoformat()}.json"
        if not path.exists():
            next_day = day + timedelta(days=1)
            rows = _soql(
                {
                    "$where": (
                        f"trip_start_timestamp >= '{day.isoformat()}T00:00:00.000' AND "
                        f"trip_start_timestamp < '{next_day.isoformat()}T00:00:00.000'"
                    ),
                    "$order": "trip_id",
                    "$limit": "1500",
                }
            )
            _write_json(path, rows)
        products.append({"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)})
        if day.day == 1 or day == end:
            _log(f"Chicago samples through {day.isoformat()}")
    return products


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-chicago", action="store_true")
    parser.add_argument("--finalize-existing", action="store_true",
                        help="write a manifest from completed files after an exhausted source job")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("--workers must be from 1 to 8")

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    _log("D1 downloads finalize started" if args.finalize_existing else "D1 downloads started")
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    tasks = _static_downloads()
    if args.finalize_existing:
        for url, destination in tasks:
            if destination.is_file() and destination.stat().st_size:
                records.append({"url": url, "path": str(destination), "bytes": destination.stat().st_size,
                                "sha256": _sha256(destination), "downloaded_utc": None})
            else:
                failures.append({"source": url, "path": str(destination), "reason": "missing after D1 attempts"})
        if not args.skip_chicago:
            failures.append({"source": CHICAGO_API, "dataset": CHICAGO_DATASET,
                             "reason": "aggregate query timed out on all 3 attempts"})
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_download, url, destination): (url, destination) for url, destination in tasks}
            for future in as_completed(futures):
                try:
                    records.append(future.result())
                except Exception as exc:
                    url, destination = futures[future]
                    failures.append({"source": url, "path": str(destination), "reason": repr(exc)})
        if not args.skip_chicago:
            try:
                records.extend(_chicago_downloads())
            except Exception as exc:
                failures.append({"source": CHICAGO_API, "dataset": CHICAGO_DATASET, "reason": repr(exc)})

    boston = DATA / "raw" / "rides" / "boston_kaggle" / "rideshare_kaggle.csv"
    if not boston.exists():
        raise FileNotFoundError(f"registered Boston data are missing: {boston}")
    records.append(
        {
            "url": "on_disk",
            "path": str(boston),
            "bytes": boston.stat().st_size,
            "sha256": _sha256(boston),
        }
    )
    manifest = {
        "protocol": "CRISP-Ride_v18_Last_Revision_20261113 Section 6 D1",
        "state": "complete_with_failures" if failures else "complete",
        "completed_utc": _utc_now(),
        "records": sorted(records, key=lambda item: item["path"]),
        "failures": failures,
    }
    _write_json(RUN_ROOT / "d1_download_manifest.json", manifest)
    _log(f"D1 downloads complete: {len(records)} records, {len(failures)} failed sources")
    return 0


if __name__ == "__main__":
    sys.exit(main())
