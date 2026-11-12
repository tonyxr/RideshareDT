#!/usr/bin/env python3
"""Run the registered V18 P2 calibration and nested mechanism table.

The workflow is content-addressed and resumable at each simulator job.  It
fits the pay rule, derives the declared NYC wait target, fits fleet and pickup
at N=500, measures city reference earnings, reruns T1.1--T1.3, and builds the
four-arm C0/A1 mechanism table required by the V18 manual.
"""

from __future__ import annotations

import argparse
import concurrent.futures
from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean, median
import subprocess
import sys
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np


BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from city_data_pipeline import NOAAWeatherIndex, resolve_city_spec  # noqa: E402
from diagnostic_logging import atomic_json  # noqa: E402
from scripts import v17_market_checks as t1  # noqa: E402
from scripts import v17_t2_sweep as t2  # noqa: E402
from scripts.v17_config import canonical_hash, load_config as load_v17  # noqa: E402
from scripts.v17_robust_common import set_option  # noqa: E402
from scripts.v18_config import load_config as load_v18  # noqa: E402
from scripts.v18_fit_pay_rule import (  # noqa: E402
    DEFAULT_INPUT, atomic_json as pay_atomic_json, fit, load_rows, transfer,
)
from scripts.v18_fleet_refit import (  # noqa: E402
    FLEET_GRID, NYC_TRIPS_PER_VEHICLE_HOUR, select_fit,
)


STEPS_PER_DAY = 6
BLOCK_HOURS = 24.0 / STEPS_PER_DAY
CUSTOMERS = 500
PRODUCTION_POOL = 20_000
LEGACY_POOL = 2_000
OPENINGS = 2
HORIZON_DAYS = 40
PICKUP_GRID = tuple(round(2.0 + 0.5 * index, 1) for index in range(13))
PAY_DIR = BASE / "calibration_artifacts/v18"
WAIT_SOURCE = ROOT / "data/raw/rides/nyc_kaggle/rideshare_data.parquet"
WEATHER_ROOT = ROOT / "data/raw/weather/noaa"
WAIT_MONTHS = ((2022, 10), (2023, 3), (2023, 6))
WAIT_MAX_SECONDS = 3600
T1_CITIES = ("nyc", "boston")
REFERENCE_CITIES = ("nyc", "boston", "chicago")
T1_JOBS = ("calm_static", "calm_memory", "shortage_static")
EFFECT_CASES = ("C0_calm", "A1_shortage_severe")
EFFECT_RIVALS = (
    "adaptive_best_response_memory",
    "profit_best_response_myopic",
    "profit_opt_mpc_grid",
)
PUBLISHED = t2.cell_spec(1.0, 1.0)
EFFECT_ARMS = ("v17_model", "stock", "pay_rule", "tlc_fleet")


@dataclass(frozen=True)
class Job:
    name: str
    command: tuple[str, ...]
    raw: Path
    manifest: Path
    log: Path
    expected: Mapping[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _option(command: Sequence[str], flag: str) -> str:
    values = list(command)
    index = values.index(flag)
    return str(values[index + 1])


def _boolean_flag(command: Sequence[str], flag: str, enabled: bool) -> list[str]:
    negative = "--no-" + flag.removeprefix("--")
    result = [value for value in map(str, command) if value not in {flag, negative}]
    result.append(flag if enabled else negative)
    return result


def _drop_option(command: Sequence[str], flag: str) -> list[str]:
    result = list(map(str, command))
    if flag not in result:
        return result
    index = result.index(flag)
    if index + 1 >= len(result) or result[index + 1].startswith("--"):
        raise ValueError(f"command option {flag} has no value")
    return result[:index] + result[index + 2:]


def _finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    return True


def _source_identity() -> dict[str, str]:
    sources = dict(t2.source_identity())
    for path in (
        BASE / "scripts/v18_p2.py",
        BASE / "scripts/v18_fit_pay_rule.py",
        BASE / "scripts/v18_fleet_refit.py",
    ):
        sources[str(path.relative_to(BASE))] = _sha256_file(path)
    for city in REFERENCE_CITIES:
        path = PAY_DIR / city / "pay_rule.json"
        if path.is_file():
            sources[str(path.relative_to(BASE))] = _sha256_file(path)
    # Supply selections are generated during this workflow and reference
    # earnings are appended after their measurement jobs finish.  Their
    # effective inputs are already embedded in each command, so hashing the
    # mutable JSON here would make a clean resume reject its own completed
    # reference jobs as stale.
    return sources


def _v18_command(
    command: Sequence[str], *, city: str, raw: Path, report_prefix: Path,
    stock: bool, linear_pay: bool, fleet_scale: float,
    base_pickup_minutes: float, production_population: bool,
    reference_earnings: float | None = None,
) -> tuple[list[str], dict[str, Any]]:
    result = set_option(command, "--policy_sweep_out", raw.resolve())
    result = set_option(result, "--report_prefix", report_prefix)
    result = set_option(result, "--train_customers", CUSTOMERS)
    result = set_option(result, "--eval_customers", CUSTOMERS)
    pool = PRODUCTION_POOL if production_population else LEGACY_POOL
    reference = "run_capacity" if production_population else "fixed"
    result = set_option(result, "--pool", pool)
    result = set_option(result, "--loyalty_pool_size", round(3.6 * CUSTOMERS))
    result = set_option(result, "--scenario_hazards", "off")
    result = set_option(result, "--supply_eta_v3", 0.6)
    if stock:
        result = set_option(result, "--supply_eta_within", 0.2)
    else:
        result = _drop_option(result, "--supply_eta_within")
    result = set_option(result, "--supply_eta_day", 0.4)
    result = set_option(result, "--stock_adjustment_rate", 0.05 if stock else 0.30)
    result = set_option(result, "--stock_floor", 0.4)
    result = set_option(result, "--active_driver_feature_reference", reference)
    result = set_option(result, "--fleet_scale", fleet_scale)
    result = set_option(result, "--base_pickup_minutes", base_pickup_minutes)
    result = set_option(result, "--driver_pay_rule", "linear" if linear_pay else "share")
    result = _boolean_flag(result, "--v3_daily_stock", stock)
    if linear_pay:
        result = set_option(result, "--pay_rule_json", PAY_DIR / city / "pay_rule.json")
    if reference_earnings is not None:
        result = set_option(result, "--supply_reference_earnings", reference_earnings)
    expected = {
        "v3_daily_stock": bool(stock),
        "supply_eta_within": 0.2 if stock else None,
        "supply_eta_day": 0.4,
        "stock_adjustment_rate": 0.05 if stock else 0.30,
        "stock_floor": 0.4,
        "stock_ceiling": 2.0,
        "active_driver_feature_reference": reference,
        "driver_pay_rule": "linear" if linear_pay else "share",
        "base_pickup_minutes": float(base_pickup_minutes),
        "scenario_hazards": "off",
    }
    return result, expected


def _validate_product(path: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not _finite(payload):
        raise ValueError(f"non-finite P2 product: {path}")
    switches = dict(payload.get("model_switches") or {})
    for key, want in expected.items():
        got = switches.get(key)
        if isinstance(want, float):
            if not math.isfinite(float(got)) or abs(float(got) - want) > 1e-12:
                raise ValueError(f"{path}: {key}={got!r}, expected {want!r}")
        elif got != want:
            raise ValueError(f"{path}: {key}={got!r}, expected {want!r}")
    return payload


def _run_job(job: Job, *, sources: Mapping[str, str], config_hash: str) -> str:
    state = t2.run_job(
        job.command, raw=job.raw, manifest=job.manifest, log=job.log,
        config_hash=config_hash, sources=sources, force=False,
        model_expected={"acceptance_shock_wage": "declared", "rate_upper_multiple": 2.5},
    )
    _validate_product(job.raw, job.expected)
    return state


def _run_parallel(
    jobs: Sequence[Job], *, workers: int, output: Path, phase: str,
    completed_before: int = 0, expected_total: int | None = None,
) -> int:
    sources = _source_identity()
    config_hash = canonical_hash(load_v18())
    completed = int(completed_before)
    total = int(expected_total if expected_total is not None else completed + len(jobs))
    atomic_json(output / "status.json", {
        "step": "P2", "state": "running", "phase": phase,
        "completed_jobs": completed, "expected_jobs": total,
        "workers": min(max(1, workers), len(jobs)) if jobs else 0,
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=BASE, text=True
        ).strip(),
    })
    if not jobs:
        return completed
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(max(1, int(workers)), len(jobs))
    ) as executor:
        futures = {
            executor.submit(_run_job, job, sources=sources, config_hash=config_hash): job
            for job in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            future.result()
            completed += 1
            atomic_json(output / "status.json", {
                "step": "P2", "state": "running", "phase": phase,
                "completed_jobs": completed, "expected_jobs": total,
                "last_completed": futures[future].name,
                "workers": min(max(1, workers), len(jobs)),
                "git_head": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=BASE, text=True
                ).strip(),
            })
    return completed


def fit_pay_rules() -> dict[str, Any]:
    nyc = fit(load_rows(DEFAULT_INPUT))
    results = {"nyc": nyc}
    for city in ("boston", "chicago"):
        results[city] = transfer(nyc, city)
    source = {
        "path": str(DEFAULT_INPUT), "size_bytes": DEFAULT_INPUT.stat().st_size,
        "sha256": _sha256_file(DEFAULT_INPUT),
    }
    for city, payload in results.items():
        value = {**payload, "input_source": source}
        pay_atomic_json(PAY_DIR / city / "pay_rule.json", value)
        results[city] = value
    return results


def _histogram_median(counts: np.ndarray) -> float:
    total = int(np.sum(counts))
    if total < 1:
        raise ValueError("wait target contains no observations")
    positions = ((total - 1) // 2, total // 2)
    cumulative = np.cumsum(counts)
    values = [int(np.searchsorted(cumulative, position + 1)) for position in positions]
    return float(sum(values) / 2.0)


def derive_wait_target(output: Path) -> dict[str, Any]:
    target_path = output / "wait_target.json"
    identity = {
        "source_path": str(WAIT_SOURCE),
        "source_size_bytes": WAIT_SOURCE.stat().st_size,
        "source_mtime_ns": WAIT_SOURCE.stat().st_mtime_ns,
        "months": [f"{year:04d}-{month:02d}" for year, month in WAIT_MONTHS],
        "weather": "NOAA nearest observation classified clear",
        "platforms": ["Uber", "Lyft"],
        "valid_wait_seconds": [1, WAIT_MAX_SECONDS],
        "weather_files": [
            {
                "path": str(path), "size_bytes": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in sorted(WEATHER_ROOT.glob("nyc_laguardia_*.csv"))
        ],
    }
    if target_path.is_file():
        cached = json.loads(target_path.read_text(encoding="utf-8"))
        if cached.get("identity") != identity:
            raise ValueError("stale P2 wait-target cache has a different source identity")
        return cached
    import pyarrow.parquet as pq

    spec = resolve_city_spec("nyc")
    weather = NOAAWeatherIndex.from_directory("nyc", WEATHER_ROOT)
    local_tz = ZoneInfo(spec.timezone)
    dates: list[date] = []
    for year, month in WAIT_MONTHS:
        current = date(year, month, 1)
        while current.month == month:
            dates.append(current)
            current = date.fromordinal(current.toordinal() + 1)
    epoch = date(1970, 1, 1)
    clear_keys = set()
    for day in dates:
        for hour in range(24):
            observation = weather.nearest(datetime(
                day.year, day.month, day.day, hour, 0, tzinfo=local_tz
            ))
            if observation is not None and observation.weather == "clear":
                clear_keys.add((day.toordinal() - epoch.toordinal()) * 24 + hour)
    histograms = np.zeros((6, WAIT_MAX_SECONDS + 1), dtype=np.int64)
    parquet = pq.ParquetFile(WAIT_SOURCE)
    batches = 0
    for batch in parquet.iter_batches(
        batch_size=1_000_000,
        columns=("date", "hour_of_day", "request_to_pickup"),
    ):
        batches += 1
        days = batch.column(0).to_numpy(zero_copy_only=False).astype("datetime64[D]").astype(np.int64)
        hours = batch.column(1).to_numpy(zero_copy_only=False).astype(np.int16)
        waits = batch.column(2).to_numpy(zero_copy_only=False).astype(float)
        keys = days * 24 + hours
        mask = (
            np.isfinite(waits)
            & (waits >= 1.0)
            & (waits <= WAIT_MAX_SECONDS)
            & np.isin(keys, tuple(clear_keys))
        )
        if not np.any(mask):
            continue
        valid_waits = np.rint(waits[mask]).astype(np.int32)
        valid_blocks = (hours[mask] // 4).astype(np.int8)
        for block in range(6):
            values = valid_waits[valid_blocks == block]
            if len(values):
                histograms[block] += np.bincount(
                    values, minlength=WAIT_MAX_SECONDS + 1
                )[:WAIT_MAX_SECONDS + 1]
    block_rows = []
    for block in range(6):
        block_rows.append({
            "block": block,
            "hours": [4 * block, 4 * block + 4],
            "observations": int(np.sum(histograms[block])),
            "median_wait_minutes": _histogram_median(histograms[block]) / 60.0,
        })
    result = {
        "schema_version": "v18-nyc-wait-target-v1",
        "identity": identity,
        "source_url": "https://www.kaggle.com/datasets/aaronweymouth/nyc-rideshare-raw-data",
        "weather_source_url": "https://www.ncei.noaa.gov/data/global-hourly/access/",
        "batches_scanned": batches,
        "observations": int(np.sum(histograms)),
        "median_wait_minutes": _histogram_median(np.sum(histograms, axis=0)) / 60.0,
        "by_four_hour_block": block_rows,
    }
    atomic_json(target_path, result)
    return result


def _measurement(
    payload: Mapping[str, Any], spec: str, *, stage: str,
    fleet_scale: float, base_pickup_minutes: float,
) -> dict[str, Any]:
    rows = list(payload["traces"][spec])
    if not rows:
        raise ValueError("P2 calibration product has no trace rows")
    completed = sum(CUSTOMERS * float(row["own_share"]) for row in rows)
    online_hours = sum(BLOCK_HOURS * float(row["active_drivers_firm1"]) for row in rows)
    result = {
        "stage": stage,
        "fleet_scale": float(fleet_scale),
        "base_pickup_minutes": float(base_pickup_minutes),
        "trips_per_online_driver_hour": completed / max(online_hours, 1e-12),
        "median_wait_minutes": median(float(row["wait"]) for row in rows),
        "fulfillment": mean(float(row["fulfillment"]) for row in rows),
        "reference_earnings_per_hour": float(payload["v26r"]["supply_reference_earnings"]),
        "trace_rows": len(rows),
    }
    if not _finite(result):
        raise ValueError(f"non-finite P2 calibration measurement: {result}")
    return result


def _calibration_job(
    config: Mapping[str, Any], output: Path, *, name: str, city: str,
    fleet_scale: float, pickup: float,
) -> Job:
    work = output / "calibration" / name
    command = t1.market_command(
        config, city=city, job="calm_static", output=work,
        openings=OPENINGS, customers=CUSTOMERS, pool=PRODUCTION_POOL,
    )
    raw = work / "raw.json"
    command = set_option(command, "--policy_sweep", t1.CARD)
    command = set_option(command, "--policy_sweep_horizon", HORIZON_DAYS * STEPS_PER_DAY)
    command, expected = _v18_command(
        command, city=city, raw=raw, report_prefix=work / "diagnostics",
        stock=True, linear_pay=True, fleet_scale=fleet_scale,
        base_pickup_minutes=pickup, production_population=True,
    )
    return Job(
        name=name, command=tuple(command), raw=raw,
        manifest=work / "manifest.json", log=work / "job.log", expected=expected,
    )


def run_supply_calibration(
    config: Mapping[str, Any], output: Path, *, workers: int,
) -> dict[str, Any]:
    wait_target = derive_wait_target(output)
    fleet_jobs = [
        _calibration_job(
            config, output, name=f"fleet_{scale:.1f}", city="nyc",
            fleet_scale=scale, pickup=3.5,
        )
        for scale in FLEET_GRID
    ]
    _run_parallel(fleet_jobs, workers=workers, output=output, phase="fleet_grid")
    measurements = [
        _measurement(
            _validate_product(job.raw, job.expected), t1.CARD,
            stage="fleet", fleet_scale=float(_option(job.command, "--fleet_scale")),
            base_pickup_minutes=3.5,
        )
        for job in fleet_jobs
    ]
    provisional = min(
        measurements,
        key=lambda row: (
            abs(float(row["trips_per_online_driver_hour"]) - NYC_TRIPS_PER_VEHICLE_HOUR),
            float(row["fleet_scale"]),
        ),
    )
    selected_scale = float(provisional["fleet_scale"])
    pickup_jobs = [
        _calibration_job(
            config, output, name=f"pickup_{pickup:.1f}", city="nyc",
            fleet_scale=selected_scale, pickup=pickup,
        )
        for pickup in PICKUP_GRID
    ]
    _run_parallel(
        pickup_jobs, workers=workers, output=output, phase="pickup_grid",
        completed_before=len(fleet_jobs),
        expected_total=len(fleet_jobs) + len(pickup_jobs),
    )
    measurements.extend(
        _measurement(
            _validate_product(job.raw, job.expected), t1.CARD,
            stage="pickup", fleet_scale=selected_scale,
            base_pickup_minutes=float(_option(job.command, "--base_pickup_minutes")),
        )
        for job in pickup_jobs
    )
    selected = select_fit(
        measurements,
        target_trips_per_hour=NYC_TRIPS_PER_VEHICLE_HOUR,
        target_median_wait=float(wait_target["median_wait_minutes"]),
    )
    selected.update({
        "city": "nyc",
        "source_urls": {
            "trips_per_vehicle_hour": "https://www.nyc.gov/assets/tlc/downloads/csv/data_reports_monthly.csv",
            "wait": wait_target["source_url"],
            "weather": wait_target["weather_source_url"],
        },
        "months": ["2022-10", "2023-03", "2023-06"],
        "wait_target": wait_target,
    })
    atomic_json(PAY_DIR / "nyc/supply.json", selected)
    baseline = {"nyc": 1.2, "boston": 1.0, "chicago": 1.1}
    ratio = float(selected["fleet_scale"]) / baseline["nyc"]
    for city in ("boston", "chicago"):
        transferred = {
            "schema_version": "v18-supply-calibration-v1",
            "city": city,
            "source": "transferred_from_nyc",
            "transfer_rule": "legacy city fleet scale multiplied by NYC fitted/legacy ratio",
            "nyc_fitted_to_legacy_ratio": ratio,
            "fleet_scale": baseline[city] * ratio,
            "base_pickup_minutes": float(selected["base_pickup_minutes"]),
            "nyc_source": {
                "fleet_scale": selected["fleet_scale"],
                "base_pickup_minutes": selected["base_pickup_minutes"],
            },
        }
        atomic_json(PAY_DIR / city / "supply.json", transferred)
    return {city: json.loads((PAY_DIR / city / "supply.json").read_text()) for city in REFERENCE_CITIES}


def run_reference_earnings(
    config: Mapping[str, Any], output: Path, *, workers: int,
) -> dict[str, Any]:
    jobs = []
    for city in REFERENCE_CITIES:
        supply = json.loads((PAY_DIR / city / "supply.json").read_text())
        jobs.append(_calibration_job(
            config, output, name=f"reference_{city}", city=city,
            fleet_scale=float(supply["fleet_scale"]),
            pickup=float(supply["base_pickup_minutes"]),
        ))
    _run_parallel(jobs, workers=workers, output=output, phase="reference_earnings")
    result = {}
    for city, job in zip(REFERENCE_CITIES, jobs):
        payload = _validate_product(job.raw, job.expected)
        value = float(payload["v26r"]["supply_reference_earnings"])
        supply_path = PAY_DIR / city / "supply.json"
        supply = json.loads(supply_path.read_text())
        supply["reference_earnings_per_hour"] = value
        supply["reference_earnings_contract"] = "measured_calm_then_fixed"
        atomic_json(supply_path, supply)
        result[city] = value
    return result


def _t1_job(
    config: Mapping[str, Any], output: Path, *, city: str, job_name: str,
) -> Job:
    work = output / "market_checks" / city / job_name
    command = t1.market_command(
        config, city=city, job=job_name, output=work,
        openings=6, customers=CUSTOMERS, pool=PRODUCTION_POOL,
    )
    supply = json.loads((PAY_DIR / city / "supply.json").read_text())
    raw = work / "raw.json"
    command, expected = _v18_command(
        command, city=city, raw=raw, report_prefix=work / "diagnostics",
        stock=True, linear_pay=True, fleet_scale=float(supply["fleet_scale"]),
        base_pickup_minutes=float(supply["base_pickup_minutes"]),
        production_population=True,
        reference_earnings=float(supply["reference_earnings_per_hour"]),
    )
    return Job(
        name=f"market_{city}_{job_name}", command=tuple(command), raw=raw,
        manifest=work / "manifest.json", log=work / "job.log", expected=expected,
    )


def run_market_checks(
    config: Mapping[str, Any], output: Path, *, workers: int,
) -> dict[str, Any]:
    jobs = [_t1_job(config, output, city=city, job_name=name)
            for city in T1_CITIES for name in T1_JOBS]
    _run_parallel(jobs, workers=workers, output=output, phase="market_checks")
    by_key = {
        (city, name): _validate_product(job.raw, job.expected)
        for (city, name), job in zip(
            ((city, name) for city in T1_CITIES for name in T1_JOBS), jobs
        )
    }
    cities = {}
    for city in T1_CITIES:
        calm = by_key[(city, "calm_static")]
        memory = by_key[(city, "calm_memory")]
        shortage = by_key[(city, "shortage_static")]
        check11 = t1._t11(city, calm, CUSTOMERS)
        reality = t1.quote_reality_gap(config, city, rows=1000)
        gap = abs(float(reality["v17_vs_v15d_relative_gap"]))
        check11.update(
            v15d_card_mean_fare=float(reality["v15d_card_mean"]),
            fare_relative_gap=gap,
            fare_pass=gap <= 0.03,
        )
        retaliation = [
            row for row in memory["traces"][t1.UNDERCUT]
            if 10 <= int(row["step"]) // STEPS_PER_DAY < 25
            and str(row["rival_memory_state"]) == "retaliation"
        ]
        post = [
            row for row in memory["traces"][t1.UNDERCUT]
            if 15 <= int(row["step"]) // STEPS_PER_DAY < 25
            and str(row["rival_memory_state"]) == "retaliation"
        ]
        check12 = {
            "retaliation_steps": len(retaliation),
            "post_undercut_retaliation_steps": len(post),
            "rival_level_effect": t1.paired_effect(
                memory, t1.UNDERCUT, t1.CARD, "firm2_level",
                start_day=10, end_day=25,
            ),
            "pass": bool(retaliation and post),
        }
        drivers = t1.paired_effect(
            shortage, t1.SHORTAGE_SURGE, t1.CARD, "active_drivers_firm1",
            start_day=14, end_day=20,
        )
        fulfillment = t1.paired_effect(
            shortage, t1.SHORTAGE_SURGE, t1.CARD, "fulfillment",
            start_day=14, end_day=20,
        )
        check13 = {
            "active_drivers": drivers, "fulfillment": fulfillment,
            "pass": bool(
                float(drivers["mean"]) > 0 and float(drivers["t"]) >= 3
                and float(fulfillment["mean"]) > 0 and float(fulfillment["t"]) >= 3
            ),
        }
        cities[city] = {
            "T1.1": check11, "T1.2": check12, "T1.3": check13,
            "scientific_expectations_pass": bool(
                check11["fulfillment_pass"] and check11["fare_pass"]
                and check12["pass"] and check13["pass"]
            ),
        }
    report = {"cities": cities, "pass": True, "gated": False}
    atomic_json(output / "market_checks/report.json", report)
    return report


def _effect_specs() -> tuple[str, ...]:
    selected = json.loads(
        (BASE / "runs/v17_results_comparators/nyc/selected.json").read_text()
    )
    return (
        PUBLISHED,
        str(selected["best_fixed_cell"]["spec"]),
        t2.cell_spec(1.5, 1.5),
        t2.cell_spec(2.0, 2.0),
        t2.cell_spec(2.5, 2.5),
    )


def _effect_job(
    config: Mapping[str, Any], output: Path, *, arm: str, case: str, rival: str,
) -> Job:
    work = output / "effects" / arm / case / rival
    raw = work / "raw.json"
    command = t2.sweep_command(
        config, city="nyc", case=case, rival=rival, output=work,
        openings=2, customers=CUSTOMERS, pool=LEGACY_POOL, days=HORIZON_DAYS,
    )
    command = set_option(command, "--policy_sweep", ";".join(_effect_specs()))
    supply = json.loads((PAY_DIR / "nyc/supply.json").read_text())
    settings = {
        "v17_model": (False, False, 1.2, 3.5),
        "stock": (True, False, 1.2, 3.5),
        "pay_rule": (True, True, 1.2, 3.5),
        "tlc_fleet": (
            True, True, float(supply["fleet_scale"]),
            float(supply["base_pickup_minutes"]),
        ),
    }[arm]
    command, expected = _v18_command(
        command, city="nyc", raw=raw, report_prefix=work / "diagnostics",
        stock=settings[0], linear_pay=settings[1], fleet_scale=settings[2],
        base_pickup_minutes=settings[3], production_population=False,
        reference_earnings=(
            float(supply["reference_earnings_per_hour"])
            if arm == "tlc_fleet" else None
        ),
    )
    return Job(
        name=f"effect|{arm}|{case}|{rival}", command=tuple(command), raw=raw,
        manifest=work / "manifest.json", log=work / "job.log", expected=expected,
    )


def _effect_summary(jobs: Sequence[Job]) -> dict[str, Any]:
    specs = _effect_specs()
    labels = ("published", "v17_best_fixed", "1.5", "2.0", "2.5")
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    stock_rows: list[Mapping[str, Any]] = []
    for job in jobs:
        _, arm, case, rival = job.name.split("|", 3)
        payload = _validate_product(job.raw, job.expected)
        for label, spec in zip(labels, specs):
            key = (arm, case, label)
            grouped.setdefault(key, []).append({"payload": payload, "spec": spec})
            if arm == "tlc_fleet" and case == "C0_calm" and label == "published":
                stock_rows.extend(payload["traces"][spec])
    rows = []
    for (arm, case, label), products in sorted(grouped.items()):
        results = []
        traces = []
        for item in products:
            results.extend(dict(item["payload"]["results"][item["spec"]]).values())
            traces.extend(item["payload"]["traces"][item["spec"]])
        window = [row for row in traces if 10 <= int(row["step"]) // STEPS_PER_DAY <= 15]
        row = {
            "arm": arm, "case": case, "card": label,
            "objective": mean(float(item["objective"]) for item in results),
            "own_profit": mean(float(item["own_profit"]) for item in results),
            "rival_profit": mean(float(item["rival_profit"]) for item in results),
            "fulfillment_days_10_15": mean(float(item["fulfillment"]) for item in window),
            "drivers_online_days_10_15": mean(float(item["active_drivers_firm1"]) for item in window),
            "stock_days_10_15": mean(float(item["stock_firm1"]) for item in window),
            "rollouts": len(results), "trace_rows": len(traces),
        }
        if not _finite(row):
            raise ValueError(f"non-finite P2 effect row: {row}")
        rows.append(row)
    after_day10 = [
        float(row["stock_firm1"]) for row in stock_rows
        if int(row["step"]) // STEPS_PER_DAY >= 10
    ]
    gate = {
        "minimum": min(after_day10), "maximum": max(after_day10),
        "maximum_absolute_drift": max(abs(value - 1.0) for value in after_day10),
    }
    gate["pass"] = bool(gate["maximum_absolute_drift"] <= 0.03)
    report = {
        "arms": list(EFFECT_ARMS), "cases": list(EFFECT_CASES),
        "rivals": list(EFFECT_RIVALS), "openings": 2,
        "customers": CUSTOMERS, "rows": rows,
        "calm_published_stock_gate": gate,
    }
    return report


def run_effects(
    config: Mapping[str, Any], output: Path, *, workers: int,
) -> dict[str, Any]:
    jobs = [
        _effect_job(config, output, arm=arm, case=case, rival=rival)
        for arm in EFFECT_ARMS for case in EFFECT_CASES for rival in EFFECT_RIVALS
    ]
    _run_parallel(jobs, workers=workers, output=output, phase="effect_table")
    report = _effect_summary(jobs)
    atomic_json(output / "effects/report.json", report)
    if not report["calm_published_stock_gate"]["pass"]:
        raise ValueError("P2 calm published-card stock drift exceeded 3%")
    return report


def finalize(output: Path) -> dict[str, Any]:
    supplies = {
        city: json.loads((PAY_DIR / city / "supply.json").read_text())
        for city in REFERENCE_CITIES
    }
    pay_rules = {
        city: json.loads((PAY_DIR / city / "pay_rule.json").read_text())
        for city in REFERENCE_CITIES
    }
    market = json.loads((output / "market_checks/report.json").read_text())
    effects = json.loads((output / "effects/report.json").read_text())
    decision = {
        "fitted_supply": {
            city: {
                "fleet_scale": float(supplies[city]["fleet_scale"]),
                "base_pickup_minutes": float(supplies[city]["base_pickup_minutes"]),
                "reference_earnings_per_hour": float(
                    supplies[city]["reference_earnings_per_hour"]
                ),
            }
            for city in REFERENCE_CITIES
        }
    }
    report = {
        "step": "P2", "state": "complete",
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=BASE, text=True
        ).strip(),
        "decision": decision,
        "pay_rules": pay_rules,
        "supply": supplies,
        "market_checks": market,
        "effect_table": effects,
        "stop_conditions": {
            "finite": True,
            "calm_published_stock_within_3_percent": bool(
                effects["calm_published_stock_gate"]["pass"]
            ),
        },
    }
    if not _finite(report) or not all(report["stop_conditions"].values()):
        raise ValueError("P2 final report failed a stop condition")
    atomic_json(output / "report.json", report)
    markdown = (
        "# V18 P2 calibration and effect table\n\n"
        f"State: complete. Calm published-card stock gate: PASS "
        f"(maximum drift {effects['calm_published_stock_gate']['maximum_absolute_drift']:.4f}).\n\n"
        "T1.1--T1.3 outcomes are reported diagnostics and are not P2 stop gates.\n"
    )
    (output / "report.md").write_text(markdown, encoding="utf-8")
    atomic_json(output / "status.json", {
        "step": "P2", "state": "completed", "pass": True,
        "phase": "complete", "report": str(output / "report.json"),
        "git_head": report["git_head"],
    })
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("all", "pay", "wait", "supply", "reference", "market", "effects", "finalize"),
        default="all",
    )
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument(
        "--output", type=Path,
        default=BASE / "runs/v18/preflight/P2",
    )
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= int(args.workers) <= 10:
        raise ValueError("P2 workers must be in [1, 10]")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.plan_only:
        print(json.dumps({
            "step": "P2", "workers": args.workers,
            "fleet_candidates": list(FLEET_GRID),
            "pickup_candidates": list(PICKUP_GRID),
            "reference_jobs": len(REFERENCE_CITIES),
            "market_jobs": len(T1_CITIES) * len(T1_JOBS),
            "effect_jobs": len(EFFECT_ARMS) * len(EFFECT_CASES) * len(EFFECT_RIVALS),
            "effect_rollouts": len(EFFECT_ARMS) * len(EFFECT_CASES)
            * len(EFFECT_RIVALS) * len(_effect_specs()) * 2,
        }, indent=2, sort_keys=True))
        return 0
    try:
        config = load_v17()
        if args.stage in {"all", "pay"}:
            fit_pay_rules()
        if args.stage in {"all", "wait"}:
            derive_wait_target(output)
        if args.stage in {"all", "supply"}:
            run_supply_calibration(config, output, workers=args.workers)
        if args.stage in {"all", "reference"}:
            run_reference_earnings(config, output, workers=args.workers)
        if args.stage in {"all", "market"}:
            run_market_checks(config, output, workers=args.workers)
        if args.stage in {"all", "effects"}:
            run_effects(config, output, workers=args.workers)
        result = finalize(output) if args.stage in {"all", "finalize"} else {
            "step": "P2", "state": "stage_complete", "stage": args.stage,
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except BaseException as exc:
        previous = {}
        status = output / "status.json"
        if status.is_file():
            previous = json.loads(status.read_text())
        atomic_json(status, {
            **previous, "step": "P2", "state": "failed",
            "error": repr(exc),
        })
        raise


if __name__ == "__main__":
    raise SystemExit(main())
