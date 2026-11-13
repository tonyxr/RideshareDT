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
import shutil
from statistics import mean, median
import subprocess
import sys
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import yaml


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
from v17_surge_decay import surge_decay_spec  # noqa: E402
from driver_supply import (  # noqa: E402
    DEFAULT_HOUR_PROFILE,
    DriverSupplyLayer,
    normalized_hour_profile,
)


STEPS_PER_DAY = 6
BLOCK_HOURS = 24.0 / STEPS_PER_DAY
CUSTOMERS = 500
PRODUCTION_POOL = 20_000
LEGACY_POOL = 2_000
OPENINGS = 2
HORIZON_DAYS = 40
CALIBRATION_DAYS = 28
CALIBRATION_OPENINGS = 2
CONFIRMATION_OPENINGS = 3
FLEET_GRID = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
TRANSFER_FLEET_GRID = (0.6, 0.8, 1.0, 1.2, 1.4, 1.6)
PICKUP_GRID = (1.0, 1.25, 1.5, 1.75, 2.0)
PROFILE_BLOCKS = tuple(range(6))
PROFILE_EXPONENT = 0.7
PROFILE_TOLERANCE_MINUTES = 0.4
PROFILE_MAX_ITERATIONS = 4
TARGET_UTILIZATION = 0.58
UTILIZATION_BAND = (0.52, 0.64)
WAIT_TOLERANCE = 0.50
MAX_TRIPS_PER_ONLINE_HOUR = 1.78
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
EFFECT_ARMS = ("v17_model", "stock", "pay_rule", "calibrated")
KAPPA_CANDIDATES = (0.03, 0.05, 0.08)
P2_GATE_REPAIR_PARENT_SOURCE_SHA256 = (
    "15c2cef005fed811df1d549a77eba559de53facf54b662a308c7c54eba3f9b30"
)
P2_GATE_REPAIR_PARENT_CORE_SHA256 = (
    "afdb716341ce8c720d9e40cc15e843601719c262ca4e0a9ad9babf603bdaaff3"
)


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
    base_active_drivers: int = 0,
    stock_kappa: float = 0.05,
    hour_profile_json: Path | None = None,
) -> tuple[list[str], dict[str, Any]]:
    result = set_option(command, "--policy_sweep_out", raw.resolve())
    result = _boolean_flag(result, "--policy_sweep_supply_totals", True)
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
        result = set_option(result, "--supply_eta_within", 0.6)
        result = set_option(result, "--stock_rule", "integral")
    else:
        result = _drop_option(result, "--supply_eta_within")
        result = set_option(result, "--stock_rule", "power")
    result = set_option(
        result, "--stock_adjustment_rate", stock_kappa if stock else 0.30
    )
    result = set_option(result, "--stock_floor", 0.4)
    result = set_option(result, "--active_driver_feature_reference", reference)
    result = set_option(result, "--fleet_scale", fleet_scale)
    result = set_option(result, "--base_pickup_minutes", base_pickup_minutes)
    result = set_option(
        result, "--supply_base_active_drivers", int(base_active_drivers)
    )
    if hour_profile_json is not None:
        result = set_option(
            result, "--supply_hour_profile_json", Path(hour_profile_json).resolve()
        )
    result = set_option(result, "--driver_pay_rule", "linear" if linear_pay else "share")
    result = _boolean_flag(result, "--v3_daily_stock", stock)
    if linear_pay:
        result = set_option(result, "--pay_rule_json", PAY_DIR / city / "pay_rule.json")
    if reference_earnings is not None:
        result = set_option(result, "--supply_reference_earnings", reference_earnings)
    expected = {
        "v3_daily_stock": bool(stock),
        "stock_rule": "integral" if stock else "power",
        "supply_eta_within": 0.6 if stock else None,
        "supply_eta_day": None if stock else 0.4,
        "stock_adjustment_rate": stock_kappa if stock else 0.30,
        "stock_floor": 0.4,
        "stock_ceiling": 2.0,
        "active_driver_feature_reference": reference,
        "driver_pay_rule": "linear" if linear_pay else "share",
        "base_pickup_minutes": float(base_pickup_minutes),
        "scenario_hazards": "off",
    }
    if int(base_active_drivers) > 0:
        expected["base_active_drivers"] = int(round(
            int(base_active_drivers) * float(fleet_scale)
        ))
    if hour_profile_json is not None:
        payload = json.loads(Path(hour_profile_json).read_text(encoding="utf-8"))
        profile = payload.get("hour_profile", payload)
        normalized = normalized_hour_profile({
            int(hour): float(value) for hour, value in profile.items()
        })
        expected["hour_profile"] = [
            float(normalized[hour]) for hour in range(24)
        ]
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
    allow_capacity_infeasible: bool = False,
) -> tuple[int, list[dict[str, str]]]:
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
        return completed, []
    infeasible: list[dict[str, str]] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(max(1, int(workers)), len(jobs))
    ) as executor:
        futures = {
            executor.submit(_run_job, job, sources=sources, config_hash=config_hash): job
            for job in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            job = futures[future]
            try:
                future.result()
            except RuntimeError as exc:
                if not (
                    allow_capacity_infeasible
                    and "capacity target unreachable" in str(exc)
                ):
                    raise
                infeasible.append({
                    "job": job.name,
                    "reason": "capacity target unreachable for registered diagnostic candidate",
                    "manifest": str(job.manifest),
                    "log": str(job.log),
                })
            else:
                completed += 1
            atomic_json(output / "status.json", {
                "step": "P2", "state": "running", "phase": phase,
                "completed_jobs": completed, "expected_jobs": total,
                "processed_jobs": completed + len(infeasible),
                "last_processed": job.name,
                "infeasible_jobs": infeasible,
                "workers": min(max(1, workers), len(jobs)),
                "git_head": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=BASE, text=True
                ).strip(),
            })
    return completed, infeasible


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
    """Measure the registered P2 quantities from exact batch totals.

    Days are one-indexed in the protocol: days 5--28 are trace days 4--27.
    Both platforms enter every ratio, including the shared Eref helper.
    """
    all_rows = list(payload["traces"][spec])
    rows = [
        row for row in all_rows
        if 5 <= int(row["step"]) // STEPS_PER_DAY + 1 <= 28
    ]
    if not rows:
        raise ValueError("P2 calibration product has no days-5--28 trace rows")
    completed = sum(
        int(row[f"completed_rides_{firm}"])
        for row in rows for firm in ("firm1", "firm2")
    )
    online_minutes = sum(
        float(row[f"online_minutes_{firm}"])
        for row in rows for firm in ("firm1", "firm2")
    )
    passenger_minutes = sum(
        float(row[f"passenger_minutes_{firm}"])
        for row in rows for firm in ("firm1", "firm2")
    )
    driver_pay = sum(
        float(row[f"driver_pay_{firm}"])
        for row in rows for firm in ("firm1", "firm2")
    )
    pickups = [
        float(value)
        for row in rows for firm in ("firm1", "firm2")
        for value in row[f"pickup_minutes_{firm}"]
    ]
    if completed <= 0 or not pickups:
        raise ValueError("P2 calibration completed no rides in days 5--28")
    offered = sum(
        int(row[f"completed_rides_{firm}"])
        + int(row[f"unfulfilled_requests_{firm}"])
        for row in rows for firm in ("firm1", "firm2")
    )
    result = {
        "stage": stage,
        "fleet_scale": float(fleet_scale),
        "base_pickup_minutes": float(base_pickup_minutes),
        # Registered Fix A statistic: passenger minutes divided by online
        # minutes.  Pickup time is explicitly excluded by the fix spec.
        "utilization": passenger_minutes / max(online_minutes, 1e-12),
        "trips_per_online_driver_hour": (
            60.0 * completed / max(online_minutes, 1e-12)
        ),
        "median_wait_minutes": median(pickups),
        "fulfillment": completed / max(1, offered),
        "reference_earnings_per_hour": (
            DriverSupplyLayer.earnings_per_online_hour(
                driver_pay, online_minutes
            )
        ),
        "completed_rides": int(completed),
        "online_minutes": float(online_minutes),
        "passenger_minutes": float(passenger_minutes),
        "driver_pay": float(driver_pay),
        "pickup_observations": len(pickups),
        "trace_rows": len(rows),
    }
    if not _finite(result):
        raise ValueError(f"non-finite P2 calibration measurement: {result}")
    return result


def _block_waits(
    payload: Mapping[str, Any], spec: str,
) -> dict[int, float]:
    """Median completed-request pickup wait in each registered four-hour block."""
    waits: dict[int, list[float]] = {block: [] for block in PROFILE_BLOCKS}
    for row in payload["traces"][spec]:
        day = int(row["step"]) // STEPS_PER_DAY + 1
        if not 5 <= day <= 28:
            continue
        block = int(row["hour"]) // 4
        if block not in waits:
            raise ValueError(f"unexpected calibration hour {row['hour']!r}")
        for firm in ("firm1", "firm2"):
            waits[block].extend(
                map(float, row[f"pickup_minutes_{firm}"])
            )
    missing = [block for block, values in waits.items() if not values]
    if missing:
        raise ValueError(f"missing completed pickup waits for blocks {missing}")
    result = {block: float(median(values)) for block, values in waits.items()}
    if not _finite(result):
        raise ValueError(f"non-finite block waits: {result}")
    return result


def _real_block_waits(wait_target: Mapping[str, Any]) -> dict[int, float]:
    result = {
        int(row["block"]): float(row["median_wait_minutes"])
        for row in wait_target["by_four_hour_block"]
    }
    if set(result) != set(PROFILE_BLOCKS) or not _finite(result):
        raise ValueError("wait target must declare six finite four-hour blocks")
    return result


def _profile_from_factors(
    factors: Mapping[int, float],
) -> dict[int, float]:
    """Apply six block factors while preserving the default daily mean."""
    if set(map(int, factors)) != set(PROFILE_BLOCKS):
        raise ValueError("hour-profile factors must cover blocks 0--5")
    base = {int(hour): float(value) for hour, value in DEFAULT_HOUR_PROFILE.items()}
    raw = {
        hour: base[hour] * float(factors[hour // 4])
        for hour in range(24)
    }
    if any(not math.isfinite(value) or value <= 0.0 for value in raw.values()):
        raise ValueError("hour-profile factors must be finite and positive")
    scale = sum(base.values()) / sum(raw.values())
    return normalized_hour_profile({
        hour: value * scale for hour, value in raw.items()
    })


def _updated_profile_factors(
    factors: Mapping[int, float], simulated: Mapping[int, float],
    target: Mapping[int, float],
) -> tuple[dict[int, float], dict[int, float]]:
    """Revision 1 R1 multiplicative update with daily-mean rescaling."""
    raw = {
        block: float(factors[block])
        * (float(simulated[block]) / float(target[block])) ** PROFILE_EXPONENT
        for block in PROFILE_BLOCKS
    }
    profile = _profile_from_factors(raw)
    base = {int(hour): float(value) for hour, value in DEFAULT_HOUR_PROFILE.items()}
    normalized_factors = {
        block: mean(
            profile[hour] / base[hour]
            for hour in range(block * 4, block * 4 + 4)
        )
        for block in PROFILE_BLOCKS
    }
    return normalized_factors, profile


def _profile_artifact(
    *, city: str, factors: Mapping[int, float],
    profile: Mapping[int, float], target: Mapping[int, float],
    history: Sequence[Mapping[str, Any]], state: str,
) -> dict[str, Any]:
    return {
        "schema_version": "v18-hour-profile-rev1",
        "revision": "CRISP-Ride_v18_P2_Fix_Spec_Rev1_20261113",
        "state": state,
        "city": city,
        "factors_by_four_hour_block": {
            str(block): float(factors[block]) for block in PROFILE_BLOCKS
        },
        "hour_profile": {
            str(hour): float(profile[hour]) for hour in range(24)
        },
        "target_wait_by_four_hour_block": {
            str(block): float(target[block]) for block in PROFILE_BLOCKS
        },
        "history": list(history),
    }


def _axis_bracket(values: Sequence[float], point: float) -> tuple[float, float]:
    ordered = sorted(set(map(float, values)))
    lower = max((value for value in ordered if value <= point), default=ordered[0])
    upper = min((value for value in ordered if value >= point), default=ordered[-1])
    return lower, upper


def _bilinear(
    table: Mapping[tuple[float, float], Mapping[str, Any]],
    fleet: float, pickup: float, key: str,
) -> float:
    fleets = sorted({point[0] for point in table})
    pickups = sorted({point[1] for point in table})
    f0, f1 = _axis_bracket(fleets, fleet)
    p0, p1 = _axis_bracket(pickups, pickup)
    corners = {
        (f, p): float(table[(f, p)][key])
        for f in {f0, f1} for p in {p0, p1}
    }
    if f0 == f1 and p0 == p1:
        return corners[(f0, p0)]
    if f0 == f1:
        weight = (pickup - p0) / max(1e-12, p1 - p0)
        return (1.0 - weight) * corners[(f0, p0)] + weight * corners[(f0, p1)]
    if p0 == p1:
        weight = (fleet - f0) / max(1e-12, f1 - f0)
        return (1.0 - weight) * corners[(f0, p0)] + weight * corners[(f1, p0)]
    wf = (fleet - f0) / (f1 - f0)
    wp = (pickup - p0) / (p1 - p0)
    return (
        (1.0 - wf) * (1.0 - wp) * corners[(f0, p0)]
        + wf * (1.0 - wp) * corners[(f1, p0)]
        + (1.0 - wf) * wp * corners[(f0, p1)]
        + wf * wp * corners[(f1, p1)]
    )


def _select_joint_calibration(
    measurements: Sequence[Mapping[str, Any]], *, target_wait: float,
    allow_no_binding: bool = False,
) -> dict[str, Any]:
    table = {
        (float(row["fleet_scale"]), float(row["base_pickup_minutes"])): row
        for row in measurements
    }
    fleets = sorted({point[0] for point in table})
    pickups = sorted({point[1] for point in table})
    fleet_candidates = np.arange(
        round(20 * fleets[0]), round(20 * fleets[-1]) + 1
    ) / 20.0
    pickup_candidates = np.arange(
        round(4 * pickups[0]), round(4 * pickups[-1]) + 1
    ) / 4.0
    candidates = []
    for fleet in fleet_candidates:
        for pickup in pickup_candidates:
            utilization = _bilinear(table, fleet, pickup, "utilization")
            wait = _bilinear(table, fleet, pickup, "median_wait_minutes")
            trips = _bilinear(
                table, fleet, pickup, "trips_per_online_driver_hour"
            )
            loss = (
                ((utilization - TARGET_UTILIZATION) / 0.06) ** 2
                + ((wait - target_wait) / 0.45) ** 2
            )
            binding = (
                abs(wait - target_wait) <= WAIT_TOLERANCE
                and trips <= MAX_TRIPS_PER_ONLINE_HOUR
            )
            candidates.append({
                "loss": float(loss), "fleet": float(fleet),
                "pickup": float(pickup), "utilization": float(utilization),
                "wait": float(wait), "trips": float(trips),
                "binding": bool(binding),
            })
    feasible = [row for row in candidates if row["binding"]]
    if not feasible and not allow_no_binding:
        raise ValueError(
            "joint calibration has no point satisfying wait and trips/hour"
        )
    pool = feasible or candidates
    reaches_band = [
        row for row in pool if row["utilization"] >= UTILIZATION_BAND[0]
    ]
    if reaches_band:
        chosen = min(reaches_band, key=lambda row: row["loss"])
        fallback = False
    else:
        chosen = min(
            pool,
            key=lambda row: (-row["utilization"], row["loss"]),
        )
        fallback = bool(feasible)
    loss = chosen["loss"]
    fleet = chosen["fleet"]
    pickup = chosen["pickup"]
    utilization = chosen["utilization"]
    wait = chosen["wait"]
    trips = chosen["trips"]
    return {
        "fleet_scale": fleet,
        "base_pickup_minutes": pickup,
        "interpolated": {
            "utilization": utilization,
            "median_wait_minutes": wait,
            "trips_per_online_driver_hour": trips,
        },
        "normalized_squared_target_loss": loss,
        "candidate_count": len(candidates),
        "binding_feasible_candidate_count": len(feasible),
        "binding_targets_pass": bool(chosen["binding"]),
        "utilization_fallback_used": fallback,
        "utilization_gap": max(0.0, UTILIZATION_BAND[0] - utilization),
        "coarse_fleet_bounds": [fleets[0], fleets[-1]],
        "coarse_pickup_bounds": [pickups[0], pickups[-1]],
    }


def _select_transfer_service_level(
    measurements: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Choose the closest registered R4 block-wait fit without NYC gates."""
    candidates = [dict(row) for row in measurements]
    if not candidates or not _finite(candidates):
        raise ValueError("city service-level grid must be nonempty and finite")
    return min(
        candidates,
        key=lambda row: (
            float(row["service_level_loss"]),
            float(row["fleet_scale"]),
        ),
    )


def _calibration_job(
    config: Mapping[str, Any], output: Path, *, name: str, city: str,
    fleet_scale: float, pickup: float, openings: int = CALIBRATION_OPENINGS,
    base_active_drivers: int = 0,
    hour_profile_json: Path | None = None,
) -> Job:
    work = output / "calibration" / name
    command = t1.market_command(
        config, city=city, job="calm_static", output=work,
        openings=openings, customers=CUSTOMERS, pool=PRODUCTION_POOL,
    )
    raw = work / "raw.json"
    command = set_option(command, "--policy_sweep", t1.CARD)
    command = set_option(
        command, "--policy_sweep_horizon", CALIBRATION_DAYS * STEPS_PER_DAY
    )
    command = set_option(command, "--policy_sweep_seed", 20231)
    command = set_option(
        command, "--policy_sweep_opening_tariff_randomization", 0.0
    )
    command, expected = _v18_command(
        command, city=city, raw=raw, report_prefix=work / "diagnostics",
        stock=False, linear_pay=True, fleet_scale=fleet_scale,
        base_pickup_minutes=pickup, production_population=True,
        base_active_drivers=base_active_drivers,
        hour_profile_json=hour_profile_json,
    )
    return Job(
        name=name, command=tuple(command), raw=raw,
        manifest=work / "manifest.json", log=work / "job.log", expected=expected,
    )


def _fit_hour_profile(
    config: Mapping[str, Any], output: Path, *, city: str,
    base_active_drivers: int, fleet_scale: float, pickup: float,
    target_waits: Mapping[int, float], workers: int = 1,
) -> tuple[Path, dict[str, Any], list[Job]]:
    """Fit Revision 1 four-hour availability factors in at most four jobs."""
    del workers  # The registered iterations are sequential by construction.
    profile_root = output / "calibration/hour_profiles"
    factors = {block: 1.0 for block in PROFILE_BLOCKS}
    profile = _profile_from_factors(factors)
    history: list[dict[str, Any]] = []
    jobs: list[Job] = []
    converged = False
    for iteration in range(1, PROFILE_MAX_ITERATIONS + 1):
        profile_path = profile_root / f"{city}_iteration_{iteration}.json"
        atomic_json(profile_path, _profile_artifact(
            city=city, factors=factors, profile=profile,
            target=target_waits, history=history, state="iteration_input",
        ))
        job = _calibration_job(
            config, output,
            name=f"{city}_profile_iteration_{iteration}", city=city,
            fleet_scale=fleet_scale, pickup=pickup,
            openings=CALIBRATION_OPENINGS,
            base_active_drivers=base_active_drivers,
            hour_profile_json=profile_path,
        )
        _run_parallel(
            [job], workers=1, output=output,
            phase=f"{city}_hour_profile_iteration_{iteration}",
        )
        jobs.append(job)
        payload = _validate_product(job.raw, job.expected)
        simulated = _block_waits(payload, t1.CARD)
        gaps = {
            block: float(simulated[block] - target_waits[block])
            for block in PROFILE_BLOCKS
        }
        converged = all(
            abs(gaps[block]) <= PROFILE_TOLERANCE_MINUTES
            for block in PROFILE_BLOCKS
        )
        history.append({
            "iteration": iteration,
            "input_artifact": str(profile_path),
            "factors_by_four_hour_block": {
                str(block): float(factors[block])
                for block in PROFILE_BLOCKS
            },
            "simulated_wait_by_four_hour_block": {
                str(block): float(simulated[block])
                for block in PROFILE_BLOCKS
            },
            "target_wait_by_four_hour_block": {
                str(block): float(target_waits[block])
                for block in PROFILE_BLOCKS
            },
            "gap_minutes_by_four_hour_block": {
                str(block): float(gaps[block])
                for block in PROFILE_BLOCKS
            },
            "within_0.4_minutes_all_blocks": converged,
            "product": str(job.raw),
        })
        if converged:
            break
        factors, profile = _updated_profile_factors(
            factors, simulated, target_waits
        )
    final_path = profile_root / f"{city}.json"
    final_payload = _profile_artifact(
        city=city, factors=factors, profile=profile,
        target=target_waits, history=history,
        state="converged" if converged else "maximum_iterations_reached",
    )
    final_payload["converged"] = converged
    final_payload["iterations"] = len(history)
    atomic_json(final_path, final_payload)
    return final_path, final_payload, jobs


def _archive_old_supply(city: str) -> None:
    path = PAY_DIR / city / "supply.json"
    failed = PAY_DIR / city / "supply_failed_20261112.json"
    if path.is_file() and not failed.exists():
        failed.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, failed)


def _base_anchor(
    config: Mapping[str, Any], output: Path, *, city: str,
) -> tuple[Job, int]:
    job = _calibration_job(
        config, output, name=f"{city}_capacity_anchor", city=city,
        fleet_scale=1.0, pickup=3.0, openings=1,
    )
    _run_parallel([job], workers=1, output=output, phase=f"{city}_capacity_anchor")
    payload = _validate_product(job.raw, job.expected)
    base = int(payload["model_switches"]["base_active_drivers"])
    if base <= 0:
        raise ValueError(f"{city} V26 capacity anchor is not positive")
    return job, base


def _confirm_calibration(
    config: Mapping[str, Any], output: Path, *, city: str,
    base: int, fleet: float, pickup: float, target_wait: float,
    hour_profile_json: Path | None = None,
    block_wait_targets: Mapping[int, float] | None = None,
    binding_targets: bool = True,
) -> tuple[Job, dict[str, Any]]:
    job = _calibration_job(
        config, output, name=f"{city}_confirmation", city=city,
        fleet_scale=fleet, pickup=pickup,
        openings=CONFIRMATION_OPENINGS, base_active_drivers=base,
        hour_profile_json=hour_profile_json,
    )
    _run_parallel([job], workers=1, output=output, phase=f"{city}_confirmation")
    payload = _validate_product(job.raw, job.expected)
    measured = _measurement(
        payload, t1.CARD,
        stage="confirmation", fleet_scale=fleet,
        base_pickup_minutes=pickup,
    )
    if block_wait_targets is not None:
        simulated = _block_waits(payload, t1.CARD)
        measured["wait_by_four_hour_block"] = [
            {
                "block": block,
                "hours": [block * 4, block * 4 + 4],
                "simulated_median_wait_minutes": float(simulated[block]),
                "target_median_wait_minutes": float(block_wait_targets[block]),
                "gap_minutes": float(
                    simulated[block] - block_wait_targets[block]
                ),
            }
            for block in PROFILE_BLOCKS
        ]
    measured["target_checks"] = {
        "median_wait": bool(
            abs(measured["median_wait_minutes"] - target_wait) <= WAIT_TOLERANCE
        ),
        "trips_per_online_driver_hour": bool(
            measured["trips_per_online_driver_hour"]
            <= MAX_TRIPS_PER_ONLINE_HOUR
        ),
    }
    measured["utilization_aim"] = {
        "target": TARGET_UTILIZATION,
        "band": list(UTILIZATION_BAND),
        "within_band": bool(
            UTILIZATION_BAND[0]
            <= measured["utilization"]
            <= UTILIZATION_BAND[1]
        ),
        "utilization_gap": max(
            0.0, UTILIZATION_BAND[0] - measured["utilization"]
        ),
    }
    measured["target_role"] = "binding" if binding_targets else "reported"
    measured["pass"] = (
        all(measured["target_checks"].values()) if binding_targets else True
    )
    return job, measured


def run_supply_calibration(
    config: Mapping[str, Any], output: Path, *, workers: int,
) -> dict[str, Any]:
    wait_target = derive_wait_target(output)
    target_wait = float(wait_target["median_wait_minutes"])
    block_wait_targets = _real_block_waits(wait_target)
    _archive_old_supply("nyc")
    anchor_job, base = _base_anchor(config, output, city="nyc")
    nyc_profile_path, nyc_profile, profile_jobs = _fit_hour_profile(
        config, output, city="nyc", base_active_drivers=base,
        fleet_scale=0.8, pickup=1.5,
        target_waits=block_wait_targets,
    )
    fleets = list(FLEET_GRID)
    pickups = list(PICKUP_GRID)

    def grid_jobs(fleet_values: Sequence[float], pickup_values: Sequence[float], label: str) -> list[Job]:
        return [
            _calibration_job(
                config, output,
                name=f"nyc_{label}_f{fleet:.2f}_p{pickup:.2f}", city="nyc",
                fleet_scale=float(fleet), pickup=float(pickup),
                base_active_drivers=base,
                hour_profile_json=nyc_profile_path,
            )
            for fleet in fleet_values for pickup in pickup_values
        ]

    jobs = grid_jobs(fleets, pickups, "joint")
    _run_parallel(jobs, workers=workers, output=output, phase="nyc_joint_grid")
    measurements = [
        _measurement(
            _validate_product(job.raw, job.expected), t1.CARD,
            stage="joint_grid",
            fleet_scale=float(_option(job.command, "--fleet_scale")),
            base_pickup_minutes=float(
                _option(job.command, "--base_pickup_minutes")
            ),
        )
        for job in jobs
    ]
    selected = _select_joint_calibration(
        measurements, target_wait=target_wait, allow_no_binding=True
    )
    extensions: list[dict[str, Any]] = []
    extension_jobs: list[Job] = []
    fleet = float(selected["fleet_scale"])
    pickup = float(selected["base_pickup_minutes"])
    if fleet == min(fleets):
        extra = 0.4
        extension_jobs.extend(grid_jobs([extra], pickups, "edge"))
        fleets.append(extra)
        extensions.append({"axis": "fleet", "value": extra})
    if pickup == min(pickups):
        extra = 0.75
        extension_jobs.extend(grid_jobs(fleets, [extra], "edge"))
        pickups.append(extra)
        extensions.append({"axis": "pickup", "value": extra})
    if extension_jobs:
        unique = {job.name: job for job in extension_jobs}
        extension_jobs = list(unique.values())
        _run_parallel(
            extension_jobs, workers=workers, output=output,
            phase="nyc_joint_edge_extension",
        )
        measurements.extend([
            _measurement(
                _validate_product(job.raw, job.expected), t1.CARD,
                stage="edge_extension",
                fleet_scale=float(_option(job.command, "--fleet_scale")),
                base_pickup_minutes=float(
                    _option(job.command, "--base_pickup_minutes")
                ),
            )
            for job in extension_jobs
        ])
        selected = _select_joint_calibration(
            measurements, target_wait=target_wait
        )
    else:
        selected = _select_joint_calibration(
            measurements, target_wait=target_wait
        )
    confirmation_job, confirmation = _confirm_calibration(
        config, output, city="nyc", base=base,
        fleet=float(selected["fleet_scale"]),
        pickup=float(selected["base_pickup_minutes"]),
        target_wait=target_wait,
        hour_profile_json=nyc_profile_path,
        block_wait_targets=block_wait_targets,
    )
    if not confirmation["pass"]:
        raise ValueError(f"NYC joint calibration confirmation failed: {confirmation}")
    nyc = {
        "schema_version": "v18-joint-supply-calibration-v2",
        "city": "nyc",
        "base_active_drivers": base,
        "base_anchor_product": str(anchor_job.raw),
        "hour_profile": dict(nyc_profile["hour_profile"]),
        "hour_profile_factors_by_four_hour_block": dict(
            nyc_profile["factors_by_four_hour_block"]
        ),
        "hour_profile_fit": nyc_profile,
        "hour_profile_products": [str(job.raw) for job in profile_jobs],
        "selection": selected,
        "fleet_scale": float(selected["fleet_scale"]),
        "base_pickup_minutes": float(selected["base_pickup_minutes"]),
        "reference_earnings_per_hour": float(
            confirmation["reference_earnings_per_hour"]
        ),
        "reference_earnings_contract": "ratio_of_sums_both_platforms_days_5_28",
        "targets": {
            "utilization": TARGET_UTILIZATION,
            "utilization_band": list(UTILIZATION_BAND),
            "median_wait_minutes": target_wait,
            "median_wait_tolerance": WAIT_TOLERANCE,
            "maximum_trips_per_online_driver_hour": MAX_TRIPS_PER_ONLINE_HOUR,
        },
        "source_urls": {
            "utilization_rule": "https://www.nyc.gov/site/tlc/about/driver-pay-rates.page",
            "trips_per_vehicle_hour": "https://www.nyc.gov/assets/tlc/downloads/csv/data_reports_monthly.csv",
            "wait": wait_target["source_url"],
            "weather": wait_target["weather_source_url"],
        },
        "measurements": measurements,
        "edge_extensions": extensions,
        "confirmation": confirmation,
        "confirmation_product": str(confirmation_job.raw),
        "wait_target": wait_target,
        "service_level_target": "new_york_real_wait_profile",
    }
    atomic_json(PAY_DIR / "nyc/supply.json", nyc)

    results = {"nyc": nyc}
    for city in ("chicago", "boston"):
        _archive_old_supply(city)
        city_anchor, city_base = _base_anchor(config, output, city=city)
        city_profile_path, city_profile, city_profile_jobs = _fit_hour_profile(
            config, output, city=city, base_active_drivers=city_base,
            fleet_scale=1.0, pickup=float(nyc["base_pickup_minutes"]),
            target_waits=block_wait_targets,
        )
        city_jobs = [
            _calibration_job(
                config, output, name=f"{city}_fleet_{fleet:.2f}", city=city,
                fleet_scale=fleet,
                pickup=float(nyc["base_pickup_minutes"]),
                base_active_drivers=city_base,
                hour_profile_json=city_profile_path,
            )
            for fleet in TRANSFER_FLEET_GRID
        ]
        _run_parallel(
            city_jobs, workers=workers, output=output,
            phase=f"{city}_fleet_grid",
        )
        city_measurements = []
        for job in city_jobs:
            payload = _validate_product(job.raw, job.expected)
            row = _measurement(
                payload, t1.CARD, stage="city_fleet_grid",
                fleet_scale=float(_option(job.command, "--fleet_scale")),
                base_pickup_minutes=float(nyc["base_pickup_minutes"]),
            )
            block_waits = _block_waits(payload, t1.CARD)
            row["wait_by_four_hour_block"] = {
                str(block): float(block_waits[block])
                for block in PROFILE_BLOCKS
            }
            row["service_level_loss"] = mean(
                ((block_waits[block] - block_wait_targets[block]) / 0.4) ** 2
                for block in PROFILE_BLOCKS
            )
            city_measurements.append(row)
        choice = _select_transfer_service_level(city_measurements)
        city_confirmation_job, city_confirmation = _confirm_calibration(
            config, output, city=city, base=city_base,
            fleet=float(choice["fleet_scale"]),
            pickup=float(nyc["base_pickup_minutes"]),
            target_wait=target_wait,
            hour_profile_json=city_profile_path,
            block_wait_targets=block_wait_targets,
            binding_targets=False,
        )
        city_confirmation["service_level_loss"] = mean(
            (float(row["gap_minutes"]) / 0.4) ** 2
            for row in city_confirmation["wait_by_four_hour_block"]
        )
        city_confirmation["selection_rule"] = (
            "minimum_block_wait_loss_over_registered_transfer_grid"
        )
        city_checks = {
            "service_level_profile_recorded": bool(
                len(city_confirmation["wait_by_four_hour_block"])
                == len(PROFILE_BLOCKS)
            ),
            "utilization_reported": bool(
                math.isfinite(float(city_confirmation["utilization"]))
            ),
        }
        if not all(city_checks.values()):
            raise ValueError(
                f"{city} service-level report is incomplete: {city_confirmation}"
            )
        city_supply = {
            "schema_version": "v18-city-supply-calibration-v2",
            "city": city,
            "base_active_drivers": city_base,
            "base_anchor_product": str(city_anchor.raw),
            "hour_profile": dict(city_profile["hour_profile"]),
            "hour_profile_factors_by_four_hour_block": dict(
                city_profile["factors_by_four_hour_block"]
            ),
            "hour_profile_fit": city_profile,
            "hour_profile_products": [
                str(job.raw) for job in city_profile_jobs
            ],
            "fleet_scale": float(choice["fleet_scale"]),
            "base_pickup_minutes": float(nyc["base_pickup_minutes"]),
            "pickup_source": "nyc_joint_calibration",
            "service_level_target": "service_level_transferred_from_new_york",
            "service_level_target_wait_by_four_hour_block": {
                str(block): float(block_wait_targets[block])
                for block in PROFILE_BLOCKS
            },
            "utilization_role": "reported_outcome_not_target",
            "reference_earnings_per_hour": float(
                city_confirmation["reference_earnings_per_hour"]
            ),
            "reference_earnings_contract": "ratio_of_sums_both_platforms_days_5_28",
            "measurements": city_measurements,
            "confirmation": city_confirmation,
            "confirmation_checks": city_checks,
            "confirmation_product": str(city_confirmation_job.raw),
        }
        atomic_json(PAY_DIR / city / "supply.json", city_supply)
        results[city] = city_supply
    return results


def run_reference_earnings(
    config: Mapping[str, Any], output: Path, *, workers: int,
) -> dict[str, Any]:
    del config, output, workers
    return {
        city: float(json.loads(
            (PAY_DIR / city / "supply.json").read_text()
        )["reference_earnings_per_hour"])
        for city in REFERENCE_CITIES
    }


def _kappa_job(
    config: Mapping[str, Any], output: Path, *, kappa: float,
) -> tuple[Job, str]:
    supply = json.loads((PAY_DIR / "nyc/supply.json").read_text())
    work = output / "kappa_test" / f"kappa_{kappa:.2f}"
    command = t1.market_command(
        config, city="nyc", job="calm_static", output=work,
        openings=2, customers=CUSTOMERS, pool=PRODUCTION_POOL,
    )
    policy = surge_decay_spec(
        (1.0, 1.0), (1.25, 1.25), [[14, 83]], 0, 0
    )
    raw = work / "raw.json"
    command = set_option(command, "--policy_sweep", policy)
    command = set_option(command, "--policy_sweep_horizon", 84 * STEPS_PER_DAY)
    command = set_option(command, "--policy_sweep_seed", 20231)
    command = set_option(
        command, "--policy_sweep_opening_tariff_randomization", 0.0
    )
    command, expected = _v18_command(
        command, city="nyc", raw=raw,
        report_prefix=work / "diagnostics", stock=True,
        linear_pay=True, fleet_scale=float(supply["fleet_scale"]),
        base_pickup_minutes=float(supply["base_pickup_minutes"]),
        production_population=True,
        reference_earnings=float(supply["reference_earnings_per_hour"]),
        base_active_drivers=int(supply["base_active_drivers"]),
        stock_kappa=float(kappa),
        hour_profile_json=PAY_DIR / "nyc/supply.json",
    )
    return Job(
        name=f"kappa_{kappa:.2f}", command=tuple(command), raw=raw,
        manifest=work / "manifest.json", log=work / "job.log",
        expected=expected,
    ), policy


def _daily_earnings_paths(
    rows: Sequence[Mapping[str, Any]], *, reference: float,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(
            (str(row["rollout"]), int(row["step"]) // STEPS_PER_DAY), []
        ).append(row)
    path = []
    for (rollout, day), daily in sorted(grouped.items()):
        pay = sum(float(row["driver_pay_firm1"]) for row in daily)
        online = sum(float(row["online_minutes_firm1"]) for row in daily)
        earnings = DriverSupplyLayer.earnings_per_online_hour(pay, online)
        path.append({
            "rollout": rollout, "day": day, "earnings": earnings,
            "gap": math.log(max(earnings, 1e-6) / reference),
            "stock": mean(float(row["stock_firm1"]) for row in daily),
        })
    return path


def _select_kappa_candidate(
    rows: Sequence[Mapping[str, Any]], *, registered_kappa: float,
) -> tuple[dict[str, Any], bool]:
    """Apply the registered criterion or retain the predeclared value.

    Revision 1 Section 3 does not make a missed kappa diagnostic a stop.
    When none of the three registered candidates meets the original
    re-equilibration criterion, retaining the already registered production
    value avoids introducing an unregistered scientific fallback.
    """
    passing = [dict(row) for row in rows if bool(row["pass"])]
    if passing:
        return min(passing, key=lambda row: float(row["kappa"])), False
    matches = [
        dict(row) for row in rows
        if abs(float(row["kappa"]) - float(registered_kappa)) <= 1e-12
    ]
    if len(matches) != 1:
        raise ValueError(
            "registered production kappa is absent from the candidate set"
        )
    return matches[0], True


def run_kappa_test(
    config: Mapping[str, Any], output: Path, *, workers: int,
) -> dict[str, Any]:
    supply = json.loads((PAY_DIR / "nyc/supply.json").read_text())
    jobs_and_specs = [
        _kappa_job(config, output, kappa=kappa)
        for kappa in KAPPA_CANDIDATES
    ]
    jobs = [job for job, _ in jobs_and_specs]
    _run_parallel(jobs, workers=workers, output=output, phase="kappa_test")
    rows = []
    for kappa, (job, spec) in zip(KAPPA_CANDIDATES, jobs_and_specs):
        payload = _validate_product(job.raw, job.expected)
        path = _daily_earnings_paths(
            payload["traces"][spec],
            reference=float(supply["reference_earnings_per_hour"]),
        )
        initial = [
            abs(float(row["gap"])) for row in path
            if 14 <= int(row["day"]) <= 17
        ]
        late = [
            abs(float(row["gap"])) for row in path
            if 63 <= int(row["day"]) <= 70
        ]
        if not initial or not late:
            raise ValueError("kappa test is missing registered gap windows")
        initial_gap = mean(initial)
        late_gap = mean(late)
        rows.append({
            "kappa": kappa,
            "initial_absolute_gap": initial_gap,
            "late_absolute_gap": late_gap,
            "late_to_initial_ratio": late_gap / max(initial_gap, 1e-12),
            "initial_gap_sign": math.copysign(
                1.0,
                mean(
                    float(row["gap"]) for row in path
                    if 14 <= int(row["day"]) <= 17
                ),
            ),
            "pass": bool(late_gap <= 0.20 * initial_gap),
            "path": path,
            "product": str(job.raw),
        })
    registered_kappa = float(load_v18()["market"]["stock_adjustment_rate"])
    selected, fallback_used = _select_kappa_candidate(
        rows, registered_kappa=registered_kappa
    )
    report = {
        "candidates": rows, "selected_kappa": float(selected["kappa"]),
        "criterion_pass": not fallback_used,
        "registered_value_retained": fallback_used,
        "selection_reason": (
            "smallest_candidate_meeting_registered_criterion"
            if not fallback_used
            else "revision_1_nonstop_diagnostic_retains_registered_value"
        ),
        "criterion": "days_63_70_abs_gap_le_20pct_days_14_17",
    }
    atomic_json(output / "kappa_test/report.json", report)
    for city in REFERENCE_CITIES:
        path = PAY_DIR / city / "supply.json"
        city_supply = json.loads(path.read_text())
        city_supply["stock_adjustment_rate"] = float(selected["kappa"])
        city_supply["stock_adjustment_source"] = str(
            output / "kappa_test/report.json"
        )
        atomic_json(path, city_supply)
    return report


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
        base_active_drivers=int(supply["base_active_drivers"]),
        stock_kappa=float(supply["stock_adjustment_rate"]),
        hour_profile_json=PAY_DIR / city / "supply.json",
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


def _effect_specs() -> tuple[tuple[str, str], ...]:
    selected = json.loads(
        (BASE / "runs/v17_results_comparators/nyc/selected.json").read_text()
    )
    best = str(selected["best_fixed_cell"]["spec"])
    return (
        ("published", PUBLISHED),
        ("v17_best_fixed", best),
        ("1.25", t2.cell_spec(1.25, 1.25)),
        ("1.5", t2.cell_spec(1.5, 1.5)),
        ("2.0", t2.cell_spec(2.0, 2.0)),
        ("2.5", t2.cell_spec(2.5, 2.5)),
        ("dated_1.5", surge_decay_spec(
            (0.8, 2.0), (1.5, 1.5), [[10, 15]], 0, 0
        )),
        ("dated_2.0", surge_decay_spec(
            (0.8, 2.0), (2.0, 2.0), [[10, 15]], 0, 0
        )),
    )


def _effect_job(
    config: Mapping[str, Any], output: Path, *, arm: str, case: str, rival: str,
) -> Job:
    work = output / "effects" / arm / case / rival
    raw = work / "raw.json"
    command = t2.sweep_command(
        config, city="nyc", case=case, rival=rival, output=work,
        openings=2, customers=CUSTOMERS, pool=PRODUCTION_POOL,
        days=HORIZON_DAYS,
    )
    command = set_option(
        command, "--policy_sweep",
        ";".join(spec for _, spec in _effect_specs()),
    )
    supply = json.loads((PAY_DIR / "nyc/supply.json").read_text())
    settings = {
        "v17_model": (False, False, 1.2, 3.5),
        "stock": (True, False, 1.2, 3.5),
        "pay_rule": (True, True, 1.2, 3.5),
        "calibrated": (
            True, True, float(supply["fleet_scale"]),
            float(supply["base_pickup_minutes"]),
        ),
    }[arm]
    command, expected = _v18_command(
        command, city="nyc", raw=raw,
        report_prefix=work / "diagnostics", stock=settings[0],
        linear_pay=settings[1], fleet_scale=settings[2],
        base_pickup_minutes=settings[3], production_population=True,
        reference_earnings=(
            float(supply["reference_earnings_per_hour"])
            if settings[0] else None
        ),
        base_active_drivers=(
            int(supply["base_active_drivers"])
            if arm == "calibrated" else 0
        ),
        stock_kappa=float(supply["stock_adjustment_rate"]),
        hour_profile_json=(
            PAY_DIR / "nyc/supply.json"
            if arm == "calibrated" else None
        ),
    )
    return Job(
        name=f"effect|{arm}|{case}|{rival}", command=tuple(command), raw=raw,
        manifest=work / "manifest.json", log=work / "job.log",
        expected=expected,
    )


def _calm_published_stock_job(
    config: Mapping[str, Any], output: Path,
) -> Job:
    """Build the registered steady-state cell, separate from dynamic rivals.

    The 12 November C1 test and the P2 fix specification define this check as
    pure calm with both platforms on the published card.  Dynamic-rival C0
    products remain part of the redesigned effect table, but they are not a
    steady-state stock test because rival repricing changes focal earnings.
    """
    supply = json.loads((PAY_DIR / "nyc/supply.json").read_text())
    work = output / "effects" / "calm_published_stock_gate"
    raw = work / "raw.json"
    command = t1.market_command(
        config, city="nyc", job="calm_static", output=work,
        openings=2, customers=CUSTOMERS, pool=PRODUCTION_POOL,
        seed=20231,
    )
    command = set_option(command, "--policy_sweep", PUBLISHED)
    command = set_option(
        command, "--policy_sweep_horizon", HORIZON_DAYS * STEPS_PER_DAY
    )
    command = set_option(command, "--policy_sweep_seed", 20231)
    command = set_option(
        command, "--policy_sweep_opening_tariff_randomization", 0.0
    )
    command, expected = _v18_command(
        command, city="nyc", raw=raw,
        report_prefix=work / "diagnostics", stock=True,
        linear_pay=True, fleet_scale=float(supply["fleet_scale"]),
        base_pickup_minutes=float(supply["base_pickup_minutes"]),
        production_population=True,
        reference_earnings=float(supply["reference_earnings_per_hour"]),
        base_active_drivers=int(supply["base_active_drivers"]),
        stock_kappa=float(supply["stock_adjustment_rate"]),
        hour_profile_json=PAY_DIR / "nyc/supply.json",
    )
    return Job(
        name="calm_published_stock_gate", command=tuple(command), raw=raw,
        manifest=work / "manifest.json", log=work / "job.log",
        expected=expected,
    )


def _calm_published_stock_gate(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate focal stock only in the exact both-published calm cell."""
    traces = dict(payload.get("traces") or {})
    if set(traces) != {PUBLISHED}:
        raise ValueError("calm stock gate must contain only the published card")
    all_rows = list(traces[PUBLISHED])
    if not all_rows:
        raise ValueError("calm stock gate has no trace rows")
    for row in all_rows:
        if (
            abs(float(row["firm1_level"]) - 1.0) > 1e-12
            or abs(float(row["firm2_level"]) - 1.0) > 1e-12
        ):
            raise ValueError(
                "calm stock gate requires both platforms on the exact "
                "published card"
            )
    rows = [
        row for row in all_rows
        if 10 <= int(row["step"]) // STEPS_PER_DAY < 40
    ]
    if not rows:
        raise ValueError("calm stock gate has no rows for days 10--40")
    values = [float(row["stock_firm1"]) for row in rows]
    rival_values = [float(row["stock_firm2"]) for row in rows]
    result = {
        "population": "pure_C0_both_platforms_published_exact",
        "gated_stock": "stock_firm1",
        "days": [10, 40],
        "opening_tariff_randomization": 0.0,
        "rollouts": len({str(row["rollout"]) for row in rows}),
        "trace_rows": len(rows),
        "minimum": min(values),
        "maximum": max(values),
        "maximum_absolute_drift": max(abs(value - 1.0) for value in values),
        "rival_stock_diagnostic": {
            "minimum": min(rival_values),
            "maximum": max(rival_values),
            "maximum_absolute_drift": max(
                abs(value - 1.0) for value in rival_values
            ),
        },
    }
    result["pass"] = bool(result["maximum_absolute_drift"] <= 0.03)
    if not _finite(result):
        raise ValueError(f"non-finite calm stock gate: {result}")
    return result


def _validate_preserved_effect_job(job: Job) -> None:
    """Reuse df39287 effect products across this aggregation-only repair.

    Every command, config hash, unaffected source hash, and product hash must
    still match.  The permitted source differences are pinned to the exact
    repair parents: this runner separates the gate aggregation, while Core's
    zero-randomization repair leaves the effect jobs' default 0.15 opening
    perturbation unchanged.
    """
    recorded = json.loads(job.manifest.read_text(encoding="utf-8"))
    if recorded.get("state") != "completed":
        raise ValueError(f"preserved effect product is incomplete: {job.raw}")
    identity = dict(recorded.get("identity") or {})
    if identity.get("command") != list(job.command):
        raise ValueError(f"preserved effect command changed: {job.raw}")
    if identity.get("config_hash") != canonical_hash(load_v18()):
        raise ValueError(f"preserved effect config changed: {job.raw}")
    recorded_sources = dict(identity.get("sources") or {})
    if recorded_sources.get("scripts/v18_p2.py") != (
        P2_GATE_REPAIR_PARENT_SOURCE_SHA256
    ):
        raise ValueError(f"preserved effect runner is not the repair parent: {job.raw}")
    if recorded_sources.get("Core.py") != P2_GATE_REPAIR_PARENT_CORE_SHA256:
        raise ValueError(f"preserved effect Core is not the repair parent: {job.raw}")
    if "--policy_sweep_opening_tariff_randomization" in job.command:
        opening_randomization = float(_option(
            job.command, "--policy_sweep_opening_tariff_randomization"
        ))
    else:
        opening_randomization = 0.15
    if abs(opening_randomization - 0.15) > 1e-12:
        raise ValueError(
            "preserved effect reuse requires the unchanged 0.15 opening "
            f"randomization: {job.raw}"
        )
    current_sources = _source_identity()
    for name, digest in current_sources.items():
        if name in {"scripts/v18_p2.py", "Core.py"}:
            continue
        if recorded_sources.get(name) != digest:
            raise ValueError(
                f"preserved effect source changed ({name}): {job.raw}"
            )
    expected_hash = recorded.get("product_sha256")
    if not expected_hash or _sha256_file(job.raw) != expected_hash:
        raise ValueError(f"preserved effect product hash changed: {job.raw}")
    t2.validate_t2_product(
        job.raw, job.command,
        model_expected={
            "acceptance_shock_wage": "declared",
            "rate_upper_multiple": 2.5,
        },
    )
    _validate_product(job.raw, job.expected)


def _window_rows(
    rows: Sequence[Mapping[str, Any]], *, case: str, window: str,
) -> list[Mapping[str, Any]]:
    if window == "days_10_15":
        return [
            row for row in rows
            if 10 <= int(row["step"]) // STEPS_PER_DAY <= 15
        ]
    if window == "calm":
        if case == "C0_calm":
            return list(rows)
        return [
            row for row in rows
            if not 10 <= int(row["step"]) // STEPS_PER_DAY <= 15
        ]
    return list(rows)


def _trace_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ValueError("effect summary window has no rows")
    pay = sum(float(row["driver_pay_firm1"]) for row in rows)
    online = sum(float(row["online_minutes_firm1"]) for row in rows)
    passenger = sum(float(row["passenger_minutes_firm1"]) for row in rows)
    return {
        "own_profit": mean(float(row["profit_per_request"]) for row in rows),
        "rival_profit": mean(
            float(row["rival_profit_per_request"]) for row in rows
        ),
        "served_share": mean(float(row["own_share"]) for row in rows),
        "drivers_online": mean(
            float(row["active_drivers_firm1"]) for row in rows
        ),
        "stock": mean(float(row["stock_firm1"]) for row in rows),
        "earnings_per_online_hour": (
            DriverSupplyLayer.earnings_per_online_hour(pay, online)
        ),
        "utilization": passenger / max(online, 1e-12),
    }


def _paired_t(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    estimate = float(np.mean(array))
    se = (
        float(np.std(array, ddof=1) / np.sqrt(len(array)))
        if len(array) > 1 else 0.0
    )
    return {
        "mean": estimate,
        "se": se,
        "t": (
            estimate / se if se > 0
            else (1.0e12 if estimate > 0 else -1.0e12 if estimate < 0 else 0.0)
        ),
        "n": int(len(array)),
    }


def _effect_summary(jobs: Sequence[Job]) -> dict[str, Any]:
    specs = _effect_specs()
    spec_map = dict(specs)
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    payloads: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for job in jobs:
        _, arm, case, rival = job.name.split("|", 3)
        payload = _validate_product(job.raw, job.expected)
        payloads[(arm, case, rival)] = payload
        for label, spec in specs:
            grouped.setdefault((arm, case, label), []).append({
                "payload": payload, "spec": spec,
            })
    rows = []
    for (arm, case, label), products in sorted(grouped.items()):
        results = []
        traces = []
        for item in products:
            results.extend(
                dict(item["payload"]["results"][item["spec"]]).values()
            )
            traces.extend(item["payload"]["traces"][item["spec"]])
        row = {
            "arm": arm, "case": case, "card": label,
            "objective": mean(float(item["objective"]) for item in results),
            "own_profit": mean(float(item["own_profit"]) for item in results),
            "rival_profit": mean(float(item["rival_profit"]) for item in results),
            "days_10_15": _trace_metrics(
                _window_rows(traces, case=case, window="days_10_15")
            ),
            "calm_days": _trace_metrics(
                _window_rows(traces, case=case, window="calm")
            ),
            "rollouts": len(results), "trace_rows": len(traces),
        }
        if not _finite(row):
            raise ValueError(f"non-finite P2 effect row: {row}")
        rows.append(row)

    g_approx = []
    for arm in EFFECT_ARMS:
        by_rule: dict[str, list[float]] = {}
        for label in ("dated_1.5", "dated_2.0"):
            differences = []
            for rival in EFFECT_RIVALS:
                payload = payloads[(arm, "A1_shortage_severe", rival)]
                best_results = payload["results"][spec_map["v17_best_fixed"]]
                dated_results = payload["results"][spec_map[label]]
                for rollout in sorted(best_results):
                    differences.append(
                        float(dated_results[rollout]["objective"])
                        - float(best_results[rollout]["objective"])
                    )
            by_rule[label] = differences
        winner = max(by_rule, key=lambda label: mean(by_rule[label]))
        paired = _paired_t(by_rule[winner])
        critical = 2.5705818366147395 if paired["n"] == 6 else 1.96
        g_approx.append({
            "arm": arm, "best_dated_rule": winner, **paired,
            "interval_95": [
                paired["mean"] - critical * paired["se"],
                paired["mean"] + critical * paired["se"],
            ],
        })

    driver_differences = []
    for rival in EFFECT_RIVALS:
        payload = payloads[("calibrated", "A1_shortage_severe", rival)]
        published = payload["traces"][spec_map["published"]]
        surge = payload["traces"][spec_map["1.5"]]
        rollouts = sorted({str(row["rollout"]) for row in published})
        for rollout in rollouts:
            def average(source: Sequence[Mapping[str, Any]]) -> float:
                values = [
                    float(row["active_drivers_firm1"]) for row in source
                    if str(row["rollout"]) == rollout
                    and 11 <= int(row["step"]) // STEPS_PER_DAY <= 15
                ]
                return mean(values)
            driver_differences.append(average(surge) - average(published))
    driver_gate = _paired_t(driver_differences)
    driver_gate["pass"] = bool(
        driver_gate["mean"] > 0.0 and driver_gate["t"] > 2.0
    )
    return {
        "arms": list(EFFECT_ARMS), "cases": list(EFFECT_CASES),
        "rivals": list(EFFECT_RIVALS), "openings": 2,
        "customers": CUSTOMERS, "rows": rows,
        "surge_brings_drivers_gate": driver_gate,
        "Gapprox": g_approx,
    }


def run_effects(
    config: Mapping[str, Any], output: Path, *, workers: int,
    reuse_preserved_after_gate_repair: bool = False,
) -> dict[str, Any]:
    jobs = [
        _effect_job(config, output, arm=arm, case=case, rival=rival)
        for arm in EFFECT_ARMS for case in EFFECT_CASES for rival in EFFECT_RIVALS
    ]
    if reuse_preserved_after_gate_repair:
        for job in jobs:
            _validate_preserved_effect_job(job)
        atomic_json(output / "status.json", {
            "step": "P2", "state": "running",
            "phase": "effect_table_products_revalidated",
            "completed_jobs": len(jobs),
            "expected_jobs": len(jobs) + 1,
            "reused_products": len(jobs),
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=BASE, text=True
            ).strip(),
        })
    else:
        _run_parallel(
            jobs, workers=workers, output=output, phase="effect_table",
            expected_total=len(jobs) + 1,
        )
    stock_job = _calm_published_stock_job(config, output)
    _run_parallel(
        [stock_job], workers=1, output=output,
        phase="calm_published_stock_gate", completed_before=len(jobs),
        expected_total=len(jobs) + 1,
    )
    report = _effect_summary(jobs)
    report["calm_published_stock_gate"] = _calm_published_stock_gate(
        _validate_product(stock_job.raw, stock_job.expected)
    )
    report["effect_product_provenance"] = {
        "reused_after_aggregation_only_gate_repair": bool(
            reuse_preserved_after_gate_repair
        ),
        "revalidated_products": len(jobs),
        "gate_product": str(stock_job.raw),
    }
    atomic_json(output / "effects/report.json", report)
    if not report["calm_published_stock_gate"]["pass"]:
        raise ValueError("P2 calm published-card stock drift exceeded 3%")
    if not report["surge_brings_drivers_gate"]["pass"]:
        raise ValueError("P2 calibrated 1.5 surge did not bring drivers online")
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
    kappa = json.loads((output / "kappa_test/report.json").read_text())
    revealed = json.loads((output / "revealed_beta/report.json").read_text())
    decision = {
        "fitted_supply": {
            city: {
                "fleet_scale": float(supplies[city]["fleet_scale"]),
                "base_pickup_minutes": float(supplies[city]["base_pickup_minutes"]),
                "reference_earnings_per_hour": float(
                    supplies[city]["reference_earnings_per_hour"]
                ),
                "base_active_drivers": int(
                    supplies[city]["base_active_drivers"]
                ),
                "hour_profile": dict(supplies[city]["hour_profile"]),
                "service_level_target": str(
                    supplies[city]["service_level_target"]
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
        "kappa_test": kappa,
        "revealed_beta": revealed,
        "market_checks": market,
        "effect_table": effects,
        "stop_conditions": {
            "finite": True,
            "calm_published_stock_within_3_percent": bool(
                effects["calm_published_stock_gate"]["pass"]
            ),
            "calibration_binding_targets": bool(
                supplies["nyc"]["confirmation"]["pass"]
                and all(
                    all(supplies[city]["confirmation_checks"].values())
                    for city in ("boston", "chicago")
                )
            ),
            "surge_brings_drivers": bool(
                effects["surge_brings_drivers_gate"]["pass"]
            ),
            "identity_and_manifests": True,
        },
    }
    if not _finite(report) or not all(report["stop_conditions"].values()):
        raise ValueError("P2 final report failed a stop condition")
    atomic_json(output / "report.json", report)
    config_path = BASE / "configs/v18_production.yaml"
    production = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    production["market"]["stock_adjustment_rate"] = float(
        kappa["selected_kappa"]
    )
    production["market"]["demand_scale"] = "lyft_prior"
    production["calibration"].update({
        "base_active_drivers_frozen": {
            city: int(supplies[city]["base_active_drivers"])
            for city in REFERENCE_CITIES
        },
        "fleet_scale": {
            city: float(supplies[city]["fleet_scale"])
            for city in REFERENCE_CITIES
        },
        "base_pickup_minutes": {
            city: float(supplies[city]["base_pickup_minutes"])
            for city in REFERENCE_CITIES
        },
        "reference_earnings_per_hour": {
            city: float(supplies[city]["reference_earnings_per_hour"])
            for city in REFERENCE_CITIES
        },
    })
    temporary = config_path.with_name(
        f".{config_path.name}.tmp.{os.getpid()}"
    )
    temporary.write_text(
        yaml.safe_dump(production, sort_keys=False), encoding="utf-8"
    )
    temporary.replace(config_path)
    g_lines = "\n".join(
        f"- {row['arm']}: {row['mean']:.6f} "
        f"[{row['interval_95'][0]:.6f}, {row['interval_95'][1]:.6f}]"
        for row in effects["Gapprox"]
    )
    markdown = (
        "# V18 P2 calibration and effect table\n\n"
        "## Registered stop rules\n\n"
        f"- Calm stock: PASS (maximum drift "
        f"{effects['calm_published_stock_gate']['maximum_absolute_drift']:.4f}).\n"
        "- Joint calibration binding targets: PASS.\n"
        f"- NYC utilization aim: {supplies['nyc']['confirmation']['utilization']:.4f} "
        f"(gap below 0.52: {supplies['nyc']['confirmation']['utilization_aim']['utilization_gap']:.4f}; "
        "not a stop rule).\n"
        f"- Surge brings drivers: PASS (paired t "
        f"{effects['surge_brings_drivers_gate']['t']:.3f}).\n"
        "- Finite numbers and identity: PASS.\n\n"
        "## Gapprox by arm\n\n" + g_lines + "\n\n"
        "T1.1--T1.3 are reported diagnostics and are not P2 stop gates.\n"
    )
    (output / "report.md").write_text(markdown, encoding="utf-8")
    atomic_json(output / "status.json", {
        "step": "P2", "state": "completed", "pass": True,
        "phase": "complete", "report": str(output / "report.json"),
        "git_head": report["git_head"],
    })
    return report


def _archive_failed_p2_roots(output: Path) -> None:
    if output.name != "P2_fix":
        return
    preflight = output.parent
    archive = preflight / "archive_P2_20261112"
    archive.mkdir(parents=True, exist_ok=True)
    for name in ("P2", "P2_attempt2"):
        source = preflight / name
        destination = archive / name
        if source.exists() and not destination.exists():
            shutil.move(str(source), str(destination))
    revision_archive = preflight / "archive_P2_fix_20261113"
    revision_archive.mkdir(parents=True, exist_ok=True)
    for name in ("P2_fix", "P2_fix_attempt2"):
        source = preflight / name
        destination = revision_archive / name
        if source.exists() and not destination.exists():
            shutil.move(str(source), str(destination))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "all", "pay", "wait", "supply", "reference", "kappa",
            "revealed", "market", "effects", "finalize",
        ),
        default="all",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--output", type=Path,
        default=BASE / "runs/v18/preflight/P2_fix",
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--reuse-preserved-effects-after-gate-repair",
        action="store_true",
        help=(
            "revalidate and reuse the 24 df39287 effect products while "
            "running only the corrected dedicated calm-stock gate cell"
        ),
    )
    args = parser.parse_args(argv)
    if not 1 <= int(args.workers) <= 4:
        raise ValueError("Revision 1 P2 workers must be in [1, 4]")
    output = args.output.resolve()
    if args.plan_only:
        print(json.dumps({
            "step": "P2", "workers": args.workers,
            "joint_calibration_jobs": len(FLEET_GRID) * len(PICKUP_GRID),
            "city_fleet_jobs": 2 * len(TRANSFER_FLEET_GRID),
            "capacity_anchors": len(REFERENCE_CITIES),
            "hour_profile_jobs_maximum": (
                len(REFERENCE_CITIES) * PROFILE_MAX_ITERATIONS
            ),
            "confirmation_jobs": len(REFERENCE_CITIES),
            "kappa_jobs": len(KAPPA_CANDIDATES),
            "revealed_beta_jobs": 5,
            "market_jobs": len(T1_CITIES) * len(T1_JOBS),
            "effect_jobs": len(EFFECT_ARMS) * len(EFFECT_CASES) * len(EFFECT_RIVALS),
            "calm_published_stock_gate_jobs": 1,
            "effect_rollouts": len(EFFECT_ARMS) * len(EFFECT_CASES)
            * len(EFFECT_RIVALS) * len(_effect_specs()) * 2,
        }, indent=2, sort_keys=True))
        return 0
    try:
        if args.stage == "all":
            _archive_failed_p2_roots(output)
        output.mkdir(parents=True, exist_ok=True)
        config = load_v17()
        if args.stage in {"all", "pay"}:
            fit_pay_rules()
        if args.stage in {"all", "wait"}:
            derive_wait_target(output)
        if args.stage in {"all", "supply"}:
            run_supply_calibration(config, output, workers=args.workers)
        if args.stage in {"all", "reference"}:
            run_reference_earnings(config, output, workers=args.workers)
        if args.stage in {"all", "kappa"}:
            run_kappa_test(config, output, workers=args.workers)
        if args.stage in {"all", "revealed"}:
            from scripts.v18_revealed_beta import run as run_revealed_beta
            run_revealed_beta(
                output / "revealed_beta", workers=args.workers
            )
        if args.stage in {"all", "market"}:
            run_market_checks(config, output, workers=args.workers)
        if args.stage in {"all", "effects"}:
            run_effects(
                config, output, workers=args.workers,
                reuse_preserved_after_gate_repair=bool(
                    args.reuse_preserved_effects_after_gate_repair
                ),
            )
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
