#!/usr/bin/env python3
"""V18 V5-V7 outcome fidelity, same-regression, and supply checks."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import date, timedelta
import gzip
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
DATA = BASE.parent / "data"
RUN = BASE / "runs/v18/realdata"
PROGRESS = BASE / "runs/v18/controller/progress.jsonl"
START_DATE = date(2023, 1, 2)
BOOTSTRAP_DRAWS = 1000
BOOTSTRAP_SEED = 20261112
sys.path.insert(0, str(BASE))

from scripts.v18_runtime import append_progress  # noqa: E402
from scripts.v18_validation.analyze import atomic_json, ghcn_daily, nyc_sample, utc_now, write_csv  # noqa: E402
from scripts.v18_validation.metrics import fidelity, ols_hc1  # noqa: E402


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def _v5_file(path_text: str) -> dict[str, Any]:
    path = Path(path_text)
    measures: dict[str, list[float]] = {key: [] for key in (
        "wait_minutes", "driver_pay_per_trip", "driver_pay_per_mile",
        "driver_pay_per_minute", "pay_fare_ratio",
    )}
    groups = []
    with gzip.open(path, "rt", newline="") as stream:
        for row in csv.DictReader(stream):
            if not (_bool(row.get("Fulfilled")) and row.get("Choice") in {"Firm1", "Firm2"}):
                continue
            distance = _float(row.get("TravelDistance"))
            duration = _float(row.get("DurationMinutes"))
            pay = _float(row.get("DriverPay"))
            wait = _float(row.get("WaitMinutes"))
            fare = _float(row.get("Price_Firm1" if row.get("Choice") == "Firm1" else "Price_Firm2"))
            values = {
                "wait_minutes": wait,
                "driver_pay_per_trip": pay,
                "driver_pay_per_mile": pay / distance if distance > 0 else float("nan"),
                "driver_pay_per_minute": pay / duration if duration > 0 else float("nan"),
                "pay_fare_ratio": pay / fare if fare > 0 else float("nan"),
            }
            if not all(np.isfinite(value) for value in values.values()):
                continue
            for key, value in values.items():
                measures[key].append(value)
            groups.append(f"{row['Day']}:{row['Block']}:{row['Weather']}")
    return {"measures": measures, "groups": groups}


def _group_summaries(values: np.ndarray, groups: Sequence[str]) -> np.ndarray:
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, group in zip(values, groups):
        if np.isfinite(value):
            grouped[str(group)].append(float(value))
    rows = []
    for group in sorted(grouped):
        sample = np.asarray(grouped[group], dtype=float)
        q = np.quantile(sample, [0.25, 0.5, 0.75])
        rows.append([sample.size, q[0], q[1], q[2]])
    return np.asarray(rows, dtype=float)


def _ratio_bootstrap(
    real: np.ndarray, real_groups: Sequence[str], synthetic: np.ndarray, synthetic_groups: Sequence[str],
) -> dict[str, Any]:
    """Day-block bootstrap of median and IQR ratios via weighted group quantiles."""
    real_summary = _group_summaries(real, real_groups)
    synthetic_summary = _group_summaries(synthetic, synthetic_groups)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    median_ratio, iqr_ratio = [], []
    for _ in range(BOOTSTRAP_DRAWS):
        r = real_summary[rng.integers(0, len(real_summary), len(real_summary))]
        s = synthetic_summary[rng.integers(0, len(synthetic_summary), len(synthetic_summary))]
        # Weight group-level quantiles by each resampled group's row count.
        rq = np.average(r[:, 1:], axis=0, weights=r[:, 0])
        sq = np.average(s[:, 1:], axis=0, weights=s[:, 0])
        median_ratio.append(sq[1] / max(abs(rq[1]), 1e-12))
        iqr_ratio.append((sq[2] - sq[0]) / max(rq[2] - rq[0], 1e-12))
    return {
        "draws": BOOTSTRAP_DRAWS,
        "unit": "day-block-weather group",
        "method": "weighted group-quantile block bootstrap",
        "median_ratio_ci95": [float(np.quantile(median_ratio, 0.025)), float(np.quantile(median_ratio, 0.975))],
        "iqr_ratio_ci95": [float(np.quantile(iqr_ratio, 0.025)), float(np.quantile(iqr_ratio, 0.975))],
    }


def stage_v5(*, workers: int, attempt: int = 1) -> dict[str, Any]:
    started = time.monotonic()
    append_progress(PROGRESS, job="realdata_V5", state="running", attempt=attempt, seconds=0)
    real_frame = nyc_sample()
    real_values = {
        "wait_minutes": real_frame["wait_minutes"].to_numpy(float),
        "driver_pay_per_trip": real_frame["driver_pay"].to_numpy(float),
        "driver_pay_per_mile": (real_frame["driver_pay"] / real_frame["distance_miles"]).to_numpy(float),
        "driver_pay_per_minute": (real_frame["driver_pay"] / real_frame["duration_minutes"]).to_numpy(float),
        "pay_fare_ratio": (real_frame["driver_pay"] / real_frame["actual_price"]).to_numpy(float),
    }
    real_groups = [f"{day}:{block}:{weather}" for day, block, weather in zip(
        real_frame["day_group"], real_frame["block"], real_frame["weather"],
    )]
    policies = ("published_card", "best_fixed", "nyc_s101")
    rows = []
    for policy in policies:
        files = sorted((BASE / f"runs/v18/eval/E10/nyc/C0/{policy}/requests").glob("*.csv.gz"))
        if len(files) != 10:
            raise ValueError(f"V5 expected 10 logs for {policy}, found {len(files)}")
        outputs = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            outputs = list(executor.map(_v5_file, map(str, files)))
        synthetic_groups = sum((output["groups"] for output in outputs), [])
        for measure in real_values:
            synthetic = np.asarray(sum((output["measures"][measure] for output in outputs), []), dtype=float)
            real = np.asarray(real_values[measure], dtype=float)
            real_keep = np.isfinite(real)
            rows.append({
                "policy": policy, "measure": measure,
                **fidelity(real[real_keep], synthetic),
                "bootstrap": _ratio_bootstrap(real[real_keep], np.asarray(real_groups)[real_keep], synthetic, synthetic_groups),
            })
    result = {
        "state": "complete", "computed_utc": utc_now(), "rows": rows,
        "platform_share_comparison": "not performed; firms are symmetric by design",
    }
    atomic_json(RUN / "v5_outcome/nyc_metrics.json", result)
    atomic_json(RUN / "v5_outcome/chicago_status.json", {
        "state": "not_applicable", "reason": "Chicago source has no wait or driver-pay fields and D1 failed",
    })
    seconds = time.monotonic() - started
    append_progress(PROGRESS, job="realdata_V5", state="complete", attempt=attempt, seconds=seconds)
    return {"rows": len(rows), "seconds": seconds}


def _e5_file(task: tuple[str, str]) -> dict[str, Any]:
    policy, path_text = task
    path = Path(path_text)
    blocks: dict[tuple[int, int], dict[str, Any]] = defaultdict(lambda: {
        "trips": 0, "log_pi": [], "wait": [], "weather": "clear", "dow": 0,
        "regime": "calm", "firm1_trips": 0, "firm1_duration": 0.0,
    })
    steps: dict[int, dict[str, Any]] = defaultdict(lambda: {
        "day": 0, "block": 0, "regime": "calm", "firm1_trips": 0, "firm1_duration": 0.0,
    })
    rollout = None
    with gzip.open(path, "rt", newline="") as stream:
        for row in csv.DictReader(stream):
            rollout = rollout or row["Rollout"]
            day, block, step = int(row["Day"]), int(row["Block"]), int(row["Step"])
            target = blocks[(day, block)]
            target["weather"] = row["Weather"]
            target["dow"] = int(row["DayOfWeek"])
            target["regime"] = row["Regime"]
            step_target = steps[step]
            step_target.update(day=day, block=block, regime=row["Regime"])
            if not (_bool(row.get("Fulfilled")) and row.get("Choice") in {"Firm1", "Firm2"}):
                continue
            target["trips"] += 1
            firm = row["Choice"]
            fare = _float(row["Price_Firm1" if firm == "Firm1" else "Price_Firm2"])
            card = _float(row["CardClearQuote"])
            wait = _float(row["WaitMinutes"])
            duration = _float(row["DurationMinutes"])
            if fare > 0 and card > 0:
                target["log_pi"].append(float(np.log(fare / card)))
            if np.isfinite(wait):
                target["wait"].append(wait)
            if firm == "Firm1":
                target["firm1_trips"] += 1
                step_target["firm1_trips"] += 1
                if np.isfinite(duration):
                    target["firm1_duration"] += duration
                    step_target["firm1_duration"] += duration
    rows = []
    for (day, block), item in blocks.items():
        if not item["trips"] or not item["log_pi"] or not item["wait"]:
            continue
        rows.append({
            "policy": policy, "rollout": rollout, "day": day, "date": (START_DATE + timedelta(days=day)).isoformat(),
            "block": block, "weekday": item["dow"], "rain": int(item["weather"] == "rain"),
            "snow": int(item["weather"] == "snow"), "holiday_week": int(item["regime"] == "holiday_wave"),
            "log_trips": float(np.log(item["trips"])), "price_index": float(np.exp(np.mean(item["log_pi"]))),
            "median_wait": float(np.median(item["wait"])),
        })
    return {"policy": policy, "rollout": rollout, "blocks": rows, "steps": dict(steps)}


def _holiday_week(day: date) -> int:
    return int(date(2023, 11, 21) <= day <= date(2023, 11, 26) or date(2023, 12, 22) <= day <= date(2024, 1, 3))


def _real_v6() -> list[dict[str, Any]]:
    weather = ghcn_daily("nyc")
    rows = []
    with (RUN / "prepared/nyc_blocks.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            observed = date.fromisoformat(row["ride_date"])
            category = weather.get(observed, {}).get("weather", "clear")
            rows.append({
                "date": observed.isoformat(), "rain": int(category == "rain"), "snow": int(category == "snow"),
                "block": int(row["block"]), "weekday": int(row["day_of_week"]),
                "holiday_week": _holiday_week(observed), "log_trips": float(np.log(float(row["trips"]))),
                "price_index": float(row["price_index"]), "median_wait": float(row["median_wait"]),
            })
    return rows


def _regression(rows: Sequence[Mapping[str, Any]], outcome: str) -> dict[str, Any]:
    regressors = ["rain", "snow", "block", "weekday", "holiday_week"]
    design = [[1.0, *(float(row[key]) for key in regressors)] for row in rows]
    return {"regressors": ["intercept", *regressors], **ols_hc1(design, [float(row[outcome]) for row in rows])}


def _coefficient_comparison(real: Mapping[str, Any], synthetic: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for index, name in enumerate(real["regressors"]):
        rb, sb = float(real["coefficients"][index]), float(synthetic["coefficients"][index])
        rci, sci = real["ci95"][index], synthetic["ci95"][index]
        rows.append({
            "regressor": name, "real": rb, "synthetic": sb,
            "sign_agreement": bool(np.sign(rb) == np.sign(sb)),
            "ci_overlap": bool(max(rci[0], sci[0]) <= min(rci[1], sci[1])),
            "ratio": sb / rb if abs(rb) > 1e-12 else None,
        })
    return rows


def _load_active_drivers(policy: str) -> dict[tuple[str, int], float]:
    raw = json.loads((BASE / f"runs/v18/eval/E5/nyc/natural_year_2023/{policy}/raw.json").read_text())
    trace = next(iter(raw["traces"].values()))
    return {(row["rollout"], int(row["step"])): float(row["active_drivers_firm1"]) for row in trace}


def _number(text: str) -> float:
    value = str(text).strip().replace(",", "")
    return float(value) if value and value != "-" else float("nan")


def _v7_real() -> dict[str, dict[str, float]]:
    result = {}
    path = DATA / "raw/validation/nyc_tlc/data_reports_monthly.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            month = row["Month/Year"]
            if row["License Class"] != "FHV - High Volume" or month not in {"2023-09", "2023-10", "2023-11", "2023-12", "2024-01", "2024-02"}:
                continue
            trips = _number(row["Trips Per Day"])
            vehicles = _number(row["Vehicles Per Day"])
            hours = _number(row["Avg Hours Per Day Per Vehicle"])
            minutes = _number(row["Avg Minutes Per Trip"])
            vehicle_hours = vehicles * hours
            result[month] = {
                "trips_per_vehicle_hour": trips / vehicle_hours,
                "passenger_only_utilization": trips * minutes / 60.0 / vehicle_hours,
            }
    return result


def _v7_synthetic(outputs: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    active = _load_active_drivers("published_card")
    totals: dict[str, dict[str, float]] = defaultdict(lambda: {"trips": 0.0, "passenger_minutes": 0.0, "driver_hours": 0.0})
    for output in outputs:
        if output["policy"] != "published_card":
            continue
        rollout = str(output["rollout"])
        for step, item in output["steps"].items():
            if item["regime"] != "calm":
                continue
            month = (START_DATE + timedelta(days=int(item["day"]))).strftime("%Y-%m")
            drivers = active.get((rollout, int(step)))
            if drivers is None:
                continue
            totals[month]["trips"] += float(item["firm1_trips"])
            totals[month]["passenger_minutes"] += float(item["firm1_duration"])
            totals[month]["driver_hours"] += drivers * 4.0
    return {
        month: {
            "trips_per_vehicle_hour": item["trips"] / item["driver_hours"],
            "passenger_only_utilization": item["passenger_minutes"] / 60.0 / item["driver_hours"],
        }
        for month, item in totals.items() if item["driver_hours"] > 0
    }


def _trend(values: Mapping[str, Mapping[str, float]], months: Sequence[str], field: str) -> float:
    y = np.asarray([values[month][field] for month in months], dtype=float)
    return float(np.polyfit(np.arange(len(y), dtype=float), y, 1)[0]) if len(y) >= 2 else 0.0


def stage_v6_v7(*, workers: int, attempt: int = 1) -> dict[str, Any]:
    started = time.monotonic()
    append_progress(PROGRESS, job="realdata_V6_V7", state="running", attempt=attempt, seconds=0)
    policies = ("published_card", "best_fixed", "dated_two_lever", "count_rule_1step", "nyc_s101", "nyc_s202", "nyc_s303")
    tasks = []
    for policy in policies:
        files = sorted((BASE / f"runs/v18/eval/E5/nyc/natural_year_2023/{policy}/requests").glob("*.csv.gz"))
        if len(files) != 6:
            raise ValueError(f"V6 expected 6 E5 logs for {policy}, found {len(files)}")
        tasks.extend((policy, str(path)) for path in files)
    outputs = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_e5_file, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), start=1):
            outputs.append(future.result())
            if index % 6 == 0:
                print(f"processed {index}/{len(tasks)} E5 logs", flush=True)

    real_rows = _real_v6()
    outcomes = ("log_trips", "price_index", "median_wait")
    real_results = {outcome: _regression(real_rows, outcome) for outcome in outcomes}
    by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for output in outputs:
        by_policy[output["policy"]].extend(output["blocks"])
        if output["policy"].startswith("nyc_s"):
            by_policy["agent_pooled"].extend(output["blocks"])
    synthetic_results = {
        policy: {outcome: _regression(rows, outcome) for outcome in outcomes}
        for policy, rows in by_policy.items()
    }
    comparisons = []
    for policy, results in synthetic_results.items():
        for outcome in outcomes:
            comparisons.extend({"policy": policy, "outcome": outcome, **row} for row in _coefficient_comparison(real_results[outcome], results[outcome]))
    atomic_json(RUN / "v6_regression/nyc_metrics.json", {
        "state": "complete", "regressors": ["rain", "snow", "block", "weekday", "holiday_week"],
        "temperature_bins": "omitted on both sides because Last Revision limits V6 to simulator weather categories",
        "real": real_results, "synthetic": synthetic_results,
    })
    write_csv(RUN / "v6_regression/comparison.csv", comparisons)
    atomic_json(RUN / "v6_regression/chicago_status.json", {
        "state": "skipped_dependency_failed", "dependency": "D1 Chicago source failed after 3 attempts",
    })

    real_v7, synthetic_v7 = _v7_real(), _v7_synthetic(outputs)
    common = sorted(set(real_v7) & set(synthetic_v7) & {"2023-09", "2023-10", "2023-11", "2023-12", "2024-01", "2024-02"})
    monthly = []
    for month in sorted(real_v7):
        synthetic = synthetic_v7.get(month)
        monthly.append({
            "month": month, "real_trips_per_vehicle_hour": real_v7[month]["trips_per_vehicle_hour"],
            "real_utilization": real_v7[month]["passenger_only_utilization"],
            "synthetic_trips_per_vehicle_hour": synthetic["trips_per_vehicle_hour"] if synthetic else None,
            "synthetic_utilization": synthetic["passenger_only_utilization"] if synthetic else None,
            "ratio": synthetic["trips_per_vehicle_hour"] / real_v7[month]["trips_per_vehicle_hour"] if synthetic else None,
        })
    result_v7 = {
        "state": "complete_with_calendar_limitation", "monthly": monthly, "common_months": common,
        "missing_synthetic_months": sorted(set(real_v7) - set(synthetic_v7)),
        "calendar_note": "E5 natural_year_2023 ends in Dec 2023, so Jan-Feb 2024 have no synthetic counterpart",
        "trend": {
            "real": _trend(real_v7, common, "trips_per_vehicle_hour"),
            "synthetic": _trend(synthetic_v7, common, "trips_per_vehicle_hour"),
        },
    }
    result_v7["trend"]["agreement"] = bool(np.sign(result_v7["trend"]["real"]) == np.sign(result_v7["trend"]["synthetic"]))
    atomic_json(RUN / "v7_supply/nyc_metrics.json", result_v7)
    seconds = time.monotonic() - started
    append_progress(PROGRESS, job="realdata_V6_V7", state="complete", attempt=attempt, seconds=seconds)
    return {"e5_logs": len(outputs), "v6_comparison_rows": len(comparisons), "v7_months": len(monthly), "seconds": seconds}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("v5", "v6v7"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.workers <= 10:
        parser.error("--workers must be 1..10")
    result = stage_v5(workers=args.workers, attempt=args.attempt) if args.stage == "v5" else stage_v6_v7(workers=args.workers, attempt=args.attempt)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
