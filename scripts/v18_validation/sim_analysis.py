#!/usr/bin/env python3
"""V2-V4 simulated-side analysis from completed V18 request logs."""

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
import statistics
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
RUN = BASE / "runs/v18/realdata"
PROGRESS = BASE / "runs/v18/controller/progress.jsonl"
sys.path.insert(0, str(BASE))

from scripts.v18_runtime import append_progress  # noqa: E402
from scripts.v18_validation.analyze import (  # noqa: E402
    _bank_frame,
    atomic_json,
    nyc_sample,
    utc_now,
    write_csv,
)
from scripts.v18_validation.metrics import (  # noqa: E402
    distribution_summary,
    ks,
    total_variation,
    variability,
    wasserstein1,
)


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def _rollout(path: Path) -> str:
    name = path.name
    return name[:-7] if name.endswith(".csv.gz") else path.stem


def _read_tripmix(files: Sequence[Path]) -> dict[str, dict[str, np.ndarray]]:
    values = {
        scope: {key: [] for key in ("distance", "duration", "hour", "weekday", "airport")}
        for scope in ("all_requests", "completed")
    }
    for path in files:
        with gzip.open(path, "rt", newline="") as stream:
            for row in csv.DictReader(stream):
                distance = _float(row.get("TravelDistance"))
                duration = _float(row.get("DurationMinutes"))
                hour = _float(row.get("Hour"))
                weekday = _float(row.get("DayOfWeek"))
                airport = 1.0 if _bool(row.get("Airport")) else 0.0
                if not all(np.isfinite(x) for x in (distance, duration, hour, weekday)):
                    continue
                scopes = ["all_requests"]
                if _bool(row.get("Fulfilled")) and str(row.get("Choice")) in {"Firm1", "Firm2"}:
                    scopes.append("completed")
                for scope in scopes:
                    values[scope]["distance"].append(distance)
                    values[scope]["duration"].append(duration)
                    values[scope]["hour"].append(hour)
                    values[scope]["weekday"].append(weekday)
                    values[scope]["airport"].append(airport)
    return {
        scope: {key: np.asarray(item, dtype=float) for key, item in columns.items()}
        for scope, columns in values.items()
    }


def _shares(values: np.ndarray, levels: Iterable[int]) -> dict[int, float]:
    return {level: float(np.mean(values == level)) for level in levels}


def _tripmix_row(name: str, real: Mapping[str, np.ndarray], other: Mapping[str, np.ndarray]) -> dict[str, Any]:
    real_block = (real["hour"].astype(int) // 4).clip(0, 5)
    other_block = (other["hour"].astype(int) // 4).clip(0, 5)
    real_weekday = real["weekday"].astype(int)
    other_weekday = other["weekday"].astype(int)
    speed_real = 60.0 * real["distance"] / np.maximum(real["duration"], 1e-12)
    speed_other = 60.0 * other["distance"] / np.maximum(other["duration"], 1e-12)
    return {
        "comparison": name,
        "n_real": int(real["distance"].size), "n_other": int(other["distance"].size),
        "distance": {
            "real": distribution_summary(real["distance"]), "other": distribution_summary(other["distance"]),
            "ks": ks(real["distance"], other["distance"]),
            "wasserstein1": wasserstein1(real["distance"], other["distance"]),
        },
        "duration": {
            "real": distribution_summary(real["duration"]), "other": distribution_summary(other["duration"]),
            "ks": ks(real["duration"], other["duration"]),
            "wasserstein1": wasserstein1(real["duration"], other["duration"]),
        },
        "hour_block_shares_real": _shares(real_block, range(6)),
        "hour_block_shares_other": _shares(other_block, range(6)),
        "tvd_hour": total_variation(_shares(real_block, range(6)), _shares(other_block, range(6))),
        "weekday_shares_real": _shares(real_weekday, range(7)),
        "weekday_shares_other": _shares(other_weekday, range(7)),
        "tvd_weekday": total_variation(_shares(real_weekday, range(7)), _shares(other_weekday, range(7))),
        "airport_share_real": float(real["airport"].mean()),
        "airport_share_other": float(other["airport"].mean()),
        "airport_share_gap": float(other["airport"].mean() - real["airport"].mean()),
        "median_speed_by_block_real": {str(block): float(np.median(speed_real[real_block == block])) for block in range(6)},
        "median_speed_by_block_other": {str(block): float(np.median(speed_other[other_block == block])) for block in range(6)},
    }


def stage_v2(*, attempt: int = 1) -> dict[str, Any]:
    started = time.monotonic()
    append_progress(PROGRESS, job="realdata_V2", state="running", attempt=attempt, seconds=0)
    e10 = sorted((BASE / "runs/v18/eval/E10/nyc/C0/published_card/requests").glob("*.csv.gz"))
    twins = sorted((RUN / "v3_events/sim/nyc").glob("*/twin/published_card/requests/*.csv.gz"))
    files = e10 + twins
    if len(e10) != 10 or len(twins) != 50:
        raise ValueError(f"V2 expected 10 E10 and 50 twin logs, found {len(e10)} and {len(twins)}")
    simulated = _read_tripmix(files)
    sample = nyc_sample()
    real = {
        "distance": sample["distance_miles"].to_numpy(float),
        "duration": sample["duration_minutes"].to_numpy(float),
        "hour": sample["hour"].to_numpy(float),
        "weekday": sample["day_of_week"].to_numpy(float),
        "airport": sample["airport"].to_numpy(float),
    }
    bank_frame = _bank_frame("nyc")
    bank = {
        "distance": bank_frame["distance_miles"].to_numpy(float),
        "duration": bank_frame["duration_minutes"].to_numpy(float),
        "hour": bank_frame["hour"].to_numpy(float),
        "weekday": bank_frame["day_of_week"].to_numpy(float),
        "airport": bank_frame["airport"].to_numpy(float),
    }
    result = {
        "state": "complete", "computed_utc": utc_now(),
        "sources": {"e10_logs": len(e10), "calm_twin_logs": len(twins)},
        "rows": [
            _tripmix_row("simulated_completed_vs_window", real, simulated["completed"]),
            _tripmix_row("simulated_all_requests_vs_window", real, simulated["all_requests"]),
            _tripmix_row("bank_vs_window_drift", real, bank),
        ],
    }
    atomic_json(RUN / "v2_tripmix/nyc_metrics.json", result)
    atomic_json(RUN / "v2_tripmix/chicago_metrics.json", {
        "state": "skipped_dependency_failed", "dependency": "D1 Chicago source failed after 3 attempts",
    })
    seconds = time.monotonic() - started
    append_progress(PROGRESS, job="realdata_V2", state="complete", attempt=attempt, seconds=seconds)
    return {"logs": len(files), "seconds": seconds, "completed_rows": int(simulated["completed"]["distance"].size)}


def _scenario(city: str, event: str) -> tuple[Path, list[int], date]:
    if event.startswith("E1_"):
        path = BASE / "scenarios/validation/nyc_e1_flood_20230929.json"
    elif event.startswith("E2_"):
        path = BASE / "scenarios/validation/e2_thanksgiving_2023.json"
    elif event.startswith("E3_"):
        path = BASE / "scenarios/validation/e3_christmas_2023.json"
    else:
        path = BASE / f"scenarios/validation/{city}_{event.lower()}.json"
    payload = json.loads(path.read_text())
    target = payload["events"][0]
    days = list(range(int(target["start_day"]), int(target["end_day"]) + 1))
    return path, days, date.fromisoformat(payload["start_date"])


def _empty_day() -> dict[str, Any]:
    return {
        "dow": None, "completed": 0, "duration": [], "speed": [], "airport": [],
        "log_pi": [], "log_pi_firm1": [], "log_pi_firm2": [], "fare_per_mile": [], "wait": [],
    }


def _aggregate_log(path: Path, keep_days: set[int] | None) -> dict[int, dict[str, float]]:
    by_day: dict[int, dict[str, Any]] = defaultdict(_empty_day)
    with gzip.open(path, "rt", newline="") as stream:
        for row in csv.DictReader(stream):
            day = int(row["Day"])
            if keep_days is not None and day not in keep_days:
                continue
            target = by_day[day]
            target["dow"] = int(row["DayOfWeek"])
            if not (_bool(row.get("Fulfilled")) and str(row.get("Choice")) in {"Firm1", "Firm2"}):
                continue
            firm = str(row["Choice"])
            distance = _float(row.get("TravelDistance"))
            duration = _float(row.get("DurationMinutes"))
            airport = 1.0 if _bool(row.get("Airport")) else 0.0
            wait = _float(row.get("WaitMinutes"))
            card = _float(row.get("CardClearQuote"))
            fare = _float(row.get("Price_Firm1" if firm == "Firm1" else "Price_Firm2"))
            target["completed"] += 1
            if np.isfinite(duration):
                target["duration"].append(duration)
            if np.isfinite(distance) and np.isfinite(duration) and duration > 0:
                target["speed"].append(60.0 * distance / duration)
            target["airport"].append(airport)
            if np.isfinite(fare) and np.isfinite(card) and fare > 0 and card > 0:
                log_pi = float(np.log(fare / card))
                target["log_pi"].append(log_pi)
                target["log_pi_firm1" if firm == "Firm1" else "log_pi_firm2"].append(log_pi)
            if np.isfinite(fare) and np.isfinite(distance) and distance > 0:
                target["fare_per_mile"].append(fare / distance)
            if np.isfinite(wait):
                target["wait"].append(wait)
    result: dict[int, dict[str, float]] = {}
    for day, item in by_day.items():
        result[day] = {
            "dow": int(item["dow"]),
            "m1_completed_trips": float(item["completed"]),
            "m2_mean_duration": float(np.mean(item["duration"])),
            "m3_median_speed": float(np.median(item["speed"])),
            "m4_airport_share": float(np.mean(item["airport"])),
            "m5_price_index": float(np.exp(np.mean(item["log_pi"]))),
            "m5_price_index_firm1": float(np.exp(np.mean(item["log_pi_firm1"]))) if item["log_pi_firm1"] else float("nan"),
            "m5_price_index_firm2": float(np.exp(np.mean(item["log_pi_firm2"]))) if item["log_pi_firm2"] else float("nan"),
            "m6_fare_per_mile": float(np.mean(item["fare_per_mile"])),
            "m7_wait": float(np.mean(item["wait"])),
        }
    return result


def _process_pair(task: tuple[str, str, str, Path, Path, list[int], str]) -> dict[str, Any]:
    city, event, policy, event_path, twin_path, days, start_text = task
    target = set(days)
    event_values = _aggregate_log(event_path, target)
    twin_values = _aggregate_log(twin_path, None)
    effects = []
    start = date.fromisoformat(start_text)
    for day in days:
        if day not in event_values or day not in twin_values:
            continue
        for metric in (
            "m1_completed_trips", "m2_mean_duration", "m3_median_speed", "m4_airport_share",
            "m5_price_index", "m5_price_index_firm1", "m5_price_index_firm2",
            "m6_fare_per_mile", "m7_wait",
        ):
            event_value = event_values[day][metric]
            twin_value = twin_values[day][metric]
            if np.isfinite(event_value) and np.isfinite(twin_value) and twin_value != 0:
                effects.append({
                    "day": day, "date": (start + timedelta(days=day)).isoformat(),
                    "metric": metric, "effect": event_value / twin_value - 1.0,
                })
    twin_daily = [
        {"day": day, "date": (start + timedelta(days=day)).isoformat(), **values}
        for day, values in twin_values.items()
    ]
    return {
        "city": city, "event": event, "policy": policy, "rollout": _rollout(event_path),
        "effects": effects, "twin_daily": twin_daily,
    }


def _pair_tasks() -> list[tuple[str, str, str, Path, Path, list[int], str]]:
    tasks = []
    root = RUN / "v3_events/sim"
    for city_root in sorted(path for path in root.iterdir() if path.is_dir()):
        city = city_root.name
        for event_root in sorted(path for path in city_root.iterdir() if path.is_dir()):
            event = event_root.name
            _, days, start = _scenario(city, event)
            for policy_root in sorted(path for path in (event_root / "event").iterdir() if path.is_dir()):
                policy = policy_root.name
                event_files = {path.name: path for path in (policy_root / "requests").glob("*.csv.gz")}
                twin_files = {path.name: path for path in (event_root / "twin" / policy / "requests").glob("*.csv.gz")}
                if set(event_files) != set(twin_files) or len(event_files) != 10:
                    raise ValueError(f"unpaired V3 request logs for {city}/{event}/{policy}")
                tasks.extend((city, event, policy, event_files[name], twin_files[name], days, start.isoformat()) for name in sorted(event_files))
    if len(tasks) != 630:
        raise ValueError(f"expected 630 paired rollout files, found {len(tasks)}")
    return tasks


def _pooled_effect_rows(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    samples: dict[tuple[str, str, str, str, str], list[float]] = defaultdict(list)
    for result in results:
        for row in result["effects"]:
            key = (result["city"], result["event"], result["policy"], row["date"], row["metric"])
            samples[key].append(float(row["effect"]))
            if "_s" in result["policy"]:
                agent_key = (result["city"], result["event"], "agent_pooled", row["date"], row["metric"])
                samples[agent_key].append(float(row["effect"]))
    rows = []
    for (city, event, policy, day, metric), sample in sorted(samples.items()):
        values = np.asarray(sample, dtype=float)
        rows.append({
            "city": city, "event": event, "policy": policy, "date": day, "metric": metric,
            "n_rollouts": len(values), "sim_mean": float(values.mean()),
            "sim_ci_low": float(np.quantile(values, 0.025)), "sim_ci_high": float(np.quantile(values, 0.975)),
        })
    return rows


def _comparison(sim_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    with (RUN / "v3_events/nyc_real.csv").open(newline="", encoding="utf-8") as stream:
        real_rows = list(csv.DictReader(stream))
    real = {(row["event"], row["date"], row["metric"]): row for row in real_rows}
    rows = []
    for sim in sim_rows:
        if sim["city"] != "nyc" or sim["metric"] not in {
            "m1_completed_trips", "m2_mean_duration", "m3_median_speed", "m4_airport_share",
            "m5_price_index", "m6_fare_per_mile", "m7_wait",
        }:
            continue
        observed = real.get((sim["event"], sim["date"], sim["metric"]))
        if observed is None:
            continue
        effect = float(observed["real_effect"])
        band = float(observed["normal_day_band"])
        mean, low, high = float(sim["sim_mean"]), float(sim["sim_ci_low"]), float(sim["sim_ci_high"])
        if abs(effect) <= band:
            label = "real not distinguishable"
        elif effect - band <= mean <= effect + band:
            label = "within band"
        elif (effect > 0 and low > 0) or (effect < 0 and high < 0):
            label = "same direction"
        elif (effect > 0 and high < 0) or (effect < 0 and low > 0):
            label = "opposite"
        else:
            label = "inconclusive"
        rows.append({**dict(sim), "real_effect": effect, "normal_day_band": band, "label": label})
    return rows


def _mean_nested(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in items[0]:
        values = [item[key] for item in items]
        if isinstance(values[0], Mapping):
            result[key] = _mean_nested(values)  # type: ignore[arg-type]
        elif isinstance(values[0], (int, float)):
            result[key] = float(np.nanmean(np.asarray(values, dtype=float)))
    return result


def _v4_variability(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    per_policy: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        for firm, metric in (("pooled", "m5_price_index"), ("firm1", "m5_price_index_firm1"), ("firm2", "m5_price_index_firm2")):
            rows = [
                (item["date"], int(item["dow"]), float(item[metric]))
                for item in result["twin_daily"] if np.isfinite(float(item[metric]))
            ]
            if len(rows) >= 3:
                per_policy[(result["city"], result["policy"], firm)].append(variability(rows))
    output = []
    for (city, policy, firm), samples in sorted(per_policy.items()):
        output.append({"city": city, "policy": policy, "firm": firm, "rollouts": len(samples), **_mean_nested(samples)})
    return output


def _twin_checks() -> dict[str, Any]:
    rows = []
    for city_root in sorted((RUN / "v3_events/sim").iterdir()):
        if not city_root.is_dir():
            continue
        for event_root in sorted(path for path in city_root.iterdir() if path.is_dir()):
            _, target_days, _ = _scenario(city_root.name, event_root.name)
            allowed = set(target_days)
            if event_root.name.startswith(("E1_", "E4_")):
                allowed |= {max(target_days) + 1, max(target_days) + 2}
            for policy_root in sorted(path for path in (event_root / "event").iterdir() if path.is_dir()):
                event_raw = json.loads((policy_root / "raw.json").read_text())
                twin_raw = json.loads((event_root / "twin" / policy_root.name / "raw.json").read_text())
                event_trace = next(iter(event_raw["traces"].values()))
                twin_trace = next(iter(twin_raw["traces"].values()))
                event_map = {(row["rollout"], int(row["step"])): tuple(row.get("shock_active", ())) for row in event_trace}
                twin_map = {(row["rollout"], int(row["step"])): tuple(row.get("shock_active", ())) for row in twin_trace}
                differences = sorted({step // 6 for key, value in event_map.items() for step in [key[1]] if value != twin_map.get(key, ())})
                rows.append({
                    "city": city_root.name, "event": event_root.name, "policy": policy_root.name,
                    "difference_days": differences, "allowed_days": sorted(allowed),
                    "unexpected_days": sorted(set(differences) - allowed),
                })
    return {"state": "complete", "checks": len(rows), "unexpected_checks": sum(bool(row["unexpected_days"]) for row in rows), "rows": rows}


def stage_v3v4(*, workers: int, attempt: int = 1) -> dict[str, Any]:
    started = time.monotonic()
    append_progress(PROGRESS, job="realdata_V3_V4_sim", state="running", attempt=attempt, seconds=0)
    tasks = _pair_tasks()
    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_process_pair, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if index % 50 == 0:
                print(f"processed {index}/{len(tasks)} paired rollout logs", flush=True)
    sim_rows = _pooled_effect_rows(results)
    comparison = _comparison(sim_rows)
    write_csv(RUN / "v3_events/sim_effects.csv", [row for row in sim_rows if row["metric"] in {
        "m1_completed_trips", "m2_mean_duration", "m3_median_speed", "m4_airport_share",
        "m5_price_index", "m6_fare_per_mile", "m7_wait",
    }])
    write_csv(RUN / "v3_events/comparison.csv", comparison)
    twin_checks = _twin_checks()
    atomic_json(RUN / "v3_events/twin_checks.json", twin_checks)
    write_csv(RUN / "v4_surge/variability.csv", _v4_variability(results))
    write_csv(RUN / "v4_surge/uplift.csv", [row for row in sim_rows if row["metric"].startswith("m5_price_index")])
    atomic_json(RUN / "v4_surge/sim_status.json", {
        "state": "complete", "paired_files": len(results), "workers": workers,
        "one_numerical_thread_per_process": True,
    })
    seconds = time.monotonic() - started
    append_progress(PROGRESS, job="realdata_V3_V4_sim", state="complete", attempt=attempt, seconds=seconds)
    return {
        "pairs": len(results), "sim_rows": len(sim_rows), "comparison_rows": len(comparison),
        "twin_unexpected_checks": twin_checks["unexpected_checks"], "seconds": seconds,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("v2", "v3v4"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.workers <= 10:
        parser.error("--workers must be 1..10")
    result = stage_v2(attempt=args.attempt) if args.stage == "v2" else stage_v3v4(workers=args.workers, attempt=args.attempt)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
