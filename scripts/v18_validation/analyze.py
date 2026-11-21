#!/usr/bin/env python3
"""Resumable V18 real-data validation analysis (Last Revision, Section 6).

Scientific disagreements are products, not stop conditions.  Missing source
data and rows rejected by leakage guards are recorded explicitly and the
independent checks continue.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
from datetime import date, datetime, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import duckdb
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[2]
ROOT = BASE.parent
DATA = ROOT / "data"
RUN = BASE / "runs/v18/realdata"
DECLARATION = BASE / "scenarios/validation/DECLARATION.json"
PROGRESS = BASE / "runs/v18/controller/progress.jsonl"

sys.path.insert(0, str(BASE))

from MarketInteraction import MarketInteraction, RideContext  # noqa: E402
from city_market_calibration import fit_tariff_anchors  # noqa: E402
from market_dynamics import default_duration_minutes  # noqa: E402
from scripts.v18_runtime import append_progress  # noqa: E402
from scripts.v18_validation.metrics import (  # noqa: E402
    day_block_bootstrap,
    fare_metrics,
    finite_tree,
    sanitize_tree,
    total_variation,
    variability,
)

WINDOW_START = date(2023, 9, 1)
WINDOW_END = date(2024, 2, 29)
BOOTSTRAP_DRAWS = 1000
SEED = 20261112
TLC_FILES = [
    DATA / "raw/validation/nyc_tlc" / f"fhvhv_tripdata_{month}.parquet"
    for month in ("2023-09", "2023-10", "2023-11", "2023-12", "2024-01", "2024-02")
]
FEDERAL_HOLIDAYS = {
    date(2023, 9, 4), date(2023, 10, 9), date(2023, 11, 10),
    date(2023, 11, 23), date(2023, 12, 25), date(2024, 1, 1),
    date(2024, 1, 15), date(2024, 2, 19),
}


def utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def atomic_json(path: Path, value: Any) -> None:
    value = sanitize_tree(value)
    if not finite_tree(value):
        raise ValueError(f"non-finite output for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_declaration() -> dict[str, Any]:
    value = json.loads(DECLARATION.read_text(encoding="utf-8"))
    if value.get("schema_version") != "v18-realdata-declaration-v1":
        raise ValueError("invalid V18 real-data declaration")
    if value.get("declared_before_metrics") is not True:
        raise ValueError("declaration does not predate metrics")
    return value


def load_tariff(city: str) -> dict[str, Any]:
    payload = json.loads((BASE / f"calibration_artifacts/v17/{city}/calibration.json").read_text())
    return dict(payload.get("calibration", payload)["tariff"])


def market(city: str, tariff: Mapping[str, Any] | None = None) -> MarketInteraction:
    names = {"nyc": "New York City", "boston": "Boston", "chicago": "Chicago"}
    value = MarketInteraction(names[city], seed=17)
    value.register_calibrated_tariff(dict(tariff or load_tariff(city)))
    value.configure_fare_model("realistic")
    return value


def quote_rows(city: str, frame: pd.DataFrame, tariff: Mapping[str, Any] | None = None) -> np.ndarray:
    mi = market(city, tariff)
    values = np.empty(len(frame), dtype=float)
    for index, row in enumerate(frame.itertuples(index=False)):
        context = RideContext(
            day_of_week=int(row.day_of_week), weather=str(row.weather), hour=int(row.hour),
            airport=bool(row.airport), service=str(row.service),
        )
        values[index] = mi.quote_price(float(row.distance_miles), float(row.duration_minutes), context)
    return values


def ghcn_daily(city: str) -> dict[date, dict[str, float | str]]:
    station = "USW00014732" if city == "nyc" else "USW00094846"
    path = DATA / "raw/validation/ghcnd" / f"{station}.csv"
    result: dict[date, dict[str, float | str]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            observed = date.fromisoformat(row["DATE"][:10])
            if observed < WINDOW_START - timedelta(days=60) or observed > WINDOW_END + timedelta(days=60):
                continue
            prcp = float(row["PRCP"].strip() or 0.0)
            snow = float(row["SNOW"].strip() or 0.0)
            weather = "snow" if snow > 0 else ("rain" if prcp > 0 else "clear")
            result[observed] = {"prcp_raw": prcp, "snow_raw": snow, "weather": weather}
    return result


def nyc_sample_query() -> str:
    files = ",".join(repr(str(path)) for path in TLC_FILES)
    return f"""
      WITH eligible AS (
        SELECT pickup_datetime, request_datetime, dropoff_datetime,
               CAST(pickup_datetime AS DATE) AS ride_date,
               hvfhs_license_num, PULocationID, DOLocationID,
               trip_miles, trip_time / 60.0 AS duration_minutes,
               base_passenger_fare, driver_pay,
               shared_match_flag, access_a_ride_flag, wav_request_flag,
               EXTRACT(hour FROM pickup_datetime)::INTEGER AS hour,
               ((EXTRACT(dow FROM pickup_datetime)::INTEGER + 6) % 7) AS day_of_week
        FROM read_parquet([{files}])
        WHERE hvfhs_license_num IN ('HV0003','HV0005')
          AND trip_miles > 0.05 AND trip_miles < 60
          AND trip_time > 30 AND base_passenger_fare > 0
          AND COALESCE(shared_match_flag, 'N') <> 'Y'
      ), ranked AS (
        SELECT *, ROW_NUMBER() OVER (
          PARTITION BY strftime(ride_date, '%Y-%m')
          ORDER BY hash(pickup_datetime, request_datetime, PULocationID, DOLocationID, trip_miles)
        ) AS sample_rank
        FROM eligible
      )
      SELECT * EXCLUDE(sample_rank) FROM ranked WHERE sample_rank <= 30000
      ORDER BY ride_date, pickup_datetime
    """


def nyc_sample() -> pd.DataFrame:
    cache = RUN / "cache/nyc_v1_sample.parquet"
    cache.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    if not cache.exists():
        connection.execute(f"COPY ({nyc_sample_query()}) TO {repr(str(cache))} (FORMAT PARQUET, COMPRESSION ZSTD)")
    frame = connection.execute(f"SELECT * FROM read_parquet({repr(str(cache))})").fetchdf()
    weather = ghcn_daily("nyc")
    frame["weather"] = [weather.get(pd.Timestamp(value).date(), {}).get("weather", "clear") for value in frame["ride_date"]]
    frame["airport"] = frame["PULocationID"].isin((1, 132, 138)) | frame["DOLocationID"].isin((1, 132, 138))
    frame["service"] = "economy"
    frame["provider"] = frame["hvfhs_license_num"].map({"HV0003": "Uber", "HV0005": "Lyft"})
    frame["actual_price"] = frame["base_passenger_fare"].astype(float)
    frame["distance_miles"] = frame["trip_miles"].astype(float)
    frame["day_group"] = frame["ride_date"].astype(str)
    frame["block"] = frame["hour"] // 4
    frame["wait_minutes"] = (
        (pd.to_datetime(frame["pickup_datetime"]) - pd.to_datetime(frame["request_datetime"])).dt.total_seconds() / 60.0
    )
    return frame


def stage_guards() -> dict[str, Any]:
    append_progress(PROGRESS, job="realdata_guards", state="running", attempt=1, seconds=0)
    raw_kaggle = DATA / "raw/rides/nyc_kaggle/rideshare_data.parquet"
    nyc_bank = DATA / "processed/nyc/trip_bank.csv"
    connection = duckdb.connect()
    bank_max = connection.execute(
        f"SELECT max(CAST(pickup_datetime AS TIMESTAMP)) FROM read_csv_auto({repr(str(nyc_bank))}, header=true)"
    ).fetchone()[0]
    validation_min = connection.execute(
        f"SELECT min(pickup_datetime) FROM read_parquet({repr(str(TLC_FILES[0]))})"
    ).fetchone()[0]

    boston_raw = DATA / "raw/rides/boston_kaggle/rideshare_kaggle.csv"
    boston_bank = DATA / "processed/boston/trip_bank.csv"
    boston = connection.execute(f"""
      WITH raw AS (
        SELECT id, CAST(datetime AS TIMESTAMP) AS dt
        FROM read_csv_auto({repr(str(boston_raw))}, header=true, ignore_errors=true)
        WHERE price IS NOT NULL
      ), bank AS (
        SELECT source_id FROM read_csv_auto({repr(str(boston_bank))}, header=true)
      )
      SELECT min(dt), max(dt),
             count(*) FILTER (WHERE dt >= TIMESTAMP '2018-12-10'),
             count(*) FILTER (WHERE dt >= TIMESTAMP '2018-12-10' AND id IN (SELECT source_id FROM bank))
      FROM raw
    """).fetchone()
    search = subprocess.run(
        ["rg", "-l", "data/raw/validation|runs/v18/realdata", "runs/v18/train", "runs/v18/eval", "calibration_artifacts", "-g", "*.json"],
        cwd=BASE, capture_output=True, text=True, check=False,
    )
    contamination = [line for line in search.stdout.splitlines() if line]
    result = {
        "schema": "v18-realdata-guards-v1",
        "checked_utc": utc_now(),
        "state": "complete_with_exclusions",
        "nyc": {
            "raw_kaggle_source": str(raw_kaggle),
            "raw_kaggle_available": raw_kaggle.exists(),
            "guard_basis": "processed_trip_bank_conservative_proxy" if not raw_kaggle.exists() else "raw_kaggle",
            "fit_proxy_max": str(bank_max),
            "validation_min": str(validation_min),
            "after_fit_proxy": bool(validation_min > bank_max),
            "excluded_rows": 0,
            "limitation": "Raw NYC Kaggle parquet is absent; exact source-date and August price-definition join cannot be computed."
            if not raw_kaggle.exists() else None,
        },
        "chicago": {
            "state": "failed_after_3_attempts",
            "dependent_real_side_checks": ["V1", "V2", "V3", "V4", "V6"],
        },
        "boston": {
            "raw_min": str(boston[0]), "raw_max": str(boston[1]),
            "test_rows_before_bank_exclusion": int(boston[2]),
            "test_rows_in_bank_excluded": int(boston[3]),
            "fit_end": "2018-12-09", "test_start": "2018-12-10",
        },
        "validation_paths_in_training_or_evaluation_products": contamination,
        "path_guard_passed": not contamination,
    }
    atomic_json(RUN / "guards.json", result)
    append_progress(PROGRESS, job="realdata_guards", state="complete", attempt=1, seconds=0)
    return result


def _bank_frame(city: str) -> pd.DataFrame:
    path = DATA / f"processed/{city}/trip_bank.csv"
    frame = pd.read_csv(path, low_memory=False)
    frame = frame[
        np.isfinite(pd.to_numeric(frame["actual_price"], errors="coerce"))
        & (pd.to_numeric(frame["actual_price"], errors="coerce") > 0)
        & (pd.to_numeric(frame["distance_miles"], errors="coerce") > 0.05)
        & (pd.to_numeric(frame["distance_miles"], errors="coerce") < 60)
        & ~frame["shared_ride"].astype(str).str.lower().isin(("true", "1", "yes"))
    ].copy()
    frame["duration_minutes"] = pd.to_numeric(frame["duration_minutes"], errors="coerce")
    missing = ~np.isfinite(frame["duration_minutes"])
    if missing.any():
        frame.loc[missing, "duration_minutes"] = default_duration_minutes(
            pd.to_numeric(frame.loc[missing, "distance_miles"]).to_numpy(),
            pd.to_numeric(frame.loc[missing, "hour"]).to_numpy(),
        )
    frame["weather"] = frame["weather"].fillna("clear")
    frame["service"] = frame["service"].fillna("economy")
    frame["airport"] = frame["airport"].astype(str).str.lower().isin(("true", "1", "yes"))
    frame["actual_price"] = pd.to_numeric(frame["actual_price"])
    frame["distance_miles"] = pd.to_numeric(frame["distance_miles"])
    frame["pickup_datetime"] = pd.to_datetime(frame["pickup_datetime"], utc=True, errors="coerce")
    frame["day_group"] = frame["pickup_datetime"].dt.date.astype(str)
    frame["block"] = pd.to_numeric(frame["hour"]) // 4
    return frame


def _baseline_predictions(bank: pd.DataFrame, validation: pd.DataFrame) -> dict[str, np.ndarray]:
    price = bank["actual_price"].to_numpy(float)
    distance = bank["distance_miles"].to_numpy(float)
    duration = bank["duration_minutes"].to_numpy(float)
    airport = bank["airport"].to_numpy(float)
    b1 = np.full(len(validation), price.mean())
    beta2 = np.linalg.lstsq(np.column_stack((np.ones(len(bank)), distance)), price, rcond=None)[0]
    beta3 = np.linalg.lstsq(
        np.column_stack((np.ones(len(bank)), distance, duration, airport)), price, rcond=None,
    )[0]
    return {
        "B1_mean": b1,
        "B2_distance": np.column_stack((np.ones(len(validation)), validation["distance_miles"].to_numpy(float))) @ beta2,
        "B3_linear": np.column_stack((
            np.ones(len(validation)), validation["distance_miles"].to_numpy(float),
            validation["duration_minutes"].to_numpy(float), validation["airport"].to_numpy(float),
        )) @ beta3,
    }


def _fare_row(name: str, actual: np.ndarray, quote: np.ndarray, groups: Sequence[str]) -> dict[str, Any]:
    metrics = fare_metrics(actual, quote)
    metrics["ci95_day_block"] = day_block_bootstrap(
        actual, quote, groups, draws=BOOTSTRAP_DRAWS, seed=SEED,
    )
    return {"name": name, **metrics}


def _nyc_breakdowns(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    distance_bins = pd.cut(frame["distance_miles"], [-np.inf, 2, 5, 10, np.inf], labels=("0-2", "2-5", "5-10", "10+"))
    dimensions = {
        "distance": distance_bins.astype(str),
        "block": frame["block"].astype(str),
        "airport": frame["airport"].astype(str),
        "month": pd.to_datetime(frame["ride_date"]).dt.strftime("%Y-%m"),
        "provider": frame["provider"].astype(str),
    }
    for dimension, values in dimensions.items():
        for level in sorted(values.unique()):
            selected = values == level
            if selected.sum() < 2:
                continue
            rows.append({"dimension": dimension, "level": level, **fare_metrics(
                frame.loc[selected, "actual_price"], frame.loc[selected, "card_quote"],
            )})
    return rows


def _boston_frames() -> tuple[pd.DataFrame, pd.DataFrame, int]:
    raw = DATA / "raw/rides/boston_kaggle/rideshare_kaggle.csv"
    columns = ["id", "datetime", "hour", "source", "destination", "cab_type", "name", "price", "distance", "surge_multiplier", "short_summary"]
    frame = pd.read_csv(raw, usecols=columns, low_memory=False)
    frame["pickup_datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    frame["actual_price"] = pd.to_numeric(frame["price"], errors="coerce")
    frame["distance_miles"] = pd.to_numeric(frame["distance"], errors="coerce")
    frame = frame[
        frame["pickup_datetime"].notna() & (frame["actual_price"] > 0)
        & (frame["distance_miles"] > 0.05) & (frame["distance_miles"] < 60)
        & ~frame["name"].fillna("").str.lower().str.contains("shared|pool")
    ].copy()
    frame["hour"] = frame["pickup_datetime"].dt.hour
    frame["day_of_week"] = frame["pickup_datetime"].dt.dayofweek
    frame["duration_minutes"] = default_duration_minutes(frame["distance_miles"].to_numpy(), frame["hour"].to_numpy())
    location = (frame["source"].fillna("") + " " + frame["destination"].fillna("")).str.lower()
    frame["airport"] = location.str.contains("airport|logan")
    premium = frame["name"].fillna("").str.lower().str.contains("black|lux|premium|select|suv|uberxl|lyft xl")
    frame["service"] = np.where(premium, "premium", "economy")
    summary = frame["short_summary"].fillna("").str.lower()
    frame["weather"] = np.where(summary.str.contains("snow|sleet"), "snow", np.where(summary.str.contains("rain|drizzle"), "rain", "clear"))
    frame["day_group"] = frame["pickup_datetime"].dt.date.astype(str)
    fit = frame[frame["pickup_datetime"].dt.date <= date(2018, 12, 9)].copy()
    test = frame[frame["pickup_datetime"].dt.date >= date(2018, 12, 10)].copy()
    bank_ids = set(pd.read_csv(DATA / "processed/boston/trip_bank.csv", usecols=["source_id"])["source_id"].astype(str))
    in_bank = test["id"].astype(str).isin(bank_ids)
    excluded = int(in_bank.sum())
    test = test[~in_bank].copy()
    return fit, test, excluded


def _fit_only_boston_tariff(fit: pd.DataFrame) -> dict[str, Any]:
    arrays = {
        "price": fit["actual_price"].to_numpy(float),
        "distance": fit["distance_miles"].to_numpy(float),
        "duration": fit["duration_minutes"].to_numpy(float),
        "hour": fit["hour"].to_numpy(int),
        "dow": fit["day_of_week"].to_numpy(int),
        "airport": fit["airport"].to_numpy(bool),
        "service": fit["service"].to_numpy(object),
        "shared": np.zeros(len(fit), dtype=bool),
        "weather": fit["weather"].to_numpy(object),
    }
    fitted = fit_tariff_anchors(arrays, duration_identified=False, service_identified=True, airport_identified=False)
    tariff = {key: value for key, value in fitted.items() if key in {
        "base_fare", "booking_fee", "per_mile", "per_minute", "airport_fee",
        "surge_hours", "day_multiplier", "weather_multiplier", "service_multiplier",
    }}
    # Apply the v17 scalar procedure: preserve the legacy-card mean on the fit part.
    legacy = MarketInteraction("Boston", seed=17)
    legacy.register_calibrated_tariff(dict(tariff))
    legacy.configure_fare_model("legacy")
    sample = fit if len(fit) <= 100000 else fit.sample(100000, random_state=SEED)
    target = np.mean([
        legacy.quote_price(float(row.distance_miles), float(row.duration_minutes), RideContext(
            day_of_week=int(row.day_of_week), weather=str(row.weather), hour=int(row.hour),
            airport=bool(row.airport), service=str(row.service),
        )) for row in sample.itertuples(index=False)
    ])
    low, high = 0.1, 5.0
    for _ in range(48):
        scale = 0.5 * (low + high)
        candidate = json.loads(json.dumps(tariff))
        candidate["per_mile"] *= scale
        candidate["per_minute"] *= scale
        achieved = quote_rows("boston", sample, candidate).mean()
        if achieved < target:
            low = scale
        else:
            high = scale
    scale = 0.5 * (low + high)
    tariff["per_mile"] *= scale
    tariff["per_minute"] *= scale
    tariff["v17_fit_only_scale"] = scale
    return tariff


def stage_v1() -> dict[str, Any]:
    append_progress(PROGRESS, job="realdata_V1", state="running", attempt=1, seconds=0)
    declaration = load_declaration()
    nyc = nyc_sample()
    nyc["card_quote"] = quote_rows("nyc", nyc)
    bank = _bank_frame("nyc")
    bank["card_quote"] = quote_rows("nyc", bank)
    baselines = _baseline_predictions(bank, nyc)
    nyc_rows = [_fare_row("Card_out_of_time", nyc["actual_price"].to_numpy(), nyc["card_quote"].to_numpy(), nyc["day_group"])]
    nyc_rows.extend(_fare_row(name, nyc["actual_price"].to_numpy(), prediction, nyc["day_group"]) for name, prediction in baselines.items())
    nyc_rows.append({"name": "Card_in_sample_bank", **fare_metrics(bank["actual_price"], bank["card_quote"])})
    nyc_rows.append({"name": "Card_last_1000_bank", **fare_metrics(bank["actual_price"].iloc[-1000:], bank["card_quote"].iloc[-1000:])})
    nyc_result = {
        "state": "complete_with_price_definition_limitation",
        "price_definition": {
            "chosen": "base_passenger_fare",
            "match_rate": None,
            "reason": "raw NYC Kaggle parquet needed for the August join is absent; Last Revision requires continuation",
        },
        "population_filters": {
            "shared_match_Y_excluded": True,
            "access_a_ride_rule": "not identifiable because Kaggle source is absent",
            "wav_rule": "not identifiable because Kaggle source is absent",
        },
        "sample_seed": SEED,
        "rows_per_month": 30000,
        "bootstrap_draws": int(declaration["bootstrap_draws"]),
        "metrics": nyc_rows,
    }
    atomic_json(RUN / "v1_fare/nyc_metrics.json", nyc_result)
    write_csv(RUN / "v1_fare/nyc_breakdowns.csv", _nyc_breakdowns(nyc))

    fit, test, excluded = _boston_frames()
    fit_tariff = _fit_only_boston_tariff(fit)
    production_quote = quote_rows("boston", test)
    procedure_quote = quote_rows("boston", test, fit_tariff)
    boston_result = {
        "state": "complete",
        "fit_rows": len(fit), "test_rows": len(test), "bank_rows_excluded": excluded,
        "duration": "default_duration_minutes fallback (Boston source has no duration)",
        "procedure_tariff": fit_tariff,
        "metrics": [
            _fare_row("Procedure_out_of_time", test["actual_price"].to_numpy(), procedure_quote, test["day_group"]),
            _fare_row("Simulator_card_rows_outside_bank_same_period", test["actual_price"].to_numpy(), production_quote, test["day_group"]),
        ],
    }
    atomic_json(RUN / "v1_fare/boston_metrics.json", boston_result)
    atomic_json(RUN / "v1_fare/chicago_metrics.json", {
        "state": "skipped_dependency_failed", "dependency": "D1 Chicago source failed after 3 attempts",
        "rounding_limit": "Fares nearest $2.50, tips nearest $1, time nearest 15 minutes",
    })
    result = {"nyc": nyc_result["state"], "boston": "complete", "chicago": "skipped_dependency_failed"}
    atomic_json(RUN / "v1_fare/status.json", result)
    append_progress(PROGRESS, job="realdata_V1", state="complete", attempt=1, seconds=0)
    return result


def _card_clear_sql(tariff: Mapping[str, Any]) -> str:
    hour_cases = " ".join(
        f"WHEN hour >= {int(lo)} AND hour < {int(hi)} THEN {float(mult)}"
        for lo, hi, mult in tariff["surge_hours"]
    )
    weekend = float(tariff.get("day_multiplier", {}).get("5", 1.0))
    return f"""round(
      {float(tariff['base_fare']) + float(tariff['booking_fee'])}
      + ({float(tariff['per_mile'])} * trip_miles + {float(tariff['per_minute'])} * duration_minutes)
        * (CASE {hour_cases} ELSE 1.0 END)
        * (CASE WHEN day_of_week >= 5 THEN {weekend} ELSE 1.0 END)
      + (CASE WHEN airport THEN {float(tariff['airport_fee'])} ELSE 0.0 END), 2
    )"""


def _nyc_base_sql() -> str:
    files = ",".join(repr(str(path)) for path in TLC_FILES)
    return f"""
      SELECT CAST(pickup_datetime AS DATE) AS ride_date,
             EXTRACT(hour FROM pickup_datetime)::INTEGER AS hour,
             ((EXTRACT(dow FROM pickup_datetime)::INTEGER + 6) % 7) AS day_of_week,
             hvfhs_license_num,
             PULocationID IN (1,132,138) OR DOLocationID IN (1,132,138) AS airport,
             trip_miles, trip_time / 60.0 AS duration_minutes,
             base_passenger_fare AS actual_price, driver_pay,
             greatest(0.0, epoch(pickup_datetime - request_datetime) / 60.0) AS wait_minutes,
             pickup_datetime, request_datetime, PULocationID, DOLocationID
      FROM read_parquet([{files}])
      WHERE hvfhs_license_num IN ('HV0003','HV0005')
        AND trip_miles > 0.05 AND trip_miles < 60
        AND trip_time > 30 AND base_passenger_fare > 0
        AND COALESCE(shared_match_flag, 'N') <> 'Y'
    """


def prepare_nyc_real() -> tuple[pd.DataFrame, pd.DataFrame]:
    connection = duckdb.connect()
    base = _nyc_base_sql()
    card = _card_clear_sql(load_tariff("nyc"))
    daily = connection.execute(f"""
      WITH rides AS ({base})
      SELECT ride_date, day_of_week, count(*) AS completed_trips,
             avg(duration_minutes) AS mean_duration,
             median(60.0 * trip_miles / duration_minutes) AS median_speed,
             avg(airport::INTEGER) AS airport_share,
             avg(actual_price / trip_miles) AS fare_per_mile,
             avg(wait_minutes) AS mean_wait,
             median(wait_minutes) AS median_wait,
             avg(driver_pay) AS driver_pay_per_trip,
             avg(driver_pay / trip_miles) AS driver_pay_per_mile,
             avg(driver_pay / duration_minutes) AS driver_pay_per_minute,
             avg(driver_pay / actual_price) AS pay_fare_ratio
      FROM rides GROUP BY ride_date, day_of_week ORDER BY ride_date
    """).fetchdf()
    pi = connection.execute(f"""
      WITH rides AS ({base}), sampled AS (
        SELECT *, ROW_NUMBER() OVER (
          PARTITION BY ride_date ORDER BY hash(pickup_datetime, request_datetime, PULocationID, DOLocationID, trip_miles)
        ) AS sample_rank FROM rides
      ), values AS (
        SELECT *, {card} AS card_clear FROM sampled WHERE sample_rank <= 50000
      )
      SELECT ride_date,
             exp(avg(ln(actual_price / card_clear))) AS price_index,
             exp(avg(ln(actual_price / card_clear)) FILTER (WHERE hvfhs_license_num='HV0003')) AS price_index_uber,
             exp(avg(ln(actual_price / card_clear)) FILTER (WHERE hvfhs_license_num='HV0005')) AS price_index_lyft,
             count(*) AS price_index_n
      FROM values WHERE card_clear > 0 GROUP BY ride_date ORDER BY ride_date
    """).fetchdf()
    daily = daily.merge(pi, on="ride_date", how="left")
    blocks = connection.execute(f"""
      WITH rides AS ({base}), values AS (
        SELECT *, floor(hour / 4)::INTEGER AS block, {card} AS card_clear FROM rides
      )
      SELECT ride_date, day_of_week, block, count(*) AS trips,
             exp(avg(ln(actual_price / card_clear))) AS price_index,
             median(wait_minutes) AS median_wait,
             avg(duration_minutes) AS mean_duration,
             median(60.0 * trip_miles / duration_minutes) AS median_speed,
             avg(airport::INTEGER) AS airport_share,
             avg(actual_price / trip_miles) AS fare_per_mile
      FROM values WHERE card_clear > 0 GROUP BY ride_date, day_of_week, block ORDER BY ride_date, block
    """).fetchdf()
    daily["ride_date"] = pd.to_datetime(daily["ride_date"]).dt.date
    blocks["ride_date"] = pd.to_datetime(blocks["ride_date"]).dt.date
    return daily, blocks


def _event_definitions() -> dict[str, tuple[date, date, date]]:
    return {
        "E1_flood_20230929": (date(2023, 9, 29), date(2023, 9, 29), date(2023, 9, 29)),
        "E2_thanksgiving_2023": (date(2023, 11, 21), date(2023, 11, 26), date(2023, 11, 23)),
        "E3_christmas_2023": (date(2023, 12, 22), date(2024, 1, 3), date(2023, 12, 25)),
        "E4_rain_20231218": (date(2023, 12, 18), date(2023, 12, 18), date(2023, 12, 18)),
        "E4_snow_20240213": (date(2024, 2, 13), date(2024, 2, 13), date(2024, 2, 13)),
    }


def _event_exclusion_dates() -> set[date]:
    excluded: set[date] = set()
    for start, end, _ in _event_definitions().values():
        cursor = start - timedelta(days=1)
        while cursor <= end + timedelta(days=1):
            excluded.add(cursor)
            cursor += timedelta(days=1)
    return excluded


def _baseline_days(target: date, weather: Mapping[date, Mapping[str, Any]]) -> list[date]:
    excluded = _event_exclusion_dates()
    result: list[date] = []
    for weeks in (4, 6):
        result = []
        for direction in (-1, 1):
            for offset in range(1, weeks + 1):
                candidate = target + timedelta(days=direction * 7 * offset)
                wx = weather.get(candidate, {})
                if candidate in excluded or candidate in FEDERAL_HOLIDAYS:
                    continue
                if float(wx.get("prcp_raw", 0.0)) >= 100 or float(wx.get("snow_raw", 0.0)) > 0:
                    continue
                result.append(candidate)
        if len(result) >= 4:
            break
    return sorted(set(result))


def _real_event_effects(daily: pd.DataFrame) -> list[dict[str, Any]]:
    weather = ghcn_daily("nyc")
    indexed = daily.set_index("ride_date")
    metrics = {
        "m1_completed_trips": "completed_trips", "m2_mean_duration": "mean_duration",
        "m3_median_speed": "median_speed", "m4_airport_share": "airport_share",
        "m5_price_index": "price_index", "m6_fare_per_mile": "fare_per_mile", "m7_wait": "mean_wait",
    }
    rows: list[dict[str, Any]] = []
    for event, (start, end, anchor) in _event_definitions().items():
        cursor = start
        event_rows = []
        while cursor <= end:
            if cursor in indexed.index:
                event_rows.append(cursor)
                baselines = [day for day in _baseline_days(cursor, weather) if day in indexed.index]
                for metric, column in metrics.items():
                    base = indexed.loc[baselines, column].astype(float).to_numpy()
                    observed = float(indexed.loc[cursor, column])
                    mean = float(np.mean(base))
                    spread = float(np.std(base, ddof=1) / mean) if len(base) > 1 and mean else 0.0
                    effect = observed / mean - 1.0 if mean else 0.0
                    rows.append({
                        "city": "nyc", "event": event, "date": cursor.isoformat(),
                        "offset": (cursor - anchor).days, "metric": metric,
                        "real_effect": effect, "normal_day_band": 1.96 * spread,
                        "z": effect / spread if spread else 0.0,
                        "baseline_days": "|".join(day.isoformat() for day in baselines),
                    })
            cursor += timedelta(days=1)
        if len(event_rows) > 1:
            for metric, column in metrics.items():
                effects = [row["real_effect"] for row in rows if row["event"] == event and row["metric"] == metric and row["date"] != "wave_average"]
                bands = [row["normal_day_band"] for row in rows if row["event"] == event and row["metric"] == metric and row["date"] != "wave_average"]
                rows.append({"city": "nyc", "event": event, "date": "wave_average", "offset": 0,
                             "metric": metric, "real_effect": float(np.mean(effects)),
                             "normal_day_band": float(np.mean(bands)), "z": 0.0, "baseline_days": "per-day"})
    return rows


def _normal_days(daily: pd.DataFrame) -> pd.DataFrame:
    weather = ghcn_daily("nyc")
    excluded = _event_exclusion_dates() | FEDERAL_HOLIDAYS
    keep = []
    for day in daily["ride_date"]:
        wx = weather.get(day, {})
        keep.append(day not in excluded and float(wx.get("prcp_raw", 0.0)) < 100 and float(wx.get("snow_raw", 0.0)) <= 0)
    return daily[np.asarray(keep, dtype=bool)].copy()


def stage_real() -> dict[str, Any]:
    append_progress(PROGRESS, job="realdata_V2_V4_real", state="running", attempt=1, seconds=0)
    daily, blocks = prepare_nyc_real()
    write_csv(RUN / "prepared/nyc_daily.csv", [{key: (value.isoformat() if isinstance(value, date) else value) for key, value in row.items()} for row in daily.to_dict("records")])
    write_csv(RUN / "prepared/nyc_blocks.csv", [{key: (value.isoformat() if isinstance(value, date) else value) for key, value in row.items()} for row in blocks.to_dict("records")])

    event_rows = _real_event_effects(daily)
    write_csv(RUN / "v3_events/nyc_real.csv", event_rows)
    atomic_json(RUN / "v3_events/chicago_real_status.json", {
        "state": "skipped_dependency_failed", "dependency": "D1 Chicago source failed after 3 attempts",
    })

    normal = _normal_days(daily)
    v4 = {
        "state": "complete",
        "normal_day": {
            platform: variability(zip(normal["ride_date"].astype(str), normal["day_of_week"], normal[column]))
            for platform, column in (("pooled", "price_index"), ("uber", "price_index_uber"), ("lyft", "price_index_lyft"))
        },
        "hour_profile": {
            str(block): float(value)
            for block, value in blocks[blocks["ride_date"].isin(set(normal["ride_date"]))].groupby("block")["price_index"].mean().items()
        },
        "event_uplift": [row for row in event_rows if row["metric"] == "m5_price_index"],
        "known_limits": ["agent cannot respond within a day", "agent cannot price by trip"],
    }
    atomic_json(RUN / "v4_surge/nyc_real.json", v4)
    atomic_json(RUN / "v4_surge/chicago_status.json", {
        "state": "skipped_dependency_failed", "dependency": "D1 Chicago source failed after 3 attempts",
    })
    result = {"daily_rows": len(daily), "block_rows": len(blocks), "event_rows": len(event_rows)}
    atomic_json(RUN / "prepared/status.json", result)
    append_progress(PROGRESS, job="realdata_V2_V4_real", state="complete", attempt=1, seconds=0)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("guards", "v1", "real", "stage1"))
    args = parser.parse_args()
    if args.stage in ("guards", "stage1"):
        print(json.dumps(stage_guards(), indent=2, default=str))
    if args.stage in ("v1", "stage1"):
        print(json.dumps(stage_v1(), indent=2, default=str))
    if args.stage in ("real", "stage1"):
        print(json.dumps(stage_real(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
