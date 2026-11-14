#!/usr/bin/env python3
"""Authoritative Last Revision market variants for every V18 runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence


BASE = Path(__file__).resolve().parents[1]
CITIES = ("nyc", "boston", "chicago")
FLEET_SCALE = {"nyc": 1.2, "chicago": 1.1, "boston": 1.0}
SENSITIVITY_REFERENCE_EARNINGS = 30.487332296170116


def _drop_value_options(values: Sequence[str], flags: set[str]) -> list[str]:
    result: list[str] = []
    index = 0
    source = list(map(str, values))
    while index < len(source):
        value = source[index]
        if value in flags:
            index += 2
            continue
        result.append(value)
        index += 1
    return result


def _set(values: Sequence[str], flag: str, value: Any) -> list[str]:
    result = _drop_value_options(values, {flag})
    result.extend((flag, str(value)))
    return result


def _boolean(values: Sequence[str], flag: str, enabled: bool) -> list[str]:
    negative = "--no-" + flag.removeprefix("--")
    result = [item for item in map(str, values) if item not in {flag, negative}]
    result.append(flag if enabled else negative)
    return result


def _demand_artifact(city: str) -> Path:
    return (
        BASE
        / "artifacts/v17_relative_demand_post_fare_20261028"
        / city
        / "demand_1.15.json"
    ).resolve()


def _profile(path: Path) -> list[float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("hour_profile", payload)
    return [float(values[str(hour)] if str(hour) in values else values[hour]) for hour in range(24)]


def market_flags(
    command: Sequence[str], *, variant: str, city: str, customers: int,
) -> tuple[list[str], dict[str, Any]]:
    """Apply one registered market variant and return expected Core switches.

    ``scenario_hazards`` is intentionally absent: the caller owns that switch.
    S1--S3 are New York-only sensitivity arms.  Production is the P2
    ``v17_model`` market and is valid in all three cities.
    """
    variant = str(variant).strip().lower()
    city = str(city).strip().lower()
    if city not in CITIES:
        raise ValueError(f"unknown V18 city: {city}")
    if variant not in {"production", "s1", "s2", "s3"}:
        raise ValueError(f"unknown V18 market variant: {variant}")
    if variant != "production" and city != "nyc":
        raise ValueError(f"V18 market variant {variant} is New York-only")
    if int(customers) <= 0:
        raise ValueError("customers must be positive")

    value_options = {
        "--supply_eta_v3", "--supply_eta_within", "--supply_eta_day",
        "--stock_rule", "--stock_adjustment_rate", "--stock_floor",
        "--driver_pay_rule", "--pay_rule_json", "--fleet_scale",
        "--base_pickup_minutes", "--supply_base_active_drivers",
        "--supply_hour_profile_json", "--supply_reference_earnings",
        "--active_driver_feature_reference", "--pool", "--train_customers",
        "--eval_customers", "--loyalty_pool_size",
        "--demand_calibration_artifact",
    }
    values = _drop_value_options(command, value_options)
    values = _boolean(values, "--v3_daily_stock", variant == "s3")
    common = (
        ("--supply_eta_v3", 0.6),
        ("--stock_rule", "integral" if variant == "s3" else "power"),
        ("--stock_adjustment_rate", 0.05 if variant == "s3" else 0.30),
        ("--stock_floor", 0.4),
        ("--active_driver_feature_reference", "run_capacity"),
        ("--pool", 20_000),
        ("--train_customers", int(customers)),
        ("--eval_customers", int(customers)),
        ("--loyalty_pool_size", round(3.6 * int(customers))),
        ("--demand_calibration_artifact", _demand_artifact(city)),
    )
    for flag, value in common:
        values = _set(values, flag, value)

    linear = variant in {"s1", "s2", "s3"}
    fleet_scale = FLEET_SCALE[city] if variant == "production" else (1.2 if variant == "s1" else 0.7)
    pickup = 3.5 if variant in {"production", "s1"} else 1.5
    base_active = 0 if variant in {"production", "s1"} else 35
    for flag, value in (
        ("--driver_pay_rule", "linear" if linear else "share"),
        ("--fleet_scale", fleet_scale),
        ("--base_pickup_minutes", pickup),
        ("--supply_base_active_drivers", base_active),
    ):
        values = _set(values, flag, value)
    if variant == "s3":
        values = _set(values, "--supply_eta_within", 0.6)
    if linear:
        values = _set(
            values, "--pay_rule_json",
            (BASE / "calibration_artifacts/v18/nyc/pay_rule.json").resolve(),
        )
    supply_json = (BASE / "calibration_artifacts/v18/nyc/supply.json").resolve()
    if variant in {"s2", "s3"}:
        values = _set(values, "--supply_hour_profile_json", supply_json)
        values = _set(
            values, "--supply_reference_earnings",
            SENSITIVITY_REFERENCE_EARNINGS,
        )

    expected: dict[str, Any] = {
        "supply_eta_v3": 0.6,
        "v3_daily_stock": variant == "s3",
        "stock_rule": "integral" if variant == "s3" else "power",
        "supply_eta_within": 0.6 if variant == "s3" else None,
        "supply_eta_day": None if variant == "s3" else 0.4,
        "stock_adjustment_rate": 0.05 if variant == "s3" else 0.30,
        "stock_floor": 0.4,
        "stock_ceiling": 2.0,
        "active_driver_feature_reference": "run_capacity",
        "driver_pay_rule": "linear" if linear else "share",
        "base_pickup_minutes": float(pickup),
        "supply_base_active_drivers": int(base_active),
    }
    if variant in {"s2", "s3"}:
        expected["supply_reference_earnings"] = SENSITIVITY_REFERENCE_EARNINGS
        expected["hour_profile"] = _profile(supply_json)
    return values, expected


__all__ = ["CITIES", "FLEET_SCALE", "SENSITIVITY_REFERENCE_EARNINGS", "market_flags"]
