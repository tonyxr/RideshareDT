#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 12:54:03 2026

@author: Xiaoru Shi
"""

from __future__ import annotations

import csv
from typing import Dict, Any, List, Optional

import mkl_config  # noqa: F401 - set oneMKL env before NumPy/Torch
import numpy as np

from calibration_presets import NYC_PUBLIC_2024


INCOME_LEVELS = ["<50k", "50k-100k", "100k-200k", "200k+"]

PRESET_PAYLOADS: Dict[str, Dict[str, Any]] = {
    "nyc_public": NYC_PUBLIC_2024,
}


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        if val is None or val == "":
            return float(default)
        return float(val)
    except (TypeError, ValueError):
        return float(default)


def _norm_probs(counts: Dict[str, float], keys: List[str]) -> Dict[str, float]:
    vals = np.array([max(0.0, float(counts.get(k, 0.0))) for k in keys], dtype=float)
    if float(vals.sum()) <= 0:
        vals = np.ones(len(keys), dtype=float)
    vals = vals / vals.sum()
    return {k: float(v) for k, v in zip(keys, vals)}


def load_rows(path: str, city: Optional[str] = None) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if city and str(row.get("City", "")).strip() != city:
                continue
            rows.append(row)
    return rows


def derive_calibration(path: str, city: Optional[str] = None) -> Dict[str, Any]:
    rows = load_rows(path, city=city)
    if not rows:
        raise ValueError(f"No calibration rows found in {path} (city={city!r}).")

    weather_counts: Dict[str, float] = {}
    service_counts: Dict[str, float] = {}
    income_counts: Dict[str, float] = {k: 0.0 for k in INCOME_LEVELS}

    airport_vals: List[float] = []
    ages: List[float] = []
    household_vals: List[float] = []
    p_new_vals: List[float] = []

    price_gap: List[float] = []
    choose_f1: List[float] = []
    loyalty_strength: List[float] = []

    for r in rows:
        weather = str(r.get("Weather", "clear") or "clear").lower()
        service = str(r.get("Service", "economy") or "economy").lower()

        weather_counts[weather] = weather_counts.get(weather, 0.0) + 1.0
        service_counts[service] = service_counts.get(service, 0.0) + 1.0

        airport_raw = str(r.get("Airport", "False") or "False").strip().lower()
        airport_vals.append(1.0 if airport_raw in {"1", "true", "yes", "y"} else 0.0)

        income = str(r.get("IncomeBracket", "50k-100k") or "50k-100k")
        if income in income_counts:
            income_counts[income] += 1.0

        age = _safe_float(r.get("Age"), default=np.nan)
        if not np.isnan(age):
            ages.append(age)

        hh = _safe_float(r.get("HouseholdSize"), default=np.nan)
        if not np.isnan(hh):
            household_vals.append(hh)

        loyalty_type = str(r.get("Loyalty", r.get("LoyaltyType", "New")) or "New").strip().lower()
        p_new_vals.append(1.0 if loyalty_type == "new" else 0.0)

        p1 = _safe_float(r.get("Price_Firm1"), default=np.nan)
        p2 = _safe_float(r.get("Price_Firm2"), default=np.nan)
        ch = str(r.get("Choice", "") or "")
        if not np.isnan(p1) and not np.isnan(p2) and ch in {"Firm1", "Firm2"}:
            price_gap.append(float(p2 - p1))
            choose_f1.append(1.0 if ch == "Firm1" else 0.0)
            loyalty_strength.append(_safe_float(r.get("LoyaltyStrength"), 0.0))

    x = np.array(price_gap, dtype=float)
    y = np.array(choose_f1, dtype=float)
    l = np.array(loyalty_strength, dtype=float) if loyalty_strength else np.array([0.0])

    if len(x) >= 10 and float(np.var(x)) > 1e-9:
        slope = float(np.cov(x, y, bias=True)[0, 1] / np.var(x))
    else:
        slope = 0.15

    slope = float(np.clip(slope, 0.02, 1.20))

    loyalty_corr = 0.0
    if len(l) == len(y) and len(y) >= 10 and float(np.var(l)) > 1e-9:
        loyalty_corr = float(np.cov(l, y, bias=True)[0, 1] / np.var(l))
    loyalty_corr = float(np.clip(abs(loyalty_corr), 0.10, 1.50))

    return {
        "sample_size": len(rows),
        "market": {
            "weather_probs": _norm_probs(weather_counts, keys=sorted(weather_counts.keys() or ["clear", "rain", "snow"])),
            "service_probs": _norm_probs(service_counts, keys=sorted(service_counts.keys() or ["economy", "premium"])),
            "airport_prob": float(np.clip(np.mean(airport_vals) if airport_vals else 0.12, 0.01, 0.60)),
        },
        "agent": {
            "age_mean": float(np.mean(ages) if ages else 36.0),
            "age_std": float(max(5.0, np.std(ages) if ages else 12.0)),
            "income_probs": _norm_probs(income_counts, keys=INCOME_LEVELS),
            "household_lambda": float(max(1.1, np.mean(household_vals) if household_vals else 2.0)),
            "p_new": float(np.clip(np.mean(p_new_vals) if p_new_vals else 0.30, 0.05, 0.95)),
        },
        "choice": {
            "price_sensitivity_scale": float(np.clip(slope / 0.35, 0.30, 2.80)),
            "loyalty_scale": float(np.clip(loyalty_corr / 0.85, 0.30, 2.80)),
        },
    }


def load_calibration_preset(name: str) -> Dict[str, Any]:
    key = str(name or "").strip().lower()
    if key not in PRESET_PAYLOADS:
        raise ValueError(f"Unknown calibration preset: {name}. Available presets: {sorted(PRESET_PAYLOADS.keys())}")

    payload = PRESET_PAYLOADS[key]
    if not isinstance(payload, dict) or "calibration" not in payload:
        raise ValueError(f"Calibration preset {name} must contain top-level 'calibration' object.")

    calibration = payload["calibration"]
    if not isinstance(calibration, dict):
        raise ValueError(f"Calibration preset {name} has invalid 'calibration' object.")
    return calibration
