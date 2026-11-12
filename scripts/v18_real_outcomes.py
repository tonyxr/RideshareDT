#!/usr/bin/env python3
"""V18 real-data outcome checks V5-V7 with explicit leakage guards."""

from __future__ import annotations

import argparse
import csv
from datetime import date
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def require_declaration(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != "v18-realdata-declaration-v1":
        raise ValueError("real-data metrics require the committed V18 declaration")
    if value.get("declared_before_metrics") is not True:
        raise ValueError("declaration does not predate real-data metrics")
    return value


def leakage_guard(rows: Sequence[Mapping[str, Any]], *, date_field: str, earliest_allowed: date) -> None:
    observed = [date.fromisoformat(str(row[date_field])[:10]) for row in rows]
    if not observed or min(observed) < earliest_allowed:
        raise ValueError(
            f"out-of-time leakage: minimum {min(observed) if observed else None} < {earliest_allowed}"
        )


def wasserstein1(a: Sequence[float], b: Sequence[float]) -> float:
    a, b = np.sort(np.asarray(a, dtype=float)), np.sort(np.asarray(b, dtype=float))
    if not a.size or not b.size:
        raise ValueError("Wasserstein inputs cannot be empty")
    quantiles = np.linspace(0.0, 1.0, max(a.size, b.size))
    return float(np.mean(np.abs(
        np.quantile(a, quantiles) - np.quantile(b, quantiles)
    )))


def ks(a: Sequence[float], b: Sequence[float]) -> float:
    a, b = np.sort(np.asarray(a, dtype=float)), np.sort(np.asarray(b, dtype=float))
    values = np.sort(np.unique(np.concatenate((a, b))))
    return float(np.max(np.abs(
        np.searchsorted(a, values, side="right") / a.size
        - np.searchsorted(b, values, side="right") / b.size
    )))


def fidelity(real: Sequence[float], synthetic: Sequence[float]) -> dict[str, Any]:
    real, synthetic = np.asarray(real, dtype=float), np.asarray(synthetic, dtype=float)
    real, synthetic = real[np.isfinite(real)], synthetic[np.isfinite(synthetic)]
    if not real.size or not synthetic.size:
        raise ValueError("fidelity inputs must contain finite observations")
    real_iqr = np.quantile(real, [0.25, 0.75])
    synthetic_iqr = np.quantile(synthetic, [0.25, 0.75])
    return {
        "median_ratio": float(np.median(synthetic) / max(abs(np.median(real)), 1e-12)),
        "iqr_ratio": float((synthetic_iqr[1] - synthetic_iqr[0]) / max(real_iqr[1] - real_iqr[0], 1e-12)),
        "wasserstein1": wasserstein1(real, synthetic),
        "ks": ks(real, synthetic),
        "n_real": int(real.size), "n_synthetic": int(synthetic.size),
    }


def ols_hc1(design: Sequence[Sequence[float]], outcome: Sequence[float]) -> dict[str, Any]:
    x, y = np.asarray(design, dtype=float), np.asarray(outcome, dtype=float)
    if x.ndim != 2 or y.shape != (x.shape[0],) or x.shape[0] <= x.shape[1]:
        raise ValueError("OLS design is invalid")
    inverse = np.linalg.pinv(x.T @ x)
    beta = inverse @ x.T @ y
    residual = y - x @ beta
    meat = x.T @ (x * (residual ** 2).reshape(-1, 1))
    scale = x.shape[0] / (x.shape[0] - x.shape[1])
    covariance = scale * inverse @ meat @ inverse
    standard_error = np.sqrt(np.maximum(0.0, np.diag(covariance)))
    return {"coefficients": beta.tolist(), "standard_errors_hc1": standard_error.tolist(),
            "ci95": np.column_stack((beta - 1.96 * standard_error, beta + 1.96 * standard_error)).tolist(),
            "n": int(x.shape[0])}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", choices=("V5", "V6", "V7"), required=True)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--synthetic", type=Path, required=True)
    parser.add_argument("--field", default="wait_minutes")
    parser.add_argument("--date-field", default="date")
    parser.add_argument("--earliest", default="2023-09-01")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    declaration = require_declaration(args.declaration)
    real, synthetic = read_csv(args.real), read_csv(args.synthetic)
    leakage_guard(real, date_field=args.date_field, earliest_allowed=date.fromisoformat(args.earliest))
    if args.check == "V5":
        result = fidelity(
            [float(row[args.field]) for row in real],
            [float(row[args.field]) for row in synthetic],
        )
    elif args.check == "V6":
        regressors = tuple(declaration["v6_regressors"])
        result = {
            "real": ols_hc1(
                [[1.0, *(float(row[key]) for key in regressors)] for row in real],
                [float(row[args.field]) for row in real],
            ),
            "synthetic": ols_hc1(
                [[1.0, *(float(row[key]) for key in regressors)] for row in synthetic],
                [float(row[args.field]) for row in synthetic],
            ),
            "regressors": ["intercept", *regressors],
        }
    else:
        real_by_month = {row["month"]: float(row[args.field]) for row in real}
        synthetic_by_month = {row["month"]: float(row[args.field]) for row in synthetic}
        months = sorted(set(real_by_month) & set(synthetic_by_month))
        if not months:
            raise ValueError("V7 has no aligned months")
        result = {"monthly_ratio": {
            month: synthetic_by_month[month] / real_by_month[month]
            for month in months
        }}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"check": args.check, **result}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
