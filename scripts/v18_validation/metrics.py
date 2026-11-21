"""Numerical helpers for the V18 real-data validation products."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


def _finite_pair(a: Sequence[float], b: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    return x[keep], y[keep]


def rankdata(values: Sequence[float]) -> np.ndarray:
    """Return average ranks, including ties, without a SciPy dependency."""
    x = np.asarray(values, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranked = np.empty(x.size, dtype=float)
    start = 0
    while start < x.size:
        stop = start + 1
        while stop < x.size and x[order[stop]] == x[order[start]]:
            stop += 1
        ranked[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranked


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    if float(np.ptp(x)) <= 1e-12 or float(np.ptp(y)) <= 1e-12:
        return float("nan")
    xc, yc = x - x.mean(), y - y.mean()
    denom = float(np.sqrt(np.dot(xc, xc) * np.dot(yc, yc)))
    return float(np.dot(xc, yc) / denom) if denom > 0 else float("nan")


def fare_metrics(actual: Sequence[float], quote: Sequence[float]) -> dict[str, float | int]:
    actual_a, quote_a = _finite_pair(actual, quote)
    if actual_a.size < 2:
        raise ValueError("fare metrics require at least two finite pairs")
    error = quote_a - actual_a
    sst = float(np.dot(actual_a - actual_a.mean(), actual_a - actual_a.mean()))
    q_center = quote_a - quote_a.mean()
    slope_denom = float(np.dot(q_center, q_center))
    return {
        "n": int(actual_a.size),
        "r": _corr(actual_a, quote_a),
        "rho": _corr(rankdata(actual_a), rankdata(quote_a)),
        "r2": float(1.0 - np.dot(error, error) / sst) if sst > 0 else float("nan"),
        "mae": float(np.mean(np.abs(error))),
        "median_ape": float(np.median(np.abs(error) / np.maximum(np.abs(actual_a), 1e-12))),
        "bias": float(np.mean(error)),
        "slope": float(np.dot(q_center, actual_a - actual_a.mean()) / slope_denom)
        if float(np.ptp(quote_a)) > 1e-12 and slope_denom > 0 else float("nan"),
    }


def day_block_bootstrap(
    actual: Sequence[float],
    quote: Sequence[float],
    groups: Sequence[str],
    *,
    draws: int,
    seed: int,
) -> dict[str, list[float | None]]:
    """Day-block intervals using sufficient statistics.

    Pearson r, R2, MAE, bias and slope are exact for each day resample.  Rho
    uses ranks from the original sample (the standard fixed-rank block
    bootstrap approximation), and median APE resamples day-level medians.
    """
    a, q = _finite_pair(actual, quote)
    g = np.asarray(groups, dtype=object)
    keep = np.isfinite(np.asarray(actual, dtype=float)) & np.isfinite(np.asarray(quote, dtype=float))
    g = g[keep]
    if a.size != g.size or not a.size:
        raise ValueError("bootstrap inputs do not align")
    ra, rq = rankdata(a), rankdata(q)
    by_group: dict[str, list[int]] = defaultdict(list)
    for index, key in enumerate(g):
        by_group[str(key)].append(index)
    keys = sorted(by_group)
    if len(keys) < 2:
        raise ValueError("day-block bootstrap needs at least two days")

    stats: dict[str, np.ndarray] = {}
    ape_medians = []
    for key in keys:
        idx = np.asarray(by_group[key], dtype=int)
        av, qv, rav, rqv = a[idx], q[idx], ra[idx], rq[idx]
        err = qv - av
        stats[key] = np.array([
            idx.size,
            av.sum(), qv.sum(), np.dot(av, av), np.dot(qv, qv), np.dot(av, qv),
            np.abs(err).sum(), err.sum(), np.dot(err, err),
            rav.sum(), rqv.sum(), np.dot(rav, rav), np.dot(rqv, rqv), np.dot(rav, rqv),
        ], dtype=float)
        ape_medians.append(float(np.median(np.abs(err) / np.maximum(np.abs(av), 1e-12))))
    matrix = np.vstack([stats[key] for key in keys])
    ape_medians_a = np.asarray(ape_medians, dtype=float)
    rng = np.random.default_rng(seed)
    values = {name: [] for name in ("r", "rho", "r2", "mae", "median_ape", "bias", "slope")}
    for _ in range(draws):
        chosen = rng.integers(0, len(keys), len(keys))
        s = matrix[chosen].sum(axis=0)
        n, sy, sx, sy2, sx2, syx, sae, se, sse, sry, srx, sry2, srx2, sryx = s
        cov = syx - sy * sx / n
        vary = sy2 - sy * sy / n
        varx = sx2 - sx * sx / n
        rcov = sryx - sry * srx / n
        rvary = sry2 - sry * sry / n
        rvarx = srx2 - srx * srx / n
        values["r"].append(cov / np.sqrt(vary * varx) if vary > 0 and varx > 0 else np.nan)
        values["rho"].append(rcov / np.sqrt(rvary * rvarx) if rvary > 0 and rvarx > 0 else np.nan)
        values["r2"].append(1.0 - sse / vary if vary > 0 else np.nan)
        values["mae"].append(sae / n)
        values["median_ape"].append(float(np.median(ape_medians_a[chosen])))
        values["bias"].append(se / n)
        values["slope"].append(cov / varx if varx > 0 else np.nan)
    intervals: dict[str, list[float | None]] = {}
    for name, sample in values.items():
        finite = np.asarray(sample, dtype=float)
        finite = finite[np.isfinite(finite)]
        intervals[name] = (
            [float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))]
            if finite.size else [None, None]
        )
    return intervals


def wasserstein1(a: Sequence[float], b: Sequence[float]) -> float:
    x = np.sort(np.asarray(a, dtype=float))
    y = np.sort(np.asarray(b, dtype=float))
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if not x.size or not y.size:
        raise ValueError("Wasserstein inputs cannot be empty")
    # Exact 1-D empirical Wasserstein distance: integrate |F_x - F_y|
    # between consecutive pooled observations.  This is O((n+m) log(n+m))
    # and avoids the superlinear millions-of-quantiles implementation.
    pooled = np.sort(np.concatenate((x, y)))
    if pooled.size < 2:
        return 0.0
    left = pooled[:-1]
    widths = np.diff(pooled)
    cdf_x = np.searchsorted(x, left, side="right") / x.size
    cdf_y = np.searchsorted(y, left, side="right") / y.size
    return float(np.dot(np.abs(cdf_x - cdf_y), widths))


def ks(a: Sequence[float], b: Sequence[float]) -> float:
    x = np.sort(np.asarray(a, dtype=float))
    y = np.sort(np.asarray(b, dtype=float))
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if not x.size or not y.size:
        raise ValueError("KS inputs cannot be empty")
    points = np.sort(np.unique(np.concatenate((x, y))))
    return float(np.max(np.abs(
        np.searchsorted(x, points, side="right") / x.size
        - np.searchsorted(y, points, side="right") / y.size
    )))


def fidelity(real: Sequence[float], synthetic: Sequence[float]) -> dict[str, Any]:
    real_a = np.asarray(real, dtype=float)
    syn_a = np.asarray(synthetic, dtype=float)
    real_a, syn_a = real_a[np.isfinite(real_a)], syn_a[np.isfinite(syn_a)]
    if not real_a.size or not syn_a.size:
        raise ValueError("fidelity inputs must be finite and nonempty")
    rq = np.quantile(real_a, [0.25, 0.75])
    sq = np.quantile(syn_a, [0.25, 0.75])
    return {
        "median_ratio": float(np.median(syn_a) / max(abs(np.median(real_a)), 1e-12)),
        "iqr_ratio": float((sq[1] - sq[0]) / max(rq[1] - rq[0], 1e-12)),
        "wasserstein1": wasserstein1(real_a, syn_a),
        "ks": ks(real_a, syn_a),
        "n_real": int(real_a.size),
        "n_synthetic": int(syn_a.size),
    }


def distribution_summary(values: Sequence[float]) -> dict[str, Any]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not x.size:
        raise ValueError("distribution is empty")
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "quantiles": {
            str(q): float(np.quantile(x, q))
            for q in (0.10, 0.25, 0.50, 0.75, 0.90, 0.99)
        },
    }


def total_variation(a: Mapping[Any, float], b: Mapping[Any, float]) -> float:
    keys = set(a) | set(b)
    return float(0.5 * sum(abs(float(a.get(k, 0.0)) - float(b.get(k, 0.0))) for k in keys))


def ols_hc1(design: Sequence[Sequence[float]], outcome: Sequence[float]) -> dict[str, Any]:
    x = np.asarray(design, dtype=float)
    y = np.asarray(outcome, dtype=float)
    keep = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    x, y = x[keep], y[keep]
    if x.ndim != 2 or y.shape != (x.shape[0],) or x.shape[0] <= x.shape[1]:
        raise ValueError("OLS design is invalid")
    inverse = np.linalg.pinv(x.T @ x)
    beta = inverse @ x.T @ y
    residual = y - x @ beta
    meat = x.T @ (x * (residual ** 2).reshape(-1, 1))
    covariance = (x.shape[0] / (x.shape[0] - x.shape[1])) * inverse @ meat @ inverse
    se = np.sqrt(np.maximum(0.0, np.diag(covariance)))
    return {
        "coefficients": beta.tolist(),
        "standard_errors_hc1": se.tolist(),
        "ci95": np.column_stack((beta - 1.96 * se, beta + 1.96 * se)).tolist(),
        "n": int(x.shape[0]),
    }


def variability(values: Iterable[tuple[str, int, float]]) -> dict[str, float | int]:
    rows = sorted((str(day), int(weekday), float(value)) for day, weekday, value in values if np.isfinite(value))
    if len(rows) < 3:
        raise ValueError("variability needs at least three days")
    raw = np.log(np.asarray([row[2] for row in rows], dtype=float))
    weekday_means = {day: float(np.mean([raw[i] for i, row in enumerate(rows) if row[1] == day])) for day in range(7)}
    adjusted = np.asarray([raw[i] - weekday_means[row[1]] for i, row in enumerate(rows)])

    def one(x: np.ndarray) -> dict[str, float]:
        return {
            "sd": float(np.std(x, ddof=1)),
            "mean_abs_change": float(np.mean(np.abs(np.diff(x)))),
            "lag1": _corr(x[:-1], x[1:]),
            "p05": float(np.quantile(x, 0.05)),
            "p95": float(np.quantile(x, 0.95)),
        }

    return {"n": len(rows), "raw": one(raw), "weekday_adjusted": one(adjusted)}


def finite_tree(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value))
    return True


def sanitize_tree(value: Any) -> Any:
    """Represent undefined numerical statistics as JSON null, never NaN."""
    if isinstance(value, Mapping):
        return {key: sanitize_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_tree(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    return value
