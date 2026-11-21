from __future__ import annotations

import math

import numpy as np

from scripts.v18_validation.metrics import (
    day_block_bootstrap,
    fare_metrics,
    fidelity,
    finite_tree,
    ols_hc1,
    rankdata,
    sanitize_tree,
    total_variation,
)


def test_rankdata_averages_ties() -> None:
    assert rankdata([3.0, 1.0, 1.0, 2.0]).tolist() == [4.0, 1.5, 1.5, 3.0]


def test_perfect_fare_metrics() -> None:
    result = fare_metrics([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])
    assert result["r"] == 1.0
    assert result["rho"] == 1.0
    assert result["r2"] == 1.0
    assert result["mae"] == 0.0
    assert result["bias"] == 0.0
    assert result["slope"] == 1.0


def test_day_block_bootstrap_is_deterministic_and_finite() -> None:
    actual = np.arange(1.0, 13.0)
    quote = actual * 1.1 + np.tile([0.1, -0.1], 6)
    groups = [f"2024-01-{day:02d}" for day in np.repeat(np.arange(1, 7), 2)]
    first = day_block_bootstrap(actual, quote, groups, draws=40, seed=17)
    second = day_block_bootstrap(actual, quote, groups, draws=40, seed=17)
    assert first == second
    assert finite_tree(first)


def test_distribution_metrics_and_total_variation() -> None:
    result = fidelity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert result["median_ratio"] == 1.0
    assert result["iqr_ratio"] == 1.0
    assert result["wasserstein1"] == 0.0
    assert result["ks"] == 0.0
    assert total_variation({0: 0.5, 1: 0.5}, {0: 0.25, 1: 0.75}) == 0.25


def test_ols_hc1_recovers_linear_coefficients() -> None:
    x = np.arange(1.0, 11.0)
    result = ols_hc1(np.column_stack((np.ones_like(x), x)), 2.0 + 3.0 * x)
    assert math.isclose(result["coefficients"][0], 2.0, abs_tol=1e-10)
    assert math.isclose(result["coefficients"][1], 3.0, abs_tol=1e-10)
    assert result["n"] == 10


def test_sanitize_tree_records_undefined_statistics_as_null() -> None:
    assert sanitize_tree({"r": float("nan"), "ci": [float("inf"), 1.0]}) == {
        "r": None,
        "ci": [None, 1.0],
    }
