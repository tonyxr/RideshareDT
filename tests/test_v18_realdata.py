from __future__ import annotations

import math
import csv
import gzip

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
from scripts.v18_validation.sim_analysis import _aggregate_log, _scenario


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


def test_constant_baseline_has_undefined_correlation_and_slope() -> None:
    result = fare_metrics([1.0, 2.0, 3.0, 4.0], [2.5, 2.5, 2.5, 2.5])
    assert math.isnan(result["r"])
    assert math.isnan(result["rho"])
    assert math.isnan(result["slope"])


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


def test_v3_scenario_target_days_are_declared() -> None:
    path, days, start = _scenario("nyc", "E1_flood_20230929")
    assert path.name == "nyc_e1_flood_20230929.json"
    assert days == [16]
    assert start.isoformat() == "2023-09-13"
    path, days, _ = _scenario("nyc", "E4_snow_20240213")
    assert path.name == "nyc_e4_snow_20240213.json"
    assert days == [16]


def test_request_log_aggregate_uses_fulfilled_market_rows(tmp_path) -> None:
    path = tmp_path / "requests.csv.gz"
    fields = [
        "Day", "DayOfWeek", "Fulfilled", "Choice", "TravelDistance", "DurationMinutes",
        "Airport", "WaitMinutes", "CardClearQuote", "Price_Firm1", "Price_Firm2",
    ]
    with gzip.open(path, "wt", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "Day": 16, "DayOfWeek": 4, "Fulfilled": True, "Choice": "Firm1",
            "TravelDistance": 2, "DurationMinutes": 10, "Airport": False,
            "WaitMinutes": 4, "CardClearQuote": 10, "Price_Firm1": 12, "Price_Firm2": 11,
        })
        writer.writerow({
            "Day": 16, "DayOfWeek": 4, "Fulfilled": False, "Choice": "Firm2",
            "TravelDistance": 4, "DurationMinutes": 20, "Airport": True,
            "WaitMinutes": 8, "CardClearQuote": 20, "Price_Firm1": 20, "Price_Firm2": 24,
        })
    result = _aggregate_log(path, {16})[16]
    assert result["m1_completed_trips"] == 1
    assert result["m2_mean_duration"] == 10
    assert result["m3_median_speed"] == 12
    assert math.isclose(result["m5_price_index"], 1.2)
