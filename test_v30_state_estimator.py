from datetime import datetime, timedelta, timezone

import pytest

from digital_twin.state_estimator import EstimatorConfig, OnlineStateEstimator


T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def estimator():
    return OnlineStateEstimator(
        initial_means={"demand": 100, "supply": 80, "rival_quote_index": 1, "shock_probability": 0},
        initial_variances={"demand": 100, "supply": 100, "rival_quote_index": 1, "shock_probability": 1},
    )


def test_repeated_measurements_reduce_uncertainty_and_converge():
    model = estimator()
    first = model.update(T0, {"demand": 120})
    second = model.update(T0 + timedelta(hours=1), {"demand": 120})
    assert abs(second.means["demand"] - 120) < abs(first.means["demand"] - 120)
    assert second.variances["demand"] < first.variances["demand"]


def test_missing_feed_advances_prediction_and_increases_uncertainty():
    model = estimator()
    before = model.update(T0, {"demand": 110})
    after = model.update(T0 + timedelta(hours=3), {})
    assert after.variances["demand"] > before.variances["demand"]
    assert after.observed == ()


def test_measurement_variance_controls_outlier_influence():
    low_trust = estimator()
    high_trust = estimator()
    a = low_trust.update(T0, {"demand": 1000}, measurement_variance={"demand": 10000})
    b = high_trust.update(T0, {"demand": 1000}, measurement_variance={"demand": 1})
    assert b.means["demand"] > a.means["demand"]


def test_forecast_uses_learned_trend_and_preserves_domains():
    model = estimator()
    model.update(T0, {"demand": 100, "shock_probability": 0.2})
    model.update(T0 + timedelta(hours=1), {"demand": 120, "shock_probability": 0.8})
    forecast = model.forecast(2)
    assert forecast["demand"] > model.state().means["demand"]
    assert 0 <= forecast["shock_probability"] <= 1


def test_snapshot_restore_is_exact():
    model = estimator()
    model.update(T0, {"demand": 110, "supply": 70})
    model.update(T0 + timedelta(hours=1), {"demand": 130})
    restored = OnlineStateEstimator.restore(model.snapshot())
    assert restored.snapshot() == model.snapshot()
    assert restored.forecast(3) == model.forecast(3)


def test_negative_internal_trend_projects_forecast_but_negative_data_fails():
    model = estimator()
    model.update(
        T0,
        {"demand": 100, "supply": 80, "rival_quote_index": 1,
         "shock_probability": 0.1},
    )
    model.update(
        T0 + timedelta(hours=1),
        {"demand": 0, "supply": 0, "rival_quote_index": 0,
         "shock_probability": 0},
    )
    forecast = model.forecast(24)
    assert forecast["demand"] == 0
    assert forecast["supply"] == 0
    assert forecast["rival_quote_index"] == 0
    assert forecast["shock_probability"] == 0

    with pytest.raises(ValueError, match="demand must be non-negative"):
        model.update(T0 + timedelta(hours=2), {"demand": -1})


def test_out_of_order_nonfinite_and_invalid_domains_fail():
    model = estimator()
    model.update(T0, {})
    with pytest.raises(ValueError, match="out-of-order"):
        model.update(T0 - timedelta(seconds=1), {})
    with pytest.raises(ValueError):
        model.update(T0 + timedelta(hours=1), {"supply": -1})
    with pytest.raises(ValueError):
        model.update(T0 + timedelta(hours=1), {"shock_probability": 1.1})


def test_aggregate_uncertainty_is_finite_and_positive():
    state = estimator().update(T0, {})
    assert 0 < state.aggregate_uncertainty < float("inf")
