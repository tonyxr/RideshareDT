import copy

import pytest

from driver_supply import DriverSupplyConfig, DriverSupplyLayer, FirmDriverBatchState


def _v3_layer() -> DriverSupplyLayer:
    layer = DriverSupplyLayer(
        seed=17,
        config=DriverSupplyConfig(
            supply_model="v3",
            persistent_stock=False,
            reference_earnings_per_hour=20.0,
            eta_v3=0.6,
            hour_profile={hour: 1.0 for hour in range(24)},
        ),
    )
    # Fixed utilisation: 2 trips per online hour at $10 pay per trip.
    state = FirmDriverBatchState(
        active_drivers=100,
        idle_drivers=80,
        busy_drivers=20,
        completed_rides=200,
        total_driver_pay=2000.0,
        total_online_minutes=6000.0,
    )
    layer.smoothed_states["Firm1"] = state
    layer.pay_index_smoothed["Firm1"] = 1.0
    return layer


def test_doubling_pay_at_fixed_utilisation_scales_online_supply_by_two_to_eta():
    layer = _v3_layer()
    pay, trips, expected = layer.supply_v3_expected_earnings("Firm1")
    assert pay == pytest.approx(10.0)
    assert trips == pytest.approx(2.0)
    assert expected == pytest.approx(20.0)
    baseline = layer.supply_v3_online_response("Firm1")

    layer.set_pay_index("Firm1", 2.0)
    pay, trips, expected = layer.supply_v3_expected_earnings("Firm1")
    doubled = layer.supply_v3_online_response("Firm1")
    assert pay == pytest.approx(20.0)
    assert trips == pytest.approx(2.0)
    assert expected == pytest.approx(40.0)
    assert doubled / baseline == pytest.approx(2.0**0.6, abs=1e-9)


def test_v3_neutral_fallback_uses_fixed_reference():
    layer = DriverSupplyLayer(
        seed=3,
        config=DriverSupplyConfig(
            supply_model="v3", reference_earnings_per_hour=24.0
        ),
    )
    assert layer.supply_v3_expected_earnings("Firm1") == (24.0, 1.0, 24.0)
    assert layer.supply_v3_online_response("Firm1") == pytest.approx(1.0)


def test_v2_path_is_bit_identical_when_only_eta_v3_changes():
    common = dict(
        supply_model="v2",
        persistent_stock=True,
        reference_earnings_per_hour=20.0,
        eta_day=0.4,
        eta_hour=0.2,
        hour_profile={hour: 1.0 for hour in range(24)},
    )
    low = DriverSupplyLayer(seed=99, config=DriverSupplyConfig(**common, eta_v3=0.0))
    high = DriverSupplyLayer(seed=99, config=DriverSupplyConfig(**common, eta_v3=9.0))
    for layer in (low, high):
        layer.set_pay_index("Firm1", 2.0)
        layer.begin_batch(500, 12, "clear")
    assert low._batch_states == high._batch_states
    assert low.expected_block_earnings("Firm1") == pytest.approx(40.0, abs=1e-9)
    assert high.expected_block_earnings("Firm1") == pytest.approx(40.0, abs=1e-9)
    low.end_batch()
    high.end_batch()
    assert copy.deepcopy(low.advance_day()) == copy.deepcopy(high.advance_day())


def test_supply_model_validation_includes_v3_only():
    assert DriverSupplyConfig(supply_model="V3").supply_model == "v3"
    with pytest.raises(ValueError, match="supply_model"):
        DriverSupplyConfig(supply_model="v4")
