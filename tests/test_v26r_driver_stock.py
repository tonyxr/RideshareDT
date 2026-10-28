"""Persistent online-driver stock: partial adjustment, incentive, shortage and recovery."""
import numpy as np
from driver_supply import DriverSupplyConfig, DriverSupplyLayer, FirmDriverBatchState


def _layer(persistent=True, kappa=0.3):
    cfg = DriverSupplyConfig(persistent_stock=persistent, stock_adjustment_rate=kappa)
    layer = DriverSupplyLayer(seed=0, config=cfg)
    for firm in ("Firm1", "Firm2"):
        st = FirmDriverBatchState(active_drivers=100, idle_drivers=80, busy_drivers=20)
        st.total_driver_pay = 100.0 * 24.0
        st.total_online_minutes = 100.0 * 60.0
        layer.smoothed_states[firm] = st
    return layer


def test_incentive_raises_target_and_stock_adjusts_monotonically():
    layer = _layer()
    base_target = layer.stock_target("Firm1")
    layer.pay_policies["Firm1"].incentive_multiplier = 1.2
    high_target = layer.stock_target("Firm1")
    assert high_target > base_target
    stocks = []
    for _ in range(10):
        stocks.append(layer.advance_day()["Firm1"])
    assert all(b >= a - 1e-12 for a, b in zip(stocks, stocks[1:]))
    assert abs(stocks[-1] - high_target) < abs(stocks[0] - high_target)


def test_shortage_lowers_stock_and_recovery_is_gradual():
    layer = _layer()
    layer.apply_shock(fleet_multiplier=0.65)
    for _ in range(5):
        low = layer.advance_day()["Firm1"]
    assert low < 0.9
    layer.apply_shock(fleet_multiplier=1.0)
    first = layer.advance_day()["Firm1"]
    assert low < first < 1.0  # recovers, but not at once
    assert layer.online_stock["Firm1"] >= layer.config.stock_floor


def test_legacy_mode_unchanged_when_persistent_stock_off():
    layer = _layer(persistent=False)
    before = dict(layer.online_stock)
    layer.advance_day()
    assert layer.online_stock == before
    layer.begin_batch(customers_per_step=100, hour=12, weather="clear")
    assert layer._batch_states["Firm1"].active_drivers > 0
