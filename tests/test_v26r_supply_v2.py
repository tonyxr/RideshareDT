"""Supply v2: pay elasticity across days, hour profile, within-day response."""
import numpy as np
from driver_supply import DriverSupplyConfig, DriverSupplyLayer, FirmDriverBatchState, normalized_hour_profile


def _layer(**kw):
    cfg = DriverSupplyConfig(supply_model="v2", persistent_stock=True, stock_adjustment_rate=0.25,
                             reference_earnings_per_hour=20.0, **kw)
    layer = DriverSupplyLayer(seed=0, config=cfg)
    layer.reset_stock()
    for firm in ("Firm1", "Firm2"):
        st = FirmDriverBatchState(active_drivers=100, idle_drivers=80, busy_drivers=20)
        st.total_driver_pay = 100.0 * 20.0
        st.total_online_minutes = 100.0 * 60.0
        layer.smoothed_states[firm] = st
    return layer


def test_hour_profile_day_exceeds_night_and_is_mean_one():
    prof = normalized_hour_profile()
    assert abs(np.mean(list(prof.values())) - 1.0) < 1e-9
    assert prof[17] > 1.2 and prof[3] < 0.5


def test_daily_stock_follows_pay_with_elasticity():
    layer = _layer(eta_day=0.4)
    for _ in range(40):
        layer._daily_pay['Firm1'], layer._daily_online['Firm1'] = 2400., 6000.
        layer.advance_day()
    assert abs(layer.online_stock["Firm1"] - 1.2 ** 0.4) < 0.01
    stocks = []
    for _ in range(40):
        layer._daily_pay['Firm1'], layer._daily_online['Firm1'] = 1600., 6000.
        stocks.append(layer.advance_day()["Firm1"])
    assert stocks[0] > stocks[-1] and abs(stocks[-1] - 0.8 ** 0.4) < 0.01
    assert all(b <= a + 1e-12 for a, b in zip(stocks, stocks[1:]))  # gradual decline


def test_within_day_response_and_night_scarcity():
    layer = _layer(eta_hour=0.2)
    layer.begin_batch(customers_per_step=500, hour=17, weather="clear")
    day = layer._batch_states["Firm1"].active_drivers
    layer.begin_batch(customers_per_step=500, hour=3, weather="clear")
    night = layer._batch_states["Firm1"].active_drivers
    assert day > 2.5 * night
    layer.set_pay_index("Firm1", 1.5)  # surge pay posted this block
    layer.begin_batch(customers_per_step=500, hour=3, weather="clear")
    night_surge = layer._batch_states["Firm1"].active_drivers
    assert 1.05 * night < night_surge < 1.2 * night  # 1.5**0.2 = 1.084


def test_v1_unchanged_by_v2_fields():
    cfg = DriverSupplyConfig()
    assert cfg.supply_model == "v1" and not cfg.persistent_stock
    layer = DriverSupplyLayer(seed=0, config=cfg)
    layer.begin_batch(customers_per_step=100, hour=3, weather="clear")
    assert layer._batch_states["Firm1"].active_drivers > 0
