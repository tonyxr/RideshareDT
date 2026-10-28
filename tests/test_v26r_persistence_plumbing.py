"""The persistence signals reach the actor frame and the privileged critic (no torch needed)."""
from collections import deque
import numpy as np
from Core import Core, REGIME_CRITIC_LATENT_DIM
from driver_supply import DriverSupplyConfig, DriverSupplyLayer, FirmDriverBatchState
from pricing_models import FirmMemoryBestResponsePricer
from market_shocks import ShockEffect


class _Stub:
    pass


def _supply(earnings_per_online_hour: float, reference: float):
    layer = DriverSupplyLayer(seed=0, config=DriverSupplyConfig(persistent_stock=True, supply_model="v2",
                                                                reference_earnings_per_hour=reference))
    layer.reset_stock()
    st = FirmDriverBatchState(active_drivers=100, idle_drivers=80, busy_drivers=20)
    st.total_driver_pay = earnings_per_online_hour * 100.0
    st.total_online_minutes = 100.0 * 60.0
    layer.smoothed_states["Firm1"] = st
    return layer


def test_public_proxies_track_earnings_returning_share_and_rival_gap():
    core = _Stub()
    core.enable_driver_supply = True
    core.driver_supply = _supply(30.0, 24.0)
    core._returning_share_ema = 0.55
    core._rival_gap_history = deque([4.0, 4.0, 2.0], maxlen=42)
    proxies = Core._persistence_public_features(core)
    assert abs(proxies["driver_earnings_index"] - 0.25) < 1e-6
    assert abs(proxies["returning_share"] - 0.55) < 1e-9
    assert abs(proxies["rival_quote_index_7d"] - (10.0 / 3.0) / 8.0) < 1e-9
    core.driver_supply = _supply(12.0, 24.0)
    assert Core._persistence_public_features(core)["driver_earnings_index"] < 0.0


def test_critic_latent_block_carries_loyalty_and_rival_state():
    core = _Stub()
    core.regime_market = True
    core._regime_last_effect = ShockEffect()
    core.market = _Stub(); core.market.active_shocks = None
    core.driver_supply = _Stub(); core.driver_supply.online_stock = {"Firm1": 0.9, "Firm2": 1.1}
    core.rider_loyalty = {0: {"Firm1": 0.6, "Firm2": 0.0}, 1: {"Firm1": 0.2, "Firm2": 0.4}}
    core._loyalty_mean = lambda firm: Core._loyalty_mean(core, firm)
    core.firm2 = FirmMemoryBestResponsePricer(seed=0)
    core.firm2.memory_state = "retaliation"
    feats = Core._regime_critic_features(core)
    assert len(feats) == REGIME_CRITIC_LATENT_DIM == 12
    assert abs(feats[6] - (-0.1)) < 1e-9 and abs(feats[7] - 0.1) < 1e-9
    assert abs(feats[8] - 0.4) < 1e-9 and abs(feats[9] - 0.2) < 1e-9
    assert feats[10] == 1.0 and feats[11] == 0.0
    core.regime_market = False
    assert Core._regime_critic_features(core) == []
