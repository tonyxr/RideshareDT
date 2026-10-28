"""Observation frame dimensions, public ablation and opt-in oracle diagnostic."""
import numpy as np
import pytest
from platform_mdp import (
    ORACLE_SHORTAGE_FEATURE_DIM,
    ObservationConfig,
    PlatformObservationModel,
    REGIME_FEATURE_DIM,
)


COEFS = {"base_fare": 3.0, "per_minute": 0.3, "per_mile": 1.2, "booking_fee": 2.0, "airport_fee": 0.0}


def _obs(model, regime):
    return model.build_observation(hour=9, day_of_week=1, weather="clear", own_coefficients=COEFS,
                                   anchor_coefficients=COEFS, regime_features=regime)


def test_dimensions_and_ablation():
    base = PlatformObservationModel(1)
    assert base.observation_dim == 24 and _obs(base, None).size == 24
    on = PlatformObservationModel(1, ObservationConfig(regime_features=True))
    assert on.observation_dim == 24 + REGIME_FEATURE_DIM
    v = _obs(on, {"storm_warning": 1.0, "transit_availability": 0.3})
    assert v.size == 30 and v[24 + 2] == 1.0 and abs(v[24 + 4] - 0.3) < 1e-6
    assert max(on.regime_feature_groups()["immediate"]) == 29
    abl = PlatformObservationModel(1, ObservationConfig(regime_features=True, regime_feature_ablation=True))
    w = _obs(abl, {"storm_warning": 1.0, "transit_availability": 0.3})
    assert w.size == 30 and w[24 + 2] == 0.0 and w[24 + 4] == 1.0


def test_persistence_proxies_extend_the_frame():
    from platform_mdp import PERSISTENCE_FEATURE_DIM
    both = PlatformObservationModel(1, ObservationConfig(regime_features=True, persistence_features=True))
    assert both.observation_dim == 24 + REGIME_FEATURE_DIM + PERSISTENCE_FEATURE_DIM == 34
    v = both.build_observation(hour=9, day_of_week=1, weather="clear", own_coefficients=COEFS, anchor_coefficients=COEFS,
                               regime_features={"storm_warning": 1.0},
                               persistence_features={"driver_earnings_index": 0.25,
                                                     "same_day_driver_supply_index": -0.4,
                                                     "returning_share": 0.6,
                                                     "rival_quote_index_7d": -2.0})
    assert v.size == 34
    assert abs(v[30] - 0.25) < 1e-6
    assert abs(v[31] + 0.4) < 1e-6
    assert abs(v[32] - 0.6) < 1e-6 and v[33] == -1.0
    assert max(both.regime_feature_groups()["immediate"]) == 33
    only = PlatformObservationModel(1, ObservationConfig(persistence_features=True))
    assert only.observation_dim == 28 and _obs(only, None).size == 28


def test_true_shortage_flag_is_explicit_opt_in_and_survives_public_ablation():
    diagnostic = PlatformObservationModel(
        1,
        ObservationConfig(
            regime_features=True,
            regime_feature_ablation=True,
            actor_true_shortage_flag=True,
            persistence_features=True,
        ),
    )
    assert diagnostic.observation_dim == (
        24 + REGIME_FEATURE_DIM + ORACLE_SHORTAGE_FEATURE_DIM + 4
    )
    v = diagnostic.build_observation(
        hour=9,
        day_of_week=1,
        weather="clear",
        own_coefficients=COEFS,
        anchor_coefficients=COEFS,
        regime_features={"storm_warning": 1.0, "driver_shortage_active": 1.0},
        persistence_features={},
    )
    assert v[24 + 2] == 0.0
    assert v[24 + REGIME_FEATURE_DIM] == 1.0


def test_same_day_supply_proxy_is_public_and_distinct_from_oracle_flag():
    from types import SimpleNamespace
    from Core import Core

    core = object.__new__(Core)
    core.enable_driver_supply = True
    core.learning_corrections = True
    core._rival_gap_history = []
    core._returning_share_ema = 0.0
    core.driver_supply = SimpleNamespace(
        online_stock={"Firm1": 0.55, "Firm2": 0.80},
        smoothed_states={"Firm1": None, "Firm2": None},
        _earnings_reference=lambda firm: 24.0,
    )
    f1 = core._persistence_public_features("Firm1")
    f2 = core._persistence_public_features("Firm2")
    assert f1["same_day_driver_supply_index"] == pytest.approx(-0.45)
    assert f2["same_day_driver_supply_index"] == pytest.approx(-0.20)
    assert "driver_shortage_active" not in f1
