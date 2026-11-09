"""Unit and contract tests for the V17 robustness workflow."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from Core import Core
from Market_models import CoefficientOverrides, MarketCoefficients
from driver_supply import DriverSupplyConfig
from optim_config import default_specs_for
from pricing_models import FirmMemoryBestResponsePricer
from scripts.v17_robust_common import check_overrides, set_option
from v17_surge_decay import surge_decay_spec, surge_decay_weight


KEYS = [
    "base_fare", "per_minute", "per_mile", "booking_fee", "airport_fee"
]


def _posting_core() -> Core:
    core = Core.__new__(Core)
    core.shared_steps_per_day = 6
    core.market = SimpleNamespace(curr_market=MarketCoefficients(
        base_fare=2.0,
        per_mile=3.0,
        per_minute=0.5,
        booking_fee=1.0,
        airport_fee=4.0,
        surge_hours=[],
        day_multiplier={},
        weather_multiplier={},
        service_multiplier={},
    ))
    core.firm1 = SimpleNamespace(
        two_lever_actions=True,
        opt_keys=KEYS,
        config=default_specs_for(KEYS),
        overrides=CoefficientOverrides(),
        supply_incentive_multiplier=1.0,
    )
    return core


def _ratios(core: Core) -> dict[str, float]:
    base = core.market.curr_market
    return {
        key: float(getattr(core.firm1.overrides, key)) / float(getattr(base, key))
        for key in KEYS
    }


def _small_core(**kwargs) -> Core:
    return Core(
        "Boston",
        seed=71,
        firm1_mode="RL",
        firm2_mode="adaptive_best_response_memory",
        total_customers_pool=8,
        simulation_sample_cap=8,
        enable_driver_supply=False,
        openai_api_key="",
        fare_model="realistic",
        action_contract="two_lever_v1",
        rivals_two_lever=True,
        **kwargs,
    )


def test_surge_decay_weight_table():
    windows = [[10, 15]]
    expected = {
        9: 0.0,
        10: 1.0,
        17: 1.0,
        18: 0.75,
        19: 0.5,
        20: 0.25,
        21: 0.0,
    }
    for day, weight in expected.items():
        assert surge_decay_weight(day, windows, 2, 3) == pytest.approx(weight)
    assert surge_decay_weight(18, [[10, 15], [18, 18]], 2, 3) == 1.0
    assert surge_decay_weight(10, [], 2, 3) == 0.0


def test_surge_decay_spec_roundtrip():
    spec = surge_decay_spec((0.8, 1.0), (1.5, 2.0), [[10, 15]], 2, 4)
    payload = Core._parse_policy_table(spec.split(":", 1)[1])
    assert payload == {
        "normal": [0.8, 1.0],
        "event": [1.5, 2.0],
        "windows": [[10, 15]],
        "hold_days": 2,
        "decay_days": 4,
    }


def test_surge_decay_posts_full_card():
    core = _posting_core()
    spec = surge_decay_spec((1.0, 1.0), (1.25, 1.25), [[1, 1]], 0, 0)
    observed = []
    for step in (0, 6, 12):
        action = Core._baseline_firm1_action(
            core,
            spec,
            step_index=step,
            action_mask=np.ones(25, dtype=bool),
            day_of_week=0,
            hour=12,
            weather="clear",
            presampled=None,
            customers_per_step=1,
        )
        assert action == -1
        observed.append(_ratios(core))
    assert observed[0] == pytest.approx({key: 1.0 for key in KEYS})
    assert observed[1] == pytest.approx({
        "base_fare": 1.0,
        "per_minute": 1.25,
        "per_mile": 1.25,
        "booking_fee": 1.0,
        "airport_fee": 1.0,
    })
    assert observed[2] == pytest.approx({key: 1.0 for key in KEYS})


def test_eta_v3_reaches_market():
    payload = {
        "v26r": {
            "effective_supply_config": {
                "eta_v3": 0.3,
                "shortage_willingness_share": 0.5,
            },
            "memory_rival_overrides": {
                "hold_mean_days": None,
                "retaliation": "on",
            },
        }
    }
    check_overrides(payload, {"supply_eta_v3": 0.3})
    assert DriverSupplyConfig().eta_v3 == pytest.approx(0.6)


def test_memory_rival_overrides():
    core = _small_core(
        memory_rival_hold_mean_days=10.0,
        memory_rival_retaliation="off",
    )
    core.shared_steps_per_day = 6
    pricer = core._make_pricer("adaptive_best_response_memory", 7)
    assert isinstance(pricer, FirmMemoryBestResponsePricer)
    assert pricer.hold_mean_steps == 60
    assert pricer.lag_steps == 10**9
    other = core._make_pricer("adaptive_best_response", 7)
    assert not isinstance(other, FirmMemoryBestResponsePricer)


def test_retaliation_off_never_retaliates():
    core = _small_core(memory_rival_retaliation="off")
    pricer = core._make_pricer("adaptive_best_response_memory", 7)
    assert pricer.lag_steps == 10**9
    pricer.undercut_steps = 1_000_000
    assert pricer.undercut_steps < pricer.lag_steps


def test_defaults_identity():
    implicit = _small_core()
    explicit = _small_core(
        memory_rival_retaliation="on",
        memory_rival_hold_mean_days=None,
    )
    implicit.shared_steps_per_day = explicit.shared_steps_per_day = 6
    left = implicit._make_pricer("adaptive_best_response_memory", 7)
    right = explicit._make_pricer("adaptive_best_response_memory", 7)
    assert left.hold_mean_steps == right.hold_mean_steps == 30
    assert left.lag_steps == right.lag_steps
    assert set_option(["python", "--x", "1"], "--x", 2) == [
        "python", "--x", "2"
    ]
    assert set_option(["python"], "--x", 2) == ["python", "--x", "2"]
