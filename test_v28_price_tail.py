"""Regression tests for the uncapped-fare demand tail and spillover choice."""

from __future__ import annotations

import numpy as np

from choice_models import CognitiveChoiceModel
from driver_supply import DriverPayPolicy
from market_engine import (
    EngineConfig,
    FleetState,
    MarketEngine,
    cognitive_probabilities,
    profile_features,
)
from market_shocks import ShockEffect
from market_dynamics import FirmReputationMemory


def _market_inputs(n: int = 160):
    profiles = [
        {
            "Age": 35,
            "IncomeBracket": "50k-100k",
            "HouseholdSize": 2,
            "EmploymentStatus": "Employed",
            "LoyaltyType": "New",
            "LoyaltyFirm": None,
            "LoyaltyStrength": 0.0,
            "PriceThreshold": 1.5,
            "ExperimentalUnboundedEconomics": True,
        }
        for _ in range(n)
    ]
    trips = {
        "distance": np.full(n, 5.0),
        "duration": np.full(n, 20.0),
        "airport": np.zeros(n, dtype=bool),
        "service": np.asarray(["economy"] * n),
    }
    return profiles, trips, np.full(n, 20.0)


def _outage_step(
    multiplier: float,
    *,
    unbounded: bool = True,
    demand_model: str = "cognitive",
    pay_basis: str = "anchor",
    spillover: bool = True,
    rider_memory: bool = False,
):
    profiles, trips, anchor = _market_inputs()
    model = CognitiveChoiceModel(seed=1)
    model.make_symmetric()
    model.experimental_unbounded_economics = unbounded
    if rider_memory:
        memory = FirmReputationMemory(alpha=1.0)
        # This is the adversarial old failure mode: a factor below one used to
        # post-multiply p(no ride), leaving a fixed ride floor at infinite fare.
        memory.price_index = {"Firm1": 0.2, "Firm2": 0.2}
        model.reputation = memory
    engine = MarketEngine(
        EngineConfig(
            enabled=True,
            fleet_per_base_request=0.5,
            demand_model=demand_model,
        )
    )
    policies = {
        firm: DriverPayPolicy(
            fare_share=0.72,
            experimental_unbounded_economics=unbounded,
        )
        for firm in ("Firm1", "Firm2")
    }
    shock = ShockEffect(
        firm_capacity_multiplier={"Firm1": 1.0, "Firm2": 0.0},
        active=("platform_outage",),
    )
    fleet = FleetState()
    online = engine.online_drivers(
        fleet,
        base_requests=100.0,
        supply_multiplier=1.0,
        weather="clear",
        shock=shock,
        reservation_wage=24.0,
    )
    return engine.step(
        model=model,
        profiles=profiles,
        trips=trips,
        hour=13,
        day_of_week=2,
        weather="clear",
        price={"Firm1": anchor * multiplier, "Firm2": anchor},
        pay_basis={
            "Firm1": anchor * multiplier if pay_basis == "fare" else anchor,
            "Firm2": anchor,
        },
        pay_policies=policies,
        fleet=fleet,
        online=online,
        block_minutes=240.0,
        realized_requests=100.0,
        shock=shock,
        reservation_wage=24.0,
        operating_cost_per_mile=0.35,
        acceptance_threshold=1.0,
        platform_variable_cost=0.45,
        spillover=spillover,
    )


def test_v28_outside_option_can_absorb_all_demand_at_extreme_fares() -> None:
    profiles, trips, _ = _market_inputs()
    model = CognitiveChoiceModel(seed=1)
    model.experimental_unbounded_economics = True
    n = len(profiles)
    p0, pf1, _ = cognitive_probabilities(
        model,
        profile_features(profiles),
        trips,
        hour=13,
        day_of_week=2,
        weather="clear",
        price1=np.full(n, 1.0e100),
        price2=np.full(n, 1.0e100),
        wait1=np.full(n, 4.0),
        wait2=np.full(n, 4.0),
        cancel1=np.zeros(n),
        cancel2=np.zeros(n),
        shock=ShockEffect(),
    )
    np.testing.assert_array_equal(p0, np.ones(n))
    assert np.all(np.isfinite(pf1))


def test_overflow_riders_reconsider_extreme_alternative_and_profit_turns_down() -> None:
    results = [_outage_step(value) for value in (2.0, 4.0, 8.0, 16.0)]
    spill = np.asarray(
        [row.metrics["Firm1"]["spillover_attempts"] for row in results]
    )
    profit = np.asarray([row.metrics["Firm1"]["profit"] for row in results])
    assert np.all(np.diff(spill) < 0.0)
    assert profit[1] < profit[0] * 0.01
    assert profit[-1] < 1.0e-10
    assert np.max(results[-1].rider["spill_retry_probability_firm1"]) == 0.0


def test_legacy_spillover_floor_is_unchanged() -> None:
    result = _outage_step(16.0, unbounded=False)
    retry = result.rider["spill_retry_probability_firm1"]
    np.testing.assert_allclose(retry, 0.75 * 0.25)
    assert np.max(result.rider["p_no_ride"]) <= 0.98


def test_uncapped_tail_is_well_posed_across_demand_pay_and_spillover_variants() -> None:
    """No supported v28 engine variant may retain captive high-price demand."""
    multipliers = (1.0, 2.0, 4.0, 8.0, 16.0, 64.0)
    for demand_model in ("cognitive", "nested_logit"):
        for pay_basis in ("anchor", "fare"):
            for spillover in (False, True):
                results = [
                    _outage_step(
                        multiplier,
                        demand_model=demand_model,
                        pay_basis=pay_basis,
                        spillover=spillover,
                    )
                    for multiplier in multipliers
                ]
                completed = np.asarray(
                    [row.metrics["Firm1"]["completed"] for row in results]
                )
                profit = np.asarray(
                    [row.metrics["Firm1"]["profit"] for row in results]
                )
                assert np.all(np.isfinite(completed))
                assert np.all(np.isfinite(profit))
                assert completed[-1] <= 1.0e-8
                assert profit[-1] < max(1.0e-8, 1.0e-6 * np.max(profit[:-1]))
                assert int(np.argmax(profit)) < len(multipliers) - 1


def test_rider_memory_odds_shift_preserves_zero_demand_at_infinite_fare() -> None:
    profiles, trips, _ = _market_inputs()
    for demand_model in ("cognitive", "nested_logit"):
        model = CognitiveChoiceModel(seed=2)
        model.experimental_unbounded_economics = True
        memory = FirmReputationMemory(alpha=1.0)
        memory.price_index = {"Firm1": 0.2, "Firm2": 0.2}
        assert memory.outside_option_factor() < 1.0
        model.reputation = memory
        # The full spillover path, not only the direct probability helper,
        # must lose all demand and revenue in the uncapped tail.
        results = [
            _outage_step(
                value,
                demand_model=demand_model,
                rider_memory=True,
            )
            for value in (4.0, 16.0, 64.0)
        ]
        completed = np.asarray(
            [row.metrics["Firm1"]["completed"] for row in results]
        )
        profit = np.asarray([row.metrics["Firm1"]["profit"] for row in results])
        assert completed[-1] <= 1.0e-8
        assert profit[-1] < max(1.0e-8, 1.0e-6 * np.max(profit[:-1]))


def test_legacy_rider_memory_probability_multiplier_is_bitwise_preserved() -> None:
    profiles, trips, _ = _market_inputs(12)
    features = profile_features(profiles)
    prices = np.full(12, 1.0e100)
    kwargs = dict(
        feats=features,
        trips=trips,
        hour=13,
        day_of_week=2,
        weather="clear",
        price1=prices,
        price2=prices,
        wait1=np.full(12, 4.0),
        wait2=np.full(12, 4.0),
        cancel1=np.zeros(12),
        cancel2=np.zeros(12),
        shock=ShockEffect(),
    )
    baseline = CognitiveChoiceModel(seed=3)
    baseline.experimental_unbounded_economics = False
    p0_without_memory, _, _ = cognitive_probabilities(baseline, **kwargs)
    remembered = CognitiveChoiceModel(seed=3)
    remembered.experimental_unbounded_economics = False
    memory = FirmReputationMemory(alpha=1.0)
    memory.price_index = {"Firm1": 0.2, "Firm2": 0.2}
    remembered.reputation = memory
    p0_with_memory, _, _ = cognitive_probabilities(remembered, **kwargs)
    expected = np.clip(
        p0_without_memory * memory.outside_option_factor(), 0.0, 0.98
    )
    np.testing.assert_array_equal(p0_with_memory, expected)
