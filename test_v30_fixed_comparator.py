import math

import pytest

from digital_twin.fixed_comparator import (
    FixedComparatorConfig, FixedFiveDimensionalComparator, TARIFF_KEYS,
)


TARGET = {
    "base_fare": 1.25,
    "per_minute": 1.60,
    "per_mile": 2.00,
    "booking_fee": 1.10,
    "airport_fee": 2.80,
}


def objective(action, seed):
    # Common seed term cancels in candidate comparisons but exercises CRN use.
    noise = ((seed * 17) % 13 - 6) * 0.001
    loss = sum((math.log(action[key]) - math.log(TARGET[key])) ** 2 for key in TARIFF_KEYS)
    return 10.0 - loss + noise


def test_optimizer_recovers_nonuniform_five_dimensional_optimum():
    optimizer = FixedFiveDimensionalComparator(
        FixedComparatorConfig(population=96, iterations=12, initial_log_std=0.7)
    )
    result = optimizer.optimize(objective, seeds=[1, 2, 3, 4], rng_seed=30)
    for key in TARIFF_KEYS:
        assert result.multipliers[key] == pytest.approx(TARGET[key], rel=0.12)
    assert len({round(value, 2) for value in result.multipliers.values()}) > 3
    assert result.mean_objective > 9.98


def test_anchor_is_explicit_candidate_and_floor_is_respected():
    optimizer = FixedFiveDimensionalComparator(
        FixedComparatorConfig(population=24, iterations=3)
    )
    def anchor_objective(action, seed):
        return -sum((value - 1.0) ** 2 for value in action.values())
    result = optimizer.optimize(anchor_objective, seeds=[10, 11], rng_seed=31)
    assert result.multipliers == {key: 1.0 for key in TARIFF_KEYS}
    assert all(value >= 1.0 for value in result.multipliers.values())


def test_search_is_deterministic_under_same_crn_and_rng():
    optimizer = FixedFiveDimensionalComparator(
        FixedComparatorConfig(population=24, iterations=4)
    )
    first = optimizer.optimize(objective, seeds=[1, 2, 3], rng_seed=32)
    second = optimizer.optimize(objective, seeds=[1, 2, 3], rng_seed=32)
    assert first == second


def test_invalid_objective_candidates_are_counted_not_clipped():
    optimizer = FixedFiveDimensionalComparator(
        FixedComparatorConfig(population=20, iterations=2, initial_log_std=3)
    )
    def sometimes_invalid(action, seed):
        return float("nan") if action["airport_fee"] > 3 else -sum(action.values())
    result = optimizer.optimize(sometimes_invalid, seeds=[1, 2], rng_seed=33)
    assert result.invalid_candidates > 0
    assert all(value >= 1 for value in result.multipliers.values())


def test_initial_contract_and_seed_integrity_fail_closed():
    optimizer = FixedFiveDimensionalComparator()
    with pytest.raises(ValueError):
        optimizer.optimize(objective, seeds=[], rng_seed=1)
    with pytest.raises(ValueError):
        optimizer.optimize(objective, seeds=[1, 1], rng_seed=1)
    with pytest.raises(ValueError):
        optimizer.optimize(
            objective, seeds=[1], rng_seed=1,
            initial_multipliers={**TARGET, "base_fare": 0.9},
        )
