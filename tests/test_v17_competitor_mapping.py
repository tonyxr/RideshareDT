"""V17 M6 competitors use the two-lever distance/time diagonal."""

import pytest

from Core import Core
from Market_models import CoefficientOverrides
from pricing_models import FirmTwinProfitOptimizerMixin


def make_core(mode="adaptive_best_response"):
    return Core(
        "Boston",
        seed=61,
        firm1_mode="RL",
        firm2_mode=mode,
        total_customers_pool=8,
        simulation_sample_cap=8,
        enable_driver_supply=False,
        openai_api_key="",
        fare_model="realistic",
        action_contract="two_lever_v1",
        rivals_two_lever=True,
    )


def assert_diagonal(core):
    card = core.market.curr_market
    rival = core.firm2.overrides
    assert rival.base_fare is None
    assert rival.booking_fee is None
    assert rival.airport_fee is None
    assert rival.per_mile / card.per_mile == pytest.approx(
        rival.per_minute / card.per_minute, abs=1e-12
    )


def test_reactive_mapping_preserves_15d_reference_target_within_one_percent():
    core = make_core("adaptive_best_response_memory")
    card = core.market.curr_market
    core.firm2.overrides = CoefficientOverrides(
        base_fare=card.base_fare * 1.08,
        booking_fee=card.booking_fee * 0.94,
        per_mile=card.per_mile * 1.17,
        per_minute=card.per_minute * 0.91,
        airport_fee=card.airport_fee * 1.12,
    )
    target = core._firm2_reference_trip_quote()
    level = core._map_firm2_to_two_lever_diagonal(target)
    posted = core._firm2_reference_trip_quote()
    assert level is not None
    assert_diagonal(core)
    assert abs(posted - target) / max(target, 1e-9) <= 0.01
    assert core.firm2.last_two_lever_target_quote == pytest.approx(target)
    assert core.firm2.last_two_lever_posted_quote == pytest.approx(posted)


def test_reactive_leader_step_posts_only_the_diagonal():
    core = make_core("adaptive_best_response")
    core._firm2_leader_step(
        previous_metrics=None,
        day_of_week=2,
        hour=12,
        weather="clear",
    )
    assert_diagonal(core)
    target = core.firm2.last_two_lever_target_quote
    posted = core.firm2.last_two_lever_posted_quote
    assert abs(posted - target) / max(target, 1e-9) <= 0.01


def test_profit_rival_search_candidates_and_selected_tariff_are_diagonal():
    core = make_core("profit_best_response_myopic")
    assert isinstance(core.firm2, FirmTwinProfitOptimizerMixin)
    core._post_grid_multiplier(core.firm2, 1.2)
    assert_diagonal(core)
    card = core.market.curr_market
    assert core.firm2.overrides.per_mile == pytest.approx(card.per_mile * 1.2)
    assert core.firm2.overrides.per_minute == pytest.approx(card.per_minute * 1.2)
    assert core._tariff_grid_multiplier(core.firm2) == pytest.approx(1.2)

    core.firm2.set_profit_plan(
        multiplier=0.9,
        improvement=0.1,
        standard_error=0.01,
        candidate_means={0.9: 1.0},
    )
    core.firm2.act(
        city_base=card.base_fare,
        city_pmin=card.per_minute,
        city_pmile=card.per_mile,
        city_booking=card.booking_fee,
        city_airport=card.airport_fee,
        hour=12,
        weather="clear",
    )
    assert_diagonal(core)
    assert core.firm2.overrides.per_mile == pytest.approx(card.per_mile * 0.9)
    assert core.firm2.overrides.per_minute == pytest.approx(card.per_minute * 0.9)


def test_flag_is_default_off_and_requires_the_two_lever_action_contract():
    legacy = Core(
        "Chicago",
        seed=62,
        firm1_mode="static",
        firm2_mode="adaptive_best_response",
        total_customers_pool=8,
        simulation_sample_cap=8,
        enable_driver_supply=False,
        openai_api_key="",
    )
    assert not legacy.rivals_two_lever
    assert not legacy.firm2.rivals_two_lever

    with pytest.raises(ValueError, match="rivals_two_lever"):
        Core(
            "Chicago",
            seed=63,
            firm1_mode="static",
            firm2_mode="adaptive_best_response",
            total_customers_pool=8,
            simulation_sample_cap=8,
            enable_driver_supply=False,
            openai_api_key="",
            rivals_two_lever=True,
        )
