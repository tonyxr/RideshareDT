"""V17 M5 absolute distance/time action-card contract."""
import copy

import pytest

from MarketInteraction import MarketInteraction, RideContext
from pricing_models import (
    FirmRLPricer,
    TWO_LEVER_LEVELS,
    TWO_LEVER_V2_LEVELS,
    TWO_LEVER_V3_LEVELS,
)


KEYS = ["base_fare", "per_minute", "per_mile", "booking_fee", "airport_fee"]


def make_pricer():
    return FirmRLPricer(
        seed=17,
        opt_keys=KEYS,
        independent_coefficient_actions=False,
        action_contract="two_lever_v1",
    )


def action_for(pricer, distance, time):
    return next(
        action for action, cell in pricer.action_to_two_lever_cell.items()
        if cell == pytest.approx((distance, time))
    )


def test_registered_grid_has_exactly_25_feasible_absolute_cells():
    pricer = make_pricer()
    market = MarketInteraction(city_name="Chicago", seed=17)
    assert pricer.agent.action_dim == 25
    assert set(pricer.action_to_two_lever_cell.values()) == {
        (distance, time)
        for distance in TWO_LEVER_LEVELS
        for time in TWO_LEVER_LEVELS
    }
    assert pricer.feasible_action_mask(market).tolist() == [True] * 25
    assert pricer.action_label(action_for(pricer, 0.8, 1.5)) == (
        "cell:distance=0.8,time=1.5"
    )


@pytest.mark.parametrize("distance,time", [
    (distance, time)
    for distance in TWO_LEVER_LEVELS
    for time in TWO_LEVER_LEVELS
])
def test_each_cell_posts_only_absolute_distance_and_time(distance, time):
    pricer = make_pricer()
    market = MarketInteraction(city_name="Chicago", seed=18)
    anchor = market.curr_market
    pricer.apply_action(action_for(pricer, distance, time), market)
    assert pricer.overrides.per_mile == pytest.approx(anchor.per_mile * distance)
    assert pricer.overrides.per_minute == pytest.approx(anchor.per_minute * time)
    assert pricer.overrides.base_fare is None
    assert pricer.overrides.booking_fee is None
    assert pricer.overrides.airport_fee is None
    assert not pricer.last_action_was_saturated


def test_repeating_a_cell_holds_the_posted_fare_exactly():
    pricer = make_pricer()
    market = MarketInteraction(city_name="New York City", seed=19)
    action = action_for(pricer, 1.2, 0.9)
    pricer.apply_action(action, market)
    before = copy.deepcopy(pricer.overrides)
    context = RideContext(
        day_of_week=2,
        hour=12,
        weather="clear",
        airport=False,
        service="economy",
    )
    quote_before = market.quote_price(
        4.0, 15.0, context, overrides=pricer.overrides
    )
    pricer.apply_action(action, market)
    quote_after = market.quote_price(
        4.0, 15.0, context, overrides=pricer.overrides
    )
    assert pricer.overrides == before
    assert quote_after == pytest.approx(quote_before, abs=0.0)
    assert pricer.last_action_was_zero_effect


def test_legacy_contract_is_unchanged_when_flag_is_off():
    legacy = FirmRLPricer(
        seed=20,
        opt_keys=KEYS,
        tariff_targets=(0.8, 1.0, 1.2),
    )
    assert not legacy.two_lever_actions
    assert legacy.action_label(0) == "hold"
    assert not hasattr(legacy, "action_to_two_lever_cell")


def test_two_lever_grid_rejects_legacy_target_inputs():
    with pytest.raises(ValueError, match="tariff_targets"):
        FirmRLPricer(
            seed=21,
            opt_keys=KEYS,
            independent_coefficient_actions=False,
            action_contract="two_lever_v1",
            tariff_targets=(0.8, 1.0),
        )


def test_core_flag_wires_and_persists_the_25_cell_contract(tmp_path):
    from Core import Core
    import torch

    core = Core(
        "Chicago",
        seed=22,
        firm1_mode="RL",
        firm2_mode="static",
        total_customers_pool=8,
        simulation_sample_cap=8,
        enable_driver_supply=False,
        openai_api_key="",
        action_contract="two_lever_v1",
    )
    assert core.action_contract == "two_lever_v1"
    assert core.firm1.two_lever_actions
    assert core.firm1.agent.action_dim == 25
    assert core._action_feature_matrix_for_pricer(core.firm1, {}).shape == (25, 20)
    artifact_path = tmp_path / "two-lever-policy.pt"
    core.save_trained_model(str(artifact_path))
    artifact = torch.load(artifact_path, weights_only=False, map_location="cpu")
    assert artifact["action_semantics"] == "two_lever_absolute_card_v1"
    assert len(artifact["action_to_two_lever_cell"]) == 25
    core.load_trained_model(str(artifact_path))


def test_t0_14_v2_grid_has_36_cells_and_v1_mapping_is_unchanged():
    v1 = make_pricer()
    v2 = FirmRLPricer(
        seed=23,
        opt_keys=KEYS,
        independent_coefficient_actions=False,
        action_contract="two_lever_v2",
    )
    assert v1.agent.action_dim == 25
    assert tuple(v1.two_lever_levels) == TWO_LEVER_LEVELS
    assert list(v1.action_to_two_lever_cell.values()) == [
        (distance, time)
        for distance in TWO_LEVER_LEVELS
        for time in TWO_LEVER_LEVELS
    ]
    assert v2.agent.action_dim == 36
    assert tuple(v2.two_lever_levels) == TWO_LEVER_V2_LEVELS
    assert list(v2.action_to_two_lever_cell.values()) == [
        (distance, time)
        for distance in TWO_LEVER_V2_LEVELS
        for time in TWO_LEVER_V2_LEVELS
    ]
    assert v1.action_label(0) == "cell:distance=0.8,time=0.8"
    assert v2.action_label(0) == "cell:distance=0.7,time=0.7"


def test_t0_14_v2_core_value_surface_and_artifact_are_36_cells(tmp_path):
    from Core import Core
    import torch

    core = Core(
        "Chicago",
        seed=24,
        firm1_mode="RL",
        firm2_mode="static",
        total_customers_pool=8,
        simulation_sample_cap=3,
        enable_driver_supply=False,
        openai_api_key="",
        action_contract="two_lever_v2",
        v16_crn_tape=True,
        v16_branch_mode="one_day_deviation",
        v16_branch_design="stencil11",
        v16_store_cap=20,
        v16_value_head="concave2d",
        v16_policy_mode="q",
    )
    assert core.v16_value_cells.shape == (36, 2)
    artifact_path = tmp_path / "two-lever-v2-policy.pt"
    core.save_trained_model(str(artifact_path))
    artifact = torch.load(artifact_path, weights_only=False, map_location="cpu")
    assert artifact["action_semantics"] == "two_lever_absolute_card_v2"
    assert len(artifact["action_to_two_lever_cell"]) == 36
    core.load_trained_model(str(artifact_path))


def test_v3_surge_range_has_exactly_the_requested_36_cells():
    v3 = FirmRLPricer(
        seed=25,
        opt_keys=KEYS,
        independent_coefficient_actions=False,
        action_contract="two_lever_v3",
    )
    assert tuple(v3.two_lever_levels) == TWO_LEVER_V3_LEVELS
    assert tuple(v3.two_lever_levels) == (0.8, 1.0, 1.25, 1.5, 2.0, 2.5)
    assert v3.agent.action_dim == 36
    assert list(v3.action_to_two_lever_cell.values()) == [
        (distance, time)
        for distance in TWO_LEVER_V3_LEVELS
        for time in TWO_LEVER_V3_LEVELS
    ]
    assert v3.action_label(35) == "cell:distance=2.5,time=2.5"


def test_v3_core_artifact_reloads_with_its_registered_grid(tmp_path):
    from Core import Core
    import torch

    core = Core(
        "Chicago",
        seed=26,
        firm1_mode="RL",
        firm2_mode="static",
        total_customers_pool=8,
        simulation_sample_cap=3,
        enable_driver_supply=False,
        openai_api_key="",
        action_contract="two_lever_v3",
        v16_crn_tape=True,
        v16_branch_mode="one_day_deviation",
        v16_branch_design="stencil11",
        v16_store_cap=20,
        v16_value_head="concave2d",
        v16_policy_mode="q",
    )
    artifact_path = tmp_path / "two-lever-v3-policy.pt"
    core.save_trained_model(str(artifact_path))
    artifact = torch.load(artifact_path, weights_only=False, map_location="cpu")
    assert artifact["action_semantics"] == "two_lever_absolute_card_v3"
    assert len(artifact["action_to_two_lever_cell"]) == 36
    core.load_trained_model(str(artifact_path))
