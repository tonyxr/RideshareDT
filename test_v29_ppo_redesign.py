import numpy as np
import pytest
import torch

from MarketInteraction import MarketInteraction
from PPOAgent import PPOAgent
from market_dynamics import RiderProfileMemory
from market_engine import profile_features
from platform_mdp import (
    LongTermProfitReward,
    LongTermProfitRewardConfig,
    PlatformObservationModel,
    TrainingStageScheduler,
)
from pricing_models import FirmRLPricer


def test_independent_action_space_never_moves_coefficients_uniformly():
    pricer = FirmRLPricer(
        seed=7,
        opt_keys=[
            "base_fare", "per_minute", "per_mile", "booking_fee", "airport_fee"
        ],
        state_frame_stack=4,
        single_state_dim=19,
        independent_coefficient_actions=True,
        experimental_unbounded_economics=True,
    )
    assert pricer.agent.action_dim == 11
    for action, mapping in pricer.action_to_steps.items():
        changed = [key for key, step in mapping.items() if step]
        assert len(changed) == (0 if action == 0 else 1)

    market = MarketInteraction(city_name="Chicago", seed=11)
    before = {
        key: float(getattr(market.curr_market, key)) for key in pricer.opt_keys
    }
    base_up = next(
        action for action, mapping in pricer.action_to_steps.items()
        if mapping.get("base_fare") == 1
    )
    pricer.apply_action(base_up, market)
    after = {
        key: float(
            getattr(pricer.overrides, key)
            if getattr(pricer.overrides, key) is not None
            else getattr(market.curr_market, key)
        )
        for key in pricer.opt_keys
    }
    assert after["base_fare"] > before["base_fare"]
    assert all(after[key] == before[key] for key in pricer.opt_keys[1:])
    assert pricer.agent.scheduled_exploration is False
    assert np.isclose(pricer.agent.initial_exploration_rate, 0.0)
    assert np.isclose(pricer.agent.final_exploration_rate, 0.0)
    assert np.isclose(pricer.agent.initial_hold_probability, 1.0 / 11.0)
    with torch.no_grad():
        logits, _ = pricer.agent.net.actor_forward(
            torch.zeros(1, 4, 19).reshape(1, -1)
        )
    probabilities = torch.softmax(logits, dim=-1).squeeze(0).numpy()
    assert np.isclose(probabilities[0], 1.0 / 11.0, atol=1e-6)
    np.testing.assert_allclose(
        probabilities[1:],
        np.full(10, 1.0 / 11.0),
        atol=1e-6,
    )


def test_unbounded_independent_action_can_move_a_zero_fee():
    pricer = FirmRLPricer(
        seed=8,
        opt_keys=["airport_fee"],
        state_frame_stack=1,
        single_state_dim=19,
        independent_coefficient_actions=True,
        experimental_unbounded_economics=True,
    )
    market = MarketInteraction(city_name="Chicago", seed=12)
    pricer.overrides.airport_fee = 0.0
    up = next(
        action for action, mapping in pricer.action_to_steps.items()
        if mapping.get("airport_fee") == 1
    )
    pricer.apply_action(up, market)
    assert float(pricer.overrides.airport_fee) > 0.0


def test_v29_independent_coefficients_never_fall_below_anchor_multiplier_one():
    market = MarketInteraction(city_name="Chicago", seed=121)
    keys = [
        "base_fare", "per_minute", "per_mile", "booking_fee", "airport_fee"
    ]
    pricer = FirmRLPricer(
        seed=122,
        opt_keys=keys,
        independent_coefficient_actions=True,
        experimental_unbounded_economics=True,
        minimum_tariff_multiplier=1.0,
    )
    for key in keys:
        anchor = float(getattr(market.curr_market, key))
        setattr(pricer.overrides, key, 1.2 * anchor)
        down = next(
            action
            for action, mapping in pricer.action_to_steps.items()
            if mapping.get(key) == -1
        )
        for _ in range(100):
            pricer.apply_action(down, market)
        value = float(getattr(pricer.overrides, key))
        assert value >= anchor
        assert value / anchor >= 1.0
        assert not pricer.feasible_action_mask(market)[down]


def test_legacy_independent_coefficients_can_still_move_below_anchor():
    market = MarketInteraction(city_name="Chicago", seed=123)
    pricer = FirmRLPricer(
        seed=124,
        opt_keys=["base_fare"],
        independent_coefficient_actions=True,
        experimental_unbounded_economics=True,
    )
    down = next(
        action for action, mapping in pricer.action_to_steps.items()
        if mapping.get("base_fare") == -1
    )
    anchor = float(market.curr_market.base_fare)
    pricer.apply_action(down, market)
    assert float(pricer.overrides.base_fare) < anchor


def test_original_ppo_samples_only_effective_actions_at_v29_floor():
    market = MarketInteraction(city_name="Chicago", seed=125)
    pricer = FirmRLPricer(
        seed=126,
        opt_keys=[
            "base_fare", "per_minute", "per_mile", "booking_fee", "airport_fee"
        ],
        state_frame_stack=4,
        single_state_dim=19,
        action_feature_dim=0,
        independent_coefficient_actions=True,
        experimental_unbounded_economics=True,
        minimum_tariff_multiplier=1.0,
        persistent_recurrence=False,
    )
    mask = pricer.feasible_action_mask(market)
    assert int(mask.sum()) == 6  # hold plus five upward actions
    state = np.zeros(4 * 19, dtype=np.float32)
    for _ in range(100):
        action, *_ = pricer.agent.act(state, action_mask=mask)
        assert mask[action]


def test_compact_foresight_observation_has_one_clear_schema():
    observer = PlatformObservationModel(
        3, compact_foresight=True, shock_features=False
    )
    observer.quote_probes["0_2"]["gap"] = 2.0
    observer.quote_probes["10_plus"]["gap"] = -3.0
    observer.demand_mix["distance_mean"] = 6.0
    observer.demand_mix["duration_mean"] = 18.0
    coefficients = {
        "base_fare": 2.0,
        "per_minute": 0.3,
        "per_mile": 1.2,
        "booking_fee": 1.0,
        "airport_fee": 2.0,
    }
    state = observer.build_observation(
        hour=8,
        day_of_week=1,
        weather="clear",
        own_coefficients=coefficients,
        anchor_coefficients=coefficients,
        city_context={"demand_intensity": 0.2, "next_demand_intensity": 0.7},
    )
    assert state.shape == (19,)
    assert np.isclose(state[12], np.arcsinh(0.2))
    assert np.isclose(state[13], np.arcsinh(0.7))
    assert np.isclose(state[15], np.log1p(0.5))
    assert np.isclose(state[16], np.log1p(0.4))
    assert state[17] > 0.0
    assert state[18] < 0.0
    names = PlatformObservationModel.feature_names(
        compact_foresight=True, shock_features=False
    )
    assert len(names) == state.size
    assert names[15:19] == (
        "mean_trip_distance",
        "mean_trip_duration",
        "quote_gap_short_trip",
        "quote_gap_long_trip",
    )


def test_compact_tariff_coordinates_remain_distinct_when_uncapped():
    observer = PlatformObservationModel(4, compact_foresight=True)
    anchor = {
        "base_fare": 2.0,
        "per_minute": 0.3,
        "per_mile": 1.2,
        "booking_fee": 1.0,
        "airport_fee": 2.0,
    }
    coordinates = []
    for multiplier in (1.5, 2.0, 10.0, 1000.0):
        own = dict(anchor)
        own["base_fare"] *= multiplier
        state = observer.build_observation(
            hour=8,
            day_of_week=1,
            weather="clear",
            own_coefficients=own,
            anchor_coefficients=anchor,
        )
        coordinates.append(float(state[4]))
    assert all(right > left for left, right in zip(coordinates, coordinates[1:]))


def test_compact_quote_curve_does_not_alias_equal_mean_different_slopes():
    coefficients = {
        "base_fare": 2.0,
        "per_minute": 0.3,
        "per_mile": 1.2,
        "booking_fee": 1.0,
        "airport_fee": 2.0,
    }
    states = []
    for short_gap, long_gap in ((-2.0, 2.0), (2.0, -2.0)):
        observer = PlatformObservationModel(41, compact_foresight=True)
        observer.quote_probes["0_2"]["gap"] = short_gap
        observer.quote_probes["10_plus"]["gap"] = long_gap
        states.append(observer.build_observation(
            hour=8,
            day_of_week=1,
            weather="clear",
            own_coefficients=coefficients,
            anchor_coefficients=coefficients,
        ))
    assert np.isclose(states[0][17], -states[1][17])
    assert np.isclose(states[0][18], -states[1][18])
    assert not np.array_equal(states[0], states[1])


def test_smdp_centering_has_consistent_units_across_hold_durations():
    gamma = 0.999
    durations = np.asarray([1, 10, 4], dtype=float)
    exposures = torch.tensor(
        [(1.0 - gamma**duration) / (1.0 - gamma) for duration in durations]
    )
    constant_rate = -4.7
    sums = constant_rate * exposures
    centered, rate = PPOAgent.center_smdp_reward_sums(sums, exposures)
    assert np.isclose(rate, constant_rate, atol=1e-6)
    np.testing.assert_allclose(centered.numpy(), 0.0, atol=2e-6)


def test_profit_with_dominance_is_profit_minus_one_hinge_constraint():
    reward = LongTermProfitReward(LongTermProfitRewardConfig(
        objective_mode="profit_with_dominance",
        own_profit_scale=2.0,
        market_share_gap_scale=0.10,
    ))
    dominant = reward.compute(
        own_profit_per_request=2.0,
        rival_profit_per_request=9.0,
        own_completed_share=0.45,
        rival_completed_share=0.35,
    )
    losing = reward.compute(
        own_profit_per_request=2.0,
        rival_profit_per_request=0.0,
        own_completed_share=0.35,
        rival_completed_share=0.45,
    )
    assert np.isclose(dominant["reward"], 1.0)
    assert np.isclose(losing["reward"], 0.0)
    assert np.isclose(losing["reward_dominance_constraint_penalty"], 1.0)


def test_two_stage_curriculum_is_environment_then_opponent():
    scheduler = TrainingStageScheduler("environment_then_opponent")
    assert [stage.name for stage in scheduler.stages] == ["environment", "opponent"]
    assert scheduler.stage_at(0.1).opponent_regime == "market"
    assert scheduler.stage_at(0.8).opponent_regime == "dynamic"
    assert scheduler.stage_at(0.8).opponent_pool_size == 1


def test_rider_profile_memory_is_identity_stable_and_profile_specific():
    memory = RiderProfileMemory(alpha=1.0, choice_weight=1.5, outside_weight=0.5)
    rider_a = {
        "ThresholdSourceProfileSHA256": "rider-a",
        "IncomeBracket": "<50k",
        "LoyaltyFirm": "Firm1",
        "LoyaltyStrength": 0.8,
        "PriceThreshold": 2.0,
    }
    rider_a_copy = dict(rider_a)
    rider_b = {
        **rider_a,
        "ThresholdSourceProfileSHA256": "rider-b",
        "LoyaltyFirm": "Firm2",
    }
    memory.update(
        [rider_a, rider_b],
        choice_target=np.asarray([0.8, -0.6]),
        outside_target=np.asarray([-0.4, 0.7]),
    )
    features = profile_features(
        [rider_a_copy, rider_b], profile_memory=memory
    )
    assert features["profile_memory_choice"][0] == pytest.approx(1.2)
    assert features["profile_memory_choice"][1] == pytest.approx(-0.9)
    assert features["profile_memory_outside"][0] == pytest.approx(-0.2)
    assert features["profile_memory_outside"][1] == pytest.approx(0.35)
    assert RiderProfileMemory.identity(rider_a) == RiderProfileMemory.identity(
        rider_a_copy
    )
    assert RiderProfileMemory.identity(rider_a) != RiderProfileMemory.identity(
        rider_b
    )

    memory.reset()
    reset = profile_features([rider_a], profile_memory=memory)
    assert reset["profile_memory_choice"][0] == pytest.approx(0.0)
    assert reset["profile_memory_outside"][0] == pytest.approx(0.0)


def test_duplicate_rider_memory_update_is_order_invariant():
    rider = {"ProfileID": "repeat", "PriceThreshold": 2.0}
    first = RiderProfileMemory(alpha=0.5)
    second = RiderProfileMemory(alpha=0.5)
    first.update(
        [rider, rider],
        choice_target=np.asarray([1.0, -0.5]),
        outside_target=np.asarray([-1.0, 0.5]),
    )
    second.update(
        [rider, rider],
        choice_target=np.asarray([-0.5, 1.0]),
        outside_target=np.asarray([0.5, -1.0]),
    )
    assert first.choice_state == second.choice_state
    assert first.outside_state == second.outside_state
    assert first.encounters == second.encounters == {"ProfileID:repeat": 2}
