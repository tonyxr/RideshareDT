"""Independent coefficient-choice and magnitude action contract."""
import numpy as np
import pytest
import torch

from MarketInteraction import MarketInteraction
from pricing_models import FirmRLPricer


KEYS = ["base_fare", "per_minute", "per_mile", "booking_fee", "airport_fee"]


def action_for(pricer, key, multiplier):
    return next(
        action
        for action, target in pricer.action_to_target_key.items()
        if target == key
        and pricer.action_to_multiplier[action] == pytest.approx(multiplier)
    )


def test_action_space_is_complete_single_coefficient_product():
    pricer = FirmRLPricer(
        seed=1,
        opt_keys=KEYS,
        state_frame_stack=2,
        single_state_dim=24,
        action_feature_dim=20,
        tariff_targets=(0.8, 1.0, 1.2),
    )
    assert len(pricer.action_to_steps) == 1 + len(KEYS) * 3
    assert pricer.action_label(0) == "hold"
    for action in range(1, len(pricer.action_to_steps)):
        mapping = pricer.action_to_steps[action]
        assert sum(value != 0 for value in mapping.values()) <= 1
        assert pricer.action_to_target_key[action] in KEYS
        assert pricer.action_to_multiplier[action] in (0.8, 1.0, 1.2)


def test_factorized_prior_is_uniform_over_hold_or_coefficient_then_magnitude():
    pricer = FirmRLPricer(
        seed=2,
        opt_keys=KEYS,
        state_frame_stack=2,
        single_state_dim=24,
        action_feature_dim=20,
        tariff_targets=(0.8, 1.0, 1.2),
    )
    logits, _ = pricer.agent.net.actor_forward(torch.zeros(1, 48))
    probabilities = torch.softmax(logits, dim=-1).detach().numpy()[0]
    assert probabilities[0] == pytest.approx(1.0 / 6.0)
    for key in KEYS:
        actions = [
            action
            for action, target in pricer.action_to_target_key.items()
            if target == key
        ]
        assert probabilities[actions].sum() == pytest.approx(1.0 / 6.0)
        np.testing.assert_allclose(
            probabilities[actions], np.full(3, 1.0 / 18.0), atol=1e-7
        )


def test_selected_action_credits_its_target_and_only_its_magnitude_row():
    pricer = FirmRLPricer(
        seed=21,
        opt_keys=KEYS,
        state_frame_stack=2,
        single_state_dim=24,
        action_feature_dim=20,
        tariff_targets=(0.8, 1.0, 1.2),
    )
    action = action_for(pricer, "per_minute", 1.2)
    logits, _ = pricer.agent.net.actor_forward(torch.zeros(1, 48))
    loss = -torch.log_softmax(logits, dim=-1)[0, action]
    loss.backward()

    target_head = pricer.agent.net.pi_target_head
    magnitude_head = pricer.agent.net.pi_magnitude_head
    assert target_head is not None and magnitude_head is not None
    assert torch.count_nonzero(target_head.bias.grad) > 0
    magnitude_grad = magnitude_head.bias.grad.reshape(len(KEYS), 3)
    selected = KEYS.index("per_minute")
    assert float(magnitude_grad[selected].abs().max()) > 0.1
    unrelated = torch.cat(
        (magnitude_grad[:selected], magnitude_grad[selected + 1 :])
    )
    assert float(unrelated.abs().max()) < 1e-7


def test_one_action_changes_only_its_selected_coefficient():
    pricer = FirmRLPricer(
        seed=3,
        opt_keys=KEYS,
        tariff_targets=(0.8, 1.0, 1.2),
    )
    market = MarketInteraction(city_name="Chicago", seed=4)
    anchor = market.curr_market
    before = {key: float(getattr(anchor, key)) for key in KEYS}

    action = action_for(pricer, "per_mile", 1.2)
    pricer.apply_action(action, market)
    assert pricer.overrides.per_mile == pytest.approx(
        np.clip(1.2 * anchor.per_mile, *pricer.config.bounds["per_mile"])
    )
    for key in set(KEYS) - {"per_mile"}:
        assert getattr(pricer.overrides, key) is None
        assert before[key] == pytest.approx(getattr(anchor, key))
    assert pricer.last_action_descriptor.target == "per_mile"

    posted_per_mile = pricer.overrides.per_mile
    pricer.apply_action(action_for(pricer, "base_fare", 0.8), market)
    assert pricer.overrides.per_mile == posted_per_mile
    assert pricer.overrides.base_fare == pytest.approx(
        np.clip(0.8 * anchor.base_fare, *pricer.config.bounds["base_fare"])
    )


def test_nondefault_driver_incentive_factor_is_rejected():
    with pytest.raises(ValueError, match="driver_incentive_levels"):
        FirmRLPricer(
            seed=5,
            opt_keys=KEYS,
            incentive_levels=(1.0, 1.1),
        )


def test_flags_off_path_preserves_whole_tariff_action_contract():
    pricer = FirmRLPricer(
        seed=6,
        opt_keys=KEYS,
        tariff_targets=(0.8, 1.0, 1.2),
        independent_coefficient_actions=False,
    )
    assert len(pricer.action_to_steps) == 4
    assert pricer.action_to_target_key == {}
    assert pricer.action_label(3) == "tariff_target:1.200x_market"
    market = MarketInteraction(city_name="Chicago", seed=7)
    pricer.apply_action(3, market)
    for key in KEYS:
        assert getattr(pricer.overrides, key) == pytest.approx(
            np.clip(
                1.2 * float(getattr(market.curr_market, key)),
                *pricer.config.bounds[key],
            )
        )
    assert pricer.last_action_descriptor.target == "whole_tariff"
