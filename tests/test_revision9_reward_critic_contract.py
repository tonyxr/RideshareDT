"""Regression tests for the revision-9 four-outcome reward/critic contract."""

import numpy as np
import pytest

from platform_mdp import LongTermProfitReward, LongTermProfitRewardConfig
from PPOAgent import PPOAgent


def _config() -> LongTermProfitRewardConfig:
    return LongTermProfitRewardConfig(
        objective_mode="profit_maximization",
        stage_specific_rewards=True,
        learning_corrections=True,
        opponent_profit_lead_weight=0.60,
        own_profit_scale=2.0,
        profit_advantage_scale=9.0,
        market_share_competitiveness_weight=0.05,
        market_share_gap_scale=0.10,
    )


def test_share_term_is_revision_scoped_and_flags_off_reward_is_unchanged():
    legacy = LongTermProfitRewardConfig(share_dominance_reward=False)
    corrected = LongTermProfitRewardConfig(learning_corrections=True)
    assert legacy.economic_weights() == {
        "own_profit": 0.8,
        "profit_lead": 0.2,
        "own_completed_share": 0.0,
        "completed_share_lead": 0.0,
    }
    assert corrected.economic_weights() == pytest.approx({
        "own_profit": 0.80,
        "profit_lead": 0.20,
        "own_completed_share": 0.0,
        "completed_share_lead": 0.0,
    })


@pytest.mark.parametrize("stage", ["market", "static", "dynamic", None])
def test_four_raw_outcomes_exactly_reconstruct_every_stage_reward(stage):
    config = _config()
    reward = LongTermProfitReward(config).compute(
        training_regime=stage,
        own_profit_per_request=1.35,
        rival_profit_per_request=0.85,
        own_completed_share=0.57,
        rival_completed_share=0.31,
    )["reward"]
    components = np.asarray([1.35, 0.85, 0.57, 0.31])
    reconstructed = components @ np.asarray(
        config.critic_component_weights(stage)
    )
    assert reconstructed == pytest.approx(reward, abs=1e-12)


def _agent() -> PPOAgent:
    return PPOAgent(
        6,
        3,
        hidden_dim=16,
        single_state_dim=2,
        frame_stack=3,
        persistent_recurrence=True,
        privileged_critic=True,
        privileged_value_only=True,
        critic_state_dim=2,
        gamma=0.99,
        learning_corrections=True,
        td_value_targets=True,
        learner_revision=4,
        value_component_dim=4,
        device="cpu",
    )


@pytest.mark.parametrize("stage", ["market", "static", "dynamic"])
def test_ppo_accepts_and_updates_exact_four_outcome_transition(stage):
    config = _config()
    model = _agent()
    model.set_value_objective_weights(config.critic_component_weights(stage))
    components = np.asarray([1.35, 0.85, 0.57, 0.31], dtype=np.float32)
    reward = float(components @ model.value_weights.cpu().numpy())
    action, state, logp, value, *_ = model.act(
        np.zeros(6, dtype=np.float32),
        critic_state=np.zeros(2, dtype=np.float32),
    )
    model.store(
        state,
        action,
        reward,
        True,
        None,
        logp,
        value,
        duration=1,
        reward_components=components,
    )
    metrics = model.update(epochs=1, batch_size=1)
    assert metrics["optimizer_steps"] > 0
    for name in ("own_profit", "rival_profit", "own_share", "rival_share"):
        assert np.isfinite(metrics[f"critic_{name}_rmse"])


def test_four_outcome_critic_rejects_truncated_two_component_transition():
    config = _config()
    model = _agent()
    model.set_value_objective_weights(config.critic_component_weights("market"))
    action, state, logp, value, *_ = model.act(
        np.zeros(6, dtype=np.float32),
        critic_state=np.zeros(2, dtype=np.float32),
    )
    with pytest.raises(ValueError, match="component count"):
        model.store(
            state,
            action,
            0.0,
            True,
            None,
            logp,
            value,
            reward_components=[0.0, 0.0],
        )


def test_value_objective_checkpoint_roundtrip_preserves_all_four_weights():
    config = _config()
    first, second = _agent(), _agent()
    first.set_value_objective_weights(config.critic_component_weights("dynamic"))
    second.load_normalizer_state_dict(first.normalizer_state_dict())
    np.testing.assert_allclose(
        second.value_weights.cpu().numpy(), first.value_weights.cpu().numpy()
    )
