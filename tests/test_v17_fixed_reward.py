"""Registered V17 fixed economic-reward and resume contracts."""

from types import SimpleNamespace

import numpy as np
import pytest

from Core import Core
from platform_mdp import LongTermProfitReward, LongTermProfitRewardConfig


def _config() -> LongTermProfitRewardConfig:
    return LongTermProfitRewardConfig(
        fixed_economic_reward_v16=True,
        stage_specific_rewards=True,
        learning_corrections=True,
        own_profit_scale=2.0,
        profit_advantage_scale=1.0,
    )


def _core(
    *, seed: int = 1781, reward: bool = True, own_scale: float = 2.0
) -> Core:
    return Core(
        "Chicago",
        seed=seed,
        firm1_mode="RL",
        firm2_mode="static",
        total_customers_pool=8,
        simulation_sample_cap=3,
        enable_driver_supply=False,
        openai_api_key="",
        v16_reward=reward,
        long_term_profit_scale=own_scale,
        long_term_profit_advantage_scale=1.0,
    )


@pytest.mark.parametrize(
    "stage", [None, "market", "static", "dynamic", "catchup", "profit"]
)
def test_fixed_reward_uses_same_registered_weights_in_every_stage(stage):
    assert _config().economic_weights(stage) == pytest.approx(
        {
            "own_profit": 0.70,
            "profit_lead": 0.30,
            "own_completed_share": 0.0,
            "completed_share_lead": 0.0,
        }
    )


def test_reward_uses_frozen_own_and_lead_scales_exactly():
    result = LongTermProfitReward(_config()).compute(
        training_regime="profit",
        own_profit_per_request=4.0,
        rival_profit_per_request=1.0,
        own_completed_share=0.05,
        rival_completed_share=0.95,
    )
    assert result["reward"] == pytest.approx(2.3)
    assert result["reward_own_profit_component"] == pytest.approx(1.4)
    assert result["reward_profit_advantage_component"] == pytest.approx(0.9)
    assert result["reward_market_share_competitiveness_component"] == 0.0
    assert result["reward_weight_own_profit"] == pytest.approx(0.70)
    assert result["reward_weight_profit_lead"] == pytest.approx(0.30)


def test_four_stored_primitives_exactly_reconstruct_fixed_reward():
    config = _config()
    components = np.asarray([4.0, 1.0, 0.05, 0.95])
    weights = np.asarray(config.critic_component_weights("market"))
    np.testing.assert_allclose(weights, [0.65, -0.30, 0.0, 0.0])
    assert float(components @ weights) == pytest.approx(2.3)

    core = SimpleNamespace(long_term_profit_reward_config=config)
    stored = Core._value_reward_components(
        core,
        SimpleNamespace(profit_per_request=4.0, completed_share=0.05),
        SimpleNamespace(profit_per_request=1.0, completed_share=0.95),
        {},
    )
    np.testing.assert_allclose(stored, components)


def test_fixed_reward_rejects_conflicting_legacy_objectives():
    with pytest.raises(ValueError, match="profit_maximization"):
        LongTermProfitRewardConfig(
            objective_mode="balanced",
            fixed_economic_reward_v16=True,
        )
    with pytest.raises(ValueError, match="legacy temporal reward"):
        LongTermProfitRewardConfig(
            fixed_economic_reward_v16=True,
            temporal_competitive_reward=True,
        )


def test_fixed_reward_flag_and_scales_are_resume_contracts(tmp_path):
    core = _core(seed=1782)
    artifact = tmp_path / "fixed-reward.pt"
    core.save_trained_model(path=str(artifact), model_id="fixed-reward")

    resumed = _core(seed=1782)
    resumed.load_trained_model(str(artifact), load_optimizer=True)
    assert resumed.long_term_profit_reward_model.mechanism_name == (
        "fixed_70_30_scaled_economic_v17"
    )

    with pytest.raises(ValueError, match="reward mechanism"):
        _core(seed=1782, reward=False).load_trained_model(
            str(artifact), load_optimizer=True
        )
    with pytest.raises(ValueError, match="own_profit_scale"):
        _core(seed=1782, own_scale=3.0).load_trained_model(
            str(artifact), load_optimizer=True
        )


def test_flags_off_reward_contract_is_unchanged():
    config = LongTermProfitRewardConfig()
    assert config.fixed_economic_reward_v16 is False
    assert config.economic_weights() == {
        "own_profit": 0.8,
        "profit_lead": 0.2,
        "own_completed_share": 0.0,
        "completed_share_lead": 0.0,
    }
