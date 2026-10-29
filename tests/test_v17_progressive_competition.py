"""Manual-aligned V17 curriculum, reward, and private supply observations."""

import numpy as np
import pytest

from Core import Core
from driver_supply import FirmDriverBatchState
from platform_mdp import (
    TrainingStageScheduler,
    V17_EXECUTION_STAGE_FRACTIONS,
    V17_PROGRESSIVE_PROFIT_LEAD_WEIGHTS,
)


def _core(*, curriculum: str = "v17_progressive") -> Core:
    return Core(
        "Chicago",
        seed=1791,
        firm1_mode="RL",
        firm2_mode="static",
        total_customers_pool=8,
        simulation_sample_cap=3,
        enable_driver_supply=True,
        openai_api_key="",
        training_curriculum=curriculum,
    )


def test_v17_progressive_uses_manual_four_stage_execution_sequence():
    scheduler = TrainingStageScheduler(
        "v17_progressive",
        # The mode owns the frozen execution allocation; arbitrary caller
        # fractions cannot silently replace the manual's sequence.
        stage_fractions=(0.20, 0.20, 0.60),
    )
    assert tuple(
        stage.end - stage.start for stage in scheduler.stages
    ) == pytest.approx(V17_EXECUTION_STAGE_FRACTIONS, abs=2e-6)
    assert [stage.name for stage in scheduler.stages] == [
        "market",
        "static_opponent",
        "price_gap_catchup",
        "profit_maximizers",
    ]
    assert [stage.opponent_regime for stage in scheduler.stages] == [
        "market",
        "static",
        "catchup",
        "profit",
    ]
    assert [stage.opponent_pool_size for stage in scheduler.stages] == [
        1, 1, 4, 8
    ]
    assert scheduler.stage_at(0.049999).opponent_regime == "market"
    assert scheduler.stage_at(0.05).opponent_regime == "static"
    assert scheduler.stage_at(0.10).opponent_regime == "catchup"
    assert scheduler.stage_at(0.40).opponent_regime == "profit"


def test_progressive_mode_preserves_reward_endpoints_and_adds_catchup_step():
    core = _core()
    config = core.long_term_profit_reward_config
    assert config.stage_specific_rewards is True
    assert config.stage_profit_lead_weights == (
        V17_PROGRESSIVE_PROFIT_LEAD_WEIGHTS
    )
    assert len(core.training_catchup_opponent_modes) == 4
    assert len(core.training_profit_opponent_modes) == 8
    for regime, competitive_weight in zip(
        ("market", "static", "catchup", "profit"),
        V17_PROGRESSIVE_PROFIT_LEAD_WEIGHTS,
    ):
        weights = config.economic_weights(regime)
        assert weights == pytest.approx({
            "own_profit": 1.0 - competitive_weight,
            "profit_lead": competitive_weight,
            "own_completed_share": 0.0,
            "completed_share_lead": 0.0,
        })
        result = core.long_term_profit_reward_model.compute(
            training_regime=regime,
            own_profit_per_request=4.0,
            rival_profit_per_request=1.0,
            own_completed_share=0.2,
            rival_completed_share=0.8,
        )
        # One common dollar scale keeps own profit's coefficient invariant;
        # progression adds only a larger direct rival-profit penalty.
        assert result["reward"] == pytest.approx(
            (4.0 - competitive_weight * 1.0) / 2.0
        )


def test_progressive_mode_does_not_overwrite_fixed_v17_reward_implementation():
    with pytest.raises(ValueError, match="separate implementation"):
        Core(
            "Chicago",
            seed=1792,
            firm1_mode="RL",
            firm2_mode="static",
            total_customers_pool=8,
            simulation_sample_cap=3,
            enable_driver_supply=False,
            openai_api_key="",
            training_curriculum="v17_progressive",
            v16_reward=True,
        )


def test_each_firm_observes_exact_current_own_driver_count_only():
    core = _core()
    core.driver_supply.config.base_active_drivers = 400
    core.driver_supply.last_states["Firm1"] = FirmDriverBatchState(
        active_drivers=120
    )
    core.driver_supply.last_states["Firm2"] = FirmDriverBatchState(
        active_drivers=310
    )
    # Deliberately contradictory smoothed values prove that this channel is
    # live and unsmoothed.
    core.driver_supply.smoothed_states["Firm1"] = FirmDriverBatchState(
        active_drivers=7
    )
    core.driver_supply.smoothed_states["Firm2"] = FirmDriverBatchState(
        active_drivers=9
    )

    firm1 = core._build_rl_state(1, 12, "Clear", perspective="Firm1")
    firm2 = core._build_rl_state(1, 12, "Clear", perspective="Firm2")
    driver_slot = 24
    assert core.platform_observation_dim == 25
    assert firm1[driver_slot] == pytest.approx(np.arcsinh(120.0 / 400.0))
    assert firm2[driver_slot] == pytest.approx(np.arcsinh(310.0 / 400.0))

    # Changing only Firm 2's private fleet cannot alter Firm 1's driver slot.
    core.driver_supply.last_states["Firm2"] = FirmDriverBatchState(
        active_drivers=25
    )
    firm1_again = core._build_rl_state(1, 12, "Clear", perspective="Firm1")
    assert firm1_again[driver_slot] == firm1[driver_slot]

    context1, supply1 = core._driver_supply_context("Firm1", "Firm2")
    context2, supply2 = core._driver_supply_context("Firm2", "Firm1")
    assert context1["active_drivers"] == 120.0
    assert context2["active_drivers"] == 25.0
    assert supply1[0] == pytest.approx(np.arcsinh(120.0 / 400.0))
    assert supply2[0] == pytest.approx(np.arcsinh(25.0 / 400.0))


def test_flags_off_keep_the_legacy_observation_width():
    core = _core(curriculum="staged")
    assert core.observation_config.current_active_driver_observation is False
    assert core.platform_observation_dim == 24


def test_progressive_reward_and_driver_schema_round_trip(tmp_path):
    core = _core()
    artifact = tmp_path / "v17-progressive.pt"
    core.save_trained_model(path=str(artifact), model_id="v17-progressive")

    resumed = _core()
    resumed.load_trained_model(str(artifact), load_optimizer=True)
    assert resumed.platform_observation_dim == 25
    assert resumed.observation_config.current_active_driver_observation is True
    assert resumed.long_term_profit_reward_config.stage_profit_lead_weights == (
        V17_PROGRESSIVE_PROFIT_LEAD_WEIGHTS
    )
