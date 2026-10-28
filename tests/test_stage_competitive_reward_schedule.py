import pytest

from Core import _parse_stage_profit_lead_weights
from platform_mdp import LongTermProfitRewardConfig, config_payload


REVISED_LEAD_SCHEDULE = (0.0, 0.55, 0.70, 0.80)


def revised_config() -> LongTermProfitRewardConfig:
    return LongTermProfitRewardConfig(
        objective_mode="profit_maximization",
        stage_specific_rewards=True,
        four_stage_curriculum_rewards=True,
        learning_corrections=True,
        opponent_profit_lead_weight=0.60,
        stage_profit_lead_weights=REVISED_LEAD_SCHEDULE,
        market_share_competitiveness_weight=0.0,
        temporal_competitive_reward=False,
    )


def test_revised_schedule_moves_reward_from_own_profit_to_profit_lead():
    config = revised_config()
    for stage, expected_lead in zip(
        ("market", "static", "catchup", "profit"),
        REVISED_LEAD_SCHEDULE,
    ):
        weights = config.economic_weights(stage)
        assert weights == pytest.approx({
            "own_profit": 1.0 - expected_lead,
            "profit_lead": expected_lead,
            "own_completed_share": 0.0,
            "completed_share_lead": 0.0,
        })
        # (1-w)*own + w*(own-rival) expands to own - w*rival:
        # Firm 1's own contribution profit always retains unit coefficient.
        assert weights["own_profit"] + weights["profit_lead"] == pytest.approx(1.0)


def test_revised_schedule_is_checkpointed_in_active_reward_payload():
    from platform_mdp import (
        ConstraintConfig,
        ObservationConfig,
        PositiveRewardConfig,
        TrainingStageScheduler,
    )

    payload = config_payload(
        ObservationConfig(),
        PositiveRewardConfig(),
        ConstraintConfig(),
        TrainingStageScheduler(),
        revised_config(),
    )
    active = payload["active_reward"]
    assert tuple(active["stage_profit_lead_weights"]) == REVISED_LEAD_SCHEDULE
    assert active["stage_economic_weights"]["profit"]["profit_lead"] == pytest.approx(0.80)


def test_stage_schedule_cli_and_validation_are_fail_closed():
    assert _parse_stage_profit_lead_weights("0,0.55,0.70,0.80") == REVISED_LEAD_SCHEDULE
    with pytest.raises(ValueError, match="four-stage"):
        LongTermProfitRewardConfig(
            stage_profit_lead_weights=REVISED_LEAD_SCHEDULE,
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        LongTermProfitRewardConfig(
            stage_specific_rewards=True,
            four_stage_curriculum_rewards=True,
            stage_profit_lead_weights=(0.0, 0.5, 0.7, 1.1),
        )
