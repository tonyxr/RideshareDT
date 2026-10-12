import pytest

from platform_mdp import (
    EpisodeProfitDominanceTracker,
    LongTermProfitRewardConfig,
    PolicyObjectiveReward,
)


def _reward():
    config = LongTermProfitRewardConfig(
        objective_mode="profit_dominance_v30",
        own_profit_scale=2.0,
        profit_advantage_scale=1.0,
        terminal_profit_dominance_weight=1.5,
    )
    return config, PolicyObjectiveReward(config)


def test_dense_reward_is_exactly_the_reported_profit_lead():
    _, reward = _reward()
    result = reward.compute(
        own_profit_per_request=4.0, rival_profit_per_request=1.0,
        own_completed_share=0.01, rival_completed_share=0.99,
    )
    assert result["reward"] == pytest.approx(4 - 1)
    assert result["reward_own_profit_component"] == 0
    assert result["reward_market_share_competitiveness_component"] == 0
    assert result["reward_applied_action_regularization"] == 0
    assert result["reward_active_component_count"] == 1


@pytest.mark.parametrize("own,rival", [(1.23, 1.14), (1.40, 1.59),
                                         (-5, -2), (-2, -5), (1000, -1000)])
def test_reward_sign_and_magnitude_match_profit_lead(own, rival):
    _, reward = _reward()
    assert reward.compute(own_profit_per_request=own,
                          rival_profit_per_request=rival)["reward"] == pytest.approx(own-rival)


def test_observed_more_profitable_but_losing_policy_is_not_rewarded_more():
    _, reward = _reward()
    def score(own, rival):
        return reward.compute(own_profit_per_request=own,
                              rival_profit_per_request=rival)["reward"]
    assert score(1.23, 1.14) > score(1.40, 1.59)
    assert score(0.65360, 0.33569) > score(1.05644, 1.12657)
    assert score(1001, 1002) == score(1, 2)  # shared market growth isn't dominance


def test_share_and_quote_diagnostics_cannot_change_v30_credit():
    _, reward = _reward()
    common = dict(own_profit_per_request=2.0, rival_profit_per_request=1.0)
    losing_share = reward.compute(
        **common, own_completed_share=0.0, rival_completed_share=1.0,
        price_gap_f2_minus_f1=-100,
    )["reward"]
    winning_share = reward.compute(
        **common, own_completed_share=1.0, rival_completed_share=0.0,
        price_gap_f2_minus_f1=100,
    )["reward"]
    assert losing_share == winning_share


def test_more_own_profit_and_more_profit_lead_each_improve_reward():
    _, reward = _reward()
    baseline = reward.compute(
        own_profit_per_request=2, rival_profit_per_request=1,
    )["reward"]
    assert reward.compute(
        own_profit_per_request=3, rival_profit_per_request=1,
    )["reward"] > baseline
    assert reward.compute(
        own_profit_per_request=2, rival_profit_per_request=0,
    )["reward"] > baseline


def test_terminal_reward_is_horizon_scaled_final_quarter_lead_and_resets():
    config, _ = _reward()
    tracker = EpisodeProfitDominanceTracker(config, horizon_steps=2)
    tracker.record(own_profit_per_request=4, rival_profit_per_request=1)
    tracker.record(own_profit_per_request=0, rival_profit_per_request=1)
    terminal = tracker.terminal()
    assert terminal["reward_terminal_profit_advantage_mean"] == pytest.approx(-1.0)
    assert terminal["reward_terminal"] == pytest.approx(-3.0)
    assert terminal["reward_terminal_observations"] == 2
    assert tracker.count == 0
    assert tracker.terminal()["reward_terminal"] == 0


def test_terminal_tracker_snapshot_restores_in_progress_episode_exactly():
    config, _ = _reward()
    tracker = EpisodeProfitDominanceTracker(config, horizon_steps=2)
    tracker.record(own_profit_per_request=4, rival_profit_per_request=1)
    tracker.record(own_profit_per_request=2, rival_profit_per_request=3)
    snapshot = tracker.snapshot()
    tracker.reset()
    tracker.record(own_profit_per_request=-100, rival_profit_per_request=100)
    tracker.restore(snapshot)
    terminal = tracker.terminal()
    assert terminal["reward_terminal_observations"] == 2
    assert terminal["reward_terminal_profit_advantage_mean"] == pytest.approx(-1.0)
    assert terminal["reward_terminal"] == pytest.approx(-3.0)


def test_terminal_tracker_rejects_invalid_snapshot():
    config, _ = _reward()
    tracker = EpisodeProfitDominanceTracker(config)
    with pytest.raises(ValueError, match="invalid episode"):
        tracker.restore({
            "tail_profit_lead_sum": float("nan"),
            "tail_count": 0,
            "horizon_steps": 2,
            "count": 1,
        })


def test_v30_reward_and_terminal_bonus_are_unclipped():
    config, reward = _reward()
    assert reward.compute(
        own_profit_per_request=1000, rival_profit_per_request=-1000,
    )["reward"] > 100
    tracker = EpisodeProfitDominanceTracker(config)
    tracker.record(own_profit_per_request=-1000, rival_profit_per_request=1000)
    assert tracker.terminal()["reward_terminal"] < -100
