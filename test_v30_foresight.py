"""Episode accounting, information separation, and delayed-credit learning."""
import numpy as np
import pytest
import torch

from platform_mdp import EpisodeProfitDominanceTracker, LongTermProfitRewardConfig
from PPOAgent import PPOAgent
from test_v30_core_contract import _core


def _tracker(horizon):
    return EpisodeProfitDominanceTracker(
        LongTermProfitRewardConfig(objective_mode="profit_dominance_v30"), horizon
    )


def test_early_wins_cannot_hide_late_losses_in_terminal_objective():
    tracker = _tracker(8)
    for lead in [10.] * 6 + [-1., -3.]:
        tracker.record(own_profit_per_request=lead, rival_profit_per_request=0.)
    result = tracker.terminal()
    assert result["reward_terminal"] == -16.
    assert result["reward_terminal_tail_observations"] == 2
    assert tracker.terminal()["reward_terminal"] == 0


@pytest.mark.parametrize("horizon", [1, 3, 8, 1200])
def test_terminal_units_scale_with_dense_horizon_not_an_invisible_mean(horizon):
    tracker = _tracker(horizon)
    for _ in range(horizon):
        tracker.record(own_profit_per_request=3., rival_profit_per_request=1.)
    assert tracker.terminal()["reward_terminal"] == 2. * horizon


def test_partial_episode_cannot_be_misreported_as_complete():
    tracker = _tracker(8)
    tracker.record(own_profit_per_request=3., rival_profit_per_request=1.)
    with pytest.raises(ValueError, match="completed planned episode"):
        tracker.terminal()


def test_clock_and_terminal_history_are_not_aliased_or_leaked_to_actor():
    core = _core(shock_observations=True)
    tracker = core.v30_episode_reward_tracker
    tracker.reset(horizon_steps=8)
    before = core._build_rl_state(day_of_week=1, hour=8, weather="clear")
    for _ in range(7):
        tracker.record(own_profit_per_request=2., rival_profit_per_request=1.)
    after = core._build_rl_state(day_of_week=1, hour=8, weather="clear")
    assert before[-2] == 1. and after[-2] == .125
    np.testing.assert_array_equal(before[:-2], after[:-2])
    context = core._build_privileged_critic_state(after)
    assert context.shape == (1,) and context[0] > 0
    assert core.firm1.agent.critic_state_dim == 1
    snapshot = tracker.snapshot()
    tracker.reset(horizon_steps=8)
    for _ in range(7):
        tracker.record(own_profit_per_request=-2., rival_profit_per_request=1.)
    np.testing.assert_array_equal(core._build_rl_state(day_of_week=1, hour=8, weather="clear"), after)
    assert core._build_privileged_critic_state(after)[0] < 0
    tracker.restore(snapshot)
    np.testing.assert_array_equal(tracker.critic_context(), context)


@pytest.mark.parametrize("seed", [17, 31, 83])
def test_ppo_learns_delayed_lead_despite_opposite_immediate_payoff(seed):
    torch.manual_seed(seed)
    agent = PPOAgent(state_dim=3, action_dim=2, hidden_dim=32,
                     gamma=.99, lam=1., lr=.003, center_rewards=False,
                     normalize_values=True, device="cpu")
    # Only the opening action determines a four-step commitment. All later
    # actions remain sampled but have no effect, requiring temporal credit.
    for update in range(90):
        for episode in range(8):
            tracker = _tracker(4)
            commitment = 0
            for step in range(4):
                state_vector = [(4-step)/4, np.log1p(4), commitment if step else 0]
                action, state, logp, value, features, mask = agent.act(state_vector)
                if step == 0:
                    commitment = action
                lead = (-.1 if commitment else .5) if step < 3 else (1. if commitment else -1.)
                tracker.record(own_profit_per_request=lead, rival_profit_per_request=0.)
                reward = lead
                if step == 3:
                    reward += .99 * tracker.terminal()["reward_terminal"]
                agent.store(state, action, reward, step == 3, None, logp, value,
                            action_features=features, action_mask=mask)
        agent.update(epochs=4, batch_size=32, bootstrap_value=0.)
    with torch.no_grad():
        logits, _ = agent.net.actor_forward(torch.tensor([[1., np.log1p(4), 0.]], dtype=torch.float32))
        farsighted_probability = torch.softmax(logits, -1)[0, 1].item()
    assert farsighted_probability > .9, (seed, farsighted_probability)
