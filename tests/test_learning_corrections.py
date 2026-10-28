"""Regression tests for the isolated learner correction, not performance gates."""
import numpy as np
import pytest
import torch

from PPOAgent import PPOAgent
from platform_mdp import TrainingStageScheduler
from platform_mdp import ObservationConfig, PlatformObservationModel


def test_stage_cosine_decay_and_restart_without_optimizer_reset():
    scheduler = TrainingStageScheduler(stage_fractions=(.1, .108, .792))
    agent = PPOAgent(4, 3, hidden_dim=8)
    optimizer = agent.actor_opt
    for first, stop in ((0, 2000), (2000, 4160), (4160, 20000)):
        rates = []
        for day in (first, (first + stop - 1) // 2, stop - 1):
            agent.set_stage_progress(scheduler.stage_progress(day, 20000))
            rates.append(agent.curr_lr)
            assert agent.critic_opt.param_groups[0]['lr'] == pytest.approx(3 * agent.curr_lr)
        assert rates[0] == pytest.approx(3e-4)
        assert rates[-1] == pytest.approx(5e-5)
        assert rates[0] > rates[1] > rates[2]
        assert agent.actor_opt is optimizer
    assert scheduler.stage_progress(4220, 20000) > scheduler.stage_progress(4219, 20000)


def test_stage_progress_small_nonintegral_and_direct_budgets():
    for scheduler in (TrainingStageScheduler(), TrainingStageScheduler(mode='direct')):
        for budget in (1, 3, 7, 192):
            for day in range(budget):
                assert 0 <= scheduler.stage_progress(day, budget) <= 1
    with pytest.raises(ValueError):
        TrainingStageScheduler().stage_progress(10, 10)


def recurrent_agent(**kwargs):
    torch.manual_seed(129)
    kwargs.setdefault('learner_revision', 2)
    agent = PPOAgent(6, 3, hidden_dim=8, single_state_dim=2, frame_stack=3,
                     persistent_recurrence=True, privileged_critic=True,
                     privileged_value_only=True, critic_state_dim=2,
                     learning_corrections=True, td_value_targets=True, **kwargs)
    # A zero initial policy head blocks encoder gradients on the first update.
    with torch.no_grad():
        agent.net.pi_head.weight.normal_(0, .1)
    return agent


def collect_episode(agent, count=4):
    for step in range(count):
        state = np.arange(6, dtype=np.float32) / 10 + step / 8
        if step:
            agent.observe_recurrent_state(state / 2, np.ones(2, dtype=np.float32))
        action, state_t, logp, value, features, mask = agent.act(state, critic_state=np.ones(2))
        agent.store(state_t, action, float(step == count - 1), step == count - 1,
                    None, logp, value, action_features=features, action_mask=mask,
                    critic_state=np.ones(2), next_value=0.)


def replay(agent, transitions):
    return agent._actor_logits_batch(torch.stack([t.state for t in transitions]),
        [t.actor_hidden for t in transitions], [(t.actor_history, t.actor_history_length) for t in transitions])


def test_recurrent_replay_exact_behavior_including_intervening_frames_and_commitment():
    agent = recurrent_agent()
    agent.exploration_commit_mean = 3.
    collect_episode(agent)
    transitions = agent.buf[:]
    logits = replay(agent, transitions)
    dist = agent._behavior_distribution(logits,
        torch.tensor([t.exploration_rate for t in transitions]),
        torch.stack([t.action_mask for t in transitions]),
        torch.tensor([t.commit_prob for t in transitions]),
        torch.tensor([t.commit_action for t in transitions]))
    logp = dist.log_prob(torch.stack([t.action for t in transitions]))
    assert torch.allclose(logp, torch.stack([t.old_logp for t in transitions]), atol=2e-6)
    assert transitions[-1].actor_history_length == 9
    assert len({id(t.actor_history) for t in transitions}) == 1
    # Nondecision frames and earlier decisions both receive policy gradients.
    history = transitions[-1].actor_history
    history[0].requires_grad_()
    history[-2].requires_grad_()
    replay(agent, transitions)[-1, 0].backward()
    assert history[0].grad.abs().sum() > 0
    assert history[-2].grad.abs().sum() > 0


def test_recurrent_replay_episode_isolation_current_weights_and_update_refresh():
    agent = recurrent_agent()
    collect_episode(agent)
    first = agent.buf[:]
    before = replay(agent, first).detach()
    agent.reset_recurrent_state()
    collect_episode(agent, 2)
    second = agent.buf[len(first):]
    assert first[0].actor_history is not second[0].actor_history
    combined = replay(agent, first + second)
    assert torch.allclose(combined[:len(first)], before, atol=1e-7)
    snapshot = agent.recurrent_state_dict()
    agent.observe_recurrent_state(np.zeros(6), np.ones(2))
    agent.load_recurrent_state_dict(snapshot)
    assert torch.equal(agent.recurrent_state_dict()['actor_history'], snapshot['actor_history'])
    with torch.no_grad():
        agent.net.actor_recurrent.weight_ih_l0.add_(.03)
    assert not torch.allclose(replay(agent, first), before, atol=1e-7)
    history_length = len(agent._actor_history)
    metrics = agent.update(epochs=2, batch_size=3)
    assert metrics['actor_optimizer_steps'] > 0
    assert len(agent._actor_history) == history_length
    frames = torch.stack(agent._actor_history).unsqueeze(0)
    with torch.no_grad():
        _, expected = agent.net.actor_recurrent(agent.net.actor_frame_encoder(frames))
    assert torch.allclose(expected, agent._actor_hidden, atol=1e-7)


def test_critic_gets_delayed_rewards_and_requested_actor_epochs():
    agent = recurrent_agent(gamma=.9, lam=1., reward_rate_beta=1.)
    with torch.no_grad():
        for parameter in agent.net.critic_parameters():
            parameter.zero_()
    collect_episode(agent)
    expected = torch.tensor([.9**3, .9**2, .9, 1.])
    initial = agent._critic_value_loss(torch.zeros(4), expected).item()
    metrics = agent.update(epochs=8, batch_size=4)
    assert metrics['critic_target_contract'] == 'discounted_reward_gae_lambda_return'
    assert metrics['critic_target_mean'] == pytest.approx(expected.mean().item())
    assert metrics['actor_optimizer_steps'] == metrics['critic_optimizer_steps'] == 8
    assert agent.reward_rate_by_regime == {}
    with torch.no_grad():
        values, _, _ = agent.net.critic_forward(torch.zeros(4, 6), torch.ones(4, 2))
    assert agent._critic_value_loss(values, expected).item() < initial


def test_observation_distinguishes_high_tariffs_profit_and_quote_gaps():
    anchor = dict(base_fare=3., per_minute=.3, per_mile=1.2, booking_fee=2., airport_fee=0.)
    model = PlatformObservationModel(4, ObservationConfig(learning_corrections=True,
                                    regime_features=True, persistence_features=True))
    frames = [model.build_observation(hour=9, day_of_week=1, weather='clear',
               own_coefficients={k: m*v for k, v in anchor.items()}, anchor_coefficients=anchor)
              for m in (1.5, 1.6, 1.8, 2.)]
    assert all(frame.shape == (34,) for frame in frames)
    assert np.all(np.diff(np.array(frames)[:, 4:8], axis=0) > 0)
    for value in (20., 40.):
        model.telemetry['profit_per_request'] = value
        for probe in model.quote_probes.values():
            probe['gap'] = value
        frame = model.build_observation(hour=9, day_of_week=1, weather='clear',
            own_coefficients=dict(anchor, airport_fee=value), anchor_coefficients=anchor)
        assert np.isfinite(frame).all()
        if value == 20.:
            previous = frame
        else:
            assert frame[8] > previous[8] and frame[10] > previous[10] and frame[18] > previous[18]


def test_public_rival_quote_index_is_not_confounded_by_own_tariff():
    from Core import Core
    from coeff_utils import set_coeff
    core = Core(market_name='Chicago', seed=5, firm1_mode='RL', firm2_mode='static',
                credit_assignment_v25=True, stage_specific_rewards=True,
                learning_corrections=True, persistence_features=True, quote_noise_dollars=0.)
    indices = []
    for own in (1.1, 1.8):
        core._rival_gap_history.clear()
        set_coeff(core.market.curr_market, core.firm1.overrides, 'base_fare', own * 3.)
        core._publish_firm2_leader_quotes(day_of_week=1, hour=10, weather='clear')
        indices.append(core._persistence_public_features()['rival_quote_index_7d'])
    assert indices[0] == pytest.approx(indices[1], abs=1e-7)
    set_coeff(core.market.curr_market, core.firm2.overrides, 'base_fare', 20.)
    core._rival_gap_history.clear()
    core._publish_firm2_leader_quotes(day_of_week=1, hour=10, weather='clear')
    assert core._persistence_public_features()['rival_quote_index_7d'] > indices[0]


def test_reward_alignment_equal_weights_common_units_and_no_penalties():
    from platform_mdp import LongTermProfitReward, LongTermProfitRewardConfig
    reward = LongTermProfitReward(LongTermProfitRewardConfig(stage_specific_rewards=True,
                                 learning_corrections=True, own_profit_scale=2.))
    def score(own, rival, regime=None, **kwargs):
        kwargs.setdefault('own_completed_share', .4)
        kwargs.setdefault('rival_completed_share', .4)
        return reward.compute(own_profit_per_request=own, rival_profit_per_request=rival,
                              training_regime=regime, **kwargs)
    assert score(2., 10., 'market')['reward'] == pytest.approx(.95)
    for regime in ('static', 'dynamic', None):
        for own, rival in ((2., 1.), (1., 2.), (-2., -1.), (0., 0.), (5., 100.)):
            result = score(own, rival, regime)
            lead = own - rival
            expected = (.475 * own + .475 * lead) / 2.
            assert result['reward'] == pytest.approx(expected)
            assert result['reward'] == pytest.approx(result['reward_own_profit_component']
                   + result['reward_profit_advantage_component']
                   + result['reward_market_share_advantage_component'])
            assert 'reward_profit_deficit_component' not in result
            assert result['reward_active_component_count'] == 3
            assert result['reward_weight_own_profit'] == result['reward_weight_profit_lead'] == .475
            assert result['reward_weight_completed_share_lead'] == .05
        assert score(2., 1., regime)['reward'] > score(1., 1., regime)['reward']
        assert score(1., 2., regime)['reward'] < score(1., 1., regime)['reward']
        # Lead has the same marginal value on both sides of parity.
        assert score(1., 1., regime)['reward'] - score(1., 2., regime)['reward'] == pytest.approx(
            score(1., 0., regime)['reward'] - score(1., 1., regime)['reward'])
        # Increasing P with L fixed and increasing L with P fixed are equally valued.
        assert score(2., 2., regime)['reward'] - score(1., 1., regime)['reward'] == pytest.approx(
            score(1., 0., regime)['reward'] - score(1., 1., regime)['reward'])
    assert score(2., 10., 'market')['reward_active_component_count'] == 2
    assert reward.mechanism_name == 'staged_profit_with_share_dominance_v27_corrected'
    assert score(1., 1., own_completed_share=.45, rival_completed_share=.35)['reward'] \
        > score(1., 1., own_completed_share=.35, rival_completed_share=.45)['reward']
    assert score(2., 1., intervention_magnitude=10., reversal=True)['reward'] == score(2., 1.)['reward']


def test_transaction_preserves_pending_decision_and_exploration_commitment():
    agent = recurrent_agent()
    agent.exploration_commit_mean = 3.
    act = agent.act(np.ones(6), critic_state=np.ones(2))
    snapshot = agent.recurrent_state_dict()
    agent.reset_recurrent_state()
    agent.act(np.zeros(6), deterministic=True, critic_state=np.zeros(2))
    agent.load_recurrent_state_dict(snapshot)
    action, state, logp, value, features, mask = act
    agent.store(state, action, 1., True, None, logp, value, action_features=features,
                action_mask=mask, critic_state=np.ones(2))
    transition = agent.buf[-1]
    dist = agent._behavior_distribution(replay(agent, [transition]), transition.exploration_rate,
             transition.action_mask.unsqueeze(0), transition.commit_prob, transition.commit_action)
    assert dist.log_prob(transition.action).item() == pytest.approx(logp.item(), abs=2e-6)


def test_corrected_critic_bootstraps_truncation_without_cross_episode_rewards():
    agent = recurrent_agent(gamma=.9, lam=1.)
    advantages, returns = agent._gae_scalar(torch.tensor([1., 100.]), torch.zeros(2),
        torch.tensor([0., 1.]), discounts=torch.tensor([.9, .9]),
        truncations=torch.tensor([1., 0.]), next_value_overrides=torch.tensor([10., 0.]),
        normalize_advantage=False)
    assert torch.allclose(returns, torch.tensor([10., 100.]))


def test_reported_firm_scores_match_equal_weight_linear_reward():
    from Core import Core
    from platform_mdp import LongTermProfitRewardConfig
    core = Core.__new__(Core)
    core.long_term_profit_reward_config = LongTermProfitRewardConfig(stage_specific_rewards=True,
        learning_corrections=True, own_profit_scale=2., profit_advantage_scale=2.)
    own, rival = core._objective_firm_scores([2., 0.], [1., 1.], [0., 0.], [0., 0.])
    np.testing.assert_allclose(own, [.7125, -.2375])
    np.testing.assert_allclose(rival, [0., .475])
    # Linearity: no extra penalty for crossing below parity.
    of_mean, _ = core._objective_firm_scores([1.], [1.], [0.], [0.])
    assert own.mean() == pytest.approx(of_mean[0])
