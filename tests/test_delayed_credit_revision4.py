"""Delayed response/credit contracts; these are not city performance evidence."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from Core import Core
from PPOAgent import PPOAgent
from platform_mdp import TrainingStageScheduler
from pricing_models import FirmMemoryBestResponsePricer, FirmMetrics

ANCHORS = dict(city_base=2.5, city_pmin=.35, city_pmile=1.5, city_booking=2., city_airport=5.)


def memory_core(revision=4, seed=7, market_clock=True):
    core = object.__new__(Core)
    core.learner_revision = revision
    core.regime_market = market_clock
    core.shared_edit_keys = ['base_fare']
    core.firm2 = FirmMemoryBestResponsePricer(seed=seed, managed_keys=['base_fare'],
                                            operational_clock=market_clock)
    core.market = SimpleNamespace(curr_market=SimpleNamespace(base_fare=2.5,
        per_minute=.35, per_mile=1.5, booking_fee=2., airport_fee=5.))
    core._publish_firm2_leader_quotes = lambda **kw: None
    core._build_rl_state = lambda **kw: np.zeros(33, dtype=np.float32)
    core._action_feature_matrix_for_pricer = lambda *args: None
    core._driver_supply_context = lambda *args: ({}, None)
    core._observed_own_minus_competitor_gap = lambda *args: core.gap
    return core


def tick(core, step, gap, cadence=6):
    core.gap = gap
    return core._firm2_leader_step(previous_metrics=FirmMetrics() if step else None,
        day_of_week=1, hour=12, weather='clear', enabled=step % cadence == 0)


@pytest.mark.parametrize('cadence', [1, 3, 6])
@pytest.mark.parametrize('gap,onset,state', [(3., 6, 'retaliation'), (-3., 12, 'accommodation')])
def test_memory_response_uses_market_periods_not_decision_calls(cadence, gap, onset, state):
    core = memory_core()
    for step in range(onset):
        tick(core, step, gap, cadence)
        assert core.firm2.memory_state != state
    tick(core, onset, gap, cadence)
    assert core.firm2.memory_state == state
    assert core.firm2.undercut_steps == 0  # a completed retaliation trigger resets evidence
    assert core.firm2.premium_steps == (onset if gap < 0 else 0)


def test_memory_updates_when_holding_but_never_posts_an_extra_price_action():
    core = memory_core()
    tick(core, 0, 3.)
    initial = core._pricer_coeff_snapshot(core.firm2)
    for step in range(1, 6):
        target, changed = tick(core, step, 3.)
        assert not changed and not target.any()
        assert core._pricer_coeff_snapshot(core.firm2) == initial
    assert core.firm2.undercut_steps == 5
    # The operational clock preserves 50% daily retention over six blocks;
    # five held-price observations therefore cover five-sixths of a day.
    assert core.firm2.reference_window_steps == 6
    assert core.firm2.fast_gap == pytest.approx(3*(1-.5**(5/6)))


def test_retaliation_duration_and_ema_are_independent_of_price_cadence():
    cores = [memory_core(seed=11), memory_core(seed=11)]
    for step in range(70):
        for core, cadence in zip(cores, [1, 6]):
            tick(core, step, 3. if step <= 6 else 0., cadence)
        a, b = (c.firm2 for c in cores)
        for field in ['retaliation_left', 'retaliation_episodes', 'undercut_steps',
                      'premium_steps', 'fast_gap', 'ema_share', 'ema_gap']:
            assert getattr(a, field) == getattr(b, field)


def test_new_clock_resets_consecutive_evidence_and_legacy_clock_is_unchanged():
    core = memory_core()
    for step in range(12):
        tick(core, step, 3. if step < 6 or step > 6 else 0.)
    assert core.firm2.retaliation_episodes == 0
    old = memory_core(revision=3, market_clock=False)
    for step in range(36):
        tick(old, step, 3.)
        assert old.firm2.retaliation_episodes == 0
    tick(old, 36, 3.)
    assert old.firm2.retaliation_episodes == 1


def agent(**kwargs):
    kwargs.setdefault('ent_coeff', .04)
    kwargs.setdefault('min_ent_coeff', .01)
    return PPOAgent(6, 3, hidden_dim=16, single_state_dim=2, frame_stack=3,
        persistent_recurrence=True, privileged_critic=True, privileged_value_only=True,
        critic_state_dim=2, gamma=.999, learning_corrections=True,
        td_value_targets=True, learner_revision=4, **kwargs)


@pytest.mark.parametrize('delay', [2, 6, 12, 30])
@pytest.mark.parametrize('value_offset', [-150., 0., 150.])
def test_delayed_reward_credit_matches_discounted_cashflows(delay, value_offset):
    model = agent(lam=.9)
    n = delay + 2
    discount = model.gamma**6
    # A flat, imperfect critic cannot predict the delayed response. Adding a
    # constant continuing value must not reverse the action's marginal credit.
    values = torch.full((n,), value_offset)
    base = torch.full((n,), (1-discount)*value_offset)
    rewards = base.clone()
    rewards[0] -= .12
    rewards[delay] += .2
    kw = dict(discounts=torch.full((n,), discount), truncations=torch.tensor([0.]*(n-1)+[1.]),
              next_value_overrides=values.clone(), normalize_advantage=False)
    advantages, returns = model._gae_scalar(rewards, values, torch.zeros(n), trace_lambda=1., **kw)
    expected = -.12 + discount**delay * .2
    assert advantages[0] == pytest.approx(expected, abs=3e-5)
    assert returns[0] == pytest.approx(value_offset+expected, abs=4e-5)
    short_trace = model._gae_scalar(rewards, values, torch.zeros(n), **kw)[0][0]
    if delay >= 6:
        assert advantages[0] > 0 and short_trace < 0


def test_actor_return_does_not_cross_resets_or_drop_truncation_bootstraps():
    model = agent(lam=.9)
    rewards = torch.tensor([1., 2., 100., 200.])
    values = torch.tensor([30., 31., -20., -19.])
    discount = torch.tensor([.9, .8, .7, .6])
    adv, ret = model._gae_scalar(rewards, values, torch.tensor([0., 0., 0., 1.]),
        normalize_advantage=False, discounts=discount, truncations=torch.tensor([0., 1., 0., 0.]),
        next_value_overrides=torch.tensor([31., 40., -19., 999.]), trace_lambda=1.)
    torch.testing.assert_close(ret, torch.tensor([1+.9*(2+.8*40), 2+.8*40, 100+.7*200, 200.]))
    torch.testing.assert_close(adv, ret-values)
    with pytest.raises(ValueError, match='truncated'):
        model._gae_scalar(rewards, values, torch.zeros(4), truncations=torch.tensor([0., 1., 0., 1.]),
            next_value_overrides=torch.full((4,), float('nan')), trace_lambda=1.)


def test_ppo_update_uses_long_actor_credit_but_keeps_bilateral_critic_targets():
    from test_fundamental_repairs import collect_components
    model = agent(lam=.9, potential_shaping_scale=.05)
    model.set_value_objective(.6)
    collect_components(model, 15, terminal=False)
    model.reset_recurrent_state()
    collect_components(model, 15, terminal=False, seed=14)
    transitions = model.buf[:]
    rewards = torch.tensor([t.reward for t in transitions])
    values = torch.stack([t.old_value for t in transitions])
    kwargs = dict(normalize_advantage=False,
        discounts=torch.tensor([t.discount for t in transitions]),
        truncations=torch.tensor([t.truncated for t in transitions]),
        next_value_overrides=torch.stack([t.next_value_override for t in transitions]))
    dones = torch.tensor([t.done for t in transitions])
    exposures = torch.tensor([
        (1.0-float(t.discount))/(1.0-model.gamma) for t in transitions
    ])
    reward_rate = sum(float(t.raw_reward) for t in transitions) / float(exposures.sum())
    centered = rewards - reward_rate * exposures
    expected_a, expected_r = model._gae_scalar(
        centered, values, dones, trace_lambda=.9, **kwargs
    )
    expected_critic = model._gae_scalar(centered, values, dones, **kwargs)[1]
    metrics = model.update(epochs=2, batch_size=30)
    assert metrics['actor_trace_lambda'] == metrics['critic_trace_lambda'] == .9
    assert metrics['actor_return_mean'] == pytest.approx(float(expected_r.mean()), abs=1e-5)
    assert metrics['critic_target_mean'] == pytest.approx(float(expected_critic.mean()), abs=1e-5)
    assert metrics['critic_own_target_mean']-.6*metrics['critic_rival_target_mean'] == pytest.approx(metrics['critic_target_mean'], abs=1e-5)
    actions = torch.stack([t.action for t in transitions])
    for action, detail in metrics['action_credit'].items():
        assert detail['raw_advantage_mean'] == pytest.approx(float(expected_a[actions==int(action)].mean()))
    assert metrics['behavior_replay_max_logp_error'] < 2e-5
    assert metrics['critic_actor_update_max_value_change'] == 0


def test_stage_exploration_uses_exact_logged_actor_uniform_mixture():
    from pricing_models import FirmRLPricer
    pricer = FirmRLPricer(seed=8, opt_keys=['base_fare'], single_state_dim=2,
        state_frame_stack=3, action_feature_dim=0, critic_state_dim=2,
        persistent_recurrence=True, privileged_critic=True, privileged_value_only=True,
        learning_corrections=True, td_value_targets=True)
    model = pricer.agent
    assert model.learner_revision == 4
    rates = []
    for progress in (0., .25, .5, .75, 1., 0.):
        model.set_stage_progress(progress)
        rates.append(model.ent_coeff)
        assert model.ent_coeff == pytest.approx(.002+.038*.5*(1+np.cos(np.pi*progress)))
    assert rates[0] == rates[-1] == .04 and rates[-2] == .002
    model.set_exploration_commit_mean(3.)
    model._commit_action = 2
    logits = torch.linspace(-18., 18., model.action_dim).reshape(1, -1)
    raw = torch.distributions.Categorical(logits=logits)
    actual = model._behavior_distribution(logits, .9, None, 0., -1)
    expected = .1 * raw.probs + .9 / model.action_dim
    torch.testing.assert_close(actual.probs, expected)
    assert actual.probs.min() >= .9 / model.action_dim
    assert model.exploration_rate > 0
    assert model._commit_continue_prob() == 0
    assert model.counterfactual_actor_weight == 0 and not model.hard_kl_stop


@pytest.mark.parametrize('opponent_due', [False, True])
@pytest.mark.parametrize('raise_in_critic', [False, True])
@pytest.mark.parametrize('shocks', [False, True])
def test_boundary_preview_sees_delayed_response_and_restores_all_touched_state(monkeypatch, opponent_due, raise_in_critic, shocks):
    import datetime as dt
    import pickle
    from market_dynamics import MarketDynamicsConfig
    from market_shocks import ShockConfig, ShockScenario, ShockEvent
    core = Core(market_name='Chicago', seed=5, firm1_mode='RL',
        firm2_mode='adaptive_best_response_memory', credit_assignment_v25=True,
        stage_specific_rewards=True, opponent_profit_lead_weight=.6,
        learning_corrections=True, learner_revision=4, persistence_features=True,
        quote_noise_dollars=0., regime_market=shocks)
    if shocks:
        core.apply_calibration({})
        core.configure_v26(market_dynamics=MarketDynamicsConfig(enabled=True,
            capacity_target_fulfillment=0., empirical_trips=False), steps_per_day=6)
        core.configure_v26r(shock_config=ShockConfig(enabled=True), supply_model='v2',
            shock_scenario=ShockScenario(start_date=dt.date(2023, 3, 6), events=[
                ShockEvent('driver_shortage', 0, 3, intensity=.65, announced=False)]))
    core.firm2.undercut_steps = 5
    monkeypatch.setattr(core, '_observed_own_minus_competitor_gap', lambda *args: 3.)
    before = core._snapshot_counterfactual_state()
    original = core.firm1.agent.value_estimates
    seen = []
    def value(*args, **kwargs):
        if shocks:
            assert core._regime_last_effect.fleet_multiplier < 1.
        seen.append((core.firm2.undercut_steps, core.firm2.retaliation_episodes,
                     core.firm2.memory_state))
        if raise_in_critic:
            raise ValueError('test preview failure')
        return original(*args, **kwargs)
    monkeypatch.setattr(core.firm1.agent, 'value_estimates', value)
    kwargs = dict(previous_metrics=FirmMetrics(), steps_per_day=6,
                  opponent_due=opponent_due, episode_seed=1234)
    if raise_in_critic:
        with pytest.raises(ValueError, match='test preview failure'):
            core._boundary_bootstrap_preview(**kwargs)
    else:
        scalar, components, potential = core._boundary_bootstrap_preview(**kwargs)
        torch.testing.assert_close(scalar, components @ core.firm1.agent.value_weights)
        assert potential is None or np.isfinite(potential)
    expected = (0 if shocks else 6, 1, 'retaliation') if opponent_due else (
        (0, 1, 'calm') if shocks else (5, 0, 'calm'))
    assert seen == [expected]
    after = core._snapshot_counterfactual_state()
    for key in ('market', 'choice_model', 'firm2', 'driver_supply', 'rng_state',
                'python_rng_state', 'observers', 'tracked'):
        assert pickle.dumps(before[key]) == pickle.dumps(after[key]), key
    torch.testing.assert_close(before['torch_rng_state'], after['torch_rng_state'], rtol=0, atol=0)
    for key in ('actor', 'critic', 'actor_history'):
        torch.testing.assert_close(before['firm1_execution']['agent']['recurrent_state'][key],
                                   after['firm1_execution']['agent']['recurrent_state'][key], rtol=0, atol=0)


@pytest.mark.parametrize('revision', [0, 1, 2, 3])
def test_old_learner_sampling_and_updates_match_the_pre_edit_code_exactly(revision):
    import sys
    import tarfile
    import types
    from pathlib import Path
    from test_fundamental_repairs import collect_components
    path = Path(__file__).resolve().parents[2]/'rideshare_v26r_corrected/runs/delayed_credit_revision4_tests_20261016/pre_edit_source.tar.gz'
    with tarfile.open(path) as archive:
        code = archive.extractfile('PPOAgent.py').read()
    name = f'_revision4_reference_{revision}'
    reference = types.ModuleType(name)
    sys.modules[name] = reference
    exec(compile(code, str(path), 'exec'), reference.__dict__)
    kwargs = dict(state_dim=6, action_dim=3, hidden_dim=16, single_state_dim=2, frame_stack=3,
        persistent_recurrence=True, privileged_critic=True, privileged_value_only=True,
        critic_state_dim=2, gamma=.999, learning_corrections=revision>0,
        td_value_targets=True, learner_revision=max(1, revision))
    torch.manual_seed(506); old = reference.PPOAgent(**kwargs)
    torch.manual_seed(506); new = PPOAgent(**kwargs)
    for model in (old, new):
        torch.manual_seed(912)
        model.set_stage_progress(.6)
        model.set_value_objective(.6)
        collect_components(model, 12)
    for a, b in zip(old.buf, new.buf):
        assert a.raw_reward == b.raw_reward
        for field in ('action', 'old_logp', 'old_value'):
            torch.testing.assert_close(getattr(a, field), getattr(b, field), rtol=0, atol=0)
    torch.manual_seed(172); old_metrics = old.update(epochs=2, batch_size=6)
    torch.manual_seed(172); new_metrics = new.update(epochs=2, batch_size=6)
    for key, value in old.net.state_dict().items():
        torch.testing.assert_close(value, new.net.state_dict()[key], rtol=0, atol=0)
    for key in ('loss', 'policy_loss', 'value_loss', 'entropy'):
        assert old_metrics[key] == new_metrics[key]


@pytest.mark.parametrize('seed', [3, 17, 91])
def test_ppo_learns_delayed_opponent_response_from_a_wrong_warm_start(seed, record_property):
    """Known-answer response game, NOT the ride simulator or a dominance test.

    An opening action pays an immediate cost; only the context-appropriate
    action yields a premium the real memory controller will accommodate after
    six days. Later actions are sampled normally but do not change this opening
    contract. This isolates delayed action credit from price-path optimization.
    """
    torch.manual_seed(seed)
    model = agent(lam=.9)
    model.set_value_objective(.6)
    model.warm_start_action(1, logit=2.)  # Wrong for the positive context.
    with torch.no_grad():
        model.net.v_head.weight.zero_()
        model.net.v_head.bias.copy_(torch.tensor([.15, .25]))
    discount = model.gamma**6
    baseline_parts = np.array([150., 250.]) * (1-discount)
    initial = torch.softmax(model.net.actor_forward(torch.ones(1, 6))[0], -1)[0, 2].item()
    budget = 300
    scheduler = TrainingStageScheduler(mode='direct')
    record_property('training_updates', budget)
    for update in range(budget):
        # Match Core: decay over the declared stage, not over an obsolete
        # 100-update budget followed by 200 updates at the terminal rates.
        model.set_stage_progress(scheduler.stage_progress(update, budget))
        assert model.stage_progress == pytest.approx(update / (budget - 1))
        if update < budget - 1:
            assert model.curr_lr > model.min_lr
        for sign in (-1., 1.) * 2:
            model.reset_recurrent_state()
            rival = FirmMemoryBestResponsePricer(seed=seed, operational_clock=True,
                                                 accommodation_steps=36)
            rival.act(hour=12, weather='clear', **ANCHORS)
            opening_action = None
            for day in range(8):
                # Like the real observation, expose the already-posted tariff.
                # Never reveal the action before it is selected. Omitting this
                # would make the critic unable to identify the pending contract.
                posted = (opening_action+1)/3 if day else 0.
                state = np.tile([sign, posted], 3).astype(np.float32)
                privileged = np.array([sign, posted], dtype=np.float32)
                if day:
                    nxt = model.value_estimates(state, critic_state=privileged)[0]
                    model.finalize_last_transition(next_value=nxt, next_value_components=model.last_preview_components)
                result = model.act(state, critic_state=privileged)
                if day == 0:
                    opening_action = result[0]
                correct = opening_action == (1 if sign < 0 else 2)
                prior_moves = rival.accommodation_moves
                for _ in range(6):
                    rival.update(FirmMetrics(), -3. if correct else 0.)
                rival.act(hour=12, weather='clear', **ANCHORS)
                parts = baseline_parts.copy()
                if day == 0 and opening_action != 0:
                    parts[0] -= .12
                if rival.accommodation_moves > prior_moves and prior_moves == 0:
                    parts[0] += .5
                last = day == 7
                model.store(result[1], result[0], float(parts[0]-.6*parts[1]),
                    False, None, result[2], result[3], duration=6, discount=discount,
                    truncated=last, reward_components=parts,
                    next_value=0. if last else None,
                    next_value_components=torch.tensor([150., 250.]) if last else None)
        metrics = model.update(epochs=8, batch_size=32)
        assert metrics['behavior_replay_max_logp_error'] < 2e-5
        if update in (99, 199):
            with torch.no_grad():
                opening = torch.tensor([[1., 0.]*3])
                p_correct = torch.softmax(model.net.actor_forward(opening)[0], -1)[0, 2].item()
            record_property(f'correct_probability_update_{update+1}', p_correct)
            record_property(f'learning_rate_update_{update+1}', model.curr_lr)
            record_property(f'entropy_coefficient_update_{update+1}', model.ent_coeff)
    probabilities = []
    for sign, correct_action in [(-1., 1), (1., 2)]:
        model.reset_recurrent_state()
        state = torch.tensor(np.tile([sign, 0.], 3), dtype=torch.float32).reshape(1, -1)
        p = torch.softmax(model.net.actor_forward(state)[0], -1)[0, correct_action].item()
        probabilities.append(p)
    record_property('initial_wrong_context_correct_probability', initial)
    record_property('final_correct_probabilities', probabilities)
    record_property('last_actor_advantage_std', metrics['actor_raw_advantage_std'])
    record_property('last_critic_rmse', metrics['critic_rmse'])
    assert model.update_calls == budget
    assert model.curr_lr == model.min_lr and model.ent_coeff == model.min_ent_coeff
    assert min(probabilities) > .70, probabilities
