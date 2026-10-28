"""Deterministic learner contracts, not market-performance acceptance gates."""
import numpy as np
import pytest
import torch

from PPOAgent import ActorCritic, PPOAgent


def agent(**kwargs):
    kwargs.setdefault('gamma', .999)
    kwargs.setdefault('learner_revision', 2)  # Retain revision-2 compatibility tests.
    return PPOAgent(6, 3, hidden_dim=8, single_state_dim=2, frame_stack=3,
                    persistent_recurrence=True, privileged_critic=True,
                    privileged_value_only=True, critic_state_dim=2,
                    learning_corrections=True, td_value_targets=True, **kwargs)


def test_fixed_value_scale_preserves_initial_raw_predictions_and_actor():
    torch.manual_seed(10)
    raw = ActorCritic(6, 3, hidden=8)
    torch.manual_seed(10)
    scaled = ActorCritic(6, 3, hidden=8, value_scale=1000.)
    states = torch.randn(7, 6)
    torch.testing.assert_close(raw.critic_forward(states)[0], scaled.critic_forward(states)[0])
    torch.testing.assert_close(raw.actor_forward(states)[0], scaled.actor_forward(states)[0], rtol=0, atol=0)


def test_scaling_does_not_change_rewards_returns_or_bootstrap_units():
    old, new = agent(learner_revision=1), agent()
    assert old.value_scale == 1 and new.value_scale == pytest.approx(1000.)
    args = (torch.tensor([2., -3., 1.]), torch.tensor([500., -100., 30.]), torch.tensor([0., 0., 1.]))
    a, r = old._gae_scalar(*args, normalize_advantage=False)
    b, s = new._gae_scalar(*args, normalize_advantage=False)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(r, s, rtol=0, atol=0)
    with pytest.raises(ValueError, match='gamma'):
        agent(gamma=1.)


@pytest.mark.parametrize('seed', [3, 17, 91])
def test_scaled_critic_fits_large_signed_state_dependent_values(seed):
    torch.manual_seed(seed)
    model = agent()
    x = torch.linspace(-1, 1, 32)
    privileged = torch.stack((x, x.square()), dim=1)
    states = torch.zeros(32, 6)
    targets = (1000. if seed != 17 else -1000.) + 400. * x
    opt = torch.optim.Adam(model.net.critic_parameters(), lr=.015)
    actor_before = [p.detach().clone() for p in model.net.actor_parameters()]
    for _ in range(250):
        pred = model.net.critic_forward(states, privileged)[0]
        loss = model._critic_value_loss(pred / model.value_scale, targets / model.value_scale)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = model.net.critic_forward(states, privileged)[0]
    assert (pred-targets).abs().mean() < 15.
    assert 1 - (pred-targets).var() / targets.var() > .98
    for before, after in zip(actor_before, model.net.actor_parameters()):
        torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_scaled_network_checkpoint_round_trip():
    first, second = agent(), agent()
    second.net.load_state_dict(first.net.state_dict())
    state, privileged = torch.randn(4, 6), torch.randn(4, 2)
    torch.testing.assert_close(first.net.critic_forward(state, privileged)[0],
                               second.net.critic_forward(state, privileged)[0], rtol=0, atol=0)


def critic_replay(model, transitions):
    contexts = [(t.actor_history, t.actor_history_length) for t in transitions]
    behavior = torch.zeros(len(transitions), model.action_dim + 2)
    for i, t in enumerate(transitions):
        behavior[i, -2:] = torch.tensor([t.exploration_rate, t.commit_prob])
        if t.commit_prob > 0:
            behavior[i, t.commit_action] = 1.
    return model._critic_outputs_batch(torch.stack([t.state for t in transitions]),
        torch.stack([t.critic_state for t in transitions]),
        [t.critic_hidden for t in transitions], contexts, behavior)[0]


def test_history_critic_replays_collection_and_backpropagates_without_actor_gradients():
    from test_learning_corrections import collect_episode
    torch.manual_seed(9)
    model = agent()
    model.exploration_commit_mean = 3.
    collect_episode(model, 5)
    first = model.buf[:]
    model.reset_recurrent_state()
    collect_episode(model, 3)
    transitions = model.buf[:]
    predictions = critic_replay(model, transitions)
    torch.testing.assert_close(predictions, torch.stack([t.old_value for t in transitions]), atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(critic_replay(model, first), predictions[:5])
    predictions.sum().backward()
    assert model.net.critic_recurrent.weight_ih_l0.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.net.actor_parameters())
    # Critic replay uses updated weights, never stale rollout hidden states.
    before = predictions.detach()
    with torch.no_grad():
        model.net.critic_recurrent.weight_ih_l0.add_(.2)
    assert not torch.allclose(critic_replay(model, transitions), before)
    model.update(epochs=2, batch_size=4)
    frames = torch.stack(model._actor_history).unsqueeze(0)
    with torch.no_grad():
        _, expected = model.net.critic_recurrent(model.net.critic_frame_encoder(frames))
    torch.testing.assert_close(expected, model._critic_hidden)


def test_value_distinguishes_history_actor_memory_and_commitment():
    torch.manual_seed(45)
    model = agent()
    state, privileged = torch.zeros(1, 6), torch.zeros(1, 2)
    memory = torch.zeros(1, 8)
    behavior = torch.zeros(1, 5)
    def value(**kwargs):
        return model.net.critic_forward(state, privileged, **kwargs)[0]
    base = value(policy_memory=memory, behavior_context=behavior)
    assert not torch.allclose(base, value(policy_memory=memory+1, behavior_context=behavior))
    assert not torch.allclose(base, value(hidden=torch.ones(1, 1, 8), policy_memory=memory, behavior_context=behavior))
    behavior[0, 1], behavior[0, -2], behavior[0, -1] = 1., .3, 2/3
    assert not torch.allclose(base, value(policy_memory=memory, behavior_context=behavior))
    model._commit_action = 1
    model.exploration_commit_mean = 3.
    context = model._value_behavior_context(False)
    assert context[0, 1] == 1 and context[0, -1] == pytest.approx(2/3)
    assert model._value_behavior_context(True).count_nonzero() == 0
    assert model._value_behavior_context(False, [True, False, True])[0, 1] == 0


def test_value_preview_matches_next_action_without_mutating_history():
    from test_learning_corrections import collect_episode
    model = agent()
    collect_episode(model, 3)
    history_length = len(model._actor_history)
    state, privileged = np.ones(6), np.ones(2)
    preview = model.value_estimates(state, critic_state=privileged)[0]
    assert len(model._actor_history) == history_length
    observed = model.act(state, critic_state=privileged)[3]
    torch.testing.assert_close(preview, observed)


def test_stage_matched_validation_changes_only_critic_targets():
    from pathlib import Path
    from Core import Core
    cache = Path(__file__).resolve().parents[2] / 'artifacts/v26_confirmatory/thresholds/chicago/llm_numeric_primary.jsonl'
    core = Core(market_name='Chicago', seed=61, total_customers_pool=60,
                credit_assignment_v25=True, stage_specific_rewards=True,
                learning_corrections=True, opponent_profit_lead_weight=.6,
                threshold_profile_source='cached', threshold_cache_path=str(cache))
    core.apply_calibration({})
    core._bootstrap_synthetic_profiles(60)
    core._build_generalization_profile_pool(60)
    calls = []
    original = core.long_term_profit_reward_model.compute
    def record(**kwargs):
        result = original(**kwargs)
        calls.append((kwargs.get('training_regime'), kwargs['own_profit_per_request'],
                      kwargs['rival_profit_per_request'],
                      kwargs.get('own_market_share', kwargs.get('own_completed_share', 0.)),
                      kwargs.get('rival_market_share', kwargs.get('rival_completed_share', 0.)),
                      result['reward']))
        return result
    core.long_term_profit_reward_model.compute = record
    reports = []
    for regime in ('market', 'static', 'dynamic', None):
        calls.clear()
        report = core._adaptive_policy_validation(customers_per_step=8, profile_sample_size=8,
            horizon=12, aggressive_horizon=12, opponent_interval_steps=6, steps_per_day=6,
            aggressive_seed_count=1, rollouts_override=[('static', 'static')],
            critic_reward_regime=regime, policy_seed=72)
        assert report['critic_reward_regime'] == (regime or 'competitive_evaluation')
        assert report['critic_policy_mode'] == 'argmax_raw'
        assert 'not_on_policy_fit' in report['critic_diagnostic_distribution']
        assert report['heldout_critic_td_samples'] > 0
        # The reward must be the stage's own linear scalarization of the raw
        # outcomes (four-stage weights, no share term under the profit preset).
        config = core.long_term_profit_reward_config
        for stage, own, rival, own_share, rival_share, reward in calls:
            weights = np.asarray(config.critic_component_weights(stage))
            assert weights.size in (2, 4)
            outcomes = np.asarray([own, rival, own_share, rival_share])[:weights.size]
            assert reward == pytest.approx(float(weights @ outcomes))
        if regime:
            assert any(stage == regime for stage, *_ in calls)
        reports.append(report)
    # Same deployed actions, rewards and economic results, regardless of the
    # reward used solely to assess the value predictions.
    for report in reports[1:]:
        assert report['rollout_summaries'] == reports[0]['rollout_summaries']
    assert reports[0]['heldout_critic_td_mae'] != reports[1]['heldout_critic_td_mae']


def test_critic_metrics_expose_raw_prediction_and_residual_units():
    from test_learning_corrections import collect_episode
    model = agent()
    collect_episode(model, 4)
    metrics = model.update(epochs=2, batch_size=4)
    assert metrics['critic_prediction_mean'] + metrics['critic_residual_mean'] == pytest.approx(metrics['critic_target_mean'], abs=1e-4)
    assert metrics['critic_rmse'] >= abs(metrics['critic_residual_mean']) - 1e-5
    assert np.isfinite(metrics['critic_explained_variance_unclipped'])
    assert metrics['explained_variance'] == metrics['critic_explained_variance_unclipped']


def test_entropy_uses_stage_time_not_updates_and_restarts_each_stage():
    from test_learning_corrections import collect_episode
    from platform_mdp import TrainingStageScheduler
    model = agent()
    scheduler = TrainingStageScheduler(stage_fractions=(.1, .108, .792))
    for first, stop in ((0, 2000), (2000, 4160), (4160, 20000)):
        rates = []
        for day in (first, (first+stop-1)//2, stop-1):
            model.set_stage_progress(scheduler.stage_progress(day, 20000))
            rates.append(model.ent_coeff)
        assert rates[0] == pytest.approx(model.base_ent_coeff)
        assert rates[-1] == pytest.approx(model.min_ent_coeff)
        assert rates[0] > rates[1] > rates[2]
    model.set_stage_progress(.2)
    coefficient = model.ent_coeff
    for _ in range(3):
        model.reset_recurrent_state()
        collect_episode(model, 4)
        model.update(epochs=2, batch_size=4)
        assert model.ent_coeff == coefficient
    fresh = agent()
    fresh.set_stage_progress(.2)  # Same stage time, zero PPO updates.
    assert fresh.ent_coeff == coefficient
    model.reset_learning_context()
    assert model.ent_coeff == coefficient


def test_schedule_save_restore_even_without_reward_rate_normalizer():
    model, loaded = agent(reward_rate_beta=0.), agent(reward_rate_beta=0.)
    model.set_stage_progress(.73)
    model.set_training_progress(.6)
    loaded.load_normalizer_state_dict(model.normalizer_state_dict())
    assert loaded.ent_coeff == model.ent_coeff
    assert loaded.curr_lr == model.curr_lr
    assert loaded.exploration_rate == model.exploration_rate
    assert loaded.stage_progress == .73
    legacy = agent(learner_revision=1, reward_rate_beta=0.)
    assert legacy.normalizer_state_dict() == {}
    entropy = legacy.ent_coeff
    legacy.set_stage_progress(.7)
    assert legacy.ent_coeff == entropy


@pytest.mark.parametrize('corrected', [False, True])
def test_pre_edit_source_identity_for_flags_off_and_legacy_policy(corrected):
    """The running jobs' source snapshot is the reference, not new code."""
    import sys
    import tarfile
    import types
    from pathlib import Path
    archive = Path(__file__).resolve().parents[2] / 'rideshare_v26r_corrected/runs/multicity_competitive_20000_20261016/source_snapshot.tar.gz'
    with tarfile.open(archive) as source:
        code = source.extractfile('PPOAgent.py').read()
    reference = types.ModuleType('_pre_repair_ppo_reference')
    sys.modules[reference.__name__] = reference
    exec(compile(code, str(archive) + '/PPOAgent.py', 'exec'), reference.__dict__)
    kwargs = dict(state_dim=6, action_dim=3, hidden_dim=8, single_state_dim=2,
        frame_stack=3, persistent_recurrence=True, privileged_critic=True,
        privileged_value_only=True, critic_state_dim=2, learning_corrections=corrected,
        td_value_targets=True)
    torch.manual_seed(918)
    old = reference.PPOAgent(**kwargs)
    torch.manual_seed(918)
    new = PPOAgent(**kwargs, learner_revision=1)
    assert old.net.state_dict().keys() == new.net.state_dict().keys()
    for key, value in old.net.state_dict().items():
        torch.testing.assert_close(value, new.net.state_dict()[key], rtol=0, atol=0)
    for i in range(8):
        state = np.arange(6, dtype=np.float32) / (i + 1)
        for model in (old, new):
            model.observe_recurrent_state(state / 2, np.ones(2))
        a = old.act(state, critic_state=np.ones(2), deterministic=True)
        b = new.act(state, critic_state=np.ones(2), deterministic=True)
        assert a[0] == b[0]
        for x, y in zip(a[1:], b[1:]):
            torch.testing.assert_close(x, y, rtol=0, atol=0)


@pytest.mark.parametrize('duration', [1, 6, 24])
def test_constant_reward_bellman_fixed_point_in_original_units(duration):
    model = agent()
    reward_rate = .8
    value = reward_rate / (1-model.gamma)
    discount = model.gamma ** duration
    reward = reward_rate * (1-discount) / (1-model.gamma)
    with torch.no_grad():
        model.net.v_head.weight.zero_()
        model.net.v_head.bias.fill_(reward_rate)
    predicted = model.value_estimates(np.zeros(6), critic_state=np.zeros(2))[0]
    assert predicted.item() == pytest.approx(value)
    advantage, returns = model._gae_scalar(torch.tensor([reward]), predicted.reshape(1),
        torch.zeros(1), discounts=torch.tensor([discount]),
        next_value_overrides=predicted.reshape(1), normalize_advantage=False)
    assert advantage.abs().max() < 1e-4
    torch.testing.assert_close(returns, predicted.reshape(1))
    with torch.no_grad():
        model.net.v_head.bias.fill_(-20.)
    assert model.value_estimates(np.zeros(6), critic_state=np.zeros(2))[0] < -19000


def test_critic_learns_different_returns_for_identical_current_states():
    torch.manual_seed(55)
    model = agent()
    positive = [torch.ones(2)] + [torch.zeros(2) for _ in range(4)]
    negative = [-torch.ones(2)] + [torch.zeros(2) for _ in range(4)]
    contexts = [(positive, 5), (negative, 5)]
    states, privileged, behavior = torch.zeros(2, 6), torch.zeros(2, 2), torch.zeros(2, 5)
    targets = torch.tensor([400., -400.])
    opt = torch.optim.Adam(model.net.critic_parameters(), lr=.01)
    for _ in range(150):
        values = model._critic_outputs_batch(states, privileged, contexts=contexts,
                                             behavior_contexts=behavior)[0]
        loss = model._critic_value_loss(values/model.value_scale, targets/model.value_scale)
        opt.zero_grad(); loss.backward(); opt.step()
    assert (values.detach() - targets).abs().max() < 10.
