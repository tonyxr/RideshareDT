"""Real-scale learner regression tests; not market-performance claims."""
import numpy as np
import pytest
import torch

from PPOAgent import PPOAgent


def agent(**kwargs):
    return PPOAgent(6, 3, hidden_dim=16, single_state_dim=2, frame_stack=3,
        persistent_recurrence=True, privileged_critic=True,
        privileged_value_only=True, critic_state_dim=2, gamma=.999,
        learning_corrections=True, td_value_targets=True, learner_revision=3, **kwargs)


@pytest.mark.parametrize('seed,offset', [(3, -100.), (17, 0.), (91, 150.)])
def test_real_scale_history_fit_with_production_optimizer(seed, offset):
    torch.manual_seed(seed)
    model = agent()
    signs = torch.tensor([-1., 1.] * 8)
    histories = [[torch.tensor([float(sign), 0.])] + [torch.zeros(2) for _ in range(4)]
                 for sign in signs]
    contexts = [(h, len(h)) for h in histories]
    states, privileged = torch.zeros(16, 6), torch.zeros(16, 2)
    behavior = torch.zeros(16, 5)
    targets = offset + 2. * signs
    with torch.no_grad():
        model.net.v_head.bias.fill_(offset / model.value_scale)
    actor = [p.detach().clone() for p in model.net.actor_parameters()]
    recurrent = model.net.critic_recurrent.weight_ih_l0.detach().clone()
    for _ in range(160):
        values = model._critic_outputs_batch(states, privileged, contexts=contexts,
                                              behavior_contexts=behavior)[0]
        if values.ndim == 2:
            values = values[:, 0]
        loss = model._critic_value_loss(values/model.value_scale, targets/model.value_scale)
        model.critic_opt.zero_grad(); loss.backward(); model.critic_opt.step()
    assert (values.detach()-targets).abs().mean() < .25
    assert 1 - (values.detach()-targets).var()/targets.var() > .95
    assert (model.net.critic_recurrent.weight_ih_l0-recurrent).norm() > .01
    for before, after in zip(actor, model.net.actor_parameters()):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert model.critic_opt.param_groups[0]['eps'] == 1e-8
    assert model.actor_opt.param_groups[0]['eps'] == 1e-5


def test_optimizer_epsilon_checkpoint_roundtrip():
    first, second = agent(), agent()
    second.load_optimizer_state_dict(first.optimizer_state_dict())
    assert second.critic_opt.param_groups[0]['eps'] == 1e-8


def test_actor_only_change_cannot_change_critic_predictions():
    torch.manual_seed(19)
    model = agent()
    history = [torch.randn(2) for _ in range(12)]
    contexts = [(history, n) for n in (3, 6, 9, 12)]
    states, privileged = torch.randn(4, 6), torch.randn(4, 2)
    def predict():
        return model._critic_outputs_batch(states, privileged, contexts=contexts,
                                            behavior_contexts=torch.zeros(4, 5))[0]
    before = predict().detach().clone()
    with torch.no_grad():
        for parameter in model.net.actor_parameters():
            parameter.add_(torch.randn_like(parameter) * .1)
    torch.testing.assert_close(predict(), before, rtol=0, atol=0)
    predict().sum().backward()
    assert all(p.grad is None for p in model.net.actor_parameters())
    assert model.net.critic_recurrent.weight_ih_l0.grad.abs().sum() > 0


@pytest.mark.parametrize('weight', [0., .5, .6, 1.])
@pytest.mark.parametrize('duration', [1, 6, 24])
def test_bilateral_constant_reward_fixed_points_and_stage_recombination(weight, duration):
    model = agent()
    rates = torch.tensor([.3, .8])
    with torch.no_grad():
        model.net.v_head.weight.zero_()
        model.net.v_head.bias.copy_(rates)
    model.set_value_objective(weight)
    state, privileged = np.zeros(6), np.zeros(2)
    value = model.value_estimates(state, critic_state=privileged)[0]
    expected = (rates[0]-weight*rates[1]) / (1-model.gamma)
    torch.testing.assert_close(value, expected)
    discount = model.gamma**duration
    parts = rates * ((1-discount)/(1-model.gamma))
    result = model.act(state, critic_state=privileged)
    model.store(result[1], result[0], float(parts @ model.value_weights), False,
                None, result[2], result[3], discount=discount, duration=duration,
                truncated=True, reward_components=parts.tolist(), next_value=value,
                next_value_components=model.last_preview_components)
    metrics = model.update(epochs=1, batch_size=1)
    assert metrics['critic_target_mean'] == pytest.approx(float(expected), abs=2e-4)
    assert metrics['critic_own_target_mean'] == pytest.approx(300., abs=2e-4)
    assert metrics['critic_rival_target_mean'] == pytest.approx(800., abs=2e-4)


def collect_components(model, length=6, *, terminal=True, seed=10):
    rng = np.random.default_rng(seed)
    for i in range(length):
        state, privileged = rng.normal(size=6), rng.normal(size=2)
        if model.buf and not model.buf[-1].next_state_finalized:
            nxt = model.value_estimates(state, critic_state=privileged)[0]
            model.finalize_last_transition(next_value=nxt,
                next_value_components=model.last_preview_components, next_potential=.1)
        result = model.act(state, critic_state=privileged)
        parts = [float(.4+.1*privileged[0]), float(.8+.2*privileged[1])]
        done = i == length-1 and terminal
        nxt = None
        if i == length-1 and not terminal:
            nxt = model.value_estimates(state, critic_state=privileged)[0]
        model.store(result[1], result[0], float(torch.tensor(parts) @ model.value_weights),
            done, None, result[2], result[3], duration=6,
            truncated=i==length-1 and not terminal,
            reward_components=parts, next_value=nxt,
            next_value_components=model.last_preview_components if nxt is not None else None,
            potential=.1, next_potential=.1 if nxt is not None or done else None)


@pytest.mark.parametrize('terminal', [True, False])
def test_component_returns_match_scalar_with_shaping_and_episode_boundaries(terminal):
    model = agent(potential_shaping_scale=.05)
    model.set_value_objective(.6)
    collect_components(model, terminal=terminal)
    model.reset_recurrent_state()
    collect_components(model, terminal=terminal, seed=12)
    metrics = model.update(epochs=2, batch_size=6)
    assert np.isfinite(metrics['critic_rival_rmse'])
    assert metrics['critic_own_target_mean']-.6*metrics['critic_rival_target_mean'] == pytest.approx(metrics['critic_target_mean'], abs=1e-4)


def test_value_objective_transaction_and_checkpoint_restore():
    model, loaded = agent(), agent()
    model.set_value_objective(.6)
    saved = model.recurrent_state_dict()
    model.set_value_objective(0.)
    model.load_recurrent_state_dict(saved)
    torch.testing.assert_close(model.value_weights, torch.tensor([1., -.6]))
    loaded.load_normalizer_state_dict(model.normalizer_state_dict())
    torch.testing.assert_close(loaded.value_weights, model.value_weights)


def test_component_collection_rejects_inconsistent_reward():
    model = agent()
    result = model.act(np.zeros(6), critic_state=np.zeros(2))
    with pytest.raises(ValueError, match='reconstruct'):
        model.store(result[1], result[0], 9., True, None, result[2], result[3],
                    reward_components=[1., 2.])


def test_sampling_and_ppo_replay_use_raw_actor_without_external_exploration():
    torch.manual_seed(15)
    model = agent(scheduled_exploration=True, initial_exploration_rate=.9)
    model.set_exploration_commit_mean(3.)
    model._commit_action = 2
    logits = torch.tensor([[1., 2., -1.]])
    expected = torch.distributions.Categorical(logits=logits)
    actual = model._behavior_distribution(logits, .9, None, 2/3, 2)
    torch.testing.assert_close(actual.probs, expected.probs, rtol=0, atol=0)
    assert model.exploration_rate == model._commit_continue_prob() == 0
    model.set_training_progress(.3)
    assert model.exploration_rate == 0
    model.set_value_objective(.6)
    collect_components(model, 12)
    transitions = model.buf[:]
    replay = model._actor_logits_batch(torch.stack([t.state for t in transitions]),
        contexts=[(t.actor_history, t.actor_history_length) for t in transitions])
    distribution = torch.distributions.Categorical(logits=replay)
    torch.testing.assert_close(distribution.log_prob(torch.stack([t.action for t in transitions])),
                               torch.stack([t.old_logp for t in transitions]), atol=1e-6, rtol=1e-6)
    assert all(t.exploration_rate == t.commit_prob == 0 for t in transitions)
    metrics = model.update(epochs=2, batch_size=12)
    assert metrics['critic_actor_update_max_value_change'] == 0
    assert sum(item['samples'] for item in metrics['action_credit'].values()) == 12
    assert metrics['behavior_replay_max_logp_error'] < 1e-6


def test_greedy_and_sampled_evaluation_are_explicit_not_substituted():
    model = agent()
    model.warm_start_action(1, logit=1.)
    counts = np.zeros(3, dtype=int)
    for _ in range(100):
        result = model.act(np.zeros(6), critic_state=np.zeros(2), deterministic=True, policy_mode='sample_raw')
        counts[result[0]] += 1
    assert np.all(counts > 0)
    assert model.act(np.zeros(6), critic_state=np.zeros(2), deterministic=True, policy_mode='argmax_raw')[0] == 1


def test_batch_requires_independent_completed_openings_not_just_many_transitions():
    from Core import Core
    from types import SimpleNamespace
    model = agent()
    core = object.__new__(Core)
    core.firm1 = SimpleNamespace(agent=model)
    core.learner_revision, core.ppo_rollout_episodes = 3, 2
    collect_components(model, 12, terminal=False)
    assert not core._ppo_rollout_ready(6)
    first = model.buf[:]
    model.reset_recurrent_state()
    collect_components(model, 6, terminal=False, seed=15)
    assert core._ppo_rollout_ready(6)
    assert first[-1].actor_history is not model.buf[-1].actor_history
    metrics = model.update(epochs=2, batch_size=18)
    assert metrics['completed_rollout_episodes'] == 2
    assert metrics['behavior_replay_max_logp_error'] < 1e-6


def test_collection_episode_lengths_preserve_existing_dynamic_horizons():
    from Core import Core
    from platform_mdp import TrainingStageScheduler
    core = object.__new__(Core)
    core.credit_assignment_v25, core.learner_revision = True, 3
    core.training_stage_scheduler = TrainingStageScheduler(dynamic_episode_days=60)
    for stage in core.training_stage_scheduler.stages:
        assert core._training_opponent_episode_days(stage, 'adaptive_best_response_memory') == 60
    assert core._training_opponent_episode_days(core.training_stage_scheduler.stages[-1], 'pi_price_gap') == 300


@pytest.mark.parametrize('revision', [0, 1, 2])
def test_pre_edit_learners_are_exactly_preserved(revision):
    import sys
    import tarfile
    import types
    from pathlib import Path
    path = Path(__file__).resolve().parents[2]/'rideshare_v26r_corrected/runs/fundamental_repairs_tests_20261016/pre_edit_source.tar.gz'
    with tarfile.open(path) as archive:
        code = archive.extractfile('PPOAgent.py').read()
    name = f'_revision3_reference_{revision}'
    reference = types.ModuleType(name); sys.modules[name] = reference
    exec(compile(code, str(path), 'exec'), reference.__dict__)
    kwargs = dict(state_dim=6, action_dim=3, hidden_dim=16, single_state_dim=2,
        frame_stack=3, persistent_recurrence=True, privileged_critic=True,
        privileged_value_only=True, critic_state_dim=2, gamma=.999,
        learning_corrections=revision>0, td_value_targets=True, learner_revision=max(1, revision))
    torch.manual_seed(751); old = reference.PPOAgent(**kwargs)
    torch.manual_seed(751); new = PPOAgent(**kwargs)
    for key, value in old.net.state_dict().items():
        torch.testing.assert_close(value, new.net.state_dict()[key], rtol=0, atol=0)
    for i in range(12):
        state, privileged = np.full(6, i/12), np.full(2, i/7)
        for model in (old, new):
            model.observe_recurrent_state(state/2, privileged)
        a = old.act(state, critic_state=privileged, deterministic=True)
        b = new.act(state, critic_state=privileged, deterministic=True)
        assert a[0] == b[0]
        for x,y in zip(a[1:], b[1:]):
            torch.testing.assert_close(x,y,rtol=0,atol=0)


def test_component_epoch_counts_and_actual_reward_are_unchanged():
    model = agent()
    model.set_value_objective(.6)
    collect_components(model, 30, terminal=False)
    model.reset_recurrent_state()
    collect_components(model, 30, terminal=False, seed=13)
    observed = [t.raw_reward for t in model.buf]
    components = [float(t.reward_components @ t.value_weights) for t in model.buf]
    np.testing.assert_allclose(observed, components, rtol=0, atol=1e-7)
    metrics = model.update(epochs=8, batch_size=64)
    assert metrics['actor_optimizer_steps'] == metrics['critic_optimizer_steps'] == 8
    assert metrics['completed_rollout_episodes'] == 2
    assert metrics['critic_actor_update_max_value_change'] == 0
    assert metrics['critic_recurrent_gradient_rms_median'] > 0


def test_ppo_learns_small_economic_action_differences_without_forced_exploration():
    """Known-answer PPO fixture: hold is legal but not the best contextual action."""
    torch.manual_seed(103)
    model = agent()
    model.set_value_objective(.6)
    for update in range(100):
        for sign in (-1., 1.) * 4:
            model.reset_recurrent_state()
            state, privileged = np.full(6, sign), np.array([sign, 0.])
            for t in range(4):
                if t:
                    nxt = model.value_estimates(state, critic_state=privileged)[0]
                    model.finalize_last_transition(next_value=nxt, next_value_components=model.last_preview_components)
                result = model.act(state, critic_state=privileged)
                correct = 1 if sign < 0 else 2
                parts = [.3 + .15*(result[0] == correct), .6]
                model.store(result[1], result[0], parts[0]-.6*parts[1], t==3,
                            None, result[2], result[3], reward_components=parts)
        model.update(epochs=4, batch_size=32)
    for sign, action in [(-1., 1), (1., 2)]:
        model.reset_recurrent_state()
        logits = model.net.actor_forward(torch.full((1, 6), sign))[0]
        assert torch.softmax(logits, -1)[0, action] > .75


@pytest.mark.parametrize('rival_offset', [100., 250.])
def test_production_width_critic_learns_both_real_scale_history_targets(rival_offset):
    torch.manual_seed(902)
    model = PPOAgent(33*16, 12, hidden_dim=192, single_state_dim=33, frame_stack=16,
        persistent_recurrence=True, privileged_critic=True, privileged_value_only=True,
        critic_state_dim=97, gamma=.999, learning_corrections=True, td_value_targets=True)
    signs = torch.tensor([-1., 1.] * 4)
    histories = [[torch.full((33,), float(sign))] + [torch.zeros(33) for _ in range(4)]
                 for sign in signs]
    contexts = [(h, len(h)) for h in histories]
    states, privileged = torch.zeros(8, 33*16), torch.zeros(8, 97)
    targets = torch.stack((150.+2*signs, rival_offset-3*signs), -1)
    with torch.no_grad():
        model.net.v_head.bias.copy_(torch.tensor([150., rival_offset])/model.value_scale)
    for _ in range(160):
        values = model._critic_outputs_batch(states, privileged, contexts=contexts,
                                            behavior_contexts=torch.zeros(8,14))[0]
        loss = sum(model._critic_value_loss(values[:,i]/model.value_scale,
                        targets[:,i]/model.value_scale) for i in range(2))/2
        model.critic_opt.zero_grad(); loss.backward(); model.critic_opt.step()
    assert (values.detach()-targets).abs().max() < .25
    assert torch.all(1-(values.detach()-targets).var(dim=0)/targets.var(dim=0) > .95)
    weights = torch.tensor([1., -.6])
    assert ((values.detach()-targets) @ weights).abs().max() < .4
