"""Regression coverage for the second revision-5 audit (no city fitting)."""
import numpy as np
import pytest
import torch

from test_learning_corrections import recurrent_agent, collect_episode, replay
from evaluation_selection import three_comparators
from model_contract import OPPONENT_PROFIT_LEAD_WEIGHT
from scripts.run_v26r_staged import command


@pytest.mark.parametrize('reset', [False, True])
@pytest.mark.parametrize('revision', [2, 4])
@pytest.mark.parametrize('count', [3, 24])
def test_transaction_preserves_episode_identity_and_full_sequence_gradient(reset, revision, count):
    agent = recurrent_agent(learner_revision=revision)
    def collect(count):
        for index in range(count):
            action, state, logp, value, features, mask = agent.act(np.arange(6)/10+index, critic_state=np.ones(2))
            components = dict(reward_components=[0., 0.], next_value_components=torch.zeros(2)) if revision >= 3 else {}
            agent.store(state, action, 0., index == count-1, None, logp, value,
                        action_features=features, action_mask=mask, critic_state=np.ones(2),
                        next_value=0., **components)
    collect(count)
    history = agent._actor_history
    before = replay(agent, agent.buf).detach().clone()
    saved = agent.recurrent_state_dict()
    if reset:
        agent.reset_recurrent_state()
    collect(2)
    agent.load_recurrent_state_dict(saved)
    assert agent._actor_history is history
    assert all(t.actor_history is history for t in agent.buf[:count])
    torch.testing.assert_close(replay(agent, agent.buf[:count]), before)
    history[0].requires_grad_()
    replay(agent, agent.buf[:count])[-1, 0].backward()
    assert history[0].grad.abs().sum() > 0


def test_disabled_commitment_is_bit_exact_to_the_legacy_distribution():
    agent = recurrent_agent()
    agent.learning_corrections = False
    logits = torch.tensor([[.41271, -.38133, 1.741], [2., -1., .121]])
    rates = torch.tensor([.3, .05])
    mask = torch.ones_like(logits, dtype=torch.bool)
    legacy = agent._exploratory_distribution(logits, rates, mask)
    disabled = agent._behavior_distribution(logits, rates, mask, torch.zeros(2), -1)
    torch.testing.assert_close(legacy.probs, disabled.probs, rtol=0, atol=0)
    torch.testing.assert_close(legacy.logits, disabled.logits, rtol=0, atol=0)


def test_missing_critic_optimizer_step_is_reported_without_crashing(monkeypatch):
    agent = recurrent_agent(learner_revision=4)
    for step in range(3):
        action, state, logp, value, features, mask = agent.act(np.ones(6)*step, critic_state=np.ones(2))
        agent.store(state, action, 0., step == 2, None, logp, value, action_features=features,
                    action_mask=mask, critic_state=np.ones(2), next_value=0.,
                    reward_components=[0., 0.], next_value_components=torch.zeros(2))
    monkeypatch.setattr(agent, '_critic_value_loss', lambda *args: torch.tensor(float('inf')))
    metrics = agent.update(epochs=1, batch_size=3)
    assert metrics['critic_optimizer_steps'] == 0
    assert metrics['actor_optimizer_steps'] > 0
    assert np.isfinite(metrics['epoch_full_rollout_kl'])


def test_launchers_share_the_unchanged_objective(tmp_path):
    cmd = command(tmp_path, integration=True)
    assert float(cmd[cmd.index('--opponent_profit_lead_weight') + 1]) == OPPONENT_PROFIT_LEAD_WEIGHT == .6


def test_hindsight_does_not_include_rl_or_privileged_online_search():
    def trace(value):
        return [dict(rollout='r__opening_1', step=i, reward=value) for i in range(6)]
    validation = {'static:1': trace(2), 'static:2': trace(1)}
    evaluation = {**validation, 'rl_actor': trace(100), 'myopic_greedy': trace(200)}
    result = three_comparators(validation, evaluation, start=0, end=1)
    assert result['hindsight_candidate_count'] == 2
    assert result['hindsight_choices'] == {'r__opening_1': 'static:1'}


def test_empty_quote_bins_are_missing_not_fictitious_zero_prices():
    from platform_mdp import ObservationConfig, PlatformObservationModel
    observer = PlatformObservationModel(seed=4, config=ObservationConfig(
        require_observed_quote_bins=True, quote_noise_dollars=0., quote_missing_probability=0.,
        quote_probe_delay_steps=0))
    observer.ingest(own_metrics={}, supply_metrics={}, crowd_stats={
        'distance_bin_0_2_count': 0, 'distance_bin_0_2_price_gap_mean': 0.,
        'distance_bin_2_5_count': 1, 'distance_bin_2_5_price_gap_mean': 2.})
    assert observer.quote_probes['0_2']['available'] == 0
    assert observer.quote_probes['2_5']['available'] == 1
    assert observer.quote_probes['2_5']['gap'] == 2.


def test_city_calibration_contract_rejects_reward_grid_and_city_mismatch(tmp_path):
    import json
    from calibration_artifact import load_city_artifact
    data = dict(format='revision6-city-calibration-v1', city='Chicago',
        opponent_profit_lead_weight=.6, grid=[.6, 1.], warm_start_target=.6,
        supply_operating_point=dict(calibration=dict(burnin_days=15)))
    path = tmp_path/'city.json'
    path.write_text(json.dumps(data))
    assert load_city_artifact(path, city='Chicago', lead_weight=.6, grid=[.6, 1.]) == data
    for overrides in (dict(city='Boston'), dict(lead_weight=.5), dict(grid=[.8, 1.])):
        kwargs = dict(city='Chicago', lead_weight=.6, grid=[.6, 1.]) | overrides
        with pytest.raises(ValueError, match='mismatch'):
            load_city_artifact(path, **kwargs)


def test_threshold_ablation_is_idempotent_after_pool_refresh():
    from market_dynamics import apply_threshold_ablation
    profiles = [dict(ProfileID=i, PriceThreshold=i+1.) for i in range(12)]
    apply_threshold_ablation(profiles, 'shuffled', seed=92)
    first = [p['PriceThreshold'] for p in profiles]
    apply_threshold_ablation(profiles, 'shuffled', seed=92)
    assert [p['PriceThreshold'] for p in profiles] == first
    assert first != [i+1 for i in range(12)]
