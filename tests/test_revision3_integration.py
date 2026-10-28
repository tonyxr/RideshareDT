"""Small multi-city implementation fixtures, never evidence of market dominance."""
import csv
import json
import subprocess

import numpy as np
import pytest
import torch

from dataclasses import fields

from platform_mdp import LongTermProfitRewardConfig
from scripts.run_v26r_staged import BASE, command, environment

STAGES = ('market', 'static', 'catchup', 'profit')


def without_learner_flags(cmd):
    """Reload commands inherit the saved learner contract instead of overriding it."""
    cmd = list(cmd)
    while '--learning_corrections' in cmd:
        cmd.remove('--learning_corrections')
    while '--learner_revision' in cmd:
        i = cmd.index('--learner_revision')
        del cmd[i:i + 2]
    return cmd


@pytest.mark.parametrize('city,slug', [('Chicago', 'chicago'), ('Boston', 'boston'), ('New York City', 'nyc')])
@pytest.mark.parametrize('revision', [3, 4])
def test_versioned_economic_component_collection_and_saved_policy(tmp_path, city, slug, revision):
    out = tmp_path / slug
    out.mkdir()
    cmd = command(out, days=80, integration=True, city=city, slug=slug,
                  update_days=1, rollout_transitions=6, lead_weight=.6)
    # Four-stage curriculum (revision 10+): 20 days per stage.
    for flag, value in {'--training_stage_market_fraction': '.25', '--training_stage_static_fraction': '.25',
                        '--training_stage_dynamic_fraction': '.25', '--training_stage_profit_fraction': '.25',
                        '--training_dynamic_episode_days': '6'}.items():
        cmd[cmd.index(flag)+1] = value
    pooled_episodes = 3 if slug == 'chicago' else 2
    cmd = without_learner_flags(cmd)
    cmd += ['--learning_corrections', '--learner_revision', str(revision), '--ppo_rollout_episodes', str(pooled_episodes)]
    with (out/'run.log').open('w') as log:
        process = subprocess.run(cmd, cwd=BASE, env=environment(), stdout=log,
                                 stderr=subprocess.STDOUT, timeout=900)
    assert process.returncode == 0, (out/'run.log').read_text()[-8000:]
    rows = list(csv.DictReader((out/f'{slug}_training_diagnostics.csv').open()))
    assert len(rows) == 80
    cfg = json.loads((out/f'{slug}_run_config.json').read_text())
    run_reward = cfg['platform_mdp']['active_reward']
    assert run_reward['market_share_competitiveness_weight'] == 0.0
    assert run_reward['opponent_profit_lead_weight'] == .6
    reward_fields = {f.name for f in fields(LongTermProfitRewardConfig)}
    run_config = LongTermProfitRewardConfig(**{k: v for k, v in run_reward.items() if k in reward_fields})
    component_count = len(run_config.critic_component_weights())
    component_names = {
        2: ('own', 'rival'),
        4: ('own_profit', 'rival_profit', 'own_share', 'rival_share'),
        6: ('own_profit', 'rival_profit', 'own_share', 'rival_share',
            'durable_profit_lead', 'adaptive_profit_residual'),
    }[component_count]
    gae_lambda = float(cmd[cmd.index('--ppo_gae_lambda') + 1])
    for stage in STAGES:
        days = 20
        selected = [r for r in rows if r['training_stage_opponent_regime'] == stage]
        assert len(selected) == days
        updates = [r for r in selected if r['ppo_update_performed'] == 'True']
        assert updates, stage
        assert sum(int(r['ppo_rollout_size']) for r in updates) == days
        assert int(updates[-1]['batch']) == int(selected[-1]['batch'])
        weights = run_reward['stage_economic_weights'][stage]
        rates = [float(r['ppo_lr']) for r in selected]
        assert np.all(np.diff(rates) <= 0), stage
        for row in selected:
            components = sum(float(row[k]) for k in (
                'reward_own_profit_component', 'reward_profit_advantage_component',
                'reward_durable_profit_advantage_component', 'reward_adaptive_profit_residual_component',
                'reward_market_share_advantage_component'))
            assert float(row['reward_base']) == pytest.approx(components, abs=1e-8)
            assert float(row['reward_own_profit_component']) == pytest.approx(
                weights['own_profit'] * float(row['avg_profit_per_request']) / 2)
            assert float(row['reward_profit_advantage_component']) == pytest.approx(
                weights['profit_lead'] * float(row['reward_profit_advantage_per_request']) / 2)
            if revision == 3:
                assert float(row['behavior_exploration_rate']) == 0
            else:
                assert float(row['behavior_exploration_rate']) == pytest.approx(
                    float(row['training_stage_exploration_rate']))
                assert float(row['ppo_lr']) == pytest.approx(
                    max(
                        5e-5,
                        3e-4 * float(row['training_stage_learning_rate_scale']),
                    )
                )
            assert int(row['ppo_schedule_total_days']) == 80
            assert int(row['ppo_schedule_stage_days']) == days
            progress = int(row['ppo_schedule_elapsed_days']) / (days-1)
            assert float(row['ppo_stage_progress']) == pytest.approx(progress)
        for row in updates:
            assert float(row['ppo_behavior_replay_max_logp_error']) < 2e-5
            assert float(row['ppo_critic_actor_update_max_value_change']) == 0
            assert float(row['ppo_critic_optimizer_epsilon']) == 1e-8
            for component in component_names:
                assert np.isfinite(float(row[f'ppo_critic_{component}_rmse']))
            assert int(row['ppo_completed_rollout_episodes']) >= pooled_episodes or row['ppo_partial_batch_flush']=='True'
            if revision == 4:
                # Revision-4 learner: both traces use the command's GAE lambda.
                assert float(row['ppo_actor_trace_lambda']) == pytest.approx(gae_lambda)
                assert float(row['ppo_critic_trace_lambda']) == pytest.approx(gae_lambda)
                assert np.isfinite(float(row['ppo_actor_return_mean']))
                assert row['ppo_actor_credit_contract'] == 'bootstrapped_episode_return_minus_collected_value'
        if revision == 4:
            for row in selected:
                # Per-stage entropy scale on the 0.04 base, clipped to [0.002, 0.04].
                expected = min(.04, max(.002, .04 * float(row['training_stage_entropy_scale'])))
                assert float(row['ppo_ent_coeff']) == pytest.approx(expected)
                assert row['rival_memory_clock'] == 'every_operational_period_before_scheduled_price_action'
    final = torch.load(out/'policy.final_training.pt', map_location='cpu', weights_only=False)
    evaluated = torch.load(out/'policy.pt', map_location='cpu', weights_only=False)
    assert final['credit_assignment_config']['learner_revision'] == revision
    assert final['credit_assignment_config']['behavior_policy_contract'] == 'raw_categorical_actor_no_external_mixture'
    assert final['policy_training_config']['rollout_independent_episodes'] == pooled_episodes
    assert final['network_state']['v_head.weight'].shape[0] == component_count
    for name, tensor in final['network_state'].items():
        torch.testing.assert_close(tensor, evaluated['network_state'][name], rtol=0, atol=0)
    assert cfg['benchmark_evaluation']['comparison_checkpoint_day'] == 80
    reload_out = out/'reload'; reload_out.mkdir()
    reload_cmd = without_learner_flags(command(reload_out, days=0, integration=True, city=city, slug=slug,
                                               lead_weight=.6))
    reload_cmd += ['--trained_model_in', str(out/'policy.final_training.pt'), '--eval_only']
    with (reload_out/'run.log').open('w') as log:
        loaded = subprocess.run(reload_cmd, cwd=BASE, env=environment(), stdout=log,
                                stderr=subprocess.STDOUT, timeout=900)
    assert loaded.returncode == 0, (reload_out/'run.log').read_text()[-6000:]
    reloaded = json.loads((reload_out/f'{slug}_run_config.json').read_text())
    assert reloaded['credit_assignment']['learner_revision'] == revision
    assert reloaded['active_reward']['opponent_profit_lead_weight'] == .6
    assert reloaded['args']['ppo_rollout_episodes'] == pooled_episodes
    assert not (reload_out/'policy.pt').exists(), 'Evaluation-only reload must not create a replacement policy'
    if revision == 4:
        assert reloaded['credit_assignment']['opponent_memory_clock'] == final['credit_assignment_config']['opponent_memory_clock']
    if revision == 4 and slug == 'chicago':
        # Exercise the production corner CLI, not only its argument construction.
        # Every controller must see identical tapes regardless of execution order.
        from scripts.v26r_case_eval import run_specs, spec_key
        from scripts.run_v26r_multicity import RIVALS
        specs = ['static:1.1', 'rl_actor', 'myopic_greedy']
        results = []
        for order, sequence in enumerate((specs, specs[::-1])):
            destination = out / f'corner_order_{order}'; destination.mkdir()
            run_specs(city, slug, 'A1_shortage_severe', sequence, 'eval', 1,
                      'full', str(out/'policy.final_training.pt'), str(destination),
                      customers=5, horizon_days=8, rivals=RIVALS, pool=300)
            results.append({spec: json.loads((destination/(spec_key(spec)+'.json')).read_text())
                            for spec in specs})
        for spec in specs:
            assert results[0][spec]['traces'] == results[1][spec]['traces'], spec
            assert results[0][spec]['results'] == results[1][spec]['results'], spec
            assert len(results[0][spec]['rollouts']) == len(RIVALS)
            assert len(results[0][spec]['traces']) == 48*len(RIVALS)
