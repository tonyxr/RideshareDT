"""Real-PyTorch end-to-end test of curriculum reward and final-policy archival."""
import csv
import json
import subprocess
import torch
import numpy as np
from scripts.run_v26r_staged import BASE, command, environment, summarize


def test_staged_real_ppo_preserves_final_weights_before_and_after_evaluation(tmp_path):
    out = tmp_path / 'staged'
    out.mkdir()
    with (out/'run.log').open('w') as log:
        completed = subprocess.run(command(out, days=192, integration=True), cwd=BASE,
            env=environment(), stdout=log, stderr=subprocess.STDOUT, timeout=900)
    assert completed.returncode == 0, (out/'run.log').read_text()[-8000:]
    with (out/'chicago_training_diagnostics.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 192
    live = [json.loads(line) for line in (out/'chicago_training_live.jsonl').read_text().splitlines()]
    assert len(live) == 192
    assert all(float(a['avg_reward']) == b['avg_reward'] for a, b in zip(rows, live))
    assert all(float(a['raw_policy_entropy']) == b['raw_policy_entropy'] for a, b in zip(rows, live))
    updates_live = [json.loads(line) for line in (out/'chicago_ppo_updates.jsonl').read_text().splitlines()]
    assert len(updates_live) == sum(r['ppo_update_performed'] == 'True' for r in rows)
    episodes = [json.loads(line) for line in (out/'chicago_episodes.jsonl').read_text().splitlines()]
    assert sum(e['days'] for e in episodes) == 192
    assert len((out/'chicago_evaluation_live.jsonl').read_text().splitlines()) == 12
    assert (out/'chicago_episode_diagnostics.png').exists()
    assert (out/'policy.stage_market.pt').exists()
    assert (out/'policy.stage_static_opponent.pt').exists()
    assert (out/'policy.stage_price_gap_catchup.pt').exists()
    assert all('shock_block_counts' in r for r in live)
    assert abs(live[-1]['behavior_exploration_rate'] - .01) < .001
    assert live[-1]['ppo_exploration_rate'] > live[-1]['behavior_exploration_rate']
    assert (out/'chicago_training_diagnostics.csv').stat().st_mtime < (out/'chicago_evaluation_diagnostics.csv').stat().st_mtime
    for regime in ('market', 'static', 'catchup', 'profit'):
        assert any(r['training_stage_opponent_regime'] == regime and
                   r['ppo_update_performed'].lower() == 'true' for r in rows)
    for r in rows:
        if r['training_stage_opponent_regime'] == 'market':
            assert float(r['reward_dominance_term']) == 0
    final = torch.load(out/'policy.final_training.pt', map_location='cpu', weights_only=False)
    after = torch.load(out/'policy.pt', map_location='cpu', weights_only=False)
    assert final['metadata']['comparison_checkpoint_day'] == 192
    assert after['metadata']['comparison_checkpoint_day'] == 192
    assert final['mdp_config']['active_reward']['stage_specific_rewards'] is True
    assert final['reward_mechanism'] == 'staged_two_term_profit_v29_corrected'
    for key, value in final['network_state'].items():
        assert torch.equal(value, after['network_state'][key]), key
    def identical(a, b):
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b)
        elif isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for k in a:
                identical(a[k], b[k])
        elif isinstance(a, (tuple, list)):
            assert len(a) == len(b)
            for x, y in zip(a, b):
                identical(x, y)
        else:
            assert a == b
    identical(final['normalizer_state'], after['normalizer_state'])
    identical(final['optimizer_state'], after['optimizer_state'])
    cfg = json.loads((out/'chicago_run_config.json').read_text())
    assert cfg['benchmark_evaluation']['comparison_checkpoint_day'] == 192
    assert cfg['benchmark_evaluation']['comparison_checkpoint_status'] == 'final_dynamic_policy'
    assert cfg['benchmark_evaluation']['objective_reward_weights'] == dict(
        own_profit=.35, profit_lead=.65, own_completed_share=0.,
        completed_share_lead=0.)
    # Isolated test fixture only: prove enabling logging/plots changes no PPO
    # weights, optimizers, value baselines or simulated training trajectory.
    unlogged = tmp_path/'unlogged'
    unlogged.mkdir()
    with (unlogged/'run.log').open('w') as log:
        second = subprocess.run(command(unlogged, days=192, integration=True) + ['--no-live_diagnostics'],
                                cwd=BASE, env=environment(), stdout=log, stderr=subprocess.STDOUT, timeout=900)
    assert second.returncode == 0, (unlogged/'run.log').read_text()[-8000:]
    without = torch.load(unlogged/'policy.final_training.pt', map_location='cpu', weights_only=False)
    identical(final['network_state'], without['network_state'])
    identical(final['optimizer_state'], without['optimizer_state'])
    identical(final['normalizer_state'], without['normalizer_state'])
    assert (out/'chicago_training_diagnostics.csv').read_bytes() == (unlogged/'chicago_training_diagnostics.csv').read_bytes()


def test_launcher_preserves_market_ppo_and_evaluation_contract(tmp_path):
    cmd = command(tmp_path)
    def value(flag):
        return cmd[cmd.index(flag) + 1]
    assert value('--train_timesteps') == '10000'
    assert value('--rl_warm_start_target') == 'auto'
    assert value('--ppo_min_rollout_transitions') == '180'
    assert value('--eval_timesteps') == '120'
    assert value('--eval_customers') == '50'
    assert value('--checkpoint_validation_horizon') == '60'
    assert '--stage_specific_rewards' in cmd
    assert '--temporal_competitive_reward' not in cmd
    assert value('--market_share_competitiveness_weight') == '0.0'
    assert value('--training_dynamic_episode_days') == '30'
    assert '--benchmark_suite' in cmd


def test_report_lists_losses_without_performance_gate(tmp_path):
    def write_csv(name, rows):
        with (tmp_path/name).open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_csv('chicago_training_diagnostics.csv', [dict(
        training_stage_opponent_regime=('market' if d < 1000 else 'static' if d < 2080 else 'dynamic'),
        ppo_update_performed='false',
    ) for d in range(10000)])
    write_csv('chicago_evaluation_diagnostics.csv', [dict(
        reward_profit_advantage_per_request=1 if d < 90 else -1,
        action_label='tariff_target:1.0' if d < 60 else 'tariff_target:1.1') for d in range(120)])
    write_csv('chicago_benchmark_summary.csv', [dict(benchmark=f'rival{i}', advantage=.1,
        final_quarter_advantage=-.2 if i == 0 else .1) for i in range(5)])
    rollout = dict(own_profit_per_request=1, rival_profit_per_request=2, late_objective_advantage=-2)
    cfg = dict(platform_mdp=dict(active_reward=dict(
            temporal_competitive_reward=False,
            market_share_competitiveness_weight=0.0)),
        credit_assignment=dict(value_components=[
            'own_profit_per_request', 'rival_profit_per_request']),
        benchmark_evaluation=dict(comparison_checkpoint_day=10000,
        comparison_checkpoint_status='final_dynamic_policy', rollout_summaries={'actual_firm2__memory__opening_1':rollout},
        action_switch_rate=.1, action_diversity=.2, mode_tail_action_switch_rate={'opening':0},
        mode_tail_action_diversity={'opening':.1}),
        v26_ood_evaluation=dict(rollout_summaries={'perturbed':rollout}))
    (tmp_path/'chicago_run_config.json').write_text(json.dumps(cfg))
    report = summarize(tmp_path)
    assert report['all_six_late_leads_positive'] is False
    assert [r['rival'] for r in report['losing_cases']] == ['actual_training_rival_paired', 'rival0']
    assert report['profit_leads'][0]['full'] == -1
    assert report['training_continuation_diagnostic']['full_lead'] == .5
    assert report['perturbed_losing_cases'][0]['rival'] == 'perturbed'
    assert report['adaptation']['switches'] == 1
    assert report['adaptation']['constant'] is False
    assert (tmp_path/'result.json').exists()
