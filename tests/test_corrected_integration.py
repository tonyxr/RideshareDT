"""Small real-PyTorch workflow verification; not a dominance experiment."""
import csv
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from scripts.run_v26r_staged import BASE, command, environment


def without_learner_flags(cmd):
    """Reload commands inherit the saved learner contract instead of overriding it."""
    cmd = list(cmd)
    while '--learning_corrections' in cmd:
        cmd.remove('--learning_corrections')
    while '--learner_revision' in cmd:
        i = cmd.index('--learner_revision')
        del cmd[i:i + 2]
    return cmd


@pytest.mark.parametrize('lead_weight,rollout', [(0.5, 180), (0.6, 30)])
def test_corrected_stages_learning_logging_and_final_policy(tmp_path, lead_weight, rollout):
    out = tmp_path / 'corrected'
    out.mkdir()
    cmd = command(out, days=192, integration=True, lead_weight=lead_weight,
                  rollout_transitions=rollout, update_days=1) + ['--learning_corrections', '--learner_revision', '2']
    with (out / 'run.log').open('w') as log:
        process = subprocess.run(cmd, cwd=BASE, env=environment(), stdout=log,
                                 stderr=subprocess.STDOUT, timeout=900)
    run_log = (out / 'run.log').read_text()
    assert process.returncode == 0, run_log[-6000:]
    with (out / 'chicago_training_diagnostics.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 192
    # The run's own reward contract: the named profit preset fixes the share
    # weight at 0 and the command does not request the temporal reward.
    run_reward = json.loads((out / 'chicago_run_config.json').read_text())['platform_mdp']['active_reward']
    assert run_reward['market_share_competitiveness_weight'] == 0.0
    assert run_reward['temporal_competitive_reward'] is False
    assert run_reward['opponent_profit_lead_weight'] == lead_weight
    stage_weights = run_reward['stage_economic_weights']
    control_bounds = {
        'market': ((1.00, .55), (1.35, .90), (.25, .12)),
        'static': ((.75, .40), (1.00, .65), (.16, .07)),
        'catchup': ((.60, .30), (.85, .45), (.12, .04)),
        'profit': ((.45, .15), (.65, .25), (.08, .01)),
    }
    for stage, days in (
        ('market', 12), ('static', 12), ('catchup', 48), ('profit', 120)
    ):
        selected = [r for r in rows if r['training_stage_opponent_regime'] == stage]
        assert len(selected) == days
        rates = [float(r['ppo_lr']) for r in selected]
        lr_scales, entropy_scales, exploration_rates = control_bounds[stage]
        np.testing.assert_allclose(
            [rates[0], rates[-1]],
            [max(5e-5, 3e-4 * lr_scales[0]),
             max(5e-5, 3e-4 * lr_scales[1])],
            atol=1e-12,
        )
        assert np.all(np.diff(rates) <= 0)
        updates = [r for r in selected if r['ppo_update_performed'].lower() == 'true']
        if not updates:
            # The final stage is intentionally quarantined when it cannot meet
            # the configured rollout threshold: learning from an incomplete
            # opponent-population cycle would bias the saved policy.
            assert stage == 'profit'
            assert rollout > days
            assert '[PPO population quarantine]' in run_log
        for row in updates:
            assert float(row['ppo_behavior_replay_max_logp_error']) < 2e-5
            assert row['ppo_critic_target_contract'] == 'discounted_reward_gae_lambda_return'
            assert row['ppo_stopped_early_kl'].lower() == 'false'
        for row in selected:
            assert row['ppo_entropy_schedule'] == 'stage_progress_cosine_base_to_min'
            assert row['ppo_critic_reward_regime'] == stage
            assert float(row['ppo_critic_value_scale']) == pytest.approx(1000.)
            progress = float(row['ppo_stage_progress'])
            cosine = .5 * (1 + np.cos(np.pi * progress))
            # Revisions 1-3 retain their archived entropy contract; the new
            # explicit per-stage entropy scaling is revision-4-only.
            assert float(row['ppo_ent_coeff']) == pytest.approx(.01)
            exploration = exploration_rates[1] + (
                exploration_rates[0] - exploration_rates[1]
            ) * cosine
            assert float(row['behavior_exploration_rate']) == pytest.approx(exploration)
            if row.get('validation_critic_reward_regime'):
                assert row['validation_critic_reward_regime'] == stage
            weights = stage_weights[stage]
            own_weight = weights['own_profit']
            rival_weight = weights['profit_lead']
            assert float(row['reward_own_profit_component']) == pytest.approx(own_weight * float(row['avg_profit_per_request']) / 2)
            assert float(row['reward_profit_advantage_component']) == pytest.approx(rival_weight * float(row['reward_profit_advantage_per_request']) / 2)
            expected = sum(float(row[k]) for k in ('reward_own_profit_component',
                           'reward_profit_advantage_component',
                           'reward_durable_profit_advantage_component',
                           'reward_adaptive_profit_residual_component',
                           'reward_market_share_advantage_component'))
            assert 'reward_profit_deficit_component' not in row
            assert abs(float(row['reward_base']) - expected) < 1e-8
    import torch
    final = torch.load(out / 'policy.final_training.pt', map_location='cpu', weights_only=False)
    evaluated = torch.load(out / 'policy.pt', map_location='cpu', weights_only=False)
    assert final['credit_assignment_config']['learning_corrections'] == 1
    assert final['credit_assignment_config']['learner_revision'] == 2
    assert final['normalizer_state']['learner_schedule']['stage_progress'] == 1.
    assert final['reward_mechanism'] == 'staged_two_term_profit_v29_corrected'
    assert final['mdp_config']['active_reward']['opponent_profit_lead_weight'] == lead_weight
    for key, tensor in final['network_state'].items():
        assert torch.equal(tensor, evaluated['network_state'][key]), key
    cfg = json.loads((out / 'chicago_run_config.json').read_text())
    assert cfg['benchmark_evaluation']['comparison_checkpoint_day'] == 192
    # Reject legacy weights explicitly; never substitute an older policy.
    legacy_cmd = command(out / 'rejected', days=0, integration=True) + [
        '--learning_corrections', '--trained_model_in', str(BASE.parent / 'rideshare_v26r/runs/multicity_20000_20261014/chicago/policy.final_training.pt')]
    # This assertion isolates the learner contract; grid conflicts have their
    # own tests and must not mask the intended rejection.
    i = legacy_cmd.index('--tariff_targets')
    del legacy_cmd[i:i+2]
    rejected = subprocess.run(legacy_cmd, cwd=BASE, env=environment(), capture_output=True, text=True, timeout=60)
    assert rejected.returncode != 0
    assert 'conflicts with the saved policy contract' in rejected.stderr
    # An earlier deficit-penalized policy cannot be reinterpreted as equal-weight.
    incompatible = dict(final, reward_mechanism='staged_profit_lead_deficit_v26r_corrected')
    incompatible_path = out / 'incompatible_reward.pt'
    torch.save(incompatible, incompatible_path)
    incompatible_cmd = without_learner_flags(command(out / 'wrong_reward', days=0, integration=True,
                                                     lead_weight=lead_weight)) + [
        '--trained_model_in', str(incompatible_path), '--eval_only']
    mismatch = subprocess.run(incompatible_cmd, cwd=BASE, env=environment(), capture_output=True,
                              text=True, timeout=60)
    assert mismatch.returncode != 0
    assert 'reward mechanism does not match runtime' in mismatch.stderr
    # Reload under the saved contract, with no retraining or explicit correction
    # flag: the checkpoint must recover its observation/reward semantics.
    reload_out = out / 'reload'
    reload_out.mkdir()
    reload_cmd = without_learner_flags(command(reload_out, days=0, integration=True)) + [
        '--trained_model_in', str(out / 'policy.final_training.pt'), '--eval_only']
    # Omit the explicit CLI default so reload must recover the saved weight.
    i = reload_cmd.index('--opponent_profit_lead_weight')
    del reload_cmd[i:i+2]
    with (reload_out / 'run.log').open('w') as log:
        loaded = subprocess.run(reload_cmd, cwd=BASE, env=environment(), stdout=log,
                                stderr=subprocess.STDOUT, timeout=900)
    assert loaded.returncode == 0, (reload_out / 'run.log').read_text()[-6000:]
    saved = json.loads((reload_out / 'chicago_run_config.json').read_text())
    assert saved['active_reward']['opponent_profit_lead_weight'] == lead_weight
    # The short eval-only command intentionally does not request the benchmark
    # suite. Check benchmark diagnostics from the training fixture above.
    report = cfg['benchmark_evaluation']
    for name, summary in report['rollout_summaries'].items():
        assert report['mode_own_objective_score'][name] == summary['objective_reward_mean']
        assert report['mode_rival_objective_score'][name] == summary['rival_objective_reward_mean']
    assert np.isfinite(report['heldout_critic_lambda_mae'])
    assert saved['loaded_trained_model_metadata']['training_days'] == 192
    with (reload_out / 'chicago_evaluation_diagnostics.csv').open() as f:
        evaluation = list(csv.DictReader(f))
    assert len(evaluation) == 12
    for row in evaluation:
        expected = sum(float(row[k]) for k in ('reward_own_profit_component',
                       'reward_profit_advantage_component',
                       'reward_durable_profit_advantage_component',
                       'reward_adaptive_profit_residual_component',
                       'reward_market_share_advantage_component'))
        assert 'reward_profit_deficit_component' not in row
        assert abs(float(row['reward_base']) - expected) < 1e-8
