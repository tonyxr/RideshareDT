"""Independent 10k-day experiment; no restart, policy substitution or performance gate."""
import argparse
import collections
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tarfile
import traceback

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from model_contract import OPPONENT_PROFIT_LEAD_WEIGHT
ROOT = BASE.parent
PYTHON = ROOT / '.venv/bin/python'


def command(out, *, days=10000, integration=False, city="Chicago", slug="chicago",
            update_days=5, rollout_transitions=180, lead_weight=OPPONENT_PROFIT_LEAD_WEIGHT,
            seed=251012, training_scenario_manifest=None,
            actor_true_shortage_flag=False):
    """City-specific validation warm start and shared extended action/baseline grid."""
    from scripts.v26r_cases import STATIC_GRID
    grid = ','.join(map(str, STATIC_GRID))
    fractions = (
        ('0.1', '0.1', '0.3', '0.5')
        if not integration
        else ('0.0625', '0.0625', '0.25', '0.625')
    )
    customers = 5 if integration else 50
    args = [str(PYTHON), '-u', str(BASE/'Core.py'),
        '--market', city, '--choice_mode', 'cognitive', '--firm1_mode', 'RL',
        '--firm2_mode', 'adaptive_best_response_memory', '--run_experiment',
        '--credit_assignment_v25', '--v26', '--deterministic_experiment_seed', '--deterministic_torch',
        '--learning_corrections', '--learner_revision', '4',
        '--seed', str(seed), '--experiment_seed', str(seed), '--policy_objective', 'profit_maximization',
        '--stage_specific_rewards', '--opponent_profit_lead_weight', str(lead_weight),
        '--market_share_competitiveness_weight', '0.0',
        '--pool', '300' if integration else '2000',
        '--threshold_profile_source', 'cached', '--threshold_cache_path',
        str(ROOT/f'artifacts/v26_confirmatory/thresholds/{slug}/llm_numeric_primary.jsonl'),
        '--calibration_json', str(ROOT/f'data/processed/{slug}/calibration.json'),
        '--gpt_threshold_mode', 'llm_numeric', '--train_timesteps', str(days),
        '--train_customers', str(customers), '--train_steps_per_day', '6',
        '--firm1_action_interval_steps', '6', '--firm2_action_interval_steps', '6',
        '--ppo_update_interval_days', str(update_days), '--ppo_min_rollout_transitions', str(rollout_transitions),
        '--ppo_batch_size', '64', '--ppo_update_epochs', '8',
        '--training_stage_market_fraction', fractions[0], '--training_stage_static_fraction', fractions[1],
        '--training_stage_dynamic_fraction', fractions[2], '--training_stage_profit_fraction', fractions[3],
        '--training_opening_tariff_randomization', '0.15',
        '--checkpoint_validation_interval_days', '60', '--checkpoint_validation_horizon', '60',
        '--checkpoint_validation_customers', str(customers), '--checkpoint_validation_openings', '2',
        '--eval_timesteps', '12' if integration else '120', '--eval_customers', str(customers),
        '--eval_policy_mode', 'argmax_raw',
        '--static_baseline_multipliers', grid,
        '--regime_market', '--shocks', 'calibrated', '--shock_intensity', '1.0', '--regime_features',
        '--tariff_targets', grid,
        '--exploration_commit_mean', '3', '--spillover_price_aware', '--rl_warm_start_target', '0' if integration else 'auto',
        '--supply_model', 'v2', '--driver_pay_basis', 'fare', '--loyalty_model', 'v2',
        '--loyalty_pool_size', str(customers * 6 * 6 // 10), '--persistence_features',
        '--training_catchup_opponent_modes', 'price_gap_catchup,price_gap_catchup_fast',
        '--training_profit_opponent_modes', 'profit_opt_pi_price_gap,profit_opt_mpc_grid,profit_opt_region_supply_demand,profit_opt_surge_driver_incentive,profit_opt_queue_service_threshold,profit_best_response_myopic,profit_best_response_forward,profit_best_response_anticipating',
        '--training_dynamic_episode_days', '30',
        '--profit_planner_sample_cap', '5' if integration else '32', '--ppo_gae_lambda', '0.90',
        '--report_prefix', os.path.relpath(out/slug, BASE),
        '--trained_model_out', str(out/'policy.pt'), '--trained_model_id', f'v26r-{slug}-seed{seed}-staged-{days}',
        '--model_registry_dir', str(out/'registry')]
    if training_scenario_manifest:
        args += ['--training_scenario_manifest', str(Path(training_scenario_manifest).resolve())]
    if actor_true_shortage_flag:
        args.append('--actor_true_shortage_flag')
    if not integration:
        args.append('--benchmark_suite')
    return args


def source_hashes():
    paths = list(BASE.glob('*.py')) + list((BASE/'scripts').glob('*.py'))
    paths += list((BASE/'scripts').glob('*.command'))
    paths += list((BASE/'tests').glob('*.py'))
    paths += list((BASE/'calibration_presets').glob('*.py'))
    paths += list((BASE/'config').rglob('*.json'))
    paths += list((BASE/'scenarios').rglob('*.json'))
    return {str(p.relative_to(BASE)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths) if not p.name.startswith('._')}


def environment():
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               KMP_USE_SHM='0', PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg')
    env.pop('PYTHONPATH', None)
    return env


def summarize(out, *, training_required=True, expected_days=10000, slug="chicago", seed=251012):
    def rows(name):
        with (out/name).open() as f:
            return list(csv.DictReader(f))
    def number(row, key):
        value = float(row[key])
        if not math.isfinite(value):
            raise ValueError(f'Nonfinite {key}')
        return value
    train = rows(f'{slug}_training_diagnostics.csv') if training_required else []
    evaluation = rows(f'{slug}_evaluation_diagnostics.csv')
    benchmarks = rows(f'{slug}_benchmark_summary.csv')
    if (training_required and len(train) != expected_days) or (len(evaluation), len(benchmarks)) != (120, 5):
        raise ValueError('Incomplete training/evaluation evidence')
    cfg = json.loads((out/f'{slug}_run_config.json').read_text())
    active_reward = cfg['platform_mdp']['active_reward']
    expected_reward_contract = {
        'temporal_competitive_reward': False,
        'market_share_competitiveness_weight': 0.0,
    }
    for key, expected in expected_reward_contract.items():
        actual = active_reward.get(key)
        if isinstance(expected, bool):
            valid = bool(actual) is expected
        else:
            valid = math.isclose(float(actual), expected, abs_tol=1e-12)
        if not valid:
            raise ValueError(f'Reward contract mismatch for {key}: {actual!r}')
    expected_components = [
        'own_profit_per_request', 'rival_profit_per_request',
    ]
    if cfg['credit_assignment'].get('value_components') != expected_components:
        raise ValueError('Saved component critic does not match the two-term reward')
    if not training_required:
        metadata = cfg.get('loaded_trained_model_metadata') or {}
        if metadata.get('training_days') != expected_days or metadata.get('comparison_checkpoint_day') != expected_days:
            raise ValueError('Recovery must use the requested final policy')
    bench = cfg['benchmark_evaluation']
    if bench['comparison_checkpoint_day'] != expected_days:
        raise ValueError('Evaluation did not use the requested final day')
    lead = [number(r, 'reward_profit_advantage_per_request') for r in evaluation]
    actual_openings = [row for name, row in bench['rollout_summaries'].items()
                       if name.startswith('actual_firm2__')]
    if not actual_openings:
        raise ValueError('Missing paired actual-rival opening results')
    cases = [dict(rival='actual_training_rival_paired', openings=len(actual_openings),
        full=statistics.mean(number(r, 'own_profit_per_request')-number(r, 'rival_profit_per_request') for r in actual_openings),
        final_quarter=statistics.mean(number(r, 'late_objective_advantage') for r in actual_openings))]
    cases += [dict(rival=r['benchmark'], full=number(r, 'advantage'),
                   final_quarter=number(r, 'final_quarter_advantage')) for r in benchmarks]
    decisions = [r['action_label'] for r in evaluation if r['action_label'] != 'no_decision']
    if not decisions:
        raise ValueError('Missing evaluation decisions')
    counts = collections.Counter(decisions)
    updates = [r for r in train if r['ppo_update_performed'].lower() in ('true', '1')]
    validations = [r for r in train if r.get('validation_critic_td_samples', '') not in ('', 'nan')]
    def rollout_leads(report):
        summaries = report['rollout_summaries']
        if not summaries:
            raise ValueError('Missing rollout summaries')
        return [dict(rival=name,
                     full=number(row, 'own_profit_per_request') - number(row, 'rival_profit_per_request'),
                     final_quarter=number(row, 'late_objective_advantage'))
                for name, row in summaries.items()]
    extra_cases = rollout_leads(bench)
    perturbed = rollout_leads(cfg['v26_ood_evaluation'])
    result = dict(city=slug, training_seed=int(seed), training_days=len(train) if training_required else expected_days,
        training_days_this_process=len(train), training_history_available=bool(train),
        training_continuation_diagnostic=dict(full_lead=statistics.mean(lead),
            final_quarter_lead=statistics.mean(lead[-30:]),
            scope='unpaired continuation, not the headline actual-rival comparison'),
        stage_days=dict(collections.Counter(r['training_stage_opponent_regime'] for r in train)),
        updates_by_stage=dict(collections.Counter(r['training_stage_opponent_regime'] for r in updates)),
        configured_min_rollout_transitions=cfg.get('args', {}).get('ppo_min_rollout_transitions'),
        final_checkpoint_day=bench['comparison_checkpoint_day'],
        checkpoint_status=bench['comparison_checkpoint_status'],
        adaptation=dict(action_counts=dict(counts), constant=len(counts)==1,
            switches=sum(a != b for a, b in zip(decisions, decisions[1:])),
            dominant_action_fraction=max(counts.values())/len(decisions)),
        latest_validation_critic=({k:validations[-1][k] for k in ('batch',
            'validation_critic_td_explained_variance', 'validation_critic_td_mae',
            'validation_critic_td_samples')} if validations else None),
        evaluated_critic_ev=bench.get('heldout_critic_td_explained_variance'),
        profit_units='own-minus-rival contribution profit, dollars per reference request (base volume normalization)',
        profit_leads=cases, losing_cases=[r for r in cases if r['full'] < 0 or r['final_quarter'] < 0],
        all_six_late_leads_positive=all(r['final_quarter'] > 0 for r in cases),
        all_evaluation_opening_leads=extra_cases,
        additional_losing_openings=[r for r in extra_cases if r['full'] < 0 or r['final_quarter'] < 0],
        perturbed_profit_leads=perturbed,
        perturbed_losing_cases=[r for r in perturbed if r['full'] < 0 or r['final_quarter'] < 0],
        benchmark_adaptation={k:bench[k] for k in ('action_switch_rate', 'action_diversity',
            'mode_tail_action_switch_rate', 'mode_tail_action_diversity')},
        benchmark_rollouts=bench.get('rollout_summaries'),
        perturbed_rival_rollouts=(cfg.get('v26_ood_evaluation') or {}).get('rollout_summaries'),
        static_myopic_comparisons={k:v.get('comparison') for k,v in
            (cfg.get('v26_baseline_evaluation') or {}).items()},
        limitations=['Single training seed, not replicated training.',
            '120 market-step evaluation (20 days), final quarter = 5 days.',
            'Reward and benchmark fixes changed; not a training-budget-only comparison.',
            'Action changes are not proof of useful adaptation; positive reward is not dominance.',
            'Original day-3000 weights are missing; no replacement evaluation was performed.'])
    (out/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    (out/'result.txt').write_text(f'FINAL DAY-{expected_days} POLICY; NO CHECKPOINT FALLBACK\n' +
        json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--city-calibration-artifact', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Refuse overwriting or implicit retries. This is artifact protection, not
    # a performance gate and never chooses or modifies a policy.
    with (out/'started.json').open('x') as f:
        json.dump(dict(pid=os.getpid(), started_at=datetime.datetime.now().isoformat()), f)
    def status(value):
        (out/'status.txt').write_text(value + '\n')
        print(datetime.datetime.now().isoformat(), value, flush=True)
    try:
        approved = json.loads((out/'verified_sources.json').read_text())
        hashes = source_hashes()
        if approved != hashes:
            raise RuntimeError('Source differs from tested version; no job started')
        with tarfile.open(out/'source_snapshot.tar.gz', 'w:gz') as archive:
            for name in hashes:
                archive.add(BASE/name, arcname=name)
        cmd = command(out) + ['--learning_corrections']
        cmd += ['--city_calibration_artifact', str(args.city_calibration_artifact.resolve())]
        (out/'command.json').write_text(json.dumps(cmd, indent=2))
        data_paths = [ROOT/'artifacts/v26_confirmatory/thresholds/chicago/llm_numeric_primary.jsonl',
                      ROOT/'data/processed/chicago/calibration.json']
        (out/'manifest.json').write_text(json.dumps(dict(
            seed=251012, training_days=10000, stage_days=[1000, 1000, 3000, 5000],
            source_hashes=hashes, python=sys.version, python_executable=str(PYTHON),
            data_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in data_paths},
            primary_policy='final day 10000, no checkpoint fallback',
            old_day3000_evaluation='unavailable: final weights not saved'), indent=2))
        status('training 10000 days, then final-policy benchmark evaluation')
        with (out/'run.log').open('x') as log:
            child = subprocess.Popen(cmd, cwd=BASE, env=environment(), stdout=log, stderr=subprocess.STDOUT)
            (out/'training_pid.txt').write_text(str(child.pid) + '\n')
            code = child.wait()
        (out/'training_exit_status.txt').write_text(str(code) + '\n')
        if code:
            raise RuntimeError(f'Training/evaluation exited {code}; no restart')
        if source_hashes() != hashes:
            raise RuntimeError('Source changed during run; results require review')
        status('summarizing adaptation, critic learning and full/late profit dominance')
        summarize(out)
        status('completed; see result.txt (performance never selects a different policy)')
        (out/'exit_status.txt').write_text('0\n')
    except BaseException:
        traceback.print_exc()
        status('failed; artifacts preserved; no restart or policy substitution')
        (out/'exit_status.txt').write_text('1\n')
        raise


if __name__ == '__main__':
    main()
