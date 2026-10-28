#!/usr/bin/env python3
"""Revision 11 decisive test: does the deterministic actor raise price in a driver shortage?

Two NYC trainings (whole-tariff actions; reactive memory/aggressive rivals; shortage
curriculum; low final entropy; held learning rate), one with the diagnostic oracle
shortage flag and one with public signals only. Each final policy is then evaluated on
the scripted severe-shortage case (A1, fleet x0.35 on days 14-19) against the memory
rival on 10 held-out openings, beside static 1.1-1.6x.

Pre-registered pass criterion (deterministic raw-actor argmax):
  mean posted multiplier on shortage days minus calm days >= 0.15x, paired over
  openings, t >= 3.
"""
from __future__ import annotations
import json, os, subprocess, sys, time, statistics as st
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent
PY = str(ROOT / '.venv/bin/python')
GRID = '0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.8,2.0'
CAL = str(ROOT / 'artifacts/revision7_direct_calibration_20261018/nyc/city_calibration.json')
sys.path.insert(0, str(BASE / 'scripts')); sys.path.insert(0, str(BASE))


def train_cmd(out: Path, days: int, oracle: bool, smoke: bool) -> list:
    fr = ('0.05', '0.05', '0.88', '0.02')
    cmd = [PY, '-u', str(BASE / 'Core.py'),
        '--market', 'New York City', '--choice_mode', 'cognitive', '--firm1_mode', 'RL',
        '--firm2_mode', 'adaptive_best_response_memory', '--run_experiment',
        '--credit_assignment_v25', '--v26', '--deterministic_experiment_seed', '--deterministic_torch',
        '--learning_corrections', '--learner_revision', '4', '--seed', '251012', '--experiment_seed', '251012',
        '--policy_objective', 'profit_maximization', '--stage_specific_rewards',
        '--opponent_profit_lead_weight', '0.6', '--market_share_competitiveness_weight', '0.0',
        '--pool', '2000', '--threshold_profile_source', 'cached',
        '--threshold_cache_path', str(ROOT / 'artifacts/v26_confirmatory/thresholds/nyc/llm_numeric_primary.jsonl'),
        '--calibration_json', str(ROOT / 'data/processed/nyc/calibration.json'),
        '--gpt_threshold_mode', 'llm_numeric', '--train_timesteps', str(days),
        '--train_customers', '50', '--train_steps_per_day', '6',
        '--firm1_action_interval_steps', '6', '--firm2_action_interval_steps', '6',
        '--ppo_update_interval_days', '5', '--ppo_min_rollout_transitions', '30' if smoke else '60',
        '--ppo_batch_size', '64', '--ppo_update_epochs', '8',
        '--training_stage_market_fraction', fr[0], '--training_stage_static_fraction', fr[1],
        '--training_stage_dynamic_fraction', fr[2], '--training_stage_profit_fraction', fr[3],
        '--training_opening_tariff_randomization', '0.15',
        '--checkpoint_validation_interval_days', '600', '--checkpoint_validation_horizon', '60',
        '--checkpoint_validation_customers', '50', '--checkpoint_validation_openings', '1',
        '--eval_timesteps', '12' if smoke else '120', '--eval_customers', '50', '--eval_policy_mode', 'argmax_raw',
        '--static_baseline_multipliers', GRID, '--regime_market', '--shocks', 'calibrated',
        '--shock_intensity', '1.0', '--regime_features', '--tariff_targets', GRID,
        '--spillover_price_aware', '--rl_warm_start_target', 'auto',
        '--supply_model', 'v2', '--driver_pay_basis', 'fare', '--loyalty_model', 'v2',
        '--loyalty_pool_size', '180', '--persistence_features',
        '--training_catchup_opponent_modes', 'adaptive_best_response_memory,adaptive_best_response_aggressive',
        '--training_profit_opponent_modes', 'profit_best_response_myopic',
        '--training_dynamic_episode_days', '30', '--profit_planner_sample_cap', '32', '--ppo_gae_lambda', '0.90',
        '--whole_tariff_actions', '--stage_entropy_end_scale', '0.05', '--stage_lr_end_scale', '0.45',
        '--city_calibration_artifact', CAL,
        '--report_prefix', os.path.relpath(out / 'nyc', BASE),
        '--trained_model_out', str(out / 'policy.pt'), '--trained_model_id', f'r11-nyc-{"oracle" if oracle else "public"}-{days}',
        '--model_registry_dir', str(out / 'registry')]
    if oracle:
        cmd.append('--actor_true_shortage_flag')
    return cmd


def env():
    e = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
             KMP_USE_SHM='0', PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg')
    e.pop('PYTHONPATH', None)
    return e


def status(root: Path, **kw):
    path = root / 'status.json'
    data = json.loads(path.read_text()) if path.exists() else {}
    data.update(kw, time=time.time())
    path.write_text(json.dumps(data, indent=1))


def launch(cmd, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    (out / 'command.json').write_text(json.dumps(cmd))
    log = open(out / 'training.log', 'w')
    return subprocess.Popen(cmd, cwd=BASE, env=env(), stdout=log, stderr=subprocess.STDOUT)


def evaluate(policy: Path, out: Path, openings: int = 10) -> dict:
    from v26r_case_eval import run_specs
    specs = ['rl_actor', 'static:1.1', 'static:1.2', 'static:1.3', 'static:1.4', 'static:1.6']
    out.mkdir(parents=True, exist_ok=True)
    run_specs('New York City', 'nyc', 'A1_shortage_severe', specs, 'eval', openings, 'full', str(policy), str(out),
              customers=50, horizon_days=40, rivals=['adaptive_best_response_memory'], pool=2000)
    return analyze(out, specs)


def analyze(out: Path, specs) -> dict:
    import hashlib
    key = lambda s: hashlib.md5(s.encode()).hexdigest()[:12]
    result = {}
    per = {}
    for spec in specs:
        d = json.loads((out / f'{key(spec)}.json').read_text())
        spd = int(d['steps_per_day'])
        rows = d['traces']
        by = {}
        for r in rows:
            day = int(r['step']) // spd
            if day < 5:
                continue
            reg = str(r.get('regime', 'calm'))
            kind = ('shortage' if ('driver_shortage' in reg and 'recovery' not in reg)
                    else 'calm' if reg == 'calm' else 'other')
            lvl = r.get('firm1_level')
            b = by.setdefault(r['rollout'], {'shortage': [], 'calm': [], 'rs': [], 'rc': []})
            if kind in ('shortage', 'calm'):
                if lvl is not None:
                    b[kind].append(float(lvl))
                b['rs' if kind == 'shortage' else 'rc'].append(float(r.get('reward') or 0.0))
        per[spec] = by
    rl = per['rl_actor']
    diffs = [st.mean(v['shortage']) - st.mean(v['calm']) for v in rl.values() if v['shortage'] and v['calm']]
    m = st.mean(diffs); se = st.stdev(diffs) / len(diffs) ** 0.5 if len(diffs) > 1 else float('nan')
    result['rl_level_shortage_minus_calm'] = {'mean': m, 'se': se, 't': m / se if se and se > 0 else float('nan'), 'n': len(diffs)}
    result['rl_level'] = {'shortage': st.mean(x for v in rl.values() for x in v['shortage']),
                          'calm': st.mean(x for v in rl.values() for x in v['calm'])}
    for spec in specs:
        result.setdefault('reward_shortage_days', {})[spec] = st.mean(x for v in per[spec].values() for x in v['rs'])
        result.setdefault('reward_calm_days', {})[spec] = st.mean(x for v in per[spec].values() for x in v['rc'])
    statics = [s for s in specs if s.startswith('static:')]
    best_calm = max(statics, key=lambda s: result['reward_calm_days'][s])
    result['best_static_calm'] = best_calm
    gaps = []
    for ro, v in rl.items():
        b = per[best_calm].get(ro)
        if b and v['rs'] and b['rs']:
            gaps.append(st.mean(v['rs']) - st.mean(b['rs']))
    if len(gaps) > 1:
        g = st.mean(gaps); gse = st.stdev(gaps) / len(gaps) ** 0.5
        result['rl_minus_best_calm_static_on_shortage_days'] = {'mean': g, 'se': gse, 't': g / gse if gse > 0 else float('nan')}
    t = result['rl_level_shortage_minus_calm']
    result['pass'] = bool(t['mean'] >= 0.15 and t['t'] >= 3)
    return result


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    root = BASE / 'runs' / f'revision11_verify_{days}'
    root.mkdir(parents=True, exist_ok=True)
    status(root, phase='smoke')
    smoke = launch(train_cmd(root / 'smoke', 192, True, True), root / 'smoke')
    if smoke.wait() != 0:
        status(root, phase='failed', error='smoke training failed; see smoke/training.log'); return 1
    try:
        smoke_eval = evaluate(root / 'smoke' / 'policy.pt', root / 'smoke' / 'eval', openings=2)
        status(root, smoke_eval=smoke_eval)
    except BaseException as exc:
        status(root, phase='failed', error=f'smoke evaluation failed: {exc!r}'); return 1
    status(root, phase='training')
    jobs = {name: launch(train_cmd(root / name, days, name == 'oracle', False), root / name)
            for name in ('oracle', 'public')}
    codes = {name: p.wait() for name, p in jobs.items()}
    status(root, exit_codes=codes)
    results = {}
    for name, code in codes.items():
        if code != 0:
            results[name] = {'error': f'training exit {code}'}; continue
        try:
            results[name] = evaluate(root / name / 'policy.pt', root / name / 'eval')
        except BaseException as exc:
            results[name] = {'error': repr(exc)}
        status(root, results=results)
    status(root, phase='complete', results=results)
    return 0


if __name__ == '__main__':
    sys.exit(main())
