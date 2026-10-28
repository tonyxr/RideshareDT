#!/usr/bin/env python3
"""Revision 11b pre-launch checks (run on the Mac via the job runner).

isolation : two identical short trainings, branch credit ON (actor weight 0) vs OFF.
            Before the first PPO update that sees branch data, the real trajectories
            must be identical; any difference means a branch leaked into the real run.
            Also reloads the trained policy through the case-evaluation path.
probe     : branch every action (actor weight 0, so the policy stays at its warm start)
            and dump the paired branch values, to test whether the credit signal
            favours higher tariffs in shortages and responds to the rival's price.
"""
from __future__ import annotations
import json, os, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_revision11b_verify as R

BASE = R.BASE


def set_flag(cmd, flag, value=None, present=True):
    cmd = list(cmd)
    if flag in cmd:
        i = cmd.index(flag)
        if not present:
            del cmd[i:i + (1 if value is None and (i + 1 >= len(cmd) or cmd[i + 1].startswith('--')) else 2)]
            return cmd
        if value is not None:
            cmd[i + 1] = str(value)
        return cmd
    if present:
        cmd += [flag] + ([str(value)] if value is not None else [])
    return cmd


def short_cmd(out: Path, days: int, oracle: bool, credit: int, weight: float, tag: str):
    cmd = R.train_cmd(out, days, oracle, True)
    cmd = set_flag(cmd, '--counterfactual_day_credit', credit)
    cmd = set_flag(cmd, '--counterfactual_actor_weight', weight)
    cmd = set_flag(cmd, '--trained_model_id', f'r11b-check-{tag}')
    return cmd


def run_parallel(jobs):
    procs = {}
    for name, (cmd, out, extra_env) in jobs.items():
        out.mkdir(parents=True, exist_ok=True)
        (out / 'command.json').write_text(json.dumps(cmd))
        env = R.env(); env.update(extra_env)
        procs[name] = (subprocess.Popen(cmd, cwd=BASE, env=env, stdout=open(out / 'training.log', 'w'),
                                        stderr=subprocess.STDOUT), time.time())
    res = {}
    for name, (p, t0) in procs.items():
        code = p.wait(); res[name] = {'exit': code, 'seconds': round(time.time() - t0, 1)}
    return res


def live_rows(out: Path):
    rows = []
    for line in open(out / 'nyc_training_live.jsonl'):
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows


def isolation(root: Path):
    days = 40
    res = run_parallel({
        'on': (short_cmd(root / 'on', days, True, 6, 0.0, 'iso-on'), root / 'on', {'R11B_CF_DUMP': str(root / 'on' / 'cf_dump.jsonl')}),
        'off': (short_cmd(root / 'off', days, True, 0, 0.0, 'iso-off'), root / 'off', {}),
    })
    report = {'runs': res}
    if all(v['exit'] == 0 for v in res.values()):
        on, off = live_rows(root / 'on'), live_rows(root / 'off')
        mism = []
        compared = 0
        fields = ['avg_reward', 'reward_raw', 'avg_fulfillment_rate', 'last_action', 'raw_policy_top_action_label',
                  'share', 'profit_lead_mean', 'shock_block_counts']
        fields = [f for f in fields if f in on[0]]
        for a, b in zip(on, off):
            compared += 1
            for f in fields:
                if a.get(f) != b.get(f):
                    mism.append({'day': a['batch'], 'field': f, 'on': a.get(f), 'off': b.get(f)})
        first_cf_update = None
        report.update(days_compared=compared, rows_on=len(on), rows_off=len(off),
                      fields=fields, mismatches=mism[:20], n_mismatches=len(mism),
                      isolation_pass=(compared >= 30 and len(on) == len(off) and not mism))
        dump = [json.loads(l) for l in open(root / 'on' / 'cf_dump.jsonl')]
        report['branch_calls'] = len(dump)
        report['branch_seconds_mean'] = sum(d['seconds'] for d in dump) / max(1, len(dump))
        report['train_seconds'] = {k: v['seconds'] for k, v in res.items()}
        # reload through the case-evaluation harness (Core restores the contracts from the artifact)
        try:
            from v26r_case_eval import run_specs
            ev = root / 'on' / 'eval'
            run_specs('New York City', 'nyc', 'A1_shortage_severe', ['rl_actor', 'static:1.2'], 'eval', 1, 'full',
                      str(root / 'on' / 'policy.pt'), str(ev), customers=50, horizon_days=20,
                      rivals=['adaptive_best_response_memory'], pool=2000)
            report['reload_eval'] = 'ok'
        except BaseException as exc:
            report['reload_eval'] = f'FAILED: {exc!r}'
    (root / 'isolation_report.json').write_text(json.dumps(report, indent=1))


def probe(root: Path, days: int):
    res = run_parallel({
        'probe_oracle': (short_cmd(root / 'probe_oracle', days, True, 14, 0.0, 'probe'), root / 'probe_oracle',
                         {'R11B_CF_DUMP': str(root / 'probe_oracle' / 'cf_dump.jsonl')}),
    })
    (root / 'probe_report.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    mode = sys.argv[1]
    root = BASE / 'runs' / 'r11b_checks'
    root.mkdir(parents=True, exist_ok=True)
    if mode == 'isolation':
        isolation(root / 'isolation')
    elif mode == 'probe':
        probe(root, int(sys.argv[2]) if len(sys.argv) > 2 else 150)
