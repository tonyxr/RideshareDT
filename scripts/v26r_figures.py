"""Read-only plots of saved evidence; never trains, selects or alters a policy."""
import csv
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from diagnostic_logging import atomic_json


def read_csv(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def number(row, key):
    value = row.get(key)
    return float(value) if value not in (None, '', 'None') else float('nan')


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        fig.savefig(path.with_suffix('.'+extension), dpi=170, bbox_inches='tight')
    plt.close(fig)


def series(rows, key, path, title, *, xkey='batch', group='training_stage', updates=False):
    sample = [r for r in rows if not updates or str(r.get('ppo_update_performed')).lower() == 'true']
    if not any(np.isfinite(number(r, key)) for r in sample):
        return False
    fig, ax = plt.subplots(figsize=(10, 4))
    for label in dict.fromkeys(r.get(group, 'all') for r in sample):
        part = [r for r in sample if r.get(group, 'all') == label]
        x = np.array([number(r, xkey) + (1 if xkey == 'batch' else 0) for r in part])
        y = np.array([number(r, key) for r in part])
        ax.plot(x, y, '.-', lw=.65, ms=2, alpha=.65, label=label)
    ax.set(title=title, xlabel='Training day' if xkey != 'day' else 'Evaluation period', ylabel=key)
    ax.grid(alpha=.2); ax.legend(fontsize=7)
    note = 'One trained seed; descriptive trajectory.'
    if xkey != 'day':
        note += ' Reward definitions differ by stage.'
    fig.text(.01, -.02, note, fontsize=8)
    save(fig, path)
    return True


def training(out, slug):
    out = Path(out); dest = out/'figures/training'
    rows = read_csv(out/f'{slug}_training_diagnostics.csv')
    daily = ('avg_reward', 'reward_base', 'avg_profit_per_request', 'avg_firm2_profit_per_request',
        'avg_profit_advantage_per_request', 'avg_market_share', 'avg_fulfillment_rate',
        'avg_wait_minutes', 'raw_policy_entropy', 'raw_policy_top_prob', 'raw_policy_hold_prob',
        'sampled_not_raw_top_rate', 'ppo_lr', 'ppo_critic_lr', 'ppo_ent_coeff',
        'behavior_exploration_rate', 'driver_stock_firm1', 'driver_stock_firm2',
        'validation_mean_profit_advantage', 'validation_worst_profit_advantage',
        'validation_critic_td_explained_variance', 'validation_critic_td_mae')
    update = ('ppo_explained_variance', 'ppo_critic_rmse', 'ppo_critic_own_rmse',
        'ppo_critic_rival_rmse', 'ppo_value_loss', 'ppo_policy_loss', 'ppo_approx_kl',
        'ppo_clipfrac', 'ppo_critic_grad_norm', 'ppo_state_action_sensitivity',
        'ppo_actor_return_std', 'ppo_actor_raw_advantage_std', 'ppo_rollout_size',
        'ppo_completed_rollout_episodes', 'ppo_behavior_replay_max_logp_error')
    missing = []
    for key in daily + update:
        if not series(rows, key, dest/key, f'{slug}: {key}', updates=key in update):
            missing.append(key)
    for stage in dict.fromkeys(r['training_stage'] for r in rows):
        part = [r for r in rows if r['training_stage'] == stage]
        for key in ('avg_reward', 'raw_policy_entropy', 'avg_profit_advantage_per_request'):
            series(part, key, dest/f'{stage}_{key}', f'{slug}: {stage} / {key}')
    episodes = read_csv(out/f'{slug}_episodes.csv')
    for key in ('reward_mean', 'entropy_mean', 'actor_lr_mean', 'critic_lr_mean',
                'entropy_coefficient_mean', 'sampled_non_greedy_mean',
                'profit_lead_mean', 'critic_ev_mean', 'critic_loss_mean', 'ppo_kl_mean'):
        if not series(episodes, key, out/'figures/episodes'/key, f'{slug}: episode {key}',
                      xkey='end_day', group='stage'):
            missing.append('episode:'+key)
    atomic_json(out/'figures/training_coverage.json', dict(missing_metrics=missing,
        days=len(rows), episodes=len(episodes), interpretation='Missing metrics are not zero or a pass.'))
    gallery(out)


def standard(out, slug):
    out = Path(out)
    rows = read_csv(out/f'{slug}_evaluation_diagnostics.csv')
    for key in ('reward', 'rl_profit', 'heuristic_profit', 'reward_profit_advantage_per_request',
                'rl_market_share', 'policy_entropy', 'policy_top_prob', 'price_gap_f2_minus_f1'):
        series(rows, key, out/'figures/evaluation'/key, f'{slug}: {key}', xkey='day')
    rows = read_csv(out/f'{slug}_benchmark_summary.csv')
    for label, keys in [('profit', ('firm1_profit_per_request', 'firm2_profit_per_request')),
                        ('share_of_reference_requests', ('firm1_completed_share', 'firm2_completed_share')),
                        ('profit_leads', ('advantage', 'final_quarter_advantage'))]:
        fig, ax = plt.subplots(figsize=(11, 5)); x = np.arange(len(rows))
        for offset, key in zip((-.2, .2), keys):
            ax.bar(x+offset, [number(r, key) for r in rows], width=.4, label=key)
        ax.set_xticks(x, [r['benchmark'] for r in rows], rotation=25, ha='right')
        ax.axhline(0, color='black', lw=.7); ax.legend(fontsize=8)
        ax.set_title(f'{slug}: standard final-policy benchmarks / {label}')
        save(fig, out/'figures/benchmarks'/label)
    gallery(out)


def corners(out, matrix_path, slug):
    out = Path(out); rows = read_csv(matrix_path)
    rl = [r for r in rows if r['policy'] == 'rl_actor']
    cases = list(dict.fromkeys(r['case'] for r in rl)); rivals = sorted({r['rival'] for r in rl})
    for key in ('full_profit_per_request', 'late_profit_per_request', 'full_profit_lead',
        'late_profit_lead', 'full_own_market_share', 'late_own_market_share',
        'full_profit_per_request_minus_static', 'late_profit_per_request_minus_static',
        'full_own_market_share_minus_static', 'late_own_market_share_minus_static'):
        grid = np.array([[number(next(r for r in rl if r['case']==c and r['rival']==v), key)
                          for v in rivals] for c in cases])
        if not np.isfinite(grid).all():
            raise ValueError(f'Incomplete corner graph evidence: {slug}/{key}')
        fig, ax = plt.subplots(figsize=(12, 8)); lim = max(.01, float(np.max(np.abs(grid))))
        signed = 'lead' in key or 'minus_static' in key
        im = ax.imshow(grid, aspect='auto', cmap='RdBu' if signed else 'viridis',
                       vmin=-lim if signed else None, vmax=lim if signed else None)
        ax.set_xticks(range(len(rivals)), rivals, rotation=35, ha='right', fontsize=8)
        ax.set_yticks(range(len(cases)), cases, fontsize=8)
        for i in range(len(cases)):
            for j in range(len(rivals)):
                ax.text(j, i, f'{grid[i,j]:.3f}', ha='center', va='center', fontsize=7,
                        bbox=dict(facecolor='white', alpha=.65, edgecolor='none'))
        ax.set_title(f'{slug}: {key}\nFull excludes declared burn-in; late = final quarter; held-out openings')
        fig.colorbar(im, ax=ax); save(fig, out/'figures/corners'/key)
    gallery(out)


def gallery(out):
    out = Path(out)
    paths = sorted(relative for p in out.rglob('*.png')
                   if not {'logs', 'preflight'}.intersection((relative := p.relative_to(out)).parts))
    atomic_json(out/'figure_index.json', dict(figures=[str(p) for p in paths], png_count=len(paths),
        note='Every plot reflects saved evidence, including losses. Openings are not independent trained agents.'))
    items = ''.join(f'<li><a href="{html.escape(str(p), quote=True)}">{html.escape(str(p))}</a></li>' for p in paths)
    (out/'figures.html').write_text('<!doctype html><meta charset="utf-8"><title>Experiment figures</title>'
        '<h1>Saved experiment figures</h1><p>One trained seed per city. No outcome-based plot selection.</p><ul>'+items+'</ul>')
