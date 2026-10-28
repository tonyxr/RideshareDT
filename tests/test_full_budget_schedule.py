"""Budget schedules and evaluation wiring, not policy-performance gates."""
import ast
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from platform_mdp import TrainingStageScheduler
from scripts.run_v26r_multicity import stage_days
from scripts.run_v26r_staged import BASE
from test_delayed_credit_revision4 import agent


@pytest.mark.parametrize('days', [1, 7, 300, 5000, 20000, 27001])
def test_rates_follow_actual_stage_budget_at_every_boundary(days):
    scheduler = TrainingStageScheduler(stage_fractions=(.1, .108, .792))
    model = agent()
    previous = {}
    counts = [0, 0, 0]
    for d in range(days):
        first, stop = scheduler.stage_window(d, days)
        phase = scheduler.stage_progress(d, days)
        stage = scheduler.stage_at(d/days).opponent_regime
        counts[('market','static','dynamic').index(stage)] += 1
        assert first <= d < stop <= days
        assert phase == (d-first)/max(1, stop-first-1)
        if d in {first, stop-1, (first+stop-1)//2}:
            model.set_stage_progress(phase)
            assert model.curr_lr == pytest.approx(5e-5+2.5e-4*.5*(1+np.cos(np.pi*phase)))
            assert model.ent_coeff == pytest.approx(
                model.min_ent_coeff
                + (model.base_ent_coeff-model.min_ent_coeff)
                * .5*(1+np.cos(np.pi*phase))
            )
            assert model.critic_opt.param_groups[0]['lr'] == pytest.approx(3*model.curr_lr)
            assert model.exploration_rate == 0
        if stage in previous:
            assert phase > previous[stage]
        previous[stage] = phase
    assert counts == stage_days(days) and sum(counts) == days


def test_budget_extension_recomputes_phase_without_resetting_optimizer():
    model = agent(); optimizer = model.actor_opt
    scheduler = TrainingStageScheduler(mode='direct')
    model.set_stage_progress(scheduler.stage_progress(99, 100))
    assert model.curr_lr == model.min_lr
    model.set_stage_progress(scheduler.stage_progress(99, 300))
    assert model.curr_lr > 2e-4 and model.ent_coeff > .03
    assert model.actor_opt is optimizer and model.update_calls == 0
    assert stage_days(20000) == [2000, 2160, 15840]


def test_sweep_cli_initializes_pools_and_honors_explicit_opponent_cadence():
    tree = ast.parse((BASE/'Core.py').read_text())
    branch = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
        and ast.unparse(n.test) == 'args.policy_sweep')
    refresh = next(i for i, n in enumerate(branch.body) if '_refresh_profile_pool' in ast.unparse(n))
    call_index = next(i for i, n in enumerate(branch.body) if 'core.run_policy_sweep' in ast.unparse(n))
    assert refresh < call_index
    call = next(n for n in ast.walk(branch.body[call_index]) if isinstance(n, ast.Call)
                and ast.unparse(n.func) == 'core.run_policy_sweep')
    value = next(k.value for k in call.keywords if k.arg == 'opponent_interval_steps')
    expression = compile(ast.Expression(value), '<cadence>', 'eval')
    argument = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute) and n.func.attr == 'add_argument'
        and n.args and isinstance(n.args[0], ast.Constant)
        and n.args[0].value == '--firm2_action_interval_days')
    parser = argparse.ArgumentParser()
    parser.add_argument(*[ast.literal_eval(a) for a in argument.args],
        **{k.arg:eval(compile(ast.Expression(k.value), '<arg>', 'eval')) for k in argument.keywords})
    for option in ('--firm2_action_interval_steps', '--firm2_action_interval_days'):
        for value, expected in ((6,6), (3,3), (0,1)):
            args = parser.parse_args([option,str(value)])
            assert vars(args) == {'firm2_action_interval_days':value}
            assert eval(expression, {'args':args}) == expected


def test_figure_pipeline_preserves_inputs_and_marks_missing_metrics(tmp_path):
    from scripts.v26r_figures import training, standard, gallery
    def write(name, rows):
        with (tmp_path/name).open('w', newline='') as f:
            writer = csv.DictWriter(f, list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write('toy_training_diagnostics.csv', [dict(batch=i, training_stage='market', avg_reward=i-2,
          ppo_update_performed=i%2==0, ppo_lr=.0003, ppo_ent_coeff=.04) for i in range(4)])
    write('toy_episodes.csv', [dict(end_day=4, stage='market', reward_mean=-.5)])
    write('toy_evaluation_diagnostics.csv', [dict(day=i, reward=-i, rl_profit=1, heuristic_profit=2) for i in range(4)])
    write('toy_benchmark_summary.csv', [dict(benchmark='rule', firm1_profit_per_request=1,
          firm2_profit_per_request=2, firm1_completed_share=.3, firm2_completed_share=.4,
          advantage=-1, final_quarter_advantage=-2)])
    before = {p.name:p.read_bytes() for p in tmp_path.glob('*.csv')}
    training(tmp_path,'toy'); standard(tmp_path,'toy')
    assert {p.name:p.read_bytes() for p in tmp_path.glob('*.csv')} == before
    assert json.loads((tmp_path/'figures/training_coverage.json').read_text())['missing_metrics']
    assert list(tmp_path.rglob('*.png')) and list(tmp_path.rglob('*.pdf'))
    assert (tmp_path/'figures.html').exists()
    fixture = tmp_path/'preflight'; fixture.mkdir()
    (fixture/'synthetic.png').write_bytes(b'not production evidence')
    gallery(tmp_path)
    assert all('preflight' not in p for p in json.loads((tmp_path/'figure_index.json').read_text())['figures'])


def test_static_selection_supports_profit_and_unchanged_reward_objective():
    from scripts.v26r_case_matrix import select
    sweep = {'traces':{f'static:{m}':[dict(rollout='rival__opening_1',step=step,
                 profit_per_request=p,reward=r) for step in range(60,240)]
                 for m,p,r in [(1,4,1),(2,3,2)]}}
    assert select(sweep,'static:','rival',10,40) == 'static:1'
    assert select(sweep,'static:','rival',10,40,column='reward') == 'static:2'


def test_corner_summary_and_figures_keep_losing_cells_and_objective_baseline(tmp_path, monkeypatch):
    from scripts import v26r_case_matrix as matrix
    from scripts.v26r_figures import corners
    monkeypatch.setattr(matrix, 'CASES', {'fixture':dict(role='test', burnin=0, onset=1, end=2)})
    traces = {}
    for spec, profit, reward in [('static:1',3,1), ('static:2',2,2),
                                  ('rl_actor',1,0), ('myopic_greedy',2.5,1.5)]:
        traces[spec] = [dict(rollout=f'{rival}__opening_1', step=step,
            profit_per_request=profit, rival_profit_per_request=4, profit_lead=profit-4,
            own_share=.3, rival_share=.5, own_market_share=.375, share_lead=-.2,
            reward=reward, firm1_level=1.1, firm2_level_effective=1.2, stock_firm1=1,
            loyalty_mean_firm1=.5) for rival in ('A','B') for step in range(24)]
    data = dict(meta=dict(horizon_steps=24, steps_per_day=6,
                rollouts=[[f'{r}__opening_1',r] for r in ('A','B')]), traces=traces)
    for seed in ('val','eval'):
        path = tmp_path/'toy/fixture'/seed; path.mkdir(parents=True)
        (path/'sweep.json').write_text(json.dumps(data))
    out = tmp_path/'toy/matrix'
    report = matrix.summarize(tmp_path, 'toy', out)
    assert len(report['late_profit_losing_cases']) == 2
    for row in report['late_profit_losing_cases']:
        assert row['selected_static'] == 'static:1'
        assert row['selected_objective_static'] == 'static:2'
        assert row['late_reward_minus_objective_static'] == -2
    corners(tmp_path/'toy', out/'matrix.csv', 'toy')
    assert len(list((out/'events').glob('*.png'))) == 2
    assert len(list((tmp_path/'toy/figures/corners').glob('*.pdf'))) == 10
    for trace in data['traces'].values():
        for row in trace:
            row.pop('reward')
    (tmp_path/'toy/fixture/val/sweep.json').write_text(json.dumps(data))
    with pytest.raises(ValueError, match='Missing reward selection evidence'):
        matrix.summarize(tmp_path, 'toy', tmp_path/'missing_objective')
