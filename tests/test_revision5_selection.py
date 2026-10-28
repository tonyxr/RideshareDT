import json
import pytest
from evaluation_selection import select_constant, three_comparators
from scripts.v26r_cases import STATIC_GRID, oracle_specs, myopic_specs, CASES
from scripts.v26r_case_eval import best_static_from, spec_key


def trace(a, b):
    return [dict(rollout=f'{r}__opening_1', step=t, reward=value, profit_per_request=9-value)
            for r, value in [('A', a), ('B', b)] for t in range(12)]


def test_single_per_rival_and_hindsight_have_distinct_information_sets():
    val = {'static:0.6': trace(4, 0), 'static:1.2': trace(2, 5)}
    ev = {'static:0.6': trace(3, 1), 'static:1.2': trace(2, 4), 'schedule:x': trace(5, 2)}
    result = three_comparators(val, ev, start=0, end=2)
    assert result['global_constant'] == 'static:1.2'
    assert result['per_rival_constants'] == {'A': 'static:0.6', 'B': 'static:1.2'}
    assert result['hindsight_choices'] == {'A__opening_1': 'schedule:x', 'B__opening_1': 'static:1.2'}
    # Changing evaluation outcomes cannot affect either validation selection.
    ev['static:0.6'] = trace(-100, 100)
    changed = three_comparators(val, ev, start=0, end=2)
    assert changed['global_constant'] == result['global_constant']
    assert changed['per_rival_constants'] == result['per_rival_constants']


def test_no_hindsight_splicing_across_incompatible_state_trajectories():
    a, b = trace(0, 0), trace(0, 0)
    for row in a:
        row['reward'] = 10 if row['step'] % 2 else 0
    for row in b:
        row['reward'] = 0 if row['step'] % 2 else 10
    result = three_comparators({'static:1': a}, {'static:1': a, 'schedule:x': b}, start=0, end=2)
    assert all(row['hindsight_best_evaluated'] == 5 for row in result['tapes'].values())


def test_reporting_profit_and_late_lead_does_not_reselect_the_objective_policy():
    from evaluation_selection import selected_outcomes
    val = {'static:1': trace(3, 3), 'static:2': trace(1, 1)}
    for spec, rows in val.items():
        for row in rows:
            row.update(rival_profit_per_request=10., profit_lead=-4. if spec=='static:1' else -2., own_market_share=.4)
    report = three_comparators(val, val, start=0, end=2)
    rows = selected_outcomes(report, val, start=0, end=2)
    assert all(row['policy'] == 'static:1' for row in rows)
    assert all(row['late_profit_lead'] == -4 and row['full_profit_per_request'] == 6 for row in rows)


def test_undefined_market_share_is_reported_missing_without_weakening_selection_evidence():
    from evaluation_selection import selected_outcomes
    evidence = {'static:1': trace(3, 3)}
    for row in evidence['static:1']:
        row.update(rival_profit_per_request=2., profit_lead=1., own_market_share=.4)
    evidence['static:1'][0]['own_market_share'] = None
    report = three_comparators(evidence, evidence, start=0, end=2)
    rows = selected_outcomes(report, evidence, start=0, end=2)
    assert rows[0]['full_own_market_share'] == pytest.approx(.4)
    evidence['static:1'][0]['reward'] = None
    with pytest.raises(ValueError, match='Nonfinite reward'):
        three_comparators(evidence, evidence, start=0, end=2)


@pytest.mark.parametrize('defect', ['missing', 'duplicate', 'nan', 'wrong_column'])
def test_incomplete_or_invalid_evidence_never_passes(defect):
    traces = {'static:1': trace(1, 2), 'static:2': trace(2, 3)}
    rows = traces['static:2']
    if defect == 'missing': rows.pop()
    elif defect == 'duplicate': rows.append(rows[0])
    elif defect == 'nan': rows[0]['reward'] = float('nan')
    else: rows[0].pop('reward')
    with pytest.raises((ValueError, KeyError)):
        select_constant(traces, start=0, end=2)


def test_ties_are_deterministic_and_objective_is_explicit():
    a = {'static:2': trace(1, 1), 'static:0.6': trace(1, 1)}
    assert select_constant(a, start=0, end=2) == 'static:0.6'
    assert select_constant(dict(reversed(list(a.items()))), start=0, end=2) == 'static:0.6'
    a['static:2'] = trace(2, 2)
    assert select_constant(a, start=0, end=2) == 'static:2'
    assert select_constant(a, start=0, end=2, column='profit_per_request') == 'static:0.6'


def test_oracle_anchor_uses_calm_window_each_rival_and_validation_only(tmp_path):
    for spec, rows in [('static:0.6', trace(3, 1)), ('static:1.2', trace(1, 3))]:
        for row in rows:
            if row['step'] >= 6:
                row['reward'] = -row['reward'] * 100  # regime must not select calm anchor
        payload = dict(spec=spec, seed_set='val', steps_per_day=6, horizon_steps=12, traces=rows)
        (tmp_path/(spec_key(spec)+'.json')).write_text(json.dumps(payload))
    assert best_static_from(tmp_path, 0, rival='A', end=1) == .6
    assert best_static_from(tmp_path, 0, rival='B', end=1) == 1.2
    path = tmp_path/(spec_key('static:0.6')+'.json')
    payload = json.loads(path.read_text()); payload['seed_set'] = 'eval'
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='validation'):
        best_static_from(tmp_path, 0, rival='A', end=1)


def test_all_oracle_and_scripted_prices_share_expanded_rl_grid():
    assert min(STATIC_GRID) == .6 and .7 in STATIC_GRID
    for case in CASES.values():
        for base in STATIC_GRID:
            for spec in oracle_specs(case, base):
                table = json.loads(spec.split(':', 1)[1])
                assert all(entry[0] in STATIC_GRID for entry in table.values())
            for spec in myopic_specs(case, base).values():
                table = json.loads(spec.split(':', 1)[1])
                assert all(entry[0] in STATIC_GRID for entry in [table['default'], *table['days'].values()])
