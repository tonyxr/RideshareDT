"""Explicit information sets for static and ex-post comparator selection.

Never splice per-step rewards from different policies: their future states differ.
"""
import math
from collections import defaultdict
from statistics import mean, stdev


def trajectory_scores(traces, *, start, end, spd=6, column='reward', rival=None,
                      allow_missing=False):
    grouped = defaultdict(list)
    for row in traces:
        name = row['rollout']
        if rival is not None and name.rsplit('__opening_', 1)[0] != rival:
            continue
        if start <= int(row['step']) / spd < end:
            if column not in row:
                raise ValueError(f'Missing {column} selection evidence in {name}')
            raw = row[column]
            value = None if raw is None else float(raw)
            if value is not None and not math.isfinite(value):
                value = None
            if value is None and not allow_missing:
                raise ValueError(f'Nonfinite {column}')
            grouped[name].append((int(row['step']), value))
    result = {}
    expected = set(range(math.ceil(start*spd), math.ceil(end*spd)))
    for name, rows in grouped.items():
        steps = [step for step, _ in rows]
        if len(steps) != len(set(steps)) or set(steps) != expected:
            raise ValueError(f'Missing or duplicate steps in {name}')
        observed = [value for _, value in rows if value is not None]
        result[name] = mean(observed) if observed else None
    return result


def select_constant(traces_by_spec, *, start, end, spd=6, column='reward', rival=None):
    scores = {spec: trajectory_scores(trace, start=start, end=end, spd=spd, column=column, rival=rival)
              for spec, trace in traces_by_spec.items() if spec.startswith('static:')}
    if not scores or any(not values for values in scores.values()):
        raise ValueError('Missing static selection evidence')
    tapes = [set(values) for values in scores.values()]
    if any(keys != tapes[0] for keys in tapes[1:]):
        raise ValueError('Static selection requires matched openings and rivals')
    def score(spec):
        groups = defaultdict(list)
        for name, value in scores[spec].items():
            groups[name.rsplit('__opening_', 1)[0]].append(value)
        return mean(mean(group) for group in groups.values())
    # A numeric, documented tie rule; directory insertion order is irrelevant.
    return min(scores, key=lambda spec: (-score(spec), float(spec.split(':')[1])))


def three_comparators(validation, evaluation, *, start, end, spd=6, column='reward'):
    """Two deployable constant choices plus ex-post best evaluated trajectory.

    Hindsight selects a whole candidate independently on each evaluation opening.
    It is the exact maximum of the evaluated candidate set, NOT a certified
    upper bound on all possible dynamic policies.
    """
    global_spec = select_constant(validation, start=start, end=end, spd=spd, column=column)
    # Hindsight is an independent comparator class, not max(RL, comparators).
    # Privileged online simulator search is also a separate information set.
    candidates = {spec: trace for spec, trace in evaluation.items()
                  if spec.startswith(('static:', 'oracle:', 'schedule:'))}
    scores = {spec: trajectory_scores(trace, start=start, end=end, spd=spd, column=column)
              for spec, trace in candidates.items()}
    if not scores or any(not v for v in scores.values()):
        raise ValueError('Missing evaluation evidence')
    tapes = set(next(iter(scores.values())))
    if any(set(v) != tapes for v in scores.values()):
        raise ValueError('Hindsight candidates must share complete evaluation openings')
    rivals = sorted({name.rsplit('__opening_', 1)[0] for name in tapes})
    per_rival = {r: select_constant(validation, start=start, end=end, spd=spd, column=column, rival=r)
                 for r in rivals}
    if global_spec not in scores or any(s not in scores for s in per_rival.values()):
        raise ValueError('Selected constant has no evaluation evidence')
    hindsight = {tape: min(scores, key=lambda spec: (-scores[spec][tape], spec)) for tape in sorted(tapes)}
    values = {name: dict(single_constant=scores[global_spec][name],
            rival_informed_constant=scores[per_rival[name.rsplit('__opening_', 1)[0]]][name],
            hindsight_best_evaluated=scores[hindsight[name]][name]) for name in sorted(tapes)}
    summaries = {}
    for rival in rivals:
        rows = [v for name, v in values.items() if name.rsplit('__opening_', 1)[0] == rival]
        def estimate(samples):
            return dict(mean=mean(samples), standard_error=stdev(samples)/math.sqrt(len(samples)) if len(samples)>1 else None,
                        openings=len(samples))
        summaries[rival] = {key: estimate([v[key] for v in rows]) for key in next(iter(values.values()))}
        summaries[rival]['paired_rival_informed_minus_single'] = estimate(
            [v['rival_informed_constant']-v['single_constant'] for v in rows])
        summaries[rival]['paired_hindsight_minus_single'] = estimate(
            [v['hindsight_best_evaluated']-v['single_constant'] for v in rows])
    return dict(objective=column, global_constant=global_spec, per_rival_constants=per_rival,
        hindsight_choices=hindsight, hindsight_candidate_count=len(scores),
        hindsight_scope='best evaluated static/oracle/schedule complete trajectory per opening; excludes RL and online myopic search; not global dynamic optimum',
        uncertainty_unit='independent opening within rival, not market blocks; hindsight selection optimism remains',
        summaries_by_rival=summaries, tapes=values)


def selected_outcomes(report, evaluation, *, start, end, spd=6):
    """Score already-selected complete policies; never reselect for an outcome."""
    columns = ('reward', 'profit_per_request', 'rival_profit_per_request', 'profit_lead', 'own_market_share')
    cache = {(spec, window, col): trajectory_scores(
                 trace, start=begin, end=end, spd=spd, column=col,
                 allow_missing=(col == 'own_market_share'))
             for spec, trace in evaluation.items() for window, begin in (('full', start), ('late', max(start, end*.75)))
             for col in columns}
    rows = []
    for tape in sorted(report['tapes']):
        rival = tape.rsplit('__opening_', 1)[0]
        choices = dict(single_constant=report['global_constant'],
            rival_informed_constant=report['per_rival_constants'][rival],
            hindsight_best_evaluated=report['hindsight_choices'][tape])
        for comparator, spec in choices.items():
            row = dict(rollout=tape, rival=rival, comparator=comparator, policy=spec)
            row.update({f'{window}_{col}': cache[spec, window, col][tape]
                        for window in ('full', 'late') for col in columns})
            rows.append(row)
    return rows
