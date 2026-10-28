"""Revision-7 pre-training audit: supply, warm starts, Gates 0/1 and rivals.

This is an evaluation-only entry point.  It never trains or loads an RL policy.
The -1.15 transferred-prior arm is the primary market contract.  The two
endpoint arms are retained only as fixed-price sensitivity analyses.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
import statistics
import sys
import traceback

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('MPLBACKEND', 'Agg')

from calibration_artifact import supply_point
from city_supply_calibration import calibrate_capacity
from diagnostic_logging import atomic_json
from evaluation_selection import select_constant, trajectory_scores
from pricing_models import FirmMetrics
from scripts.audit_city_calibration import CITIES, RIVALS, audit, load_core
from scripts.run_v26r_staged import source_hashes
from scripts.v26r_cases import CASES, STATIC_GRID

PRIMARY = 1.15
SENSITIVITY = (0.9, 1.4)
STATIC_SPECS = tuple(f'static:{value:g}' for value in STATIC_GRID)
GATE_CASES = ('A1_shortage_severe', 'S1_promotion_bait')


def estimate(values):
    values = [float(value) for value in values]
    return dict(mean=statistics.mean(values), standard_error=(
        statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None), n=len(values))


def fixed_landscape(traces, *, start, end, rival=None):
    rows = {}
    for spec, trace in traces.items():
        if not spec.startswith('static:'):
            continue
        scores = trajectory_scores(trace, start=start, end=end, rival=rival)
        rows[spec] = estimate(scores.values())
    if not rows:
        raise ValueError('fixed-price landscape is empty')
    order = sorted(rows, key=lambda spec: (-rows[spec]['mean'], float(spec.split(':')[1])))
    return rows, order


def gate0_report(validation, evaluation, *, start=15, end=40):
    landscape, order = fixed_landscape(validation, start=start, end=end)
    best, runner_up = order[:2]
    best_value = float(best.split(':')[1])
    best_eval = trajectory_scores(evaluation[best], start=start, end=end)
    runner_eval = trajectory_scores(evaluation[runner_up], start=start, end=end)
    names = sorted(set(best_eval) & set(runner_eval))
    paired = estimate([best_eval[name] - runner_eval[name] for name in names])
    inside = min(STATIC_GRID) < best_value < max(STATIC_GRID)
    unique = landscape[best]['mean'] > landscape[runner_up]['mean']
    return dict(definition='unique validation-selected single constant strictly inside the declared grid',
        passed=bool(inside and unique), best_constant=best, runner_up=runner_up,
        strictly_inside_grid=inside, unique_validation_argmax=unique,
        validation_landscape=landscape, heldout_best_minus_runner_up=paired,
        selection_window_days=[start, end])


def gate1_report(validation, evaluation, case):
    onset, end = int(case['onset']), int(case['end']) + 1
    burnin = int(case.get('burnin', 10))
    calm = select_constant(validation, start=burnin, end=onset)
    regime = select_constant(validation, start=onset, end=end)
    calm_value, regime_value = (float(value.split(':')[1]) for value in (calm, regime))
    calm_scores = trajectory_scores(evaluation[calm], start=onset, end=end)
    regime_scores = trajectory_scores(evaluation[regime], start=onset, end=end)
    names = sorted(set(calm_scores) & set(regime_scores))
    gain = estimate([regime_scores[name] - calm_scores[name] for name in names])
    shift = abs(regime_value - calm_value)
    se = gain['standard_error']
    return dict(definition='validation best fixed price shifts by at least 0.2x from pre-event calm to event days',
        passed=bool(shift >= .2 - 1e-12), calm_constant=calm, regime_constant=regime,
        absolute_shift=shift, direction=('raise' if regime_value > calm_value else
            'cut' if regime_value < calm_value else 'none'),
        heldout_regime_objective_gain=gain,
        heldout_gain_clears_three_se=bool(se is not None and se > 0 and gain['mean'] > 3*se),
        calm_window_days=[burnin, onset], regime_window_days=[onset, end])


def run_fixed_sweep(core, *, openings, horizon_days, seed, evaluation_profiles):
    return core.run_policy_sweep(specs=STATIC_SPECS, rivals=RIVALS,
        openings=openings, horizon=horizon_days*6, customers_per_step=50,
        steps_per_day=6, firm1_interval_steps=6, opponent_interval_steps=6,
        policy_seed=seed, evaluation_profiles=evaluation_profiles)


def tariff_index(pricer, anchors):
    values = [float(getattr(pricer.overrides, key)) / float(anchors[key])
              for key in pricer.managed_keys if float(anchors[key]) > 0]
    return statistics.mean(values)


def stable_metrics(reference):
    return FirmMetrics(total=50, reference_total=50, chosen=25, completed=25,
        wins=25, revenue=50*float(reference), profit=10., dispatch_offers=25,
        wait_minutes=125., pickup_minutes=125.)


def impulse_reports(core, *, memory_seeds=1000):
    base = core.market.curr_market
    anchors = {key: float(getattr(base, key)) for key in
        ('base_fare', 'per_minute', 'per_mile', 'booking_fee', 'airport_fee')}
    supply_state = dict(driver_earnings_per_online_hour=core.driver_supply._earnings_reference('Firm2'),
        driver_earnings_per_hour=core.driver_supply._earnings_reference('Firm2'),
        online_minutes=60., fulfillment_rate=1., acceptance_rate=1., wait_minutes=5.,
        pickup_minutes=5., idle_share=.3, utilization=.7)

    def act(pricer):
        pricer.act(city_base=anchors['base_fare'], city_pmin=anchors['per_minute'],
            city_pmile=anchors['per_mile'], city_booking=anchors['booking_fee'],
            city_airport=anchors['airport_fee'], hour=12, weather='clear')

    mpc = core._make_pricer('mpc_grid', 601)
    act(mpc)
    mpc_rows = []
    gap = 2.0 * float(mpc.currency_unit)
    for day in range(12):
        for _ in range(6):
            mpc.update(stable_metrics(8*mpc.currency_unit), gap if 2 <= day <= 7 else 0.,
                       supply_state=supply_state)
        before = tariff_index(mpc, anchors)
        act(mpc)
        after = tariff_index(mpc, anchors)
        mpc_rows.append(dict(day=day, held_undercut=2 <= day <= 7,
            direction=int(mpc.last_direction), reason=mpc.last_reason,
            tariff_index=after, change=after-before))
    active = [row for row in mpc_rows if row['held_undercut']]
    changes = [row['change'] for row in active if abs(row['change']) > 1e-12]
    mpc_report = dict(spec='six-day held Firm-1 undercut; one decision per day',
        passed=bool(changes and all(change < 0 for change in changes)),
        reacted=bool(changes), jitter_reversals=sum(
            1 for left, right in zip(changes, changes[1:]) if left*right < 0), trace=mpc_rows)

    holds, recoveries, lags, failures, restoration_errors = [], [], [], [], []
    expected_hold_days = 1. / (1. - (1. - 1./30.)**6)
    for seed in range(memory_seeds):
        rival = core._make_pricer('adaptive_best_response_memory', 10_000 + seed)
        act(rival)
        opening = tariff_index(rival, anchors)
        first_retaliation = first_recovery = None
        restored = False
        retaliation_days = recovery_days = 0
        for day in range(120):
            day_gap = 2.0 * float(rival.currency_unit) if day == 0 else 0.
            for _ in range(6):
                rival.update(stable_metrics(8*rival.currency_unit), day_gap,
                    supply_state=supply_state)
            act(rival)
            if rival.memory_state == 'retaliation':
                retaliation_days += 1
                first_retaliation = day if first_retaliation is None else first_retaliation
            if rival.memory_state == 'recovery':
                recovery_days += 1
                first_recovery = day if first_recovery is None else first_recovery
                targets = dict(rival._pre_retaliation_tariff or {})
                restored = bool(targets) and all(
                    target is None or abs(float(getattr(rival.overrides, key)) - float(target)) <= 1e-12
                    for key, target in targets.items())
            if restored:
                break
        if first_retaliation is None or first_recovery is None or not restored:
            failures.append(seed)
            continue
        lags.append(first_retaliation + 1)
        holds.append(retaliation_days)
        recoveries.append(recovery_days)
        restoration_errors.append(abs(tariff_index(rival, anchors) - opening))
    hold_est = estimate(holds) if holds else dict(mean=None, standard_error=None, n=0)
    memory_report = dict(spec='one-day sustained undercut, then release; six observations and one action per day',
        passed=bool(not failures and lags and set(lags) == {1} and
            abs(hold_est['mean'] - expected_hold_days) <= 3*hold_est['standard_error'] and
            recoveries and max(restoration_errors, default=float('inf')) <= 1e-12),
        configured_lag_days=1., observed_lag_days=estimate(lags) if lags else None,
        configured_hold_distribution='geometric mean 30 market blocks',
        implied_mean_hold_days=expected_hold_days, observed_hold_days=hold_est,
        configured_recovery='one inverse coefficient step per day until the pre-retaliation tariff is restored',
        observed_recovery_days=estimate(recoveries) if recoveries else None,
        maximum_final_tariff_index_error=max(restoration_errors, default=None),
        failed_seeds=failures)
    return dict(mpc_held_undercut=mpc_report, memory_lag_hold_recovery=memory_report)


def run_city(slug, output, demand_root, *, openings, sensitivity_openings, horizon_days):
    city_out = output/slug
    city_out.mkdir()
    primary = demand_root/slug/f'demand_{PRIMARY:g}.json'
    if not primary.exists():
        raise FileNotFoundError(primary)
    core = load_core(slug, city_out/'primary_setup', demand_artifact=primary)
    atomic_json(city_out/'primary_supply_operating_point.json', supply_point(core))
    verified = calibrate_capacity(core, sample_size=50, days=45, burnin_days=15,
        measure_reference=False, verify_only=True)
    atomic_json(city_out/'primary_supply_verification.json', verified)
    audit(slug, output/'primary_calibration', openings=openings,
        horizon_days=horizon_days, burnin_days=15, core=core)
    city_artifact = output/'primary_calibration'/slug/'city_calibration.json'
    validation = json.loads((output/'primary_calibration'/slug/'validation_statics.json').read_text())
    evaluation = json.loads((output/'primary_calibration'/slug/'evaluation_statics.json').read_text())
    gate0 = gate0_report(validation['traces'], evaluation['traces'], start=15, end=horizon_days)
    atomic_json(city_out/'gate0.json', gate0)
    atomic_json(city_out/'rival_impulses.json', impulse_reports(core))

    sensitivities = {}
    for target in SENSITIVITY:
        artifact = demand_root/slug/f'demand_{target:g}.json'
        sensitivity_core = load_core(slug, city_out/f'sensitivity_{target:g}_setup', demand_artifact=artifact)
        val = run_fixed_sweep(sensitivity_core, openings=sensitivity_openings,
            horizon_days=horizon_days, seed=20231, evaluation_profiles=False)
        test = run_fixed_sweep(sensitivity_core, openings=sensitivity_openings,
            horizon_days=horizon_days, seed=99173, evaluation_profiles=True)
        result = gate0_report(val['traces'], test['traces'], start=15, end=horizon_days)
        result['supply_operating_point'] = supply_point(sensitivity_core)
        sensitivities[str(-target)] = result
        atomic_json(city_out/f'sensitivity_{target:g}.json', result)
    del core

    gates1 = {}
    for name in GATE_CASES:
        case = CASES[name]
        scenario = BASE/case['scenario']
        scenario_core = load_core(slug, city_out/f'{name}_setup', city_artifact=city_artifact,
            shock_scenario=scenario)
        val = run_fixed_sweep(scenario_core, openings=openings, horizon_days=horizon_days,
            seed=20231, evaluation_profiles=False)
        test = run_fixed_sweep(scenario_core, openings=openings, horizon_days=horizon_days,
            seed=99173, evaluation_profiles=True)
        atomic_json(city_out/f'{name}_validation.json', val)
        atomic_json(city_out/f'{name}_evaluation.json', test)
        gates1[name] = gate1_report(val['traces'], test['traces'], case)
        atomic_json(city_out/f'{name}_gate1.json', gates1[name])
    return dict(primary_elasticity=-PRIMARY, gate0=gate0, gate1=gates1,
        sensitivities=sensitivities, city_calibration=str(city_artifact),
        rival_impulses=json.loads((city_out/'rival_impulses.json').read_text()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--demand-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cities', nargs='+', choices=tuple(CITIES), default=tuple(CITIES))
    parser.add_argument('--openings', type=int, default=10)
    parser.add_argument('--sensitivity-openings', type=int, default=3)
    parser.add_argument('--horizon-days', type=int, default=40)
    args = parser.parse_args()
    if args.openings < 2 or args.sensitivity_openings < 2 or args.horizon_days < 40:
        parser.error('require >=2 openings and a >=40-day horizon')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    before = source_hashes()
    atomic_json(out/'manifest.json', dict(started=dt.datetime.now().isoformat(), training=False,
        primary_elasticity=-PRIMARY, sensitivity_elasticities=[-x for x in SENSITIVITY],
        cities=args.cities, openings=args.openings, sensitivity_openings=args.sensitivity_openings,
        horizon_days=args.horizon_days, demand_root=str(args.demand_root.resolve()),
        source_hashes=before, gate0='unique best single constant strictly inside grid',
        gate1='absolute calm-to-event best-constant shift >= 0.2x'))
    results, failures = {}, {}
    try:
        for slug in args.cities:
            (out/'status.txt').write_text(f'{slug}: supply calibration, warm start, gates and impulses\n')
            try:
                results[slug] = run_city(slug, out, args.demand_root.resolve(),
                    openings=args.openings, sensitivity_openings=args.sensitivity_openings,
                    horizon_days=args.horizon_days)
            except BaseException as error:
                failures[slug] = dict(error=str(error), traceback=traceback.format_exc())
                atomic_json(out/f'{slug}_failure.json', failures[slug])
                print(failures[slug]['traceback'], flush=True)
            atomic_json(out/'results.json', results)
        if before != source_hashes():
            raise RuntimeError('source changed during audit')
        atomic_json(out/'failures.json', failures)
        if failures:
            raise RuntimeError(f'failed cities: {sorted(failures)}')
        (out/'status.txt').write_text('completed: no training; all gate outcomes retained\n')
        (out/'exit_status.txt').write_text('0\n')
    except BaseException:
        (out/'status.txt').write_text('failed: evidence preserved; no retry or training\n')
        (out/'exit_status.txt').write_text('1\n')
        raise


if __name__ == '__main__':
    main()
