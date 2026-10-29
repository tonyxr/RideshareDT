"""Local, no-training city calibration and elasticity audit. Explicit CLI only.

Reference earnings are measured on a separate calm anchor-price calibration
and then fixed across policies. Capacity uses the installed v2 supply model;
empirical validation of driver income still requires firm data.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent
sys.path.insert(0, str(BASE))
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('MPLBACKEND', 'Agg')
from diagnostic_logging import atomic_json
from evaluation_selection import three_comparators, selected_outcomes
from calibration_artifact import supply_point
from scripts.run_v26r_staged import command
from scripts.v26r_cases import STATIC_GRID

CITIES = {'nyc': 'New York City', 'boston': 'Boston', 'chicago': 'Chicago'}
RIVALS = ('adaptive_best_response_memory', 'adaptive_best_response_aggressive')
SOURCES = [dict(title='List, Muir, Pope & Sun, Left-Digit Bias at Lyft (2023)',
    url='https://s3.amazonaws.com/fieldexperiments-papers2/papers/00771.pdf',
    estimate='Standard-product conversion elasticity about -0.9 to -1.4; Shared -1.7 to -3.3',
    context='US Lyft price-offer experiments; other options held fixed, not total market demand'),
    dict(title='Cohen et al., Using Big Data to Estimate Consumer Surplus (2016)',
    url='https://www.nber.org/papers/w22627',
    context='Local short-run surge discontinuities, four US cities. Not a universal city calibration target.'),
    dict(title='Christensen, Nino & Osman, The Demand for Mobility (2025 revision)',
    url='https://www.nber.org/papers/w31330',
    context='Egypt, three-month discounts; 50% discount more than quadruples Uber usage. '
            'Long-horizon and total-mobility estimands differ from this choice audit.')]


def load_core(slug, output, *, integration=False, demand_artifact=None,
              city_artifact=None, shock_scenario=None, before_supply=False,
              calibration_json=None):
    """Reuse production CLI configuration, stop at the requested sweep boundary."""
    import Core
    captured = []
    class Captured(Exception):
        pass
    def capture(core, **kwargs):
        captured.append(core)
        raise Captured()
    argv = command(output, days=0, city=CITIES[slug], slug=slug, integration=integration)
    argv[argv.index('--rl_warm_start_target')+1] = '0'
    if calibration_json is not None:
        argv[argv.index('--calibration_json') + 1] = str(calibration_json)
    argv += ['--learning_corrections', '--learner_revision', '4', '--no-live_diagnostics',
             '--shock_scenario', str(shock_scenario or BASE/'scenarios/calm.json'),
             '--policy_sweep', 'static:1']
    if demand_artifact is not None:
        argv += ['--demand_calibration_artifact', str(demand_artifact)]
    if city_artifact is not None:
        argv += ['--city_calibration_artifact', str(city_artifact)]
    previous, sweep = sys.argv, Core.Core.run_policy_sweep
    supply = Core.Core.configure_v26r
    try:
        sys.argv = argv[2:]
        Core.Core.run_policy_sweep = capture
        if before_supply:
            Core.Core.configure_v26r = capture
        Core.main()
    except Captured:
        return captured[0]
    finally:
        sys.argv = previous
        Core.Core.run_policy_sweep = sweep
        Core.Core.configure_v26r = supply
    raise RuntimeError('CLI did not reach the no-training capture point')


def elasticity(core, *, n=2000):
    """Exact expected choices on cached held-out riders and empirical trip draws.

    Supply, waits, reputation and loyalty are held fixed; this is demand, not
    completed rides. No GPT calls or threshold edits. Derivatives are log arcs.
    """
    import numpy as np
    from MarketInteraction import RideContext
    rng = np.random.default_rng(501710)
    model = copy.deepcopy(core.choice_model)
    model.reputation = None
    pool = core.evaluation_profile_pool
    if not pool:
        raise ValueError('No held-out rider cache')
    records, fares, thresholds = [], [], []
    for index in rng.integers(0, len(pool), size=n):
        profile = dict(pool[int(index)])
        hour = int(rng.choice(24, p=np.array(core.market.hour_probs)/sum(core.market.hour_probs)))
        day = int(rng.choice(7, p=np.array(core.market.day_of_week_probs)/sum(core.market.day_of_week_probs)))
        trip = core.market.sample_operational_trips(rng, hour=hour, day_of_week=day, size=1)
        if trip is None:
            raise ValueError('City audit requires the actual empirical trip bank')
        distance, duration = float(trip['distance'][0]), float(trip['duration'][0])
        airport, service = bool(trip['airport'][0]), str(trip['service'][0])
        context = RideContext(day_of_week=day, hour=hour, weather='clear', airport=airport, service=service)
        fare = float(core.market.quote_price(distance, duration, context, overrides=None))
        scenario = dict(City=core.market_name, DistanceMiles=distance, DurationMinutes=duration,
                        Airport=airport, Service=service, Hour=hour, DayOfWeek=day, Weather='clear',
                        WaitEstimateFirm1=5., WaitEstimateFirm2=5., CancelRiskFirm1=0., CancelRiskFirm2=0.)
        scenario['ReferenceFare'] = fare
        records.append((profile, scenario, fare))
        fares.append(fare)
        thresholds.append(float(profile['PriceThreshold']))
    def demand(multiplier, both=False):
        p = [model.probabilities(profile, scenario, fare*multiplier, fare*(multiplier if both else 1.))
             for profile, scenario, fare in records]
        return {'own': float(np.mean([row['Firm1'] for row in p])),
                'market': float(np.mean([1-row['NoRide'] for row in p]))}
    curves = {str(m): {'own_price_only': demand(m), 'both_prices': demand(m, True)} for m in STATIC_GRID}
    arcs = []
    for level in (1., 1.4, 2.):
        for delta in (.01, .05, .10):
            for both in (False, True):
                low, high = demand(level*(1-delta), both), demand(level*(1+delta), both)
                arcs.append(dict(level=level, perturbation_fraction=delta,
                    price_intervention='both' if both else 'own_only',
                    **{f'{metric}_elasticity': float(np.log(high[metric]/low[metric]) /
                        np.log((1+delta)/(1-delta))) for metric in low}))
    typical = float(np.median(fares))
    return dict(requested_choices_not_completed_rides=True, n=n, rng_seed=501710,
        typical_fare_median=typical, typical_fare_mean=float(np.mean(fares)),
        fare_q10_q90=np.quantile(fares, [.1, .9]).tolist(),
        raw_threshold_median=float(np.median(thresholds)),
        threshold_over_city_typical_fare_median=float(np.median(thresholds)/typical),
        threshold_over_matched_trip_fare_q10_q50_q90=np.quantile(np.array(thresholds)/fares,[.1,.5,.9]).tolist(),
        threshold_clipped_in_current_choice_count=(0 if getattr(core.choice_model, 'fare_normalized', False)
            else int(np.sum((np.array(thresholds)<.25)|(np.array(thresholds)>8)))),
        curves=curves, elasticities=arcs, sources=SOURCES,
        limitations=['Conditional on existing request opportunities: no app-entry elasticity.',
            'Threshold/fare is dimensionless tolerance, NOT a demand elasticity.',
            'Own-price conversion and joint-price market elasticity are different estimands.',
            'Cached GPT thresholds are behavioral priors, not observed willingness-to-pay.',
            'City fare anchors inherit dataset limitations, including inferred Boston durations.'])


def audit(slug, output, *, openings=10, horizon_days=40, burnin_days=15, core=None):
    out = output/slug
    out.mkdir(parents=True, exist_ok=False)
    core = load_core(slug, out) if core is None else core
    atomic_json(out/'capacity.json', core.v26_settings['driver_capacity_calibration'])
    print(f'[audit] {slug}: capacity complete; measuring cached-rider elasticity', flush=True)
    atomic_json(out/'elasticity.json', elasticity(core))
    traces = {}
    for split, seed in [('validation', 20231), ('evaluation', 99173)]:
        print(f'[audit] {slug}: {split} constants, 13 targets x 2 rivals x {openings} openings x {horizon_days} days', flush=True)
        sweep = core.run_policy_sweep(specs=[f'static:{m:g}' for m in STATIC_GRID], rivals=RIVALS,
            openings=openings, horizon=horizon_days*6, customers_per_step=50, steps_per_day=6,
            firm1_interval_steps=6, opponent_interval_steps=6, policy_seed=seed,
            evaluation_profiles=(split == 'evaluation'))
        atomic_json(out/f'{split}_statics.json', sweep)
        traces[split] = sweep['traces']
        print(f'[audit] {slug}: {split} saved', flush=True)
    report = three_comparators(traces['validation'], traces['evaluation'], start=burnin_days, end=horizon_days)
    report['window'] = dict(burnin_days=burnin_days, horizon_days=horizon_days, openings_per_rival=openings)
    report['warm_start_target'] = float(report['global_constant'].split(':')[1])
    report['outcomes'] = selected_outcomes(report, traces['evaluation'], start=burnin_days, end=horizon_days)
    report['losing_outcomes'] = [row for row in report['outcomes']
                               if row['full_profit_lead'] < 0 or row['late_profit_lead'] < 0]
    report['hindsight_limitation'] = 'This calibration evaluates constants only; no dynamic hindsight search or RL performance claim.'
    atomic_json(out/'comparators.json', report)
    atomic_json(out/'city_calibration.json', dict(format='revision6-city-calibration-v1',
        city=core.market_name, grid=list(STATIC_GRID),
        demand_calibration=getattr(core.choice_model, 'demand_spec', None),
        opponent_profit_lead_weight=core.long_term_profit_reward_config.opponent_profit_lead_weight,
        warm_start_target=report['warm_start_target'], warm_start_selection=report,
        supply_operating_point=supply_point(core)))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cities', nargs='+', choices=list(CITIES), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--openings', type=int, default=10)
    parser.add_argument('--horizon-days', type=int, default=40)
    parser.add_argument('--burnin-days', type=int, default=15)
    args = parser.parse_args()
    if args.openings < 2 or args.horizon_days <= args.burnin_days or args.burnin_days < 15:
        parser.error('require >=2 openings and a scored window after >=15 burn-in days')
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output/'manifest.json', dict(training=False, cities=args.cities,
        source_hashes={str(p.relative_to(BASE)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted([*BASE.glob('*.py'), *BASE.glob('scripts/*.py'),
                                        *BASE.glob('scenarios/*.json')])}, grid=STATIC_GRID,
        reward='.4 own_profit/2 + .6 (own_profit-rival_profit)/2',
        validation_seed=20231, evaluation_seed=99173, worker_count=1,
        openings_per_rival=args.openings, horizon_days=args.horizon_days, burnin_days=args.burnin_days))
    results = {}
    for slug in args.cities:
        atomic_json(args.output/'status.json', dict(city=slug, stage='calibration_and_validation', completed=list(results)))
        try:
            results[slug] = audit(slug, args.output, openings=args.openings,
                                 horizon_days=args.horizon_days, burnin_days=args.burnin_days)
        except Exception as error:
            results[slug] = dict(error=str(error), traceback=traceback.format_exc(), completed=False)
            atomic_json(args.output/slug/'failure.json', results[slug])
            print(results[slug]['traceback'], flush=True)
            # Independent cities continue; failed city is never retried or
            # assigned another reference, fleet, target or policy.
    atomic_json(args.output/'results.json', results)
    failed = [slug for slug, result in results.items() if 'error' in result]
    atomic_json(args.output/'status.json', dict(stage='failed' if failed else 'complete', failed_cities=failed))
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
