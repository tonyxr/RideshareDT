"""Demand-only transferred-prior audit: no training, capacity fitting or rewards."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
os.environ.setdefault('MPLBACKEND', 'Agg')
from diagnostic_logging import atomic_json
from joint_demand import CONTRACT, JointChoiceModel, fit_local_own_elasticity
from scripts.audit_city_calibration import CITIES, load_core, SOURCES

LIMITATIONS = [
    'Transferred Lyft Standard conversion prior, not a city-specific causal estimate.',
    'Anchor participation and non-price utilities retain existing contextual priors; they are not validated by fitting elasticity.',
    'Relative inverse GPT thresholds are a heterogeneity prior, not observed willingness to pay.',
    'Conditional logit substitution (IIA) is assumed, not identified by the single own-price moment.',
    'Request opportunities are fixed: app-entry, long-run market expansion and equilibrium supply are not identified.',
    'No reward, constant-tariff optimum or RL performance enters calibration or parameter selection.',
]


def records(core, *, evaluation, n, seed):
    import numpy as np
    from MarketInteraction import RideContext
    pool = core.evaluation_profile_pool if evaluation else core.training_profile_pool
    if not pool:
        raise ValueError('required disjoint rider partition is empty')
    rng = np.random.default_rng(seed)
    result = []
    for index in rng.integers(len(pool), size=n):
        profile = dict(pool[int(index)])
        hour = int(rng.choice(24, p=np.array(core.market.hour_probs)/sum(core.market.hour_probs)))
        day = int(rng.choice(7, p=np.array(core.market.day_of_week_probs)/sum(core.market.day_of_week_probs)))
        trip = core.market.sample_operational_trips(rng, hour=hour, day_of_week=day, size=1)
        if trip is None:
            raise ValueError('an empirical trip bank is required')
        distance, duration = float(trip['distance'][0]), float(trip['duration'][0])
        airport, service = bool(trip['airport'][0]), str(trip['service'][0])
        context = RideContext(day_of_week=day, hour=hour, weather='clear', airport=airport, service=service)
        fare = float(core.market.quote_price(distance, duration, context, overrides=None))
        scenario = dict(City=core.market_name, ReferenceFare=fare, DistanceMiles=distance,
            DurationMinutes=duration, Hour=hour, DayOfWeek=day, Airport=airport, Service=service,
            Weather='clear', WaitEstimateFirm1=5., WaitEstimateFirm2=5.,
            CancelRiskFirm1=0., CancelRiskFirm2=0.)
        result.append((profile, scenario, fare))
    return result


def responses(model, rows, grid):
    import numpy as np
    def demand(own, rival):
        values = [model.probabilities(p, s, f*own, f*rival) for p,s,f in rows]
        return {key: float(np.mean([v[key] for v in values])) for key in ('Firm1', 'Firm2', 'NoRide')}
    curves = [dict(multiplier=float(m), own_only=demand(m, 1.), both=demand(m, m)) for m in grid]
    elasticities = []
    for delta in (.01, .05):
        for intervention in ('own', 'rival', 'both'):
            lo, hi = 1-delta, 1+delta
            a, b = ((lo, 1.), (hi, 1.)) if intervention == 'own' else (
                ((1., lo), (1., hi)) if intervention == 'rival' else ((lo, lo), (hi, hi)))
            low, high = demand(*a), demand(*b)
            low['market'], high['market'] = 1-low['NoRide'], 1-high['NoRide']
            elasticities.append(dict(intervention=intervention, delta=delta,
                **{key: float(np.log(high[key]/low[key])/np.log(hi/lo)) for key in low}))
    return dict(n=len(rows), curves=curves, elasticities=elasticities)


def audit(slug, out, targets, n, *, price_sensitivity_units='legacy',
          calibration_root=None):
    import numpy as np
    city_out = out/slug
    city_out.mkdir()
    calibration = (
        Path(calibration_root).resolve()/slug/'calibration.json'
        if calibration_root is not None
        else BASE.parent/f'data/processed/{slug}/calibration.json'
    )
    core = load_core(
        slug,
        city_out/'setup',
        before_supply=True,
        calibration_json=calibration,
    )
    legacy = copy.deepcopy(core.choice_model)
    legacy.reputation = None
    provisional = dict(format=CONTRACT, city=core.market_name, typical_fare=core.city_reference_fare,
        threshold_reference=1., beta=1., evidence={'type': 'calibration_scaffold_not_for_deployment'})
    if price_sensitivity_units != 'legacy':
        provisional['price_sensitivity_units'] = price_sensitivity_units
    core.price_sensitivity_units = price_sensitivity_units
    core.configure_joint_demand(provisional)
    core._refresh_profile_pool(rides_per_timestep=50)
    fit = records(core, evaluation=False, n=n, seed=70231)
    heldout = records(core, evaluation=True, n=n, seed=79173)
    grid = [round(float(x), 2) for x in np.arange(.6, 2.001, .1)]
    base = copy.deepcopy(provisional)
    base['typical_fare'] = float(np.median([row[2] for row in fit]))
    base['threshold_reference'] = float(np.median([row[0]['PriceThreshold'] for row in fit]))
    cache = Path(core.threshold_cache_path)
    base['evidence'] = dict(type='transferred_external_prior', source=SOURCES[0],
        fit_partition='training', heldout_partition='evaluation', fit_seed=70231, heldout_seed=79173,
        records_per_partition=n, cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
        city_calibration_sha256=hashlib.sha256(calibration.read_bytes()).hexdigest(),
        limitations=LIMITATIONS, fitting='exact local own-price log derivative at parity')
    training_ids = {id(p) for p in core.training_profile_pool}
    if training_ids.intersection(id(p) for p in core.evaluation_profile_pool):
        raise ValueError('training and held-out rider partitions overlap')
    report = dict(city=core.market_name, limitations=LIMITATIONS,
                  fit_riders=len(core.training_profile_pool), heldout_riders=len(core.evaluation_profile_pool),
                  legacy_heldout=responses(legacy, heldout, grid), arms=[])
    for target in targets:
        spec = copy.deepcopy(base)
        model = JointChoiceModel.from_legacy(legacy, spec)
        spec['beta'] = fit_local_own_elasticity(model, fit, target)
        spec['evidence']['target_own_elasticity'] = target
        model.demand_spec = copy.deepcopy(spec)
        name = f'demand_{abs(target):g}.json'
        atomic_json(city_out/name, spec)
        arm = dict(target=target, artifact=name, beta=spec['beta'],
                   fit=responses(model, fit, [1.]), heldout=responses(model, heldout, grid))
        arm['groups'] = {}
        for label, subset in {
            'low_income': [r for r in heldout if r[0]['IncomeBracket'] == '<50k'],
            'higher_income': [r for r in heldout if r[0]['IncomeBracket'] != '<50k'],
            'below_typical_fare': [r for r in heldout if r[2] < spec['typical_fare']],
            'above_typical_fare': [r for r in heldout if r[2] >= spec['typical_fare']],
            'airport': [r for r in heldout if r[1]['Airport']],
        }.items():
            arm['groups'][label] = responses(model, subset, [1.]) if subset else {'n': 0, 'missing': True}
        storm = [(p, dict(s, Weather='snow', ShockReservationMultiplier=1.25,
                    ShockPriceSensitivityMultiplier=.8, ShockTransitAvailability=.5), f) for p,s,f in heldout]
        arm['storm_sensitivity_scenario'] = responses(model, storm, [.8, 1., 1.2])
        report['arms'].append(arm)
        atomic_json(city_out/'demand_audit.json', report)
        print(f'{slug}: completed prior {target:g}; beta={spec["beta"]:.6g}', flush=True)
    figures(report, city_out)
    return report


def figures(report, out):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    arms = [('Legacy', report['legacy_heldout'])] + [(f'Prior {a["target"]:g}', a['heldout']) for a in report['arms']]
    for label, data in arms:
        x = [r['multiplier'] for r in data['curves']]
        axes[0].plot(x, [r['own_only']['Firm1'] for r in data['curves']], label=label)
        axes[1].plot(x, [1-r['own_only']['NoRide'] for r in data['curves']], label=label)
        axes[2].plot(x, [1-r['both']['NoRide'] for r in data['curves']], label=label)
    for ax, title in zip(axes, ['Own requests: rival fixed', 'Market participation: rival fixed', 'Market participation: both fares move']):
        ax.set(title=title, xlabel='Fare / reference fare', ylabel='Probability', ylim=(0, 1))
        ax.grid(alpha=.2)
    axes[0].legend()
    fig.suptitle(report['city']+' — transferred-prior sensitivity, NOT empirical validation')
    fig.tight_layout()
    fig.savefig(out/'demand_curves.png', dpi=200)
    fig.savefig(out/'demand_curves.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--elasticities', nargs='+', type=float, required=True)
    parser.add_argument('--records', type=int, default=2000)
    parser.add_argument('--price-sensitivity-units', choices=('legacy', 'relative'), default='legacy')
    parser.add_argument('--calibration-root', type=Path)
    args = parser.parse_args()
    if args.records < 1 or any(not (-float('inf') < v < 0) for v in args.elasticities):
        parser.error('positive record count and negative finite elasticity targets are required')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    from scripts.run_v26r_staged import source_hashes
    atomic_json(out/'sources.json', source_hashes())
    atomic_json(out/'command.json', sys.argv)
    (out/'pid.txt').write_text(str(os.getpid())+'\n')
    try:
        for slug in CITIES:
            (out/'status.txt').write_text(f'{slug}: demand-only calibration and held-out diagnostics\n')
            audit(slug, out, args.elasticities, args.records,
                  price_sensitivity_units=args.price_sensitivity_units,
                  calibration_root=args.calibration_root)
        if json.loads((out/'sources.json').read_text()) != source_hashes():
            raise RuntimeError('source changed during the demand audit')
        (out/'status.txt').write_text('completed: all prior arms retained; no RL/supply inference\n')
        (out/'exit_status.txt').write_text('0\n')
    except BaseException:
        (out/'status.txt').write_text('failed: artifacts preserved; no automatic retry\n')
        (out/'exit_status.txt').write_text('1\n')
        raise


if __name__ == '__main__':
    main()
