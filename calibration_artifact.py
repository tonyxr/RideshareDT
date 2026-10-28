"""Exact reuse of a measured city operating point and validation warm start."""
import json
from dataclasses import asdict
from pathlib import Path

SUPPLY_CONTRACT = 'revision6-posted-pay-entry-stochastic-rounding'


def validate_supply_point(point, *, customers, explicit):
    """Never silently reinterpret a saved economic or observation contract."""
    if point.get('contract') != SUPPLY_CONTRACT:
        raise ValueError('saved supply point predates revision6; recalibration is required')
    if point['calibration']['sample_size'] != customers:
        raise ValueError('saved supply calibration request volume differs from this run')
    for key, requested in explicit.items():
        saved = point.get('fleet_scale', 1.) if key == 'fleet_scale' else point['config'][key]
        if saved != requested:
            raise ValueError(f'explicit supply setting {key}={requested} conflicts with saved {saved}; '
                             'use a newly calibrated operating point for sensitivity experiments')


def supply_point(core):
    return dict(contract=SUPPLY_CONTRACT, city=core.market_name, config=asdict(core.driver_supply.config),
        demand_calibration=getattr(core.choice_model, 'demand_spec', None),
        fleet_scale=core.market_dynamics.fleet_scale,
        hour_profile=dict(core.driver_supply.hour_profile),
        pay_policies={f: asdict(p) for f, p in core.driver_supply.pay_policies.items()},
        calibration=core.v26_settings['driver_capacity_calibration'])


def load_city_artifact(path, *, city, lead_weight, grid):
    data = json.loads(Path(path).read_text())
    if data.get('format') != 'revision6-city-calibration-v1':
        raise ValueError('unsupported city calibration artifact; rerun the repaired audit')
    if data['city'] != city or data['opponent_profit_lead_weight'] != lead_weight:
        raise ValueError('city calibration market/reward contract mismatch')
    # Revision 15 level-step policies pass grid=None: they have no absolute
    # target grid and never apply the artifact's warm start.
    if grid is not None and (sorted(data['grid']) != sorted(grid) or data['warm_start_target'] not in grid):
        raise ValueError('city calibration action-grid contract mismatch')
    if data['supply_operating_point']['calibration']['burnin_days'] < 15:
        raise ValueError('city operating point lacks the required calibration burn-in')
    return data
