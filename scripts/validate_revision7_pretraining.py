"""Fail-closed eligibility check for revision-7 production training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REQUIRED_CITIES = ('nyc', 'boston', 'chicago')
REQUIRED_GATE1 = ('A1_shortage_severe', 'S1_promotion_bait')


def validate(results):
    failures = []
    for city in REQUIRED_CITIES:
        row = results.get(city)
        if not isinstance(row, dict):
            failures.append(f'{city}: missing result')
            continue
        if row.get('primary_elasticity') != -1.15:
            failures.append(f'{city}: primary elasticity is not -1.15')
        if not row.get('gate0', {}).get('passed'):
            failures.append(f'{city}: Gate 0 failed')
        for case in REQUIRED_GATE1:
            if not row.get('gate1', {}).get(case, {}).get('passed'):
                failures.append(f'{city}: Gate 1 failed for {case}')
        impulses = row.get('rival_impulses', {})
        if not impulses.get('mpc_held_undercut', {}).get('passed'):
            failures.append(f'{city}: MPC impulse failed')
        if not impulses.get('memory_lag_hold_recovery', {}).get('passed'):
            failures.append(f'{city}: memory-rival impulse failed')
        calibration = Path(str(row.get('city_calibration', '')))
        if not calibration.is_file():
            failures.append(f'{city}: missing city calibration artifact')
            continue
        saved = json.loads(calibration.read_text())
        demand = saved.get('demand_calibration') or {}
        if demand.get('evidence', {}).get('target_own_elasticity') != -1.15:
            failures.append(f'{city}: city artifact does not embed the -1.15 demand arm')
        if saved.get('supply_operating_point', {}).get('demand_calibration') != demand:
            failures.append(f'{city}: supply point and demand contract differ')
        if saved.get('warm_start_target') != float(row['gate0']['best_constant'].split(':')[1]):
            failures.append(f'{city}: warm start differs from Gate-0 fixed-price selection')
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    results = json.loads(args.results.read_text())
    failures = validate(results)
    report = dict(eligible=not failures, failures=failures,
        rule='all cities: -1.15 contract, Gate 0, both Gate-1 cases and both rival impulses pass')
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    if failures:
        raise SystemExit('\n'.join(failures))


if __name__ == '__main__':
    main()
