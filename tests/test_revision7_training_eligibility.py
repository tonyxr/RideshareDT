"""Training cannot start on incomplete or failed revision-7 evidence."""
import json

from scripts.validate_revision7_pretraining import validate


def city(tmp_path, slug):
    calibration = tmp_path/f'{slug}.json'
    demand = dict(evidence=dict(target_own_elasticity=-1.15))
    calibration.write_text(json.dumps(dict(demand_calibration=demand,
        supply_operating_point=dict(demand_calibration=demand), warm_start_target=1.1)))
    return dict(primary_elasticity=-1.15, gate0=dict(passed=True, best_constant='static:1.1'),
        gate1={name: dict(passed=True) for name in ('A1_shortage_severe', 'S1_promotion_bait')},
        rival_impulses=dict(mpc_held_undercut=dict(passed=True),
            memory_lag_hold_recovery=dict(passed=True)), city_calibration=str(calibration))


def test_complete_pretraining_evidence_is_eligible(tmp_path):
    results = {slug: city(tmp_path, slug) for slug in ('nyc', 'boston', 'chicago')}
    assert validate(results) == []


def test_every_required_failure_blocks_training(tmp_path):
    fields = [
        lambda row: row['gate0'].__setitem__('passed', False),
        lambda row: row['gate1']['A1_shortage_severe'].__setitem__('passed', False),
        lambda row: row['gate1']['S1_promotion_bait'].__setitem__('passed', False),
        lambda row: row['rival_impulses']['mpc_held_undercut'].__setitem__('passed', False),
        lambda row: row['rival_impulses']['memory_lag_hold_recovery'].__setitem__('passed', False),
        lambda row: row.__setitem__('primary_elasticity', -.9),
    ]
    for mutate in fields:
        results = {slug: city(tmp_path, slug) for slug in ('nyc', 'boston', 'chicago')}
        mutate(results['nyc'])
        assert validate(results)
