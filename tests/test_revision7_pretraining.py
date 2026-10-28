"""Pre-training audit regressions; no city sweep or RL training."""
from types import SimpleNamespace

import pytest

from market_dynamics import rescale_pricer_to_market
from pricing_models import FirmMPCGridPricer, FirmMemoryBestResponsePricer
from scripts.audit_revision7_pretraining import gate0_report, gate1_report, impulse_reports


def traces(score):
    rows = []
    for rival in ('memory', 'aggressive'):
        for opening in (1, 2):
            for step in range(240):
                day = step // 6
                rows.append(dict(rollout=f'{rival}__opening_{opening}', step=step,
                    reward=float(score(day)), profit_per_request=0.,
                    rival_profit_per_request=0., profit_lead=0., own_market_share=.5))
    return rows


def test_gate0_requires_a_unique_interior_fixed_price():
    validation = {f'static:{m:g}': traces(lambda day, m=m: -(m-1.)**2)
                  for m in (.6, .9, 1., 1.1, 2.)}
    evaluation = {key: list(value) for key, value in validation.items()}
    report = gate0_report(validation, evaluation)
    assert report['passed']
    assert report['best_constant'] == 'static:1'

    edge = {f'static:{m:g}': traces(lambda day, m=m: m) for m in (.6, 1., 2.)}
    assert not gate0_report(edge, edge)['passed']


def test_gate1_uses_separate_pre_event_and_event_windows():
    case = dict(onset=14, end=19)
    validation = {}
    for m in (.6, .8, 1., 1.2, 1.4, 2.):
        validation[f'static:{m:g}'] = traces(
            lambda day, m=m: -(m-(1.2 if 14 <= day <= 19 else 1.))**2)
    report = gate1_report(validation, validation, case)
    assert report['passed']
    assert report['calm_constant'] == 'static:1'
    assert report['regime_constant'] == 'static:1.2'
    assert report['absolute_shift'] == pytest.approx(.2)


def test_production_rival_impulses_match_day_based_contracts():
    anchors = dict(base_fare=3., per_minute=.4, per_mile=1.5,
                   booking_fee=2., airport_fee=5.)

    class Supply:
        @staticmethod
        def _earnings_reference(firm):
            return 12.

    class Core:
        market = SimpleNamespace(curr_market=SimpleNamespace(**anchors))
        driver_supply = Supply()

        @staticmethod
        def _make_pricer(mode, seed):
            cls = FirmMPCGridPricer if mode == 'mpc_grid' else FirmMemoryBestResponsePricer
            pricer = cls(seed=seed, managed_keys=list(anchors))
            pricer.operational_clock = True
            pricer.reference_window_steps = 6
            pricer.currency_unit = 1.
            pricer.earnings_reference = 12.
            pricer.ema_driver_earnings = 12.
            rescale_pricer_to_market(pricer, anchors)
            return pricer

    report = impulse_reports(Core(), memory_seeds=100)
    assert report['mpc_held_undercut']['passed']
    assert report['mpc_held_undercut']['jitter_reversals'] == 0
    memory = report['memory_lag_hold_recovery']
    assert memory['passed']
    assert memory['observed_lag_days']['mean'] == 1.
    assert memory['maximum_final_tariff_index_error'] == 0.
