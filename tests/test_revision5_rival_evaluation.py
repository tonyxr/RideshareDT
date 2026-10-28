"""Delayed rival response and independent comparator contracts."""
import copy
import json
from types import SimpleNamespace
import numpy as np
import pytest
from pricing_models import (FirmMetrics, FirmMemoryBestResponsePricer,
    FirmMPCGridPricer, FirmSurgeDriverIncentivePricer, FirmRegionSupplyDemandPricer,
    FirmAdaptiveBestResponsePricer, FirmAggressiveAdaptiveBestResponsePricer,
    FirmPIPriceGapPricer, FirmQueueServiceThresholdPricer)
from Core import Core
from scripts.v26r_cases import CASES, oracle_specs
from market_dynamics import ControllerPerturbation

ANCHORS = dict(city_base=3., city_pmin=.4, city_pmile=1.5, city_booking=2., city_airport=0., hour=12, weather='clear')


@pytest.mark.parametrize('factory', [FirmAdaptiveBestResponsePricer,
    FirmAggressiveAdaptiveBestResponsePricer, FirmMemoryBestResponsePricer,
    FirmPIPriceGapPricer, FirmQueueServiceThresholdPricer, FirmMPCGridPricer,
    FirmSurgeDriverIncentivePricer, FirmRegionSupplyDemandPricer])
def test_zero_revenue_day_is_valid_for_every_adaptive_rival(factory):
    p = factory(seed=4)
    p.operational_clock = True
    p.reference_window_steps = 6
    for _ in range(6):
        p.update(FirmMetrics(revenue=0., total=10), 0.,
                 supply_state={'driver_earnings_per_online_hour': 0.})
        p.act(**ANCHORS)
    assert p.revenue_target == 0.
    assert np.isfinite(p.last_response_score)


def test_negative_absolute_mpc_scores_still_choose_the_improvement(monkeypatch):
    p = FirmMPCGridPricer(managed_keys=['base_fare'])
    p.operational_clock = True
    monkeypatch.setattr(p, '_candidate_score', lambda key, direction: -10 + direction)
    p.act(**ANCHORS)
    assert p.last_direction == 1 and p.last_reason == 'mpc_grid'
    assert p.last_response_score == 1


def test_mpc_scores_are_invariant_to_city_currency_units():
    scores = []
    for unit in (.4, 1., 2.):
        p = FirmMPCGridPricer(managed_keys=['base_fare'])
        p.operational_clock = True
        p.currency_unit = unit
        p.ema_revenue = .8 * unit
        p.revenue_target = .7 * unit
        p.ema_gap = .5
        p.config.step['base_fare'] = .1 * unit
        scores.append([p._candidate_score('base_fare', direction) for direction in (-1, 0, 1)])
    np.testing.assert_allclose(scores, np.repeat([scores[1]], 3, axis=0), rtol=1e-14, atol=1e-14)


def test_surge_gap_sign_does_not_raise_an_already_dearer_tariff():
    p = FirmSurgeDriverIncentivePricer(managed_keys=['base_fare'])
    p.operational_clock = True
    p.ema_gap = 5.
    p.act(**ANCHORS)
    assert p.last_direction == -1


def test_regional_rival_uses_another_coefficient_when_per_mile_is_exhausted():
    p = FirmRegionSupplyDemandPricer(managed_keys=['per_mile', 'base_fare'])
    p.operational_clock = True
    p._ensure_initialized(p._anchor_map(3., .4, 1.5, 2., 0.))
    p.overrides.per_mile = p.config.bounds['per_mile'][1]
    p.ema_choice_share = .8
    p.ema_completed_share = .1
    p.ema_idle_share = 0
    p.act(**ANCHORS)
    assert p.last_direction == 1 and p.last_selected_key == 'base_fare'


def test_memory_rival_observes_full_day_and_has_no_zero_earnings_startup():
    p = FirmMemoryBestResponsePricer(operational_clock=True)
    p.reference_window_steps = 6
    for i in range(6):
        p.update(FirmMetrics(revenue=10*(i+1), total=10), 0.,
                 supply_state={'driver_earnings_per_online_hour': 24.})
    assert p.revenue_target == pytest.approx(3.5)
    assert p.ema_driver_earnings == 24.


def test_rival_price_signals_use_city_currency_units():
    rivals = []
    for unit in (.5, 1., 2.):
        p = FirmMemoryBestResponsePricer(operational_clock=True)
        p.currency_unit = unit
        for _ in range(3):
            p.update(FirmMetrics(total=10, revenue=20*unit), 2*unit)
        rivals.append(p)
    assert all(p.ema_gap == rivals[0].ema_gap and p.fast_gap == rivals[0].fast_gap for p in rivals)


def test_rival_does_not_treat_its_own_promotion_as_competitor_price_pressure():
    from platform_mdp import PlatformObservationModel, ObservationConfig
    p = PlatformObservationModel(seed=1, config=ObservationConfig(
        require_observed_quote_bins=True, quote_noise_dollars=0., quote_missing_probability=0., quote_probe_delay_steps=0))
    p.ingest(own_metrics={}, supply_metrics={}, gap_sign=-1., crowd_stats={
        'distance_bin_2_5_count': 5, 'distance_bin_2_5_price_gap_mean': -4.,
        'distance_bin_2_5_firm2_promotion_discount': 4.})
    core = Core.__new__(Core)
    core.platform_observers = {'Firm2': p}
    assert core._observed_own_minus_competitor_gap('Firm2') == 0.


def test_retaliation_can_recover_without_a_permanent_downward_ratchet():
    p = FirmMemoryBestResponsePricer(seed=2, managed_keys=['base_fare'],
        operational_clock=True, lag_steps=2, hold_mean_steps=2)
    p._ensure_initialized(p._anchor_map(3., .4, 1.5, 2., 0.))
    m = FirmMetrics(total=100, revenue=100, chosen=50, completed=50)
    for _ in range(2):
        p.update(m, 2.)
    assert p.undercut_steps == 0
    p.act(**ANCHORS)
    cut = p.overrides.base_fare
    assert cut < 3
    for _ in range(20):
        p.update(m, 0.)
        p.act(**ANCHORS)
        if p.last_reason == 'retaliation_recovery':
            assert p.overrides.base_fare > cut
            break
    else:
        pytest.fail('retaliation never recovered after the undercut ended')


def test_memory_perturbations_include_timing_and_response_parameters():
    p = FirmMemoryBestResponsePricer()
    draw = ControllerPerturbation().apply(p, np.random.default_rng(5))
    assert {'lag_steps', 'hold_mean_steps', 'accommodation_steps', 'undercut_threshold', 'accommodation_cap'} <= draw.keys()
    assert p.lag_steps >= 1 and p.hold_mean_steps >= 1


def test_oracle_expresses_opposing_and_compound_recovery_actions():
    labels = {'rival_promotion': .6, 'driver_shortage': 1.8,
              'driver_shortage+rival_promotion': 1.2,
              'driver_shortage_recovery+rival_promotion': .9}
    specs = oracle_specs(CASES['S6_shortage_during_promotion'], 1., labels)
    table = json.loads(specs[0].split(':', 1)[1])
    assert {k: v[0] for k, v in table.items() if k != 'calm'} == labels


def test_unknown_oracle_regime_is_an_error_not_a_calm_policy():
    core = Core.__new__(Core)
    core._regime_key_for_hour = lambda hour: 'new_combined_event'
    with pytest.raises(ValueError, match='no target'):
        core._baseline_firm1_action('oracle:{"calm":[1,1]}', step_index=0,
            action_mask=np.ones(1), day_of_week=0, hour=0, weather='clear',
            presampled=None, customers_per_step=1)
