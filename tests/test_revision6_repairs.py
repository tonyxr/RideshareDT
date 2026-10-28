"""Direct regressions for the twelve October-18 defects, not profit gates."""
import copy
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pytest
from driver_supply import DriverSupplyConfig, DriverSupplyLayer, FirmDriverBatchState
from pricing_models import (FirmMetrics, FirmMPCGridPricer, FirmMemoryBestResponsePricer,
    FirmAdaptiveBestResponsePricer, FirmAggressiveAdaptiveBestResponsePricer,
    FirmPIPriceGapPricer, FirmQueueServiceThresholdPricer, FirmRegionSupplyDemandPricer,
    FirmSurgeDriverIncentivePricer)
from calibration_artifact import SUPPLY_CONTRACT, validate_supply_point
from scripts.v26r_cases import CASES, STATIC_GRID, oracle_specs, required_regime_labels

RIVALS = [FirmAdaptiveBestResponsePricer, FirmAggressiveAdaptiveBestResponsePricer,
    FirmMemoryBestResponsePricer, FirmMPCGridPricer, FirmPIPriceGapPricer,
    FirmQueueServiceThresholdPricer, FirmRegionSupplyDemandPricer, FirmSurgeDriverIncentivePricer]


def supply(**kwargs):
    args = dict(supply_model='v2', persistent_stock=True, base_active_drivers=40,
        reference_earnings_per_hour=12., linear_demand_scaling=True,
        availability_block_hours=4., hour_profile={h: 1. for h in range(24)})
    return DriverSupplyLayer(seed=47, config=DriverSupplyConfig(**(args | kwargs)))


@pytest.mark.parametrize('kappa', [.3, 1.])
@pytest.mark.parametrize('empty_days', [1, 40])
def test_empty_blocks_do_not_absorb_supply_and_positive_pay_recovers(kappa, empty_days):
    s = supply(stock_adjustment_rate=kappa)
    initial = s.online_stock.copy()
    s.advance_day()
    assert s.online_stock == initial  # No fictitious day before the opening.
    for _ in range(empty_days):
        for block in range(6):
            s.begin_batch(50, block*4, 'clear')
            s.end_batch()  # Online, no rides: genuinely zero earned income.
        s.advance_day()
    s.set_pay_index('Firm1', 1.2)
    observed = []
    for _ in range(10):
        for block in range(6):
            s.begin_batch(50, block*4, 'clear')
            state = s._batch_states['Firm1']
            observed.append(state.active_drivers)
            state.total_driver_pay = state.total_online_minutes / 60 * 14.4
            s.end_batch()
        s.advance_day()
    assert sum(observed) > 0
    assert s.online_stock['Firm1'] > .9
    assert s.reference_earnings['Firm1'] == 12.


def test_complete_outage_and_exact_zero_stock_reenter_without_driver_floor():
    s = supply(initial_online_stock={'Firm1': 0., 'Firm2': 1.})
    s.apply_shock(firm_capacity={'Firm1': 0.})
    for _ in range(6):
        s.begin_batch(50, 12, 'clear')
        assert s._batch_states['Firm1'].active_drivers == 0
        s.end_batch()
    s.advance_day()
    assert s.online_stock['Firm1'] == pytest.approx(.3)
    s.apply_shock()
    counts = []
    for _ in range(60):
        s.begin_batch(50, 12, 'clear')
        counts.append(s._batch_states['Firm1'].active_drivers)
        s.end_batch()
    assert max(counts) > 0
    s.set_pay_index('Firm1', 0.)
    s.begin_batch(50, 12, 'clear')
    assert s._batch_states['Firm1'].active_drivers == 0  # No hidden pay/driver floor.


def test_actual_zero_income_still_reduces_daily_stock():
    s = supply()
    s.begin_batch(500, 12, 'clear'); s.end_batch(); s.advance_day()
    assert s.online_stock['Firm1'] == pytest.approx(.7)
    assert s.expected_block_earnings('Firm1') == 12.  # Posted, not realized.


def test_fractional_fleets_preserve_expected_volume_and_small_pay_response():
    means = []
    for customers, pay in [(50, 1.), (300, 1.), (50, 1.1)]:
        s = supply(base_active_drivers=37)
        s.set_pay_index('Firm1', pay)
        counts = []
        for _ in range(4000):
            s.begin_batch(customers, 12, 'clear')
            counts.append(s._batch_states['Firm1'].active_drivers)
        means.append(np.mean(counts))
        expected = 37*customers/500*pay**.2
        assert abs(means[-1]-expected) < .035
    assert means[2] > means[0]
    assert abs(means[1]/6-means[0]) < .04
    # This proves unbiased counts, NOT equality of nonlinear fulfillment.


@pytest.mark.parametrize('factory', RIVALS)
def test_daily_ema_clock_and_reference_use_complete_day_and_online_units(factory):
    def make(steps):
        p = factory(seed=1)
        p.operational_clock = True
        p.reference_window_steps = steps
        p.earnings_reference = 12.
        return p
    daily, blocks = make(1), make(6)
    for p in (daily, blocks):
        for _ in range(p.reference_window_steps):
            p.update(FirmMetrics(total=10, revenue=20, chosen=5, completed=5), 0.,
                {'driver_earnings_per_online_hour': 12., 'driver_earnings_per_hour': 60.})
        assert p.ema_driver_earnings == 12.
        assert p.ema_revenue == p.revenue_target == 2.
    for _ in range(10):
        for p in (daily, blocks):
            for _ in range(p.reference_window_steps):
                p.update(FirmMetrics(total=10, revenue=10, chosen=2, completed=2), 3.,
                    {'driver_earnings_per_online_hour': 9., 'driver_earnings_per_hour': 80.})
        for key in ('ema_revenue', 'ema_gap', 'ema_choice_share', 'ema_driver_earnings'):
            assert getattr(daily, key) == pytest.approx(getattr(blocks, key), abs=1e-12)
        if factory is FirmMemoryBestResponsePricer:
            assert daily.fast_gap == pytest.approx(blocks.fast_gap, abs=1e-12)


def test_revenue_reference_and_ema_do_not_depend_on_first_block_order():
    values = []
    for order in ([1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]):
        p = FirmMemoryBestResponsePricer(operational_clock=True)
        for value in order:
            p.update(FirmMetrics(total=10, revenue=10*value), 0.)
        values.append((p.ema_revenue, p.revenue_target))
    assert values == [(3.5, 3.5), (3.5, 3.5)]


@pytest.mark.parametrize('gap', [-6., -3., -1., 1., 3., 6.])
def test_mpc_moves_when_candidate_improves_hold_at_large_gap(gap):
    p = FirmMPCGridPricer(seed=1, managed_keys=['base_fare'])
    p.operational_clock = True
    p.ema_revenue = p.revenue_target = 8.
    p.ema_gap = gap
    p.ema_choice_share = p.ema_completed_share = .5
    hold = p._candidate_score('base_fare', 0)
    assert max(p._candidate_score('base_fare', d) for d in (-1, 1)) > hold
    p.act(city_base=3., city_pmin=.4, city_pmile=1.5, city_booking=2., city_airport=0., hour=12, weather='clear')
    assert p.last_selected_key == 'base_fare'
    assert p.last_direction == (-1 if gap > 0 else 1)


@pytest.mark.parametrize('key,value', [('eta_day', .7), ('eta_hour', .5),
    ('stock_adjustment_rate', .8), ('supply_model', 'v1'), ('fleet_scale', .8)])
def test_saved_supply_point_cannot_override_explicit_settings(key, value):
    point = dict(contract=SUPPLY_CONTRACT, config=asdict(supply().config),
                 fleet_scale=1., calibration=dict(sample_size=50))
    validate_supply_point(point, customers=50, explicit={})
    with pytest.raises(ValueError, match='conflicts'):
        validate_supply_point(point, customers=50, explicit={key: value})
    with pytest.raises(ValueError, match='volume'):
        validate_supply_point(point, customers=300, explicit={})
    point.pop('contract')
    with pytest.raises(ValueError, match='predates'):
        validate_supply_point(point, customers=50, explicit={})


def test_supply_features_are_scale_invariant_and_unsaturated():
    values = []
    for fleet, reference in [(20, 8.), (200, 24.), (2000, 80.)]:
        s = supply(base_active_drivers=fleet, reference_earnings_per_hour=reference)
        s._feature_capacity_reference = fleet
        row = []
        for ratio in (1., 2., 4.):
            s.smoothed_states['Firm1'] = FirmDriverBatchState(active_drivers=int(fleet*ratio),
                total_online_minutes=60., total_driver_pay=reference*ratio)
            row.append(s.state_features_for_firm('Firm1')[[0, 6]])
        values.append(row)
        np.testing.assert_allclose(row, np.repeat(np.arcsinh([1., 2., 4.])[:, None], 2, axis=1))
    np.testing.assert_array_equal(values[0], values[1])
    np.testing.assert_array_equal(values[0], values[2])


@pytest.mark.parametrize('name', CASES)
def test_oracle_has_full_scripted_label_support_even_without_fitted_label(name):
    labels = required_regime_labels(CASES[name])
    specs = oracle_specs(CASES[name], 1., {}, required_labels=labels)
    if set(labels) == {'calm'}:
        assert not specs
        return
    tables = [json.loads(s.split(':', 1)[1]) for s in specs]
    for table in tables:
        assert set(table) == set(labels)
    for label in set(labels)-{'calm'}:
        assert {table[label][0] for table in tables} == set(STATIC_GRID)


def test_preburnin_label_initializers_use_only_validation_data(tmp_path):
    from scripts.v26r_case_eval import best_regime_targets_from, spec_key
    for target in STATIC_GRID:
        spec = f'static:{target:g}'
        data = dict(seed_set='val', steps_per_day=6, traces=[
            dict(rollout='r__opening_1', step=0, regime='winter_storm_forecast', reward=-(target-1.2)**2),
            dict(rollout='r__opening_1', step=90, regime='calm', reward=1.)])
        (tmp_path/(spec_key(spec)+'.json')).write_text(json.dumps(data))
    result = best_regime_targets_from(tmp_path, rival='r', burnin=15)
    assert result['winter_storm_forecast'] == 1.2


@pytest.mark.parametrize('corrected,revision,expected', [(False, 4, False), (False, 1, False),
    (True, 1, False), (True, 2, False), (True, 3, False), (True, 4, True)])
def test_epoch_kl_stop_never_changes_legacy_or_revision_one_to_three(corrected, revision, expected):
    from test_learning_corrections import recurrent_agent
    agent = recurrent_agent(learner_revision=revision, hard_kl_stop=True)
    agent.learning_corrections = corrected
    agent.target_kl = .02
    assert not agent._stop_for_epoch_kl(.029)
    assert agent._stop_for_epoch_kl(.04) is expected
    assert agent._stop_for_epoch_kl(20.) is expected
    agent.hard_kl_stop = False
    assert not agent._stop_for_epoch_kl(20.)


def test_smoke_cli_has_actual_volume_correct_cadence_and_unchanged_objective(tmp_path):
    from scripts.run_revision6_smoke import training_command
    from platform_mdp import TrainingStageScheduler
    for slug in ('nyc', 'boston', 'chicago'):
        cmd = training_command(tmp_path/slug, slug, tmp_path/'calibration.json')
        for flag, value in {'--train_customers': '50', '--eval_customers': '50',
            '--opponent_profit_lead_weight': '.6', '--ppo_rollout_episodes': '2',
            '--ppo_min_rollout_transitions': '30', '--learner_revision': '4'}.items():
            assert float(cmd[cmd.index(flag)+1]) == float(value)
    scheduler = TrainingStageScheduler(stage_fractions=(.1, .108, .792))
    assert [scheduler.stage_window(day, 2400) for day in (0, 240, 500)] == [(0, 240), (240, 500), (500, 2400)]


def test_supply_opening_restores_observation_clock():
    s = supply()
    s.begin_batch(50, 12, 'clear'); s.end_batch()
    s.reset_stock()
    s.config.initial_online_stock = {'Firm1': .8, 'Firm2': .9}
    s.reset_stock()
    s.advance_day()
    assert s.online_stock == {'Firm1': .8, 'Firm2': .9}
