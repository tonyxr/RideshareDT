"""Real cached-city simulator tests; no policy training or performance claim."""
import copy
import numpy as np
import pytest
from Core import Core
from scripts.audit_city_calibration import load_core, elasticity
from city_supply_calibration import calibrate_capacity as actual_calibration


@pytest.fixture(params=['nyc', 'boston', 'chicago'])
def core(request, tmp_path, monkeypatch):
    installed = []
    def capacity(core, **kwargs):
        installed.append((core.driver_supply.supply_v2, core.loyalty_v2,
                          core.driver_supply.config.reference_earnings_per_hour))
        return dict(enabled=False, test_only=True, sample_size=kwargs['sample_size'])
    # Capacity parameter fitting is a separately confirmed empirical experiment;
    # this fixture tests the installed production simulator and ordering.
    monkeypatch.setattr('city_supply_calibration.calibrate_capacity', capacity)
    result = load_core(request.param, tmp_path, integration=True)
    assert installed and installed[0][0:2] == (True, True) and installed[0][2] > 0
    assert result.long_term_profit_reward_config.opponent_profit_lead_weight == .6
    assert min(result.tariff_targets) == .6
    return result


def test_real_branches_preserve_real_transition_and_rider_aliases(core):
    core._reset_adaptive_episode(tariff_reset_fraction=.15, episode_seed=151)
    profiles = core._sample_profiles_from_pool(12, generalization_fraction=0.)
    snapshot = core._snapshot_counterfactual_state(profiles)
    original_stocks = copy.deepcopy(core.rider_loyalty)
    samples = []
    for branches in (0, 3, len(core.firm1.action_to_steps)):
        core._restore_counterfactual_state(snapshot)
        core.counterfactual_branches = branches
        core._counterfactual_action_values(selected_action=1, policy_top_action=1, policy_second_action=2,
            action_mask=core.firm1.feasible_action_mask(core.market), day_of_week=2,
            hour=12, weather='clear', customers_per_step=12, sampled_profiles=profiles,
            supply_reference_customers=12)
        assert core.rider_loyalty == original_stocks
        core.firm1.apply_action(1, core.market)
        result = core.simulate_batch(day_of_week=2, hour=12, weather='clear', customers_per_step=12,
            sampled_profiles=profiles, supply_reference_customers=12, collect_rows=True)
        samples.append((result, copy.deepcopy(core.rider_loyalty), copy.deepcopy(core.last_welfare_stats)))
    assert samples[0] == samples[1] == samples[2]


def test_validation_cannot_mutate_training_rider_or_regime_state(core):
    core._reset_adaptive_episode(tariff_reset_fraction=.15, episode_seed=152)
    pool = core._active_rider_pool
    state = copy.deepcopy(core.rider_loyalty)
    core._regime_rival_switch_applied = {'test': True}
    before = copy.deepcopy(core._regime_rival_switch_applied)
    metrics = core.run_policy_sweep(specs=['static:1'], rivals=['static'], openings=1,
        horizon=12, customers_per_step=5, steps_per_day=6, firm1_interval_steps=6,
        opponent_interval_steps=6, policy_seed=3151)
    assert metrics['traces']['static:1']
    assert core._active_rider_pool is pool
    assert core.rider_loyalty == state
    assert core._regime_rival_switch_applied == before
    for rider in pool:
        assert rider['_LoyaltyV2'] is core.rider_loyalty[rider['_RiderID']]


def test_policy_order_invariance_with_actual_loyalty_and_supply(core):
    kwargs = dict(rivals=['adaptive_best_response_memory'], openings=1, horizon=12,
        customers_per_step=5, steps_per_day=6, firm1_interval_steps=6,
        opponent_interval_steps=6, policy_seed=3152)
    a = core.run_policy_sweep(specs=['static:0.6', 'static:1.4'], **kwargs)
    b = core.run_policy_sweep(specs=['static:1.4', 'static:0.6'], **kwargs)
    assert a['traces'] == b['traces']
    assert a['results'] == b['results']


def test_myopic_scores_whole_day_and_restores_history_supply_and_loyalty(core):
    import pickle
    core._reset_adaptive_episode(tariff_reset_fraction=0., episode_seed=819)
    profiles = core._sample_profiles_from_pool(5, generalization_fraction=0.)
    before = core._snapshot_counterfactual_state(profiles)
    history = core.firm1.agent._actor_history
    kwargs = dict(step_index=0, holding_steps=6, opponent_interval_steps=6,
        day_of_week=2, hour=0, weather='clear', presampled=(5, profiles),
        customers_per_step=5, evaluation_profiles=False)
    scores = core._holding_interval_action_values([1, 4], **kwargs)
    reverse = core._holding_interval_action_values([4, 1], **kwargs)
    assert scores == reverse and all(np.isfinite(v) for v in scores.values())
    after = core._snapshot_counterfactual_state(profiles)
    for key in ('riders', 'market', 'choice_model', 'firm2', 'driver_supply', 'observers', 'tracked'):
        assert pickle.dumps(before[key]) == pickle.dumps(after[key]), key
    assert core.firm1.agent._actor_history is history


def test_city_units_use_actual_price_level_and_no_heldout_checkpoint_selection(core):
    choice = core.active_calibration['choice']
    reference = choice['reference_fare'] * choice['price_level_index']
    assert core.city_reference_fare == pytest.approx(reference)
    assert core.observation_config.gap_scale_dollars == pytest.approx(.1 * reference)
    assert core.observation_config.telemetry_delay_steps == 0
    report = core._adaptive_policy_validation(customers_per_step=5, profile_sample_size=5,
        horizon=6, steps_per_day=6, opponent_interval_steps=6, firm1_interval_steps=6,
        rollouts_override=[('static__opening_1', 'static')], evaluation_profiles=False)
    assert report['profile_partition'] == 'training_validation'
    assert report['validation_profile_source'] == 'training_partition_validation'
    assert report['horizon_days'] == 1
    assert report['checkpoint_eligible'] and not report['durability_horizon_sufficient']


def test_zero_airport_anchor_cannot_dilute_the_city_tariff_index(core):
    from pricing_models import CoefficientOverrides
    for target in (.6, 1., 2.):
        values = {key: float(getattr(core.market.curr_market, key)) * target
                  for key in ('base_fare', 'per_mile', 'per_minute', 'booking_fee', 'airport_fee')}
        for firm in (core.firm1, core.firm2):
            firm.overrides = CoefficientOverrides(**values)
            assert core._firm_tariff_index(firm) == pytest.approx(target)
            core.regime_market = False  # Calm calibration disables the shock process, not city units.
            assert core._firm_tariff_index(firm) == pytest.approx(target)
            core.regime_market = True


def test_privileged_money_features_are_city_relative_and_do_not_saturate(core):
    public = np.zeros(core.platform_observation_dim, dtype=np.float32)
    offset = core.platform_observation_dim + 10
    observed = []
    for multiple in (1., 4., 8.):
        core.last_profitpr = multiple * core.city_reference_fare
        core.last_firm2_profitpr = -multiple * core.city_reference_fare
        critic = core._build_privileged_critic_state(public)
        observed.append(critic[offset + 2])
        assert critic[offset + 2] == pytest.approx(np.arcsinh(multiple))
        assert critic[offset + 6] == pytest.approx(-np.arcsinh(multiple))
    assert observed[0] < observed[1] < observed[2]


def test_saved_operating_point_is_used_without_recalibration(core, monkeypatch):
    from calibration_artifact import supply_point
    point = supply_point(core)
    point['config']['base_active_drivers'] = 137
    point['config']['reference_earnings_per_hour'] = 9.25
    point['config']['initial_online_stock'] = {'Firm1': .83, 'Firm2': .91}
    def forbidden(*args, **kwargs):
        pytest.fail('saved operating point was recalibrated')
    monkeypatch.setattr('city_supply_calibration.calibrate_capacity', forbidden)
    core.configure_v26r(supply_model='v2', supply_operating_point=point,
                       supply_calibration_customers=point['calibration']['sample_size'])
    assert core.driver_supply.config.base_active_drivers == 137
    assert core.driver_supply.reference_earnings == {'Firm1': 9.25, 'Firm2': 9.25}
    assert core.driver_supply.online_stock == {'Firm1': .83, 'Firm2': .91}


def test_elasticity_audit_is_rng_read_only_and_finite(core):
    before = [copy.deepcopy(owner.rng.bit_generator.state) for owner in
              (core, core.choice_model, core.driver_supply, core.market)]
    stocks = copy.deepcopy(core.rider_loyalty)
    result = elasticity(core, n=12)
    assert before == [owner.rng.bit_generator.state for owner in
                      (core, core.choice_model, core.driver_supply, core.market)]
    assert core.rider_loyalty == stocks
    assert result['typical_fare_median'] > 0
    assert len(result['elasticities']) == 18
    assert all(np.isfinite(row['own_elasticity']) and np.isfinite(row['market_elasticity'])
               for row in result['elasticities'])


def test_real_capacity_calibration_uses_supported_market_api_and_restores_state(core):
    core.market_dynamics.capacity_target_fulfillment = .95
    before = copy.deepcopy(core.market.shocks.snapshot())
    riders = copy.deepcopy(core.rider_loyalty)
    result = actual_calibration(core, sample_size=20, days=3, burnin_days=1, iterations=1)
    assert result['supply_model'] == 'v2' and len(result['candidates']) == 4
    assert core.driver_supply.config.base_active_drivers == result['base_active_drivers']
    assert core.market.shocks.snapshot() == before
    assert core.rider_loyalty == riders


def test_capacity_failure_restores_fleet_shocks_and_loyalty(core, monkeypatch):
    core.market_dynamics.capacity_target_fulfillment = .95
    before = core._snapshot_counterfactual_state()
    def fail(**kwargs):
        raise RuntimeError('injected capacity failure')
    monkeypatch.setattr(core, 'simulate_batch', fail)
    with pytest.raises(RuntimeError, match='injected capacity failure'):
        actual_calibration(core, sample_size=20, days=3, burnin_days=1, iterations=1)
    assert core.driver_supply.config == before['driver_supply'].config
    assert core.market.shocks.snapshot() == before['market'].shocks.snapshot()
    assert core.rider_loyalty == before['riders']['ledger']


@pytest.mark.parametrize('spillover', [False, True])
def test_platform_promotion_preserves_driver_pay_on_direct_and_retry_dispatch(core, monkeypatch, spillover):
    from choice_models import ChoiceResult
    from driver_supply import DispatchResult
    core._reset_adaptive_episode(tariff_reset_fraction=0., episode_seed=515)
    core._firm2_regime_price_multiplier = .7
    core.market_dynamics.spillover_probability = 1.
    monkeypatch.setattr(core.choice_model, 'choose', lambda *args, **kwargs:
                        ChoiceResult('Firm1' if spillover else 'Firm2', [], 'test'))
    monkeypatch.setattr(core.choice_model, '_no_ride_probability', lambda *args: 0.)
    payments = []
    def dispatch(firm, fare, *args, **kwargs):
        if firm == 'Firm1':
            return DispatchResult(fulfilled=False, reject_reason='test only')
        reference = kwargs['pay_reference_fare']
        payments.append((fare, reference))
        return DispatchResult(fulfilled=True, driver_pay=.72*reference, platform_variable_cost=.45)
    monkeypatch.setattr(core.driver_supply, 'dispatch', dispatch)
    profiles = core._sample_profiles_from_pool(5, generalization_fraction=0.)
    _, _, firm2, _, _, _ = core.simulate_batch(day_of_week=2, hour=12, weather='clear',
        customers_per_step=5, sampled_profiles=profiles, collect_rows=True, supply_reference_customers=5)
    assert len(payments) == 5
    assert all(fare == pytest.approx(.7*reference) for fare, reference in payments)
    assert firm2.driver_pay == pytest.approx(sum(.72*ref for _, ref in payments))
    assert firm2.profit == pytest.approx(sum(fare-.72*ref-.45 for fare, ref in payments))
