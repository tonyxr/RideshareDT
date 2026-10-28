"""Real cached-city plumbing with synthetic demand parameters, not calibration."""
import copy
import json
import numpy as np
import pytest
from Core import Core
from joint_demand import JointChoiceModel
from test_joint_demand import specification
from test_revision5_integration import core
from scripts.audit_city_calibration import load_core
from scripts.audit_joint_demand import records, responses
from calibration_artifact import supply_point


def install(core):
    core.configure_joint_demand(specification(core.market_name, typical_fare=core.city_reference_fare))


def test_real_city_choice_rollout_and_counterfactual_rollback(core):
    install(core)
    core._reset_adaptive_episode(tariff_reset_fraction=0., episode_seed=771)
    profiles = core._sample_profiles_from_pool(12, generalization_fraction=0.)
    before = core._snapshot_counterfactual_state(profiles)
    samples = []
    for _ in range(2):
        core._restore_counterfactual_state(before)
        result = core.simulate_batch(day_of_week=2, hour=12, weather='clear', customers_per_step=12,
            sampled_profiles=profiles, supply_reference_customers=12, collect_rows=True)
        samples.append((result, copy.deepcopy(core.rider_loyalty)))
        assert isinstance(core.choice_model, JointChoiceModel)
    assert samples[0] == samples[1]


def test_new_demand_changes_participation_and_sweep_order_does_not_change_results(core):
    install(core)
    rows = records(core, evaluation=True, n=24, seed=99)
    observed = responses(core.choice_model, rows, [1., 1.3])
    assert observed['curves'][1]['own_only']['NoRide'] > observed['curves'][0]['own_only']['NoRide']
    kwargs = dict(rivals=['static'], openings=1, horizon=6, customers_per_step=5,
                  steps_per_day=6, firm1_interval_steps=6, opponent_interval_steps=6, policy_seed=918)
    a = core.run_policy_sweep(specs=['static:.8', 'static:1.3'], **kwargs)
    b = core.run_policy_sweep(specs=['static:1.3', 'static:.8'], **kwargs)
    assert a['traces'] == b['traces']


def test_model_artifact_carries_and_enforces_demand_contract(core, tmp_path):
    import torch
    install(core)
    path = tmp_path/'joint-policy.pt'
    core.save_trained_model(str(path))
    saved = torch.load(path, weights_only=False, map_location='cpu')
    assert saved['demand_calibration'] == core.choice_model.demand_spec
    assert saved['supply_operating_point']['demand_calibration'] == core.choice_model.demand_spec
    core.load_trained_model(str(path))
    core.choice_model.demand_spec['beta'] *= 2
    with pytest.raises(ValueError, match='demand contracts'):
        core.load_trained_model(str(path))


def test_supply_point_from_old_demand_cannot_be_reused(core):
    point = supply_point(core)
    install(core)
    with pytest.raises(ValueError, match='different demand model'):
        core.configure_v26r(supply_model='v2', supply_operating_point=point,
                            supply_calibration_customers=point['calibration']['sample_size'])


@pytest.mark.parametrize('slug,city', [('nyc', 'New York City'), ('boston', 'Boston'), ('chicago', 'Chicago')])
def test_cli_installs_demand_before_capacity_fit(slug, city, tmp_path, monkeypatch):
    path = tmp_path/'demand.json'
    path.write_text(json.dumps(specification(city)))
    seen = []
    def capacity(core, **kwargs):
        seen.append(isinstance(core.choice_model, JointChoiceModel))
        return dict(enabled=False, test_only=True, sample_size=kwargs['sample_size'])
    monkeypatch.setattr('city_supply_calibration.calibrate_capacity', capacity)
    result = load_core(slug, tmp_path, integration=True, demand_artifact=path)
    assert seen == [True]
    assert result.v26_settings['demand_calibration']['city'] == city


def test_raw_thresholds_not_silently_clipped_for_joint_model(core, tmp_path):
    install(core)
    source = json.loads(open(core.threshold_cache_path).readline())
    source['PriceThreshold'] = 12.5
    path = tmp_path/'profiles.jsonl'
    path.write_text(json.dumps(source)+'\n')
    core.threshold_cache_path = str(path)
    assert core._try_load_threshold_cache(1)
    assert core.synthetic_profile_pool[0]['PriceThreshold'] == 12.5


def test_relative_units_reject_legacy_beta_and_accept_matching_refit(core):
    core.price_sensitivity_units = 'relative'
    with pytest.raises(ValueError, match='price-sensitivity units'):
        core.configure_joint_demand(
            specification(core.market_name, typical_fare=core.city_reference_fare)
        )
    spec = specification(
        core.market_name,
        typical_fare=core.city_reference_fare,
        price_sensitivity_units='relative',
    )
    core.configure_joint_demand(spec)
    assert core.choice_model.price_sensitivity_units == 'relative'


@pytest.mark.parametrize('slug', ['nyc', 'boston', 'chicago'])
def test_actual_audit_entrypoint_all_prior_arms_no_supply_or_training(slug, tmp_path, monkeypatch):
    from scripts.audit_joint_demand import audit
    def prohibited(*args, **kwargs):
        raise AssertionError('demand audit must not fit supply or run training')
    monkeypatch.setattr(Core, '_calibrate_driver_capacity', prohibited)
    monkeypatch.setattr(Core, 'run_experiment', prohibited)
    report = audit(slug, tmp_path, [-.9, -1.15, -1.4], n=12)
    assert [a['target'] for a in report['arms']] == [-.9, -1.15, -1.4]
    for arm in report['arms']:
        spec = json.loads((tmp_path/slug/arm['artifact']).read_text())
        assert spec['evidence']['fit_partition'] == 'training'
        assert spec['evidence']['heldout_partition'] == 'evaluation'
        assert arm['heldout']['n'] == 12
    assert (tmp_path/slug/'demand_curves.png').stat().st_size > 1000
