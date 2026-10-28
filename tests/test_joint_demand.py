"""Demand equations, invariances and contracts; not evidence that RL wins."""
import copy
import numpy as np
import pytest
from choice_models import CognitiveChoiceModel
from joint_demand import CONTRACT, JointChoiceModel, fit_local_own_elasticity

PROFILE = dict(IncomeBracket='50k-100k', HouseholdSize=2, PriceThreshold=2.5,
               LoyaltyType='New', LoyaltyStrength=0, LoyaltyFirm=None)
SCENARIO = dict(ReferenceFare=23.22, DistanceMiles=4., DurationMinutes=15.,
                Hour=12, DayOfWeek=2, Weather='clear', Airport=False, Service='economy',
                WaitEstimateFirm1=5., WaitEstimateFirm2=5.)


def specification(city='New York City', **kwargs):
    return dict(format=CONTRACT, city=city, beta=2., typical_fare=23.22,
                threshold_reference=2.5, evidence={'type': 'synthetic_test_only'}) | kwargs


def model(**kwargs):
    legacy = CognitiveChoiceModel(seed=51)
    legacy.make_symmetric()
    legacy.fare_normalized = True
    return JointChoiceModel.from_legacy(legacy, specification(**kwargs))


@pytest.mark.parametrize('threshold', [.05, .5, 2.5, 5., 50.])
@pytest.mark.parametrize('fare', [5., 23.22, 60., 200.])
def test_all_alternatives_jointly_respond_without_price_cliffs(threshold, fare):
    m = model()
    profile, scenario = dict(PROFILE, PriceThreshold=threshold), dict(SCENARIO, ReferenceFare=fare)
    values = [m.probabilities(profile, scenario, fare*x, fare) for x in (.6, .8, 1., 1.1, 1.3, 2., 10.)]
    assert all(sum(v.values()) == pytest.approx(1.) for v in values)
    assert all(all(np.isfinite(p) and 0 <= p <= 1 for p in v.values()) for v in values)
    assert all(a['Firm1'] >= b['Firm1'] for a, b in zip(values, values[1:]))
    assert all(a['Firm2'] <= b['Firm2'] and a['NoRide'] <= b['NoRide'] for a, b in zip(values, values[1:]))
    assert values[4]['NoRide'] > values[2]['NoRide']  # Former min-price gate held this constant.


def test_exact_elasticity_fit_recovers_synthetic_parameter_and_joint_market_response():
    m = model()
    records = [(dict(PROFILE, PriceThreshold=t), dict(SCENARIO, ReferenceFare=f), f)
               for t, f in [(1., 8.), (2.5, 23.22), (4., 60.)]]
    for target in (-.9, -1.15, -1.4):
        beta = fit_local_own_elasticity(m, records, target)
        m.demand_spec['beta'] = beta
        def demand(multiplier, both=False):
            rows = [m.probabilities(p, s, f*multiplier, f*(multiplier if both else 1)) for p,s,f in records]
            return np.mean([r['Firm1'] for r in rows]), np.mean([1-r['NoRide'] for r in rows])
        delta = 1e-5
        low, high = demand(1-delta)[0], demand(1+delta)[0]
        measured = np.log(high/low)/np.log((1+delta)/(1-delta))
        assert measured == pytest.approx(target, abs=1e-7)
        assert demand(1.05, True)[1] < demand(.95, True)[1]


@pytest.mark.parametrize('city,typical,threshold,references', [
    ('New York City', 23.02, 2.5, (11., 23.02, 48.)),
    ('Boston', 17.165, 1.85, (8., 17.165, 37.)),
    ('Chicago', 12.97, 1.8, (6., 12.97, 31.)),
])
def test_relative_units_refit_same_elasticity_target_per_city(
        city, typical, threshold, references):
    m = model(
        city=city,
        typical_fare=typical,
        threshold_reference=threshold,
        price_sensitivity_units='relative',
    )
    rows = [
        (dict(PROFILE, PriceThreshold=value),
         dict(SCENARIO, ReferenceFare=fare), fare)
        for value, fare in zip((.6 * threshold, threshold, 1.8 * threshold), references)
    ]
    target = -1.15
    m.demand_spec['beta'] = fit_local_own_elasticity(m, rows, target)
    delta = 1e-5
    def level(multiplier):
        return np.mean([
            m.probabilities(profile, scenario, fare * multiplier, fare)['Firm1']
            for profile, scenario, fare in rows
        ])
    measured = np.log(level(1 + delta) / level(1 - delta)) / np.log((1 + delta) / (1 - delta))
    assert measured == pytest.approx(target, rel=.01)


def test_relative_units_are_trip_reference_scaled_and_require_declared_refit():
    relative = model(price_sensitivity_units='relative')
    _, low_slope, low_reference = relative.components(
        PROFILE, dict(SCENARIO, ReferenceFare=10.))
    _, high_slope, high_reference = relative.components(
        PROFILE, dict(SCENARIO, ReferenceFare=50.))
    assert low_slope * low_reference == pytest.approx(high_slope * high_reference)
    with pytest.raises(ValueError, match='price_sensitivity_units'):
        model(price_sensitivity_units='unknown')


def test_currency_invariance_and_firm_exchange_symmetry():
    values = []
    for scale in (.01, .53, 1., 2., 100.):
        m = model(typical_fare=23.22*scale, threshold_reference=2.5*scale)
        m.currency_unit = m.price_level_index = scale
        m.outside_option_base_fare = 2.9*scale
        p = dict(PROFILE, PriceThreshold=2.5*scale,
                 ReservationPrice=35*scale, OutsideOptionCost=4*scale,
                 _LoyaltyV2={'Firm1': .4, 'Firm2': .1})
        # Explicit costs below legacy clipping ceilings; use implicit scalable priors at extremes.
        p.pop('ReservationPrice'); p.pop('OutsideOptionCost')
        s = dict(SCENARIO, ReferenceFare=23.22*scale)
        v = m.probabilities(p, s, 25*scale, 20*scale)
        swapped = dict(p, _LoyaltyV2={'Firm1': .1, 'Firm2': .4})
        w = m.probabilities(swapped, s, 20*scale, 25*scale)
        assert v['Firm1'] == pytest.approx(w['Firm2'])
        assert v['NoRide'] == pytest.approx(w['NoRide'])
        values.append(v)
    for v in values:
        assert v == pytest.approx(values[0])


def test_rng_purity_sampling_and_no_state_changes():
    m = model()
    p, s = copy.deepcopy(PROFILE), copy.deepcopy(SCENARIO)
    state = copy.deepcopy(m.rng.bit_generator.state)
    expected = m.probabilities(p, s, 25., 23.22)
    assert m.rng.bit_generator.state == state and p == PROFILE and s == SCENARIO
    counts = dict.fromkeys(expected, 0)
    for _ in range(6000):
        counts[m.choose(p, s, 25., 23.22).choice] += 1
    for name in expected:
        assert counts[name]/6000 == pytest.approx(expected[name], abs=.025)


def test_spillover_uses_same_utilities_without_unavailable_firm():
    m = model()
    for fare in (10., 23.22, 40., 1000.):
        p = m.probabilities(PROFILE, SCENARIO, 23.22, fare)
        actual = m.retry_probability(PROFILE, SCENARIO, 'Firm2', fare)
        assert actual == pytest.approx(p['Firm2']/(p['Firm2']+p['NoRide']))


@pytest.mark.parametrize('bad', [0, -1, float('nan'), float('inf')])
def test_bad_inputs_are_errors_not_replaced_parameters(bad):
    with pytest.raises(ValueError):
        model(beta=bad)
    with pytest.raises(ValueError):
        model().probabilities(dict(PROFILE, PriceThreshold=bad), SCENARIO, 20., 20.)


def test_unknown_anchor_and_mismatched_record_fail():
    s = dict(SCENARIO); s.pop('ReferenceFare')
    with pytest.raises(KeyError):
        model().probabilities(PROFILE, s, 20., 20.)
    with pytest.raises(ValueError, match='reference fare'):
        fit_local_own_elasticity(model(), [(PROFILE, SCENARIO, 20.)], -1.15)


def test_memory_waits_and_shock_channels_remain_active():
    m = model()
    base = m.probabilities(PROFILE, SCENARIO, 23.22, 23.22)
    loyal = m.probabilities(dict(PROFILE, _LoyaltyV2={'Firm1': .7, 'Firm2': 0.}), SCENARIO, 23.22, 23.22)
    waiting = m.probabilities(PROFILE, dict(SCENARIO, WaitEstimateFirm1=15), 23.22, 23.22)
    storm = m.probabilities(PROFILE, dict(SCENARIO, ShockReservationMultiplier=1.25,
                            ShockTransitAvailability=.5), 23.22, 23.22)
    assert loyal['Firm1'] > base['Firm1'] > waiting['Firm1']
    assert storm['NoRide'] < base['NoRide']
