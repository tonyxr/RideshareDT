import copy
from types import SimpleNamespace
import numpy as np
import pytest
from simulation_noise import isolated_batch_noise, rider_streams
from choice_models import CognitiveChoiceModel


def test_policy_conditional_draws_do_not_shift_later_rider_or_batch_tapes():
    def core():
        return SimpleNamespace(rng=np.random.default_rng(1),
            choice_model=SimpleNamespace(rng=np.random.default_rng(2)),
            driver_supply=SimpleNamespace(supply_v2=True, rng=np.random.default_rng(3)))
    @isolated_batch_noise
    def batch(c, extra):
        seen = []
        for index in range(10):
            rider_streams(c, index)
            seen.append([c.rng.random(), c.choice_model.rng.random(), c.driver_supply.rng.random()])
            if index % 2:
                for owner in (c, c.choice_model, c.driver_supply):
                    owner.rng.random(extra)
        return seen
    a, b = core(), core()
    for _ in range(4):
        assert batch(a, 0) == batch(b, 100)


def test_noise_stream_objects_restore_even_on_exception():
    core = SimpleNamespace(rng=np.random.default_rng(1),
        choice_model=SimpleNamespace(rng=np.random.default_rng(2)),
        driver_supply=SimpleNamespace(supply_v2=True, rng=np.random.default_rng(3)))
    original = (core.rng, core.choice_model.rng, core.driver_supply.rng)
    @isolated_batch_noise
    def fail(c):
        rider_streams(c, 9)
        raise RuntimeError('test')
    with pytest.raises(RuntimeError):
        fail(core)
    assert (core.rng, core.choice_model.rng, core.driver_supply.rng) == original


def test_exact_choice_probability_api_does_not_change_rng_or_sampler():
    from test_v26r_loyalty_v2 import PROFILE, SCENARIO
    model = CognitiveChoiceModel(seed=891)
    before = copy.deepcopy(model.rng.bit_generator.state)
    expected = model.probabilities(PROFILE, SCENARIO, 20., 21.)
    assert sum(expected.values()) == pytest.approx(1.)
    assert model.rng.bit_generator.state == before
    counts = {key: 0 for key in expected}
    for _ in range(10000):
        counts[model.choose(PROFILE, SCENARIO, 20., 21.).choice] += 1
    for key in expected:
        assert counts[key]/10000 == pytest.approx(expected[key], abs=.02)


def test_own_and_market_price_elasticity_are_not_conflated():
    from test_v26r_loyalty_v2 import PROFILE, SCENARIO
    model = CognitiveChoiceModel(seed=891)
    low = model.probabilities(PROFILE, SCENARIO, 19., 20.)
    high = model.probabilities(PROFILE, SCENARIO, 21., 20.)
    assert low['Firm1'] > high['Firm1']
    both = model.probabilities(PROFILE, SCENARIO, 21., 21.)
    assert both['NoRide'] > high['NoRide']


def test_normalized_price_response_is_invariant_to_city_currency_units():
    from test_v26r_loyalty_v2 import PROFILE, SCENARIO
    results = []
    for scale in (.5, 1., 2.):
        model = CognitiveChoiceModel()
        model.make_symmetric()
        model.fare_normalized = True
        model.currency_unit = scale
        model.price_level_index = scale
        model.outside_option_base_fare = 2.9*scale
        profile = dict(PROFILE, PriceThreshold=1.5*scale,
                       ReservationPrice=40*scale, OutsideOptionCost=5*scale)
        results.append(model.probabilities(profile, SCENARIO, 20*scale, 21*scale))
    for row in results:
        assert row == pytest.approx(results[0])


def test_symmetric_premium_has_no_firm1_comfort_bonus():
    from test_v26r_loyalty_v2 import PROFILE, SCENARIO
    model = CognitiveChoiceModel()
    model.make_symmetric()
    model.fare_normalized = True
    result = model.probabilities(dict(PROFILE, LoyaltyFirm=None, LoyaltyStrength=0),
        dict(SCENARIO, Service='premium'), 20., 20.)
    assert result['Firm1'] == pytest.approx(result['Firm2'])
