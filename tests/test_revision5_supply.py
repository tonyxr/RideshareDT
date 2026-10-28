import copy
import numpy as np
import pytest
from driver_supply import DriverSupplyConfig, DriverSupplyLayer, FirmDriverBatchState


def layer(earnings=10., **kwargs):
    supply = DriverSupplyLayer(seed=1, config=DriverSupplyConfig(supply_model='v2',
        persistent_stock=True, reservation_wage_per_hour=20., **kwargs))
    for firm in ('Firm1', 'Firm2'):
        supply.smoothed_states[firm] = FirmDriverBatchState(total_driver_pay=earnings * 100,
                                                         total_online_minutes=6000)
    return supply


def test_reference_cannot_be_relearned_from_cheap_policy_or_reset():
    low, high = layer(10.), layer(30.)
    for _ in range(40):
        for supply, earnings in ((low, 10.), (high, 30.)):
            for firm in ('Firm1', 'Firm2'):
                supply._daily_pay[firm] = earnings * 100
                supply._daily_online[firm] = 6000
        low.advance_day(); high.advance_day()
    assert low.reference_earnings == high.reference_earnings == {'Firm1': 20., 'Firm2': 20.}
    assert low.online_stock['Firm1'] < 1 < high.online_stock['Firm1']
    low.reset_stock()
    assert low.reference_earnings['Firm1'] == 20.
    assert low.smoothed_states['Firm1'].total_online_minutes == 0
    assert low.online_stock['Firm1'] == 1.


def test_explicit_reference_stays_fixed_and_city_rates_are_distinct():
    for reference in (8., 20., 35.):
        supply = layer(10., reference_earnings_per_hour=reference)
        for _ in range(5):
            supply.advance_day()
        supply.reset_stock()
        assert supply.reference_earnings['Firm1'] == reference


def test_outage_not_applied_to_workforce_twice():
    calm, outage = layer(20.), layer(20.)
    outage.apply_shock(firm_capacity={'Firm1': .25})
    for _ in range(3):
        calm.advance_day(); outage.advance_day()
    assert calm.online_stock == outage.online_stock
    for supply in (calm, outage):
        supply.begin_batch(500, 12, 'clear')
    assert outage._batch_states['Firm1'].active_drivers == pytest.approx(
        calm._batch_states['Firm1'].active_drivers * .25, abs=1)
    outage.apply_shock()
    outage.begin_batch(500, 12, 'clear')
    assert abs(calm._batch_states['Firm1'].active_drivers - outage._batch_states['Firm1'].active_drivers) <= 1


def test_complete_outage_is_zero_capacity_not_an_implicit_five_percent_floor():
    supply = layer(20.)
    supply.apply_shock(firm_capacity={'Firm1': 0.})
    supply.begin_batch(500, 12, 'clear')
    assert supply._batch_states['Firm1'].active_drivers == 0


def test_whole_block_availability_not_random_hour_within_block():
    supply = layer(20., availability_block_hours=4)
    for start in range(0, 24, 4):
        expected = np.mean([supply.hour_profile[h] for h in range(start, start + 4)])
        assert all(supply.hour_availability(h) == expected for h in range(start, start + 4))


def test_config_post_init_is_live():
    with pytest.raises(ValueError, match='acceptance_mode'):
        DriverSupplyConfig(acceptance_mode='typo')


def test_driver_can_complete_multiple_rides_but_not_simultaneously():
    supply = layer(20., base_active_drivers=1, availability_block_hours=4,
                   hour_profile={h: 1 for h in range(24)}, pickup_noise_sigma=0.)
    supply.begin_batch(500, 12, 'clear')
    supply.advance_request_time(0.)
    kwargs = dict(firm='Firm1', rider_fare=100., distance_miles=1., duration_minutes=10.)
    assert supply.dispatch(**kwargs).fulfilled
    assert not supply.dispatch(**kwargs).fulfilled
    supply.advance_request_time(.25)
    assert supply.dispatch(**kwargs).fulfilled
    assert supply._batch_states['Firm1'].completed_rides == 2
    assert supply._batch_states['Firm1'].total_online_minutes == 240


def test_trip_crossing_block_remains_busy_then_releases():
    supply = layer(20., base_active_drivers=1, availability_block_hours=4,
                   hour_profile={h: 1 for h in range(24)}, pickup_noise_sigma=0.)
    supply.begin_batch(500, 12, 'clear')
    supply.advance_request_time(.99)
    assert supply.dispatch('Firm1', 100., 1., 30.).fulfilled
    supply.end_batch()
    supply.begin_batch(500, 16, 'clear')
    assert supply._batch_states['Firm1'].idle_drivers == 0
    supply.advance_request_time(.25)
    assert supply._batch_states['Firm1'].idle_drivers >= 1


def test_zero_measured_pay_and_severe_shortage_are_not_replaced_or_floored():
    supply = layer(0.)
    assert supply.expected_block_earnings('Firm1') == 20.  # Posted offer, not realized income.
    supply.apply_shock(fleet_multiplier=.1)
    for _ in range(40):
        supply._daily_online = {'Firm1': 600., 'Firm2': 600.}
        supply._daily_pay = {'Firm1': 0., 'Firm2': 0.}
        supply.advance_day()
    assert supply.online_stock['Firm1'] < .01


@pytest.mark.parametrize('steps', [5, 6, 7, 24])
def test_fractional_hour_blocks_cover_full_day(steps):
    supply = layer(20., availability_block_hours=24/steps)
    total = sum(supply.hour_availability((b+.5)*24/steps) * 24/steps for b in range(steps))
    assert total == pytest.approx(24.)
    edges = np.round(np.linspace(0, 24, steps + 1)).astype(int)
    for block in range(steps):
        expected = supply.hour_availability((block+.5)*24/steps)
        assert all(supply.hour_availability(hour) == expected
                   for hour in range(edges[block], edges[block+1]))


def test_pay_index_and_earnings_share_block_clock():
    supply = layer(20.)
    supply.set_pay_index('Firm1', 2.)
    supply.begin_batch(500, 12, 'clear')
    supply.end_batch()
    value = supply.pay_index_smoothed['Firm1']
    assert value == pytest.approx(1.35)
    supply.advance_day()
    assert supply.pay_index_smoothed['Firm1'] == value
