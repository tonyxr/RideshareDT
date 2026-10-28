import pytest

from driver_supply import DriverSupplyConfig, DriverSupplyLayer


def _layer(omega: float, seed: int = 41) -> DriverSupplyLayer:
    return DriverSupplyLayer(
        seed=seed,
        config=DriverSupplyConfig(
            supply_model="v3",
            persistent_stock=False,
            base_active_drivers=10000,
            reference_earnings_per_hour=20.0,
            eta_v3=0.6,
            shortage_willingness_share=omega,
            hour_profile={hour: 1.0 for hour in range(24)},
        ),
    )


def _card_capacity_multiplier(layer: DriverSupplyLayer) -> float:
    return layer.shock_fleet_multiplier * layer.supply_v3_online_response("Firm1")


def test_omega_half_preserves_card_shortage_severity_within_two_percent():
    hard = _layer(0.0)
    split = _layer(0.5)
    for layer in (hard, split):
        layer.apply_shock(fleet_multiplier=0.4, wage_multiplier=1.0)

    # Registered decomposition: sqrt(m) is hard and m^(-.5/eta) is
    # willingness.  At the card, their combined supply multiplier remains m.
    assert hard.shock_fleet_multiplier == pytest.approx(0.4)
    assert hard.shock_wage_multiplier == pytest.approx(1.0)
    assert split.shock_fleet_multiplier == pytest.approx(0.4**0.5)
    assert split.shortage_willingness_multiplier == pytest.approx(0.4 ** (-0.5 / 0.6))
    assert _card_capacity_multiplier(hard) == pytest.approx(0.4, abs=1e-12)
    assert _card_capacity_multiplier(split) == pytest.approx(0.4, abs=1e-12)

    hard.begin_batch(500, 12, "clear")
    split.begin_batch(500, 12, "clear")
    hard_online = hard._batch_states["Firm1"].active_drivers
    split_online = split._batch_states["Firm1"].active_drivers
    assert abs(split_online / hard_online - 1.0) <= 0.02


def test_default_zero_share_leaves_v3_shock_unsplit():
    layer = _layer(0.0)
    layer.apply_shock(fleet_multiplier=0.55, wage_multiplier=1.2)
    assert layer.raw_shock_fleet_multiplier == pytest.approx(0.55)
    assert layer.shock_fleet_multiplier == pytest.approx(0.55)
    assert layer.raw_shock_wage_multiplier == pytest.approx(1.2)
    assert layer.shock_wage_multiplier == pytest.approx(1.2)
    assert layer.shortage_willingness_multiplier == pytest.approx(1.0)


def test_non_shortage_and_complete_outage_are_not_reinterpreted():
    layer = _layer(0.5)
    layer.apply_shock(fleet_multiplier=1.1, wage_multiplier=1.0)
    assert layer.shock_fleet_multiplier == pytest.approx(1.1)
    assert layer.shock_wage_multiplier == pytest.approx(1.0)
    layer.apply_shock(fleet_multiplier=0.0, wage_multiplier=1.0)
    assert layer.shock_fleet_multiplier == 0.0
    assert layer.shock_wage_multiplier == 1.0


@pytest.mark.parametrize("omega", [-0.01, 1.01, float("nan")])
def test_shortage_willingness_share_is_bounded(omega):
    with pytest.raises(ValueError, match="shortage_willingness_share"):
        DriverSupplyConfig(shortage_willingness_share=omega)
