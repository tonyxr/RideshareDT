"""Corrected-model gates from the November 10 V17 rerun protocol."""

from types import SimpleNamespace

import pytest

from Core import Core
from MarketInteraction import MarketInteraction
from driver_supply import DriverSupplyConfig, DriverSupplyLayer
from market_dynamics import RELATIVE_BOUNDS, market_relative_bounds, rescale_pricer_to_market
from pricing_models import FirmRLPricer
from scripts import (
    v17_market_checks,
    v17_model_effect,
    v17_probe,
    v17_results_eval,
    v17_t2_sweep,
)
from scripts.v17_config import check_model, core_args, load_config


KEYS = ["base_fare", "per_minute", "per_mile", "booking_fee", "airport_fee"]


def _shortage_layer(mode: str, *, omega: float = 0.5, eta: float = 0.6):
    layer = DriverSupplyLayer(
        seed=17,
        config=DriverSupplyConfig(
            supply_model="v3",
            eta_v3=eta,
            shortage_willingness_share=omega,
            acceptance_shock_wage=mode,
            reference_earnings_per_hour=24.0,
        ),
    )
    layer.apply_shock(fleet_multiplier=0.35, wage_multiplier=1.15)
    return layer


def _acceptance(layer, *, miles=5.0, minutes=15.0, pickup=4.0, pay=20.0):
    return layer.driver_acceptance_ratio(
        driver_pay=pay,
        distance_miles=miles,
        duration_minutes=minutes,
        pickup_minutes=pickup,
        weather="clear",
    )


def _manual_ratio(layer, factor, *, miles=5.0, minutes=15.0, pickup=4.0, pay=20.0):
    cost = layer.config.operating_cost_per_mile * (miles + max(0.1, pickup * 0.28))
    net = max(0.0, pay - cost)
    required = layer.config.reservation_wage_per_hour * factor * ((minutes + pickup) / 60.0)
    return net / required


def test_acceptance_default_is_legacy():
    layer = _shortage_layer("legacy")
    assert DriverSupplyConfig().acceptance_shock_wage == "legacy"
    assert layer.shock_wage_multiplier == pytest.approx(2.76, rel=5e-3)
    assert _acceptance(layer) == pytest.approx(
        _manual_ratio(layer, layer.shock_wage_multiplier), abs=1e-12
    )


def test_acceptance_declared():
    declared = _shortage_layer("declared")
    assert _acceptance(declared) == pytest.approx(
        _manual_ratio(declared, 1.15), abs=1e-12
    )
    for omega, eta in ((0.25, 0.3), (0.75, 1.0)):
        other = _shortage_layer("declared", omega=omega, eta=eta)
        assert _acceptance(other) == pytest.approx(_acceptance(declared), abs=1e-12)
        legacy = _shortage_layer("legacy", omega=omega, eta=eta)
        assert other.supply_v3_online_response("Firm1") == pytest.approx(
            legacy.supply_v3_online_response("Firm1"), abs=1e-12
        )


def test_acceptance_no_shock_identity():
    legacy = _shortage_layer("legacy")
    declared = _shortage_layer("declared")
    for layer in (legacy, declared):
        layer.apply_shock(fleet_multiplier=1.0, wage_multiplier=1.0)
    for miles in (0.5, 3.0, 12.0):
        for minutes in (4.0, 18.0, 45.0):
            assert _acceptance(legacy, miles=miles, minutes=minutes) == pytest.approx(
                _acceptance(declared, miles=miles, minutes=minutes), abs=1e-12
            )


def test_rate_cap_default():
    anchors = {key: value for key, value in zip(KEYS, (3.0, 0.4, 1.9, 2.75, 6.0))}
    actual = market_relative_bounds(anchors)
    for key in KEYS:
        lo, hi = RELATIVE_BOUNDS[key]
        floor = 0.01 if key in {"base_fare", "per_minute", "per_mile"} else 0.0
        assert actual[key] == pytest.approx(
            (max(floor, lo * anchors[key]), max(floor + 1e-6, hi * anchors[key]))
        )


def test_rate_cap_lifted():
    market = MarketInteraction(city_name="New York City", seed=18)
    anchors = {key: float(getattr(market.curr_market, key)) for key in KEYS}
    default = market_relative_bounds(anchors)
    lifted = market_relative_bounds(anchors, rate_upper=2.5)
    assert lifted["per_mile"][1] == pytest.approx(2.5 * anchors["per_mile"])
    assert lifted["per_minute"][1] == pytest.approx(2.5 * anchors["per_minute"])
    for key in ("base_fare", "booking_fee", "airport_fee"):
        assert lifted[key] == pytest.approx(default[key])

    pricer = FirmRLPricer(
        seed=18,
        opt_keys=KEYS,
        independent_coefficient_actions=False,
        action_contract="two_lever_v3",
    )
    rescale_pricer_to_market(pricer, anchors, rate_upper=2.5)
    action = next(
        index for index, cell in pricer.action_to_two_lever_cell.items()
        if cell == (2.5, 2.5)
    )
    pricer.apply_action(action, market)
    assert pricer.overrides.per_mile == pytest.approx(2.5 * anchors["per_mile"])
    assert pricer.overrides.per_minute == pytest.approx(2.5 * anchors["per_minute"])
    assert not pricer.last_action_was_saturated


def test_model_flags_in_every_builder(tmp_path):
    config = load_config()
    builders = (
        core_args(config, city="nyc", seed=101, output=tmp_path, days=20, branch_horizon_days=5),
        v17_results_eval._base_command(
            config, city="nyc", case="C0_calm", specs=("static:1",),
            raw=tmp_path / "eval.json", report_prefix=tmp_path / "eval",
        ),
        v17_t2_sweep.sweep_command(
            config, case="C0_calm", rival="adaptive_best_response_memory",
            output=tmp_path, openings=1, customers=5, pool=20,
        ),
        v17_market_checks.market_command(
            config, city="nyc", job="calm_static", output=tmp_path,
            openings=1, customers=5, pool=20,
        ),
        v17_probe.validation_command(
            config, horizon=5, day=20, case="C0_calm", output=tmp_path,
            validation_days=1, validation_openings=1,
        ),
    )
    for command in builders:
        assert command.count("--acceptance_shock_wage") == 1
        assert command[command.index("--acceptance_shock_wage") + 1] == "declared"
        assert command.count("--rate_upper_multiple") == 1
        assert float(command[command.index("--rate_upper_multiple") + 1]) == 2.5


def test_check_model_rejects_legacy_product():
    config = load_config()
    with pytest.raises(ValueError, match="requested model"):
        check_model({}, config)
    with pytest.raises(ValueError, match="requested model"):
        check_model(
            {"model_switches": {"acceptance_shock_wage": "legacy", "rate_upper_multiple": 2.0}},
            config,
        )


def test_flags_reported():
    config = SimpleNamespace(
        acceptance_shock_wage="declared", eta_v3=0.6,
        shortage_willingness_share=0.5,
    )
    fake = SimpleNamespace(
        firm1_mode="RL",
        market_name="New York City",
        driver_supply=SimpleNamespace(config=config),
        market_dynamics=SimpleNamespace(
            rate_upper_multiple=2.5, unfulfilled_spillover=True,
        ),
        loyalty_model="v2",
        threshold_ablation_mode="as_loaded",
        v26r_settings={},
        _sweep_trace=[],
    )
    fake._check_supply_calibration_volume = lambda customers: None
    fake._configured_primary_objective = lambda: "profit"
    fake._model_switches = lambda: Core._model_switches(fake)

    def validation(**kwargs):
        fake._sweep_trace = []
        return {
            "rollout_summaries": {
                name: {
                    "objective_reward_mean": 1.0,
                    "own_profit_per_request": 2.0,
                    "rival_profit_per_request": 1.0,
                    "own_completed_share": 0.5,
                    "rival_completed_share": 0.5,
                    "fulfillment": 0.9,
                    "steps": 1,
                }
                for name, _ in kwargs["rollouts_override"]
            }
        }

    fake._adaptive_policy_validation = validation
    product = Core.run_policy_sweep(
        fake, specs=("static:1",), rivals=("static",), openings=1,
        horizon=1, customers_per_step=1, steps_per_day=1,
        firm1_interval_steps=1, opponent_interval_steps=1, policy_seed=1,
    )
    assert product["model_switches"]["acceptance_shock_wage"] == "declared"
    assert product["model_switches"]["rate_upper_multiple"] == 2.5


def test_model_effect_gate_uses_manual_stop_condition_only():
    checks = {
        "calm_first_three_maximum_difference": 0.25,
        "corrected_published_service": 0.75,
        "legacy_published_service": 0.43,
        "legacy_2_5_vs_2_0_maximum_difference": 0.10,
        "corrected_2_5_vs_2_0_maximum_difference": 0.0,
    }
    gate = v17_model_effect._manual_gate(checks)
    assert gate["pass"] is True
    assert gate["stop_conditions"] == {
        "corrected_published_service_improves": True,
    }
    assert not all(gate["diagnostic_expectations"].values())

    checks["corrected_published_service"] = checks["legacy_published_service"]
    assert v17_model_effect._manual_gate(checks)["pass"] is False


def test_scientific_diagnostics_do_not_stop_part_c():
    market = v17_market_checks._workflow_status({"passed": False})
    assert market == {
        "state": "completed",
        "pass": True,
        "scientific_expectations_pass": False,
    }
    sweep = v17_t2_sweep._workflow_status(
        {"edge_checks": {"requires_grid_review": True}}
    )
    assert sweep == {
        "state": "passed",
        "pass": True,
        "grid_review_recommended": True,
    }
