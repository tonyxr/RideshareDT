import copy

import numpy as np
import pytest

from scripts import v17_config, v17_market_checks as checks


def _trace(rollout, values, column):
    return [
        {"rollout": rollout, "step": day * checks.STEPS_PER_DAY, column: value}
        for day, value in enumerate(values)
    ]


def test_t1_command_uses_complete_v17_market_without_invalid_v3_stock(tmp_path):
    command = checks.market_command(
        v17_config.load_config(),
        city="nyc",
        job="shortage_static",
        output=tmp_path,
        openings=6,
        customers=50,
        pool=2000,
    )
    assert command[command.index("--supply_model") + 1] == "v3"
    assert command[command.index("--fleet_scale") + 1] == "1.2"
    assert "--persistent_driver_stock" not in command
    assert command[command.index("--shortage_willingness_share") + 1] == "0.5"
    assert command[command.index("--price_sensitivity_units") + 1] == "relative"
    assert command[command.index("--fare_model") + 1] == "realistic"
    assert command[command.index("--action_contract") + 1] == "two_lever_v1"
    assert "--rivals_two_lever" in command
    assert "--current_active_driver_observation" in command


def test_paired_effect_uses_opening_level_differences():
    treated = []
    control = []
    for rollout, lift in (("r__opening_1", 2.0), ("r__opening_2", 4.0)):
        treated += _trace(rollout, [10.0 + lift] * 3, "drivers")
        control += _trace(rollout, [10.0] * 3, "drivers")
    result = checks.paired_effect(
        {"traces": {"treated": treated, "control": control}},
        "treated",
        "control",
        "drivers",
        start_day=0,
        end_day=3,
    )
    assert result["n"] == 2
    assert result["mean"] == pytest.approx(3.0)
    assert result["se"] == pytest.approx(1.0)
    assert result["t"] == pytest.approx(3.0)


def test_market_command_rejects_config_drift(tmp_path):
    config = copy.deepcopy(v17_config.load_config())
    config["observation"]["opponent_active_drivers"] = True
    with pytest.raises(ValueError, match="own-only"):
        checks.market_command(
            config,
            city="boston",
            job="calm_static",
            output=tmp_path,
            openings=1,
            customers=10,
            pool=100,
        )


def test_revb_chicago_t11_command_uses_chicago_inputs_and_card_only_can_be_set(
    tmp_path,
):
    command = checks.market_command(
        v17_config.load_config(),
        city="chicago",
        job="calm_static",
        output=tmp_path,
        openings=6,
        customers=500,
        pool=2000,
    )
    assert command[command.index("--market") + 1] == "Chicago"
    card_only = checks._replace_command_value(command, "--policy_sweep", checks.CARD)
    assert card_only[card_only.index("--policy_sweep") + 1] == checks.CARD
    assert card_only[card_only.index("--fleet_scale") + 1] == "1.1"


def test_two_lever_direct_post_comparator_returns_no_action_sentinel():
    from types import SimpleNamespace

    from Core import Core

    core = Core.__new__(Core)
    core.firm1 = SimpleNamespace(two_lever_actions=True)
    action = Core._baseline_firm1_action(
        core,
        "opening_hold",
        step_index=0,
        action_mask=np.ones(25, dtype=bool),
        day_of_week=0,
        hour=0,
        weather="clear",
        presampled=None,
        customers_per_step=1,
    )
    assert action == -1


def test_two_lever_direct_post_keeps_fixed_card_charges():
    from types import SimpleNamespace

    from Core import Core
    from Market_models import CoefficientOverrides, MarketCoefficients
    from optim_config import default_specs_for

    core = Core.__new__(Core)
    base = MarketCoefficients(
        base_fare=2.0,
        per_mile=3.0,
        per_minute=0.5,
        booking_fee=1.0,
        airport_fee=4.0,
        surge_hours=[],
        day_multiplier={},
        weather_multiplier={},
        service_multiplier={},
    )
    keys = ["base_fare", "per_minute", "per_mile", "booking_fee", "airport_fee"]
    core.market = SimpleNamespace(curr_market=base)
    core.firm1 = SimpleNamespace(
        two_lever_actions=True,
        opt_keys=keys,
        config=default_specs_for(keys),
        overrides=CoefficientOverrides(),
        supply_incentive_multiplier=1.0,
    )
    Core._post_firm1_target(core, 1.25)
    assert core.firm1.overrides.per_mile == pytest.approx(3.75)
    assert core.firm1.overrides.per_minute == pytest.approx(0.625)
    assert core.firm1.overrides.base_fare is None
    assert core.firm1.overrides.booking_fee is None
    assert core.firm1.overrides.airport_fee is None


def test_reality_gap_uses_exact_requested_rows():
    result = checks.quote_reality_gap(v17_config.load_config(), "chicago", rows=10)
    assert result["rows"] == 10
    assert result["v17_card_mean"] > 0.0
    assert result["v15d_card_mean"] > 0.0
    assert result["v17_mae"] >= 0.0
