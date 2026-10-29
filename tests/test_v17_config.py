import copy
from pathlib import Path

import pytest

from scripts import v17_config


def _config():
    return v17_config.load_config()


def test_prefreeze_config_is_valid_but_cannot_launch_production():
    config = _config()
    v17_config.validate_config(config)
    with pytest.raises(ValueError, match="T3 must freeze"):
        v17_config.validate_config(config, require_frozen=True)


def test_command_wires_manual_order_and_user_reward_without_fixed_reward(tmp_path):
    command = v17_config.core_args(
        _config(),
        city="nyc",
        seed=101,
        output=tmp_path,
        days=200,
        branch_horizon_days=7,
    )
    assert command[command.index("--training_curriculum") + 1] == "v17_progressive"
    assert "--v16_reward" not in command
    assert "--current_active_driver_observation" in command
    assert command[command.index("--supply_model") + 1] == "v3"
    assert command[command.index("--fleet_scale") + 1] == "1.2"
    assert "--persistent_driver_stock" not in command
    assert command[command.index("--action_contract") + 1] == "two_lever_v1"
    assert command[command.index("--v16_policy_mode") + 1] == "q"
    assert command[command.index("--v16_branch_design") + 1] == "stencil11"
    assert command[command.index("--v16_store_cap") + 1] == "20000"
    assert command[command.index("--ppo_update_interval_days") + 1] == "7"


def test_all_city_demand_inputs_are_relative_refits():
    config = _config()
    v17_config.validate_config(config)
    for slug in v17_config.CITY_SLUGS:
        path = v17_config.resolve_input(
            config["city_inputs"][slug]["demand_calibration"]
        )
        assert path.is_file()


@pytest.mark.parametrize("bad_horizon", [4, 6, 8])
def test_only_t3_horizon_candidates_can_render_command(tmp_path, bad_horizon):
    with pytest.raises(ValueError, match="T3-selected"):
        v17_config.core_args(
            _config(),
            city="boston",
            seed=202,
            output=Path(tmp_path),
            days=200,
            branch_horizon_days=bad_horizon,
        )


def test_config_hash_changes_on_contract_drift():
    first = _config()
    second = copy.deepcopy(first)
    second["reward"]["stage_profit_lead_weights"][3] = 0.8
    assert v17_config.canonical_hash(first) != v17_config.canonical_hash(second)


def test_config_rejects_invalid_city_fleet_scale():
    config = _config()
    config["city_inputs"]["nyc"]["fleet_scale"] = 0.0
    with pytest.raises(ValueError, match="fleet_scale"):
        v17_config.validate_config(config)


def test_t0_14_config_switch_accepts_only_the_registered_six_level_grid(tmp_path):
    config = _config()
    config["action"] = {
        "contract": "two_lever_v2",
        "distance_levels": [0.7, 0.8, 0.9, 1.0, 1.2, 1.5],
        "time_levels": [0.7, 0.8, 0.9, 1.0, 1.2, 1.5],
        "base_and_booking_fixed": True,
    }
    v17_config.validate_config(config)
    command = v17_config.core_args(
        config,
        city="nyc",
        seed=101,
        output=tmp_path,
        days=200,
        branch_horizon_days=7,
    )
    assert command[command.index("--action_contract") + 1] == "two_lever_v2"

    config["action"]["distance_levels"][0] = 0.6
    with pytest.raises(ValueError, match="action grid"):
        v17_config.validate_config(config)
