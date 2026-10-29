import copy

import pytest

from scripts import v17_t2_sweep as t2
from scripts.v17_config import load_config


def _result(own, rival, own_share=0.5, rival_share=0.5):
    return {
        "own_profit": own,
        "rival_profit": rival,
        "own_share": own_share,
        "rival_share": rival_share,
    }


def test_t2_plan_is_exactly_1350_v1_rollouts():
    config = load_config()
    assert len(t2.fixed_specs(config)) == 25
    assert len(t2.CASES) == 2
    assert len(t2.RIVALS) == 9
    assert t2.expected_rollouts(config) == 1350


def test_t2_command_wires_production_market_and_live_driver_contract(tmp_path):
    config = load_config()
    command = t2.sweep_command(
        config,
        case="A1_shortage_severe",
        rival="adaptive_best_response_memory",
        output=tmp_path,
        openings=3,
        customers=100,
        pool=2000,
    )
    assert "--current_active_driver_observation" in command
    assert command[command.index("--supply_model") + 1] == "v3"
    assert command[command.index("--shortage_willingness_share") + 1] == "0.5"
    assert "--persistent_driver_stock" not in command
    assert command[command.index("--action_contract") + 1] == "two_lever_v1"
    assert command[command.index("--policy_sweep_openings") + 1] == "3"
    assert command[command.index("--policy_sweep_horizon") + 1] == "240"
    assert command[command.index("--shock_scenario") + 1].endswith(
        "scenarios/v17_t2_shortage_severe.json"
    )


def test_t2_v2_switch_expands_the_same_plan_to_36_cells():
    config = load_config()
    config = copy.deepcopy(config)
    config["action"] = {
        "contract": "two_lever_v2",
        "distance_levels": [0.7, 0.8, 0.9, 1.0, 1.2, 1.5],
        "time_levels": [0.7, 0.8, 0.9, 1.0, 1.2, 1.5],
        "base_and_booking_fixed": True,
    }
    assert len(t2.fixed_specs(config)) == 36
    assert t2.expected_rollouts(config) == 1944


def test_t2_analysis_uses_final_objective_and_reports_components():
    low = t2.cell_spec(0.8, 0.8)
    high = t2.cell_spec(1.0, 1.0)
    raw = {
        "results": {
            low: {
                "r__opening_1": _result(4.0, 3.0),
                "r__opening_2": _result(4.0, 3.0),
                "r__opening_3": _result(4.0, 3.0),
            },
            high: {
                "r__opening_1": _result(3.5, 1.0),
                "r__opening_2": _result(3.5, 1.0),
                "r__opening_3": _result(3.5, 1.0),
            },
        },
        "traces": {
            low: [
                {
                    "regime": "calm",
                    "profit_per_request": 4.0,
                    "rival_profit_per_request": 3.0,
                    "own_share": 0.5,
                    "rival_share": 0.5,
                }
            ],
            high: [
                {
                    "regime": "calm",
                    "profit_per_request": 3.5,
                    "rival_profit_per_request": 1.0,
                    "own_share": 0.55,
                    "rival_share": 0.45,
                }
            ],
        },
    }
    report = t2.analyze_job(raw)
    assert report["best_fixed"]["final"]["spec"] == high
    assert report["best_fixed"]["own_profit"]["spec"] == low
    assert report["best_fixed"]["final"]["final_objective"] == pytest.approx(2.75)
    assert report["best_fixed"]["final"]["own_profit"] == 3.5
    assert report["best_fixed"]["final"]["competitor_profit"] == 1.0
