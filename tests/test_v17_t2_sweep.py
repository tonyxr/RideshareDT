import copy
import json

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


def _small_product(tmp_path):
    specs = (t2.cell_spec(0.8, 0.8), t2.cell_spec(1.0, 1.0))
    rival = "profit_best_response_myopic"
    openings = tuple(f"{rival}__opening_{index}" for index in (1, 2))
    command = [
        "python", "Core.py",
        "--market", "New York City",
        "--policy_sweep", ";".join(specs),
        "--policy_sweep_rivals", rival,
        "--policy_sweep_openings", "2",
        "--policy_sweep_horizon", "2",
        "--eval_customers", "10",
        "--policy_sweep_seed", str(t2.VALIDATION_SEED),
        "--action_contract", "two_lever_v3",
        "--fleet_scale", "1.2",
    ]
    results = {
        spec: {
            opening: {
                **_result(3.0, 2.0),
                "fulfillment": 0.95,
                "steps": 2,
            }
            for opening in openings
        }
        for spec in specs
    }
    traces = {
        spec: [
            {
                "rollout": opening,
                "regime": "calm",
                "profit_per_request": 3.0,
                "rival_profit_per_request": 2.0,
                "own_share": 0.5,
                "rival_share": 0.5,
            }
            for opening in openings
            for _ in range(2)
        ]
        for spec in specs
    }
    payload = {
        "protocol": "v26r_policy_sweep_common_random_numbers",
        "city": "New York City",
        "horizon": 2,
        "customers_per_step": 10,
        "steps_per_day": t2.STEPS_PER_DAY,
        "policy_seed": t2.VALIDATION_SEED,
        "rollouts": [[opening, rival] for opening in openings],
        "v26r": {
            "action_contract": "two_lever_v3",
            "supply_model": "v3",
            "persistent_driver_stock": False,
            "effective_fleet_scale": 1.2,
        },
        "results": results,
        "traces": traces,
    }
    path = tmp_path / "product.json"
    path.write_text(json.dumps(payload))
    return path, command, payload


def test_t2_plan_is_3240_surge_range_rollouts_with_five_openings():
    config = load_config()
    assert len(t2.fixed_specs(config)) == 36
    assert len(t2.CASES) == 2
    assert len(t2.RIVALS) == 9
    assert t2.OPENINGS == 5
    assert t2.expected_rollouts(config) == 3240
    assert t2.estimated_wall_hours(config, openings=5, workers=6) == pytest.approx(9.6)


def test_t2_command_wires_production_market_and_live_driver_contract(tmp_path):
    config = load_config()
    command = t2.sweep_command(
        config,
        case="A1_shortage_severe",
        rival="adaptive_best_response_memory",
        output=tmp_path,
        openings=5,
        customers=100,
        pool=2000,
    )
    assert "--current_active_driver_observation" in command
    assert command[command.index("--supply_model") + 1] == "v3"
    assert command[command.index("--shortage_willingness_share") + 1] == "0.5"
    assert "--persistent_driver_stock" not in command
    assert command[command.index("--action_contract") + 1] == "two_lever_v3"
    assert command[command.index("--policy_sweep_openings") + 1] == "5"
    assert command[command.index("--policy_sweep_horizon") + 1] == "240"
    assert command[command.index("--shock_scenario") + 1].endswith(
        "scenarios/v17_t2_shortage_severe.json"
    )


def test_results_comparator_plan_is_registered_648_rollouts(tmp_path):
    config = load_config()
    rivals = (
        "adaptive_best_response_memory",
        "profit_best_response_myopic",
        "profit_opt_mpc_grid",
    )
    assert t2.expected_rollouts(config, openings=3, rivals=rivals) == 648
    command = t2.sweep_command(
        config,
        case="C0_calm",
        rival=rivals[0],
        output=tmp_path,
        openings=3,
        customers=50,
        pool=2000,
        city="boston",
        days=28,
    )
    assert command[command.index("--market") + 1] == "Boston"
    assert command[command.index("--eval_customers") + 1] == "50"
    assert command[command.index("--policy_sweep_horizon") + 1] == "168"
    assert command[command.index("--fleet_scale") + 1] == "1.0"


def test_selected_comparators_pool_cells_and_build_declared_day_rule(
    tmp_path, monkeypatch
):
    config = load_config()
    rival = "adaptive_best_response_memory"
    specs = t2.fixed_specs(config)

    def payload_for(case):
        results = {}
        for spec in specs:
            distance, time = t2.parse_cell(spec)
            own = 10.0 - abs(distance - 1.5) - abs(time - 1.5)
            competitor = 4.0
            if case == "A1_shortage_severe" and distance == time == 2.0:
                own = 20.0
            results[spec] = {
                f"{rival}__opening_1": _result(own, competitor)
            }
        return {"results": results}

    def fake_validate(path, command):
        case = path.parent.name
        return payload_for(case)

    monkeypatch.setattr(t2, "validate_t2_product", fake_validate)
    commands = {(case, rival): [case] for case in t2.CASES}
    selected = t2.select_comparators(
        config,
        tmp_path,
        city="chicago",
        openings=1,
        days=28,
        rivals=(rival,),
        commands=commands,
    )
    assert selected["city"] == "chicago"
    assert selected["surge_rule"]["event_level"] == 2.0
    shortage_rule = selected["surge_rule"]["per_case"][
        "A1_shortage_severe"
    ]
    assert shortage_rule["event_days"] == [10, 11, 12, 13, 14, 15]
    assert shortage_rule["spec"].startswith("schedule:")
    assert selected["surge_rule"]["per_case"]["C0_calm"][
        "event_days"
    ] == []


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
    assert t2.expected_rollouts(config) == 3240


def test_t2_analysis_uses_final_objective_and_reports_components():
    low = t2.cell_spec(0.8, 0.8)
    high = t2.cell_spec(1.0, 1.0)
    openings = [f"r__opening_{index}" for index in range(1, 4)]
    raw = {
        "results": {
            low: {opening: _result(4.0, 3.0) for opening in openings},
            high: {opening: _result(3.5, 1.0) for opening in openings},
        },
        "traces": {
            low: [
                {
                    "rollout": opening,
                    "regime": "calm",
                    "profit_per_request": 4.0,
                    "rival_profit_per_request": 3.0,
                    "own_share": 0.5,
                    "rival_share": 0.5,
                }
                for opening in openings
            ],
            high: [
                {
                    "rollout": opening,
                    "regime": "calm",
                    "profit_per_request": 3.5,
                    "rival_profit_per_request": 1.0,
                    "own_share": 0.55,
                    "rival_share": 0.45,
                }
                for opening in openings
            ],
        },
    }
    report = t2.analyze_job(raw)
    assert report["best_fixed"]["final"]["spec"] == high
    assert report["best_fixed"]["own_profit"]["spec"] == low
    assert report["best_fixed"]["final"]["final_objective"] == pytest.approx(2.75)
    assert report["best_fixed"]["final"]["own_profit"] == 3.5
    assert report["best_fixed"]["final"]["competitor_profit"] == 1.0
    assert report["best_fixed"]["final"]["final_objective_se"] == 0.0
    assert report["adaptation_value"]["final"]["selection_method"] == (
        "leave_one_opening_out"
    )
    assert report["cost_of_competing_for_lead"]["own_profit"] == pytest.approx(-0.5)


def test_t2_product_validator_accepts_complete_and_rejects_truncation(tmp_path):
    path, command, payload = _small_product(tmp_path)
    t2.validate_t2_product(path, command)
    spec = next(iter(payload["results"]))
    opening = next(iter(payload["results"][spec]))
    payload["results"][spec][opening]["steps"] = 1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="truncated"):
        t2.validate_t2_product(path, command)


def test_t2_product_validator_rejects_missing_trace_step(tmp_path):
    path, command, payload = _small_product(tmp_path)
    spec = next(iter(payload["traces"]))
    payload["traces"][spec].pop()
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="trace coverage"):
        t2.validate_t2_product(path, command)


def test_regime_table_for_each_opening_is_fit_on_the_other_openings():
    low = t2.cell_spec(0.8, 0.8)
    high = t2.cell_spec(2.5, 2.5)
    openings = [f"r__opening_{index}" for index in range(1, 4)]
    own_by_spec = {
        low: [10.0, 0.0, 0.0],
        high: [0.0, 10.0, 10.0],
    }
    raw = {
        "results": {
            spec: {
                opening: _result(values[index], 0.0)
                for index, opening in enumerate(openings)
            }
            for spec, values in own_by_spec.items()
        },
        "traces": {
            spec: [
                {
                    "rollout": opening,
                    "regime": "calm",
                    "profit_per_request": values[index],
                    "rival_profit_per_request": 0.0,
                    "own_share": 0.5,
                    "rival_share": 0.5,
                }
                for index, opening in enumerate(openings)
            ]
            for spec, values in own_by_spec.items()
        },
    }
    report = t2.analyze_job(raw)
    loo = report["regime_table"]["final"]["calm"][
        "leave_one_opening_out"
    ]
    assert [row["held_out_opening"] for row in loo] == openings
    assert [row["spec"] for row in loo] == [high, low, low]
    assert report["adaptation_value"]["final"]["switch"]["n_openings"] == 3
