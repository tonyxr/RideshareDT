import json

import pytest

from scripts import v17_results_eval as evaluation
from scripts.v17_config import load_config
from scripts.v17_results_train import RUN_BY_NAME


def _selection():
    return {
        "best_fixed_cell": {"spec": evaluation.cell_spec(1.0, 1.5)},
        "own_profit_best_fixed_cell": {
            "spec": evaluation.cell_spec(1.5, 2.0)
        },
        "surge_rule": {"event_level": 2.0},
    }


def _small_product(tmp_path):
    config = load_config()
    raw = tmp_path / "product.json"
    command = evaluation.evaluation_command(
        config,
        city="boston",
        case="C0_calm",
        specs=(evaluation.cell_spec(1.0, 1.0),),
        raw=raw,
        report_prefix=tmp_path / "report",
    )
    command[command.index("--policy_sweep_openings") + 1] = "1"
    command[command.index("--policy_sweep_rivals") + 1] = "pi_price_gap"
    command[command.index("--policy_sweep_horizon") + 1] = "2"
    spec = evaluation.cell_spec(1.0, 1.0)
    rollout = "pi_price_gap__opening_1"
    payload = {
        "protocol": "v26r_policy_sweep_common_random_numbers",
        "city": "Boston",
        "horizon": 2,
        "customers_per_step": evaluation.CUSTOMERS,
        "steps_per_day": evaluation.STEPS_PER_DAY,
        "policy_seed": evaluation.EVAL_SEED,
        "rollouts": [[rollout, "pi_price_gap"]],
        "v26r": {
            "action_contract": "two_lever_v3",
            "supply_model": "v3",
            "persistent_driver_stock": False,
            "effective_fleet_scale": 1.0,
        },
        "results": {
            spec: {
                rollout: {
                    "own_profit": 2.0,
                    "rival_profit": 1.0,
                    "own_share": 0.5,
                    "rival_share": 0.5,
                    "fulfillment": 0.9,
                    "steps": 2,
                }
            }
        },
        "traces": {
            spec: [
                {
                    "rollout": rollout,
                    "regime": "calm",
                    "profit_per_request": 2.0,
                    "rival_profit_per_request": 1.0,
                    "own_share": 0.5,
                    "rival_share": 0.5,
                }
                for _ in range(2)
            ]
        },
    }
    raw.write_text(json.dumps(payload), encoding="utf-8")
    return raw, command, payload


def test_registered_section7_rollout_count_is_3030():
    assert evaluation.expected_rollouts() == {
        "main_rl": 1050,
        "ablation_rl": 180,
        "comparators": 1800,
        "total": 3030,
    }
    assert len(evaluation.RIVALS) == 10
    assert len(evaluation.CASES) == 5
    assert evaluation.OPENINGS == 3
    assert evaluation.HORIZON_DAYS == 40
    assert evaluation.CUSTOMERS == 50


def test_all_registered_case_inputs_exist():
    assert all(path.is_file() for path in evaluation.CASES.values())
    assert evaluation.CASES["A3_storm_transit_late"].as_posix().endswith(
        "scenarios/cases/storm_transit_late.json"
    )


def test_comparator_specs_extend_selected_surge_to_held_out_cases():
    specs = evaluation.comparator_specs(
        _selection(), "A3_storm_transit_late"
    )
    assert tuple(specs) == evaluation.COMPARATOR_LABELS
    schedule = json.loads(specs["surge_rule"].split(":", 1)[1])
    assert schedule["default"] == [1.0, 1.0]
    assert schedule["days"]
    assert all(value == [2.0, 1.0] for value in schedule["days"].values())


def test_rl_commands_preserve_q_and_plain_ppo_variants(tmp_path):
    config = load_config()
    main = evaluation.evaluation_command(
        config, city="nyc", case="C0_calm", specs=("rl_actor",),
        raw=tmp_path / "main.json", report_prefix=tmp_path / "main",
        run=RUN_BY_NAME["nyc_s101"], policy=tmp_path / "policy.pt",
    )
    ppo = evaluation.evaluation_command(
        config, city="nyc", case="C0_calm", specs=("rl_actor",),
        raw=tmp_path / "ppo.json", report_prefix=tmp_path / "ppo",
        run=RUN_BY_NAME["nyc_s101_plain_ppo"], policy=tmp_path / "ppo.pt",
    )
    assert main[main.index("--v16_policy_mode") + 1] == "q"
    assert "--v17_move_radius" in main
    assert ppo[ppo.index("--v16_policy_mode") + 1] == "ppo"
    assert "--v16_value_head" not in ppo
    assert "--v17_move_radius" not in ppo
    assert main[main.index("--policy_sweep_seed") + 1] == "77311"


def test_no_driver_observation_command_matches_saved_contract(tmp_path):
    config = load_config()
    command = evaluation.evaluation_command(
        config, city="nyc", case="C0_calm", specs=("rl_actor",),
        raw=tmp_path / "no-driver.json",
        report_prefix=tmp_path / "no-driver",
        run=RUN_BY_NAME["nyc_s101_no_driver_obs"],
        policy=tmp_path / "policy.pt",
    )
    assert "--v17_ablate_driver_observation" in command
    assert "--current_active_driver_observation" not in command


def test_evaluation_validator_accepts_complete_and_rejects_truncation(tmp_path):
    raw, command, payload = _small_product(tmp_path)
    evaluation.validate_eval_product(raw, command)
    spec = next(iter(payload["results"]))
    rollout = next(iter(payload["results"][spec]))
    payload["results"][spec][rollout]["steps"] = 1
    raw.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="truncated"):
        evaluation.validate_eval_product(raw, command)


def test_split_comparator_product_has_one_policy_and_preserves_rollouts(tmp_path):
    raw, command, payload = _small_product(tmp_path)
    spec = next(iter(payload["results"]))
    split = tmp_path / "split.json"
    evaluation._split_comparator_product(payload, spec=spec, output=split)
    checked = evaluation.validate_eval_product(split, command, specs=(spec,))
    assert list(checked["results"]) == [spec]
