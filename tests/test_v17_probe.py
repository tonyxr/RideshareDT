import json

import pytest

from scripts import v17_probe as t3
from scripts.v17_config import load_config


def test_t3_training_command_matches_revb_probe_contract(tmp_path):
    command = t3.training_command(
        load_config(), horizon=5, output=tmp_path
    )
    assert command[command.index("--train_timesteps") + 1] == "200"
    assert command[command.index("--v16_branch_horizon_days") + 1] == "5"
    assert command[command.index("--v16_audit_every") + 1] == "4"
    assert command[command.index("--v17_probe_second_tape_fraction") + 1] == "0.25"
    assert command[command.index("--v17_probe_checkpoint_days") + 1] == "100,150,200"
    assert "--current_active_driver_observation" in command
    assert command[command.index("--supply_model") + 1] == "v3"
    assert command[command.index("--shortage_willingness_share") + 1] == "0.5"


def test_t3_validation_command_uses_exact_cases_competitors_and_final_policy(tmp_path):
    command = t3.validation_command(
        load_config(), horizon=7, day=150,
        case="A1_shortage_severe", output=tmp_path,
    )
    assert command[command.index("--policy_sweep") + 1] == "rl_actor"
    assert command[command.index("--policy_sweep_openings") + 1] == "3"
    assert command[command.index("--policy_sweep_horizon") + 1] == "168"
    assert command[command.index("--policy_sweep_rivals") + 1] == ",".join(
        t3.VALIDATION_RIVALS
    )
    assert command[command.index("--trained_model_in") + 1].endswith(
        "h7/checkpoints/day_150.pt"
    )
    assert command[command.index("--shock_scenario") + 1].endswith(
        "scenarios/v17_t2_shortage_severe.json"
    )


def test_t3_validation_ranks_on_final_objective_and_reports_components(tmp_path):
    output = tmp_path
    for day, shift in zip(t3.CHECKPOINT_DAYS, (0.0, 0.1, 0.2)):
        for case in t3.CASES:
            rows = {}
            for rival_index, rival in enumerate(t3.VALIDATION_RIVALS):
                for opening in range(t3.VALIDATION_OPENINGS):
                    rows[f"{rival}__opening_{opening + 1}"] = {
                        "own_profit": 4.0 + shift + rival_index * 0.01,
                        "rival_profit": 2.0,
                    }
            path = output / f"h5/validation/day_{day}/{case}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"results": {"rl_actor": rows}}))
    report = t3.analyze_validation(output, 5)
    assert report["pass"] is True
    assert report["scores"][100]["final_objective"] == pytest.approx(2.51)
    assert report["scores"][100]["own_profit"] == pytest.approx(4.01)
    assert report["scores"][100]["competitor_profit"] == pytest.approx(2.0)
    assert all(row["pass"] for row in report["comparisons"])


def test_t3_plan_counts_two_horizons_three_checkpoints_and_two_cases():
    expected = (
        len(t3.HORIZONS) * len(t3.CHECKPOINT_DAYS) * len(t3.CASES)
        * len(t3.VALIDATION_RIVALS) * t3.VALIDATION_OPENINGS
    )
    assert expected == 108
