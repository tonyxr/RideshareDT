import json
import sys

import pytest

from scripts import v17_probe as t3
from scripts.v17_config import ACTION_LEVELS, canonical_hash, load_config
from v17_objective import OBJECTIVE_NAME


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
    assert command[command.index("--action_contract") + 1] == "two_lever_v3"
    assert command[command.index("--checkpoint_validation_customers") + 1] == "50"
    assert command[command.index("--training_resume_state") + 1].endswith(
        "h5/resume/training_state.pt"
    )
    assert command[command.index("--training_resume_every_days") + 1] == "1"


def test_t3_short_command_uses_60_days_and_three_short_checkpoints(tmp_path):
    command = t3.training_command(
        load_config(), horizon=5, output=tmp_path,
        training_days=t3.SHORT_TRAINING_DAYS,
        checkpoint_days=t3.SHORT_CHECKPOINT_DAYS,
    )
    assert command[command.index("--train_timesteps") + 1] == "60"
    assert command[command.index("--v17_probe_checkpoint_days") + 1] == "30,45,60"


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
    assert report["scores"][100]["profit_lead"] == pytest.approx(2.01)
    assert all(row["pass"] for row in report["comparisons"])


def test_t3_plan_counts_two_horizons_three_checkpoints_and_two_cases():
    expected = (
        len(t3.HORIZONS) * len(t3.CHECKPOINT_DAYS) * len(t3.CASES)
        * len(t3.VALIDATION_RIVALS) * t3.VALIDATION_OPENINGS
    )
    assert expected == 108


def _valid_t2_dependency():
    config = load_config()
    contract = config["action"]["contract"]
    status = {
        "state": "passed",
        "pass": True,
        "config_hash": canonical_hash(config),
        "action_contract": contract,
    }
    report = {
        "action_contract": contract,
        "levels": list(ACTION_LEVELS[contract]),
        "openings": t3.REQUIRED_T2_OPENINGS,
        "ranking_objective": OBJECTIVE_NAME,
    }
    return config, status, report


def test_t3_requires_current_passed_widened_t2():
    config, status, report = _valid_t2_dependency()
    t3.validate_t2_dependency(config, status, report)


@pytest.mark.parametrize(
    ("target", "key", "value"),
    (
        ("status", "state", "failed"),
        ("status", "config_hash", "stale"),
        ("status", "action_contract", "two_lever_v1"),
        ("report", "action_contract", "two_lever_v1"),
        ("report", "levels", [0.8, 0.9, 1.0, 1.2, 1.5]),
        ("report", "openings", 3),
        ("report", "ranking_objective", "own_profit"),
    ),
)
def test_t3_rejects_failed_or_stale_t2(target, key, value):
    config, status, report = _valid_t2_dependency()
    (status if target == "status" else report)[key] = value
    with pytest.raises(ValueError):
        t3.validate_t2_dependency(config, status, report)


def test_t3_rejects_stale_t2_before_starting_training(monkeypatch, tmp_path):
    config, status, report = _valid_t2_dependency()
    status["config_hash"] = "stale"
    status_path = tmp_path / "t2_status.json"
    report_path = tmp_path / "t2_report.json"
    status_path.write_text(json.dumps(status))
    report_path.write_text(json.dumps(report))
    monkeypatch.setattr(t3, "T2_STATUS", status_path)
    monkeypatch.setattr(t3, "T2_REPORT", report_path)
    monkeypatch.setattr(
        t3,
        "_run_cached_job",
        lambda *args, **kwargs: pytest.fail("training must not start"),
    )
    output = tmp_path / "t3"
    monkeypatch.setattr(
        sys, "argv", ["v17_probe.py", "--output", str(output)]
    )
    with pytest.raises(ValueError, match="stale configuration hash"):
        t3.main()
    saved = json.loads((output / "status.json").read_text())
    assert saved["state"] == "failed"
    assert "stale configuration hash" in saved["error"]
