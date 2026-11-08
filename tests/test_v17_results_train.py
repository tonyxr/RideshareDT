from pathlib import Path

import pytest

from scripts import v17_results_train as train


def _value(command, option):
    return command[command.index(option) + 1]


def test_registered_run_matrix_is_exact():
    assert [spec.name for spec in train.RUN_SPECS] == [
        "nyc_s101",
        "nyc_s202",
        "nyc_s303",
        "boston_s101",
        "boston_s202",
        "chicago_s101",
        "chicago_s202",
        "nyc_s101_no_driver_obs",
        "nyc_s101_own_profit",
        "nyc_s101_plain_ppo",
    ]


@pytest.mark.parametrize(
    ("name", "flag"),
    [
        ("nyc_s101_no_driver_obs", "--v17_ablate_driver_observation"),
        ("nyc_s101_own_profit", "--v17_ablate_profit_lead"),
    ],
)
def test_ablation_commands_are_labeled(name, flag, tmp_path):
    command = train.build_command(
        train.RUN_BY_NAME[name], days=20, output_root=tmp_path
    )
    assert flag in command
    assert _value(command, "--training_resume_every_days") == "1"
    assert _value(command, "--training_resume_state").endswith(
        f"{name}/resume/training_state.pt"
    )


def test_plain_ppo_omits_counterfactual_branch_and_q_flags(tmp_path):
    command = train.build_command(
        train.RUN_BY_NAME["nyc_s101_plain_ppo"],
        days=20,
        output_root=tmp_path,
    )
    assert _value(command, "--v16_policy_mode") == "ppo"
    for option in (
        "--v16_branch_mode",
        "--v16_branch_horizon_days",
        "--v16_branch_design",
        "--v16_audit_every",
        "--v16_store_cap",
        "--v16_value_head",
        "--v17_move_radius",
        "--v17_temperature_mode",
    ):
        assert option not in command


def test_main_command_uses_registered_q_learner_and_explicit_days(tmp_path):
    command = train.build_command(
        train.RUN_BY_NAME["nyc_s101"], days=1200, output_root=tmp_path
    )
    assert _value(command, "--train_timesteps") == "1200"
    assert _value(command, "--v16_branch_horizon_days") == "5"
    assert _value(command, "--v16_policy_mode") == "q"
    assert _value(command, "--v17_move_radius") == "1"
    assert _value(command, "--v17_temperature_mode") == "relative"
    report_prefix = _value(command, "--report_prefix")
    assert not Path(report_prefix).is_absolute()
    assert (train.BASE / report_prefix).resolve() == (
        tmp_path / "nyc_s101" / "nyc"
    ).resolve()


def test_run_selection_rejects_unknown_and_duplicates():
    assert train._selected_specs("nyc_s101,boston_s101") == (
        train.RUN_BY_NAME["nyc_s101"],
        train.RUN_BY_NAME["boston_s101"],
    )
    with pytest.raises(ValueError, match="unknown"):
        train._selected_specs("not-a-run")
    with pytest.raises(ValueError, match="duplicate"):
        train._selected_specs("nyc_s101,nyc_s101")


def test_status_payload_has_auditable_command_and_paths(tmp_path):
    spec = train.RUN_BY_NAME["nyc_s101"]
    command = ["python", "Core.py"]
    payload = train._status_payload(
        spec,
        state="running",
        days=20,
        run_dir=Path(tmp_path) / spec.name,
        command=command,
        pid=123,
    )
    assert payload["state"] == "running"
    assert payload["pid"] == 123
    assert payload["days_expected"] == 20
    assert payload["command"] == command


def test_retry_attempt_prefixes_are_unique_and_preserve_prior_files(tmp_path):
    run_dir = tmp_path / "nyc_s101"
    first_id, first_prefix = train._reserve_attempt_prefix(run_dir, "nyc")
    (first_prefix.parent / "nyc_events.jsonl").write_text(
        "prior evidence\n", encoding="utf-8"
    )
    second_id, second_prefix = train._reserve_attempt_prefix(run_dir, "nyc")

    assert first_id == "attempt-0001"
    assert second_id == "attempt-0002"
    assert first_prefix != second_prefix
    assert (first_prefix.parent / "nyc_events.jsonl").read_text(
        encoding="utf-8"
    ) == "prior evidence\n"


def test_command_can_use_reserved_attempt_journal(tmp_path):
    spec = train.RUN_BY_NAME["nyc_s101"]
    prefix = tmp_path / spec.name / "attempts/attempt-0001/nyc"
    command = train.build_command(
        spec,
        days=20,
        output_root=tmp_path,
        diagnostic_prefix=prefix,
    )
    value = _value(command, "--report_prefix")
    assert not Path(value).is_absolute()
    assert (train.BASE / value).resolve() == prefix.resolve()
