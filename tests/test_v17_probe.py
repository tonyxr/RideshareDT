import copy
import json
import sys

import pytest

from scripts import v17_probe as t3
from scripts.v17_config import ACTION_LEVELS, canonical_hash, load_config
from v17_objective import OBJECTIVE_NAME


def _policy_sweep_payload(
    *, config=None, days=t3.VALIDATION_DAYS,
    openings=t3.VALIDATION_OPENINGS, shift=0.0,
):
    config = load_config() if config is None else config
    steps = days * t3.STEPS_PER_DAY
    rollouts = [
        (f"{rival}__opening_{opening}", rival)
        for rival in t3.VALIDATION_RIVALS
        for opening in range(1, openings + 1)
    ]
    results = {}
    traces = []
    for rival_index, (rollout, _) in enumerate(rollouts):
        own_profit = 4.0 + shift + rival_index * 0.01
        rival_profit = 2.0
        results[rollout] = {
            "own_profit": own_profit,
            "rival_profit": rival_profit,
            "own_share": 0.5,
            "rival_share": 0.5,
            "fulfillment": 0.95,
            "steps": steps,
        }
        traces.extend(
            {
                "rollout": rollout,
                "step": step,
                "profit_per_request": own_profit,
                "rival_profit_per_request": rival_profit,
            }
            for step in range(steps)
        )
    return {
        "protocol": "v26r_policy_sweep_common_random_numbers",
        "city": "New York City",
        "horizon": steps,
        "customers_per_step": 50,
        "steps_per_day": t3.STEPS_PER_DAY,
        "policy_seed": t3.VALIDATION_SEED,
        "rollouts": [list(row) for row in rollouts],
        "v26r": {
            "action_contract": config["action"]["contract"],
            "supply_model": "v3",
            "persistent_driver_stock": False,
            "effective_fleet_scale": config["city_inputs"]["nyc"][
                "fleet_scale"
            ],
        },
        "results": {"rl_actor": results},
        "traces": {"rl_actor": traces},
    }


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
    assert "--checkpoint_validation_customers" not in command
    assert command[command.index("--training_resume_state") + 1].endswith(
        "h5/resume/training_state.pt"
    )
    assert command[command.index("--training_resume_every_days") + 1] == "1"
    assert "--external_evaluation_only" in command
    assert "--eval_timesteps" not in command
    assert "--checkpoint_validation_horizon" not in command
    t3.validate_training_command_contract(command)


def test_t3_rejects_embedded_short_evaluation(tmp_path):
    command = t3.training_command(
        load_config(), horizon=5, output=tmp_path
    )
    command.extend(("--eval_timesteps", "2"))
    with pytest.raises(ValueError, match="shorter than"):
        t3.validate_training_command_contract(command)


def test_t3_external_scoring_must_be_long_and_replicated():
    t3.validate_external_scoring_contract(
        checkpoint_days=t3.CHECKPOINT_DAYS,
        validation_days=t3.VALIDATION_DAYS,
        validation_openings=t3.VALIDATION_OPENINGS,
    )
    with pytest.raises(ValueError, match="too short"):
        t3.validate_external_scoring_contract(
            checkpoint_days=t3.CHECKPOINT_DAYS,
            validation_days=2,
            validation_openings=t3.VALIDATION_OPENINGS,
        )
    with pytest.raises(ValueError, match="openings"):
        t3.validate_external_scoring_contract(
            checkpoint_days=t3.CHECKPOINT_DAYS,
            validation_days=t3.VALIDATION_DAYS,
            validation_openings=1,
        )


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
            path = output / f"h5/validation/day_{day}/{case}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(_policy_sweep_payload(shift=shift)))
    report = t3.analyze_validation(output, 5)
    assert report["pass"] is True
    assert report["scores"][100]["final_objective"] == pytest.approx(2.54)
    assert report["scores"][100]["own_profit"] == pytest.approx(4.04)
    assert report["scores"][100]["competitor_profit"] == pytest.approx(2.0)
    assert report["scores"][100]["profit_lead"] == pytest.approx(2.04)
    assert all(row["pass"] for row in report["comparisons"])


def test_t3_policy_sweep_validator_rejects_truncated_rollout(tmp_path):
    payload = _policy_sweep_payload()
    rollout = next(iter(payload["results"]["rl_actor"]))
    payload["results"]["rl_actor"][rollout]["steps"] -= 1
    path = tmp_path / "truncated.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="truncated"):
        t3.validate_policy_sweep_product(
            path,
            config=load_config(),
            validation_days=t3.VALIDATION_DAYS,
            validation_openings=t3.VALIDATION_OPENINGS,
        )


def test_t3_policy_sweep_validator_rejects_missing_trace_step(tmp_path):
    payload = _policy_sweep_payload()
    payload["traces"]["rl_actor"].pop()
    path = tmp_path / "missing-trace.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="trace coverage"):
        t3.validate_policy_sweep_product(
            path,
            config=load_config(),
            validation_days=t3.VALIDATION_DAYS,
            validation_openings=t3.VALIDATION_OPENINGS,
        )


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
        "config_hash": canonical_hash(config),
        "action_contract": contract,
        "levels": list(ACTION_LEVELS[contract]),
        "openings": t3.REQUIRED_T2_OPENINGS,
        "ranking_objective": OBJECTIVE_NAME,
    }
    return config, status, report


def test_t3_requires_current_passed_widened_t2():
    config, status, report = _valid_t2_dependency()
    t3.validate_t2_dependency(config, status, report)


def test_explicit_legacy_t2_acceptance_is_validated_and_audited(tmp_path):
    root = tmp_path / "legacy_t2"
    root.mkdir()
    status = {
        "phase": "T2", "state": "passed", "pass": True,
        "action_contract": "two_lever_v1",
    }
    job = {
        "best_fixed": {"final": {
            "own_profit": 3.0,
            "competitor_profit": 2.0,
            "final_objective": 1.5,
            "n": 3,
        }},
        "adaptation_value": {"final": {
            "switch_minus_fixed": {"final_objective": 0.1},
        }},
    }
    report = {
        "phase": "T2",
        "action_contract": "two_lever_v1",
        "levels": list(ACTION_LEVELS["two_lever_v1"]),
        "openings": 3,
        "ranking_objective": OBJECTIVE_NAME,
        "jobs": {
            case: {f"rival_{index}": copy.deepcopy(job) for index in range(9)}
            for case in t3.CASES
        },
    }
    (root / "status.json").write_text(json.dumps(status))
    (root / "report.json").write_text(json.dumps(report))

    loaded_status, loaded_report, provenance = (
        t3.load_accepted_legacy_t2_dependency(root)
    )
    assert loaded_status["pass"] is True
    assert loaded_report["action_contract"] == "two_lever_v1"
    assert provenance["mode"].startswith("explicit_user_acceptance")
    assert provenance["report_sha256"] == t3._sha256(root / "report.json")


def test_legacy_t2_acceptance_rejects_incomplete_evidence(tmp_path):
    root = tmp_path / "legacy_t2"
    root.mkdir()
    (root / "status.json").write_text(json.dumps({
        "phase": "T2", "state": "passed", "pass": True,
        "action_contract": "two_lever_v1",
    }))
    (root / "report.json").write_text(json.dumps({
        "phase": "T2",
        "action_contract": "two_lever_v1",
        "levels": list(ACTION_LEVELS["two_lever_v1"]),
        "openings": 3,
        "ranking_objective": OBJECTIVE_NAME,
        "jobs": {},
    }))
    with pytest.raises(ValueError, match="case coverage"):
        t3.load_accepted_legacy_t2_dependency(root)


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
    status["report_sha256"] = t3._sha256(report_path)
    status_path.write_text(json.dumps(status))
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
