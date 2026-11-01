"""Exact process-level training continuation contracts."""

import json

import pytest

import Core as core_module
from Core import (
    Core,
    TRAINING_RESUME_FORMAT,
    _atomic_write_training_resume_state,
    _load_training_resume_state,
    _validate_training_resume_contract,
)
from scripts import v17_probe


def _payload():
    contract = {
        "train_timesteps": 200,
        "core_source_sha256": "registered-source",
    }
    return {
        "format": TRAINING_RESUME_FORMAT,
        "saved_at_utc": "2026-10-30T00:00:00Z",
        "completed_days": 87,
        "contract": contract,
        "core": {"ppo_buffer": [1, 2, 3], "market_day": 87},
        "loop_state": {"pending_count": 4},
        "global_rng_state": {"python": (3, (), None)},
    }


def test_exact_training_state_is_atomic_and_has_lightweight_progress(tmp_path):
    path = tmp_path / "resume" / "training_state.pt"
    payload = _payload()

    _atomic_write_training_resume_state(str(path), payload)
    loaded = _load_training_resume_state(str(path))

    assert loaded["completed_days"] == 87
    assert loaded["core"]["ppo_buffer"] == [1, 2, 3]
    assert _validate_training_resume_contract(
        loaded, payload["contract"]
    ) == 87
    progress = json.loads(
        (tmp_path / "resume/training_state.pt.progress.json").read_text()
    )
    assert progress == {
        "format": TRAINING_RESUME_FORMAT,
        "completed_days": 87,
        "training_days": 200,
        "saved_at_utc": "2026-10-30T00:00:00Z",
    }
    assert not list(path.parent.glob("*.tmp.*"))


def test_exact_training_resume_fails_closed_on_contract_drift():
    payload = _payload()
    changed = dict(payload["contract"], train_timesteps=201)
    with pytest.raises(ValueError, match="refusing to mix experiments"):
        _validate_training_resume_contract(payload, changed)


def test_exact_state_round_trips_the_live_core_not_only_policy_weights(
    tmp_path,
):
    core = Core(
        "Chicago",
        seed=1881,
        firm1_mode="RL",
        firm2_mode="static",
        total_customers_pool=8,
        simulation_sample_cap=3,
        enable_driver_supply=False,
        openai_api_key="",
        action_contract="two_lever_v1",
        v16_crn_tape=True,
        v16_branch_mode="one_day_deviation",
        v16_branch_design="stencil11",
        v16_store_cap=20,
    )
    core.training_logs = [{"batch": 0, "avg_reward": 1.25}]
    payload = _payload()
    payload["completed_days"] = 1
    payload["core"] = core
    path = tmp_path / "live_core.pt"

    _atomic_write_training_resume_state(str(path), payload)
    restored = _load_training_resume_state(str(path))["core"]

    assert isinstance(restored, Core)
    assert restored.training_logs == core.training_logs
    assert restored.market.curr_market == core.market.curr_market
    assert restored.firm1.agent.optimizer_state_dict().keys() == (
        core.firm1.agent.optimizer_state_dict().keys()
    )


def test_t3_partial_product_can_finish_from_exact_resume(
    monkeypatch, tmp_path
):
    product = tmp_path / "day_200.pt"
    product.write_bytes(b"partial-product")
    resume = tmp_path / "resume/training_state.pt"
    resume.parent.mkdir(parents=True)
    resume.write_bytes(b"exact-state")
    manifest = tmp_path / "training_manifest.json"
    command = [
        "python", "Core.py", "--training_resume_state", str(resume)
    ]
    calls = []
    monkeypatch.setattr(
        v17_probe, "_run",
        lambda command, *, log, env: calls.append(list(command)),
    )

    result = v17_probe._run_cached_job(
        command,
        product=product,
        manifest=manifest,
        log=tmp_path / "training.log",
        identity={"command": command},
        force=False,
    )

    assert result == "completed"
    assert calls == [command]
    saved_manifest = json.loads(manifest.read_text())
    assert saved_manifest["state"] == "completed"
    assert saved_manifest["identity"] == {"command": command}
    assert saved_manifest["product_sha256"] == v17_probe._sha256(product)


def test_t3_resume_rejects_changed_prelaunch_source_identity(tmp_path):
    product = tmp_path / "day_200.pt"
    product.write_bytes(b"partial-product")
    resume = tmp_path / "resume/training_state.pt"
    resume.parent.mkdir(parents=True)
    resume.write_bytes(b"exact-state")
    manifest = tmp_path / "training_manifest.json"
    manifest.write_text(json.dumps({
        "state": "running",
        "identity": {"sources": {"Core.py": "old"}},
    }))
    command = [
        "python", "Core.py", "--training_resume_state", str(resume)
    ]
    with pytest.raises(ValueError, match="stale T3 cache identity"):
        v17_probe._run_cached_job(
            command,
            product=product,
            manifest=manifest,
            log=tmp_path / "training.log",
            identity={"sources": {"Core.py": "new"}},
            force=False,
        )


def test_interrupted_training_continues_after_last_completed_day(
    monkeypatch, tmp_path
):
    core = Core(
        "Chicago",
        seed=1882,
        firm1_mode="RL",
        firm2_mode="heuristic",
        total_customers_pool=10,
        simulation_sample_cap=2,
        enable_driver_supply=False,
        openai_api_key="",
        training_curriculum="direct",
    )
    core.active_calibration = {}
    state = tmp_path / "exact_state.pt"
    real_writer = core_module._atomic_write_training_resume_state

    class PlannedInterruption(RuntimeError):
        pass

    def interrupt_after_day_two(path, payload):
        real_writer(path, payload)
        if int(payload["completed_days"]) == 2:
            raise PlannedInterruption("simulated process loss")

    run_args = dict(
        train_timesteps=4,
        train_customers_per_step=3,
        eval_timesteps=1,
        eval_customers_per_step=3,
        train_steps_per_day=1,
        ppo_update_interval_days=2,
        ppo_min_rollout_transitions=1,
        stochastic_training=False,
        firm1_action_interval_steps=1,
        firm2_action_interval_days=1,
        checkpoint_validation_horizon=2,
        checkpoint_validation_interval_days=100,
        checkpoint_validation_customers_per_step=3,
        checkpoint_validation_openings=1,
        evaluate_benchmark_suite=False,
        model_registry_dir=None,
        diagnostic_prefix=None,
        training_resume_state=str(state),
        training_resume_every_days=1,
    )
    monkeypatch.setattr(
        core_module,
        "_atomic_write_training_resume_state",
        interrupt_after_day_two,
    )
    with pytest.raises(PlannedInterruption):
        core.run_experiment(**run_args)
    assert [row["batch"] for row in core.training_logs] == [0, 1]

    payload = _load_training_resume_state(str(state))
    monkeypatch.setattr(
        core_module,
        "_atomic_write_training_resume_state",
        real_writer,
    )
    resumed = payload["core"]
    resumed.run_experiment(
        **run_args, training_resume_payload=payload
    )

    assert [row["batch"] for row in resumed.training_logs] == [0, 1, 2, 3]


def test_external_evaluation_only_suppresses_legacy_end_of_run_scoring(
    monkeypatch, tmp_path
):
    core = Core(
        "Chicago",
        seed=1883,
        firm1_mode="RL",
        firm2_mode="heuristic",
        total_customers_pool=10,
        simulation_sample_cap=2,
        enable_driver_supply=False,
        openai_api_key="",
        training_curriculum="direct",
    )
    core.active_calibration = {}
    monkeypatch.setattr(
        core,
        "_adaptive_policy_validation",
        lambda *args, **kwargs: pytest.fail(
            "legacy embedded evaluation must remain disabled"
        ),
    )
    training, evaluation = core.run_experiment(
        train_timesteps=1,
        train_customers_per_step=3,
        eval_timesteps=2,
        eval_customers_per_step=3,
        train_steps_per_day=1,
        ppo_update_interval_days=2,
        ppo_min_rollout_transitions=1,
        stochastic_training=False,
        firm1_action_interval_steps=1,
        firm2_action_interval_days=1,
        checkpoint_validation_horizon=2,
        checkpoint_validation_interval_days=1,
        checkpoint_validation_customers_per_step=3,
        checkpoint_validation_openings=1,
        evaluate_benchmark_suite=False,
        model_registry_dir=None,
        diagnostic_prefix=str(tmp_path / "external-only"),
        external_evaluation_only=True,
    )
    assert len(training) == 1
    assert evaluation == []
    assert core.competitive_backtest_report == {
        "passed": None,
        "status": "deferred_to_external_checkpoint_scoring",
        "failures": [],
        "evaluation_steps": 0,
    }
