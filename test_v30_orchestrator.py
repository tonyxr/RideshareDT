import json

import pytest

from digital_twin.orchestrator import CYCLE_STAGES, ContinuousOptimizationOrchestrator


def _callbacks(calls, failing=None):
    callbacks = {}
    for stage in CYCLE_STAGES:
        def callback(manifest, stage=stage):
            calls.append(stage)
            if stage == failing:
                raise RuntimeError(f"failed:{stage}")
            return {"stage": stage, "input_stage_count": len(manifest.completed_stages)}
        callbacks[stage] = callback
    return callbacks


def test_cycle_commits_each_stage_and_never_auto_deploys(tmp_path):
    calls = []
    runner = ContinuousOptimizationOrchestrator(
        root=tmp_path, cycle_id="cycle-1", configuration={"market": "chicago"},
        callbacks=_callbacks(calls),
    )
    manifest = runner.run()
    assert calls == list(CYCLE_STAGES)
    assert manifest.status == "shadow_complete"
    assert manifest.deployment_authorization is None
    stored = json.loads(runner.manifest_path.read_text())
    assert stored["completed_stages"] == list(CYCLE_STAGES)


def test_failed_stage_is_recorded_and_resume_starts_there(tmp_path):
    calls = []
    failing = {"stage": "calibrate"}
    callbacks = _callbacks(calls)
    original = callbacks["calibrate"]

    def calibrate(manifest):
        if failing["stage"]:
            raise RuntimeError("calibration feed unavailable")
        return original(manifest)

    callbacks["calibrate"] = calibrate
    runner = ContinuousOptimizationOrchestrator(
        root=tmp_path, cycle_id="cycle-2", configuration={"x": 1}, callbacks=callbacks,
    )
    with pytest.raises(RuntimeError):
        runner.run()
    failed = runner.load_or_create()
    assert failed.status == "failed"
    assert failed.completed_stages == ["ingest", "assimilate_state"]
    failing["stage"] = None
    resumed = runner.run()
    assert resumed.status == "shadow_complete"
    assert calls.count("ingest") == 1 and calls.count("assimilate_state") == 1


def test_configuration_change_is_rejected_on_resume(tmp_path):
    first = ContinuousOptimizationOrchestrator(
        root=tmp_path, cycle_id="cycle", configuration={"x": 1},
        callbacks=_callbacks([]),
    )
    first.load_or_create()
    second = ContinuousOptimizationOrchestrator(
        root=tmp_path, cycle_id="cycle", configuration={"x": 2},
        callbacks=_callbacks([]),
    )
    with pytest.raises(ValueError):
        second.load_or_create()


def test_deployment_requires_explicit_complete_authorization(tmp_path):
    runner = ContinuousOptimizationOrchestrator(
        root=tmp_path, cycle_id="cycle", configuration={"x": 1},
        callbacks=_callbacks([]),
    )
    with pytest.raises(RuntimeError):
        runner.authorize_deployment(authorizer="owner", change_ticket="T-1", policy_artifact_sha256="abc")
    runner.run()
    with pytest.raises(ValueError, match="SHA-256"):
        runner.authorize_deployment(
            authorizer="owner", change_ticket="T-1",
            policy_artifact_sha256="abc",
        )
    deployed = runner.authorize_deployment(
        authorizer="owner", change_ticket="T-1", policy_artifact_sha256="a" * 64,
    )
    assert deployed.status == "deployed"
    assert deployed.deployment_authorization["change_ticket"] == "T-1"
