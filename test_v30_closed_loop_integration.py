from dataclasses import asdict
from datetime import datetime, timedelta, timezone

import numpy as np

from digital_twin.calibration import RecursiveLinearCalibrator
from digital_twin.experiments import ExperimentArm, StableExperimentAssigner
from digital_twin.identity_store import IdentityStateStore
from digital_twin.ledger import DecisionEvent, EventLedger, ObservationEvent, OutcomeEvent
from digital_twin.observation import ClosedLoopFeatures, V30ObservationContract
from digital_twin.off_policy import LoggedDecision, OffPolicyEvaluator
from digital_twin.orchestrator import ContinuousOptimizationOrchestrator
from digital_twin.policy_service import ShadowPolicy, ShadowPolicyService
from digital_twin.state_estimator import OnlineStateEstimator
from digital_twin.twin_ensemble import TwinMember, VersionedTwinEnsemble


T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_periodic_data_to_shadow_cycle_composes_existing_v30_components(tmp_path):
    ledger_path = tmp_path / "events.sqlite"
    identity_path = tmp_path / "identity.sqlite"
    observation_contract = V30ObservationContract(shock_features=False)
    state = OnlineStateEstimator(
        initial_means={
            "demand": 100.0, "supply": 80.0,
            "rival_quote_index": 1.0, "shock_probability": 0.0,
        }
    )
    calibrator = RecursiveLinearCalibrator(("fare", "wait_minutes"))
    ensemble = VersionedTwinEnsemble((
        TwinMember("low", {"elasticity": -1.0}, "market-v30"),
        TwinMember("high", {"elasticity": -3.0}, "market-v30"),
    ))
    assignment = StableExperimentAssigner(
        experiment_id="v30-cycle", salt="test-secret",
        arms=(ExperimentArm("behavior", 0.5), ExperimentArm("candidate", 0.5)),
    ).assign("rider:001", stratum="Chicago")

    def ingest(_manifest):
        with IdentityStateStore(identity_path) as identities:
            identities.register(
                "rider", "rider:001", traits={"segment": "commuter"},
            )
            identities.update_rider_experience(
                "rider:001", platform_experience=0.5,
                outside_experience=-0.25, alpha=0.2,
            )
        with EventLedger(ledger_path) as ledger:
            inserted = ledger.append_batch((
                DecisionEvent(
                    event_id="decision-1", request_id="request-1",
                    event_time=T0, market="Chicago", policy_id="behavior-v1",
                    observation_schema=observation_contract.schema_sha256,
                    observation={"fare": 12.0, "wait_minutes": 4.0},
                    action={"name": assignment.arm},
                    action_probabilities={"hold": 0.5, "base_up": 0.5},
                    tariff_before={"base_fare": 2.0},
                    tariff_after={"base_fare": 2.1}, rider_id="rider:001",
                ),
                OutcomeEvent(
                    event_id="outcome-1", request_id="request-1",
                    event_time=T0 + timedelta(minutes=20), market="Chicago",
                    accepted=True, completed=True, cancelled=False,
                    fare=12.0, driver_payment=7.0, contribution_profit=3.5,
                    wait_minutes=4.0, trip_minutes=16.0,
                ),
                ObservationEvent(
                    event_id="market-1", event_time=T0, market="Chicago",
                    source="collaborating-firm-feed",
                    values={
                        "demand": 120.0, "supply": 90.0,
                        "rival_quote_index": 1.05, "shock_probability": 0.1,
                    }, freshness_seconds=60.0,
                ),
            ))
        return {"inserted": inserted, "assignment": asdict(assignment)}

    def assimilate(_manifest):
        with EventLedger(ledger_path) as ledger:
            row = ledger.events(event_type="observation")[-1]
        updated = state.update(row["event_time"], row["values"])
        return {"state": asdict(updated)}

    def calibrate(_manifest):
        with EventLedger(ledger_path) as ledger:
            joined = ledger.joined_decisions(completed_only=True)[0]
        decision = joined["decision"]
        outcome = joined["outcome"]
        update = calibrator.update(
            {
                "fare": float(decision["observation"]["fare"]),
                "wait_minutes": float(decision["observation"]["wait_minutes"]),
            },
            float(outcome["contribution_profit"]),
        )
        return {"updates": calibrator.updates, "parameters": update["coefficients"]}

    def update_twin(_manifest):
        posterior = ensemble.update_weights(
            {"profit": 3.5},
            {"low": {"profit": 3.3}, "high": {"profit": 2.0}},
            observation_variance={"profit": 0.25},
        )
        return {"weights": posterior.weights, "updates": ensemble.updates}

    def train(_manifest):
        return {"policy_id": "candidate-v30", "mode": "existing-ppo-stack"}

    def evaluate(_manifest):
        estimate = OffPolicyEvaluator().evaluate((
            LoggedDecision(3.5, 0.5, 0.6, 3.0, 3.2, "day-1"),
            LoggedDecision(2.5, 0.5, 0.4, 2.8, 3.0, "day-2"),
        ))
        return {"estimate": asdict(estimate)}

    def shadow(_manifest):
        policy = ShadowPolicy(
            policy_id="candidate-v30", action_names=("hold", "base_up"),
            probability_policy=lambda observation: np.asarray((0.25, 0.75)),
            observation_schema_sha256=observation_contract.schema_sha256,
        )
        base = {name: 0.0 for name in observation_contract.base_names}
        with EventLedger(ledger_path) as ledger:
            recommendation = ShadowPolicyService(
                policy=policy, observation_contract=observation_contract,
                ledger=ledger,
                tariff_transition=lambda action, tariff: {
                    **tariff,
                    "base_fare": tariff["base_fare"]
                    + (0.1 if action == "base_up" else 0.0),
                },
            ).recommend(
                event_id="shadow-1", request_id="shadow-request-1",
                event_time=T0 + timedelta(days=1), market="Chicago",
                base_observation=base,
                closed_loop=ClosedLoopFeatures(
                    demand_now=120.0, demand_forecast=125.0,
                    supply_now=90.0, supply_forecast=92.0,
                    state_uncertainty=0.1, data_age_seconds=60.0,
                    decision_interval_seconds=3600.0,
                    episode_steps_remaining=900, episode_horizon_steps=1200,
                ),
                tariff_before={"base_fare": 2.1},
            )
        return {
            "policy_id": recommendation.policy_id,
            "action": recommendation.action_name,
            "deployment_authorized": False,
        }

    callbacks = {
        "ingest": ingest, "assimilate_state": assimilate,
        "calibrate": calibrate, "update_twin": update_twin,
        "train": train, "evaluate": evaluate, "shadow": shadow,
    }
    runner = ContinuousOptimizationOrchestrator(
        root=tmp_path / "cycles", cycle_id="cycle-1",
        configuration={"market": "Chicago", "policy": "v30"},
        callbacks=callbacks,
    )
    manifest = runner.run()

    assert manifest.status == "shadow_complete"
    assert manifest.deployment_authorization is None
    assert manifest.stage_outputs["ingest"]["inserted"] == 3
    assert manifest.stage_outputs["calibrate"]["updates"] == 1
    assert manifest.stage_outputs["update_twin"]["weights"]["low"] > 0.5
    assert manifest.stage_outputs["evaluate"]["estimate"]["support_warning"] is False
    assert manifest.stage_outputs["shadow"]["action"] in {"hold", "base_up"}
    with EventLedger(ledger_path) as ledger:
        shadow_event = next(
            row for row in ledger.events(event_type="decision")
            if row["event_id"] == "shadow-1"
        )
    assert shadow_event["action"]["name"] == manifest.stage_outputs["shadow"]["action"]
    assert shadow_event["action_probabilities"] == {"hold": 0.25, "base_up": 0.75}
    assert shadow_event["context"]["policy_mode"] == "sample_raw"
    assert shadow_event["context"]["deployment_authorized"] is False
