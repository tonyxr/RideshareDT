from datetime import datetime, timedelta, timezone
import sqlite3

import pytest

from digital_twin.ledger import DecisionEvent, EventLedger, ObservationEvent, OutcomeEvent


NOW = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)


def decision(event_id="d1", request_id="r1", when=NOW):
    return DecisionEvent(
        event_id=event_id,
        request_id=request_id,
        event_time=when,
        market="Chicago",
        policy_id="policy-v30-a",
        observation_schema="v30.observation.1",
        observation={"demand": 1.2},
        action={"base_fare": 1.05},
        action_probabilities={"hold": 0.25, "base_up": 0.75},
        tariff_before={"base_fare": 1.0},
        tariff_after={"base_fare": 1.05},
        rider_id="rider:abc",
    )


def outcome(event_id="o1", request_id="r1", when=NOW + timedelta(minutes=20)):
    return OutcomeEvent(
        event_id=event_id,
        request_id=request_id,
        event_time=when,
        market="Chicago",
        accepted=True,
        completed=True,
        cancelled=False,
        fare=18.0,
        driver_payment=12.0,
        contribution_profit=4.0,
        wait_minutes=5.0,
    )


def test_idempotent_append_and_conflicting_duplicate(tmp_path):
    with EventLedger(tmp_path / "events.sqlite") as ledger:
        assert ledger.append_decision(decision()) is True
        assert ledger.append_decision(decision()) is False
        conflicting = decision()
        conflicting = DecisionEvent(**{**conflicting.__dict__, "policy_id": "different"})
        with pytest.raises(ValueError, match="conflicting duplicate"):
            ledger.append_decision(conflicting)


def test_delayed_outcome_join_and_completed_filter(tmp_path):
    path = tmp_path / "events.sqlite"
    with EventLedger(path) as ledger:
        ledger.append_decision(decision())
        assert ledger.joined_decisions() == [{"decision": decision().canonical(), "outcome": None}]
    with EventLedger(path) as ledger:
        ledger.append_outcome(outcome())
        joined = ledger.joined_decisions(completed_only=True)
        assert joined[0]["decision"]["request_id"] == "r1"
        assert joined[0]["outcome"]["contribution_profit"] == 4.0


def test_event_time_order_is_independent_of_ingestion_order(tmp_path):
    with EventLedger(tmp_path / "events.sqlite") as ledger:
        ledger.append_batch([
            decision("d2", "r2", NOW + timedelta(hours=1)),
            decision("d1", "r1", NOW),
        ])
        assert [row["event_id"] for row in ledger.events()] == ["d1", "d2"]


def test_batch_is_atomic_on_invalid_or_conflicting_event(tmp_path):
    with EventLedger(tmp_path / "events.sqlite") as ledger:
        ledger.append_decision(decision())
        with pytest.raises(ValueError):
            ledger.append_batch([
                decision("d2", "r2"),
                DecisionEvent(**{**decision().__dict__, "policy_id": "conflict"}),
            ])
        assert [row["event_id"] for row in ledger.events()] == ["d1"]


def test_decision_requires_normalized_behavior_probabilities():
    with pytest.raises(ValueError, match="sum to one"):
        DecisionEvent(**{**decision().__dict__, "action_probabilities": {"hold": 0.8}}).canonical()


def test_naive_timestamp_is_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        decision(when=datetime(2026, 1, 1)).canonical()


def test_impossible_completion_is_rejected():
    with pytest.raises(ValueError, match="must have been accepted"):
        OutcomeEvent(**{**outcome().__dict__, "accepted": False}).canonical()


def test_observation_freshness_and_source_filter(tmp_path):
    event = ObservationEvent(
        event_id="weather-1",
        event_time=NOW,
        market="Chicago",
        source="weather",
        values={"rain_probability": 0.4},
        freshness_seconds=30,
    )
    with EventLedger(tmp_path / "events.sqlite") as ledger:
        ledger.append_observation(event)
        assert ledger.events(event_type="observation", market="Chicago")[0]["source"] == "weather"
    with pytest.raises(ValueError):
        ObservationEvent(**{**event.__dict__, "freshness_seconds": -1}).canonical()


@pytest.mark.parametrize("freshness", [float("nan"), float("inf"), -1.0])
def test_observation_freshness_must_be_finite_and_nonnegative(freshness):
    event = ObservationEvent(
        event_id="weather-invalid", event_time=NOW, market="Chicago",
        source="weather", values={"rain_probability": 0.4},
        freshness_seconds=freshness,
    )
    with pytest.raises(ValueError, match="freshness_seconds"):
        event.canonical()


def test_tariff_and_outcome_physical_domains_are_fail_closed():
    with pytest.raises(ValueError, match="tariffs"):
        DecisionEvent(**{
            **decision().__dict__, "tariff_after": {"base_fare": -1.0},
        }).canonical()
    with pytest.raises(ValueError, match="durations"):
        OutcomeEvent(**{**outcome().__dict__, "wait_minutes": -0.1}).canonical()
    with pytest.raises(ValueError, match="completed and cancelled"):
        OutcomeEvent(**{**outcome().__dict__, "cancelled": True}).canonical()


def test_schema_survives_process_restart(tmp_path):
    path = tmp_path / "events.sqlite"
    with EventLedger(path) as ledger:
        ledger.append_batch([decision(), outcome()])
    with EventLedger(path) as ledger:
        assert len(ledger.events()) == 2
        assert ledger.joined_decisions()[0]["outcome"]["completed"] is True
