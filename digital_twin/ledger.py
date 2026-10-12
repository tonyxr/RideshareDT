"""Durable event ledger for periodic collaborating-firm data feeds.

The ledger records what the policy knew and did separately from outcomes that
may arrive hours or days later. Inserts are idempotent by event id and reject a
conflicting payload under the same id. SQLite is used deliberately: it gives a
small deployment a transactional reference implementation without introducing
an external service dependency. Production connectors can preserve this API.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Iterator, Mapping, Optional

from .contracts import EVENT_SCHEMA_VERSION


def _utc_iso(value: str | datetime) -> str:
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise TypeError("event_time must be an ISO string or datetime")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("event_time must be timezone-aware")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _json(value: Mapping[str, Any] | None) -> str:
    return json.dumps(dict(value or {}), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _required_text(value: str, name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return result


@dataclass(frozen=True)
class DecisionEvent:
    event_id: str
    request_id: str
    event_time: str | datetime
    market: str
    policy_id: str
    observation_schema: str
    observation: Mapping[str, Any]
    action: Mapping[str, Any]
    action_probabilities: Mapping[str, float]
    tariff_before: Mapping[str, float]
    tariff_after: Mapping[str, float]
    rider_id: Optional[str] = None
    driver_id: Optional[str] = None
    context: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = EVENT_SCHEMA_VERSION

    def canonical(self) -> dict[str, Any]:
        if self.schema_version != EVENT_SCHEMA_VERSION:
            raise ValueError("unsupported decision-event schema")
        probabilities = {str(key): float(value) for key, value in self.action_probabilities.items()}
        if not probabilities or any(
            not math.isfinite(value) or value < 0.0
            for value in probabilities.values()
        ):
            raise ValueError(
                "action probabilities must be present, finite and non-negative"
            )
        total = sum(probabilities.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError("action probabilities must sum to one")
        tariff_before = {
            str(key): float(value) for key, value in self.tariff_before.items()
        }
        tariff_after = {
            str(key): float(value) for key, value in self.tariff_after.items()
        }
        if (
            not tariff_before
            or set(tariff_before) != set(tariff_after)
            or any(
                not math.isfinite(value) or value < 0.0
                for value in (*tariff_before.values(), *tariff_after.values())
            )
        ):
            raise ValueError(
                "before/after tariffs must share non-empty keys and finite non-negative values"
            )
        return {
            "schema_version": self.schema_version,
            "event_id": _required_text(self.event_id, "event_id"),
            "request_id": _required_text(self.request_id, "request_id"),
            "event_time": _utc_iso(self.event_time),
            "market": _required_text(self.market, "market"),
            "policy_id": _required_text(self.policy_id, "policy_id"),
            "observation_schema": _required_text(self.observation_schema, "observation_schema"),
            "observation": dict(self.observation),
            "action": dict(self.action),
            "action_probabilities": probabilities,
            "tariff_before": tariff_before,
            "tariff_after": tariff_after,
            "rider_id": None if self.rider_id is None else _required_text(self.rider_id, "rider_id"),
            "driver_id": None if self.driver_id is None else _required_text(self.driver_id, "driver_id"),
            "context": dict(self.context),
        }


@dataclass(frozen=True)
class OutcomeEvent:
    event_id: str
    request_id: str
    event_time: str | datetime
    market: str
    accepted: bool
    completed: bool
    cancelled: bool
    fare: float
    driver_payment: float
    contribution_profit: float
    wait_minutes: Optional[float] = None
    trip_minutes: Optional[float] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = EVENT_SCHEMA_VERSION

    def canonical(self) -> dict[str, Any]:
        if self.schema_version != EVENT_SCHEMA_VERSION:
            raise ValueError("unsupported outcome-event schema")
        numeric = {
            "fare": float(self.fare),
            "driver_payment": float(self.driver_payment),
            "contribution_profit": float(self.contribution_profit),
        }
        if self.wait_minutes is not None:
            numeric["wait_minutes"] = float(self.wait_minutes)
        if self.trip_minutes is not None:
            numeric["trip_minutes"] = float(self.trip_minutes)
        if any(not math.isfinite(value) for value in numeric.values()):
            raise ValueError("outcome numeric fields must be finite")
        if numeric["fare"] < 0.0 or numeric["driver_payment"] < 0.0:
            raise ValueError("fare and driver payment must be non-negative")
        if any(
            numeric.get(key, 0.0) < 0.0
            for key in ("wait_minutes", "trip_minutes")
        ):
            raise ValueError("wait and trip durations must be non-negative")
        if self.completed and not self.accepted:
            raise ValueError("a completed ride must have been accepted")
        if self.completed and self.cancelled:
            raise ValueError("a ride cannot be completed and cancelled")
        return {
            "schema_version": self.schema_version,
            "event_id": _required_text(self.event_id, "event_id"),
            "request_id": _required_text(self.request_id, "request_id"),
            "event_time": _utc_iso(self.event_time),
            "market": _required_text(self.market, "market"),
            "accepted": bool(self.accepted),
            "completed": bool(self.completed),
            "cancelled": bool(self.cancelled),
            **numeric,
            "wait_minutes": None if self.wait_minutes is None else numeric["wait_minutes"],
            "trip_minutes": None if self.trip_minutes is None else numeric["trip_minutes"],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ObservationEvent:
    event_id: str
    event_time: str | datetime
    market: str
    source: str
    values: Mapping[str, Any]
    freshness_seconds: float
    schema_version: str = EVENT_SCHEMA_VERSION

    def canonical(self) -> dict[str, Any]:
        if self.schema_version != EVENT_SCHEMA_VERSION:
            raise ValueError("unsupported observation-event schema")
        freshness = float(self.freshness_seconds)
        if not math.isfinite(freshness) or freshness < 0.0:
            raise ValueError("freshness_seconds must be finite and non-negative")
        return {
            "schema_version": self.schema_version,
            "event_id": _required_text(self.event_id, "event_id"),
            "event_time": _utc_iso(self.event_time),
            "market": _required_text(self.market, "market"),
            "source": _required_text(self.source, "source"),
            "values": dict(self.values),
            "freshness_seconds": freshness,
        }


class EventLedger:
    """Transactional event store with delayed decision/outcome joins."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.path), timeout=30.0)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self._create_schema()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "EventLedger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                request_id TEXT,
                event_time TEXT NOT NULL,
                market TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                payload_sha256 TEXT NOT NULL,
                ingested_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_events_time ON events(event_time, event_id);
            CREATE INDEX IF NOT EXISTS idx_events_request ON events(request_id, event_type);
            """
        )
        self.connection.commit()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        try:
            self.connection.execute("BEGIN IMMEDIATE")
            yield
        except BaseException:
            self.connection.rollback()
            raise
        else:
            self.connection.commit()

    def _insert(self, event_type: str, payload: Mapping[str, Any]) -> bool:
        canonical_json = _json(payload)
        digest = _fingerprint(payload)
        existing = self.connection.execute(
            "SELECT event_type, payload_sha256 FROM events WHERE event_id=?",
            (payload["event_id"],),
        ).fetchone()
        if existing is not None:
            if existing["event_type"] == event_type and existing["payload_sha256"] == digest:
                return False
            raise ValueError(f"conflicting duplicate event_id={payload['event_id']!r}")
        self.connection.execute(
            "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                payload["event_id"], event_type, payload.get("request_id"),
                payload["event_time"], payload["market"], canonical_json, digest,
                datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            ),
        )
        return True

    def append_decision(self, event: DecisionEvent) -> bool:
        with self.transaction():
            return self._insert("decision", event.canonical())

    def append_outcome(self, event: OutcomeEvent) -> bool:
        with self.transaction():
            return self._insert("outcome", event.canonical())

    def append_observation(self, event: ObservationEvent) -> bool:
        with self.transaction():
            return self._insert("observation", event.canonical())

    def append_batch(self, events: Iterable[DecisionEvent | OutcomeEvent | ObservationEvent]) -> int:
        inserted = 0
        with self.transaction():
            for event in events:
                if isinstance(event, DecisionEvent):
                    inserted += int(self._insert("decision", event.canonical()))
                elif isinstance(event, OutcomeEvent):
                    inserted += int(self._insert("outcome", event.canonical()))
                elif isinstance(event, ObservationEvent):
                    inserted += int(self._insert("observation", event.canonical()))
                else:
                    raise TypeError(f"unsupported event type: {type(event).__name__}")
        return inserted

    def events(self, *, event_type: Optional[str] = None, market: Optional[str] = None) -> list[dict[str, Any]]:
        conditions: list[str] = []
        params: list[str] = []
        if event_type is not None:
            conditions.append("event_type=?")
            params.append(str(event_type))
        if market is not None:
            conditions.append("market=?")
            params.append(str(market))
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        rows = self.connection.execute(
            "SELECT event_type, payload_json FROM events" + where + " ORDER BY event_time, event_id",
            params,
        ).fetchall()
        return [{"event_type": row["event_type"], **json.loads(row["payload_json"])} for row in rows]

    def joined_decisions(self, *, completed_only: bool = False) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT d.payload_json AS decision_json, o.payload_json AS outcome_json
            FROM events d
            LEFT JOIN events o ON o.request_id=d.request_id AND o.event_type='outcome'
            WHERE d.event_type='decision'
            ORDER BY d.event_time, d.event_id, o.event_time, o.event_id
            """
        ).fetchall()
        result = []
        for row in rows:
            outcome = None if row["outcome_json"] is None else json.loads(row["outcome_json"])
            if completed_only and (outcome is None or not outcome["completed"]):
                continue
            result.append({"decision": json.loads(row["decision_json"]), "outcome": outcome})
        return result
