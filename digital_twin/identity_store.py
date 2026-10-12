"""Persistent pseudonymous rider and driver state for v30."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Mapping, Optional

from .contracts import IDENTITY_SCHEMA_VERSION


PROHIBITED_PII_KEYS = frozenset({
    "name", "first_name", "last_name", "email", "phone", "address",
    "license_number", "social_security_number",
})


def _canonical(value: Mapping[str, Any]) -> str:
    return json.dumps(dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _validate_entity(entity_type: str, entity_id: str, traits: Mapping[str, Any]) -> None:
    if entity_type not in {"rider", "driver"}:
        raise ValueError("entity_type must be rider or driver")
    if not str(entity_id).strip():
        raise ValueError("entity_id must be a non-empty pseudonymous identifier")
    prohibited = PROHIBITED_PII_KEYS.intersection(str(key).lower() for key in traits)
    if prohibited:
        raise ValueError("direct PII is prohibited: " + ",".join(sorted(prohibited)))


@dataclass(frozen=True)
class EntityRecord:
    entity_type: str
    entity_id: str
    traits: dict[str, Any]
    state: dict[str, Any]
    version: int
    created_at: str
    updated_at: str
    schema_version: str = IDENTITY_SCHEMA_VERSION


class IdentityStateStore:
    """Transactional store with immutable traits and versioned mutable state."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.path), timeout=30.0)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS entity_state (
                entity_type TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                schema_version TEXT NOT NULL,
                traits_json TEXT NOT NULL,
                traits_sha256 TEXT NOT NULL,
                state_json TEXT NOT NULL,
                version INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(entity_type, entity_id)
            )
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "IdentityStateStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def register(
        self,
        entity_type: str,
        entity_id: str,
        *,
        traits: Mapping[str, Any],
        initial_state: Optional[Mapping[str, Any]] = None,
    ) -> EntityRecord:
        _validate_entity(entity_type, entity_id, traits)
        traits_json = _canonical(traits)
        traits_hash = _hash(traits)
        state_json = _canonical(initial_state or {})
        timestamp = _now()
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            existing = self.connection.execute(
                "SELECT traits_sha256 FROM entity_state WHERE entity_type=? AND entity_id=?",
                (entity_type, entity_id),
            ).fetchone()
            if existing is not None:
                if existing["traits_sha256"] != traits_hash:
                    raise ValueError("immutable entity traits conflict with existing identity")
            else:
                self.connection.execute(
                    "INSERT INTO entity_state VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?)",
                    (
                        entity_type, entity_id, IDENTITY_SCHEMA_VERSION,
                        traits_json, traits_hash, state_json, timestamp, timestamp,
                    ),
                )
            self.connection.commit()
        except BaseException:
            self.connection.rollback()
            raise
        record = self.get(entity_type, entity_id)
        assert record is not None
        return record

    def get(self, entity_type: str, entity_id: str) -> Optional[EntityRecord]:
        row = self.connection.execute(
            "SELECT * FROM entity_state WHERE entity_type=? AND entity_id=?",
            (entity_type, entity_id),
        ).fetchone()
        if row is None:
            return None
        if row["schema_version"] != IDENTITY_SCHEMA_VERSION:
            raise ValueError("identity-store schema mismatch")
        return EntityRecord(
            entity_type=row["entity_type"], entity_id=row["entity_id"],
            traits=json.loads(row["traits_json"]), state=json.loads(row["state_json"]),
            version=int(row["version"]), created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def update_state(
        self,
        entity_type: str,
        entity_id: str,
        state: Mapping[str, Any],
        *,
        expected_version: int,
    ) -> EntityRecord:
        state_json = _canonical(state)
        timestamp = _now()
        cursor = self.connection.execute(
            """
            UPDATE entity_state
            SET state_json=?, version=version+1, updated_at=?
            WHERE entity_type=? AND entity_id=? AND version=?
            """,
            (state_json, timestamp, entity_type, entity_id, int(expected_version)),
        )
        if cursor.rowcount != 1:
            self.connection.rollback()
            current = self.get(entity_type, entity_id)
            if current is None:
                raise KeyError(f"unknown {entity_type} identity {entity_id!r}")
            raise RuntimeError(
                f"stale identity-state update: expected={expected_version}, current={current.version}"
            )
        self.connection.commit()
        record = self.get(entity_type, entity_id)
        assert record is not None
        return record

    @staticmethod
    def _ewma(previous: float, observed: float, alpha: float) -> float:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        values = (float(previous), float(observed))
        if any(value != value or abs(value) == float("inf") for value in values):
            raise ValueError("memory observations must be finite")
        return (1.0 - alpha) * values[0] + alpha * values[1]

    def update_rider_experience(
        self,
        rider_id: str,
        *,
        platform_experience: float,
        outside_experience: float,
        alpha: float,
    ) -> EntityRecord:
        record = self.get("rider", rider_id)
        if record is None:
            raise KeyError(f"unknown rider identity {rider_id!r}")
        state = dict(record.state)
        state["platform_experience_ewma"] = self._ewma(
            state.get("platform_experience_ewma", 0.0), platform_experience, alpha
        )
        state["outside_experience_ewma"] = self._ewma(
            state.get("outside_experience_ewma", 0.0), outside_experience, alpha
        )
        state["encounters"] = int(state.get("encounters", 0)) + 1
        return self.update_state("rider", rider_id, state, expected_version=record.version)

    def update_driver_experience(
        self,
        driver_id: str,
        *,
        availability: float,
        acceptance: float,
        earnings_per_hour: float,
        alpha: float,
    ) -> EntityRecord:
        record = self.get("driver", driver_id)
        if record is None:
            raise KeyError(f"unknown driver identity {driver_id!r}")
        state = dict(record.state)
        for key, observation in (
            ("availability_ewma", availability),
            ("acceptance_ewma", acceptance),
            ("earnings_per_hour_ewma", earnings_per_hour),
        ):
            state[key] = self._ewma(state.get(key, 0.0), observation, alpha)
        state["encounters"] = int(state.get("encounters", 0)) + 1
        return self.update_state("driver", driver_id, state, expected_version=record.version)

    def counts(self) -> dict[str, int]:
        rows = self.connection.execute(
            "SELECT entity_type, COUNT(*) AS count FROM entity_state GROUP BY entity_type"
        ).fetchall()
        return {row["entity_type"]: int(row["count"]) for row in rows}
