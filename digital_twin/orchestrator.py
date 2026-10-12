"""Crash-safe orchestration for a continuous v30 digital-twin cycle.

The orchestrator records work; it does not alter learner updates, reject policy
improvements, or restore a fallback policy.  Evaluation and shadowing are
ordinary stages.  Live deployment is a separate, explicit authorization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from .contracts import V30_CONTRACT_VERSION


CYCLE_STAGES = (
    "ingest",
    "assimilate_state",
    "calibrate",
    "update_twin",
    "train",
    "evaluate",
    "shadow",
)


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    data = _canonical_json(payload)
    with temporary.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass
class CycleManifest:
    cycle_id: str
    configuration_sha256: str
    created_at: str
    updated_at: str
    status: str = "running"
    completed_stages: list[str] = field(default_factory=list)
    stage_outputs: dict[str, Mapping[str, Any]] = field(default_factory=dict)
    failure: Optional[Mapping[str, Any]] = None
    deployment_authorization: Optional[Mapping[str, Any]] = None
    contract_version: str = V30_CONTRACT_VERSION

    def payload(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "cycle_id": self.cycle_id,
            "configuration_sha256": self.configuration_sha256,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
            "completed_stages": list(self.completed_stages),
            "stage_outputs": dict(self.stage_outputs),
            "failure": self.failure,
            "deployment_authorization": self.deployment_authorization,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "CycleManifest":
        if payload.get("contract_version") != V30_CONTRACT_VERSION:
            raise ValueError("unsupported optimization-cycle contract")
        completed = list(payload.get("completed_stages", []))
        if completed != list(CYCLE_STAGES[: len(completed)]):
            raise ValueError("cycle manifest contains a non-prefix stage history")
        return cls(
            cycle_id=str(payload["cycle_id"]),
            configuration_sha256=str(payload["configuration_sha256"]),
            created_at=str(payload["created_at"]),
            updated_at=str(payload["updated_at"]),
            status=str(payload["status"]),
            completed_stages=completed,
            stage_outputs=dict(payload.get("stage_outputs", {})),
            failure=payload.get("failure"),
            deployment_authorization=payload.get("deployment_authorization"),
            contract_version=str(payload["contract_version"]),
        )


StageCallback = Callable[[CycleManifest], Mapping[str, Any]]


class ContinuousOptimizationOrchestrator:
    def __init__(
        self,
        *,
        root: str | Path,
        cycle_id: str,
        configuration: Mapping[str, Any],
        callbacks: Mapping[str, StageCallback],
    ):
        self.root = Path(root)
        self.cycle_id = str(cycle_id).strip()
        if not self.cycle_id:
            raise ValueError("cycle id is required")
        self.configuration = dict(configuration)
        self.configuration_sha256 = _sha256(self.configuration)
        missing = [stage for stage in CYCLE_STAGES if stage not in callbacks]
        extra = sorted(set(callbacks) - set(CYCLE_STAGES))
        if missing or extra:
            raise ValueError(f"callback contract mismatch; missing={missing}, extra={extra}")
        self.callbacks = dict(callbacks)
        self.cycle_dir = self.root / self.cycle_id
        self.manifest_path = self.cycle_dir / "manifest.json"

    def _write(self, manifest: CycleManifest) -> None:
        manifest.updated_at = _now()
        _atomic_json(self.manifest_path, manifest.payload())

    def load_or_create(self) -> CycleManifest:
        if self.manifest_path.exists():
            payload = json.loads(self.manifest_path.read_text())
            manifest = CycleManifest.from_payload(payload)
            if manifest.cycle_id != self.cycle_id:
                raise ValueError("cycle id does not match its manifest")
            if manifest.configuration_sha256 != self.configuration_sha256:
                raise ValueError("cycle configuration changed during resume")
            return manifest
        stamp = _now()
        manifest = CycleManifest(
            cycle_id=self.cycle_id,
            configuration_sha256=self.configuration_sha256,
            created_at=stamp,
            updated_at=stamp,
        )
        self._write(manifest)
        return manifest

    def run(self) -> CycleManifest:
        manifest = self.load_or_create()
        if manifest.status == "shadow_complete" or manifest.status == "deployed":
            return manifest
        if manifest.status == "failed":
            # A retry is explicit and resumes the first unfinished stage.  No
            # model state is silently restored or substituted.
            manifest.status = "running"
            manifest.failure = None
            self._write(manifest)
        for stage in CYCLE_STAGES[len(manifest.completed_stages) :]:
            try:
                output = dict(self.callbacks[stage](manifest))
                # Force JSON validation before committing stage completion.
                _canonical_json(output)
            except BaseException as error:
                manifest.status = "failed"
                manifest.failure = {
                    "stage": stage,
                    "type": type(error).__name__,
                    "message": str(error),
                    "at": _now(),
                }
                self._write(manifest)
                raise
            manifest.stage_outputs[stage] = output
            manifest.completed_stages.append(stage)
            manifest.status = "shadow_complete" if stage == "shadow" else "running"
            manifest.failure = None
            self._write(manifest)
        return manifest

    def authorize_deployment(
        self,
        *,
        authorizer: str,
        change_ticket: str,
        policy_artifact_sha256: str,
    ) -> CycleManifest:
        manifest = self.load_or_create()
        if manifest.status != "shadow_complete" or manifest.completed_stages != list(CYCLE_STAGES):
            raise RuntimeError("deployment requires a completed shadow cycle")
        fields = {
            "authorizer": str(authorizer).strip(),
            "change_ticket": str(change_ticket).strip(),
            "policy_artifact_sha256": str(policy_artifact_sha256).strip(),
        }
        if any(not value for value in fields.values()):
            raise ValueError("deployment authorization fields must be non-empty")
        digest = fields["policy_artifact_sha256"].lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("policy_artifact_sha256 must be a full SHA-256 hex digest")
        fields["policy_artifact_sha256"] = digest
        manifest.deployment_authorization = {**fields, "authorized_at": _now()}
        manifest.status = "deployed"
        self._write(manifest)
        return manifest
