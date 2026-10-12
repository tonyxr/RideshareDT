"""Stable randomized assignments for controlled v30 field experiments."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Sequence


@dataclass(frozen=True)
class ExperimentArm:
    name: str
    probability: float


@dataclass(frozen=True)
class ExperimentAssignment:
    experiment_id: str
    unit_id: str
    stratum: str
    arm: str
    assignment_probability: float
    assignment_digest: str


class StableExperimentAssigner:
    """Hash-based randomization with exact, logged assignment propensities."""

    def __init__(self, *, experiment_id: str, salt: str, arms: Sequence[ExperimentArm]):
        self.experiment_id = str(experiment_id).strip()
        self.salt = str(salt)
        self.arms = tuple(arms)
        if not self.experiment_id or not self.salt or not self.arms:
            raise ValueError("experiment id, secret salt and arms are required")
        if len({arm.name for arm in self.arms}) != len(self.arms):
            raise ValueError("experiment arm names must be unique")
        probabilities = [float(arm.probability) for arm in self.arms]
        if any(not math.isfinite(value) or value <= 0 for value in probabilities):
            raise ValueError("experiment probabilities must be finite and positive")
        if abs(sum(probabilities) - 1.0) > 1e-12:
            raise ValueError("experiment probabilities must sum to one")

    @staticmethod
    def _validated_unit_id(unit_id: str) -> str:
        result = str(unit_id).strip()
        if not result:
            raise ValueError("pseudonymous experiment unit id is required")
        # Direct contact identifiers are never valid randomization keys.
        if "@" in result or result.replace("+", "").replace("-", "").isdigit():
            raise ValueError("experiment unit id must be pseudonymous, not direct PII")
        return result

    def assign(self, unit_id: str, *, stratum: str = "all") -> ExperimentAssignment:
        unit = self._validated_unit_id(unit_id)
        stratum_value = str(stratum).strip() or "all"
        payload = "\x1f".join((self.salt, self.experiment_id, stratum_value, unit)).encode()
        digest = hashlib.sha256(payload).hexdigest()
        uniform = int(digest[:16], 16) / float(1 << 64)
        cumulative = 0.0
        selected = self.arms[-1]
        for arm in self.arms:
            cumulative += float(arm.probability)
            if uniform < cumulative:
                selected = arm
                break
        return ExperimentAssignment(
            experiment_id=self.experiment_id,
            unit_id=unit,
            stratum=stratum_value,
            arm=selected.name,
            assignment_probability=float(selected.probability),
            assignment_digest=digest,
        )
