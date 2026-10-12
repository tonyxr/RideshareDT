"""Versioned probabilistic ensemble of plausible v30 twin parameterizations."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping, Sequence

import numpy as np

from .contracts import TWIN_SCHEMA_VERSION


@dataclass(frozen=True)
class TwinMember:
    member_id: str
    parameters: dict[str, float]
    structural_version: str

    def canonical(self) -> dict[str, object]:
        if not self.member_id or not self.structural_version:
            raise ValueError("twin member identity/version must be non-empty")
        values = {str(key): float(value) for key, value in self.parameters.items()}
        if not values or any(not math.isfinite(value) for value in values.values()):
            raise ValueError("twin parameters must be non-empty and finite")
        return {
            "member_id": self.member_id,
            "parameters": values,
            "structural_version": self.structural_version,
        }


@dataclass(frozen=True)
class EnsemblePrediction:
    mean: dict[str, float]
    variance: dict[str, float]
    member_predictions: dict[str, dict[str, float]]
    weights: dict[str, float]


class VersionedTwinEnsemble:
    def __init__(self, members: Sequence[TwinMember], *, weights: Sequence[float] | None = None):
        if not members or len({member.member_id for member in members}) != len(members):
            raise ValueError("twin members must be non-empty with unique identities")
        self.members = tuple(members)
        for member in self.members:
            member.canonical()
        initial = np.ones(len(members), dtype=float) if weights is None else np.asarray(weights, dtype=float)
        if initial.shape != (len(members),) or np.any(initial < 0) or not np.all(np.isfinite(initial)):
            raise ValueError("ensemble weights are invalid")
        if initial.sum() <= 0:
            raise ValueError("at least one ensemble weight must be positive")
        self.weights = initial / initial.sum()
        self.updates = 0
        canonical = [member.canonical() for member in self.members]
        self.ensemble_sha256 = hashlib.sha256(
            json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def predict(self, member_predictions: Mapping[str, Mapping[str, float]]) -> EnsemblePrediction:
        expected_ids = {member.member_id for member in self.members}
        if set(member_predictions) != expected_ids:
            raise ValueError("predictions must cover every ensemble member exactly")
        output_keys: set[str] | None = None
        canonical: dict[str, dict[str, float]] = {}
        for member in self.members:
            values = {str(key): float(value) for key, value in member_predictions[member.member_id].items()}
            if not values or any(not math.isfinite(value) for value in values.values()):
                raise ValueError("member predictions must be finite and non-empty")
            if output_keys is None:
                output_keys = set(values)
            elif set(values) != output_keys:
                raise ValueError("all twin members must predict the same outputs")
            canonical[member.member_id] = values
        assert output_keys is not None
        mean: dict[str, float] = {}
        variance: dict[str, float] = {}
        for key in sorted(output_keys):
            vector = np.asarray([canonical[member.member_id][key] for member in self.members])
            average = float(self.weights @ vector)
            mean[key] = average
            variance[key] = float(self.weights @ ((vector - average) ** 2))
        return EnsemblePrediction(
            mean=mean, variance=variance, member_predictions=canonical,
            weights={member.member_id: float(weight) for member, weight in zip(self.members, self.weights)},
        )

    def update_weights(
        self,
        observed: Mapping[str, float],
        member_predictions: Mapping[str, Mapping[str, float]],
        *,
        observation_variance: Mapping[str, float],
        minimum_weight: float = 1e-9,
    ) -> EnsemblePrediction:
        prediction = self.predict(member_predictions)
        keys = set(prediction.mean)
        if set(observed) != keys or set(observation_variance) != keys:
            raise ValueError("observation and variance outputs must match predictions")
        log_likelihood = np.zeros(len(self.members), dtype=float)
        for index, member in enumerate(self.members):
            for key in keys:
                variance = float(observation_variance[key])
                value = float(observed[key])
                if variance <= 0 or not math.isfinite(variance) or not math.isfinite(value):
                    raise ValueError("observations must be finite and variances positive")
                error = value - prediction.member_predictions[member.member_id][key]
                log_likelihood[index] += -0.5 * (math.log(2 * math.pi * variance) + error * error / variance)
        log_posterior = np.log(np.maximum(self.weights, minimum_weight)) + log_likelihood
        log_posterior -= np.max(log_posterior)
        posterior = np.exp(log_posterior)
        posterior = np.maximum(posterior, float(minimum_weight))
        self.weights = posterior / posterior.sum()
        self.updates += 1
        return self.predict(member_predictions)

    def sample_member(self, rng: np.random.Generator) -> TwinMember:
        index = int(rng.choice(len(self.members), p=self.weights))
        return self.members[index]

    def snapshot(self) -> dict[str, object]:
        return {
            "schema_version": TWIN_SCHEMA_VERSION,
            "ensemble_sha256": self.ensemble_sha256,
            "members": [member.canonical() for member in self.members],
            "weights": self.weights.tolist(),
            "updates": self.updates,
        }

    @classmethod
    def restore(cls, snapshot: Mapping[str, object]) -> "VersionedTwinEnsemble":
        if snapshot.get("schema_version") != TWIN_SCHEMA_VERSION:
            raise ValueError("twin-ensemble schema mismatch")
        members = [TwinMember(**dict(item)) for item in snapshot["members"]]
        ensemble = cls(members, weights=snapshot["weights"])
        if ensemble.ensemble_sha256 != snapshot.get("ensemble_sha256"):
            raise ValueError("twin-ensemble member fingerprint mismatch")
        # Loading must not renormalize an already normalized posterior: even a
        # one-ULP change breaks deterministic replay and artifact hashes.
        stored_weights = np.asarray(snapshot["weights"], dtype=float)
        if not np.isclose(stored_weights.sum(), 1.0, rtol=0.0, atol=1e-12):
            raise ValueError("stored twin-ensemble weights do not sum to one")
        ensemble.weights = stored_weights.copy()
        ensemble.updates = int(snapshot["updates"])
        return ensemble
