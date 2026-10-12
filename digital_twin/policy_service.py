"""Replayable, audit-first shadow inference for v30 policies.

The service deliberately does not mutate a live tariff.  It produces a
recommendation, records the exact behavior distribution and the counterfactual
tariff, and leaves deployment to a separately authorized adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from typing import Callable, Mapping, Sequence

import numpy as np

from .ledger import DecisionEvent, EventLedger
from .observation import ClosedLoopFeatures, V30ObservationContract


ProbabilityPolicy = Callable[[np.ndarray], Sequence[float]]
TariffTransition = Callable[[str, Mapping[str, float]], Mapping[str, float]]


@dataclass(frozen=True)
class ShadowPolicy:
    policy_id: str
    action_names: tuple[str, ...]
    probability_policy: ProbabilityPolicy
    observation_schema_sha256: str

    def probabilities(self, observation: np.ndarray) -> dict[str, float]:
        raw = np.asarray(self.probability_policy(observation.copy()), dtype=np.float64)
        if raw.shape != (len(self.action_names),):
            raise ValueError("policy probability width does not match action names")
        if not np.all(np.isfinite(raw)) or np.any(raw < 0.0):
            raise ValueError("policy probabilities must be finite and non-negative")
        total = float(raw.sum())
        if total <= 0.0:
            raise ValueError("policy probability mass must be positive")
        normalized = raw / total
        return {name: float(normalized[index]) for index, name in enumerate(self.action_names)}


@dataclass(frozen=True)
class ShadowRecommendation:
    policy_id: str
    action_name: str
    action_probabilities: Mapping[str, float]
    tariff_before: Mapping[str, float]
    tariff_after: Mapping[str, float]
    observation_schema_sha256: str
    ledger_inserted: bool


class ShadowPolicyService:
    """Samples the learned policy, with idempotent per-event random draws."""

    def __init__(
        self,
        *,
        policy: ShadowPolicy,
        observation_contract: V30ObservationContract,
        ledger: EventLedger,
        tariff_transition: TariffTransition,
        sampling_seed: int = 0,
    ):
        if policy.observation_schema_sha256 != observation_contract.schema_sha256:
            raise ValueError("policy and serving observation schemas do not match")
        if not policy.policy_id.strip() or not policy.action_names:
            raise ValueError("shadow policy requires an id and actions")
        if len(set(policy.action_names)) != len(policy.action_names):
            raise ValueError("shadow action names must be unique")
        self.policy = policy
        self.observation_contract = observation_contract
        self.ledger = ledger
        self.tariff_transition = tariff_transition
        self.sampling_seed = int(sampling_seed)

    def recommend(
        self,
        *,
        event_id: str,
        request_id: str,
        event_time: str | datetime,
        market: str,
        base_observation: Mapping[str, float],
        closed_loop: ClosedLoopFeatures,
        tariff_before: Mapping[str, float],
        context: Mapping[str, object] | None = None,
    ) -> ShadowRecommendation:
        observation = self.observation_contract.from_production(base_observation, closed_loop)
        probabilities = self.policy.probabilities(observation)
        # Argmax would execute a different incremental policy and falsely log
        # its softmax as the behavior propensity. Key the draw to the event so
        # retries reproduce the recommendation without consuming a new draw.
        draw_key = json.dumps([
            self.sampling_seed, self.policy.policy_id, event_id, request_id,
        ]).encode()
        draw_seed = int.from_bytes(hashlib.sha256(draw_key).digest()[:8], "big")
        action_name = str(np.random.default_rng(draw_seed).choice(
            self.policy.action_names,
            p=[probabilities[name] for name in self.policy.action_names],
        ))
        tariff_after = {
            str(key): float(value)
            for key, value in self.tariff_transition(action_name, tariff_before).items()
        }
        if set(tariff_after) != set(tariff_before):
            raise ValueError("tariff transition must preserve the registered coefficient keys")
        if any(not math.isfinite(value) for value in tariff_after.values()):
            raise ValueError("shadow tariff must be finite")
        observation_payload = {
            name: float(observation[index])
            for index, name in enumerate(self.observation_contract.names)
        }
        inserted = self.ledger.append_decision(
            DecisionEvent(
                event_id=event_id,
                request_id=request_id,
                event_time=event_time,
                market=market,
                policy_id=self.policy.policy_id,
                observation_schema=self.observation_contract.schema_sha256,
                observation=observation_payload,
                action={"name": action_name, "shadow_only": True},
                action_probabilities=probabilities,
                tariff_before=tariff_before,
                tariff_after=tariff_after,
                context={
                    **dict(context or {}),
                    "serving_mode": "shadow",
                    "policy_mode": "sample_raw",
                    "sampling_seed": self.sampling_seed,
                    "deployment_authorized": False,
                },
            )
        )
        return ShadowRecommendation(
            policy_id=self.policy.policy_id,
            action_name=action_name,
            action_probabilities=probabilities,
            tariff_before=dict(tariff_before),
            tariff_after=tariff_after,
            observation_schema_sha256=self.observation_contract.schema_sha256,
            ledger_inserted=inserted,
        )
