"""Production/simulator observation parity for the concise v30 MDP."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping, Sequence

import numpy as np

from platform_mdp import PlatformObservationModel
from .contracts import OBSERVATION_SCHEMA_VERSION


V30_CLOSED_LOOP_FEATURES = (
    "forecast_demand_log_ratio",
    "forecast_supply_log_ratio",
    "state_uncertainty_log1p",
    "data_age_log1p",
)
V30_EPISODE_FEATURES = ("episode_remaining_fraction", "episode_horizon_log1p")


@dataclass(frozen=True)
class ClosedLoopFeatures:
    demand_now: float
    demand_forecast: float
    supply_now: float
    supply_forecast: float
    state_uncertainty: float
    data_age_seconds: float
    decision_interval_seconds: float
    episode_steps_remaining: int
    episode_horizon_steps: int

    def encoded(self) -> tuple[float, ...]:
        values = (
            self.demand_now, self.demand_forecast, self.supply_now,
            self.supply_forecast, self.state_uncertainty,
            self.data_age_seconds, self.decision_interval_seconds,
            self.episode_steps_remaining, self.episode_horizon_steps,
        )
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError("closed-loop observation inputs must be finite")
        if min(self.demand_now, self.demand_forecast, self.supply_now, self.supply_forecast) < 0:
            raise ValueError("demand and supply values must be non-negative")
        if self.state_uncertainty < 0 or self.data_age_seconds < 0 or self.decision_interval_seconds <= 0:
            raise ValueError("uncertainty/age must be non-negative and interval positive")
        if self.episode_horizon_steps <= 0 or not 0 <= self.episode_steps_remaining <= self.episode_horizon_steps:
            raise ValueError("episode clock must have a positive horizon and valid remaining steps")
        epsilon = 1e-9
        return (
            math.log((self.demand_forecast + epsilon) / (self.demand_now + epsilon)),
            math.log((self.supply_forecast + epsilon) / (self.supply_now + epsilon)),
            math.log1p(self.state_uncertainty),
            math.log1p(self.data_age_seconds / self.decision_interval_seconds),
            self.episode_steps_remaining / self.episode_horizon_steps,
            math.log1p(self.episode_horizon_steps),
        )


class V30ObservationContract:
    """One ordered observation schema shared by simulation and serving."""

    def __init__(self, *, shock_features: bool = True):
        self.shock_features = bool(shock_features)
        self.base_names = PlatformObservationModel.feature_names(
            shock_features=self.shock_features, compact_foresight=True
        )
        self.names = tuple(self.base_names) + V30_CLOSED_LOOP_FEATURES + V30_EPISODE_FEATURES
        payload = {
            "schema_version": OBSERVATION_SCHEMA_VERSION,
            "names": self.names,
            "transforms": {
                "forecast_ratios": "natural_log_unbounded",
                "uncertainty": "log1p_nonnegative",
                "data_age": "log1p_age_over_decision_interval",
                "episode_clock": "remaining_fraction_and_log1p_horizon_steps",
            },
        }
        self.schema_sha256 = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @property
    def dimension(self) -> int:
        return len(self.names)

    def _base_from_mapping(self, values: Mapping[str, float]) -> np.ndarray:
        missing = [name for name in self.base_names if name not in values]
        extra = sorted(set(values) - set(self.base_names))
        if missing or extra:
            raise ValueError(f"base observation schema mismatch; missing={missing}, extra={extra}")
        return np.asarray([float(values[name]) for name in self.base_names], dtype=np.float32)

    def _combine(self, base: np.ndarray, closed_loop: ClosedLoopFeatures) -> np.ndarray:
        if base.shape != (len(self.base_names),):
            raise ValueError(
                f"base observation width mismatch: expected={len(self.base_names)}, got={base.shape}"
            )
        result = np.concatenate(
            [base.astype(np.float32, copy=False), np.asarray(closed_loop.encoded(), dtype=np.float32)]
        )
        if result.shape != (self.dimension,) or not np.all(np.isfinite(result)):
            raise ValueError("v30 observation must be finite and match the registered width")
        return result

    def from_simulator(
        self, base_vector: Sequence[float] | np.ndarray, closed_loop: ClosedLoopFeatures
    ) -> np.ndarray:
        return self._combine(np.asarray(base_vector, dtype=np.float32), closed_loop)

    def from_production(
        self, base_values: Mapping[str, float], closed_loop: ClosedLoopFeatures
    ) -> np.ndarray:
        return self._combine(self._base_from_mapping(base_values), closed_loop)

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": OBSERVATION_SCHEMA_VERSION,
            "schema_sha256": self.schema_sha256,
            "dimension": self.dimension,
            "feature_names": list(self.names),
            "base_contract": "v29_compact_foresight_preserved",
            "closed_loop_features": list(V30_CLOSED_LOOP_FEATURES),
            "episode_features": list(V30_EPISODE_FEATURES),
        }
