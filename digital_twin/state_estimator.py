"""Small online state estimator for the v30 fast feedback loop."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import math
from typing import Mapping, Optional

from .contracts import STATE_SCHEMA_VERSION


STATE_KEYS = ("demand", "supply", "rival_quote_index", "shock_probability")


def _time(value: str | datetime) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if not isinstance(parsed, datetime) or parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("state-estimator timestamps must be timezone-aware")
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True)
class EstimatorConfig:
    process_variance_per_hour: Mapping[str, float] = field(default_factory=lambda: {
        "demand": 4.0,
        "supply": 4.0,
        "rival_quote_index": 0.01,
        "shock_probability": 0.005,
    })
    measurement_variance: Mapping[str, float] = field(default_factory=lambda: {
        "demand": 25.0,
        "supply": 25.0,
        "rival_quote_index": 0.04,
        "shock_probability": 0.02,
    })
    trend_alpha: float = 0.10


@dataclass(frozen=True)
class LatentMarketState:
    event_time: str
    means: dict[str, float]
    variances: dict[str, float]
    trends_per_hour: dict[str, float]
    observed: tuple[str, ...]
    schema_version: str = STATE_SCHEMA_VERSION

    @property
    def aggregate_uncertainty(self) -> float:
        normalized = []
        for key in STATE_KEYS:
            scale = max(abs(self.means[key]), 1.0)
            normalized.append(math.sqrt(max(self.variances[key], 0.0)) / scale)
        return math.sqrt(sum(value * value for value in normalized) / len(normalized))


class OnlineStateEstimator:
    """Independent scalar Kalman filters plus a smoothed local trend."""

    def __init__(
        self,
        *,
        config: Optional[EstimatorConfig] = None,
        initial_means: Optional[Mapping[str, float]] = None,
        initial_variances: Optional[Mapping[str, float]] = None,
    ):
        self.config = config or EstimatorConfig()
        self.means = {key: float((initial_means or {}).get(key, 0.0)) for key in STATE_KEYS}
        self.variances = {key: float((initial_variances or {}).get(key, 100.0)) for key in STATE_KEYS}
        self.trends = {key: 0.0 for key in STATE_KEYS}
        self.last_time: Optional[datetime] = None
        self.last_observed = tuple()
        self._validate_config()

    def _validate_config(self) -> None:
        for name, values in (
            ("process", self.config.process_variance_per_hour),
            ("measurement", self.config.measurement_variance),
        ):
            missing = set(STATE_KEYS) - set(values)
            if missing or any(float(values[key]) < 0 or not math.isfinite(float(values[key])) for key in STATE_KEYS):
                raise ValueError(f"invalid {name} variances; missing={sorted(missing)}")
        if not 0.0 <= self.config.trend_alpha <= 1.0:
            raise ValueError("trend_alpha must be in [0, 1]")

    def _elapsed_hours(self, event_time: datetime) -> float:
        if self.last_time is None:
            return 0.0
        delta = (event_time - self.last_time).total_seconds() / 3600.0
        if delta < 0:
            raise ValueError("out-of-order state update")
        return delta

    @staticmethod
    def _domain(key: str, value: float) -> float:
        """Validate an externally observed state value."""
        if not math.isfinite(value):
            raise ValueError(f"non-finite measurement for {key}")
        if key in {"demand", "supply", "rival_quote_index"} and value < 0:
            raise ValueError(f"{key} must be non-negative")
        if key == "shock_probability" and not 0.0 <= value <= 1.0:
            raise ValueError("shock_probability must be in [0, 1]")
        return value

    @staticmethod
    def _project_state(key: str, value: float) -> float:
        """Project an internal trend prediction onto its natural domain."""
        if not math.isfinite(value):
            raise ValueError(f"non-finite state prediction for {key}")
        if key == "shock_probability":
            return min(1.0, max(0.0, value))
        if key in {"demand", "supply", "rival_quote_index"}:
            return max(0.0, value)
        raise KeyError(f"unknown latent-state key: {key}")

    def update(
        self,
        event_time: str | datetime,
        measurements: Mapping[str, Optional[float]],
        *,
        measurement_variance: Optional[Mapping[str, float]] = None,
    ) -> LatentMarketState:
        when = _time(event_time)
        elapsed = self._elapsed_hours(when)
        predicted = {
            key: self._project_state(
                key, self.means[key] + self.trends[key] * elapsed
            )
            for key in STATE_KEYS
        }
        predicted_variance = {
            key: self.variances[key]
            + float(self.config.process_variance_per_hour[key]) * elapsed
            for key in STATE_KEYS
        }
        observed = []
        for key in STATE_KEYS:
            raw = measurements.get(key)
            if raw is None:
                self.means[key] = predicted[key]
                self.variances[key] = predicted_variance[key]
                continue
            observation = self._domain(key, float(raw))
            variance = float(
                (measurement_variance or {}).get(key, self.config.measurement_variance[key])
            )
            if variance <= 0 or not math.isfinite(variance):
                raise ValueError("measurement variance must be finite and positive")
            gain = predicted_variance[key] / (predicted_variance[key] + variance)
            innovation = observation - predicted[key]
            posterior = predicted[key] + gain * innovation
            self.means[key] = self._project_state(key, posterior)
            self.variances[key] = max(0.0, (1.0 - gain) * predicted_variance[key])
            if elapsed > 0:
                observed_trend = innovation / elapsed
                alpha = self.config.trend_alpha
                self.trends[key] = (1.0 - alpha) * self.trends[key] + alpha * observed_trend
            observed.append(key)
        self.last_time = when
        self.last_observed = tuple(observed)
        return self.state()

    def forecast(self, hours: float) -> dict[str, float]:
        horizon = float(hours)
        if horizon < 0 or not math.isfinite(horizon):
            raise ValueError("forecast horizon must be finite and non-negative")
        result = {}
        for key in STATE_KEYS:
            value = self.means[key] + self.trends[key] * horizon
            result[key] = self._project_state(key, value)
        return result

    def state(self) -> LatentMarketState:
        event_time = (
            self.last_time.isoformat().replace("+00:00", "Z")
            if self.last_time is not None else "1970-01-01T00:00:00Z"
        )
        return LatentMarketState(
            event_time=event_time,
            means=dict(self.means), variances=dict(self.variances),
            trends_per_hour=dict(self.trends), observed=self.last_observed,
        )

    def snapshot(self) -> dict[str, object]:
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "config": asdict(self.config),
            "state": asdict(self.state()),
        }

    @classmethod
    def restore(cls, snapshot: Mapping[str, object]) -> "OnlineStateEstimator":
        if snapshot.get("schema_version") != STATE_SCHEMA_VERSION:
            raise ValueError("state-estimator schema mismatch")
        config = EstimatorConfig(**dict(snapshot["config"]))
        state = dict(snapshot["state"])
        estimator = cls(
            config=config, initial_means=state["means"], initial_variances=state["variances"]
        )
        estimator.trends = {key: float(value) for key, value in dict(state["trends_per_hour"]).items()}
        estimator.last_time = _time(str(state["event_time"]))
        estimator.last_observed = tuple(state.get("observed", ()))
        return estimator
