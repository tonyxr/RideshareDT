from __future__ import annotations

"""Platform-facing MDP and constrained-control primitives.

This module deliberately sits *outside* the market simulator.  The simulator may
retain complete ground truth, while a firm receives only operational telemetry
and noisy/delayed competitor quote probes that a real ride-hailing platform
could plausibly collect.  Reward is non-negative business utility.  Gap,
service, margin, and intervention stability are separate soft costs used by a
constrained optimizer; no action identity is rewarded or penalized directly.
"""

from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Deque, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


SEGMENTS: Tuple[str, ...] = ("0_2", "2_5", "5_10", "10_plus")


@dataclass(frozen=True)
class ObservationConfig:
    telemetry_delay_steps: int = 1
    quote_probe_interval_steps: int = 3
    quote_probe_delay_steps: int = 1
    quote_noise_dollars: float = 0.18
    quote_missing_probability: float = 0.03
    market_share_noise: float = 0.015
    demand_mix_noise: float = 0.02
    max_quote_age_steps: int = 24
    gap_scale_dollars: float = 4.0
    revenue_scale: float = 25.0
    profit_scale: float = 10.0
    wait_scale_minutes: float = 15.0
    driver_pay_scale: float = 20.0
    driver_earnings_scale: float = 80.0


@dataclass(frozen=True)
class PositiveRewardConfig:
    """Configuration for the stationary, positive business objective.

    All four terms are own-platform outcome levels rather than changes,
    competitor-relative bonuses, or action-dependent shaping.  The weights may
    be any non-negative finite values and are normalized before use, making CLI
    tuning intuitive without changing the reward's [0, 1] scale.
    """

    profit_weight: float = 0.38
    revenue_weight: float = 0.22
    completed_demand_weight: float = 0.20
    service_weight: float = 0.20
    revenue_scale: float = 25.0
    profit_scale: float = 10.0
    completed_share_target: float = 0.42
    wait_target_minutes: float = 7.0
    minimum_reward: float = 1e-4

    def __post_init__(self) -> None:
        weights = np.asarray(
            [
                self.profit_weight,
                self.revenue_weight,
                self.completed_demand_weight,
                self.service_weight,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(weights)):
            raise ValueError("positive reward weights must all be finite")
        if np.any(weights < 0.0):
            raise ValueError("positive reward weights must all be non-negative")
        if float(weights.sum()) <= 0.0:
            raise ValueError("at least one positive reward weight must be greater than zero")

    def normalized_weights(self) -> np.ndarray:
        weights = np.asarray(
            [
                self.profit_weight,
                self.revenue_weight,
                self.completed_demand_weight,
                self.service_weight,
            ],
            dtype=float,
        )
        return weights / float(weights.sum())


@dataclass(frozen=True)
class ConstraintConfig:
    target_gap: float = 0.75
    overall_tolerance: float = 0.45
    # Long-trip gaps are at least as important as short-trip gaps because a
    # small per-mile error compounds with distance.  Short trips get slightly
    # more room for fixed-fee and quote-probe noise.
    segment_tolerances: Tuple[float, float, float, float] = (0.65, 0.60, 0.55, 0.55)
    gap_softness: float = 0.50
    fulfillment_floor: float = 0.78
    wait_limit_minutes: float = 8.0
    margin_floor: float = 0.10
    intervention_rate_budget: float = 0.35
    reversal_horizon: int = 4
    oscillation_window: int = 12
    multiplier_lr: float = 0.035
    multiplier_max: float = 1.5
    cost_ema_alpha: float = 0.10
    cost_budgets: Tuple[float, ...] = (
        0.01, 0.01,  # aggregate overprice / underprice
        0.02, 0.02, 0.025, 0.03,  # distance segments
        0.01, 0.01, 0.01, 0.03,  # fulfillment, wait, margin, oscillation
    )


@dataclass(frozen=True)
class TrainingStage:
    name: str
    start: float
    end: float
    opponent_cadence_multiplier: int
    freeze_opponent: bool
    action_hold_steps: int
    constraint_scale: float
    exploration_rate: float
    entropy_scale: float
    learning_rate_scale: float
    episode_days: int
    opponent_pool_size: int
    tariff_reset_fraction: float


@dataclass
class OperationalClock:
    """Shared decision clock used by training, validation, and evaluation."""

    period: int = 0

    def due(self, interval_steps: int) -> bool:
        return self.period % max(1, int(interval_steps)) == 0

    def advance(self) -> None:
        self.period += 1

    def reset(self) -> None:
        self.period = 0


class TrainingStageScheduler:
    """Curriculum for learning control before full strategic competition."""

    def __init__(self, mode: str = "staged") -> None:
        mode = str(mode or "staged").strip().lower()
        if mode not in {"staged", "direct"}:
            raise ValueError("training curriculum must be 'staged' or 'direct'")
        self.mode = mode
        if mode == "direct":
            self.stages = (
                TrainingStage(
                    "direct", 0.0, 1.0, 1, False, 1, 1.0, 0.08, 0.80, 1.0,
                    64, 8, 0.20,
                ),
            )
        else:
            self.stages = (
                TrainingStage(
                    "foundation", 0.0, 0.18, 1, True, 2, 0.35, 0.18, 1.00, 1.00,
                    48, 1, 0.12,
                ),
                TrainingStage(
                    "robustness", 0.18, 0.48, 2, False, 2, 0.70, 0.13, 0.95, 1.00,
                    64, 4, 0.18,
                ),
                TrainingStage(
                    "competition", 0.48, 0.85, 1, False, 1, 1.00, 0.08, 0.80, 0.90,
                    72, 8, 0.22,
                ),
                TrainingStage(
                    "consolidation", 0.85, 1.000001, 1, False, 2, 1.10, 0.04, 0.65, 0.70,
                    64, 6, 0.16,
                ),
            )

    def stage_at(self, progress: float) -> TrainingStage:
        p = float(np.clip(progress, 0.0, 1.0))
        for stage in self.stages:
            if stage.start <= p < stage.end:
                return stage
        return self.stages[-1]

    def as_dict(self) -> Dict[str, Any]:
        return {"mode": self.mode, "stages": [asdict(stage) for stage in self.stages]}


class PlatformObservationModel:
    """Noisy/delayed observation channel for one platform.

    Ground-truth rider thresholds, competitor coefficients, competitor profit,
    and exact current segment gaps never appear in the observation.  Segment
    gaps become visible only as sampled public-quote estimates with uncertainty,
    age, and missingness.
    """

    TELEMETRY_KEYS: Tuple[str, ...] = (
        "chosen_share_estimate",
        "completed_share_estimate",
        "revenue_per_request",
        "profit_per_request",
        "fulfillment_rate",
        "acceptance_rate",
        "wait_minutes",
        "driver_pay_per_request",
        "idle_driver_share",
        "utilization",
        "driver_earnings_per_hour",
        "telemetry_age",
    )
    DEMAND_KEYS: Tuple[str, ...] = (
        "distance_mean",
        "distance_std",
        "distance_q25",
        "distance_q75",
        "duration_mean",
        "duration_std",
        "airport_rate",
        "long_trip_share",
    )
    observation_dim: int = 61
    action_feature_dim: int = 20

    def __init__(self, seed: int, config: Optional[ObservationConfig] = None) -> None:
        self.config = config or ObservationConfig()
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.reset()

    def reset(self) -> None:
        self.step = 0
        self._telemetry_queue: Deque[Dict[str, float]] = deque()
        self._probe_queue: Deque[Dict[str, Dict[str, float]]] = deque()
        self.telemetry = {key: 0.0 for key in self.TELEMETRY_KEYS}
        self.telemetry.update({"fulfillment_rate": 1.0, "acceptance_rate": 1.0, "idle_driver_share": 1.0})
        self.previous_telemetry = dict(self.telemetry)
        self.demand_mix = {key: 0.0 for key in self.DEMAND_KEYS}
        self.demand_mix.update({"distance_mean": 4.0, "distance_q25": 2.0, "distance_q75": 7.0})
        self.quote_probes = {
            segment: {"gap": 0.0, "uncertainty": 1.0, "age": float(self.config.max_quote_age_steps), "available": 0.0}
            for segment in SEGMENTS
        }
        self._latest_true_gaps = {segment: 0.0 for segment in SEGMENTS}

    def snapshot(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "rng_state": self.rng.bit_generator.state,
            "telemetry_queue": list(self._telemetry_queue),
            "probe_queue": list(self._probe_queue),
            "telemetry": dict(self.telemetry),
            "previous_telemetry": dict(self.previous_telemetry),
            "demand_mix": dict(self.demand_mix),
            "quote_probes": {k: dict(v) for k, v in self.quote_probes.items()},
            "latest_true_gaps": dict(self._latest_true_gaps),
        }

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        self.step = int(snapshot.get("step", 0))
        self.rng = np.random.default_rng(self.seed)
        if snapshot.get("rng_state") is not None:
            self.rng.bit_generator.state = snapshot["rng_state"]
        self._telemetry_queue = deque(dict(x) for x in snapshot.get("telemetry_queue", []))
        self._probe_queue = deque(
            {k: dict(v) for k, v in x.items()} for x in snapshot.get("probe_queue", [])
        )
        self.telemetry = dict(snapshot.get("telemetry", self.telemetry))
        self.previous_telemetry = dict(snapshot.get("previous_telemetry", self.previous_telemetry))
        self.demand_mix = dict(snapshot.get("demand_mix", self.demand_mix))
        self.quote_probes = {k: dict(v) for k, v in snapshot.get("quote_probes", self.quote_probes).items()}
        self._latest_true_gaps = dict(snapshot.get("latest_true_gaps", self._latest_true_gaps))

    @staticmethod
    def _bounded_noise(rng: np.random.Generator, scale: float, bound: float) -> float:
        return float(np.clip(rng.normal(0.0, max(0.0, scale)), -bound, bound))

    def ingest(
        self,
        *,
        own_metrics: Mapping[str, float],
        supply_metrics: Optional[Mapping[str, float]],
        crowd_stats: Mapping[str, float],
        gap_sign: float = 1.0,
    ) -> None:
        """Ingest simulator truth into delayed/noisy platform measurement queues."""
        self.step += 1
        supply = supply_metrics or {}
        share_noise = self._bounded_noise(self.rng, self.config.market_share_noise, 0.08)
        telemetry = {
            "chosen_share_estimate": float(np.clip(float(own_metrics.get("chosen_share", 0.0)) + share_noise, 0.0, 1.0)),
            "completed_share_estimate": float(np.clip(float(own_metrics.get("completed_share", 0.0)) + 0.75 * share_noise, 0.0, 1.0)),
            "revenue_per_request": float(own_metrics.get("revenue_per_request", 0.0)),
            "profit_per_request": float(own_metrics.get("profit_per_request", 0.0)),
            "fulfillment_rate": float(np.clip(own_metrics.get("fulfillment_rate", 1.0), 0.0, 1.0)),
            "acceptance_rate": float(np.clip(own_metrics.get("acceptance_rate", 1.0), 0.0, 1.0)),
            "wait_minutes": float(max(0.0, own_metrics.get("wait_minutes", 0.0))),
            "driver_pay_per_request": float(max(0.0, own_metrics.get("driver_pay_per_request", 0.0))),
            "idle_driver_share": float(np.clip(supply.get("idle_driver_share", 0.0), 0.0, 1.0)),
            "utilization": float(np.clip(supply.get("utilization", 0.0), 0.0, 1.0)),
            "driver_earnings_per_hour": float(max(0.0, supply.get("driver_earnings_per_hour", 0.0))),
            "telemetry_age": 0.0,
        }
        self._telemetry_queue.append(telemetry)
        delay = max(0, int(self.config.telemetry_delay_steps))
        if len(self._telemetry_queue) > delay:
            self.previous_telemetry = dict(self.telemetry)
            self.telemetry = self._telemetry_queue.popleft()
        else:
            self.telemetry["telemetry_age"] = float(self.telemetry.get("telemetry_age", 0.0) + 1.0)

        mix_noise = self.config.demand_mix_noise
        self.demand_mix = {
            "distance_mean": max(0.0, float(crowd_stats.get("distance_mean", 4.0)) * (1.0 + self._bounded_noise(self.rng, mix_noise, 0.10))),
            "distance_std": max(0.0, float(crowd_stats.get("distance_std", 0.0)) * (1.0 + self._bounded_noise(self.rng, mix_noise, 0.10))),
            "distance_q25": max(0.0, float(crowd_stats.get("distance_q25", 2.0))),
            "distance_q75": max(0.0, float(crowd_stats.get("distance_q75", 7.0))),
            "duration_mean": max(0.0, float(crowd_stats.get("duration_mean", 0.0))),
            "duration_std": max(0.0, float(crowd_stats.get("duration_std", 0.0))),
            "airport_rate": float(np.clip(crowd_stats.get("airport_rate", 0.0), 0.0, 1.0)),
            "long_trip_share": float(np.clip(crowd_stats.get("long_trip_share", 0.0), 0.0, 1.0)),
        }

        for probe in self.quote_probes.values():
            probe["age"] = float(min(self.config.max_quote_age_steps, probe.get("age", 0.0) + 1.0))

        if self.step == 1 or self.step % max(1, int(self.config.quote_probe_interval_steps)) == 0:
            sampled: Dict[str, Dict[str, float]] = {}
            for segment in SEGMENTS:
                raw = float(gap_sign) * float(crowd_stats.get(f"distance_bin_{segment}_price_gap_mean", 0.0))
                self._latest_true_gaps[segment] = raw
                available = float(self.rng.random() >= self.config.quote_missing_probability)
                uncertainty = float(self.config.quote_noise_dollars * (1.0 + 0.40 * self.rng.random()))
                estimate = raw + self._bounded_noise(self.rng, uncertainty, 3.0 * uncertainty) if available else 0.0
                sampled[segment] = {
                    "gap": float(estimate),
                    "uncertainty": uncertainty,
                    "age": 0.0,
                    "available": available,
                }
            self._probe_queue.append(sampled)
        probe_delay = max(0, int(self.config.quote_probe_delay_steps))
        if len(self._probe_queue) > probe_delay:
            released = self._probe_queue.popleft()
            for segment, probe in released.items():
                if probe.get("available", 0.0) > 0.0:
                    self.quote_probes[segment] = dict(probe)
                else:
                    self.quote_probes[segment]["available"] = 0.0

    def quote_snapshot(self) -> Dict[str, Dict[str, float]]:
        return {segment: dict(values) for segment, values in self.quote_probes.items()}

    def build_observation(
        self,
        *,
        hour: int,
        day_of_week: int,
        weather: str,
        own_coefficients: Mapping[str, float],
        anchor_coefficients: Mapping[str, float],
        last_action: Optional[Mapping[str, float]] = None,
    ) -> np.ndarray:
        hour_f = float(int(hour) % 24)
        day_f = float(int(day_of_week) % 7)
        weather_code = {"clear": 0.0, "cloudy": 0.33, "rain": 0.66, "snow": 1.0}.get(str(weather).lower(), 0.0)
        time_features = [
            np.sin(2.0 * np.pi * hour_f / 24.0),
            np.cos(2.0 * np.pi * hour_f / 24.0),
            np.sin(2.0 * np.pi * day_f / 7.0),
            np.cos(2.0 * np.pi * day_f / 7.0),
            float(day_f >= 5),
            float((7 <= hour_f < 10) or (16 <= hour_f < 19)),
            float(hour_f < 6 or hour_f >= 22),
            weather_code,
        ]
        coefficient_features = []
        for key in ("base_fare", "per_minute", "per_mile", "booking_fee", "airport_fee"):
            anchor = float(anchor_coefficients.get(key, 1.0))
            coefficient_features.append(float(np.clip((float(own_coefficients.get(key, anchor)) - anchor) / max(abs(anchor), 1e-6), -1.0, 1.0)))

        c = self.config
        t = self.telemetry
        telemetry_features = [
            float(np.clip(t.get("chosen_share_estimate", 0.0), 0.0, 1.0)),
            float(np.clip(t.get("completed_share_estimate", 0.0), 0.0, 1.0)),
            float(np.clip(t.get("revenue_per_request", 0.0) / c.revenue_scale, 0.0, 1.5)),
            float(np.clip(t.get("profit_per_request", 0.0) / c.profit_scale, -1.0, 1.0)),
            float(np.clip(t.get("fulfillment_rate", 1.0), 0.0, 1.0)),
            float(np.clip(t.get("acceptance_rate", 1.0), 0.0, 1.0)),
            float(np.clip(t.get("wait_minutes", 0.0) / c.wait_scale_minutes, 0.0, 1.5)),
            float(np.clip(t.get("driver_pay_per_request", 0.0) / c.driver_pay_scale, 0.0, 1.5)),
            float(np.clip(t.get("idle_driver_share", 0.0), 0.0, 1.0)),
            float(np.clip(t.get("utilization", 0.0), 0.0, 1.0)),
            float(np.clip(t.get("driver_earnings_per_hour", 0.0) / c.driver_earnings_scale, 0.0, 1.5)),
            float(np.clip(t.get("telemetry_age", 0.0) / max(1, c.max_quote_age_steps), 0.0, 1.0)),
        ]
        p = self.previous_telemetry
        trend_features = [
            float(np.clip((t.get("chosen_share_estimate", 0.0) - p.get("chosen_share_estimate", 0.0)) / 0.15, -1.0, 1.0)),
            float(np.clip((t.get("completed_share_estimate", 0.0) - p.get("completed_share_estimate", 0.0)) / 0.15, -1.0, 1.0)),
            float(np.clip((t.get("revenue_per_request", 0.0) - p.get("revenue_per_request", 0.0)) / c.revenue_scale, -1.0, 1.0)),
            float(np.clip((t.get("profit_per_request", 0.0) - p.get("profit_per_request", 0.0)) / c.profit_scale, -1.0, 1.0)),
            float(np.clip((t.get("fulfillment_rate", 1.0) - p.get("fulfillment_rate", 1.0)) / 0.25, -1.0, 1.0)),
            float(np.clip((t.get("wait_minutes", 0.0) - p.get("wait_minutes", 0.0)) / c.wait_scale_minutes, -1.0, 1.0)),
        ]
        d = self.demand_mix
        demand_features = [
            float(np.clip(d.get("distance_mean", 0.0) / 12.0, 0.0, 1.5)),
            float(np.clip(d.get("distance_std", 0.0) / 8.0, 0.0, 1.5)),
            float(np.clip(d.get("distance_q25", 0.0) / 12.0, 0.0, 1.5)),
            float(np.clip(d.get("distance_q75", 0.0) / 16.0, 0.0, 1.5)),
            float(np.clip(d.get("duration_mean", 0.0) / 45.0, 0.0, 1.5)),
            float(np.clip(d.get("duration_std", 0.0) / 30.0, 0.0, 1.5)),
            float(np.clip(d.get("airport_rate", 0.0), 0.0, 1.0)),
            float(np.clip(d.get("long_trip_share", 0.0), 0.0, 1.0)),
        ]
        probe_features = []
        for segment in SEGMENTS:
            probe = self.quote_probes[segment]
            probe_features.extend([
                float(np.clip((probe.get("gap", 0.0)) / c.gap_scale_dollars, -1.5, 1.5)),
                float(np.clip(probe.get("uncertainty", 1.0) / c.gap_scale_dollars, 0.0, 1.0)),
                float(np.clip(probe.get("age", c.max_quote_age_steps) / max(1, c.max_quote_age_steps), 0.0, 1.0)),
                float(np.clip(probe.get("available", 0.0), 0.0, 1.0)),
            ])
        action = last_action or {}
        action_features = [
            float(np.clip(action.get("direction", 0.0), -1.0, 1.0)),
            float(np.clip(action.get("target_index", 0.0), 0.0, 1.0)),
            float(np.clip(action.get("magnitude", 0.0) / 2.0, 0.0, 1.0)),
            float(np.clip(action.get("reversal", 0.0), 0.0, 1.0)),
            float(np.clip(action.get("recent_intervention_rate", 0.0), 0.0, 1.0)),
            float(np.clip(action.get("time_since_intervention", 1.0), 0.0, 1.0)),
        ]
        result = np.asarray(
            time_features
            + coefficient_features
            + telemetry_features
            + trend_features
            + demand_features
            + probe_features
            + action_features,
            dtype=np.float32,
        )
        if result.size != self.observation_dim:
            raise RuntimeError(f"platform observation dimension mismatch: {result.size} != {self.observation_dim}")
        return np.nan_to_num(result, nan=0.0, posinf=1.5, neginf=-1.5)

    def build_action_features(
        self,
        *,
        action_steps: Mapping[int, Mapping[str, int]],
        action_keys: Sequence[str],
        own_coefficients: Mapping[str, float],
        anchor_coefficients: Mapping[str, float],
        coefficient_steps: Mapping[str, float],
        coefficient_bounds: Mapping[str, Tuple[float, float]],
        step_scale: float,
        target_gap: float,
    ) -> np.ndarray:
        representative = {
            "0_2": {"base_fare": 1.0, "per_minute": 8.0, "per_mile": 1.5, "booking_fee": 1.0, "airport_fee": 0.0},
            "2_5": {"base_fare": 1.0, "per_minute": 14.0, "per_mile": 3.5, "booking_fee": 1.0, "airport_fee": 0.05},
            "5_10": {"base_fare": 1.0, "per_minute": 24.0, "per_mile": 7.0, "booking_fee": 1.0, "airport_fee": 0.15},
            "10_plus": {"base_fare": 1.0, "per_minute": 38.0, "per_mile": 13.0, "booking_fee": 1.0, "airport_fee": 0.30},
        }
        rows = []
        for action_id in range(len(action_steps)):
            mapping = dict(action_steps.get(action_id, {}))
            active = [(k, int(v)) for k, v in mapping.items() if int(v) != 0 and k in action_keys]
            if not active:
                rows.append([1.0, 0.0, *([0.0] * 5), 0.0, 1.0, 1.0, *([0.0] * 4), *([0.0] * 4), 1.0, 0.0])
                continue
            all_keys = ("base_fare", "per_minute", "per_mile", "booking_fee", "airport_fee")
            direction_by_key = {key: int(direction) for key, direction in active}
            # Signed lever membership preserves the identity of mixed-direction
            # short/long rebalancing actions.
            key_signature = [float(direction_by_key.get(key, 0)) for key in all_keys]
            global_direction = float(np.clip(np.mean([direction for _, direction in active]), -1.0, 1.0))
            relative_deviations = []
            lower_distances = []
            upper_distances = []
            delta_by_key: Dict[str, float] = {}
            for key, direction in active:
                anchor = float(anchor_coefficients[key])
                current = float(own_coefficients[key])
                lb, ub = coefficient_bounds[key]
                width = max(1e-6, float(ub - lb))
                relative_deviations.append(
                    float(np.clip((current - anchor) / max(abs(anchor), 1e-6), -1.0, 1.0))
                )
                lower_distances.append(float(np.clip((current - lb) / width, 0.0, 1.0)))
                upper_distances.append(float(np.clip((ub - current) / width, 0.0, 1.0)))
                delta_by_key[key] = float(direction) * float(coefficient_steps[key]) * float(step_scale)
            impacts = [
                float(sum(delta_by_key[key] * representative[segment][key] for key, _ in active))
                for segment in SEGMENTS
            ]
            improvements = []
            uncertainty_values = []
            for segment, impact in zip(SEGMENTS, impacts):
                probe = self.quote_probes[segment]
                estimate = float(probe.get("gap", target_gap))
                before = abs(estimate - target_gap)
                after = abs((estimate - impact) - target_gap)
                improvements.append(float(np.clip((before - after) / max(self.config.gap_scale_dollars, 1e-6), -1.0, 1.0)))
                uncertainty_values.append(float(probe.get("uncertainty", self.config.gap_scale_dollars)))
            rows.append([
                0.0,
                global_direction,
                *key_signature,
                float(np.mean(relative_deviations)),
                float(np.min(lower_distances)),
                float(np.min(upper_distances)),
                *[float(np.clip(v / 20.0, -1.0, 1.0)) for v in impacts],
                *improvements,
                float(np.clip(np.mean(uncertainty_values) / self.config.gap_scale_dollars, 0.0, 1.0)),
                float(len(active) / max(1, len(action_keys))),
            ])
        result = np.asarray(rows, dtype=np.float32)
        if result.shape[1] != self.action_feature_dim:
            raise RuntimeError(f"action feature dimension mismatch: {result.shape[1]} != {self.action_feature_dim}")
        return result


class PositiveBusinessReward:
    def __init__(self, config: Optional[PositiveRewardConfig] = None) -> None:
        self.config = config or PositiveRewardConfig()

    @staticmethod
    def _positive_saturation(value: float, scale: float) -> float:
        return float(np.clip(1.0 - np.exp(-max(0.0, float(value)) / max(1e-6, float(scale))), 0.0, 1.0))

    def compute(self, metrics: Mapping[str, float]) -> Dict[str, float]:
        c = self.config
        profit_score = self._positive_saturation(metrics.get("profit_per_request", 0.0), c.profit_scale)
        revenue_score = self._positive_saturation(metrics.get("revenue_per_request", 0.0), c.revenue_scale)
        # A hard cap at the target made all sufficiently popular tariffs look
        # identical to the actor.  Smooth saturation retains useful marginal
        # credit for additional completed demand without letting volume dominate.
        completed_score = self._positive_saturation(
            metrics.get("completed_share", 0.0),
            max(0.05, 0.70 * c.completed_share_target),
        )
        fulfillment = float(np.clip(metrics.get("fulfillment_rate", 1.0), 0.0, 1.0))
        acceptance = float(np.clip(metrics.get("acceptance_rate", 1.0), 0.0, 1.0))
        wait_score = float(np.exp(-max(0.0, metrics.get("wait_minutes", 0.0) - c.wait_target_minutes) / max(c.wait_target_minutes, 1e-6)))
        service_score = float(np.clip(0.55 * fulfillment + 0.25 * acceptance + 0.20 * wait_score, 0.0, 1.0))
        weights = c.normalized_weights()
        reward = float(np.dot(weights, [profit_score, revenue_score, completed_score, service_score]))
        reward = float(np.clip(reward, c.minimum_reward, 1.0))
        return {
            "reward": reward,
            "reward_raw": reward,
            "reward_base": reward,
            "reward_positive_profit": profit_score,
            "reward_positive_revenue": revenue_score,
            "reward_positive_completed_demand": completed_score,
            "reward_positive_service": service_score,
            "reward_weight_profit": float(weights[0]),
            "reward_weight_revenue": float(weights[1]),
            "reward_weight_completed_demand": float(weights[2]),
            "reward_weight_service": float(weights[3]),
        }


class ActionStabilityTracker:
    def __init__(self, config: ConstraintConfig) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.recent_interventions: Deque[int] = deque(maxlen=max(2, self.config.oscillation_window))
        self.last_target = "hold"
        self.last_direction = 0
        self.steps_since_intervention = self.config.reversal_horizon + 1
        self.last_reversal = 0.0
        self.last_cost = 0.0

    def snapshot(self) -> Dict[str, Any]:
        return {
            "recent_interventions": list(self.recent_interventions),
            "last_target": self.last_target,
            "last_direction": self.last_direction,
            "steps_since_intervention": self.steps_since_intervention,
            "last_reversal": self.last_reversal,
            "last_cost": self.last_cost,
        }

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        self.recent_interventions = deque(
            [int(x) for x in snapshot.get("recent_interventions", [])],
            maxlen=max(2, self.config.oscillation_window),
        )
        self.last_target = str(snapshot.get("last_target", "hold"))
        self.last_direction = int(snapshot.get("last_direction", 0))
        self.steps_since_intervention = int(snapshot.get("steps_since_intervention", self.config.reversal_horizon + 1))
        self.last_reversal = float(snapshot.get("last_reversal", 0.0))
        self.last_cost = float(snapshot.get("last_cost", 0.0))

    @staticmethod
    def _smooth_excess(excess: float, softness: float = 0.25) -> float:
        x = max(0.0, float(excess))
        return float(1.0 - np.exp(-x / max(1e-6, softness)))

    def record(self, *, action_event: bool, target: str, direction: int) -> float:
        self.steps_since_intervention += 1
        if not action_event:
            self.last_reversal = 0.0
            self.recent_interventions.append(0)
            rate = float(np.mean(self.recent_interventions)) if self.recent_interventions else 0.0
            self.last_cost = float(
                0.30 * self._smooth_excess(
                    rate - self.config.intervention_rate_budget,
                    softness=0.20,
                )
            )
            return self.last_cost
        non_hold = int(direction != 0 and str(target) != "hold")
        self.recent_interventions.append(non_hold)
        reversal = float(
            non_hold
            and self.last_target == str(target)
            and self.last_direction == -int(direction)
            and self.steps_since_intervention <= self.config.reversal_horizon
        )
        if non_hold:
            self.last_target = str(target)
            self.last_direction = int(direction)
            self.steps_since_intervention = 0
        rate = float(np.mean(self.recent_interventions)) if self.recent_interventions else 0.0
        rate_cost = self._smooth_excess(rate - self.config.intervention_rate_budget, softness=0.20)
        self.last_reversal = reversal
        self.last_cost = float(np.clip(0.70 * reversal + 0.30 * rate_cost, 0.0, 1.0))
        return self.last_cost

    def features(self, action_keys: Sequence[str]) -> Dict[str, float]:
        target_index = 0.0
        active_targets = [
            target for target in str(self.last_target).split("+") if target in action_keys
        ]
        if active_targets:
            target_index = float(np.mean([
                (action_keys.index(target) + 1) / max(1, len(action_keys))
                for target in active_targets
            ]))
        return {
            "direction": float(self.last_direction),
            "target_index": target_index,
            "magnitude": 0.0,
            "reversal": float(self.last_reversal),
            "recent_intervention_rate": float(np.mean(self.recent_interventions)) if self.recent_interventions else 0.0,
            "time_since_intervention": float(np.clip(self.steps_since_intervention / max(1, self.config.oscillation_window), 0.0, 1.0)),
        }


class SoftConstraintController:
    names: Tuple[str, ...] = (
        "gap_overprice",
        "gap_underprice",
        "gap_0_2",
        "gap_2_5",
        "gap_5_10",
        "gap_10_plus",
        "fulfillment",
        "wait",
        "margin",
        "oscillation",
    )

    def __init__(self, config: Optional[ConstraintConfig] = None) -> None:
        self.config = config or ConstraintConfig()
        if len(self.config.cost_budgets) != len(self.names):
            raise ValueError("constraint cost budgets must match constraint names")
        self.lambdas = np.zeros(len(self.names), dtype=np.float32)
        self.cost_ema = np.zeros(len(self.names), dtype=np.float32)

    @staticmethod
    def _soft_cost(excess: float, softness: float) -> float:
        x = max(0.0, float(excess))
        return float(1.0 - np.exp(-((x / max(1e-6, softness)) ** 2)))

    def compute(
        self,
        *,
        observer: PlatformObservationModel,
        fulfillment_rate: float,
        wait_minutes: float,
        profit_margin: float,
        oscillation_cost: float,
    ) -> Dict[str, float]:
        c = self.config
        available = [p for p in observer.quote_probes.values() if p.get("available", 0.0) > 0.0]
        if available:
            weights = np.asarray([1.0 / max(0.05, float(p.get("uncertainty", 1.0))) for p in available])
            aggregate_gap = float(np.average([float(p.get("gap", c.target_gap)) for p in available], weights=weights))
            aggregate_uncertainty = float(np.average([float(p.get("uncertainty", 0.0)) for p in available], weights=weights))
        else:
            aggregate_gap = c.target_gap
            aggregate_uncertainty = 0.0
        lower = c.target_gap - c.overall_tolerance
        upper = c.target_gap + c.overall_tolerance
        costs: Dict[str, float] = {
            "gap_overprice": self._soft_cost((lower - aggregate_gap) + 0.5 * aggregate_uncertainty, c.gap_softness),
            "gap_underprice": self._soft_cost((aggregate_gap - upper) + 0.5 * aggregate_uncertainty, c.gap_softness),
        }
        for segment, tolerance in zip(SEGMENTS, c.segment_tolerances):
            probe = observer.quote_probes[segment]
            if probe.get("available", 0.0) <= 0.0:
                costs[f"gap_{segment}"] = 0.0
                continue
            robust_error = abs(float(probe.get("gap", c.target_gap)) - c.target_gap) + 0.5 * float(probe.get("uncertainty", 0.0))
            costs[f"gap_{segment}"] = self._soft_cost(robust_error - tolerance, c.gap_softness)
        costs.update({
            "fulfillment": self._soft_cost(c.fulfillment_floor - float(fulfillment_rate), 0.15),
            "wait": self._soft_cost(float(wait_minutes) - c.wait_limit_minutes, 4.0),
            "margin": self._soft_cost(c.margin_floor - float(profit_margin), 0.08),
            "oscillation": float(np.clip(oscillation_cost, 0.0, 1.0)),
        })
        return costs

    def vector(self, costs: Mapping[str, float]) -> np.ndarray:
        return np.asarray([float(np.clip(costs.get(name, 0.0), 0.0, 1.0)) for name in self.names], dtype=np.float32)

    def update(self, costs: Mapping[str, float], scale: float = 1.0) -> None:
        values = self.vector(costs)
        alpha = float(np.clip(self.config.cost_ema_alpha, 0.0, 1.0))
        self.cost_ema = (1.0 - alpha) * self.cost_ema + alpha * values
        budgets = np.asarray(self.config.cost_budgets, dtype=np.float32)
        self.lambdas = np.clip(
            self.lambdas + float(self.config.multiplier_lr) * float(max(0.0, scale)) * (self.cost_ema - budgets),
            0.0,
            float(self.config.multiplier_max),
        ).astype(np.float32)

    def snapshot(self) -> Dict[str, Any]:
        return {"lambdas": self.lambdas.tolist(), "cost_ema": self.cost_ema.tolist()}

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        self.lambdas = np.asarray(snapshot.get("lambdas", self.lambdas), dtype=np.float32)
        self.cost_ema = np.asarray(snapshot.get("cost_ema", self.cost_ema), dtype=np.float32)

    def diagnostics(self, costs: Mapping[str, float]) -> Dict[str, float]:
        result: Dict[str, float] = {}
        for index, name in enumerate(self.names):
            result[f"constraint_cost_{name}"] = float(costs.get(name, 0.0))
            result[f"constraint_lambda_{name}"] = float(self.lambdas[index])
            result[f"constraint_cost_ema_{name}"] = float(self.cost_ema[index])
            result[f"constraint_budget_{name}"] = float(self.config.cost_budgets[index])
        return result


def config_payload(
    observation: ObservationConfig,
    reward: PositiveRewardConfig,
    constraints: ConstraintConfig,
    stages: TrainingStageScheduler,
) -> Dict[str, Any]:
    return {
        "observation": asdict(observation),
        "positive_reward": asdict(reward),
        "constraints": asdict(constraints),
        "training_curriculum": stages.as_dict(),
    }
