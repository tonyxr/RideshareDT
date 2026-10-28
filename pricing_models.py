from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xiaoru Shi
Firm Pricing controllers

Firm1: RL (with PPO/Actor-Critic)
Firm2: Heuristic dynamic pricing (schedule + competitive response + guardrails)

"""

from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence, Tuple
import copy
import mkl_config  # noqa: F401 - set oneMKL env before NumPy/Torch
mkl_config  # imported intentionally for its process-wide environment side effects
import numpy as np
from collections import deque

from PPOAgent import PPOAgent
from Market_models import CoefficientOverrides
from optim_config import default_specs_for

@dataclass
class ActionDescriptor:
    """Executed RL-selected coefficient intervention metadata.

    PPO chooses hold or one coefficient and one absolute target for that
    coefficient. Bounds are enforced by the application layer.
    """

    action_id: int = 0
    target: str = "hold"
    direction: int = 0
    intended_step: float = 0.0
    realized_delta: float = 0.0
    realized_delta_norm: float = 0.0
    pre_value: float = 0.0
    post_value: float = 0.0
    lower_distance: float = 1.0
    upper_distance: float = 1.0
    magnitude_multiplier: float = 0.0
    magnitude_level: float = 0.0
    repeat_count: int = 0
    was_clipped: bool = False
    is_reversal: bool = False
    reversal_count: int = 0

@dataclass
class FirmMetrics:
    revenue: float = 0.0
    profit: float = 0.0
    wins: int = 0
    total: int = 0
    chosen: int = 0
    completed: int = 0
    unfulfilled: int = 0
    driver_pay: float = 0.0
    wait_minutes: float = 0.0
    pickup_minutes: float = 0.0
    driver_rejections: int = 0
    dispatch_offers: int = 0
    # v26: when request volume varies by hour, money outcomes are normalized by
    # the run's *base* requests per step so the reward is proportional to total
    # contribution profit (a busy hour counts more than a quiet one).  Shares
    # remain ratios over the step's realized requests.  0 = legacy behavior.
    reference_total: int = 0
    # v26 multi-homing: completed rides that first chose the other platform.
    spillover_completed: int = 0
    # Retry attempts received from riders who initially chose the other firm.
    # These are dispatch opportunities, not additional initial choices.
    spillover_attempts: int = 0

    @property
    def _money_denominator(self) -> int:
        return int(self.reference_total) if int(self.reference_total) > 0 else int(self.total)

    @property
    def share(self) -> float:
        return (self.wins / self.total) if self.total > 0 else 0.0

    @property
    def chosen_share(self) -> float:
        return (self.chosen / self.total) if self.total > 0 else 0.0

    @property
    def completed_share(self) -> float:
        return (self.completed / self.total) if self.total > 0 else 0.0

    @property
    def rev_per_request(self) -> float:
        denominator = self._money_denominator
        return (self.revenue / denominator) if denominator > 0 else 0.0
    
    @property
    def profit_per_request(self) -> float:
        denominator = self._money_denominator
        return (self.profit / denominator) if denominator > 0 else 0.0
    
    @property
    def fulfillment_rate(self) -> float:
        if self.dispatch_offers > 0:
            return self.completed / self.dispatch_offers
        return (self.completed / self.chosen) if self.chosen > 0 else 1.0

    @property
    def avg_wait_minutes(self) -> float:
        return (self.wait_minutes / self.completed) if self.completed > 0 else 0.0

    @property
    def driver_acceptance_rate(self) -> float:
        return ((self.dispatch_offers - self.driver_rejections) / self.dispatch_offers) if self.dispatch_offers > 0 else 1.0
    
def build_discrete_action_space(opt_keys: List[str], max_coeffs: int = 5) -> Tuple[Dict[int, Dict[str, int]], List[str]]:
    """Build hold plus one fixed up/down step for each pricing coefficient."""
    all_keys = ["base_fare", "per_minute", "per_mile", "booking_fee", "airport_fee"]
    active_keys = [k for k in all_keys if k in list(opt_keys)[:max_coeffs]]
    action_to_steps: Dict[int, Dict[str, int]] = {0: {key: 0 for key in active_keys}}

    def add_action(changes: Dict[str, int]) -> None:
        mapping = {key: int(changes.get(key, 0)) for key in active_keys}
        if not any(mapping.values()):
            return
        if mapping in action_to_steps.values():
            return
        action_to_steps[len(action_to_steps)] = mapping

    for key in active_keys:
        for step in (-1, 1):
            add_action({key: step})

    return action_to_steps, active_keys


def build_absolute_tariff_action_space(
    opt_keys: List[str],
    max_coeffs: int = 5,
    multipliers: Optional[Sequence[float]] = None,
) -> Tuple[
    Dict[int, Dict[str, int]],
    Dict[int, float],
    Dict[int, str],
    List[str],
]:
    """Legacy hold-or-whole-tariff action space used when revisions are off."""
    all_keys = ["base_fare", "per_minute", "per_mile", "booking_fee", "airport_fee"]
    active_keys = [k for k in all_keys if k in list(opt_keys)[:max_coeffs]]
    resolved = tuple(
        float(x)
        for x in (
            multipliers
            if multipliers
            else (0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30)
        )
    )
    action_to_steps: Dict[int, Dict[str, int]] = {
        0: {key: 0 for key in active_keys}
    }
    action_to_multiplier: Dict[int, float] = {}
    for multiplier in resolved:
        action = len(action_to_steps)
        direction = int(np.sign(multiplier - 1.0))
        action_to_steps[action] = {
            candidate: direction for candidate in active_keys
        }
        action_to_multiplier[action] = multiplier
    return action_to_steps, action_to_multiplier, {}, active_keys


def build_independent_coefficient_action_space(
    opt_keys: List[str],
    max_coeffs: int = 5,
    multipliers: Optional[Sequence[float]] = None,
) -> Tuple[
    Dict[int, Dict[str, int]],
    Dict[int, float],
    Dict[int, str],
    List[str],
]:
    """Build one hold plus independent coefficient/target interventions.

    Every non-hold action changes exactly one managed coefficient. The action
    id is a compact encoding of two policy decisions: which coefficient to
    change, and the absolute target multiplier for that coefficient. Repeating
    an action keeps that coefficient at the chosen target rather than causing
    incremental drift. Other coefficients remain exactly as posted.
    """
    all_keys = ["base_fare", "per_minute", "per_mile", "booking_fee", "airport_fee"]
    active_keys = [k for k in all_keys if k in list(opt_keys)[:max_coeffs]]
    multipliers = tuple(
        float(x) for x in (multipliers if multipliers else (0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30))
    )
    action_to_steps: Dict[int, Dict[str, int]] = {
        0: {key: 0 for key in active_keys}
    }
    action_to_multiplier: Dict[int, float] = {}
    action_to_target_key: Dict[int, str] = {}
    if not multipliers or not all(np.isfinite(multipliers)):
        raise ValueError("coefficient target multipliers must be finite and non-empty")
    if len(set(multipliers)) != len(multipliers):
        raise ValueError("coefficient target multipliers must be unique")
    for key in active_keys:
        for multiplier in multipliers:
            action = len(action_to_steps)
            direction = int(np.sign(float(multiplier) - 1.0))
            action_to_steps[action] = {
                candidate: direction if candidate == key else 0
                for candidate in active_keys
            }
            action_to_multiplier[action] = float(multiplier)
            action_to_target_key[action] = str(key)
    return (
        action_to_steps,
        action_to_multiplier,
        action_to_target_key,
        active_keys,
    )


# V17 M5: one absolute card cell controls distance and time components only.
TWO_LEVER_ACTION_SEMANTICS = "two_lever_absolute_card_v1"
TWO_LEVER_LEVELS: Tuple[float, ...] = (0.8, 0.9, 1.0, 1.2, 1.5)


def build_two_lever_action_space(
    opt_keys: List[str],
    levels: Sequence[float] = TWO_LEVER_LEVELS,
) -> Tuple[Dict[int, Dict[str, int]], Dict[int, Tuple[float, float]], List[str]]:
    """Return the 25 absolute (distance, time) cells registered for V17."""
    required = ("per_mile", "per_minute")
    if not set(required).issubset(opt_keys):
        raise ValueError("two_lever_v1 requires per_mile and per_minute")
    resolved = tuple(float(value) for value in levels)
    if (
        resolved != TWO_LEVER_LEVELS
        or len(set(resolved)) != len(resolved)
        or not all(np.isfinite(value) and value > 0.0 for value in resolved)
    ):
        raise ValueError(f"two_lever_v1 levels are fixed at {TWO_LEVER_LEVELS}")
    action_to_steps: Dict[int, Dict[str, int]] = {}
    action_to_cell: Dict[int, Tuple[float, float]] = {}
    for distance_multiplier in resolved:
        for time_multiplier in resolved:
            action = len(action_to_cell)
            action_to_cell[action] = (
                float(distance_multiplier),
                float(time_multiplier),
            )
            action_to_steps[action] = {
                key: (
                    int(np.sign(distance_multiplier - 1.0))
                    if key == "per_mile"
                    else int(np.sign(time_multiplier - 1.0))
                    if key == "per_minute"
                    else 0
                )
                for key in opt_keys
            }
    return action_to_steps, action_to_cell, list(required)


# ---------------------------------------------------------------------------
# Revision 15d: compact hybrid relative action contract.
# ---------------------------------------------------------------------------
HYBRID_RELATIVE_ACTION_SEMANTICS = "hybrid_relative_tariff_steps_v1"
HYBRID_RELATIVE_STEP_MAGNITUDES: Tuple[float, ...] = (0.10, 0.20)


def build_hybrid_relative_action_space(
    opt_keys: List[str],
    max_coeffs: int = 5,
    step_magnitudes: Sequence[float] = HYBRID_RELATIVE_STEP_MAGNITUDES,
) -> Tuple[
    Dict[int, Dict[str, int]],
    Dict[int, float],
    Dict[int, str],
    List[str],
]:
    """Hold, whole-tariff steps, and per-coefficient relative steps.

    A step is additive in city-anchor units: ``+0.10`` moves the posted
    coefficient from (for example) 0.80x to 0.90x its city anchor.  This avoids
    Revision 15c's one-decision jump between unrelated absolute targets while
    retaining a whole-tariff response for market shocks.
    """
    all_keys = ["base_fare", "per_minute", "per_mile", "booking_fee", "airport_fee"]
    active_keys = [key for key in all_keys if key in list(opt_keys)[:max_coeffs]]
    if not active_keys:
        raise ValueError("hybrid relative actions require at least one coefficient")
    magnitudes = tuple(sorted(set(float(value) for value in step_magnitudes)))
    if (
        not magnitudes
        or any(not np.isfinite(value) or value <= 0.0 for value in magnitudes)
    ):
        raise ValueError("relative step magnitudes must be finite, unique, and positive")
    deltas = tuple(-value for value in reversed(magnitudes)) + magnitudes
    action_to_steps: Dict[int, Dict[str, int]] = {
        0: {key: 0 for key in active_keys}
    }
    action_to_relative_step: Dict[int, float] = {}
    action_to_target_key: Dict[int, str] = {}

    for target in ("whole_tariff", *active_keys):
        for delta in deltas:
            action = len(action_to_steps)
            action_to_steps[action] = {
                key: int(np.sign(delta))
                if target == "whole_tariff" or key == target
                else 0
                for key in active_keys
            }
            action_to_relative_step[action] = float(delta)
            action_to_target_key[action] = str(target)
    return (
        action_to_steps,
        action_to_relative_step,
        action_to_target_key,
        active_keys,
    )


# ---------------------------------------------------------------------------
# Revision 15: relative level steps on a fixed, validated tariff structure.
# ---------------------------------------------------------------------------
LEVEL_STEP_ACTION_SEMANTICS = "relative_level_step_on_fixed_structure"
TARIFF_STRUCTURE_FORMAT = "revision15-tariff-structure-v1"
LEVEL_STEP_DELTAS: Tuple[float, ...] = (-0.2, -0.1, 0.1, 0.2)
TARIFF_LEVEL_GRID_STEP = 0.1
TARIFF_LEVEL_KEYS: Tuple[str, ...] = (
    "base_fare",
    "per_minute",
    "per_mile",
    "booking_fee",
    "airport_fee",
)
# Typical-ride exposure of each coefficient: one base fare, 14.65 minutes,
# 3.95 miles, one booking fee and a 7.75% airport incidence. Core uses the
# same weights for fare-equivalent action impacts.
LEVEL_STEP_EXPOSURE: Dict[str, float] = {
    "base_fare": 1.0,
    "per_minute": 14.65,
    "per_mile": 3.95,
    "booking_fee": 1.0,
    "airport_fee": 0.0775,
}


def snap_tariff_level(value: float) -> float:
    """Return ``value`` on the 0.1 level lattice, or raise if it is off-lattice."""
    scaled = float(value) / TARIFF_LEVEL_GRID_STEP
    if not np.isfinite(scaled):
        raise ValueError(f"tariff level must be finite: {value!r}")
    nearest = int(round(scaled))
    if abs(scaled - nearest) > 1e-6:
        raise ValueError(
            f"tariff level {value!r} is not on the {TARIFF_LEVEL_GRID_STEP} lattice"
        )
    return round(nearest * TARIFF_LEVEL_GRID_STEP, 6)


def tariff_level_of(
    multipliers: Mapping[str, float], anchors: Mapping[str, float]
) -> float:
    """Exposure-weighted fare level of per-coefficient anchor multipliers."""
    keys = list(multipliers)
    numerator = sum(
        float(anchors[key]) * float(multipliers[key]) * LEVEL_STEP_EXPOSURE[key]
        for key in keys
    )
    denominator = sum(
        float(anchors[key]) * LEVEL_STEP_EXPOSURE[key] for key in keys
    )
    if not denominator > 0.0:
        raise ValueError("tariff level requires positive exposure-weighted anchors")
    return float(numerator / denominator)


def validate_tariff_structure(
    payload: Mapping[str, Any],
    opt_keys: Sequence[str],
    level_bounds: Tuple[float, float],
) -> Dict[str, Any]:
    """Validate a Revision 15 structure payload. A bad payload is never repaired."""
    if payload.get("format") != TARIFF_STRUCTURE_FORMAT:
        raise ValueError(
            f"tariff structure format must be {TARIFF_STRUCTURE_FORMAT!r}, "
            f"got {payload.get('format')!r}"
        )
    structure = {
        str(key): float(value) for key, value in dict(payload["structure"]).items()
    }
    anchors = {
        str(key): float(value) for key, value in dict(payload["anchors"]).items()
    }
    if set(structure) != set(opt_keys) or set(anchors) != set(opt_keys):
        raise ValueError(
            "tariff structure and anchors must name exactly the managed "
            f"coefficients {sorted(opt_keys)}"
        )
    if any(not np.isfinite(value) or value <= 0.0 for value in structure.values()):
        raise ValueError("tariff structure multipliers must be finite and positive")
    # A city calibration may legitimately identify a zero coefficient (Boston's
    # public data, for example, contains no separately recoverable airport fee).
    # The level contract only needs a positive exposure-weighted *total* anchor;
    # requiring every component to be positive would make an otherwise valid
    # city impossible to train.  Negative and non-finite anchors remain invalid,
    # and tariff_level_of below rejects an all-zero/zero-exposure payload.
    if any(not np.isfinite(value) or value < 0.0 for value in anchors.values()):
        raise ValueError("tariff structure anchors must be finite and nonnegative")
    level = tariff_level_of(structure, anchors)
    if abs(level - 1.0) > 1e-6:
        raise ValueError(
            f"tariff structure must be normalized to level 1.0, got {level:.9f}"
        )
    lower = snap_tariff_level(level_bounds[0])
    upper = snap_tariff_level(level_bounds[1])
    if not 0.0 < lower < upper:
        raise ValueError(f"invalid tariff level bounds {level_bounds!r}")
    reference = snap_tariff_level(payload["reference_level"])
    if not lower - 1e-9 <= reference <= upper + 1e-9:
        raise ValueError(
            f"reference level {reference} lies outside the level bounds "
            f"[{lower}, {upper}]"
        )
    return {
        "format": TARIFF_STRUCTURE_FORMAT,
        "structure": {key: structure[key] for key in opt_keys},
        "anchors": {key: anchors[key] for key in opt_keys},
        "reference_level": reference,
        "source": dict(payload.get("source", {}) or {}),
    }


def build_level_step_action_space(
    opt_keys: List[str],
    max_coeffs: int = 5,
    deltas: Sequence[float] = LEVEL_STEP_DELTAS,
) -> Tuple[Dict[int, Dict[str, int]], Dict[int, float], List[str]]:
    """Hold plus relative steps of the whole-tariff level on a fixed structure."""
    all_keys = list(TARIFF_LEVEL_KEYS)
    active_keys = [k for k in all_keys if k in list(opt_keys)[:max_coeffs]]
    if not active_keys:
        raise ValueError("level-step actions require at least one managed coefficient")
    resolved = tuple(float(delta) for delta in deltas)
    if not resolved or len(set(resolved)) != len(resolved):
        raise ValueError("level steps must be unique and non-empty")
    for delta in resolved:
        if delta == 0.0:
            raise ValueError("a zero level step duplicates hold")
        snap_tariff_level(abs(delta))
    action_to_steps: Dict[int, Dict[str, int]] = {
        0: {key: 0 for key in active_keys}
    }
    action_to_level_step: Dict[int, float] = {}
    for delta in resolved:
        action = len(action_to_steps)
        action_to_steps[action] = {
            key: int(np.sign(delta)) for key in active_keys
        }
        action_to_level_step[action] = float(delta)
    return action_to_steps, action_to_level_step, active_keys


class FirmStaticPricer:
    def __init__(self):
        self.overrides = CoefficientOverrides()

    def act(self, **kwargs) -> None:
        # no-op: coefficients remain fixed
        return
    
    def update(self, *args, **kwargs) -> None:
        pass

class FirmHeuristicPricer:
    """
    Stable baseline:
    - schedule targets for peak & bad weather
    - competitive response using EMA share and EMA price-gap
    - cooldown to prevent thrashing
    """

    def __init__(self, seed: Optional[int] = None, managed_keys: Optional[List[str]] = None):
        self.rng = np.random.default_rng(seed)
        self.overrides = CoefficientOverrides()
        keys = list(managed_keys) if managed_keys is not None else ["base_fare", "per_minute"]
        editable = {"base_fare", "per_minute", "per_mile", "booking_fee", "airport_fee"}
        self.managed_keys = tuple(k for k in keys if k in editable)
        if not self.managed_keys:
            self.managed_keys = ("base_fare", "per_minute")

        self.config = default_specs_for(list(self.managed_keys))
        self.action_to_steps, self.action_keys = build_discrete_action_space(list(self.managed_keys))
        self.step_base_fare = self.config.step.get("base_fare", 0.10)
        self.step_per_minute = self.config.step.get("per_minute", 0.01)
        self.step_per_mile = self.config.step.get("per_mile", 0.05)
        self.step_booking_fee = self.config.step.get("booking_fee", 0.10)
        self.step_airport_fee = self.config.step.get("airport_fee", 0.25)
        self.base_bounds = self.config.bounds.get("base_fare", (1.50, 6.00))
        self.pmin_bounds = self.config.bounds.get("per_minute", (0.10, 1.00))
        self.pmile_bounds = self.config.bounds.get("per_mile", (0.50, 4.00))
        self.booking_bounds = self.config.bounds.get("booking_fee", (0.00, 6.00))
        self.airport_bounds = self.config.bounds.get("airport_fee", (0.00, 15.00))
        self.single_state_dim = 89
        self.action_feature_dim = 19
        self.last_state_features = None
        self.last_action_features = None

        self.target_share = 0.50
        self.share_band = 0.03
        self.price_gap_threshold = 0.50  # Firm2 - Firm1

        self.alpha = 0.05
        self.ema_share = 0.50
        self.ema_gap = 0.0

        self.cooldown = 0
        self.cooldown_H = 8
        
    def observe_state(self, state_features: np.ndarray, action_features: Optional[np.ndarray] = None) -> None:
        """Cache the latest observation/action features for parity diagnostics."""
        self.last_state_features = np.asarray(state_features, dtype=np.float32).copy()
        self.last_action_features = (
            None if action_features is None else np.asarray(action_features, dtype=np.float32).copy()
        )

    def action_steps(self, action: int) -> Dict[str, int]:
        return dict(self.action_to_steps.get(int(action), {}))

    @staticmethod
    def _ema(x: float, ema: float, a: float) -> float:
        return (1 - a) * ema + a * x

    def act(
        self,
        city_base: float,
        city_pmin: float,
        hour: int,
        weather: str,
        city_pmile: Optional[float] = None,
        city_booking: Optional[float] = None,
        city_airport: Optional[float] = None,
    ):
        # initialize once
        if "base_fare" in self.managed_keys and self.overrides.base_fare is None:
            self.overrides.base_fare = float(city_base)
        if "per_minute" in self.managed_keys and self.overrides.per_minute is None:
            self.overrides.per_minute = float(city_pmin)
        if "per_mile" in self.managed_keys and self.overrides.per_mile is None:
            self.overrides.per_mile = float(city_pmile if city_pmile is not None else 1.50)
        if "booking_fee" in self.managed_keys and self.overrides.booking_fee is None:
            self.overrides.booking_fee = float(city_booking if city_booking is not None else 2.00)
        if "airport_fee" in self.managed_keys and self.overrides.airport_fee is None:
            self.overrides.airport_fee = float(city_airport if city_airport is not None else 5.00)

        if self.cooldown > 0:
            self.cooldown -= 1
            return

        peak = (7 <= hour < 10) or (16 <= hour < 19)
        bad_weather = weather in ("rain", "snow")

        # baseline targets (schedule)
        base_target = city_base + (0.20 if peak else 0.0) + (0.10 if bad_weather else 0.0)
        pmin_target = city_pmin + (0.01 if peak else 0.0) + (0.005 if bad_weather else 0.0)
        pmile_anchor = float(city_pmile if city_pmile is not None else 1.50)
        booking_anchor = float(city_booking if city_booking is not None else 2.00)
        airport_anchor = float(city_airport if city_airport is not None else 5.00)
        pmile_target = pmile_anchor + (0.08 if peak else 0.0) + (0.05 if bad_weather else 0.0)
        booking_target = booking_anchor + (0.08 if peak else 0.0) + (0.04 if bad_weather else 0.0)
        airport_target = airport_anchor + (0.20 if peak else 0.0)

        losing = self.ema_share < (self.target_share - self.share_band)
        overpriced = self.ema_gap > self.price_gap_threshold

        winning = self.ema_share > (self.target_share + self.share_band)
        underpriced = self.ema_gap < (-self.price_gap_threshold)

        chosen = None
        direction = 0

        preferred = "base" if (peak or bad_weather) else "pmin"
        if losing and overpriced:
            chosen = preferred
            direction = -1
        elif winning and underpriced:
            chosen = preferred
            direction = +1
        else:
            # Pick the most off-target managed coefficient instead of walking a
            # fixed if/elif priority list.  The old priority meant base/minute
            # corrections could permanently starve booking and airport fees,
            # leaving them flat in coefficient trajectory plots even though they
            # were listed as managed keys.
            target_specs = [
                ("base", "base_fare", base_target, 0.15),
                ("pmin", "per_minute", pmin_target, 0.02),
                ("pmile", "per_mile", pmile_target, 0.08),
                ("booking", "booking_fee", booking_target, 0.04),
                ("airport", "airport_fee", airport_target, 0.10),
            ]
            candidates = []
            for action_name, coeff_name, target, tolerance in target_specs:
                if coeff_name not in self.managed_keys:
                    continue
                curr = getattr(self.overrides, coeff_name)
                if curr is None:
                    continue
                delta = float(target) - float(curr)
                if abs(delta) >= float(tolerance):
                    candidates.append((abs(delta) / max(float(tolerance), 1e-6), action_name, delta))
            if candidates:
                _, chosen, delta = max(candidates, key=lambda x: x[0])
                direction = +1 if delta > 0.0 else -1
        
        if preferred == "base" and "base_fare" not in self.managed_keys:
            chosen = "pmin" if "per_minute" in self.managed_keys else None
        elif preferred == "pmin" and "per_minute" not in self.managed_keys:
            chosen = "base" if "base_fare" in self.managed_keys else None
            
        if chosen is None or direction == 0:
            return

        if chosen == "base" and "base_fare" in self.managed_keys:
            self.overrides.base_fare = float(np.clip(
                float(self.overrides.base_fare) + direction * self.step_base_fare, *self.base_bounds
            ))
        elif chosen == "pmin" and "per_minute" in self.managed_keys:
            self.overrides.per_minute = float(np.clip(
                float(self.overrides.per_minute) + direction * self.step_per_minute, *self.pmin_bounds
            ))
        elif chosen == "pmile" and "per_mile" in self.managed_keys:
            self.overrides.per_mile = float(np.clip(
                float(self.overrides.per_mile) + direction * self.step_per_mile, *self.pmile_bounds
            ))
        elif chosen == "booking" and "booking_fee" in self.managed_keys:
            self.overrides.booking_fee = float(np.clip(
                float(self.overrides.booking_fee) + direction * self.step_booking_fee, *self.booking_bounds
            ))
        elif chosen == "airport" and "airport_fee" in self.managed_keys:
            self.overrides.airport_fee = float(np.clip(
                float(self.overrides.airport_fee) + direction * self.step_airport_fee, *self.airport_bounds
            ))
        # One-price-coefficient action constraint: the dynamic competitor may
        # manage all five coefficients over time, but this decision point only
        # changes the single selected coefficient above.

        self.cooldown = self.cooldown_H

    def update(
        self,
        metrics: FirmMetrics,
        price_gap_mean: float,
        supply_state: Optional[Dict[str, float]] = None,
        supply_state_vector: Optional[np.ndarray] = None,
    ):
        del supply_state, supply_state_vector
        self.ema_share = self._ema(metrics.share, self.ema_share, self.alpha)
        self.ema_gap = self._ema(price_gap_mean, self.ema_gap, self.alpha)
        
class FirmMarginGuardrailPricer(FirmHeuristicPricer):
    """
    Heuristic pricer with conservative margin/share guardrails.

    This keeps the existing schedule and competitive response behavior, then
    nudges prices back toward a safer band when recent revenue per request or
    win share indicates the firm may be pricing too aggressively or too cheaply.
    """

    def __init__(self, seed: Optional[int] = None, managed_keys: Optional[List[str]] = None):
        super().__init__(seed=seed, managed_keys=managed_keys)
        self.min_rev_per_request = 6.00
        self.high_share_threshold = 0.5
        self.low_share_threshold = 0.38
        self.guardrail_cooldown_H = 6

    def update(
        self,
        metrics: FirmMetrics,
        price_gap_mean: float,
        supply_state: Optional[Dict[str, float]] = None,
        supply_state_vector: Optional[np.ndarray] = None,
    ):
        super().update(
            metrics=metrics,
            price_gap_mean=price_gap_mean,
            supply_state=supply_state,
            supply_state_vector=supply_state_vector,
        )

        if self.cooldown > 0:
            return

        # If share is high but revenue per request is weak, add a small price
        # increase to protect margin. If share is low while the firm is already
        # more expensive than its competitor, trim prices to recover demand.
        increase_for_margin = (
            metrics.total > 0
            and metrics.share >= self.high_share_threshold
            and metrics.rev_per_request < self.min_rev_per_request
        )
        cut_for_share = metrics.total > 0 and metrics.share <= self.low_share_threshold and price_gap_mean > 0.0

        if not (increase_for_margin or cut_for_share):
            return

        direction = 1 if increase_for_margin else -1
        # Respect the same one-coefficient-per-decision constraint as the PPO
        # controller.  Prefer base fare for broad margin/share corrections, then
        # fall back to per-minute if base fare is not managed.
        if "base_fare" in self.managed_keys and self.overrides.base_fare is not None:
            self.overrides.base_fare = float(np.clip(
                float(self.overrides.base_fare) + direction * (0.5 * self.step_base_fare),
                *self.base_bounds,
            ))
            self.cooldown = self.guardrail_cooldown_H
            return
        if "per_minute" in self.managed_keys and self.overrides.per_minute is not None:
            self.overrides.per_minute = float(np.clip(
                float(self.overrides.per_minute) + direction * (0.5 * self.step_per_minute),
                *self.pmin_bounds,
            ))
            self.cooldown = self.guardrail_cooldown_H
            
class FirmAdaptiveBestResponsePricer:
    """Bounded lagged adaptive best-response competitor for Firm 2.

    The controller observes smoothed market share, revenue per request, and the
    realized Firm2-Firm1 price gap.  It then applies a myopic threshold rule to
    one bounded coefficient at a time.  This is stronger than a static baseline
    but remains interpretable and weaker than Firm1's PPO policy because it has
    no learned value function or delayed-credit optimization.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        managed_keys: Optional[List[str]] = None,
        *,
        target_share: float = 0.50,
        alpha_share: float = 0.20,
        alpha_revenue: float = 0.20,
        alpha_gap: float = 0.20,
        k_share: float = 1.00,
        k_gap: float = 0.60,
        k_revenue: float = 0.40,
        k_supply: float = 0.85,
        response_threshold: float = 0.05,
        cooldown_batches: int = 3,
        step_scale: float = 1.00,
        revenue_target: Optional[float] = None,
    ):
        self.rng = np.random.default_rng(seed)
        self.overrides = CoefficientOverrides()
        editable = {"base_fare", "per_minute", "per_mile", "booking_fee", "airport_fee"}
        keys = list(managed_keys) if managed_keys is not None else ["base_fare", "per_minute", "per_mile"]
        self.managed_keys = tuple(k for k in keys if k in editable)
        if not self.managed_keys:
            self.managed_keys = ("base_fare", "per_minute", "per_mile")
        self.config = default_specs_for(list(self.managed_keys))
        
        self.action_to_steps, self.action_keys = build_discrete_action_space(list(self.managed_keys))
        self.single_state_dim = 89
        self.action_feature_dim = 19
        self.last_state_features = None
        self.last_action_features = None

        self.target_share = float(np.clip(target_share, 0.0, 1.0))
        self.alpha_share = float(np.clip(alpha_share, 0.0, 1.0))
        self.alpha_revenue = float(np.clip(alpha_revenue, 0.0, 1.0))
        self.alpha_gap = float(np.clip(alpha_gap, 0.0, 1.0))
        self.k_share = float(max(0.0, k_share))
        self.k_gap = float(max(0.0, k_gap))
        self.k_revenue = float(max(0.0, k_revenue))
        self.k_supply = float(max(0.0, k_supply))
        self.response_threshold = float(max(0.0, response_threshold))
        self.cooldown_H = int(max(0, cooldown_batches))
        self.cooldown = 0
        self.step_scale = float(max(0.0, step_scale))
        self.revenue_target = None if revenue_target is None else float(max(1e-6, revenue_target))

        self.ema_share = self.target_share
        self.ema_choice_share = self.target_share
        self.ema_completed_share = self.target_share
        self.ema_revenue = 0.0 if self.revenue_target is None else self.revenue_target
        self.ema_gap = 0.0
        self.ema_fulfillment = 1.0
        self.ema_acceptance = 1.0
        self.ema_wait = 0.0
        self.ema_pickup = 0.0
        self.ema_driver_earnings = 0.0
        self.ema_idle_share = 1.0
        self.ema_utilization = 0.0
        self.last_supply_state_vector = np.zeros(8, dtype=np.float32)
        self.last_supply_stress = 0.0
        self.last_demand_shortfall = 0.0
        self.last_response_score = 0.0
        self.last_selected_key = "hold"
        self.last_direction = 0
        self.last_reason = "init"
    
    def observe_state(self, state_features: np.ndarray, action_features: Optional[np.ndarray] = None) -> None:
        """Cache the same observation/action-feature tensors exposed to the RL policy."""
        self.last_state_features = np.asarray(state_features, dtype=np.float32).copy()
        self.last_action_features = (
            None if action_features is None else np.asarray(action_features, dtype=np.float32).copy()
        )

    def action_steps(self, action: int) -> Dict[str, int]:
        return dict(self.action_to_steps.get(int(action), {}))
    
    @staticmethod
    def _ema(x: float, ema: float, a: float) -> float:
        return (1 - a) * float(ema) + a * float(x)

    def _anchor_map(
        self,
        city_base: float,
        city_pmin: float,
        city_pmile: Optional[float],
        city_booking: Optional[float],
        city_airport: Optional[float],
    ) -> Dict[str, float]:
        return {
            "base_fare": float(city_base),
            "per_minute": float(city_pmin),
            "per_mile": float(city_pmile if city_pmile is not None else 1.50),
            "booking_fee": float(city_booking if city_booking is not None else 2.00),
            "airport_fee": float(city_airport if city_airport is not None else 5.00),
        }

    def _ensure_initialized(self, anchors: Dict[str, float]) -> None:
        for key in self.managed_keys:
            if getattr(self.overrides, key, None) is None:
                setattr(self.overrides, key, float(anchors[key]))

    def _response_score(self) -> float:
        # Use chosen-share for demand pressure and completed-share/fulfillment
        # for service realization. Ride-hailing pricing literature treats
        # dynamic prices as a two-sided balancing instrument, so fulfillment,
        # waiting time, driver acceptance, and driver earnings should dampen
        # demand-chasing cuts when the real bottleneck is supply.
        choice_share = float(np.clip(self.ema_choice_share, 0.0, 1.0))
        completed_share = float(np.clip(self.ema_completed_share, 0.0, 1.0))
        demand_shortfall = float(np.clip(self.target_share - choice_share, -1.0, 1.0))
        gap_pressure = float(np.clip(self.ema_gap / 2.0, -1.0, 1.0))
        if self.revenue_target is None:
            revenue_surplus = 0.0
        else:
            # A zero-revenue opening is valid. A percentage relative to zero
            # is undefined; express that change in the city's currency unit.
            revenue_scale = self.revenue_target if self.revenue_target > 0 else float(getattr(self, 'currency_unit', 1.))
            revenue_surplus = float(
                np.clip((self.ema_revenue - self.revenue_target) / revenue_scale, -1.0, 1.0)
            )
        fulfillment_stress = float(np.clip((0.78 - self.ema_fulfillment) / 0.38, 0.0, 1.0))
        acceptance_stress = float(np.clip((0.70 - self.ema_acceptance) / 0.35, 0.0, 1.0))
        wait_stress = float(np.clip((self.ema_wait - 8.0) / 10.0, 0.0, 1.0))
        pickup_stress = float(np.clip((self.ema_pickup - 8.0) / 10.0, 0.0, 1.0))
        earnings_reference = float(getattr(self, 'earnings_reference', 24.0))
        earnings_stress = float(np.clip((earnings_reference - self.ema_driver_earnings) / earnings_reference, 0.0, 1.0))
        idle_stress = float(np.clip((0.22 - self.ema_idle_share) / 0.22, 0.0, 1.0))
        utilization_stress = float(np.clip((self.ema_utilization - 0.78) / 0.22, 0.0, 1.0))
        completion_gap = float(np.clip(choice_share - completed_share, 0.0, 1.0))
        supply_stress = float(np.clip(
            0.22 * fulfillment_stress
            + 0.18 * acceptance_stress
            + 0.14 * wait_stress
            + 0.12 * pickup_stress
            + 0.18 * earnings_stress
            + 0.08 * idle_stress
            + 0.08 * utilization_stress
            + 0.18 * completion_gap,
            0.0,
            1.0,
        ))
        self.last_supply_stress = supply_stress
        self.last_demand_shortfall = demand_shortfall
        return float(
            self.k_share * demand_shortfall
            + self.k_gap * gap_pressure
            - self.k_revenue * revenue_surplus
            - self.k_supply * supply_stress
        )

    def _select_coefficient(self, direction: int) -> Optional[str]:
        if direction < 0:
            priority = ("base_fare", "booking_fee", "per_mile", "per_minute", "airport_fee")
        else:
            priority = ("per_mile", "per_minute", "booking_fee", "base_fare", "airport_fee")
        candidates: List[Tuple[float, int, str]] = []
        for rank, key in enumerate(priority):
            if key not in self.managed_keys:
                continue
            curr = getattr(self.overrides, key, None)
            if curr is None:
                continue
            lb, ub = self.config.bounds[key]
            width = max(1e-6, float(ub - lb))
            room = (float(curr) - float(lb)) if direction < 0 else (float(ub) - float(curr))
            room_frac = float(np.clip(room / width, 0.0, 1.0))
            if room_frac > 1e-6:
                candidates.append((room_frac, -rank, key))
        if not candidates:
            return None
        return max(candidates)[2]

    def _apply_step(self, key: str, direction: int) -> bool:
        curr = getattr(self.overrides, key, None)
        if curr is None:
            return False
        step = float(self.config.step[key]) * self.step_scale
        lb, ub = self.config.bounds[key]
        new_value = float(np.clip(float(curr) + float(np.sign(direction)) * step, lb, ub))
        if abs(new_value - float(curr)) <= 1e-12:
            return False
        setattr(self.overrides, key, new_value)
        return True

    def act(
        self,
        city_base: float,
        city_pmin: float,
        hour: int,
        weather: str,
        city_pmile: Optional[float] = None,
        city_booking: Optional[float] = None,
        city_airport: Optional[float] = None,
    ) -> None:
        del hour, weather
        anchors = self._anchor_map(city_base, city_pmin, city_pmile, city_booking, city_airport)
        self._ensure_initialized(anchors)
        self.last_selected_key = "hold"
        self.last_direction = 0
        if self.cooldown > 0:
            self.cooldown -= 1
            self.last_reason = "cooldown"
            return

        score = self._response_score()
        self.last_response_score = score
        if abs(score) <= self.response_threshold:
            self.last_reason = "threshold_hold"
            return

        direction = -1 if score > 0.0 else 1
        key = self._select_coefficient(direction)
        if key is None:
            self.last_reason = "no_room"
            return

        moved = self._apply_step(key, direction)
        self.last_selected_key = key if moved else "hold"
        self.last_direction = int(direction) if moved else 0
        self.last_reason = "adaptive_response" if moved else "bounded"
        if moved:
            self.cooldown = self.cooldown_H

    def update(
        self,
        metrics: FirmMetrics,
        price_gap_mean: float,
        supply_state: Optional[Dict[str, float]] = None,
        supply_state_vector: Optional[np.ndarray] = None,
    ) -> None:
        price_gap_mean = float(price_gap_mean) / float(getattr(self, 'currency_unit', 1.))
        alpha_share, alpha_revenue, alpha_gap = self.alpha_share, self.alpha_revenue, self.alpha_gap
        if getattr(self, 'operational_clock', False):
            # Parameters describe one day, not each telemetry block.
            blocks = int(self.reference_window_steps)
            alpha_share, alpha_revenue, alpha_gap = [1 - (1-a)**(1/blocks)
                for a in (alpha_share, alpha_revenue, alpha_gap)]
            seen = getattr(self, '_reference_blocks', 0)
            if seen < self.reference_window_steps:
                self._reference_blocks = seen + 1
                self._reference_revenue = getattr(self, '_reference_revenue', 0.) + metrics.revenue
                self._reference_requests = getattr(self, '_reference_requests', 0.) + metrics._money_denominator
                self.revenue_target = self._reference_revenue / max(1., self._reference_requests)
                # Seed both from the SAME complete-day statistic. Until that
                # day is complete the accumulating statistic is provisional.
                self.ema_revenue = self.revenue_target
            if (not getattr(self, '_supply_observed', False) and supply_state
                    and supply_state.get('online_minutes', 1.) > 0):
                self.ema_driver_earnings = float(supply_state['driver_earnings_per_online_hour'])
                self._supply_observed = True
        if self.revenue_target is None:
            self.revenue_target = float(max(1e-6, metrics.rev_per_request))
            self.ema_revenue = self.revenue_target
        self.ema_share = self._ema(metrics.share, self.ema_share, alpha_share)
        self.ema_choice_share = self._ema(metrics.chosen_share, self.ema_choice_share, alpha_share)
        self.ema_completed_share = self._ema(metrics.completed_share, self.ema_completed_share, alpha_share)
        if not (getattr(self, 'operational_clock', False) and seen < self.reference_window_steps):
            self.ema_revenue = self._ema(metrics.rev_per_request, self.ema_revenue, alpha_revenue)
        self.ema_gap = self._ema(price_gap_mean, self.ema_gap, alpha_gap)
        supply = supply_state or {}
        self.ema_fulfillment = self._ema(
            supply.get("fulfillment_rate", metrics.fulfillment_rate),
            self.ema_fulfillment,
            alpha_share,
        )
        self.ema_acceptance = self._ema(
            supply.get("acceptance_rate", metrics.driver_acceptance_rate),
            self.ema_acceptance,
            alpha_share,
        )
        self.ema_wait = self._ema(supply.get("avg_wait_minutes", metrics.avg_wait_minutes), self.ema_wait, alpha_share)
        self.ema_pickup = self._ema(
            supply.get("avg_pickup_minutes", metrics.avg_wait_minutes),
            self.ema_pickup,
            alpha_share,
        )
        if not getattr(self, 'operational_clock', False) or supply.get('online_minutes', 1.) > 0:
            self.ema_driver_earnings = self._ema(
                supply.get("driver_earnings_per_online_hour" if getattr(self, 'operational_clock', False)
                           else "driver_earnings_per_hour", self.ema_driver_earnings),
                self.ema_driver_earnings, alpha_share)
        self.ema_idle_share = self._ema(supply.get("idle_driver_share", self.ema_idle_share), self.ema_idle_share, alpha_share)
        self.ema_utilization = self._ema(supply.get("utilization", self.ema_utilization), self.ema_utilization, alpha_share)
        # These benchmarks bypass the base act() response calculation. Keep
        # legacy controllers' act-time diagnostic cadence exactly unchanged.
        if getattr(self, "refresh_supply_stress_on_update", False):
            self._response_score()
        if supply_state_vector is not None:
            vec = np.asarray(supply_state_vector, dtype=np.float32).reshape(-1)
            self.last_supply_state_vector = np.nan_to_num(vec[:8], nan=0.0, posinf=1.0, neginf=0.0)

class FirmPIPriceGapPricer(FirmAdaptiveBestResponsePricer):
    """PI feedback benchmark for price-gap control.

    Literature basis: Fayed, Nilsson, and Geroliminis (Transportation Research
    Part C, 2024) use PI control to regulate a ride-hailing price gap.  This
    benchmark applies the same controlled-variable idea to the Firm2-Firm1
    average price gap available in this simulator.
    """

    def __init__(self, seed: Optional[int] = None, managed_keys: Optional[List[str]] = None):
        super().__init__(
            seed=seed,
            managed_keys=managed_keys,
            alpha_share=0.18,
            alpha_revenue=0.18,
            alpha_gap=0.25,
            cooldown_batches=2,
            step_scale=1.00,
            response_threshold=0.06,
        )
        self.target_gap = 0.0
        self.kp_gap = 0.85
        self.ki_gap = 0.18
        self.integral_gap_error = 0.0
        self.integral_clip = 2.0

    def act(
        self,
        city_base: float,
        city_pmin: float,
        hour: int,
        weather: str,
        city_pmile: Optional[float] = None,
        city_booking: Optional[float] = None,
        city_airport: Optional[float] = None,
    ) -> None:
        del hour, weather
        anchors = self._anchor_map(city_base, city_pmin, city_pmile, city_booking, city_airport)
        self._ensure_initialized(anchors)
        self.last_selected_key = "hold"
        self.last_direction = 0
        if self.cooldown > 0:
            self.cooldown -= 1
            self.last_reason = "cooldown"
            return

        gap_error = float(self.target_gap - self.ema_gap)
        self.integral_gap_error = float(np.clip(
            self.integral_gap_error + gap_error,
            -self.integral_clip,
            self.integral_clip,
        ))
        control = self.kp_gap * gap_error + self.ki_gap * self.integral_gap_error
        self.last_response_score = control
        if abs(control) <= self.response_threshold:
            self.last_reason = "pi_gap_hold"
            return

        direction = 1 if control > 0.0 else -1
        key = self._select_coefficient(direction)
        if key is None:
            self.last_reason = "no_room"
            return
        moved = self._apply_step(key, direction)
        self.last_selected_key = key if moved else "hold"
        self.last_direction = int(direction) if moved else 0
        self.last_reason = "pi_price_gap" if moved else "bounded"
        if moved:
            self.cooldown = self.cooldown_H


class FirmRegionSupplyDemandPricer(FirmAdaptiveBestResponsePricer):
    """Regional supply-demand benchmark using context buckets as pseudo-regions.

    Literature basis: Shi, Lu, and Cao (Applied Intelligence, 2024) segment the
    ride-hailing market and set regional prices from local demand-supply
    imbalance.  This simulator lacks explicit spatial cells, so peak/weather
    contexts and smoothed choice/completion shares serve as coarse operational
    regions and local imbalance proxies.
    """

    def __init__(self, seed: Optional[int] = None, managed_keys: Optional[List[str]] = None):
        super().__init__(
            seed=seed,
            managed_keys=managed_keys,
            alpha_share=0.22,
            alpha_revenue=0.18,
            alpha_gap=0.18,
            k_share=1.10,
            k_gap=0.20,
            k_revenue=0.15,
            k_supply=0.95,
            cooldown_batches=2,
            step_scale=1.15,
            response_threshold=0.08,
        )

    def act(
        self,
        city_base: float,
        city_pmin: float,
        hour: int,
        weather: str,
        city_pmile: Optional[float] = None,
        city_booking: Optional[float] = None,
        city_airport: Optional[float] = None,
    ) -> None:
        anchors = self._anchor_map(city_base, city_pmin, city_pmile, city_booking, city_airport)
        self._ensure_initialized(anchors)
        self.last_selected_key = "hold"
        self.last_direction = 0
        if self.cooldown > 0:
            self.cooldown -= 1
            self.last_reason = "cooldown"
            return

        peak = (7 <= int(hour) < 10) or (16 <= int(hour) < 19)
        bad_weather = str(weather).lower() in ("rain", "snow", "storm")
        context_pressure = 0.08 * float(peak) + 0.06 * float(bad_weather)
        demand_index = float(np.clip(self.ema_choice_share / max(self.target_share, 1e-6), 0.25, 2.0))
        realized_supply_index = float(np.clip(
            self.ema_completed_share / max(self.target_share, 1e-6)
            + 0.50 * max(0.0, self.ema_idle_share - 0.20),
            0.25,
            2.0,
        ))
        imbalance = float(np.clip((demand_index / max(realized_supply_index, 1e-6)) - 1.0, -1.0, 1.0))
        score = imbalance + context_pressure
        self.last_response_score = score
        if abs(score) <= self.response_threshold:
            self.last_reason = "regional_hold"
            return

        direction = 1 if score > 0.0 else -1
        key_priority = ("per_mile", "base_fare", "per_minute", "booking_fee", "airport_fee")
        key = next((k for k in key_priority if k in self.managed_keys
                    and getattr(self.overrides, k, None) is not None
                    and (not getattr(self, 'operational_clock', False) or
                         (getattr(self.overrides, k) < self.config.bounds[k][1] if direction > 0
                          else getattr(self.overrides, k) > self.config.bounds[k][0]))), None)
        if key is None:
            key = self._select_coefficient(direction)
        if key is None:
            self.last_reason = "no_room"
            return
        moved = self._apply_step(key, direction)
        self.last_selected_key = key if moved else "hold"
        self.last_direction = int(direction) if moved else 0
        self.last_reason = "regional_supply_demand" if moved else "bounded"
        if moved:
            self.cooldown = self.cooldown_H


class FirmQueueServiceThresholdPricer(FirmAdaptiveBestResponsePricer):
    """Queue/service-quality threshold benchmark for dynamic service pricing.

    Literature basis: recent queueing models of ride-hailing dynamic service
    pricing choose prices from system state, including congestion, service
    completion, and retrial/blocked demand.  This benchmark implements the same
    state-threshold structure with the simulator's wait, fulfillment,
    acceptance, utilization, and idle-driver signals.
    """

    def __init__(self, seed: Optional[int] = None, managed_keys: Optional[List[str]] = None):
        super().__init__(
            seed=seed,
            managed_keys=managed_keys,
            alpha_share=0.25,
            alpha_revenue=0.15,
            alpha_gap=0.15,
            cooldown_batches=2,
            step_scale=1.00,
            response_threshold=0.07,
        )
        self.target_fulfillment = 0.86
        self.target_acceptance = 0.72
        self.target_wait = 7.0

    def act(
        self,
        city_base: float,
        city_pmin: float,
        hour: int,
        weather: str,
        city_pmile: Optional[float] = None,
        city_booking: Optional[float] = None,
        city_airport: Optional[float] = None,
    ) -> None:
        del hour, weather
        anchors = self._anchor_map(city_base, city_pmin, city_pmile, city_booking, city_airport)
        self._ensure_initialized(anchors)
        self.last_selected_key = "hold"
        self.last_direction = 0
        if self.cooldown > 0:
            self.cooldown -= 1
            self.last_reason = "cooldown"
            return

        service_pressure = float(np.clip(
            0.40 * max(0.0, self.target_fulfillment - self.ema_fulfillment) / max(self.target_fulfillment, 1e-6)
            + 0.25 * max(0.0, self.target_acceptance - self.ema_acceptance) / max(self.target_acceptance, 1e-6)
            + 0.25 * max(0.0, self.ema_wait - self.target_wait) / max(self.target_wait, 1e-6)
            + 0.20 * max(0.0, self.ema_utilization - 0.78) / 0.22,
            0.0,
            1.0,
        ))
        idle_supply = float(np.clip(self.ema_idle_share - 0.28, 0.0, 1.0))
        demand_shortfall = float(np.clip(self.target_share - self.ema_choice_share, 0.0, 1.0))
        score = service_pressure - 0.45 * idle_supply - 0.55 * demand_shortfall
        self.last_response_score = score
        if abs(score) <= self.response_threshold:
            self.last_reason = "queue_service_hold"
            return

        direction = 1 if score > 0.0 else -1
        key = self._select_coefficient(direction)
        if key is None:
            self.last_reason = "no_room"
            return
        moved = self._apply_step(key, direction)
        self.last_selected_key = key if moved else "hold"
        self.last_direction = int(direction) if moved else 0
        self.last_reason = "queue_service_threshold" if moved else "bounded"
        if moved:
            self.cooldown = self.cooldown_H


class FirmSurgeDriverIncentivePricer(FirmAdaptiveBestResponsePricer):
    """Two-sided surge/incentive benchmark.

    Literature basis: Chen, Zheng, Ke, and Yang (Transportation Research Part B,
    2020) jointly study surge pricing, commission, and incentives for on-demand
    ride services.  The benchmark exposes a driver incentive multiplier consumed
    by the driver supply layer and uses rider-facing surge steps when demand is
    high or when supply needs to be rationed.
    """

    refresh_supply_stress_on_update = True

    def __init__(self, seed: Optional[int] = None, managed_keys: Optional[List[str]] = None):
        super().__init__(
            seed=seed,
            managed_keys=managed_keys,
            alpha_share=0.22,
            alpha_revenue=0.20,
            alpha_gap=0.18,
            cooldown_batches=2,
            step_scale=1.10,
            response_threshold=0.08,
        )
        self.supply_incentive_multiplier = 1.0
        self.supply_incentive_step = 0.025
        self.supply_incentive_bounds = (0.90, 1.15)

    def act(
        self,
        city_base: float,
        city_pmin: float,
        hour: int,
        weather: str,
        city_pmile: Optional[float] = None,
        city_booking: Optional[float] = None,
        city_airport: Optional[float] = None,
    ) -> None:
        del hour, weather
        anchors = self._anchor_map(city_base, city_pmin, city_pmile, city_booking, city_airport)
        self._ensure_initialized(anchors)
        self.last_selected_key = "hold"
        self.last_direction = 0
        if self.cooldown > 0:
            self.cooldown -= 1
            self.last_reason = "cooldown"
            return

        demand_pressure = float(np.clip(self.ema_choice_share - self.target_share, -1.0, 1.0))
        supply_stress = float(np.clip(self.last_supply_stress, 0.0, 1.0))
        revenue_gap = 0.0 if self.revenue_target is None else float(np.clip(
            (self.revenue_target - self.ema_revenue) / (self.revenue_target or float(getattr(self, 'currency_unit', 1.))),
            -1.0,
            1.0,
        ))
        if supply_stress > 0.22:
            self.supply_incentive_multiplier = float(np.clip(
                self.supply_incentive_multiplier + self.supply_incentive_step,
                *self.supply_incentive_bounds,
            ))
        elif self.ema_idle_share > 0.32:
            self.supply_incentive_multiplier = float(np.clip(
                self.supply_incentive_multiplier - self.supply_incentive_step,
                *self.supply_incentive_bounds,
            ))

        incentive_pressure = float(self.supply_incentive_multiplier - 1.0)
        score = (
            0.65 * demand_pressure
            + 0.40 * supply_stress
            + 0.30 * revenue_gap
            + (-0.20 if getattr(self, 'operational_clock', False) else 0.20) * self.ema_gap
            + 0.40 * incentive_pressure
        )
        self.last_response_score = score
        if abs(score) <= self.response_threshold:
            self.last_reason = "surge_incentive_hold"
            return

        direction = 1 if score > 0.0 else -1
        key = self._select_coefficient(direction)
        if key is None:
            self.last_reason = "no_room"
            return
        moved = self._apply_step(key, direction)
        self.last_selected_key = key if moved else "hold"
        self.last_direction = int(direction) if moved else 0
        self.last_reason = "surge_driver_incentive" if moved else "bounded"
        if moved:
            self.cooldown = self.cooldown_H


class FirmMPCGridPricer(FirmAdaptiveBestResponsePricer):
    """Short-horizon grid-search control benchmark.

    Literature basis: Nourinejad and Ramezani (Transportation Research Part B,
    2020) use model-predictive control for ride-sourcing fare/wage decisions.
    This benchmark is the simulator-compatible receding-horizon analogue: score
    all feasible one-step rider-fare moves against a local profit/service
    objective and apply the best move.
    """

    refresh_supply_stress_on_update = True

    def __init__(self, seed: Optional[int] = None, managed_keys: Optional[List[str]] = None):
        super().__init__(
            seed=seed,
            managed_keys=managed_keys,
            alpha_share=0.20,
            alpha_revenue=0.20,
            alpha_gap=0.20,
            cooldown_batches=1,
            step_scale=1.00,
            response_threshold=0.03,
        )

    def _candidate_score(self, key: str, direction: int) -> float:
        unit = float(getattr(self, 'currency_unit', 1.)) if getattr(self, 'operational_clock', False) else 1.
        step = float(self.config.step[key]) / unit
        revenue = self.ema_revenue / unit
        reference = None if self.revenue_target is None else self.revenue_target / unit
        price_effect = float(direction) * step
        relative_step = float(np.clip(price_effect / max(1.0, revenue), -0.25, 0.25))
        own_demand = float(np.clip(self.ema_choice_share, 0.0, 1.0))
        predicted_choice_share = float(np.clip(own_demand - 0.80 * relative_step, 0.0, 1.0))
        predicted_completed_share = float(np.clip(
            self.ema_completed_share - 0.50 * max(0.0, relative_step) + 0.20 * max(0.0, -relative_step),
            0.0,
            1.0,
        ))
        supply_stress = float(np.clip(self.last_supply_stress, 0.0, 1.0))
        revenue_gap = 0.0 if self.revenue_target is None else float(np.clip(
            (self.revenue_target - self.ema_revenue) / (self.revenue_target or float(getattr(self, 'currency_unit', 1.))),
            -1.0,
            1.0,
        ))
        gap_after = self.ema_gap + price_effect
        revenue_bonus = (revenue + price_effect) * predicted_choice_share / max(reference or 1.0, 1.0)
        unmet_demand_penalty = max(0.0, self.target_share - predicted_choice_share)
        service_penalty = (
            0.55 * supply_stress
            + 0.25 * max(0.0, self.ema_wait - 7.0) / 7.0
            + 0.35 * max(0.0, self.target_share - predicted_completed_share)
        )
        recovery_bonus = 0.20 * revenue_gap * float(direction)
        gap_penalty = 0.20 * abs(gap_after)
        volatility_penalty = 0.05 * abs(relative_step)
        return float(revenue_bonus + recovery_bonus - unmet_demand_penalty - service_penalty - gap_penalty - volatility_penalty)

    def act(
        self,
        city_base: float,
        city_pmin: float,
        hour: int,
        weather: str,
        city_pmile: Optional[float] = None,
        city_booking: Optional[float] = None,
        city_airport: Optional[float] = None,
    ) -> None:
        del hour, weather
        anchors = self._anchor_map(city_base, city_pmin, city_pmile, city_booking, city_airport)
        self._ensure_initialized(anchors)
        self.last_selected_key = "hold"
        self.last_direction = 0
        if self.cooldown > 0:
            self.cooldown -= 1
            self.last_reason = "cooldown"
            return

        # Hold has the same operating rewards and gap penalties as moves.
        # Comparing an absolute penalised score with zero freezes the controller.
        hold_score = self._candidate_score(self.managed_keys[0], 0) if getattr(self, 'operational_clock', False) else 0.
        best_score = hold_score
        best_key: Optional[str] = None
        best_direction = 0
        for key in self.managed_keys:
            if getattr(self.overrides, key, None) is None:
                continue
            for direction in (-1, 1):
                curr = float(getattr(self.overrides, key))
                lb, ub = self.config.bounds[key]
                if direction < 0 and curr <= lb:
                    continue
                if direction > 0 and curr >= ub:
                    continue
                score = self._candidate_score(key, direction)
                if score > best_score:
                    best_score = score
                    best_key = key
                    best_direction = direction

        self.last_response_score = best_score - hold_score
        # Scores already include the cost of changing tariff. A second fixed
        # response threshold discards profitable small-coefficient moves.
        threshold = 0. if getattr(self, 'operational_clock', False) else self.response_threshold
        if best_key is None or best_score - hold_score <= threshold:
            self.last_reason = "mpc_grid_hold"
            return
        moved = self._apply_step(best_key, best_direction)
        self.last_selected_key = best_key if moved else "hold"
        self.last_direction = int(best_direction) if moved else 0
        self.last_reason = "mpc_grid" if moved else "bounded"
        if moved:
            self.cooldown = self.cooldown_H


PROFIT_BEST_RESPONSE_GRID: Tuple[float, ...] = (
    0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20,
    1.30, 1.40, 1.50, 1.60, 1.80, 2.00,
)


class FirmPriceGapCatchUpPricer(FirmAdaptiveBestResponsePricer):
    """Training-only rival that mostly follows the observed public price gap.

    Unlike the five canonical benchmarks, this controller is deliberately a
    simple catch-up mechanism.  Ninety percent of its response score comes
    from Firm2-minus-Firm1 observed quote gap and ten percent from its own
    choice-share shortfall.  It changes one coefficient per decision.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        managed_keys: Optional[List[str]] = None,
        *,
        alpha_gap: float = 0.45,
        cooldown_batches: int = 2,
        step_scale: float = 1.00,
    ) -> None:
        super().__init__(
            seed=seed,
            managed_keys=managed_keys,
            alpha_share=0.12,
            alpha_revenue=0.10,
            alpha_gap=alpha_gap,
            cooldown_batches=cooldown_batches,
            step_scale=step_scale,
            response_threshold=0.035,
        )
        self.gap_response_weight = 0.90
        self.share_response_weight = 0.10

    def act(
        self,
        city_base: float,
        city_pmin: float,
        hour: int,
        weather: str,
        city_pmile: Optional[float] = None,
        city_booking: Optional[float] = None,
        city_airport: Optional[float] = None,
    ) -> None:
        del hour, weather
        anchors = self._anchor_map(
            city_base, city_pmin, city_pmile, city_booking, city_airport
        )
        self._ensure_initialized(anchors)
        self.last_selected_key = "hold"
        self.last_direction = 0
        if self.cooldown > 0:
            self.cooldown -= 1
            self.last_reason = "cooldown"
            return
        currency = float(max(1e-6, getattr(self, "currency_unit", 1.0)))
        normalized_gap = float(np.clip(self.ema_gap / currency, -1.5, 1.5))
        share_shortfall = float(np.clip(
            self.target_share - self.ema_choice_share, -0.5, 0.5
        ))
        score = (
            self.gap_response_weight * normalized_gap
            + self.share_response_weight * share_shortfall
        )
        self.last_response_score = score
        if abs(score) <= self.response_threshold:
            self.last_reason = "price_gap_catchup_hold"
            return
        # Positive Firm2-minus-Firm1 gap means Firm 2 is dearer: cut price.
        direction = -1 if score > 0.0 else 1
        key = self._select_coefficient(direction)
        if key is None:
            self.last_reason = "no_room"
            return
        moved = self._apply_step(key, direction)
        self.last_selected_key = key if moved else "hold"
        self.last_direction = int(direction) if moved else 0
        self.last_reason = "price_gap_catchup" if moved else "bounded"
        if moved:
            self.cooldown = self.cooldown_H


class FirmFastPriceGapCatchUpPricer(FirmPriceGapCatchUpPricer):
    """Faster, larger-step catch-up rival used for Stage 3 stress coverage."""

    def __init__(self, seed: Optional[int] = None, managed_keys: Optional[List[str]] = None):
        super().__init__(
            seed=seed,
            managed_keys=managed_keys,
            alpha_gap=0.70,
            cooldown_batches=1,
            step_scale=1.50,
        )


class FirmTwinProfitOptimizerMixin:
    """Interface for simulator-backed Firm 2 own-profit best responses.

    Candidate evaluation belongs to :class:`Core`, which owns the digital
    twin and can transactionally restore rider, supply, loyalty, and RNG state.
    These controllers only declare their information/horizon contract and
    apply the selected 13-point grid tariff. Under the V17 M6 flag, that grid
    is the distance/time diagonal and all fixed components stay on the card.
    """

    source_benchmark = "myopic"
    profit_horizon_days = 1
    anticipate_firm1_reaction = False
    profit_candidate_grid = PROFIT_BEST_RESPONSE_GRID

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pending_profit_multiplier: Optional[float] = None
        self.last_profit_candidate = 1.0
        self.last_profit_improvement = 0.0
        self.last_profit_standard_error = 0.0
        self.last_profit_candidate_means: Dict[str, float] = {}
        self.profit_plan_decisions = 0
        self.firm1_reaction_table: Dict[str, List[float]] = {}
        self._last_own_grid_multiplier: Optional[float] = None

    @staticmethod
    def _tariff_multiplier(
        tariff: Mapping[str, float], anchors: Mapping[str, float]
    ) -> float:
        ratios = [
            float(tariff[key]) / max(1e-9, float(anchors[key]))
            for key in tariff
            if key in anchors and float(anchors[key]) > 0.0
        ]
        return float(np.mean(ratios)) if ratios else 1.0

    def observe_firm1_posted_tariff(
        self,
        *,
        firm1_tariff: Mapping[str, float],
        anchors: Mapping[str, float],
    ) -> None:
        """Update fictitious-play data using only already-posted public prices."""
        firm1_multiplier = self._tariff_multiplier(firm1_tariff, anchors)
        if self._last_own_grid_multiplier is not None:
            key = f"{self._last_own_grid_multiplier:.2f}"
            values = self.firm1_reaction_table.setdefault(key, [])
            values.append(float(firm1_multiplier))
            if len(values) > 64:
                del values[:-64]

    def predicted_firm1_multiplier(
        self, candidate: float, fallback: float
    ) -> float:
        if not self.anticipate_firm1_reaction or not self.firm1_reaction_table:
            return float(fallback)
        key = min(
            self.firm1_reaction_table,
            key=lambda value: abs(float(value) - float(candidate)),
        )
        values = self.firm1_reaction_table[key]
        return float(np.mean(values)) if values else float(fallback)

    def set_profit_plan(
        self,
        *,
        multiplier: Optional[float],
        improvement: float,
        standard_error: float,
        candidate_means: Mapping[float, float],
    ) -> None:
        self.pending_profit_multiplier = (
            float("nan") if multiplier is None else float(multiplier)
        )
        if multiplier is not None:
            self.last_profit_candidate = float(multiplier)
        self.last_profit_improvement = float(improvement)
        self.last_profit_standard_error = float(standard_error)
        self.last_profit_candidate_means = {
            f"{float(key):.2f}": float(value)
            for key, value in candidate_means.items()
        }
        self.profit_plan_decisions += 1

    def act(
        self,
        city_base: float,
        city_pmin: float,
        hour: int,
        weather: str,
        city_pmile: Optional[float] = None,
        city_booking: Optional[float] = None,
        city_airport: Optional[float] = None,
    ) -> None:
        del hour, weather
        anchors = self._anchor_map(
            city_base, city_pmin, city_pmile, city_booking, city_airport
        )
        self._ensure_initialized(anchors)
        target = self.pending_profit_multiplier
        self.pending_profit_multiplier = None
        self.last_selected_key = "hold"
        self.last_direction = 0
        if target is None:
            self.last_reason = "profit_plan_missing_hold"
            return
        if not np.isfinite(target):
            self.last_reason = "profit_hysteresis_hold"
            return
        before = self._tariff_multiplier(
            {key: float(getattr(self.overrides, key)) for key in self.managed_keys},
            anchors,
        )
        changed = False
        two_lever = bool(getattr(self, "rivals_two_lever", False))
        keys = (
            ("per_mile", "per_minute")
            if two_lever
            else tuple(self.managed_keys)
        )
        if two_lever:
            for key in ("base_fare", "booking_fee", "airport_fee"):
                setattr(self.overrides, key, None)
        for key in keys:
            lower, upper = self.config.bounds[key]
            value = float(np.clip(float(anchors[key]) * float(target), lower, upper))
            changed |= abs(value - float(getattr(self.overrides, key))) > 1e-12
            setattr(self.overrides, key, value)
        self._last_own_grid_multiplier = float(target)
        self.last_direction = int(np.sign(float(target) - before)) if changed else 0
        self.last_selected_key = "tariff_grid" if changed else "hold"
        self.last_reason = (
            f"profit_best_response_{self.source_benchmark}"
            if changed
            else "profit_hysteresis_hold"
        )


class FirmProfitOptimizedPIPriceGapPricer(
    FirmTwinProfitOptimizerMixin, FirmPIPriceGapPricer
):
    source_benchmark = "pi_price_gap"
    profit_horizon_days = 1


class FirmProfitOptimizedMPCGridPricer(
    FirmTwinProfitOptimizerMixin, FirmMPCGridPricer
):
    source_benchmark = "mpc_grid"
    profit_horizon_days = 3


class FirmProfitOptimizedRegionSupplyDemandPricer(
    FirmTwinProfitOptimizerMixin, FirmRegionSupplyDemandPricer
):
    source_benchmark = "region_supply_demand"
    profit_horizon_days = 3


class FirmProfitOptimizedQueueServiceThresholdPricer(
    FirmTwinProfitOptimizerMixin, FirmQueueServiceThresholdPricer
):
    source_benchmark = "queue_service_threshold"
    profit_horizon_days = 3


class FirmProfitOptimizedSurgeDriverIncentivePricer(
    FirmTwinProfitOptimizerMixin, FirmSurgeDriverIncentivePricer
):
    source_benchmark = "surge_driver_incentive"
    profit_horizon_days = 5


class FirmMyopicProfitBestResponsePricer(FirmProfitOptimizedPIPriceGapPricer):
    source_benchmark = "myopic_one_day"


class FirmForwardProfitBestResponsePricer(FirmProfitOptimizedMPCGridPricer):
    source_benchmark = "forward_k_day"


class FirmAnticipatingProfitBestResponsePricer(FirmProfitOptimizedMPCGridPricer):
    source_benchmark = "anticipating_fictitious_play"
    anticipate_firm1_reaction = True



class FirmAggressiveAdaptiveBestResponsePricer(FirmAdaptiveBestResponsePricer):
    """Stress-test variant with faster signal updates and shorter cooldown."""

    def __init__(self, seed: Optional[int] = None, managed_keys: Optional[List[str]] = None):
        super().__init__(
            seed=seed,
            managed_keys=managed_keys,
            alpha_share=0.35,
            alpha_revenue=0.35,
            alpha_gap=0.35,
            cooldown_batches=1,
            step_scale=1.50,
            response_threshold=0.04,
        )

class FirmMemoryBestResponsePricer(FirmAdaptiveBestResponsePricer):
    """Best-response rival with memory: slow signals, retaliation and accommodation.

    ``price_gap_mean`` passed to :meth:`update` is Firm2 minus Firm1 (positive
    when Firm 1 undercuts).  State: consecutive undercut / premium steps, the
    remaining retaliation hold, and a fast gap EMA used while matching.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        managed_keys: Optional[List[str]] = None,
        *,
        alpha_slow: float = 0.03,            # per-day alpha with operational_clock (~23-day half-life)
        undercut_threshold: float = 1.00,    # dollars of Firm 1 undercut that count
        lag_steps: int = 6,                  # one day of sustained undercut before retaliating
        hold_mean_steps: int = 30,           # geometric hold, mean five days
        accommodation_steps: int = 12,       # two days of Firm 1 premium before following up
        accommodation_cap: float = 0.5,      # follow at most this share of the premium gap
        operational_clock: bool = False,
    ):
        super().__init__(
            seed=seed,
            managed_keys=managed_keys,
            alpha_share=float(alpha_slow),
            alpha_revenue=float(alpha_slow),
            alpha_gap=0.35,
            cooldown_batches=1,
            step_scale=1.50,
            response_threshold=0.04,
        )
        self.undercut_threshold = float(max(0.0, undercut_threshold))
        self.lag_steps = int(max(1, lag_steps))
        self.hold_mean_steps = int(max(1, hold_mean_steps))
        self.accommodation_steps = int(max(1, accommodation_steps))
        self.accommodation_cap = float(np.clip(accommodation_cap, 0.0, 1.0))
        self.undercut_steps = 0
        self.premium_steps = 0
        self.retaliation_left = 0
        self.retaliation_episodes = 0
        self.accommodation_moves = 0
        self.fast_gap = 0.0
        self.memory_state = "calm"
        # Revision 4 receives one lagged observation per market period, even
        # when its tariff is not due to change. Legacy artifacts count calls.
        self.operational_clock = bool(operational_clock)
        self.reference_window_steps = 6
        self._pre_retaliation_tariff = None

    def update(
        self,
        metrics: FirmMetrics,
        price_gap_mean: float,
        supply_state: Optional[Dict[str, float]] = None,
        supply_state_vector: Optional[np.ndarray] = None,
    ) -> None:
        super().update(metrics, price_gap_mean, supply_state=supply_state, supply_state_vector=supply_state_vector)
        gap = float(price_gap_mean) / float(getattr(self, 'currency_unit', 1.))
        fast_alpha = 1 - .5**(1/self.reference_window_steps) if self.operational_clock else .5
        self.fast_gap = (1-fast_alpha) * self.fast_gap + fast_alpha * gap
        self.undercut_steps = self.undercut_steps + 1 if gap > self.undercut_threshold else 0
        self.premium_steps = self.premium_steps + 1 if gap < -self.undercut_threshold else 0
        if self.operational_clock:
            self.retaliation_left = max(0, self.retaliation_left - 1)
        if self.retaliation_left == 0 and self.undercut_steps >= self.lag_steps:
            if self.operational_clock:
                if self._pre_retaliation_tariff is None:
                    self._pre_retaliation_tariff = {key: getattr(self.overrides, key) for key in self.managed_keys}
                self.undercut_steps = 0
                self.premium_steps = 0
            self.retaliation_left = int(self.rng.geometric(1.0 / float(self.hold_mean_steps)))
            self.retaliation_episodes += 1

    def act(
        self,
        city_base: float,
        city_pmin: float,
        hour: int,
        weather: str,
        city_pmile: Optional[float] = None,
        city_booking: Optional[float] = None,
        city_airport: Optional[float] = None,
    ) -> None:
        anchors = self._anchor_map(city_base, city_pmin, city_pmile, city_booking, city_airport)
        self._ensure_initialized(anchors)
        self.last_selected_key = "hold"
        self.last_direction = 0
        if self.retaliation_left > 0:
            # Retaliation: match the undercut (step down while still dearer),
            # then hold the match for the rest of the geometric window.
            if not self.operational_clock:
                self.retaliation_left -= 1
            self.memory_state = "retaliation"
            if self.fast_gap > 0.25 * self.undercut_threshold:
                key = self._select_coefficient(-1)
                if key is not None and self._apply_step(key, -1):
                    self.last_selected_key, self.last_direction = key, -1
                    self.last_reason = "retaliation_match"
                    return
            self.last_reason = "retaliation_hold"
            return
        if self.operational_clock and self._pre_retaliation_tariff is not None and self.undercut_steps == 0:
            for key, target in self._pre_retaliation_tariff.items():
                if target is None:
                    continue
                current = float(getattr(self.overrides, key))
                if current < target - 1e-10:
                    setattr(self.overrides, key, min(target, current + self.config.step[key]))
                    self.memory_state = 'recovery'
                    self.last_selected_key, self.last_direction = key, 1
                    self.last_reason = 'retaliation_recovery'
                    return
            self._pre_retaliation_tariff = None
            self.premium_steps = 0
        if self.premium_steps >= self.accommodation_steps and self.fast_gap < -self.undercut_threshold:
            # Accommodation: follow a sustained Firm 1 premium part of the way.
            self.memory_state = "accommodation"
            if abs(self.fast_gap) * self.accommodation_cap > self.undercut_threshold * 0.5:
                key = self._select_coefficient(+1)
                if key is not None and self._apply_step(key, +1):
                    self.last_selected_key, self.last_direction = key, +1
                    self.last_reason = "accommodation"
                    self.accommodation_moves += 1
                    self.cooldown = self.cooldown_H
                    return
            self.last_reason = "accommodation_hold"
            return
        self.memory_state = "calm"
        super().act(
            city_base=city_base, city_pmin=city_pmin, hour=hour, weather=weather,
            city_pmile=city_pmile, city_booking=city_booking, city_airport=city_airport,
        )


class FirmRandomWalkPricer(FirmHeuristicPricer):
    """Heuristic pricer with occasional bounded random exploration."""

    def __init__(self, seed: Optional[int] = None, managed_keys: Optional[List[str]] = None):
        super().__init__(seed=seed, managed_keys=managed_keys)
        self.exploration_prob = 0.20

    def act(
        self,
        city_base: float,
        city_pmin: float,
        hour: int,
        weather: str,
        city_pmile: Optional[float] = None,
        city_booking: Optional[float] = None,
        city_airport: Optional[float] = None,
    ):
        super().act(
            city_base=city_base,
            city_pmin=city_pmin,
            hour=hour,
            weather=weather,
            city_pmile=city_pmile,
            city_booking=city_booking,
            city_airport=city_airport,
        )

        if self.cooldown > 0 or float(self.rng.random()) >= self.exploration_prob:
            return

        choices = [k for k in self.managed_keys if getattr(self.overrides, k, None) is not None]
        if not choices:
            return

        key = str(self.rng.choice(choices))
        direction = int(self.rng.choice([-1, 1]))
        if key == "base_fare":
            self.overrides.base_fare = float(np.clip(float(self.overrides.base_fare) + direction * self.step_base_fare, *self.base_bounds))
        elif key == "per_minute":
            self.overrides.per_minute = float(np.clip(float(self.overrides.per_minute) + direction * self.step_per_minute, *self.pmin_bounds))
        elif key == "per_mile":
            self.overrides.per_mile = float(np.clip(float(self.overrides.per_mile) + direction * self.step_per_mile, *self.pmile_bounds))
        elif key == "booking_fee":
            self.overrides.booking_fee = float(np.clip(float(self.overrides.booking_fee) + direction * self.step_booking_fee, *self.booking_bounds))
        elif key == "airport_fee":
            self.overrides.airport_fee = float(np.clip(float(self.overrides.airport_fee) + direction * self.step_airport_fee, *self.airport_bounds))
        self.cooldown = self.cooldown_H

        
class FirmRLPricer:
    MAX_MANIPULATED_COEFFS = 5

    def _ensure_internal_state(self) -> None:
        """Backstop against partially initialized objects from stale environments."""
        if not hasattr(self, "opt_keys"):
            self.opt_keys = ["base_fare", "per_minute"]
        if len(self.opt_keys) > self.MAX_MANIPULATED_COEFFS:
            self.opt_keys = list(self.opt_keys[: self.MAX_MANIPULATED_COEFFS])
        if not hasattr(self, "config"):
            self.config = default_specs_for(self.opt_keys)
    
    def __init__(
        self,
        seed: Optional[int],
        opt_keys: List[str],
        state_frame_stack: int = 16,
        single_state_dim: int = 24,
        action_feature_dim: int = 20,
        constraint_dim: int = 0,
        response_dim: int = 0,
        opponent_forecast_dim: int = 0,
        feature_groups: Optional[Dict[str, Sequence[int]]] = None,
        incentive_levels: Sequence[float] = (1.0,),
        tariff_targets: Optional[Sequence[float]] = None,
        gamma: float = 0.999,
        gae_lambda: float = 0.99,
        state_action_mi_weight: float = 0.0,
        direct_actor_credit_weight: float = 0.0,
        critic_state_dim: int = 0,
        privileged_critic: bool = False,
        privileged_value_only: bool = False,
        persistent_recurrence: bool = False,
        counterfactual_q_coeff: float = 0.0,
        counterfactual_actor_weight: float = 0.0,
        potential_shaping_scale: float = 0.0,
        hard_kl_stop: bool = False,
        reward_rate_beta: float = 0.0,
        td_value_targets: bool = False,
        learning_corrections: bool = False,
        learner_revision: int = 4,
        value_component_dim: int = 2,
        independent_coefficient_actions: bool = True,
        level_step_actions: bool = False,
        hybrid_relative_actions: bool = False,
        action_contract: str = "legacy",
        relative_action_steps: Sequence[float] = HYBRID_RELATIVE_STEP_MAGNITUDES,
        tariff_structure: Optional[Mapping[str, Any]] = None,
        tariff_level_bounds: Optional[Tuple[float, float]] = None,
        entropy_coeff_start: Optional[float] = None,
    ):
        # The revised path selects exactly one coefficient and one absolute
        # target. The legacy path is retained verbatim for flags-off identity.
        self.independent_coefficient_actions = bool(
            independent_coefficient_actions
        )
        self.hybrid_relative_actions = bool(hybrid_relative_actions)
        self.action_contract = str(action_contract)
        if self.action_contract not in ("legacy", "two_lever_v1"):
            raise ValueError("action_contract must be legacy or two_lever_v1")
        self.two_lever_actions = self.action_contract == "two_lever_v1"
        # Revision 15: hold or a relative step of the tariff level L on a fixed
        # structure S; every coefficient is posted as anchor * S * L.
        self.level_step_actions = bool(level_step_actions)
        if sum((
            self.level_step_actions,
            self.independent_coefficient_actions,
            self.hybrid_relative_actions,
            self.two_lever_actions,
        )) > 1:
            raise ValueError(
                "level-step, absolute-target, hybrid-relative, and two-lever actions are exclusive"
            )
        if entropy_coeff_start is not None and not (
            np.isfinite(float(entropy_coeff_start)) and float(entropy_coeff_start) > 0.0
        ):
            raise ValueError("entropy_coeff_start must be finite and positive")
        self.opt_keys = list(opt_keys[: self.MAX_MANIPULATED_COEFFS])
        if len(opt_keys) > self.MAX_MANIPULATED_COEFFS:
            print(
                f"[FirmRLPricer] Received {len(opt_keys)} opt_keys; "
                f"limiting to first {self.MAX_MANIPULATED_COEFFS}: {self.opt_keys}"
            )
        self.config = default_specs_for(self.opt_keys)
        self.overrides = CoefficientOverrides()
        self.state_frame_stack = int(max(1, state_frame_stack))
        self._state_history: Deque[np.ndarray] = deque(maxlen=self.state_frame_stack)
        
        # Compatibility values retained for reports; absolute targets do not
        # decay or accumulate.
        self.base_step_scale = 1.0
        self.converged_step_scale = 1.0
        self.step_scale = 1.0
        self.repeat_action_decay = 1.0
        self.repeat_action_min_scale = 1.0
        self._last_applied_action = -1
        self._repeat_action_count = 0
        self.base_max_relative_dev = float("inf")
        self.converged_max_relative_dev = float("inf")
        self.max_relative_dev = float("inf")
        # Keep supply-side control available as an internal multiplier, but do
        # not expose it as a PPO action by default.  Early optimization is
        # easier when PPO only credits rider-facing price moves.
        self.supply_incentive_multiplier = 1.0
        self.supply_step = 0.025
        self.supply_min_multiplier = 0.90
        self.supply_max_multiplier = 1.15
        self.action_feature_dim = int(max(0, action_feature_dim))
        self.last_state_features = None
        self.last_action_features = None
        self.last_action_descriptor = ActionDescriptor()
        self.last_action_was_saturated = False
        self.last_action_was_zero_effect = False
        self.last_action_step_units: Dict[str, float] = {}
        self._last_action_target = "hold"
        self._last_action_direction = 0
        self._reversal_count = 0
        self.recovery_share_threshold = 0.25
        self.recovery_gap_threshold = -0.05
        self.aggressive_actions = set()
        self.allow_aggressive_actions = True
        
        if self.two_lever_actions:
            if tariff_targets:
                raise ValueError("two_lever_v1 does not use tariff_targets")
            if tariff_structure is not None or tariff_level_bounds is not None:
                raise ValueError("two_lever_v1 does not use a tariff structure")
            (
                self.action_to_steps,
                self.action_to_two_lever_cell,
                self.action_keys,
            ) = build_two_lever_action_space(self.opt_keys)
            self.action_to_multiplier = {}
            self.action_to_target_key = {}
            self.action_to_level_step = {}
            self.action_to_relative_step = {}
            self.tariff_structure = None
            self.tariff_level_bounds = None
        elif self.hybrid_relative_actions:
            if tariff_targets:
                raise ValueError("hybrid relative actions do not use tariff_targets")
            if tariff_structure is not None or tariff_level_bounds is not None:
                raise ValueError("hybrid relative actions do not use a tariff structure")
            (
                self.action_to_steps,
                self.action_to_relative_step,
                self.action_to_target_key,
                self.action_keys,
            ) = build_hybrid_relative_action_space(
                self.opt_keys,
                self.MAX_MANIPULATED_COEFFS,
                step_magnitudes=relative_action_steps,
            )
            self.action_to_multiplier = {}
            self.action_to_level_step = {}
            self.tariff_structure = None
            self.tariff_level_bounds = None
        elif self.level_step_actions:
            if tariff_targets:
                raise ValueError("level-step actions do not use tariff_targets")
            if tariff_structure is None or tariff_level_bounds is None:
                raise ValueError(
                    "level-step actions require tariff_structure and tariff_level_bounds"
                )
            self.tariff_structure: Optional[Dict[str, Any]] = validate_tariff_structure(
                tariff_structure, self.opt_keys, tariff_level_bounds
            )
            self.tariff_level_bounds: Optional[Tuple[float, float]] = (
                snap_tariff_level(tariff_level_bounds[0]),
                snap_tariff_level(tariff_level_bounds[1]),
            )
            (
                self.action_to_steps,
                self.action_to_level_step,
                self.action_keys,
            ) = build_level_step_action_space(
                self.opt_keys, self.MAX_MANIPULATED_COEFFS
            )
            self.action_to_multiplier = {}
            self.action_to_target_key = {}
            self.action_to_relative_step = {}
        else:
            action_builder = (
                build_independent_coefficient_action_space
                if self.independent_coefficient_actions
                else build_absolute_tariff_action_space
            )
            (
                self.action_to_steps,
                self.action_to_multiplier,
                self.action_to_target_key,
                self.action_keys,
            ) = action_builder(
                self.opt_keys,
                self.MAX_MANIPULATED_COEFFS,
                multipliers=tariff_targets,
            )
            self.action_to_level_step: Dict[int, float] = {}
            self.action_to_relative_step: Dict[int, float] = {}
            self.tariff_structure = None
            self.tariff_level_bounds = None
        # Posted tariff level; None until a level is posted (and after a
        # comparator posts a tariff outside the level contract).
        self.tariff_level: Optional[float] = None
        self.tariff_targets: Tuple[float, ...] = tuple(sorted(set(self.action_to_multiplier.values())))
        self.base_action_count = len(self.action_to_steps)
        levels = [float(x) for x in (incentive_levels or (1.0,))] or [1.0]
        if (
            self.independent_coefficient_actions
            or self.level_step_actions
            or self.hybrid_relative_actions
            or self.two_lever_actions
        ):
            if len(levels) != 1 or abs(levels[0] - 1.0) > 1e-9:
                raise ValueError(
                    "independent coefficient actions require driver_incentive_levels=(1.0,)"
                )
        elif abs(levels[0] - 1.0) > 1e-9:
            levels = [1.0] + [x for x in levels if abs(x - 1.0) > 1e-9]
        self.incentive_levels: Tuple[float, ...] = tuple(levels)
        self.action_to_incentive: Dict[int, float] = {a: 1.0 for a in self.action_to_steps}
        if not self.independent_coefficient_actions and len(self.incentive_levels) > 1:
            base_steps = dict(self.action_to_steps)
            base_multipliers = dict(self.action_to_multiplier)
            for level_index, level in enumerate(self.incentive_levels[1:], start=1):
                for base_action, steps in base_steps.items():
                    action = int(base_action) + level_index * self.base_action_count
                    self.action_to_steps[action] = dict(steps)
                    if base_action in base_multipliers:
                        self.action_to_multiplier[action] = float(
                            base_multipliers[base_action]
                        )
                    self.action_to_incentive[action] = float(level)
        action_dim = len(self.action_to_steps)
        target_indices = None
        magnitude_indices = None
        if self.independent_coefficient_actions:
            target_indices = [0]
            magnitude_indices = [-1]
            magnitude_lookup = {
                value: index for index, value in enumerate(self.tariff_targets)
            }
            for action in range(1, action_dim):
                target_indices.append(
                    1 + self.action_keys.index(self.action_to_target_key[action])
                )
                magnitude_indices.append(
                    magnitude_lookup[self.action_to_multiplier[action]]
                )
        elif self.hybrid_relative_actions:
            target_indices = [0]
            magnitude_indices = [-1]
            targets = ["whole_tariff", *self.action_keys]
            deltas = tuple(sorted(set(self.action_to_relative_step.values())))
            magnitude_lookup = {value: index for index, value in enumerate(deltas)}
            for action in range(1, action_dim):
                target_indices.append(
                    1 + targets.index(self.action_to_target_key[action])
                )
                magnitude_indices.append(
                    magnitude_lookup[self.action_to_relative_step[action]]
                )
        
        # Each compact public frame contains time/weather, own tariff position,
        # delayed own outcomes, current demand, public rival quote gaps, and
        # prior-action timing. Ordered frame stacking supplies trajectory and
        # momentum without exposing rival internals or hand-built trends.
        self.single_state_dim = int(max(1, single_state_dim))
        state_dim = self.single_state_dim * self.state_frame_stack

        # Initialize PPO agent.
        self.agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            gamma=float(gamma),
            lam=float(gae_lambda),
            hidden_dim=192,
            clip_eps=0.20,
            final_clip_eps=0.10,
            max_grad_norm=0.8,
            ent_coeff=(
                float(entropy_coeff_start)
                if entropy_coeff_start is not None
                else 0.040 if learning_corrections and learner_revision >= 4 else 0.010
            ),
            min_ent_coeff=0.002 if learning_corrections and learner_revision >= 4 else 0.0005,
            max_ent_coeff=(
                float(entropy_coeff_start)
                if entropy_coeff_start is not None
                else 0.040
            ),
            target_entropy_fraction=0.30,
            entropy_adaptation_rate=0.75,
            ent_decay=0.992,
            target_kl=0.020,
            max_lr=6.0e-4,
            value_clip_eps=0.30,
            initial_exploration_rate=0.30,
            final_exploration_rate=0.05,
            action_feature_dim=self.action_feature_dim,
            constraint_dim=int(max(0, constraint_dim)),
            response_dim=int(max(0, response_dim)),
            opponent_forecast_dim=int(max(0, opponent_forecast_dim)),
            response_coeff=0.0,
            # The Core training loop now records the exact discounted return for
            # each held pricing decision.  A second future-reward redistribution
            # would double count the same delayed outcomes.
            delayed_reward_horizon=1,
            delayed_reward_blend=0.0,
            single_state_dim=self.single_state_dim,
            frame_stack=self.state_frame_stack,
            state_action_mi_coeff=float(max(0.0, state_action_mi_weight)),
            direct_actor_credit_weight=float(
                np.clip(direct_actor_credit_weight, 0.0, 1.0)
            ),
            feature_groups=feature_groups,
            critic_state_dim=(
                int(critic_state_dim) if int(critic_state_dim) > 0 else state_dim
            ),
            privileged_critic=bool(privileged_critic),
            privileged_value_only=bool(privileged_value_only),
            persistent_recurrence=bool(persistent_recurrence),
            enable_counterfactual_q=float(counterfactual_q_coeff) > 0.0,
            action_q_coeff=float(max(0.0, counterfactual_q_coeff)),
            counterfactual_actor_weight=float(
                np.clip(counterfactual_actor_weight, 0.0, 1.0)
            ),
            potential_shaping_scale=float(max(0.0, potential_shaping_scale)),
            hard_kl_stop=bool(hard_kl_stop),
            reward_rate_beta=float(np.clip(reward_rate_beta, 0.0, 1.0)),
            td_value_targets=bool(td_value_targets),
            learning_corrections=bool(learning_corrections),
            learner_revision=learner_revision,
            value_component_dim=value_component_dim,
            action_target_indices=target_indices,
            action_magnitude_indices=magnitude_indices,
            scheduled_exploration=bool(td_value_targets),
            exploration_fraction=1.00,
            exploration_warmup_fraction=0.05,
            min_action_visits=1,
        )
        
    def observe_state(self, state_features: np.ndarray, action_features: Optional[np.ndarray] = None) -> None:
        """Cache the latest observation/action features for parity diagnostics."""
        self.last_state_features = np.asarray(state_features, dtype=np.float32).copy()
        self.last_action_features = (
            None if action_features is None else np.asarray(action_features, dtype=np.float32).copy()
        )
    
    def action_steps(self, action: int) -> Dict[str, int]:
        """Return the decoded bounded coefficient intervention for diagnostics."""
        return dict(self.action_to_steps.get(int(action), {}))

    def action_label(self, action: int) -> str:
        """Compact label for the active action-space contract."""
        if self.two_lever_actions:
            cell = self.action_to_two_lever_cell.get(int(action))
            return (
                "invalid"
                if cell is None
                else f"cell:distance={cell[0]:.1f},time={cell[1]:.1f}"
            )
        if int(action) == 0:
            return "hold"
        if self.level_step_actions:
            step = self.action_to_level_step.get(int(action))
            return "invalid" if step is None else f"level_step:{step:+.2f}"
        if self.hybrid_relative_actions:
            step = self.action_to_relative_step.get(int(action))
            target = self.action_to_target_key.get(int(action))
            return (
                "invalid"
                if step is None or target is None
                else f"{target}_step:{step:+.2f}"
            )
        multiplier = self.action_to_multiplier.get(int(action))
        if not self.independent_coefficient_actions:
            return (
                "invalid"
                if multiplier is None
                else f"tariff_target:{multiplier:.3f}x_market"
            )
        target = self.action_to_target_key.get(int(action))
        return (
            "invalid"
            if multiplier is None or target is None
            else f"{target}:{multiplier:.3f}x_anchor"
        )
        
    def last_action_magnitude(self) -> float:
        """Mean realized movement in standard coefficient-step units.

        This is invariant to the very different numerical ranges of airport,
        per-minute, and fixed-fee coefficients.  Moving every managed lever by
        one configured step has magnitude 1; one lever has magnitude 1/5.
        """
        return float(
            np.clip(
                sum(abs(v) for v in getattr(self, "last_action_step_units", {}).values())
                / max(1, len(self.opt_keys)),
                0.0,
                1.0,
            )
        )
        
    def reset_state_history(self) -> None:
        """Clear stacked observation/action history at train/eval episode boundaries."""
        self._state_history.clear()
        self._last_applied_action = -1
        self._repeat_action_count = 0
        self._last_action_target = "hold"
        self._last_action_direction = 0
        self._reversal_count = 0
        self.last_action_normalized_gap = {}
        self.last_action_step_units = {}
        self.last_action_descriptor = ActionDescriptor()
        self.last_action_was_saturated = False
        self.last_action_was_zero_effect = False
        self.agent.reset_learning_context()

    def snapshot_execution_state(self) -> Dict[str, Any]:
        """Capture mutable inference state without copying policy parameters."""
        attribute_names = (
            "last_state_features",
            "last_action_features",
            "last_action_descriptor",
            "last_action_was_saturated",
            "last_action_was_zero_effect",
            "last_action_normalized_gap",
            "last_action_step_units",
            "last_action_steps",
            "_last_applied_action",
            "_repeat_action_count",
            "_last_action_target",
            "_last_action_direction",
            "_reversal_count",
            "_last_share_hint",
            "_last_gap_hint",
            "_last_fulfillment_hint",
            "supply_incentive_multiplier",
            "tariff_level",
        )
        return {
            "history": [np.asarray(frame, dtype=np.float32).copy() for frame in self._state_history],
            "attributes": {
                name: copy.deepcopy(getattr(self, name))
                for name in attribute_names
                if hasattr(self, name)
            },
            "agent": {
                "action_visits": self.agent.action_visits.copy(),
                "action_ever_feasible": self.agent.action_ever_feasible.copy(),
                "last_action_exploration_rate": float(self.agent.last_action_exploration_rate),
                "last_continuous_magnitude": float(self.agent.last_continuous_magnitude),
                "last_constraint_values": self.agent.last_constraint_values.detach().clone(),
                "last_risk_value": self.agent.last_risk_value.detach().clone(),
                "last_response_pred": self.agent.last_response_pred.detach().clone(),
                "recurrent_state": self.agent.recurrent_state_dict(),
                "learning_context": self.agent.normalizer_state_dict(),
            },
        }

    def restore_execution_state(self, snapshot: Mapping[str, Any]) -> None:
        """Restore state captured by :meth:`snapshot_execution_state`."""
        self._state_history = deque(
            [
                np.asarray(frame, dtype=np.float32).copy()
                for frame in snapshot.get("history", [])
            ],
            maxlen=self.state_frame_stack,
        )
        for name, value in dict(snapshot.get("attributes", {})).items():
            setattr(self, name, copy.deepcopy(value))
        agent_state = dict(snapshot.get("agent", {}))
        if "action_visits" in agent_state:
            self.agent.action_visits = np.asarray(
                agent_state["action_visits"], dtype=np.int64
            ).copy()
        if "action_ever_feasible" in agent_state:
            self.agent.action_ever_feasible = np.asarray(
                agent_state["action_ever_feasible"], dtype=bool
            ).copy()
        for name in ("last_action_exploration_rate", "last_continuous_magnitude"):
            if name in agent_state:
                setattr(self.agent, name, float(agent_state[name]))
        for name in ("last_constraint_values", "last_risk_value", "last_response_pred"):
            if name in agent_state:
                setattr(self.agent, name, agent_state[name].detach().clone().to(self.agent.device))
        self.agent.load_recurrent_state_dict(agent_state.get("recurrent_state"))
        self.agent.load_normalizer_state_dict(
            agent_state.get("learning_context")
        )
        
    def update_response_context(self, share: float, gap: float, fulfillment: float) -> None:
        """Cache latest market response signals for state/action features."""
        self._last_share_hint = float(np.clip(share, 0.0, 1.0))
        self._last_gap_hint = float(np.nan_to_num(gap, nan=0.0, posinf=3.0, neginf=-3.0))
        self._last_fulfillment_hint = float(np.clip(fulfillment, 0.0, 1.0))

    def stack_state(self, state: np.ndarray, commit: bool = True) -> np.ndarray:
        """Return a fixed-width frame stack ending with the current state.

        The first observation is repeated to fill the stack, avoiding an all-zero
        cold start that would be out-of-distribution relative to later states.
        """
        current = np.asarray(state, dtype=np.float32).reshape(-1)
        if current.size != self.single_state_dim:
            raise ValueError(f"Single state dim mismatch: got {current.size}, expected {self.single_state_dim}")
        history = self._state_history if commit else deque(self._state_history, maxlen=self.state_frame_stack)
        if not history:
            for _ in range(self.state_frame_stack - 1):
                history.append(current.copy())
        history.append(current.copy())
        frames = list(history)
        while len(frames) < self.state_frame_stack:
            frames.insert(0, current.copy())
        return np.concatenate(frames[-self.state_frame_stack:]).astype(np.float32, copy=False)

    def feasible_action_mask(self, market_interaction) -> np.ndarray:
        """Return the stationary physical action set.

        Repeating a coefficient target keeps that coefficient posted, so it is
        intentionally legal. No objective or business rule masks an action.
        Under the Revision 15 level contract a step is legal exactly when the
        resulting level stays inside the level bounds; with no posted level
        (a comparator posted the tariff) only hold is defined.
        """
        if self.two_lever_actions:
            del market_interaction
            return np.ones(len(self.action_to_steps), dtype=bool)
        if self.level_step_actions:
            mask = np.zeros(len(self.action_to_steps), dtype=bool)
            mask[0] = True
            if self.tariff_level is None:
                return mask
            lower, upper = self.tariff_level_bounds
            for action, step in self.action_to_level_step.items():
                target = round(float(self.tariff_level) + float(step), 6)
                mask[int(action)] = bool(lower - 1e-9 <= target <= upper + 1e-9)
            return mask
        if self.hybrid_relative_actions:
            mask = np.zeros(len(self.action_to_steps), dtype=bool)
            mask[0] = True
            base = market_interaction.curr_market
            for action, relative_step in self.action_to_relative_step.items():
                target = self.action_to_target_key[int(action)]
                target_keys = (
                    tuple(self.opt_keys)
                    if target == "whole_tariff"
                    else (str(target),)
                )
                effects = []
                for key in target_keys:
                    anchor = float(getattr(base, key))
                    current_raw = getattr(self.overrides, key)
                    current = anchor if current_raw is None else float(current_raw)
                    lower, upper = self.config.bounds[key]
                    posted = float(np.clip(
                        current + anchor * float(relative_step), lower, upper
                    ))
                    effects.append(abs(posted - current) > 1e-8)
                # Whole-tariff steps remain feasible when at least one lever
                # can move. Per-coefficient cap hits are true no-ops and are
                # removed before behavior-policy sampling and PPO likelihoods.
                mask[int(action)] = bool(any(effects))
            return mask
        del market_interaction
        return np.ones(len(self.action_to_steps), dtype=bool)

    def current_tariff_level(self) -> Optional[float]:
        """Posted Revision 15 tariff level, or None when none is posted."""
        return None if self.tariff_level is None else float(self.tariff_level)

    def structure_values(self, level: float, market_interaction) -> Dict[str, float]:
        """Bounded coefficient values anchor * S * level for one lattice level."""
        if not self.level_step_actions:
            raise RuntimeError("structure_values requires the level-step contract")
        resolved = snap_tariff_level(level)
        base = market_interaction.curr_market
        structure = self.tariff_structure["structure"]
        anchors = self.tariff_structure["anchors"]
        values: Dict[str, float] = {}
        for key in self.opt_keys:
            anchor = float(getattr(base, key))
            if not np.isclose(anchor, float(anchors[key]), rtol=1e-9, atol=1e-12):
                raise ValueError(
                    f"runtime anchor {key}={anchor!r} differs from the tariff "
                    f"structure anchor {anchors[key]!r}"
                )
            lb, ub = self.config.bounds[key]
            values[key] = float(np.clip(anchor * float(structure[key]) * resolved, lb, ub))
        return values

    def post_tariff_level(self, level: float, market_interaction) -> Dict[str, float]:
        """Post anchor * S * level for every managed coefficient."""
        if not self.level_step_actions:
            raise RuntimeError("post_tariff_level requires the level-step contract")
        resolved = snap_tariff_level(level)
        lower, upper = self.tariff_level_bounds
        if not lower - 1e-9 <= resolved <= upper + 1e-9:
            raise ValueError(
                f"tariff level {resolved} lies outside [{lower}, {upper}]"
            )
        values = self.structure_values(resolved, market_interaction)
        for key, value in values.items():
            setattr(self.overrides, key, float(value))
        self.tariff_level = resolved
        return values

    def release_tariff_level(self) -> None:
        """Record that a comparator posted a tariff outside the level contract."""
        if self.level_step_actions:
            self.tariff_level = None

    def level_target_values(self, market_interaction) -> Dict[int, Dict[str, float]]:
        """Post-action coefficient values for every non-hold level step.

        Steps that would leave the bounds are clipped here only to keep their
        (masked) feature rows finite; ``feasible_action_mask`` excludes them.
        """
        if not self.level_step_actions or self.tariff_level is None:
            return {}
        lower, upper = self.tariff_level_bounds
        result: Dict[int, Dict[str, float]] = {}
        for action, step in self.action_to_level_step.items():
            target = float(np.clip(round(float(self.tariff_level) + float(step), 6), lower, upper))
            result[int(action)] = self.structure_values(target, market_interaction)
        return result
    
    def action_descriptor_vector(self) -> np.ndarray:
        """Compact vector describing the last executed option for PPO auxiliary credit."""
        d = getattr(self, "last_action_descriptor", ActionDescriptor())
        targets = [part for part in str(d.target).split("+") if part in self.action_keys]
        target_idx = (
            -1.0
            if not targets
            else float(np.mean([self.action_keys.index(part) for part in targets]))
        )
        return np.asarray(
            [
                float(d.direction),
                float((target_idx + 1) / max(1, len(self.action_keys))),
                float(np.clip(d.realized_delta_norm, -1.0, 1.0)),
                float(np.clip(abs(d.magnitude_level or d.magnitude_multiplier), 0.0, 2.0) / 2.0),
                float(np.clip(d.lower_distance, 0.0, 1.0)),
                float(np.clip(d.upper_distance, 0.0, 1.0)),
                float(np.clip(d.repeat_count / 10.0, 0.0, 1.0)),
                1.0 if d.was_clipped else 0.0,
            ],
            dtype=np.float32,
        )

    def build_action_feature_matrix(self, market_interaction, crowd_context: Optional[Dict[str, float]] = None) -> np.ndarray:
        """Fallback 20-column causal features for standalone pricer use."""
        ctx = crowd_context or {}
        base = market_interaction.curr_market
        rows: List[List[float]] = []
        near_threshold = float(np.clip(ctx.get("near_threshold_share", 0.0), 0.0, 1.0))
        threshold_mean = float(np.clip(ctx.get("price_threshold_mean", 1.5) / 8.0, 0.0, 1.0))
        no_ride_rate = float(np.clip(ctx.get("no_ride_rate", 0.0), 0.0, 1.0))
        peak = float(np.clip(ctx.get("peak_context", 0.0), 0.0, 1.0))
        airport_rate = float(np.clip(ctx.get("airport_rate", 0.0), 0.0, 1.0))
        level_targets = self.level_target_values(market_interaction)
        for action in range(len(self.action_to_steps)):
            target_key = self.action_to_target_key.get(int(action))
            multiplier = self.action_to_multiplier.get(int(action))
            relative_step = self.action_to_relative_step.get(int(action))
            level_values = level_targets.get(int(action))
            two_lever_cell = (
                self.action_to_two_lever_cell.get(int(action))
                if self.two_lever_actions else None
            )
            if (
                multiplier is None
                and relative_step is None
                and level_values is None
                and two_lever_cell is None
            ):
                rows.append([
                    1.0, 0.0, *([0.0] * 5), 0.0, 1.0, 1.0,
                    *([0.0] * 4),
                    near_threshold, no_ride_rate, peak, airport_rate,
                    threshold_mean, 0.0,
                ])
                continue
            target_keys = (
                ["per_mile", "per_minute"]
                if two_lever_cell is not None
                else list(self.opt_keys)
                if target_key is None or target_key == "whole_tariff"
                else [str(target_key)]
            )
            active: List[Tuple[str, int]] = []
            targets: Dict[str, float] = {}
            for key in target_keys:
                anchor = float(getattr(base, key))
                curr_raw = getattr(self.overrides, key)
                current = anchor if curr_raw is None else float(curr_raw)
                lb, ub = self.config.bounds[key]
                target = (
                    float(anchor * two_lever_cell[0 if key == "per_mile" else 1])
                    if two_lever_cell is not None
                    else float(level_values[key])
                    if level_values is not None
                    else float(np.clip(
                        current + anchor * float(relative_step), lb, ub
                    ))
                    if relative_step is not None
                    else float(np.clip(anchor * float(multiplier), lb, ub))
                )
                targets[key] = target
                active.append((key, int(np.sign(target - current))))
            directions = {key: int(direction) for key, direction in active}
            signature = [
                float(key in directions)
                for key in ("base_fare", "per_minute", "per_mile", "booking_fee", "airport_fee")
            ]
            relative, lower, upper, deltas = [], [], [], {}
            for key, direction in active:
                curr = getattr(self.overrides, key)
                anchor = float(getattr(base, key))
                curr_f = anchor if curr is None else float(curr)
                lb, ub = self.config.bounds[key]
                width = max(1e-6, float(ub - lb))
                relative.append(float(np.clip((curr_f - anchor) / max(abs(anchor), 1e-6), -1.0, 1.0)))
                lower.append(float(np.clip((curr_f - lb) / width, 0.0, 1.0)))
                upper.append(float(np.clip((ub - curr_f) / width, 0.0, 1.0)))
                deltas[key] = float(targets[key] - curr_f)
            exposures = (
                {"base_fare": 1.0, "per_minute": 8.0, "per_mile": 1.5, "booking_fee": 1.0, "airport_fee": 0.0},
                {"base_fare": 1.0, "per_minute": 14.0, "per_mile": 3.5, "booking_fee": 1.0, "airport_fee": 0.05},
                {"base_fare": 1.0, "per_minute": 24.0, "per_mile": 7.0, "booking_fee": 1.0, "airport_fee": 0.15},
                {"base_fare": 1.0, "per_minute": 38.0, "per_mile": 13.0, "booking_fee": 1.0, "airport_fee": 0.30},
            )
            impacts = [
                float(sum(deltas[key] * exposure[key] for key, _ in active))
                for exposure in exposures
            ]
            rows.append([
                0.0,
                float(np.clip(np.mean([direction for _, direction in active]), -1.0, 1.0)),
                *signature,
                float(np.mean(relative)),
                float(np.min(lower)),
                float(np.min(upper)),
                *[float(np.clip(impact / 20.0, -1.0, 1.0)) for impact in impacts],
                near_threshold,
                no_ride_rate,
                peak,
                airport_rate,
                threshold_mean,
                float(len(active) / max(1, len(self.action_keys))),
            ])
        result = np.asarray(rows, dtype=np.float32)
        if result.shape[1] != self.action_feature_dim:
            raise RuntimeError(
                f"standalone action feature dimension mismatch: {result.shape[1]} != {self.action_feature_dim}"
            )
        return result
        
    def configure_training_controls(self, progress: float, reward_converged: bool, reward_std: float) -> None:
        """Advance only PPO's predeclared behavior-coverage schedule.

        Reward convergence remains diagnostic: it cannot freeze the policy or
        alter the action space.  The scheduled uniform component includes hold
        and is accounted for by PPO's behavior likelihood.
        """
        del reward_converged, reward_std
        self.agent.set_training_progress(progress)
        self.allow_aggressive_actions = True
        self.step_scale = 1.0
        self.max_relative_dev = float("inf")

    def stabilize_after_batch(
        self,
        share: float,
        price_gap_f2_minus_f1: float,
        city_base: float,
        city_pmin: float,
        city_pmile: Optional[float] = None,
        city_booking: Optional[float] = None,
        city_airport: Optional[float] = None,
        profit_per_request: float = 0.0,
        fulfillment_rate: float = 1.0,
        target_price_gap: float = 0.0,
        target_gap_tolerance: float = 0.35,
    ) -> Dict[str, object]:
        """Compatibility hook; no rule modifies a learned tariff."""
        del (
            share,
            price_gap_f2_minus_f1,
            city_base,
            city_pmin,
            city_pmile,
            city_booking,
            city_airport,
            profit_per_request,
            fulfillment_rate,
            target_price_gap,
            target_gap_tolerance,
        )
        before = {k: getattr(self.overrides, k) for k in self.opt_keys}
        return {
            "applied": False,
            "reasons": [],
            "deltas": {},
            "before": before,
            "after": dict(before),
        }
            
    def apply_action(self, action: int, market_interaction) -> None:
        """Apply the configured revised or flags-off tariff action contract."""
        self.last_action_normalized_gap = {}
        self.last_action_step_units = {}
        self.last_action_was_saturated = False
        self.last_action_was_zero_effect = False
        self.last_action_descriptor = ActionDescriptor(action_id=int(action))
        if action not in self.action_to_steps:
            return
        full_action = int(action)
        if (
            not self.independent_coefficient_actions
            and len(self.incentive_levels) > 1
        ):
            self.supply_incentive_multiplier = float(
                np.clip(self.action_to_incentive.get(full_action, 1.0), 0.85, 1.25)
            )
            action = full_action % int(self.base_action_count)

        if self.two_lever_actions:
            self._apply_two_lever_cell(
                int(action), int(full_action), market_interaction
            )
            return

        if int(action) == 0:
            self._last_applied_action = int(action)
            self._repeat_action_count = 0
            self._last_action_target = "hold"
            self._last_action_direction = 0
            self._reversal_count = 0
            self.last_action_steps = {
                key: 0 for key in self.opt_keys
            }
            self.last_action_descriptor = ActionDescriptor(
                action_id=int(full_action), target="hold", direction=0
            )
            return

        if self.level_step_actions:
            self._apply_level_step(int(action), int(full_action), market_interaction)
            return

        multiplier = self.action_to_multiplier.get(int(action))
        relative_step = self.action_to_relative_step.get(int(action))
        target_key = self.action_to_target_key.get(int(action))
        if self.hybrid_relative_actions:
            if (
                relative_step is None
                or target_key not in {"whole_tariff", *self.opt_keys}
            ):
                return
        elif multiplier is None or (
            self.independent_coefficient_actions and target_key not in self.opt_keys
        ):
            return
        if int(action) == self._last_applied_action:
            self._repeat_action_count += 1
        else:
            self._last_applied_action = int(action)
            self._repeat_action_count = 1

        base = market_interaction.curr_market
        pre_values: Dict[str, float] = {}
        realized_by_key: Dict[str, float] = {}
        lower_distances: List[float] = []
        upper_distances: List[float] = []
        directions: Dict[str, int] = {}
        clipped_any = False
        target_keys = (
            (str(target_key),)
            if self.independent_coefficient_actions
            else (
                (str(target_key),)
                if self.hybrid_relative_actions
                and target_key != "whole_tariff"
                else tuple(self.opt_keys)
            )
        )
        for key in target_keys:
            anchor = float(getattr(base, key))
            current_raw = getattr(self.overrides, key)
            current = anchor if current_raw is None else float(current_raw)
            lb, ub = self.config.bounds[key]
            raw_target = float(
                current + anchor * float(relative_step)
                if self.hybrid_relative_actions
                else anchor * float(multiplier)
            )
            target_value = float(np.clip(raw_target, lb, ub))
            clipped_any = bool(
                clipped_any or abs(target_value - raw_target) > 1e-8
            )
            pre_values[key] = current
            setattr(self.overrides, key, target_value)
            width = max(1e-6, float(ub - lb))
            realized = float(target_value - current)
            realized_by_key[key] = realized
            directions[key] = int(np.sign(realized))
            lower_distances.append(float(np.clip((target_value - lb) / width, 0.0, 1.0)))
            upper_distances.append(float(np.clip((ub - target_value) / width, 0.0, 1.0)))

        self.last_action_was_zero_effect = bool(
            all(abs(delta) <= 1e-8 for delta in realized_by_key.values())
        )
        self.last_action_was_saturated = clipped_any
        self.last_action_steps = {
            key: int(directions.get(key, 0)) for key in self.opt_keys
        }
        self.last_action_step_units = {
            key: float(realized_by_key.get(key, 0.0) / max(1e-8, float(self.config.step[key])))
            for key in self.opt_keys
        }
        self.last_action_normalized_gap = {
            key: float(realized_by_key.get(key, 0.0))
            / max(1e-6, self.config.bounds[key][1] - self.config.bounds[key][0])
            for key in self.opt_keys
        }
        # Direction is the realized move from the previously posted tariff,
        # not the target's side of the city anchor. For example, 1.30x ->
        # 1.15x is a downward move even though the new level remains above
        # market. This is the direction exposed in the next observation frame.
        aggregate_direction = int(np.sign(sum(realized_by_key.values())))
        target = (
            str(target_key)
            if self.independent_coefficient_actions or self.hybrid_relative_actions
            else "whole_tariff"
        )
        is_reversal = bool(
            target == str(getattr(self, "_last_action_target", "hold"))
            and aggregate_direction != 0
            and aggregate_direction == -int(getattr(self, "_last_action_direction", 0) or 0)
        )
        self._reversal_count = int(getattr(self, "_reversal_count", 0) + 1) if is_reversal else 0
        self._last_action_target = target
        self._last_action_direction = aggregate_direction
        self.last_action_descriptor = ActionDescriptor(
            action_id=int(full_action),
            target=target,
            direction=aggregate_direction,
            intended_step=float(sum(realized_by_key.values())),
            realized_delta=float(sum(realized_by_key.values())),
            realized_delta_norm=float(sum(
                self.last_action_normalized_gap.values()
            )),
            pre_value=float(np.mean(list(pre_values.values()))),
            post_value=(
                float(getattr(self.overrides, str(target_key)))
                if (
                    self.independent_coefficient_actions
                    or (
                        self.hybrid_relative_actions
                        and target_key != "whole_tariff"
                    )
                )
                else float(np.mean([
                    float(getattr(self.overrides, key)) for key in self.opt_keys
                ]))
            ),
            lower_distance=float(min(lower_distances) if lower_distances else 0.0),
            upper_distance=float(min(upper_distances) if upper_distances else 0.0),
            magnitude_multiplier=(
                float(relative_step)
                if self.hybrid_relative_actions
                else float(multiplier)
            ),
            magnitude_level=(
                float(relative_step)
                if self.hybrid_relative_actions
                else float(multiplier)
            ),
            repeat_count=int(self._repeat_action_count),
            was_clipped=clipped_any,
            is_reversal=is_reversal,
            reversal_count=int(self._reversal_count),
        )

    def _apply_two_lever_cell(
        self, action: int, full_action: int, market_interaction
    ) -> None:
        """Post one absolute V17 card cell; repeating it leaves fares unchanged."""
        cell = self.action_to_two_lever_cell.get(int(action))
        if cell is None:
            return
        base = market_interaction.curr_market
        multipliers = {
            "per_mile": float(cell[0]),
            "per_minute": float(cell[1]),
        }
        pre_values: Dict[str, float] = {}
        realized_by_key: Dict[str, float] = {}
        lower_distances: List[float] = []
        upper_distances: List[float] = []
        clipped_any = False
        for key, multiplier in multipliers.items():
            anchor = float(getattr(base, key))
            current_raw = getattr(self.overrides, key)
            current = anchor if current_raw is None else float(current_raw)
            lower, upper = self.config.bounds[key]
            raw_target = anchor * multiplier
            target = float(np.clip(raw_target, lower, upper))
            clipped_any = bool(
                clipped_any
                or not np.isclose(target, raw_target, rtol=0.0, atol=1e-9)
            )
            pre_values[key] = current
            realized_by_key[key] = target - current
            setattr(self.overrides, key, target)
            width = max(1e-9, upper - lower)
            lower_distances.append(
                float(np.clip((target - lower) / width, 0.0, 1.0))
            )
            upper_distances.append(
                float(np.clip((upper - target) / width, 0.0, 1.0))
            )
        # M5 exposes only distance and time. Every other rate-card component
        # remains at the calibrated card even if this object was previously
        # used by another action contract.
        for key in self.opt_keys:
            if key not in multipliers:
                setattr(self.overrides, key, None)
        repeated = int(action) == self._last_applied_action
        self._repeat_action_count = self._repeat_action_count + 1 if repeated else 1
        self._last_applied_action = int(action)
        self.last_action_was_zero_effect = all(
            abs(delta) <= 1e-9 for delta in realized_by_key.values()
        )
        self.last_action_was_saturated = clipped_any
        self.last_action_steps = {
            key: int(np.sign(realized_by_key.get(key, 0.0)))
            for key in self.opt_keys
        }
        self.last_action_step_units = {
            key: float(
                realized_by_key.get(key, 0.0)
                / max(1e-9, self.config.step[key])
            )
            for key in self.opt_keys
        }
        self.last_action_normalized_gap = {
            key: float(realized_by_key.get(key, 0.0))
            / max(
                1e-9,
                self.config.bounds[key][1] - self.config.bounds[key][0],
            )
            for key in self.opt_keys
        }
        aggregate_direction = int(np.sign(sum(realized_by_key.values())))
        self._last_action_target = "per_mile+per_minute"
        self._last_action_direction = aggregate_direction
        self._reversal_count = 0
        self.last_action_descriptor = ActionDescriptor(
            action_id=int(full_action),
            target="per_mile+per_minute",
            direction=aggregate_direction,
            intended_step=float(sum(realized_by_key.values())),
            realized_delta=float(sum(realized_by_key.values())),
            realized_delta_norm=float(
                sum(self.last_action_normalized_gap.values())
            ),
            pre_value=float(np.mean(list(pre_values.values()))),
            post_value=float(np.mean([
                float(self.overrides.per_mile),
                float(self.overrides.per_minute),
            ])),
            lower_distance=float(min(lower_distances)),
            upper_distance=float(min(upper_distances)),
            magnitude_multiplier=float(np.mean(cell)),
            magnitude_level=float(np.mean(cell)),
            repeat_count=int(self._repeat_action_count),
            was_clipped=clipped_any,
        )

    def _apply_level_step(self, action: int, full_action: int, market_interaction) -> None:
        """Move the posted level by one registered step and repost S * L."""
        if self.tariff_level is None:
            raise RuntimeError(
                "a level step requires a tariff posted through post_tariff_level"
            )
        step = self.action_to_level_step.get(int(action))
        if step is None:
            raise ValueError(f"unknown level-step action {action}")
        lower, upper = self.tariff_level_bounds
        new_level = round(float(self.tariff_level) + float(step), 6)
        if not lower - 1e-9 <= new_level <= upper + 1e-9:
            raise ValueError(
                f"masked level step executed: {self.tariff_level} {step:+.2f} "
                f"leaves [{lower}, {upper}]"
            )
        new_level = snap_tariff_level(new_level)
        if int(action) == self._last_applied_action:
            self._repeat_action_count += 1
        else:
            self._last_applied_action = int(action)
            self._repeat_action_count = 1
        base = market_interaction.curr_market
        structure = self.tariff_structure["structure"]
        values = self.structure_values(new_level, market_interaction)
        pre_values: Dict[str, float] = {}
        realized_by_key: Dict[str, float] = {}
        lower_distances: List[float] = []
        upper_distances: List[float] = []
        clipped_any = False
        for key in self.opt_keys:
            anchor = float(getattr(base, key))
            current_raw = getattr(self.overrides, key)
            current = anchor if current_raw is None else float(current_raw)
            lb, ub = self.config.bounds[key]
            raw_target = float(anchor * float(structure[key]) * new_level)
            target_value = float(values[key])
            clipped_any = bool(clipped_any or abs(target_value - raw_target) > 1e-8)
            pre_values[key] = current
            setattr(self.overrides, key, target_value)
            width = max(1e-6, float(ub - lb))
            realized_by_key[key] = float(target_value - current)
            lower_distances.append(float(np.clip((target_value - lb) / width, 0.0, 1.0)))
            upper_distances.append(float(np.clip((ub - target_value) / width, 0.0, 1.0)))
        previous_level = float(self.tariff_level)
        self.tariff_level = new_level
        self.last_action_was_zero_effect = bool(
            all(abs(delta) <= 1e-8 for delta in realized_by_key.values())
        )
        self.last_action_was_saturated = clipped_any
        self.last_action_steps = {
            key: int(np.sign(realized_by_key.get(key, 0.0))) for key in self.opt_keys
        }
        self.last_action_step_units = {
            key: float(realized_by_key.get(key, 0.0) / max(1e-8, float(self.config.step[key])))
            for key in self.opt_keys
        }
        self.last_action_normalized_gap = {
            key: float(realized_by_key.get(key, 0.0))
            / max(1e-6, self.config.bounds[key][1] - self.config.bounds[key][0])
            for key in self.opt_keys
        }
        aggregate_direction = int(np.sign(float(step)))
        target = "tariff_level"
        is_reversal = bool(
            target == str(getattr(self, "_last_action_target", "hold"))
            and aggregate_direction != 0
            and aggregate_direction == -int(getattr(self, "_last_action_direction", 0) or 0)
        )
        self._reversal_count = int(getattr(self, "_reversal_count", 0) + 1) if is_reversal else 0
        self._last_action_target = target
        self._last_action_direction = aggregate_direction
        self.last_action_descriptor = ActionDescriptor(
            action_id=int(full_action),
            target=target,
            direction=aggregate_direction,
            intended_step=float(step),
            realized_delta=float(new_level - previous_level),
            realized_delta_norm=float(sum(self.last_action_normalized_gap.values())),
            pre_value=float(previous_level),
            post_value=float(new_level),
            lower_distance=float(min(lower_distances) if lower_distances else 0.0),
            upper_distance=float(min(upper_distances) if upper_distances else 0.0),
            magnitude_multiplier=float(new_level),
            magnitude_level=float(new_level),
            repeat_count=int(self._repeat_action_count),
            was_clipped=clipped_any,
            is_reversal=is_reversal,
            reversal_count=int(self._reversal_count),
        )
