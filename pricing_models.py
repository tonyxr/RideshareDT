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

    Legacy PPO chooses an absolute whole-tariff target whose historical bounds
    are enforced by the application layer. In v28, the same descriptor records
    an unbounded scale of the currently posted tariff.
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
    numerical_rejection: bool = False
    numerical_rejection_reason: str = ""
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
) -> Tuple[
    Dict[int, Dict[str, int]],
    Dict[int, float],
    Dict[int, str],
    List[str],
]:
    """Build hold plus stable absolute managed-tariff market targets.

    One policy decision posts a coherent tariff over the experiment's fixed
    1-5 coefficient scope instead of changing a single coefficient and asking
    PPO to remember a multi-action composition. Every non-hold action sets all
    managed coefficients to the same multiplier of their city anchors.
    Repeating a target keeps that tariff posted and can never create
    incremental drift.
    """
    all_keys = ["base_fare", "per_minute", "per_mile", "booking_fee", "airport_fee"]
    active_keys = [k for k in all_keys if k in list(opt_keys)[:max_coeffs]]
    multipliers = (0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30)
    action_to_steps: Dict[int, Dict[str, int]] = {
        0: {key: 0 for key in active_keys}
    }
    action_to_multiplier: Dict[int, float] = {}
    action_to_target_key: Dict[int, str] = {}
    for multiplier in multipliers:
        action = len(action_to_steps)
        direction = int(np.sign(float(multiplier) - 1.0))
        action_to_steps[action] = {
            candidate: direction for candidate in active_keys
        }
        action_to_multiplier[action] = float(multiplier)
    return (
        action_to_steps,
        action_to_multiplier,
        action_to_target_key,
        active_keys,
    )

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
            revenue_surplus = float(
                np.clip((self.ema_revenue - self.revenue_target) / self.revenue_target, -1.0, 1.0)
            )
        fulfillment_stress = float(np.clip((0.78 - self.ema_fulfillment) / 0.38, 0.0, 1.0))
        acceptance_stress = float(np.clip((0.70 - self.ema_acceptance) / 0.35, 0.0, 1.0))
        wait_stress = float(np.clip((self.ema_wait - 8.0) / 10.0, 0.0, 1.0))
        pickup_stress = float(np.clip((self.ema_pickup - 8.0) / 10.0, 0.0, 1.0))
        earnings_stress = float(np.clip((24.0 - self.ema_driver_earnings) / 24.0, 0.0, 1.0))
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
        if self.revenue_target is None:
            self.revenue_target = float(max(1e-6, metrics.rev_per_request))
            self.ema_revenue = self.revenue_target
        self.ema_share = self._ema(metrics.share, self.ema_share, self.alpha_share)
        self.ema_choice_share = self._ema(metrics.chosen_share, self.ema_choice_share, self.alpha_share)
        self.ema_completed_share = self._ema(metrics.completed_share, self.ema_completed_share, self.alpha_share)
        self.ema_revenue = self._ema(metrics.rev_per_request, self.ema_revenue, self.alpha_revenue)
        self.ema_gap = self._ema(price_gap_mean, self.ema_gap, self.alpha_gap)
        supply = supply_state or {}
        self.ema_fulfillment = self._ema(
            supply.get("fulfillment_rate", metrics.fulfillment_rate),
            self.ema_fulfillment,
            self.alpha_share,
        )
        self.ema_acceptance = self._ema(
            supply.get("acceptance_rate", metrics.driver_acceptance_rate),
            self.ema_acceptance,
            self.alpha_share,
        )
        self.ema_wait = self._ema(supply.get("avg_wait_minutes", metrics.avg_wait_minutes), self.ema_wait, self.alpha_share)
        self.ema_pickup = self._ema(
            supply.get("avg_pickup_minutes", metrics.avg_wait_minutes),
            self.ema_pickup,
            self.alpha_share,
        )
        self.ema_driver_earnings = self._ema(
            supply.get("driver_earnings_per_hour", self.ema_driver_earnings),
            self.ema_driver_earnings,
            self.alpha_share,
        )
        self.ema_idle_share = self._ema(supply.get("idle_driver_share", self.ema_idle_share), self.ema_idle_share, self.alpha_share)
        self.ema_utilization = self._ema(supply.get("utilization", self.ema_utilization), self.ema_utilization, self.alpha_share)
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
        key = next((k for k in key_priority if k in self.managed_keys and getattr(self.overrides, k, None) is not None), None)
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
            (self.revenue_target - self.ema_revenue) / self.revenue_target,
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
            + 0.20 * self.ema_gap
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
        step = float(self.config.step[key])
        price_effect = float(direction) * step
        relative_step = float(np.clip(price_effect / max(1.0, self.ema_revenue), -0.25, 0.25))
        own_demand = float(np.clip(self.ema_choice_share, 0.0, 1.0))
        predicted_choice_share = float(np.clip(own_demand - 0.80 * relative_step, 0.0, 1.0))
        predicted_completed_share = float(np.clip(
            self.ema_completed_share - 0.50 * max(0.0, relative_step) + 0.20 * max(0.0, -relative_step),
            0.0,
            1.0,
        ))
        supply_stress = float(np.clip(self.last_supply_stress, 0.0, 1.0))
        revenue_gap = 0.0 if self.revenue_target is None else float(np.clip(
            (self.revenue_target - self.ema_revenue) / self.revenue_target,
            -1.0,
            1.0,
        ))
        gap_after = self.ema_gap + price_effect
        revenue_bonus = (self.ema_revenue + price_effect) * predicted_choice_share / max(self.revenue_target or 1.0, 1.0)
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

        best_score = 0.0
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

        self.last_response_score = best_score
        if best_key is None or best_score <= self.response_threshold:
            self.last_reason = "mpc_grid_hold"
            return
        moved = self._apply_step(best_key, best_direction)
        self.last_selected_key = best_key if moved else "hold"
        self.last_direction = int(best_direction) if moved else 0
        self.last_reason = "mpc_grid" if moved else "bounded"
        if moved:
            self.cooldown = self.cooldown_H



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
    # This is not an economic price bound. It leaves headroom for float32
    # policy features and downstream fare arithmetic; actions crossing it are
    # rejected atomically and reported rather than silently clipped.
    NUMERICAL_COEFFICIENT_LIMIT = float(np.finfo(np.float32).max / 1024.0)

    def _ensure_internal_state(self) -> None:
        """Backstop against partially initialized objects from stale environments."""
        if not hasattr(self, "opt_keys"):
            self.opt_keys = ["base_fare", "per_minute"]
        if len(self.opt_keys) > self.MAX_MANIPULATED_COEFFS:
            self.opt_keys = list(self.opt_keys[: self.MAX_MANIPULATED_COEFFS])
        if not hasattr(self, "config"):
            self.config = default_specs_for(self.opt_keys)
        if not hasattr(self, "experimental_unbounded_economics"):
            self.experimental_unbounded_economics = False
        if not hasattr(self, "minimum_tariff_multiplier"):
            self.minimum_tariff_multiplier = 0.0
        if not hasattr(self, "numerical_action_rejection_count"):
            self.numerical_action_rejection_count = 0
        if not hasattr(self, "last_action_numerical_rejection"):
            self.last_action_numerical_rejection = ""
    
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
        center_rewards: bool = True,
        td_value_targets: bool = False,
        normalize_values: bool = False,
        experimental_unbounded_economics: bool = False,
        independent_coefficient_actions: bool = False,
        minimum_tariff_multiplier: float = 0.0,
    ):
        # One action posts a stable target over the complete managed tariff
        # scope. This removes the hidden multi-action composition problem from
        # credit assignment regardless of whether the experiment controls
        # one, several, or all five price coefficients.
        self.opt_keys = list(opt_keys[: self.MAX_MANIPULATED_COEFFS])
        if len(opt_keys) > self.MAX_MANIPULATED_COEFFS:
            print(
                f"[FirmRLPricer] Received {len(opt_keys)} opt_keys; "
                f"limiting to first {self.MAX_MANIPULATED_COEFFS}: {self.opt_keys}"
            )
        self.config = default_specs_for(self.opt_keys)
        self.experimental_unbounded_economics = bool(
            experimental_unbounded_economics
        )
        self.independent_coefficient_actions = bool(
            independent_coefficient_actions
        )
        self.minimum_tariff_multiplier = float(minimum_tariff_multiplier)
        if (
            not np.isfinite(self.minimum_tariff_multiplier)
            or self.minimum_tariff_multiplier < 0.0
        ):
            raise ValueError("minimum_tariff_multiplier must be finite and non-negative")
        self.overrides = CoefficientOverrides()
        self.state_frame_stack = int(max(1, state_frame_stack))
        self._state_history: Deque[np.ndarray] = deque(maxlen=self.state_frame_stack)
        
        # Compatibility values retained for reports. Independent v29 actions
        # apply one fixed additive step to one coefficient; the scale is kept
        # stationary so the action has the same meaning throughout training.
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
        self.last_action_numerical_rejection = ""
        self.numerical_action_rejection_count = 0
        self.last_action_step_units: Dict[str, float] = {}
        self._last_action_target = "hold"
        self._last_action_direction = 0
        self._reversal_count = 0
        self.recovery_share_threshold = 0.25
        self.recovery_gap_threshold = -0.05
        self.aggressive_actions = set()
        self.allow_aggressive_actions = True
        
        # Index 0 holds the tariff. Independent mode exposes one up/down pair
        # per coefficient. The legacy whole-tariff path retains its absolute
        # target vocabulary; v29 never uses that uniform control.
        if self.independent_coefficient_actions:
            self.action_to_steps, self.action_keys = build_discrete_action_space(
                self.opt_keys, self.MAX_MANIPULATED_COEFFS
            )
            self.action_to_multiplier = {}
            self.action_to_target_key = {
                action: next(
                    (key for key, step in mapping.items() if int(step) != 0),
                    "hold",
                )
                for action, mapping in self.action_to_steps.items()
            }
        else:
            (
                self.action_to_steps,
                self.action_to_multiplier,
                self.action_to_target_key,
                self.action_keys,
            ) = build_absolute_tariff_action_space(
                self.opt_keys,
                self.MAX_MANIPULATED_COEFFS,
            )
        action_dim = len(self.action_to_steps)
        
        # Each compact public frame contains clock, five own tariff positions,
        # delayed own outcomes, demand/route-basis context, and public rival
        # quote endpoints. Ordered frame stacking supplies trajectory and
        # momentum without exposing rival internals or duplicating action IDs.
        self.single_state_dim = int(max(1, single_state_dim))
        state_dim = self.single_state_dim * self.state_frame_stack

        # Initialize PPO agent.
        self.agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            gamma=float(gamma),
            normalize_values=normalize_values,
            lam=float(gae_lambda),
            hidden_dim=192,
            clip_eps=0.20,
            final_clip_eps=0.10,
            # The previous 0.8 threshold clipped 45/50 critic updates in the
            # readiness run while critic explained variance stayed near zero.
            # Five preserves protection against non-finite explosions without
            # flattening ordinary raw-return value gradients.
            max_grad_norm=5.0,
            ent_coeff=0.010,
            min_ent_coeff=0.0005,
            max_ent_coeff=0.040,
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
            center_rewards=bool(center_rewards),
            td_value_targets=bool(td_value_targets),
            # The original PPO stack samples directly from its categorical
            # actor.  Independent coefficient actions must not silently add an
            # epsilon-uniform controller: it changes the behavior policy and
            # can make an unconstrained tariff walk look like learned action.
            scheduled_exploration=bool(td_value_targets),
            # No hand-authored hold prior.  Entropy in PPO supplies exploration
            # and every one of the eleven actions begins with equal logits.
            initial_hold_probability=None,
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
        """Compact human-readable label for the selected tariff operation."""
        if int(action) == 0:
            return "hold"
        if self.independent_coefficient_actions:
            mapping = self.action_to_steps.get(int(action), {})
            active = [(key, step) for key, step in mapping.items() if step]
            if len(active) != 1:
                return "invalid"
            key, step = active[0]
            return f"{key}:{'up' if step > 0 else 'down'}"
        multiplier = self.action_to_multiplier.get(int(action))
        if multiplier is None:
            return "invalid"
        relation = "current" if self.experimental_unbounded_economics else "market"
        operation = "tariff_scale" if self.experimental_unbounded_economics else "tariff_target"
        return f"{operation}:{multiplier:.3f}x_{relation}"

    @classmethod
    def _numerically_safe_scaled_value(
        cls, current: float, multiplier: float
    ) -> Tuple[Optional[float], str]:
        """Scale a coefficient without imposing an economic range.

        Returning ``None`` is an explicit atomic rejection signal. Values are
        never projected onto ``NUMERICAL_COEFFICIENT_LIMIT`` because doing so
        would turn overflow protection into a hidden policy action.
        """
        current_f = float(current)
        multiplier_f = float(multiplier)
        if not np.isfinite(current_f) or not np.isfinite(multiplier_f):
            return None, "non_finite_input"
        try:
            target = current_f * multiplier_f
        except OverflowError:
            return None, "float_overflow"
        if not np.isfinite(target):
            return None, "non_finite_product"
        if abs(target) > cls.NUMERICAL_COEFFICIENT_LIMIT:
            return None, "float32_headroom_exceeded"
        return float(target), ""

    @classmethod
    def _numerically_safe_added_value(
        cls, current: float, delta: float
    ) -> Tuple[Optional[float], str]:
        """Add an uncapped coefficient step with numerical checks only."""
        current_f = float(current)
        delta_f = float(delta)
        if not np.isfinite(current_f) or not np.isfinite(delta_f):
            return None, "non_finite_input"
        target = current_f + delta_f
        if not np.isfinite(target):
            return None, "non_finite_sum"
        if abs(target) > cls.NUMERICAL_COEFFICIENT_LIMIT:
            return None, "float32_headroom_exceeded"
        return float(target), ""
        
    def last_action_magnitude(self) -> float:
        """Mean realized movement in standard coefficient-step units.

        This is invariant to the very different numerical ranges of airport,
        per-minute, and fixed-fee coefficients.  Moving every managed lever by
        one configured step has magnitude 1; one lever has magnitude 1/5.
        """
        if self.independent_coefficient_actions:
            return float(
                sum(abs(v) for v in getattr(
                    self, "last_action_step_units", {}
                ).values())
                / max(1, len(self.opt_keys))
            )
        if self.experimental_unbounded_economics:
            multiplier = float(
                getattr(self.last_action_descriptor, "magnitude_multiplier", 1.0)
            )
            return abs(multiplier - 1.0)
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
        self.last_action_numerical_rejection = ""
        self.agent.reset_learning_context()

    def snapshot_execution_state(self) -> Dict[str, Any]:
        """Capture mutable inference state without copying policy parameters."""
        attribute_names = (
            "last_state_features",
            "last_action_features",
            "last_action_descriptor",
            "last_action_was_saturated",
            "last_action_was_zero_effect",
            "last_action_numerical_rejection",
            "numerical_action_rejection_count",
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
        """Return the physically effective action set.

        Repeating a whole-tariff target means keep that coherent tariff posted,
        so it is intentionally legal.  In the v29 independent-action contract,
        a downward action at the explicit coefficient floor is excluded because
        executing it would be a mislabeled no-op.  No objective or business
        performance rule masks an action.
        """
        mask = np.ones(len(self.action_to_steps), dtype=bool)
        if not (
            self.independent_coefficient_actions
            and self.minimum_tariff_multiplier > 0.0
        ):
            return mask
        base = market_interaction.curr_market
        for action, mapping in self.action_to_steps.items():
            active = [
                (key, int(direction))
                for key, direction in mapping.items()
                if int(direction) != 0
            ]
            if len(active) != 1:
                continue
            key, direction = active[0]
            if direction >= 0:
                continue
            anchor = float(getattr(base, key))
            current_raw = getattr(self.overrides, key)
            current = anchor if current_raw is None else float(current_raw)
            floor = anchor * self.minimum_tariff_multiplier
            if current <= floor + 1e-10 * max(1.0, abs(floor)):
                mask[int(action)] = False
        mask[0] = True
        return mask
    
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
                float(np.clip(d.magnitude_level or d.magnitude_multiplier, 0.0, 2.0) / 2.0),
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
        raw_threshold_mean = float(ctx.get("price_threshold_mean", 1.5))
        threshold_mean = (
            float(np.log1p(raw_threshold_mean) / np.log(9.0))
            if self.experimental_unbounded_economics and raw_threshold_mean > 0.0
            else float(np.clip(raw_threshold_mean / 8.0, 0.0, 1.0))
        )
        no_ride_rate = float(np.clip(ctx.get("no_ride_rate", 0.0), 0.0, 1.0))
        peak = float(np.clip(ctx.get("peak_context", 0.0), 0.0, 1.0))
        airport_rate = float(np.clip(ctx.get("airport_rate", 0.0), 0.0, 1.0))
        for action in range(len(self.action_to_steps)):
            target_key = self.action_to_target_key.get(int(action))
            multiplier = self.action_to_multiplier.get(int(action))
            if multiplier is None:
                rows.append([
                    1.0, 0.0, *([0.0] * 5), 0.0, 1.0, 1.0,
                    *([0.0] * 4),
                    near_threshold, no_ride_rate, peak, airport_rate,
                    threshold_mean, 0.0,
                ])
                continue
            target_keys = (
                list(self.opt_keys)
                if target_key is None
                else [str(target_key)]
            )
            active: List[Tuple[str, int]] = []
            targets: Dict[str, float] = {}
            for key in target_keys:
                anchor = float(getattr(base, key))
                curr_raw = getattr(self.overrides, key)
                current = anchor if curr_raw is None else float(curr_raw)
                lb, ub = self.config.bounds[key]
                if self.experimental_unbounded_economics:
                    candidate, _ = self._numerically_safe_scaled_value(
                        current, multiplier
                    )
                    target = current if candidate is None else candidate
                else:
                    target = float(np.clip(anchor * multiplier, lb, ub))
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
                if self.experimental_unbounded_economics:
                    candidate, _ = self._numerically_safe_scaled_value(
                        curr_f, multiplier
                    )
                    numerical_room = 1.0 if candidate is not None else 0.0
                    lower.append(numerical_room)
                    upper.append(numerical_room)
                else:
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
        """Apply a legacy anchor target or an unbounded v28 tariff scale."""
        self.last_action_normalized_gap = {}
        self.last_action_step_units = {}
        self.last_action_was_saturated = False
        self.last_action_was_zero_effect = False
        self.last_action_numerical_rejection = ""
        self.last_action_descriptor = ActionDescriptor(action_id=int(action))
        if action not in self.action_to_steps:
            return

        if self.independent_coefficient_actions:
            self._apply_independent_coefficient_action(action, market_interaction)
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
                action_id=int(action), target="hold", direction=0
            )
            return

        multiplier = self.action_to_multiplier.get(int(action))
        if multiplier is None:
            return
        if int(action) == self._last_applied_action:
            self._repeat_action_count += 1
        else:
            self._last_applied_action = int(action)
            self._repeat_action_count = 1

        base = market_interaction.curr_market
        pre_values: Dict[str, float] = {}
        target_values: Dict[str, float] = {}
        realized_by_key: Dict[str, float] = {}
        lower_distances: List[float] = []
        upper_distances: List[float] = []
        directions: Dict[str, int] = {}
        clipped_any = False

        # Resolve the complete action before mutating any coefficient. This
        # makes numerical overflow a transparent all-or-nothing rejection.
        for key in self.opt_keys:
            anchor = float(getattr(base, key))
            current_raw = getattr(self.overrides, key)
            current = anchor if current_raw is None else float(current_raw)
            lb, ub = self.config.bounds[key]
            pre_values[key] = current
            if self.experimental_unbounded_economics:
                target_value, rejection_reason = self._numerically_safe_scaled_value(
                    current, multiplier
                )
                if target_value is None:
                    self.last_action_numerical_rejection = (
                        f"{key}:{rejection_reason}"
                    )
                    self.numerical_action_rejection_count += 1
                    self.last_action_was_zero_effect = True
                    self.last_action_steps = {
                        candidate: 0 for candidate in self.opt_keys
                    }
                    pre_mean = float(np.mean(list(pre_values.values())))
                    self.last_action_descriptor = ActionDescriptor(
                        action_id=int(action),
                        target="whole_tariff_scale",
                        direction=0,
                        pre_value=pre_mean,
                        post_value=pre_mean,
                        lower_distance=0.0,
                        upper_distance=0.0,
                        magnitude_multiplier=float(multiplier),
                        magnitude_level=float(multiplier),
                        repeat_count=int(self._repeat_action_count),
                        numerical_rejection=True,
                        numerical_rejection_reason=self.last_action_numerical_rejection,
                    )
                    return
            else:
                raw_target = float(anchor * multiplier)
                target_value = float(np.clip(raw_target, lb, ub))
                clipped_any = bool(
                    clipped_any or abs(target_value - raw_target) > 1e-8
                )
            target_values[key] = float(target_value)

        for key in self.opt_keys:
            anchor = float(getattr(base, key))
            current = pre_values[key]
            target_value = target_values[key]
            lb, ub = self.config.bounds[key]
            setattr(self.overrides, key, target_value)
            width = max(1e-6, float(ub - lb))
            realized = float(target_value - current)
            realized_by_key[key] = realized
            directions[key] = int(np.sign(realized))
            if self.experimental_unbounded_economics:
                lower_distances.append(1.0)
                upper_distances.append(1.0)
            else:
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
        self.last_action_normalized_gap = {}
        for key in self.opt_keys:
            if self.experimental_unbounded_economics:
                anchor = float(getattr(base, key))
                denominator = max(
                    1e-6, abs(float(pre_values[key])), abs(anchor)
                )
            else:
                denominator = max(
                    1e-6,
                    self.config.bounds[key][1] - self.config.bounds[key][0],
                )
            self.last_action_normalized_gap[key] = float(
                realized_by_key.get(key, 0.0) / denominator
            )
        # Direction is the realized move from the previously posted tariff,
        # not the target's side of the city anchor. For example, 1.30x ->
        # 1.15x is a downward move even though the new level remains above
        # market. This is the direction exposed in the next observation frame.
        aggregate_direction = int(np.sign(sum(realized_by_key.values())))
        target = (
            "whole_tariff_scale"
            if self.experimental_unbounded_economics
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
            action_id=int(action),
            target=target,
            direction=aggregate_direction,
            intended_step=float(sum(realized_by_key.values())),
            realized_delta=float(sum(realized_by_key.values())),
            realized_delta_norm=float(sum(
                self.last_action_normalized_gap[key]
                for key in self.opt_keys
            )),
            pre_value=float(np.mean(list(pre_values.values()))),
            post_value=float(np.mean([
                float(getattr(self.overrides, key)) for key in self.opt_keys
            ])),
            lower_distance=float(min(lower_distances) if lower_distances else 0.0),
            upper_distance=float(min(upper_distances) if upper_distances else 0.0),
            magnitude_multiplier=float(multiplier),
            magnitude_level=float(multiplier),
            repeat_count=int(self._repeat_action_count),
            was_clipped=clipped_any,
            is_reversal=is_reversal,
            reversal_count=int(self._reversal_count),
        )

    def _apply_independent_coefficient_action(self, action: int, market_interaction) -> None:
        """Move exactly one tariff coefficient; never apply a uniform tariff move."""
        mapping = dict(self.action_to_steps.get(int(action), {}))
        active = [(key, int(step)) for key, step in mapping.items() if int(step)]
        if not active:
            self._last_applied_action = int(action)
            self._repeat_action_count = 0
            self._last_action_target = "hold"
            self._last_action_direction = 0
            self._reversal_count = 0
            self.last_action_steps = {key: 0 for key in self.opt_keys}
            self.last_action_descriptor = ActionDescriptor(
                action_id=int(action), target="hold", direction=0
            )
            return
        if len(active) != 1:
            raise RuntimeError("independent coefficient action changed multiple levers")

        key, direction = active[0]
        base = market_interaction.curr_market
        anchor = float(getattr(base, key))
        current_raw = getattr(self.overrides, key)
        current = anchor if current_raw is None else float(current_raw)
        step = float(self.config.step[key]) * float(self.step_scale)
        lb, ub = self.config.bounds[key]
        clipped = False
        if self.experimental_unbounded_economics:
            # Preserve the original independent-action semantics: one action
            # adds or subtracts one configured step from one coefficient. The
            # v28 contract removes economic bounds, not the meaning or size of
            # the action. Additive steps also keep a zero-valued fee reversible
            # instead of making zero an absorbing state under multiplication.
            target, reason = self._numerically_safe_added_value(
                current, float(direction) * step
            )
            if target is None:
                self.last_action_numerical_rejection = f"{key}:{reason}"
                self.numerical_action_rejection_count += 1
                self.last_action_was_zero_effect = True
                self.last_action_steps = {candidate: 0 for candidate in self.opt_keys}
                self.last_action_descriptor = ActionDescriptor(
                    action_id=int(action), target=key, direction=0,
                    pre_value=current, post_value=current,
                    numerical_rejection=True,
                    numerical_rejection_reason=self.last_action_numerical_rejection,
                )
                return
            multiplier = (
                float(target / current) if abs(current) > 1e-8 else 1.0
            )
        else:
            raw_target = current + float(direction) * step
            target = float(np.clip(raw_target, lb, ub))
            clipped = bool(abs(target - raw_target) > 1e-8)
            multiplier = float(target / current) if abs(current) > 1e-8 else 1.0

        multiplier_floor_value = float(anchor) * float(
            self.minimum_tariff_multiplier
        )
        if target < multiplier_floor_value:
            target = multiplier_floor_value
            clipped = True
            multiplier = (
                float(target / current) if abs(current) > 1e-8 else 1.0
            )

        if int(action) == self._last_applied_action:
            self._repeat_action_count += 1
        else:
            self._last_applied_action = int(action)
            self._repeat_action_count = 1
        realized = float(target - current)
        realized_direction = int(np.sign(realized))
        previous_target = str(getattr(self, "_last_action_target", "hold"))
        previous_direction = int(getattr(self, "_last_action_direction", 0) or 0)
        reversal = bool(
            previous_target == key
            and realized_direction != 0
            and realized_direction == -previous_direction
        )
        self._reversal_count = self._reversal_count + 1 if reversal else 0
        self._last_action_target = key
        self._last_action_direction = realized_direction
        setattr(self.overrides, key, float(target))
        self.last_action_was_saturated = clipped
        self.last_action_was_zero_effect = abs(realized) <= 1e-8
        self.last_action_steps = {
            candidate: realized_direction if candidate == key else 0
            for candidate in self.opt_keys
        }
        self.last_action_step_units = {
            candidate: (
                realized / max(1e-8, float(self.config.step[candidate]))
                if candidate == key else 0.0
            )
            for candidate in self.opt_keys
        }
        denominator = (
            max(abs(current), abs(anchor), 1e-6)
            if self.experimental_unbounded_economics
            else max(1e-6, ub - lb)
        )
        self.last_action_normalized_gap = {
            candidate: realized / denominator if candidate == key else 0.0
            for candidate in self.opt_keys
        }
        lower_distance = (
            float((target - multiplier_floor_value) / max(abs(anchor), 1e-6))
            if self.experimental_unbounded_economics
            and self.minimum_tariff_multiplier > 0.0
            else 1.0
            if self.experimental_unbounded_economics
            else float(np.clip((target - lb) / max(1e-6, ub - lb), 0.0, 1.0))
        )
        upper_distance = 1.0 if self.experimental_unbounded_economics else float(
            np.clip((ub - target) / max(1e-6, ub - lb), 0.0, 1.0)
        )
        self.last_action_descriptor = ActionDescriptor(
            action_id=int(action), target=key, direction=realized_direction,
            intended_step=float(direction) * step, realized_delta=realized,
            realized_delta_norm=float(realized / denominator),
            pre_value=current, post_value=float(target),
            lower_distance=lower_distance, upper_distance=upper_distance,
            magnitude_multiplier=multiplier, magnitude_level=abs(realized) / max(step, 1e-8),
            repeat_count=int(self._repeat_action_count), was_clipped=clipped,
            is_reversal=reversal, reversal_count=int(self._reversal_count),
        )
