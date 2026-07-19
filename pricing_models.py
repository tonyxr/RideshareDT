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
from typing import Deque, Optional, Dict, List, Tuple
import mkl_config  # noqa: F401 - set oneMKL env before NumPy/Torch
import numpy as np
from collections import deque

from PPOAgent import PPOAgent
from Market_models import CoefficientOverrides
from optim_config import default_specs_for

@dataclass
class ActionDescriptor:
    """Executed RL-selected coefficient intervention metadata.

    PPO chooses hold, or exactly one rider-facing price coefficient together
    with its direction and a continuous magnitude.  Bounds are still enforced
    by the simulator-facing application step, but there is no separate lower-
    layer heuristic deciding how large the selected manipulation should be.
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
        return (self.revenue / self.total) if self.total > 0 else 0.0
    
    @property
    def profit_per_request(self) -> float:
        return (self.profit / self.total) if self.total > 0 else 0.0
    
    @property
    def fulfillment_rate(self) -> float:
        return (self.completed / self.chosen) if self.chosen > 0 else 1.0

    @property
    def avg_wait_minutes(self) -> float:
        return (self.wait_minutes / self.completed) if self.completed > 0 else 0.0

    @property
    def driver_acceptance_rate(self) -> float:
        return ((self.dispatch_offers - self.driver_rejections) / self.dispatch_offers) if self.dispatch_offers > 0 else 1.0
    
def build_discrete_action_space(opt_keys: List[str], max_coeffs: int = 5) -> Tuple[Dict[int, Dict[str, int]], List[str]]:
    """Build single-lever and coordinated tariff interventions.

    A one-coefficient action space cannot express common pricing responses such
    as moving fixed fees together, changing time and distance rates together, or
    rebalancing short- versus long-trip prices.  The expanded representation
    keeps every action interpretable while allowing the actor to make coherent
    multivariate moves with one shared continuous magnitude.
    """
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

    coordinated_groups = (
        ("fixed_fees", ("base_fare", "booking_fee")),
        ("usage_rates", ("per_minute", "per_mile")),
        ("core_tariff", ("base_fare", "per_minute", "per_mile", "booking_fee")),
        ("airport_trip", ("base_fare", "booking_fee", "airport_fee")),
    )
    for _, keys in coordinated_groups:
        present = [key for key in keys if key in active_keys]
        if len(present) >= 2:
            for direction in (-1, 1):
                add_action({key: direction for key in present})

    # Segment rebalancing changes fixed and distance-sensitive levers in
    # opposite directions.  This lets the controller react when short- and
    # long-trip quote gaps move differently.
    fixed = [key for key in ("base_fare", "booking_fee") if key in active_keys]
    variable = [key for key in ("per_minute", "per_mile") if key in active_keys]
    if fixed and variable:
        for short_direction in (-1, 1):
            changes = {key: short_direction for key in fixed}
            changes.update({key: -short_direction for key in variable})
            add_action(changes)
    return action_to_steps, active_keys

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
        super().update(metrics=metrics, price_gap_mean=price_gap_mean)

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
        state_frame_stack: int = 4,
        single_state_dim: int = 89,
        action_feature_dim: int = 19,
        constraint_dim: int = 5,
        response_dim: int = 12,
    ):
        # Structured response-aware action design:
        #   1) the PPO actor selects hold, a single lever, or a coordinated
        #      multivariate tariff template;
        #   2) a shared continuous magnitude controls the intervention size;
        #   3) the simulator provides realized response targets used by the PPO
        #      response head/world-model auxiliary loss.
        # This removes the former lower-layer magnitude heuristic: the learned
        # policy owns both "what to manipulate" and "how much to manipulate",
        # while the application layer only clips to feasibility bounds.
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
        
        self.base_step_scale = 1.75
        self.converged_step_scale = 1.0
        self.step_scale = self.base_step_scale
        self.repeat_action_decay = 0.85
        self.repeat_action_min_scale = 0.50
        self._last_applied_action = -1
        self._repeat_action_count = 0
        self.base_max_relative_dev = 0.60
        self.converged_max_relative_dev = 0.45
        self.max_relative_dev = self.base_max_relative_dev
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
        self._last_action_target = "hold"
        self._last_action_direction = 0
        self._reversal_count = 0
        self.recovery_share_threshold = 0.25
        self.recovery_gap_threshold = -0.05
        self.aggressive_actions = set()
        self.allow_aggressive_actions = True
        
        # Hybrid manipulation actions.  Index 0 is hold/status quo; remaining
        # actions include both individual and coordinated coefficient changes.
        # PPO's continuous magnitude head supplies the shared step multiplier.
        self.action_to_steps, self.action_keys = build_discrete_action_space(self.opt_keys, self.MAX_MANIPULATED_COEFFS)
        action_dim = len(self.action_to_steps)
        
        # State includes cyclical/flag time context, richer demand/WTP context,
        # recent EMA/delta features, direct supply state, own and opponent fare-
        # coefficient deltas, action-memory/oscillation stress features, and
        # constrained-MDP context. Opponent deltas help PPO respond to heuristic
        # rivals whose coefficients move before share/gap metrics fully react.
        # Frame stacking appends recent encoded states so PPO can infer hidden
        # demand/supply feedback without requiring a recurrent policy.
        self.single_state_dim = int(max(1, single_state_dim))
        state_dim = self.single_state_dim * self.state_frame_stack

        # Initialize PPO agent.
        self.agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=192,
            clip_eps=0.20,
            final_clip_eps=0.10,
            max_grad_norm=0.8,
            ent_coeff=0.010,
            min_ent_coeff=0.0005,
            ent_decay=0.992,
            target_kl=0.030,
            max_lr=6.0e-4,
            value_clip_eps=0.30,
            initial_exploration_rate=0.20,
            final_exploration_rate=0.02,
            action_feature_dim=self.action_feature_dim,
            constraint_dim=int(max(1, constraint_dim)),
            response_dim=int(max(1, response_dim)),
            action_q_coeff=0.10,
            response_coeff=0.12,
            delayed_reward_horizon=8,
            delayed_reward_blend=0.30,
            single_state_dim=self.single_state_dim,
            frame_stack=self.state_frame_stack,
            state_action_mi_coeff=0.025,
            collapse_rescue_updates=8,
            exploration_fraction=0.65,
            exploration_warmup_fraction=0.10,
            min_action_visits=1,
            exploration_rescue_rate=0.08,
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
        """Compact human-readable label for a discrete vector action."""
        steps = self.action_steps(action)
        if not steps or all(int(v) == 0 for v in steps.values()):
            return "hold"
        return ",".join(f"{k}:{int(v):+d}@continuous" for k, v in steps.items() if int(v) != 0)
        
    def last_action_magnitude(self) -> float:
        """Return the normalized size of the most recently applied action."""
        return float(
            np.clip(
                sum(abs(v) for v in getattr(self, "last_action_normalized_gap", {}).values()),
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
        self.last_action_descriptor = ActionDescriptor()
        self.last_action_was_saturated = False
        self.last_action_was_zero_effect = False
        
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
        """Mask interventions that cannot produce a nonzero bounded coefficient move."""
        mask = np.zeros(len(self.action_to_steps), dtype=bool)
        mask[0] = True
        base = market_interaction.curr_market
        for action, step_map in self.action_to_steps.items():
            if int(action) == 0:
                continue
            active = [(k, int(v)) for k, v in step_map.items() if k in self.opt_keys and int(v) != 0]
            if not active:
                continue
            feasible = True
            for key, direction in active:
                anchor = float(getattr(base, key))
                current_raw = getattr(self.overrides, key)
                current = anchor if current_raw is None else float(current_raw)
                lb, ub = self.config.bounds[key]
                floor = max(float(lb), anchor * (1.0 - self.max_relative_dev))
                ceil = min(float(ub), anchor * (1.0 + self.max_relative_dev))
                if floor > ceil:
                    floor, ceil = float(lb), float(ub)
                if direction < 0:
                    feasible = feasible and current > floor + 1e-8
                else:
                    feasible = feasible and current < ceil - 1e-8
            mask[int(action)] = bool(feasible)
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
        """Return per-option causal features used by the action-conditioned PPO head.

        Features expose the option identity, a neutral continuous-magnitude prior,
        expected unit price impact on short/long/airport/peak trips, bound pressure,
        and recent crowd-response distribution summaries.
        """
        ctx = crowd_context or {}
        base = market_interaction.curr_market
        rows: List[List[float]] = []
        near_threshold = float(np.clip(ctx.get("near_threshold_share", 0.0), 0.0, 1.0))
        threshold_mean = float(np.clip(ctx.get("price_threshold_mean", 1.5) / 8.0, 0.0, 1.0))
        no_ride_rate = float(np.clip(ctx.get("no_ride_rate", 0.0), 0.0, 1.0))
        peak = float(np.clip(ctx.get("peak_context", 0.0), 0.0, 1.0))
        airport_rate = float(np.clip(ctx.get("airport_rate", 0.0), 0.0, 1.0))
        for action in range(len(self.action_to_steps)):
            step_map = self.action_steps(action)
            active = [(k, int(v)) for k, v in step_map.items() if int(v) != 0 and k in self.action_keys]
            if not active:
                rows.append([1.0, 0.0, *([0.0] * 5), 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, near_threshold, threshold_mean, no_ride_rate, peak, airport_rate])
                continue
            key, direction = active[0]
            key_onehot = [1.0 if key == k else 0.0 for k in ["base_fare", "per_minute", "per_mile", "booking_fee", "airport_fee"]]
            curr = getattr(self.overrides, key)
            anchor = float(getattr(base, key))
            curr_f = anchor if curr is None else float(curr)
            lb, ub = self.config.bounds[key]
            width = max(1e-6, float(ub - lb))
            rel_dev = float(np.clip((curr_f - anchor) / max(abs(anchor), 1e-6), -1.0, 1.0))
            lower_dist = float(np.clip((curr_f - lb) / width, 0.0, 1.0))
            upper_dist = float(np.clip((ub - curr_f) / width, 0.0, 1.0))
            # The realized magnitude is sampled after this matrix is built. Use
            # a neutral unit multiplier so action scoring stays about target/
            # direction feasibility while the continuous head learns size.
            delta = float(direction) * self.config.step[key] * self.step_scale
            # Approximate heterogeneous price exposure before time/weather/service multipliers.
            short_impact = delta if key in {"base_fare", "booking_fee"} else delta * (2.0 if key == "per_mile" else 8.0 if key == "per_minute" else 0.0)
            long_impact = delta if key in {"base_fare", "booking_fee"} else delta * (10.0 if key == "per_mile" else 28.0 if key == "per_minute" else 0.0)
            airport_impact = delta if key == "airport_fee" else short_impact
            rows.append([
                0.0,
                float(direction),
                *key_onehot,
                rel_dev,
                lower_dist,
                upper_dist,
                float(np.clip(delta / width, -1.0, 1.0)),
                float(np.clip(short_impact / 8.0, -1.0, 1.0)),
                float(np.clip(long_impact / 20.0, -1.0, 1.0)),
                float(np.clip(airport_impact / 12.0, -1.0, 1.0)),
                near_threshold,
                threshold_mean,
                no_ride_rate,
                peak,
                airport_rate,
            ])
        return np.asarray(rows, dtype=np.float32)
        
    def configure_training_controls(self, progress: float, reward_converged: bool, reward_std: float) -> None:
        """Adapt exploration and action aggressiveness across training phases."""
        p = float(np.clip(progress, 0.0, 1.0))
        stable = bool(reward_converged or (p > 0.80 and reward_std <= 0.03))

        self.allow_aggressive_actions = not stable
        self.step_scale = self.converged_step_scale if stable else self.base_step_scale
        self.max_relative_dev = self.converged_max_relative_dev if stable else self.base_max_relative_dev
        self.agent.adapt_optimization(progress=p, reward_converged=stable)
        self.agent.adapt_entropy(progress=p, reward_converged=stable)
        self.agent.adapt_exploration(
            progress=p,
            reward_converged=stable,
            reward_std=reward_std,
        )
    
    @staticmethod
    def _bounded_relative_move(value: float, anchor: float, max_relative_dev: float, lb: float, ub: float) -> float:
        """Clamp a value to both absolute bounds and a relative envelope around anchor."""
        floor = max(lb, anchor * (1.0 - max_relative_dev))
        ceil = min(ub, anchor * (1.0 + max_relative_dev))
        if floor > ceil:
            floor, ceil = lb, ub
        return float(np.clip(value, floor, ceil))

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
        """Guard against extreme dynamic-market failure modes and report effects.

        The guardrail is intentionally small: PPO still owns the strategy, but
        it should not remain stuck in obviously dominated regions such as
        overpricing with collapsing share, negative-profit share buying, or
        severe driver-fulfillment stress.  Returning a structured diagnostic lets
        evaluation separate learned RL actions from rule-based safety projection.
        """
        before = {k: getattr(self.overrides, k) for k in self.opt_keys}
        moved: Dict[str, float] = {}
        reasons: List[str] = []
        share_f = float(np.clip(share, 0.0, 1.0))
        gap = float(price_gap_f2_minus_f1)
        profit = float(profit_per_request)
        fulfill = float(np.clip(fulfillment_rate, 0.0, 1.0))
        target_gap = float(target_price_gap)
        gap_tol = float(max(0.05, target_gap_tolerance))
        lower_gap = min(target_gap - gap_tol, target_gap)
        upper_gap = target_gap + gap_tol
        overpricing = share_f < 0.22 or gap < (max(self.recovery_gap_threshold, lower_gap) - 0.50)
        over_discounting = gap > 2.25 and share_f >= self.recovery_share_threshold
        loss_buying_share = profit < -1.00 and share_f >= self.recovery_share_threshold and gap >= 0.00
        supply_stress = fulfill < 0.60 and profit >= -0.50
        if not (overpricing or over_discounting or loss_buying_share or supply_stress):
            return {"applied": False, "reasons": [], "deltas": {}, "before": before, "after": dict(before)}

        anchors = {
            "base_fare": float(city_base),
            "per_minute": float(city_pmin),
            "per_mile": float(city_pmile if city_pmile is not None else 1.50),
            "booking_fee": float(city_booking if city_booking is not None else max(0.0, city_base - 0.25)),
            "airport_fee": float(city_airport if city_airport is not None else 5.0),
        }

        def move(key: str, direction: int, multiplier: float = 1.0) -> None:
            if key not in self.opt_keys:
                return
            curr = getattr(self.overrides, key)
            anchor = anchors[key]
            if curr is None:
                curr = anchor
            step = self.config.step[key] * self.step_scale * float(multiplier)
            lb, ub = self.config.bounds[key]
            old = float(curr)
            nudged = old + float(np.sign(direction)) * step
            new_value = self._bounded_relative_move(nudged, anchor, self.max_relative_dev, lb, ub)
            setattr(self.overrides, key, new_value)
            delta = float(new_value - old)
            if abs(delta) > 1e-12:
                moved[key] = moved.get(key, 0.0) + delta
        
        if overpricing:
            reasons.append("overpricing")
            # Restore competitiveness first through fixed fees, then segment
            # levers if the mean gap shows a severe broad overprice.
            move("base_fare", -1)
            move("booking_fee", -1)
            move("per_minute", -1, 0.5)
            if gap < -1.00:
                move("airport_fee", -1)
            if gap < -2.00:
                move("per_mile", -1, 0.75)
        elif over_discounting:
            reasons.append("over_discounting")
            # If the realized market gap is already wider than the intended
            # modest discount, lift broad/fixed fare levers before allowing the
            # policy to keep compensating through one distance-heavy coefficient.
            move("base_fare", +1)
            move("booking_fee", +1)
            move("per_minute", +1, 0.5)
            if gap > 2.25:
                move("per_mile", +1, 0.5)
        elif loss_buying_share:
            reasons.append("loss_buying_share")
            # Repair unit economics without immediately giving up all share.
            move("per_mile", +1, 0.75)
            move("per_minute", +1, 0.75)
            if gap > 0.75:
                move("booking_fee", +1, 0.5)
        elif supply_stress:
            reasons.append("supply_stress")
            # Temper demand only when Firm1 is at or above its intended discount
            # band.  Otherwise this guardrail becomes a one-way variable-fare
            # ratchet that can erase competitiveness and trigger oscillations.
            if gap >= upper_gap:
                move("per_minute", +1, 0.25)
                move("per_mile", +1, 0.25)
        after = {k: getattr(self.overrides, k) for k in self.opt_keys}
        return {
            "applied": bool(moved),
            "reasons": reasons,
            "deltas": moved,
            "before": before,
            "after": after,
        }
            
    def apply_action(self, action: int, market_interaction) -> None:
        """Map discrete steps back into concrete market coefficient overrides."""
        self.last_action_normalized_gap = {}
        self.last_action_was_saturated = False
        self.last_action_was_zero_effect = False
        self.last_action_descriptor = ActionDescriptor(action_id=int(action))
        if action not in self.action_to_steps:
            return
        
        if (not self.allow_aggressive_actions) and action in self.aggressive_actions:
            action = 0
        
        step_map = dict(self.action_to_steps[action])
        if not any(int(v) != 0 for v in step_map.values()):
            self._last_applied_action = int(action)
            self._repeat_action_count = 0
            self._last_action_target = "hold"
            self._last_action_direction = 0
            self._reversal_count = 0
            self.last_action_steps = step_map
            self.last_action_descriptor = ActionDescriptor(action_id=int(action), target="hold", direction=0)
            return
        
        if int(action) == self._last_applied_action and int(action) != 0:
            self._repeat_action_count += 1
        else:
            self._last_applied_action = int(action)
            self._repeat_action_count = 1 if int(action) != 0 else 0
        

        self.last_action_steps = dict(step_map)
        fare_step_map = {k: v for k, v in step_map.items() if k in self.opt_keys and int(v) != 0}
        supply_step = int(step_map.get("supply_incentive", 0))
        if supply_step:
            self.supply_incentive_multiplier = float(np.clip(
                self.supply_incentive_multiplier + supply_step * self.supply_step * self.step_scale,
                self.supply_min_multiplier,
                self.supply_max_multiplier,
            ))
        if not fare_step_map:
            self.last_action_normalized_gap = {"supply_incentive": float(supply_step) * self.supply_step} if supply_step else {}
            return
        
        action_magnitude = float(np.clip(getattr(self.agent, "last_continuous_magnitude", 1.0), 0.0, 2.0))
        multipliers = {k: action_magnitude for k in fare_step_map.keys()}
        scaled_steps = {
            k: self.config.step[k]
            * self.step_scale
            * multipliers[k]
            for k in fare_step_map.keys()
        }
        pre_values = {}
        for k in fare_step_map.keys():
            curr = getattr(self.overrides, k)
            pre_values[k] = float(getattr(market_interaction.curr_market, k) if curr is None else curr)
        bounds = {k: self.config.bounds[k] for k in fare_step_map.keys()}
        invalid_bound_actions = []
        for key, direction in fare_step_map.items():
            lb, ub = bounds[key]
            width = max(1e-6, float(ub - lb))
            pre = float(pre_values.get(key, getattr(market_interaction.curr_market, key)))
            anchor = float(getattr(market_interaction.curr_market, key))
            effective_floor = max(float(lb), anchor * (1.0 - self.max_relative_dev))
            effective_ceil = min(float(ub), anchor * (1.0 + self.max_relative_dev))
            if effective_floor > effective_ceil:
                effective_floor, effective_ceil = float(lb), float(ub)
            lower_distance = float((pre - lb) / width)
            upper_distance = float((ub - pre) / width)
            if int(direction) < 0 and (lower_distance <= 1e-4 or pre <= effective_floor + 1e-8):
                invalid_bound_actions.append((key, int(direction), pre, lb, ub))
            elif int(direction) > 0 and (upper_distance <= 1e-4 or pre >= effective_ceil - 1e-8):
                invalid_bound_actions.append((key, int(direction), pre, lb, ub))
        if invalid_bound_actions and len(invalid_bound_actions) == len(fare_step_map):
            key, direction, pre, lb, ub = invalid_bound_actions[0]
            width = max(1e-6, float(ub - lb))
            if int(action) == self._last_applied_action and int(action) != 0:
                self._repeat_action_count += 1
            else:
                self._last_applied_action = int(action)
                self._repeat_action_count = 1 if int(action) != 0 else 0
            self._last_action_target = str(key)
            self._last_action_direction = int(direction)
            self.last_action_was_saturated = True
            self.last_action_was_zero_effect = True
            self.last_action_steps = dict(step_map)
            self.last_action_descriptor = ActionDescriptor(
                action_id=int(action),
                target=str(key),
                direction=int(direction),
                intended_step=0.0,
                realized_delta=0.0,
                realized_delta_norm=0.0,
                pre_value=float(pre),
                post_value=float(pre),
                lower_distance=float(np.clip((pre - lb) / width, 0.0, 1.0)),
                upper_distance=float(np.clip((ub - pre) / width, 0.0, 1.0)),
                magnitude_multiplier=0.0,
                magnitude_level=0.0,
                repeat_count=int(self._repeat_action_count),
                was_clipped=True,
                is_reversal=False,
                reversal_count=int(self._reversal_count),
            )
            return
        # Detailed normalized one-unit gap per coefficient:
        # one step => config.step[k], normalized by feasible range width.
        self.last_action_normalized_gap = {
            k: float(fare_step_map[k]) * float(scaled_steps[k]) / max(1e-6, float(bounds[k][1] - bounds[k][0]))
            for k in fare_step_map.keys()
        }
        if supply_step:
            self.last_action_normalized_gap["supply_incentive"] = float(supply_step) * self.supply_step
        market_interaction.apply_step_actions_to_overrides(
            overrides=self.overrides,
            action_steps=fare_step_map,
            step_size=scaled_steps,
            bounds=bounds,
        )
        
        base = market_interaction.curr_market
        realized_by_key: Dict[str, float] = {}
        intended_by_key: Dict[str, float] = {}
        lower_distances: List[float] = []
        upper_distances: List[float] = []
        clipped_any = False
        for key in fare_step_map:
            anchor = float(getattr(base, key))
            lb, ub = bounds[key]
            current = float(getattr(self.overrides, key))
            bounded = self._bounded_relative_move(current, anchor, self.max_relative_dev, lb, ub)
            setattr(self.overrides, key, bounded)
            width = max(1e-6, float(ub - lb))
            pre = float(pre_values.get(key, anchor))
            realized = float(bounded - pre)
            intended = float(np.sign(fare_step_map[key]) * scaled_steps[key])
            realized_by_key[key] = realized
            intended_by_key[key] = intended
            if abs(realized) <= 1e-8 and abs(intended) > 1e-8:
                self.last_action_was_zero_effect = True
            lower_distances.append(float(np.clip((bounded - lb) / width, 0.0, 1.0)))
            upper_distances.append(float(np.clip((ub - bounded) / width, 0.0, 1.0)))
            clipped_any = clipped_any or bool(abs(realized - intended) > 1e-8)

        target = "+".join(sorted(fare_step_map))
        signed_intended = float(sum(
            intended_by_key[key] / max(1e-6, bounds[key][1] - bounds[key][0])
            for key in fare_step_map
        ))
        aggregate_direction = int(np.sign(signed_intended))
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
            intended_step=float(sum(intended_by_key.values())),
            realized_delta=float(sum(realized_by_key.values())),
            realized_delta_norm=float(sum(
                realized_by_key[key] / max(1e-6, bounds[key][1] - bounds[key][0])
                for key in fare_step_map
            )),
            pre_value=float(np.mean(list(pre_values.values()))),
            post_value=float(np.mean([float(getattr(self.overrides, key)) for key in fare_step_map])),
            lower_distance=float(min(lower_distances) if lower_distances else 0.0),
            upper_distance=float(min(upper_distances) if upper_distances else 0.0),
            magnitude_multiplier=float(action_magnitude),
            magnitude_level=float(action_magnitude),
            repeat_count=int(self._repeat_action_count),
            was_clipped=clipped_any,
            is_reversal=is_reversal,
            reversal_count=int(self._reversal_count),
        )
