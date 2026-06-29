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
import numpy as np
from collections import deque
from itertools import product

from PPOAgent import PPOAgent
from Market_models import CoefficientOverrides
from optim_config import default_specs_for

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

        self.step_base_fare = 0.10
        self.step_per_minute = 0.01
        self.step_per_mile = 0.05
        self.step_booking_fee = 0.10
        self.step_airport_fee = 0.10
        self.base_bounds = (1.50, 6.00)
        self.pmin_bounds = (0.10, 1.00)
        self.pmile_bounds = (0.50, 4.00)
        self.booking_bounds = (0.00, 6.00)
        self.airport_bounds = (0.00, 12.00)

        self.target_share = 0.50
        self.share_band = 0.03
        self.price_gap_threshold = 0.50  # Firm2 - Firm1

        self.alpha = 0.2
        self.ema_share = 0.50
        self.ema_gap = 0.0

        self.cooldown = 0
        self.cooldown_H = 3

    @staticmethod
    def _ema(x: float, ema: float, a: float) -> float:
        return (1 - a) * ema + a * x

    def act(self, city_base: float, city_pmin: float, hour: int, weather: str):
        # initialize once
        if "base_fare" in self.managed_keys and self.overrides.base_fare is None:
            self.overrides.base_fare = float(city_base)
        if "per_minute" in self.managed_keys and self.overrides.per_minute is None:
            self.overrides.per_minute = float(city_pmin)
        if "per_mile" in self.managed_keys and self.overrides.per_mile is None:
            self.overrides.per_mile = 1.50
        if "booking_fee" in self.managed_keys and self.overrides.booking_fee is None:
            self.overrides.booking_fee = 2.00
        if "airport_fee" in self.managed_keys and self.overrides.airport_fee is None:
            self.overrides.airport_fee = 5.00

        if self.cooldown > 0:
            self.cooldown -= 1
            return

        peak = (7 <= hour < 10) or (16 <= hour < 19)
        bad_weather = weather in ("rain", "snow")

        # baseline targets (schedule)
        base_target = city_base + (0.20 if peak else 0.0) + (0.10 if bad_weather else 0.0)
        pmin_target = city_pmin + (0.01 if peak else 0.0) + (0.005 if bad_weather else 0.0)
        pmile_target = 1.50 + (0.08 if peak else 0.0) + (0.05 if bad_weather else 0.0)
        booking_target = 2.00 + (0.08 if peak else 0.0) + (0.04 if bad_weather else 0.0)
        airport_target = 5.00 + (0.20 if peak else 0.0)

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

    def update(self, metrics: FirmMetrics, price_gap_mean: float):
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
        self.high_share_threshold = 0.62
        self.low_share_threshold = 0.38
        self.guardrail_cooldown_H = 2

    def update(self, metrics: FirmMetrics, price_gap_mean: float):
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


class FirmRandomWalkPricer(FirmHeuristicPricer):
    """Heuristic pricer with occasional bounded random exploration."""

    def __init__(self, seed: Optional[int] = None, managed_keys: Optional[List[str]] = None):
        super().__init__(seed=seed, managed_keys=managed_keys)
        self.exploration_prob = 0.20

    def act(self, city_base: float, city_pmin: float, hour: int, weather: str):
        super().act(city_base=city_base, city_pmin=city_pmin, hour=hour, weather=weather)

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
    
    def __init__(self, seed: Optional[int], opt_keys: List[str], state_frame_stack: int = 4):
        # Shared action manipulates up to five pricing coefficients per step.
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
        
        self.base_step_scale = 0.95
        self.converged_step_scale = 0.5
        self.step_scale = self.base_step_scale
        self.repeat_action_decay = 0.55
        self.repeat_action_min_scale = 0.20
        self._last_applied_action = -1
        self._repeat_action_count = 0
        self.base_max_relative_dev = 0.35
        self.converged_max_relative_dev = 0.25
        self.max_relative_dev = self.base_max_relative_dev
        # Keep supply-side control available as an internal multiplier, but do
        # not expose it as a PPO action by default.  Early optimization is
        # easier when PPO only credits rider-facing price moves.
        self.supply_incentive_multiplier = 1.0
        self.supply_step = 0.025
        self.supply_min_multiplier = 0.90
        self.supply_max_multiplier = 1.15
        self.recovery_share_threshold = 0.30
        self.recovery_gap_threshold = -0.05
        self.aggressive_actions = set()
        self.allow_aggressive_actions = True
        
        # Response-aware MDP action: a centralized pricing intervention is a
        # bounded vector of coefficient adjustments, not a command to one rider
        # and not a one-knob nudge.  Use the discrete product {-1, 0, +1}^d so
        # PPO can learn coordinated fare updates across all managed coefficients
        # in one decision while projection below enforces feasibility.
        all_keys = ["base_fare", "per_minute", "per_mile", "booking_fee", "airport_fee"]
        active_keys = [k for k in all_keys if k in self.opt_keys]
        

        self.action_to_steps: Dict[int, Dict[str, int]] = {}
        for a_idx, step_tuple in enumerate(product((-1, 0, 1), repeat=len(active_keys))):
            self.action_to_steps[a_idx] = {
                key: int(step) for key, step in zip(active_keys, step_tuple)
            }
        # Put the hold action at index 0 for stable cold starts and diagnostics.
        hold_idx = next(
            (idx for idx, steps in self.action_to_steps.items() if all(v == 0 for v in steps.values())),
            0,
        )
        if hold_idx != 0:
            self.action_to_steps[0], self.action_to_steps[hold_idx] = (
                self.action_to_steps[hold_idx],
                self.action_to_steps[0],
            )
        action_dim = len(self.action_to_steps)
        self.action_keys = list(active_keys)
        
        # direct supply-state features, five normalized fare-coefficient deltas,
        # ten belief/action-memory stress features, and seven constrained-MDP
        # context features. Frame stacking appends recent encoded states so PPO
        # can infer hidden demand/supply feedback without requiring a recurrent
        # policy.
        self.single_state_dim = 50
        state_dim = self.single_state_dim * self.state_frame_stack

        # Initialize PPO agent.
        self.agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=192,
            clip_eps=0.20,
            final_clip_eps=0.10,
            max_grad_norm=0.8,
            ent_coeff=0.024,
            min_ent_coeff=0.0005,
            ent_decay=0.992,
            target_kl=0.050,
            max_lr=1.2e-3,
            value_clip_eps=0.30,
            initial_exploration_rate=0.58,
            final_exploration_rate=0.02,
            exploration_fraction=0.90,
            exploration_warmup_fraction=0.35,
            min_action_visits=2,
            exploration_rescue_rate=0.25,
        )
    
    def action_steps(self, action: int) -> Dict[str, int]:
        """Return the decoded bounded coefficient intervention for diagnostics."""
        return dict(self.action_to_steps.get(int(action), {}))

    def action_label(self, action: int) -> str:
        """Compact human-readable label for a discrete vector action."""
        steps = self.action_steps(action)
        if not steps or all(int(v) == 0 for v in steps.values()):
            return "hold"
        return ",".join(f"{k}:{int(v):+d}" for k, v in steps.items() if int(v) != 0)
        
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
        self.last_action_normalized_gap = {}

    def stack_state(self, state: np.ndarray) -> np.ndarray:
        """Return a fixed-width frame stack ending with the current state.

        The first observation is repeated to fill the stack, avoiding an all-zero
        cold start that would be out-of-distribution relative to later states.
        """
        current = np.asarray(state, dtype=np.float32).reshape(-1)
        if current.size != self.single_state_dim:
            raise ValueError(f"Single state dim mismatch: got {current.size}, expected {self.single_state_dim}")
        if not self._state_history:
            for _ in range(self.state_frame_stack - 1):
                self._state_history.append(current.copy())
        self._state_history.append(current.copy())
        frames = list(self._state_history)
        while len(frames) < self.state_frame_stack:
            frames.insert(0, current.copy())
        return np.concatenate(frames[-self.state_frame_stack:]).astype(np.float32, copy=False)
        
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
    ) -> None:
        """Guard against extreme dynamic-market failure modes.

        The guardrail is intentionally small: PPO still owns the strategy, but
        it should not remain stuck in obviously dominated regions such as
        overpricing with collapsing share, negative-profit share buying, or
        severe driver-fulfillment stress.  Adjust all rider-facing fare knobs
        around market anchors instead of only base fare/per-minute.
        """
        share_f = float(np.clip(share, 0.0, 1.0))
        gap = float(price_gap_f2_minus_f1)
        profit = float(profit_per_request)
        fulfill = float(np.clip(fulfillment_rate, 0.0, 1.0))
        target_gap = float(target_price_gap)
        gap_tol = float(max(0.05, target_gap_tolerance))
        lower_gap = min(target_gap - gap_tol, target_gap)
        upper_gap = target_gap + gap_tol
        overpricing = share_f < self.recovery_share_threshold or gap < max(self.recovery_gap_threshold, lower_gap)
        over_discounting = gap > 1.50 and share_f >= self.recovery_share_threshold
        loss_buying_share = profit < 0.0 and share_f >= self.recovery_share_threshold and gap >= -0.50
        supply_stress = fulfill < 0.75 and profit >= -0.50
        if not (overpricing or over_discounting or loss_buying_share or supply_stress):
            return

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
            nudged = float(curr) + float(np.sign(direction)) * step
            setattr(
                self.overrides,
                key,
                self._bounded_relative_move(nudged, anchor, self.max_relative_dev, lb, ub),
            )
        
        if overpricing:
            # Restore competitiveness first through fixed fees, then segment
            # levers if the mean gap shows a severe broad overprice.
            move("base_fare", -1)
            move("booking_fee", -1)
            move("per_minute", -1, 0.5)
            if gap < -1.00:
                move("airport_fee", -1)
            if gap < -2.00:
                move("per_mile", -1, 0.75)
            return
        
        if over_discounting:
            # If the realized market gap is already wider than the intended
            # modest discount, lift broad/fixed fare levers before allowing the
            # policy to keep compensating through one distance-heavy coefficient.
            move("base_fare", +1)
            move("booking_fee", +1)
            move("per_minute", +1, 0.5)
            if gap > 2.25:
                move("per_mile", +1, 0.5)
            return

        if loss_buying_share:
            # Repair unit economics without immediately giving up all share.
            move("per_mile", +1, 0.75)
            move("per_minute", +1, 0.75)
            if gap > 0.75:
                move("booking_fee", +1, 0.5)
            return

        if supply_stress:
            # Temper demand only when Firm1 is at or above its intended discount
            # band.  Otherwise this guardrail becomes a one-way variable-fare
            # ratchet that can erase competitiveness and trigger oscillations.
            if gap >= upper_gap:
                move("per_minute", +1, 0.25)
                move("per_mile", +1, 0.25)
            
    def apply_action(self, action: int, market_interaction) -> None:
        """Map discrete steps back into concrete market coefficient overrides."""
        self.last_action_normalized_gap = {}
        if action not in self.action_to_steps:
            return
        
        
        if (not self.allow_aggressive_actions) and action in self.aggressive_actions:
            action = 0
        
        step_map = dict(self.action_to_steps[action])
        if not any(int(v) != 0 for v in step_map.values()):
            self._last_applied_action = int(action)
            self._repeat_action_count = 0
            self.last_action_steps = step_map
            return
        
        if int(action) == self._last_applied_action and int(action) != 0:
            self._repeat_action_count += 1
        else:
            self._last_applied_action = int(action)
            self._repeat_action_count = 1 if int(action) != 0 else 0
        repeat_scale = 1.0
        if self._repeat_action_count > 1:
            repeat_scale = max(
                self.repeat_action_min_scale,
                self.repeat_action_decay ** float(self._repeat_action_count - 1),
            )

        self.last_action_steps = dict(step_map)
        fare_step_map = {k: v for k, v in step_map.items() if k in self.opt_keys and int(v) != 0}
        supply_step = int(step_map.get("supply_incentive", 0))
        if supply_step:
            self.supply_incentive_multiplier = float(np.clip(
                self.supply_incentive_multiplier + supply_step * self.supply_step * self.step_scale * repeat_scale,
                self.supply_min_multiplier,
                self.supply_max_multiplier,
            ))
        if not fare_step_map:
            self.last_action_normalized_gap = {"supply_incentive": float(supply_step) * self.supply_step} if supply_step else {}
            return

        scaled_steps = {k: self.config.step[k] * self.step_scale * repeat_scale for k in fare_step_map.keys()}
        bounds = {k: self.config.bounds[k] for k in fare_step_map.keys()}
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
        for key in fare_step_map:
            anchor = float(getattr(base, key))
            lb, ub = bounds[key]
            current = float(getattr(self.overrides, key))
            setattr(
                self.overrides,
                key,
                self._bounded_relative_move(current, anchor, self.max_relative_dev, lb, ub),
            )

