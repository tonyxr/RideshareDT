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
from typing import Optional, Dict, List, Tuple
import numpy as np

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
            if "base_fare" in self.managed_keys and abs(float(self.overrides.base_fare) - base_target) > 0.15:
                chosen = "base"
                direction = +1 if float(self.overrides.base_fare) < base_target else -1
            elif "per_minute" in self.managed_keys and abs(float(self.overrides.per_minute) - pmin_target) > 0.02:
                chosen = "pmin"
                direction = +1 if float(self.overrides.per_minute) < pmin_target else -1
            elif "per_mile" in self.managed_keys and abs(float(self.overrides.per_mile) - pmile_target) > 0.08:
                chosen = "pmile"
                direction = +1 if float(self.overrides.per_mile) < pmile_target else -1
            elif "booking_fee" in self.managed_keys and abs(float(self.overrides.booking_fee) - booking_target) > 0.12:
                chosen = "booking"
                direction = +1 if float(self.overrides.booking_fee) < booking_target else -1
            elif "airport_fee" in self.managed_keys and abs(float(self.overrides.airport_fee) - airport_target) > 0.20:
                chosen = "airport"
                direction = +1 if float(self.overrides.airport_fee) < airport_target else -1
        
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
        if "base_fare" in self.managed_keys and self.overrides.base_fare is not None:
            self.overrides.base_fare = float(np.clip(
                float(self.overrides.base_fare) + direction * (0.5 * self.step_base_fare),
                *self.base_bounds,
            ))
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
    
    def __init__(self, seed: Optional[int], opt_keys: List[str]):
        # Shared action manipulates up to five pricing coefficients per step.
        self.opt_keys = list(opt_keys[: self.MAX_MANIPULATED_COEFFS])
        if len(opt_keys) > self.MAX_MANIPULATED_COEFFS:
            print(
                f"[FirmRLPricer] Received {len(opt_keys)} opt_keys; "
                f"limiting to first {self.MAX_MANIPULATED_COEFFS}: {self.opt_keys}"
            )
        self.config = default_specs_for(self.opt_keys)
        self.overrides = CoefficientOverrides()
        
        self.base_step_scale = 0.95
        self.converged_step_scale = 0.5
        self.step_scale = self.base_step_scale
        self.base_max_relative_dev = 0.35
        self.converged_max_relative_dev = 0.25
        self.max_relative_dev = self.base_max_relative_dev
        self.recovery_share_threshold = 0.30
        self.recovery_gap_threshold = -0.05
        self.aggressive_actions = set()
        self.allow_aggressive_actions = True
        
        # Use a compact macro-action space so rewards map to clear pricing moves.
        # The driver layer is observed but external: all actions still change only
        # rider-facing fare coefficients, never driver-pay policy.
        
        all_keys = ["base_fare", "per_minute", "per_mile", "booking_fee", "airport_fee"]
        active_keys = [k for k in all_keys if k in self.opt_keys]
        def zero_steps() -> Dict[str, int]:
            return {k: 0 for k in active_keys}

        self.action_to_steps: Dict[int, Dict[str, int]] = {0: zero_steps()}
        macro_specs = [
            {"base_fare": -1, "booking_fee": -1},
            {"base_fare": +1, "booking_fee": +1},
            {"per_minute": -1, "per_mile": -1},
            {"per_minute": +1, "per_mile": +1},
            {"base_fare": -1, "per_minute": -1, "per_mile": -1, "booking_fee": -1},
            {"base_fare": +1, "per_minute": +1, "per_mile": +1, "booking_fee": +1},
            {"airport_fee": -1},
            {"airport_fee": +1},
        ]
        a_idx = 1
        for spec in macro_specs:
            action = zero_steps()
            for key, step in spec.items():
                if key in action:
                    action[key] = int(step)
            if any(v != 0 for v in action.values()):
                self.action_to_steps[a_idx] = action
                a_idx += 1
        action_dim = len(self.action_to_steps)
        
        # State encoder emits 20 target-centered market/service features plus
        # normalized fare-coefficient deltas for the active rider-price knobs.
        state_dim = 20 + len(active_keys)

        # Initialize PPO agent.
        self.agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=192,
            clip_eps=0.2,
            max_grad_norm=0.8,
            ent_coeff=0.030,
            min_ent_coeff=0.0008,
            ent_decay=0.990,
            target_kl=0.030,
            max_lr=7e-4,
            value_clip_eps=0.20,
        )
        
    def configure_training_controls(self, progress: float, reward_converged: bool, reward_std: float) -> None:
        """Adapt exploration and action aggressiveness across training phases."""
        p = float(np.clip(progress, 0.0, 1.0))
        stable = bool(reward_converged or (p > 0.80 and reward_std <= 0.03))

        self.allow_aggressive_actions = not stable
        self.step_scale = self.converged_step_scale if stable else self.base_step_scale
        self.max_relative_dev = self.converged_max_relative_dev if stable else self.base_max_relative_dev
        self.agent.adapt_entropy(progress=p, reward_converged=stable)
    
    @staticmethod
    def _bounded_relative_move(value: float, anchor: float, max_relative_dev: float, lb: float, ub: float) -> float:
        """Clamp a value to both absolute bounds and a relative envelope around anchor."""
        floor = max(lb, anchor * (1.0 - max_relative_dev))
        ceil = min(ub, anchor * (1.0 + max_relative_dev))
        if floor > ceil:
            floor, ceil = lb, ub
        return float(np.clip(value, floor, ceil))

    def stabilize_after_batch(self, share: float, price_gap_f2_minus_f1: float, city_base: float, city_pmin: float) -> None:
        """Light-touch guardrails to avoid prolonged share-collapse from runaway pricing."""
        if share >= self.recovery_share_threshold and price_gap_f2_minus_f1 >= self.recovery_gap_threshold:
            return

        for key, anchor in (("base_fare", float(city_base)), ("per_minute", float(city_pmin))):
            curr = getattr(self.overrides, key)
            if curr is None:
                curr = anchor
            step = self.config.step[key] * self.step_scale
            lb, ub = self.config.bounds[key]
            nudged = float(curr) - step
            setattr(
                self.overrides,
                key,
                self._bounded_relative_move(nudged, anchor, self.max_relative_dev, lb, ub),
            )
        
    def apply_action(self, action: int, market_interaction) -> None:
        """Map discrete steps back into concrete market coefficient overrides."""
        if action not in self.action_to_steps:
            return
        
        
        if (not self.allow_aggressive_actions) and action in self.aggressive_actions:
            action = 0
        
        step_map = {k: v for k, v in self.action_to_steps[action].items() if k in self.opt_keys}
        if not step_map:
            return
        
        scaled_steps = {k: self.config.step[k] * self.step_scale for k in step_map.keys()}
        bounds = {k: self.config.bounds[k] for k in step_map.keys()}
        # Detailed normalized one-unit gap per coefficient:
        # one step => config.step[k], normalized by feasible range width.
        self.last_action_normalized_gap = {
            k: float(step_map[k]) * float(scaled_steps[k]) / max(1e-6, float(bounds[k][1] - bounds[k][0]))
            for k in step_map.keys()
        }
        market_interaction.apply_step_actions_to_overrides(
            overrides=self.overrides,
            action_steps=step_map,
            step_size=scaled_steps,
            bounds=bounds,
        )

