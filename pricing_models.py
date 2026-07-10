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

        self.alpha = 0.05
        self.ema_share = 0.50
        self.ema_gap = 0.0

        self.cooldown = 0
        self.cooldown_H = 8

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
        self.high_share_threshold = 0.5
        self.low_share_threshold = 0.38
        self.guardrail_cooldown_H = 6

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
        # Flat discrete response-aware action design:
        #   1) the PPO actor selects hold, or one coefficient/direction pair;
        #   2) the same PPO action also selects the magnitude bucket for that
        #      intervention;
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
        self.action_feature_dim = 19
        self.last_action_descriptor = ActionDescriptor()
        self._last_action_target = "hold"
        self._last_action_direction = 0
        self._reversal_count = 0
        self.recovery_share_threshold = 0.25
        self.recovery_gap_threshold = -0.05
        self.aggressive_actions = set()
        self.allow_aggressive_actions = True
        
        # Hybrid manipulation actions.  Index 0 is hold/status quo; every
        # other discrete action chooses exactly one coefficient and direction.
        # PPO's continuous magnitude head supplies the step multiplier.
        all_keys = ["base_fare", "per_minute", "per_mile", "booking_fee", "airport_fee"]
        active_keys = [k for k in all_keys if k in self.opt_keys]
        

        self.action_to_steps: Dict[int, Dict[str, int]] = {}
        self.action_to_steps[0] = {key: 0 for key in active_keys}
        action_idx = 1
        for key in active_keys:
            for step in (-1, 1):
                self.action_to_steps[action_idx] = {
                    k: (int(step) if k == key else 0) for k in active_keys
                }
                action_idx += 1
        action_dim = len(self.action_to_steps)
        self.action_keys = list(active_keys)
        
        # State includes cyclical/flag time context, richer demand/WTP context,
        # recent EMA/delta features, direct supply state, own and opponent fare-
        # coefficient deltas, action-memory/oscillation stress features, and
        # constrained-MDP context. Opponent deltas help PPO respond to heuristic
        # rivals whose coefficients move before share/gap metrics fully react.
        # Frame stacking appends recent encoded states so PPO can infer hidden
        # demand/supply feedback without requiring a recurrent policy.
        self.single_state_dim = 89
        state_dim = self.single_state_dim * self.state_frame_stack

        # Initialize PPO agent.
        self.agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=192,
            clip_eps=0.60,
            final_clip_eps=0.35,
            max_grad_norm=0.8,
            ent_coeff=0.024,
            min_ent_coeff=0.0005,
            ent_decay=0.992,
            target_kl=0.500,
            max_lr=1.2e-3,
            value_clip_eps=0.30,
            initial_exploration_rate=0.58,
            final_exploration_rate=0.02,
            action_feature_dim=self.action_feature_dim,
            response_dim=12,
            action_q_coeff=0.10,
            exploration_fraction=0.90,
            exploration_warmup_fraction=0.35,
            min_action_visits=1,
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
        
    def update_response_context(self, share: float, gap: float, fulfillment: float) -> None:
        """Cache latest market response signals for state/action features."""
        self._last_share_hint = float(np.clip(share, 0.0, 1.0))
        self._last_gap_hint = float(np.nan_to_num(gap, nan=0.0, posinf=3.0, neginf=-3.0))
        self._last_fulfillment_hint = float(np.clip(fulfillment, 0.0, 1.0))

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
    
    def action_descriptor_vector(self) -> np.ndarray:
        """Compact vector describing the last executed option for PPO auxiliary credit."""
        d = getattr(self, "last_action_descriptor", ActionDescriptor())
        target_idx = -1 if d.target == "hold" else self.action_keys.index(d.target) if d.target in self.action_keys else -1
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
            bounded = self._bounded_relative_move(current, anchor, self.max_relative_dev, lb, ub)
            setattr(self.overrides, key, bounded)
            width = max(1e-6, float(ub - lb))
            pre = float(pre_values.get(key, anchor))
            realized = float(bounded - pre)
            intended = float(np.sign(fare_step_map[key]) * scaled_steps[key])
            direction = int(np.sign(fare_step_map[key]))
            is_reversal = bool(
                str(key) == str(getattr(self, "_last_action_target", "hold"))
                and direction != 0
                and direction == -int(getattr(self, "_last_action_direction", 0) or 0)
            )
            self._reversal_count = int(getattr(self, "_reversal_count", 0) + 1) if is_reversal else 0
            self._last_action_target = str(key)
            self._last_action_direction = int(direction)
            self.last_action_descriptor = ActionDescriptor(
                action_id=int(action),
                target=str(key),
                direction=direction,
                intended_step=float(intended),
                realized_delta=realized,
                realized_delta_norm=float(realized / width),
                pre_value=pre,
                post_value=float(bounded),
                lower_distance=float(np.clip((bounded - lb) / width, 0.0, 1.0)),
                upper_distance=float(np.clip((ub - bounded) / width, 0.0, 1.0)),
                magnitude_multiplier=float(multipliers[key]),
                magnitude_level=float(multipliers[key]),
                repeat_count=int(self._repeat_action_count),
                was_clipped=bool(abs(realized - intended) > 1e-8),
                is_reversal=is_reversal,
                reversal_count=int(self._reversal_count),
            )
