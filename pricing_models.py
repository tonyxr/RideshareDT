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

from WPOAgent import PPOAgent
from Market_models import CoefficientOverrides
from optim_config import default_specs_for

@dataclass
class FirmMetrics:
    revenue: float = 0.0
    wins: int = 0
    total: int = 0

    @property
    def share(self) -> float:
        return (self.wins / self.total) if self.total > 0 else 0.0

    @property
    def rev_per_request(self) -> float:
        return (self.revenue / self.total) if self.total > 0 else 0.0


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
        self.managed_keys = tuple(k for k in keys if k in {"base_fare", "per_minute"})
        if not self.managed_keys:
            self.managed_keys = ("base_fare", "per_minute")

        self.step_base_fare = 0.10
        self.step_per_minute = 0.01
        self.base_bounds = (1.50, 6.00)
        self.pmin_bounds = (0.10, 1.00)

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

        if self.cooldown > 0:
            self.cooldown -= 1
            return

        peak = (7 <= hour < 10) or (16 <= hour < 19)
        bad_weather = weather in ("rain", "snow")

        # baseline targets (schedule)
        base_target = city_base + (0.20 if peak else 0.0) + (0.10 if bad_weather else 0.0)
        pmin_target = city_pmin + (0.01 if peak else 0.0) + (0.005 if bad_weather else 0.0)

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

        self.cooldown = self.cooldown_H

    def update(self, metrics: FirmMetrics, price_gap_mean: float):
        self.ema_share = self._ema(metrics.share, self.ema_share, self.alpha)
        self.ema_gap = self._ema(price_gap_mean, self.ema_gap, self.alpha)
        
class FirmRLPricer:
    MAX_MANIPULATED_COEFFS = 2

    def _ensure_internal_state(self) -> None:
        """Backstop against partially initialized objects from stale environments."""
        if not hasattr(self, "opt_keys"):
            self.opt_keys = ["base_fare", "per_minute"]
        if len(self.opt_keys) > self.MAX_MANIPULATED_COEFFS:
            self.opt_keys = list(self.opt_keys[: self.MAX_MANIPULATED_COEFFS])
        if not hasattr(self, "config"):
            self.config = default_specs_for(self.opt_keys)
    
    
    def _build_action_cost_matrix(self, action_dim: int) -> np.ndarray:
        """Build a symmetric Wasserstein ground-cost matrix over discrete actions."""
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")

        vectors = [
            self.action_vectors.get(i, np.zeros(2, dtype=float))
            for i in range(action_dim)
        ]
        cost = np.zeros((action_dim, action_dim), dtype=np.float32)
        for i in range(action_dim):
            for j in range(i + 1, action_dim):
                d = float(np.linalg.norm(vectors[i] - vectors[j], ord=2))
                cost[i, j] = d
                cost[j, i] = d
        return cost
    
    def __init__(self, seed: Optional[int], opt_keys: List[str]):
        # Keep action semantics simple and stable: one shared action can only
        # manipulate up to two coefficients per step.
        self.opt_keys = list(opt_keys[: self.MAX_MANIPULATED_COEFFS])
        if len(opt_keys) > self.MAX_MANIPULATED_COEFFS:
            print(
                f"[FirmRLPricer] Received {len(opt_keys)} opt_keys; "
                f"limiting to first {self.MAX_MANIPULATED_COEFFS}: {self.opt_keys}"
            )
        self.config = default_specs_for(self.opt_keys)
        self.overrides = CoefficientOverrides()
        
        self.step_scale = 0.5
        self.max_relative_dev = 0.30
        self.recovery_share_threshold = 0.12
        self.recovery_gap_threshold = -0.35
        
        
        # Action space: no-op + (down/up) per managed coefficient.
        # The 5-action layout is fixed for stability and for Wasserstein costs.
        action_dim = 5
        
        self.action_to_steps: Dict[int, Dict[str, int]] = {
            0: {"base_fare": 0, "per_minute": 0},
            1: {"base_fare": -1, "per_minute": 0},
            2: {"base_fare": +1, "per_minute": 0},
            3: {"base_fare": 0, "per_minute": -1},
            4: {"base_fare": 0, "per_minute": +1},
        }
        self.action_vectors = {
            a: np.array([d["base_fare"], d["per_minute"]], dtype=float)
            for a, d in self.action_to_steps.items()
        }

        # Keep state compact and below 10 features.
        state_dim = 9

        # Wasserstein geometry: penalize large policy moves in coefficient-step space.
        self.cost_matrix = self._build_action_cost_matrix(action_dim)
        
        # Initialize PPO agent.
        self.agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            clip_eps=0.2,
            max_grad_norm=0.8,
            ent_coeff=0.01,
        )
    
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
        
        step_map = {
            k: v for k, v in self.action_to_steps[action].items() if k in self.opt_keys
        }
        if not step_map:
            return
        
        scaled_steps = {k: self.config.step[k] * self.step_scale for k in step_map.keys()}
        bounds = {k: self.config.bounds[k] for k in step_map.keys()}
        market_interaction.apply_step_actions_to_overrides(
            overrides=self.overrides,
            action_steps=step_map,
            step_size=scaled_steps,
            bounds=bounds,
        )

