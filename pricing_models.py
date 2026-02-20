from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xiaoru Shi
Firm Pricing controllers

Firm1: RL (with WPO/Actor-Critic)
Firm2: Heuristic dynamic pricing (schedule + competitive response + guardrails)

"""

from dataclasses import dataclass
from typing import Optional, Dict, List
import numpy as np

from WPOAgent import WassersteinWPOAgent
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

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.overrides = CoefficientOverrides()

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
        if self.overrides.base_fare is None:
            self.overrides.base_fare = float(city_base)
        if self.overrides.per_minute is None:
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

        if losing and overpriced:
            chosen = "base" if (peak or bad_weather) else "pmin"
            direction = -1
        elif winning and underpriced:
            chosen = "base" if (peak or bad_weather) else "pmin"
            direction = +1
        else:
            if abs(float(self.overrides.base_fare) - base_target) > 0.15:
                chosen = "base"
                direction = +1 if float(self.overrides.base_fare) < base_target else -1
            elif abs(float(self.overrides.per_minute) - pmin_target) > 0.02:
                chosen = "pmin"
                direction = +1 if float(self.overrides.per_minute) < pmin_target else -1

        if chosen is None or direction == 0:
            return

        if chosen == "base":
            self.overrides.base_fare = float(np.clip(
                float(self.overrides.base_fare) + direction * self.step_base_fare, *self.base_bounds
            ))
        else:
            self.overrides.per_minute = float(np.clip(
                float(self.overrides.per_minute) + direction * self.step_per_minute, *self.pmin_bounds
            ))

        self.cooldown = self.cooldown_H

    def update(self, metrics: FirmMetrics, price_gap_mean: float):
        self.ema_share = self._ema(metrics.share, self.ema_share, self.alpha)
        self.ema_gap = self._ema(price_gap_mean, self.ema_gap, self.alpha)
        
class FirmRLPricer:
    def __init__(self, seed: Optional[int], opt_keys: List[str]):
        self.opt_keys = opt_keys
        self.config = default_specs_for(opt_keys)
        self.overrides = CoefficientOverrides()
        
        self.step_scale = 0.5
        # State: 10 context features + length of optimized coefficients
        state_dim = 10 + len(opt_keys)
        action_dim = 3 # 0: Decrease, 1: No-op, 2: Increase
        
        # Initialize Agent
        self.agent = WassersteinWPOAgent(
            state_dim=state_dim, 
            action_dim=action_dim, 
            cost_matrix=np.eye(action_dim)
        )
        
    def apply_action(self, action: int):
        """Maps the discrete RL action to coefficient updates."""
        for key in self.opt_keys:
            current_val = getattr(self.overrides, key) or 1.0 
            step = self.config.step[key] * self.step_scale
            
            if action == 0: # Decrease
                new_val = current_val - step
            elif action == 2: # Increase
                new_val = current_val + step
            else: # No-op
                new_val = current_val
        
            # Keep within specified bounds
            setattr(self.overrides, key, np.clip(new_val, *self.config.bounds[key]))
