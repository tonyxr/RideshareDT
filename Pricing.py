#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xiaoru Shi
Firm Pricing controllers

Firm1: RL (with WPO/Actor-Critic)
Firm2: Heuristic dynamic pricing (schedule + competitive response + guardrails)

"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np

from Market_models import CoefficientOverrides

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
    
class Firm1RLPricer:
    """
    RL-based coefficient controller.
    For now: simple placeholder policy + clean interface (ready for WPO).
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        
        # Start with no overrides (use city priors)
        self.overrides = CoefficientOverrides
        
        # Step sizes
        self.step_base_fare = 0.10
        self.step_per_minute = 0.01
        
        # Bound for overrides (relative realism; tune)
        self.base_fare_bounds = (1.50, 6.00)
        self.per_minute_bounds = (0.10, 1.00)
        
        self.last_metrics = FirmMetrics()
        
    def act(self, state: Dict[str, Any]) -> None:
        """
        Placeholder action:
        - randomly nudge base_fare or per_minute slightly
        Replace this with Actor-Critic policy action selection.
        """
        if self.overrides.base_fare is None:
            self.overrides.base_fare = state["city_base_fare"]
        if self.overrides.per_minute is None:
            self.overrides.per_minute = state["city_per_minute"]

        # Simple exploration; swap with learned action distribution
        k = self.rng.choice(["base_fare", "per_minute", "noop"], p=[0.45, 0.45, 0.10])
        if k == "noop":
            return

        direction = self.rng.choice([-1, +1])

        if k == "base_fare":
            new_val = float(self.overrides.base_fare) + direction * self.step_base_fare
            lo, hi = self.base_fare_bounds
            self.overrides.base_fare = float(np.clip(new_val, lo, hi))

        if k == "per_minute":
            new_val = float(self.overrides.per_minute) + direction * self.step_per_minute
            lo, hi = self.per_minute_bounds
            self.overrides.per_minute = float(np.clip(new_val, lo, hi))


    def update(self, metrics: FirmMetrics) -> None:
        """
        Hook for RL learning updates (WPO):
        store transitions (s,a,r,s'), update networks periodically.
        """
        self.last_metrics = metrics
        
class Firm2HeuristicPricer:
    """
    Heuristic dynamic pricing baseline:
    - context schedule targets (weather/peak)
    - competitive response via share + price gap
    - stability via EMA + hysteresis + cooldown
    """
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.overrides = CoefficientOverrides()
        
        # Steps / bounds
        self.step_base_fare = 0.10
        self.step_per_minute = 0.01
        self.base_fare_bounds = (1.50, 6.00)
        self.per_minute_bounds = (0.10, 1.00)
        
        # Competitive thresholds
        self.target_share = 0.50
        self.share_band = 0.03
        self.price_gap_threshold = 0.50  # dollars
        
        # EMA smoothing
        self.alpha = 0.2
        self.ema_share = 0.50
        self.ema_revpr = 0.0
        self.ema_price_gap = 0.0
        
        # Cooldown to avoid thrashing
        self.cooldown = 0
        self.cooldown_H = 3
        
        self.last_metrics = FirmMetrics()
        
    def _ema_update(self, x: float, ema: float) -> float:
        return (1 - self.alpha) * ema + self.alpha * x
    
    def act(self, state: Dict[str, Any]) -> None:
        """
        state requires:
        - city_base_fare, city_per_minute
        - hour, weather
        - ema_share, ema_revpr, ema_price_gap (passed/maintained externally or updated in update())
        """
        if self.overrides.base_fare is None:
            self.overrides.base_fare = state["city_base_fare"]
        if self.overrides.per_minute is None:
            self.overrides.per_minute = state["city_per_minute"]
            
        if self.cooldown > 0:
            self.cooldown -= 1
            return 
    
        hour = int(state["hour"])
        weather = str(state["weather"])
        
        # (1) Baseline schedule targets
        peak = (7 <= hour < 10) or (16 <= hour < 19)
        bad_weather = weather in ("rain", "snow")
        
        base_target = state["city_base_fare"] + (0.20 if peak else 0.0) + (0.10 if bad_weather else 0.0)
        permin_target = state["city_per_minute"] + (0.01 if peak else 0.0) + (0.005 if bad_weather else 0.0)
        
        # (2) Competitive response
        losing_share = self.ema_share < (self.target_share - self.share_band)
        overpriced = self.ema_price_gap > self.price_gap_threshold # Firm2 - Firm1
        
        winning_share = self.ema_share > (self.target_share + self.share_band)
        underpriced = self.ema_price_gap < (-self.price_gap_threshold)
        
        # Decide one-step update (single lever)
        chosen = None
        direction = 0
        
        if losing_share and overpriced: 
            # cut price: reduce base first in peak/bad weather, else per_minute
            chosen = "base_fare" if (peak or bad_weather) else "per_minute"
            direction = -1
        elif winning_share and underpriced:
            # capture margin
            chosen = "base_fare" if (peak or bad_weather) else "per_minute"
            
        else:
            if abs(float(self.overrides.base_fare) - base_target) > 0.15:
                chosen = "base_fare"
                direction = +1 if float(self.overrides.base_fare) < base_target else -1
            elif abs(float(self.overrides.per_minute) - permin_target) > 0.02:
                chosen = "per_minute"
                direction = +1 if float(self.overrides.per_minute) < permin_target else -1
        
        if chosen is None or direction == 0:
            return
        
        if chosen == "base_fare":
            new_val = float(self.overrides.base_fare) + direction * self.step_base_fare
            low, high = self.fare_bounds
            self.overrides.base_fare = float(np.clip(new_val, low, high))
            
        if chosen == "per_minute":
            new_val = float(self.overrides.per_minute) + direction * self.step_per_minute
            low, high = self.per_minute_bounds
            self.overrides.per_minute = float(np.clip(new_val, low, high))
            
        self.cooldown = self.cooldown_H
                
    def update(self, metrics: FirmMetrics, price_gap_mean: float) -> None:
        self.last_metrics = metrics
        self.ema_share = self._ema_update(metrics.share, self.ema_share)
        self.ema_revpr = self._ema_update(metrics.rev_per_request, self.ema_revpr)
        self.ema_price_gap = self._ema_update(price_gap_mean, self.ema_price_gap)
        
    
    
    
    