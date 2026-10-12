from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 12:57:14 2026

@author: Xiaoru Shi
- Market coefficient definitions (city priors) _ coefficient overrides per firm

re-written with GPT on Feb 8, 2026
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass(frozen=True)
class MarketCoefficients:
    # Core scalars
    base_fare: float
    per_mile: float
    per_minute: float
    booking_fee: float

    # Additive fee
    airport_fee: float

    # Multipliers / schedules
    surge_hours: List[Tuple[int, int, float]]            # (start_hour, end_hour, multiplier)
    day_multiplier: Dict[int, float]                     # 0..6 -> multiplier
    weather_multiplier: Dict[str, float]                 # "clear"/"rain"/"snow" -> multiplier
    service_multiplier: Dict[str, float]                 # "economy"/"premium" -> multiplier

@dataclass
class CoefficientOverrides:
    # scalars
    base_fare: Optional[float] = None
    per_mile: Optional[float] = None
    per_minute: Optional[float] = None
    booking_fee: Optional[float] = None
    airport_fee: Optional[float] = None

    # dict overrides (partial updates supported by copying base dict once)
    day_multiplier: Optional[Dict[int, float]] = None
    weather_multiplier: Optional[Dict[str, float]] = None
    service_multiplier: Optional[Dict[str, float]] = None