from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  8 19:47:22 2026

@author: Xiaoru Shi
"""

"""
State encoder:
compact context + last-batch summaries + selected coefficient deltas.

The state intentionally keeps only low-variance signals so the policy is easier to
learn and less sensitive to transient competitor behavior.
"""

from typing import List
import numpy as np

from Market_models import MarketCoefficients, CoefficientOverrides
from coeff_utils import get_coeff

CoeffKey = str


def build_state_vector(
    base: MarketCoefficients,
    ov_firm1: CoefficientOverrides,
    opt_keys: List[CoeffKey],
    day_of_week: int,
    hour: int,
    weather: str,
    airport_rate_last: float,
    mean_distance_last: float,
    firm2_ema_share: float,
    firm2_ema_gap: float,
    firm2_cooldown: float,
) -> np.ndarray:
    # Compact fixed features (stable and low-noise):
    #  - weekend flag instead of full day-of-week
    #  - bad-weather indicator instead of full one-hot weather
    is_weekend = 1.0 if int(day_of_week) >= 5 else 0.0
    is_bad_weather = 1.0 if weather in ("rain", "snow") else 0.0

    # Fixed features (stable, Markov-ish)
    fixed = [
        hour / 23.0,
        is_weekend,
        is_bad_weather,
        float(np.clip(airport_rate_last, 0.0, 1.0)),
        float(np.clip(mean_distance_last / 20.0, 0.0, 1.0)),
        float(np.clip(firm2_ema_share, 0.0, 1.0)),
    ]

    # Coefficient features: normalized relative deltas vs base
    coef_feats = []
    base_empty = CoefficientOverrides()
    for k in opt_keys:
        cur = get_coeff(base, ov_firm1, k)
        base_val = get_coeff(base, base_empty, k)
        denom = abs(base_val) + 1e-6
        coef_feats.append((cur - base_val) / denom)

    return np.array(fixed + coef_feats, dtype=np.float32)