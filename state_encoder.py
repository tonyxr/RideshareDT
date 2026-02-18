from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  8 19:47:22 2026

@author: Xiaoru Shi
"""

"""
State encoder:
fixed context + competitor memory + last-batch summaries + selected coefficient deltas.
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
    w_keys = ["clear", "rain", "snow"]
    w_onehot = [1.0 if weather == k else 0.0 for k in w_keys]
    if sum(w_onehot) == 0:
        w_onehot = [0.0, 0.0, 0.0]

    # Fixed features (stable, Markov-ish)
    fixed = [
        day_of_week / 6.0,
        hour / 23.0,
        *w_onehot,
        float(np.clip(airport_rate_last, 0.0, 1.0)),
        float(np.clip(mean_distance_last / 20.0, 0.0, 1.0)),  # normalize ~ [0,20] miles
        float(firm2_ema_share),
        float(np.clip(firm2_ema_gap / 5.0, -2.0, 2.0)),       # scale price gap
        float(np.clip(firm2_cooldown / 5.0, 0.0, 1.0)),
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