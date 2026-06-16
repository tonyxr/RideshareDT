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

from typing import List, Optional
import numpy as np

from Market_models import MarketCoefficients, CoefficientOverrides
from coeff_utils import get_coeff

CoeffKey = str


def build_state_vector(
    base: MarketCoefficients,
    ov_firm1: CoefficientOverrides,
    opt_keys: List[CoeffKey],
    ride_ctx_vec: np.ndarray,
    airport_rate_last: float,
    mean_distance_last: float,
    firm2_ema_share: float,
    firm2_ema_gap: float,
    firm2_cooldown: float,
    firm1_last_share: float,
    firm1_last_revpr: float,
    firm1_last_gap: float,
    firm1_last_reward: float,
    driver_state_vec: Optional[np.ndarray] = None,
) -> np.ndarray:
    
    ride_ctx = np.asarray(ride_ctx_vec, dtype=np.float32).reshape(-1)
    if ride_ctx.size < 3:
        ride_ctx = np.pad(ride_ctx, (0, 3 - ride_ctx.size), mode="constant")
    ride_ctx = ride_ctx[:3]

    # Coefficient features: normalized relative deltas vs base
    base_empty = CoefficientOverrides()
    coef_feats = []
    for k in opt_keys[:5]:
        cur = get_coeff(base, ov_firm1, k)
        base_val = get_coeff(base, base_empty, k)
        coef_feats.append((cur - base_val) / (abs(base_val) + 1e-6))

    while len(coef_feats) < 5:
        coef_feats.append(0.0)

    fixed = [
        float(np.clip(ride_ctx[0], 0.0, 1.0)),
        float(np.clip(ride_ctx[1], 0.0, 1.0)),
        float(np.clip(ride_ctx[2], 0.0, 1.0)),
        float(np.clip(firm1_last_share, 0.0, 1.0)),
        float(np.clip(firm2_ema_share, 0.0, 1.0)),
        float(np.clip((firm1_last_gap + 4.0) / 8.0, 0.0, 1.0)),
        float(np.clip((firm2_ema_gap + 4.0) / 8.0, 0.0, 1.0)),
        float(np.clip(firm1_last_revpr / 30.0, 0.0, 1.0)),
        float(np.clip((firm1_last_reward + 1.0) / 2.0, 0.0, 1.0)),
        float(np.clip(airport_rate_last, 0.0, 1.0)),
        float(np.clip(mean_distance_last / 12.0, 0.0, 1.0)),
        float(np.clip(firm2_cooldown / 5.0, 0.0, 1.0)),
    ]
    driver_feats = np.asarray(driver_state_vec if driver_state_vec is not None else [], dtype=np.float32).reshape(-1)
    if driver_feats.size < 8:
        driver_feats = np.pad(driver_feats, (0, 8 - driver_feats.size), mode="constant")
    driver_feats = driver_feats[:8]
    return np.array(fixed + list(driver_feats) + coef_feats, dtype=np.float32)