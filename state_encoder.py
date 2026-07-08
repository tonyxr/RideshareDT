from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  8 19:47:22 2026

@author: Xiaoru Shi
"""

"""
State encoder:
compact context + normalized demand/supply health + selected coefficient deltas.

The state intentionally exposes target-centered signals instead of raw, redundant
metrics.  Driver supply remains an external layer: the policy can observe service
health and supply stress, but it still only controls rider-facing price actions.
"""

from typing import List, Optional
import mkl_config  # noqa: F401 - set oneMKL env before NumPy/Torch
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
    firm1_last_profitpr: float = 0.0,
    firm1_last_fulfillment: float = 1.0,
    firm1_last_acceptance: float = 1.0,
    firm1_last_wait: float = 0.0,
    firm1_last_driver_paypr: float = 0.0,
    driver_state_vec: Optional[np.ndarray] = None,
    supply_incentive_multiplier: float = 1.0,
    last_action_magnitude: float = 0.0,
    repeat_action_count: float = 0.0,
    max_relative_dev: float = 0.35,
    constraint_multipliers: Optional[np.ndarray] = None,
    constraint_curriculum_scale: float = 1.0,
) -> np.ndarray:
    
    ride_ctx = np.asarray(ride_ctx_vec, dtype=np.float32).reshape(-1)
    if ride_ctx.size < 3:
        ride_ctx = np.pad(ride_ctx, (0, 3 - ride_ctx.size), mode="constant")
    ride_ctx = ride_ctx[:3]
    supply = np.asarray(
        driver_state_vec if driver_state_vec is not None else np.zeros(8),
        dtype=np.float32,
    ).reshape(-1)
    if supply.size < 8:
        supply = np.pad(supply, (0, 8 - supply.size), mode="constant")
    supply = np.nan_to_num(supply[:8], nan=0.0, posinf=1.0, neginf=0.0)
    supply = np.clip(supply, 0.0, 1.0)

    # Coefficient features: normalized relative deltas vs base
    base_empty = CoefficientOverrides()
    coef_feats = []
    for k in opt_keys[:5]:
        cur = get_coeff(base, ov_firm1, k)
        base_val = get_coeff(base, base_empty, k)
        coef_feats.append((cur - base_val) / (abs(base_val) + 1e-6))

    while len(coef_feats) < 5:
        coef_feats.append(0.0)
    coef_arr = np.asarray(coef_feats[:5], dtype=np.float32)
        
    share = float(np.clip(firm1_last_share, 0.0, 1.0))
    fulfillment = float(np.clip(firm1_last_fulfillment, 0.0, 1.0))
    acceptance = float(np.clip(firm1_last_acceptance, 0.0, 1.0))
    # The incoming share is completed rides / all requests, so it already
    # includes fulfillment and must not be multiplied by fulfillment again.
    served_share = share
    gap = float(firm1_last_gap)
    profitpr = float(firm1_last_profitpr)

    fixed = [
        float(np.clip(ride_ctx[0], 0.0, 1.0)),
        float(np.clip(ride_ctx[1], 0.0, 1.0)),
        float(np.clip(ride_ctx[2], 0.0, 1.0)),
        share,
        float(np.clip(firm2_ema_share, 0.0, 1.0)),
        float(np.clip(served_share / 0.50, 0.0, 1.0)),
        float(np.clip((gap + 3.0) / 6.0, 0.0, 1.0)),
        float(np.clip((firm2_ema_gap + 3.0) / 6.0, 0.0, 1.0)),
        float(np.clip((profitpr + 6.0) / 18.0, 0.0, 1.0)),
        float(np.clip(firm1_last_revpr / 30.0, 0.0, 1.0)),
        float(np.clip((firm1_last_reward + 1.0) / 2.0, 0.0, 1.0)),
        float(np.clip(airport_rate_last, 0.0, 1.0)),
        float(np.clip(mean_distance_last / 12.0, 0.0, 1.0)),
        float(np.clip(firm2_cooldown / 5.0, 0.0, 1.0)),
        fulfillment,
        acceptance,
        float(np.clip(firm1_last_wait / 20.0, 0.0, 1.0)),
        float(np.clip(firm1_last_driver_paypr / 20.0, 0.0, 1.0)),
        float(np.clip((0.30 - served_share) / 0.30, 0.0, 1.0)),
        float(np.clip((0.60 - acceptance) / 0.60, 0.0, 1.0)),
    ]
    # Belief/POMDP proxy features: expose current policy posture and subsystem
    # stress so a feed-forward PPO policy can infer hidden supply/demand
    # feedback from direct outputs.  Frame stacking adds temporal derivatives;
    # these features make the current frame itself Markov-closer by encoding
    # action memory, price distance-to-anchor, and collapse risk.
    avg_abs_coef_dev = float(
        np.clip(np.mean(np.abs(coef_arr)) / max(float(max_relative_dev), 1e-6), 0.0, 1.0)
    )
    max_abs_coef_dev = float(
        np.clip(np.max(np.abs(coef_arr)) / max(float(max_relative_dev), 1e-6), 0.0, 1.0)
    )
    overpricing_stress = float(np.clip((0.15 - gap) / 3.0, 0.0, 1.0))
    overdiscount_stress = float(np.clip((gap - 1.50) / 3.0, 0.0, 1.0))
    service_stress = float(np.clip(max(0.75 - fulfillment, 0.0) / 0.75, 0.0, 1.0))
    wait_stress = float(np.clip((firm1_last_wait - 5.0) / 15.0, 0.0, 1.0))
    profit_stress = float(np.clip((-profitpr) / 6.0, 0.0, 1.0))
    collapse_stress = float(np.clip((0.12 - served_share) / 0.12, 0.0, 1.0))
    action_memory = [
        float(np.clip((supply_incentive_multiplier - 0.90) / 0.30, 0.0, 1.0)),
        float(np.clip(last_action_magnitude, 0.0, 1.0)),
        float(np.clip(repeat_action_count / 10.0, 0.0, 1.0)),
        avg_abs_coef_dev,
        max_abs_coef_dev,
        overpricing_stress,
        overdiscount_stress,
        service_stress,
        wait_stress,
        max(profit_stress, collapse_stress),
    ]
    constraints = np.asarray(
        constraint_multipliers if constraint_multipliers is not None else np.zeros(5),
        dtype=np.float32,
    ).reshape(-1)
    if constraints.size < 5:
        constraints = np.pad(constraints, (0, 5 - constraints.size), mode="constant")
    constraints = np.clip(constraints[:5], 0.0, 1.0)
    constrained_context = [
        float(np.clip(constraint_curriculum_scale, 0.0, 1.0)),
        float(np.clip(np.mean(constraints), 0.0, 1.0)),
    ]
    return np.array(
        fixed + supply.tolist() + coef_feats + action_memory + constraints.tolist() + constrained_context,
        dtype=np.float32,
    )
