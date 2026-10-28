from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  8 16:37:07 2026

@author: Xiaoru Shi
"""

"""
Helper file

Experiment configuration: choose which coefficients to optimize,
and define step sizes + bounds per key.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

CoeffKey = str


@dataclass(frozen=True)
class OptimConfig:
    opt_keys: List[CoeffKey]
    step: Dict[CoeffKey, float]
    bounds: Dict[CoeffKey, Tuple[float, float]]
    cost_weight: Dict[CoeffKey, float]


def default_specs_for(keys: List[CoeffKey]) -> OptimConfig:
    step: Dict[CoeffKey, float] = {}
    bounds: Dict[CoeffKey, Tuple[float, float]] = {}
    cost_weight: Dict[CoeffKey, float] = {}

    for k in keys:
        # scalars
        if k == "base_fare":
            step[k] = 0.10
            bounds[k] = (1.50, 6.00)
        elif k == "per_minute":
            step[k] = 0.01
            bounds[k] = (0.10, 1.00)
        elif k == "per_mile":
            step[k] = 0.05
            bounds[k] = (0.50, 4.00)
        elif k == "booking_fee":
            step[k] = 0.10
            bounds[k] = (0.0, 6.0)
        elif k == "airport_fee":
            step[k] = 0.25
            bounds[k] = (0.0, 15.0)

        # dict-entry multipliers
        elif k.startswith("weather_multiplier."):
            step[k] = 0.02
            bounds[k] = (0.80, 2.00)
        elif k.startswith("service_multiplier."):
            step[k] = 0.05
            bounds[k] = (0.80, 3.00)
        elif k.startswith("day_multiplier."):
            step[k] = 0.02
            bounds[k] = (0.80, 1.50)

        else:
            step[k] = 0.01
            bounds[k] = (0.0, 10.0)

        cost_weight[k] = 1.0

    return OptimConfig(opt_keys=keys, step=step, bounds=bounds, cost_weight=cost_weight)


