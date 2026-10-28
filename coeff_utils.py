from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  8 16:57:10 2026

@author: Xiaoru Shi

Generic coefficient get/set/update for scalar + dict-entry coefficients.

Helper file

CoeffKey examples:
- "base_fare"
- "airport_fee"
- "weather_multiplier.rain"
- "day_multiplier.5"
"""

from typing import Any, Tuple, Optional
from copy import deepcopy

from Market_models import MarketCoefficients, CoefficientOverrides

CoeffKey = str


def _parse_key(key: CoeffKey) -> Tuple[str, Optional[Any]]:
    if "." not in key:
        return key, None

    attr, sub = key.split(".", 1)
    if attr == "day_multiplier":
        sub = int(sub)
    return attr, sub


def get_coeff(base: MarketCoefficients, ov: CoefficientOverrides, key: CoeffKey) -> float:
    attr, sub = _parse_key(key)

    if sub is None:
        v = getattr(ov, attr)
        return float(getattr(base, attr) if v is None else v)

    ov_map = getattr(ov, attr)
    base_map = getattr(base, attr)
    if ov_map is None:
        return float(base_map.get(sub, 1.0))
    return float(ov_map.get(sub, base_map.get(sub, 1.0)))


def set_coeff(base: MarketCoefficients, ov: CoefficientOverrides, key: CoeffKey, value: float) -> None:
    attr, sub = _parse_key(key)

    if sub is None:
        setattr(ov, attr, float(value))
        return

    ov_map = getattr(ov, attr)
    if ov_map is None:
        ov_map = deepcopy(getattr(base, attr))
        setattr(ov, attr, ov_map)
    ov_map[sub] = float(value)