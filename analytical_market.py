"""Small analytical market used as a v28 known-answer diagnostic.

The simulator is intentionally richer than this module.  These closed-form
results isolate the economic conditions under which a constant tariff is
optimal and the conditions that create value for adaptation:

* exponential demand with no binding capacity has a constant optimal markup;
* changing only demand volume does not change that markup;
* elasticity or capacity changes move the optimum; and
* lagged driver participation can make an incentive paid before a shortage
  profitable.

No production action bound is defined here.  The analytical demand curve tends
to zero as price tends to infinity, which is the v28 well-posedness contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class ExponentialMarketRegime:
    """One-period demand/cost/capacity regime.

    Potential requests follow ``A exp(-eta p)``.  ``capacity`` is the number
    of requests that can be completed during the period.  The served quantity
    is the smaller of demand and capacity.
    """

    demand_scale: float
    elasticity: float
    marginal_cost: float
    capacity: float = math.inf

    def __post_init__(self) -> None:
        if not math.isfinite(self.demand_scale) or self.demand_scale <= 0.0:
            raise ValueError("demand_scale must be finite and positive")
        if not math.isfinite(self.elasticity) or self.elasticity <= 0.0:
            raise ValueError("elasticity must be finite and positive")
        if not math.isfinite(self.marginal_cost) or self.marginal_cost < 0.0:
            raise ValueError("marginal_cost must be finite and non-negative")
        if math.isnan(self.capacity) or self.capacity <= 0.0:
            raise ValueError("capacity must be positive")

    def demand(self, price: float) -> float:
        price = float(price)
        if not math.isfinite(price) or price < 0.0:
            raise ValueError("price must be finite and non-negative")
        return float(self.demand_scale * math.exp(-self.elasticity * price))

    def served(self, price: float) -> float:
        return float(min(self.demand(price), self.capacity))

    def contribution_profit(self, price: float) -> float:
        return float((float(price) - self.marginal_cost) * self.served(price))

    @property
    def unconstrained_optimal_price(self) -> float:
        """Closed-form monopoly price when capacity does not bind."""
        return float(self.marginal_cost + 1.0 / self.elasticity)

    @property
    def capacity_clearing_price(self) -> float:
        """Lowest price whose expected demand does not exceed capacity."""
        if math.isinf(self.capacity) or self.capacity >= self.demand_scale:
            return 0.0
        return float(math.log(self.demand_scale / self.capacity) / self.elasticity)

    @property
    def optimal_price(self) -> float:
        """Global optimum over the non-negative, uncapped price domain."""
        return float(max(0.0, self.unconstrained_optimal_price, self.capacity_clearing_price))


@dataclass(frozen=True)
class LaggedDriverSupply:
    """One-step partial adjustment of available drivers to an incentive."""

    adjustment_rate: float = 0.35
    incentive_elasticity: float = 0.75
    incentive_cost_per_driver: float = 2.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.adjustment_rate <= 1.0:
            raise ValueError("adjustment_rate must lie in [0, 1]")
        if not math.isfinite(self.incentive_elasticity) or self.incentive_elasticity < 0.0:
            raise ValueError("incentive_elasticity must be finite and non-negative")
        if not math.isfinite(self.incentive_cost_per_driver) or self.incentive_cost_per_driver < 0.0:
            raise ValueError("incentive cost must be finite and non-negative")

    def next_capacity(
        self,
        current_capacity: float,
        normal_capacity: float,
        incentive_multiplier: float,
        shock_multiplier: float = 1.0,
    ) -> float:
        values = (
            current_capacity,
            normal_capacity,
            incentive_multiplier,
            shock_multiplier,
        )
        if any(not math.isfinite(float(value)) or float(value) <= 0.0 for value in values):
            raise ValueError("capacities, incentive, and shock multiplier must be finite and positive")
        target = float(normal_capacity) * float(shock_multiplier) * (
            float(incentive_multiplier) ** self.incentive_elasticity
        )
        return float(
            (1.0 - self.adjustment_rate) * float(current_capacity)
            + self.adjustment_rate * target
        )

    def incentive_cost(self, normal_capacity: float, incentive_multiplier: float) -> float:
        if incentive_multiplier <= 0.0 or not math.isfinite(incentive_multiplier):
            raise ValueError("incentive multiplier must be finite and positive")
        incremental = max(0.0, float(incentive_multiplier) - 1.0)
        return float(self.incentive_cost_per_driver * float(normal_capacity) * incremental)


def anticipatory_shortage_value(
    *,
    current_capacity: float,
    normal_capacity: float,
    shortage_capacity_multiplier: float,
    next_demand_scale: float,
    next_elasticity: float,
    marginal_cost: float,
    incentive_multiplier: float,
    supply: LaggedDriverSupply,
) -> Dict[str, float]:
    """Value of choosing a driver incentive one period before a shortage."""
    capacity = supply.next_capacity(
        current_capacity=current_capacity,
        normal_capacity=normal_capacity,
        incentive_multiplier=incentive_multiplier,
        shock_multiplier=shortage_capacity_multiplier,
    )
    regime = ExponentialMarketRegime(
        demand_scale=next_demand_scale,
        elasticity=next_elasticity,
        marginal_cost=marginal_cost,
        capacity=capacity,
    )
    gross = regime.contribution_profit(regime.optimal_price)
    incentive_cost = supply.incentive_cost(normal_capacity, incentive_multiplier)
    return {
        "incentive_multiplier": float(incentive_multiplier),
        "next_capacity": float(capacity),
        "optimal_price": float(regime.optimal_price),
        "gross_contribution_profit": float(gross),
        "incentive_cost": float(incentive_cost),
        "net_value": float(gross - incentive_cost),
    }


def best_anticipatory_incentive(
    incentive_candidates: Sequence[float] | Iterable[float],
    **kwargs: float | LaggedDriverSupply,
) -> Dict[str, float]:
    """Evaluate a diagnostic candidate set and return the best net value.

    The candidate set is a diagnostic numerical search, not an action-space
    cap.  The production learner retains its unbounded positive action domain.
    """
    rows = [
        anticipatory_shortage_value(
            incentive_multiplier=float(candidate),
            **kwargs,  # type: ignore[arg-type]
        )
        for candidate in incentive_candidates
    ]
    if not rows:
        raise ValueError("at least one incentive candidate is required")
    return max(rows, key=lambda row: row["net_value"])


def price_tail(regime: ExponentialMarketRegime, prices: Sequence[float]) -> np.ndarray:
    """Return demand and profit over an explicit uncapped diagnostic sequence."""
    return np.asarray(
        [
            (float(price), regime.demand(float(price)), regime.contribution_profit(float(price)))
            for price in prices
        ],
        dtype=np.float64,
    )


__all__ = [
    "ExponentialMarketRegime",
    "LaggedDriverSupply",
    "anticipatory_shortage_value",
    "best_anticipatory_incentive",
    "price_tail",
]
