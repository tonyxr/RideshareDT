"""Single source of truth for the CRISP-Ride V17 final objective."""

from __future__ import annotations

from typing import Any, Iterable, Mapping


COMPETITOR_PROFIT_WEIGHT = 0.75
OBJECTIVE_NAME = "own_profit_minus_0.75_competitor_profit"


def objective(
    own_profit: float,
    competitor_profit: float,
    *,
    w: float = COMPETITOR_PROFIT_WEIGHT,
) -> float:
    """Return the registered V17 policy-ranking objective."""
    return float(own_profit) - float(w) * float(competitor_profit)


def final_objective(
    own_profit: float,
    competitor_profit: float,
    *,
    competitor_weight: float = COMPETITOR_PROFIT_WEIGHT,
) -> float:
    """Descriptive alias retained for existing report code."""
    return objective(own_profit, competitor_profit, w=competitor_weight)


def profit_report(own_profit: float, competitor_profit: float) -> dict[str, float]:
    """Report the objective, both auditable components, and their lead."""
    own = float(own_profit)
    competitor = float(competitor_profit)
    return {
        "own_profit": own,
        "competitor_profit": competitor,
        "final_objective": objective(own, competitor),
        "profit_lead": own - competitor,
    }


def rank_profit_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    own_key: str = "own_profit",
    competitor_key: str = "competitor_profit",
) -> list[dict[str, Any]]:
    """Rank rows high-to-low while retaining both component profits."""
    enriched = []
    for row in rows:
        value = dict(row)
        value.update(profit_report(value[own_key], value[competitor_key]))
        enriched.append(value)
    return sorted(
        enriched,
        key=lambda row: float(row["final_objective"]),
        reverse=True,
    )
