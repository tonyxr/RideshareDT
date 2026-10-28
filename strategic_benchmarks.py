"""Theory-facing summaries for simulated two-firm tariff payoff matrices."""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np


def _matrix(values: Sequence[Sequence[float]], size: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (size, size) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite {size}x{size} matrix")
    return array


def solve_tariff_game(
    grid: Sequence[float],
    firm1_profit: Sequence[Sequence[float]],
    firm2_profit: Sequence[Sequence[float]],
) -> Dict[str, Any]:
    """Compute pure Nash and Firm-1 Stackelberg points on one payoff grid.

    Rows are Firm 1 tariffs and columns are Firm 2 tariffs. Ties are resolved
    toward the lower tariff, making the result deterministic without claiming
    that a mixed-strategy equilibrium was estimated.
    """
    prices = np.asarray(tuple(float(value) for value in grid), dtype=float)
    if prices.ndim != 1 or prices.size < 2 or not np.all(np.isfinite(prices)):
        raise ValueError("grid must contain at least two finite tariffs")
    n = int(prices.size)
    p1 = _matrix(firm1_profit, n, "firm1_profit")
    p2 = _matrix(firm2_profit, n, "firm2_profit")
    firm1_best = np.argmax(p1, axis=0)
    firm2_best = np.argmax(p2, axis=1)
    pure = [
        (row, col)
        for row in range(n)
        for col in range(n)
        if int(firm1_best[col]) == row and int(firm2_best[row]) == col
    ]
    pure_rows = [
        {
            "firm1_multiplier": float(prices[row]),
            "firm2_multiplier": float(prices[col]),
            "firm1_profit_per_request": float(p1[row, col]),
            "firm2_profit_per_request": float(p2[row, col]),
            "firm1_profit_lead": float(p1[row, col] - p2[row, col]),
        }
        for row, col in pure
    ]
    if pure:
        nash_row, nash_col = max(
            pure,
            key=lambda item: (
                p1[item] + p2[item],
                -prices[item[0]],
                -prices[item[1]],
            ),
        )
        nash = next(
            row for row in pure_rows
            if row["firm1_multiplier"] == float(prices[nash_row])
            and row["firm2_multiplier"] == float(prices[nash_col])
        )
    else:
        nash = None

    # Firm 1 commits first; Firm 2 is the follower and best-responds by row.
    follower_columns = np.asarray(firm2_best, dtype=int)
    leader_payoffs = p1[np.arange(n), follower_columns]
    leader_row = int(np.argmax(leader_payoffs))
    follower_col = int(follower_columns[leader_row])
    stackelberg = {
        "firm1_multiplier": float(prices[leader_row]),
        "firm2_multiplier": float(prices[follower_col]),
        "firm1_profit_per_request": float(p1[leader_row, follower_col]),
        "firm2_profit_per_request": float(p2[leader_row, follower_col]),
        "firm1_profit_lead": float(
            p1[leader_row, follower_col] - p2[leader_row, follower_col]
        ),
    }
    return {
        "grid": prices.tolist(),
        "pure_nash": pure_rows,
        "selected_static_nash": nash,
        "stackelberg_firm1_leader": stackelberg,
        "firm2_best_response_multiplier_by_firm1": {
            f"{prices[row]:g}": float(prices[follower_columns[row]])
            for row in range(n)
        },
        "second_mover_advantage_at_stackelberg": bool(
            stackelberg["firm2_profit_per_request"]
            > stackelberg["firm1_profit_per_request"]
        ),
        "interpretation": (
            "Own profit and profit lead are separate. A negative leader lead "
            "can be the theoretical second-mover advantage, not a model error."
        ),
    }


def cycle_diagnostics(values: Sequence[float]) -> Dict[str, float | bool]:
    """Report price cycling without treating it as a convergence failure."""
    series = np.asarray(tuple(float(value) for value in values), dtype=float)
    if series.size < 3:
        return {"observations": int(series.size), "cycle_detected": False}
    changes = np.diff(series)
    nonzero = changes[np.abs(changes) > 1e-12]
    reversals = int(np.sum(nonzero[1:] * nonzero[:-1] < 0.0)) if nonzero.size > 1 else 0
    return {
        "observations": int(series.size),
        "switch_rate": float(np.mean(np.abs(changes) > 1e-12)),
        "direction_reversal_rate": float(reversals / max(1, nonzero.size - 1)),
        "amplitude": float(np.max(series) - np.min(series)),
        "cycle_detected": bool(reversals >= 2 and np.ptp(series) > 1e-6),
        "interpretation": (
            "With capacity constraints, alternating-move Edgeworth cycles are "
            "an economic result to report rather than automatically a bug."
        ),
    }
