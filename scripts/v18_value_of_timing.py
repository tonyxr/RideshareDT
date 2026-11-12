#!/usr/bin/env python3
"""Compute the dated-minus-fixed timing gain and the agent's share."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np


def timing(agent: Sequence[float], fixed: Sequence[float], dated: Sequence[float], *, draws: int = 10_000, seed: int = 18) -> dict:
    agent, fixed, dated = map(lambda x: np.asarray(x, dtype=float), (agent, fixed, dated))
    if not (agent.shape == fixed.shape == dated.shape) or agent.size == 0:
        raise ValueError("timing inputs must be aligned and non-empty")
    gain = dated - fixed
    numerator = agent - fixed
    rng = np.random.default_rng(seed)
    gain_draws, ratio_draws = [], []
    for _ in range(draws):
        idx = rng.integers(0, gain.size, gain.size)
        g = float(np.mean(gain[idx])); n = float(np.mean(numerator[idx]))
        gain_draws.append(g)
        if abs(g) > 1e-12:
            ratio_draws.append(n / g)
    ci = np.quantile(gain_draws, [0.025, 0.975]).tolist()
    excludes_zero = ci[0] > 0 or ci[1] < 0
    return {"gain": float(np.mean(gain)), "gain_ci95": ci,
            "agent_share": float(np.mean(numerator) / np.mean(gain)) if excludes_zero else None,
            "agent_share_ci95": np.quantile(ratio_draws, [0.025, 0.975]).tolist() if excludes_zero else None,
            "reported": bool(excludes_zero)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON with aligned agent, best_fixed and dated arrays")
    parser.add_argument("--output", type=Path, default=Path("runs/v18/results/value_of_timing.json"))
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    result = timing(data["agent"], data["best_fixed"], data["dated_two_lever"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
