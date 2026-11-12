#!/usr/bin/env python3
"""Seed-level paired differences, two-level bootstrap and Holm correction."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


def two_level_bootstrap(
    differences: Mapping[str, Iterable[float]], *, draws: int = 10_000,
    seed: int = 18,
) -> dict[str, Any]:
    vectors = {key: np.asarray(list(values), dtype=float) for key, values in differences.items()}
    if not vectors or any(vector.size == 0 or not np.all(np.isfinite(vector)) for vector in vectors.values()):
        raise ValueError("bootstrap requires finite, non-empty cells for every seed")
    rng = np.random.default_rng(seed)
    seeds = tuple(vectors)
    sample = np.empty(draws, dtype=float)
    for index in range(draws):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        sample[index] = np.mean([
            np.mean(rng.choice(vectors[name], size=vectors[name].size, replace=True))
            for name in sampled_seeds
        ])
    seed_means = {name: float(np.mean(vector)) for name, vector in vectors.items()}
    return {
        "mean": float(np.mean(list(seed_means.values()))),
        "ci95": np.quantile(sample, [0.025, 0.975]).tolist(),
        "seed_differences": seed_means,
        "seed_range": [min(seed_means.values()), max(seed_means.values())],
        "draws": int(draws),
    }


def holm(pvalues: Mapping[str, float], alpha: float = 0.05) -> dict[str, Any]:
    ordered = sorted((float(value), key) for key, value in pvalues.items())
    result: dict[str, Any] = {}
    still_rejecting = True
    total = len(ordered)
    for rank, (pvalue, key) in enumerate(ordered, 1):
        threshold = float(alpha) / (total - rank + 1)
        reject = bool(still_rejecting and pvalue <= threshold)
        still_rejecting &= reject
        result[key] = {"p": pvalue, "threshold": threshold, "reject": reject}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON mapping comparison -> seed -> cell differences")
    parser.add_argument("--output", type=Path, default=Path("runs/v18/results/stats.json"))
    parser.add_argument("--draws", type=int, default=10_000)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    result = {name: two_level_bootstrap(rows, draws=args.draws) for name, rows in data.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
