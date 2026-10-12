"""Full five-dimensional fixed-tariff comparator for v30 evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Mapping, Sequence

import numpy as np


TARIFF_KEYS = ("base_fare", "per_minute", "per_mile", "booking_fee", "airport_fee")


@dataclass(frozen=True)
class FixedComparatorConfig:
    population: int = 48
    iterations: int = 8
    elite_fraction: float = 0.20
    initial_log_std: float = 0.50
    smoothing: float = 0.25
    minimum_elites: int = 3


@dataclass(frozen=True)
class FixedComparatorResult:
    multipliers: dict[str, float]
    mean_objective: float
    seed_objectives: tuple[float, ...]
    evaluations: int
    invalid_candidates: int
    coordinate_mean: tuple[float, ...]
    coordinate_std: tuple[float, ...]


class FixedFiveDimensionalComparator:
    """CEM search over all five independent, upper-unbounded multipliers."""

    def __init__(self, config: FixedComparatorConfig | None = None):
        self.config = config or FixedComparatorConfig()
        if self.config.population < 4 or self.config.iterations < 1:
            raise ValueError("comparator search budget is too small")
        if not 0 < self.config.elite_fraction < 1:
            raise ValueError("elite_fraction must be in (0,1)")
        if self.config.minimum_elites < 2:
            raise ValueError("minimum_elites must be at least two")

    @staticmethod
    def action(coordinates: Sequence[float]) -> dict[str, float]:
        vector = np.asarray(coordinates, dtype=float)
        if vector.shape != (len(TARIFF_KEYS),) or not np.all(np.isfinite(vector)):
            raise ValueError("fixed comparator coordinates must be five finite values")
        positive_log = np.maximum(vector, 0.0)
        if np.any(positive_log > math.log(np.finfo(np.float64).max)):
            raise FloatingPointError("fixed comparator multiplier overflow")
        values = np.exp(positive_log)
        return {key: float(value) for key, value in zip(TARIFF_KEYS, values)}

    @staticmethod
    def _score(
        objective: Callable[[Mapping[str, float], int], float],
        action: Mapping[str, float],
        seeds: Sequence[int],
    ) -> tuple[float, tuple[float, ...]]:
        values = tuple(float(objective(action, int(seed))) for seed in seeds)
        if not values or any(not math.isfinite(value) for value in values):
            return -math.inf, values
        return float(np.mean(values)), values

    def optimize(
        self,
        objective: Callable[[Mapping[str, float], int], float],
        *,
        seeds: Sequence[int],
        rng_seed: int,
        initial_multipliers: Mapping[str, float] | None = None,
    ) -> FixedComparatorResult:
        if not seeds or len(set(int(seed) for seed in seeds)) != len(seeds):
            raise ValueError("comparator seeds must be non-empty and unique")
        initial = {key: 1.0 for key in TARIFF_KEYS}
        if initial_multipliers is not None:
            if set(initial_multipliers) != set(TARIFF_KEYS):
                raise ValueError("initial multipliers must cover all five tariff keys")
            initial.update({key: float(value) for key, value in initial_multipliers.items()})
        if any(value < 1.0 or not math.isfinite(value) for value in initial.values()):
            raise ValueError("initial fixed multipliers must be finite and >=1.0")
        mean = np.log(np.asarray([initial[key] for key in TARIFF_KEYS], dtype=float))
        std = np.full(len(TARIFF_KEYS), self.config.initial_log_std, dtype=float)
        rng = np.random.default_rng(int(rng_seed))
        best_action = dict(initial)
        best_score, best_values = self._score(objective, best_action, seeds)
        evaluations = 1
        invalid = int(not math.isfinite(best_score))
        elite_count = max(
            self.config.minimum_elites,
            int(math.ceil(self.config.population * self.config.elite_fraction)),
        )
        elite_count = min(elite_count, self.config.population)
        for _ in range(self.config.iterations):
            coordinates = rng.normal(mean, std, size=(self.config.population, len(TARIFF_KEYS)))
            coordinates[0] = mean
            scored: list[tuple[float, np.ndarray, dict[str, float], tuple[float, ...]]] = []
            for coordinate in coordinates:
                evaluations += 1
                try:
                    action = self.action(coordinate)
                    score, values = self._score(objective, action, seeds)
                except (ValueError, OverflowError, FloatingPointError):
                    invalid += 1
                    continue
                if not math.isfinite(score):
                    invalid += 1
                    continue
                scored.append((score, coordinate, action, values))
                if score > best_score:
                    best_score, best_values, best_action = score, values, action
            if len(scored) < elite_count:
                raise RuntimeError("insufficient finite comparator candidates")
            scored.sort(key=lambda row: row[0], reverse=True)
            elite = np.stack([row[1] for row in scored[:elite_count]])
            elite_mean = elite.mean(axis=0)
            elite_std = elite.std(axis=0, ddof=1)
            smooth = self.config.smoothing
            mean = smooth * mean + (1.0 - smooth) * elite_mean
            std = smooth * std + (1.0 - smooth) * np.maximum(elite_std, 1e-4)
        return FixedComparatorResult(
            multipliers=best_action, mean_objective=best_score,
            seed_objectives=best_values, evaluations=evaluations,
            invalid_candidates=invalid, coordinate_mean=tuple(mean.tolist()),
            coordinate_std=tuple(std.tolist()),
        )
