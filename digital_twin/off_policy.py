"""Off-policy estimators for logged v30 pricing decisions."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Optional

import numpy as np


@dataclass(frozen=True)
class LoggedDecision:
    reward: float
    behavior_probability: float
    target_probability: float
    q_logged_action: Optional[float] = None
    v_target_policy: Optional[float] = None
    cluster_id: str = ""

    def validated(self) -> "LoggedDecision":
        numeric = (
            float(self.reward), float(self.behavior_probability),
            float(self.target_probability),
        )
        if any(not math.isfinite(value) for value in numeric):
            raise ValueError("logged reward/propensities must be finite")
        if not 0 < numeric[1] <= 1 or not 0 <= numeric[2] <= 1:
            raise ValueError("behavior probability must be positive; target probability must be in [0,1]")
        if (self.q_logged_action is None) != (self.v_target_policy is None):
            raise ValueError("doubly robust evaluation requires both Q and V estimates")
        if self.q_logged_action is not None and any(
            not math.isfinite(float(value))
            for value in (self.q_logged_action, self.v_target_policy)
        ):
            raise ValueError("Q/V estimates must be finite")
        return self


@dataclass(frozen=True)
class OffPolicyEstimate:
    count: int
    ips: float
    snips: float
    doubly_robust: Optional[float]
    effective_sample_size: float
    maximum_importance_weight: float
    weight_coefficient_of_variation: float
    support_warning: bool
    cluster_count: int


class OffPolicyEvaluator:
    """Unclipped IPS, self-normalized IPS and doubly robust evaluation."""

    def __init__(self, *, minimum_effective_fraction: float = 0.10):
        if not 0 < minimum_effective_fraction <= 1:
            raise ValueError("minimum effective fraction must be in (0,1]")
        self.minimum_effective_fraction = float(minimum_effective_fraction)

    def evaluate(self, decisions: Iterable[LoggedDecision]) -> OffPolicyEstimate:
        rows = [row.validated() for row in decisions]
        if not rows:
            raise ValueError("off-policy evaluation requires logged decisions")
        rewards = np.asarray([float(row.reward) for row in rows])
        weights = np.asarray([
            float(row.target_probability) / float(row.behavior_probability)
            for row in rows
        ])
        if not np.all(np.isfinite(weights)):
            raise ValueError("importance weights are non-finite")
        ips = float(np.mean(weights * rewards))
        total_weight = float(weights.sum())
        snips = float(np.sum(weights * rewards) / total_weight) if total_weight > 0 else math.nan
        denominator = float(np.sum(weights * weights))
        ess = total_weight * total_weight / denominator if denominator > 0 else 0.0
        dr: Optional[float]
        if all(row.q_logged_action is not None for row in rows):
            q = np.asarray([float(row.q_logged_action) for row in rows])
            v = np.asarray([float(row.v_target_policy) for row in rows])
            dr = float(np.mean(v + weights * (rewards - q)))
        elif any(row.q_logged_action is not None for row in rows):
            raise ValueError("Q/V estimates must be available for every row or none")
        else:
            dr = None
        mean_weight = float(weights.mean())
        coefficient = (
            float(weights.std(ddof=1) / mean_weight)
            if len(weights) > 1 and mean_weight > 0 else 0.0
        )
        clusters = {row.cluster_id for row in rows if row.cluster_id}
        return OffPolicyEstimate(
            count=len(rows), ips=ips, snips=snips, doubly_robust=dr,
            effective_sample_size=ess,
            maximum_importance_weight=float(weights.max(initial=0.0)),
            weight_coefficient_of_variation=coefficient,
            support_warning=(ess < self.minimum_effective_fraction * len(rows)),
            cluster_count=len(clusters),
        )

    def cluster_bootstrap(
        self,
        decisions: Iterable[LoggedDecision],
        *,
        replicates: int,
        seed: int,
        statistic: str = "doubly_robust",
    ) -> tuple[float, float]:
        rows = [row.validated() for row in decisions]
        groups: dict[str, list[LoggedDecision]] = {}
        for index, row in enumerate(rows):
            groups.setdefault(row.cluster_id or f"row:{index}", []).append(row)
        if len(groups) < 2 or replicates < 100:
            raise ValueError("cluster bootstrap requires >=2 clusters and >=100 replicates")
        keys = tuple(sorted(groups))
        rng = np.random.default_rng(int(seed))
        values = []
        for _ in range(int(replicates)):
            selected = rng.choice(keys, size=len(keys), replace=True)
            sample = [row for key in selected for row in groups[str(key)]]
            estimate = self.evaluate(sample)
            value = getattr(estimate, statistic)
            if value is None or not math.isfinite(float(value)):
                raise ValueError(f"statistic {statistic!r} is unavailable")
            values.append(float(value))
        return tuple(float(value) for value in np.quantile(values, [0.025, 0.975]))
