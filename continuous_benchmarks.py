"""CRN-safe continuous-control benchmarks for the v28 learner ladder.

All optimizers in this module receive an explicit tuple of episode seeds.  A
candidate is always evaluated on *the same* seeds as every other candidate,
which makes static, schedule, CEM/MPC, and learned-policy differences paired
comparisons rather than independent Monte Carlo estimates.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Set, Tuple

import numpy as np

from recurrent_sac import ActionTransform, RecurrentSACAgent


class DailyEnvironment(Protocol):
    observation_dim: int

    def reset(self, seed: int) -> np.ndarray: ...

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Mapping[str, Any]]: ...


Policy = Callable[[np.ndarray, int], np.ndarray]
EnvironmentFactory = Callable[[], DailyEnvironment]
# Large enough to rank below any plausible market return, but small enough
# that population means/history aggregation cannot overflow to infinity.
NUMERICAL_CANDIDATE_PENALTY = -1.0e12


def _safe_policy_mean(
    environment_factory: EnvironmentFactory,
    policy_factory: Callable[[], Policy],
    seeds: Sequence[int],
) -> float:
    """Score a CEM candidate, penalizing numerical overflow without clipping.

    Only floating-point overflow/non-finiteness is converted to a finite CEM
    score.  Contract and programming errors continue to fail loudly.
    """
    try:
        return float(
            evaluate_policy(environment_factory, policy_factory, seeds)["mean"]
        )
    except (FloatingPointError, OverflowError):
        return float(NUMERICAL_CANDIDATE_PENALTY)


@dataclass(frozen=True)
class PairedEstimate:
    mean: float
    standard_error: float
    median: float
    values: Tuple[float, ...]

    @classmethod
    def from_values(cls, values: Sequence[float]) -> "PairedEstimate":
        array = np.asarray(values, dtype=np.float64)
        if array.size == 0:
            raise ValueError("cannot summarize an empty sample")
        standard_error = (
            float(array.std(ddof=1) / np.sqrt(array.size)) if array.size > 1 else float("nan")
        )
        return cls(
            mean=float(array.mean()),
            standard_error=standard_error,
            median=float(np.median(array)),
            values=tuple(float(value) for value in array),
        )


def rollout_policy(
    environment: DailyEnvironment,
    policy: Policy,
    *,
    seed: int,
) -> Dict[str, Any]:
    observation = environment.reset(int(seed))
    total_reward = 0.0
    daily_rewards = []
    actions = []
    infos = []
    day = 0
    done = False
    while not done:
        action = np.asarray(policy(observation.copy(), day), dtype=np.float64)
        observation, reward, done, info = environment.step(action)
        total_reward += float(reward)
        daily_rewards.append(float(reward))
        actions.append(action.tolist())
        infos.append(dict(info))
        day += 1
    return {
        "seed": int(seed),
        "return": float(total_reward),
        "days": int(day),
        "daily_rewards": daily_rewards,
        "actions": actions,
        "infos": infos,
    }


def evaluate_policy(
    environment_factory: EnvironmentFactory,
    policy_factory: Callable[[], Policy],
    seeds: Sequence[int],
) -> Dict[str, Any]:
    episodes = [
        rollout_policy(environment_factory(), policy_factory(), seed=int(seed))
        for seed in seeds
    ]
    estimate = PairedEstimate.from_values([episode["return"] for episode in episodes])
    return {
        "mean": estimate.mean,
        "standard_error": estimate.standard_error,
        "median": estimate.median,
        "values": list(estimate.values),
        "episodes": episodes,
    }


def paired_difference(left: Mapping[str, Any], right: Mapping[str, Any]) -> Dict[str, Any]:
    left_values = np.asarray(left["values"], dtype=np.float64)
    right_values = np.asarray(right["values"], dtype=np.float64)
    if left_values.shape != right_values.shape:
        raise ValueError("paired evaluations have different seed counts")
    estimate = PairedEstimate.from_values(left_values - right_values)
    return {
        "mean": estimate.mean,
        "standard_error": estimate.standard_error,
        "median": estimate.median,
        "values": list(estimate.values),
        "t": (
            estimate.mean / estimate.standard_error
            if np.isfinite(estimate.standard_error) and estimate.standard_error > 0.0
            else float("nan")
        ),
    }


def evaluate_uncapped_price_tail(
    environment_factory: EnvironmentFactory,
    seeds: Sequence[int],
    *,
    multipliers: Sequence[float] = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0),
) -> Dict[str, Any]:
    """Preflight that rejects a price-independent demand/revenue tail.

    Each fixed tariff uses the identical episode seeds (CRN).  The test does
    not impose a price bound: it verifies that the last doubling is not a
    statistically meaningful profit improvement and that Firm 1's completed
    demand at the largest tested fare is negligible relative to the grid.
    """
    grid = tuple(float(value) for value in multipliers)
    if len(grid) < 3 or any(not np.isfinite(value) or value <= 0.0 for value in grid):
        raise ValueError("price-tail grid needs at least three finite positive multipliers")
    if any(right <= left for left, right in zip(grid, grid[1:])):
        raise ValueError("price-tail multipliers must be strictly increasing")
    evaluations = []
    for multiplier in grid:
        evaluation = evaluate_policy(
            environment_factory,
            lambda value=multiplier: FixedPolicy([value, 0.0, 1.0]),
            seeds,
        )
        completed = []
        for episode in evaluation["episodes"]:
            summaries = [
                dict(info.get("summary", {}) or {})
                for info in episode.get("infos", [])
            ]
            completed.append(
                float(np.mean([
                    float(summary.get("completed_share_firm1", 0.0))
                    for summary in summaries
                ]))
                if summaries
                else 0.0
            )
        evaluations.append(
            {
                "multiplier": multiplier,
                "mean": float(evaluation["mean"]),
                "standard_error": float(evaluation["standard_error"]),
                "values": [float(value) for value in evaluation["values"]],
                "mean_completed_share_firm1": float(np.mean(completed)),
            }
        )
    penultimate = {"values": evaluations[-2]["values"]}
    endpoint = {"values": evaluations[-1]["values"]}
    tail_difference = paired_difference(endpoint, penultimate)
    means = np.asarray([row["mean"] for row in evaluations], dtype=float)
    completed_means = np.asarray(
        [row["mean_completed_share_firm1"] for row in evaluations], dtype=float
    )
    standard_error = float(tail_difference["standard_error"])
    numerical_tolerance = 1e-9 * max(1.0, float(np.max(np.abs(means))))
    statistical_tolerance = (
        3.0 * standard_error if np.isfinite(standard_error) else numerical_tolerance
    )
    best_index = int(np.argmax(means))
    demand_tolerance = max(1e-4, 0.05 * float(np.max(completed_means)))
    endpoint_demand_vanishes = bool(completed_means[-1] <= demand_tolerance)
    no_rising_profit_tail = bool(
        float(tail_difference["mean"])
        <= max(numerical_tolerance, statistical_tolerance)
    )
    endpoint_not_optimal = bool(best_index < len(grid) - 1)
    base_result: Dict[str, Any] = {
        "name": "uncapped price-tail well-posedness",
        "criterion": (
            "largest fare is not optimal, final doubling has no significant "
            "profit gain, and completed demand vanishes"
        ),
        "seeds": [int(seed) for seed in seeds],
        "grid": evaluations,
        "best_multiplier": float(grid[best_index]),
        "tail_difference_last_minus_previous": tail_difference,
        "profit_tail_tolerance": float(
            max(numerical_tolerance, statistical_tolerance)
        ),
        "endpoint_completed_share_tolerance": float(demand_tolerance),
        "endpoint_not_optimal": endpoint_not_optimal,
        "no_rising_profit_tail": no_rising_profit_tail,
        "endpoint_demand_vanishes": endpoint_demand_vanishes,
        "passed": bool(
            np.all(np.isfinite(means))
            and endpoint_not_optimal
            and no_rising_profit_tail
            and endpoint_demand_vanishes
        ),
    }

    def stress_row(label: float, action: Sequence[float]) -> Dict[str, Any]:
        physical = np.asarray(action, dtype=np.float64)
        try:
            evaluation = evaluate_policy(
                environment_factory,
                lambda: FixedPolicy(physical),
                seeds,
            )
        except (FloatingPointError, OverflowError) as exc:
            return {
                "label": float(label),
                "action": physical.tolist(),
                "numerically_valid": False,
                "numerical_error": f"{type(exc).__name__}: {exc}",
                "mean": float(NUMERICAL_CANDIDATE_PENALTY),
                "values": [float(NUMERICAL_CANDIDATE_PENALTY)] * len(seeds),
            }
        return {
            "label": float(label),
            "action": physical.tolist(),
            "numerically_valid": True,
            "mean": float(evaluation["mean"]),
            "standard_error": float(evaluation["standard_error"]),
            "values": [float(value) for value in evaluation["values"]],
        }

    def one_sided_status(name: str, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        valid = bool(all(bool(row["numerically_valid"]) for row in rows))
        axis_means = np.asarray([float(row["mean"]) for row in rows])
        best = int(np.argmax(axis_means))
        comparison = paired_difference(rows[-1], rows[-2]) if valid else {
            "mean": float(NUMERICAL_CANDIDATE_PENALTY),
            "standard_error": float("nan"),
            "median": float(NUMERICAL_CANDIDATE_PENALTY),
            "values": [float(NUMERICAL_CANDIDATE_PENALTY)] * len(seeds),
            "t": float("nan"),
        }
        se = float(comparison["standard_error"])
        tolerance = max(
            1e-9 * max(1.0, float(np.max(np.abs(axis_means)))),
            3.0 * se if np.isfinite(se) else 0.0,
        )
        endpoint_not_optimal = bool(
            float(axis_means[-1]) <= float(np.max(axis_means[:-1])) + tolerance
        )
        passed = bool(
            valid
            and endpoint_not_optimal
            and float(comparison["mean"]) <= tolerance
        )
        return {
            "name": name,
            "rows": list(rows),
            "numerically_valid": valid,
            "endpoint_not_optimal": endpoint_not_optimal,
            "argmax_index": best,
            "tail_difference_last_minus_previous": comparison,
            "profit_tail_tolerance": float(tolerance),
            "passed": passed,
        }

    def two_sided_status(name: str, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        valid = bool(all(bool(row["numerically_valid"]) for row in rows))
        axis_means = np.asarray([float(row["mean"]) for row in rows])
        best = int(np.argmax(axis_means))
        if valid:
            left = paired_difference(rows[0], rows[1])
            right = paired_difference(rows[-1], rows[-2])
        else:
            invalid = {
                "mean": float(NUMERICAL_CANDIDATE_PENALTY),
                "standard_error": float("nan"),
                "median": float(NUMERICAL_CANDIDATE_PENALTY),
                "values": [float(NUMERICAL_CANDIDATE_PENALTY)] * len(seeds),
                "t": float("nan"),
            }
            left = dict(invalid)
            right = dict(invalid)
        errors = [float(left["standard_error"]), float(right["standard_error"])]
        tolerance = max(
            1e-9 * max(1.0, float(np.max(np.abs(axis_means)))),
            *(3.0 * value for value in errors if np.isfinite(value)),
        )
        interior_best = float(np.max(axis_means[1:-1]))
        endpoint_not_optimal = bool(
            float(axis_means[0]) <= interior_best + tolerance
            and float(axis_means[-1]) <= interior_best + tolerance
        )
        passed = bool(
            valid
            and endpoint_not_optimal
            and float(left["mean"]) <= tolerance
            and float(right["mean"]) <= tolerance
        )
        return {
            "name": name,
            "rows": list(rows),
            "numerically_valid": valid,
            "endpoint_not_optimal": endpoint_not_optimal,
            "argmax_index": best,
            "left_extreme_minus_inward": left,
            "right_extreme_minus_inward": right,
            "profit_tail_tolerance": float(tolerance),
            "passed": passed,
        }

    incentive_values = (1.0, 2.0, 4.0, 8.0, 16.0)
    incentive = one_sided_status(
        "driver-incentive positive tail",
        [stress_row(value, [1.0, 0.0, value]) for value in incentive_values],
    )
    incentive_low_values = (1.0, 0.5, 0.25, 0.125, 0.0625)
    incentive_low = one_sided_status(
        "driver-incentive open lower tail",
        [
            stress_row(value, [1.0, 0.0, value])
            for value in incentive_low_values
        ],
    )
    surge_values = (-16.0, -8.0, -4.0, 0.0, 4.0, 8.0, 16.0)
    surge = two_sided_status(
        "surge-gain signed tails",
        [stress_row(value, [1.0, value, 1.0]) for value in surge_values],
    )
    joint_values = (1.0, 2.0, 4.0, 8.0, 16.0)
    joint_positive = one_sided_status(
        "joint positive-control tail",
        [
            stress_row(
                value,
                [value, 4.0 * float(np.log2(value)), value],
            )
            for value in joint_values
        ],
    )
    joint_negative = one_sided_status(
        "joint negative-surge tail",
        [
            stress_row(
                value,
                [value, -4.0 * float(np.log2(value)), value],
            )
            for value in joint_values
        ],
    )
    joint_low_incentive = one_sided_status(
        "joint high-price/high-surge/low-incentive tail",
        [
            stress_row(
                value,
                [value, 4.0 * float(np.log2(value)), 1.0 / value],
            )
            for value in joint_values
        ],
    )
    stress_axes = {
        "driver_incentive": incentive,
        "driver_incentive_low": incentive_low,
        "surge_gain": surge,
        "joint_positive": joint_positive,
        "joint_negative_surge": joint_negative,
        "joint_low_incentive": joint_low_incentive,
    }
    base_result["stress_axes"] = stress_axes
    base_result["numerical_actions_valid"] = bool(
        all(bool(axis["numerically_valid"]) for axis in stress_axes.values())
    )
    base_result["all_action_tails_well_posed"] = bool(
        all(bool(axis["passed"]) for axis in stress_axes.values())
    )
    base_result["passed"] = bool(
        base_result["passed"]
        and base_result["numerical_actions_valid"]
        and base_result["all_action_tails_well_posed"]
    )
    base_result["criterion"] += (
        "; upper/lower incentive, signed-surge, and joint action tails are "
        "finite and their far endpoints are not optimal or rising"
    )
    return base_result


class FixedPolicy:
    def __init__(self, action: Sequence[float]) -> None:
        self.action = np.asarray(action, dtype=np.float64)

    def __call__(self, observation: np.ndarray, day: int) -> np.ndarray:
        del observation, day
        return self.action.copy()


class PublicSchedulePolicy:
    """Switch between normal/event actions using only a public observation."""

    def __init__(
        self,
        normal_action: Sequence[float],
        event_action: Sequence[float],
        public_indicator: Callable[[np.ndarray], bool],
    ) -> None:
        self.normal_action = np.asarray(normal_action, dtype=np.float64)
        self.event_action = np.asarray(event_action, dtype=np.float64)
        self.public_indicator = public_indicator

    def __call__(self, observation: np.ndarray, day: int) -> np.ndarray:
        del day
        selected = self.event_action if self.public_indicator(observation) else self.normal_action
        return selected.copy()


class ClairvoyantEventWindowPolicy:
    """Diagnostic schedule that is explicitly allowed to know latent event days.

    This is an oracle upper bound, never an admissible learner or promotion
    policy.  It is useful for measuring how much value is available from
    reacting to an otherwise latent shock without leaking the schedule into
    the learner's observation.
    """

    def __init__(
        self,
        normal_action: Sequence[float],
        event_action: Sequence[float],
        event_days: Sequence[int],
    ) -> None:
        self.normal_action = np.asarray(normal_action, dtype=np.float64)
        self.event_action = np.asarray(event_action, dtype=np.float64)
        self.event_days: Set[int] = {int(day) for day in event_days}

    def __call__(self, observation: np.ndarray, day: int) -> np.ndarray:
        del observation
        selected = self.event_action if int(day) in self.event_days else self.normal_action
        return selected.copy()


def evaluate_clairvoyant_event_schedule(
    environment_factory: EnvironmentFactory,
    normal_action: Sequence[float],
    event_action: Sequence[float],
    event_days_factory: Callable[[int], Sequence[int]],
    seeds: Sequence[int],
) -> Dict[str, Any]:
    """Evaluate a seed-specific latent-event schedule under episode CRN."""
    episodes = []
    for seed in seeds:
        event_days = tuple(int(day) for day in event_days_factory(int(seed)))
        episode = rollout_policy(
            environment_factory(),
            ClairvoyantEventWindowPolicy(
                normal_action, event_action, event_days
            ),
            seed=int(seed),
        )
        episode["oracle_event_days"] = list(event_days)
        episodes.append(episode)
    estimate = PairedEstimate.from_values([episode["return"] for episode in episodes])
    return {
        "mean": estimate.mean,
        "standard_error": estimate.standard_error,
        "median": estimate.median,
        "values": list(estimate.values),
        "episodes": episodes,
        "fair_comparison": False,
        "information": "clairvoyant latent event window known by seed",
    }


class RecurrentAgentPolicy:
    def __init__(self, agent: RecurrentSACAgent, deterministic: bool = True) -> None:
        self.agent = agent
        self.deterministic = bool(deterministic)
        self.hidden = None
        self.previous_action: Optional[np.ndarray] = None
        self.previous_reward = 0.0

    def __call__(self, observation: np.ndarray, day: int) -> np.ndarray:
        del day
        action, self.hidden, _ = self.agent.act(
            observation,
            previous_action=self.previous_action,
            previous_reward=self.previous_reward,
            hidden=self.hidden,
            deterministic=self.deterministic,
        )
        self.previous_action = action.copy()
        # ``rollout_policy`` cannot pass reward into a stateless callable.  The
        # public runner uses ``rollout_recurrent_agent`` below when recurrence
        # must include previous reward; this class remains useful for policies
        # whose observation already contains the previous outcome.
        return action


def rollout_recurrent_agent(
    environment: DailyEnvironment,
    agent: RecurrentSACAgent,
    *,
    seed: int,
    deterministic: bool = True,
) -> Dict[str, Any]:
    observation = environment.reset(int(seed))
    hidden = None
    previous_action = None
    previous_reward = 0.0
    done = False
    total_reward = 0.0
    daily_rewards = []
    actions = []
    infos = []
    day = 0
    while not done:
        action, hidden, action_info = agent.act(
            observation,
            previous_action=previous_action,
            previous_reward=previous_reward,
            hidden=hidden,
            deterministic=deterministic,
        )
        observation, reward, done, info = environment.step(action)
        total_reward += float(reward)
        previous_action = action.copy()
        previous_reward = float(reward)
        daily_rewards.append(float(reward))
        actions.append(action.tolist())
        infos.append({**dict(info), "actor": action_info})
        day += 1
    return {
        "seed": int(seed),
        "return": float(total_reward),
        "days": int(day),
        "daily_rewards": daily_rewards,
        "actions": actions,
        "infos": infos,
    }


def evaluate_recurrent_agent(
    environment_factory: EnvironmentFactory,
    agent: RecurrentSACAgent,
    seeds: Sequence[int],
    *,
    deterministic: bool = True,
) -> Dict[str, Any]:
    episodes = [
        rollout_recurrent_agent(
            environment_factory(), agent, seed=int(seed), deterministic=deterministic
        )
        for seed in seeds
    ]
    estimate = PairedEstimate.from_values([episode["return"] for episode in episodes])
    return {
        "mean": estimate.mean,
        "standard_error": estimate.standard_error,
        "median": estimate.median,
        "values": list(estimate.values),
        "episodes": episodes,
    }


@dataclass
class CEMConfig:
    population: int = 64
    elite_fraction: float = 0.15
    iterations: int = 6
    initial_std: float = 0.75
    minimum_std: float = 0.03
    smoothing: float = 0.20
    seed: int = 0

    def __post_init__(self) -> None:
        if self.population < 2:
            raise ValueError("CEM population must be at least two")
        if not 0.0 < self.elite_fraction <= 1.0:
            raise ValueError("elite_fraction must lie in (0, 1]")
        if self.iterations <= 0 or self.initial_std <= 0.0 or self.minimum_std <= 0.0:
            raise ValueError("CEM iterations and standard deviations must be positive")


class CEMOptimizer:
    """Cross-entropy optimizer in policy-coordinate space."""

    def __init__(self, dimension: int, config: Optional[CEMConfig] = None) -> None:
        self.dimension = int(dimension)
        self.config = config or CEMConfig()
        self.rng = np.random.default_rng(int(self.config.seed))

    def optimize(
        self,
        objective: Callable[[np.ndarray], float],
        *,
        initial_mean: Optional[Sequence[float]] = None,
        initial_std: Optional[Sequence[float] | float] = None,
    ) -> Dict[str, Any]:
        mean = np.zeros(self.dimension, dtype=np.float64)
        if initial_mean is not None:
            mean = np.asarray(initial_mean, dtype=np.float64).reshape(self.dimension)
        if initial_std is None:
            std = np.full(self.dimension, float(self.config.initial_std), dtype=np.float64)
        else:
            std = np.broadcast_to(np.asarray(initial_std, dtype=np.float64), (self.dimension,)).copy()
        # A singleton "elite set" has zero empirical variance.  With the
        # small populations used by smoke diagnostics that immediately
        # collapses CEM around one lucky draw and makes both warm starts and
        # gates optimizer-luck driven.  Two samples are the minimum needed to
        # estimate dispersion; ceil also avoids rounding a positive fraction
        # down prematurely.
        elite_count = min(
            int(self.config.population),
            max(
                2,
                int(np.ceil(self.config.population * self.config.elite_fraction)),
            ),
        )
        best_coordinates = mean.copy()
        best_score = float("-inf")
        history = []
        for iteration in range(int(self.config.iterations)):
            candidates = self.rng.normal(
                loc=mean, scale=std, size=(int(self.config.population), self.dimension)
            )
            # Always re-evaluate the current center.  This makes small-budget
            # runs less sensitive to one unlucky population draw.
            candidates[0] = mean
            scores = np.asarray([float(objective(candidate)) for candidate in candidates])
            if not np.all(np.isfinite(scores)):
                raise FloatingPointError("CEM objective returned a non-finite score")
            order = np.argsort(scores)
            elites = candidates[order[-elite_count:]]
            elite_scores = scores[order[-elite_count:]]
            candidate_mean = elites.mean(axis=0)
            candidate_std = elites.std(axis=0)
            smooth = float(np.clip(self.config.smoothing, 0.0, 1.0))
            mean = smooth * mean + (1.0 - smooth) * candidate_mean
            std = np.maximum(
                smooth * std + (1.0 - smooth) * candidate_std,
                float(self.config.minimum_std),
            )
            index = int(np.argmax(scores))
            if float(scores[index]) > best_score:
                best_score = float(scores[index])
                best_coordinates = candidates[index].copy()
            history.append(
                {
                    "iteration": iteration,
                    "elite_count": int(elite_count),
                    "best_score": float(scores[index]),
                    "elite_mean_score": float(elite_scores.mean()),
                    "mean": mean.tolist(),
                    "std": std.tolist(),
                }
            )
        return {
            "coordinates": best_coordinates,
            "score": best_score,
            "elite_count": int(elite_count),
            "final_mean": mean,
            "final_std": std,
            "history": history,
        }


def optimize_static_policy(
    environment_factory: EnvironmentFactory,
    action_transform: ActionTransform,
    seeds: Sequence[int],
    *,
    config: Optional[CEMConfig] = None,
    initial_action: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Optimize one action for the entire episode on paired CRN seeds."""
    initial_coordinates = (
        np.zeros(action_transform.dim, dtype=np.float64)
        if initial_action is None
        else action_transform.physical_to_coordinates(initial_action)
    )
    optimizer = CEMOptimizer(action_transform.dim, config=config)

    def objective(coordinates: np.ndarray) -> float:
        try:
            physical = action_transform.coordinates_to_physical(coordinates)
        except FloatingPointError:
            return float(NUMERICAL_CANDIDATE_PENALTY)
        return _safe_policy_mean(
            environment_factory,
            lambda: FixedPolicy(physical),
            seeds,
        )

    result = optimizer.optimize(objective, initial_mean=initial_coordinates)
    result["action"] = action_transform.coordinates_to_physical(result["coordinates"]).tolist()
    result["seeds"] = [int(seed) for seed in seeds]
    return result


def optimize_static_tariff(
    environment_factory: EnvironmentFactory,
    action_transform: ActionTransform,
    seeds: Sequence[int],
    *,
    config: Optional[CEMConfig] = None,
    fixed_surge_gain: float = 0.0,
    fixed_driver_incentive: float = 1.0,
) -> Dict[str, Any]:
    """Robustly optimize a fixed tariff on an uncapped positive domain.

    A one-dimensional CEM can miss a broad optimum with a tiny smoke budget.
    Here the search expands geometrically in both directions until an actual
    downturn brackets the best coarse point, then refines that bracket.  The
    expansion limit is a fail-fast compute safeguard, not an economic action
    bound: reaching it raises instead of returning an edge solution.
    """
    if action_transform.names[:3] != (
        "base_multiplier",
        "surge_gain",
        "driver_incentive_multiplier",
    ):
        raise ValueError("static-tariff helper requires the standard market control order")
    search_config = config or CEMConfig()
    evaluations: Dict[float, float] = {}

    def objective_coordinate(coordinate: float) -> float:
        key = float(coordinate)
        if key in evaluations:
            return evaluations[key]
        try:
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                base = float(np.exp(key))
        except (FloatingPointError, OverflowError):
            return float(NUMERICAL_CANDIDATE_PENALTY)
        if not np.isfinite(base) or base <= 0.0:
            score = float(NUMERICAL_CANDIDATE_PENALTY)
            evaluations[key] = score
            return score
        action = np.array(
            [base, float(fixed_surge_gain), float(fixed_driver_incentive)],
            dtype=np.float64,
        )
        score = _safe_policy_mean(
            environment_factory,
            lambda: FixedPolicy(action),
            seeds,
        )
        evaluations[key] = float(score)
        return float(score)

    step = float(np.log(2.0))
    center = 0.0
    objective_coordinate(center)
    max_expansions = 32
    for direction in (-1.0, 1.0):
        previous_coordinate = center
        previous_score = objective_coordinate(previous_coordinate)
        found_downturn = False
        for expansion in range(1, max_expansions + 1):
            coordinate = direction * step * expansion
            score = objective_coordinate(coordinate)
            if score < previous_score:
                found_downturn = True
                break
            previous_coordinate = coordinate
            previous_score = score
        if not found_downturn:
            raise RuntimeError(
                "uncapped static-tariff search found no downturn after "
                f"{max_expansions} doublings toward direction {direction:+.0f}"
            )

    ordered = sorted(evaluations)
    best_coordinate = max(ordered, key=lambda value: evaluations[value])
    best_index = ordered.index(best_coordinate)
    if best_index == 0 or best_index == len(ordered) - 1:
        raise RuntimeError("static tariff optimum was not bracketed by downturns")
    left = float(ordered[best_index - 1])
    right = float(ordered[best_index + 1])
    refinement_history: List[Dict[str, Any]] = []
    refinement_rounds = max(2, int(search_config.iterations))
    for iteration in range(refinement_rounds):
        grid = np.linspace(left, right, 9)
        scores = np.asarray([objective_coordinate(float(value)) for value in grid])
        winner = int(np.argmax(scores))
        if winner == 0:
            new_left, new_right = float(grid[0]), float(grid[1])
        elif winner == len(grid) - 1:
            new_left, new_right = float(grid[-2]), float(grid[-1])
        else:
            new_left, new_right = float(grid[winner - 1]), float(grid[winner + 1])
        refinement_history.append(
            {
                "iteration": int(iteration),
                "left_log_multiplier": float(left),
                "right_log_multiplier": float(right),
                "best_log_multiplier": float(grid[winner]),
                "best_score": float(scores[winner]),
            }
        )
        left, right = new_left, new_right

    best_coordinate = max(evaluations, key=lambda value: evaluations[value])
    best_score = float(evaluations[best_coordinate])
    best_base = float(np.exp(best_coordinate))
    result: Dict[str, Any] = {
        "coordinates": np.asarray([best_coordinate], dtype=np.float64),
        "score": best_score,
        "final_mean": np.asarray([best_coordinate], dtype=np.float64),
        "final_std": np.asarray([(right - left) / 2.0], dtype=np.float64),
        "history": refinement_history,
        "search_method": "adaptive_log2_bracket_then_grid_refine",
        "coarse_to_fine_evaluations": [
            {
                "log_multiplier": float(coordinate),
                "multiplier": float(np.exp(coordinate)),
                "score": float(evaluations[coordinate]),
            }
            for coordinate in sorted(evaluations)
        ],
        "bracket_verified": True,
    }
    result["action"] = [
        best_base,
        float(fixed_surge_gain),
        float(fixed_driver_incentive),
    ]
    result["seeds"] = [int(seed) for seed in seeds]
    return result


def optimize_public_schedule(
    environment_factory: EnvironmentFactory,
    action_transform: ActionTransform,
    seeds: Sequence[int],
    public_indicator: Callable[[np.ndarray], bool],
    normal_action: Sequence[float],
    *,
    config: Optional[CEMConfig] = None,
) -> Dict[str, Any]:
    """Optimize only the event action around a fixed full-control rule.

    Locking the normal regime to the fitted three-dimensional rule isolates
    the value of switching on public shock information.  Otherwise the
    schedule can win merely by discovering a better permanent surge or driver
    incentive and the adaptivity denominator is not comparable with RL.
    """
    normal = np.asarray(normal_action, dtype=np.float64).reshape(action_transform.dim)
    base = action_transform.physical_to_coordinates(normal)
    optimizer = CEMOptimizer(action_transform.dim, config=config)

    def objective(coordinates: np.ndarray) -> float:
        try:
            event = action_transform.coordinates_to_physical(coordinates)
        except FloatingPointError:
            return float(NUMERICAL_CANDIDATE_PENALTY)
        return _safe_policy_mean(
            environment_factory,
            lambda: PublicSchedulePolicy(normal, event, public_indicator),
            seeds,
        )

    result = optimizer.optimize(objective, initial_mean=base)
    event = action_transform.coordinates_to_physical(result["coordinates"])
    result["normal_action"] = normal.tolist()
    result["event_action"] = event.tolist()
    result["seeds"] = [int(seed) for seed in seeds]
    result["normal_regime_fixed_to"] = "best_fixed_full_3d_rule"
    return result


def optimize_clairvoyant_event_action(
    environment_factory: EnvironmentFactory,
    action_transform: ActionTransform,
    seeds: Sequence[int],
    event_days_factory: Callable[[int], Sequence[int]],
    normal_action: Sequence[float],
    *,
    config: Optional[CEMConfig] = None,
) -> Dict[str, Any]:
    """Optimize only the latent-event action around a fixed normal rule.

    Fixing ``normal_action`` to the validation-selected full three-control
    rule makes the oracle gain a clean switching opportunity.  It cannot
    absorb a permanently better surge/incentive rule into its normal regime.
    """
    normal = np.asarray(normal_action, dtype=np.float64).reshape(action_transform.dim)
    initial_coordinates = action_transform.physical_to_coordinates(normal)
    optimizer = CEMOptimizer(action_transform.dim, config=config)

    def objective(coordinates: np.ndarray) -> float:
        try:
            event = action_transform.coordinates_to_physical(coordinates)
        except FloatingPointError:
            return float(NUMERICAL_CANDIDATE_PENALTY)
        try:
            return float(
                evaluate_clairvoyant_event_schedule(
                    environment_factory,
                    normal,
                    event,
                    event_days_factory,
                    seeds,
                )["mean"]
            )
        except (FloatingPointError, OverflowError):
            return float(NUMERICAL_CANDIDATE_PENALTY)

    result = optimizer.optimize(objective, initial_mean=initial_coordinates)
    event_action = action_transform.coordinates_to_physical(result["coordinates"])
    result["normal_action"] = normal.tolist()
    result["event_action"] = event_action.tolist()
    result["seeds"] = [int(seed) for seed in seeds]
    result["fair_comparison"] = False
    result["information"] = "clairvoyant latent event window known by seed"
    result["normal_regime_fixed_to"] = "best_fixed_full_3d_rule"
    return result


class RecedingHorizonCEM:
    """CEM/MPC upper bound over exact simulator-state branches.

    ``ContinuousMarketEnv`` intentionally supports only
    ``reveal_latent=True`` until a tested public-history belief filter exists.
    Consequently this planner is a perfect-state diagnostic upper bound and
    must not be used in a fair learner promotion gate.
    """

    def __init__(
        self,
        action_transform: ActionTransform,
        horizon: int = 7,
        particles: int = 4,
        discount: float = 1.0,
        config: Optional[CEMConfig] = None,
        reveal_latent: bool = True,
    ) -> None:
        self.action_transform = action_transform
        self.horizon = int(max(1, horizon))
        self.particles = int(max(1, particles))
        self.discount = float(discount)
        self.config = config or CEMConfig(population=32, iterations=4)
        self.reveal_latent = bool(reveal_latent)
        self.calls = 0

    def _branch(self, environment: DailyEnvironment, seed: int) -> DailyEnvironment:
        if hasattr(environment, "planning_branch"):
            return environment.planning_branch(  # type: ignore[attr-defined]
                int(seed), reveal_latent=self.reveal_latent
            )
        if not self.reveal_latent:
            raise TypeError("environment has no public-history belief branch")
        return copy.deepcopy(environment)

    def plan(self, environment: DailyEnvironment, *, seed: int) -> Dict[str, Any]:
        dimension = self.horizon * self.action_transform.dim
        config = copy.copy(self.config)
        config.seed = int(seed) + self.calls * 104_729
        optimizer = CEMOptimizer(dimension, config=config)
        particle_seeds = [int(seed) + 7_919 * (index + 1) for index in range(self.particles)]

        def objective(flat: np.ndarray) -> float:
            coordinates = flat.reshape(self.horizon, self.action_transform.dim)
            try:
                actions = self.action_transform.coordinates_to_physical(coordinates)
            except FloatingPointError:
                return float(NUMERICAL_CANDIDATE_PENALTY)
            returns = []
            try:
                for particle_seed in particle_seeds:
                    branch = self._branch(environment, particle_seed)
                    value = 0.0
                    weight = 1.0
                    for action in actions:
                        _, reward, done, _ = branch.step(action)
                        value += weight * float(reward)
                        weight *= self.discount
                        if done:
                            break
                    returns.append(value)
            except (FloatingPointError, OverflowError):
                return float(NUMERICAL_CANDIDATE_PENALTY)
            return float(np.mean(returns))

        result = optimizer.optimize(objective)
        plan_coordinates = result["coordinates"].reshape(
            self.horizon, self.action_transform.dim
        )
        plan_actions = self.action_transform.coordinates_to_physical(plan_coordinates)
        self.calls += 1
        return {
            **result,
            "action": plan_actions[0].copy(),
            "plan": plan_actions,
            "particle_seeds": particle_seeds,
            "reveal_latent": self.reveal_latent,
        }


__all__ = [
    "CEMConfig",
    "CEMOptimizer",
    "ClairvoyantEventWindowPolicy",
    "FixedPolicy",
    "PairedEstimate",
    "PublicSchedulePolicy",
    "RecedingHorizonCEM",
    "evaluate_policy",
    "evaluate_clairvoyant_event_schedule",
    "evaluate_recurrent_agent",
    "evaluate_uncapped_price_tail",
    "optimize_public_schedule",
    "optimize_clairvoyant_event_action",
    "optimize_static_policy",
    "optimize_static_tariff",
    "paired_difference",
    "rollout_policy",
    "rollout_recurrent_agent",
]
