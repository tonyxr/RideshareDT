"""Auditable, paired evaluation and publication reporting for v28 policies.

This module is deliberately separate from the learner and simulator.  It does
not select hyperparameters or train a policy.  Instead it evaluates already
frozen policies under common random numbers, records operational and welfare
outcomes, and renders an immutable report artifact.

Two details are important for interpreting its output:

* Statistical uncertainty is clustered at the episode seed.  Intraday blocks
  and days are measurements within a seed, not independent replicates.
* The current :class:`continuous_market_env.ContinuousMarketEnv` returns a
  compact daily summary.  During a live evaluation this module attaches a
  read-only recorder to the fresh episode Core so exact intraday numerators
  for fares, pay, and welfare can be aggregated.  When only compact historical
  episode dictionaries are available, unsupported metrics are reported as
  missing rather than reconstructed from assumptions.

The final-test guard is intentionally persistent.  Claiming a final test split
consumes it before the first rollout, so a crash cannot silently turn the test
set into a repeatable model-selection set.
"""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np


REPORT_SCHEMA = "v28_evaluation_report/1.0"
FROZEN_SCHEMA = "v28_frozen_evaluation/1.0"


class EvaluationContractError(RuntimeError):
    """Raised when an evaluation would violate split or freezing semantics."""


class FinalTestAlreadyConsumed(EvaluationContractError):
    """Raised when a run attempts to inspect its final test split twice."""


class ReportIntegrityError(RuntimeError):
    """Raised when a frozen report no longer matches its content hash."""


class DailyEnvironment(Protocol):
    profile_split: str

    def reset(self, seed: int) -> np.ndarray: ...

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Mapping[str, Any]]: ...


class ContextualPolicy(Protocol):
    def select_action(
        self,
        observation: np.ndarray,
        *,
        day: int,
        previous_action: Optional[np.ndarray],
        previous_reward: float,
    ) -> np.ndarray: ...


EnvironmentFactory = Callable[[], DailyEnvironment]


class FixedEvaluationPolicy:
    """A frozen three-control action suitable for a static benchmark."""

    def __init__(self, action: Sequence[float]) -> None:
        array = np.asarray(action, dtype=np.float64).reshape(-1)
        if array.shape != (3,) or not np.all(np.isfinite(array)):
            raise ValueError("fixed evaluation action must be three finite values")
        self.action = array

    def __call__(self, observation: np.ndarray, day: int) -> np.ndarray:
        del observation, day
        return self.action.copy()


class RecurrentAgentEvaluationPolicy:
    """Reward-aware evaluation adapter for a frozen recurrent SAC-like agent."""

    def __init__(self, agent: Any, *, deterministic: bool = True) -> None:
        if not callable(getattr(agent, "act", None)):
            raise TypeError("recurrent evaluation agent must expose act(...)")
        self.agent = agent
        self.deterministic = bool(deterministic)
        self.hidden: Any = None

    def select_action(
        self,
        observation: np.ndarray,
        *,
        day: int,
        previous_action: Optional[np.ndarray],
        previous_reward: float,
    ) -> np.ndarray:
        del day
        action, self.hidden, _ = self.agent.act(
            observation,
            previous_action=previous_action,
            previous_reward=float(previous_reward),
            hidden=self.hidden,
            deterministic=self.deterministic,
        )
        return np.asarray(action, dtype=np.float64)


class ClairvoyantEventPolicyFactory:
    """Seed-aware factory for a clearly labelled latent-event upper bound."""

    def __init__(
        self,
        normal_action: Sequence[float],
        event_action: Sequence[float],
    ) -> None:
        self.normal_action = np.asarray(normal_action, dtype=np.float64).reshape(-1)
        self.event_action = np.asarray(event_action, dtype=np.float64).reshape(-1)
        if self.normal_action.shape != (3,) or self.event_action.shape != (3,):
            raise ValueError("clairvoyant actions must each have shape (3,)")

    def for_episode(self, environment: Any, seed: int) -> Any:
        if not callable(getattr(environment, "clairvoyant_event_days", None)):
            raise EvaluationContractError(
                "clairvoyant oracle requires environment.clairvoyant_event_days(seed)"
            )
        event_days = {
            int(day) for day in environment.clairvoyant_event_days(int(seed))
        }
        normal = self.normal_action.copy()
        event = self.event_action.copy()

        def policy(observation: np.ndarray, day: int) -> np.ndarray:
            del observation
            return (event if int(day) in event_days else normal).copy()

        return policy


@dataclass(frozen=True)
class BootstrapConfig:
    replicates: int = 5000
    confidence: float = 0.95
    seed: int = 28_104

    def __post_init__(self) -> None:
        if self.replicates < 100:
            raise ValueError("bootstrap replicates must be at least 100")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("bootstrap confidence must lie in (0, 1)")


@dataclass(frozen=True)
class EvaluationContext:
    """Declares how a policy suite may use a rider-profile split."""

    run_id: str
    split: str
    purpose: str = "validation_diagnostic"
    policies_frozen: bool = False
    selection_performed: bool = False
    final_test_authorized: bool = False
    ledger_path: Optional[Path] = None

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id must be non-empty")
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        if self.split == "test":
            if self.selection_performed:
                raise EvaluationContractError(
                    "the final test split cannot rank, tune, gate, or select policies"
                )
            if not self.policies_frozen:
                raise EvaluationContractError(
                    "all policies and benchmark actions must be frozen before final test"
                )
            if not self.final_test_authorized:
                raise EvaluationContractError(
                    "final test use requires an explicit final_test_authorized=True declaration"
                )
            if self.ledger_path is None:
                raise EvaluationContractError(
                    "final test use requires a persistent one-shot ledger path"
                )


METRIC_CATALOG: Dict[str, Dict[str, str]] = {
    "episode_return": {
        "label": "Episode return",
        "unit": "objective units / episode",
        "scope": "Firm 1 learner objective",
        "direction": "higher",
    },
    "profit_per_request_firm1": {
        "label": "Firm 1 contribution profit",
        "unit": "currency / reference request",
        "scope": "Firm 1",
        "direction": "higher",
    },
    "completed_request_share_firm1": {
        "label": "Firm 1 completed-request share",
        "unit": "share of all requests",
        "scope": "Firm 1",
        "direction": "higher",
    },
    "platform_share_of_completed_rides_firm1": {
        "label": "Firm 1 platform share",
        "unit": "share of completed platform rides",
        "scope": "Firm 1 vs Firm 2",
        "direction": "higher",
    },
    "market_completion_rate": {
        "label": "Market completion rate",
        "unit": "completed rides / request",
        "scope": "both platforms",
        "direction": "higher",
    },
    "fulfillment_rate_firm1": {
        "label": "Firm 1 fulfillment",
        "unit": "completed rides / dispatch offer",
        "scope": "Firm 1",
        "direction": "higher",
    },
    "mean_wait_minutes_firm1": {
        "label": "Firm 1 wait",
        "unit": "minutes / completed ride",
        "scope": "Firm 1",
        "direction": "lower",
    },
    "mean_fare_paid_firm1": {
        "label": "Firm 1 mean fare",
        "unit": "currency / completed ride",
        "scope": "Firm 1",
        "direction": "context",
    },
    "driver_pay_per_completed_firm1": {
        "label": "Firm 1 driver pay",
        "unit": "currency / completed ride",
        "scope": "Firm 1",
        "direction": "higher",
    },
    "driver_pay_per_request_firm1": {
        "label": "Firm 1 driver pay",
        "unit": "currency / request",
        "scope": "Firm 1",
        "direction": "higher",
    },
    "driver_earnings_per_online_hour_firm1": {
        "label": "Firm 1 driver earnings",
        "unit": "currency / online hour",
        "scope": (
            "Firm 1 online supply; current-block earnings weighted by online "
            "driver-hours (fleet telemetry, with exact batch-state fallback)"
        ),
        "direction": "higher",
    },
    "mean_fare_paid_market": {
        "label": "Market mean fare",
        "unit": "currency / completed ride",
        "scope": "both platforms",
        "direction": "context",
    },
    "consumer_surplus_per_request_market": {
        "label": "Consumer surplus",
        "unit": "currency / request",
        "scope": "both platforms; reservation fare minus paid fare",
        "direction": "higher",
    },
    "driver_pay_per_request_market": {
        "label": "Market driver pay",
        "unit": "currency / request",
        "scope": "both platforms",
        "direction": "higher",
    },
    "total_surplus_per_request_market": {
        "label": "Total modeled surplus",
        "unit": "currency / request",
        "scope": "consumer surplus + platform contribution + gross driver pay",
        "direction": "higher",
    },
    "equity_served_gap_high_minus_low": {
        "label": "High-minus-low income served-rate gap",
        "unit": "share points",
        "scope": "200k+ minus <50k; both platforms",
        "direction": "context",
    },
}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return value


def _stable_seed(base: int, label: str) -> int:
    digest = hashlib.sha256(f"{int(base)}|{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _finite_or_none(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def summarize_seed_values(
    values: Sequence[float],
    *,
    bootstrap: BootstrapConfig,
    label: str,
) -> Dict[str, Any]:
    """Summarize independent episode-seed values with a percentile CI."""
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "available": False,
            "n_seeds": 0,
            "mean": None,
            "median": None,
            "standard_error": None,
            "ci_lower": None,
            "ci_upper": None,
            "confidence": bootstrap.confidence,
            "method": "seed_cluster_percentile_bootstrap",
            "reason": "no finite seed-level measurements",
        }
    result: Dict[str, Any] = {
        "available": True,
        "n_seeds": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "standard_error": (
            float(np.std(array, ddof=1) / math.sqrt(array.size))
            if array.size > 1
            else None
        ),
        "ci_lower": None,
        "ci_upper": None,
        "confidence": bootstrap.confidence,
        "method": "seed_cluster_percentile_bootstrap",
        "bootstrap_replicates": int(bootstrap.replicates),
        "values": [float(item) for item in array],
    }
    if array.size > 1:
        rng = np.random.default_rng(_stable_seed(bootstrap.seed, label))
        indices = rng.integers(0, array.size, size=(bootstrap.replicates, array.size))
        means = np.mean(array[indices], axis=1)
        alpha = (1.0 - bootstrap.confidence) / 2.0
        result["ci_lower"] = float(np.quantile(means, alpha))
        result["ci_upper"] = float(np.quantile(means, 1.0 - alpha))
    else:
        result["reason_ci_unavailable"] = "at least two episode seeds are required"
    return result


class OneShotTestLedger:
    """Atomically record the sole permitted inspection of a final test split."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.claim_path = self.path.with_suffix(self.path.suffix + ".claimed")

    def consume(
        self,
        *,
        run_id: str,
        seeds: Sequence[int],
        policy_manifest: Mapping[str, Any],
    ) -> Dict[str, Any]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        claim = {
            "schema": "v28_final_test_claim/1.0",
            "run_id": str(run_id),
            "seed_sha256": _sha256_json([int(seed) for seed in seeds]),
            "policy_manifest_sha256": _sha256_json(policy_manifest),
            "claimed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "status": "consumed_before_rollout",
        }
        try:
            descriptor = os.open(
                self.claim_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
        except FileExistsError as exc:
            existing = ""
            try:
                existing = self.claim_path.read_text(encoding="utf-8")
            except OSError:
                pass
            raise FinalTestAlreadyConsumed(
                f"final test already claimed at {self.claim_path}; {existing[:300]}"
            ) from exc
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(_canonical_json(claim) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(_json_safe(claim), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.path)
        return claim


def _income_metric_name(prefix: str, bracket: str) -> str:
    return f"{prefix}__{bracket}"


def _ratio(numerator: float, denominator: float) -> Optional[float]:
    if denominator <= 0.0 or not math.isfinite(numerator) or not math.isfinite(denominator):
        return None
    value = numerator / denominator
    return float(value) if math.isfinite(value) else None


class _CoreBlockRecorder:
    """Capture exact block totals without changing simulator decisions."""

    def __init__(self, environment: Any) -> None:
        self.environment = environment
        self.core = getattr(environment, "core", None)
        self.blocks: List[Dict[str, Any]] = []
        self._original: Optional[Callable[..., Any]] = None

    @property
    def available(self) -> bool:
        return self.core is not None and callable(getattr(self.core, "simulate_batch", None))

    def attach(self) -> None:
        if not self.available:
            return
        self._original = self.core.simulate_batch

        def recorded(*args: Any, **kwargs: Any) -> Any:
            assert self._original is not None
            result = self._original(*args, **kwargs)
            self.blocks.append(self._capture(result))
            return result

        self.core.simulate_batch = recorded

    def detach(self) -> None:
        if self._original is not None:
            self.core.simulate_batch = self._original
            self._original = None

    def _capture(self, result: Any) -> Dict[str, Any]:
        row: Dict[str, Any] = {"source": "core_intraday_block"}
        try:
            firm1, firm2 = result[1], result[2]
            total = float(getattr(firm1, "total", 0.0))
            money_requests = float(getattr(firm1, "reference_total", 0.0) or total)
            row.update(
                requests=total,
                money_requests=money_requests,
                firm1_completed=float(getattr(firm1, "completed", 0.0)),
                firm2_completed=float(getattr(firm2, "completed", 0.0)),
                firm1_offers=float(getattr(firm1, "dispatch_offers", 0.0)),
                firm1_profit=float(getattr(firm1, "profit", 0.0)),
                firm2_profit=float(getattr(firm2, "profit", 0.0)),
                firm1_revenue=float(getattr(firm1, "revenue", 0.0)),
                firm2_revenue=float(getattr(firm2, "revenue", 0.0)),
                firm1_driver_pay=float(getattr(firm1, "driver_pay", 0.0)),
                firm2_driver_pay=float(getattr(firm2, "driver_pay", 0.0)),
                firm1_wait_minutes=float(getattr(firm1, "wait_minutes", 0.0)),
            )
        except (IndexError, TypeError):
            row["capture_error"] = "simulate_batch did not return FirmMetrics positions 1 and 2"
            return row

        welfare = copy.deepcopy(getattr(self.core, "last_welfare_stats", {}) or {})
        row["welfare_by_income"] = {
            str(bracket): [float(value) for value in values]
            for bracket, values in dict(welfare.get("by_income", {}) or {}).items()
        }
        row["market_driver_pay"] = float(
            welfare.get(
                "driver_pay",
                row.get("firm1_driver_pay", 0.0) + row.get("firm2_driver_pay", 0.0),
            )
        )
        states = getattr(getattr(self.core, "driver_supply", None), "last_states", {}) or {}
        state = states.get("Firm1") if isinstance(states, Mapping) else None
        if state is not None:
            row["firm1_online_minutes"] = float(
                getattr(state, "total_online_minutes", 0.0)
            )
            row["firm1_driver_pay_weighted"] = float(
                getattr(state, "total_driver_pay", 0.0)
            )
        fleet = dict((getattr(self.core, "last_v27_step", {}) or {}).get("fleet", {}) or {})
        fleet1 = dict(fleet.get("Firm1", {}) or {})
        if fleet1:
            earnings = _finite_or_none(
                fleet1.get("earnings_per_online_hour")
            )
            online_drivers = _finite_or_none(fleet1.get("online_drivers"))
            block_minutes: Optional[float] = None
            block_duration = getattr(self.core, "_v27_block_minutes", None)
            if callable(block_duration):
                block_minutes = _finite_or_none(block_duration())
            if block_minutes is None:
                config = getattr(self.environment, "config", None)
                steps = _finite_or_none(getattr(config, "steps_per_day", None))
                if steps is not None and steps > 0.0:
                    block_minutes = 24.0 * 60.0 / steps
            row["firm1_earnings_reported"] = earnings
            if (
                earnings is not None
                and online_drivers is not None
                and online_drivers >= 0.0
                and block_minutes is not None
                and block_minutes > 0.0
            ):
                online_minutes = online_drivers * block_minutes
                row["firm1_reported_online_minutes"] = online_minutes
                row["firm1_reported_driver_pay_equivalent"] = (
                    earnings * online_minutes / 60.0
                )
        return row


def _sum_blocks(blocks: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    keys = (
        "requests",
        "money_requests",
        "firm1_completed",
        "firm2_completed",
        "firm1_offers",
        "firm1_profit",
        "firm2_profit",
        "firm1_revenue",
        "firm2_revenue",
        "firm1_driver_pay",
        "firm2_driver_pay",
        "firm1_wait_minutes",
        "market_driver_pay",
        "firm1_online_minutes",
        "firm1_driver_pay_weighted",
        "firm1_reported_online_minutes",
        "firm1_reported_driver_pay_equivalent",
    )
    totals: Dict[str, Any] = {key: 0.0 for key in keys}
    present: set[str] = set()
    welfare: Dict[str, List[float]] = {}
    for block in blocks:
        for key in keys:
            value = _finite_or_none(block.get(key))
            if value is not None:
                totals[key] += value
                present.add(key)
        for bracket, values in dict(block.get("welfare_by_income", {}) or {}).items():
            numbers = [float(value) for value in values]
            bucket = welfare.setdefault(str(bracket), [0.0] * len(numbers))
            if len(bucket) < len(numbers):
                bucket.extend([0.0] * (len(numbers) - len(bucket)))
            for index, value in enumerate(numbers):
                bucket[index] += value
    totals["_present"] = sorted(present)
    totals["welfare_by_income"] = welfare
    totals["block_count"] = int(len(blocks))
    return totals


def _metrics_from_block_totals(totals: Mapping[str, Any]) -> Tuple[Dict[str, Optional[float]], Dict[str, str]]:
    present = set(totals.get("_present", []))
    requests = float(totals.get("requests", 0.0))
    money_requests = float(totals.get("money_requests", 0.0))
    completed1 = float(totals.get("firm1_completed", 0.0))
    completed2 = float(totals.get("firm2_completed", 0.0))
    completed_market = completed1 + completed2
    # Prefer the current-block public fleet telemetry, weighted by online
    # driver-hours.  The exact FleetBatchState is an equivalent fallback in
    # the vector engine and avoids an unweighted mean of hourly earnings.
    if (
        "firm1_reported_online_minutes" in present
        and float(totals.get("firm1_reported_online_minutes", 0.0)) > 0.0
    ):
        earnings_pay = float(totals.get("firm1_reported_driver_pay_equivalent", 0.0))
        earnings_minutes = float(totals.get("firm1_reported_online_minutes", 0.0))
        earnings_required = {
            "firm1_reported_driver_pay_equivalent",
            "firm1_reported_online_minutes",
        }
    else:
        earnings_pay = float(totals.get("firm1_driver_pay_weighted", 0.0))
        earnings_minutes = float(totals.get("firm1_online_minutes", 0.0))
        earnings_required = {"firm1_driver_pay_weighted", "firm1_online_minutes"}

    values: Dict[str, Optional[float]] = {
        "profit_per_request_firm1": _ratio(
            float(totals.get("firm1_profit", 0.0)), money_requests
        ),
        "completed_request_share_firm1": _ratio(completed1, requests),
        "platform_share_of_completed_rides_firm1": _ratio(completed1, completed_market),
        "market_completion_rate": _ratio(completed_market, requests),
        "fulfillment_rate_firm1": _ratio(
            completed1, float(totals.get("firm1_offers", 0.0))
        ),
        "mean_wait_minutes_firm1": _ratio(
            float(totals.get("firm1_wait_minutes", 0.0)), completed1
        ),
        "mean_fare_paid_firm1": _ratio(
            float(totals.get("firm1_revenue", 0.0)), completed1
        ),
        "driver_pay_per_completed_firm1": _ratio(
            float(totals.get("firm1_driver_pay", 0.0)), completed1
        ),
        "driver_pay_per_request_firm1": _ratio(
            float(totals.get("firm1_driver_pay", 0.0)), requests
        ),
        "driver_earnings_per_online_hour_firm1": _ratio(
            60.0 * earnings_pay,
            earnings_minutes,
        ),
    }
    missing: Dict[str, str] = {}
    required = {
        "profit_per_request_firm1": {"firm1_profit", "money_requests"},
        "completed_request_share_firm1": {"firm1_completed", "requests"},
        "platform_share_of_completed_rides_firm1": {"firm1_completed", "firm2_completed"},
        "market_completion_rate": {"firm1_completed", "firm2_completed", "requests"},
        "fulfillment_rate_firm1": {"firm1_completed", "firm1_offers"},
        "mean_wait_minutes_firm1": {"firm1_wait_minutes", "firm1_completed"},
        "mean_fare_paid_firm1": {"firm1_revenue", "firm1_completed"},
        "driver_pay_per_completed_firm1": {"firm1_driver_pay", "firm1_completed"},
        "driver_pay_per_request_firm1": {"firm1_driver_pay", "requests"},
        "driver_earnings_per_online_hour_firm1": earnings_required,
    }
    for metric, keys in required.items():
        if not keys.issubset(present):
            values[metric] = None
            missing[metric] = "exact Core block numerator/denominator was not exposed"
        elif values[metric] is None:
            missing[metric] = "metric denominator was zero"

    welfare = dict(totals.get("welfare_by_income", {}) or {})
    if welfare:
        matrix = np.asarray(list(welfare.values()), dtype=np.float64)
        if matrix.ndim == 2 and matrix.shape[1] >= 6:
            aggregate = np.sum(matrix, axis=0)
            welfare_requests = float(aggregate[0])
            welfare_served = float(aggregate[1])
            market_profit = float(totals.get("firm1_profit", 0.0)) + float(
                totals.get("firm2_profit", 0.0)
            )
            consumer_surplus = _ratio(float(aggregate[3]), welfare_requests)
            platform_profit = _ratio(market_profit, welfare_requests)
            market_driver_pay = _ratio(
                float(totals.get("market_driver_pay", 0.0)), welfare_requests
            )
            values.update(
                mean_fare_paid_market=_ratio(float(aggregate[2]), welfare_served),
                consumer_surplus_per_request_market=consumer_surplus,
                driver_pay_per_request_market=market_driver_pay,
                total_surplus_per_request_market=(
                    consumer_surplus + platform_profit + market_driver_pay
                    if consumer_surplus is not None
                    and platform_profit is not None
                    and market_driver_pay is not None
                    else None
                ),
            )
            served_rates: Dict[str, float] = {}
            for bracket, raw in sorted(welfare.items()):
                row = np.asarray(raw, dtype=np.float64)
                if row.size < 6:
                    continue
                bracket_requests = float(row[0])
                bracket_served = float(row[1])
                served_rate = _ratio(bracket_served, bracket_requests)
                if served_rate is not None:
                    served_rates[str(bracket)] = served_rate
                values[_income_metric_name("equity_served_rate", str(bracket))] = served_rate
                values[_income_metric_name("equity_mean_fare_paid", str(bracket))] = _ratio(
                    float(row[2]), bracket_served
                )
                values[_income_metric_name("equity_surplus_per_request", str(bracket))] = _ratio(
                    float(row[3]), bracket_requests
                )
                values[_income_metric_name("equity_mean_wait_minutes", str(bracket))] = _ratio(
                    float(row[5]), bracket_served
                )
            if "<50k" in served_rates and "200k+" in served_rates:
                values["equity_served_gap_high_minus_low"] = (
                    served_rates["200k+"] - served_rates["<50k"]
                )
            else:
                values["equity_served_gap_high_minus_low"] = None
                missing["equity_served_gap_high_minus_low"] = (
                    "both <50k and 200k+ income brackets were not observed"
                )
    else:
        for metric in (
            "mean_fare_paid_market",
            "consumer_surplus_per_request_market",
            "driver_pay_per_request_market",
            "total_surplus_per_request_market",
            "equity_served_gap_high_minus_low",
        ):
            values[metric] = None
            missing[metric] = "income-bracket welfare totals were not exposed"
    return values, missing


def _metrics_from_daily_infos(
    infos: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Optional[float]], Dict[str, str]]:
    """Fallback for compact historical rollouts without Core block totals."""
    summaries = [dict(info.get("summary", {}) or {}) for info in infos]

    def mean_of(key: str) -> Optional[float]:
        numbers = [_finite_or_none(row.get(key)) for row in summaries]
        finite = [value for value in numbers if value is not None]
        return float(np.mean(finite)) if finite else None

    completed1 = mean_of("completed_share_firm1")
    completed2 = mean_of("completed_share_firm2")
    values: Dict[str, Optional[float]] = {
        "profit_per_request_firm1": mean_of("profit_firm1"),
        "completed_request_share_firm1": completed1,
        "market_completion_rate": (
            completed1 + completed2
            if completed1 is not None and completed2 is not None
            else None
        ),
        "platform_share_of_completed_rides_firm1": (
            _ratio(completed1, completed1 + completed2)
            if completed1 is not None and completed2 is not None
            else None
        ),
        "fulfillment_rate_firm1": mean_of("fulfillment_firm1"),
        "mean_wait_minutes_firm1": mean_of("wait_firm1"),
        "driver_earnings_per_online_hour_firm1": mean_of("earnings_firm1"),
    }
    missing: Dict[str, str] = {
        "mean_fare_paid_firm1": "compact daily summary does not contain fare totals",
        "driver_pay_per_completed_firm1": "compact daily summary does not contain driver-pay totals",
        "driver_pay_per_request_firm1": "compact daily summary does not contain driver-pay totals",
        "mean_fare_paid_market": "compact daily summary does not contain welfare totals",
        "consumer_surplus_per_request_market": "compact daily summary does not contain welfare totals",
        "driver_pay_per_request_market": "compact daily summary does not contain welfare totals",
        "total_surplus_per_request_market": "compact daily summary does not contain welfare totals",
        "equity_served_gap_high_minus_low": "compact daily summary does not contain income-bracket totals",
    }
    for metric in missing:
        values[metric] = None
    for metric, value in values.items():
        if value is None and metric not in missing:
            missing[metric] = "metric was not present in compact daily summaries"
    return values, missing


def _select_action(
    policy: Any,
    observation: np.ndarray,
    *,
    day: int,
    previous_action: Optional[np.ndarray],
    previous_reward: float,
) -> np.ndarray:
    if hasattr(policy, "select_action"):
        action = policy.select_action(
            observation.copy(),
            day=int(day),
            previous_action=(None if previous_action is None else previous_action.copy()),
            previous_reward=float(previous_reward),
        )
    elif callable(policy):
        action = policy(observation.copy(), int(day))
    else:
        raise TypeError("policy must be callable or implement select_action")
    array = np.asarray(action, dtype=np.float64).reshape(-1)
    if array.shape != (3,):
        raise ValueError("evaluation policy action must have shape (3,)")
    if not np.all(np.isfinite(array)):
        raise FloatingPointError("evaluation policy emitted a non-finite action")
    return array


def _instantiate_policy(factory: Any, environment: Any, seed: int) -> Any:
    """Create an episode-local policy, including seed-aware oracle factories."""
    if callable(getattr(factory, "for_episode", None)):
        return factory.for_episode(environment, int(seed))
    if not callable(factory):
        raise TypeError("policy factory must be callable or implement for_episode")
    return factory()


def rollout_policy_with_metrics(
    environment: DailyEnvironment,
    policy: Any,
    *,
    seed: int,
    expected_split: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one episode and preserve daily actions, shocks, and exact metrics."""
    observation = environment.reset(int(seed))
    actual_split = getattr(environment, "profile_split", None)
    if expected_split is not None and actual_split is not None and actual_split != expected_split:
        raise EvaluationContractError(
            f"environment profile_split={actual_split!r}, expected {expected_split!r}"
        )
    recorder = _CoreBlockRecorder(environment)
    recorder.attach()
    total_reward = 0.0
    previous_reward = 0.0
    previous_action: Optional[np.ndarray] = None
    day = 0
    done = False
    daily: List[Dict[str, Any]] = []
    infos: List[Dict[str, Any]] = []
    all_actions: List[List[float]] = []
    try:
        while not done:
            action = _select_action(
                policy,
                np.asarray(observation),
                day=day,
                previous_action=previous_action,
                previous_reward=previous_reward,
            )
            block_start = len(recorder.blocks)
            observation, reward, done, raw_info = environment.step(action)
            info = copy.deepcopy(dict(raw_info))
            new_blocks = recorder.blocks[block_start:]
            if new_blocks:
                day_metrics, day_missing = _metrics_from_block_totals(_sum_blocks(new_blocks))
                day_source = "core_intraday_blocks"
            else:
                day_metrics, day_missing = _metrics_from_daily_infos([info])
                day_source = "environment_daily_summary"
            row = {
                "day": int(info.get("day", day)),
                "date": info.get("date"),
                "weather": info.get("weather"),
                "reward": float(reward),
                "action": [float(value) for value in action],
                "active_shocks": [str(name) for name in info.get("active_shocks", [])],
                "shock_phases": [str(name) for name in info.get("shock_phases", [])],
                "public_shock": _json_safe(info.get("public_shock", {})),
                "block_tariff_multipliers": [
                    float(value) for value in info.get("block_tariff_multipliers", [])
                ],
                "metrics": _json_safe(day_metrics),
                "metric_missing": dict(day_missing),
                "metric_source": day_source,
            }
            daily.append(row)
            infos.append(info)
            all_actions.append(row["action"])
            total_reward += float(reward)
            previous_reward = float(reward)
            previous_action = action.copy()
            day += 1
    finally:
        recorder.detach()

    if recorder.blocks:
        metrics, missing = _metrics_from_block_totals(_sum_blocks(recorder.blocks))
        metric_source = "core_intraday_blocks"
    else:
        metrics, missing = _metrics_from_daily_infos(infos)
        metric_source = "environment_daily_summary"
    metrics["episode_return"] = float(total_reward)
    return {
        "seed": int(seed),
        "return": float(total_reward),
        "days": int(day),
        "actions": all_actions,
        "daily_trace": daily,
        "metrics": _json_safe(metrics),
        "metric_missing": missing,
        "metric_source": metric_source,
    }


def _policy_metric_summaries(
    episodes: Sequence[Mapping[str, Any]],
    *,
    bootstrap: BootstrapConfig,
    policy_name: str,
) -> Dict[str, Any]:
    metric_names = sorted(
        {
            str(metric)
            for episode in episodes
            for metric in dict(episode.get("metrics", {}) or {})
        }
    )
    summaries: Dict[str, Any] = {}
    for metric in metric_names:
        by_seed: Dict[int, float] = {}
        missing: Dict[int, str] = {}
        for episode in episodes:
            seed = int(episode["seed"])
            value = _finite_or_none(dict(episode.get("metrics", {}) or {}).get(metric))
            if value is None:
                reason = dict(episode.get("metric_missing", {}) or {}).get(
                    metric, "metric unavailable for this seed"
                )
                missing[seed] = str(reason)
            else:
                by_seed[seed] = value
        estimate = summarize_seed_values(
            list(by_seed.values()), bootstrap=bootstrap, label=f"{policy_name}|{metric}"
        )
        estimate["seed_values"] = {str(seed): value for seed, value in sorted(by_seed.items())}
        estimate["missing_by_seed"] = {
            str(seed): reason for seed, reason in sorted(missing.items())
        }
        estimate["complete_seed_coverage"] = len(by_seed) == len(episodes)
        estimate["catalog"] = METRIC_CATALOG.get(
            metric,
            {
                "label": metric.replace("_", " "),
                "unit": "model units",
                "scope": "see metric name",
                "direction": "context",
            },
        )
        summaries[metric] = estimate
    return summaries


def paired_policy_comparison(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    bootstrap: BootstrapConfig,
    label: str,
) -> Dict[str, Any]:
    """Compare policies on matched seed-level metric values."""
    left_episodes = {int(row["seed"]): row for row in left.get("episodes", [])}
    right_episodes = {int(row["seed"]): row for row in right.get("episodes", [])}
    if set(left_episodes) != set(right_episodes):
        raise ValueError("paired policy comparison requires identical episode seeds")
    metric_names = sorted(
        {
            str(key)
            for episodes in (left_episodes, right_episodes)
            for episode in episodes.values()
            for key in dict(episode.get("metrics", {}) or {})
        }
    )
    result: Dict[str, Any] = {}
    for metric in metric_names:
        differences: List[float] = []
        paired_seeds: List[int] = []
        missing_seeds: List[int] = []
        for seed in sorted(left_episodes):
            left_value = _finite_or_none(left_episodes[seed].get("metrics", {}).get(metric))
            right_value = _finite_or_none(right_episodes[seed].get("metrics", {}).get(metric))
            if left_value is None or right_value is None:
                missing_seeds.append(seed)
                continue
            paired_seeds.append(seed)
            differences.append(left_value - right_value)
        estimate = summarize_seed_values(
            differences, bootstrap=bootstrap, label=f"paired|{label}|{metric}"
        )
        estimate.update(
            paired_seeds=paired_seeds,
            missing_seeds=missing_seeds,
            contrast="left_minus_right",
            catalog=METRIC_CATALOG.get(metric, {}),
        )
        result[metric] = estimate
    return result


def compute_adaptivity_index(
    learner: Mapping[str, Any],
    static: Mapping[str, Any],
    oracle: Mapping[str, Any],
    *,
    bootstrap: BootstrapConfig,
    metric: str = "episode_return",
    minimum_opportunity: float = 1e-12,
) -> Dict[str, Any]:
    """Estimate ``(learner-static)/(oracle-static)`` on paired seeds.

    The oracle is an explicitly unfair information upper bound.  The index is
    withheld when its mean opportunity is nonpositive or numerically empty.
    """
    episode_maps = [
        {int(row["seed"]): row for row in evaluation.get("episodes", [])}
        for evaluation in (learner, static, oracle)
    ]
    seeds = set(episode_maps[0])
    if any(set(mapping) != seeds for mapping in episode_maps[1:]):
        raise ValueError("adaptivity index requires identical learner/static/oracle seeds")
    rows: List[Tuple[float, float, float]] = []
    used_seeds: List[int] = []
    missing_seeds: List[int] = []
    for seed in sorted(seeds):
        values = [
            _finite_or_none(mapping[seed].get("metrics", {}).get(metric))
            for mapping in episode_maps
        ]
        if any(value is None for value in values):
            missing_seeds.append(seed)
            continue
        rows.append((float(values[0]), float(values[1]), float(values[2])))
        used_seeds.append(seed)
    if not rows:
        return {
            "available": False,
            "metric": metric,
            "reason": "no seeds have all three policy measurements",
            "used_seeds": [],
            "missing_seeds": missing_seeds,
            "oracle_is_fair_policy": False,
        }
    array = np.asarray(rows, dtype=np.float64)
    learner_gain = float(np.mean(array[:, 0] - array[:, 1]))
    oracle_gain = float(np.mean(array[:, 2] - array[:, 1]))
    if oracle_gain <= float(minimum_opportunity):
        return {
            "available": False,
            "metric": metric,
            "learner_minus_static": learner_gain,
            "oracle_minus_static": oracle_gain,
            "reason": "oracle opportunity is nonpositive or too small",
            "used_seeds": used_seeds,
            "missing_seeds": missing_seeds,
            "oracle_is_fair_policy": False,
        }
    estimate = learner_gain / oracle_gain
    result: Dict[str, Any] = {
        "available": True,
        "metric": metric,
        "value": float(estimate),
        "learner_minus_static": learner_gain,
        "oracle_minus_static": oracle_gain,
        "n_seeds": int(array.shape[0]),
        "used_seeds": used_seeds,
        "missing_seeds": missing_seeds,
        "ci_lower": None,
        "ci_upper": None,
        "confidence": bootstrap.confidence,
        "method": "paired_seed_cluster_ratio_bootstrap",
        "oracle_is_fair_policy": False,
        "oracle_information": "latent shock timing/state may be known to the oracle",
    }
    if array.shape[0] > 1:
        rng = np.random.default_rng(
            _stable_seed(bootstrap.seed, f"adaptivity|{metric}")
        )
        indices = rng.integers(
            0, array.shape[0], size=(bootstrap.replicates, array.shape[0])
        )
        sampled = array[indices]
        numerators = np.mean(sampled[:, :, 0] - sampled[:, :, 1], axis=1)
        denominators = np.mean(sampled[:, :, 2] - sampled[:, :, 1], axis=1)
        valid = np.isfinite(denominators) & (denominators > minimum_opportunity)
        ratios = numerators[valid] / denominators[valid]
        result["valid_bootstrap_replicates"] = int(ratios.size)
        if ratios.size >= max(100, bootstrap.replicates // 10):
            alpha = (1.0 - bootstrap.confidence) / 2.0
            result["ci_lower"] = float(np.quantile(ratios, alpha))
            result["ci_upper"] = float(np.quantile(ratios, 1.0 - alpha))
        else:
            result["reason_ci_unavailable"] = (
                "too many bootstrap samples had a nonpositive oracle opportunity"
            )
    else:
        result["reason_ci_unavailable"] = "at least two episode seeds are required"
    return result


def shock_aligned_action_summary(
    evaluations: Mapping[str, Mapping[str, Any]],
    *,
    bootstrap: BootstrapConfig,
    days_before: int = 3,
    days_after: int = 5,
) -> Dict[str, Any]:
    """Align physical controls to each shock's first active day.

    Multiple onsets of the same shock within one episode are averaged before
    uncertainty is calculated, retaining the episode seed as the sampling
    unit.
    """
    if days_before < 0 or days_after < 0:
        raise ValueError("alignment windows must be nonnegative")
    output: Dict[str, Any] = {
        "days_before": int(days_before),
        "days_after": int(days_after),
        "action_names": [
            "base_tariff_multiplier",
            "surge_gain",
            "driver_incentive_multiplier",
        ],
        "policies": {},
    }
    for policy_name, evaluation in evaluations.items():
        # event -> relative day -> seed -> list[action]
        collected: Dict[str, Dict[int, Dict[int, List[np.ndarray]]]] = {}
        onset_rows: List[Dict[str, Any]] = []
        for episode in evaluation.get("episodes", []):
            seed = int(episode["seed"])
            trace = list(episode.get("daily_trace", []))
            active_sets = [set(map(str, row.get("active_shocks", []))) for row in trace]
            for index, active in enumerate(active_sets):
                previous = active_sets[index - 1] if index else set()
                for event in sorted(active - previous):
                    onset_rows.append(
                        {
                            "policy": policy_name,
                            "seed": seed,
                            "shock": event,
                            "onset_day": int(trace[index].get("day", index)),
                            "onset_date": trace[index].get("date"),
                        }
                    )
                    for relative in range(-days_before, days_after + 1):
                        target = index + relative
                        if target < 0 or target >= len(trace):
                            continue
                        action = np.asarray(trace[target].get("action", []), dtype=np.float64)
                        if action.shape != (3,) or not np.all(np.isfinite(action)):
                            continue
                        collected.setdefault(event, {}).setdefault(relative, {}).setdefault(
                            seed, []
                        ).append(action)
        event_output: Dict[str, Any] = {}
        for event, by_relative in sorted(collected.items()):
            rows: List[Dict[str, Any]] = []
            for relative in range(-days_before, days_after + 1):
                seed_actions = by_relative.get(relative, {})
                episode_means = {
                    seed: np.mean(np.stack(actions), axis=0)
                    for seed, actions in seed_actions.items()
                    if actions
                }
                row: Dict[str, Any] = {
                    "relative_day": int(relative),
                    "n_episode_seeds": int(len(episode_means)),
                    "actions": {},
                }
                for index, action_name in enumerate(output["action_names"]):
                    values = [float(vector[index]) for vector in episode_means.values()]
                    row["actions"][action_name] = summarize_seed_values(
                        values,
                        bootstrap=bootstrap,
                        label=f"shock|{policy_name}|{event}|{relative}|{action_name}",
                    )
                rows.append(row)
            event_output[event] = rows
        output["policies"][policy_name] = {
            "events": event_output,
            "onsets": onset_rows,
        }
    return output


def evaluate_policy_suite(
    environment_factory: EnvironmentFactory,
    policy_factories: Mapping[str, Any],
    seeds: Sequence[int],
    *,
    context: EvaluationContext,
    policy_manifest: Mapping[str, Any],
    bootstrap: BootstrapConfig = BootstrapConfig(),
    baseline_policy: Optional[str] = None,
    learner_policy: Optional[str] = None,
    oracle_policy: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate frozen policies with common random numbers across seed clusters."""
    normalized_seeds = [int(seed) for seed in seeds]
    if not normalized_seeds or len(set(normalized_seeds)) != len(normalized_seeds):
        raise ValueError("evaluation seeds must be non-empty and unique")
    if not policy_factories:
        raise ValueError("at least one policy is required")
    if set(policy_factories) != set(policy_manifest):
        raise EvaluationContractError(
            "policy_manifest must name every evaluated policy exactly once"
        )
    claim: Optional[Dict[str, Any]] = None
    if context.split == "test":
        claim = OneShotTestLedger(context.ledger_path).consume(
            run_id=context.run_id,
            seeds=normalized_seeds,
            policy_manifest=policy_manifest,
        )

    evaluations: Dict[str, Any] = {}
    for name, factory in policy_factories.items():
        episodes = []
        for seed in normalized_seeds:
            environment = environment_factory()
            policy = _instantiate_policy(factory, environment, seed)
            episodes.append(
                rollout_policy_with_metrics(
                    environment,
                    policy,
                    seed=seed,
                    expected_split=context.split,
                )
            )
        evaluations[name] = {
            "policy_manifest": _json_safe(policy_manifest[name]),
            "episodes": episodes,
            "metrics": _policy_metric_summaries(
                episodes, bootstrap=bootstrap, policy_name=name
            ),
        }

    comparisons: Dict[str, Any] = {}
    if baseline_policy is not None:
        if baseline_policy not in evaluations:
            raise KeyError(f"unknown baseline policy {baseline_policy!r}")
        for name, evaluation in evaluations.items():
            if name == baseline_policy:
                continue
            comparison_name = f"{name}_minus_{baseline_policy}"
            comparisons[comparison_name] = paired_policy_comparison(
                evaluation,
                evaluations[baseline_policy],
                bootstrap=bootstrap,
                label=comparison_name,
            )

    adaptivity: Optional[Dict[str, Any]] = None
    supplied = [learner_policy, baseline_policy, oracle_policy]
    if learner_policy is not None or oracle_policy is not None:
        if not all(name is not None for name in supplied):
            raise ValueError(
                "learner_policy, baseline_policy, and oracle_policy are all required "
                "to compute adaptivity"
            )
        assert learner_policy is not None and baseline_policy is not None and oracle_policy is not None
        for name in (learner_policy, baseline_policy, oracle_policy):
            if name not in evaluations:
                raise KeyError(f"unknown adaptivity policy {name!r}")
        adaptivity = compute_adaptivity_index(
            evaluations[learner_policy],
            evaluations[baseline_policy],
            evaluations[oracle_policy],
            bootstrap=bootstrap,
        )

    shock_alignment = shock_aligned_action_summary(
        evaluations, bootstrap=bootstrap
    )
    report: Dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "run_id": context.run_id,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "metadata": _json_safe(metadata or {}),
        "evaluation_contract": {
            "profile_split": context.split,
            "purpose": context.purpose,
            "policies_frozen_before_evaluation": bool(context.policies_frozen),
            "selection_performed_on_this_split": bool(context.selection_performed),
            "final_test_one_shot": context.split == "test",
            "final_test_claim": claim,
            "common_random_numbers": True,
            "episode_seeds": normalized_seeds,
            "seed_is_statistical_unit": True,
            "bootstrap": {
                "replicates": bootstrap.replicates,
                "confidence": bootstrap.confidence,
                "seed": bootstrap.seed,
            },
        },
        "metric_catalog": METRIC_CATALOG,
        "policies": evaluations,
        "paired_comparisons": comparisons,
        "adaptivity_index": adaptivity,
        "shock_aligned_actions": shock_alignment,
        "limitations": [
            "The oracle is an information upper bound and is not a deployable fair-policy comparator.",
            "Income-bracket welfare is market-wide because the simulator currently aggregates both platforms by bracket.",
            "Consumer surplus uses modeled reservation fare minus fare paid; it is not compensating variation.",
            "Bootstrap intervals quantify episode-seed Monte Carlo variation, not calibration or structural-model uncertainty.",
        ],
    }
    return _json_safe(report)


def freeze_report(
    report: Mapping[str, Any],
    destination: str | Path,
    *,
    frozen_at_utc: Optional[str] = None,
) -> Path:
    """Write a content-addressed immutable evaluation report."""
    payload = copy.deepcopy(_json_safe(dict(report)))
    if payload.get("schema") != REPORT_SCHEMA:
        raise ValueError(f"expected report schema {REPORT_SCHEMA!r}")
    payload.pop("freeze", None)
    digest = _sha256_json(payload)
    payload["freeze"] = {
        "schema": FROZEN_SCHEMA,
        "frozen_at_utc": frozen_at_utc
        or dt.datetime.now(dt.timezone.utc).isoformat(),
        "content_sha256": digest,
    }
    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path


def load_frozen_report(path: str | Path) -> Dict[str, Any]:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    freeze = dict(payload.get("freeze", {}) or {})
    if freeze.get("schema") != FROZEN_SCHEMA:
        raise ReportIntegrityError("report has no recognized freeze record")
    expected = str(freeze.get("content_sha256", ""))
    content = copy.deepcopy(payload)
    content.pop("freeze", None)
    observed = _sha256_json(content)
    if not expected or observed != expected:
        raise ReportIntegrityError(
            f"frozen report hash mismatch: expected {expected}, observed {observed}"
        )
    if content.get("schema") != REPORT_SCHEMA:
        raise ReportIntegrityError("frozen report payload schema is unsupported")
    return payload


def _metric_summary(
    report: Mapping[str, Any], policy: str, metric: str
) -> Mapping[str, Any]:
    return dict(
        report.get("policies", {})
        .get(policy, {})
        .get("metrics", {})
        .get(metric, {})
        or {}
    )


def _error_bar(summary: Mapping[str, Any]) -> Tuple[float, float, float]:
    mean = _finite_or_none(summary.get("mean"))
    lower = _finite_or_none(summary.get("ci_lower"))
    upper = _finite_or_none(summary.get("ci_upper"))
    if mean is None:
        return float("nan"), 0.0, 0.0
    return (
        mean,
        max(0.0, mean - lower) if lower is not None else 0.0,
        max(0.0, upper - mean) if upper is not None else 0.0,
    )


def _plot_policy_performance(report: Mapping[str, Any], destination: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    policies = list(report.get("policies", {}))
    if not policies:
        return False
    palette = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9"]
    panels = [
        ("episode_return", "Episode return"),
        ("profit_per_request_firm1", "Firm 1 profit / request"),
        ("completed_request_share_firm1", "Firm 1 completed-request share"),
        ("fulfillment_rate_firm1", "Firm 1 fulfillment"),
    ]
    with plt.rc_context(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
        }
    ):
        figure, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
        for axis, (metric, title) in zip(axes.flat, panels):
            plotted = []
            for index, policy in enumerate(policies):
                summary = _metric_summary(report, policy, metric)
                mean, lower_error, upper_error = _error_bar(summary)
                if not math.isfinite(mean):
                    continue
                axis.errorbar(
                    mean,
                    len(plotted),
                    xerr=np.asarray([[lower_error], [upper_error]]),
                    fmt="o",
                    markersize=7,
                    capsize=3,
                    color=palette[index % len(palette)],
                    linewidth=1.5,
                )
                plotted.append(policy)
            axis.set_yticks(range(len(plotted)), [name.replace("_", " ") for name in plotted])
            axis.axvline(0.0, color="#777777", linewidth=0.7, alpha=0.5)
            axis.grid(axis="x", alpha=0.20)
            axis.set_title(title)
            axis.set_xlabel(METRIC_CATALOG.get(metric, {}).get("unit", "model units"))
        figure.suptitle("Paired-seed policy outcomes (bootstrap confidence intervals)", fontsize=13)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=300, bbox_inches="tight")
        plt.close(figure)
    return True


def _plot_stakeholders(report: Mapping[str, Any], destination: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    policies = list(report.get("policies", {}))
    panels = [
        ("mean_wait_minutes_firm1", "Rider wait", "minutes"),
        ("mean_fare_paid_firm1", "Firm 1 fare", "currency / ride"),
        ("driver_earnings_per_online_hour_firm1", "Driver earnings", "currency / online hour"),
        ("consumer_surplus_per_request_market", "Consumer surplus", "currency / request"),
    ]
    if not policies:
        return False
    palette = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9"]
    with plt.rc_context({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}):
        figure, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
        for axis, (metric, title, unit) in zip(axes.flat, panels):
            x = np.arange(len(policies))
            means: List[float] = []
            errors: List[Tuple[float, float]] = []
            available: List[bool] = []
            for policy in policies:
                mean, low, high = _error_bar(_metric_summary(report, policy, metric))
                means.append(0.0 if not math.isfinite(mean) else mean)
                errors.append((low, high))
                available.append(math.isfinite(mean))
            colors = [palette[index % len(palette)] if ok else "#DDDDDD" for index, ok in enumerate(available)]
            axis.bar(x, means, color=colors, alpha=0.90)
            for index, ok in enumerate(available):
                if ok:
                    axis.errorbar(
                        index,
                        means[index],
                        yerr=np.asarray([[errors[index][0]], [errors[index][1]]]),
                        fmt="none",
                        ecolor="black",
                        capsize=3,
                        linewidth=1,
                    )
                else:
                    axis.text(index, 0.0, "NA", ha="center", va="bottom", fontsize=8)
            axis.set_xticks(x, [name.replace("_", " ") for name in policies], rotation=25, ha="right")
            axis.set_title(title, fontweight="bold")
            axis.set_ylabel(unit)
            axis.grid(axis="y", alpha=0.20)
        figure.suptitle("Stakeholder outcomes (gray/NA means not observed)", fontsize=13)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=300, bbox_inches="tight")
        plt.close(figure)
    return True


def _plot_equity(report: Mapping[str, Any], destination: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    policies = list(report.get("policies", {}))
    all_metrics = {
        metric
        for policy in policies
        for metric in report.get("policies", {}).get(policy, {}).get("metrics", {})
    }
    prefix = "equity_served_rate__"
    discovered = {metric[len(prefix):] for metric in all_metrics if metric.startswith(prefix)}
    preferred = ("<50k", "50-100k", "100-200k", "200k+", "unknown")
    brackets = [name for name in preferred if name in discovered]
    brackets.extend(sorted(discovered - set(brackets)))
    if not policies or not brackets:
        return False
    palette = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9"]
    x = np.arange(len(brackets), dtype=float)
    width = min(0.8 / len(policies), 0.25)
    with plt.rc_context({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}):
        figure, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
        for index, policy in enumerate(policies):
            values = []
            lower = []
            upper = []
            for bracket in brackets:
                mean, lo, hi = _error_bar(
                    _metric_summary(report, policy, prefix + bracket)
                )
                values.append(mean)
                lower.append(lo)
                upper.append(hi)
            offset = (index - (len(policies) - 1) / 2.0) * width
            axis.bar(
                x + offset,
                np.nan_to_num(values, nan=0.0),
                width=width,
                label=policy.replace("_", " "),
                color=palette[index % len(palette)],
            )
            finite = np.isfinite(values)
            if np.any(finite):
                axis.errorbar(
                    (x + offset)[finite],
                    np.asarray(values)[finite],
                    yerr=np.asarray([lower, upper])[:, finite],
                    fmt="none",
                    ecolor="black",
                    linewidth=0.8,
                    capsize=2,
                )
        axis.set_xticks(x, brackets)
        axis.set_ylabel("Completed rides / requests")
        axis.set_title("Market-wide service rate by rider income bracket", fontweight="bold")
        axis.grid(axis="y", alpha=0.20)
        axis.legend(frameon=False, ncol=min(3, len(policies)))
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=300, bbox_inches="tight")
        plt.close(figure)
    return True


def _plot_shock_alignment(report: Mapping[str, Any], destination: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    aligned = dict(report.get("shock_aligned_actions", {}) or {})
    policies = dict(aligned.get("policies", {}) or {})
    event_names = sorted(
        {
            event
            for policy in policies.values()
            for event in dict(policy.get("events", {}) or {})
        }
    )
    if not event_names:
        return False
    palette = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9"]
    action_names = list(aligned.get("action_names", []))
    with plt.rc_context({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}):
        figure, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
        line_styles = ("-", "--", ":", "-.")
        for policy_index, (policy_name, policy) in enumerate(policies.items()):
            event_rows = dict(policy.get("events", {}) or {})
            for event_index, event in enumerate(event_names):
                rows = event_rows.get(event, [])
                for action_index, action_name in enumerate(action_names):
                    days: List[int] = []
                    means: List[float] = []
                    lowers: List[float] = []
                    uppers: List[float] = []
                    for row in rows:
                        summary = dict(row.get("actions", {}).get(action_name, {}) or {})
                        mean = _finite_or_none(summary.get("mean"))
                        if mean is None:
                            continue
                        lower = _finite_or_none(summary.get("ci_lower"))
                        upper = _finite_or_none(summary.get("ci_upper"))
                        days.append(int(row["relative_day"]))
                        means.append(mean)
                        lowers.append(mean if lower is None else lower)
                        uppers.append(mean if upper is None else upper)
                    if not days:
                        continue
                    label = policy_name.replace("_", " ")
                    if len(event_names) > 1:
                        label += f" · {event.replace('_', ' ')}"
                    color = palette[policy_index % len(palette)]
                    axes[action_index].plot(
                        days,
                        means,
                        marker="o",
                        markersize=3.5,
                        linewidth=1.5,
                        linestyle=line_styles[event_index % len(line_styles)],
                        label=label,
                        color=color,
                    )
                    axes[action_index].fill_between(
                        days, lowers, uppers, color=color, alpha=0.10
                    )
        titles = ["Base tariff multiplier", "Surge gain", "Driver incentive multiplier"]
        for index, axis in enumerate(axes):
            axis.axvline(0, color="black", linestyle="--", linewidth=0.9)
            axis.set_title(titles[index], fontweight="bold")
            axis.set_xlabel("Days from shock onset")
            axis.grid(alpha=0.20)
        axes[0].set_ylabel("Physical action")
        axes[-1].legend(frameon=False)
        figure.suptitle(
            "Policy actions aligned to physical shock onset\n"
            "shading: episode-seed bootstrap confidence interval",
            fontsize=12,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=300, bbox_inches="tight")
        plt.close(figure)
    return True


def _format_interval(summary: Mapping[str, Any], digits: int = 3) -> str:
    mean = _finite_or_none(summary.get("mean"))
    if mean is None:
        return "NA"
    lower = _finite_or_none(summary.get("ci_lower"))
    upper = _finite_or_none(summary.get("ci_upper"))
    if lower is None or upper is None:
        return f"{mean:.{digits}f} (CI unavailable)"
    confidence = 100.0 * float(summary.get("confidence", 0.95))
    return f"{mean:.{digits}f} [{confidence:.0f}% CI {lower:.{digits}f}, {upper:.{digits}f}]"


def _markdown_report(report: Mapping[str, Any], figures: Sequence[str]) -> str:
    contract = dict(report.get("evaluation_contract", {}) or {})
    policies = list(report.get("policies", {}))
    lines = [
        f"# v28 evaluation: {report.get('run_id', 'unnamed run')}",
        "",
        "## Evaluation contract",
        "",
        f"- Profile split: `{contract.get('profile_split', 'unknown')}`",
        f"- Purpose: `{contract.get('purpose', 'unknown')}`",
        f"- Common random numbers: `{contract.get('common_random_numbers', False)}`",
        f"- Independent statistical unit: episode seed (`n={len(contract.get('episode_seeds', []))}`)",
        f"- Selection performed on this split: `{contract.get('selection_performed_on_this_split', False)}`",
        f"- Final test one-shot: `{contract.get('final_test_one_shot', False)}`",
        "",
        "## Primary outcomes",
        "",
        "| Policy | Episode return | Firm 1 profit/request | Completed-request share | Fulfillment |",
        "|---|---:|---:|---:|---:|",
    ]
    for policy in policies:
        summaries = report["policies"][policy].get("metrics", {})
        lines.append(
            "| "
            + policy.replace("_", " ")
            + " | "
            + " | ".join(
                _format_interval(summaries.get(metric, {}))
                for metric in (
                    "episode_return",
                    "profit_per_request_firm1",
                    "completed_request_share_firm1",
                    "fulfillment_rate_firm1",
                )
            )
            + " |"
        )
    lines.extend(["", "## Paired return contrasts", ""])
    if report.get("paired_comparisons"):
        lines.extend(["| Contrast | Left minus right | Paired seeds |", "|---|---:|---:|"])
        for contrast, metrics in report["paired_comparisons"].items():
            summary = metrics.get("episode_return", {})
            lines.append(
                f"| {contrast.replace('_', ' ')} | {_format_interval(summary)} | {summary.get('n_seeds', 0)} |"
            )
    else:
        lines.append("No baseline contrast was requested.")
    adaptivity = report.get("adaptivity_index")
    lines.extend(["", "## Adaptivity", ""])
    if isinstance(adaptivity, Mapping) and adaptivity.get("available"):
        lower = adaptivity.get("ci_lower")
        upper = adaptivity.get("ci_upper")
        interval = (
            f" [{lower:.3f}, {upper:.3f}]"
            if lower is not None and upper is not None
            else " (CI unavailable)"
        )
        lines.append(
            f"The learner captures **{float(adaptivity['value']):.3f}** of the "
            f"static-to-oracle return opportunity{interval}. The oracle is an unfair "
            "information upper bound, not a deployable policy."
        )
    elif adaptivity:
        lines.append(f"Adaptivity index unavailable: {adaptivity.get('reason', 'unspecified reason')}.")
    else:
        lines.append("Adaptivity was not requested.")
    lines.extend(["", "## Metric availability", ""])
    for policy in policies:
        missing = [
            metric
            for metric, summary in report["policies"][policy].get("metrics", {}).items()
            if not summary.get("available", False) or not summary.get("complete_seed_coverage", True)
        ]
        lines.append(
            f"- **{policy.replace('_', ' ')}:** "
            + ("all recorded metrics available" if not missing else "unavailable/partial: " + ", ".join(missing))
        )
    if figures:
        lines.extend(["", "## Figures", ""])
        for figure in figures:
            lines.append(f"![{Path(figure).stem}]({figure})")
    lines.extend(["", "## Limitations", ""])
    for limitation in report.get("limitations", []):
        lines.append(f"- {limitation}")
    lines.append("")
    freeze = dict(report.get("freeze", {}) or {})
    lines.append(
        f"Frozen content SHA-256: `{freeze.get('content_sha256', 'not frozen')}`"
    )
    return "\n".join(lines) + "\n"


def render_frozen_report(
    frozen_report_path: str | Path,
    output_directory: str | Path,
) -> Dict[str, Any]:
    """Verify and render JSON, Markdown, and publication-quality PNGs."""
    report = load_frozen_report(frozen_report_path)
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    canonical_path = output / "report.json"
    canonical_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    plotters = [
        ("policy_performance.png", _plot_policy_performance),
        ("stakeholder_outcomes.png", _plot_stakeholders),
        ("income_equity.png", _plot_equity),
        ("shock_adaptation.png", _plot_shock_alignment),
    ]
    figures: List[str] = []
    for name, plotter in plotters:
        destination = output / name
        if plotter(report, destination):
            figures.append(name)
    markdown_path = output / "report.md"
    markdown_path.write_text(_markdown_report(report, figures), encoding="utf-8")
    manifest = {
        "frozen_report": str(Path(frozen_report_path).expanduser().resolve()),
        "verified_sha256": report["freeze"]["content_sha256"],
        "json": str(canonical_path),
        "markdown": str(markdown_path),
        "figures": [str(output / name) for name in figures],
    }
    (output / "render_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


__all__ = [
    "BootstrapConfig",
    "ClairvoyantEventPolicyFactory",
    "EvaluationContext",
    "EvaluationContractError",
    "FinalTestAlreadyConsumed",
    "FixedEvaluationPolicy",
    "METRIC_CATALOG",
    "OneShotTestLedger",
    "REPORT_SCHEMA",
    "RecurrentAgentEvaluationPolicy",
    "ReportIntegrityError",
    "compute_adaptivity_index",
    "evaluate_policy_suite",
    "freeze_report",
    "load_frozen_report",
    "paired_policy_comparison",
    "render_frozen_report",
    "rollout_policy_with_metrics",
    "shock_aligned_action_summary",
    "summarize_seed_values",
]
