from __future__ import annotations

"""Platform-facing POMDP and robust objective-control primitives.

The simulator may retain complete ground truth, while a firm receives only
operational telemetry and noisy/delayed competitor quote probes.  Profit and
served-market competitiveness are separate policy objectives with separate
reward mechanisms.  They intentionally share the same observation, action,
PPO, staged-training, and reporting contracts.  Public quote gaps remain
diagnostics rather than a proxy for competitiveness.  Service feasibility,
completed-market coverage, waiting time, and margin remain explicit audit
channels for the named objectives.
"""

from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Deque, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


SEGMENTS: Tuple[str, ...] = ("0_2", "2_5", "5_10", "10_plus")
POLICY_OBJECTIVES: Tuple[str, ...] = (
    "profit_maximization",
    "competitiveness",
    "profit_10_competitiveness_90",
    "profit_30_competitiveness_70",
    "profit_50_competitiveness_50",
    "profit_70_competitiveness_30",
    "profit_90_competitiveness_10",
)

OBJECTIVE_PROFIT_SHARES: Dict[str, float] = {
    "profit_maximization": 1.0,
    "competitiveness": 0.0,
    "profit_10_competitiveness_90": 0.10,
    "profit_30_competitiveness_70": 0.30,
    "profit_50_competitiveness_50": 0.50,
    "profit_70_competitiveness_30": 0.70,
    "profit_90_competitiveness_10": 0.90,
}

# Use the smallest non-degenerate economic decomposition for each objective.
# Profit contains only profit terms. Competitiveness retains a small own-profit
# term for sustainable service plus two completed-share terms. Zero-weight
# terms remain visible in the common report schema but cannot send PPO credit.
ENDPOINT_OBJECTIVE_REWARD_WEIGHTS: Dict[str, Dict[str, float]] = {
    "profit_maximization": {
        "own_profit": 0.80,
        "profit_lead": 0.20,
        "own_completed_share": 0.00,
        "completed_share_lead": 0.00,
    },
    "competitiveness": {
        "own_profit": 0.10,
        "profit_lead": 0.00,
        "own_completed_share": 0.20,
        "completed_share_lead": 0.70,
    },
}


def objective_profit_share(objective: str) -> float:
    """Return the preregistered profit share of an objective mixture."""
    name = str(objective or "profit_maximization").strip().lower()
    if name not in OBJECTIVE_PROFIT_SHARES:
        raise ValueError(
            f"policy objective must be one of {', '.join(POLICY_OBJECTIVES)}"
        )
    return float(OBJECTIVE_PROFIT_SHARES[name])


def objective_reward_weights(objective: str) -> Dict[str, float]:
    """Return endpoint or preregistered convex-mixture economic weights."""
    name = str(objective or "profit_maximization").strip().lower()
    alpha = objective_profit_share(name)
    profit = ENDPOINT_OBJECTIVE_REWARD_WEIGHTS["profit_maximization"]
    competitive = ENDPOINT_OBJECTIVE_REWARD_WEIGHTS["competitiveness"]
    return {
        key: float(alpha * profit[key] + (1.0 - alpha) * competitive[key])
        for key in profit
    }


def policy_objective_defaults(objective: str) -> Dict[str, float]:
    """Return the complete, fixed reward preset for a named objective."""
    name = str(objective or "profit_maximization").strip().lower()
    weights = objective_reward_weights(name)
    return {
        "long_term_profit_weight": float(weights["own_profit"]),
        "profit_dominance_weight": float(weights["profit_lead"]),
        "market_share_competitiveness_weight": float(
            weights["completed_share_lead"]
        ),
        "market_share_target_gap": 0.10,
        "market_share_gap_scale": 0.10,
        "market_share_level_weight": float(weights["own_completed_share"]),
        "price_competitiveness_weight": 0.0,
        "target_price_gap": 0.75,
        "price_gap_reward_scale": 1.0,
        "intervention_cost_weight": 0.0,
        "reversal_cost_weight": 0.0,
    }


def conditional_competitive_shares(
    own_choice_share: float,
    rival_choice_share: float,
) -> Tuple[float, float, float]:
    """Return two-firm choice shares that are complementary by construction.

    ``chosen / all incoming requests`` is not a two-firm market share because
    it also contains the outside/no-ride option.  Completed shares are even
    less suitable: their sum is the market's completion coverage and therefore
    also moves with driver supply.  Conditioning on customers who selected
    either platform isolates the competitive choice signal:

    ``own / (own + rival)`` and ``rival / (own + rival)``.

    The third return value is the outside-option share among all requests.
    When neither firm is selected, the competitive prior is an even split.
    """
    own = float(np.clip(np.nan_to_num(own_choice_share, nan=0.0), 0.0, 1.0))
    rival = float(
        np.clip(np.nan_to_num(rival_choice_share, nan=0.0), 0.0, 1.0)
    )
    competitive_total = own + rival
    outside = float(np.clip(1.0 - competitive_total, 0.0, 1.0))
    if competitive_total <= 1e-12:
        return 0.5, 0.5, outside
    own_market = float(own / competitive_total)
    # Compute the complement explicitly so floating-point drift cannot make
    # the two reported market shares sum to anything other than one.
    rival_market = float(1.0 - own_market)
    return own_market, rival_market, outside


@dataclass(frozen=True)
class ObservationConfig:
    telemetry_delay_steps: int = 1
    quote_probe_interval_steps: int = 3
    quote_probe_delay_steps: int = 1
    quote_noise_dollars: float = 0.18
    quote_missing_probability: float = 0.03
    market_share_noise: float = 0.015
    demand_mix_noise: float = 0.02
    max_quote_age_steps: int = 24
    gap_scale_dollars: float = 4.0
    revenue_scale: float = 25.0
    profit_scale: float = 10.0
    wait_scale_minutes: float = 15.0
    driver_pay_scale: float = 20.0
    driver_earnings_scale: float = 80.0


@dataclass(frozen=True)
class PositiveRewardConfig:
    """Configuration for the stationary, positive business objective.

    All four terms are own-platform outcome levels rather than changes,
    competitor-relative bonuses, or action-dependent shaping.  The weights may
    be any non-negative finite values and are normalized before use, making CLI
    tuning intuitive without changing the reward's [0, 1] scale.
    """

    profit_weight: float = 0.38
    revenue_weight: float = 0.22
    completed_demand_weight: float = 0.20
    service_weight: float = 0.20
    revenue_scale: float = 25.0
    profit_scale: float = 10.0
    completed_share_target: float = 0.42
    wait_target_minutes: float = 7.0
    minimum_reward: float = 1e-4

    def __post_init__(self) -> None:
        weights = np.asarray(
            [
                self.profit_weight,
                self.revenue_weight,
                self.completed_demand_weight,
                self.service_weight,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(weights)):
            raise ValueError("positive reward weights must all be finite")
        if np.any(weights < 0.0):
            raise ValueError("positive reward weights must all be non-negative")
        if float(weights.sum()) <= 0.0:
            raise ValueError("at least one positive reward weight must be greater than zero")

    def normalized_weights(self) -> np.ndarray:
        weights = np.asarray(
            [
                self.profit_weight,
                self.revenue_weight,
                self.completed_demand_weight,
                self.service_weight,
            ],
            dtype=float,
        )
        return weights / float(weights.sum())


@dataclass(frozen=True)
class LongTermProfitRewardConfig:
    """Configuration shared by the objective-specific reward mechanisms.

    ``profit_per_request`` is contribution profit divided by *all incoming
    requests*, so it already incorporates price, conversion, fulfillment,
    variable cost, and demand volume. The named profit reward is linear in this
    quantity. A pointwise monotone nonlinear transform is not sufficient here:
    under stochastic outcomes it can change the ordering of policies by
    expected profit. Profit lead is a smaller competitive economic term; it is
    not revenue, market share, margin, or profit per completed ride.

    Both objectives draw from four reported linear economic outcomes: Firm 1
    profit, its profit lead, its completed-request share, and its completed-
    share lead. Zero-weight outcomes do not affect the selected objective.
    Scales are unit conversions, not targets; rewards are never clipped. Quote
    gaps, action changes, service thresholds, and terminal bonuses are
    diagnostics only and cannot alter PPO credit.
    """

    objective_mode: str = "profit_maximization"
    own_profit_scale: float = 2.0
    profit_advantage_scale: float = 1.0
    own_profit_weight: float = 0.0
    profit_advantage_weight: float = 1.0
    dominance_quality_scale: float = 4.0
    market_share_competitiveness_weight: float = 0.0
    market_share_target_gap: float = 0.10
    market_share_gap_scale: float = 0.10
    own_completed_share_scale: float = 0.40
    market_share_level_weight: float = 0.25
    # Retained for report compatibility; service never enters either reward.
    market_share_service_floor: float = 0.78
    # Deprecated reward field retained so old artifacts deserialize. Public
    # quote gaps are diagnostics/constraints and never enter named rewards.
    price_competitiveness_weight: float = 0.0
    price_gap_target: float = 0.75
    price_gap_scale: float = 1.0
    intervention_cost_weight: float = 0.0
    reversal_cost_weight: float = 0.0
    # Retained only for report compatibility; direct rewards are not clipped.
    minimum_reward: float = 0.0
    maximum_reward: float = 1.0

    def __post_init__(self) -> None:
        objective = str(self.objective_mode or "").strip().lower()
        if objective not in POLICY_OBJECTIVES:
            raise ValueError(
                f"objective_mode must be one of {', '.join(POLICY_OBJECTIVES)}"
            )
        values = np.asarray(
            [
                self.own_profit_scale,
                self.profit_advantage_scale,
                self.own_profit_weight,
                self.profit_advantage_weight,
                self.dominance_quality_scale,
                self.market_share_competitiveness_weight,
                self.market_share_target_gap,
                self.market_share_gap_scale,
                self.own_completed_share_scale,
                self.market_share_level_weight,
                self.market_share_service_floor,
                self.price_competitiveness_weight,
                self.price_gap_target,
                self.price_gap_scale,
                self.intervention_cost_weight,
                self.reversal_cost_weight,
                self.minimum_reward,
                self.maximum_reward,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("long-term profit reward configuration must be finite")
        if (
            self.own_profit_scale <= 0.0
            or self.profit_advantage_scale <= 0.0
            or self.dominance_quality_scale <= 0.0
            or self.market_share_gap_scale <= 0.0
            or self.own_completed_share_scale <= 0.0
            or self.price_gap_scale <= 0.0
        ):
            raise ValueError("long-term profit reward scales must be positive")
        if (
            self.own_profit_weight < 0.0
            or self.profit_advantage_weight < 0.0
            or self.market_share_competitiveness_weight < 0.0
            or self.price_competitiveness_weight < 0.0
        ):
            raise ValueError("long-term profit reward weights must be non-negative")
        if not 0.0 <= self.market_share_target_gap <= 1.0:
            raise ValueError("market_share_target_gap must be in [0, 1]")
        if not 0.0 <= self.market_share_level_weight <= 1.0:
            raise ValueError("market_share_level_weight must be in [0, 1]")
        if not 0.0 <= self.market_share_service_floor <= 1.0:
            raise ValueError("market_share_service_floor must be in [0, 1]")
        if self.intervention_cost_weight < 0.0 or self.reversal_cost_weight < 0.0:
            raise ValueError("long-term profit action costs must be non-negative")
        if self.minimum_reward >= self.maximum_reward:
            raise ValueError("minimum_reward must be smaller than maximum_reward")

    def normalized_weights(self) -> Tuple[float, float, float]:
        total = float(
            self.own_profit_weight
            + self.profit_advantage_weight
            + self.market_share_competitiveness_weight
        )
        if total <= 0.0:
            return (0.0, 0.0, 0.0)
        return (
            float(self.own_profit_weight / total),
            float(self.profit_advantage_weight / total),
            float(self.market_share_competitiveness_weight / total),
        )

    def economic_weights(self) -> Dict[str, float]:
        """The four active weights; legacy tunable fields cannot alter them."""
        return objective_reward_weights(self.objective_mode)


class _ObjectiveRewardBase:
    """Shared metric extraction for objective-owned reward implementations."""

    mechanism_name = "base"

    def __init__(self, config: LongTermProfitRewardConfig) -> None:
        self.config = config

    def _metrics(
        self,
        *,
        own_profit_per_request: float,
        rival_profit_per_request: float,
        own_completed_share: float = 0.0,
        rival_completed_share: float = 0.0,
        own_market_share: Optional[float] = None,
        rival_market_share: Optional[float] = None,
        own_fulfillment_rate: float = 1.0,
        price_gap_f2_minus_f1: Optional[float] = None,
        price_gap_abs_error: Optional[float] = None,
        intervention_magnitude: float = 0.0,
        reversal: float = 0.0,
    ) -> Dict[str, float]:
        c = self.config
        own_profit = float(np.nan_to_num(own_profit_per_request, nan=0.0))
        rival_profit = float(np.nan_to_num(rival_profit_per_request, nan=0.0))
        intervention = float(np.clip(intervention_magnitude, 0.0, 1.0))
        reversal_flag = float(np.clip(reversal, 0.0, 1.0))
        # These bounded profit utilities are retained for explicit balanced and
        # custom objectives. Named profit maximization uses the separate exact
        # linear own-profit signal below, so its ordering cannot be changed by
        # clipping or nonlinear transformation. Risk, service, coverage, and
        # margin remain separate audit and feasibility channels.
        profit_gain_utility = float(max(0.0, own_profit) / c.own_profit_scale)
        profit_loss_utility = float(max(0.0, -own_profit) / c.own_profit_scale)
        scaled_profit = float(own_profit / c.own_profit_scale)
        own_utility = float(
            0.5 * (1.0 + scaled_profit / (1.0 + abs(scaled_profit)))
        )
        advantage_utility = float(
            np.tanh((own_profit - rival_profit) / c.profit_advantage_scale)
        )
        profit_quality = float(
            1.0 - np.exp(-max(0.0, own_profit) / c.dominance_quality_scale)
        )
        # Explicit market-share inputs are completed rides divided by all
        # incoming requests. This makes the optimized signal include both
        # customer choice and successful fulfillment; a policy cannot improve
        # it by winning a vanishing conditional market while serving nobody.
        own_share = float(np.clip(
            np.nan_to_num(
                own_completed_share
                if own_market_share is None
                else own_market_share,
                nan=0.0,
            ),
            0.0,
            1.0,
        ))
        rival_share = float(np.clip(
            np.nan_to_num(
                rival_completed_share
                if rival_market_share is None
                else rival_market_share,
                nan=0.0,
            ),
            0.0,
            1.0,
        ))
        fulfillment_gate = float(np.clip(
            np.nan_to_num(own_fulfillment_rate, nan=0.0), 0.0, 1.0
        ))
        share_advantage = float(own_share - rival_share)
        share_gap_error = float(
            share_advantage - c.market_share_target_gap
        )
        # Legacy target-attainment diagnostics are retained for old report
        # schemas. Named competitiveness never consumes them; its active
        # reward is the unclipped linear completed-share lead below.
        market_share_target_level = float(
            np.clip(0.5 * (1.0 + c.market_share_target_gap), 1e-6, 1.0)
        )
        target_gap = float(c.market_share_target_gap)
        if target_gap > 1e-6:
            recovery_scale = float(max(
                target_gap,
                c.market_share_gap_scale,
            ))
            market_share_target_progress = float(min(
                1.0,
                np.exp(
                    np.log(2.0)
                    * (share_advantage - target_gap)
                    / recovery_scale
                ),
            ))
        else:
            # A zero-gap objective regards parity as attainment.  Preserve a
            # smooth recovery signal on the losing side using the configured
            # share scale.
            market_share_target_progress = float(
                1.0
                if share_advantage >= 0.0
                else np.exp(
                    np.log(2.0)
                    * share_advantage
                    / max(c.market_share_gap_scale, 1e-6)
                )
            )
        share_dominance_utility = float(
            0.5
            * (
                1.0
                + np.tanh(
                    share_gap_error / c.market_share_gap_scale
                )
            )
        )
        # Retain the legacy service-adjusted share utility only for
        # backward-compatible report columns. Named objectives do not use it;
        # their four active terms are the linear economic outcomes above.
        service_shortfall = float(max(
            0.0, c.market_share_service_floor - fulfillment_gate
        ))
        # A linear penalty is intentionally steep near the floor.  With a
        # quadratic term, an almost-monopoly at 67% fulfillment could still
        # outrank a durable 75/25 lead at 80% fulfillment because the local
        # penalty was too small.
        service_shortfall_penalty = float(5.0 * service_shortfall)
        target_shortfall = float(max(0.0, -share_gap_error))
        share_utility = float(
            share_advantage - service_shortfall_penalty
        )
        observed_gap = float(
            c.price_gap_target
            if price_gap_f2_minus_f1 is None
            else np.nan_to_num(price_gap_f2_minus_f1, nan=c.price_gap_target)
        )
        signed_gap_error = float(observed_gap - c.price_gap_target)
        # When available, callers pass MAE across all public quote
        # opportunities/segments. Falling back to the absolute signed-mean
        # error preserves compatibility with older checkpoints and tests.
        gap_abs_error = float(
            abs(signed_gap_error)
            if price_gap_abs_error is None
            else max(
                0.0,
                float(
                    np.nan_to_num(
                        price_gap_abs_error,
                        nan=abs(signed_gap_error),
                        posinf=abs(signed_gap_error),
                        neginf=abs(signed_gap_error),
                    )
                ),
            )
        )
        # Dense, bounded, and non-saturating at ordinary dollar errors.  The
        # configured scale has an interpretable meaning: it is the MAE that
        # receives exactly 0.5 utility.
        competitiveness_utility = float(
            c.price_gap_scale / (c.price_gap_scale + gap_abs_error)
        )
        intervention_cost = float(c.intervention_cost_weight * intervention)
        reversal_cost = float(c.reversal_cost_weight * reversal_flag)
        return {
            "reward_own_profit_linear": float(own_profit / c.own_profit_scale),
            "reward_profit_advantage_linear": float(
                (own_profit - rival_profit) / c.profit_advantage_scale
            ),
            "reward_own_profit_utility": own_utility,
            "reward_profit_gain_utility": profit_gain_utility,
            "reward_profit_loss_utility": profit_loss_utility,
            "reward_profit_advantage_utility": advantage_utility,
            "reward_profit_quality_gate": profit_quality,
            "reward_market_share_utility": share_utility,
            "reward_market_share_level_utility": own_share,
            "reward_market_share_dominance_utility": share_dominance_utility,
            "reward_market_share_service_gate": fulfillment_gate,
            "reward_market_share_service_floor": float(
                c.market_share_service_floor
            ),
            "reward_market_share_service_shortfall": service_shortfall,
            "reward_market_share_service_penalty": service_shortfall_penalty,
            "reward_market_share": own_share,
            "reward_market_share_linear": own_share,
            "reward_own_completed_share_linear": float(
                own_share / c.own_completed_share_scale
            ),
            "reward_market_share_advantage_linear": float(
                share_advantage / c.market_share_gap_scale
            ),
            "reward_rival_market_share": rival_share,
            "reward_completed_share": own_share,
            "reward_rival_completed_share": rival_share,
            "reward_market_share_advantage": share_advantage,
            "reward_market_share_target_gap": float(
                c.market_share_target_gap
            ),
            "reward_market_share_target_level": market_share_target_level,
            "reward_market_share_target_progress": (
                market_share_target_progress
            ),
            "reward_market_share_target_achieved": float(
                share_advantage >= c.market_share_target_gap
            ),
            "reward_market_share_gap_error": share_gap_error,
            "reward_market_share_target_shortfall": target_shortfall,
            "reward_price_competitiveness_utility": competitiveness_utility,
            "reward_price_gap": observed_gap,
            "reward_price_gap_target": float(c.price_gap_target),
            "reward_price_gap_error": signed_gap_error,
            "reward_price_gap_abs_error": gap_abs_error,
            "reward_intervention_cost": intervention_cost,
            "reward_reversal_cost": reversal_cost,
            "reward_intervention_magnitude": intervention,
            "reward_reversal_indicator": reversal_flag,
            "reward_profit_per_request": own_profit,
            "reward_rival_profit_per_request": rival_profit,
            "reward_profit_advantage_per_request": own_profit - rival_profit,
            "reward_relative_profit_definition": own_profit - rival_profit,
        }

    def _finish(
        self,
        metrics: Dict[str, float],
        *,
        own_profit_component: float,
        profit_lead_component: float,
        own_share_component: float,
        share_lead_component: float,
    ) -> Dict[str, float]:
        weights = self.config.economic_weights()
        base = float(
            own_profit_component
            + profit_lead_component
            + own_share_component
            + share_lead_component
        )
        return {
            "reward": base,
            "reward_raw": base,
            "reward_base": base,
            "reward_own_profit_component": float(own_profit_component),
            "reward_profit_advantage_component": float(profit_lead_component),
            "reward_market_share_competitiveness_component": float(
                own_share_component + share_lead_component
            ),
            "reward_market_share_level_component": float(own_share_component),
            "reward_market_share_advantage_component": float(
                share_lead_component
            ),
            "reward_price_competitiveness_component": 0.0,
            "reward_applied_action_regularization": 0.0,
            "reward_active_component_count": float(sum(
                float(value) > 0.0 for value in weights.values()
            )),
            "reward_weight_own_profit": float(weights["own_profit"]),
            "reward_weight_profit_lead": float(weights["profit_lead"]),
            "reward_weight_own_completed_share": float(
                weights["own_completed_share"]
            ),
            "reward_weight_completed_share_lead": float(
                weights["completed_share_lead"]
            ),
            **metrics,
        }


class ProfitMaximizationReward(_ObjectiveRewardBase):
    """Profit-only two-term normalized economic reward."""

    mechanism_name = "economic_profit_focus_v22"

    def compute(self, **kwargs: float) -> Dict[str, float]:
        metrics = self._metrics(**kwargs)
        weights = self.config.economic_weights()
        return self._finish(
            metrics,
            own_profit_component=(
                weights["own_profit"] * metrics["reward_own_profit_linear"]
            ),
            profit_lead_component=(
                weights["profit_lead"]
                * metrics["reward_profit_advantage_linear"]
            ),
            own_share_component=(
                weights["own_completed_share"]
                * metrics["reward_own_completed_share_linear"]
            ),
            share_lead_component=(
                weights["completed_share_lead"]
                * metrics["reward_market_share_advantage_linear"]
            ),
        )


class MarketShareCompetitivenessReward(_ObjectiveRewardBase):
    """Completed-share-focused sustainable three-term normalized reward."""

    mechanism_name = "economic_competitiveness_focus_v22"

    def compute(self, **kwargs: float) -> Dict[str, float]:
        metrics = self._metrics(**kwargs)
        weights = self.config.economic_weights()
        return self._finish(
            metrics,
            own_profit_component=(
                weights["own_profit"] * metrics["reward_own_profit_linear"]
            ),
            profit_lead_component=(
                weights["profit_lead"]
                * metrics["reward_profit_advantage_linear"]
            ),
            own_share_component=(
                weights["own_completed_share"]
                * metrics["reward_own_completed_share_linear"]
            ),
            share_lead_component=(
                weights["completed_share_lead"]
                * metrics["reward_market_share_advantage_linear"]
            ),
        )


class MixedEconomicReward(_ObjectiveRewardBase):
    """Preregistered convex mixture of profit and competitiveness rewards."""

    mechanism_name = "economic_profit_competitiveness_mix_v25"

    def compute(self, **kwargs: float) -> Dict[str, float]:
        metrics = self._metrics(**kwargs)
        weights = self.config.economic_weights()
        return self._finish(
            metrics,
            own_profit_component=(
                weights["own_profit"] * metrics["reward_own_profit_linear"]
            ),
            profit_lead_component=(
                weights["profit_lead"]
                * metrics["reward_profit_advantage_linear"]
            ),
            own_share_component=(
                weights["own_completed_share"]
                * metrics["reward_own_completed_share_linear"]
            ),
            share_lead_component=(
                weights["completed_share_lead"]
                * metrics["reward_market_share_advantage_linear"]
            ),
        )


class RobustObjectiveOptimizer:
    """Rank checkpoints by the same reward, emphasizing its sustained tail."""

    mechanism_name = "durable_economic_objective_v22"

    def __init__(
        self,
        *,
        objective: str,
        own_profit_scale: float,
        profit_advantage_target: float,
        market_share_advantage_target: float,
        fulfillment_floor: float,
        completion_coverage_floor: float,
        profit_advantage_scale: float = 1.0,
        market_share_gap_scale: float = 0.10,
        own_completed_share_scale: float = 0.40,
        dominance_rate_floor: float = 0.90,
        tail_weight: float = 0.60,
    ) -> None:
        self.objective = str(objective).strip().lower()
        if self.objective not in POLICY_OBJECTIVES:
            raise ValueError(f"unsupported policy objective: {objective!r}")
        self.own_profit_scale = float(max(1e-6, own_profit_scale))
        self.profit_advantage_scale = float(
            max(1e-6, profit_advantage_scale)
        )
        self.profit_advantage_target = float(
            max(0.0, profit_advantage_target)
        )
        self.market_share_advantage_target = float(
            max(0.0, market_share_advantage_target)
        )
        self.fulfillment_floor = float(
            np.clip(fulfillment_floor, 0.0, 1.0)
        )
        self.completion_coverage_floor = float(
            np.clip(completion_coverage_floor, 0.0, 1.0)
        )
        self.market_share_gap_scale = float(
            max(1e-6, market_share_gap_scale)
        )
        self.own_completed_share_scale = float(
            max(1e-6, own_completed_share_scale)
        )
        self.dominance_rate_floor = float(
            np.clip(dominance_rate_floor, 0.0, 1.0)
        )
        self.tail_weight = float(np.clip(tail_weight, 0.0, 1.0))
        self.weights = objective_reward_weights(self.objective)

    @staticmethod
    def _finite_vector(values: Sequence[float], name: str) -> np.ndarray:
        result = np.asarray(values, dtype=float).reshape(-1)
        if result.size == 0:
            raise ValueError(f"{name} cannot be empty")
        if not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must be finite")
        return result

    @classmethod
    def _unit_vector(cls, values: Sequence[float], name: str) -> np.ndarray:
        result = cls._finite_vector(values, name)
        if np.any((result < 0.0) | (result > 1.0)):
            raise ValueError(f"{name} must be in [0, 1]")
        return result

    def summarize(
        self,
        *,
        own_profit: Sequence[float],
        rival_profit: Sequence[float],
        own_market_share: Sequence[float],
        rival_market_share: Sequence[float],
        fulfillment: Sequence[float],
        completion_coverage: Sequence[float],
        objective_dominance_rate: Optional[Sequence[float]] = None,
        tail_objective_reward: Optional[Sequence[float]] = None,
    ) -> Dict[str, Any]:
        own_profit_v = self._finite_vector(own_profit, "own_profit")
        rival_profit_v = self._finite_vector(rival_profit, "rival_profit")
        own_share_v = self._unit_vector(
            own_market_share, "own_market_share"
        )
        rival_share_v = self._unit_vector(
            rival_market_share, "rival_market_share"
        )
        fulfillment_v = self._unit_vector(fulfillment, "fulfillment")
        coverage_v = self._unit_vector(
            completion_coverage, "completion_coverage"
        )
        widths = {
            vector.size
            for vector in (
                own_profit_v,
                rival_profit_v,
                own_share_v,
                rival_share_v,
                fulfillment_v,
                coverage_v,
            )
        }
        if len(widths) != 1:
            raise ValueError("robust objective metric vectors must align")

        objective_values = (
            self.weights["own_profit"]
            * own_profit_v / self.own_profit_scale
            + self.weights["profit_lead"]
            * (own_profit_v - rival_profit_v) / self.profit_advantage_scale
            + self.weights["own_completed_share"]
            * own_share_v / self.own_completed_share_scale
            + self.weights["completed_share_lead"]
            * (own_share_v - rival_share_v) / self.market_share_gap_scale
        )

        # Preserve the detailed audit fields for report compatibility.  None of
        # them participate in checkpoint eligibility or ordering.
        if objective_dominance_rate is None:
            dominance_rate_v = np.ones_like(objective_values, dtype=float)
        else:
            dominance_rate_v = self._finite_vector(
                objective_dominance_rate, "objective_dominance_rate"
            )
            if dominance_rate_v.size != own_profit_v.size:
                raise ValueError(
                    "objective_dominance_rate must align with robust metrics"
                )
            dominance_rate_v = np.clip(dominance_rate_v, 0.0, 1.0)
        objective_mean = float(np.mean(objective_values))
        objective_worst = float(np.min(objective_values))
        objective_std = float(np.std(objective_values))
        if tail_objective_reward is None:
            tail_values = objective_values
        else:
            tail_values = self._finite_vector(
                tail_objective_reward, "tail_objective_reward"
            )
            if tail_values.size != objective_values.size:
                raise ValueError(
                    "tail_objective_reward must align with robust metrics"
                )
        tail_objective_mean = float(np.mean(tail_values))
        objective_score = float(
            (1.0 - self.tail_weight) * objective_mean
            + self.tail_weight * tail_objective_mean
        )
        feasible = dominance_rate_v >= self.dominance_rate_floor
        normalized_mode_shortfall = np.maximum(
            0.0,
            self.dominance_rate_floor - dominance_rate_v,
        ) / max(self.dominance_rate_floor, 1e-6)
        pass_fraction = float(np.mean(feasible))
        max_shortfall = float(np.max(normalized_mode_shortfall))
        mean_shortfall = float(np.mean(normalized_mode_shortfall))
        eligible = bool(np.all(feasible))
        # Eligibility and dominance rates are business diagnostics. Checkpoint
        # ordering must not introduce a second optimization objective that PPO
        # never saw, so the selection key is exactly the mean normalized reward.
        selection_key = (objective_score,)
        return {
            "optimizer_mechanism": self.mechanism_name,
            "objective_score": objective_score,
            "objective_mean": objective_mean,
            "objective_worst": objective_worst,
            "objective_std": objective_std,
            "tail_objective_mean": tail_objective_mean,
            "tail_weight": self.tail_weight,
            "reward_weights": dict(self.weights),
            "mean_objective_dominance_rate": float(
                np.mean(dominance_rate_v)
            ),
            "worst_objective_dominance_rate": float(
                np.min(dominance_rate_v)
            ),
            "mode_objective_dominance_rate": dominance_rate_v.tolist(),
            "dominance_rate_floor": float(self.dominance_rate_floor),
            "feasible_mode_fraction": pass_fraction,
            "maximum_normalized_shortfall": max_shortfall,
            "mean_normalized_shortfall": mean_shortfall,
            "mode_normalized_shortfall": normalized_mode_shortfall.tolist(),
            "mode_feasible": feasible.astype(float).tolist(),
            "checkpoint_eligible": eligible,
            "selection_key": selection_key,
        }

    @staticmethod
    def candidate_is_better(
        candidate_key: Sequence[float],
        best_key: Optional[Sequence[float]],
        *,
        require_eligible: bool,
    ) -> bool:
        candidate = tuple(float(value) for value in candidate_key)
        if not candidate or not all(np.isfinite(candidate)):
            return False
        del require_eligible
        if best_key is None:
            return True
        best = tuple(float(value) for value in best_key)
        # Equal durability and objective results prefer the later candidate,
        # preserving completed dynamic-stage training.
        return candidate >= best


class PolicyObjectiveReward:
    """Facade selecting one reward mechanism from the configured objective."""

    _mechanisms = {
        "profit_maximization": ProfitMaximizationReward,
        "competitiveness": MarketShareCompetitivenessReward,
        "profit_10_competitiveness_90": MixedEconomicReward,
        "profit_30_competitiveness_70": MixedEconomicReward,
        "profit_50_competitiveness_50": MixedEconomicReward,
        "profit_70_competitiveness_30": MixedEconomicReward,
        "profit_90_competitiveness_10": MixedEconomicReward,
    }

    def __init__(self, config: Optional[LongTermProfitRewardConfig] = None) -> None:
        self.config = config or LongTermProfitRewardConfig()
        objective = str(self.config.objective_mode).strip().lower()
        implementation = self._mechanisms[objective]
        self.implementation = implementation(self.config)
        self.mechanism_name = self.implementation.mechanism_name

    def compute(self, **kwargs: float) -> Dict[str, float]:
        return self.implementation.compute(**kwargs)


class LongTermProfitReward(PolicyObjectiveReward):
    """Compatibility name for the objective-dispatching reward facade."""


# Import compatibility for code that referenced the old class name. The
# implementation is intentionally market-share based.
PriceCompetitivenessReward = MarketShareCompetitivenessReward


@dataclass(frozen=True)
class ConstraintConfig:
    target_gap: float = 0.75
    overall_tolerance: float = 0.45
    # Long-trip gaps are at least as important as short-trip gaps because a
    # small per-mile error compounds with distance.  Short trips get slightly
    # more room for fixed-fee and quote-probe noise.
    segment_tolerances: Tuple[float, float, float, float] = (0.65, 0.60, 0.55, 0.55)
    gap_softness: float = 0.50
    fulfillment_floor: float = 0.78
    # Conditional share can look dominant while nearly every rider chooses the
    # outside option. This floor prevents "winning" a vanishing market and
    # protects completed demand and revenue volume.
    completion_coverage_floor: float = 0.55
    wait_limit_minutes: float = 8.0
    margin_floor: float = 0.10
    intervention_rate_budget: float = 0.35
    reversal_horizon: int = 4
    oscillation_window: int = 12
    multiplier_lr: float = 0.01
    multiplier_max: float = 2.0
    cost_ema_alpha: float = 0.10
    # Only operational feasibility is optimized as a constraint.  Fare-gap
    # channels remain available in diagnostics and the observation model.
    cost_budgets: Tuple[float, ...] = (0.01, 0.01, 0.01, 0.01)


@dataclass(frozen=True)
class TrainingStage:
    name: str
    start: float
    end: float
    opponent_regime: str
    opponent_cadence_multiplier: int
    freeze_opponent: bool
    action_hold_steps: int
    constraint_scale: float
    exploration_rate: float
    entropy_scale: float
    learning_rate_scale: float
    episode_days: int
    opponent_pool_size: int
    tariff_reset_fraction: float


@dataclass
class OperationalClock:
    """Shared decision clock used by training, validation, and evaluation."""

    period: int = 0

    def due(self, interval_steps: int) -> bool:
        return self.period % max(1, int(interval_steps)) == 0

    def advance(self) -> None:
        self.period += 1

    def reset(self) -> None:
        self.period = 0


class TrainingStageScheduler:
    """Environment-only curriculum for progressively harder competition.

    Staging never changes the PPO objective or optimizer.  It changes only the
    source of Firm 2's posted tariff: the city-market reference, a fixed rival,
    and finally the configured dynamic rival. The first two stages establish
    the basics; the final 60% supplies multiple long adaptive-opponent episodes.
    """

    def __init__(
        self,
        mode: str = "staged",
        *,
        stage_fractions: Tuple[float, float, float] = (0.20, 0.20, 0.60),
        dynamic_episode_days: int = 600,
        tariff_reset_fraction: float = 0.15,
    ) -> None:
        requested = str(mode or "staged").strip().lower()
        if requested not in {"staged", "direct"}:
            raise ValueError("training curriculum must be 'staged' or 'direct'")
        fractions = np.asarray(stage_fractions, dtype=float).reshape(-1)
        if (
            fractions.size != 3
            or not np.all(np.isfinite(fractions))
            or np.any(fractions <= 0.0)
            or not np.isclose(float(fractions.sum()), 1.0)
        ):
            raise ValueError(
                "training stage fractions must be three positive values summing to 1"
            )
        episode_days = int(dynamic_episode_days)
        if episode_days <= 0:
            raise ValueError("dynamic_episode_days must be positive")
        reset_fraction = float(tariff_reset_fraction)
        if not np.isfinite(reset_fraction) or not 0.02 <= reset_fraction <= 0.40:
            raise ValueError("tariff_reset_fraction must be in [0.02, 0.40]")
        self.mode = requested
        if requested == "direct":
            self.stages = (
                TrainingStage(
                    "direct", 0.0, 1.000001, "dynamic", 1, False, 1, 0.0,
                    0.0, 1.0, 1.0, episode_days, 1, reset_fraction,
                ),
            )
        else:
            # Decimal CLI fractions such as 0.10 + 0.20 should switch exactly
            # at the documented 0.30 boundary, not one float ulp later.
            market_end = float(round(float(fractions[0]), 12))
            static_end = float(round(float(fractions[0] + fractions[1]), 12))
            self.stages = (
                TrainingStage(
                    "market", 0.0, market_end, "market", 1, True, 1, 0.0,
                    0.0, 1.0, 1.0, 0, 1, reset_fraction,
                ),
                TrainingStage(
                    "static_opponent", market_end, static_end, "static", 1,
                    True, 1, 0.0, 0.0, 1.0, 1.0, 0, 1, reset_fraction,
                ),
                TrainingStage(
                    "dynamic_opponent", static_end, 1.000001, "dynamic", 1,
                    False, 1, 0.0, 0.0, 1.0, 1.0, episode_days, 4,
                    reset_fraction,
                ),
            )

    def stage_at(self, progress: float) -> TrainingStage:
        p = float(np.clip(progress, 0.0, 1.0))
        for stage in self.stages:
            if stage.start <= p < stage.end:
                return stage
        return self.stages[-1]

    def smooth_controls_at(self, progress: float) -> Dict[str, float]:
        """Return constant PPO controls; stages change the environment only."""
        del progress
        return {
            "exploration_rate": 0.0,
            "entropy_scale": 1.0,
            "learning_rate_scale": 1.0,
        }

    def as_dict(self) -> Dict[str, Any]:
        return {"mode": self.mode, "stages": [asdict(stage) for stage in self.stages]}


class PlatformObservationModel:
    """Noisy/delayed observation channel for one platform.

    Ground-truth rider thresholds, competitor coefficients, competitor profit,
    and exact current segment gaps never appear in the observation.  Segment
    gaps become visible only as sampled public-quote estimates with uncertainty,
    age, and missingness.
    """

    TELEMETRY_KEYS: Tuple[str, ...] = (
        "chosen_share_estimate",
        "completed_share_estimate",
        "revenue_per_request",
        "profit_per_request",
        "fulfillment_rate",
        "acceptance_rate",
        "wait_minutes",
        "driver_pay_per_request",
        "idle_driver_share",
        "utilization",
        "driver_earnings_per_hour",
        "telemetry_age",
    )
    DEMAND_KEYS: Tuple[str, ...] = (
        "distance_mean",
        "distance_std",
        "distance_q25",
        "distance_q75",
        "duration_mean",
        "duration_std",
        "airport_rate",
        "long_trip_share",
    )
    CITY_CONTEXT_KEYS: Tuple[str, ...] = (
        "demand_intensity",
    )
    # Layout (single frame):
    #   0:4    cyclical hour and day
    #   4:9    each own tariff coefficient relative to its market anchor
    #   9:14   delayed own operating outcomes
    #   14:18  current demand conditions (including weather-driven intensity)
    #   18:22  public Firm2-minus-Firm1 quote gaps by trip segment
    #   22:24  previous action direction and time since intervention
    # Temporal momentum is supplied by the ordered frame stack, so explicit
    # level duplicates, hand-built trend features, and static city constants
    # are not repeated inside every frame.
    observation_dim: int = 24
    action_feature_dim: int = 20

    @classmethod
    def feature_groups(cls) -> Dict[str, Tuple[int, ...]]:
        """Return semantic observation partitions for hierarchical encoders.

        Opponent features contain only observable consequences and public quote
        history.  The simulator-side opponent mode or controller parameters are
        deliberately absent.  City features are continuous calibrated market
        descriptors, so a policy cannot memorize an integer city identifier.
        """
        return {
            "immediate": tuple(
                list(range(0, 18)) + list(range(22, 24))
            ),
            "opponent": tuple(range(18, 22)),
            "city": tuple(),
        }

    def __init__(self, seed: int, config: Optional[ObservationConfig] = None) -> None:
        self.config = config or ObservationConfig()
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.reset()

    def reset(self) -> None:
        self.step = 0
        self._telemetry_queue: Deque[Dict[str, float]] = deque()
        self._probe_queue: Deque[Dict[str, Dict[str, float]]] = deque()
        self.telemetry = {key: 0.0 for key in self.TELEMETRY_KEYS}
        self.telemetry.update({"fulfillment_rate": 1.0, "acceptance_rate": 1.0, "idle_driver_share": 1.0})
        self.previous_telemetry = dict(self.telemetry)
        self.demand_mix = {key: 0.0 for key in self.DEMAND_KEYS}
        self.demand_mix.update({"distance_mean": 4.0, "distance_q25": 2.0, "distance_q75": 7.0})
        self.quote_probes = {
            segment: {"gap": 0.0, "uncertainty": 1.0, "age": float(self.config.max_quote_age_steps), "available": 0.0}
            for segment in SEGMENTS
        }
        self.previous_quote_probes = {
            segment: dict(values) for segment, values in self.quote_probes.items()
        }
        self._latest_true_gaps = {segment: 0.0 for segment in SEGMENTS}

    def snapshot(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "rng_state": self.rng.bit_generator.state,
            "telemetry_queue": list(self._telemetry_queue),
            "probe_queue": list(self._probe_queue),
            "telemetry": dict(self.telemetry),
            "previous_telemetry": dict(self.previous_telemetry),
            "demand_mix": dict(self.demand_mix),
            "quote_probes": {k: dict(v) for k, v in self.quote_probes.items()},
            "previous_quote_probes": {
                k: dict(v) for k, v in self.previous_quote_probes.items()
            },
            "latest_true_gaps": dict(self._latest_true_gaps),
        }

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        self.step = int(snapshot.get("step", 0))
        self.rng = np.random.default_rng(self.seed)
        if snapshot.get("rng_state") is not None:
            self.rng.bit_generator.state = snapshot["rng_state"]
        self._telemetry_queue = deque(dict(x) for x in snapshot.get("telemetry_queue", []))
        self._probe_queue = deque(
            {k: dict(v) for k, v in x.items()} for x in snapshot.get("probe_queue", [])
        )
        self.telemetry = dict(snapshot.get("telemetry", self.telemetry))
        self.previous_telemetry = dict(snapshot.get("previous_telemetry", self.previous_telemetry))
        self.demand_mix = dict(snapshot.get("demand_mix", self.demand_mix))
        self.quote_probes = {k: dict(v) for k, v in snapshot.get("quote_probes", self.quote_probes).items()}
        self.previous_quote_probes = {
            k: dict(v)
            for k, v in snapshot.get(
                "previous_quote_probes", self.previous_quote_probes
            ).items()
        }
        self._latest_true_gaps = dict(snapshot.get("latest_true_gaps", self._latest_true_gaps))

    @staticmethod
    def _bounded_noise(rng: np.random.Generator, scale: float, bound: float) -> float:
        return float(np.clip(rng.normal(0.0, max(0.0, scale)), -bound, bound))

    def ingest(
        self,
        *,
        own_metrics: Mapping[str, float],
        supply_metrics: Optional[Mapping[str, float]],
        crowd_stats: Mapping[str, float],
        gap_sign: float = 1.0,
    ) -> None:
        """Ingest simulator truth into delayed/noisy platform measurement queues."""
        self.step += 1
        supply = supply_metrics or {}
        share_noise = self._bounded_noise(self.rng, self.config.market_share_noise, 0.08)
        telemetry = {
            "chosen_share_estimate": float(np.clip(float(own_metrics.get("chosen_share", 0.0)) + share_noise, 0.0, 1.0)),
            "completed_share_estimate": float(np.clip(float(own_metrics.get("completed_share", 0.0)) + 0.75 * share_noise, 0.0, 1.0)),
            "revenue_per_request": float(own_metrics.get("revenue_per_request", 0.0)),
            "profit_per_request": float(own_metrics.get("profit_per_request", 0.0)),
            "fulfillment_rate": float(np.clip(own_metrics.get("fulfillment_rate", 1.0), 0.0, 1.0)),
            "acceptance_rate": float(np.clip(own_metrics.get("acceptance_rate", 1.0), 0.0, 1.0)),
            "wait_minutes": float(max(0.0, own_metrics.get("wait_minutes", 0.0))),
            "driver_pay_per_request": float(max(0.0, own_metrics.get("driver_pay_per_request", 0.0))),
            "idle_driver_share": float(np.clip(supply.get("idle_driver_share", 0.0), 0.0, 1.0)),
            "utilization": float(np.clip(supply.get("utilization", 0.0), 0.0, 1.0)),
            "driver_earnings_per_hour": float(max(0.0, supply.get("driver_earnings_per_hour", 0.0))),
            "telemetry_age": 0.0,
        }
        self._telemetry_queue.append(telemetry)
        delay = max(0, int(self.config.telemetry_delay_steps))
        if len(self._telemetry_queue) > delay:
            self.previous_telemetry = dict(self.telemetry)
            self.telemetry = self._telemetry_queue.popleft()
        else:
            self.telemetry["telemetry_age"] = float(self.telemetry.get("telemetry_age", 0.0) + 1.0)

        mix_noise = self.config.demand_mix_noise
        self.demand_mix = {
            "distance_mean": max(0.0, float(crowd_stats.get("distance_mean", 4.0)) * (1.0 + self._bounded_noise(self.rng, mix_noise, 0.10))),
            "distance_std": max(0.0, float(crowd_stats.get("distance_std", 0.0)) * (1.0 + self._bounded_noise(self.rng, mix_noise, 0.10))),
            "distance_q25": max(0.0, float(crowd_stats.get("distance_q25", 2.0))),
            "distance_q75": max(0.0, float(crowd_stats.get("distance_q75", 7.0))),
            "duration_mean": max(0.0, float(crowd_stats.get("duration_mean", 0.0))),
            "duration_std": max(0.0, float(crowd_stats.get("duration_std", 0.0))),
            "airport_rate": float(np.clip(crowd_stats.get("airport_rate", 0.0), 0.0, 1.0)),
            "long_trip_share": float(np.clip(crowd_stats.get("long_trip_share", 0.0), 0.0, 1.0)),
        }

        for probe in self.quote_probes.values():
            probe["age"] = float(min(self.config.max_quote_age_steps, probe.get("age", 0.0) + 1.0))

        if self.step == 1 or self.step % max(1, int(self.config.quote_probe_interval_steps)) == 0:
            sampled: Dict[str, Dict[str, float]] = {}
            for segment in SEGMENTS:
                raw = float(gap_sign) * float(crowd_stats.get(f"distance_bin_{segment}_price_gap_mean", 0.0))
                self._latest_true_gaps[segment] = raw
                available = float(self.rng.random() >= self.config.quote_missing_probability)
                uncertainty = float(self.config.quote_noise_dollars * (1.0 + 0.40 * self.rng.random()))
                estimate = raw + self._bounded_noise(self.rng, uncertainty, 3.0 * uncertainty) if available else 0.0
                sampled[segment] = {
                    "gap": float(estimate),
                    "uncertainty": uncertainty,
                    "age": 0.0,
                    "available": available,
                }
            self._probe_queue.append(sampled)
        probe_delay = max(0, int(self.config.quote_probe_delay_steps))
        if len(self._probe_queue) > probe_delay:
            released = self._probe_queue.popleft()
            for segment, probe in released.items():
                self.previous_quote_probes[segment] = dict(
                    self.quote_probes[segment]
                )
                if probe.get("available", 0.0) > 0.0:
                    self.quote_probes[segment] = dict(probe)
                else:
                    self.quote_probes[segment]["available"] = 0.0

    def quote_snapshot(self) -> Dict[str, Dict[str, float]]:
        return {segment: dict(values) for segment, values in self.quote_probes.items()}

    def publish_public_quote_gaps(
        self,
        gaps: Mapping[str, float],
        *,
        gap_sign: float = 1.0,
    ) -> None:
        """Publish Firm 2's committed leader quote before Firm 1 acts.

        A posted price is public at the point of sale, so this channel has no
        telemetry delay or missingness.  It retains bounded quote noise and
        uncertainty, exposes only representative ride-price gaps, and never
        exposes the leader's coefficients, controller type, or private state.
        Clearing queued pre-commit probes prevents an older delayed quote from
        overwriting the newly posted leader price on the next market ingest.
        """
        self._probe_queue.clear()
        for segment in SEGMENTS:
            raw = float(gap_sign) * float(gaps.get(segment, 0.0))
            self._latest_true_gaps[segment] = raw
            uncertainty = float(
                self.config.quote_noise_dollars
                * (1.0 + 0.40 * self.rng.random())
            )
            estimate = raw + self._bounded_noise(
                self.rng,
                uncertainty,
                3.0 * uncertainty,
            )
            self.previous_quote_probes[segment] = dict(
                self.quote_probes[segment]
            )
            self.quote_probes[segment] = {
                "gap": float(estimate),
                "uncertainty": uncertainty,
                "age": 0.0,
                "available": 1.0,
            }

    def build_observation(
        self,
        *,
        hour: int,
        day_of_week: int,
        weather: str,
        own_coefficients: Mapping[str, float],
        anchor_coefficients: Mapping[str, float],
        last_action: Optional[Mapping[str, float]] = None,
        city_context: Optional[Mapping[str, float]] = None,
    ) -> np.ndarray:
        hour_f = float(int(hour) % 24)
        day_f = float(int(day_of_week) % 7)
        # Weather already enters the observable demand-intensity descriptor.
        # Keeping it a second time would spend a scarce frame slot on a
        # duplicate while hiding which tariff coefficient the actor has
        # already placed at an absolute target.
        del weather
        time_features = [
            np.sin(2.0 * np.pi * hour_f / 24.0),
            np.cos(2.0 * np.pi * hour_f / 24.0),
            np.sin(2.0 * np.pi * day_f / 7.0),
            np.cos(2.0 * np.pi * day_f / 7.0),
        ]
        tariff_keys = (
            "base_fare",
            "per_minute",
            "per_mile",
            "booking_fee",
            "airport_fee",
        )
        tariff_features = []
        for key in tariff_keys:
            anchor = float(anchor_coefficients.get(key, 0.0))
            own = float(own_coefficients.get(key, anchor))
            if abs(anchor) <= 1e-8:
                relative = 0.0 if abs(own) <= 1e-8 else np.sign(own) * 0.5
            else:
                relative = own / anchor - 1.0
            tariff_features.append(float(np.clip(relative, -0.5, 0.5)))

        c = self.config
        t = self.telemetry
        telemetry_features = [
            float(np.clip(t.get("completed_share_estimate", 0.0), 0.0, 1.0)),
            float(np.clip(t.get("profit_per_request", 0.0) / c.profit_scale, -1.0, 1.0)),
            float(np.clip(t.get("fulfillment_rate", 1.0), 0.0, 1.0)),
            float(np.clip(t.get("wait_minutes", 0.0) / c.wait_scale_minutes, 0.0, 1.5)),
            float(np.clip(t.get("utilization", 0.0), 0.0, 1.0)),
        ]
        numeric_city_context = city_context or {}
        d = self.demand_mix
        demand_features = [
            float(np.clip(numeric_city_context.get("demand_intensity", 0.0), -1.5, 1.5)),
            float(np.clip(d.get("distance_mean", 0.0) / 12.0, 0.0, 1.5)),
            float(np.clip(d.get("duration_mean", 0.0) / 45.0, 0.0, 1.5)),
            float(np.clip(d.get("airport_rate", 0.0), 0.0, 1.0)),
        ]
        probe_features = []
        for segment in SEGMENTS:
            probe = self.quote_probes[segment]
            probe_features.append(
                float(np.clip((probe.get("gap", 0.0)) / c.gap_scale_dollars, -1.5, 1.5)),
            )
        action = last_action or {}
        action_features = [
            float(np.clip(action.get("direction", 0.0), -1.0, 1.0)),
            float(np.clip(action.get("time_since_intervention", 1.0), 0.0, 1.0)),
        ]
        result = np.asarray(
            time_features
            + tariff_features
            + telemetry_features
            + demand_features
            + probe_features
            + action_features,
            dtype=np.float32,
        )
        if result.size != self.observation_dim:
            raise RuntimeError(f"platform observation dimension mismatch: {result.size} != {self.observation_dim}")
        return np.nan_to_num(result, nan=0.0, posinf=1.5, neginf=-1.5)

    def build_action_features(
        self,
        *,
        action_steps: Mapping[int, Mapping[str, int]],
        action_keys: Sequence[str],
        own_coefficients: Mapping[str, float],
        anchor_coefficients: Mapping[str, float],
        coefficient_steps: Mapping[str, float],
        coefficient_bounds: Mapping[str, Tuple[float, float]],
        step_scale: float,
        target_gap: float,
        action_target_keys: Optional[Mapping[int, str]] = None,
        action_target_multipliers: Optional[Mapping[int, float]] = None,
    ) -> np.ndarray:
        # Retained for artifact/API compatibility. Fare gaps are observations,
        # not an action objective, so action features must not identify the
        # hand-authored move that minimizes a target gap.
        del target_gap
        representative = {
            "0_2": {"base_fare": 1.0, "per_minute": 8.0, "per_mile": 1.5, "booking_fee": 1.0, "airport_fee": 0.0},
            "2_5": {"base_fare": 1.0, "per_minute": 14.0, "per_mile": 3.5, "booking_fee": 1.0, "airport_fee": 0.05},
            "5_10": {"base_fare": 1.0, "per_minute": 24.0, "per_mile": 7.0, "booking_fee": 1.0, "airport_fee": 0.15},
            "10_plus": {"base_fare": 1.0, "per_minute": 38.0, "per_mile": 13.0, "booking_fee": 1.0, "airport_fee": 0.30},
        }
        rows = []
        for action_id in range(len(action_steps)):
            mapping = dict(action_steps.get(action_id, {}))
            target_key = (
                None
                if action_target_keys is None
                else action_target_keys.get(action_id)
            )
            target_multiplier = (
                None
                if action_target_multipliers is None
                else action_target_multipliers.get(action_id)
            )
            absolute_deltas: Dict[str, float] = {}
            if target_multiplier is not None:
                targeted_keys = (
                    [target_key]
                    if target_key in action_keys
                    else list(action_keys)
                )
                active = []
                for key in targeted_keys:
                    anchor = float(anchor_coefficients[key])
                    current = float(own_coefficients[key])
                    lb, ub = coefficient_bounds[key]
                    target_value = float(np.clip(
                        anchor * float(target_multiplier), lb, ub
                    ))
                    delta = float(target_value - current)
                    absolute_deltas[key] = delta
                    # Keep zero-effect absolute targets distinct from hold.
                    active.append((key, int(np.sign(delta))))
            else:
                active = [
                    (k, int(v)) for k, v in mapping.items()
                    if int(v) != 0 and k in action_keys
                ]
            if not active:
                rows.append([1.0, 0.0, *([0.0] * 5), 0.0, 1.0, 1.0, *([0.0] * 4), *([0.0] * 4), 1.0, 0.0])
                continue
            all_keys = ("base_fare", "per_minute", "per_mile", "booking_fee", "airport_fee")
            direction_by_key = {key: int(direction) for key, direction in active}
            # Absolute target actions need lever identity even when the target
            # is exactly the current market level and realized direction is 0.
            key_signature = [
                float(key in direction_by_key) for key in all_keys
            ]
            global_direction = float(np.clip(np.mean([direction for _, direction in active]), -1.0, 1.0))
            relative_deviations = []
            lower_distances = []
            upper_distances = []
            delta_by_key: Dict[str, float] = {}
            for key, direction in active:
                anchor = float(anchor_coefficients[key])
                current = float(own_coefficients[key])
                lb, ub = coefficient_bounds[key]
                width = max(1e-6, float(ub - lb))
                relative_deviations.append(
                    float(np.clip((current - anchor) / max(abs(anchor), 1e-6), -1.0, 1.0))
                )
                lower_distances.append(float(np.clip((current - lb) / width, 0.0, 1.0)))
                upper_distances.append(float(np.clip((ub - current) / width, 0.0, 1.0)))
                delta_by_key[key] = (
                    absolute_deltas[key]
                    if key in absolute_deltas
                    else float(direction)
                    * float(coefficient_steps[key])
                    * float(step_scale)
                )
            impacts = [
                float(sum(delta_by_key[key] * representative[segment][key] for key, _ in active))
                for segment in SEGMENTS
            ]
            projected_relative_fares = []
            uncertainty_values = []
            for segment, impact in zip(SEGMENTS, impacts):
                probe = self.quote_probes[segment]
                current_fare = float(sum(
                    float(
                        own_coefficients.get(
                            key, anchor_coefficients.get(key, 0.0)
                        )
                    )
                    * representative[segment][key]
                    for key in all_keys
                ))
                anchor_fare = float(sum(
                    float(anchor_coefficients.get(key, 0.0))
                    * representative[segment][key]
                    for key in all_keys
                ))
                projected_relative_fares.append(float(np.clip(
                    (current_fare + impact - anchor_fare)
                    / max(abs(anchor_fare), 1e-6),
                    -1.0,
                    1.0,
                )))
                uncertainty_values.append(float(probe.get("uncertainty", self.config.gap_scale_dollars)))
            rows.append([
                0.0,
                global_direction,
                *key_signature,
                float(np.mean(relative_deviations)),
                float(np.min(lower_distances)),
                float(np.min(upper_distances)),
                *[float(np.clip(v / 20.0, -1.0, 1.0)) for v in impacts],
                *projected_relative_fares,
                float(np.clip(np.mean(uncertainty_values) / self.config.gap_scale_dollars, 0.0, 1.0)),
                float(len(active) / max(1, len(action_keys))),
            ])
        result = np.asarray(rows, dtype=np.float32)
        if result.shape[1] != self.action_feature_dim:
            raise RuntimeError(f"action feature dimension mismatch: {result.shape[1]} != {self.action_feature_dim}")
        return result


class PositiveBusinessReward:
    def __init__(self, config: Optional[PositiveRewardConfig] = None) -> None:
        self.config = config or PositiveRewardConfig()

    @staticmethod
    def _positive_saturation(value: float, scale: float) -> float:
        return float(np.clip(1.0 - np.exp(-max(0.0, float(value)) / max(1e-6, float(scale))), 0.0, 1.0))

    def compute(self, metrics: Mapping[str, float]) -> Dict[str, float]:
        c = self.config
        profit_score = self._positive_saturation(metrics.get("profit_per_request", 0.0), c.profit_scale)
        revenue_score = self._positive_saturation(metrics.get("revenue_per_request", 0.0), c.revenue_scale)
        # A hard cap at the target made all sufficiently popular tariffs look
        # identical to the actor.  Smooth saturation retains useful marginal
        # credit for additional completed demand without letting volume dominate.
        completed_score = self._positive_saturation(
            metrics.get("completed_share", 0.0),
            max(0.05, 0.70 * c.completed_share_target),
        )
        fulfillment = float(np.clip(metrics.get("fulfillment_rate", 1.0), 0.0, 1.0))
        acceptance = float(np.clip(metrics.get("acceptance_rate", 1.0), 0.0, 1.0))
        wait_score = float(np.exp(-max(0.0, metrics.get("wait_minutes", 0.0) - c.wait_target_minutes) / max(c.wait_target_minutes, 1e-6)))
        service_score = float(np.clip(0.55 * fulfillment + 0.25 * acceptance + 0.20 * wait_score, 0.0, 1.0))
        weights = c.normalized_weights()
        reward = float(np.dot(weights, [profit_score, revenue_score, completed_score, service_score]))
        reward = float(np.clip(reward, c.minimum_reward, 1.0))
        return {
            "reward": reward,
            "reward_raw": reward,
            "reward_base": reward,
            "reward_positive_profit": profit_score,
            "reward_positive_revenue": revenue_score,
            "reward_positive_completed_demand": completed_score,
            "reward_positive_service": service_score,
            "reward_weight_profit": float(weights[0]),
            "reward_weight_revenue": float(weights[1]),
            "reward_weight_completed_demand": float(weights[2]),
            "reward_weight_service": float(weights[3]),
        }


class ActionStabilityTracker:
    def __init__(self, config: ConstraintConfig) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.recent_interventions: Deque[int] = deque(maxlen=max(2, self.config.oscillation_window))
        self.last_target = "hold"
        self.last_direction = 0
        self.steps_since_intervention = self.config.reversal_horizon + 1
        self.decision_index = 0
        self.last_direction_by_target: Dict[str, int] = {}
        self.last_decision_by_target: Dict[str, int] = {}
        self.last_reversal = 0.0
        self.last_cost = 0.0

    def snapshot(self) -> Dict[str, Any]:
        return {
            "recent_interventions": list(self.recent_interventions),
            "last_target": self.last_target,
            "last_direction": self.last_direction,
            "steps_since_intervention": self.steps_since_intervention,
            "decision_index": self.decision_index,
            "last_direction_by_target": dict(self.last_direction_by_target),
            "last_decision_by_target": dict(self.last_decision_by_target),
            "last_reversal": self.last_reversal,
            "last_cost": self.last_cost,
        }

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        self.recent_interventions = deque(
            [int(x) for x in snapshot.get("recent_interventions", [])],
            maxlen=max(2, self.config.oscillation_window),
        )
        self.last_target = str(snapshot.get("last_target", "hold"))
        self.last_direction = int(snapshot.get("last_direction", 0))
        self.steps_since_intervention = int(snapshot.get("steps_since_intervention", self.config.reversal_horizon + 1))
        self.decision_index = int(snapshot.get("decision_index", 0))
        self.last_direction_by_target = {
            str(k): int(v)
            for k, v in dict(snapshot.get("last_direction_by_target", {})).items()
        }
        self.last_decision_by_target = {
            str(k): int(v)
            for k, v in dict(snapshot.get("last_decision_by_target", {})).items()
        }
        self.last_reversal = float(snapshot.get("last_reversal", 0.0))
        self.last_cost = float(snapshot.get("last_cost", 0.0))

    @staticmethod
    def _smooth_excess(excess: float, softness: float = 0.25) -> float:
        x = max(0.0, float(excess))
        return float(1.0 - np.exp(-x / max(1e-6, softness)))

    def record(
        self,
        *,
        action_event: bool,
        target: str,
        direction: int,
        directions: Optional[Mapping[str, int]] = None,
    ) -> float:
        self.steps_since_intervention += 1
        if not action_event:
            self.last_reversal = 0.0
            self.recent_interventions.append(0)
            rate = float(np.mean(self.recent_interventions)) if self.recent_interventions else 0.0
            self.last_cost = float(
                0.30 * self._smooth_excess(
                    rate - self.config.intervention_rate_budget,
                    softness=0.20,
                )
            )
            return self.last_cost
        self.decision_index += 1
        coefficient_directions = {
            str(key): int(np.sign(value))
            for key, value in dict(directions or {}).items()
            if int(np.sign(value)) != 0
        }
        if not coefficient_directions and direction != 0 and str(target) != "hold":
            coefficient_directions = {str(target): int(np.sign(direction))}
        non_hold = int(bool(coefficient_directions))
        self.recent_interventions.append(non_hold)
        reversal = 0.0
        for coefficient, current_direction in coefficient_directions.items():
            previous_direction = int(self.last_direction_by_target.get(coefficient, 0))
            previous_decision = int(
                self.last_decision_by_target.get(
                    coefficient, -self.config.reversal_horizon - 1
                )
            )
            if (
                previous_direction == -current_direction
                and self.decision_index - previous_decision <= self.config.reversal_horizon
            ):
                reversal = 1.0
            self.last_direction_by_target[coefficient] = current_direction
            self.last_decision_by_target[coefficient] = self.decision_index
        if non_hold:
            self.last_target = str(target)
            self.last_direction = int(direction)
            self.steps_since_intervention = 0
        rate = float(np.mean(self.recent_interventions)) if self.recent_interventions else 0.0
        rate_cost = self._smooth_excess(rate - self.config.intervention_rate_budget, softness=0.20)
        self.last_reversal = reversal
        self.last_cost = float(np.clip(0.70 * reversal + 0.30 * rate_cost, 0.0, 1.0))
        return self.last_cost

    def features(self, action_keys: Sequence[str]) -> Dict[str, float]:
        target_index = 0.0
        active_targets = [
            target for target in str(self.last_target).split("+") if target in action_keys
        ]
        if active_targets:
            target_index = float(np.mean([
                (action_keys.index(target) + 1) / max(1, len(action_keys))
                for target in active_targets
            ]))
        return {
            "direction": float(self.last_direction),
            "target_index": target_index,
            "magnitude": 0.0,
            "reversal": float(self.last_reversal),
            "recent_intervention_rate": float(np.mean(self.recent_interventions)) if self.recent_interventions else 0.0,
            "time_since_intervention": float(np.clip(self.steps_since_intervention / max(1, self.config.oscillation_window), 0.0, 1.0)),
        }


class SoftConstraintController:
    diagnostic_names: Tuple[str, ...] = (
        "gap_overprice",
        "gap_underprice",
        "gap_0_2",
        "gap_2_5",
        "gap_5_10",
        "gap_10_plus",
        "fulfillment",
        "coverage",
        "wait",
        "margin",
        "oscillation",
    )
    # These are true operating-feasibility constraints. Oscillation remains a
    # diagnostic, but is intentionally not a Lagrangian target: reacting to an
    # evolving opponent is part of the desired policy, not a violation.
    names: Tuple[str, ...] = ("fulfillment", "coverage", "wait", "margin")

    def __init__(self, config: Optional[ConstraintConfig] = None) -> None:
        self.config = config or ConstraintConfig()
        if len(self.config.cost_budgets) != len(self.names):
            raise ValueError("constraint cost budgets must match constraint names")
        self.lambdas = np.zeros(len(self.names), dtype=np.float32)
        self.cost_ema = np.zeros(len(self.names), dtype=np.float32)

    @staticmethod
    def _soft_cost(excess: float, softness: float) -> float:
        x = max(0.0, float(excess))
        return float(1.0 - np.exp(-((x / max(1e-6, softness)) ** 2)))

    def compute(
        self,
        *,
        observer: PlatformObservationModel,
        fulfillment_rate: float,
        wait_minutes: float,
        profit_margin: float,
        oscillation_cost: float,
        completion_coverage: float = 1.0,
    ) -> Dict[str, float]:
        c = self.config
        available = [p for p in observer.quote_probes.values() if p.get("available", 0.0) > 0.0]
        if available:
            weights = np.asarray([1.0 / max(0.05, float(p.get("uncertainty", 1.0))) for p in available])
            aggregate_gap = float(np.average([float(p.get("gap", c.target_gap)) for p in available], weights=weights))
            aggregate_uncertainty = float(np.average([float(p.get("uncertainty", 0.0)) for p in available], weights=weights))
        else:
            aggregate_gap = c.target_gap
            aggregate_uncertainty = 0.0
        lower = c.target_gap - c.overall_tolerance
        upper = c.target_gap + c.overall_tolerance
        costs: Dict[str, float] = {
            "gap_overprice": self._soft_cost((lower - aggregate_gap) + 0.5 * aggregate_uncertainty, c.gap_softness),
            "gap_underprice": self._soft_cost((aggregate_gap - upper) + 0.5 * aggregate_uncertainty, c.gap_softness),
        }
        for segment, tolerance in zip(SEGMENTS, c.segment_tolerances):
            probe = observer.quote_probes[segment]
            if probe.get("available", 0.0) <= 0.0:
                costs[f"gap_{segment}"] = 0.0
                continue
            robust_error = abs(float(probe.get("gap", c.target_gap)) - c.target_gap) + 0.5 * float(probe.get("uncertainty", 0.0))
            costs[f"gap_{segment}"] = self._soft_cost(robust_error - tolerance, c.gap_softness)
        fulfillment_cost = self._soft_cost(
            c.fulfillment_floor - float(fulfillment_rate), 0.15
        )
        coverage_cost = self._soft_cost(
            c.completion_coverage_floor - float(completion_coverage), 0.15
        )
        costs.update({
            # Fulfillment and completed-market coverage have opposite pricing
            # remedies in important states: excess unserved demand calls for
            # less demand pressure, while safe spare capacity and a large
            # outside option call for more.  Combining them with ``max`` made
            # the constraint critic learn one ambiguous service signal.
            "fulfillment": fulfillment_cost,
            "coverage": coverage_cost,
            "wait": self._soft_cost(float(wait_minutes) - c.wait_limit_minutes, 4.0),
            "margin": self._soft_cost(c.margin_floor - float(profit_margin), 0.08),
            "oscillation": float(np.clip(oscillation_cost, 0.0, 1.0)),
        })
        return costs

    def vector(self, costs: Mapping[str, float]) -> np.ndarray:
        return np.asarray([float(np.clip(costs.get(name, 0.0), 0.0, 1.0)) for name in self.names], dtype=np.float32)

    def update(self, costs: Mapping[str, float], scale: float = 1.0) -> None:
        values = self.vector(costs)
        alpha = float(np.clip(self.config.cost_ema_alpha, 0.0, 1.0))
        self.cost_ema = (1.0 - alpha) * self.cost_ema + alpha * values
        budgets = np.asarray(self.config.cost_budgets, dtype=np.float32)
        self.lambdas = np.clip(
            self.lambdas + float(self.config.multiplier_lr) * float(max(0.0, scale)) * (self.cost_ema - budgets),
            0.0,
            float(self.config.multiplier_max),
        ).astype(np.float32)

    def snapshot(self) -> Dict[str, Any]:
        return {"lambdas": self.lambdas.tolist(), "cost_ema": self.cost_ema.tolist()}

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        lambdas = np.asarray(
            snapshot.get("lambdas", self.lambdas), dtype=np.float32
        ).reshape(-1)
        cost_ema = np.asarray(
            snapshot.get("cost_ema", self.cost_ema), dtype=np.float32
        ).reshape(-1)
        expected = len(self.names)
        if lambdas.size != expected or cost_ema.size != expected:
            raise ValueError(
                "soft-constraint snapshot width does not match active "
                f"constraints: lambdas={lambdas.size}, cost_ema={cost_ema.size}, "
                f"expected={expected} ({', '.join(self.names)})"
            )
        self.lambdas = np.clip(
            lambdas, 0.0, float(self.config.multiplier_max)
        ).astype(np.float32)
        self.cost_ema = np.clip(cost_ema, 0.0, 1.0).astype(np.float32)

    def diagnostics(self, costs: Mapping[str, float]) -> Dict[str, float]:
        result: Dict[str, float] = {}
        active_index = {name: index for index, name in enumerate(self.names)}
        for name in self.diagnostic_names:
            result[f"constraint_cost_{name}"] = float(costs.get(name, 0.0))
            if name in active_index:
                index = active_index[name]
                result[f"constraint_lambda_{name}"] = float(self.lambdas[index])
                result[f"constraint_cost_ema_{name}"] = float(self.cost_ema[index])
                result[f"constraint_budget_{name}"] = float(self.config.cost_budgets[index])
                result[f"constraint_active_{name}"] = 1.0
            else:
                result[f"constraint_lambda_{name}"] = 0.0
                result[f"constraint_cost_ema_{name}"] = 0.0
                result[f"constraint_budget_{name}"] = 0.0
                result[f"constraint_active_{name}"] = 0.0
        return result


def config_payload(
    observation: ObservationConfig,
    reward: PositiveRewardConfig,
    constraints: ConstraintConfig,
    stages: TrainingStageScheduler,
    long_term_reward: Optional[LongTermProfitRewardConfig] = None,
) -> Dict[str, Any]:
    return {
        "observation": asdict(observation),
        "active_reward": {
            "type": "objective_separated_policy_reward_v8",
            "mechanism": PolicyObjectiveReward(
                long_term_reward or LongTermProfitRewardConfig()
            ).mechanism_name,
            "relative_profit_definition": (
                "own contribution profit per incoming request minus rival "
                "contribution profit per incoming request"
            ),
            "market_share_competitiveness_definition": (
                "linear Firm1-minus-Firm2 completed rides per incoming request "
                "divided by 0.10 with no target cap; conditional choice share, "
                "profit, public quote gaps, service, wait, and margin are "
                "diagnostics"
            ),
            "robust_checkpoint_optimizer": (
                RobustObjectiveOptimizer.mechanism_name
            ),
            "price_gap_metric_definition": (
                "mean(abs((Firm2 public quote - Firm1 public quote) - "
                "target_price_gap)) across quote opportunities; diagnostic only"
            ),
            **asdict(long_term_reward or LongTermProfitRewardConfig()),
        },
        "positive_reward": asdict(reward),
        "constraints": asdict(constraints),
        "active_constraint_names": [],
        "diagnostic_constraint_names": list(SoftConstraintController.diagnostic_names),
        "training_curriculum": stages.as_dict(),
    }
