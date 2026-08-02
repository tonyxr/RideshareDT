from __future__ import annotations

"""Platform-facing POMDP and constrained-control primitives.

The simulator may retain complete ground truth, while a firm receives only
operational telemetry and noisy/delayed competitor quote probes.  Profit and
market-share competitiveness are separate policy objectives with separate
reward mechanisms.  They intentionally share the same observation, action,
PPO, staged-training, and reporting contracts.  Public quote gaps remain
diagnostics rather than a proxy for competitiveness.  Service feasibility,
waiting time, margin, and intervention stability remain genuine CMDP
constraints.
"""

from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Deque, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


SEGMENTS: Tuple[str, ...] = ("0_2", "2_5", "5_10", "10_plus")
POLICY_OBJECTIVES: Tuple[str, ...] = (
    "profit_maximization",
    "competitiveness",
    "balanced",
    "custom",
)


def policy_objective_defaults(objective: str) -> Dict[str, float]:
    """Return the complete reward preset for a named policy objective.

    The profit preset intentionally reproduces the v6 reward weights.  The
    other presets change only the economic objective; they do not swap the
    policy architecture, observation model, action space, or training stages.
    ``custom`` has no preset because its values come directly from the CLI or a
    saved artifact.
    """
    name = str(objective or "profit_maximization").strip().lower()
    if name not in POLICY_OBJECTIVES:
        raise ValueError(
            f"policy objective must be one of {', '.join(POLICY_OBJECTIVES)}"
        )
    presets: Dict[str, Dict[str, float]] = {
        "profit_maximization": {
            "long_term_profit_weight": 1.0,
            "profit_dominance_weight": 0.10,
            "market_share_competitiveness_weight": 0.0,
            "market_share_target_gap": 0.10,
            "market_share_gap_scale": 0.05,
            "market_share_level_weight": 0.25,
            "price_competitiveness_weight": 0.0,
            "target_price_gap": 0.75,
            "price_gap_reward_scale": 1.0,
        },
        "competitiveness": {
            # Profitability is a certification constraint for this objective,
            # never an additive source of reward. The learned target is a
            # durable ten-point conditional choice-share lead, matching the
            # intended market-leadership behavior.
            "long_term_profit_weight": 0.0,
            "profit_dominance_weight": 0.0,
            "market_share_competitiveness_weight": 1.0,
            "market_share_target_gap": 0.10,
            "market_share_gap_scale": 0.05,
            "market_share_level_weight": 0.25,
            "price_competitiveness_weight": 0.0,
            "target_price_gap": 0.75,
            "price_gap_reward_scale": 1.0,
        },
        "balanced": {
            "long_term_profit_weight": 0.70,
            "profit_dominance_weight": 0.20,
            "market_share_competitiveness_weight": 0.50,
            "market_share_target_gap": 0.10,
            "market_share_gap_scale": 0.05,
            "market_share_level_weight": 0.25,
            "price_competitiveness_weight": 0.0,
            "target_price_gap": 0.75,
            "price_gap_reward_scale": 1.0,
        },
        "custom": {},
    }
    return dict(presets[name])


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
    variable cost, and demand volume.  Relative profit is exactly
    ``own_profit_per_request - rival_profit_per_request``; it is not revenue,
    market share, margin, or profit per completed ride. ``asinh`` preserves the
    sign and remains approximately linear around zero while preventing rare
    market shocks from dominating a PPO rollout.

    Named objectives do not infer behavior from weights:

    * ``profit_maximization`` uses only own and relative contribution profit.
    * ``competitiveness`` uses Firm 1's conditional two-firm choice share,
      gated by Firm 1 fulfillment so unserviceable demand cannot earn a high
      score.
    * ``balanced`` deliberately combines the two mechanisms.
    * ``custom`` preserves the legacy expert-weighted combination.

    ``market_share_target_gap`` is the desired Firm 1 minus Firm 2 conditional
    choice-share advantage. It is a certification threshold, not a saturating
    reward target. ``market_share_gap_scale`` and ``market_share_level_weight``
    are retained for old checkpoint compatibility. Quote-gap fields are
    diagnostics only; named objectives never optimize them.
    """

    objective_mode: str = "profit_maximization"
    own_profit_scale: float = 4.0
    profit_advantage_scale: float = 2.5
    # Absolute contribution profit is the objective. Rival-relative profit is
    # deliberately only a small tie-breaker and is gated by own profitability,
    # so PPO cannot earn a high return merely by winning a destructive price war.
    own_profit_weight: float = 1.0
    profit_advantage_weight: float = 0.10
    dominance_quality_scale: float = 4.0
    market_share_competitiveness_weight: float = 0.0
    market_share_target_gap: float = 0.10
    market_share_gap_scale: float = 0.05
    market_share_level_weight: float = 0.25
    # Competitiveness is valuable only when the platform can serve the demand
    # it wins. Below this floor the reward subtracts twice the service
    # shortfall, giving PPO a direct reason to raise/rebalance price instead of
    # continuing an unfulfillable price war.
    market_share_service_floor: float = 0.78
    # Deprecated reward field retained so old artifacts deserialize. Public
    # quote gaps are diagnostics/constraints and never enter named rewards.
    price_competitiveness_weight: float = 0.0
    price_gap_target: float = 0.75
    price_gap_scale: float = 1.0
    intervention_cost_weight: float = 0.0
    reversal_cost_weight: float = 0.0
    minimum_reward: float = -2.0
    maximum_reward: float = 2.0

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
        relevant_weight = {
            "profit_maximization": (
                self.own_profit_weight + self.profit_advantage_weight
            ),
            "competitiveness": self.market_share_competitiveness_weight,
            "balanced": (
                self.own_profit_weight
                + self.profit_advantage_weight
                + self.market_share_competitiveness_weight
            ),
            "custom": (
                self.own_profit_weight
                + self.profit_advantage_weight
                + self.market_share_competitiveness_weight
            ),
        }[objective]
        if relevant_weight <= 0.0:
            raise ValueError(
                f"{objective} requires a positive objective-relevant reward weight"
            )
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
        return (
            float(self.own_profit_weight / total),
            float(self.profit_advantage_weight / total),
            float(self.market_share_competitiveness_weight / total),
        )


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
        own_utility = float(np.arcsinh(own_profit / c.own_profit_scale))
        advantage_utility = float(
            np.tanh((own_profit - rival_profit) / c.profit_advantage_scale)
        )
        profit_quality = float(
            1.0 - np.exp(-max(0.0, own_profit) / c.dominance_quality_scale)
        )
        # Explicit market-share inputs use the conditional two-firm choice
        # definition. The completed-share aliases remain only so old model
        # artifacts and callers can still be evaluated.
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
        share_dominance_utility = float(
            0.5
            * (
                1.0
                + np.tanh(
                    share_gap_error / c.market_share_gap_scale
                )
            )
        )
        # Dense, monotone action credit. Unlike the former tanh-at-target
        # objective, this continues to distinguish every improvement in Firm
        # 1's share. The explicit service shortfall makes an unfulfillable
        # near-monopoly worse than a smaller durable lead; without it, share
        # times fulfillment still overvalued extreme discounting.
        service_shortfall = float(max(
            0.0, c.market_share_service_floor - fulfillment_gate
        ))
        service_shortfall_penalty = float(2.0 * service_shortfall)
        share_utility = float(
            own_share * fulfillment_gate - service_shortfall_penalty
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
            "reward_own_profit_utility": own_utility,
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
            "reward_rival_market_share": rival_share,
            "reward_completed_share": own_share,
            "reward_rival_completed_share": rival_share,
            "reward_market_share_advantage": share_advantage,
            "reward_market_share_target_gap": float(
                c.market_share_target_gap
            ),
            "reward_market_share_gap_error": share_gap_error,
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
        own_component: float,
        advantage_component: float,
        competitiveness_component: float,
    ) -> Dict[str, float]:
        c = self.config
        base = float(
            own_component + advantage_component + competitiveness_component
        )
        raw_reward = float(
            base
            - metrics["reward_intervention_cost"]
            - metrics["reward_reversal_cost"]
        )
        return {
            "reward": float(
                np.clip(raw_reward, c.minimum_reward, c.maximum_reward)
            ),
            "reward_raw": raw_reward,
            "reward_base": base,
            "reward_own_profit_component": float(own_component),
            "reward_profit_advantage_component": float(advantage_component),
            "reward_market_share_competitiveness_component": float(
                competitiveness_component
            ),
            "reward_price_competitiveness_component": 0.0,
            **metrics,
        }


class ProfitMaximizationReward(_ObjectiveRewardBase):
    """V6 economic objective with an explicitly defined relative-profit term."""

    mechanism_name = "profit_maximization_v6"

    def compute(self, **kwargs: float) -> Dict[str, float]:
        metrics = self._metrics(**kwargs)
        c = self.config
        return self._finish(
            metrics,
            own_component=(
                c.own_profit_weight * metrics["reward_own_profit_utility"]
            ),
            advantage_component=(
                c.profit_advantage_weight
                * metrics["reward_profit_quality_gate"]
                * metrics["reward_profit_advantage_utility"]
            ),
            competitiveness_component=0.0,
        )


class MarketShareCompetitivenessReward(_ObjectiveRewardBase):
    """Create a serviceable Firm 1 choice-share lead without profit shaping."""

    mechanism_name = "serviceable_conditional_choice_share_v10"

    def compute(self, **kwargs: float) -> Dict[str, float]:
        metrics = self._metrics(**kwargs)
        # Named competitiveness always has unit scale. Its configured weight is
        # an enablement/compatibility field, not a route for profit terms to
        # leak back into the objective.
        return self._finish(
            metrics,
            own_component=0.0,
            advantage_component=0.0,
            competitiveness_component=metrics["reward_market_share_utility"],
        )


class BalancedPolicyReward(_ObjectiveRewardBase):
    """Deliberate blend of normalized profit and competitiveness mechanisms."""

    mechanism_name = "balanced_profit_and_market_share"

    def compute(self, **kwargs: float) -> Dict[str, float]:
        metrics = self._metrics(**kwargs)
        c = self.config
        own_weight, advantage_weight, gap_weight = c.normalized_weights()
        return self._finish(
            metrics,
            own_component=(
                own_weight * metrics["reward_own_profit_utility"]
            ),
            advantage_component=(
                advantage_weight
                * metrics["reward_profit_quality_gate"]
                * metrics["reward_profit_advantage_utility"]
            ),
            competitiveness_component=(
                gap_weight * metrics["reward_market_share_utility"]
            ),
        )


class CustomPolicyReward(_ObjectiveRewardBase):
    """Backward-compatible unnormalized expert weighting."""

    mechanism_name = "custom_weighted"

    def compute(self, **kwargs: float) -> Dict[str, float]:
        metrics = self._metrics(**kwargs)
        c = self.config
        return self._finish(
            metrics,
            own_component=(
                c.own_profit_weight * metrics["reward_own_profit_utility"]
            ),
            advantage_component=(
                c.profit_advantage_weight
                * metrics["reward_profit_quality_gate"]
                * metrics["reward_profit_advantage_utility"]
            ),
            competitiveness_component=(
                c.market_share_competitiveness_weight
                * metrics["reward_market_share_utility"]
            ),
        )


class PolicyObjectiveReward:
    """Facade selecting one reward mechanism from the configured objective."""

    _mechanisms = {
        "profit_maximization": ProfitMaximizationReward,
        "competitiveness": MarketShareCompetitivenessReward,
        "balanced": BalancedPolicyReward,
        "custom": CustomPolicyReward,
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
    wait_limit_minutes: float = 8.0
    margin_floor: float = 0.10
    intervention_rate_budget: float = 0.35
    reversal_horizon: int = 4
    oscillation_window: int = 12
    multiplier_lr: float = 0.035
    multiplier_max: float = 1.5
    cost_ema_alpha: float = 0.10
    # Only operational feasibility is optimized as a constraint.  Fare-gap
    # channels remain available in diagnostics and the observation model.
    cost_budgets: Tuple[float, ...] = (0.01, 0.01, 0.01)


@dataclass(frozen=True)
class TrainingStage:
    name: str
    start: float
    end: float
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
    """Curriculum for learning control before full strategic competition."""

    def __init__(self, mode: str = "staged") -> None:
        mode = str(mode or "staged").strip().lower()
        if mode not in {"staged", "direct"}:
            raise ValueError("training curriculum must be 'staged' or 'direct'")
        self.mode = mode
        if mode == "direct":
            self.stages = (
                TrainingStage(
                    # A normal training day contains four operational
                    # decisions. Repeated 256-day episodes therefore rehearse
                    # the full 1,000-period deployment horizon, including the
                    # rival's late catch-up, before both firms receive a new
                    # fair opening tariff.
                    "direct", 0.0, 1.0, 1, False, 1, 1.0, 0.05, 0.45, 0.85,
                    256, 8, 0.18,
                ),
            )
        else:
            self.stages = (
                TrainingStage(
                    "foundation", 0.0, 0.18, 1, True, 1, 0.35, 0.18, 1.00, 1.00,
                    48, 1, 0.12,
                ),
                TrainingStage(
                    "robustness", 0.18, 0.42, 2, False, 1, 0.70, 0.12, 0.75, 0.90,
                    64, 4, 0.18,
                ),
                TrainingStage(
                    "competition", 0.42, 0.66, 1, False, 1, 1.00, 0.05, 0.45, 0.70,
                    96, 8, 0.18,
                ),
                TrainingStage(
                    "consolidation", 0.66, 1.000001, 1, False, 1, 1.00, 0.01, 0.00, 0.65,
                    2048, 8, 0.08,
                ),
            )

    def stage_at(self, progress: float) -> TrainingStage:
        p = float(np.clip(progress, 0.0, 1.0))
        for stage in self.stages:
            if stage.start <= p < stage.end:
                return stage
        return self.stages[-1]

    def smooth_controls_at(self, progress: float) -> Dict[str, float]:
        """Return continuous exploration/entropy/LR controls at stage edges.

        Opponent pools and episode horizons remain genuinely staged, while PPO
        controls transition with a smoothstep over the first 20% of each new
        stage. This avoids an instantaneous change in rollout distribution and
        value targets—the main source of artificial reward spikes at curriculum
        boundaries.
        """
        p = float(np.clip(progress, 0.0, 1.0))
        current = self.stage_at(p)
        index = self.stages.index(current)
        if index == 0:
            return {
                "exploration_rate": float(current.exploration_rate),
                "entropy_scale": float(current.entropy_scale),
                "learning_rate_scale": float(current.learning_rate_scale),
            }
        previous = self.stages[index - 1]
        transition_width = max(1e-6, 0.20 * (current.end - current.start))
        fraction = float(np.clip((p - current.start) / transition_width, 0.0, 1.0))
        blend = fraction * fraction * (3.0 - 2.0 * fraction)

        def interpolate(before: float, after: float) -> float:
            return float(before + blend * (after - before))

        return {
            "exploration_rate": interpolate(
                previous.exploration_rate, current.exploration_rate
            ),
            "entropy_scale": interpolate(
                previous.entropy_scale, current.entropy_scale
            ),
            "learning_rate_scale": interpolate(
                previous.learning_rate_scale, current.learning_rate_scale
            ),
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
        "trip_distance_scale",
        "trip_duration_scale",
        "airport_intensity",
        "income_index",
        "price_elasticity",
        "loyalty_intensity",
        "new_rider_share",
        "driver_cost_mile",
        "driver_cost_minute",
        "supply_intensity",
        "weather_sensitivity",
    )
    # Layout (single frame):
    #   0:25   immediate time/tariff/operational measurements
    #   25:31  observable outcome changes caused by market/opponent dynamics
    #   31:43  numeric city/digital-twin context (never a city identity)
    #   43:51  current demand-mix measurements
    #   51:75  public competitor quote history and its changes
    #   75:81  the agent's own previous intervention
    observation_dim: int = 81
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
                list(range(0, 25))
                + list(range(43, 51))
                + list(range(75, 81))
            ),
            "opponent": tuple(
                list(range(25, 31))
                + list(range(51, 75))
            ),
            "city": tuple(range(31, 43)),
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
        weather_code = {"clear": 0.0, "cloudy": 0.33, "rain": 0.66, "snow": 1.0}.get(str(weather).lower(), 0.0)
        time_features = [
            np.sin(2.0 * np.pi * hour_f / 24.0),
            np.cos(2.0 * np.pi * hour_f / 24.0),
            np.sin(2.0 * np.pi * day_f / 7.0),
            np.cos(2.0 * np.pi * day_f / 7.0),
            float(day_f >= 5),
            float((7 <= hour_f < 10) or (16 <= hour_f < 19)),
            float(hour_f < 6 or hour_f >= 22),
            weather_code,
        ]
        coefficient_features = []
        for key in ("base_fare", "per_minute", "per_mile", "booking_fee", "airport_fee"):
            anchor = float(anchor_coefficients.get(key, 1.0))
            coefficient_features.append(float(np.clip((float(own_coefficients.get(key, anchor)) - anchor) / max(abs(anchor), 1e-6), -1.0, 1.0)))

        c = self.config
        t = self.telemetry
        telemetry_features = [
            float(np.clip(t.get("chosen_share_estimate", 0.0), 0.0, 1.0)),
            float(np.clip(t.get("completed_share_estimate", 0.0), 0.0, 1.0)),
            float(np.clip(t.get("revenue_per_request", 0.0) / c.revenue_scale, 0.0, 1.5)),
            float(np.clip(t.get("profit_per_request", 0.0) / c.profit_scale, -1.0, 1.0)),
            float(np.clip(t.get("fulfillment_rate", 1.0), 0.0, 1.0)),
            float(np.clip(t.get("acceptance_rate", 1.0), 0.0, 1.0)),
            float(np.clip(t.get("wait_minutes", 0.0) / c.wait_scale_minutes, 0.0, 1.5)),
            float(np.clip(t.get("driver_pay_per_request", 0.0) / c.driver_pay_scale, 0.0, 1.5)),
            float(np.clip(t.get("idle_driver_share", 0.0), 0.0, 1.0)),
            float(np.clip(t.get("utilization", 0.0), 0.0, 1.0)),
            float(np.clip(t.get("driver_earnings_per_hour", 0.0) / c.driver_earnings_scale, 0.0, 1.5)),
            float(np.clip(t.get("telemetry_age", 0.0) / max(1, c.max_quote_age_steps), 0.0, 1.0)),
        ]
        p = self.previous_telemetry
        trend_features = [
            float(np.clip((t.get("chosen_share_estimate", 0.0) - p.get("chosen_share_estimate", 0.0)) / 0.15, -1.0, 1.0)),
            float(np.clip((t.get("completed_share_estimate", 0.0) - p.get("completed_share_estimate", 0.0)) / 0.15, -1.0, 1.0)),
            float(np.clip((t.get("revenue_per_request", 0.0) - p.get("revenue_per_request", 0.0)) / c.revenue_scale, -1.0, 1.0)),
            float(np.clip((t.get("profit_per_request", 0.0) - p.get("profit_per_request", 0.0)) / c.profit_scale, -1.0, 1.0)),
            float(np.clip((t.get("fulfillment_rate", 1.0) - p.get("fulfillment_rate", 1.0)) / 0.25, -1.0, 1.0)),
            float(np.clip((t.get("wait_minutes", 0.0) - p.get("wait_minutes", 0.0)) / c.wait_scale_minutes, -1.0, 1.0)),
        ]
        numeric_city_context = city_context or {}
        city_features = [
            float(np.clip(numeric_city_context.get(key, 0.0), -1.5, 1.5))
            for key in self.CITY_CONTEXT_KEYS
        ]
        d = self.demand_mix
        demand_features = [
            float(np.clip(d.get("distance_mean", 0.0) / 12.0, 0.0, 1.5)),
            float(np.clip(d.get("distance_std", 0.0) / 8.0, 0.0, 1.5)),
            float(np.clip(d.get("distance_q25", 0.0) / 12.0, 0.0, 1.5)),
            float(np.clip(d.get("distance_q75", 0.0) / 16.0, 0.0, 1.5)),
            float(np.clip(d.get("duration_mean", 0.0) / 45.0, 0.0, 1.5)),
            float(np.clip(d.get("duration_std", 0.0) / 30.0, 0.0, 1.5)),
            float(np.clip(d.get("airport_rate", 0.0), 0.0, 1.0)),
            float(np.clip(d.get("long_trip_share", 0.0), 0.0, 1.0)),
        ]
        probe_features = []
        probe_trend_features = []
        for segment in SEGMENTS:
            probe = self.quote_probes[segment]
            previous_probe = self.previous_quote_probes[segment]
            probe_features.extend([
                float(np.clip((probe.get("gap", 0.0)) / c.gap_scale_dollars, -1.5, 1.5)),
                float(np.clip(probe.get("uncertainty", 1.0) / c.gap_scale_dollars, 0.0, 1.0)),
                float(np.clip(probe.get("age", c.max_quote_age_steps) / max(1, c.max_quote_age_steps), 0.0, 1.0)),
                float(np.clip(probe.get("available", 0.0), 0.0, 1.0)),
            ])
            both_available = float(
                probe.get("available", 0.0) > 0.0
                and previous_probe.get("available", 0.0) > 0.0
            )
            probe_trend_features.extend([
                float(
                    np.clip(
                        (
                            probe.get("gap", 0.0)
                            - previous_probe.get("gap", 0.0)
                        )
                        / c.gap_scale_dollars,
                        -1.5,
                        1.5,
                    )
                    * both_available
                ),
                float(
                    np.clip(
                        probe.get("available", 0.0)
                        - previous_probe.get("available", 0.0),
                        -1.0,
                        1.0,
                    )
                ),
            ])
        action = last_action or {}
        action_features = [
            float(np.clip(action.get("direction", 0.0), -1.0, 1.0)),
            float(np.clip(action.get("target_index", 0.0), 0.0, 1.0)),
            float(np.clip(action.get("magnitude", 0.0) / 2.0, 0.0, 1.0)),
            float(np.clip(action.get("reversal", 0.0), 0.0, 1.0)),
            float(np.clip(action.get("recent_intervention_rate", 0.0), 0.0, 1.0)),
            float(np.clip(action.get("time_since_intervention", 1.0), 0.0, 1.0)),
        ]
        result = np.asarray(
            time_features
            + coefficient_features
            + telemetry_features
            + trend_features
            + city_features
            + demand_features
            + probe_features
            + probe_trend_features
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
            active = [(k, int(v)) for k, v in mapping.items() if int(v) != 0 and k in action_keys]
            if not active:
                rows.append([1.0, 0.0, *([0.0] * 5), 0.0, 1.0, 1.0, *([0.0] * 4), *([0.0] * 4), 1.0, 0.0])
                continue
            all_keys = ("base_fare", "per_minute", "per_mile", "booking_fee", "airport_fee")
            direction_by_key = {key: int(direction) for key, direction in active}
            # Signed lever membership preserves the identity of mixed-direction
            # short/long rebalancing actions.
            key_signature = [float(direction_by_key.get(key, 0)) for key in all_keys]
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
                delta_by_key[key] = float(direction) * float(coefficient_steps[key]) * float(step_scale)
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
        "wait",
        "margin",
        "oscillation",
    )
    # These are true operating-feasibility constraints. Oscillation remains a
    # diagnostic, but is intentionally not a Lagrangian target: reacting to an
    # evolving opponent is part of the desired policy, not a violation.
    names: Tuple[str, ...] = ("fulfillment", "wait", "margin")

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
        costs.update({
            "fulfillment": self._soft_cost(c.fulfillment_floor - float(fulfillment_rate), 0.15),
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
        self.lambdas = np.asarray(snapshot.get("lambdas", self.lambdas), dtype=np.float32)
        self.cost_ema = np.asarray(snapshot.get("cost_ema", self.cost_ema), dtype=np.float32)

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
            "type": "objective_separated_policy_reward_v4",
            "mechanism": PolicyObjectiveReward(
                long_term_reward or LongTermProfitRewardConfig()
            ).mechanism_name,
            "relative_profit_definition": (
                "own contribution profit per incoming request minus rival "
                "contribution profit per incoming request"
            ),
            "market_share_competitiveness_definition": (
                "Firm1 choice share conditional on choosing either firm, "
                "multiplied by Firm1 fulfillment, minus twice any fulfillment "
                "shortfall below the service floor; the two reported market "
                "shares are exact complements and the configured lead is a "
                "certification threshold rather than a reward plateau"
            ),
            "price_gap_metric_definition": (
                "mean(abs((Firm2 public quote - Firm1 public quote) - "
                "target_price_gap)) across quote opportunities; diagnostic only"
            ),
            **asdict(long_term_reward or LongTermProfitRewardConfig()),
        },
        "positive_reward": asdict(reward),
        "constraints": asdict(constraints),
        "active_constraint_names": list(SoftConstraintController.names),
        "diagnostic_constraint_names": list(SoftConstraintController.diagnostic_names),
        "training_curriculum": stages.as_dict(),
    }
