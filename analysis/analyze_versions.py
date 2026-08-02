#!/usr/bin/env python3
"""Normalize RideResponse experiment artifacts into comparable run metrics."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
OUTPUT = ROOT / "analysis" / "generated_version_metrics.json"


PRIMARY_PREFIXES = [
    "policy_dynamic_v4_seed4303",
    "policy_long_profit_seed4303",
    "policy_profit_v5_seed4303",
    "policy_profit_v6_seed4303",
    "policy_profit_v6_1_seed4303",
    "policy_profit_v6_2_seed4303",
    "policy_profit_v7_seed4303_top2",
    "policy_profit_v7_1_seed4303_top2",
    "policy_profit_v7_2_seed4500_top2",
    "dynamic_v8_seed4500_economic",
    "dynamic_v8_seed4500_gate",
    "dynamic_v8_seed4500_gate2",
    "dynamic_v8_seed4500_gate3",
    "dynamic_v8_seed4500_longgate",
    "dynamic_v8_seed9127_direct",
    "dynamic_v8_seed9127_direct_symmetric",
    "dynamic_v8_seed9127_final",
    "dynamic_v8_seed9127_final_eval1000",
    "dynamic_v8_seed9127_gate",
    "dynamic_v8_seed9127_mi50",
    "dynamic_v8_seed27183_gate",
    "dynamic_v9f_seed4500",
    "dynamic_v13_seed4500",
    "policy_profit_dynamic_v18_seed4500",
    "policy_profit_dynamic_v19_seed4500",
]


CONFIG_FIELDS = [
    "firm2_mode",
    "training_curriculum",
    "train_timesteps",
    "eval_timesteps",
    "train_steps_per_day",
    "train_customers",
    "eval_customers",
    "ppo_update_interval_days",
    "ppo_min_rollout_transitions",
    "ppo_batch_size",
    "ppo_update_epochs",
    "ppo_gamma",
    "ppo_gae_lambda",
    "state_frame_stack",
    "state_action_mi_weight",
    "long_term_profit_weight",
    "profit_dominance_weight",
    "long_term_profit_scale",
    "long_term_profit_advantage_scale",
    "intervention_cost_weight",
    "reversal_cost_weight",
    "enable_constrained_reward",
    "disable_constrained_reward",
    "constraint_penalty_scale",
    "constraint_curriculum_start_scale",
    "constraint_curriculum_mid_scale",
    "constraint_curriculum_end_scale",
    "checkpoint_validation_horizon",
    "checkpoint_validation_interval_days",
    "checkpoint_validation_customers",
    "eval_policy_mode",
    "eval_guardrail_mode",
    "choice_mode",
    "firm1_action_interval_steps",
    "firm2_action_interval_days",
    "pool",
    "threshold_profile_source",
]


def numeric(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def finite(values: Iterable[Any]) -> list[float]:
    return [number for value in values if math.isfinite(number := numeric(value))]


def mean(values: Iterable[Any]) -> float:
    nums = finite(values)
    return fmean(nums) if nums else math.nan


def std(values: Iterable[Any]) -> float:
    nums = finite(values)
    return pstdev(nums) if len(nums) > 1 else (0.0 if nums else math.nan)


def fraction(values: Iterable[Any], predicate) -> float:
    nums = finite(values)
    return fmean([1.0 if predicate(value) else 0.0 for value in nums]) if nums else math.nan


def slope(values: Iterable[Any]) -> float:
    nums = finite(values)
    if len(nums) < 2:
        return math.nan
    n = len(nums)
    x_mean = (n - 1) / 2.0
    y_mean = fmean(nums)
    denominator = sum((index - x_mean) ** 2 for index in range(n))
    if denominator <= 0:
        return math.nan
    return sum(
        (index - x_mean) * (value - y_mean)
        for index, value in enumerate(nums)
    ) / denominator


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def tail(rows: list[dict[str, str]], fraction_size: float = 0.25) -> list[dict[str, str]]:
    if not rows:
        return []
    count = max(20, int(math.ceil(len(rows) * fraction_size)))
    return rows[-min(len(rows), count):]


def early_settled(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    if not rows:
        return []
    settled = rows[min(len(rows) - 1, len(rows) // 10):]
    return settled[: min(len(settled), max(20, len(rows) // 4))]


def column(rows: list[dict[str, str]], name: str) -> list[Any]:
    return [row.get(name) for row in rows]


def action_switch_rate(rows: list[dict[str, str]]) -> float:
    actions = [row.get("action") for row in rows if row.get("action") not in (None, "")]
    if len(actions) < 2:
        return math.nan
    return fmean([1.0 if current != previous else 0.0 for previous, current in zip(actions, actions[1:])])


def coefficient_change_rate(rows: list[dict[str, str]]) -> float:
    keys = ["base_fare", "per_minute", "per_mile", "booking_fee", "airport_fee"]
    if len(rows) < 2:
        return math.nan
    changes = 0
    comparisons = 0
    for previous, current in zip(rows, rows[1:]):
        deltas = []
        for key in keys:
            before = numeric(previous.get(f"firm1_{key}"))
            after = numeric(current.get(f"firm1_{key}"))
            if math.isfinite(before) and math.isfinite(after):
                deltas.append(abs(after - before))
        if deltas:
            comparisons += 1
            changes += int(any(delta > 1e-9 for delta in deltas))
    return changes / comparisons if comparisons else math.nan


def fare_equivalent_change_rate(rows: list[dict[str, str]]) -> float:
    exposures = {
        "base_fare": 1.0,
        "per_minute": 15.0,
        "per_mile": 4.0,
        "booking_fee": 1.0,
        "airport_fee": 0.05,
    }
    if len(rows) < 2:
        return math.nan
    changes = 0
    comparisons = 0
    for previous, current in zip(rows, rows[1:]):
        total = 0.0
        found = False
        for key, exposure in exposures.items():
            before = numeric(previous.get(f"firm1_{key}"))
            after = numeric(current.get(f"firm1_{key}"))
            if math.isfinite(before) and math.isfinite(after):
                total += abs(after - before) * exposure
                found = True
        if found:
            comparisons += 1
            changes += int(total >= 0.05)
    return changes / comparisons if comparisons else math.nan


def endpoint(rows: list[dict[str, str]], name: str) -> float:
    if not rows:
        return math.nan
    return numeric(rows[-1].get(name))


def summarize_evaluation(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        return {}
    late = tail(rows)
    early = early_settled(rows)
    reward_early = mean(column(early, "reward"))
    reward_late = mean(column(late, "reward"))
    actions = [row.get("action") for row in late if row.get("action") not in (None, "")]
    result = {
        "days": len(rows),
        "tail_days": len(late),
        "reward_mean": mean(column(rows, "reward")),
        "reward_tail_mean": reward_late,
        "reward_tail_std": std(column(late, "reward")),
        "reward_tail_slope": slope(column(late, "reward")),
        "reward_retention": (
            reward_late / reward_early
            if math.isfinite(reward_early) and abs(reward_early) > 1e-9
            else math.nan
        ),
        "rl_completed_share_tail": mean(column(late, "rl_completed_share")),
        "rival_completed_share_tail": mean(column(late, "heuristic_completed_share")),
        "completed_share_advantage_tail": mean(
            numeric(row.get("rl_completed_share")) - numeric(row.get("heuristic_completed_share"))
            for row in late
        ),
        "rl_revenue_tail": mean(column(late, "rl_revenue")),
        "rival_revenue_tail": mean(column(late, "heuristic_revenue")),
        "revenue_advantage_tail": mean(
            numeric(row.get("rl_revenue")) - numeric(row.get("heuristic_revenue"))
            for row in late
        ),
        "rl_profit_tail": mean(column(late, "rl_profit")),
        "rival_profit_tail": mean(column(late, "heuristic_profit")),
        "profit_advantage_tail": mean(
            numeric(row.get("rl_profit")) - numeric(row.get("heuristic_profit"))
            for row in late
        ),
        "price_gap_tail": mean(column(late, "price_gap_f2_minus_f1")),
        "price_gap_abs_error_tail": mean(column(late, "price_gap_abs_error")),
        "gap_violation_025_tail": mean(column(late, "gap_violation_025")),
        "gap_violation_050_tail": mean(column(late, "gap_violation_050")),
        "fulfillment_tail": mean(column(late, "rl_fulfillment_rate")),
        "wait_minutes_tail": mean(column(late, "rl_avg_wait_minutes")),
        "rival_collapse_share_rate_tail": fraction(
            column(late, "heuristic_completed_share"), lambda value: value <= 0.01
        ),
        "rival_collapse_profit_rate_tail": fraction(
            column(late, "heuristic_profit"), lambda value: value <= 0.05
        ),
        "action_diversity_tail": len(set(actions)),
        "action_switch_rate_tail": action_switch_rate(late),
        "coefficient_change_rate_tail": coefficient_change_rate(late),
        "fare_equivalent_change_rate_tail": fare_equivalent_change_rate(late),
        "zero_effect_rate_tail": fraction(column(late, "action_zero_effect"), lambda value: value >= 0.5),
        "saturated_action_rate_tail": fraction(column(late, "action_saturated"), lambda value: value >= 0.5),
        "policy_entropy_tail": mean(column(late, "policy_entropy")),
        "policy_action_margin_tail": mean(column(late, "policy_action_margin")),
    }
    for firm in ("firm1", "firm2"):
        for coefficient in ("base_fare", "per_minute", "per_mile", "booking_fee", "airport_fee"):
            result[f"{firm}_{coefficient}_end"] = endpoint(rows, f"{firm}_{coefficient}")
    return result


def summarize_training(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        return {}
    late = tail(rows)
    result = {
        "batches": len(rows),
        "reward_tail_mean": mean(column(late, "avg_reward")),
        "reward_tail_std": std(column(late, "avg_reward")),
        "share_tail": mean(column(late, "avg_completed_share")),
        "rival_share_tail": mean(column(late, "avg_firm2_completed_share")),
        "profit_tail": mean(column(late, "avg_profit_per_request")),
        "rival_profit_tail": mean(column(late, "avg_firm2_profit_per_request")),
        "profit_advantage_tail": mean(column(late, "avg_profit_advantage_per_request")),
        "price_gap_tail": mean(column(late, "avg_price_gap_f2_minus_f1")),
        "price_gap_abs_error_tail": mean(column(late, "avg_price_gap_abs_error")),
        "ppo_approx_kl_tail": mean(column(late, "ppo_approx_kl")),
        "ppo_clipfrac_tail": mean(column(late, "ppo_clipfrac")),
        "ppo_entropy_fraction_tail": mean(column(late, "ppo_policy_entropy_fraction")),
        "ppo_explained_variance_tail": mean(column(late, "ppo_explained_variance")),
        "ppo_action_diversity_tail": mean(column(late, "ppo_rollout_action_diversity")),
        "ppo_state_action_sensitivity_tail": mean(column(late, "ppo_state_action_sensitivity")),
        "ppo_state_action_mi_tail": mean(column(late, "ppo_state_action_mi")),
        "validation_score_best": max(finite(column(rows, "validation_score")), default=math.nan),
        "validation_score_last": next(
            (
                value
                for row in reversed(rows)
                if math.isfinite(value := numeric(row.get("validation_score")))
            ),
            math.nan,
        ),
        "validation_worst_profit_advantage_last": next(
            (
                value
                for row in reversed(rows)
                if math.isfinite(
                    value := numeric(row.get("validation_worst_profit_advantage"))
                )
            ),
            math.nan,
        ),
        "validation_profit_win_rate_last": next(
            (
                value
                for row in reversed(rows)
                if math.isfinite(value := numeric(row.get("validation_profit_win_rate")))
            ),
            math.nan,
        ),
        "validation_policy_dynamicity_last": next(
            (
                value
                for row in reversed(rows)
                if math.isfinite(value := numeric(row.get("validation_policy_dynamicity")))
            ),
            math.nan,
        ),
    }
    return result


def summarize_config(config: dict[str, Any]) -> dict[str, Any]:
    args = config.get("args", {})
    active = config.get("active_reward", {})
    backtest = config.get("competitive_backtest") or {}
    metadata = config.get("loaded_trained_model_metadata") or {}
    checkpoint = metadata.get("checkpoint_validation") or {}
    result = {key: args.get(key) for key in CONFIG_FIELDS}
    result.update(
        {
            "reward_type": active.get("type"),
            "reward_own_profit_weight": active.get("own_profit_weight"),
            "reward_profit_advantage_weight": active.get("profit_advantage_weight"),
            "reward_own_profit_scale": active.get("own_profit_scale"),
            "reward_profit_advantage_scale": active.get("profit_advantage_scale"),
            "reward_dominance_quality_scale": active.get("dominance_quality_scale"),
            "backtest_passed": backtest.get("passed"),
            "backtest_failures": backtest.get("failures", []),
            "backtest_late_share_advantage": backtest.get("late_completed_share_advantage"),
            "backtest_late_profit_advantage": backtest.get("late_profit_advantage_per_request"),
            "backtest_reward_retention": backtest.get("late_evaluation_reward_retention"),
            "backtest_late_action_diversity": backtest.get("late_action_diversity"),
            "backtest_late_fare_change_rate": backtest.get("late_fare_equivalent_change_rate"),
            "checkpoint_best_day": metadata.get("best_validation_day"),
            "checkpoint_best_score": metadata.get("best_validation_score"),
            "checkpoint_eligible": checkpoint.get("checkpoint_eligible"),
            "checkpoint_mean_profit_advantage": checkpoint.get("mean_profit_advantage"),
            "checkpoint_worst_profit_advantage": checkpoint.get("worst_profit_advantage"),
            "checkpoint_profit_win_rate": checkpoint.get("profit_win_rate"),
            "checkpoint_policy_dynamicity": checkpoint.get("policy_dynamicity"),
            "checkpoint_mode_profit_advantages": checkpoint.get("mode_profit_advantages", {}),
        }
    )
    return result


def version_number(prefix: str) -> str:
    match = re.search(r"(?:^|_)(v\d+(?:_\d+)?|v\d+f)(?:_|$)", prefix)
    return match.group(1) if match else prefix


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def main() -> None:
    runs: list[dict[str, Any]] = []
    for prefix in PRIMARY_PREFIXES:
        config_path = ARTIFACTS / f"{prefix}_run_config.json"
        evaluation_path = ARTIFACTS / f"{prefix}_evaluation_diagnostics.csv"
        training_path = ARTIFACTS / f"{prefix}_training_diagnostics.csv"
        if not config_path.exists() and not evaluation_path.exists():
            continue
        config = (
            json.loads(config_path.read_text(encoding="utf-8"))
            if config_path.exists()
            else {}
        )
        evaluation = read_csv(evaluation_path)
        training = read_csv(training_path)
        run = {
            "prefix": prefix,
            "version": version_number(prefix),
            "config_path": str(config_path) if config_path.exists() else None,
            "evaluation_path": str(evaluation_path) if evaluation_path.exists() else None,
            "training_path": str(training_path) if training_path.exists() else None,
            "config": summarize_config(config),
            "evaluation": summarize_evaluation(evaluation),
            "training": summarize_training(training),
        }
        runs.append(run)

    threshold_files = sorted(ARTIFACTS.glob("*_price_gap_threshold_distribution.csv"))
    threshold_summaries = {}
    for path in threshold_files:
        prefix = path.name.removesuffix("_price_gap_threshold_distribution.csv")
        if prefix not in PRIMARY_PREFIXES:
            continue
        rows = read_csv(path)
        threshold_summaries[prefix] = rows

    result = {
        "run_count": len(runs),
        "runs": runs,
        "threshold_distributions": threshold_summaries,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(json_safe(result), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT} with {len(runs)} runs")


if __name__ == "__main__":
    main()
