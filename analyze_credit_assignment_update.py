from __future__ import annotations

"""Compare v23 and v24 learning signals and frozen benchmark outcomes."""

import argparse
import csv
import json
import math
import os
from typing import Dict, Iterable, List, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


BENCHMARKS = (
    "pi_price_gap",
    "mpc_grid",
    "region_supply_demand",
    "surge_driver_incentive",
    "queue_service_threshold",
)
VERSIONS = ("v23", "v24")
OBJECTIVES = ("profit", "competitiveness")
COLORS = {
    "v23": "#607D8B",
    "v24": "#1565C0",
    "v25": "#00897B",
}


def _read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def _values(rows: Iterable[Mapping[str, str]], key: str) -> np.ndarray:
    return np.asarray([_number(row.get(key)) for row in rows], dtype=float)


def _mean(rows: Sequence[Mapping[str, str]], key: str) -> float:
    values = _values(rows, key)
    return float(np.nanmean(values)) if np.any(np.isfinite(values)) else float("nan")


def _last(rows: Sequence[Mapping[str, str]], key: str) -> float:
    values = _values(rows, key)
    values = values[np.isfinite(values)]
    return float(values[-1]) if len(values) else float("nan")


def _sum(rows: Sequence[Mapping[str, str]], key: str) -> float:
    values = _values(rows, key)
    return float(np.nansum(values)) if np.any(np.isfinite(values)) else float("nan")


def _is_true(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _load_case(training_prefix: str, frozen_prefix: str) -> Dict[str, object]:
    diagnostics = _read_csv(f"{training_prefix}_training_diagnostics.csv")
    updates = [row for row in diagnostics if _is_true(row.get("ppo_update_performed"))]
    validations = [
        row for row in diagnostics if math.isfinite(_number(row.get("validation_score")))
    ]
    benchmark_rows = _read_csv(f"{frozen_prefix}_benchmark_summary.csv")
    by_benchmark = {row["benchmark"]: row for row in benchmark_rows}
    missing = sorted(set(BENCHMARKS) - set(by_benchmark))
    if missing:
        raise ValueError(f"Missing benchmark rows in {frozen_prefix}: {missing}")
    with open(f"{frozen_prefix}_run_config.json", encoding="utf-8") as handle:
        run_config = json.load(handle)
    benchmark_eval = run_config["benchmark_evaluation"]
    return {
        "diagnostics": diagnostics,
        "updates": updates,
        "validations": validations,
        "benchmarks": by_benchmark,
        "benchmark_evaluation": benchmark_eval,
    }


def _summary_row(version: str, objective: str, case: Mapping[str, object]) -> Dict[str, object]:
    updates = case["updates"]
    validations = case["validations"]
    benchmarks = case["benchmarks"]
    evaluation = case["benchmark_evaluation"]
    assert isinstance(updates, list)
    assert isinstance(validations, list)
    assert isinstance(benchmarks, dict)
    assert isinstance(evaluation, dict)
    final_validation = validations[-1]
    benchmark_rows = [benchmarks[name] for name in BENCHMARKS]
    advantage_key = (
        "validation_worst_profit_advantage"
        if objective == "profit"
        else "validation_worst_completed_share_advantage"
    )
    return {
        "version": version,
        "objective": objective,
        "ppo_update_count": len(updates),
        "critic_explained_variance_mean": _mean(updates, "ppo_explained_variance"),
        "critic_explained_variance_final": _last(updates, "ppo_explained_variance"),
        "critic_loss_mean": _mean(updates, "ppo_critic_loss"),
        "state_action_sensitivity_mean": _mean(updates, "ppo_state_action_sensitivity"),
        "state_action_sensitivity_final": _last(updates, "ppo_state_action_sensitivity"),
        "approx_kl_mean": _mean(updates, "ppo_approx_kl"),
        "early_kl_stop_count": sum(_is_true(row.get("ppo_stopped_early_kl")) for row in updates),
        "actor_optimizer_steps": _sum(updates, "ppo_actor_optimizer_steps"),
        "critic_optimizer_steps": _sum(updates, "ppo_critic_optimizer_steps"),
        "action_q_loss_mean": _mean(updates, "ppo_action_q_loss"),
        "counterfactual_target_coverage_mean": _mean(
            updates, "ppo_counterfactual_target_coverage"
        ),
        "final_validation_day": _number(final_validation.get("batch")),
        "final_validation_score": _number(final_validation.get("validation_score")),
        "final_validation_worst_objective_advantage": _number(
            final_validation.get(advantage_key)
        ),
        "final_validation_action_diversity": _number(
            final_validation.get("validation_action_diversity")
        ),
        "final_validation_action_switch_rate": _number(
            final_validation.get("validation_action_switch_rate")
        ),
        "final_validation_policy_dynamicity": _number(
            final_validation.get("validation_policy_dynamicity")
        ),
        "frozen_score": _number(evaluation.get("score")),
        "frozen_feasible_mode_fraction": _number(
            evaluation.get("objective_feasible_mode_fraction")
        ),
        "frozen_benchmark_pass_count": sum(
            _is_true(row.get("passes_objective")) for row in benchmark_rows
        ),
        "frozen_mean_firm1_profit_per_request": float(
            np.mean([_number(row["firm1_profit_per_request"]) for row in benchmark_rows])
        ),
        "frozen_mean_objective_advantage": float(
            np.mean([_number(row["advantage"]) for row in benchmark_rows])
        ),
        "frozen_worst_final_quarter_advantage": float(
            np.min([_number(row["final_quarter_advantage"]) for row in benchmark_rows])
        ),
        "frozen_worst_rolling_dominance_rate": float(
            np.min([_number(row["rolling_dominance_rate"]) for row in benchmark_rows])
        ),
    }


def _plot_learning(cases: Mapping[tuple[str, str], Mapping[str, object]], path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex="col")
    for row_index, objective in enumerate(OBJECTIVES):
        for version in VERSIONS:
            updates = cases[(version, objective)]["updates"]
            assert isinstance(updates, list)
            x = np.arange(1, len(updates) + 1)
            axes[row_index, 0].plot(
                x,
                _values(updates, "ppo_explained_variance"),
                label=version,
                color=COLORS[version],
                linewidth=1.8,
            )
            sensitivity = np.maximum(
                _values(updates, "ppo_state_action_sensitivity"), 1e-8
            )
            axes[row_index, 1].plot(
                x,
                sensitivity,
                label=version,
                color=COLORS[version],
                linewidth=1.8,
            )
        axes[row_index, 0].axhline(0.0, color="black", linewidth=0.8)
        axes[row_index, 0].set_ylabel(
            f"{objective.title()}\ncritic explained variance"
        )
        axes[row_index, 1].set_ylabel("State-action sensitivity")
        axes[row_index, 1].set_yscale("log")
        for axis in axes[row_index]:
            axis.grid(linestyle="--", alpha=0.25)
            axis.legend(loc="best")
    axes[-1, 0].set_xlabel("PPO update")
    axes[-1, 1].set_xlabel("PPO update")
    fig.suptitle(
        f"{VERSIONS[-1]} Credit-Assignment Update: Learning-Signal Audit"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_validation(cases: Mapping[tuple[str, str], Mapping[str, object]], path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for row_index, objective in enumerate(OBJECTIVES):
        advantage_key = (
            "validation_worst_profit_advantage"
            if objective == "profit"
            else "validation_worst_completed_share_advantage"
        )
        scale = 1.0 if objective == "profit" else 100.0
        for version in VERSIONS:
            validations = cases[(version, objective)]["validations"]
            assert isinstance(validations, list)
            x = _values(validations, "batch")
            axes[row_index, 0].plot(
                x,
                _values(validations, "validation_score"),
                marker="o",
                label=version,
                color=COLORS[version],
            )
            axes[row_index, 1].plot(
                x,
                scale * _values(validations, advantage_key),
                marker="o",
                label=version,
                color=COLORS[version],
            )
        axes[row_index, 0].set_ylabel(f"{objective.title()}\nvalidation score")
        unit = "$ / request" if objective == "profit" else "percentage points"
        axes[row_index, 1].set_ylabel(f"Worst objective advantage ({unit})")
        axes[row_index, 1].axhline(0.0, color="black", linewidth=0.8)
        for axis in axes[row_index]:
            axis.set_xlabel("Training day")
            axis.grid(linestyle="--", alpha=0.25)
            axis.legend(loc="best")
    fig.suptitle(
        f"{VERSIONS[0]} vs {VERSIONS[-1]}: Held-Out Validation Through Training"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_benchmarks(cases: Mapping[tuple[str, str], Mapping[str, object]], path: str) -> None:
    labels = [name.replace("_", "\n") for name in BENCHMARKS]
    x = np.arange(len(BENCHMARKS), dtype=float)
    width = 0.36
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex="col")
    for row_index, objective in enumerate(OBJECTIVES):
        for version_index, version in enumerate(VERSIONS):
            benchmarks = cases[(version, objective)]["benchmarks"]
            assert isinstance(benchmarks, dict)
            rows = [benchmarks[name] for name in BENCHMARKS]
            offset = (version_index - 0.5) * width
            scale = 1.0 if objective == "profit" else 100.0
            bars = axes[row_index, 0].bar(
                x + offset,
                [scale * _number(row["advantage"]) for row in rows],
                width,
                label=version,
                color=COLORS[version],
                alpha=0.9,
            )
            for bar, row in zip(bars, rows):
                if not _is_true(row.get("passes_objective")):
                    bar.set_hatch("///")
                    bar.set_alpha(0.5)
            axes[row_index, 1].bar(
                x + offset,
                [_number(row["firm1_profit_per_request"]) for row in rows],
                width,
                label=version,
                color=COLORS[version],
                alpha=0.9,
            )
        unit = "$ / request" if objective == "profit" else "percentage points"
        axes[row_index, 0].set_ylabel(f"{objective.title()}\nobjective advantage ({unit})")
        axes[row_index, 1].set_ylabel("Firm 1 profit / request ($)")
        axes[row_index, 0].axhline(0.0, color="black", linewidth=0.8)
        for axis in axes[row_index]:
            axis.grid(axis="y", linestyle="--", alpha=0.25)
            axis.legend(loc="best")
            axis.set_xticks(x, labels)
    fig.suptitle(
        "Independent Frozen-Policy Audit Across Five NYC Benchmarks\n"
        "Hatched bars failed the configured objective criterion"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> None:
    global VERSIONS
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--versions", nargs=2, default=list(VERSIONS), metavar=("OLD", "NEW")
    )
    pre_args, _ = pre_parser.parse_known_args()
    VERSIONS = tuple(str(version) for version in pre_args.versions)
    parser = argparse.ArgumentParser(parents=[pre_parser])
    for version in VERSIONS:
        for objective in OBJECTIVES:
            parser.add_argument(f"--{version}_{objective}_training_prefix", required=True)
            parser.add_argument(f"--{version}_{objective}_frozen_prefix", required=True)
    parser.add_argument("--out_prefix", required=True)
    args = parser.parse_args()

    cases: Dict[tuple[str, str], Dict[str, object]] = {}
    for version in VERSIONS:
        for objective in OBJECTIVES:
            cases[(version, objective)] = _load_case(
                getattr(args, f"{version}_{objective}_training_prefix"),
                getattr(args, f"{version}_{objective}_frozen_prefix"),
            )

    rows = [
        _summary_row(version, objective, cases[(version, objective)])
        for objective in OBJECTIVES
        for version in VERSIONS
    ]
    os.makedirs(os.path.dirname(args.out_prefix) or ".", exist_ok=True)
    summary_path = f"{args.out_prefix}_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    learning_path = f"{args.out_prefix}_learning_signals.png"
    validation_path = f"{args.out_prefix}_validation_progress.png"
    benchmark_path = f"{args.out_prefix}_frozen_benchmarks.png"
    _plot_learning(cases, learning_path)
    _plot_validation(cases, validation_path)
    _plot_benchmarks(cases, benchmark_path)
    for path in (summary_path, learning_path, validation_path, benchmark_path):
        print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
