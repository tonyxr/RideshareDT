#!/usr/bin/env python3
"""Generate a large, deterministic graph gallery from saved live evidence.

This script is read-only with respect to policies and simulations. It accepts
multiple ``label=run_root`` inputs, plots every available city independently,
then produces like-for-like revision and city comparisons. Missing metrics are
recorded rather than silently graphed as zero.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CITIES = ("nyc", "boston", "chicago")
METRICS = (
    "avg_reward",
    "reward_base",
    "reward_actor_credit",
    "avg_profit_per_request",
    "avg_firm2_profit_per_request",
    "avg_profit_advantage_per_request",
    "avg_rev_per_request",
    "avg_firm2_rev_per_request",
    "avg_market_share",
    "avg_firm2_market_share",
    "share_gap",
    "avg_fulfillment_rate",
    "avg_completion_coverage",
    "avg_wait_minutes",
    "firm1_tariff_level",
    "raw_policy_entropy",
    "raw_policy_top_prob",
    "raw_policy_hold_prob",
    "behavior_exploration_rate",
    "ppo_lr",
    "ppo_critic_lr",
    "ppo_ent_coeff",
    "ppo_explained_variance",
    "ppo_critic_explained_variance_unclipped",
    "ppo_value_loss",
    "ppo_policy_loss",
    "ppo_approx_kl",
    "ppo_rollout_action_diversity",
    "ppo_action_coverage",
    "ppo_state_action_sensitivity",
    "ppo_continuous_magnitude_mean",
    "ppo_continuous_magnitude_std",
    "counterfactual_action_coverage",
    "objective_optimizer_action_change_rate",
    "objective_optimizer_action_margin",
    "objective_optimizer_q_spread",
    "driver_stock_firm1",
    "driver_stock_firm2",
    "avg_price_gap_f2_minus_f1",
)
COEFFICIENTS = (
    "base_fare",
    "per_minute",
    "per_mile",
    "booking_fee",
    "airport_fee",
)


def finite(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    for row in rows:
        row["day"] = int(row.get("batch", len(rows) - 1)) + 1
        row["share_gap"] = finite(row.get("avg_market_share")) - finite(
            row.get("avg_firm2_market_share")
        )
    return rows


def live_path(root: Path, city: str) -> Path | None:
    path = root / city / f"{city}_training_live.jsonl"
    return path if path.is_file() else None


def rolling(values: np.ndarray, width: int = 50) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=float)
    for index in range(values.size):
        sample = values[max(0, index - width + 1):index + 1]
        sample = sample[np.isfinite(sample)]
        if sample.size:
            result[index] = float(sample.mean())
    return result


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=155, bbox_inches="tight")
    plt.close(fig)


def plot_series(rows: list[dict[str, Any]], key: str, path: Path, title: str) -> bool:
    x = np.asarray([row["day"] for row in rows], dtype=float)
    y = np.asarray([finite(row.get(key)) for row in rows], dtype=float)
    if not np.isfinite(y).any():
        return False
    fig, ax = plt.subplots(figsize=(11, 4.5))
    stages = [str(row.get("training_stage", "unknown")) for row in rows]
    start = 0
    palette = ("#edf8fb", "#f7fcf0", "#fff7ec", "#f1eef6")
    stage_order: list[str] = []
    for index in range(1, len(stages) + 1):
        if index == len(stages) or stages[index] != stages[start]:
            stage = stages[start]
            if stage not in stage_order:
                stage_order.append(stage)
            ax.axvspan(x[start], x[index - 1], color=palette[(len(stage_order)-1) % len(palette)], alpha=.55)
            start = index
    ax.plot(x, y, color="#7b8da6", alpha=.22, lw=.55, label="daily")
    ax.plot(x, rolling(y), color="#17365d", lw=1.8, label="50-day mean")
    ax.axhline(0, color="black", lw=.6, alpha=.5)
    ax.set(title=title, xlabel="training day", ylabel=key)
    ax.grid(alpha=.18)
    ax.legend(fontsize=8)
    fig.text(.01, -.015, "Stage backgrounds follow the saved training_stage field; one trained seed.", fontsize=8)
    save(fig, path)
    return True


def plot_coefficients(rows: list[dict[str, Any]], path: Path, title: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    x = np.asarray([row["day"] for row in rows])
    for firm, ax in (("firm1", axes[0]), ("firm2", axes[1])):
        for name in COEFFICIENTS:
            y = np.asarray([finite(row.get(f"{firm}_{name}")) for row in rows])
            if np.isfinite(y).any():
                ax.plot(x, rolling(y, 20), lw=1.25, label=name)
        ax.set_ylabel(f"{firm} coefficient")
        ax.grid(alpha=.2)
        ax.legend(ncol=5, fontsize=8)
    axes[-1].set_xlabel("training day")
    fig.suptitle(title)
    save(fig, path)


def plot_stage_summary(rows: list[dict[str, Any]], path: Path, title: str) -> None:
    keys = (
        "avg_reward",
        "avg_profit_advantage_per_request",
        "share_gap",
        "avg_fulfillment_rate",
    )
    stages = list(dict.fromkeys(str(row.get("training_stage")) for row in rows))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, key in zip(axes.ravel(), keys):
        values = []
        for stage in stages:
            sample = np.asarray([
                finite(row.get(key)) for row in rows
                if str(row.get("training_stage")) == stage
            ])
            values.append(float(np.nanmean(sample)))
        ax.bar(np.arange(len(stages)), values, color="#4c78a8")
        ax.axhline(0, color="black", lw=.6)
        ax.set_xticks(np.arange(len(stages)), stages, rotation=20, ha="right")
        ax.set_title(key)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle(title)
    save(fig, path)


def plot_opponents(rows: list[dict[str, Any]], path: Path, title: str) -> None:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("training_active_firm2_mode", "unknown"))].append(row)
    labels = [label for label, sample in groups.items() if len(sample) >= 2]
    keys = ("avg_reward", "avg_profit_advantage_per_request", "share_gap", "avg_fulfillment_rate")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for ax, key in zip(axes.ravel(), keys):
        values = [float(np.nanmean([finite(row.get(key)) for row in groups[label]])) for label in labels]
        ax.barh(np.arange(len(labels)), values, color="#59a14f")
        ax.axvline(0, color="black", lw=.6)
        ax.set_yticks(np.arange(len(labels)), labels, fontsize=8)
        ax.set_title(key)
        ax.grid(axis="x", alpha=.2)
    fig.suptitle(title)
    save(fig, path)


def plot_actions(rows: list[dict[str, Any]], path: Path, title: str) -> None:
    labels = [str(row.get("last_action_label", "unknown")) for row in rows]
    counts = Counter(labels)
    chosen = counts.most_common(20)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].barh(np.arange(len(chosen)), [count for _, count in chosen], color="#f28e2b")
    axes[0].set_yticks(np.arange(len(chosen)), [label for label, _ in chosen], fontsize=8)
    axes[0].invert_yaxis(); axes[0].set_title("top sampled action labels")
    x = np.asarray([row["day"] for row in rows])
    for key in ("raw_policy_hold_prob", "raw_policy_top_prob", "sampled_not_raw_top_rate", "ppo_rollout_action_diversity"):
        y = np.asarray([finite(row.get(key)) for row in rows])
        if np.isfinite(y).any():
            axes[1].plot(x, rolling(y), label=key)
    axes[1].set_title("policy action behavior (50-day means)")
    axes[1].grid(alpha=.2); axes[1].legend(fontsize=8)
    fig.suptitle(title)
    save(fig, path)


def plot_distance_bins(rows: list[dict[str, Any]], path: Path, title: str) -> None:
    bins = ("0_2", "2_5", "5_10", "10_plus")
    metrics = ("firm1_choice_share", "firm1_completed_share", "firm1_revpr", "price_gap_mean")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    x = np.asarray([row["day"] for row in rows])
    for ax, metric in zip(axes.ravel(), metrics):
        for distance in bins:
            key = f"distance_bin_{distance}_{metric}"
            y = np.asarray([finite(row.get(key)) for row in rows])
            if np.isfinite(y).any():
                ax.plot(x, rolling(y), label=distance)
        ax.set_title(metric); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle(title)
    save(fig, path)


def comparison_plot(
    datasets: dict[tuple[str, str], list[dict[str, Any]]],
    city: str,
    key: str,
    path: Path,
) -> bool:
    fig, ax = plt.subplots(figsize=(12, 5))
    plotted = False
    for (label, candidate_city), rows in datasets.items():
        if candidate_city != city:
            continue
        x = np.asarray([row["day"] for row in rows])
        y = np.asarray([finite(row.get(key)) for row in rows])
        if np.isfinite(y).any():
            ax.plot(x, rolling(y), lw=1.5, label=label)
            plotted = True
    if not plotted:
        plt.close(fig)
        return False
    ax.axhline(0, color="black", lw=.6)
    ax.set(title=f"{city}: {key} across runs", xlabel="training day", ylabel=key)
    ax.grid(alpha=.2); ax.legend(fontsize=8)
    save(fig, path)
    return True


def city_comparison(
    datasets: dict[tuple[str, str], list[dict[str, Any]]],
    label: str,
    key: str,
    path: Path,
) -> bool:
    fig, ax = plt.subplots(figsize=(12, 5))
    plotted = False
    for city in CITIES:
        rows = datasets.get((label, city), [])
        if not rows:
            continue
        x = np.asarray([row["day"] for row in rows])
        y = np.asarray([finite(row.get(key)) for row in rows])
        if np.isfinite(y).any():
            ax.plot(x, rolling(y), lw=1.5, label=city)
            plotted = True
    if not plotted:
        plt.close(fig)
        return False
    ax.axhline(0, color="black", lw=.6)
    ax.set(title=f"{label}: multicity {key}", xlabel="training day", ylabel=key)
    ax.grid(alpha=.2); ax.legend()
    save(fig, path)
    return True


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def evaluation_summary_path(root: Path, city: str) -> Path | None:
    for directory in (
        "full_evaluation_summary_all_cases",
        "full_evaluation_summary",
    ):
        path = root / city / directory / "benchmark_summary.csv"
        if path.is_file():
            return path
    return None


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def mean_by(rows: list[dict[str, Any]], group: str, key: str) -> dict[str, float]:
    samples: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = finite(row.get(key))
        if math.isfinite(value):
            samples[str(row.get(group, "unknown"))].append(value)
    return {name: float(np.mean(values)) for name, values in samples.items() if values}


def plot_evaluation_rivals(rows: list[dict[str, Any]], key: str, path: Path, title: str) -> bool:
    selected = [
        row for row in rows
        if row.get("window") == "post_burnin"
        and row.get("policy_spec") == "rl_actor"
    ]
    values = mean_by(selected, "rival", key)
    if not values:
        return False
    labels = list(values); data = [values[label] for label in labels]
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(np.arange(len(labels)), data, color="#4e79a7")
    ax.axhline(0, color="black", lw=.7)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=55, ha="right", fontsize=8)
    ax.set(title=title, ylabel=key)
    ax.grid(axis="y", alpha=.2)
    save(fig, path)
    return True


def plot_evaluation_cross_run(
    summaries: dict[tuple[str, str], list[dict[str, Any]]],
    city: str,
    key: str,
    path: Path,
) -> bool:
    series_by_run: dict[str, dict[str, float]] = {}
    rivals: list[str] = []
    for (label, candidate_city), rows in summaries.items():
        if candidate_city != city:
            continue
        selected = [
            row for row in rows
            if row.get("window") == "post_burnin"
            and row.get("policy_spec") == "rl_actor"
        ]
        values = mean_by(selected, "rival", key)
        if values:
            series_by_run[label] = values
            for rival in values:
                if rival not in rivals:
                    rivals.append(rival)
    if not series_by_run:
        return False
    fig, ax = plt.subplots(figsize=(15, 7))
    width = .8 / max(1, len(series_by_run))
    x = np.arange(len(rivals))
    for index, (label, values) in enumerate(series_by_run.items()):
        ax.bar(
            x - .4 + width / 2 + index * width,
            [values.get(rival, np.nan) for rival in rivals],
            width=width,
            label=label,
        )
    ax.axhline(0, color="black", lw=.7)
    ax.set_xticks(x, rivals, rotation=55, ha="right", fontsize=8)
    ax.set(title=f"{city}: held-out {key} across completed runs", ylabel=key)
    ax.grid(axis="y", alpha=.2); ax.legend(fontsize=8)
    save(fig, path)
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="label=/absolute/run/root")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    roots: dict[str, Path] = {}
    for item in args.run:
        label, separator, value = item.partition("=")
        if not separator or not label or not value:
            parser.error("--run must be label=path")
        roots[label] = Path(value).resolve()

    datasets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    coverage: list[dict[str, Any]] = []
    missing: list[str] = []
    for label, root in roots.items():
        for city in CITIES:
            path = live_path(root, city)
            if path is None:
                missing.append(f"{label}/{city}:training_live")
                continue
            rows = load_jsonl(path)
            datasets[(label, city)] = rows
            destination = output / label / city
            observed = []
            for key in METRICS:
                if plot_series(rows, key, destination / "series" / f"{key}.png", f"{label} / {city}: {key}"):
                    observed.append(key)
            plot_coefficients(rows, destination / "tariff_coefficients.png", f"{label} / {city}: tariff coefficients")
            plot_stage_summary(rows, destination / "stage_summary.png", f"{label} / {city}: stage means")
            plot_opponents(rows, destination / "opponent_summary.png", f"{label} / {city}: opponent means")
            plot_actions(rows, destination / "action_behavior.png", f"{label} / {city}: action behavior")
            plot_distance_bins(rows, destination / "distance_bins.png", f"{label} / {city}: distance-bin outcomes")
            coverage.append({
                "run": label,
                "city": city,
                "days": len(rows),
                "latest_stage": rows[-1].get("training_stage") if rows else None,
                "metrics_plotted": len(observed),
                "source": str(path),
            })

    comparison_keys = (
        "avg_reward", "avg_profit_per_request", "avg_firm2_profit_per_request",
        "avg_profit_advantage_per_request", "avg_market_share", "share_gap",
        "avg_fulfillment_rate", "raw_policy_entropy", "ppo_explained_variance",
        "ppo_rollout_action_diversity", "firm1_tariff_level",
    )
    for city in CITIES:
        for key in comparison_keys:
            comparison_plot(datasets, city, key, output / "comparisons" / city / f"{key}.png")
    for label in roots:
        for key in comparison_keys:
            city_comparison(datasets, label, key, output / "comparisons" / label / f"{key}.png")

    evaluation_keys = (
        "objective_reward",
        "firm1_profit_per_request",
        "firm2_profit_per_request",
        "profit_lead_per_request",
        "firm1_revenue_per_request",
        "firm2_revenue_per_request",
        "firm1_market_share",
        "firm2_market_share",
        "firm1_fulfillment",
        "firm2_fulfillment",
        "policy_switch_rate",
    )
    summaries: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for label, root in roots.items():
        for city in CITIES:
            path = evaluation_summary_path(root, city)
            if path is None:
                missing.append(f"{label}/{city}:completed_evaluation_summary")
                continue
            rows = read_csv(path)
            summaries[(label, city)] = rows
            for key in evaluation_keys:
                plot_evaluation_rivals(
                    rows,
                    key,
                    output / label / city / "evaluation" / f"{key}.png",
                    f"{label} / {city}: adaptive policy by evaluation rival",
                )
    for city in CITIES:
        for key in evaluation_keys:
            plot_evaluation_cross_run(
                summaries,
                city,
                key,
                output / "comparisons" / city / "evaluation" / f"{key}.png",
            )

    write_csv(output / "coverage.csv", coverage)
    pngs = sorted(path.relative_to(output) for path in output.rglob("*.png"))
    index = {
        "created_at": __import__("time").time(),
        "runs": {label: str(root) for label, root in roots.items()},
        "coverage": coverage,
        "missing": missing,
        "png_count": len(pngs),
        "figures": [str(path) for path in pngs],
        "evidence_note": "Descriptive saved training evidence; no policy selection or simulation mutation.",
    }
    (output / "figure_index.json").write_text(json.dumps(index, indent=2))
    links = "".join(
        f'<li><a href="{html.escape(str(path), quote=True)}">{html.escape(str(path))}</a></li>'
        for path in pngs
    )
    (output / "figures.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>R15c/R15d figures</title>"
        "<h1>Revision 15c / 15d training figures</h1>"
        "<p>Saved evidence only; reward scales differ by curriculum stage.</p><ul>"
        + links + "</ul>"
    )
    print(json.dumps({"status": "complete", "png_count": len(pngs), "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
