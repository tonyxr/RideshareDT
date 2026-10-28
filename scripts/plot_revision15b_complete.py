#!/usr/bin/env python3
"""Create a complete Revision 15b training/evaluation figure set."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


VARIANTS = ["R15-A", "R15-B", "R15-C"]
VARIANT_COLORS = {
    "R15-A": "#0072B2",
    "R15-B": "#D55E00",
    "R15-C": "#009E73",
}
SCENARIOS = [
    "C0_calm",
    "A1_shortage_severe",
    "A2_shortage_moderate_null",
    "A3_storm_transit",
    "A5_rival_outage",
    "S1_promotion_bait",
    "S2_undercut_trap",
]
SCENARIO_LABELS = {
    "C0_calm": "Calm",
    "A1_shortage_severe": "Severe\nshortage",
    "A2_shortage_moderate_null": "Moderate\nshortage",
    "A3_storm_transit": "Storm /\ntransit",
    "A5_rival_outage": "Rival\noutage",
    "S1_promotion_bait": "Promotion\nbait",
    "S2_undercut_trap": "Undercut\ntrap",
}
STAGES = [
    (1, 200, "Market", "#e8f1fb"),
    (201, 400, "Static rival", "#f4eee4"),
    (401, 2600, "Catch-up rivals", "#e8f4eb"),
    (2601, 4000, "Profit maximizers", "#f5e8ee"),
]
STAGE_KEY_TO_LABEL = {
    "market_environment": "Market",
    "market": "Market",
    "static_opponent": "Static rival",
    "static": "Static rival",
    "catchup_opponents": "Catch-up rivals",
    "catchup": "Catch-up rivals",
    "price_gap_catchup": "Catch-up rivals",
    "profit_maximizers": "Profit maximizers",
    "profit": "Profit maximizers",
}


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def arr(rows, key):
    return np.asarray([
        np.nan if row.get(key) is None else float(row.get(key)) for row in rows
    ], dtype=float)


def rolling(values, window=40):
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values).astype(float)
    clean = np.where(np.isfinite(values), values, 0.0)
    sums = np.cumsum(np.insert(clean, 0, 0.0))
    counts = np.cumsum(np.insert(valid, 0, 0.0))
    for i in range(len(values)):
        lo = max(0, i + 1 - window)
        count = counts[i + 1] - counts[lo]
        if count:
            out[i] = (sums[i + 1] - sums[lo]) / count
    return out


def finite_xy(rows, x_key, y_key):
    points = [
        (float(row[x_key]) + (1.0 if x_key == "batch" else 0.0), float(row[y_key]))
        for row in rows
        if row.get(x_key) is not None and row.get(y_key) is not None
    ]
    if not points:
        return np.array([]), np.array([])
    return np.asarray(points).T


def style_axes(ax, grid=True):
    if grid:
        ax.grid(True, axis="y", color="#d9dee5", linewidth=0.7, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)


def shade_stages(ax, labels=False):
    for index, (start, stop, label, color) in enumerate(STAGES):
        ax.axvspan(start, stop, color=color, alpha=0.42, zorder=0)
        if labels:
            ax.text(
                (start + stop) / 2,
                0.985 - 0.045 * (index % 2),
                label,
                ha="center",
                va="top",
                fontsize=8,
                color="#444444",
                clip_on=False,
                transform=ax.get_xaxis_transform(),
            )
    for boundary in [200, 400, 2600]:
        ax.axvline(boundary, color="#9ba3ad", linewidth=0.8, alpha=0.8)


def save(fig, out_dir: Path, name: str):
    path = out_dir / name
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(path)


def spec_kind(spec: str):
    if spec == "rl_actor":
        return "Adaptive RL", None
    if "frozen_rl_tariff" in spec:
        return "Frozen final tariff", None
    match = re.search(r"level_([0-9.]+)\.json", spec)
    if match:
        return f"Static {match.group(1)}", float(match.group(1))
    return spec, None


def mean_ci(values):
    vals = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if len(vals) == 0:
        return np.nan, np.nan
    mean = float(np.mean(vals))
    if len(vals) == 1:
        return mean, 0.0
    se = float(np.std(vals, ddof=1) / math.sqrt(len(vals)))
    return mean, 1.96 * se


def load_eval(eval_root: Path):
    data = defaultdict(dict)
    for variant in VARIANTS:
        for scenario in SCENARIOS:
            scenario_dir = eval_root / variant / "evaluation" / scenario
            if not scenario_dir.exists():
                continue
            for path in scenario_dir.glob("*.json"):
                payload = read_json(path)
                label, level = spec_kind(payload["spec"])
                result_rows = list(payload.get("results", {}).values())
                metrics = {}
                for key in [
                    "objective",
                    "own_profit",
                    "rival_profit",
                    "profit_lead",
                    "own_share",
                    "fulfillment",
                ]:
                    metrics[key] = mean_ci([row.get(key) for row in result_rows])
                data[(variant, scenario)][label] = {
                    "level": level,
                    "metrics": metrics,
                    "path": path,
                    "payload": payload,
                }
    return data


def chosen_benchmarks(eval_data, variant, scenario):
    candidates = eval_data[(variant, scenario)]
    static_items = [
        (label, item)
        for label, item in candidates.items()
        if label.startswith("Static ") and item["level"] is not None
    ]
    best_label, best_item = max(
        static_items,
        key=lambda pair: pair[1]["metrics"]["objective"][0],
    )
    return {
        "Adaptive RL": candidates["Adaptive RL"],
        "Frozen final tariff": candidates["Frozen final tariff"],
        "Best static": best_item,
    }, best_label


def plot_training_learning(train_rows, ppo_rows, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    fig.suptitle("Revision 15b training: learning signals", fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    for variant in VARIANTS:
        rows = train_rows[variant]
        days = np.arange(1, len(rows) + 1)
        ax.plot(days, rolling(arr(rows, "avg_reward"), 35), color=VARIANT_COLORS[variant], label=variant, linewidth=1.6)
    ax.set_title("Reward trend (35-day rolling mean)")
    ax.set_xlabel("Training day")
    ax.set_ylabel("Average reward")
    shade_stages(ax, labels=True)
    style_axes(ax)
    ax.legend(frameon=False, ncol=3, fontsize=8)

    ax = axes[0, 1]
    for variant in VARIANTS:
        rows = train_rows[variant]
        days = np.arange(1, len(rows) + 1)
        ax.plot(days, rolling(arr(rows, "raw_policy_entropy"), 35), color=VARIANT_COLORS[variant], label=variant, linewidth=1.5)
    ax.set_title("Actor policy entropy (35-day rolling mean)")
    ax.set_xlabel("Training day")
    ax.set_ylabel("Entropy (nats)")
    shade_stages(ax)
    style_axes(ax)

    ax = axes[1, 0]
    for variant in VARIANTS:
        x, y = finite_xy(ppo_rows[variant], "batch", "ppo_explained_variance")
        ax.plot(x, y, marker="o", markersize=2.8, linewidth=1.0, color=VARIANT_COLORS[variant], label=variant)
    ax.axhline(0, color="#555555", linewidth=0.9)
    ax.axhline(1, color="#555555", linewidth=0.7, linestyle="--")
    ax.set_title("Critic explained variance at PPO updates")
    ax.set_xlabel("Training day")
    ax.set_ylabel("Explained variance")
    shade_stages(ax)
    style_axes(ax)

    ax = axes[1, 1]
    ax2 = ax.twinx()
    for variant in VARIANTS:
        rows = train_rows[variant]
        days = np.arange(1, len(rows) + 1)
        ax.plot(days, arr(rows, "training_stage_exploration_rate"), color=VARIANT_COLORS[variant], linewidth=1.4, label=f"{variant} exploration")
        ax2.plot(days, arr(rows, "ppo_lr"), color=VARIANT_COLORS[variant], linewidth=1.0, linestyle="--", alpha=0.75)
    ax.set_title("Exploration and actor learning-rate schedules")
    ax.set_xlabel("Training day")
    ax.set_ylabel("Exploration rate")
    ax2.set_ylabel("Actor learning rate (dashed)")
    shade_stages(ax)
    style_axes(ax)
    ax2.spines["top"].set_visible(False)
    ax2.tick_params(labelsize=8)
    save(fig, out_dir, "training_learning_signals.png")


def plot_training_economics(train_rows, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    fig.suptitle("Revision 15b training: economics and competitive position", fontsize=16, fontweight="bold")

    ax = axes[0, 0]
    for variant in VARIANTS:
        rows = train_rows[variant]
        days = np.arange(1, len(rows) + 1)
        ax.plot(days, rolling(arr(rows, "avg_profit_per_request"), 40), color=VARIANT_COLORS[variant], linewidth=1.5, label=f"{variant} Firm 1")
        ax.plot(days, rolling(arr(rows, "avg_firm2_profit_per_request"), 40), color=VARIANT_COLORS[variant], linewidth=1.0, linestyle="--", alpha=0.75, label=f"{variant} Firm 2")
    ax.set_title("Profit per request")
    ax.set_xlabel("Training day")
    ax.set_ylabel("Profit / request")
    shade_stages(ax, labels=True)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=7, ncol=2)

    ax = axes[0, 1]
    for variant in VARIANTS:
        rows = train_rows[variant]
        days = np.arange(1, len(rows) + 1)
        ax.plot(days, rolling(arr(rows, "avg_profit_advantage_per_request"), 40), color=VARIANT_COLORS[variant], linewidth=1.5, label=variant)
    ax.axhline(0, color="#555555", linewidth=0.9)
    ax.set_title("Profit lead over Firm 2")
    ax.set_xlabel("Training day")
    ax.set_ylabel("Profit lead / request")
    shade_stages(ax)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    for variant in VARIANTS:
        rows = train_rows[variant]
        days = np.arange(1, len(rows) + 1)
        ax.plot(days, rolling(arr(rows, "avg_market_share"), 40), color=VARIANT_COLORS[variant], linewidth=1.5, label=f"{variant} Firm 1")
        ax.plot(days, rolling(arr(rows, "avg_firm2_market_share"), 40), color=VARIANT_COLORS[variant], linewidth=1.0, linestyle="--", alpha=0.75, label=f"{variant} Firm 2")
    ax.set_title("Completed market share")
    ax.set_xlabel("Training day")
    ax.set_ylabel("Share")
    shade_stages(ax)
    style_axes(ax)

    ax = axes[1, 1]
    for variant in VARIANTS:
        rows = train_rows[variant]
        days = np.arange(1, len(rows) + 1)
        ax.plot(days, rolling(arr(rows, "firm1_tariff_level"), 15), color=VARIANT_COLORS[variant], linewidth=1.4, label=variant)
    ax.set_title("Firm 1 tariff level")
    ax.set_xlabel("Training day")
    ax.set_ylabel("Tariff multiplier")
    shade_stages(ax)
    style_axes(ax)
    save(fig, out_dir, "training_economics.png")


def action_bucket(label):
    if not label:
        return "unknown"
    match = re.search(r"level_step:([+-]?[0-9.]+)", label)
    if match:
        value = float(match.group(1))
        if abs(value) < 1e-9:
            return "hold"
        return f"{value:+.2f}"
    return label


def plot_policy_dynamics(train_rows, out_dir):
    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    fig.suptitle("Revision 15b training: policy dynamics", fontsize=16, fontweight="bold")

    ax = fig.add_subplot(gs[0, 0])
    for variant in VARIANTS:
        labels = [row.get("raw_policy_top_action_label") for row in train_rows[variant]]
        switches = np.zeros(len(labels), dtype=float)
        switches[1:] = [float(labels[i] != labels[i - 1]) for i in range(1, len(labels))]
        ax.plot(np.arange(1, len(labels) + 1), rolling(switches, 100), color=VARIANT_COLORS[variant], linewidth=1.5, label=variant)
    ax.set_title("Raw-policy action switch rate (100-day window)")
    ax.set_xlabel("Training day")
    ax.set_ylabel("Switch probability")
    shade_stages(ax, labels=True)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8)

    ax = fig.add_subplot(gs[0, 1])
    for variant in VARIANTS:
        rows = train_rows[variant]
        ax.plot(np.arange(1, len(rows) + 1), rolling(arr(rows, "raw_policy_top_prob"), 50), color=VARIANT_COLORS[variant], linewidth=1.5, label=variant)
    ax.set_title("Confidence in raw top action (50-day mean)")
    ax.set_xlabel("Training day")
    ax.set_ylabel("Top-action probability")
    shade_stages(ax)
    style_axes(ax)

    ax = fig.add_subplot(gs[1, 0])
    action_order = ["-0.20", "-0.10", "hold", "+0.10", "+0.20"]
    matrix = []
    row_labels = []
    for variant in VARIANTS:
        by_stage = defaultdict(Counter)
        for row in train_rows[variant]:
            stage = STAGE_KEY_TO_LABEL.get(row.get("training_stage", ""), row.get("training_stage", "unknown"))
            by_stage[stage][action_bucket(row.get("raw_policy_top_action_label"))] += 1
        for _, _, stage_label, _ in STAGES:
            counts = by_stage[stage_label]
            total = sum(counts.values()) or 1
            matrix.append([counts.get(action, 0) / total for action in action_order])
            row_labels.append(f"{variant} · {stage_label}")
    image = ax.imshow(np.asarray(matrix), aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(action_order)), action_order)
    ax.set_yticks(range(len(row_labels)), row_labels, fontsize=7)
    ax.set_xlabel("Raw top tariff-level action")
    ax.set_title("Action distribution by training stage")
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            if value >= 0.08:
                ax.text(j, i, f"{value:.0%}", ha="center", va="center", fontsize=7, color="white" if value > 0.55 else "black")
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02, label="Share of days")

    ax = fig.add_subplot(gs[1, 1])
    x = np.arange(len(STAGES))
    width = 0.23
    for idx, variant in enumerate(VARIANTS):
        means = []
        lows = []
        highs = []
        rows = train_rows[variant]
        for start, stop, _, _ in STAGES:
            vals = arr(rows[start - 1 : stop], "firm1_tariff_level")
            vals = vals[np.isfinite(vals)]
            means.append(np.mean(vals))
            lows.append(np.quantile(vals, 0.1))
            highs.append(np.quantile(vals, 0.9))
        means = np.asarray(means)
        yerr = np.vstack([means - np.asarray(lows), np.asarray(highs) - means])
        ax.errorbar(x + (idx - 1) * width, means, yerr=yerr, fmt="o", capsize=3, color=VARIANT_COLORS[variant], label=variant)
    ax.set_xticks(x, [stage[2] for stage in STAGES], rotation=15, ha="right")
    ax.set_ylabel("Tariff level (mean; 10–90% range)")
    ax.set_title("Tariff levels by stage")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8)
    save(fig, out_dir, "training_policy_dynamics.png")


def plot_coefficients(train_rows, out_dir):
    metrics = [
        ("firm1_base_fare", "Base fare"),
        ("firm1_per_minute", "Per minute"),
        ("firm1_per_mile", "Per mile"),
        ("firm1_booking_fee", "Booking fee"),
        ("firm1_airport_fee", "Airport fee"),
        ("avg_price_gap_f2_minus_f1", "Price gap: Firm 2 − Firm 1"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    fig.suptitle("Revision 15b training: tariff coefficients and price gap", fontsize=16, fontweight="bold")
    for ax, (key, title) in zip(axes.flat, metrics):
        for variant in VARIANTS:
            rows = train_rows[variant]
            ax.plot(np.arange(1, len(rows) + 1), rolling(arr(rows, key), 30), color=VARIANT_COLORS[variant], linewidth=1.3, label=variant)
        if "gap" in key:
            ax.axhline(0, color="#555555", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("Training day")
        ax.set_ylabel("Fare units" if "gap" not in key else "Fare difference")
        shade_stages(ax)
        style_axes(ax)
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=3)
    save(fig, out_dir, "training_coefficients.png")


def plot_critic(ppo_rows, benchmark_data, out_dir):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    fig.suptitle("Revision 15b critic diagnostics", fontsize=16, fontweight="bold")
    line_specs = [
        ("ppo_explained_variance", "Overall explained variance", None),
        ("ppo_critic_rmse", "Critic RMSE", None),
        ("ppo_critic_loss", "Critic loss", "log"),
        ("ppo_approx_kl", "Approximate KL divergence", None),
    ]
    for ax, (key, title, yscale) in zip(axes.flat[:4], line_specs):
        for variant in VARIANTS:
            x, y = finite_xy(ppo_rows[variant], "batch", key)
            ax.plot(x, y, color=VARIANT_COLORS[variant], marker="o", markersize=2.2, linewidth=1.0, label=variant)
        if key == "ppo_explained_variance":
            ax.axhline(0, color="#555555", linewidth=0.8)
        if key == "ppo_approx_kl":
            ax.axhline(0.02, color="#555555", linestyle="--", linewidth=0.8, label="Target KL")
        if yscale:
            positive = [line.get_ydata() for line in ax.lines if len(line.get_ydata())]
            if positive and all(np.nanmin(values) > 0 for values in positive):
                ax.set_yscale(yscale)
        ax.set_title(title)
        ax.set_xlabel("Training day")
        shade_stages(ax)
        style_axes(ax)
    axes[0, 0].legend(frameon=False, fontsize=8)

    x = np.arange(len(VARIANTS))
    width = 0.34
    td_ev = [benchmark_data[v].get("heldout_critic_td_explained_variance", np.nan) for v in VARIANTS]
    lam_ev = [benchmark_data[v].get("heldout_critic_lambda_explained_variance", np.nan) for v in VARIANTS]
    ax = axes[1, 1]
    ax.bar(x - width / 2, td_ev, width, label="TD target", color="#56B4E9")
    ax.bar(x + width / 2, lam_ev, width, label="Lambda target", color="#E69F00")
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_xticks(x, VARIANTS)
    ax.set_ylabel("Explained variance")
    ax.set_title("Held-out critic explained variance (27 samples)")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8)

    td_mae = [benchmark_data[v].get("heldout_critic_td_mae", np.nan) for v in VARIANTS]
    lam_mae = [benchmark_data[v].get("heldout_critic_lambda_mae", np.nan) for v in VARIANTS]
    ax = axes[1, 2]
    ax.bar(x - width / 2, td_mae, width, label="TD target", color="#56B4E9")
    ax.bar(x + width / 2, lam_mae, width, label="Lambda target", color="#E69F00")
    ax.set_xticks(x, VARIANTS)
    ax.set_ylabel("Mean absolute error")
    ax.set_title("Held-out critic MAE (27 samples)")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8)
    save(fig, out_dir, "critic_diagnostics.png")


def plot_eval_benchmarks(eval_data, out_dir):
    metric_defs = [
        ("objective", "Objective", "Objective score"),
        ("own_profit", "Firm 1 profit", "Profit / request"),
        ("own_share", "Firm 1 market share", "Completed share"),
    ]
    compare_labels = ["Adaptive RL", "Frozen final tariff", "Best static"]
    compare_colors = ["#0072B2", "#999999", "#E69F00"]
    fig, axes = plt.subplots(3, 3, figsize=(19, 13), constrained_layout=True)
    fig.suptitle("Held-out evaluation: adaptive RL versus tariff benchmarks", fontsize=17, fontweight="bold")
    x = np.arange(len(SCENARIOS))
    width = 0.24
    for row_idx, variant in enumerate(VARIANTS):
        for col_idx, (metric, title, ylabel) in enumerate(metric_defs):
            ax = axes[row_idx, col_idx]
            for bar_idx, (label, color) in enumerate(zip(compare_labels, compare_colors)):
                means, cis = [], []
                for scenario in SCENARIOS:
                    selected, _ = chosen_benchmarks(eval_data, variant, scenario)
                    mean, ci = selected[label]["metrics"][metric]
                    means.append(mean)
                    cis.append(ci)
                ax.bar(
                    x + (bar_idx - 1) * width,
                    means,
                    width,
                    yerr=cis,
                    capsize=2,
                    color=color,
                    alpha=0.9,
                    label=label,
                )
            ax.set_xticks(x, [SCENARIO_LABELS[s] for s in SCENARIOS], fontsize=7)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{variant} · {title}")
            style_axes(ax)
            if row_idx == 0 and col_idx == 0:
                ax.legend(frameon=False, fontsize=8, ncol=3, loc="best")
    save(fig, out_dir, "evaluation_benchmark_comparison.png")


def plot_eval_gains(eval_data, out_dir):
    metrics = [
        ("objective", "Objective gain"),
        ("own_profit", "Profit gain / request"),
        ("own_share", "Market-share gain"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(17, 8), constrained_layout=True)
    fig.suptitle("Held-out adaptive advantage over fixed-tariff benchmarks", fontsize=16, fontweight="bold")
    targets = [("Frozen final tariff", "Adaptive − frozen"), ("Best static", "Adaptive − best static")]
    cmap = LinearSegmentedColormap.from_list("div", ["#b2182b", "#f7f7f7", "#2166ac"])
    for row_idx, (target, row_title) in enumerate(targets):
        for col_idx, (metric, title) in enumerate(metrics):
            matrix = np.zeros((len(VARIANTS), len(SCENARIOS)))
            for i, variant in enumerate(VARIANTS):
                for j, scenario in enumerate(SCENARIOS):
                    selected, _ = chosen_benchmarks(eval_data, variant, scenario)
                    matrix[i, j] = selected["Adaptive RL"]["metrics"][metric][0] - selected[target]["metrics"][metric][0]
            bound = max(np.max(np.abs(matrix)), 1e-6)
            ax = axes[row_idx, col_idx]
            image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=-bound, vmax=bound)
            ax.set_xticks(range(len(SCENARIOS)), [SCENARIO_LABELS[s] for s in SCENARIOS], fontsize=7)
            ax.set_yticks(range(len(VARIANTS)), VARIANTS)
            ax.set_title(f"{row_title}\n{title}")
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    value = matrix[i, j]
                    fmt = "+.3f" if metric != "own_share" else "+.3f"
                    ax.text(j, i, format(value, fmt), ha="center", va="center", fontsize=7, color="white" if abs(value) > bound * 0.55 else "black")
            fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    save(fig, out_dir, "evaluation_relative_gains.png")


def plot_dynamic_opponents(benchmark_data, out_dir):
    mode_order = [
        "actual_firm2__adaptive_best_response_memory__opening_1",
        "training_class__adaptive_best_response_aggressive",
        "training_class__profit_best_response_myopic",
    ]
    mode_labels = ["Memory rival", "Aggressive catch-up", "Myopic profit maximizer"]
    metrics = [
        ("own_profit_per_request", "Firm 1 profit / request"),
        ("objective_advantage_mean", "Objective advantage"),
        ("own_completed_share", "Firm 1 completed share"),
        ("objective_reward_mean", "Objective reward"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    fig.suptitle("Benchmark-opponent evaluation at the final checkpoint", fontsize=16, fontweight="bold")
    x = np.arange(len(mode_order))
    width = 0.24
    for ax, (metric, title) in zip(axes.flat, metrics):
        for idx, variant in enumerate(VARIANTS):
            summaries = benchmark_data[variant].get("rollout_summaries", {})
            values = [summaries.get(mode, {}).get(metric, np.nan) for mode in mode_order]
            ax.bar(x + (idx - 1) * width, values, width, color=VARIANT_COLORS[variant], label=variant)
        if "advantage" in metric:
            ax.axhline(0, color="#555555", linewidth=0.8)
        ax.set_xticks(x, mode_labels, rotation=12, ha="right")
        ax.set_title(title)
        style_axes(ax)
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=3)
    save(fig, out_dir, "dynamic_opponent_benchmarks.png")


def plot_acceptance(acceptance, out_dir):
    gates = ["P1", "P2", "P3", "P4", "Promotion bait", "Undercut trap", "Storm/transit", "Rival outage"]
    matrix = np.zeros((len(VARIANTS), len(gates)))
    for i, variant in enumerate(VARIANTS):
        row = acceptance[variant]
        matrix[i] = [
            row["P1"]["pass"],
            row["P2"]["pass"],
            row["P3"]["pass"],
            row["P4"]["pass"],
            row["P5"]["S1_promotion_bait"]["pass"],
            row["P5"]["S2_undercut_trap"]["pass"],
            row["robustness"]["A3_storm_transit"]["pass"],
            row["robustness"]["A5_rival_outage"]["pass"],
        ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    fig.suptitle("Held-out acceptance and adaptive-pricing evidence", fontsize=16, fontweight="bold")
    ax = axes[0, 0]
    cmap = LinearSegmentedColormap.from_list("passfail", ["#cc6677", "#44aa99"])
    ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(gates)), gates, rotation=25, ha="right")
    ax.set_yticks(range(len(VARIANTS)), VARIANTS)
    ax.set_title("Acceptance gates")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, "PASS" if matrix[i, j] else "FAIL", ha="center", va="center", fontsize=8, color="white")

    ax = axes[0, 1]
    means = [acceptance[v]["level_shortage_minus_calm"]["mean"] for v in VARIANTS]
    lower = [acceptance[v]["level_shortage_minus_calm"]["lower_95"] for v in VARIANTS]
    upper = []
    for v in VARIANTS:
        stats = acceptance[v]["level_shortage_minus_calm"]
        upper.append(stats["mean"] + 1.96 * stats["se"])
    yerr = np.vstack([np.asarray(means) - np.asarray(lower), np.asarray(upper) - np.asarray(means)])
    ax.bar(VARIANTS, means, yerr=yerr, capsize=4, color=[VARIANT_COLORS[v] for v in VARIANTS])
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_title("Shortage tariff response versus calm")
    ax.set_ylabel("Tariff-level difference (95% CI)")
    style_axes(ax)

    x = np.arange(2)
    width = 0.24
    strategic = ["S1_promotion_bait", "S2_undercut_trap"]
    strategic_labels = ["Promotion bait", "Undercut trap"]
    ax = axes[1, 0]
    for idx, variant in enumerate(VARIANTS):
        means, lows, highs = [], [], []
        for scenario in strategic:
            stats = acceptance[variant]["P5"][scenario]["reward_vs_frozen"]
            means.append(stats["mean"])
            lows.append(stats["lower_95"])
            highs.append(stats["mean"] + 1.96 * stats["se"])
        means = np.asarray(means)
        yerr = np.vstack([means - np.asarray(lows), np.asarray(highs) - means])
        ax.bar(x + (idx - 1) * width, means, width, yerr=yerr, capsize=3, color=VARIANT_COLORS[variant], label=variant)
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_xticks(x, strategic_labels)
    ax.set_ylabel("Reward gain over frozen (95% CI)")
    ax.set_title("Strategic-opponent reward response")
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    for idx, variant in enumerate(VARIANTS):
        means, lows, highs = [], [], []
        for scenario in strategic:
            stats = acceptance[variant]["P5"][scenario]["reference_fare_change"]
            means.append(stats["mean"])
            lows.append(stats["lower_95"])
            highs.append(stats["mean"] + 1.96 * stats["se"])
        means = np.asarray(means)
        yerr = np.vstack([means - np.asarray(lows), np.asarray(highs) - means])
        ax.bar(x + (idx - 1) * width, means, width, yerr=yerr, capsize=3, color=VARIANT_COLORS[variant], label=variant)
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_xticks(x, strategic_labels)
    ax.set_ylabel("Reference-fare change (95% CI)")
    ax.set_title("Strategic-opponent price response")
    style_axes(ax)
    save(fig, out_dir, "evaluation_acceptance.png")


def trace_mean(payload, key):
    traces = payload.get("traces", [])
    by_step = defaultdict(list)
    for row in traces:
        value = row.get(key)
        if value is not None and np.isfinite(value):
            by_step[int(row["step"])].append(float(value))
    steps = sorted(by_step)
    return np.asarray(steps), np.asarray([np.mean(by_step[step]) for step in steps])


def plot_eval_trajectories(eval_data, out_dir):
    scenarios = ["A1_shortage_severe", "S1_promotion_bait", "S2_undercut_trap"]
    metrics = [
        ("firm1_tariff_level", "Tariff level", None),
        ("profit_lead", "Profit lead / request", 0),
        ("share_lead", "Market-share lead", 0),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(17, 11), constrained_layout=True)
    fig.suptitle("Held-out adaptive trajectories against the memory rival", fontsize=16, fontweight="bold")
    for col, scenario in enumerate(scenarios):
        for row_idx, (metric, ylabel, zero) in enumerate(metrics):
            ax = axes[row_idx, col]
            for variant in VARIANTS:
                payload = eval_data[(variant, scenario)]["Adaptive RL"]["payload"]
                steps, values = trace_mean(payload, metric)
                days = steps / float(payload.get("steps_per_day", 6))
                ax.plot(days, rolling(values, 6), color=VARIANT_COLORS[variant], linewidth=1.4, label=variant)
                if row_idx == 0:
                    for event in payload.get("v26r", {}).get("scenario", {}).get("events", []):
                        ax.axvspan(event.get("start_day", 0), event.get("end_day", 0), color="#999999", alpha=0.12)
            if zero is not None:
                ax.axhline(zero, color="#555555", linewidth=0.8)
            ax.set_xlabel("Evaluation day")
            ax.set_ylabel(ylabel)
            if row_idx == 0:
                ax.set_title(SCENARIO_LABELS[scenario].replace("\n", " "))
            style_axes(ax)
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=3)
    save(fig, out_dir, "evaluation_adaptive_trajectories.png")


def plot_static_levels(eval_data, out_dir):
    best_levels = np.zeros((len(VARIANTS), len(SCENARIOS)))
    rl_gain = np.zeros_like(best_levels)
    frozen_gain = np.zeros_like(best_levels)
    for i, variant in enumerate(VARIANTS):
        for j, scenario in enumerate(SCENARIOS):
            selected, label = chosen_benchmarks(eval_data, variant, scenario)
            best_levels[i, j] = float(label.split()[-1])
            adaptive = selected["Adaptive RL"]["metrics"]["objective"][0]
            static = selected["Best static"]["metrics"]["objective"][0]
            frozen = selected["Frozen final tariff"]["metrics"]["objective"][0]
            rl_gain[i, j] = adaptive - static
            frozen_gain[i, j] = adaptive - frozen
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), constrained_layout=True)
    fig.suptitle("Static-tariff frontier and adaptive objective gains", fontsize=16, fontweight="bold")
    panels = [
        (best_levels, "Best static tariff level", "viridis", 0.7, 1.1),
        (rl_gain, "Adaptive − best static objective", "coolwarm", -np.max(np.abs(rl_gain)), np.max(np.abs(rl_gain))),
        (frozen_gain, "Adaptive − frozen objective", "coolwarm", -np.max(np.abs(frozen_gain)), np.max(np.abs(frozen_gain))),
    ]
    for ax, (matrix, title, cmap, vmin, vmax) in zip(axes, panels):
        image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(SCENARIOS)), [SCENARIO_LABELS[s] for s in SCENARIOS], fontsize=7)
        ax.set_yticks(range(len(VARIANTS)), VARIANTS)
        ax.set_title(title)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:+.3f}" if title.startswith("Adaptive") else f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=7, color="white" if abs(matrix[i, j]) > max(abs(vmin), abs(vmax)) * 0.55 else "black")
        fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    save(fig, out_dir, "evaluation_static_frontier.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    train_rows = {}
    ppo_rows = {}
    benchmark_data = {}
    for variant in VARIANTS:
        variant_root = args.run_root / "train" / variant / "training"
        train_rows[variant] = read_jsonl(variant_root / "nyc_training_live.jsonl")
        ppo_rows[variant] = read_jsonl(variant_root / "nyc_ppo_updates.jsonl")
        benchmark_data[variant] = read_json(variant_root / "nyc_benchmark_partial.json")

    eval_data = load_eval(args.run_root / "evaluate")
    acceptance = read_json(args.run_root / "evaluate" / "acceptance_partial.json")

    plot_training_learning(train_rows, ppo_rows, args.out)
    plot_training_economics(train_rows, args.out)
    plot_policy_dynamics(train_rows, args.out)
    plot_coefficients(train_rows, args.out)
    plot_critic(ppo_rows, benchmark_data, args.out)
    plot_eval_benchmarks(eval_data, args.out)
    plot_eval_gains(eval_data, args.out)
    plot_dynamic_opponents(benchmark_data, args.out)
    plot_acceptance(acceptance, args.out)
    plot_eval_trajectories(eval_data, args.out)
    plot_static_levels(eval_data, args.out)


if __name__ == "__main__":
    main()
