#!/usr/bin/env python3
"""Plot episode-averaged Revision 15b held-out evaluation trajectories.

The evaluation artifacts store 10 independent 40-day openings per policy.
This script first averages the six intraday observations within each episode-day,
then averages the daily paths across episodes. Shaded bands are 95% confidence
intervals across the 10 episode-level daily means.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


VARIANTS = ["R15-A", "R15-B", "R15-C"]
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
    "C0_calm": "Calm market",
    "A1_shortage_severe": "Severe driver shortage",
    "A2_shortage_moderate_null": "Moderate driver shortage",
    "A3_storm_transit": "Winter storm / transit suspension",
    "A5_rival_outage": "Firm 2 platform outage",
    "S1_promotion_bait": "Firm 2 promotion bait",
    "S2_undercut_trap": "Firm 2 undercut trap",
}
POLICY_COLORS = {
    "Adaptive RL": "#0072B2",
    "Frozen final tariff": "#777777",
    "Best static": "#E69F00",
}
FIRM_STYLES = {"Firm 1": "-", "Firm 2": "--"}


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def spec_kind(spec: str):
    if spec == "rl_actor":
        return "Adaptive RL", None
    if "frozen_rl_tariff" in spec:
        return "Frozen final tariff", None
    match = re.search(r"level_([0-9.]+)\.json", spec)
    if match:
        return f"Static {match.group(1)}", float(match.group(1))
    return spec, None


def objective_mean(payload):
    values = [
        float(row["objective"])
        for row in payload.get("results", {}).values()
        if row.get("objective") is not None
    ]
    return float(np.mean(values)) if values else -math.inf


def load_evaluations(eval_root: Path):
    data = defaultdict(dict)
    for variant in VARIANTS:
        for scenario in SCENARIOS:
            scenario_dir = eval_root / variant / "evaluation" / scenario
            for path in sorted(scenario_dir.glob("*.json")):
                payload = read_json(path)
                label, level = spec_kind(payload["spec"])
                data[(variant, scenario)][label] = {
                    "payload": payload,
                    "level": level,
                    "objective": objective_mean(payload),
                }
    return data


def chosen_policies(data, variant, scenario):
    entries = data[(variant, scenario)]
    statics = [
        (label, item)
        for label, item in entries.items()
        if label.startswith("Static ") and item["level"] is not None
    ]
    best_static_label, best_static = max(statics, key=lambda pair: pair[1]["objective"])
    return {
        "Adaptive RL": entries["Adaptive RL"],
        "Frozen final tariff": entries["Frozen final tariff"],
        "Best static": best_static,
    }, best_static_label


def episode_daily(payload, key):
    """Return episode x day means, aggregating intraday steps first."""
    steps_per_day = int(payload.get("steps_per_day", 6))
    grouped = defaultdict(lambda: defaultdict(list))
    for row in payload.get("traces", []):
        value = row.get(key)
        # Static-tariff traces predate the explicit tariff-level diagnostic but
        # retain the equivalent realized level under ``firm1_level``.
        if value is None and key == "firm1_tariff_level":
            value = row.get("firm1_level")
        if value is None:
            continue
        rollout = str(row.get("rollout", "unknown"))
        day = int(row["step"]) // steps_per_day
        grouped[rollout][day].append(float(value))
    if not grouped:
        return np.empty((0, 0), dtype=float)
    days = int(payload["horizon_steps"]) // steps_per_day
    matrix = np.full((len(grouped), days), np.nan, dtype=float)
    for row_index, rollout in enumerate(sorted(grouped)):
        for day, values in grouped[rollout].items():
            matrix[row_index, day] = float(np.mean(values))
    return matrix


def mean_ci(matrix):
    if matrix.size == 0:
        return np.array([]), np.array([]), np.array([])
    mean = np.nanmean(matrix, axis=0)
    count = np.sum(np.isfinite(matrix), axis=0)
    std = np.nanstd(matrix, axis=0, ddof=1)
    half = np.divide(
        1.96 * std,
        np.sqrt(count),
        out=np.zeros_like(std),
        where=count > 1,
    )
    return mean, mean - half, mean + half


def style_axis(ax):
    ax.grid(True, axis="y", color="#d9dee5", linewidth=0.7, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.set_xlim(1, 40)
    ax.set_xticks([1, 10, 20, 30, 40])


def shade_events(ax, payload):
    events = payload.get("v26r", {}).get("scenario", {}).get("events", [])
    for event in events:
        start = float(event["start_day"]) + 1.0
        end = float(event["end_day"]) + 1.0
        ax.axvspan(start, end, color="#CC79A7", alpha=0.10, zorder=0)


def plot_series(ax, matrix, color, linestyle, label, percent=False, band=True):
    mean, low, high = mean_ci(matrix)
    if mean.size == 0:
        return
    days = np.arange(1, len(mean) + 1)
    factor = 100.0 if percent else 1.0
    ax.plot(
        days,
        factor * mean,
        color=color,
        linestyle=linestyle,
        linewidth=1.45,
        label=label,
        zorder=3,
    )
    if band:
        ax.fill_between(days, factor * low, factor * high, color=color, alpha=0.07, linewidth=0)


def metric_panel(ax, policies, key_f1, key_f2, title, ylabel, percent=False):
    for policy_label, item in policies.items():
        payload = item["payload"]
        color = POLICY_COLORS[policy_label]
        plot_series(
            ax,
            episode_daily(payload, key_f1),
            color,
            FIRM_STYLES["Firm 1"],
            f"{policy_label} · Firm 1",
            percent=percent,
        )
        if key_f2:
            plot_series(
                ax,
                episode_daily(payload, key_f2),
                color,
                FIRM_STYLES["Firm 2"],
                f"{policy_label} · Firm 2",
                percent=percent,
            )
        shade_events(ax, payload)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=9)
    style_axis(ax)


def plot_scenario(data, scenario, out_dir):
    fig, axes = plt.subplots(3, 4, figsize=(21, 12), sharex=True, constrained_layout=True)
    fig.suptitle(
        f"Held-out evaluation over time — {SCENARIO_LABELS[scenario]}",
        fontsize=16,
        fontweight="bold",
    )
    static_labels = []
    for row_index, variant in enumerate(VARIANTS):
        policies, best_static_label = chosen_policies(data, variant, scenario)
        static_labels.append(f"{variant}: {best_static_label.replace('Static ', '')}×")

        metric_panel(
            axes[row_index, 0], policies,
            "profit_per_request", "rival_profit_per_request",
            "Profit", "Profit per request",
        )
        metric_panel(
            axes[row_index, 1], policies,
            "own_share", "rival_share",
            "Market share", "Completed choice share (%)", percent=True,
        )
        metric_panel(
            axes[row_index, 2], policies,
            "fulfillment", None,
            "Fulfillment", "Firm 1 fulfillment (%)", percent=True,
        )
        metric_panel(
            axes[row_index, 3], policies,
            "firm1_level", "firm2_level_effective",
            "Effective price level", "Price multiplier",
        )
        axes[row_index, 0].text(
            -0.28, 0.5, variant,
            transform=axes[row_index, 0].transAxes,
            rotation=90, ha="center", va="center",
            fontsize=12, fontweight="bold",
        )
        for col in range(4):
            if row_index == 2:
                axes[row_index, col].set_xlabel("Evaluation day", fontsize=9)

    policy_handles = [
        Line2D([0], [0], color=color, lw=2.2, label=label)
        for label, color in POLICY_COLORS.items()
    ]
    firm_handles = [
        Line2D([0], [0], color="#222222", lw=1.8, linestyle=style, label=firm)
        for firm, style in FIRM_STYLES.items()
    ]
    event_handle = Line2D([0], [0], color="#CC79A7", lw=7, alpha=0.22, label="Scenario event window")
    fig.legend(
        handles=policy_handles + firm_handles + [event_handle],
        loc="upper center", bbox_to_anchor=(0.5, 0.965),
        ncol=6, frameon=False, fontsize=9,
    )
    fig.text(
        0.5, 0.015,
        "Daily mean across 10 episodes; translucent bands are 95% CIs. "
        f"Best static tariff by variant: {', '.join(static_labels)}. "
        "Firm 2 fulfillment and both firms' revenue were not recorded in the evaluation traces.",
        ha="center", va="bottom", fontsize=9, color="#444444",
    )
    output = out_dir / f"episode_paths_{scenario}.png"
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(output)


def aggregate_policy_matrices(data, variant, policy_label, key):
    matrices = []
    for scenario in SCENARIOS:
        policies, _ = chosen_policies(data, variant, scenario)
        matrix = episode_daily(policies[policy_label]["payload"], key)
        if matrix.size:
            matrices.append(matrix)
    return np.vstack(matrices) if matrices else np.empty((0, 0), dtype=float)


def plot_aggregate(data, out_dir):
    fig, axes = plt.subplots(3, 4, figsize=(21, 12), sharex=True, constrained_layout=True)
    fig.suptitle(
        "Held-out evaluation over time — average across all seven scenarios",
        fontsize=16,
        fontweight="bold",
    )
    panel_defs = [
        ("profit_per_request", "rival_profit_per_request", "Profit", "Profit per request", False),
        ("own_share", "rival_share", "Market share", "Completed choice share (%)", True),
        ("fulfillment", None, "Fulfillment", "Firm 1 fulfillment (%)", True),
        ("firm1_level", "firm2_level_effective", "Effective price level", "Price multiplier", False),
    ]
    for row_index, variant in enumerate(VARIANTS):
        for col_index, (key_f1, key_f2, title, ylabel, percent) in enumerate(panel_defs):
            ax = axes[row_index, col_index]
            for policy_label, color in POLICY_COLORS.items():
                plot_series(
                    ax,
                    aggregate_policy_matrices(data, variant, policy_label, key_f1),
                    color,
                    FIRM_STYLES["Firm 1"],
                    f"{policy_label} · Firm 1",
                    percent=percent,
                )
                if key_f2:
                    plot_series(
                        ax,
                        aggregate_policy_matrices(data, variant, policy_label, key_f2),
                        color,
                        FIRM_STYLES["Firm 2"],
                        f"{policy_label} · Firm 2",
                        percent=percent,
                    )
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.set_ylabel(ylabel, fontsize=9)
            style_axis(ax)
            if row_index == 2:
                ax.set_xlabel("Evaluation day", fontsize=9)
        axes[row_index, 0].text(
            -0.28, 0.5, variant,
            transform=axes[row_index, 0].transAxes,
            rotation=90, ha="center", va="center",
            fontsize=12, fontweight="bold",
        )

    policy_handles = [
        Line2D([0], [0], color=color, lw=2.2, label=label)
        for label, color in POLICY_COLORS.items()
    ]
    firm_handles = [
        Line2D([0], [0], color="#222222", lw=1.8, linestyle=style, label=firm)
        for firm, style in FIRM_STYLES.items()
    ]
    fig.legend(
        handles=policy_handles + firm_handles,
        loc="upper center", bbox_to_anchor=(0.5, 0.965),
        ncol=5, frameon=False, fontsize=9,
    )
    fig.text(
        0.5, 0.015,
        "Daily mean across 70 scenario-episodes (7 scenarios × 10 openings); translucent bands are 95% CIs. "
        "Firm 2 fulfillment and both firms' revenue were not recorded in the evaluation traces.",
        ha="center", va="bottom", fontsize=9, color="#444444",
    )
    output = out_dir / "episode_paths_all_scenarios.png"
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(output)


def summary_rows(data):
    rows = []
    for variant in VARIANTS:
        for scenario in SCENARIOS:
            policies, static_label = chosen_policies(data, variant, scenario)
            for policy_label, item in policies.items():
                payload = item["payload"]
                row = {
                    "variant": variant,
                    "scenario": scenario,
                    "policy": policy_label,
                    "best_static_level": static_label.replace("Static ", "") if policy_label == "Best static" else None,
                }
                for out_key, trace_key in [
                    ("firm1_profit_per_request", "profit_per_request"),
                    ("firm2_profit_per_request", "rival_profit_per_request"),
                    ("firm1_market_share", "own_share"),
                    ("firm2_market_share", "rival_share"),
                    ("firm1_fulfillment", "fulfillment"),
                    ("firm1_price_level", "firm1_level"),
                    ("firm2_tariff_level", "firm2_level_effective"),
                ]:
                    matrix = episode_daily(payload, trace_key)
                    row[out_key] = float(np.nanmean(matrix)) if matrix.size else None
                rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    out_dir = (args.out_dir or (run_dir / "figures" / "episode_averages")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_evaluations(run_dir / "evaluate")
    for scenario in SCENARIOS:
        plot_scenario(data, scenario, out_dir)
    plot_aggregate(data, out_dir)

    summary_path = out_dir / "episode_average_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_rows(data), handle, indent=2)
    print(summary_path)


if __name__ == "__main__":
    main()
