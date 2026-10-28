#!/usr/bin/env python3
"""Episode-paired policy/rival summaries with explicit exposure labels."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/rideshare-matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "scripts"))
from diagnostic_logging import atomic_json  # noqa: E402
from v26r_cases import CASES  # noqa: E402


BASE_RIVAL_GROUPS = {
    "static": ("static",),
    "catch_up": (
        "adaptive_best_response_memory",
        "adaptive_best_response_aggressive",
        "price_gap_catchup",
        "price_gap_catchup_fast",
    ),
    "legacy_objective": (
        "pi_price_gap",
        "mpc_grid",
        "region_supply_demand",
        "surge_driver_incentive",
        "queue_service_threshold",
    ),
    "profit_optimized": (
        "profit_opt_pi_price_gap",
        "profit_opt_mpc_grid",
        "profit_opt_region_supply_demand",
        "profit_opt_surge_driver_incentive",
        "profit_opt_queue_service_threshold",
    ),
    "profit_best_response": (
        "profit_best_response_myopic",
        "profit_best_response_forward",
        "profit_best_response_anticipating",
    ),
}
TRAINING_EXPOSED_RIVALS = (
    BASE_RIVAL_GROUPS["catch_up"]
    + BASE_RIVAL_GROUPS["profit_optimized"]
    + BASE_RIVAL_GROUPS["profit_best_response"]
)
GENUINELY_HELD_OUT_RIVALS = (
    BASE_RIVAL_GROUPS["static"] + BASE_RIVAL_GROUPS["legacy_objective"]
)
# A static rival has no controller parameters to perturb.  Each older
# rule-based controller receives an out-of-distribution parameter draw that is
# deterministic and common across every Firm 1 comparator.
PERTURBED_HELD_OUT_RIVALS = tuple(
    f"{rival}@perturbed0" for rival in BASE_RIVAL_GROUPS["legacy_objective"]
)
RIVAL_GROUPS = {
    **BASE_RIVAL_GROUPS,
    "legacy_objective_perturbed": PERTURBED_HELD_OUT_RIVALS,
}
RIVALS = tuple(rival for values in RIVAL_GROUPS.values() for rival in values)
RIVAL_GROUP = {rival: group for group, values in RIVAL_GROUPS.items() for rival in values}
RIVAL_EXPOSURE = {
    **{rival: "training_exposed" for rival in TRAINING_EXPOSED_RIVALS},
    **{rival: "genuinely_held_out" for rival in GENUINELY_HELD_OUT_RIVALS},
    **{
        rival: "genuinely_held_out_perturbed"
        for rival in PERTURBED_HELD_OUT_RIVALS
    },
}
METRICS = (
    "objective_reward",
    "firm1_profit_per_request",
    "firm2_profit_per_request",
    "profit_lead_per_request",
    "firm1_revenue_per_request",
    "firm2_revenue_per_request",
    "firm1_fulfillment",
    "firm2_fulfillment",
    "firm1_market_share",
    "firm2_market_share",
    "firm1_completed_request_share",
    "firm2_completed_request_share",
    "outside_or_unfulfilled_share",
    "firm1_tariff_level",
    "firm2_tariff_level",
)


def policy_class(spec: str) -> str:
    if spec == "rl_actor":
        return "adaptive_learned_policy"
    if spec == "rl_actor_frozen":
        return "learned_opening_action_then_frozen"
    if spec == "opening_hold":
        return "randomized_opening_tariff_frozen"
    if spec.startswith("static:"):
        return "fixed_whole_tariff"
    if spec.startswith("static_tariff:"):
        return "revision15b_searched_differentiated_tariff"
    return "other_comparator"


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def rival_of(rollout: str) -> str:
    return str(rollout).rsplit("__opening_", 1)[0]


def row_metrics(row: dict[str, Any]) -> dict[str, float | None]:
    own = finite(row.get("own_share"))
    rival = finite(row.get("rival_share"))
    total = (own or 0.0) + (rival or 0.0)
    market_share = (own / total) if own is not None and total > 0.0 else finite(row.get("own_market_share"))
    level = finite(row.get("firm1_tariff_level"))
    if level is None:
        level = finite(row.get("firm1_level"))
    return {
        "objective_reward": finite(row.get("reward")),
        "firm1_profit_per_request": finite(row.get("profit_per_request")),
        "firm2_profit_per_request": finite(row.get("rival_profit_per_request")),
        "profit_lead_per_request": finite(row.get("profit_lead")),
        "firm1_revenue_per_request": finite(row.get("revenue_per_request")),
        "firm2_revenue_per_request": finite(row.get("rival_revenue_per_request")),
        "firm1_fulfillment": finite(row.get("fulfillment")),
        "firm2_fulfillment": finite(row.get("rival_fulfillment")),
        "firm1_market_share": market_share,
        "firm2_market_share": (1.0 - market_share) if market_share is not None else None,
        # Absolute completed shares use all incoming requests as denominator,
        # exactly matching the Stage 4 reward.  Keep the two-firm-normalized
        # market share above as a separate, explicitly named measure.
        "firm1_completed_request_share": own,
        "firm2_completed_request_share": rival,
        "outside_or_unfulfilled_share": (
            max(0.0, 1.0 - total)
            if own is not None and rival is not None
            else None
        ),
        "firm1_tariff_level": level,
        "firm2_tariff_level": finite(row.get("firm2_level_effective", row.get("firm2_level"))),
    }


def average(values: Iterable[float | None]) -> float | None:
    clean = [value for value in values if value is not None and math.isfinite(value)]
    return mean(clean) if clean else None


def episode_switch_rate(rows: list[dict[str, Any]], steps_per_day: int) -> float | None:
    daily: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        value = row_metrics(row)["firm1_tariff_level"]
        if value is not None:
            daily[int(row["step"]) // steps_per_day].append(value)
    levels = [mean(daily[day]) for day in sorted(daily)]
    if len(levels) < 2:
        return None
    return sum(abs(right - left) > 1e-8 for left, right in zip(levels, levels[1:])) / (len(levels) - 1)


def summarize_case(path: Path, case_name: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    data = json.loads(path.read_text())
    traces = dict(data.get("traces", {}))
    required_specs = {"rl_actor", "rl_actor_frozen", "opening_hold"}
    if required_specs - set(traces):
        raise ValueError(
            f"{path}: missing policy comparators {sorted(required_specs - set(traces))}"
        )
    if not any(spec.startswith("static:") for spec in traces):
        raise ValueError(f"{path}: static tariff grid is missing")
    steps_per_day = int(data["meta"]["steps_per_day"])
    horizon_days = int(data["meta"]["horizon_steps"]) // steps_per_day
    openings = int(data["meta"]["openings"])
    burnin = int(CASES[case_name].get("burnin", 5))
    late_start = max(burnin, int(math.floor(0.75 * horizon_days)))
    by_episode: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    by_day: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    expected_rollouts = len(RIVALS) * openings
    expected_steps = expected_rollouts * horizon_days * steps_per_day
    for spec, rows in traces.items():
        if len(rows) != expected_steps:
            raise ValueError(
                f"{path}: {spec} has {len(rows)} trace steps, expected {expected_steps}"
            )
        observed_rollouts = {str(row["rollout"]) for row in rows}
        if len(observed_rollouts) != expected_rollouts:
            raise ValueError(
                f"{path}: {spec} has {len(observed_rollouts)} rollouts, "
                f"expected {expected_rollouts}"
            )
        for row in rows:
            rival = rival_of(row["rollout"])
            if rival not in RIVAL_GROUP:
                raise ValueError(f"{path}: unexpected rival {rival!r}")
            opening = str(row["rollout"])
            day = int(row["step"]) // steps_per_day
            by_episode[(str(spec), rival, opening)].append(row)
            by_day[(str(spec), rival, opening, day)].append(row)

    episode_rows: list[dict[str, Any]] = []
    windows = {"full": (0, horizon_days), "post_burnin": (burnin, horizon_days), "late": (late_start, horizon_days)}
    for (spec, rival, opening), episode in by_episode.items():
        for window, (start, end) in windows.items():
            selected = [row for row in episode if start <= int(row["step"]) // steps_per_day < end]
            metrics = [row_metrics(row) for row in selected]
            output = {
                "case": case_name,
                "policy_spec": spec,
                "policy_class": policy_class(spec),
                "rival": rival,
                "rival_group": RIVAL_GROUP[rival],
                "evaluation_exposure": RIVAL_EXPOSURE[rival],
                "opening": opening,
                "window": window,
                "start_day": start,
                "end_day": end,
                "steps": len(selected),
                **{name: average(item[name] for item in metrics) for name in METRICS},
            }
            output["policy_switch_rate"] = episode_switch_rate(selected, steps_per_day)
            episode_rows.append(output)

    daily_episode_rows: list[dict[str, Any]] = []
    for (spec, rival, opening, day), sample in by_day.items():
        metrics = [row_metrics(row) for row in sample]
        daily_episode_rows.append({
            "case": case_name,
            "policy_spec": spec,
            "policy_class": policy_class(spec),
            "rival": rival,
            "rival_group": RIVAL_GROUP[rival],
            "evaluation_exposure": RIVAL_EXPOSURE[rival],
            "opening": opening,
            "day": day,
            **{name: average(item[name] for item in metrics) for name in METRICS},
        })
    return episode_rows, daily_episode_rows


def group_means(rows: list[dict[str, Any]], keys: tuple[str, ...], values: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    output = []
    for key_values, sample in sorted(groups.items()):
        item = dict(zip(keys, key_values))
        item["episodes"] = len(sample)
        for value in values:
            item[value] = average(finite(row.get(value)) for row in sample)
        output.append(item)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty summary {path}")
    keys = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_overview(rows: list[dict[str, Any]], path: Path, city: str) -> None:
    chosen = [
        row for row in rows
        if row["window"] == "post_burnin" and row["policy_spec"] == "rl_actor"
    ]
    aggregate = group_means(chosen, ("rival", "rival_group"), METRICS + ("policy_switch_rate",))
    aggregate.sort(key=lambda row: RIVALS.index(row["rival"]))
    labels = [row["rival"] for row in aggregate]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), constrained_layout=True)
    panels = (
        ("firm1_profit_per_request", "firm2_profit_per_request", "Contribution profit / request"),
        ("firm1_revenue_per_request", "firm2_revenue_per_request", "Revenue / request"),
        ("firm1_market_share", "firm2_market_share", "Two-firm completed market share"),
        ("firm1_fulfillment", "firm2_fulfillment", "Fulfillment rate"),
    )
    for ax, (left, right, title) in zip(axes.flat, panels):
        ax.plot(x, [row[left] for row in aggregate], marker="o", linewidth=1.5, label="Firm 1 (RL)")
        ax.plot(x, [row[right] for row in aggregate], marker="s", linewidth=1.2, label="Firm 2 benchmark")
        ax.set_title(title)
        ax.set_xticks(x, labels, rotation=55, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
    fig.suptitle(
        f"{city}: adaptive learned policy against every evaluation rival "
        "(episode-paired, post-burn-in)"
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_heatmaps(rows: list[dict[str, Any]], path: Path, city: str) -> None:
    chosen = [
        row for row in rows
        if row["window"] == "post_burnin" and row["policy_spec"] == "rl_actor"
    ]
    aggregate = group_means(chosen, ("case", "rival"), ("profit_lead_per_request", "firm1_market_share", "policy_switch_rate"))
    lookup = {(row["case"], row["rival"]): row for row in aggregate}
    # The bounded Revision 15d screen deliberately evaluates only six of the
    # twelve configured scenarios before the extension fills in the rest.
    # Plot only the cases present in this summary, preserving the canonical
    # CASES order.  Indexing every configured case here made an otherwise
    # successful bounded evaluation fail while looking up the first omitted
    # A2/static cell.
    observed_cases = {str(row["case"]) for row in chosen}
    cases = [case for case in CASES if case in observed_cases]
    unexpected_cases = sorted(observed_cases - set(CASES))
    if unexpected_cases:
        raise ValueError(f"heatmap contains unknown cases: {unexpected_cases}")
    if not cases:
        raise ValueError("heatmap has no post-burn-in rl_actor cases")
    missing_cells = [
        (case, rival)
        for case in cases
        for rival in RIVALS
        if (case, rival) not in lookup
    ]
    if missing_cells:
        raise ValueError(
            "heatmap is missing evaluated case/rival cells: "
            f"{missing_cells[:10]}"
        )
    fig, axes = plt.subplots(3, 1, figsize=(19, 12), constrained_layout=True)
    panels = (
        ("profit_lead_per_request", "Firm 1 minus Firm 2 profit / request", "coolwarm"),
        (
            "firm1_market_share",
            "Firm 1 share of two-firm completed rides",
            "viridis",
        ),
        ("policy_switch_rate", "Firm 1 daily tariff switch rate", "magma"),
    )
    for ax, (metric, title, cmap) in zip(axes, panels):
        matrix = np.asarray([[lookup[(case, rival)][metric] for rival in RIVALS] for case in cases], dtype=float)
        image = ax.imshow(matrix, aspect="auto", cmap=cmap)
        ax.set_yticks(np.arange(len(cases)), cases)
        ax.set_xticks(np.arange(len(RIVALS)), RIVALS, rotation=55, ha="right")
        ax.set_title(title)
        fig.colorbar(image, ax=ax, fraction=0.015, pad=0.01)
    fig.suptitle(f"{city}: evaluation cases × opponents (adaptive learned policy)")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_case_timeseries(daily: list[dict[str, Any]], case_name: str, path: Path, city: str) -> None:
    case_rows = [
        row for row in daily
        if row["case"] == case_name and row["policy_spec"] == "rl_actor"
    ]
    category = group_means(case_rows, ("rival_group", "day"), METRICS)
    groups = list(RIVAL_GROUPS)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    panels = (
        ("firm1_profit_per_request", "firm2_profit_per_request", "Profit / request"),
        ("firm1_revenue_per_request", "firm2_revenue_per_request", "Revenue / request"),
        (
            "firm1_market_share",
            "firm2_market_share",
            "Two-firm completed-ride share",
        ),
        ("firm1_fulfillment", "firm2_fulfillment", "Fulfillment"),
    )
    for ax, (left, right, title) in zip(axes.flat, panels):
        for group in groups:
            sample = sorted((row for row in category if row["rival_group"] == group), key=lambda row: row["day"])
            days = [row["day"] for row in sample]
            line = ax.plot(days, [row[left] for row in sample], linewidth=1.3, label=f"{group}: F1")[0]
            ax.plot(days, [row[right] for row in sample], linewidth=1.0, linestyle="--", color=line.get_color(), label=f"{group}: F2")
        ax.set_title(title)
        ax.set_xlabel("Evaluation day")
        ax.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=5)
    fig.suptitle(f"{city} — {case_name}: episode-averaged trajectories by opponent family")
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def comparator_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Paired dynamic/frozen/fixed summaries without hiding static-grid search.

    ``best_static_spec`` is selected on the same reported evaluation cells and
    is therefore explicitly an ex-post grid oracle, not a deployable held-out
    policy-selection result.  Every individual static grid row remains in the
    main CSV so readers can inspect the complete comparison.
    """
    aggregates = group_means(
        rows,
        (
            "case", "rival", "rival_group", "evaluation_exposure", "window",
            "policy_spec", "policy_class",
        ),
        METRICS + ("policy_switch_rate",),
    )
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in aggregates:
        grouped[(row["case"], row["rival"], row["window"])].append(row)
    output: list[dict[str, Any]] = []
    for (case, rival, window), sample in sorted(grouped.items()):
        by_spec = {row["policy_spec"]: row for row in sample}
        required = {"rl_actor", "rl_actor_frozen", "opening_hold"}
        if required - set(by_spec):
            raise ValueError(
                f"{case}/{rival}/{window}: missing {sorted(required - set(by_spec))}"
            )
        static = [row for row in sample if row["policy_spec"].startswith("static:")]
        if not static:
            raise ValueError(f"{case}/{rival}/{window}: missing fixed tariff grid")
        best_static = min(
            static,
            key=lambda row: (
                -float(row["objective_reward"]),
                float(row["policy_spec"].split(":", 1)[1]),
            ),
        )
        dynamic = by_spec["rl_actor"]
        frozen = by_spec["rl_actor_frozen"]
        opening = by_spec["opening_hold"]
        searched = [
            row for row in sample
            if row["policy_spec"].startswith("static_tariff:")
        ]
        if len(searched) != 1:
            raise ValueError(
                f"{case}/{rival}/{window}: expected one searched differentiated "
                f"tariff, found {len(searched)}"
            )
        searched_tariff = searched[0]
        item: dict[str, Any] = {
            "case": case,
            "rival": rival,
            "rival_group": dynamic["rival_group"],
            "evaluation_exposure": dynamic["evaluation_exposure"],
            "window": window,
            "best_static_spec": best_static["policy_spec"],
            "best_static_selection": "ex_post_same_evaluation_cell_grid_oracle",
        }
        for metric in METRICS + ("policy_switch_rate",):
            item[f"adaptive_{metric}"] = dynamic.get(metric)
            item[f"frozen_actor_{metric}"] = frozen.get(metric)
            item[f"opening_hold_{metric}"] = opening.get(metric)
            item[f"best_static_{metric}"] = best_static.get(metric)
            item[f"r15b_searched_{metric}"] = searched_tariff.get(metric)
            if finite(dynamic.get(metric)) is not None:
                item[f"adaptive_minus_frozen_{metric}"] = (
                    float(dynamic[metric]) - float(frozen[metric])
                    if finite(frozen.get(metric)) is not None else None
                )
                item[f"adaptive_minus_best_static_{metric}"] = (
                    float(dynamic[metric]) - float(best_static[metric])
                    if finite(best_static.get(metric)) is not None else None
                )
            else:
                item[f"adaptive_minus_frozen_{metric}"] = None
                item[f"adaptive_minus_best_static_{metric}"] = None
        output.append(item)
    return output


def plot_policy_comparators(rows: list[dict[str, Any]], path: Path, city: str) -> None:
    selected = [row for row in rows if row["window"] == "post_burnin"]
    labels = [
        "Adaptive actor", "Actor frozen after first move", "Opening hold",
        "Best fixed grid*", "R15b searched tariff",
    ]
    prefixes = (
        "adaptive", "frozen_actor", "opening_hold", "best_static",
        "r15b_searched",
    )
    panels = (
        ("objective_reward", "Objective reward"),
        ("firm1_profit_per_request", "Firm 1 profit / request"),
        ("firm1_market_share", "Firm 1 two-firm market share"),
        ("firm1_fulfillment", "Firm 1 fulfillment"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    for ax, (metric, title) in zip(axes.flat, panels):
        values = [average(finite(row.get(f"{prefix}_{metric}")) for row in selected) for prefix in prefixes]
        ax.bar(
            np.arange(len(labels)), values,
            color=("#2f6f9f", "#7da7c4", "#999999", "#d9903d", "#7a5195"),
        )
        ax.set_xticks(np.arange(len(labels)), labels, rotation=18, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(
        f"{city}: matched learned-policy, frozen-policy and fixed-tariff controls\n"
        "*best fixed grid is an ex-post evaluation-cell oracle, not held-out selection"
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", required=True, type=Path)
    parser.add_argument("--slug", required=True)
    parser.add_argument("--city", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--cases",
        default="all",
        help="Configured case name, comma-separated subset, or 'all' (default).",
    )
    parser.add_argument(
        "--openings",
        type=int,
        default=10,
        help="Expected independent openings per case/rival/policy cell.",
    )
    args = parser.parse_args()
    selected_cases = (
        list(CASES)
        if args.cases == "all"
        else [name.strip() for name in args.cases.split(",") if name.strip()]
    )
    unknown_cases = [name for name in selected_cases if name not in CASES]
    if not selected_cases or unknown_cases or args.openings < 1:
        parser.error(
            "--cases must select configured cases and --openings must be "
            f"positive; unknown={unknown_cases}"
        )
    episode_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    for case_name in selected_cases:
        path = args.evaluation_root / args.slug / case_name / "eval" / "sweep.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        case_episode, case_daily = summarize_case(path, case_name)
        episode_rows.extend(case_episode)
        daily_rows.extend(case_daily)

    observed_specs = sorted({row["policy_spec"] for row in episode_rows})
    expected_episodes = (
        len(selected_cases)
        * len(RIVALS)
        * int(args.openings)
        * 3
        * len(observed_specs)
    )
    if len(episode_rows) != expected_episodes:
        raise ValueError(
            f"incomplete evaluation: {len(episode_rows)} episode-window rows, "
            f"expected {expected_episodes}"
        )
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "benchmark_episode_metrics.csv", episode_rows)
    aggregates = group_means(
        episode_rows,
        (
            "case", "policy_spec", "policy_class", "rival", "rival_group",
            "evaluation_exposure", "window",
        ),
        METRICS + ("policy_switch_rate",),
    )
    write_csv(output / "benchmark_summary.csv", aggregates)
    daily_averages = group_means(
        daily_rows,
        (
            "case", "policy_spec", "policy_class", "rival", "rival_group",
            "evaluation_exposure", "day",
        ),
        METRICS,
    )
    write_csv(output / "benchmark_daily_trajectories.csv", daily_averages)

    comparisons = comparator_rows(episode_rows)
    write_csv(output / "policy_comparator_summary.csv", comparisons)

    post = [
        row for row in episode_rows
        if row["window"] == "post_burnin" and row["policy_spec"] == "rl_actor"
    ]
    overall = group_means(
        post,
        ("rival", "rival_group", "evaluation_exposure"),
        METRICS + ("policy_switch_rate",),
    )
    overall.sort(key=lambda row: RIVALS.index(row["rival"]))
    atomic_json(output / "benchmark_summary.json", {
        "city": args.city,
        "slug": args.slug,
        "cases": list(selected_cases),
        "policy_specs": observed_specs,
        "rival_groups": {key: list(value) for key, value in RIVAL_GROUPS.items()},
        "rival_exposure": dict(RIVAL_EXPOSURE),
        "training_exposed_rivals": list(TRAINING_EXPOSED_RIVALS),
        "genuinely_held_out_rivals": list(GENUINELY_HELD_OUT_RIVALS),
        "perturbed_held_out_rivals": list(PERTURBED_HELD_OUT_RIVALS),
        "evaluation_protocol": (
            "mean within each opening, then mean across openings and evaluation cases; "
            "training-exposed and genuinely-held-out rivals are reported separately"
        ),
        "fixed_tariff_comparator_contract": (
            "all static grid points are retained; best_static in policy_comparator_summary "
            "is an ex-post same-cell grid oracle and is not called held-out selection"
        ),
        "share_metric_contract": {
            "firm1_market_share": "Firm 1 / (Firm 1 + Firm 2 completed rides)",
            "firm2_market_share": "Firm 2 / (Firm 1 + Firm 2 completed rides)",
            "firm1_completed_request_share": "Firm 1 completed rides / all incoming requests",
            "firm2_completed_request_share": "Firm 2 completed rides / all incoming requests",
            "outside_or_unfulfilled_share": "1 - both firms' completed-request shares",
        },
        "overall_post_burnin": overall,
    })
    plot_overview(episode_rows, output / "benchmark_overview.png", args.city)
    plot_heatmaps(episode_rows, output / "benchmark_case_heatmaps.png", args.city)
    plot_policy_comparators(
        comparisons, output / "policy_comparator_overview.png", args.city
    )
    case_dir = output / "case_trajectories"
    case_dir.mkdir(exist_ok=True)
    for case_name in selected_cases:
        plot_case_timeseries(daily_averages, case_name, case_dir / f"{case_name}.png", args.city)
    print(json.dumps({
        "status": "complete", "city": args.city, "rivals": len(RIVALS),
        "policy_specs": len(observed_specs), "cases": len(selected_cases),
        "output": str(output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
