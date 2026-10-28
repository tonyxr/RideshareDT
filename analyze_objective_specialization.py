from __future__ import annotations

"""Compare profit- and competitiveness-policy benchmark outcomes."""

import argparse
import csv
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


BENCHMARKS = (
    "pi_price_gap",
    "mpc_grid",
    "region_supply_demand",
    "surge_driver_incentive",
    "queue_service_threshold",
)


def _read_trajectory(prefix: str, benchmark: str) -> List[Dict[str, str]]:
    path = f"{prefix}_benchmark_{benchmark}_trajectory.csv"
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mean(rows: List[Dict[str, str]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profit_prefix", required=True)
    parser.add_argument("--competitiveness_prefix", required=True)
    parser.add_argument("--out_prefix", required=True)
    args = parser.parse_args()

    policy_prefixes = {
        "Profit policy": args.profit_prefix,
        "Competitiveness policy": args.competitiveness_prefix,
    }
    summary: List[Dict[str, object]] = []
    for benchmark in BENCHMARKS:
        for policy, prefix in policy_prefixes.items():
            rows = _read_trajectory(prefix, benchmark)
            firm1_profit = _mean(rows, "firm1_profit_per_request")
            firm2_profit = _mean(rows, "firm2_profit_per_request")
            profit_gap = _mean(rows, "profit_gap_per_request")
            firm1_share = _mean(rows, "firm1_completed_share")
            firm2_share = _mean(rows, "firm2_completed_share")
            share_gap = _mean(rows, "completed_share_gap")
            summary.append(
                {
                    "benchmark": benchmark,
                    "policy": policy,
                    "mean_firm1_profit_per_request": firm1_profit,
                    "mean_firm2_profit_per_request": firm2_profit,
                    "mean_profit_gap_per_request": profit_gap,
                    "mean_firm1_completed_share": firm1_share,
                    "mean_firm2_completed_share": firm2_share,
                    "mean_completed_share_gap": share_gap,
                    "profit_dominance_rate": float(
                        np.mean(
                            [
                                float(row["profit_gap_per_request"]) > 0.0
                                for row in rows
                            ]
                        )
                    ),
                    "completed_share_dominance_rate": float(
                        np.mean(
                            [
                                float(row["completed_share_gap"]) > 0.0
                                for row in rows
                            ]
                        )
                    ),
                }
            )

    csv_path = f"{args.out_prefix}_summary.csv"
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    labels = [benchmark.replace("_", "\n") for benchmark in BENCHMARKS]
    x = np.arange(len(BENCHMARKS), dtype=float)
    width = 0.36
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    for offset, (policy, color) in zip(
        (-width / 2, width / 2),
        (("Profit policy", "tab:blue"), ("Competitiveness policy", "tab:orange")),
    ):
        selected = [row for row in summary if row["policy"] == policy]
        axes[0].bar(
            x + offset,
            [float(row["mean_firm1_profit_per_request"]) for row in selected],
            width,
            label=policy,
            color=color,
        )
        axes[1].bar(
            x + offset,
            [100.0 * float(row["mean_completed_share_gap"]) for row in selected],
            width,
            label=policy,
            color=color,
        )
    axes[1].axhline(0.0, color="black", linewidth=0.9)
    axes[0].set_ylabel("Firm 1 contribution profit / request ($)")
    axes[1].set_ylabel("Firm 1 completed-share lead (pp)")
    axes[1].set_xticks(x, labels)
    for axis in axes:
        axis.grid(axis="y", linestyle="--", alpha=0.25)
        axis.legend(loc="best")
    fig.suptitle(
        "Economic Objective Specialization Across Five NYC Benchmarks\n"
        "Profit policy: 100% profit terms | Competitiveness policy: 90% completed-share terms"
    )
    fig.tight_layout()
    plot_path = f"{args.out_prefix}_performance.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {csv_path}")
    print(f"Saved graph -> {plot_path}")


if __name__ == "__main__":
    main()
