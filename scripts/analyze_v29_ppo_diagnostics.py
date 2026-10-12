#!/usr/bin/env python3
"""Create a compact, evidence-first diagnostic report for one v29 PPO run."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np


COEFFICIENTS = (
    "base_fare", "per_minute", "per_mile", "booking_fee", "airport_fee"
)


def number(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")


def finite(values: Iterable[float]) -> list[float]:
    return [float(value) for value in values if math.isfinite(float(value))]


def stats(values: Iterable[float]) -> dict[str, float | None]:
    resolved = finite(values)
    if not resolved:
        return {"min": None, "mean": None, "max": None}
    return {
        "min": min(resolved),
        "mean": float(np.mean(resolved)),
        "max": max(resolved),
    }


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize(run_dir: Path) -> dict[str, Any]:
    training_path = next(run_dir.glob("*_training_diagnostics.csv"))
    evaluation_path = next(run_dir.glob("*_evaluation_diagnostics.csv"))
    config_path = next(run_dir.glob("*_run_config.json"))
    training = load_rows(training_path)
    evaluation = load_rows(evaluation_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    update_rows = [
        row for row in training
        if row.get("ppo_update_performed", "").lower() == "true"
    ]
    stage_names = list(dict.fromkeys(row.get("training_stage", "") for row in training))
    stages: dict[str, Any] = {}
    for stage in stage_names:
        rows = [row for row in training if row.get("training_stage") == stage]
        stages[stage] = {
            "days": len(rows),
            "mean_reward": stats(number(row, "avg_reward") for row in rows)["mean"],
            "mean_profit_advantage": stats(
                number(row, "avg_profit_advantage_per_request") for row in rows
            )["mean"],
            "mean_completed_share_advantage": stats(
                number(row, "avg_completed_share")
                - number(row, "avg_firm2_completed_share")
                for row in rows
            )["mean"],
        }

    actions = Counter(int(float(row["last_action"])) for row in training)
    raw_actions = Counter(int(float(row["raw_policy_top_action"])) for row in training)
    action_labels: dict[int, str] = {}
    for row in training:
        action = int(float(row["last_action"]))
        if action not in action_labels:
            steps = json.loads(row.get("last_action_steps") or "{}")
            active = [(key, int(value)) for key, value in steps.items() if int(value)]
            action_labels[action] = (
                "hold" if not active else
                f"{active[0][0]}:{'up' if active[0][1] > 0 else 'down'}"
            )

    validations = []
    for row in training:
        if math.isfinite(number(row, "validation_score")):
            validations.append({
                "day": int(float(row["batch"])) + 1,
                "stage": row["training_stage"],
                "score": number(row, "validation_score"),
                "reward": number(row, "validation_reward"),
                "profit_per_request": number(row, "validation_profit_per_request"),
                "rival_profit_per_request": number(row, "validation_firm2_profit_per_request"),
                "completed_share": number(row, "validation_completed_share"),
                "rival_completed_share": number(row, "validation_firm2_completed_share"),
                "action_diversity": number(row, "validation_action_diversity"),
                "action_switch_rate": number(row, "validation_action_switch_rate"),
                "critic_td_explained_variance": number(
                    row, "validation_critic_td_explained_variance"
                ),
                "critic_td_mae": number(row, "validation_critic_td_mae"),
                "eligible": row.get("policy_checkpoint_eligible", "").lower() == "true",
            })

    coefficient_summary = {}
    for key in COEFFICIENTS:
        values = [number(row, f"firm1_{key}") for row in training]
        coefficient_summary[key] = {
            **stats(values),
            "final": values[-1] if values else None,
        }

    tail = evaluation[-max(1, len(evaluation) // 4):]
    certification = dict(config.get("training_certification") or {})
    backtest = dict(config.get("competitive_backtest") or {})
    critic_ev = stats(number(row, "ppo_explained_variance") for row in update_rows)
    sensitivity = stats(number(row, "ppo_state_action_sensitivity") for row in update_rows)
    raw_concentration = stats(
        number(row, "ppo_raw_argmax_concentration") for row in update_rows
    )
    findings = []
    if (critic_ev["mean"] or 0.0) < 0.10:
        findings.append(
            "critic_failed_to_explain_returns: mean PPO explained variance below 0.10"
        )
    if (sensitivity["mean"] or 0.0) < 0.01:
        findings.append(
            "policy_nearly_state_independent: mean state/action sensitivity below 0.01"
        )
    if (raw_concentration["mean"] or 0.0) > 0.80:
        findings.append(
            "raw_argmax_collapse: one raw action dominated most rollout states"
        )
    if validations and validations[-1]["profit_per_request"] < 0.0:
        findings.append("final_validation_profit_negative")
    if backtest.get("passed") is False:
        findings.append("durability_audit_failed")

    return {
        "run_directory": str(run_dir.resolve()),
        "training_days": len(training),
        "evaluation_steps": len(evaluation),
        "stages": stages,
        "ppo": {
            "fresh_update_batches": len(update_rows),
            "actor_optimizer_steps": sum(
                int(number(row, "ppo_actor_optimizer_steps")) for row in update_rows
            ),
            "critic_optimizer_steps": sum(
                int(number(row, "ppo_critic_optimizer_steps")) for row in update_rows
            ),
            "explained_variance": critic_ev,
            "value_loss": stats(number(row, "ppo_value_loss") for row in update_rows),
            "approx_kl": stats(number(row, "ppo_approx_kl") for row in update_rows),
            "clip_fraction": stats(number(row, "ppo_clipfrac") for row in update_rows),
            "policy_entropy_fraction": stats(
                number(row, "ppo_policy_entropy_fraction") for row in update_rows
            ),
            "state_action_sensitivity": sensitivity,
            "raw_argmax_concentration": raw_concentration,
            "forced_uniform_exploration_rate": stats(
                number(row, "ppo_exploration_rate") for row in update_rows
            ),
        },
        "actions": {
            "executed": {
                action_labels.get(action, str(action)): count
                for action, count in sorted(actions.items())
            },
            "raw_argmax": {
                action_labels.get(action, str(action)): count
                for action, count in sorted(raw_actions.items())
            },
        },
        "coefficients": coefficient_summary,
        "validations": validations,
        "evaluation": {
            "mean_profit_per_request": stats(
                number(row, "rl_profit") for row in evaluation
            )["mean"],
            "mean_rival_profit_per_request": stats(
                number(row, "heuristic_profit") for row in evaluation
            )["mean"],
            "mean_completed_share": stats(
                number(row, "rl_market_share") for row in evaluation
            )["mean"],
            "mean_rival_completed_share": stats(
                number(row, "heuristic_market_share") for row in evaluation
            )["mean"],
            "tail_completed_share_advantage": stats(
                number(row, "rl_market_share") - number(row, "heuristic_market_share")
                for row in tail
            )["mean"],
        },
        "certification_diagnostics": certification,
        "durability_diagnostics": backtest,
        "primary_findings": findings,
    }


def render_plot(run_dir: Path, report: dict[str, Any], output: Path) -> None:
    training = load_rows(next(run_dir.glob("*_training_diagnostics.csv")))
    x = np.arange(1, len(training) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    axes[0, 0].plot(x, [number(r, "reward_moving_avg20") for r in training], label="reward MA20")
    axes[0, 0].plot(x, [number(r, "avg_profit_advantage_per_request") for r in training], alpha=.65, label="profit advantage")
    axes[0, 0].plot(x, [number(r, "avg_completed_share") - number(r, "avg_firm2_completed_share") for r in training], alpha=.65, label="completed-share advantage")
    axes[0, 0].axhline(0, color="black", lw=.7)
    axes[0, 0].set_title("Training objectives")
    axes[0, 0].legend(fontsize=8)
    for key in COEFFICIENTS:
        axes[0, 1].plot(x, [number(r, f"firm1_{key}") for r in training], label=key)
    axes[0, 1].set_title("Independent coefficient paths")
    axes[0, 1].legend(fontsize=8, ncol=2)
    updates = [r for r in training if r.get("ppo_update_performed", "").lower() == "true"]
    ux = [int(float(r["batch"])) + 1 for r in updates]
    axes[1, 0].plot(ux, [number(r, "ppo_explained_variance") for r in updates], label="critic explained variance")
    axes[1, 0].plot(ux, [number(r, "ppo_state_action_sensitivity") for r in updates], label="state/action sensitivity")
    axes[1, 0].axhline(0, color="black", lw=.7)
    axes[1, 0].set_title("Learning signal")
    axes[1, 0].legend(fontsize=8)
    labels = list(report["actions"]["raw_argmax"])
    values = [report["actions"]["raw_argmax"][label] for label in labels]
    axes[1, 1].barh(labels, values)
    axes[1, 1].set_title("Raw actor argmax by training day")
    for axis in axes.flat:
        axis.grid(alpha=.2)
    fig.suptitle("v29 PPO diagnostic summary", fontsize=15)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    report = summarize(run_dir)
    json_path = run_dir / "model_diagnostic_report.json"
    md_path = run_dir / "model_diagnostic_report.md"
    plot_path = run_dir / "model_diagnostic_summary.png"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# v29 PPO model diagnostic",
        "",
        f"- Training days: {report['training_days']}",
        f"- Evaluation steps: {report['evaluation_steps']}",
        f"- Fresh PPO batches: {report['ppo']['fresh_update_batches']}",
        f"- Mean critic explained variance: {report['ppo']['explained_variance']['mean']:.6f}",
        f"- Mean state/action sensitivity: {report['ppo']['state_action_sensitivity']['mean']:.6f}",
        "",
        "## Primary findings",
        "",
        *[f"- {item}" for item in report["primary_findings"]],
        "",
        "## Validation snapshots",
        "",
    ]
    for row in report["validations"]:
        lines.append(
            f"- day {row['day']} ({row['stage']}): score={row['score']:.4f}, "
            f"profit={row['profit_per_request']:.3f}, "
            f"share advantage={row['completed_share'] - row['rival_completed_share']:+.3f}, "
            f"critic EV={row['critic_td_explained_variance']:.4f}"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    render_plot(run_dir, report, plot_path)
    print(json_path)
    print(md_path)
    print(plot_path)


if __name__ == "__main__":
    main()
