#!/usr/bin/env python3
"""Compile the v25→v29 cumulative ablation ladder into tables and figures."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib.pyplot as plt


METRICS = (
    "reward",
    "rl_profit",
    "heuristic_profit",
    "profit_advantage",
    "rl_completed_share",
    "heuristic_completed_share",
    "completed_share_advantage",
    "rl_fulfillment_rate",
    "rl_avg_wait_minutes",
    "policy_entropy",
)


def _number(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def _mean_finite(values: list[float]) -> float:
    kept = [value for value in values if math.isfinite(value)]
    return mean(kept) if kept else math.nan


def _find_one(directory: Path, suffix: str) -> Path | None:
    values = sorted(directory.glob(f"*{suffix}"))
    return values[0] if values else None


def _read_rows(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _run_record(stage: dict[str, Any], run_dir: Path, seed: int) -> dict[str, Any]:
    status_path = run_dir / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    evaluation = _read_rows(_find_one(run_dir, "evaluation_diagnostics.csv"))
    training = _read_rows(_find_one(run_dir, "training_diagnostics.csv"))
    record: dict[str, Any] = {
        "stage_id": stage["stage_id"],
        "label": stage["label"],
        "seed": seed,
        "exit_code": status.get("exit_code"),
        "seconds": _number(status.get("seconds")),
        "evaluation_rows": len(evaluation),
        "training_rows": len(training),
    }
    for metric in METRICS:
        record[metric] = _mean_finite([_number(row.get(metric)) for row in evaluation])
    if not math.isfinite(record["profit_advantage"]):
        record["profit_advantage"] = record["rl_profit"] - record["heuristic_profit"]
    if not math.isfinite(record["completed_share_advantage"]):
        record["completed_share_advantage"] = (
            record["rl_completed_share"] - record["heuristic_completed_share"]
        )
    actions = [str(row.get("action_label", "")) for row in evaluation if row.get("action_label")]
    record["evaluation_action_count"] = len(set(actions))
    record["evaluation_hold_fraction"] = (
        sum(action == "hold" for action in actions) / len(actions) if actions else math.nan
    )
    record["ppo_optimizer_steps"] = max(
        [_number(row.get("ppo_optimizer_steps")) for row in training] or [math.nan]
    )
    record["ppo_state_action_mi"] = _mean_finite(
        [_number(row.get("ppo_state_action_mi")) for row in training]
    )
    return record


def _summaries(records: list[dict[str, Any]], stage_order: list[str]) -> list[dict[str, Any]]:
    output = []
    previous: dict[str, Any] | None = None
    for stage_id in stage_order:
        rows = [row for row in records if row["stage_id"] == stage_id and row.get("exit_code") == 0]
        if not rows:
            continue
        summary: dict[str, Any] = {
            "stage_id": stage_id,
            "label": rows[0]["label"],
            "successful_seeds": len(rows),
        }
        for metric in METRICS + ("evaluation_action_count", "evaluation_hold_fraction", "ppo_optimizer_steps", "ppo_state_action_mi"):
            values = [_number(row.get(metric)) for row in rows]
            values = [value for value in values if math.isfinite(value)]
            summary[metric] = mean(values) if values else math.nan
            summary[metric + "_se"] = (
                stdev(values) / math.sqrt(len(values)) if len(values) > 1 else math.nan
            )
        if previous is not None:
            for metric in ("reward", "profit_advantage", "completed_share_advantage"):
                summary["delta_from_previous_" + metric] = summary[metric] - previous[metric]
        output.append(summary)
        previous = summary
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _plot(summaries: list[dict[str, Any]], output: Path) -> None:
    labels = [row["stage_id"].split("_", 1)[0].upper() for row in summaries]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    panels = (
        ("profit_advantage", "Profit advantage / request", 0.0),
        ("completed_share_advantage", "Completed-share advantage", 0.0),
        ("reward", "Evaluation reward", None),
        ("evaluation_action_count", "Distinct evaluation actions", None),
    )
    for axis, (metric, title, reference) in zip(axes.flat, panels):
        values = [_number(row.get(metric)) for row in summaries]
        errors = [_number(row.get(metric + "_se")) for row in summaries]
        errors = [0.0 if not math.isfinite(value) else value for value in errors]
        axis.errorbar(labels, values, yerr=errors, marker="o", capsize=4, linewidth=2)
        if reference is not None:
            axis.axhline(reference, color="black", linewidth=1, alpha=0.5)
        axis.set_title(title)
        axis.grid(alpha=0.25)
    fig.suptitle("Cumulative v25 → v29 ablation (matched seeds)", fontsize=15)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    options = parser.parse_args()
    root = options.run_dir.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    stages = manifest["stages"]
    records = []
    for stage in stages:
        for seed in manifest["seeds"]:
            records.append(_run_record(stage, root / stage["stage_id"] / f"seed_{seed}", int(seed)))
    summaries = _summaries(records, [stage["stage_id"] for stage in stages])
    _write_csv(root / "per_seed_results.csv", records)
    _write_csv(root / "stage_summary.csv", summaries)
    (root / "results.json").write_text(
        json.dumps({"records": records, "summaries": summaries}, indent=2, sort_keys=True) + "\n"
    )
    if summaries:
        _plot(summaries, root / "ablation_summary.png")
    print(root / "stage_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
