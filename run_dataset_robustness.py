#!/usr/bin/env python3
"""Run multi-seed NYC dataset validation and plot robustness diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import mkl_config  # noqa: F401
import numpy as np


DEFAULT_SEED_PAIRS: Tuple[Tuple[int, int], ...] = (
    (4500, 104729),
    (99173, 130363),
    (27183, 15485863),
    (65537, 32452843),
)
DURATION_MODES: Tuple[str, ...] = ("actual_if_available", "predicted_only")


def _parse_seed_pairs(value: str) -> Tuple[Tuple[int, int], ...]:
    pairs: List[Tuple[int, int]] = []
    for item in value.split(","):
        policy_seed, dataset_seed = item.strip().split(":", 1)
        pairs.append((int(policy_seed), int(dataset_seed)))
    if len(pairs) < 2:
        raise argparse.ArgumentTypeError("provide at least two policy:dataset seed pairs")
    return tuple(pairs)


def _read_numeric_csv(path: Path) -> Dict[str, np.ndarray]:
    columns = {
        "actual": [],
        "predicted": [],
        "anchor": [],
        "actual_duration": [],
        "predicted_duration": [],
    }
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            columns["actual"].append(float(row["actual_paid"]))
            columns["predicted"].append(float(row["rl_predicted_price"]))
            columns["anchor"].append(float(row["anchor_tariff_price"]))
            if row.get("actual_duration_minutes") not in (None, ""):
                columns["actual_duration"].append(float(row["actual_duration_minutes"]))
                columns["predicted_duration"].append(float(row["predicted_duration_minutes"]))
    return {key: np.asarray(values, dtype=float) for key, values in columns.items()}


def _metrics(run_id: str, mode: str, policy_seed: int, dataset_seed: int, data: Dict[str, np.ndarray]) -> Dict[str, float]:
    actual = data["actual"]
    predicted = data["predicted"]
    anchor = data["anchor"]
    error = predicted - actual
    abs_error = np.abs(error)
    slope, intercept = np.polyfit(actual, predicted, 1)
    correlation = float(np.corrcoef(actual, predicted)[0, 1])
    anchor_mae = float(np.mean(np.abs(anchor - actual)))
    return {
        "run_id": run_id,
        "duration_mode": mode,
        "policy_seed": int(policy_seed),
        "dataset_seed": int(dataset_seed),
        "rows": int(actual.size),
        "mae": float(np.mean(abs_error)),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "bias": float(np.mean(error)),
        "correlation": correlation,
        "r_squared_correlation": float(correlation * correlation),
        "calibration_slope": float(slope),
        "calibration_intercept": float(intercept),
        "within_2_dollars": float(np.mean(abs_error <= 2.0)),
        "within_5_dollars": float(np.mean(abs_error <= 5.0)),
        "anchor_mae": anchor_mae,
        "improvement_vs_anchor": float(anchor_mae - np.mean(abs_error)),
        "duration_mae": (
            float(np.mean(np.abs(data["predicted_duration"] - data["actual_duration"])))
            if data["actual_duration"].size
            else float("nan")
        ),
    }


def _write_metrics_csv(path: Path, metrics: Sequence[Dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)


def _plot_reports(
    output_dir: Path,
    runs: Sequence[Tuple[Dict[str, float], Dict[str, np.ndarray]]],
    seed_pairs: Sequence[Tuple[int, int]],
) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_paths: List[Path] = []
    global_max = max(
        float(max(np.max(data["actual"]), np.max(data["predicted"])))
        for _, data in runs
    )
    plot_max = float(np.ceil(global_max / 25.0) * 25.0)

    fig, axes = plt.subplots(
        len(DURATION_MODES),
        len(seed_pairs),
        figsize=(4.4 * len(seed_pairs), 4.2 * len(DURATION_MODES)),
        sharex=True,
        sharey=True,
    )
    for ax, (metric, data) in zip(np.asarray(axes).reshape(-1), runs):
        rng = np.random.default_rng(int(metric["policy_seed"]) + int(metric["dataset_seed"]))
        count = min(1500, data["actual"].size)
        indices = rng.choice(data["actual"].size, size=count, replace=False)
        ax.scatter(
            data["actual"][indices],
            data["predicted"][indices],
            s=8,
            alpha=0.25,
        )
        ax.plot([0.0, plot_max], [0.0, plot_max], "r--", linewidth=1.0)
        mode_label = (
            "Observed duration"
            if metric["duration_mode"] == "actual_if_available"
            else "Predicted duration only"
        )
        ax.set_title(
            f"{mode_label}\npolicy={metric['policy_seed']}, data={metric['dataset_seed']}\n"
            f"MAE=${metric['mae']:.2f}, r={metric['correlation']:.3f}, "
            f"slope={metric['calibration_slope']:.2f}"
        )
        ax.set_xlim(0.0, plot_max)
        ax.set_ylim(0.0, plot_max)
        ax.grid(alpha=0.15)
    for ax in np.asarray(axes)[-1, :]:
        ax.set_xlabel("Actual customer price ($)")
    for ax in np.asarray(axes)[:, 0]:
        ax.set_ylabel("RL predicted price ($)")
    fig.suptitle("NYC Price Robustness Across Independent Policy and Dataset Seeds")
    fig.tight_layout()
    scatter_path = output_dir / "robustness_price_scatter_facets.png"
    fig.savefig(scatter_path, dpi=160)
    plt.close(fig)
    plot_paths.append(scatter_path)

    x = np.arange(len(seed_pairs))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    metric_specs = (
        ("mae", "Price MAE ($)"),
        ("correlation", "Price correlation"),
        ("calibration_slope", "Calibration slope"),
        ("bias", "Price bias ($)"),
    )
    for mode in DURATION_MODES:
        mode_runs = [
            metric for metric, _ in runs if metric["duration_mode"] == mode
        ]
        label = (
            "Observed duration"
            if mode == "actual_if_available"
            else "Predicted duration only"
        )
        for ax, (key, ylabel) in zip(axes.reshape(-1), metric_specs):
            ax.plot(
                x,
                [float(metric[key]) for metric in mode_runs],
                marker="o",
                label=label,
            )
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.2)
    for ax in axes[-1, :]:
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"p{policy}\nd{dataset}" for policy, dataset in seed_pairs]
        )
        ax.set_xlabel("Policy seed / dataset seed")
    axes[0, 0].legend(loc="best")
    fig.suptitle("Robustness Metrics Across Independent NYC Samples")
    fig.tight_layout()
    metrics_path = output_dir / "robustness_metrics_by_seed.png"
    fig.savefig(metrics_path, dpi=160)
    plt.close(fig)
    plot_paths.append(metrics_path)

    labels = [
        f"{'obs' if metric['duration_mode'] == 'actual_if_available' else 'pred'}\n"
        f"p{metric['policy_seed']}\nd{metric['dataset_seed']}"
        for metric, _ in runs
    ]
    errors = [
        np.abs(data["predicted"] - data["actual"])
        for _, data in runs
    ]
    fig, ax = plt.subplots(figsize=(14, 6))
    try:
        ax.boxplot(errors, tick_labels=labels, showfliers=False)
    except TypeError:  # Matplotlib < 3.9
        ax.boxplot(errors, labels=labels, showfliers=False)
    ax.set_ylabel("Absolute price error ($)")
    ax.set_title("Price Error Distribution Across Robustness Runs")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    error_path = output_dir / "robustness_absolute_error_boxplot.png"
    fig.savefig(error_path, dpi=160)
    plt.close(fig)
    plot_paths.append(error_path)

    return plot_paths


def _aggregate(metrics: Sequence[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    keys = ("mae", "rmse", "bias", "correlation", "calibration_slope", "improvement_vs_anchor")
    for mode in DURATION_MODES:
        subset = [metric for metric in metrics if metric["duration_mode"] == mode]
        result[mode] = {}
        for key in keys:
            values = np.asarray([float(metric[key]) for metric in subset], dtype=float)
            result[mode][f"{key}_mean"] = float(np.mean(values))
            result[mode][f"{key}_std"] = float(np.std(values))
            result[mode][f"{key}_min"] = float(np.min(values))
            result[mode][f"{key}_max"] = float(np.max(values))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        help="Exact registered model id, immutable archive id, or .pt path.",
    )
    parser.add_argument(
        "--dataset-root",
        default="/Users/huali/.cache/kagglehub/datasets/aaronweymouth/nyc-rideshare-raw-data/versions/1",
    )
    parser.add_argument("--rows", type=int, default=3000)
    parser.add_argument(
        "--seed-pairs",
        type=_parse_seed_pairs,
        default=DEFAULT_SEED_PAIRS,
        help="Comma-separated policy_seed:dataset_seed pairs.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
    )
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    model_label = Path(str(args.model)).stem
    safe_label = "".join(
        char if char.isalnum() or char in "._-" else "_"
        for char in model_label
    ).strip("._-") or "trained_policy"
    output_dir_arg = (
        args.output_dir
        or f"artifacts/{safe_label}_nyc_robustness"
    )
    output_dir = (project_dir / output_dir_arg).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    runs: List[Tuple[Dict[str, float], Dict[str, np.ndarray]]] = []
    for duration_mode in DURATION_MODES:
        for policy_seed, dataset_seed in args.seed_pairs:
            run_id = f"{duration_mode}_p{policy_seed}_d{dataset_seed}"
            prefix = output_dir / run_id
            csv_path = prefix.with_name(prefix.name + "_rows.csv")
            log_path = prefix.with_name(prefix.name + ".log")
            command = [
                sys.executable,
                str(project_dir / "Core.py"),
                "--eval_only",
                "--dataset_only",
                "--market",
                "New York City",
                "--trained_model_in",
                args.model,
                "--model_registry_dir",
                str(project_dir / "artifacts" / "trained_models"),
                "--firm2_mode",
                "static",
                "--seed",
                str(policy_seed),
                "--deterministic_experiment_seed",
                "--deterministic_torch",
                "--compare_with_dataset",
                "--dataset_root",
                str(Path(args.dataset_root).expanduser().resolve()),
                "--dataset_glob",
                "*.parquet",
                "--comparison_limit",
                str(args.rows),
                "--dataset_preview_rows",
                "0",
                "--comparison_policy_mode",
                "argmax",
                "--comparison_policy_seed",
                str(policy_seed),
                "--comparison_dataset_seed",
                str(dataset_seed),
                "--comparison_duration_mode",
                duration_mode,
                "--comparison_out",
                str(csv_path),
                "--comparison_plot_prefix",
                str(prefix),
                "--report_prefix",
                str(prefix) + "_run",
            ]
            with log_path.open("w", encoding="utf-8") as log_handle:
                completed = subprocess.run(
                    command,
                    cwd=project_dir,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"robustness run {run_id} failed; inspect {log_path}"
                )
            data = _read_numeric_csv(csv_path)
            runs.append(
                (
                    _metrics(
                        run_id,
                        duration_mode,
                        policy_seed,
                        dataset_seed,
                        data,
                    ),
                    data,
                )
            )

    metrics = [metric for metric, _ in runs]
    metrics_path = output_dir / "robustness_metrics.csv"
    _write_metrics_csv(metrics_path, metrics)
    aggregate = _aggregate(metrics)
    summary_path = output_dir / "robustness_summary.json"
    summary_path.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    plots = _plot_reports(output_dir, runs, args.seed_pairs)
    print(json.dumps({
        "runs": len(runs),
        "rows_per_run": args.rows,
        "metrics_csv": str(metrics_path),
        "summary_json": str(summary_path),
        "plots": [str(path) for path in plots],
        "aggregate": aggregate,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
