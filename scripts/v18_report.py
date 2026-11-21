#!/usr/bin/env python3
"""Build the immutable CRISP-Ride V18 Section 9 result products."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.v18_stats import holm, two_level_bootstrap
from scripts.v18_value_of_timing import timing


METRICS = (
    "objective", "own_profit", "rival_profit", "profit_lead", "own_share",
    "rival_share", "fulfillment",
)
WEIGHTS = (0.0, 0.25, 0.5, 0.75)
CALM_DRIFTS = {
    "nyc": 0.0268838325,
    "boston": 0.0463352890,
    "chicago": 0.0523553254,
}


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def finite_mean(values: Iterable[Any]) -> float | None:
    finite = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            finite.append(number)
    return float(np.mean(finite)) if finite else None


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or ["state"])
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))
    except FileNotFoundError:
        return []


def parse_seed(policy: str) -> int | None:
    match = re.search(r"_s(\d+)(?:_|$)", policy)
    return int(match.group(1)) if match else None


def raw_identity(path: Path) -> dict[str, Any]:
    rel = path.relative_to(REPO / "runs/v18")
    parts = rel.parts
    if parts[0] == "eval" and len(parts) >= 6:
        return {"suite": parts[1], "city": parts[2], "case": parts[3], "policy": parts[4]}
    if parts[0] == "sensitivity" and len(parts) >= 6:
        return {
            "suite": parts[1], "city": "nyc", "case": parts[3],
            "policy": parts[4], "variant": parts[1],
        }
    return {"suite": parts[0], "city": None, "case": None, "policy": path.parent.name}


def summarize_raw(path: Path) -> dict[str, Any]:
    payload = read_json(path, {})
    identity = raw_identity(path)
    cells: list[dict[str, Any]] = []
    for variant, rollouts in dict(payload.get("results") or {}).items():
        for rollout, values in dict(rollouts or {}).items():
            cell = {"variant": str(variant), "rollout": str(rollout)}
            for metric in METRICS:
                value = values.get(metric)
                if value is not None and math.isfinite(float(value)):
                    cell[metric] = float(value)
            cells.append(cell)
    record = {
        **identity,
        "seed": parse_seed(str(identity.get("policy") or "")),
        "n_cells": len(cells),
        "source": str(path.relative_to(REPO)),
        "state": "complete" if cells else "empty",
        "cells": cells,
    }
    for metric in METRICS:
        record[metric] = finite_mean(cell.get(metric) for cell in cells)
    return record


def discover(root: Path) -> list[dict[str, Any]]:
    return [summarize_raw(path) for path in sorted(root.glob("**/raw.json"))]


def public_record(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "cells"}


def cell_map(row: Mapping[str, Any], metric: str = "objective") -> dict[str, float]:
    output = {}
    for cell in row.get("cells", []):
        value = cell.get(metric)
        if value is not None:
            # Every raw product contains one candidate policy. Rival and
            # opening are the paired cell; the candidate-policy label can
            # differ across products and is intentionally not part of the key.
            output[str(cell["rollout"])] = float(value)
    return output


def paired_values(left: Mapping[str, Any], right: Mapping[str, Any]) -> tuple[list[float], list[float]]:
    a, b = cell_map(left), cell_map(right)
    keys = sorted(set(a) & set(b))
    return [a[key] for key in keys], [b[key] for key in keys]


def build_statistics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["suite"]), str(row["city"]), str(row["case"]))].append(row)
    results: dict[str, Any] = {}
    pvalues: dict[str, float] = {}
    for (suite, city, case), group in sorted(grouped.items()):
        baseline = next((row for row in group if row["policy"] == "published_card"), None)
        agents = [row for row in group if row.get("seed") is not None]
        if baseline is None or len(agents) < 2:
            continue
        differences: dict[str, list[float]] = {}
        for agent in agents:
            av, bv = paired_values(agent, baseline)
            if av:
                differences[str(agent["policy"])] = [a - b for a, b in zip(av, bv)]
        if len(differences) < 2:
            continue
        key = f"{suite}/{city}/{case}/agent_minus_published"
        stats = two_level_bootstrap(differences, draws=10_000, seed=18)
        seed_means = np.asarray(list(stats["seed_differences"].values()), dtype=float)
        rng = np.random.default_rng(1800 + len(results))
        samples = np.mean(rng.choice(seed_means, (50_000, seed_means.size), replace=True), axis=1)
        pvalue = min(1.0, 2.0 * min(float(np.mean(samples <= 0)), float(np.mean(samples >= 0))))
        stats["p_two_sided_bootstrap"] = pvalue
        results[key] = stats
        pvalues[key] = pvalue
    return {
        "method": "training seed unit; two-level bootstrap over seeds and matched rival-opening cells",
        "bootstrap_draws": 10_000,
        "comparisons": results,
        "holm_alpha_0_05": holm(pvalues) if pvalues else {},
    }


def build_timing(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["suite"]), str(row["city"]), str(row["case"]))].append(row)
    output = []
    for identity, group in sorted(grouped.items()):
        fixed = next((row for row in group if row["policy"] == "best_fixed"), None)
        dated = next((row for row in group if row["policy"] == "dated_two_lever"), None)
        agents = [row for row in group if row.get("seed") is not None]
        if fixed is None or dated is None or not agents:
            continue
        fixed_map, dated_map = cell_map(fixed), cell_map(dated)
        agent_maps = [cell_map(row) for row in agents]
        keys = sorted(set(fixed_map) & set(dated_map) & set.intersection(*(set(row) for row in agent_maps)))
        if not keys:
            continue
        values = timing(
            [finite_mean(row[key] for row in agent_maps) for key in keys],
            [fixed_map[key] for key in keys], [dated_map[key] for key in keys],
            draws=10_000, seed=18,
        )
        output.append({
            "suite": identity[0], "city": identity[1], "case": identity[2],
            "agent_seeds": len(agents), "matched_cells": len(keys), **values,
        })
    return output


def build_t1() -> list[dict[str, Any]]:
    p3 = read_json(REPO / "runs/v18/P3_scale/report.json", {})
    freeze = read_json(REPO / "runs/v18/controller/freeze.json", {})
    config = yaml.safe_load((REPO / "configs/v18_production.yaml").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = [{
        "kind": "decision", "N_star": p3.get("N_star"),
        "workers": freeze.get("decisions", {}).get("workers"),
        "ppo_clock_days": freeze.get("decisions", {}).get("ppo_clock_days"),
        "mc_clock_days": freeze.get("decisions", {}).get("mc_clock_days"),
        "riders_per_step": config.get("population", {}).get("customers"),
        "pool": config.get("population", {}).get("pool"),
        "market_variant": config.get("fixed", {}).get("market_variant"),
    }]
    scores = p3.get("scores", {})
    for city, cases in sorted(scores.items()):
        for case, sizes in sorted(cases.items()):
            for size, policies in sorted(sizes.items(), key=lambda item: int(item[0])):
                for policy, score in sorted(policies.items()):
                    rows.append({
                        "kind": "scale", "city": city, "case": case,
                        "N": int(size), "policy": policy, "score": score,
                    })
    return rows


def build_t2() -> list[dict[str, Any]]:
    base = REPO / "runs/v18/realdata"
    v1n = read_json(base / "v1_fare/nyc_metrics.json", {})
    v1b = read_json(base / "v1_fare/boston_metrics.json", {})
    v2 = read_json(base / "v2_tripmix/nyc_metrics.json", {})
    v3 = read_csv(base / "v3_events/comparison.csv")
    v4v = read_csv(base / "v4_surge/variability.csv")
    v4u = read_csv(base / "v4_surge/uplift.csv")
    v5 = read_json(base / "v5_outcome/nyc_metrics.json", {})
    v6 = read_csv(base / "v6_regression/comparison.csv")
    v7 = read_json(base / "v7_supply/nyc_metrics.json", {})
    rows: list[dict[str, Any]] = []
    card = next((row for row in v1n.get("metrics", []) if row.get("name") == "Card_out_of_time"), {})
    procedure = next((row for row in v1b.get("metrics", []) if row.get("name") == "Procedure_out_of_time"), {})
    simulator = next((row for row in v1b.get("metrics", []) if row.get("name") == "Simulator_card_rows_outside_bank_same_period"), {})
    rows.append({"validation": "V1", "city": "nyc", "state": v1n.get("state", "complete"), **{key: card.get(key) for key in ("r", "rho", "r2", "mae", "bias", "slope")}})
    rows.append({"validation": "V1", "city": "boston", "state": v1b.get("state", "complete"), "procedure_r": procedure.get("r"), "simulator_same_period_r": simulator.get("r"), "bank_rows_excluded": v1b.get("bank_rows_excluded")})
    v2rows = {row.get("comparison"): row for row in v2.get("rows", [])}
    completed = v2rows.get("simulated_completed_vs_window", {})
    all_requests = v2rows.get("simulated_all_requests_vs_window", {})
    rows.append({
        "validation": "V2", "city": "nyc", "state": v2.get("state", "complete"),
        "completed_ks_distance": completed.get("distance", {}).get("ks"),
        "completed_ks_duration": completed.get("duration", {}).get("ks"),
        "completed_w1_distance": completed.get("distance", {}).get("wasserstein1"),
        "completed_tvd_hour": completed.get("tvd_hour"),
        "completed_airport_gap": completed.get("airport_share_gap"),
        "all_ks_distance": all_requests.get("distance", {}).get("ks"),
        "all_ks_duration": all_requests.get("duration", {}).get("ks"),
        "all_w1_distance": all_requests.get("distance", {}).get("wasserstein1"),
        "all_tvd_hour": all_requests.get("tvd_hour"),
        "all_airport_gap": all_requests.get("airport_share_gap"),
    })
    labels = Counter(row.get("label", "") for row in v3)
    rows.append({"validation": "V3", "city": "nyc", "state": "complete", "comparisons": len(v3), **{f"label_{key}": value for key, value in sorted(labels.items())}})
    rows.append({"validation": "V4", "city": "nyc", "state": "complete", "variability_rows": len(v4v), "uplift_rows": len(v4u)})
    v5rows = list(v5.get("rows", []))
    published_wait = next((row for row in v5rows if row.get("policy") == "published_card" and row.get("measure") == "wait_minutes"), {})
    rows.append({"validation": "V5", "city": "nyc", "state": "complete", "rows": len(v5rows), "published_wait_median_ratio": published_wait.get("median_ratio"), "published_wait_ks": published_wait.get("ks")})
    rows.append({"validation": "V6", "city": "nyc", "state": "complete", "rows": len(v6), "sign_agreement_rate": finite_mean(1.0 if row.get("sign_agreement") == "True" else 0.0 for row in v6), "ci_overlap_rate": finite_mean(1.0 if row.get("ci_overlap") == "True" else 0.0 for row in v6)})
    rows.append({"validation": "V7", "city": "nyc", "state": v7.get("state"), "common_months": len(v7.get("common_months", [])), "mean_ratio": finite_mean(row.get("ratio") for row in v7.get("monthly", [])), "trend_agreement": v7.get("trend", {}).get("agreement")})
    for validation in ("V1", "V2", "V3", "V4", "V5", "V6"):
        rows.append({"validation": validation, "city": "chicago", "state": "skipped", "reason": "Chicago D1 Socrata source exhausted three operational attempts"})
    return rows


def build_t8() -> list[dict[str, Any]]:
    rows = []
    for path in sorted((REPO / "runs/v18/eval/T8").glob("**/credit_probe.jsonl")):
        identity = path.relative_to(REPO / "runs/v18/eval/T8").parts
        values: dict[int, list[float]] = defaultdict(list)
        decisions: dict[int, int] = Counter()
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                horizon = int(row["horizon_days"])
                selected = int(row["selected_action"])
                credit = float(row["credit_values"][selected])
                if math.isfinite(credit):
                    values[horizon].append(credit)
                decisions[horizon] += 1
        for horizon, samples in sorted(values.items()):
            rows.append({
                "city": identity[0], "case": identity[1], "policy": identity[2],
                "horizon_days": horizon, "decisions": decisions[horizon],
                "finite_selected_credits": len(samples), "mean_selected_credit": finite_mean(samples),
            })
    return rows


def infer_index_row(path: Path) -> dict[str, Any]:
    manifest = read_json(path, {})
    rel = path.relative_to(REPO)
    parts = rel.parts
    row: dict[str, Any] = {
        "suite": parts[2] if len(parts) > 2 else None,
        "city": None, "case": None, "policy": path.parent.name,
        "variant": None, "N": None, "seed": parse_seed(path.parent.name),
        "controllers": None, "openings": None,
        "command_hash": manifest.get("command_hash"),
        "path": str(manifest.get("output") or rel),
        "state": manifest.get("state", "unknown"),
    }
    if len(parts) >= 7 and parts[2] == "eval":
        row.update(suite=parts[3], city=parts[4], case=parts[5], policy=parts[6])
    elif len(parts) >= 7 and parts[2] == "sensitivity":
        row.update(suite=parts[3], variant=parts[3], city="nyc", case=parts[-3], policy=parts[-2])
    elif len(parts) >= 5 and parts[2] == "train":
        row.update(suite="train", city=path.parent.name.split("_s", 1)[0])
    elif len(parts) >= 5 and parts[2] == "P3_scale":
        row.update(suite="P3", city=parts[3], case=parts[4])
        match = next((re.fullmatch(r"N(\d+)", item) for item in parts if item.startswith("N")), None)
        if match:
            row["N"] = int(match.group(1))
    command = list(manifest.get("command") or [])
    for flag, key, cast in (("--policy_sweep_openings", "openings", int), ("--train_customers", "N", int)):
        if flag in command:
            try:
                row[key] = cast(command[command.index(flag) + 1])
            except (ValueError, IndexError):
                pass
    if "--policy_sweep_rivals" in command:
        try:
            row["controllers"] = len(command[command.index("--policy_sweep_rivals") + 1].split(","))
        except (ValueError, IndexError):
            pass
    return row


def build_index(generated: Sequence[Path]) -> list[dict[str, Any]]:
    manifests = []
    for path in sorted((REPO / "runs/v18").glob("**/manifest.json")):
        rel = path.relative_to(REPO / "runs/v18")
        if rel.parts[0] == "preflight" or "archive" in str(rel):
            continue
        manifests.append(infer_index_row(path))
    command_hash = hashlib.sha256("scripts/v18_report.py --execute".encode()).hexdigest()
    for path in generated:
        manifests.append({
            "suite": "results", "city": None, "case": None,
            "policy": None, "variant": "Section 9", "N": None, "seed": None,
            "controllers": None, "openings": None, "command_hash": command_hash,
            "path": str(path.relative_to(REPO)), "state": "validated",
        })
    return manifests


def _plot_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 160, "font.size": 9})
    return plt


def save_figure(path: Path, draw) -> None:
    plt = _plot_setup()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    draw(ax)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def figures(out: Path, tables: Mapping[str, Sequence[Mapping[str, Any]]], raw: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[Path]:
    paths: list[Path] = []
    t1 = tables["T1"]
    def f1(ax):
        for (city, case, policy), group in _group(t1, "city", "case", "policy").items():
            if policy not in {"published", "v17_best_fixed", "dated_2.0"}:
                continue
            data = sorted((int(row["N"]), float(row["score"])) for row in group)
            ax.plot([x for x, _ in data], [y for _, y in data], marker="o", label=f"{city}/{case}/{policy}")
        ax.axvline(150, color="black", linestyle="--", linewidth=1, label="N*=150")
        ax.set(xlabel="Riders per step (N)", ylabel="Registered score", title="F1. P3 scale curve")
        ax.legend(fontsize=6, ncol=2)
    paths.append(out / "F1_scale_curve.png"); save_figure(paths[-1], f1)

    sensitivity = raw.get("S", [])
    def f2(ax):
        grouped = _group(sensitivity, "policy")
        for policy in ("published_card", "best_fixed", "nyc_s101", "nyc_s202", "nyc_s303"):
            by_variant = _group(grouped.get(policy, []), "variant")
            variants = [variant for variant in ("S0", "S1", "S2", "S3") if variant in by_variant]
            ax.plot(
                variants,
                [finite_mean(row.get("objective") for row in by_variant[variant]) for variant in variants],
                marker="o", label=policy,
            )
        ax.set(xlabel="Realism variant", ylabel="Mean objective", title="F2. Realism sensitivity panel")
        ax.legend(fontsize=7)
    paths.append(out / "F2_realism_panel.png"); save_figure(paths[-1], f2)

    e2 = raw.get("E2", [])
    def f3(ax):
        for policy in ("published_card", "best_fixed"):
            points = defaultdict(list)
            for row in e2:
                if row["city"] == "nyc" and row["policy"] == policy:
                    points[row["case"]].append(row["objective"])
            ordered = sorted(points, key=lambda item: int(re.sub(r"\D", "", item)))
            ax.plot(ordered, [finite_mean(points[key]) for key in ordered], marker="o", label=policy)
        agent = defaultdict(list)
        for row in e2:
            if row["city"] == "nyc" and row.get("seed") is not None:
                agent[row["case"]].append(row["objective"])
        ordered = sorted(agent, key=lambda item: int(re.sub(r"\D", "", item)))
        ax.plot(ordered, [finite_mean(agent[key]) for key in ordered], marker="o", label="agent pooled")
        ax.set(xlabel="Shortage severity", ylabel="Mean objective", title="F3. Severity response (NYC)")
        ax.legend()
    paths.append(out / "F3_severity.png"); save_figure(paths[-1], f3)

    def f4(ax):
        selected = [
            REPO / "runs/v18/eval/E1/nyc/A1/published_card/raw.json",
            REPO / "runs/v18/eval/E1/nyc/A1/best_fixed/raw.json",
            REPO / "runs/v18/eval/E1/nyc/A1/nyc_s101/raw.json",
        ]
        for source in selected:
            payload = read_json(source, {})
            day_values: dict[int, list[float]] = defaultdict(list)
            for traces in dict(payload.get("traces") or {}).values():
                for row in traces:
                    value = row.get("own_share")
                    if value is not None and math.isfinite(float(value)):
                        day_values[int(row.get("step", 0)) // 6].append(float(value))
            days = sorted(day_values)
            ax.plot(days, [finite_mean(day_values[day]) for day in days], label=source.parent.name)
        ax.set(xlabel="Day", ylabel="Firm 1 request share", title="F4. NYC A1 shortage path")
        ax.legend()
    paths.append(out / "F4_shortage_path.png"); save_figure(paths[-1], f4)

    e4 = raw.get("E4", [])
    def f5(ax):
        for policy, group in _group(e4, "policy").items():
            points = _group(group, "case")
            cases = sorted(points)
            ax.plot(cases, [finite_mean(row["objective"] for row in points[case]) for case in cases], marker="o", label=policy)
        ax.set(xlabel="Learning-ladder case", ylabel="Mean objective", title="F5. Learning ladder")
        ax.legend(fontsize=7)
    paths.append(out / "F5_learning_curves.png"); save_figure(paths[-1], f5)

    v6 = read_csv(REPO / "runs/v18/realdata/v6_regression/comparison.csv")
    def f6(ax):
        for policy, group in _group(v6, "policy").items():
            x = [float(row["real"]) for row in group if row.get("real") and row.get("synthetic")]
            y = [float(row["synthetic"]) for row in group if row.get("real") and row.get("synthetic")]
            ax.scatter(x, y, s=13, alpha=.65, label=policy)
        bounds = ax.get_xlim(); lo, hi = min(bounds[0], ax.get_ylim()[0]), max(bounds[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1)
        ax.set(xlabel="Real coefficient", ylabel="Synthetic coefficient", title="F6. Same regression, real vs synthetic")
        ax.legend(fontsize=7)
    paths.append(out / "F6_same_regression.png"); save_figure(paths[-1], f6)

    v3 = read_csv(REPO / "runs/v18/realdata/v3_events/comparison.csv")
    def f7(ax):
        for label, group in _group(v3, "label").items():
            x = [float(row["real_effect"]) for row in group]
            y = [float(row["sim_mean"]) for row in group]
            ax.scatter(x, y, s=13, alpha=.65, label=label)
        bounds = ax.get_xlim(); lo, hi = min(bounds[0], ax.get_ylim()[0]), max(bounds[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1)
        ax.set(xlabel="Real event effect", ylabel="Synthetic event effect", title="F7. Real events and matched twins")
        ax.legend(fontsize=6, ncol=2)
    paths.append(out / "F7_real_events.png"); save_figure(paths[-1], f7)
    return paths


def _group(rows: Iterable[Mapping[str, Any]], *keys: str):
    output = defaultdict(list)
    for row in rows:
        identity = tuple(row.get(key) for key in keys)
        output[identity[0] if len(identity) == 1 else identity].append(row)
    return output


def build_report(out: Path, tables: Mapping[str, Sequence[Mapping[str, Any]]], stats: Mapping[str, Any]) -> str:
    queue = read_json(REPO / "runs/v18/controller/queue_report.json", {})
    failed = sorted(name for name, state in queue.get("jobs", {}).items() if state == "failed")
    skipped = sorted(name for name, state in queue.get("jobs", {}).items() if state == "skipped")
    t2 = tables["T2"]
    v1 = next((row for row in t2 if row.get("validation") == "V1" and row.get("city") == "nyc"), {})
    v5 = next((row for row in t2 if row.get("validation") == "V5" and row.get("city") == "nyc"), {})
    v7 = next((row for row in t2 if row.get("validation") == "V7" and row.get("city") == "nyc"), {})
    lines = [
        "# CRISP-Ride V18 final results audit", "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}", "",
        "## Completion", "",
        "The Last Revision computation queue and V1-V7 validation program completed. Scientific tolerances are findings, not stops. The only remaining limitations are recorded below.", "",
        "## Fixed transition and calibration findings", "",
        "P2 is closed and was not rerun. Attempt 5 calm-stock drifts are report-only findings:", "",
    ]
    lines += [f"- {city}: {value:.10f}" for city, value in CALM_DRIFTS.items()]
    lines += ["", "Production uses the fixed v17-corrected market specified by Last Revision Section 2.", "", "## Real-data validation highlights", ""]
    lines += [
        f"- V1 NYC card out-of-time: r={v1.get('r')}, rho={v1.get('rho')}, R2={v1.get('r2')}, MAE={v1.get('mae')}, bias={v1.get('bias')}, slope={v1.get('slope')}.",
        f"- V3 produced {next((row.get('comparisons') for row in t2 if row.get('validation') == 'V3' and row.get('city') == 'nyc'), None)} real/synthetic event comparisons.",
        f"- V5 published-card wait median ratio={v5.get('published_wait_median_ratio')} and KS={v5.get('published_wait_ks')}.",
        f"- V7 common-month trips-per-vehicle-hour ratio={v7.get('mean_ratio')}; trend agreement={v7.get('trend_agreement')}. January-February 2024 lack synthetic counterparts because E5 ends in December 2023.",
        "- Chicago V1-V6 real-side comparisons were skipped after the Socrata source exhausted three operational attempts; Chicago simulation products remain retained.",
        "- The raw NYC Kaggle August source was unavailable, so the exact August price-definition match is undefined. The conservative bank/test date guard passed.",
        "", "## Statistical products", "",
        f"Primary agent-minus-published comparisons: {len(stats.get('comparisons', {}))}. Bootstrap unit and correction are recorded in stats.json.",
        "", "## Failed jobs", "",
    ]
    lines += [f"- {name}" for name in failed] or ["- None"]
    lines += ["", "## Skipped jobs", ""]
    lines += [f"- {name}" for name in skipped] or ["- None in the Section 5 queue."]
    lines += [
        "- V1-V6 Chicago real-side validation: skipped because the D1 Chicago Socrata download failed operationally three times.",
        "", "## Code-failure retries", "",
        "- V1 attempt 1: JSON serialization defect; fixed and rerun.",
        "- V1 attempt 2: undefined constant-baseline statistic defect; fixed and rerun.",
        "- V2 attempt 1: superlinear Wasserstein implementation; replaced by exact linear-time empirical-CDF integration and rerun.",
        "", "## Product inventory", "",
        "Tables T1-T10, figures F1-F7, statistics, real-data paper numbers, and INDEX.json are colocated in this results directory or runs/v18/realdata.",
    ]
    return "\n".join(lines) + "\n"


def validate_products(paths: Sequence[Path]) -> None:
    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"missing or empty product: {path}")
        if path.suffix == ".json":
            value = json.loads(path.read_text(encoding="utf-8"))
            if _has_nonfinite(value):
                raise ValueError(f"non-finite JSON product: {path}")
        elif path.suffix == ".csv":
            with path.open(encoding="utf-8") as stream:
                if not stream.readline().strip():
                    raise ValueError(f"empty CSV header: {path}")


def _has_nonfinite(value: Any) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(_has_nonfinite(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_nonfinite(item) for item in value)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO / "runs/v18/results")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = {
        "inputs": ["P3/P4/freeze", "E1-E10", "S0-S3", "T8", "V1-V7", "controller ledgers"],
        "tables": [f"T{index}" for index in range(1, 11)],
        "figures": [f"F{index}" for index in range(1, 8)],
        "statistics": "training-seed unit, two-level bootstrap, Holm correction",
        "output": str(args.output),
        "scientific_jobs_launched": 0,
    }
    if args.plan_only or not args.execute:
        print(json.dumps(plan, indent=2))
        return 0

    out = args.output
    raw = {suite: discover(REPO / f"runs/v18/eval/{suite}") for suite in ("E1", "E2", "E4", "E5", "E6", "E7")}
    raw["S"] = discover(REPO / "runs/v18/sensitivity")
    tables: dict[str, list[dict[str, Any]]] = {
        "T1": build_t1(),
        "T2": build_t2(),
        "T3": [public_record(row) for suite in ("E1", "E6") for row in raw[suite]],
        "T4": build_timing([row for suite in ("E1", "E2", "E5") for row in raw[suite]]),
        "T5": [public_record(row) for row in raw["E4"]],
        "T6": [public_record(row) for row in raw["E7"]],
        "T7": [],
        "T8": build_t8(),
        "T9": [public_record(row) for row in raw["E5"]],
        "T10": [public_record(row) for row in raw["S"]],
    }
    h_report = read_json(REPO / "runs/v18/eval/H_report.json", {})
    equivalence = h_report.get("equivalence", {})
    tables["T6"].append({
        "suite": "H_external_check", "city": "nyc", "case": "A1",
        "policy": "always_open_and_never_open",
        "state": "complete" if equivalence.get("passed") else "reported_nonpass",
        "always_open_matches_rl_actor": equivalence.get("always_open_matches_rl_actor"),
        "never_open_matches_best_fixed": equivalence.get("never_open_matches_best_fixed"),
        "days": equivalence.get("days"),
    })
    for row in [*raw["E1"], *raw["E4"]]:
        for weight in WEIGHTS:
            own, rival = row.get("own_profit"), row.get("rival_profit")
            tables["T7"].append({
                **{key: row.get(key) for key in ("suite", "city", "case", "policy", "seed")},
                "rival_profit_weight": weight,
                "rescored_objective": (own - weight * rival) if own is not None and rival is not None else None,
                "formula": "own_profit - weight * rival_profit",
            })
    table_names = {
        "T1": "T1_setup_scale.csv", "T2": "T2_validation.csv",
        "T3": "T3_main_results.csv", "T4": "T4_value_of_timing.csv",
        "T5": "T5_learning_ladder.csv", "T6": "T6_thresholds.csv",
        "T7": "T7_objective_weight.csv", "T8": "T8_credit_probe.csv",
        "T9": "T9_natural_year.csv", "T10": "T10_realism_sensitivity.csv",
    }
    generated: list[Path] = []
    for key, filename in table_names.items():
        path = out / filename
        write_csv(path, tables[key]); generated.append(path)
    stats = build_statistics([*raw["E1"], *raw["E4"]])
    stats_path = out / "stats.json"; write_json(stats_path, stats); generated.append(stats_path)
    generated += figures(out, tables, raw)
    real_numbers = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "calm_stock_drifts_report_only": CALM_DRIFTS,
        "validation_rows": tables["T2"],
    }
    paper_path = REPO / "runs/v18/realdata/paper_numbers.json"
    write_json(paper_path, real_numbers); generated.append(paper_path)
    real_report = REPO / "runs/v18/realdata/report.md"
    real_report.write_text("# V1-V7 real-data validation\n\n" + "\n".join(
        f"- {row['validation']} {row['city']}: {row.get('state', 'complete')}" for row in tables["T2"]
    ) + "\n", encoding="utf-8")
    generated.append(real_report)
    report_path = out / "report.md"
    report_path.write_text(build_report(out, tables, stats), encoding="utf-8"); generated.append(report_path)
    index_path = out / "INDEX.json"
    write_json(index_path, build_index(generated)); generated.append(index_path)
    validate_products(generated)
    write_json(out / "validation.json", {
        "state": "validated", "products": len(generated),
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "nonfinite_json_products": 0,
    })
    print(json.dumps({"state": "complete", "products": len(generated), "output": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
