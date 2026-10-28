#!/usr/bin/env python3
"""v26 confirmatory experiment (2026-10 review response).

Phases
------
prepare    Re-run city preprocessing with whole-file systematic sampling and a
           50,000-row empirical trip bank; writes schema-2 calibrations with
           data-calibrated tariff anchors, weather demand intensity, city price
           level, transit fare, and driver pay share.
external   Chicago TNP shared-trip benchmark for every threshold source.
profiles   Numeric CrowdLLM thresholds (``--gpt_threshold_mode llm_numeric``)
           for the *same* 20,000 riders and cold-start rides as the v25
           population, plus two 2,000-rider numeric prompt repeats.
train      Seeded training cells (see ``cells``):
             main      cities x {profit, competitiveness} x seeds, LLM-numeric
             mixtures  cities x five profit/competitiveness mixtures x seeds
             ablation  cities x endpoints x {rubric, constant_mean, shuffled}
             channels  cities x endpoints x seeds x {rider memory on, supply
                       feedback off, both, 1.4x fleet}: forward-looking channel
                       decomposition and capacity-slack sensitivity
           Every run uses the v26 environment, held-out benchmarks, opponent
           domain randomization, per-opening environment reseeding, truncation
           bootstrapping, Firm-1-seat baselines (static grid, opening hold,
           myopic one-step oracle), and perturbed out-of-distribution rivals.
analyze    Across-seed tables (mean and 95% t-intervals) for objective,
           RL-minus-best-static, RL-minus-myopic, held-out benchmark and OOD
           results, threshold ablations, and the external benchmark.

Usage::

    python scripts/run_v26_experiment.py --phase prepare
    python scripts/run_v26_experiment.py --phase external
    OPENAI_API_KEY=... python scripts/run_v26_experiment.py --phase profiles
    python scripts/run_v26_experiment.py --phase train --cells main --jobs 3
    python scripts/run_v26_experiment.py --phase analyze
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "v26_confirmatory"
V25_EXPERIMENT = PROJECT_ROOT / "artifacts" / "three_city_full_20260825"
SEEDS = (4500, 4501, 4502)
PROMPT_MODEL = "gpt-4.1"
DATASET_SEED = 104729
POLICY_SEED = 99173
STEPS_PER_DAY = 6            # ordered 4-hour blocks
TRAIN_DAYS = 6100            # 36,600 market steps, same simulated volume as v25
POPULATION_DECISIONS = 1080  # 4 training classes x 3 openings x 15 days x 6 steps
EVAL_STEPS = 1200            # 200 days
BENCHMARKS = ("pi_price_gap", "mpc_grid", "region_supply_demand", "surge_driver_incentive", "queue_service_threshold")
CITIES: Mapping[str, Mapping[str, str]] = {
    "nyc": {"market": "New York City", "ride_root": "data/raw/rides/nyc_kaggle"},
    "boston": {"market": "Boston", "ride_root": "data/raw/rides/boston_kaggle"},
    "chicago": {"market": "Chicago", "ride_root": "data/raw/rides/chicago_kaggle"},
}
ENDPOINTS = (
    {"label": "endpoint_profit", "objective": "profit_maximization"},
    {"label": "endpoint_competitiveness", "objective": "competitiveness"},
)
MIXTURES = tuple(
    {"label": f"p{p}c{100 - p}", "objective": f"profit_{p}_competitiveness_{100 - p}"}
    for p in (10, 30, 50, 70, 90)
)
ABLATIONS = ("rubric", "constant_mean", "shuffled")
# Forward-looking channel decomposition.  The ``llm_numeric`` main cell is the
# base environment (driver-supply feedback on, rider memory off).  RL's gain
# over the best static tariff and over the myopic one-step oracle is compared
# across environments that add or remove intertemporal channels.
CHANNEL_ARMS: Dict[str, List[str]] = {
    "ch_memory": ["--rider_memory"],
    "ch_nosupply": ["--disable_supply_feedback"],
    "ch_memory_nosupply": ["--rider_memory", "--disable_supply_feedback"],
    # Capacity slack: the calibrated market sits near fleet capacity, which
    # caps what any price cut can win; 1.4x fleet tests that dependence.
    "ch_fleet_slack": ["--fleet_scale", "1.4"],
}
WELFARE_KEYS = (
    "welfare_consumer_surplus_per_request", "welfare_total_surplus_per_request",
    "welfare_served_rate", "welfare_mean_fare_paid", "equity_served_gap_high_minus_low",
    "equity_surplus_per_request__<50k", "equity_surplus_per_request__200k+",
)
ROLLOUT_METRICS = (
    "objective_reward_mean", "objective_advantage_mean",
    "objective_dominance_rate", "late_objective_advantage",
    "own_profit_per_request", "own_completed_share", "fulfillment",
    *WELFARE_KEYS,
)
BASELINE_METRICS = (
    "rl_objective", "rl_minus_best_static", "rl_minus_myopic",
    "rl_minus_opening_hold",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _t_interval(values: Sequence[float]) -> Dict[str, float]:
    import statistics
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    n = len(clean)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    mean = statistics.fmean(clean)
    if n == 1:
        return {"n": 1, "mean": mean, "lo": float("nan"), "hi": float("nan")}
    sd = statistics.stdev(clean)
    t = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}.get(n, 1.96)
    half = t * sd / math.sqrt(n)
    return {"n": n, "mean": mean, "lo": mean - half, "hi": mean + half}


def _aggregate_rows(
    rows: Sequence[Mapping[str, Any]],
    group_fields: Sequence[str],
    metrics: Sequence[str],
) -> List[Dict[str, Any]]:
    """Aggregate numeric metrics with a 95% t-interval per declared cell."""
    keys = sorted({tuple(row.get(field) for field in group_fields) for row in rows})
    aggregate: List[Dict[str, Any]] = []
    for key in keys:
        group = [
            row for row in rows
            if tuple(row.get(field) for field in group_fields) == key
        ]
        entry: Dict[str, Any] = dict(zip(group_fields, key))
        for metric in metrics:
            interval = _t_interval([
                row.get(metric) for row in group if row.get(metric) is not None
            ])
            entry.update({f"{metric}_{name}": value for name, value in interval.items()})
        aggregate.append(entry)
    return aggregate


def _write_analysis_plots(
    out: Path,
    rollout_aggregate: Sequence[Mapping[str, Any]],
    baseline_aggregate: Sequence[Mapping[str, Any]],
    external_rows: Sequence[Mapping[str, Any]],
) -> List[str]:
    """Render compact confirmatory-result figures from aggregate tables."""
    if not rollout_aggregate and not baseline_aggregate and not external_rows:
        return []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # fail loudly; plotting is part of the phase contract
        raise RuntimeError(
            "the analyze phase requires matplotlib; install requirement.txt"
        ) from exc

    written: List[str] = []

    baseline = [
        row for row in baseline_aggregate
        if row.get("arm") == "llm_numeric"
        and str(row.get("policy", "")).startswith("endpoint_")
    ]
    if baseline:
        labels = [
            f"{row['city']} | {str(row['policy']).replace('endpoint_', '')} | {row['opponent']}"
            + (" | OOD" if row.get("evaluation_set") == "ood_perturbed" else "")
            for row in baseline
        ]
        y = list(range(len(baseline)))
        fig, ax = plt.subplots(figsize=(11, max(5.0, 0.28 * len(labels))))
        for offset, metric, label, color in (
            (-0.14, "rl_minus_best_static", "RL - best static", "#2563eb"),
            (0.14, "rl_minus_myopic", "RL - myopic", "#ea580c"),
        ):
            means = [float(row.get(f"{metric}_mean", float("nan"))) for row in baseline]
            lower = [
                max(0.0, mean - float(row.get(f"{metric}_lo", mean)))
                if math.isfinite(mean) and math.isfinite(float(row.get(f"{metric}_lo", float("nan"))))
                else 0.0
                for mean, row in zip(means, baseline)
            ]
            upper = [
                max(0.0, float(row.get(f"{metric}_hi", mean)) - mean)
                if math.isfinite(mean) and math.isfinite(float(row.get(f"{metric}_hi", float("nan"))))
                else 0.0
                for mean, row in zip(means, baseline)
            ]
            ax.errorbar(means, [value + offset for value in y], xerr=[lower, upper],
                        fmt="o", capsize=3, label=label, color=color)
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_yticks(y, labels)
        ax.set_xlabel("Objective advantage (mean and 95% t-interval)")
        ax.set_title("v26 policy advantage over Firm-1-seat baselines")
        ax.legend(loc="best")
        fig.tight_layout()
        path = out / "v26_baseline_advantage.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        written.append(str(path))

    rollout = [
        row for row in rollout_aggregate
        if row.get("arm") == "llm_numeric"
        and str(row.get("policy", "")).startswith("endpoint_")
        and bool(row.get("held_out_benchmark"))
    ]
    if rollout:
        labels = [
            f"{row['city']} | {str(row['policy']).replace('endpoint_', '')} | {row['opponent']}"
            for row in rollout
        ]
        means = [float(row.get("objective_advantage_mean_mean", float("nan"))) for row in rollout]
        colors = ["#15803d" if value >= 0.0 else "#b91c1c" for value in means]
        fig, ax = plt.subplots(figsize=(11, max(5.0, 0.26 * len(labels))))
        ax.barh(range(len(labels)), means, color=colors)
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("Mean objective advantage")
        ax.set_title("Held-out and OOD benchmark performance")
        fig.tight_layout()
        path = out / "v26_held_out_objective_advantage.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        written.append(str(path))

    if external_rows:
        labels = [str(row["source"]) for row in external_rows]
        values = [float(row.get("spearman_predicted_vs_observed", float("nan"))) for row in external_rows]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(labels, values, color="#7c3aed")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_ylabel("Spearman predicted vs observed")
        ax.set_title("Chicago external threshold benchmark")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        path = out / "v26_external_threshold_benchmark.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        written.append(str(path))

    return written


class Experiment:
    def __init__(self, root: Path, python: str, jobs: int, dry_run: bool, seeds: Sequence[int]):
        self.root = root
        self.python = python
        self.jobs = max(1, int(jobs))
        self.dry_run = bool(dry_run)
        self.seeds = tuple(int(s) for s in seeds)
        if not self.seeds:
            raise ValueError("at least one experiment seed is required")
        self.lock = threading.Lock()
        for child in ("logs", "thresholds", "runs", "models", "analyzed", "external"):
            (self.root / child).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def _run(self, key: str, command: Sequence[str], log: Path, env: Optional[Mapping[str, str]] = None) -> None:
        record = {"task": key, "command": list(command), "started_utc": _now(), "dry_run": self.dry_run}
        with self.lock:
            with (self.root / "commands.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
        if self.dry_run:
            print("[dry-run]", key, " ".join(command), flush=True)
            return
        log.parent.mkdir(parents=True, exist_ok=True)
        started = datetime.now(timezone.utc)
        with log.open("w", encoding="utf-8") as handle:
            result = subprocess.run(list(command), cwd=PROJECT_ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                    text=True, env=dict(os.environ if env is None else env))
        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        with self.lock:
            with (self.root / "commands.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"task": key, "returncode": result.returncode, "elapsed_seconds": elapsed}) + "\n")
        if result.returncode != 0:
            raise RuntimeError(f"{key} failed (rc={result.returncode}); see {log}")
        print(f"[done] {key} {elapsed / 3600:.2f}h", flush=True)

    def calibration(self, city: str) -> Path:
        return PROJECT_ROOT / "data" / "processed" / city / "calibration.json"

    def numeric_cache(self, city: str, label: str = "primary") -> Path:
        return self.root / "thresholds" / city / f"llm_numeric_{label}.jsonl"

    def _validate_numeric_cache(
        self, city: str, label: str = "primary", expected_rows: int = 20000,
    ) -> Dict[str, Any]:
        """Fail closed on incomplete, fallback, or mismatched numeric profiles."""
        path = self.numeric_cache(city, label)
        if not path.is_file():
            raise FileNotFoundError(f"Missing numeric threshold cache: {path}")
        count = 0
        clipped = 0
        errors: List[str] = []
        expected_market = CITIES[city]["market"]
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                count += 1
                try:
                    row = json.loads(line)
                    threshold = float(row["PriceThreshold"])
                    raw = float(row["ThresholdRawLLMValue"])
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    errors.append(f"line {line_number}: malformed numeric threshold ({exc})")
                    continue
                if row.get("ProfileMarket") != expected_market:
                    errors.append(f"line {line_number}: market={row.get('ProfileMarket')!r}")
                if row.get("ThresholdPromptProtocol") != "numeric_profile_threshold_v11":
                    errors.append(f"line {line_number}: wrong prompt protocol")
                if row.get("ThresholdPromptModel") != PROMPT_MODEL:
                    errors.append(f"line {line_number}: wrong prompt model")
                if row.get("PriceThresholdSource") != "llm_numeric":
                    errors.append(f"line {line_number}: fallback/non-numeric source")
                if not (math.isfinite(threshold) and 0.50 <= threshold <= 5.00):
                    errors.append(f"line {line_number}: threshold out of bounds")
                if not math.isfinite(raw):
                    errors.append(f"line {line_number}: non-finite raw LLM value")
                clipped += int(bool(row.get("ThresholdLLMClipped", False)))
                if len(errors) >= 20:
                    break
        if count != int(expected_rows):
            errors.append(f"expected {expected_rows} rows, found {count}")
        if errors:
            raise ValueError(f"Invalid numeric threshold cache {path}: " + "; ".join(errors))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        report = {
            "path": str(path), "rows": count, "clipped_rows": clipped, "sha256": digest,
        }
        print(f"[validated] {city}/{label} {json.dumps(report, sort_keys=True)}", flush=True)
        return report

    def rubric_cache(self, city: str) -> Path:
        return V25_EXPERIMENT / "raw" / "thresholds" / city / "primary.jsonl"

    def prefix(self, city: str, label: str, seed: int, arm: str = "llm_numeric") -> Path:
        return self.root / "runs" / city / arm / f"{label}_seed{seed}"

    # ------------------------------------------------------------------
    def prepare(self) -> None:
        self._run("prepare", [self.python, "scripts/prepare_city_data.py", "--city", "all",
                              "--sample-rows", "5000", "--calibration-ride-rows", "20000",
                              "--trip-bank-rows", "50000"], self.root / "logs" / "prepare.log")

    def external(self) -> None:
        sources = [f"rubric_v10_audited={self.rubric_cache('chicago')}"]
        if self.numeric_cache("chicago").exists() or self.dry_run:
            sources.append(f"llm_numeric={self.numeric_cache('chicago')}")
        command = [self.python, "scripts/external_threshold_benchmark.py",
                   "--out", str(self.root / "external" / "chicago")]
        for source in sources:
            command.extend(["--threshold-source", source])
        self._run("external:chicago", command, self.root / "logs" / "external_chicago.log")

    def _profile_command(self, city: str, label: str, size: int, repeat_source: Path) -> List[str]:
        out = self.numeric_cache(city, label)
        prefix = self.root / "thresholds" / city / f"bootstrap_{label}"
        return [
            self.python, "Core.py", "--market", CITIES[city]["market"], "--model", PROMPT_MODEL,
            "--seed", str(self.seeds[0]), "--days", "1", "--timesteps", "1", "--customers", "1",
            "--pool", str(size), "--threshold_profile_source", "generated",
            "--threshold_cache_path", str(out), "--gpt_threshold_batch_size", "20",
            "--gpt_threshold_coldstart_rides", "5", "--gpt_threshold_mode", "llm_numeric",
            "--threshold_prompt_repeat_source", str(repeat_source),
            "--calibration_json", str(self.calibration(city)), "--v26",
            "--out", str(prefix) + "_market.csv", "--report_prefix", str(prefix.relative_to(PROJECT_ROOT)),
        ]

    def profiles(self) -> None:
        if not os.environ.get("OPENAI_API_KEY") and not self.dry_run:
            raise SystemExit("profiles phase requires OPENAI_API_KEY (numeric CrowdLLM elicitation)")
        for city in CITIES:
            for label, size in (("primary", 20000), ("repeat_1", 2000), ("repeat_2", 2000)):
                if self.numeric_cache(city, label).exists():
                    self._validate_numeric_cache(city, label, size)
                    print(f"[skip] {city}/{label} validated cache exists", flush=True)
                    continue
                self._run(f"profiles:{city}:{label}",
                          self._profile_command(city, label, size, self.rubric_cache(city)),
                          self.root / "logs" / f"profiles_{city}_{label}.log")
                self._validate_numeric_cache(city, label, size)

    # ------------------------------------------------------------------
    def _train_command(self, city: str, policy: Mapping[str, str], seed: int, arm: str) -> List[str]:
        prefix = self.prefix(city, policy["label"], seed, arm)
        model = self.root / "models" / city / arm / f"{policy['label']}_seed{seed}.pt"
        cache = self.rubric_cache(city) if arm in ABLATIONS else self.numeric_cache(city)
        endpoint = policy["label"].startswith("endpoint")
        command = [
            self.python, "Core.py", "--market", CITIES[city]["market"], "--choice_mode", "cognitive",
            "--firm1_mode", "RL", "--firm2_mode", "adaptive_best_response_aggressive",
            "--run_experiment", "--credit_assignment_v25", "--v26",
            "--deterministic_experiment_seed", "--deterministic_torch",
            "--seed", str(seed), "--experiment_seed", str(seed),
            "--policy_objective", policy["objective"],
            "--pool", "20000", "--threshold_profile_source", "cached", "--threshold_cache_path", str(cache),
            "--calibration_json", str(self.calibration(city)),
            "--train_timesteps", str(TRAIN_DAYS), "--train_customers", "100",
            "--train_steps_per_day", str(STEPS_PER_DAY),
            "--firm1_action_interval_steps", "1", "--firm2_action_interval_steps", "1",
            "--ppo_update_interval_days", "5", "--ppo_min_rollout_transitions", str(POPULATION_DECISIONS),
            "--ppo_batch_size", "64", "--ppo_update_epochs", "8",
            "--training_stage_market_fraction", "0.05", "--training_stage_static_fraction", "0.05",
            "--training_stage_dynamic_fraction", "0.90", "--training_dynamic_episode_days", "15",
            "--training_opening_tariff_randomization", "0.15",
            "--checkpoint_validation_interval_days", "610", "--checkpoint_validation_horizon", "600",
            "--checkpoint_validation_customers", "100", "--checkpoint_validation_openings", "2",
            "--eval_timesteps", str(EVAL_STEPS), "--eval_customers", "100",
            "--eval_policy_mode", "argmax_raw",
            "--report_prefix", str(prefix.relative_to(PROJECT_ROOT)),
            "--trained_model_out", str(model),
            "--trained_model_id", f"{EXPERIMENT_ID}-{city}-{arm}-{policy['label']}-seed{seed}",
            "--model_registry_dir", str(self.root / "registry" / city / arm / policy["label"]),
            "--compare_with_dataset", "--dataset_root", str(PROJECT_ROOT / CITIES[city]["ride_root"]),
            "--dataset_city", CITIES[city]["market"],
            "--weather_root", str(PROJECT_ROOT / "data" / "raw" / "weather" / "noaa"),
            "--comparison_out", str(prefix) + "_observed_price_comparison.csv",
            "--comparison_plot_prefix", str(Path(str(prefix) + "_observed_price").relative_to(PROJECT_ROOT)),
            "--comparison_limit", "5000", "--dataset_preview_rows", "0",
            "--comparison_policy_mode", "argmax_raw",
            "--comparison_policy_seed", str(POLICY_SEED), "--comparison_dataset_seed", str(DATASET_SEED),
        ]
        if arm in ABLATIONS:
            command.extend(["--threshold_ablation", "as_loaded" if arm == "rubric" else arm])
        elif arm == "llm_numeric" or arm in CHANNEL_ARMS:
            command.extend(["--gpt_threshold_mode", "llm_numeric"])
        command.extend(CHANNEL_ARMS.get(arm, []))
        if endpoint:
            # Held-out benchmarks, perturbed OOD rivals, and Firm-1-seat
            # baselines (static grid, opening hold, myopic oracle) on the
            # identical rollout seeds.
            command.extend(["--benchmark_suite"])
        return command

    def cells(self, which: str) -> List[tuple]:
        out: List[tuple] = []
        if which in {"main", "all"}:
            out += [(c, p, s, "llm_numeric") for c in CITIES for p in ENDPOINTS for s in self.seeds]
        if which in {"mixtures", "all"}:
            out += [(c, p, s, "llm_numeric") for c in CITIES for p in MIXTURES for s in self.seeds]
        if which in {"ablation", "all"}:
            out += [(c, p, s, arm) for c in CITIES for p in ENDPOINTS for s in self.seeds for arm in ABLATIONS]
        if which in {"channels", "all"}:
            out += [(c, p, s, arm) for c in CITIES for p in ENDPOINTS for s in self.seeds for arm in CHANNEL_ARMS]
        return out

    def train(
        self,
        which: str,
        *,
        selected_cities: Sequence[str] = (),
        selected_policies: Sequence[str] = (),
    ) -> None:
        jobs = self.cells(which)
        city_filter = {str(city).strip() for city in selected_cities if str(city).strip()}
        policy_filter = {
            str(policy).strip() for policy in selected_policies if str(policy).strip()
        }
        unknown_cities = city_filter - set(CITIES)
        known_policies = {
            policy["label"] for policy in (*ENDPOINTS, *MIXTURES)
        }
        unknown_policies = policy_filter - known_policies
        if unknown_cities:
            raise ValueError(f"unknown training cities: {sorted(unknown_cities)}")
        if unknown_policies:
            raise ValueError(f"unknown training policies: {sorted(unknown_policies)}")
        if city_filter:
            jobs = [job for job in jobs if job[0] in city_filter]
        if policy_filter:
            jobs = [job for job in jobs if job[1]["label"] in policy_filter]
        if not jobs:
            raise ValueError("training selection produced no experiment cells")
        if not self.dry_run:
            numeric_cities = {
                city for city, _, _, arm in jobs
                if arm == "llm_numeric" or arm in CHANNEL_ARMS
            }
            for city in sorted(numeric_cities):
                self._validate_numeric_cache(city)

        def execute(city, policy, seed, arm):
            prefix = self.prefix(city, policy["label"], seed, arm)
            completion_path = Path(str(prefix) + "_complete.json")
            if completion_path.exists():
                return f"skip {city}/{arm}/{policy['label']}/seed{seed}"
            key = f"train:{city}:{arm}:{policy['label']}:seed{seed}"
            self._run(key, self._train_command(city, policy, seed, arm),
                      self.root / "logs" / f"train_{city}_{arm}_{policy['label']}_seed{seed}.log")
            if not self.dry_run:
                _atomic_json(completion_path, {
                    "task": key,
                    "completed_utc": _now(),
                    "run_config": str(Path(str(prefix) + "_run_config.json")),
                })
            return key

        with ThreadPoolExecutor(max_workers=self.jobs) as pool:
            futures = [pool.submit(execute, *job) for job in jobs]
            for future in as_completed(futures):
                print("[completed]", future.result(), flush=True)

    # ------------------------------------------------------------------
    def analyze(self) -> None:
        rows: List[Dict[str, Any]] = []
        baseline_rows: List[Dict[str, Any]] = []
        for city in CITIES:
            for arm in ("llm_numeric", *ABLATIONS, *CHANNEL_ARMS):
                for policy in (*ENDPOINTS, *MIXTURES):
                    for seed in self.seeds:
                        prefix = self.prefix(city, policy["label"], seed, arm)
                        config_path = Path(str(prefix) + "_run_config.json")
                        completion_path = Path(str(prefix) + "_complete.json")
                        if not config_path.exists() or not completion_path.exists():
                            continue
                        config = json.loads(config_path.read_text(encoding="utf-8"))
                        bench = config.get("benchmark_evaluation") or {}
                        summaries = bench.get("rollout_summaries", {})
                        ood = (config.get("v26_ood_evaluation") or {}).get("rollout_summaries", {})
                        for name, summary in {**summaries, **{f"ood::{k}": v for k, v in ood.items()}}.items():
                            rows.append({
                                "city": city, "arm": arm, "policy": policy["label"], "seed": seed,
                                "opponent": name,
                                "held_out_benchmark": name.split("@")[0] in BENCHMARKS or name.startswith("ood::"),
                                **{k: summary.get(k) for k in ROLLOUT_METRICS},
                            })
                        baseline_report = config.get("v26_baseline_evaluation") or {}
                        for evaluation_set in (
                            "in_distribution_and_benchmarks", "ood_perturbed"
                        ):
                            comparison = (
                                (baseline_report.get(evaluation_set) or {})
                                .get("comparison") or {}
                            )
                            for opponent, comp in comparison.items():
                                baseline_rows.append({
                                    "city": city, "arm": arm,
                                    "policy": policy["label"], "seed": seed,
                                    "evaluation_set": evaluation_set,
                                    "opponent": opponent, **comp,
                                })
        out = self.root / "analyzed"
        _write_csv(out / "v26_rollout_summaries.csv", rows)
        _write_csv(out / "v26_baseline_comparisons.csv", baseline_rows)
        rollout_aggregate = _aggregate_rows(
            rows,
            ("city", "arm", "policy", "opponent", "held_out_benchmark"),
            ROLLOUT_METRICS,
        )
        baseline_aggregate = _aggregate_rows(
            baseline_rows,
            ("city", "arm", "policy", "evaluation_set", "opponent"),
            BASELINE_METRICS,
        )
        _write_csv(out / "v26_rollout_across_seeds.csv", rollout_aggregate)
        _write_csv(out / "v26_baseline_across_seeds.csv", baseline_aggregate)

        external_rows: List[Dict[str, Any]] = []
        external_path = self.root / "external" / "chicago" / "chicago_shared_trip_benchmark.json"
        if external_path.exists():
            external = json.loads(external_path.read_text(encoding="utf-8"))
            for source, summary in (external.get("sources") or {}).items():
                row: Dict[str, Any] = {
                    "benchmark": external.get("benchmark"),
                    "trips_used": external.get("trips_used"),
                    "pumas_used": external.get("pumas_used"),
                    "source": source,
                }
                row.update({
                    key: value for key, value in summary.items()
                    if not isinstance(value, (dict, list))
                })
                row.update({
                    f"composition_only_{key}": value
                    for key, value in (summary.get("composition_only") or {}).items()
                })
                external_rows.append(row)
        _write_csv(out / "v26_external_benchmark.csv", external_rows)
        plots = _write_analysis_plots(
            out, rollout_aggregate, baseline_aggregate, external_rows
        )
        _atomic_json(out / "analysis_manifest.json", {
            "created_utc": _now(),
            "rollout_rows": len(rows),
            "rollout_aggregate_rows": len(rollout_aggregate),
            "baseline_rows": len(baseline_rows),
            "baseline_aggregate_rows": len(baseline_aggregate),
            "external_rows": len(external_rows),
            "plots": plots,
        })
        print(
            f"[analyze] {len(rows)} rollout rows, {len(baseline_rows)} baseline rows, "
            f"{len(external_rows)} external rows, {len(plots)} plots -> {out}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--phase", choices=("prepare", "external", "profiles", "train", "analyze"), required=True)
    parser.add_argument("--cells", choices=("main", "mixtures", "ablation", "channels", "all"), default="main")
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "artifacts" / EXPERIMENT_ID))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    parser.add_argument(
        "--cities",
        default="",
        help="optional comma-separated training-city keys (nyc,boston,chicago)",
    )
    parser.add_argument(
        "--policies",
        default="",
        help="optional comma-separated training policy labels",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    experiment = Experiment(Path(args.output_root).resolve(), args.python, args.jobs, args.dry_run,
                            [int(s) for s in args.seeds.split(",") if s.strip()])
    if args.phase == "train":
        experiment.train(
            args.cells,
            selected_cities=[item for item in args.cities.split(",") if item.strip()],
            selected_policies=[item for item in args.policies.split(",") if item.strip()],
        )
    else:
        getattr(experiment, args.phase)()


if __name__ == "__main__":
    main()
