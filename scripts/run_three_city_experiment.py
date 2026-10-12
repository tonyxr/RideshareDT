#!/usr/bin/env python3
"""Run the paper's three-part NYC/Boston/Chicago experiment reproducibly."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLBACKEND", "Agg")
_MPLCONFIGDIR = PROJECT_ROOT / "artifacts" / "three_city_full_20260825" / "matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

from analysis.three_part_experiment_report import (  # noqa: E402
    analyze_benchmarks,
    analyze_crowdllm,
    analyze_crowdllm_robustness,
    analyze_dataset_comparisons,
    analyze_training_progress,
)
from city_data_pipeline import resolve_city_spec  # noqa: E402


EXPERIMENT_ID = "three_city_full_20260825"
PRIMARY_SEED = 4500
ROBUSTNESS_SEEDS = (99173, 27183)
PROMPT_REPEAT_LABELS = ("prompt_repeat_1", "prompt_repeat_2")
PROMPT_REPEAT_MIN_GPT_RATE = 0.95
PROMPT_REPEAT_MAX_KS = 0.15
PROMPT_REPEAT_MAX_ABS_MEAN_SHIFT = 0.20
PROMPT_REPEAT_MAX_PAIRED_MAE = 0.35
PROMPT_REPEAT_MIN_PAIRED_CORRELATION = 0.50
PROMPT_REPEAT_MIN_CONTEXT_EXACT_MATCH = 0.75
CROWDLLM_PRODUCTION_BATCH_SIZE = 1
CROWDLLM_MIN_INCOME_MONOTONE_STEPS = 3
CROWDLLM_MIN_HIGH_LOW_INCOME_GAP = 1.25
CROWDLLM_MIN_RETURNING_NEW_GAP = 0.25
CROWDLLM_MAX_PROTECTED_GENDER_GAP = 0.25
CROWDLLM_MIN_CONTEXT_CLASSES = 2
CROWDLLM_MAX_CONTEXT_CLASS_RATE = 0.90
CROWDLLM_MIN_CONTEXT_CLASSIFICATION_ACCURACY = 1.00
CROWDLLM_MIN_RAW_CONTEXT_CLASSIFICATION_ACCURACY = 0.85
CROWDLLM_MAX_CONTEXT_CORRECTION_RATE = 0.15
CROWDLLM_PROMPT_MODEL = "gpt-4.1"
DATASET_SEED = 104729
POLICY_SEED = 99173
BENCHMARKS = (
    "pi_price_gap",
    "mpc_grid",
    "region_supply_demand",
    "surge_driver_incentive",
    "queue_service_threshold",
)
CITIES: Mapping[str, Mapping[str, str]] = {
    "nyc": {
        "market": "New York City",
        "ride_root": "data/raw/rides/nyc_kaggle",
        "ride_semantics": "realized_trip",
    },
    "boston": {
        "market": "Boston",
        "ride_root": "data/raw/rides/boston_kaggle",
        "ride_semantics": "price_quote",
    },
    "chicago": {
        "market": "Chicago",
        "ride_root": "data/raw/rides/chicago_kaggle",
        "ride_semantics": "realized_trip",
    },
}
POLICIES: Sequence[Mapping[str, Any]] = (
    {"label": "endpoint_profit", "objective": "profit_maximization", "benchmark": True},
    {"label": "endpoint_competitiveness", "objective": "competitiveness", "benchmark": True},
    {"label": "p10c90", "objective": "profit_10_competitiveness_90", "benchmark": False},
    {"label": "p30c70", "objective": "profit_30_competitiveness_70", "benchmark": False},
    {"label": "p50c50", "objective": "profit_50_competitiveness_50", "benchmark": False},
    {"label": "p70c30", "objective": "profit_70_competitiveness_30", "benchmark": False},
    {"label": "p90c10", "objective": "profit_90_competitiveness_10", "benchmark": False},
)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Experiment:
    def __init__(self, root: Path, python: str, profile_size: int, jobs: int, dry_run: bool):
        self.root = root
        self.python = python
        self.profile_size = int(profile_size)
        self.jobs = int(max(1, jobs))
        self.dry_run = bool(dry_run)
        self.status_path = self.root / "status.json"
        self.command_path = self.root / "commands.jsonl"
        self.lock = threading.Lock()
        self.root.mkdir(parents=True, exist_ok=True)
        for child in ("raw/logs", "raw/thresholds", "raw/runs", "models", "registry", "analyzed"):
            (self.root / child).mkdir(parents=True, exist_ok=True)

    def _record(self, item: Mapping[str, Any]) -> None:
        with self.lock:
            with self.command_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(dict(item), sort_keys=True) + "\n")

    def _set_status(self, key: str, state: str, detail: str = "") -> None:
        with self.lock:
            payload: Dict[str, Any] = {}
            if self.status_path.exists():
                payload = json.loads(self.status_path.read_text(encoding="utf-8"))
            payload.setdefault("experiment_id", EXPERIMENT_ID)
            payload.setdefault("tasks", {})[key] = {
                "state": state,
                "detail": detail,
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            }
            _atomic_json(self.status_path, payload)

    def _run(self, key: str, command: Sequence[str], log_path: Path, env: Mapping[str, str] | None = None) -> None:
        self._record({
            "task": key,
            "command": list(command),
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "dry_run": self.dry_run,
        })
        if self.dry_run:
            print("[dry-run]", key, " ".join(command), flush=True)
            return
        self._set_status(key, "running")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started = datetime.now(timezone.utc)
        try:
            with log_path.open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    list(command),
                    cwd=PROJECT_ROOT,
                    env=dict(os.environ if env is None else env),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
        except KeyboardInterrupt:
            self._set_status(key, "interrupted", f"log={log_path}")
            raise
        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        self._record({"task": key, "returncode": result.returncode, "elapsed_seconds": elapsed})
        if result.returncode != 0:
            self._set_status(key, "failed", f"returncode={result.returncode}; log={log_path}")
            raise RuntimeError(f"{key} failed; inspect {log_path}")
        self._set_status(key, "completed", f"elapsed_seconds={elapsed:.1f}")

    def calibration(self, city: str) -> Path:
        return PROJECT_ROOT / "data" / "processed" / city / "calibration.json"

    def threshold_cache(self, city: str, label: str = "primary") -> Path:
        return self.root / "raw" / "thresholds" / city / f"{label}.jsonl"

    def run_prefix(self, city: str, label: str) -> Path:
        return self.root / "raw" / "runs" / city / label

    def model_path(self, city: str, label: str) -> Path:
        return self.root / "models" / city / f"{label}.pt"

    @staticmethod
    def _core_relative(path: Path) -> str:
        """Keep Core's report-prefix safety normalization inside the experiment tree."""
        try:
            return str(path.resolve().relative_to(PROJECT_ROOT))
        except ValueError as exc:
            raise ValueError(f"Core output must remain under {PROJECT_ROOT}: {path}") from exc

    def write_manifest(self) -> None:
        code_files = (
            "Core.py",
            "MarketInteraction.py",
            "GenerateAgent.py",
            "PPOAgent.py",
            "platform_mdp.py",
            "pricing_models.py",
            "city_data_pipeline.py",
            "analysis/three_part_experiment_report.py",
            "scripts/audit_threshold_caches.py",
            "scripts/run_three_city_experiment.py",
        )
        sources: List[Dict[str, Any]] = []
        for city in CITIES:
            report_path = PROJECT_ROOT / "data" / "processed" / city / "preprocessing_report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            sources.extend({"city": city, **item} for item in report["sources"])
        manifest = {
            "schema_version": 1,
            "experiment_id": EXPERIMENT_ID,
            "frozen_utc": datetime.now(timezone.utc).isoformat(),
            "research_questions": {
                "part1": "City-conditioned CrowdLLM profiles and threshold construct/stability diagnostics.",
                "part2": "Endpoint policies against five fixed benchmark controllers on matched held-out trajectories.",
                "part3": "Five objective-mixture policies on the same 5,000 observed source records per city.",
            },
            "cities": CITIES,
            "policies": list(POLICIES),
            "benchmark_controllers": BENCHMARKS,
            "seeds": {
                "training": PRIMARY_SEED,
                "crowdllm_prompt_repeat_profile_seed": PRIMARY_SEED,
                "crowdllm_prompt_repeats": list(PROMPT_REPEAT_LABELS),
                "dataset_robustness": list(ROBUSTNESS_SEEDS),
                "observed_dataset": DATASET_SEED,
                "policy_scoring": POLICY_SEED,
            },
            "sample_sizes": {
                "primary_profiles_per_city": self.profile_size,
                "prompt_repeat_profiles_per_city": 2000,
                "rationale_profiles_per_city": 200,
                "observed_records_per_city": 5000,
            },
            "crowdllm_threshold_evidence": {
                "protocol": "city_reference_tariffs_v1",
                "prompt_protocol": "structured_context_rubric_v10",
                "prompt_model": CROWDLLM_PROMPT_MODEL,
                "api_temperature": 0.0,
                "firm1_city_reference_multiplier": 1.00,
                "firm2_city_reference_multiplier": 1.10,
                "live_random_opening_tariffs_excluded": True,
            },
            "crowdllm_prompt_repeat_acceptance": {
                "minimum_gpt_completion_rate": PROMPT_REPEAT_MIN_GPT_RATE,
                "maximum_ks_distance": PROMPT_REPEAT_MAX_KS,
                "maximum_absolute_mean_shift_dollars": PROMPT_REPEAT_MAX_ABS_MEAN_SHIFT,
                "maximum_paired_mae_dollars": PROMPT_REPEAT_MAX_PAIRED_MAE,
                "minimum_paired_correlation": PROMPT_REPEAT_MIN_PAIRED_CORRELATION,
                "minimum_context_adjustment_exact_match_rate": PROMPT_REPEAT_MIN_CONTEXT_EXACT_MATCH,
                "criteria_frozen_before_pilot_review": True,
            },
            "crowdllm_construct_acceptance": {
                "production_batch_size": CROWDLLM_PRODUCTION_BATCH_SIZE,
                "deduplication": "one API classification per unique four-count context signature",
                "maximum_possible_signatures_for_five_rides": 1296,
                "minimum_income_adjacent_monotone_steps_of_3": CROWDLLM_MIN_INCOME_MONOTONE_STEPS,
                "minimum_high_minus_low_income_threshold_dollars": CROWDLLM_MIN_HIGH_LOW_INCOME_GAP,
                "minimum_returning_minus_new_threshold_dollars": CROWDLLM_MIN_RETURNING_NEW_GAP,
                "maximum_absolute_protected_gender_gap_dollars": CROWDLLM_MAX_PROTECTED_GENDER_GAP,
                "minimum_context_adjustment_classes": CROWDLLM_MIN_CONTEXT_CLASSES,
                "maximum_largest_context_class_rate": CROWDLLM_MAX_CONTEXT_CLASS_RATE,
                "minimum_audited_context_classification_accuracy": CROWDLLM_MIN_CONTEXT_CLASSIFICATION_ACCURACY,
                "minimum_raw_gpt_context_classification_accuracy": CROWDLLM_MIN_RAW_CONTEXT_CLASSIFICATION_ACCURACY,
                "maximum_context_correction_rate": CROWDLLM_MAX_CONTEXT_CORRECTION_RATE,
                "audit_protocol": "deterministic_context_validation_v1",
                "criteria_frozen_before_batch_size_pilot_review": True,
            },
            "training": {
                "days": 9150,
                "steps_per_day": 4,
                "customers_per_step": 100,
                "stages": [0.05, 0.05, 0.90],
                "dynamic_episode_days": 15,
                "action_interval_steps": 5,
                "ppo_update_interval_days": 10,
                "ppo_min_rollout_transitions": 732,
                "ppo_batch_size": 48,
                "ppo_epochs": 8,
                "mixture_checkpoint_interval_days": 915,
                "mixture_checkpoint_horizon": 300,
                "endpoint_checkpoint_interval_days": 1830,
                "endpoint_checkpoint_horizon": 1000,
                "checkpoint_openings": 2,
                "final_evaluation_steps": 1000,
                "final_evaluation_customers": 200,
                "final_five_controller_benchmark_steps": 1000,
                "actor_initialization": "zero_logits_uniform_prior",
                "deterministic_prelearning_tie_break": "hold",
                "static_policy_is_legal": True,
                "movement_reward_or_action_quota": False,
                "constrained_mdp": False,
            },
            "interpretation_guardrail": (
                "Boston rows are posted fare quotes, not completed trips; its Part 3 result is a quote-scale diagnostic."
            ),
            "environment": {
                "python_executable": self.python,
                "python_version": platform.python_version(),
                "platform": platform.platform(),
            },
            "code_sha256": {
                name: _sha256(PROJECT_ROOT / name) for name in code_files
            },
            "source_manifest": sources,
        }
        _atomic_json(self.root / "experiment_manifest.json", manifest)

    def preflight(self) -> None:
        self.write_manifest()
        for city, settings in CITIES.items():
            calibration = json.loads(self.calibration(city).read_text(encoding="utf-8"))
            if calibration["city"] != settings["market"]:
                raise ValueError(f"{city} calibration market mismatch")
            if calibration["ride_semantics"] != settings["ride_semantics"]:
                raise ValueError(f"{city} ride semantics mismatch")
            market = calibration["calibration"]["market"]
            if max(market["hour_probs"].values()) > 0.20:
                raise ValueError(f"{city} calibration is concentrated in one hour")
            if max(market["day_of_week_probs"].values()) > 0.35:
                raise ValueError(f"{city} calibration is concentrated in one day")
            resolve_city_spec(settings["market"])
        self._set_status("preflight", "completed", "three city contracts validated")

    def _profile_command(self, city: str, seed: int, size: int, label: str, rationales: bool) -> List[str]:
        settings = CITIES[city]
        prefix = self.root / "raw" / "thresholds" / city / f"bootstrap_{label}"
        command = [
            self.python, "Core.py",
            "--market", settings["market"],
            "--model", CROWDLLM_PROMPT_MODEL,
            "--seed", str(seed),
            "--days", "1", "--timesteps", "1", "--customers", "1",
            "--pool", str(size),
            "--threshold_profile_source", "generated",
            "--threshold_cache_path", str(self.threshold_cache(city, label)),
            "--gpt_threshold_batch_size", str(CROWDLLM_PRODUCTION_BATCH_SIZE),
            "--gpt_threshold_coldstart_rides", "5",
            "--calibration_json", str(self.calibration(city)),
            "--out", str(prefix) + "_market.csv",
            "--profiles_out", str(prefix) + "_profiles.csv",
            "--report_prefix", self._core_relative(prefix),
        ]
        if label in {*PROMPT_REPEAT_LABELS, "rationale_audit"}:
            command.extend([
                "--threshold_prompt_repeat_source",
                str(self.threshold_cache(city, "primary")),
            ])
        if rationales:
            command.append("--gpt_threshold_include_rationales")
        return command

    @staticmethod
    def _validate_profiles(path: Path, market: str, expected: int, require_rationale: bool) -> None:
        count = gpt = rationale = raw_correct = audited_correct = corrections = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                count += 1
                gpt += int(row.get("PriceThresholdSource") == "gpt")
                rationale += int(bool(str(row.get("PriceThresholdRationale", "")).strip()))
                if row.get("ProfileMarket") != market:
                    raise ValueError(f"profile market mismatch in {path}")
                if row.get("ThresholdEvidenceProtocol") != "city_reference_tariffs_v1":
                    raise ValueError(f"obsolete threshold evidence protocol in {path}")
                if row.get("ThresholdPromptProtocol") != "structured_context_rubric_v10":
                    raise ValueError(f"obsolete threshold prompt protocol in {path}")
                if row.get("ThresholdPromptModel") != CROWDLLM_PROMPT_MODEL:
                    raise ValueError(f"obsolete threshold prompt model in {path}")
                if row.get("ThresholdAuditProtocol") != "deterministic_context_validation_v1":
                    raise ValueError(f"obsolete threshold audit protocol in {path}")
                expected_context = float(row["ThresholdExpectedContextAdjustment"])
                raw_context = float(row["ThresholdRawGPTContextAdjustment"])
                audited_context = float(row["ThresholdContextAdjustment"])
                raw_correct += int(math.isclose(raw_context, expected_context))
                audited_correct += int(math.isclose(audited_context, expected_context))
                corrections += int(bool(row.get("ThresholdContextWasCorrected", False)))
                expected_batch_size = CROWDLLM_PRODUCTION_BATCH_SIZE
                if int(row.get("ThresholdPromptBatchSize", -1)) != expected_batch_size:
                    raise ValueError(f"obsolete threshold batch size in {path}")
                if row.get("ThresholdPromptDeduplication") != "context_signature_v1":
                    raise ValueError(f"obsolete threshold deduplication in {path}")
        if count != expected:
            raise ValueError(f"{path}: expected {expected} profiles, got {count}")
        if gpt / count < 0.95:
            raise ValueError(f"{path}: GPT completion rate {gpt / count:.3f} < 0.95")
        if require_rationale and rationale / count < 0.95:
            raise ValueError(f"{path}: rationale completion rate {rationale / count:.3f} < 0.95")
        if audited_correct / count < CROWDLLM_MIN_CONTEXT_CLASSIFICATION_ACCURACY:
            raise ValueError(f"{path}: audited context accuracy is incomplete")
        if not require_rationale:
            if raw_correct / count < CROWDLLM_MIN_RAW_CONTEXT_CLASSIFICATION_ACCURACY:
                raise ValueError(f"{path}: raw GPT context accuracy {raw_correct / count:.3f} is too low")
            if corrections / count > CROWDLLM_MAX_CONTEXT_CORRECTION_RATE:
                raise ValueError(f"{path}: context correction rate {corrections / count:.3f} is too high")

    @staticmethod
    def _prompt_input_signature(path: Path, limit: int) -> str:
        excluded = {
            "PriceThreshold",
            "PriceThresholdRationale",
            "PriceThresholdSource",
            "ThresholdEvidenceProtocol",
            "ThresholdEvidenceFirm1Multiplier",
            "ThresholdEvidenceFirm2Multiplier",
            "ThresholdPromptProtocol",
            "ThresholdPromptBatchSize",
        }
        digest = hashlib.sha256()
        count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                payload = {
                    key: value for key, value in row.items()
                    if key not in excluded and not str(key).startswith("Threshold")
                }
                digest.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
                digest.update(b"\n")
                count += 1
                if count >= limit:
                    break
        if count != limit:
            raise ValueError(f"{path}: prompt signature has {count} rows, expected {limit}")
        return digest.hexdigest()

    def profiles(self) -> None:
        profile_jobs = []
        for city in CITIES:
            profile_jobs.append((city, PRIMARY_SEED, self.profile_size, "primary", False))
            for label in PROMPT_REPEAT_LABELS:
                # Holding the underlying profiles/rides fixed isolates API and
                # prompt repeatability from population-composition changes.
                profile_jobs.append((city, PRIMARY_SEED, 2000, label, False))
            profile_jobs.append((city, 65537, 200, "rationale_audit", True))
        for city, seed, size, label, rationales in profile_jobs:
            key = f"profiles:{city}:{label}"
            path = self.threshold_cache(city, label)
            try:
                self._validate_profiles(path, CITIES[city]["market"], size, rationales)
                if label in {*PROMPT_REPEAT_LABELS, "rationale_audit"}:
                    if self._prompt_input_signature(path, size) != self._prompt_input_signature(
                        self.threshold_cache(city, "primary"), size
                    ):
                        raise ValueError(
                            f"{path}: prompt inputs differ from the primary population"
                        )
                self._set_status(key, "completed", "validated existing cache")
                continue
            except (FileNotFoundError, ValueError):
                pass
            self._run(
                key,
                self._profile_command(city, seed, size, label, rationales),
                self.root / "raw" / "logs" / f"profiles_{city}_{label}.log",
            )
            if not self.dry_run:
                self._validate_profiles(path, CITIES[city]["market"], size, rationales)
                if label in {*PROMPT_REPEAT_LABELS, "rationale_audit"}:
                    primary_signature = self._prompt_input_signature(
                        self.threshold_cache(city, "primary"), size
                    )
                    repeat_signature = self._prompt_input_signature(path, size)
                    if repeat_signature != primary_signature:
                        raise ValueError(
                            f"{path}: prompt inputs differ from the primary population"
                        )

    def _training_command(self, city: str, policy: Mapping[str, Any]) -> List[str]:
        settings = CITIES[city]
        label = str(policy["label"])
        endpoint_benchmark = bool(policy["benchmark"])
        checkpoint_interval_days = 1830 if endpoint_benchmark else 915
        checkpoint_horizon = 1000 if endpoint_benchmark else 300
        prefix = self.run_prefix(city, label)
        model = self.model_path(city, label)
        command = [
            self.python, "Core.py",
            "--market", settings["market"],
            "--choice_mode", "cognitive",
            "--firm1_mode", "RL",
            "--firm2_mode", "adaptive_best_response_aggressive",
            "--run_experiment", "--credit_assignment_v25",
            "--deterministic_experiment_seed", "--deterministic_torch",
            "--seed", str(PRIMARY_SEED), "--experiment_seed", str(PRIMARY_SEED),
            "--policy_objective", str(policy["objective"]),
            "--pool", str(self.profile_size),
            "--threshold_profile_source", "cached",
            "--threshold_cache_path", str(self.threshold_cache(city)),
            "--calibration_json", str(self.calibration(city)),
            "--train_timesteps", "9150", "--train_customers", "100",
            "--train_steps_per_day", "4",
            "--firm1_action_interval_steps", "5",
            "--firm2_action_interval_steps", "5",
            "--ppo_update_interval_days", "10",
            "--ppo_min_rollout_transitions", "732",
            "--ppo_batch_size", "48", "--ppo_update_epochs", "8",
            "--training_stage_market_fraction", "0.05",
            "--training_stage_static_fraction", "0.05",
            "--training_stage_dynamic_fraction", "0.90",
            "--training_dynamic_episode_days", "15",
            "--training_opening_tariff_randomization", "0.15",
            "--checkpoint_validation_interval_days", str(checkpoint_interval_days),
            "--checkpoint_validation_horizon", str(checkpoint_horizon),
            "--checkpoint_validation_customers", "100",
            "--checkpoint_validation_openings", "2",
            "--eval_timesteps", "1000", "--eval_customers", "200",
            "--eval_policy_mode", "argmax_raw",
            "--report_prefix", self._core_relative(prefix),
            "--trained_model_out", str(model),
            "--trained_model_id", f"{EXPERIMENT_ID}-{city}-{label}-seed{PRIMARY_SEED}",
            "--model_registry_dir", str(self.root / "registry" / city / label),
            "--compare_with_dataset",
            "--dataset_root", str(PROJECT_ROOT / settings["ride_root"]),
            "--dataset_city", settings["market"],
            "--weather_root", str(PROJECT_ROOT / "data" / "raw" / "weather" / "noaa"),
            "--comparison_out", str(prefix) + "_observed_price_comparison.csv",
            "--comparison_plot_prefix", self._core_relative(
                Path(str(prefix) + "_observed_price")
            ),
            "--comparison_limit", "5000", "--dataset_preview_rows", "0",
            "--comparison_policy_mode", "argmax_raw",
            "--comparison_policy_seed", str(POLICY_SEED),
            "--comparison_dataset_seed", str(DATASET_SEED),
        ]
        if endpoint_benchmark:
            command.extend(("--benchmark_suite", "--require_competitive_backtest"))
        return command

    def _validate_run(self, city: str, policy: Mapping[str, Any], days: int = 9150, observed: int = 5000) -> None:
        label = str(policy["label"])
        prefix = self.run_prefix(city, label)
        required = (
            self.model_path(city, label),
            Path(str(prefix) + "_run_config.json"),
            Path(str(prefix) + "_training_diagnostics.csv"),
            Path(str(prefix) + "_evaluation_diagnostics.csv"),
            Path(str(prefix) + "_observed_price_comparison.csv"),
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(", ".join(missing))
        if len(_read_csv(required[2])) != days:
            raise ValueError(f"{city}/{label}: incomplete training diagnostics")
        if len(_read_csv(required[4])) != observed:
            raise ValueError(f"{city}/{label}: incomplete observed-price comparison")
        config = json.loads(required[1].read_text(encoding="utf-8"))
        args = config["args"]
        if args["market"] != CITIES[city]["market"] or args["policy_objective"] != policy["objective"]:
            raise ValueError(f"{city}/{label}: run contract mismatch")
        if bool(policy["benchmark"]):
            summary = _read_csv(Path(str(prefix) + "_benchmark_summary.csv"))
            if [row["benchmark"] for row in summary] != list(BENCHMARKS):
                raise ValueError(f"{city}/{label}: benchmark suite mismatch")
            for benchmark in BENCHMARKS:
                trajectory = _read_csv(Path(str(prefix) + f"_benchmark_{benchmark}_trajectory.csv"))
                if len(trajectory) != 1000:
                    raise ValueError(f"{city}/{label}/{benchmark}: incomplete trajectory")

    def train(self) -> None:
        jobs = [(city, policy) for city in CITIES for policy in POLICIES]

        def execute(city: str, policy: Mapping[str, Any]) -> str:
            label = str(policy["label"])
            key = f"train:{city}:{label}"
            try:
                self._validate_run(city, policy)
                self._set_status(key, "completed", "validated existing run")
                return key
            except (FileNotFoundError, ValueError, KeyError):
                pass
            self._run(
                key,
                self._training_command(city, policy),
                self.root / "raw" / "logs" / f"train_{city}_{label}.log",
            )
            if not self.dry_run:
                try:
                    self._validate_run(city, policy)
                except Exception as exc:
                    self._set_status(key, "failed", f"post-run validation: {exc}")
                    raise
                self._set_status(key, "completed", "process and artifact validation passed")
            return key

        with ThreadPoolExecutor(max_workers=self.jobs) as executor:
            futures = [executor.submit(execute, city, policy) for city, policy in jobs]
            for future in as_completed(futures):
                print("[completed]", future.result(), flush=True)

    def analyze(self) -> None:
        if self.dry_run:
            print("[dry-run] analyze all completed artifacts", flush=True)
            return
        part1_rows: List[Dict[str, Any]] = []
        part2_rows: List[Dict[str, Any]] = []
        part3_rows: List[Dict[str, Any]] = []
        mixture_policies = [policy for policy in POLICIES if str(policy["label"]).startswith("p")]
        for city, settings in CITIES.items():
            city_out = self.root / "analyzed" / city
            city_out.mkdir(parents=True, exist_ok=True)
            crowd = analyze_crowdllm(
                str(self.threshold_cache(city)), str(city_out), DATASET_SEED
            )
            construct_failures = []
            if int(crowd["income_adjacent_monotone_steps"]) < CROWDLLM_MIN_INCOME_MONOTONE_STEPS:
                construct_failures.append("income monotonicity")
            if float(crowd["income_high_minus_low_dollars"]) < CROWDLLM_MIN_HIGH_LOW_INCOME_GAP:
                construct_failures.append("high-low income gap")
            if float(crowd["returning_minus_new_raw_dollars"]) < CROWDLLM_MIN_RETURNING_NEW_GAP:
                construct_failures.append("returning-new loyalty gap")
            if abs(float(crowd["female_minus_nonfemale_raw_dollars"])) > CROWDLLM_MAX_PROTECTED_GENDER_GAP:
                construct_failures.append("protected-gender neutrality")
            if int(crowd["context_adjustment_unique_values"]) < CROWDLLM_MIN_CONTEXT_CLASSES:
                construct_failures.append("context-class diversity")
            if float(crowd["context_adjustment_largest_class_rate"]) > CROWDLLM_MAX_CONTEXT_CLASS_RATE:
                construct_failures.append("context-class concentration")
            if float(crowd["context_classification_accuracy"]) < CROWDLLM_MIN_CONTEXT_CLASSIFICATION_ACCURACY:
                construct_failures.append("context-classification accuracy")
            if float(crowd["raw_gpt_context_classification_accuracy"]) < CROWDLLM_MIN_RAW_CONTEXT_CLASSIFICATION_ACCURACY:
                construct_failures.append("raw GPT context-classification accuracy")
            if float(crowd["context_correction_rate"]) > CROWDLLM_MAX_CONTEXT_CORRECTION_RATE:
                construct_failures.append("context-correction rate")
            if construct_failures:
                raise ValueError(
                    f"{city} failed frozen CrowdLLM construct criteria: "
                    f"{', '.join(construct_failures)}"
                )
            part1_rows.append({"city": settings["market"], **crowd})
            analyze_crowdllm_robustness(
                str(self.threshold_cache(city)),
                [
                    f"{label}={self.threshold_cache(city, label)}"
                    for label in PROMPT_REPEAT_LABELS
                ],
                str(self.threshold_cache(city, "rationale_audit")),
                str(city_out),
            )
            robustness_rows = _read_csv(
                city_out / "part1_crowdllm_robustness.csv"
            )
            for row in robustness_rows:
                if row["population"] not in PROMPT_REPEAT_LABELS:
                    continue
                failures = []
                if float(row["gpt_source_rate"]) < PROMPT_REPEAT_MIN_GPT_RATE:
                    failures.append("GPT completion rate")
                if float(row["ks_distance_from_primary"]) > PROMPT_REPEAT_MAX_KS:
                    failures.append("KS distance")
                if float(row["absolute_mean_shift_from_primary"]) > PROMPT_REPEAT_MAX_ABS_MEAN_SHIFT:
                    failures.append("absolute mean shift")
                if float(row["paired_threshold_mae"]) > PROMPT_REPEAT_MAX_PAIRED_MAE:
                    failures.append("paired MAE")
                if float(row["paired_threshold_correlation"]) < PROMPT_REPEAT_MIN_PAIRED_CORRELATION:
                    failures.append("paired correlation")
                if float(row["paired_context_exact_match_rate"]) < PROMPT_REPEAT_MIN_CONTEXT_EXACT_MATCH:
                    failures.append("context-adjustment exact match")
                if failures:
                    raise ValueError(
                        f"{city}/{row['population']} failed frozen CrowdLLM repeatability "
                        f"criteria: {', '.join(failures)}"
                    )
            for endpoint in (POLICIES[0], POLICIES[1]):
                endpoint_out = city_out / "part2" / str(endpoint["label"])
                endpoint_out.mkdir(parents=True, exist_ok=True)
                rows = analyze_benchmarks(
                    str(self.run_prefix(city, str(endpoint["label"]))),
                    str(endpoint_out),
                    DATASET_SEED + 1,
                )
                part2_rows.extend({
                    "city": settings["market"],
                    "objective": endpoint["objective"],
                    **row,
                } for row in rows)
            comparison_specs = [
                f"{policy['label']}={self.run_prefix(city, str(policy['label']))}_observed_price_comparison.csv"
                for policy in mixture_policies
            ]
            rows = analyze_dataset_comparisons(
                comparison_specs, str(city_out), DATASET_SEED + 2
            )
            part3_rows.extend({
                "city": settings["market"],
                "ride_semantics": settings["ride_semantics"],
                **row,
            } for row in rows)
            analyze_training_progress(
                [
                    f"{policy['label']}={self.run_prefix(city, str(policy['label']))}"
                    for policy in mixture_policies
                ],
                str(city_out),
            )
        cross = self.root / "analyzed" / "cross_city"
        _write_csv(cross / "part1_city_key_statistics.csv", part1_rows)
        _write_csv(cross / "part2_city_benchmark_key_statistics.csv", part2_rows)
        _write_csv(cross / "part3_city_objective_key_statistics.csv", part3_rows)
        self._set_status("analysis", "completed", "all tables and figures generated")

    def smoke(self) -> None:
        city = "boston"
        smoke_root = self.root / "smoke"
        cache = smoke_root / "profiles.jsonl"
        env = dict(os.environ)
        env["OPENAI_API_KEY"] = ""
        if not cache.exists():
            command = self._profile_command(city, 2601, 100, "smoke", False)
            cache_target = self.threshold_cache(city, "smoke")
            command[command.index(str(cache_target))] = str(cache)
            self._run("smoke:profiles", command, smoke_root / "profiles.log", env=env)
        prefix = smoke_root / "run"
        model = smoke_root / "policy.pt"
        command = [
            self.python, "Core.py", "--market", "Boston",
            "--firm1_mode", "RL", "--firm2_mode", "adaptive_best_response_aggressive",
            "--run_experiment", "--credit_assignment_v25",
            "--deterministic_experiment_seed", "--seed", "2601", "--experiment_seed", "2601",
            "--policy_objective", "profit_50_competitiveness_50",
            "--pool", "100", "--threshold_profile_source", "cached",
            "--threshold_cache_path", str(cache),
            "--calibration_json", str(self.calibration(city)),
            "--train_timesteps", "40", "--train_customers", "20", "--train_steps_per_day", "4",
            "--firm1_action_interval_steps", "5", "--firm2_action_interval_steps", "5",
            "--ppo_update_interval_days", "5", "--ppo_min_rollout_transitions", "8",
            "--ppo_batch_size", "8", "--ppo_update_epochs", "2",
            "--training_stage_market_fraction", "0.10", "--training_stage_static_fraction", "0.10",
            "--training_stage_dynamic_fraction", "0.80", "--training_dynamic_episode_days", "5",
            "--checkpoint_validation_interval_days", "20", "--checkpoint_validation_horizon", "20",
            "--checkpoint_validation_customers", "20", "--checkpoint_validation_openings", "1",
            "--eval_timesteps", "20", "--eval_customers", "20", "--benchmark_suite",
            "--report_prefix", self._core_relative(prefix),
            "--trained_model_out", str(model),
            "--trained_model_id", "three-city-smoke-boston",
            "--model_registry_dir", str(smoke_root / "registry"),
            "--compare_with_dataset", "--dataset_root", str(PROJECT_ROOT / CITIES[city]["ride_root"]),
            "--dataset_city", "Boston", "--weather_root", str(PROJECT_ROOT / "data/raw/weather/noaa"),
            "--comparison_out", str(prefix) + "_observed_price_comparison.csv",
            "--comparison_plot_prefix", self._core_relative(
                Path(str(prefix) + "_observed_price")
            ),
            "--comparison_limit", "20", "--dataset_preview_rows", "0",
        ]
        self._run("smoke:training", command, smoke_root / "training.log")
        if not self.dry_run:
            training_rows = _read_csv(Path(str(prefix) + "_training_diagnostics.csv"))
            benchmarks = _read_csv(Path(str(prefix) + "_benchmark_summary.csv"))
            observed = _read_csv(Path(str(prefix) + "_observed_price_comparison.csv"))
            if len(training_rows) != 40 or len(benchmarks) != 5 or len(observed) != 20:
                raise ValueError("smoke output cardinality mismatch")
            required_fields = {
                "unserved_demand_rate",
                "mean_price_gap_abs_error",
                "selected_action_feasibility_rate",
            }
            if not required_fields.issubset(benchmarks[0]):
                raise ValueError("smoke benchmark diagnostics are incomplete")
            self._set_status("smoke", "completed", "40-day end-to-end contract passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("preflight", "smoke", "profiles", "train", "analyze", "all"),
        default="all",
    )
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "artifacts" / EXPERIMENT_ID),
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--profile-size", type=int, default=20000)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    experiment = Experiment(
        Path(args.output_root).expanduser().resolve(),
        str(Path(args.python).expanduser().resolve()),
        args.profile_size,
        args.jobs,
        args.dry_run,
    )
    phases = (
        ("preflight", "smoke", "profiles", "train", "analyze")
        if args.phase == "all" else (args.phase,)
    )
    for phase in phases:
        print(f"[phase] {phase}", flush=True)
        getattr(experiment, phase)()


if __name__ == "__main__":
    main()
