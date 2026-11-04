#!/usr/bin/env python3
"""Run and score the RevB V17 T3 learner probes (H=5 and H=7)."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from scripts.v17_config import (  # noqa: E402
    ACTION_LEVELS,
    DEFAULT_CONFIG,
    canonical_hash,
    core_args,
    load_config,
    resolve_input,
    validate_config,
)
from v17_objective import OBJECTIVE_NAME, profit_report  # noqa: E402


PYTHON = BASE.parent / ".venv/bin/python"
CORE = BASE / "Core.py"
DEFAULT_OUTPUT = BASE / "runs/v17_t3"
SHORT_OUTPUT = BASE / "runs/v17_t3_short"
T2_REPORT = BASE / "runs/v17_t2/report.json"
T2_STATUS = BASE / "runs/v17_t2/status.json"
LEGACY_T2_ROOT = BASE / "runs/v17_t2_two_lever_v1_20261101_pre_v3"
HORIZONS = (5, 7)
TRAINING_DAYS = 200
CHECKPOINT_DAYS = (100, 150, 200)
SHORT_TRAINING_DAYS = 60
SHORT_CHECKPOINT_DAYS = (30, 45, 60)
VALIDATION_DAYS = 28
VALIDATION_OPENINGS = 3
MIN_EXTERNAL_EVALUATION_STEPS = 100
REQUIRED_T2_OPENINGS = 5
VALIDATION_SEED = 55101
STEPS_PER_DAY = 6
DRIVER_SLOT = 24
VALIDATION_RIVALS = (
    "adaptive_best_response_memory",
    "profit_best_response_myopic",
    "profit_opt_mpc_grid",
)
CASES = {
    "C0_calm": None,
    "A1_shortage_severe": BASE / "scenarios/v17_t2_shortage_severe.json",
}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temp.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_t2_dependency(
    config: Mapping[str, Any],
    status: Mapping[str, Any],
    report: Mapping[str, Any],
) -> None:
    """Reject missing, failed, or stale T2 evidence before scoring T3."""
    validate_config(config)
    expected_hash = canonical_hash(config)
    contract = str(config["action"]["contract"])
    expected_levels = tuple(float(value) for value in ACTION_LEVELS[contract])
    report_levels = tuple(float(value) for value in report.get("levels", ()))
    if status.get("state") != "passed" or status.get("pass") is not True:
        raise ValueError("T3 requires a passed T2 status")
    if status.get("config_hash") != expected_hash:
        raise ValueError("T3 T2 dependency uses a stale configuration hash")
    if report.get("config_hash") != expected_hash:
        raise ValueError("T3 T2 report uses a stale configuration hash")
    if status.get("action_contract") != contract:
        raise ValueError("T3 T2 status uses a stale action contract")
    if report.get("action_contract") != contract:
        raise ValueError("T3 T2 report uses a stale action contract")
    if report_levels != expected_levels:
        raise ValueError("T3 T2 report uses stale action levels")
    if int(report.get("openings", 0)) != REQUIRED_T2_OPENINGS:
        raise ValueError(
            f"T3 requires T2 with {REQUIRED_T2_OPENINGS} openings"
        )
    if report.get("ranking_objective") != OBJECTIVE_NAME:
        raise ValueError("T3 T2 report uses the wrong ranking objective")


def load_t2_dependency(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    status = json.loads(T2_STATUS.read_text(encoding="utf-8"))
    report = json.loads(T2_REPORT.read_text(encoding="utf-8"))
    if status.get("report_sha256") != _sha256(T2_REPORT):
        raise ValueError("T3 T2 status/report hash mismatch")
    validate_t2_dependency(config, status, report)
    return status, report


def load_accepted_legacy_t2_dependency(
    root: Path = LEGACY_T2_ROOT,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load the completed pre-widening T2 under an explicit user waiver.

    This is deliberately separate from ``load_t2_dependency`` so the normal
    current-hash/current-grid gate remains fail-closed.  The waiver is only
    exposed for the one short widened-grid T3; it never converts the legacy
    result into a current T2 pass.
    """
    status_path = root / "status.json"
    report_path = root / "report.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    legacy_contract = "two_lever_v1"
    legacy_levels = tuple(float(value) for value in ACTION_LEVELS[legacy_contract])
    if status.get("phase") != "T2" or report.get("phase") != "T2":
        raise ValueError("accepted legacy evidence is not a T2 result")
    if status.get("state") != "passed" or status.get("pass") is not True:
        raise ValueError("accepted legacy T2 did not pass")
    if (
        status.get("action_contract") != legacy_contract
        or report.get("action_contract") != legacy_contract
    ):
        raise ValueError("accepted legacy T2 uses an unexpected action contract")
    if tuple(float(value) for value in report.get("levels", ())) != legacy_levels:
        raise ValueError("accepted legacy T2 uses unexpected action levels")
    if int(report.get("openings", 0)) < 3:
        raise ValueError("accepted legacy T2 has insufficient openings")
    if report.get("ranking_objective") != OBJECTIVE_NAME:
        raise ValueError("accepted legacy T2 uses the wrong ranking objective")
    jobs = dict(report.get("jobs", {}) or {})
    if set(jobs) != set(CASES):
        raise ValueError("accepted legacy T2 case coverage is incomplete")
    expected_jobs = len(CASES) * 9
    observed_jobs = 0
    for case_jobs in jobs.values():
        if len(case_jobs) != 9:
            raise ValueError("accepted legacy T2 rival coverage is incomplete")
        for job in case_jobs.values():
            observed_jobs += 1
            final = dict(job.get("best_fixed", {}) or {}).get("final", {})
            for key in ("own_profit", "competitor_profit", "final_objective"):
                value = float(final.get(key, math.nan))
                if not math.isfinite(value):
                    raise ValueError(
                        f"accepted legacy T2 has invalid {key} evidence"
                    )
            if int(final.get("n", 0)) < 3:
                raise ValueError("accepted legacy T2 opening coverage is incomplete")
    if observed_jobs != expected_jobs:
        raise ValueError("accepted legacy T2 job coverage is incomplete")
    provenance = {
        "mode": "explicit_user_acceptance_of_completed_legacy_t2",
        "root": str(root.resolve()),
        "status_sha256": _sha256(status_path),
        "report_sha256": _sha256(report_path),
        "action_contract": legacy_contract,
        "levels": list(legacy_levels),
        "openings": int(report["openings"]),
        "known_deviation": (
            "T2 predates the two_lever_v3 widening; the short T3 itself uses "
            "the current widened grid and the current configuration hash."
        ),
    }
    return status, report, provenance


def _set_option(command: list[str], flag: str, value: str) -> None:
    if flag not in command:
        command.extend((flag, value))
        return
    command[command.index(flag) + 1] = value


def _report_prefix(path: Path) -> str:
    """Return a Core-safe repo-relative prefix, including for temp tests."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(BASE))
    except ValueError:
        suffix = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:12]
        return f"artifacts/v17_probe_external_{suffix}"


def training_command(
    config: Mapping[str, Any], *, horizon: int, output: Path,
    training_days: int = TRAINING_DAYS,
    checkpoint_days: Sequence[int] = CHECKPOINT_DAYS,
) -> list[str]:
    run_dir = output / f"h{horizon}"
    command = [
        str(PYTHON.resolve()),
        "-u",
        str(CORE),
        *core_args(
            config,
            city="nyc",
            seed=101,
            output=run_dir / "training",
            days=training_days,
            branch_horizon_days=horizon,
        ),
    ]
    _set_option(command, "--v16_audit_every", "4")
    _set_option(command, "--eval_customers", "50")
    _set_option(
        command,
        "--report_prefix",
        _report_prefix(run_dir / "training/nyc"),
    )
    command.extend(
        (
            "--v17_probe_mode",
            "--v17_probe_second_tape_fraction",
            "0.25",
            "--v17_probe_checkpoint_days",
            ",".join(str(day) for day in checkpoint_days),
            "--v17_probe_checkpoint_dir",
            str((run_dir / "checkpoints").resolve()),
            "--training_resume_state",
            str((run_dir / "resume/training_state.pt").resolve()),
            "--training_resume_every_days",
            "1",
        )
    )
    return command


def validation_command(
    config: Mapping[str, Any],
    *,
    horizon: int,
    day: int,
    case: str,
    output: Path,
    checkpoint_root: Path | None = None,
    validation_days: int = VALIDATION_DAYS,
    validation_openings: int = VALIDATION_OPENINGS,
) -> list[str]:
    validate_config(config)
    inputs = config["city_inputs"]["nyc"]
    checkpoint_base = output if checkpoint_root is None else checkpoint_root
    checkpoint = checkpoint_base / f"h{horizon}/checkpoints/day_{day}.pt"
    raw = output / f"h{horizon}/validation/day_{day}/{case}.json"
    prefix = output / f"h{horizon}/validation/day_{day}/{case}"
    command = [
        str(PYTHON.resolve()), "-u", str(CORE),
        "--market", str(inputs["market_name"]),
        "--choice_mode", "cognitive",
        "--firm1_mode", "RL",
        "--firm2_mode", VALIDATION_RIVALS[0],
        "--credit_assignment_v25", "--v26",
        "--deterministic_experiment_seed", "--deterministic_torch",
        "--learning_corrections", "--learner_revision", "4",
        "--training_curriculum", "v17_progressive",
        "--current_active_driver_observation",
        "--seed", str(VALIDATION_SEED),
        "--experiment_seed", str(VALIDATION_SEED),
        "--policy_objective", "profit_maximization",
        "--pool", "2000",
        "--threshold_profile_source", "cached",
        "--threshold_cache_path", str(resolve_input(inputs["threshold_cache"])),
        "--calibration_json", str(resolve_input(inputs["calibration_json"])),
        "--demand_calibration_artifact", str(
            resolve_input(inputs["demand_calibration"])
        ),
        "--gpt_threshold_mode", "llm_numeric",
        "--train_timesteps", "0",
        "--train_customers", "50",
        "--eval_customers", "50",
        "--train_steps_per_day", str(STEPS_PER_DAY),
        "--firm1_action_interval_steps", str(STEPS_PER_DAY),
        "--firm2_action_interval_steps", str(STEPS_PER_DAY),
        "--no-live_diagnostics",
        "--regime_market", "--regime_features", "--spillover_price_aware",
        "--supply_model", "v3", "--driver_pay_basis", "fare",
        "--loyalty_model", "v2", "--loyalty_pool_size", "180",
        "--persistence_features", "--fare_model", "realistic",
        "--shortage_willingness_share", "0.5",
        "--fleet_scale", str(float(inputs.get("fleet_scale", 1.0))),
        "--price_sensitivity_units", "relative",
        "--action_contract", str(config["action"]["contract"]),
        "--rivals_two_lever", "--v16_crn_tape",
        "--v16_branch_mode", "one_day_deviation",
        "--v16_branch_horizon_days", str(horizon),
        "--v16_branch_design", "stencil11",
        "--v16_store_cap", "20000",
        "--v16_value_head", "concave2d",
        "--v16_policy_mode", "q",
        "--trained_model_in", str(checkpoint.resolve()), "--eval_only",
        "--policy_sweep", "rl_actor",
        "--policy_sweep_out", str(raw.resolve()),
        "--policy_sweep_openings", str(int(validation_openings)),
        "--policy_sweep_rivals", ",".join(VALIDATION_RIVALS),
        "--policy_sweep_horizon", str(int(validation_days) * STEPS_PER_DAY),
        "--policy_sweep_seed", str(VALIDATION_SEED),
        "--report_prefix", _report_prefix(prefix),
    ]
    scenario = CASES[case]
    if scenario is not None:
        command.extend(("--shock_scenario", str(scenario.resolve())))
    return command


def _option_value(command: Sequence[str], flag: str) -> str | None:
    values = list(command)
    if flag not in values:
        return None
    index = values.index(flag)
    if index + 1 >= len(values):
        raise ValueError(f"{flag} has no value")
    return str(values[index + 1])


def validate_training_command_contract(command: Sequence[str]) -> None:
    """Reject expensive V17 training commands with invalid phase contracts."""
    values = list(command)
    if "--external_evaluation_only" not in values:
        raise ValueError(
            "V17 training must defer Core's in-process evaluation"
        )
    eval_steps = _option_value(values, "--eval_timesteps")
    if eval_steps is not None and int(eval_steps) < MIN_EXTERNAL_EVALUATION_STEPS:
        raise ValueError(
            "V17 cannot embed an evaluation shorter than the durability minimum"
        )
    if _option_value(values, "--training_resume_every_days") != "1":
        raise ValueError("V17 training must save exact continuation every day")
    if not _option_value(values, "--training_resume_state"):
        raise ValueError("V17 training requires an exact continuation path")


def validate_external_scoring_contract(
    *,
    checkpoint_days: Sequence[int],
    validation_days: int,
    validation_openings: int,
) -> None:
    steps = int(validation_days) * STEPS_PER_DAY
    if steps < MIN_EXTERNAL_EVALUATION_STEPS:
        raise ValueError(
            "external checkpoint evaluation is too short for durability: "
            f"{steps} < {MIN_EXTERNAL_EVALUATION_STEPS} steps"
        )
    if int(validation_openings) < 3:
        raise ValueError("external checkpoint evaluation requires >=3 openings")
    if len(tuple(checkpoint_days)) < 3:
        raise ValueError("T3 scoring requires at least three checkpoints")
    if len(CASES) < 2 or len(VALIDATION_RIVALS) < 3:
        raise ValueError("T3 scoring case or rival coverage was reduced")


def registered_score_checkpoint_days(value: str) -> tuple[int, ...]:
    """Parse a score-only schedule without creating an unregistered T3."""
    try:
        days = tuple(
            int(item.strip()) for item in value.split(",") if item.strip()
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "checkpoint days must be comma-separated integers"
        ) from exc
    if days not in (CHECKPOINT_DAYS, SHORT_CHECKPOINT_DAYS):
        raise argparse.ArgumentTypeError(
            "score-only checkpoint days must be 30,45,60 or 100,150,200"
        )
    return days


def validate_checkpoint_contract(
    path: Path,
    *,
    config: Mapping[str, Any],
    horizon: int,
    day: int,
) -> dict[str, Any]:
    """Fail before scoring if a policy does not match the requested experiment."""
    artifact = _load_artifact(path)
    if artifact.get("format") != "ride-response-platform-policy":
        raise ValueError(f"unsupported checkpoint format: {path}")
    if artifact.get("market_name") != "New York City":
        raise ValueError(f"checkpoint market mismatch: {path}")
    metadata = dict(artifact.get("metadata", {}) or {})
    if int(metadata.get("training_seed", -1)) != 101:
        raise ValueError(f"checkpoint training seed mismatch: {path}")
    if int(metadata.get("training_days", -1)) < int(day):
        raise ValueError(f"checkpoint training length mismatch: {path}")
    if int(metadata.get("comparison_checkpoint_day", -1)) != int(day):
        raise ValueError(f"checkpoint day metadata mismatch: {path}")
    if int(metadata.get("branch_horizon_days", -1)) != int(horizon):
        raise ValueError(f"checkpoint branch horizon mismatch: {path}")

    contract = str(config["action"]["contract"])
    levels = tuple(float(value) for value in ACTION_LEVELS[contract])
    expected_cells = {
        (distance, time) for distance in levels for time in levels
    }
    mapping = dict(artifact.get("action_to_two_lever_cell", {}) or {})
    actual_cells = {
        (float(cell[0]), float(cell[1])) for cell in mapping.values()
    }
    if actual_cells != expected_cells or len(mapping) != len(expected_cells):
        raise ValueError(f"checkpoint action grid mismatch: {path}")

    settings = dict(artifact.get("v26_settings", {}).get("v26r", {}) or {})
    if settings.get("action_contract") != contract:
        raise ValueError(f"checkpoint action contract mismatch: {path}")
    if settings.get("supply_model") != "v3":
        raise ValueError(f"checkpoint supply model mismatch: {path}")
    expected_settings = {
        "v16_branch_mode": "one_day_deviation",
        "v16_branch_design": "stencil11",
        "v16_value_head": "concave2d",
        "v16_policy_mode": "q",
        "v16_store_cap": 20000,
        "v16_branch_horizon_days": int(horizon),
        "shortage_willingness_share": 0.5,
        "driver_pay_basis": "fare",
        "spillover_price_aware": True,
    }
    for key, expected in expected_settings.items():
        if settings.get(key) != expected:
            raise ValueError(f"checkpoint {key} mismatch: {path}")
    if contract == "two_lever_v3":
        branch_contract = {
            "branch_profit_planner_sample_cap": int(
                config["credit"]["branch_profit_planner_sample_cap"]
            ),
            "branch_profit_planner_horizon_days": int(
                config["credit"]["branch_profit_planner_horizon_days"]
            ),
            "branch_profit_planner_candidate_radius": int(
                config["credit"]["branch_profit_planner_candidate_radius"]
            ),
        }
        for key, expected in branch_contract.items():
            if settings.get(key) != expected:
                raise ValueError(f"checkpoint {key} mismatch: {path}")
    if bool(settings.get("persistent_driver_stock")):
        raise ValueError(f"checkpoint unexpectedly persists driver stock: {path}")
    supply = dict(settings.get("effective_supply_config", {}) or {})
    if float(supply.get("eta_v3", float("nan"))) != 0.6:
        raise ValueError(f"checkpoint eta_v3 mismatch: {path}")
    if float(settings.get("effective_fleet_scale", float("nan"))) != float(
        config["city_inputs"]["nyc"]["fleet_scale"]
    ):
        raise ValueError(f"checkpoint fleet scale mismatch: {path}")
    observation = dict(
        artifact.get("mdp_config", {}).get("observation", {}) or {}
    )
    if observation.get("current_active_driver_observation") is not True:
        raise ValueError(f"checkpoint lacks exact active-driver input: {path}")
    if observation.get("opponent_features") is not False:
        raise ValueError(f"checkpoint exposes opponent private features: {path}")
    reward = dict(artifact.get("mdp_config", {}).get("active_reward", {}) or {})
    if tuple(reward.get("stage_profit_lead_weights", ())) != (
        0.0, 0.5, 0.625, 0.75
    ):
        raise ValueError(f"checkpoint reward contract mismatch: {path}")
    if any(float(value) != 0.0 for value in reward.get(
        "stage_market_share_competitiveness_weights", ()
    )):
        raise ValueError(f"checkpoint contains market-share shaping: {path}")
    if artifact.get("policy_objective") != "profit_maximization":
        raise ValueError(f"checkpoint policy objective mismatch: {path}")
    value_model = dict(artifact.get("v16_value_model_config", {}) or {})
    expected_value_model = {
        "head": "concave2d",
        "fit_steps": 50,
        "fit_batch_size": 64,
        "polyak": 0.99,
    }
    for key, expected in expected_value_model.items():
        if value_model.get(key) != expected:
            raise ValueError(f"checkpoint value-model {key} mismatch: {path}")
    q_policy = dict(artifact.get("v16_q_policy_config", {}) or {})
    if q_policy.get("mode") != "q" or q_policy.get("target_source") != (
        "polyak_value_target"
    ):
        raise ValueError(f"checkpoint Q-policy contract mismatch: {path}")
    demand = json.loads(
        resolve_input(
            config["city_inputs"]["nyc"]["demand_calibration"]
        ).read_text(encoding="utf-8")
    )
    if artifact.get("demand_calibration") != demand:
        raise ValueError(f"checkpoint demand calibration mismatch: {path}")
    population = dict(artifact.get("case_study_population", {}) or {})
    threshold_path = resolve_input(
        config["city_inputs"]["nyc"]["threshold_cache"]
    )
    if (
        population.get("threshold_profile_source") != "cached"
        or population.get("threshold_cache_sha256") != _sha256(threshold_path)
        or int(population.get("profile_pool_size", -1)) != 2000
    ):
        raise ValueError(f"checkpoint rider-population mismatch: {path}")
    return artifact


def validate_policy_sweep_product(
    path: Path,
    *,
    config: Mapping[str, Any],
    validation_days: int,
    validation_openings: int,
) -> dict[str, Any]:
    """Reject truncated, stale, or malformed external evaluation products."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_steps = int(validation_days) * STEPS_PER_DAY
    expected_rollouts = {
        f"{rival}__opening_{opening}"
        for rival in VALIDATION_RIVALS
        for opening in range(1, int(validation_openings) + 1)
    }
    if payload.get("protocol") != "v26r_policy_sweep_common_random_numbers":
        raise ValueError(f"wrong policy-sweep protocol: {path}")
    scalar_contract = {
        "city": "New York City",
        "horizon": expected_steps,
        "customers_per_step": 50,
        "steps_per_day": STEPS_PER_DAY,
        "policy_seed": VALIDATION_SEED,
    }
    for key, expected in scalar_contract.items():
        if payload.get(key) != expected:
            raise ValueError(
                f"policy-sweep {key} mismatch in {path}: "
                f"{payload.get(key)!r} != {expected!r}"
            )
    v26r = dict(payload.get("v26r", {}) or {})
    if v26r.get("action_contract") != config["action"]["contract"]:
        raise ValueError(f"policy-sweep action contract mismatch: {path}")
    if (
        v26r.get("supply_model") != "v3"
        or bool(v26r.get("persistent_driver_stock"))
        or float(v26r.get("effective_fleet_scale", float("nan")))
        != float(config["city_inputs"]["nyc"]["fleet_scale"])
    ):
        raise ValueError(f"policy-sweep supply contract mismatch: {path}")
    declared_rollouts = {
        str(row[0]) for row in payload.get("rollouts", ())
        if isinstance(row, (list, tuple)) and len(row) == 2
    }
    results = dict(payload.get("results", {}).get("rl_actor", {}) or {})
    if declared_rollouts != expected_rollouts or set(results) != expected_rollouts:
        raise ValueError(f"policy-sweep rollout coverage mismatch: {path}")
    finite_keys = (
        "own_profit", "rival_profit", "own_share", "rival_share",
        "fulfillment",
    )
    for rollout, row in results.items():
        if int(row.get("steps", -1)) != expected_steps:
            raise ValueError(
                f"policy-sweep rollout {rollout} is truncated: {path}"
            )
        if not all(
            np.isfinite(float(row.get(key, float("nan"))))
            for key in finite_keys
        ):
            raise ValueError(
                f"policy-sweep rollout {rollout} has non-finite metrics: {path}"
            )
    traces = list(payload.get("traces", {}).get("rl_actor", ()) or ())
    trace_counts = {rollout: 0 for rollout in expected_rollouts}
    for row in traces:
        rollout = str(row.get("rollout", ""))
        if rollout not in trace_counts:
            raise ValueError(f"policy-sweep trace has unknown rollout: {path}")
        if not all(
            np.isfinite(float(row.get(key, float("nan"))))
            for key in ("profit_per_request", "rival_profit_per_request")
        ):
            raise ValueError(f"policy-sweep trace has non-finite profit: {path}")
        trace_counts[rollout] += 1
    if any(count != expected_steps for count in trace_counts.values()):
        raise ValueError(f"policy-sweep trace coverage mismatch: {path}")
    return payload


def _run(
    command: Sequence[str], *, log: Path, env: Mapping[str, str]
) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    resume_state = _training_resume_path(command)
    append = bool(
        resume_state is not None and resume_state.is_file() and log.is_file()
    )
    with log.open("a" if append else "w", encoding="utf-8") as handle:
        if append:
            handle.write(
                "\n>>> V17 runner restart: resuming from exact saved "
                f"training state {resume_state}\n"
            )
            handle.flush()
        completed = subprocess.run(
            list(command), cwd=BASE, env=dict(env), stdout=handle,
            stderr=subprocess.STDOUT, text=True, check=False,
        )
    if completed.returncode:
        raise RuntimeError(
            f"command failed with exit {completed.returncode}; see {log}"
        )


def _training_resume_path(command: Sequence[str]) -> Path | None:
    try:
        index = list(command).index("--training_resume_state")
    except ValueError:
        return None
    if index + 1 >= len(command):
        raise ValueError("--training_resume_state has no path")
    return Path(command[index + 1]).expanduser().resolve()


def _run_cached_job(
    command: Sequence[str],
    *,
    product: Path,
    manifest: Path,
    log: Path,
    identity: Mapping[str, Any],
    force: bool,
) -> str:
    expected_identity = dict(identity)
    recorded: dict[str, Any] | None = None
    if manifest.is_file() and not force:
        recorded = json.loads(manifest.read_text(encoding="utf-8"))
        recorded_identity = dict(recorded.get("identity", recorded))
        if recorded_identity != expected_identity:
            raise ValueError(f"stale T3 cache identity: {product}")
        if product.is_file() and recorded.get("state") in (None, "completed"):
            expected_product_hash = recorded.get("product_sha256")
            if (
                expected_product_hash is not None
                and _sha256(product) != str(expected_product_hash)
            ):
                raise ValueError(f"T3 cached product hash changed: {product}")
            return "cached"
    if (product.exists() or manifest.exists()) and not force:
        # Training checkpoints can exist before Core finishes. Resume only
        # when the exact daily state exists and the pre-launch identity above
        # proves that command and source hashes did not change.
        resume_state = _training_resume_path(command)
        if resume_state is None or not resume_state.is_file():
            raise ValueError(f"partial T3 cache requires --force: {product}")
    _write_json(manifest, {
        "state": "running",
        "identity": expected_identity,
    })
    env = dict(os.environ)
    env.update(
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1", MPLBACKEND="Agg",
    )
    try:
        _run(command, log=log, env=env)
    except BaseException:
        _write_json(manifest, {
            "state": "failed",
            "identity": expected_identity,
        })
        raise
    if not product.is_file():
        raise RuntimeError(f"T3 command did not create {product}")
    _write_json(manifest, {
        "state": "completed",
        "identity": expected_identity,
        "product_sha256": _sha256(product),
    })
    return "completed"


def _require_training_products(
    output: Path, horizon: int,
    checkpoint_days: Sequence[int] = CHECKPOINT_DAYS,
) -> None:
    required = [
        output / f"h{horizon}/checkpoints/day_{day}.pt"
        for day in checkpoint_days
    ]
    required.append(
        output / f"h{horizon}/training/nyc_training_diagnostics.csv"
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"T3 H={horizon} training is missing required products: {missing}"
        )


def _load_artifact(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _value_model(artifact: Mapping[str, Any]):
    from v17_value_model import V17ConcaveValueModel

    config = dict(artifact["v16_value_model_config"])
    model = V17ConcaveValueModel(
        observation_dim=int(config["observation_dim"]),
        frame_stack=int(config["frame_stack"]),
        auxiliary_names=tuple(config["auxiliary_names"]),
        curvature_floor=float(config["curvature_floor"]),
    )
    model.load_state_dict(artifact["v16_value_target_model_state"])
    model.eval()
    mapping = {
        int(key): tuple(value)
        for key, value in artifact["action_to_two_lever_cell"].items()
    }
    cells = torch.tensor(
        [mapping[index] for index in range(len(mapping))], dtype=torch.float32
    )
    return model, cells, mapping, config


def _last_rows(artifact: Mapping[str, Any], days: int = 60) -> list[dict[str, Any]]:
    rows = list(artifact["v16_branch_store_state"]["rows"])
    end = max(int(row["day"]) for row in rows)
    return [row for row in rows if int(row["day"]) > end - days]


def analyze_learning(path: Path, levels: Sequence[float]) -> dict[str, Any]:
    artifact = _load_artifact(path)
    model, cells, mapping, model_config = _value_model(artifact)
    rows = _last_rows(artifact)
    audit_rows = [row for row in rows if bool(row.get("audit"))]
    errors: list[float] = []
    noise_differences: list[float] = []
    ranks: list[float] = []
    chosen: dict[str, list[tuple[float, float]]] = {"calm": [], "shortage": []}
    sensitivities: dict[str, list[float]] = {"calm": [], "shortage": []}

    for row in rows:
        mask = np.asarray(row["action_credit_mask"], dtype=bool)
        credits = np.asarray(row["action_credits"], dtype=float)
        selected = int(row["selected_action"])
        values = credits[mask]
        selected_value = float(credits[selected])
        rank = float(np.sum(values > selected_value) / max(1, values.size - 1))
        ranks.append(rank)
        events = [str(value) for value in row.get("regime_events", [])]
        group = (
            "shortage" if any("driver_shortage" in value for value in events)
            else "calm" if str(row.get("regime_label")) == "calm"
            else ""
        )
        if group:
            chosen[group].append(tuple(mapping[selected]))

    for row in audit_rows:
        observation = torch.tensor(
            row["observation_window"], dtype=torch.float32
        ).reshape(1, -1)
        with torch.no_grad():
            q = model(observation, cells)["q"][0].numpy()
        incumbent = int(row["incumbent_action"])
        predicted = q - q[incumbent]
        label_mask = np.asarray(row["action_credit_mask"], dtype=bool)
        fit_mask = np.asarray(row["action_fit_mask"], dtype=bool)
        holdout = label_mask & ~fit_mask
        credits = np.asarray(row["action_credits"], dtype=float)
        errors.extend((predicted[holdout] - credits[holdout]).tolist())
        if bool(row.get("second_tape")):
            second = np.asarray(row["second_tape_action_credits"], dtype=float)
            second_mask = np.asarray(row["second_tape_credit_mask"], dtype=bool)
            paired = holdout & second_mask & np.isfinite(second)
            noise_differences.extend((credits[paired] - second[paired]).tolist())

        events = [str(value) for value in row.get("regime_events", [])]
        group = (
            "shortage" if any("driver_shortage" in value for value in events)
            else "calm" if str(row.get("regime_label")) == "calm"
            else ""
        )
        if group:
            flat = torch.tensor(
                row["observation_window"], dtype=torch.float32,
                requires_grad=True,
            )
            slot = (
                (int(model_config["frame_stack"]) - 1)
                * int(model_config["observation_dim"])
                + DRIVER_SLOT
            )
            per_cell: list[float] = []
            output = model(flat.reshape(1, -1), cells)["q"][0]
            for index in range(output.numel()):
                gradient = torch.autograd.grad(
                    output[index], flat, retain_graph=True
                )[0]
                per_cell.append(abs(float(gradient[slot].item())))
            sensitivities[group].append(float(np.mean(per_cell)))

    error_rmse = float(np.sqrt(np.mean(np.square(errors)))) if errors else math.inf
    noise_sd = (
        float(np.std(noise_differences, ddof=1) / np.sqrt(2.0))
        if len(noise_differences) > 1 else 0.0
    )
    level_index = {float(value): index for index, value in enumerate(levels)}
    calm_mean = np.mean(chosen["calm"], axis=0) if chosen["calm"] else np.array([np.nan, np.nan])
    shortage_mean = (
        np.mean(chosen["shortage"], axis=0)
        if chosen["shortage"] else np.array([np.nan, np.nan])
    )
    def index_coordinate(value: float) -> float:
        positions = np.arange(len(levels), dtype=float)
        return float(np.interp(value, np.asarray(levels, dtype=float), positions))
    grid_steps = float(sum(
        abs(index_coordinate(float(shortage_mean[i])) - index_coordinate(float(calm_mean[i])))
        for i in range(2)
    ))
    del level_index
    return {
        "t3_1": {
            "audit_holdout_rmse": error_rmse,
            "label_noise_sd": noise_sd,
            "holdout_targets": len(errors),
            "paired_noise_targets": len(noise_differences),
            "pass": bool(errors and noise_sd > 0.0 and error_rmse <= noise_sd),
        },
        "t3_2": {
            "mean_sampled_cell_rank": float(np.mean(ranks)) if ranks else math.inf,
            "rows": len(ranks),
            "pass": bool(ranks and float(np.mean(ranks)) <= 0.35),
        },
        "t3_3": {
            "calm_mean_cell": calm_mean.tolist(),
            "shortage_mean_cell": shortage_mean.tolist(),
            "manhattan_grid_steps": grid_steps,
            "calm_days": len(chosen["calm"]),
            "shortage_days": len(chosen["shortage"]),
        },
        "t3_7": {
            "driver_slot": DRIVER_SLOT,
            "calm_mean_abs_dq_dslot": (
                float(np.mean(sensitivities["calm"]))
                if sensitivities["calm"] else None
            ),
            "shortage_mean_abs_dq_dslot": (
                float(np.mean(sensitivities["shortage"]))
                if sensitivities["shortage"] else None
            ),
            "calm_audits": len(sensitivities["calm"]),
            "shortage_audits": len(sensitivities["shortage"]),
            "gate": False,
        },
    }


def analyze_timing(
    csv_path: Path, expected_days: int = TRAINING_DAYS
) -> dict[str, Any]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    stage_seconds: dict[str, list[float]] = {}
    rss: list[float] = []
    critical = (
        "avg_reward", "avg_profit_per_request", "avg_firm2_profit_per_request",
        "v16_q_temperature", "v16_value_fit_total_loss",
        "v17_probe_simulated_day_seconds", "v17_probe_max_rss_mb",
    )
    finite = True
    for row in rows:
        stage = str(row["training_stage"])
        seconds = float(row["v17_probe_simulated_day_seconds"])
        stage_seconds.setdefault(stage, []).append(seconds)
        rss.append(float(row["v17_probe_max_rss_mb"]))
        for key in critical:
            value = float(row[key])
            if key == "v16_value_fit_total_loss" and int(row["v16_value_fit_calls_total"]) == 0:
                continue
            finite = bool(finite and np.isfinite(value))
    return {
        "seconds_per_day_by_stage": {
            stage: float(np.mean(values)) for stage, values in stage_seconds.items()
        },
        "minutes_per_day_stage_mix": float(
            np.mean([value for values in stage_seconds.values() for value in values]) / 60.0
        ),
        "max_rss_mb": float(max(rss)) if rss else math.inf,
        "training_days": len(rows),
        "no_nan": bool(finite and len(rows) == expected_days),
    }


def analyze_validation(
    output: Path, horizon: int,
    checkpoint_days: Sequence[int] = CHECKPOINT_DAYS,
    *,
    config: Mapping[str, Any] | None = None,
    validation_days: int = VALIDATION_DAYS,
    validation_openings: int = VALIDATION_OPENINGS,
) -> dict[str, Any]:
    active_config = load_config() if config is None else config
    scores: dict[int, dict[str, float]] = {}
    rollout_values: dict[int, dict[str, float]] = {}
    for day in checkpoint_days:
        values: dict[str, float] = {}
        own: list[float] = []
        rival: list[float] = []
        for case in CASES:
            payload = validate_policy_sweep_product(
                output / f"h{horizon}/validation/day_{day}/{case}.json",
                config=active_config,
                validation_days=int(validation_days),
                validation_openings=int(validation_openings),
            )
            for rollout, row in payload["results"]["rl_actor"].items():
                key = f"{case}/{rollout}"
                own_profit = float(row["own_profit"])
                rival_profit = float(row["rival_profit"])
                values[key] = profit_report(
                    own_profit, rival_profit
                )["final_objective"]
                own.append(own_profit)
                rival.append(rival_profit)
        array = np.asarray(list(values.values()), dtype=float)
        scores[day] = {
            **profit_report(float(np.mean(own)), float(np.mean(rival))),
            "se": float(np.std(array, ddof=1) / np.sqrt(array.size)),
            "rollouts": int(array.size),
        }
        rollout_values[day] = values
    comparisons = []
    passed = True
    for earlier, later in zip(checkpoint_days, checkpoint_days[1:]):
        keys = sorted(rollout_values[earlier])
        differences = np.asarray([
            rollout_values[later][key] - rollout_values[earlier][key]
            for key in keys
        ], dtype=float)
        mean = float(np.mean(differences))
        se = float(np.std(differences, ddof=1) / np.sqrt(differences.size))
        comparison_pass = bool(mean >= -2.0 * se)
        passed = bool(passed and comparison_pass)
        comparisons.append({
            "earlier_day": earlier, "later_day": later,
            "paired_change": mean, "paired_se": se,
            "lower_2se": mean + 2.0 * se, "pass": comparison_pass,
        })
    return {"scores": scores, "comparisons": comparisons, "pass": passed}


def _adaptation_found(t2: Mapping[str, Any]) -> tuple[bool, float]:
    values = [
        float(job["adaptation_value"]["final"]["switch_minus_fixed"]["final_objective"])
        for job in t2["jobs"]["A1_shortage_severe"].values()
    ]
    mean = float(np.mean(values))
    return bool(mean > 0.0 and sum(value > 0.0 for value in values) >= len(values) / 2), mean


def _performance_cores() -> int:
    try:
        result = subprocess.run(
            ("sysctl", "-n", "hw.perflevel0.physicalcpu"),
            capture_output=True, text=True, check=True,
        )
        return max(1, int(result.stdout.strip()))
    except Exception:
        return max(1, int(os.cpu_count() or 2))


def _production_days(minutes_per_day: float) -> dict[str, Any]:
    cores = _performance_cores()
    concurrent_runs = max(1, cores - 1)
    batches = int(math.ceil(9 / concurrent_runs))
    days = min(3000, int(math.floor(4 * 1440 / (batches * minutes_per_day))))
    return {
        "performance_cores": cores,
        "concurrent_runs": concurrent_runs,
        "batches": batches,
        "minutes_per_simulated_day": minutes_per_day,
        "training_days": days,
        "below_1000_contingency": bool(days < 1000),
    }


def _render_report(report: Mapping[str, Any]) -> str:
    title = "# V17 T3 short learner probe" if report.get("short_probe") else "# V17 T3 learner probe"
    lines = [title, ""]
    if report.get("short_probe"):
        lines.extend((
            "This is the single widened-grid freeze probe requested after T2.",
            "",
        ))
    lines.append(f"Selected horizon: H={report['selected_horizon']} days")
    lines.append(f"Overall: {'PASS' if report['pass'] else 'FAIL'}")
    lines.append("")
    for horizon, result in report["horizons"].items():
        lines.extend((
            f"## H={horizon}", "",
            f"- T3.1 audit RMSE {result['t3_1']['audit_holdout_rmse']:.6f}; label-noise SD {result['t3_1']['label_noise_sd']:.6f}; {'PASS' if result['t3_1']['pass'] else 'FAIL'}",
            f"- T3.2 sampled-cell rank {result['t3_2']['mean_sampled_cell_rank']:.4f}; {'PASS' if result['t3_2']['pass'] else 'FAIL'}",
            f"- T3.3 calm/shortage separation {result['t3_3']['manhattan_grid_steps']:.3f} grid steps; {'PASS' if result['t3_3']['pass'] else 'FAIL'}",
            f"- T3.4 validation stability: {'PASS' if result['t3_4']['pass'] else 'FAIL'} (own and competitor profit reported in report.json)",
            f"- T3.6 {result['t3_6']['minutes_per_day_stage_mix']:.4f} minutes/simulated day; max RSS {result['t3_6']['max_rss_mb']:.1f} MB; no-NaN {'PASS' if result['t3_6']['no_nan'] else 'FAIL'}",
            f"- T3.7 |dQ/d own-driver slot| calm={result['t3_7']['calm_mean_abs_dq_dslot']} shortage={result['t3_7']['shortage_mean_abs_dq_dslot']} (recorded, not a gate)",
            "",
        ))
    plan = report["production_length"]
    lines.append(
        f"T3.6 production length: D={plan['training_days']} with {plan['concurrent_runs']} concurrent run(s) in {plan['batches']} batch(es)."
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Score existing immutable checkpoints without starting training.",
    )
    parser.add_argument(
        "--horizon", type=int, choices=HORIZONS,
        help="Limit --score-only to one saved branch horizon.",
    )
    parser.add_argument(
        "--checkpoint-source", type=Path,
        help="Root containing hN/checkpoints for --score-only.",
    )
    parser.add_argument(
        "--checkpoint-days",
        type=registered_score_checkpoint_days,
        help=(
            "Registered checkpoint schedule for --score-only: 30,45,60 or "
            "100,150,200 (default)."
        ),
    )
    parser.add_argument(
        "--action-contract", choices=tuple(ACTION_LEVELS),
        help="Historical action contract override for --score-only.",
    )
    parser.add_argument(
        "--evaluation-days", type=int, default=VALIDATION_DAYS,
        help="Days per external policy-sweep rollout (default: 28).",
    )
    parser.add_argument(
        "--validation-openings", type=int, default=VALIDATION_OPENINGS,
        help="Independent openings per rival and case (default: 3).",
    )
    parser.add_argument(
        "--short", action="store_true",
        help="Run a 60-day diagnostic T3 with checkpoints at days 30, 45, and 60.",
    )
    parser.add_argument(
        "--accept-completed-legacy-t2", action="store_true",
        help=(
            "Explicitly accept the completed archived two_lever_v1 T2 only "
            "for the one short widened-grid T3."
        ),
    )
    args = parser.parse_args()
    if args.short and args.score_only:
        parser.error("--short and --score-only are mutually exclusive")
    if args.horizon is not None and not args.score_only:
        parser.error("--horizon is currently restricted to --score-only")
    if args.checkpoint_source is not None and not args.score_only:
        parser.error("--checkpoint-source requires --score-only")
    if args.checkpoint_days is not None and not args.score_only:
        parser.error("--checkpoint-days requires --score-only")
    if args.action_contract is not None and not args.score_only:
        parser.error("--action-contract override requires --score-only")
    if args.accept_completed_legacy_t2 and not args.short:
        parser.error("--accept-completed-legacy-t2 requires --short")
    config = load_config(args.config)
    if args.action_contract is not None:
        config = copy.deepcopy(config)
        levels = ACTION_LEVELS[str(args.action_contract)]
        config["action"].update(
            contract=str(args.action_contract),
            distance_levels=list(levels),
            time_levels=list(levels),
        )
    validate_config(config)
    checkpoint_days = (
        tuple(args.checkpoint_days)
        if args.score_only and args.checkpoint_days is not None
        else SHORT_CHECKPOINT_DAYS if args.short else CHECKPOINT_DAYS
    )
    training_days = (
        max(checkpoint_days)
        if args.score_only
        else SHORT_TRAINING_DAYS if args.short else TRAINING_DAYS
    )
    output = (args.output or (SHORT_OUTPUT if args.short else DEFAULT_OUTPUT)).resolve()
    checkpoint_source = (
        args.checkpoint_source or output
    ).resolve()
    active_horizons = (
        (int(args.horizon),) if args.horizon is not None else HORIZONS
    )
    validate_external_scoring_contract(
        checkpoint_days=checkpoint_days,
        validation_days=int(args.evaluation_days),
        validation_openings=int(args.validation_openings),
    )
    config_hash = canonical_hash(config)
    sources = {
        str(path.relative_to(BASE)): _sha256(path)
        for path in (
            CORE, BASE / "scripts/v17_config.py", Path(__file__).resolve(),
            BASE / "v17_value_model.py", BASE / "v17_branch_store.py",
        )
    }
    plan = {
        "phase": (
            "T3-score-only" if args.score_only
            else "T3-short" if args.short
            else "T3"
        ),
        "state": "planned", "pass": False,
        "short_probe": bool(args.short),
        "eligible_to_freeze_config": not args.score_only,
        "ranking_objective": OBJECTIVE_NAME,
        "score_only": bool(args.score_only),
        "horizons": list(active_horizons), "training_days": training_days,
        "checkpoint_days": list(checkpoint_days),
        "validation_cases": list(CASES),
        "validation_rivals": list(VALIDATION_RIVALS),
        "evaluation_days": int(args.evaluation_days),
        "evaluation_steps": int(args.evaluation_days) * STEPS_PER_DAY,
        "validation_openings": int(args.validation_openings),
        "validation_rollouts": len(active_horizons) * len(checkpoint_days) * len(CASES) * len(VALIDATION_RIVALS) * int(args.validation_openings),
        "checkpoint_source": str(checkpoint_source),
        "config_hash": config_hash,
        "t2_dependency_mode": (
            "accepted_completed_legacy"
            if args.accept_completed_legacy_t2
            else "strict_current"
        ),
    }
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return 0
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "status.json", {**plan, "state": "running"})
    try:
        if args.score_only:
            checkpoint_contracts: dict[str, Any] = {}
            for horizon in active_horizons:
                for day in checkpoint_days:
                    checkpoint = (
                        checkpoint_source
                        / f"h{horizon}/checkpoints/day_{day}.pt"
                    )
                    if not checkpoint.is_file():
                        raise FileNotFoundError(checkpoint)
                    validate_checkpoint_contract(
                        checkpoint,
                        config=config,
                        horizon=horizon,
                        day=day,
                    )
                    checkpoint_contracts[f"h{horizon}/day_{day}"] = {
                        "path": str(checkpoint),
                        "sha256": _sha256(checkpoint),
                    }

            validation_jobs = []
            for horizon in active_horizons:
                for day in checkpoint_days:
                    for case in CASES:
                        command = validation_command(
                            config,
                            horizon=horizon,
                            day=day,
                            case=case,
                            output=output,
                            checkpoint_root=checkpoint_source,
                            validation_days=int(args.evaluation_days),
                            validation_openings=int(args.validation_openings),
                        )
                        product = (
                            output
                            / f"h{horizon}/validation/day_{day}/{case}.json"
                        )
                        identity = {
                            "command": command,
                            "config_hash": config_hash,
                            "sources": sources,
                            "checkpoint_sha256": checkpoint_contracts[
                                f"h{horizon}/day_{day}"
                            ]["sha256"],
                        }
                        validation_jobs.append(
                            (horizon, day, case, command, product, identity)
                        )
            with ThreadPoolExecutor(
                max_workers=max(1, int(args.workers))
            ) as executor:
                futures = {
                    executor.submit(
                        _run_cached_job,
                        command,
                        product=product,
                        manifest=product.with_suffix(".manifest.json"),
                        log=product.with_suffix(".log"),
                        identity=identity,
                        force=args.force,
                    ): (horizon, day, case)
                    for horizon, day, case, command, product, identity
                    in validation_jobs
                }
                completed_validation = []
                for future in as_completed(futures):
                    future.result()
                    completed_validation.append(futures[future])
                    _write_json(output / "status.json", {
                        **plan,
                        "state": "running",
                        "completed_validation_jobs": len(
                            completed_validation
                        ),
                        "validation_jobs": len(validation_jobs),
                    })

            scored: dict[str, Any] = {}
            levels = ACTION_LEVELS[str(config["action"]["contract"])]
            scientific_pass = True
            for horizon in active_horizons:
                result = analyze_learning(
                    checkpoint_source
                    / f"h{horizon}/checkpoints/day_{training_days}.pt",
                    levels,
                )
                result["t3_4"] = analyze_validation(
                    output,
                    horizon,
                    checkpoint_days,
                    config=config,
                    validation_days=int(args.evaluation_days),
                    validation_openings=int(args.validation_openings),
                )
                result["scientific_pass"] = bool(
                    result["t3_1"]["pass"]
                    and result["t3_2"]["pass"]
                    and result["t3_4"]["pass"]
                )
                scientific_pass = bool(
                    scientific_pass and result["scientific_pass"]
                )
                scored[str(horizon)] = result
            report = {
                **plan,
                "state": "completed",
                "execution_pass": True,
                "pass": scientific_pass,
                "eligible_to_freeze_config": False,
                "checkpoint_contracts": checkpoint_contracts,
                "horizons": scored,
            }
            _write_json(output / "report.json", report)
            lines = [
                "# V17 historical checkpoint score",
                "",
                "Training was not restarted. Checkpoints were evaluated by "
                f"{int(args.evaluation_days)}-day external policy sweeps "
                f"({int(args.evaluation_days) * STEPS_PER_DAY} steps each).",
                "",
            ]
            for horizon, result in scored.items():
                lines.extend((
                    f"## H={horizon}",
                    "",
                    f"- T3.1: {'PASS' if result['t3_1']['pass'] else 'FAIL'}",
                    f"- T3.2: {'PASS' if result['t3_2']['pass'] else 'FAIL'}",
                    f"- T3.4 checkpoint stability: {'PASS' if result['t3_4']['pass'] else 'FAIL'}",
                    f"- Scientific result: {'PASS' if result['scientific_pass'] else 'FAIL'}",
                    "",
                ))
            (output / "report.md").write_text(
                "\n".join(lines), encoding="utf-8"
            )
            _write_json(output / "status.json", {
                **plan,
                "state": "completed",
                "execution_pass": True,
                "pass": scientific_pass,
                "checkpoint_contracts": checkpoint_contracts,
                "report": str((output / "report.json").resolve()),
                "report_sha256": _sha256(output / "report.json"),
            })
            return 0

        # T3's adaptation gate depends on T2. Validate that dependency before
        # launching either expensive training horizon.
        if args.accept_completed_legacy_t2:
            _, t2, t2_provenance = load_accepted_legacy_t2_dependency()
            plan["t2_dependency"] = t2_provenance
            _write_json(output / "status.json", {**plan, "state": "running"})
        else:
            _, t2 = load_t2_dependency(config)
        training_jobs = []
        for horizon in active_horizons:
            command = training_command(
                config, horizon=horizon, output=output,
                training_days=training_days,
                checkpoint_days=checkpoint_days,
            )
            validate_training_command_contract(command)
            product = output / f"h{horizon}/checkpoints/day_{training_days}.pt"
            identity = {
                "command": command, "config_hash": config_hash,
                "sources": sources,
            }
            training_jobs.append((horizon, command, product, identity))
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(
                    _run_cached_job, command, product=product,
                    manifest=output / f"h{horizon}/training_manifest.json",
                    log=output / f"h{horizon}/training.log",
                    identity=identity, force=args.force,
                ): horizon
                for horizon, command, product, identity in training_jobs
            }
            completed_training = []
            for future in as_completed(futures):
                horizon = futures[future]
                future.result()
                _require_training_products(output, horizon, checkpoint_days)
                completed_training.append(horizon)
                _write_json(output / "status.json", {
                    **plan, "state": "running",
                    "completed_training_horizons": sorted(completed_training),
                })

        validation_jobs = []
        for horizon in active_horizons:
            for day in checkpoint_days:
                for case in CASES:
                    command = validation_command(
                        config,
                        horizon=horizon,
                        day=day,
                        case=case,
                        output=output,
                        validation_days=int(args.evaluation_days),
                        validation_openings=int(args.validation_openings),
                    )
                    product = output / f"h{horizon}/validation/day_{day}/{case}.json"
                    identity = {
                        "command": command, "config_hash": config_hash,
                        "sources": sources,
                        "checkpoint_sha256": _sha256(
                            output / f"h{horizon}/checkpoints/day_{day}.pt"
                        ),
                    }
                    validation_jobs.append((horizon, day, case, command, product, identity))
        with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
            futures = {
                executor.submit(
                    _run_cached_job, command, product=product,
                    manifest=product.with_suffix(".manifest.json"),
                    log=product.with_suffix(".log"), identity=identity,
                    force=args.force,
                ): (horizon, day, case)
                for horizon, day, case, command, product, identity in validation_jobs
            }
            completed_validation = []
            for future in as_completed(futures):
                key = futures[future]
                future.result()
                completed_validation.append(key)
                _write_json(output / "status.json", {
                    **plan, "state": "running",
                    "completed_training_horizons": list(active_horizons),
                    "completed_validation_jobs": len(completed_validation),
                    "validation_jobs": len(validation_jobs),
                })

        adaptation_found, adaptation_mean = _adaptation_found(t2)
        levels = ACTION_LEVELS[str(config["action"]["contract"])]
        horizons: dict[str, Any] = {}
        for horizon in active_horizons:
            result = analyze_learning(
                output / f"h{horizon}/checkpoints/day_{training_days}.pt", levels
            )
            result["t3_3"].update(
                adaptation_found_in_t2=adaptation_found,
                t2_mean_adaptation_value=adaptation_mean,
                pass_=bool(
                    result["t3_3"]["manhattan_grid_steps"] >= 1.0
                    if adaptation_found else False
                ),
            )
            result["t3_3"]["pass"] = result["t3_3"].pop("pass_")
            result["t3_4"] = analyze_validation(
                output,
                horizon,
                checkpoint_days,
                config=config,
                validation_days=int(args.evaluation_days),
                validation_openings=int(args.validation_openings),
            )
            result["t3_6"] = analyze_timing(
                output / f"h{horizon}/training/nyc_training_diagnostics.csv",
                expected_days=training_days,
            )
            result["gate_pass"] = bool(
                result["t3_1"]["pass"]
                and result["t3_2"]["pass"]
                and result["t3_3"]["pass"]
                and result["t3_4"]["pass"]
                and result["t3_6"]["no_nan"]
            )
            horizons[str(horizon)] = result

        def selection_key(horizon: int) -> tuple[float, ...]:
            result = horizons[str(horizon)]
            passes = sum(bool(result[f"t3_{index}"]["pass"]) for index in (1, 2, 3))
            noise = max(result["t3_1"]["label_noise_sd"], 1e-12)
            return (
                float(passes),
                -float(result["t3_1"]["audit_holdout_rmse"] / noise),
                -float(result["t3_2"]["mean_sampled_cell_rank"]),
                float(result["t3_3"]["manhattan_grid_steps"]),
                -float(horizon),
            )
        selected = max(active_horizons, key=selection_key)
        selected_result = horizons[str(selected)]
        production = _production_days(
            float(selected_result["t3_6"]["minutes_per_day_stage_mix"])
        )
        report = {
            **plan, "state": "passed" if selected_result["gate_pass"] else "failed",
            "pass": bool(selected_result["gate_pass"]),
            "horizons": horizons,
            "selected_horizon": selected,
            "selection_rule": "most T3.1-T3.3 passes, then lower error/noise, lower rank, larger regime separation, shorter H",
            "production_length": production,
        }
        _write_json(output / "report.json", report)
        (output / "report.md").write_text(_render_report(report), encoding="utf-8")
        _write_json(output / "status.json", {
            **plan,
            "state": report["state"], "pass": report["pass"],
            "eligible_to_freeze_config": True,
            "selected_horizon": selected,
            "probe_training_days": training_days,
            "recommended_production_days": production["training_days"],
            "report": str((output / "report.json").resolve()),
            "report_sha256": _sha256(output / "report.json"),
        })
        return 0 if report["pass"] else 2
    except Exception as exc:
        _write_json(output / "status.json", {
            **plan, "state": "failed", "pass": False,
            "error": f"{type(exc).__name__}: {exc}",
        })
        raise


if __name__ == "__main__":
    raise SystemExit(main())
