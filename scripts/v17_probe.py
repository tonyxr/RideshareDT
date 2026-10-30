#!/usr/bin/env python3
"""Run and score the RevB V17 T3 learner probes (H=5 and H=7)."""

from __future__ import annotations

import argparse
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
HORIZONS = (5, 7)
TRAINING_DAYS = 200
CHECKPOINT_DAYS = (100, 150, 200)
SHORT_TRAINING_DAYS = 60
SHORT_CHECKPOINT_DAYS = (30, 45, 60)
VALIDATION_DAYS = 28
VALIDATION_OPENINGS = 3
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
            "--checkpoint_validation_interval_days",
            "100000",
            "--checkpoint_validation_horizon",
            "2",
            "--checkpoint_validation_customers",
            "50",
            "--checkpoint_validation_openings",
            "1",
            "--eval_timesteps",
            "2",
            "--eval_customers",
            "50",
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
) -> list[str]:
    validate_config(config)
    inputs = config["city_inputs"]["nyc"]
    checkpoint = output / f"h{horizon}/checkpoints/day_{day}.pt"
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
        "--policy_sweep_openings", str(VALIDATION_OPENINGS),
        "--policy_sweep_rivals", ",".join(VALIDATION_RIVALS),
        "--policy_sweep_horizon", str(VALIDATION_DAYS * STEPS_PER_DAY),
        "--policy_sweep_seed", str(VALIDATION_SEED),
        "--report_prefix", _report_prefix(prefix),
    ]
    scenario = CASES[case]
    if scenario is not None:
        command.extend(("--shock_scenario", str(scenario.resolve())))
    return command


def _run(
    command: Sequence[str], *, log: Path, env: Mapping[str, str]
) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            list(command), cwd=BASE, env=dict(env), stdout=handle,
            stderr=subprocess.STDOUT, text=True, check=False,
        )
    if completed.returncode:
        raise RuntimeError(
            f"command failed with exit {completed.returncode}; see {log}"
        )


def _run_cached_job(
    command: Sequence[str],
    *,
    product: Path,
    manifest: Path,
    log: Path,
    identity: Mapping[str, Any],
    force: bool,
) -> str:
    if product.is_file() and manifest.is_file() and not force:
        if json.loads(manifest.read_text(encoding="utf-8")) != dict(identity):
            raise ValueError(f"stale T3 cache identity: {product}")
        return "cached"
    if (product.exists() or manifest.exists()) and not force:
        raise ValueError(f"partial T3 cache requires --force: {product}")
    env = dict(os.environ)
    env.update(
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1", MPLBACKEND="Agg",
    )
    _run(command, log=log, env=env)
    if not product.is_file():
        raise RuntimeError(f"T3 command did not create {product}")
    _write_json(manifest, identity)
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
) -> dict[str, Any]:
    scores: dict[int, dict[str, float]] = {}
    rollout_values: dict[int, dict[str, float]] = {}
    for day in checkpoint_days:
        values: dict[str, float] = {}
        own: list[float] = []
        rival: list[float] = []
        for case in CASES:
            payload = json.loads(
                (output / f"h{horizon}/validation/day_{day}/{case}.json")
                .read_text(encoding="utf-8")
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
            "This diagnostic run is not eligible to freeze the production configuration.",
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
        "--short", action="store_true",
        help="Run a 60-day diagnostic T3 with checkpoints at days 30, 45, and 60.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    validate_config(config)
    training_days = SHORT_TRAINING_DAYS if args.short else TRAINING_DAYS
    checkpoint_days = SHORT_CHECKPOINT_DAYS if args.short else CHECKPOINT_DAYS
    output = (args.output or (SHORT_OUTPUT if args.short else DEFAULT_OUTPUT)).resolve()
    config_hash = canonical_hash(config)
    sources = {
        str(path.relative_to(BASE)): _sha256(path)
        for path in (
            CORE, BASE / "scripts/v17_config.py", Path(__file__).resolve(),
            BASE / "v17_value_model.py", BASE / "v17_branch_store.py",
        )
    }
    plan = {
        "phase": "T3-short" if args.short else "T3",
        "state": "planned", "pass": False,
        "short_probe": bool(args.short),
        "eligible_to_freeze_config": not args.short,
        "ranking_objective": OBJECTIVE_NAME,
        "horizons": list(HORIZONS), "training_days": training_days,
        "checkpoint_days": list(checkpoint_days),
        "validation_cases": list(CASES),
        "validation_rivals": list(VALIDATION_RIVALS),
        "validation_rollouts": len(HORIZONS) * len(checkpoint_days) * len(CASES) * len(VALIDATION_RIVALS) * VALIDATION_OPENINGS,
        "config_hash": config_hash,
    }
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return 0
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "status.json", {**plan, "state": "running"})
    try:
        training_jobs = []
        for horizon in HORIZONS:
            command = training_command(
                config, horizon=horizon, output=output,
                training_days=training_days,
                checkpoint_days=checkpoint_days,
            )
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
        for horizon in HORIZONS:
            for day in checkpoint_days:
                for case in CASES:
                    command = validation_command(
                        config, horizon=horizon, day=day, case=case, output=output
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
                    "completed_training_horizons": list(HORIZONS),
                    "completed_validation_jobs": len(completed_validation),
                    "validation_jobs": len(validation_jobs),
                })

        t2_status = json.loads(T2_STATUS.read_text(encoding="utf-8"))
        t2 = json.loads(T2_REPORT.read_text(encoding="utf-8"))
        validate_t2_dependency(config, t2_status, t2)
        adaptation_found, adaptation_mean = _adaptation_found(t2)
        levels = ACTION_LEVELS[str(config["action"]["contract"])]
        horizons: dict[str, Any] = {}
        for horizon in HORIZONS:
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
                output, horizon, checkpoint_days
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
        selected = max(HORIZONS, key=selection_key)
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
            "phase": plan["phase"], "state": report["state"], "pass": report["pass"],
            "short_probe": bool(args.short),
            "eligible_to_freeze_config": not args.short,
            "selected_horizon": selected,
            "probe_training_days": training_days,
            "recommended_production_days": production["training_days"],
            "report": str((output / "report.json").resolve()),
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
