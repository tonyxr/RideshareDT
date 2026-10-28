#!/usr/bin/env python3
"""Full-budget Revision 15c training, benchmark evaluation and reality-gap analysis.

Each invocation with --city owns one complete, restart-free city pipeline.  The
launcher intentionally starts three separate jobs so a failure in one city does
not interrupt or rewrite evidence from another city.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/rideshare-matplotlib-cache")

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent
PYTHON = ROOT / ".venv/bin/python"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(HERE))

from diagnostic_logging import atomic_json  # noqa: E402
import run_revision14_verify as r14  # noqa: E402
from run_v26r_staged import environment, source_hashes  # noqa: E402
from summarize_revision15b_benchmarks import RIVAL_GROUPS, RIVALS  # noqa: E402
from v26r_cases import CASES  # noqa: E402
import v26r_figures  # noqa: E402


CITIES = {
    "nyc": {"market": "New York City", "ride_root": "data/raw/rides/nyc_kaggle"},
    "boston": {"market": "Boston", "ride_root": "data/raw/rides/boston_kaggle"},
    "chicago": {"market": "Chicago", "ride_root": "data/raw/rides/chicago_kaggle"},
}
CITY_CALIBRATION_ROOT = BASE / "calibration_artifacts"
CATCHUP_MODES = (
    "adaptive_best_response_memory",
    "adaptive_best_response_aggressive",
    "price_gap_catchup",
    "price_gap_catchup_fast",
)
PROFIT_MODES = (
    "profit_opt_pi_price_gap",
    "profit_opt_mpc_grid",
    "profit_opt_region_supply_demand",
    "profit_opt_surge_driver_incentive",
    "profit_opt_queue_service_threshold",
    "profit_best_response_myopic",
    "profit_best_response_forward",
    "profit_best_response_anticipating",
)
STAGE_KEYS = ("market", "static", "catchup", "profit")
STAGE_FRACTIONS = {"market": 0.05, "static": 0.05, "catchup": 0.375, "profit": 0.525}
MAGNITUDE_TARGETS = tuple(round(value / 10.0, 1) for value in range(6, 21))
STAGE_PROFIT_LEAD_WEIGHTS = (0.0, 0.35, 0.55, 0.65)
STAGE_MARKET_SHARE_WEIGHTS = (0.0, 0.05, 0.10, 0.25)


def manipulated_coefficient_count(slug: str) -> int:
    """Return the largest identifiable prefix of the public tariff contract.

    Core exposes a documented prefix of
    ``base_fare, per_minute, per_mile, booking_fee, airport_fee``.  Boston's
    quote data do not identify an airport surcharge and its calibrated airport
    anchor is therefore exactly zero.  Exposing airport multipliers in that
    market creates fifteen guaranteed no-op actions (zero times any multiplier
    remains zero), which wastes exploration and corrupts actor credit.  Read
    the calibration rather than special-casing the city name so a future data
    refresh can restore the fifth lever automatically.
    """
    path = ROOT / "data" / "processed" / slug / "calibration.json"
    with path.open() as handle:
        payload = json.load(handle)
    calibration = dict(payload.get("calibration", {}) or {})
    tariff = dict(calibration.get("tariff", {}) or {})
    trip_bank = dict(calibration.get("trip_bank", {}) or {})
    prefix = ("base_fare", "per_minute", "per_mile", "booking_fee")
    missing = [
        key for key in prefix
        if not math.isfinite(float(tariff.get(key, float("nan"))))
        or float(tariff.get(key, 0.0)) <= 0.0
    ]
    if missing:
        raise ValueError(
            f"{slug}: nonpositive or missing required tariff anchors: {missing}"
        )
    airport_anchor = float(tariff.get("airport_fee", 0.0))
    airport_identified = bool(trip_bank.get("airport_identified", True))
    return 5 if airport_identified and math.isfinite(airport_anchor) and airport_anchor > 0.0 else 4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def option(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def remove_option(command: list[str], flag: str) -> None:
    while flag in command:
        index = command.index(flag)
        del command[index:index + 2]


def add_flag(command: list[str], flag: str) -> None:
    if flag not in command:
        command.append(flag)


def resolve_stage_fractions(days: int, stage_days: str = "") -> dict[str, float]:
    """Resolve an optional exact four-stage day allocation.

    Fractions are passed to Core, but accepting days at the launcher boundary
    keeps long-run contracts auditable and avoids rounding ambiguity.
    """
    if not stage_days:
        return dict(STAGE_FRACTIONS)
    try:
        values = tuple(int(value.strip()) for value in stage_days.split(","))
    except ValueError as exc:
        raise ValueError("--stage-days must contain four comma-separated integers") from exc
    if len(values) != 4 or any(value <= 0 for value in values):
        raise ValueError("--stage-days must contain four positive integers")
    if sum(values) != int(days):
        raise ValueError(f"--stage-days sum to {sum(values)}, expected --days={days}")
    return {key: value / float(days) for key, value in zip(STAGE_KEYS, values)}


def build_training_command(
    slug: str,
    out: Path,
    days: int,
    seed: int,
    stage_fractions: dict[str, float] | None = None,
) -> list[str]:
    settings = CITIES[slug]
    stage_fractions = dict(stage_fractions or STAGE_FRACTIONS)
    command = r14.r14_train_command(
        out,
        days,
        oracle=False,
        smoke=False,
        nmcc_branches=5,
        nmcc_actor_weight=0.8,
        nmcc_policy_coeff=0.0,
        momentum=True,
        opponent=True,
        seed=seed,
    )
    for flag in ("--level_step_actions", "--whole_tariff_actions"):
        while flag in command:
            command.remove(flag)
    for flag in (
        "--tariff_structure",
        "--tariff_level_bounds",
        "--stage_entropy_end_scale",
        "--stage_lr_end_scale",
    ):
        remove_option(command, flag)
    for flag, value in (
        ("--market", settings["market"]),
        ("--threshold_cache_path", ROOT / f"artifacts/v26_confirmatory/thresholds/{slug}/llm_numeric_primary.jsonl"),
        ("--calibration_json", ROOT / f"data/processed/{slug}/calibration.json"),
        ("--city_calibration_artifact", CITY_CALIBRATION_ROOT / slug / "city_calibration.json"),
        ("--training_stage_market_fraction", stage_fractions["market"]),
        ("--training_stage_static_fraction", stage_fractions["static"]),
        ("--training_stage_dynamic_fraction", stage_fractions["catchup"]),
        ("--training_stage_profit_fraction", stage_fractions["profit"]),
        ("--training_catchup_opponent_modes", ",".join(CATCHUP_MODES)),
        ("--training_profit_opponent_modes", ",".join(PROFIT_MODES)),
        ("--report_prefix", os.path.relpath(out / slug, BASE)),
        ("--trained_model_out", out / "policy.pt"),
        ("--trained_model_id", f"r15c-{slug}-public-{days}-s{seed}-full-benchmark"),
        ("--model_registry_dir", out / "registry"),
        ("--rl_warm_start_target", "0"),
        ("--tariff_targets", ",".join(str(value) for value in MAGNITUDE_TARGETS)),
        ("--manipulated_price_coefficients", manipulated_coefficient_count(slug)),
        ("--ppo_entropy_start", "0.01"),
        ("--ppo_min_rollout_transitions", "30"),
        ("--branch_imitation_coeff", "0.0"),
        ("--stage_profit_lead_weights", ",".join(str(value) for value in STAGE_PROFIT_LEAD_WEIGHTS)),
        ("--stage_market_share_competitiveness_weights", ",".join(str(value) for value in STAGE_MARKET_SHARE_WEIGHTS)),
    ):
        r14.set_option(command, flag, str(value))

    # The standard five-policy report remains available beside the wider held-out
    # sweep.  Kaggle comparison is performed by Core only after the final policy
    # has been saved, so actual price is never an actor input.
    add_flag(command, "--benchmark_suite")
    add_flag(command, "--compare_with_dataset")
    for flag, value in (
        ("--dataset_root", ROOT / settings["ride_root"]),
        ("--dataset_city", settings["market"]),
        ("--weather_root", ROOT / "data/raw/weather/noaa"),
        ("--comparison_out", out / f"{slug}_observed_price_comparison.csv"),
        ("--comparison_plot_prefix", os.path.relpath(out / f"{slug}_reality_gap", BASE)),
        ("--comparison_limit", "5000"),
        ("--dataset_preview_rows", "0"),
        ("--comparison_policy_mode", "argmax_raw"),
        ("--comparison_policy_seed", "99173"),
        ("--comparison_dataset_seed", "104729"),
    ):
        r14.set_option(command, flag, str(value))
    assert_city_command(command, slug, out, days, stage_fractions)
    return command


def assert_city_command(
    command: list[str],
    slug: str,
    out: Path,
    days: int,
    stage_fractions: dict[str, float] | None = None,
) -> None:
    stage_fractions = dict(stage_fractions or STAGE_FRACTIONS)
    if "--learning_corrections" not in command:
        raise ValueError(f"{slug}: Revision 15c requires learning corrections")
    expected = {
        "--market": CITIES[slug]["market"],
        "--train_timesteps": str(days),
        "--training_stage_market_fraction": str(stage_fractions["market"]),
        "--training_stage_static_fraction": str(stage_fractions["static"]),
        "--training_stage_dynamic_fraction": str(stage_fractions["catchup"]),
        "--training_stage_profit_fraction": str(stage_fractions["profit"]),
        "--training_catchup_opponent_modes": ",".join(CATCHUP_MODES),
        "--training_profit_opponent_modes": ",".join(PROFIT_MODES),
        "--rl_warm_start_target": "0",
        "--nmcc_branches": "5",
        "--branch_imitation_coeff": "0.0",
        "--ppo_min_rollout_transitions": "30",
        "--tariff_targets": ",".join(str(value) for value in MAGNITUDE_TARGETS),
        "--manipulated_price_coefficients": str(manipulated_coefficient_count(slug)),
        "--stage_profit_lead_weights": ",".join(str(value) for value in STAGE_PROFIT_LEAD_WEIGHTS),
        "--stage_market_share_competitiveness_weights": ",".join(str(value) for value in STAGE_MARKET_SHARE_WEIGHTS),
    }
    for flag, value in expected.items():
        if option(command, flag) != value:
            raise ValueError(f"{slug}: {flag}={option(command, flag)!r}, expected {value!r}")
    for flag in ("--benchmark_suite", "--compare_with_dataset"):
        if flag not in command:
            raise ValueError(f"{slug}: missing {flag}")
    for flag in (
        "--level_step_actions",
        "--whole_tariff_actions",
        "--tariff_structure",
        "--actor_true_shortage_flag",
        "--stage_entropy_end_scale",
        "--stage_lr_end_scale",
    ):
        if flag in command:
            raise ValueError(f"{slug}: forbidden R15c option {flag}")
    required = (
        Path(option(command, "--threshold_cache_path")),
        Path(option(command, "--calibration_json")),
        Path(option(command, "--city_calibration_artifact")),
        Path(option(command, "--dataset_root")),
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{slug}: missing inputs: {missing}")
    if Path(option(command, "--trained_model_out")) != out / "policy.pt":
        raise ValueError(f"{slug}: trained model output escaped city run")


def evaluation_command(slug: str, out: Path, policy: Path) -> list[str]:
    return [
        str(PYTHON), "-u", str(HERE / "v26r_case_eval.py"),
        CITIES[slug]["market"], slug, "all",
        "--seed-set", "eval",
        "--openings", "10",
        "--policy", str(policy.resolve()),
        "--output-root", str((out / "full_evaluation").resolve()),
        "--customers", "50",
        "--pool", "2000",
        "--rivals", ",".join(RIVALS),
        "--horizon-days", "40",
        "--channels", "full",
        "--chunk", "1",
        "--budget-seconds", "1000000000",
        "--specs", "rl_actor",
    ]


def summarizer_command(slug: str, out: Path) -> list[str]:
    return [
        str(PYTHON), "-u", str(HERE / "summarize_revision15b_benchmarks.py"),
        "--evaluation-root", str((out / "full_evaluation").resolve()),
        "--slug", slug,
        "--city", CITIES[slug]["market"],
        "--output", str((out / "full_evaluation_summary").resolve()),
    ]


def update_status(out: Path, **values: Any) -> None:
    path = out / "status.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data.update(values, updated_at=time.time())
    atomic_json(path, data)


def run_process(command: list[str], out: Path, phase: str, log_name: str) -> None:
    atomic_json(out / f"{phase}_command.json", {"argv": command, "cwd": str(BASE)})
    log_path = out / log_name
    child_env = environment()
    child_env["MPLCONFIGDIR"] = os.environ["MPLCONFIGDIR"]
    with log_path.open("w") as log:
        child = subprocess.Popen(
            command,
            cwd=BASE,
            env=child_env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        update_status(out, phase=phase, state="running", child_pid=child.pid, log=str(log_path))
        while True:
            try:
                code = child.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                update_status(
                    out,
                    phase=phase,
                    state="running",
                    child_pid=child.pid,
                    log_bytes=log_path.stat().st_size,
                )
    atomic_json(out / f"{phase}_exit.json", {"returncode": code, "completed_at": time.time()})
    if code != 0:
        raise RuntimeError(f"{phase} exited {code}; artifacts retained and no retry attempted")


def count_csv(path: Path) -> int:
    with path.open() as handle:
        return sum(1 for _ in csv.DictReader(handle))


def reality_gap_summary(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 5000:
        raise ValueError(f"{path}: expected 5000 reality-gap rows, found {len(rows)}")
    observed_key = "observed_price" if "observed_price" in rows[0] else "actual_paid"
    observed = np_array(rows, observed_key)
    predicted = np_array(rows, "rl_predicted_price")
    anchor = np_array(rows, "anchor_tariff_price")
    semantics = sorted({str(row.get("price_semantics", "unknown")) for row in rows})
    return {
        "rows": len(rows),
        "price_semantics": semantics[0] if len(semantics) == 1 else semantics,
        "observed_price_column": observed_key,
        "observed_price_mean": float(observed.mean()),
        # Compatibility alias for consumers created before the semantic audit.
        "actual_price_mean": float(observed.mean()),
        "predicted_price_mean": float(predicted.mean()),
        "anchor_price_mean": float(anchor.mean()),
        "mae": float(abs(predicted - observed).mean()),
        "anchor_mae": float(abs(anchor - observed).mean()),
        "mae_improvement_vs_anchor": float(abs(anchor - observed).mean() - abs(predicted - observed).mean()),
        "rmse": float(math.sqrt(float(((predicted - observed) ** 2).mean()))),
        "bias": float((predicted - observed).mean()),
        "correlation": correlation(observed, predicted),
        "within_2_dollars": float((abs(predicted - observed) <= 2.0).mean()),
        "within_5_dollars": float((abs(predicted - observed) <= 5.0).mean()),
    }


def np_array(rows: list[dict[str, str]], key: str):
    import numpy as np

    values = np.asarray([float(row[key]) for row in rows], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite reality-gap values in {key}")
    return values


def correlation(left, right) -> float | None:
    import numpy as np

    left_values = np.asarray(left, dtype=float)
    right_values = np.asarray(right, dtype=float)
    if left_values.size < 2 or right_values.size != left_values.size:
        return None
    if float(np.std(left_values)) <= 1e-12 or float(np.std(right_values)) <= 1e-12:
        return None
    value = float(np.corrcoef(left_values, right_values)[0, 1])
    return value if math.isfinite(value) else None


def validate_training(
    slug: str,
    out: Path,
    days: int,
    stage_fractions: dict[str, float] | None = None,
) -> dict[str, Any]:
    stage_fractions = dict(stage_fractions or STAGE_FRACTIONS)
    diagnostics = out / f"{slug}_training_diagnostics.csv"
    evaluations = out / f"{slug}_evaluation_diagnostics.csv"
    benchmarks = out / f"{slug}_benchmark_summary.csv"
    gap = out / f"{slug}_observed_price_comparison.csv"
    for path in (diagnostics, evaluations, benchmarks, gap, out / "policy.final_training.pt"):
        if not path.is_file():
            raise FileNotFoundError(path)
    if count_csv(diagnostics) != days:
        raise ValueError(f"{slug}: incomplete training diagnostics")
    if count_csv(evaluations) != 120 or count_csv(benchmarks) != 5:
        raise ValueError(f"{slug}: incomplete standard evaluation")
    with diagnostics.open() as handle:
        stage_counts = Counter(row["training_stage"] for row in csv.DictReader(handle))
    expected_counts = {
        "market": round(days * stage_fractions["market"]),
        "static_opponent": round(days * stage_fractions["static"]),
        "price_gap_catchup": round(days * stage_fractions["catchup"]),
        "profit_maximizers": (
            days
            - round(days * stage_fractions["market"])
            - round(days * stage_fractions["static"])
            - round(days * stage_fractions["catchup"])
        ),
    }
    if dict(stage_counts) != expected_counts:
        raise ValueError(f"{slug}: stage counts {dict(stage_counts)} != {expected_counts}")
    summary = reality_gap_summary(gap)
    atomic_json(out / "reality_gap_summary.json", summary)
    return {"stage_days": dict(stage_counts), "reality_gap": summary}


def validate_full_evaluation(slug: str, out: Path) -> dict[str, Any]:
    details: dict[str, Any] = {}
    for case_name in CASES:
        path = out / "full_evaluation" / slug / case_name / "eval" / "sweep.json"
        data = json.loads(path.read_text())
        if set(data.get("traces", {})) != {"rl_actor"}:
            raise ValueError(f"{path}: missing frozen RL policy trace")
        trace = data["traces"]["rl_actor"]
        rollouts = Counter(row["rollout"] for row in trace)
        if len(rollouts) != len(RIVALS) * 10 or set(rollouts.values()) != {240}:
            raise ValueError(f"{path}: incomplete rival/opening/step coverage")
        observed = {str(row.get("rival_controller")) for row in trace}
        if set(RIVALS) - observed:
            raise ValueError(f"{path}: rival controllers not exercised: {sorted(set(RIVALS) - observed)}")
        required = {"revenue_per_request", "rival_revenue_per_request", "fulfillment", "rival_fulfillment"}
        if trace and required - set(trace[0]):
            raise ValueError(f"{path}: missing expanded business metrics {sorted(required - set(trace[0]))}")
        details[case_name] = {"rollouts": len(rollouts), "steps": len(trace), "rivals": len(observed)}
    return details


def prepare_root(
    root: Path,
    days: int,
    seed: int,
    stage_fractions: dict[str, float] | None = None,
) -> None:
    stage_fractions = dict(stage_fractions or STAGE_FRACTIONS)
    root.mkdir(parents=True, exist_ok=False)
    inputs = []
    for slug, settings in CITIES.items():
        inputs.extend([
            ROOT / f"data/processed/{slug}/calibration.json",
            ROOT / f"data/processed/{slug}/trip_bank.csv",
            ROOT / f"artifacts/v26_confirmatory/thresholds/{slug}/llm_numeric_primary.jsonl",
            CITY_CALIBRATION_ROOT / slug / "city_calibration.json",
            ROOT / settings["ride_root"],
        ])
    missing = [str(path) for path in inputs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing multicity inputs: {missing}")
    atomic_json(root / "manifest.json", {
        "revision": "15c",
        "created_at": time.time(),
        "days_per_city": days,
        "seed": seed,
        "cities": CITIES,
        "stage_fractions": stage_fractions,
        "stage_days": {key: int(round(days * value)) for key, value in stage_fractions.items()},
        "catchup_training_opponents": list(CATCHUP_MODES),
        "profit_training_opponents": list(PROFIT_MODES),
        "action_contract": {
            "kind": "factorized_coefficient_then_magnitude",
            "coefficient_targets_by_city": {
                slug: manipulated_coefficient_count(slug) for slug in CITIES
            },
            "magnitude_targets": list(MAGNITUDE_TARGETS),
            "joint_actions_including_hold_by_city": {
                slug: 1 + manipulated_coefficient_count(slug) * len(MAGNITUDE_TARGETS)
                for slug in CITIES
            },
            "identifiability_rule": (
                "exclude airport_fee when the public trip calibration does not "
                "identify an airport surcharge or its tariff anchor is zero"
            ),
        },
        "stage_profit_lead_weights": list(STAGE_PROFIT_LEAD_WEIGHTS),
        "stage_market_share_competitiveness_weights": list(STAGE_MARKET_SHARE_WEIGHTS),
        "heldout_evaluation_opponent_groups": {key: list(value) for key, value in RIVAL_GROUPS.items()},
        "heldout_cases": list(CASES),
        "evaluation_openings_per_case_opponent": 10,
        "evaluation_horizon_days": 40,
        "kaggle_rows_per_city": 5000,
        "source_hashes": source_hashes(),
        "execution": "three independent city jobs; no automatic retry or policy substitution",
    })


def city_job(args: argparse.Namespace) -> int:
    slug = args.city
    out = args.output.resolve()
    stage_fractions = resolve_stage_fractions(args.days, args.stage_days)
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / "started.json", {"pid": os.getpid(), "started_at": time.time(), "city": slug})
    command = build_training_command(slug, out, args.days, args.seed, stage_fractions)
    atomic_json(out / "pipeline_contract.json", {
        "city": CITIES[slug]["market"],
        "days": args.days,
        "seed": args.seed,
        "action_contract": "factorized_coefficient_then_magnitude",
        "manipulated_price_coefficients": manipulated_coefficient_count(slug),
        "joint_actions_including_hold": (
            1 + manipulated_coefficient_count(slug) * len(MAGNITUDE_TARGETS)
        ),
        "magnitude_targets": list(MAGNITUDE_TARGETS),
        "stage_profit_lead_weights": list(STAGE_PROFIT_LEAD_WEIGHTS),
        "stage_market_share_competitiveness_weights": list(STAGE_MARKET_SHARE_WEIGHTS),
        "stage_fractions": stage_fractions,
        "stage_days": {key: int(round(args.days * value)) for key, value in stage_fractions.items()},
        "training_catchup_opponents": list(CATCHUP_MODES),
        "training_profit_opponents": list(PROFIT_MODES),
        "evaluation_rivals": list(RIVALS),
        "evaluation_cases": list(CASES),
    })
    policy = out / "policy.final_training.pt"
    try:
        run_process(command, out, "training", "training.log")
        training = validate_training(slug, out, args.days, stage_fractions)
        policy_digest = sha256(policy)
        v26r_figures.training(out, slug)
        v26r_figures.standard(out, slug)
        update_status(out, phase="full_benchmark_evaluation", state="running", policy_sha256=policy_digest, training=training)
        run_process(evaluation_command(slug, out, policy), out, "full_benchmark_evaluation", "full_benchmark_evaluation.log")
        evaluation = validate_full_evaluation(slug, out)
        run_process(summarizer_command(slug, out), out, "benchmark_summary", "benchmark_summary.log")
        if sha256(policy) != policy_digest:
            raise ValueError("trained policy changed during held-out evaluation")
        v26r_figures.gallery(out)
        update_status(
            out,
            phase="complete",
            state="complete",
            completed_at=time.time(),
            policy_sha256=policy_digest,
            training=training,
            evaluation=evaluation,
        )
        atomic_json(out / "exit_status.json", {"returncode": 0, "completed_at": time.time()})
        return 0
    except BaseException as exc:
        traceback.print_exc()
        update_status(out, phase="failed", state="failed", error=repr(exc), failed_at=time.time())
        atomic_json(out / "exit_status.json", {"returncode": 1, "error": repr(exc), "failed_at": time.time()})
        return 1
    finally:
        if args.launch_label:
            subprocess.Popen(
                ["/bin/launchctl", "remove", args.launch_label],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--city", choices=tuple(CITIES))
    parser.add_argument("--days", type=int, default=4000)
    parser.add_argument(
        "--stage-days",
        default="",
        help="exact market,static,catchup,profit days; must sum to --days",
    )
    parser.add_argument("--seed", type=int, default=251012)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--launch-label", default="")
    args = parser.parse_args()
    if args.days < 1:
        parser.error("--days must be positive")
    try:
        stage_fractions = resolve_stage_fractions(args.days, args.stage_days)
    except ValueError as exc:
        parser.error(str(exc))
    if args.prepare_only:
        if args.city:
            parser.error("--prepare-only and --city are mutually exclusive")
        prepare_root(args.output.resolve(), args.days, args.seed, stage_fractions)
        return 0
    if not args.city:
        parser.error("--city is required unless --prepare-only is used")
    return city_job(args)


if __name__ == "__main__":
    raise SystemExit(main())
