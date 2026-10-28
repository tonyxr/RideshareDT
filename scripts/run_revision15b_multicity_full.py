#!/usr/bin/env python3
"""Full-budget Revision 15b training, benchmark evaluation and reality-gap analysis.

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
from pricing_models import TARIFF_LEVEL_KEYS, tariff_level_of, validate_tariff_structure  # noqa: E402
import run_revision14_verify as r14  # noqa: E402
import run_revision15_verify as r15  # noqa: E402
from run_v26r_staged import environment, source_hashes  # noqa: E402
from summarize_revision15b_benchmarks import RIVAL_GROUPS, RIVALS  # noqa: E402
from v26r_cases import CASES  # noqa: E402
import v26r_figures  # noqa: E402


CITIES = {
    "nyc": {"market": "New York City", "ride_root": "data/raw/rides/nyc_kaggle"},
    "boston": {"market": "Boston", "ride_root": "data/raw/rides/boston_kaggle"},
    "chicago": {"market": "Chicago", "ride_root": "data/raw/rides/chicago_kaggle"},
}
DEFAULT_STRUCTURE = (
    BASE / "runs/revision15b_direct_20261023_012422/structure/tariff_structure.json"
)
CITY_CALIBRATION_ROOT = ROOT / "artifacts/revision7_direct_calibration_20261018"
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
STAGE_FRACTIONS = {"market": 0.05, "static": 0.05, "catchup": 0.375, "profit": 0.525}


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


def city_anchors(slug: str) -> dict[str, float]:
    payload = json.loads((ROOT / f"data/processed/{slug}/calibration.json").read_text())
    tariff = payload["calibration"]["tariff"]
    return {key: float(tariff[key]) for key in TARIFF_LEVEL_KEYS}


def write_city_structure(slug: str, template_path: Path, destination: Path) -> Path:
    template = json.loads(template_path.read_text())
    anchors = city_anchors(slug)
    raw_shape = {key: float(template["structure"][key]) for key in TARIFF_LEVEL_KEYS}
    city_level = tariff_level_of(raw_shape, anchors)
    normalized = {key: value / city_level for key, value in raw_shape.items()}
    payload = validate_tariff_structure(
        {
            "format": template["format"],
            "structure": normalized,
            "anchors": anchors,
            "reference_level": float(template["reference_level"]),
            "source": {
                "revision": "15b",
                "transfer_kind": "dimensionless R15b tariff shape, exposure-renormalized to city anchors",
                "source_structure": str(template_path.resolve()),
                "source_sha256": sha256(template_path),
                "city": slug,
                "pre_normalization_city_level": city_level,
            },
        },
        list(TARIFF_LEVEL_KEYS),
        (0.6, 2.0),
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(destination, payload)
    return destination.resolve()


def build_training_command(slug: str, out: Path, days: int, seed: int, structure: Path) -> list[str]:
    settings = CITIES[slug]
    command = r15.r15_train_command(
        out,
        days,
        oracle=False,
        structure=structure,
        smoke=False,
        imitation=0.5,
        seed=seed,
    )
    for flag, value in (
        ("--market", settings["market"]),
        ("--threshold_cache_path", ROOT / f"artifacts/v26_confirmatory/thresholds/{slug}/llm_numeric_primary.jsonl"),
        ("--calibration_json", ROOT / f"data/processed/{slug}/calibration.json"),
        ("--city_calibration_artifact", CITY_CALIBRATION_ROOT / slug / "city_calibration.json"),
        ("--training_stage_market_fraction", "0.05"),
        ("--training_stage_static_fraction", "0.05"),
        ("--training_stage_dynamic_fraction", "0.375"),
        ("--training_stage_profit_fraction", "0.525"),
        ("--training_catchup_opponent_modes", ",".join(CATCHUP_MODES)),
        ("--training_profit_opponent_modes", ",".join(PROFIT_MODES)),
        ("--report_prefix", os.path.relpath(out / slug, BASE)),
        ("--trained_model_out", out / "policy.pt"),
        ("--trained_model_id", f"r15b-{slug}-public-{days}-s{seed}-full-benchmark"),
        ("--model_registry_dir", out / "registry"),
        ("--tariff_structure", structure),
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
    assert_city_command(command, slug, out, days)
    return command


def assert_city_command(command: list[str], slug: str, out: Path, days: int) -> None:
    r15.assert_level_contract(command)
    expected = {
        "--market": CITIES[slug]["market"],
        "--train_timesteps": str(days),
        "--training_stage_market_fraction": "0.05",
        "--training_stage_static_fraction": "0.05",
        "--training_stage_dynamic_fraction": "0.375",
        "--training_stage_profit_fraction": "0.525",
        "--training_catchup_opponent_modes": ",".join(CATCHUP_MODES),
        "--training_profit_opponent_modes": ",".join(PROFIT_MODES),
        "--rl_warm_start_target": "0",
        "--nmcc_branches": "5",
        "--branch_imitation_coeff": "0.5",
        "--ppo_min_rollout_transitions": "30",
        "--stage_profit_lead_weights": "0.3,0.3,0.3,0.3",
    }
    for flag, value in expected.items():
        if option(command, flag) != value:
            raise ValueError(f"{slug}: {flag}={option(command, flag)!r}, expected {value!r}")
    for flag in ("--benchmark_suite", "--compare_with_dataset", "--level_step_actions"):
        if flag not in command:
            raise ValueError(f"{slug}: missing {flag}")
    for flag in ("--tariff_targets", "--whole_tariff_actions", "--actor_true_shortage_flag"):
        if flag in command:
            raise ValueError(f"{slug}: forbidden R15-B option {flag}")
    required = (
        Path(option(command, "--threshold_cache_path")),
        Path(option(command, "--calibration_json")),
        Path(option(command, "--city_calibration_artifact")),
        Path(option(command, "--tariff_structure")),
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
    actual = np_array(rows, "actual_paid")
    predicted = np_array(rows, "rl_predicted_price")
    anchor = np_array(rows, "anchor_tariff_price")
    return {
        "rows": len(rows),
        "actual_price_mean": float(actual.mean()),
        "predicted_price_mean": float(predicted.mean()),
        "anchor_price_mean": float(anchor.mean()),
        "mae": float(abs(predicted - actual).mean()),
        "anchor_mae": float(abs(anchor - actual).mean()),
        "mae_improvement_vs_anchor": float(abs(anchor - actual).mean() - abs(predicted - actual).mean()),
        "rmse": float(math.sqrt(float(((predicted - actual) ** 2).mean()))),
        "bias": float((predicted - actual).mean()),
        "correlation": float(correlation(actual, predicted)),
        "within_2_dollars": float((abs(predicted - actual) <= 2.0).mean()),
        "within_5_dollars": float((abs(predicted - actual) <= 5.0).mean()),
    }


def np_array(rows: list[dict[str, str]], key: str):
    import numpy as np

    values = np.asarray([float(row[key]) for row in rows], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite reality-gap values in {key}")
    return values


def correlation(left, right) -> float:
    import numpy as np

    return np.corrcoef(left, right)[0, 1] if len(left) > 1 else float("nan")


def validate_training(slug: str, out: Path, days: int) -> dict[str, Any]:
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
        "market": round(days * 0.05),
        "static_opponent": round(days * 0.05),
        "price_gap_catchup": round(days * 0.375),
        "profit_maximizers": days - round(days * 0.05) - round(days * 0.05) - round(days * 0.375),
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


def prepare_root(root: Path, days: int, seed: int, template: Path) -> None:
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
        "revision": "15b",
        "created_at": time.time(),
        "days_per_city": days,
        "seed": seed,
        "cities": CITIES,
        "stage_fractions": STAGE_FRACTIONS,
        "stage_days": {key: int(round(days * value)) for key, value in STAGE_FRACTIONS.items()},
        "catchup_training_opponents": list(CATCHUP_MODES),
        "profit_training_opponents": list(PROFIT_MODES),
        "heldout_evaluation_opponent_groups": {key: list(value) for key, value in RIVAL_GROUPS.items()},
        "heldout_cases": list(CASES),
        "evaluation_openings_per_case_opponent": 10,
        "evaluation_horizon_days": 40,
        "kaggle_rows_per_city": 5000,
        "structure_template": str(template.resolve()),
        "structure_template_sha256": sha256(template),
        "source_hashes": source_hashes(),
        "execution": "three independent city jobs; no automatic retry or policy substitution",
    })


def city_job(args: argparse.Namespace) -> int:
    slug = args.city
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / "started.json", {"pid": os.getpid(), "started_at": time.time(), "city": slug})
    structure = write_city_structure(slug, args.structure_template.resolve(), out / "structure/tariff_structure.json")
    command = build_training_command(slug, out, args.days, args.seed, structure)
    atomic_json(out / "pipeline_contract.json", {
        "city": CITIES[slug]["market"],
        "days": args.days,
        "seed": args.seed,
        "structure": str(structure),
        "training_catchup_opponents": list(CATCHUP_MODES),
        "training_profit_opponents": list(PROFIT_MODES),
        "evaluation_rivals": list(RIVALS),
        "evaluation_cases": list(CASES),
    })
    policy = out / "policy.final_training.pt"
    try:
        run_process(command, out, "training", "training.log")
        training = validate_training(slug, out, args.days)
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
    parser.add_argument("--seed", type=int, default=251012)
    parser.add_argument("--structure-template", type=Path, default=DEFAULT_STRUCTURE)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--launch-label", default="")
    args = parser.parse_args()
    if args.days < 1:
        parser.error("--days must be positive")
    if not args.structure_template.is_file():
        parser.error(f"missing Revision 15b structure template: {args.structure_template}")
    if args.prepare_only:
        if args.city:
            parser.error("--prepare-only and --city are mutually exclusive")
        prepare_root(args.output.resolve(), args.days, args.seed, args.structure_template.resolve())
        return 0
    if not args.city:
        parser.error("--city is required unless --prepare-only is used")
    return city_job(args)


if __name__ == "__main__":
    raise SystemExit(main())
