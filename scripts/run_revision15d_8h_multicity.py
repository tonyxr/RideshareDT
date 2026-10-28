#!/usr/bin/env python3
"""Eight-hour Revision 15d protocol with auditable compute stratification.

The policy still receives 6,000 real training days per city, two complete
14-day trajectories per PPO rollout, all feasible hybrid actions at every NMCC
call, and the full 12-day rival-response credit horizon. Runtime is bounded by
sampling those exhaustive NMCC calls sparsely and by using a stratified,
parallel evaluation rather than the publication-scale Cartesian product.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BASE))

from diagnostic_logging import atomic_json  # noqa: E402
import run_revision15d_multicity_full as full  # noqa: E402


CITIES = full.CITIES
STAGE_KEYS = full.STAGE_KEYS
STAGE_FRACTIONS = full.STAGE_FRACTIONS

# Prime cadence avoids repeatedly landing on the same 14-day episode phase or
# the same position in the deterministic opponent rotation.
NMCC_EVERY_K_DECISIONS = 211
STANDARD_EVAL_DAYS = 14
REALITY_GAP_ROWS = 1000
EVALUATION_OPENINGS = 2
EVALUATION_HORIZON_DAYS = 14
EVALUATION_WORKERS = 3
EVALUATION_CASES = (
    "A1_shortage_severe",
    "A3_storm_transit",
    "S1_promotion_bait",
    "S4_rival_switch",
    "S6_shortage_during_promotion",
    "C0_calm",
)


def resolve_stage_fractions(
    days: int, stage_days: str = ""
) -> dict[str, float]:
    """Preserve exact integer stage counts through float progress boundaries.

    ``TrainingStageScheduler`` rounds cumulative boundaries to 12 decimals.
    A rational boundary such as 2/48 can therefore round upward and include
    day index 2 in the preceding stage.  Put requested integer boundaries a
    tiny, auditable amount below the following day's exact progress while
    preserving the requested rounded counts and a unit total.
    """
    fractions = full.resolve_stage_fractions(days, stage_days)
    if not stage_days:
        return fractions
    counts = tuple(int(value.strip()) for value in stage_days.split(","))
    epsilon = 1e-10
    boundaries = [
        sum(counts[:index]) / float(days) - epsilon
        for index in range(1, len(counts))
    ]
    values = (
        boundaries[0],
        boundaries[1] - boundaries[0],
        boundaries[2] - boundaries[1],
        1.0 - boundaries[2],
    )
    return dict(zip(STAGE_KEYS, values))


def evaluation_specs(slug: str) -> tuple[str, ...]:
    """Adaptive, frozen, calibrated-fixed and searched-fixed controls."""
    return (
        "rl_actor",
        "rl_actor_frozen",
        "opening_hold",
        "static:1",
        f"static_tariff:@{full.r15b_searched_tariff_path(slug).resolve()}",
    )


def evaluation_contract(
    *, openings: int = EVALUATION_OPENINGS
) -> dict[str, Any]:
    return {
        "protocol": "revision15d_stratified_eight_hour",
        "cases": list(EVALUATION_CASES),
        "case_selection": (
            "calm plus severe shortage, storm/transit, promotion bait, "
            "rival switch and compound shortage/promotion"
        ),
        "rivals": list(full.RIVALS),
        "policy_specs_by_city": {
            slug: list(evaluation_specs(slug)) for slug in CITIES
        },
        "openings_per_case_rival_policy": int(openings),
        "horizon_days": EVALUATION_HORIZON_DAYS,
        "case_workers_per_city": EVALUATION_WORKERS,
        "training_exposed_opponents": list(full.TRAINING_EXPOSED_RIVALS),
        "genuinely_held_out_opponents": list(full.GENUINELY_HELD_OUT_RIVALS),
        "perturbed_genuinely_held_out_opponents": list(
            full.PERTURBED_HELD_OUT_RIVALS
        ),
        "limitation": (
            "screening evaluation, not the publication-scale 12-case x "
            "10-opening x 17-policy Cartesian product"
        ),
    }


def build_training_command(
    slug: str,
    out: Path,
    days: int,
    seed: int,
    stage_fractions: dict[str, float],
    *,
    nmcc_every_k_decisions: int = NMCC_EVERY_K_DECISIONS,
    nmcc_horizon_days: int = full.NMCC_HORIZON_DAYS,
    nmcc_actor_weight: float | None = None,
    nmcc_stages: str | None = None,
) -> list[str]:
    command = full.build_training_command(
        slug, out, days, seed, stage_fractions
    )
    for flag, value in (
        ("--nmcc_every_k_decisions", nmcc_every_k_decisions),
        ("--nmcc_horizon_days", nmcc_horizon_days),
        ("--checkpoint_validation_interval_days", days),
        ("--checkpoint_validation_horizon", STANDARD_EVAL_DAYS),
        ("--eval_timesteps", STANDARD_EVAL_DAYS),
        ("--comparison_limit", REALITY_GAP_ROWS),
        ("--static_baseline_multipliers", "0.8,1.0,1.2"),
        (
            "--trained_model_id",
            f"r15d-8h-{slug}-public-{days}-s{seed}-hybrid-relative",
        ),
    ):
        full.r14.set_option(command, flag, str(value))
    if nmcc_actor_weight is not None:
        full.r14.set_option(
            command, "--nmcc_actor_weight", str(nmcc_actor_weight)
        )
    if nmcc_stages is not None:
        full.r14.set_option(command, "--nmcc_stages", nmcc_stages)
    if full.option(command, "--nmcc_horizon_days") != str(
        nmcc_horizon_days
    ):
        raise ValueError("the configured NMCC horizon was not wired")
    if full.option(command, "--nmcc_branches") != str(
        full.hybrid_action_count(slug)
    ):
        raise ValueError("the eight-hour protocol must replay every action")
    return command


def evaluation_command(
    slug: str,
    out: Path,
    policy: Path,
    *,
    openings: int = EVALUATION_OPENINGS,
) -> list[str]:
    return [
        str(full.inherited.PYTHON),
        "-u",
        str(HERE / "v26r_case_eval.py"),
        CITIES[slug]["market"],
        slug,
        ",".join(EVALUATION_CASES),
        "--seed-set",
        "eval",
        "--openings",
        str(openings),
        "--policy",
        str(policy.resolve()),
        "--output-root",
        str((out / "full_evaluation").resolve()),
        "--customers",
        "50",
        "--pool",
        "2000",
        "--rivals",
        ",".join(full.RIVALS),
        "--horizon-days",
        str(EVALUATION_HORIZON_DAYS),
        "--channels",
        "full",
        "--chunk",
        "1",
        "--workers",
        str(EVALUATION_WORKERS),
        "--budget-seconds",
        "1000000000",
        "--specs",
        ";".join(evaluation_specs(slug)),
    ]


def summarizer_command(
    slug: str,
    out: Path,
    *,
    openings: int = EVALUATION_OPENINGS,
) -> list[str]:
    return [
        str(full.inherited.PYTHON),
        "-u",
        str(HERE / "summarize_revision15b_benchmarks.py"),
        "--evaluation-root",
        str((out / "full_evaluation").resolve()),
        "--slug",
        slug,
        "--city",
        CITIES[slug]["market"],
        "--output",
        str((out / "full_evaluation_summary").resolve()),
        "--cases",
        ",".join(EVALUATION_CASES),
        "--openings",
        str(openings),
    ]


def count_csv(path: Path) -> int:
    with path.open() as handle:
        return sum(1 for _ in csv.DictReader(handle))


def reality_gap_summary(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != REALITY_GAP_ROWS:
        raise ValueError(
            f"{path}: expected {REALITY_GAP_ROWS} rows, found {len(rows)}"
        )
    observed_key = (
        "observed_price" if "observed_price" in rows[0] else "actual_paid"
    )
    observed = np.asarray([float(row[observed_key]) for row in rows])
    predicted = np.asarray([float(row["rl_predicted_price"]) for row in rows])
    anchor = np.asarray([float(row["anchor_tariff_price"]) for row in rows])
    if not all(np.isfinite(values).all() for values in (observed, predicted, anchor)):
        raise ValueError(f"{path}: non-finite price comparison")
    return {
        "rows": len(rows),
        "observed_price_mean": float(observed.mean()),
        "predicted_price_mean": float(predicted.mean()),
        "anchor_price_mean": float(anchor.mean()),
        "mae": float(np.abs(predicted - observed).mean()),
        "anchor_mae": float(np.abs(anchor - observed).mean()),
        "rmse": float(math.sqrt(float(np.square(predicted - observed).mean()))),
        "bias": float((predicted - observed).mean()),
    }


def validate_training(
    slug: str,
    out: Path,
    days: int,
    stage_fractions: dict[str, float],
) -> dict[str, Any]:
    diagnostics = out / f"{slug}_training_diagnostics.csv"
    evaluations = out / f"{slug}_evaluation_diagnostics.csv"
    benchmarks = out / f"{slug}_benchmark_summary.csv"
    gap = out / f"{slug}_observed_price_comparison.csv"
    policy = out / "policy.final_training.pt"
    for path in (diagnostics, evaluations, benchmarks, gap, policy):
        if not path.is_file():
            raise FileNotFoundError(path)
    expected_rows = {
        diagnostics: days,
        evaluations: STANDARD_EVAL_DAYS,
        benchmarks: 5,
        gap: REALITY_GAP_ROWS,
    }
    for path, expected in expected_rows.items():
        observed = count_csv(path)
        if observed != expected:
            raise ValueError(f"{path}: {observed} rows, expected {expected}")
    with diagnostics.open() as handle:
        stage_counts = Counter(
            row["training_stage"] for row in csv.DictReader(handle)
        )
    expected_stage_counts = {
        "market": round(days * stage_fractions["market"]),
        "static_opponent": round(days * stage_fractions["static"]),
        "price_gap_catchup": round(days * stage_fractions["catchup"]),
    }
    expected_stage_counts["profit_maximizers"] = days - sum(
        expected_stage_counts.values()
    )
    if dict(stage_counts) != expected_stage_counts:
        raise ValueError(
            f"{slug}: stage counts {dict(stage_counts)} != "
            f"{expected_stage_counts}"
        )
    gap_summary = reality_gap_summary(gap)
    atomic_json(out / "reality_gap_summary.json", gap_summary)
    return {"stage_days": dict(stage_counts), "reality_gap": gap_summary}


def validate_full_evaluation(
    slug: str,
    out: Path,
    *,
    openings: int = EVALUATION_OPENINGS,
) -> dict[str, Any]:
    expected_specs = set(evaluation_specs(slug))
    expected_rollouts = len(full.RIVALS) * int(openings)
    expected_steps = EVALUATION_HORIZON_DAYS * 6
    details: dict[str, Any] = {}
    for case_name in EVALUATION_CASES:
        path = (
            out
            / "full_evaluation"
            / slug
            / case_name
            / "eval"
            / "sweep.json"
        )
        data = json.loads(path.read_text())
        traces = dict(data.get("traces", {}))
        if set(traces) != expected_specs:
            raise ValueError(f"{path}: incomplete policy set")
        for spec, trace in traces.items():
            rollouts = Counter(str(row["rollout"]) for row in trace)
            if len(rollouts) != expected_rollouts or set(rollouts.values()) != {
                expected_steps
            }:
                raise ValueError(f"{path}: incomplete trace for {spec}")
            rivals = {
                name.rsplit("__opening_", 1)[0] for name in rollouts
            }
            if rivals != set(full.RIVALS):
                raise ValueError(f"{path}: incomplete rivals for {spec}")
        details[case_name] = {
            "rivals": len(full.RIVALS),
            "policy_specs": len(expected_specs),
            "rollouts_per_spec": expected_rollouts,
            "steps_per_rollout": expected_steps,
        }
    return details


def prepare_root(
    root: Path,
    days: int,
    seed: int,
    stage_fractions: dict[str, float],
    *,
    nmcc_every_k_decisions: int = NMCC_EVERY_K_DECISIONS,
    nmcc_horizon_days: int = full.NMCC_HORIZON_DAYS,
    nmcc_actor_weight: float | None = None,
    nmcc_stages: str | None = None,
    evaluation_openings: int = EVALUATION_OPENINGS,
) -> None:
    full.prepare_root(root, days, seed, stage_fractions)
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest.update(
        {
            "revision": (
                "15d-full-counterfactual-today"
                if int(nmcc_every_k_decisions) == 1
                else "15d-8h"
            ),
            "compute_budget_hours": 8,
            "nmcc_every_k_decisions": int(nmcc_every_k_decisions),
            "nmcc_horizon_days": int(nmcc_horizon_days),
            "nmcc_branches": "all feasible hybrid actions",
            "nmcc_actor_weight": nmcc_actor_weight,
            "nmcc_stages": nmcc_stages,
            "full_counterfactual_every_eligible_decision": (
                int(nmcc_every_k_decisions) == 1
            ),
            "standard_evaluation_days": STANDARD_EVAL_DAYS,
            "reality_gap_rows": REALITY_GAP_ROWS,
            "evaluation_openings_per_case_opponent": int(
                evaluation_openings
            ),
            "evaluation_horizon_days": EVALUATION_HORIZON_DAYS,
            "kaggle_rows_per_city": REALITY_GAP_ROWS,
            "evaluation_contract": evaluation_contract(
                openings=evaluation_openings
            ),
            "source_hashes": full.source_hashes(),
        }
    )
    atomic_json(path, manifest)


def city_job(args: argparse.Namespace) -> int:
    slug = args.city
    out = args.output.resolve()
    stage_fractions = resolve_stage_fractions(args.days, args.stage_days)
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(
        out / "started.json",
        {"pid": os.getpid(), "started_at": time.time(), "city": slug},
    )
    command = build_training_command(
        slug,
        out,
        args.days,
        args.seed,
        stage_fractions,
        nmcc_every_k_decisions=args.nmcc_every_k_decisions,
        nmcc_horizon_days=args.nmcc_horizon_days,
        nmcc_actor_weight=args.nmcc_actor_weight,
        nmcc_stages=args.nmcc_stages,
    )
    atomic_json(
        out / "pipeline_contract.json",
        {
            "revision": (
                "15d-full-counterfactual-today"
                if args.nmcc_every_k_decisions == 1
                else "15d-8h"
            ),
            "city": CITIES[slug]["market"],
            "days": args.days,
            "seed": args.seed,
            "stage_fractions": stage_fractions,
            "action_contract": full.action_contract(),
            "learning_contract": {
                **full.learning_contract(),
                "nmcc_every_k_decisions": args.nmcc_every_k_decisions,
                "nmcc_horizon_days": args.nmcc_horizon_days,
                "nmcc_actor_weight": args.nmcc_actor_weight,
                "nmcc_stages": args.nmcc_stages,
                "full_counterfactual_every_eligible_decision": (
                    args.nmcc_every_k_decisions == 1
                ),
            },
            "evaluation_contract": evaluation_contract(
                openings=args.evaluation_openings
            ),
        },
    )
    policy = out / "policy.final_training.pt"
    try:
        full.inherited.run_process(command, out, "training", "training.log")
        training = validate_training(
            slug, out, args.days, stage_fractions
        )
        policy_digest = full.inherited.sha256(policy)
        full.inherited.v26r_figures.training(out, slug)
        full.inherited.v26r_figures.standard(out, slug)
        full.inherited.update_status(
            out,
            phase="stratified_benchmark_evaluation",
            state="running",
            policy_sha256=policy_digest,
            training=training,
        )
        full.inherited.run_process(
            evaluation_command(
                slug, out, policy, openings=args.evaluation_openings
            ),
            out,
            "full_benchmark_evaluation",
            "full_benchmark_evaluation.log",
        )
        evaluation = validate_full_evaluation(
            slug, out, openings=args.evaluation_openings
        )
        full.inherited.run_process(
            summarizer_command(
                slug, out, openings=args.evaluation_openings
            ),
            out,
            "benchmark_summary",
            "benchmark_summary.log",
        )
        if full.inherited.sha256(policy) != policy_digest:
            raise ValueError("trained policy changed during evaluation")
        full.inherited.v26r_figures.gallery(out)
        full.inherited.update_status(
            out,
            phase="complete",
            state="complete",
            completed_at=time.time(),
            policy_sha256=policy_digest,
            training=training,
            evaluation=evaluation,
        )
        atomic_json(
            out / "exit_status.json",
            {"returncode": 0, "completed_at": time.time()},
        )
        return 0
    except BaseException as exc:
        traceback.print_exc()
        full.inherited.update_status(
            out,
            phase="failed",
            state="failed",
            error=repr(exc),
            failed_at=time.time(),
        )
        atomic_json(
            out / "exit_status.json",
            {"returncode": 1, "error": repr(exc), "failed_at": time.time()},
        )
        return 1
    finally:
        if args.launch_label:
            import subprocess

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
    parser.add_argument("--days", type=int, default=6000)
    parser.add_argument("--stage-days", default="")
    parser.add_argument("--seed", type=int, default=251025)
    parser.add_argument(
        "--nmcc-every-k-decisions",
        type=int,
        default=NMCC_EVERY_K_DECISIONS,
    )
    parser.add_argument(
        "--nmcc-horizon-days",
        type=int,
        default=full.NMCC_HORIZON_DAYS,
    )
    parser.add_argument("--nmcc-actor-weight", type=float)
    parser.add_argument("--nmcc-stages")
    parser.add_argument(
        "--evaluation-openings", type=int, default=EVALUATION_OPENINGS
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--launch-label", default="")
    args = parser.parse_args()
    if args.days < 1:
        parser.error("--days must be positive")
    if args.nmcc_every_k_decisions < 1:
        parser.error("--nmcc-every-k-decisions must be positive")
    if args.nmcc_horizon_days < 1:
        parser.error("--nmcc-horizon-days must be positive")
    if args.nmcc_actor_weight is not None and not (
        0.0 <= args.nmcc_actor_weight <= 1.0
    ):
        parser.error("--nmcc-actor-weight must be in [0, 1]")
    if args.evaluation_openings < 1:
        parser.error("--evaluation-openings must be positive")
    try:
        stage_fractions = resolve_stage_fractions(
            args.days, args.stage_days
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.prepare_only:
        if args.city:
            parser.error("--prepare-only and --city are mutually exclusive")
        prepare_root(
            args.output.resolve(),
            args.days,
            args.seed,
            stage_fractions,
            nmcc_every_k_decisions=args.nmcc_every_k_decisions,
            nmcc_horizon_days=args.nmcc_horizon_days,
            nmcc_actor_weight=args.nmcc_actor_weight,
            nmcc_stages=args.nmcc_stages,
            evaluation_openings=args.evaluation_openings,
        )
        return 0
    if not args.city:
        parser.error("--city is required unless --prepare-only is used")
    return city_job(args)


if __name__ == "__main__":
    raise SystemExit(main())
