#!/usr/bin/env python3
"""Dedicated Revision 15d full-counterfactual protocol.

The three cities run independently and concurrently.  Every real pricing
decision in every curriculum stage replays every *feasible* hybrid action for
the full rival-memory horizon under common random numbers.  The resulting
action advantages supervise the sampled-action PPO advantage, the all-pairs
ranking loss, branch imitation, and the diagnostic Q head.

The default 12-hour, 196-day schedule is deliberately aligned to 14-day trajectories:
14 market, 14 static, 56 catch-up, and 112 profit-maximizer days.  It gives
every catch-up opponent one complete episode, every profit-maximizer one
complete episode, and 14 actor/critic updates per city while retaining a
12-day causal credit horizon.  The evaluation is broad but bounded: all 23 rivals, five
policy comparators, six representative cases, one paired opening, 14 days,
plus the standard evaluation and a 1,000-row Kaggle reality-gap analysis.
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


# Keep three concurrently trained cities from multiplying BLAS worker pools.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BASE))

from diagnostic_logging import atomic_json  # noqa: E402
import run_revision15d_8h_multicity as bounded  # noqa: E402
import run_revision15d_multicity_full as full  # noqa: E402


CITIES = full.CITIES
STAGE_KEYS = full.STAGE_KEYS
DEFAULT_DAYS = 196
DEFAULT_STAGE_DAYS = (14, 14, 56, 112)
COMPUTE_BUDGET_HOURS = 12
NMCC_HORIZON_DAYS = 12
NMCC_ACTOR_WEIGHT = 1.0
NMCC_POLICY_COEFF = 0.25
BRANCH_IMITATION_COEFF = 0.10
TRAJECTORY_DAYS = 14
ROLLOUT_EPISODES = 1
MIN_ROLLOUT_TRANSITIONS = TRAJECTORY_DAYS * ROLLOUT_EPISODES
NMCC_STAGES = "market,static,catchup,profit"
EVALUATION_OPENINGS = 1
PROFIT_PLANNER_SAMPLE_CAP = 8

# Measured on the stopped 7-day exhaustive pilot and the completed bounded
# six-case evaluation.  The contingency covers the longer horizon and slower
# profit-planning opponents.
RUNTIME_BASIS = {
    "measured_nonprofit_nmcc_seconds_per_decision_12day": 27.5,
    "measured_profit_nmcc_seconds_per_decision_cap32": 281.1,
    "production_profit_planner_sample_cap": PROFIT_PLANNER_SAMPLE_CAP,
    "measured_profit_nmcc_seconds_per_decision_cap8": 129.6,
    "projected_training_hours": 4.67,
    "slow_profit_controller_contingency_hours": 1.80,
    "measured_six_case_evaluation_hours": 2.60,
    "expected_total_hours": 9.07,
    "hard_planning_envelope_hours": COMPUTE_BUDGET_HOURS,
}


def exact_stage_days(days: int, text: str = "") -> tuple[int, int, int, int]:
    if text:
        try:
            values = tuple(int(value.strip()) for value in text.split(","))
        except ValueError as exc:
            raise ValueError("--stage-days must contain four integers") from exc
    elif days == DEFAULT_DAYS:
        values = DEFAULT_STAGE_DAYS
    else:
        # Preserve the dedicated proportions for controlled smoke runs.
        first = max(1, round(days * DEFAULT_STAGE_DAYS[0] / DEFAULT_DAYS))
        second = max(1, round(days * DEFAULT_STAGE_DAYS[1] / DEFAULT_DAYS))
        third = max(1, round(days * DEFAULT_STAGE_DAYS[2] / DEFAULT_DAYS))
        fourth = days - first - second - third
        values = (first, second, third, fourth)
    if len(values) != 4 or any(value < 1 for value in values):
        raise ValueError("all four curriculum stages need at least one day")
    if sum(values) != int(days):
        raise ValueError(
            f"--stage-days sum to {sum(values)}, expected --days={days}"
        )
    return values


def stage_fractions(days: int, values: tuple[int, int, int, int]) -> dict[str, float]:
    return bounded.resolve_stage_fractions(
        days, ",".join(str(value) for value in values)
    )


def nmcc_dump_path(out: Path, slug: str) -> Path:
    return out / f"{slug}_nmcc_full_credit.jsonl"


def build_training_command(
    slug: str,
    out: Path,
    days: int,
    seed: int,
    fractions: dict[str, float],
    protocol_profile: str = "12h",
) -> list[str]:
    command = bounded.build_training_command(
        slug,
        out,
        days,
        seed,
        fractions,
        nmcc_every_k_decisions=1,
        nmcc_horizon_days=NMCC_HORIZON_DAYS,
        nmcc_actor_weight=NMCC_ACTOR_WEIGHT,
        nmcc_stages=NMCC_STAGES,
    )
    for flag, value in (
        ("--nmcc_policy_coeff", NMCC_POLICY_COEFF),
        ("--nmcc_baseline", "hold"),
        ("--nmcc_advantage_scale", "rms"),
        ("--nmcc_dump", nmcc_dump_path(out, slug).resolve()),
        ("--branch_imitation_coeff", BRANCH_IMITATION_COEFF),
        ("--profit_planner_sample_cap", PROFIT_PLANNER_SAMPLE_CAP),
        ("--training_dynamic_episode_days", TRAJECTORY_DAYS),
        ("--ppo_rollout_episodes", ROLLOUT_EPISODES),
        ("--ppo_min_rollout_transitions", MIN_ROLLOUT_TRANSITIONS),
        ("--ppo_batch_size", TRAJECTORY_DAYS),
        ("--ppo_update_interval_days", 2),
        ("--ppo_update_epochs", 12),
        (
            "--trained_model_id",
            f"r15d-{protocol_profile}-{slug}-full-cf-{days}-s{seed}",
        ),
    ):
        full.r14.set_option(command, flag, str(value))

    expected = {
        "--nmcc_branches": str(full.hybrid_action_count(slug)),
        "--nmcc_every_k_decisions": "1",
        "--nmcc_horizon_days": str(NMCC_HORIZON_DAYS),
        "--nmcc_actor_weight": str(NMCC_ACTOR_WEIGHT),
        "--nmcc_policy_coeff": str(NMCC_POLICY_COEFF),
        "--nmcc_stages": NMCC_STAGES,
        "--branch_imitation_coeff": str(BRANCH_IMITATION_COEFF),
        "--profit_planner_sample_cap": str(PROFIT_PLANNER_SAMPLE_CAP),
        "--ppo_rollout_episodes": str(ROLLOUT_EPISODES),
        "--ppo_min_rollout_transitions": str(MIN_ROLLOUT_TRANSITIONS),
    }
    mismatches = {
        flag: (full.option(command, flag), value)
        for flag, value in expected.items()
        if full.option(command, flag) != value
    }
    if mismatches:
        raise ValueError(f"full-credit wiring mismatch: {mismatches}")
    return command


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def validate_full_counterfactual_credit(
    slug: str,
    out: Path,
    days: int,
    expected_stage_days: tuple[int, int, int, int],
) -> dict[str, Any]:
    dump = nmcc_dump_path(out, slug)
    if not dump.is_file():
        raise FileNotFoundError(dump)
    records = [
        json.loads(line) for line in dump.read_text().splitlines() if line.strip()
    ]
    if len(records) != days:
        raise ValueError(
            f"{slug}: {len(records)} NMCC decisions, expected {days}"
        )
    observed_days = [int(row["day"]) for row in records]
    if observed_days != list(range(days)):
        raise ValueError(f"{slug}: NMCC did not cover every training day")
    expected_counts = {
        key: value
        for key, value in zip(STAGE_KEYS, expected_stage_days)
        if value > 0
    }
    observed_counts = Counter(str(row["stage"]) for row in records)
    if dict(observed_counts) != expected_counts:
        raise ValueError(
            f"{slug}: NMCC stage counts {dict(observed_counts)} != "
            f"{expected_counts}"
        )

    seconds = 0.0
    feasible_counts: list[int] = []
    spreads: list[float] = []
    for row in records:
        if int(row["horizon_days"]) != NMCC_HORIZON_DAYS:
            raise ValueError(f"{slug}: truncated NMCC horizon in {row}")
        feasible = int(row["feasible_action_count"])
        computed = int(row["computed_action_count"])
        advantages = dict(row["values"])
        raw_values = dict(row["raw_values"])
        if not (
            feasible >= 2
            and computed == feasible
            and len(advantages) == feasible
            and len(raw_values) == feasible
            and math.isclose(
                float(row["advantage_coverage"]), 1.0, abs_tol=1e-12
            )
        ):
            raise ValueError(f"{slug}: incomplete feasible-action credit {row}")
        if row["selected"] not in advantages or "hold" not in advantages:
            raise ValueError(f"{slug}: selected/hold branch missing in {row}")
        values = [float(value) for value in advantages.values()]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{slug}: non-finite action advantage in {row}")
        if not math.isclose(float(advantages["hold"]), 0.0, abs_tol=1e-6):
            raise ValueError(f"{slug}: hold-relative advantage is not zero")
        feasible_counts.append(feasible)
        spreads.append(max(values) - min(values))
        seconds += float(row["seconds"])

    diagnostics = out / f"{slug}_training_diagnostics.csv"
    with diagnostics.open() as handle:
        updates = [
            row for row in csv.DictReader(handle)
            if _truthy(row.get("ppo_update_performed", ""))
        ]
    minimum_updates = max(1, days // TRAJECTORY_DAYS - 4)
    if len(updates) < minimum_updates:
        raise ValueError(
            f"{slug}: only {len(updates)} PPO updates, expected at least "
            f"{minimum_updates}"
        )
    for row in updates:
        rollout = int(float(row["ppo_rollout_size"]))
        branch_rows = int(float(row["ppo_branch_imitation_rows"]))
        if not math.isclose(
            float(row["ppo_nmcc_valid_fraction"]), 1.0, abs_tol=1e-6
        ):
            raise ValueError(f"{slug}: selected-action NMCC advantage missing")
        if not math.isclose(
            float(row["ppo_counterfactual_actor_weight"]),
            NMCC_ACTOR_WEIGHT,
            abs_tol=1e-6,
        ):
            raise ValueError(f"{slug}: full counterfactual actor weight missing")
        if branch_rows != rollout or rollout < 1:
            raise ValueError(
                f"{slug}: complete all-action targets did not reach every "
                f"actor row ({branch_rows}/{rollout})"
            )
        if int(float(row["ppo_actor_optimizer_steps"])) < 1:
            raise ValueError(f"{slug}: actor did not update from full credit")

    return {
        "decisions": len(records),
        "stage_decisions": dict(observed_counts),
        "action_advantage_coverage": 1.0,
        "feasible_action_count_min": min(feasible_counts),
        "feasible_action_count_max": max(feasible_counts),
        "advantage_spread_mean": sum(spreads) / len(spreads),
        "nmcc_seconds": seconds,
        "ppo_updates": len(updates),
        "actor_weight": NMCC_ACTOR_WEIGHT,
        "all_action_ranking_coeff": NMCC_POLICY_COEFF,
        "branch_imitation_coeff": BRANCH_IMITATION_COEFF,
    }


def prepare_root(
    root: Path,
    days: int,
    seed: int,
    fractions: dict[str, float],
    counts: tuple[int, int, int, int],
    compute_budget_hours: float = COMPUTE_BUDGET_HOURS,
    protocol_profile: str = "12h",
) -> None:
    bounded.prepare_root(
        root,
        days,
        seed,
        fractions,
        nmcc_every_k_decisions=1,
        nmcc_horizon_days=NMCC_HORIZON_DAYS,
        nmcc_actor_weight=NMCC_ACTOR_WEIGHT,
        nmcc_stages=NMCC_STAGES,
        evaluation_openings=EVALUATION_OPENINGS,
    )
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(
        {
            "revision": f"15d-{protocol_profile}-full-counterfactual",
            "protocol_profile": protocol_profile,
            "compute_budget_hours": compute_budget_hours,
            "training_days_per_city": days,
            "stage_days": dict(zip(STAGE_KEYS, counts)),
            "runtime_basis": RUNTIME_BASIS,
            "full_counterfactual_contract": {
                "every_training_decision": True,
                "every_curriculum_stage": list(STAGE_KEYS),
                "every_feasible_hybrid_action": True,
                "horizon_days": NMCC_HORIZON_DAYS,
                "common_random_numbers": True,
                "actor_weight": NMCC_ACTOR_WEIGHT,
                "all_action_ranking_coeff": NMCC_POLICY_COEFF,
                "branch_imitation_coeff": BRANCH_IMITATION_COEFF,
                "audit_dump_per_city": "<city>_nmcc_full_credit.jsonl",
            },
            "update_contract": {
                "trajectory_days": TRAJECTORY_DAYS,
                "rollout_episodes_per_update": ROLLOUT_EPISODES,
                "minimum_transitions": MIN_ROLLOUT_TRANSITIONS,
                "expected_complete_updates_per_city": days
                // TRAJECTORY_DAYS,
            },
        }
    )
    atomic_json(manifest_path, manifest)


def city_job(args: argparse.Namespace) -> int:
    slug = args.city
    out = args.output.resolve()
    counts = exact_stage_days(args.days, args.stage_days)
    fractions = stage_fractions(args.days, counts)
    out.mkdir(parents=True, exist_ok=False)
    started_at = time.time()
    atomic_json(
        out / "started.json",
        {
            "pid": os.getpid(),
            "started_at": started_at,
            "city": slug,
            "planning_deadline_at": started_at
            + args.compute_budget_hours * 3600,
        },
    )
    command = build_training_command(
        slug,
        out,
        args.days,
        args.seed,
        fractions,
        protocol_profile=args.protocol_profile,
    )
    if args.training_smoke:
        for flag, value in (
            ("--checkpoint_validation_horizon", 1),
            ("--eval_timesteps", 1),
            ("--comparison_limit", 20),
        ):
            full.r14.set_option(command, flag, str(value))
    atomic_json(
        out / "pipeline_contract.json",
        {
            "revision": f"15d-{args.protocol_profile}-full-counterfactual",
            "protocol_profile": args.protocol_profile,
            "compute_budget_hours": args.compute_budget_hours,
            "city": CITIES[slug]["market"],
            "days": args.days,
            "seed": args.seed,
            "stage_days": dict(zip(STAGE_KEYS, counts)),
            "stage_fractions": fractions,
            "action_contract": full.action_contract(),
            "learning_contract": {
                **full.learning_contract(),
                "trajectory_days": TRAJECTORY_DAYS,
                "independent_trajectories_per_update": ROLLOUT_EPISODES,
                "minimum_transitions_per_update": MIN_ROLLOUT_TRANSITIONS,
                "nmcc_every_k_decisions": 1,
                "nmcc_horizon_days": NMCC_HORIZON_DAYS,
                "nmcc_actor_weight": NMCC_ACTOR_WEIGHT,
                "nmcc_policy_coeff": NMCC_POLICY_COEFF,
                "nmcc_stages": NMCC_STAGES,
                "branch_imitation_coeff": BRANCH_IMITATION_COEFF,
                "full_counterfactual_every_decision": True,
            },
            "evaluation_contract": bounded.evaluation_contract(
                openings=EVALUATION_OPENINGS
            ),
            "runtime_basis": RUNTIME_BASIS,
        },
    )
    policy = out / "policy.final_training.pt"
    try:
        full.inherited.run_process(command, out, "training", "training.log")
        full_credit = validate_full_counterfactual_credit(
            slug, out, args.days, counts
        )
        if args.training_smoke:
            training = {"full_counterfactual": full_credit}
            policy_digest = full.inherited.sha256(policy)
            full.inherited.update_status(
                out,
                phase="training_smoke_complete",
                state="complete",
                completed_at=time.time(),
                elapsed_hours=(time.time() - started_at) / 3600.0,
                policy_sha256=policy_digest,
                training=training,
            )
            atomic_json(
                out / "exit_status.json",
                {"returncode": 0, "completed_at": time.time()},
            )
            return 0
        training = bounded.validate_training(
            slug, out, args.days, fractions
        )
        training["full_counterfactual"] = full_credit
        policy_digest = full.inherited.sha256(policy)
        full.inherited.v26r_figures.training(out, slug)
        full.inherited.v26r_figures.standard(out, slug)
        full.inherited.update_status(
            out,
            phase="six_case_benchmark_evaluation",
            state="running",
            policy_sha256=policy_digest,
            training=training,
        )
        full.inherited.run_process(
            bounded.evaluation_command(
                slug,
                out,
                policy,
                openings=EVALUATION_OPENINGS,
            ),
            out,
            "full_benchmark_evaluation",
            "full_benchmark_evaluation.log",
        )
        evaluation = bounded.validate_full_evaluation(
            slug, out, openings=EVALUATION_OPENINGS
        )
        full.inherited.run_process(
            bounded.summarizer_command(
                slug, out, openings=EVALUATION_OPENINGS
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
            elapsed_hours=(time.time() - started_at) / 3600.0,
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

            # ``launchctl submit`` jobs otherwise restart after a normal exit.
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
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    parser.add_argument("--stage-days", default="")
    parser.add_argument("--seed", type=int, default=251026)
    parser.add_argument("--compute-budget-hours", type=float, default=COMPUTE_BUDGET_HOURS)
    parser.add_argument("--protocol-profile", default="12h")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--training-smoke", action="store_true")
    parser.add_argument("--launch-label", default="")
    args = parser.parse_args()
    if args.days < 4:
        parser.error("--days must allow all four stages")
    try:
        counts = exact_stage_days(args.days, args.stage_days)
        fractions = stage_fractions(args.days, counts)
    except ValueError as exc:
        parser.error(str(exc))
    if args.prepare_only:
        if args.city or args.training_smoke:
            parser.error(
                "--prepare-only is mutually exclusive with --city and "
                "--training-smoke"
            )
        prepare_root(
            args.output.resolve(),
            args.days,
            args.seed,
            fractions,
            counts,
            compute_budget_hours=args.compute_budget_hours,
            protocol_profile=args.protocol_profile,
        )
        return 0
    if not args.city:
        parser.error("--city is required unless --prepare-only is used")
    return city_job(args)


if __name__ == "__main__":
    raise SystemExit(main())
