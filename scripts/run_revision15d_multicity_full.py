#!/usr/bin/env python3
"""Revision 15d multicity pipeline: hybrid relative actions and dense updates.

The inherited 15c pipeline still owns validation, held-out benchmark evaluation,
Kaggle reality-gap analysis, process supervision, and plotting.  This launcher
replaces only the training/action contract and records that contract explicitly.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BASE))

from diagnostic_logging import atomic_json  # noqa: E402
import run_revision14_verify as r14  # noqa: E402
import run_revision15c_multicity_full as inherited  # noqa: E402
from run_v26r_staged import source_hashes  # noqa: E402
from summarize_revision15b_benchmarks import (  # noqa: E402
    GENUINELY_HELD_OUT_RIVALS,
    PERTURBED_HELD_OUT_RIVALS,
    RIVAL_EXPOSURE,
    RIVAL_GROUPS,
    RIVALS,
    TRAINING_EXPOSED_RIVALS,
)
from v26r_cases import CASES, STATIC_GRID  # noqa: E402


CITIES = inherited.CITIES
CATCHUP_MODES = inherited.CATCHUP_MODES
PROFIT_MODES = inherited.PROFIT_MODES
STAGE_KEYS = inherited.STAGE_KEYS
STAGE_FRACTIONS = inherited.STAGE_FRACTIONS
STAGE_PROFIT_LEAD_WEIGHTS = inherited.STAGE_PROFIT_LEAD_WEIGHTS
STAGE_MARKET_SHARE_WEIGHTS = inherited.STAGE_MARKET_SHARE_WEIGHTS

RELATIVE_ACTION_STEPS = (0.10, 0.20)
TRAJECTORY_DAYS = 14
ROLLOUT_EPISODES = 2
MIN_ROLLOUT_TRANSITIONS = TRAJECTORY_DAYS * ROLLOUT_EPISODES
PPO_UPDATE_INTERVAL_DAYS = 2
PPO_BATCH_SIZE = 14
PPO_UPDATE_EPOCHS = 12
PPO_GAMMA = 0.995
PPO_GAE_LAMBDA = 0.90
# The memory rival reacts after one day, retaliates for about 5.5 days and
# recovers over about 4.3 more. Twelve days makes the complete response visible
# to paired branch credit instead of stopping before retaliation/recovery ends.
NMCC_HORIZON_DAYS = 12
NMCC_EVERY_K_DECISIONS = 3
BRANCH_IMITATION_COEFF = 0.05
EVALUATION_SPECS = (
    "rl_actor",
    "rl_actor_frozen",
    "opening_hold",
    *(f"static:{target:g}" for target in STATIC_GRID),
)
R15B_SEARCH_ROOT = BASE / "evaluation_assets" / "r15b_searched_tariffs"


manipulated_coefficient_count = inherited.manipulated_coefficient_count
option = inherited.option
remove_option = inherited.remove_option
add_flag = inherited.add_flag
resolve_stage_fractions = inherited.resolve_stage_fractions


def hybrid_action_count(slug: str) -> int:
    """Hold + four signed magnitudes for whole tariff and every live lever."""
    return 1 + 4 * (1 + manipulated_coefficient_count(slug))


def r15b_searched_tariff_path(slug: str) -> Path:
    return R15B_SEARCH_ROOT / f"{slug}.json"


def evaluation_specs(slug: str) -> tuple[str, ...]:
    """Matched controls plus the city-specific Revision 15b searched tariff."""
    path = r15b_searched_tariff_path(slug).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{slug}: missing Revision 15b searched tariff {path}")
    return (*EVALUATION_SPECS, f"static_tariff:@{path}")


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
        nmcc_branches=hybrid_action_count(slug),
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
        "--tariff_targets",
        "--stage_entropy_end_scale",
        "--stage_lr_end_scale",
    ):
        remove_option(command, flag)
    add_flag(command, "--hybrid_relative_actions")
    for flag, value in (
        ("--market", settings["market"]),
        ("--threshold_cache_path", inherited.ROOT / f"artifacts/v26_confirmatory/thresholds/{slug}/llm_numeric_primary.jsonl"),
        ("--calibration_json", inherited.ROOT / f"data/processed/{slug}/calibration.json"),
        ("--city_calibration_artifact", inherited.CITY_CALIBRATION_ROOT / slug / "city_calibration.json"),
        ("--training_stage_market_fraction", stage_fractions["market"]),
        ("--training_stage_static_fraction", stage_fractions["static"]),
        ("--training_stage_dynamic_fraction", stage_fractions["catchup"]),
        ("--training_stage_profit_fraction", stage_fractions["profit"]),
        ("--training_catchup_opponent_modes", ",".join(CATCHUP_MODES)),
        ("--training_profit_opponent_modes", ",".join(PROFIT_MODES)),
        ("--report_prefix", os.path.relpath(out / slug, BASE)),
        ("--trained_model_out", out / "policy.pt"),
        ("--trained_model_id", f"r15d-{slug}-public-{days}-s{seed}-hybrid-relative"),
        ("--model_registry_dir", out / "registry"),
        ("--rl_warm_start_target", "0"),
        ("--relative_action_steps", ",".join(str(value) for value in RELATIVE_ACTION_STEPS)),
        ("--manipulated_price_coefficients", manipulated_coefficient_count(slug)),
        ("--ppo_entropy_start", "0.01"),
        ("--training_dynamic_episode_days", TRAJECTORY_DAYS),
        ("--ppo_rollout_episodes", ROLLOUT_EPISODES),
        ("--ppo_update_interval_days", PPO_UPDATE_INTERVAL_DAYS),
        ("--ppo_min_rollout_transitions", MIN_ROLLOUT_TRANSITIONS),
        ("--ppo_batch_size", PPO_BATCH_SIZE),
        ("--ppo_update_epochs", PPO_UPDATE_EPOCHS),
        ("--ppo_gamma", PPO_GAMMA),
        ("--ppo_gae_lambda", PPO_GAE_LAMBDA),
        ("--nmcc_branches", hybrid_action_count(slug)),
        ("--nmcc_horizon_days", NMCC_HORIZON_DAYS),
        ("--nmcc_every_k_decisions", NMCC_EVERY_K_DECISIONS),
        ("--branch_imitation_coeff", BRANCH_IMITATION_COEFF),
        ("--stage_profit_lead_weights", ",".join(str(value) for value in STAGE_PROFIT_LEAD_WEIGHTS)),
        ("--stage_market_share_competitiveness_weights", ",".join(str(value) for value in STAGE_MARKET_SHARE_WEIGHTS)),
    ):
        r14.set_option(command, flag, str(value))

    add_flag(command, "--benchmark_suite")
    add_flag(command, "--compare_with_dataset")
    for flag, value in (
        ("--dataset_root", inherited.ROOT / settings["ride_root"]),
        ("--dataset_city", settings["market"]),
        ("--weather_root", inherited.ROOT / "data/raw/weather/noaa"),
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
        "--relative_action_steps": ",".join(str(value) for value in RELATIVE_ACTION_STEPS),
        "--manipulated_price_coefficients": str(manipulated_coefficient_count(slug)),
        "--training_dynamic_episode_days": str(TRAJECTORY_DAYS),
        "--ppo_rollout_episodes": str(ROLLOUT_EPISODES),
        "--ppo_update_interval_days": str(PPO_UPDATE_INTERVAL_DAYS),
        "--ppo_min_rollout_transitions": str(MIN_ROLLOUT_TRANSITIONS),
        "--ppo_batch_size": str(PPO_BATCH_SIZE),
        "--ppo_update_epochs": str(PPO_UPDATE_EPOCHS),
        "--ppo_gamma": str(PPO_GAMMA),
        "--ppo_gae_lambda": str(PPO_GAE_LAMBDA),
        "--nmcc_branches": str(hybrid_action_count(slug)),
        "--nmcc_horizon_days": str(NMCC_HORIZON_DAYS),
        "--nmcc_every_k_decisions": str(NMCC_EVERY_K_DECISIONS),
        "--branch_imitation_coeff": str(BRANCH_IMITATION_COEFF),
        "--stage_profit_lead_weights": ",".join(str(value) for value in STAGE_PROFIT_LEAD_WEIGHTS),
        "--stage_market_share_competitiveness_weights": ",".join(str(value) for value in STAGE_MARKET_SHARE_WEIGHTS),
    }
    if "--learning_corrections" not in command or "--hybrid_relative_actions" not in command:
        raise ValueError(f"{slug}: Revision 15d requires corrected hybrid-relative learning")
    for flag, value in expected.items():
        if option(command, flag) != value:
            raise ValueError(f"{slug}: {flag}={option(command, flag)!r}, expected {value!r}")
    for flag in ("--benchmark_suite", "--compare_with_dataset"):
        if flag not in command:
            raise ValueError(f"{slug}: missing {flag}")
    for flag in (
        "--level_step_actions",
        "--whole_tariff_actions",
        "--tariff_targets",
        "--tariff_structure",
        "--actor_true_shortage_flag",
        "--stage_entropy_end_scale",
        "--stage_lr_end_scale",
    ):
        if flag in command:
            raise ValueError(f"{slug}: forbidden Revision 15d option {flag}")
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
    """Full paired policy/rival/case matrix with correctly labelled exposure."""
    return [
        str(inherited.PYTHON), "-u", str(HERE / "v26r_case_eval.py"),
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
        "--workers", "3",
        "--budget-seconds", "1000000000",
        "--specs", ";".join(evaluation_specs(slug)),
    ]


def summarizer_command(slug: str, out: Path) -> list[str]:
    return [
        str(inherited.PYTHON), "-u",
        str(HERE / "summarize_revision15b_benchmarks.py"),
        "--evaluation-root", str((out / "full_evaluation").resolve()),
        "--slug", slug,
        "--city", CITIES[slug]["market"],
        "--output", str((out / "full_evaluation_summary").resolve()),
    ]


def validate_full_evaluation(slug: str, out: Path) -> dict[str, object]:
    """Reject missing comparators, opponent variants, or paired trace cells."""
    details: dict[str, object] = {}
    expected_specs = set(evaluation_specs(slug))
    steps_per_rollout = 40 * 6
    for case_name in CASES:
        path = out / "full_evaluation" / slug / case_name / "eval" / "sweep.json"
        data = json.loads(path.read_text())
        traces = dict(data.get("traces", {}))
        if set(traces) != expected_specs:
            raise ValueError(
                f"{path}: policy specs {sorted(traces)} != {sorted(expected_specs)}"
            )
        for spec, trace in traces.items():
            rollouts = Counter(str(row["rollout"]) for row in trace)
            if len(rollouts) != len(RIVALS) * 10 or set(rollouts.values()) != {
                steps_per_rollout
            }:
                raise ValueError(
                    f"{path}: incomplete rival/opening/step coverage for {spec}"
                )
            observed_rivals = {
                str(name).rsplit("__opening_", 1)[0] for name in rollouts
            }
            if observed_rivals != set(RIVALS):
                raise ValueError(
                    f"{path}: wrong rollout rival set for {spec}: "
                    f"{sorted(observed_rivals ^ set(RIVALS))}"
                )
            observed_controllers = {
                str(row.get("rival_controller")) for row in trace
            }
            if set(RIVALS) - observed_controllers:
                raise ValueError(
                    f"{path}: rival controllers not exercised for {spec}: "
                    f"{sorted(set(RIVALS) - observed_controllers)}"
                )
            required = {
                "revenue_per_request", "rival_revenue_per_request",
                "fulfillment", "rival_fulfillment",
            }
            if trace and required - set(trace[0]):
                raise ValueError(
                    f"{path}: missing business metrics "
                    f"{sorted(required - set(trace[0]))}"
                )
        details[case_name] = {
            "rivals": len(RIVALS),
            "policy_specs": len(expected_specs),
            "rollouts_per_spec": len(RIVALS) * 10,
            "steps_per_spec": len(RIVALS) * 10 * steps_per_rollout,
        }
    return details


def action_contract() -> dict:
    return {
        "kind": "hybrid_relative_tariff_steps_v1",
        "targets": "hold, whole tariff, and each identifiable coefficient",
        "signed_relative_steps": [-0.20, -0.10, 0.10, 0.20],
        "joint_actions_including_hold_by_city": {
            slug: hybrid_action_count(slug) for slug in CITIES
        },
        "coefficient_targets_by_city": {
            slug: manipulated_coefficient_count(slug) for slug in CITIES
        },
        "no_op_cap_actions_masked": True,
        "boston_airport_rule": (
            "use four coefficients while the public calibration does not "
            "identify a positive airport surcharge"
        ),
    }


def learning_contract() -> dict:
    return {
        "trajectory_days": TRAJECTORY_DAYS,
        "independent_trajectories_per_update": ROLLOUT_EPISODES,
        "minimum_transitions_per_update": MIN_ROLLOUT_TRANSITIONS,
        "update_check_interval_days": PPO_UPDATE_INTERVAL_DAYS,
        "minibatch_size": PPO_BATCH_SIZE,
        "epochs_per_update": PPO_UPDATE_EPOCHS,
        "gamma": PPO_GAMMA,
        "gae_lambda": PPO_GAE_LAMBDA,
        "nmcc_horizon_days": NMCC_HORIZON_DAYS,
        "nmcc_truncate_at_real_episode_boundary": False,
        "nmcc_every_k_decisions": NMCC_EVERY_K_DECISIONS,
        "nmcc_branches": "all feasible hybrid actions",
        "branch_imitation_coeff": BRANCH_IMITATION_COEFF,
        "memory_rival_response_window_days": {
            "reaction_lag": 1.0,
            "retaliation_hold": 5.5,
            "recovery": 4.3,
            "total": 10.8,
        },
    }


def evaluation_contract() -> dict:
    searched = {
        slug: r15b_searched_tariff_path(slug).resolve() for slug in CITIES
    }
    missing = [str(path) for path in searched.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing Revision 15b searched tariffs: {missing}")
    return {
        "policy_specs_by_city": {
            slug: list(evaluation_specs(slug)) for slug in CITIES
        },
        "policy_comparators": {
            "adaptive": "rl_actor",
            "learned_tariff_frozen_after_opening_action": "rl_actor_frozen",
            "randomized_opening_tariff_frozen": "opening_hold",
            "fixed_whole_tariff_grid": [
                spec for spec in EVALUATION_SPECS if spec.startswith("static:")
            ],
            "revision15b_city_searched_tariff": {
                slug: {
                    "spec": f"static_tariff:@{path}",
                    "path": str(path),
                    "sha256": inherited.sha256(path),
                }
                for slug, path in searched.items()
            },
        },
        "opponent_groups": {key: list(value) for key, value in RIVAL_GROUPS.items()},
        "opponent_exposure_labels": dict(RIVAL_EXPOSURE),
        "training_exposed_opponents": list(TRAINING_EXPOSED_RIVALS),
        "genuinely_held_out_opponents": list(GENUINELY_HELD_OUT_RIVALS),
        "perturbed_genuinely_held_out_opponents": list(PERTURBED_HELD_OUT_RIVALS),
        "parallel_case_workers_per_city": 3,
        "parallelism_contract": (
            "cases use disjoint deterministic caches and may run concurrently; "
            "rollouts within a case and all policy-training transitions remain "
            "sequential"
        ),
        "best_fixed_reporting": (
            "retain every fixed grid result; any best-fixed same-evaluation-cell "
            "summary is labelled ex-post and is not held-out policy selection"
        ),
    }


def prepare_root(root: Path, days: int, seed: int, stage_fractions=None) -> None:
    inherited.prepare_root(root, days, seed, stage_fractions)
    manifest = json.loads((root / "manifest.json").read_text())
    # Revision 15c called every benchmark opponent and every case held out,
    # even where the controller class or scenario family appeared in training.
    manifest.pop("heldout_evaluation_opponent_groups", None)
    manifest.pop("heldout_cases", None)
    manifest.update({
        "revision": "15d",
        "action_contract": action_contract(),
        "learning_contract": learning_contract(),
        "evaluation_contract": evaluation_contract(),
        "evaluation_cases": list(CASES),
        "evaluation_case_roles": {
            name: str(config.get("role", "unspecified"))
            for name, config in CASES.items()
        },
        "source_hashes": source_hashes(),
    })
    atomic_json(root / "manifest.json", manifest)


def city_job(args: argparse.Namespace) -> int:
    slug = args.city
    out = args.output.resolve()
    stage_fractions = resolve_stage_fractions(args.days, args.stage_days)
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / "started.json", {
        "pid": os.getpid(), "started_at": time.time(), "city": slug,
    })
    command = build_training_command(slug, out, args.days, args.seed, stage_fractions)
    atomic_json(out / "pipeline_contract.json", {
        "revision": "15d",
        "city": CITIES[slug]["market"],
        "days": args.days,
        "seed": args.seed,
        "action_contract": action_contract(),
        "learning_contract": learning_contract(),
        "evaluation_contract": evaluation_contract(),
        "stage_fractions": stage_fractions,
        "stage_days": {
            key: int(round(args.days * value))
            for key, value in stage_fractions.items()
        },
        "training_catchup_opponents": list(CATCHUP_MODES),
        "training_profit_opponents": list(PROFIT_MODES),
        "evaluation_rivals": list(RIVALS),
        "evaluation_cases": list(CASES),
    })
    policy = out / "policy.final_training.pt"
    try:
        inherited.run_process(command, out, "training", "training.log")
        training = inherited.validate_training(
            slug, out, args.days, stage_fractions
        )
        policy_digest = inherited.sha256(policy)
        inherited.v26r_figures.training(out, slug)
        inherited.v26r_figures.standard(out, slug)
        inherited.update_status(
            out,
            phase="full_benchmark_evaluation",
            state="running",
            policy_sha256=policy_digest,
            training=training,
        )
        inherited.run_process(
            evaluation_command(slug, out, policy),
            out,
            "full_benchmark_evaluation",
            "full_benchmark_evaluation.log",
        )
        evaluation = validate_full_evaluation(slug, out)
        inherited.run_process(
            summarizer_command(slug, out),
            out,
            "benchmark_summary",
            "benchmark_summary.log",
        )
        if inherited.sha256(policy) != policy_digest:
            raise ValueError("trained policy changed during benchmark evaluation")
        inherited.v26r_figures.gallery(out)
        inherited.update_status(
            out,
            phase="complete",
            state="complete",
            completed_at=time.time(),
            policy_sha256=policy_digest,
            training=training,
            evaluation=evaluation,
        )
        atomic_json(out / "exit_status.json", {
            "returncode": 0, "completed_at": time.time(),
        })
        return 0
    except BaseException as exc:
        traceback.print_exc()
        inherited.update_status(
            out,
            phase="failed",
            state="failed",
            error=repr(exc),
            failed_at=time.time(),
        )
        atomic_json(out / "exit_status.json", {
            "returncode": 1, "error": repr(exc), "failed_at": time.time(),
        })
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
    parser.add_argument("--days", type=int, default=4000)
    parser.add_argument("--stage-days", default="")
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
