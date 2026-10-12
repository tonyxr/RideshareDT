#!/usr/bin/env python3
"""Run a cumulative, v25-rooted ablation ladder for the v29 redesign.

The frozen ``baseline_v25`` tree is rung 0.  Every later rung uses the active
tree and adds exactly one documented layer.  Commands, source hashes, tests,
logs, and exit status are persisted before moving to the next rung.  The
runner is intentionally sequential so stages do not compete for CPU or alter
the interpretation of wall-clock measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "baseline_v25"


@dataclass(frozen=True)
class Budget:
    train_steps: int
    train_customers: int
    eval_steps: int
    eval_customers: int
    validation_horizon: int
    validation_interval_days: int
    validation_customers: int
    validation_openings: int
    rollout: int
    batch: int
    epochs: int
    pool: int


BUDGETS = {
    "smoke": Budget(240, 100, 120, 100, 120, 60, 100, 1, 12, 12, 2, 512),
    "pilot": Budget(2400, 250, 400, 250, 300, 150, 250, 3, 24, 24, 4, 5000),
    "full": Budget(7200, 1000, 1000, 1000, 1000, 600, 1000, 5, 32, 32, 6, 20000),
}


@dataclass(frozen=True)
class Stage:
    stage_id: str
    label: str
    implementation: str
    added_change: str
    test_files: tuple[str, ...]
    extra_args: tuple[str, ...] = ()


STAGES = (
    Stage(
        "s0_frozen_v25",
        "Frozen v25",
        "baseline",
        "None: exact restored v25 reference",
        (
            "test_core_ppo_fixes.py",
            "test_platform_mdp.py",
            "test_ppo_agent.py",
            "test_v25_credit_assignment.py",
        ),
    ),
    Stage(
        "s1_active_v25_contract",
        "Active code, v25 contract",
        "active",
        "Code refactors only; v26-v29 features remain disabled",
        (
            "test_core_ppo_fixes.py",
            "test_platform_mdp.py",
            "test_ppo_agent.py",
            "test_v25_credit_assignment.py",
        ),
    ),
    Stage(
        "s2_v29_setup",
        "+ v29 data/setup",
        "active",
        "Schema-v2 city calibration, trip bank and native profile cache",
        (
            "test_city_data_pipeline.py",
            "test_model_registry_and_dataset_validation.py",
            "test_price_threshold_flow.py",
            "test_v28_threshold_batch.py",
        ),
        ("--gpt_threshold_mode", "llm_numeric"),
    ),
    Stage(
        "s3_v29_simulator",
        "+ v29 simulator",
        "active",
        "Expected-flow v27 market, calibrated shocks and rider-profile memory",
        (
            "test_v27_market_engine.py",
            "test_v27_core_integration.py",
            "test_v28_shock_sampling.py",
            "test_market_generalization.py",
        ),
        (
            "--v27",
            "--shocks",
            "calibrated",
            "--public_shock_features",
            "--rider_memory",
            "--gpt_threshold_mode",
            "llm_numeric",
        ),
    ),
    Stage(
        "s4_v29_mdp",
        "+ v29 MDP",
        "active",
        "Five independent upper-unbounded coefficients, anchor floor and concise foresight state",
        (
            "test_v29_ppo_redesign.py",
            "test_platform_mdp.py",
            "test_v28_unbounded_economics.py",
            "test_v29_readiness_contracts.py",
        ),
        (
            "--v28",
            "--shocks",
            "calibrated",
            "--public_shock_features",
            "--rider_memory",
            "--independent_coefficient_actions",
            "--compact_foresight_observation",
            "--manipulated_price_coefficients",
            "5",
            "--minimum_tariff_multiplier",
            "1.0",
            "--state_frame_stack",
            "4",
            "--no-persistent_recurrence",
        ),
    ),
    Stage(
        "s5_v29_learner",
        "+ v29 learner",
        "active",
        "Explicit v29 PPO contract, profit-with-dominance reward and environment-then-opponent curriculum",
        (
            "test_v29_ppo_redesign.py",
            "test_v29_readiness_contracts.py",
            "test_ppo_agent.py",
            "test_v24_credit_assignment.py",
        ),
        (
            "--v29",
            "--shocks",
            "calibrated",
            "--public_shock_features",
            "--training_curriculum",
            "environment_then_opponent",
            "--policy_objective",
            "profit_with_dominance",
        ),
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _common_training_args(stage: Stage, budget: Budget, seed: int, out: Path) -> list[str]:
    prefix = out / "chicago"
    objective = "profit_with_dominance" if stage.stage_id == "s5_v29_learner" else "profit_50_competitiveness_50"
    args = [
        "--run_experiment",
        "--market", "Chicago",
        "--choice_mode", "cognitive",
        "--firm1_mode", "RL",
        "--firm2_mode", "adaptive_best_response_aggressive",
        "--deterministic_experiment_seed",
        "--deterministic_torch",
        "--seed", str(seed),
        "--experiment_seed", str(seed),
        "--policy_objective", objective,
        "--pool", str(budget.pool),
        "--train_timesteps", str(budget.train_steps),
        "--train_customers", str(budget.train_customers),
        "--train_steps_per_day", "4",
        "--firm1_action_interval_steps", "4",
        "--firm2_action_interval_steps", "4",
        "--ppo_update_interval_days", "5",
        "--ppo_min_rollout_transitions", str(budget.rollout),
        "--ppo_batch_size", str(budget.batch),
        "--ppo_update_epochs", str(budget.epochs),
        "--training_stage_market_fraction", "0.10",
        "--training_stage_static_fraction", "0.10",
        "--training_stage_dynamic_fraction", "0.80",
        "--training_dynamic_episode_days", "30",
        "--checkpoint_validation_interval_days", str(budget.validation_interval_days),
        "--checkpoint_validation_horizon", str(budget.validation_horizon),
        "--checkpoint_validation_customers", str(budget.validation_customers),
        "--checkpoint_validation_openings", str(budget.validation_openings),
        "--eval_timesteps", str(budget.eval_steps),
        "--eval_customers", str(budget.eval_customers),
        "--benchmark_suite",
        "--report_prefix", str(prefix),
        "--trained_model_out", str(out / "policy.pt"),
        "--trained_model_id", f"v25-v29-ablation-{stage.stage_id}-seed{seed}",
        "--model_registry_dir", str(out / "registry"),
    ]
    if stage.implementation == "baseline":
        return args + [
            "--credit_assignment_v25",
            "--threshold_profile_source", "cached",
            "--threshold_cache_path", str(
                ROOT / "artifacts/three_city_full_20260825/raw/thresholds/chicago/primary.jsonl"
            ),
            "--calibration_json", str(BASELINE / "data/processed/chicago/calibration.json"),
        ]

    active = args + [
        "--threshold_profile_source", "cached",
        "--threshold_cache_path", str(
            ROOT / (
                "artifacts/three_city_full_20260825/raw/thresholds/chicago/primary.jsonl"
                if stage.stage_id == "s1_active_v25_contract"
                else "artifacts/v28_thresholds/chicago/native_uncapped_primary.jsonl"
            )
        ),
        "--calibration_json", str(
            BASELINE / "data/processed/chicago/calibration.json"
            if stage.stage_id == "s1_active_v25_contract"
            else ROOT / "data/processed/chicago/calibration.json"
        ),
        "--trip_bank_path", str(ROOT / "data/processed/chicago/trip_bank.csv"),
        "--baseline_suite", "static_grid,myopic_greedy,anchor_hold",
        "--static_baseline_multipliers", "0.6,0.8,1.0,1.2,1.4,1.5,1.75,2.0,3.0,4.0",
        "--ood_perturbed_variants", "0",
        "--unrestricted_policy_experiment",
        "--defer_final_test",
    ]
    if stage.stage_id != "s5_v29_learner":
        active.append("--credit_assignment_v25")
    return active + list(stage.extra_args)


def command_for(stage: Stage, budget: Budget, seed: int, out: Path) -> list[str]:
    core = BASELINE / "Core.py" if stage.implementation == "baseline" else ROOT / "Core.py"
    return [sys.executable, "-u", str(core)] + _common_training_args(stage, budget, seed, out)


def test_command(stage: Stage) -> list[str]:
    base = BASELINE if stage.implementation == "baseline" else ROOT
    command = [sys.executable, "-m", "pytest", "-q", *[str(base / item) for item in stage.test_files]]
    if stage.stage_id == "s2_v29_setup":
        # The default test interpreter has no pyarrow. The preprocessed bundle
        # contracts remain covered here; raw-Parquet I/O is a separately
        # reported runtime dependency check rather than a model regression.
        command.extend(["-k", "not test_downloaded_bundles_pass_small_backtest_when_sources_exist"])
    return command


def _run_logged(command: list[str], cwd: Path, log_path: Path, env: dict[str, str]) -> tuple[int, float]:
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + shlex.join(command) + "\n")
        log.flush()
        result = subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
    return int(result.returncode), float(time.monotonic() - started)


def _selected_stages(values: Iterable[str]) -> list[Stage]:
    wanted = set(values)
    if not wanted or "all" in wanted:
        return list(STAGES)
    by_id = {stage.stage_id: stage for stage in STAGES}
    unknown = wanted - set(by_id)
    if unknown:
        raise SystemExit("unknown stages: " + ", ".join(sorted(unknown)))
    return [stage for stage in STAGES if stage.stage_id in wanted]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", choices=tuple(BUDGETS), default="smoke")
    parser.add_argument("--seeds", type=int, nargs="+", default=[17101])
    parser.add_argument("--stages", nargs="+", default=["all"])
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/v25_v29_ablation")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--tests-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    options = parser.parse_args()

    budget = BUDGETS[options.budget]
    stages = _selected_stages(options.stages)
    root_out = options.out.resolve() / options.budget
    root_out.mkdir(parents=True, exist_ok=True)
    source_files = [
        ROOT / "Core.py", ROOT / "PPOAgent.py", ROOT / "platform_mdp.py",
        ROOT / "pricing_models.py", ROOT / "choice_models.py",
        ROOT / "market_dynamics.py", ROOT / "market_engine.py",
    ]
    manifest = {
        "schema": "v25_v29_cumulative_ablation_v1",
        "budget": asdict(budget),
        "seeds": options.seeds,
        "stages": [asdict(stage) for stage in stages],
        "source_sha256": {str(path.relative_to(ROOT)): _sha256(path) for path in source_files},
    }
    _write_json(root_out / "manifest.json", manifest)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    env.setdefault("MPLCONFIGDIR", str(root_out / ".mpl"))

    for stage in stages:
        if not options.skip_tests:
            stage_test_dir = root_out / "tests" / stage.stage_id
            stage_test_dir.mkdir(parents=True, exist_ok=True)
            test_status = stage_test_dir / "status.json"
            if not (options.resume and test_status.exists() and json.loads(test_status.read_text()).get("exit_code") == 0):
                command = test_command(stage)
                if options.dry_run:
                    print("TEST", stage.stage_id, shlex.join(command))
                else:
                    code, seconds = _run_logged(
                        command,
                        BASELINE if stage.implementation == "baseline" else ROOT,
                        stage_test_dir / "pytest.log",
                        env,
                    )
                    _write_json(test_status, {"exit_code": code, "seconds": seconds, "command": command})
                    if code:
                        print(f"{stage.stage_id}: tests failed; see {stage_test_dir / 'pytest.log'}", file=sys.stderr)
                        return code

        for seed in options.seeds:
            if options.tests_only:
                continue
            run_out = root_out / stage.stage_id / f"seed_{seed}"
            run_out.mkdir(parents=True, exist_ok=True)
            status_path = run_out / "status.json"
            if options.resume and status_path.exists() and json.loads(status_path.read_text()).get("exit_code") == 0:
                continue
            command = command_for(stage, budget, seed, run_out)
            _write_json(
                run_out / "run_manifest.json",
                {"stage": asdict(stage), "budget": asdict(budget), "seed": seed, "command": command},
            )
            if options.dry_run:
                print("RUN", stage.stage_id, seed, shlex.join(command))
                continue
            cwd = BASELINE if stage.implementation == "baseline" else ROOT
            code, seconds = _run_logged(command, cwd, run_out / "run.log", env)
            _write_json(status_path, {"exit_code": code, "seconds": seconds})
            if code:
                print(f"{stage.stage_id}/seed_{seed}: run failed; see {run_out / 'run.log'}", file=sys.stderr)
                return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
