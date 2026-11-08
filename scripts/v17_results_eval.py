#!/usr/bin/env python3
"""Run the registered CRISP-Ride V17 held-out results evaluation.

The launcher is content-addressed and resumable only at completed Core-job
boundaries.  Every simulator product is validated for full Cartesian
coverage, finite metrics, exact horizon length, common-random-number seed,
market contract, source identity, and input hashes before it is accepted.
"""

from __future__ import annotations

import argparse
import concurrent.futures
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence


BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from diagnostic_logging import atomic_json  # noqa: E402
from scripts.v17_config import (  # noqa: E402
    CITY_SLUGS,
    canonical_hash,
    load_config,
    resolve_input,
    validate_config,
)
from scripts.v17_results_train import RUN_SPECS, RunSpec  # noqa: E402
from scripts.v17_t2_sweep import cell_spec  # noqa: E402


PYTHON = ROOT / ".venv/bin/python"
CORE = BASE / "Core.py"
EVAL_SEED = 77_311
OPENINGS = 3
HORIZON_DAYS = 40
STEPS_PER_DAY = 6
CUSTOMERS = 50
POOL = 2000
RIVALS = (
    "adaptive_best_response_memory",
    "profit_best_response_myopic",
    "profit_best_response_forward",
    "profit_best_response_anticipating",
    "profit_opt_mpc_grid",
    "pi_price_gap",
    "region_supply_demand",
    "queue_service_threshold",
    "pi_price_gap@perturbed0",
    "queue_service_threshold@perturbed0",
)
CASES = {
    "C0_calm": BASE / "scenarios/calm.json",
    "A1_shortage_severe": BASE / "scenarios/v17_t2_shortage_severe.json",
    "A3_storm_transit_late": BASE / "scenarios/cases/storm_transit_late.json",
    "S1_promotion_bait": BASE / "scenarios/cases/promotion_bait.json",
    "S6_shortage_during_promotion": (
        BASE / "scenarios/cases/shortage_during_promotion.json"
    ),
}
ABLATION_CASES = ("C0_calm", "A1_shortage_severe")
SURGE_EVENT_KINDS = ("driver_shortage", "winter_storm")
COMPARATOR_LABELS = (
    "published_card",
    "best_fixed_cell",
    "own_profit_best_fixed_cell",
    "surge_rule",
)
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "MPLBACKEND": "Agg",
    "PYTHONDONTWRITEBYTECODE": "1",
}


@dataclass(frozen=True)
class EvalJob:
    name: str
    city: str
    case: str
    kind: str
    labels: tuple[str, ...]
    specs: tuple[str, ...]
    raw: Path
    outputs: tuple[Path, ...]
    manifest: Path
    log: Path
    command: tuple[str, ...]
    input_hashes: Mapping[str, str]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_json(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _option(command: Sequence[str], flag: str) -> str:
    values = list(command)
    try:
        index = values.index(flag)
    except ValueError as exc:
        raise ValueError(f"evaluation command is missing {flag}") from exc
    if index + 1 >= len(values):
        raise ValueError(f"evaluation command has no value for {flag}")
    return str(values[index + 1])


def source_identity() -> dict[str, str]:
    paths = (
        BASE / "Core.py",
        BASE / "driver_supply.py",
        BASE / "market_dynamics.py",
        BASE / "market_shocks.py",
        BASE / "platform_mdp.py",
        BASE / "pricing_models.py",
        BASE / "v17_objective.py",
        BASE / "scripts/v17_config.py",
        BASE / "scripts/v17_results_eval.py",
        *CASES.values(),
    )
    return {str(path.relative_to(BASE)): _sha256_file(path) for path in paths}


def _event_days(case: str) -> tuple[int, ...]:
    scenario = json.loads(CASES[case].read_text(encoding="utf-8"))
    days: set[int] = set()
    for event in scenario.get("events", ()):
        if str(event.get("kind")) not in SURGE_EVENT_KINDS:
            continue
        days.update(range(int(event["start_day"]), int(event["end_day"]) + 1))
    return tuple(sorted(days))


def _schedule_spec(level: float, case: str) -> str:
    payload = {
        "default": [1.0, 1.0],
        "days": {
            str(day): [float(level), 1.0] for day in _event_days(case)
        },
    }
    return "schedule:" + json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    )


def comparator_specs(selected: Mapping[str, Any], case: str) -> dict[str, str]:
    return {
        "published_card": cell_spec(1.0, 1.0),
        "best_fixed_cell": str(selected["best_fixed_cell"]["spec"]),
        "own_profit_best_fixed_cell": str(
            selected["own_profit_best_fixed_cell"]["spec"]
        ),
        "surge_rule": _schedule_spec(
            float(selected["surge_rule"]["event_level"]), case
        ),
    }


def expected_rollouts() -> dict[str, int]:
    main_rl = 7 * len(CASES) * len(RIVALS) * OPENINGS
    ablation_rl = 3 * len(ABLATION_CASES) * len(RIVALS) * OPENINGS
    comparators = (
        len(CITY_SLUGS)
        * len(CASES)
        * len(COMPARATOR_LABELS)
        * len(RIVALS)
        * OPENINGS
    )
    return {
        "main_rl": main_rl,
        "ablation_rl": ablation_rl,
        "comparators": comparators,
        "total": main_rl + ablation_rl + comparators,
    }


def _base_command(
    config: Mapping[str, Any], *, city: str, case: str,
    specs: Sequence[str], raw: Path, report_prefix: Path,
) -> list[str]:
    validate_config(config)
    inputs = config["city_inputs"][city]
    return [
        str(PYTHON.resolve()), "-u", str(CORE.resolve()),
        "--market", str(inputs["market_name"]),
        "--choice_mode", "cognitive",
        "--firm1_mode", "RL",
        "--firm2_mode", RIVALS[0],
        "--credit_assignment_v25", "--v26",
        "--deterministic_experiment_seed", "--deterministic_torch",
        "--learning_corrections", "--learner_revision", "4",
        "--training_curriculum", "v17_progressive",
        "--current_active_driver_observation",
        "--seed", str(EVAL_SEED),
        "--experiment_seed", str(EVAL_SEED),
        "--policy_objective", "profit_maximization",
        "--pool", str(POOL),
        "--threshold_profile_source", "cached",
        "--threshold_cache_path", str(resolve_input(inputs["threshold_cache"])),
        "--calibration_json", str(resolve_input(inputs["calibration_json"])),
        "--demand_calibration_artifact", str(
            resolve_input(inputs["demand_calibration"])
        ),
        "--gpt_threshold_mode", "llm_numeric",
        "--train_timesteps", "0",
        "--train_customers", str(CUSTOMERS),
        "--eval_customers", str(CUSTOMERS),
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
        "--shock_scenario", str(CASES[case].resolve()),
        "--policy_sweep", ";".join(specs),
        "--policy_sweep_out", str(raw.resolve()),
        "--policy_sweep_openings", str(OPENINGS),
        "--policy_sweep_rivals", ",".join(RIVALS),
        "--policy_sweep_horizon", str(HORIZON_DAYS * STEPS_PER_DAY),
        "--policy_sweep_seed", str(EVAL_SEED),
        "--report_prefix", os.path.relpath(report_prefix, BASE),
    ]


def evaluation_command(
    config: Mapping[str, Any], *, city: str, case: str,
    specs: Sequence[str], raw: Path, report_prefix: Path,
    run: RunSpec | None = None, policy: Path | None = None,
) -> list[str]:
    command = _base_command(
        config, city=city, case=case, specs=specs,
        raw=raw, report_prefix=report_prefix,
    )
    if run is None:
        return command
    if policy is None:
        raise ValueError("RL evaluation requires a policy artifact")
    command.extend(("--trained_model_in", str(policy.resolve()), "--eval_only"))
    if run.kind == "plain_ppo":
        command.extend(("--v16_policy_mode", "ppo"))
    else:
        command.extend((
            "--v16_branch_mode", "one_day_deviation",
            "--v16_branch_horizon_days", "5",
            "--v16_branch_design", "stencil11",
            "--v16_store_cap", "20000",
            "--v16_value_head", "concave2d",
            "--v16_policy_mode", "q",
            "--v17_move_radius", "1",
            "--v17_temperature_mode", "relative",
            "--v17_tau_rel_start", "0.5",
            "--v17_tau_rel_end", "0.15",
        ))
        if run.kind == "no_driver_observation":
            command.append("--v17_ablate_driver_observation")
        elif run.kind == "own_profit":
            command.append("--v17_ablate_profit_lead")
    return command


def validate_eval_product(
    path: Path, command: Sequence[str], *, specs: Sequence[str] | None = None,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_specs = set(
        specs if specs is not None else _option(command, "--policy_sweep").split(";")
    )
    rivals = tuple(_option(command, "--policy_sweep_rivals").split(","))
    openings = int(_option(command, "--policy_sweep_openings"))
    horizon = int(_option(command, "--policy_sweep_horizon"))
    expected_rollout_names = {
        f"{rival}__opening_{opening}"
        for rival in rivals for opening in range(1, openings + 1)
    }
    if payload.get("protocol") != "v26r_policy_sweep_common_random_numbers":
        raise ValueError(f"wrong evaluation protocol: {path}")
    expected_scalars = {
        "city": _option(command, "--market"),
        "horizon": horizon,
        "customers_per_step": int(_option(command, "--eval_customers")),
        "steps_per_day": STEPS_PER_DAY,
        "policy_seed": int(_option(command, "--policy_sweep_seed")),
    }
    for key, expected in expected_scalars.items():
        if payload.get(key) != expected:
            raise ValueError(
                f"evaluation {key} mismatch in {path}: "
                f"{payload.get(key)!r} != {expected!r}"
            )
    v26r = dict(payload.get("v26r", {}) or {})
    if (
        v26r.get("action_contract") != _option(command, "--action_contract")
        or v26r.get("supply_model") != "v3"
        or bool(v26r.get("persistent_driver_stock"))
        or float(v26r.get("effective_fleet_scale", float("nan")))
        != float(_option(command, "--fleet_scale"))
    ):
        raise ValueError(f"evaluation market contract mismatch: {path}")
    declared = {
        str(row[0])
        for row in payload.get("rollouts", ())
        if isinstance(row, (list, tuple)) and len(row) == 2
    }
    if declared != expected_rollout_names:
        raise ValueError(f"evaluation rollout declaration mismatch: {path}")
    results = dict(payload.get("results", {}) or {})
    traces = dict(payload.get("traces", {}) or {})
    if set(results) != expected_specs or set(traces) != expected_specs:
        raise ValueError(f"evaluation policy coverage mismatch: {path}")
    finite_results = (
        "own_profit", "rival_profit", "own_share", "rival_share", "fulfillment"
    )
    finite_traces = (
        "profit_per_request", "rival_profit_per_request",
        "own_share", "rival_share",
    )
    for spec in expected_specs:
        spec_results = dict(results[spec] or {})
        if set(spec_results) != expected_rollout_names:
            raise ValueError(f"evaluation opening coverage mismatch: {path}")
        for rollout, row in spec_results.items():
            if int(row.get("steps", -1)) != horizon:
                raise ValueError(f"evaluation rollout is truncated: {path}/{rollout}")
            if not all(
                math.isfinite(float(row.get(key, float("nan"))))
                for key in finite_results
            ):
                raise ValueError(f"evaluation has non-finite result: {path}/{rollout}")
        counts = {rollout: 0 for rollout in expected_rollout_names}
        for row in list(traces[spec] or ()):
            rollout = str(row.get("rollout", ""))
            if rollout not in counts:
                raise ValueError(f"evaluation trace has unknown rollout: {path}")
            if not str(row.get("regime", "")):
                raise ValueError(f"evaluation trace has no regime: {path}")
            if not all(
                math.isfinite(float(row.get(key, float("nan"))))
                for key in finite_traces
            ):
                raise ValueError(f"evaluation trace is non-finite: {path}")
            counts[rollout] += 1
        if any(count != horizon for count in counts.values()):
            raise ValueError(f"evaluation trace coverage mismatch: {path}")
    return payload


def _split_comparator_product(
    payload: Mapping[str, Any], *, spec: str, output: Path,
) -> None:
    split = dict(payload)
    split["results"] = {spec: payload["results"][spec]}
    split["traces"] = {spec: payload["traces"][spec]}
    atomic_json(output, split)


def _identity(job: EvalJob, config_hash: str, sources: Mapping[str, str]) -> dict[str, Any]:
    command = list(job.command)
    return {
        "command": command,
        "command_sha256": _sha256_json(command),
        "config_hash": config_hash,
        "sources": dict(sources),
        "inputs": dict(job.input_hashes),
        "labels": list(job.labels),
        "specs": list(job.specs),
    }


def run_job(
    job: EvalJob, *, config_hash: str, sources: Mapping[str, str],
) -> str:
    identity = _identity(job, config_hash, sources)
    if job.manifest.is_file():
        recorded = json.loads(job.manifest.read_text(encoding="utf-8"))
        if recorded.get("identity") != identity:
            raise ValueError(f"stale evaluation cache identity: {job.name}")
        if recorded.get("state") != "completed":
            raise ValueError(f"incomplete evaluation cache requires review: {job.name}")
        if _sha256_file(job.raw) != recorded.get("raw_sha256"):
            raise ValueError(f"evaluation raw product hash changed: {job.name}")
        validate_eval_product(job.raw, job.command)
        output_hashes = dict(recorded.get("output_sha256", {}) or {})
        for label, spec, output in zip(job.labels, job.specs, job.outputs):
            if not output.is_file() or _sha256_file(output) != output_hashes.get(label):
                raise ValueError(f"evaluation output hash changed: {job.name}/{label}")
            validate_eval_product(output, job.command, specs=(spec,))
        return "cached"
    existing = [path for path in (job.raw, *job.outputs) if path.is_file()]
    if existing:
        raise ValueError(f"partial evaluation cache requires review: {job.name}")
    for path in (job.raw, job.manifest, job.log, *job.outputs):
        path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(job.manifest, {"state": "running", "identity": identity})
    env = dict(os.environ)
    env.update(THREAD_ENV)
    try:
        proc = subprocess.run(
            list(job.command), cwd=BASE, env=env, capture_output=True, text=True
        )
        job.log.write_text(proc.stdout + proc.stderr, encoding="utf-8")
        if proc.returncode != 0 or not job.raw.is_file():
            raise RuntimeError(
                f"evaluation Core job failed ({proc.returncode}): {job.name}\n"
                + (proc.stdout + proc.stderr)[-8000:]
            )
        payload = validate_eval_product(job.raw, job.command)
        if job.kind == "comparators":
            for spec, output in zip(job.specs, job.outputs):
                _split_comparator_product(payload, spec=spec, output=output)
        for spec, output in zip(job.specs, job.outputs):
            validate_eval_product(output, job.command, specs=(spec,))
    except BaseException as exc:
        atomic_json(job.manifest, {
            "state": "failed", "identity": identity, "error": repr(exc),
        })
        raise
    atomic_json(job.manifest, {
        "state": "completed",
        "identity": identity,
        "raw_sha256": _sha256_file(job.raw),
        "output_sha256": {
            label: _sha256_file(output)
            for label, output in zip(job.labels, job.outputs)
        },
    })
    return "completed"


def _load_selected(path: Path, city: str) -> dict[str, Any]:
    status_path = path / city / "status.json"
    selected_path = path / city / "selected.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("state") not in {"passed", "requires_grid_review"}:
        raise ValueError(f"comparator selection is incomplete for {city}")
    if not status.get("selected_sha256"):
        raise ValueError(f"comparator selection has no recorded hash for {city}")
    if _sha256_file(selected_path) != status["selected_sha256"]:
        raise ValueError(f"comparator selector hash changed for {city}")
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    if selected.get("city") != city:
        raise ValueError(f"comparator selector city mismatch for {city}")
    return selected


def build_jobs(
    config: Mapping[str, Any], *, runs_root: Path,
    comparator_root: Path, output: Path,
) -> tuple[EvalJob, ...]:
    selected = {
        city: _load_selected(comparator_root, city) for city in CITY_SLUGS
    }
    jobs: list[EvalJob] = []
    for run in RUN_SPECS:
        cases = CASES if run.kind == "main" else ABLATION_CASES
        run_dir = runs_root / run.name
        status_path = run_dir / "status.json"
        policy = run_dir / "policy.pt"
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if (
            status.get("state") != "completed"
            or int(status.get("days_expected", -1)) != 1200
            or not policy.is_file()
        ):
            raise ValueError(f"training run is not durably complete: {run.name}")
        for case in cases:
            product = output / run.city / case / f"{run.name}.json"
            raw = product
            manifest = output / "_manifests" / run.city / case / f"{run.name}.json"
            log = output / "_logs" / run.city / case / f"{run.name}.log"
            prefix = output / "_diagnostics" / run.city / case / run.name
            command = evaluation_command(
                config, city=run.city, case=case, specs=("rl_actor",),
                raw=raw, report_prefix=prefix, run=run, policy=policy,
            )
            jobs.append(EvalJob(
                name=f"{run.city}/{case}/{run.name}",
                city=run.city, case=case, kind="rl",
                labels=(run.name,), specs=("rl_actor",), raw=raw,
                outputs=(product,), manifest=manifest, log=log,
                command=tuple(command),
                input_hashes={
                    "policy.pt": _sha256_file(policy),
                    "status.json": _sha256_file(status_path),
                },
            ))
    for city in CITY_SLUGS:
        selected_path = comparator_root / city / "selected.json"
        for case in CASES:
            policy_specs = comparator_specs(selected[city], case)
            labels = tuple(policy_specs)
            specs = tuple(policy_specs.values())
            raw = output / "_combined" / city / case / "comparators.json"
            outputs = tuple(output / city / case / f"{label}.json" for label in labels)
            manifest = output / "_manifests" / city / case / "comparators.json"
            log = output / "_logs" / city / case / "comparators.log"
            prefix = output / "_diagnostics" / city / case / "comparators"
            command = evaluation_command(
                config, city=city, case=case, specs=specs,
                raw=raw, report_prefix=prefix,
            )
            jobs.append(EvalJob(
                name=f"{city}/{case}/comparators",
                city=city, case=case, kind="comparators",
                labels=labels, specs=specs, raw=raw, outputs=outputs,
                manifest=manifest, log=log, command=tuple(command),
                input_hashes={"selected.json": _sha256_file(selected_path)},
            ))
    return tuple(jobs)


def _alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except (OSError, TypeError, ValueError):
        return False
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--runs-root", type=Path, default=BASE / "runs/v17_results")
    parser.add_argument(
        "--comparators", type=Path,
        default=BASE / "runs/v17_results_comparators",
    )
    parser.add_argument(
        "--output", type=Path, default=BASE / "runs/v17_results_eval"
    )
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("workers must be positive")
    config = load_config()
    validate_config(config)
    output = args.output.resolve()
    runs_root = args.runs_root.resolve()
    comparator_root = args.comparators.resolve()
    jobs = build_jobs(
        config, runs_root=runs_root,
        comparator_root=comparator_root, output=output,
    )
    rollouts = expected_rollouts()
    expected_jobs = 7 * len(CASES) + 3 * len(ABLATION_CASES) + 15
    if len(jobs) != expected_jobs or rollouts["total"] != 3030:
        raise ValueError("registered Section 7 Cartesian coverage drifted")
    git_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=BASE, text=True
    ).strip()
    plan = {
        "schema_version": "v17-results-evaluation-status-v1",
        "phase": "held_out_evaluation",
        "attempt_id": "attempt-0001",
        "created_at_utc": _utc_now(),
        "git_head": git_head,
        "config_hash": canonical_hash(config),
        "evaluation_seed": EVAL_SEED,
        "workers": int(args.workers),
        "jobs": len(jobs),
        "rollouts": rollouts,
        "cities": list(CITY_SLUGS),
        "cases": list(CASES),
        "rivals": list(RIVALS),
        "openings": OPENINGS,
        "horizon_days": HORIZON_DAYS,
        "customers_per_step": CUSTOMERS,
    }
    if args.plan_only:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "status.json"
    if status_path.is_file():
        previous = json.loads(status_path.read_text(encoding="utf-8"))
        if previous.get("state") == "running" and _alive(previous.get("pid", -1)):
            raise SystemExit(
                f"evaluation is already running as pid {previous['pid']}"
            )
        if previous.get("state") == "completed":
            print("[evaluation] already completed; validating cached products")
    sources = source_identity()
    completed = 0
    cached = 0
    atomic_json(status_path, {
        **plan, "state": "running", "pass": False,
        "pid": os.getpid(), "completed_jobs": 0, "cached_jobs": 0,
    })
    try:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(int(args.workers), len(jobs))
        ) as executor:
            futures = {
                executor.submit(
                    run_job, job,
                    config_hash=plan["config_hash"], sources=sources,
                ): job
                for job in jobs
            }
            for future in concurrent.futures.as_completed(futures):
                job = futures[future]
                result = future.result()
                completed += 1
                cached += int(result == "cached")
                print(
                    f"[evaluation] {job.name}: {result} "
                    f"({completed}/{len(jobs)})",
                    flush=True,
                )
                atomic_json(status_path, {
                    **plan, "state": "running", "pass": False,
                    "pid": os.getpid(), "completed_jobs": completed,
                    "cached_jobs": cached, "last_completed": job.name,
                })
        manifest_hashes = {
            str(job.manifest.relative_to(output)): _sha256_file(job.manifest)
            for job in jobs
        }
        atomic_json(output / "manifest.json", {
            **plan,
            "state": "completed",
            "source_identity": sources,
            "job_manifests": manifest_hashes,
        })
        atomic_json(status_path, {
            **plan, "state": "completed", "pass": True,
            "pid": os.getpid(), "completed_jobs": completed,
            "cached_jobs": cached,
            "manifest": str(output / "manifest.json"),
            "manifest_sha256": _sha256_file(output / "manifest.json"),
        })
        return 0
    except BaseException as exc:
        atomic_json(status_path, {
            **plan, "state": "failed", "pass": False,
            "pid": os.getpid(), "completed_jobs": completed,
            "cached_jobs": cached, "error": repr(exc),
        })
        raise


if __name__ == "__main__":
    raise SystemExit(main())
