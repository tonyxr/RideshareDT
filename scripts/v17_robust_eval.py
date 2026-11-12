#!/usr/bin/env python3
"""Content-addressed evaluation runner for CRISP-Ride V17 robustness suites."""

from __future__ import annotations

import argparse
import concurrent.futures
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence


BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from diagnostic_logging import atomic_json  # noqa: E402
from scripts import v17_results_eval as registered  # noqa: E402
from scripts.v17_config import (  # noqa: E402
    canonical_hash,
    check_command_model,
    load_config,
    validate_config,
)
from scripts.v17_results_train import RunSpec  # noqa: E402
from scripts.v17_results_train import RUN_SPECS  # noqa: E402
from scripts.v17_t2_sweep import cell_spec  # noqa: E402
from scripts.v17_robust_select import LEFTOUT_VARIANTS  # noqa: E402
from scripts.v17_robust_common import (  # noqa: E402
    check_overrides,
    robust_sources,
    schedule_spec,
    set_option,
)
from v17_surge_decay import scenario_windows, surge_decay_spec  # noqa: E402


ROBUST_ROOT = BASE / "runs/v17_robust"
EVAL_ROOT = ROBUST_ROOT / "eval"
COMPARATOR_ROOT = BASE / "runs/v17_results_comparators"
RUNS_ROOT = BASE / "runs/v17_results"
WP1_SELECTED = ROBUST_ROOT / "select/wp1"
POINT_SELECTED = ROBUST_ROOT / "select/point"
RULE_SELECTED = ROBUST_ROOT / "select/rule"
LEFTOUT_SELECTED = ROBUST_ROOT / "select/leftout"
LADDER_CASES = {
    "L080_shortage_mild": BASE / "scenarios/cases/v17_shortage_i080.json",
    "L065_shortage_moderate": BASE / "scenarios/cases/v17_shortage_i065.json",
    "L050_shortage_strong": BASE / "scenarios/cases/v17_shortage_i050.json",
}
OPTIONAL_LADDER_CASES = {
    "L035_d3": BASE / "scenarios/cases/v17_shortage_i035_d3.json",
    "L035_d10": BASE / "scenarios/cases/v17_shortage_i035_d10.json",
}
POINT_VALUES = tuple(
    (eta, omega)
    for eta in (0.3, 0.6, 1.0)
    for omega in (0.25, 0.5, 0.75)
)
RETALIATION_LEVELS = {
    "off": {"memory_rival_retaliation": "off"},
    "hold2.5": {"memory_rival_hold_mean_days": 2.5},
    "hold5": {"memory_rival_hold_mean_days": 5.0},
    "hold10": {"memory_rival_hold_mean_days": 10.0},
}
METRICS = ("own_profit", "rival_profit", "own_share", "rival_share", "fulfillment")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    return registered._sha256_file(path)


def _case_command_name(case: str) -> str:
    return case if case in registered.CASES else "A1_shortage_severe"


def make_job(
    config: Mapping[str, Any],
    *,
    suite: str,
    point: str,
    city: str,
    case: str,
    scenario: Path,
    labels: Sequence[str],
    specs: Sequence[str],
    output_root: Path,
    overrides: Mapping[str, Any] | None = None,
    run: RunSpec | None = None,
    policy: Path | None = None,
    rivals: Sequence[str] | None = None,
    openings: int | None = None,
    input_hashes: Mapping[str, str] | None = None,
) -> registered.EvalJob:
    """Build one registered evaluator job with robustness-only overrides."""
    labels = tuple(str(label) for label in labels)
    specs = tuple(str(spec) for spec in specs)
    if len(labels) != len(specs) or not labels:
        raise ValueError("labels and specs must be non-empty and aligned")
    overrides = dict(overrides or {})
    output_root = output_root.resolve()
    kind = "rl" if run is not None else "comparators"
    leaf = output_root / point / city / case
    if kind == "rl":
        if len(labels) != 1 or specs != ("rl_actor",):
            raise ValueError("RL jobs require one label and the rl_actor spec")
        raw = leaf / f"{labels[0]}.json"
        outputs = (raw,)
        manifest_name = labels[0]
    else:
        raw = output_root / "_combined" / point / city / case / "comparators.json"
        outputs = tuple(leaf / f"{label}.json" for label in labels)
        manifest_name = "comparators"
    manifest = (
        output_root / "_manifests" / point / city / case / f"{manifest_name}.json"
    )
    log = output_root / "_logs" / point / city / case / f"{manifest_name}.log"
    prefix = output_root / "_diagnostics" / point / city / case / manifest_name
    command = registered.evaluation_command(
        config,
        city=city,
        case=_case_command_name(case),
        specs=specs,
        raw=raw,
        report_prefix=prefix,
        run=run,
        policy=policy,
    )
    command = set_option(command, "--shock_scenario", str(scenario.resolve()))
    if rivals is not None:
        command = set_option(
            command, "--policy_sweep_rivals", ",".join(str(v) for v in rivals)
        )
        if tuple(rivals) == ("static",):
            command = set_option(command, "--firm2_mode", "static")
    if openings is not None:
        command = set_option(command, "--policy_sweep_openings", int(openings))
    flag_map = {
        "supply_eta_v3": "--supply_eta_v3",
        "shortage_willingness_share": "--shortage_willingness_share",
        "memory_rival_hold_mean_days": "--memory_rival_hold_mean_days",
        "memory_rival_retaliation": "--memory_rival_retaliation",
        "loyalty_model": "--loyalty_model",
        "threshold_ablation": "--threshold_ablation",
    }
    for name, value in overrides.items():
        key = str(name).lstrip("-").replace("-", "_")
        if key == "disable_spillover":
            command = [item for item in command if item != "--disable_spillover"]
            if bool(value):
                command.append("--disable_spillover")
            continue
        if key not in flag_map:
            raise ValueError(f"unknown evaluation override {name!r}")
        command = set_option(command, flag_map[key], value)
    hashes = {"scenario": _sha256_file(scenario.resolve())}
    hashes.update(dict(input_hashes or {}))
    return registered.EvalJob(
        name=f"{suite}/{point}/{city}/{case}/{manifest_name}",
        city=city,
        case=case,
        kind=kind,
        labels=labels,
        specs=specs,
        raw=raw,
        outputs=outputs,
        manifest=manifest,
        log=log,
        command=tuple(command),
        input_hashes=hashes,
    )


def run_robust_job(
    job: registered.EvalJob,
    *,
    config_hash: str,
    sources: Mapping[str, str],
    overrides: Mapping[str, Any] | None = None,
    model_expected: Mapping[str, Any] | None = None,
) -> str:
    result = registered.run_job(
        job, config_hash=config_hash, sources=sources,
        model_expected=model_expected,
    )
    for output in job.outputs:
        payload = json.loads(output.read_text(encoding="utf-8"))
        check_overrides(payload, dict(overrides or {}))
    return result


def execute_jobs(
    jobs: Sequence[registered.EvalJob],
    *,
    suite: str,
    output: Path,
    workers: int,
    config: Mapping[str, Any],
    sources: Mapping[str, str],
    overrides_by_name: Mapping[str, Mapping[str, Any]] | None = None,
    plan_only: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run, validate, and summarize a complete robustness suite."""
    output = output.resolve()
    overrides_by_name = dict(overrides_by_name or {})
    plan = {
        "schema_version": "v17-robust-evaluation-status-v1",
        "suite": str(suite),
        "created_at_utc": _utc_now(),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=BASE, text=True
        ).strip(),
        "config_hash": canonical_hash(config),
        "workers": int(workers),
        "jobs": len(jobs),
        "evaluation_seed": registered.EVAL_SEED,
        "horizon_days": registered.HORIZON_DAYS,
        "customers_per_step": registered.CUSTOMERS,
        **dict(metadata or {}),
    }
    if plan_only:
        print(json.dumps({
            **plan,
            "job_names": [job.name for job in jobs],
        }, indent=2, sort_keys=True))
        return plan
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "status.json"
    atomic_json(status_path, {
        **plan, "state": "running", "pass": False,
        "pid": os.getpid(), "completed_jobs": 0, "cached_jobs": 0,
    })
    completed = 0
    cached = 0
    try:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(max(1, int(workers)), max(1, len(jobs)))
        ) as executor:
            futures = {
                executor.submit(
                    run_robust_job,
                    job,
                    config_hash=plan["config_hash"],
                    sources=sources,
                    overrides=overrides_by_name.get(job.name, {}),
                ): job
                for job in jobs
            }
            for future in concurrent.futures.as_completed(futures):
                job = futures[future]
                result = future.result()
                completed += 1
                cached += int(result == "cached")
                print(
                    f"[{suite}] {job.name}: {result} "
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
        manifest_path = output / "manifest.json"
        atomic_json(manifest_path, {
            **plan,
            "state": "completed",
            "source_identity": dict(sources),
            "job_manifests": manifest_hashes,
        })
        final = {
            **plan, "state": "completed", "pass": True,
            "pid": os.getpid(), "completed_jobs": completed,
            "cached_jobs": cached, "manifest": str(manifest_path),
            "manifest_sha256": _sha256_file(manifest_path),
        }
        atomic_json(status_path, final)
        return final
    except BaseException as exc:
        atomic_json(status_path, {
            **plan, "state": "failed", "pass": False,
            "pid": os.getpid(), "completed_jobs": completed,
            "cached_jobs": cached, "error": repr(exc),
        })
        raise


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _selected(city: str) -> tuple[Path, dict[str, Any]]:
    path = COMPARATOR_ROOT / city / "selected.json"
    return path, _load(path)


def _surge_selected(city: str) -> tuple[Path, dict[str, Any]]:
    path = WP1_SELECTED / city / "selected_surge.json"
    return path, _load(path)


def _scenario(path: Path) -> dict[str, Any]:
    return _load(path)


def _point_id(eta: float, omega: float) -> str:
    return f"eta{eta:g}_omega{omega:g}"


def _parse_points(raw: str) -> tuple[tuple[float, float], ...]:
    if raw == "all":
        return POINT_VALUES
    result: list[tuple[float, float]] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token.startswith("eta") or "_omega" not in token:
            raise ValueError(f"bad point id: {token}")
        eta, omega = token[3:].split("_omega", 1)
        result.append((float(eta), float(omega)))
    return tuple(result)


def _parse_cities(raw: str, default: Sequence[str]) -> tuple[str, ...]:
    cities = tuple(token.strip() for token in str(raw).split(",") if token.strip())
    if not cities:
        cities = tuple(default)
    unknown = sorted(set(cities) - {"nyc", "boston", "chicago"})
    if unknown:
        raise ValueError(f"unknown city: {', '.join(unknown)}")
    return cities


def _main_runs(city: str) -> tuple[RunSpec, ...]:
    return tuple(run for run in RUN_SPECS if run.city == city and run.kind == "main")


def _require_run(
    run: RunSpec, *, days: int | None = 1200, root: Path = RUNS_ROOT,
) -> tuple[Path, Path, dict[str, Any]]:
    run_dir = root / run.name
    status_path = run_dir / "status.json"
    policy = run_dir / "policy.pt"
    status = _load(status_path)
    check_command_model(status.get("command"), load_config())
    if status.get("state") != "completed" or not policy.is_file():
        raise ValueError(f"training run is not complete: {run.name}")
    if days is not None and int(status.get("days_expected", -1)) != int(days):
        raise ValueError(f"training day count mismatch: {run.name}")
    return status_path, policy, status


def _rebuild_surge(row: Mapping[str, Any], scenario: Path) -> str:
    payload = _scenario(scenario)
    return surge_decay_spec(
        row.get("normal", row.get("normal_cell")),
        row["event"], scenario_windows(payload),
        int(row.get("hold_days", 0)), int(row.get("decay_days", 0)),
    )


def _selection_surge(row: Mapping[str, Any], selected: Mapping[str, Any], scenario: Path) -> str:
    return surge_decay_spec(
        selected["normal_cell"], row["event"], scenario_windows(_scenario(scenario)),
        int(row.get("hold_days", 0)), int(row.get("decay_days", 0)),
    )


def _identity_diff(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> dict[str, Any]:
    left = dict(actual.get("results", {}) or {})
    right = dict(expected.get("results", {}) or {})
    if len(left) != 1 or len(right) != 1:
        raise ValueError("identity product must contain exactly one policy")
    left_rows = dict(next(iter(left.values())) or {})
    right_rows = dict(next(iter(right.values())) or {})
    if set(left_rows) != set(right_rows):
        raise ValueError("identity rollout keys differ")
    maximum = 0.0
    differences = []
    for rollout in sorted(left_rows):
        for metric in METRICS:
            delta = abs(float(left_rows[rollout][metric]) - float(right_rows[rollout][metric]))
            maximum = max(maximum, delta)
            if delta > 1e-12:
                differences.append((rollout, metric, delta))
    return {"pass": not differences, "maximum_absolute_difference": maximum, "differences": differences[:50]}


def _build_wp1(config: Mapping[str, Any], output: Path) -> tuple[list[registered.EvalJob], dict[str, dict[str, Any]], list[Path]]:
    jobs: list[registered.EvalJob] = []
    paths: list[Path] = []
    for city in ("nyc", "boston", "chicago"):
        selected_path, selected = _surge_selected(city)
        paths.append(selected_path)
        for case, scenario in registered.CASES.items():
            labels = ("surge_two_lever", "surge_decay")
            specs = tuple(_selection_surge(selected[label], selected, scenario) for label in labels)
            jobs.append(make_job(
                config, suite="wp1", point="nominal", city=city, case=case,
                scenario=scenario, labels=labels, specs=specs, output_root=output,
                input_hashes={"selected_surge.json": _sha256_file(selected_path)},
            ))
    return jobs, {}, paths


def _build_wp3(
    config: Mapping[str, Any], output: Path, *, optional: bool,
) -> tuple[list[registered.EvalJob], dict[str, dict[str, Any]], list[Path]]:
    cases = dict(LADDER_CASES)
    if optional:
        cases.update(OPTIONAL_LADDER_CASES)
    jobs: list[registered.EvalJob] = []
    paths: list[Path] = list(cases.values())
    for run in (row for row in RUN_SPECS if row.kind == "main"):
        status_path, policy, _ = _require_run(run)
        for case, scenario in cases.items():
            jobs.append(make_job(
                config, suite="wp3", point="nominal", city=run.city, case=case,
                scenario=scenario, labels=(run.name,), specs=("rl_actor",),
                output_root=output, run=run, policy=policy,
                input_hashes={"policy.pt": _sha256_file(policy), "status.json": _sha256_file(status_path)},
            ))
    for city in ("nyc", "boston", "chicago"):
        comparator_path, selected = _selected(city)
        surge_path, surge = _surge_selected(city)
        paths.extend((comparator_path, surge_path))
        for case, scenario in cases.items():
            scenario_payload = _scenario(scenario)
            labels = (
                "published_card", "best_fixed_cell", "own_profit_best_fixed_cell",
                "surge_rule", "surge_two_lever", "surge_decay",
            )
            specs = (
                cell_spec(1.0, 1.0),
                str(selected["best_fixed_cell"]["spec"]),
                str(selected["own_profit_best_fixed_cell"]["spec"]),
                schedule_spec(float(selected["surge_rule"]["event_level"]), scenario_payload),
                _selection_surge(surge["surge_two_lever"], surge, scenario),
                _selection_surge(surge["surge_decay"], surge, scenario),
            )
            jobs.append(make_job(
                config, suite="wp3", point="nominal", city=city, case=case,
                scenario=scenario, labels=labels, specs=specs, output_root=output,
                input_hashes={
                    "selected.json": _sha256_file(comparator_path),
                    "selected_surge.json": _sha256_file(surge_path),
                },
            ))
    return jobs, {}, paths


def _build_wp2a(
    config: Mapping[str, Any], output: Path, *, cities: Sequence[str], points: str,
) -> tuple[list[registered.EvalJob], dict[str, dict[str, Any]], list[Path]]:
    jobs: list[registered.EvalJob] = []
    overrides_by_name: dict[str, dict[str, Any]] = {}
    paths: list[Path] = []
    scenario = registered.CASES["A1_shortage_severe"]
    for city in cities:
        comparator_path, selected = _selected(city)
        paths.append(comparator_path)
        for eta, omega in _parse_points(points):
            point = _point_id(eta, omega)
            point_path = POINT_SELECTED / city / point / "selected_point.json"
            point_selected = _load(point_path)
            paths.append(point_path)
            overrides = {"supply_eta_v3": eta, "shortage_willingness_share": omega}
            labels = ("published_card", "best_fixed_nominal", "best_fixed_point", "surge_decay_point")
            specs = (
                cell_spec(1.0, 1.0),
                str(selected["best_fixed_cell"]["spec"]),
                str(point_selected["best_fixed_point"]["spec"]),
                str(point_selected["surge_decay_point"]["spec"]),
            )
            comparator = make_job(
                config, suite="wp2a", point=point, city=city, case="A1_shortage_severe",
                scenario=scenario, labels=labels, specs=specs, output_root=output,
                overrides=overrides,
                input_hashes={
                    "selected.json": _sha256_file(comparator_path),
                    "selected_point.json": _sha256_file(point_path),
                },
            )
            jobs.append(comparator)
            overrides_by_name[comparator.name] = overrides
            for run in _main_runs(city):
                status_path, policy, _ = _require_run(run)
                job = make_job(
                    config, suite="wp2a", point=point, city=city, case="A1_shortage_severe",
                    scenario=scenario, labels=(run.name,), specs=("rl_actor",),
                    output_root=output, overrides=overrides, run=run, policy=policy,
                    input_hashes={"policy.pt": _sha256_file(policy), "status.json": _sha256_file(status_path)},
                )
                jobs.append(job)
                overrides_by_name[job.name] = overrides
    return jobs, overrides_by_name, paths


def _build_wp2b(
    config: Mapping[str, Any], output: Path,
) -> tuple[list[registered.EvalJob], dict[str, dict[str, Any]], list[Path]]:
    jobs: list[registered.EvalJob] = []
    overrides_by_name: dict[str, dict[str, Any]] = {}
    paths: list[Path] = []
    rival = ("adaptive_best_response_memory",)
    for city in ("nyc", "chicago"):
        comparator_path, selected = _selected(city)
        surge_path, surge = _surge_selected(city)
        paths.extend((comparator_path, surge_path))
        for level, overrides in RETALIATION_LEVELS.items():
            for case in ("C0_calm", "A1_shortage_severe"):
                scenario = registered.CASES[case]
                labels = ("published_card", "best_fixed_cell", "surge_decay")
                specs = (
                    cell_spec(1.0, 1.0), str(selected["best_fixed_cell"]["spec"]),
                    _selection_surge(surge["surge_decay"], surge, scenario),
                )
                comparator = make_job(
                    config, suite="wp2b", point=level, city=city, case=case,
                    scenario=scenario, labels=labels, specs=specs, output_root=output,
                    overrides=overrides, rivals=rival, openings=10,
                    input_hashes={"selected.json": _sha256_file(comparator_path), "selected_surge.json": _sha256_file(surge_path)},
                )
                jobs.append(comparator)
                overrides_by_name[comparator.name] = dict(overrides)
                for run in _main_runs(city):
                    status_path, policy, _ = _require_run(run)
                    job = make_job(
                        config, suite="wp2b", point=level, city=city, case=case,
                        scenario=scenario, labels=(run.name,), specs=("rl_actor",),
                        output_root=output, overrides=overrides, run=run, policy=policy,
                        rivals=rival, openings=10,
                        input_hashes={"policy.pt": _sha256_file(policy), "status.json": _sha256_file(status_path)},
                    )
                    jobs.append(job)
                    overrides_by_name[job.name] = dict(overrides)
    return jobs, overrides_by_name, paths


def _build_wp4(
    config: Mapping[str, Any], output: Path,
) -> tuple[list[registered.EvalJob], dict[str, dict[str, Any]], list[Path]]:
    jobs: list[registered.EvalJob] = []
    matched_path = ROBUST_ROOT / "wp4/matched_days.json"
    matched = int(_load(matched_path)["matched_days"])
    paths = [matched_path]
    runs = (
        (RunSpec("nyc_s101_ppo_tuned", "nyc", 101, "plain_ppo"), 1200, tuple(registered.CASES)),
        (RunSpec("nyc_s101_ppo_matched", "nyc", 101, "plain_ppo"), matched, tuple(registered.CASES)),
        (RunSpec("nyc_s101_plain_ppo", "nyc", 101, "plain_ppo"), 1200, ("A3_storm_transit_late", "S1_promotion_bait", "S6_shortage_during_promotion")),
    )
    for run, days, cases in runs:
        status_path, policy, _ = _require_run(run, days=days)
        paths.extend((status_path, policy))
        for case in cases:
            jobs.append(make_job(
                config, suite="wp4", point="nominal", city="nyc", case=case,
                scenario=registered.CASES[case], labels=(run.name,), specs=("rl_actor",),
                output_root=output, run=run, policy=policy,
                input_hashes={"policy.pt": _sha256_file(policy), "status.json": _sha256_file(status_path)},
            ))
    return jobs, {}, paths


def _build_abl(
    config: Mapping[str, Any], output: Path,
) -> tuple[list[registered.EvalJob], dict[str, dict[str, Any]], list[Path]]:
    """Evaluate both registered ablations on a second seed and all five cases."""
    jobs: list[registered.EvalJob] = []
    paths: list[Path] = []
    runs = (
        (
            RunSpec("nyc_s101_no_driver_obs", "nyc", 101, "no_driver_observation"),
            ("A3_storm_transit_late", "S1_promotion_bait", "S6_shortage_during_promotion"),
        ),
        (
            RunSpec("nyc_s101_own_profit", "nyc", 101, "own_profit"),
            ("A3_storm_transit_late", "S1_promotion_bait", "S6_shortage_during_promotion"),
        ),
        (
            RunSpec("nyc_s202_no_driver_obs", "nyc", 202, "no_driver_observation"),
            tuple(registered.CASES),
        ),
        (
            RunSpec("nyc_s202_own_profit", "nyc", 202, "own_profit"),
            tuple(registered.CASES),
        ),
    )
    for run, cases in runs:
        status_path, policy, _ = _require_run(run)
        paths.extend((status_path, policy))
        for case in cases:
            jobs.append(make_job(
                config, suite="abl", point="nominal", city="nyc", case=case,
                scenario=registered.CASES[case], labels=(run.name,), specs=("rl_actor",),
                output_root=output, run=run, policy=policy,
                input_hashes={
                    "policy.pt": _sha256_file(policy),
                    "status.json": _sha256_file(status_path),
                },
            ))
    return jobs, {}, paths


def _build_rule(
    config: Mapping[str, Any], output: Path,
) -> tuple[list[registered.EvalJob], dict[str, dict[str, Any]], list[Path]]:
    """Evaluate each city's selected live-signal rule on all registered cases."""
    jobs: list[registered.EvalJob] = []
    paths: list[Path] = []
    for city in ("nyc", "boston", "chicago"):
        selected_path = RULE_SELECTED / city / "selected_rule.json"
        selected = _load(selected_path)
        spec = str(selected["reactive_surge"]["spec"])
        if not spec.startswith("reactive_surge:"):
            raise ValueError(f"wrong reactive rule selection for {city}: {spec}")
        paths.append(selected_path)
        for case, scenario in registered.CASES.items():
            jobs.append(make_job(
                config, suite="rule", point="nominal", city=city, case=case,
                scenario=scenario, labels=("reactive_surge",), specs=(spec,),
                output_root=output,
                input_hashes={"selected_rule.json": _sha256_file(selected_path)},
            ))
    return jobs, {}, paths


def _leftout_selected(city: str, variant: str) -> tuple[Path, dict[str, Any]]:
    path = LEFTOUT_SELECTED / city / variant / "selected_leftout.json"
    return path, _load(path)


def _build_leftout(
    config: Mapping[str, Any], output: Path,
    *, cities: Sequence[str] = ("nyc", "boston", "chicago"),
) -> tuple[list[registered.EvalJob], dict[str, dict[str, Any]], list[Path]]:
    """Evaluate partial-model selected fares back in the full market."""
    jobs: list[registered.EvalJob] = []
    paths: list[Path] = []
    for city in cities:
        full_path = LEFTOUT_SELECTED / city / "full" / "selected_leftout.json"
        available = [
            variant for variant in LEFTOUT_VARIANTS
            if (LEFTOUT_SELECTED / city / variant / "selected_leftout.json").is_file()
        ]
        if not full_path.is_file() or not available:
            continue
        selections: list[tuple[str, Path, dict[str, Any]]] = [
            ("full", full_path, _load(full_path))
        ]
        selections.extend(
            (variant, *_leftout_selected(city, variant))
            for variant in available
        )
        paths.extend(path for _variant, path, _selected in selections)
        labels: list[str] = []
        for variant, _path, _selected in selections:
            labels.extend((f"fixed@{variant}", f"dated@{variant}"))
        for case, scenario in registered.CASES.items():
            specs: list[str] = []
            for _variant, _path, selected in selections:
                normal = selected["best_fixed_cell"]["cell"]
                event = selected["best_shortage_cell"]["cell"]
                specs.extend((
                    str(selected["best_fixed_cell"]["spec"]),
                    surge_decay_spec(
                        normal, event, scenario_windows(_scenario(scenario)), 0, 0,
                    ),
                ))
            jobs.append(make_job(
                config,
                suite="leftout",
                point="nominal",
                city=city,
                case=case,
                scenario=scenario,
                labels=tuple(labels),
                specs=tuple(specs),
                output_root=output,
                input_hashes={
                    f"selected_leftout_{variant}.json": _sha256_file(path)
                    for variant, path, _selected in selections
                },
            ))
    return jobs, {}, paths


def _build_leftout_market(
    config: Mapping[str, Any], output: Path, *, cities: Sequence[str],
) -> tuple[list[registered.EvalJob], dict[str, dict[str, Any]], list[Path]]:
    """Evaluate the published card under each one-piece-omitted market."""
    jobs: list[registered.EvalJob] = []
    overrides_by_name: dict[str, dict[str, Any]] = {}
    paths: list[Path] = []
    for city in cities:
        for variant, settings in LEFTOUT_VARIANTS.items():
            overrides = dict(settings["overrides"])
            rivals = tuple(settings["rivals"])
            for case, scenario in registered.CASES.items():
                job = make_job(
                    config,
                    suite="leftout_market",
                    point=variant,
                    city=city,
                    case=case,
                    scenario=scenario,
                    labels=("published_card",),
                    specs=(cell_spec(1.0, 1.0),),
                    output_root=output,
                    overrides=overrides,
                    rivals=rivals,
                )
                jobs.append(job)
                overrides_by_name[job.name] = overrides
    return jobs, overrides_by_name, paths


def _built_in_checks(suite: str, jobs: Sequence[registered.EvalJob], output: Path) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    if suite == "wp1":
        for job in jobs:
            if job.case not in {"C0_calm", "S1_promotion_bait"}:
                continue
            expected = _load(BASE / "runs/v17_results_eval" / job.city / job.case / "best_fixed_cell.json")
            for product in job.outputs:
                result = _identity_diff(_load(product), expected)
                result["product"] = str(product)
                checks.append(result)
    elif suite == "wp2a":
        for job in jobs:
            if job.kind != "rl" or "/eta0.6_omega0.5/" not in f"/{job.name}/":
                continue
            expected = _load(BASE / "runs/v17_results_eval" / job.city / job.case / job.outputs[0].name)
            result = _identity_diff(_load(job.outputs[0]), expected)
            result["product"] = str(job.outputs[0])
            checks.append(result)
    elif suite == "wp2b":
        for job in jobs:
            if not job.name.startswith("wp2b/off/"):
                continue
            payload = _load(job.outputs[0])
            retaliation_rows = [
                row for rows in dict(payload.get("traces", {})).values()
                for row in rows if str(row.get("rival_memory_state", "")) == "retaliation"
            ]
            checks.append({"pass": not retaliation_rows, "product": str(job.outputs[0]), "retaliation_rows": len(retaliation_rows)})
    failures = [row for row in checks if not row.get("pass")]
    if failures:
        raise ValueError(f"{suite} built-in identity check failed: {failures[:5]}")
    return checks


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", required=True,
        choices=(
            "wp1", "wp2a", "wp2b", "wp3", "wp4", "abl", "rule",
            "leftout", "leftout_market",
        ),
    )
    parser.add_argument("--city", default="")
    parser.add_argument("--points", default="all")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--optional-duration-cases", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.workers > 8:
        parser.error("--workers must not exceed 8 while PPO training may be active")
    config = load_config()
    validate_config(config)
    output = EVAL_ROOT / args.suite
    if args.suite == "wp1":
        jobs, overrides, paths = _build_wp1(config, output)
    elif args.suite == "wp3":
        jobs, overrides, paths = _build_wp3(config, output, optional=args.optional_duration_cases)
    elif args.suite == "wp2a":
        cities = _parse_cities(args.city, ("nyc",))
        jobs, overrides, paths = _build_wp2a(config, output, cities=cities, points=args.points)
    elif args.suite == "wp2b":
        jobs, overrides, paths = _build_wp2b(config, output)
    elif args.suite == "wp4":
        jobs, overrides, paths = _build_wp4(config, output)
    elif args.suite == "abl":
        jobs, overrides, paths = _build_abl(config, output)
    elif args.suite == "rule":
        jobs, overrides, paths = _build_rule(config, output)
    elif args.suite == "leftout":
        cities = _parse_cities(args.city, ("nyc", "boston", "chicago"))
        jobs, overrides, paths = _build_leftout(config, output, cities=cities)
        if not jobs:
            raise ValueError("leftout suite has no completed city selections")
    elif args.suite == "leftout_market":
        cities = _parse_cities(args.city, ("nyc", "chicago"))
        jobs, overrides, paths = _build_leftout_market(
            config, output, cities=cities,
        )
    else:
        raise AssertionError(args.suite)
    sources = robust_sources(tuple(dict.fromkeys(path.resolve() for path in paths if path.is_file())))
    metadata = {
        "points": sorted({job.name.split("/")[1] for job in jobs}),
        "cities": sorted({job.city for job in jobs}),
        "cases": sorted({job.case for job in jobs}),
    }
    if args.plan_only:
        execute_jobs(
            jobs, suite=args.suite, output=output, workers=args.workers,
            config=config, sources=sources, overrides_by_name=overrides,
            plan_only=True, metadata=metadata,
        )
        return 0
    try:
        result = execute_jobs(
            jobs, suite=args.suite, output=output, workers=args.workers,
            config=config, sources=sources, overrides_by_name=overrides,
            metadata=metadata,
        )
        checks = _built_in_checks(args.suite, jobs, output)
        result["built_in_checks"] = checks
        atomic_json(output / "status.json", result)
        return 0
    except BaseException as exc:
        current = {}
        try:
            current = _load(output / "status.json")
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        atomic_json(output / "status.json", {
            **current, "state": "failed", "pass": False,
            "error": repr(exc), "failed_at_utc": _utc_now(),
        })
        raise


if __name__ == "__main__":
    raise SystemExit(main())
