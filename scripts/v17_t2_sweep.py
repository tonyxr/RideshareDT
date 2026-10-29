#!/usr/bin/env python3
"""Run the RevB V17 T2 adaptation-value sweep on production market wiring.

The sweep is content-addressed and resumable at the case/competitor boundary.
Each worker evaluates every fixed distance/time cell on common-random-number
validation openings. Analysis uses the registered final objective, retains
both component profits and completed shares, reports opening-level standard
errors, and evaluates each regime table on an opening excluded from its fit.
"""

from __future__ import annotations

import argparse
import concurrent.futures
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence


BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from diagnostic_logging import atomic_json  # noqa: E402
from scripts.v17_config import (  # noqa: E402
    ACTION_LEVELS,
    DEFAULT_CONFIG,
    canonical_hash,
    load_config,
    resolve_input,
    validate_config,
)
from v17_objective import OBJECTIVE_NAME, profit_report  # noqa: E402


PYTHON = ROOT / ".venv/bin/python"
STEPS_PER_DAY = 6
HORIZON_DAYS = 40
ONSET_DAY = 10
OPENINGS = 5
VALIDATION_SEED = 20_231
PRIOR_T2_ROLLOUTS = 1350
PRIOR_T2_WORKERS = 4
PRIOR_T2_WALL_HOURS = 6.0
RIVALS = (
    "adaptive_best_response_memory",
    "profit_opt_pi_price_gap",
    "profit_opt_mpc_grid",
    "profit_opt_region_supply_demand",
    "profit_opt_surge_driver_incentive",
    "profit_opt_queue_service_threshold",
    "profit_best_response_myopic",
    "profit_best_response_forward",
    "profit_best_response_anticipating",
)
CASES = {
    "C0_calm": BASE / "scenarios/calm.json",
    "A1_shortage_severe": BASE / "scenarios/v17_t2_shortage_severe.json",
}


def _sha256_json(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_identity() -> dict[str, str]:
    paths = (
        BASE / "Core.py",
        BASE / "driver_supply.py",
        BASE / "market_shocks.py",
        BASE / "platform_mdp.py",
        BASE / "pricing_models.py",
        BASE / "v17_objective.py",
        BASE / "scripts/v17_t2_sweep.py",
        *CASES.values(),
    )
    return {str(path.relative_to(BASE)): _sha256_file(path) for path in paths}


def cell_spec(distance: float, time: float) -> str:
    table = {
        "airport_fee": 1.0,
        "base_fare": 1.0,
        "booking_fee": 1.0,
        "per_mile": float(distance),
        "per_minute": float(time),
    }
    return "static_tariff:" + json.dumps(
        table, sort_keys=True, separators=(",", ":")
    )


def parse_cell(spec: str) -> tuple[float, float]:
    if not str(spec).startswith("static_tariff:"):
        raise ValueError(f"not a T2 fixed-cell spec: {spec}")
    table = json.loads(str(spec).split(":", 1)[1])
    return float(table["per_mile"]), float(table["per_minute"])


def fixed_specs(config: Mapping[str, Any]) -> tuple[str, ...]:
    contract = str(config["action"]["contract"])
    levels = ACTION_LEVELS[contract]
    return tuple(cell_spec(distance, time) for distance in levels for time in levels)


def expected_rollouts(config: Mapping[str, Any], openings: int = OPENINGS) -> int:
    return len(fixed_specs(config)) * len(CASES) * len(RIVALS) * int(openings)


def estimated_wall_hours(
    config: Mapping[str, Any], *, openings: int, workers: int
) -> float:
    """Scale the completed V1 T2 wall time by rollouts and worker count."""
    return (
        PRIOR_T2_WALL_HOURS
        * expected_rollouts(config, openings) / PRIOR_T2_ROLLOUTS
        * PRIOR_T2_WORKERS / max(1, int(workers))
    )


def sweep_command(
    config: Mapping[str, Any],
    *,
    case: str,
    rival: str,
    output: Path,
    openings: int,
    customers: int,
    pool: int,
) -> list[str]:
    """Render one isolated case/competitor Core sweep."""
    validate_config(config)
    if case not in CASES:
        raise ValueError(f"unknown T2 case: {case}")
    if rival not in RIVALS:
        raise ValueError(f"unknown T2 rival: {rival}")
    inputs = config["city_inputs"]["nyc"]
    raw = output / "raw" / case / f"{rival}.json"
    prefix = output / "logs" / case / rival
    return [
        str(PYTHON.resolve()),
        "-u",
        str(BASE / "Core.py"),
        "--market", str(inputs["market_name"]),
        "--choice_mode", "cognitive",
        "--firm1_mode", "RL",
        "--firm2_mode", rival,
        "--run_experiment",
        "--credit_assignment_v25",
        "--v26",
        "--deterministic_experiment_seed",
        "--deterministic_torch",
        "--learning_corrections",
        "--learner_revision", "4",
        "--training_curriculum", "v17_progressive",
        "--current_active_driver_observation",
        "--seed", "4500",
        "--experiment_seed", "4500",
        "--policy_objective", "profit_maximization",
        "--pool", str(int(pool)),
        "--threshold_profile_source", "cached",
        "--threshold_cache_path", str(resolve_input(inputs["threshold_cache"])),
        "--calibration_json", str(resolve_input(inputs["calibration_json"])),
        "--demand_calibration_artifact", str(
            resolve_input(inputs["demand_calibration"])
        ),
        "--gpt_threshold_mode", "llm_numeric",
        "--train_timesteps", "0",
        "--train_customers", str(int(customers)),
        "--eval_customers", str(int(customers)),
        "--train_steps_per_day", str(STEPS_PER_DAY),
        "--firm1_action_interval_steps", str(STEPS_PER_DAY),
        "--firm2_action_interval_steps", str(STEPS_PER_DAY),
        "--no-live_diagnostics",
        "--regime_market",
        "--regime_features",
        "--spillover_price_aware",
        "--supply_model", "v3",
        "--driver_pay_basis", "fare",
        "--loyalty_model", "v2",
        "--loyalty_pool_size", str(int(customers) * STEPS_PER_DAY * 6 // 10),
        "--persistence_features",
        "--fare_model", "realistic",
        "--shortage_willingness_share", "0.5",
        "--fleet_scale", str(float(inputs.get("fleet_scale", 1.0))),
        "--price_sensitivity_units", "relative",
        "--action_contract", str(config["action"]["contract"]),
        "--rivals_two_lever",
        "--shock_scenario", str(CASES[case]),
        "--policy_sweep", ";".join(fixed_specs(config)),
        "--policy_sweep_out", str(raw),
        "--policy_sweep_openings", str(int(openings)),
        "--policy_sweep_rivals", rival,
        "--policy_sweep_horizon", str(HORIZON_DAYS * STEPS_PER_DAY),
        "--policy_sweep_seed", str(VALIDATION_SEED),
        "--report_prefix", str(prefix),
    ]


def _job_paths(output: Path, case: str, rival: str) -> tuple[Path, Path, Path]:
    raw = output / "raw" / case / f"{rival}.json"
    manifest = output / "manifests" / case / f"{rival}.json"
    log = output / "logs" / case / f"{rival}.log"
    return raw, manifest, log


def run_job(
    command: Sequence[str],
    *,
    raw: Path,
    manifest: Path,
    log: Path,
    config_hash: str,
    sources: Mapping[str, str],
    force: bool,
) -> str:
    identity = {
        "command": list(command),
        "command_sha256": _sha256_json(list(command)),
        "config_hash": str(config_hash),
        "sources": dict(sources),
    }
    if raw.is_file() and manifest.is_file() and not force:
        if json.loads(manifest.read_text(encoding="utf-8")) != identity:
            raise ValueError(f"stale T2 cache has a different identity: {raw}")
        json.loads(raw.read_text(encoding="utf-8"))
        return "cached"
    if (raw.is_file() or manifest.is_file()) and not force:
        raise ValueError(f"partial T2 cache must be repaired explicitly: {raw}")
    raw.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        MPLBACKEND="Agg",
        PYTHONDONTWRITEBYTECODE="1",
    )
    proc = subprocess.run(
        list(command), cwd=BASE, env=env, capture_output=True, text=True
    )
    log.write_text(proc.stdout + proc.stderr, encoding="utf-8")
    if proc.returncode != 0 or not raw.is_file():
        raise RuntimeError(
            f"T2 simulator job failed ({proc.returncode}): {raw}\n"
            + (proc.stdout + proc.stderr)[-4000:]
        )
    atomic_json(manifest, identity)
    return "completed"


def _sample_se(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    average = mean(values)
    variance = sum((value - average) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance / len(values))


def _metric_row(row: Mapping[str, Any]) -> dict[str, float]:
    own = float(row["own_profit"])
    competitor = float(row.get("competitor_profit", row.get("rival_profit")))
    value = profit_report(own, competitor)
    value.update(
        profit_lead=own - competitor,
        own_completed_share=float(
            row["own_completed_share"]
            if "own_completed_share" in row else row["own_share"]
        ),
        competitor_completed_share=float(
            row["competitor_completed_share"]
            if "competitor_completed_share" in row else row["rival_share"]
        ),
    )
    return value


METRIC_KEYS = (
    "final_objective",
    "own_profit",
    "competitor_profit",
    "profit_lead",
    "own_completed_share",
    "competitor_completed_share",
)


def _metric_mean(rows: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    values = [_metric_row(row) for row in rows]
    if not values:
        raise ValueError("cannot summarize an empty T2 metric set")
    result: dict[str, float] = {}
    for key in METRIC_KEYS:
        sample = [float(row[key]) for row in values]
        result[key] = mean(sample)
        result[f"{key}_se"] = _sample_se(sample)
    result["n_openings"] = len(values)
    return result


def _trace_opening_metrics(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["rollout"])].append(row)
    result = {}
    for opening, opening_rows in grouped.items():
        result[opening] = _metric_mean(
            {
                "own_profit": row["profit_per_request"],
                "rival_profit": row["rival_profit_per_request"],
                "own_share": row["own_share"],
                "rival_share": row["rival_share"],
            }
            for row in opening_rows
        )
    return result


def _trace_metric_mean(rows: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    return _metric_mean(_trace_opening_metrics(rows).values())


def _paired_difference(
    first: Mapping[str, Mapping[str, Any]],
    second: Mapping[str, Mapping[str, Any]],
) -> dict[str, float]:
    openings = sorted(set(first) & set(second))
    if not openings:
        raise ValueError("paired T2 comparison has no common openings")
    result: dict[str, float] = {"n_openings": len(openings)}
    for key in METRIC_KEYS:
        differences = [
            float(first[opening][key]) - float(second[opening][key])
            for opening in openings
        ]
        result[key] = mean(differences)
        result[f"{key}_se"] = _sample_se(differences)
    return result


def _best_spec(
    summaries: Mapping[str, Mapping[str, float]], objective: str
) -> str:
    key = "final_objective" if objective == "final" else "own_profit"
    return max(
        summaries,
        key=lambda spec: (
            float(summaries[spec][key]),
            -parse_cell(spec)[0],
            -parse_cell(spec)[1],
        ),
    )


def _labeled_cell(spec: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
    distance, time = parse_cell(spec)
    return {
        "spec": spec,
        "distance_level": distance,
        "time_level": time,
        **dict(metrics),
    }


def analyze_job(raw: Mapping[str, Any]) -> dict[str, Any]:
    results = dict(raw["results"])
    traces = dict(raw["traces"])
    opening_sets = {tuple(sorted(dict(rows))) for rows in results.values()}
    if len(opening_sets) != 1:
        raise ValueError("T2 fixed cells do not share the same openings")
    opening_names = list(next(iter(opening_sets)))
    if len(opening_names) < 2:
        raise ValueError("leave-one-opening-out analysis requires at least two openings")
    overall = {
        spec: _metric_mean(dict(openings).values())
        for spec, openings in results.items()
    }
    best_final_spec = _best_spec(overall, "final")
    best_own_spec = _best_spec(overall, "own")

    regime_rows: dict[
        str, dict[str, dict[str, list[Mapping[str, Any]]]]
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for spec, rows in traces.items():
        for row in rows:
            regime_rows[str(row["regime"])][spec][str(row["rollout"])].append(row)

    tables: dict[str, dict[str, dict[str, Any]]] = {
        "final": {},
        "own_profit": {},
    }
    selected_specs: dict[str, dict[str, str]] = {"final": {}, "own_profit": {}}
    loo_specs: dict[str, dict[str, dict[str, str]]] = {
        "final": defaultdict(dict),
        "own_profit": defaultdict(dict),
    }
    for regime, by_spec in sorted(regime_rows.items()):
        summaries = {
            spec: _trace_metric_mean(
                row for opening_rows in rows.values() for row in opening_rows
            )
            for spec, rows in by_spec.items()
        }
        for objective in ("final", "own_profit"):
            chosen = _best_spec(
                summaries, "final" if objective == "final" else "own"
            )
            selected_specs[objective][regime] = chosen
            table = _labeled_cell(chosen, summaries[chosen])
            table["selection_method"] = "all_openings_for_later_evaluation"
            table["leave_one_opening_out"] = []
            for held_out in opening_names:
                fit_summaries = {}
                for spec, rows_by_opening in by_spec.items():
                    fit_rows = [
                        row
                        for opening, rows in rows_by_opening.items()
                        if opening != held_out
                        for row in rows
                    ]
                    if fit_rows:
                        fit_summaries[spec] = _trace_metric_mean(fit_rows)
                loo_chosen = _best_spec(
                    fit_summaries, "final" if objective == "final" else "own"
                )
                loo_specs[objective][regime][held_out] = loo_chosen
                held_out_rows = by_spec[loo_chosen].get(held_out, [])
                if not held_out_rows:
                    continue
                table["leave_one_opening_out"].append({
                    "held_out_opening": held_out,
                    **_labeled_cell(
                        loo_chosen,
                        _trace_metric_mean(held_out_rows),
                    ),
                })
            tables[objective][regime] = table

    adaptation: dict[str, Any] = {}
    for objective in ("final", "own_profit"):
        switch_by_opening = {}
        for opening in opening_names:
            switched_rows = []
            for regime in selected_specs[objective]:
                chosen = loo_specs[objective][regime][opening]
                switched_rows.extend(regime_rows[regime][chosen].get(opening, []))
            switch_by_opening[opening] = _trace_metric_mean(switched_rows)
        switch_metrics = _metric_mean(switch_by_opening.values())
        fixed_spec = best_final_spec if objective == "final" else best_own_spec
        fixed_by_opening = _trace_opening_metrics(traces[fixed_spec])
        fixed_metrics = _metric_mean(fixed_by_opening.values())
        adaptation[objective] = {
            "selection_method": "leave_one_opening_out",
            "switch": switch_metrics,
            "fixed": _labeled_cell(fixed_spec, fixed_metrics),
            "switch_minus_fixed": _paired_difference(
                switch_by_opening, fixed_by_opening
            ),
        }

    best_final_by_opening = {
        opening: _metric_row(row)
        for opening, row in dict(results[best_final_spec]).items()
    }
    best_own_by_opening = {
        opening: _metric_row(row)
        for opening, row in dict(results[best_own_spec]).items()
    }

    return {
        "best_fixed": {
            "final": _labeled_cell(best_final_spec, overall[best_final_spec]),
            "own_profit": _labeled_cell(best_own_spec, overall[best_own_spec]),
        },
        "regime_table": tables,
        "adaptation_value": adaptation,
        "cost_of_competing_for_lead": {
            "definition": "final-objective-best fixed minus own-profit-best fixed",
            **_paired_difference(best_final_by_opening, best_own_by_opening),
        },
    }


def analyze(config: Mapping[str, Any], output: Path, openings: int) -> dict[str, Any]:
    jobs: dict[str, Any] = {}
    levels = ACTION_LEVELS[str(config["action"]["contract"])]
    lower_edge = float(levels[0])
    upper_edge = float(levels[-1])
    edges: dict[str, dict[str, Any]] = {}
    for case in CASES:
        jobs[case] = {}
        edges[case] = {
            "lower_distance": [], "lower_time": [],
            "upper_distance": [], "upper_time": [],
            "lower_any": [], "upper_any": [],
        }
        for rival in RIVALS:
            raw, _, _ = _job_paths(output, case, rival)
            report = analyze_job(json.loads(raw.read_text(encoding="utf-8")))
            jobs[case][rival] = report
            chosen = report["best_fixed"]["final"]
            distance = float(chosen["distance_level"])
            time = float(chosen["time_level"])
            lower_distance = math.isclose(distance, lower_edge, abs_tol=1e-12)
            lower_time = math.isclose(time, lower_edge, abs_tol=1e-12)
            upper_distance = math.isclose(distance, upper_edge, abs_tol=1e-12)
            upper_time = math.isclose(time, upper_edge, abs_tol=1e-12)
            for key, hit in (
                ("lower_distance", lower_distance), ("lower_time", lower_time),
                ("upper_distance", upper_distance), ("upper_time", upper_time),
                ("lower_any", lower_distance or lower_time),
                ("upper_any", upper_distance or upper_time),
            ):
                if hit:
                    edges[case][key].append(rival)
    majority_hits = [
        {"case": case, "edge": edge, "count": len(summary[edge])}
        for case, summary in edges.items()
        for edge in ("lower_any", "upper_any")
        if len(summary[edge]) > len(RIVALS) / 2.0
    ]
    edge_checks = {
        "lower_level": lower_edge,
        "upper_level": upper_edge,
        "competitors_total": len(RIVALS),
        "by_case": edges,
        "majority_hits": majority_hits,
        "requires_grid_review": bool(majority_hits),
    }
    return {
        "phase": "T2",
        "omega": 0.5,
        "city": "nyc",
        "horizon_days": HORIZON_DAYS,
        "onset_day": ONSET_DAY,
        "openings": int(openings),
        "competitors": list(RIVALS),
        "cases": list(CASES),
        "action_contract": str(config["action"]["contract"]),
        "levels": list(levels),
        "rollouts": expected_rollouts(config, openings),
        "ranking_objective": OBJECTIVE_NAME,
        "jobs": jobs,
        "edge_checks": edge_checks,
        "lower_edge_check": {
            "level": lower_edge,
            "by_case": {case: data["lower_any"] for case, data in edges.items()},
        },
        "upper_edge_check": {
            "level": upper_edge,
            "by_case": {case: data["upper_any"] for case, data in edges.items()},
        },
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# V17 T2 adaptation-value sweep",
        "",
        f"Action contract: `{report['action_contract']}`",
        f"Rollouts: {report['rollouts']}",
        "",
        "Regime tables are fitted leave-one-opening-out; every uncertainty below is an opening-level standard error.",
        "",
        "| Case | Competitor | Final-best cell | Own profit | Competitor profit | Final objective | Own-profit-best cell | Cost of competition | Adaptation value |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case, rivals in report["jobs"].items():
        for rival, result in rivals.items():
            best = result["best_fixed"]["final"]
            adaptation = result["adaptation_value"]["final"][
                "switch_minus_fixed"
            ]
            own_best = result["best_fixed"]["own_profit"]
            cost = result["cost_of_competing_for_lead"]
            lines.append(
                f"| {case} | {rival} | "
                f"({best['distance_level']:g}, {best['time_level']:g}) | "
                f"{best['own_profit']:.4f} +/- {best['own_profit_se']:.4f} | "
                f"{best['competitor_profit']:.4f} +/- {best['competitor_profit_se']:.4f} | "
                f"{best['final_objective']:.4f} +/- {best['final_objective_se']:.4f} | "
                f"({own_best['distance_level']:g}, {own_best['time_level']:g}) | "
                f"{cost['own_profit']:.4f} +/- {cost['own_profit_se']:.4f} | "
                f"{adaptation['final_objective']:.4f} +/- {adaptation['final_objective_se']:.4f} |"
            )
    edge = report["edge_checks"]
    lines.extend(("", "## Two-sided grid-edge check", ""))
    for case, summary in edge["by_case"].items():
        lines.append(
            f"- {case}: lower {len(summary['lower_any'])}/{edge['competitors_total']} "
            f"({', '.join(summary['lower_any']) or 'none'}); upper "
            f"{len(summary['upper_any'])}/{edge['competitors_total']} "
            f"({', '.join(summary['upper_any']) or 'none'})."
        )
    lines.extend((
        "",
        f"Grid review required: **{'YES' if edge['requires_grid_review'] else 'NO'}**.",
        "",
    ))
    return "\n".join(lines)


def _alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=BASE / "runs/v17_t2")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--openings", type=int, default=OPENINGS)
    parser.add_argument("--customers", type=int, default=100)
    parser.add_argument("--pool", type=int, default=2000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    validate_config(config)
    if args.workers < 1 or args.openings < 1 or args.customers < 1:
        parser.error("workers, openings, and customers must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_hash = canonical_hash(config)
    commands = {
        (case, rival): sweep_command(
            config,
            case=case,
            rival=rival,
            output=output,
            openings=args.openings,
            customers=args.customers,
            pool=args.pool,
        )
        for case in CASES
        for rival in RIVALS
    }
    plan = {
        "phase": "T2",
        "jobs": len(commands),
        "workers": int(args.workers),
        "rollouts": expected_rollouts(config, args.openings),
        "action_contract": config["action"]["contract"],
        "config_hash": config_hash,
        "estimated_wall_hours": estimated_wall_hours(
            config, openings=args.openings, workers=args.workers
        ),
        "runtime_basis": {
            "prior_rollouts": PRIOR_T2_ROLLOUTS,
            "prior_workers": PRIOR_T2_WORKERS,
            "prior_wall_hours": PRIOR_T2_WALL_HOURS,
        },
    }
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return 0

    status_path = output / "status.json"
    if status_path.is_file() and not args.force:
        previous = json.loads(status_path.read_text(encoding="utf-8"))
        if previous.get("state") == "running" and _alive(previous.get("pid", -1)):
            raise SystemExit(
                f"T2 is already running as pid {previous['pid']}; not duplicating"
            )
    atomic_json(
        status_path,
        {
            **plan,
            "state": "running",
            "pass": False,
            "pid": os.getpid(),
            "completed_jobs": 0,
        },
    )
    sources = source_identity()
    completed = 0
    try:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(int(args.workers), len(commands))
        ) as executor:
            futures = {}
            for (case, rival), command in commands.items():
                raw, manifest, log = _job_paths(output, case, rival)
                future = executor.submit(
                    run_job,
                    command,
                    raw=raw,
                    manifest=manifest,
                    log=log,
                    config_hash=config_hash,
                    sources=sources,
                    force=args.force,
                )
                futures[future] = (case, rival)
            for future in concurrent.futures.as_completed(futures):
                case, rival = futures[future]
                result = future.result()
                completed += 1
                print(
                    f"[T2] {case}/{rival}: {result} ({completed}/{len(commands)})",
                    flush=True,
                )
                atomic_json(
                    status_path,
                    {
                        **plan,
                        "state": "running",
                        "pass": False,
                        "pid": os.getpid(),
                        "completed_jobs": completed,
                        "last_completed": [case, rival],
                    },
                )
        report = analyze(config, output, args.openings)
        atomic_json(output / "report.json", report)
        (output / "report.md").write_text(
            render_markdown(report), encoding="utf-8"
        )
        review = bool(report["edge_checks"]["requires_grid_review"])
        atomic_json(
            status_path,
            {
                **plan,
                "state": "requires_grid_review" if review else "passed",
                "pass": not review,
                "pid": os.getpid(),
                "completed_jobs": completed,
                "report": str(output / "report.json"),
                "lower_edge_check": report["lower_edge_check"],
                "upper_edge_check": report["upper_edge_check"],
            },
        )
        return 2 if review else 0
    except BaseException as exc:
        atomic_json(
            status_path,
            {
                **plan,
                "state": "failed",
                "pass": False,
                "pid": os.getpid(),
                "completed_jobs": completed,
                "error": repr(exc),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
