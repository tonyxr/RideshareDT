#!/usr/bin/env python3
"""Run the RevB V17 T2 adaptation-value sweep on production market wiring.

The sweep is content-addressed and resumable at the case/competitor boundary.
Each worker evaluates every fixed distance/time cell on three common-random-
number validation openings.  Analysis uses the registered final objective,
while retaining both component profits and completed shares in every table.
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
from v17_objective import final_objective, profit_report  # noqa: E402


PYTHON = ROOT / ".venv/bin/python"
STEPS_PER_DAY = 6
HORIZON_DAYS = 40
ONSET_DAY = 10
OPENINGS = 3
VALIDATION_SEED = 20_231
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


def _metric_mean(rows: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    values = list(rows)
    if not values:
        raise ValueError("cannot summarize an empty T2 metric set")
    own = mean(float(row["own_profit"]) for row in values)
    competitor = mean(float(row["rival_profit"]) for row in values)
    result = profit_report(own, competitor)
    result.update(
        profit_lead=own - competitor,
        own_completed_share=mean(float(row["own_share"]) for row in values),
        competitor_completed_share=mean(
            float(row["rival_share"]) for row in values
        ),
        n=len(values),
    )
    return result


def _trace_metric_mean(rows: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    return _metric_mean(
        {
            "own_profit": row["profit_per_request"],
            "rival_profit": row["rival_profit_per_request"],
            "own_share": row["own_share"],
            "rival_share": row["rival_share"],
        }
        for row in rows
    )


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
    overall = {
        spec: _metric_mean(dict(openings).values())
        for spec, openings in results.items()
    }
    best_final_spec = _best_spec(overall, "final")
    best_own_spec = _best_spec(overall, "own")

    regime_rows: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for spec, rows in traces.items():
        for row in rows:
            regime_rows[str(row["regime"])][spec].append(row)

    tables: dict[str, dict[str, dict[str, Any]]] = {
        "final": {},
        "own_profit": {},
    }
    selected_specs: dict[str, dict[str, str]] = {"final": {}, "own_profit": {}}
    for regime, by_spec in sorted(regime_rows.items()):
        summaries = {
            spec: _trace_metric_mean(rows) for spec, rows in by_spec.items()
        }
        for objective in ("final", "own_profit"):
            chosen = _best_spec(
                summaries, "final" if objective == "final" else "own"
            )
            selected_specs[objective][regime] = chosen
            tables[objective][regime] = _labeled_cell(
                chosen, summaries[chosen]
            )

    adaptation: dict[str, Any] = {}
    for objective in ("final", "own_profit"):
        switched_rows = []
        for regime, chosen in selected_specs[objective].items():
            switched_rows.extend(regime_rows[regime][chosen])
        switch_metrics = _trace_metric_mean(switched_rows)
        fixed_spec = best_final_spec if objective == "final" else best_own_spec
        fixed_metrics = _trace_metric_mean(traces[fixed_spec])
        adaptation[objective] = {
            "switch": switch_metrics,
            "fixed": _labeled_cell(fixed_spec, fixed_metrics),
            "switch_minus_fixed": {
                key: float(switch_metrics[key]) - float(fixed_metrics[key])
                for key in (
                    "final_objective",
                    "own_profit",
                    "competitor_profit",
                    "profit_lead",
                    "own_completed_share",
                    "competitor_completed_share",
                )
            },
        }

    return {
        "best_fixed": {
            "final": _labeled_cell(best_final_spec, overall[best_final_spec]),
            "own_profit": _labeled_cell(best_own_spec, overall[best_own_spec]),
        },
        "regime_table": tables,
        "adaptation_value": adaptation,
    }


def analyze(config: Mapping[str, Any], output: Path, openings: int) -> dict[str, Any]:
    jobs: dict[str, Any] = {}
    calm_edge = []
    levels = ACTION_LEVELS[str(config["action"]["contract"])]
    lower_edge = float(levels[0])
    for case in CASES:
        jobs[case] = {}
        for rival in RIVALS:
            raw, _, _ = _job_paths(output, case, rival)
            report = analyze_job(json.loads(raw.read_text(encoding="utf-8")))
            jobs[case][rival] = report
            if case == "C0_calm":
                chosen = report["best_fixed"]["final"]
                calm_edge.append(
                    math.isclose(
                        float(chosen["distance_level"]), lower_edge, abs_tol=1e-12
                    )
                    or math.isclose(
                        float(chosen["time_level"]), lower_edge, abs_tol=1e-12
                    )
                )
    edge_count = sum(calm_edge)
    trigger = (
        str(config["action"]["contract"]) == "two_lever_v1"
        and edge_count > len(RIVALS) / 2.0
    )
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
        "ranking_objective": "own_profit_minus_0.75_competitor_profit",
        "jobs": jobs,
        "lower_edge_check": {
            "level": lower_edge,
            "competitors_on_edge": edge_count,
            "competitors_total": len(RIVALS),
            "most_competitors": edge_count > len(RIVALS) / 2.0,
            "switch_to_v2_and_rerun": trigger,
        },
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# V17 T2 adaptation-value sweep",
        "",
        f"Action contract: `{report['action_contract']}`",
        f"Rollouts: {report['rollouts']}",
        "",
        "| Case | Competitor | Final-best cell | Own profit | Competitor profit | Final objective | Adaptation value |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for case, rivals in report["jobs"].items():
        for rival, result in rivals.items():
            best = result["best_fixed"]["final"]
            adaptation = result["adaptation_value"]["final"][
                "switch_minus_fixed"
            ]["final_objective"]
            lines.append(
                f"| {case} | {rival} | "
                f"({best['distance_level']:.1f}, {best['time_level']:.1f}) | "
                f"{best['own_profit']:.4f} | {best['competitor_profit']:.4f} | "
                f"{best['final_objective']:.4f} | {adaptation:.4f} |"
            )
    edge = report["lower_edge_check"]
    lines.extend([
        "",
        f"Lower-edge count: {edge['competitors_on_edge']}/{edge['competitors_total']}.",
        f"V2 rerun required: **{'YES' if edge['switch_to_v2_and_rerun'] else 'NO'}**.",
        "",
    ])
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
    parser.add_argument("--workers", type=int, default=4)
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
        rerun = bool(report["lower_edge_check"]["switch_to_v2_and_rerun"])
        atomic_json(
            status_path,
            {
                **plan,
                "state": "requires_v2_rerun" if rerun else "passed",
                "pass": not rerun,
                "pid": os.getpid(),
                "completed_jobs": completed,
                "report": str(output / "report.json"),
                "lower_edge_check": report["lower_edge_check"],
            },
        )
        return 2 if rerun else 0
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
