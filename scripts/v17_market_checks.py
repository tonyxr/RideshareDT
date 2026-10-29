#!/usr/bin/env python3
"""Run and summarize the CRISP-Ride V17 T1 market checks.

The expensive simulator sweeps are resumable: each raw JSON is written once,
validated against a content-addressed command manifest, and reused on relaunch.
T1.1--T1.5 use six paired openings in NYC and Boston; T1.6 quotes 1,000
held-out city trip-bank rows at the card in all three cities.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent
sys.path.insert(0, str(BASE))

from MarketInteraction import MarketInteraction, RideContext  # noqa: E402
from diagnostic_logging import atomic_json  # noqa: E402
from scripts.v17_config import (  # noqa: E402
    DEFAULT_CONFIG,
    canonical_hash,
    load_config,
    resolve_input,
    validate_config,
)

PYTHON = ROOT / ".venv/bin/python"
STEPS_PER_DAY = 6
T1_CITIES = ("nyc", "boston")
REALITY_CITIES = ("nyc", "boston", "chicago")
CARD = "static:1"


def schedule(*, default: float = 1.0, days: Mapping[int, float]) -> str:
    payload = {
        "default": [float(default), 1.0],
        "days": {str(day): [float(level), 1.0] for day, level in days.items()},
    }
    return "schedule:" + json.dumps(payload, sort_keys=True, separators=(",", ":"))


UNDERCUT = schedule(days={day: 0.8 for day in range(10, 15)})
FARE_RISE = schedule(days={day: 1.25 for day in range(14, 17)})
SHORTAGE_SURGE = FARE_RISE


JOBS: dict[str, dict[str, Any]] = {
    "calm_static": {
        "scenario": "scenarios/calm.json",
        "rival": "static",
        "days": 30,
        "specs": (CARD, FARE_RISE),
    },
    "calm_memory": {
        "scenario": "scenarios/calm.json",
        "rival": "adaptive_best_response_memory",
        "days": 30,
        "specs": (CARD, UNDERCUT),
    },
    "shortage_static": {
        "scenario": "scenarios/cases/shortage_severe_late.json",
        "rival": "static",
        "days": 24,
        "specs": (CARD, SHORTAGE_SURGE),
    },
    "outage_static": {
        "scenario": "scenarios/cases/own_outage_late.json",
        "rival": "static",
        "days": 24,
        "specs": (CARD,),
    },
}


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_identity() -> dict[str, str]:
    names = (
        "Core.py",
        "MarketInteraction.py",
        "driver_supply.py",
        "market_shocks.py",
        "platform_mdp.py",
        "pricing_models.py",
        "scripts/v17_market_checks.py",
    )
    return {name: _sha256_file(BASE / name) for name in names}


def _write_status(output: Path, *, state: str, **fields: Any) -> None:
    atomic_json(output / "status.json", {"state": state, **fields})


def market_command(
    config: Mapping[str, Any],
    *,
    city: str,
    job: str,
    output: Path,
    openings: int,
    customers: int,
    pool: int,
    seed: int = 17_101,
) -> list[str]:
    validate_config(config)
    if city not in T1_CITIES:
        raise ValueError(f"T1 market sweeps are only declared for {T1_CITIES}")
    if job not in JOBS:
        raise ValueError(f"unknown T1 job: {job}")
    inputs = config["city_inputs"][city]
    spec = JOBS[job]
    raw = output / "raw" / city / f"{job}.json"
    prefix = output / "logs" / city / job
    return [
        str(PYTHON.resolve()),
        "-u",
        str(BASE / "Core.py"),
        "--market", inputs["market_name"],
        "--choice_mode", "cognitive",
        "--firm1_mode", "RL",
        "--firm2_mode", spec["rival"],
        "--run_experiment",
        "--credit_assignment_v25",
        "--v26",
        "--deterministic_experiment_seed",
        "--deterministic_torch",
        "--learning_corrections",
        "--learner_revision", "4",
        "--training_curriculum", "v17_progressive",
        "--current_active_driver_observation",
        "--seed", str(seed),
        "--experiment_seed", str(seed),
        "--policy_objective", "profit_maximization",
        "--pool", str(pool),
        "--threshold_profile_source", "cached",
        "--threshold_cache_path", str(resolve_input(inputs["threshold_cache"])),
        "--calibration_json", str(resolve_input(inputs["calibration_json"])),
        "--demand_calibration_artifact", str(resolve_input(inputs["demand_calibration"])),
        "--gpt_threshold_mode", "llm_numeric",
        "--train_timesteps", "0",
        "--train_customers", str(customers),
        "--eval_customers", str(customers),
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
        "--loyalty_pool_size", str(customers * STEPS_PER_DAY * 6 // 10),
        "--persistence_features",
        "--fare_model", "realistic",
        "--shortage_willingness_share", "0.5",
        "--fleet_scale", str(float(inputs.get("fleet_scale", 1.0))),
        "--price_sensitivity_units", "relative",
        "--action_contract", "two_lever_v1",
        "--rivals_two_lever",
        "--shock_scenario", str(BASE / spec["scenario"]),
        "--policy_sweep", ";".join(spec["specs"]),
        "--policy_sweep_out", str(raw),
        "--policy_sweep_openings", str(openings),
        "--policy_sweep_rivals", spec["rival"],
        "--policy_sweep_horizon", str(int(spec["days"]) * STEPS_PER_DAY),
        "--policy_sweep_seed", "20231",
        "--report_prefix", str(prefix),
    ]


def _run_job(
    command: Sequence[str],
    raw: Path,
    manifest: Path,
    *,
    force: bool,
    config_hash: str,
) -> None:
    identity = {
        "command": list(command),
        "command_sha256": _sha256_json(list(command)),
        "config_hash": str(config_hash),
        "sources": source_identity(),
    }
    if raw.is_file() and manifest.is_file() and not force:
        if json.loads(manifest.read_text(encoding="utf-8")) != identity:
            raise ValueError(f"stale T1 cache has a different command: {raw}")
        json.loads(raw.read_text(encoding="utf-8"))
        return
    raw.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    log = manifest.with_suffix(".log")
    env = dict(os.environ)
    env.update(
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        MPLBACKEND="Agg",
    )
    proc = subprocess.run(
        list(command), cwd=BASE, env=env, capture_output=True, text=True
    )
    log.write_text(proc.stdout + proc.stderr, encoding="utf-8")
    if proc.returncode != 0 or not raw.is_file():
        raise RuntimeError(
            f"T1 simulator job failed ({proc.returncode}): {raw}\n"
            + (proc.stdout + proc.stderr)[-4000:]
        )
    atomic_json(manifest, identity)


def _rows_by_rollout(
    rows: Iterable[Mapping[str, Any]],
    *,
    start_day: int,
    end_day: int,
) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        day = int(row["step"]) // STEPS_PER_DAY
        if start_day <= day < end_day:
            grouped[str(row["rollout"])].append(row)
    return dict(grouped)


def _opening_means(
    rows: Iterable[Mapping[str, Any]],
    column: str,
    *,
    start_day: int,
    end_day: int,
) -> dict[str, float]:
    grouped = _rows_by_rollout(rows, start_day=start_day, end_day=end_day)
    return {
        rollout: mean(float(row[column]) for row in values)
        for rollout, values in grouped.items()
        if values
    }


def estimate(values: Sequence[float]) -> dict[str, float | int]:
    sample = [float(value) for value in values]
    if not sample:
        raise ValueError("cannot estimate an empty sample")
    avg = mean(sample)
    se = stdev(sample) / math.sqrt(len(sample)) if len(sample) > 1 else 0.0
    if se > 0.0:
        t_value = avg / se
    elif avg > 0.0:
        t_value = float("inf")
    elif avg < 0.0:
        t_value = float("-inf")
    else:
        t_value = 0.0
    return {"n": len(sample), "mean": avg, "se": se, "t": t_value}


def paired_effect(
    data: Mapping[str, Any],
    treated: str,
    control: str,
    column: str,
    *,
    start_day: int,
    end_day: int,
) -> dict[str, float | int]:
    treatment = _opening_means(
        data["traces"][treated], column, start_day=start_day, end_day=end_day
    )
    baseline = _opening_means(
        data["traces"][control], column, start_day=start_day, end_day=end_day
    )
    names = sorted(set(treatment) & set(baseline))
    if not names:
        raise ValueError(f"no paired openings for {column}")
    return estimate([treatment[name] - baseline[name] for name in names])


def _read_raw(output: Path, city: str, job: str) -> dict[str, Any]:
    return json.loads((output / "raw" / city / f"{job}.json").read_text())


def _legacy_supply_reference(city: str) -> dict[str, Any]:
    path = ROOT / "rideshare_v26r_revision15d/calibration_artifacts" / city / "city_calibration.json"
    data = json.loads(path.read_text(encoding="utf-8"))["supply_operating_point"]
    return data


def _t11(city: str, data: Mapping[str, Any], customers: int) -> dict[str, Any]:
    card_rows = list(data["traces"][CARD])
    tail = _rows_by_rollout(card_rows, start_day=15, end_day=30)
    rows = [row for values in tail.values() for row in values]
    if not rows:
        raise ValueError(f"{city}: missing calm card tail")
    fulfillment = mean(float(row["fulfillment"]) for row in rows)
    fare = mean(float(row["mean_fare_paid"]) for row in rows)
    reference = _legacy_supply_reference(city)
    calibration = reference["calibration"]
    reference_fulfillment = float(calibration["anchor_fulfillment"])
    # The T1.1 fare reference is the 15d calibrated card, evaluated on the
    # same city trip bank in T1.6; it is attached during final summarization.
    by_hour: dict[str, dict[str, float]] = {}
    for hour in sorted({int(row["hour"]) for row in rows}):
        hour_rows = [row for row in rows if int(row["hour"]) == hour]
        expected = (
            float(reference["config"]["base_active_drivers"])
            * float(customers)
            / float(reference["config"]["reference_customers"])
            * float(reference["hour_profile"][str(hour)])
        )
        by_hour[str(hour)] = {
            "v17_mean": mean(float(row["active_drivers_firm1"]) for row in hour_rows),
            "v15d_calibrated_card": expected,
        }
    return {
        "v17_fulfillment": fulfillment,
        "v15d_fulfillment": reference_fulfillment,
        "fulfillment_absolute_gap": abs(fulfillment - reference_fulfillment),
        "fulfillment_pass": abs(fulfillment - reference_fulfillment) <= 0.02,
        "v17_mean_fare_paid": fare,
        "active_drivers_by_hour": by_hour,
    }


def quote_reality_gap(
    config: Mapping[str, Any], city: str, *, rows: int = 1000
) -> dict[str, Any]:
    inputs = config["city_inputs"][city]
    calibration_path = resolve_input(inputs["calibration_json"])
    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    calibration = dict(payload.get("calibration", payload))
    refit = dict(payload.get("v17_fare_refit", {}) or {})
    if refit:
        source_path = Path(str(refit["source_calibration"])).resolve()
        source_payload = json.loads(source_path.read_text(encoding="utf-8"))
        legacy_tariff = dict(
            source_payload.get("calibration", source_payload)["tariff"]
        )
    else:
        legacy_tariff = dict(calibration["tariff"])
    trip_bank = calibration_path.parent / str(calibration["trip_bank"]["path"])
    market_name = str(inputs["market_name"])
    legacy = MarketInteraction(market_name, seed=17)
    realistic = MarketInteraction(market_name, seed=17)
    legacy.register_calibrated_tariff(legacy_tariff)
    legacy.configure_fare_model("legacy")
    realistic.register_calibrated_tariff(dict(calibration["tariff"]))
    realistic.configure_fare_model("realistic")
    observed: list[float] = []
    legacy_quotes: list[float] = []
    realistic_quotes: list[float] = []
    with trip_bank.open(newline="", encoding="utf-8") as handle:
        heldout_rows = list(csv.DictReader(handle))[-int(rows):]
        for row in heldout_rows:
            distance = float(row["distance_miles"])
            duration = (
                float(row["duration_minutes"])
                if str(row.get("duration_minutes", "")).strip()
                else max(5.0, 60.0 * distance / 22.0)
            )
            ctx = RideContext(
                day_of_week=int(row["day_of_week"]),
                weather=str(row["weather"]),
                hour=int(row["hour"]),
                airport=str(row["airport"]).strip().lower() in {"1", "true", "yes"},
                service=str(row["service"]),
            )
            observed.append(float(row["actual_price"]))
            legacy_quotes.append(legacy.quote_price(distance, duration, ctx))
            realistic_quotes.append(realistic.quote_price(distance, duration, ctx))
    if len(observed) != int(rows):
        raise ValueError(f"{city}: expected {rows} trip-bank rows, found {len(observed)}")
    actual = np.asarray(observed, dtype=float)
    old = np.asarray(legacy_quotes, dtype=float)
    new = np.asarray(realistic_quotes, dtype=float)
    correlation = float(np.corrcoef(actual, new)[0, 1])
    return {
        "rows": len(observed),
        "partition": "final rows from the declared 20% held-out trip-bank partition",
        "observed_mean": float(actual.mean()),
        "v15d_card_mean": float(old.mean()),
        "v17_card_mean": float(new.mean()),
        "v17_vs_v15d_relative_gap": float(new.mean() / old.mean() - 1.0),
        "v17_bias": float((new - actual).mean()),
        "v17_mae": float(np.abs(new - actual).mean()),
        "v15d_mae": float(np.abs(old - actual).mean()),
        "v17_rmse": float(np.sqrt(np.mean((new - actual) ** 2))),
        "v17_correlation": correlation,
    }


def summarize_city(
    config: Mapping[str, Any], output: Path, city: str, *, customers: int
) -> dict[str, Any]:
    calm = _read_raw(output, city, "calm_static")
    memory = _read_raw(output, city, "calm_memory")
    shortage = _read_raw(output, city, "shortage_static")
    outage = _read_raw(output, city, "outage_static")
    reality = quote_reality_gap(config, city, rows=1000)
    t11 = _t11(city, calm, customers)
    fare_gap = abs(float(reality["v17_vs_v15d_relative_gap"]))
    t11.update(
        v15d_card_mean_fare=float(reality["v15d_card_mean"]),
        fare_relative_gap=fare_gap,
        fare_pass=fare_gap <= 0.03,
    )

    undercut_rows = _rows_by_rollout(
        memory["traces"][UNDERCUT], start_day=10, end_day=25
    )
    memory_states = [
        str(row["rival_memory_state"])
        for values in undercut_rows.values()
        for row in values
    ]
    retaliation_steps = sum(state == "retaliation" for state in memory_states)
    post_undercut_retaliation = sum(
        str(row["rival_memory_state"]) == "retaliation"
        for row in memory["traces"][UNDERCUT]
        if 15 <= int(row["step"]) // STEPS_PER_DAY < 25
    )
    t12 = {
        "retaliation_steps": retaliation_steps,
        "post_undercut_retaliation_steps": post_undercut_retaliation,
        "rival_level_effect": paired_effect(
            memory, UNDERCUT, CARD, "firm2_level", start_day=10, end_day=25
        ),
        "pass": retaliation_steps > 0 and post_undercut_retaliation > 0,
    }

    t13_drivers = paired_effect(
        shortage,
        SHORTAGE_SURGE,
        CARD,
        "active_drivers_firm1",
        start_day=14,
        end_day=20,
    )
    t13_fulfillment = paired_effect(
        shortage,
        SHORTAGE_SURGE,
        CARD,
        "fulfillment",
        start_day=14,
        end_day=20,
    )
    t13 = {
        "window_days": [14, 20],
        "active_drivers": t13_drivers,
        "fulfillment": t13_fulfillment,
        "pass": (
            float(t13_drivers["mean"]) > 0.0
            and float(t13_drivers["t"]) >= 3.0
            and float(t13_fulfillment["mean"]) > 0.0
            and float(t13_fulfillment["t"]) >= 3.0
        ),
    }

    t14 = {
        "earnings_per_online_hour": paired_effect(
            calm,
            FARE_RISE,
            CARD,
            "earnings_online_hour_firm1",
            start_day=14,
            end_day=20,
        ),
        "online_drivers": paired_effect(
            calm,
            FARE_RISE,
            CARD,
            "active_drivers_firm1",
            start_day=14,
            end_day=20,
        ),
        "supply_stock_state": paired_effect(
            calm,
            FARE_RISE,
            CARD,
            "stock_firm1",
            start_day=14,
            end_day=20,
        ),
    }

    pre_loyalty = _opening_means(
        outage["traces"][CARD],
        "loyalty_mean_firm1",
        start_day=9,
        end_day=12,
    )
    post_loyalty = _opening_means(
        outage["traces"][CARD],
        "loyalty_mean_firm1",
        start_day=15,
        end_day=22,
    )
    paired = sorted(set(pre_loyalty) & set(post_loyalty))
    t15 = {
        "post_minus_pre_own_loyalty": estimate(
            [post_loyalty[name] - pre_loyalty[name] for name in paired]
        ),
        "reported_only": True,
    }

    return {
        "T1.1": t11,
        "T1.2": t12,
        "T1.3": t13,
        "T1.4": t14,
        "T1.5": t15,
        "T1.6": reality,
        "required_pass": bool(t11["fulfillment_pass"] and t11["fare_pass"] and t12["pass"] and t13["pass"]),
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# V17 T1 market checks",
        "",
        f"Config hash: `{report['config_hash']}`",
        f"Overall required checks: **{'PASS' if report['passed'] else 'FAIL'}**",
        "",
        "| City | T1.1 fulfillment | T1.1 fare | T1.2 retaliation | T1.3 supply/fulfillment | Required |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for city, result in report["cities"].items():
        lines.append(
            f"| {city} | {'PASS' if result['T1.1']['fulfillment_pass'] else 'FAIL'} "
            f"({result['T1.1']['fulfillment_absolute_gap']:.3f}) | "
            f"{'PASS' if result['T1.1']['fare_pass'] else 'FAIL'} "
            f"({result['T1.1']['fare_relative_gap']:.1%}) | "
            f"{'PASS' if result['T1.2']['pass'] else 'FAIL'} | "
            f"{'PASS' if result['T1.3']['pass'] else 'FAIL'} | "
            f"{'PASS' if result['required_pass'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "T1.4, T1.5, and T1.6 are descriptive outputs, as specified in the execution manual.",
            "Full machine-readable results are in `report.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=BASE / "runs/v17_t1")
    parser.add_argument("--openings", type=int, default=6)
    parser.add_argument(
        "--customers",
        type=int,
        default=500,
        help="Requests per block; 500 is the declared city-calibration reference volume.",
    )
    parser.add_argument("--pool", type=int, default=2000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one opening with 12 customers and a 200-rider pool.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    validate_config(config)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    openings = 1 if args.smoke else max(1, int(args.openings))
    customers = 12 if args.smoke else max(1, int(args.customers))
    pool = 200 if args.smoke else max(1, int(args.pool))
    manifest = {
        "phase": "T1",
        "config": str(args.config.resolve()),
        "config_hash": canonical_hash(config),
        "openings": openings,
        "customers": customers,
        "pool": pool,
        "smoke": bool(args.smoke),
        "jobs": JOBS,
    }
    atomic_json(output / "manifest.json", manifest)
    _write_status(output, state="running", completed_jobs=0, total_jobs=8)
    completed = 0
    try:
        for city in T1_CITIES:
            for job in JOBS:
                command = market_command(
                    config,
                    city=city,
                    job=job,
                    output=output,
                    openings=openings,
                    customers=customers,
                    pool=pool,
                )
                raw = output / "raw" / city / f"{job}.json"
                command_manifest = output / "commands" / city / f"{job}.json"
                _run_job(
                    command,
                    raw,
                    command_manifest,
                    force=bool(args.force),
                    config_hash=manifest["config_hash"],
                )
                completed += 1
                _write_status(
                    output,
                    state="running",
                    completed_jobs=completed,
                    total_jobs=8,
                    current_city=city,
                    current_job=job,
                )
        cities = {
            city: summarize_city(config, output, city, customers=customers)
            for city in T1_CITIES
        }
        reality = {
            city: quote_reality_gap(config, city, rows=1000)
            for city in REALITY_CITIES
        }
        for city in T1_CITIES:
            cities[city]["T1.6"] = reality[city]
        report = {
            **manifest,
            "cities": cities,
            "reality_gap_all_cities": reality,
            "passed": all(result["required_pass"] for result in cities.values()),
        }
        atomic_json(output / "report.json", report)
        (output / "report.md").write_text(render_markdown(report), encoding="utf-8")
        _write_status(
            output,
            state="passed" if report["passed"] else "failed",
            completed_jobs=completed,
            total_jobs=8,
            report=str(output / "report.json"),
        )
        print(render_markdown(report))
        return 0 if report["passed"] else 1
    except Exception as exc:
        _write_status(
            output,
            state="error",
            completed_jobs=completed,
            total_jobs=8,
            error=repr(exc),
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
