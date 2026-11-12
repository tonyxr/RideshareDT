#!/usr/bin/env python3
"""Select registered V17 robustness comparators with cached Core sweeps."""

from __future__ import annotations

import argparse
import concurrent.futures
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence


BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from diagnostic_logging import atomic_json  # noqa: E402
from scripts.v17_config import ACTION_LEVELS, canonical_hash, load_config, validate_config  # noqa: E402
from scripts import v17_t2_sweep as t2  # noqa: E402
from scripts.v17_robust_common import check_overrides, robust_sources, set_option  # noqa: E402
from v17_surge_decay import scenario_windows, surge_decay_spec  # noqa: E402


RIVALS = (
    "adaptive_best_response_memory",
    "profit_best_response_myopic",
    "profit_opt_mpc_grid",
)
LEFTOUT_VARIANTS: dict[str, dict[str, Any]] = {
    "drivers_fixed": {
        "overrides": {"supply_eta_v3": 0.01, "shortage_willingness_share": 0.0},
        "rivals": RIVALS,
    },
    "no_loyalty": {"overrides": {"loyalty_model": "v1"}, "rivals": RIVALS},
    "static_rival": {"overrides": {}, "rivals": ("static",)},
    "no_spillover": {"overrides": {"disable_spillover": True}, "rivals": RIVALS},
    "thresholds_mean": {
        "overrides": {"threshold_ablation": "constant_mean"}, "rivals": RIVALS,
    },
    "thresholds_shuffled": {
        "overrides": {"threshold_ablation": "shuffled"}, "rivals": RIVALS,
    },
}
HOLD_DAYS = (0, 1, 2, 4)
DECAY_DAYS = (0, 2, 4, 6)
ETA_VALUES = (0.3, 0.6, 1.0)
OMEGA_VALUES = (0.25, 0.5, 0.75)
SCENARIO = BASE / "scenarios/v17_t2_shortage_severe.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    return t2._sha256_file(path)


def _selected_comparators(city: str) -> tuple[Path, dict[str, Any]]:
    path = BASE / "runs/v17_results_comparators" / city / "selected.json"
    return path, _load(path)


def _normal_cell(city: str) -> tuple[float, float]:
    _, selected = _selected_comparators(city)
    row = selected["best_fixed_cell"]
    if "distance_level" in row:
        return float(row["distance_level"]), float(row["time_level"])
    return t2.parse_cell(str(row["spec"]))


def _surge_payload(spec: str) -> dict[str, Any]:
    if not str(spec).startswith("surge_decay:"):
        raise ValueError(f"not a surge-decay spec: {spec}")
    return json.loads(str(spec).split(":", 1)[1])


def _spec_tiebreak(spec: str, normal: Sequence[float]) -> tuple[float, float, float]:
    if str(spec).startswith("static_tariff:"):
        distance, time = t2.parse_cell(spec)
        return 0.0, abs(distance - float(normal[0])) + abs(time - float(normal[1])), distance + time
    payload = _surge_payload(spec)
    event = payload["event"]
    return (
        float(payload.get("hold_days", 0)) + float(payload.get("decay_days", 0)),
        abs(float(event[0]) - float(normal[0])) + abs(float(event[1]) - float(normal[1])),
        float(event[0]) + float(event[1]),
    )


def _rank_specs(
    summaries: Mapping[str, Mapping[str, Any]], normal: Sequence[float]
) -> list[str]:
    return sorted(
        summaries,
        key=lambda spec: (
            -float(summaries[spec]["final_objective"]),
            *_spec_tiebreak(spec, normal),
            spec,
        ),
    )


def _summary_rows(products: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    pooled: dict[str, list[Mapping[str, Any]]] = {}
    for payload in products:
        for spec, openings in dict(payload["results"]).items():
            pooled.setdefault(str(spec), []).extend(dict(openings).values())
    summaries = {spec: t2._metric_mean(rows) for spec, rows in pooled.items()}
    if not summaries or any(not math.isfinite(float(row["final_objective"])) for row in summaries.values()):
        raise ValueError("selector produced no finite candidates")
    return summaries


def _run_stage(
    *, city: str, specs: Sequence[str], output: Path, overrides: Mapping[str, Any],
    workers: int, config: Mapping[str, Any], plan_only: bool,
    case: str = "A1_shortage_severe",
    rivals: Sequence[str] = RIVALS, openings: int = 3, days: int = 28,
    with_products: bool = False,
) -> Any:
    output = output.resolve()
    jobs: list[tuple[str, list[str], Path, Path, Path]] = []
    for rival in rivals:
        command_rival = rival if rival in t2.RIVALS else t2.RIVALS[0]
        command = t2.sweep_command(
            config, case=case, rival=command_rival, output=output,
            openings=openings, customers=50, pool=2000, city=city, days=days,
        )
        if rival != command_rival:
            command = set_option(command, "--firm2_mode", rival)
            command = set_option(command, "--policy_sweep_rivals", rival)
        command = set_option(command, "--policy_sweep", ";".join(specs))
        for key, value in overrides.items():
            normalized = str(key).lstrip("-").replace("-", "_")
            if normalized == "disable_spillover":
                command = [item for item in command if item != "--disable_spillover"]
                if bool(value):
                    command.append("--disable_spillover")
                continue
            flag = {
                "supply_eta_v3": "--supply_eta_v3",
                "shortage_willingness_share": "--shortage_willingness_share",
                "loyalty_model": "--loyalty_model",
                "threshold_ablation": "--threshold_ablation",
            }[normalized]
            command = set_option(command, flag, value)
        raw, manifest, log = t2._job_paths(output, case, rival)
        jobs.append((rival, command, raw, manifest, log))
    if plan_only:
        return ({}, []) if with_products else {}
    sources = robust_sources((t2.CASES[case],))
    config_hash = canonical_hash(config)

    def execute(job: tuple[str, list[str], Path, Path, Path]):
        rival, command, raw, manifest, log = job
        result = t2.run_job(
            command, raw=raw, manifest=manifest, log=log,
            config_hash=config_hash, sources=sources, force=False,
        )
        payload = _load(raw)
        check_overrides(payload, overrides)
        return rival, result, payload

    products: list[Mapping[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
        futures = [executor.submit(execute, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            rival, state, payload = future.result()
            print(f"[{city}] {rival}: {state}", flush=True)
            products.append(payload)
    summaries = _summary_rows(products)
    return (summaries, products) if with_products else summaries


def _grid_neighbors(event: Sequence[float], levels: Sequence[float]) -> tuple[tuple[float, float], ...]:
    d_index = min(range(len(levels)), key=lambda i: abs(levels[i] - float(event[0])))
    t_index = min(range(len(levels)), key=lambda i: abs(levels[i] - float(event[1])))
    neighbors: set[tuple[float, float]] = set()
    for di in (-1, 0, 1):
        for ti in (-1, 0, 1):
            if di == 0 and ti == 0:
                continue
            d = levels[min(max(0, d_index + di), len(levels) - 1)]
            t = levels[min(max(0, t_index + ti), len(levels) - 1)]
            if (d, t) != (levels[d_index], levels[t_index]):
                neighbors.add((float(d), float(t)))
    return tuple(sorted(neighbors))


def _candidate_row(spec: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
    if str(spec).startswith("surge_decay:"):
        payload = _surge_payload(spec)
        return {
            "event": [float(value) for value in payload["event"]],
            "hold_days": int(payload["hold_days"]),
            "decay_days": int(payload["decay_days"]),
            "spec": spec,
            **dict(metrics),
        }
    distance, time = t2.parse_cell(spec)
    return {
        "cell": [distance, time], "distance_level": distance,
        "time_level": time, "spec": spec, **dict(metrics),
    }


def _point_values(raw: str) -> tuple[tuple[float, float], ...]:
    if raw == "all":
        return tuple((eta, omega) for eta in ETA_VALUES for omega in OMEGA_VALUES)
    points: list[tuple[float, float]] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token.startswith("eta") or "_omega" not in token:
            raise ValueError(f"bad point id: {token}")
        eta_text, omega_text = token[3:].split("_omega", 1)
        points.append((float(eta_text), float(omega_text)))
    return tuple(points)


def _point_id(eta: float, omega: float) -> str:
    return f"eta{eta:g}_omega{omega:g}"


def _reactive_surge_spec(
    normal: Sequence[float], *, fill_ref: float, fill_gain: float,
    wait_ref: float, wait_gain: float, storm_gain: float,
    max_level: float = 2.5,
) -> str:
    return "reactive_surge:" + json.dumps({
        "normal": [float(normal[0]), float(normal[1])],
        "fill_ref": float(fill_ref), "fill_gain": float(fill_gain),
        "wait_ref": float(wait_ref), "wait_gain": float(wait_gain),
        "storm_gain": float(storm_gain), "max_level": float(max_level),
    }, sort_keys=True, separators=(",", ":"))


def _rule_wait_ref(city: str, normal: Sequence[float]) -> tuple[Path, float]:
    path = (
        BASE / "runs/v17_results_comparators" / city / "raw/C0_calm"
        / "adaptive_best_response_memory.json"
    )
    payload = _load(path)
    spec = t2.cell_spec(float(normal[0]), float(normal[1]))
    rows = list(dict(payload.get("traces") or {}).get(spec) or ())
    waits = [float(row["wait"]) for row in rows if math.isfinite(float(row.get("wait", float("nan"))))]
    if not waits:
        raise ValueError(f"no finite calm wait rows for {city} normal card")
    return path, float(sum(waits) / len(waits))


def _shortage_scores(
    products: Iterable[Mapping[str, Any]], specs: Sequence[str],
) -> dict[str, float]:
    """Score fixed cards on A1 days 10--15 using the registered shortage metric."""
    values: dict[str, list[float]] = {str(spec): [] for spec in specs}
    for payload in products:
        steps_per_day = int(payload.get("steps_per_day", 6))
        for spec, rows in dict(payload.get("traces") or {}).items():
            if spec not in values:
                continue
            for row in rows:
                day = int(row["step"]) // steps_per_day
                if 10 <= day <= 15:
                    score = (
                        float(row["profit_per_request"])
                        - 0.75 * float(row["rival_profit_per_request"])
                    )
                    if math.isfinite(score):
                        values[spec].append(score)
    if any(not rows for rows in values.values()):
        missing = sorted(spec for spec, rows in values.items() if not rows)
        raise ValueError(f"left-out selection lacks A1 day-10--15 rows: {missing[:5]}")
    return {spec: float(sum(rows) / len(rows)) for spec, rows in values.items()}


def _leftout_selection(
    *, city: str, variant: str, normal: Sequence[float],
    specs: Sequence[str], summaries: Mapping[str, Mapping[str, Any]],
    shortage_products: Iterable[Mapping[str, Any]],
    overrides: Mapping[str, Any], rivals: Sequence[str],
    input_hashes: Mapping[str, str],
) -> dict[str, Any]:
    shortage_scores = _shortage_scores(shortage_products, specs)
    best_fixed = _rank_specs(summaries, normal)[0]
    best_shortage = sorted(
        specs,
        key=lambda spec: (
            -shortage_scores[spec], *_spec_tiebreak(spec, normal), spec,
        ),
    )[0]
    candidates = {
        spec: {
            **dict(summaries[spec]),
            "shortage_score": shortage_scores[spec],
        }
        for spec in specs
    }
    return {
        "schema_version": "v17-leftout-selection-v1",
        "created_at_utc": _utc_now(),
        "city": city,
        "variant": variant,
        "overrides": dict(overrides),
        "rivals": list(rivals),
        "normal_cell": [float(normal[0]), float(normal[1])],
        "best_fixed_cell": _candidate_row(best_fixed, candidates[best_fixed]),
        "best_shortage_cell": _candidate_row(
            best_shortage, candidates[best_shortage]
        ),
        "candidates": candidates,
        "selected_sha256_inputs": dict(input_hashes),
    }


def _full_leftout_products(city: str) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], dict[str, str]]:
    products: list[Mapping[str, Any]] = []
    shortage_products: list[Mapping[str, Any]] = []
    hashes: dict[str, str] = {}
    for case in ("C0_calm", "A1_shortage_severe"):
        for rival in RIVALS:
            path = (
                BASE / "runs/v17_results_comparators" / city
                / "raw" / case / f"{rival}.json"
            )
            payload = _load(path)
            products.append(payload)
            if case == "A1_shortage_severe":
                shortage_products.append(payload)
            hashes[str(path.relative_to(BASE))] = _sha256_file(path)
    return products, shortage_products, hashes


def _run_city(args: argparse.Namespace, city: str, config: Mapping[str, Any]) -> dict[str, Any]:
    output = args.output.resolve() / city
    days = int(getattr(args, "days", 28))
    openings = int(getattr(args, "openings", 3))
    normal = _normal_cell(city)
    scenario = _load(SCENARIO)
    windows = scenario_windows(scenario)
    levels = tuple(float(value) for value in ACTION_LEVELS[str(config["action"]["contract"])])
    comparator_path, _ = _selected_comparators(city)
    base_meta = {
        "city": city, "validation_seed": t2.VALIDATION_SEED,
        "horizon_days": days, "openings": openings, "rivals": list(RIVALS),
        "tuning_case": "A1_shortage_severe", "windows": windows,
        "normal_cell": list(normal),
        "selected_sha256_inputs": {"comparators_selected.json": _sha256_file(comparator_path)},
    }
    if args.mode == "wp1a":
        specs = tuple(
            surge_decay_spec(normal, (distance, time), windows, 0, 0)
            for distance in levels for time in levels
        )
        if args.plan_only:
            return {**base_meta, "mode": args.mode, "candidates": len(specs), "jobs": 3}
        summaries = _run_stage(
            city=city, specs=specs, output=output, overrides={}, workers=args.workers,
            config=config, plan_only=False,
        )
        ranked = _rank_specs(summaries, normal)
        payload = {
            "schema_version": "v17-robust-surge-stage-a-v1", **base_meta,
            "stage_a_top3": [_surge_payload(spec)["event"] for spec in ranked[:3]],
            "candidates": {spec: dict(summaries[spec]) for spec in ranked},
        }
        atomic_json(output / "selected_stage_a.json", payload)
        return payload
    if args.mode == "wp1b":
        if args.source is None:
            raise ValueError("wp1b requires --from")
        source = args.source.resolve() / city / "selected_stage_a.json"
        if not source.is_file() and (args.source / "selected_stage_a.json").is_file():
            source = args.source / "selected_stage_a.json"
        stage_a = _load(source)
        top3 = tuple(tuple(float(v) for v in pair) for pair in stage_a["stage_a_top3"])
        specs = tuple(
            surge_decay_spec(normal, event, windows, hold, decay)
            for event in top3 for hold in HOLD_DAYS for decay in DECAY_DAYS
        )
        if args.plan_only:
            return {**base_meta, "mode": args.mode, "candidates": len(specs), "jobs": 3}
        summaries = _run_stage(
            city=city, specs=specs, output=output, overrides={}, workers=args.workers,
            config=config, plan_only=False,
        )
        ranked = _rank_specs(summaries, normal)
        stage_a_summaries = dict(stage_a["candidates"])
        two_spec = _rank_specs(stage_a_summaries, normal)[0]
        decay_spec = ranked[0]
        payload = {
            "schema_version": "v17-robust-surge-selection-v1", **base_meta,
            "surge_two_lever": _candidate_row(two_spec, stage_a_summaries[two_spec]),
            "surge_decay": _candidate_row(decay_spec, summaries[decay_spec]),
            "stage_a_top3": [list(pair) for pair in top3],
            "candidates": {spec: dict(summaries[spec]) for spec in ranked},
        }
        atomic_json(output / "selected_surge.json", payload)
        canonical = BASE / "runs/v17_robust/select/wp1" / city / "selected_surge.json"
        atomic_json(canonical, payload)
        return payload
    if args.mode == "rule":
        wait_path, wait_ref = _rule_wait_ref(city, normal)
        specs = tuple(
            _reactive_surge_spec(
                normal, fill_ref=fill_ref, fill_gain=fill_gain,
                wait_ref=wait_ref, wait_gain=wait_gain,
                storm_gain=storm_gain,
            )
            for fill_ref in (0.90, 0.95)
            for fill_gain in (1.0, 2.0, 4.0)
            for wait_gain in (0.0, 0.5)
            for storm_gain in (0.0, 0.25)
        )
        if args.plan_only:
            return {
                **base_meta, "mode": args.mode, "candidates": len(specs),
                "jobs": 6, "wait_ref": wait_ref,
            }
        per_case_workers = max(1, args.workers // 2)
        case_summaries: dict[str, dict[str, dict[str, float]]] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_cases = {
                executor.submit(
                    _run_stage, city=city, specs=specs, output=output,
                    overrides={}, workers=per_case_workers, config=config,
                    plan_only=False, case=case,
                ): case
                for case in ("C0_calm", "A1_shortage_severe")
            }
            for future in concurrent.futures.as_completed(future_cases):
                case_summaries[future_cases[future]] = future.result()
        candidates: dict[str, dict[str, Any]] = {}
        for spec in specs:
            params = json.loads(spec.split(":", 1)[1])
            per_case = {
                case: dict(case_summaries[case][spec])
                for case in ("C0_calm", "A1_shortage_severe")
            }
            candidates[spec] = {
                "parameters": params,
                "per_case": per_case,
                "final_objective": float(sum(
                    row["final_objective"] for row in per_case.values()
                ) / len(per_case)),
            }
        ranked = sorted(candidates, key=lambda spec: (
            -float(candidates[spec]["final_objective"]),
            float(candidates[spec]["parameters"]["fill_gain"]), spec,
        ))
        payload = {
            "schema_version": "v17-robust-reactive-rule-selection-v1",
            **base_meta, "mode": args.mode, "wait_ref": wait_ref,
            "wait_ref_source": str(wait_path),
            "reactive_surge": {"spec": ranked[0], **candidates[ranked[0]]},
            "candidates": {spec: candidates[spec] for spec in ranked},
        }
        atomic_json(output / "selected_rule.json", payload)
        return payload
    if args.mode == "leftout":
        variant = str(args.variant or "")
        if variant not in LEFTOUT_VARIANTS:
            raise ValueError(
                "leftout mode requires --variant in "
                + ", ".join(sorted(LEFTOUT_VARIANTS))
            )
        settings = LEFTOUT_VARIANTS[variant]
        overrides = dict(settings["overrides"])
        rivals = tuple(settings["rivals"])
        specs = tuple(t2.fixed_specs(config))
        variant_output = output / variant
        if args.plan_only:
            return {
                **base_meta,
                "mode": args.mode,
                "variant": variant,
                "overrides": overrides,
                "rivals": list(rivals),
                "candidates": len(specs),
                "jobs": 2 * len(rivals),
            }
        per_case_workers = max(1, args.workers // 2)
        case_results: dict[str, tuple[dict[str, dict[str, float]], list[Mapping[str, Any]]]] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_cases = {
                executor.submit(
                    _run_stage,
                    city=city,
                    specs=specs,
                    output=variant_output,
                    overrides=overrides,
                    workers=per_case_workers,
                    config=config,
                    plan_only=False,
                    case=case,
                    rivals=rivals,
                    openings=openings,
                    days=days,
                    with_products=True,
                ): case
                for case in ("C0_calm", "A1_shortage_severe")
            }
            for future in concurrent.futures.as_completed(future_cases):
                case_results[future_cases[future]] = future.result()
        products = [
            payload
            for _summaries, case_products in case_results.values()
            for payload in case_products
        ]
        summaries = _summary_rows(products)
        shortage_products = case_results["A1_shortage_severe"][1]
        input_hashes = {
            str(path.relative_to(BASE)): _sha256_file(path)
            for path in sorted(variant_output.glob("raw/*/*.json"))
        }
        payload = _leftout_selection(
            city=city,
            variant=variant,
            normal=normal,
            specs=specs,
            summaries=summaries,
            shortage_products=shortage_products,
            overrides=overrides,
            rivals=rivals,
            input_hashes=input_hashes,
        )
        atomic_json(variant_output / "selected_leftout.json", payload)

        full_products, full_shortage, full_hashes = _full_leftout_products(city)
        full_payload = _leftout_selection(
            city=city,
            variant="full",
            normal=normal,
            specs=specs,
            summaries=_summary_rows(full_products),
            shortage_products=full_shortage,
            overrides={},
            rivals=RIVALS,
            input_hashes=full_hashes,
        )
        atomic_json(output / "full" / "selected_leftout.json", full_payload)
        return payload
    if args.mode == "point":
        if args.source is None:
            raise ValueError("point mode requires --from selected_surge root/file")
        source = args.source.resolve()
        if source.is_dir():
            candidates = (source / city / "selected_surge.json", source / "selected_surge.json")
            source = next((path for path in candidates if path.is_file()), candidates[0])
        selected_surge = _load(source)
        winner = selected_surge["surge_decay"]
        event = tuple(float(v) for v in winner["event"])
        hold = int(winner["hold_days"])
        decay = int(winner["decay_days"])
        fixed_specs = tuple(t2.cell_spec(d, t) for d in levels for t in levels)
        base_surge = tuple(
            surge_decay_spec(normal, event, windows, h, d)
            for h in HOLD_DAYS for d in DECAY_DAYS
        )
        neighbor_specs = tuple(
            surge_decay_spec(normal, neighbor, windows, hold, decay)
            for neighbor in _grid_neighbors(event, levels)
        )
        specs = tuple(dict.fromkeys((*fixed_specs, *base_surge, *neighbor_specs)))
        point_payloads: dict[str, Any] = {}
        for eta, omega in _point_values(args.points):
            point = _point_id(eta, omega)
            point_output = output / point
            overrides = {
                "supply_eta_v3": eta,
                "shortage_willingness_share": omega,
            }
            if args.plan_only:
                point_payloads[point] = {"candidates": len(specs), "jobs": 3, "overrides": overrides}
                continue
            summaries = _run_stage(
                city=city, specs=specs, output=point_output, overrides=overrides,
                workers=args.workers, config=config, plan_only=False,
            )
            fixed = {spec: summaries[spec] for spec in fixed_specs}
            surges = {spec: summaries[spec] for spec in specs if spec not in fixed}
            best_fixed = _rank_specs(fixed, normal)[0]
            best_surge = _rank_specs(surges, normal)[0]
            payload = {
                "schema_version": "v17-robust-point-selection-v1", **base_meta,
                "point": point, "overrides": overrides,
                "best_fixed_point": _candidate_row(best_fixed, summaries[best_fixed]),
                "surge_decay_point": _candidate_row(best_surge, summaries[best_surge]),
                "candidates": {spec: dict(summaries[spec]) for spec in _rank_specs(summaries, normal)},
                "selected_surge_sha256": _sha256_file(source),
            }
            atomic_json(point_output / "selected_point.json", payload)
            point_payloads[point] = payload
        return {**base_meta, "mode": args.mode, "points": point_payloads}
    raise AssertionError(args.mode)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", required=True,
        choices=("wp1a", "wp1b", "point", "rule", "leftout")
    )
    parser.add_argument("--city", required=True, help="Comma-separated city slugs")
    parser.add_argument("--points", default="all")
    parser.add_argument("--variant", choices=tuple(sorted(LEFTOUT_VARIANTS)))
    parser.add_argument("--days", type=int, default=28)
    parser.add_argument("--openings", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--from", dest="source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status-name", default="status.json")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 8:
        parser.error("--workers must be in [1, 8]")
    if args.days < 1 or args.openings < 1:
        parser.error("--days and --openings must be positive")
    cities = tuple(token.strip() for token in args.city.split(",") if token.strip())
    unknown = sorted(set(cities) - {"nyc", "boston", "chicago"})
    if unknown:
        parser.error(f"unknown city: {', '.join(unknown)}")
    if Path(args.status_name).name != args.status_name:
        parser.error("--status-name must be a file name, not a path")
    status_path = args.output.resolve() / args.status_name
    config = load_config()
    validate_config(config)
    summary = {
        "schema_version": "v17-robust-selector-status-v1",
        "created_at_utc": _utc_now(), "mode": args.mode,
        "state": "planned" if args.plan_only else "running",
        "workers": args.workers, "cities": list(cities), "results": {},
    }
    if not args.plan_only:
        atomic_json(status_path, summary)
    try:
        for city in cities:
            summary["results"][city] = _run_city(args, city, config)
        summary["state"] = "planned" if args.plan_only else "completed"
        summary["pass"] = not args.plan_only
        if args.plan_only:
            print(json.dumps(summary, indent=2, sort_keys=True))
        else:
            atomic_json(status_path, summary)
        return 0
    except BaseException as exc:
        summary.update(state="failed", pass_=False, error=repr(exc))
        if not args.plan_only:
            summary["pass"] = False
            summary.pop("pass_", None)
            atomic_json(status_path, summary)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
