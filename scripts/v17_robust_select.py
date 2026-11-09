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
) -> dict[str, dict[str, float]]:
    output = output.resolve()
    jobs: list[tuple[str, list[str], Path, Path, Path]] = []
    for rival in RIVALS:
        command = t2.sweep_command(
            config, case="A1_shortage_severe", rival=rival, output=output,
            openings=3, customers=50, pool=2000, city=city, days=28,
        )
        command = set_option(command, "--policy_sweep", ";".join(specs))
        for key, value in overrides.items():
            flag = {
                "supply_eta_v3": "--supply_eta_v3",
                "shortage_willingness_share": "--shortage_willingness_share",
            }[key]
            command = set_option(command, flag, value)
        raw, manifest, log = t2._job_paths(output, "A1_shortage_severe", rival)
        jobs.append((rival, command, raw, manifest, log))
    if plan_only:
        return {}
    sources = robust_sources((SCENARIO,))
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
    return _summary_rows(products)


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


def _run_city(args: argparse.Namespace, city: str, config: Mapping[str, Any]) -> dict[str, Any]:
    output = args.output.resolve() / city
    normal = _normal_cell(city)
    scenario = _load(SCENARIO)
    windows = scenario_windows(scenario)
    levels = tuple(float(value) for value in ACTION_LEVELS[str(config["action"]["contract"])])
    comparator_path, _ = _selected_comparators(city)
    base_meta = {
        "city": city, "validation_seed": t2.VALIDATION_SEED,
        "horizon_days": 28, "openings": 3, "rivals": list(RIVALS),
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
    parser.add_argument("--mode", required=True, choices=("wp1a", "wp1b", "point"))
    parser.add_argument("--city", required=True, help="Comma-separated city slugs")
    parser.add_argument("--points", default="all")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--from", dest="source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 8:
        parser.error("--workers must be in [1, 8]")
    cities = tuple(token.strip() for token in args.city.split(",") if token.strip())
    unknown = sorted(set(cities) - {"nyc", "boston", "chicago"})
    if unknown:
        parser.error(f"unknown city: {', '.join(unknown)}")
    config = load_config()
    validate_config(config)
    summary = {
        "schema_version": "v17-robust-selector-status-v1",
        "created_at_utc": _utc_now(), "mode": args.mode,
        "state": "planned" if args.plan_only else "running",
        "workers": args.workers, "cities": list(cities), "results": {},
    }
    if not args.plan_only:
        atomic_json(args.output.resolve() / "status.json", summary)
    try:
        for city in cities:
            summary["results"][city] = _run_city(args, city, config)
        summary["state"] = "planned" if args.plan_only else "completed"
        summary["pass"] = not args.plan_only
        if args.plan_only:
            print(json.dumps(summary, indent=2, sort_keys=True))
        else:
            atomic_json(args.output.resolve() / "status.json", summary)
        return 0
    except BaseException as exc:
        summary.update(state="failed", pass_=False, error=repr(exc))
        if not args.plan_only:
            summary["pass"] = False
            summary.pop("pass_", None)
            atomic_json(args.output.resolve() / "status.json", summary)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
