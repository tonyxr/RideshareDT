#!/usr/bin/env python3
"""Build V18 reference candidates and select them from saved landscapes."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from scripts import v17_t2_sweep as t2
from scripts.v17_config import load_config as load_v17_config
from scripts.v17_robust_common import set_option
from scripts.v18_config import load_config as load_v18_config, validate_config
from scripts.v18_market import market_flags
from scripts.v18_runtime import Job, run_jobs
from v17_surge_decay import scenario_windows, surge_decay_spec
LEVELS = (0.8, 1.0, 1.25, 1.5, 2.0, 2.5)
SELECTION_RIVALS = (
    "adaptive_best_response_memory", "adaptive_best_response_aggressive",
    "profit_best_response_myopic", "profit_best_response_forward",
    "profit_best_response_anticipating", "profit_opt_mpc_grid",
)
CASES = ("C0_calm", "A1_shortage_severe", "v17_shortage_i065")
STAGE3_RIVALS = (
    "adaptive_best_response_memory",
    "profit_best_response_myopic",
    "profit_opt_mpc_grid",
)


def cell_spec(distance: float, time: float) -> str:
    return t2.cell_spec(float(distance), float(time))


def card_grid() -> list[dict[str, Any]]:
    return [
        {"distance": distance, "time": time, "spec": cell_spec(distance, time)}
        for distance in LEVELS for time in LEVELS
    ]


def _grid_edge_selected(best_row: Mapping[str, Any], event: Sequence[float]) -> bool:
    return 2.5 in (
        float(best_row["distance"]), float(best_row["time"]), *map(float, event)
    )


def count_rule_candidates(
    *, normal: Sequence[float], event_cards: Sequence[Sequence[float]],
    norm: Sequence[float],
) -> list[str]:
    values: list[str] = []
    for event in event_cards:
        for move_limit in (0, 1):
            for theta in (0.5, 0.6, 0.7, 0.8, 0.9):
                payload = {"normal": list(normal), "event": list(event),
                           "norm": list(norm), "theta": theta, "mode": "step",
                           "theta_lo": 0.5, "move_limit": move_limit}
                values.append("count_rule:" + json.dumps(payload, sort_keys=True, separators=(",", ":")))
            for theta_lo in (0.4, 0.6):
                payload = {"normal": list(normal), "event": list(event),
                           "norm": list(norm), "theta": 0.9, "mode": "ramp",
                           "theta_lo": theta_lo, "move_limit": move_limit}
                values.append("count_rule:" + json.dumps(payload, sort_keys=True, separators=(",", ":")))
    if len(values) != 84:
        raise RuntimeError("registered count-rule grid must contain 84 candidates")
    return values


def reactive_candidates(normal: Sequence[float]) -> list[str]:
    values: list[str] = []
    for fill_gain in (1.0, 2.0):
        for wait_gain in (0.25, 0.5):
            for storm_gain in (0.0, 0.25):
                for max_level in (1.5, 2.0, 2.5):
                    payload = {"normal": list(normal), "fill_ref": 0.9,
                               "fill_gain": fill_gain, "wait_ref": 8.0,
                               "wait_gain": wait_gain, "storm_gain": storm_gain,
                               "max_level": max_level}
                    values.append("reactive_surge:" + json.dumps(payload, sort_keys=True, separators=(",", ":")))
    if len(values) != 24:
        raise RuntimeError("registered reactive-surge grid must contain 24 candidates")
    return values


def _mean(rows: Sequence[Mapping[str, Any]], spec: str, cases: Sequence[str]) -> float:
    values = [float(row["objective"]) for row in rows
              if row.get("spec") == spec and row.get("case") in cases]
    return float(sum(values) / len(values)) if values else -math.inf


def select(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    calm = [row for row in rows if row.get("case") == "C0_calm" and row.get("kind") == "card"]
    shortage = [row for row in rows if row.get("case") == "A1_shortage_severe" and row.get("kind") == "card"]
    if len({row.get("spec") for row in calm}) != 36 or len({row.get("spec") for row in shortage}) != 36:
        raise ValueError("selection needs all 36 cards in both C0 and A1")
    best_fixed = max(calm, key=lambda row: (float(row["objective"]), str(row["spec"])))
    event = max(shortage, key=lambda row: (float(row["objective"]), str(row["spec"])))
    rules = [row for row in rows if row.get("kind") in {"count_rule", "reactive_surge"}]
    selected_rules: dict[str, Any] = {}
    for kind in ("count_rule", "reactive_surge"):
        candidates = {str(row["spec"]) for row in rules if row.get("kind") == kind}
        if kind == "count_rule":
            for move_limit in (0, 1):
                subset = [spec for spec in candidates if f'"move_limit":{move_limit}' in spec]
                if subset:
                    winner = max(subset, key=lambda spec: (_mean(rules, spec, CASES), spec))
                    selected_rules[f"{kind}_{move_limit}step"] = {
                        "spec": winner, "objective": _mean(rules, winner, CASES)
                    }
        elif candidates:
            winner = max(candidates, key=lambda spec: (_mean(rules, spec, CASES), spec))
            selected_rules[kind] = {"spec": winner, "objective": _mean(rules, winner, CASES)}
    edge = any(
        2.5 in (float(row.get("distance", -1)), float(row.get("time", -1)))
        for row in (best_fixed, event)
    )
    return {
        "schema_version": "v18-reference-selection-v1",
        "best_fixed": dict(best_fixed),
        "dated_two_lever": {
            "normal_spec": best_fixed["spec"], "event_spec": event["spec"],
            "spec": "surge_decay:" + json.dumps({
                "normal": [best_fixed["distance"], best_fixed["time"]],
                "event": [event["distance"], event["time"]],
                "windows": "realized", "hold_days": 0, "decay_days": 0,
            }, sort_keys=True, separators=(",", ":")),
        },
        **selected_rules,
        "published_card": {"spec": cell_spec(1.0, 1.0)},
        "grid_edge_selected": bool(edge),
        "landscape": [dict(row) for row in rows],
    }


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _command(
    config: Mapping[str, Any], *, city: str, case: str,
    scenario: Path, specs: Sequence[str], rivals: Sequence[str],
    openings: int, output: Path, variant: str = "production",
    customers: int | None = None, seed: int = 20_231,
    overrides: Mapping[str, Any] | None = None,
) -> list[str]:
    n_star = int(customers or config["decisions"]["N_star"])
    base = t2.sweep_command(
        load_v17_config(), city=city, case="C0_calm",
        rival=str(rivals[0]) if str(rivals[0]) in t2.RIVALS else str(t2.RIVALS[0]), output=output.parent,
        openings=openings, customers=n_star, pool=20_000, days=28,
    )
    values, _ = market_flags(
        base, variant=variant, city=city, customers=n_star,
    )
    for flag, value in (
        ("--scenario_hazards", "off"),
        ("--seed", seed), ("--experiment_seed", seed),
        ("--policy_sweep_seed", seed),
        ("--shock_scenario", scenario.resolve()),
        ("--policy_sweep", ";".join(specs)),
        ("--policy_sweep_rivals", ",".join(rivals)),
        ("--policy_sweep_openings", openings),
        ("--policy_sweep_out", output.resolve()),
        ("--report_prefix", output.parent / "diagnostics"),
    ):
        values = set_option(values, flag, value)
    if tuple(rivals) == ("static",):
        values = set_option(values, "--firm2_mode", "static")
    for key, value in dict(overrides or {}).items():
        if key == "disable_spillover":
            values = [item for item in values if item != "--disable_spillover"]
            if value:
                values.append("--disable_spillover")
        else:
            values = set_option(values, "--" + key, value)
    return values


def _objective(payload: Mapping[str, Any], spec: str) -> float:
    rows = dict(payload["results"])[spec]
    return float(sum(float(row["objective"]) for row in rows.values()) / len(rows))


def _drivers_by_block(payload: Mapping[str, Any], spec: str) -> list[float]:
    groups: dict[int, list[float]] = {block: [] for block in range(6)}
    for row in payload["traces"][spec]:
        day = int(row["step"]) // 6 + 1
        if 5 <= day <= 27:
            groups[int(row["hour"]) // 4].append(float(row["active_drivers_firm1"]))
    if any(not values for values in groups.values()):
        raise ValueError("published-card driver norm is missing a four-hour block")
    return [sum(groups[block]) / len(groups[block]) for block in range(6)]


def _stage3_count_rules(
    normal: Sequence[float], event: Sequence[float], norm: Sequence[float],
) -> list[str]:
    result: list[str] = []
    for move_limit in (0, 1):
        for theta in (0.5, 0.6, 0.7, 0.8, 0.9):
            payload = {
                "normal": list(normal), "event": list(event), "norm": list(norm),
                "theta": theta, "mode": "step", "theta_lo": 0.5,
                "move_limit": move_limit,
            }
            result.append("count_rule:" + json.dumps(payload, sort_keys=True, separators=(",", ":")))
        for theta_lo in (0.4, 0.6):
            payload = {
                "normal": list(normal), "event": list(event), "norm": list(norm),
                "theta": 0.9, "mode": "ramp", "theta_lo": theta_lo,
                "move_limit": move_limit,
            }
            result.append("count_rule:" + json.dumps(payload, sort_keys=True, separators=(",", ":")))
    if len(result) != 14:
        raise AssertionError("Last Revision count-rule grid must contain 14 candidates")
    return result


def run_selection(
    config: Mapping[str, Any], *, city: str, output: Path, workers: int,
    variant: str = "production", customers: int | None = None,
    seed: int = 20_231, overrides: Mapping[str, Any] | None = None,
    selection_rivals: Sequence[str] = SELECTION_RIVALS,
    selection_openings: int = 2, stage3: bool = True,
) -> dict[str, Any]:
    validate_config(config, require_frozen=True)
    output.mkdir(parents=True, exist_ok=True)
    cards = card_grid()
    card_specs = tuple(row["spec"] for row in cards)
    published = cell_spec(1.0, 1.0)
    stage1_raw = output / "stage1_calm_cards.json"
    stage1 = Job(
        name=f"selection/{city}/stage1",
        command=tuple(_command(
            config, city=city, case="C0_calm",
            scenario=BASE / "scenarios/calm.json", specs=card_specs,
            rivals=selection_rivals, openings=selection_openings, output=stage1_raw,
            variant=variant, customers=customers, seed=seed,
            overrides=overrides,
        )),
        output=stage1_raw, log=output / "stage1.log",
        manifest=output / "stage1_manifest.json",
    )
    if run_jobs((stage1,), cwd=BASE, workers=1)[stage1.name] not in {"complete", "cached"}:
        raise RuntimeError("selection stage 1 failed")
    p1 = json.loads(stage1_raw.read_text(encoding="utf-8"))
    scored = [
        (row, _objective(p1, row["spec"])) for row in cards
    ]
    best_row, best_objective = max(
        scored,
        key=lambda item: (
            item[1], -(float(item[0]["distance"]) + float(item[0]["time"])),
            -float(item[0]["distance"]),
        ),
    )
    published_drivers = _drivers_by_block(p1, published)

    shortage_path = BASE / "scenarios/v17_t2_shortage_severe.json"
    windows = scenario_windows(json.loads(shortage_path.read_text(encoding="utf-8")))
    dated_specs = tuple(
        surge_decay_spec(
            (best_row["distance"], best_row["time"]),
            (row["distance"], row["time"]), windows, 0, 0,
        )
        for row in cards
    )
    stage2_raw = output / "stage2_dated_cards.json"
    stage2 = Job(
        name=f"selection/{city}/stage2",
        command=tuple(_command(
            config, city=city, case="A1_shortage_severe",
            scenario=shortage_path, specs=dated_specs,
            rivals=selection_rivals, openings=selection_openings, output=stage2_raw,
            variant=variant, customers=customers, seed=seed,
            overrides=overrides,
        )), output=stage2_raw, log=output / "stage2.log",
        manifest=output / "stage2_manifest.json",
    )
    if run_jobs((stage2,), cwd=BASE, workers=1)[stage2.name] not in {"complete", "cached"}:
        raise RuntimeError("selection stage 2 failed")
    p2 = json.loads(stage2_raw.read_text(encoding="utf-8"))
    dated_spec = max(dated_specs, key=lambda spec: (_objective(p2, spec), spec))
    dated_table = json.loads(dated_spec.split(":", 1)[1])
    event = list(dated_table["event"])
    normal = [float(best_row["distance"]), float(best_row["time"])]
    norm = [published_drivers[(block - 1) % 6] for block in range(6)]
    result = {
        "schema_version": "v18-reference-selection-last-revision-v1",
        "city": city, "variant": variant,
        "published_card": {"spec": published},
        "best_fixed": {**best_row, "objective": best_objective},
        "dated_two_lever": {"spec": dated_spec, "objective": _objective(p2, dated_spec)},
        "published_drivers_by_block": published_drivers,
        "count_norm_shifted": norm,
        "grid_edge_selected": _grid_edge_selected(best_row, event),
        "landscapes": {"stage1": str(stage1_raw), "stage2": str(stage2_raw)},
    }
    if not stage3:
        atomic_json(output / "selected.json", result)
        return result
    count_specs = _stage3_count_rules(normal, event, norm)
    reactive_specs = reactive_candidates(normal)
    all_stage3 = tuple(count_specs + reactive_specs)
    case_paths = {
        "C0_calm": BASE / "scenarios/calm.json",
        "A1_shortage_severe": shortage_path,
        "v17_shortage_i065": BASE / "scenarios/cases/v17_shortage_i065.json",
    }
    stage3_jobs: list[Job] = []
    for case, scenario in case_paths.items():
        raw = output / f"stage3_{case}.json"
        stage3_jobs.append(Job(
            name=f"selection/{city}/stage3/{case}",
            command=tuple(_command(
                config, city=city, case=case, scenario=scenario,
                specs=all_stage3, rivals=STAGE3_RIVALS, openings=2, output=raw,
                variant=variant, customers=customers, seed=seed,
                overrides=overrides,
            )), output=raw, log=output / f"stage3_{case}.log",
            manifest=output / f"stage3_{case}_manifest.json",
        ))
    states = run_jobs(stage3_jobs, cwd=BASE, workers=min(workers, len(stage3_jobs)))
    if any(state not in {"complete", "cached"} for state in states.values()):
        raise RuntimeError("selection stage 3 failed")
    stage3_payloads = [json.loads(job.output.read_text(encoding="utf-8")) for job in stage3_jobs]
    mean_objective = lambda spec: sum(_objective(payload, spec) for payload in stage3_payloads) / len(stage3_payloads)
    count_selected = {}
    for move_limit in (0, 1):
        candidates = [spec for spec in count_specs if f'"move_limit":{move_limit}' in spec]
        winner = max(candidates, key=lambda spec: (mean_objective(spec), spec))
        count_selected[f"count_rule_{move_limit}step"] = {
            "spec": winner, "objective": mean_objective(winner),
        }
    reactive = max(reactive_specs, key=lambda spec: (mean_objective(spec), spec))
    result.update({
        **count_selected,
        "reactive_surge": {"spec": reactive, "objective": mean_objective(reactive)},
    })
    result["landscapes"]["stage3"] = [str(job.output) for job in stage3_jobs]
    atomic_json(output / "selected.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", choices=("nyc", "boston", "chicago"), required=True)
    parser.add_argument("--landscape", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--emit-plan", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--variant", choices=("production", "s1", "s2", "s3"), default="production")
    parser.add_argument("--customers", type=int)
    parser.add_argument("--seed", type=int, default=20_231)
    parser.add_argument("--config", type=Path, default=BASE / "configs/v18_production.yaml")
    args = parser.parse_args()
    output = args.output or BASE / f"runs/v18/select/{args.city}/selected.json"
    if args.run:
        payload = run_selection(
            load_v18_config(args.config), city=args.city,
            output=output.parent, workers=args.workers, variant=args.variant,
            customers=args.customers, seed=args.seed,
        )
    elif args.emit_plan:
        payload = {"cards": card_grid(), "selection_rivals": SELECTION_RIVALS,
                   "cases": CASES, "seed": 20231, "days": 28, "openings": 3}
    else:
        if args.landscape is None:
            raise ValueError("--landscape is required unless --emit-plan is used")
        payload = select(json.loads(args.landscape.read_text(encoding="utf-8")))
    atomic_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
