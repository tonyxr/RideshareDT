#!/usr/bin/env python3
"""Build V18 reference candidates and select them from saved landscapes."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


BASE = Path(__file__).resolve().parents[1]
LEVELS = (0.8, 1.0, 1.25, 1.5, 2.0, 2.5)
SELECTION_RIVALS = (
    "adaptive_best_response_memory", "adaptive_best_response_aggressive",
    "profit_best_response_myopic", "profit_best_response_forward",
    "profit_best_response_anticipating", "profit_opt_mpc_grid",
)
CASES = ("C0_calm", "A1_shortage_severe", "v17_shortage_i065")


def cell_spec(distance: float, time: float) -> str:
    return "cell:" + json.dumps([float(distance), float(time)], separators=(",", ":"))


def card_grid() -> list[dict[str, Any]]:
    return [
        {"distance": distance, "time": time, "spec": cell_spec(distance, time)}
        for distance in LEVELS for time in LEVELS
    ]


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", choices=("nyc", "boston", "chicago"), required=True)
    parser.add_argument("--landscape", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--emit-plan", action="store_true")
    args = parser.parse_args()
    output = args.output or BASE / f"runs/v18/select/{args.city}/selected.json"
    if args.emit_plan:
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
