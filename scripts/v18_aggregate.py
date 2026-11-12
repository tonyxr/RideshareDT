#!/usr/bin/env python3
"""Aggregate V18 policy-sweep traces to daily and rollout CSV products."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping


METRICS = (
    "profit_per_request", "rival_profit_per_request", "own_share",
    "rival_share", "mean_fare_paid", "earnings_online_hour_firm1",
    "fulfillment", "wait", "reward",
)


def rows_from_product(path: Path, steps_per_day: int = 6) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    output: list[dict[str, Any]] = []
    for policy, traces in dict(payload.get("traces") or {}).items():
        for row in traces:
            value = dict(row)
            value.update({
                "source": str(path), "policy": str(policy),
                "day": int(value.get("step", 0)) // int(steps_per_day),
                "retaliation_flag": int(bool(value.get("rival_memory_state"))),
            })
            output.append(value)
    return output


def aggregate(rows: Iterable[Mapping[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(key) for key in keys), []).append(row)
    result: list[dict[str, Any]] = []
    for identity, group in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        record = dict(zip(keys, identity))
        record["steps"] = len(group)
        for metric in METRICS:
            values = [float(row[metric]) for row in group if row.get(metric) is not None]
            record[metric] = mean(values) if values else float("nan")
        record["retaliation_share"] = mean(float(row.get("retaliation_flag", 0)) for row in group)
        result.append(record)
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/v18/results"))
    parser.add_argument("--steps-per-day", type=int, default=6)
    args = parser.parse_args()
    rows = [row for path in args.inputs for row in rows_from_product(path, args.steps_per_day)]
    daily = aggregate(rows, ("source", "policy", "rollout", "day"))
    rollouts = aggregate(rows, ("source", "policy", "rollout"))
    write_csv(args.output / "daily.csv", daily)
    write_csv(args.output / "rollouts.csv", rollouts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
