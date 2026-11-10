#!/usr/bin/env python3
"""Compute registered V17 training-cost accounting for robustness WP4."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from diagnostic_logging import atomic_json  # noqa: E402
from scripts.v17_results_train import RUN_SPECS  # noqa: E402
from Core import Core  # noqa: E402,F401


COLUMNS = (
    "run", "kind", "city", "wall_hours", "decisions", "market_days",
    "branch_rollouts", "branch_days", "nmcc_hours", "rival_planner_hours",
    "ppo_updates", "actor_steps", "value_fit_steps",
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _float(row: Mapping[str, Any], key: str) -> float:
    text = str(row.get(key, "") or "").strip()
    value = float(text) if text else 0.0
    if not math.isfinite(value):
        raise ValueError(f"non-finite {key}")
    return value


def _events_wall_hours(path: Path) -> float:
    started = None
    completed = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("phase") == "started" and started is None:
                started = datetime.fromisoformat(str(row["time_utc"]).replace("Z", "+00:00"))
            if row.get("phase") == "simulation_complete":
                completed = datetime.fromisoformat(str(row["time_utc"]).replace("Z", "+00:00"))
    if started is None or completed is None or completed <= started:
        raise ValueError(f"events do not bound a completed simulation: {path}")
    return (completed - started).total_seconds() / 3600.0


def _branch_counts(path: Path) -> tuple[int, int]:
    import torch

    artifact = torch.load(path, map_location="cpu", weights_only=False)
    state = artifact.get("v16_branch_store_state") or {}
    if not state:
        core = artifact.get("core")
        store = getattr(core, "v16_branch_store", None)
        if store is not None:
            state = store.state_dict()
    rows = list(state.get("rows") or ())
    rollouts = 0
    days = 0
    for row in rows:
        count = sum(bool(value) for value in row.get("action_credit_mask", ()))
        if bool(row.get("second_tape")):
            count *= 2
        horizon = int(row.get("horizon_days", 5) or 5)
        rollouts += count
        days += count * horizon
    return rollouts, days


def _diagnostic_path(status: Mapping[str, Any], city: str) -> Path:
    prefix = Path(str(status["diagnostic_prefix"]))
    return prefix.with_name(f"{prefix.name}_training_diagnostics.csv")


def _event_path(status: Mapping[str, Any], city: str) -> Path:
    prefix = Path(str(status["diagnostic_prefix"]))
    return prefix.with_name(f"{prefix.name}_events.jsonl")


def accounting_row(root: Path, spec: Any) -> dict[str, Any]:
    run_dir = root / spec.name
    status = _load(run_dir / "status.json")
    if status.get("state") != "completed" or int(status.get("days_expected", -1)) != 1200:
        raise ValueError(f"run is not a completed 1200-day product: {spec.name}")
    diagnostics = _diagnostic_path(status, spec.city)
    with diagnostics.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1200:
        raise ValueError(f"expected 1200 diagnostic rows: {spec.name}")
    last = rows[-1]
    branch_rollouts, branch_days = _branch_counts(run_dir / "resume/training_state.pt")
    if spec.kind != "plain_ppo" and branch_rollouts <= 0:
        raise ValueError(f"missing branch-store accounting: {spec.name}")
    row = {
        "run": spec.name,
        "kind": spec.kind,
        "city": spec.city,
        "wall_hours": _events_wall_hours(_event_path(status, spec.city)),
        "decisions": len(rows),
        "market_days": int(float(last["batch"])) + 1,
        "branch_rollouts": branch_rollouts,
        "branch_days": branch_days,
        "nmcc_hours": _float(last, "nmcc_seconds_total") / 3600.0,
        "rival_planner_hours": (
            _float(last, "firm2_profit_planner_seconds_total")
            + _float(last, "firm2_branch_profit_planner_seconds_total")
        ) / 3600.0,
        "ppo_updates": sum(_truthy(item.get("ppo_update_performed")) for item in rows),
        "actor_steps": int(sum(_float(item, "ppo_actor_optimizer_steps") for item in rows)),
        "value_fit_steps": int(_float(last, "v16_value_optimizer_steps_total")),
    }
    for key in ("wall_hours", "nmcc_hours", "rival_planner_hours"):
        if not math.isfinite(float(row[key])):
            raise ValueError(f"non-finite accounting value {spec.name}/{key}")
    return row


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=BASE / "runs/v17_results")
    parser.add_argument("--output", type=Path, default=BASE / "runs/v17_robust/wp4")
    args = parser.parse_args(argv)
    root = args.runs_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = [accounting_row(root, spec) for spec in RUN_SPECS]
    by_name = {row["run"]: row for row in rows}
    main_hours = float(by_name["nyc_s101"]["wall_hours"])
    ppo_hours = float(by_name["nyc_s101_plain_ppo"]["wall_hours"])
    matched_days = int(math.ceil((1200.0 * main_hours / ppo_hours) / 30.0) * 30)
    if matched_days < 1200:
        raise ValueError("matched PPO day count cannot be below 1200")
    csv_path = output / "compute_accounting.csv"
    temporary = csv_path.with_name(f".{csv_path.name}.tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, csv_path)
    atomic_json(output / "matched_days.json", {
        "schema_version": "v17-robust-wp4-matched-days-v1",
        "matched_days": matched_days,
        "nyc_s101_wall_hours": main_hours,
        "nyc_s101_plain_ppo_wall_hours": ppo_hours,
        "ratio": main_hours / ppo_hours,
        "rounding": "ceil to a multiple of 30 days",
        "compute_accounting_csv": str(csv_path),
    })
    print(json.dumps({"matched_days": matched_days, "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
