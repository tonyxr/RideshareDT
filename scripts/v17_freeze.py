#!/usr/bin/env python3
"""Freeze V17 production days and branch horizon from one passed T3 probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from scripts.v17_config import (
    DEFAULT_CONFIG,
    canonical_hash,
    load_config,
    validate_config,
)


DEFAULT_T3_ROOT = BASE / "runs/v17_t3_short"
MIN_FREEZE_EVALUATION_DAYS = 28
STEPS_PER_DAY = 6
MIN_FREEZE_OPENINGS = 3
SHORT_TRAINING_DAYS = 60
SHORT_CHECKPOINT_DAYS = (30, 45, 60)
HORIZONS = (5, 7)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_freeze_evidence(
    config: Mapping[str, Any],
    status: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    report_path: Path,
) -> tuple[int, int]:
    """Return selected horizon and production days, failing closed on drift."""
    validate_config(config)
    if config.get("status") != "prefreeze":
        raise ValueError("only a prefreeze V17 configuration can be frozen")
    if status.get("state") != "passed" or status.get("pass") is not True:
        raise ValueError("freezing requires a passed T3 status")
    if status.get("eligible_to_freeze_config") is not True:
        raise ValueError("T3 status is not eligible to freeze the configuration")
    if status.get("config_hash") != canonical_hash(config):
        raise ValueError("T3 status uses a stale prefreeze configuration hash")
    if status.get("report_sha256") != _sha256(report_path):
        raise ValueError("T3 report hash does not match its status")
    if report.get("state") != "passed" or report.get("pass") is not True:
        raise ValueError("freezing requires a passed T3 report")
    if report.get("config_hash") != canonical_hash(config):
        raise ValueError("T3 report uses a stale prefreeze configuration hash")
    if (
        status.get("phase") != "T3-short"
        or report.get("phase") != "T3-short"
        or report.get("short_probe") is not True
    ):
        raise ValueError("only the registered widened-grid short T3 can freeze")
    evaluation_days = int(report.get("evaluation_days", 0))
    if (
        evaluation_days < MIN_FREEZE_EVALUATION_DAYS
        or int(report.get("evaluation_steps", 0))
        != evaluation_days * STEPS_PER_DAY
        or int(report.get("validation_openings", 0))
        < MIN_FREEZE_OPENINGS
    ):
        raise ValueError("T3 external evaluation is too short to freeze")
    if (
        int(report.get("training_days", 0)) != SHORT_TRAINING_DAYS
        or tuple(report.get("checkpoint_days", ())) != SHORT_CHECKPOINT_DAYS
        or tuple(report.get("horizons", {}).keys())
        != tuple(str(value) for value in HORIZONS)
    ):
        raise ValueError("T3 freeze probe coverage is incomplete")
    horizon = int(report.get("selected_horizon", -1))
    days = int(
        dict(report.get("production_length", {}) or {}).get(
            "training_days", -1
        )
    )
    if horizon not in tuple(config["credit"]["horizon_candidates_from_T3"]):
        raise ValueError("T3 selected an unregistered branch horizon")
    if days < 4:
        raise ValueError("T3 recommended an invalid production length")
    return horizon, days


def freeze_config(
    config_path: Path,
    status_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    config = load_config(config_path)
    status = json.loads(status_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    prefreeze_hash = canonical_hash(config)
    horizon, days = validate_freeze_evidence(
        config, status, report, report_path=report_path
    )
    config["status"] = "frozen"
    config["credit"]["horizon_days"] = horizon
    config["training"]["days"] = days
    config["freeze_evidence"] = {
        "phase": str(status.get("phase")),
        "prefreeze_config_hash": prefreeze_hash,
        "t3_report": str(report_path.resolve()),
        "t3_report_sha256": _sha256(report_path),
        "selected_horizon": horizon,
        "production_days": days,
    }
    validate_config(config, require_frozen=True)
    temp = config_path.with_suffix(config_path.suffix + ".tmp")
    temp.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    temp.replace(config_path)
    return {
        "state": "frozen",
        "config": str(config_path.resolve()),
        "prefreeze_config_hash": prefreeze_hash,
        "frozen_config_hash": canonical_hash(config),
        "selected_horizon": horizon,
        "production_days": days,
        "t3_report_sha256": _sha256(report_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--status", type=Path, default=DEFAULT_T3_ROOT / "status.json"
    )
    parser.add_argument(
        "--report", type=Path, default=DEFAULT_T3_ROOT / "report.json"
    )
    args = parser.parse_args()
    result = freeze_config(
        args.config.resolve(), args.status.resolve(), args.report.resolve()
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
