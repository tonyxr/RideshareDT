#!/usr/bin/env python3
"""Probe the manual-authorized V17 T1.1 fleet-size refit.

The probe changes only ``--fleet_scale``.  It holds the V3 elasticity,
calibrated demand, tariff, scenario, seeds, and calm-card policy fixed, and
uses common-random-number openings for every candidate.  The selected scale
is reported but never written into the production configuration automatically.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Sequence

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from diagnostic_logging import atomic_json
from scripts.v17_config import DEFAULT_CONFIG, canonical_hash, load_config
from scripts.v17_market_checks import (
    CALIBRATION_CITIES,
    CARD,
    _read_raw,
    _run_job,
    _t11,
    market_command,
)


def _replace_flag(command: Sequence[str], flag: str, value: str) -> list[str]:
    result = list(command)
    index = result.index(flag)
    result[index + 1] = str(value)
    return result


def run_candidate(
    config: dict,
    *,
    city: str,
    scale: float,
    output: Path,
    openings: int,
    customers: int,
    pool: int,
    force: bool,
) -> dict:
    candidate = copy.deepcopy(config)
    candidate["city_inputs"][city]["fleet_scale"] = float(scale)
    candidate_dir = output / f"scale_{scale:.4f}"
    command = market_command(
        candidate,
        city=city,
        job="calm_static",
        output=candidate_dir,
        openings=openings,
        customers=customers,
        pool=pool,
    )
    # Fleet fitting needs only the calm card.  Removing the fare-rise arm does
    # not change the card rollout because every opening uses its own CRN tape.
    command = _replace_flag(command, "--policy_sweep", CARD)
    raw = candidate_dir / "raw" / city / "calm_static.json"
    manifest = candidate_dir / "commands" / city / "calm_static.json"
    _run_job(
        command,
        raw,
        manifest,
        force=force,
        config_hash=canonical_hash(candidate),
    )
    result = _t11(city, _read_raw(candidate_dir, city, "calm_static"), customers)
    return {
        "fleet_scale": float(scale),
        "fulfillment": float(result["v17_fulfillment"]),
        "reference_fulfillment": float(result["v15d_fulfillment"]),
        "absolute_gap": float(result["fulfillment_absolute_gap"]),
        "within_two_points": bool(result["fulfillment_pass"]),
        "raw": str(raw),
    }


def select_candidate(candidates: Sequence[dict]) -> dict:
    if not candidates:
        raise ValueError("at least one fleet-scale candidate is required")
    passing = [item for item in candidates if item["within_two_points"]]
    pool = passing or list(candidates)
    return min(pool, key=lambda item: (item["absolute_gap"], item["fleet_scale"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--city", choices=CALIBRATION_CITIES, default="nyc")
    parser.add_argument("--candidates", default="1.0,1.1,1.2,1.3,1.4")
    parser.add_argument("--openings", type=int, default=2)
    parser.add_argument("--customers", type=int, default=500)
    parser.add_argument("--pool", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=Path("runs/v17_t1_fleet_refit"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    scales = [float(item) for item in args.candidates.split(",") if item.strip()]
    if any(not 0.5 <= scale <= 2.0 for scale in scales):
        raise ValueError("candidate fleet scales must lie in [0.5, 2.0]")
    config = load_config(args.config)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    results = [
        run_candidate(
            config,
            city=args.city,
            scale=scale,
            output=output,
            openings=max(1, int(args.openings)),
            customers=max(1, int(args.customers)),
            pool=max(1, int(args.pool)),
            force=bool(args.force),
        )
        for scale in scales
    ]
    selected = select_candidate(results)
    report = {
        "contract": "v17-t1.1-fleet-N-only-refit-v1",
        "city": args.city,
        "openings": max(1, int(args.openings)),
        "customers": max(1, int(args.customers)),
        "fixed_eta_v3": 0.6,
        "candidates": results,
        "selected": selected,
        "selection_is_within_two_points": bool(selected["within_two_points"]),
        "note": "Selection is provisional until the six-opening T1 rerun passes.",
    }
    atomic_json(output / "report.json", report)
    print(json.dumps(report, indent=2))
    return 0 if selected["within_two_points"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
