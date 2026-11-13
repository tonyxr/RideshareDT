#!/usr/bin/env python3
"""Measure the V18 revealed-optimality demand scale as a sensitivity only."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from diagnostic_logging import atomic_json  # noqa: E402
from scripts import v17_market_checks as t1  # noqa: E402
from scripts import v17_t2_sweep as t2  # noqa: E402
from scripts.v17_config import load_config as load_v17  # noqa: E402
from scripts.v17_robust_common import set_option  # noqa: E402
from scripts import v18_p2 as p2  # noqa: E402


BETA_FACTORS = (1.0, 1.5, 2.0, 3.0, 4.0)
LEVELS = (0.90, 0.95, 1.00, 1.05, 1.10)
SOURCE = BASE / "artifacts/v17_relative_demand_post_fare_20261028/nyc/demand_1.15.json"
ARTIFACT = BASE / "artifacts/v18_revealed_demand/nyc/demand_foc.json"


def _candidate_artifact(output: Path, factor: float) -> Path:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    source["beta"] = float(source["beta"]) * float(factor)
    source["evidence"] = {
        **dict(source["evidence"]),
        "type": "revealed_optimality_sensitivity_candidate",
        "production_use": False,
        "beta_factor_relative_to_lyft_prior": float(factor),
    }
    path = output / "candidate_artifacts" / f"demand_factor_{factor:g}.json"
    atomic_json(path, source)
    return path


def _job(
    config: Mapping[str, Any], output: Path, *, factor: float,
) -> p2.Job:
    supply = json.loads((p2.PAY_DIR / "nyc/supply.json").read_text())
    work = output / f"factor_{factor:g}"
    command = t1.market_command(
        config, city="nyc", job="calm_static", output=work,
        openings=2, customers=p2.CUSTOMERS, pool=p2.PRODUCTION_POOL,
    )
    raw = work / "raw.json"
    specs = [t2.cell_spec(level, level) for level in LEVELS]
    command = set_option(command, "--policy_sweep", ";".join(specs))
    command = set_option(
        command, "--policy_sweep_horizon", 28 * p2.STEPS_PER_DAY
    )
    command = set_option(command, "--policy_sweep_seed", 20231)
    command = set_option(
        command, "--policy_sweep_opening_tariff_randomization", 0.0
    )
    command = set_option(
        command, "--demand_calibration_artifact",
        _candidate_artifact(output, factor),
    )
    command, expected = p2._v18_command(
        command, city="nyc", raw=raw,
        report_prefix=work / "diagnostics", stock=False,
        linear_pay=True, fleet_scale=float(supply["fleet_scale"]),
        base_pickup_minutes=float(supply["base_pickup_minutes"]),
        production_population=True,
        base_active_drivers=int(supply["base_active_drivers"]),
    )
    return p2.Job(
        name=f"revealed_beta_{factor:g}", command=tuple(command), raw=raw,
        manifest=work / "manifest.json", log=work / "job.log",
        expected=expected,
    )


def _quadratic_result(payload: Mapping[str, Any]) -> dict[str, Any]:
    profits = []
    for level in LEVELS:
        spec = t2.cell_spec(level, level)
        results = payload["results"][spec]
        profits.append(float(np.mean([
            float(row["own_profit"]) for row in results.values()
        ])))
    coefficients = np.polyfit(np.asarray(LEVELS), np.asarray(profits), 2)
    a, b, c = map(float, coefficients)
    maximizer = -b / (2.0 * a) if a < 0.0 else float(LEVELS[int(np.argmax(profits))])
    return {
        "levels": list(LEVELS), "own_profit_per_request": profits,
        "quadratic": {"level2": a, "level": b, "intercept": c},
        "quadratic_maximizer": maximizer,
        "concave": bool(a < 0.0),
    }


def _interpolated_factor(rows: Sequence[Mapping[str, Any]]) -> float:
    points = sorted(
        (float(row["factor"]), float(row["quadratic_maximizer"]) - 1.0)
        for row in rows
    )
    for (f0, y0), (f1, y1) in zip(points, points[1:]):
        if y0 == 0.0:
            return f0
        if y0 * y1 <= 0.0 and y1 != y0:
            return f0 + (f1 - f0) * (-y0) / (y1 - y0)
    nearest = sorted(points, key=lambda point: abs(point[1]))[:2]
    (f0, y0), (f1, y1) = sorted(nearest)
    if y1 == y0:
        return f0
    return float(np.clip(
        f0 + (f1 - f0) * (-y0) / (y1 - y0),
        BETA_FACTORS[0], BETA_FACTORS[-1],
    ))


def run(output: Path, *, workers: int = 4) -> dict[str, Any]:
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = load_v17()
    jobs = [_job(config, output, factor=factor) for factor in BETA_FACTORS]
    p2._run_parallel(
        jobs, workers=workers, output=output, phase="revealed_beta"
    )
    rows = []
    for factor, job in zip(BETA_FACTORS, jobs):
        payload = p2._validate_product(job.raw, job.expected)
        rows.append({"factor": factor, **_quadratic_result(payload)})
    factor_foc = _interpolated_factor(rows)
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    beta_prior = float(source["beta"])
    prior_elasticity = float(
        source["evidence"].get("target_own_elasticity", -1.15)
    )
    beta_foc = factor_foc * beta_prior
    implied_elasticity = factor_foc * prior_elasticity
    artifact = copy.deepcopy(source)
    artifact["beta"] = beta_foc
    artifact["evidence"] = {
        "type": "revealed_optimality_sensitivity",
        "production_use": False,
        "production_artifact": str(SOURCE.relative_to(BASE)),
        "production_target_own_elasticity": prior_elasticity,
        "beta_factor_relative_to_lyft_prior": factor_foc,
        "implied_local_own_elasticity": implied_elasticity,
        "elasticity_method": (
            "Exact proportional scaling of the local derivative whose prior "
            "beta was fitted by fit_local_own_elasticity"
        ),
        "procedure": {
            "beta_factors": list(BETA_FACTORS), "levels": list(LEVELS),
            "openings": 2, "days": 28, "stock_frozen": True,
            "rival": "published_card",
        },
        "quadratic_fits": rows,
        "limitations": [
            "Sensitivity only; not used for training, selection, or production.",
            "The revealed optimum is a static platform-profit condition, not external causal demand evidence.",
        ],
    }
    if not math.isfinite(beta_foc) or beta_foc <= 0.0:
        raise ValueError("revealed-beta interpolation produced an invalid beta")
    atomic_json(ARTIFACT, artifact)
    report = {
        "state": "complete", "production_demand_scale": "lyft_prior",
        "beta_prior": beta_prior, "beta_foc": beta_foc,
        "factor_foc": factor_foc,
        "implied_local_own_elasticity": implied_elasticity,
        "fits": rows, "artifact": str(ARTIFACT),
    }
    atomic_json(output / "report.json", report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 10:
        raise ValueError("workers must be in [1, 10]")
    print(json.dumps(run(args.output, workers=args.workers), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
