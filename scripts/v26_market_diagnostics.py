#!/usr/bin/env python3
"""Simulator-only diagnostics of the v26 market (no learning, no API calls).

For each city calibration it reports
  * the static landscape: Firm 1 at a fixed tariff multiplier against an
    anchor-priced Firm 2 (choice share, completed share, fulfillment, profit);
  * elasticities: market (both firms +10%) and Firm 1 own-price (+10% alone);
  * forward-looking channels: the impulse response of a one-day 0.8x Firm 1
    promotion, with rider memory and driver-supply feedback switched on/off,
    on common random numbers against an identical no-promotion path.

Riders use the deterministic rubric thresholds (no OpenAI key needed), so
these numbers describe the environment, not a particular CrowdLLM cache.

    python scripts/v26_market_diagnostics.py --city nyc --fleet-scale 1.0
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from calibration_utils import load_calibration_json  # noqa: E402
from market_dynamics import MarketDynamicsConfig  # noqa: E402

CITY_NAMES = {"nyc": "New York City", "boston": "Boston", "chicago": "Chicago"}
HOURS = (2, 6, 10, 14, 18, 22)


def build(city: str, calibration: str, *, seed: int, memory: bool, supply: bool,
          fleet_scale: float, capacity_target: float, pay_basis: str):
    import Core as core_module
    name = CITY_NAMES[city]
    with contextlib.redirect_stdout(io.StringIO()):
        core = core_module.Core(market_name=name, seed=seed, firm1_mode="static",
                                firm2_mode="static", total_customers_pool=3000)
        cal = load_calibration_json(calibration, expected_city=name)
        cal["_source_dir"] = str(Path(calibration).resolve().parent)
        core.apply_calibration(cal)
        core.configure_v26(
            market_dynamics=MarketDynamicsConfig(
                enabled=True, rider_memory=memory, supply_feedback=supply,
                fleet_scale=fleet_scale, capacity_target_fulfillment=capacity_target,
                driver_pay_basis=pay_basis,
            ),
            steps_per_day=6,
        )
    return core


def set_tariffs(core, m1: float, m2: float) -> None:
    base = core.market.curr_market
    for key in core.shared_edit_keys:
        setattr(core.firm1.overrides, key, m1 * float(getattr(base, key)))
        setattr(core.firm2.overrides, key, m2 * float(getattr(base, key)))


def rollout(core, schedule, *, n: int, seed: int):
    core = copy.deepcopy(core)
    core.choice_model.rng = np.random.default_rng(seed + 1)
    core.driver_supply.rng = np.random.default_rng(seed + 2)
    core.rng = np.random.default_rng(seed + 3)
    if getattr(core.choice_model, "reputation", None) is not None:
        core.choice_model.reputation.reset()
    generator = copy.deepcopy(core.agent_gen)
    generator.rng = np.random.default_rng(seed + 4)
    out = []
    for step, (m1, m2) in enumerate(schedule):
        set_tariffs(core, m1, m2)
        profiles = generator.sample_profiles(n)
        _, f1, f2, _, _, _ = core.simulate_batch(
            day_of_week=2, weather="clear", hour=HOURS[step % 6], customers_per_step=n,
            sampled_profiles=profiles, collect_rows=False, supply_reference_customers=n,
        )
        no_ride = float(core.last_crowd_response_stats.get("no_ride_rate", 0.0))
        out.append([f1.chosen / n, f1.completed / n, f2.completed / n, f1.profit / n,
                    f2.profit / n, f1.completed / max(1, f1.dispatch_offers), no_ride])
    return np.asarray(out)


def landscape(core, *, n: int, seeds: int):
    rows = {}
    for m in (0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3):
        paths = [rollout(core, [(m, 1.0)] * 18, n=n, seed=100 * s)[6:] for s in range(seeds)]
        a = np.mean([p.mean(axis=0) for p in paths], axis=0)
        rows[f"{m:.1f}"] = dict(zip(("firm1_choice_share", "firm1_completed_share", "firm2_completed_share",
                                     "firm1_profit_per_request", "firm2_profit_per_request",
                                     "firm1_dispatch_fulfillment", "no_ride_rate"), map(float, a)))
    parity = rows["1.0"]
    up = [rollout(core, [(1.1, 1.1)] * 18, n=n, seed=100 * s)[6:].mean(axis=0) for s in range(seeds)]
    up = np.mean(up, axis=0)
    elasticity = {
        "market_rideshare_demand": float(((1 - up[6]) / (1 - parity["no_ride_rate"]) - 1) / 0.10),
        "firm1_own_choice": float((rows["1.1"]["firm1_choice_share"] / parity["firm1_choice_share"] - 1) / 0.10),
        "firm1_own_completed": float((rows["1.1"]["firm1_completed_share"] / parity["firm1_completed_share"] - 1) / 0.10),
    }
    return rows, elasticity


def impulse(args, city, calibration):
    results = {}
    schedule_promo = [(0.8 if 6 <= t < 12 else 1.0, 1.0) for t in range(36)]
    schedule_base = [(1.0, 1.0)] * 36
    for memory, supply in ((False, False), (False, True), (True, False), (True, True)):
        core = build(city, calibration, seed=args.seed, memory=memory, supply=supply,
                     fleet_scale=args.fleet_scale, capacity_target=0.0, pay_basis=args.pay_basis)
        diffs = [rollout(core, schedule_promo, n=args.customers, seed=10 * s)
                 - rollout(core, schedule_base, n=args.customers, seed=10 * s) for s in range(args.seeds)]
        d = np.mean(diffs, axis=0)
        results[f"memory={memory},supply_feedback={supply}"] = {
            "completed_share_lift_promo_day": float(d[6:12, 1].mean()),
            "completed_share_lift_next_day": float(d[12:18, 1].mean()),
            "completed_share_lift_days_2_3": float(d[18:30, 1].mean()),
            "profit_per_request_sum_promo_day": float(d[6:12, 3].sum()),
            "profit_per_request_sum_after_4_days": float(d[12:36, 3].sum()),
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--city", choices=sorted(CITY_NAMES), default="nyc")
    parser.add_argument("--calibration", default="")
    parser.add_argument("--fleet-scale", type=float, default=1.0)
    parser.add_argument("--pay-basis", choices=("anchor", "fare"), default="anchor")
    parser.add_argument("--customers", type=int, default=400)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--skip-impulse", action="store_true")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    calibration = args.calibration or str(PROJECT_ROOT / "data" / "processed" / args.city / "calibration.json")
    core = build(args.city, calibration, seed=args.seed, memory=False, supply=True,
                 fleet_scale=args.fleet_scale, capacity_target=0.95, pay_basis=args.pay_basis)
    rows, elasticity = landscape(core, n=args.customers, seeds=args.seeds)
    report = {
        "city": args.city, "calibration": calibration, "fleet_scale": args.fleet_scale,
        "pay_basis": args.pay_basis,
        "capacity_calibration": core.v26_settings.get("driver_capacity_calibration"),
        "static_landscape_vs_anchor_rival": rows, "elasticities": elasticity,
    }
    if not args.skip_impulse:
        report["promotion_impulse_response"] = impulse(args, args.city, calibration)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
