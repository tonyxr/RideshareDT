#!/usr/bin/env python3
"""Does adapting to shocks pay?  Simulator-only diagnostics of the v27 market.

For every shock type a scripted scenario places one event in a ten-day window
(onset on day 3).  Firm 2 posts the anchor tariff (default; see ``--rival``).
Firm 1 tariff multipliers are evaluated on common random numbers (riders,
trips, request volume, weather): every policy of a comparison faces the same
draws.  Three policy classes, from least to most adaptive:

1. ``static``: one multiplier for the whole window (grid);
2. ``event switching``: the best static multiplier outside the event and a
   separate multiplier on event-active blocks (and, for public shocks, from the
   public warning onwards: *foresight*);
3. ``oracle schedule``: an open-loop multiplier per segment (each day, and the
   event-active blocks of each event day separately), found by coordinate
   ascent on *training* seeds and evaluated on disjoint *test* seeds.  It knows
   the shock path, so it bounds what any adaptive policy can gain; the placebo
   window (no event) measures how much of that gain is overfitting.

Reported per shock type (Firm 1 profit per base request, summed over the
window; test-seed means with paired standard errors):

* ``static_profile`` and ``best_static``;
* ``switching``: best event multiplier given the static normal one,
  ``delta_m`` = m_event - m_static and its paired gain;
* ``oracle``: the schedule (multiplier by day relative to onset), its gain over
  the best static policy, and the same gain in the placebo window.

Acceptance (pre-registered in SHOCK_MARKET_PROPOSAL_2026-10-07.md): a shock
type supports the adaptive-pricing claim when the regime-optimal multiplier
moves by at least 0.10 and the adaptive gain is at least 3 paired SE above zero
on test seeds while the placebo gain is not.

Riders use the offline rubric thresholds (no OpenAI key).

    python scripts/v27_shock_diagnostics.py --city boston --out artifacts/v27_diagnostics/shock_diagnostics_boston.json
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import datetime as dt
import io
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from calibration_utils import load_calibration_json  # noqa: E402
from market_dynamics import MarketDynamicsConfig  # noqa: E402
from market_engine import EngineConfig  # noqa: E402
from market_shocks import (  # noqa: E402
    ShockConfig,
    ShockEvent,
    ShockProcess,
    ShockScenario,
    holiday_dates,
)

CITY_NAMES = {"nyc": "New York City", "boston": "Boston", "chicago": "Chicago"}
WINDOW_DAYS = 10
ONSET_DAY = 3
STEPS_PER_DAY = 6
PUBLIC_KINDS = {"winter_storm", "holiday_wave", "large_event"}
DEFAULT_GRID = (0.85, 0.92, 1.00, 1.08, 1.15, 1.30, 1.50, 1.75)
QUIET_CONFIG = dict(
    storm_daily_hazard={}, event_rate_per_day=0.0, shortage_rate_per_day=0.0,
    outage_rate_per_day=0.0, transit_disruption_rate_per_day=0.0,
    rival_promotion_rate_per_day=0.0, drift_sd=0.0,
)
Segment = Tuple[str, int]  # ("D", day) or ("E", day) for event-active blocks


def scenario_for(kind: str, start: dt.date) -> Tuple[ShockConfig, ShockScenario]:
    """Scripted ten-day window with one event of ``kind`` from day 3."""
    d = ONSET_DAY
    types: Tuple[str, ...] = (kind,) if kind != "placebo" else ()
    config = ShockConfig(enabled=True, mode="calibrated", types=types, **QUIET_CONFIG)
    events: List[ShockEvent] = []
    forecasts: Dict[int, bool] = {}
    if kind == "winter_storm":
        events.append(ShockEvent("winter_storm", d, d + 1, intensity=1.0, detail={"transit_suspended": True}))
        forecasts = {d: True, d + 1: True}
    elif kind == "large_event":
        events.append(ShockEvent("large_event", d, d, intensity=1.30, start_block=3, end_block=5, announced=True))
    elif kind == "driver_shortage":
        events.append(ShockEvent("driver_shortage", d, d + 2, intensity=0.70, announced=False))
    elif kind == "platform_outage":
        events.append(ShockEvent("platform_outage", d, d, intensity=0.30, start_block=2, end_block=4,
                                 announced=False, firm="Firm2"))
    elif kind == "transit_disruption":
        events.append(ShockEvent("transit_disruption", d, d + 2))
    elif kind == "rival_promotion":
        events.append(ShockEvent("rival_promotion", d, d + 4, announced=False))
    elif kind == "holiday_wave":
        # Thanksgiving wave = 2 days before through 3 after: onset on day 3.
        anchor = holiday_dates(start.year)["thanksgiving"]
        start = anchor - dt.timedelta(days=d + 2)
    elif kind != "placebo":
        raise ValueError(f"unknown scenario kind {kind!r}")
    return config, ShockScenario(start_date=start, events=events, forecasts=forecasts)


def build_core(city: str, calibration: str, *, seed: int, pool: int, target_utilization: float,
               rider_memory: bool = False):
    import Core as core_module
    name = CITY_NAMES[city]
    with contextlib.redirect_stdout(io.StringIO()):
        core = core_module.Core(market_name=name, seed=seed, firm1_mode="static", firm2_mode="static",
                                total_customers_pool=pool, openai_api_key="")
        cal = load_calibration_json(calibration, expected_city=name)
        cal["_source_dir"] = str(Path(calibration).resolve().parent)
        core.apply_calibration(cal)
        core.configure_v26(market_dynamics=MarketDynamicsConfig(enabled=True, capacity_target_fulfillment=0.0,
                                                                rider_memory=bool(rider_memory)),
                           steps_per_day=STEPS_PER_DAY)
        core.configure_v27(engine_config=EngineConfig(enabled=True, target_utilization=target_utilization),
                           shock_config=ShockConfig(enabled=True, types=(), **QUIET_CONFIG))
        core._sample_profiles_from_pool(10)  # build the offline threshold pool once
    return core


def set_multiplier(core, firm, m: float) -> None:
    base = core.market.curr_market
    for key in core.shared_edit_keys:
        lo, hi = firm.config.bounds[key] if hasattr(firm, "config") else (-np.inf, np.inf)
        setattr(firm.overrides, key, float(np.clip(m * float(getattr(base, key)), lo, hi)))


def surge_multiplier(core) -> float:
    """Exploratory rival: a myopic scarcity surge on Firm 2's own telemetry.

    Warning: this rule is exploitable (a Firm 1 price rise shifts demand to
    Firm 2, which then surges), so it inflates every high static multiplier.
    """
    report = (getattr(core, "last_v27_step", {}) or {}).get("fleet", {}).get("Firm2")
    if not report:
        return 1.0
    shortfall = 1.0 - float(report["served_fraction"])
    excess = max(0.0, float(report["utilization"]) - 0.60)
    return float(np.clip(1.0 + 1.0 * shortfall + 2.0 * excess, 1.0, 1.75))


def rollout(base_core, kind: str, *, seed: int, policy: Callable[[Segment, bool], float],
            customers: int, start: dt.date, rival: str = "anchor") -> Dict[str, Any]:
    """One window under ``policy(segment, warned) -> multiplier``."""
    core = copy.deepcopy(base_core)
    config, scenario = scenario_for(kind, start)
    core.shock_config = config
    core._reseed_environment(int(seed))
    process = ShockProcess(config, seed=int(seed) + 31, steps_per_day=STEPS_PER_DAY)
    process.scenario = scenario
    core.market.install_shocks(process)
    core.driver_supply.v27_fleet.reset()
    core.last_v27_step = {}
    if getattr(core.choice_model, "reputation", None) is not None:
        core.choice_model.reputation.reset()
    set_multiplier(core, core.firm2, 1.0)
    profit = share = served = 0.0
    segments: List[Segment] = []
    for day in range(WINDOW_DAYS):
        ctx = core.market.sample_day_context()
        for _ in range(STEPS_PER_DAY):
            hour = core.market.sample_timestep_hour().hour
            effect = core.market.shock_effect(hour)
            if kind == "placebo":
                active = ONSET_DAY <= day <= ONSET_DAY + 1
            else:
                active = kind in effect.active
            public = effect.public
            warned = kind in PUBLIC_KINDS and bool(
                public.get("storm_warning", 0.0) > 0.0
                or (kind == "holiday_wave" and public.get("holiday_proximity", 0.0) > 0.0
                    and day >= ONSET_DAY - 1)
                or public.get("event_announced", 0.0) > 0.0
            )
            segment: Segment = ("E" if active else "D", day)
            segments.append(segment)
            set_multiplier(core, core.firm1, policy(segment, warned))
            if rival == "surge":
                set_multiplier(core, core.firm2, surge_multiplier(core))
            n = core._step_customers(customers, hour=hour, day_of_week=ctx.day_of_week, weather=ctx.weather)
            profiles = core._sample_profiles_from_pool(core._effective_simulation_sample_size(n))
            _, m1, _, _, _, _ = core.simulate_batch(
                ctx.day_of_week, ctx.weather, hour, n, profiles, False, customers
            )
            profit += float(m1.profit_per_request)
            share += float(m1.completed_share)
            served += float(m1.fulfillment_rate)
    steps = WINDOW_DAYS * STEPS_PER_DAY
    return {"profit": profit, "completed_share": share / steps, "fulfillment": served / steps,
            "segments": segments}


def paired(diff: np.ndarray) -> Dict[str, float]:
    mean = float(diff.mean())
    se = float(diff.std(ddof=1) / np.sqrt(diff.size)) if diff.size > 1 else float("nan")
    t = mean / se if se and se > 0 else (0.0 if abs(mean) < 1e-12 else float("nan"))
    return {"mean": mean, "se": se, "t": float(t)}


def evaluate_kind(base_core, kind: str, *, train_seeds: Sequence[int], test_seeds: Sequence[int],
                  grid: Sequence[float], customers: int, start: dt.date, rival: str,
                  sweeps: int) -> Dict[str, Any]:
    def profits(policy: Callable[[Segment, bool], float], seeds: Sequence[int]) -> np.ndarray:
        return np.array([
            rollout(base_core, kind, seed=s, policy=policy, customers=customers, start=start, rival=rival)["profit"]
            for s in seeds
        ])

    # 1. Static landscape (train seeds choose the multiplier; test seeds report).
    static_train = {m: profits(lambda seg, w, m=m: m, train_seeds) for m in grid}
    best_static = max(grid, key=lambda m: float(static_train[m].mean()))
    static_test = profits(lambda seg, w: best_static, test_seeds)

    # 2. Event switching around the static normal multiplier.
    switch_train = {m: profits(lambda seg, w, m=m: m if seg[0] == "E" else best_static, train_seeds) for m in grid}
    best_event = max(grid, key=lambda m: float(switch_train[m].mean()))
    switch_test = profits(lambda seg, w: best_event if seg[0] == "E" else best_static, test_seeds)
    result: Dict[str, Any] = {
        "kind": kind,
        "static_profile_train": {str(m): float(v.mean()) for m, v in static_train.items()},
        "best_static": best_static,
        "best_static_test_profit": float(static_test.mean()),
        "switching": {
            "event_profile_train": {str(m): float(v.mean()) for m, v in switch_train.items()},
            "best_event_multiplier": best_event,
            "delta_m": float(best_event - best_static),
            "gain_test": paired(switch_test - static_test),
        },
    }
    if kind in PUBLIC_KINDS:
        fore_test = profits(lambda seg, w: best_event if (seg[0] == "E" or w) else best_static, test_seeds)
        result["switching"]["foresight_gain_test"] = paired(fore_test - switch_test)

    # 3. Open-loop oracle schedule by segment (coordinate ascent on train seeds).
    probe = rollout(base_core, kind, seed=train_seeds[0], policy=lambda seg, w: best_static,
                    customers=customers, start=start, rival=rival)
    ordered: List[Segment] = list(dict.fromkeys(probe["segments"]))
    schedule: Dict[Segment, float] = {seg: best_static for seg in ordered}
    current = float(static_train[best_static].mean())
    for _ in range(int(sweeps)):
        improved = False
        for seg in ordered:
            best_m, best_v = schedule[seg], current
            for m in grid:
                if m == schedule[seg]:
                    continue
                trial = dict(schedule)
                trial[seg] = m
                value = float(profits(lambda s, w, t=trial: t.get(s, best_static), train_seeds).mean())
                if value > best_v + 1e-9:
                    best_m, best_v = m, value
            if best_m != schedule[seg]:
                schedule[seg], current, improved = best_m, best_v, True
        if not improved:
            break
    oracle_test = profits(lambda s, w: schedule.get(s, best_static), test_seeds)
    result["oracle"] = {
        "schedule": {f"{k[0]}{k[1] - ONSET_DAY:+d}": v for k, v in schedule.items()},
        "train_profit": current,
        "gain_test": paired(oracle_test - static_test),
        "max_shift": float(max(abs(v - best_static) for v in schedule.values())),
    }
    anchor = rollout(base_core, kind, seed=test_seeds[0], policy=lambda seg, w: 1.0,
                     customers=customers, start=start, rival=rival)
    result["anchor_window"] = {k: v for k, v in anchor.items() if k != "segments"}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--city", choices=sorted(CITY_NAMES), default="boston")
    parser.add_argument("--calibration", default="")
    parser.add_argument("--kinds", default="placebo,winter_storm,holiday_wave,large_event,driver_shortage,"
                                          "platform_outage,transit_disruption,rival_promotion")
    parser.add_argument("--grid", default=",".join(str(m) for m in DEFAULT_GRID))
    parser.add_argument("--train-seeds", type=int, default=4)
    parser.add_argument("--test-seeds", type=int, default=8)
    parser.add_argument("--sweeps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--customers", type=int, default=80)
    parser.add_argument("--pool", type=int, default=1500)
    parser.add_argument("--target-utilization", type=float, default=0.60)
    parser.add_argument("--start", default="2023-01-16", help="Window start for non-holiday scenarios.")
    parser.add_argument("--rival", choices=("anchor", "surge"), default="anchor",
                        help="Firm 2: the anchor tariff (default) or an exploratory scarcity-surge rule.")
    parser.add_argument("--rider-memory", action="store_true",
                        help="Enable rider price/reliability memory (penalizes persistently high tariffs).")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    calibration = args.calibration or str(PROJECT_ROOT / "data" / "processed" / args.city / "calibration.json")
    grid = tuple(float(x) for x in args.grid.split(",") if x.strip())
    train_seeds = [int(args.seed) + 101 * i for i in range(int(args.train_seeds))]
    test_seeds = [int(args.seed) + 7919 + 101 * i for i in range(int(args.test_seeds))]
    started = time.time()
    core = build_core(args.city, calibration, seed=int(args.seed), pool=int(args.pool),
                      target_utilization=float(args.target_utilization), rider_memory=bool(args.rider_memory))
    report: Dict[str, Any] = {
        "city": args.city,
        "grid": list(grid),
        "train_seeds": train_seeds,
        "test_seeds": test_seeds,
        "customers": int(args.customers),
        "rival": args.rival,
        "rider_memory": bool(args.rider_memory),
        "fleet_calibration": core.v27_settings.get("fleet_calibration", {}),
        "results": [],
    }
    start = dt.date.fromisoformat(args.start)
    placebo_gain: Optional[Dict[str, float]] = None
    for kind in [k.strip() for k in args.kinds.split(",") if k.strip()]:
        t0 = time.time()
        result = evaluate_kind(core, kind, train_seeds=train_seeds, test_seeds=test_seeds, grid=grid,
                               customers=int(args.customers), start=start, rival=args.rival,
                               sweeps=int(args.sweeps))
        if kind == "placebo":
            placebo_gain = result["oracle"]["gain_test"]
        sw, orc = result["switching"], result["oracle"]
        result["supports_adaptivity"] = bool(
            orc["max_shift"] >= 0.10 and orc["gain_test"]["t"] >= 3.0
            and (placebo_gain is None or kind == "placebo"
                 or orc["gain_test"]["mean"] > placebo_gain["mean"] + 3.0 * orc["gain_test"]["se"])
        ) if kind != "placebo" else False
        report["results"].append(result)
        fore = sw.get("foresight_gain_test")
        print(
            f"[{args.city}] {kind:18s} static*={result['best_static']:.2f} "
            f"event*={sw['best_event_multiplier']:.2f} (dm={sw['delta_m']:+.2f}, "
            f"gain={sw['gain_test']['mean']:+.2f}±{sw['gain_test']['se']:.2f})"
            + (f" foresight={fore['mean']:+.2f}±{fore['se']:.2f}" if fore else "")
            + f" | oracle gain={orc['gain_test']['mean']:+.2f}±{orc['gain_test']['se']:.2f} "
            f"(t={orc['gain_test']['t']:.1f}) schedule={orc['schedule']} "
            f"supports={result['supports_adaptivity']} [{time.time() - t0:.0f}s]",
            flush=True,
        )
    report["elapsed_seconds"] = round(time.time() - started, 1)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2, default=str))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
