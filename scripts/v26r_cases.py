"""Corner-case registry for the v26r evaluation matrix (see claude/corner_cases_v26r_2026-10-14.md).

Each case names a scripted scenario, the day the regime starts and ends, the regime keys the
oracle may condition on (``Core._regime_key_for_hour`` labels), optional schedule specs that
play the "myopic" response (bait matcher, undercut), and the role of the case in the paper.
Days are simulator days from the opening; evaluation uses a 10-day burn-in so the rival has
converged before anything happens.
"""
from __future__ import annotations

from typing import Any, Dict, List

STATIC_GRID = (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0)
BURNIN_DAYS = 10
HORIZON_DAYS = 40


def _schedule(default: float, days: Dict[int, float]) -> str:
    import json
    return "schedule:" + json.dumps({"default": [default, 1.0], "days": {str(d): [m, 1.0] for d, m in days.items()}},
                                     separators=(",", ":"))


CASES: Dict[str, Dict[str, Any]] = {
    "A1_shortage_severe": dict(
        scenario="scenarios/cases/shortage_severe_late.json", onset=14, end=19,
        regimes=["driver_shortage"], role="train+eval",
        why="surge rules react to waits after supply fell and release early; the shortage persists",
    ),
    "A2_shortage_moderate_null": dict(
        scenario="scenarios/cases/shortage_moderate_null.json", onset=14, end=18,
        regimes=["driver_shortage"], role="null",
        why="capacity does not bind; any policy that moves here is over-fitting",
    ),
    "A3_storm_transit": dict(
        scenario="scenarios/cases/storm_transit_late.json", onset=16, end=17,
        regimes=["winter_storm", "winter_storm_forecast"], role="eval",
        why="forecast is public but a rule has no model of the rebound or supply recovery",
    ),
    "A4_holiday_null": dict(
        scenario="scenarios/holiday.json", onset=7, end=12, burnin=5,
        regimes=["holiday_wave"], role="null",
        why="demand-mix change only; outside option weak",
    ),
    "A5_rival_outage": dict(
        scenario="scenarios/cases/rival_outage.json", onset=12, end=14,
        regimes=["platform_outage"], role="train+eval",
        why="rival's stranded riders spill over; surge on their waits loses them, holding builds loyalty",
    ),
    "S1_promotion_bait": dict(
        scenario="scenarios/cases/promotion_bait.json", onset=12, end=16,
        regimes=["rival_promotion"], role="train+eval",
        myopic={"match_promotion": lambda m: _schedule(m, {d: round(m - 0.25, 2) for d in range(12, 17)})},
        why="matching earns during the promotion and is punished by the memory rival for ten days after",
    ),
    "S2_undercut_trap": dict(
        scenario="scenarios/cases/undercut_trap.json", onset=12, end=16,
        regimes=[], role="train+eval",
        myopic={"cut_20pct_5d": lambda m: _schedule(m, {d: round(m - 0.2, 2) for d in range(12, 17)}),
                "cut_10pct_5d": lambda m: _schedule(m, {d: round(m - 0.1, 2) for d in range(12, 17)})},
        why="a share-chasing rule sees +12 share points during the cut and never the retaliation after it",
    ),
    "S3_commitment_premium": dict(
        scenario="scenarios/cases/undercut_trap.json", onset=12, end=40,
        regimes=[], role="eval",
        myopic={"premium_10pct_held": lambda m: _schedule(m, {d: round(m + 0.1, 2) for d in range(12, 40)}),
                "premium_10pct_3d": lambda m: _schedule(m, {d: round(m + 0.1, 2) for d in range(12, 15)})},
        why="one-step planners re-optimise each step and never discover accommodation",
    ),
    "S4_rival_switch": dict(
        scenario="scenarios/cases/rival_switch_late.json", onset=16, end=40,
        regimes=["rival_switched"], role="train+eval",
        why="the best constant differs by rival; nothing in a rule re-identifies the rival",
    ),
    "S5_own_outage_loyalty": dict(
        scenario="scenarios/cases/own_outage_late.json", onset=12, end=14,
        regimes=["platform_outage"], role="eval",
        myopic={"ration_during_outage": lambda m: _schedule(m, {d: round(m + 0.3, 2) for d in range(12, 15)}),
                "rebuild_after_outage": lambda m: _schedule(m, {**{d: round(m + 0.3, 2) for d in range(12, 15)},
                                                               **{d: round(m + 0.1, 2) for d in range(15, 25)}})},
        why="loyalty halves after stranding; pricing to current waits keeps over-attracting demand",
    ),
    "S6_shortage_during_promotion": dict(
        scenario="scenarios/cases/shortage_during_promotion.json", onset=12, end=19,
        regimes=["rival_promotion", "driver_shortage"], role="holdout",
        why="the promotion says cut, the shortage says raise; a single-parameter response is wrong for one",
    ),
    "C0_calm": dict(
        scenario="scenarios/calm.json", onset=12, end=40,
        regimes=[], role="null",
        why="stationarity arm: the best constant must win",
    ),
}


def static_specs() -> List[str]:
    return [f"static:{m:g}" for m in STATIC_GRID]


def required_regime_labels(case, horizon_days=HORIZON_DAYS, calibration=None):
    """Enumerate the scripted support before seeing any validation/test rewards.

    Includes the full opening, not just scored days or decision-block labels.
    Calendar holidays and all forecast/compound/recovery labels are included.
    """
    from pathlib import Path
    from Core import load_shock_scenario
    from market_shocks import ShockConfig, ShockProcess, regime_label
    config = ShockConfig.from_calibration(calibration or {}, enabled=True,
        storm_daily_hazard={}, event_rate_per_day=0., shortage_rate_per_day=0.,
        outage_rate_per_day=0., transit_disruption_rate_per_day=0.,
        rival_promotion_rate_per_day=0., drift_sd=0., conditional_sampling=False)
    process = ShockProcess(config, seed=0, steps_per_day=6)
    process.scenario = load_shock_scenario(str(Path(__file__).resolve().parents[1]/case['scenario']))
    process.reset()
    labels, switched = {'calm'}, False
    for day in range(horizon_days):
        process.advance_day()
        switched |= any(e.kind == 'rival_switch' and e.covers(day, 0) for e in process.events)
        for block in range(6):
            labels.add(regime_label(process.effect(block), switched))
    return sorted(labels)


def oracle_specs(case: Dict[str, Any], calm_target: float, regime_targets=None, required_labels=None) -> List[str]:
    """On-grid regime tables anchored to a rival-specific calm validation optimum.

    This is a searched policy class, not an upper bound or an optimal schedule.
    Compound labels and recovery remain distinct in the simulator.
    """
    import json
    specs: List[str] = []
    regimes = list(case.get("regimes", []))
    if regime_targets is None:
        # Declaration-only API. Production supplies the actual labels found in
        # complete validation traces, including compound recovery phases.
        labels = sorted(set(regimes + [key+'_recovery' for key in regimes
                            if not key.endswith('_forecast') and key != 'rival_switched']))
        regime_targets = {label: calm_target for label in labels}
    regime_targets = dict(regime_targets)
    # Missing-fit labels are explicit SEARCH coordinates, not a runtime
    # fallback to calm. Every grid value is evaluated for each coordinate.
    # An initializer is not reported as a fitted/optimal target.
    for label in required_labels or []:
        if label != 'calm' and label not in regime_targets:
            regime_targets[label] = STATIC_GRID[0]
    if not regime_targets:
        return specs
    if any(value not in STATIC_GRID for value in [calm_target, *regime_targets.values()]):
        raise ValueError('oracle targets must belong to the declared tariff grid')
    base = {"calm": [calm_target, 1.0], **{key: [value, 1.0] for key, value in regime_targets.items() if key != 'calm'}}
    def encode(table):
        return 'oracle:' + json.dumps(table, sort_keys=True, separators=(',', ':'))
    specs.append(encode(base))
    # Independent label targets plus coordinate neighbourhood. This searched
    # class can express opposing responses, but is NOT a global dynamic oracle.
    for target in STATIC_GRID:
        specs.append(encode({key: ([calm_target, 1.] if key == 'calm' else [target, 1.]) for key in base}))
        for label in regime_targets:
            if label != 'calm':
                specs.append(encode({**base, label: [target, 1.]}))
    return list(dict.fromkeys(specs))


def myopic_specs(case: Dict[str, Any], calm_target: float) -> Dict[str, str]:
    import json
    specs = {}
    for name, fn in dict(case.get("myopic", {})).items():
        table = json.loads(fn(calm_target).split(':', 1)[1])
        for entry in [table['default'], *table['days'].values()]:
            entry[0] = min(STATIC_GRID, key=lambda value: (abs(value-entry[0]), value))
        specs[name] = 'schedule:' + json.dumps(table, separators=(',', ':'))
    return specs
