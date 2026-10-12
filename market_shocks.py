"""v27 stochastic shocks to demand, supply, market nature, and the rival.

A ``ShockProcess`` lives inside ``MarketInteraction`` (so counterfactual
snapshots and validation rebuilds copy it) and turns a dated calendar into a
per-block ``ShockEffect``: the channel multipliers the v27 market engine
applies.  Shocks are drawn from the process's own seeded stream, so every
policy evaluated on the same rollout faces the identical shock path.

Design rule (see SHOCK_MARKET_PROPOSAL_2026-10-07.md): a shock must change
more than request volume.  Each type moves at least one of rider mix and
price sensitivity, the supply/demand ratio, the outside option, the rival, or
a persistent state (deferred demand).  Magnitudes below are documented priors;
``ShockConfig.from_calibration`` overrides them with data estimates where the
city calibration provides ``shock_calibration``.
"""

from __future__ import annotations

import datetime as _dt
import copy
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

SHOCK_TYPES: Tuple[str, ...] = (
    "holiday_wave",
    "winter_storm",
    "large_event",
    "driver_shortage",
    "platform_outage",
    "transit_disruption",
    "rival_promotion",
    "demand_drift",
)

# v28 conditional-demand segments.  The trip bank does not identify trip
# purpose directly, so non-airport segments are deliberately described as
# modelled strata (and never as observed labels).  Airport direction is kept
# separate because departure and arrival waves have different timing.
SHOCK_SEGMENTS: Tuple[str, ...] = (
    "airport_origin",
    "airport_destination",
    "commute",
    "event",
    "essential",
    "discretionary",
)

# Public shock features appended to the platform observation when enabled.
PUBLIC_SHOCK_FEATURE_KEYS: Tuple[str, ...] = (
    "holiday_active",
    "holiday_proximity",
    "storm_warning",
    "storm_active",
    "transit_availability",
    "event_announced",
)

# Fixed-date and rule-based US holidays with multi-day travel waves.  Offsets
# are days relative to the holiday; the wave covers ``pre`` days before and
# ``post`` days after, with the airport surge strongest at the edges (outbound
# before, return after).
HOLIDAY_RULES: Tuple[Tuple[str, str, int, int], ...] = (
    ("thanksgiving", "nov_4th_thu", 2, 3),
    ("christmas_new_year", "dec_25", 3, 9),
    ("independence_day", "jul_04", 2, 2),
    ("memorial_day", "may_last_mon", 3, 1),
    ("labor_day", "sep_1st_mon", 3, 1),
)


def holiday_dates(year: int) -> Dict[str, _dt.date]:
    def nth_weekday(month: int, weekday: int, n: int) -> _dt.date:
        day = _dt.date(year, month, 1)
        offset = (weekday - day.weekday()) % 7
        return day + _dt.timedelta(days=offset + 7 * (n - 1))

    def last_weekday(month: int, weekday: int) -> _dt.date:
        day = _dt.date(year, month + 1, 1) - _dt.timedelta(days=1)
        return day - _dt.timedelta(days=(day.weekday() - weekday) % 7)

    return {
        "thanksgiving": nth_weekday(11, 3, 4),
        "christmas_new_year": _dt.date(year, 12, 25),
        "independence_day": _dt.date(year, 7, 4),
        "memorial_day": last_weekday(5, 0),
        "labor_day": nth_weekday(9, 0, 1),
    }


@dataclass
class ShockEffect:
    """Channel multipliers for one market block (neutral defaults)."""

    volume_multiplier: float = 1.0
    airport_share_multiplier: float = 1.0
    duration_multiplier: float = 1.0
    price_sensitivity_multiplier: float = 1.0
    airport_price_sensitivity_multiplier: float = 1.0
    reservation_multiplier: float = 1.0
    transit_availability: float = 1.0
    fleet_multiplier: float = 1.0
    firm_capacity_multiplier: Dict[str, float] = field(
        default_factory=lambda: {"Firm1": 1.0, "Firm2": 1.0}
    )
    reservation_wage_multiplier: float = 1.0
    operating_cost_multiplier: float = 1.0
    firm2_price_multiplier: float = 1.0
    # Conditional distribution channels.  They are consumed by
    # ``shock_sampling.apply_conditional_shock_distribution``.  Keeping them
    # on the effect makes a shock path a complete, serialisable object and
    # lets counterfactual branches share the exact same conditional sampler.
    segment_volume_multiplier: Dict[str, float] = field(
        default_factory=lambda: {name: 1.0 for name in SHOCK_SEGMENTS}
    )
    segment_duration_multiplier: Dict[str, float] = field(
        default_factory=lambda: {name: 1.0 for name in SHOCK_SEGMENTS}
    )
    segment_price_sensitivity_multiplier: Dict[str, float] = field(
        default_factory=lambda: {name: 1.0 for name in SHOCK_SEGMENTS}
    )
    segment_reservation_multiplier: Dict[str, float] = field(
        default_factory=lambda: {name: 1.0 for name in SHOCK_SEGMENTS}
    )
    segment_outside_cost_multiplier: Dict[str, float] = field(
        default_factory=lambda: {name: 1.0 for name in SHOCK_SEGMENTS}
    )
    segment_outside_inconvenience_multiplier: Dict[str, float] = field(
        default_factory=lambda: {name: 1.0 for name in SHOCK_SEGMENTS}
    )
    # Explicit event phase metadata.  ``phase`` is the highest-priority
    # physical/public phase; ``event_phases`` retains overlapping events.
    phase: str = "inactive"
    phase_progress: float = 0.0
    severity: float = 0.0
    event_phases: Dict[str, str] = field(default_factory=dict)
    driver_supply_phase: str = "normal"
    # Storm-only volume factor (part of ``volume_multiplier``); the suppressed
    # share of requests is partly deferred to the days after the storm.
    storm_volume_factor: float = 1.0
    weather_override: Optional[str] = None
    emergency: bool = False
    active: Tuple[str, ...] = ()
    public: Dict[str, float] = field(default_factory=dict)

    @property
    def neutral(self) -> bool:
        return not self.active

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ShockConfig:
    """Shock hazards and effect priors (documented assumptions, see module doc)."""

    enabled: bool = False
    mode: str = "calibrated"  # calibrated | stress
    intensity_scale: float = 1.0
    types: Tuple[str, ...] = SHOCK_TYPES
    public_features: bool = True
    emergency_surge_cap: float = 0.0  # 0 = no cap; else max multiplier vs anchor
    # v28: resample the empirical trip/profile caches conditionally on the
    # active regime.  With shocks disabled the legacy v27 path is untouched.
    conditional_sampling: bool = True
    # This label is intentionally explicit in artifacts.  Defaults below are
    # documented priors, not empirical calibration.
    calibration_source: str = "documented_prior"
    # Holiday travel wave
    holiday_airport_share_peak: float = 2.2
    holiday_commute_volume: float = 0.75
    holiday_airport_price_sensitivity: float = 0.70
    holiday_fleet: float = 0.90
    # Winter storm (hazard per day by month, Dec-Mar by default)
    storm_daily_hazard: Dict[int, float] = field(default_factory=lambda: {
        12: 0.035, 1: 0.05, 2: 0.045, 3: 0.025, 11: 0.01, 4: 0.005,
    })
    storm_duration_days: Tuple[int, int] = (1, 3)
    storm_forecast_lead_days: int = 1
    storm_forecast_miss_prob: float = 0.15
    storm_false_alarm_prob: float = 0.02
    storm_transit_suspension_prob: float = 0.5
    storm_transit_degraded: float = 0.5
    storm_discretionary_volume: float = 0.55
    storm_airport_share: float = 0.6
    storm_duration_multiplier: float = 1.35
    storm_fleet: float = 0.65
    storm_reservation_multiplier: float = 1.25
    storm_price_sensitivity: float = 0.80
    storm_deferred_share: float = 0.5
    storm_rebound_days: int = 2
    storm_supply_recovery_days: int = 2
    # Large events
    event_rate_per_day: float = 0.10
    # City-wide request multiplier while a large event lets out (concerts,
    # games, conventions): a localized surge diluted over the whole city.
    event_volume_multiplier: Tuple[float, float] = (1.15, 1.45)
    event_price_sensitivity: float = 0.85
    event_announce_prob: float = 0.7
    # Supply shocks
    shortage_rate_per_day: float = 0.02
    shortage_duration_days: Tuple[int, int] = (2, 6)
    shortage_fleet: Tuple[float, float] = (0.6, 0.8)
    shortage_wage_multiplier: float = 1.15
    shortage_recovery_days: int = 4
    outage_rate_per_day: float = 0.01
    outage_blocks: Tuple[int, int] = (1, 3)
    outage_capacity: Tuple[float, float] = (0.2, 0.4)
    # Market nature
    transit_disruption_rate_per_day: float = 0.008
    transit_disruption_days: Tuple[int, int] = (1, 4)
    transit_disruption_availability: float = 0.3
    # Rival
    rival_promotion_rate_per_day: float = 0.015
    rival_promotion_days: Tuple[int, int] = (3, 7)
    rival_promotion_multiplier: float = 0.80
    # Demand drift (AR(1) on the daily log level)
    drift_persistence: float = 0.95
    drift_sd: float = 0.03
    # Training sampler: share of episode openings placed near a shock window
    training_shock_opening_share: float = 0.5

    def __post_init__(self) -> None:
        if self.mode not in {"calibrated", "stress"}:
            raise ValueError("shock mode must be 'calibrated' or 'stress'")
        unknown = set(self.types) - set(SHOCK_TYPES)
        if unknown:
            raise ValueError(f"unknown shock types: {sorted(unknown)}")
        self.intensity_scale = float(max(0.0, self.intensity_scale))
        self.emergency_surge_cap = float(max(0.0, self.emergency_surge_cap))
        if self.mode == "stress":
            # Stress tests: rarer-than-calibrated events made common and strong.
            self.event_rate_per_day = max(self.event_rate_per_day, 0.25)
            self.shortage_rate_per_day = max(self.shortage_rate_per_day, 0.06)
            self.outage_rate_per_day = max(self.outage_rate_per_day, 0.05)
            self.transit_disruption_rate_per_day = max(self.transit_disruption_rate_per_day, 0.04)
            self.rival_promotion_rate_per_day = max(self.rival_promotion_rate_per_day, 0.05)

    def scaled(self, value: float) -> float:
        """Move a multiplier away from 1 by the intensity scale."""
        return float(1.0 + (float(value) - 1.0) * self.intensity_scale)

    @classmethod
    def from_calibration(cls, calibration: Optional[Mapping[str, Any]], **overrides: Any) -> "ShockConfig":
        """Return a config with an auditable calibration provenance label.

        Missing ``shock_calibration`` keeps the documented priors and is
        labelled as such; it is never silently represented as empirically
        calibrated.
        """
        params: Dict[str, Any] = {}
        shock_cal = dict((calibration or {}).get("shock_calibration", {}) or {})
        allowed = set(cls.__dataclass_fields__)
        for key, value in shock_cal.items():
            if key in allowed and value is not None:
                if key == "storm_daily_hazard":
                    value = {int(k): float(v) for k, v in dict(value).items()}
                params[key] = value
        params["calibration_source"] = (
            "city_shock_calibration" if shock_cal else "documented_prior_missing_shock_calibration"
        )
        params.update(overrides)
        return cls(**params)

    def as_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["storm_daily_hazard"] = {str(k): v for k, v in self.storm_daily_hazard.items()}
        return payload


@dataclass
class _Event:
    kind: str
    start_day: int
    end_day: int  # inclusive
    intensity: float = 1.0
    start_block: int = 0
    end_block: int = 10**9  # inclusive, within start/end days
    announced: bool = True
    firm: str = ""
    detail: Dict[str, Any] = field(default_factory=dict)

    def covers(self, day: int, block: int) -> bool:
        if day < self.start_day or day > self.end_day:
            return False
        if day == self.start_day and block < self.start_block:
            return False
        if day == self.end_day and block > self.end_block:
            return False
        return True

    def start_tick(self, steps_per_day: int) -> int:
        return int(self.start_day) * int(steps_per_day) + int(self.start_block)

    def end_tick(self, steps_per_day: int) -> int:
        return int(self.end_day) * int(steps_per_day) + min(
            int(self.end_block), int(steps_per_day) - 1
        )


ShockEvent = _Event  # public name for scripted scenarios


@dataclass
class ShockScenario:
    """A scripted episode: fixed opening date plus pre-scheduled events.

    Used for scenario evaluation and diagnostics (e.g. "a two-day storm with
    transit suspended starting on day 3").  Random arrivals still follow the
    process configuration, so a pure scenario uses a config whose hazards are 0.
    ``forecasts`` maps a day index to the public storm warning for that day.
    """

    start_date: _dt.date
    events: List[_Event] = field(default_factory=list)
    forecasts: Dict[int, bool] = field(default_factory=dict)


class ShockProcess:
    """Dated shock schedule for one episode, revealed day by day."""

    def __init__(self, config: ShockConfig, seed: int, steps_per_day: int = 6) -> None:
        self.config = config
        self.seed = int(seed)
        self.steps_per_day = int(max(1, steps_per_day))
        self.rng = np.random.default_rng([self.seed, 27_027])
        self.start_date: _dt.date = _dt.date(2023, 1, 9)
        self.day_index: int = -1
        self.events: List[_Event] = []
        self.drift_log_level: float = 0.0
        self.backlog: float = 0.0  # deferred requests (realized-request units)
        self.release_rate: float = 0.0
        self.storm_forecast: Dict[int, bool] = {}
        self.last_effect = ShockEffect()
        self.scenario: Optional[ShockScenario] = None

    def snapshot(self) -> Dict[str, Any]:
        """Return all mutable process state needed for exact branch replay.

        ``MarketInteraction`` is deep-copied by the existing counterfactual
        machinery, but an explicit contract makes deterministic CRN tests and
        external planners independent of that implementation detail.
        """
        return {
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
            "start_date": self.start_date.isoformat(),
            "day_index": int(self.day_index),
            "events": copy.deepcopy(self.events),
            "drift_log_level": float(self.drift_log_level),
            "backlog": float(self.backlog),
            "release_rate": float(self.release_rate),
            "storm_forecast": copy.deepcopy(self.storm_forecast),
            "last_effect": copy.deepcopy(self.last_effect),
            "scenario": copy.deepcopy(self.scenario),
            "steps_per_day": int(self.steps_per_day),
        }

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        """Restore a :meth:`snapshot` without drawing any random numbers."""
        self.rng.bit_generator.state = copy.deepcopy(snapshot["rng_state"])
        self.start_date = _dt.date.fromisoformat(str(snapshot["start_date"]))
        self.day_index = int(snapshot["day_index"])
        self.events = copy.deepcopy(list(snapshot["events"]))
        self.drift_log_level = float(snapshot["drift_log_level"])
        self.backlog = float(snapshot["backlog"])
        self.release_rate = float(snapshot["release_rate"])
        self.storm_forecast = copy.deepcopy(dict(snapshot["storm_forecast"]))
        self.last_effect = copy.deepcopy(snapshot["last_effect"])
        self.scenario = copy.deepcopy(snapshot.get("scenario"))
        self.steps_per_day = int(snapshot.get("steps_per_day", self.steps_per_day))

    # -------------------------------------------------------------- episode
    def reset(self, *, start_date: Optional[_dt.date] = None, episode_seed: Optional[int] = None,
              training_opening: bool = False) -> None:
        if episode_seed is not None:
            self.rng = np.random.default_rng([int(episode_seed), 27_028])
        self.day_index = -1
        self.events = []
        self.drift_log_level = 0.0
        self.backlog = 0.0
        self.release_rate = 0.0
        self.storm_forecast = {}
        self.last_effect = ShockEffect()
        if self.scenario is not None:
            # Scripted episode: fixed date, pre-scheduled events and warnings.
            self.start_date = self.scenario.start_date
            self.events = [replace(e, detail=dict(e.detail)) for e in self.scenario.events]
            self.storm_forecast = {int(k): bool(v) for k, v in self.scenario.forecasts.items()}
            return
        if start_date is None:
            start_date = self._sample_start_date(training_opening)
        self.start_date = start_date

    def _sample_start_date(self, training_opening: bool) -> _dt.date:
        year = int(self.rng.choice([2022, 2023]))
        if training_opening and self.rng.random() < self.config.training_shock_opening_share:
            # Place the opening a few days before a holiday or in storm season
            # so rare windows are learned (training only; evaluation uses the
            # natural calendar or an explicit scenario date).
            if self.rng.random() < 0.5:
                name = str(self.rng.choice([h[0] for h in HOLIDAY_RULES]))
                anchor = holiday_dates(year)[name]
                return anchor - _dt.timedelta(days=int(self.rng.integers(2, 8)))
            month = int(self.rng.choice([12, 1, 2]))
            return _dt.date(year if month == 12 else year + 1, month, int(self.rng.integers(1, 28)))
        return _dt.date(year, 1, 1) + _dt.timedelta(days=int(self.rng.integers(0, 365)))

    @property
    def current_date(self) -> _dt.date:
        return self.start_date + _dt.timedelta(days=max(0, self.day_index))

    def date_for(self, day_index: int) -> _dt.date:
        return self.start_date + _dt.timedelta(days=int(day_index))

    def _recovery_blocks(self, event: _Event) -> int:
        """Physical recovery tail retained after an event's active window."""
        if "recovery_blocks" in event.detail:
            return int(max(0, event.detail["recovery_blocks"]))
        if event.kind == "driver_shortage":
            return int(max(0, self.config.shortage_recovery_days)) * self.steps_per_day
        if event.kind == "winter_storm":
            return int(max(0, self.config.storm_supply_recovery_days)) * self.steps_per_day
        if event.kind in {"large_event", "platform_outage"}:
            return 1
        return 0

    def _phase(self, event: _Event, day: int, block: int) -> Tuple[str, float, float]:
        """Return ``(phase, progress, physical_strength)`` for an event.

        Progress is within the named phase.  Strength is one at the peak and
        decays linearly through recovery.  Forecasts are public information
        only and therefore have zero physical strength.
        """
        steps = self.steps_per_day
        tick = int(day) * steps + int(block)
        start = event.start_tick(steps)
        end = event.end_tick(steps)
        if tick < start:
            lead = start - tick
            forecastable = event.announced or event.kind in {"holiday_wave", "winter_storm"}
            if forecastable and lead <= max(steps, self.config.storm_forecast_lead_days * steps):
                return "forecast", float(np.clip(1.0 - lead / max(1, steps), 0.0, 1.0)), 0.0
            return "inactive", 0.0, 0.0
        if tick <= end:
            span = max(1, end - start + 1)
            progress = float((tick - start + 1) / span)
            # The first quarter is a ramp for multi-block/day events.  A
            # one-block event is its own peak.  The label exposes onset to the
            # diagnostics; configured event magnitudes remain the physical
            # effect throughout the active window for v27 compatibility.
            if span > 1 and progress <= 0.25:
                return "ramp", float(np.clip(progress / 0.25, 0.0, 1.0)), 1.0
            return "peak", progress, 1.0
        recovery = self._recovery_blocks(event)
        if recovery > 0 and tick <= end + recovery:
            elapsed = tick - end
            progress = float(np.clip(elapsed / recovery, 0.0, 1.0))
            return "recovery", progress, float(max(0.0, 1.0 - progress))
        return "inactive", 0.0, 0.0

    def _keep_event(self, event: _Event, day: int) -> bool:
        end_tick = event.end_tick(self.steps_per_day) + self._recovery_blocks(event)
        return end_tick >= (int(day) - 1) * self.steps_per_day

    # ------------------------------------------------------------------ days
    def advance_day(self) -> _dt.date:
        """Reveal the next day; schedule events that start then (or are forecast)."""
        self.day_index += 1
        day = self.day_index
        c = self.config
        types = set(c.types)
        date = self.date_for(day)
        # Demand drift
        if "demand_drift" in types:
            self.drift_log_level = (
                c.drift_persistence * self.drift_log_level
                + float(self.rng.normal(0.0, c.drift_sd)) * c.intensity_scale
            )
        # Holidays (deterministic calendar), scheduled once per year in view
        if "holiday_wave" in types:
            for name, _, pre, post in HOLIDAY_RULES:
                for year in (date.year - 1, date.year, date.year + 1):
                    anchor = holiday_dates(year)[name]
                    first = (anchor - _dt.timedelta(days=pre) - self.start_date).days
                    last = (anchor + _dt.timedelta(days=post) - self.start_date).days
                    if first <= day + c.storm_forecast_lead_days and last >= day and not any(
                        e.kind == "holiday_wave" and e.detail.get("name") == name and e.start_day == first
                        for e in self.events
                    ):
                        self.events.append(_Event(
                            "holiday_wave", first, last, intensity=1.0,
                            detail={"name": name, "anchor_day": (anchor - self.start_date).days,
                                    "pre": pre, "post": post},
                        ))
        # Storms: drawn one day ahead so a (noisy) forecast exists
        if "winter_storm" in types:
            target = day + int(c.storm_forecast_lead_days)
            if target not in self.storm_forecast:
                tdate = self.date_for(target)
                hazard = float(c.storm_daily_hazard.get(tdate.month, 0.0))
                busy = any(e.kind == "winter_storm" and e.covers(target, 0) for e in self.events)
                occurs = (not busy) and self.rng.random() < hazard
                if occurs:
                    length = int(self.rng.integers(c.storm_duration_days[0], c.storm_duration_days[1] + 1))
                    suspended = bool(self.rng.random() < c.storm_transit_suspension_prob)
                    self.events.append(_Event(
                        "winter_storm", target, target + length - 1,
                        intensity=float(self.rng.uniform(0.7, 1.3)),
                        detail={"transit_suspended": suspended},
                    ))
                forecast = (occurs and self.rng.random() >= c.storm_forecast_miss_prob) or (
                    (not occurs) and self.rng.random() < c.storm_false_alarm_prob
                )
                self.storm_forecast[target] = bool(forecast)
        blocks = self.steps_per_day
        # Large events (single day, 1-3 blocks, mostly evenings)
        if "large_event" in types and self.rng.random() < c.event_rate_per_day:
            length = int(self.rng.integers(1, 4))
            start_block = int(self.rng.integers(max(0, blocks // 2), blocks))
            self.events.append(_Event(
                "large_event", day, day, intensity=float(self.rng.uniform(*c.event_volume_multiplier)),
                start_block=start_block, end_block=min(blocks - 1, start_block + length - 1),
                announced=bool(self.rng.random() < c.event_announce_prob),
            ))
        if "driver_shortage" in types and self.rng.random() < c.shortage_rate_per_day:
            length = int(self.rng.integers(c.shortage_duration_days[0], c.shortage_duration_days[1] + 1))
            self.events.append(_Event(
                "driver_shortage", day, day + length - 1,
                intensity=float(self.rng.uniform(*c.shortage_fleet)), announced=False,
            ))
        if "platform_outage" in types and self.rng.random() < c.outage_rate_per_day:
            start_block = int(self.rng.integers(0, blocks))
            length = int(self.rng.integers(c.outage_blocks[0], c.outage_blocks[1] + 1))
            self.events.append(_Event(
                "platform_outage", day, day, intensity=float(self.rng.uniform(*c.outage_capacity)),
                start_block=start_block, end_block=min(blocks - 1, start_block + length - 1),
                announced=False, firm=str(self.rng.choice(["Firm1", "Firm2"])),
            ))
        if "transit_disruption" in types and self.rng.random() < c.transit_disruption_rate_per_day:
            length = int(self.rng.integers(c.transit_disruption_days[0], c.transit_disruption_days[1] + 1))
            self.events.append(_Event("transit_disruption", day, day + length - 1))
        if "rival_promotion" in types and self.rng.random() < c.rival_promotion_rate_per_day:
            length = int(self.rng.integers(c.rival_promotion_days[0], c.rival_promotion_days[1] + 1))
            self.events.append(_Event("rival_promotion", day, day + length - 1, announced=False))
        # Keep physical recovery tails (especially driver supply) instead of
        # making capacity jump back to normal at midnight after an event.
        self.events = [e for e in self.events if self._keep_event(e, day)]
        return date

    def day_weather_override(self) -> Optional[str]:
        """Weather forced by a shock on the current day (a storm day is snowy)."""
        day = self.day_index
        if day < 0 or "winter_storm" not in self.config.types or self.config.intensity_scale <= 0.0:
            return None
        for e in self.events:
            if e.kind == "winter_storm" and e.start_day <= day <= e.end_day:
                return "snow"
        return None

    # ----------------------------------------------------------------- block
    def effect(self, block: int) -> ShockEffect:
        """Composite effect for the current day and intra-day block."""
        c = self.config
        day = max(0, self.day_index)
        effect = ShockEffect()
        active: List[str] = []
        public = {key: 0.0 for key in PUBLIC_SHOCK_FEATURE_KEYS}
        public["transit_availability"] = 1.0
        s = c.scaled
        volume = float(np.exp(self.drift_log_level))

        phase_candidates: List[Tuple[int, float, str, float, str]] = []

        def phased(target: float, strength: float) -> float:
            """Interpolate a multiplier through the phase, then scale severity."""
            return s(1.0 + (float(target) - 1.0) * float(np.clip(strength, 0.0, 1.0)))

        def segment(channel: Dict[str, float], name: str, multiplier: float) -> None:
            channel[name] = float(channel.get(name, 1.0)) * float(multiplier)

        for e in self.events:
            phase, progress, strength = self._phase(e, day, int(block))
            if phase != "inactive":
                effect.event_phases[e.kind] = phase
                priority = {"peak": 4, "ramp": 3, "recovery": 2, "forecast": 1}.get(phase, 0)
                phase_candidates.append((priority, float(e.intensity), phase, progress, e.kind))
            if e.kind == "holiday_wave":
                anchor_day = int(e.detail["anchor_day"])
                distance = day - anchor_day
                if not e.covers(day, block):
                    continue
                # Holiday phases are tied to the travel wave rather than the
                # event object's generic quarter-span ramp.
                if distance < 0:
                    phase, progress = "ramp", float(np.clip(1.0 + distance / max(1, int(e.detail.get("pre", 1))), 0.0, 1.0))
                elif distance == 0:
                    phase, progress = "peak", 1.0
                else:
                    phase, progress = "recovery", float(np.clip(distance / max(1, int(e.detail.get("post", 1))), 0.0, 1.0))
                effect.event_phases[e.kind] = phase
                phase_candidates.append(({"peak": 4, "ramp": 3, "recovery": 2}[phase], 1.0, phase, progress, e.kind))
                active.append("holiday_wave")
                public["holiday_active"] = 1.0
                # Outbound wave before, return wave after, quieter holiday itself.
                edge = 1.0 if distance != 0 else 0.6
                airport_multiplier = s(1.0 + (c.holiday_airport_share_peak - 1.0) * edge)
                effect.airport_share_multiplier *= airport_multiplier
                effect.volume_multiplier *= s(c.holiday_commute_volume if day != anchor_day else 0.65)
                effect.airport_price_sensitivity_multiplier *= s(c.holiday_airport_price_sensitivity)
                effect.fleet_multiplier *= s(c.holiday_fleet)
                # Before the holiday, trips *to* the airport dominate; after
                # it, arriving passengers create trips *from* the airport.
                if distance < 0:
                    segment(effect.segment_volume_multiplier, "airport_destination", airport_multiplier)
                    segment(effect.segment_volume_multiplier, "airport_origin", phased(1.30, 1.0))
                elif distance > 0:
                    segment(effect.segment_volume_multiplier, "airport_origin", airport_multiplier)
                    segment(effect.segment_volume_multiplier, "airport_destination", phased(1.20, 1.0))
                else:
                    segment(effect.segment_volume_multiplier, "airport_origin", airport_multiplier)
                    segment(effect.segment_volume_multiplier, "airport_destination", airport_multiplier)
                segment(effect.segment_volume_multiplier, "commute", s(c.holiday_commute_volume))
                segment(effect.segment_volume_multiplier, "discretionary", phased(0.85, 1.0))
            elif e.kind == "winter_storm":
                if phase == "forecast" or phase == "inactive":
                    continue
                k = float(e.intensity)
                suspended = bool(e.detail.get("transit_suspended", False))
                if phase == "recovery":
                    # Demand/weather return normally; the driver stock
                    # recovers gradually rather than jumping at event end.
                    active.append("winter_storm_recovery")
                    supply_target = 1.0 - (1.0 - c.storm_fleet) * k
                    effect.fleet_multiplier *= phased(supply_target, strength)
                    effect.driver_supply_phase = "recovery"
                    continue
                active.append("winter_storm")
                public["storm_active"] = 1.0
                avail = 0.0 if suspended else c.storm_transit_degraded
                effect.transit_availability = min(effect.transit_availability, float(np.clip(
                    phased(avail, strength), 0.0, 1.0)))
                storm_target = 1.0 - (1.0 - c.storm_discretionary_volume) * k
                storm_factor = float(np.clip(phased(storm_target, strength), 0.05, 1.0))
                effect.volume_multiplier *= storm_factor
                effect.storm_volume_factor = min(effect.storm_volume_factor, storm_factor)
                effect.airport_share_multiplier *= phased(c.storm_airport_share, strength)
                effect.duration_multiplier *= phased(1.0 + (c.storm_duration_multiplier - 1.0) * k, strength)
                effect.fleet_multiplier *= phased(1.0 - (1.0 - c.storm_fleet) * k, strength)
                effect.reservation_multiplier *= phased(c.storm_reservation_multiplier, strength)
                effect.price_sensitivity_multiplier *= phased(c.storm_price_sensitivity, strength)
                effect.driver_supply_phase = phase
                segment(effect.segment_volume_multiplier, "discretionary", phased(c.storm_discretionary_volume, strength))
                segment(effect.segment_volume_multiplier, "commute", phased(0.65, strength))
                segment(effect.segment_volume_multiplier, "airport_origin", phased(c.storm_airport_share, strength))
                segment(effect.segment_volume_multiplier, "airport_destination", phased(c.storm_airport_share, strength))
                segment(effect.segment_volume_multiplier, "essential", phased(0.90, strength))
                segment(effect.segment_duration_multiplier, "airport_origin", phased(1.10, strength))
                segment(effect.segment_duration_multiplier, "airport_destination", phased(1.10, strength))
                segment(effect.segment_reservation_multiplier, "essential", phased(1.08, strength))
                effect.weather_override = "snow"
                effect.emergency = True
            elif e.kind == "large_event":
                if e.announced and phase in {"forecast", "ramp", "peak"}:
                    public["event_announced"] = 1.0
                if phase in {"forecast", "inactive"}:
                    continue
                name = "large_event_recovery" if phase == "recovery" else "large_event"
                active.append(name)
                effect.volume_multiplier *= phased(e.intensity, strength)
                effect.price_sensitivity_multiplier *= phased(c.event_price_sensitivity, strength)
                segment(effect.segment_volume_multiplier, "event", phased(max(1.0, e.intensity * 1.35), strength))
                segment(effect.segment_volume_multiplier, "discretionary", phased(1.10, strength))
                segment(effect.segment_duration_multiplier, "event", phased(1.18, strength))
                segment(effect.segment_price_sensitivity_multiplier, "event", phased(0.85, strength))
                segment(effect.segment_reservation_multiplier, "event", phased(1.12, strength))
                segment(effect.segment_outside_cost_multiplier, "event", phased(1.20, strength))
                segment(effect.segment_outside_inconvenience_multiplier, "event", phased(1.15, strength))
            elif e.kind == "driver_shortage":
                if phase in {"forecast", "inactive"}:
                    continue
                name = "driver_shortage_recovery" if phase == "recovery" else "driver_shortage"
                active.append(name)
                effect.fleet_multiplier *= phased(e.intensity, strength)
                effect.reservation_wage_multiplier *= phased(c.shortage_wage_multiplier, strength)
                effect.driver_supply_phase = phase
            elif e.kind == "platform_outage":
                if phase in {"forecast", "inactive"}:
                    continue
                active.append("platform_outage_recovery" if phase == "recovery" else "platform_outage")
                caps = dict(effect.firm_capacity_multiplier)
                caps[e.firm] = caps.get(e.firm, 1.0) * phased(e.intensity, strength)
                effect.firm_capacity_multiplier = caps
            elif e.kind == "transit_disruption":
                if not e.covers(day, block):
                    continue
                active.append("transit_disruption")
                effect.transit_availability = min(effect.transit_availability, float(np.clip(
                    1.0 - (1.0 - c.transit_disruption_availability) * c.intensity_scale, 0.0, 1.0)))
                segment(effect.segment_volume_multiplier, "commute", phased(1.30, 1.0))
                segment(effect.segment_volume_multiplier, "essential", phased(1.10, 1.0))
                segment(effect.segment_outside_cost_multiplier, "commute", phased(1.20, 1.0))
                segment(effect.segment_outside_inconvenience_multiplier, "commute", phased(1.25, 1.0))
            elif e.kind == "rival_promotion":
                if not e.covers(day, block):
                    continue
                active.append("rival_promotion")
                effect.firm2_price_multiplier *= s(c.rival_promotion_multiplier)
        if "holiday_wave" in c.types:
            # Public calendar knowledge: closeness to the nearest holiday anchor
            # (1 on the day, 0 at a week or more away), known well in advance.
            date = self.date_for(day)
            nearest = min(
                abs((anchor - date).days)
                for year in (date.year - 1, date.year, date.year + 1)
                for anchor in holiday_dates(year).values()
            )
            public["holiday_proximity"] = float(np.clip(1.0 - nearest / 7.0, 0.0, 1.0))
        warning = 1.0 if self.storm_forecast.get(day + int(c.storm_forecast_lead_days), False) else 0.0
        public["storm_warning"] = max(warning, public["storm_active"])
        public["transit_availability"] = float(effect.transit_availability)
        effect.volume_multiplier *= volume
        effect.active = tuple(sorted(set(active)))
        effect.public = public
        if phase_candidates:
            _, severity, phase, progress, _ = max(
                phase_candidates, key=lambda item: (item[0], item[1])
            )
            effect.phase = str(phase)
            effect.phase_progress = float(progress)
            effect.severity = float(severity)
        self.last_effect = effect
        return effect

    # -------------------------------------------------------- deferred demand
    def defer(self, suppressed_requests: float) -> None:
        """Queue a share of storm-suppressed requests (realized-request units)."""
        amount = max(0.0, float(suppressed_requests)) * float(self.config.storm_deferred_share)
        if amount > 0.0:
            self.backlog += amount

    def release_backlog(self, storm_active: bool) -> float:
        """Deferred requests released this step.

        Nothing is released while a storm is active; afterwards the backlog
        drains at a constant rate over ``storm_rebound_days``.
        """
        if storm_active or self.backlog <= 0.0:
            return 0.0
        if self.release_rate <= 0.0:
            steps = int(max(1, self.config.storm_rebound_days)) * self.steps_per_day
            self.release_rate = self.backlog / float(steps)
        amount = min(self.backlog, self.release_rate)
        self.backlog -= amount
        if self.backlog <= 1e-9:
            self.backlog = 0.0
            self.release_rate = 0.0
        return float(amount)

    def summary(self) -> Dict[str, Any]:
        return {
            "start_date": self.start_date.isoformat(),
            "day_index": int(self.day_index),
            "backlog": float(self.backlog),
            "events": [
                {"kind": e.kind, "start_day": e.start_day, "end_day": e.end_day,
                 "intensity": round(float(e.intensity), 4), "firm": e.firm,
                 "announced": e.announced, **{k: v for k, v in e.detail.items() if k != "anchor_day"}}
                for e in self.events
            ],
        }
