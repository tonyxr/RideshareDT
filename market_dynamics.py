"""City-calibrated operational market dynamics (v26 environment contract).

This module carries the environment changes introduced after the 2026-10-05
review.  Every component is data-driven when a city's processed bundle is
available and is switched off by default so legacy (v25) runs and tests stay
reproducible.

Components
----------
``MarketDynamicsConfig``
    Feature switches recorded in every run configuration and model artifact.
``TripBank``
    Empirical operational trips (distance, duration, airport, service, hour,
    weekday) resampled from the city's normalized public ride records.  This
    replaces the former city-independent ``Exponential(4 mi)`` trip draw.
``MarketCalendar``
    Sequential calendar: consecutive weekdays, persistent weather (a Markov
    chain whose stationary law equals the calibrated weather frequencies), and
    ordered intra-day time blocks whose request volume follows the calibrated
    hour-of-day profile.  Demand volume and partially tracking driver supply
    therefore move predictably through the day, which gives a forward-looking
    pricing policy something to anticipate.
``ControllerPerturbation``
    Seeded parameter perturbation of rule-based competitors for opponent
    domain randomization (training) and out-of-distribution evaluation.
``market_relative_config``
    Tariff bounds and step sizes expressed relative to the city's anchor tariff
    so a new city is not clipped by New-York-dollar bounds.
"""

from __future__ import annotations

import csv
import functools
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

V26_ENVIRONMENT_CONTRACT = "v26_city_calibrated_sequential_market"

TARIFF_KEYS: Tuple[str, ...] = (
    "base_fare",
    "per_minute",
    "per_mile",
    "booking_fee",
    "airport_fee",
)

# Step sizes of the historical NYC controller configuration expressed as a
# fraction of the NYC anchor tariff.  Re-expressing them relatively preserves
# the controllers' NYC behavior while scaling correctly to other cities.
RELATIVE_STEP: Dict[str, float] = {
    "base_fare": 0.10 / 3.00,
    "per_minute": 0.01 / 0.40,
    "per_mile": 0.05 / 1.90,
    "booking_fee": 0.10 / 2.75,
    "airport_fee": 0.25 / 6.00,
}
RELATIVE_BOUNDS: Dict[str, Tuple[float, float]] = {
    "base_fare": (0.40, 2.00),
    "per_minute": (0.40, 2.00),
    "per_mile": (0.40, 2.00),
    "booking_fee": (0.00, 2.00),
    "airport_fee": (0.00, 2.00),
}

DEFAULT_RUSH_SPEED_MPH = 18.0
DEFAULT_OFFPEAK_SPEED_MPH = 25.0


@dataclass
class MarketDynamicsConfig:
    """Switches for the v26 city-calibrated sequential market."""

    enabled: bool = False
    sequential_calendar: bool = True
    weather_persistence: float = 0.70
    hourly_demand_volume: bool = True
    demand_noise: str = "poisson"
    max_volume_multiple: float = 4.0
    supply_tracking: float = 0.50
    empirical_trips: bool = True
    trip_hour_window: int = 1
    trip_jitter: float = 0.03
    price_level_scaling: bool = True
    symmetric_firms: bool = True
    linear_driver_scaling: bool = True
    driver_reference_customers: float = 500.0
    market_relative_bounds: bool = True
    calibrated_tariff_anchors: bool = True
    # A rider whose chosen platform cannot dispatch a driver re-requests on the
    # other platform (multi-homing) with this probability; the retry is less
    # likely when the other quote exceeds the rider's reservation fare.  In the
    # v25 market such riders vanished, so a firm could raise its completed-
    # share *lead* by absorbing demand it could not serve.
    unfulfilled_spillover: bool = True
    spillover_probability: float = 0.75
    # Rider memory (forward-looking demand channel, off by default).  Riders
    # remember each platform's recent tariff level and dispatch reliability as
    # exponentially smoothed reputations (reference-price / goodwill model).
    # Platform choice tilts toward the cheaper-remembered and more reliable
    # firm, and a high remembered market price level raises the outside-option
    # probability, so today's tariff moves tomorrow's demand.
    rider_memory: bool = False
    memory_alpha: float = 0.10
    memory_price_weight: float = 1.50
    memory_reliability_weight: float = 1.00
    memory_outside_weight: float = 1.00
    # Driver online-supply response to recent earnings (existing v25 channel).
    supply_feedback: bool = True
    # Driver pay basis.  ``fare`` (v25): pay = share x the rider's fare, so a
    # rider discount cuts driver pay and acceptance one-for-one.  ``anchor``:
    # pay = share x the city-anchor fare for the same trip/context (surge
    # included), i.e. platform-funded discounts as under upfront driver pay and
    # per-mile/minute pay floors.
    driver_pay_basis: str = "anchor"
    # Fleet-capacity calibration.  The legacy fleet (260 drivers per 500
    # requests, one trip per driver per step) puts the symmetric anchor market
    # on the edge of capacity, so any undercut mostly creates unserved
    # requests.  With a positive target, ``base_active_drivers`` is set by
    # bisection so that both firms at the anchor tariff fulfil this share of
    # dispatches over a weekday (CRN, all blocks).  0 keeps the legacy fleet.
    capacity_target_fulfillment: float = 0.95
    # Multiplies the (calibrated) fleet: a sensitivity knob for how much idle
    # capacity a price cut can draw on (1.0 = calibrated market).
    fleet_scale: float = 1.0

    def __post_init__(self) -> None:
        self.weather_persistence = float(np.clip(self.weather_persistence, 0.0, 0.98))
        self.supply_tracking = float(np.clip(self.supply_tracking, 0.0, 1.0))
        self.max_volume_multiple = float(max(1.0, self.max_volume_multiple))
        self.trip_hour_window = int(max(0, self.trip_hour_window))
        self.trip_jitter = float(np.clip(self.trip_jitter, 0.0, 0.25))
        self.driver_reference_customers = float(max(1.0, self.driver_reference_customers))
        self.spillover_probability = float(np.clip(self.spillover_probability, 0.0, 1.0))
        self.memory_alpha = float(np.clip(self.memory_alpha, 0.0, 1.0))
        self.memory_price_weight = float(max(0.0, self.memory_price_weight))
        self.memory_reliability_weight = float(max(0.0, self.memory_reliability_weight))
        self.memory_outside_weight = float(max(0.0, self.memory_outside_weight))
        self.capacity_target_fulfillment = float(np.clip(self.capacity_target_fulfillment, 0.0, 0.995))
        self.fleet_scale = float(np.clip(self.fleet_scale, 0.25, 4.0))
        if str(self.driver_pay_basis) not in {"fare", "anchor"}:
            raise ValueError("driver_pay_basis must be 'fare' or 'anchor'")
        if str(self.demand_noise) not in {"poisson", "none"}:
            raise ValueError("demand_noise must be 'poisson' or 'none'")

    def as_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["contract"] = V26_ENVIRONMENT_CONTRACT if self.enabled else "legacy_v25_market"
        return payload


# --------------------------------------------------------------------------
# Empirical trips
# --------------------------------------------------------------------------
def _parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def _parse_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def _airport_location(value: Any, city: str) -> bool:
    """Identify an airport endpoint from the normalized trip-bank location."""
    text = str(value or "").strip().lower()
    if any(token in text for token in (
        "airport", "jfk", "laguardia", "lga", "newark", "ewr", "o'hare", "ohare", "midway",
    )):
        return True
    try:
        location_id = int(float(text))
    except (TypeError, ValueError):
        return False
    city_key = str(city).strip().lower()
    if "new york" in city_key:
        return location_id in {1, 132, 138}
    if "chicago" in city_key:
        return location_id in {56, 76}
    return False


def default_duration_minutes(distance: np.ndarray, hour: np.ndarray) -> np.ndarray:
    rush = ((hour >= 7) & (hour < 10)) | ((hour >= 16) & (hour < 19))
    speed = np.where(rush, DEFAULT_RUSH_SPEED_MPH, DEFAULT_OFFPEAK_SPEED_MPH)
    return np.maximum(3.0, 60.0 * distance / speed)


class TripBank:
    """Immutable empirical trip distribution for one city.

    Sampling is conditional on hour-of-day (with a +/- window) and on the
    weekday/weekend split, falling back progressively when a cell is sparse.
    When the source has no observed duration (Boston quote data), durations
    come from an hour-dependent speed profile; that fallback is reported in
    ``provenance`` rather than silently presented as data.
    """

    MIN_CELL = 25

    def __init__(
        self,
        *,
        city: str,
        distance: np.ndarray,
        duration: np.ndarray,
        hour: np.ndarray,
        day_of_week: np.ndarray,
        airport: np.ndarray,
        service: np.ndarray,
        price: np.ndarray,
        weather: Optional[np.ndarray] = None,
        pickup_location: Optional[np.ndarray] = None,
        dropoff_location: Optional[np.ndarray] = None,
        airport_identified: bool = True,
        service_identified: bool = True,
        source: str = "",
    ) -> None:
        self.city = str(city)
        self.distance = np.asarray(distance, dtype=float)
        hour_arr = np.asarray(hour, dtype=int) % 24
        raw_duration = np.asarray(duration, dtype=float)
        observed = np.isfinite(raw_duration) & (raw_duration > 0.0)
        self.duration_observed_fraction = float(np.mean(observed)) if observed.size else 0.0
        if self.duration_observed_fraction >= 0.5:
            fallback = default_duration_minutes(self.distance, hour_arr)
            self.duration = np.where(observed, raw_duration, fallback)
            self.duration_source = "observed"
        else:
            self.duration = default_duration_minutes(self.distance, hour_arr)
            self.duration_source = "assumed_hourly_speed_profile"
        self.hour = hour_arr
        self.day_of_week = np.asarray(day_of_week, dtype=int) % 7
        self.airport = np.asarray(airport, dtype=bool)
        self.service = np.asarray(service, dtype=object)
        self.price = np.asarray(price, dtype=float)
        if weather is None:
            self.weather = np.full(self.distance.size, "unknown", dtype=object)
        else:
            self.weather = np.array(
                [str(value or "unknown").strip().lower() or "unknown" for value in weather],
                dtype=object,
            )
        pickup = (
            np.full(self.distance.size, "", dtype=object)
            if pickup_location is None else np.asarray(pickup_location, dtype=object)
        )
        dropoff = (
            np.full(self.distance.size, "", dtype=object)
            if dropoff_location is None else np.asarray(dropoff_location, dtype=object)
        )
        self.pickup_location = pickup.copy()
        self.dropoff_location = dropoff.copy()
        self.airport_origin = np.array(
            [_airport_location(value, self.city) for value in pickup], dtype=bool
        ) & self.airport
        self.airport_destination = np.array(
            [_airport_location(value, self.city) for value in dropoff], dtype=bool
        ) & self.airport
        self.airport_identified = bool(airport_identified)
        self.service_identified = bool(service_identified)
        self.source = str(source)
        if self.distance.size == 0:
            raise ValueError(f"empty trip bank for {city}")
        weekend = self.day_of_week >= 5
        self._cells: Dict[Tuple[int, int], np.ndarray] = {}
        self._hour_cells: Dict[int, np.ndarray] = {}
        self._all = np.arange(self.distance.size)
        for h in range(24):
            in_hour = self.hour == h
            self._hour_cells[h] = np.flatnonzero(in_hour)
            for w in (0, 1):
                self._cells[(h, w)] = np.flatnonzero(in_hour & (weekend == bool(w)))

    def __deepcopy__(self, memo: Dict[int, Any]) -> "TripBank":
        # Immutable: counterfactual branch snapshots must not copy the bank.
        return self

    def __len__(self) -> int:
        return int(self.distance.size)

    @classmethod
    def from_csv(
        cls,
        path: Path | str,
        *,
        city: str,
        airport_identified: bool = True,
        service_identified: bool = True,
        max_distance_miles: float = 60.0,
    ) -> "TripBank":
        columns: Dict[str, List[Any]] = {
            key: [] for key in (
                "distance", "duration", "hour", "dow", "airport", "service", "price",
                "weather", "pickup_location", "dropoff_location",
            )
        }
        with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                distance = _parse_float(row.get("distance_miles"))
                if not (0.05 <= distance <= max_distance_miles):
                    continue
                columns["distance"].append(distance)
                columns["duration"].append(_parse_float(row.get("duration_minutes")))
                columns["hour"].append(int(_parse_float(row.get("hour")) if row.get("hour") not in (None, "") else 12))
                columns["dow"].append(int(_parse_float(row.get("day_of_week")) if row.get("day_of_week") not in (None, "") else 2))
                columns["airport"].append(_parse_bool(row.get("airport")))
                columns["service"].append(str(row.get("service") or "economy").strip().lower() or "economy")
                columns["price"].append(_parse_float(row.get("actual_price")))
                columns["weather"].append(str(row.get("weather") or "unknown"))
                columns["pickup_location"].append(row.get("pickup_location") or "")
                columns["dropoff_location"].append(row.get("dropoff_location") or "")
        return cls(
            city=city,
            distance=np.asarray(columns["distance"]),
            duration=np.asarray(columns["duration"]),
            hour=np.asarray(columns["hour"]),
            day_of_week=np.asarray(columns["dow"]),
            airport=np.asarray(columns["airport"]),
            service=np.asarray(columns["service"], dtype=object),
            price=np.asarray(columns["price"]),
            weather=np.asarray(columns["weather"], dtype=object),
            pickup_location=np.asarray(columns["pickup_location"], dtype=object),
            dropoff_location=np.asarray(columns["dropoff_location"], dtype=object),
            airport_identified=airport_identified,
            service_identified=service_identified,
            source=str(path),
        )

    def _candidates(self, hour: int, weekend: bool, window: int) -> np.ndarray:
        cell = self._cells.get((int(hour) % 24, int(bool(weekend))))
        if cell is not None and cell.size >= self.MIN_CELL:
            return cell
        pooled = [
            self._hour_cells[(int(hour) + offset) % 24]
            for offset in range(-window, window + 1)
        ]
        merged = np.concatenate(pooled) if pooled else np.empty(0, dtype=int)
        if merged.size >= self.MIN_CELL:
            return merged
        return self._all

    def sample(
        self,
        rng: np.random.Generator,
        *,
        hour: int,
        day_of_week: int,
        size: int,
        window: int = 1,
        jitter: float = 0.03,
        weather: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        count = int(max(0, size))
        candidates = self._candidates(hour, int(day_of_week) >= 5, window)
        requested_weather = str(weather or "").strip().lower()
        if requested_weather:
            exact = candidates[self.weather[candidates] == requested_weather]
            if exact.size >= self.MIN_CELL:
                candidates = exact
            elif requested_weather in {"rain", "snow", "storm"}:
                adverse = candidates[np.isin(self.weather[candidates], ["rain", "snow", "storm"])]
                if adverse.size >= self.MIN_CELL:
                    candidates = adverse
        index = candidates[rng.integers(0, candidates.size, size=count)]
        scale = (
            np.exp(rng.normal(0.0, jitter, size=count)) if jitter > 0.0 else np.ones(count)
        )
        distance = np.round(np.clip(self.distance[index] * scale, 0.05, None), 2)
        duration = np.clip(self.duration[index] * scale, 1.0, 240.0)
        return {
            "distance": distance,
            "duration": duration,
            "airport": self.airport[index],
            "airport_origin": self.airport_origin[index],
            "airport_destination": self.airport_destination[index],
            "service": self.service[index],
            "source_weather": self.weather[index],
            "pickup_location": self.pickup_location[index],
            "dropoff_location": self.dropoff_location[index],
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "city": self.city,
            "rows": len(self),
            "source": self.source,
            "distance_mean": float(np.mean(self.distance)),
            "distance_median": float(np.median(self.distance)),
            "duration_mean": float(np.mean(self.duration)),
            "duration_source": self.duration_source,
            "duration_observed_fraction": self.duration_observed_fraction,
            "airport_rate": float(np.mean(self.airport)),
            "airport_identified": self.airport_identified,
            "airport_origin_rate": float(np.mean(self.airport_origin)),
            "airport_destination_rate": float(np.mean(self.airport_destination)),
            "weather_identified_fraction": float(np.mean(self.weather != "unknown")),
            "service_identified": self.service_identified,
        }


# --------------------------------------------------------------------------
# Sequential calendar and demand volume
# --------------------------------------------------------------------------
@dataclass
class StepContext:
    day_index: int
    day_of_week: int
    weather: str
    block: int
    hour: int
    volume_multiplier: float
    supply_multiplier: float


@dataclass
class MarketCalendar:
    """Calendar state; lives inside ``MarketInteraction`` so CRN snapshots copy it."""

    day_index: int = -1
    day_of_week: int = 0
    weather: str = "clear"
    block: int = -1
    steps_per_day: int = 6
    _day_hours: List[int] = field(default_factory=list)
    _day_block_shares: List[float] = field(default_factory=list)

    def reset(self, rng: np.random.Generator, *, day_of_week_probs: np.ndarray, weather_probs: Mapping[str, float]) -> None:
        self.day_index = -1
        self.block = -1
        self.day_of_week = int(rng.choice(7, p=_normalized(day_of_week_probs)))
        self.weather = _draw_weather(rng, weather_probs)
        self._day_hours = []
        self._day_block_shares = []

    def advance_day(
        self,
        rng: np.random.Generator,
        *,
        steps_per_day: int,
        hour_probs: np.ndarray,
        weather_probs: Mapping[str, float],
        persistence: float,
    ) -> None:
        if self.day_index >= 0:
            self.day_of_week = (self.day_of_week + 1) % 7
            if float(rng.random()) >= float(persistence):
                self.weather = _draw_weather(rng, weather_probs)
        self.day_index += 1
        self.block = -1
        self.steps_per_day = int(max(1, steps_per_day))
        probs = _normalized(hour_probs)
        hours: List[int] = []
        shares: List[float] = []
        for lo, hi in block_hour_ranges(self.steps_per_day):
            block_hours = np.arange(lo, hi) % 24
            weights = probs[block_hours]
            share = float(weights.sum())
            within = weights / share if share > 0 else np.ones_like(weights) / len(weights)
            hours.append(int(rng.choice(block_hours, p=within)))
            shares.append(share)
        self._day_hours = hours
        self._day_block_shares = shares

    def next_block(self) -> Tuple[int, int, float]:
        self.block += 1
        if self.block >= len(self._day_hours):
            raise RuntimeError("calendar day exhausted; call advance_day first")
        return self.block, self._day_hours[self.block], self._day_block_shares[self.block]

    @property
    def day_exhausted(self) -> bool:
        return self.day_index < 0 or self.block + 1 >= len(self._day_hours)


@functools.lru_cache(maxsize=64)
def block_hour_ranges(steps_per_day: int) -> Tuple[Tuple[int, int], ...]:
    """Integer [lo, hi) hour range of each intra-day block.

    The single definition used by the calendar, the hourly volume/supply
    multipliers, and the observed-fare trajectory lookup, so all three agree
    for any number of blocks (including ones that do not divide 24).
    """
    steps = int(max(1, steps_per_day))
    edges = np.linspace(0, 24, steps + 1)
    ranges = []
    for b in range(steps):
        lo, hi = int(round(edges[b])), int(round(edges[b + 1]))
        ranges.append((lo, max(hi, lo + 1)))
    return tuple(ranges)


def block_index_for_hour(hour: int, steps_per_day: int) -> int:
    hour = int(hour) % 24
    ranges = block_hour_ranges(int(max(1, steps_per_day)))
    for index, (lo, hi) in enumerate(ranges):
        if lo <= hour < hi:
            return index
    return len(ranges) - 1


def _normalized(values: Iterable[float]) -> np.ndarray:
    arr = np.clip(np.asarray(list(values), dtype=float), 0.0, None)
    total = float(arr.sum())
    if total <= 0.0:
        return np.full(arr.size, 1.0 / max(1, arr.size))
    return arr / total


def _draw_weather(rng: np.random.Generator, weather_probs: Mapping[str, float]) -> str:
    keys = list(weather_probs.keys())
    probs = _normalized(weather_probs[key] for key in keys)
    return str(rng.choice(keys, p=probs))


def expected_volume_multiplier(
    *,
    block_share: float,
    steps_per_day: int,
    day_of_week: int,
    day_of_week_probs: np.ndarray,
    weather: str,
    weather_demand_multiplier: Mapping[str, float],
) -> float:
    """Expected requests relative to the run's base customers per step."""
    hourly = float(block_share) * float(max(1, steps_per_day))
    weekday = 7.0 * float(_normalized(day_of_week_probs)[int(day_of_week) % 7])
    weather_factor = float(weather_demand_multiplier.get(str(weather), 1.0))
    return float(max(0.0, hourly * weekday * weather_factor))


def supply_multiplier(*, block_share: float, steps_per_day: int, tracking: float) -> float:
    """Driver online multiplier: tracks only part of the hourly demand cycle."""
    hourly = float(block_share) * float(max(1, steps_per_day))
    return float((1.0 - tracking) + tracking * hourly)


def realized_request_count(
    rng: np.random.Generator,
    *,
    base_customers: int,
    multiplier: float,
    noise: str,
    max_multiple: float,
) -> int:
    expected = float(max(0.0, base_customers * multiplier))
    expected = min(expected, float(base_customers) * float(max_multiple))
    if noise == "poisson":
        count = int(rng.poisson(expected))
    else:
        count = int(round(expected))
    return int(max(1, count))


# --------------------------------------------------------------------------
# Tariff bounds relative to the city anchors
# --------------------------------------------------------------------------
def market_relative_bounds(anchors: Mapping[str, float]) -> Dict[str, Tuple[float, float]]:
    bounds: Dict[str, Tuple[float, float]] = {}
    for key in TARIFF_KEYS:
        anchor = float(anchors.get(key, 0.0))
        lo, hi = RELATIVE_BOUNDS[key]
        floor = 0.01 if key in {"base_fare", "per_minute", "per_mile"} else 0.0
        bounds[key] = (max(floor, lo * anchor), max(floor + 1e-6, hi * anchor))
    return bounds


def market_relative_steps(anchors: Mapping[str, float]) -> Dict[str, float]:
    return {
        key: float(max(1e-4, RELATIVE_STEP[key] * float(anchors.get(key, 0.0))))
        for key in TARIFF_KEYS
    }


def rescale_pricer_to_market(pricer: Any, anchors: Mapping[str, float]) -> None:
    """Replace absolute NYC-dollar bounds/steps on any pricer with relative ones."""
    bounds = market_relative_bounds(anchors)
    steps = market_relative_steps(anchors)
    config = getattr(pricer, "config", None)
    if config is not None and hasattr(config, "bounds"):
        new_bounds = dict(config.bounds)
        new_steps = dict(config.step)
        for key in list(new_bounds):
            if key in bounds:
                new_bounds[key] = bounds[key]
                new_steps[key] = steps[key]
        pricer.config = type(config)(
            opt_keys=list(config.opt_keys),
            step=new_steps,
            bounds=new_bounds,
            cost_weight=dict(config.cost_weight),
        )
    legacy = {
        "base_fare": ("step_base_fare", "base_bounds"),
        "per_minute": ("step_per_minute", "pmin_bounds"),
        "per_mile": ("step_per_mile", "pmile_bounds"),
        "booking_fee": ("step_booking_fee", "booking_bounds"),
        "airport_fee": ("step_airport_fee", "airport_bounds"),
    }
    for key, (step_attr, bound_attr) in legacy.items():
        if hasattr(pricer, step_attr):
            setattr(pricer, step_attr, steps[key])
        if hasattr(pricer, bound_attr):
            setattr(pricer, bound_attr, bounds[key])


# --------------------------------------------------------------------------
# Controller perturbation (opponent domain randomization / OOD evaluation)
# --------------------------------------------------------------------------
MULTIPLICATIVE_PARAMETERS: Tuple[str, ...] = (
    "k_share", "k_gap", "k_revenue", "k_supply", "response_threshold",
    "step_scale", "alpha_share", "alpha_revenue", "alpha_gap", "kp_gap",
    "ki_gap", "supply_incentive_step", "alpha", "exploration_prob",
    "share_band", "price_gap_threshold",
)
ADDITIVE_PARAMETERS: Dict[str, Tuple[float, float, float]] = {
    # name: (sd, lower, upper)
    "target_share": (0.05, 0.30, 0.70),
    "target_gap": (0.50, -2.00, 2.00),
    "target_fulfillment": (0.05, 0.60, 0.98),
    "target_acceptance": (0.05, 0.50, 0.95),
    "target_wait": (1.50, 3.00, 14.00),
}


@dataclass(frozen=True)
class ControllerPerturbation:
    scale: float = 0.25
    cooldown_jitter: int = 1

    def apply(self, pricer: Any, rng: np.random.Generator) -> Dict[str, float]:
        """Perturb numeric controller parameters in place; return the draw."""
        draw: Dict[str, float] = {}
        if self.scale <= 0.0:
            return draw
        for name in MULTIPLICATIVE_PARAMETERS:
            value = getattr(pricer, name, None)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                factor = float(np.exp(rng.normal(0.0, self.scale)))
                new_value = float(value) * factor
                if name.startswith("alpha") or name == "exploration_prob":
                    new_value = float(np.clip(new_value, 0.01, 0.95))
                setattr(pricer, name, new_value)
                draw[name] = factor
        for name, (sd, lo, hi) in ADDITIVE_PARAMETERS.items():
            value = getattr(pricer, name, None)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                delta = float(rng.normal(0.0, sd * self.scale / 0.25))
                setattr(pricer, name, float(np.clip(float(value) + delta, lo, hi)))
                draw[name] = delta
        cooldown = getattr(pricer, "cooldown_H", None)
        if isinstance(cooldown, int) and self.cooldown_jitter > 0:
            jitter = int(rng.integers(-self.cooldown_jitter, self.cooldown_jitter + 1))
            pricer.cooldown_H = int(max(0, cooldown + jitter))
            draw["cooldown_H"] = float(jitter)
        setattr(pricer, "_perturbation_draw", dict(draw))
        return draw


def perturbed_mode_name(mode: str, variant: int) -> str:
    return f"{mode}@perturbed{int(variant)}"


def split_perturbed_mode(mode: str) -> Tuple[str, Optional[int]]:
    text = str(mode)
    if "@perturbed" not in text:
        return text, None
    base, _, index = text.partition("@perturbed")
    return base, int(index)


# --------------------------------------------------------------------------
# Threshold ablation
# --------------------------------------------------------------------------
THRESHOLD_ABLATION_MODES: Tuple[str, ...] = (
    "as_loaded",
    "constant_mean",
    "shuffled",
    "rubric",
)


def apply_threshold_ablation(
    profiles: List[Dict[str, Any]],
    mode: str,
    *,
    seed: int,
    rubric: Optional[Any] = None,
) -> Dict[str, Any]:
    """Rewrite ``PriceThreshold`` in place for a downstream ablation arm.

    ``constant_mean`` keeps the population mean but removes all heterogeneity;
    ``shuffled`` keeps the marginal distribution but breaks the link to rider
    attributes; ``rubric`` recomputes the transparent rule (requires a callable
    ``rubric(profile) -> float``).  The original value is preserved under
    ``PriceThresholdOriginal`` for audit.
    """
    name = str(mode or "as_loaded")
    if name not in THRESHOLD_ABLATION_MODES:
        raise ValueError(f"threshold ablation must be one of {THRESHOLD_ABLATION_MODES}")
    values = np.array([float(p.get("PriceThreshold", np.nan)) for p in profiles], dtype=float)
    finite = values[np.isfinite(values)]
    summary = {"mode": name, "profiles": len(profiles)}
    if name == "as_loaded" or not profiles:
        summary["mean"] = float(np.mean(finite)) if finite.size else None
        return summary
    for profile in profiles:
        profile.setdefault("PriceThresholdOriginal", profile.get("PriceThreshold"))
    if name == "constant_mean":
        mean = float(np.mean(finite))
        for profile in profiles:
            profile["PriceThreshold"] = mean
            profile["PriceThresholdSource"] = "ablation_constant_mean"
    elif name == "shuffled":
        rng = np.random.default_rng(int(seed))
        permuted = rng.permutation(values)
        for profile, value in zip(profiles, permuted):
            profile["PriceThreshold"] = float(value)
            profile["PriceThresholdSource"] = "ablation_shuffled"
    elif name == "rubric":
        if rubric is None:
            raise ValueError("rubric ablation requires a rubric callable")
        for profile in profiles:
            profile["PriceThreshold"] = float(rubric(profile))
            profile["PriceThresholdSource"] = "ablation_rubric"
    after = np.array([float(p["PriceThreshold"]) for p in profiles])
    summary.update({"mean": float(np.mean(after)), "std": float(np.std(after))})
    return summary


# --------------------------------------------------------------------------
# Rider memory: platform reputations (forward-looking demand channel)
# --------------------------------------------------------------------------
FIRMS: Tuple[str, str] = ("Firm1", "Firm2")


@dataclass
class FirmReputationMemory:
    """Exponentially smoothed price and reliability reputations per platform.

    ``price_index`` is the platform's posted tariff level relative to the
    city anchor (1.0 = anchor); ``reliability`` is its recent dispatch
    fulfillment.  Both start at the neutral market state each episode, are
    updated once per market step from realized outcomes, and live inside the
    choice model so counterfactual snapshots and CRN validation copy them.
    """

    alpha: float = 0.10
    price_weight: float = 1.50
    reliability_weight: float = 1.00
    outside_weight: float = 1.00
    price_index: Dict[str, float] = field(default_factory=lambda: {f: 1.0 for f in FIRMS})
    reliability: Dict[str, float] = field(default_factory=lambda: {f: 1.0 for f in FIRMS})
    updates: int = 0

    @classmethod
    def from_config(cls, config: "MarketDynamicsConfig") -> "FirmReputationMemory":
        return cls(
            alpha=float(config.memory_alpha),
            price_weight=float(config.memory_price_weight),
            reliability_weight=float(config.memory_reliability_weight),
            outside_weight=float(config.memory_outside_weight),
        )

    def reset(self) -> None:
        self.price_index = {f: 1.0 for f in FIRMS}
        self.reliability = {f: 1.0 for f in FIRMS}
        self.updates = 0

    def update(self, firm: str, *, price_index: float, fulfillment: Optional[float]) -> None:
        a = float(self.alpha)
        self.price_index[firm] = (1.0 - a) * self.price_index[firm] + a * float(
            np.clip(price_index, 0.2, 3.0)
        )
        if fulfillment is not None and np.isfinite(fulfillment):
            self.reliability[firm] = (1.0 - a) * self.reliability[firm] + a * float(
                np.clip(fulfillment, 0.0, 1.0)
            )

    def choice_utility(self, price_sensitivity: float) -> float:
        """Firm 1 minus Firm 2 latent utility from remembered reputations."""
        price = self.price_weight * float(price_sensitivity) * (
            self.price_index["Firm2"] - self.price_index["Firm1"]
        )
        reliability = self.reliability_weight * (
            self.reliability["Firm1"] - self.reliability["Firm2"]
        )
        return float(price + reliability)

    def outside_option_factor(self) -> float:
        """Multiplier on no-ride *odds* from the remembered price level."""
        level = 0.5 * (self.price_index["Firm1"] + self.price_index["Firm2"]) - 1.0
        return float(np.clip(np.exp(self.outside_weight * level), 0.5, 2.0))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "price_index": dict(self.price_index),
            "reliability": dict(self.reliability),
            "updates": int(self.updates),
        }


@dataclass
class RiderProfileMemory:
    """Persistent, identity-specific rider experience.

    Profile traits (threshold, income and loyalty) remain immutable.  This
    store adds only an exponentially smoothed memory of the rider's own
    platform and outside-option experience.  It is keyed by a stable profile
    identity, so drawing the same rider again produces behavior consistent
    with that rider's history rather than only a market-wide reputation.
    """

    alpha: float = 0.10
    choice_weight: float = 1.00
    outside_weight: float = 1.00
    choice_state: Dict[str, float] = field(default_factory=dict)
    outside_state: Dict[str, float] = field(default_factory=dict)
    encounters: Dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: "MarketDynamicsConfig") -> "RiderProfileMemory":
        return cls(
            alpha=float(config.memory_alpha),
            choice_weight=float(config.memory_reliability_weight),
            outside_weight=float(config.memory_outside_weight),
        )

    @staticmethod
    def identity(profile: Mapping[str, Any]) -> str:
        for key in (
            "ThresholdSourceProfileSHA256",
            "ThresholdPromptIdentitySHA256",
            "ProfileID",
            "ProfileId",
            "RiderID",
            "RiderId",
        ):
            value = str(profile.get(key, "") or "").strip()
            if value:
                return f"{key}:{value}"
        stable = {
            str(key): value
            for key, value in profile.items()
            if not str(key).startswith("Threshold")
            and str(key) not in {"ColdstartRides", "PriceThresholdRationale"}
        }
        encoded = json.dumps(
            stable, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
        return "profile_sha256:" + hashlib.sha256(encoded).hexdigest()

    def reset(self) -> None:
        self.choice_state.clear()
        self.outside_state.clear()
        self.encounters.clear()

    def features(
        self, profiles: Iterable[Mapping[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        identities = [self.identity(profile) for profile in profiles]
        choice = np.asarray(
            [self.choice_state.get(identity, 0.0) for identity in identities],
            dtype=float,
        )
        outside = np.asarray(
            [self.outside_state.get(identity, 0.0) for identity in identities],
            dtype=float,
        )
        return self.choice_weight * choice, self.outside_weight * outside

    def update(
        self,
        profiles: Iterable[Mapping[str, Any]],
        *,
        choice_target: np.ndarray,
        outside_target: np.ndarray,
    ) -> None:
        rows = list(profiles)
        choice = np.asarray(choice_target, dtype=float).reshape(-1)
        outside = np.asarray(outside_target, dtype=float).reshape(-1)
        if choice.size != len(rows) or outside.size != len(rows):
            raise ValueError("rider-memory targets must match the profile count")
        if not np.all(np.isfinite(choice)) or not np.all(np.isfinite(outside)):
            raise ValueError("rider-memory targets must be finite")

        # Sampling is with replacement. Aggregate duplicate identities before
        # the EWMA so the result is invariant to their order within a block.
        grouped: Dict[str, List[Tuple[float, float]]] = {}
        for profile, target_choice, target_outside in zip(rows, choice, outside):
            grouped.setdefault(self.identity(profile), []).append(
                (
                    float(np.clip(target_choice, -1.0, 1.0)),
                    float(np.clip(target_outside, -1.0, 1.0)),
                )
            )
        a = float(np.clip(self.alpha, 0.0, 1.0))
        for identity, observations in grouped.items():
            choice_mean = float(np.mean([value[0] for value in observations]))
            outside_mean = float(np.mean([value[1] for value in observations]))
            self.choice_state[identity] = (
                (1.0 - a) * self.choice_state.get(identity, 0.0)
                + a * choice_mean
            )
            self.outside_state[identity] = (
                (1.0 - a) * self.outside_state.get(identity, 0.0)
                + a * outside_mean
            )
            self.encounters[identity] = self.encounters.get(identity, 0) + len(
                observations
            )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "identities": len(self.encounters),
            "encounters": int(sum(self.encounters.values())),
            "alpha": float(self.alpha),
        }
