"""Auditable v28 shock calibration from normalized trip banks only.

The estimators in this module are intentionally conservative.  They use only
fields present in ``trip_bank.csv`` and never reinterpret missing outcomes as
observations.  In particular, the module cannot identify driver supply,
elasticities, rival promotions, outages, or transit disruptions.

All uncertainty is clustered by calendar day.  Request-volume estimates are
sample-frequency ratios and therefore require a realized-trip bank whose
sampling preserves relative source frequency; that assumption is reported in
the artifact because it is not encoded in the CSV itself.
"""

from __future__ import annotations

import calendar
import csv
import datetime as dt
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from market_shocks import HOLIDAY_RULES, ShockConfig, holiday_dates


SCHEMA_VERSION = 1
WEATHER_CATEGORIES = ("rain", "snow")
CONTROL_FIELDS = {
    "enabled",
    "mode",
    "intensity_scale",
    "types",
    "public_features",
    "emergency_surge_cap",
    "conditional_sampling",
    "calibration_source",
}


@dataclass(frozen=True)
class TripObservation:
    date: dt.date
    hour: int
    day_of_week: int
    weather: str
    duration: Optional[float]
    airport: bool
    airport_origin: bool
    airport_destination: bool


@dataclass
class Aggregate:
    rows: int = 0
    airport: int = 0
    airport_origin: int = 0
    airport_destination: int = 0
    duration_sum: float = 0.0
    duration_n: int = 0
    weather_counts: Counter = field(default_factory=Counter)

    def add(self, observation: TripObservation) -> None:
        self.rows += 1
        self.airport += int(observation.airport)
        self.airport_origin += int(observation.airport_origin)
        self.airport_destination += int(observation.airport_destination)
        if observation.duration is not None:
            self.duration_sum += float(observation.duration)
            self.duration_n += 1
        self.weather_counts[observation.weather] += 1

    @property
    def weather(self) -> str:
        if not self.weather_counts:
            return "unknown"
        return str(self.weather_counts.most_common(1)[0][0])


def _finite_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _boolean(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "t"}


def _airport_endpoint(city: str, value: Any) -> bool:
    text = str(value or "").strip().lower()
    if any(token in text for token in (
        "airport", "jfk", "laguardia", "lga", "newark", "ewr",
        "o'hare", "ohare", "midway", "logan",
    )):
        return True
    try:
        location_id = int(float(text))
    except (TypeError, ValueError):
        return False
    city_key = str(city).strip().lower()
    if "new york" in city_key or city_key == "nyc":
        return location_id in {1, 132, 138}
    if "chicago" in city_key:
        return location_id in {56, 76}
    return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_trip_bank(path: Path) -> Tuple[List[TripObservation], Dict[str, Any]]:
    """Load the normalized CSV without pandas and return auditable metadata."""
    observations: List[TripObservation] = []
    cities: Counter = Counter()
    semantics: Counter = Counter()
    weather_sources: Counter = Counter()
    malformed = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"pickup_datetime", "hour", "day_of_week", "weather", "airport"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        for row in reader:
            try:
                date = dt.date.fromisoformat(str(row.get("pickup_datetime", ""))[:10])
                hour = int(float(row.get("hour", 12)))
                day_of_week = int(float(row.get("day_of_week", date.weekday())))
            except (TypeError, ValueError):
                malformed += 1
                continue
            city = str(row.get("city") or "unknown").strip() or "unknown"
            airport = _boolean(row.get("airport"))
            pickup_airport = _airport_endpoint(city, row.get("pickup_location"))
            dropoff_airport = _airport_endpoint(city, row.get("dropoff_location"))
            duration = _finite_float(row.get("duration_minutes"))
            if duration is not None and duration <= 0.0:
                duration = None
            observations.append(TripObservation(
                date=date,
                hour=hour % 24,
                day_of_week=day_of_week % 7,
                weather=str(row.get("weather") or "unknown").strip().lower() or "unknown",
                duration=duration,
                airport=airport,
                airport_origin=bool(airport and pickup_airport),
                airport_destination=bool(airport and dropoff_airport),
            ))
            cities[city] += 1
            semantics[str(row.get("ride_semantics") or "unknown").strip()] += 1
            weather_sources[str(row.get("weather_source") or "unknown").strip()] += 1
    if not observations:
        raise ValueError(f"{path} contains no usable trip observations")
    city = cities.most_common(1)[0][0]
    ride_semantics = semantics.most_common(1)[0][0]
    return observations, {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "rows": len(observations),
        "malformed_rows_skipped": malformed,
        "city": city,
        "ride_semantics": ride_semantics,
        "city_values": dict(cities),
        "ride_semantics_values": dict(semantics),
        "weather_source_values": dict(weather_sources),
    }


def _aggregate(
    observations: Sequence[TripObservation],
) -> Tuple[Dict[dt.date, Aggregate], Dict[Tuple[dt.date, int], Aggregate]]:
    days: DefaultDict[dt.date, Aggregate] = defaultdict(Aggregate)
    cells: DefaultDict[Tuple[dt.date, int], Aggregate] = defaultdict(Aggregate)
    for observation in observations:
        days[observation.date].add(observation)
        cells[(observation.date, observation.hour)].add(observation)
    return dict(days), dict(cells)


def _percentile(values: Sequence[float], probability: float) -> Optional[float]:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return None
    position = min(1.0, max(0.0, float(probability))) * (len(finite) - 1)
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return finite[lo]
    weight = position - lo
    return finite[lo] * (1.0 - weight) + finite[hi] * weight


def _kish_effective_n(weights: Iterable[float]) -> float:
    values = [max(0.0, float(value)) for value in weights]
    total = sum(values)
    square = sum(value * value for value in values)
    return (total * total / square) if square > 0.0 else 0.0


def _ratio_evidence(
    pairs: Sequence[Tuple[dt.date, float, float, float]],
    *,
    eligible: bool,
    ineligible_reason: str = "",
    min_clusters: int = 5,
    bootstrap_replicates: int = 1000,
    seed: int = 0,
    method: str,
    source_fields: Sequence[str],
) -> Dict[str, Any]:
    usable = [
        (date, float(numerator), float(denominator), float(weight))
        for date, numerator, denominator, weight in pairs
        if math.isfinite(float(numerator))
        and math.isfinite(float(denominator))
        and float(denominator) > 0.0
        and float(weight) > 0.0
    ]
    numerator = sum(item[1] for item in usable)
    denominator = sum(item[2] for item in usable)
    estimate = numerator / denominator if denominator > 0.0 else None
    bootstrap: List[float] = []
    if len(usable) >= 2 and int(bootstrap_replicates) > 0:
        rng = random.Random(int(seed))
        for _ in range(int(bootstrap_replicates)):
            sampled = [usable[rng.randrange(len(usable))] for _ in usable]
            sample_denominator = sum(item[2] for item in sampled)
            if sample_denominator > 0.0:
                bootstrap.append(sum(item[1] for item in sampled) / sample_denominator)
    ci = None
    if bootstrap:
        ci = [_percentile(bootstrap, 0.025), _percentile(bootstrap, 0.975)]
    effective_n = _kish_effective_n(item[3] for item in usable)
    reasons: List[str] = []
    if not eligible:
        reasons.append(ineligible_reason or "outcome is not identified by this trip bank")
    if len(usable) < int(min_clusters):
        reasons.append(f"fewer than {int(min_clusters)} independent calendar-day clusters")
    if effective_n < max(2.0, 0.6 * float(min_clusters)):
        reasons.append("effective calendar-day sample size is too small")
    if estimate is None or not math.isfinite(float(estimate)) or float(estimate) <= 0.0:
        reasons.append("ratio is not finite and positive")
    if ci and ci[0] is not None and ci[1] is not None and ci[0] > 0.0:
        if ci[1] / ci[0] > 4.0:
            reasons.append("95% interval spans more than a four-fold range")
    return {
        "estimate": estimate,
        "ci95_cluster_bootstrap": ci,
        "cluster_count": len(usable),
        "effective_n_calendar_days": effective_n,
        "raw_target_rows": int(round(sum(item[3] for item in usable))),
        "ready_for_training_override": not reasons,
        "readiness_reasons": reasons,
        "cluster_unit": "calendar_day",
        "bootstrap_replicates": int(bootstrap_replicates),
        "method": method,
        "source_fields": list(source_fields),
    }


def _mean_evidence(
    daily_values: Sequence[Tuple[dt.date, float, float]],
    *,
    bootstrap_replicates: int,
    seed: int,
    source_fields: Sequence[str],
    method: str,
) -> Dict[str, Any]:
    usable = [
        (date, float(total), float(weight))
        for date, total, weight in daily_values
        if math.isfinite(float(total)) and float(weight) > 0.0
    ]
    total_weight = sum(item[2] for item in usable)
    estimate = sum(item[1] for item in usable) / total_weight if total_weight else None
    bootstrap: List[float] = []
    if len(usable) >= 2:
        rng = random.Random(int(seed))
        for _ in range(int(bootstrap_replicates)):
            sampled = [usable[rng.randrange(len(usable))] for _ in usable]
            weight = sum(item[2] for item in sampled)
            if weight > 0.0:
                bootstrap.append(sum(item[1] for item in sampled) / weight)
    return {
        "estimate": estimate,
        "ci95_cluster_bootstrap": (
            [_percentile(bootstrap, 0.025), _percentile(bootstrap, 0.975)]
            if bootstrap else None
        ),
        "cluster_count": len(usable),
        "effective_n_calendar_days": _kish_effective_n(item[2] for item in usable),
        "raw_rows": int(round(total_weight)),
        "cluster_unit": "calendar_day",
        "bootstrap_replicates": int(bootstrap_replicates),
        "method": method,
        "source_fields": list(source_fields),
    }


def _pooled_control(cells: Sequence[Aggregate]) -> Dict[str, Optional[float]]:
    rows = sum(cell.rows for cell in cells)
    duration_n = sum(cell.duration_n for cell in cells)
    return {
        "count": statistics.fmean(cell.rows for cell in cells) if cells else None,
        "airport_rate": (sum(cell.airport for cell in cells) / rows) if rows else None,
        "airport_origin_rate": (
            sum(cell.airport_origin for cell in cells) / rows if rows else None
        ),
        "airport_destination_rate": (
            sum(cell.airport_destination for cell in cells) / rows if rows else None
        ),
        "duration_mean": (
            sum(cell.duration_sum for cell in cells) / duration_n if duration_n else None
        ),
    }


def _weather_evidence(
    cells: Mapping[Tuple[dt.date, int], Aggregate],
    *,
    ride_semantics: str,
    airport_identified: bool,
    duration_identified: bool,
    bootstrap_replicates: int,
    seed: int,
) -> Dict[str, Any]:
    years = sorted({date.year for date, _ in cells})
    holiday_years = range(min(years) - 1, max(years) + 1) if years else ()
    holiday_window_dates = set(_holiday_windows(holiday_years))
    exact_controls: DefaultDict[Tuple[int, int, int], List[Tuple[dt.date, Aggregate]]] = defaultdict(list)
    relaxed_controls: DefaultDict[Tuple[int, int], List[Tuple[dt.date, Aggregate]]] = defaultdict(list)
    for (date, hour), cell in cells.items():
        if cell.weather != "clear" or date in holiday_window_dates:
            continue
        exact_controls[(date.month, date.weekday(), hour)].append((date, cell))
        relaxed_controls[(date.month, hour)].append((date, cell))

    output: Dict[str, Any] = {}
    for weather_index, weather in enumerate(WEATHER_CATEGORIES):
        daily: DefaultDict[dt.date, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        exact_matches = relaxed_matches = unmatched = 0
        control_cell_ids = set()
        for (date, hour), cell in cells.items():
            if cell.weather != weather or date in holiday_window_dates:
                continue
            controls = [
                (control_date, control)
                for control_date, control in exact_controls.get(
                    (date.month, date.weekday(), hour), []
                )
                if control_date != date
            ]
            if len(controls) >= 2:
                exact_matches += 1
            else:
                controls = [
                    (control_date, control)
                    for control_date, control in relaxed_controls.get(
                        (date.month, hour), []
                    )
                    if control_date != date
                ]
                if controls:
                    relaxed_matches += 1
            if not controls:
                unmatched += 1
                continue
            for control_date, _ in controls:
                control_cell_ids.add((control_date, hour))
            expected = _pooled_control([control for _, control in controls])
            target = daily[date]
            target["rows"] += cell.rows
            target["volume_actual"] += cell.rows
            target["volume_expected"] += float(expected["count"] or 0.0)
            if expected["airport_rate"] is not None:
                target["airport_actual"] += cell.airport
                target["airport_expected"] += cell.rows * float(expected["airport_rate"])
                target["airport_rows"] += cell.rows
            if expected["duration_mean"] is not None and cell.duration_n:
                target["duration_actual"] += cell.duration_sum
                target["duration_expected"] += cell.duration_n * float(expected["duration_mean"])
                target["duration_rows"] += cell.duration_n

        def pairs(actual: str, expected: str, weight: str) -> List[Tuple[dt.date, float, float, float]]:
            return [
                (date, values[actual], values[expected], values[weight])
                for date, values in sorted(daily.items())
                if values[expected] > 0.0 and values[weight] > 0.0
            ]

        prefix = (
            f"Observed {weather} date-hour cells compared with clear cells matched "
            "on calendar month, weekday, and hour; month-hour fallback is reported."
        )
        metrics = {
            "request_volume_ratio": _ratio_evidence(
                pairs("volume_actual", "volume_expected", "rows"),
                eligible=ride_semantics == "realized_trip",
                ineligible_reason=(
                    "quote observations do not identify realized request volume"
                ),
                bootstrap_replicates=bootstrap_replicates,
                seed=seed + weather_index * 101 + 1,
                method=prefix,
                source_fields=("pickup_datetime", "hour", "weather", "ride_semantics"),
            ),
            "airport_share_ratio": _ratio_evidence(
                pairs("airport_actual", "airport_expected", "airport_rows"),
                eligible=airport_identified,
                ineligible_reason="airport status is absent or uninformative",
                bootstrap_replicates=bootstrap_replicates,
                seed=seed + weather_index * 101 + 2,
                method=prefix,
                source_fields=("airport", "pickup_location", "dropoff_location", "weather"),
            ),
            "duration_ratio": _ratio_evidence(
                pairs("duration_actual", "duration_expected", "duration_rows"),
                eligible=duration_identified,
                ineligible_reason="observed trip duration is absent",
                bootstrap_replicates=bootstrap_replicates,
                seed=seed + weather_index * 101 + 3,
                method=prefix,
                source_fields=("duration_minutes", "weather", "pickup_datetime", "hour"),
            ),
        }
        output[weather] = {
            "metrics": metrics,
            "matching": {
                "exact_target_cells": exact_matches,
                "relaxed_target_cells": relaxed_matches,
                "unmatched_target_cells": unmatched,
                "unique_control_cells": len(control_cell_ids),
            },
        }
    return output


def _holiday_windows(years: Iterable[int]) -> Dict[dt.date, Tuple[str, int, dt.date]]:
    windows: Dict[dt.date, Tuple[str, int, dt.date]] = {}
    rule_by_name = {name: (pre, post) for name, _, pre, post in HOLIDAY_RULES}
    for year in years:
        for name, anchor in holiday_dates(int(year)).items():
            pre, post = rule_by_name[name]
            for offset in range(-int(pre), int(post) + 1):
                windows[anchor + dt.timedelta(days=offset)] = (name, offset, anchor)
    return windows


def _holiday_evidence(
    days: Mapping[dt.date, Aggregate],
    *,
    ride_semantics: str,
    airport_identified: bool,
    airport_direction_identified: bool,
    duration_identified: bool,
    bootstrap_replicates: int,
    seed: int,
) -> Dict[str, Any]:
    years = sorted({date.year for date in days})
    # Include the previous year's Christmas/New-Year wave when the observed
    # range begins in January.
    windows = _holiday_windows(range(min(years) - 1, max(years) + 1))
    holiday_dates_set = set(windows)
    grouped: DefaultDict[str, DefaultDict[dt.date, Dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float))
    )
    event_anchors = set()
    unmatched_days = 0
    control_dates = set()
    for date, (name, offset, anchor) in sorted(windows.items()):
        target = days.get(date)
        if target is None:
            continue
        matched_dates = [
            candidate
            for delta in (7, -7, 14, -14, 21, -21, 28, -28)
            for candidate in (date + dt.timedelta(days=delta),)
            if candidate in days and candidate not in holiday_dates_set
            and days[candidate].weather == target.weather
        ]
        controls = [days[candidate] for candidate in matched_dates]
        if not controls:
            unmatched_days += 1
            continue
        control_dates.update(matched_dates)
        event_anchors.add((name, anchor))
        expected = _pooled_control(controls)
        groups = ["all_wave"]
        if offset != 0:
            groups.append("non_anchor_wave")
        if offset < 0:
            groups.append("pre_holiday")
        elif offset > 0:
            groups.append("post_holiday")
        else:
            groups.append("holiday_anchor")
        for group in groups:
            values = grouped[group][date]
            values["rows"] += target.rows
            values["volume_actual"] += target.rows
            values["volume_expected"] += float(expected["count"] or 0.0)
            if expected["airport_rate"] is not None:
                values["airport_actual"] += target.airport
                values["airport_expected"] += target.rows * float(expected["airport_rate"])
                values["airport_rows"] += target.rows
            if expected["airport_origin_rate"] is not None:
                values["origin_actual"] += target.airport_origin
                values["origin_expected"] += target.rows * float(expected["airport_origin_rate"])
                values["origin_rows"] += target.rows
            if expected["airport_destination_rate"] is not None:
                values["destination_actual"] += target.airport_destination
                values["destination_expected"] += target.rows * float(expected["airport_destination_rate"])
                values["destination_rows"] += target.rows
            if expected["duration_mean"] is not None and target.duration_n:
                values["duration_actual"] += target.duration_sum
                values["duration_expected"] += target.duration_n * float(expected["duration_mean"])
                values["duration_rows"] += target.duration_n

    output: Dict[str, Any] = {}
    group_order = ("all_wave", "non_anchor_wave", "pre_holiday", "post_holiday", "holiday_anchor")
    for group_index, group in enumerate(group_order):
        daily = grouped.get(group, {})

        def pairs(actual: str, expected: str, weight: str) -> List[Tuple[dt.date, float, float, float]]:
            return [
                (date, values[actual], values[expected], values[weight])
                for date, values in sorted(daily.items())
                if values[expected] > 0.0 and values[weight] > 0.0
            ]

        method = (
            "US holiday-wave dates from market_shocks.HOLIDAY_RULES; each observed "
            "event day is compared with available same-weekday dates at +/-7, 14, "
            "21, and 28 days with the same dominant weather, excluding all holiday windows."
        )
        output[group] = {
            "request_volume_ratio": _ratio_evidence(
                pairs("volume_actual", "volume_expected", "rows"),
                eligible=ride_semantics == "realized_trip",
                ineligible_reason="quote observations do not identify realized request volume",
                bootstrap_replicates=bootstrap_replicates,
                seed=seed + group_index * 17 + 1,
                method=method,
                source_fields=("pickup_datetime", "ride_semantics"),
            ),
            "airport_share_ratio": _ratio_evidence(
                pairs("airport_actual", "airport_expected", "airport_rows"),
                eligible=airport_identified,
                ineligible_reason="airport status is absent or uninformative",
                bootstrap_replicates=bootstrap_replicates,
                seed=seed + group_index * 17 + 2,
                method=method,
                source_fields=("airport", "pickup_datetime"),
            ),
            "airport_origin_share_ratio": _ratio_evidence(
                pairs("origin_actual", "origin_expected", "origin_rows"),
                eligible=airport_direction_identified,
                ineligible_reason=(
                    "airport endpoint direction is absent or insufficiently classified"
                ),
                bootstrap_replicates=bootstrap_replicates,
                seed=seed + group_index * 17 + 3,
                method=method,
                source_fields=("airport", "pickup_location", "pickup_datetime"),
            ),
            "airport_destination_share_ratio": _ratio_evidence(
                pairs("destination_actual", "destination_expected", "destination_rows"),
                eligible=airport_direction_identified,
                ineligible_reason=(
                    "airport endpoint direction is absent or insufficiently classified"
                ),
                bootstrap_replicates=bootstrap_replicates,
                seed=seed + group_index * 17 + 4,
                method=method,
                source_fields=("airport", "dropoff_location", "pickup_datetime"),
            ),
            "duration_ratio": _ratio_evidence(
                pairs("duration_actual", "duration_expected", "duration_rows"),
                eligible=duration_identified,
                ineligible_reason="observed trip duration is absent",
                bootstrap_replicates=bootstrap_replicates,
                seed=seed + group_index * 17 + 5,
                method=method,
                source_fields=("duration_minutes", "pickup_datetime"),
            ),
        }
    return {
        "groups": output,
        "observed_holiday_events": len(event_anchors),
        "matched_control_days": len(control_dates),
        "unmatched_event_days": unmatched_days,
    }


def _wilson(successes: int, trials: int, z: float = 1.959963984540054) -> Optional[List[float]]:
    if trials <= 0:
        return None
    p = float(successes) / float(trials)
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def _storm_hazard(days: Mapping[dt.date, Aggregate]) -> Dict[str, Any]:
    if not days:
        return {"monthly_onset_hazard": {}, "duration_days": {}}
    observed_dates = set(days)
    start, end = min(observed_dates), max(observed_dates)
    month_groups: DefaultDict[Tuple[int, int], List[dt.date]] = defaultdict(list)
    cursor = start
    while cursor <= end:
        month_groups[(cursor.year, cursor.month)].append(cursor)
        cursor += dt.timedelta(days=1)
    month_records: DefaultDict[int, List[Tuple[int, int]]] = defaultdict(list)
    # A single snow-labelled hour is weather exposure, not a city-wide storm
    # day.  Hazard/duration estimates therefore require snow in at least 25%
    # of that day's sampled rows and at least five rows of daily coverage.
    # Weather response ratios above still use every snow-labelled date-hour.
    snow_days = {
        date for date, aggregate in days.items()
        if aggregate.rows >= 5
        and aggregate.weather_counts.get("snow", 0) / aggregate.rows >= 0.25
    }
    complete_dates = set()
    for (_, month), expected_dates in sorted(month_groups.items()):
        observed = [date for date in expected_dates if date in observed_dates]
        coverage = len(observed) / max(1, len(expected_dates))
        if coverage < 0.90 or len(observed) < 20:
            continue
        complete_dates.update(observed)
        onsets = 0
        for date in observed:
            previous = date - dt.timedelta(days=1)
            if date in snow_days and previous not in snow_days:
                onsets += 1
        month_records[month].append((onsets, len(observed)))

    monthly: Dict[str, Any] = {}
    for month in sorted(month_records):
        onsets = sum(record[0] for record in month_records[month])
        trials = sum(record[1] for record in month_records[month])
        estimate = onsets / trials if trials else None
        interval = _wilson(onsets, trials)
        reasons = []
        if onsets < 5:
            reasons.append("fewer than five observed snow-event onsets")
        if trials < 40:
            reasons.append("fewer than 40 high-coverage calendar days")
        if interval and interval[0] > 0.0 and interval[1] / interval[0] > 4.0:
            reasons.append("95% interval spans more than a four-fold hazard range")
        monthly[str(month)] = {
            "estimate": estimate,
            "ci95_wilson": interval,
            "snow_event_onsets": onsets,
            "effective_n_calendar_days": trials,
            "ready_for_training_override": not reasons,
            "readiness_reasons": reasons,
            "method": (
                "Snow-event onsets divided by observed days, using only year-month "
                "panels with >=90% date coverage and >=20 observed days."
            ),
            "source_fields": ["pickup_datetime", "weather"],
        }

    runs: List[int] = []
    for date in sorted(snow_days & complete_dates):
        if date - dt.timedelta(days=1) in snow_days:
            continue
        length = 1
        while date + dt.timedelta(days=length) in snow_days:
            length += 1
        runs.append(length)
    duration_reasons = []
    if len(runs) < 5:
        duration_reasons.append("fewer than five observed snow events in high-coverage months")
    duration = {
        "event_count": len(runs),
        "observed_run_lengths": runs,
        "median": _percentile(runs, 0.50),
        "p10": _percentile(runs, 0.10),
        "p90": _percentile(runs, 0.90),
        "ready_for_training_override": not duration_reasons,
        "readiness_reasons": duration_reasons,
        "method": (
            "Contiguous high-coverage calendar days with snow in at least 25% "
            "of sampled rows and at least five daily rows."
        ),
        "source_fields": ["pickup_datetime", "weather"],
    }
    return {"monthly_onset_hazard": monthly, "duration_days": duration}


def _coverage(days: Mapping[dt.date, Aggregate]) -> Dict[str, Any]:
    dates = sorted(days)
    span = (dates[-1] - dates[0]).days + 1
    per_year = Counter(date.year for date in dates)
    return {
        "first_date": dates[0].isoformat(),
        "last_date": dates[-1].isoformat(),
        "observed_calendar_days": len(dates),
        "calendar_span_days": span,
        "date_coverage_fraction": len(dates) / span,
        "observed_days_by_year": {str(year): count for year, count in sorted(per_year.items())},
    }


def _prior_reason(field_name: str) -> str:
    if field_name.startswith("shortage_") or field_name in {
        "holiday_fleet", "storm_fleet", "storm_supply_recovery_days",
        "outage_rate_per_day", "outage_blocks", "outage_capacity",
    }:
        return "trip banks contain no online-driver, fleet, wage, or platform-capacity event labels"
    if field_name in {
        "holiday_airport_price_sensitivity", "storm_reservation_multiplier",
        "storm_price_sensitivity", "event_price_sensitivity",
    }:
        return "trip banks contain no rider choice counterfactuals identifying elasticity or reservation responses"
    if field_name.startswith("rival_promotion"):
        return "trip banks contain no labelled rival-promotion episodes or rival policy state"
    if field_name.startswith("transit_disruption"):
        return "trip banks contain no labelled transit service state"
    if field_name.startswith("event_"):
        return "trip banks contain no labelled large-event calendar or event footprint"
    if field_name in {"storm_forecast_lead_days", "storm_forecast_miss_prob", "storm_false_alarm_prob"}:
        return "trip banks contain realized weather but no forecast vintages"
    if field_name in {"storm_transit_suspension_prob", "storm_transit_degraded"}:
        return "trip banks contain no transit operations during storms"
    if field_name == "storm_discretionary_volume":
        return "trip purpose is unobserved; total snow-day volume cannot identify discretionary-trip volume"
    if field_name == "holiday_commute_volume":
        return "trip purpose is unobserved; total holiday volume cannot identify commute volume"
    if field_name == "storm_deferred_share" or field_name == "storm_rebound_days":
        return "individual deferred and recovered requests are not linked across days"
    if field_name in {"drift_persistence", "drift_sd"}:
        return "trip-bank sampling and secular source changes are not separable from latent demand drift"
    if field_name == "training_shock_opening_share":
        return "training curriculum exposure is a design choice, not a population parameter"
    return "no directly corresponding identifiable field exists in trip_bank.csv"


def _empirical_provenance(
    source: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return a self-contained lineage record for one empirical field."""
    return {
        "kind": "empirical_trip_bank_estimate",
        "path": str(source.get("path", "")),
        "sha256": str(source.get("sha256", "")),
        "ride_semantics": str(source.get("ride_semantics", "unknown")),
        "source_fields": list(evidence.get("source_fields", [])),
        "method": str(evidence.get("method", "")),
        "cluster_unit": str(evidence.get("cluster_unit", "calendar_day")),
    }


def _prior_provenance() -> Dict[str, Any]:
    return {
        "kind": "documented_prior",
        "source": "market_shocks.ShockConfig",
        "empirical_override": False,
    }


def _field_decisions(
    *,
    weather: Mapping[str, Any],
    holidays: Mapping[str, Any],
    hazard: Mapping[str, Any],
    airport_identified: bool,
    duration_identified: bool,
    city: str,
    source: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    priors = ShockConfig().as_dict()
    decisions: Dict[str, Any] = {}
    candidate: Dict[str, Any] = {}

    def consider(field_name: str, evidence: Mapping[str, Any], evidence_path: str) -> None:
        prior = priors[field_name]
        if bool(evidence.get("ready_for_training_override")):
            value = evidence.get("estimate")
            candidate[field_name] = value
            decisions[field_name] = {
                "status": "candidate_empirical_override_review_required",
                "value": value,
                "prior": prior,
                "evidence_path": evidence_path,
                "provenance": _empirical_provenance(source, evidence),
                "evidence_summary": {
                    "estimate": evidence.get("estimate"),
                    "ci95": evidence.get("ci95_cluster_bootstrap"),
                    "effective_n_calendar_days": evidence.get(
                        "effective_n_calendar_days"
                    ),
                    "cluster_count": evidence.get("cluster_count"),
                    "method": evidence.get("method"),
                    "source_fields": evidence.get("source_fields"),
                },
            }
        else:
            decisions[field_name] = {
                "status": "retained_prior",
                "value": prior,
                "evidence_path": evidence_path,
                "reason": "; ".join(evidence.get("readiness_reasons", []))
                or "empirical evidence did not pass readiness gates",
                "provenance": _prior_provenance(),
            }

    holiday_airport = (
        holidays.get("groups", {}).get("non_anchor_wave", {}).get("airport_share_ratio", {})
    )
    if airport_identified:
        consider(
            "holiday_airport_share_peak",
            holiday_airport,
            "holiday.groups.non_anchor_wave.airport_share_ratio",
        )
    else:
        decisions["holiday_airport_share_peak"] = {
            "status": "retained_prior",
            "value": priors["holiday_airport_share_peak"],
            "reason": (
                "Boston airport status is absent and must not be inferred"
                if "boston" in city.lower()
                else "airport status is absent or uninformative"
            ),
            "provenance": _prior_provenance(),
        }

    snow = weather.get("snow", {}).get("metrics", {})
    if airport_identified:
        consider(
            "storm_airport_share",
            snow.get("airport_share_ratio", {}),
            "weather.snow.metrics.airport_share_ratio",
        )
    else:
        decisions["storm_airport_share"] = {
            "status": "retained_prior",
            "value": priors["storm_airport_share"],
            "reason": (
                "Boston airport status is absent and must not be inferred"
                if "boston" in city.lower()
                else "airport status is absent or uninformative"
            ),
            "provenance": _prior_provenance(),
        }
    if duration_identified:
        consider(
            "storm_duration_multiplier",
            snow.get("duration_ratio", {}),
            "weather.snow.metrics.duration_ratio",
        )
    else:
        decisions["storm_duration_multiplier"] = {
            "status": "retained_prior",
            "value": priors["storm_duration_multiplier"],
            "reason": (
                "Boston duration is absent and must not be inferred"
                if "boston" in city.lower()
                else "observed duration is absent"
            ),
            "provenance": _prior_provenance(),
        }

    monthly = hazard.get("monthly_onset_hazard", {})
    resolved_hazard = dict(priors["storm_daily_hazard"])
    ready_months = []
    for month, record in monthly.items():
        if record.get("ready_for_training_override"):
            resolved_hazard[str(month)] = record["estimate"]
            ready_months.append(str(month))
    if ready_months:
        candidate["storm_daily_hazard"] = resolved_hazard
        decisions["storm_daily_hazard"] = {
            "status": "candidate_empirical_override_review_required",
            "value": resolved_hazard,
            "empirically_overridden_months": ready_months,
            "prior_retained_months": sorted(set(resolved_hazard) - set(ready_months)),
            "evidence_path": "storm_hazard.monthly_onset_hazard",
            "provenance": {
                **_empirical_provenance(
                    source,
                    {
                        "source_fields": ["pickup_datetime", "weather"],
                        "method": "High-coverage month snow-event onset hazard.",
                        "cluster_unit": "calendar_day",
                    },
                ),
                "prior_source_for_unready_months": "market_shocks.ShockConfig",
            },
            "evidence_summary": {
                month: monthly[month] for month in ready_months
            },
        }
    else:
        decisions["storm_daily_hazard"] = {
            "status": "retained_prior",
            "value": priors["storm_daily_hazard"],
            "reason": "no month passed onset-count and calendar-coverage gates",
            "provenance": _prior_provenance(),
        }

    duration_record = hazard.get("duration_days", {})
    if duration_record.get("ready_for_training_override"):
        lower = max(1, int(round(float(duration_record.get("p10") or 1.0))))
        upper = max(lower, int(round(float(duration_record.get("p90") or lower))))
        candidate["storm_duration_days"] = [lower, upper]
        decisions["storm_duration_days"] = {
            "status": "candidate_empirical_override_review_required",
            "value": [lower, upper],
            "prior": priors["storm_duration_days"],
            "evidence_path": "storm_hazard.duration_days",
            "provenance": _empirical_provenance(source, duration_record),
            "evidence_summary": dict(duration_record),
        }
    else:
        decisions["storm_duration_days"] = {
            "status": "retained_prior",
            "value": priors["storm_duration_days"],
            "reason": "; ".join(duration_record.get("readiness_reasons", [])),
            "provenance": _prior_provenance(),
        }

    for field_name, prior in priors.items():
        if field_name in CONTROL_FIELDS or field_name in decisions:
            continue
        decisions[field_name] = {
            "status": "retained_prior",
            "value": prior,
            "reason": _prior_reason(field_name),
            "provenance": _prior_provenance(),
        }
    return decisions, candidate


def analyze_trip_bank(
    path: Path,
    *,
    bootstrap_replicates: int = 1000,
    seed: int = 2801,
) -> Dict[str, Any]:
    observations, source = load_trip_bank(Path(path))
    days, cells = _aggregate(observations)
    airport_rows = sum(observation.airport for observation in observations)
    airport_direction_rows = sum(
        observation.airport_origin or observation.airport_destination
        for observation in observations
    )
    duration_rows = sum(observation.duration is not None for observation in observations)
    airport_identified = bool(airport_rows > 0)
    airport_direction_coverage = (
        airport_direction_rows / airport_rows if airport_rows else 0.0
    )
    airport_direction_identified = bool(
        airport_identified and airport_direction_coverage >= 0.90
    )
    duration_identified = bool(duration_rows / len(observations) >= 0.50)
    city = str(source["city"])
    semantics = str(source["ride_semantics"])

    baseline_airport = _mean_evidence(
        [
            (date, float(aggregate.airport), float(aggregate.rows))
            for date, aggregate in sorted(days.items())
        ],
        bootstrap_replicates=bootstrap_replicates,
        seed=seed + 1,
        source_fields=("airport", "pickup_datetime"),
        method="Trip-level airport share with uncertainty clustered by calendar day.",
    )
    baseline_airport["identified"] = airport_identified
    baseline_airport_origin = _mean_evidence(
        [
            (date, float(aggregate.airport_origin), float(aggregate.rows))
            for date, aggregate in sorted(days.items())
        ],
        bootstrap_replicates=bootstrap_replicates,
        seed=seed + 3,
        source_fields=("airport", "pickup_location", "pickup_datetime"),
        method=(
            "Trip-level airport-origin share with uncertainty clustered by "
            "calendar day; requires airport endpoint classification coverage."
        ),
    )
    baseline_airport_origin["identified"] = airport_direction_identified
    baseline_airport_destination = _mean_evidence(
        [
            (date, float(aggregate.airport_destination), float(aggregate.rows))
            for date, aggregate in sorted(days.items())
        ],
        bootstrap_replicates=bootstrap_replicates,
        seed=seed + 4,
        source_fields=("airport", "dropoff_location", "pickup_datetime"),
        method=(
            "Trip-level airport-destination share with uncertainty clustered by "
            "calendar day; requires airport endpoint classification coverage."
        ),
    )
    baseline_airport_destination["identified"] = airport_direction_identified
    baseline_duration = _mean_evidence(
        [
            (date, float(aggregate.duration_sum), float(aggregate.duration_n))
            for date, aggregate in sorted(days.items())
            if aggregate.duration_n
        ],
        bootstrap_replicates=bootstrap_replicates,
        seed=seed + 2,
        source_fields=("duration_minutes", "pickup_datetime"),
        method="Trip-level mean duration with uncertainty clustered by calendar day.",
    )
    baseline_duration["identified"] = duration_identified
    weather = _weather_evidence(
        cells,
        ride_semantics=semantics,
        airport_identified=airport_identified,
        duration_identified=duration_identified,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed + 100,
    )
    holidays = _holiday_evidence(
        days,
        ride_semantics=semantics,
        airport_identified=airport_identified,
        airport_direction_identified=airport_direction_identified,
        duration_identified=duration_identified,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed + 200,
    )
    hazard = _storm_hazard(days)
    decisions, candidate = _field_decisions(
        weather=weather,
        holidays=holidays,
        hazard=hazard,
        airport_identified=airport_identified,
        duration_identified=duration_identified,
        city=city,
        source=source,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "city": city,
        "source": source,
        "coverage": _coverage(days),
        "identifiability": {
            "realized_request_volume": semantics == "realized_trip",
            "airport": airport_identified,
            "airport_direction": airport_direction_identified,
            "airport_direction_classification_coverage": airport_direction_coverage,
            "duration": duration_identified,
            "driver_supply": False,
            "rider_elasticity": False,
            "rival_promotions": False,
            "trip_purpose": False,
            "sampling_frequency_assumption": (
                "Relative request-volume ratios are conditional on the trip-bank sampling "
                "preserving relative source frequency; the CSV itself does not encode that guarantee."
            ),
        },
        "baseline": {
            "airport_share": baseline_airport,
            "airport_origin_share": baseline_airport_origin,
            "airport_destination_share": baseline_airport_destination,
            "duration_minutes": baseline_duration,
            "weather_row_counts": dict(Counter(observation.weather for observation in observations)),
        },
        "weather": weather,
        "holiday": holidays,
        "storm_hazard": hazard,
        "field_decisions": decisions,
        "training_candidate": {
            "status": "review_required_not_applied",
            "shock_calibration": candidate,
            "safe_to_merge_automatically": False,
            "reason": (
                "Estimates describe the observed trip-bank sampling frame; review confidence "
                "intervals, source coverage, and the snow-regime versus severe-storm distinction."
            ),
        },
    }


def build_artifact(
    trip_banks: Sequence[Path],
    *,
    bootstrap_replicates: int = 1000,
    seed: int = 2801,
) -> Dict[str, Any]:
    cities: Dict[str, Any] = {}
    for index, path in enumerate(trip_banks):
        result = analyze_trip_bank(
            Path(path),
            bootstrap_replicates=bootstrap_replicates,
            seed=seed + index * 10007,
        )
        key = str(result["city"])
        cities[key] = result
    return {
        "schema": "ride-response-v28-trip-bank-shock-readiness",
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "data_boundary": "trip_bank.csv only",
        "automatic_calibration_mutation": False,
        "cities": cities,
        "global_non_identifiable_channels": [
            "driver shortages and fleet response",
            "rider price elasticity and reservation response",
            "rival promotions or rival policy switches",
            "platform outages",
            "transit disruptions",
            "large-event identity and footprint",
            "trip purpose (commute/discretionary/essential)",
        ],
    }


def write_artifact(artifact: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number:.3f}" if math.isfinite(number) else "n/a"


def render_markdown(artifact: Mapping[str, Any]) -> str:
    lines = [
        "# v28 trip-bank shock calibration readiness",
        "",
        "This report uses `trip_bank.csv` only. It does not modify any city calibration file.",
        "Every proposed override requires review; missing channels retain the documented `ShockConfig` prior.",
        "",
    ]
    for city, report in artifact.get("cities", {}).items():
        coverage = report["coverage"]
        ident = report["identifiability"]

        def interval_text(metric: Mapping[str, Any], key: str) -> str:
            ci = metric.get(key)
            if isinstance(ci, list) and len(ci) == 2:
                return f"{_fmt(metric.get('estimate'))} [{_fmt(ci[0])}, {_fmt(ci[1])}]"
            return _fmt(metric.get("estimate"))

        def baseline_metric_text(metric: Mapping[str, Any]) -> str:
            if not bool(metric.get("identified", False)):
                return "n/a (not identified)"
            return interval_text(metric, "ci95_cluster_bootstrap")

        lines.extend([
            f"## {city}",
            "",
            f"- Source: `{report['source']['path']}` (`{report['source']['sha256'][:12]}…`)",
            f"- Rows: {report['source']['rows']:,}; semantics: `{report['source']['ride_semantics']}`",
            f"- Dates: {coverage['first_date']} to {coverage['last_date']}; "
            f"{coverage['observed_calendar_days']}/{coverage['calendar_span_days']} days observed "
            f"({_fmt(coverage['date_coverage_fraction'])})",
            f"- Identified: request volume={ident['realized_request_volume']}, "
            f"airport={ident['airport']}, duration={ident['duration']}",
            f"- Airport direction classified={ident['airport_direction']} "
            f"(coverage among airport-labelled rows="
            f"{_fmt(ident['airport_direction_classification_coverage'])})",
        ])
        baseline = report["baseline"]
        if ident["airport"]:
            lines.append(
                "- Baseline airport trip share: "
                + interval_text(
                    baseline["airport_share"], "ci95_cluster_bootstrap"
                )
            )
        if ident["airport_direction"]:
            lines.append(
                "- Baseline direction shares (from / to airport): "
                + interval_text(
                    baseline["airport_origin_share"], "ci95_cluster_bootstrap"
                )
                + " / "
                + interval_text(
                    baseline["airport_destination_share"],
                    "ci95_cluster_bootstrap",
                )
            )
        if ident["duration"]:
            lines.append(
                "- Baseline trip duration, minutes: "
                + interval_text(
                    baseline["duration_minutes"], "ci95_cluster_bootstrap"
                )
            )
        lines.extend([
            "",
            "### Candidate training overrides (review required)",
            "",
            "| ShockConfig field | Candidate | Evidence |",
            "|---|---:|---|",
        ])
        candidate = report["training_candidate"]["shock_calibration"]
        if candidate:
            for field_name, value in sorted(candidate.items()):
                decision = report["field_decisions"][field_name]
                summary = decision.get("evidence_summary", {})
                if field_name == "storm_daily_hazard":
                    evidence_note = (
                        "monthly onset hazards; empirical months="
                        + ",".join(decision.get("empirically_overridden_months", []))
                    )
                elif field_name == "storm_duration_days":
                    evidence_note = (
                        f"events={summary.get('event_count', 'n/a')}; "
                        f"{decision.get('evidence_path', '')}"
                    )
                else:
                    ci = summary.get("ci95")
                    ci_text = (
                        f"[{_fmt(ci[0])}, {_fmt(ci[1])}]"
                        if isinstance(ci, list) and len(ci) == 2 else "n/a"
                    )
                    evidence_note = (
                        f"95% CI {ci_text}; effective N="
                        f"{_fmt(summary.get('effective_n_calendar_days'))}; "
                        f"{decision.get('evidence_path', '')}"
                    )
                lines.append(
                    f"| `{field_name}` | `{json.dumps(value, sort_keys=True)}` | "
                    f"{evidence_note} |"
                )
        else:
            lines.append("| _none_ | — | No field passed the readiness gates. |")
        lines.extend([
            "",
            "### Observable ratios",
            "",
            "| Regime | Volume (95% CI) | Airport share (95% CI) | Duration (95% CI) | Effective N days |",
            "|---|---:|---:|---:|---:|",
        ])
        for weather in WEATHER_CATEGORIES:
            metrics = report["weather"][weather]["metrics"]

            def metric_text(metric: Mapping[str, Any]) -> str:
                ci = metric.get("ci95_cluster_bootstrap")
                interval = (
                    f" [{_fmt(ci[0])}, {_fmt(ci[1])}]"
                    if isinstance(ci, list) and len(ci) == 2 else ""
                )
                suffix = "" if metric.get("ready_for_training_override") else " (not ready)"
                return f"{_fmt(metric.get('estimate'))}{interval}{suffix}"

            lines.append(
                f"| {weather} vs matched clear | "
                f"{metric_text(metrics['request_volume_ratio'])} | "
                f"{metric_text(metrics['airport_share_ratio'])} | "
                f"{metric_text(metrics['duration_ratio'])} | "
                f"{_fmt(metrics['request_volume_ratio']['effective_n_calendar_days'])} |"
            )
        holiday_metrics = report["holiday"]["groups"]["non_anchor_wave"]
        lines.append(
            f"| holiday non-anchor wave vs matched weekday | "
            f"{metric_text(holiday_metrics['request_volume_ratio'])} | "
            f"{metric_text(holiday_metrics['airport_share_ratio'])} | "
            f"{metric_text(holiday_metrics['duration_ratio'])} | "
            f"{_fmt(holiday_metrics['request_volume_ratio']['effective_n_calendar_days'])} |"
        )
        pre_holiday = report["holiday"]["groups"]["pre_holiday"]
        post_holiday = report["holiday"]["groups"]["post_holiday"]
        lines.extend([
            "",
            "### Airport-direction evidence",
            "",
            "| Estimand | Ratio/share (95% CI) | Effective N days |",
            "|---|---:|---:|",
            "| Baseline from-airport share | "
            f"{baseline_metric_text(baseline['airport_origin_share'])} | "
            f"{_fmt(baseline['airport_origin_share']['effective_n_calendar_days']) if ident['airport_direction'] else 'n/a'} |",
            "| Baseline to-airport share | "
            f"{baseline_metric_text(baseline['airport_destination_share'])} | "
            f"{_fmt(baseline['airport_destination_share']['effective_n_calendar_days']) if ident['airport_direction'] else 'n/a'} |",
            "| Pre-holiday to-airport share ratio | "
            f"{metric_text(pre_holiday['airport_destination_share_ratio'])} | "
            f"{_fmt(pre_holiday['airport_destination_share_ratio']['effective_n_calendar_days'])} |",
            "| Post-holiday from-airport share ratio | "
            f"{metric_text(post_holiday['airport_origin_share_ratio'])} | "
            f"{_fmt(post_holiday['airport_origin_share_ratio']['effective_n_calendar_days'])} |",
            "",
            "### Snow-event timing evidence",
            "",
            "| Month | Daily onset hazard (Wilson 95% CI) | Onsets | Effective N days | Ready |",
            "|---:|---:|---:|---:|---:|",
        ])
        monthly_hazard = report["storm_hazard"]["monthly_onset_hazard"]
        if monthly_hazard:
            for month, evidence in sorted(
                monthly_hazard.items(), key=lambda item: int(item[0])
            ):
                lines.append(
                    f"| {month} | {interval_text(evidence, 'ci95_wilson')} | "
                    f"{evidence['snow_event_onsets']} | "
                    f"{evidence['effective_n_calendar_days']} | "
                    f"{evidence['ready_for_training_override']} |"
                )
        else:
            lines.append("| — | n/a | 0 | 0 | False |")
        duration_evidence = report["storm_hazard"]["duration_days"]
        lines.extend([
            "",
            "Snow-event duration proxy: "
            f"events={duration_evidence['event_count']}, "
            f"p10/median/p90={_fmt(duration_evidence['p10'])}/"
            f"{_fmt(duration_evidence['median'])}/"
            f"{_fmt(duration_evidence['p90'])} days; "
            f"ready={duration_evidence['ready_for_training_override']}.",
        ])
        retained = [
            (name, decision)
            for name, decision in report["field_decisions"].items()
            if decision["status"] == "retained_prior"
        ]
        lines.extend([
            "",
            "### Retained priors / blocked channels",
            "",
        ])
        for name, decision in sorted(retained):
            lines.append(f"- `{name}`: {decision.get('reason', 'not identifiable from this trip bank')}.")
        lines.extend([
            "",
            "> Volume ratios are sample-frequency ratios. They identify demand only for realized-trip "
            "banks and only if trip-bank sampling preserves relative source frequency.",
            "",
        ])
    lines.extend([
        "## What cannot feed training from these files",
        "",
    ])
    for channel in artifact.get("global_non_identifiable_channels", []):
        lines.append(f"- {channel}")
    lines.extend([
        "",
        "No Boston airport or duration parameter is estimated: those columns are absent/uninformative in the Boston quote bank.",
        "",
    ])
    return "\n".join(lines)


def write_markdown(artifact: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(artifact), encoding="utf-8")
