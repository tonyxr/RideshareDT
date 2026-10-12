"""Streaming, city-aware preprocessing for public ride, weather, and ACS data.

The module deliberately keeps data semantics separate from field-name parsing.
NYC and Chicago contain realized trips, while the Boston Kaggle source contains
posted quotes.  A canonical record therefore carries both ``ride_semantics``
and ``price_semantics`` so downstream evaluation cannot silently treat a quote
as a completed trip or a tip-inclusive total as a posted tariff.

The implementation is streaming and has no pandas dependency.  Large CSV and
Parquet inputs can be sampled without materializing the full source in memory.
"""

from __future__ import annotations

import bisect
import csv
import hashlib
import io
import json
import math
import random
import re
import struct
import zipfile
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

try:
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover - exercised by a dependency error test
    pq = None


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CATALOG_PATH = PROJECT_ROOT / "config" / "city_datasets.json"


class DataValidationError(ValueError):
    """A source row cannot be mapped to the canonical ride contract."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = str(code)


@dataclass(frozen=True)
class CityDataSpec:
    key: str
    market_name: str
    aliases: Tuple[str, ...]
    timezone: str
    ride_semantics: str
    price_semantics: str
    kaggle_slug: str
    ride_root: str
    ride_glob: str
    weather_glob: str
    weather_station: str
    pums_person: str
    pums_household: str
    puma_geography: str
    puma_name_pattern: str
    state_fips: str
    place_fips: str
    sources: Mapping[str, str]
    # v26 generic onboarding.  ``field_map`` maps canonical ride fields to the
    # source's column names/units (see config/city_dataset_template.json), so a
    # new city needs a catalog entry, not code.  ``market`` holds city facts
    # used by the simulator (e.g. transit fare for the outside option).
    field_map: Mapping[str, Any] = None  # type: ignore[assignment]
    market: Mapping[str, Any] = None  # type: ignore[assignment]
    status: str = "available"


@dataclass(frozen=True)
class WeatherObservation:
    timestamp_utc: datetime
    weather: str
    temperature_c: Optional[float]
    precipitation_mm: Optional[float]
    wind_speed_mps: Optional[float]
    visibility_m: Optional[float]
    station: str


@dataclass(frozen=True)
class CanonicalRideRecord:
    city: str
    ride_semantics: str
    price_semantics: str
    source_id: str
    pickup_datetime: Optional[datetime]
    hour: int
    day_of_week: int
    distance_miles: float
    duration_minutes: Optional[float]
    actual_price: float
    actual_price_source: str
    distance_source: str
    duration_source: str
    provider: str
    service: str
    shared_ride: bool
    airport: bool
    pickup_location: str
    dropoff_location: str
    weather: str
    weather_source: str
    temperature_c: Optional[float]
    precipitation_mm: Optional[float]
    wind_speed_mps: Optional[float]
    visibility_m: Optional[float]
    surge_multiplier: Optional[float]
    driver_pay: Optional[float]
    platform_profit: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["pickup_datetime"] = (
            self.pickup_datetime.isoformat() if self.pickup_datetime else ""
        )
        return result


def load_city_catalog(path: Path | str = DEFAULT_CATALOG_PATH) -> Dict[str, CityDataSpec]:
    catalog_path = Path(path)
    with catalog_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if int(payload.get("schema_version", 0)) != 1:
        raise DataValidationError("catalog_schema", "unsupported city catalog schema")
    result: Dict[str, CityDataSpec] = {}
    for key, raw in payload.get("cities", {}).items():
        result[str(key)] = CityDataSpec(
            key=str(key),
            market_name=str(raw["market_name"]),
            aliases=tuple(str(item) for item in raw.get("aliases", ())),
            timezone=str(raw["timezone"]),
            ride_semantics=str(raw["ride_semantics"]),
            price_semantics=str(raw["price_semantics"]),
            kaggle_slug=str(raw["kaggle_slug"]),
            ride_root=str(raw["ride_root"]),
            ride_glob=str(raw["ride_glob"]),
            weather_glob=str(raw["weather_glob"]),
            weather_station=str(raw["weather_station"]),
            pums_person=str(raw["pums_person"]),
            pums_household=str(raw["pums_household"]),
            puma_geography=str(raw["puma_geography"]),
            puma_name_pattern=str(raw["puma_name_pattern"]),
            state_fips=str(raw["state_fips"]),
            place_fips=str(raw["place_fips"]),
            sources=dict(raw.get("sources", {})),
            field_map=dict(raw.get("field_map", {}) or {}),
            market=dict(raw.get("market", {}) or {}),
            status=str(raw.get("status", "available")),
        )
    if not result:
        raise DataValidationError("empty_catalog", f"no cities found in {catalog_path}")
    return result


def resolve_city_spec(city: str, catalog: Optional[Mapping[str, CityDataSpec]] = None) -> CityDataSpec:
    specs = dict(catalog or load_city_catalog())
    requested = " ".join(str(city or "").strip().lower().split())
    for key, spec in specs.items():
        candidates = {key.lower(), spec.market_name.lower(), *(a.lower() for a in spec.aliases)}
        if requested in candidates:
            return spec
    supported = ", ".join(sorted(spec.market_name for spec in specs.values()))
    raise DataValidationError("unsupported_city", f"unsupported city {city!r}; choose one of: {supported}")


def default_source_paths(city: str, data_root: Path | str) -> Dict[str, Path]:
    spec = resolve_city_spec(city)
    root = Path(data_root)
    demographics = root / "demographics" / "acs_pums"
    return {
        "rides": root / spec.ride_root,
        "weather": root / "weather" / "noaa",
        "pums_person": demographics / spec.pums_person,
        "pums_household": demographics / spec.pums_household,
        "puma_geography": demographics / spec.puma_geography,
    }


def _field_name(value: Any) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower())).strip("_")


class _LooseRow:
    def __init__(self, raw: Mapping[str, Any]):
        self.raw = raw
        self._values: Dict[str, Tuple[Any, str]] = {}
        for original, value in raw.items():
            self._values.setdefault(_field_name(original), (value, str(original)))

    def pick(self, *names: str) -> Tuple[Any, str]:
        for name in names:
            item = self._values.get(_field_name(name))
            if item is not None and not _is_missing(item[0]):
                return item
        return None, ""


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in {"", "nan", "none", "null", "na", "n/a"}


def _number(value: Any) -> Optional[float]:
    if _is_missing(value):
        return None
    if isinstance(value, (int, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    text = str(value).strip().replace(",", "").replace("$", "")
    try:
        result = float(text)
        return result if math.isfinite(result) else None
    except ValueError:
        matches = re.findall(r"-?\d+(?:\.\d+)?", text)
        if not matches:
            return None
        numbers = [float(item) for item in matches]
        if len(numbers) >= 2 and (" to " in text.lower() or "-" in text.lstrip("-")):
            return float(sum(numbers[:2]) / 2.0)
        return numbers[0]


def _integer(value: Any) -> Optional[int]:
    parsed = _number(value)
    return int(parsed) if parsed is not None else None


def _boolean(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if _is_missing(value):
        return bool(default)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "t"}:
        return True
    if text in {"0", "false", "no", "n", "f"}:
        return False
    return bool(default)


def _parse_datetime(value: Any, tz_name: str) -> Optional[datetime]:
    if value is None or _is_missing(value):
        return None
    local_tz = ZoneInfo(tz_name)
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime.combine(value, time.min)
    elif isinstance(value, (int, float)) or str(value).strip().replace(".", "", 1).isdigit():
        numeric = _number(value)
        if numeric is None:
            return None
        # Unix seconds and milliseconds are the only supported numeric forms.
        if abs(numeric) > 1e12:
            numeric /= 1000.0
        if abs(numeric) < 1e8:
            return None
        return datetime.fromtimestamp(numeric, tz=timezone.utc).astimezone(local_tz)
    else:
        text = str(value).strip().replace("Z", "+00:00")
        parsed = None
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            for fmt in (
                "%m/%d/%Y %I:%M:%S %p",
                "%m/%d/%Y %H:%M:%S",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d",
            ):
                try:
                    parsed = datetime.strptime(text, fmt)
                    break
                except ValueError:
                    continue
        if parsed is None:
            return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=local_tz)
    return parsed.astimezone(local_tz)


def _service_level(text: str) -> str:
    value = str(text or "").strip().lower()
    premium = ("black", "lux", "premium", "select", "suv", "uberxl", "lyft xl")
    return "premium" if any(token in value for token in premium) else "economy"


def _embedded_weather(row: _LooseRow) -> Tuple[str, Optional[float], Optional[float], Optional[float], Optional[float]]:
    summaries = [
        row.pick(name)[0]
        for name in ("weather", "short_summary", "long_summary", "icon", "summary")
    ]
    text = " ".join(str(value or "") for value in summaries).strip().lower()
    if any(token in text for token in ("snow", "sleet", "blizzard", "flurr")):
        weather = "snow"
    elif any(token in text for token in ("rain", "drizzle", "storm", "shower")):
        weather = "rain"
    else:
        weather = "clear"
    temp_f, _ = row.pick("temperature")
    precip_in, _ = row.pick("precip_intensity", "precipIntensity")
    wind_mph, _ = row.pick("wind_speed", "windSpeed")
    visibility_miles, _ = row.pick("visibility")
    temp = _number(temp_f)
    precip = _number(precip_in)
    wind = _number(wind_mph)
    visibility = _number(visibility_miles)
    return (
        weather,
        ((temp - 32.0) * 5.0 / 9.0) if temp is not None else None,
        (precip * 25.4) if precip is not None else None,
        (wind * 0.44704) if wind is not None else None,
        (visibility * 1609.344) if visibility is not None else None,
    )


def _airport_from_coordinates(lat: Optional[float], lon: Optional[float]) -> bool:
    if lat is None or lon is None:
        return False
    # O'Hare and Midway. A 3-mile radius accommodates centroid rounding in the
    # public Chicago source without labelling whole quadrants as airport trips.
    for airport_lat, airport_lon in ((41.9742, -87.9073), (41.7868, -87.7522)):
        lat_scale = 69.0 * (lat - airport_lat)
        lon_scale = 69.0 * math.cos(math.radians(airport_lat)) * (lon - airport_lon)
        if math.hypot(lat_scale, lon_scale) <= 3.0:
            return True
    return False


def _validate_required_numeric(value: Optional[float], code: str, label: str, upper: float) -> float:
    if value is None:
        raise DataValidationError(f"missing_{code}", f"missing {label}")
    if not math.isfinite(value) or value <= 0.0 or value > upper:
        raise DataValidationError(f"invalid_{code}", f"invalid {label}: {value!r}")
    return float(value)


_DISTANCE_UNITS = {"miles": 1.0, "mi": 1.0, "km": 0.621371, "kilometers": 0.621371, "meters": 0.000621371, "m": 0.000621371}
_DURATION_UNITS = {"minutes": 1.0, "min": 1.0, "seconds": 1.0 / 60.0, "s": 1.0 / 60.0, "hours": 60.0, "h": 60.0}


def _map_columns(spec_value: Any) -> Tuple[str, ...]:
    if spec_value is None:
        return ()
    if isinstance(spec_value, str):
        return (spec_value,)
    if isinstance(spec_value, Mapping):
        return tuple(str(item) for item in spec_value.get("columns", ()))
    return tuple(str(item) for item in spec_value)


def _normalize_with_field_map(spec: "CityDataSpec", row: "_LooseRow") -> Tuple[Any, ...]:
    """Map one row through a catalog field map (generic city onboarding).

    Field map keys (all optional except price, distance, and pickup_datetime):
      price            {"columns": [...]} first non-missing column, or
      price_components {"add": [...], "subtract": [...]} summed components
      distance         {"columns": [...], "unit": "miles|km|meters"}
      duration         {"columns": [...], "unit": "minutes|seconds|hours"}
      pickup_datetime  [...]
      provider, pickup_location, dropoff_location, service, surge_multiplier,
      driver_pay       [...]
      shared           {"columns": [...], "true_values": [...]} or pooled counts
      airport          {"location_ids": [...], "keywords": [...],
                        "coordinates": [[lat, lon, radius_miles], ...],
                        "lat_columns": [...], "lon_columns": [...]}
    """
    fm = spec.field_map
    price = None
    price_source = ""
    components = fm.get("price_components")
    if isinstance(components, Mapping):
        total = 0.0
        used: List[str] = []
        for name in components.get("add", ()):
            value, source = row.pick(name)
            number = _number(value)
            if number is None:
                total = float("nan")
                break
            total += number
            used.append(source)
        for name in components.get("subtract", ()):
            value, source = row.pick(name)
            total -= _number(value) or 0.0
            used.append(f"-{source}")
        if math.isfinite(total):
            price, price_source = total, "+".join(used)
    if price is None:
        value, price_source = row.pick(*_map_columns(fm.get("price")))
        price = _number(value)
    distance_spec = fm.get("distance", {})
    value, distance_source = row.pick(*_map_columns(distance_spec))
    distance = _number(value)
    if distance is not None and isinstance(distance_spec, Mapping):
        distance *= _DISTANCE_UNITS[str(distance_spec.get("unit", "miles")).lower()]
    duration_spec = fm.get("duration", {})
    value, duration_source = row.pick(*_map_columns(duration_spec))
    duration = _number(value)
    if duration is not None and isinstance(duration_spec, Mapping):
        duration *= _DURATION_UNITS[str(duration_spec.get("unit", "minutes")).lower()]
    dt_raw, _ = row.pick(*_map_columns(fm.get("pickup_datetime")))
    pickup_dt = _parse_datetime(dt_raw, spec.timezone)
    provider, _ = row.pick(*_map_columns(fm.get("provider")))
    pickup, _ = row.pick(*_map_columns(fm.get("pickup_location")))
    dropoff, _ = row.pick(*_map_columns(fm.get("dropoff_location")))
    airport_spec = fm.get("airport", {}) or {}
    airport = False
    ids = {str(item) for item in airport_spec.get("location_ids", ())}
    if ids and (str(_integer(pickup)) in ids or str(_integer(dropoff)) in ids):
        airport = True
    keywords = tuple(str(k).lower() for k in airport_spec.get("keywords", ()))
    if keywords and any(k in f"{pickup or ''} {dropoff or ''}".lower() for k in keywords):
        airport = True
    for lat_col, lon_col in zip(airport_spec.get("lat_columns", ()), airport_spec.get("lon_columns", ())):
        lat, _ = row.pick(lat_col)
        lon, _ = row.pick(lon_col)
        lat_f, lon_f = _number(lat), _number(lon)
        if lat_f is None or lon_f is None:
            continue
        for a_lat, a_lon, radius in airport_spec.get("coordinates", ()):
            dy = (lat_f - float(a_lat)) * 69.0
            dx = (lon_f - float(a_lon)) * 69.0 * math.cos(math.radians(float(a_lat)))
            if math.hypot(dx, dy) <= float(radius):
                airport = True
    service_text, _ = row.pick(*_map_columns(fm.get("service")))
    shared_spec = fm.get("shared", {}) or {}
    shared_value, _ = row.pick(*_map_columns(shared_spec))
    true_values = {str(v).lower() for v in (shared_spec.get("true_values", ("true", "1", "yes")) if isinstance(shared_spec, Mapping) else ("true", "1", "yes"))}
    shared = str(shared_value).strip().lower() in true_values if shared_value is not None else (
        "shared" in str(service_text or "").lower() or "pool" in str(service_text or "").lower()
    )
    surge_raw, _ = row.pick(*_map_columns(fm.get("surge_multiplier")))
    pay_raw, _ = row.pick(*_map_columns(fm.get("driver_pay")))
    return (
        price, price_source, distance, distance_source, duration, duration_source,
        pickup_dt, provider, pickup, dropoff, airport, service_text, shared,
        _number(surge_raw), pay_raw, None,
    )


def normalize_ride_record(
    city: str,
    raw: Mapping[str, Any],
    weather_index: Optional["NOAAWeatherIndex"] = None,
) -> CanonicalRideRecord:
    """Map one raw city row into the canonical model-operation contract."""

    spec = resolve_city_spec(city)
    row = _LooseRow(raw)
    source_id, _ = row.pick("source_id", "trip_id", "id", "uuid")
    explicit_city, _ = row.pick("city")
    if explicit_city and resolve_city_spec(str(explicit_city)).key != spec.key:
        raise DataValidationError(
            "city_mismatch",
            f"row city {explicit_city!r} does not match requested {spec.market_name}",
        )

    if spec.field_map:
        mapped = _normalize_with_field_map(spec, row)
        (price, price_source, distance, distance_source, duration, duration_source,
         pickup_dt, provider, pickup, dropoff, airport, service_text, shared,
         surge_multiplier, driver_pay, platform_profit) = mapped
    elif spec.key == "nyc":
        price_raw, price_source = row.pick("actual_price", "passenger_fare", "Passenger Fare")
        distance_raw, distance_source = row.pick("distance_miles", "trip_length", "Trip Length")
        duration_raw, duration_source = row.pick(
            "duration_minutes", "total_ride_time", "request_to_dropoff", "on_scene_to_dropoff"
        )
        price = _number(price_raw)
        distance = _number(distance_raw)
        duration = _number(duration_raw)
        if duration is not None and _field_name(duration_source) != "duration_minutes":
            duration /= 60.0
        dt_raw, _ = row.pick("pickup_datetime", "datetime", "date")
        pickup_dt = _parse_datetime(dt_raw, spec.timezone)
        explicit_hour, _ = row.pick("hour", "hour_of_day")
        if pickup_dt is not None and _integer(explicit_hour) is not None:
            pickup_dt = pickup_dt.replace(hour=int(_integer(explicit_hour) or 0))
        provider, _ = row.pick("provider", "business", "company")
        pickup, _ = row.pick("pickup_location", "pickup_zone", "source")
        dropoff, _ = row.pick("dropoff_location", "dropoff_zone", "destination")
        pickup_zone = _integer(pickup)
        dropoff_zone = _integer(dropoff)
        airport = bool({pickup_zone, dropoff_zone}.intersection({1, 132, 138}))
        driver_pay, _ = row.pick("driver_pay", "driver_total_pay")
        platform_profit, _ = row.pick("platform_profit", "rideshare_profit")
        service_text, _ = row.pick("service", "product_name", "name")
        shared = "shared" in str(service_text or "").lower()
        surge_multiplier = None
    elif spec.key == "boston":
        price_raw, price_source = row.pick("actual_price", "price")
        distance_raw, distance_source = row.pick("distance_miles", "distance")
        duration_raw, duration_source = row.pick("duration_minutes", "duration")
        price = _number(price_raw)
        distance = _number(distance_raw)
        duration = _number(duration_raw)
        dt_raw, _ = row.pick("pickup_datetime", "datetime", "timestamp")
        pickup_dt = _parse_datetime(dt_raw, spec.timezone)
        provider, _ = row.pick("provider", "cab_type", "company")
        pickup, _ = row.pick("pickup_location", "source")
        dropoff, _ = row.pick("dropoff_location", "destination")
        text = f"{pickup or ''} {dropoff or ''}".lower()
        airport = any(token in text for token in ("airport", "logan"))
        driver_pay = None
        platform_profit = None
        service_text, _ = row.pick("service", "name", "product_name")
        shared = any(token in str(service_text or "").lower() for token in ("shared", "pool"))
        surge_raw, _ = row.pick("surge_multiplier")
        surge_multiplier = _number(surge_raw)
    elif spec.key == "chicago":
        fare_raw, fare_source = row.pick("fare")
        charges_raw, charges_source = row.pick("additional_charges")
        total_raw, total_source = row.pick("trip_total", "actual_price")
        tip_raw, tip_source = row.pick("tip")
        fare = _number(fare_raw)
        charges = _number(charges_raw)
        total = _number(total_raw)
        tip = _number(tip_raw) or 0.0
        if fare is not None and charges is not None:
            price = fare + charges
            price_source = f"{fare_source}+{charges_source}"
        elif total is not None:
            price = total - tip
            price_source = f"{total_source}-{tip_source or 'Tip'}"
        else:
            price = None
            price_source = ""
        distance_raw, distance_source = row.pick("distance_miles", "trip_miles")
        duration_raw, duration_source = row.pick("duration_minutes", "trip_seconds")
        distance = _number(distance_raw)
        duration = _number(duration_raw)
        if duration is not None and _field_name(duration_source) == "trip_seconds":
            duration /= 60.0
        dt_raw, _ = row.pick("pickup_datetime", "trip_start_timestamp")
        pickup_dt = _parse_datetime(dt_raw, spec.timezone)
        provider, _ = row.pick("provider", "company")
        pickup, _ = row.pick("pickup_location", "pickup_community_area")
        dropoff, _ = row.pick("dropoff_location", "dropoff_community_area")
        pickup_area = _integer(pickup)
        dropoff_area = _integer(dropoff)
        pickup_lat, _ = row.pick("pickup_centroid_latitude")
        pickup_lon, _ = row.pick("pickup_centroid_longitude")
        dropoff_lat, _ = row.pick("dropoff_centroid_latitude")
        dropoff_lon, _ = row.pick("dropoff_centroid_longitude")
        airport = (
            bool({pickup_area, dropoff_area}.intersection({56, 76}))
            or _airport_from_coordinates(_number(pickup_lat), _number(pickup_lon))
            or _airport_from_coordinates(_number(dropoff_lat), _number(dropoff_lon))
        )
        pooled, _ = row.pick("trips_pooled")
        authorized, _ = row.pick("shared_trip_authorized")
        shared = (_integer(pooled) or 0) > 1 or _boolean(authorized)
        service_text = "shared" if shared else "economy"
        driver_pay = None
        platform_profit = None
        surge_multiplier = None
    else:  # pragma: no cover - catalog validation keeps this unreachable
        raise DataValidationError("unsupported_adapter", f"no adapter for {spec.key}")

    price_value = _validate_required_numeric(price, "price", "comparable price", 2500.0)
    distance_value = _validate_required_numeric(distance, "distance", "distance miles", 500.0)
    if duration is not None and (not math.isfinite(duration) or duration <= 0.0 or duration > 1440.0):
        raise DataValidationError("invalid_duration", f"invalid duration minutes: {duration!r}")

    explicit_hour, _ = row.pick("hour", "hour_of_day")
    explicit_day, _ = row.pick("day_of_week", "weekday")
    hour = _integer(explicit_hour)
    day_of_week = _integer(explicit_day)
    if pickup_dt is not None:
        hour = pickup_dt.hour if hour is None else hour
        day_of_week = pickup_dt.weekday() if day_of_week is None else day_of_week
    hour = 12 if hour is None else int(hour)
    day_of_week = 2 if day_of_week is None else int(day_of_week)
    if not 0 <= hour <= 23:
        raise DataValidationError("invalid_hour", f"invalid hour: {hour}")
    if not 0 <= day_of_week <= 6:
        raise DataValidationError("invalid_day_of_week", f"invalid day_of_week: {day_of_week}")

    explicit_airport, _ = row.pick("airport")
    if explicit_airport is not None:
        airport = _boolean(explicit_airport)
    explicit_service, _ = row.pick("service")
    service = _service_level(str(explicit_service or service_text or ""))

    embedded = _embedded_weather(row)
    weather, temperature_c, precipitation_mm, wind_speed_mps, visibility_m = embedded
    weather_source = "embedded" if any(value is not None for value in embedded[1:]) else "fallback_clear"
    if weather_index is not None and pickup_dt is not None:
        observation = weather_index.nearest(pickup_dt)
        if observation is not None:
            weather = observation.weather
            temperature_c = observation.temperature_c
            precipitation_mm = observation.precipitation_mm
            wind_speed_mps = observation.wind_speed_mps
            visibility_m = observation.visibility_m
            weather_source = "noaa_global_hourly"

    if source_id is None:
        stable_payload = json.dumps(dict(raw), default=str, sort_keys=True, separators=(",", ":"))
        source_id = hashlib.sha256(stable_payload.encode("utf-8")).hexdigest()[:24]

    return CanonicalRideRecord(
        city=spec.market_name,
        ride_semantics=spec.ride_semantics,
        price_semantics=spec.price_semantics,
        source_id=str(source_id),
        pickup_datetime=pickup_dt,
        hour=hour,
        day_of_week=day_of_week,
        distance_miles=distance_value,
        duration_minutes=float(duration) if duration is not None else None,
        actual_price=price_value,
        actual_price_source=str(price_source),
        distance_source=str(distance_source),
        duration_source=str(duration_source),
        provider=str(provider or ("TNP" if spec.key == "chicago" else "")),
        service=service,
        shared_ride=bool(shared),
        airport=bool(airport),
        pickup_location=str(pickup or ""),
        dropoff_location=str(dropoff or ""),
        weather=str(weather),
        weather_source=weather_source,
        temperature_c=temperature_c,
        precipitation_mm=precipitation_mm,
        wind_speed_mps=wind_speed_mps,
        visibility_m=visibility_m,
        surge_multiplier=surge_multiplier,
        driver_pay=_number(driver_pay),
        platform_profit=_number(platform_profit),
    )


def _scaled_noaa(value: Any, scale: float, missing: int = 9999) -> Optional[float]:
    if _is_missing(value):
        return None
    token = str(value).split(",", 1)[0].strip()
    try:
        raw = int(token)
    except ValueError:
        return None
    if abs(raw) >= missing:
        return None
    return float(raw) / float(scale)


def _noaa_precipitation(row: Mapping[str, Any]) -> Optional[float]:
    depths: List[float] = []
    for key in ("AA1", "AA2", "AA3", "AA4"):
        value = row.get(key)
        if _is_missing(value):
            continue
        parts = str(value).split(",")
        if len(parts) < 2:
            continue
        try:
            depth = int(parts[1])
        except ValueError:
            continue
        if 0 <= depth < 9999:
            depths.append(float(depth) / 10.0)
    return max(depths) if depths else None


def _noaa_weather(row: Mapping[str, Any], temperature_c: Optional[float], precip_mm: Optional[float]) -> str:
    text = " ".join(str(row.get(key, "")) for key in ("REM", "AW1", "MW1", "MW2")).upper()
    if re.search(r"(^|[ +])-?(SN|SG|PL|GR|GS)([ +]|$)", text) or any(
        code in text for code in (",7", ",85", ",86")
    ):
        return "snow"
    if re.search(r"(^|[ +])-?(RA|DZ|SHRA|TSRA|FZRA)([ +]|$)", text):
        return "snow" if temperature_c is not None and temperature_c <= 0.5 else "rain"
    if precip_mm is not None and precip_mm > 0.0:
        return "snow" if temperature_c is not None and temperature_c <= 0.5 else "rain"
    return "clear"


class NOAAWeatherIndex:
    """Nearest-observation index for NOAA Global Hourly CSV files."""

    def __init__(self, observations: Sequence[WeatherObservation], max_gap_minutes: float = 120.0):
        ordered = sorted(observations, key=lambda item: item.timestamp_utc)
        self.observations = tuple(ordered)
        self._timestamps = tuple(item.timestamp_utc.timestamp() for item in ordered)
        self.max_gap_seconds = float(max_gap_minutes) * 60.0

    @classmethod
    def from_directory(
        cls,
        city: str,
        weather_root: Path | str,
        max_gap_minutes: float = 120.0,
    ) -> "NOAAWeatherIndex":
        spec = resolve_city_spec(city)
        files = sorted(Path(weather_root).glob(spec.weather_glob))
        if not files:
            raise FileNotFoundError(
                f"no NOAA files match {spec.weather_glob!r} under {Path(weather_root)}"
            )
        observations: List[WeatherObservation] = []
        for path in files:
            with path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    raw_date = row.get("DATE")
                    if _is_missing(raw_date):
                        continue
                    try:
                        stamp = datetime.fromisoformat(str(raw_date).replace("Z", "+00:00"))
                    except ValueError:
                        continue
                    if stamp.tzinfo is None:
                        stamp = stamp.replace(tzinfo=timezone.utc)
                    else:
                        stamp = stamp.astimezone(timezone.utc)
                    temp = _scaled_noaa(row.get("TMP"), 10.0)
                    precip = _noaa_precipitation(row)
                    wind = None
                    wind_parts = str(row.get("WND", "")).split(",")
                    if len(wind_parts) >= 4:
                        wind = _scaled_noaa(wind_parts[3], 10.0)
                    visibility = _scaled_noaa(row.get("VIS"), 1.0, missing=999999)
                    observations.append(
                        WeatherObservation(
                            timestamp_utc=stamp,
                            weather=_noaa_weather(row, temp, precip),
                            temperature_c=temp,
                            precipitation_mm=precip,
                            wind_speed_mps=wind,
                            visibility_m=visibility,
                            station=str(row.get("STATION", spec.weather_station)),
                        )
                    )
        if not observations:
            raise DataValidationError("empty_weather", f"no valid observations found for {spec.market_name}")
        return cls(observations, max_gap_minutes=max_gap_minutes)

    def nearest(self, local_datetime: datetime) -> Optional[WeatherObservation]:
        if not self.observations:
            return None
        stamp = local_datetime
        if stamp.tzinfo is None:
            raise DataValidationError("naive_datetime", "weather lookup requires a timezone-aware datetime")
        target = stamp.astimezone(timezone.utc).timestamp()
        index = bisect.bisect_left(self._timestamps, target)
        candidates = []
        if index < len(self.observations):
            candidates.append(self.observations[index])
        if index > 0:
            candidates.append(self.observations[index - 1])
        if not candidates:
            return None
        best = min(candidates, key=lambda item: abs(item.timestamp_utc.timestamp() - target))
        gap = abs(best.timestamp_utc.timestamp() - target)
        return best if gap <= self.max_gap_seconds else None

    def category_probabilities(self) -> Dict[str, float]:
        # One record per station-hour prevents irregular sub-hourly reporting
        # from overweighting severe periods.
        hourly: Dict[Tuple[str, int], str] = {}
        for observation in self.observations:
            hour = int(observation.timestamp_utc.timestamp() // 3600)
            hourly[(observation.station, hour)] = observation.weather
        counts = {"clear": 0, "rain": 0, "snow": 0}
        for value in hourly.values():
            counts[value if value in counts else "clear"] += 1
        total = float(sum(counts.values()))
        return {key: value / total for key, value in counts.items()} if total else {}


def discover_ride_files(city: str, ride_root: Path | str, pattern: Optional[str] = None) -> List[Path]:
    spec = resolve_city_spec(city)
    root = Path(ride_root)
    chosen = pattern or spec.ride_glob
    files = sorted(path for path in root.glob(chosen) if path.is_file())
    if not files and "**" not in chosen:
        files = sorted(path for path in root.glob(f"**/{chosen}") if path.is_file())
    return files


_SYSTEMATIC_CSV = {"enabled": False}


def systematic_csv_sampling(enabled: Optional[bool] = None) -> bool:
    """Get/set whether bounded CSV reads use a whole-file systematic sample.

    The legacy reader took the first N rows after a small offset, so a source
    ordered by date (Chicago 2018-2022) produced a calendar-biased sample.  The
    v26 preprocessing enables a deterministic systematic sample over the
    complete file (two passes: count rows, then keep evenly spaced rows).
    """
    if enabled is not None:
        _SYSTEMATIC_CSV["enabled"] = bool(enabled)
    return bool(_SYSTEMATIC_CSV["enabled"])


def _systematic_csv_sample(source: Path, rows: int, seed: int) -> Iterator[Mapping[str, Any]]:
    with source.open("rb") as handle:
        total = max(0, sum(chunk.count(b"\n") for chunk in iter(lambda: handle.read(1 << 24), b"")) - 1)
    # Invalid source rows are discarded during normalization.  A small sample
    # can therefore lose more than the old ten-percent reserve (Boston's
    # deterministic 250-row validation accepted only 249).  Use both a
    # proportional reserve and a fixed floor so validation and small smoke
    # samples have enough candidates without scanning the entire source.
    target = int(min(total, max(rows + 1024, math.ceil(rows * 1.25))))
    if target <= 0:
        return
    spacing = float(total) / float(target)
    start = (seed % 1_000_003) / 1_000_003.0 * spacing
    wanted = {int(start + i * spacing) for i in range(target)}
    sampled: List[Mapping[str, Any]] = []
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            if index in wanted:
                sampled.append(row)
    random.Random(int(seed)).shuffle(sampled)
    yield from sampled


def iter_tabular_rows(
    path: Path | str,
    sample_seed: Optional[int] = None,
    max_rows_hint: Optional[int] = None,
) -> Iterator[Mapping[str, Any]]:
    source = Path(path)
    lower = source.name.lower()
    if lower.endswith(".csv"):
        if max_rows_hint is not None and int(max_rows_hint) > 0 and systematic_csv_sampling():
            yield from _systematic_csv_sample(source, int(max_rows_hint), int(sample_seed or 0))
            return
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            offset = int(sample_seed or 0) % 100003
            for index, row in enumerate(reader):
                if index < offset:
                    continue
                yield row
        return
    if lower.endswith(".csv.gz"):
        import gzip

        with gzip.open(source, "rt", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            offset = int(sample_seed or 0) % 100003
            for index, row in enumerate(reader):
                if index >= offset:
                    yield row
        return
    if lower.endswith(".zip"):
        with zipfile.ZipFile(source) as archive:
            for member in sorted(archive.namelist()):
                if not member.lower().endswith(".csv"):
                    continue
                with archive.open(member) as raw_handle:
                    handle = io.TextIOWrapper(raw_handle, encoding="utf-8-sig", newline="")
                    for row in csv.DictReader(handle):
                        yield row
        return
    if lower.endswith(".parquet"):
        if pq is None:
            raise RuntimeError("Parquet support requires pyarrow")
        parquet_file = pq.ParquetFile(source)
        if max_rows_hint is not None and int(max_rows_hint) > 0:
            # Large ride Parquet files are commonly ordered by provider, date,
            # and hour.  Stopping after the first N rows therefore creates a
            # severely biased calibration sample.  Draw a deterministic
            # systematic sample over the complete physical row range instead.
            # A reserve tolerates invalid source rows; shuffling
            # the small in-memory sample ensures downstream early stopping does
            # not discard the end of the source period.
            total_rows = int(parquet_file.metadata.num_rows)
            requested_rows = int(max_rows_hint)
            target_rows = int(min(
                total_rows,
                max(requested_rows + 1024, math.ceil(requested_rows * 1.25)),
            ))
            if target_rows <= 0:
                return
            spacing = float(total_rows) / float(target_rows)
            start = (int(sample_seed or 0) % 1_000_003) / 1_000_003.0 * spacing
            next_position = float(start)
            sampled_rows: List[Mapping[str, Any]] = []
            global_start = 0
            for batch in parquet_file.iter_batches(batch_size=65_536):
                global_end = global_start + len(batch)
                indices: List[int] = []
                while (
                    len(sampled_rows) + len(indices) < target_rows
                    and next_position < global_end
                ):
                    if next_position >= global_start:
                        indices.append(int(next_position) - global_start)
                    next_position += spacing
                if indices:
                    sampled_rows.extend(batch.take(indices).to_pylist())
                global_start = global_end
                if len(sampled_rows) >= target_rows:
                    break
            random.Random(int(sample_seed or 0)).shuffle(sampled_rows)
            yield from sampled_rows
            return
        groups = list(range(parquet_file.num_row_groups))
        if sample_seed is not None and groups:
            offset = int(sample_seed) % len(groups)
            groups = groups[offset:] + groups[:offset]
        for group in groups:
            for batch in parquet_file.iter_batches(batch_size=4096, row_groups=[group]):
                rows = batch.to_pylist()
                if sample_seed is not None and len(rows) > 1:
                    # Stable rotation changes bounded samples without requiring
                    # NumPy or loading an entire row group.
                    offset = int(sample_seed) % len(rows)
                    rows = rows[offset:] + rows[:offset]
                yield from rows
        return
    raise DataValidationError("unsupported_format", f"unsupported tabular file: {source}")


def iter_normalized_rides(
    city: str,
    ride_root: Path | str,
    weather_root: Optional[Path | str] = None,
    max_rows: Optional[int] = None,
    sample_seed: Optional[int] = None,
    rejection_counts: Optional[Dict[str, int]] = None,
) -> Iterator[CanonicalRideRecord]:
    spec = resolve_city_spec(city)
    files = discover_ride_files(spec.key, ride_root)
    if not files:
        raise FileNotFoundError(f"no ride files found under {Path(ride_root)} for {spec.market_name}")
    weather = (
        NOAAWeatherIndex.from_directory(spec.key, weather_root)
        if weather_root is not None
        else None
    )
    accepted = 0
    for file_index, path in enumerate(files):
        seed = None if sample_seed is None else int(sample_seed) + file_index * 1000003
        for raw in iter_tabular_rows(
            path,
            sample_seed=seed,
            max_rows_hint=(
                None
                if max_rows is None
                else max(1, int(max_rows) - accepted)
            ),
        ):
            try:
                record = normalize_ride_record(spec.key, raw, weather_index=weather)
            except DataValidationError as exc:
                if rejection_counts is not None:
                    rejection_counts[exc.code] = rejection_counts.get(exc.code, 0) + 1
                continue
            yield record
            accepted += 1
            if max_rows is not None and accepted >= int(max_rows):
                return


def _read_dbf_records(payload: bytes) -> Iterator[Dict[str, str]]:
    if len(payload) < 33:
        raise DataValidationError("invalid_dbf", "DBF payload is truncated")
    record_count = struct.unpack("<I", payload[4:8])[0]
    header_length = struct.unpack("<H", payload[8:10])[0]
    record_length = struct.unpack("<H", payload[10:12])[0]
    fields: List[Tuple[str, int]] = []
    cursor = 32
    while cursor + 32 <= header_length and payload[cursor] != 0x0D:
        descriptor = payload[cursor : cursor + 32]
        name = descriptor[:11].split(b"\x00", 1)[0].decode("ascii", errors="ignore")
        length = int(descriptor[16])
        fields.append((name, length))
        cursor += 32
    for record_index in range(record_count):
        start = header_length + record_index * record_length
        record = payload[start : start + record_length]
        if len(record) < record_length or record[:1] == b"*":
            continue
        values: Dict[str, str] = {}
        offset = 1
        for name, length in fields:
            values[name] = record[offset : offset + length].decode("latin-1", errors="ignore").strip()
            offset += length
        yield values


def puma_codes_for_city(city: str, geography_zip: Path | str) -> Tuple[str, ...]:
    spec = resolve_city_spec(city)
    path = Path(geography_zip)
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".dbf")]
        if len(members) != 1:
            raise DataValidationError("puma_dbf", f"expected one DBF in {path}, found {len(members)}")
        records = list(_read_dbf_records(archive.read(members[0])))
    pattern = re.compile(spec.puma_name_pattern, flags=re.IGNORECASE)
    codes = []
    for record in records:
        name = record.get("NAMELSAD20") or record.get("NAME20") or ""
        code = record.get("PUMACE20") or record.get("PUMA20") or ""
        if pattern.search(name) and code:
            codes.append(str(code).zfill(5))
    if not codes:
        raise DataValidationError(
            "empty_city_pumas",
            f"pattern {spec.puma_name_pattern!r} matched no PUMAs in {path}",
        )
    return tuple(sorted(set(codes)))


class _WeightedMoments:
    def __init__(self) -> None:
        self.weight = 0.0
        self.total = 0.0
        self.total_square = 0.0

    def add(self, value: Optional[float], weight: Optional[float]) -> None:
        if value is None or weight is None or weight <= 0.0 or not math.isfinite(value):
            return
        self.weight += weight
        self.total += weight * value
        self.total_square += weight * value * value

    @property
    def mean(self) -> Optional[float]:
        return self.total / self.weight if self.weight else None

    @property
    def std(self) -> Optional[float]:
        if not self.weight:
            return None
        mean = self.total / self.weight
        return math.sqrt(max(0.0, self.total_square / self.weight - mean * mean))


def _pums_csv_rows(path: Path | str, prefix: str) -> Iterator[Mapping[str, str]]:
    with zipfile.ZipFile(path) as archive:
        members = [
            name for name in archive.namelist()
            if Path(name).name.lower().startswith(prefix.lower()) and name.lower().endswith(".csv")
        ]
        if len(members) != 1:
            raise DataValidationError("pums_member", f"expected one {prefix} CSV in {path}, found {len(members)}")
        with archive.open(members[0]) as raw:
            handle = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
            yield from csv.DictReader(handle)


def _normalized_distribution(counts: Mapping[str, float], keys: Sequence[str]) -> Dict[str, float]:
    total = float(sum(max(0.0, float(counts.get(key, 0.0))) for key in keys))
    if total <= 0.0:
        return {key: 1.0 / len(keys) for key in keys}
    return {key: max(0.0, float(counts.get(key, 0.0))) / total for key in keys}


def derive_demographic_calibration(
    city: str,
    person_zip: Path | str,
    household_zip: Path | str,
    geography_zip: Path | str,
) -> Dict[str, Any]:
    """Derive weighted adult/household priors from city-labelled 2020 PUMAs."""

    spec = resolve_city_spec(city)
    pumas = set(puma_codes_for_city(spec.key, geography_zip))
    age = _WeightedMoments()
    gender = {"Male": 0.0, "Female": 0.0}
    marital = {"Single": 0.0, "Married": 0.0, "Divorced": 0.0, "Widowed": 0.0}
    person_rows = 0
    for row in _pums_csv_rows(person_zip, "psam_p"):
        if str(row.get("PUMA", "")).zfill(5) not in pumas:
            continue
        age_value = _number(row.get("AGEP"))
        if age_value is None or age_value < 18.0:
            continue
        weight = _number(row.get("PWGTP"))
        if weight is None or weight <= 0.0:
            continue
        person_rows += 1
        age.add(age_value, weight)
        sex = _integer(row.get("SEX"))
        if sex == 1:
            gender["Male"] += weight
        elif sex == 2:
            gender["Female"] += weight
        mar = _integer(row.get("MAR"))
        if mar == 1:
            marital["Married"] += weight
        elif mar == 2:
            marital["Widowed"] += weight
        elif mar in {3, 4}:
            marital["Divorced"] += weight
        elif mar == 5:
            marital["Single"] += weight

    household_size = _WeightedMoments()
    income = {"<50k": 0.0, "50k-100k": 0.0, "100k-200k": 0.0, "200k+": 0.0}
    household_rows = 0
    for row in _pums_csv_rows(household_zip, "psam_h"):
        if str(row.get("PUMA", "")).zfill(5) not in pumas:
            continue
        weight = _number(row.get("WGTP"))
        size = _number(row.get("NP"))
        income_value = _number(row.get("HINCP"))
        if weight is None or weight <= 0.0 or size is None or size <= 0.0:
            continue
        household_rows += 1
        household_size.add(size, weight)
        if income_value is not None:
            if income_value < 50000.0:
                income["<50k"] += weight
            elif income_value < 100000.0:
                income["50k-100k"] += weight
            elif income_value < 200000.0:
                income["100k-200k"] += weight
            else:
                income["200k+"] += weight

    if person_rows == 0 or household_rows == 0 or age.mean is None or household_size.mean is None:
        raise DataValidationError("empty_demographics", f"no usable city PUMS records for {spec.market_name}")
    return {
        "puma_codes": sorted(pumas),
        "person_sample_rows": person_rows,
        "household_sample_rows": household_rows,
        "agent": {
            "age_mean": float(age.mean),
            "age_std": float(max(5.0, age.std or 0.0)),
            "income_probs": _normalized_distribution(income, tuple(income)),
            "marital_probs": _normalized_distribution(marital, tuple(marital)),
            "gender_probs": _normalized_distribution(gender, tuple(gender)),
            "household_lambda": float(max(1.1, household_size.mean)),
        },
        "geography_note": (
            "PUMS records are filtered to 2020 PUMAs whose official TIGER names "
            f"match {spec.puma_name_pattern!r}; city-level ACS margins should be "
            "used for final raking when exact boundary equivalence is required."
        ),
    }


def derive_city_calibration(
    city: str,
    data_root: Path | str,
    ride_sample_rows: int = 20000,
    sample_seed: int = 4500,
) -> Dict[str, Any]:
    spec = resolve_city_spec(city)
    paths = default_source_paths(spec.key, data_root)
    weather_index = NOAAWeatherIndex.from_directory(spec.key, paths["weather"])
    rejections: Dict[str, int] = {}
    service = {"economy": 0.0, "premium": 0.0}
    day_of_week = {str(index): 0.0 for index in range(7)}
    hour = {str(index): 0.0 for index in range(24)}
    airport = 0
    shared = 0
    distances = _WeightedMoments()
    durations = _WeightedMoments()
    prices = _WeightedMoments()
    records = list(
        iter_normalized_rides(
            spec.key,
            paths["rides"],
            weather_root=paths["weather"],
            max_rows=ride_sample_rows,
            sample_seed=sample_seed,
            rejection_counts=rejections,
        )
    )
    for record in records:
        service[record.service] = service.get(record.service, 0.0) + 1.0
        day_of_week[str(record.day_of_week)] += 1.0
        hour[str(record.hour)] += 1.0
        airport += int(record.airport)
        shared += int(record.shared_ride)
        distances.add(record.distance_miles, 1.0)
        durations.add(record.duration_minutes, 1.0)
        prices.add(record.actual_price, 1.0)
    if not records:
        raise DataValidationError("empty_rides", f"no usable ride records for {spec.market_name}")
    demographics = derive_demographic_calibration(
        spec.key,
        paths["pums_person"],
        paths["pums_household"],
        paths["puma_geography"],
    )
    service_identified = spec.key == "boston"
    shared_identified = spec.key in {"boston", "chicago"}
    airport_identified = not (spec.key == "boston" and airport == 0)
    return {
        "schema_version": 1,
        "city": spec.market_name,
        "sources": dict(spec.sources),
        "ride_semantics": spec.ride_semantics,
        "price_semantics": spec.price_semantics,
        "sample_size": len(records),
        "quality": {
            "ride_rows_accepted": len(records),
            "ride_rejections": rejections,
            "person_sample_rows": demographics["person_sample_rows"],
            "household_sample_rows": demographics["household_sample_rows"],
            "puma_codes": demographics["puma_codes"],
            "geography_note": demographics["geography_note"],
            "identifiable_market_fields": {
                "airport_probability": airport_identified,
                "premium_service_probability": service_identified,
                "shared_ride_probability": shared_identified,
            },
        },
        "calibration": {
            "market": {
                "weather_probs": weather_index.category_probabilities(),
                "day_of_week_probs": _normalized_distribution(
                    day_of_week, tuple(str(index) for index in range(7))
                ),
                "hour_probs": _normalized_distribution(
                    hour, tuple(str(index) for index in range(24))
                ),
                "service_probs": (
                    _normalized_distribution(service, ("economy", "premium"))
                    if service_identified
                    else None
                ),
                "airport_prob": (airport / len(records)) if airport_identified else None,
                "shared_ride_prob": (shared / len(records)) if shared_identified else None,
                "distance_mean_miles": distances.mean,
                "distance_std_miles": distances.std,
                "duration_mean_minutes": durations.mean,
                "duration_std_minutes": durations.std,
                "observed_price_mean": prices.mean,
                "observed_price_std": prices.std,
            },
            "agent": demographics["agent"],
            "choice": {},
        },
    }


def validate_city_bundle(city: str, data_root: Path | str, sample_rows: int = 100) -> Dict[str, Any]:
    spec = resolve_city_spec(city)
    paths = default_source_paths(spec.key, data_root)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing required city sources: " + ", ".join(missing))
    ride_files = discover_ride_files(spec.key, paths["rides"])
    if not ride_files:
        raise FileNotFoundError(f"no ride files for {spec.market_name}")
    for path in (paths["pums_person"], paths["pums_household"], paths["puma_geography"]):
        with zipfile.ZipFile(path) as archive:
            bad = archive.testzip()
            if bad is not None:
                raise DataValidationError("zip_crc", f"CRC failure in {path}: {bad}")
    weather = NOAAWeatherIndex.from_directory(spec.key, paths["weather"])
    rejections: Dict[str, int] = {}
    records = list(
        iter_normalized_rides(
            spec.key,
            paths["rides"],
            weather_root=paths["weather"],
            max_rows=sample_rows,
            sample_seed=4500,
            rejection_counts=rejections,
        )
    )
    if len(records) < sample_rows:
        raise DataValidationError(
            "insufficient_rides", f"accepted {len(records)} of requested {sample_rows} rows"
        )
    return {
        "city": spec.market_name,
        "ride_semantics": spec.ride_semantics,
        "price_semantics": spec.price_semantics,
        "ride_files": [str(path.resolve()) for path in ride_files],
        "weather_observations": len(weather.observations),
        "puma_codes": list(puma_codes_for_city(spec.key, paths["puma_geography"])),
        "sample_rows_accepted": len(records),
        "sample_rejections": rejections,
        "sample_weather_join_rate": sum(r.weather_source == "noaa_global_hourly" for r in records) / len(records),
        "sample_duration_coverage": sum(r.duration_minutes is not None for r in records) / len(records),
    }


CANONICAL_RIDE_COLUMNS: Tuple[str, ...] = tuple(CanonicalRideRecord.__dataclass_fields__.keys())
