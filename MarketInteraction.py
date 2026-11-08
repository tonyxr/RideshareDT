from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xiaoru Shi

- Scenario generation + price quote function

- Factor alignment:
    - Weather + Day are shared for all rides in a single "day"
    - Hour is shared for all rides in a single timestep
    - Airport and Service are sampled per ride from distributions
    
rewritten Feb 6, 2026
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import mkl_config  # set oneMKL env before NumPy/Torch
del mkl_config
import numpy as np
from fast_scalar import sclip

from Market_models import MarketCoefficients, CoefficientOverrides
from market_shocks import ShockEffect, ShockProcess
from calibration_presets import NYC_PUBLIC_2024
from market_dynamics import (
    MarketCalendar,
    MarketDynamicsConfig,
    TripBank,
    expected_volume_multiplier,
    supply_multiplier as dynamics_supply_multiplier,
)

NYC_MARKET_CAL = NYC_PUBLIC_2024.get("calibration", {}).get("market", {})
NYC_WEATHER_FROM_DATA = NYC_MARKET_CAL.get("weather_probs", {}) if isinstance(NYC_MARKET_CAL, dict) else {}
NYC_SERVICE_FROM_DATA = NYC_MARKET_CAL.get("service_probs", {}) if isinstance(NYC_MARKET_CAL, dict) else {}
# Approximate long-run weather-state frequencies by city
CITY_WEATHER_HISTORY: Dict[str, Dict[str, float]] = {
    "General": {"clear": 0.60, "rain": 0.28, "snow": 0.12},
    "Seattle": {"clear": 0.56, "rain": 0.36, "snow": 0.08},
    "New York City": {
        "clear": float(NYC_WEATHER_FROM_DATA.get("clear", 0.58)),
        "rain": float(NYC_WEATHER_FROM_DATA.get("rain", 0.29)),
        "snow": float(NYC_WEATHER_FROM_DATA.get("snow", 0.13)),
    },
    "Boston": {"clear": 0.57, "rain": 0.29, "snow": 0.14},
    "Chicago": {"clear": 0.54, "rain": 0.27, "snow": 0.19},
}

@dataclass(frozen=True)
class DayContext:
    day_of_week: int  # 0..6
    weather: str      # clear/rain/snow


@dataclass(frozen=True)
class TimeContext:
    hour: int         # 0..23


@dataclass(frozen=True)
class SyntheticRide:
    day_of_week: int
    weather: str
    hour: int
    airport: bool
    service: str
    distance_miles: float
    duration_minutes: float
    
@dataclass(frozen=True)
class RideContext:
    day_of_week: int
    weather: str
    hour: int
    airport: bool
    service: str  


@dataclass(frozen=True)
class FareQuote:
    """Auditable split between the rider charge and platform economics."""

    rider_fare: float
    platform_fare: float
    pass_through: float


class MarketInteraction:
    MARKET_ALIASES: Dict[str, str] = {
        "general": "General",
        "seattle": "Seattle",
        "new york city": "New York City",
        "new york": "New York City",
        "nyc": "New York City",
        "boston": "Boston",
        "boston ma": "Boston",
        "boston massachusetts": "Boston",
        "chicago": "Chicago",
    }

    def __init__(self, city_name: str = "Seattle", seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.market_list: Dict[str, MarketCoefficients] = {}
        self._init_markets()
        self.current_city = self.resolve_market_name(city_name)
        self.curr_market: MarketCoefficients = self.market_list[self.current_city]

        self._reset_probability_priors()
        self._apply_probability_variation()
        # v26 sequential/city-calibrated dynamics (disabled = legacy behavior).
        self.dynamics = MarketDynamicsConfig()
        self.dynamics_steps_per_day = 6
        self.trip_bank: Optional[TripBank] = None
        self.weather_demand_multiplier: Dict[str, float] = {}
        self.calendar = MarketCalendar()
        self.calendar_wraps = 0
        self.preset_market: Optional[MarketCoefficients] = None
        # v26r stochastic regime process (None = the v26 market).  Lives here
        # so validation rebuilds and counterfactual copies carry it with the
        # calendar.  Inactive unless the config is enabled AND the sequential
        # calendar is on, so the legacy i.i.d. market is untouched.
        self.shocks: Optional[ShockProcess] = None
        self.shock_training_openings = False
        self.shock_training_episode_days = 30
        # V17 M4.  Legacy is the byte-compatible default.
        self.fare_model = "legacy"

    def configure_fare_model(self, fare_model: str = "legacy") -> None:
        model = str(fare_model).strip().lower()
        if model not in {"legacy", "realistic"}:
            raise ValueError("fare_model must be 'legacy' or 'realistic'")
        self.fare_model = model

    # ---------- v26 dynamics ----------
    def configure_dynamics(
        self,
        config: MarketDynamicsConfig,
        *,
        steps_per_day: Optional[int] = None,
        trip_bank: Optional[TripBank] = None,
        weather_demand_multiplier: Optional[Dict[str, float]] = None,
    ) -> None:
        self.dynamics = config
        if steps_per_day is not None:
            self.dynamics_steps_per_day = int(max(1, steps_per_day))
        if trip_bank is not None:
            self.trip_bank = trip_bank
        if weather_demand_multiplier is not None:
            self.weather_demand_multiplier = {
                str(k): float(v) for k, v in weather_demand_multiplier.items()
            }
        self.calendar = MarketCalendar(steps_per_day=self.dynamics_steps_per_day)

    @property
    def sequential_calendar(self) -> bool:
        return bool(self.dynamics.enabled and self.dynamics.sequential_calendar)

    def register_calibrated_tariff(self, tariff: Dict[str, object], *, city: Optional[str] = None) -> None:
        """Install data-calibrated anchors for ``city`` without resetting priors."""
        name = str(city or self.current_city)
        if name not in self.market_list:
            self.market_list[name] = self.curr_market
            self.MARKET_ALIASES = dict(self.MARKET_ALIASES)
            self.MARKET_ALIASES[name.lower()] = name
        previous = self.market_list[name]
        if self.preset_market is None and name == self.current_city:
            self.preset_market = previous
        surge = [
            (int(item[0]), int(item[1]), float(item[2]))
            for item in tariff.get("surge_hours", previous.surge_hours)  # type: ignore[union-attr]
        ]
        day_multiplier = {
            int(k): float(v)
            for k, v in dict(tariff.get("day_multiplier", previous.day_multiplier)).items()  # type: ignore[arg-type]
        }
        weather_multiplier = dict(previous.weather_multiplier)
        weather_multiplier.update({
            str(k): float(v)
            for k, v in dict(tariff.get("weather_multiplier", {})).items()  # type: ignore[arg-type]
        })
        service_multiplier = dict(previous.service_multiplier)
        service_multiplier.update({
            str(k): float(v)
            for k, v in dict(tariff.get("service_multiplier", {})).items()  # type: ignore[arg-type]
        })
        market = MarketCoefficients(
            base_fare=float(tariff["base_fare"]),
            per_mile=float(tariff["per_mile"]),
            per_minute=float(tariff["per_minute"]),
            booking_fee=float(tariff["booking_fee"]),
            airport_fee=float(tariff["airport_fee"]),
            surge_hours=surge,
            day_multiplier=day_multiplier,
            weather_multiplier=weather_multiplier,
            service_multiplier=service_multiplier,
        )
        self.market_list[name] = market
        if name == self.current_city:
            self.curr_market = market

    def demand_volume_multiplier(self, *, hour: int, day_of_week: int, weather: str) -> float:
        if not (self.dynamics.enabled and self.dynamics.hourly_demand_volume):
            return 1.0
        steps = int(max(1, self.dynamics_steps_per_day))
        share = self._block_share(int(hour), steps)
        return expected_volume_multiplier(
            block_share=share,
            steps_per_day=steps,
            day_of_week=int(day_of_week),
            day_of_week_probs=np.asarray(self.day_of_week_probs, dtype=float),
            weather=str(weather),
            weather_demand_multiplier=self.weather_demand_multiplier,
        )

    def driver_supply_multiplier(self, *, hour: int) -> float:
        if not (self.dynamics.enabled and self.dynamics.hourly_demand_volume):
            return 1.0
        steps = int(max(1, self.dynamics_steps_per_day))
        return dynamics_supply_multiplier(
            block_share=self._block_share(int(hour), steps),
            steps_per_day=steps,
            tracking=float(self.dynamics.supply_tracking),
        )

    def _block_share(self, hour: int, steps: int) -> float:
        edges = np.linspace(0, 24, steps + 1)
        block = int(min(steps - 1, np.searchsorted(edges, int(hour) % 24, side="right") - 1))
        lo, hi = int(round(edges[block])), int(round(edges[block + 1]))
        probs = np.asarray(self.hour_probs, dtype=float)
        probs = probs / max(1e-12, float(probs.sum()))
        return float(probs[np.arange(lo, max(hi, lo + 1)) % 24].sum())

    def sample_operational_trips(
        self,
        rng: np.random.Generator,
        *,
        hour: int,
        day_of_week: int,
        size: int,
    ) -> Optional[Dict[str, np.ndarray]]:
        """Empirical trips when enabled, else ``None`` (caller keeps legacy draw)."""
        if not (self.dynamics.enabled and self.dynamics.empirical_trips and self.trip_bank is not None):
            return None
        trips = self.trip_bank.sample(
            rng,
            hour=int(hour),
            day_of_week=int(day_of_week),
            size=int(size),
            window=int(self.dynamics.trip_hour_window),
            jitter=float(self.dynamics.trip_jitter),
        )
        if not self.trip_bank.airport_identified:
            trips["airport"] = rng.random(int(size)) < float(self.airport_prob)
        if not self.trip_bank.service_identified:
            keys = list(self.service_probs.keys())
            probs = np.array([self.service_probs[k] for k in keys], dtype=float)
            trips["service"] = rng.choice(keys, size=int(size), p=probs / probs.sum())
        else:
            known = set(self.curr_market.service_multiplier.keys())
            trips["service"] = np.array(
                [value if value in known else "economy" for value in trips["service"]],
                dtype=object,
            )
        return trips

    def set_market(self, name: str) -> None:
        self.current_city = self.resolve_market_name(name)
        self.curr_market = self.market_list[self.current_city]
        self._reset_probability_priors()
        self._apply_probability_variation()

    def resolve_market_name(self, name: str) -> str:
        """Return one canonical supported market; never silently relabel data."""
        requested = " ".join(str(name or "").strip().split())
        alias = self.MARKET_ALIASES.get(requested.lower())
        if alias in self.market_list:
            return str(alias)
        casefolded = {
            str(market).casefold(): str(market)
            for market in self.market_list
        }
        resolved = casefolded.get(requested.casefold())
        if resolved is not None:
            return resolved
        supported = ", ".join(sorted(self.market_list))
        raise ValueError(
            f"unsupported market {name!r}; choose one of: {supported}"
        )

    def _init_markets(self) -> None:
        self.market_list["General"] = MarketCoefficients(
            base_fare=2.50, per_mile=1.50, per_minute=0.25, booking_fee=2.00,
            airport_fee=4.00,
            surge_hours=[(0, 2, 1.60), (2, 5, 0.90), (5, 9, 1.25), (9, 15, 1.00),
                         (15, 19, 1.40), (19, 23, 1.40), (23, 24, 1.00)],
            day_multiplier={5: 1.10, 6: 1.15},
            weather_multiplier={"clear": 1.0, "rain": 1.15, "snow": 1.30},
            service_multiplier={"economy": 1.0, "premium": 1.70},
        )

        self.market_list["Seattle"] = MarketCoefficients(
            base_fare=2.60, per_mile=1.55, per_minute=0.26, booking_fee=2.05,
            airport_fee=5.00,
            surge_hours=[(0, 2, 1.30), (2, 5, 1.00), (5, 9, 1.50),
                         (9, 15, 1.20), (15, 19, 1.70), (19, 23, 1.70), (23, 24, 0.90)],
            day_multiplier={5: 1.10, 6: 1.15},
            weather_multiplier={"clear": 1.0, "rain": 1.20, "snow": 1.35},
            service_multiplier={"economy": 1.0, "premium": 1.60},
        )

        self.market_list["New York City"] = MarketCoefficients(
            base_fare=3.00, per_mile=1.90, per_minute=0.40, booking_fee=2.75,
            airport_fee=6.00,
            surge_hours=[(0, 6, 1.0), (6, 10, 1.40), (10, 15, 1.05), (15, 20, 1.60), (20, 24, 1.30)],
            day_multiplier={5: 1.10, 6: 1.15},
            weather_multiplier={"clear": 1.0, "rain": 1.25, "snow": 1.40},
            service_multiplier={"economy": 1.0, "premium": 1.80},
        )

        self.market_list["Boston"] = MarketCoefficients(
            base_fare=2.50, per_mile=1.55, per_minute=0.26, booking_fee=2.05,
            airport_fee=4.50,
            surge_hours=[(0, 6, 1.0), (6, 10, 1.30), (10, 15, 1.00),
                         (15, 20, 1.45), (20, 24, 1.20)],
            day_multiplier={5: 1.10, 6: 1.12},
            weather_multiplier={"clear": 1.0, "rain": 1.20, "snow": 1.38},
            service_multiplier={"economy": 1.0, "premium": 1.72},
        )

        self.market_list["Chicago"] = MarketCoefficients(
            base_fare=2.40, per_mile=1.45, per_minute=0.24, booking_fee=1.95,
            airport_fee=5.00,
            surge_hours=[(6, 9, 1.25), (9, 15, 1.00), (15, 19, 1.45), (19, 23, 1.15), (23, 24, 1.55)],
            day_multiplier={5: 1.10, 6: 1.15},
            weather_multiplier={"clear": 1.0, "rain": 1.18, "snow": 1.35},
            service_multiplier={"economy": 1.0, "premium": 1.65},
        )
    
    
    @staticmethod
    def _normalize_probs(vals: np.ndarray) -> np.ndarray:
        arr = sclip(np.array(vals, dtype=float), 1e-6, None)
        s = float(arr.sum())
        if s <= 0.0:
            return np.ones_like(arr, dtype=float) / float(len(arr))
        return arr / s

    def _reset_probability_priors(self) -> None:
        self._base_weather_probs = self._weather_probs_for_city(self.current_city)
        self._base_day_of_week_probs = np.ones(7, dtype=float) / 7.0
        self._base_hour_probs = self._default_hour_probabilities()
        self._base_airport_prob = (
            float(NYC_MARKET_CAL.get("airport_prob", 0.12))
            if self.current_city == "New York City"
            else 0.12
        )
        service_keys = list(self.curr_market.service_multiplier.keys())
        if self.current_city == "New York City" and NYC_SERVICE_FROM_DATA:
            values = np.array(
                [float(NYC_SERVICE_FROM_DATA.get(key, 0.0)) for key in service_keys],
                dtype=float,
            )
        else:
            values = np.array(
                [0.85 if key == "economy" else 0.15 for key in service_keys],
                dtype=float,
            )
        values = self._normalize_probs(values)
        self._base_service_probs = {
            key: float(value) for key, value in zip(service_keys, values)
        }
        self.weather_probs = dict(self._base_weather_probs)
        self.day_of_week_probs = self._base_day_of_week_probs.copy()
        self.hour_probs = self._base_hour_probs.copy()
        self.airport_prob = float(self._base_airport_prob)
        self.service_probs = dict(self._base_service_probs)
        self._base_distance_mean_miles = 6.9
        self._base_distance_std_miles = float(np.sqrt(2.3) * 3.0)
        self._base_duration_mean_minutes: Optional[float] = None
        self._base_duration_std_minutes: Optional[float] = None
        self.distance_mean_miles = self._base_distance_mean_miles
        self.distance_std_miles = self._base_distance_std_miles
        self.duration_mean_minutes = self._base_duration_mean_minutes
        self.duration_std_minutes = self._base_duration_std_minutes

    def _default_hour_probabilities(self) -> np.ndarray:
        weights = np.ones(24, dtype=float)
        for start, end, multiplier in self.curr_market.surge_hours:
            start %= 24
            end %= 24
            if start <= end:
                weights[start:end] *= float(multiplier)
            else:
                weights[start:24] *= float(multiplier)
                weights[0:end] *= float(multiplier)
        return self._normalize_probs(weights)

    @classmethod
    def _indexed_probabilities(
        cls,
        values: Optional[Dict[str, float]],
        size: int,
    ) -> Optional[np.ndarray]:
        if not isinstance(values, dict) or not values:
            return None
        probabilities = np.array(
            [float(values.get(str(index), values.get(index, 0.0))) for index in range(size)],
            dtype=float,
        )
        if not np.all(np.isfinite(probabilities)) or float(np.sum(probabilities)) <= 0.0:
            return None
        return cls._normalize_probs(probabilities)

    def set_probability_priors(
        self,
        *,
        weather_probs: Optional[Dict[str, float]] = None,
        day_of_week_probs: Optional[Dict[str, float]] = None,
        hour_probs: Optional[Dict[str, float]] = None,
        airport_prob: Optional[float] = None,
        service_probs: Optional[Dict[str, float]] = None,
        distance_mean_miles: Optional[float] = None,
        distance_std_miles: Optional[float] = None,
        duration_mean_minutes: Optional[float] = None,
        duration_std_minutes: Optional[float] = None,
    ) -> None:
        """Install empirical priors as the base for subsequent run jitter."""

        if isinstance(weather_probs, dict) and weather_probs:
            keys = list(self.curr_market.weather_multiplier.keys())
            values = self._normalize_probs(
                np.array([float(weather_probs.get(key, 0.0)) for key in keys], dtype=float)
            )
            self._base_weather_probs = {
                key: float(value) for key, value in zip(keys, values)
            }
        calibrated_days = self._indexed_probabilities(day_of_week_probs, 7)
        if calibrated_days is not None:
            self._base_day_of_week_probs = calibrated_days
        calibrated_hours = self._indexed_probabilities(hour_probs, 24)
        if calibrated_hours is not None:
            self._base_hour_probs = calibrated_hours
        if airport_prob is not None:
            self._base_airport_prob = float(sclip(float(airport_prob), 0.01, 0.60))
        if isinstance(service_probs, dict) and service_probs:
            keys = list(self.curr_market.service_multiplier.keys())
            values = self._normalize_probs(
                np.array([float(service_probs.get(key, 0.0)) for key in keys], dtype=float)
            )
            self._base_service_probs = {
                key: float(value) for key, value in zip(keys, values)
            }
        if distance_mean_miles is not None and float(distance_mean_miles) > 0.0:
            self._base_distance_mean_miles = float(distance_mean_miles)
        if distance_std_miles is not None and float(distance_std_miles) > 0.0:
            self._base_distance_std_miles = float(distance_std_miles)
        if duration_mean_minutes is not None and float(duration_mean_minutes) > 0.0:
            self._base_duration_mean_minutes = float(duration_mean_minutes)
        if duration_std_minutes is not None and float(duration_std_minutes) > 0.0:
            self._base_duration_std_minutes = float(duration_std_minutes)
        self.weather_probs = dict(self._base_weather_probs)
        self.day_of_week_probs = self._base_day_of_week_probs.copy()
        self.hour_probs = self._base_hour_probs.copy()
        self.airport_prob = float(self._base_airport_prob)
        self.service_probs = dict(self._base_service_probs)
        self.distance_mean_miles = float(self._base_distance_mean_miles)
        self.distance_std_miles = float(self._base_distance_std_miles)
        self.duration_mean_minutes = self._base_duration_mean_minutes
        self.duration_std_minutes = self._base_duration_std_minutes

    def _apply_probability_variation(self, jitter_scale: float = 0.05) -> None:
        """Apply slight run-level perturbations to weather/ride nature priors."""
        j = float(max(0.0, jitter_scale))

        base_weather = self._base_weather_probs
        weather_keys = list(self.curr_market.weather_multiplier.keys())
        weather_vals = np.array([float(base_weather.get(k, 0.0)) for k in weather_keys], dtype=float)
        weather_vals = weather_vals + self.rng.normal(0.0, j, size=len(weather_vals))
        weather_vals = self._normalize_probs(weather_vals)
        self.weather_probs = {k: float(v) for k, v in zip(weather_keys, weather_vals)}

        self.day_of_week_probs = self._normalize_probs(
            self._base_day_of_week_probs
            + self.rng.normal(0.0, j / 7.0, size=7)
        )
        self.hour_probs = self._normalize_probs(
            self._base_hour_probs
            + self.rng.normal(0.0, j / 24.0, size=24)
        )

        airport_base = float(self._base_airport_prob)
        self.airport_prob = float(sclip(airport_base + self.rng.normal(0.0, 0.05 * j), 0.03, 0.35))

        service_keys = list(self.curr_market.service_multiplier.keys())
        base_service = np.array(
            [float(self._base_service_probs.get(k, 0.0)) for k in service_keys],
            dtype=float,
        )
        service_vals = self._normalize_probs(base_service + self.rng.normal(0.0, j, size=len(service_keys)))
        self.service_probs = {k: float(v) for k, v in zip(service_keys, service_vals)}
        moment_jitter = float(self.rng.normal(0.0, 0.10 * j))
        self.distance_mean_miles = float(max(
            0.25, self._base_distance_mean_miles * (1.0 + moment_jitter)
        ))
        self.distance_std_miles = float(max(
            0.10, self._base_distance_std_miles * (1.0 + moment_jitter)
        ))
        self.duration_mean_minutes = (
            None
            if self._base_duration_mean_minutes is None
            else float(max(1.0, self._base_duration_mean_minutes * (1.0 + moment_jitter)))
        )
        self.duration_std_minutes = (
            None
            if self._base_duration_std_minutes is None
            else float(max(0.5, self._base_duration_std_minutes * (1.0 + moment_jitter)))
        )

    def refresh_run_probabilities(self, jitter_scale: float = 0.05) -> None:
        """Public wrapper to refresh run-level weather/ride priors with slight perturbation."""
        self._apply_probability_variation(jitter_scale=jitter_scale)

    def _weather_probs_for_city(self, city: str) -> Dict[str, float]:
        base = CITY_WEATHER_HISTORY.get(city, CITY_WEATHER_HISTORY["General"])
        keys = list(self.curr_market.weather_multiplier.keys())
        vals = np.array([float(base.get(k, 0.0)) for k in keys], dtype=float)
        if vals.sum() <= 0:
            vals = np.ones(len(keys), dtype=float)
        vals = vals / vals.sum()
        return {k: float(v) for k, v in zip(keys, vals)}

    # ---------- Scenario sampling ----------
    def _iid_day_context(self) -> DayContext:
        day = int(self.rng.choice(np.arange(7), p=self.day_of_week_probs))
        keys = list(self.weather_probs.keys())
        probs = np.array([self.weather_probs[k] for k in keys], dtype=float)
        probs = probs / probs.sum()
        weather = str(self.rng.choice(keys, p=probs))
        return DayContext(day_of_week=day, weather=weather)

    def _iid_hour(self) -> TimeContext:
        hour = int(self.rng.choice(np.arange(24), p=self.hour_probs))
        return TimeContext(hour=hour)

    def sample_day_context(
        self,
        *,
        calendar_rng: Optional[np.random.Generator] = None,
        weather_rng: Optional[np.random.Generator] = None,
        hour_rng: Optional[np.random.Generator] = None,
    ) -> DayContext:
        """Next operating day.

        Legacy: an i.i.d. weekday and weather.  v26 sequential calendar:
        consecutive weekdays, persistent weather, and a fresh ordered set of
        intra-day time blocks (consumed by ``sample_timestep_hour``).
        """
        if not self.sequential_calendar:
            if calendar_rng is None and weather_rng is None and hour_rng is None:
                return self._iid_day_context()
            primary_rng = self.rng if calendar_rng is None else calendar_rng
            weather_source = primary_rng if weather_rng is None else weather_rng
            day = int(primary_rng.choice(7, p=self.day_of_week_probs))
            keys = list(self.weather_probs.keys())
            probs = np.array([self.weather_probs[k] for k in keys], dtype=float)
            probs = probs / probs.sum()
            weather = str(weather_source.choice(keys, p=probs))
            return DayContext(day_of_week=day, weather=weather)
        primary_rng = self.rng if calendar_rng is None else calendar_rng
        weather_source = primary_rng if weather_rng is None else weather_rng
        shocks = self.active_shocks
        if self.calendar.day_index < 0:
            self.calendar.reset(
                primary_rng,
                day_of_week_probs=np.asarray(self.day_of_week_probs, dtype=float),
                weather_probs=self.weather_probs,
                weather_rng=weather_source,
            )
            if shocks is not None:
                shocks.reset(
                    training_opening=bool(self.shock_training_openings),
                    training_episode_days=int(self.shock_training_episode_days),
                )
        self.calendar.advance_day(
            primary_rng,
            steps_per_day=self.dynamics_steps_per_day,
            hour_probs=np.asarray(self.hour_probs, dtype=float),
            weather_probs=self.weather_probs,
            persistence=float(self.dynamics.weather_persistence),
            weather_rng=weather_source,
            hour_rng=(primary_rng if hour_rng is None else hour_rng),
        )
        weather = str(self.calendar.weather)
        if shocks is not None:
            # v26r: a dated calendar.  The weekday follows the date so holidays
            # fall on their real weekdays; a winter storm overrides the day's
            # weather without disturbing the weather chain's persistence.
            shocks.steps_per_day = int(max(1, self.dynamics_steps_per_day))
            date = shocks.advance_day()
            self.calendar.day_of_week = int(date.weekday())
            override = shocks.day_weather_override()
            if override:
                weather = str(override)
        return DayContext(
            day_of_week=int(self.calendar.day_of_week),
            weather=weather,
        )

    def sample_timestep_hour(
        self, rng: Optional[np.random.Generator] = None
    ) -> TimeContext:
        """Next operating step's hour (ordered blocks in v26, i.i.d. otherwise)."""
        if not self.sequential_calendar:
            if rng is None:
                return self._iid_hour()
            return TimeContext(
                hour=int(rng.choice(np.arange(24), p=self.hour_probs))
            )
        if self.calendar.day_index < 0:
            self.sample_day_context()
        if self.calendar.block + 1 >= len(self.calendar._day_hours):
            # More steps were requested than the configured blocks per day.
            # Wrap within the same day rather than silently skipping a date.
            self.calendar.block = -1
            self.calendar_wraps += 1
        _, hour, _ = self.calendar.next_block()
        return TimeContext(hour=int(hour))

    def sample_airport_flag(
        self, rng: Optional[np.random.Generator] = None
    ) -> bool:
        source = self.rng if rng is None else rng
        return bool(source.random() < self.airport_prob)

    # ---------- v26r regime process ----------
    @property
    def active_shocks(self) -> Optional[ShockProcess]:
        shocks = self.shocks
        if shocks is None or not shocks.config.enabled or not self.sequential_calendar:
            return None
        return shocks

    def install_shocks(self, process: Optional[ShockProcess]) -> None:
        self.shocks = process
        self.calendar.day_index = -1

    def block_index_for_hour(self, hour: int) -> int:
        steps = int(max(1, self.dynamics_steps_per_day))
        edges = np.linspace(0, 24, steps + 1)
        return int(min(steps - 1, np.searchsorted(edges, int(hour) % 24, side="right") - 1))

    def shock_effect(self, hour: int) -> ShockEffect:
        """Composite regime effect for the current day and ``hour``'s block."""
        shocks = self.active_shocks
        if shocks is None or shocks.day_index < 0:
            return ShockEffect()
        return shocks.effect(self.block_index_for_hour(int(hour)))

    def shock_public_features(self, hour: int) -> Dict[str, float]:
        shocks = self.active_shocks
        if shocks is None or shocks.day_index < 0 or not shocks.config.public_features:
            return {}
        return dict(self.shock_effect(int(hour)).public)

    def sample_service(
        self, rng: Optional[np.random.Generator] = None
    ) -> str:
        keys = list(self.service_probs.keys())
        probs = np.array([self.service_probs[k] for k in keys], dtype=float)
        probs = probs / probs.sum()
        source = self.rng if rng is None else rng
        return str(source.choice(keys, p=probs))

    # ---------- Pricing ----------
    def _time_multiplier(self, hour: int) -> float:
        for start, end, mult in self.curr_market.surge_hours:
            start %= 24
            end %= 24
            if start <= end:
                if start <= hour < end:
                    return float(mult)
            else:
                if hour >= start or hour < end:
                    return float(mult)
        return 1.0

    def quote_components(
        self,
        distance_miles: float,
        duration_minutes: float,
        ctx: RideContext,
        overrides: Optional[CoefficientOverrides] = None,
        extra_fees: float = 0.0,
    ) -> FareQuote:
        m = self.curr_market
        o = overrides or CoefficientOverrides()

        base_fare = m.base_fare if o.base_fare is None else float(o.base_fare)
        per_mile = m.per_mile if o.per_mile is None else float(o.per_mile)
        per_minute = m.per_minute if o.per_minute is None else float(o.per_minute)
        booking_fee = m.booking_fee if o.booking_fee is None else float(o.booking_fee)
        airport_fee = m.airport_fee if o.airport_fee is None else float(o.airport_fee)

        day_map = m.day_multiplier if o.day_multiplier is None else o.day_multiplier
        weather_map = m.weather_multiplier if o.weather_multiplier is None else o.weather_multiplier
        service_map = m.service_multiplier if o.service_multiplier is None else o.service_multiplier

        base = base_fare + booking_fee
        variable = per_mile * max(distance_miles, 0.0) + per_minute * max(duration_minutes, 0.0)

        tod = self._time_multiplier(ctx.hour)
        dow = float(day_map.get(ctx.day_of_week, 1.0))
        wco = float(weather_map.get(ctx.weather, 1.0))
        sco = float(service_map.get(ctx.service, 1.0))

        context = tod * dow * wco * sco
        if self.fare_model == "realistic":
            # V17 M4: the fixed charges stay fixed; context affects only the
            # time/distance portion.  Airport and external fees are collected
            # from the rider but excluded from platform revenue and driver pay.
            platform_fare = max(base + variable * context, 0.0)
            pass_through = max(
                (airport_fee if ctx.airport else 0.0) + float(extra_fees),
                0.0,
            )
            rider_fare = round(platform_fare + pass_through, 2)
            pass_through = round(min(pass_through, rider_fare), 2)
            platform_fare = round(rider_fare - pass_through, 2)
            return FareQuote(rider_fare, platform_fare, pass_through)

        # Historical behavior: every charge is platform fare and the context
        # multiplier applies to fixed and variable components alike.
        price = (base + variable) * context
        if ctx.airport:
            price += airport_fee
        price += float(extra_fees)
        rider_fare = round(max(price, 0.0), 2)
        return FareQuote(rider_fare, rider_fare, 0.0)

    def quote_price(
        self,
        distance_miles: float,
        duration_minutes: float,
        ctx: RideContext,
        overrides: Optional[CoefficientOverrides] = None,
        extra_fees: float = 0.0,
    ) -> float:
        return self.quote_components(
            distance_miles,
            duration_minutes,
            ctx,
            overrides=overrides,
            extra_fees=extra_fees,
        ).rider_fare
    
    def flatten_ride_context(self, ctx: RideContext) -> np.ndarray:
        """Numerical encoding of a RideContext for RL state construction."""
        weather_keys = list(self.curr_market.weather_multiplier.keys())
        service_keys = list(self.curr_market.service_multiplier.keys())

        weather_one_hot = [1.0 if ctx.weather == k else 0.0 for k in weather_keys]
        service_one_hot = [1.0 if ctx.service == k else 0.0 for k in service_keys]
        weekend = 1.0 if int(ctx.day_of_week) >= 5 else 0.0

        feats: List[float] = [
            float(sclip(ctx.hour / 23.0, 0.0, 1.0)),
            float(sclip(ctx.day_of_week / 6.0, 0.0, 1.0)),
            weekend,
            1.0 if bool(ctx.airport) else 0.0,
            *weather_one_hot,
            *service_one_hot,
        ]
        return np.array(feats, dtype=np.float32)

    def apply_step_actions_to_overrides(
        self,
        overrides: CoefficientOverrides,
        action_steps: Dict[str, int],
        step_size: Dict[str, float],
        bounds: Dict[str, Tuple[float, float]],
    ) -> None:
        """Map discrete coefficient steps {-1,0,1} to concrete market override updates."""
        for key, step_dir in action_steps.items():
            if step_dir == 0:
                continue

            current = getattr(overrides, key)
            if current is None:
                current = float(getattr(self.curr_market, key))

            delta = float(step_size[key]) * float(np.sign(step_dir))
            lb, ub = bounds[key]
            setattr(overrides, key, float(sclip(float(current) + delta, lb, ub)))

    def generate_ride(self) -> SyntheticRide:
        """Sample a synthetic ride request for bootstrapping profile cold-start scenarios."""
        # Cold-start evidence rides are an i.i.d. draw of plausible trips and
        # must never advance the operational v26 calendar.
        day_ctx = self._iid_day_context()
        time_ctx = self._iid_hour()
        airport = self.sample_airport_flag()
        service = self.sample_service()

        distance_mean = float(max(0.25, self.distance_mean_miles))
        distance_std = float(max(0.10, self.distance_std_miles))
        distance_shape = float(max(0.10, (distance_mean / distance_std) ** 2))
        distance_scale = float(max(0.01, distance_std ** 2 / distance_mean))
        distance_miles = float(sclip(
            self.rng.gamma(shape=distance_shape, scale=distance_scale),
            0.25,
            100.0,
        ))
        if self.duration_mean_minutes is None:
            avg_speed_mph = float(sclip(
                self.rng.normal(loc=18.0, scale=4.0), 8.0, 35.0
            ))
            duration_minutes = float(
                (distance_miles / max(avg_speed_mph, 1e-6)) * 60.0
            )
        else:
            duration_mean = float(max(1.0, self.duration_mean_minutes))
            duration_std = float(max(0.5, self.duration_std_minutes or 0.25 * duration_mean))
            distance_slope = float(min(
                duration_mean / distance_mean,
                0.75 * duration_std / distance_std,
            ))
            residual_std = float(np.sqrt(max(
                0.25,
                duration_std ** 2 - (distance_slope * distance_std) ** 2,
            )))
            duration_minutes = float(
                duration_mean
                + distance_slope * (distance_miles - distance_mean)
                + self.rng.normal(0.0, residual_std)
            )
        duration_minutes = float(sclip(duration_minutes, 1.0, 180.0))

        return SyntheticRide(
            day_of_week=int(day_ctx.day_of_week),
            weather=str(day_ctx.weather),
            hour=int(time_ctx.hour),
            airport=bool(airport),
            service=str(service),
            distance_miles=distance_miles,
            duration_minutes=duration_minutes,
        )
