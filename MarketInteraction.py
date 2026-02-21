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
from typing import Dict, Any, Optional, Tuple
import numpy as np

from Market_models import MarketCoefficients, CoefficientOverrides

# Approximate long-run weather-state frequencies by city
CITY_WEATHER_HISTORY: Dict[str, Dict[str, float]] = {
    "General": {"clear": 0.60, "rain": 0.28, "snow": 0.12},
    "Seattle": {"clear": 0.56, "rain": 0.36, "snow": 0.08},
    "New York City": {"clear": 0.58, "rain": 0.29, "snow": 0.13},
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
class RideContext:
    day_of_week: int
    weather: str
    hour: int
    airport: bool
    service: str      # economy/premium


class MarketInteraction:
    def __init__(self, city_name: str = "Seattle", seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.market_list: Dict[str, MarketCoefficients] = {}
        self._init_markets()
        self.current_city = city_name if city_name in self.market_list else "General"
        self.curr_market: MarketCoefficients = self.market_list[self.current_city]

        # scenario priors (can be city-conditioned later)
        self.airport_prob = 0.12
        self.service_probs = {"economy": 0.85, "premium": 0.15}
        self.weather_probs = self._weather_probs_for_city(self.current_city)

    def set_market(self, name: str) -> None:
        self.current_city = name if name in self.market_list else "General"
        self.curr_market = self.market_list[self.current_city]
        self.weather_probs = self._weather_probs_for_city(self.current_city)

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

        self.market_list["Chicago"] = MarketCoefficients(
            base_fare=2.40, per_mile=1.45, per_minute=0.24, booking_fee=1.95,
            airport_fee=5.00,
            surge_hours=[(6, 9, 1.25), (9, 15, 1.00), (15, 19, 1.45), (19, 23, 1.15), (23, 24, 1.55)],
            day_multiplier={5: 1.10, 6: 1.15},
            weather_multiplier={"clear": 1.0, "rain": 1.18, "snow": 1.35},
            service_multiplier={"economy": 1.0, "premium": 1.65},
        )
        
    def _weather_probs_for_city(self, city: str) -> Dict[str, float]:
        base = CITY_WEATHER_HISTORY.get(city, CITY_WEATHER_HISTORY["General"])
        keys = list(self.curr_market.weather_multiplier.keys())
        vals = np.array([float(base.get(k, 0.0)) for k in keys], dtype=float)
        if vals.sum() <= 0:
            vals = np.ones(len(keys), dtype=float)
        vals = vals / vals.sum()
        return {k: float(v) for k, v in zip(keys, vals)}

    # ---------- Scenario sampling ----------
    def sample_day_context(self) -> DayContext:
        day = int(self.rng.integers(0, 7))
        keys = list(self.weather_probs.keys())
        probs = np.array([self.weather_probs[k] for k in keys], dtype=float)
        probs = probs / probs.sum()
        weather = str(self.rng.choice(keys, p=probs))
        return DayContext(day_of_week=day, weather=weather)

    def sample_timestep_hour(self) -> TimeContext:
        weights = np.ones(24, dtype=float)
        for start, end, mult in self.curr_market.surge_hours:
            start %= 24
            end %= 24
            if start <= end:
                weights[start:end] *= float(mult) if start != end else float(mult)
            else:
                weights[start:24] *= float(mult)
                weights[0:end] *= float(mult)

        probs = weights / weights.sum() if weights.sum() > 0 else np.ones(24) / 24
        hour = int(self.rng.choice(np.arange(24), p=probs))
        return TimeContext(hour=hour)

    def sample_airport_flag(self) -> bool:
        return bool(self.rng.random() < self.airport_prob)

    def sample_service(self) -> str:
        keys = list(self.service_probs.keys())
        probs = np.array([self.service_probs[k] for k in keys], dtype=float)
        probs = probs / probs.sum()
        return str(self.rng.choice(keys, p=probs))

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

    def quote_price(
        self,
        distance_miles: float,
        duration_minutes: float,
        ctx: RideContext,
        overrides: Optional[CoefficientOverrides] = None,
        extra_fees: float = 0.0,
    ) -> float:
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

        price = (base + variable) * tod * dow * wco * sco
        if ctx.airport:
            price += airport_fee

        price += float(extra_fees)
        return round(max(price, 0.0), 2)
    