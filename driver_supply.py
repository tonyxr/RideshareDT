#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 14:13:52 2026

@author: Xiaoru Shi
"""

from __future__ import annotations

"""Driver supply, acceptance, matching, and optional OpenStreetMap/OSMnx hooks.

The first implementation is intentionally lightweight and batch-friendly: Firm 1's
RL agent still controls rider-facing fare coefficients only (Option A), while the
driver layer responds endogenously through supply, wait times, and acceptance.
Driver pay is intentionally tied to each firm's current rider-facing price so
supply responds directly to pricing policy rather than a separate rate card.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import importlib
import importlib.util

import numpy as np


@dataclass
class DriverPayPolicy:
    """Simple driver compensation policy linked to the firm's current fare."""

    fare_share: float = 0.72
    minimum_pay: float = 3.50
    incentive_multiplier: float = 1.0

@dataclass
class DriverSupplyConfig:
    """Configuration for driver supply and dispatch acceptance.

    acceptance_mode controls how dispatch acceptance is resolved after the
    earnings-based acceptance probability is calculated:
    - expected: deterministic thresholding for lower-variance training runs.
    - stochastic: Bernoulli sampling for simulation runs that need randomness.
    """
    
    base_active_drivers: int = 260
    reservation_wage_per_hour: float = 24.0
    operating_cost_per_mile: float = 0.35
    acceptance_ratio_threshold: float = 1.0
    max_pickup_minutes: float = 16.0
    base_pickup_minutes: float = 3.5
    osmnx_pickup_minutes_factor: float = 1.0
    supply_elasticity: float = 0.20
    online_temperature: float = 18.0
    platform_variable_cost: float = 0.45
    state_smoothing_alpha: float = 0.35
    pickup_noise_sigma: float = 0.03
    
    acceptance_mode: str = "expected"
    expected_acceptance_cutoff: float = 0.55

    def __post_init__(self) -> None:
        self.acceptance_mode = str(self.acceptance_mode).lower().strip()
        if self.acceptance_mode not in {"expected", "stochastic"}:
            raise ValueError(
                "DriverSupplyConfig.acceptance_mode must be either "
                "'expected' or 'stochastic'"
            )
        self.expected_acceptance_cutoff = float(np.clip(self.expected_acceptance_cutoff, 0.0, 1.0))

@dataclass
class FirmDriverBatchState:
    active_drivers: int = 0
    idle_drivers: int = 0
    busy_drivers: int = 0
    offered_dispatches: int = 0
    accepted_dispatches: int = 0
    rejected_dispatches: int = 0
    completed_rides: int = 0
    unfulfilled_requests: int = 0
    total_driver_pay: float = 0.0
    total_pickup_minutes: float = 0.0
    total_pickup_miles: float = 0.0
    total_wait_minutes: float = 0.0
    total_online_minutes: float = 0.0
    total_engaged_minutes: float = 0.0
    total_acceptance_probability: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        if self.offered_dispatches <= 0:
            return 1.0
        if self.total_acceptance_probability > 0.0:
            return self.total_acceptance_probability / self.offered_dispatches
        return self.accepted_dispatches / self.offered_dispatches

    @property
    def fulfillment_rate(self) -> float:
        denom = self.completed_rides + self.unfulfilled_requests
        return self.completed_rides / denom if denom > 0 else 1.0

    @property
    def avg_wait_minutes(self) -> float:
        return self.total_wait_minutes / self.completed_rides if self.completed_rides > 0 else 0.0

    @property
    def avg_pickup_minutes(self) -> float:
        return self.total_pickup_minutes / self.completed_rides if self.completed_rides > 0 else 0.0

    @property
    def driver_earnings_per_hour(self) -> float:
        hours = self.total_engaged_minutes / 60.0
        return self.total_driver_pay / hours if hours > 0 else 0.0

    @property
    def idle_driver_share(self) -> float:
        return self.idle_drivers / self.active_drivers if self.active_drivers > 0 else 0.0

    @property
    def utilization(self) -> float:
        return self.busy_drivers / self.active_drivers if self.active_drivers > 0 else 0.0


@dataclass
class DispatchResult:
    fulfilled: bool
    driver_pay: float = 0.0
    platform_variable_cost: float = 0.0
    pickup_minutes: float = 0.0
    pickup_miles: float = 0.0
    wait_minutes: float = 0.0
    acceptance_probability: float = 0.0
    reject_reason: str = ""


class OpenStreetMapRouter:
    """Optional OSMnx-backed road-network hook with deterministic fallback.

    This class intentionally avoids importing OSMnx at module import time. When a
    graph is requested and OSMnx is installed, it is imported dynamically. This
    keeps normal experiments runnable in environments without OSMnx while leaving
    a clear integration point for city-network routing.
    """

    def __init__(self, place_name: str, network_type: str = "drive", use_osmnx: bool = False):
        self.place_name = str(place_name)
        self.network_type = str(network_type)
        self.use_osmnx = bool(use_osmnx)
        self.graph = None
        self.network_pickup_factor = 1.0
        if self.use_osmnx and importlib.util.find_spec("osmnx") is not None:
            ox = importlib.import_module("osmnx")
            self.graph = ox.graph_from_place(self.place_name, network_type=self.network_type)
            self.graph = ox.add_edge_speeds(self.graph)
            self.graph = ox.add_edge_travel_times(self.graph)
            self.network_pickup_factor = self._estimate_pickup_factor()

    def _estimate_pickup_factor(self) -> float:
        """Estimate a stable pickup-time factor from the OSMnx drive graph.

        The simulator does not route every synthetic pickup because that would be
        too slow for PPO rollouts.  Instead, OSMnx supplies a city-specific
        network friction scalar derived from edge speed and circuity.
        """
        if self.graph is None:
            return 1.0
        speeds = []
        circuities = []
        for _, _, data in self.graph.edges(data=True):
            length = float(data.get("length", 0.0) or 0.0)
            speed = float(data.get("speed_kph", 0.0) or 0.0)
            travel_time = float(data.get("travel_time", 0.0) or 0.0)
            if speed > 0.0:
                speeds.append(speed)
            if length > 0.0 and travel_time > 0.0 and speed > 0.0:
                implied = (speed * 1000.0 / 3600.0) * travel_time
                circuities.append(implied / max(length, 1e-6))
        median_speed = float(np.median(speeds)) if speeds else 22.0
        median_circuity = float(np.median(circuities)) if circuities else 1.15
        speed_factor = 24.0 / max(8.0, median_speed)
        return float(np.clip(speed_factor * median_circuity, 0.75, 1.75))

    @property
    def enabled(self) -> bool:
        return self.graph is not None


class DriverSupplyLayer:
    """Batch-level two-sided driver supply and matching simulator."""

    def __init__(
        self,
        seed: Optional[int] = None,
        config: Optional[DriverSupplyConfig] = None,
        firm1_pay_policy: Optional[DriverPayPolicy] = None,
        firm2_pay_policy: Optional[DriverPayPolicy] = None,
        use_osmnx: bool = False,
        osmnx_place: str = "New York City, New York, USA",
    ):
        self.rng = np.random.default_rng(seed)
        self.config = config or DriverSupplyConfig()
        self.pay_policies = {
            "Firm1": firm1_pay_policy or DriverPayPolicy(),
            "Firm2": firm2_pay_policy or DriverPayPolicy(fare_share=0.70),
        }
        self.router = OpenStreetMapRouter(place_name=osmnx_place, use_osmnx=use_osmnx)
        self.last_states = {"Firm1": FirmDriverBatchState(), "Firm2": FirmDriverBatchState()}
        self.smoothed_states = {"Firm1": FirmDriverBatchState(), "Firm2": FirmDriverBatchState()}
        self._batch_states = {"Firm1": FirmDriverBatchState(), "Firm2": FirmDriverBatchState()}

    def begin_batch(self, customers_per_step: int, hour: int, weather: str) -> None:
        self._batch_states = {}
        demand_scale = max(0.25, float(customers_per_step) / 500.0)
        peak = 1.18 if (7 <= int(hour) < 10 or 16 <= int(hour) < 19) else 1.0
        weather_supply = 0.88 if str(weather).lower() in {"rain", "snow", "storm"} else 1.0
        for firm, bias in (("Firm1", 1.0), ("Firm2", 0.96)):
            previous = self.smoothed_states.get(firm, self.last_states[firm])
            previous_earnings = previous.driver_earnings_per_hour or self.config.reservation_wage_per_hour
            online_response = 1.0 + self.config.supply_elasticity * np.tanh(
                (previous_earnings - self.config.reservation_wage_per_hour) / max(1e-6, self.config.online_temperature)
            )
            active = int(max(1, round(self.config.base_active_drivers * demand_scale * peak * weather_supply * bias * online_response)))
            busy = int(np.clip(round(active * (0.18 if peak > 1.0 else 0.12)), 0, max(0, active - 1)))
            idle = max(1, active - busy)
            state = FirmDriverBatchState(active_drivers=active, idle_drivers=idle, busy_drivers=busy)
            state.total_online_minutes = float(active * 60.0)
            self._batch_states[firm] = state

    def estimate_pickup_minutes(self, firm: str, airport: bool = False) -> float:
        state = self._batch_states.get(firm, FirmDriverBatchState(active_drivers=1, idle_drivers=1))
        supply_ratio = state.idle_drivers / max(1.0, float(state.active_drivers))
        scarcity = 1.0 / max(0.08, supply_ratio)
        airport_factor = 1.18 if airport else 1.0
        noise = float(self.rng.lognormal(mean=0.0, sigma=max(0.0, self.config.pickup_noise_sigma)))
        network_factor = float(getattr(self.router, "network_pickup_factor", 1.0))
        config_factor = float(np.clip(self.config.osmnx_pickup_minutes_factor, 0.5, 2.0))
        return float(np.clip(self.config.base_pickup_minutes * scarcity * airport_factor * network_factor * config_factor * noise, 1.0, 25.0))

    def estimate_service_quality(self, firm: str, airport: bool = False) -> Dict[str, float]:
        pickup = self.estimate_pickup_minutes(firm, airport=airport)
        cancel_risk = float(np.clip((pickup - 6.0) / 18.0, 0.0, 0.95))
        return {"pickup_minutes": pickup, "cancel_risk": cancel_risk}

    def quote_driver_pay(
        self,
        firm: str,
        rider_fare: float,
        distance_miles: float,
        duration_minutes: float,
        pickup_minutes: float,
        airport: bool,
    ) -> float:
        del distance_miles, duration_minutes, pickup_minutes, airport
        policy = self.pay_policies[firm]
        base_pay = max(policy.minimum_pay, policy.fare_share * max(0.0, float(rider_fare)))
        return float(max(policy.minimum_pay, base_pay * float(np.clip(policy.incentive_multiplier, 0.85, 1.25))))

    def driver_acceptance_ratio(
        self,
        driver_pay: float,
        distance_miles: float,
        duration_minutes: float,
        pickup_minutes: float,
        weather: str = "clear",
    ) -> float:
        """Return net earnings divided by required earnings for the ride context."""
        weather_penalty = {"clear": 1.0, "cloudy": 1.03, "rain": 1.18, "snow": 1.32, "storm": 1.40}.get(str(weather).lower(), 1.0)
        miles = max(0.0, float(distance_miles))
        pickup = max(0.0, float(pickup_minutes))
        minutes = max(0.0, float(duration_minutes)) + pickup
        operating_cost = self.config.operating_cost_per_mile * (miles + max(0.1, pickup * 0.28))
        net_earnings = max(0.0, float(driver_pay) - operating_cost)
        required_earnings = self.config.reservation_wage_per_hour * (minutes / 60.0) * weather_penalty
        return float(net_earnings / max(1e-6, required_earnings))

    def driver_acceptance_probability(
        self,
        driver_pay: float,
        distance_miles: float,
        duration_minutes: float,
        pickup_minutes: float,
        weather: str = "clear",
    ) -> float:
        """Map the earnings/ride-nature ratio directly to willingness to accept."""
        ratio = self.driver_acceptance_ratio(
            driver_pay=driver_pay,
            distance_miles=distance_miles,
            duration_minutes=duration_minutes,
            pickup_minutes=pickup_minutes,
            weather=weather,
        )
        return float(np.clip(ratio / max(1e-6, self.config.acceptance_ratio_threshold), 0.0, 1.0))
    
    def _resolve_acceptance(self, feasible: bool, acceptance_probability: float) -> bool:
        """Resolve whether a feasible dispatch is accepted under the configured mode."""
        if not feasible:
            return False
        probability = float(np.clip(acceptance_probability, 0.0, 1.0))
        if self.config.acceptance_mode == "stochastic":
            return bool(self.rng.random() < probability)
        # Expected mode is deterministic for lower-variance PPO training, but use
        # a soft acceptability cutoff instead of requiring probability saturation.
        # This keeps fulfillment from becoming an all-or-nothing cliff around the
        # earnings threshold.
        return bool(probability >= float(self.config.expected_acceptance_cutoff))

    def dispatch(
        self,
        firm: str,
        rider_fare: float,
        distance_miles: float,
        duration_minutes: float,
        airport: bool = False,
        weather: str = "clear",
    ) -> DispatchResult:
        state = self._batch_states.setdefault(firm, FirmDriverBatchState(active_drivers=1, idle_drivers=1))
        state.offered_dispatches += 1
        pickup_minutes = self.estimate_pickup_minutes(firm, airport=airport)
        pickup_miles = max(0.1, pickup_minutes * 0.28)
        driver_pay = self.quote_driver_pay(firm, rider_fare, distance_miles, duration_minutes, pickup_minutes, airport)
        accept_prob = self.driver_acceptance_probability(
            driver_pay=driver_pay,
            distance_miles=distance_miles,
            duration_minutes=duration_minutes,
            pickup_minutes=pickup_minutes,
            weather=weather,
        )
        feasible = state.idle_drivers > 0 and pickup_minutes <= self.config.max_pickup_minutes
        state.total_acceptance_probability += accept_prob if feasible else 0.0
        accepted = self._resolve_acceptance(feasible=feasible, acceptance_probability=accept_prob)
        if not accepted:
            state.rejected_dispatches += 1
            state.unfulfilled_requests += 1
            reason = "NO_IDLE_DRIVER" if state.idle_drivers <= 0 else "LOW_EARNINGS_RATIO_OR_LONG_PICKUP"
            return DispatchResult(
                fulfilled=False,
                driver_pay=0.0,
                pickup_minutes=pickup_minutes,
                pickup_miles=pickup_miles,
                wait_minutes=pickup_minutes,
                acceptance_probability=accept_prob,
                reject_reason=reason,
            )

        state.accepted_dispatches += 1
        state.completed_rides += 1
        state.idle_drivers = max(0, state.idle_drivers - 1)
        state.busy_drivers += 1
        state.total_driver_pay += driver_pay
        state.total_pickup_minutes += pickup_minutes
        state.total_pickup_miles += pickup_miles
        state.total_engaged_minutes += pickup_minutes + max(0.0, float(duration_minutes))
        state.total_wait_minutes += pickup_minutes
        platform_cost = self.config.platform_variable_cost + 0.05 * max(0.0, float(duration_minutes))
        return DispatchResult(
            fulfilled=True,
            driver_pay=driver_pay,
            platform_variable_cost=float(platform_cost),
            pickup_minutes=pickup_minutes,
            pickup_miles=pickup_miles,
            wait_minutes=pickup_minutes,
            acceptance_probability=accept_prob,
        )

    def end_batch(self) -> Dict[str, FirmDriverBatchState]:
        self.last_states = {firm: state for firm, state in self._batch_states.items()}
        alpha = float(np.clip(self.config.state_smoothing_alpha, 0.0, 1.0))
        for firm, state in self.last_states.items():
            prev = self.smoothed_states.get(firm, FirmDriverBatchState())
            self.smoothed_states[firm] = FirmDriverBatchState(
                active_drivers=int(round((1.0 - alpha) * prev.active_drivers + alpha * state.active_drivers)),
                idle_drivers=int(round((1.0 - alpha) * prev.idle_drivers + alpha * state.idle_drivers)),
                busy_drivers=int(round((1.0 - alpha) * prev.busy_drivers + alpha * state.busy_drivers)),
                offered_dispatches=int(round((1.0 - alpha) * prev.offered_dispatches + alpha * state.offered_dispatches)),
                accepted_dispatches=int(round((1.0 - alpha) * prev.accepted_dispatches + alpha * state.accepted_dispatches)),
                rejected_dispatches=int(round((1.0 - alpha) * prev.rejected_dispatches + alpha * state.rejected_dispatches)),
                completed_rides=int(round((1.0 - alpha) * prev.completed_rides + alpha * state.completed_rides)),
                unfulfilled_requests=int(round((1.0 - alpha) * prev.unfulfilled_requests + alpha * state.unfulfilled_requests)),
                total_driver_pay=(1.0 - alpha) * prev.total_driver_pay + alpha * state.total_driver_pay,
                total_pickup_minutes=(1.0 - alpha) * prev.total_pickup_minutes + alpha * state.total_pickup_minutes,
                total_pickup_miles=(1.0 - alpha) * prev.total_pickup_miles + alpha * state.total_pickup_miles,
                total_wait_minutes=(1.0 - alpha) * prev.total_wait_minutes + alpha * state.total_wait_minutes,
                total_online_minutes=(1.0 - alpha) * prev.total_online_minutes + alpha * state.total_online_minutes,
                total_engaged_minutes=(1.0 - alpha) * prev.total_engaged_minutes + alpha * state.total_engaged_minutes,
                total_acceptance_probability=(
                    (1.0 - alpha) * prev.total_acceptance_probability
                    + alpha * state.total_acceptance_probability
                ),
            )
        return self.last_states

    def state_features_for_firm1(self) -> np.ndarray:
        f1 = self.smoothed_states.get("Firm1", self.last_states.get("Firm1", FirmDriverBatchState()))
        f2 = self.smoothed_states.get("Firm2", self.last_states.get("Firm2", FirmDriverBatchState()))
        features = [
            np.clip(f1.active_drivers / 800.0, 0.0, 1.0),
            np.clip(f1.idle_driver_share, 0.0, 1.0),
            np.clip(f1.utilization, 0.0, 1.0),
            np.clip(f1.acceptance_rate, 0.0, 1.0),
            np.clip(f1.fulfillment_rate, 0.0, 1.0),
            np.clip(f1.avg_wait_minutes / 20.0, 0.0, 1.0),
            np.clip(f1.driver_earnings_per_hour / 60.0, 0.0, 1.0),
            np.clip((f2.avg_wait_minutes - f1.avg_wait_minutes + 10.0) / 20.0, 0.0, 1.0),
        ]
        return np.asarray(features, dtype=np.float32)