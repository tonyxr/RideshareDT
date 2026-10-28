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

import mkl_config  # noqa: F401 - set oneMKL env before NumPy/Torch
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
    platform_cost_per_minute: float = 0.05
    state_smoothing_alpha: float = 0.35
    pickup_noise_sigma: float = 0.03
    
    acceptance_mode: str = "expected"
    expected_acceptance_cutoff: float = 0.55
    # v26 environment contract.  Legacy defaults reproduce v25 exactly.
    firm2_supply_bias: float = 0.96
    linear_demand_scaling: bool = False
    reference_customers: float = 500.0
    legacy_peak_factor: bool = True
    # v26r regime market.  ``persistent_stock`` replaces the per-batch
    # ``online_response`` by a daily partial-adjustment stock of online drivers
    # (the persistent state that makes anticipatory incentives pay).  Off by
    # default, so the v26 supply layer is unchanged.
    persistent_stock: bool = False
    stock_adjustment_rate: float = 0.30     # share of the gap to target closed per day
    stock_floor: float = 0.40               # minimum stock relative to base fleet
    incentive_earnings_weight: float = 1.0  # incentive multiplier -> perceived earnings
    # v26r supply model v2: constant-elasticity participation in pay and a
    # calibrated hour-of-day availability profile (more drivers by day).
    #   S_{d+1} = (1-kappa) S_d + kappa (E_d / E_ref)^eta_day
    #   A(h)    = N * S_d * phi(h) * weather * posted_pay_index^eta_hour
    # Realized earnings move the daily stock, not the posted-pay response too.
    supply_model: str = "v1"                # "v1" (v26 behaviour) | "v2" | V17 "v3"
    eta_day: float = 0.40                   # daily labour-supply elasticity (Chen et al. 2019: 0.3-0.6)
    eta_hour: float = 0.20                  # within-day online response to current pay
    eta_v3: float = 0.60                    # V17 response to expected earnings per online hour
    shortage_willingness_share: float = 0.0 # V17 M2 omega; default off
    reference_earnings_per_hour: float = 0.0  # 0 selects the fixed opportunity wage, never policy earnings
    availability_block_hours: float = 1.0
    driver_minute_capacity: bool = True  # used only by supply v2
    initial_online_stock: Optional[Dict[str, float]] = None
    hour_profile: Optional[Dict[int, float]] = None  # phi(h); None = DEFAULT_HOUR_PROFILE
    stock_ceiling: float = 2.0

    def __post_init__(self) -> None:
        self.supply_model = str(self.supply_model).lower().strip()
        if self.supply_model not in {"v1", "v2", "v3"}:
            raise ValueError("supply_model must be v1, v2, or v3")
        if not np.isfinite(self.eta_v3) or self.eta_v3 < 0:
            raise ValueError("eta_v3 must be finite and nonnegative")
        if (not np.isfinite(self.shortage_willingness_share)
                or not 0.0 <= self.shortage_willingness_share <= 1.0):
            raise ValueError("shortage_willingness_share must lie in [0, 1]")
        self.acceptance_mode = str(self.acceptance_mode).lower().strip()
        if self.acceptance_mode not in {"expected", "stochastic"}:
            raise ValueError("acceptance_mode must be 'expected' or 'stochastic'")
        if not 0 <= self.expected_acceptance_cutoff <= 1:
            raise ValueError("expected_acceptance_cutoff must lie in [0, 1]")


# Stylised active-vehicle profile by hour (mean 1), shaped on the NYC TLC
# active-vehicle curve: a 3-4 am trough near a third of the mean, a morning
# ramp, and an evening peak around 17:00.  A city calibration may override it.
DEFAULT_HOUR_PROFILE: Dict[int, float] = {
    0: 0.55, 1: 0.45, 2: 0.38, 3: 0.35, 4: 0.40, 5: 0.55, 6: 0.80, 7: 1.05,
    8: 1.20, 9: 1.15, 10: 1.05, 11: 1.05, 12: 1.10, 13: 1.10, 14: 1.15, 15: 1.25,
    16: 1.35, 17: 1.40, 18: 1.35, 19: 1.25, 20: 1.10, 21: 1.00, 22: 0.85, 23: 0.70,
}


def normalized_hour_profile(profile: Optional[Dict[int, float]] = None) -> Dict[int, float]:
    raw = {int(h): float(v) for h, v in (profile or DEFAULT_HOUR_PROFILE).items()}
    values = np.array([raw.get(h, 1.0) for h in range(24)], dtype=float)
    values = np.maximum(values, 1e-3)
    values = values / float(values.mean())
    return {h: float(values[h]) for h in range(24)}

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
    def driver_earnings_per_online_hour(self) -> float:
        """Pay per hour online (utilisation x pay): what participation responds to."""
        hours = self.total_online_minutes / 60.0
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
        # v26r: persistent stock (relative to the base fleet) and the active
        # regime multipliers (neutral = 1.0).  ``begin_batch`` reads them.
        self.online_stock: Dict[str, float] = {"Firm1": 1.0, "Firm2": 1.0}
        # v2 state: pay index posted this block (relative to its smoothed level),
        # the learned reference earnings and the warm-up counter.
        self.pay_index: Dict[str, float] = {"Firm1": 1.0, "Firm2": 1.0}
        self.pay_index_smoothed: Dict[str, float] = {"Firm1": 1.0, "Firm2": 1.0}
        self.reference_earnings: Dict[str, float] = {"Firm1": 0.0, "Firm2": 0.0}
        self.hour_profile: Dict[int, float] = normalized_hour_profile(self.config.hour_profile)
        self.shock_fleet_multiplier: float = 1.0
        self.shock_wage_multiplier: float = 1.0
        self.raw_shock_fleet_multiplier: float = 1.0
        self.raw_shock_wage_multiplier: float = 1.0
        self.shortage_willingness_multiplier: float = 1.0
        self.shock_firm_capacity: Dict[str, float] = {"Firm1": 1.0, "Firm2": 1.0}
        self._available_at = {f: np.empty(0) for f in self.pay_policies}
        self._batch_minute = 0.0
        self._daily_pay = {f: 0.0 for f in self.pay_policies}
        self._daily_online = {f: 0.0 for f in self.pay_policies}
        self._observed_blocks = 0
        if self.supply_v2 or self.supply_v3:
            self.reset_stock()

    # ---------- v26r regime market ----------
    def reset_stock(self) -> None:
        self.online_stock = {f: float((self.config.initial_online_stock or {}).get(f, 1.0))
                             for f in ("Firm1", "Firm2")}
        self.pay_index = {"Firm1": 1.0, "Firm2": 1.0}
        self.pay_index_smoothed = {"Firm1": 1.0, "Firm2": 1.0}
        reference = float(self.config.reference_earnings_per_hour)
        if (self.supply_v2 or self.supply_v3) and reference == 0:
            reference = float(self.config.reservation_wage_per_hour)
        self.reference_earnings = {f: reference for f in ("Firm1", "Firm2")}
        self._available_at = {f: np.empty(0) for f in self.pay_policies}
        self._batch_minute = 0.0
        self._daily_pay = {f: 0.0 for f in self.pay_policies}
        self._daily_online = {f: 0.0 for f in self.pay_policies}
        if self.supply_v2 or self.supply_v3:
            self._observed_blocks = 0
            # New episodes cannot inherit the previous policy's pay history.
            self.last_states = {f: FirmDriverBatchState() for f in ("Firm1", "Firm2")}
            self.smoothed_states = {f: FirmDriverBatchState() for f in ("Firm1", "Firm2")}
            self._batch_states = {f: FirmDriverBatchState() for f in ("Firm1", "Firm2")}

    @property
    def supply_v2(self) -> bool:
        return str(self.config.supply_model).lower() == "v2"

    @property
    def supply_v3(self) -> bool:
        return str(self.config.supply_model).lower() == "v3"

    def set_pay_index(self, firm: str, index: float) -> None:
        """Posted pay level this block relative to the anchor (fare multiplier x incentive)."""
        if self.supply_v2 or self.supply_v3:
            if not np.isfinite(index) or index < 0:
                raise ValueError('posted pay index must be finite and nonnegative')
            self.pay_index[firm] = float(index)
        else:
            self.pay_index[firm] = float(max(0.05, index))

    def hour_availability(self, hour: int) -> float:
        width = float(self.config.availability_block_hours)
        if not 0 < width <= 24:
            raise ValueError('availability block duration must be in (0, 24] hours')
        # Match OperationalCalendar's rounded hour bins, including cadences
        # that do not divide 24. Availability itself integrates the exact
        # fractional block duration rather than truncating its hours.
        steps = int(round(24. / width))
        edges = np.round(np.linspace(0., 24., steps + 1))
        block = int(np.searchsorted(edges, float(hour) % 24., side='right') - 1)
        start = block * width
        stop = start + width
        return float(sum(self.hour_profile[h % 24] *
                         max(0., min(stop, h+1) - max(start, h))
                         for h in range(int(np.floor(start)), int(np.ceil(stop)))) / width)

    def _earnings_reference(self, firm: str) -> float:
        ref = float(self.reference_earnings.get(firm, 0.0))
        if ref > 0.0:
            return ref
        if not (self.supply_v2 or self.supply_v3):
            previous = self.smoothed_states.get(firm, self.last_states.get(firm, FirmDriverBatchState()))
            return float(previous.driver_earnings_per_online_hour or self.config.reservation_wage_per_hour)
        raise ValueError('supply v2/v3 requires positive fixed reference earnings')

    def expected_block_earnings(self, firm: str) -> float:
        """Posted earning opportunity for prospective drivers (online-hour units).

        v2 uses posted pay for the within-day response; realized utilization
        already enters the daily workforce adjustment. Multiplying both by
        realized earnings counted that signal twice and made one empty block
        an absorbing zero. This is an offer response, not a realized-pay floor.
        """
        if self.supply_v2:
            return self._earnings_reference(firm) * float(self.pay_index[firm])
        previous = self.smoothed_states.get(firm, self.last_states.get(firm, FirmDriverBatchState()))
        smoothed = (float(previous.driver_earnings_per_online_hour)
                    if previous.total_online_minutes > 0 else self._earnings_reference(firm))
        ratio = float(self.pay_index.get(firm, 1.0)) / max(1e-6, float(self.pay_index_smoothed.get(firm, 1.0)))
        return smoothed * ratio

    def supply_v3_expected_earnings(self, firm: str) -> Tuple[float, float, float]:
        """Return pay/trip, trips/hour, and their product for V17 supply v3.

        The most recent public block fixes utilisation (trips per online hour).
        The posted pay index moves expected pay per trip relative to that block.
        With no exposed block, the fixed city earnings reference is the neutral
        opportunity.  This makes the registered M1 mechanism explicit and
        auditable instead of hiding it inside an earnings-per-hour shortcut.
        """
        if not self.supply_v3:
            raise ValueError("supply_v3_expected_earnings requires supply_model='v3'")
        previous = self.smoothed_states.get(
            firm, self.last_states.get(firm, FirmDriverBatchState())
        )
        online_hours = float(previous.total_online_minutes) / 60.0
        if online_hours > 0.0 and previous.completed_rides > 0:
            trips_per_hour = float(previous.completed_rides) / online_hours
            observed_pay_per_trip = (
                float(previous.total_driver_pay) / float(previous.completed_rides)
            )
            pay_ratio = float(self.pay_index.get(firm, 1.0)) / max(
                1e-12, float(self.pay_index_smoothed.get(firm, 1.0))
            )
            pay_per_trip = observed_pay_per_trip * pay_ratio
        else:
            trips_per_hour = 1.0
            pay_per_trip = self._earnings_reference(firm) * float(
                self.pay_index.get(firm, 1.0)
            )
        expected = pay_per_trip * trips_per_hour
        return float(pay_per_trip), float(trips_per_hour), float(expected)

    def supply_v3_online_response(self, firm: str) -> float:
        """Continuous V17 online-driver response before fleet realization."""
        _, _, expected = self.supply_v3_expected_earnings(firm)
        wage_reference = self._earnings_reference(firm) * float(
            self.shock_wage_multiplier
        )
        if not np.isfinite(expected) or not np.isfinite(wage_reference) or wage_reference <= 0:
            raise ValueError("supply v3 requires finite earnings and a positive reference")
        return float(max(0.0, expected / wage_reference) ** float(self.config.eta_v3))

    def apply_shock(self, *, fleet_multiplier: float = 1.0, wage_multiplier: float = 1.0,
                    firm_capacity: Optional[Dict[str, float]] = None) -> None:
        """Install the current block's regime multipliers (from ``ShockEffect``)."""
        if self.supply_v2 or self.supply_v3:
            values = [fleet_multiplier, wage_multiplier, *(firm_capacity or {}).values()]
            if any(not np.isfinite(v) or v < 0 for v in values) or wage_multiplier == 0:
                raise ValueError('supply shocks require nonnegative capacity and positive wage multipliers')
            self.raw_shock_fleet_multiplier = float(fleet_multiplier)
            self.raw_shock_wage_multiplier = float(wage_multiplier)
            hard_fleet = float(fleet_multiplier)
            willingness = 1.0
            if self.supply_v3 and 0.0 < fleet_multiplier < 1.0:
                omega = float(self.config.shortage_willingness_share)
                if omega > 0.0:
                    if self.config.eta_v3 <= 0.0:
                        raise ValueError("positive shortage willingness requires eta_v3 > 0")
                    # Split an original shortage m into a hard capacity part
                    # m^(1-omega) and a reservation-wage part.  Because v3
                    # participation is proportional to wage^(-eta), choosing
                    # the latter as m^(-omega/eta) keeps total card-price
                    # severity exactly m while making omega recoverable by pay.
                    hard_fleet = float(fleet_multiplier) ** (1.0 - omega)
                    willingness = float(fleet_multiplier) ** (
                        -omega / float(self.config.eta_v3)
                    )
            self.shock_fleet_multiplier = hard_fleet
            self.shortage_willingness_multiplier = willingness
            self.shock_wage_multiplier = float(wage_multiplier) * willingness
            self.shock_firm_capacity = {f: float((firm_capacity or {}).get(f, 1.)) for f in ('Firm1', 'Firm2')}
            return
        self.shock_fleet_multiplier = float(max(0.05, fleet_multiplier))
        self.shock_wage_multiplier = float(max(0.05, wage_multiplier))
        self.shock_firm_capacity = {
            "Firm1": float(max(0.05, (firm_capacity or {}).get("Firm1", 1.0))),
            "Firm2": float(max(0.05, (firm_capacity or {}).get("Firm2", 1.0))),
        }

    def stock_target(self, firm: str) -> float:
        """Earnings-driven target stock for ``firm`` under the current regime."""
        cfg = self.config
        previous = self.smoothed_states.get(firm, self.last_states.get(firm, FirmDriverBatchState()))
        earnings = float(previous.driver_earnings_per_hour or cfg.reservation_wage_per_hour)
        incentive = float(np.clip(self.pay_policies[firm].incentive_multiplier, 0.85, 1.25))
        perceived = earnings * (1.0 + float(cfg.incentive_earnings_weight) * (incentive - 1.0))
        wage = float(cfg.reservation_wage_per_hour) * float(self.shock_wage_multiplier)
        target = 1.0 + float(cfg.supply_elasticity) * float(np.tanh(
            (perceived - wage) / max(1e-6, float(cfg.online_temperature))
        ))
        return float(target)

    def advance_day(self) -> Dict[str, float]:
        """Daily partial adjustment of each firm's online stock toward target.

        Called once per simulated day by Core (after the last block).  The
        regime fleet multiplier enters the target (a shortage lowers it) so a
        lost stock recovers only over the following days.
        """
        if not self.config.persistent_stock:
            return dict(self.online_stock)
        if self.supply_v2 and self._observed_blocks == 0 and not any(self._daily_online.values()):
            return dict(self.online_stock)  # Opening, not a zero-exposure day.
        kappa = float(np.clip(self.config.stock_adjustment_rate, 0.0, 1.0))
        floor = float(max(0.0, self.config.stock_floor))
        ceiling = float(max(floor, self.config.stock_ceiling))
        for firm in ("Firm1", "Firm2"):
            if self.supply_v2:
                # Actual daily gross online earnings already include incentives.
                # Zero rides WITH exposure are zero earnings. Without exposure,
                # entrants respond to the posted opportunity, not an undefined
                # 0/0 rate. Thus an outage does not remove the future workforce.
                perceived = (60. * self._daily_pay[firm] / self._daily_online[firm]
                             if self._daily_online[firm] > 0 else self.expected_block_earnings(firm))
                wage_scale = float(self.shock_wage_multiplier)
                ratio = perceived / (float(self.reference_earnings[firm]) * wage_scale)
                target = max(0., ratio) ** float(self.config.eta_day)
                # Platform outages affect dispatch availability in begin_batch,
                # not the persistent workforce a second time.
                target *= float(self.shock_fleet_multiplier)
            else:
                target = self.stock_target(firm) * float(self.shock_fleet_multiplier) \
                    * float(self.shock_firm_capacity.get(firm, 1.0))
            updated = (1.0 - kappa) * float(self.online_stock[firm]) + kappa * target
            self.online_stock[firm] = float(updated if self.supply_v2 else np.clip(updated, floor, ceiling))
            self._daily_pay[firm] = self._daily_online[firm] = 0.0
            if not self.supply_v2:
                self.pay_index_smoothed[firm] = float(
                    (1.0 - kappa) * self.pay_index_smoothed.get(firm, 1.0) + kappa * self.pay_index.get(firm, 1.0))
        self._observed_blocks = 0
        return dict(self.online_stock)

    def begin_batch(
        self,
        customers_per_step: int,
        hour: int,
        weather: str,
        supply_multiplier: float = 1.0,
    ) -> None:
        """Start one market step.

        ``customers_per_step`` sets fleet capacity.  Legacy mode floors the
        capacity scale at 0.25, which made the request/driver ratio differ
        between 100-customer training and 200-customer evaluation; the v26
        ``linear_demand_scaling`` keeps that ratio invariant.  In v26 callers
        pass the run's base customers here (capacity) and an hourly
        ``supply_multiplier`` that tracks only part of the demand cycle, while
        realized requests vary with the calibrated hour-of-day volume.
        """
        self._batch_states = {}
        if self.config.linear_demand_scaling:
            demand_scale = max(1e-3, float(customers_per_step) / float(self.config.reference_customers))
        else:
            demand_scale = max(0.25, float(customers_per_step) / 500.0)
        rush = 7 <= int(hour) < 10 or 16 <= int(hour) < 19
        peak = 1.18 if (rush and self.config.legacy_peak_factor) else 1.0
        busy_peak = rush
        weather_supply = 0.88 if str(weather).lower() in {"rain", "snow", "storm"} else 1.0
        hourly = float(max(0.05, supply_multiplier))
        for firm, bias in (("Firm1", 1.0), ("Firm2", float(self.config.firm2_supply_bias))):
            previous = self.smoothed_states.get(firm, self.last_states[firm])
            previous_earnings = previous.driver_earnings_per_hour or self.config.reservation_wage_per_hour
            if self.supply_v2:
                # v26r supply v2: calibrated hour profile x daily stock x the
                # within-day online response to the pay posted this block.
                ref = self._earnings_reference(firm)
                hour_response = max(0., self.expected_block_earnings(firm) / ref) \
                    ** float(self.config.eta_hour)
                online_response = float(self.online_stock.get(firm, 1.0)) * float(
                    self.shock_firm_capacity.get(firm, 1.0)
                ) * hour_response
                peak = 1.0
                hourly = self.hour_availability(int(hour))
            elif self.supply_v3:
                # V17 M1: participation responds to expected earnings, written
                # explicitly as pay per trip times trips per online hour.
                online_response = self.supply_v3_online_response(firm) * float(
                    self.shock_fleet_multiplier
                ) * float(self.shock_firm_capacity.get(firm, 1.0))
                peak = 1.0
                hourly = self.hour_availability(int(hour))
            elif self.config.persistent_stock:
                # v26r: the daily stock carries the lagged supply response; the
                # block-level regime capacity cut (an outage) applies at once.
                online_response = float(self.online_stock.get(firm, 1.0)) * float(
                    self.shock_firm_capacity.get(firm, 1.0)
                )
            else:
                online_response = 1.0 + self.config.supply_elasticity * np.tanh(
                    (previous_earnings - self.config.reservation_wage_per_hour) / max(1e-6, self.config.online_temperature)
                )
                if self.shock_fleet_multiplier != 1.0 or self.shock_firm_capacity.get(firm, 1.0) != 1.0:
                    online_response *= float(self.shock_fleet_multiplier) * float(self.shock_firm_capacity.get(firm, 1.0))
            expected_active = self.config.base_active_drivers * demand_scale * peak * hourly * weather_supply * bias * online_response
            if self.supply_v2 or self.supply_v3:
                # Unbiased integer realization, including fleets below one.
                # No minimum driver and no deterministic rounding dead zone.
                whole = int(np.floor(expected_active))
                active = whole + int(self.rng.random() < expected_active - whole)
                self._feature_capacity_reference = self.config.base_active_drivers * demand_scale
            else:
                active = int(max(1, round(expected_active)))
            busy = int(np.clip(round(active * (0.18 if busy_peak else 0.12)), 0, max(0, active - 1)))
            idle = max(1, active - busy)
            state = FirmDriverBatchState(active_drivers=active, idle_drivers=idle, busy_drivers=busy)
            state.total_online_minutes = float(active * 60.0)
            if (self.supply_v2 or self.supply_v3) and self.config.driver_minute_capacity:
                duration = 60. * self.config.availability_block_hours
                # Trips crossing a block boundary keep their remaining time.
                remaining = np.maximum(0., self._available_at[firm] - duration)
                remaining = np.sort(remaining)[-active:] if active else np.empty(0)
                self._available_at[firm] = np.pad(remaining, (active-len(remaining), 0))
                state.total_online_minutes = active * duration
                state.total_engaged_minutes = float(np.minimum(remaining, duration).sum())
                state.idle_drivers = int(np.count_nonzero(self._available_at[firm] <= 0))
                state.busy_drivers = active - state.idle_drivers
            self._batch_states[firm] = state
        self._batch_minute = 0.0

    def advance_request_time(self, fraction: float) -> None:
        """Release completed drivers at the next within-block request arrival."""
        if not ((self.supply_v2 or self.supply_v3) and self.config.driver_minute_capacity):
            return
        minute = float(fraction) * 60. * self.config.availability_block_hours
        if not 0 <= fraction <= 1 or minute < self._batch_minute:
            raise ValueError('request arrivals must be ordered within the block')
        self._batch_minute = minute
        for firm, state in self._batch_states.items():
            state.idle_drivers = int(np.count_nonzero(self._available_at[firm] <= minute))
            state.busy_drivers = state.active_drivers - state.idle_drivers

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
        required_earnings = self.config.reservation_wage_per_hour * float(self.shock_wage_multiplier) * (minutes / 60.0) * weather_penalty
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
        pay_reference_fare: Optional[float] = None,
    ) -> DispatchResult:
        """Offer one ride to the firm's drivers.

        ``pay_reference_fare`` (v26 ``driver_pay_basis='anchor'``) is the
        city-anchor fare for the same trip and context; driver pay is a share of
        it, so rider discounts/premiums are absorbed by the platform instead of
        moving driver pay one-for-one (upfront pay / regulated pay floors).
        """
        state = self._batch_states.setdefault(firm, FirmDriverBatchState(active_drivers=1, idle_drivers=1))
        state.offered_dispatches += 1
        pickup_minutes = self.estimate_pickup_minutes(firm, airport=airport)
        pickup_miles = max(0.1, pickup_minutes * 0.28)
        pay_fare = float(rider_fare if pay_reference_fare is None else pay_reference_fare)
        driver_pay = self.quote_driver_pay(firm, pay_fare, distance_miles, duration_minutes, pickup_minutes, airport)
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
        engaged = pickup_minutes + max(0.0, float(duration_minutes))
        if (self.supply_v2 or self.supply_v3) and self.config.driver_minute_capacity:
            driver = int(np.argmin(self._available_at[firm]))
            self._available_at[firm][driver] = self._batch_minute + engaged
            engaged = min(engaged, 60. * self.config.availability_block_hours - self._batch_minute)
        state.total_engaged_minutes += engaged
        state.total_wait_minutes += pickup_minutes
        platform_cost = self.config.platform_variable_cost + self.config.platform_cost_per_minute * max(0.0, float(duration_minutes))
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
        self._observed_blocks += 1
        self.last_states = {firm: state for firm, state in self._batch_states.items()}
        alpha = float(np.clip(self.config.state_smoothing_alpha, 0.0, 1.0))
        for firm, state in self.last_states.items():
            self._daily_pay[firm] += state.total_driver_pay
            self._daily_online[firm] += state.total_online_minutes
            if self.supply_v2 or self.supply_v3:
                # Same block clock and EMA weight as the associated earnings.
                self.pay_index_smoothed[firm] = ((1-alpha) * self.pay_index_smoothed[firm]
                                                 + alpha * self.pay_index[firm])
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

    def state_features_for_firm(self, firm: str, opponent: str = "Firm2") -> np.ndarray:
        own = self.smoothed_states.get(firm, self.last_states.get(firm, FirmDriverBatchState()))
        other = self.smoothed_states.get(opponent, self.last_states.get(opponent, FirmDriverBatchState()))
        features = [
            (np.arcsinh(own.active_drivers / getattr(self, '_feature_capacity_reference', self.config.base_active_drivers))
             if (self.supply_v2 or self.supply_v3) else np.clip(own.active_drivers / 800.0, 0.0, 1.0)),
            np.clip(own.idle_driver_share, 0.0, 1.0),
            np.clip(own.utilization, 0.0, 1.0),
            np.clip(own.acceptance_rate, 0.0, 1.0),
            np.clip(own.fulfillment_rate, 0.0, 1.0),
            np.clip(own.avg_wait_minutes / 20.0, 0.0, 1.0),
            (np.arcsinh(own.driver_earnings_per_online_hour / self._earnings_reference(firm))
             if (self.supply_v2 or self.supply_v3) else np.clip(own.driver_earnings_per_hour / 60.0, 0.0, 1.0)),
            np.clip((other.avg_wait_minutes - own.avg_wait_minutes + 10.0) / 20.0, 0.0, 1.0),
        ]
        return np.asarray(features, dtype=np.float32)

    def state_features_for_firm1(self) -> np.ndarray:
        return self.state_features_for_firm("Firm1", opponent="Firm2")
