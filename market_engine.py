"""v27 market engine: vectorized expected-flow market step.

Replaces the v25/v26 per-rider sampling loop (enabled by ``--v27``) with:

* **Expected rider flows.** Each sampled rider contributes the *probability* of
  choosing no ride / Firm 1 / Firm 2 (the same cognitive utility as
  ``choice_models.CognitiveChoiceModel``, vectorized) instead of one sampled
  choice.  Monte Carlo noise from choice and dispatch sampling disappears; the
  remaining randomness is real (request volume, rider mix, trips, weather,
  shocks).  Driver acceptance is the expected acceptance probability.
* **Flow-based fleet capacity.** Each online driver supplies
  ``block_minutes`` of driver time per block and every completed trip uses
  ``pickup + trip`` minutes.  Pickup time rises as idle supply falls
  (square-root matching: ``pickup = p_ref / sqrt(1 - utilization)``), which
  caps utilization where pickup reaches the maximum tolerated pickup.  This
  replaces "one trip per driver per step".
* **Multi-homing and lagged driver supply.** Multi-homing drivers run both
  apps and are *shared* capacity in real time: their time goes to each
  platform in proportion to the driver minutes its requests need (tilted
  toward the app that pays more per engaged minute).  Single-homing drivers
  serve one platform.  Participation responds with a lag to adaptively
  expected earnings per online hour: market-wide for multi-homers, the own
  platform's for single-homers.  So a platform that prices itself out of
  demand cedes the shared drivers at once and loses its own drivers over days.
* **Shock channels** from ``market_shocks.ShockEffect``.

Units: rider-side metrics are in *sampled-rider* units (as in v26, where every
sampled rider counted once); capacity, driver time and driver earnings use
*realized-request* units (``weight = realized_requests / sampled riders``).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from driver_supply import FirmDriverBatchState, resolve_driver_incentive_multiplier
from gpt_threshold_utils import read_profile_price_threshold
from market_shocks import ShockEffect

INCOME_SCORE = {"<50k": 0.0, "50k-100k": 0.3, "100k-200k": 0.7, "200k+": 1.0}
EMPLOYMENT_FLEX = {"Student": 0.35, "Employed": 0.70, "Unemployed": 0.25, "Retired": 0.30}
INCOME_BRACKETS = ("<50k", "50k-100k", "100k-200k", "200k+")
BAD_WEATHER = {"rain", "snow", "storm"}
FIRMS = ("Firm1", "Firm2")
DEMAND_MODELS = ("cognitive", "nested_logit")


@dataclass
class EngineConfig:
    enabled: bool = False
    demand_model: str = "cognitive"  # cognitive | nested_logit
    pickup_ref_minutes: float = 3.5
    max_pickup_minutes: float = 16.0
    airport_pickup_factor: float = 1.18
    bad_weather_pickup_factor: float = 1.10
    # Online drivers per base request (all firms); set by calibration so the
    # anchor market runs at ``target_utilization``.
    fleet_per_base_request: float = 0.08
    target_utilization: float = 0.60
    # Driver participation: online fleet x (1 + elasticity * tanh((E - E_ref)/T))
    # with E the adaptively expected earnings per online hour and E_ref the
    # anchor-market earnings found by calibration (so the anchor is a steady state).
    participation_elasticity: float = 0.25
    participation_temperature_share: float = 0.75  # T as a share of E_ref
    reference_earnings_per_hour: float = 0.0  # 0 = use the reservation wage
    expectation_rate: float = 1.0 / 12.0  # per step (about two days at 6 steps/day)
    # Multi-homing: this share of drivers runs both apps (shared capacity
    # allocated within each step); the rest single-home, split by
    # ``firm1_base_fleet_share``.  ``allocation_sensitivity`` tilts the shared
    # time toward the app paying more per engaged minute (logit on the
    # relative pay gap; 0 = in proportion to requested driver minutes).
    multihoming_share: float = 0.5
    allocation_sensitivity: float = 3.0
    firm1_base_fleet_share: float = 0.5
    spillover_probability: float = 0.75
    # Outside option penalty when transit is unavailable (city price-level $).
    transit_penalty_dollars: float = 8.0
    transit_penalty_inconvenience: float = 1.5
    # Rider wait (ETA) refinement passes; 2 = predict with last step's ETA, then
    # correct with this step's solved pickup.
    wait_iterations: int = 2

    def __post_init__(self) -> None:
        if self.demand_model not in DEMAND_MODELS:
            raise ValueError(f"demand_model must be one of {DEMAND_MODELS}")
        if not (0.0 < self.pickup_ref_minutes < self.max_pickup_minutes):
            raise ValueError("need 0 < pickup_ref_minutes < max_pickup_minutes")
        self.fleet_per_base_request = float(max(1e-6, self.fleet_per_base_request))
        self.target_utilization = float(np.clip(self.target_utilization, 0.05, self.max_utilization - 0.01))
        self.participation_elasticity = float(np.clip(self.participation_elasticity, 0.0, 0.95))
        self.participation_temperature_share = float(max(0.05, self.participation_temperature_share))
        self.multihoming_share = float(np.clip(self.multihoming_share, 0.0, 1.0))
        self.firm1_base_fleet_share = float(np.clip(self.firm1_base_fleet_share, 0.05, 0.95))
        self.expectation_rate = float(np.clip(self.expectation_rate, 0.0, 1.0))
        self.allocation_sensitivity = float(max(0.0, self.allocation_sensitivity))
        self.spillover_probability = float(np.clip(self.spillover_probability, 0.0, 1.0))
        self.wait_iterations = int(max(1, self.wait_iterations))

    @property
    def max_utilization(self) -> float:
        ratio = float(self.pickup_ref_minutes) / float(self.max_pickup_minutes)
        return float(1.0 - ratio * ratio)

    def as_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["max_utilization"] = self.max_utilization
        return payload


@dataclass
class FleetState:
    """Persistent driver-side state (lives on ``DriverSupplyLayer.v27_fleet``)."""

    expected_earnings: float = float("nan")  # $/online hour, market-wide (multi-homers)
    expected_earnings_firm: Dict[str, float] = field(
        default_factory=lambda: {f: float("nan") for f in FIRMS}
    )  # $/online hour on each platform (its single-homers)
    shared_firm1_share: float = float("nan")  # last step's share of multi-homer time at Firm 1
    utilization: Dict[str, float] = field(default_factory=lambda: {f: 0.0 for f in FIRMS})
    pickup_minutes: Dict[str, float] = field(default_factory=lambda: {f: float("nan") for f in FIRMS})
    earnings_per_hour: Dict[str, float] = field(default_factory=lambda: {f: 0.0 for f in FIRMS})
    online_drivers: Dict[str, float] = field(default_factory=lambda: {f: 0.0 for f in FIRMS})
    # When True, participation and allocation stay at their reference values
    # (used by capacity calibration).
    frozen: bool = False

    def reset(self) -> None:
        self.expected_earnings = float("nan")
        self.expected_earnings_firm = {f: float("nan") for f in FIRMS}
        self.shared_firm1_share = float("nan")
        self.utilization = {f: 0.0 for f in FIRMS}
        self.pickup_minutes = {f: float("nan") for f in FIRMS}
        self.earnings_per_hour = {f: 0.0 for f in FIRMS}
        self.online_drivers = {f: 0.0 for f in FIRMS}


@dataclass
class OnlineFleet:
    """Drivers online in one step: single-homers per platform plus multi-homers."""

    dedicated: Dict[str, float]
    shared: float

    @property
    def total(self) -> float:
        return float(sum(self.dedicated.values()) + self.shared)


@dataclass
class FleetBatchState(FirmDriverBatchState):
    """Driver-side step summary with float counts (realized-request units).

    Earnings are per *online* hour (idle time included), the quantity that
    drives participation in v27; the v25 base class divides by engaged time.
    """

    @property
    def driver_earnings_per_hour(self) -> float:  # type: ignore[override]
        hours = self.total_online_minutes / 60.0
        return self.total_driver_pay / hours if hours > 0 else 0.0

    @property
    def utilization(self) -> float:  # type: ignore[override]
        return self.busy_drivers / self.active_drivers if self.active_drivers > 0 else 0.0


_STATE_FIELDS = (
    "active_drivers", "idle_drivers", "busy_drivers", "offered_dispatches",
    "accepted_dispatches", "rejected_dispatches", "completed_rides",
    "unfulfilled_requests", "total_driver_pay", "total_pickup_minutes",
    "total_pickup_miles", "total_wait_minutes", "total_online_minutes",
    "total_engaged_minutes", "total_acceptance_probability",
)


def smooth_batch_state(previous: Optional[FirmDriverBatchState], current: FleetBatchState,
                       alpha: float) -> FleetBatchState:
    """Exponential smoothing without the v25 integer rounding."""
    a = float(np.clip(alpha, 0.0, 1.0))
    if previous is None or not isinstance(previous, FleetBatchState):
        return FleetBatchState(**{name: float(getattr(current, name)) for name in _STATE_FIELDS})
    return FleetBatchState(**{
        name: (1.0 - a) * float(getattr(previous, name)) + a * float(getattr(current, name))
        for name in _STATE_FIELDS
    })


# --------------------------------------------------------------------------
# Rider features and quotes
# --------------------------------------------------------------------------
_FEATURE_FIELDS = (
    "income", "household", "age", "employment_flex", "loyalty_sign",
    "loyalty_missing", "loyalty_strength", "returning", "threshold",
    "reservation_override", "outside_cost_override", "inconvenience_override",
    "income_bracket_index",
)


def _profile_feature_row(profile: Mapping[str, Any]) -> Tuple[float, ...]:
    """Rider features with exactly the defaults of ``CognitiveChoiceModel``."""
    loyalty = profile.get("LoyaltyFirm")
    income = str(profile.get("IncomeBracket", "50k-100k"))
    explicit_res = profile.get("ReservationPrice", profile.get("RideshareReservationPrice"))
    explicit_cost = profile.get("OutsideOptionCost", profile.get("AlternativeTransportCost"))
    explicit_inc = profile.get("OutsideOptionInconvenience")
    return (
        INCOME_SCORE.get(income, 0.3),
        float(int(profile.get("HouseholdSize", 1) or 1)),
        float(int(profile.get("Age", 35) or 35)),
        EMPLOYMENT_FLEX.get(str(profile.get("EmploymentStatus", "Employed")), 0.45),
        1.0 if loyalty == "Firm1" else (-1.0 if loyalty == "Firm2" else 0.0),
        1.0 if loyalty is None else 0.0,
        float(profile.get("LoyaltyStrength", 0.0) or 0.0),
        1.0 if str(profile.get("LoyaltyType", "New")) == "Returning" else 0.0,
        read_profile_price_threshold(
            profile,
            0.25,
            8.00,
        ),
        float(explicit_res) if explicit_res is not None else float("nan"),
        float(explicit_cost) if explicit_cost is not None else float("nan"),
        float(explicit_inc) if explicit_inc is not None else float("nan"),
        float(INCOME_BRACKETS.index(income)) if income in INCOME_BRACKETS else -1.0,
    )


def profile_features(
    profiles: Sequence[Mapping[str, Any]],
    profile_memory: Any = None,
) -> Dict[str, np.ndarray]:
    """Column arrays of rider features (recomputed every step: profiles may change)."""
    rows = [_profile_feature_row(profile) for profile in profiles]
    matrix = np.asarray(rows, dtype=float).reshape(-1, len(_FEATURE_FIELDS))
    features = {name: matrix[:, j] for j, name in enumerate(_FEATURE_FIELDS)}
    if profile_memory is None:
        features["profile_memory_choice"] = np.zeros(len(profiles), dtype=float)
        features["profile_memory_outside"] = np.zeros(len(profiles), dtype=float)
    else:
        choice, outside = profile_memory.features(profiles)
        features["profile_memory_choice"] = np.asarray(choice, dtype=float)
        features["profile_memory_outside"] = np.asarray(outside, dtype=float)
    return features


def quote_vector(
    market_coefficients: Any,
    overrides: Any,
    *,
    time_multiplier: float,
    day_of_week: int,
    weather: str,
    distance: np.ndarray,
    duration: np.ndarray,
    airport: np.ndarray,
    service: Sequence[Any],
) -> np.ndarray:
    """Vectorized ``MarketInteraction.quote_price`` (same formula; cent rounding)."""
    m = market_coefficients

    def pick(name: str) -> Any:
        value = getattr(overrides, name, None) if overrides is not None else None
        return getattr(m, name) if value is None else value

    base = float(pick("base_fare")) + float(pick("booking_fee"))
    per_mile = float(pick("per_mile"))
    per_minute = float(pick("per_minute"))
    airport_fee = float(pick("airport_fee"))
    dow = float(pick("day_multiplier").get(int(day_of_week), 1.0))
    wco = float(pick("weather_multiplier").get(str(weather), 1.0))
    service_map = pick("service_multiplier")
    sco = np.array([float(service_map.get(str(s), 1.0)) for s in service], dtype=float)
    distance = np.asarray(distance, dtype=float)
    duration = np.asarray(duration, dtype=float)
    variable = per_mile * np.maximum(distance, 0.0) + per_minute * np.maximum(duration, 0.0)
    price = (base + variable) * float(time_multiplier) * dow * wco * sco
    price = price + airport_fee * np.asarray(airport, dtype=bool).astype(float)
    return np.round(np.maximum(price, 0.0), 2)


def apply_emergency_cap(price: np.ndarray, anchor: np.ndarray, cap: float, emergency: bool) -> np.ndarray:
    """Cap fares at ``cap`` x the anchor fare during a declared emergency."""
    if not emergency or cap <= 0.0:
        return price
    return np.minimum(price, np.round(cap * np.asarray(anchor, dtype=float), 2))


# --------------------------------------------------------------------------
# Rider choice
# --------------------------------------------------------------------------
def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid with no artificial probability floor.

    A clipped latent leaves ``sigmoid(-60)`` demand at every finite fare.
    Multiplying that floor by an uncapped price eventually makes revenue grow
    without bound.  ``logaddexp`` is stable over the full float64 domain and
    lets demand converge to exactly zero/one in the numerical tails.
    """
    values = np.asarray(x, dtype=float)
    return np.exp(-np.logaddexp(0.0, -values))


def _outside_probability_ceiling(model: Any) -> float:
    """Preserve the historical 2% ride floor except in the v28 experiment."""
    return (
        1.0
        if bool(getattr(model, "experimental_unbounded_economics", False))
        else 0.98
    )


def _premium(service: Sequence[Any]) -> np.ndarray:
    return np.array([str(s).lower() == "premium" for s in service], dtype=float)


def _trip_multiplier(trips: Mapping[str, Any], key: str) -> np.ndarray:
    """Return an optional per-trip v28 channel, neutral for v27 callers."""
    n = len(np.asarray(trips["distance"]))
    value = trips.get(key)
    if value is None:
        return np.ones(n, dtype=float)
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size != n or not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"trip channel {key!r} must be finite, non-negative, and length {n}")
    return array


def memory_utility(reputation: Any, sensitivity: np.ndarray) -> np.ndarray:
    """Vectorized ``FirmReputationMemory.choice_utility`` (linear in sensitivity)."""
    if reputation is None:
        return np.zeros_like(sensitivity, dtype=float)
    intercept = float(reputation.choice_utility(0.0))
    slope = float(reputation.choice_utility(1.0)) - intercept
    return intercept + slope * np.asarray(sensitivity, dtype=float)


def outside_memory_log_odds(reputation: Any) -> float:
    """Finite reputation shift for the outside-option log odds.

    ``outside_option_factor`` historically multiplied a probability after the
    sigmoid.  Any factor below one then forced ``P(no ride)`` below one even
    at an infinite fare, creating a captive-demand revenue tail.  Treating it
    as an odds factor preserves its intended direction while retaining the
    correct limiting probability.
    """
    if reputation is None:
        return 0.0
    factor = float(reputation.outside_option_factor())
    if not np.isfinite(factor) or factor <= 0.0:
        raise FloatingPointError("outside-option reputation odds must be positive and finite")
    return float(np.log(factor))


def outside_probability_from_logit(
    model: Any,
    logit: np.ndarray,
    *,
    legacy_preclip: bool = True,
) -> np.ndarray:
    """Versioned rider-memory treatment for an outside-option logit."""
    reputation = getattr(model, "reputation", None)
    unbounded = bool(getattr(model, "experimental_unbounded_economics", False))
    adjusted_logit = np.asarray(logit, dtype=float)
    if reputation is not None and unbounded:
        adjusted_logit = adjusted_logit + outside_memory_log_odds(reputation)
    probability = _sigmoid(adjusted_logit)
    if reputation is not None and not unbounded:
        # Exact v27 compatibility: legacy memory multiplied probability after
        # the sigmoid.  Cognitive/nested-main pre-clipped to .98 whereas the
        # old single-platform retry path did not, hence the explicit switch.
        if legacy_preclip:
            probability = np.clip(
                probability, 0.0, _outside_probability_ceiling(model)
            )
        probability = probability * float(reputation.outside_option_factor())
    return np.clip(probability, 0.0, _outside_probability_ceiling(model))


def rider_reservation(model: Any, feats: Mapping[str, np.ndarray], trips: Mapping[str, Any],
                      *, hour: int, weather: str, shock: ShockEffect) -> np.ndarray:
    """Vectorized ``BaseChoiceModel._reservation_price`` x the shock multiplier."""
    rush = (7 <= hour < 10) or (16 <= hour < 19)
    bad = str(weather).lower() in BAD_WEATHER
    premium = _premium(trips["service"])
    airport = np.asarray(trips["airport"], dtype=bool).astype(float)
    reservation = 11.0 + 26.0 * feats["income"]
    reservation = reservation + 0.95 * trips["distance"] + 0.10 * trips["duration"]
    reservation = reservation + 8.0 * premium + 7.0 * airport
    reservation = reservation + 2.5 * float(rush) + 2.5 * float(bad)
    reservation = reservation + feats["returning"] * (2.0 + 3.0 * feats["loyalty_strength"])
    reservation = reservation - 1.25 * np.minimum(np.maximum(feats["household"] - 1.0, 0.0), 4.0)
    level = float(getattr(model, "price_level_index", 1.0))
    reservation = np.clip(reservation * level, 5.0 * level, 120.0 * level)
    override = feats["reservation_override"]
    reservation = np.where(np.isfinite(override), np.clip(override, 5.0, 120.0), reservation)
    return (
        reservation
        * float(shock.reservation_multiplier)
        * _trip_multiplier(trips, "reservation_multiplier")
    )


def _outside_option(model: Any, feats: Mapping[str, np.ndarray], trips: Mapping[str, Any], *,
                    hour: int, weather: str, shock: ShockEffect,
                    penalty_dollars: float, penalty_inconvenience: float) -> Tuple[np.ndarray, np.ndarray]:
    rush = (7 <= hour < 10) or (16 <= hour < 19)
    bad = str(weather).lower() in BAD_WEATHER
    airport = np.asarray(trips["airport"], dtype=bool).astype(float)
    distance = trips["distance"]
    premium = _premium(trips["service"])
    level = float(getattr(model, "price_level_index", 1.0))
    base_fare = float(getattr(model, "outside_option_base_fare", 2.90))
    cost = np.clip(base_fare + level * (0.20 * distance + 9.0 * airport), 0.0, 80.0 * level)
    cost_override = feats["outside_cost_override"]
    cost = np.where(np.isfinite(cost_override), np.clip(cost_override, 0.0, 80.0), cost)
    inconvenience = 0.35 + 0.05 * np.minimum(distance, 15.0) + 0.45 * float(bad) + 0.55 * airport
    inconvenience = inconvenience + 0.20 * float(rush) + premium * (0.25 + 0.25 * feats["income"])
    inconvenience = np.clip(inconvenience, 0.0, 4.0)
    inc_override = feats["inconvenience_override"]
    inconvenience = np.where(np.isfinite(inc_override), np.clip(inc_override, 0.0, 4.0), inconvenience)
    cost = cost * _trip_multiplier(trips, "outside_option_cost_multiplier")
    inconvenience = inconvenience * _trip_multiplier(
        trips, "outside_option_inconvenience_multiplier"
    )
    # Transit shock: unavailable transit makes the outside option dearer and
    # less convenient (applied on top of any explicit profile value).
    unavailable = 1.0 - float(np.clip(shock.transit_availability, 0.0, 1.0))
    cost = cost + unavailable * float(penalty_dollars) * level
    inconvenience = inconvenience + unavailable * float(penalty_inconvenience)
    return cost, inconvenience


def cognitive_outside_probability(
    model: Any,
    feats: Mapping[str, np.ndarray],
    trips: Mapping[str, Any],
    *,
    hour: int,
    weather: str,
    candidate_price: np.ndarray,
    shock: ShockEffect,
    reservation: Optional[np.ndarray] = None,
    transit_penalty_dollars: float = 8.0,
    transit_penalty_inconvenience: float = 1.5,
) -> np.ndarray:
    """Probability of choosing the outside option over one available fare.

    This is the absolute-fare component of ``cognitive_probabilities``.  It is
    also used when a rider's first platform cannot serve them: the rider must
    reconsider the remaining quote against transit/no-ride instead of being
    transferred with a fixed probability independent of its price.
    """
    if reservation is None:
        reservation = rider_reservation(
            model, feats, trips, hour=hour, weather=weather, shock=shock
        )
    outside_cost, inconvenience = _outside_option(
        model,
        feats,
        trips,
        hour=hour,
        weather=weather,
        shock=shock,
        penalty_dollars=transit_penalty_dollars,
        penalty_inconvenience=transit_penalty_inconvenience,
    )
    candidate = np.asarray(candidate_price, dtype=float)
    denominator = np.maximum(np.asarray(reservation, dtype=float), 1e-6)
    affordability = (candidate - reservation) / denominator
    savings = np.maximum(0.0, candidate - outside_cost) / denominator
    latent = (
        -2.15
        + 4.00 * affordability
        + 0.65 * savings
        + 0.08 * np.minimum(np.maximum(feats["household"] - 1.0, 0.0), 4.0)
        - 0.70 * inconvenience
        - 0.35 * feats["loyalty_strength"]
        - 0.35 * feats["income"]
        + feats.get("profile_memory_outside", 0.0)
    )
    return outside_probability_from_logit(model, latent)


def cognitive_probabilities(
    model: Any,
    feats: Mapping[str, np.ndarray],
    trips: Mapping[str, Any],
    *,
    hour: int,
    day_of_week: int,
    weather: str,
    price1: np.ndarray,
    price2: np.ndarray,
    wait1: np.ndarray,
    wait2: np.ndarray,
    cancel1: np.ndarray,
    cancel2: np.ndarray,
    shock: ShockEffect,
    transit_penalty_dollars: float = 8.0,
    transit_penalty_inconvenience: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (P(no ride), P(Firm1 | rideshare), reservation) per rider.

    Mirrors ``CognitiveChoiceModel.choose`` term by term (unit-tested against
    it at neutral shocks).  Shock channels: price sensitivity (all riders, and
    airport riders separately), reservation fare, and transit availability.
    """
    rush = (7 <= hour < 10) or (16 <= hour < 19)
    weekend = int(day_of_week) >= 5
    bad = str(weather).lower() in BAD_WEATHER
    airport = np.asarray(trips["airport"], dtype=bool).astype(float)
    distance = trips["distance"]
    premium = _premium(trips["service"])
    income = feats["income"]
    household = feats["household"]

    sensitivity_shock = float(shock.price_sensitivity_multiplier) * np.where(
        airport > 0, float(shock.airport_price_sensitivity_multiplier), 1.0
    ) * _trip_multiplier(trips, "price_sensitivity_multiplier")
    price_sensitivity = (
        0.30 + 0.45 * (1.0 - income) + 0.05 * np.minimum(np.maximum(household - 1.0, 0.0), 3.0)
    ) * float(model.price_sensitivity_scale) * sensitivity_shock
    urgency = 0.40 * float(rush) + 0.35 * float(bad) + 0.30 * airport
    shopping = np.maximum(0.45, 1.0 - 0.45 * urgency)
    gap = price2 - price1
    wait_gap = wait2 - wait1
    cancel_gap = cancel2 - cancel1
    ratio = np.abs(gap) / np.maximum(feats["threshold"], 1e-6)
    salience = np.clip(0.35 + 0.65 * np.minimum(ratio, 1.0) + 0.25 * np.maximum(ratio - 1.0, 0.0), 0.20, 2.25)
    price_term = price_sensitivity * np.sign(gap) * np.abs(gap) ** 0.92 * shopping * salience

    loyalty_term = 0.85 * float(model.loyalty_scale) * feats["loyalty_strength"] * feats["loyalty_sign"]
    near_tie = float(getattr(model, "near_tie_habit_firm2", 0.03))
    habit = np.where(
        (feats["loyalty_missing"] > 0.0) & (np.abs(gap) < 0.60) & (near_tie != 0.0),
        -near_tie * (2.0 if weekend else 1.0),
        0.0,
    )
    risk_aversion = 0.20 + 0.30 * (feats["age"] >= 55).astype(float) + 0.20 * feats["employment_flex"]
    pressure = risk_aversion * (0.45 * float(bad) + 0.35 * airport + 0.20 * float(rush))
    bias = float(getattr(model, "reliability_bias_firm1", 0.02))
    wait_value = (
        0.035 + 0.035 * feats["employment_flex"] + 0.025 * np.maximum(float(rush), airport)
    ) * float(model.reliability_scale)
    reliability = pressure * bias * float(model.reliability_scale) + wait_value * wait_gap + 0.85 * pressure * cancel_gap
    comfort = premium * (0.05 + 0.10 * income)
    trip_scale = 1.0 + 0.06 * np.minimum(distance / 10.0, 2.0)
    reputation = getattr(model, "reputation", None)
    memory = memory_utility(reputation, price_sensitivity)
    latent = (
        trip_scale * (price_term + loyalty_term + habit + reliability + comfort)
        + memory
        + feats.get("profile_memory_choice", 0.0)
    )
    temperature = np.clip(0.70 - 0.15 * float(rush) - 0.10 * float(bad) - 0.05 * airport, 0.35, 0.90)
    p_f1 = _sigmoid(latent / temperature)

    reservation = rider_reservation(model, feats, trips, hour=hour, weather=weather, shock=shock)
    best = np.minimum(price1, price2)
    p0 = cognitive_outside_probability(
        model,
        feats,
        trips,
        hour=hour,
        weather=weather,
        candidate_price=best,
        shock=shock,
        reservation=reservation,
        transit_penalty_dollars=transit_penalty_dollars,
        transit_penalty_inconvenience=transit_penalty_inconvenience,
    )
    return p0, p_f1, reservation


@dataclass
class NestedLogitParams:
    """Initial values; to be calibrated by simulated method of moments."""

    reference_fare: float = 23.16
    alpha0: float = 2.2
    alpha_low_income: float = 0.8
    alpha_household: float = 0.1
    kink: float = 2.0
    beta_wait: float = 0.06
    beta_cancel: float = 1.0
    beta_inconvenience: float = 0.7
    loyalty: float = 0.8
    nest_scale: float = 0.35
    outside_constant: float = 0.2
    reservation_weight: float = 1.0
    # Rider-memory price sensitivity per unit of alpha / alpha0 (keeps the
    # memory channel on the same scale as in the cognitive model).
    memory_sensitivity: float = 0.5


def nested_logit_probabilities(
    model: Any,
    feats: Mapping[str, np.ndarray],
    trips: Mapping[str, Any],
    *,
    hour: int,
    day_of_week: int,
    weather: str,
    price1: np.ndarray,
    price2: np.ndarray,
    wait1: np.ndarray,
    wait2: np.ndarray,
    cancel1: np.ndarray,
    cancel2: np.ndarray,
    shock: ShockEffect,
    params: Optional[NestedLogitParams] = None,
    transit_penalty_dollars: float = 8.0,
    transit_penalty_inconvenience: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mixed nested logit: outside option vs a rideshare nest {Firm1, Firm2}.

    V_if = -alpha_i * g(p_if) - beta_w * wait_if - beta_c * cancel_if + loyalty_if,
    with price in relative units (fare / city reference fare) and a
    reference-dependent kink: the part of a fare *gap* beyond the rider's
    CrowdLLM threshold theta_i weighs ``kink`` times more.
    V_i0 = const - alpha_i * outside_cost_i / ref - beta_inc * inconvenience_i
           + w * log(ref / reservation_i).
    Returns the same triple as ``cognitive_probabilities``.
    """
    del day_of_week
    p = params or NestedLogitParams()
    reservation = rider_reservation(model, feats, trips, hour=hour, weather=weather, shock=shock)
    level = float(getattr(model, "price_level_index", 1.0))
    ref = max(1e-6, float(p.reference_fare) * level)
    airport = np.asarray(trips["airport"], dtype=bool).astype(float)
    sensitivity_shock = float(shock.price_sensitivity_multiplier) * np.where(
        airport > 0, float(shock.airport_price_sensitivity_multiplier), 1.0
    ) * _trip_multiplier(trips, "price_sensitivity_multiplier")
    alpha = p.alpha0 * (
        1.0 + p.alpha_low_income * (1.0 - feats["income"])
        + p.alpha_household * np.minimum(np.maximum(feats["household"] - 1.0, 0.0), 3.0)
    ) * sensitivity_shock * float(getattr(model, "price_sensitivity_scale", 1.0))
    theta = feats["threshold"] * level / ref

    def price_disutility(price: np.ndarray, other: np.ndarray) -> np.ndarray:
        excess = np.maximum(0.0, (price - other) / ref - theta)
        return price / ref + (p.kink - 1.0) * excess

    v1 = (-alpha * price_disutility(price1, price2) - p.beta_wait * wait1 - p.beta_cancel * cancel1
          + p.loyalty * feats["loyalty_strength"] * (feats["loyalty_sign"] > 0))
    v2 = (-alpha * price_disutility(price2, price1) - p.beta_wait * wait2 - p.beta_cancel * cancel2
          + p.loyalty * feats["loyalty_strength"] * (feats["loyalty_sign"] < 0))
    reputation = getattr(model, "reputation", None)
    v1 = v1 + memory_utility(reputation, p.memory_sensitivity * alpha / max(1e-6, p.alpha0))
    v1 = v1 + feats.get("profile_memory_choice", 0.0)
    lam = float(np.clip(p.nest_scale, 0.05, 1.0))
    inclusive = lam * np.logaddexp(v1 / lam, v2 / lam)
    p_f1 = _sigmoid((v1 - v2) / lam)
    outside_cost, inconvenience = _outside_option(
        model, feats, trips, hour=hour, weather=weather, shock=shock,
        penalty_dollars=transit_penalty_dollars, penalty_inconvenience=transit_penalty_inconvenience,
    )
    v0 = p.outside_constant - alpha * outside_cost / ref - p.beta_inconvenience * inconvenience
    v0 = v0 + p.reservation_weight * np.log(ref / np.maximum(reservation, 1e-6))
    v0 = v0 + feats.get("profile_memory_outside", 0.0)
    p0 = outside_probability_from_logit(model, v0 - inclusive)
    return p0, p_f1, reservation


def single_platform_ride_probability(
    model: Any,
    feats: Mapping[str, np.ndarray],
    trips: Mapping[str, Any],
    *,
    firm: str,
    hour: int,
    weather: str,
    price: np.ndarray,
    wait: np.ndarray,
    cancel: np.ndarray,
    shock: ShockEffect,
    demand_model: str = "cognitive",
    nested_params: Optional[NestedLogitParams] = None,
    reservation: Optional[np.ndarray] = None,
    transit_penalty_dollars: float = 8.0,
    transit_penalty_inconvenience: float = 1.5,
) -> np.ndarray:
    """Probability of accepting one remaining platform versus outside option.

    The quantity is intentionally *not* conditioned on having tried the first
    platform.  It is a fresh discrete-choice decision after non-service, which
    prevents capacity spillover from becoming a price-independent captive
    market.  Cognitive and nested-logit engines retain their own absolute-fare
    specifications.
    """
    if firm not in FIRMS:
        raise ValueError(f"unknown firm {firm!r}")
    if demand_model == "cognitive":
        p0 = cognitive_outside_probability(
            model,
            feats,
            trips,
            hour=hour,
            weather=weather,
            candidate_price=price,
            shock=shock,
            reservation=reservation,
            transit_penalty_dollars=transit_penalty_dollars,
            transit_penalty_inconvenience=transit_penalty_inconvenience,
        )
        return np.clip(1.0 - p0, 0.0, 1.0)
    if demand_model != "nested_logit":
        raise ValueError(f"unknown demand model {demand_model!r}")

    p = nested_params or NestedLogitParams()
    if reservation is None:
        reservation = rider_reservation(
            model, feats, trips, hour=hour, weather=weather, shock=shock
        )
    level = float(getattr(model, "price_level_index", 1.0))
    ref = max(1e-6, float(p.reference_fare) * level)
    airport = np.asarray(trips["airport"], dtype=bool).astype(float)
    sensitivity_shock = float(shock.price_sensitivity_multiplier) * np.where(
        airport > 0,
        float(shock.airport_price_sensitivity_multiplier),
        1.0,
    ) * _trip_multiplier(trips, "price_sensitivity_multiplier")
    alpha = p.alpha0 * (
        1.0
        + p.alpha_low_income * (1.0 - feats["income"])
        + p.alpha_household
        * np.minimum(np.maximum(feats["household"] - 1.0, 0.0), 3.0)
    ) * sensitivity_shock * float(
        getattr(model, "price_sensitivity_scale", 1.0)
    )
    loyalty_match = (
        feats["loyalty_sign"] > 0
        if firm == "Firm1"
        else feats["loyalty_sign"] < 0
    )
    platform_utility = (
        -alpha * np.asarray(price, dtype=float) / ref
        - p.beta_wait * np.asarray(wait, dtype=float)
        - p.beta_cancel * np.asarray(cancel, dtype=float)
        + p.loyalty * feats["loyalty_strength"] * loyalty_match
    )
    reputation = getattr(model, "reputation", None)
    if firm == "Firm1":
        platform_utility = platform_utility + memory_utility(
            reputation, p.memory_sensitivity * alpha / max(1e-6, p.alpha0)
        )
    outside_cost, inconvenience = _outside_option(
        model,
        feats,
        trips,
        hour=hour,
        weather=weather,
        shock=shock,
        penalty_dollars=transit_penalty_dollars,
        penalty_inconvenience=transit_penalty_inconvenience,
    )
    outside_utility = (
        p.outside_constant
        - alpha * outside_cost / ref
        - p.beta_inconvenience * inconvenience
        + p.reservation_weight
        * np.log(ref / np.maximum(np.asarray(reservation, dtype=float), 1e-6))
    )
    p0 = outside_probability_from_logit(
        model, outside_utility - platform_utility, legacy_preclip=False
    )
    return np.clip(1.0 - p0, 0.0, 1.0)


# --------------------------------------------------------------------------
# Capacity and drivers
# --------------------------------------------------------------------------
_CAPACITY_GRID = 513


def pickup_minutes(utilization: float, cfg: EngineConfig, factor: float = 1.0) -> float:
    idle = max(1e-3, 1.0 - float(utilization))
    return float(min(25.0, cfg.pickup_ref_minutes * float(factor) / math.sqrt(idle)))


def solve_capacity(
    demand: np.ndarray,
    trip_minutes: np.ndarray,
    driver_minutes: float,
    cfg: EngineConfig,
    pickup_factor: float = 1.0,
) -> Tuple[float, float, float]:
    """Served fraction phi, utilization u and pickup minutes for one firm.

    ``demand`` are dispatchable requests (realized units, after driver
    acceptance) and ``trip_minutes`` their in-vehicle durations.  With
    A = sum(q * d) and B = sum(q), utilization solves u = (A + B * pickup(u)) / T.
    If even u_max cannot carry all demand, phi < 1 rations requests so that
    u = u_max: phi = u_max * T / (A + B * pickup(u_max)).
    """
    A = float(np.sum(demand * trip_minutes))
    B = float(np.sum(demand))
    T = float(driver_minutes)
    u_max = cfg.max_utilization
    if B <= 1e-12:
        return 1.0, 0.0, pickup_minutes(0.0, cfg, pickup_factor)
    if T <= 1e-9:
        return 0.0, u_max, pickup_minutes(u_max, cfg, pickup_factor)
    # Sustainable served fraction at utilization u: h(u) = u T / (A + B pickup(u)).
    # pickup is convex, so h may peak inside [0, u_max].  If max h >= 1 every
    # request is served at the *lowest* utilization with h(u) = 1 (the stable
    # equilibrium); otherwise requests are rationed at phi = max h.
    grid = np.linspace(0.0, u_max, _CAPACITY_GRID)
    pick = np.minimum(25.0, cfg.pickup_ref_minutes * float(pickup_factor) / np.sqrt(np.maximum(1e-3, 1.0 - grid)))
    h = grid * T / (A + B * pick)
    best = int(np.argmax(h))
    if h[best] < 1.0:
        u = float(grid[best])
        return float(h[best]), u, pickup_minutes(u, cfg, pickup_factor)
    first = int(np.argmax(h >= 1.0))  # first grid point that is sustainable (>= 1)
    lo, hi = float(grid[max(0, first - 1)]), float(grid[first])
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if mid * T - (A + B * pickup_minutes(mid, cfg, pickup_factor)) < 0.0:
            lo = mid
        else:
            hi = mid
    return 1.0, hi, pickup_minutes(hi, cfg, pickup_factor)


WEATHER_DRIVER_PENALTY = {"clear": 1.0, "cloudy": 1.03, "rain": 1.18, "snow": 1.32, "storm": 1.40}


def driver_acceptance(
    pay: np.ndarray,
    distance: np.ndarray,
    duration: np.ndarray,
    pickup: float,
    *,
    reservation_wage: float,
    operating_cost_per_mile: float,
    weather: str,
    threshold: float,
) -> np.ndarray:
    """Expected acceptance: ``DriverSupplyLayer.driver_acceptance_probability`` vectorized."""
    penalty = WEATHER_DRIVER_PENALTY.get(str(weather).lower(), 1.0)
    pick = max(0.0, float(pickup))
    minutes = np.maximum(duration, 0.0) + pick
    cost = float(operating_cost_per_mile) * (np.maximum(distance, 0.0) + max(0.1, pick * 0.28))
    net = np.maximum(0.0, pay - cost)
    required = float(reservation_wage) * (minutes / 60.0) * penalty
    ratio = net / np.maximum(required, 1e-6)
    return np.clip(ratio / max(float(threshold), 1e-6), 0.0, 1.0)


def driver_pay_vector(policy: Any, basis: np.ndarray) -> np.ndarray:
    """``DriverSupplyLayer.quote_driver_pay`` vectorized."""
    minimum = float(policy.minimum_pay)
    base_pay = np.maximum(minimum, float(policy.fare_share) * np.maximum(basis, 0.0))
    incentive = resolve_driver_incentive_multiplier(policy)
    with np.errstate(over="ignore", invalid="ignore"):
        driver_pay = base_pay * incentive
    if bool(getattr(policy, "experimental_unbounded_economics", False)) and not np.all(np.isfinite(driver_pay)):
        raise OverflowError(
            "v28 vectorized driver pay overflowed while applying incentive "
            f"multiplier {incentive!r}"
        )
    return np.maximum(minimum, driver_pay)


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------
@dataclass
class StepResult:
    metrics: Dict[str, Dict[str, float]]  # sampled-rider units
    rider: Dict[str, np.ndarray]
    fleet: Dict[str, Dict[str, float]]
    batch_states: Dict[str, FleetBatchState]  # realized units
    welfare_by_income: Dict[str, List[float]]  # sampled-rider units
    weight: float


class MarketEngine:
    """One vectorized market step plus driver-supply dynamics."""

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.nested_logit = NestedLogitParams()

    def as_dict(self) -> Dict[str, Any]:
        payload = self.config.as_dict()
        if self.config.demand_model == "nested_logit":
            payload["nested_logit"] = asdict(self.nested_logit)
        return payload

    # ------------------------------------------------------------- drivers
    def reference_earnings(self, reservation_wage: float) -> float:
        ref = float(self.config.reference_earnings_per_hour)
        return ref if ref > 0.0 else float(reservation_wage)

    def _participation(self, expected: float, reference: float, hurdle: float, frozen: bool) -> float:
        cfg = self.config
        if frozen:
            return 1.0
        value = expected if math.isfinite(expected) else reference
        return 1.0 + cfg.participation_elasticity * math.tanh(
            (value - hurdle) / max(1e-6, cfg.participation_temperature_share * reference)
        )

    def online_drivers(
        self,
        fleet: FleetState,
        *,
        base_requests: float,
        supply_multiplier: float,
        weather: str,
        shock: ShockEffect,
        reservation_wage: float,
    ) -> OnlineFleet:
        cfg = self.config
        weather_supply = 0.88 if str(weather).lower() in BAD_WEATHER else 1.0
        reference = self.reference_earnings(reservation_wage)
        # A higher reservation wage (driver shortage) lowers participation at
        # the same expected earnings.
        hurdle = reference * float(shock.reservation_wage_multiplier)
        potential = (
            cfg.fleet_per_base_request * float(base_requests) * float(supply_multiplier)
            * weather_supply * float(shock.fleet_multiplier)
        )
        shared = potential * cfg.multihoming_share * self._participation(
            fleet.expected_earnings, reference, hurdle, fleet.frozen
        )
        split = {"Firm1": cfg.firm1_base_fleet_share, "Firm2": 1.0 - cfg.firm1_base_fleet_share}
        dedicated = {
            f: potential * (1.0 - cfg.multihoming_share) * split[f] * self._participation(
                fleet.expected_earnings_firm.get(f, float("nan")), reference, hurdle, fleet.frozen
            )
            for f in FIRMS
        }
        return OnlineFleet(dedicated=dedicated, shared=float(shared))

    def update_fleet(self, fleet: FleetState, earnings: Mapping[str, float], pooled: float,
                     *, reservation_wage: float) -> None:
        cfg = self.config
        reference = self.reference_earnings(reservation_wage)
        fleet.earnings_per_hour = {f: float(earnings[f]) for f in FIRMS}
        if fleet.frozen:
            return
        rate = cfg.expectation_rate

        def ema(previous: float, value: float) -> float:
            previous = previous if math.isfinite(previous) else reference
            return (1.0 - rate) * previous + rate * float(value)

        fleet.expected_earnings = ema(fleet.expected_earnings, pooled)
        fleet.expected_earnings_firm = {
            f: ema(fleet.expected_earnings_firm.get(f, float("nan")), earnings[f]) for f in FIRMS
        }

    def allocate_shared(self, need: Mapping[str, float], pay_rate: Mapping[str, float],
                        access: Mapping[str, float]) -> Dict[str, float]:
        """Share of multi-homer time at each platform.

        Pings reach multi-homers in proportion to the driver minutes each
        platform's requests need; drivers lean toward the app with the higher
        pay per engaged minute; ``access`` (an outage) scales a platform's reach.
        """
        cfg = self.config
        mean_rate = 0.5 * (pay_rate["Firm1"] + pay_rate["Firm2"])
        weights = {}
        for f, g in (("Firm1", "Firm2"), ("Firm2", "Firm1")):
            gap = (pay_rate[f] - pay_rate[g]) / mean_rate if mean_rate > 1e-9 else 0.0
            tilt = 2.0 / (1.0 + math.exp(-float(np.clip(cfg.allocation_sensitivity * gap, -60.0, 60.0))))
            weights[f] = max(0.0, float(need[f])) * float(access.get(f, 1.0)) * tilt
        total = weights["Firm1"] + weights["Firm2"]
        if total <= 1e-12:
            a1, a2 = float(access.get("Firm1", 1.0)), float(access.get("Firm2", 1.0))
            denom = a1 + a2
            return {"Firm1": a1 / denom if denom > 0 else 0.5, "Firm2": a2 / denom if denom > 0 else 0.5}
        return {f: weights[f] / total for f in FIRMS}

    # ----------------------------------------------------------------- step
    def choice_probabilities(self, model: Any, feats: Mapping[str, np.ndarray], trips: Mapping[str, Any],
                             **kwargs: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        cfg = self.config
        kwargs.setdefault("transit_penalty_dollars", cfg.transit_penalty_dollars)
        kwargs.setdefault("transit_penalty_inconvenience", cfg.transit_penalty_inconvenience)
        if cfg.demand_model == "nested_logit":
            return nested_logit_probabilities(model, feats, trips, params=self.nested_logit, **kwargs)
        return cognitive_probabilities(model, feats, trips, **kwargs)

    def step(
        self,
        *,
        model: Any,
        profiles: Sequence[Mapping[str, Any]],
        trips: Mapping[str, Any],
        hour: int,
        day_of_week: int,
        weather: str,
        price: Mapping[str, np.ndarray],
        pay_basis: Mapping[str, np.ndarray],
        pay_policies: Mapping[str, Any],
        fleet: FleetState,
        online: OnlineFleet,
        block_minutes: float,
        realized_requests: float,
        shock: ShockEffect,
        reservation_wage: float,
        operating_cost_per_mile: float,
        acceptance_threshold: float,
        platform_variable_cost: float,
        spillover: bool,
        rider_weights: Optional[np.ndarray] = None,
    ) -> StepResult:
        """One market step.

        ``trips['duration']`` must already include shocks.  ``rider_weights``
        (normalized here to mean 1) reweight sampled riders, e.g. to tilt the
        trip mix toward airport trips during a holiday wave without extra draws.
        """
        cfg = self.config
        n = len(profiles)
        if n == 0:
            raise ValueError("market step needs at least one sampled rider")
        weight = float(realized_requests) / float(n)
        if rider_weights is None:
            w = np.ones(n)
        else:
            w = np.asarray(rider_weights, dtype=float).reshape(-1)
            if w.size != n or not np.all(np.isfinite(w)) or np.any(w < 0.0) or float(w.sum()) <= 0.0:
                raise ValueError("rider_weights must be finite, non-negative, one per rider, not all zero")
            w = w * (n / float(w.sum()))
        wq = weight * w  # realized requests represented by each sampled rider
        profile_memory = getattr(model, "profile_memory", None)
        feats = profile_features(profiles, profile_memory=profile_memory)
        distance = np.asarray(trips["distance"], dtype=float)
        duration = np.asarray(trips["duration"], dtype=float)
        airport = np.asarray(trips["airport"], dtype=bool)
        trips = {**trips, "distance": distance, "duration": duration, "airport": airport}
        bad = str(weather).lower() in BAD_WEATHER
        pickup_factor = cfg.bad_weather_pickup_factor if bad else 1.0
        airport_factor = np.where(airport, cfg.airport_pickup_factor, 1.0)
        prices = {f: np.asarray(price[f], dtype=float) for f in FIRMS}
        wage = float(reservation_wage) * float(shock.reservation_wage_multiplier)
        op_cost = float(operating_cost_per_mile) * float(shock.operating_cost_multiplier)
        pay = {f: driver_pay_vector(pay_policies[f], np.asarray(pay_basis[f], dtype=float)) for f in FIRMS}
        block = float(block_minutes)
        access = {f: float(np.clip(shock.firm_capacity_multiplier.get(f, 1.0), 0.0, 1.0)) for f in FIRMS}
        dedicated_minutes = {f: float(online.dedicated[f]) * block * access[f] for f in FIRMS}
        shared_minutes = float(online.shared) * block
        nominal_pickup = pickup_minutes(cfg.target_utilization, cfg, pickup_factor) * airport_factor

        def allocate(load: Mapping[str, np.ndarray]) -> Tuple[Dict[str, float], Dict[str, float]]:
            """Driver minutes per platform: own single-homers + a share of multi-homers."""
            need = {f: float(np.sum(load[f] * (duration + nominal_pickup))) for f in FIRMS}
            rate = {
                f: (float(np.sum(load[f] * pay[f])) / need[f]) if need[f] > 1e-12 else 0.0
                for f in FIRMS
            }
            share = self.allocate_shared(need, rate, access)
            return share, {f: dedicated_minutes[f] + share[f] * shared_minutes for f in FIRMS}

        # Riders see each platform's ETA: start from last step's pickup and,
        # on later passes, this step's solved pickup.
        eta = {}
        for f in FIRMS:
            last = float(fleet.pickup_minutes.get(f, float("nan")))
            eta[f] = last if math.isfinite(last) else pickup_minutes(0.0, cfg, pickup_factor)
        retry_probability = {f: np.zeros(n, dtype=float) for f in FIRMS}
        for _ in range(cfg.wait_iterations):
            waits = {f: eta[f] * airport_factor for f in FIRMS}
            cancel = {f: np.clip((waits[f] - 6.0) / 18.0, 0.0, 0.95) for f in FIRMS}
            p0, pf1, reservation = self.choice_probabilities(
                model, feats, trips, hour=int(hour), day_of_week=int(day_of_week), weather=str(weather),
                price1=prices["Firm1"], price2=prices["Firm2"], wait1=waits["Firm1"], wait2=waits["Firm2"],
                cancel1=cancel["Firm1"], cancel2=cancel["Firm2"], shock=shock,
            )
            choose = {"Firm1": (1.0 - p0) * pf1, "Firm2": (1.0 - p0) * (1.0 - pf1)}
            accept = {
                f: driver_acceptance(
                    pay[f], distance, duration, eta[f],
                    reservation_wage=wage, operating_cost_per_mile=op_cost,
                    weather=weather, threshold=acceptance_threshold,
                )
                for f in FIRMS
            }
            # First pass: own demand only.
            own_load = {f: choose[f] * accept[f] * wq for f in FIRMS}
            _, driver_minutes = allocate(own_load)
            phi0 = {
                f: solve_capacity(own_load[f], duration, driver_minutes[f], cfg, pickup_factor)[0]
                for f in FIRMS
            }
            # Riders not served on their first choice re-request on the other app.
            spill_in = {f: np.zeros(n) for f in FIRMS}
            if spillover:
                for f, g in (("Firm1", "Firm2"), ("Firm2", "Firm1")):
                    failed = choose[f] * (1.0 - accept[f] * phi0[f])
                    if bool(
                        getattr(model, "experimental_unbounded_economics", False)
                    ):
                        remaining_ride_probability = single_platform_ride_probability(
                            model,
                            feats,
                            trips,
                            firm=g,
                            hour=int(hour),
                            weather=str(weather),
                            price=prices[g],
                            wait=waits[g],
                            cancel=cancel[g],
                            shock=shock,
                            demand_model=cfg.demand_model,
                            nested_params=self.nested_logit,
                            reservation=reservation,
                            transit_penalty_dollars=cfg.transit_penalty_dollars,
                            transit_penalty_inconvenience=cfg.transit_penalty_inconvenience,
                        )
                        retry = (
                            cfg.spillover_probability
                            * remaining_ride_probability
                        )
                    else:
                        # Exact v27 behavior for reproducible legacy results.
                        retry = cfg.spillover_probability * np.where(
                            prices[g] <= reservation, 1.0, 0.25
                        )
                    retry_probability[g] = retry
                    spill_in[g] = failed * retry
            load = {f: (choose[f] + spill_in[f]) * accept[f] * wq for f in FIRMS}
            shared_share, driver_minutes = allocate(load)
            solved = {
                f: solve_capacity(load[f], duration, driver_minutes[f], cfg, pickup_factor)
                for f in FIRMS
            }
            eta = {f: float(solved[f][2]) for f in FIRMS}
        phi = {f: float(solved[f][0]) for f in FIRMS}
        util = {f: float(solved[f][1]) for f in FIRMS}
        pick = {f: float(solved[f][2]) for f in FIRMS}
        served_own = {f: choose[f] * accept[f] * phi[f] for f in FIRMS}
        served_spill = {f: spill_in[f] * accept[f] * phi[f] for f in FIRMS}
        served = {f: served_own[f] + served_spill[f] for f in FIRMS}
        trip_cost = float(platform_variable_cost) + 0.05 * duration
        metrics: Dict[str, Dict[str, float]] = {}
        batch_states: Dict[str, FleetBatchState] = {}
        fleet_report: Dict[str, Dict[str, float]] = {}
        earnings: Dict[str, float] = {}
        online_eff = {f: float(online.dedicated[f]) + shared_share[f] * float(online.shared) for f in FIRMS}
        for f in FIRMS:
            pickup_f = pick[f] * airport_factor
            chosen = float(np.sum(w * choose[f]))
            spill_attempts = float(np.sum(w * spill_in[f]))
            offers = chosen + spill_attempts
            completed = float(np.sum(w * served[f]))
            driver_pay = float(np.sum(w * served[f] * pay[f]))
            accepted_offers = float(np.sum(w * (choose[f] + spill_in[f]) * accept[f]))
            metrics[f] = {
                "chosen": chosen,
                "completed": completed,
                "dispatch_offers": offers,
                # v26 semantics: every dispatch that does not complete counts as
                # unfulfilled and as a driver rejection.
                "unfulfilled": max(0.0, offers - completed),
                "driver_rejections": max(0.0, offers - completed),
                "acceptance_declines": max(0.0, offers - accepted_offers),
                "capacity_shortfall": max(0.0, accepted_offers - completed),
                "revenue": float(np.sum(w * served[f] * prices[f])),
                "driver_pay": driver_pay,
                "profit": float(np.sum(w * served[f] * (prices[f] - pay[f] - trip_cost))),
                "wait_minutes": float(np.sum(w * served[f] * pickup_f)),
                "pickup_minutes": float(np.sum(w * served[f] * pickup_f)),
                "spillover_completed": float(np.sum(w * served_spill[f])),
                "spillover_attempts": spill_attempts,
            }
            online_minutes = online_eff[f] * block
            earnings[f] = (driver_pay * weight) / (online_minutes / 60.0) if online_minutes > 1e-9 else 0.0
            busy = util[f] * driver_minutes[f] / block
            batch_states[f] = FleetBatchState(
                active_drivers=online_eff[f],
                idle_drivers=max(0.0, online_eff[f] - busy),
                busy_drivers=busy,
                offered_dispatches=offers * weight,
                accepted_dispatches=completed * weight,
                rejected_dispatches=max(0.0, offers - completed) * weight,
                completed_rides=completed * weight,
                unfulfilled_requests=max(0.0, offers - completed) * weight,
                total_driver_pay=driver_pay * weight,
                total_pickup_minutes=float(np.sum(wq * served[f] * pickup_f)),
                total_pickup_miles=float(np.sum(wq * served[f] * np.maximum(0.1, pickup_f * 0.28))),
                total_wait_minutes=float(np.sum(wq * served[f] * pickup_f)),
                total_online_minutes=online_minutes,
                total_engaged_minutes=float(np.sum(wq * served[f] * (duration + pickup_f))),
                total_acceptance_probability=accepted_offers * weight,
            )
            fleet_report[f] = {
                "online_drivers": online_eff[f],
                "single_homing_drivers": float(online.dedicated[f]),
                "multihoming_time_share": float(shared_share[f]),
                "driver_minutes": float(driver_minutes[f]),
                "utilization": util[f],
                "pickup_minutes": pick[f],
                "served_fraction": phi[f],
                "earnings_per_online_hour": float(earnings[f]),
            }
        # Welfare by income bracket: [requests, served, paid, surplus, no_ride, wait]
        welfare: Dict[str, List[float]] = {}
        served_total = served["Firm1"] + served["Firm2"]
        paid = served["Firm1"] * prices["Firm1"] + served["Firm2"] * prices["Firm2"]
        surplus = (served["Firm1"] * np.maximum(0.0, reservation - prices["Firm1"])
                   + served["Firm2"] * np.maximum(0.0, reservation - prices["Firm2"]))
        wait_total = (served["Firm1"] * pick["Firm1"] + served["Firm2"] * pick["Firm2"]) * airport_factor
        bracket = feats["income_bracket_index"]
        labels = list(INCOME_BRACKETS) + ["unknown"]
        for index, name in enumerate(labels):
            mask = bracket == (index if name != "unknown" else -1.0)
            if not np.any(mask):
                continue
            wm = w[mask]
            welfare[name] = [
                float(np.sum(wm)),
                float(np.sum(wm * served_total[mask])),
                float(np.sum(wm * paid[mask])),
                float(np.sum(wm * surplus[mask])),
                float(np.sum(wm * p0[mask])),
                float(np.sum(wm * wait_total[mask])),
            ]
        rider = {
            "p_no_ride": p0, "p_firm1_given_ride": pf1, "reservation": reservation,
            "choose_firm1": choose["Firm1"], "choose_firm2": choose["Firm2"],
            "served_firm1": served["Firm1"], "served_firm2": served["Firm2"],
            "served_spill_firm1": served_spill["Firm1"], "served_spill_firm2": served_spill["Firm2"],
            "spill_retry_probability_firm1": retry_probability["Firm1"],
            "spill_retry_probability_firm2": retry_probability["Firm2"],
            "accept_firm1": accept["Firm1"], "accept_firm2": accept["Firm2"],
            "pay_firm1": pay["Firm1"], "pay_firm2": pay["Firm2"],
            "wait_firm1": pick["Firm1"] * airport_factor, "wait_firm2": pick["Firm2"] * airport_factor,
            "weight": w,
        }
        if profile_memory is not None:
            served_total_unweighted = np.clip(
                served["Firm1"] + served["Firm2"], 0.0, 1.0
            )
            served_gap = served["Firm1"] - served["Firm2"]
            choice_target = np.divide(
                served_gap,
                np.maximum(served_total_unweighted, 1e-8),
                out=np.zeros_like(served_gap),
                where=served_total_unweighted > 1e-8,
            )
            outside_or_failed = np.clip(1.0 - served_total_unweighted, 0.0, 1.0)
            profile_memory.update(
                profiles,
                choice_target=choice_target,
                outside_target=2.0 * outside_or_failed - 1.0,
            )
        fleet.utilization = dict(util)
        fleet.pickup_minutes = dict(pick)
        fleet.online_drivers = dict(online_eff)
        fleet.shared_firm1_share = float(shared_share["Firm1"])
        total_minutes = sum(online_eff.values()) * block
        pooled = (
            sum(metrics[f]["driver_pay"] for f in FIRMS) * weight / (total_minutes / 60.0)
            if total_minutes > 1e-9 else self.reference_earnings(reservation_wage)
        )
        self.update_fleet(fleet, earnings, pooled, reservation_wage=reservation_wage)
        return StepResult(
            metrics=metrics, rider=rider, fleet=fleet_report, batch_states=batch_states,
            welfare_by_income=welfare, weight=weight,
        )
