"""Conditional trip/rider resampling for the v28 shock environment.

The v27 engine multiplied request weights after independently sampling trips
and riders.  That changes totals, but it does not make a holiday or storm draw
from a different joint population.  This module conditions the *existing*
trip bank and profile cache on the active shock regime before the market step.

Important provenance boundary: the source rows remain empirical/cache rows,
but trip-purpose strata and profile-to-segment propensities are modelled priors
unless a future ``shock_calibration`` artifact supplies estimates.  Diagnostics
therefore say ``modelled_segment`` rather than ``observed_segment``.

CRN contract
------------
When enabled, exactly three uniforms per input row are consumed on every call,
including neutral calls.  A neutral effect returns the original ordering.  An
active effect maps the same uniforms through a different weighted CDF, so two
policies evaluated from the same snapshot see identical exogenous draws.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from market_shocks import SHOCK_SEGMENTS, ShockEffect


@dataclass(frozen=True)
class ConditionalSamplingDiagnostics:
    enabled: bool
    changed: bool
    phase: str
    active: Tuple[str, ...]
    source: str
    airport_share_before: float
    airport_share_after: float
    mean_duration_before: float
    mean_duration_after: float
    weather_requested: str
    weather_match_share_before: float
    weather_match_share_after: float
    airport_direction_source: str
    duration_source: str
    synthetic_airport_candidates: int
    segment_counts: Dict[str, int]

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _is_neutral_map(values: Mapping[str, float]) -> bool:
    return all(abs(float(values.get(name, 1.0)) - 1.0) <= 1e-12 for name in SHOCK_SEGMENTS)


def _distribution_changes(effect: ShockEffect) -> bool:
    maps = (
        effect.segment_volume_multiplier,
        effect.segment_duration_multiplier,
        effect.segment_price_sensitivity_multiplier,
        effect.segment_reservation_multiplier,
        effect.segment_outside_cost_multiplier,
        effect.segment_outside_inconvenience_multiplier,
    )
    return not all(_is_neutral_map(values) for values in maps)


def _weighted_indices(weights: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or w.size == 0 or not np.all(np.isfinite(w)) or np.any(w < 0.0):
        raise ValueError("conditional sampling weights must be finite and non-negative")
    total = float(w.sum())
    if total <= 0.0:
        raise ValueError("conditional sampling weights cannot all be zero")
    cdf = np.cumsum(w / total)
    cdf[-1] = 1.0
    return np.searchsorted(cdf, np.asarray(uniforms, dtype=float), side="right").clip(0, w.size - 1)


def _weights_change_distribution(weights: np.ndarray) -> bool:
    """Return whether weights differ after normalization.

    A vector such as ``[4, 4, 4]`` is the same categorical distribution as
    ``[1, 1, 1]``.  Treating the former as a change needlessly bootstrap-
    resamples an already conditioned trip cell, reducing its effective sample
    size and breaking the neutral-call identity guarantee.
    """
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or w.size == 0 or not np.all(np.isfinite(w)) or np.any(w < 0.0):
        raise ValueError("conditional sampling weights must be finite and non-negative")
    total = float(w.sum())
    if total <= 0.0:
        raise ValueError("conditional sampling weights cannot all be zero")
    normalized = w / total
    uniform = np.full(w.size, 1.0 / float(w.size), dtype=float)
    return not bool(np.allclose(normalized, uniform, rtol=1e-12, atol=1e-12))


def _index_trip_arrays(trips: Mapping[str, Any], index: np.ndarray, n: int) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in trips.items():
        array = np.asarray(value)
        if array.ndim >= 1 and array.shape[0] == n:
            result[key] = array[index].copy()
        else:
            result[key] = value
    return result


def _airport_origin_probability(hour: int) -> float:
    """Weak time-of-day prior used only when the bank lacks direction labels."""
    h = int(hour) % 24
    if 5 <= h < 12:
        return 0.35  # morning airport trips more often head to the airport
    if 16 <= h < 24:
        return 0.62  # afternoon/evening trips more often leave the airport
    return 0.50


def _candidate_segments(
    trips: Mapping[str, Any],
    effect: ShockEffect,
    *,
    hour: int,
    day_of_week: int,
    direction_draw: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, int]:
    n = int(len(direction_draw))
    airport = np.asarray(trips["airport"], dtype=bool).copy()
    distance = np.asarray(trips["distance"], dtype=float)
    service = np.asarray(trips["service"], dtype=object)
    synthetic = 0

    airport_wanted = max(
        float(effect.segment_volume_multiplier.get("airport_origin", 1.0)),
        float(effect.segment_volume_multiplier.get("airport_destination", 1.0)),
    ) > 1.0 + 1e-12
    if airport_wanted and not np.any(airport) and n > 0:
        # A city source with no airport label cannot contribute observed
        # airport rows.  Use the longest cached trips as an explicit synthetic
        # fallback rather than pretending the source identified direction.
        count = max(1, min(n, int(np.ceil(0.10 * n))))
        candidates = np.argsort(distance, kind="mergesort")[-count:]
        airport[candidates] = True
        synthetic = int(count)

    segment = np.full(n, "essential", dtype=object)
    p_origin = _airport_origin_probability(hour)
    observed_origin = np.asarray(
        trips.get("airport_origin", np.zeros(n, dtype=bool)), dtype=bool
    ) & airport
    observed_destination = np.asarray(
        trips.get("airport_destination", np.zeros(n, dtype=bool)), dtype=bool
    ) & airport
    direction_known = observed_origin | observed_destination
    airport_origin = observed_origin | (airport & ~direction_known & (direction_draw < p_origin))
    segment[airport_origin] = "airport_origin"
    segment[airport & ~airport_origin] = "airport_destination"

    non_airport = ~airport
    event_active = abs(float(effect.segment_volume_multiplier.get("event", 1.0)) - 1.0) > 1e-12
    event_candidate = non_airport & event_active & (
        (direction_draw >= 0.65)
        | np.array([str(value).lower() == "premium" for value in service], dtype=bool)
    )
    segment[event_candidate] = "event"
    remaining = non_airport & ~event_candidate
    rush = (7 <= int(hour) < 10) or (16 <= int(hour) < 19)
    if int(day_of_week) < 5 and rush:
        commute = remaining & (direction_draw < 0.62)
        segment[commute] = "commute"
        remaining &= ~commute
    discretionary = remaining & (
        (int(day_of_week) >= 5)
        | (int(hour) >= 19)
        | np.array([str(value).lower() == "premium" for value in service], dtype=bool)
    )
    segment[discretionary] = "discretionary"
    return segment, airport, synthetic


def _profile_weights(profiles: Sequence[Mapping[str, Any]], segment: str) -> np.ndarray:
    """Weak, protected-attribute-free propensity prior for cache resampling."""
    employed = np.array(
        [str(profile.get("EmploymentStatus", "")) == "Employed" for profile in profiles], dtype=float
    )
    income_rank = np.array(
        [
            {"<50k": 0.0, "50k-100k": 1.0, "100k-200k": 2.0, "200k+": 3.0}.get(
                str(profile.get("IncomeBracket", "")), 1.0
            )
            for profile in profiles
        ],
        dtype=float,
    )
    if segment in {"airport_origin", "airport_destination"}:
        return 0.70 + 0.18 * income_rank + 0.25 * employed
    if segment == "commute":
        return 0.55 + 1.10 * employed
    if segment == "event":
        return 0.80 + 0.10 * income_rank + 0.10 * employed
    if segment == "discretionary":
        return 0.90 + 0.05 * income_rank
    return np.ones(len(profiles), dtype=float)


def _weather_weights(source_weather: np.ndarray, requested: str) -> np.ndarray:
    """Hierarchical weather match: exact, adverse-family, then no conditioning.

    This works both with a weather-conditioned ``TripBank`` cell and with a
    sparse fallback cell.  If the cache contains neither an exact nor a
    related adverse condition, weights stay neutral rather than inventing
    synthetic weather rows.
    """
    source = np.array([str(value).strip().lower() for value in source_weather], dtype=object)
    requested = str(requested or "").strip().lower()
    if not requested or requested == "unknown":
        return np.ones(source.size, dtype=float)
    exact = source == requested
    adverse_set = np.isin(source, ["rain", "snow", "storm"])
    requested_adverse = requested in {"rain", "snow", "storm"}
    if np.any(exact):
        weights = np.full(source.size, 0.50, dtype=float)
        weights[source == "unknown"] = 1.0
        if requested_adverse:
            weights[adverse_set] = 2.0
        weights[exact] = 4.0
        return weights
    if requested_adverse and np.any(adverse_set):
        weights = np.ones(source.size, dtype=float)
        weights[adverse_set] = 3.0
        return weights
    return np.ones(source.size, dtype=float)


def apply_conditional_shock_distribution(
    profiles: Sequence[Mapping[str, Any]],
    trips: Mapping[str, Any],
    effect: ShockEffect,
    rng: np.random.Generator,
    *,
    hour: int,
    day_of_week: int,
    enabled: bool = True,
    calibration_source: str = "documented_prior",
    weather: str = "",
    weather_conditioned_upstream: bool = False,
    airport_identified: bool = True,
    duration_source: str = "unknown",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], ConditionalSamplingDiagnostics]:
    """Condition cached profiles/trips on one shock effect.

    The output has the same number of rows.  Extra trip arrays are consumed by
    ``market_engine`` and default to one outside this path, so legacy callers
    remain compatible.
    """
    n = len(profiles)
    if n == 0:
        raise ValueError("conditional shock sampling requires at least one profile")
    for key in ("distance", "duration", "airport", "service"):
        if key not in trips or len(np.asarray(trips[key])) != n:
            raise ValueError(f"trip field {key!r} must contain one value per profile")

    airport_before = float(np.mean(np.asarray(trips["airport"], dtype=bool)))
    duration_before = float(np.mean(np.asarray(trips["duration"], dtype=float)))
    requested_weather = str(weather or "").strip().lower()
    source_weather = np.asarray(
        trips.get("source_weather", np.full(n, "unknown", dtype=object)), dtype=object
    )
    weather_before = float(np.mean(source_weather == requested_weather)) if requested_weather else float("nan")
    observed_directions = bool(
        np.any(np.asarray(trips.get("airport_origin", np.zeros(n)), dtype=bool))
        or np.any(np.asarray(trips.get("airport_destination", np.zeros(n)), dtype=bool))
    )
    direction_source = (
        "observed_endpoints_with_prior_for_unknown"
        if bool(airport_identified) and observed_directions
        else "partial_pool_synthetic_direction"
    )
    if not enabled:
        diagnostics = ConditionalSamplingDiagnostics(
            enabled=False, changed=False, phase=str(effect.phase), active=tuple(effect.active),
            source=str(calibration_source), airport_share_before=airport_before,
            airport_share_after=airport_before, mean_duration_before=duration_before,
            mean_duration_after=duration_before, weather_requested=requested_weather,
            weather_match_share_before=weather_before, weather_match_share_after=weather_before,
            airport_direction_source=direction_source, duration_source=str(duration_source),
            synthetic_airport_candidates=0,
            segment_counts={},
        )
        return [dict(profile) for profile in profiles], dict(trips), diagnostics

    # Fixed draw budget is part of the CRN contract.  Do not move these draws
    # behind an ``if active`` branch.
    trip_uniform = rng.random(n)
    profile_uniform = rng.random(n)
    direction_uniform = rng.random(n)

    # ``TripBank.sample(..., weather=...)`` may already have selected the best
    # available weather cell from the full empirical bank.  Reweighting its
    # finite bootstrap here would condition twice.  Standalone callers can
    # continue to request weather conditioning in this function.
    weather_weights = (
        np.ones(n, dtype=float)
        if weather_conditioned_upstream
        else _weather_weights(source_weather, requested_weather)
    )
    changed = _distribution_changes(effect) or _weights_change_distribution(weather_weights)
    if not changed:
        diagnostics = ConditionalSamplingDiagnostics(
            enabled=True, changed=False, phase=str(effect.phase), active=tuple(effect.active),
            source=str(calibration_source), airport_share_before=airport_before,
            airport_share_after=airport_before, mean_duration_before=duration_before,
            mean_duration_after=duration_before, weather_requested=requested_weather,
            weather_match_share_before=weather_before, weather_match_share_after=weather_before,
            airport_direction_source=direction_source, duration_source=str(duration_source),
            synthetic_airport_candidates=0,
            segment_counts={},
        )
        return [dict(profile) for profile in profiles], dict(trips), diagnostics

    candidate_segment, candidate_airport, synthetic = _candidate_segments(
        trips, effect, hour=int(hour), day_of_week=int(day_of_week), direction_draw=direction_uniform
    )
    trip_weights = np.array(
        [float(effect.segment_volume_multiplier.get(str(name), 1.0)) for name in candidate_segment],
        dtype=float,
    )
    trip_weights *= weather_weights
    trip_index = _weighted_indices(trip_weights, trip_uniform)
    selected_segment = candidate_segment[trip_index]
    selected_airport = candidate_airport[trip_index]
    sampled_trips = _index_trip_arrays(trips, trip_index, n)
    sampled_trips["airport"] = selected_airport.astype(bool)
    sampled_trips["shock_segment"] = selected_segment.copy()
    sampled_trips["airport_direction"] = np.array(
        [
            "from_airport" if name == "airport_origin" else
            "to_airport" if name == "airport_destination" else "none"
            for name in selected_segment
        ],
        dtype=object,
    )

    def segment_array(values: Mapping[str, float]) -> np.ndarray:
        return np.array([float(values.get(str(name), 1.0)) for name in selected_segment], dtype=float)

    duration_multiplier = segment_array(effect.segment_duration_multiplier)
    sampled_trips["duration"] = np.asarray(sampled_trips["duration"], dtype=float) * duration_multiplier
    sampled_trips["price_sensitivity_multiplier"] = segment_array(
        effect.segment_price_sensitivity_multiplier
    )
    sampled_trips["reservation_multiplier"] = segment_array(effect.segment_reservation_multiplier)
    sampled_trips["outside_option_cost_multiplier"] = segment_array(
        effect.segment_outside_cost_multiplier
    )
    sampled_trips["outside_option_inconvenience_multiplier"] = segment_array(
        effect.segment_outside_inconvenience_multiplier
    )

    sampled_profiles: List[Dict[str, Any]] = []
    profile_weight_cache: Dict[str, np.ndarray] = {}
    for i, name_obj in enumerate(selected_segment):
        name = str(name_obj)
        if name not in profile_weight_cache:
            profile_weight_cache[name] = _profile_weights(profiles, name)
        profile_index = int(_weighted_indices(profile_weight_cache[name], profile_uniform[i:i + 1])[0])
        profile = dict(profiles[profile_index])
        profile["ShockSegment"] = name
        profile["ShockPhase"] = str(effect.phase)
        sampled_profiles.append(profile)

    counts = {
        name: int(np.sum(selected_segment == name))
        for name in SHOCK_SEGMENTS
        if np.any(selected_segment == name)
    }
    airport_after = float(np.mean(selected_airport))
    duration_after = float(np.mean(np.asarray(sampled_trips["duration"], dtype=float)))
    sampled_weather = np.asarray(
        sampled_trips.get("source_weather", np.full(n, "unknown", dtype=object)), dtype=object
    )
    weather_after = float(np.mean(sampled_weather == requested_weather)) if requested_weather else float("nan")
    diagnostics = ConditionalSamplingDiagnostics(
        enabled=True, changed=True, phase=str(effect.phase), active=tuple(effect.active),
        source=str(calibration_source), airport_share_before=airport_before,
        airport_share_after=airport_after, mean_duration_before=duration_before,
        mean_duration_after=duration_after, weather_requested=requested_weather,
        weather_match_share_before=weather_before, weather_match_share_after=weather_after,
        airport_direction_source=direction_source, duration_source=str(duration_source),
        synthetic_airport_candidates=synthetic,
        segment_counts=counts,
    )
    return sampled_profiles, sampled_trips, diagnostics


__all__ = ["ConditionalSamplingDiagnostics", "apply_conditional_shock_distribution"]
