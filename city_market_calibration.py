"""Data-driven market calibration for any city with public ride records.

The functions here turn canonical ride records (``city_data_pipeline``) into
the v26 market contract:

* ``fit_tariff_anchors``: city tariff anchors (base fare, per-minute, per-mile,
  booking fee, airport fee) plus time-block, weekend, weather, and service
  multipliers estimated from observed fares or posted quotes;
* ``weather_demand_multipliers``: relative request intensity under each
  weather state (ride share in that state divided by the state's NOAA
  frequency);
* ``price_level_index``: the city's reference-trip anchor fare relative to the
  NYC reference, used to scale rider reservation fares and outside-option costs
  so the same behavioral constants are meaningful in cheaper/costlier cities;
* ``driver_fare_share``: driver pay divided by passenger fare when the source
  reports driver pay (NYC FHVHV); other cities inherit the reference share.

All estimators are deliberately simple, robust (medians, bounded least
squares), and dependency-free so a new city can be onboarded from a field map
and its public files alone.  Every quantity is reported with its sample size
and fit quality so a paper can state exactly what is calibrated.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# NYC 2022-2023 reference used to express every other city's price level: the
# anchor quote of ``REFERENCE_TRIP`` under the calibrated NYC anchors.  The
# rider reservation-fare and outside-option constants were developed for NYC,
# so NYC keeps index 1.0 and other cities are scaled by their own reference
# fare.  The value used is stored in every calibration file.
REFERENCE_TRIP = {"distance_miles": 4.0, "duration_minutes": 15.0}
DEFAULT_REFERENCE_FARE = 23.16  # NYC 2022-23 calibrated anchors (5,000-row sample): 2.63+2.41+2.12*4+0.64*15
DEFAULT_DRIVER_FARE_SHARE = 0.72
BASE_TO_BOOKING_SPLIT = 3.00 / (3.00 + 2.75)  # NYC preset structure of the fixed fee
TEMPLATE_MILE_MINUTE = (1.90, 0.40)  # NYC preset ratio for unidentified duration
TIME_BLOCKS: Tuple[Tuple[int, int], ...] = ((0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 24))


def _get(record: Any, key: str) -> Any:
    if isinstance(record, Mapping):
        return record.get(key)
    return getattr(record, key, None)


def _float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def records_to_arrays(records: Iterable[Any]) -> Dict[str, np.ndarray]:
    cols: Dict[str, List[Any]] = {k: [] for k in (
        "price", "distance", "duration", "hour", "dow", "airport", "service",
        "shared", "weather", "driver_pay",
    )}
    for record in records:
        cols["price"].append(_float(_get(record, "actual_price")))
        cols["distance"].append(_float(_get(record, "distance_miles")))
        cols["duration"].append(_float(_get(record, "duration_minutes")))
        cols["hour"].append(int(_float(_get(record, "hour")) if _get(record, "hour") not in (None, "") else 12))
        cols["dow"].append(int(_float(_get(record, "day_of_week")) if _get(record, "day_of_week") not in (None, "") else 2))
        cols["airport"].append(_bool(_get(record, "airport")))
        cols["service"].append(str(_get(record, "service") or "economy").lower())
        cols["shared"].append(_bool(_get(record, "shared_ride")))
        cols["weather"].append(str(_get(record, "weather") or "clear").lower())
        cols["driver_pay"].append(_float(_get(record, "driver_pay")))
    out = {k: np.asarray(v) for k, v in cols.items()}
    for k in ("price", "distance", "duration", "driver_pay"):
        out[k] = out[k].astype(float)
    out["hour"] = out["hour"].astype(int) % 24
    out["dow"] = out["dow"].astype(int) % 7
    out["airport"] = out["airport"].astype(bool)
    out["shared"] = out["shared"].astype(bool)
    return out


def _block_of(hour: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(hour, dtype=int) // 4, 0, len(TIME_BLOCKS) - 1)


def _nonnegative_lstsq(design: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Least squares with a nonnegativity active set (tiny problems only)."""
    active = list(range(design.shape[1]))
    coef = np.zeros(design.shape[1])
    for _ in range(design.shape[1] + 1):
        sol, *_ = np.linalg.lstsq(design[:, active], target, rcond=None)
        if np.all(sol >= 0.0):
            coef[:] = 0.0
            coef[active] = sol
            return coef
        drop = active[int(np.argmin(sol))]
        active.remove(drop)
        if not active:
            break
    return coef


def fit_tariff_anchors(
    arrays: Mapping[str, np.ndarray],
    *,
    duration_identified: bool,
    service_identified: bool,
    airport_identified: bool,
    iterations: int = 4,
) -> Dict[str, Any]:
    """Estimate the multiplicative tariff structure used by ``quote_price``.

    quote = (base + booking + per_mile*D + per_minute*T) * H(block) * Wk(day)
            * Wx(weather) * S(service) + airport_fee * airport
    Shared/pooled fares are excluded because they are discounted products.
    """
    price = np.asarray(arrays["price"], dtype=float)
    distance = np.asarray(arrays["distance"], dtype=float)
    duration = np.asarray(arrays["duration"], dtype=float)
    hour = np.asarray(arrays["hour"], dtype=int)
    dow = np.asarray(arrays["dow"], dtype=int)
    airport = np.asarray(arrays["airport"], dtype=bool)
    service = np.asarray(arrays["service"], dtype=object)
    shared = np.asarray(arrays["shared"], dtype=bool)
    weather = np.asarray(arrays["weather"], dtype=object)
    usable = (
        np.isfinite(price) & (price > 0.0) & np.isfinite(distance)
        & (distance > 0.05) & (distance < 60.0) & ~shared
    )
    if duration_identified:
        usable &= np.isfinite(duration) & (duration > 0.5)
    if int(usable.sum()) < 50:
        raise ValueError("too few usable fare records to calibrate tariff anchors")
    p, d, h, w = price[usable], distance[usable], hour[usable], dow[usable]
    t = duration[usable] if duration_identified else np.full(d.shape, np.nan)
    ap = airport[usable] if airport_identified else np.zeros(d.shape, dtype=bool)
    sv = service[usable]
    wx = weather[usable]
    block = _block_of(h)
    weekend = w >= 5
    premium = (sv == "premium") if service_identified else np.zeros(d.shape, dtype=bool)

    mult = np.ones_like(p)
    airport_fee = 0.0
    coef = np.zeros(3)
    block_mult = np.ones(len(TIME_BLOCKS))
    weekend_mult = 1.0
    weather_mult = {"clear": 1.0, "rain": 1.0, "snow": 1.0}
    premium_mult = 1.0
    for _ in range(int(max(1, iterations))):
        y = (p - airport_fee * ap) / mult
        if duration_identified:
            design = np.column_stack([np.ones_like(d), d, t])
        else:
            design = np.column_stack([np.ones_like(d), d])
        base_mask = ~ap & ~premium
        sol = _nonnegative_lstsq(design[base_mask], y[base_mask])
        if duration_identified:
            coef = sol
        else:
            coef = np.array([sol[0], sol[1], 0.0])
        linear = design @ (sol if duration_identified else sol)
        linear = np.maximum(linear, 1e-6)
        ratio = (p - airport_fee * ap) / linear
        for b in range(len(TIME_BLOCKS)):
            sel = (block == b) & base_mask & ~weekend
            if sel.sum() >= 20:
                block_mult[b] = float(np.median(ratio[sel]))
        block_mult = block_mult / float(np.median(block_mult))
        offweek = ratio / block_mult[block]
        if weekend.any() and (~weekend).any():
            weekend_mult = float(np.median(offweek[weekend & base_mask]) / np.median(offweek[~weekend & base_mask]))
        wk = np.where(weekend, weekend_mult, 1.0)
        for state in ("rain", "snow"):
            sel = (wx == state) & base_mask
            clear = (wx == "clear") & base_mask
            if sel.sum() >= 20 and clear.sum() >= 20:
                weather_mult[state] = float(np.median(offweek[sel] / wk[sel]) / np.median(offweek[clear] / wk[clear]))
        wxm = np.array([weather_mult.get(str(x), 1.0) for x in wx])
        if service_identified and premium.sum() >= 20 and (~premium).sum() >= 20:
            premium_mult = float(np.median(offweek[premium] / wk[premium] / wxm[premium])
                                 / np.median(offweek[~premium] / wk[~premium] / wxm[~premium]))
        mult = block_mult[block] * wk * wxm * np.where(premium, premium_mult, 1.0)
        if airport_identified and ap.sum() >= 20:
            residual = p[ap] - (design[ap] @ (sol if duration_identified else sol)) * mult[ap]
            airport_fee = float(max(0.0, np.median(residual)))

    fixed, per_mile, per_minute = float(coef[0]), float(coef[1]), float(coef[2])
    duration_split = "estimated"
    if not duration_identified:
        # Price slope s per mile mixes per-mile and per-minute components.  Split
        # it with the NYC preset structure at the city's assumed mean speed.
        mph = 22.0
        tm, tt = TEMPLATE_MILE_MINUTE
        minute_per_mile = 60.0 / mph
        share_mile = tm / (tm + tt * minute_per_mile)
        slope = per_mile
        per_mile = slope * share_mile
        per_minute = slope * (1.0 - share_mile) / minute_per_mile
        duration_split = f"template_split_at_{mph:.0f}mph"
    base_fare = fixed * BASE_TO_BOOKING_SPLIT
    booking_fee = fixed - base_fare
    fitted = ((fixed + coef[1] * d + (coef[2] * t if duration_identified else 0.0)) * mult) + airport_fee * ap
    resid = p - fitted
    surge_hours = [
        (int(lo), int(hi), float(np.clip(block_mult[i], 0.5, 3.0)))
        for i, (lo, hi) in enumerate(TIME_BLOCKS)
    ]
    return {
        "base_fare": float(max(0.25, base_fare)),
        "booking_fee": float(max(0.0, booking_fee)),
        "per_mile": float(max(0.05, per_mile)),
        "per_minute": float(max(0.0, per_minute)),
        "airport_fee": float(airport_fee),
        "surge_hours": surge_hours,
        "day_multiplier": {"5": float(np.clip(weekend_mult, 0.7, 1.6)), "6": float(np.clip(weekend_mult, 0.7, 1.6))},
        "weather_multiplier": {k: float(np.clip(v, 0.7, 2.0)) for k, v in weather_mult.items()},
        "service_multiplier": {"economy": 1.0, "premium": float(np.clip(premium_mult if service_identified else 1.7, 1.0, 3.5))},
        "fit": {
            "rows": int(usable.sum()),
            "mae": float(np.mean(np.abs(resid))),
            "r2": float(1.0 - np.var(resid) / max(1e-12, np.var(p))),
            "duration_identified": bool(duration_identified),
            "duration_split": duration_split,
            "service_identified": bool(service_identified),
            "airport_identified": bool(airport_identified),
            "shared_rows_excluded": int((shared & np.isfinite(price)).sum()),
            "note": (
                "Observed prices include platform-specific surge and fees; anchors are the "
                "multiplicative tariff that best reproduces the city's typical fare, not a "
                "published rate card."
            ),
        },
    }


def reference_anchor_fare(tariff: Mapping[str, Any], trip: Mapping[str, float] = REFERENCE_TRIP) -> float:
    return float(
        tariff["base_fare"] + tariff["booking_fee"]
        + tariff["per_mile"] * trip["distance_miles"]
        + tariff["per_minute"] * trip["duration_minutes"]
    )


def price_level_index(tariff: Mapping[str, Any], reference_fare: float = DEFAULT_REFERENCE_FARE) -> float:
    return float(np.clip(reference_anchor_fare(tariff) / max(1e-6, float(reference_fare)), 0.25, 4.0))


def weather_demand_multipliers(
    arrays: Mapping[str, np.ndarray],
    weather_probs: Mapping[str, float],
    *,
    minimum_rows: int = 50,
) -> Dict[str, float]:
    """Relative request intensity: P(weather | ride) / P(weather)."""
    wx = np.asarray(arrays["weather"], dtype=object)
    total = float(wx.size)
    result: Dict[str, float] = {}
    for state, prior in weather_probs.items():
        count = float(np.sum(wx == state))
        if total <= 0 or prior <= 0 or count < minimum_rows:
            result[str(state)] = 1.0
            continue
        result[str(state)] = float(np.clip((count / total) / float(prior), 0.5, 2.0))
    # Normalize so the stationary expectation is one.
    expectation = sum(float(weather_probs.get(k, 0.0)) * v for k, v in result.items())
    if expectation > 0:
        result = {k: v / expectation for k, v in result.items()}
    return result


def driver_fare_share(arrays: Mapping[str, np.ndarray]) -> Optional[Dict[str, float]]:
    pay = np.asarray(arrays.get("driver_pay", []), dtype=float)
    price = np.asarray(arrays.get("price", []), dtype=float)
    ok = np.isfinite(pay) & np.isfinite(price) & (price > 2.0) & (pay > 0.0)
    if int(ok.sum()) < 50:
        return None
    ratio = pay[ok] / price[ok]
    ratio = ratio[(ratio > 0.2) & (ratio < 1.5)]
    if ratio.size < 50:
        return None
    return {"median": float(np.median(ratio)), "rows": int(ratio.size)}


def market_dynamics_calibration(
    records: Sequence[Any],
    *,
    weather_probs: Mapping[str, float],
    duration_identified: bool,
    service_identified: bool,
    airport_identified: bool,
    outside_option_base_fare: Optional[float] = None,
    reference_fare: float = DEFAULT_REFERENCE_FARE,
) -> Dict[str, Any]:
    arrays = records_to_arrays(records)
    tariff = fit_tariff_anchors(
        arrays,
        duration_identified=duration_identified,
        service_identified=service_identified,
        airport_identified=airport_identified,
    )
    share = driver_fare_share(arrays)
    return {
        "tariff": tariff,
        "weather_demand_multiplier": weather_demand_multipliers(arrays, weather_probs),
        "choice": {
            "price_level_index": price_level_index(tariff, reference_fare),
            "reference_fare": float(reference_fare),
            "reference_trip": dict(REFERENCE_TRIP),
            **({"outside_option_base_fare": float(outside_option_base_fare)}
               if outside_option_base_fare is not None else {}),
        },
        "supply": {
            "driver_fare_share": float(share["median"]) if share else DEFAULT_DRIVER_FARE_SHARE,
            "driver_fare_share_source": "observed_driver_pay" if share else "reference_default",
            "driver_fare_share_rows": int(share["rows"]) if share else 0,
        },
    }
