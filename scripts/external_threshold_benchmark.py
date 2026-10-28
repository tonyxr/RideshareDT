"""External revealed-preference benchmark for rider price-gap thresholds.

No open dataset records individual riders choosing between two simultaneous
ride-hailing quotes, so a rider-level price-gap threshold cannot be validated
directly.  The City of Chicago Transportation Network Providers (TNP) trip
records do, however, contain an open revealed-preference price/service trade-
off: ``Shared Trip Authorized`` marks riders who accepted a pooled ride (longer,
less certain trip) in exchange for a lower upfront fare.  Pooling was offered
from November 2018 until it was suspended in March 2020.

Under the threshold construct, a rider authorizes sharing when the shared
discount exceeds her price-gap threshold plus a pooling inconvenience cost::

    P(authorize | trip, area) = P(theta_i + kappa < discount_trip)

This script therefore tests two externally falsifiable implications of any
threshold population (rubric, LLM-numeric, constant, shuffled, ...):

1. Cross-area gradient.  Riders picked up in lower-income PUMAs should
   authorize sharing more often.  A threshold source predicts each PUMA's rate
   from that PUMA's ACS PUMS income/household composition.  The test is the
   correlation between predicted and observed PUMA authorization rates.
2. Discount response.  Authorization should rise with the trip's discount in
   the way implied by the CDF of theta (kappa is the only fitted parameter,
   matched to the citywide rate).

Both tests are aggregate (ecological) checks of construct direction and scale;
neither validates an individual rider's threshold.  The script uses only local
files already downloaded by the city pipeline and has no network dependency.

Usage::

    python scripts/external_threshold_benchmark.py \
        --threshold-source rubric=artifacts/.../raw/thresholds/chicago/primary.jsonl \
        --threshold-source llm_numeric=artifacts/.../thresholds_llm_numeric/chicago/primary.jsonl \
        --out artifacts/external_validation/chicago
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import struct
import sys
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from city_data_pipeline import _read_dbf_records, puma_codes_for_city  # noqa: E402

INCOME_KEYS = ("<50k", "50k-100k", "100k-200k", "200k+")
POOLING_START = datetime(2018, 11, 1)
POOLING_END = datetime(2020, 3, 1)


# --------------------------------------------------------------------------
# Geography: minimal ESRI shapefile polygon reader and point-in-polygon.
# --------------------------------------------------------------------------
def read_polygons(shp_payload: bytes) -> List[List[np.ndarray]]:
    """Return one list of rings (N x 2 lon/lat arrays) per shapefile record."""
    records: List[List[np.ndarray]] = []
    cursor = 100
    size = len(shp_payload)
    while cursor + 8 <= size:
        _, content_words = struct.unpack(">ii", shp_payload[cursor : cursor + 8])
        cursor += 8
        content = shp_payload[cursor : cursor + 2 * content_words]
        cursor += 2 * content_words
        shape_type = struct.unpack("<i", content[:4])[0]
        if shape_type == 0:
            records.append([])
            continue
        if shape_type not in (5, 15, 25):
            raise ValueError(f"unsupported shape type {shape_type}")
        num_parts, num_points = struct.unpack("<ii", content[36:44])
        parts = list(struct.unpack(f"<{num_parts}i", content[44 : 44 + 4 * num_parts]))
        offset = 44 + 4 * num_parts
        points = np.frombuffer(
            content[offset : offset + 16 * num_points], dtype="<f8"
        ).reshape(num_points, 2)
        parts.append(num_points)
        records.append(
            [points[parts[i] : parts[i + 1]].copy() for i in range(num_parts)]
        )
    return records


def point_in_rings(lon: float, lat: float, rings: Sequence[np.ndarray]) -> bool:
    """Even-odd rule across all rings (holes handled automatically)."""
    inside = False
    for ring in rings:
        x = ring[:, 0]
        y = ring[:, 1]
        x2 = np.roll(x, -1)
        y2 = np.roll(y, -1)
        crosses = ((y > lat) != (y2 > lat)) & (
            lon < (x2 - x) * (lat - y) / np.where(y2 - y == 0.0, 1e-300, y2 - y) + x
        )
        inside ^= bool(np.count_nonzero(crosses) % 2)
    return inside


class PumaLocator:
    def __init__(self, geography_zip: Path, keep_codes: Iterable[str]):
        keep = {str(code).zfill(5) for code in keep_codes}
        with zipfile.ZipFile(geography_zip) as archive:
            shp = [n for n in archive.namelist() if n.lower().endswith(".shp")][0]
            dbf = [n for n in archive.namelist() if n.lower().endswith(".dbf")][0]
            polygons = read_polygons(archive.read(shp))
            attributes = list(_read_dbf_records(archive.read(dbf)))
        self.items: List[Tuple[str, List[np.ndarray], Tuple[float, float, float, float]]] = []
        for rings, attrs in zip(polygons, attributes):
            code = str(attrs.get("PUMACE20") or attrs.get("PUMA20") or "").zfill(5)
            if code not in keep or not rings:
                continue
            allpts = np.vstack(rings)
            bbox = (allpts[:, 0].min(), allpts[:, 1].min(), allpts[:, 0].max(), allpts[:, 1].max())
            self.items.append((code, rings, bbox))
        self._cache: Dict[Tuple[float, float], Optional[str]] = {}

    def locate(self, lon: float, lat: float) -> Optional[str]:
        key = (round(lon, 6), round(lat, 6))
        if key in self._cache:
            return self._cache[key]
        found = None
        for code, rings, (x0, y0, x1, y1) in self.items:
            if x0 <= lon <= x1 and y0 <= lat <= y1 and point_in_rings(lon, lat, rings):
                found = code
                break
        self._cache[key] = found
        return found


# --------------------------------------------------------------------------
# ACS PUMS composition by PUMA.
# --------------------------------------------------------------------------
def income_bracket(value: float) -> str:
    if value < 50000.0:
        return "<50k"
    if value < 100000.0:
        return "50k-100k"
    if value < 200000.0:
        return "100k-200k"
    return "200k+"


def household_class(size: int) -> str:
    return "1-2" if size <= 2 else ("3-4" if size <= 4 else "5+")


def puma_composition(household_zip: Path, pumas: Iterable[str]) -> Dict[str, Dict[str, object]]:
    keep = {str(code).zfill(5) for code in pumas}
    cells: Dict[str, Dict[Tuple[str, str], float]] = defaultdict(lambda: defaultdict(float))
    incomes: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    with zipfile.ZipFile(household_zip) as archive:
        member = [n for n in archive.namelist() if Path(n).name.lower().startswith("psam_h") and n.lower().endswith(".csv")][0]
        with archive.open(member) as raw:
            for row in csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")):
                code = str(row.get("PUMA", "")).zfill(5)
                if code not in keep:
                    continue
                try:
                    weight = float(row.get("WGTP") or 0.0)
                    size = int(float(row.get("NP") or 0))
                    hincp = float(row.get("HINCP") or "nan")
                    adj = float(row.get("ADJINC") or 1_000_000) / 1_000_000.0
                except ValueError:
                    continue
                if weight <= 0.0 or size <= 0 or not math.isfinite(hincp):
                    continue
                value = hincp * adj
                cells[code][(income_bracket(value), household_class(size))] += weight
                incomes[code].append((value, weight))
    result: Dict[str, Dict[str, object]] = {}
    for code, table in cells.items():
        total = sum(table.values())
        pairs = sorted(incomes[code])
        values = np.array([v for v, _ in pairs])
        weights = np.cumsum([w for _, w in pairs])
        median = float(values[np.searchsorted(weights, 0.5 * weights[-1])])
        result[code] = {
            "cells": {f"{i}|{h}": w / total for (i, h), w in table.items()},
            "median_household_income": median,
            "households_weight": total,
        }
    return result


# --------------------------------------------------------------------------
# Threshold sources: conditional theta distributions by income x household.
# --------------------------------------------------------------------------
def load_threshold_cells(path: Path) -> Dict[str, np.ndarray]:
    cells: Dict[str, List[float]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            theta = row.get("PriceThreshold")
            if theta is None:
                continue
            key = f"{row.get('IncomeBracket')}|{household_class(int(row.get('HouseholdSize', 1) or 1))}"
            cells[key].append(float(theta))
            cells["__all__"].append(float(theta))
    return {key: np.asarray(values, dtype=float) for key, values in cells.items()}


def constant_cells(reference: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    mean = float(np.mean(reference["__all__"]))
    return {key: np.full(1, mean) for key in reference}


def shuffled_cells(reference: Mapping[str, np.ndarray], seed: int = 4500) -> Dict[str, np.ndarray]:
    """Same marginal theta distribution, demographic link destroyed."""
    pooled = np.asarray(reference["__all__"], dtype=float)
    return {key: pooled for key in reference}


def predicted_authorization(
    cells: Mapping[str, np.ndarray],
    composition: Mapping[str, float],
    discounts: np.ndarray,
    kappa: float,
) -> float:
    """Mixture probability that theta + kappa < discount, averaged over trips."""
    fallback = cells["__all__"]
    total = 0.0
    for key, weight in composition.items():
        thetas = np.sort(cells.get(key, fallback))
        if thetas.size == 0:
            thetas = np.sort(fallback)
        cdf = np.searchsorted(thetas, discounts - kappa, side="right") / float(thetas.size)
        total += float(weight) * float(np.mean(cdf))
    return total


# --------------------------------------------------------------------------
# Chicago TNP stream.
# --------------------------------------------------------------------------
def parse_timestamp(text: str) -> Optional[datetime]:
    for fmt in ("%m/%d/%Y %I:%M:%S %p", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _strided_rows(handle, stride: int, offset: int) -> Iterator[Dict[str, str]]:
    header = next(csv.reader([handle.readline()]))
    for index, line in enumerate(handle):
        if stride > 1 and (index - offset) % stride:
            continue
        values = next(csv.reader([line]))
        yield dict(zip(header, values))


def stream_chicago(
    path: Path,
    locator: PumaLocator,
    max_rows: Optional[int],
    stride: int = 1,
    offset: int = 0,
) -> Dict[str, np.ndarray]:
    columns: Dict[str, List[float]] = defaultdict(list)
    puma_ids: List[str] = []
    read = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in _strided_rows(handle, max(1, int(stride)), int(offset)):
            read += 1
            if max_rows and read > max_rows:
                break
            stamp = parse_timestamp(row.get("Trip Start Timestamp", ""))
            if stamp is None or not (POOLING_START <= stamp < POOLING_END):
                continue
            try:
                miles = float(row["Trip Miles"])
                seconds = float(row["Trip Seconds"])
                fare = float(row["Fare"])
                lat = float(row["Pickup Centroid Latitude"])
                lon = float(row["Pickup Centroid Longitude"])
            except (KeyError, TypeError, ValueError):
                continue
            if not (0.2 <= miles <= 60.0 and 120.0 <= seconds <= 7200.0 and 2.0 <= fare <= 200.0):
                continue
            code = locator.locate(lon, lat)
            if code is None:
                continue
            columns["miles"].append(miles)
            columns["minutes"].append(seconds / 60.0)
            columns["fare"].append(fare)
            columns["hour"].append(float(stamp.hour))
            columns["shared"].append(1.0 if str(row.get("Shared Trip Authorized", "")).strip().lower() == "true" else 0.0)
            puma_ids.append(code)
    result = {key: np.asarray(values, dtype=float) for key, values in columns.items()}
    result["puma"] = np.asarray(puma_ids)
    return result


def fit_fare_model(miles: np.ndarray, minutes: np.ndarray, hour: np.ndarray, fare: np.ndarray) -> np.ndarray:
    peak = ((hour >= 7) & (hour < 10)) | ((hour >= 16) & (hour < 19))
    design = np.column_stack([np.ones_like(miles), miles, minutes, peak.astype(float)])
    coef, *_ = np.linalg.lstsq(design, fare, rcond=None)
    return coef


def predict_fare(coef: np.ndarray, miles: np.ndarray, minutes: np.ndarray, hour: np.ndarray) -> np.ndarray:
    peak = ((hour >= 7) & (hour < 10)) | ((hour >= 16) & (hour < 19))
    return coef[0] + coef[1] * miles + coef[2] * minutes + coef[3] * peak


def fit_kappa(cells, compositions, trip_discounts_by_puma, trip_counts, citywide_rate) -> float:
    """One scalar pooling-inconvenience cost matched to the citywide rate."""
    lo, hi = -10.0, 15.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        total = 0.0
        weight = 0.0
        for code, discounts in trip_discounts_by_puma.items():
            total += trip_counts[code] * predicted_authorization(cells, compositions[code], discounts, mid)
            weight += trip_counts[code]
        if total / weight > citywide_rate:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def logistic_puma_effects(
    shared: np.ndarray,
    puma: np.ndarray,
    codes: Sequence[str],
    log_fare: np.ndarray,
    hour: np.ndarray,
) -> Dict[str, float]:
    """PUMA fixed effects on the log-odds of authorization, adjusted for trip
    size (log solo fare) and peak period, via IRLS.  Returned effects are
    re-centered to have trip-weighted mean zero."""
    keep = np.isin(puma, list(codes))
    y = shared[keep]
    peak = (((hour >= 7) & (hour < 10)) | ((hour >= 16) & (hour < 19)))[keep].astype(float)
    dummies = np.column_stack([(puma[keep] == code).astype(float) for code in codes])
    x = np.column_stack([dummies, log_fare[keep] - np.mean(log_fare[keep]), peak - np.mean(peak)])
    beta = np.zeros(x.shape[1])
    for _ in range(50):
        eta = np.clip(x @ beta, -30, 30)
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.maximum(mu * (1.0 - mu), 1e-9)
        z = eta + (y - mu) / w
        xtw = x.T * w
        step = np.linalg.solve(xtw @ x + 1e-8 * np.eye(x.shape[1]), xtw @ z)
        if np.max(np.abs(step - beta)) < 1e-8:
            beta = step
            break
        beta = step
    effects = beta[: len(codes)]
    counts = dummies.sum(axis=0)
    effects = effects - np.average(effects, weights=counts)
    return {code: float(value) for code, value in zip(codes, effects)}


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "data" / "raw"))
    parser.add_argument("--rides", default="rides/chicago_kaggle/Transportation_Network_Providers_-_Trips__2018_-_2022_.csv")
    parser.add_argument("--household-zip", default="demographics/acs_pums/csv_hil.zip")
    parser.add_argument("--geography-zip", default="demographics/acs_pums/tl_2024_17_puma20.zip")
    parser.add_argument("--threshold-source", action="append", default=[],
                        help="name=path/to/thresholds.jsonl (repeatable)")
    parser.add_argument("--min-trips-per-puma", type=int, default=300)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--row-stride", type=int, default=1,
                        help="parse every k-th source row (systematic sample)")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "artifacts" / "external_validation" / "chicago"))
    args = parser.parse_args(argv)

    root = Path(args.data_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pumas = puma_codes_for_city("chicago", root / args.geography_zip)
    locator = PumaLocator(root / args.geography_zip, pumas)
    composition = puma_composition(root / args.household_zip, pumas)
    trips = stream_chicago(root / args.rides, locator, args.max_rows or None, stride=args.row_stride)
    if trips["fare"].size == 0:
        raise SystemExit("no pooling-era Chicago trips were located inside city PUMAs")

    solo = trips["shared"] < 0.5
    coef = fit_fare_model(trips["miles"][solo], trips["minutes"][solo], trips["hour"][solo], trips["fare"][solo])
    expected_solo = predict_fare(coef, trips["miles"], trips["minutes"], trips["hour"])
    shared = ~solo
    observed_discount = expected_solo[shared] - trips["fare"][shared]
    # Discount offered to every trip is approximated by the median relative
    # discount realized on authorized trips, applied to the trip's solo fare.
    relative_discount = float(np.median(observed_discount / np.maximum(expected_solo[shared], 1.0)))
    trip_discount = relative_discount * expected_solo

    citywide_rate = float(np.mean(trips["shared"]))
    codes = sorted(code for code in set(trips["puma"].tolist())
                   if code in composition and int(np.sum(trips["puma"] == code)) >= args.min_trips_per_puma)
    observed = {}
    discounts_by_puma = {}
    counts = {}
    for code in codes:
        mask = trips["puma"] == code
        observed[code] = float(np.mean(trips["shared"][mask]))
        # Quantile summaries keep the mixture CDF exact to ~0.1% while making
        # the one-parameter calibration cheap.
        discounts_by_puma[code] = np.quantile(trip_discount[mask], np.linspace(0.0005, 0.9995, 1000))
        counts[code] = int(mask.sum())
    compositions = {code: composition[code]["cells"] for code in codes}
    adjusted = logistic_puma_effects(
        trips["shared"], trips["puma"], codes, np.log(np.maximum(expected_solo, 1.0)), trips["hour"]
    )
    adjusted_vec = np.array([adjusted[code] for code in codes])
    citywide_quantiles = np.quantile(trip_discount, np.linspace(0.0005, 0.9995, 1000))
    citywide_discounts = {code: citywide_quantiles for code in codes}
    median_income = np.array([composition[code]["median_household_income"] for code in codes])
    observed_vec = np.array([observed[code] for code in codes])

    sources: Dict[str, Dict[str, np.ndarray]] = {}
    for item in args.threshold_source:
        name, _, path = item.partition("=")
        sources[name] = load_threshold_cells(Path(path))
    if sources:
        reference = next(iter(sources.values()))
        sources.setdefault("constant_mean", constant_cells(reference))
        sources.setdefault("shuffled_demographics", shuffled_cells(reference))

    # Discount-band response: observed authorization by discount decile.
    deciles = np.quantile(trip_discount, np.linspace(0, 1, 11))
    band = np.clip(np.searchsorted(deciles, trip_discount, side="right") - 1, 0, 9)
    band_rows = []
    for b in range(10):
        mask = band == b
        band_rows.append({
            "band": b,
            "discount_mean": float(np.mean(trip_discount[mask])),
            "observed_rate": float(np.mean(trips["shared"][mask])),
            "trips": int(mask.sum()),
        })

    report: Dict[str, object] = {
        "benchmark": "chicago_tnp_shared_trip_authorization_v1",
        "period": [POOLING_START.date().isoformat(), POOLING_END.date().isoformat()],
        "trips_used": int(trips["fare"].size),
        "pumas_used": len(codes),
        "citywide_authorization_rate": citywide_rate,
        "solo_fare_model": {"intercept": coef[0], "per_mile": coef[1], "per_minute": coef[2], "peak": coef[3]},
        "median_relative_shared_discount": relative_discount,
        "median_dollar_discount": float(np.median(trip_discount)),
        "income_gradient": {
            "pearson_observed_rate_vs_median_income": float(np.corrcoef(median_income, observed_vec)[0, 1]),
            "spearman_observed_rate_vs_median_income": spearman(median_income, observed_vec),
            "pearson_trip_adjusted_effect_vs_log_income": float(np.corrcoef(np.log(median_income), adjusted_vec)[0, 1]),
            "spearman_trip_adjusted_effect_vs_median_income": spearman(median_income, adjusted_vec),
        },
        "sources": {},
        "interpretation": (
            "Aggregate construct test only: PUMA-level authorization is an ecological "
            "outcome and pooling also carries a time cost; neither test identifies an "
            "individual rider's threshold."
        ),
    }
    puma_rows = []
    for name, cells in sources.items():
        kappa = fit_kappa(cells, compositions, discounts_by_puma, counts, citywide_rate)
        predicted = np.array([
            predicted_authorization(cells, compositions[code], discounts_by_puma[code], kappa)
            for code in codes
        ])
        band_pred = [
            predicted_authorization(cells, {"__all__": 1.0}, trip_discount[band == b], kappa)
            for b in range(10)
        ]
        for row, value in zip(band_rows, band_pred):
            row[f"predicted_{name}"] = float(value)
        # Composition-only prediction: every PUMA faces the citywide discount
        # distribution, so predicted differences come only from the source's
        # income/household -> threshold mapping.  Compared with trip-adjusted
        # PUMA effects (log-odds), this is the cleanest construct test.
        kappa_c = fit_kappa(cells, compositions, citywide_discounts, counts, citywide_rate)
        composition_only = np.array([
            predicted_authorization(cells, compositions[code], citywide_quantiles, kappa_c)
            for code in codes
        ])
        comp_logit = np.log(np.clip(composition_only, 1e-6, 1 - 1e-6) / np.clip(1 - composition_only, 1e-6, 1))
        varies = np.std(composition_only) > 1e-9
        weights = np.array([counts[c] for c in codes], dtype=float)
        rmse = float(np.sqrt(np.average((predicted - observed_vec) ** 2, weights=weights)))
        report["sources"][name] = {
            "kappa_pooling_cost_dollars": kappa,
            "pearson_predicted_vs_observed": (
                float(np.corrcoef(predicted, observed_vec)[0, 1]) if np.std(predicted) > 1e-12 else 0.0
            ),
            "spearman_predicted_vs_observed": spearman(predicted, observed_vec) if np.std(predicted) > 1e-12 else 0.0,
            "weighted_rmse": rmse,
            "predicted_rate_range": [float(predicted.min()), float(predicted.max())],
            "band_mae": float(np.mean([abs(r["observed_rate"] - r[f"predicted_{name}"]) for r in band_rows])),
            "theta_mean": float(np.mean(cells["__all__"])),
            "composition_only": {
                "pearson_logit_vs_trip_adjusted_effect": float(np.corrcoef(comp_logit, adjusted_vec)[0, 1]) if varies else 0.0,
                "spearman_vs_trip_adjusted_effect": spearman(composition_only, adjusted_vec) if varies else 0.0,
                "predicted_logit_spread": float(np.ptp(comp_logit)),
                "observed_adjusted_effect_spread": float(np.ptp(adjusted_vec)),
                "slope_observed_on_predicted_logit": (
                    float(np.polyfit(comp_logit, adjusted_vec, 1)[0]) if varies else 0.0
                ),
            },
        }
        for code, value in zip(codes, predicted):
            puma_rows.append({"source": name, "puma": code, "predicted_rate": float(value)})

    with (out / "chicago_shared_trip_benchmark.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=float)
    with (out / "chicago_puma_observed.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["puma", "trips", "observed_rate", "trip_adjusted_log_odds_effect", "median_household_income"])
        writer.writeheader()
        for code in codes:
            writer.writerow({"puma": code, "trips": counts[code], "observed_rate": observed[code],
                             "trip_adjusted_log_odds_effect": adjusted[code],
                             "median_household_income": composition[code]["median_household_income"]})
    if puma_rows:
        with (out / "chicago_puma_predicted.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(puma_rows[0]))
            writer.writeheader()
            writer.writerows(puma_rows)
    with (out / "chicago_discount_bands.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(band_rows[0]))
        writer.writeheader()
        writer.writerows(band_rows)
    print(json.dumps(report, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
