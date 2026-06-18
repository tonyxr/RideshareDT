from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Xiaoru Shi
"""
"""
End-to-end two-firm market simulator + Firm1 RL training.

Defaults aim for stable learning:
- large customers_per_step
- normalized reward + coefficient movement penalty
- clipped PPO policy/value updates
"""

import argparse
import csv
import glob
import gzip
import io
import importlib.util
import json
import os
import urllib.error
import urllib.request
import random
import time
import re
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional


# Suppress Intel oneMKL CPU deprecation warning on legacy (non-AVX) machines unless
# the user has already chosen an instruction policy in the environment.
os.environ.setdefault("MKL_ENABLE_INSTRUCTIONS", "SSE4_2")

import numpy as np
import torch


from MarketInteraction import MarketInteraction, RideContext
from Market_models import CoefficientOverrides
from GenerateAgent import GenerateAgent
from choice_models import ParametricChoiceModel, LLMChoiceModel, ChoiceResult
from pricing_models import (
    FirmMetrics,
    FirmStaticPricer,
    FirmHeuristicPricer,
    FirmMarginGuardrailPricer,
    FirmRandomWalkPricer,
    FirmRLPricer,
)
from coeff_utils import get_coeff, set_coeff
from state_encoder import build_state_vector
from calibration_utils import derive_calibration, load_calibration_preset
from gpt_threshold_utils import (
    build_threshold_profile,
    clip_price_threshold,
    diagnose_gpt_threshold_usage,
    format_gpt_threshold_usage_summary,
    increment_gpt_threshold_usage,
    is_retryable_gpt_threshold_http_status,
    new_gpt_threshold_usage_counts,
    summarize_priced_coldstart_rides,
)
from driver_supply import DriverSupplyConfig, DriverSupplyLayer

import kagglehub

try:
    import pyarrow.parquet as pq
except Exception:
    pq = None

def _resolve_dataset_path() -> str:
    """Download the Kaggle dataset only when explicitly needed."""
    dataset_path = kagglehub.dataset_download("aaronweymouth/nyc-rideshare-raw-data")
    print("Path to dataset files:", dataset_path)
    return dataset_path

def _is_missing(v: Any) -> bool:
    if v is None:
        return True
    s = str(v).strip().lower()
    return s in {"", "nan", "none", "null", "na"}


def _to_float(v: Any) -> Optional[float]:
    if _is_missing(v):
        return None
    try:
        return float(v)
    except Exception:
        return None
    
def _to_numeric(v: Any) -> Optional[float]:
    """Best-effort numeric parser for noisy dataset fields (currency, ranges, units)."""
    fv = _to_float(v)
    if fv is not None:
        return fv
    if _is_missing(v):
        return None

    text = str(v).strip().lower().replace(",", "")
    if not text:
        return None

    nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", text)]
    if not nums:
        return None

    # Fields like "$12-16" or "12 to 16" should map to a single usable value.
    if len(nums) >= 2 and ("-" in text or " to " in text):
        return float(sum(nums[:2]) / 2.0)
    return float(nums[0])


def _to_int(v: Any) -> Optional[int]:
    fv = _to_float(v)
    if fv is None:
        return None
    return int(fv)

def _to_minutes(v: Any) -> Optional[float]:
    if _is_missing(v):
        return None
    if isinstance(v, np.timedelta64):
        return float(v / np.timedelta64(1, "m"))
    fv = _to_numeric(v)
    if fv is not None:
        val = float(fv)
        # Heuristic: very large values are likely stored in ns/us/ms or sec.
        if val > 1e12:
            return val / 60_000_000_000.0
        if val > 1e9:
            return val / 60_000_000.0
        if val > 10000:
            return val / 60.0
        return val

    try:
        import pandas as pd
        td = pd.to_timedelta(v)
        return float(td.total_seconds() / 60.0)
    except Exception:
        return None


def _pick_value(row: Dict[str, Any], names: List[str]) -> Any:
    for k in names:
        if k in row and not _is_missing(row[k]):
            return row[k]
    return None


def _pick_first_parsed(
    row: Dict[str, Any],
    names: List[str],
    parser,
    with_key: bool = False,
) -> Any:
    for k in names:
        if k not in row or _is_missing(row[k]):
            continue
        val = parser(row[k])
        if val is not None:
            return (val, k) if with_key else val
    return (None, None) if with_key else None


def _pick_first_numeric(row: Dict[str, Any], names: List[str]) -> Optional[float]:
    return _pick_first_parsed(row, names, _to_numeric)


def _pick_first_minutes(row: Dict[str, Any], names: List[str]) -> Optional[float]:
    return _pick_first_parsed(row, names, _to_minutes)

def _pick_first_minutes_with_key(row: Dict[str, Any], names: List[str]) -> Tuple[Optional[float], Optional[str]]:
    val, key = _pick_first_parsed(row, names, _to_minutes, with_key=True)
    return val, key

def _pick_first_numeric_with_key(row: Dict[str, Any], names: List[str]) -> Tuple[Optional[float], Optional[str]]:
    val, key = _pick_first_parsed(row, names, _to_numeric, with_key=True)
    return val, key

def _normalize_row_keys(row: Dict[str, Any]) -> Dict[str, Any]:
    """Return a row view that supports loose key matching across datasets."""
    out = dict(row)
    for k, v in row.items():
        kn = str(k).replace("\ufeff", "").strip()
        out.setdefault(kn, v)
        out.setdefault(kn.lower(), v)
        out.setdefault(kn.replace(" ", "_"), v)
        out.setdefault(kn.lower().replace(" ", "_"), v)
        out.setdefault(kn.replace("-", "_"), v)
        out.setdefault(kn.lower().replace("-", "_"), v)
    return out

def _json_preview(obj: Any) -> str:
    """JSON preview helper that tolerates datetimes/timedeltas/decimals."""
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return str(obj)

def _discover_dataset_files(dataset_root: str, dataset_glob: str) -> List[str]:
    """Discover tabular files for dataset comparison; tolerate compressed variants."""
    base_patterns = [dataset_glob]
    g = dataset_glob.lower()
    if g == "*.csv":
        base_patterns.extend(["*.csv.gz", "*.zip"])
    elif g.endswith(".csv"):
        base_patterns.extend([dataset_glob + ".gz", dataset_glob + ".zip"])

    patterns: List[str] = []
    for pat in base_patterns:
        patterns.append(os.path.join(dataset_root, pat))
        if "**" not in pat:
            patterns.append(os.path.join(dataset_root, "**", pat))

    files: List[str] = []
    for pat in patterns:
        files.extend(glob.glob(pat, recursive=True))
    keep_ext = (".csv", ".csv.gz", ".zip", ".parquet")
    return sorted({f for f in files if os.path.isfile(f) and f.lower().endswith(keep_ext)})


def _iter_tabular_rows(fpath: str):
    """Yield rows from .csv, .csv.gz, .parquet, or .zip archives containing csv files."""
    lower = fpath.lower()
    if lower.endswith(".csv"):
        with open(fpath, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return
            yield reader.fieldnames, reader
        return

    if lower.endswith(".csv.gz"):
        with gzip.open(fpath, "rt", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return
            yield reader.fieldnames, reader
        return

    if lower.endswith(".zip"):
        with zipfile.ZipFile(fpath, "r") as zf:
            for member in zf.namelist():
                if not member.lower().endswith(".csv"):
                    continue
                with zf.open(member, "r") as raw:
                    text_stream = io.TextIOWrapper(raw, encoding="utf-8", newline="")
                    reader = csv.DictReader(text_stream)
                    if not reader.fieldnames:
                        continue
                    yield reader.fieldnames, reader
        return

    if lower.endswith(".parquet"):
        if pq is None:
            return
        parquet_file = pq.ParquetFile(fpath)
        fieldnames = parquet_file.schema.names
        for batch in parquet_file.iter_batches(batch_size=10000):
            records = batch.to_pylist()
            if not records:
                continue
            yield fieldnames, records
        return

def _infer_service_level(row: Dict[str, Any]) -> str:
    service_text = str(_pick_value(row, ["service", "name", "cab_type", "product_name", "product_id", "business", "Business"]) or "").lower()
    premium_markers = ["black", "lux", "luxury", "prem", "premium", "select", "suv"]
    return "premium" if any(tok in service_text for tok in premium_markers) else "economy"


def _infer_weather(row: Dict[str, Any]) -> str:
    text = str(_pick_value(row, ["weather", "icon", "short_summary", "long_summary", "summary"]) or "").lower()
    if "snow" in text or "sleet" in text or "blizzard" in text:
        return "snow"
    if "rain" in text or "storm" in text or "drizzle" in text:
        return "rain"
    return "clear"


def _infer_airport_trip(row: Dict[str, Any]) -> bool:
    text = " ".join(
        str(_pick_value(row, ["source", "destination", "pickup", "dropoff", "pickup_zone", "dropoff_zone"]) or "").lower().split()
    )
    airport_tokens = ["airport", "jfk", "lga", "ewr", "laguardia", "newark"]
    return any(tok in text for tok in airport_tokens)


def _infer_hour_day(row: Dict[str, Any]) -> Tuple[int, int]:
    hour = _to_int(_pick_value(row, ["hour", "pickup_hour"]))
    day = _to_int(_pick_value(row, ["day_of_week", "weekday", "day"]))

    dt_raw = _pick_value(row, ["datetime", "pickup_datetime", "time_stamp", "timestamp", "date"])
    dt: Optional[datetime] = None
    if dt_raw is not None:
        txt = str(dt_raw).strip()
        if txt.isdigit():
            sec = int(txt)
            if sec > 1_000_000_000_000:
                sec = sec // 1000
            try:
                dt = datetime.fromtimestamp(sec)
            except Exception:
                dt = None
        if dt is None:
            try:
                dt = datetime.fromisoformat(txt.replace("Z", "+00:00"))
            except Exception:
                dt = None

    if dt is not None:
        if hour is None:
            hour = int(dt.hour)
        if day is None:
            day = int(dt.weekday())

    if day is not None and day >= 1 and day <= 7:
        day = (day - 1) % 7

    return int(hour if hour is not None else 12), int(day if day is not None else 2)


def _parse_kv_floats(s: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not s:
        return out
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Bad entry '{part}'. Use key=value, e.g. base_fare=2.8")
        k, v = part.split("=", 1)
        out[k.strip()] = float(v.strip())
    return out


class Core:
    def __init__(
        self,
        market_name: str,
        seed: Optional[int] = None,
        choice_mode: str = "cognitive",
        model_name: str = "gpt-4o-mini",
        openai_api_key: Optional[str] = None,
        firm1_mode: str = "RL",
        firm2_mode: str = "static",
        firm1_static_values: str = "",
        firm2_static_values: str = "",
        total_customers_pool: int = 20000,
        deterministic_torch: bool = False,
        reward_share_weight: float = 0.34,
        reward_revenue_weight: float = 0.33,
        reward_overprice_weight: float = 0.20,
        reward_rev_scale: float = 25.0,
        reward_competitive_weight: float = 0.33,
        reward_trend_weight: float = 0.0,
        reward_profit_scale: float = 12.0,
        reward_underprice_weight: float = 0.15,
        reward_acceptable_discount: float = 2.00,
        min_profit_margin: float = 0.08,
        driver_cost_per_mile: float = 0.85,
        driver_cost_per_minute: float = 0.12,
        fixed_trip_cost: float = 1.25,
        airport_cost: float = 2.00,
        enable_driver_supply: bool = True,
        use_osmnx: bool = False,
        osmnx_place: Optional[str] = None,
        driver_base_active: int = 260,
        driver_reservation_wage: float = 24.0,
        driver_acceptance_mode: str = "expected",
        driver_state_smoothing: float = 0.35,
        driver_reward_fulfillment_weight: float = 0.06,
        driver_reward_wait_weight: float = 0.04,
        driver_reward_reject_weight: float = 0.03,
        driver_reward_warmup_fraction: float = 0.30,
        threshold_cache_path: str = "",
        reuse_threshold_cache: bool = False,
        threshold_profile_source: str = "generated",
        save_threshold_cache: bool = True,
        gpt_threshold_include_rationales: bool = False,
        gpt_threshold_coldstart_rides: int = 5,
        gpt_threshold_send_raw_rides: bool = False,
        gpt_threshold_batch_size: int = 20,
        gpt_threshold_max_retries: int = 2,
        gpt_threshold_failure_pause: float = 1.0,
    ):
        self.rng = np.random.default_rng(seed)
        self.market = MarketInteraction(city_name=market_name, seed=seed)
        self.seed = int(seed) if seed is not None else int(np.random.SeedSequence().generate_state(1)[0])
        self.total_customers_pool = int(total_customers_pool)
        
        self._seed_all_rngs(self.seed, deterministic_torch=deterministic_torch)
        self.rng = np.random.default_rng(self.seed)
        self.market = MarketInteraction(city_name=market_name, seed=self.seed)
        
        self.market.set_market(market_name)
        self.market_name = market_name

        self.agent_gen = GenerateAgent(seed=self.seed, total_customers=total_customers_pool, city_name = market_name)
        self.profile_pool_multiplier = 2
        self.model_name = model_name
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.synthetic_profile_pool: List[Dict[str, Any]] = []
        self.gpt_threshold_batch_size = int(max(1, gpt_threshold_batch_size))
        self.gpt_threshold_max_retries = int(max(0, gpt_threshold_max_retries))
        self.gpt_threshold_error_counts: Dict[str, int] = {}
        self.gpt_threshold_last_error = ""
        self.gpt_threshold_request_counts: Dict[str, int] = new_gpt_threshold_usage_counts()
        self.gpt_threshold_failure_pause = float(max(0.0, gpt_threshold_failure_pause))
        self.threshold_cache_path = str(threshold_cache_path or "")
        requested_profile_source = str(threshold_profile_source or "generated").strip().lower()
        if requested_profile_source not in {"generated", "cached"}:
            raise ValueError("threshold_profile_source must be either 'generated' or 'cached'")
        self.strict_cached_profiles = requested_profile_source == "cached"
        self.reuse_threshold_cache = bool(reuse_threshold_cache)
        # The legacy alias means "reuse if available", not "cached-only".
        # Explicit --threshold_profile_source cached retains strict behavior.
        if reuse_threshold_cache and self.threshold_cache_path:
            requested_profile_source = "cached"
        self.threshold_profile_source = requested_profile_source
        self.save_threshold_cache = bool(save_threshold_cache)
        self.gpt_threshold_include_rationales = bool(gpt_threshold_include_rationales)
        self.gpt_threshold_coldstart_rides = int(np.clip(gpt_threshold_coldstart_rides, 1, 10))
        self.gpt_threshold_send_raw_rides = bool(gpt_threshold_send_raw_rides)
        self.training_stable_window = 30
        self.training_stable_tol = 0.012
        self.reward_convergence_window = 60
        self.reward_convergence_tol = 0.018
        self.reward_trend_tol = 0.007
        self.convergence_min_days = 80
        self.convergence_required_streak = 5
        self._convergence_streak = 0
        
        # choice model
        self.choice_mode = choice_mode
        if choice_mode in {"llm", "cognitive"}:
            self.choice_model = LLMChoiceModel(model_name=model_name, api_key=self.openai_api_key, seed=self.seed)
            if self.openai_api_key is None:
                print("[Core] No OpenAI key found; GPT price-threshold bootstrapping will use deterministic fallback thresholds.")
        else:
            self.choice_model = ParametricChoiceModel(seed=self.seed)

        # firms
        self.firm1_mode = firm1_mode
        self.firm2_mode = firm2_mode

        dynamic_pricer_modes = {"heuristic", "heuristic_margin", "heuristic_random"}
        if self.firm1_mode not in {"RL", "static", *dynamic_pricer_modes}:
            raise ValueError("firm1_mode must be one of: RL, heuristic, heuristic_margin, heuristic_random, static")
        if self.firm2_mode not in {"static", *dynamic_pricer_modes}:
            raise ValueError("firm2_mode must be one of: heuristic, heuristic_margin, heuristic_random, static")

        self.opt_keys = ["base_fare", "per_minute", "per_mile", "booking_fee", "airport_fee"]
        self.shared_edit_keys = list(self.opt_keys)
        
        pricer_by_mode = {
            "heuristic": FirmHeuristicPricer,
            "heuristic_margin": FirmMarginGuardrailPricer,
            "heuristic_random": FirmRandomWalkPricer,
        }

        if self.firm1_mode == "RL":
            self.firm1 = FirmRLPricer(seed=self.seed, opt_keys=self.shared_edit_keys)
        elif self.firm1_mode in pricer_by_mode:
            self.firm1 = pricer_by_mode[self.firm1_mode](seed=self.seed, managed_keys=self.shared_edit_keys)
        else:
            self.firm1 = FirmStaticPricer()

        if self.firm2_mode in pricer_by_mode:
            self.firm2 = pricer_by_mode[self.firm2_mode](seed=self.seed + 1, managed_keys=self.shared_edit_keys)
        else:
            self.firm2 = FirmStaticPricer()
            
        print(f"Using random seed: {self.seed}")

        # initialize both firms with arbitrary starting coefficients on optimized dimensions
        self._initialize_arbitrary_starting_coefficients()


        # apply static overrides (if any)
        f1_vals = _parse_kv_floats(firm1_static_values)
        f2_vals = _parse_kv_floats(firm2_static_values)
        f1_vals, f2_vals = self._restrict_static_overrides(f1_vals, f2_vals, self.shared_edit_keys)


        for k, v in f1_vals.items():
            set_coeff(self.market.curr_market, self.firm1.overrides, k, v)
        for k, v in f2_vals.items():
            set_coeff(self.market.curr_market, self.firm2.overrides, k, v)

        # last batch summaries (optional; can be logged)
        self.airport_rate_last = self.market.airport_prob
        self.mean_distance_last = 4.0
        self.last_share = 0.5
        self.last_revpr = 0.0
        self.last_gap = 0.0
        self.last_reward = 0.0
        self.last_profitpr = 0.0
        self.last_fulfillment = 1.0
        self.last_acceptance = 1.0
        self.last_wait = 0.0
        self.last_driver_paypr = 0.0
        
        self.training_logs = []
        self.evaluation_logs = []
        
        self.run_logs = []
        self.convergence_day: Optional[int] = None
        self.convergence_window_std_at_day: Optional[float] = None
        self.convergence_delta_per_day_at_day: Optional[float] = None
        
        self.reward_share_weight = float(max(0.0, reward_share_weight))
        self.reward_revenue_weight = float(max(0.0, reward_revenue_weight))
        self.reward_overprice_weight = float(max(0.0, reward_overprice_weight))
        self.reward_underprice_weight = float(max(0.0, reward_underprice_weight))
        self.reward_acceptable_discount = float(max(0.0, reward_acceptable_discount))
        self.min_profit_margin = float(np.clip(min_profit_margin, 0.0, 0.95))
        self.reward_target_price_gap = 0.5 * self.reward_acceptable_discount
        self.driver_cost_per_mile = float(max(0.0, driver_cost_per_mile))
        self.driver_cost_per_minute = float(max(0.0, driver_cost_per_minute))
        self.fixed_trip_cost = float(max(0.0, fixed_trip_cost))
        self.airport_cost = float(max(0.0, airport_cost))
        self.enable_driver_supply = bool(enable_driver_supply)
        self.driver_supply = DriverSupplyLayer(
            seed=self.seed + 17,
            config=DriverSupplyConfig(
                base_active_drivers=int(max(1, driver_base_active)),
                reservation_wage_per_hour=float(max(1.0, driver_reservation_wage)),
                operating_cost_per_mile=self.driver_cost_per_mile,
                platform_variable_cost=self.fixed_trip_cost,
                acceptance_mode=str(driver_acceptance_mode),
                state_smoothing_alpha=float(np.clip(driver_state_smoothing, 0.0, 1.0)),
            ),
            use_osmnx=bool(use_osmnx),
            osmnx_place=osmnx_place or f"{market_name}, USA",
        )
        self.reward_rev_scale = float(max(1e-6, reward_rev_scale))
        self.reward_profit_scale = float(max(1e-6, reward_profit_scale))
        # Kept as the CLI/API name for backward compatibility; this weight is
        # folded into the simple market-share objective below.
        self.reward_competitive_weight = float(max(0.0, reward_competitive_weight))
        self.reward_trend_weight = float(max(0.0, reward_trend_weight))

        # Simple balanced reward: normalize the two decision objectives only.
        # ``reward_competitive_weight`` is kept for CLI/API compatibility and is
        # folded into the customer/market-share side instead of adding a third
        # objective that can reward dominance without profit.
        self.reward_share_weight += self.reward_competitive_weight
        denom = self.reward_share_weight + self.reward_revenue_weight
        if denom <= 0.0:
            self.reward_share_weight = 0.50
            self.reward_revenue_weight = 0.50
            denom = 1.0

        self.reward_share_weight /= denom
        self.reward_revenue_weight /= denom
        self.reward_competitive_weight = 0.0

        # In a three-option market (Firm1/Firm2/NoRide), sustainable dominance
        # starts well below 50% absolute request share.  Reward dominance as a
        # momentum-preserving advantage zone instead of only as monopoly share.
        self.reward_dominance_threshold = 0.30
        self.reward_dominance_full_credit_share = 0.45
        self.reward_trend_scale = 0.20
        # Keep the default objective stationary. Trend shaping is opt-in because
        # non-potential momentum rewards can change the optimal pricing policy
        # and keep PPO chasing short-lived deltas instead of the business target.
        self.reward_momentum_weight = self.reward_trend_weight
        self.driver_reward_fulfillment_weight = float(max(0.0, driver_reward_fulfillment_weight))
        self.driver_reward_wait_weight = float(max(0.0, driver_reward_wait_weight))
        self.driver_reward_reject_weight = float(max(0.0, driver_reward_reject_weight))
        self.driver_reward_warmup_fraction = float(np.clip(driver_reward_warmup_fraction, 0.0, 1.0))
        self.driver_reward_scale_current = 1.0 if self.driver_reward_warmup_fraction <= 0.0 else 0.0
        
        # Keep minibatches close to the default rollout size
        # (5 days x 10 synthetic steps/day) instead of silently forcing
        # full-batch PPO updates through an oversized batch setting.
        
        self.ppo_update_epochs = 5
        self.ppo_batch_size = 64
        
        print(
            "[RewardConfig] "
            f"share={self.reward_share_weight:.2f}, "
            f"profit={self.reward_revenue_weight:.2f}, "
            f"price_gap_penalty={max(self.reward_overprice_weight, self.reward_underprice_weight):.2f}, "
            f"target_gap(F2-F1)=${self.reward_target_price_gap:.2f}, "
            f"profit_scale={self.reward_profit_scale:.2f}, "
            f"dominance={self.reward_competitive_weight:.2f}, "
            f"trend={self.reward_trend_weight:.2f}, "
            f"momentum={self.reward_momentum_weight:.2f}, "
            f"driver_fulfillment={self.driver_reward_fulfillment_weight:.2f}, "
            f"driver_wait={self.driver_reward_wait_weight:.2f}, "
            f"driver_reject={self.driver_reward_reject_weight:.2f}, "
            f"driver_warmup={self.driver_reward_warmup_fraction:.2f}"
        )
        
    
    def apply_calibration(self, calibration: Dict[str, Any]) -> None:
        """Apply calibration outputs to market priors, agent priors, and choice sensitivity scales."""
        market_cal = calibration.get("market", {}) if isinstance(calibration, dict) else {}
        agent_cal = calibration.get("agent", {}) if isinstance(calibration, dict) else {}
        choice_cal = calibration.get("choice", {}) if isinstance(calibration, dict) else {}

        weather_probs = market_cal.get("weather_probs")
        if isinstance(weather_probs, dict) and weather_probs:
            keys = list(self.market.curr_market.weather_multiplier.keys())
            vals = np.array([float(weather_probs.get(k, 0.0)) for k in keys], dtype=float)
            s = float(vals.sum())
            if s > 0:
                vals = vals / s
                self.market.weather_probs = {k: float(v) for k, v in zip(keys, vals)}

        service_probs = market_cal.get("service_probs")
        if isinstance(service_probs, dict) and service_probs:
            keys = list(self.market.curr_market.service_multiplier.keys())
            vals = np.array([float(service_probs.get(k, 0.0)) for k in keys], dtype=float)
            s = float(vals.sum())
            if s > 0:
                vals = vals / s
                self.market.service_probs = {k: float(v) for k, v in zip(keys, vals)}

        airport_prob = market_cal.get("airport_prob")
        if airport_prob is not None:
            self.market.airport_prob = float(np.clip(float(airport_prob), 0.01, 0.60))

        if isinstance(agent_cal, dict) and agent_cal:
            if "age_mean" in agent_cal:
                self.agent_gen.age_mean = float(agent_cal["age_mean"])
            if "age_std" in agent_cal:
                self.agent_gen.age_std = float(max(5.0, float(agent_cal["age_std"])))
            if "household_lambda" in agent_cal:
                self.agent_gen.household_lambda = float(max(1.1, float(agent_cal["household_lambda"])))
            if "p_new" in agent_cal:
                self.agent_gen.p_new = float(np.clip(float(agent_cal["p_new"]), 0.05, 0.95))
            if "income_probs" in agent_cal and isinstance(agent_cal["income_probs"], dict):
                keys = list(self.agent_gen.income_names)
                vals = np.array([float(agent_cal["income_probs"].get(k, 0.0)) for k in keys], dtype=float)
                s = float(vals.sum())
                if s > 0:
                    self.agent_gen.income_probs = vals / s
            self.agent_gen._build_population()

        if hasattr(self.choice_model, "apply_calibration") and isinstance(choice_cal, dict):
            self.choice_model.apply_calibration(choice_cal)
    
    @staticmethod
    def _restrict_static_overrides(
        f1_vals: Dict[str, float],
        f2_vals: Dict[str, float],
        shared_keys: List[str],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Force both firms to edit the same coefficient set and cap Firm2 to Firm1 scope."""
        allowed = set(shared_keys)

        f1_filtered = {k: v for k, v in f1_vals.items() if k in allowed}
        dropped_f1 = [k for k in f1_vals.keys() if k not in allowed]
        if dropped_f1:
            print(f"[Core] Ignoring Firm1 static keys outside shared edit set: {dropped_f1}")

        f2_allowed = set(f1_filtered.keys())
        f2_filtered = {k: v for k, v in f2_vals.items() if k in f2_allowed}
        dropped_f2 = [k for k in f2_vals.keys() if k not in f2_allowed]
        if dropped_f2:
            print(f"[Core] Ignoring Firm2 static keys not edited by Firm1: {dropped_f2}")

        return f1_filtered, f2_filtered
    
    @staticmethod
    def _seed_all_rngs(seed: int, deterministic_torch: bool = False) -> None:
        """Seed Python/NumPy/Torch RNGs for reproducible simulations."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Optional stricter mode for deterministic Torch kernels.
        if deterministic_torch:
            torch.use_deterministic_algorithms(True)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.benchmark = False

    @staticmethod
    def _ema(curr: float, prev: float, alpha: float = 0.2) -> float:
        return (1.0 - alpha) * float(prev) + alpha * float(curr)
    
    @staticmethod
    def estimate_duration(miles: float, hod: int) -> float:
        mph = 18.0 if (7 <= hod < 10 or 16 <= hod < 19) else 25.0
        return max(5.0, 60.0 * miles / max(8.0, mph))
    
    def _initialize_arbitrary_starting_coefficients(self) -> None:
        """Set both firms to arbitrary (seeded-random) initial values for optimized coefficients."""
        base = self.market.curr_market
        lo, hi = 0.85, 1.15

        f1_base = float(base.base_fare) * float(self.rng.uniform(lo, hi))
        f1_pmin = float(base.per_minute) * float(self.rng.uniform(lo, hi))
        f1_pmile = float(base.per_mile) * float(self.rng.uniform(lo, hi))
        f1_book = float(base.booking_fee) * float(self.rng.uniform(lo, hi))
        f1_air = float(base.airport_fee) * float(self.rng.uniform(lo, hi))
        f2_base = float(base.base_fare) * float(self.rng.uniform(lo, hi))
        f2_pmin = float(base.per_minute) * float(self.rng.uniform(lo, hi))
        f2_pmile = float(base.per_mile) * float(self.rng.uniform(lo, hi))
        f2_book = float(base.booking_fee) * float(self.rng.uniform(lo, hi))
        f2_air = float(base.airport_fee) * float(self.rng.uniform(lo, hi))

        self.firm1.overrides.base_fare = max(0.1, f1_base)
        self.firm1.overrides.per_minute = max(0.01, f1_pmin)
        self.firm1.overrides.per_mile = max(0.01, f1_pmile)
        self.firm1.overrides.booking_fee = max(0.0, f1_book)
        self.firm1.overrides.airport_fee = max(0.0, f1_air)
        self.firm2.overrides.base_fare = max(0.1, f2_base)
        self.firm2.overrides.per_minute = max(0.01, f2_pmin)
        self.firm2.overrides.per_mile = max(0.01, f2_pmile)
        self.firm2.overrides.booking_fee = max(0.0, f2_book)
        self.firm2.overrides.airport_fee = max(0.0, f2_air)
        
    def _estimate_trip_cost(self, distance_miles: float, duration_minutes: float, airport: bool) -> float:
        """Approximate platform contribution cost for a completed ride."""
        return float(
            self.fixed_trip_cost
            + self.driver_cost_per_mile * max(0.0, float(distance_miles))
            + self.driver_cost_per_minute * max(0.0, float(duration_minutes))
            + (self.airport_cost if airport else 0.0)
        )

    @staticmethod
    def _profit_margin(metrics: FirmMetrics) -> float:
        if metrics.revenue <= 0.0:
            return 0.0
        return float(metrics.profit / max(metrics.revenue, 1e-6))
        
    @staticmethod
    def _smooth_positive_penalty(value: float, scale: float) -> float:
        """Smoothly map a non-negative violation to [0, 1) without hard clipping."""
        x = max(0.0, float(value))
        s = max(1e-6, float(scale))
        y = x / (x + s)
        return float(y * y)
    
    @staticmethod
    def _finite_float(value: float, default: float = 0.0) -> float:
        """Return a scalar float with NaN/infinite values mapped to a safe default."""
        try:
            out = float(value)
        except (TypeError, ValueError):
            return float(default)
        return out if np.isfinite(out) else float(default)

    def _reward_components(
        self,
        share: float,
        rev_per_request: float,
        price_gap_f2_minus_f1: float = 0.0,
        profit_per_request: Optional[float] = None,
        profit_margin: Optional[float] = None,
        fulfillment_rate: float = 1.0,
        avg_wait_minutes: float = 0.0,
        driver_acceptance_rate: float = 1.0,
    ) -> Dict[str, float]:
        """Return a small, robust reward for the customer-driver-firm loop.

        The reward intentionally has two primary terms:
        1. **Market term**: completed customer share, so riders choosing the firm
           only counts when driver supply can fulfill those rides.
        2. **Profit term**: signed profit per request, so losses are penalized and
           profitable operation is rewarded.

        This avoids corner-case incentives where the agent can maximize share by
        destructive underpricing, maximize margin with no customers, or improve
        driver acceptance by raising prices without serving demand.  All inputs
        are clipped/sanitized before use so empty batches, NaNs, infinities, zero
        revenue, and extreme prices still produce a finite reward in [-1, 1].
        """
        
        share_f = float(np.clip(self._finite_float(share), 0.0, 1.0))
        profitpr = self._finite_float(
            rev_per_request if profit_per_request is None else profit_per_request
        )
        margin = self._finite_float(profit_margin)
        gap = self._finite_float(price_gap_f2_minus_f1)
        fulfillment_term = float(np.clip(self._finite_float(fulfillment_rate, 1.0), 0.0, 1.0))
        # FirmMetrics.share is already completed rides / all requests. Do not
        # multiply it by fulfillment again: that previously squared supply
        # failures and overwhelmed both the market signal and PPO advantages.
        served_share = share_f
        market_term = float(np.tanh((served_share - self.reward_dominance_threshold) / 0.15))

        # Firm objective: signed and bounded contribution profit per request.
        profit_term = float(np.tanh(profitpr / self.reward_profit_scale))
        # Use one symmetric price-gap objective. A positive gap means Firm1 is
        # cheaper than Firm2, so the default target is a modest $1 discount when
        # the acceptable discount is $2. This gives the agent one clear target
        # instead of separate and potentially conflicting over/underprice rules.
        price_gap_deviation = float(gap - self.reward_target_price_gap)
        price_gap_penalty = self._smooth_positive_penalty(
            abs(price_gap_deviation), scale=max(1.0, self.reward_acceptable_discount)
        )
        margin_shortfall = float(
            max(0.0, self.min_profit_margin - margin) / max(self.min_profit_margin, 1e-6)
        )
        unfulfilled_penalty = self._smooth_positive_penalty(0.75 - fulfillment_term, scale=0.35)
        margin_penalty = self._smooth_positive_penalty(margin_shortfall, scale=1.0)
        price_discipline_penalty = max(price_gap_penalty, margin_penalty)
        # The average is deliberately the complete business objective: completed
        # customer share captures riders and driver supply, while signed
        # contribution profit captures the firm and driver compensation.  The
        # gap term is only price discipline, not a competing business objective.
        share_component = self.reward_share_weight * market_term
        profit_component = self.reward_revenue_weight * profit_term
        price_gap_weight = max(self.reward_overprice_weight, self.reward_underprice_weight)
        price_gap_component = price_gap_weight * price_discipline_penalty

        driver_scale = float(np.clip(self.driver_reward_scale_current, 0.0, 1.0))
        fulfillment_component = (
            driver_scale
            * self.driver_reward_fulfillment_weight
            * (2.0 * fulfillment_term - 1.0)
        )
        wait_penalty = (
            driver_scale
            * self.driver_reward_wait_weight
            * self._smooth_positive_penalty(max(0.0, avg_wait_minutes - 5.0), scale=5.0)
        )
        rejection_penalty = (
            driver_scale
            * self.driver_reward_reject_weight
            * (1.0 - float(np.clip(driver_acceptance_rate, 0.0, 1.0)))
        )

        raw = (
            share_component
            + profit_component
            + fulfillment_component
            - price_gap_component
            - wait_penalty
            - rejection_penalty
        )
        base = float(np.clip(raw, -1.0, 1.0))
        return {
            "reward_base": base,
            "reward_base_unclipped": float(raw),
            "reward_share_component": float(share_component),
            "reward_profit_component": float(profit_component),
            "reward_margin_penalty": float(margin_penalty),
            "reward_price_discipline_penalty": float(price_discipline_penalty),
            "reward_price_gap": float(gap),
            "reward_target_price_gap": float(self.reward_target_price_gap),
            "reward_price_gap_deviation": float(price_gap_deviation),
            "reward_price_gap_penalty": float(price_gap_penalty),
            "reward_fulfillment_term": float(fulfillment_term),
            "reward_profit_term": float(profit_term),
            "reward_served_share": float(served_share),
            "reward_driver_unfulfilled_penalty": float(unfulfilled_penalty),
            "reward_driver_fulfillment_component": float(fulfillment_component),
            "reward_driver_wait_penalty": float(wait_penalty),
            "reward_driver_rejection_penalty": float(rejection_penalty),
        }

    def _reward_base(
        self,
        share: float,
        rev_per_request: float,
        price_gap_f2_minus_f1: float = 0.0,
        profit_per_request: Optional[float] = None,
        profit_margin: Optional[float] = None,
        fulfillment_rate: float = 1.0,
        avg_wait_minutes: float = 0.0,
        driver_acceptance_rate: float = 1.0,
    ) -> float:
        """Balanced business reward: completed market share plus signed profit."""
        return float(
            self._reward_components(
                share=share,
                rev_per_request=rev_per_request,
                price_gap_f2_minus_f1=price_gap_f2_minus_f1,
                profit_per_request=profit_per_request,
                profit_margin=profit_margin,
                fulfillment_rate=fulfillment_rate,
                avg_wait_minutes=avg_wait_minutes,
                driver_acceptance_rate=driver_acceptance_rate,
            )["reward_base"]
        )

    def _reward_diagnostics(
        self,
        share: float,
        rev_per_request: float,
        mean_gap: float,
        prev_share: float,
        prev_rev_per_request: float,
        prev_profit_per_request: Optional[float] = None,
        prev_gap: Optional[float] = None,
        profit_per_request: Optional[float] = None,
        profit_margin: Optional[float] = None,
        fulfillment_rate: float = 1.0,
        avg_wait_minutes: float = 0.0,
        driver_acceptance_rate: float = 1.0,
    ) -> Dict[str, float]:
        """Return shaped reward and component diagnostics for trajectory analysis."""
        share_f = float(np.clip(share, 0.0, 1.0))
        revpr = float(max(0.0, rev_per_request))
        profitpr = float(revpr if profit_per_request is None else profit_per_request)
        margin = float(0.0 if profit_margin is None else profit_margin)
        gap = self._finite_float(mean_gap)
        
        components = self._reward_components(
            share=share_f,
            rev_per_request=revpr,
            price_gap_f2_minus_f1=gap,
            profit_per_request=profitpr,
            profit_margin=margin,
            fulfillment_rate=fulfillment_rate,
            avg_wait_minutes=avg_wait_minutes,
            driver_acceptance_rate=driver_acceptance_rate,
        )
        base_reward = float(components["reward_base"])
        
        dominance_width = max(1e-6, self.reward_dominance_full_credit_share - self.reward_dominance_threshold)
        dominance_term = float(np.clip((share_f - self.reward_dominance_threshold) / dominance_width, 0.0, 1.0))
        share_delta = float(np.clip(share_f - float(prev_share), -0.20, 0.20) / 0.20)
        rev_delta = float(np.clip((revpr - float(prev_rev_per_request)) / self.reward_profit_scale, -0.20, 0.20) / 0.20)
        prev_profit = float(profitpr if prev_profit_per_request is None else prev_profit_per_request)
        profit_delta = float(np.clip((profitpr - prev_profit) / self.reward_profit_scale, -0.20, 0.20) / 0.20)
        prev_gap_f = float(gap if prev_gap is None else prev_gap)
        # Positive gap movement usually means Firm1 is becoming less overpriced or
        # more strategically discounted.  Cap it because extreme discounts are
        # already penalized by the base reward.
        gap_delta = float(np.clip((gap - prev_gap_f) / 2.0, -1.0, 1.0))
        trend_term = float(0.40 * share_delta + 0.35 * profit_delta + 0.25 * gap_delta)
        pricing_discipline = float(
            1.0 - 2.0 * components["reward_price_gap_penalty"]
        )
        efficiency_term = float(0.5 * trend_term + 0.5 * pricing_discipline)
        momentum_component = float(self.reward_momentum_weight * self.reward_trend_scale * trend_term)
        raw_reward = float(base_reward + momentum_component)
        # The base terms are already smoothly bounded. A second tanh compressed
        # useful differences between good and excellent outcomes and weakened
        # PPO's advantage signal without changing safety.
        reward = float(np.clip(raw_reward, -1.0, 1.0))

        return {
            "reward": reward,
            "reward_raw": raw_reward,
            "reward_base": float(base_reward),
            "reward_dominance_term": dominance_term,
            # Backward-compatible diagnostic column name used by existing plots/CSVs.
            "reward_competitive_term": dominance_term,
            "reward_share_delta": share_delta,
            "reward_rev_delta": rev_delta,
            "reward_trend_term": trend_term,
            "reward_profit_delta": profit_delta,
            "reward_gap_delta": gap_delta,
            "reward_momentum_component": momentum_component,
            "reward_pricing_discipline": pricing_discipline,
            "reward_efficiency_term": efficiency_term,
            "reward_profit_per_request": profitpr,
            "reward_profit_margin": margin,
            "reward_fulfillment_rate": float(np.clip(fulfillment_rate, 0.0, 1.0)),
            "reward_avg_wait_minutes": float(avg_wait_minutes),
            "reward_driver_acceptance_rate": float(np.clip(driver_acceptance_rate, 0.0, 1.0)),
            **components,
        }

    def _compute_rl_reward(self, m1: FirmMetrics, mean_gap: float) -> float:
        """Low-complexity reward shaping for better PPO learning signal."""
        diagnostics = self._reward_diagnostics(
            share=float(m1.share),
            rev_per_request=float(m1.rev_per_request),
            mean_gap=float(mean_gap),
            profit_per_request=float(m1.profit_per_request),
            profit_margin=self._profit_margin(m1),
            fulfillment_rate=float(m1.fulfillment_rate),
            avg_wait_minutes=float(m1.avg_wait_minutes),
            driver_acceptance_rate=float(m1.driver_acceptance_rate),
            prev_share=float(self.last_share),
            prev_rev_per_request=float(self.last_revpr),
            prev_profit_per_request=float(self.last_profitpr),
            prev_gap=float(self.last_gap),
        )
        reward = float(diagnostics["reward"])
        self.last_reward = reward
        return reward
    
    def _initialize_run_distributions(self) -> None:
        """Run-level slight variations to demographics/weather/ride nature priors."""
        self.agent_gen.apply_probability_variation(jitter_scale=0.05)
        self.market.refresh_run_probabilities(jitter_scale=0.05)

    def _refresh_profile_pool(self, rides_per_timestep: int) -> None:
        """Generate synthetic customer profiles at t=0 with minimum 2x timestep demand."""
        min_pool = int(max(self.total_customers_pool, self.profile_pool_multiplier * int(rides_per_timestep)))
        if self.agent_gen.total_customers != min_pool:
            self.agent_gen.total_customers = min_pool
        self.agent_gen.apply_probability_variation(jitter_scale=0.05)
        self._bootstrap_synthetic_profiles(pool_size=min_pool)

    def _build_coldstart_rides(self, n: int = 10) -> List[Dict[str, Any]]:
        """Generate rider-specific cold-start rides before simulation begins."""
        out: List[Dict[str, Any]] = []
        for _ in range(int(max(1, n))):
            r = self.market.generate_ride()
            out.append({
                "Hour": int(r.hour),
                "Weather": str(r.weather),
                "DistanceMiles": float(r.distance_miles),
                "DurationMinutes": float(r.duration_minutes),
                "Service": str(r.service),
                "Airport": bool(r.airport),
                "DayOfWeek": int(r.day_of_week),
            })
        return out

    @staticmethod
    def _extract_response_text(payload: Dict[str, Any]) -> str:
        """Best-effort extraction from Responses API payload variants."""
        if isinstance(payload.get("output_text"), str) and payload.get("output_text"):
            return str(payload["output_text"])
        output = payload.get("output")
        if not isinstance(output, list):
            return ""
        chunks: List[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            for content in item.get("content", []) or []:
                if isinstance(content, dict):
                    txt = content.get("text")
                    if isinstance(txt, str) and txt:
                        chunks.append(txt)
        return "\n".join(chunks).strip()
    
    @staticmethod
    def _parse_response_json(text: str) -> Dict[str, Any]:
        """Parse model JSON, tolerating simple markdown code fences."""
        cleaned = str(text or "").strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned).strip()
        return json.loads(cleaned) if cleaned else {}
    
    @staticmethod
    def _gpt_threshold_max_output_tokens(batch_size: int, include_rationales: bool = True) -> int:
        """Return an output-token budget large enough for strict JSON batch responses."""
        # Rationale-free production mode keeps output small; debug mode budgets more
        # tokens because free-text rationales can otherwise truncate JSON responses.
        per_profile = 250 if include_rationales else 70
        floor = 1200 if include_rationales else 500
        return int(max(floor, per_profile * int(max(1, batch_size))))

    @staticmethod
    def _fallback_price_threshold(profile: Dict[str, Any], priced_rides: List[Dict[str, Any]]) -> float:
        """Estimate the fare-gap threshold at which price becomes salient for a rider."""
        income_score = {"<50k": 0.0, "50k-100k": 0.3, "100k-200k": 0.7, "200k+": 1.0}.get(
            str(profile.get("IncomeBracket", "50k-100k")),
            0.3,
        )
        household = int(profile.get("HouseholdSize", 1) or 1)
        loyalty_strength = float(profile.get("LoyaltyStrength", 0.0) or 0.0)
        is_returning = str(profile.get("LoyaltyType", "New")) == "Returning"
        mean_fare = 15.0
        if priced_rides:
            fares = [float(r.get("firm1_price", 0.0)) for r in priced_rides] + [float(r.get("firm2_price", 0.0)) for r in priced_rides]
            valid_fares = [fare for fare in fares if fare > 0.0]
            if valid_fares:
                mean_fare = float(np.mean(valid_fares))

        threshold = 0.55 + 1.65 * income_score + 0.04 * mean_fare
        threshold -= 0.18 * min(max(household - 1, 0), 3)
        threshold += (0.35 + 0.45 * loyalty_strength) if is_returning else 0.0
        return clip_price_threshold(threshold)

    def _price_coldstart_rides(self, rides: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Attach current firm prices to cold-start ride contexts."""
        priced: List[Dict[str, Any]] = []
        for r in rides[: self.gpt_threshold_coldstart_rides]:
            rc = RideContext(
                day_of_week=int(r["DayOfWeek"]),
                weather=str(r["Weather"]),
                hour=int(r["Hour"]),
                airport=bool(r["Airport"]),
                service=str(r["Service"]),
            )
            p1 = float(self.market.quote_price(float(r.get("DistanceMiles", 0.0)), float(r.get("DurationMinutes", 15.0)), rc, overrides=self.firm1.overrides))
            p2 = float(self.market.quote_price(float(r.get("DistanceMiles", 0.0)), float(r.get("DurationMinutes", 15.0)), rc, overrides=self.firm2.overrides))
            priced.append({**r, "firm1_price": p1, "firm2_price": p2})
        return priced

    def _threshold_fallback_result(self, profile: Dict[str, Any], priced_rides: List[Dict[str, Any]], rationale: str = "Deterministic fallback price threshold (no API response).") -> Dict[str, Any]:
        return {
            "price_threshold": self._fallback_price_threshold(profile, priced_rides),
            "rationale": rationale,
            "source": "fallback",
        }
        
        
    def _record_gpt_threshold_error(self, reason: str, detail: str = "") -> None:
        """Track GPT threshold failures without flooding logs during large bootstraps."""
        key = str(reason or "unknown")
        self.gpt_threshold_error_counts[key] = int(self.gpt_threshold_error_counts.get(key, 0) + 1)
        if self.gpt_threshold_error_counts[key] <= 5:
            suffix = f": {detail}" if detail else ""
            print(f"[GPT threshold fallback] {key}{suffix}")

    def _gpt_threshold_schema(self) -> Dict[str, Any]:
        """Responses API structured-output schema for one aggregate threshold per profile."""
        item_properties = {
            "profile_index": {"type": "integer"},
            "price_threshold": {"type": "number", "minimum": 0.50, "maximum": 5.00},
        }
        required = ["profile_index", "price_threshold"]
        if self.gpt_threshold_include_rationales:
            item_properties["rationale"] = {"type": "string"}
            required.append("rationale")
        return {
            "type": "json_schema",
            "name": "price_threshold_batch",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "thresholds": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": item_properties,
                            "required": required,
                        },
                    }
                },
                "required": ["thresholds"],
            },
        }

    def _gpt_batch_price_thresholds(self, batch: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]]) -> List[Dict[str, Any]]:
        """
        Infer price thresholds for multiple profiles in one API request.

        Batching reduces the default 20,000-profile bootstrap from 20,000 API calls
        to roughly pool_size / gpt_threshold_batch_size calls, improving throughput
        and reducing rate-limit pressure while keeping deterministic fallback safety.
        """
        prepared: List[Dict[str, Any]] = []
        results: List[Dict[str, Any]] = []
        increment_gpt_threshold_usage(self.gpt_threshold_request_counts, "batches_total")
        increment_gpt_threshold_usage(self.gpt_threshold_request_counts, "profiles_requested", len(batch))
        for i, (profile, rides) in enumerate(batch):
            priced = self._price_coldstart_rides(rides)
            evidence = {
                "profile_index": i,
                "profile": profile,
                "coldstart_ride_summary": summarize_priced_coldstart_rides(priced[: self.gpt_threshold_coldstart_rides]),
                "instruction": (
                    "Aggregate all coldstart_rides for this one profile into exactly one profile-level threshold. "
                    "Use the summary distribution and profile attributes to infer the smallest fare gap that would make "
                    "price a primary decision factor for this rider across similar rides; do not answer separately per ride."
                ),
            }
            if self.gpt_threshold_send_raw_rides:
                evidence["coldstart_rides"] = priced[: self.gpt_threshold_coldstart_rides]
            prepared.append(evidence)
            results.append(self._threshold_fallback_result(profile, priced))

        if not prepared:
            return results
        
        if not self.openai_api_key:
            increment_gpt_threshold_usage(self.gpt_threshold_request_counts, "batches_skipped_no_key")
            increment_gpt_threshold_usage(self.gpt_threshold_request_counts, "profiles_fallback", len(results))
            return results

        increment_gpt_threshold_usage(self.gpt_threshold_request_counts, "batches_attempted")

        prompt = {
            "task": (
                "Infer exactly one aggregate rider-level price threshold for each profile item. The threshold is the "
                "smallest dollar fare gap at which that customer starts treating price as a primary decision factor, "
                f"after considering the profile and all {self.gpt_threshold_coldstart_rides} coldstart ride summaries together. The coldstart evidence is "
                "for one profile-level judgment; it is not a set of separate threshold requests."
            ),
            "calibration_guide": {
                "do": [
                    "Use profile economics first, then coldstart price-gap distribution as grounding evidence.",
                    "Lower thresholds for lower income, larger households, new/no-loyalty riders, and repeated small gaps.",
                    "Raise thresholds for higher income, strong loyalty to either firm, airport/rush/weather urgency, and premium-heavy rides.",
                    "Pick a continuous dollar value to the nearest $0.25 when useful; do not default to only integers or half-dollars.",
                    "If rationale is requested, cite aggregate profile signals and coldstart summary; avoid generic phrases.",
                ],
                "do_not": [
                    "Do not copy the mean_absolute_price_gap as the threshold unless the profile evidence specifically supports it.",
                    "Do not use max_absolute_price_gap as the threshold; max gaps are only upper-bound evidence.",
                    "Do not make one threshold per ride or duplicate a profile_index.",
                ],
                "interpretation": (
                    "A threshold below the observed mean gap means price becomes primary even for modest differences; "
                    "a threshold above the mean gap means brand, convenience, or context often outweighs price until gaps are larger."
                ),
            },
            "profiles": prepared,
            "output_constraints": {
                "format": "json_only",
                "one_threshold_per_profile": True,
                "forbidden": "Do not return per-ride thresholds. Do not create more than one row for the same profile_index.",
                "expected_threshold_count": len(prepared),
                "price_threshold_range_usd": [0.50, 5.00],
                "response_shape": (
                    {"thresholds": [{"profile_index": 0, "price_threshold": 2.25, "rationale": "short aggregate explanation using profile and summary evidence"}]}
                    if self.gpt_threshold_include_rationales
                    else {"thresholds": [{"profile_index": 0, "price_threshold": 2.25}]}
                ),
            },
        }
        payload = {
            "model": self.model_name,
            "input": json.dumps(prompt),
            "max_output_tokens": self._gpt_threshold_max_output_tokens(len(prepared), include_rationales=self.gpt_threshold_include_rationales),
            "text": {"format": self._gpt_threshold_schema()},
        }

        last_error = ""
        for attempt in range(self.gpt_threshold_max_retries + 1):
            try:
                req = urllib.request.Request(
                    "https://api.openai.com/v1/responses",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Authorization": f"Bearer {self.openai_api_key}", "Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=45) as resp:
                    response_payload = json.loads(resp.read().decode("utf-8"))

                text = self._extract_response_text(response_payload)
                parsed = self._parse_response_json(text)
                threshold_rows = parsed.get("thresholds", [])
                if not isinstance(threshold_rows, list):
                    raise ValueError("Response JSON did not contain a thresholds array")

                seen_indices = set()
                duplicate_indices = 0
                for item in threshold_rows:
                    if not isinstance(item, dict):
                        continue
                    idx_raw = item.get("profile_index", item.get("index", -1))
                    try:
                        idx = int(idx_raw)
                    except (TypeError, ValueError):
                        continue
                    if idx < 0 or idx >= len(results):
                        continue
                    if idx in seen_indices:
                        duplicate_indices += 1
                        continue
                    raw_threshold = item.get("price_threshold", item.get("price_threshold_usd"))
                    if raw_threshold is None:
                        continue
                    results[idx] = {
                        "price_threshold": clip_price_threshold(raw_threshold),
                        "rationale": str(item.get("rationale", "")),
                        "source": "gpt",
                    }
                    seen_indices.add(idx)
                seen = len(seen_indices)
                if duplicate_indices:
                    self._record_gpt_threshold_error("duplicate_profile_thresholds", f"duplicates={duplicate_indices}")
                if seen == 0:
                    raise ValueError("Response JSON contained no usable threshold rows")
                if seen < len(results):
                    increment_gpt_threshold_usage(self.gpt_threshold_request_counts, "batches_partial")
                    self._record_gpt_threshold_error("partial_batch", f"usable={seen}/{len(results)}")
                else:
                    increment_gpt_threshold_usage(self.gpt_threshold_request_counts, "batches_succeeded")
                increment_gpt_threshold_usage(self.gpt_threshold_request_counts, "profiles_gpt", seen)
                increment_gpt_threshold_usage(self.gpt_threshold_request_counts, "profiles_fallback", len(results) - seen)
                return results
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", errors="replace")[:300]
                last_error = f"HTTP {e.code} {detail}"
                if not is_retryable_gpt_threshold_http_status(e.code) or attempt >= self.gpt_threshold_max_retries:
                    break
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                if attempt >= self.gpt_threshold_max_retries:
                    break
            time.sleep(float(min(8.0, 1.5 * (2 ** attempt))))
        
        self.gpt_threshold_last_error = last_error
        increment_gpt_threshold_usage(self.gpt_threshold_request_counts, "batches_failed")
        increment_gpt_threshold_usage(self.gpt_threshold_request_counts, "profiles_fallback", len(results))
        self._record_gpt_threshold_error("batch_failed", last_error)
        return results

    def _gpt_profile_price_threshold(self, profile: Dict[str, Any], rides: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compatibility wrapper for callers that need a single threshold."""
        return self._gpt_batch_price_thresholds([(profile, rides)])[0]
    
    def _try_load_threshold_cache(self, pool_size: int) -> bool:
        """Load cached threshold-enriched profiles when cached profile mode is requested."""
        path = self.threshold_cache_path
        if self.threshold_profile_source != "cached" and not self.reuse_threshold_cache:
            return False
        if not path:
            if self.reuse_threshold_cache and not self.strict_cached_profiles:
                print("[ThresholdCache] --reuse_threshold_cache ignored because no --threshold_cache_path was supplied.")
                return False
            raise ValueError("threshold_profile_source='cached' requires --threshold_cache_path")
        if not os.path.exists(path):
            if self.reuse_threshold_cache and not self.strict_cached_profiles:
                print(f"[ThresholdCache] Cache not found at {path}; generating and saving a new pool.")
                self.threshold_profile_source = "generated"
                return False
            raise FileNotFoundError(f"Threshold profile cache not found: {path}")
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "PriceThreshold" not in row:
                    continue
                row["PriceThreshold"] = clip_price_threshold(row["PriceThreshold"])
                rows.append(row)
                if len(rows) >= int(pool_size):
                    break
        if len(rows) < int(pool_size):
            raise ValueError(
                f"Threshold profile cache has {len(rows)} usable profiles; need {int(pool_size)}. "
                "Use threshold_profile_source='generated' to create a completely new profile pool, "
                "or provide a larger cache for threshold_profile_source='cached'."
            )
        self.synthetic_profile_pool = rows[: int(pool_size)]
        self.gpt_threshold_request_counts = new_gpt_threshold_usage_counts()
        print(f"[ThresholdCache] Loaded {len(self.synthetic_profile_pool)} cached threshold profiles from {path}.")
        return True

    def _save_threshold_cache(self) -> None:
        """Persist threshold-enriched profiles as JSONL for fast repeated experiments."""
        path = self.threshold_cache_path
        if not (self.save_threshold_cache and path and self.synthetic_profile_pool):
            return
        _ensure_parent_dir(path)
        with open(path, "w", encoding="utf-8") as f:
            for row in self.synthetic_profile_pool:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        print(f"[ThresholdCache] Saved {len(self.synthetic_profile_pool)} threshold profiles -> {path}")

    def _bootstrap_synthetic_profiles(self, pool_size: int) -> None:
        if self._try_load_threshold_cache(pool_size):
            return
        if self.threshold_profile_source != "generated":
            raise ValueError(f"Unsupported threshold_profile_source={self.threshold_profile_source!r}")
        base_profiles = self.agent_gen.sample_profiles(int(max(1, pool_size)))
        self.synthetic_profile_pool = []
        threshold_source_counts: Dict[str, int] = {}
        self.gpt_threshold_error_counts = {}
        self.gpt_threshold_last_error = ""
        self.gpt_threshold_request_counts = new_gpt_threshold_usage_counts()
        batch_size = int(max(1, self.gpt_threshold_batch_size))
        for start in range(0, len(base_profiles), batch_size):
            profile_batch = list(base_profiles[start:start + batch_size])
            rides_batch = [(p, self._build_coldstart_rides(n=self.gpt_threshold_coldstart_rides)) for p in profile_batch]
            failed_before = int(self.gpt_threshold_request_counts.get("batches_failed", 0))
            thresholds = self._gpt_batch_price_thresholds(rides_batch)
            failed_after = int(self.gpt_threshold_request_counts.get("batches_failed", 0))
            for (p, coldstart_rides), threshold in zip(rides_batch, thresholds):
                src = str(threshold.get("source", "fallback"))
                threshold_source_counts[src] = int(threshold_source_counts.get(src, 0) + 1)
                self.synthetic_profile_pool.append(build_threshold_profile(p, coldstart_rides, threshold))
            transient_error = any(
                token in self.gpt_threshold_last_error
                for token in ("RemoteDisconnected", "Connection refused", "ConnectionReset", "Timeout", "timed out", "URLError")
            )
            if (
                self.openai_api_key
                and failed_after > failed_before
                and transient_error
                and self.gpt_threshold_failure_pause > 0.0
            ):
                time.sleep(self.gpt_threshold_failure_pause)

        total = int(sum(threshold_source_counts.values()))
        summary_parts = []
        for k, v in sorted(threshold_source_counts.items()):
            pct = 100.0 * float(v) / float(max(1, total))
            summary_parts.append(f"{k}={v} ({pct:.1f}%)")
        print(
            f">>> Price-threshold bootstrap sources (batch_size={batch_size}): "
            + ", ".join(summary_parts)
        )
        request_summary = format_gpt_threshold_usage_summary(self.gpt_threshold_request_counts)
        if request_summary:
            print(f">>> GPT threshold API utilization: {request_summary}")
        diagnostic_notes = diagnose_gpt_threshold_usage(
            self.gpt_threshold_request_counts,
            self.gpt_threshold_error_counts,
            self.gpt_threshold_max_retries,
        )
        if diagnostic_notes:
            print(">>> GPT threshold diagnosis: " + "; ".join(diagnostic_notes))
        if self.gpt_threshold_error_counts:
            print(
                ">>> GPT threshold fallback errors: "
                + ", ".join(f"{k}={v}" for k, v in sorted(self.gpt_threshold_error_counts.items()))
            )
        self._save_threshold_cache()

    def _sample_profiles_from_pool(self, n: int) -> List[Dict[str, Any]]:
        if not self.synthetic_profile_pool:
            self._bootstrap_synthetic_profiles(pool_size=max(self.total_customers_pool, n))
        idx = self.rng.integers(0, len(self.synthetic_profile_pool), size=int(max(0, n)))
        # Simulation and choice models treat profiles as read-only. Reusing the
        # pool dictionaries avoids thousands of allocations per timestep.
        return [self.synthetic_profile_pool[int(i)] for i in idx]

    def run_experiment(
        self,
        train_timesteps: int = 1000,
        train_customers_per_step: int = 5000,
        eval_timesteps: int = 200,
        eval_customers_per_step: int = 1000,
        profiles_out: Optional[str] = None,
        profiles_log_limit: int = 200000,
        train_steps_per_day: int = 10,
        ppo_update_interval_days: int = 5,
        stochastic_training: bool = True,
    ):
        """Run workflow: synthetic-data RL training (day/timestep cadence), then held-out evaluation."""
        self._initialize_run_distributions()
        self._refresh_profile_pool(rides_per_timestep=train_customers_per_step)
        
        if stochastic_training:
            exp_seed = int(np.random.SeedSequence().generate_state(1)[0])
            self._seed_all_rngs(exp_seed)
            self.rng = np.random.default_rng(exp_seed)
            self.market = MarketInteraction(city_name=self.market_name, seed=exp_seed)
            self.market.set_market(self.market_name)
            self.agent_gen = GenerateAgent(seed=exp_seed, total_customers=self.total_customers_pool, city_name=self.market_name)
            print(f">>> run_experiment stochastic seed={exp_seed} (independent of --seed for robustness)")
            self._initialize_arbitrary_starting_coefficients()
            self._initialize_run_distributions()
            self._refresh_profile_pool(rides_per_timestep=train_customers_per_step)

        print(
            f">>> Synthetic setup: profile_pool={self.agent_gen.total_customers}, "
            f"train days={train_timesteps} x {train_steps_per_day} steps/day x {train_customers_per_step} rides, "
            f"eval timesteps={eval_timesteps} x {eval_customers_per_step} rides"
        )
        if self.firm1_mode == "RL":
            update_every = int(max(1, ppo_update_interval_days))
            print(
                f">>> PPO rollout/update cadence: {update_every} day(s) per optimizer step "
                f"(~{update_every * max(1, int(train_steps_per_day))} transitions/update)."
            )
        
        print(">>> Training RL agent on synthetic NYC-calibrated sampling (day/timestep cadence)...")
        reward_history: List[float] = []
        sampled_profile_rows: List[Dict[str, Any]] = []
        profile_limit_reached = False

        # If evaluation opponent is static, use the default dynamic heuristic during training
        # to match the long-horizon `run` behavior. Preserve explicit dynamic alternatives.
        orig_firm2, orig_firm2_mode = self.firm2, self.firm2_mode
        if self.firm2_mode == "static":
            self.firm2 = FirmHeuristicPricer(seed=self.seed + 1, managed_keys=self.shared_edit_keys)
            self.firm2_mode = "heuristic"
        
        update_every = int(max(1, ppo_update_interval_days))
        last_ppo_metrics = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "approx_kl": 0.0,
            "clipfrac": 0.0,
            "entropy": 0.0,
            "policy_entropy": 0.0,
            "policy_entropy_fraction": 1.0,
            "ent_coeff": 0.0,
            "exploration_rate": 0.0,
            "action_coverage": 0.0,
            "explained_variance": 0.0,
            "optimizer_steps": 0,
            "lr": 0.0,
            "stopped_early_kl": False,
        }

        for d in range(train_timesteps):
            warmup_days = max(1.0, float(train_timesteps) * self.driver_reward_warmup_fraction)
            self.driver_reward_scale_current = float(np.clip((d + 1) / warmup_days, 0.0, 1.0)) if self.enable_driver_supply else 0.0
            day_ctx = self.market.sample_day_context()
            hours = [self.market.sample_timestep_hour().hour for _ in range(max(1, int(train_steps_per_day)))]

            reward_sum = 0.0
            base_reward_sum = 0.0
            raw_reward_sum = 0.0
            competitive_sum = 0.0
            trend_sum = 0.0
            discipline_sum = 0.0
            share_delta_sum = 0.0
            rev_delta_sum = 0.0
            efficiency_sum = 0.0
            reward_component_sums: Dict[str, float] = defaultdict(float)
            share_sum = 0.0
            revpr_sum = 0.0
            profitpr_sum = 0.0
            gap_sum = 0.0
            fulfillment_sum = 0.0
            wait_sum = 0.0
            driver_accept_sum = 0.0
            driver_paypr_sum = 0.0
            action_counts: Counter[int] = Counter()
            last_action = -1
            recovery_guardrail_count = 0

            for t, hour in enumerate(hours):
                base = self.market.curr_market
                rl_step = None
                if self.firm1_mode == "RL":
                    s_vec = self._build_rl_state(day_of_week=day_ctx.day_of_week, hour=hour, weather=day_ctx.weather)
                    action, s_ts, logits, val = self.firm1.agent.act(s_vec)
                    self.firm1.apply_action(action, self.market)
                    action_counts[int(action)] += 1
                    last_action = int(action)
                    rl_step = (action, s_ts, logits, val)
                elif self.firm1_mode != "static":
                    self.firm1.act(city_base=base.base_fare, city_pmin=base.per_minute, hour=hour, weather=day_ctx.weather)

                self.firm2.act(city_base=base.base_fare, city_pmin=base.per_minute, hour=hour, weather=day_ctx.weather)

                sampled_profiles = self._sample_profiles_from_pool(train_customers_per_step)
                if profiles_out and not profile_limit_reached:
                    remaining = int(max(0, profiles_log_limit - len(sampled_profile_rows)))
                    if remaining > 0:
                        sampled_profile_rows.extend({"Phase": "train", "Day": int(d), "Timestep": int(t), **p} for p in sampled_profiles[:remaining])
                    profile_limit_reached = len(sampled_profile_rows) >= int(max(0, profiles_log_limit))

                _, m1, m2, mean_gap, _, _ = self.simulate_batch(
                    day_of_week=day_ctx.day_of_week,
                    weather=day_ctx.weather,
                    hour=hour,
                    customers_per_step=train_customers_per_step,
                    sampled_profiles=sampled_profiles,
                    collect_rows=False,
                )

                if self.firm2_mode != "static":
                    self.firm2.update(metrics=m2, price_gap_mean=mean_gap)
                    
                reward_diag = self._reward_diagnostics(
                    share=float(m1.share),
                    rev_per_request=float(m1.rev_per_request),
                    mean_gap=float(mean_gap),
                    prev_share=float(self.last_share),
                    prev_rev_per_request=float(self.last_revpr),
                    prev_profit_per_request=float(self.last_profitpr),
                    prev_gap=float(self.last_gap),
                    profit_per_request=float(m1.profit_per_request),
                    profit_margin=self._profit_margin(m1),
                    fulfillment_rate=float(m1.fulfillment_rate),
                    avg_wait_minutes=float(m1.avg_wait_minutes),
                    driver_acceptance_rate=float(m1.driver_acceptance_rate),
                )

                if self.firm1_mode == "RL" and rl_step is not None:
                    action, s_ts, logits, val = rl_step
                    reward = float(reward_diag["reward"])
                    self.last_reward = reward
                    done = (t == len(hours) - 1)
                    self.firm1.agent.store(s_ts, action, reward, done, None, logits, val)
                    if done:
                        before_base = getattr(self.firm1.overrides, "base_fare", None)
                        before_pmin = getattr(self.firm1.overrides, "per_minute", None)
                        self.firm1.stabilize_after_batch(
                            share=float(m1.share),
                            price_gap_f2_minus_f1=float(mean_gap),
                            city_base=float(base.base_fare),
                            city_pmin=float(base.per_minute),
                            city_pmile=float(base.per_mile),
                            city_booking=float(base.booking_fee),
                            city_airport=float(base.airport_fee),
                            profit_per_request=float(m1.profit_per_request),
                            fulfillment_rate=float(m1.fulfillment_rate),
                        )
                        after_base = getattr(self.firm1.overrides, "base_fare", None)
                        after_pmin = getattr(self.firm1.overrides, "per_minute", None)
                        if before_base != after_base or before_pmin != after_pmin:
                            recovery_guardrail_count += 1
                    reward_sum += reward
                elif self.firm1_mode != "RL":
                    reward_sum += float(reward_diag["reward_base"])

                base_reward_sum += float(reward_diag["reward_base"])
                raw_reward_sum += float(reward_diag["reward_raw"])
                competitive_sum += float(reward_diag["reward_competitive_term"])
                trend_sum += float(reward_diag["reward_trend_term"])
                discipline_sum += float(reward_diag["reward_pricing_discipline"])
                share_delta_sum += float(reward_diag["reward_share_delta"])
                rev_delta_sum += float(reward_diag["reward_rev_delta"])
                efficiency_sum += float(reward_diag["reward_efficiency_term"])
                for component_key in (
                    "reward_share_component",
                    "reward_profit_component",
                    "reward_momentum_component",
                    "reward_profit_delta",
                    "reward_gap_delta",
                    "reward_margin_penalty",
                    "reward_price_discipline_penalty",
                    "reward_price_gap_deviation",
                    "reward_price_gap_penalty",
                    "reward_served_share",
                    "reward_driver_unfulfilled_penalty",
                    "reward_driver_fulfillment_component",
                    "reward_driver_wait_penalty",
                    "reward_driver_rejection_penalty",
                ):
                    reward_component_sums[component_key] += float(reward_diag.get(component_key, 0.0))

                share_sum += float(m1.share)
                revpr_sum += float(m1.rev_per_request)
                profitpr_sum += float(m1.profit_per_request)
                gap_sum += float(mean_gap)
                fulfillment_sum += float(m1.fulfillment_rate)
                wait_sum += float(m1.avg_wait_minutes)
                driver_accept_sum += float(m1.driver_acceptance_rate)
                driver_paypr_sum += float(m1.driver_pay / max(1, m1.total))
                self.last_share = float(m1.share)
                self.last_revpr = float(m1.rev_per_request)
                self.last_gap = float(mean_gap)
                self.last_profitpr = float(m1.profit_per_request)
                self.last_fulfillment = float(m1.fulfillment_rate)
                self.last_acceptance = float(m1.driver_acceptance_rate)
                self.last_wait = float(m1.avg_wait_minutes)
                self.last_driver_paypr = float(m1.driver_pay / max(1, m1.total))

            ppo_metrics = dict(last_ppo_metrics)
            if self.firm1_mode == "RL":
                should_update = ((d + 1) % update_every == 0) or ((d + 1) == train_timesteps)
                if should_update:
                    ppo_metrics = self.firm1.agent.update(epochs=self.ppo_update_epochs, batch_size=self.ppo_batch_size)
                    last_ppo_metrics = dict(ppo_metrics)
                
            n_steps = max(1, len(hours))
            avg_reward = float(reward_sum / n_steps)
            reward_history.append(float(avg_reward))
                    
            moving_window = reward_history[-min(len(reward_history), 20):]
            moving_avg20 = float(np.mean(moving_window)) if moving_window else 0.0
            moving_std20 = float(np.std(moving_window)) if moving_window else 0.0
            avg_share = float(share_sum / n_steps)
            avg_revpr = float(revpr_sum / n_steps)
            avg_profitpr = float(profitpr_sum / n_steps)
            avg_gap = float(gap_sum / n_steps)
            avg_fulfillment = float(fulfillment_sum / n_steps)
            avg_wait = float(wait_sum / n_steps)
            avg_driver_accept = float(driver_accept_sum / n_steps)
            avg_driver_paypr = float(driver_paypr_sum / n_steps)
            dominant_action = int(action_counts.most_common(1)[0][0]) if action_counts else -1
            dominant_action_rate = float(action_counts.most_common(1)[0][1] / n_steps) if action_counts else 0.0

            log_row = {
                "batch": d,
                "avg_reward": float(avg_reward),
                "reward_moving_avg20": moving_avg20,
                "reward_moving_std20": moving_std20,
                "reward_base": float(base_reward_sum / n_steps),
                "reward_raw": float(raw_reward_sum / n_steps),
                "reward_dominance_term": float(competitive_sum / n_steps),
                "reward_competitive_term": float(competitive_sum / n_steps),
                "reward_trend_term": float(trend_sum / n_steps),
                "reward_pricing_discipline": float(discipline_sum / n_steps),
                "reward_share_delta": float(share_delta_sum / n_steps),
                "reward_rev_delta": float(rev_delta_sum / n_steps),
                "reward_efficiency_term": float(efficiency_sum / n_steps),
                **{
                    k: float(v / n_steps)
                    for k, v in sorted(reward_component_sums.items())
                },
                "avg_share": avg_share,
                "avg_rev_per_request": avg_revpr,
                "avg_profit_per_request": avg_profitpr,
                "avg_price_gap_f2_minus_f1": avg_gap,
                "avg_fulfillment_rate": avg_fulfillment,
                "avg_wait_minutes": avg_wait,
                "driver_acceptance_rate": avg_driver_accept,
                "driver_pay_per_request": avg_driver_paypr,
                "driver_reward_scale": float(self.driver_reward_scale_current),
                "last_action": int(last_action),
                "dominant_action": dominant_action,
                "dominant_action_rate": dominant_action_rate,
                "action_counts": json.dumps(dict(sorted(action_counts.items())), sort_keys=True),
                "recovery_guardrail_count": int(recovery_guardrail_count),
                "loss": float(ppo_metrics.get("loss", 0.0)),
                "ppo_policy_loss": float(ppo_metrics.get("policy_loss", 0.0)),
                "ppo_value_loss": float(ppo_metrics.get("value_loss", 0.0)),
                "ppo_approx_kl": float(ppo_metrics.get("approx_kl", 0.0)),
                "ppo_clipfrac": float(ppo_metrics.get("clipfrac", 0.0)),
                "ppo_entropy": float(ppo_metrics.get("entropy", 0.0)),
                "ppo_policy_entropy": float(ppo_metrics.get("policy_entropy", 0.0)),
                "ppo_policy_entropy_fraction": float(ppo_metrics.get("policy_entropy_fraction", 1.0)),
                "ppo_ent_coeff": float(ppo_metrics.get("ent_coeff", 0.0)),
                "ppo_exploration_rate": float(ppo_metrics.get("exploration_rate", 0.0)),
                "ppo_action_coverage": float(ppo_metrics.get("action_coverage", 0.0)),
                "ppo_explained_variance": float(ppo_metrics.get("explained_variance", 0.0)),
                "ppo_optimizer_steps": int(ppo_metrics.get("optimizer_steps", 0)),
                "ppo_lr": float(ppo_metrics.get("lr", 0.0)),
                "ppo_stopped_early_kl": bool(ppo_metrics.get("stopped_early_kl", False)),
            }
            for k in self.shared_edit_keys:
                log_row[f"firm1_{k}"] = float(get_coeff(self.market.curr_market, self.firm1.overrides, k))
                log_row[f"firm2_{k}"] = float(get_coeff(self.market.curr_market, self.firm2.overrides, k))
            self.training_logs.append(log_row)
            
            if self.firm1_mode == "RL":
                # Keep exploration/entropy controls synchronized in the primary
                # experiment workflow as well as in ``run``. Previously this
                # path stayed at the initial 40% uniform mixture for the entire
                # training run, preventing the learned policy from exploiting.
                convergence_window = reward_history[-min(len(reward_history), 40):]
                convergence_std = (
                    float(np.std(convergence_window))
                    if len(convergence_window) >= 2
                    else 1.0
                )
                convergence_delta = (
                    float(abs(convergence_window[-1] - convergence_window[0])
                          / max(1, len(convergence_window) - 1))
                    if len(convergence_window) >= 2
                    else 1.0
                )
                reward_converged = bool(
                    len(convergence_window) >= 40
                    and convergence_std <= self.reward_convergence_tol
                    and convergence_delta <= self.reward_trend_tol
                    and float(ppo_metrics.get("action_coverage", 0.0)) >= 0.95
                    and float(ppo_metrics.get("policy_entropy_fraction", 1.0)) <= 0.40
                )
                self.firm1.configure_training_controls(
                    progress=float((d + 1) / max(1, train_timesteps)),
                    reward_converged=reward_converged,
                    reward_std=convergence_std,
                )
            
            if (d + 1) % max(1, train_timesteps // 10) == 0:
                print(
                    f"  [train {d+1}/{train_timesteps}] reward={float(avg_reward):.3f} "
                    f"moving_avg20={moving_avg20:.3f} share={avg_share:.3f} "
                    f"revPR={avg_revpr:.2f} profitPR={avg_profitpr:.2f} "
                    f"gap(F2-F1)={avg_gap:.2f} fulfill={avg_fulfillment:.3f} "
                    f"accept={avg_driver_accept:.3f} "
                    f"KL={float(ppo_metrics.get('approx_kl', 0.0)):.4f} "
                    f"clip={float(ppo_metrics.get('clipfrac', 0.0)):.3f}"
                )
                
            
        self.firm2, self.firm2_mode = orig_firm2, orig_firm2_mode
        
        print(">>> Evaluating RL agent against static/heuristic opponent with shared profile pool...")
        self.driver_reward_scale_current = 1.0 if self.enable_driver_supply else 0.0
        eval_rewards: List[float] = []
        # Reset reward-trend baselines so evaluation reward reflects evaluation dynamics,
        # not trailing deltas from the end of training.
        eval_last_share = float(self.last_share)
        eval_last_revpr = float(self.last_revpr)
        eval_last_profitpr = float(self.last_profitpr)
        eval_last_gap = float(self.last_gap)
        for t in range(eval_timesteps):
            day_ctx = self.market.sample_day_context()
            hour = self.market.sample_timestep_hour().hour
            sampled_profiles = self._sample_profiles_from_pool(eval_customers_per_step)
            if profiles_out and not profile_limit_reached:
                remaining = int(max(0, profiles_log_limit - len(sampled_profile_rows)))
                if remaining > 0:
                    sampled_profile_rows.extend({"Phase": "eval", "Timestep": int(t), **p} for p in sampled_profiles[:remaining])
                profile_limit_reached = len(sampled_profile_rows) >= int(max(0, profiles_log_limit))

            base = self.market.curr_market
            action = -1
            if self.firm1_mode == "RL":
                s_vec = self._build_rl_state(day_of_week=day_ctx.day_of_week, hour=hour, weather=day_ctx.weather)
                action, *_ = self.firm1.agent.act(s_vec, deterministic=True)
                self.firm1.apply_action(action, self.market)
                action = int(action)
            elif self.firm1_mode != "static":
                self.firm1.act(city_base=base.base_fare, city_pmin=base.per_minute, hour=hour, weather=day_ctx.weather)

            if self.firm2_mode != "static":
                self.firm2.act(city_base=base.base_fare, city_pmin=base.per_minute, hour=hour, weather=day_ctx.weather)

            _, m1, m2, mean_gap, _, _ = self.simulate_batch(
                day_of_week=day_ctx.day_of_week,
                weather=day_ctx.weather,
                hour=hour,
                customers_per_step=eval_customers_per_step,
                sampled_profiles=sampled_profiles,
                collect_rows=False,
            )
                
            # Use the same shaped reward family as training for consistent trajectory logs.
            # Compute with local baselines to avoid mutating training history state.
            share = float(np.clip(m1.share, 0.0, 1.0))
            revpr = float(max(0.0, m1.rev_per_request))
            reward_diag = self._reward_diagnostics(
                share=share,
                rev_per_request=revpr,
                mean_gap=float(mean_gap),
                prev_share=eval_last_share,
                prev_rev_per_request=eval_last_revpr,
                profit_per_request=float(m1.profit_per_request),
                prev_profit_per_request=eval_last_profitpr,
                prev_gap=eval_last_gap,
                profit_margin=self._profit_margin(m1),
                fulfillment_rate=float(m1.fulfillment_rate),
                avg_wait_minutes=float(m1.avg_wait_minutes),
                driver_acceptance_rate=float(m1.driver_acceptance_rate),
            )
            eval_reward = float(reward_diag["reward"])
            
            if self.firm1_mode == "RL":
                # Evaluation uses a deterministic argmax policy. Repeating the
                # same incremental macro action can otherwise walk coefficients
                # to a boundary even after the resulting market collapses.
                # Apply the same business safety projection used in training so
                # evaluation measures the deployed controller, not unbounded
                # accumulation of one action.
                self.firm1.stabilize_after_batch(
                    share=float(m1.share),
                    price_gap_f2_minus_f1=float(mean_gap),
                    city_base=float(base.base_fare),
                    city_pmin=float(base.per_minute),
                    city_pmile=float(base.per_mile),
                    city_booking=float(base.booking_fee),
                    city_airport=float(base.airport_fee),
                    profit_per_request=float(m1.profit_per_request),
                    fulfillment_rate=float(m1.fulfillment_rate),
                )

            eval_last_share = float(share)
            eval_last_revpr = float(revpr)
            eval_last_profitpr = float(m1.profit_per_request)
            eval_last_gap = float(mean_gap)
            eval_rewards.append(float(eval_reward))
            moving_window = eval_rewards[-min(len(eval_rewards), 20):]
            moving_avg20 = float(np.mean(moving_window)) if moving_window else 0.0
            moving_std20 = float(np.std(moving_window)) if moving_window else 0.0
            log_row = {
                "day": t,
                "reward": float(eval_reward),
                "reward_moving_avg20": moving_avg20,
                "reward_moving_std20": moving_std20,
                "reward_raw": float(reward_diag["reward_raw"]),
                "reward_base": float(reward_diag["reward_base"]),
                "reward_dominance_term": float(reward_diag["reward_dominance_term"]),
                "reward_competitive_term": float(reward_diag["reward_competitive_term"]),
                "reward_trend_term": float(reward_diag["reward_trend_term"]),
                "reward_pricing_discipline": float(reward_diag["reward_pricing_discipline"]),
                "reward_share_delta": float(reward_diag["reward_share_delta"]),
                "reward_rev_delta": float(reward_diag["reward_rev_delta"]),
                "reward_efficiency_term": float(reward_diag["reward_efficiency_term"]),
                "rl_share": float(m1.share),
                "heuristic_share": float(m2.share),
                "rl_revenue": float(m1.rev_per_request),
                "heuristic_revenue": float(m2.rev_per_request),
                "rl_profit": float(m1.profit_per_request),
                "heuristic_profit": float(m2.profit_per_request),
                "rl_fulfillment_rate": float(m1.fulfillment_rate),
                "rl_avg_wait_minutes": float(m1.avg_wait_minutes),
                "rl_driver_acceptance_rate": float(m1.driver_acceptance_rate),
                "rl_driver_pay_per_request": float(m1.driver_pay / max(1, m1.total)),
                "driver_reward_scale": float(self.driver_reward_scale_current),
                "price_gap_f2_minus_f1": float(mean_gap),
                "action": int(action),
            }
            for k in self.shared_edit_keys:
                log_row[f"firm1_{k}"] = float(get_coeff(self.market.curr_market, self.firm1.overrides, k))
                log_row[f"firm2_{k}"] = float(get_coeff(self.market.curr_market, self.firm2.overrides, k))
            self.evaluation_logs.append(log_row)
            self.last_share = float(m1.share)
            self.last_revpr = float(m1.rev_per_request)
            self.last_gap = float(mean_gap)
            self.last_profitpr = float(m1.profit_per_request)
            self.last_fulfillment = float(m1.fulfillment_rate)
            self.last_acceptance = float(m1.driver_acceptance_rate)
            self.last_wait = float(m1.avg_wait_minutes)
            self.last_driver_paypr = float(m1.driver_pay / max(1, m1.total))

            if (t + 1) % max(1, eval_timesteps // 10) == 0:
                print(
                    f"  [eval {t+1}/{eval_timesteps}] reward={float(eval_reward):.3f} "
                    f"moving_avg20={moving_avg20:.3f} share={float(m1.share):.3f} "
                    f"revPR={float(m1.rev_per_request):.2f} gap(F2-F1)={float(mean_gap):.2f}"
                )

        if profiles_out:
            _ensure_parent_dir(profiles_out)
            _write_csv(profiles_out, sampled_profile_rows)
            print(f">>> Saved sampled profiles -> {profiles_out} (rows={len(sampled_profile_rows)})")
            if profile_limit_reached:
                print(f">>> Profile export capped at profiles_log_limit={profiles_log_limit} rows.")

        if reward_history:
            print(f">>> Training reward trajectory summary: min={float(np.min(reward_history)):.3f}, max={float(np.max(reward_history)):.3f}, final={float(reward_history[-1]):.3f}")
        if eval_rewards:
            print(f">>> Testing reward trajectory summary: min={float(np.min(eval_rewards)):.3f}, max={float(np.max(eval_rewards)):.3f}, final={float(eval_rewards[-1]):.3f}")
            
        print("Experiment Complete.")
        return self.training_logs, self.evaluation_logs
    
    def compare_trained_rl_to_dataset(
       self,
       dataset_root: str,
       dataset_glob: str = "*.parquet",
       out_csv: Optional[str] = None,
       out_plot_prefix: Optional[str] = None,
       max_rows: int = 50000,
       preview_rows: int = 5,
   ) -> Dict[str, Any]:
       files = _discover_dataset_files(dataset_root=dataset_root, dataset_glob=dataset_glob)

       if not files:
           raise FileNotFoundError(
               f"No parquet dataset files found under dataset_root={dataset_root!r} with glob={dataset_glob!r}"
           )

       rows_out: List[Dict[str, Any]] = []
       abs_err: List[float] = []
       sq_err: List[float] = []
       signed_err: List[float] = []
       abs_pct: List[float] = []
       dur_abs_err: List[float] = []
       dur_sq_err: List[float] = []
       dur_abs_pct: List[float] = []

       processed = 0
       kept = 0

       actual_price_cols = [
           "price", "fare", "fare_amount", "total_amount", "final_price", "paid", "cost",
           "estimated_price", "trip_price", "amount", "total_fare", "price_usd",
           "avg_price", "avg_fare", "fare_usd", "price_estimate", "estimate", "passenger_fare", "Passenger Fare",
       ]
       lower_price_cols = ["low_estimate", "minimum", "min_estimate", "fare_low"]
       upper_price_cols = ["high_estimate", "maximum", "max_estimate", "fare_high"]
       distance_cols = [
           "distance", "distance_miles", "trip_miles", "miles", "trip_distance", "DistanceMiles", "TravelDistance",
           "trip_length", "Trip Length",
       ]
       duration_cols = [
           "duration", "duration_minutes", "trip_duration", "trip_time", "duration_secs", "DurationMinutes",
           "trip_duration_minutes", "duration_min", "travel_time", "total_ride_time", "Total Ride Time",
           "on_scene_to_dropoff", "On Scene to Dropoff", "request_to_dropoff", "Request to Dropoff",
       ]
       duration_seconds_cols = ["duration_secs", "duration_seconds", "trip_duration_seconds", "eta_seconds"]
       dropoff_time_cols = ["dropoff_datetime", "tpep_dropoff_datetime", "lpep_dropoff_datetime"]
       pickup_time_cols = ["pickup_datetime", "tpep_pickup_datetime", "lpep_pickup_datetime"]
       
       skipped_missing_price = 0
       skipped_missing_distance = 0
       files_with_rows = 0
       preview_printed = False

       for fpath in files:
           file_kept = 0
           try:
               for fieldnames, reader in _iter_tabular_rows(fpath):
                   if (not preview_printed) and preview_rows > 0:
                       print(f"[Dataset Preview] file={fpath}")
                       print(f"[Dataset Preview] columns={fieldnames}")

                   for idx, raw in enumerate(reader):
                       if (not preview_printed) and preview_rows > 0 and idx < preview_rows:
                           sample_keys = list(raw.keys())[:12]
                           sample = {k: raw.get(k) for k in sample_keys}
                           print(f"[Dataset Preview] row_{idx}: {_json_preview(sample)}")
                       if (not preview_printed) and preview_rows > 0 and idx + 1 >= preview_rows:
                           preview_printed = True
                       processed += 1
                       if kept >= max_rows:
                           break
                       
                       raw = _normalize_row_keys(raw)

                       actual_paid = _pick_first_numeric(raw, actual_price_cols)
                       if actual_paid is None:
                           low = _pick_first_numeric(raw, lower_price_cols)
                           high = _pick_first_numeric(raw, upper_price_cols)
                           if low is not None and high is not None:
                               actual_paid = float((low + high) / 2.0)
                           elif low is not None:
                               actual_paid = float(low)
                           elif high is not None:
                               actual_paid = float(high)

                       distance, distance_key = _pick_first_numeric_with_key(raw, distance_cols)
                       if distance is not None and distance_key == "distance_km":
                           distance = float(distance) * 0.621371

                       if actual_paid is None:
                           skipped_missing_price += 1
                           continue
                       if distance is None:
                           skipped_missing_distance += 1
                           continue

                       hour, day_of_week = _infer_hour_day(raw)
                       actual_duration, duration_key = _pick_first_minutes_with_key(raw, duration_cols)
                       second_based_duration_keys = set(duration_seconds_cols + [
                           "request_to_dropoff",
                           "request_to_pickup",
                           "on_scene_to_pickup",
                           "on_scene_to_dropoff",
                           "total_ride_time",
                           "trip_time",
                       ])
                       if actual_duration is not None and duration_key in second_based_duration_keys:
                           actual_duration = float(actual_duration) / 60.0
                       if actual_duration is None:
                           pickup_raw = _pick_value(raw, pickup_time_cols)
                           dropoff_raw = _pick_value(raw, dropoff_time_cols)
                           if pickup_raw is not None and dropoff_raw is not None:
                               try:
                                   pickup_dt = datetime.fromisoformat(str(pickup_raw).replace("Z", "+00:00"))
                                   dropoff_dt = datetime.fromisoformat(str(dropoff_raw).replace("Z", "+00:00"))
                                   secs = (dropoff_dt - pickup_dt).total_seconds()
                                   if secs > 0:
                                       actual_duration = float(secs / 60.0)
                               except Exception:
                                   pass

                       predicted_duration = float(self.estimate_duration(float(distance), hour))
                       duration = float(actual_duration) if actual_duration is not None else predicted_duration

                       if actual_duration is not None:
                           d_err = float(predicted_duration - float(actual_duration))
                           dae = float(abs(d_err))
                           dur_abs_err.append(dae)
                           dur_sq_err.append(float(d_err * d_err))
                           if float(actual_duration) > 1e-6:
                               dur_abs_pct.append(float(dae / float(actual_duration)))

                       weather = _infer_weather(raw)
                       airport = _infer_airport_trip(raw)
                       service = _infer_service_level(raw)

                       ctx = RideContext(
                           day_of_week=int(np.clip(day_of_week, 0, 6)),
                           weather=weather,
                           hour=int(np.clip(hour, 0, 23)),
                           airport=bool(airport),
                           service=service,
                       )
                       # Keep coefficient overrides fixed during dataset comparison.
                       # Applying sequential RL actions row-by-row introduces policy drift
                       # unrelated to each observed ride and creates strong directional bias.
                       rl_price = self.market.quote_price(
                           distance_miles=float(max(0.0, distance)),
                           duration_minutes=float(max(0.0, duration)),
                           ctx=ctx,
                           overrides=self.firm1.overrides,
                       )

                       err = float(rl_price - actual_paid)
                       ae = float(abs(err))
                       abs_err.append(ae)
                       sq_err.append(float(err * err))
                       signed_err.append(err)
                       if actual_paid > 1e-6:
                           abs_pct.append(float(ae / actual_paid))

                       rows_out.append({
                           "source_file": os.path.basename(fpath),
                           "actual_paid": float(actual_paid),
                           "rl_predicted_price": float(rl_price),
                           "price_error": err,
                           "abs_error": ae,
                           "distance_miles": float(distance),
                           "actual_duration_minutes": float(actual_duration) if actual_duration is not None else None,
                           "predicted_duration_minutes": float(predicted_duration),
                           "duration_minutes_used_for_price": float(duration),
                           "hour": int(ctx.hour),
                           "day_of_week": int(ctx.day_of_week),
                           "weather": str(ctx.weather),
                           "airport": bool(ctx.airport),
                           "service": str(ctx.service),
                       })
                       kept += 1
                       file_kept += 1

                   if kept >= max_rows:
                       break
                   
           except Exception as exc:
               print(f"[Dataset Preview] skipped file={fpath} due to read error: {exc}")
               continue
              
           if file_kept > 0:
               files_with_rows += 1
           if kept >= max_rows:
               break

       if out_csv:
           _ensure_parent_dir(out_csv)
           _write_csv(out_csv, rows_out)
           
       if out_plot_prefix and rows_out:
           self._plot_dataset_validation(rows_out=rows_out, out_plot_prefix=out_plot_prefix)
       
       if rows_out:
           self._print_kaggle_analysis(rows_out=rows_out)

       self._print_convergence_analysis()
       summary = {
           "dataset_root": dataset_root,
           "files_scanned": len(files),
           "rows_processed": int(processed),
           "rows_compared": int(kept),
           "rows_skipped_missing_price": int(skipped_missing_price),
           "rows_skipped_missing_distance": int(skipped_missing_distance),
           "files_with_comparable_rows": int(files_with_rows),
           "mae": float(np.mean(abs_err)) if abs_err else None,
           "rmse": float(np.sqrt(np.mean(sq_err))) if sq_err else None,
           "mape": float(np.mean(abs_pct)) if abs_pct else None,
           "bias": float(np.mean(signed_err)) if signed_err else None,
           "duration_mae_minutes": float(np.mean(dur_abs_err)) if dur_abs_err else None,
           "duration_rmse_minutes": float(np.sqrt(np.mean(dur_sq_err))) if dur_sq_err else None,
           "duration_mape": float(np.mean(dur_abs_pct)) if dur_abs_pct else None,
           "out_csv": out_csv,
           "out_plot_prefix": out_plot_prefix,
       }
       self._print_reality_gap_check(summary)
       self._print_sensitivity_analysis()
       return summary
   
    def _print_reality_gap_check(self, summary: Dict[str, Any]) -> None:
        mae = summary.get("mae")
        mape = summary.get("mape")
        bias = summary.get("bias")
        if mae is None:
            print(">>> Reality gap check: insufficient comparable Kaggle rows.")
            return
        bias_dir = "overpricing" if (bias or 0.0) > 0 else "underpricing"
        print(
            ">>> Reality gap check: "
            f"mae={float(mae):.3f}, mape={float(mape or 0.0):.3f}, bias={float(bias or 0.0):.3f} ({bias_dir})."
        )

    def _print_sensitivity_analysis(self) -> None:
        base_ctx = RideContext(day_of_week=2, weather="clear", hour=14, airport=False, service="economy")
        base_distance = 3.0
        base_duration = float(self.estimate_duration(base_distance, base_ctx.hour))
        base_price = float(self.market.quote_price(base_distance, base_duration, base_ctx, overrides=self.firm1.overrides))

        def _quote(distance: float, duration: float, hour: int, weather: str, airport: bool, service: str) -> float:
            ctx = RideContext(day_of_week=2, weather=weather, hour=int(np.clip(hour, 0, 23)), airport=airport, service=service)
            return float(self.market.quote_price(distance, duration, ctx, overrides=self.firm1.overrides))

        dist_up = _quote(base_distance * 1.2, base_duration, base_ctx.hour, base_ctx.weather, base_ctx.airport, base_ctx.service)
        dur_up = _quote(base_distance, base_duration * 1.2, base_ctx.hour, base_ctx.weather, base_ctx.airport, base_ctx.service)
        rush = _quote(base_distance, base_duration, 18, base_ctx.weather, base_ctx.airport, base_ctx.service)
        rain = _quote(base_distance, base_duration, base_ctx.hour, "rain", base_ctx.airport, base_ctx.service)
        airport = _quote(base_distance, base_duration, base_ctx.hour, base_ctx.weather, True, base_ctx.service)
        premium = _quote(base_distance, base_duration, base_ctx.hour, base_ctx.weather, base_ctx.airport, "premium")

        print(
            ">>> Sensitivity analysis (trained RL pricing, baseline ride=3mi clear weekday 14:00): "
            f"base={base_price:.2f}, +20%dist={dist_up:.2f}, +20%duration={dur_up:.2f}, rush18={rush:.2f}, rain={rain:.2f}, airport={airport:.2f}, premium={premium:.2f}"
        )
        
    def _print_kaggle_analysis(self, rows_out: List[Dict[str, Any]]) -> None:
        """Print concise dataset-vs-model diagnostics for quick sanity checks."""
        if not rows_out:
            print(">>> Kaggle comparison: no comparable rows available.")
            return

        price_errors = np.array([float(r.get("price_error", 0.0)) for r in rows_out], dtype=float)
        abs_errors = np.abs(price_errors)
        actual_prices = np.array([float(r.get("actual_paid", 0.0)) for r in rows_out], dtype=float)

        with np.errstate(divide="ignore", invalid="ignore"):
            ape = np.where(actual_prices > 1e-6, abs_errors / actual_prices, np.nan)

        duration_errors = [
            abs(float(r["predicted_duration_minutes"]) - float(r["actual_duration_minutes"]))
            for r in rows_out
            if r.get("actual_duration_minutes") is not None and r.get("predicted_duration_minutes") is not None
        ]

        print(
            ">>> Kaggle comparison summary: "
            f"rows={len(rows_out)}, "
            f"price_mae={float(np.mean(abs_errors)):.3f}, "
            f"price_rmse={float(np.sqrt(np.mean(np.square(price_errors)))):.3f}, "
            f"price_bias={float(np.mean(price_errors)):.3f}, "
            f"price_mape={float(np.nanmean(ape)):.3f}"
        )

        if duration_errors:
            print(
                ">>> Kaggle duration summary: "
                f"duration_mae_minutes={float(np.mean(duration_errors)):.3f}, "
                f"duration_rmse_minutes={float(np.sqrt(np.mean(np.square(duration_errors)))):.3f}"
            )

    def _print_convergence_analysis(self) -> None:
        """Print light-weight diagnostics for training/evaluation reward trajectories."""
        train_rewards = [float(x.get("avg_reward", 0.0)) for x in self.training_logs]
        eval_rewards = [float(x.get("reward", 0.0)) for x in self.evaluation_logs]

        if train_rewards:
            tail = train_rewards[-min(len(train_rewards), 40):]
            print(
                ">>> Convergence (train): "
                f"n={len(train_rewards)}, mean_last{len(tail)}={float(np.mean(tail)):.3f}, std_last{len(tail)}={float(np.std(tail)):.4f}"
            )
        if eval_rewards:
            tail = eval_rewards[-min(len(eval_rewards), 40):]
            print(
                ">>> Convergence (eval): "
                f"n={len(eval_rewards)}, mean_last{len(tail)}={float(np.mean(tail)):.3f}, std_last{len(tail)}={float(np.std(tail)):.4f}"
            )
            
    @staticmethod
    def _plot_dataset_validation(rows_out: List[Dict[str, Any]], out_plot_prefix: str) -> None:
        if importlib.util.find_spec("matplotlib") is None:
            print("[WARN] matplotlib not installed; skipping RL dataset validation graphs.")
            return

        import matplotlib.pyplot as plt

        prices_actual = [float(r["actual_paid"]) for r in rows_out if r.get("actual_paid") is not None]
        prices_pred = [float(r["rl_predicted_price"]) for r in rows_out if r.get("rl_predicted_price") is not None]

        if prices_actual and prices_pred:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(prices_actual, prices_pred, s=10, alpha=0.35)
            lo = float(min(prices_actual + prices_pred))
            hi = float(max(prices_actual + prices_pred))
            ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.2, label="ideal match")
            ax.set_title("RL Predicted Price vs Actual Customer Price")
            ax.set_xlabel("Actual customer price")
            ax.set_ylabel("RL predicted price")
            ax.legend(loc="best")
            fig.tight_layout()
            out = f"{out_plot_prefix}_price_match_scatter.png"
            _ensure_parent_dir(out)
            fig.savefig(out, dpi=150)
            print(f"Saved graph -> {out}")
            plt.close(fig)

        duration_pairs = [
            (float(r["actual_duration_minutes"]), float(r["predicted_duration_minutes"]))
            for r in rows_out
            if r.get("actual_duration_minutes") is not None and r.get("predicted_duration_minutes") is not None
        ]
        if duration_pairs:
            actual_dur = [a for a, _ in duration_pairs]
            pred_dur = [p for _, p in duration_pairs]
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(actual_dur, pred_dur, s=10, alpha=0.35, color="tab:orange")
            lo = float(min(actual_dur + pred_dur))
            hi = float(max(actual_dur + pred_dur))
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.2, label="ideal match")
            ax.set_title("Predicted Total Time vs Actual Duration")
            ax.set_xlabel("Actual trip duration (minutes)")
            ax.set_ylabel("Predicted total time (minutes)")
            ax.legend(loc="best")
            fig.tight_layout()
            out = f"{out_plot_prefix}_duration_match_scatter.png"
            _ensure_parent_dir(out)
            fig.savefig(out, dpi=150)
            print(f"Saved graph -> {out}")
            plt.close(fig)
            
    def simulate_day_cycle(self, day_ctx, rides, is_training):
        """Runs one 200-ride cycle. Primarily used by run_experiment."""
        hour = 12
        base = self.market.curr_market

        rl_step = None
        if self.firm1_mode == "RL":
            s_vec = self._build_rl_state(day_of_week=day_ctx.day_of_week, hour=hour, weather=day_ctx.weather)
            action, s_ts, logits, val = self.firm1.agent.act(s_vec)
            self.firm1.apply_action(action, self.market)
            rl_step = (action, s_ts, logits, val)
        elif self.firm1_mode != "static":
            self.firm1.act(city_base=base.base_fare, city_pmin=base.per_minute, hour=hour, weather=day_ctx.weather)

        if self.firm2_mode != "static":
            self.firm2.act(city_base=base.base_fare, city_pmin=base.per_minute, hour=hour, weather=day_ctx.weather)

        results, m1, m2, gap, air, dist = self.simulate_batch(day_ctx.day_of_week, day_ctx.weather, hour, rides)
        
        reward = self._compute_rl_reward(m1, gap)


        if is_training and self.firm1_mode == "RL" and rl_step is not None:
            action, s_ts, logits, val = rl_step
            self.firm1.agent.store(s_ts, action, float(reward), True, None, logits, val)
            self.firm1.stabilize_after_batch(
                share=float(m1.share),
                price_gap_f2_minus_f1=float(gap),
                city_base=float(self.market.curr_market.base_fare),
                city_pmin=float(self.market.curr_market.per_minute),
                city_pmile=float(self.market.curr_market.per_mile),
                city_booking=float(self.market.curr_market.booking_fee),
                city_airport=float(self.market.curr_market.airport_fee),
                profit_per_request=float(m1.profit_per_request),
                fulfillment_rate=float(m1.fulfillment_rate),
            )
        
        self.last_share = float(m1.share)
        self.last_revpr = float(m1.rev_per_request)
        self.last_gap = float(gap)
        self.last_profitpr = float(m1.profit_per_request)
        self.last_fulfillment = float(m1.fulfillment_rate)
        self.last_acceptance = float(m1.driver_acceptance_rate)
        self.last_wait = float(m1.avg_wait_minutes)
        self.last_driver_paypr = float(m1.driver_pay / max(1, m1.total))
        
        self.airport_rate_last = air
        self.mean_distance_last = dist

        return results, m1, m2, float(reward)

    
    def _build_rl_state(self, day_of_week: int, hour: int, weather: str) -> np.ndarray:
        """Build state vector for Firm1 RL controller from current market + recent summaries."""
        f2_ema_share = getattr(self.firm2, "ema_share", 0.5)
        f2_ema_gap = getattr(self.firm2, "ema_gap", 0.0)
        
        f2_cooldown = float(getattr(self.firm2, "cooldown", 0))

        # Keep contextual encoding compact and low-variance.
        weather_code = {"clear": 0.0, "cloudy": 0.33, "rain": 0.66, "snow": 1.0}.get(str(weather).lower(), 0.0)
        ride_ctx_vec = np.array(
            [
                float(np.clip(day_of_week / 6.0, 0.0, 1.0)),
                float(np.clip(hour / 23.0, 0.0, 1.0)),
                weather_code,
            ],
            dtype=np.float32,
        )
        
        return build_state_vector(
            base=self.market.curr_market,
            ov_firm1=self.firm1.overrides,
            opt_keys=self.opt_keys,
            ride_ctx_vec=ride_ctx_vec,
            airport_rate_last=self.airport_rate_last,
            mean_distance_last=self.mean_distance_last,
            firm2_ema_share=float(f2_ema_share),
            firm2_ema_gap=float(f2_ema_gap),
            firm2_cooldown=f2_cooldown,
            firm1_last_share=float(self.last_share),
            firm1_last_revpr=float(self.last_revpr),
            firm1_last_gap=float(self.last_gap),
            firm1_last_reward=float(self.last_reward),
            firm1_last_profitpr=float(self.last_profitpr),
            firm1_last_fulfillment=float(self.last_fulfillment),
            firm1_last_acceptance=float(self.last_acceptance),
            firm1_last_wait=float(self.last_wait),
            firm1_last_driver_paypr=float(self.last_driver_paypr),
            driver_state_vec=self.driver_supply.state_features_for_firm1() if self.enable_driver_supply else None,
        )

    def simulate_batch(
        self,
        day_of_week: int,
        weather: str,
        hour: int,
        customers_per_step: int,
        sampled_profiles: Optional[List[Dict[str, Any]]] = None,
        collect_rows: bool = True,
    ) -> Tuple[List[Dict[str, Any]], FirmMetrics, FirmMetrics, float, float, float]:
        rows: List[Dict[str, Any]] = []
        firm1 = FirmMetrics()
        firm2 = FirmMetrics()
        gap_sum = 0.0
        profile_count = 0

        airport_count = 0
        dist_sum = 0.0
        
        profiles = sampled_profiles if sampled_profiles is not None else self.agent_gen.sample_profiles(customers_per_step)
        if self.enable_driver_supply:
            self.driver_supply.begin_batch(customers_per_step=customers_per_step, hour=hour, weather=weather)

        for profile in profiles:
            profile_count += 1
            # trip-specific distance (scenario-side)
            travel_distance = round(float(self.rng.exponential(4.0)), 2)

            airport = self.market.sample_airport_flag()
            service = self.market.sample_service()
            airport_count += int(airport)
            dist_sum += float(travel_distance)

            duration = self.estimate_duration(travel_distance, hour)

            ctx = RideContext(
                day_of_week=day_of_week,
                weather=weather,
                hour=hour,
                airport=airport,
                service=service,
            )

            p1 = self.market.quote_price(travel_distance, duration, ctx, overrides=self.firm1.overrides)
            p2 = self.market.quote_price(travel_distance, duration, ctx, overrides=self.firm2.overrides)

            gap_sum += p2 - p1
            
            if self.enable_driver_supply:
                service_q1 = self.driver_supply.estimate_service_quality("Firm1", airport=airport)
                service_q2 = self.driver_supply.estimate_service_quality("Firm2", airport=airport)
                pickup_estimate1 = float(service_q1["pickup_minutes"])
                pickup_estimate2 = float(service_q2["pickup_minutes"])
                cancel_risk1 = float(service_q1["cancel_risk"])
                cancel_risk2 = float(service_q2["cancel_risk"])
            else:
                pickup_estimate1 = pickup_estimate2 = 0.0
                cancel_risk1 = cancel_risk2 = 0.0

            scenario = {
                "City": self.market_name,
                "DistanceMiles": float(travel_distance),
                "DurationMinutes": float(round(duration, 2)),
                "DayOfWeek": int(day_of_week),
                "Hour": int(hour),
                "Weather": str(weather),
                "Airport": bool(airport),
                "Service": str(service),
                "WaitEstimateFirm1": pickup_estimate1,
                "WaitEstimateFirm2": pickup_estimate2,
                "CancelRiskFirm1": cancel_risk1,
                "CancelRiskFirm2": cancel_risk2,
            }

            choice_res: ChoiceResult = self.choice_model.choose(profile, scenario, p1, p2)
            choice = choice_res.choice

            firm1.total += 1
            firm2.total += 1
            fulfilled = False
            match_status = "NoRide" if choice == "NoRide" else "NotDispatched"
            driver_pay = 0.0
            pickup_minutes = 0.0
            wait_minutes = 0.0
            acceptance_prob = 0.0
            if choice == "Firm1":
                firm1.chosen += 1
                if self.enable_driver_supply:
                    dispatch = self.driver_supply.dispatch("Firm1", p1, travel_distance, duration, airport=airport)
                    firm1.dispatch_offers += 1
                    acceptance_prob = float(dispatch.acceptance_probability)
                    pickup_minutes = float(dispatch.pickup_minutes)
                    wait_minutes = float(dispatch.wait_minutes)
                    driver_pay = float(dispatch.driver_pay)
                    if dispatch.fulfilled:
                        fulfilled = True
                        firm1.completed += 1
                        firm1.wins += 1
                        firm1.revenue += float(p1)
                        firm1.driver_pay += driver_pay
                        firm1.wait_minutes += wait_minutes
                        firm1.pickup_minutes += pickup_minutes
                        firm1.profit += float(p1) - driver_pay - float(dispatch.platform_variable_cost)
                        match_status = "Completed"
                    else:
                        firm1.unfulfilled += 1
                        firm1.driver_rejections += 1
                        match_status = dispatch.reject_reason
                else:
                    trip_cost = self._estimate_trip_cost(travel_distance, duration, airport)
                    fulfilled = True
                    firm1.completed += 1
                    firm1.wins += 1
                    firm1.revenue += float(p1)
                    firm1.profit += float(p1) - trip_cost
                    match_status = "Completed"
            elif choice == "Firm2":
                firm2.chosen += 1
                if self.enable_driver_supply:
                    dispatch = self.driver_supply.dispatch("Firm2", p2, travel_distance, duration, airport=airport)
                    firm2.dispatch_offers += 1
                    acceptance_prob = float(dispatch.acceptance_probability)
                    pickup_minutes = float(dispatch.pickup_minutes)
                    wait_minutes = float(dispatch.wait_minutes)
                    driver_pay = float(dispatch.driver_pay)
                    if dispatch.fulfilled:
                        fulfilled = True
                        firm2.completed += 1
                        firm2.wins += 1
                        firm2.revenue += float(p2)
                        firm2.driver_pay += driver_pay
                        firm2.wait_minutes += wait_minutes
                        firm2.pickup_minutes += pickup_minutes
                        firm2.profit += float(p2) - driver_pay - float(dispatch.platform_variable_cost)
                        match_status = "Completed"
                    else:
                        firm2.unfulfilled += 1
                        firm2.driver_rejections += 1
                        match_status = dispatch.reject_reason
                else:
                    trip_cost = self._estimate_trip_cost(travel_distance, duration, airport)
                    fulfilled = True
                    firm2.completed += 1
                    firm2.wins += 1
                    firm2.revenue += float(p2)
                    firm2.profit += float(p2) - trip_cost
                    match_status = "Completed"
            # "NoRide" / outside-option choices remain demand opportunities.
            # Driver-layer unfulfilled choices also remain demand opportunities but
            # generate no completed win or rideshare revenue.
            
            if collect_rows:
                rows.append({
                    "City": self.market_name,
                    "DayOfWeek": day_of_week,
                    "Weather": weather,
                    "Hour": hour,
                    "Airport": airport,
                    "Service": service,
                    "TravelDistance": travel_distance,
                    "Price_Firm1": p1,
                    "Price_Firm2": p2,
                    "Choice": choice,
                    "Fulfilled": bool(fulfilled),
                    "MatchStatus": match_status,
                    "DriverPay": float(driver_pay),
                    "PickupMinutes": float(pickup_minutes),
                    "WaitMinutes": float(wait_minutes),
                    "DriverAcceptanceProbability": float(acceptance_prob),
                    "WaitEstimateFirm1": pickup_estimate1,
                    "WaitEstimateFirm2": pickup_estimate2,
                    "ReasonCodes": ",".join(choice_res.reason_codes),
                    "ShortReason": choice_res.short_reason,
                    **profile,
                })

        if self.enable_driver_supply:
            self.driver_supply.end_batch()
        denominator = max(1, profile_count)
        mean_gap = float(gap_sum / denominator)
        airport_rate = float(airport_count / denominator)
        mean_dist = float(dist_sum / denominator)
        return rows, firm1, firm2, mean_gap, airport_rate, mean_dist

    def run(
        self,
        days: int,
        timesteps_per_day: int,
        customers_per_step: int,
        out_path: Optional[str] = None,
        profiles_out: Optional[str] = None,
        profiles_log_limit: int = 200000,
    ) -> List[Dict[str, Any]]:
        all_rows: List[Dict[str, Any]] = []
        stream_rows = bool(out_path)
        csv_file = None
        csv_writer: Optional[csv.DictWriter] = None
        sampled_profile_rows: List[Dict[str, Any]] = []
        profile_limit_reached = False
        
        if stream_rows:
            _ensure_parent_dir(out_path)
            csv_file = open(out_path, "w", newline="", encoding="utf-8")

        self._initialize_run_distributions()
        self._refresh_profile_pool(rides_per_timestep=customers_per_step)
        self.convergence_day = None
        self.convergence_window_std_at_day = None
        self.convergence_delta_per_day_at_day = None
        self._convergence_streak = 0
        self.driver_reward_scale_current = 1.0 if self.enable_driver_supply else 0.0
        
        for d in range(days):
            day_ctx = self.market.sample_day_context()
            hours = [self.market.sample_timestep_hour().hour for _ in range(timesteps_per_day)]

            # day accumulators for logging
            share_sum = 0.0
            revpr_sum = 0.0
            profit_sum = 0.0
            gap_sum = 0.0
            reward_sum = 0.0
            fulfillment_sum = 0.0
            wait_sum = 0.0
            driver_accept_sum = 0.0
            driver_paypr_sum = 0.0
            reward_component_sums: Dict[str, float] = defaultdict(float)
            action_counts: Counter[int] = Counter()
            last_action = -1
            
            share_sum_two = 0.0
            revpr_sum_two = 0.0
            profit_sum_two = 0.0
            no_ride_share_sum = 0.0

            for t in range(timesteps_per_day):
                hour = hours[t]
                base = self.market.curr_market

                # Firm 1 action
                rl_step = None
                if self.firm1_mode == "RL":
                    s_vec = self._build_rl_state(day_of_week=day_ctx.day_of_week, hour=hour, weather=day_ctx.weather)
                    action, s_ts, logits, val = self.firm1.agent.act(s_vec)
                    self.firm1.apply_action(action, self.market)
                    action_counts[int(action)] += 1
                    last_action = int(action)
                    rl_step = (action, s_ts, logits, val)
                elif self.firm1_mode != "static":
                    self.firm1.act(city_base=base.base_fare, city_pmin=base.per_minute, hour=hour, weather=day_ctx.weather)

                # Firm 2 action
                if self.firm2_mode != "static":
                    self.firm2.act(city_base=base.base_fare, city_pmin=base.per_minute, hour=hour, weather=day_ctx.weather)
                
                sampled_profiles = self._sample_profiles_from_pool(customers_per_step)
                if profiles_out and not profile_limit_reached:
                    remaining = int(max(0, profiles_log_limit - len(sampled_profile_rows)))
                    if remaining > 0:
                        sampled_profile_rows.extend(
                            {
                                "Phase": "run",
                                "Day": int(d),
                                "Timestep": int(t),
                                **p,
                            }
                            for p in sampled_profiles[:remaining]
                        )
                    profile_limit_reached = len(sampled_profile_rows) >= int(max(0, profiles_log_limit))

                rows, m1, m2, mean_gap, airport_rate, mean_dist = self.simulate_batch(
                    day_of_week=day_ctx.day_of_week,
                    weather=day_ctx.weather,
                    hour=hour,
                    customers_per_step=customers_per_step,
                    sampled_profiles=sampled_profiles,
                )
                if stream_rows:
                    if rows and csv_writer is None:
                        csv_writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
                        csv_writer.writeheader()
                    if rows:
                        csv_writer.writerows(rows)
                else:
                    all_rows.extend(rows)

                # update heuristic memory (only if heuristic)
                if self.firm1_mode != "static" and self.firm1_mode != "RL":
                    self.firm1.update(metrics=m1, price_gap_mean=-mean_gap)  # note sign: Firm1 - Firm2
                if self.firm2_mode != "static":
                    self.firm2.update(metrics=m2, price_gap_mean=mean_gap)   # Firm2 - Firm1
                    
                # RL memory + reward shaping
                reward_diag = self._reward_diagnostics(
                    share=float(m1.share),
                    rev_per_request=float(m1.rev_per_request),
                    mean_gap=float(mean_gap),
                    prev_share=float(self.last_share),
                    prev_rev_per_request=float(self.last_revpr),
                    prev_profit_per_request=float(self.last_profitpr),
                    prev_gap=float(self.last_gap),
                    profit_per_request=float(m1.profit_per_request),
                    profit_margin=self._profit_margin(m1),
                    fulfillment_rate=float(m1.fulfillment_rate),
                    avg_wait_minutes=float(m1.avg_wait_minutes),
                    driver_acceptance_rate=float(m1.driver_acceptance_rate),
                )
                if self.firm1_mode == "RL" and rl_step is not None:
                    action, s_ts, logits, val = rl_step
                    reward = float(reward_diag["reward"])
                    self.last_reward = reward
                    done = (t == timesteps_per_day - 1)
                    self.firm1.agent.store(s_ts, action, float(reward), done, None, logits, val)
                    if done:
                        self.firm1.stabilize_after_batch(
                            share=float(m1.share),
                            price_gap_f2_minus_f1=float(mean_gap),
                            city_base=float(base.base_fare),
                            city_pmin=float(base.per_minute),
                            city_pmile=float(base.per_mile),
                            city_booking=float(base.booking_fee),
                            city_airport=float(base.airport_fee),
                            profit_per_request=float(m1.profit_per_request),
                            fulfillment_rate=float(m1.fulfillment_rate),
                        )
                    reward_sum += float(reward)
                for component_key in (
                    "reward_share_component",
                    "reward_profit_component",
                    "reward_momentum_component",
                    "reward_profit_delta",
                    "reward_gap_delta",
                    "reward_margin_penalty",
                    "reward_price_discipline_penalty",
                    "reward_price_gap_deviation",
                    "reward_price_gap_penalty",
                    "reward_served_share",
                    "reward_driver_unfulfilled_penalty",
                    "reward_driver_fulfillment_component",
                    "reward_driver_wait_penalty",
                    "reward_driver_rejection_penalty",
                ):
                    reward_component_sums[component_key] += float(reward_diag.get(component_key, 0.0))

                share_sum += float(m1.share)
                revpr_sum += float(m1.rev_per_request)
                profit_sum += float(m1.profit_per_request)
                gap_sum += float(mean_gap)
                fulfillment_sum += float(m1.fulfillment_rate)
                wait_sum += float(m1.avg_wait_minutes)
                driver_accept_sum += float(m1.driver_acceptance_rate)
                driver_paypr_sum += float(m1.driver_pay / max(1, m1.total))
                
                share_sum_two += float(m2.share)
                revpr_sum_two += float(m2.rev_per_request)
                profit_sum_two += float(m2.profit_per_request)
                no_ride_share_sum += float(max(0.0, 1.0 - float(m1.share) - float(m2.share)))
                
                self.last_share = float(m1.share)
                self.last_revpr = float(m1.rev_per_request)
                self.last_gap = float(mean_gap)
                self.last_profitpr = float(m1.profit_per_request)
                self.last_fulfillment = float(m1.fulfillment_rate)
                self.last_acceptance = float(m1.driver_acceptance_rate)
                self.last_wait = float(m1.avg_wait_minutes)
                self.last_driver_paypr = float(m1.driver_pay / max(1, m1.total))
                self.airport_rate_last = airport_rate
                self.mean_distance_last = mean_dist
            
            ppo_metrics = {
                "loss": 0.0,
                "approx_kl": 0.0,
                "clipfrac": 0.0,
                "ent_coeff": 0.0,
                "exploration_rate": 0.0,
                "policy_entropy_fraction": 1.0,
                "action_coverage": 0.0,
                "explained_variance": 0.0,
            }
            if self.firm1_mode == "RL":
                ppo_metrics = self.firm1.agent.update(epochs=self.ppo_update_epochs, batch_size=self.ppo_batch_size)
                
            #print("firm 1 revenue per request sum", str(revpr_sum))
            #print("firm 1 market share sum", str(share_sum))
            
            #print("firm 2 revenue per request sum", str(revpr_sum_two))
            #print("firm 2 market share sum", str(share_sum_two))

            avg_share = share_sum / max(1, timesteps_per_day)
            avg_revpr = revpr_sum / max(1, timesteps_per_day)
            avg_gap = gap_sum / max(1, timesteps_per_day)
            avg_profitpr = profit_sum / max(1, timesteps_per_day)
            avg_no_ride_share = no_ride_share_sum / max(1, timesteps_per_day)
            avg_fulfillment = fulfillment_sum / max(1, timesteps_per_day)
            avg_wait = wait_sum / max(1, timesteps_per_day)
            avg_driver_accept = driver_accept_sum / max(1, timesteps_per_day)
            avg_driver_paypr = driver_paypr_sum / max(1, timesteps_per_day)
            dominant_action = int(action_counts.most_common(1)[0][0]) if action_counts else -1
            dominant_action_rate = (
                float(action_counts.most_common(1)[0][1] / max(1, timesteps_per_day))
                if action_counts
                else 0.0
            )
            avg_reward = (
                (reward_sum / max(1, timesteps_per_day))
                if self.firm1_mode == "RL"
                else self._reward_base(
                    avg_share,
                    avg_revpr,
                    price_gap_f2_minus_f1=avg_gap,
                    profit_per_request=avg_profitpr,
                    fulfillment_rate=avg_fulfillment,
                    avg_wait_minutes=avg_wait,
                    driver_acceptance_rate=avg_driver_accept,
                )
            )
            self.run_logs.append({
                "day": d + 1,
                "avg_share": float(avg_share),
                "avg_revpr": float(avg_revpr),
                "avg_profitpr": float(avg_profitpr),
                "avg_gap": float(avg_gap),
                "avg_no_ride_share": float(avg_no_ride_share),
                "avg_fulfillment_rate": float(avg_fulfillment),
                "avg_wait_minutes": float(avg_wait),
                "driver_acceptance_rate": float(avg_driver_accept),
                "driver_pay_per_request": float(avg_driver_paypr),
                "driver_reward_scale": float(self.driver_reward_scale_current),
                "avg_reward": float(avg_reward),
                **{
                    k: float(v / max(1, timesteps_per_day))
                    for k, v in sorted(reward_component_sums.items())
                },
                "last_action": int(last_action),
                "dominant_action": dominant_action,
                "dominant_action_rate": dominant_action_rate,
                "action_counts": json.dumps(dict(sorted(action_counts.items())), sort_keys=True),
                "ppo_approx_kl": float(ppo_metrics.get("approx_kl", 0.0)),
                "ppo_clipfrac": float(ppo_metrics.get("clipfrac", 0.0)),
                "ppo_entropy": float(ppo_metrics.get("entropy", 0.0)),
                "ppo_policy_entropy": float(ppo_metrics.get("policy_entropy", 0.0)),
                "ppo_policy_entropy_fraction": float(ppo_metrics.get("policy_entropy_fraction", 1.0)),
                "ppo_ent_coeff": float(ppo_metrics.get("ent_coeff", 0.0)),
                "ppo_exploration_rate": float(ppo_metrics.get("exploration_rate", 0.0)),
                "ppo_action_coverage": float(ppo_metrics.get("action_coverage", 0.0)),
                "ppo_explained_variance": float(ppo_metrics.get("explained_variance", 0.0)),
            })
            for k in self.shared_edit_keys:
                self.run_logs[-1][f"firm1_{k}"] = float(get_coeff(base, self.firm1.overrides, k))
                self.run_logs[-1][f"firm2_{k}"] = float(get_coeff(base, self.firm2.overrides, k))
            
            recent_rewards = [float(x["avg_reward"]) for x in self.run_logs[-self.reward_convergence_window:]]
            reward_std = float(np.std(recent_rewards)) if recent_rewards else 1.0
            reward_delta = (
                float(abs(recent_rewards[-1] - recent_rewards[0]) / max(1, len(recent_rewards) - 1))
                if len(recent_rewards) >= 2
                else 1.0
            )
            reward_converged = (
                len(recent_rewards) >= self.reward_convergence_window
                and (d + 1) >= self.convergence_min_days
                and reward_std <= self.reward_convergence_tol
                and reward_delta <= self.reward_trend_tol
                and float(ppo_metrics.get("approx_kl", 0.0)) <= 0.025
                and float(ppo_metrics.get("clipfrac", 0.0)) <= 0.22
                and float(ppo_metrics.get("action_coverage", 0.0)) >= 0.95
                and float(ppo_metrics.get("policy_entropy_fraction", 1.0)) <= 0.40
            )
            
            self._convergence_streak = int(self._convergence_streak + 1) if reward_converged else 0
            reward_converged = bool(self._convergence_streak >= self.convergence_required_streak)
            self.run_logs[-1]["reward_window_std"] = float(reward_std)
            self.run_logs[-1]["reward_window_delta"] = float(reward_delta)
            self.run_logs[-1]["reward_convergence_streak"] = int(self._convergence_streak)
            self.run_logs[-1]["reward_converged"] = bool(reward_converged)
            if reward_converged and self.convergence_day is None:
                self.convergence_day = int(d + 1)
                self.convergence_window_std_at_day = float(reward_std)
                self.convergence_delta_per_day_at_day = float(reward_delta)
                print(
                    f"[Convergence] Optimization converged at day {self.convergence_day} "
                    f"(window_std={reward_std:.4f}, delta/day={reward_delta:.4f})."
                )
            
            # print every ~10% of days
            k = max(1, days // 10)
            if (d + 1) % k == 0 or (d + 1) == 1 or (d + 1) == days:

                print(
                    f"[Day {d+1}/{days}] avg_share(F1)={avg_share:.3f} avg_revPR(F1)=${avg_revpr:.2f} "
                    f"avg_profitPR(F1)=${avg_profitpr:.2f} avg_gap(F2-F1)=${avg_gap:.2f} "
                    f"no_ride_share={avg_no_ride_share:.3f} fulfill={avg_fulfillment:.3f} "
                    f"accept={avg_driver_accept:.3f}"
                )
                if self.firm1_mode == "RL":
                    print(
                        f"  [PPO] KL={float(ppo_metrics.get('approx_kl', 0.0)):.4f} "
                        f"clipfrac={float(ppo_metrics.get('clipfrac', 0.0)):.3f} "
                        f"ent={float(ppo_metrics.get('entropy', 0.0)):.3f} "
                        f"policy_ent={float(ppo_metrics.get('policy_entropy_fraction', 1.0)):.3f} "
                        f"ent_coeff={float(ppo_metrics.get('ent_coeff', 0.0)):.4f} "
                        f"explore={float(ppo_metrics.get('exploration_rate', 0.0)):.3f} "
                        f"coverage={float(ppo_metrics.get('action_coverage', 0.0)):.2f} "
                        f"lr={float(ppo_metrics.get('lr', 0.0)):.6f} "
                        f"kl_stop={bool(ppo_metrics.get('stopped_early_kl', False))}"
                    )
                    
            if self.firm1_mode == "RL":
                progress = float((d + 1) / max(1, days))
                self.firm1.configure_training_controls(
                    progress=progress,
                    reward_converged=reward_converged,
                    reward_std=reward_std,
                )
        
        if self.run_logs:
            final = self.run_logs[-1]
            summary_std = (
                float(self.convergence_window_std_at_day)
                if self.convergence_window_std_at_day is not None
                else float(final.get("reward_window_std", 0.0))
            )
            summary_delta = (
                float(self.convergence_delta_per_day_at_day)
                if self.convergence_delta_per_day_at_day is not None
                else float(final.get("reward_window_delta", 0.0))
            )
            print(
                "[Convergence Summary] "
                f"converged_day={self.convergence_day if self.convergence_day is not None else 'not reached'} "
                f"window_std={summary_std:.4f} "
                f"delta/day={summary_delta:.4f}"
            )
            
        if profiles_out:
            _ensure_parent_dir(profiles_out)
            _write_csv(profiles_out, sampled_profile_rows)
            print(f">>> Saved sampled profiles -> {profiles_out} (rows={len(sampled_profile_rows)})")
            if profile_limit_reached:
                print(f">>> Profile export capped at profiles_log_limit={profiles_log_limit} rows.")
                
        if csv_file is not None:
            csv_file.close()
            print(f"Saved -> {out_path}")

        return all_rows
    
    def get_final_manipulated_coefficients(self) -> Dict[str, Dict[str, float]]:
        """Return final values for the shared manipulated coefficient set for each firm."""
        base = self.market.curr_market
        return {
            "firm1": {
                k: float(get_coeff(base, self.firm1.overrides, k)) for k in self.shared_edit_keys
            },
            "firm2": {
                k: float(get_coeff(base, self.firm2.overrides, k)) for k in self.shared_edit_keys
            },
        }


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("No rows to write.")
        return
    cols: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                cols.append(key)
    _ensure_parent_dir(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

def _ensure_parent_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def _write_distribution_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    counters = {
        "DayOfWeek": Counter(),
        "Weather": Counter(),
        "Hour": Counter(),
        "Airport": Counter(),
        "Service": Counter(),
        "Choice": Counter(),
        "DistanceBin": Counter(),
    }

    for r in rows:
        counters["DayOfWeek"][str(r.get("DayOfWeek"))] += 1
        counters["Weather"][str(r.get("Weather"))] += 1
        counters["Hour"][str(r.get("Hour"))] += 1
        counters["Airport"][str(r.get("Airport"))] += 1
        counters["Service"][str(r.get("Service"))] += 1
        counters["Choice"][str(r.get("Choice"))] += 1

        dist = float(r.get("TravelDistance", 0.0))
        if dist < 2:
            b = "0-2"
        elif dist < 5:
            b = "2-5"
        elif dist < 10:
            b = "5-10"
        else:
            b = "10+"
        counters["DistanceBin"][b] += 1

    total = max(1, len(rows))
    out_rows: List[Dict[str, Any]] = []
    for param, c in counters.items():
        if param in {"DayOfWeek", "Hour"}:
            sorted_items = sorted(c.items(), key=lambda kv: int(kv[0]))
        elif param == "Airport":
            sorted_items = sorted(c.items(), key=lambda kv: (kv[0] != "False", kv[0]))
        else:
            sorted_items = sorted(c.items(), key=lambda kv: kv[0])

        for v, n in sorted_items:
            out_rows.append({
                "parameter": param,
                "value": v,
                "count": int(n),
                "share": float(n / total),
            })

    _ensure_parent_dir(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["parameter", "value", "count", "share"])
        w.writeheader()
        w.writerows(out_rows)


def _plot_reports(
    rows: List[Dict[str, Any]],
    run_logs: List[Dict[str, Any]],
    training_logs: List[Dict[str, Any]],
    evaluation_logs: List[Dict[str, Any]],
    prefix: str,
    threshold_profiles: Optional[List[Dict[str, Any]]] = None,
) -> None:
    for name, logs in (("training_diagnostics", training_logs), ("evaluation_diagnostics", evaluation_logs)):
        if logs:
            out = f"{prefix}_{name}.csv"
            _write_csv(out, logs)
            print(f"Saved -> {out}")
    
    
    threshold_rows = list(threshold_profiles or [])
    if not threshold_rows and rows:
        seen_thresholds = set()
        for r in rows:
            if "PriceThreshold" not in r:
                continue
            key = (r.get("ProfileID", id(r)), r.get("PriceThreshold"), r.get("PriceThresholdSource"))
            if key in seen_thresholds:
                continue
            seen_thresholds.add(key)
            threshold_rows.append(r)

    if threshold_rows:
        summary: Dict[str, List[float]] = {}
        for r in threshold_rows:
            try:
                value = float(r.get("PriceThreshold"))
            except (TypeError, ValueError):
                continue
            source = str(r.get("PriceThresholdSource", "unknown"))
            summary.setdefault(source, []).append(value)
        if summary:
            summary_path = f"{prefix}_price_threshold_distribution.csv"
            _ensure_parent_dir(summary_path)
            with open(summary_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["source", "count", "mean", "std", "min", "p25", "median", "p75", "max"])
                w.writeheader()
                for source, vals in sorted(summary.items()):
                    arr = np.asarray(vals, dtype=float)
                    w.writerow({
                        "source": source,
                        "count": int(arr.size),
                        "mean": float(np.mean(arr)),
                        "std": float(np.std(arr)),
                        "min": float(np.min(arr)),
                        "p25": float(np.percentile(arr, 25)),
                        "median": float(np.percentile(arr, 50)),
                        "p75": float(np.percentile(arr, 75)),
                        "max": float(np.max(arr)),
                    })
            print(f"Saved -> {summary_path}")
            
    if importlib.util.find_spec("matplotlib") is None:
        print("[WARN] matplotlib not installed; skipping graph generation.")
        return

    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    
    def _extract_reward(entry: Dict[str, Any]) -> float:
        """Extract reward value from heterogeneous log schemas."""
        if "reward" in entry:
            return float(entry["reward"])
        if "avg_reward" in entry:
            return float(entry["avg_reward"])
        if "rl_share" in entry and "rl_revenue" in entry:
            return float(
                np.clip(
                    (0.60 * np.clip(float(entry["rl_share"]), 0.0, 1.0))
                    + (0.20 * np.tanh((float(entry["rl_revenue"]) - 10.0) / 8.0)),
                    -1.0,
                    1.0,
                )
            )
        return 0.0
    
    _ensure_parent_dir(prefix + "_dummy")
    
    if threshold_rows:
        threshold_values_by_source: Dict[str, List[float]] = {}
        for r in threshold_rows:
            try:
                value = float(r.get("PriceThreshold"))
            except (TypeError, ValueError):
                continue
            threshold_values_by_source.setdefault(str(r.get("PriceThresholdSource", "unknown")), []).append(value)
        if threshold_values_by_source:
            fig, ax = plt.subplots(figsize=(8, 4))
            bins = np.linspace(0.5, 5.0, 19)
            for source, vals in sorted(threshold_values_by_source.items()):
                weights = np.ones(len(vals), dtype=float) * (100.0 / max(1, len(vals)))
                ax.hist(vals, bins=bins, weights=weights, alpha=0.55, label=f"{source} (n={len(vals)})")
            ax.set_title("Profile Price-Threshold Distribution")
            ax.set_xlabel("PriceThreshold ($ fare gap)")
            ax.set_ylabel("% of profiles within source")
            ax.legend(loc="best")
            fig.tight_layout()
            out = f"{prefix}_dist_PriceThreshold.png"
            _ensure_parent_dir(out)
            fig.savefig(out, dpi=150)
            print(f"Saved graph -> {out}")
            plt.close(fig)

    # 1) Ride parameter distributions
    params = ["DayOfWeek", "Weather", "Hour", "Airport", "Service", "Choice"]
    for p in params:
        c = Counter(str(r.get(p)) for r in rows)
        if not c:
            continue

        if p in {"DayOfWeek", "Hour"}:
            xs = sorted(c.keys(), key=lambda x: int(x))
        elif p == "Airport":
            xs = sorted(c.keys(), key=lambda x: (x != "False", x))
        else:
            xs = sorted(c.keys())

        total = max(1, sum(c.values()))
        ys = [100.0 * c[x] / total for x in xs]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(xs, ys)
        ax.set_title(f"Ride Distribution by {p}")
        ax.set_xlabel(p)
        ax.set_ylabel("% of rides")
        fig.tight_layout()
        out = f"{prefix}_dist_{p}.png"
        _ensure_parent_dir(out)
        fig.savefig(out, dpi=150)
        print(f"Saved graph -> {out}")
        plt.close(fig)

    # 2) Distance histogram
    if rows:
        dvals = [float(r.get("TravelDistance", 0.0)) for r in rows]
        weights = np.ones(len(dvals), dtype=float) * (100.0 / max(1, len(dvals)))
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(dvals, bins=20, weights=weights)
        ax.set_title("Ride Distance Distribution")
        ax.set_xlabel("TravelDistance")
        ax.set_ylabel("% of rides")
        fig.tight_layout()
        out = f"{prefix}_dist_TravelDistance.png"
        _ensure_parent_dir(out)
        fig.savefig(out, dpi=150)
        print(f"Saved graph -> {out}")
        plt.close(fig)

    def _plot_timeseries(ax: Any, xs: List[int], ys: List[float], *args: Any, **kwargs: Any) -> Any:
        """Plot a time series with visible single-point smoke-test output."""
        kwargs.setdefault("marker", "o")
        kwargs.setdefault("markersize", 4)
        line_objs = ax.plot(xs, ys, *args, **kwargs)
        if len(xs) == 1:
            x = float(xs[0])
            ax.set_xlim(x - 0.5, x + 0.5)
        return line_objs
    
    def _moving_average(values: List[float], window: int = 20) -> List[float]:
        out: List[float] = []
        for i in range(len(values)):
            start = max(0, i + 1 - int(max(1, window)))
            out.append(float(np.mean(values[start : i + 1])))
        return out

    # 3) run reward trajectory + convergence diagnostics
    if run_logs:
        xs = [int(r["day"]) for r in run_logs]
        ys = [_extract_reward(r) for r in run_logs]
        converged_days = [int(r["day"]) for r in run_logs if bool(r.get("reward_converged", False))]
        fig, ax = plt.subplots(figsize=(9, 4))
        _plot_timeseries(ax, xs, ys, label="avg_reward")
        if converged_days:
            conv_day = int(converged_days[0])
            conv_reward = float(next((r["avg_reward"] for r in run_logs if int(r["day"]) == conv_day), ys[-1]))
            ax.axvline(conv_day, color="tab:green", linestyle="--", linewidth=1.2, label=f"converged day={conv_day}")
            ax.scatter([conv_day], [conv_reward], color="tab:green", zorder=3)
        ax.set_title("Run Reward Trajectory")
        ax.set_xlabel("Day")
        ax.set_ylabel("Reward")
        ax.legend(loc="best")
        fig.tight_layout()
        out = f"{prefix}_reward_run.png"
        _ensure_parent_dir(out)
        fig.savefig(out, dpi=150)
        print(f"Saved graph -> {out}")
        plt.close(fig)
        
        stds = [float(r.get("reward_window_std", np.nan)) for r in run_logs]
        deltas = [float(r.get("reward_window_delta", np.nan)) for r in run_logs]
        if any(np.isfinite(v) for v in stds) or any(np.isfinite(v) for v in deltas):
            fig, ax1 = plt.subplots(figsize=(9, 4))
            _plot_timeseries(ax1, xs, stds, color="tab:blue", label="window std")
            ax1.set_xlabel("Day")
            ax1.set_ylabel("Reward window std", color="tab:blue")
            ax1.tick_params(axis="y", labelcolor="tab:blue")

            ax2 = ax1.twinx()
            _plot_timeseries(ax2, xs, deltas, color="tab:orange", label="|delta| per day")
            ax2.set_ylabel("Reward trend magnitude", color="tab:orange")
            ax2.tick_params(axis="y", labelcolor="tab:orange")

            if converged_days:
                ax1.axvline(converged_days[0], color="tab:green", linestyle="--", linewidth=1.2)

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
            ax1.set_title("Optimization Convergence Diagnostics")
            fig.tight_layout()
            out = f"{prefix}_convergence_run.png"
            _ensure_parent_dir(out)
            fig.savefig(out, dpi=150)
            print(f"Saved graph -> {out}")
            plt.close(fig)

    # 4) training trajectory (run_experiment)
    if training_logs:
        xs = [int(r["batch"]) + 1 for r in training_logs]
        ys = [float(r["avg_reward"]) for r in training_logs]
        fig, ax = plt.subplots(figsize=(9, 4))
        _plot_timeseries(ax, xs, ys, label="avg reward", alpha=0.45)
        if len(ys) > 1:
            _plot_timeseries(ax, xs, _moving_average(ys, 20), label="moving avg20", marker="", linewidth=2.0)
        if any("reward_base" in r for r in training_logs):
            base_ys = [float(r.get("reward_base", np.nan)) for r in training_logs]
            _plot_timeseries(ax, xs, base_ys, label="base reward", marker="", linestyle="--", alpha=0.8)
        
        y_min, y_max = float(np.nanmin(ys)), float(np.nanmax(ys))
        y_span = max(1e-6, y_max - y_min)
        y_pad = max(0.02, 0.10 * y_span)
        y_lo, y_hi = y_min - y_pad, y_max + y_pad

        # Use denser major ticks (with minor ticks in-between) so reward movements are easier to inspect.
        tick_candidates = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
        y_total = max(1e-6, y_hi - y_lo)
        major_step = next((step for step in tick_candidates if (y_total / step) <= 12), tick_candidates[-1])

        ax.set_ylim(y_lo, y_hi)
        ax.yaxis.set_major_locator(MultipleLocator(major_step))
        ax.yaxis.set_minor_locator(MultipleLocator(max(major_step / 2.0, 0.005)))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.grid(axis="y", which="major", linestyle="--", alpha=0.35)
        ax.grid(axis="y", which="minor", linestyle=":", alpha=0.20)
        
        ax.set_title("Training Reward Trajectory")
        ax.set_xlabel("Batch")
        ax.set_ylabel("Avg Reward")
        ax.legend(loc="best")
        fig.tight_layout()
        out = f"{prefix}_reward_training.png"
        _ensure_parent_dir(out)
        fig.savefig(out, dpi=150)
        print(f"Saved graph -> {out}")
        plt.close(fig)

    # 5) evaluation trajectory (run_experiment)
    if evaluation_logs:
        xs = [int(r["day"]) for r in evaluation_logs]
        ys = [_extract_reward(r) for r in evaluation_logs]
        fig, ax = plt.subplots(figsize=(9, 4))
        _plot_timeseries(ax, xs, ys, label="reward", alpha=0.45)
        if len(ys) > 1:
            _plot_timeseries(ax, xs, _moving_average(ys, 20), label="moving avg20", marker="", linewidth=2.0)
        if any("reward_base" in r for r in evaluation_logs):
            base_ys = [float(r.get("reward_base", np.nan)) for r in evaluation_logs]
            _plot_timeseries(ax, xs, base_ys, label="base reward", marker="", linestyle="--", alpha=0.8)
        ax.set_title("Evaluation Reward Trajectory")
        ax.set_xlabel("Day")
        ax.set_ylabel("Reward")
        ax.legend(loc="best")
        fig.tight_layout()
        out = f"{prefix}_reward_evaluation.png"
        _ensure_parent_dir(out)
        fig.savefig(out, dpi=150)
        print(f"Saved graph -> {out}")
        plt.close(fig)
        
    def _plot_experiment_diagnostics(logs: List[Dict[str, Any]], label: str, x_key: str) -> None:
        if not logs:
            return
        xs = [int(r[x_key]) + (1 if x_key == "batch" else 0) for r in logs]

        share_key = "avg_share" if "avg_share" in logs[0] else "rl_share"
        revenue_key = "avg_rev_per_request" if "avg_rev_per_request" in logs[0] else "rl_revenue"
        gap_key = "avg_price_gap_f2_minus_f1" if "avg_price_gap_f2_minus_f1" in logs[0] else "price_gap_f2_minus_f1"
        if all(k in logs[0] for k in (share_key, revenue_key, gap_key)):
            fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
            _plot_timeseries(axes[0], xs, [float(r[share_key]) for r in logs], label="share", color="tab:blue")
            axes[0].set_ylabel("Share")
            axes[0].grid(axis="y", linestyle="--", alpha=0.25)
            _plot_timeseries(axes[1], xs, [float(r[revenue_key]) for r in logs], label="rev/request", color="tab:green")
            axes[1].set_ylabel("Revenue/request")
            axes[1].grid(axis="y", linestyle="--", alpha=0.25)
            _plot_timeseries(axes[2], xs, [float(r[gap_key]) for r in logs], label="F2-F1 gap", color="tab:orange")
            axes[2].axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
            axes[2].set_ylabel("Price gap")
            axes[2].set_xlabel("Batch" if x_key == "batch" else "Day")
            axes[2].grid(axis="y", linestyle="--", alpha=0.25)
            fig.suptitle(f"{label.title()} Business Diagnostics")
            fig.tight_layout()
            out = f"{prefix}_{label}_business_diagnostics.png"
            _ensure_parent_dir(out)
            fig.savefig(out, dpi=150)
            print(f"Saved graph -> {out}")
            plt.close(fig)

        ppo_keys = [
            "ppo_approx_kl",
            "ppo_clipfrac",
            "ppo_entropy",
            "ppo_policy_entropy_fraction",
            "ppo_ent_coeff",
            "ppo_exploration_rate",
            "ppo_action_coverage",
            "ppo_explained_variance",
        ]
        present_ppo = [k for k in ppo_keys if any(k in r for r in logs)]
        if present_ppo:
            fig, ax = plt.subplots(figsize=(9, 4))
            for k in present_ppo:
                _plot_timeseries(ax, xs, [float(r.get(k, np.nan)) for r in logs], label=k.replace("ppo_", ""), marker="")
            ax.set_title(f"{label.title()} PPO Diagnostics")
            ax.set_xlabel("Batch" if x_key == "batch" else "Day")
            ax.set_ylabel("Value")
            ax.legend(loc="best")
            ax.grid(axis="y", linestyle="--", alpha=0.25)
            fig.tight_layout()
            out = f"{prefix}_{label}_ppo_diagnostics.png"
            _ensure_parent_dir(out)
            fig.savefig(out, dpi=150)
            print(f"Saved graph -> {out}")
            plt.close(fig)

        coeff_keys: List[str] = sorted(
            {
                k[len("firm1_"):]
                for r in logs
                for k in r.keys()
                if k.startswith("firm1_") and ("firm2_" + k[len("firm1_"):]) in r
            }
        )
        for coeff in coeff_keys:
            fig, ax = plt.subplots(figsize=(9, 4))
            _plot_timeseries(ax, xs, [float(r.get(f"firm1_{coeff}", np.nan)) for r in logs], label=f"Firm1 {coeff}")
            _plot_timeseries(ax, xs, [float(r.get(f"firm2_{coeff}", np.nan)) for r in logs], label=f"Firm2 {coeff}")
            ax.set_title(f"{label.title()} Coefficient Trajectory: {coeff}")
            ax.set_xlabel("Batch" if x_key == "batch" else "Day")
            ax.set_ylabel("Coefficient value")
            ax.legend(loc="best")
            fig.tight_layout()
            out = f"{prefix}_{label}_coeff_{coeff}_trajectory.png"
            _ensure_parent_dir(out)
            fig.savefig(out, dpi=150)
            print(f"Saved graph -> {out}")
            plt.close(fig)

    _plot_experiment_diagnostics(training_logs, "training", "batch")
    _plot_experiment_diagnostics(evaluation_logs, "evaluation", "day")

    # 6) manipulated coefficient trajectories
    if run_logs:
        xs = [int(r["day"]) for r in run_logs]
        coeff_keys: List[str] = sorted(
            {
                k[len("firm1_"):]
                for r in run_logs
                for k in r.keys()
                if k.startswith("firm1_") and ("firm2_" + k[len("firm1_"):]) in r
            }
        )
        for coeff in coeff_keys:
            y1: List[float] = []
            y2: List[float] = []
            valid = True
            for r in run_logs:
                v1 = r.get(f"firm1_{coeff}")
                v2 = r.get(f"firm2_{coeff}")
                if v1 is None or v2 is None:
                    valid = False
                    break
                y1.append(float(v1))
                y2.append(float(v2))
            if not valid or not y1:
                continue

            fig, ax = plt.subplots(figsize=(9, 4))
            _plot_timeseries(ax, xs, y1, label=f"Firm1 {coeff}")
            _plot_timeseries(ax, xs, y2, label=f"Firm2 {coeff}")
            ax.set_title(f"Coefficient Trajectory: {coeff}")
            ax.set_xlabel("Day")
            ax.set_ylabel("Coefficient value")
            ax.legend(loc="best")
            fig.tight_layout()
            out = f"{prefix}_coeff_{coeff}_trajectory.png"
            _ensure_parent_dir(out)
            fig.savefig(out, dpi=150)
            print(f"Saved graph -> {out}")
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", type=str, default="New York City")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--timesteps", type=int, default=8)
    parser.add_argument("--customers", type=int, default=500)
    parser.add_argument("--choice_mode", type=str, default="cognitive", choices=["parametric", "cognitive", "llm"])
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--openai_api_key", type=str, default=None, help="Optional OpenAI API key override. If omitted, OPENAI_API_KEY env var is used.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed. If omitted, a new seed is generated each run.")
    parser.add_argument("--out", type=str, default="market_runs.csv")

    dynamic_mode_choices = ["heuristic", "heuristic_margin", "heuristic_random"]
    parser.add_argument("--firm1_mode", type=str, default="heuristic", choices=["RL", *dynamic_mode_choices, "static"])
    parser.add_argument("--firm2_mode", type=str, default="static", choices=[*dynamic_mode_choices, "static"])

    parser.add_argument("--firm1_static_values", type=str, default="")
    parser.add_argument("--firm2_static_values", type=str, default="")

    parser.add_argument("--pool", type=int, default=20000, help="Static customer pool size.")
    parser.add_argument(
        "--deterministic_torch",
        action="store_true",
        help="Use deterministic Torch kernels where available (may reduce performance).",
    )
    
    parser.add_argument("--report_prefix", type=str, default="artifacts/report")
    parser.add_argument("--run_experiment", action="store_true", help="Run workflow-aligned training/eval experiment")
    parser.add_argument("--train_timesteps", type=int, default=1000)
    parser.add_argument("--train_customers", type=int, default=5000)
    parser.add_argument("--train_steps_per_day", type=int, default=10, help="Synthetic training timesteps per day (run_experiment mode).")
    parser.add_argument("--ppo_update_interval_days", type=int, default=5, help="How many synthetic training days to collect before each PPO optimizer update.")
    parser.add_argument("--deterministic_experiment_seed", action="store_true", help="If set, keep run_experiment fully deterministic with --seed.")
    parser.add_argument("--eval_timesteps", type=int, default=200)
    parser.add_argument("--eval_customers", type=int, default=1000)
    parser.add_argument("--profiles_out", type=str, default="artifacts/sampled_profiles.csv")
    parser.add_argument("--profiles_log_limit", type=int, default=200000)
    parser.add_argument("--reward_share_weight", type=float, default=0.34, help="Positive reward weight for absolute market share; normalized with revenue and dominance weights.")
    parser.add_argument("--reward_revenue_weight", type=float, default=0.33, help="Positive reward weight for revenue per request; normalized with share and dominance weights.")
    parser.add_argument("--reward_overprice_weight", type=float, default=0.20, help="Penalty weight when Firm1 is more expensive than Firm2.")
    parser.add_argument("--reward_rev_scale", type=float, default=25.0, help="Revenue/request value that maps to full revenue reward credit.")
    parser.add_argument("--reward_competitive_weight", type=float, default=0.33, help="Backward-compatible name for sustained-dominance reward weight; normalized with share and revenue weights.")
    parser.add_argument("--reward_trend_weight", type=float, default=0.0, help="Optional small trend-shaping weight for short-term share/profit improvements.")
    parser.add_argument("--reward_profit_scale", type=float, default=12.0, help="Profit/request value that maps to full profit reward credit.")
    parser.add_argument("--reward_underprice_weight", type=float, default=0.15, help="Penalty weight for destructive underpricing or sub-target profit margin.")
    parser.add_argument("--reward_acceptable_discount", type=float, default=2.0, help="Firm1 may be this many dollars cheaper than Firm2 before underpricing penalty starts.")
    parser.add_argument("--min_profit_margin", type=float, default=0.08, help="Minimum contribution margin target used by the margin discipline penalty.")
    parser.add_argument("--driver_cost_per_mile", type=float, default=0.85, help="Approximate per-mile contribution cost for completed rides.")
    parser.add_argument("--driver_cost_per_minute", type=float, default=0.12, help="Approximate per-minute contribution cost for completed rides.")
    parser.add_argument("--fixed_trip_cost", type=float, default=1.25, help="Approximate fixed platform/driver cost for completed rides.")
    parser.add_argument("--airport_cost", type=float, default=2.0, help="Approximate extra cost for airport rides.")
    parser.add_argument("--disable_driver_supply", action="store_true", help="Disable the driver supply/matching layer and use legacy completed-ride costs.")
    parser.add_argument("--use_osmnx", action="store_true", help="Load an OpenStreetMap drive network through OSMnx for future spatial routing hooks.")
    parser.add_argument("--osmnx_place", type=str, default="", help="OSMnx place string; defaults to '<market>, USA'.")
    parser.add_argument("--driver_base_active", type=int, default=260, help="Baseline active drivers for the batch-level driver supply model.")
    parser.add_argument("--driver_reservation_wage", type=float, default=24.0, help="Driver reservation wage used by acceptance and online-supply response.")
    parser.add_argument("--driver_acceptance_mode", type=str, default="expected", choices=["expected", "stochastic"], help="Use deterministic expected driver acceptance for faster RL convergence, or stochastic Bernoulli acceptance for realism.")
    parser.add_argument("--driver_state_smoothing", type=float, default=0.35, help="EMA smoothing alpha for driver features fed into the RL state.")
    parser.add_argument("--driver_reward_fulfillment_weight", type=float, default=0.06, help="Reward bonus weight for completed/chosen fulfillment when driver supply is enabled.")
    parser.add_argument("--driver_reward_wait_weight", type=float, default=0.04, help="Reward penalty weight for long average pickup wait when driver supply is enabled.")
    parser.add_argument("--driver_reward_reject_weight", type=float, default=0.03, help="Reward penalty weight for driver rejection when driver supply is enabled.")
    parser.add_argument("--driver_reward_warmup_fraction", type=float, default=0.30, help="Fraction of training over which driver reward terms ramp from zero to full strength.")
    parser.add_argument("--threshold_cache_path", type=str, default="", help="Optional JSONL path for cached GPT/fallback threshold-enriched profiles.")
    parser.add_argument("--threshold_profile_source", type=str, default="generated", choices=["generated", "cached"], help="Use completely newly generated threshold profiles, or load threshold profiles from --threshold_cache_path.")
    parser.add_argument("--reuse_threshold_cache", action="store_true", help="Deprecated alias for --threshold_profile_source cached.")
    parser.add_argument("--no_save_threshold_cache", action="store_true", help="Do not write threshold profiles to --threshold_cache_path after bootstrap.")
    parser.add_argument("--gpt_threshold_include_rationales", action="store_true", help="Request GPT rationales for thresholds; slower but useful for debugging.")
    parser.add_argument("--gpt_threshold_coldstart_rides", type=int, default=5, help="Cold-start rides per profile sent/summarized for threshold bootstrapping.")
    parser.add_argument("--gpt_threshold_send_raw_rides", action="store_true", help="Send raw cold-start ride rows in addition to compact summaries; slower but more detailed.")
    parser.add_argument("--gpt_threshold_batch_size", type=int, default=20, help="Profiles per GPT price-threshold API request. Larger values improve API utilization but increase payload size.")
    parser.add_argument("--gpt_threshold_max_retries", type=int, default=2, help="Retries per GPT price-threshold batch before deterministic fallback is used.")
    parser.add_argument("--gpt_threshold_failure_pause", type=float, default=1.0, help="Seconds to pause after a failed GPT threshold batch before sending the next batch; helps avoid connection-refused cascades.")
    parser.add_argument("--calibration_csv", type=str, default="", help="Optional historical CSV used to calibrate priors and choice sensitivity.")
    parser.add_argument("--calibration_city", type=str, default="", help="Optional city filter for --calibration_csv; defaults to --market if omitted.")
    parser.add_argument("--calibration_preset", type=str, default="nyc_public", choices=["", "nyc_public"], help="Built-in preset calibration. Defaults to nyc_public (NYC TLC + ACS + weather priors).")
    parser.add_argument("--compare_with_dataset", action="store_true", help="After training/run, compare RL-implied prices against actual paid prices in the Kaggle rideshare dataset files.")
    parser.add_argument("--dataset_glob", type=str, default="*.parquet", help="Glob for dataset file discovery under kagglehub download path.")
    parser.add_argument(
        "--comparison_out", "--comparison-out",
        type=str,
        nargs="?",
        const="artifacts/rl_dataset_price_comparison.csv",
        default="artifacts/rl_dataset_price_comparison.csv",
        help="Output CSV for row-level RL-vs-actual comparison. If provided without a value, defaults to artifacts/rl_dataset_price_comparison.csv.",
    )
    parser.add_argument(
        "--comparison_plot_prefix", "--comparison-plot-prefix",
        type=str,
        nargs="?",
        const="artifacts/rl_dataset_validation",
        default="artifacts/rl_dataset_validation",
        help="Prefix for validation graphs (price/time match) against dataset. If provided without a value, defaults to artifacts/rl_dataset_validation.",
    )
    parser.add_argument("--comparison_limit", type=int, default=50000, help="Max number of dataset rows to score during RL-vs-actual comparison.")
    parser.add_argument("--dataset_preview_rows", type=int, default=5, help="How many raw dataset rows to print once as format preview during RL-vs-actual comparison.")

    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        print(f"[WARN] Ignoring unrecognized CLI args: {unknown_args}")
        
    code_dir = os.path.dirname(os.path.abspath(__file__))

    def _pin_graph_prefix_to_code_dir(prefix: str, flag_name: str) -> str:
        """Force graph outputs to live beside Core.py, even when absolute prefixes are passed."""
        normalized = os.path.normpath(prefix)
        if os.path.isabs(normalized):
            rewritten = os.path.join(code_dir, os.path.basename(normalized))
            print(
                f"[WARN] {flag_name} was absolute ({prefix}); "
                f"rewriting to {rewritten} so graphs are generated in code folder."
            )
            return rewritten
        return os.path.join(code_dir, normalized)

    args.report_prefix = _pin_graph_prefix_to_code_dir(args.report_prefix, "--report_prefix")
    args.comparison_plot_prefix = _pin_graph_prefix_to_code_dir(
        args.comparison_plot_prefix,
        "--comparison_plot_prefix",
    )

    core = Core(
        market_name=args.market,
        seed=args.seed,
        choice_mode=args.choice_mode,
        model_name=args.model,
        openai_api_key=args.openai_api_key,
        firm1_mode=args.firm1_mode,
        firm2_mode=args.firm2_mode,
        firm1_static_values=args.firm1_static_values,
        firm2_static_values=args.firm2_static_values,
        total_customers_pool=args.pool,
        deterministic_torch=args.deterministic_torch,
        reward_share_weight=args.reward_share_weight,
        reward_revenue_weight=args.reward_revenue_weight,
        reward_overprice_weight=args.reward_overprice_weight,
        reward_rev_scale=args.reward_rev_scale,
        reward_competitive_weight=args.reward_competitive_weight,
        reward_trend_weight=args.reward_trend_weight,
        reward_profit_scale=args.reward_profit_scale,
        reward_underprice_weight=args.reward_underprice_weight,
        reward_acceptable_discount=args.reward_acceptable_discount,
        min_profit_margin=args.min_profit_margin,
        driver_cost_per_mile=args.driver_cost_per_mile,
        driver_cost_per_minute=args.driver_cost_per_minute,
        fixed_trip_cost=args.fixed_trip_cost,
        airport_cost=args.airport_cost,
        enable_driver_supply=not args.disable_driver_supply,
        use_osmnx=args.use_osmnx,
        osmnx_place=args.osmnx_place or None,
        driver_base_active=args.driver_base_active,
        driver_reservation_wage=args.driver_reservation_wage,
        driver_acceptance_mode=args.driver_acceptance_mode,
        driver_state_smoothing=args.driver_state_smoothing,
        driver_reward_fulfillment_weight=args.driver_reward_fulfillment_weight,
        driver_reward_wait_weight=args.driver_reward_wait_weight,
        driver_reward_reject_weight=args.driver_reward_reject_weight,
        driver_reward_warmup_fraction=args.driver_reward_warmup_fraction,
        threshold_cache_path=args.threshold_cache_path,
        reuse_threshold_cache=args.reuse_threshold_cache,
        threshold_profile_source=args.threshold_profile_source,
        save_threshold_cache=not args.no_save_threshold_cache,
        gpt_threshold_include_rationales=args.gpt_threshold_include_rationales,
        gpt_threshold_coldstart_rides=args.gpt_threshold_coldstart_rides,
        gpt_threshold_send_raw_rides=args.gpt_threshold_send_raw_rides,
        gpt_threshold_batch_size=args.gpt_threshold_batch_size,
        gpt_threshold_max_retries=args.gpt_threshold_max_retries,
        gpt_threshold_failure_pause=args.gpt_threshold_failure_pause,
    )
    
    if args.calibration_preset:
        calibration = load_calibration_preset(args.calibration_preset)
        core.apply_calibration(calibration)
        print(f"[Calibration] Applied preset={args.calibration_preset} for market={args.market}.")
        print(f"[Calibration] {json.dumps(calibration, indent=2)}")

    if args.calibration_csv:
        city_filter = args.calibration_city if args.calibration_city else args.market
        calibration = derive_calibration(args.calibration_csv, city=city_filter)
        core.apply_calibration(calibration)
        print(f"[Calibration] Applied using {calibration['sample_size']} rows from {args.calibration_csv} (city={city_filter}).")
        print(f"[Calibration] {json.dumps(calibration, indent=2)}")


    if args.run_experiment:
        core.run_experiment(
            train_timesteps=args.train_timesteps,
            train_customers_per_step=args.train_customers,
            eval_timesteps=args.eval_timesteps,
            eval_customers_per_step=args.eval_customers,
            profiles_out=args.profiles_out,
            profiles_log_limit=args.profiles_log_limit,
            train_steps_per_day=args.train_steps_per_day,
            ppo_update_interval_days=args.ppo_update_interval_days,
            stochastic_training=not args.deterministic_experiment_seed,
        )
        rows = []
    else:
        estimated_rows = int(args.days) * int(args.timesteps) * int(args.customers)
        stream_threshold = 1_000_000
        stream_to_disk = estimated_rows > stream_threshold
        if stream_to_disk:
            print(
                f"[Run] Large output detected ({estimated_rows:,} rows). "
                f"Streaming rows directly to {args.out} to avoid high memory usage."
            )
            
        rows = core.run(
            days=args.days,
            timesteps_per_day=args.timesteps,
            customers_per_step=args.customers,
            out_path=args.out if stream_to_disk else None,
            profiles_out=args.profiles_out,
            profiles_log_limit=args.profiles_log_limit,
        )
        final_coeffs = core.get_final_manipulated_coefficients()
        print(
            f"[Final Coefficients after {args.days} days] "
            f"Firm1={json.dumps(final_coeffs['firm1'], sort_keys=True)} "
            f"Firm2={json.dumps(final_coeffs['firm2'], sort_keys=True)}"
        )
        if not stream_to_disk:
            _write_csv(args.out, rows)
            print(f"Saved -> {args.out}")

    if rows:
        dist_csv = f"{args.report_prefix}_ride_distributions.csv"
        _write_distribution_csv(dist_csv, rows)
        print(f"Saved -> {dist_csv}")

    _plot_reports(
        rows=rows,
        run_logs=core.run_logs,
        training_logs=core.training_logs,
        evaluation_logs=core.evaluation_logs,
        prefix=args.report_prefix,
        threshold_profiles=core.synthetic_profile_pool,
    )
    
    if args.compare_with_dataset:
        summary = core.compare_trained_rl_to_dataset(
            dataset_root=_resolve_dataset_path(),
            dataset_glob=args.dataset_glob,
            out_csv=args.comparison_out,
            out_plot_prefix=args.comparison_plot_prefix,
            max_rows=args.comparison_limit,
            preview_rows=args.dataset_preview_rows,
        )
        print(f"[RL vs Actual] {json.dumps(summary, indent=2)}")

if __name__ == "__main__":
    main()