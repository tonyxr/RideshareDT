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
- Wasserstein trust region in policy update
"""

import argparse
import csv
import glob
import gzip
import io
import importlib.util
import json
import os
import random
import sys
import re
import zipfile
from collections import Counter
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
from pricing_models import FirmMetrics, FirmStaticPricer, FirmHeuristicPricer, FirmRLPricer
from coeff_utils import get_coeff, set_coeff
from state_encoder import build_state_vector
from calibration_utils import derive_calibration, load_calibration_preset

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


def _pick_first_numeric(row: Dict[str, Any], names: List[str]) -> Optional[float]:
    for k in names:
        if k not in row or _is_missing(row[k]):
            continue
        val = _to_numeric(row[k])
        if val is not None:
            return val
    return None


def _pick_first_numeric_with_key(row: Dict[str, Any], names: List[str]) -> Tuple[Optional[float], Optional[str]]:
    for k in names:
        if k not in row or _is_missing(row[k]):
            continue
        val = _to_numeric(row[k])
        if val is not None:
            return val, k
    return None, None

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
    keep_ext = (".csv", ".csv.gz", ".zip")
    return sorted({f for f in files if os.path.isfile(f) and f.lower().endswith(keep_ext)})


def _iter_tabular_rows(fpath: str):
    """Yield rows from .csv, .csv.gz, or .zip archives containing csv files."""
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
        reward_share_weight: float = 0.550,
        reward_revenue_weight: float = 0.45,
        reward_overprice_weight: float = 0.35,
        reward_rev_scale: float = 25.0,
        reward_competitive_weight: float = 0.12,
        reward_trend_weight: float = 0.08,
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
        self.training_stable_window = 20
        self.training_stable_tol = 0.015
        self.reward_convergence_window = 40
        self.reward_convergence_tol = 0.025
        self.reward_trend_tol = 0.01
        
        # choice model
        self.choice_mode = choice_mode
        if choice_mode in {"llm", "cognitive"}:
            self.choice_model = LLMChoiceModel(model_name=model_name, api_key=openai_api_key, seed=self.seed)
            if openai_api_key:
                print("[Core] --openai_api_key is ignored; API integration is retired.")
        else:
            self.choice_model = ParametricChoiceModel(seed=self.seed)

        # firms
        self.firm1_mode = firm1_mode
        self.firm2_mode = firm2_mode

        if self.firm1_mode not in {"RL", "heuristic", "static"}:
            raise ValueError("firm1_mode must be one of: RL, heuristic, static")
        if self.firm2_mode not in {"heuristic", "static"}:
            raise ValueError("firm2_mode must be one of: heuristic, static")

        self.opt_keys = ["base_fare", "per_minute"]
        self.shared_edit_keys = list(self.opt_keys)

        if self.firm1_mode == "RL":
            self.firm1 = FirmRLPricer(seed=self.seed, opt_keys=self.shared_edit_keys)
        elif self.firm1_mode == "heuristic":
            self.firm1 = FirmHeuristicPricer(seed=self.seed, managed_keys=self.shared_edit_keys)
        else:
            self.firm1 = FirmStaticPricer()

        if self.firm2_mode == "heuristic":
            self.firm2 = FirmHeuristicPricer(seed=self.seed + 1, managed_keys=self.shared_edit_keys)
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
        
        self.training_logs = []
        self.evaluation_logs = []
        
        self.run_logs = []
        self.convergence_day: Optional[int] = None
        
        self.reward_share_weight = float(max(0.0, reward_share_weight))
        self.reward_revenue_weight = float(max(0.0, reward_revenue_weight))
        self.reward_overprice_weight = float(max(0.0, reward_overprice_weight))
        self.reward_rev_scale = float(max(1e-6, reward_rev_scale))
        self.reward_competitive_weight = float(max(0.0, reward_competitive_weight))
        self.reward_trend_weight = float(max(0.0, reward_trend_weight))

        denom = self.reward_share_weight + self.reward_revenue_weight
        if denom <= 0.0:
            self.reward_share_weight = 0.6
            self.reward_revenue_weight = 0.4
            denom = 1.0

        self.reward_share_weight /= denom
        self.reward_revenue_weight /= denom
        
        self.reward_competitive_scale = 0.75
        self.reward_trend_scale = 0.75
        self.reward_softsign_temp = 1.25

        self.ppo_update_epochs = 5
        self.ppo_batch_size = 256
        
        print(
            "[RewardConfig] "
            f"share={self.reward_share_weight:.2f}, "
            f"revenue={self.reward_revenue_weight:.2f}, "
            f"overprice_penalty={self.reward_overprice_weight:.2f}, "
            f"rev_scale={self.reward_rev_scale:.2f}, "
            f"competitive={self.reward_competitive_weight:.2f}, "
            f"trend={self.reward_trend_weight:.2f}"
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
        f2_base = float(base.base_fare) * float(self.rng.uniform(lo, hi))
        f2_pmin = float(base.per_minute) * float(self.rng.uniform(lo, hi))

        self.firm1.overrides.base_fare = max(0.1, f1_base)
        self.firm1.overrides.per_minute = max(0.01, f1_pmin)
        self.firm2.overrides.base_fare = max(0.1, f2_base)
        self.firm2.overrides.per_minute = max(0.01, f2_pmin)
        
    def _reward_base(
        self,
        share: float,
        rev_per_request: float,
        price_gap_f2_minus_f1: float = 0.0,
    ) -> float:
        """Simplified reward: market share + revenue with light overpricing penalty."""
        
        
        share_f = float(np.clip(share, 0.0, 1.0))
        rev_term = float(np.clip(rev_per_request / self.reward_rev_scale, 0.0, 1.0))
        overprice_gap = float(max(0.0, -price_gap_f2_minus_f1))
        overprice_penalty = float(np.clip((overprice_gap / 2.5) ** 2, 0.0, 1.0))
        
        raw = (
            (self.reward_share_weight * share_f)
            + (self.reward_revenue_weight * rev_term)
            - (self.reward_overprice_weight * overprice_penalty)
        )
        return float(np.clip(raw, -1.0, 1.0))

    def _compute_rl_reward(self, m1: FirmMetrics, mean_gap: float) -> float:
        """Low-complexity reward shaping for better PPO learning signal."""
        
        base_reward = self._reward_base(
            share=float(m1.share),
            rev_per_request=float(m1.rev_per_request),
            price_gap_f2_minus_f1=float(mean_gap),
        )
        share = float(np.clip(m1.share, 0.0, 1.0))
        revpr = float(max(0.0, m1.rev_per_request))

        # Encourage beating the 50% share mark (dense, centered around 0).
        competitive_term = float(np.clip((share - 0.5) / 0.5, -1.0, 1.0))

        # Reward local improvement to reduce variance and speed up adaptation.
        share_delta = float(np.clip(share - self.last_share, -0.20, 0.20) / 0.20)
        rev_delta = float(np.clip((revpr - self.last_revpr) / self.reward_rev_scale, -0.20, 0.20) / 0.20)
        trend_term = 0.5 * (share_delta + rev_delta)
        
        # Simple action-linked signal: reward outcomes that improve share/revenue
        # while not drifting too far into uncompetitive pricing (large negative gap).
        pricing_discipline = float(np.clip(mean_gap / 2.0, -1.0, 1.0))
        efficiency_term = 0.5 * trend_term + 0.5 * pricing_discipline

        raw_reward = (
            base_reward
            + self.reward_competitive_weight * self.reward_competitive_scale * competitive_term
            + self.reward_trend_weight * self.reward_trend_scale * efficiency_term
        )
        # Softsign-like compression keeps gradients informative while avoiding hard clipping saturation.
        reward = float(np.tanh(raw_reward / self.reward_softsign_temp))

        self.last_reward = float(reward)
        return float(reward)
    
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

    def run_experiment(
        self,
        train_timesteps: int = 1000,
        train_customers_per_step: int = 5000,
        eval_timesteps: int = 200,
        eval_customers_per_step: int = 1000,
        profiles_out: Optional[str] = None,
        profiles_log_limit: int = 200000,
    ):
        """Run workflow: synthetic generation summary, on-the-fly RL training, then evaluation."""
        self._initialize_run_distributions()
        self._refresh_profile_pool(rides_per_timestep=train_customers_per_step)

        print(
            f">>> Synthetic setup: profile_pool={self.agent_gen.total_customers}, "
            f"train timesteps={train_timesteps} x {train_customers_per_step} rides, "
            f"eval timesteps={eval_timesteps} x {eval_customers_per_step} rides"
        )
        
        print(">>> Training RL agent on-the-fly (early exploration prioritized)...")
        reward_history: List[float] = []
        stable_count = 0
        stability_notice_emitted = False
        sampled_profile_rows: List[Dict[str, Any]] = []
        profile_limit_reached = False

        for t in range(train_timesteps):
            day_ctx = self.market.sample_day_context()
            hour = self.market.sample_timestep_hour().hour
            sampled_profiles = self.agent_gen.sample_profiles(train_customers_per_step)
            if profiles_out and not profile_limit_reached:
                remaining = int(max(0, profiles_log_limit - len(sampled_profile_rows)))
                if remaining > 0:
                    sampled_profile_rows.extend({"Phase": "train", "Timestep": int(t), **p} for p in sampled_profiles[:remaining])
                profile_limit_reached = len(sampled_profile_rows) >= int(max(0, profiles_log_limit))

            base = self.market.curr_market
            rl_step = None
            if self.firm1_mode == "RL":
                s_vec = self._build_rl_state(day_of_week=day_ctx.day_of_week, hour=hour, weather=day_ctx.weather)
                action, s_ts, logits, val = self.firm1.agent.act(s_vec)
                self.firm1.apply_action(action, self.market)
                rl_step = (action, s_ts, logits, val)
            elif self.firm1_mode == "heuristic":
                self.firm1.act(city_base=base.base_fare, city_pmin=base.per_minute, hour=hour, weather=day_ctx.weather)

            if self.firm2_mode == "heuristic":
                self.firm2.act(city_base=base.base_fare, city_pmin=base.per_minute, hour=hour, weather=day_ctx.weather)

            _, m1, _, mean_gap, _, _ = self.simulate_batch(
                day_of_week=day_ctx.day_of_week,
                weather=day_ctx.weather,
                hour=hour,
                customers_per_step=train_customers_per_step,
                sampled_profiles=sampled_profiles,
            )
            
            reward = self._compute_rl_reward(m1, mean_gap)
            if self.firm1_mode == "RL" and rl_step is not None:
                action, s_ts, logits, val = rl_step
                self.firm1.agent.store(s_ts, action, float(reward), False, None, logits, val)
                ppo_metrics = self.firm1.agent.update(epochs=self.ppo_update_epochs, batch_size=self.ppo_batch_size)
            else:
                ppo_metrics = {"loss": 0.0}
                
            reward_history.append(float(reward))
            if len(reward_history) >= self.training_stable_window:
                recent = reward_history[-self.training_stable_window:]
                if float(np.std(recent)) <= self.training_stable_tol:
                    stable_count += 1
                else:
                    stable_count = 0
                    
            self.training_logs.append({
                "batch": t,
                "avg_reward": float(reward),
                "loss": float(ppo_metrics.get("loss", 0.0)),
            })
            
            if (t + 1) % max(1, train_timesteps // 10) == 0:
                window = reward_history[-min(len(reward_history), 20):]
                moving_avg = float(np.mean(window)) if window else 0.0
                print(f"  [train {t+1}/{train_timesteps}] reward={float(reward):.3f} moving_avg20={moving_avg:.3f}")

            if stable_count >= 3:
                if not stability_notice_emitted:
                    print(
                        f">>> Reward stabilized at timestep {t+1}; "
                        "continuing training to complete all requested timesteps."
                    )
                    stability_notice_emitted = True

            self.last_share = float(m1.share)
            self.last_revpr = float(m1.rev_per_request)
            self.last_gap = float(mean_gap)
        
        print(">>> Evaluating RL agent against static/heuristic opponent with shared profile pool...")
        eval_rewards: List[float] = []
        for t in range(eval_timesteps):
            day_ctx = self.market.sample_day_context()
            hour = self.market.sample_timestep_hour().hour
            sampled_profiles = self.agent_gen.sample_profiles(eval_customers_per_step)
            if profiles_out and not profile_limit_reached:
                remaining = int(max(0, profiles_log_limit - len(sampled_profile_rows)))
                if remaining > 0:
                    sampled_profile_rows.extend({"Phase": "eval", "Timestep": int(t), **p} for p in sampled_profiles[:remaining])
                profile_limit_reached = len(sampled_profile_rows) >= int(max(0, profiles_log_limit))

            base = self.market.curr_market
            if self.firm1_mode == "RL":
                s_vec = self._build_rl_state(day_of_week=day_ctx.day_of_week, hour=hour, weather=day_ctx.weather)
                action, *_ = self.firm1.agent.act(s_vec)
                self.firm1.apply_action(action, self.market)
            elif self.firm1_mode == "heuristic":
                self.firm1.act(city_base=base.base_fare, city_pmin=base.per_minute, hour=hour, weather=day_ctx.weather)

            if self.firm2_mode == "heuristic":
                self.firm2.act(city_base=base.base_fare, city_pmin=base.per_minute, hour=hour, weather=day_ctx.weather)

            _, m1, m2, mean_gap, _, _ = self.simulate_batch(
                day_of_week=day_ctx.day_of_week,
                weather=day_ctx.weather,
                hour=hour,
                customers_per_step=eval_customers_per_step,
                sampled_profiles=sampled_profiles,
            )
            eval_reward = self._reward_base(
                share=float(m1.share),
                rev_per_request=float(m1.rev_per_request),
                price_gap_f2_minus_f1=float(mean_gap),
            )
            eval_rewards.append(float(eval_reward))
            self.evaluation_logs.append({
                "day": t,
                "rl_share": float(m1.share),
                "heuristic_share": float(m2.share),
                "rl_revenue": float(m1.rev_per_request),
                "reward": float(eval_reward),
            })
            self.last_share = float(m1.share)
            self.last_revpr = float(m1.rev_per_request)
            self.last_gap = float(mean_gap)

            if (t + 1) % max(1, eval_timesteps // 10) == 0:
                window = eval_rewards[-min(len(eval_rewards), 20):]
                moving_avg = float(np.mean(window)) if window else 0.0
                print(f"  [eval {t+1}/{eval_timesteps}] reward={float(eval_reward):.3f} moving_avg20={moving_avg:.3f}")

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
       """Validate trained RL pricing on raw NYC rides by predicting trip time + paid price."""
       patterns = [os.path.join(dataset_root, dataset_glob)]
       if "**" not in dataset_glob:
           patterns.append(os.path.join(dataset_root, "**", dataset_glob))

       files: List[str] = []
       for pat in patterns:
           files.extend(glob.glob(pat, recursive=True))
       files = sorted({f for f in files if os.path.isfile(f) and f.lower().endswith(('.csv', '.parquet'))})

       if not files:
           raise FileNotFoundError(
               f"No CSV/Parquet files found under dataset_root={dataset_root!r} with glob={dataset_glob!r}"
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
                           print(f"[Dataset Preview] row_{idx}: {json.dumps(sample, ensure_ascii=False)}")
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
                       actual_duration = _pick_first_numeric(raw, duration_cols)
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
                       if actual_duration is not None:
                           # Normalize common second-based fields to minutes.
                           if any((k in raw and not _is_missing(raw.get(k))) for k in duration_seconds_cols):
                               actual_duration = float(actual_duration) / 60.0
                           elif actual_duration > 300.0:
                               actual_duration = float(actual_duration) / 60.0
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
                   
           except Exception:
               continue
           if kept >= max_rows:
               break        
           if file_kept > 0:
               files_with_rows += 1

       if out_csv:
           _ensure_parent_dir(out_csv)
           _write_csv(out_csv, rows_out)
           
       if out_plot_prefix and rows_out:
           self._plot_dataset_validation(rows_out=rows_out, out_plot_prefix=out_plot_prefix)

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
       return summary
   
    @staticmethod
    def _plot_dataset_validation(rows_out: List[Dict[str, Any]], out_plot_prefix: str) -> None:
        if importlib.util.find_spec("matplotlib") is None:
            print("[WARN] matplotlib not installed; skipping RL dataset validation graphs.")
            return

        import matplotlib.pyplot as plt

        prices_actual = [float(r["actual_paid"]) for r in rows_out if r.get("actual_paid") is not None]
        prices_pred = [float(r["rl_predicted_price"]) for r in rows_out if r.get("rl_predicted_price") is not None]

        if prices_actual and prices_pred:
            plt.figure(figsize=(7, 5))
            plt.scatter(prices_actual, prices_pred, s=10, alpha=0.35)
            lo = float(min(prices_actual + prices_pred))
            hi = float(max(prices_actual + prices_pred))
            plt.plot([lo, hi], [lo, hi], "r--", linewidth=1.2, label="ideal match")
            plt.title("RL Predicted Price vs Actual Customer Price")
            plt.xlabel("Actual customer price")
            plt.ylabel("RL predicted price")
            plt.legend(loc="best")
            plt.tight_layout()
            out = f"{out_plot_prefix}_price_match_scatter.png"
            _ensure_parent_dir(out)
            plt.savefig(out, dpi=150)
            plt.close()

        duration_pairs = [
            (float(r["actual_duration_minutes"]), float(r["predicted_duration_minutes"]))
            for r in rows_out
            if r.get("actual_duration_minutes") is not None and r.get("predicted_duration_minutes") is not None
        ]
        if duration_pairs:
            actual_dur = [a for a, _ in duration_pairs]
            pred_dur = [p for _, p in duration_pairs]
            plt.figure(figsize=(7, 5))
            plt.scatter(actual_dur, pred_dur, s=10, alpha=0.35, color="tab:orange")
            lo = float(min(actual_dur + pred_dur))
            hi = float(max(actual_dur + pred_dur))
            plt.plot([lo, hi], [lo, hi], "k--", linewidth=1.2, label="ideal match")
            plt.title("Predicted Total Time vs Actual Duration")
            plt.xlabel("Actual trip duration (minutes)")
            plt.ylabel("Predicted total time (minutes)")
            plt.legend(loc="best")
            plt.tight_layout()
            out = f"{out_plot_prefix}_duration_match_scatter.png"
            _ensure_parent_dir(out)
            plt.savefig(out, dpi=150)
            plt.close()
            
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
        elif self.firm1_mode == "heuristic":
            self.firm1.act(city_base=base.base_fare, city_pmin=base.per_minute, hour=hour, weather=day_ctx.weather)

        if self.firm2_mode == "heuristic":
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
            )
        
        self.last_share = float(m1.share)
        self.last_revpr = float(m1.rev_per_request)
        self.last_gap = float(gap)
        
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
        )

    def simulate_batch(
        self,
        day_of_week: int,
        weather: str,
        hour: int,
        customers_per_step: int,
        sampled_profiles: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[Dict[str, Any]], FirmMetrics, FirmMetrics, float, float, float]:
        rows: List[Dict[str, Any]] = []
        firm1 = FirmMetrics()
        firm2 = FirmMetrics()
        gaps: List[float] = []

        airport_count = 0
        dist_sum = 0.0
        
        profiles = sampled_profiles if sampled_profiles is not None else self.agent_gen.sample_profiles(customers_per_step)

        for profile in profiles:
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

            gaps.append(p2 - p1)

            scenario = {
                "City": self.market_name,
                "DistanceMiles": float(travel_distance),
                "DurationMinutes": float(round(duration, 2)),
                "DayOfWeek": int(day_of_week),
                "Hour": int(hour),
                "Weather": str(weather),
                "Airport": bool(airport),
                "Service": str(service),
            }

            choice_res: ChoiceResult = self.choice_model.choose(profile, scenario, p1, p2)
            choice = choice_res.choice

            firm1.total += 1
            firm2.total += 1
            if choice == "Firm1":
                firm1.wins += 1
                firm1.revenue += float(p1)
            else:
                firm2.wins += 1
                firm2.revenue += float(p2)

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
                "ReasonCodes": ",".join(choice_res.reason_codes),
                "ShortReason": choice_res.short_reason,
                **profile,
            })

        mean_gap = float(np.mean(gaps)) if gaps else 0.0
        airport_rate = float(airport_count / max(1, customers_per_step))
        mean_dist = float(dist_sum / max(1, customers_per_step))
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
        
        for d in range(days):
            day_ctx = self.market.sample_day_context()
            hours = [self.market.sample_timestep_hour().hour for _ in range(timesteps_per_day)]

            # day accumulators for logging
            share_sum = 0.0
            revpr_sum = 0.0
            gap_sum = 0.0
            reward_sum = 0.0
            
            share_sum_two = 0.0
            revpr_sum_two = 0.0

            for t in range(timesteps_per_day):
                hour = hours[t]
                base = self.market.curr_market

                # Firm 1 action
                rl_step = None
                if self.firm1_mode == "RL":
                    s_vec = self._build_rl_state(day_of_week=day_ctx.day_of_week, hour=hour, weather=day_ctx.weather)
                    action, s_ts, logits, val = self.firm1.agent.act(s_vec)
                    self.firm1.apply_action(action, self.market)
                    rl_step = (action, s_ts, logits, val)
                elif self.firm1_mode == "heuristic":
                    self.firm1.act(city_base=base.base_fare, city_pmin=base.per_minute, hour=hour, weather=day_ctx.weather)

                # Firm 2 action
                if self.firm2_mode == "heuristic":
                    self.firm2.act(city_base=base.base_fare, city_pmin=base.per_minute, hour=hour, weather=day_ctx.weather)
                
                sampled_profiles = self.agent_gen.sample_profiles(customers_per_step)
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
                if self.firm1_mode == "heuristic":
                    self.firm1.update(metrics=m1, price_gap_mean=-mean_gap)  # note sign: Firm1 - Firm2
                if self.firm2_mode == "heuristic":
                    self.firm2.update(metrics=m2, price_gap_mean=mean_gap)   # Firm2 - Firm1
                    
                # RL memory + reward shaping
                if self.firm1_mode == "RL" and rl_step is not None:
                    action, s_ts, logits, val = rl_step
                    reward = self._compute_rl_reward(m1, mean_gap)
                    done = (t == timesteps_per_day - 1)
                    self.firm1.agent.store(s_ts, action, float(reward), done, None, logits, val)
                    if done:
                        self.firm1.stabilize_after_batch(
                            share=float(m1.share),
                            price_gap_f2_minus_f1=float(mean_gap),
                            city_base=float(base.base_fare),
                            city_pmin=float(base.per_minute),
                        )
                    reward_sum += float(reward)

                share_sum += float(m1.share)
                revpr_sum += float(m1.rev_per_request)
                gap_sum += float(mean_gap)
                
                share_sum_two += float(m2.share)
                revpr_sum_two += float(m2.rev_per_request)
                
                self.last_share = float(m1.share)
                self.last_revpr = float(m1.rev_per_request)
                self.last_gap = float(mean_gap)
                self.airport_rate_last = airport_rate
                self.mean_distance_last = mean_dist
            
            ppo_metrics = {"loss": 0.0, "approx_kl": 0.0, "clipfrac": 0.0, "ent_coeff": 0.0}
            if self.firm1_mode == "RL":
                ppo_metrics = self.firm1.agent.update(epochs=self.ppo_update_epochs, batch_size=self.ppo_batch_size)
                
            #print("firm 1 revenue per request sum", str(revpr_sum))
            #print("firm 1 market share sum", str(share_sum))
            
            #print("firm 2 revenue per request sum", str(revpr_sum_two))
            #print("firm 2 market share sum", str(share_sum_two))

            avg_share = share_sum / max(1, timesteps_per_day)
            avg_revpr = revpr_sum / max(1, timesteps_per_day)
            avg_gap = gap_sum / max(1, timesteps_per_day)
            avg_reward = (reward_sum / max(1, timesteps_per_day)) if self.firm1_mode == "RL" else self._reward_base(avg_share, avg_revpr, price_gap_f2_minus_f1=avg_gap)
            self.run_logs.append({
                "day": d + 1,
                "avg_share": float(avg_share),
                "avg_revpr": float(avg_revpr),
                "avg_gap": float(avg_gap),
                "avg_reward": float(avg_reward),
                "ppo_approx_kl": float(ppo_metrics.get("approx_kl", 0.0)),
                "ppo_clipfrac": float(ppo_metrics.get("clipfrac", 0.0)),
                "ppo_entropy": float(ppo_metrics.get("entropy", 0.0)),
                "ppo_ent_coeff": float(ppo_metrics.get("ent_coeff", 0.0)),
            })
            
            recent_rewards = [float(x["avg_reward"]) for x in self.run_logs[-self.reward_convergence_window:]]
            reward_std = float(np.std(recent_rewards)) if recent_rewards else 1.0
            reward_delta = (
                float(abs(recent_rewards[-1] - recent_rewards[0]) / max(1, len(recent_rewards) - 1))
                if len(recent_rewards) >= 2
                else 1.0
            )
            reward_converged = (
                len(recent_rewards) >= self.reward_convergence_window
                and reward_std <= self.reward_convergence_tol
                and reward_delta <= self.reward_trend_tol
            )
            self.run_logs[-1]["reward_window_std"] = float(reward_std)
            self.run_logs[-1]["reward_window_delta"] = float(reward_delta)
            self.run_logs[-1]["reward_converged"] = bool(reward_converged)
            if reward_converged and self.convergence_day is None:
                self.convergence_day = int(d + 1)
                print(
                    f"[Convergence] Optimization converged at day {self.convergence_day} "
                    f"(window_std={reward_std:.4f}, delta/day={reward_delta:.4f})."
                )
            
            # print every ~10% of days
            k = max(1, days // 10)
            if (d + 1) % k == 0 or (d + 1) == 1 or (d + 1) == days:

                print(
                    f"[Day {d+1}/{days}] avg_share(F1)={avg_share:.3f} avg_revPR(F1)=${avg_revpr:.2f} "
                    f"avg_gap(F2-F1)=${avg_gap:.2f}"
                )
                if self.firm1_mode == "RL":
                    print(
                        f"  [PPO] KL={float(ppo_metrics.get('approx_kl', 0.0)):.4f} "
                        f"clipfrac={float(ppo_metrics.get('clipfrac', 0.0)):.3f} "
                        f"ent={float(ppo_metrics.get('entropy', 0.0)):.3f} "
                        f"ent_coeff={float(ppo_metrics.get('ent_coeff', 0.0)):.4f}"
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
            print(
                "[Convergence Summary] "
                f"converged_day={self.convergence_day if self.convergence_day is not None else 'not reached'} "
                f"window_std={float(final.get('reward_window_std', 0.0)):.4f} "
                f"delta/day={float(final.get('reward_window_delta', 0.0)):.4f}"
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
    cols = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
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
) -> None:
    if importlib.util.find_spec("matplotlib") is None:
        print("[WARN] matplotlib not installed; skipping graph generation.")
        return

    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    _ensure_parent_dir(prefix + "_dummy")

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
        plt.figure(figsize=(8, 4))
        plt.bar(xs, ys)
        plt.title(f"Ride Distribution by {p}")
        plt.xlabel(p)
        plt.ylabel("% of rides")
        plt.tight_layout()
        out = f"{prefix}_dist_{p}.png"
        _ensure_parent_dir(out)
        plt.savefig(out, dpi=150)
        plt.close()

    # 2) Distance histogram
    if rows:
        dvals = [float(r.get("TravelDistance", 0.0)) for r in rows]
        weights = np.ones(len(dvals), dtype=float) * (100.0 / max(1, len(dvals)))
        plt.figure(figsize=(8, 4))
        plt.hist(dvals, bins=20, weights=weights)
        plt.title("Ride Distance Distribution")
        plt.xlabel("TravelDistance")
        plt.ylabel("% of rides")
        plt.tight_layout()
        out = f"{prefix}_dist_TravelDistance.png"
        _ensure_parent_dir(out)
        plt.savefig(out, dpi=150)
        plt.close()

    # 3) run reward trajectory + convergence diagnostics
    if run_logs:
        xs = [int(r["day"]) for r in run_logs]
        ys = [float(r["avg_reward"]) for r in run_logs]
        converged_days = [int(r["day"]) for r in run_logs if bool(r.get("reward_converged", False))]
        plt.figure(figsize=(9, 4))
        plt.plot(xs, ys, label="avg_reward")
        if converged_days:
            conv_day = int(converged_days[0])
            conv_reward = float(next((r["avg_reward"] for r in run_logs if int(r["day"]) == conv_day), ys[-1]))
            plt.axvline(conv_day, color="tab:green", linestyle="--", linewidth=1.2, label=f"converged day={conv_day}")
            plt.scatter([conv_day], [conv_reward], color="tab:green", zorder=3)
        plt.title("Run Reward Trajectory")
        plt.xlabel("Day")
        plt.ylabel("Reward")
        plt.legend(loc="best")
        plt.tight_layout()
        out = f"{prefix}_reward_run.png"
        _ensure_parent_dir(out)
        plt.savefig(out, dpi=150)
        plt.close()
        
        stds = [float(r.get("reward_window_std", np.nan)) for r in run_logs]
        deltas = [float(r.get("reward_window_delta", np.nan)) for r in run_logs]
        if any(np.isfinite(v) for v in stds) or any(np.isfinite(v) for v in deltas):
            fig, ax1 = plt.subplots(figsize=(9, 4))
            ax1.plot(xs, stds, color="tab:blue", label="window std")
            ax1.set_xlabel("Day")
            ax1.set_ylabel("Reward window std", color="tab:blue")
            ax1.tick_params(axis="y", labelcolor="tab:blue")

            ax2 = ax1.twinx()
            ax2.plot(xs, deltas, color="tab:orange", label="|delta| per day")
            ax2.set_ylabel("Reward trend magnitude", color="tab:orange")
            ax2.tick_params(axis="y", labelcolor="tab:orange")

            if converged_days:
                ax1.axvline(converged_days[0], color="tab:green", linestyle="--", linewidth=1.2)

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
            plt.title("Optimization Convergence Diagnostics")
            plt.tight_layout()
            out = f"{prefix}_convergence_run.png"
            _ensure_parent_dir(out)
            plt.savefig(out, dpi=150)
            plt.close(fig)

    # 4) training trajectory (run_experiment)
    if training_logs:
        xs = [int(r["batch"]) + 1 for r in training_logs]
        ys = [float(r["avg_reward"]) for r in training_logs]
        plt.figure(figsize=(9, 4))
        plt.plot(xs, ys)
        
        y_min, y_max = float(min(ys)), float(max(ys))
        y_span = max(1e-6, y_max - y_min)
        y_pad = max(0.02, 0.10 * y_span)
        y_lo, y_hi = y_min - y_pad, y_max + y_pad

        # Use denser major ticks (with minor ticks in-between) so reward movements are easier to inspect.
        tick_candidates = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
        y_total = max(1e-6, y_hi - y_lo)
        major_step = next((step for step in tick_candidates if (y_total / step) <= 12), tick_candidates[-1])

        ax = plt.gca()
        ax.set_ylim(y_lo, y_hi)
        ax.yaxis.set_major_locator(MultipleLocator(major_step))
        ax.yaxis.set_minor_locator(MultipleLocator(max(major_step / 2.0, 0.005)))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.grid(axis="y", which="major", linestyle="--", alpha=0.35)
        ax.grid(axis="y", which="minor", linestyle=":", alpha=0.20)
        
        plt.title("Training Reward Trajectory")
        plt.xlabel("Batch")
        plt.ylabel("Avg Reward")
        plt.tight_layout()
        out = f"{prefix}_reward_training.png"
        _ensure_parent_dir(out)
        plt.savefig(out, dpi=150)
        plt.close()

    # 5) evaluation trajectory (run_experiment)
    if evaluation_logs:
        xs = [int(r["day"]) for r in evaluation_logs]
        ys = [
            float(np.clip((0.60 * np.clip(float(r["rl_share"]), 0.0, 1.0)) + (0.20 * np.tanh((float(r["rl_revenue"]) - 10.0) / 8.0)), -1.0, 1.0))
            for r in evaluation_logs
        ]
        plt.figure(figsize=(9, 4))
        plt.plot(xs, ys)
        plt.title("Evaluation Reward Trajectory")
        plt.xlabel("Day")
        plt.ylabel("Reward")
        plt.tight_layout()
        out = f"{prefix}_reward_evaluation.png"
        _ensure_parent_dir(out)
        plt.savefig(out, dpi=150)
        plt.close()


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

    parser.add_argument("--firm1_mode", type=str, default="heuristic", choices=["RL", "heuristic", "static"])
    parser.add_argument("--firm2_mode", type=str, default="static", choices=["heuristic", "static"])

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
    parser.add_argument("--eval_timesteps", type=int, default=200)
    parser.add_argument("--eval_customers", type=int, default=1000)
    parser.add_argument("--profiles_out", type=str, default="artifacts/sampled_profiles.csv")
    parser.add_argument("--profiles_log_limit", type=int, default=200000)
    parser.add_argument("--reward_share_weight", type=float, default=0.60)
    parser.add_argument("--reward_revenue_weight", type=float, default=0.40)
    parser.add_argument("--reward_overprice_weight", type=float, default=0.35)
    parser.add_argument("--reward_rev_scale", type=float, default=25.0)
    parser.add_argument("--reward_competitive_weight", type=float, default=0.12)
    parser.add_argument("--reward_trend_weight", type=float, default=0.08)
    parser.add_argument("--calibration_csv", type=str, default="", help="Optional historical CSV used to calibrate priors and choice sensitivity.")
    parser.add_argument("--calibration_city", type=str, default="", help="Optional city filter for --calibration_csv; defaults to --market if omitted.")
    parser.add_argument("--calibration_preset", type=str, default="nyc_public", choices=["", "nyc_public"], help="Built-in preset calibration. Defaults to nyc_public (NYC TLC + ACS + weather priors).")
    parser.add_argument("--compare_with_dataset", action="store_true", help="After training/run, compare RL-implied prices against actual paid prices in the Kaggle CSV files.")
    parser.add_argument("--dataset_glob", type=str, default="*.csv", help="Glob for dataset CSV discovery under kagglehub download path.")
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