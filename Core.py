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
import copy
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
import sys
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional


# Suppress Intel oneMKL CPU deprecation warning on legacy (non-AVX) machines unless
# the user has already chosen an instruction policy in the environment.
import mkl_config  # noqa: F401 - set oneMKL env before NumPy/Torch

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
    FirmAdaptiveBestResponsePricer,
    FirmAggressiveAdaptiveBestResponsePricer,
    FirmMarginGuardrailPricer,
    FirmRandomWalkPricer,
    FirmPIPriceGapPricer,
    FirmRegionSupplyDemandPricer,
    FirmQueueServiceThresholdPricer,
    FirmSurgeDriverIncentivePricer,
    FirmMPCGridPricer,
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
from driver_supply import DriverSupplyConfig, DriverSupplyLayer, FirmDriverBatchState
from platform_mdp import (
    ActionStabilityTracker,
    ConstraintConfig,
    ObservationConfig,
    PlatformObservationModel,
    LongTermProfitReward,
    LongTermProfitRewardConfig,
    PositiveBusinessReward,
    PositiveRewardConfig,
    SoftConstraintController,
    OperationalClock,
    TrainingStageScheduler,
    config_payload as platform_mdp_config_payload,
)

try:
    import kagglehub
except Exception:
    kagglehub = None

try:
    import pyarrow.parquet as pq
except Exception:
    pq = None

def _resolve_dataset_path() -> str:
    """Download the Kaggle dataset only when explicitly needed."""
    if kagglehub is None:
        raise RuntimeError("kagglehub is required only for --compare_with_dataset; install it to use that option")
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
        simulation_sample_cap: int = 2000,
        deterministic_torch: bool = False,
        positive_reward_profit_weight: float = 0.38,
        positive_reward_revenue_weight: float = 0.22,
        positive_reward_completed_demand_weight: float = 0.20,
        positive_reward_service_weight: float = 0.20,
        long_term_profit_scale: float = 4.0,
        long_term_profit_advantage_scale: float = 2.5,
        long_term_profit_weight: float = 1.0,
        profit_dominance_weight: float = 0.10,
        dominance_quality_scale: float = 4.0,
        intervention_cost_weight: float = 0.0,
        reversal_cost_weight: float = 0.0,
        ppo_gamma: float = 0.995,
        ppo_gae_lambda: float = 0.97,
        reward_share_weight: float = 0.40,
        reward_revenue_weight: float = 0.35,
        reward_profit_weight: Optional[float] = None,
        reward_price_gap_weight: Optional[float] = None,
        reward_hold_inaction_weight: float = 0.06,
        reward_corrective_action_weight: float = 0.08,
        reward_baseline_loss_weight: float = 0.12,
        reward_overprice_weight: float = 0.20,
        reward_rev_scale: float = 25.0,
        reward_competitive_weight: float = 0.15,
        reward_trend_weight: float = 0.0,
        reward_profit_scale: float = 12.0,
        reward_underprice_weight: float = 0.15,
        reward_acceptable_discount: float = 2.00,
        min_profit_margin: float = 0.08,
        reward_action_change_weight: float = 0.0,
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
        driver_expected_acceptance_cutoff: float = 0.65,
        driver_state_smoothing: float = 0.35,
        driver_reward_fulfillment_weight: float = 0.15,
        driver_reward_wait_weight: float = 0.0,
        driver_reward_reject_weight: float = 0.0,
        driver_reward_unfulfilled_weight: float = 0.05,
        driver_reward_warmup_fraction: float = 0.60,
        constrained_reward: bool = False,
        constraint_lr: float = 0.03,
        constraint_penalty_scale: float = 0.10,
        constraint_curriculum_start_scale: float = 0.25,
        constraint_curriculum_mid_scale: float = 0.60,
        constraint_curriculum_end_scale: float = 1.00,
        gap_band_fraction: float = 0.75,
        gap_penalty_scale_fraction: float = 0.75,
        target_price_gap: Optional[float] = None,
        price_gap_tolerance: float = 0.45,
        segment_gap_tolerances: Tuple[float, float, float, float] = (0.65, 0.60, 0.55, 0.55),
        telemetry_delay_steps: int = 1,
        quote_probe_interval_steps: int = 3,
        quote_probe_delay_steps: int = 1,
        quote_noise_dollars: float = 0.18,
        quote_missing_probability: float = 0.03,
        training_curriculum: str = "staged",
        ppo_batch_size: int = 256,
        ppo_update_epochs: int = 8,
        state_frame_stack: int = 4,
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
        self.simulation_sample_cap = int(max(0, simulation_sample_cap))
        
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

        dynamic_pricer_modes = {
            "heuristic",
            "heuristic_margin",
            "heuristic_random",
            "adaptive_best_response",
            "adaptive_best_response_aggressive",
            "pi_price_gap",
            "region_supply_demand",
            "queue_service_threshold",
            "surge_driver_incentive",
            "mpc_grid",
        }
        if self.firm1_mode not in {"RL", "static", *dynamic_pricer_modes}:
            raise ValueError(
                "firm1_mode must be one of: RL, heuristic, heuristic_margin, heuristic_random, "
                "adaptive_best_response, adaptive_best_response_aggressive, pi_price_gap, "
                "region_supply_demand, queue_service_threshold, surge_driver_incentive, "
                "mpc_grid, static"
            )
        if self.firm2_mode not in {"static", *dynamic_pricer_modes}:
            raise ValueError(
                "firm2_mode must be one of: heuristic, heuristic_margin, heuristic_random, "
                "adaptive_best_response, adaptive_best_response_aggressive, pi_price_gap, "
                "region_supply_demand, queue_service_threshold, surge_driver_incentive, "
                "mpc_grid, static"
            )
        self.opponent_is_dynamic = self.firm2_mode != "static"

        self.opt_keys = ["base_fare", "per_minute", "per_mile", "booking_fee", "airport_fee"]
        self.shared_edit_keys = list(self.opt_keys)

        # The market simulator retains full ground truth.  Everything below is
        # the platform-facing POMDP boundary: delayed own telemetry, sampled
        # public quote probes, positive business utility, and separate soft
        # constraint channels for safe/credible operation.
        mdp_target_gap = float(
            0.5 * max(0.0, reward_acceptable_discount)
            if target_price_gap is None
            else target_price_gap
        )
        self.observation_config = ObservationConfig(
            telemetry_delay_steps=int(max(0, telemetry_delay_steps)),
            quote_probe_interval_steps=int(max(1, quote_probe_interval_steps)),
            quote_probe_delay_steps=int(max(0, quote_probe_delay_steps)),
            quote_noise_dollars=float(max(0.0, quote_noise_dollars)),
            quote_missing_probability=float(np.clip(quote_missing_probability, 0.0, 0.95)),
            revenue_scale=float(max(1e-6, reward_rev_scale)),
            profit_scale=float(max(1e-6, reward_profit_scale)),
        )
        self.positive_reward_config = PositiveRewardConfig(
            profit_weight=float(positive_reward_profit_weight),
            revenue_weight=float(positive_reward_revenue_weight),
            completed_demand_weight=float(positive_reward_completed_demand_weight),
            service_weight=float(positive_reward_service_weight),
            revenue_scale=float(max(1e-6, reward_rev_scale)),
            profit_scale=float(max(1e-6, reward_profit_scale)),
        )
        self.long_term_profit_reward_config = LongTermProfitRewardConfig(
            own_profit_scale=float(long_term_profit_scale),
            profit_advantage_scale=float(long_term_profit_advantage_scale),
            own_profit_weight=float(long_term_profit_weight),
            profit_advantage_weight=float(profit_dominance_weight),
            dominance_quality_scale=float(dominance_quality_scale),
            intervention_cost_weight=float(intervention_cost_weight),
            reversal_cost_weight=float(reversal_cost_weight),
        )
        self.constraint_config = ConstraintConfig(
            target_gap=mdp_target_gap,
            overall_tolerance=float(max(0.05, price_gap_tolerance)),
            segment_tolerances=tuple(float(max(0.05, x)) for x in segment_gap_tolerances),
            margin_floor=float(np.clip(min_profit_margin, 0.0, 0.95)),
            multiplier_lr=float(max(0.0, constraint_lr)),
        )
        self.platform_observers = {
            "Firm1": PlatformObservationModel(self.seed + 101, self.observation_config),
            "Firm2": PlatformObservationModel(self.seed + 202, self.observation_config),
        }
        self.positive_reward_model = PositiveBusinessReward(self.positive_reward_config)
        self.long_term_profit_reward_model = LongTermProfitReward(
            self.long_term_profit_reward_config
        )
        self.soft_constraints = SoftConstraintController(self.constraint_config)
        self.action_stability = ActionStabilityTracker(self.constraint_config)
        self.training_stage_scheduler = TrainingStageScheduler(training_curriculum)
        self.current_training_stage = self.training_stage_scheduler.stage_at(0.0)
        
        pricer_by_mode = {
            "heuristic": FirmHeuristicPricer,
            "heuristic_margin": FirmMarginGuardrailPricer,
            "heuristic_random": FirmRandomWalkPricer,
            "adaptive_best_response": FirmAdaptiveBestResponsePricer,
            "adaptive_best_response_aggressive": FirmAggressiveAdaptiveBestResponsePricer,
            "pi_price_gap": FirmPIPriceGapPricer,
            "region_supply_demand": FirmRegionSupplyDemandPricer,
            "queue_service_threshold": FirmQueueServiceThresholdPricer,
            "surge_driver_incentive": FirmSurgeDriverIncentivePricer,
            "mpc_grid": FirmMPCGridPricer,
        }
        self.pricer_by_mode = dict(pricer_by_mode)

        if self.firm1_mode == "RL":
            effective_frame_stack = int(max(1, state_frame_stack))
            self.firm1 = FirmRLPricer(
                seed=self.seed,
                opt_keys=self.shared_edit_keys,
                state_frame_stack=effective_frame_stack,
                single_state_dim=PlatformObservationModel.observation_dim,
                action_feature_dim=PlatformObservationModel.action_feature_dim,
                constraint_dim=len(self.soft_constraints.names),
                response_dim=12,
                feature_groups=PlatformObservationModel.feature_groups(),
                gamma=float(ppo_gamma),
                gae_lambda=float(ppo_gae_lambda),
            )
        elif self.firm1_mode in pricer_by_mode:
            self.firm1 = pricer_by_mode[self.firm1_mode](seed=self.seed, managed_keys=self.shared_edit_keys)
        else:
            self.firm1 = FirmStaticPricer()

        if self.firm2_mode in pricer_by_mode:
            self.firm2 = pricer_by_mode[self.firm2_mode](seed=self.seed + 1, managed_keys=self.shared_edit_keys)
        else:
            self.firm2 = FirmStaticPricer()
        self._assert_competitor_parity()
        
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
        self.last_firm2_share = 0.5
        self.last_firm2_revpr = 0.0
        self.last_firm2_profitpr = 0.0
        self.last_firm2_fulfillment = 1.0
        self.last_firm2_acceptance = 1.0
        self.last_firm2_wait = 0.0
        self.last_firm2_driver_paypr = 0.0
        self.ema_share = 0.5
        self.ema_revpr = 0.0
        self.ema_profitpr = 0.0
        self.ema_gap = 0.0
        self.ema_fulfillment = 1.0
        self.ema_firm2_share = 0.5
        self.ema_firm2_revpr = 0.0
        self.ema_firm2_profitpr = 0.0
        self.ema_firm2_gap = 0.0
        self.ema_firm2_fulfillment = 1.0
        self.prev_action_target = "hold"
        
        self.training_logs = []
        self.evaluation_logs = []
        
        self.run_logs = []
        self.convergence_day: Optional[int] = None
        self.convergence_window_std_at_day: Optional[float] = None
        self.convergence_delta_per_day_at_day: Optional[float] = None
        
        self.reward_share_weight = float(max(0.0, reward_share_weight))
        configured_revenue_weight = float(max(0.0, reward_revenue_weight))
        if reward_profit_weight is None:
            self.reward_profit_weight = 0.40 * configured_revenue_weight
            self.reward_revenue_weight = 0.60 * configured_revenue_weight
        else:
            self.reward_revenue_weight = configured_revenue_weight
            self.reward_profit_weight = float(max(0.0, reward_profit_weight))
        self.reward_price_gap_weight = float(
            max(0.0, reward_price_gap_weight)
            if reward_price_gap_weight is not None
            else max(float(max(0.0, reward_overprice_weight)), float(max(0.0, reward_underprice_weight)))
        )
        self.reward_hold_inaction_weight = float(max(0.0, reward_hold_inaction_weight))
        self.reward_corrective_action_weight = float(max(0.0, reward_corrective_action_weight))
        self.reward_baseline_loss_weight = float(max(0.0, reward_baseline_loss_weight))
        self.reward_overprice_weight = float(max(0.0, reward_overprice_weight))
        self.reward_underprice_weight = float(max(0.0, reward_underprice_weight))
        self.reward_acceptable_discount = float(max(0.0, reward_acceptable_discount))
        self.min_profit_margin = float(np.clip(min_profit_margin, 0.0, 0.95))
        self.reward_target_price_gap = float(self.constraint_config.target_gap)
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
                expected_acceptance_cutoff=float(np.clip(driver_expected_acceptance_cutoff, 0.0, 1.0)),
                state_smoothing_alpha=float(np.clip(driver_state_smoothing, 0.0, 1.0)),
            ),
            use_osmnx=bool(use_osmnx),
            osmnx_place=osmnx_place or f"{market_name}, USA",
        )
        self.reward_rev_scale = float(max(1e-6, reward_rev_scale))
        self.reward_profit_scale = float(max(1e-6, reward_profit_scale))
        self.reward_competitive_weight = float(max(0.0, reward_competitive_weight))
        self.reward_trend_weight = float(max(0.0, reward_trend_weight))

        # Keep the core reward aligned with the screenshot design: market-share
        # balance and revenue improvement are the main terms, acceptance/service is
        # explicit, and competitive dominance is smaller momentum support.  Do not
        # renormalize these weights; their defaults are intended as interpretable
        # reward coefficients and the final reward is clipped for PPO stability.
        denom = self.reward_revenue_weight + self.reward_profit_weight + self.reward_share_weight + self.reward_competitive_weight
        if denom <= 0.0:
            self.reward_share_weight = 0.40
            self.reward_revenue_weight = 0.21
            self.reward_profit_weight = 0.14
            self.reward_competitive_weight = 0.15

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
        self.reward_action_change_weight = float(max(0.0, reward_action_change_weight))
        self.driver_reward_fulfillment_weight = float(max(0.0, driver_reward_fulfillment_weight))
        self.driver_reward_wait_weight = float(max(0.0, driver_reward_wait_weight))
        self.driver_reward_reject_weight = float(max(0.0, driver_reward_reject_weight))
        self.driver_reward_unfulfilled_weight = float(max(0.0, driver_reward_unfulfilled_weight))
        self.driver_reward_warmup_fraction = float(np.clip(driver_reward_warmup_fraction, 0.0, 1.0))
        self.driver_reward_scale_current = 1.0 if self.driver_reward_warmup_fraction <= 0.0 else 0.0
        self.constrained_reward = bool(constrained_reward)
        self.constraint_lr = float(max(0.0, constraint_lr))
        self.constraint_penalty_scale = float(max(0.0, constraint_penalty_scale))
        self.constraint_curriculum_start_scale = float(max(0.0, constraint_curriculum_start_scale))
        self.constraint_curriculum_mid_scale = float(max(0.0, constraint_curriculum_mid_scale))
        self.constraint_curriculum_end_scale = float(max(0.0, constraint_curriculum_end_scale))
        self.gap_band_fraction = float(max(1e-6, gap_band_fraction))
        self.gap_penalty_scale_fraction = float(max(1e-6, gap_penalty_scale_fraction))
        self.constraint_curriculum_scale = 1.0
        # Adaptive Lagrange multipliers for safety/service constraints.  These
        # make the reward a constrained-MDP objective instead of a fixed weighted
        # sum: pressure rises only when the observed stochastic layer violates a
        # service floor, then decays when the policy recovers.
        # Compatibility view for existing diagnostics. PPO receives only the
        # active operational-feasibility vector from ``soft_constraints``.
        self.constraint_lambdas: Dict[str, float] = {
            name: 0.0 for name in self.soft_constraints.names
        }
        self.constraint_lambda_max = float(self.constraint_config.multiplier_max)
        
        # Keep PPO optimization controls explicit so longer rollout windows can
        # use minibatches instead of silently falling back to full-batch updates.
        
        self.ppo_update_epochs = int(max(1, ppo_update_epochs))
        self.ppo_batch_size = int(max(1, ppo_batch_size))
        
        print(
            "[LongTermProfitReward] "
            f"own={self.long_term_profit_reward_config.own_profit_weight:.3f}"
            f"*asinh(profit/{self.long_term_profit_reward_config.own_profit_scale:.2f}) + "
            f"dominance={self.long_term_profit_reward_config.profit_advantage_weight:.3f}"
            f"*profit_quality*tanh((profit-rival_profit)/"
            f"{self.long_term_profit_reward_config.profit_advantage_scale:.2f}), "
            f"profit_quality=1-exp(-max(profit,0)/"
            f"{self.long_term_profit_reward_config.dominance_quality_scale:.2f}); "
            f"intervention_cost={self.long_term_profit_reward_config.intervention_cost_weight:.4f}, "
            f"reversal_cost={self.long_term_profit_reward_config.reversal_cost_weight:.4f}, "
            f"gamma={self.firm1.agent.gamma if self.firm1_mode == 'RL' else float('nan'):.4f}, "
            f"lambda={self.firm1.agent.lam if self.firm1_mode == 'RL' else float('nan'):.3f}."
        )
        print(
            "[ConstraintConfig] "
            f"constrained={int(self.constrained_reward)}, "
            f"constraint_lr={self.constraint_lr:.3f}, "
            f"active={self.soft_constraints.names}, "
            f"margin_floor={self.constraint_config.margin_floor:.3f}; "
            "fare-gap channels are diagnostics only."
        )
    
    def _sync_driver_incentive_multipliers(self) -> None:
        """Expose pricer-side incentive controls to the driver pay layer.

        Benchmark policies inspired by two-sided ride-hailing pricing literature
        can adjust a ``supply_incentive_multiplier``.  The driver supply layer
        consumes that value through each firm's pay policy before dispatches in
        the next batch are evaluated.
        """
        if not self.enable_driver_supply:
            return
        for firm_name, pricer in (("Firm1", self.firm1), ("Firm2", self.firm2)):
            policy = self.driver_supply.pay_policies.get(firm_name)
            if policy is None:
                continue
            policy.incentive_multiplier = float(np.clip(
                getattr(pricer, "supply_incentive_multiplier", 1.0),
                0.85,
                1.25,
            ))
    
    def _assert_competitor_parity(self) -> None:
        """Validate that non-static competitors share Firm1's observable action/state interface."""
        if self.firm2_mode == "static" or not hasattr(self.firm1, "action_to_steps"):
            return
        mismatches: List[str] = []
        if getattr(self.firm1, "action_keys", None) != getattr(self.firm2, "action_keys", None):
            mismatches.append(
                f"action_keys Firm1={getattr(self.firm1, 'action_keys', None)} "
                f"Firm2={getattr(self.firm2, 'action_keys', None)}"
            )
        if getattr(self.firm1, "action_to_steps", None) != getattr(self.firm2, "action_to_steps", None):
            mismatches.append("action_to_steps")
        f1_cfg = getattr(self.firm1, "config", None)
        f2_cfg = getattr(self.firm2, "config", None)
        if f1_cfg is not None and f2_cfg is not None:
            if dict(f1_cfg.step) != dict(f2_cfg.step):
                mismatches.append(f"step Firm1={dict(f1_cfg.step)} Firm2={dict(f2_cfg.step)}")
            if dict(f1_cfg.bounds) != dict(f2_cfg.bounds):
                mismatches.append(f"bounds Firm1={dict(f1_cfg.bounds)} Firm2={dict(f2_cfg.bounds)}")
        if mismatches:
            raise ValueError("Firm1/Firm2 parity violation; only decision policy may differ: " + "; ".join(mismatches))
        print(
            f"[PricingParity] Firm1 and Firm2 share actions={len(getattr(self.firm1, 'action_to_steps', {}))}, "
            f"keys={list(getattr(self.firm1, 'action_keys', []))}."
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

    def _make_pricer(self, mode: str, seed: int):
        """Construct a fresh opponent without carrying hidden controller state."""
        mode = str(mode).strip().lower()
        cls = self.pricer_by_mode.get(mode)
        if cls is None:
            return FirmStaticPricer()
        return cls(seed=int(seed), managed_keys=self.shared_edit_keys)

    def _reset_adaptive_episode(
        self,
        *,
        tariff_reset_fraction: float,
        opponent_mode: Optional[str] = None,
        episode_seed: Optional[int] = None,
    ) -> None:
        """Randomize initial conditions and clear only environment-side memory.

        The learned policy and optimizer remain intact.  This creates many short
        strategic trajectories instead of one 3,000-day path whose terminal
        tariff can be memorized as the solution.
        """
        if opponent_mode is not None:
            self.firm2 = self._make_pricer(
                opponent_mode,
                self.seed + 10_000 if episode_seed is None else int(episode_seed),
            )
        fraction = float(np.clip(tariff_reset_fraction, 0.02, 0.40))
        base = self.market.curr_market
        for firm in (self.firm1, self.firm2):
            config = getattr(firm, "config", None)
            for key in self.shared_edit_keys:
                anchor = float(getattr(base, key))
                value = anchor * float(self.rng.uniform(1.0 - fraction, 1.0 + fraction))
                if config is not None and key in getattr(config, "bounds", {}):
                    lb, ub = config.bounds[key]
                    value = float(np.clip(value, lb, ub))
                elif key in {"base_fare", "per_minute", "per_mile"}:
                    value = max(0.01, value)
                else:
                    value = max(0.0, value)
                set_coeff(base, firm.overrides, key, value)
        if self.firm1_mode == "RL":
            self.firm1.reset_state_history()
        for observer in self.platform_observers.values():
            observer.reset()
        self.action_stability.reset()
        self.last_crowd_response_stats = {}
        self.last_share = 0.5
        self.last_revpr = 0.0
        self.last_profitpr = 0.0
        self.last_gap = 0.0
        self.last_fulfillment = 1.0
        self.last_acceptance = 1.0
        self.last_wait = 0.0
        self.last_firm2_share = 0.5
        self.last_firm2_revpr = 0.0
        self.last_firm2_profitpr = 0.0
        
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
    
    def _driver_supply_context(self, firm: str, opponent: str) -> Tuple[Dict[str, float], np.ndarray]:
        """Return same-firm supply diagnostics for heuristic/RL parity."""
        if not self.enable_driver_supply:
            return {}, np.zeros(8, dtype=np.float32)
        state = self.driver_supply.smoothed_states.get(
            firm,
            self.driver_supply.last_states.get(firm, FirmDriverBatchState()),
        )
        context = {
            "active_drivers": float(state.active_drivers),
            "idle_driver_share": float(state.idle_driver_share),
            "utilization": float(state.utilization),
            "acceptance_rate": float(state.acceptance_rate),
            "fulfillment_rate": float(state.fulfillment_rate),
            "avg_wait_minutes": float(state.avg_wait_minutes),
            "avg_pickup_minutes": float(state.avg_pickup_minutes),
            "driver_earnings_per_hour": float(state.driver_earnings_per_hour),
        }
        return context, self.driver_supply.state_features_for_firm(firm, opponent=opponent)
    
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
    
    def _effective_reward_weights(self) -> Dict[str, float]:
        """Return objective weights for static-vs-dynamic opponent regimes.

        Static-opponent runs are mostly revenue-management/profit-maximization:
        competitive share and gap discipline should not dominate unit economics.
        Dynamic-opponent runs keep the configured blended objective because
        Firm2's reactions make market position a state variable.
        """
        if bool(getattr(self, "opponent_is_dynamic", False)):
            return {
                "share": 0.55 * float(self.reward_share_weight),
                "revenue": 0.45 * float(self.reward_revenue_weight),
                "profit": 1.65 * float(self.reward_profit_weight),
                "competitive": 0.60 * float(self.reward_competitive_weight),
                "price_gap": 0.70 * float(self.reward_price_gap_weight),
                "service": float(self.driver_reward_fulfillment_weight),
            }
        return {
            "share": 0.15 * float(self.reward_share_weight),
            "revenue": 0.55 * float(self.reward_revenue_weight),
            "profit": 2.85 * float(self.reward_profit_weight),
            "competitive": 0.0,
            "price_gap": 0.20 * float(self.reward_price_gap_weight),
            "service": 0.50 * float(self.driver_reward_fulfillment_weight),
        }
    
    def _constraint_violations(
        self,
        share: float,
        price_gap_f2_minus_f1: float,
        profit_margin: float,
        fulfillment_rate: float,
        avg_wait_minutes: float,
    ) -> Dict[str, float]:
        """Return normalized constrained-MDP violations from observable outputs."""
        share_f = float(np.clip(self._finite_float(share), 0.0, 1.0))
        gap = self._finite_float(price_gap_f2_minus_f1)
        margin = self._finite_float(profit_margin)
        fulfill = float(np.clip(self._finite_float(fulfillment_rate, 1.0), 0.0, 1.0))
        wait = max(0.0, self._finite_float(avg_wait_minutes))
        target_gap = float(self.reward_target_price_gap)
        # Keep the constrained-MDP gap band proportional to the configured
        # acceptable-discount range rather than to any particular cached profile
        # or one-off evaluation trajectory.
        gap_band = float(
            np.clip(
                self.gap_band_fraction * float(self.reward_acceptable_discount),
                0.25 * float(self.reward_acceptable_discount),
                float(self.reward_acceptable_discount),
            )
        )
        return {
            "share_floor": self._smooth_positive_penalty(0.18 - share_f, scale=0.18),
            "fulfillment_floor": self._smooth_positive_penalty(0.78 - fulfill, scale=0.35),
            "wait_limit": self._smooth_positive_penalty(wait - 7.0, scale=7.0),
            "gap_band": self._smooth_positive_penalty(abs(gap - target_gap) - gap_band, scale=gap_band),
            "margin_floor": self._smooth_positive_penalty(
                max(0.0, self.min_profit_margin - margin) / max(self.min_profit_margin, 1e-6),
                scale=1.0,
            ),
        }

    def _update_constraint_multipliers(self, reward_diag: Dict[str, float]) -> None:
        """Update dual variables from cost channels, never from reward terms."""
        if not self.constrained_reward or self.constraint_lr <= 0.0:
            return
        costs = {
            name: float(reward_diag.get(f"constraint_cost_{name}", 0.0))
            for name in self.soft_constraints.names
        }
        stage_scale = float(getattr(self.current_training_stage, "constraint_scale", 1.0))
        self.soft_constraints.update(costs, scale=stage_scale)
        self.constraint_lambdas = {
            name: float(self.soft_constraints.lambdas[index])
            for index, name in enumerate(self.soft_constraints.names)
        }
            
    def _constraint_vector_from_diag(self, reward_diag: Dict[str, float]) -> np.ndarray:
        """Ordered constraint-cost vector for PPO constraint critics."""
        costs = {
            name: float(reward_diag.get(f"constraint_cost_{name}", 0.0))
            for name in self.soft_constraints.names
        }
        return self.soft_constraints.vector(costs)

    def _risk_cost_from_diag(self, reward_diag: Dict[str, float]) -> float:
        """Tail-risk proxy used by the PPO risk critic and Lagrangian advantage."""
        operational = np.asarray(
            [
                float(reward_diag.get("constraint_cost_fulfillment", 0.0)),
                float(reward_diag.get("constraint_cost_wait", 0.0)),
                float(reward_diag.get("constraint_cost_margin", 0.0)),
            ],
            dtype=float,
        )
        if operational.size == 0:
            return 0.0
        # A mean-only risk target hides rare but severe service failures.  Blend
        # expected pressure with the worst currently visible tail channel.
        return float(np.clip(0.35 * np.mean(operational) + 0.65 * np.max(operational), 0.0, 1.0))

    def _response_target_from_metrics(self, m1: FirmMetrics, m2: FirmMetrics, mean_gap: float) -> np.ndarray:
        """Current observable response snapshot, used to form causal deltas."""
        telemetry = self.platform_observers["Firm1"].telemetry
        demand = self.platform_observers["Firm1"].demand_mix
        probes = self.platform_observers["Firm1"].quote_probes
        return np.asarray(
            [
                # Put public opponent-response probes first.  These are the
                # highest-value prediction targets for strategic adaptation.
                *[
                    float(np.clip(probes[s].get("gap", 0.0) / self.observation_config.gap_scale_dollars, -1.5, 1.5))
                    for s in ("0_2", "2_5", "5_10", "10_plus")
                ],
                float(np.clip(telemetry.get("chosen_share_estimate", 0.0), 0.0, 1.0)),
                float(np.clip(telemetry.get("completed_share_estimate", 0.0), 0.0, 1.0)),
                float(np.clip(telemetry.get("revenue_per_request", 0.0) / self.reward_rev_scale, 0.0, 1.5)),
                float(np.clip(telemetry.get("profit_per_request", 0.0) / self.reward_profit_scale, -1.0, 1.0)),
                float(np.clip(telemetry.get("fulfillment_rate", 1.0), 0.0, 1.0)),
                float(np.clip(telemetry.get("wait_minutes", 0.0) / 15.0, 0.0, 1.5)),
                float(np.clip(demand.get("distance_mean", 0.0) / 12.0, 0.0, 1.5)),
                float(np.clip(demand.get("long_trip_share", 0.0), 0.0, 1.0)),
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _response_delta(baseline: np.ndarray, future: np.ndarray) -> np.ndarray:
        """Bounded action-conditioned market change for response supervision."""
        before = np.asarray(baseline, dtype=np.float32).reshape(-1)
        after = np.asarray(future, dtype=np.float32).reshape(-1)
        if before.shape != after.shape:
            raise ValueError(f"response target shapes differ: {before.shape} != {after.shape}")
        return np.clip(after - before, -2.0, 2.0).astype(np.float32)

    def _sync_agent_optimization_context(self) -> None:
        """Push current Lagrange multipliers and risk weight into the PPO agent."""
        if self.firm1_mode != "RL" or not hasattr(getattr(self.firm1, "agent", None), "set_optimization_context"):
            return
        if not self.constrained_reward:
            self.firm1.agent.set_optimization_context(
                np.zeros(len(self.soft_constraints.names), dtype=np.float32),
                risk_coeff=0.0,
                constraints_active=False,
                risk_active=False,
            )
            return
        scale = float(getattr(self.current_training_stage, "constraint_scale", 1.0))
        lambdas = np.asarray(self.soft_constraints.lambdas, dtype=np.float32).copy()
        # Fulfillment, wait, and contribution margin are economic feasibility
        # constraints. Tariff oscillation is an outcome diagnostic rather than
        # a feasibility condition: charging it through the Lagrangian made
        # "hold forever" the easiest way to reduce the actor loss.
        for index, name in enumerate(self.soft_constraints.names):
            if name == "oscillation":
                lambdas[index] = 0.0
        lambdas *= scale
        self.firm1.agent.set_optimization_context(
            lambdas,
            risk_coeff=float(self.constraint_penalty_scale * scale),
            constraints_active=True,
            risk_active=bool(self.constraint_penalty_scale * scale > 0.0),
        )

    def _ppo_bootstrap_estimates(
        self,
        *,
        day_of_week: int,
        hour: int,
        weather: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Estimate continuing-task values without advancing the frame history."""
        raw_state = self._build_rl_state(
            day_of_week=day_of_week,
            hour=hour,
            weather=weather,
            perspective="Firm1",
        )
        action_features = self._action_feature_matrix_for_pricer(
            self.firm1, getattr(self, "last_crowd_response_stats", {}) or {}
        )
        stacked_state = self.firm1.stack_state(raw_state, commit=False)
        return self.firm1.agent.value_estimates(stacked_state, action_features=action_features)

    def _configure_constraint_curriculum(self, progress: float) -> None:
        """Ramp constrained-RL pressure from easy curriculum to full realism."""
        p = float(np.clip(progress, 0.0, 1.0))
        # Use a smooth, configurable curriculum instead of thresholds fitted to
        # one cached profile pool.  The defaults ramp to the user-provided base
        # penalty; callers can set a stronger end scale for stricter studies.
        if p < 0.25:
            self.constraint_curriculum_scale = self.constraint_curriculum_start_scale
        elif p < 0.60:
            mid_progress = (p - 0.25) / 0.35
            self.constraint_curriculum_scale = float(
                self.constraint_curriculum_start_scale
                + (self.constraint_curriculum_mid_scale - self.constraint_curriculum_start_scale)
                * np.clip(mid_progress, 0.0, 1.0)
            )
        else:
            late_progress = (p - 0.60) / 0.40
            self.constraint_curriculum_scale = float(
                self.constraint_curriculum_mid_scale
                + (self.constraint_curriculum_end_scale - self.constraint_curriculum_mid_scale)
                * np.clip(late_progress, 0.0, 1.0)
            )
            
    @staticmethod
    def _policy_action_diagnostics(logits: torch.Tensor, action: int) -> Dict[str, float]:
        """Return compact deterministic-policy diagnostics for eval logs."""
        with torch.no_grad():
            probs_t = torch.softmax(logits.detach().float(), dim=-1).reshape(-1)
            probs = np.asarray(probs_t.cpu().tolist(), dtype=float)
        if probs.size == 0:
            return {
                "policy_action_prob": 0.0,
                "policy_top_action": -1.0,
                "policy_top_prob": 0.0,
                "policy_entropy": 0.0,
                "policy_hold_prob": 0.0,
            }
        safe_probs = np.clip(probs.astype(float), 1e-12, 1.0)
        top_action = int(np.argmax(probs))
        action_i = int(action)
        action_prob = float(probs[action_i]) if 0 <= action_i < probs.size else 0.0
        return {
            "policy_action_prob": action_prob,
            "policy_top_action": float(top_action),
            "policy_top_prob": float(probs[top_action]),
            "policy_entropy": float(-np.sum(safe_probs * np.log(safe_probs))),
            "policy_hold_prob": float(probs[0]) if probs.size > 0 else 0.0,
        }
    
    def _coeff_snapshot(self) -> Dict[str, float]:
        """Current manipulated Firm1 coefficients, resolved against market anchors."""
        if self.firm1_mode != "RL":
            return {}
        return {
            k: float(get_coeff(self.market.curr_market, self.firm1.overrides, k))
            for k in self.shared_edit_keys
        }

    @staticmethod
    def _coeff_delta(after: Dict[str, float], before: Dict[str, float]) -> Dict[str, float]:
        keys = sorted(set(after) | set(before))
        return {
            k: float(after.get(k, 0.0) - before.get(k, 0.0))
            for k in keys
            if abs(float(after.get(k, 0.0) - before.get(k, 0.0))) > 1e-12
        }
    
    def _action_feature_matrix_for_pricer(self, pricer, crowd_context: Optional[Dict[str, float]] = None) -> Optional[np.ndarray]:
        """Build action consequences from own tariff and observable quote probes."""
        if not hasattr(pricer, "action_to_steps") or not hasattr(pricer, "action_keys"):
            return None
        base = self.market.curr_market
        config = getattr(pricer, "config", None)
        if config is None:
            return None
        firm_name = "Firm2" if pricer is self.firm2 else "Firm1"
        anchors = {key: float(getattr(base, key)) for key in self.opt_keys}
        coefficients = {
            key: float(get_coeff(base, pricer.overrides, key)) for key in self.opt_keys
        }
        return self.platform_observers[firm_name].build_action_features(
            action_steps=getattr(pricer, "action_to_steps", {}),
            action_keys=list(getattr(pricer, "action_keys", [])),
            own_coefficients=coefficients,
            anchor_coefficients=anchors,
            coefficient_steps=dict(config.step),
            coefficient_bounds=dict(config.bounds),
            step_scale=float(getattr(pricer, "step_scale", 1.0)),
            target_gap=float(self.constraint_config.target_gap),
        )

    def _publish_pricing_observation(
        self,
        *,
        day_of_week: int,
        hour: int,
        weather: str,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Expose the same market observation set to both firms before decisions."""
        firm1_state_vec = self._build_rl_state(day_of_week=day_of_week, hour=hour, weather=weather, perspective="Firm1")
        firm2_state_vec = self._build_rl_state(day_of_week=day_of_week, hour=hour, weather=weather, perspective="Firm2")
        crowd_context = getattr(self, "last_crowd_response_stats", {})
        f1_action_features = self._action_feature_matrix_for_pricer(self.firm1, crowd_context)
        f2_action_features = self._action_feature_matrix_for_pricer(self.firm2, crowd_context)
        if hasattr(self.firm1, "observe_state"):
            self.firm1.observe_state(firm1_state_vec, f1_action_features)
        if hasattr(self.firm2, "observe_state"):
            # Same schema, firm-specific values: Firm2 receives its own supply,
            # fulfillment, coefficient distances, and Firm1 as the opponent.
            self.firm2.observe_state(firm2_state_vec, f2_action_features)
        return firm1_state_vec, f1_action_features, f2_action_features

    def _ingest_platform_observations(self, m1: FirmMetrics, m2: FirmMetrics) -> None:
        """Release one simulated market outcome through realistic measurement channels."""
        crowd_stats = dict(getattr(self, "last_crowd_response_stats", {}) or {})
        for firm_name, opponent_name, metrics, gap_sign in (
            ("Firm1", "Firm2", m1, 1.0),
            ("Firm2", "Firm1", m2, -1.0),
        ):
            supply, _ = self._driver_supply_context(firm_name, opponent_name)
            own_metrics = {
                "chosen_share": float(metrics.chosen_share),
                "completed_share": float(metrics.completed_share),
                "revenue_per_request": float(metrics.rev_per_request),
                "profit_per_request": float(metrics.profit_per_request),
                "fulfillment_rate": float(metrics.fulfillment_rate),
                "acceptance_rate": float(metrics.driver_acceptance_rate),
                "wait_minutes": float(metrics.avg_wait_minutes),
                "driver_pay_per_request": float(metrics.driver_pay / max(1, metrics.total)),
            }
            self.platform_observers[firm_name].ingest(
                own_metrics=own_metrics,
                supply_metrics=supply,
                crowd_stats=crowd_stats,
                gap_sign=gap_sign,
            )

    def _observed_own_minus_competitor_gap(self, firm_name: str) -> float:
        """Return an uncertainty-weighted public-quote estimate, never true gap."""
        probes = self.platform_observers[firm_name].quote_probes.values()
        available = [probe for probe in probes if float(probe.get("available", 0.0)) > 0.0]
        if not available:
            return 0.0
        weights = np.asarray(
            [1.0 / max(0.05, float(probe.get("uncertainty", 1.0))) for probe in available],
            dtype=float,
        )
        competitor_minus_own = float(np.average(
            [float(probe.get("gap", 0.0)) for probe in available],
            weights=weights,
        ))
        return -competitor_minus_own

    def _restore_coeff_snapshot(self, snapshot: Dict[str, float]) -> None:
        """Restore Firm1 overrides to an absolute coefficient snapshot."""
        if self.firm1_mode != "RL":
            return
        for k, value in snapshot.items():
            set_coeff(self.market.curr_market, self.firm1.overrides, k, float(value))

    def _apply_eval_guardrail(
        self,
        mode: str,
        *,
        metrics: FirmMetrics,
        price_gap_f2_minus_f1: float,
        base,
    ) -> Dict[str, Any]:
        """Apply, skip, or only log the post-batch RL safety projection."""
        if self.firm1_mode != "RL" or not hasattr(self.firm1, "stabilize_after_batch"):
            return {"applied": False, "mode": "unavailable", "reasons": [], "deltas": {}}
        guardrail_mode = str(mode or "deployed").strip().lower()
        if guardrail_mode not in {"deployed", "off", "log_only"}:
            guardrail_mode = "deployed"
        before = self._coeff_snapshot()
        if guardrail_mode == "off":
            return {"applied": False, "mode": guardrail_mode, "reasons": [], "deltas": {}, "before": before, "after": dict(before)}
        diag = self.firm1.stabilize_after_batch(
            share=float(metrics.share),
            price_gap_f2_minus_f1=float(price_gap_f2_minus_f1),
            city_base=float(base.base_fare),
            city_pmin=float(base.per_minute),
            city_pmile=float(base.per_mile),
            city_booking=float(base.booking_fee),
            city_airport=float(base.airport_fee),
            profit_per_request=float(metrics.profit_per_request),
            fulfillment_rate=float(metrics.fulfillment_rate),
            target_price_gap=float(self.reward_target_price_gap),
        )
        after = self._coeff_snapshot()
        if guardrail_mode == "log_only":
            self._restore_coeff_snapshot(before)
            after = dict(before)
        diag = dict(diag or {})
        diag.update(
            {
                "mode": guardrail_mode,
                "applied": bool(diag.get("applied", False)) and guardrail_mode == "deployed",
                "recommended": bool(diag.get("applied", False)),
                "before": before,
                "after": after,
                "deltas": self._coeff_delta(after, before) if guardrail_mode == "deployed" else dict(diag.get("deltas", {})),
            }
        )
        return diag
    
    @staticmethod
    def _format_distance_segment_diagnostics(stats: Dict[str, Any]) -> str:
        """Compact distance-bin diagnostics for terminal progress logs."""
        pieces: List[str] = []
        for label, display in (
            ("0_2", "0-2"),
            ("2_5", "2-5"),
            ("5_10", "5-10"),
            ("10_plus", "10+"),
        ):
            choice = float(stats.get(f"distance_bin_{label}_firm1_choice_share", np.nan))
            completed = float(stats.get(f"distance_bin_{label}_firm1_completed_share", np.nan))
            gap = float(stats.get(f"distance_bin_{label}_price_gap_mean", np.nan))
            if np.isfinite(choice) and np.isfinite(completed) and np.isfinite(gap):
                pieces.append(f"{display}:ch={choice:.2f}/done={completed:.2f}/gap={gap:.2f}")
        return " | ".join(pieces)

    @staticmethod
    def _segment_balance_penalty_from_stats(stats: Dict[str, Any]) -> float:
        """Penalty for policies that win only one distance segment or create floods."""
        completed_shares: List[float] = []
        flood_penalties: List[float] = []
        gap_penalties: List[float] = []
        for label in ("0_2", "2_5", "5_10", "10_plus"):
            choice = float(stats.get(f"distance_bin_{label}_firm1_choice_share", np.nan))
            completed = float(stats.get(f"distance_bin_{label}_firm1_completed_share", np.nan))
            gap = float(stats.get(f"distance_bin_{label}_price_gap_mean", np.nan))
            if np.isfinite(completed):
                completed_shares.append(float(np.clip(completed, 0.0, 1.0)))
            if np.isfinite(choice) and np.isfinite(completed):
                flood_penalties.append(float(np.clip(choice - completed - 0.25, 0.0, 1.0)))
            if np.isfinite(gap):
                gap_penalties.append(float(np.clip((abs(gap) - 3.0) / 3.0, 0.0, 1.0)))
        imbalance = 0.0
        if len(completed_shares) >= 2:
            imbalance = float(np.clip(np.std(completed_shares) / 0.30, 0.0, 1.0))
        flood = float(np.mean(flood_penalties)) if flood_penalties else 0.0
        gap_extreme = float(np.mean(gap_penalties)) if gap_penalties else 0.0
        return float(np.clip(0.50 * imbalance + 0.35 * flood + 0.15 * gap_extreme, 0.0, 1.0))

    def _validation_score_from_metrics(
        self,
        *,
        reward: float,
        share: float,
        revpr: float,
        profitpr: float,
        fulfillment: float,
        gap: float,
        segment_stats: Optional[Dict[str, Any]] = None,
        rival_profitpr: float = 0.0,
        rival_share: float = 0.0,
        constraint_costs: Optional[Dict[str, float]] = None,
    ) -> float:
        """Profit and profit-dominance checkpoint score.

        The score intentionally has no share, revenue, price-gap, or action
        diversity or feasibility penalty. Those quantities remain diagnostics
        but cannot change which policy is selected.
        """
        config = self.long_term_profit_reward_config
        profit_quality = float(
            1.0
            - np.exp(
                -max(0.0, float(profitpr))
                / config.dominance_quality_scale
            )
        )
        dominance_tie_breaker = float(
            config.profit_advantage_scale
            * np.tanh(
                (float(profitpr) - float(rival_profitpr))
                / config.profit_advantage_scale
            )
        )
        return float(
            config.own_profit_weight * float(profitpr)
            + config.profit_advantage_weight
            * profit_quality
            * dominance_tie_breaker
        )

    @staticmethod
    def _smoothed_convergence_statistics(
        rewards: Sequence[float],
        *,
        window: int,
        smoothing: int = 20,
    ) -> Tuple[int, float, float]:
        """Measure convergence on rolling expected reward, not raw market noise.

        A day's realized reward contains demand, weather, profile, supply, and
        opponent-response noise.  The policy has converged only when a sequence
        of trailing expected-reward estimates is stable.  Returning the number
        of usable estimates lets callers retain their existing minimum-window
        requirements without accidentally declaring convergence on a short run.
        """
        finite = np.asarray(
            [float(value) for value in rewards if np.isfinite(value)],
            dtype=float,
        )
        smooth_width = int(max(1, smoothing))
        if finite.size < smooth_width:
            return 0, 1.0, 1.0
        kernel = np.full(smooth_width, 1.0 / smooth_width, dtype=float)
        rolling = np.convolve(finite, kernel, mode="valid")
        recent = rolling[-int(max(1, window)):]
        if recent.size < 2:
            return int(recent.size), 1.0, 1.0
        reward_std = float(np.std(recent))
        reward_delta = float(
            abs(float(recent[-1]) - float(recent[0]))
            / max(1, int(recent.size) - 1)
        )
        return int(recent.size), reward_std, reward_delta

    def _legacy_reward_components(
        self,
        share: float,
        rev_per_request: float,
        price_gap_f2_minus_f1: float = 0.0,
        profit_per_request: Optional[float] = None,
        profit_margin: Optional[float] = None,
        fulfillment_rate: float = 1.0,
        avg_wait_minutes: float = 0.0,
        driver_acceptance_rate: float = 1.0,
        action_change_magnitude: float = 0.0,
        completed_share: Optional[float] = None,
        baseline_share: float = 0.0,
        baseline_completed_share: Optional[float] = None,
        baseline_rev_per_request: float = 0.0,
        baseline_profit_per_request: float = 0.0,
    ) -> Dict[str, float]:
        """Compute the core duopoly pricing reward.

        Literature-grounded objective for crowd-interactive ride-hailing:
        (1) preserve a direct unit-economics signal (profit/revenue), as in
        dynamic-pricing and dispatch RL; (2) keep demand/service observables
        positive and dense so simulated riders/drivers shape learning without
        sparse failure penalties; (3) treat operational constraints as separate
        cost/diagnostic signals, matching constrained-RL practice instead of
        burying adaptive Lagrange costs in the scalar reward; and (4) use
        bounded reward shaping for price-gap discipline so acceptable gaps earn
        positive credit rather than imposing a penalty.
        """
        
        choice_share_f = float(np.clip(self._finite_float(share), 0.0, 1.0))
        served_share = float(np.clip(
            self._finite_float(share if completed_share is None else completed_share),
            0.0,
            1.0,
        ))
        rival_share = float(np.clip(
            self._finite_float(baseline_share if baseline_completed_share is None else baseline_completed_share),
            0.0,
            1.0,
        ))
        revpr = max(0.0, self._finite_float(rev_per_request))
        profitpr = self._finite_float(revpr if profit_per_request is None else profit_per_request)
        rival_revpr = max(0.0, self._finite_float(baseline_rev_per_request))
        rival_profitpr = self._finite_float(baseline_profit_per_request)
        margin = self._finite_float(profit_margin)
        gap = self._finite_float(price_gap_f2_minus_f1)
        fulfill = float(np.clip(self._finite_float(fulfillment_rate, 1.0), 0.0, 1.0))
        wait = max(0.0, self._finite_float(avg_wait_minutes))
        driver_accept = float(np.clip(self._finite_float(driver_acceptance_rate, 1.0), 0.0, 1.0))
        
        has_rival = rival_revpr > 0.0 or rival_share > 0.0 or abs(rival_profitpr) > 1e-9
        profit_scale = max(self.reward_profit_scale, abs(rival_profitpr), abs(profitpr), 1.0)
        revenue_scale = max(self.reward_rev_scale, rival_revpr, revpr, 1.0)
        absolute_profit_term = float(np.tanh(profitpr / profit_scale))
        absolute_revenue_term = float(np.clip(revpr / revenue_scale, 0.0, 1.0))
        share_floor_term = float(np.clip((served_share - 0.30) / 0.25, -1.0, 1.0))
        if has_rival:
            relative_profit_term = float(np.clip((profitpr - rival_profitpr) / profit_scale, -1.0, 1.0))
            relative_revenue_term = float(np.clip((revpr - rival_revpr) / revenue_scale, -1.0, 1.0))
            relative_share_term = float(np.clip((served_share - rival_share) / 0.25, -1.0, 1.0))
            # Robust duopoly learning should beat the rival without learning a
            # degenerate low-profit niche.  Blend relative advantage with the
            # absolute business level so static and heuristic opponents share
            # the same success criterion: profitable served demand.
            profit_term = float(np.clip(0.65 * relative_profit_term + 0.35 * absolute_profit_term, -1.0, 1.0))
            revenue_term = float(np.clip(0.60 * relative_revenue_term + 0.40 * absolute_revenue_term, -1.0, 1.0))
            share_term = float(np.clip(0.65 * relative_share_term + 0.35 * share_floor_term, -1.0, 1.0))
        else:
            relative_profit_term = 0.0
            relative_revenue_term = 0.0
            relative_share_term = 0.0
            profit_term = absolute_profit_term
            revenue_term = absolute_revenue_term
            share_term = share_floor_term
        target_gap = float(self.reward_target_price_gap)
        price_gap_deviation = float(gap - target_gap)
        
        gap_error = float(gap - target_gap)
        gap_scale = float(max(1e-6, self.reward_acceptable_discount))
        gap_abs_error = float(abs(gap_error))
        gap_excess = float(max(0.0, gap_abs_error - gap_scale))
        price_gap_satisfaction = float(
            1.0 if gap_excess <= 0.0 else np.exp(-0.5 * (gap_excess / gap_scale) ** 2)
        )
        price_gap_in_range = float(gap_abs_error <= gap_scale)
        demand_loss = float(np.clip(max(0.0, (rival_share if has_rival else 0.30) - served_share) / 0.30, 0.0, 1.0))
        wait_satisfaction = float(np.clip(1.0 - wait / 12.0, 0.0, 1.0))
        service_quality = float(np.clip(0.60 * fulfill + 0.25 * driver_accept + 0.15 * wait_satisfaction, 0.0, 1.0))
        constraint_violations = self._constraint_violations(
            share=served_share,
            price_gap_f2_minus_f1=gap,
            profit_margin=margin,
            fulfillment_rate=fulfill,
            avg_wait_minutes=wait,
        )
        # Constraint values remain diagnostics/critic costs only; the scalar reward
        # intentionally avoids adaptive Lagrangian penalties for a simpler,
        # positively shaped objective.
            
        weights = self._effective_reward_weights()
        profit_component = weights["profit"] * profit_term
        revenue_component = weights["revenue"] * revenue_term
        share_component = weights["share"] * share_term
        competitive_component = weights["competitive"] * share_term
        price_gap_component = weights["price_gap"] * price_gap_satisfaction
        service_component = weights["service"] * service_quality

        raw = (
            profit_component
            + revenue_component
            + share_component
            + competitive_component
            + price_gap_component
            + service_component
        )
        base = float(np.clip(raw, -1.0, 1.0))
        positive_profit = float(max(0.0, profit_term))
        positive_revenue = float(max(0.0, revenue_term))
        positive_share = float(max(0.0, share_term))
        profit_objective = float(np.clip((profit_term + 1.0) / 2.0, 0.0, 1.0))
        revenue_objective = float(np.clip((revenue_term + 1.0) / 2.0, 0.0, 1.0))
        share_objective = float(np.clip((share_term + 1.0) / 2.0, 0.0, 1.0))
        return {
            "reward_base": base,
            "reward_base_unclipped": float(raw),
            "reward_revenue_component": float(revenue_component),
            "reward_profit_component": float(profit_component),
            "reward_share_component": float(share_component),
            "reward_unit_economics_component": float(revenue_component + profit_component),
            "reward_dominance_component": float(competitive_component),
            "reward_price_gap_component": float(price_gap_component),
            "reward_service_component": float(service_component),
            "reward_objective_dynamic_opponent": float(bool(getattr(self, "opponent_is_dynamic", False))),
            **{f"reward_effective_weight_{k}": float(v) for k, v in weights.items()},
            "reward_baseline_share": float(rival_share),
            "reward_baseline_choice_share": float(np.clip(self._finite_float(baseline_share), 0.0, 1.0)),
            "reward_baseline_completed_share": float(rival_share),
            "reward_completed_share": float(served_share),
            "reward_baseline_rev_per_request": float(rival_revpr),
            "reward_baseline_profit_per_request": float(rival_profitpr),
            "reward_absolute_profit_term": float(absolute_profit_term),
            "reward_relative_profit_term": float(relative_profit_term),
            "reward_absolute_revenue_term": float(absolute_revenue_term),
            "reward_relative_revenue_term": float(relative_revenue_term),
            "reward_relative_share_term": float(relative_share_term),
            "reward_revenue_improvement": float(revenue_term),
            "reward_positive_revenue_improvement": positive_revenue,
            "reward_share_improvement": float(share_term),
            "reward_positive_share_improvement": positive_share,
            "reward_profit_improvement": float(profit_term),
            "reward_positive_profit_improvement": positive_profit,
            "reward_revenue_objective": revenue_objective,
            "reward_profit_objective": profit_objective,
            "reward_share_objective": share_objective,
            "reward_dominance_objective": share_objective,
            "reward_dominance_advantage": float(share_term),
            "reward_fulfillment_objective": float(fulfill),
            "reward_service_quality": float(service_quality),
            "reward_wait_satisfaction": float(wait_satisfaction),
            "reward_driver_acceptance_objective": float(driver_accept),
            "reward_demand_loss": float(demand_loss),
            "reward_choice_share": float(choice_share_f),
            "reward_price_gap": float(gap),
            "reward_target_price_gap": float(target_gap),
            "reward_price_gap_deviation": float(gap_error),
            "reward_price_gap_abs_error": float(gap_abs_error),
            "reward_price_gap_satisfaction": float(price_gap_satisfaction),
            "reward_price_gap_in_range": float(price_gap_in_range),
            "reward_gap_tolerance": float(gap_scale),
            "reward_fulfillment_term": float(fulfill),
            "reward_profit_term": float(profit_term),
            "reward_served_share": float(served_share),
            **{f"constraint_violation_{k}": float(v) for k, v in constraint_violations.items()},
            **{f"constraint_lambda_{k}": float(v) for k, v in self.constraint_lambdas.items()},
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
        action_change_magnitude: float = 0.0,
        competitor_share: float = 0.0,
        competitor_profit_per_request: float = 0.0,
    ) -> float:
        """Active profit utility used by lightweight reporting paths."""
        del price_gap_f2_minus_f1, profit_margin, competitor_share
        del share, fulfillment_rate, driver_acceptance_rate, avg_wait_minutes
        return float(self.long_term_profit_reward_model.compute(
            own_profit_per_request=(
                rev_per_request if profit_per_request is None else profit_per_request
            ),
            rival_profit_per_request=competitor_profit_per_request,
            intervention_magnitude=action_change_magnitude,
            reversal=0.0,
        )["reward"])

    def _legacy_reward_diagnostics(
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
        action_change_magnitude: float = 0.0,
        completed_share: Optional[float] = None,
        baseline_share: float = 0.0,
        baseline_completed_share: Optional[float] = None,
        baseline_rev_per_request: float = 0.0,
        baseline_profit_per_request: float = 0.0,
        action_event: Optional[bool] = None,
    ) -> Dict[str, float]:
        """Return shaped reward and component diagnostics for trajectory analysis."""
        share_f = float(np.clip(share, 0.0, 1.0))
        revpr = float(max(0.0, rev_per_request))
        profitpr = float(revpr if profit_per_request is None else profit_per_request)
        margin = float(0.0 if profit_margin is None else profit_margin)
        gap = self._finite_float(mean_gap)
        
        components = self._legacy_reward_components(
            share=share_f,
            rev_per_request=revpr,
            price_gap_f2_minus_f1=gap,
            profit_per_request=profitpr,
            profit_margin=margin,
            fulfillment_rate=fulfillment_rate,
            avg_wait_minutes=avg_wait_minutes,
            driver_acceptance_rate=driver_acceptance_rate,
            action_change_magnitude=action_change_magnitude,
            completed_share=completed_share,
            baseline_share=baseline_share,
            baseline_completed_share=baseline_completed_share,
            baseline_rev_per_request=baseline_rev_per_request,
            baseline_profit_per_request=baseline_profit_per_request,
        )
        base_reward = float(components["reward_base"])
        
        dominance_width = max(1e-6, self.reward_dominance_full_credit_share - self.reward_dominance_threshold)
        dominance_term = float(np.clip((share_f - self.reward_dominance_threshold) / dominance_width, 0.0, 1.0))
        share_delta = float(np.clip(share_f - float(prev_share), -0.20, 0.20) / 0.20)
        rev_delta = float(np.clip((revpr - float(prev_rev_per_request)) / self.reward_profit_scale, -0.20, 0.20) / 0.20)
        prev_profit = float(profitpr if prev_profit_per_request is None else prev_profit_per_request)
        profit_delta = float(np.clip((profitpr - prev_profit) / self.reward_profit_scale, -0.20, 0.20) / 0.20)
        prev_gap_f = float(gap if prev_gap is None else prev_gap)
        gap_delta = float(np.clip((abs(prev_gap_f - self.reward_target_price_gap) - abs(gap - self.reward_target_price_gap)) / 2.0, -1.0, 1.0))
        # The processed reward is intentionally short: core duopoly reward plus
        # small market-response momentum. Other returned fields are diagnostics only.
        trend_term = float(0.50 * profit_delta + 0.30 * share_delta + 0.20 * gap_delta)
        pricing_discipline = float(components["reward_price_gap_satisfaction"])
        efficiency_term = float(0.65 * components["reward_profit_objective"] + 0.35 * components["reward_share_objective"])
        momentum_component = float(self.reward_momentum_weight * self.reward_trend_scale * trend_term)
        
        action_desc = getattr(getattr(self, "firm1", None), "last_action_descriptor", None)
        action_direction = int(getattr(action_desc, "direction", 0) or 0)
        saturated_action = bool(getattr(getattr(self, "firm1", None), "last_action_was_saturated", False))
        zero_effect_action = bool(getattr(getattr(self, "firm1", None), "last_action_was_zero_effect", False))
        if action_event is False:
            action_direction = 0
            saturated_action = False
            zero_effect_action = False
        gap_error = float(gap - self.reward_target_price_gap)
        gap_scale = float(max(self.reward_acceptable_discount, 1e-6))
        overprice_pressure = float(np.clip(max(0.0, -gap_error) / gap_scale, 0.0, 1.0))
        underprice_pressure = float(
            np.clip(max(0.0, gap_error - self.reward_acceptable_discount) / gap_scale, 0.0, 1.0)
        )
        margin_shortfall = float(
            np.clip(
                max(0.0, self.min_profit_margin - margin) / max(self.min_profit_margin, 1e-6),
                0.0,
                1.0,
            )
        )
        baseline_loss = float(
            np.clip(
                max(
                    0.0,
                    -(
                        0.65 * float(components["reward_relative_profit_term"])
                        + 0.35 * float(components["reward_relative_revenue_term"])
                    ),
                ),
                0.0,
                1.0,
            )
        )
        demand_pressure = float(components.get("reward_demand_loss", 0.0))
        service_pressure = float(1.0 - components.get("reward_service_quality", 1.0))
        correction_pressure = float(
            np.clip(
                max(
                    overprice_pressure,
                    underprice_pressure,
                    baseline_loss,
                    demand_pressure,
                    margin_shortfall,
                    service_pressure,
                ),
                0.0,
                1.0,
            )
        )
        corrective_action_bonus = 0.0
        if (
            not zero_effect_action
            and not saturated_action
            and action_direction < 0
            and overprice_pressure > 0.0
            and gap_delta > 0.0
        ):
            corrective_action_bonus = self.reward_corrective_action_weight * min(
                overprice_pressure,
                gap_delta,
            )
        elif not zero_effect_action and not saturated_action and action_direction > 0 and (
            (underprice_pressure > 0.0 and gap_delta > 0.0)
            or (margin_shortfall > 0.0 and overprice_pressure <= 0.05 and profit_delta > 0.0)
        ):
            corrective_action_bonus = self.reward_corrective_action_weight * max(
                min(underprice_pressure, max(gap_delta, 0.0)),
                min(margin_shortfall, max(profit_delta, 0.0)),
            )
        response_component = float(corrective_action_bonus)
        hold_inaction_penalty = float(
            self.reward_hold_inaction_weight * correction_pressure
            if action_direction == 0
            else 0.0
        )
        baseline_loss_penalty = float(self.reward_baseline_loss_weight * baseline_loss)
        overprice_penalty = float(self.reward_overprice_weight * overprice_pressure)
        underprice_penalty = float(
            self.reward_underprice_weight * max(underprice_pressure, margin_shortfall)
        )
        action_change_penalty = float(
            self.reward_action_change_weight * np.clip(abs(action_change_magnitude), 0.0, 1.0)
        )
        action_realization_penalty = float(
            max(self.reward_action_change_weight, 0.25 * self.reward_hold_inaction_weight)
            * float(zero_effect_action or saturated_action)
        )
        raw_reward = float(
            base_reward
            + momentum_component
            + corrective_action_bonus
            - hold_inaction_penalty
            - baseline_loss_penalty
            - overprice_penalty
            - underprice_penalty
            - action_change_penalty
            - action_realization_penalty
        )
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
            "reward_response_component": float(response_component),
            "reward_corrective_action_bonus": float(corrective_action_bonus),
            "reward_hold_inaction_penalty": float(hold_inaction_penalty),
            "reward_baseline_loss_component": float(-baseline_loss_penalty),
            "reward_baseline_loss_penalty": float(baseline_loss_penalty),
            "reward_overprice_component": float(-overprice_penalty),
            "reward_overprice_penalty": float(overprice_penalty),
            "reward_underprice_component": float(-underprice_penalty),
            "reward_underprice_penalty": float(underprice_penalty),
            "reward_action_change_penalty": float(action_change_penalty),
            "reward_action_realization_penalty": float(action_realization_penalty),
            "reward_correction_pressure": float(correction_pressure),
            "reward_zero_effect_action": float(zero_effect_action),
            "reward_saturated_action": float(saturated_action),
            "reward_action_reversal": 0.0,
            "reward_oscillation_count": 0.0,
            "reward_action_target_airport_context": 0.0,
            "reward_crowd_near_threshold_share": 0.0,
            "reward_crowd_no_ride_rate": 0.0,
            "reward_pricing_discipline": pricing_discipline,
            "reward_efficiency_term": efficiency_term,
            "reward_profit_per_request": profitpr,
            "reward_profit_margin": margin,
            "reward_fulfillment_rate": float(np.clip(fulfillment_rate, 0.0, 1.0)),
            "reward_avg_wait_minutes": float(avg_wait_minutes),
            "reward_driver_acceptance_rate": float(np.clip(driver_acceptance_rate, 0.0, 1.0)),
            **components,
        }

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
        action_change_magnitude: float = 0.0,
        completed_share: Optional[float] = None,
        baseline_share: float = 0.0,
        baseline_completed_share: Optional[float] = None,
        baseline_rev_per_request: float = 0.0,
        baseline_profit_per_request: float = 0.0,
        action_event: Optional[bool] = None,
    ) -> Dict[str, float]:
        """Long-run contribution-profit reward plus diagnostic decomposition.

        Share, revenue, fare gaps, and one-step KPI improvements are deliberately
        excluded from the optimized scalar. They remain logged because they are
        useful explanations of *why* profit changed. Service feasibility,
        waiting, and margin are learned through separate constraint critics
        rather than being mixed into the business objective. Tariff oscillation
        is logged but not optimized, so legitimate reactions are not suppressed.
        """
        profitpr = float(rev_per_request if profit_per_request is None else profit_per_request)
        margin = float(0.0 if profit_margin is None else profit_margin)
        completed = float(share if completed_share is None else completed_share)
        positive = self.positive_reward_model.compute({
            "profit_per_request": profitpr,
            "revenue_per_request": float(rev_per_request),
            "completed_share": completed,
            "fulfillment_rate": float(fulfillment_rate),
            "acceptance_rate": float(driver_acceptance_rate),
            "wait_minutes": float(avg_wait_minutes),
        })
        descriptor = getattr(getattr(self, "firm1", None), "last_action_descriptor", None)
        target = str(getattr(descriptor, "target", "hold"))
        direction = int(getattr(descriptor, "direction", 0) or 0)
        oscillation = self.action_stability.record(
            action_event=bool(action_event),
            target=target,
            direction=direction,
            directions=getattr(getattr(self, "firm1", None), "last_action_steps", {}),
        )
        active_reward = self.long_term_profit_reward_model.compute(
            own_profit_per_request=profitpr,
            rival_profit_per_request=float(baseline_profit_per_request),
            intervention_magnitude=(
                float(action_change_magnitude) if bool(action_event) else 0.0
            ),
            reversal=float(self.action_stability.last_reversal),
        )
        costs = self.soft_constraints.compute(
            observer=self.platform_observers["Firm1"],
            fulfillment_rate=float(fulfillment_rate),
            wait_minutes=float(avg_wait_minutes),
            profit_margin=margin,
            oscillation_cost=oscillation,
        )
        discipline = float(np.clip(1.0 - max(costs.get("gap_overprice", 0.0), costs.get("gap_underprice", 0.0)), 0.0, 1.0))
        efficiency = float(
            0.50 * positive["reward_positive_profit"]
            + 0.30 * positive["reward_positive_revenue"]
            + 0.20 * positive["reward_positive_completed_demand"]
        )
        profit_delta = float(
            profitpr - float(prev_profit_per_request or 0.0)
        )
        revenue_delta = float(rev_per_request - prev_rev_per_request)
        share_delta = float(completed - prev_share)
        rival_completed = float(
            baseline_share if baseline_completed_share is None else baseline_completed_share
        )
        result: Dict[str, float] = {
            **positive,
            **active_reward,
            **self.soft_constraints.diagnostics(costs),
            "reward_competitive_term": active_reward["reward_profit_advantage_component"],
            "reward_dominance_term": active_reward["reward_profit_advantage_utility"],
            "reward_trend_term": 0.0,
            "reward_share_delta": share_delta,
            "reward_rev_delta": revenue_delta,
            "reward_profit_delta": profit_delta,
            "reward_gap_delta": 0.0,
            "reward_momentum_component": 0.0,
            "reward_response_component": 0.0,
            "reward_pricing_discipline": discipline,
            "reward_efficiency_term": efficiency,
            "reward_profit_per_request": profitpr,
            "reward_profit_margin": margin,
            "reward_fulfillment_rate": float(np.clip(fulfillment_rate, 0.0, 1.0)),
            "reward_avg_wait_minutes": float(max(0.0, avg_wait_minutes)),
            "reward_driver_acceptance_rate": float(np.clip(driver_acceptance_rate, 0.0, 1.0)),
            "reward_service_quality": positive["reward_positive_service"],
            "reward_profit_objective": active_reward["reward_own_profit_utility"],
            "reward_revenue_objective": 0.0,
            "reward_share_objective": 0.0,
            "reward_price_gap_satisfaction": discipline,
            "reward_price_gap_deviation": abs(float(mean_gap) - self.constraint_config.target_gap),
            "reward_price_gap_abs_error": abs(float(mean_gap) - self.constraint_config.target_gap),
            "reward_gap_tolerance": self.constraint_config.overall_tolerance,
            "reward_price_gap_in_range": float(abs(float(mean_gap) - self.constraint_config.target_gap) <= self.constraint_config.overall_tolerance),
            "reward_choice_share": float(np.clip(share, 0.0, 1.0)),
            "reward_completed_share": float(np.clip(share if completed_share is None else completed_share, 0.0, 1.0)),
            "reward_served_share": float(np.clip(share if completed_share is None else completed_share, 0.0, 1.0)),
            "reward_zero_effect_action": float(bool(getattr(getattr(self, "firm1", None), "last_action_was_zero_effect", False))),
            "reward_saturated_action": float(bool(getattr(getattr(self, "firm1", None), "last_action_was_saturated", False))),
            "reward_action_reversal": float(self.action_stability.last_reversal),
            "reward_oscillation_count": float(oscillation),
            "reward_relative_profit_term": active_reward["reward_profit_advantage_utility"],
            "reward_relative_revenue_term": float(
                float(rev_per_request) - float(baseline_rev_per_request)
            ),
            "reward_relative_share_term": completed - rival_completed,
            "reward_absolute_profit_term": active_reward["reward_own_profit_utility"],
            "reward_absolute_revenue_term": float(rev_per_request),
            "reward_legacy_positive_business_utility": float(positive["reward"]),
        }
        # Legacy diagnostic columns remain zero so existing CSV/plot readers do
        # not break.  None of these fields participates in optimization.
        for key in (
            "reward_revenue_component", "reward_share_component", "reward_profit_component",
            "reward_unit_economics_component", "reward_price_gap_component", "reward_service_component",
            "reward_corrective_action_bonus", "reward_hold_inaction_penalty",
            "reward_baseline_loss_penalty", "reward_baseline_loss_component",
            "reward_overprice_penalty", "reward_overprice_component",
            "reward_underprice_penalty", "reward_underprice_component",
            "reward_action_change_penalty", "reward_action_realization_penalty",
            "reward_correction_pressure", "reward_revenue_improvement", "reward_share_improvement",
            "reward_profit_improvement", "reward_demand_loss", "reward_baseline_share",
            "reward_baseline_rev_per_request", "reward_baseline_profit_per_request",
            "reward_absolute_profit_term", "reward_relative_profit_term",
            "reward_absolute_revenue_term", "reward_relative_revenue_term", "reward_relative_share_term",
            "reward_wait_satisfaction", "reward_driver_acceptance_objective",
        ):
            result.setdefault(key, 0.0)
        # Compatibility aliases for old dashboards; costs are not reward penalties.
        result.update({
            "constraint_violation_share_floor": 0.0,
            "constraint_violation_fulfillment_floor": costs["fulfillment"],
            "constraint_violation_wait_limit": costs["wait"],
            "constraint_violation_gap_band": max(costs[name] for name in (
                "gap_overprice", "gap_underprice", "gap_0_2", "gap_2_5", "gap_5_10", "gap_10_plus"
            )),
            "constraint_violation_margin_floor": costs["margin"],
        })
        return result

    def _compute_rl_reward(self, m1: FirmMetrics, mean_gap: float, m2: Optional[FirmMetrics] = None) -> float:
        """Low-complexity reward shaping for better PPO learning signal."""
        diagnostics = self._reward_diagnostics(
            share=float(m1.chosen_share),
            completed_share=float(m1.completed_share),
            rev_per_request=float(m1.rev_per_request),
            mean_gap=float(mean_gap),
            profit_per_request=float(m1.profit_per_request),
            profit_margin=self._profit_margin(m1),
            fulfillment_rate=float(m1.fulfillment_rate),
            avg_wait_minutes=float(m1.avg_wait_minutes),
            driver_acceptance_rate=float(m1.driver_acceptance_rate),
            action_change_magnitude=float(getattr(getattr(self, "firm1", None), "last_action_magnitude", lambda: 0.0)()),
            baseline_share=float(m2.chosen_share) if m2 is not None else 0.0,
            baseline_completed_share=float(m2.completed_share) if m2 is not None else 0.0,
            baseline_rev_per_request=float(m2.rev_per_request) if m2 is not None else 0.0,
            baseline_profit_per_request=float(m2.profit_per_request) if m2 is not None else 0.0,
            prev_share=float(self.last_share),
            prev_rev_per_request=float(self.last_revpr),
            prev_profit_per_request=float(self.last_profitpr),
            prev_gap=float(self.last_gap),
            action_event=self.firm1_mode == "RL",
        )
        reward = float(diagnostics["reward"])
        self.last_reward = reward
        self.last_reward_diagnostics = dict(diagnostics)
        self._update_constraint_multipliers(diagnostics)
        return reward
    
    def _initialize_run_distributions(self) -> None:
        """Run-level slight variations to demographics/weather/ride nature priors."""
        self.agent_gen.apply_probability_variation(jitter_scale=0.05)
        self.market.refresh_run_probabilities(jitter_scale=0.05)
        self.last_crowd_response_stats = {}

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
    
    def _effective_simulation_sample_size(self, requested: int, collect_rows: bool = False) -> int:
        """Return the Monte Carlo profile count used for aggregate simulations."""
        requested_i = int(max(0, requested))
        if collect_rows or self.simulation_sample_cap <= 0:
            return requested_i
        return int(min(requested_i, self.simulation_sample_cap))

    def save_trained_model(self, path: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save a portable Firm1 policy artifact independent of the opponent."""
        if self.firm1_mode != "RL":
            raise ValueError("trained model artifacts require firm1_mode='RL'")
        output = os.path.abspath(os.path.expanduser(path))
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        artifact = {
            "format": "ride-response-platform-policy",
            "version": 7,
            "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "market_name": self.market_name,
            "opt_keys": list(self.shared_edit_keys),
            "action_to_steps": copy.deepcopy(self.firm1.action_to_steps),
            "single_state_dim": int(self.firm1.single_state_dim),
            "state_frame_stack": int(self.firm1.state_frame_stack),
            "action_feature_dim": int(self.firm1.action_feature_dim),
            "constraint_names": list(self.soft_constraints.names),
            "response_dim": int(self.firm1.agent.response_dim),
            "observation_feature_groups": {
                name: [int(index) for index in indices]
                for name, indices in PlatformObservationModel.feature_groups().items()
            },
            "city_context_keys": list(PlatformObservationModel.CITY_CONTEXT_KEYS),
            "architecture": "hierarchical-public-history-city-context-ppo",
            "gamma": float(self.firm1.agent.gamma),
            "gae_lambda": float(self.firm1.agent.lam),
            "network_state": {
                key: value.detach().cpu() for key, value in self.firm1.agent.net.state_dict().items()
            },
            "optimizer_state": self.firm1.agent.optimizer_state_dict(),
            "normalizer_state": self.firm1.agent.normalizer_state_dict(),
            # Tariffs, opponent controller state, observation queues, and dual
            # variables are deliberately excluded.  The artifact is a policy,
            # not a terminal training-world snapshot.
            "mdp_config": platform_mdp_config_payload(
                self.observation_config,
                self.positive_reward_config,
                self.constraint_config,
                self.training_stage_scheduler,
                self.long_term_profit_reward_config,
            ),
            "metadata": dict(metadata or {}),
        }
        torch.save(artifact, output)
        print(f">>> Saved portable trained policy -> {output}")
        return output

    def load_trained_model(
        self,
        path: str,
        *,
        load_optimizer: bool = False,
        restore_tariff: bool = False,
    ) -> Dict[str, Any]:
        """Load a policy without loading or replacing the benchmark opponent."""
        if self.firm1_mode != "RL":
            raise ValueError("trained model artifacts require firm1_mode='RL'")
        source = os.path.abspath(os.path.expanduser(path))
        try:
            artifact = torch.load(source, map_location=self.firm1.agent.device, weights_only=False)
        except TypeError:
            artifact = torch.load(source, map_location=self.firm1.agent.device)
        if artifact.get("format") != "ride-response-platform-policy":
            raise ValueError(f"unsupported trained model artifact: {source}")
        if int(artifact.get("version", 0)) < 7:
            raise ValueError(
                "trained model predates the version-7 public-history opponent encoder, "
                "continuous city context, and isolated actor/critic/response design; "
                "retrain the policy"
            )
        expected = {
            "single_state_dim": int(self.firm1.single_state_dim),
            "state_frame_stack": int(self.firm1.state_frame_stack),
            "action_feature_dim": int(self.firm1.action_feature_dim),
        }
        for key, value in expected.items():
            if int(artifact.get(key, -1)) != value:
                raise ValueError(
                    f"trained model {key}={artifact.get(key)} is incompatible with runtime {value}"
                )
        if list(artifact.get("opt_keys", [])) != list(self.shared_edit_keys):
            raise ValueError("trained model pricing keys do not match runtime pricing keys")
        if artifact.get("action_to_steps") != self.firm1.action_to_steps:
            raise ValueError("trained model action mapping does not match runtime action mapping")
        expected_groups = {
            name: [int(index) for index in indices]
            for name, indices in PlatformObservationModel.feature_groups().items()
        }
        if artifact.get("observation_feature_groups") != expected_groups:
            raise ValueError(
                "trained model observation feature groups do not match the runtime "
                "immediate/opponent/city hierarchy"
            )
        if list(artifact.get("city_context_keys", [])) != list(
            PlatformObservationModel.CITY_CONTEXT_KEYS
        ):
            raise ValueError("trained model city-context schema does not match runtime")
        if list(artifact.get("constraint_names", [])) != list(self.soft_constraints.names):
            raise ValueError(
                "trained model constraint critics do not match runtime active constraints; "
                f"artifact={artifact.get('constraint_names')} runtime={list(self.soft_constraints.names)}"
            )
        try:
            self.firm1.agent.net.load_state_dict(artifact["network_state"])
        except RuntimeError as exc:
            raise ValueError(
                "trained model network is incompatible with the hierarchical "
                "actor/critic architecture; retrain and save a version-7 artifact"
            ) from exc
        self.firm1.agent.load_normalizer_state_dict(
            artifact.get("normalizer_state")
        )
        if load_optimizer and artifact.get("optimizer_state"):
            self.firm1.agent.load_optimizer_state_dict(
                artifact["optimizer_state"]
            )
        if restore_tariff:
            for key, value in artifact.get("firm1_coefficients", {}).items():
                if key in self.shared_edit_keys:
                    set_coeff(self.market.curr_market, self.firm1.overrides, key, float(value))
        if artifact.get("soft_constraints"):
            self.soft_constraints.restore(artifact["soft_constraints"])
        self.constraint_lambdas = {
            name: float(self.soft_constraints.lambdas[index])
            for index, name in enumerate(self.soft_constraints.names)
        }
        self.firm1.reset_state_history()
        for observer in self.platform_observers.values():
            observer.reset()
        self.action_stability.reset()
        self._sync_agent_optimization_context()
        print(
            f">>> Loaded trained policy <- {source}; benchmark opponent remains {self.firm2_mode}."
        )
        return artifact

    def _adaptive_policy_validation(
        self,
        *,
        customers_per_step: int,
        profile_sample_size: int,
        horizon: int = 192,
    ) -> Dict[str, Any]:
        """Evaluate the policy in closed loop against unseen adaptive rivals.

        Validation must call ``agent.act`` repeatedly.  Scoring a single market
        batch at the current training tariff selects a tariff snapshot, not an
        adaptive policy, and leaks the terminal training state into evaluation.
        """
        saved = {
            "market": copy.deepcopy(self.market),
            "firm1_overrides": copy.deepcopy(self.firm1.overrides),
            "firm2": copy.deepcopy(self.firm2),
            "firm1_execution": self.firm1.snapshot_execution_state(),
            "observers": {name: observer.snapshot() for name, observer in self.platform_observers.items()},
            "stability": self.action_stability.snapshot(),
            "crowd": dict(getattr(self, "last_crowd_response_stats", {}) or {}),
            "driver_supply": copy.deepcopy(self.driver_supply),
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
        }
        tracked_names = (
            "last_share", "last_revpr", "last_profitpr", "last_gap",
            "last_fulfillment", "last_acceptance", "last_wait",
            "last_firm2_share", "last_firm2_revpr", "last_firm2_profitpr",
        )
        saved["last_values"] = {name: getattr(self, name) for name in tracked_names}
        mode_scores: List[float] = []
        rewards: List[float] = []
        own_shares: List[float] = []
        rival_shares: List[float] = []
        own_revenues: List[float] = []
        rival_revenues: List[float] = []
        own_profits: List[float] = []
        rival_profits: List[float] = []
        gaps: List[float] = []
        fulfillment: List[float] = []
        actions: List[int] = []
        validation_mode_names: List[str] = []
        mode_profit_advantages: List[float] = []
        mode_own_profit_means: List[float] = []
        mode_rival_profit_means: List[float] = []
        try:
            validation_modes = (
                "static",
                "adaptive_best_response_aggressive",
                "pi_price_gap",
                "queue_service_threshold",
                "region_supply_demand",
            )
            for rollout_index, mode in enumerate(validation_modes):
                self._reset_adaptive_episode(
                    tariff_reset_fraction=0.22,
                    opponent_mode=mode,
                    episode_seed=self.seed + 90_000 + 97 * rollout_index,
                )
                clock = OperationalClock()
                prev_share = 0.5
                prev_rev = 0.0
                prev_profit = 0.0
                prev_gap = 0.0
                mode_rewards: List[float] = []
                mode_own_profits: List[float] = []
                mode_rival_profits: List[float] = []
                mode_constraint_costs: Dict[str, List[float]] = {
                    name: [] for name in self.soft_constraints.names
                }
                for _ in range(max(2, int(horizon))):
                    ctx = self.market.sample_day_context()
                    hour = self.market.sample_timestep_hour().hour
                    raw_state, action_features, _ = self._publish_pricing_observation(
                        day_of_week=ctx.day_of_week,
                        hour=hour,
                        weather=ctx.weather,
                    )
                    state = self.firm1.stack_state(raw_state)
                    action_mask = self.firm1.feasible_action_mask(self.market)
                    action, *_ = self.firm1.agent.act(
                        state,
                        deterministic=True,
                        action_features=action_features,
                        action_mask=action_mask,
                    )
                    self.firm1.apply_action(action, self.market)
                    actions.append(int(action))
                    if clock.due(2):
                        base = self.market.curr_market
                        self.firm2.act(
                            city_base=base.base_fare,
                            city_pmin=base.per_minute,
                            city_pmile=base.per_mile,
                            city_booking=base.booking_fee,
                            city_airport=base.airport_fee,
                            hour=hour,
                            weather=ctx.weather,
                        )
                    profiles = self._sample_profiles_from_pool(profile_sample_size)
                    _, m1, m2, gap, _, _ = self.simulate_batch(
                        day_of_week=ctx.day_of_week,
                        weather=ctx.weather,
                        hour=hour,
                        customers_per_step=customers_per_step,
                        sampled_profiles=profiles,
                        collect_rows=False,
                    )
                    self._ingest_platform_observations(m1, m2)
                    if clock.due(2):
                        supply, supply_vec = self._driver_supply_context("Firm2", "Firm1")
                        self.firm2.update(
                            metrics=m2,
                            price_gap_mean=self._observed_own_minus_competitor_gap("Firm2"),
                            supply_state=supply,
                            supply_state_vector=supply_vec,
                        )
                    diag = self._reward_diagnostics(
                        share=float(m1.chosen_share),
                        completed_share=float(m1.completed_share),
                        rev_per_request=float(m1.rev_per_request),
                        mean_gap=float(gap),
                        prev_share=prev_share,
                        prev_rev_per_request=prev_rev,
                        prev_profit_per_request=prev_profit,
                        prev_gap=prev_gap,
                        profit_per_request=float(m1.profit_per_request),
                        profit_margin=self._profit_margin(m1),
                        fulfillment_rate=float(m1.fulfillment_rate),
                        avg_wait_minutes=float(m1.avg_wait_minutes),
                        driver_acceptance_rate=float(m1.driver_acceptance_rate),
                        baseline_share=float(m2.chosen_share),
                        baseline_completed_share=float(m2.completed_share),
                        baseline_rev_per_request=float(m2.rev_per_request),
                        baseline_profit_per_request=float(m2.profit_per_request),
                        action_event=True,
                    )
                    mode_own_profits.append(float(m1.profit_per_request))
                    mode_rival_profits.append(float(m2.profit_per_request))
                    for name in self.soft_constraints.names:
                        mode_constraint_costs[name].append(
                            float(diag.get(f"constraint_cost_{name}", 0.0))
                        )
                    mode_rewards.append(float(diag["reward"]))
                    rewards.append(float(diag["reward"]))
                    own_shares.append(float(m1.completed_share))
                    rival_shares.append(float(m2.completed_share))
                    own_revenues.append(float(m1.rev_per_request))
                    rival_revenues.append(float(m2.rev_per_request))
                    own_profits.append(float(m1.profit_per_request))
                    rival_profits.append(float(m2.profit_per_request))
                    gaps.append(float(gap))
                    fulfillment.append(float(m1.fulfillment_rate))
                    prev_share = float(m1.completed_share)
                    prev_rev = float(m1.rev_per_request)
                    prev_profit = float(m1.profit_per_request)
                    prev_gap = float(gap)
                    clock.advance()
                # Score the long-run half of each closed-loop rollout. This
                # rejects policies that win only during their initial tariff
                # transient and then collapse into an unprofitable price war.
                # while still retaining the full trace in returned diagnostics.
                tail_start = min(
                    len(mode_own_profits) - 1,
                    max(0, len(mode_own_profits) // 2),
                )
                tail_own_profit = float(np.mean(mode_own_profits[tail_start:]))
                tail_rival_profit = float(np.mean(mode_rival_profits[tail_start:]))
                validation_mode_names.append(str(mode))
                mode_own_profit_means.append(tail_own_profit)
                mode_rival_profit_means.append(tail_rival_profit)
                mode_profit_advantages.append(tail_own_profit - tail_rival_profit)
                mode_scores.append(self._validation_score_from_metrics(
                    reward=float(np.mean(mode_rewards[tail_start:])),
                    share=0.0,
                    revpr=0.0,
                    profitpr=tail_own_profit,
                    fulfillment=0.0,
                    gap=0.0,
                    rival_profitpr=tail_rival_profit,
                    constraint_costs={
                        name: float(np.mean(values[tail_start:]))
                        for name, values in mode_constraint_costs.items()
                    },
                ))
        finally:
            self.market = saved["market"]
            self.firm1.overrides = saved["firm1_overrides"]
            self.firm2 = saved["firm2"]
            self.firm1.restore_execution_state(saved["firm1_execution"])
            for name, snapshot in saved["observers"].items():
                self.platform_observers[name].restore(snapshot)
            self.action_stability.restore(saved["stability"])
            self.last_crowd_response_stats = saved["crowd"]
            self.driver_supply = saved["driver_supply"]
            self.rng.bit_generator.state = saved["rng_state"]
            for name, value in saved["last_values"].items():
                setattr(self, name, value)
        mean_mode_score = float(np.mean(mode_scores)) if mode_scores else -float("inf")
        mode_score_std = float(np.std(mode_scores)) if mode_scores else float("inf")
        worst_mode_score = float(np.min(mode_scores)) if mode_scores else -float("inf")
        mean_profit_advantage = (
            float(np.mean(mode_profit_advantages))
            if mode_profit_advantages else -float("inf")
        )
        worst_profit_advantage = (
            float(np.min(mode_profit_advantages))
            if mode_profit_advantages else -float("inf")
        )
        profit_win_rate = (
            float(np.mean(np.asarray(mode_profit_advantages) > 0.0))
            if mode_profit_advantages else 0.0
        )
        return {
            "score": (
                0.35 * mean_mode_score
                + 0.65 * worst_mode_score
                - 0.15 * mode_score_std
            ),
            "mode_score_mean": mean_mode_score,
            "mode_score_std": mode_score_std,
            "mode_score_worst": worst_mode_score,
            "mean_profit_advantage": mean_profit_advantage,
            "worst_profit_advantage": worst_profit_advantage,
            "profit_win_rate": profit_win_rate,
            "mode_profit_advantages": {
                name: float(advantage)
                for name, advantage in zip(
                    validation_mode_names, mode_profit_advantages
                )
            },
            "mode_own_profit_per_request": {
                name: float(profit)
                for name, profit in zip(
                    validation_mode_names, mode_own_profit_means
                )
            },
            "mode_rival_profit_per_request": {
                name: float(profit)
                for name, profit in zip(
                    validation_mode_names, mode_rival_profit_means
                )
            },
            "reward": float(np.mean(rewards)) if rewards else 0.0,
            "share": float(np.mean(own_shares)) if own_shares else 0.0,
            "rival_share": float(np.mean(rival_shares)) if rival_shares else 0.0,
            "rev_per_request": float(np.mean(own_revenues)) if own_revenues else 0.0,
            "rival_rev_per_request": float(np.mean(rival_revenues)) if rival_revenues else 0.0,
            "profit_per_request": float(np.mean(own_profits)) if own_profits else 0.0,
            "rival_profit_per_request": float(np.mean(rival_profits)) if rival_profits else 0.0,
            "fulfillment_rate": float(np.mean(fulfillment)) if fulfillment else 0.0,
            "price_gap": float(np.mean(gaps)) if gaps else 0.0,
            "action_diversity": float(len(set(actions)) / max(1, len(self.firm1.action_to_steps))),
            "action_switch_rate": float(np.mean(np.diff(actions) != 0)) if len(actions) > 1 else 0.0,
        }

    def run_experiment(
        self,
        train_timesteps: int = 1500,
        train_customers_per_step: int = 5000,
        eval_timesteps: int = 200,
        eval_customers_per_step: int = 1000,
        profiles_out: Optional[str] = None,
        profiles_log_limit: int = 200000,
        train_steps_per_day: int = 10,
        ppo_update_interval_days: int = 20,
        ppo_min_rollout_transitions: int = 256,
        stochastic_training: bool = True,
        firm1_action_interval_steps: int = -1,
        firm2_action_interval_days: int = -1,
        eval_policy_mode: str = "argmax",
        eval_policy_temperature: float = 0.50,
        eval_top2_margin: float = 0.05,
        eval_guardrail_mode: str = "log_only",
        training_enabled: bool = True,
        trained_model_out: Optional[str] = None,
    ):
        """Run workflow: synthetic-data RL training (day/timestep cadence), then held-out evaluation."""
        training_enabled = bool(training_enabled)
        if not training_enabled:
            train_timesteps = 0
            stochastic_training = False
        self._initialize_run_distributions()
        for observer in self.platform_observers.values():
            observer.reset()
        self.action_stability.reset()
        if self.firm1_mode == "RL":
            self.firm1.reset_state_history()
        
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
            if self.firm1_mode == "RL":
                self.firm1.reset_state_history()
        self._refresh_profile_pool(rides_per_timestep=train_customers_per_step)
        train_profile_sample_size = self._effective_simulation_sample_size(train_customers_per_step, collect_rows=False)
        eval_profile_sample_size = self._effective_simulation_sample_size(eval_customers_per_step, collect_rows=False)

        print(
            f">>> Synthetic setup: profile_pool={self.agent_gen.total_customers}, "
            f"train days={train_timesteps} x {train_steps_per_day} steps/day x {train_customers_per_step} rides, "
            f"eval timesteps={eval_timesteps} x {eval_customers_per_step} rides, "
            f"mc_samples(train/eval)={train_profile_sample_size}/{eval_profile_sample_size}"
        )
        if self.firm1_mode == "RL":
            update_every = int(max(1, ppo_update_interval_days))
            action_interval_steps = int(firm1_action_interval_steps)
            if action_interval_steps <= 0:
                # Default to one pricing decision per market timestep for both
                # static and dynamic opponents.  Static-opponent training used
                # to hold a single price action for an entire synthetic day,
                # which produced too few PPO decisions and made it hard for
                # Firm1 to learn coefficient-specific profit improvements.
                action_interval_steps = 1
            action_interval_steps = int(max(1, action_interval_steps))
            base_action_interval_steps = int(action_interval_steps)
            approx_decisions = int(np.ceil(max(1, int(train_steps_per_day)) / float(action_interval_steps)))
            print(
                f">>> PPO rollout/update cadence: {update_every} day(s) per optimizer step "
                f"(~{update_every * approx_decisions} price decisions/update; "
                f"minimum rollout={max(32, int(ppo_min_rollout_transitions))} transitions; "
                f"Firm1 holds each price action for {action_interval_steps} step(s))."
            )
        else:
            action_interval_steps = 1
            base_action_interval_steps = 1
        # The public argument retains its legacy name for CLI compatibility,
        # but cadence is measured on the shared operational decision clock.
        firm2_interval_steps = int(firm2_action_interval_days)
        if firm2_interval_steps <= 0:
            firm2_interval_steps = (
                max(1, int(np.ceil(max(1, int(ppo_update_interval_days)) / 2.0)))
                if self.firm2_mode != "static"
                else max(1, int(ppo_update_interval_days))
            )
        firm2_interval_steps = int(max(1, firm2_interval_steps))
        if self.firm2_mode != "static":
            print(
                f">>> Firm2 heuristic cadence: manipulate/update every "
                f"{firm2_interval_steps} operational decision period(s)."
            )
        
        if training_enabled:
            print(f">>> Training RL agent on synthetic {self.market_name} sampling (day/timestep cadence)...")
        else:
            print(">>> Evaluation-only mode: using loaded Firm1 policy without optimizer updates.")
        reward_history: List[float] = []
        sampled_profile_rows: List[Dict[str, Any]] = []
        profile_limit_reached = False

        # Keep the competitor at the same strategy level in training and
        # evaluation.  A static competitor remains static; a heuristic competitor
        # remains heuristic.  This avoids the previous train/eval distribution
        # shift where RL trained against a heuristic opponent but evaluated
        # against a static opponent.
        opponent_mode = str(self.firm2_mode)
        print(f">>> Competitor mode for both training and evaluation: {opponent_mode}")
        benchmark_opponent = copy.deepcopy(self.firm2)
        opponent_population = [
            opponent_mode,
            # Robust stages must contain both non-reactive and reactive rivals.
            # Static episodes teach monopoly-like margin discovery; adaptive
            # episodes teach anticipation and selective competitive response.
            "static",
            "adaptive_best_response",
            "adaptive_best_response_aggressive",
            "pi_price_gap",
            "queue_service_threshold",
            "region_supply_demand",
            "surge_driver_incentive",
            "mpc_grid",
            "heuristic_margin",
            "heuristic_random",
        ]
        opponent_population = list(dict.fromkeys(opponent_population))
        training_clock = OperationalClock()
        
        update_every = int(max(1, ppo_update_interval_days))
        best_checkpoint: Optional[Dict[str, Any]] = None
        best_validation_score = -float("inf")
        minimum_rollout = int(max(32, ppo_min_rollout_transitions))
        validation_interval = max(150, 8 * update_every)
        latest_validation: Dict[str, float] = {}
        last_ppo_metrics = {
            "loss": np.nan,
            "policy_loss": np.nan,
            "value_loss": np.nan,
            "critic_loss": np.nan,
            "critic_grad_norm": np.nan,
            "constraint_value_loss": np.nan,
            "risk_value_loss": np.nan,
            "action_q_loss": np.nan,
            "response_loss": np.nan,
            "lagrangian_adv_mean": np.nan,
            "lagrangian_adv_std": np.nan,
            "constraint_lambda_mean": np.nan,
            "risk_coeff": np.nan,
            "approx_kl": np.nan,
            "clipfrac": np.nan,
            "entropy": np.nan,
            "policy_entropy": np.nan,
            "discrete_policy_entropy": np.nan,
            "magnitude_entropy": np.nan,
            "policy_entropy_fraction": 1.0,
            "ent_coeff": float(self.firm1.agent.ent_coeff) if self.firm1_mode == "RL" else np.nan,
            "clip_eps": float(self.firm1.agent.clip_eps) if self.firm1_mode == "RL" else np.nan,
            "exploration_rate": float(self.firm1.agent.exploration_rate) if self.firm1_mode == "RL" else np.nan,
            "action_coverage": float(self.firm1.agent._action_coverage()) if self.firm1_mode == "RL" else np.nan,
            "explained_variance": np.nan,
            "optimizer_steps": 0,
            "actor_optimizer_steps": 0,
            "critic_optimizer_steps": 0,
            "lr": float(self.firm1.agent.curr_lr) if self.firm1_mode == "RL" else np.nan,
            "critic_lr": float(self.firm1.agent.critic_opt.param_groups[0]["lr"]) if self.firm1_mode == "RL" else np.nan,
            "update_performed": False,
            "learning_signal_ok": False,
            "run_mode": "rollout_collection",
        }

        active_stage_name: Optional[str] = None
        # Once the policy has demonstrated stable reward and robust validation
        # profit dominance, do not re-open the high-exploration schedule because
        # of one noisy later rollout.  This latch controls optimization only; it
        # never changes simulator outcomes or evaluation actions.
        training_convergence_latched = False
        latched_convergence_std = float("inf")
        episode_end_day = 0
        for d in range(train_timesteps):
            train_progress = float(d + 1) / max(1.0, float(train_timesteps))
            self.current_training_stage = self.training_stage_scheduler.stage_at(train_progress)
            episode_days = int(max(1, self.current_training_stage.episode_days))
            stage_changed = active_stage_name != self.current_training_stage.name
            episode_reset = bool(d == 0 or stage_changed or d >= episode_end_day)
            if episode_reset:
                pool = opponent_population[: max(1, self.current_training_stage.opponent_pool_size)]
                selected_mode = str(self.rng.choice(pool))
                self._reset_adaptive_episode(
                    tariff_reset_fraction=self.current_training_stage.tariff_reset_fraction,
                    opponent_mode=selected_mode,
                    episode_seed=self.seed + 50_000 + d,
                )
                training_clock.reset()
                active_stage_name = self.current_training_stage.name
                episode_end_day = min(train_timesteps, d + episode_days)
                print(
                    f">>> [curriculum episode] day={d + 1} stage={self.current_training_stage.name} "
                    f"opponent={selected_mode} horizon={episode_days}d "
                    f"tariff_reset=±{100.0 * self.current_training_stage.tariff_reset_fraction:.0f}%"
                )
            next_stage_name = self.current_training_stage.name
            if (d + 1) < train_timesteps:
                next_progress = float(d + 2) / max(1.0, float(train_timesteps))
                next_stage_name = self.training_stage_scheduler.stage_at(
                    next_progress
                ).name
            day_regime_boundary = bool(
                (d + 1) >= episode_end_day
                or next_stage_name != self.current_training_stage.name
                or (d + 1) == train_timesteps
            )
            stage_action_interval_steps = int(max(
                base_action_interval_steps,
                self.current_training_stage.action_hold_steps,
            ))
            effective_firm2_interval_steps = int(max(
                1,
                firm2_interval_steps * self.current_training_stage.opponent_cadence_multiplier,
            ))
            if self.firm1_mode == "RL":
                self.firm1.agent.set_stage_controls(
                    exploration_rate=self.current_training_stage.exploration_rate,
                    entropy_scale=self.current_training_stage.entropy_scale,
                    learning_rate_scale=self.current_training_stage.learning_rate_scale,
                )
                # Stage controls are reapplied at the start of every simulated
                # day.  Preserve a validated convergence decision across that
                # reset so late training cannot reopen exploration and destroy
                # a robustly profitable policy.
                if training_convergence_latched:
                    self.firm1.configure_training_controls(
                        progress=train_progress,
                        reward_converged=True,
                        reward_std=latched_convergence_std,
                    )
            self._configure_constraint_curriculum(train_progress)
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
            choice_share_sum = 0.0
            completed_share_sum = 0.0
            firm2_choice_share_sum = 0.0
            firm2_completed_share_sum = 0.0
            firm2_revpr_sum = 0.0
            firm2_profitpr_sum = 0.0
            revpr_sum = 0.0
            profitpr_sum = 0.0
            gap_sum = 0.0
            fulfillment_sum = 0.0
            wait_sum = 0.0
            driver_accept_sum = 0.0
            driver_paypr_sum = 0.0
            action_counts: Counter[int] = Counter()
            raw_top_action_counts: Counter[int] = Counter()
            sampled_not_top_count = 0
            train_policy_diag_sums: Dict[str, float] = defaultdict(float)
            train_policy_diag_count = 0
            last_action = -1
            recovery_guardrail_count = 0
            recovery_guardrail_recommend_count = 0
            recovery_guardrail_reasons: Counter[str] = Counter()
            pending_rl_step = None
            pending_reward_sum = 0.0
            pending_constraint_sum = np.zeros(len(self.soft_constraints.names), dtype=np.float32)
            pending_risk_sum = 0.0
            pending_discount = 1.0
            pending_response_sum = np.zeros(12, dtype=np.float32)
            pending_response_baseline = np.zeros(12, dtype=np.float32)
            pending_count = 0

            for t, hour in enumerate(hours):
                base = self.market.curr_market
                rl_step = None
                if self.firm1_mode == "RL":
                    raw_state_vec, action_features, _ = self._publish_pricing_observation(
                        day_of_week=day_ctx.day_of_week, hour=hour, weather=day_ctx.weather
                    )
                    s_vec = self.firm1.stack_state(raw_state_vec)
                    decision_due = (t % stage_action_interval_steps) == 0
                    if decision_due:
                        pending_response_baseline = self._response_target_from_metrics(
                            FirmMetrics(), FirmMetrics(), 0.0
                        )
                        action_mask = self.firm1.feasible_action_mask(self.market)
                        raw_diag = self.firm1.agent.policy_diagnostics(
                            s_vec, action_features=action_features, action_mask=action_mask
                        )
                        action, s_ts, old_logp, val, af_ts, mask_ts = self.firm1.agent.act(
                            s_vec, action_features=action_features, action_mask=action_mask
                        )
                        self.firm1.apply_action(action, self.market)
                        action_counts[int(action)] += 1
                        raw_top = int(raw_diag.get("policy_top_action", -1))
                        raw_top_action_counts[raw_top] += 1
                        sampled_not_top_count += int(raw_top >= 0 and int(action) != raw_top)
                        for diag_key, diag_value in raw_diag.items():
                            train_policy_diag_sums[diag_key] += float(diag_value)
                        train_policy_diag_count += 1
                        last_action = int(action)
                        rl_step = (action, s_ts, old_logp, val, af_ts, mask_ts)
                        pending_rl_step = rl_step
                        pending_reward_sum = 0.0
                        pending_constraint_sum = np.zeros(len(self.soft_constraints.names), dtype=np.float32)
                        pending_risk_sum = 0.0
                        pending_discount = 1.0
                        pending_response_sum = np.zeros(12, dtype=np.float32)
                        pending_count = 0
                    else:
                        rl_step = None
                elif self.firm1_mode != "static":
                    self.firm1.act(
                        city_base=base.base_fare,
                        city_pmin=base.per_minute,
                        city_pmile=base.per_mile,
                        city_booking=base.booking_fee,
                        city_airport=base.airport_fee,
                        hour=hour,
                        weather=day_ctx.weather,
                    )

                firm2_due = bool(
                    not isinstance(self.firm2, FirmStaticPricer)
                    and not self.current_training_stage.freeze_opponent
                    and training_clock.due(effective_firm2_interval_steps)
                )
                if firm2_due:
                    self.firm2.act(
                        city_base=base.base_fare,
                        city_pmin=base.per_minute,
                        city_pmile=base.per_mile,
                        city_booking=base.booking_fee,
                        city_airport=base.airport_fee,
                        hour=hour,
                        weather=day_ctx.weather,
                    )

                sampled_profiles = self._sample_profiles_from_pool(train_profile_sample_size)
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
                self._ingest_platform_observations(m1, m2)

                if firm2_due:
                    f2_supply, f2_supply_vec = self._driver_supply_context("Firm2", "Firm1")
                    self.firm2.update(
                        metrics=m2,
                        price_gap_mean=self._observed_own_minus_competitor_gap("Firm2"),
                        supply_state=f2_supply,
                        supply_state_vector=f2_supply_vec,
                    )
                    
                action_movement = 0.0
                if self.firm1_mode == "RL" and rl_step is not None:
                    action_movement = float(self.firm1.last_action_magnitude())
                    
                reward_diag = self._reward_diagnostics(
                    share=float(m1.chosen_share),
                    completed_share=float(m1.completed_share),
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
                    action_change_magnitude=action_movement,
                    baseline_share=float(m2.chosen_share),
                    baseline_completed_share=float(m2.completed_share),
                    baseline_rev_per_request=float(m2.rev_per_request),
                    baseline_profit_per_request=float(m2.profit_per_request),
                    action_event=rl_step is not None,
                )

                if self.firm1_mode == "RL" and pending_rl_step is not None:
                    reward = float(reward_diag["reward"])
                    self.last_reward = reward
                    self._update_constraint_multipliers(reward_diag)
                    pending_reward_sum += pending_discount * reward
                    pending_constraint_sum += (
                        pending_discount * self._constraint_vector_from_diag(reward_diag)
                    )
                    pending_risk_sum += (
                        pending_discount * self._risk_cost_from_diag(reward_diag)
                    )
                    pending_response_sum += self._response_target_from_metrics(m1, m2, mean_gap)
                    pending_count += 1
                    pending_discount *= float(self.firm1.agent.gamma)
                    end_of_day = (t == len(hours) - 1)
                    episode_truncated = bool(end_of_day and day_regime_boundary)
                    interval_closed = ((t + 1) % stage_action_interval_steps == 0) or end_of_day
                    if interval_closed and pending_count > 0:
                        action, s_ts, old_logp, val, af_ts, mask_ts = pending_rl_step
                        next_v = None
                        next_constraint_v = None
                        next_risk_v = None
                        if episode_truncated:
                            (
                                next_v,
                                next_constraint_v,
                                next_risk_v,
                            ) = self._ppo_bootstrap_estimates(
                                day_of_week=day_ctx.day_of_week,
                                hour=hour,
                                weather=day_ctx.weather,
                            )
                        self.firm1.agent.store(
                            s_ts,
                            action,
                            float(pending_reward_sum),
                            False,
                            None,
                            old_logp,
                            val,
                            constraint_costs=pending_constraint_sum,
                            risk_cost=float(pending_risk_sum),
                            response_target=self._response_delta(
                                pending_response_baseline,
                                pending_response_sum / max(1, pending_count),
                            ),
                            action_features=af_ts,
                            action_mask=mask_ts,
                            discount=float(pending_discount),
                            duration=int(pending_count),
                            truncated=episode_truncated,
                            next_value=next_v,
                            next_constraint_values=next_constraint_v,
                            next_risk_value=next_risk_v,
                        )
                        pending_rl_step = None
                        pending_reward_sum = 0.0
                        pending_constraint_sum = np.zeros(len(self.soft_constraints.names), dtype=np.float32)
                        pending_risk_sum = 0.0
                        pending_discount = 1.0
                        pending_response_sum = np.zeros(12, dtype=np.float32)
                        pending_count = 0
                    if end_of_day:
                        before_snapshot = self._coeff_snapshot()
                        guardrail_diag = self._apply_eval_guardrail(
                            "log_only",
                            metrics=m1,
                            price_gap_f2_minus_f1=float(mean_gap),
                            base=base,
                        )
                        after_snapshot = self._coeff_snapshot()
                        guardrail_delta = self._coeff_delta(after_snapshot, before_snapshot)
                        if guardrail_delta:
                            recovery_guardrail_count += 1
                        if guardrail_diag and guardrail_diag.get("reasons"):
                            recovery_guardrail_recommend_count += 1
                            recovery_guardrail_reasons.update(str(r) for r in guardrail_diag.get("reasons", []))
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
                    "reward_revenue_component",
                    "reward_share_component",
                    "reward_profit_component",
                    "reward_unit_economics_component",
                    "reward_price_gap_component",
                    "reward_service_component",
                    "reward_corrective_action_bonus",
                    "reward_hold_inaction_penalty",
                    "reward_baseline_loss_penalty",
                    "reward_overprice_penalty",
                    "reward_underprice_penalty",
                    "reward_action_change_penalty",
                    "reward_action_realization_penalty",
                    "reward_correction_pressure",
                    "reward_revenue_improvement",
                    "reward_share_improvement",
                    "reward_profit_improvement",
                    "reward_demand_loss",
                    "reward_baseline_share",
                    "reward_baseline_rev_per_request",
                    "reward_baseline_profit_per_request",
                    "reward_absolute_profit_term",
                    "reward_relative_profit_term",
                    "reward_absolute_revenue_term",
                    "reward_relative_revenue_term",
                    "reward_relative_share_term",
                    "reward_own_profit_utility",
                    "reward_profit_advantage_utility",
                    "reward_own_profit_component",
                    "reward_profit_advantage_component",
                    "reward_intervention_cost",
                    "reward_reversal_cost",
                    "reward_intervention_magnitude",
                    "reward_profit_advantage_per_request",
                    "reward_momentum_component",
                    "reward_profit_delta",
                    "reward_gap_delta",
                    "reward_price_gap_deviation",
                    "reward_price_gap_satisfaction",
                    "reward_service_quality",
                    "reward_wait_satisfaction",
                    "reward_driver_acceptance_objective",
                    "reward_price_gap_in_range",
                    "reward_gap_tolerance",
                    "reward_price_gap_abs_error",
                    "reward_served_share",
                    "constraint_violation_share_floor",
                    "constraint_violation_fulfillment_floor",
                    "constraint_violation_wait_limit",
                    "constraint_violation_gap_band",
                    "constraint_violation_margin_floor",
                    "reward_zero_effect_action",
                    "reward_saturated_action",
                ):
                    reward_component_sums[component_key] += float(reward_diag.get(component_key, 0.0))
                for component_key, component_value in reward_diag.items():
                    if component_key.startswith(("reward_positive_", "constraint_cost_", "constraint_lambda_")):
                        reward_component_sums[component_key] += float(component_value)

                share_sum += float(m1.share)
                choice_share_sum += float(m1.chosen_share)
                completed_share_sum += float(m1.completed_share)
                firm2_choice_share_sum += float(m2.chosen_share)
                firm2_completed_share_sum += float(m2.completed_share)
                firm2_revpr_sum += float(m2.rev_per_request)
                firm2_profitpr_sum += float(m2.profit_per_request)
                revpr_sum += float(m1.rev_per_request)
                profitpr_sum += float(m1.profit_per_request)
                gap_sum += float(mean_gap)
                fulfillment_sum += float(m1.fulfillment_rate)
                wait_sum += float(m1.avg_wait_minutes)
                driver_accept_sum += float(m1.driver_acceptance_rate)
                driver_paypr_sum += float(m1.driver_pay / max(1, m1.total))
                self._update_recent_response_emas(m1, mean_gap, m2=m2)
                self.last_share = float(m1.share)
                self.last_revpr = float(m1.rev_per_request)
                self.last_gap = float(mean_gap)
                self.last_profitpr = float(m1.profit_per_request)
                self.last_fulfillment = float(m1.fulfillment_rate)
                self.last_acceptance = float(m1.driver_acceptance_rate)
                self.last_wait = float(m1.avg_wait_minutes)
                self.last_driver_paypr = float(m1.driver_pay / max(1, m1.total))
                self.last_firm2_share = float(m2.share)
                self.last_firm2_revpr = float(m2.rev_per_request)
                self.last_firm2_profitpr = float(m2.profit_per_request)
                self.last_firm2_fulfillment = float(m2.fulfillment_rate)
                self.last_firm2_acceptance = float(m2.driver_acceptance_rate)
                self.last_firm2_wait = float(m2.avg_wait_minutes)
                self.last_firm2_driver_paypr = float(m2.driver_pay / max(1, m2.total))
                training_clock.advance()

            ppo_metrics = {
                **{
                    key: np.nan
                    for key in (
                        "loss", "policy_loss", "value_loss", "constraint_value_loss",
                        "risk_value_loss", "action_q_loss", "response_loss", "lagrangian_adv_mean",
                        "lagrangian_adv_std", "constraint_lambda_mean", "risk_coeff",
                        "approx_kl", "clipfrac", "entropy", "policy_entropy",
                        "discrete_policy_entropy", "magnitude_entropy", "explained_variance",
                        "rollout_action_diversity", "rollout_reward_std", "credited_reward_std",
                        "continuous_magnitude_mean", "continuous_magnitude_std",
                        "raw_policy_argmax_concentration", "raw_policy_hold_probability",
                        "state_action_sensitivity", "state_action_mi",
                    )
                },
                "policy_entropy_fraction": float(self.firm1.agent.last_policy_entropy_fraction) if self.firm1_mode == "RL" else np.nan,
                "ent_coeff": float(self.firm1.agent.ent_coeff) if self.firm1_mode == "RL" else np.nan,
                "clip_eps": float(self.firm1.agent.clip_eps) if self.firm1_mode == "RL" else np.nan,
                "exploration_rate": float(self.firm1.agent.exploration_rate) if self.firm1_mode == "RL" else np.nan,
                "action_coverage": float(self.firm1.agent._action_coverage()) if self.firm1_mode == "RL" else np.nan,
                "optimizer_steps": 0,
                "lr": float(self.firm1.agent.curr_lr) if self.firm1_mode == "RL" else np.nan,
                "update_performed": False,
                "learning_signal_ok": False,
                "run_mode": "rollout_collection",
            }
            if self.firm1_mode == "RL":
                should_update = bool(
                    (
                        (d + 1) % update_every == 0
                        or (d + 1) == train_timesteps
                    )
                    and self.firm1.agent.buffer_size >= minimum_rollout
                )
                if should_update:
                    self._sync_agent_optimization_context()
                    bootstrap_v, bootstrap_constraint_v, bootstrap_risk_v = self._ppo_bootstrap_estimates(
                        day_of_week=day_ctx.day_of_week,
                        hour=hours[-1],
                        weather=day_ctx.weather,
                    )
                    ppo_metrics = self.firm1.agent.update(
                        epochs=self.ppo_update_epochs,
                        batch_size=self.ppo_batch_size,
                        bootstrap_value=bootstrap_v,
                        bootstrap_constraint_values=bootstrap_constraint_v,
                        bootstrap_risk_value=bootstrap_risk_v,
                    )
                    last_ppo_metrics = dict(ppo_metrics)
                
            n_steps = max(1, len(hours))
            avg_reward = float(reward_sum / n_steps)
            reward_history.append(float(avg_reward))
                    
            moving_window = reward_history[-min(len(reward_history), 20):]
            moving_avg20 = float(np.mean(moving_window)) if moving_window else 0.0
            moving_std20 = float(np.std(moving_window)) if moving_window else 0.0
            avg_share = float(share_sum / n_steps)
            avg_choice_share = float(choice_share_sum / n_steps)
            avg_completed_share = float(completed_share_sum / n_steps)
            avg_firm2_choice_share = float(firm2_choice_share_sum / n_steps)
            avg_firm2_completed_share = float(firm2_completed_share_sum / n_steps)
            avg_firm2_revpr = float(firm2_revpr_sum / n_steps)
            avg_firm2_profitpr = float(firm2_profitpr_sum / n_steps)
            avg_revpr = float(revpr_sum / n_steps)
            avg_profitpr = float(profitpr_sum / n_steps)
            avg_gap = float(gap_sum / n_steps)
            avg_fulfillment = float(fulfillment_sum / n_steps)
            avg_wait = float(wait_sum / n_steps)
            decision_count = max(1, sum(action_counts.values()))
            avg_driver_accept = float(driver_accept_sum / n_steps)
            avg_driver_paypr = float(driver_paypr_sum / n_steps)
            dominant_action = int(action_counts.most_common(1)[0][0]) if action_counts else -1
            dominant_action_rate = float(action_counts.most_common(1)[0][1] / decision_count) if action_counts else 0.0
            dominant_raw_top_action = int(raw_top_action_counts.most_common(1)[0][0]) if raw_top_action_counts else -1
            dominant_raw_top_action_rate = float(raw_top_action_counts.most_common(1)[0][1] / decision_count) if raw_top_action_counts else 0.0
            avg_policy_diag = {
                k: float(v / max(1, train_policy_diag_count))
                for k, v in train_policy_diag_sums.items()
            }
            
            dominant_action_steps = (
                self.firm1.action_steps(dominant_action)
                if self.firm1_mode == "RL" and hasattr(self.firm1, "action_steps")
                else {}
            )
            dominant_action_label = (
                self.firm1.action_label(dominant_action)
                if self.firm1_mode == "RL" and hasattr(self.firm1, "action_label")
                else ""
            )
            last_action_steps = (
                self.firm1.action_steps(last_action)
                if self.firm1_mode == "RL" and hasattr(self.firm1, "action_steps")
                else {}
            )

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
                "avg_choice_share": avg_choice_share,
                "avg_completed_share": avg_completed_share,
                "avg_firm2_choice_share": avg_firm2_choice_share,
                "avg_firm2_completed_share": avg_firm2_completed_share,
                "avg_rev_per_request": avg_revpr,
                "avg_profit_per_request": avg_profitpr,
                "avg_firm2_rev_per_request": avg_firm2_revpr,
                "avg_firm2_profit_per_request": avg_firm2_profitpr,
                "avg_profit_advantage_per_request": (
                    avg_profitpr - avg_firm2_profitpr
                ),
                "avg_price_gap_f2_minus_f1": avg_gap,
                "avg_price_gap_abs_error": float(abs(avg_gap - self.reward_target_price_gap)),
                "avg_gap_violation_025": float(abs(avg_gap - self.reward_target_price_gap) > 0.25),
                "avg_gap_violation_050": float(abs(avg_gap - self.reward_target_price_gap) > 0.50),
                "constraint_curriculum_scale": float(self.constraint_curriculum_scale),
                "training_stage": str(self.current_training_stage.name),
                "training_stage_constraint_scale": float(self.current_training_stage.constraint_scale),
                "training_stage_action_hold_steps": int(stage_action_interval_steps),
                "training_stage_opponent_frozen": bool(self.current_training_stage.freeze_opponent),
                "avg_fulfillment_rate": avg_fulfillment,
                "avg_wait_minutes": avg_wait,
                "driver_acceptance_rate": avg_driver_accept,
                "driver_pay_per_request": avg_driver_paypr,
                "driver_reward_scale": float(self.driver_reward_scale_current),
                "last_action": int(last_action),
                "last_action_steps": json.dumps(last_action_steps, sort_keys=True),
                "dominant_action": dominant_action,
                "dominant_action_label": dominant_action_label,
                "dominant_action_steps": json.dumps(dominant_action_steps, sort_keys=True),
                "dominant_action_rate": dominant_action_rate,
                "action_counts": json.dumps(dict(sorted(action_counts.items())), sort_keys=True),
                "raw_top_action_counts": json.dumps(dict(sorted(raw_top_action_counts.items())), sort_keys=True),
                "sampled_not_raw_top_count": int(sampled_not_top_count),
                "sampled_not_raw_top_rate": float(sampled_not_top_count / decision_count),
                "raw_policy_top_action": dominant_raw_top_action,
                "raw_policy_top_action_label": (
                    self.firm1.action_label(dominant_raw_top_action)
                    if self.firm1_mode == "RL" and hasattr(self.firm1, "action_label")
                    else ""
                ),
                "raw_policy_top_action_rate": dominant_raw_top_action_rate,
                "raw_policy_hold_prob": float(avg_policy_diag.get("policy_hold_prob", 0.0)),
                "raw_policy_top_prob": float(avg_policy_diag.get("policy_top_prob", 0.0)),
                "raw_policy_action_margin": float(avg_policy_diag.get("policy_action_margin", 0.0)),
                "raw_policy_entropy": float(avg_policy_diag.get("policy_entropy", 0.0)),
                "recovery_guardrail_count": int(recovery_guardrail_count),
                "recovery_guardrail_recommend_count": int(recovery_guardrail_recommend_count),
                "recovery_guardrail_reasons": json.dumps(dict(sorted(recovery_guardrail_reasons.items())), sort_keys=True),
                "loss": float(ppo_metrics.get("loss", 0.0)),
                "ppo_policy_loss": float(ppo_metrics.get("policy_loss", 0.0)),
                "ppo_value_loss": float(ppo_metrics.get("value_loss", 0.0)),
                "ppo_critic_loss": float(ppo_metrics.get("critic_loss", 0.0)),
                "ppo_critic_grad_norm": float(ppo_metrics.get("critic_grad_norm", 0.0)),
                "ppo_constraint_value_loss": float(ppo_metrics.get("constraint_value_loss", 0.0)),
                "ppo_risk_value_loss": float(ppo_metrics.get("risk_value_loss", 0.0)),
                "ppo_action_q_loss": float(ppo_metrics.get("action_q_loss", 0.0)),
                "ppo_response_loss": float(ppo_metrics.get("response_loss", 0.0)),
                "ppo_lagrangian_adv_mean": float(ppo_metrics.get("lagrangian_adv_mean", 0.0)),
                "ppo_lagrangian_adv_std": float(ppo_metrics.get("lagrangian_adv_std", 0.0)),
                "ppo_constraint_lambda_mean": float(ppo_metrics.get("constraint_lambda_mean", 0.0)),
                "ppo_risk_coeff": float(ppo_metrics.get("risk_coeff", 0.0)),
                "ppo_approx_kl": float(ppo_metrics.get("approx_kl", 0.0)),
                "ppo_target_kl": float(
                    ppo_metrics.get("target_kl", self.firm1.agent.target_kl)
                ),
                "ppo_clipfrac": float(ppo_metrics.get("clipfrac", 0.0)),
                "ppo_entropy": float(ppo_metrics.get("entropy", 0.0)),
                "ppo_policy_entropy": float(ppo_metrics.get("policy_entropy", 0.0)),
                "ppo_discrete_policy_entropy": float(ppo_metrics.get("discrete_policy_entropy", np.nan)),
                "ppo_magnitude_entropy": float(ppo_metrics.get("magnitude_entropy", np.nan)),
                "ppo_policy_entropy_fraction": float(ppo_metrics.get("policy_entropy_fraction", 1.0)),
                "ppo_ent_coeff": float(ppo_metrics.get("ent_coeff", 0.0)),
                "ppo_clip_eps": float(ppo_metrics.get("clip_eps", 0.0)),
                "ppo_exploration_rate": float(ppo_metrics.get("exploration_rate", 0.0)),
                "ppo_action_coverage": float(ppo_metrics.get("action_coverage", 0.0)),
                "ppo_explained_variance": float(ppo_metrics.get("explained_variance", 0.0)),
                "ppo_optimizer_steps": int(ppo_metrics.get("optimizer_steps", 0)),
                "ppo_actor_optimizer_steps": int(ppo_metrics.get("actor_optimizer_steps", 0)),
                "ppo_critic_optimizer_steps": int(ppo_metrics.get("critic_optimizer_steps", 0)),
                "ppo_lr": float(ppo_metrics.get("lr", 0.0)),
                "ppo_critic_lr": float(ppo_metrics.get("critic_lr", 0.0)),
                "ppo_critic_target_mean": float(ppo_metrics.get("critic_target_mean", np.nan)),
                "ppo_critic_target_std": float(ppo_metrics.get("critic_target_std", np.nan)),
                "ppo_update_performed": bool(ppo_metrics.get("update_performed", False)),
                "ppo_run_mode": str(ppo_metrics.get("run_mode", "rollout_collection")),
                "ppo_learning_signal_ok": bool(ppo_metrics.get("learning_signal_ok", False)),
                "ppo_rollout_action_diversity": float(ppo_metrics.get("rollout_action_diversity", np.nan)),
                "ppo_rollout_reward_std": float(ppo_metrics.get("rollout_reward_std", np.nan)),
                "ppo_credited_reward_std": float(ppo_metrics.get("credited_reward_std", np.nan)),
                "ppo_continuous_magnitude_mean": float(ppo_metrics.get("continuous_magnitude_mean", np.nan)),
                "ppo_continuous_magnitude_std": float(ppo_metrics.get("continuous_magnitude_std", np.nan)),
                "ppo_raw_argmax_concentration": float(ppo_metrics.get("raw_policy_argmax_concentration", np.nan)),
                "ppo_raw_hold_probability": float(ppo_metrics.get("raw_policy_hold_probability", np.nan)),
                "ppo_state_action_sensitivity": float(ppo_metrics.get("state_action_sensitivity", np.nan)),
                "ppo_state_action_mi": float(ppo_metrics.get("state_action_mi", np.nan)),
                "ppo_rollout_size": int(ppo_metrics.get("rollout_size", 0)),
                "ppo_effective_batch_size": int(ppo_metrics.get("effective_batch_size", 0)),
                "ppo_stopped_early_kl": bool(ppo_metrics.get("stopped_early_kl", False)),
            }
            
            for bin_label in ("0_2", "2_5", "5_10", "10_plus"):
                for metric in ("firm1_choice_share", "firm1_completed_share", "firm1_revpr", "completed_rev", "price_gap_mean"):
                    stat_key = f"distance_bin_{bin_label}_{metric}"
                    log_row[stat_key] = float(getattr(self, "last_crowd_response_stats", {}).get(stat_key, np.nan))
            for k in self.shared_edit_keys:
                log_row[f"firm1_{k}"] = float(get_coeff(self.market.curr_market, self.firm1.overrides, k))
                log_row[f"firm2_{k}"] = float(get_coeff(self.market.curr_market, self.firm2.overrides, k))
            validation_score = np.nan
            if self.firm1_mode == "RL" and ((d + 1) % validation_interval == 0 or (d + 1) == train_timesteps):
                validation = self._adaptive_policy_validation(
                    customers_per_step=eval_customers_per_step,
                    profile_sample_size=eval_profile_sample_size,
                    horizon=192,
                )
                latest_validation = dict(validation)
                validation_score = float(validation["score"])
                log_row["validation_reward"] = float(validation["reward"])
                log_row["validation_score"] = float(validation_score)
                log_row["validation_share"] = float(validation["share"])
                log_row["validation_completed_share"] = float(validation["share"])
                log_row["validation_firm2_completed_share"] = float(validation["rival_share"])
                log_row["validation_rev_per_request"] = float(validation["rev_per_request"])
                log_row["validation_firm2_rev_per_request"] = float(validation["rival_rev_per_request"])
                log_row["validation_profit_per_request"] = float(validation["profit_per_request"])
                log_row["validation_firm2_profit_per_request"] = float(validation["rival_profit_per_request"])
                log_row["validation_fulfillment_rate"] = float(validation["fulfillment_rate"])
                log_row["validation_price_gap_f2_minus_f1"] = float(validation["price_gap"])
                log_row["validation_action_diversity"] = float(validation["action_diversity"])
                log_row["validation_action_switch_rate"] = float(validation["action_switch_rate"])
                log_row["validation_mean_profit_advantage"] = float(
                    validation["mean_profit_advantage"]
                )
                log_row["validation_worst_profit_advantage"] = float(
                    validation["worst_profit_advantage"]
                )
                log_row["validation_profit_win_rate"] = float(
                    validation["profit_win_rate"]
                )
                for mode, advantage in validation["mode_profit_advantages"].items():
                    log_row[f"validation_profit_advantage_{mode}"] = float(advantage)
                print(
                    f">>> [validation snapshot] day={d+1} score={validation_score:.3f} "
                    f"reward={validation['reward']:.3f} "
                    f"share={validation['share']:.3f} "
                    f"rivalShare={validation['rival_share']:.3f} "
                    f"revPR={validation['rev_per_request']:.2f} "
                    f"rivalRevPR={validation['rival_rev_per_request']:.2f} "
                    f"profitPR={validation['profit_per_request']:.2f} "
                    f"rivalProfitPR={validation['rival_profit_per_request']:.2f} "
                    f"profitAdv={validation['mean_profit_advantage']:+.2f} "
                    f"worstProfitAdv={validation['worst_profit_advantage']:+.2f} "
                    f"wins={validation['profit_win_rate']:.0%} "
                    f"actions={validation['action_diversity']:.2f} "
                    f"switch={validation['action_switch_rate']:.2f}"
                )
                mode_advantages = " | ".join(
                    f"{mode}:{advantage:+.2f}"
                    for mode, advantage in validation["mode_profit_advantages"].items()
                )
                if mode_advantages:
                    print(f"    [validation opponents profitAdv] {mode_advantages}")
                if float(validation_score) > float(best_validation_score):
                    best_validation_score = float(validation_score)
                    best_checkpoint = {
                        "day": int(d + 1),
                        "score": float(validation_score),
                        "agent_state": {
                            k: v.detach().cpu().clone()
                            for k, v in self.firm1.agent.net.state_dict().items()
                        },
                        "normalizer_state": copy.deepcopy(
                            self.firm1.agent.normalizer_state_dict()
                        ),
                        "validation": dict(validation),
                    }
                    print(
                        f">>> [validation checkpoint] new_best day={d+1} "
                        f"score={validation_score:.3f}"
                    )
            else:
                log_row["validation_score"] = np.nan
            self.training_logs.append(log_row)
            
            if self.firm1_mode == "RL":
                # Keep exploration/entropy controls synchronized in the primary
                # experiment workflow as well as in ``run``. Previously this
                # path stayed at the initial 40% uniform mixture for the entire
                # training run, preventing the learned policy from exploiting.
                (
                    convergence_count,
                    convergence_std,
                    convergence_delta,
                ) = self._smoothed_convergence_statistics(
                    reward_history,
                    window=40,
                    smoothing=20,
                )
                control_ppo_metrics = last_ppo_metrics
                convergence_now = bool(
                    convergence_count >= 40
                    and convergence_std <= self.reward_convergence_tol
                    and convergence_delta <= self.reward_trend_tol
                    and bool(control_ppo_metrics.get("learning_signal_ok", False))
                    and float(control_ppo_metrics.get("state_action_sensitivity", 0.0)) >= 0.005
                    and bool(latest_validation)
                    and float(latest_validation.get("profit_win_rate", 0.0)) >= 1.0
                    and float(
                        latest_validation.get(
                            "worst_profit_advantage", -float("inf")
                        )
                    ) > 0.0
                )
                training_convergence_latched = bool(
                    training_convergence_latched or convergence_now
                )
                if convergence_now:
                    latched_convergence_std = float(convergence_std)
                reward_converged = training_convergence_latched
                log_row["reward_converged"] = bool(reward_converged)
                log_row["reward_convergence_now"] = bool(convergence_now)
                self.firm1.configure_training_controls(
                    progress=float((d + 1) / max(1, train_timesteps)),
                    reward_converged=reward_converged,
                    reward_std=convergence_std,
                )
            
            if (d + 1) % max(1, train_timesteps // 10) == 0:
                progress_line = (
                    f"  [train {d+1}/{train_timesteps}] reward={float(avg_reward):.3f} "
                    f"moving_avg20={moving_avg20:.3f} share={avg_share:.3f} "
                    f"revPR={avg_revpr:.2f} profitPR={avg_profitpr:.2f} "
                    f"rivalProfitPR={avg_firm2_profitpr:.2f} "
                    f"profitAdv={avg_profitpr - avg_firm2_profitpr:+.2f} "
                    f"gap(F2-F1)={avg_gap:.2f} fulfill={avg_fulfillment:.3f} "
                    f"accept={avg_driver_accept:.3f}"
                )
                if bool(ppo_metrics.get("update_performed", False)):
                    progress_line += (
                        f" PPO_update steps={int(ppo_metrics.get('optimizer_steps', 0))} "
                        f"KL={float(ppo_metrics.get('approx_kl', np.nan)):.6g} "
                        f"clip={float(ppo_metrics.get('clipfrac', np.nan)):.4f}"
                    )
                else:
                    progress_line += f" PPO=collecting ({len(self.firm1.agent.buf)} transitions buffered)"
                print(progress_line)
                print(
                    f"    [train action summary] decisions={sum(action_counts.values())} "
                    f"dominant={dominant_action_label or 'none'} "
                    f"rate={dominant_action_rate:.1%} "
                    f"steps={json.dumps(dominant_action_steps, sort_keys=True)} "
                    f"last={last_action} "
                    f"lastSteps={json.dumps(last_action_steps, sort_keys=True)} "
                    f"sampledOffTop={sampled_not_top_count}"
                )
                
                segment_line = self._format_distance_segment_diagnostics(getattr(self, "last_crowd_response_stats", {}) or {})
                if segment_line:
                    print(f"    [train segments] {segment_line}")
        if self.firm1_mode == "RL" and best_checkpoint is not None:
            self.firm1.agent.net.load_state_dict(
                {k: v.to(self.firm1.agent.device) for k, v in best_checkpoint["agent_state"].items()}
            )
            self.firm1.agent.load_normalizer_state_dict(
                best_checkpoint["normalizer_state"]
            )
            print(
                f">>> Restored validation-best policy from train day {best_checkpoint['day']} "
                f"(score={best_checkpoint['score']:.3f}) for evaluation."
            )
        # Evaluation always starts from a fresh market/opponent state.  Only the
        # policy weights survive checkpoint selection.
        self.firm2 = copy.deepcopy(benchmark_opponent)
        self._reset_adaptive_episode(tariff_reset_fraction=0.18)

        if training_enabled and trained_model_out and self.firm1_mode == "RL":
            self.save_trained_model(
                trained_model_out,
                metadata={
                    "best_validation_day": None if best_checkpoint is None else int(best_checkpoint["day"]),
                    "best_validation_score": None if best_checkpoint is None else float(best_checkpoint["score"]),
                    "training_opponent": str(opponent_mode),
                    "training_days": int(train_timesteps),
                },
            )

        
        print(f">>> Evaluating RL agent against same-level {opponent_mode} opponent with shared profile pool...")
        if self.firm1_mode == "RL":
            self.firm1.reset_state_history()
        for observer in self.platform_observers.values():
            observer.reset()
        self.action_stability.reset()
        self.driver_reward_scale_current = 1.0 if self.enable_driver_supply else 0.0
        eval_rewards: List[float] = []
        # Reset reward-trend baselines so evaluation reward reflects evaluation dynamics,
        # not trailing deltas from the end of training.
        eval_last_share = float(self.last_share)
        eval_last_revpr = float(self.last_revpr)
        eval_last_profitpr = float(self.last_profitpr)
        eval_last_gap = float(self.last_gap)
        eval_action_interval_steps = int(base_action_interval_steps)
        eval_clock = OperationalClock()
        for t in range(eval_timesteps):
            day_ctx = self.market.sample_day_context()
            hour = self.market.sample_timestep_hour().hour
            sampled_profiles = self._sample_profiles_from_pool(eval_profile_sample_size)
            if profiles_out and not profile_limit_reached:
                remaining = int(max(0, profiles_log_limit - len(sampled_profile_rows)))
                if remaining > 0:
                    sampled_profile_rows.extend({"Phase": "eval", "Timestep": int(t), **p} for p in sampled_profiles[:remaining])
                profile_limit_reached = len(sampled_profile_rows) >= int(max(0, profiles_log_limit))

            base = self.market.curr_market
            action = -1
            eval_policy_diag: Dict[str, float] = {}
            coeff_pre_action = self._coeff_snapshot()
            coeff_post_action = dict(coeff_pre_action)
            coeff_post_projection = dict(coeff_pre_action)
            if self.firm1_mode == "RL":
                if (t % eval_action_interval_steps) == 0:
                    raw_state_vec, action_features, _ = self._publish_pricing_observation(
                        day_of_week=day_ctx.day_of_week, hour=hour, weather=day_ctx.weather
                    )
                    s_vec = self.firm1.stack_state(raw_state_vec)
                    action_mask = self.firm1.feasible_action_mask(self.market)
                    eval_policy_diag = self.firm1.agent.policy_diagnostics(
                        s_vec,
                        action_features=action_features,
                        action_mask=action_mask,
                        temperature=1.0,
                    )
                    action, *_ = self.firm1.agent.act(
                        s_vec,
                        deterministic=True,
                        action_features=action_features,
                        action_mask=action_mask,
                        policy_mode=eval_policy_mode,
                        policy_temperature=eval_policy_temperature,
                        top2_margin=eval_top2_margin,
                    )
                    coeff_pre_action = self._coeff_snapshot()
                    self.firm1.apply_action(action, self.market)
                    coeff_post_action = self._coeff_snapshot()
                    coeff_post_projection = dict(coeff_post_action)
                    action = int(action)
                else:
                    action = -1
            elif self.firm1_mode != "static":
                self.firm1.act(
                    city_base=base.base_fare,
                    city_pmin=base.per_minute,
                    city_pmile=base.per_mile,
                    city_booking=base.booking_fee,
                    city_airport=base.airport_fee,
                    hour=hour,
                    weather=day_ctx.weather,
                )

            # Evaluation timesteps are operational decision periods, not the
            # substeps of a synthetic training day.
            firm2_eval_interval_steps = max(1, firm2_interval_steps)
            firm2_eval_due = bool(
                not isinstance(self.firm2, FirmStaticPricer)
                and eval_clock.due(firm2_eval_interval_steps)
            )
            if firm2_eval_due:
                self.firm2.act(
                    city_base=base.base_fare,
                    city_pmin=base.per_minute,
                    city_pmile=base.per_mile,
                    city_booking=base.booking_fee,
                    city_airport=base.airport_fee,
                    hour=hour,
                    weather=day_ctx.weather,
                )

            _, m1, m2, mean_gap, _, _ = self.simulate_batch(
                day_of_week=day_ctx.day_of_week,
                weather=day_ctx.weather,
                hour=hour,
                customers_per_step=eval_customers_per_step,
                sampled_profiles=sampled_profiles,
                collect_rows=False,
            )
            self._ingest_platform_observations(m1, m2)
            if firm2_eval_due:
                f2_supply, f2_supply_vec = self._driver_supply_context("Firm2", "Firm1")
                self.firm2.update(
                    metrics=m2,
                    price_gap_mean=self._observed_own_minus_competitor_gap("Firm2"),
                    supply_state=f2_supply,
                    supply_state_vector=f2_supply_vec,
                )
            eval_clock.advance()
            
            action_movement = 0.0
            if self.firm1_mode == "RL" and action >= 0:
                action_movement = float(self.firm1.last_action_magnitude())
                
            # Use the same shaped reward family as training for consistent trajectory logs.
            # Compute with local baselines to avoid mutating training history state.
            share = float(np.clip(m1.chosen_share, 0.0, 1.0))
            revpr = float(max(0.0, m1.rev_per_request))
            reward_diag = self._reward_diagnostics(
                share=share,
                completed_share=float(m1.completed_share),
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
                action_change_magnitude=action_movement,
                baseline_share=float(m2.chosen_share),
                baseline_completed_share=float(m2.completed_share),
                baseline_rev_per_request=float(m2.rev_per_request),
                baseline_profit_per_request=float(m2.profit_per_request),
                action_event=action >= 0,
            )
            eval_reward = float(reward_diag["reward"])
            guardrail_diag: Dict[str, Any] = {"applied": False, "recommended": False, "reasons": [], "deltas": {}}
            
            if self.firm1_mode == "RL":
                guardrail_diag = self._apply_eval_guardrail(
                    eval_guardrail_mode,
                    metrics=m1,
                    price_gap_f2_minus_f1=float(mean_gap),
                    base=base,
                )
                coeff_post_projection = self._coeff_snapshot()

            eval_last_share = float(share)
            eval_last_revpr = float(revpr)
            eval_last_profitpr = float(m1.profit_per_request)
            eval_last_gap = float(mean_gap)
            f2_supply_diag, _ = self._driver_supply_context("Firm2", "Firm1")
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
                "reward_own_profit_utility": float(reward_diag["reward_own_profit_utility"]),
                "reward_profit_advantage_utility": float(reward_diag["reward_profit_advantage_utility"]),
                "reward_own_profit_component": float(reward_diag["reward_own_profit_component"]),
                "reward_profit_advantage_component": float(reward_diag["reward_profit_advantage_component"]),
                "reward_intervention_cost": float(reward_diag["reward_intervention_cost"]),
                "reward_reversal_cost": float(reward_diag["reward_reversal_cost"]),
                "reward_intervention_magnitude": float(reward_diag["reward_intervention_magnitude"]),
                "reward_profit_advantage_per_request": float(reward_diag["reward_profit_advantage_per_request"]),
                "reward_share_component": float(reward_diag.get("reward_share_component", 0.0)),
                "reward_profit_component": float(reward_diag.get("reward_profit_component", 0.0)),
                "reward_revenue_component": float(reward_diag.get("reward_revenue_component", 0.0)),
                "reward_unit_economics_component": float(reward_diag.get("reward_unit_economics_component", 0.0)),
                "reward_price_gap_component": float(reward_diag.get("reward_price_gap_component", 0.0)),
                "reward_corrective_action_bonus": float(reward_diag.get("reward_corrective_action_bonus", 0.0)),
                "reward_hold_inaction_penalty": float(reward_diag.get("reward_hold_inaction_penalty", 0.0)),
                "reward_baseline_loss_penalty": float(reward_diag.get("reward_baseline_loss_penalty", 0.0)),
                "reward_overprice_penalty": float(reward_diag.get("reward_overprice_penalty", 0.0)),
                "reward_underprice_penalty": float(reward_diag.get("reward_underprice_penalty", 0.0)),
                "reward_action_change_penalty": float(reward_diag.get("reward_action_change_penalty", 0.0)),
                "reward_action_realization_penalty": float(reward_diag.get("reward_action_realization_penalty", 0.0)),
                "reward_correction_pressure": float(reward_diag.get("reward_correction_pressure", 0.0)),
                "reward_revenue_improvement": float(reward_diag.get("reward_revenue_improvement", 0.0)),
                "reward_share_improvement": float(reward_diag.get("reward_share_improvement", 0.0)),
                "reward_profit_improvement": float(reward_diag.get("reward_profit_improvement", 0.0)),
                "reward_demand_loss": float(reward_diag.get("reward_demand_loss", 0.0)),
                "reward_baseline_share": float(reward_diag.get("reward_baseline_share", 0.0)),
                "reward_choice_share": float(reward_diag.get("reward_choice_share", 0.0)),
                "reward_completed_share": float(reward_diag.get("reward_completed_share", 0.0)),
                "reward_baseline_choice_share": float(reward_diag.get("reward_baseline_choice_share", 0.0)),
                "reward_baseline_completed_share": float(reward_diag.get("reward_baseline_completed_share", 0.0)),
                "reward_baseline_rev_per_request": float(reward_diag.get("reward_baseline_rev_per_request", 0.0)),
                "reward_baseline_profit_per_request": float(reward_diag.get("reward_baseline_profit_per_request", 0.0)),
                "reward_absolute_profit_term": float(reward_diag.get("reward_absolute_profit_term", 0.0)),
                "reward_relative_profit_term": float(reward_diag.get("reward_relative_profit_term", 0.0)),
                "reward_absolute_revenue_term": float(reward_diag.get("reward_absolute_revenue_term", 0.0)),
                "reward_relative_revenue_term": float(reward_diag.get("reward_relative_revenue_term", 0.0)),
                "reward_relative_share_term": float(reward_diag.get("reward_relative_share_term", 0.0)),
                "reward_price_gap_abs_error": float(reward_diag.get("reward_price_gap_abs_error", 0.0)),
                "reward_service_component": float(reward_diag.get("reward_service_component", 0.0)),
                "reward_service_quality": float(reward_diag.get("reward_service_quality", 0.0)),
                "reward_wait_satisfaction": float(reward_diag.get("reward_wait_satisfaction", 0.0)),
                "reward_driver_acceptance_objective": float(reward_diag.get("reward_driver_acceptance_objective", 0.0)),
                "reward_price_gap_satisfaction": float(reward_diag.get("reward_price_gap_satisfaction", 0.0)),
                "reward_price_gap_in_range": float(reward_diag.get("reward_price_gap_in_range", 0.0)),
                "reward_gap_tolerance": float(reward_diag.get("reward_gap_tolerance", 0.0)),
                "reward_zero_effect_action": float(reward_diag.get("reward_zero_effect_action", 0.0)),
                "reward_saturated_action": float(reward_diag.get("reward_saturated_action", 0.0)),
                "constraint_violation_share_floor": float(reward_diag.get("constraint_violation_share_floor", 0.0)),
                "constraint_violation_fulfillment_floor": float(
                    reward_diag.get("constraint_violation_fulfillment_floor", 0.0)
                ),
                "constraint_violation_wait_limit": float(reward_diag.get("constraint_violation_wait_limit", 0.0)),
                "constraint_violation_gap_band": float(reward_diag.get("constraint_violation_gap_band", 0.0)),
                "constraint_violation_margin_floor": float(reward_diag.get("constraint_violation_margin_floor", 0.0)),
                "rl_share": float(m1.share),
                "rl_chosen_share": float(m1.chosen_share),
                "rl_completed_share": float(m1.completed_share),
                "heuristic_share": float(m2.share),
                "heuristic_chosen_share": float(m2.chosen_share),
                "heuristic_completed_share": float(m2.completed_share),
                "rl_revenue": float(m1.rev_per_request),
                "heuristic_revenue": float(m2.rev_per_request),
                "rl_profit": float(m1.profit_per_request),
                "heuristic_profit": float(m2.profit_per_request),
                "rl_fulfillment_rate": float(m1.fulfillment_rate),
                "rl_avg_wait_minutes": float(m1.avg_wait_minutes),
                "rl_driver_acceptance_rate": float(m1.driver_acceptance_rate),
                "rl_driver_pay_per_request": float(m1.driver_pay / max(1, m1.total)),
                "heuristic_fulfillment_rate": float(m2.fulfillment_rate),
                "heuristic_avg_wait_minutes": float(m2.avg_wait_minutes),
                "heuristic_driver_acceptance_rate": float(m2.driver_acceptance_rate),
                "heuristic_driver_pay_per_request": float(m2.driver_pay / max(1, m2.total)),
                "heuristic_supply_stress": float(getattr(self.firm2, "last_supply_stress", 0.0)),
                "heuristic_demand_shortfall": float(getattr(self.firm2, "last_demand_shortfall", 0.0)),
                "heuristic_supply_idle_share": float(f2_supply_diag.get("idle_driver_share", 0.0)),
                "heuristic_supply_utilization": float(f2_supply_diag.get("utilization", 0.0)),
                "heuristic_supply_driver_earnings_per_hour": float(f2_supply_diag.get("driver_earnings_per_hour", 0.0)),
                "driver_reward_scale": float(self.driver_reward_scale_current),
                "price_gap_f2_minus_f1": float(mean_gap),
                "price_gap_abs_error": float(abs(float(mean_gap) - self.reward_target_price_gap)),
                "gap_violation_025": float(abs(float(mean_gap) - self.reward_target_price_gap) > 0.25),
                "gap_violation_050": float(abs(float(mean_gap) - self.reward_target_price_gap) > 0.50),
                "constraint_curriculum_scale": float(self.constraint_curriculum_scale),
                "eval_policy_mode": str(eval_policy_mode),
                "eval_guardrail_mode": str(eval_guardrail_mode),
                "policy_top_action": int(eval_policy_diag.get("policy_top_action", -1)),
                "policy_top_action_label": (
                    self.firm1.action_label(int(eval_policy_diag.get("policy_top_action", -1)))
                    if self.firm1_mode == "RL" and hasattr(self.firm1, "action_label") and int(eval_policy_diag.get("policy_top_action", -1)) >= 0
                    else ""
                ),
                "policy_second_action": int(eval_policy_diag.get("policy_second_action", -1)),
                "policy_top_prob": float(eval_policy_diag.get("policy_top_prob", 0.0)),
                "policy_second_prob": float(eval_policy_diag.get("policy_second_prob", 0.0)),
                "policy_action_margin": float(eval_policy_diag.get("policy_action_margin", 0.0)),
                "policy_hold_prob": float(eval_policy_diag.get("policy_hold_prob", 0.0)),
                "policy_entropy": float(eval_policy_diag.get("policy_entropy", 0.0)),
                "policy_selected_is_top": float(action == int(eval_policy_diag.get("policy_top_action", -999))),
                "guardrail_applied": bool(guardrail_diag.get("applied", False)),
                "guardrail_recommended": bool(guardrail_diag.get("recommended", False)),
                "guardrail_reasons": json.dumps(list(guardrail_diag.get("reasons", []))),
                "guardrail_deltas": json.dumps(guardrail_diag.get("deltas", {}), sort_keys=True),
                "coeff_delta_rl_action": json.dumps(self._coeff_delta(coeff_post_action, coeff_pre_action), sort_keys=True),
                "coeff_delta_projection": json.dumps(self._coeff_delta(coeff_post_projection, coeff_post_action), sort_keys=True),
                "action_zero_effect": bool(getattr(self.firm1, "last_action_was_zero_effect", False)) if self.firm1_mode == "RL" else False,
                "action_saturated": bool(getattr(self.firm1, "last_action_was_saturated", False)) if self.firm1_mode == "RL" else False,
                "action": int(action),
                "action_label": (
                    self.firm1.action_label(action)
                    if self.firm1_mode == "RL" and hasattr(self.firm1, "action_label") and int(action) >= 0
                    else ("no_decision" if self.firm1_mode == "RL" else "")
                ),
                "action_steps": json.dumps(
                    self.firm1.action_steps(action)
                    if self.firm1_mode == "RL" and hasattr(self.firm1, "action_steps")
                    else {},
                    sort_keys=True,
                ),
            }
            for bin_label in ("0_2", "2_5", "5_10", "10_plus"):
                for metric in ("firm1_choice_share", "firm1_completed_share", "firm1_revpr", "completed_rev", "price_gap_mean"):
                    stat_key = f"distance_bin_{bin_label}_{metric}"
                    log_row[stat_key] = float(getattr(self, "last_crowd_response_stats", {}).get(stat_key, np.nan))
            for k in self.shared_edit_keys:
                log_row[f"firm1_{k}"] = float(get_coeff(self.market.curr_market, self.firm1.overrides, k))
                log_row[f"firm2_{k}"] = float(get_coeff(self.market.curr_market, self.firm2.overrides, k))
            for diag_key, diag_value in reward_diag.items():
                if diag_key.startswith(("reward_positive_", "constraint_cost_", "constraint_lambda_")):
                    log_row[diag_key] = float(diag_value)
            self.evaluation_logs.append(log_row)
            self._update_recent_response_emas(m1, mean_gap, m2=m2)
            self.last_share = float(m1.share)
            self.last_revpr = float(m1.rev_per_request)
            self.last_gap = float(mean_gap)
            self.last_profitpr = float(m1.profit_per_request)
            self.last_fulfillment = float(m1.fulfillment_rate)
            self.last_acceptance = float(m1.driver_acceptance_rate)
            self.last_wait = float(m1.avg_wait_minutes)
            self.last_driver_paypr = float(m1.driver_pay / max(1, m1.total))
            self.last_firm2_share = float(m2.share)
            self.last_firm2_revpr = float(m2.rev_per_request)
            self.last_firm2_profitpr = float(m2.profit_per_request)
            self.last_firm2_fulfillment = float(m2.fulfillment_rate)
            self.last_firm2_acceptance = float(m2.driver_acceptance_rate)
            self.last_firm2_wait = float(m2.avg_wait_minutes)
            self.last_firm2_driver_paypr = float(m2.driver_pay / max(1, m2.total))

            if (t + 1) % max(1, eval_timesteps // 10) == 0:
                print(
                    f"  [eval {t+1}/{eval_timesteps}] reward={float(eval_reward):.3f} "
                    f"moving_avg20={moving_avg20:.3f} "
                    f"share={float(m1.completed_share):.3f} "
                    f"rivalShare={float(m2.completed_share):.3f} "
                    f"revPR={float(m1.rev_per_request):.2f} "
                    f"rivalRevPR={float(m2.rev_per_request):.2f} "
                    f"profitPR={float(m1.profit_per_request):.2f} "
                    f"rivalProfitPR={float(m2.profit_per_request):.2f} "
                    f"profitAdv={float(m1.profit_per_request - m2.profit_per_request):+.2f} "
                    f"gap(F2-F1)={float(mean_gap):.2f} "
                    f"action={log_row['action_label'] or 'none'}"
                )
                segment_line = self._format_distance_segment_diagnostics(getattr(self, "last_crowd_response_stats", {}) or {})
                if segment_line:
                    print(f"    [eval segments] {segment_line}")

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
        if self.training_logs:
            training_tail = self.training_logs[-min(100, len(self.training_logs)):]
            print(
                f">>> Training competitive summary (last {len(training_tail)} days): "
                f"share={float(np.mean([row['avg_completed_share'] for row in training_tail])):.3f} "
                f"rivalShare={float(np.mean([row['avg_firm2_completed_share'] for row in training_tail])):.3f} "
                f"revPR={float(np.mean([row['avg_rev_per_request'] for row in training_tail])):.2f} "
                f"rivalRevPR={float(np.mean([row['avg_firm2_rev_per_request'] for row in training_tail])):.2f} "
                f"profitPR={float(np.mean([row['avg_profit_per_request'] for row in training_tail])):.2f} "
                f"rivalProfitPR={float(np.mean([row['avg_firm2_profit_per_request'] for row in training_tail])):.2f} "
                f"profitAdv={float(np.mean([row['avg_profit_advantage_per_request'] for row in training_tail])):+.2f}"
            )
        if self.evaluation_logs:
            print(
                f">>> Evaluation competitive summary ({len(self.evaluation_logs)} days): "
                f"share={float(np.mean([row['rl_completed_share'] for row in self.evaluation_logs])):.3f} "
                f"rivalShare={float(np.mean([row['heuristic_completed_share'] for row in self.evaluation_logs])):.3f} "
                f"revPR={float(np.mean([row['rl_revenue'] for row in self.evaluation_logs])):.2f} "
                f"rivalRevPR={float(np.mean([row['heuristic_revenue'] for row in self.evaluation_logs])):.2f} "
                f"profitPR={float(np.mean([row['rl_profit'] for row in self.evaluation_logs])):.2f} "
                f"rivalProfitPR={float(np.mean([row['heuristic_profit'] for row in self.evaluation_logs])):.2f} "
                f"profitAdv={float(np.mean([row['rl_profit'] - row['heuristic_profit'] for row in self.evaluation_logs])):+.2f}"
            )
            
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
        response_baseline = np.zeros(12, dtype=np.float32)
        if self.firm1_mode == "RL":
            raw_state, action_features, _ = self._publish_pricing_observation(
                day_of_week=day_ctx.day_of_week,
                hour=hour,
                weather=day_ctx.weather,
            )
            s_vec = self.firm1.stack_state(raw_state)
            response_baseline = self._response_target_from_metrics(
                FirmMetrics(), FirmMetrics(), 0.0
            )
            action_mask = self.firm1.feasible_action_mask(self.market)
            action, s_ts, old_logp, val, af_ts, mask_ts = self.firm1.agent.act(
                s_vec, action_features=action_features, action_mask=action_mask
            )
            self.firm1.apply_action(action, self.market)
            rl_step = (action, s_ts, old_logp, val, af_ts, mask_ts)
        elif self.firm1_mode != "static":
            self.firm1.act(
                city_base=base.base_fare,
                city_pmin=base.per_minute,
                city_pmile=base.per_mile,
                city_booking=base.booking_fee,
                city_airport=base.airport_fee,
                hour=hour,
                weather=day_ctx.weather,
            )

        if self.firm2_mode != "static":
            self.firm2.act(
                city_base=base.base_fare,
                city_pmin=base.per_minute,
                city_pmile=base.per_mile,
                city_booking=base.booking_fee,
                city_airport=base.airport_fee,
                hour=hour,
                weather=day_ctx.weather,
            )

        results, m1, m2, gap, air, dist = self.simulate_batch(day_ctx.day_of_week, day_ctx.weather, hour, rides)
        self._ingest_platform_observations(m1, m2)
        
        reward = self._compute_rl_reward(m1, gap, m2=m2)


        if is_training and self.firm1_mode == "RL" and rl_step is not None:
            action, s_ts, old_logp, val, af_ts, mask_ts = rl_step
            reward_diag = getattr(self, "last_reward_diagnostics", {})
            self.firm1.agent.store(
                s_ts,
                action,
                float(reward),
                False,
                None,
                old_logp,
                val,
                constraint_costs=self._constraint_vector_from_diag(reward_diag),
                risk_cost=self._risk_cost_from_diag(reward_diag),
                response_target=self._response_delta(
                    response_baseline,
                    self._response_target_from_metrics(m1, m2, gap),
                ),
                action_features=af_ts,
                action_mask=mask_ts,
            )
            self._apply_eval_guardrail(
                "log_only",
                metrics=m1,
                price_gap_f2_minus_f1=float(gap),
                base=self.market.curr_market,
            )
        
        self._update_recent_response_emas(m1, gap, m2=m2)
        self.last_share = float(m1.share)
        self.last_revpr = float(m1.rev_per_request)
        self.last_gap = float(gap)
        self.last_profitpr = float(m1.profit_per_request)
        self.last_fulfillment = float(m1.fulfillment_rate)
        self.last_acceptance = float(m1.driver_acceptance_rate)
        self.last_wait = float(m1.avg_wait_minutes)
        self.last_driver_paypr = float(m1.driver_pay / max(1, m1.total))
        self.last_firm2_share = float(m2.share)
        self.last_firm2_revpr = float(m2.rev_per_request)
        self.last_firm2_profitpr = float(m2.profit_per_request)
        self.last_firm2_fulfillment = float(m2.fulfillment_rate)
        self.last_firm2_acceptance = float(m2.driver_acceptance_rate)
        self.last_firm2_wait = float(m2.avg_wait_minutes)
        self.last_firm2_driver_paypr = float(m2.driver_pay / max(1, m2.total))
        
        self.airport_rate_last = air
        self.mean_distance_last = dist

        return results, m1, m2, float(reward)
    
    
    def _update_recent_response_emas(self, m1: FirmMetrics, mean_gap: float, alpha: float = 0.20, m2: Optional[FirmMetrics] = None) -> None:
        """Maintain short-term EMA features for both firms' observations."""
        self.ema_share = self._ema(float(m1.share), float(getattr(self, "ema_share", 0.5)), alpha)
        self.ema_revpr = self._ema(float(m1.rev_per_request), float(getattr(self, "ema_revpr", 0.0)), alpha)
        self.ema_profitpr = self._ema(float(m1.profit_per_request), float(getattr(self, "ema_profitpr", 0.0)), alpha)
        self.ema_gap = self._ema(float(mean_gap), float(getattr(self, "ema_gap", 0.0)), alpha)
        self.ema_fulfillment = self._ema(float(m1.fulfillment_rate), float(getattr(self, "ema_fulfillment", 1.0)), alpha)
        if m2 is not None:
            self.ema_firm2_share = self._ema(float(m2.share), float(getattr(self, "ema_firm2_share", 0.5)), alpha)
            self.ema_firm2_revpr = self._ema(float(m2.rev_per_request), float(getattr(self, "ema_firm2_revpr", 0.0)), alpha)
            self.ema_firm2_profitpr = self._ema(float(m2.profit_per_request), float(getattr(self, "ema_firm2_profitpr", 0.0)), alpha)
            self.ema_firm2_gap = self._ema(float(-mean_gap), float(getattr(self, "ema_firm2_gap", 0.0)), alpha)
            self.ema_firm2_fulfillment = self._ema(float(m2.fulfillment_rate), float(getattr(self, "ema_firm2_fulfillment", 1.0)), alpha)

    @staticmethod
    def _bounded_context(value: float, center: float, scale: float) -> float:
        """Normalize a public/calibrated market descriptor without city identity."""
        return float(np.clip((float(value) - float(center)) / max(1e-6, float(scale)), -1.5, 1.5))

    def _build_city_context(self, *, hour: int, day_of_week: int, weather: str) -> Dict[str, float]:
        """Build continuous market context shared across city twins.

        These descriptors are either supplied by digital-twin calibration or
        estimated from previously realized aggregate demand.  No city name,
        opponent label, controller parameters, or future simulator state is
        exposed to the policy.
        """
        base = self.market.curr_market
        demand = dict(getattr(self, "last_crowd_response_stats", {}) or {})

        time_multiplier = float(self.market._time_multiplier(int(hour)))
        day_multiplier = float(base.day_multiplier.get(int(day_of_week), 1.0))
        weather_multiplier = float(base.weather_multiplier.get(str(weather), 1.0))
        demand_multiplier = time_multiplier * day_multiplier * weather_multiplier

        distance_mean = float(demand.get("distance_mean", 6.0) or 6.0)
        duration_mean = float(demand.get("duration_mean", 24.0) or 24.0)
        airport_intensity = float(getattr(self.market, "airport_prob", 0.12))

        income_probs = np.asarray(
            getattr(self.agent_gen, "income_probs", [0.25, 0.25, 0.25, 0.25]),
            dtype=np.float64,
        ).reshape(-1)
        income_scores = np.linspace(0.0, 1.0, max(1, income_probs.size))
        income_total = float(income_probs.sum())
        income_index = float(
            np.dot(income_probs / income_total, income_scores)
            if income_total > 0.0
            else 0.5
        )

        choice_price_scale = float(getattr(self.choice_model, "price_sensitivity_scale", 1.0))
        choice_loyalty_scale = float(getattr(self.choice_model, "loyalty_scale", 1.0))
        loyalty_mean = 0.5 * (
            float(getattr(self.agent_gen, "loy_lo", 0.4))
            + float(getattr(self.agent_gen, "loy_hi", 1.0))
        )
        new_rider_share = float(getattr(self.agent_gen, "p_new", 0.30))

        supply_config = getattr(self.driver_supply, "config", None)
        base_active_drivers = float(getattr(supply_config, "base_active_drivers", 260.0))
        weather_probs = dict(getattr(self.market, "weather_probs", {}) or {})
        expected_weather_multiplier = sum(
            float(probability) * float(base.weather_multiplier.get(name, 1.0))
            for name, probability in weather_probs.items()
        )
        if not weather_probs:
            expected_weather_multiplier = 1.0

        return {
            "demand_intensity": self._bounded_context(demand_multiplier, 1.15, 0.45),
            "trip_distance_scale": self._bounded_context(distance_mean, 6.0, 4.0),
            "trip_duration_scale": self._bounded_context(duration_mean, 24.0, 16.0),
            "airport_intensity": self._bounded_context(airport_intensity, 0.12, 0.10),
            "income_index": self._bounded_context(income_index, 0.50, 0.30),
            "price_elasticity": self._bounded_context(choice_price_scale, 1.0, 0.60),
            "loyalty_intensity": self._bounded_context(
                loyalty_mean * choice_loyalty_scale, 0.70, 0.40
            ),
            "new_rider_share": self._bounded_context(new_rider_share, 0.30, 0.20),
            "driver_cost_mile": self._bounded_context(self.driver_cost_per_mile, 0.85, 0.40),
            "driver_cost_minute": self._bounded_context(self.driver_cost_per_minute, 0.12, 0.08),
            "supply_intensity": self._bounded_context(base_active_drivers, 260.0, 120.0),
            "weather_sensitivity": self._bounded_context(
                expected_weather_multiplier, 1.08, 0.16
            ),
        }
    
    def _build_rl_state(self, day_of_week: int, hour: int, weather: str, perspective: str = "Firm1") -> np.ndarray:
        """Build the platform's operational observation, not simulator truth.

        A firm knows its own active tariff and delayed internal KPIs.  Rival
        prices are represented only by sampled public-quote estimates already
        held by the observer.  Rider thresholds, rival coefficients, rival
        controller state, and rival profit are intentionally unavailable.
        """
        firm_name = "Firm2" if str(perspective) == "Firm2" else "Firm1"
        own = self.firm2 if firm_name == "Firm2" else self.firm1
        base = self.market.curr_market
        anchors = {key: float(getattr(base, key)) for key in self.opt_keys}
        coefficients = {
            key: float(get_coeff(base, own.overrides, key)) for key in self.opt_keys
        }
        last_action = self.action_stability.features(self.shared_edit_keys)
        if firm_name == "Firm2":
            last_action = {
                "direction": 0.0,
                "target_index": 0.0,
                "magnitude": 0.0,
                "reversal": 0.0,
                "recent_intervention_rate": 0.0,
                "time_since_intervention": 1.0,
            }
        return self.platform_observers[firm_name].build_observation(
            hour=hour,
            day_of_week=day_of_week,
            weather=weather,
            own_coefficients=coefficients,
            anchor_coefficients=anchors,
            last_action=last_action,
            city_context=self._build_city_context(
                hour=hour,
                day_of_week=day_of_week,
                weather=weather,
            ),
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
        threshold_values: List[float] = []
        distance_values: List[float] = []
        duration_values: List[float] = []
        low_income_count = 0
        near_threshold_count = 0
        firm1_gap_below_threshold_count = 0
        no_ride_count = 0
        long_trip_count = 0
        premium_count = 0
        distance_bin_counts: Dict[str, int] = defaultdict(int)
        distance_bin_firm1_choices: Dict[str, int] = defaultdict(int)
        distance_bin_firm1_completed: Dict[str, int] = defaultdict(int)
        distance_bin_gap_sum: Dict[str, float] = defaultdict(float)
        distance_bin_firm1_revenue_sum: Dict[str, float] = defaultdict(float)
        
        profile_sample_size = self._effective_simulation_sample_size(customers_per_step, collect_rows=collect_rows)
        if sampled_profiles is None:
            profiles = self.agent_gen.sample_profiles(profile_sample_size)
        elif collect_rows or len(sampled_profiles) <= profile_sample_size:
            profiles = sampled_profiles
        else:
            profiles = sampled_profiles[:profile_sample_size]
        if self.enable_driver_supply:
            self._sync_driver_incentive_multipliers()
            self.driver_supply.begin_batch(customers_per_step=customers_per_step, hour=hour, weather=weather)

        for profile in profiles:
            profile_count += 1
            # trip-specific distance (scenario-side)
            travel_distance = round(float(self.rng.exponential(4.0)), 2)
            
            if travel_distance < 2.0:
                distance_bin = "0_2"
            elif travel_distance < 5.0:
                distance_bin = "2_5"
            elif travel_distance < 10.0:
                distance_bin = "5_10"
            else:
                distance_bin = "10_plus"
            distance_bin_counts[distance_bin] += 1

            airport = self.market.sample_airport_flag()
            service = self.market.sample_service()
            airport_count += int(airport)
            dist_sum += float(travel_distance)
            distance_values.append(float(travel_distance))

            duration = self.estimate_duration(travel_distance, hour)
            duration_values.append(float(duration))

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
            
            threshold = float(np.clip(float(profile.get("PriceThreshold", 1.50) or 1.50), 0.25, 8.00))
            threshold_values.append(threshold)
            low_income_count += int(str(profile.get("IncomeBracket", "")).strip() == "<50k")
            near_threshold_count += int(0.75 <= abs(float(p2 - p1)) / max(threshold, 1e-6) <= 1.25)
            long_trip_count += int(float(travel_distance) >= 8.0)
            firm1_gap_below_threshold_count += int(abs(float(p2 - p1)) <= threshold)
            premium_count += int(str(service).lower() in {"premium", "xl", "black"})

            choice_res: ChoiceResult = self.choice_model.choose(profile, scenario, p1, p2)
            choice = choice_res.choice
            no_ride_count += int(choice == "NoRide")

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
                    dispatch = self.driver_supply.dispatch("Firm1", p1, travel_distance, duration, airport=airport, weather=weather)
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
                    dispatch = self.driver_supply.dispatch("Firm2", p2, travel_distance, duration, airport=airport, weather=weather)
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
            distance_bin_gap_sum[distance_bin] += float(p2 - p1)
            if choice == "Firm1":
                distance_bin_firm1_choices[distance_bin] += 1
                if fulfilled:
                    distance_bin_firm1_completed[distance_bin] += 1
                    distance_bin_firm1_revenue_sum[distance_bin] += float(p1)
            
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
        threshold_arr = np.asarray(threshold_values or [1.5], dtype=float)
        distance_arr = np.asarray(distance_values or [self.mean_distance_last], dtype=float)
        duration_arr = np.asarray(duration_values or [0.0], dtype=float)
        distance_bin_stats: Dict[str, float] = {}
        for label in ("0_2", "2_5", "5_10", "10_plus"):
            count = max(1, int(distance_bin_counts.get(label, 0)))
            completed = max(1, int(distance_bin_firm1_completed.get(label, 0)))
            distance_bin_stats[f"distance_bin_{label}_count"] = float(distance_bin_counts.get(label, 0))
            distance_bin_stats[f"distance_bin_{label}_firm1_choice_share"] = float(distance_bin_firm1_choices.get(label, 0) / count)
            distance_bin_stats[f"distance_bin_{label}_firm1_completed_share"] = float(distance_bin_firm1_completed.get(label, 0) / count)
            distance_bin_stats[f"distance_bin_{label}_firm1_revpr"] = float(distance_bin_firm1_revenue_sum.get(label, 0.0) / count)
            distance_bin_stats[f"distance_bin_{label}_completed_rev"] = float(distance_bin_firm1_revenue_sum.get(label, 0.0) / completed)
            distance_bin_stats[f"distance_bin_{label}_price_gap_mean"] = float(distance_bin_gap_sum.get(label, 0.0) / count)
            
        self.last_crowd_response_stats = {
            "price_threshold_mean": float(np.mean(threshold_arr)),
            "price_threshold_std": float(np.std(threshold_arr)),
            "price_threshold_q25": float(np.quantile(threshold_arr, 0.25)),
            "price_threshold_q50": float(np.quantile(threshold_arr, 0.50)),
            "price_threshold_q75": float(np.quantile(threshold_arr, 0.75)),
            "firm1_gap_below_threshold_share": float(firm1_gap_below_threshold_count / denominator),
            "firm1_price_below_wtp_share": float(firm1_gap_below_threshold_count / denominator),
            "near_threshold_share": float(near_threshold_count / denominator),
            **distance_bin_stats,
            "low_income_share": float(low_income_count / denominator),
            "airport_rate": float(airport_count / denominator),
            "long_trip_share": float(long_trip_count / denominator),
            "premium_share": float(premium_count / denominator),
            "distance_mean": float(np.mean(distance_arr)),
            "distance_std": float(np.std(distance_arr)),
            "distance_q25": float(np.quantile(distance_arr, 0.25)),
            "distance_q50": float(np.quantile(distance_arr, 0.50)),
            "distance_q75": float(np.quantile(distance_arr, 0.75)),
            "duration_mean": float(np.mean(duration_arr)),
            "duration_std": float(np.std(duration_arr)),
            "peak_context": float((7 <= int(hour) < 10) or (16 <= int(hour) < 19)),
            "night_context": float(int(hour) < 6 or int(hour) >= 22),
            "weekend_context": float(int(day_of_week) >= 5),
            "no_ride_rate": float(no_ride_count / denominator),
        }
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
        if self.firm1_mode == "RL":
            self.firm1.reset_state_history()
        self._refresh_profile_pool(rides_per_timestep=customers_per_step)
        self.convergence_day = None
        self.convergence_window_std_at_day = None
        self.convergence_delta_per_day_at_day = None
        self._convergence_streak = 0
        self.driver_reward_scale_current = 1.0 if self.enable_driver_supply else 0.0
        
        for d in range(days):
            train_progress = float(d + 1) / max(1.0, float(days))
            self._configure_constraint_curriculum(train_progress)
            day_ctx = self.market.sample_day_context()
            hours = [self.market.sample_timestep_hour().hour for _ in range(timesteps_per_day)]

            # day accumulators for logging
            share_sum = 0.0
            choice_share_sum = 0.0
            completed_share_sum = 0.0
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
            choice_share_sum_two = 0.0
            completed_share_sum_two = 0.0
            revpr_sum_two = 0.0
            profit_sum_two = 0.0
            no_ride_share_sum = 0.0

            for t in range(timesteps_per_day):
                hour = hours[t]
                base = self.market.curr_market

                # Firm 1 action
                rl_step = None
                if self.firm1_mode == "RL":
                    raw_state_vec, action_features, _ = self._publish_pricing_observation(
                        day_of_week=day_ctx.day_of_week, hour=hour, weather=day_ctx.weather
                    )
                    s_vec = self.firm1.stack_state(raw_state_vec)
                    # Evaluation/deployment runs reflect the learned policy, not
                    # PPO's training-time uniform exploration mixture. Use
                    # --run_experiment for optimizer updates and rollout training.
                    action_mask = self.firm1.feasible_action_mask(self.market)
                    action, s_ts, old_logp, val, af_ts, mask_ts = self.firm1.agent.act(
                        s_vec,
                        deterministic=True,
                        action_features=action_features,
                        action_mask=action_mask,
                    )
                    self.firm1.apply_action(action, self.market)
                    action_counts[int(action)] += 1
                    last_action = int(action)
                    rl_step = (action, s_ts, old_logp, val, af_ts, mask_ts)
                elif self.firm1_mode != "static":
                    self.firm1.act(
                        city_base=base.base_fare,
                        city_pmin=base.per_minute,
                        city_pmile=base.per_mile,
                        city_booking=base.booking_fee,
                        city_airport=base.airport_fee,
                        hour=hour,
                        weather=day_ctx.weather,
                    )

                # Firm 2 action
                if self.firm2_mode != "static":
                    self.firm2.act(
                        city_base=base.base_fare,
                        city_pmin=base.per_minute,
                        city_pmile=base.per_mile,
                        city_booking=base.booking_fee,
                        city_airport=base.airport_fee,
                        hour=hour,
                        weather=day_ctx.weather,
                    )
                
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
                self._ingest_platform_observations(m1, m2)
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
                    self.firm1.update(
                        metrics=m1,
                        price_gap_mean=self._observed_own_minus_competitor_gap("Firm1"),
                    )
                if self.firm2_mode != "static":
                    f2_supply, f2_supply_vec = self._driver_supply_context("Firm2", "Firm1")
                    self.firm2.update(
                        metrics=m2,
                        price_gap_mean=self._observed_own_minus_competitor_gap("Firm2"),
                        supply_state=f2_supply,
                        supply_state_vector=f2_supply_vec,
                    )
                    
                # RL memory + reward shaping
                action_movement = 0.0
                if self.firm1_mode == "RL" and rl_step is not None:
                    action_movement = float(self.firm1.last_action_magnitude())
                reward_diag = self._reward_diagnostics(
                    share=float(m1.chosen_share),
                    completed_share=float(m1.completed_share),
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
                    action_change_magnitude=action_movement,
                    baseline_share=float(m2.chosen_share),
                    baseline_completed_share=float(m2.completed_share),
                    baseline_rev_per_request=float(m2.rev_per_request),
                    baseline_profit_per_request=float(m2.profit_per_request),
                    action_event=rl_step is not None,
                )
                if self.firm1_mode == "RL" and rl_step is not None:
                    action, s_ts, old_logp, val, af_ts, mask_ts = rl_step
                    reward = float(reward_diag["reward"])
                    self.last_reward = reward
                    self._update_constraint_multipliers(reward_diag)
                    done = (t == timesteps_per_day - 1)
                    # Do not append evaluation transitions to the PPO rollout buffer;
                    # optimizer updates are training-only, and stale eval samples can
                    # contaminate a later training continuation.
                    if done:
                        self._apply_eval_guardrail(
                            "deployed",
                            metrics=m1,
                            price_gap_f2_minus_f1=float(mean_gap),
                            base=base,
                        )
                    reward_sum += float(reward)
                for component_key in (
                    "reward_share_component",
                    "reward_profit_component",
                    "reward_relative_share_term",
                    "reward_relative_profit_term",
                    "reward_own_profit_utility",
                    "reward_profit_advantage_utility",
                    "reward_own_profit_component",
                    "reward_profit_advantage_component",
                    "reward_intervention_cost",
                    "reward_reversal_cost",
                    "reward_intervention_magnitude",
                    "reward_profit_advantage_per_request",
                    "reward_momentum_component",
                    "reward_hold_inaction_penalty",
                    "reward_baseline_loss_penalty",
                    "reward_overprice_penalty",
                    "reward_underprice_penalty",
                    "reward_action_change_penalty",
                    "reward_action_realization_penalty",
                    "reward_correction_pressure",
                    "reward_profit_delta",
                    "reward_gap_delta",
                    "reward_price_gap_deviation",
                    "reward_price_gap_satisfaction",
                    "reward_service_component",
                    "reward_service_quality",
                    "reward_wait_satisfaction",
                    "reward_driver_acceptance_objective",
                    "reward_price_gap_in_range",
                    "reward_gap_tolerance",
                    "reward_price_gap_abs_error",
                    "reward_served_share",
                    "constraint_violation_share_floor",
                    "constraint_violation_fulfillment_floor",
                    "constraint_violation_wait_limit",
                    "constraint_violation_gap_band",
                    "constraint_violation_margin_floor",
                    "reward_zero_effect_action",
                    "reward_saturated_action",
                ):
                    reward_component_sums[component_key] += float(reward_diag.get(component_key, 0.0))

                share_sum += float(m1.share)
                choice_share_sum += float(m1.chosen_share)
                completed_share_sum += float(m1.completed_share)
                revpr_sum += float(m1.rev_per_request)
                profit_sum += float(m1.profit_per_request)
                gap_sum += float(mean_gap)
                fulfillment_sum += float(m1.fulfillment_rate)
                wait_sum += float(m1.avg_wait_minutes)
                driver_accept_sum += float(m1.driver_acceptance_rate)
                driver_paypr_sum += float(m1.driver_pay / max(1, m1.total))
                
                share_sum_two += float(m2.share)
                choice_share_sum_two += float(m2.chosen_share)
                completed_share_sum_two += float(m2.completed_share)
                revpr_sum_two += float(m2.rev_per_request)
                profit_sum_two += float(m2.profit_per_request)
                no_ride_share_sum += float(max(0.0, 1.0 - float(m1.share) - float(m2.share)))
                
                self._update_recent_response_emas(m1, mean_gap, m2=m2)
                self.last_share = float(m1.share)
                self.last_revpr = float(m1.rev_per_request)
                self.last_gap = float(mean_gap)
                self.last_profitpr = float(m1.profit_per_request)
                self.last_fulfillment = float(m1.fulfillment_rate)
                self.last_acceptance = float(m1.driver_acceptance_rate)
                self.last_wait = float(m1.avg_wait_minutes)
                self.last_driver_paypr = float(m1.driver_pay / max(1, m1.total))
                self.last_firm2_share = float(m2.share)
                self.last_firm2_revpr = float(m2.rev_per_request)
                self.last_firm2_profitpr = float(m2.profit_per_request)
                self.last_firm2_fulfillment = float(m2.fulfillment_rate)
                self.last_firm2_acceptance = float(m2.driver_acceptance_rate)
                self.last_firm2_wait = float(m2.avg_wait_minutes)
                self.last_firm2_driver_paypr = float(m2.driver_pay / max(1, m2.total))
                self.airport_rate_last = airport_rate
                self.mean_distance_last = mean_dist
            
            ppo_metrics = {
                key: np.nan
                for key in (
                    "loss", "approx_kl", "clipfrac", "entropy", "policy_entropy",
                    "discrete_policy_entropy", "magnitude_entropy", "explained_variance",
                    "constraint_value_loss", "risk_value_loss", "response_loss",
                    "lagrangian_adv_mean", "lagrangian_adv_std", "constraint_lambda_mean",
                    "risk_coeff",
                )
            }
            if self.firm1_mode == "RL":
                self._sync_agent_optimization_context()
                ppo_metrics.update({
                    "ent_coeff": float(self.firm1.agent.ent_coeff),
                    "clip_eps": float(self.firm1.agent.clip_eps),
                    "lr": float(self.firm1.agent.curr_lr),
                    "exploration_rate": 0.0,
                    "action_coverage": float(self.firm1.agent._action_coverage()),
                    "policy_entropy_fraction": float(self.firm1.agent.last_policy_entropy_fraction),
                    "optimizer_steps": 0,
                    "update_performed": False,
                    "run_mode": "evaluation_no_update",
                })
                
            #print("firm 1 revenue per request sum", str(revpr_sum))
            #print("firm 1 market share sum", str(share_sum))
            
            #print("firm 2 revenue per request sum", str(revpr_sum_two))
            #print("firm 2 market share sum", str(share_sum_two))

            avg_share = share_sum / max(1, timesteps_per_day)
            avg_choice_share = choice_share_sum / max(1, timesteps_per_day)
            avg_completed_share = completed_share_sum / max(1, timesteps_per_day)
            avg_revpr = revpr_sum / max(1, timesteps_per_day)
            avg_gap = gap_sum / max(1, timesteps_per_day)
            avg_profitpr = profit_sum / max(1, timesteps_per_day)
            avg_no_ride_share = no_ride_share_sum / max(1, timesteps_per_day)
            avg_fulfillment = fulfillment_sum / max(1, timesteps_per_day)
            avg_wait = wait_sum / max(1, timesteps_per_day)
            avg_driver_accept = driver_accept_sum / max(1, timesteps_per_day)
            avg_driver_paypr = driver_paypr_sum / max(1, timesteps_per_day)
            avg_share_two = share_sum_two / max(1, timesteps_per_day)
            avg_choice_share_two = choice_share_sum_two / max(1, timesteps_per_day)
            avg_completed_share_two = completed_share_sum_two / max(1, timesteps_per_day)
            dominant_action = int(action_counts.most_common(1)[0][0]) if action_counts else -1
            dominant_action_rate = (
                float(action_counts.most_common(1)[0][1] / max(1, timesteps_per_day))
                if action_counts
                else 0.0
            )
            dominant_action_steps = (
                self.firm1.action_steps(dominant_action)
                if self.firm1_mode == "RL" and hasattr(self.firm1, "action_steps")
                else {}
            )
            dominant_action_label = (
                self.firm1.action_label(dominant_action)
                if self.firm1_mode == "RL" and hasattr(self.firm1, "action_label")
                else ""
            )
            last_action_steps = (
                self.firm1.action_steps(last_action)
                if self.firm1_mode == "RL" and hasattr(self.firm1, "action_steps")
                else {}
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
                "avg_choice_share": float(avg_choice_share),
                "avg_completed_share": float(avg_completed_share),
                "avg_share_firm2": float(avg_share_two),
                "avg_choice_share_firm2": float(avg_choice_share_two),
                "avg_completed_share_firm2": float(avg_completed_share_two),
                "avg_revpr": float(avg_revpr),
                "avg_profitpr": float(avg_profitpr),
                "avg_gap": float(avg_gap),
                "avg_gap_abs_error": float(abs(avg_gap - self.reward_target_price_gap)),
                "avg_gap_violation_025": float(abs(avg_gap - self.reward_target_price_gap) > 0.25),
                "avg_gap_violation_050": float(abs(avg_gap - self.reward_target_price_gap) > 0.50),
                "constraint_curriculum_scale": float(self.constraint_curriculum_scale),
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
                "last_action_steps": json.dumps(last_action_steps, sort_keys=True),
                "dominant_action": dominant_action,
                "dominant_action_label": dominant_action_label,
                "dominant_action_steps": json.dumps(dominant_action_steps, sort_keys=True),
                "dominant_action_rate": dominant_action_rate,
                "action_counts": json.dumps(dict(sorted(action_counts.items())), sort_keys=True),
                "ppo_approx_kl": float(ppo_metrics.get("approx_kl", 0.0)),
                "ppo_clipfrac": float(ppo_metrics.get("clipfrac", 0.0)),
                "ppo_entropy": float(ppo_metrics.get("entropy", 0.0)),
                "ppo_policy_entropy": float(ppo_metrics.get("policy_entropy", 0.0)),
                "ppo_policy_entropy_fraction": float(ppo_metrics.get("policy_entropy_fraction", 1.0)),
                "ppo_ent_coeff": float(ppo_metrics.get("ent_coeff", 0.0)),
                "ppo_clip_eps": float(ppo_metrics.get("clip_eps", 0.0)),
                "ppo_exploration_rate": float(ppo_metrics.get("exploration_rate", 0.0)),
                "ppo_action_coverage": float(ppo_metrics.get("action_coverage", 0.0)),
                "ppo_explained_variance": float(ppo_metrics.get("explained_variance", 0.0)),
                "ppo_constraint_value_loss": float(ppo_metrics.get("constraint_value_loss", 0.0)),
                "ppo_risk_value_loss": float(ppo_metrics.get("risk_value_loss", 0.0)),
                "ppo_response_loss": float(ppo_metrics.get("response_loss", 0.0)),
                "ppo_lagrangian_adv_mean": float(ppo_metrics.get("lagrangian_adv_mean", 0.0)),
                "ppo_lagrangian_adv_std": float(ppo_metrics.get("lagrangian_adv_std", 0.0)),
                "ppo_constraint_lambda_mean": float(ppo_metrics.get("constraint_lambda_mean", 0.0)),
                "ppo_risk_coeff": float(ppo_metrics.get("risk_coeff", 0.0)),
                "ppo_optimizer_steps": int(ppo_metrics.get("optimizer_steps", 0)),
                "ppo_update_performed": bool(ppo_metrics.get("update_performed", False)),
                "ppo_run_mode": str(ppo_metrics.get("run_mode", "training_update")),
            })
            for k in self.shared_edit_keys:
                self.run_logs[-1][f"firm1_{k}"] = float(get_coeff(base, self.firm1.overrides, k))
                self.run_logs[-1][f"firm2_{k}"] = float(get_coeff(base, self.firm2.overrides, k))
            
            (
                convergence_count,
                reward_std,
                reward_delta,
            ) = self._smoothed_convergence_statistics(
                [float(x["avg_reward"]) for x in self.run_logs],
                window=self.reward_convergence_window,
                smoothing=20,
            )
            reward_converged = (
                convergence_count >= self.reward_convergence_window
                and (d + 1) >= self.convergence_min_days
                and reward_std <= self.reward_convergence_tol
                and reward_delta <= self.reward_trend_tol
                and bool(ppo_metrics.get("learning_signal_ok", True))
                and float(ppo_metrics.get("state_action_sensitivity", 0.005)) >= 0.005
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
                        f"  [PPO evaluation] optimizer=disabled "
                        f"last_train_policy_ent={float(ppo_metrics.get('policy_entropy_fraction', 1.0)):.3f} "
                        f"ent_coeff={float(ppo_metrics.get('ent_coeff', 0.0)):.4f} "
                        f"coverage={float(ppo_metrics.get('action_coverage', 0.0)):.2f} "
                        f"lr={float(ppo_metrics.get('lr', 0.0)):.6f}"
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
            summary_path = f"{prefix}_price_gap_threshold_distribution.csv"
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
            ax.set_title("Profile Fare-Gap Threshold Distribution")
            ax.set_xlabel("Fare-gap salience threshold ($)")
            ax.set_ylabel("% of profiles within source")
            ax.legend(loc="best")
            fig.tight_layout()
            out = f"{prefix}_dist_PriceGapThreshold.png"
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

        # Directly comparable two-firm business outcomes.  The original
        # dashboard showed only Firm 1 share/revenue plus a fare gap, which
        # could make a low-price, low-profit policy look successful.  Preserve
        # that diagnostic below and add the actual competitive objective.
        if x_key == "batch":
            competitive_keys = (
                ("avg_completed_share", "avg_firm2_completed_share"),
                ("avg_rev_per_request", "avg_firm2_rev_per_request"),
                ("avg_profit_per_request", "avg_firm2_profit_per_request"),
            )
        else:
            competitive_keys = (
                ("rl_completed_share", "heuristic_completed_share"),
                ("rl_revenue", "heuristic_revenue"),
                ("rl_profit", "heuristic_profit"),
            )
        if all(
            own_key in logs[0] and rival_key in logs[0]
            for own_key, rival_key in competitive_keys
        ):
            fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
            ylabels = ("Completed market share", "Revenue / request", "Profit / request")
            for ax, (own_key, rival_key), ylabel in zip(
                axes, competitive_keys, ylabels
            ):
                own = [float(r.get(own_key, np.nan)) for r in logs]
                rival = [float(r.get(rival_key, np.nan)) for r in logs]
                _plot_timeseries(
                    ax, xs, own, label="Firm 1 raw", color="tab:blue",
                    alpha=0.20, linewidth=0.8,
                )
                _plot_timeseries(
                    ax, xs, rival, label="Firm 2 raw", color="tab:orange",
                    alpha=0.20, linewidth=0.8,
                )
                _plot_timeseries(
                    ax, xs, _moving_average(own, 20),
                    label="Firm 1 MA20", color="tab:blue",
                    marker="", linewidth=2.0,
                )
                _plot_timeseries(
                    ax, xs, _moving_average(rival, 20),
                    label="Firm 2 MA20", color="tab:orange",
                    marker="", linewidth=2.0,
                )
                ax.set_ylabel(ylabel)
                ax.grid(axis="y", linestyle="--", alpha=0.25)
                ax.legend(loc="best", ncol=2, fontsize=8)
            axes[0].set_ylim(-0.02, 1.02)
            axes[2].axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
            axes[2].set_xlabel("Training day" if x_key == "batch" else "Evaluation day")
            fig.suptitle(f"{label.title()} Two-Firm Competitive Performance")
            fig.tight_layout()
            out = f"{prefix}_{label}_competitive_performance.png"
            _ensure_parent_dir(out)
            fig.savefig(out, dpi=150)
            print(f"Saved graph -> {out}")
            plt.close(fig)

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

        # PPO metrics exist only when an optimizer update is performed. Plotting
        # them on every simulator day used to join sparse update values through
        # zero/NaN placeholders and mixed quantities with incompatible scales.
        # That produced the misleading vertical spikes in the old dashboard.
        ppo_logs = [
            r for r in logs
            if bool(r.get("ppo_update_performed", False))
            and np.isfinite(float(r.get("ppo_approx_kl", np.nan)))
        ]
        if ppo_logs:
            ppo_xs = [
                int(r[x_key]) + (1 if x_key == "batch" else 0)
                for r in ppo_logs
            ]
            fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)

            trust_ax = axes[0]
            trust_twin = trust_ax.twinx()
            _plot_timeseries(
                trust_ax,
                ppo_xs,
                [float(r.get("ppo_approx_kl", np.nan)) for r in ppo_logs],
                label="approx KL",
                color="tab:blue",
            )
            target_kl = np.asarray(
                [float(r.get("ppo_target_kl", np.nan)) for r in ppo_logs],
                dtype=float,
            )
            if np.isfinite(target_kl).any():
                _plot_timeseries(
                    trust_ax,
                    ppo_xs,
                    target_kl,
                    label="target KL",
                    color="tab:blue",
                    marker="",
                    linestyle="--",
                    alpha=0.65,
                )
            _plot_timeseries(
                trust_twin,
                ppo_xs,
                [float(r.get("ppo_clipfrac", np.nan)) for r in ppo_logs],
                label="clip fraction",
                color="tab:orange",
            )
            trust_ax.set_ylabel("KL divergence")
            trust_twin.set_ylabel("Clip fraction")
            trust_twin.set_ylim(bottom=0.0)
            trust_ax.set_title("Trust-region behavior")

            exploration_ax = axes[1]
            exploration_twin = exploration_ax.twinx()
            for key, display, color in (
                ("ppo_policy_entropy_fraction", "policy entropy fraction", "tab:red"),
                ("ppo_exploration_rate", "exploration rate", "tab:brown"),
                ("ppo_action_coverage", "action coverage", "tab:pink"),
            ):
                _plot_timeseries(
                    exploration_ax,
                    ppo_xs,
                    [float(r.get(key, np.nan)) for r in ppo_logs],
                    label=display,
                    color=color,
                )
            _plot_timeseries(
                exploration_twin,
                ppo_xs,
                [float(r.get("ppo_ent_coeff", np.nan)) for r in ppo_logs],
                label="entropy coefficient",
                color="tab:purple",
            )
            exploration_ax.set_ylabel("Fraction / rate")
            exploration_ax.set_ylim(-0.02, 1.02)
            exploration_twin.set_ylabel("Entropy coefficient")
            exploration_ax.set_title("Exploration and policy concentration")

            critic_ax = axes[2]
            critic_twin = critic_ax.twinx()
            _plot_timeseries(
                critic_ax,
                ppo_xs,
                [float(r.get("ppo_explained_variance", np.nan)) for r in ppo_logs],
                label="explained variance",
                color="tab:gray",
            )
            for key, display, color in (
                ("ppo_critic_loss", "critic loss", "tab:green"),
                ("ppo_value_loss", "value loss", "tab:olive"),
            ):
                _plot_timeseries(
                    critic_twin,
                    ppo_xs,
                    [float(r.get(key, np.nan)) for r in ppo_logs],
                    label=display,
                    color=color,
                )
            critic_ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.35)
            critic_ax.set_ylim(-1.05, 1.05)
            critic_ax.set_ylabel("Explained variance")
            critic_twin.set_ylabel("Loss")
            critic_twin.set_yscale("symlog", linthresh=1e-3)
            critic_ax.set_title("Critic fit")

            credit_ax = axes[3]
            credit_twin = credit_ax.twinx()
            _plot_timeseries(
                credit_ax,
                ppo_xs,
                [float(r.get("ppo_rollout_action_diversity", np.nan)) for r in ppo_logs],
                label="rollout action diversity",
                color="tab:cyan",
            )
            _plot_timeseries(
                credit_ax,
                ppo_xs,
                [
                    1.0 - float(r.get("ppo_raw_argmax_concentration", np.nan))
                    for r in ppo_logs
                ],
                label="policy adaptive mass",
                color="tab:blue",
            )
            for key, display, color in (
                ("ppo_state_action_sensitivity", "state-action sensitivity", "tab:orange"),
                ("ppo_state_action_mi", "state-action mutual information", "tab:green"),
            ):
                _plot_timeseries(
                    credit_twin,
                    ppo_xs,
                    [float(r.get(key, np.nan)) for r in ppo_logs],
                    label=display,
                    color=color,
                )
            credit_ax.set_ylim(-0.02, 1.02)
            credit_ax.set_ylabel("Diversity / adaptive mass")
            credit_twin.set_ylabel("State dependence")
            credit_ax.set_title("Action credit and state dependence")
            credit_ax.set_xlabel("Training day" if x_key == "batch" else "Day")

            twins = {
                trust_ax: trust_twin,
                exploration_ax: exploration_twin,
                critic_ax: critic_twin,
                credit_ax: credit_twin,
            }
            for ax in axes:
                ax.grid(axis="y", linestyle="--", alpha=0.25)
                twin = twins[ax]
                handles, labels_left = ax.get_legend_handles_labels()
                twin_handles, labels_right = twin.get_legend_handles_labels()
                handles += twin_handles
                labels_left += labels_right
                if handles:
                    ax.legend(handles, labels_left, loc="best", fontsize=8)

            fig.suptitle(
                f"{label.title()} PPO Update Diagnostics "
                f"({len(ppo_logs)} optimizer updates)"
            )
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
            
def _write_run_config(prefix: str, args: argparse.Namespace, core: Core) -> None:
    """Persist the exact run configuration for reproducible experiments."""
    out = f"{prefix}_run_config.json"
    payload = {
        "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "argv": sys.argv,
        "user_seed": args.seed,
        "experiment_seed": args.experiment_seed,
        "effective_seed": core.seed,
        "deterministic_experiment_seed": bool(args.deterministic_experiment_seed),
        "eval_policy_mode": args.eval_policy_mode,
        "eval_guardrail_mode": args.eval_guardrail_mode,
        "report_prefix": args.report_prefix,
        "platform_mdp": platform_mdp_config_payload(
            core.observation_config,
            core.positive_reward_config,
            core.constraint_config,
            core.training_stage_scheduler,
            core.long_term_profit_reward_config,
        ),
        "active_reward": {
            "type": "discounted_long_term_contribution_profit",
            **vars(core.long_term_profit_reward_config),
            "gamma": float(core.firm1.agent.gamma) if core.firm1_mode == "RL" else None,
            "gae_lambda": float(core.firm1.agent.lam) if core.firm1_mode == "RL" else None,
        },
        "args": {
            k: (str(v) if not isinstance(v, (str, int, float, bool, type(None), list, dict)) else v)
            for k, v in vars(args).items()
        },
    }
    _ensure_parent_dir(out)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"Saved -> {out}")

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

    dynamic_mode_choices = [
        "heuristic",
        "heuristic_margin",
        "heuristic_random",
        "adaptive_best_response",
        "adaptive_best_response_aggressive",
        "pi_price_gap",
        "region_supply_demand",
        "queue_service_threshold",
        "surge_driver_incentive",
        "mpc_grid",
    ]
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
    parser.add_argument("--trained_model_in", type=str, default="", help="Load a portable Firm1 policy artifact before evaluation or continued training.")
    parser.add_argument("--trained_model_out", type=str, default="", help="Save the validation-best Firm1 policy artifact before evaluation.")
    parser.add_argument("--eval_only", action="store_true", help="Skip training and evaluate --trained_model_in directly against --firm2_mode.")
    parser.add_argument("--train_timesteps", type=int, default=1500)
    parser.add_argument("--train_customers", type=int, default=5000)
    parser.add_argument("--train_steps_per_day", type=int, default=10, help="Synthetic training timesteps per day (run_experiment mode).")
    parser.add_argument("--ppo_update_interval_days", type=int, default=20, help="How many synthetic training days to collect before each PPO optimizer update; larger values produce longer, lower-variance PPO rollouts.")
    parser.add_argument("--ppo_min_rollout_transitions", type=int, default=256, help="Minimum complete on-policy pricing decisions required before PPO may update; stage boundaries truncate GAE but do not trigger undersized optimizer steps.")
    
    parser.add_argument("--firm1_action_interval_steps", type=int, default=-1, help="Training/eval timesteps to hold each Firm1 PPO price action; <=0 means one pricing decision every market timestep for both static and dynamic opponents.")
    parser.add_argument("--firm2_action_interval_days", type=int, default=-1, help="Legacy option name: operational decision periods between Firm2 manipulations/updates; <=0 derives cadence from the PPO update interval.")
    parser.add_argument("--ppo_batch_size", type=int, default=256, help="PPO minibatch size used when optimizing each rollout buffer.")
    parser.add_argument("--ppo_update_epochs", type=int, default=8, help="Number of optimization epochs per PPO rollout buffer.")
    parser.add_argument("--state_frame_stack", type=int, default=4, help="Number of recent platform observation frames concatenated for history-aware PPO.")
    parser.add_argument("--training_curriculum", type=str, default="staged", choices=["staged", "direct"], help="Use staged opponent/cadence/optimization training or train directly in full competition.")
    parser.add_argument("--deterministic_experiment_seed", action="store_true", help="If set, keep run_experiment fully deterministic with --seed.")
    parser.add_argument("--experiment_seed", type=int, default=None, help="Optional explicit run_experiment seed. Overrides stochastic child-seed generation and is recorded in run config.")
    parser.add_argument("--eval_policy_mode", type=str, default="argmax", choices=["argmax", "sample_raw", "sample_low_temp", "top2_margin"], help="Evaluation action-selection mode for the learned policy.")
    parser.add_argument("--eval_policy_temperature", type=float, default=0.50, help="Temperature for --eval_policy_mode sample_low_temp.")
    parser.add_argument("--eval_top2_margin", type=float, default=0.05, help="Top-probability margin required before deterministic top action is used in --eval_policy_mode top2_margin.")
    parser.add_argument("--eval_guardrail_mode", type=str, default="log_only", choices=["deployed", "off", "log_only"], help="Whether evaluation applies, disables, or only logs the post-batch business guardrail.")
    parser.add_argument("--eval_timesteps", type=int, default=200)
    parser.add_argument("--eval_customers", type=int, default=1000)
    parser.add_argument("--profiles_out", type=str, default="artifacts/sampled_profiles.csv")
    parser.add_argument("--profiles_log_limit", type=int, default=200000)
    parser.add_argument("--positive_reward_profit_weight", type=float, default=0.38, help="Active positive-reward weight for profit per incoming request. All four active weights are normalized to sum to one.")
    parser.add_argument("--positive_reward_revenue_weight", type=float, default=0.22, help="Active positive-reward weight for revenue per incoming request.")
    parser.add_argument("--positive_reward_completed_demand_weight", type=float, default=0.20, help="Active positive-reward weight for completed demand share.")
    parser.add_argument("--positive_reward_service_weight", type=float, default=0.20, help="Active positive-reward weight for fulfillment, driver acceptance, and wait quality.")
    parser.add_argument("--long_term_profit_scale", type=float, default=4.0, help="Scale inside asinh(own contribution profit per incoming request / scale).")
    parser.add_argument("--long_term_profit_advantage_scale", type=float, default=2.5, help="Scale inside tanh((own-rival contribution profit per incoming request) / scale).")
    parser.add_argument("--long_term_profit_weight", type=float, default=1.0, help="Primary weight on own long-run contribution-profit utility.")
    parser.add_argument("--profit_dominance_weight", type=float, default=0.10, help="Small profitability-gated tie-breaker for outperforming the rival.")
    parser.add_argument("--dominance_quality_scale", type=float, default=4.0, help="Own-profit scale controlling when the rival-relative tie-breaker becomes economically meaningful.")
    parser.add_argument("--intervention_cost_weight", type=float, default=0.0, help="Optional cost on realized tariff movement in standard step units; zero leaves economically useful reactions unpenalized.")
    parser.add_argument("--reversal_cost_weight", type=float, default=0.0, help="Optional cost on reversing the same tariff target within four decisions; zero avoids inducing a fixed tariff.")
    parser.add_argument("--ppo_gamma", type=float, default=0.995, help="Operational-timestep discount used by semi-MDP decision returns.")
    parser.add_argument("--ppo_gae_lambda", type=float, default=0.97, help="GAE lambda applied once per pricing decision with gamma^duration discount.")
    parser.add_argument("--reward_share_weight", type=float, default=0.40, help="Deprecated legacy diagnostic weight; does not enter the active scalar reward.")
    parser.add_argument("--reward_revenue_weight", type=float, default=0.35, help="Deprecated legacy diagnostic weight; use --positive_reward_revenue_weight.")
    parser.add_argument("--reward_profit_weight", type=float, default=None, help="Deprecated legacy diagnostic weight; use --positive_reward_profit_weight.")
    parser.add_argument("--reward_price_gap_weight", type=float, default=None, help="Deprecated legacy diagnostic weight; price gap is an adaptive soft constraint.")
    parser.add_argument("--reward_hold_inaction_weight", type=float, default=0.06, help="Deprecated compatibility option; hold has no active reward penalty.")
    parser.add_argument("--reward_corrective_action_weight", type=float, default=0.08, help="Deprecated compatibility option; corrective actions receive no active scalar-reward bonus.")
    parser.add_argument("--reward_baseline_loss_weight", type=float, default=0.12, help="Deprecated compatibility option; competitor-relative losses do not enter active reward.")
    parser.add_argument("--reward_overprice_weight", type=float, default=0.20, help="Deprecated compatibility option; overpricing is handled by a soft constraint.")
    parser.add_argument("--reward_rev_scale", type=float, default=25.0, help="Revenue/request e-folding scale for positive saturation; a value equal to the scale receives about 63.2%% component credit.")
    parser.add_argument("--reward_competitive_weight", type=float, default=0.15, help="Deprecated legacy diagnostic weight; competitor dominance does not enter active reward.")
    parser.add_argument("--reward_trend_weight", type=float, default=0.0, help="Deprecated legacy diagnostic weight; short-term deltas do not enter active reward.")
    parser.add_argument("--reward_profit_scale", type=float, default=12.0, help="Profit/request e-folding scale for positive saturation; a value equal to the scale receives about 63.2%% component credit.")
    parser.add_argument("--reward_underprice_weight", type=float, default=0.15, help="Deprecated compatibility option; underpricing and margin are handled by soft constraints.")
    parser.add_argument("--reward_acceptable_discount", type=float, default=2.0, help="Deprecated compatibility value used only to derive target_price_gap when that explicit target is omitted.")
    parser.add_argument("--min_profit_margin", type=float, default=0.08, help="Minimum contribution margin used by the adaptive margin soft constraint.")
    parser.add_argument("--target_price_gap", type=float, default=None, help="Target public quote gap F2-F1 used by soft constraints; defaults to half reward_acceptable_discount for compatibility.")
    parser.add_argument("--price_gap_tolerance", type=float, default=0.45, help="Aggregate two-sided soft-constraint tolerance around target_price_gap.")
    parser.add_argument("--segment_gap_tolerances", type=float, nargs=4, default=[0.65, 0.60, 0.55, 0.55], metavar=("SHORT", "MEDIUM", "LONG", "XLONG"), help="Soft gap tolerances for 0-2, 2-5, 5-10, and 10+ mile quote probes; long trips default tighter because per-mile errors compound.")
    parser.add_argument("--telemetry_delay_steps", type=int, default=1, help="Delay applied to own operational KPI observations.")
    parser.add_argument("--quote_probe_interval_steps", type=int, default=3, help="Operational periods between public competitor quote samples.")
    parser.add_argument("--quote_probe_delay_steps", type=int, default=1, help="Delay before sampled competitor quotes reach the policy.")
    parser.add_argument("--quote_noise_dollars", type=float, default=0.18, help="Standard deviation of public competitor quote measurement noise.")
    parser.add_argument("--quote_missing_probability", type=float, default=0.03, help="Probability that a scheduled competitor quote probe is unavailable.")
    parser.add_argument("--driver_cost_per_mile", type=float, default=0.85, help="Approximate per-mile contribution cost for completed rides.")
    parser.add_argument("--reward_action_change_weight", type=float, default=0.0, help="Deprecated legacy-diagnostic option; it does not enter the active long-term-profit reward.")
    parser.add_argument("--driver_cost_per_minute", type=float, default=0.12, help="Approximate per-minute contribution cost for completed rides.")
    parser.add_argument("--fixed_trip_cost", type=float, default=1.25, help="Approximate fixed platform/driver cost for completed rides.")
    parser.add_argument("--airport_cost", type=float, default=2.0, help="Approximate extra cost for airport rides.")
    parser.add_argument("--disable_driver_supply", action="store_true", help="Disable the driver supply/matching layer and use legacy completed-ride costs.")
    parser.add_argument("--use_osmnx", action="store_true", help="Load an OpenStreetMap drive network through OSMnx for future spatial routing hooks.")
    parser.add_argument("--osmnx_place", type=str, default="", help="OSMnx place string; defaults to '<market>, USA'.")
    parser.add_argument("--driver_base_active", type=int, default=260, help="Baseline active drivers for the batch-level driver supply model.")
    parser.add_argument("--driver_reservation_wage", type=float, default=24.0, help="Driver reservation wage used by acceptance and online-supply response.")
    parser.add_argument("--driver_acceptance_mode", type=str, default="expected", choices=["expected", "stochastic"], help="Use deterministic expected driver acceptance for faster RL convergence, or stochastic Bernoulli acceptance for realism.")
    parser.add_argument("--driver_expected_acceptance_cutoff", type=float, default=0.65, help="Minimum expected driver acceptance probability required to count an offered ride as accepted in expected driver acceptance mode.")
    parser.add_argument("--driver_state_smoothing", type=float, default=0.35, help="EMA smoothing alpha for driver features fed into the RL state.")
    parser.add_argument("--driver_reward_fulfillment_weight", type=float, default=0.15, help="Service-quality reward weight for fulfilled rides when driver supply is enabled.")
    parser.add_argument("--driver_reward_wait_weight", type=float, default=0.0, help="Legacy option retained for compatibility; scalar reward uses positive wait satisfaction instead.")
    parser.add_argument("--driver_reward_reject_weight", type=float, default=0.0, help="Legacy option retained for compatibility; scalar reward uses positive driver-acceptance satisfaction instead.")
    parser.add_argument("--driver_reward_unfulfilled_weight", type=float, default=0.05, help="Legacy option retained for compatibility; scalar reward uses positive fulfillment/service quality instead.")
    parser.add_argument("--driver_reward_warmup_fraction", type=float, default=0.60, help="Fraction of training over which driver reward terms ramp from zero to full strength.")
    parser.add_argument("--enable_constrained_reward", action="store_true", help="Opt in to operational constraint multipliers; default training optimizes profit and profit dominance only.")
    parser.add_argument("--disable_constrained_reward", action="store_true", help="Deprecated compatibility flag; unconstrained profit optimization is already the default.")
    parser.add_argument("--constraint_lr", type=float, default=0.03, help="Learning rate for constraint multipliers used by diagnostics/constraint critics.")
    parser.add_argument("--constraint_penalty_scale", type=float, default=0.10, help="Risk-advantage coefficient at full constraint curriculum scale.")
    parser.add_argument("--constraint_curriculum_start_scale", type=float, default=0.25, help="Legacy constraint-diagnostic curriculum scale used during early exploration.")
    parser.add_argument("--constraint_curriculum_mid_scale", type=float, default=0.60, help="Legacy constraint-diagnostic curriculum scale reached by mid-training.")
    parser.add_argument("--constraint_curriculum_end_scale", type=float, default=1.00, help="Legacy constraint-diagnostic curriculum scale at the end of training.")
    parser.add_argument("--gap_band_fraction", type=float, default=0.75, help="Fraction of reward_acceptable_discount allowed before adaptive gap-band violations activate.")
    parser.add_argument("--gap_penalty_scale_fraction", type=float, default=0.75, help="Legacy compatibility option; price-gap reward now uses reward_acceptable_discount as its tolerance.")
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
    parser.add_argument("--calibration_preset", type=str, default="", choices=["", "nyc_public"], help="Optional built-in calibration preset. Use nyc_public explicitly for NYC TLC + ACS + weather priors.")
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
    if args.eval_only and not args.trained_model_in:
        parser.error("--eval_only requires --trained_model_in")
    if args.trained_model_in:
        args.firm1_mode = "RL"
        args.run_experiment = True
        artifact_path = os.path.abspath(os.path.expanduser(args.trained_model_in))
        try:
            artifact_header = torch.load(artifact_path, map_location="cpu", weights_only=False)
        except TypeError:
            artifact_header = torch.load(artifact_path, map_location="cpu")
        if artifact_header.get("format") != "ride-response-platform-policy":
            parser.error(f"unsupported --trained_model_in artifact: {artifact_path}")
        args.state_frame_stack = int(artifact_header.get("state_frame_stack", args.state_frame_stack))
        saved_mdp = artifact_header.get("mdp_config", {})
        saved_observation = saved_mdp.get("observation", {})
        saved_reward = saved_mdp.get("positive_reward", {})
        saved_active_reward = saved_mdp.get("active_reward", {})
        saved_constraints = saved_mdp.get("constraints", {})
        saved_curriculum = saved_mdp.get("training_curriculum", {})
        args.telemetry_delay_steps = int(saved_observation.get("telemetry_delay_steps", args.telemetry_delay_steps))
        args.quote_probe_interval_steps = int(saved_observation.get("quote_probe_interval_steps", args.quote_probe_interval_steps))
        args.quote_probe_delay_steps = int(saved_observation.get("quote_probe_delay_steps", args.quote_probe_delay_steps))
        args.quote_noise_dollars = float(saved_observation.get("quote_noise_dollars", args.quote_noise_dollars))
        args.quote_missing_probability = float(saved_observation.get("quote_missing_probability", args.quote_missing_probability))
        args.positive_reward_profit_weight = float(saved_reward.get("profit_weight", args.positive_reward_profit_weight))
        args.positive_reward_revenue_weight = float(saved_reward.get("revenue_weight", args.positive_reward_revenue_weight))
        args.positive_reward_completed_demand_weight = float(
            saved_reward.get("completed_demand_weight", args.positive_reward_completed_demand_weight)
        )
        args.positive_reward_service_weight = float(saved_reward.get("service_weight", args.positive_reward_service_weight))
        args.reward_rev_scale = float(saved_reward.get("revenue_scale", args.reward_rev_scale))
        args.reward_profit_scale = float(saved_reward.get("profit_scale", args.reward_profit_scale))
        args.long_term_profit_scale = float(
            saved_active_reward.get("own_profit_scale", args.long_term_profit_scale)
        )
        args.long_term_profit_advantage_scale = float(
            saved_active_reward.get(
                "profit_advantage_scale", args.long_term_profit_advantage_scale
            )
        )
        args.long_term_profit_weight = float(
            saved_active_reward.get("own_profit_weight", args.long_term_profit_weight)
        )
        args.profit_dominance_weight = float(
            saved_active_reward.get(
                "profit_advantage_weight", args.profit_dominance_weight
            )
        )
        args.dominance_quality_scale = float(
            saved_active_reward.get(
                "dominance_quality_scale", args.dominance_quality_scale
            )
        )
        args.intervention_cost_weight = float(
            saved_active_reward.get(
                "intervention_cost_weight", args.intervention_cost_weight
            )
        )
        args.reversal_cost_weight = float(
            saved_active_reward.get("reversal_cost_weight", args.reversal_cost_weight)
        )
        args.ppo_gamma = float(artifact_header.get("gamma", args.ppo_gamma))
        args.ppo_gae_lambda = float(
            artifact_header.get("gae_lambda", args.ppo_gae_lambda)
        )
        args.target_price_gap = float(saved_constraints.get("target_gap", args.target_price_gap or 0.0))
        args.price_gap_tolerance = float(saved_constraints.get("overall_tolerance", args.price_gap_tolerance))
        args.segment_gap_tolerances = list(saved_constraints.get("segment_tolerances", args.segment_gap_tolerances))
        args.min_profit_margin = float(saved_constraints.get("margin_floor", args.min_profit_margin))
        args.training_curriculum = str(saved_curriculum.get("mode", args.training_curriculum))
        
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
    effective_seed = args.experiment_seed if args.experiment_seed is not None else args.seed

    core = Core(
        market_name=args.market,
        seed=effective_seed,
        choice_mode=args.choice_mode,
        model_name=args.model,
        openai_api_key=args.openai_api_key,
        firm1_mode=args.firm1_mode,
        firm2_mode=args.firm2_mode,
        firm1_static_values=args.firm1_static_values,
        firm2_static_values=args.firm2_static_values,
        total_customers_pool=args.pool,
        deterministic_torch=args.deterministic_torch,
        positive_reward_profit_weight=args.positive_reward_profit_weight,
        positive_reward_revenue_weight=args.positive_reward_revenue_weight,
        positive_reward_completed_demand_weight=args.positive_reward_completed_demand_weight,
        positive_reward_service_weight=args.positive_reward_service_weight,
        long_term_profit_scale=args.long_term_profit_scale,
        long_term_profit_advantage_scale=args.long_term_profit_advantage_scale,
        long_term_profit_weight=args.long_term_profit_weight,
        profit_dominance_weight=args.profit_dominance_weight,
        dominance_quality_scale=args.dominance_quality_scale,
        intervention_cost_weight=args.intervention_cost_weight,
        reversal_cost_weight=args.reversal_cost_weight,
        ppo_gamma=args.ppo_gamma,
        ppo_gae_lambda=args.ppo_gae_lambda,
        reward_share_weight=args.reward_share_weight,
        reward_revenue_weight=args.reward_revenue_weight,
        reward_profit_weight=args.reward_profit_weight,
        reward_price_gap_weight=args.reward_price_gap_weight,
        reward_hold_inaction_weight=args.reward_hold_inaction_weight,
        reward_corrective_action_weight=args.reward_corrective_action_weight,
        reward_baseline_loss_weight=args.reward_baseline_loss_weight,
        reward_overprice_weight=args.reward_overprice_weight,
        reward_rev_scale=args.reward_rev_scale,
        reward_competitive_weight=args.reward_competitive_weight,
        reward_trend_weight=args.reward_trend_weight,
        reward_profit_scale=args.reward_profit_scale,
        reward_underprice_weight=args.reward_underprice_weight,
        reward_acceptable_discount=args.reward_acceptable_discount,
        min_profit_margin=args.min_profit_margin,
        reward_action_change_weight=args.reward_action_change_weight,
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
        driver_expected_acceptance_cutoff=args.driver_expected_acceptance_cutoff,
        driver_state_smoothing=args.driver_state_smoothing,
        driver_reward_fulfillment_weight=args.driver_reward_fulfillment_weight,
        driver_reward_wait_weight=args.driver_reward_wait_weight,
        driver_reward_reject_weight=args.driver_reward_reject_weight,
        driver_reward_unfulfilled_weight=args.driver_reward_unfulfilled_weight,
        driver_reward_warmup_fraction=args.driver_reward_warmup_fraction,
        constrained_reward=bool(
            args.enable_constrained_reward and not args.disable_constrained_reward
        ),
        constraint_lr=args.constraint_lr,
        constraint_penalty_scale=args.constraint_penalty_scale,
        constraint_curriculum_start_scale=args.constraint_curriculum_start_scale,
        constraint_curriculum_mid_scale=args.constraint_curriculum_mid_scale,
        constraint_curriculum_end_scale=args.constraint_curriculum_end_scale,
        gap_band_fraction=args.gap_band_fraction,
        gap_penalty_scale_fraction=args.gap_penalty_scale_fraction,
        target_price_gap=args.target_price_gap,
        price_gap_tolerance=args.price_gap_tolerance,
        segment_gap_tolerances=tuple(args.segment_gap_tolerances),
        telemetry_delay_steps=args.telemetry_delay_steps,
        quote_probe_interval_steps=args.quote_probe_interval_steps,
        quote_probe_delay_steps=args.quote_probe_delay_steps,
        quote_noise_dollars=args.quote_noise_dollars,
        quote_missing_probability=args.quote_missing_probability,
        training_curriculum=args.training_curriculum,
        ppo_batch_size=args.ppo_batch_size,
        ppo_update_epochs=args.ppo_update_epochs,
        state_frame_stack=args.state_frame_stack,
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

    if args.trained_model_in:
        core.load_trained_model(
            args.trained_model_in,
            load_optimizer=not args.eval_only,
            restore_tariff=False,
        )


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
            ppo_min_rollout_transitions=args.ppo_min_rollout_transitions,
            stochastic_training=(not args.deterministic_experiment_seed and args.experiment_seed is None),
            firm1_action_interval_steps=args.firm1_action_interval_steps,
            firm2_action_interval_days=args.firm2_action_interval_days,
            eval_policy_mode=args.eval_policy_mode,
            eval_policy_temperature=args.eval_policy_temperature,
            eval_top2_margin=args.eval_top2_margin,
            eval_guardrail_mode=args.eval_guardrail_mode,
            training_enabled=not args.eval_only,
            trained_model_out=args.trained_model_out or None,
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
    _write_run_config(args.report_prefix, args, core)
    
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
