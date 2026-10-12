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
import hashlib
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
from typing import Dict, Any, List, Mapping, Tuple, Optional


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
from calibration_utils import derive_calibration, load_calibration_json, load_calibration_preset
from city_data_pipeline import (
    DataValidationError,
    NOAAWeatherIndex,
    normalize_ride_record,
    resolve_city_spec,
)
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
    POLICY_OBJECTIVES,
    PlatformObservationModel,
    LongTermProfitReward,
    LongTermProfitRewardConfig,
    PositiveBusinessReward,
    PositiveRewardConfig,
    RobustObjectiveOptimizer,
    SoftConstraintController,
    OperationalClock,
    TrainingStage,
    TrainingStageScheduler,
    conditional_competitive_shares,
    config_payload as platform_mdp_config_payload,
    objective_profit_share,
    objective_reward_weights,
    policy_objective_defaults,
)
from model_registry import (
    allocate_archive_path,
    append_manifest,
    default_model_id,
    list_registered_models,
    resolve_model_reference,
    safe_model_id,
)


# The case study's deployment comparison is intentionally fixed to these five
# public-policy baselines.  Keeping one canonical list prevents training,
# checkpoint selection, certification, and reporting from silently evaluating
# different opponents.
BENCHMARK_OPPONENT_MODES: Tuple[str, ...] = (
    "pi_price_gap",
    "mpc_grid",
    "region_supply_demand",
    "surge_driver_incentive",
    "queue_service_threshold",
)

CROWDLLM_THRESHOLD_EVIDENCE_PROTOCOL = "city_reference_tariffs_v1"
CROWDLLM_THRESHOLD_PROMPT_PROTOCOL = "structured_context_rubric_v10"
CROWDLLM_THRESHOLD_AUDIT_PROTOCOL = "deterministic_context_validation_v1"
CROWDLLM_REFERENCE_FIRM1_MULTIPLIER = 1.00
CROWDLLM_REFERENCE_FIRM2_MULTIPLIER = 1.10

# Experiments choose a prefix of this fixed, documented order.  The resolved
# count is part of the trained-policy contract, so validation and evaluation
# cannot silently expose the actor or its opponent to a different tariff scope.
MANIPULATABLE_PRICE_COEFFICIENT_KEYS: Tuple[str, ...] = (
    "base_fare",
    "per_minute",
    "per_mile",
    "booking_fee",
    "airport_fee",
)

# The auxiliary world model predicts twelve public market-response deltas plus
# the realized Firm 2 leader intervention that preceded Firm 1's action (up to
# five normalized coefficient moves and one intervention indicator). The
# active audit vector is sized from the experiment's locked coefficient scope.
# Simulator-private coefficients are teacher labels during training only;
# deployment inference uses posted quotes and public history.
MARKET_RESPONSE_DIM = 12
OPPONENT_ACTION_FORECAST_DIM = 6
RESPONSE_TARGET_DIM = MARKET_RESPONSE_DIM + OPPONENT_ACTION_FORECAST_DIM
DURABILITY_MIN_STEPS = 100
PRIVILEGED_CRITIC_STATE_DIM = 64
V25_CRITIC_OPPONENT_MODES: Tuple[str, ...] = (
    "static",
    "adaptive_best_response_aggressive",
    "adaptive_best_response",
    "heuristic",
    "heuristic_margin",
    "heuristic_random",
    "pi_price_gap",
    "mpc_grid",
    "region_supply_demand",
    "surge_driver_incentive",
    "queue_service_threshold",
)
V25_PRIVILEGED_CRITIC_STATE_DIM = (
    PRIVILEGED_CRITIC_STATE_DIM + len(V25_CRITIC_OPPONENT_MODES)
)

try:
    import kagglehub
except Exception:
    kagglehub = None

try:
    import pyarrow.parquet as pq
except Exception:
    pq = None

def _resolve_dataset_path(city: str = "nyc") -> str:
    """Download the Kaggle dataset only when explicitly needed."""
    if kagglehub is None:
        raise RuntimeError("kagglehub is required only for --compare_with_dataset; install it to use that option")
    spec = resolve_city_spec(city)
    dataset_path = kagglehub.dataset_download(spec.kaggle_slug)
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


def _canonical_json(obj: Any) -> str:
    """Stable JSON used to audit the exact raw dataset record."""
    return json.dumps(
        obj,
        ensure_ascii=False,
        default=str,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

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


def _iter_tabular_rows(fpath: str, sample_seed: Optional[int] = None):
    """Yield rows from supported tabular files.

    For parquet inputs, ``sample_seed`` deterministically randomizes both row
    group order and rows within each emitted batch. This makes bounded
    validation runs independent samples instead of repeatedly scoring the
    beginning of a chronologically sorted dataset.
    """
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
            raise RuntimeError(
                "Parquet validation requires pyarrow. Install it in the active "
                "environment (for example: conda install -c conda-forge pyarrow)."
            )
        parquet_file = pq.ParquetFile(fpath)
        fieldnames = parquet_file.schema.names
        rng = np.random.default_rng(sample_seed)
        group_order = list(range(parquet_file.num_row_groups))
        if sample_seed is not None:
            group_order = [int(group) for group in rng.permutation(group_order)]
        # Round-robin row groups so a bounded comparison covers multiple
        # temporal partitions. Seeded group ordering makes repeated robustness
        # runs draw different partitions without loading the full parquet file.
        # Each group also receives a bounded seeded batch offset. Without this,
        # different seeds that select the same large row group would still read
        # the identical first 1,000 rows from that group.
        iterators = []
        for group in group_order:
            iterator = iter(
                parquet_file.iter_batches(batch_size=1000, row_groups=[group])
            )
            if sample_seed is not None:
                group_rows = int(parquet_file.metadata.row_group(group).num_rows)
                group_batches = max(1, int(np.ceil(group_rows / 1000.0)))
                skip_batches = int(rng.integers(0, min(group_batches, 256)))
                for _ in range(skip_batches):
                    try:
                        next(iterator)
                    except StopIteration:
                        break
            iterators.append(iterator)
        while iterators:
            active = []
            for iterator in iterators:
                try:
                    batch = next(iterator)
                except StopIteration:
                    continue
                active.append(iterator)
                records = batch.to_pylist()
                if records:
                    if sample_seed is not None and len(records) > 1:
                        order = rng.permutation(len(records))
                        records = [records[int(index)] for index in order]
                    yield fieldnames, records
            iterators = active
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
    location_fields = [
        "source", "destination", "pickup", "dropoff", "pickup_zone", "dropoff_zone",
        "pickup_location", "dropoff_location",
    ]
    values = [row.get(name) for name in location_fields if row.get(name) is not None]
    text = " ".join(str(value).lower() for value in values)
    airport_tokens = ["airport", "jfk", "lga", "ewr", "laguardia", "newark"]
    if any(tok in text for tok in airport_tokens):
        return True
    # NYC TLC taxi-zone ids: Newark Airport=1, JFK=132, LaGuardia=138.
    airport_zone_ids = {1, 132, 138}
    return any((_to_int(value) in airport_zone_ids) for value in values)


def _infer_hour_day(row: Dict[str, Any]) -> Tuple[int, int]:
    hour = _to_int(_pick_value(row, ["hour", "pickup_hour", "hour_of_day"]))
    day = _to_int(_pick_value(row, ["day_of_week", "weekday", "day"]))
    day_was_explicit = day is not None

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

    if day_was_explicit and day is not None and day >= 1 and day <= 7:
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
        manipulated_price_coefficients: int = 5,
        total_customers_pool: int = 20000,
        simulation_sample_cap: int = 2000,
        deterministic_torch: bool = False,
        positive_reward_profit_weight: float = 0.38,
        positive_reward_revenue_weight: float = 0.22,
        positive_reward_completed_demand_weight: float = 0.20,
        positive_reward_service_weight: float = 0.20,
        long_term_profit_scale: float = 2.0,
        long_term_profit_advantage_scale: float = 1.0,
        policy_objective: str = "profit_maximization",
        long_term_profit_weight: float = 1.0,
        profit_dominance_weight: float = 0.0,
        dominance_quality_scale: float = 4.0,
        market_share_competitiveness_weight: float = 0.0,
        market_share_target_gap: float = 0.10,
        market_share_gap_scale: float = 0.10,
        market_share_level_weight: float = 0.25,
        price_competitiveness_weight: float = 0.0,
        price_gap_reward_scale: float = 1.0,
        intervention_cost_weight: float = 0.0,
        reversal_cost_weight: float = 0.0,
        ppo_gamma: float = 0.999,
        ppo_gae_lambda: float = 0.99,
        reward_rev_scale: float = 25.0,
        reward_profit_scale: float = 12.0,
        min_profit_margin: float = 0.08,
        min_completion_coverage: float = 0.55,
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
        constrained_reward: bool = False,
        constraint_lr: float = 0.01,
        constraint_penalty_scale: float = 0.10,
        constraint_curriculum_start_scale: float = 0.25,
        constraint_curriculum_mid_scale: float = 0.60,
        constraint_curriculum_end_scale: float = 1.00,
        target_price_gap: float = 0.75,
        price_gap_tolerance: float = 0.45,
        segment_gap_tolerances: Tuple[float, float, float, float] = (0.65, 0.60, 0.55, 0.55),
        telemetry_delay_steps: int = 1,
        quote_probe_interval_steps: int = 3,
        quote_probe_delay_steps: int = 1,
        quote_noise_dollars: float = 0.18,
        quote_missing_probability: float = 0.03,
        training_curriculum: str = "staged",
        training_stage_market_fraction: float = 0.20,
        training_stage_static_fraction: float = 0.20,
        training_stage_dynamic_fraction: float = 0.60,
        training_dynamic_episode_days: int = 600,
        training_opening_tariff_randomization: float = 0.15,
        ppo_batch_size: int = 48,
        ppo_update_epochs: int = 8,
        state_action_mi_weight: float = 0.0,
        direct_actor_credit_weight: float = 0.0,
        state_frame_stack: int = 16,
        credit_assignment_v24: bool = False,
        credit_assignment_v25: bool = False,
        privileged_critic: Optional[bool] = None,
        persistent_recurrence: Optional[bool] = None,
        counterfactual_branches: Optional[int] = None,
        counterfactual_q_coeff: Optional[float] = None,
        counterfactual_actor_weight: Optional[float] = None,
        potential_shaping_scale: Optional[float] = None,
        hard_kl_stop: Optional[bool] = None,
        reward_rate_beta: Optional[float] = None,
        threshold_cache_path: str = "",
        threshold_profile_source: str = "generated",
        threshold_prompt_repeat_source: str = "",
        save_threshold_cache: bool = True,
        gpt_threshold_include_rationales: bool = False,
        gpt_threshold_coldstart_rides: int = 5,
        gpt_threshold_send_raw_rides: bool = False,
        gpt_threshold_batch_size: int = 20,
        gpt_threshold_max_retries: int = 2,
        gpt_threshold_failure_pause: float = 1.0,
        training_generalization_fraction: float = 0.0,
        evaluation_generalization_fraction: float = 1.0,
    ):
        self.seed = int(seed) if seed is not None else int(np.random.SeedSequence().generate_state(1)[0])
        self.total_customers_pool = int(total_customers_pool)
        self.simulation_sample_cap = int(max(0, simulation_sample_cap))
        
        self._seed_all_rngs(self.seed, deterministic_torch=deterministic_torch)
        self.rng = np.random.default_rng(self.seed)
        self.market = MarketInteraction(city_name=market_name, seed=self.seed)

        self.market_name = str(self.market.current_city)

        self.agent_gen = GenerateAgent(
            seed=self.seed,
            total_customers=total_customers_pool,
            city_name=self.market_name,
        )
        self.profile_pool_multiplier = 2
        self.model_name = model_name
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.synthetic_profile_pool: List[Dict[str, Any]] = []
        # Both partitions come from the same case-study population (including
        # the same cached threshold source when requested).  Evaluation uses a
        # deterministic holdout partition, not a synthetic cross-city threshold
        # distribution that changes the problem being evaluated.
        self.training_profile_pool: List[Dict[str, Any]] = []
        self.evaluation_profile_pool: List[Dict[str, Any]] = []
        self.generalization_profile_pool: List[Dict[str, Any]] = []
        self.generalization_profiles_by_city: Dict[str, List[Dict[str, Any]]] = {}
        self.training_generalization_fraction = float(np.clip(
            training_generalization_fraction, 0.0, 1.0
        ))
        self.evaluation_generalization_fraction = float(np.clip(
            evaluation_generalization_fraction, 0.0, 1.0
        ))
        self.gpt_threshold_batch_size = int(max(1, gpt_threshold_batch_size))
        self.gpt_threshold_max_retries = int(max(0, gpt_threshold_max_retries))
        self.gpt_threshold_error_counts: Dict[str, int] = {}
        self.gpt_threshold_last_error = ""
        self.gpt_threshold_request_counts: Dict[str, int] = new_gpt_threshold_usage_counts()
        self.gpt_threshold_failure_pause = float(max(0.0, gpt_threshold_failure_pause))
        self.threshold_cache_path = str(threshold_cache_path or "")
        self.threshold_prompt_repeat_source = str(
            threshold_prompt_repeat_source or ""
        )
        requested_profile_source = str(threshold_profile_source or "generated").strip().lower()
        if requested_profile_source not in {"generated", "cached"}:
            raise ValueError("threshold_profile_source must be either 'generated' or 'cached'")
        self.strict_cached_profiles = requested_profile_source == "cached"
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
        # The default production clock is five market steps. run_experiment
        # replaces this with the resolved shared value before collecting data.
        self.shared_action_interval_steps = 5
        self.shared_steps_per_day = 10

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

        manipulated_count = int(manipulated_price_coefficients)
        if not 1 <= manipulated_count <= len(MANIPULATABLE_PRICE_COEFFICIENT_KEYS):
            raise ValueError("manipulated_price_coefficients must be between 1 and 5")
        self.manipulated_price_coefficient_count = manipulated_count
        self.opt_keys = list(
            MANIPULATABLE_PRICE_COEFFICIENT_KEYS[:manipulated_count]
        )
        self.shared_edit_keys = list(self.opt_keys)

        v25 = bool(credit_assignment_v25)
        v24 = bool(credit_assignment_v24 or v25)
        self.credit_assignment_v25 = v25
        self.training_dynamic_opponent_modes = tuple(dict.fromkeys((
            str(self.firm2_mode),
            "adaptive_best_response",
            "heuristic",
            "heuristic_random",
            *(
                (
                    "pi_price_gap",
                    "mpc_grid",
                    "region_supply_demand",
                    "surge_driver_incentive",
                    "queue_service_threshold",
                )
                if v25
                else ()
            ),
        )))
        self.privileged_critic = bool(
            v24 if privileged_critic is None else privileged_critic
        )
        self.persistent_recurrence = bool(
            v24 if persistent_recurrence is None else persistent_recurrence
        )
        self.counterfactual_branches = int(max(
            0,
            (3 if v24 else 0)
            if counterfactual_branches is None
            else counterfactual_branches,
        ))
        self.counterfactual_q_coeff = float(max(
            0.0,
            (0.50 if v24 else 0.0)
            if counterfactual_q_coeff is None
            else counterfactual_q_coeff,
        ))
        self.counterfactual_actor_weight = float(np.clip(
            # V25's one-market-step CRN branches are deliberately critic-only:
            # their immediate ordering is not the five-step semi-MDP objective
            # and can point against profit after demand, supply, and the rival
            # respond. Clipped PPO/GAE over the complete held-action return is
            # therefore the only default actor credit. The v24 value is kept
            # strictly for artifact/ablation compatibility.
            (0.0 if v25 else 0.20 if v24 else 0.0)
            if counterfactual_actor_weight is None
            else counterfactual_actor_weight,
            0.0,
            1.0,
        ))
        self.potential_shaping_scale = float(max(
            0.0,
            (0.05 if v24 else 0.0)
            if potential_shaping_scale is None
            else potential_shaping_scale,
        ))
        self.hard_kl_stop = bool(
            v24 if hard_kl_stop is None else hard_kl_stop
        )
        self.reward_rate_beta = float(np.clip(
            # V25 updates a regime-keyed lagged reward-rate cache only after
            # the current population's advantages have been fixed.  A modest
            # beta prevents one noisy completed population from replacing the
            # baseline coordinate system wholesale.
            (0.20 if v25 else 0.05 if v24 else 0.0)
            if reward_rate_beta is None
            else reward_rate_beta,
            0.0,
            1.0,
        ))
        if self.counterfactual_branches > 0 and self.counterfactual_q_coeff <= 0.0:
            raise ValueError(
                "counterfactual branches require counterfactual_q_coeff > 0"
            )
        self.credit_assignment_config = {
            "version": (
                25
                if v25
                else 24
                if any((
                    self.privileged_critic,
                    self.persistent_recurrence,
                    self.counterfactual_branches > 0,
                    self.potential_shaping_scale > 0.0,
                    self.hard_kl_stop,
                    self.reward_rate_beta > 0.0,
                ))
                else 23
            ),
            "privileged_critic": self.privileged_critic,
            "privileged_critic_state_dim": (
                (
                    V25_PRIVILEGED_CRITIC_STATE_DIM
                    if v25
                    else PRIVILEGED_CRITIC_STATE_DIM
                )
                if self.privileged_critic
                else 0
            ),
            "persistent_recurrence": self.persistent_recurrence,
            "counterfactual_branches": self.counterfactual_branches,
            "counterfactual_q_coeff": self.counterfactual_q_coeff,
            "counterfactual_actor_weight": self.counterfactual_actor_weight,
            "potential_shaping_scale": self.potential_shaping_scale,
            "hard_kl_stop": self.hard_kl_stop,
            "reward_rate_beta": self.reward_rate_beta,
        }
        if v25:
            self.credit_assignment_config.update({
                "transition_contract": (
                    "discounted_smdp_sum_actual_next_decision"
                ),
                "counterfactual_q_contract": (
                    "independent_action_feature_pairwise_diagnostic"
                ),
                "recurrent_ingestion": "every_public_frame",
                "critic_target_contract": "actual_next_decision_td0",
                "privileged_value_contract": "direct_centralized_state_mlp",
                "actor_credit_contract": "ppo_complete_smdp_return_only",
                "opponent_loss_weighting": "equal_terminal_regime_mass",
                "reward_baseline_contract": (
                    "lagged_opponent_regime_action_independent"
                ),
                "behavior_policy_contract": (
                    "cosine_actor_uniform_mixture_exact_ppo_ratio"
                ),
                "actor_initialization_contract": (
                    "zero_logits_uniform_prior_deterministic_hold_tie_break"
                ),
                "critic_opponent_context": (
                    "training_only_controller_class_one_hot"
                ),
            })

        # The market simulator retains full ground truth.  Everything below is
        # the platform-facing POMDP boundary: delayed own telemetry, sampled
        # public quote probes, positive business utility, and separate
        # operational audit/feasibility channels.
        mdp_target_gap = float(target_price_gap)
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
        objective_preset = policy_objective_defaults(str(policy_objective))
        self.long_term_profit_reward_config = LongTermProfitRewardConfig(
            objective_mode=str(policy_objective),
            own_profit_scale=float(long_term_profit_scale),
            profit_advantage_scale=float(long_term_profit_advantage_scale),
            own_profit_weight=float(
                objective_preset["long_term_profit_weight"]
            ),
            profit_advantage_weight=float(
                objective_preset["profit_dominance_weight"]
            ),
            dominance_quality_scale=float(dominance_quality_scale),
            market_share_competitiveness_weight=float(
                objective_preset["market_share_competitiveness_weight"]
            ),
            market_share_target_gap=float(market_share_target_gap),
            market_share_gap_scale=float(market_share_gap_scale),
            market_share_level_weight=float(
                objective_preset["market_share_level_weight"]
            ),
            price_competitiveness_weight=float(
                objective_preset["price_competitiveness_weight"]
            ),
            price_gap_target=mdp_target_gap,
            price_gap_scale=float(price_gap_reward_scale),
            intervention_cost_weight=float(
                objective_preset["intervention_cost_weight"]
            ),
            reversal_cost_weight=float(
                objective_preset["reversal_cost_weight"]
            ),
        )
        # Named objectives use the semi-MDP return as their only actor credit.
        # An immediate-delta blend double-counts the same held action and can
        # point against the final objective while an adaptive rival reacts.
        del direct_actor_credit_weight
        resolved_direct_actor_credit_weight = 0.0
        self.constraint_config = ConstraintConfig(
            target_gap=mdp_target_gap,
            overall_tolerance=float(max(0.05, price_gap_tolerance)),
            segment_tolerances=tuple(float(max(0.05, x)) for x in segment_gap_tolerances),
            margin_floor=float(np.clip(min_profit_margin, 0.0, 0.95)),
            completion_coverage_floor=float(
                np.clip(min_completion_coverage, 0.0, 1.0)
            ),
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
        self.objective_optimizer = RobustObjectiveOptimizer(
            objective=str(policy_objective),
            own_profit_scale=float(long_term_profit_scale),
            profit_advantage_scale=float(long_term_profit_advantage_scale),
            profit_advantage_target=0.0,
            market_share_advantage_target=0.0,
            fulfillment_floor=float(self.constraint_config.fulfillment_floor),
            completion_coverage_floor=float(
                self.constraint_config.completion_coverage_floor
            ),
            market_share_gap_scale=float(market_share_gap_scale),
            own_completed_share_scale=float(
                self.long_term_profit_reward_config.own_completed_share_scale
            ),
        )
        self.last_price_gap_metrics: Dict[str, float] = {
            "signed_mean": float(mdp_target_gap),
            "mae": 0.0,
            "rmse": 0.0,
            "p90_abs_error": 0.0,
            "within_tolerance_rate": 1.0,
            "target": float(mdp_target_gap),
        }
        self.soft_constraints = SoftConstraintController(self.constraint_config)
        self.action_stability = ActionStabilityTracker(self.constraint_config)
        self.training_stage_scheduler = TrainingStageScheduler(
            training_curriculum,
            stage_fractions=(
                training_stage_market_fraction,
                training_stage_static_fraction,
                training_stage_dynamic_fraction,
            ),
            dynamic_episode_days=training_dynamic_episode_days,
            tariff_reset_fraction=training_opening_tariff_randomization,
        )
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
                constraint_dim=0,
                response_dim=0,
                opponent_forecast_dim=0,
                feature_groups=PlatformObservationModel.feature_groups(),
                gamma=float(ppo_gamma),
                gae_lambda=float(ppo_gae_lambda),
                state_action_mi_weight=float(state_action_mi_weight),
                direct_actor_credit_weight=resolved_direct_actor_credit_weight,
                critic_state_dim=(
                    V25_PRIVILEGED_CRITIC_STATE_DIM
                    if v25
                    else PRIVILEGED_CRITIC_STATE_DIM
                ),
                privileged_critic=self.privileged_critic,
                privileged_value_only=v25,
                persistent_recurrence=self.persistent_recurrence,
                counterfactual_q_coeff=self.counterfactual_q_coeff,
                counterfactual_actor_weight=self.counterfactual_actor_weight,
                potential_shaping_scale=self.potential_shaping_scale,
                hard_kl_stop=self.hard_kl_stop,
                reward_rate_beta=self.reward_rate_beta,
                td_value_targets=v25,
            )
        elif self.firm1_mode in pricer_by_mode:
            self.firm1 = pricer_by_mode[self.firm1_mode](seed=self.seed, managed_keys=self.shared_edit_keys)
        else:
            self.firm1 = FirmStaticPricer()
        self.counterfactual_action_counts = np.zeros(
            self.firm1.agent.action_dim if self.firm1_mode == "RL" else 0,
            dtype=np.int64,
        )

        if self.firm2_mode in pricer_by_mode:
            self.firm2 = pricer_by_mode[self.firm2_mode](seed=self.seed + 1, managed_keys=self.shared_edit_keys)
        else:
            self.firm2 = FirmStaticPricer()
        setattr(self.firm2, "_ride_response_mode", str(self.firm2_mode))
        self._assert_competitor_parity()
        print(
            "[PricingControl] "
            f"manipulated_price_coefficients={self.manipulated_price_coefficient_count}; "
            f"keys={','.join(self.shared_edit_keys)}; "
            "scope_locked_for_training_validation_evaluation=true."
        )
        
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
        self.last_share = 0.0
        self.last_revpr = 0.0
        self.last_gap = 0.0
        self.last_reward = 0.0
        self.last_profitpr = 0.0
        self.last_fulfillment = 1.0
        self.last_completion_coverage = 0.0
        self.last_acceptance = 1.0
        self.last_wait = 0.0
        self.last_driver_paypr = 0.0
        self.last_firm2_share = 0.0
        self.last_firm2_revpr = 0.0
        self.last_firm2_profitpr = 0.0
        self._last_cycle_firm2_metrics = None
        self.objective_credit_ready = False
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
        self.last_action_mask_regime = "all_bounded_actions"
        
        self.training_logs = []
        self.evaluation_logs = []
        
        self.run_logs = []
        self.convergence_day: Optional[int] = None
        self.convergence_window_std_at_day: Optional[float] = None
        self.convergence_delta_per_day_at_day: Optional[float] = None
        self.training_certification_report: Optional[Dict[str, Any]] = None
        
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
        # The constrained-reward argument remains readable for older callers
        # but cannot reactivate operational shaping, action shielding, or
        # auxiliary critics.  The training curriculum is environment-only and
        # was already resolved above.
        del constrained_reward
        self.constrained_reward = False
        self.constraint_lr = float(max(0.0, constraint_lr))
        self.constraint_penalty_scale = float(max(0.0, constraint_penalty_scale))
        self.constraint_curriculum_start_scale = float(max(0.0, constraint_curriculum_start_scale))
        self.constraint_curriculum_mid_scale = float(max(0.0, constraint_curriculum_mid_scale))
        self.constraint_curriculum_end_scale = float(max(0.0, constraint_curriculum_end_scale))
        self.constraint_curriculum_scale = 1.0
        # Constraint fields remain as diagnostics for artifact/report
        # compatibility. They do not enter either named reward, PPO gradients,
        # or checkpoint ordering.
        self.constraint_lambdas: Dict[str, float] = {
            name: 0.0 for name in self.soft_constraints.names
        }
        self.constraint_lambda_max = float(self.constraint_config.multiplier_max)
        
        # Keep PPO optimization controls explicit so longer rollout windows can
        # use minibatches instead of silently falling back to full-batch updates.
        
        self.ppo_update_epochs = int(max(1, ppo_update_epochs))
        self.ppo_batch_size = int(max(1, ppo_batch_size))
        
        objective_name = self.long_term_profit_reward_config.objective_mode
        reward_weights = self.long_term_profit_reward_config.economic_weights()
        active_terms = int(sum(
            float(value) > 0.0 for value in reward_weights.values()
        ))
        objective_definition = (
            f"linear {active_terms}-term economics: "
            f"{reward_weights['own_profit']:.0%}*own_profit/$"
            f"{self.long_term_profit_reward_config.own_profit_scale:.2f} + "
            f"{reward_weights['profit_lead']:.0%}*profit_lead/$"
            f"{self.long_term_profit_reward_config.profit_advantage_scale:.2f} + "
            f"{reward_weights['own_completed_share']:.0%}*own_completed_share/"
            f"{self.long_term_profit_reward_config.own_completed_share_scale:.0%} + "
            f"{reward_weights['completed_share_lead']:.0%}*completed_share_lead/"
            f"{self.long_term_profit_reward_config.market_share_gap_scale:.0%}; "
            "unclipped, no terminal bonus, action cost, threshold, or quote-gap term"
        )
        print(
            "[PolicyObjectiveReward] "
            f"objective={objective_name}; "
            f"mechanism={self.long_term_profit_reward_model.mechanism_name}; "
            f"definition={objective_definition}; "
            f"intervention_cost={self.long_term_profit_reward_config.intervention_cost_weight:.4f}, "
            f"reversal_cost={self.long_term_profit_reward_config.reversal_cost_weight:.4f}, "
            f"gamma={self.firm1.agent.gamma if self.firm1_mode == 'RL' else float('nan'):.4f}, "
            f"lambda={self.firm1.agent.lam if self.firm1_mode == 'RL' else float('nan'):.4f}."
        )
        print(
            "[ConstraintConfig] "
            "optimization=disabled; "
            f"diagnostics={self.soft_constraints.names}, "
            f"margin_floor={self.constraint_config.margin_floor:.3f}; "
            f"coverage_floor={self.constraint_config.completion_coverage_floor:.3f}; "
            "all operational channels are reporting-only."
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
            "[PricingParity] Firm1 and Firm2 share managed keys, bounds, "
            "coefficient steps, and the operational decision clock; their "
            "policy-specific action representations may differ."
        )
    
    def apply_calibration(self, calibration: Dict[str, Any]) -> None:
        """Apply calibration outputs to market priors, agent priors, and choice sensitivity scales."""
        self.active_calibration = copy.deepcopy(calibration) if isinstance(calibration, dict) else {}
        market_cal = calibration.get("market", {}) if isinstance(calibration, dict) else {}
        agent_cal = calibration.get("agent", {}) if isinstance(calibration, dict) else {}
        choice_cal = calibration.get("choice", {}) if isinstance(calibration, dict) else {}

        weather_probs = market_cal.get("weather_probs")
        day_of_week_probs = market_cal.get("day_of_week_probs")
        hour_probs = market_cal.get("hour_probs")
        service_probs = market_cal.get("service_probs")
        airport_prob = market_cal.get("airport_prob")
        self.market.set_probability_priors(
            weather_probs=(weather_probs if isinstance(weather_probs, dict) else None),
            day_of_week_probs=(
                day_of_week_probs if isinstance(day_of_week_probs, dict) else None
            ),
            hour_probs=(hour_probs if isinstance(hour_probs, dict) else None),
            service_probs=(service_probs if isinstance(service_probs, dict) else None),
            airport_prob=(float(airport_prob) if airport_prob is not None else None),
            distance_mean_miles=market_cal.get("distance_mean_miles"),
            distance_std_miles=market_cal.get("distance_std_miles"),
            duration_mean_minutes=market_cal.get("duration_mean_minutes"),
            duration_std_minutes=market_cal.get("duration_std_minutes"),
        )

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
            if "marital_probs" in agent_cal and isinstance(agent_cal["marital_probs"], dict):
                keys = list(self.agent_gen.marital_names)
                vals = np.array([float(agent_cal["marital_probs"].get(k, 0.0)) for k in keys], dtype=float)
                s = float(vals.sum())
                if s > 0:
                    self.agent_gen.marital_probs = vals / s
            if "gender_probs" in agent_cal and isinstance(agent_cal["gender_probs"], dict):
                keys = list(self.agent_gen.gender_names)
                vals = np.array([float(agent_cal["gender_probs"].get(k, 0.0)) for k in keys], dtype=float)
                s = float(vals.sum())
                if s > 0:
                    self.agent_gen.gender_probs = vals / s
            self.agent_gen.freeze_current_priors_as_base()
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
        for firm in (self.firm1, self.firm2):
            for key in self.shared_edit_keys:
                anchor = float(getattr(base, key))
                value = anchor * float(self.rng.uniform(lo, hi))
                floor = 0.01 if key in {
                    "base_fare", "per_minute", "per_mile"
                } else 0.0
                setattr(firm.overrides, key, max(floor, value))

    def _make_pricer(self, mode: str, seed: int):
        """Construct a fresh opponent without carrying hidden controller state."""
        mode = str(mode).strip().lower()
        cls = self.pricer_by_mode.get(mode)
        pricer = (
            FirmStaticPricer()
            if cls is None
            else cls(seed=int(seed), managed_keys=self.shared_edit_keys)
        )
        setattr(
            pricer,
            "_ride_response_mode",
            mode if cls is not None else "static",
        )
        return pricer

    def _set_firm2_mode_preserving_tariff(self, mode: str, seed: int) -> None:
        """Change only Firm 2's controller, preserving its posted tariff."""
        base = self.market.curr_market
        current_tariff = {
            key: float(get_coeff(base, self.firm2.overrides, key))
            for key in self.shared_edit_keys
        }
        self.firm2 = self._make_pricer(mode, int(seed))
        for key, value in current_tariff.items():
            set_coeff(base, self.firm2.overrides, key, value)

    def _set_firm2_to_market_reference(self, seed: int) -> None:
        """Use the configured city's base tariff as the market-stage reference."""
        self.firm2 = self._make_pricer("static", int(seed))
        base = self.market.curr_market
        for key in self.shared_edit_keys:
            set_coeff(
                base,
                self.firm2.overrides,
                key,
                float(getattr(base, key)),
            )

    def _training_stage_opponent_mode(
        self,
        stage: TrainingStage,
        configured_mode: str,
        episode_seed: Optional[int] = None,
    ) -> str:
        """Resolve the controller used by one curriculum stage."""
        if str(stage.opponent_regime) == "dynamic":
            modes = tuple(getattr(
                self,
                "training_dynamic_opponent_modes",
                (),
            )) or tuple(dict.fromkeys((
                str(configured_mode),
                "adaptive_best_response",
                "heuristic",
                "heuristic_random",
            )))
            if episode_seed is None:
                return str(configured_mode)
            # Episode seeds advance by exactly ``episode_days`` in the training
            # loop, so this deterministic rotation covers every controller
            # before repeating instead of randomly omitting one in a short run.
            episode_index = int(episode_seed) // max(1, int(stage.episode_days))
            return str(modes[episode_index % len(modes)])
        return "static"

    def _training_opponent_episode_days(
        self,
        stage: TrainingStage,
        opponent_mode: str,
    ) -> int:
        """Match training episode length to the controller response timescale."""
        base_days = int(max(1, stage.episode_days))
        if (
            self.credit_assignment_v25
            and str(stage.opponent_regime) == "dynamic"
            and str(opponent_mode) in {
                "pi_price_gap",
                "region_supply_demand",
            }
        ):
            return int(max(300, base_days))
        return base_days

    def _training_opponent_population(self) -> Tuple[str, ...]:
        """Return one statistically balanced v25 opponent population cycle.

        Fast controllers supply three independently reset short episodes per
        cycle; the two slow controllers already supply 300-day trajectories
        and appear once. Loss weighting later gives every *controller class*
        equal mass, so replication improves coverage without changing the
        intended opponent objective.
        """
        modes = tuple(self.training_dynamic_opponent_modes)
        if not self.credit_assignment_v25:
            return modes
        slow = {"pi_price_gap", "region_supply_demand"}
        return tuple(
            mode
            for mode in modes
            for _ in range(1 if mode in slow else 3)
        )

    def _enter_training_stage(
        self,
        stage: TrainingStage,
        *,
        configured_mode: str,
        episode_seed: int,
        opponent_mode_override: Optional[str] = None,
    ) -> str:
        """Open a fresh episode for a curriculum stage and return its controller.

        Policy and optimizer state are intentionally retained.  Tariffs,
        controller state, observation history, and simulator episode metrics
        are reset so a return never spans two opponent regimes.
        """
        selected_mode = (
            str(opponent_mode_override)
            if opponent_mode_override is not None
            else self._training_stage_opponent_mode(
                stage, configured_mode, episode_seed=int(episode_seed)
            )
        )
        self._reset_adaptive_episode(
            tariff_reset_fraction=float(stage.tariff_reset_fraction),
            opponent_mode=selected_mode,
            episode_seed=int(episode_seed),
            symmetric_tariffs=False,
        )
        if str(stage.opponent_regime) == "market":
            self._set_firm2_to_market_reference(int(episode_seed) + 1)
        return selected_mode

    def _reset_adaptive_episode(
        self,
        *,
        tariff_reset_fraction: float,
        opponent_mode: Optional[str] = None,
        episode_seed: Optional[int] = None,
        symmetric_tariffs: bool = False,
    ) -> None:
        """Create one independently randomized strategic trajectory opening.

        Training calls this at stage boundaries and at each long episode in the
        dynamic stage. Validation and evaluation use the same opening
        randomization for independent long trajectories.
        """
        if opponent_mode is not None:
            self.firm2 = self._make_pricer(
                opponent_mode,
                self.seed + 10_000 if episode_seed is None else int(episode_seed),
            )
        fraction = float(np.clip(tariff_reset_fraction, 0.02, 0.40))
        reset_rng = (
            np.random.default_rng(int(episode_seed))
            if episode_seed is not None
            else self.rng
        )
        base = self.market.curr_market
        common_values: Dict[str, float] = {}
        for firm_index, firm in enumerate((self.firm1, self.firm2)):
            config = getattr(firm, "config", None)
            for key in self.shared_edit_keys:
                anchor = float(getattr(base, key))
                if bool(symmetric_tariffs) and firm_index > 0:
                    value = float(common_values[key])
                else:
                    value = anchor * float(
                        reset_rng.uniform(1.0 - fraction, 1.0 + fraction)
                    )
                if config is not None and key in getattr(config, "bounds", {}):
                    lb, ub = config.bounds[key]
                    value = float(np.clip(value, lb, ub))
                elif key in {"base_fare", "per_minute", "per_mile"}:
                    value = max(0.01, value)
                else:
                    value = max(0.0, value)
                set_coeff(base, firm.overrides, key, value)
                if bool(symmetric_tariffs) and firm_index == 0:
                    common_values[key] = float(value)
        if self.firm1_mode == "RL":
            self.firm1.reset_state_history()
        for observer in self.platform_observers.values():
            observer.reset()
        self.action_stability.reset()
        self.last_crowd_response_stats = {}
        self.last_share = 0.0
        self.last_revpr = 0.0
        self.last_profitpr = 0.0
        self.last_gap = 0.0
        self.last_fulfillment = 1.0
        self.last_completion_coverage = 0.0
        self.last_acceptance = 1.0
        self.last_wait = 0.0
        self.last_firm2_share = 0.0
        self.last_firm2_revpr = 0.0
        self.last_firm2_profitpr = 0.0
        self.objective_credit_ready = False
        self.last_action_mask_regime = "all_bounded_actions"
        
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
    
    def _update_constraint_multipliers(self, reward_diag: Dict[str, float]) -> None:
        """Compatibility no-op; operational metrics are diagnostics only."""
        del reward_diag

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
                float(reward_diag.get("constraint_cost_coverage", 0.0)),
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
        """Keep PPO optimization tied only to its scalar economic reward."""
        if self.firm1_mode != "RL" or not hasattr(getattr(self.firm1, "agent", None), "set_optimization_context"):
            return
        self.firm1.agent.set_optimization_context(
            np.zeros(len(self.soft_constraints.names), dtype=np.float32),
            risk_coeff=0.0,
            constraints_active=False,
            risk_active=False,
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
        critic_state = (
            self._build_privileged_critic_state(raw_state)
            if self.privileged_critic
            else None
        )
        return self.firm1.agent.value_estimates(
            stacked_state,
            action_features=action_features,
            critic_state=critic_state,
        )

    def _configure_constraint_curriculum(self, progress: float) -> None:
        """Compatibility no-op; there is no constraint curriculum."""
        del progress
        self.constraint_curriculum_scale = 0.0
            
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

    def _pricer_coeff_snapshot(self, pricer) -> Dict[str, float]:
        """Resolved public tariff coefficients for either firm."""
        overrides = getattr(pricer, "overrides", CoefficientOverrides())
        return {
            key: float(get_coeff(self.market.curr_market, overrides, key))
            for key in self.shared_edit_keys
        }

    def _public_firm2_quote_gaps(
        self,
        *,
        day_of_week: int,
        hour: int,
        weather: str,
    ) -> Dict[str, float]:
        """Return representative posted-price gaps after Firm 2 commits.

        These are ordinary public ride quotes, not controller coefficients.
        Firm 1 receives the same four distance-segment view already used by the
        observation model and therefore can identify a hidden policy only from
        its observable price process and subsequent market outcomes.
        """
        public_rides = {
            "0_2": (1.2, 8.0, False),
            "2_5": (3.5, 14.0, False),
            "5_10": (7.0, 24.0, False),
            "10_plus": (13.0, 38.0, True),
        }
        gaps: Dict[str, float] = {}
        for segment, (distance, duration, airport) in public_rides.items():
            context = RideContext(
                day_of_week=int(day_of_week),
                weather=str(weather),
                hour=int(hour),
                airport=bool(airport),
                service="economy",
            )
            firm1_quote = self.market.quote_price(
                float(distance),
                float(duration),
                context,
                overrides=self.firm1.overrides,
            )
            firm2_quote = self.market.quote_price(
                float(distance),
                float(duration),
                context,
                overrides=self.firm2.overrides,
            )
            gaps[segment] = float(firm2_quote - firm1_quote)
        return gaps

    def _publish_firm2_leader_quotes(
        self,
        *,
        day_of_week: int,
        hour: int,
        weather: str,
    ) -> None:
        """Make the committed Firm 2 move observable before Firm 1 reacts."""
        self.platform_observers["Firm1"].publish_public_quote_gaps(
            self._public_firm2_quote_gaps(
                day_of_week=day_of_week,
                hour=hour,
                weather=weather,
            )
        )

    def _firm2_leader_step(
        self,
        *,
        previous_metrics: Optional[FirmMetrics],
        day_of_week: int,
        hour: int,
        weather: str,
        enabled: bool = True,
    ) -> Tuple[np.ndarray, bool]:
        """Execute one configured Firm 2 decision before Firm 1's decision.

        The controller receives only lagged realized metrics and its existing
        public observation interface.  Its native ``update`` and ``act``
        methods are called without policy substitutions or hand-written action
        overrides.  The returned target is a training-only audit/auxiliary
        label; Firm 1 observes the resulting public quotes, never this label.
        """
        empty_target = np.zeros(
            len(self.shared_edit_keys) + 1, dtype=np.float32
        )
        if (
            not bool(enabled)
            or isinstance(self.firm2, FirmStaticPricer)
        ):
            # Holding is still a committed public price. Publish it so the
            # follower sees a freshly posted static/no-change quote before its
            # first decision as well as on periods between dynamic moves.
            self._publish_firm2_leader_quotes(
                day_of_week=day_of_week,
                hour=hour,
                weather=weather,
            )
            return empty_target, False

        if hasattr(self.firm2, "observe_state"):
            firm2_state = self._build_rl_state(
                day_of_week=day_of_week,
                hour=hour,
                weather=weather,
                perspective="Firm2",
            )
            firm2_features = self._action_feature_matrix_for_pricer(
                self.firm2,
                getattr(self, "last_crowd_response_stats", {}) or {},
            )
            self.firm2.observe_state(firm2_state, firm2_features)

        if previous_metrics is not None:
            supply, supply_vector = self._driver_supply_context(
                "Firm2", "Firm1"
            )
            self.firm2.update(
                metrics=previous_metrics,
                price_gap_mean=self._observed_own_minus_competitor_gap(
                    "Firm2"
                ),
                supply_state=supply,
                supply_state_vector=supply_vector,
            )

        before = self._pricer_coeff_snapshot(self.firm2)
        base = self.market.curr_market
        self.firm2.act(
            city_base=base.base_fare,
            city_pmin=base.per_minute,
            city_pmile=base.per_mile,
            city_booking=base.booking_fee,
            city_airport=base.airport_fee,
            hour=hour,
            weather=weather,
        )
        after = self._pricer_coeff_snapshot(self.firm2)
        target = self._opponent_action_target(before, after, self.firm2)
        self._publish_firm2_leader_quotes(
            day_of_week=day_of_week,
            hour=hour,
            weather=weather,
        )
        return target, bool(target[-1] > 0.0)

    def _opponent_action_target(
        self,
        before: Mapping[str, float],
        after: Mapping[str, float],
        pricer,
    ) -> np.ndarray:
        """Normalize Firm 2's current leader intervention for supervision.

        Targets describe direction and approximate number of controller steps,
        not absolute private tariff levels.  They therefore teach the public-
        history model the public-price process that preceded the follower
        action while keeping the deployed actor's observation boundary
        unchanged.
        """
        config = getattr(pricer, "config", None)
        steps = getattr(config, "step", {}) if config is not None else {}
        fallback = {
            "base_fare": 0.10,
            "per_minute": 0.01,
            "per_mile": 0.05,
            "booking_fee": 0.10,
            "airport_fee": 0.25,
        }
        moves = []
        for key in self.shared_edit_keys:
            scale = max(1e-6, float(steps.get(key, fallback[key])))
            delta = float(after.get(key, 0.0) - before.get(key, 0.0))
            moves.append(float(np.clip(delta / scale, -2.0, 2.0)))
        intervened = float(any(abs(value) > 1e-6 for value in moves))
        return np.asarray([*moves, intervened], dtype=np.float32)

    @staticmethod
    def _finalize_opponent_action_sum(values: np.ndarray) -> np.ndarray:
        """Aggregate any rival moves within one Firm 1 reaction horizon."""
        result = np.asarray(values, dtype=np.float32).reshape(-1).copy()
        if not 2 <= result.size <= OPPONENT_ACTION_FORECAST_DIM:
            raise ValueError(
                "opponent action target has incompatible size: "
                f"{result.size}; expected 2-{OPPONENT_ACTION_FORECAST_DIM}"
            )
        result[:-1] = np.clip(result[:-1], -2.0, 2.0)
        result[-1] = float(result[-1] > 0.0)
        return result

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
        if int(getattr(getattr(pricer, "agent", None), "action_feature_dim", 0)) <= 0:
            return None
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
            action_target_keys=getattr(pricer, "action_to_target_key", None),
            action_target_multipliers=getattr(
                pricer, "action_to_multiplier", None
            ),
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
        completed_share_advantage: Optional[float] = None,
    ) -> float:
        """Return the checkpoint score owned by the configured objective.

        Every policy is ranked by its realized normalized training reward.
        The profit pipeline selects on Firm1 contribution profit and the
        competitiveness pipeline on the Firm1-minus-Firm2 completed-share gap.
        Neither uses proxy KPIs.
        """
        del completed_share_advantage
        return float(reward)

    @staticmethod
    def _evaluation_checkpoint(
        best_checkpoint: Optional[Mapping[str, Any]],
        latest_checkpoint: Optional[Mapping[str, Any]],
        best_observed_checkpoint: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[Optional[Mapping[str, Any]], str]:
        """Use the final policy unless it fails direct durability certification.

        V25's dynamic phase repeats a stationary opponent-population design,
        so retaining the best previously durable validation snapshot is a
        standard checkpoint safeguard rather than a change to reward, action
        feasibility, or deployment behavior.
        """
        if latest_checkpoint is not None:
            validation = dict(latest_checkpoint.get("validation", {}) or {})
            durability = validation.get("checkpoint_eligible")
            if durability is True:
                return latest_checkpoint, "final_dynamic_policy_durable"
            if durability is False:
                if best_checkpoint is not None:
                    best_validation = dict(
                        best_checkpoint.get("validation", {}) or {}
                    )
                    if best_validation.get("checkpoint_eligible") is True:
                        return (
                            best_checkpoint,
                            "best_durable_dynamic_policy_fallback",
                        )
                if best_observed_checkpoint is not None:
                    return (
                        best_observed_checkpoint,
                        "best_observed_dynamic_policy_not_yet_durable",
                    )
                return latest_checkpoint, "final_dynamic_policy_not_yet_durable"
            return latest_checkpoint, "final_dynamic_policy"
        return None, "final_actor"

    def _configured_primary_objective(self) -> str:
        """Return the explicit objective used for selection and certification."""
        config = getattr(self, "long_term_profit_reward_config", None)
        if config is None:
            return "profit_maximization"
        configured = str(
            getattr(config, "objective_mode", "profit_maximization")
            or "profit_maximization"
        ).strip().lower()
        if configured not in POLICY_OBJECTIVES:
            raise ValueError(f"unsupported policy objective: {configured!r}")
        return configured

    def _objective_firm_scores(
        self,
        own_profit: Sequence[float],
        rival_profit: Sequence[float],
        own_share: Sequence[float],
        rival_share: Sequence[float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return symmetric firm scores under the configured scalar objective.

        PPO maximizes Firm 1's score.  Swapping the two firms gives the
        comparison score used only for mixed-objective dominance diagnostics;
        it does not add a reward, constraint, or action requirement.
        """
        own_profit_v = np.asarray(own_profit, dtype=float)
        rival_profit_v = np.asarray(rival_profit, dtype=float)
        own_share_v = np.asarray(own_share, dtype=float)
        rival_share_v = np.asarray(rival_share, dtype=float)
        if len({
            own_profit_v.size,
            rival_profit_v.size,
            own_share_v.size,
            rival_share_v.size,
        }) != 1:
            raise ValueError("objective score inputs must be aligned")
        config = self.long_term_profit_reward_config
        weights = config.economic_weights()

        def score(
            first_profit: np.ndarray,
            second_profit: np.ndarray,
            first_share: np.ndarray,
            second_share: np.ndarray,
        ) -> np.ndarray:
            return (
                weights["own_profit"]
                * first_profit
                / config.own_profit_scale
                + weights["profit_lead"]
                * (first_profit - second_profit)
                / config.profit_advantage_scale
                + weights["own_completed_share"]
                * first_share
                / config.own_completed_share_scale
                + weights["completed_share_lead"]
                * (first_share - second_share)
                / config.market_share_gap_scale
            )

        return (
            score(own_profit_v, rival_profit_v, own_share_v, rival_share_v),
            score(rival_profit_v, own_profit_v, rival_share_v, own_share_v),
        )

    def _objective_dominance_path(
        self,
        own_profit: Sequence[float],
        rival_profit: Sequence[float],
        own_share: Sequence[float],
        rival_share: Sequence[float],
    ) -> np.ndarray:
        """Return the objective-consistent Firm 1 minus Firm 2 path."""
        objective = self._configured_primary_objective()
        if objective == "profit_maximization":
            return np.asarray(own_profit, dtype=float) - np.asarray(
                rival_profit, dtype=float
            )
        if objective == "competitiveness":
            return np.asarray(own_share, dtype=float) - np.asarray(
                rival_share, dtype=float
            )
        own_score, rival_score = self._objective_firm_scores(
            own_profit, rival_profit, own_share, rival_share
        )
        return own_score - rival_score

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
        """Active configured objective used by lightweight reporting paths."""
        del profit_margin
        del fulfillment_rate, driver_acceptance_rate, avg_wait_minutes
        return float(self.long_term_profit_reward_model.compute(
            own_profit_per_request=(
                rev_per_request if profit_per_request is None else profit_per_request
            ),
            rival_profit_per_request=competitor_profit_per_request,
            own_completed_share=float(share),
            rival_completed_share=float(competitor_share),
            price_gap_f2_minus_f1=price_gap_f2_minus_f1,
            intervention_magnitude=action_change_magnitude,
            reversal=0.0,
        )["reward"])

    def _reward_diagnostics(
        self,
        share: float,
        rev_per_request: float,
        mean_gap: float,
        prev_share: float,
        prev_rev_per_request: float,
        prev_profit_per_request: Optional[float] = None,
        prev_rival_profit_per_request: Optional[float] = None,
        prev_rival_market_share: Optional[float] = None,
        prev_fulfillment_rate: Optional[float] = None,
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
        price_gap_abs_error: Optional[float] = None,
    ) -> Dict[str, float]:
        """Objective-specific reward plus a uniform diagnostic decomposition.

        Both named objectives use the same four normalized outcomes: own
        contribution profit, profit lead, own completed-request share, and
        completed-share lead. Only their fixed weights differ. Public quote
        gaps, action changes, thresholds, and one-step KPI changes remain
        diagnostics; no terminal reward is added.
        """
        profitpr = float(rev_per_request if profit_per_request is None else profit_per_request)
        margin = float(0.0 if profit_margin is None else profit_margin)
        completed = float(share if completed_share is None else completed_share)
        conditional_choice_share, rival_conditional_choice_share, outside_option_share = (
            conditional_competitive_shares(share, baseline_share)
        )
        rival_completed = float(
            baseline_share
            if baseline_completed_share is None
            else baseline_completed_share
        )
        # The optimized competitiveness metric is actual served demand. It is
        # not renormalized after the outside option or failed fulfillment.
        market_share = float(np.clip(completed, 0.0, 1.0))
        rival_market_share = float(np.clip(rival_completed, 0.0, 1.0))
        completion_coverage = float(np.clip(
            market_share + rival_market_share,
            0.0,
            1.0,
        ))
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
            own_completed_share=completed,
            rival_completed_share=rival_completed,
            own_market_share=market_share,
            rival_market_share=rival_market_share,
            own_fulfillment_rate=float(fulfillment_rate),
            price_gap_f2_minus_f1=float(mean_gap),
            price_gap_abs_error=price_gap_abs_error,
            intervention_magnitude=(
                float(action_change_magnitude) if bool(action_event) else 0.0
            ),
            reversal=float(self.action_stability.last_reversal),
        )
        previous_market_share = float(np.clip(prev_share, 0.0, 1.0))
        previous_rival_market_share = float(np.clip(
            0.0
            if prev_rival_market_share is None
            else prev_rival_market_share,
            0.0,
            1.0,
        ))
        previous_objective = self.long_term_profit_reward_model.compute(
            own_profit_per_request=float(prev_profit_per_request or 0.0),
            rival_profit_per_request=float(prev_rival_profit_per_request or 0.0),
            own_market_share=previous_market_share,
            rival_market_share=previous_rival_market_share,
            own_fulfillment_rate=float(
                fulfillment_rate
                if prev_fulfillment_rate is None
                else prev_fulfillment_rate
            ),
            price_gap_f2_minus_f1=float(prev_gap or mean_gap),
            intervention_magnitude=0.0,
            reversal=0.0,
        )
        objective_delta_diagnostic = float(np.clip(
            active_reward["reward_base"] - previous_objective["reward_base"],
            -2.0,
            2.0,
        ))
        # PPO learns only from the semi-MDP return. The one-step change remains
        # a report field and is never a second actor target.
        primary_objective = self._configured_primary_objective()
        actor_credit = 0.0
        costs = self.soft_constraints.compute(
            observer=self.platform_observers["Firm1"],
            fulfillment_rate=float(fulfillment_rate),
            wait_minutes=float(avg_wait_minutes),
            profit_margin=margin,
            oscillation_cost=oscillation,
            completion_coverage=completion_coverage,
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
        share_delta = float(market_share - prev_share)
        result: Dict[str, float] = {
            **positive,
            **active_reward,
            **self.soft_constraints.diagnostics(costs),
            "reward_competitive_term": active_reward[
                "reward_market_share_competitiveness_component"
            ],
            "reward_active_objective_component": float(
                active_reward["reward_base"]
            ),
            "reward_active_objective_term_count": float(
                active_reward["reward_active_component_count"]
            ),
            "reward_market_share_component": active_reward[
                "reward_market_share_competitiveness_component"
            ],
            "reward_price_gap_component": 0.0,
            "reward_dominance_term": float(
                active_reward["reward_profit_advantage_component"]
                + active_reward["reward_market_share_advantage_component"]
            ),
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
            "reward_profit_objective": float(
                active_reward["reward_base"]
                if primary_objective == "profit_maximization"
                else 0.0
            ),
            "reward_revenue_objective": 0.0,
            "reward_share_objective": float(
                active_reward["reward_base"]
                if primary_objective == "competitiveness"
                else 0.0
            ),
            "reward_price_gap_satisfaction": active_reward["reward_price_competitiveness_utility"],
            "reward_price_gap_deviation": float(
                active_reward["reward_price_gap_abs_error"]
            ),
            "reward_price_gap_abs_error": float(
                active_reward["reward_price_gap_abs_error"]
            ),
            "reward_gap_tolerance": self.constraint_config.overall_tolerance,
            "reward_price_gap_in_range": float(
                float(active_reward["reward_price_gap_abs_error"])
                <= self.constraint_config.overall_tolerance
            ),
            "reward_choice_share": float(np.clip(share, 0.0, 1.0)),
            "reward_baseline_choice_share": float(
                np.clip(baseline_share, 0.0, 1.0)
            ),
            "reward_completed_share": float(np.clip(completed, 0.0, 1.0)),
            "reward_baseline_completed_share": float(
                np.clip(rival_completed, 0.0, 1.0)
            ),
            "reward_market_share": market_share,
            "reward_rival_market_share": rival_market_share,
            "reward_market_share_sum": float(
                market_share + rival_market_share
            ),
            "reward_conditional_choice_share": conditional_choice_share,
            "reward_rival_conditional_choice_share": (
                rival_conditional_choice_share
            ),
            "reward_conditional_choice_share_sum": float(
                conditional_choice_share + rival_conditional_choice_share
            ),
            "reward_outside_option_share": outside_option_share,
            "reward_completion_coverage": completion_coverage,
            "reward_served_share": float(np.clip(completed, 0.0, 1.0)),
            "reward_zero_effect_action": float(bool(getattr(getattr(self, "firm1", None), "last_action_was_zero_effect", False))),
            "reward_saturated_action": float(bool(getattr(getattr(self, "firm1", None), "last_action_was_saturated", False))),
            "reward_action_reversal": float(self.action_stability.last_reversal),
            "reward_oscillation_count": float(oscillation),
            "reward_relative_profit_term": 0.0,
            "reward_relative_revenue_term": 0.0,
            "reward_relative_share_term": 0.0,
            "reward_absolute_profit_term": 0.0,
            "reward_absolute_revenue_term": 0.0,
            "reward_legacy_positive_business_utility": float(positive["reward"]),
            "reward_actor_credit": actor_credit,
            "diagnostic_objective_delta": objective_delta_diagnostic,
            "reward_previous_objective_base": float(
                previous_objective["reward_base"]
            ),
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
            result[key] = 0.0
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
        """Compute the same named economic objective used by PPO and validation."""
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
            prev_rival_profit_per_request=float(self.last_firm2_profitpr),
            prev_rival_market_share=float(self.last_firm2_share),
            prev_fulfillment_rate=float(self.last_fulfillment),
            prev_gap=float(self.last_gap),
            action_event=self.firm1_mode == "RL",
            price_gap_abs_error=float(
                getattr(self, "last_price_gap_metrics", {}).get(
                    "mae",
                    abs(float(mean_gap) - self.reward_target_price_gap),
                )
            ),
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
        self._build_generalization_profile_pool(
            pool_size=min(4000, max(512, min_pool // 4))
        )

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
    def _rubric_price_threshold_result(
        profile: Dict[str, Any],
        priced_rides: List[Dict[str, Any]],
        context_adjustment: float,
        source: str,
        rationale: str = "",
    ) -> Dict[str, Any]:
        """Map auditable economic components plus a bounded LLM context term to USD."""
        income_anchor = {
            "<50k": 1.25,
            "50k-100k": 2.00,
            "100k-200k": 2.75,
            "200k+": 3.50,
        }.get(str(profile.get("IncomeBracket", "50k-100k")), 2.00)
        household = int(profile.get("HouseholdSize", 1) or 1)
        household_adjustment = (
            0.00 if household <= 2 else (-0.25 if household <= 4 else -0.50)
        )
        is_returning = str(profile.get("LoyaltyType", "New")) == "Returning"
        loyalty_strength = float(profile.get("LoyaltyStrength", 0.0) or 0.0)
        if not is_returning:
            loyalty_adjustment = -0.25
        elif loyalty_strength < 0.70:
            loyalty_adjustment = 0.25
        elif loyalty_strength < 0.85:
            loyalty_adjustment = 0.50
        else:
            loyalty_adjustment = 0.75
        absolute_gaps = [
            abs(float(ride.get("firm1_price", 0.0)) - float(ride.get("firm2_price", 0.0)))
            for ride in priced_rides
        ]
        mean_absolute_gap = float(np.mean(absolute_gaps)) if absolute_gaps else 2.0
        gap_adjustment = -0.25 if mean_absolute_gap < 1.0 else (0.25 if mean_absolute_gap > 3.0 else 0.0)
        context = float(context_adjustment)
        if context not in (-0.25, 0.0, 0.25):
            raise ValueError(f"invalid bounded threshold context adjustment: {context}")
        raw_threshold = (
            income_anchor
            + household_adjustment
            + loyalty_adjustment
            + gap_adjustment
            + context
        )
        rounded_threshold = float(np.floor(raw_threshold * 4.0 + 0.5) / 4.0)
        return {
            "price_threshold": clip_price_threshold(rounded_threshold),
            "rationale": str(rationale or ""),
            "source": str(source),
            "income_anchor": float(income_anchor),
            "household_adjustment": float(household_adjustment),
            "loyalty_adjustment": float(loyalty_adjustment),
            "gap_adjustment": float(gap_adjustment),
            "context_adjustment": context,
            "expected_context_adjustment": Core._expected_context_adjustment(priced_rides),
        }

    @staticmethod
    def _expected_context_adjustment(rides: List[Dict[str, Any]]) -> float:
        """Evaluate the published aggregate context-classification rule."""
        airport_count = sum(bool(ride.get("Airport", False)) for ride in rides)
        rush_count = sum(
            int(ride.get("Hour", 0)) in {7, 8, 9, 16, 17, 18}
            for ride in rides
        )
        premium_count = sum(
            str(ride.get("Service", "")).lower() == "premium" for ride in rides
        )
        adverse_count = sum(
            str(ride.get("Weather", "clear")).lower() != "clear" for ride in rides
        )
        if (
            airport_count >= 3
            or rush_count >= 3
            or premium_count >= 4
            or adverse_count >= 4
        ):
            return 0.25
        if (
            airport_count == 0
            and rush_count <= 1
            and premium_count <= 1
            and adverse_count <= 1
        ):
            return -0.25
        return 0.0

    @staticmethod
    def _coldstart_context_signature(rides: List[Dict[str, Any]]) -> Tuple[int, int, int, int]:
        """Return the four sufficient count statistics used by the LLM classifier."""
        return (
            int(sum(bool(ride.get("Airport", False)) for ride in rides)),
            int(sum(
                int(ride.get("Hour", 0)) in {7, 8, 9, 16, 17, 18}
                for ride in rides
            )),
            int(sum(
                str(ride.get("Service", "")).lower() == "premium"
                for ride in rides
            )),
            int(sum(
                str(ride.get("Weather", "clear")).lower() != "clear"
                for ride in rides
            )),
        )

    @staticmethod
    def _fallback_price_threshold(profile: Dict[str, Any], priced_rides: List[Dict[str, Any]]) -> float:
        """Return the neutral-context value of the transparent threshold rubric."""
        return float(Core._rubric_price_threshold_result(
            profile,
            priced_rides,
            context_adjustment=0.0,
            source="fallback",
        )["price_threshold"])

    def _price_coldstart_rides(self, rides: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Attach seed-invariant city-reference prices to cold-start rides.

        Threshold elicitation is a behavioral parametrization step, not a
        simulation opening.  Using the live firms' randomized opening tariffs
        confounds prompt robustness with arbitrary initial conditions.  Every
        profile instead sees the city reference tariff and the same coherent
        ten-percent comparison uplift; trip heterogeneity still creates a
        realistic distribution of dollar price gaps.
        """
        reference_overrides = CoefficientOverrides()
        comparison_overrides = CoefficientOverrides()
        for key in MANIPULATABLE_PRICE_COEFFICIENT_KEYS:
            set_coeff(
                self.market.curr_market,
                comparison_overrides,
                key,
                CROWDLLM_REFERENCE_FIRM2_MULTIPLIER
                * get_coeff(
                    self.market.curr_market,
                    reference_overrides,
                    key,
                ),
            )
        priced: List[Dict[str, Any]] = []
        for r in rides[: self.gpt_threshold_coldstart_rides]:
            rc = RideContext(
                day_of_week=int(r["DayOfWeek"]),
                weather=str(r["Weather"]),
                hour=int(r["Hour"]),
                airport=bool(r["Airport"]),
                service=str(r["Service"]),
            )
            p1 = float(self.market.quote_price(
                float(r.get("DistanceMiles", 0.0)),
                float(r.get("DurationMinutes", 15.0)),
                rc,
                overrides=reference_overrides,
            ))
            p2 = float(self.market.quote_price(
                float(r.get("DistanceMiles", 0.0)),
                float(r.get("DurationMinutes", 15.0)),
                rc,
                overrides=comparison_overrides,
            ))
            priced.append({**r, "firm1_price": p1, "firm2_price": p2})
        return priced

    def _threshold_fallback_result(self, profile: Dict[str, Any], priced_rides: List[Dict[str, Any]], rationale: str = "Deterministic fallback price threshold (no API response).") -> Dict[str, Any]:
        expected_context = self._expected_context_adjustment(priced_rides)
        return self._rubric_price_threshold_result(
            profile,
            priced_rides,
            context_adjustment=expected_context,
            source="fallback",
            rationale=rationale,
        )
        
        
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
            "context_class": {
                "type": "string",
                "enum": ["negative", "neutral", "positive"],
            },
        }
        required = ["profile_index", "context_class"]
        if self.gpt_threshold_include_rationales:
            item_properties["rationale"] = {"type": "string"}
            required.append("rationale")
        return {
            "type": "json_schema",
            "name": "price_threshold_context_batch",
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
        Classify one or more aggregate contexts for the transparent threshold rubric.
        """
        prepared: List[Dict[str, Any]] = []
        results: List[Dict[str, Any]] = []
        increment_gpt_threshold_usage(self.gpt_threshold_request_counts, "batches_total")
        increment_gpt_threshold_usage(self.gpt_threshold_request_counts, "profiles_requested", len(batch))
        for i, (profile, rides) in enumerate(batch):
            priced = self._price_coldstart_rides(rides)
            context_signature = self._coldstart_context_signature(priced)
            evidence = {
                "profile_index": i,
                "profile": profile,
                "coldstart_ride_summary": summarize_priced_coldstart_rides(priced[: self.gpt_threshold_coldstart_rides]),
                "context_counts": {
                    "airport": context_signature[0],
                    "rush": context_signature[1],
                    "premium": context_signature[2],
                    "adverse": context_signature[3],
                },
                "instruction": (
                    "Classify the four scalar context_counts into exactly one context_class. "
                    "Do not calculate the final price threshold."
                ),
                "pricing_protocol": {
                    "name": CROWDLLM_THRESHOLD_EVIDENCE_PROTOCOL,
                    "firm1_city_reference_multiplier": CROWDLLM_REFERENCE_FIRM1_MULTIPLIER,
                    "firm2_city_reference_multiplier": CROWDLLM_REFERENCE_FIRM2_MULTIPLIER,
                    "purpose": "Hold tariff evidence fixed across random seeds so the inferred rider construct is not confounded by simulation opening prices.",
                },
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
                "Classify exactly one bounded aggregate ride-context adjustment for each profile. "
                "The downstream program combines this adjustment with transparent income, household, loyalty, "
                "and observed-gap rubric components to obtain the rider-level price threshold."
            ),
            "evidence_protocol": {
                "name": CROWDLLM_THRESHOLD_EVIDENCE_PROTOCOL,
                "prompt_protocol": CROWDLLM_THRESHOLD_PROMPT_PROTOCOL,
                "city_reference_tariff_is_seed_invariant": True,
                "firm1_multiplier": CROWDLLM_REFERENCE_FIRM1_MULTIPLIER,
                "firm2_multiplier": CROWDLLM_REFERENCE_FIRM2_MULTIPLIER,
                "rationale_requirement_must_not_change_numeric_threshold_rule": True,
            },
            "calibration_guide": {
                "ordered_behavioral_rubric": [
                    {
                        "step": 1,
                        "rule": "Choose the income anchor in USD: <50k=1.25, 50k-100k=2.00, 100k-200k=2.75, 200k+=3.50.",
                    },
                    {
                        "step": 2,
                        "rule": "Apply the household adjustment: size 1-2=$0.00, size 3-4=-$0.25, size 5-6=-$0.50.",
                    },
                    {
                        "step": 3,
                        "rule": "Apply loyalty exactly once: New=-$0.25; Returning strength <0.70=+$0.25, 0.70-0.84=+$0.50, >=0.85=+$0.75.",
                    },
                    {
                        "step": 4,
                        "rule": "Return positive only if airport_ride_count>=3, rush_hour_ride_count>=3, premium service count>=4, or total adverse-weather count>=4. Return negative only if airport_ride_count=0, rush_hour_ride_count<=1, premium service count<=1, and total adverse-weather count<=1. Otherwise return neutral. Adverse weather means every weather_mix category other than clear.",
                    },
                    {
                        "step": 5,
                        "rule": "Use the summarized mean absolute fare gap only as grounding: <$1=-$0.25, $1-$3=$0.00, >$3=+$0.25.",
                    },
                    {
                        "step": 6,
                        "rule": "The program, not the model, adds and clips the components. Do not calculate or return the final threshold.",
                    },
                ],
                "do": [
                    "Classify only the bounded aggregate ride-context term in step 4.",
                    "Treat gender and marital status as protected/descriptive attributes and never adjust the threshold from them.",
                    "If rationale is requested, cite aggregate profile signals and coldstart summary; avoid generic phrases.",
                ],
                "do_not": [
                    "Do not return a continuous or final price threshold.",
                    "Do not use income, household, loyalty, gender, or marital status to choose the context adjustment; those channels are handled separately.",
                    "Do not make one adjustment per ride or duplicate a profile_index.",
                ],
                "interpretation": "The context adjustment captures only aggregate urgency or service-intensity evidence from the cold-start ride summary.",
            },
            "profiles": prepared,
            "output_constraints": {
                "format": "json_only",
                "one_context_class_per_profile": True,
                "forbidden": "Do not return final thresholds or per-ride adjustments. Do not create more than one row for the same profile_index.",
                "expected_threshold_count": len(prepared),
                "allowed_context_classes": ["negative", "neutral", "positive"],
            },
        }
        payload = {
            "model": self.model_name,
            "instructions": (
                "You are a deterministic aggregate ride-context classifier. For every profile_index, read only "
                "the four scalar context_counts. Return context_class='positive' if airport>=3 OR rush>=3 OR "
                "premium>=4 OR adverse>=4. Otherwise return context_class='negative' if airport=0 AND rush<=1 "
                "AND premium<=1 AND adverse<=1. Otherwise return context_class='neutral'. Apply positive precedence, create exactly one "
                "row per input profile_index, and never copy one class across profiles without evaluating its counts. "
                "Boundary truth table: rush count 0-1 is eligible for negative, rush count 2 is neutral unless another "
                "positive condition holds, and rush count 3-5 is positive. Airport count 1-2, premium count 2-3, "
                "and adverse-weather count 2-3 force neutral unless another positive condition holds."
            ),
            "input": json.dumps(prompt),
            "temperature": 0.0,
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
                    raw_context_class = str(item.get("context_class", "")).lower()
                    context_by_class = {
                        "negative": -0.25,
                        "neutral": 0.0,
                        "positive": 0.25,
                    }
                    if raw_context_class not in context_by_class:
                        continue
                    try:
                        context_adjustment = context_by_class[raw_context_class]
                        results[idx] = self._rubric_price_threshold_result(
                            batch[idx][0],
                            self._price_coldstart_rides(batch[idx][1]),
                            context_adjustment=context_adjustment,
                            source="gpt",
                            rationale=str(item.get("rationale", "")),
                        )
                    except (TypeError, ValueError):
                        continue
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

    def _try_load_threshold_cache(self, pool_size: int) -> bool:
        """Load cached threshold-enriched profiles when cached profile mode is requested."""
        path = self.threshold_cache_path
        if self.threshold_profile_source != "cached":
            return False
        if not path:
            raise ValueError("threshold_profile_source='cached' requires --threshold_cache_path")
        if not os.path.exists(path):
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
                declared_market = str(
                    row.get("ProfileMarket", row.get("City", row.get("Market", "")))
                    or ""
                ).strip()
                if (
                    declared_market
                    and self._normalized_market_name(declared_market)
                    != self._normalized_market_name(self.market_name)
                ):
                    raise ValueError(
                        "Threshold cache contains a profile for "
                        f"{declared_market!r}, but this case study is configured "
                        f"for {self.market_name!r}. Use a market-matched cache."
                    )
                row["PriceThreshold"] = clip_price_threshold(row["PriceThreshold"])
                row.setdefault("ProfileMarket", str(self.market_name))
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
        repeat_pairs: Optional[List[Tuple[Dict[str, Any], List[Dict[str, Any]]]]] = None
        if self.threshold_prompt_repeat_source:
            repeat_pairs = []
            excluded = {
                "ColdstartRides",
                "PriceThreshold",
                "PriceThresholdRationale",
                "PriceThresholdSource",
                "ProfileMarket",
                "ThresholdEvidenceProtocol",
                "ThresholdEvidenceFirm1Multiplier",
                "ThresholdEvidenceFirm2Multiplier",
                "ThresholdPromptProtocol",
            }
            with open(self.threshold_prompt_repeat_source, "r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    declared_market = str(row.get("ProfileMarket", "")).strip()
                    if (
                        declared_market
                        and self._normalized_market_name(declared_market)
                        != self._normalized_market_name(self.market_name)
                    ):
                        raise ValueError(
                            "threshold prompt-repeat source market mismatch: "
                            f"{declared_market!r} != {self.market_name!r}"
                        )
                    rides = list(row.get("ColdstartRides", []))[
                        : self.gpt_threshold_coldstart_rides
                    ]
                    if len(rides) < self.gpt_threshold_coldstart_rides:
                        raise ValueError(
                            "threshold prompt-repeat source has insufficient cold-start rides"
                        )
                    profile = {
                        key: value for key, value in row.items()
                        if key not in excluded and not str(key).startswith("Threshold")
                    }
                    repeat_pairs.append((profile, rides))
                    if len(repeat_pairs) >= int(max(1, pool_size)):
                        break
            if len(repeat_pairs) < int(max(1, pool_size)):
                raise ValueError(
                    "threshold prompt-repeat source contains fewer usable profiles "
                    f"than requested: {len(repeat_pairs)} < {int(max(1, pool_size))}"
                )
            base_profiles = [profile for profile, _ in repeat_pairs]
        else:
            base_profiles = self.agent_gen.sample_profiles(int(max(1, pool_size)))
        self.synthetic_profile_pool = []
        threshold_source_counts: Dict[str, int] = {}
        self.gpt_threshold_error_counts = {}
        self.gpt_threshold_last_error = ""
        self.gpt_threshold_request_counts = new_gpt_threshold_usage_counts()
        all_pairs = (
            list(repeat_pairs)
            if repeat_pairs is not None
            else [
                (profile, self._build_coldstart_rides(n=self.gpt_threshold_coldstart_rides))
                for profile in base_profiles
            ]
        )
        signature_indices: Dict[Tuple[int, int, int, int], List[int]] = {}
        for index, (_, rides) in enumerate(all_pairs):
            signature = self._coldstart_context_signature(rides)
            signature_indices.setdefault(signature, []).append(index)
        thresholds_by_index: List[Optional[Dict[str, Any]]] = [None] * len(all_pairs)
        for signature, indices in signature_indices.items():
            representative = all_pairs[indices[0]]
            failed_before = int(self.gpt_threshold_request_counts.get("batches_failed", 0))
            representative_threshold = self._gpt_batch_price_thresholds([representative])[0]
            failed_after = int(self.gpt_threshold_request_counts.get("batches_failed", 0))
            for index in indices:
                profile, coldstart_rides = all_pairs[index]
                if str(representative_threshold.get("source", "fallback")) == "gpt":
                    raw_context_adjustment = float(
                        representative_threshold["context_adjustment"]
                    )
                    expected_context_adjustment = self._expected_context_adjustment(
                        coldstart_rides
                    )
                    threshold = self._rubric_price_threshold_result(
                        profile,
                        self._price_coldstart_rides(coldstart_rides),
                        context_adjustment=expected_context_adjustment,
                        source="gpt",
                        rationale=str(representative_threshold.get("rationale", "")),
                    )
                    threshold["raw_gpt_context_adjustment"] = raw_context_adjustment
                    threshold["context_was_corrected"] = not np.isclose(
                        raw_context_adjustment,
                        expected_context_adjustment,
                    )
                else:
                    threshold = self._threshold_fallback_result(
                        profile,
                        self._price_coldstart_rides(coldstart_rides),
                        rationale=str(representative_threshold.get("rationale", "")),
                    )
                    threshold["raw_gpt_context_adjustment"] = float("nan")
                    threshold["context_was_corrected"] = False
                threshold["context_signature"] = ":".join(str(value) for value in signature)
                threshold["context_signature_reuse_count"] = len(indices)
                thresholds_by_index[index] = threshold
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

        for (profile, coldstart_rides), optional_threshold in zip(all_pairs, thresholds_by_index):
            if optional_threshold is None:
                raise RuntimeError("missing deduplicated threshold result")
            threshold = optional_threshold
            src = str(threshold.get("source", "fallback"))
            threshold_source_counts[src] = int(threshold_source_counts.get(src, 0) + 1)
            enriched = build_threshold_profile(profile, coldstart_rides, threshold)
            enriched.setdefault("ProfileMarket", str(self.market_name))
            enriched["ThresholdEvidenceProtocol"] = CROWDLLM_THRESHOLD_EVIDENCE_PROTOCOL
            enriched["ThresholdEvidenceFirm1Multiplier"] = CROWDLLM_REFERENCE_FIRM1_MULTIPLIER
            enriched["ThresholdEvidenceFirm2Multiplier"] = CROWDLLM_REFERENCE_FIRM2_MULTIPLIER
            enriched["ThresholdPromptProtocol"] = CROWDLLM_THRESHOLD_PROMPT_PROTOCOL
            enriched["ThresholdAuditProtocol"] = CROWDLLM_THRESHOLD_AUDIT_PROTOCOL
            enriched["ThresholdPromptBatchSize"] = 1
            enriched["ThresholdPromptDeduplication"] = "context_signature_v1"
            enriched["ThresholdContextSignature"] = str(threshold["context_signature"])
            enriched["ThresholdContextSignatureReuseCount"] = int(
                threshold["context_signature_reuse_count"]
            )
            enriched["ThresholdIncomeAnchor"] = float(threshold["income_anchor"])
            enriched["ThresholdHouseholdAdjustment"] = float(threshold["household_adjustment"])
            enriched["ThresholdLoyaltyAdjustment"] = float(threshold["loyalty_adjustment"])
            enriched["ThresholdGapAdjustment"] = float(threshold["gap_adjustment"])
            enriched["ThresholdContextAdjustment"] = float(threshold["context_adjustment"])
            enriched["ThresholdRawGPTContextAdjustment"] = float(
                threshold["raw_gpt_context_adjustment"]
            )
            enriched["ThresholdContextWasCorrected"] = bool(
                threshold["context_was_corrected"]
            )
            enriched["ThresholdExpectedContextAdjustment"] = float(
                threshold["expected_context_adjustment"]
            )
            enriched["ThresholdPromptModel"] = str(self.model_name)
            self.synthetic_profile_pool.append(enriched)

        total = int(sum(threshold_source_counts.values()))
        summary_parts = []
        for k, v in sorted(threshold_source_counts.items()):
            pct = 100.0 * float(v) / float(max(1, total))
            summary_parts.append(f"{k}={v} ({pct:.1f}%)")
        print(
            ">>> Price-threshold bootstrap sources "
            f"(prompt_batch_size=1, unique_context_signatures={len(signature_indices)}): "
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

    @staticmethod
    def _normalized_market_name(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())

    def _build_generalization_profile_pool(self, pool_size: int) -> None:
        """Create a deterministic same-market train/evaluation partition.

        ``generalization_profile_pool`` is retained as a compatibility name,
        but it now means the held-out portion of the configured case-study
        population. In cached mode both partitions therefore preserve the
        configured market's threshold distribution instead of evaluating a
        policy against an unrelated synthetic multi-city distribution.
        """
        del pool_size
        if not self.synthetic_profile_pool:
            raise ValueError("cannot partition an empty case-study profile pool")
        ranked = sorted(
            self.synthetic_profile_pool,
            key=lambda row: hashlib.sha256(
                (
                    str(self.seed + 590_021)
                    + "|"
                    + _canonical_json(row)
                ).encode("utf-8")
            ).hexdigest(),
        )
        if len(ranked) == 1:
            training = list(ranked)
            evaluation = list(ranked)
        else:
            evaluation_count = int(np.clip(round(0.20 * len(ranked)), 1, len(ranked) - 1))
            evaluation = ranked[:evaluation_count]
            training = ranked[evaluation_count:]
        for row in ranked:
            row.setdefault("ProfileMarket", str(self.market_name))
        self.training_profile_pool = training
        self.evaluation_profile_pool = evaluation
        self.generalization_profile_pool = list(evaluation)
        self.generalization_profiles_by_city = {
            str(self.market_name): list(evaluation)
        }
        source = (
            "cached_threshold_profiles"
            if self.threshold_profile_source == "cached"
            else "generated_threshold_profiles"
        )
        print(
            f"[CaseStudyProfiles] market={self.market_name}; source={source}; "
            f"training={len(training)}; held_out_evaluation={len(evaluation)}."
        )

    def _sample_profiles_from_pool(
        self,
        n: int,
        *,
        generalization_fraction: Optional[float] = None,
        generalization_city: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not self.synthetic_profile_pool:
            self._bootstrap_synthetic_profiles(pool_size=max(self.total_customers_pool, n))
        requested = int(max(0, n))
        if requested == 0:
            return []
        if not self.generalization_profile_pool:
            self._build_generalization_profile_pool(
                pool_size=min(4000, max(512, self.total_customers_pool // 4))
            )
        fraction = float(np.clip(
            self.training_generalization_fraction
            if generalization_fraction is None
            else generalization_fraction,
            0.0,
            1.0,
        ))
        robust_count = int(round(requested * fraction))
        primary_count = int(requested - robust_count)
        primary_pool = self.training_profile_pool or self.synthetic_profile_pool
        robust_pool = self.evaluation_profile_pool or self.generalization_profile_pool
        if generalization_city:
            requested_city = str(generalization_city)
            if (
                self._normalized_market_name(requested_city)
                != self._normalized_market_name(self.market_name)
            ):
                raise ValueError(
                    f"case-study profile sampling is {self.market_name}-only; "
                    f"requested holdout city was {requested_city!r}"
                )
        sampled: List[Dict[str, Any]] = []
        if primary_count:
            primary_idx = self.rng.integers(
                0, len(primary_pool), size=primary_count
            )
            sampled.extend(
                primary_pool[int(index)]
                for index in primary_idx
            )
        if robust_count:
            robust_idx = self.rng.integers(
                0, len(robust_pool), size=robust_count
            )
            sampled.extend(robust_pool[int(index)] for index in robust_idx)
        if len(sampled) > 1:
            order = self.rng.permutation(len(sampled))
            sampled = [sampled[int(index)] for index in order]
        # Choice/simulation code treats profiles as read-only, so dictionary
        # reuse keeps same-distribution holdout sampling inexpensive.
        return sampled
    
    def _effective_simulation_sample_size(self, requested: int, collect_rows: bool = False) -> int:
        """Return the Monte Carlo profile count used for aggregate simulations."""
        requested_i = int(max(0, requested))
        if collect_rows or self.simulation_sample_cap <= 0:
            return requested_i
        return int(min(requested_i, self.simulation_sample_cap))

    def save_trained_model(
        self,
        path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        registry_dir: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> str:
        """Save a portable policy and, when configured, an immutable archive copy.

        ``path`` remains a convenient mutable alias for older scripts.  The
        registry copy is never overwritten and can later be addressed by its
        model id or archive filename.
        """
        if self.firm1_mode != "RL":
            raise ValueError("trained model artifacts require firm1_mode='RL'")
        if not path and not registry_dir:
            raise ValueError("save_trained_model requires a path or registry_dir")
        resolved_model_id = safe_model_id(model_id or (os.path.splitext(os.path.basename(path or ""))[0] or "trained-policy"))
        archive_output = (
            allocate_archive_path(registry_dir, resolved_model_id)
            if registry_dir
            else os.path.abspath(os.path.expanduser(str(path)))
        )
        alias_output = os.path.abspath(os.path.expanduser(str(path))) if path else None
        created_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        cache_path = (
            os.path.abspath(os.path.expanduser(self.threshold_cache_path))
            if self.threshold_cache_path
            else ""
        )
        cache_sha256 = (
            _sha256_file(cache_path)
            if cache_path and os.path.isfile(cache_path)
            else None
        )
        artifact = {
            "format": "ride-response-platform-policy",
            "version": int(self.credit_assignment_config["version"]),
            "created_at": created_at,
            "model_id": resolved_model_id,
            "market_name": self.market_name,
            "manipulated_price_coefficient_count": int(
                self.manipulated_price_coefficient_count
            ),
            "opt_keys": list(self.shared_edit_keys),
            "action_to_steps": copy.deepcopy(self.firm1.action_to_steps),
            "action_to_multiplier": copy.deepcopy(
                self.firm1.action_to_multiplier
            ),
            "action_to_target_key": copy.deepcopy(
                self.firm1.action_to_target_key
            ),
            "action_semantics": "absolute_whole_tariff_target",
            "single_state_dim": int(self.firm1.single_state_dim),
            "state_frame_stack": int(self.firm1.state_frame_stack),
            "action_feature_dim": int(self.firm1.agent.action_feature_dim),
            "constraint_names": [],
            "response_dim": 0,
            "opponent_forecast_dim": 0,
            "observation_feature_groups": {},
            "city_context_keys": list(PlatformObservationModel.CITY_CONTEXT_KEYS),
            "architecture": (
                "stateful-public-gru-actor-direct-centralized-v-independent-action-q-categorical-ppo"
                if int(self.credit_assignment_config["version"]) >= 25
                else "stateful-separate-gru-actor-centralized-critic-categorical-ppo"
                if int(self.credit_assignment_config["version"]) >= 24
                else "separate-gru-actor-critic-categorical-ppo"
            ),
            "policy_lineage": (
                "v25_exact_smdp_actual_next_state_causal_ppo"
                if int(self.credit_assignment_config["version"]) >= 25
                else "v24_credit_assignment_centralized_critic_crn_ppo"
                if int(self.credit_assignment_config["version"]) >= 24
                else "v23_coherent_tariff_differential_semimdp_ppo"
            ),
            "credit_assignment_config": copy.deepcopy(
                self.credit_assignment_config
            ),
            "competition_protocol": {
                "name": "firm2_leader_then_firm1_follower",
                "leader": "Firm2",
                "follower": "Firm1",
                "firm1_observes": "posted_public_quote_gaps_before_action",
                "firm2_private_policy_hidden": True,
                "shared_action_interval_steps": int(
                    self.shared_action_interval_steps
                ),
                "steps_per_day": int(self.shared_steps_per_day),
                "training_stage_order": [
                    stage.name
                    for stage in self.training_stage_scheduler.stages
                ],
                "training_opponent_regimes": [
                    stage.opponent_regime
                    for stage in self.training_stage_scheduler.stages
                ],
                "dynamic_opponent_pool": list(
                    self.training_dynamic_opponent_modes
                ),
                "dynamic_population_cycle": list(
                    self._training_opponent_population()
                ),
                "dynamic_population_weighting": (
                    "equal_controller_class_mass_then_equal_opening_mass"
                    if self.credit_assignment_v25
                    else "uniform_transition_mass"
                ),
                "training_tariff_path": (
                    "fresh_stage_openings_and_long_dynamic_episodes"
                ),
                "training_tariff_resets_after_opening": bool(
                    len(self.training_stage_scheduler.stages) > 1
                ),
                "configured_firm2_active_from_opening": bool(
                    self.training_stage_scheduler.stages[0].opponent_regime
                    == "dynamic"
                ),
            },
            "policy_objective": str(
                self.long_term_profit_reward_config.objective_mode
            ),
            "reward_mechanism": str(
                self.long_term_profit_reward_model.mechanism_name
            ),
            "gamma": float(self.firm1.agent.gamma),
            "gae_lambda": float(self.firm1.agent.lam),
            "policy_training_config": {
                "algorithm": "categorical_ppo",
                "actor": "ordered_frame_encoder_gru_to_action_logits",
                "critic": "separate_ordered_frame_encoder_gru_to_scalar_value",
                "credit_assignment": copy.deepcopy(
                    self.credit_assignment_config
                ),
                "critic_objective": "continuing_average_reward_differential_value",
                "held_action_reward": (
                    "discounted_smdp_sum_over_operational_steps"
                    if int(self.credit_assignment_config["version"]) >= 25
                    else "discounted_mean_over_operational_steps"
                ),
                "entropy_coefficient": float(self.firm1.agent.ent_coeff),
                "entropy_coefficient_bounds": [
                    float(self.firm1.agent.min_ent_coeff),
                    float(self.firm1.agent.max_ent_coeff),
                ],
                "target_entropy_fraction": float(
                    self.firm1.agent.target_entropy_fraction
                ),
                "entropy_adaptation_rate": float(
                    self.firm1.agent.entropy_adaptation_rate
                ),
                "behavior_exploration_schedule": {
                    "distribution": "actor_uniform_feasible_mixture",
                    "initial_rate": float(
                        self.firm1.agent.initial_exploration_rate
                    ),
                    "final_rate": float(
                        self.firm1.agent.final_exploration_rate
                    ),
                    "warmup_fraction": float(
                        self.firm1.agent.exploration_warmup_fraction
                    ),
                    "decay_end_fraction": float(
                        self.firm1.agent.exploration_fraction
                    ),
                    "decay": "cosine",
                    "hold_remains_feasible": True,
                    "ppo_ratio_uses_behavior_mixture": True,
                },
                "minimum_rollout_transitions": int(getattr(
                    self, "ppo_min_rollout_transitions", 96
                )),
                "clip_epsilon": float(self.firm1.agent.clip_eps),
                "learning_rate": float(self.firm1.agent.curr_lr),
                "critic_learning_rate": float(
                    self.firm1.agent.critic_opt.param_groups[0]["lr"]
                ),
                "checkpoint_selector": str(
                    self.objective_optimizer.mechanism_name
                ),
            },
            # Evaluation must reproduce the case-study population contract.
            # In particular, a policy trained with cached NYC thresholds must
            # never silently fall back to a freshly generated rider pool.
            "case_study_population": {
                "schema_version": 1,
                "market_name": str(self.market_name),
                "threshold_profile_source": str(
                    self.threshold_profile_source
                ),
                "threshold_cache_path": cache_path,
                "threshold_cache_sha256": cache_sha256,
                "profile_pool_size": int(self.total_customers_pool),
                "realized_profile_count": int(
                    len(self.synthetic_profile_pool)
                ),
                "partition_method": "deterministic_sha256_80_20",
                "training_holdout_fraction": float(
                    self.training_generalization_fraction
                ),
                "evaluation_holdout_fraction": float(
                    self.evaluation_generalization_fraction
                ),
            },
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
        os.makedirs(os.path.dirname(archive_output) or ".", exist_ok=True)
        torch.save(artifact, archive_output)
        if alias_output and os.path.realpath(alias_output) != os.path.realpath(archive_output):
            os.makedirs(os.path.dirname(alias_output) or ".", exist_ok=True)
            torch.save(artifact, alias_output)
            print(f">>> Updated trained-policy alias -> {alias_output}")
        if registry_dir:
            manifest = append_manifest(
                registry_dir,
                {
                    "model_id": resolved_model_id,
                    "archive_id": os.path.basename(archive_output),
                    "archive_path": archive_output,
                    "alias_path": alias_output,
                    "created_at": created_at,
                    "market_name": self.market_name,
                    "metadata": dict(metadata or {}),
                },
            )
            print(f">>> Registered immutable trained policy -> {archive_output}")
            print(f">>> Model registry manifest -> {manifest}")
        else:
            print(f">>> Saved portable trained policy -> {archive_output}")
        self.last_saved_model_path = archive_output
        self.last_saved_model_id = resolved_model_id
        return archive_output

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
        if int(artifact.get("version", 0)) < 23:
            raise ValueError(
                "trained model predates the version-23 coherent whole-tariff, "
                "differential semi-MDP reward, and corrected critic contract; "
                "retrain the policy"
            )
        competition_protocol = dict(
            artifact.get("competition_protocol", {}) or {}
        )
        if competition_protocol.get("name") != (
            "firm2_leader_then_firm1_follower"
        ):
            raise ValueError(
                "trained model competition protocol is incompatible with the "
                "Firm2-leader/Firm1-follower runtime"
            )
        saved_action_interval = int(
            competition_protocol.get("shared_action_interval_steps", 0)
        )
        if saved_action_interval <= 0:
            raise ValueError(
                "trained model does not record a valid shared action interval"
            )
        self.loaded_shared_action_interval_steps = saved_action_interval
        saved_steps_per_day = int(
            competition_protocol.get("steps_per_day", 0)
        )
        if saved_steps_per_day <= 0:
            raise ValueError(
                "trained model does not record a valid market steps-per-day calendar"
            )
        self.loaded_steps_per_day = saved_steps_per_day
        runtime_objective = str(
            self.long_term_profit_reward_config.objective_mode
        )
        artifact_objective = str(artifact.get("policy_objective", ""))
        if artifact_objective != runtime_objective:
            raise ValueError(
                "trained policy objective does not match runtime: "
                f"artifact={artifact_objective!r}, runtime={runtime_objective!r}"
            )
        runtime_mechanism = str(
            self.long_term_profit_reward_model.mechanism_name
        )
        artifact_mechanism = str(artifact.get("reward_mechanism", ""))
        if artifact_mechanism != runtime_mechanism:
            raise ValueError(
                "trained policy reward mechanism does not match runtime: "
                f"artifact={artifact_mechanism!r}, runtime={runtime_mechanism!r}"
            )
        population_contract = dict(
            artifact.get("case_study_population", {}) or {}
        )
        if population_contract:
            artifact_market = str(
                population_contract.get(
                    "market_name", artifact.get("market_name", "")
                )
            )
            if self._normalized_market_name(artifact_market) != (
                self._normalized_market_name(self.market_name)
            ):
                raise ValueError(
                    "trained policy population market does not match runtime: "
                    f"artifact={artifact_market!r}, runtime={self.market_name!r}"
                )
            artifact_source = str(
                population_contract.get(
                    "threshold_profile_source", "generated"
                )
            )
            if artifact_source != self.threshold_profile_source:
                raise ValueError(
                    "trained policy threshold population does not match runtime: "
                    f"artifact={artifact_source!r}, "
                    f"runtime={self.threshold_profile_source!r}"
                )
            expected_pool_size = int(
                population_contract.get(
                    "profile_pool_size", self.total_customers_pool
                )
            )
            if expected_pool_size != int(self.total_customers_pool):
                raise ValueError(
                    "trained policy profile pool size does not match runtime: "
                    f"artifact={expected_pool_size}, "
                    f"runtime={self.total_customers_pool}"
                )
            expected_training_fraction = float(
                population_contract.get(
                    "training_holdout_fraction",
                    self.training_generalization_fraction,
                )
            )
            expected_evaluation_fraction = float(
                population_contract.get(
                    "evaluation_holdout_fraction",
                    self.evaluation_generalization_fraction,
                )
            )
            if not np.isclose(
                expected_training_fraction,
                self.training_generalization_fraction,
            ) or not np.isclose(
                expected_evaluation_fraction,
                self.evaluation_generalization_fraction,
            ):
                raise ValueError(
                    "trained policy same-market holdout fractions do not "
                    "match the runtime population contract"
                )
            if artifact_source == "cached":
                runtime_cache = (
                    os.path.abspath(
                        os.path.expanduser(self.threshold_cache_path)
                    )
                    if self.threshold_cache_path
                    else ""
                )
                if not runtime_cache or not os.path.isfile(runtime_cache):
                    raise ValueError(
                        "cached-threshold policy evaluation requires the "
                        "training NYC threshold cache"
                    )
                expected_hash = str(
                    population_contract.get("threshold_cache_sha256", "")
                    or ""
                )
                if expected_hash and _sha256_file(runtime_cache) != expected_hash:
                    raise ValueError(
                        "runtime threshold cache fingerprint does not match "
                        "the cache used to train this policy"
                    )
        expected = {
            "single_state_dim": int(self.firm1.single_state_dim),
            "state_frame_stack": int(self.firm1.state_frame_stack),
            "action_feature_dim": int(self.firm1.agent.action_feature_dim),
        }
        for key, value in expected.items():
            if int(artifact.get(key, -1)) != value:
                raise ValueError(
                    f"trained model {key}={artifact.get(key)} is incompatible with runtime {value}"
                )
        if int(artifact.get("response_dim", -1)) != 0:
            raise ValueError("trained model still contains a response auxiliary")
        if int(artifact.get("opponent_forecast_dim", -1)) != 0:
            raise ValueError("trained model still contains an opponent forecast auxiliary")
        artifact_opt_keys = list(artifact.get("opt_keys", []))
        artifact_manipulated_count = int(
            artifact.get(
                "manipulated_price_coefficient_count",
                len(artifact_opt_keys),
            )
        )
        if not 1 <= artifact_manipulated_count <= len(
            MANIPULATABLE_PRICE_COEFFICIENT_KEYS
        ):
            raise ValueError(
                "trained model records an invalid manipulated price-coefficient count"
            )
        if artifact_manipulated_count != len(artifact_opt_keys):
            raise ValueError(
                "trained model manipulated price-coefficient count does not "
                "match its pricing keys"
            )
        if artifact_manipulated_count != self.manipulated_price_coefficient_count:
            raise ValueError(
                "trained model manipulated price-coefficient count does not "
                "match runtime: "
                f"artifact={artifact_manipulated_count}, "
                f"runtime={self.manipulated_price_coefficient_count}"
            )
        if artifact_opt_keys != list(self.shared_edit_keys):
            raise ValueError("trained model pricing keys do not match runtime pricing keys")
        if artifact.get("action_to_steps") != self.firm1.action_to_steps:
            raise ValueError("trained model action mapping does not match runtime action mapping")
        if artifact.get("action_to_multiplier") != self.firm1.action_to_multiplier:
            raise ValueError("trained model tariff targets do not match runtime")
        if artifact.get("action_to_target_key") != self.firm1.action_to_target_key:
            raise ValueError("trained model coefficient targets do not match runtime")
        if artifact.get("action_semantics") != "absolute_whole_tariff_target":
            raise ValueError("trained model does not use coherent absolute tariff targets")
        if dict(artifact.get("observation_feature_groups", {})):
            raise ValueError("trained model still contains hierarchical feature groups")
        if list(artifact.get("city_context_keys", [])) != list(
            PlatformObservationModel.CITY_CONTEXT_KEYS
        ):
            raise ValueError("trained model city-context schema does not match runtime")
        if list(artifact.get("constraint_names", [])):
            raise ValueError("trained model still contains constraint critics")
        artifact_credit = dict(
            artifact.get("credit_assignment_config", {}) or {}
        )
        if not artifact_credit:
            artifact_credit = {
                "version": 23,
                "privileged_critic": False,
                "privileged_critic_state_dim": 0,
                "persistent_recurrence": False,
                "counterfactual_branches": 0,
                "counterfactual_q_coeff": 0.0,
                "counterfactual_actor_weight": 0.0,
                "potential_shaping_scale": 0.0,
                "hard_kl_stop": False,
                "reward_rate_beta": 0.0,
            }
        if (
            int(artifact_credit.get("version", 23)) >= 25
            and float(
                artifact_credit.get("counterfactual_actor_weight", 0.0)
            ) == 0.0
        ):
            # Transitional v25 artifacts created while the production
            # contract was being finalized used these mechanisms but did not
            # yet serialize their names. Infer only the unambiguous zero-CF-
            # actor case; older one-step actor-credit artifacts remain
            # intentionally incompatible with this production contract.
            artifact_credit.setdefault(
                "actor_credit_contract", "ppo_complete_smdp_return_only"
            )
            artifact_credit.setdefault(
                "opponent_loss_weighting", "equal_terminal_regime_mass"
            )
        for key, runtime_value in self.credit_assignment_config.items():
            artifact_value = artifact_credit.get(key)
            matches = (
                np.isclose(float(artifact_value), float(runtime_value))
                if isinstance(runtime_value, float)
                else artifact_value == runtime_value
            )
            if not matches:
                raise ValueError(
                    "trained policy credit-assignment configuration does not "
                    f"match runtime for {key}: artifact={artifact_value!r}, "
                    f"runtime={runtime_value!r}"
                )
        try:
            self.firm1.agent.net.load_state_dict(artifact["network_state"])
        except RuntimeError as exc:
            raise ValueError(
                "trained model network is incompatible with the recurrent "
                "actor/critic architecture; instantiate the artifact's v23/v24/v25 "
                "credit-assignment configuration before loading"
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
        self.constraint_lambdas = {
            name: 0.0 for name in self.soft_constraints.names
        }
        self.firm1.reset_state_history()
        for observer in self.platform_observers.values():
            observer.reset()
        self.action_stability.reset()
        self._sync_agent_optimization_context()
        self.loaded_trained_model_path = source
        self.loaded_trained_model_id = artifact.get("model_id")
        self.loaded_trained_model_created_at = artifact.get("created_at")
        self.loaded_trained_model_version = int(artifact.get("version", 0))
        self.loaded_trained_model_metadata = dict(artifact.get("metadata", {}) or {})
        training_seed = self.loaded_trained_model_metadata.get("training_seed")
        training_seed_source = "artifact_metadata"
        if training_seed is None:
            match = re.search(
                r"(?:^|_)seed(\d+)(?:_|$)",
                str(self.loaded_trained_model_id or ""),
            )
            if match:
                training_seed = int(match.group(1))
                training_seed_source = "legacy_model_id_inference"
            else:
                training_seed_source = "unavailable"
        self.loaded_trained_model_training_seed = (
            None if training_seed is None else int(training_seed)
        )
        self.loaded_trained_model_training_seed_source = training_seed_source
        self.loaded_trained_model_sha256 = _sha256_file(source)
        print(
            f">>> Loaded trained policy <- {source}; "
            f"model_id={self.loaded_trained_model_id}, "
            f"training_seed={self.loaded_trained_model_training_seed} "
            f"({self.loaded_trained_model_training_seed_source}); "
            f"benchmark opponent remains {self.firm2_mode}."
        )
        return artifact

    @staticmethod
    def _validation_opponent_rollouts(
        actual_mode: str,
        actual_seed_count: int,
        include_benchmarks: bool = False,
    ) -> List[Tuple[str, str]]:
        """Build checkpoint or reporting rollouts without mixing their roles."""
        mode = str(actual_mode)
        seed_count = int(max(1, actual_seed_count))
        rollouts = [
            (f"actual_firm2__{mode}__opening_{index + 1}", mode)
            for index in range(seed_count)
        ]
        if include_benchmarks:
            rollouts.extend(
                (benchmark, benchmark) for benchmark in BENCHMARK_OPPONENT_MODES
            )
        return rollouts

    @staticmethod
    def _full_horizon_validation_means(
        **series: Sequence[float],
    ) -> Dict[str, float]:
        """Return aligned, finite full-rollout means for checkpoint scoring."""
        if not series:
            raise ValueError("validation metric series cannot be empty")
        vectors = {
            name: np.asarray(values, dtype=float).reshape(-1)
            for name, values in series.items()
        }
        lengths = {vector.size for vector in vectors.values()}
        if len(lengths) != 1 or 0 in lengths:
            raise ValueError("validation metric series must be non-empty and aligned")
        for name, vector in vectors.items():
            if not np.all(np.isfinite(vector)):
                raise ValueError(f"validation metric {name} must be finite")
        return {
            name: float(np.mean(vector))
            for name, vector in vectors.items()
        }

    def _adaptive_policy_validation(
        self,
        *,
        customers_per_step: int,
        profile_sample_size: int,
        horizon: int = 192,
        aggressive_horizon: Optional[int] = None,
        policy_mode: str = "argmax_raw",
        policy_temperature: float = 0.50,
        policy_seed: Optional[int] = None,
        opponent_interval_steps: int = 10,
        firm1_interval_steps: Optional[int] = None,
        steps_per_day: int = 10,
        aggressive_seed_count: int = 5,
        include_benchmarks: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate the actor against the configured Firm 2 on held-out data.

        Validation must call ``agent.act`` repeatedly.  Scoring a single market
        batch at the current training tariff selects a tariff snapshot, not an
        adaptive policy, and leaks the terminal training state into evaluation.
        """
        saved = {
            "market": copy.deepcopy(self.market),
            "agent_gen": self.agent_gen,
            "firm1_overrides": copy.deepcopy(self.firm1.overrides),
            "firm2": copy.deepcopy(self.firm2),
            "firm1_execution": self.firm1.snapshot_execution_state(),
            "observers": {name: observer.snapshot() for name, observer in self.platform_observers.items()},
            "stability": self.action_stability.snapshot(),
            "crowd": dict(getattr(self, "last_crowd_response_stats", {}) or {}),
            "driver_supply": copy.deepcopy(self.driver_supply),
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
            "torch_rng_state": torch.random.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        }
        tracked_names = (
            "last_share", "last_revpr", "last_profitpr", "last_gap",
            "last_fulfillment", "last_completion_coverage",
            "last_acceptance", "last_wait", "last_action_mask_regime",
            "last_firm2_share", "last_firm2_revpr", "last_firm2_profitpr",
        )
        saved["last_values"] = {name: getattr(self, name) for name in tracked_names}
        mode_scores: List[float] = []
        rewards: List[float] = []
        own_shares: List[float] = []
        rival_shares: List[float] = []
        own_completed_shares: List[float] = []
        rival_completed_shares: List[float] = []
        own_revenues: List[float] = []
        rival_revenues: List[float] = []
        own_profits: List[float] = []
        rival_profits: List[float] = []
        gaps: List[float] = []
        fulfillment: List[float] = []
        completion_coverage: List[float] = []
        actions: List[int] = []
        validation_mode_names: List[str] = []
        mode_profit_advantages: List[float] = []
        mode_share_advantages: List[float] = []
        mode_objective_advantages: List[float] = []
        mode_gap_errors: List[float] = []
        mode_own_profit_means: List[float] = []
        mode_rival_profit_means: List[float] = []
        mode_own_share_means: List[float] = []
        mode_rival_share_means: List[float] = []
        mode_fulfillment_means: List[float] = []
        mode_completion_coverage_means: List[float] = []
        mode_tail_objective_rewards: List[float] = []
        mode_profit_advantage_decay: List[float] = []
        mode_objective_dominance_rates: List[float] = []
        mode_tail_dynamicity: Dict[str, float] = {}
        mode_tail_action_diversity: Dict[str, float] = {}
        mode_tail_action_switch_rate: Dict[str, float] = {}
        mode_leader_intervention_rates: Dict[str, float] = {}
        mode_selected_action_feasibility: Dict[str, float] = {}
        validation_mode_cities: Dict[str, str] = {}
        benchmark_trajectories: Dict[str, Dict[str, List[float]]] = {}
        heldout_critic_predictions: List[float] = []
        heldout_critic_targets: List[float] = []
        try:
            resolved_policy_seed = int(
                self.seed + 810_001 if policy_seed is None else policy_seed
            )
            resolved_actual_seed_count = int(max(1, aggressive_seed_count))
            validation_rollouts = self._validation_opponent_rollouts(
                str(self.firm2_mode),
                resolved_actual_seed_count,
                include_benchmarks=include_benchmarks,
            )
            resolved_firm1_interval = int(
                max(
                    1,
                    opponent_interval_steps
                    if firm1_interval_steps is None
                    else firm1_interval_steps,
                )
            )
            resolved_steps_per_day = int(max(1, steps_per_day))
            for rollout_index, (rollout_name, mode) in enumerate(validation_rollouts):
                rollout_seed = resolved_policy_seed + 10_007 * rollout_index
                validation_city = str(self.market_name)
                validation_mode_cities[str(rollout_name)] = validation_city
                torch.manual_seed(rollout_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(rollout_seed)
                self.rng = np.random.default_rng(rollout_seed + 1)
                # Each validation rollout receives an independently seeded
                # configured-market path and the same-distribution held-out
                # profile split.
                # Opponent strategy is the experimental variable; city and
                # threshold distribution are deliberately held fixed.
                self.market = MarketInteraction(
                    city_name=validation_city, seed=rollout_seed + 2
                )
                self.market.set_market(validation_city)
                self.market.refresh_run_probabilities(jitter_scale=0.15)
                self.agent_gen = GenerateAgent(
                    seed=rollout_seed + 4,
                    total_customers=max(512, profile_sample_size),
                    city_name=validation_city,
                )
                self.agent_gen.apply_probability_variation(jitter_scale=0.18)
                self.driver_supply = copy.deepcopy(saved["driver_supply"])
                self._reset_adaptive_episode(
                    tariff_reset_fraction=0.15,
                    opponent_mode=mode,
                    episode_seed=rollout_seed + 3,
                    symmetric_tariffs=False,
                )
                clock = OperationalClock()
                prev_share = 0.5
                prev_rev = 0.0
                prev_profit = 0.0
                prev_rival_share = 0.5
                prev_rival_profit = 0.0
                prev_fulfillment = 1.0
                prev_gap = 0.0
                mode_rewards: List[float] = []
                mode_own_shares: List[float] = []
                mode_rival_shares: List[float] = []
                mode_own_profits: List[float] = []
                mode_rival_profits: List[float] = []
                mode_gaps: List[float] = []
                mode_gap_abs_errors: List[float] = []
                mode_fulfillment: List[float] = []
                mode_completion_coverage: List[float] = []
                mode_actions: List[int] = []
                mode_action_feasible: List[float] = []
                mode_critic_records: List[
                    Tuple[float, float, float, float, float, float]
                ] = []
                pending_critic_value: Optional[float] = None
                pending_critic_reward = 0.0
                pending_critic_discount = 1.0
                pending_critic_potential = 0.0
                previous_firm2_metrics: Optional[FirmMetrics] = None
                leader_decision_count = 0
                leader_intervention_count = 0
                rollout_day_ctx = None
                # ``aggressive_horizon`` is retained as an API-compatibility
                # name. The canonical suite applies the requested long horizon
                # uniformly to the actual rival and all five deployment benchmarks.
                mode_horizon = int(max(
                    horizon,
                    horizon if aggressive_horizon is None else aggressive_horizon,
                ))
                for step_index in range(max(2, mode_horizon)):
                    if (
                        rollout_day_ctx is None
                        or step_index % resolved_steps_per_day == 0
                    ):
                        rollout_day_ctx = self.market.sample_day_context()
                    ctx = rollout_day_ctx
                    hour = self.market.sample_timestep_hour().hour
                    rival_due = clock.due(opponent_interval_steps)
                    _, leader_intervened = self._firm2_leader_step(
                        previous_metrics=previous_firm2_metrics,
                        day_of_week=ctx.day_of_week,
                        hour=hour,
                        weather=ctx.weather,
                        enabled=rival_due,
                    )
                    if rival_due and not isinstance(
                        self.firm2, FirmStaticPricer
                    ):
                        leader_decision_count += 1
                        leader_intervention_count += int(leader_intervened)
                    raw_state, action_features, _ = self._publish_pricing_observation(
                        day_of_week=ctx.day_of_week,
                        hour=hour,
                        weather=ctx.weather,
                    )
                    state = self.firm1.stack_state(raw_state)
                    critic_state = (
                        self._build_privileged_critic_state(raw_state)
                        if self.privileged_critic
                        else None
                    )
                    action = -1
                    firm1_due = (step_index % resolved_firm1_interval) == 0
                    if firm1_due:
                        action_mask = self._objective_action_mask(
                            self.firm1.feasible_action_mask(self.market),
                            fulfillment_rate=self.last_fulfillment,
                        )
                        (
                            action,
                            _,
                            _,
                            decision_value,
                            _,
                            _,
                        ) = self.firm1.agent.act(
                            state,
                            deterministic=True,
                            action_features=action_features,
                            action_mask=action_mask,
                            policy_mode=policy_mode,
                            policy_temperature=policy_temperature,
                            critic_state=critic_state,
                        )
                        mode_action_feasible.append(float(
                            0 <= int(action) < len(action_mask)
                            and bool(action_mask[int(action)])
                        ))
                        current_potential = (
                            self._public_reward_potential(raw_state)
                            if self.potential_shaping_scale > 0.0
                            else 0.0
                        )
                        if pending_critic_value is not None:
                            mode_critic_records.append((
                                float(pending_critic_value),
                                float(pending_critic_reward),
                                float(pending_critic_discount),
                                float(pending_critic_potential),
                                float(current_potential),
                                float(decision_value.item()),
                            ))
                        pending_critic_value = float(decision_value.item())
                        pending_critic_reward = 0.0
                        pending_critic_discount = 1.0
                        pending_critic_potential = float(current_potential)
                        self.firm1.apply_action(action, self.market)
                        actions.append(int(action))
                        mode_actions.append(int(action))
                    else:
                        self.firm1.agent.observe_recurrent_state(
                            state, critic_state=critic_state
                        )
                    profiles = self._sample_profiles_from_pool(
                        profile_sample_size,
                        generalization_fraction=1.0,
                        generalization_city=validation_city,
                    )
                    _, m1, m2, gap, _, _ = self.simulate_batch(
                        day_of_week=ctx.day_of_week,
                        weather=ctx.weather,
                        hour=hour,
                        customers_per_step=customers_per_step,
                        sampled_profiles=profiles,
                        collect_rows=False,
                    )
                    self._ingest_platform_observations(m1, m2)
                    previous_firm2_metrics = m2
                    diag = self._reward_diagnostics(
                        share=float(m1.chosen_share),
                        completed_share=float(m1.completed_share),
                        rev_per_request=float(m1.rev_per_request),
                        mean_gap=float(gap),
                        prev_share=prev_share,
                        prev_rev_per_request=prev_rev,
                        prev_profit_per_request=prev_profit,
                        prev_rival_profit_per_request=prev_rival_profit,
                        prev_rival_market_share=prev_rival_share,
                        prev_fulfillment_rate=prev_fulfillment,
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
                        action_event=firm1_due,
                        price_gap_abs_error=float(
                            self.last_price_gap_metrics["mae"]
                        ),
                    )
                    mode_own_profits.append(float(m1.profit_per_request))
                    mode_rival_profits.append(float(m2.profit_per_request))
                    mode_own_shares.append(
                        float(diag["reward_market_share"])
                    )
                    mode_rival_shares.append(
                        float(diag["reward_rival_market_share"])
                    )
                    mode_gaps.append(float(gap))
                    mode_gap_abs_errors.append(
                        float(diag["reward_price_gap_abs_error"])
                    )
                    mode_rewards.append(float(diag["reward"]))
                    if pending_critic_value is not None:
                        pending_critic_reward += (
                            pending_critic_discount * float(diag["reward"])
                        )
                        pending_critic_discount *= float(
                            self.firm1.agent.gamma
                        )
                    rewards.append(float(diag["reward"]))
                    own_shares.append(float(diag["reward_market_share"]))
                    rival_shares.append(
                        float(diag["reward_rival_market_share"])
                    )
                    own_completed_shares.append(float(m1.completed_share))
                    rival_completed_shares.append(float(m2.completed_share))
                    own_revenues.append(float(m1.rev_per_request))
                    rival_revenues.append(float(m2.rev_per_request))
                    own_profits.append(float(m1.profit_per_request))
                    rival_profits.append(float(m2.profit_per_request))
                    gaps.append(float(gap))
                    fulfillment.append(float(m1.fulfillment_rate))
                    mode_fulfillment.append(float(m1.fulfillment_rate))
                    realized_coverage = float(np.clip(
                        m1.completed_share + m2.completed_share, 0.0, 1.0
                    ))
                    completion_coverage.append(realized_coverage)
                    mode_completion_coverage.append(realized_coverage)
                    self.last_fulfillment = float(m1.fulfillment_rate)
                    self.last_completion_coverage = realized_coverage
                    prev_share = float(diag["reward_market_share"])
                    prev_rev = float(m1.rev_per_request)
                    prev_profit = float(m1.profit_per_request)
                    prev_rival_share = float(
                        diag["reward_rival_market_share"]
                    )
                    prev_rival_profit = float(m2.profit_per_request)
                    prev_fulfillment = float(m1.fulfillment_rate)
                    prev_gap = float(gap)
                    clock.advance()
                if mode_critic_records:
                    exposures = np.asarray([
                        (
                            float(max(1, resolved_firm1_interval))
                            if abs(1.0 - self.firm1.agent.gamma) <= 1e-12
                            else (1.0 - record[2])
                            / (1.0 - self.firm1.agent.gamma)
                        )
                        for record in mode_critic_records
                    ], dtype=float)
                    raw_interval_rewards = np.asarray(
                        [record[1] for record in mode_critic_records],
                        dtype=float,
                    )
                    # Match the critic's training target with the lagged
                    # action-independent baseline. Using this validation
                    # rollout's own realized mean would leak its actions into
                    # the target and overstate explained variance.
                    validation_reward_rate = float(
                        self.firm1.agent.reward_rate_by_regime.get(
                            str(mode), 0.0
                        )
                        if self.credit_assignment_v25
                        else np.sum(raw_interval_rewards)
                        / max(1e-8, float(np.sum(exposures)))
                    )
                    for record, exposure in zip(
                        mode_critic_records, exposures
                    ):
                        (
                            prediction,
                            raw_interval_reward,
                            interval_discount,
                            potential,
                            next_potential,
                            next_prediction,
                        ) = record
                        shaped_reward = float(raw_interval_reward)
                        if self.potential_shaping_scale > 0.0:
                            shaped_reward += self.potential_shaping_scale * (
                                interval_discount * next_potential - potential
                            )
                        target = (
                            shaped_reward
                            - validation_reward_rate * float(exposure)
                            + interval_discount * next_prediction
                        )
                        heldout_critic_predictions.append(float(prediction))
                        heldout_critic_targets.append(float(target))
                # The tail is the same per-step economic reward, aggregated
                # separately so transient opening performance cannot outrank
                # durable performance. It introduces no new reward component.
                tail_start = min(
                    len(mode_own_profits) - 1,
                    max(0, (3 * len(mode_own_profits)) // 4),
                )
                tail_own_profit = float(np.mean(mode_own_profits[tail_start:]))
                tail_rival_profit = float(np.mean(mode_rival_profits[tail_start:]))
                tail_own_share = float(np.mean(mode_own_shares[tail_start:]))
                tail_rival_share = float(np.mean(mode_rival_shares[tail_start:]))
                late_share_advantage = tail_own_share - tail_rival_share
                objective_advantage_path = self._objective_dominance_path(
                    mode_own_profits[tail_start:],
                    mode_rival_profits[tail_start:],
                    mode_own_shares[tail_start:],
                    mode_rival_shares[tail_start:],
                )
                objective_advantage_target = 0.0
                dominance_window = int(
                    max(1, min(20, objective_advantage_path.size))
                )
                objective_rolling_advantage = np.convolve(
                    objective_advantage_path,
                    np.full(
                        dominance_window,
                        1.0 / dominance_window,
                        dtype=float,
                    ),
                    mode="valid",
                )
                objective_dominance_rate = float(np.mean(
                    objective_rolling_advantage >= objective_advantage_target
                ))
                early_end = max(1, len(mode_own_profits) // 4)
                early_advantage = float(
                    np.mean(mode_own_profits[:early_end])
                    - np.mean(mode_rival_profits[:early_end])
                )
                late_advantage = tail_own_profit - tail_rival_profit
                tail_gap_error = float(
                    np.mean(mode_gap_abs_errors[tail_start:])
                )
                decision_tail_start = max(0, (3 * len(mode_actions)) // 4)
                tail_actions = mode_actions[decision_tail_start:]
                effective_tail_actions = [
                    self._economic_action_group(int(action))
                    for action in tail_actions
                ]
                tail_action_diversity = float(
                    len(set(effective_tail_actions)) / 3.0
                )
                tail_action_switch_rate = float(
                    np.mean(np.diff(effective_tail_actions) != 0)
                    if len(effective_tail_actions) > 1 else 0.0
                )
                tail_dynamicity = float(
                    min(1.0, tail_action_diversity / (2.0 / 3.0))
                    * min(1.0, tail_action_switch_rate / 0.10)
                )
                mode_tail_dynamicity[str(rollout_name)] = tail_dynamicity
                mode_tail_action_diversity[str(rollout_name)] = tail_action_diversity
                mode_tail_action_switch_rate[str(rollout_name)] = tail_action_switch_rate
                mode_leader_intervention_rates[str(rollout_name)] = float(
                    leader_intervention_count / max(1, leader_decision_count)
                )
                mode_selected_action_feasibility[str(rollout_name)] = float(
                    np.mean(mode_action_feasible)
                    if mode_action_feasible else 1.0
                )
                full_means = self._full_horizon_validation_means(
                    reward=mode_rewards,
                    own_profit=mode_own_profits,
                    rival_profit=mode_rival_profits,
                    own_market_share=mode_own_shares,
                    rival_market_share=mode_rival_shares,
                    fulfillment=mode_fulfillment,
                    completion_coverage=mode_completion_coverage,
                )
                validation_mode_names.append(str(rollout_name))
                mode_own_profit_means.append(full_means["own_profit"])
                mode_rival_profit_means.append(full_means["rival_profit"])
                mode_own_share_means.append(full_means["own_market_share"])
                mode_rival_share_means.append(full_means["rival_market_share"])
                mode_fulfillment_means.append(full_means["fulfillment"])
                mode_completion_coverage_means.append(
                    full_means["completion_coverage"]
                )
                mode_tail_objective_rewards.append(float(
                    np.mean(mode_rewards[tail_start:])
                ))
                mode_profit_advantages.append(late_advantage)
                mode_share_advantages.append(late_share_advantage)
                mode_objective_advantages.append(float(
                    np.mean(objective_advantage_path)
                ))
                mode_objective_dominance_rates.append(
                    objective_dominance_rate
                )
                mode_gap_errors.append(tail_gap_error)
                mode_profit_advantage_decay.append(
                    max(0.0, early_advantage - late_advantage)
                )
                mode_scores.append(self._validation_score_from_metrics(
                    reward=full_means["reward"],
                    completed_share_advantage=late_share_advantage,
                ))
                if (
                    include_benchmarks
                    and str(rollout_name) in BENCHMARK_OPPONENT_MODES
                ):
                    benchmark_trajectories[str(rollout_name)] = {
                        "day": [
                            float(index + 1)
                            for index in range(len(mode_rewards))
                        ],
                        "normalized_objective_reward": [
                            float(value) for value in mode_rewards
                        ],
                        "firm1_profit_per_request": [
                            float(value) for value in mode_own_profits
                        ],
                        "firm2_profit_per_request": [
                            float(value) for value in mode_rival_profits
                        ],
                        "firm1_completed_share": [
                            float(value) for value in mode_own_shares
                        ],
                        "firm2_completed_share": [
                            float(value) for value in mode_rival_shares
                        ],
                        "firm1_fulfillment_rate": [
                            float(value) for value in mode_fulfillment
                        ],
                        "completion_coverage": [
                            float(value) for value in mode_completion_coverage
                        ],
                        "price_gap_abs_error": [
                            float(value) for value in mode_gap_abs_errors
                        ],
                    }
        finally:
            self.market = saved["market"]
            self.agent_gen = saved["agent_gen"]
            self.firm1.overrides = saved["firm1_overrides"]
            self.firm2 = saved["firm2"]
            self.firm1.restore_execution_state(saved["firm1_execution"])
            for name, snapshot in saved["observers"].items():
                self.platform_observers[name].restore(snapshot)
            self.action_stability.restore(saved["stability"])
            self.last_crowd_response_stats = saved["crowd"]
            self.driver_supply = saved["driver_supply"]
            self.rng.bit_generator.state = saved["rng_state"]
            torch.random.set_rng_state(saved["torch_rng_state"])
            if torch.cuda.is_available() and saved["cuda_rng_state"] is not None:
                torch.cuda.set_rng_state_all(saved["cuda_rng_state"])
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
        mean_share_advantage = (
            float(np.mean(mode_share_advantages))
            if mode_share_advantages else -float("inf")
        )
        worst_share_advantage = (
            float(np.min(mode_share_advantages))
            if mode_share_advantages else -float("inf")
        )
        mean_own_profit = (
            float(np.mean(mode_own_profit_means))
            if mode_own_profit_means else -float("inf")
        )
        worst_own_profit = (
            float(np.min(mode_own_profit_means))
            if mode_own_profit_means else -float("inf")
        )
        mean_fulfillment = (
            float(np.mean(mode_fulfillment_means))
            if mode_fulfillment_means else 0.0
        )
        worst_fulfillment = (
            float(np.min(mode_fulfillment_means))
            if mode_fulfillment_means else 0.0
        )
        mean_completion_coverage = (
            float(np.mean(mode_completion_coverage_means))
            if mode_completion_coverage_means else 0.0
        )
        worst_completion_coverage = (
            float(np.min(mode_completion_coverage_means))
            if mode_completion_coverage_means else 0.0
        )
        mean_gap_error = (
            float(np.mean(mode_gap_errors))
            if mode_gap_errors else float("inf")
        )
        worst_gap_error = (
            float(np.max(mode_gap_errors))
            if mode_gap_errors else float("inf")
        )
        profit_win_rate = (
            float(np.mean(np.asarray(mode_profit_advantages) > 0.0))
            if mode_profit_advantages else 0.0
        )
        share_win_rate = (
            float(np.mean(np.asarray(mode_share_advantages) > 0.0))
            if mode_share_advantages else 0.0
        )
        robust_dominance_bonus = float(
            0.50 * np.tanh(
                worst_profit_advantage
                / max(
                    1e-6,
                    self.long_term_profit_reward_config.profit_advantage_scale,
                )
            )
        )
        incomplete_win_penalty = float(0.50 * (1.0 - profit_win_rate))
        worst_advantage_decay = (
            float(np.max(mode_profit_advantage_decay))
            if mode_profit_advantage_decay else float("inf")
        )
        durability_decay_penalty = float(
            0.35 * np.tanh(
                worst_advantage_decay
                / max(
                    1e-6,
                    self.long_term_profit_reward_config.profit_advantage_scale,
                )
            )
        )
        action_diversity = float(
            len(set(actions)) / max(1, len(self.firm1.action_to_steps))
        )
        action_switch_rate = float(
            np.mean(np.diff(actions) != 0) if len(actions) > 1 else 0.0
        )
        aggressive_dynamicities = [
            float(value)
            for name, value in mode_tail_dynamicity.items()
            if str(name).startswith("adaptive_best_response_aggressive")
        ]
        policy_dynamicity = float(
            min(aggressive_dynamicities)
            if aggressive_dynamicities
            else 0.0
        )
        # Responsiveness is reported, but it is not a source of reward and not
        # a certification requirement.  Requiring late tariff movement made a
        # genuinely converged dominant policy ineligible and selected action
        # churn instead.  Economic durability is established directly by the
        # worst-opponent outcome gaps below.
        minimum_checkpoint_dynamicity = 0.0
        objective = self._configured_primary_objective()
        profit_dominance_gap = 0.0
        optimization = self.objective_optimizer.summarize(
            own_profit=mode_own_profit_means,
            rival_profit=mode_rival_profit_means,
            own_market_share=mode_own_share_means,
            rival_market_share=mode_rival_share_means,
            fulfillment=mode_fulfillment_means,
            completion_coverage=mode_completion_coverage_means,
            objective_dominance_rate=mode_objective_dominance_rates,
            tail_objective_reward=mode_tail_objective_rewards,
        )
        durability_horizon_sufficient = bool(
            int(max(
                horizon,
                horizon if aggressive_horizon is None else aggressive_horizon,
            ))
            >= DURABILITY_MIN_STEPS
        )
        checkpoint_eligible = bool(
            optimization["checkpoint_eligible"]
            and durability_horizon_sufficient
        )
        objective_selection_key = list(optimization["selection_key"])
        collapsed_policy_penalty = float(0.50 * (1.0 - policy_dynamicity))
        checkpoint_score = float(optimization["objective_score"])
        critic_predictions = np.asarray(
            heldout_critic_predictions, dtype=float
        )
        critic_targets = np.asarray(heldout_critic_targets, dtype=float)
        heldout_critic_td_mae = float(
            np.mean(np.abs(critic_targets - critic_predictions))
            if critic_targets.size
            else float("nan")
        )
        heldout_target_variance = float(
            np.var(critic_targets) if critic_targets.size else 0.0
        )
        heldout_critic_td_explained_variance = float(
            1.0
            - np.var(critic_targets - critic_predictions)
            / heldout_target_variance
            if critic_targets.size >= 2 and heldout_target_variance > 1e-8
            else float("nan")
        )
        mode_own_objective_scores, mode_rival_objective_scores = (
            self._objective_firm_scores(
                mode_own_profit_means,
                mode_rival_profit_means,
                mode_own_share_means,
                mode_rival_share_means,
            )
        )
        return {
            "score": checkpoint_score,
            "objective_optimizer": str(optimization["optimizer_mechanism"]),
            "objective_selection_key": objective_selection_key,
            "objective_full_horizon_mean": float(
                optimization["objective_mean"]
            ),
            "objective_tail_mean": float(
                optimization["tail_objective_mean"]
            ),
            "objective_tail_weight": float(optimization["tail_weight"]),
            "objective_reward_weights": dict(
                optimization["reward_weights"]
            ),
            "objective_feasible_mode_fraction": float(
                optimization["feasible_mode_fraction"]
            ),
            "objective_maximum_normalized_shortfall": float(
                optimization["maximum_normalized_shortfall"]
            ),
            "objective_mean_normalized_shortfall": float(
                optimization["mean_normalized_shortfall"]
            ),
            "mean_objective_dominance_rate": float(
                optimization["mean_objective_dominance_rate"]
            ),
            "worst_objective_dominance_rate": float(
                optimization["worst_objective_dominance_rate"]
            ),
            "objective_dominance_rate_floor": float(
                optimization["dominance_rate_floor"]
            ),
            "mode_objective_dominance_rate": {
                name: float(value)
                for name, value in zip(
                    validation_mode_names,
                    optimization["mode_objective_dominance_rate"],
                )
            },
            "objective_mode_normalized_shortfall": {
                name: float(value)
                for name, value in zip(
                    validation_mode_names,
                    optimization["mode_normalized_shortfall"],
                )
            },
            "objective_mode_feasible": {
                name: bool(value)
                for name, value in zip(
                    validation_mode_names,
                    optimization["mode_feasible"],
                )
            },
            "primary_objective": objective,
            "market_share_target_gap": float(
                self.long_term_profit_reward_config.market_share_target_gap
            ),
            "profit_advantage_target": float(profit_dominance_gap),
            "fulfillment_floor": float(
                self.constraint_config.fulfillment_floor
            ),
            "completion_coverage_floor": float(
                self.constraint_config.completion_coverage_floor
            ),
            "mode_score_mean": mean_mode_score,
            "mode_score_std": mode_score_std,
            "mode_score_worst": worst_mode_score,
            "mean_profit_advantage": mean_profit_advantage,
            "worst_profit_advantage": worst_profit_advantage,
            "mean_completed_share_advantage": mean_share_advantage,
            "worst_completed_share_advantage": worst_share_advantage,
            "completed_share_win_rate": share_win_rate,
            "mean_market_share_advantage": mean_share_advantage,
            "worst_market_share_advantage": worst_share_advantage,
            "market_share_win_rate": share_win_rate,
            "mean_own_profit_per_request": mean_own_profit,
            "worst_own_profit_per_request": worst_own_profit,
            "mean_fulfillment_rate": mean_fulfillment,
            "worst_fulfillment_rate": worst_fulfillment,
            "mean_completion_coverage": mean_completion_coverage,
            "worst_completion_coverage": worst_completion_coverage,
            "profit_win_rate": profit_win_rate,
            "mean_price_gap_abs_error": mean_gap_error,
            "worst_price_gap_abs_error": worst_gap_error,
            "robust_dominance_bonus": robust_dominance_bonus,
            "incomplete_win_penalty": incomplete_win_penalty,
            "worst_profit_advantage_decay": worst_advantage_decay,
            "durability_decay_penalty": durability_decay_penalty,
            "policy_dynamicity": policy_dynamicity,
            "minimum_checkpoint_dynamicity": minimum_checkpoint_dynamicity,
            "checkpoint_eligible": checkpoint_eligible,
            "heldout_critic_td_explained_variance": (
                heldout_critic_td_explained_variance
            ),
            "heldout_critic_td_mae": heldout_critic_td_mae,
            "heldout_critic_td_samples": int(critic_targets.size),
            "durability_horizon_sufficient": durability_horizon_sufficient,
            "durability_minimum_horizon": DURABILITY_MIN_STEPS,
            "collapsed_policy_penalty": collapsed_policy_penalty,
            "mode_profit_advantages": {
                name: float(advantage)
                for name, advantage in zip(
                    validation_mode_names, mode_profit_advantages
                )
            },
            "mode_completed_share_advantages": {
                name: float(advantage)
                for name, advantage in zip(
                    validation_mode_names, mode_share_advantages
                )
            },
            "mode_objective_advantages": {
                name: float(advantage)
                for name, advantage in zip(
                    validation_mode_names, mode_objective_advantages
                )
            },
            "mode_own_market_share": {
                name: float(share)
                for name, share in zip(
                    validation_mode_names, mode_own_share_means
                )
            },
            "mode_rival_market_share": {
                name: float(share)
                for name, share in zip(
                    validation_mode_names, mode_rival_share_means
                )
            },
            "mode_profit_advantage_decay": {
                name: float(decay)
                for name, decay in zip(
                    validation_mode_names, mode_profit_advantage_decay
                )
            },
            "mode_price_gap_abs_error": {
                name: float(error)
                for name, error in zip(validation_mode_names, mode_gap_errors)
            },
            "mode_tail_dynamicity": dict(mode_tail_dynamicity),
            "mode_tail_action_diversity": dict(mode_tail_action_diversity),
            "mode_tail_action_switch_rate": dict(mode_tail_action_switch_rate),
            "mode_firm2_leader_intervention_rates": dict(
                mode_leader_intervention_rates
            ),
            "mode_selected_action_feasibility": dict(
                mode_selected_action_feasibility
            ),
            "competition_protocol": "firm2_leader_then_firm1_follower",
            "validation_policy_mode": str(policy_mode),
            "validation_policy_seed": int(
                self.seed + 810_001 if policy_seed is None else policy_seed
            ),
            "validation_base_horizon": int(horizon),
            "validation_aggressive_horizon": int(
                horizon if aggressive_horizon is None else aggressive_horizon
            ),
            "validation_benchmark_horizon": int(max(
                horizon,
                horizon if aggressive_horizon is None else aggressive_horizon,
            )),
            "validation_aggressive_seed_count": int(
                resolved_actual_seed_count
            ),
            "validation_opponent_interval_steps": int(
                max(1, opponent_interval_steps)
            ),
            "validation_firm1_interval_steps": int(
                resolved_firm1_interval
            ),
            "validation_steps_per_day": int(resolved_steps_per_day),
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
            "mode_own_objective_score": {
                name: float(score)
                for name, score in zip(
                    validation_mode_names, mode_own_objective_scores
                )
            },
            "mode_rival_objective_score": {
                name: float(score)
                for name, score in zip(
                    validation_mode_names, mode_rival_objective_scores
                )
            },
            "mode_fulfillment_rate": {
                name: float(rate)
                for name, rate in zip(
                    validation_mode_names, mode_fulfillment_means
                )
            },
            "mode_completion_coverage": {
                name: float(rate)
                for name, rate in zip(
                    validation_mode_names, mode_completion_coverage_means
                )
            },
            "mode_validation_cities": dict(validation_mode_cities),
            "benchmark_trajectories": benchmark_trajectories,
            "validation_profile_source": "same_market_threshold_holdout",
            "reward": float(np.mean(rewards)) if rewards else 0.0,
            "share": float(np.mean(own_shares)) if own_shares else 0.0,
            "rival_share": float(np.mean(rival_shares)) if rival_shares else 0.0,
            "completed_share": float(np.mean(own_completed_shares)) if own_completed_shares else 0.0,
            "rival_completed_share": float(np.mean(rival_completed_shares)) if rival_completed_shares else 0.0,
            "rev_per_request": float(np.mean(own_revenues)) if own_revenues else 0.0,
            "rival_rev_per_request": float(np.mean(rival_revenues)) if rival_revenues else 0.0,
            "profit_per_request": float(np.mean(own_profits)) if own_profits else 0.0,
            "rival_profit_per_request": float(np.mean(rival_profits)) if rival_profits else 0.0,
            "fulfillment_rate": float(np.mean(fulfillment)) if fulfillment else 0.0,
            "completion_coverage": float(np.mean(completion_coverage)) if completion_coverage else 0.0,
            "price_gap": float(np.mean(gaps)) if gaps else 0.0,
            "action_diversity": action_diversity,
            "action_switch_rate": action_switch_rate,
        }

    @staticmethod
    def _normalized_smoothed_trend(
        values: Sequence[float],
        window: int = 20,
    ) -> float:
        """Return fitted start-to-finish change after moving-average smoothing."""
        array = np.asarray(values, dtype=float)
        array = array[np.isfinite(array)]
        if array.size < 3:
            return 0.0
        resolved_window = int(max(1, min(window, array.size // 3 or 1)))
        if resolved_window > 1:
            kernel = np.ones(resolved_window, dtype=float) / resolved_window
            array = np.convolve(array, kernel, mode="valid")
        if array.size < 2:
            return 0.0
        x = np.linspace(0.0, 1.0, array.size)
        return float(np.polyfit(x, array, 1)[0])

    def _fare_equivalent_action_impact(self, action: int) -> float:
        """Approximate broad-market dollars changed by a discrete action.

        The exposure weights represent a typical 15-minute, four-mile ride;
        airport fee receives only its approximate population exposure. This
        prevents an airport-only oscillation from satisfying the dynamic-policy
        gate while the coefficients affecting nearly every ride remain fixed.
        """
        if self.firm1_mode != "RL" or not hasattr(self.firm1, "action_steps"):
            return 0.0
        multiplier = getattr(
            self.firm1, "action_to_multiplier", {}
        ).get(int(action))
        target_key = getattr(
            self.firm1, "action_to_target_key", {}
        ).get(int(action))
        if multiplier is not None and target_key is None:
            base = self.market.curr_market
            exposure = {
                "base_fare": 1.0,
                "per_minute": 14.65,
                "per_mile": 3.95,
                "booking_fee": 1.0,
                "airport_fee": 0.0775,
            }
            return abs(float(multiplier) - 1.0) * float(sum(
                float(getattr(base, key)) * float(exposure.get(key, 0.0))
                for key in self.firm1.opt_keys
            ))
        if multiplier is not None and target_key is not None:
            base = self.market.curr_market
            exposure = {
                "base_fare": 1.0,
                "per_minute": 14.65,
                "per_mile": 3.95,
                "booking_fee": 1.0,
                "airport_fee": 0.0775,
            }
            return (
                abs(float(multiplier) - 1.0)
                * float(getattr(base, target_key))
                * float(exposure.get(target_key, 0.0))
            )
        steps = self.firm1.action_steps(int(action))
        exposure = {
            "base_fare": 1.0,
            "per_minute": 14.65,
            "per_mile": 3.95,
            "booking_fee": 1.0,
            "airport_fee": 0.0775,
        }
        config_steps = getattr(getattr(self.firm1, "config", None), "step", {})
        return float(sum(
            abs(float(direction))
            * float(config_steps.get(key, 0.0))
            * float(exposure.get(key, 0.0))
            for key, direction in steps.items()
        ))

    def _economic_action_group(self, action: int) -> int:
        """Map an action to lower/neutral/higher representative total fare."""
        if self.firm1_mode != "RL" or not hasattr(self.firm1, "action_steps"):
            return 0
        multiplier = getattr(
            self.firm1, "action_to_multiplier", {}
        ).get(int(action))
        target_key = getattr(
            self.firm1, "action_to_target_key", {}
        ).get(int(action))
        if multiplier is not None:
            return int(np.sign(float(multiplier) - 1.0))
        steps = self.firm1.action_steps(int(action))
        exposure = {
            "base_fare": 1.0,
            "per_minute": 14.65,
            "per_mile": 3.95,
            "booking_fee": 1.0,
            "airport_fee": 0.0775,
        }
        config_steps = getattr(getattr(self.firm1, "config", None), "step", {})
        signed_impact = float(sum(
            float(direction)
            * float(config_steps.get(key, 0.0))
            * float(exposure.get(key, 0.0))
            for key, direction in steps.items()
        ))
        if signed_impact < -0.05:
            return -1
        if signed_impact > 0.05:
            return 1
        return 0

    def _objective_action_mask(
        self,
        base_mask: np.ndarray,
        *,
        fulfillment_rate: Optional[float] = None,
        completion_coverage: Optional[float] = None,
        share_advantage: Optional[float] = None,
    ) -> np.ndarray:
        """Preserve the stationary physical action set."""
        del fulfillment_rate, completion_coverage, share_advantage
        mask = np.asarray(base_mask, dtype=bool).copy()
        if mask.size == 0:
            return mask
        if not np.any(mask):
            mask[0] = True
        self.last_action_mask_regime = "all_tariff_targets_available"
        # No objective, business, or operational heuristic removes an action.
        return mask

    def _competitive_durability_backtest(self) -> Dict[str, Any]:
        """Audit learned dynamics without changing actions or simulator state.

        The audit follows the named policy objective. Profit maximization
        requires a durable contribution-profit result; competitiveness requires
        a durable completed-request-share lead;
        balanced requires both leads. Action changes remain diagnostics and
        never count as evidence of economic quality.

        Raw on-policy training reward is intentionally diagnostic only: PPO
        exploration and a learning adaptive rival make its time slope a poor
        convergence test. Evaluation reward is also not required to rise
        monotonically because evaluation freezes the policy. A sound frozen
        policy must retain reward while the rival and market evolve.
        """
        training = list(self.training_logs)
        evaluation = list(self.evaluation_logs)
        if not evaluation:
            return {
                "passed": False,
                "failures": ["evaluation_logs_missing"],
            }

        tail_size = max(20, len(evaluation) // 4)
        tail = evaluation[-tail_size:]
        settled_start = min(len(evaluation) - 1, len(evaluation) // 10)
        settled = evaluation[settled_start:]
        training_rewards = [float(row.get("avg_reward", np.nan)) for row in training]
        evaluation_rewards = [float(row.get("reward", np.nan)) for row in settled]
        validation_scores = [
            float(row.get("validation_score", np.nan))
            for row in training
            if np.isfinite(float(row.get("validation_score", np.nan)))
        ]
        if not training:
            loaded_metadata = dict(
                getattr(self, "loaded_trained_model_metadata", {}) or {}
            )
            saved_validation = dict(
                loaded_metadata.get("checkpoint_validation", {}) or {}
            )
            current_validation = dict(
                getattr(self, "benchmark_evaluation_report", {}) or {}
            )
            validation_scores = [
                float(report["score"])
                for report in (saved_validation, current_validation)
                if np.isfinite(float(report.get("score", np.nan)))
            ]

        def competitive_shares_from_row(
            row: Mapping[str, Any],
        ) -> Tuple[float, float]:
            if (
                "rl_market_share" in row
                and "heuristic_market_share" in row
            ):
                return (
                    float(row["rl_market_share"]),
                    float(row["heuristic_market_share"]),
                )
            return (
                float(row.get("rl_completed_share", 0.0)),
                float(row.get("heuristic_completed_share", 0.0)),
            )

        late_share_advantages = [
            own_market - rival_market
            for own_market, rival_market in (
                competitive_shares_from_row(row) for row in tail
            )
        ]
        late_profit_advantages = [
            float(row["rl_profit"]) - float(row["heuristic_profit"])
            for row in tail
        ]
        late_share_advantage = float(np.mean(late_share_advantages))
        late_profit_advantage = float(np.mean(late_profit_advantages))
        late_objective_advantages = self._objective_dominance_path(
            [float(row["rl_profit"]) for row in tail],
            [float(row["heuristic_profit"]) for row in tail],
            [competitive_shares_from_row(row)[0] for row in tail],
            [competitive_shares_from_row(row)[1] for row in tail],
        )
        late_objective_advantage = float(np.mean(late_objective_advantages))

        def rolling_dominance_rate(
            values: Sequence[float],
            *,
            window: int = 20,
            minimum_gap: float = 0.0,
        ) -> float:
            finite = np.asarray(
                [float(value) for value in values if np.isfinite(value)],
                dtype=float,
            )
            if finite.size == 0:
                return 0.0
            resolved = int(max(1, min(window, finite.size)))
            rolling = np.convolve(
                finite,
                np.full(resolved, 1.0 / resolved, dtype=float),
                mode="valid",
            )
            return float(np.mean(rolling >= float(minimum_gap)))
        target_gap = float(
            getattr(
                getattr(self, "long_term_profit_reward_config", None),
                "price_gap_target",
                0.75,
            )
        )
        gap_error_values = [
            float(
                row.get(
                    "price_gap_abs_error",
                    abs(
                        float(row.get("price_gap_f2_minus_f1", np.nan))
                        - target_gap
                    ),
                )
            )
            for row in tail
            if np.isfinite(
                float(
                    row.get(
                        "price_gap_abs_error",
                        abs(
                            float(row.get("price_gap_f2_minus_f1", np.nan))
                            - target_gap
                        ),
                    )
                )
            )
        ]
        late_gap_abs_error = (
            float(np.mean(gap_error_values))
            if gap_error_values
            else float("inf")
        )
        late_own_profit = float(np.mean([
            float(row["rl_profit"]) for row in tail
        ]))
        late_fulfillment_rate = float(np.mean([
            float(row.get("rl_fulfillment_rate", 1.0)) for row in tail
        ]))
        late_completion_coverage = float(np.mean([
            float(row.get(
                "completion_coverage",
                np.clip(
                    float(row.get("rl_completed_share", 0.0))
                    + float(row.get("heuristic_completed_share", 0.0)),
                    0.0,
                    1.0,
                ),
            ))
            for row in tail
        ]))
        training_reward_trend = self._normalized_smoothed_trend(training_rewards)
        evaluation_reward_trend = self._normalized_smoothed_trend(evaluation_rewards)
        validation_score_trend = self._normalized_smoothed_trend(
            validation_scores,
            window=3,
        )
        # Compare adjacent settled windows. The former reference used the
        # earliest post-reset quarter, so normal rival catch-up made a stable
        # late equilibrium look like non-convergence.
        reference_end = max(1, len(evaluation) - tail_size)
        reference_start = max(settled_start, reference_end - tail_size)
        pre_late_reward_window = evaluation[reference_start:reference_end]
        if not pre_late_reward_window:
            pre_late_reward_window = settled[: min(tail_size, len(settled))]
        pre_late_evaluation_reward = float(np.mean([
            float(row.get("reward", np.nan))
            for row in pre_late_reward_window
        ]))
        late_evaluation_reward = float(np.mean([
            float(row.get("reward", np.nan))
            for row in tail
        ]))
        reward_floor = float(
            getattr(
                getattr(self, "long_term_profit_reward_config", None),
                "minimum_reward",
                -2.0,
            )
        )
        # Shift by the declared reward floor before forming a ratio. Directly
        # dividing signed rewards turns a stable negative series into a huge
        # negative "retention" value even when nothing degraded.
        evaluation_reward_retention = float(
            (late_evaluation_reward - reward_floor + 1e-6)
            / max(1e-6, pre_late_evaluation_reward - reward_floor + 1e-6)
        )

        coeff_change_count = 0
        economic_change_count = 0
        exposure = {
            "base_fare": 1.0,
            "per_minute": 15.0,
            "per_mile": 4.0,
            "booking_fee": 1.0,
            "airport_fee": 0.05,
        }
        for previous, current in zip(tail, tail[1:]):
            deltas = [
                (
                    key,
                    abs(
                        float(current.get(f"firm1_{key}", np.nan))
                        - float(previous.get(f"firm1_{key}", np.nan))
                    ),
                )
                for key in self.shared_edit_keys
                if np.isfinite(float(current.get(f"firm1_{key}", np.nan)))
                and np.isfinite(float(previous.get(f"firm1_{key}", np.nan)))
            ]
            if any(delta > 1e-9 for _, delta in deltas):
                coeff_change_count += 1
            fare_equivalent_delta = float(sum(
                delta * float(exposure.get(key, 0.0))
                for key, delta in deltas
            ))
            if fare_equivalent_delta >= 0.05:
                economic_change_count += 1
        late_coefficient_change_rate = float(
            coeff_change_count / max(1, len(tail) - 1)
        )
        late_economic_change_rate = float(
            economic_change_count / max(1, len(tail) - 1)
        )
        effective_actions = [
            self._economic_action_group(int(row.get("action", -1)))
            for row in tail
        ]
        late_action_diversity = int(len(set(effective_actions)))

        objective = self._configured_primary_objective()
        # Dominance has its ordinary economic meaning: Firm 1 stays ahead of
        # Firm 2. Arbitrary +$0.25 and +10-point margins are not part of either
        # named objective and therefore cannot decide certification.
        share_dominance_gap = 0.0
        profit_dominance_gap = 0.0
        dominance_rate_floor = 0.90
        late_share_rolling_dominance_rate = rolling_dominance_rate(
            late_share_advantages,
            minimum_gap=share_dominance_gap,
        )
        late_profit_rolling_dominance_rate = rolling_dominance_rate(
            late_profit_advantages,
            minimum_gap=profit_dominance_gap,
        )
        late_objective_rolling_dominance_rate = rolling_dominance_rate(
            late_objective_advantages,
            minimum_gap=0.0,
        )
        failures: List[str] = []
        if len(evaluation) < DURABILITY_MIN_STEPS:
            failures.append("evaluation_horizon_too_short_for_durability")
        if objective == "competitiveness":
            if late_share_advantage < share_dominance_gap:
                failures.append("late_market_share_advantage_below_target")
            if late_share_rolling_dominance_rate < dominance_rate_floor:
                failures.append("late_market_share_dominance_not_consistent")
        elif objective == "profit_maximization":
            if late_own_profit <= 0.0:
                failures.append("late_own_profit_not_positive")
            if late_profit_advantage < profit_dominance_gap:
                failures.append("late_profit_advantage_below_target")
            if late_profit_rolling_dominance_rate < dominance_rate_floor:
                failures.append("late_profit_dominance_not_consistent")
        else:
            if late_objective_advantage < 0.0:
                failures.append("late_mixed_objective_advantage_below_parity")
            if late_objective_rolling_dominance_rate < dominance_rate_floor:
                failures.append("late_mixed_objective_dominance_not_consistent")
        minimum_validation_snapshots = 2 if training else 1
        if len(validation_scores) < minimum_validation_snapshots:
            failures.append("held_out_validation_snapshots_insufficient")
        # Validation score is a noisy diagnostic measured against fresh market
        # paths and openings.  A fitted slope across a handful of snapshots is
        # not an economic failure criterion: it can reject a final policy that
        # wins every held-out opening and remains dominant throughout the long
        # frozen-policy evaluation.  Durability is decided by the objective's
        # late rolling dominance and evaluation-reward retention below; the
        # score trend remains in the report for inspection only.
        if (
            not np.isfinite(evaluation_reward_retention)
            or evaluation_reward_retention < 0.75
        ):
            failures.append("late_evaluation_reward_not_retained")
        return {
            "passed": not failures,
            "failures": failures,
            "primary_objective": objective,
            "evaluation_days": int(len(evaluation)),
            "late_window_days": int(len(tail)),
            "late_completed_share_advantage": late_share_advantage,
            "late_completed_share_rolling_dominance_rate": (
                late_share_rolling_dominance_rate
            ),
            "late_market_share_advantage": late_share_advantage,
            "late_market_share_rolling_dominance_rate": (
                late_share_rolling_dominance_rate
            ),
            "late_own_profit_per_request": late_own_profit,
            "late_fulfillment_rate": late_fulfillment_rate,
            "late_completion_coverage": late_completion_coverage,
            "late_profit_advantage_per_request": late_profit_advantage,
            "late_profit_rolling_dominance_rate": (
                late_profit_rolling_dominance_rate
            ),
            "late_objective_advantage": late_objective_advantage,
            "late_objective_rolling_dominance_rate": (
                late_objective_rolling_dominance_rate
            ),
            "late_price_gap_abs_error": late_gap_abs_error,
            "target_price_gap": target_gap,
            "held_out_validation_snapshots": int(len(validation_scores)),
            "held_out_validation_score_fitted_change": validation_score_trend,
            "training_smoothed_reward_fitted_change": training_reward_trend,
            "settled_evaluation_smoothed_reward_fitted_change": evaluation_reward_trend,
            "pre_late_evaluation_reward": pre_late_evaluation_reward,
            # Compatibility alias retained for existing report readers.
            "early_settled_evaluation_reward": pre_late_evaluation_reward,
            "late_evaluation_reward": late_evaluation_reward,
            "late_evaluation_reward_retention": evaluation_reward_retention,
            "late_coefficient_change_rate": late_coefficient_change_rate,
            "late_fare_equivalent_change_rate": late_economic_change_rate,
            "late_action_diversity": late_action_diversity,
            "criteria": {
                "primary_objective": objective,
                "late_completed_share_advantage_min": (
                    share_dominance_gap
                    if objective == "competitiveness"
                    else None
                ),
                "late_market_share_advantage_min": (
                    share_dominance_gap
                    if objective == "competitiveness"
                    else None
                ),
                "late_completed_share_rolling_dominance_rate_min": (
                    dominance_rate_floor
                    if objective == "competitiveness"
                    else None
                ),
                "late_own_profit_per_request_min_exclusive": (
                    0.0 if objective == "profit_maximization" else None
                ),
                "late_fulfillment_rate_min": None,
                "late_completion_coverage_min": None,
                "late_profit_advantage_per_request_min": (
                    profit_dominance_gap
                    if objective == "profit_maximization"
                    else None
                ),
                "late_profit_rolling_dominance_rate_min": (
                    dominance_rate_floor
                    if objective == "profit_maximization"
                    else None
                ),
                "late_objective_advantage_min": (
                    0.0
                    if objective not in {
                        "profit_maximization", "competitiveness"
                    }
                    else None
                ),
                "late_objective_rolling_dominance_rate_min": (
                    dominance_rate_floor
                    if objective not in {
                        "profit_maximization", "competitiveness"
                    }
                    else None
                ),
                "late_price_gap_abs_error_max": (
                    None
                ),
                "held_out_validation_snapshots_min": (
                    minimum_validation_snapshots
                ),
                "evaluation_steps_min": DURABILITY_MIN_STEPS,
                "held_out_validation_score_fitted_change_min": None,
                "late_evaluation_reward_retention_min": 0.75,
                "late_fare_equivalent_change_rate_min": None,
                "late_action_diversity_min": None,
            },
        }

    def run_experiment(
        self,
        train_timesteps: int = 6000,
        train_customers_per_step: int = 5000,
        eval_timesteps: int = 1000,
        eval_customers_per_step: int = 1000,
        profiles_out: Optional[str] = None,
        profiles_log_limit: int = 200000,
        train_steps_per_day: int = 10,
        ppo_update_interval_days: int = 10,
        ppo_min_rollout_transitions: int = 96,
        stochastic_training: bool = True,
        firm1_action_interval_steps: int = -1,
        firm2_action_interval_days: int = -1,
        eval_policy_mode: str = "argmax_raw",
        eval_policy_temperature: float = 0.25,
        eval_policy_seed: Optional[int] = None,
        eval_guardrail_mode: str = "off",
        checkpoint_validation_horizon: int = 0,
        checkpoint_validation_interval_days: int = 400,
        checkpoint_validation_customers_per_step: int = 0,
        checkpoint_validation_openings: int = 3,
        evaluate_benchmark_suite: bool = False,
        training_enabled: bool = True,
        trained_model_out: Optional[str] = None,
        model_registry_dir: Optional[str] = "artifacts/trained_models",
        trained_model_id: Optional[str] = None,
    ):
        """Run workflow: synthetic-data RL training (day/timestep cadence), then held-out evaluation."""
        training_enabled = bool(training_enabled)
        if not training_enabled:
            train_timesteps = 0
            stochastic_training = False
        train_steps_per_day = int(max(1, train_steps_per_day))
        loaded_steps_per_day = int(
            getattr(self, "loaded_steps_per_day", 0) or 0
        )
        if (
            loaded_steps_per_day > 0
            and train_steps_per_day != loaded_steps_per_day
        ):
            raise ValueError(
                "requested market steps per day do not match the trained policy: "
                f"requested={train_steps_per_day}, "
                f"artifact={loaded_steps_per_day}"
            )
        self.shared_steps_per_day = train_steps_per_day
        if (
            training_enabled
            and self.firm1_mode == "RL"
            and self.training_stage_scheduler.mode == "staged"
        ):
            if int(train_timesteps) < len(self.training_stage_scheduler.stages):
                raise ValueError(
                    "staged training requires at least three --train_timesteps "
                    "so market, static_opponent, and dynamic_opponent each run"
                )
            if str(self.firm2_mode) == "static":
                raise ValueError(
                    "staged training requires a dynamic --firm2_mode for its "
                    "final stage; use --training_curriculum direct only for a "
                    "static-opponent ablation"
                )
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
            if getattr(self, "active_calibration", None):
                self.apply_calibration(self.active_calibration)
            print(f">>> run_experiment stochastic seed={exp_seed} (independent of --seed for robustness)")
            self._initialize_arbitrary_starting_coefficients()
            self._initialize_run_distributions()
            if self.firm1_mode == "RL":
                self.firm1.reset_state_history()
        profile_demand_per_step = (
            train_customers_per_step
            if training_enabled
            else eval_customers_per_step
        )
        self._refresh_profile_pool(rides_per_timestep=profile_demand_per_step)
        train_profile_sample_size = self._effective_simulation_sample_size(train_customers_per_step, collect_rows=False)
        eval_profile_sample_size = self._effective_simulation_sample_size(eval_customers_per_step, collect_rows=False)

        print(
            f">>> Synthetic setup: profile_pool={self.agent_gen.total_customers}, "
            f"train days={train_timesteps} x {train_steps_per_day} steps/day x {train_customers_per_step} rides, "
            f"eval timesteps={eval_timesteps} x {eval_customers_per_step} rides, "
            f"mc_samples(train/eval)={train_profile_sample_size}/{eval_profile_sample_size}"
        )
        update_every = int(max(1, ppo_update_interval_days))
        requested_firm1_interval = int(firm1_action_interval_steps)
        requested_firm2_interval = int(firm2_action_interval_days)
        loaded_action_interval = int(
            getattr(self, "loaded_shared_action_interval_steps", 0) or 0
        )
        if (
            requested_firm1_interval > 0
            and requested_firm2_interval > 0
            and requested_firm1_interval != requested_firm2_interval
        ):
            raise ValueError(
                "Firm1 and Firm2 must use the same action interval; received "
                f"{requested_firm1_interval} and {requested_firm2_interval}"
            )
        shared_action_interval_steps = int(max(
            1,
            requested_firm1_interval
            if requested_firm1_interval > 0
            else requested_firm2_interval
            if requested_firm2_interval > 0
            else loaded_action_interval
            if loaded_action_interval > 0
            else int(np.ceil(update_every / 2.0)),
        ))
        if (
            loaded_action_interval > 0
            and shared_action_interval_steps != loaded_action_interval
        ):
            raise ValueError(
                "requested action interval does not match the trained policy: "
                f"requested={shared_action_interval_steps}, "
                f"artifact={loaded_action_interval}"
            )
        self.shared_action_interval_steps = shared_action_interval_steps
        action_interval_steps = shared_action_interval_steps
        base_action_interval_steps = shared_action_interval_steps
        firm2_interval_steps = shared_action_interval_steps
        if self.firm1_mode == "RL":
            approx_decisions = int(np.ceil(
                update_every * max(1, int(train_steps_per_day))
                / float(shared_action_interval_steps)
            ))
            trace_multiplier = float(
                self.firm1.agent.gamma**shared_action_interval_steps
                * self.firm1.agent.lam
            )
            effective_trace_actions = float(
                1.0 / max(1e-9, 1.0 - trace_multiplier)
            )
            actions_per_day = float(
                max(1, train_steps_per_day) / shared_action_interval_steps
            )
            print(
                f">>> Shared pricing cadence: Firm1 and Firm2 act every "
                f"{shared_action_interval_steps} market step(s); PPO checks for "
                f"an update every {update_every} day(s) "
                f"(~{approx_decisions} complete reaction-cycle transitions/check; "
                f"minimum rollout={max(1, int(ppo_min_rollout_transitions))}; "
                f"GAE effective horizon≈{effective_trace_actions:.0f} actions/"
                f"{effective_trace_actions / max(actions_per_day, 1e-9):.0f} days)."
            )
        
        if training_enabled:
            print(f">>> Training RL agent on synthetic {self.market_name} sampling (day/timestep cadence)...")
        else:
            print(">>> Evaluation-only mode: using loaded Firm1 policy without optimizer updates.")
        reward_history: List[float] = []
        sampled_profile_rows: List[Dict[str, Any]] = []
        profile_limit_reached = False

        opponent_mode = str(self.firm2_mode)
        dynamic_training_pool = self._training_opponent_population()
        stage_day_counts = Counter(
            self.training_stage_scheduler.stage_at(
                float(day) / max(1.0, float(train_timesteps))
            ).name
            for day in range(int(max(0, train_timesteps)))
        )
        print(
            ">>> Training curriculum: "
            + " -> ".join(
                f"{stage.name}({stage_day_counts.get(stage.name, 0)} day(s))"
                for stage in self.training_stage_scheduler.stages
            )
            + "; dynamic episode opponent population="
            + ",".join(dynamic_training_pool)
            + "."
        )
        benchmark_opponent = copy.deepcopy(self.firm2)
        training_clock = OperationalClock()
        
        update_every = int(max(1, ppo_update_interval_days))
        best_checkpoint: Optional[Dict[str, Any]] = None
        best_validation_score = -float("inf")
        best_validation_selection_key: Optional[Tuple[float, ...]] = None
        best_observed_checkpoint: Optional[Dict[str, Any]] = None
        best_observed_selection_key: Optional[Tuple[float, ...]] = None
        latest_checkpoint: Optional[Dict[str, Any]] = None
        minimum_rollout = int(max(1, ppo_min_rollout_transitions))
        self.ppo_min_rollout_transitions = minimum_rollout
        validation_interval = int(max(
            update_every,
            checkpoint_validation_interval_days,
        ))
        resolved_checkpoint_horizon = int(
            max(
                2,
                min(
                    1000,
                    eval_timesteps
                    if int(checkpoint_validation_horizon) <= 0
                    else int(checkpoint_validation_horizon),
                ),
            )
        )
        resolved_checkpoint_customers = int(
            eval_customers_per_step
            if int(checkpoint_validation_customers_per_step) <= 0
            else max(1, int(checkpoint_validation_customers_per_step))
        )
        resolved_checkpoint_openings = int(max(1, checkpoint_validation_openings))
        if training_enabled and self.firm1_mode == "RL":
            print(
                ">>> Checkpoint validation: "
                f"horizon={resolved_checkpoint_horizon}, "
                f"opponents={opponent_mode}"
                + (
                    "+five_benchmark_classes, "
                    if self.credit_assignment_v25
                    else ", "
                )
                + f"rides/step={resolved_checkpoint_customers}, "
                f"openings={resolved_checkpoint_openings}, "
                "score=40% full-horizon + 60% final-quarter mean of the "
                "same normalized objective; selection=final durable dynamic "
                "policy, with best-certified fallback only if final fails."
            )
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
            "action_q_cql_loss": np.nan,
            "response_loss": np.nan,
            "opponent_forecast_loss": np.nan,
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
            "target_entropy_fraction": float(self.firm1.agent.target_entropy_fraction) if self.firm1_mode == "RL" else np.nan,
            "entropy_fraction_error": np.nan,
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
        # Convergence is reported only. It never pauses PPO updates, changes
        # entropy, changes the action set, or selects an early policy.
        training_reward_stability_seen = False
        # Pricing decisions and their semi-MDP returns can span synthetic day
        # boundaries.  Keeping this state outside the reporting-day loop makes
        # credit follow the operational clock and the rival reaction cadence.
        pending_rl_step = None
        pending_reward_sum = 0.0
        pending_discount = 1.0
        pending_reward_weight = 0.0
        pending_count = 0
        previous_firm2_metrics: Optional[FirmMetrics] = None
        last_ppo_update_day = 0
        active_episode_end_day = 0
        active_episode_days = 0
        dynamic_episode_ordinal = 0
        active_episode_opponent_mode = opponent_mode
        for d in range(train_timesteps):
            # Elapsed progress (rather than completed progress) ensures day 1
            # belongs to the first stage and the final third belongs to the
            # configured dynamic opponent.
            train_progress = float(d) / max(1.0, float(train_timesteps))
            self.current_training_stage = self.training_stage_scheduler.stage_at(train_progress)
            stage_minimum_rollout = int(
                minimum_rollout
                if self.current_training_stage.opponent_regime == "dynamic"
                else min(96, minimum_rollout)
            )
            stage_changed = active_stage_name != self.current_training_stage.name
            episode_expired = bool(
                not stage_changed
                and self.current_training_stage.episode_days > 0
                and d >= active_episode_end_day
            )
            selected_mode = (
                active_episode_opponent_mode
                if self.current_training_stage.opponent_regime == "dynamic"
                else self._training_stage_opponent_mode(
                    self.current_training_stage, opponent_mode
                )
            )
            if stage_changed or episode_expired:
                if d > 0 and self.firm1_mode == "RL" and self.firm1.agent.buffer_size:
                    unresolved = any(
                        not transition.next_state_finalized
                        for transition in self.firm1.agent.buf
                    )
                    if unresolved:
                        raise RuntimeError(
                            "PPO rollout crossed a training episode boundary "
                            "with an unresolved transition"
                        )
                    if stage_changed:
                        raise RuntimeError(
                            "PPO rollout crossed a curriculum-stage boundary; "
                            "the mandatory stage flush did not run"
                        )
                if stage_changed:
                    dynamic_episode_ordinal = 0
                elif episode_expired:
                    dynamic_episode_ordinal += 1
                opponent_mode_override = (
                    dynamic_training_pool[
                        dynamic_episode_ordinal % len(dynamic_training_pool)
                    ]
                    if self.current_training_stage.opponent_regime == "dynamic"
                    else None
                )
                selected_mode = self._enter_training_stage(
                    self.current_training_stage,
                    configured_mode=opponent_mode,
                    episode_seed=self.seed + 50_000 + d,
                    opponent_mode_override=opponent_mode_override,
                )
                active_episode_opponent_mode = selected_mode
                training_clock.reset()
                previous_firm2_metrics = None
                active_stage_name = self.current_training_stage.name
                active_episode_days = int(
                    self._training_opponent_episode_days(
                        self.current_training_stage, selected_mode
                    )
                    if self.current_training_stage.episode_days > 0
                    else 0
                )
                active_episode_end_day = int(
                    min(
                        train_timesteps,
                        d + active_episode_days,
                    )
                    if self.current_training_stage.episode_days > 0
                    else 0
                )
                opponent_label = (
                    "city_market_reference"
                    if self.current_training_stage.opponent_regime == "market"
                    else selected_mode
                )
                print(
                    f">>> [training episode opening] day={d + 1} "
                    f"stage={self.current_training_stage.name} "
                    f"opponent={opponent_label} "
                    + (
                        f"horizon={active_episode_days}day(s) "
                        if active_episode_days > 0
                        else ""
                    )
                    +
                    f"episode_tariff_randomization=±"
                    f"{100.0 * self.current_training_stage.tariff_reset_fraction:.0f}% "
                    "policy_and_optimizers=retained"
                )
            next_stage_name = self.current_training_stage.name
            if (d + 1) < train_timesteps:
                next_progress = float(d + 1) / max(1.0, float(train_timesteps))
                next_stage_name = self.training_stage_scheduler.stage_at(
                    next_progress
                ).name
            stage_transition_boundary = bool(
                next_stage_name != self.current_training_stage.name
            )
            dynamic_episode_boundary = bool(
                self.current_training_stage.episode_days > 0
                and active_episode_end_day > 0
                and (d + 1) >= active_episode_end_day
                and (d + 1) < train_timesteps
            )
            day_regime_boundary = bool(
                stage_transition_boundary
                or dynamic_episode_boundary
                or (d + 1) == train_timesteps
            )
            mandatory_flush_boundary = bool(
                stage_transition_boundary
                or (
                    (d + 1) == train_timesteps
                    and not (
                        self.credit_assignment_v25
                        and self.current_training_stage.opponent_regime
                        == "dynamic"
                    )
                )
            )
            effective_firm2_interval_steps = int(max(
                1,
                firm2_interval_steps * self.current_training_stage.opponent_cadence_multiplier,
            ))
            # Both firms use one operational decision cycle. Firm 1's stored
            # return spans the entire interval before either firm acts again.
            stage_action_interval_steps = int(
                effective_firm2_interval_steps
            )
            day_ctx = self.market.sample_day_context()
            hours = [self.market.sample_timestep_hour().hour for _ in range(max(1, int(train_steps_per_day)))]

            reward_sum = 0.0
            base_reward_sum = 0.0
            raw_reward_sum = 0.0
            relative_profit_sum = 0.0
            price_competitiveness_sum = 0.0
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
            market_share_sum = 0.0
            firm2_market_share_sum = 0.0
            outside_option_share_sum = 0.0
            completion_coverage_sum = 0.0
            firm2_revpr_sum = 0.0
            firm2_profitpr_sum = 0.0
            revpr_sum = 0.0
            profitpr_sum = 0.0
            gap_sum = 0.0
            gap_abs_error_sum = 0.0
            gap_rmse_sum = 0.0
            gap_p90_sum = 0.0
            gap_within_tolerance_sum = 0.0
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
            action_mask_regimes: Counter[str] = Counter()
            firm2_leader_decision_count = 0
            firm2_leader_intervention_count = 0
            firm2_leader_action_sum = np.zeros(
                len(self.shared_edit_keys) + 1, dtype=np.float32
            )
            inline_ppo_metrics: Optional[Dict[str, Any]] = None
            boundary_flush_transitions = 0
            for t, hour in enumerate(hours):
                base = self.market.curr_market
                rl_step = None
                interval_closed_this_step = False
                firm2_due = bool(
                    not isinstance(self.firm2, FirmStaticPricer)
                    and not self.current_training_stage.freeze_opponent
                    and training_clock.due(effective_firm2_interval_steps)
                )
                firm2_action_target, firm2_intervened = (
                    self._firm2_leader_step(
                        previous_metrics=previous_firm2_metrics,
                        day_of_week=day_ctx.day_of_week,
                        hour=hour,
                        weather=day_ctx.weather,
                        enabled=firm2_due,
                    )
                )
                if firm2_due:
                    firm2_leader_decision_count += 1
                    firm2_leader_intervention_count += int(firm2_intervened)
                    firm2_leader_action_sum += firm2_action_target

                # Profiles are sampled before Firm 1 acts because optional
                # counterfactual branches must replay the exact same riders and
                # exogenous random stream as the selected real action.
                sampled_profiles = self._sample_profiles_from_pool(
                    train_profile_sample_size
                )
                if profiles_out and not profile_limit_reached:
                    remaining = int(max(
                        0, profiles_log_limit - len(sampled_profile_rows)
                    ))
                    if remaining > 0:
                        sampled_profile_rows.extend(
                            {
                                "Phase": "train",
                                "Day": int(d),
                                "Timestep": int(t),
                                **profile,
                            }
                            for profile in sampled_profiles[:remaining]
                        )
                    profile_limit_reached = (
                        len(sampled_profile_rows) >= int(max(0, profiles_log_limit))
                    )

                # Firm 2 has now committed.  This publication builds Firm 1's
                # state from the new posted quotes, the current market context,
                # and lagged supply/customer outcomes before Firm 1 reacts.
                if self.firm1_mode == "RL":
                    raw_state_vec, action_features, _ = self._publish_pricing_observation(
                        day_of_week=day_ctx.day_of_week, hour=hour, weather=day_ctx.weather
                    )
                    s_vec = self.firm1.stack_state(raw_state_vec)
                    critic_state = (
                        self._build_privileged_critic_state(raw_state_vec)
                        if self.privileged_critic
                        else None
                    )
                    decision_due = training_clock.due(
                        stage_action_interval_steps
                    )
                    if decision_due:
                        # A held-action transition ends after its final market
                        # outcome, but its true semi-MDP successor is the next
                        # state on which the actor can actually choose: after
                        # the new context and Firm 2 leader quote are public.
                        # Finalize deferred value/potential targets here, before
                        # either PPO updates or the next action is sampled.
                        if (
                            self.firm1.agent.buffer_size > 0
                            and not self.firm1.agent.buf[-1].next_state_finalized
                        ):
                            next_v, _, _ = self.firm1.agent.value_estimates(
                                s_vec,
                                action_features=action_features,
                                critic_state=critic_state,
                            )
                            self.firm1.agent.finalize_last_transition(
                                next_value=next_v,
                                next_potential=(
                                    self._public_reward_potential(raw_state_vec)
                                    if self.potential_shaping_scale > 0.0
                                    else None
                                ),
                            )
                            pre_action_update_due = bool(
                                not day_regime_boundary
                                and (d + 1 - last_ppo_update_day) >= update_every
                                and self.firm1.agent.buffer_size >= stage_minimum_rollout
                            )
                            if pre_action_update_due:
                                self._sync_agent_optimization_context()
                                inline_ppo_metrics = self.firm1.agent.update(
                                    epochs=self.ppo_update_epochs,
                                    batch_size=self.ppo_batch_size,
                                    bootstrap_value=next_v,
                                )
                                last_ppo_metrics = dict(inline_ppo_metrics)
                                last_ppo_update_day = int(d + 1)
                        action_mask = self._objective_action_mask(
                            self.firm1.feasible_action_mask(self.market)
                        )
                        action_mask_regimes[
                            str(self.last_action_mask_regime)
                        ] += 1
                        raw_diag = self.firm1.agent.policy_diagnostics(
                            s_vec,
                            action_features=action_features,
                            action_mask=action_mask,
                        )
                        action, s_ts, old_logp, val, af_ts, mask_ts = self.firm1.agent.act(
                            s_vec,
                            action_features=action_features,
                            action_mask=action_mask,
                            critic_state=critic_state,
                        )
                        counterfactual_values, counterfactual_mask = (
                            self._counterfactual_action_values(
                                selected_action=int(action),
                                policy_top_action=int(
                                    raw_diag.get("policy_top_action", action)
                                ),
                                policy_second_action=int(
                                    raw_diag.get("policy_second_action", action)
                                ),
                                action_mask=np.asarray(action_mask, dtype=bool),
                                day_of_week=day_ctx.day_of_week,
                                hour=hour,
                                weather=day_ctx.weather,
                                customers_per_step=train_customers_per_step,
                                sampled_profiles=sampled_profiles,
                            )
                            if self.counterfactual_branches > 0
                            else (
                                np.full(
                                    self.firm1.agent.action_dim,
                                    np.nan,
                                    dtype=np.float32,
                                ),
                                np.zeros(
                                    self.firm1.agent.action_dim,
                                    dtype=bool,
                                ),
                            )
                        )
                        # Branch restoration replaces the copied market object.
                        # Refresh this local reference before the real action.
                        base = self.market.curr_market
                        self.firm1.apply_action(action, self.market)
                        action_counts[int(action)] += 1
                        raw_top = int(raw_diag.get("policy_top_action", -1))
                        raw_top_action_counts[raw_top] += 1
                        sampled_not_top_count += int(raw_top >= 0 and int(action) != raw_top)
                        for diag_key, diag_value in raw_diag.items():
                            train_policy_diag_sums[diag_key] += float(diag_value)
                        train_policy_diag_count += 1
                        last_action = int(action)
                        rl_step = (
                            action, s_ts, old_logp, val, af_ts, mask_ts,
                            critic_state, counterfactual_values,
                            counterfactual_mask,
                        )
                        pending_rl_step = rl_step
                        pending_reward_sum = 0.0
                        pending_discount = 1.0
                        pending_reward_weight = 0.0
                        pending_count = 0
                    else:
                        rl_step = None
                        self.firm1.agent.observe_recurrent_state(
                            s_vec, critic_state=critic_state
                        )
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
                _, m1, m2, mean_gap, _, _ = self.simulate_batch(
                    day_of_week=day_ctx.day_of_week,
                    weather=day_ctx.weather,
                    hour=hour,
                    customers_per_step=train_customers_per_step,
                    sampled_profiles=sampled_profiles,
                    collect_rows=False,
                )
                self._ingest_platform_observations(m1, m2)
                previous_firm2_metrics = m2
                    
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
                    prev_rival_profit_per_request=float(
                        self.last_firm2_profitpr
                    ),
                    prev_rival_market_share=float(self.last_firm2_share),
                    prev_fulfillment_rate=float(self.last_fulfillment),
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
                        price_gap_abs_error=float(
                            self.last_price_gap_metrics["mae"]
                        ),
                    )

                if self.firm1_mode == "RL" and pending_rl_step is not None:
                    reward = float(reward_diag["reward"])
                    self.last_reward = reward
                    pending_reward_sum += pending_discount * reward
                    pending_reward_weight += pending_discount
                    pending_count += 1
                    pending_discount *= float(self.firm1.agent.gamma)
                    end_of_day = (t == len(hours) - 1)
                    episode_truncated = bool(end_of_day and day_regime_boundary)
                    interval_closed = bool(
                        ((training_clock.period + 1) % stage_action_interval_steps)
                        == 0
                        or episode_truncated
                    )
                    if interval_closed and pending_count > 0:
                        (
                            action, s_ts, old_logp, val, af_ts, mask_ts,
                            critic_state, counterfactual_values,
                            counterfactual_mask,
                        ) = pending_rl_step
                        # A curriculum/opponent episode reset changes tariffs,
                        # observation history, opponent controller, and reward-
                        # rate context. It is therefore a true learning episode
                        # boundary, not a time-limit cut whose pre-reset value can
                        # be bootstrapped into the next regime.
                        next_v = None
                        next_constraint_v = None
                        next_risk_v = None
                        transition_reward = float(pending_reward_sum)
                        if not self.credit_assignment_v25:
                            transition_reward /= float(max(
                                1e-12, pending_reward_weight
                            ))
                        self.firm1.agent.store(
                            s_ts,
                            action,
                            transition_reward,
                            episode_truncated,
                            None,
                            old_logp,
                            val,
                            action_features=af_ts,
                            action_mask=mask_ts,
                            discount=float(pending_discount),
                            duration=int(pending_count),
                            truncated=episode_truncated,
                            next_value=next_v,
                            next_constraint_values=next_constraint_v,
                            next_risk_value=next_risk_v,
                            critic_state=critic_state,
                            counterfactual_action_values=counterfactual_values,
                            counterfactual_action_mask=counterfactual_mask,
                            potential=(
                                self._public_reward_potential(
                                    np.asarray(
                                        s_ts[-PlatformObservationModel.observation_dim:].cpu().tolist(),
                                        dtype=np.float32,
                                    )
                                )
                                if self.potential_shaping_scale > 0.0
                                else None
                            ),
                            next_potential=None,
                            reward_regime=(
                                "city_market_reference"
                                if self.current_training_stage.opponent_regime
                                == "market"
                                else str(active_episode_opponent_mode)
                            ),
                        )
                        interval_closed_this_step = True
                        pending_rl_step = None
                        pending_reward_sum = 0.0
                        pending_discount = 1.0
                        pending_reward_weight = 0.0
                        pending_count = 0
                    if end_of_day:
                        before_snapshot = self._coeff_snapshot()
                        guardrail_diag = self._apply_eval_guardrail(
                            "off",
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
                relative_profit_sum += float(
                    reward_diag["reward_profit_advantage_component"]
                )
                price_competitiveness_sum += float(
                    reward_diag[
                        "reward_market_share_competitiveness_component"
                    ]
                )
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
                    "reward_active_objective_component",
                    "reward_active_objective_term_count",
                    "reward_dominance_term",
                    "reward_own_profit_utility",
                    "reward_profit_advantage_utility",
                    "reward_market_share_utility",
                    "reward_market_share_level_utility",
                    "reward_market_share_dominance_utility",
                    "reward_market_share_competitiveness_component",
                    "reward_market_share_level_component",
                    "reward_market_share_advantage_component",
                    "reward_market_share_advantage",
                    "reward_market_share_target_gap",
                    "reward_market_share_target_level",
                    "reward_market_share_target_progress",
                    "reward_market_share_target_achieved",
                    "reward_market_share_gap_error",
                    "reward_market_share_target_shortfall",
                    "reward_conditional_choice_share",
                    "reward_rival_conditional_choice_share",
                    "reward_conditional_choice_share_sum",
                    "reward_own_profit_component",
                    "reward_profit_advantage_component",
                    "reward_intervention_cost",
                    "reward_reversal_cost",
                    "reward_intervention_magnitude",
                    "reward_profit_advantage_per_request",
                    "reward_actor_credit",
                    "reward_previous_objective_base",
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
                _, _, outside_option_share = (
                    conditional_competitive_shares(
                        m1.chosen_share, m2.chosen_share
                    )
                )
                market_share = float(m1.completed_share)
                firm2_market_share = float(m2.completed_share)
                market_share_sum += market_share
                firm2_market_share_sum += firm2_market_share
                outside_option_share_sum += outside_option_share
                completion_coverage_sum += float(np.clip(
                    m1.completed_share + m2.completed_share, 0.0, 1.0
                ))
                firm2_revpr_sum += float(m2.rev_per_request)
                firm2_profitpr_sum += float(m2.profit_per_request)
                revpr_sum += float(m1.rev_per_request)
                profitpr_sum += float(m1.profit_per_request)
                gap_sum += float(mean_gap)
                gap_abs_error_sum += float(self.last_price_gap_metrics["mae"])
                gap_rmse_sum += float(self.last_price_gap_metrics["rmse"])
                gap_p90_sum += float(
                    self.last_price_gap_metrics["p90_abs_error"]
                )
                gap_within_tolerance_sum += float(
                    self.last_price_gap_metrics["within_tolerance_rate"]
                )
                fulfillment_sum += float(m1.fulfillment_rate)
                wait_sum += float(m1.avg_wait_minutes)
                driver_accept_sum += float(m1.driver_acceptance_rate)
                driver_paypr_sum += float(m1.driver_pay / max(1, m1.total))
                self._update_recent_response_emas(m1, mean_gap, m2=m2)
                self.last_share = market_share
                self.last_revpr = float(m1.rev_per_request)
                self.last_gap = float(mean_gap)
                self.last_profitpr = float(m1.profit_per_request)
                self.last_fulfillment = float(m1.fulfillment_rate)
                self.last_completion_coverage = float(np.clip(
                    m1.completed_share + m2.completed_share, 0.0, 1.0
                ))
                self.last_acceptance = float(m1.driver_acceptance_rate)
                self.last_wait = float(m1.avg_wait_minutes)
                self.last_driver_paypr = float(m1.driver_pay / max(1, m1.total))
                self.last_firm2_share = firm2_market_share
                self.last_firm2_revpr = float(m2.rev_per_request)
                self.last_firm2_profitpr = float(m2.profit_per_request)
                self.last_firm2_fulfillment = float(m2.fulfillment_rate)
                self.last_firm2_acceptance = float(m2.driver_acceptance_rate)
                self.last_firm2_wait = float(m2.avg_wait_minutes)
                self.last_firm2_driver_paypr = float(m2.driver_pay / max(1, m2.total))
                self.objective_credit_ready = True
                training_clock.advance()

                end_of_day = (t == len(hours) - 1)
                update_due = bool(
                    (mandatory_flush_boundary and end_of_day)
                    or (
                        dynamic_episode_boundary
                        and end_of_day
                        and self.firm1.agent.buffer_size >= stage_minimum_rollout
                    )
                    or (
                        not day_regime_boundary
                        and (d + 1 - last_ppo_update_day) >= update_every
                    )
                )
                if (
                    self.firm1_mode == "RL"
                    and interval_closed_this_step
                    and update_due
                    and (
                        self.firm1.agent.buffer_size >= stage_minimum_rollout
                        or (
                            mandatory_flush_boundary
                            and self.firm1.agent.buffer_size > 0
                        )
                    )
                    and pending_rl_step is None
                    and self.firm1.agent.buf[-1].next_state_finalized
                ):
                    buffered_transitions = int(
                        self.firm1.agent.buffer_size
                    )
                    if buffered_transitions < stage_minimum_rollout:
                        boundary_flush_transitions = buffered_transitions
                        print(
                            ">>> [PPO boundary flush] learning from "
                            f"{buffered_transitions} complete transition(s) "
                            "before the continuing-task boundary; no recent "
                            "dynamic-market experience is discarded."
                        )
                    self._sync_agent_optimization_context()
                    (
                        bootstrap_v,
                        bootstrap_constraint_v,
                        bootstrap_risk_v,
                    ) = self._ppo_bootstrap_estimates(
                        day_of_week=day_ctx.day_of_week,
                        hour=hour,
                        weather=day_ctx.weather,
                    )
                    inline_ppo_metrics = self.firm1.agent.update(
                        epochs=self.ppo_update_epochs,
                        batch_size=self.ppo_batch_size,
                        bootstrap_value=bootstrap_v,
                        bootstrap_constraint_values=bootstrap_constraint_v,
                        bootstrap_risk_value=bootstrap_risk_v,
                    )
                    last_ppo_metrics = dict(inline_ppo_metrics)
                    last_ppo_update_day = int(d + 1)

            ppo_metrics = {
                **{
                    key: np.nan
                    for key in (
                        "loss", "policy_loss", "value_loss", "constraint_value_loss",
                        "risk_value_loss", "action_q_loss", "action_q_cql_loss", "response_loss", "opponent_forecast_loss", "lagrangian_adv_mean",
                        "lagrangian_adv_std", "constraint_lambda_mean", "risk_coeff",
                        "approx_kl", "clipfrac", "entropy", "policy_entropy",
                        "discrete_policy_entropy", "magnitude_entropy", "explained_variance",
                        "rollout_action_diversity", "rollout_reward_std", "credited_reward_std",
                        "continuous_magnitude_mean", "continuous_magnitude_std",
                        "raw_policy_argmax_concentration", "raw_policy_hold_probability",
                        "state_action_sensitivity", "state_action_mi",
                        "direct_actor_credit_mean", "direct_actor_credit_std",
                        "direct_actor_credit_weight",
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
            if inline_ppo_metrics is not None:
                ppo_metrics = dict(inline_ppo_metrics)
            if self.firm1_mode == "RL":
                boundary_flush_due = bool(
                    mandatory_flush_boundary
                    and self.firm1.agent.buffer_size > 0
                )
                should_update = bool(
                    (
                        (d + 1) % update_every == 0
                        or day_regime_boundary
                        or (d + 1) == train_timesteps
                    )
                    and (
                        self.firm1.agent.buffer_size >= stage_minimum_rollout
                        or boundary_flush_due
                    )
                    and pending_rl_step is None
                    and inline_ppo_metrics is None
                    and self.firm1.agent.buf[-1].next_state_finalized
                )
                if should_update:
                    buffered_transitions = int(
                        self.firm1.agent.buffer_size
                    )
                    if buffered_transitions < stage_minimum_rollout:
                        boundary_flush_transitions = buffered_transitions
                        print(
                            ">>> [PPO boundary flush] learning from "
                            f"{buffered_transitions} complete transition(s) "
                            "before the continuing-task boundary; no recent "
                            "dynamic-market experience is discarded."
                        )
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
                    last_ppo_update_day = int(d + 1)
                
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
            avg_market_share = float(market_share_sum / n_steps)
            avg_firm2_market_share = float(firm2_market_share_sum / n_steps)
            avg_outside_option_share = float(
                outside_option_share_sum / n_steps
            )
            avg_completion_coverage = float(
                completion_coverage_sum / n_steps
            )
            avg_firm2_revpr = float(firm2_revpr_sum / n_steps)
            avg_firm2_profitpr = float(firm2_profitpr_sum / n_steps)
            avg_revpr = float(revpr_sum / n_steps)
            avg_profitpr = float(profitpr_sum / n_steps)
            avg_gap = float(gap_sum / n_steps)
            avg_gap_abs_error = float(gap_abs_error_sum / n_steps)
            avg_gap_rmse = float(gap_rmse_sum / n_steps)
            avg_gap_p90 = float(gap_p90_sum / n_steps)
            avg_gap_within_tolerance = float(
                gap_within_tolerance_sum / n_steps
            )
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
                "profile_population": "same_market_training_partition",
                "profile_generalization_fraction": float(
                    self.training_generalization_fraction
                ),
                "policy_objective": str(
                    self.long_term_profit_reward_config.objective_mode
                ),
                "reward_mechanism": str(
                    self.long_term_profit_reward_model.mechanism_name
                ),
                "avg_reward": float(avg_reward),
                "reward_moving_avg20": moving_avg20,
                "reward_moving_std20": moving_std20,
                "reward_base": float(base_reward_sum / n_steps),
                "reward_raw": float(raw_reward_sum / n_steps),
                "reward_dominance_term": float(
                    reward_component_sums["reward_dominance_term"]
                    / n_steps
                ),
                "reward_competitive_term": float(
                    price_competitiveness_sum / n_steps
                ),
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
                "avg_market_share": avg_market_share,
                "avg_firm2_market_share": avg_firm2_market_share,
                "avg_market_share_sum": float(
                    avg_market_share + avg_firm2_market_share
                ),
                "avg_outside_option_share": avg_outside_option_share,
                "avg_completion_coverage": avg_completion_coverage,
                "avg_rev_per_request": avg_revpr,
                "avg_profit_per_request": avg_profitpr,
                "avg_firm2_rev_per_request": avg_firm2_revpr,
                "avg_firm2_profit_per_request": avg_firm2_profitpr,
                "avg_profit_advantage_per_request": (
                    avg_profitpr - avg_firm2_profitpr
                ),
                "avg_price_gap_f2_minus_f1": avg_gap,
                "avg_price_gap_abs_error": avg_gap_abs_error,
                "avg_price_gap_rmse": avg_gap_rmse,
                "avg_price_gap_p90_abs_error": avg_gap_p90,
                "avg_price_gap_within_tolerance_rate": avg_gap_within_tolerance,
                "price_gap_target": float(self.reward_target_price_gap),
                "price_gap_tolerance": float(
                    self.constraint_config.overall_tolerance
                ),
                "avg_gap_violation_025": float(avg_gap_abs_error > 0.25),
                "avg_gap_violation_050": float(avg_gap_abs_error > 0.50),
                "constraint_curriculum_scale": float(self.constraint_curriculum_scale),
                "training_stage": str(self.current_training_stage.name),
                "training_stage_constraint_scale": float(self.current_training_stage.constraint_scale),
                "training_stage_action_hold_steps": int(stage_action_interval_steps),
                "training_stage_opponent_frozen": bool(self.current_training_stage.freeze_opponent),
                "training_stage_opponent_regime": str(
                    self.current_training_stage.opponent_regime
                ),
                "training_stage_boundary_flush_transitions": int(
                    boundary_flush_transitions
                ),
                "competition_protocol": "firm2_leader_then_firm1_follower",
                "training_configured_firm2_mode": str(opponent_mode),
                "training_active_firm2_mode": str(
                    "city_market_reference"
                    if self.current_training_stage.opponent_regime == "market"
                    else selected_mode
                ),
                "firm2_leader_decision_count": int(
                    firm2_leader_decision_count
                ),
                "firm2_leader_intervention_count": int(
                    firm2_leader_intervention_count
                ),
                "firm2_leader_intervention_rate": float(
                    firm2_leader_intervention_count
                    / max(1, firm2_leader_decision_count)
                ),
                "firm2_leader_action_steps_mean": json.dumps(
                    (
                        firm2_leader_action_sum
                        / max(1, firm2_leader_decision_count)
                    ).astype(float).tolist()
                ),
                "avg_fulfillment_rate": avg_fulfillment,
                "avg_wait_minutes": avg_wait,
                "driver_acceptance_rate": avg_driver_accept,
                "driver_pay_per_request": avg_driver_paypr,
                "last_action": int(last_action),
                "last_action_steps": json.dumps(last_action_steps, sort_keys=True),
                "dominant_action": dominant_action,
                "dominant_action_label": dominant_action_label,
                "dominant_action_steps": json.dumps(dominant_action_steps, sort_keys=True),
                "dominant_action_rate": dominant_action_rate,
                "action_counts": json.dumps(dict(sorted(action_counts.items())), sort_keys=True),
                "raw_top_action_counts": json.dumps(dict(sorted(raw_top_action_counts.items())), sort_keys=True),
                "action_mask_regimes": json.dumps(
                    dict(sorted(action_mask_regimes.items())), sort_keys=True
                ),
                "last_action_mask_regime": str(
                    self.last_action_mask_regime
                ),
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
                "objective_optimizer_ready": bool(
                    avg_policy_diag.get("optimizer_ready", 0.0) >= 0.5
                ),
                "objective_optimizer_action_change_rate": float(
                    avg_policy_diag.get("optimizer_action_changed", 0.0)
                ),
                "objective_optimizer_q_spread": float(
                    avg_policy_diag.get("optimizer_q_spread", 0.0)
                ),
                "objective_optimizer_action_margin": float(
                    avg_policy_diag.get("optimizer_action_margin", 0.0)
                ),
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
                "ppo_action_q_cql_loss": float(
                    ppo_metrics.get("action_q_cql_loss", 0.0)
                ),
                "ppo_counterfactual_policy_loss": float(
                    ppo_metrics.get("counterfactual_policy_loss", 0.0)
                ),
                "ppo_objective_q_optimizer_steps": int(
                    ppo_metrics.get("objective_q_optimizer_steps", 0)
                ),
                "ppo_objective_q_update_calls": int(
                    ppo_metrics.get("objective_q_update_calls", 0)
                ),
                "ppo_counterfactual_target_coverage": float(
                    ppo_metrics.get("counterfactual_target_coverage", 0.0)
                ),
                "counterfactual_action_coverage": float(
                    np.mean(self.counterfactual_action_counts > 0)
                    if self.counterfactual_action_counts.size
                    else 0.0
                ),
                "counterfactual_action_count_cv": float(
                    np.std(self.counterfactual_action_counts)
                    / max(1e-8, float(np.mean(self.counterfactual_action_counts)))
                    if np.any(self.counterfactual_action_counts > 0)
                    else 0.0
                ),
                "counterfactual_action_counts": json.dumps(
                    self.counterfactual_action_counts.astype(int).tolist()
                ),
                "ppo_counterfactual_actor_weight": float(
                    ppo_metrics.get("counterfactual_actor_weight", 0.0)
                ),
                "ppo_potential_shaping_delta_mean": float(
                    ppo_metrics.get("potential_shaping_delta_mean", 0.0)
                ),
                "ppo_average_reward_rate": float(
                    ppo_metrics.get("average_reward_rate", 0.0)
                ),
                "ppo_rollout_reward_exposure": float(
                    ppo_metrics.get("rollout_reward_exposure", 0.0)
                ),
                "ppo_rollout_reward_rate_residual": float(
                    ppo_metrics.get("rollout_reward_rate_residual", 0.0)
                ),
                "ppo_reward_rate_segment_count": int(
                    ppo_metrics.get("reward_rate_segment_count", 0)
                ),
                "ppo_reward_rate_segment_std": float(
                    ppo_metrics.get("reward_rate_segment_std", 0.0)
                ),
                "ppo_reward_rate_lagged_baseline_fraction": float(
                    ppo_metrics.get(
                        "reward_rate_lagged_baseline_fraction", 0.0
                    )
                ),
                "ppo_opponent_segment_count": int(
                    ppo_metrics.get("opponent_segment_count", 0)
                ),
                "ppo_opponent_regime_count": int(
                    ppo_metrics.get("opponent_regime_count", 0)
                ),
                "ppo_opponent_sample_weight_min": float(
                    ppo_metrics.get("opponent_sample_weight_min", 1.0)
                ),
                "ppo_opponent_sample_weight_max": float(
                    ppo_metrics.get("opponent_sample_weight_max", 1.0)
                ),
                "ppo_reward_rate_beta": float(
                    ppo_metrics.get("reward_rate_beta", 0.0)
                ),
                "ppo_response_loss": float(ppo_metrics.get("response_loss", 0.0)),
                "ppo_opponent_forecast_loss": float(
                    ppo_metrics.get("opponent_forecast_loss", 0.0)
                ),
                "ppo_lagrangian_adv_mean": float(ppo_metrics.get("lagrangian_adv_mean", 0.0)),
                "ppo_lagrangian_adv_std": float(ppo_metrics.get("lagrangian_adv_std", 0.0)),
                "ppo_direct_actor_credit_mean": float(
                    ppo_metrics.get("direct_actor_credit_mean", 0.0)
                ),
                "ppo_direct_actor_credit_std": float(
                    ppo_metrics.get("direct_actor_credit_std", 0.0)
                ),
                "ppo_direct_actor_credit_weight": float(
                    ppo_metrics.get("direct_actor_credit_weight", 0.0)
                ),
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
                "ppo_target_entropy_fraction": float(
                    ppo_metrics.get("target_entropy_fraction", np.nan)
                ),
                "ppo_entropy_fraction_error": float(
                    ppo_metrics.get("entropy_fraction_error", np.nan)
                ),
                "ppo_clip_eps": float(ppo_metrics.get("clip_eps", 0.0)),
                "ppo_exploration_rate": float(ppo_metrics.get("exploration_rate", 0.0)),
                "ppo_action_coverage": float(ppo_metrics.get("action_coverage", 0.0)),
                "ppo_explained_variance": float(ppo_metrics.get("explained_variance", 0.0)),
                "ppo_optimizer_steps": int(ppo_metrics.get("optimizer_steps", 0)),
                "ppo_actor_optimizer_steps": int(ppo_metrics.get("actor_optimizer_steps", 0)),
                "ppo_critic_optimizer_steps": int(ppo_metrics.get("critic_optimizer_steps", 0)),
                "ppo_actor_epochs_requested": int(
                    ppo_metrics.get("actor_epochs_requested", 0)
                ),
                "ppo_critic_epochs_requested": int(
                    ppo_metrics.get("critic_epochs_requested", 0)
                ),
                "ppo_lr": float(ppo_metrics.get("lr", 0.0)),
                "ppo_critic_lr": float(ppo_metrics.get("critic_lr", 0.0)),
                "ppo_critic_target_mean": float(ppo_metrics.get("critic_target_mean", np.nan)),
                "ppo_critic_target_std": float(ppo_metrics.get("critic_target_std", np.nan)),
                "ppo_gae_return_mean": float(
                    ppo_metrics.get("gae_return_mean", np.nan)
                ),
                "ppo_gae_return_std": float(
                    ppo_metrics.get("gae_return_std", np.nan)
                ),
                "ppo_critic_target_contract": str(
                    ppo_metrics.get("critic_target_contract", "gae_lambda_return")
                ),
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
                "ppo_transition_duration_mean": float(
                    ppo_metrics.get("transition_duration_mean", np.nan)
                ),
                "ppo_transition_discount_mean": float(
                    ppo_metrics.get("transition_discount_mean", np.nan)
                ),
                "ppo_truncation_count": int(
                    ppo_metrics.get("truncation_count", 0)
                ),
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
                # Validate the same seeded actor distribution used for final
                # evaluation. There is no learned Q reranker or
                # deployment-only action rule.
                self.firm1.agent.deployment_optimizer_enabled = False
                validation = self._adaptive_policy_validation(
                    customers_per_step=resolved_checkpoint_customers,
                    profile_sample_size=resolved_checkpoint_customers,
                    horizon=resolved_checkpoint_horizon,
                    aggressive_horizon=resolved_checkpoint_horizon,
                    policy_mode=eval_policy_mode,
                    policy_temperature=eval_policy_temperature,
                    policy_seed=eval_policy_seed,
                    opponent_interval_steps=firm2_interval_steps,
                    firm1_interval_steps=firm2_interval_steps,
                    steps_per_day=train_steps_per_day,
                    aggressive_seed_count=resolved_checkpoint_openings,
                    include_benchmarks=bool(self.credit_assignment_v25),
                )
                validation = dict(validation)
                validation["deployment_variant"] = "raw_actor"
                validation["deployment_policy_mode"] = str(eval_policy_mode)
                latest_validation = dict(validation)
                validation_score = float(validation["score"])
                log_row["validation_reward"] = float(validation["reward"])
                log_row["validation_score"] = float(validation_score)
                log_row["validation_critic_td_explained_variance"] = float(
                    validation["heldout_critic_td_explained_variance"]
                )
                log_row["validation_critic_td_mae"] = float(
                    validation["heldout_critic_td_mae"]
                )
                log_row["validation_critic_td_samples"] = int(
                    validation["heldout_critic_td_samples"]
                )
                log_row["validation_objective_optimizer"] = str(
                    validation["objective_optimizer"]
                )
                log_row["validation_feasible_mode_fraction"] = float(
                    validation["objective_feasible_mode_fraction"]
                )
                log_row["validation_maximum_normalized_shortfall"] = float(
                    validation["objective_maximum_normalized_shortfall"]
                )
                log_row["validation_mean_objective_dominance_rate"] = float(
                    validation["mean_objective_dominance_rate"]
                )
                log_row["validation_worst_objective_dominance_rate"] = float(
                    validation["worst_objective_dominance_rate"]
                )
                log_row["validation_deployment_variant"] = str(
                    validation["deployment_variant"]
                )
                log_row["validation_raw_actor_score"] = float(
                    validation["score"]
                )
                log_row["validation_share"] = float(validation["share"])
                log_row["validation_completed_share"] = float(
                    validation["completed_share"]
                )
                log_row["validation_firm2_completed_share"] = float(
                    validation["rival_completed_share"]
                )
                log_row["validation_rev_per_request"] = float(validation["rev_per_request"])
                log_row["validation_firm2_rev_per_request"] = float(validation["rival_rev_per_request"])
                log_row["validation_profit_per_request"] = float(validation["profit_per_request"])
                log_row["validation_firm2_profit_per_request"] = float(validation["rival_profit_per_request"])
                log_row["validation_fulfillment_rate"] = float(validation["fulfillment_rate"])
                log_row["validation_mean_fulfillment_rate"] = float(
                    validation["mean_fulfillment_rate"]
                )
                log_row["validation_worst_fulfillment_rate"] = float(
                    validation["worst_fulfillment_rate"]
                )
                log_row["validation_completion_coverage"] = float(
                    validation["completion_coverage"]
                )
                log_row["validation_mean_completion_coverage"] = float(
                    validation["mean_completion_coverage"]
                )
                log_row["validation_worst_completion_coverage"] = float(
                    validation["worst_completion_coverage"]
                )
                log_row["validation_price_gap_f2_minus_f1"] = float(validation["price_gap"])
                log_row["validation_action_diversity"] = float(validation["action_diversity"])
                log_row["validation_action_switch_rate"] = float(validation["action_switch_rate"])
                log_row["validation_profile_source"] = str(
                    validation["validation_profile_source"]
                )
                log_row["validation_cities"] = json.dumps(
                    validation["mode_validation_cities"], sort_keys=True
                )
                log_row["validation_mean_profit_advantage"] = float(
                    validation["mean_profit_advantage"]
                )
                log_row["validation_worst_profit_advantage"] = float(
                    validation["worst_profit_advantage"]
                )
                log_row["validation_profit_win_rate"] = float(
                    validation["profit_win_rate"]
                )
                log_row["validation_mean_completed_share_advantage"] = float(
                    validation["mean_completed_share_advantage"]
                )
                log_row["validation_worst_completed_share_advantage"] = float(
                    validation["worst_completed_share_advantage"]
                )
                log_row["validation_completed_share_win_rate"] = float(
                    validation["completed_share_win_rate"]
                )
                log_row["validation_market_share"] = float(
                    validation["share"]
                )
                log_row["validation_firm2_market_share"] = float(
                    validation["rival_share"]
                )
                log_row["validation_mean_market_share_advantage"] = float(
                    validation["mean_market_share_advantage"]
                )
                log_row["validation_worst_market_share_advantage"] = float(
                    validation["worst_market_share_advantage"]
                )
                log_row["validation_market_share_win_rate"] = float(
                    validation["market_share_win_rate"]
                )
                log_row["validation_robust_dominance_bonus"] = float(
                    validation["robust_dominance_bonus"]
                )
                log_row["validation_incomplete_win_penalty"] = float(
                    validation["incomplete_win_penalty"]
                )
                log_row["validation_durability_decay_penalty"] = float(
                    validation["durability_decay_penalty"]
                )
                log_row["validation_collapsed_policy_penalty"] = float(
                    validation["collapsed_policy_penalty"]
                )
                log_row["validation_policy_dynamicity"] = float(
                    validation["policy_dynamicity"]
                )
                log_row["validation_aggressive_horizon"] = int(
                    validation["validation_aggressive_horizon"]
                )
                log_row["validation_benchmark_horizon"] = int(
                    validation["validation_benchmark_horizon"]
                )
                for mode, advantage in validation["mode_profit_advantages"].items():
                    log_row[f"validation_profit_advantage_{mode}"] = float(advantage)
                for mode, decay in validation[
                    "mode_profit_advantage_decay"
                ].items():
                    log_row[f"validation_profit_advantage_decay_{mode}"] = float(
                        decay
                    )
                for mode, dynamicity in validation[
                    "mode_tail_dynamicity"
                ].items():
                    log_row[f"validation_tail_dynamicity_{mode}"] = float(
                        dynamicity
                    )
                print(
                    f">>> [validation snapshot] day={d+1} score={validation_score:.3f} "
                    f"reward={validation['reward']:.3f} "
                    f"share={validation['share']:.3f} "
                    f"rivalShare={validation['rival_share']:.3f} "
                    f"revPR={validation['rev_per_request']:.2f} "
                    f"rivalRevPR={validation['rival_rev_per_request']:.2f} "
                    f"profitPR={validation['profit_per_request']:.2f} "
                    f"rivalProfitPR={validation['rival_profit_per_request']:.2f} "
                    f"fullProfitAdv={validation['profit_per_request'] - validation['rival_profit_per_request']:+.2f} "
                    f"tailProfitAdv={validation['mean_profit_advantage']:+.2f} "
                    f"worstTailProfitAdv={validation['worst_profit_advantage']:+.2f} "
                    f"tailWins={validation['profit_win_rate']:.0%} "
                    f"fullShareAdv={validation['share'] - validation['rival_share']:+.1%} "
                    f"tailShareAdv={validation['mean_completed_share_advantage']:+.1%} "
                    f"worstTailShareAdv={validation['worst_completed_share_advantage']:+.1%} "
                    f"tailShareWins={validation['completed_share_win_rate']:.0%} "
                    f"fulfill={validation['mean_fulfillment_rate']:.1%} "
                    f"worstFulfill={validation['worst_fulfillment_rate']:.1%} "
                    f"coverage={validation['mean_completion_coverage']:.1%} "
                    f"worstCoverage={validation['worst_completion_coverage']:.1%} "
                    f"criticTD_EV={validation['heldout_critic_td_explained_variance']:+.3f} "
                    f"criticTD_MAE={validation['heldout_critic_td_mae']:.3f} "
                    f"deploy={validation['deployment_variant']} "
                    f"actions={validation['action_diversity']:.2f} "
                    f"switch={validation['action_switch_rate']:.2f}"
                )
                configured_objective = self._configured_primary_objective()
                if configured_objective == "competitiveness":
                    reported_advantages = validation[
                        "mode_completed_share_advantages"
                    ]
                    advantage_label = "tailCompletedShareAdv"
                    advantage_format = "+.1%"
                elif configured_objective == "profit_maximization":
                    reported_advantages = validation["mode_profit_advantages"]
                    advantage_label = "tailProfitAdv"
                    advantage_format = "+.2f"
                else:
                    reported_advantages = validation[
                        "mode_objective_advantages"
                    ]
                    advantage_label = "tailObjectiveAdv"
                    advantage_format = "+.3f"
                mode_advantages = " | ".join(
                    f"{mode}:{format(advantage, advantage_format)}"
                    for mode, advantage in reported_advantages.items()
                )
                if mode_advantages:
                    print(
                        f"    [validation opponents {advantage_label}] "
                        f"{mode_advantages}"
                    )
                if configured_objective == "profit_maximization":
                    mode_decays = " | ".join(
                        f"{mode}:{decay:.2f}"
                        for mode, decay in validation[
                            "mode_profit_advantage_decay"
                        ].items()
                    )
                    if mode_decays:
                        print(
                            "    [validation early-to-late profitAdv decay] "
                            f"{mode_decays}"
                        )
                selection_key = tuple(
                    float(value)
                    for value in validation.get(
                        "objective_selection_key", ()
                    )
                )
                candidate_checkpoint = {
                    "day": int(d + 1),
                    "training_stage": str(self.current_training_stage.name),
                    "training_opponent_regime": str(
                        self.current_training_stage.opponent_regime
                    ),
                    "score": float(validation_score),
                    "agent_state": {
                        k: v.detach().cpu().clone()
                        for k, v in self.firm1.agent.net.state_dict().items()
                    },
                    "normalizer_state": copy.deepcopy(
                        self.firm1.agent.normalizer_state_dict()
                    ),
                    "optimizer_state": copy.deepcopy(
                        self.firm1.agent.optimizer_state_dict()
                    ),
                    "validation": dict(validation),
                }
                checkpoint_stage_eligible = bool(
                    self.current_training_stage.opponent_regime == "dynamic"
                )
                log_row["validation_checkpoint_stage_eligible"] = bool(
                    checkpoint_stage_eligible
                )
                if checkpoint_stage_eligible:
                    latest_checkpoint = candidate_checkpoint
                    observed_selection_key = (
                        float(validation.get(
                            "objective_feasible_mode_fraction", 0.0
                        )),
                        -float(validation.get(
                            "objective_maximum_normalized_shortfall",
                            float("inf"),
                        )),
                        float(validation_score),
                    )
                    if observed_selection_key >= (
                        best_observed_selection_key
                        if best_observed_selection_key is not None
                        else (-float("inf"),) * 3
                    ):
                        best_observed_selection_key = observed_selection_key
                        best_observed_checkpoint = candidate_checkpoint
                    if (
                        bool(validation.get("checkpoint_eligible"))
                        and self.objective_optimizer.candidate_is_better(
                            selection_key,
                            best_validation_selection_key,
                            require_eligible=False,
                        )
                    ):
                        best_validation_selection_key = selection_key
                        best_validation_score = float(validation_score)
                        best_checkpoint = candidate_checkpoint
                        print(
                            f">>> [validation checkpoint] new_best day={d+1} "
                            f"stage={self.current_training_stage.name} "
                            f"score={validation_score:.3f} "
                            "durability="
                            f"{'PASS' if validation.get('checkpoint_eligible') else 'FAIL'}"
                        )
                else:
                    print(
                        ">>> [validation checkpoint] diagnostic_only "
                        f"day={d+1} stage={self.current_training_stage.name}; "
                        "selection begins in dynamic_opponent."
                    )
            else:
                log_row["validation_score"] = np.nan
                log_row["validation_checkpoint_stage_eligible"] = False
            self.training_logs.append(log_row)
            
            if self.firm1_mode == "RL":
                # Keep the predeclared exploration schedule synchronized in
                # the primary experiment workflow as well as in ``run``.
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
                    train_progress >= 0.75
                    and
                    convergence_count >= 40
                    and convergence_std <= self.reward_convergence_tol
                    and convergence_delta <= self.reward_trend_tol
                    and int(control_ppo_metrics.get("optimizer_steps", 0)) > 0
                )
                training_reward_stability_seen = bool(
                    training_reward_stability_seen or convergence_now
                )
                reward_converged = training_reward_stability_seen
                checkpoint_available = bool(latest_validation)
                log_row["reward_converged"] = bool(reward_converged)
                log_row["reward_convergence_now"] = bool(convergence_now)
                log_row["reward_window_std"] = float(convergence_std)
                log_row["reward_window_delta"] = float(convergence_delta)
                log_row["policy_checkpoint_eligible"] = checkpoint_available
                self.firm1.configure_training_controls(
                    progress=float((d + 1) / max(1, train_timesteps)),
                    # Convergence/checkpoints are observations, never commands
                    # to freeze or narrow a still-learning dynamic policy.
                    reward_converged=False,
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
        self.incomplete_population_rollout_transitions = 0
        if (
            self.firm1_mode == "RL"
            and self.credit_assignment_v25
            and self.firm1.agent.buffer_size > 0
        ):
            self.incomplete_population_rollout_transitions = int(
                self.firm1.agent.discard_rollout()
            )
            if self.training_logs:
                self.training_logs[-1][
                    "incomplete_population_rollout_quarantined"
                ] = int(self.incomplete_population_rollout_transitions)
            print(
                ">>> [PPO population quarantine] retained policy from the "
                "last complete opponent-stratified update; excluded "
                f"{self.incomplete_population_rollout_transitions} terminal "
                "transition(s) from an incomplete final population cycle."
            )
        evaluation_checkpoint, evaluation_checkpoint_status = (
            self._evaluation_checkpoint(
                best_checkpoint,
                latest_checkpoint,
                best_observed_checkpoint,
            )
        )
        if (
            training_enabled
            and self.firm1_mode == "RL"
            and best_checkpoint is None
        ):
            print(
                ">>> [validation checkpoint] no validation snapshot was found; "
                "evaluation will use the final actor weights."
            )
        if self.firm1_mode == "RL" and evaluation_checkpoint is not None:
            self.firm1.agent.net.load_state_dict(
                {
                    k: v.to(self.firm1.agent.device)
                    for k, v in evaluation_checkpoint["agent_state"].items()
                }
            )
            self.firm1.agent.load_normalizer_state_dict(
                evaluation_checkpoint["normalizer_state"]
            )
            if evaluation_checkpoint.get("optimizer_state"):
                self.firm1.agent.load_optimizer_state_dict(
                    evaluation_checkpoint["optimizer_state"]
                )
            self.firm1.agent.deployment_optimizer_enabled = False
            print(
                ">>> Restored comparison policy from train day "
                f"{evaluation_checkpoint['day']} "
                f"(score={evaluation_checkpoint['score']:.3f}, "
                f"stage={evaluation_checkpoint.get('training_stage', 'dynamic_opponent')}, "
                f"status={evaluation_checkpoint_status}, "
                "deployment_variant="
                f"{evaluation_checkpoint['validation'].get('deployment_variant', 'raw_actor')}) "
                "for evaluation."
            )
        if evaluation_checkpoint is not None:
            self.benchmark_evaluation_report = dict(
                evaluation_checkpoint["validation"]
            )
        elif latest_validation:
            self.benchmark_evaluation_report = dict(latest_validation)
        if self.firm1_mode == "RL" and bool(evaluate_benchmark_suite):
            # Always produce a fresh apples-to-apples suite from the weights
            # actually selected for comparison.  Stored checkpoint validation
            # is selection evidence, not a substitute for the requested
            # benchmark product, and certification is intentionally irrelevant
            # to whether this evaluation runs.
            self.benchmark_evaluation_report = self._adaptive_policy_validation(
                customers_per_step=resolved_checkpoint_customers,
                profile_sample_size=resolved_checkpoint_customers,
                # Checkpoint validation is deliberately economical, but the
                # final five-controller comparison is a distinct experimental
                # product and must use the requested evaluation horizon.
                horizon=max(2, int(eval_timesteps)),
                aggressive_horizon=max(2, int(eval_timesteps)),
                policy_mode=eval_policy_mode,
                policy_temperature=eval_policy_temperature,
                policy_seed=(
                    int(eval_policy_seed) + 1_000_003
                    if eval_policy_seed is not None
                    else int(self.seed) + 1_810_004
                ),
                opponent_interval_steps=firm2_interval_steps,
                firm1_interval_steps=firm2_interval_steps,
                steps_per_day=train_steps_per_day,
                aggressive_seed_count=3,
                include_benchmarks=True,
            )
        if getattr(self, "benchmark_evaluation_report", None):
            self.benchmark_evaluation_report.update({
                "comparison_checkpoint_status": evaluation_checkpoint_status,
                "comparison_checkpoint_day": (
                    None
                    if evaluation_checkpoint is None
                    else int(evaluation_checkpoint["day"])
                ),
                "comparison_checkpoint_stage": (
                    None
                    if evaluation_checkpoint is None
                    else str(evaluation_checkpoint.get("training_stage", ""))
                ),
                "deployment_variant": "raw_actor",
                "deployment_policy_mode": str(eval_policy_mode),
                "comparison_runs_independently_of_certification": True,
                "training_dynamic_opponent_modes": list(
                    self.training_dynamic_opponent_modes
                ),
                "evaluation_profile_partition_held_out": bool(
                    self.evaluation_generalization_fraction > 0.0
                ),
            })
        # Evaluation starts from a fresh independently randomized opening under
        # the same ±15% protocol used by dynamic-stage training and validation.
        self.firm2 = copy.deepcopy(benchmark_opponent)
        self._reset_adaptive_episode(
            tariff_reset_fraction=0.15,
            symmetric_tariffs=False,
        )

        pending_model_metadata: Optional[Dict[str, Any]] = None
        if (
            training_enabled
            and self.firm1_mode == "RL"
            and (trained_model_out or model_registry_dir)
        ):
            pending_model_metadata = {
                "best_validation_day": (
                    None if best_checkpoint is None else int(best_checkpoint["day"])
                ),
                "best_validation_score": (
                    None if best_checkpoint is None else float(best_checkpoint["score"])
                ),
                "best_validation_stage": (
                    None
                    if best_checkpoint is None
                    else str(best_checkpoint.get("training_stage", ""))
                ),
                "latest_validation_day": (
                    None
                    if latest_checkpoint is None
                    else int(latest_checkpoint["day"])
                ),
                "latest_validation_score": (
                    None
                    if latest_checkpoint is None
                    else float(latest_checkpoint["score"])
                ),
                "latest_validation_stage": (
                    None
                    if latest_checkpoint is None
                    else str(latest_checkpoint.get("training_stage", ""))
                ),
                "comparison_checkpoint_status": str(
                    evaluation_checkpoint_status
                ),
                "comparison_checkpoint_day": (
                    None
                    if evaluation_checkpoint is None
                    else int(evaluation_checkpoint["day"])
                ),
                "comparison_checkpoint_stage": (
                    None
                    if evaluation_checkpoint is None
                    else str(evaluation_checkpoint.get("training_stage", ""))
                ),
                "training_opponent": str(opponent_mode),
                "training_days": int(train_timesteps),
                "training_seed": int(self.seed),
                "checkpoint_validation_horizon": int(
                    resolved_checkpoint_horizon
                ),
                "checkpoint_validation_customers_per_step": int(
                    resolved_checkpoint_customers
                ),
                "checkpoint_validation_openings": int(
                    resolved_checkpoint_openings
                ),
                "checkpoint_policy_mode": str(eval_policy_mode),
                "checkpoint_deployment_variant": (
                    None
                    if evaluation_checkpoint is None
                    else str(
                        evaluation_checkpoint["validation"].get(
                            "deployment_variant", "raw_actor"
                        )
                    )
                ),
                "checkpoint_policy_seed": int(
                    self.seed + 810_001
                    if eval_policy_seed is None
                    else eval_policy_seed
                ),
                "generalization_profile_source": "same_market_threshold_holdout",
                "training_generalization_fraction": float(
                    self.training_generalization_fraction
                ),
                "evaluation_generalization_fraction": float(
                    self.evaluation_generalization_fraction
                ),
                "generalization_cities": sorted(
                    str(city) for city in self.generalization_profiles_by_city
                ),
                "checkpoint_validation": (
                    None
                    if best_checkpoint is None
                    else dict(best_checkpoint["validation"])
                ),
                "latest_checkpoint_validation": (
                    None
                    if latest_checkpoint is None
                    else dict(latest_checkpoint["validation"])
                ),
            }

        
        print(
            f">>> Evaluating RL agent against same-level {opponent_mode} opponent "
            f"with a held-out {self.market_name} profile partition..."
        )
        if self.firm1_mode == "RL":
            self.firm1.reset_state_history()
        for observer in self.platform_observers.values():
            observer.reset()
        self.action_stability.reset()
        resolved_eval_policy_seed = int(self.seed + 700_001 if eval_policy_seed is None else eval_policy_seed)
        if str(eval_policy_mode) in {"sample_raw", "sample_low_temp"}:
            torch.manual_seed(resolved_eval_policy_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(resolved_eval_policy_seed)
            print(
                f">>> Evaluation policy mode={eval_policy_mode} seed={resolved_eval_policy_seed} "
                f"temperature={float(eval_policy_temperature):.3f}"
            )
        eval_rewards: List[float] = []
        # Reset reward-trend baselines so evaluation reward reflects evaluation dynamics,
        # not trailing deltas from the end of training.
        eval_last_share = float(self.last_share)
        eval_last_revpr = float(self.last_revpr)
        eval_last_profitpr = float(self.last_profitpr)
        eval_last_rival_market_share = float(self.last_firm2_share)
        eval_last_rival_profitpr = float(self.last_firm2_profitpr)
        eval_last_fulfillment = float(self.last_fulfillment)
        eval_last_gap = float(self.last_gap)
        # Match the reaction horizon used for late-stage PPO credit: Firm 1
        # chooses once, observes the rival's response cycle, then chooses again.
        # Acting every market tick at deployment would recreate the train/eval
        # cadence mismatch and obscure which action earned the outcome.
        eval_action_interval_steps = int(
            max(
                base_action_interval_steps,
                firm2_interval_steps if self.firm2_mode != "static" else 1,
            )
        )
        eval_clock = OperationalClock()
        eval_previous_firm2_metrics: Optional[FirmMetrics] = None
        eval_day_ctx = None
        for t in range(eval_timesteps):
            if eval_day_ctx is None or t % train_steps_per_day == 0:
                eval_day_ctx = self.market.sample_day_context()
            day_ctx = eval_day_ctx
            hour = self.market.sample_timestep_hour().hour
            # Evaluation uses rows held out from training but drawn from the
            # same market and threshold source.
            sampled_profiles = self._sample_profiles_from_pool(
                eval_profile_sample_size,
                generalization_fraction=self.evaluation_generalization_fraction,
            )
            if profiles_out and not profile_limit_reached:
                remaining = int(max(0, profiles_log_limit - len(sampled_profile_rows)))
                if remaining > 0:
                    sampled_profile_rows.extend({"Phase": "eval", "Timestep": int(t), **p} for p in sampled_profiles[:remaining])
                profile_limit_reached = len(sampled_profile_rows) >= int(max(0, profiles_log_limit))

            # Evaluation timesteps are operational decision periods.  On each
            # paired epoch Firm 2 executes first from lagged outcomes, posts
            # its new public quote, and only then may Firm 1 react.
            firm2_eval_interval_steps = max(1, firm2_interval_steps)
            firm2_eval_due = bool(
                not isinstance(self.firm2, FirmStaticPricer)
                and eval_clock.due(firm2_eval_interval_steps)
            )
            firm2_eval_action_target, firm2_eval_intervened = (
                self._firm2_leader_step(
                    previous_metrics=eval_previous_firm2_metrics,
                    day_of_week=day_ctx.day_of_week,
                    hour=hour,
                    weather=day_ctx.weather,
                    enabled=firm2_eval_due,
                )
            )

            base = self.market.curr_market
            action = -1
            eval_policy_diag: Dict[str, float] = {}
            coeff_pre_action = self._coeff_snapshot()
            coeff_post_action = dict(coeff_pre_action)
            coeff_post_projection = dict(coeff_pre_action)
            if self.firm1_mode == "RL":
                # Commit every public market frame, including periods where the
                # current tariff is held.  The opponent GRU therefore sees the
                # full Firm 2 reaction trajectory rather than a sparse sample
                # taken only when Firm 1 is about to move.
                raw_state_vec, action_features, _ = self._publish_pricing_observation(
                    day_of_week=day_ctx.day_of_week,
                    hour=hour,
                    weather=day_ctx.weather,
                )
                s_vec = self.firm1.stack_state(raw_state_vec)
                critic_state = (
                    self._build_privileged_critic_state(raw_state_vec)
                    if self.privileged_critic
                    else None
                )
                if (t % eval_action_interval_steps) == 0:
                    action_mask = self._objective_action_mask(
                        self.firm1.feasible_action_mask(self.market)
                    )
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
                        critic_state=critic_state,
                    )
                    coeff_pre_action = self._coeff_snapshot()
                    self.firm1.apply_action(action, self.market)
                    coeff_post_action = self._coeff_snapshot()
                    coeff_post_projection = dict(coeff_post_action)
                    action = int(action)
                else:
                    action = -1
                    self.firm1.agent.observe_recurrent_state(
                        s_vec, critic_state=critic_state
                    )
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

            _, m1, m2, mean_gap, _, _ = self.simulate_batch(
                day_of_week=day_ctx.day_of_week,
                weather=day_ctx.weather,
                hour=hour,
                customers_per_step=eval_customers_per_step,
                sampled_profiles=sampled_profiles,
                collect_rows=False,
            )
            self._ingest_platform_observations(m1, m2)
            eval_previous_firm2_metrics = m2
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
                prev_rival_profit_per_request=eval_last_rival_profitpr,
                prev_rival_market_share=eval_last_rival_market_share,
                prev_fulfillment_rate=eval_last_fulfillment,
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
                price_gap_abs_error=float(self.last_price_gap_metrics["mae"]),
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

            market_share = float(reward_diag["reward_market_share"])
            rival_market_share = float(
                reward_diag["reward_rival_market_share"]
            )
            eval_last_share = market_share
            eval_last_revpr = float(revpr)
            eval_last_profitpr = float(m1.profit_per_request)
            eval_last_rival_market_share = rival_market_share
            eval_last_rival_profitpr = float(m2.profit_per_request)
            eval_last_fulfillment = float(m1.fulfillment_rate)
            eval_last_gap = float(mean_gap)
            f2_supply_diag, _ = self._driver_supply_context("Firm2", "Firm1")
            eval_rewards.append(float(eval_reward))
            moving_window = eval_rewards[-min(len(eval_rewards), 20):]
            moving_avg20 = float(np.mean(moving_window)) if moving_window else 0.0
            moving_std20 = float(np.std(moving_window)) if moving_window else 0.0
            log_row = {
                "day": t,
                "profile_population": "same_market_threshold_holdout",
                "profile_generalization_fraction": float(
                    self.evaluation_generalization_fraction
                ),
                "policy_objective": str(
                    self.long_term_profit_reward_config.objective_mode
                ),
                "reward_mechanism": str(
                    self.long_term_profit_reward_model.mechanism_name
                ),
                "reward": float(eval_reward),
                "reward_moving_avg20": moving_avg20,
                "reward_moving_std20": moving_std20,
                "reward_raw": float(reward_diag["reward_raw"]),
                "reward_base": float(reward_diag["reward_base"]),
                "reward_dominance_term": float(reward_diag["reward_dominance_term"]),
                "reward_competitive_term": float(reward_diag["reward_competitive_term"]),
                "reward_active_objective_component": float(
                    reward_diag["reward_active_objective_component"]
                ),
                "reward_active_objective_term_count": float(
                    reward_diag["reward_active_objective_term_count"]
                ),
                "reward_trend_term": float(reward_diag["reward_trend_term"]),
                "reward_pricing_discipline": float(reward_diag["reward_pricing_discipline"]),
                "reward_share_delta": float(reward_diag["reward_share_delta"]),
                "reward_rev_delta": float(reward_diag["reward_rev_delta"]),
                "reward_efficiency_term": float(reward_diag["reward_efficiency_term"]),
                "reward_own_profit_utility": float(reward_diag["reward_own_profit_utility"]),
                "reward_profit_advantage_utility": float(reward_diag["reward_profit_advantage_utility"]),
                "reward_own_profit_component": float(reward_diag["reward_own_profit_component"]),
                "reward_profit_advantage_component": float(reward_diag["reward_profit_advantage_component"]),
                "reward_market_share_utility": float(reward_diag["reward_market_share_utility"]),
                "reward_market_share_level_utility": float(reward_diag["reward_market_share_level_utility"]),
                "reward_market_share_dominance_utility": float(reward_diag["reward_market_share_dominance_utility"]),
                "reward_market_share_competitiveness_component": float(reward_diag["reward_market_share_competitiveness_component"]),
                "reward_market_share_advantage": float(reward_diag["reward_market_share_advantage"]),
                "reward_market_share_target_gap": float(reward_diag["reward_market_share_target_gap"]),
                "reward_market_share_target_level": float(
                    reward_diag["reward_market_share_target_level"]
                ),
                "reward_market_share_target_progress": float(
                    reward_diag["reward_market_share_target_progress"]
                ),
                "reward_market_share_target_achieved": float(
                    reward_diag["reward_market_share_target_achieved"]
                ),
                "reward_market_share_gap_error": float(reward_diag["reward_market_share_gap_error"]),
                "reward_market_share_target_shortfall": float(
                    reward_diag["reward_market_share_target_shortfall"]
                ),
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
                "reward_conditional_choice_share": float(
                    reward_diag.get("reward_conditional_choice_share", 0.0)
                ),
                "reward_rival_conditional_choice_share": float(
                    reward_diag.get("reward_rival_conditional_choice_share", 0.0)
                ),
                "reward_conditional_choice_share_sum": float(
                    reward_diag.get("reward_conditional_choice_share_sum", 0.0)
                ),
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
                "rl_market_share": market_share,
                "heuristic_market_share": rival_market_share,
                "market_share_sum": float(
                    market_share + rival_market_share
                ),
                "outside_option_share": float(
                    reward_diag["reward_outside_option_share"]
                ),
                "completion_coverage": float(
                    reward_diag["reward_completion_coverage"]
                ),
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
                "price_gap_f2_minus_f1": float(mean_gap),
                "price_gap_abs_error": float(
                    self.last_price_gap_metrics["mae"]
                ),
                "price_gap_rmse": float(
                    self.last_price_gap_metrics["rmse"]
                ),
                "price_gap_p90_abs_error": float(
                    self.last_price_gap_metrics["p90_abs_error"]
                ),
                "price_gap_within_tolerance_rate": float(
                    self.last_price_gap_metrics["within_tolerance_rate"]
                ),
                "price_gap_target": float(self.reward_target_price_gap),
                "price_gap_tolerance": float(
                    self.constraint_config.overall_tolerance
                ),
                "gap_violation_025": float(
                    self.last_price_gap_metrics["mae"] > 0.25
                ),
                "gap_violation_050": float(
                    self.last_price_gap_metrics["mae"] > 0.50
                ),
                "constraint_curriculum_scale": float(self.constraint_curriculum_scale),
                "competition_protocol": "firm2_leader_then_firm1_follower",
                "firm2_leader_decision": bool(firm2_eval_due),
                "firm2_leader_intervened": bool(firm2_eval_intervened),
                "firm2_leader_action_steps": json.dumps(
                    firm2_eval_action_target.astype(float).tolist()
                ),
                "eval_policy_mode": str(eval_policy_mode),
                "eval_policy_seed": int(resolved_eval_policy_seed),
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
                "objective_optimizer_ready": bool(
                    eval_policy_diag.get("optimizer_ready", 0.0) >= 0.5
                ),
                "objective_optimizer_top_action": int(
                    eval_policy_diag.get("optimizer_top_action", -1)
                ),
                "objective_optimizer_top_prob": float(
                    eval_policy_diag.get("optimizer_top_prob", 0.0)
                ),
                "objective_optimizer_action_margin": float(
                    eval_policy_diag.get("optimizer_action_margin", 0.0)
                ),
                "objective_optimizer_action_changed": float(
                    eval_policy_diag.get("optimizer_action_changed", 0.0)
                ),
                "objective_optimizer_q_spread": float(
                    eval_policy_diag.get("optimizer_q_spread", 0.0)
                ),
                "policy_selected_is_top": float(
                    action
                    == int(
                        eval_policy_diag.get(
                            "optimizer_top_action",
                            eval_policy_diag.get("policy_top_action", -999),
                        )
                    )
                ),
                "guardrail_applied": bool(guardrail_diag.get("applied", False)),
                "guardrail_recommended": bool(guardrail_diag.get("recommended", False)),
                "guardrail_reasons": json.dumps(list(guardrail_diag.get("reasons", []))),
                "guardrail_deltas": json.dumps(guardrail_diag.get("deltas", {}), sort_keys=True),
                "coeff_delta_rl_action": json.dumps(self._coeff_delta(coeff_post_action, coeff_pre_action), sort_keys=True),
                "coeff_delta_projection": json.dumps(self._coeff_delta(coeff_post_projection, coeff_post_action), sort_keys=True),
                "action_zero_effect": bool(getattr(self.firm1, "last_action_was_zero_effect", False)) if self.firm1_mode == "RL" else False,
                "action_saturated": bool(getattr(self.firm1, "last_action_was_saturated", False)) if self.firm1_mode == "RL" else False,
                "action": int(action),
                "action_mask_regime": str(
                    self.last_action_mask_regime
                ),
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
            self.last_share = market_share
            self.last_revpr = float(m1.rev_per_request)
            self.last_gap = float(mean_gap)
            self.last_profitpr = float(m1.profit_per_request)
            self.last_fulfillment = float(m1.fulfillment_rate)
            self.last_completion_coverage = float(np.clip(
                m1.completed_share + m2.completed_share, 0.0, 1.0
            ))
            self.last_acceptance = float(m1.driver_acceptance_rate)
            self.last_wait = float(m1.avg_wait_minutes)
            self.last_driver_paypr = float(m1.driver_pay / max(1, m1.total))
            self.last_firm2_share = rival_market_share
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
                    f"marketShare={market_share:.3f} "
                    f"rivalMarketShare={rival_market_share:.3f} "
                    f"completionCoverage={float(reward_diag['reward_completion_coverage']):.3f} "
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
                f"marketShare={float(np.mean([row['avg_market_share'] for row in training_tail])):.3f} "
                f"rivalMarketShare={float(np.mean([row['avg_firm2_market_share'] for row in training_tail])):.3f} "
                f"completionCoverage={float(np.mean([row['avg_completion_coverage'] for row in training_tail])):.3f} "
                f"revPR={float(np.mean([row['avg_rev_per_request'] for row in training_tail])):.2f} "
                f"rivalRevPR={float(np.mean([row['avg_firm2_rev_per_request'] for row in training_tail])):.2f} "
                f"profitPR={float(np.mean([row['avg_profit_per_request'] for row in training_tail])):.2f} "
                f"rivalProfitPR={float(np.mean([row['avg_firm2_profit_per_request'] for row in training_tail])):.2f} "
                f"profitAdv={float(np.mean([row['avg_profit_advantage_per_request'] for row in training_tail])):+.2f}"
            )
        if self.evaluation_logs:
            print(
                f">>> Evaluation competitive summary ({len(self.evaluation_logs)} days): "
                f"marketShare={float(np.mean([row['rl_market_share'] for row in self.evaluation_logs])):.3f} "
                f"rivalMarketShare={float(np.mean([row['heuristic_market_share'] for row in self.evaluation_logs])):.3f} "
                f"completionCoverage={float(np.mean([row['completion_coverage'] for row in self.evaluation_logs])):.3f} "
                f"revPR={float(np.mean([row['rl_revenue'] for row in self.evaluation_logs])):.2f} "
                f"rivalRevPR={float(np.mean([row['heuristic_revenue'] for row in self.evaluation_logs])):.2f} "
                f"profitPR={float(np.mean([row['rl_profit'] for row in self.evaluation_logs])):.2f} "
                f"rivalProfitPR={float(np.mean([row['heuristic_profit'] for row in self.evaluation_logs])):.2f} "
                f"profitAdv={float(np.mean([row['rl_profit'] - row['heuristic_profit'] for row in self.evaluation_logs])):+.2f}"
            )

        self.competitive_backtest_report = self._competitive_durability_backtest()
        report = self.competitive_backtest_report
        status = "PASS" if report.get("passed") else "FAIL"
        print(
            f">>> Competitive durability backtest: {status}; "
            f"lateMarketShareAdv={float(report.get('late_market_share_advantage', np.nan)):+.3f}, "
            f"marketShareMA20Wins={float(report.get('late_market_share_rolling_dominance_rate', np.nan)):.1%}, "
            f"lateProfitAdv={float(report.get('late_profit_advantage_per_request', np.nan)):+.3f}, "
            f"profitMA20Wins={float(report.get('late_profit_rolling_dominance_rate', np.nan)):.1%}, "
            f"validationTrend={float(report.get('held_out_validation_score_fitted_change', np.nan)):+.3f}, "
            f"evalRewardRetention={float(report.get('late_evaluation_reward_retention', np.nan)):.1%}, "
            f"lateFareMove={float(report.get('late_fare_equivalent_change_rate', np.nan)):.1%}, "
            f"lateActions={int(report.get('late_action_diversity', 0))}"
        )
        if report.get("failures"):
            print(
                "    [backtest failures] "
                + " | ".join(str(item) for item in report["failures"])
            )
        if pending_model_metadata is not None:
            pending_model_metadata["competitive_backtest"] = dict(report)
            backtest_passed = bool(report.get("passed", False))
            pending_model_metadata["competitive_backtest_passed"] = (
                backtest_passed
            )
            pending_model_metadata["certification_status"] = "not_applicable"
            self.training_certification_report = {
                "competitive_backtest_passed": backtest_passed,
                "status": "not_applicable",
            }
            saved_policy_path = self.save_trained_model(
                trained_model_out,
                metadata=pending_model_metadata,
                registry_dir=model_registry_dir,
                model_id=trained_model_id,
            )
            # Verify that the archived artifact is independently loadable.
            self.load_trained_model(saved_policy_path, load_optimizer=False)

        print("Experiment Complete.")
        return self.training_logs, self.evaluation_logs
    
    def _predict_dataset_ride_with_policy(
        self,
        *,
        distance: float,
        duration: float,
        ctx: RideContext,
        policy_mode: str,
        policy_temperature: float,
    ) -> Dict[str, Any]:
        """Make one independent, row-conditioned counterfactual policy quote.

        The PPO policy controls an aggregate tariff rather than directly
        regressing a fare.  For offline validation, each ride therefore starts
        from the city anchor tariff, supplies its observed time and demand mix
        as the current platform context, takes one learned tariff action, and
        prices the ride with the resulting coefficients.  Rows never mutate the
        next row's tariff or state history.
        """
        if self.firm1_mode != "RL":
            raise ValueError("dataset policy validation requires firm1_mode='RL'")

        observer = self.platform_observers["Firm1"]
        observer.reset()
        long_trip = float(distance >= 10.0)
        observer.demand_mix.update({
            "distance_mean": float(distance),
            "distance_std": 0.0,
            "distance_q25": float(distance),
            "distance_q75": float(distance),
            "duration_mean": float(duration),
            "duration_std": 0.0,
            "airport_rate": float(bool(ctx.airport)),
            "long_trip_share": long_trip,
        })
        self.last_crowd_response_stats = {
            "distance_mean": float(distance),
            "duration_mean": float(duration),
            "airport_rate": float(bool(ctx.airport)),
            "long_trip_share": long_trip,
            "peak_context": float((7 <= ctx.hour < 10) or (16 <= ctx.hour < 19)),
        }
        self.firm1.overrides = CoefficientOverrides()
        self.firm1.reset_state_history()
        self.action_stability.reset()

        raw_state = self._build_rl_state(
            day_of_week=ctx.day_of_week,
            hour=ctx.hour,
            weather=ctx.weather,
        )
        action_features = self._action_feature_matrix_for_pricer(
            self.firm1, self.last_crowd_response_stats
        )
        state = self.firm1.stack_state(raw_state, commit=False)
        critic_state = (
            self._build_privileged_critic_state(raw_state)
            if self.privileged_critic
            else None
        )
        action_mask = self._objective_action_mask(
            self.firm1.feasible_action_mask(self.market)
        )
        policy_diag = self.firm1.agent.policy_diagnostics(
            state,
            action_features=action_features,
            action_mask=action_mask,
            temperature=1.0,
        )
        baseline_price = float(self.market.quote_price(
            distance_miles=float(max(0.0, distance)),
            duration_minutes=float(max(0.0, duration)),
            ctx=ctx,
            overrides=self.firm1.overrides,
        ))
        action, *_ = self.firm1.agent.act(
            state,
            deterministic=True,
            action_features=action_features,
            action_mask=action_mask,
            policy_mode=policy_mode,
            policy_temperature=policy_temperature,
            critic_state=critic_state,
        )
        self.firm1.apply_action(int(action), self.market)
        predicted_price = float(self.market.quote_price(
            distance_miles=float(max(0.0, distance)),
            duration_minutes=float(max(0.0, duration)),
            ctx=ctx,
            overrides=self.firm1.overrides,
        ))
        coefficients = self._coeff_snapshot()
        return {
            "predicted_price": predicted_price,
            "anchor_price": baseline_price,
            "action": int(action),
            "action_label": self.firm1.action_label(int(action)),
            "action_magnitude": float(self.firm1.agent.last_continuous_magnitude),
            "policy_top_action": int(policy_diag.get("policy_top_action", -1)),
            "policy_top_prob": float(policy_diag.get("policy_top_prob", 0.0)),
            "policy_second_prob": float(policy_diag.get("policy_second_prob", 0.0)),
            "policy_action_margin": float(policy_diag.get("policy_action_margin", 0.0)),
            "policy_entropy": float(policy_diag.get("policy_entropy", 0.0)),
            "objective_optimizer_ready": bool(
                policy_diag.get("optimizer_ready", 0.0) >= 0.5
            ),
            "objective_optimizer_top_action": int(
                policy_diag.get("optimizer_top_action", -1)
            ),
            "objective_optimizer_action_margin": float(
                policy_diag.get("optimizer_action_margin", 0.0)
            ),
            "coefficients": coefficients,
        }

    def compare_trained_rl_to_dataset(
       self,
       dataset_root: str,
       dataset_glob: str = "auto",
       out_csv: Optional[str] = None,
       out_plot_prefix: Optional[str] = None,
       max_rows: int = 5000,
       preview_rows: int = 5,
       policy_mode: str = "argmax_raw",
       policy_temperature: float = 0.25,
       policy_seed: int = 99173,
       dataset_sample_seed: int = 104729,
       duration_mode: str = "actual_if_available",
       dataset_city: str = "",
       weather_root: str = "",
   ) -> Dict[str, Any]:
       market_spec = resolve_city_spec(self.market_name)
       data_spec = resolve_city_spec(dataset_city or self.market_name)
       if market_spec.key != data_spec.key:
           raise ValueError(
               f"Dataset city {data_spec.market_name!r} does not match active "
               f"policy market {market_spec.market_name!r}. Cross-city transfer "
               "must be run as an explicitly configured experiment, not by "
               "silently relabelling source rows."
           )
       if duration_mode not in {"actual_if_available", "predicted_only"}:
           raise ValueError(
               "duration_mode must be 'actual_if_available' or 'predicted_only'"
           )
       policy_artifact_path = (
           getattr(self, "loaded_trained_model_path", None)
           or getattr(self, "last_saved_model_path", None)
       )
       if not policy_artifact_path:
           raise ValueError(
               "Dataset validation requires a saved trained policy artifact. "
               "Load one with --trained_model_in or train with model registry/output enabled."
           )
       policy_artifact_path = os.path.abspath(
           os.path.expanduser(str(policy_artifact_path))
       )
       policy_model_id = (
           getattr(self, "loaded_trained_model_id", None)
           or getattr(self, "last_saved_model_id", None)
       )
       policy_training_seed = getattr(
           self, "loaded_trained_model_training_seed", None
       )
       policy_training_seed_source = getattr(
           self, "loaded_trained_model_training_seed_source", "unavailable"
       )
       policy_artifact_sha256 = getattr(
           self, "loaded_trained_model_sha256", None
       )
       if policy_artifact_sha256 is None and os.path.isfile(policy_artifact_path):
           policy_artifact_sha256 = _sha256_file(policy_artifact_path)
       resolved_dataset_glob = (
           data_spec.ride_glob
           if str(dataset_glob or "").strip().lower() in {"", "auto"}
           else str(dataset_glob)
       )
       files = _discover_dataset_files(
           dataset_root=dataset_root,
           dataset_glob=resolved_dataset_glob,
       )

       if not files:
           raise FileNotFoundError(
               f"No dataset files found under dataset_root={dataset_root!r} with glob={dataset_glob!r}"
           )

       rows_out: List[Dict[str, Any]] = []
       abs_err: List[float] = []
       sq_err: List[float] = []
       signed_err: List[float] = []
       abs_pct: List[float] = []
       anchor_abs_err: List[float] = []
       dur_abs_err: List[float] = []
       dur_sq_err: List[float] = []
       dur_abs_pct: List[float] = []

       processed = 0
       kept = 0

       skipped_missing_price = 0
       skipped_missing_distance = 0
       skipped_invalid_price = 0
       skipped_invalid_distance = 0
       normalization_rejections: Counter = Counter()
       weather_joined = 0
       files_with_rows = 0
       preview_printed = False
       weather_index = (
           NOAAWeatherIndex.from_directory(data_spec.key, weather_root)
           if str(weather_root or "").strip()
           else None
       )

       saved_policy_state = {
           "firm1_overrides": copy.deepcopy(self.firm1.overrides),
           "firm1_execution": self.firm1.snapshot_execution_state(),
           "observer": self.platform_observers["Firm1"].snapshot(),
           "stability": self.action_stability.snapshot(),
           "crowd": dict(getattr(self, "last_crowd_response_stats", {}) or {}),
       }
       torch.manual_seed(int(policy_seed))
       if torch.cuda.is_available():
           torch.cuda.manual_seed_all(int(policy_seed))

       try:
        for file_index, fpath in enumerate(files):
           file_kept = 0
           try:
               file_sample_seed = int(dataset_sample_seed) + 1000003 * int(file_index)
               for fieldnames, reader in _iter_tabular_rows(
                   fpath,
                   sample_seed=file_sample_seed,
               ):
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

                       raw_source_record = dict(raw)
                       raw_record_json = _canonical_json(raw_source_record)
                       raw_record_sha256 = hashlib.sha256(
                           raw_record_json.encode("utf-8")
                       ).hexdigest()
                       try:
                           canonical = normalize_ride_record(
                               data_spec.key,
                               raw_source_record,
                               weather_index=weather_index,
                           )
                       except DataValidationError as exc:
                           normalization_rejections[exc.code] += 1
                           if exc.code == "missing_price":
                               skipped_missing_price += 1
                           elif exc.code == "missing_distance":
                               skipped_missing_distance += 1
                           elif exc.code == "invalid_price":
                               skipped_invalid_price += 1
                           elif exc.code == "invalid_distance":
                               skipped_invalid_distance += 1
                           continue

                       actual_paid = float(canonical.actual_price)
                       actual_price_key = canonical.actual_price_source
                       distance = float(canonical.distance_miles)
                       distance_key = canonical.distance_source
                       actual_duration = canonical.duration_minutes
                       duration_key = canonical.duration_source
                       hour = int(canonical.hour)
                       day_of_week = int(canonical.day_of_week)
                       weather = str(canonical.weather)
                       airport = bool(canonical.airport)
                       service = str(canonical.service)
                       weather_joined += int(
                           canonical.weather_source == "noaa_global_hourly"
                       )

                       predicted_duration = float(self.estimate_duration(float(distance), hour))
                       duration = (
                           float(predicted_duration)
                           if duration_mode == "predicted_only"
                           else (
                               float(actual_duration)
                               if actual_duration is not None
                               else float(predicted_duration)
                           )
                       )

                       if actual_duration is not None:
                           d_err = float(predicted_duration - float(actual_duration))
                           dae = float(abs(d_err))
                           dur_abs_err.append(dae)
                           dur_sq_err.append(float(d_err * d_err))
                           if float(actual_duration) > 1e-6:
                               dur_abs_pct.append(float(dae / float(actual_duration)))

                       ctx = RideContext(
                           day_of_week=int(np.clip(day_of_week, 0, 6)),
                           weather=weather,
                           hour=int(np.clip(hour, 0, 23)),
                           airport=bool(airport),
                           service=service,
                       )
                       policy_input = {
                           "market": market_spec.market_name,
                           "distance_miles": float(max(0.0, distance)),
                           "duration_minutes": float(max(0.0, duration)),
                           "duration_mode": str(duration_mode),
                           "hour": int(ctx.hour),
                           "day_of_week": int(ctx.day_of_week),
                           "weather": str(ctx.weather),
                           "airport": bool(ctx.airport),
                           "service": str(ctx.service),
                       }
                       policy_prediction = self._predict_dataset_ride_with_policy(
                           distance=float(max(0.0, distance)),
                           duration=float(max(0.0, duration)),
                           ctx=ctx,
                           policy_mode=policy_mode,
                           policy_temperature=policy_temperature,
                       )
                       rl_price = float(policy_prediction["predicted_price"])
                       anchor_abs_err.append(abs(float(policy_prediction["anchor_price"]) - actual_paid))

                       err = float(rl_price - actual_paid)
                       ae = float(abs(err))
                       abs_err.append(ae)
                       sq_err.append(float(err * err))
                       signed_err.append(err)
                       if actual_paid > 1e-6:
                           abs_pct.append(float(ae / actual_paid))

                       rows_out.append({
                           "source_file": os.path.basename(fpath),
                           "source_path": os.path.abspath(fpath),
                           "source_record_sha256": raw_record_sha256,
                           "source_stream_position": int(processed),
                           "raw_record_json": raw_record_json,
                           "dataset_city": data_spec.market_name,
                           "ride_semantics": canonical.ride_semantics,
                           "price_semantics": canonical.price_semantics,
                           "actual_paid_source_column": str(actual_price_key),
                           "actual_paid": float(actual_paid),
                           "actual_paid_excluded_from_policy_input": True,
                           "rl_predicted_price": float(rl_price),
                           "anchor_tariff_price": float(policy_prediction["anchor_price"]),
                           "price_error": err,
                           "abs_error": ae,
                           "distance_miles": float(distance),
                           "distance_source_column": str(distance_key),
                           "actual_duration_minutes": float(actual_duration) if actual_duration is not None else None,
                           "actual_duration_source_column": (
                               str(duration_key) if duration_key is not None else ""
                           ),
                           "predicted_duration_minutes": float(predicted_duration),
                           "duration_minutes_used_for_price": float(duration),
                           "duration_mode": str(duration_mode),
                           "dataset_sample_seed": int(dataset_sample_seed),
                           "hour": int(ctx.hour),
                           "day_of_week": int(ctx.day_of_week),
                           "weather": str(ctx.weather),
                           "weather_source": canonical.weather_source,
                           "temperature_c": canonical.temperature_c,
                           "precipitation_mm": canonical.precipitation_mm,
                           "wind_speed_mps": canonical.wind_speed_mps,
                           "visibility_m": canonical.visibility_m,
                           "airport": bool(ctx.airport),
                           "service": str(ctx.service),
                           "shared_ride": bool(canonical.shared_ride),
                           "surge_multiplier": canonical.surge_multiplier,
                           "raw_business": canonical.provider,
                           "raw_pickup_location": canonical.pickup_location,
                           "raw_dropoff_location": canonical.dropoff_location,
                           "raw_date": (
                               canonical.pickup_datetime.isoformat()
                               if canonical.pickup_datetime is not None
                               else ""
                           ),
                           "raw_time_of_day": "",
                           "policy_input_json": _canonical_json(policy_input),
                           "trained_policy_model_id": str(policy_model_id or ""),
                           "trained_policy_artifact_path": policy_artifact_path,
                           "trained_policy_artifact_sha256": str(
                               policy_artifact_sha256 or ""
                           ),
                           "trained_policy_training_seed": (
                               policy_training_seed
                               if policy_training_seed is not None
                               else ""
                           ),
                           "trained_policy_training_seed_source": str(
                               policy_training_seed_source
                           ),
                           "policy_action_sampling_seed": int(policy_seed),
                           "policy_mode": str(policy_mode),
                           "policy_action": int(policy_prediction["action"]),
                           "policy_action_label": str(policy_prediction["action_label"]),
                           "policy_action_magnitude": float(policy_prediction["action_magnitude"]),
                           "policy_top_action": int(policy_prediction["policy_top_action"]),
                           "policy_top_prob": float(policy_prediction["policy_top_prob"]),
                           "policy_second_prob": float(policy_prediction["policy_second_prob"]),
                           "policy_action_margin": float(policy_prediction["policy_action_margin"]),
                           "policy_entropy": float(policy_prediction["policy_entropy"]),
                           **{
                               f"predicted_{key}": float(value)
                               for key, value in policy_prediction["coefficients"].items()
                           },
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
       finally:
           self.firm1.overrides = saved_policy_state["firm1_overrides"]
           self.firm1.restore_execution_state(saved_policy_state["firm1_execution"])
           self.platform_observers["Firm1"].restore(saved_policy_state["observer"])
           self.action_stability.restore(saved_policy_state["stability"])
           self.last_crowd_response_stats = saved_policy_state["crowd"]

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
           "dataset_glob": resolved_dataset_glob,
           "dataset_city": data_spec.market_name,
           "ride_semantics": data_spec.ride_semantics,
           "price_semantics": data_spec.price_semantics,
           "files_scanned": len(files),
           "rows_processed": int(processed),
           "rows_compared": int(kept),
           "rows_skipped_missing_price": int(skipped_missing_price),
           "rows_skipped_missing_distance": int(skipped_missing_distance),
           "rows_skipped_invalid_price": int(skipped_invalid_price),
           "rows_skipped_invalid_distance": int(skipped_invalid_distance),
           "normalization_rejections": dict(normalization_rejections),
           "weather_join_rate": float(weather_joined / kept) if kept else None,
           "observed_duration_coverage": float(
               np.mean([row["actual_duration_minutes"] is not None for row in rows_out])
           ) if rows_out else None,
           "files_with_comparable_rows": int(files_with_rows),
           "mae": float(np.mean(abs_err)) if abs_err else None,
           "anchor_tariff_mae": float(np.mean(anchor_abs_err)) if anchor_abs_err else None,
           "policy_mae_improvement_vs_anchor": (
               float(np.mean(anchor_abs_err) - np.mean(abs_err))
               if anchor_abs_err and abs_err else None
           ),
           "rmse": float(np.sqrt(np.mean(sq_err))) if sq_err else None,
           "mape": float(np.mean(abs_pct)) if abs_pct else None,
           "bias": float(np.mean(signed_err)) if signed_err else None,
           "median_absolute_error": float(np.median(abs_err)) if abs_err else None,
           "within_2_dollars": float(np.mean(np.asarray(abs_err) <= 2.0)) if abs_err else None,
           "within_5_dollars": float(np.mean(np.asarray(abs_err) <= 5.0)) if abs_err else None,
           "actual_price_mean": float(np.mean([row["actual_paid"] for row in rows_out])) if rows_out else None,
           "predicted_price_mean": float(np.mean([row["rl_predicted_price"] for row in rows_out])) if rows_out else None,
           "price_correlation": (
               float(np.corrcoef(
                   [row["actual_paid"] for row in rows_out],
                   [row["rl_predicted_price"] for row in rows_out],
               )[0, 1]) if len(rows_out) > 1 else None
           ),
           "duration_mae_minutes": float(np.mean(dur_abs_err)) if dur_abs_err else None,
           "duration_rmse_minutes": float(np.sqrt(np.mean(dur_sq_err))) if dur_sq_err else None,
           "duration_mape": float(np.mean(dur_abs_pct)) if dur_abs_pct else None,
           "out_csv": out_csv,
           "out_plot_prefix": out_plot_prefix,
           "policy_mode": str(policy_mode),
           "policy_temperature": float(policy_temperature),
           "policy_seed": int(policy_seed),
           "dataset_sample_seed": int(dataset_sample_seed),
           "duration_mode": str(duration_mode),
           "trained_policy_model_id": policy_model_id,
           "trained_policy_artifact_path": policy_artifact_path,
           "trained_policy_artifact_sha256": policy_artifact_sha256,
           "trained_policy_training_seed": policy_training_seed,
           "trained_policy_training_seed_source": policy_training_seed_source,
           "actual_paid_excluded_from_policy_input": True,
       }
       self._print_reality_gap_check(summary)
       self._print_sensitivity_analysis()
       return summary
   
    def _print_reality_gap_check(self, summary: Dict[str, Any]) -> None:
        mae = summary.get("mae")
        mape = summary.get("mape")
        bias = summary.get("bias")
        anchor_mae = summary.get("anchor_tariff_mae")
        improvement = summary.get("policy_mae_improvement_vs_anchor")
        if mae is None:
            print(">>> Reality gap check: insufficient comparable Kaggle rows.")
            return
        bias_dir = "overpricing" if (bias or 0.0) > 0 else "underpricing"
        print(
            ">>> Reality gap check: "
            f"policy_mae={float(mae):.3f}, anchor_mae={float(anchor_mae or 0.0):.3f}, "
            f"policy_improvement={float(improvement or 0.0):+.3f}, "
            f"mape={float(mape or 0.0):.3f}, bias={float(bias or 0.0):.3f} ({bias_dir})."
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
            ">>> Sensitivity analysis (active tariff after restoring the simulation state; baseline ride=3mi clear weekday 14:00): "
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

        action_counts = Counter(str(r.get("policy_action_label", "unknown")) for r in rows_out)
        print(
            ">>> Kaggle learned-policy actions: "
            + ", ".join(f"{label}={count}" for label, count in action_counts.most_common(8))
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
        response_baseline = np.zeros(MARKET_RESPONSE_DIM, dtype=np.float32)
        opponent_action_target, _ = self._firm2_leader_step(
            previous_metrics=getattr(
                self, "_last_cycle_firm2_metrics", None
            ),
            day_of_week=day_ctx.day_of_week,
            hour=hour,
            weather=day_ctx.weather,
            enabled=self.firm2_mode != "static",
        )
        if self.firm1_mode == "RL":
            raw_state, action_features, _ = self._publish_pricing_observation(
                day_of_week=day_ctx.day_of_week,
                hour=hour,
                weather=day_ctx.weather,
            )
            s_vec = self.firm1.stack_state(raw_state)
            critic_state = (
                self._build_privileged_critic_state(raw_state)
                if self.privileged_critic
                else None
            )
            response_baseline = self._response_target_from_metrics(
                FirmMetrics(), FirmMetrics(), 0.0
            )
            action_mask = self._objective_action_mask(
                self.firm1.feasible_action_mask(self.market)
            )
            action, s_ts, old_logp, val, af_ts, mask_ts = self.firm1.agent.act(
                s_vec,
                action_features=action_features,
                action_mask=action_mask,
                critic_state=critic_state,
            )
            self.firm1.apply_action(action, self.market)
            rl_step = (
                action, s_ts, old_logp, val, af_ts, mask_ts, critic_state
            )
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

        results, m1, m2, gap, air, dist = self.simulate_batch(day_ctx.day_of_week, day_ctx.weather, hour, rides)
        self._ingest_platform_observations(m1, m2)
        self._last_cycle_firm2_metrics = m2
        
        reward = self._compute_rl_reward(m1, gap, m2=m2)


        if is_training and self.firm1_mode == "RL" and rl_step is not None:
            (
                action, s_ts, old_logp, val, af_ts, mask_ts, critic_state
            ) = rl_step
            reward_diag = getattr(self, "last_reward_diagnostics", {})
            next_v = None
            next_public_potential = None
            if self.potential_shaping_scale > 0.0:
                next_v, _, _ = self._ppo_bootstrap_estimates(
                    day_of_week=day_ctx.day_of_week,
                    hour=hour,
                    weather=day_ctx.weather,
                )
                next_public_potential = self._public_reward_potential(
                    self._build_rl_state(
                        day_of_week=day_ctx.day_of_week,
                        hour=hour,
                        weather=day_ctx.weather,
                        perspective="Firm1",
                    )
                )
            self.firm1.agent.store(
                s_ts,
                action,
                float(reward),
                False,
                None,
                old_logp,
                val,
                action_features=af_ts,
                action_mask=mask_ts,
                critic_state=critic_state,
                next_value=next_v,
                potential=(
                    self._public_reward_potential(
                        np.asarray(
                            s_ts[-PlatformObservationModel.observation_dim:].cpu().tolist(),
                            dtype=np.float32,
                        )
                    )
                    if self.potential_shaping_scale > 0.0
                    else None
                ),
                next_potential=(
                    next_public_potential
                    if self.potential_shaping_scale > 0.0
                    else None
                ),
            )
            self._apply_eval_guardrail(
                "off",
                metrics=m1,
                price_gap_f2_minus_f1=float(gap),
                base=self.market.curr_market,
            )
        
        self._update_recent_response_emas(m1, gap, m2=m2)
        market_share = float(m1.completed_share)
        rival_market_share = float(m2.completed_share)
        self.last_share = market_share
        self.last_revpr = float(m1.rev_per_request)
        self.last_gap = float(gap)
        self.last_profitpr = float(m1.profit_per_request)
        self.last_fulfillment = float(m1.fulfillment_rate)
        self.last_completion_coverage = float(np.clip(
            m1.completed_share + m2.completed_share, 0.0, 1.0
        ))
        self.last_acceptance = float(m1.driver_acceptance_rate)
        self.last_wait = float(m1.avg_wait_minutes)
        self.last_driver_paypr = float(m1.driver_pay / max(1, m1.total))
        self.last_firm2_share = rival_market_share
        self.last_firm2_revpr = float(m2.rev_per_request)
        self.last_firm2_profitpr = float(m2.profit_per_request)
        self.last_firm2_fulfillment = float(m2.fulfillment_rate)
        self.last_firm2_acceptance = float(m2.driver_acceptance_rate)
        self.last_firm2_wait = float(m2.avg_wait_minutes)
        self.last_firm2_driver_paypr = float(m2.driver_pay / max(1, m2.total))
        self.objective_credit_ready = True
        
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
        """Return the current observable demand intensity."""
        base = self.market.curr_market
        time_multiplier = float(self.market._time_multiplier(int(hour)))
        day_multiplier = float(base.day_multiplier.get(int(day_of_week), 1.0))
        weather_multiplier = float(base.weather_multiplier.get(str(weather), 1.0))
        demand_multiplier = time_multiplier * day_multiplier * weather_multiplier
        return {
            "demand_intensity": self._bounded_context(demand_multiplier, 1.15, 0.45),
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

    def _build_privileged_critic_state(
        self,
        public_state: np.ndarray,
    ) -> np.ndarray:
        """Build a fixed training-only centralized state summary.

        The actor never receives this tensor. It combines the public Firm 1
        observation with exact bilateral tariffs, undelayed bilateral outcomes,
        driver-supply state, and rival-controller execution state so the value
        baseline does not have to infer hidden causes from delayed rewards.
        """
        public = np.asarray(public_state, dtype=np.float32).reshape(-1)
        if public.size != PlatformObservationModel.observation_dim:
            raise ValueError(
                "public state width changed while building privileged critic state: "
                f"{public.size} != {PlatformObservationModel.observation_dim}"
            )
        base = self.market.curr_market
        tariff_ratios: List[float] = []
        for firm in (self.firm1, self.firm2):
            overrides = getattr(firm, "overrides", CoefficientOverrides())
            for key in MANIPULATABLE_PRICE_COEFFICIENT_KEYS:
                anchor = float(max(1e-6, getattr(base, key)))
                tariff_ratios.append(float(np.clip(
                    get_coeff(base, overrides, key) / anchor - 1.0,
                    -1.5,
                    1.5,
                )))

        exact_outcomes = [
            float(np.clip(self.last_share, 0.0, 1.0)),
            float(np.clip(self.last_revpr / self.reward_rev_scale, -2.0, 2.0)),
            float(np.clip(self.last_profitpr / self.reward_profit_scale, -2.0, 2.0)),
            float(np.clip(self.last_fulfillment, 0.0, 1.0)),
            float(np.clip(self.last_firm2_share, 0.0, 1.0)),
            float(np.clip(self.last_firm2_revpr / self.reward_rev_scale, -2.0, 2.0)),
            float(np.clip(self.last_firm2_profitpr / self.reward_profit_scale, -2.0, 2.0)),
            float(np.clip(self.last_firm2_fulfillment, 0.0, 1.0)),
        ]
        if self.enable_driver_supply:
            supply = np.concatenate((
                self.driver_supply.state_features_for_firm(
                    "Firm1", opponent="Firm2"
                ),
                self.driver_supply.state_features_for_firm(
                    "Firm2", opponent="Firm1"
                ),
            )).astype(np.float32, copy=False)
        else:
            supply = np.zeros(16, dtype=np.float32)
        opponent_execution = [
            float(np.clip(
                (float(getattr(self.firm2, "_last_applied_action", -1)) + 1.0)
                / max(1.0, float(getattr(self.firm1.agent, "action_dim", 1))),
                0.0,
                1.0,
            )),
            float(np.clip(
                float(getattr(self.firm2, "_repeat_action_count", 0)) / 10.0,
                0.0,
                1.0,
            )),
            float(np.clip(float(getattr(self.firm2, "step_scale", 1.0)), 0.0, 2.0) / 2.0),
            float(np.clip(
                float(getattr(self.firm2, "supply_incentive_multiplier", 1.0))
                - 1.0,
                -0.5,
                0.5,
            )),
            float(self.firm2_mode != "static"),
            float(bool(self.objective_credit_ready)),
        ]
        opponent_mode = str(getattr(
            self.firm2,
            "_ride_response_mode",
            self.firm2_mode,
        ))
        opponent_context = (
            [
                float(opponent_mode == candidate)
                for candidate in V25_CRITIC_OPPONENT_MODES
            ]
            if self.credit_assignment_v25
            else []
        )
        state = np.concatenate((
            public,
            np.asarray(tariff_ratios, dtype=np.float32),
            np.asarray(exact_outcomes, dtype=np.float32),
            supply,
            np.asarray(opponent_execution, dtype=np.float32),
            np.asarray(opponent_context, dtype=np.float32),
        )).astype(np.float32, copy=False)
        expected_critic_dim = int(self.firm1.agent.critic_state_dim)
        if state.size != expected_critic_dim:
            raise AssertionError(
                "privileged critic schema must remain fixed at "
                f"{expected_critic_dim}, got {state.size}"
            )
        return np.nan_to_num(state, nan=0.0, posinf=2.0, neginf=-2.0)

    def _public_reward_potential(self, public_state: np.ndarray) -> float:
        """Fixed public-state potential used only for optional redistribution."""
        state = np.asarray(public_state, dtype=np.float32).reshape(-1)
        if state.size != PlatformObservationModel.observation_dim:
            raise ValueError(
                "reward potential requires one public observation frame"
            )
        completed_share = float(state[9])
        profit = float(state[10])
        fulfillment = float(state[11])
        alpha = objective_profit_share(self._configured_primary_objective())
        profit_potential = (
            0.80 * profit + 0.10 * completed_share + 0.10 * fulfillment
        )
        competitiveness_potential = (
            0.20 * profit + 0.70 * completed_share + 0.10 * fulfillment
        )
        potential = (
            alpha * profit_potential
            + (1.0 - alpha) * competitiveness_potential
        )
        return float(np.clip(potential, -1.0, 1.0))

    def _snapshot_counterfactual_state(self) -> Dict[str, Any]:
        """Capture every mutable component touched by a one-step branch."""
        tracked = (
            "last_share", "last_revpr", "last_profitpr", "last_gap",
            "last_fulfillment", "last_completion_coverage", "last_acceptance",
            "last_wait", "last_driver_paypr", "last_firm2_share",
            "last_firm2_revpr", "last_firm2_profitpr",
            "last_firm2_fulfillment", "last_firm2_acceptance",
            "last_firm2_wait", "last_firm2_driver_paypr",
            "objective_credit_ready", "airport_rate_last", "mean_distance_last",
        )
        return {
            "market": copy.deepcopy(self.market),
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
            "python_rng_state": random.getstate(),
            "torch_rng_state": torch.random.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
            "choice_model": copy.deepcopy(self.choice_model),
            "firm1_overrides": copy.deepcopy(self.firm1.overrides),
            "firm1_execution": self.firm1.snapshot_execution_state(),
            "firm2": copy.deepcopy(self.firm2),
            "driver_supply": copy.deepcopy(self.driver_supply),
            "observers": {
                name: observer.snapshot()
                for name, observer in self.platform_observers.items()
            },
            "stability": self.action_stability.snapshot(),
            "crowd": copy.deepcopy(
                getattr(self, "last_crowd_response_stats", {})
            ),
            "price_gap": copy.deepcopy(self.last_price_gap_metrics),
            "reward_diagnostics": copy.deepcopy(
                getattr(self, "last_reward_diagnostics", {})
            ),
            "tracked": {name: copy.deepcopy(getattr(self, name)) for name in tracked},
        }

    def _restore_counterfactual_state(self, snapshot: Mapping[str, Any]) -> None:
        """Rollback a branch so the selected real action sees identical noise."""
        self.market = copy.deepcopy(snapshot["market"])
        self.rng.bit_generator.state = copy.deepcopy(snapshot["rng_state"])
        random.setstate(snapshot["python_rng_state"])
        torch.random.set_rng_state(snapshot["torch_rng_state"])
        if torch.cuda.is_available() and snapshot.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(snapshot["cuda_rng_state"])
        self.choice_model = copy.deepcopy(snapshot["choice_model"])
        self.firm1.overrides = copy.deepcopy(snapshot["firm1_overrides"])
        self.firm1.restore_execution_state(snapshot["firm1_execution"])
        self.firm2 = copy.deepcopy(snapshot["firm2"])
        self.driver_supply = copy.deepcopy(snapshot["driver_supply"])
        for name, observer_snapshot in snapshot["observers"].items():
            self.platform_observers[name].restore(observer_snapshot)
        self.action_stability.restore(snapshot["stability"])
        self.last_crowd_response_stats = copy.deepcopy(snapshot["crowd"])
        self.last_price_gap_metrics = copy.deepcopy(snapshot["price_gap"])
        self.last_reward_diagnostics = copy.deepcopy(
            snapshot["reward_diagnostics"]
        )
        for name, value in snapshot["tracked"].items():
            setattr(self, name, copy.deepcopy(value))

    def _counterfactual_action_values(
        self,
        *,
        selected_action: int,
        policy_top_action: int,
        policy_second_action: int,
        action_mask: np.ndarray,
        day_of_week: int,
        hour: int,
        weather: str,
        customers_per_step: int,
        sampled_profiles: List[Dict[str, Any]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate a coverage-balanced action subset under common random numbers.

        These observed targets train the independent Q diagnostic and the
        low-weight pairwise PPO policy-improvement term. No unobserved target
        enters either loss, and no branch action is forced into the real or
        deployed policy.
        """
        del policy_top_action, policy_second_action
        values = np.full(self.firm1.agent.action_dim, np.nan, dtype=np.float32)
        mask = np.asarray(action_mask, dtype=bool).reshape(-1)
        if self.counterfactual_branches <= 0:
            return values, np.zeros_like(mask)
        feasible = np.flatnonzero(mask).astype(int).tolist()
        candidates: List[int] = []
        selected = int(selected_action)
        if selected in feasible:
            candidates.append(selected)
        # The v24 top/runner-up rule repeatedly supervised the same few actions,
        # leaving most of the Q surface unconstrained. Select alternatives by
        # least prior CRN exposure. This changes only which *simulated* branches
        # are labeled; it never changes the real or deployed policy action.
        alternatives = sorted(
            (action for action in feasible if action != selected),
            key=lambda action: (
                int(self.counterfactual_action_counts[action]), action
            ),
        )
        candidates.extend(
            alternatives[: max(0, self.counterfactual_branches - len(candidates))]
        )
        for action in candidates:
            self.counterfactual_action_counts[action] += 1
        snapshot = self._snapshot_counterfactual_state()
        try:
            for action in candidates:
                self._restore_counterfactual_state(snapshot)
                self.firm1.apply_action(action, self.market)
                _, m1, m2, mean_gap, _, _ = self.simulate_batch(
                    day_of_week=day_of_week,
                    weather=weather,
                    hour=hour,
                    customers_per_step=customers_per_step,
                    sampled_profiles=sampled_profiles,
                    collect_rows=False,
                )
                diag = self._reward_diagnostics(
                    share=float(m1.chosen_share),
                    completed_share=float(m1.completed_share),
                    rev_per_request=float(m1.rev_per_request),
                    mean_gap=float(mean_gap),
                    prev_share=float(self.last_share),
                    prev_rev_per_request=float(self.last_revpr),
                    prev_profit_per_request=float(self.last_profitpr),
                    prev_rival_profit_per_request=float(self.last_firm2_profitpr),
                    prev_rival_market_share=float(self.last_firm2_share),
                    prev_fulfillment_rate=float(self.last_fulfillment),
                    prev_gap=float(self.last_gap),
                    profit_per_request=float(m1.profit_per_request),
                    profit_margin=self._profit_margin(m1),
                    fulfillment_rate=float(m1.fulfillment_rate),
                    avg_wait_minutes=float(m1.avg_wait_minutes),
                    driver_acceptance_rate=float(m1.driver_acceptance_rate),
                    action_change_magnitude=float(
                        self.firm1.last_action_magnitude()
                    ),
                    baseline_share=float(m2.chosen_share),
                    baseline_completed_share=float(m2.completed_share),
                    baseline_rev_per_request=float(m2.rev_per_request),
                    baseline_profit_per_request=float(m2.profit_per_request),
                    action_event=True,
                    price_gap_abs_error=float(self.last_price_gap_metrics["mae"]),
                )
                values[action] = float(diag["reward"])
        finally:
            self._restore_counterfactual_state(snapshot)
        target_mask = np.isfinite(values)
        if bool(target_mask.any()):
            # Common-noise relative action values are the identifiable signal;
            # their action-independent level is already handled by V/rho.
            values[target_mask] -= float(np.mean(values[target_mask]))
        return values, target_mask

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
        gap_abs_errors: List[float] = []
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

            realized_gap = float(p2 - p1)
            gap_sum += realized_gap
            gap_abs_errors.append(
                abs(realized_gap - float(self.reward_target_price_gap))
            )
            
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
                "City": str(getattr(self.market, "current_city", self.market_name)),
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
                    "City": str(getattr(self.market, "current_city", self.market_name)),
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
        gap_error_array = np.asarray(gap_abs_errors or [0.0], dtype=float)
        self.last_price_gap_metrics = {
            # Signed direction: positive means Firm2 is more expensive.
            "signed_mean": mean_gap,
            # Public-quote alignment diagnostic. Taking the absolute error
            # before aggregation prevents overpricing in one segment from
            # cancelling underpricing in another. Named competitiveness uses
            # completed rides per incoming request instead.
            "mae": float(np.mean(gap_error_array)),
            "rmse": float(np.sqrt(np.mean(np.square(gap_error_array)))),
            "p90_abs_error": float(np.quantile(gap_error_array, 0.90)),
            "within_tolerance_rate": float(np.mean(
                gap_error_array <= float(self.constraint_config.overall_tolerance)
            )),
            "target": float(self.reward_target_price_gap),
        }
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
        run_previous_firm2_metrics: Optional[FirmMetrics] = None
        
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
            gap_abs_error_sum = 0.0
            gap_rmse_sum = 0.0
            gap_p90_sum = 0.0
            gap_within_tolerance_sum = 0.0
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

                self._firm2_leader_step(
                    previous_metrics=run_previous_firm2_metrics,
                    day_of_week=day_ctx.day_of_week,
                    hour=hour,
                    weather=day_ctx.weather,
                    enabled=self.firm2_mode != "static",
                )

                # Firm 1 acts only after the configured Firm 2 controller has
                # committed its posted price for this market period.
                rl_step = None
                if self.firm1_mode == "RL":
                    raw_state_vec, action_features, _ = self._publish_pricing_observation(
                        day_of_week=day_ctx.day_of_week, hour=hour, weather=day_ctx.weather
                    )
                    s_vec = self.firm1.stack_state(raw_state_vec)
                    critic_state = (
                        self._build_privileged_critic_state(raw_state_vec)
                        if self.privileged_critic
                        else None
                    )
                    # Evaluation/deployment runs reflect the learned policy, not
                    # PPO's training-time uniform exploration mixture. Use
                    # --run_experiment for optimizer updates and rollout training.
                    action_mask = self._objective_action_mask(
                        self.firm1.feasible_action_mask(self.market)
                    )
                    action, s_ts, old_logp, val, af_ts, mask_ts = self.firm1.agent.act(
                        s_vec,
                        deterministic=True,
                        action_features=action_features,
                        action_mask=action_mask,
                        critic_state=critic_state,
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
                run_previous_firm2_metrics = m2
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
                    prev_rival_profit_per_request=float(
                        self.last_firm2_profitpr
                    ),
                    prev_rival_market_share=float(self.last_firm2_share),
                    prev_fulfillment_rate=float(self.last_fulfillment),
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
                    price_gap_abs_error=float(
                        self.last_price_gap_metrics["mae"]
                    ),
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
                            "off",
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
                    "reward_market_share_utility",
                    "reward_market_share_level_utility",
                    "reward_market_share_dominance_utility",
                    "reward_market_share_competitiveness_component",
                    "reward_market_share_advantage",
                    "reward_market_share_target_gap",
                    "reward_market_share_gap_error",
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
                gap_abs_error_sum += float(self.last_price_gap_metrics["mae"])
                gap_rmse_sum += float(self.last_price_gap_metrics["rmse"])
                gap_p90_sum += float(
                    self.last_price_gap_metrics["p90_abs_error"]
                )
                gap_within_tolerance_sum += float(
                    self.last_price_gap_metrics["within_tolerance_rate"]
                )
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
                market_share = float(m1.completed_share)
                rival_market_share = float(m2.completed_share)
                self.last_share = market_share
                self.last_revpr = float(m1.rev_per_request)
                self.last_gap = float(mean_gap)
                self.last_profitpr = float(m1.profit_per_request)
                self.last_fulfillment = float(m1.fulfillment_rate)
                self.last_completion_coverage = float(np.clip(
                    m1.completed_share + m2.completed_share, 0.0, 1.0
                ))
                self.last_acceptance = float(m1.driver_acceptance_rate)
                self.last_wait = float(m1.avg_wait_minutes)
                self.last_driver_paypr = float(m1.driver_pay / max(1, m1.total))
                self.last_firm2_share = rival_market_share
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
                    "constraint_value_loss", "risk_value_loss", "response_loss", "opponent_forecast_loss",
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
                    "exploration_rate": float(self.firm1.agent.exploration_rate),
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
            avg_gap_abs_error = gap_abs_error_sum / max(1, timesteps_per_day)
            avg_gap_rmse = gap_rmse_sum / max(1, timesteps_per_day)
            avg_gap_p90 = gap_p90_sum / max(1, timesteps_per_day)
            avg_gap_within_tolerance = gap_within_tolerance_sum / max(
                1, timesteps_per_day
            )
            avg_profitpr = profit_sum / max(1, timesteps_per_day)
            avg_no_ride_share = no_ride_share_sum / max(1, timesteps_per_day)
            avg_fulfillment = fulfillment_sum / max(1, timesteps_per_day)
            avg_wait = wait_sum / max(1, timesteps_per_day)
            avg_driver_accept = driver_accept_sum / max(1, timesteps_per_day)
            avg_driver_paypr = driver_paypr_sum / max(1, timesteps_per_day)
            avg_share_two = share_sum_two / max(1, timesteps_per_day)
            avg_choice_share_two = choice_share_sum_two / max(1, timesteps_per_day)
            avg_completed_share_two = completed_share_sum_two / max(1, timesteps_per_day)
            avg_revpr_two = revpr_sum_two / max(1, timesteps_per_day)
            avg_profitpr_two = profit_sum_two / max(1, timesteps_per_day)
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
                    avg_completed_share,
                    avg_revpr,
                    price_gap_f2_minus_f1=avg_gap,
                    profit_per_request=avg_profitpr,
                    competitor_share=avg_completed_share_two,
                    competitor_profit_per_request=avg_profitpr_two,
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
                "avg_revpr_firm2": float(avg_revpr_two),
                "avg_profitpr_firm2": float(avg_profitpr_two),
                "avg_gap": float(avg_gap),
                "avg_gap_abs_error": float(avg_gap_abs_error),
                "avg_gap_rmse": float(avg_gap_rmse),
                "avg_gap_p90_abs_error": float(avg_gap_p90),
                "avg_gap_within_tolerance_rate": float(
                    avg_gap_within_tolerance
                ),
                "avg_gap_target": float(self.reward_target_price_gap),
                "avg_gap_tolerance": float(
                    self.constraint_config.overall_tolerance
                ),
                "avg_gap_violation_025": float(avg_gap_abs_error > 0.25),
                "avg_gap_violation_050": float(avg_gap_abs_error > 0.50),
                "constraint_curriculum_scale": float(self.constraint_curriculum_scale),
                "avg_no_ride_share": float(avg_no_ride_share),
                "avg_fulfillment_rate": float(avg_fulfillment),
                "avg_wait_minutes": float(avg_wait),
                "driver_acceptance_rate": float(avg_driver_accept),
                "driver_pay_per_request": float(avg_driver_paypr),
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
                "ppo_opponent_forecast_loss": float(
                    ppo_metrics.get("opponent_forecast_loss", 0.0)
                ),
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
    benchmark_report: Optional[Mapping[str, Any]] = None,
    diagnostic_plot_stride: int = 10,
    fare_plot_stride_days: int = 10,
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

    if benchmark_report:
        report = dict(benchmark_report)
        objective = str(report.get("primary_objective", "profit_maximization"))
        if objective == "competitiveness":
            own_values = dict(report.get("mode_own_market_share", {}))
            rival_values = dict(report.get("mode_rival_market_share", {}))
            target_gap = 0.0
            outcome_label = "Completed rides / incoming request"
            gap_label = "Completed-share advantage (Firm 1 - benchmark)"
        elif objective == "profit_maximization":
            own_values = dict(report.get("mode_own_profit_per_request", {}))
            rival_values = dict(report.get("mode_rival_profit_per_request", {}))
            target_gap = 0.0
            outcome_label = "Contribution profit / request ($)"
            gap_label = "Profit advantage / request ($)"
        else:
            own_values = dict(report.get("mode_own_objective_score", {}))
            rival_values = dict(report.get("mode_rival_objective_score", {}))
            target_gap = 0.0
            outcome_label = "Normalized mixed-objective score"
            gap_label = "Mixed-objective advantage (Firm 1 - benchmark)"
        modes = [mode for mode in BENCHMARK_OPPONENT_MODES if mode in own_values]
        if modes:
            validation_cities = dict(
                report.get("mode_validation_cities", {}) or {}
            )
            benchmark_market = str(
                next(
                    iter(validation_cities.values()),
                    report.get("market_name", "configured market"),
                )
            )
            own = np.asarray([float(own_values[mode]) for mode in modes])
            rival = np.asarray([float(rival_values[mode]) for mode in modes])
            advantages = own - rival
            fulfillment = dict(report.get("mode_fulfillment_rate", {}))
            completion_coverage = dict(
                report.get("mode_completion_coverage", {})
            )
            own_profit = dict(report.get("mode_own_profit_per_request", {}))
            rival_profit = dict(report.get("mode_rival_profit_per_request", {}))
            own_completed_share = dict(report.get("mode_own_market_share", {}))
            rival_completed_share = dict(report.get("mode_rival_market_share", {}))
            gap_abs_error = dict(report.get("mode_price_gap_abs_error", {}))
            selected_action_feasibility = dict(
                report.get("mode_selected_action_feasibility", {})
            )
            tail_advantages = dict(
                report.get(
                    "mode_profit_advantages"
                    if objective == "profit_maximization"
                    else "mode_completed_share_advantages"
                    if objective == "competitiveness"
                    else "mode_objective_advantages",
                    {},
                )
            )
            dominance_rates = dict(
                report.get("mode_objective_dominance_rate", {}) or {}
            )
            dominance_floor = float(
                report.get("objective_dominance_rate_floor", 0.90)
            )
            rows_out = []
            for mode, own_value, rival_value, advantage in zip(
                modes, own, rival, advantages
            ):
                tail_advantage = float(
                    tail_advantages.get(mode, -float("inf"))
                )
                dominance_rate = float(dominance_rates.get(mode, 0.0))
                passes = bool(
                    tail_advantage > target_gap
                    and dominance_rate >= dominance_floor
                    and (
                        objective != "profit_maximization"
                        or float(own_profit.get(mode, -float("inf"))) > 0.0
                    )
                )
                rows_out.append({
                    "market": benchmark_market,
                    "objective": objective,
                    "comparison_checkpoint_status": str(
                        report.get(
                            "comparison_checkpoint_status",
                            "unknown",
                        )
                    ),
                    "comparison_checkpoint_day": report.get(
                        "comparison_checkpoint_day"
                    ),
                    "deployment_variant": str(
                        report.get("deployment_variant", "unknown")
                    ),
                    "deployment_policy_mode": str(
                        report.get("deployment_policy_mode", "unknown")
                    ),
                    "certification_independent": bool(
                        report.get(
                            "comparison_runs_independently_of_certification",
                            True,
                        )
                    ),
                    "opponent_class_seen_in_training": bool(
                        mode in set(
                            report.get("training_dynamic_opponent_modes", [])
                        )
                    ),
                    "trajectory_held_out": True,
                    "profile_partition_held_out": bool(
                        report.get(
                            "evaluation_profile_partition_held_out",
                            False,
                        )
                    ),
                    "benchmark": mode,
                    "firm1_outcome": float(own_value),
                    "benchmark_outcome": float(rival_value),
                    "advantage": float(advantage),
                    "required_advantage": float(target_gap),
                    "final_quarter_advantage": tail_advantage,
                    "rolling_dominance_rate": dominance_rate,
                    "required_rolling_dominance_rate": dominance_floor,
                    "firm1_profit_per_request": float(own_profit.get(mode, np.nan)),
                    "firm2_profit_per_request": float(rival_profit.get(mode, np.nan)),
                    "firm1_completed_share": float(
                        own_completed_share.get(mode, np.nan)
                    ),
                    "firm2_completed_share": float(
                        rival_completed_share.get(mode, np.nan)
                    ),
                    "firm1_fulfillment_rate": float(fulfillment.get(mode, np.nan)),
                    "completion_coverage": float(
                        completion_coverage.get(mode, np.nan)
                    ),
                    "unserved_demand_rate": float(
                        1.0 - completion_coverage.get(mode, np.nan)
                    ),
                    "mean_price_gap_abs_error": float(
                        gap_abs_error.get(mode, np.nan)
                    ),
                    "selected_action_feasibility_rate": float(
                        selected_action_feasibility.get(mode, np.nan)
                    ),
                    "passes_objective": passes,
                })
            summary_out = f"{prefix}_benchmark_summary.csv"
            _write_csv(summary_out, rows_out)
            print(f"Saved -> {summary_out}")

            x = np.arange(len(modes), dtype=float)
            labels = [mode.replace("_", "\n") for mode in modes]
            fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            width = 0.36
            axes[0].bar(x - width / 2, own, width, label="Firm 1 PPO", color="tab:blue")
            axes[0].bar(x + width / 2, rival, width, label="Benchmark", color="tab:orange")
            axes[0].set_ylabel(outcome_label)
            axes[0].legend(loc="best")
            axes[0].grid(axis="y", linestyle="--", alpha=0.25)
            colors = ["tab:green" if row["passes_objective"] else "tab:red" for row in rows_out]
            axes[1].bar(x, advantages, color=colors, alpha=0.85)
            axes[1].axhline(0.0, color="black", linewidth=0.8)
            axes[1].axhline(
                target_gap,
                color="tab:green",
                linestyle="--",
                linewidth=1.2,
                label="dominance boundary (Firm 1 = benchmark)",
            )
            axes[1].set_ylabel(gap_label)
            axes[1].set_xticks(x, labels)
            axes[1].legend(loc="best")
            axes[1].grid(axis="y", linestyle="--", alpha=0.25)
            fig.suptitle(
                f"{benchmark_market} Firm 1 PPO vs Five Benchmark Policies "
                f"({objective.replace('_', ' ')})"
            )
            fig.tight_layout()
            out = f"{prefix}_benchmark_performance.png"
            fig.savefig(out, dpi=150)
            print(f"Saved graph -> {out}")
            plt.close(fig)

            trajectories = dict(
                report.get("benchmark_trajectories", {}) or {}
            )
            for mode in modes:
                trajectory = dict(trajectories.get(mode, {}) or {})
                required = (
                    "day",
                    "normalized_objective_reward",
                    "firm1_profit_per_request",
                    "firm2_profit_per_request",
                    "firm1_completed_share",
                    "firm2_completed_share",
                    "firm1_fulfillment_rate",
                    "completion_coverage",
                    "price_gap_abs_error",
                )
                if not all(trajectory.get(key) for key in required):
                    continue
                length = min(len(trajectory[key]) for key in required)
                if length <= 0:
                    continue
                day_values = np.asarray(trajectory["day"][:length], dtype=float)
                reward_values = np.asarray(
                    trajectory["normalized_objective_reward"][:length],
                    dtype=float,
                )
                firm1_profit = np.asarray(
                    trajectory["firm1_profit_per_request"][:length], dtype=float
                )
                firm2_profit = np.asarray(
                    trajectory["firm2_profit_per_request"][:length], dtype=float
                )
                firm1_share = np.asarray(
                    trajectory["firm1_completed_share"][:length], dtype=float
                )
                firm2_share = np.asarray(
                    trajectory["firm2_completed_share"][:length], dtype=float
                )
                firm1_fulfillment = np.asarray(
                    trajectory["firm1_fulfillment_rate"][:length], dtype=float
                )
                coverage = np.asarray(
                    trajectory["completion_coverage"][:length], dtype=float
                )
                gap_abs_error = np.asarray(
                    trajectory["price_gap_abs_error"][:length], dtype=float
                )
                trajectory_rows = [
                    {
                        "day": int(day_values[index]),
                        "benchmark": mode,
                        "objective": objective,
                        "normalized_objective_reward": float(reward_values[index]),
                        "firm1_profit_per_request": float(firm1_profit[index]),
                        "firm2_profit_per_request": float(firm2_profit[index]),
                        "profit_gap_per_request": float(
                            firm1_profit[index] - firm2_profit[index]
                        ),
                        "firm1_completed_share": float(firm1_share[index]),
                        "firm2_completed_share": float(firm2_share[index]),
                        "completed_share_gap": float(
                            firm1_share[index] - firm2_share[index]
                        ),
                        "firm1_fulfillment_rate": float(
                            firm1_fulfillment[index]
                        ),
                        "completion_coverage": float(coverage[index]),
                        "unserved_demand_rate": float(1.0 - coverage[index]),
                        "price_gap_abs_error": float(gap_abs_error[index]),
                    }
                    for index in range(length)
                ]
                trajectory_out = f"{prefix}_benchmark_{mode}_trajectory.csv"
                _write_csv(trajectory_out, trajectory_rows)
                print(f"Saved -> {trajectory_out}")

                smooth_window = max(5, min(50, length // 8 or 1))

                def _benchmark_ma(values: np.ndarray) -> np.ndarray:
                    cumulative = np.cumsum(
                        np.insert(np.asarray(values, dtype=float), 0, 0.0)
                    )
                    result = np.empty(length, dtype=float)
                    for index in range(length):
                        start = max(0, index + 1 - smooth_window)
                        result[index] = (
                            cumulative[index + 1] - cumulative[start]
                        ) / float(index + 1 - start)
                    return result

                smooth_reward = _benchmark_ma(reward_values)
                dominance_rate = float(np.mean(smooth_reward > 0.0))
                fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
                axes[0].scatter(
                    day_values,
                    reward_values,
                    color="0.55",
                    alpha=0.10,
                    s=7,
                    linewidths=0,
                    label="realized normalized reward",
                )
                axes[0].plot(
                    day_values,
                    smooth_reward,
                    color="tab:purple",
                    linewidth=2.2,
                    label=f"MA{smooth_window}; positive={dominance_rate:.1%}",
                )
                axes[0].axhline(
                    0.0,
                    color="tab:green",
                    linestyle="--",
                    linewidth=1.2,
                    label="zero objective reward",
                )
                axes[0].set_ylabel("Normalized objective reward")
                axes[0].legend(loc="best")

                metric_panels = (
                    (
                        firm1_profit,
                        firm2_profit,
                        "Contribution profit / request ($)",
                    ),
                    (
                        firm1_share,
                        firm2_share,
                        "Completed rides / incoming request",
                    ),
                )
                for axis, (own_series, rival_series, ylabel) in zip(
                    axes[1:], metric_panels
                ):
                    axis.plot(
                        day_values,
                        own_series,
                        color="tab:blue",
                        alpha=0.10,
                        linewidth=0.7,
                        label="Firm 1 raw",
                    )
                    axis.plot(
                        day_values,
                        rival_series,
                        color="tab:orange",
                        alpha=0.10,
                        linewidth=0.7,
                        label="Firm 2 raw",
                    )
                    axis.plot(
                        day_values,
                        _benchmark_ma(own_series),
                        color="tab:blue",
                        linewidth=2.0,
                        label=f"Firm 1 MA{smooth_window}",
                    )
                    axis.plot(
                        day_values,
                        _benchmark_ma(rival_series),
                        color="tab:orange",
                        linewidth=2.0,
                        label=f"Firm 2 MA{smooth_window}",
                    )
                    axis.set_ylabel(ylabel)
                    axis.legend(loc="best", ncol=2, fontsize=8)
                    axis.grid(axis="y", linestyle="--", alpha=0.25)
                axes[2].set_xlabel("Benchmark evaluation day")
                axes[0].grid(axis="y", linestyle="--", alpha=0.25)
                fig.suptitle(
                    "Firm 1 PPO vs "
                    f"{mode.replace('_', ' ')} ({objective.replace('_', ' ')})"
                )
                fig.tight_layout()
                trajectory_plot = f"{prefix}_benchmark_{mode}_performance.png"
                fig.savefig(trajectory_plot, dpi=150)
                print(f"Saved graph -> {trajectory_plot}")
                plt.close(fig)
    
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

    def _ewma(
        values: List[float],
        span: int = 50,
    ) -> List[float]:
        """Expected-reward trace with a consistent effective window."""
        if not values:
            return []
        smoothed = [float(values[0])]
        resolved_span = int(max(1, min(span, len(values))))
        weight = float(2.0 / (resolved_span + 1.0))
        for value in values[1:]:
            smoothed.append(
                float(weight * float(value) + (1.0 - weight) * smoothed[-1])
            )
        return smoothed

    def _report_smoothing_window(values: List[float]) -> int:
        """Use the same readable smoothing convention in every report."""
        if len(values) <= 1:
            return 1
        return int(max(5, min(50, len(values) // 8)))

    def _rolling_reward_summary(
        values: List[float],
        window: int,
    ) -> Tuple[List[float], List[float], List[float]]:
        """Return expected reward and a 95% uncertainty band.

        Realized reward varies with hour, weather, demand mix, supply, and the
        rival even after a policy has converged.  Reporting a connected raw
        line made that irreducible market noise look like optimizer
        oscillation.  Every workflow now uses this same trailing expectation
        and standard-error convention.
        """
        means: List[float] = []
        lower: List[float] = []
        upper: List[float] = []
        resolved = int(max(1, window))
        for index in range(len(values)):
            start = max(0, index + 1 - resolved)
            sample = np.asarray(values[start : index + 1], dtype=float)
            sample = sample[np.isfinite(sample)]
            if sample.size == 0:
                means.append(float("nan"))
                lower.append(float("nan"))
                upper.append(float("nan"))
                continue
            mean = float(np.mean(sample))
            standard_error = float(
                np.std(sample, ddof=1) / np.sqrt(sample.size)
                if sample.size > 1
                else 0.0
            )
            radius = 1.96 * standard_error
            means.append(mean)
            lower.append(mean - radius)
            upper.append(mean + radius)
        return means, lower, upper

    def _plot_reward_expectation(
        ax: Any,
        xs: List[int],
        values: List[float],
        *,
        window: int,
        label: str,
    ) -> List[float]:
        """Draw one uniform, convergence-readable reward view."""
        expected, lower, upper = _rolling_reward_summary(values, window)
        ax.scatter(
            xs,
            values,
            label="realized reward",
            color="0.55",
            alpha=0.10,
            s=7,
            linewidths=0,
        )
        ax.fill_between(
            xs,
            lower,
            upper,
            color="tab:blue",
            alpha=0.16,
            linewidth=0,
            label="95% expected-reward band",
        )
        _plot_timeseries(
            ax,
            xs,
            expected,
            label=label,
            color="tab:blue",
            marker="",
            linewidth=2.4,
        )
        return expected

    # 3) run reward trajectory + convergence diagnostics
    if run_logs:
        xs = [int(r["day"]) for r in run_logs]
        ys = [_extract_reward(r) for r in run_logs]
        converged_days = [int(r["day"]) for r in run_logs if bool(r.get("reward_converged", False))]
        fig, ax = plt.subplots(figsize=(9, 4))
        smooth_window = _report_smoothing_window(ys)
        expected = _plot_reward_expectation(
            ax,
            xs,
            ys,
            window=smooth_window,
            label=f"expected reward ({smooth_window}-day mean)",
        )
        ax.axhline(
            0.0,
            color="tab:green",
            linestyle="--",
            linewidth=1.1,
            label="zero objective reward",
        )
        if converged_days:
            conv_day = int(converged_days[0])
            conv_index = max(0, min(len(expected) - 1, xs.index(conv_day)))
            conv_reward = float(expected[conv_index])
            ax.axvline(conv_day, color="tab:green", linestyle="--", linewidth=1.2, label=f"converged day={conv_day}")
            ax.scatter([conv_day], [conv_reward], color="tab:green", zorder=3)
        ax.set_title("Objective Reward Convergence")
        ax.set_xlabel("Day")
        ax.set_ylabel("Normalized dominance reward")
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
        smooth_window = _report_smoothing_window(ys)
        fig, ax = plt.subplots(figsize=(9, 4))
        _plot_reward_expectation(
            ax,
            xs,
            ys,
            window=smooth_window,
            label=f"expected reward ({smooth_window}-day mean)",
        )
        ax.axhline(
            0.0,
            color="tab:green",
            linestyle="--",
            linewidth=1.1,
            label="zero objective reward",
        )
        if any("reward_base" in r for r in training_logs):
            base_ys = [float(r.get("reward_base", np.nan)) for r in training_logs]
            if not np.allclose(
                np.asarray(base_ys, dtype=float),
                np.asarray(ys, dtype=float),
                equal_nan=True,
            ):
                _plot_timeseries(
                    ax, xs, _moving_average(base_ys, smooth_window),
                    label="unclipped objective mean", marker="",
                    linestyle="--", alpha=0.8,
                )
        converged_batches = [
            int(r["batch"]) + 1
            for r in training_logs
            if bool(r.get("reward_converged", False))
        ]
        if converged_batches:
            ax.axvline(
                converged_batches[0],
                color="tab:green",
                linestyle="--",
                linewidth=1.2,
                label=(
                    "in-sample reward convergence "
                    f"day={converged_batches[0]}"
                ),
            )
        
        y_min, y_max = float(np.nanmin(ys)), float(np.nanmax(ys))
        y_span = max(1e-6, y_max - y_min)
        y_pad = max(0.02, 0.10 * y_span)
        y_lo, y_hi = min(y_min - y_pad, 0.0), max(y_max + y_pad, 0.0)

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
        
        validation_points = [
            (int(r["batch"]) + 1, float(r["validation_score"]))
            for r in training_logs
            if np.isfinite(float(r.get("validation_score", np.nan)))
        ]
        validation_axis = None
        if validation_points:
            validation_axis = ax.twinx()
            validation_axis.plot(
                [point[0] for point in validation_points],
                [point[1] for point in validation_points],
                color="purple",
                marker="D",
                linestyle="--",
                linewidth=1.8,
                label="selected held-out score",
            )
            q_points = [
                (int(r["batch"]) + 1, float(r["validation_objective_q_score"]))
                for r in training_logs
                if np.isfinite(
                    float(r.get("validation_objective_q_score", np.nan))
                )
            ]
            raw_points = [
                (int(r["batch"]) + 1, float(r["validation_raw_actor_score"]))
                for r in training_logs
                if np.isfinite(
                    float(r.get("validation_raw_actor_score", np.nan))
                )
            ]
            if q_points:
                validation_axis.plot(
                    [point[0] for point in q_points],
                    [point[1] for point in q_points],
                    color="tab:cyan",
                    marker="o",
                    linestyle=":",
                    linewidth=1.1,
                    alpha=0.85,
                    label="objective-Q candidate",
                )
            if raw_points:
                validation_axis.plot(
                    [point[0] for point in raw_points],
                    [point[1] for point in raw_points],
                    color="tab:orange",
                    marker="s",
                    linestyle=":",
                    linewidth=1.1,
                    alpha=0.85,
                    label="raw-actor candidate",
                )
            validation_axis.set_ylabel("Held-out validation score", color="purple")
            validation_axis.tick_params(axis="y", labelcolor="purple")

        stage_boundaries: List[Tuple[int, str]] = []
        previous_stage = ""
        for row in training_logs:
            stage_name = str(row.get("training_stage", "") or "")
            if stage_name and stage_name != previous_stage:
                stage_boundaries.append((int(row["batch"]) + 1, stage_name))
                previous_stage = stage_name
        for boundary_index, (boundary_x, stage_name) in enumerate(
            stage_boundaries
        ):
            if boundary_index > 0:
                ax.axvline(
                    boundary_x,
                    color="0.45",
                    linestyle=":",
                    linewidth=1.0,
                    alpha=0.65,
                )
            ax.text(
                boundary_x,
                0.02,
                stage_name,
                transform=ax.get_xaxis_transform(),
                rotation=90,
                va="bottom",
                ha="left",
                color="0.35",
                fontsize=8,
            )

        ax.set_title("Training Objective Reward Convergence")
        ax.set_xlabel("Batch")
        ax.set_ylabel("Normalized dominance reward")
        handles, labels = ax.get_legend_handles_labels()
        if validation_axis is not None:
            extra_handles, extra_labels = validation_axis.get_legend_handles_labels()
            handles += extra_handles
            labels += extra_labels
        ax.legend(handles, labels, loc="best")
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
        smooth_window = _report_smoothing_window(ys)
        fig, ax = plt.subplots(figsize=(9, 4))
        _plot_reward_expectation(
            ax,
            xs,
            ys,
            window=smooth_window,
            label=f"expected reward ({smooth_window}-day mean)",
        )
        ax.axhline(
            0.0,
            color="tab:green",
            linestyle="--",
            linewidth=1.1,
            label="zero objective reward",
        )
        if any("reward_base" in r for r in evaluation_logs):
            base_ys = [float(r.get("reward_base", np.nan)) for r in evaluation_logs]
            if not np.allclose(
                np.asarray(base_ys, dtype=float),
                np.asarray(ys, dtype=float),
                equal_nan=True,
            ):
                _plot_timeseries(
                    ax, xs, _moving_average(base_ys, smooth_window),
                    label="unclipped objective mean", marker="",
                    linestyle="--", alpha=0.8,
                )
        tail_size = max(20, len(ys) // 4)
        if tail_size <= len(ys):
            late_mean = float(np.nanmean(ys[-tail_size:]))
            ax.axhline(
                late_mean,
                color="purple",
                linestyle=":",
                linewidth=1.5,
                label=f"late mean={late_mean:.3f}",
            )
        ax.set_title("Evaluation Objective Reward Stability (Frozen Policy)")
        ax.set_xlabel("Day")
        ax.set_ylabel("Normalized dominance reward")
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

        def _sample_logs(
            source: List[Dict[str, Any]], stride: int
        ) -> List[Dict[str, Any]]:
            """Retain endpoints, stage openings, and regular diagnostic points."""
            stride = max(1, int(stride))
            if stride <= 1 or len(source) <= 2:
                return list(source)
            selected: List[Dict[str, Any]] = []
            previous_stage = None
            for index, row in enumerate(source):
                x_value = int(row[x_key]) + (1 if x_key == "batch" else 0)
                stage = str(row.get("training_stage", "") or "")
                stage_changed = bool(stage and stage != previous_stage)
                if (
                    index == 0
                    or index == len(source) - 1
                    or x_value % stride == 0
                    or stage_changed
                ):
                    selected.append(row)
                if stage:
                    previous_stage = stage
            return selected

        # Directly comparable two-firm business outcomes.  The original
        # dashboard showed only Firm 1 share/revenue plus a fare gap, which
        # could make a low-price, low-profit policy look successful.  Preserve
        # that diagnostic below and add the actual competitive objective.
        if x_key == "batch":
            competitive_keys = (
                ("avg_market_share", "avg_firm2_market_share"),
                ("avg_rev_per_request", "avg_firm2_rev_per_request"),
                ("avg_profit_per_request", "avg_firm2_profit_per_request"),
            )
        else:
            competitive_keys = (
                ("rl_market_share", "heuristic_market_share"),
                ("rl_revenue", "heuristic_revenue"),
                ("rl_profit", "heuristic_profit"),
            )
        if all(
            own_key in logs[0] and rival_key in logs[0]
            for own_key, rival_key in competitive_keys
        ):
            fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
            ylabels = (
                "Completed-request share\n(sum = completion coverage)",
                "Revenue / request",
                "Profit / request",
            )
            for ax, (own_key, rival_key), ylabel in zip(
                axes[:3], competitive_keys, ylabels
            ):
                own = [float(r.get(own_key, np.nan)) for r in logs]
                rival = [float(r.get(rival_key, np.nan)) for r in logs]
                smooth_window = _report_smoothing_window(own)
                _plot_timeseries(
                    ax, xs, own, label="Firm 1 raw", color="tab:blue",
                    alpha=0.08, linewidth=0.6,
                )
                _plot_timeseries(
                    ax, xs, rival, label="Firm 2 raw", color="tab:orange",
                    alpha=0.08, linewidth=0.6,
                )
                _plot_timeseries(
                    ax, xs, _moving_average(own, smooth_window),
                    label=f"Firm 1 MA{smooth_window}", color="tab:blue",
                    marker="", linewidth=2.0,
                )
                _plot_timeseries(
                    ax, xs, _moving_average(rival, smooth_window),
                    label=f"Firm 2 MA{smooth_window}", color="tab:orange",
                    marker="", linewidth=2.0,
                )
                ax.set_ylabel(ylabel)
                ax.grid(axis="y", linestyle="--", alpha=0.25)
                ax.legend(loc="best", ncol=2, fontsize=8)
            axes[0].set_ylim(-0.02, 1.02)
            axes[2].axhline(0.0, color="black", linewidth=0.8, alpha=0.35)
            objective = str(
                logs[0].get("policy_objective", "profit_maximization")
            )
            if objective == "competitiveness":
                own_key, rival_key = competitive_keys[0]
                target_gap = 0.0
                gap_label = "Market-share gap (Firm 1 − Firm 2)"
                target_label = "dominance boundary (Firm 1 = Firm 2)"
                objective_gaps = [
                    float(row.get(own_key, np.nan))
                    - float(row.get(rival_key, np.nan))
                    for row in logs
                ]
            elif objective == "profit_maximization":
                own_key, rival_key = competitive_keys[2]
                target_gap = 0.0
                gap_label = "Profit gap / request (Firm 1 − Firm 2)"
                target_label = "dominance boundary (Firm 1 = Firm 2)"
                objective_gaps = [
                    float(row.get(own_key, np.nan))
                    - float(row.get(rival_key, np.nan))
                    for row in logs
                ]
            else:
                weights = objective_reward_weights(objective)
                target_gap = 0.0
                gap_label = "Normalized mixed-objective gap (Firm 1 − Firm 2)"
                target_label = "objective parity boundary"
                objective_gaps = []
                for row in logs:
                    own_profit = float(row.get(competitive_keys[2][0], np.nan))
                    rival_profit = float(row.get(competitive_keys[2][1], np.nan))
                    own_share = float(row.get(competitive_keys[0][0], np.nan))
                    rival_share = float(row.get(competitive_keys[0][1], np.nan))
                    objective_gaps.append(float(
                        weights["own_profit"] * (own_profit - rival_profit) / 2.0
                        + 2.0 * weights["profit_lead"]
                        * (own_profit - rival_profit)
                        + weights["own_completed_share"]
                        * (own_share - rival_share) / 0.40
                        + 2.0 * weights["completed_share_lead"]
                        * (own_share - rival_share) / 0.10
                    ))
            smooth_window = _report_smoothing_window(objective_gaps)
            axes[3].scatter(
                xs,
                objective_gaps,
                color="0.55",
                alpha=0.10,
                s=7,
                linewidths=0,
                label="realized objective gap",
            )
            _plot_timeseries(
                axes[3],
                xs,
                _moving_average(objective_gaps, smooth_window),
                color="tab:purple",
                marker="",
                linewidth=2.2,
                label=f"objective gap mean ({smooth_window}d)",
            )
            axes[3].axhline(
                target_gap,
                color="tab:green",
                linestyle="--",
                linewidth=1.3,
                label=target_label,
            )
            axes[3].set_ylabel(gap_label)
            axes[3].set_xlabel(
                "Training day" if x_key == "batch" else "Evaluation day"
            )
            axes[3].grid(axis="y", linestyle="--", alpha=0.25)
            axes[3].legend(loc="best", fontsize=8)
            fig.suptitle(
                f"{label.title()} Uniform Two-Firm Performance "
                f"({objective.replace('_', ' ')})"
            )
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
            business_logs = _sample_logs(logs, diagnostic_plot_stride)
            business_xs = [
                int(r[x_key]) + (1 if x_key == "batch" else 0)
                for r in business_logs
            ]
            fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
            _plot_timeseries(axes[0], business_xs, [float(r[share_key]) for r in business_logs], label="share", color="tab:blue")
            axes[0].set_ylabel("Completed request share")
            axes[0].grid(axis="y", linestyle="--", alpha=0.25)
            _plot_timeseries(axes[1], business_xs, [float(r[revenue_key]) for r in business_logs], label="rev/request", color="tab:green")
            axes[1].set_ylabel("Revenue/request")
            axes[1].grid(axis="y", linestyle="--", alpha=0.25)
            _plot_timeseries(axes[2], business_xs, [float(r[gap_key]) for r in business_logs], label="F2-F1 gap", color="tab:orange")
            axes[2].axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
            axes[2].set_ylabel("Price gap")
            axes[2].set_xlabel("Batch" if x_key == "batch" else "Day")
            axes[2].grid(axis="y", linestyle="--", alpha=0.25)
            fig.suptitle(
                f"{label.title()} Business Diagnostics "
                f"(every {max(1, int(diagnostic_plot_stride))} periods)"
            )
            fig.tight_layout()
            out = f"{prefix}_{label}_business_diagnostics.png"
            _ensure_parent_dir(out)
            fig.savefig(out, dpi=150)
            print(f"Saved graph -> {out}")
            plt.close(fig)

        gap_error_key = (
            "avg_price_gap_abs_error"
            if x_key == "batch"
            else "price_gap_abs_error"
        )
        gap_p90_key = (
            "avg_price_gap_p90_abs_error"
            if x_key == "batch"
            else "price_gap_p90_abs_error"
        )
        if gap_key in logs[0] and gap_error_key in logs[0]:
            target = float(logs[0].get("price_gap_target", 0.0))
            tolerance = float(logs[0].get("price_gap_tolerance", 0.45))
            signed_gaps = [float(r.get(gap_key, np.nan)) for r in logs]
            gap_mae = [float(r.get(gap_error_key, np.nan)) for r in logs]
            fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
            _plot_timeseries(
                axes[0], xs, signed_gaps, label="signed mean gap F2-F1",
                color="tab:orange", alpha=0.25, linewidth=0.8, marker="",
            )
            _plot_timeseries(
                axes[0], xs, _moving_average(signed_gaps, 20),
                label="signed gap MA20", color="tab:orange",
                linewidth=2.0, marker="",
            )
            axes[0].axhline(
                target, color="black", linestyle="--", linewidth=1.1,
                label=f"target=${target:.2f}",
            )
            axes[0].set_ylabel("Signed gap ($)")
            axes[0].legend(loc="best")
            axes[0].grid(axis="y", linestyle="--", alpha=0.25)

            _plot_timeseries(
                axes[1], xs, gap_mae, label="gap MAE",
                color="tab:red", alpha=0.25, linewidth=0.8, marker="",
            )
            _plot_timeseries(
                axes[1], xs, _moving_average(gap_mae, 20),
                label="gap MAE MA20", color="tab:red",
                linewidth=2.0, marker="",
            )
            if gap_p90_key in logs[0]:
                _plot_timeseries(
                    axes[1], xs,
                    [float(r.get(gap_p90_key, np.nan)) for r in logs],
                    label="p90 absolute error", color="tab:purple",
                    alpha=0.55, linewidth=1.0, marker="",
                )
            axes[1].axhline(
                tolerance, color="black", linestyle=":", linewidth=1.1,
                label=f"MAE tolerance=${tolerance:.2f}",
            )
            axes[1].set_ylabel("Absolute error ($)")
            axes[1].set_xlabel(
                "Training day" if x_key == "batch" else "Evaluation day"
            )
            axes[1].legend(loc="best")
            axes[1].grid(axis="y", linestyle="--", alpha=0.25)
            fig.suptitle(
                f"{label.title()} Price-Gap Performance "
                "(public-quote alignment diagnostic; not policy objective)"
            )
            fig.tight_layout()
            out = f"{prefix}_{label}_price_gap_performance.png"
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
        fare_logs = _sample_logs(logs, fare_plot_stride_days)
        fare_xs = [
            int(r[x_key]) + (1 if x_key == "batch" else 0)
            for r in fare_logs
        ]
        for coeff in coeff_keys:
            fig, ax = plt.subplots(figsize=(9, 4))
            _plot_timeseries(ax, fare_xs, [float(r.get(f"firm1_{coeff}", np.nan)) for r in fare_logs], label=f"Firm1 {coeff}")
            _plot_timeseries(ax, fare_xs, [float(r.get(f"firm2_{coeff}", np.nan)) for r in fare_logs], label=f"Firm2 {coeff}")
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

        representative_weights = {
            "base_fare": 1.0,
            "per_minute": 14.65,
            "per_mile": 3.95,
            "booking_fee": 1.0,
            "airport_fee": 0.0775,
        }
        if all(
            f"firm{firm}_{coefficient}" in fare_logs[0]
            for firm in (1, 2)
            for coefficient in representative_weights
        ):
            representative_fares: Dict[int, List[float]] = {1: [], 2: []}
            for row in fare_logs:
                for firm in (1, 2):
                    representative_fares[firm].append(
                        float(
                            sum(
                                weight
                                * float(row[f"firm{firm}_{coefficient}"])
                                for coefficient, weight in representative_weights.items()
                            )
                        )
                    )
            fig, ax = plt.subplots(figsize=(10, 4.5))
            _plot_timeseries(
                ax,
                fare_xs,
                representative_fares[1],
                label="Firm 1 representative fare",
                color="tab:blue",
            )
            _plot_timeseries(
                ax,
                fare_xs,
                representative_fares[2],
                label="Firm 2 representative fare",
                color="tab:orange",
            )
            ax.set_title(
                f"{label.title()} Representative Fare Trajectory "
                f"(every {max(1, int(fare_plot_stride_days))} days)"
            )
            ax.set_xlabel("Training day" if x_key == "batch" else "Evaluation day")
            ax.set_ylabel("Representative fare ($)")
            ax.grid(axis="y", linestyle="--", alpha=0.25)
            ax.legend(loc="best")
            fig.tight_layout()
            out = f"{prefix}_{label}_fare_trajectory.png"
            _ensure_parent_dir(out)
            fig.savefig(out, dpi=150)
            print(f"Saved graph -> {out}")
            plt.close(fig)

    _plot_experiment_diagnostics(training_logs, "training", "batch")
    _plot_experiment_diagnostics(evaluation_logs, "evaluation", "day")

    # 6) manipulated coefficient trajectories
    if run_logs:
        stride = max(1, int(fare_plot_stride_days))
        sampled_run_logs = [
            row
            for index, row in enumerate(run_logs)
            if index == 0
            or index == len(run_logs) - 1
            or int(row["day"]) % stride == 0
        ]
        xs = [int(r["day"]) for r in sampled_run_logs]
        coeff_keys: List[str] = sorted(
            {
                k[len("firm1_"):]
                for r in sampled_run_logs
                for k in r.keys()
                if k.startswith("firm1_") and ("firm2_" + k[len("firm1_"):]) in r
            }
        )
        for coeff in coeff_keys:
            y1: List[float] = []
            y2: List[float] = []
            valid = True
            for r in sampled_run_logs:
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
    prefix_seed_labels = [
        int(value)
        for value in re.findall(r"(?:^|[_-])seed(\d+)(?=$|[_-])", str(prefix))
    ]
    prefix_seed_mismatch = bool(
        prefix_seed_labels and int(core.seed) not in prefix_seed_labels
    )
    payload = {
        "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "argv": sys.argv,
        "user_seed": args.seed,
        "experiment_seed": args.experiment_seed,
        "effective_seed": core.seed,
        "report_prefix_seed_labels": prefix_seed_labels,
        "report_prefix_seed_mismatch": prefix_seed_mismatch,
        "deterministic_experiment_seed": bool(args.deterministic_experiment_seed),
        "eval_policy_mode": args.eval_policy_mode,
        "eval_policy_seed": args.eval_policy_seed,
        "eval_guardrail_mode": args.eval_guardrail_mode,
        "pricing_control": {
            "manipulated_price_coefficient_count": int(
                core.manipulated_price_coefficient_count
            ),
            "manipulated_price_coefficient_keys": list(core.shared_edit_keys),
            "scope_locked_for_training_validation_evaluation": True,
        },
        "trained_model_archive": getattr(core, "last_saved_model_path", None),
        "trained_model_id": getattr(core, "last_saved_model_id", None),
        "loaded_trained_model_path": getattr(core, "loaded_trained_model_path", None),
        "loaded_trained_model_id": getattr(core, "loaded_trained_model_id", None),
        "loaded_trained_model_sha256": getattr(core, "loaded_trained_model_sha256", None),
        "loaded_trained_model_created_at": getattr(
            core, "loaded_trained_model_created_at", None
        ),
        "loaded_trained_model_training_seed": getattr(
            core, "loaded_trained_model_training_seed", None
        ),
        "loaded_trained_model_training_seed_source": getattr(
            core, "loaded_trained_model_training_seed_source", None
        ),
        "loaded_trained_model_metadata": getattr(
            core, "loaded_trained_model_metadata", None
        ),
        "competitive_backtest": getattr(
            core, "competitive_backtest_report", None
        ),
        "training_certification": getattr(
            core, "training_certification_report", None
        ),
        "benchmark_evaluation": getattr(
            core, "benchmark_evaluation_report", None
        ),
        "generalization": {
            "profile_source": "same_market_threshold_holdout",
            "training_fraction": float(core.training_generalization_fraction),
            "evaluation_fraction": float(core.evaluation_generalization_fraction),
            "evaluation_uses_cached_threshold_profiles": bool(
                core.threshold_profile_source == "cached"
            ),
            "pool_size": int(len(core.generalization_profile_pool)),
            "cities": sorted(
                str(city) for city in core.generalization_profiles_by_city
            ),
        },
        "report_prefix": args.report_prefix,
        "platform_mdp": platform_mdp_config_payload(
            core.observation_config,
            core.positive_reward_config,
            core.constraint_config,
            core.training_stage_scheduler,
            core.long_term_profit_reward_config,
        ),
        "active_reward": {
            "type": "objective_separated_policy_reward_v8",
            "mechanism": core.long_term_profit_reward_model.mechanism_name,
            "relative_profit_definition": (
                "own contribution profit per incoming request minus rival "
                "contribution profit per incoming request"
            ),
            "market_share_competitiveness_definition": (
                "linear Firm1-minus-Firm2 completed rides per incoming request "
                "divided by 0.10 with no target cap; conditional choice share, "
                "profit, public quote gaps, wait, and margin are reporting "
                "diagnostics only"
            ),
            "robust_checkpoint_optimizer": str(
                core.objective_optimizer.mechanism_name
            ),
            "price_gap_metric_definition": (
                "mean(abs((Firm2 public quote - Firm1 public quote) - "
                "target_price_gap)) across quote opportunities"
            ),
            **vars(core.long_term_profit_reward_config),
            "gamma": float(core.firm1.agent.gamma) if core.firm1_mode == "RL" else None,
            "gae_lambda": float(core.firm1.agent.lam) if core.firm1_mode == "RL" else None,
        },
        "credit_assignment": copy.deepcopy(core.credit_assignment_config),
        "args": {
            k: (str(v) if not isinstance(v, (str, int, float, bool, type(None), list, dict)) else v)
            for k, v in vars(args).items()
        },
    }
    _ensure_parent_dir(out)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"Saved -> {out}")
    if prefix_seed_mismatch:
        print(
            "[WARN] Report-prefix seed label does not match the effective run "
            f"seed: prefix labels={prefix_seed_labels}, effective_seed={core.seed}. "
            "The run-config and trained-policy identity are authoritative."
        )

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
    parser.add_argument(
        "--manipulated_price_coefficients",
        "--manipulated_coefficient_count",
        dest="manipulated_price_coefficients",
        type=int,
        choices=range(1, 6),
        default=5,
        metavar="N",
        help=(
            "Number of price coefficients controlled by both firms (1-5), in "
            "the fixed order base_fare, per_minute, per_mile, booking_fee, "
            "airport_fee. A trained artifact locks this scope for validation "
            "and evaluation."
        ),
    )

    parser.add_argument("--pool", type=int, default=20000, help="Static customer pool size.")
    parser.add_argument(
        "--deterministic_torch",
        action="store_true",
        help="Use deterministic Torch kernels where available (may reduce performance).",
    )
    
    parser.add_argument("--report_prefix", type=str, default="artifacts/report")
    parser.add_argument(
        "--diagnostic_plot_stride",
        type=int,
        default=10,
        help=(
            "Plot business-diagnostic markers every N training/evaluation "
            "periods while retaining all periods in performance curves."
        ),
    )
    parser.add_argument(
        "--fare_plot_stride_days",
        type=int,
        default=10,
        help="Plot coefficient and representative-fare trajectories every N days.",
    )
    parser.add_argument("--run_experiment", action="store_true", help="Run workflow-aligned training/eval experiment")
    parser.add_argument("--trained_model_in", type=str, default="", help="Load a policy by explicit path, immutable registry model id/archive filename, or 'latest'.")
    parser.add_argument("--trained_model_out", type=str, default="", help="Optional mutable alias for the final completed dynamic-stage policy. Every training run is also saved immutably in --model_registry_dir.")
    parser.add_argument("--trained_model_id", type=str, default="", help="Stable human-readable id recorded for this training run in the model registry.")
    parser.add_argument("--model_registry_dir", type=str, default="artifacts/trained_models", help="Directory containing immutable trained-policy archives and manifest.jsonl.")
    parser.add_argument("--list_trained_models", action="store_true", help="List available immutable trained-policy archives and exit.")
    parser.add_argument("--eval_only", action="store_true", help="Skip training and evaluate --trained_model_in directly against --firm2_mode.")
    parser.add_argument("--dataset_only", action="store_true", help="Load --trained_model_in and run only --compare_with_dataset, without synthetic evaluation/profile bootstrap.")
    parser.add_argument("--train_timesteps", type=int, default=6000)
    parser.add_argument("--train_customers", type=int, default=5000)
    parser.add_argument("--train_steps_per_day", type=int, default=10, help="Synthetic training timesteps per day (run_experiment mode).")
    parser.add_argument("--ppo_update_interval_days", type=int, default=10, help="Synthetic training days collected before each PPO optimizer update. The default yields enough updates for state-dependent learning.")
    parser.add_argument("--ppo_min_rollout_transitions", type=int, default=96, help="Minimum complete pricing decisions per PPO rollout (default 96, about 48 simulated days at the standard cadence).")
    
    parser.add_argument("--firm1_action_interval_steps", type=int, default=-1, help="Shared Firm1/Firm2 market steps per pricing decision; if both cadence options are provided they must match.")
    parser.add_argument("--firm2_action_interval_days", "--firm2_action_interval_steps", dest="firm2_action_interval_days", type=int, default=-1, help="Legacy name and step-based alias for the shared Firm1/Firm2 pricing interval; <=0 derives cadence from the PPO update interval.")
    parser.add_argument("--ppo_batch_size", type=int, default=48, help="PPO minibatch size; two minibatches per default credit-complete rollout provide enough updates without tiny noisy batches.")
    parser.add_argument("--ppo_update_epochs", "--ppo_epochs", dest="ppo_update_epochs", type=int, default=8, help="Maximum clipped-PPO epochs per rollout; v24 may stop actor epochs early at target KL.")
    parser.add_argument(
        "--state_action_mi_weight",
        type=float,
        default=0.0,
        help="Deprecated compatibility option; PPO uses no MI regularizer.",
    )
    parser.add_argument(
        "--direct_actor_credit_weight",
        type=float,
        default=0.0,
        help=(
            "Deprecated actor-only improvement blend. The default is zero "
            "because semi-MDP returns already attribute the full held-action "
            "horizon and immediate deltas are dominated by market noise."
        ),
    )
    parser.add_argument("--state_frame_stack", type=int, default=16, help="Ordered public frames read by the separate recurrent actor and critic (default: 16 frames, covering several recent pricing actions).")
    parser.add_argument(
        "--credit_assignment_v24",
        action="store_true",
        help=(
            "Enable the staged v24 credit-assignment defaults: training-only "
            "centralized critic, stateful recurrence, three CRN branches, "
            "critic Q supervision, potential shaping, learned reward rate, and "
            "a hard PPO KL stop. No constrained-MDP mechanism is enabled."
        ),
    )
    parser.add_argument(
        "--credit_assignment_v25",
        action="store_true",
        help=(
            "Enable the corrected v25 causal-learning contract: exact discounted "
            "SMDP rewards, actual next-decision bootstrap and potential, every-frame "
            "actor recurrence, direct centralized TD value learning, an independent "
            "action-feature CRN critic, and regime-local reward-rate centering. It "
            "adds no movement reward, action quota, or constrained-MDP restriction."
        ),
    )
    parser.add_argument(
        "--privileged_critic",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Independently enable/disable the training-only centralized critic.",
    )
    parser.add_argument(
        "--persistent_recurrence",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Carry actor/critic GRU state across pricing decisions.",
    )
    parser.add_argument(
        "--counterfactual_branches",
        type=int,
        default=None,
        help="Number of common-random-number action branches per real decision (0 disables).",
    )
    parser.add_argument("--counterfactual_q_coeff", type=float, default=None)
    parser.add_argument("--counterfactual_actor_weight", type=float, default=None)
    parser.add_argument(
        "--potential_shaping_scale",
        type=float,
        default=None,
        help="Scale of optional potential-based reward redistribution (0 disables).",
    )
    parser.add_argument(
        "--hard_kl_stop",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Stop PPO actor epochs once approximate KL exceeds target_kl.",
    )
    parser.add_argument(
        "--reward_rate_beta",
        type=float,
        default=None,
        help="EMA rate for the continuing-task reward baseline (0 uses v23 rollout centering).",
    )
    parser.add_argument(
        "--training_curriculum",
        type=str,
        default="staged",
        choices=["staged", "direct"],
        help=(
            "Training environment sequence. staged (default) uses market, static "
            "opponent, then a longer configured dynamic-opponent stage (20/20/60%%). "
            "direct is retained only as an ablation."
        ),
    )
    parser.add_argument("--training_stage_market_fraction", type=float, default=0.20, help="Fraction of staged training using the city-market reference.")
    parser.add_argument("--training_stage_static_fraction", type=float, default=0.20, help="Fraction of staged training using the fixed opponent.")
    parser.add_argument("--training_stage_dynamic_fraction", type=float, default=0.60, help="Fraction of staged training using the configured adaptive opponent.")
    parser.add_argument("--training_dynamic_episode_days", type=int, default=600, help="Days per independently randomized, varied-opponent episode in the final dynamic stage.")
    parser.add_argument("--training_opening_tariff_randomization", type=float, default=0.15, help="Symmetric tariff-randomization fraction at each stage or dynamic-episode opening.")
    parser.add_argument("--deterministic_experiment_seed", action="store_true", help="If set, keep run_experiment fully deterministic with --experiment_seed (or the resolved run seed).")
    parser.add_argument("--experiment_seed", type=int, default=None, help="Optional explicit run_experiment seed. Overrides stochastic child-seed generation and is recorded in run config.")
    parser.add_argument("--eval_policy_mode", type=str, default="argmax_raw", choices=["argmax", "argmax_raw", "sample_raw", "sample_low_temp"], help="Evaluation action-selection mode. The default deploys the categorical actor's deterministic mode; repeating a whole-tariff target keeps that tariff stable.")
    parser.add_argument("--eval_policy_temperature", type=float, default=0.25, help="Temperature for --eval_policy_mode sample_low_temp.")
    parser.add_argument("--eval_policy_seed", type=int, default=None, help="Torch sampling seed for stochastic evaluation modes; defaults to effective run seed + 700001.")
    parser.add_argument("--eval_guardrail_mode", type=str, default="off", choices=["off"], help="Business guardrails are disabled; retained as a compatibility field.")
    parser.add_argument(
        "--eval_timesteps",
        type=int,
        default=1000,
        help=(
            "Frozen-policy deployment audit horizon. The 1,000-period default "
            "covers delayed adaptive-rival catch-up instead of certifying the "
            "opening transient."
        ),
    )
    parser.add_argument("--eval_customers", type=int, default=1000)
    parser.add_argument(
        "--benchmark_suite",
        action="store_true",
        help=(
            "After policy loading or training, run the canonical five-opponent "
            "suite in the configured market and emit one comparable performance graph."
        ),
    )
    parser.add_argument(
        "--require_competitive_backtest",
        action="store_true",
        help=(
            "Compatibility flag: run and report the late deployment audit. "
            "Its result does not alter reward, checkpoint selection, saving, "
            "registration, or deployed actions."
        ),
    )
    parser.add_argument(
        "--checkpoint_validation_horizon", "--validation_horizon",
        dest="checkpoint_validation_horizon",
        type=int,
        default=0,
        help=(
            "Held-out steps against the configured opponent for scalar-objective "
            "checkpoint selection. Zero matches --eval_timesteps; range 2-1000."
        ),
    )
    parser.add_argument(
        "--checkpoint_validation_interval_days", "--validation_interval_days",
        dest="checkpoint_validation_interval_days",
        type=int,
        default=400,
        help=(
            "Training days between held-out scalar-objective checkpoint "
            "evaluations against the configured opponent."
        ),
    )
    parser.add_argument(
        "--checkpoint_validation_customers", "--validation_customers",
        dest="checkpoint_validation_customers",
        type=int,
        default=0,
        help=(
            "Rides sampled per checkpoint-validation market step; <=0 reuses "
            "--eval_customers. This affects checkpoint selection only; the "
            "final deployment audit always uses --eval_customers."
        ),
    )
    parser.add_argument(
        "--checkpoint_validation_openings", "--validation_openings",
        dest="checkpoint_validation_openings",
        type=int,
        default=3,
        help="Independent tariff and market openings used by checkpoint validation.",
    )
    parser.add_argument("--profiles_out", type=str, default="artifacts/sampled_profiles.csv")
    parser.add_argument("--profiles_log_limit", type=int, default=200000)
    parser.add_argument(
        "--policy_objective",
        type=str,
        default="profit_maximization",
        choices=list(POLICY_OBJECTIVES),
        help=(
            "Named normalized economic objective. The two endpoints use "
            "80/20 percent own-profit/profit-lead and 10/20/70 percent "
            "own-profit/own-completed-share/completed-share-lead. The five "
            "profit_N_competitiveness_M choices are preregistered convex "
            "mixtures of those endpoints. Zero-weight report fields cannot "
            "affect PPO."
        ),
    )
    parser.add_argument("--positive_reward_profit_weight", type=float, default=0.38, help="Deprecated reporting-only weight; it cannot change PPO reward.")
    parser.add_argument("--positive_reward_revenue_weight", type=float, default=0.22, help="Deprecated reporting-only weight; it cannot change PPO reward.")
    parser.add_argument("--positive_reward_completed_demand_weight", type=float, default=0.20, help="Deprecated reporting-only weight; it cannot change PPO reward.")
    parser.add_argument("--positive_reward_service_weight", type=float, default=0.20, help="Deprecated reporting-only weight; it cannot change PPO reward.")
    parser.add_argument("--long_term_profit_scale", type=float, default=2.0, help="Dollar unit used to normalize Firm 1 contribution profit per incoming request.")
    parser.add_argument("--long_term_profit_advantage_scale", type=float, default=1.0, help="Positive dollar-per-request unit used to normalize the Firm1-minus-Firm2 profit reward.")
    parser.add_argument("--long_term_profit_weight", type=float, default=1.0, help="Deprecated and ignored; the named objective selects the fixed current reward weights.")
    parser.add_argument(
        "--profit_dominance_weight",
        type=float,
        default=0.0,
        help=(
            "Deprecated and ignored; profit lead has a fixed objective-specific reward weight."
        ),
    )
    parser.add_argument("--dominance_quality_scale", type=float, default=4.0, help="Deprecated reporting-only scale; there is no rival-relative tie-breaker.")
    parser.add_argument("--market_share_competitiveness_weight", type=float, default=0.0, help="Deprecated and ignored; completed-share lead has a fixed objective-specific reward weight.")
    parser.add_argument("--market_share_target_gap", type=float, default=0.10, help="Reported completed-request-share lead target for Firm 1 over Firm 2; it does not alter PPO reward or checkpoint ordering.")
    parser.add_argument("--market_share_gap_scale", type=float, default=0.10, help="Completed-share-lead normalization used by both objectives; 0.10 means ten percentage points.")
    parser.add_argument("--market_share_level_weight", type=float, default=0.25, help="Deprecated compatibility field; current objective weights are fixed and logged.")
    parser.add_argument("--price_competitiveness_weight", type=float, default=0.0, help="Deprecated compatibility field. Public quote gaps are reported as diagnostics but are not part of named objective rewards.")
    parser.add_argument("--price_gap_reward_scale", type=float, default=1.0, help="Deprecated compatibility scale for stored quote-gap diagnostics.")
    parser.add_argument("--intervention_cost_weight", type=float, default=0.0, help="Deprecated and ignored; PPO has no action-cost shaping.")
    parser.add_argument("--reversal_cost_weight", type=float, default=0.0, help="Deprecated and ignored; PPO has no reversal shaping.")
    parser.add_argument("--ppo_gamma", type=float, default=0.999, help="Operational-step discount for continuing-market semi-MDP credit.")
    parser.add_argument("--ppo_gae_lambda", type=float, default=0.99, help="GAE lambda per pricing decision; with a five-step action interval the default trace spans about 67 decisions.")
    parser.add_argument("--reward_rev_scale", type=float, default=25.0, help="Revenue/request e-folding scale for positive saturation; a value equal to the scale receives about 63.2%% component credit.")
    parser.add_argument("--reward_profit_scale", type=float, default=12.0, help="Profit/request e-folding scale for positive saturation; a value equal to the scale receives about 63.2%% component credit.")
    parser.add_argument("--min_profit_margin", type=float, default=0.08, help="Reporting-only contribution-margin reference; it does not alter PPO reward or checkpoint ordering.")
    parser.add_argument("--min_completion_coverage", type=float, default=0.55, help="Reporting-only combined completed-request-share reference; it does not alter PPO reward or checkpoint ordering.")
    parser.add_argument("--target_price_gap", type=float, default=0.75, help="Diagnostic target public quote gap Firm2-Firm1. This does not enter either named objective reward.")
    parser.add_argument("--price_gap_tolerance", type=float, default=0.45, help="Aggregate two-sided soft-constraint tolerance around target_price_gap.")
    parser.add_argument("--segment_gap_tolerances", type=float, nargs=4, default=[0.65, 0.60, 0.55, 0.55], metavar=("SHORT", "MEDIUM", "LONG", "XLONG"), help="Soft gap tolerances for 0-2, 2-5, 5-10, and 10+ mile quote probes; long trips default tighter because per-mile errors compound.")
    parser.add_argument("--telemetry_delay_steps", type=int, default=1, help="Delay applied to own operational KPI observations.")
    parser.add_argument("--quote_probe_interval_steps", type=int, default=3, help="Operational periods between public competitor quote samples.")
    parser.add_argument("--quote_probe_delay_steps", type=int, default=1, help="Delay before sampled competitor quotes reach the policy.")
    parser.add_argument("--quote_noise_dollars", type=float, default=0.18, help="Standard deviation of public competitor quote measurement noise.")
    parser.add_argument("--quote_missing_probability", type=float, default=0.03, help="Probability that a scheduled competitor quote probe is unavailable.")
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
    parser.add_argument("--driver_expected_acceptance_cutoff", type=float, default=0.65, help="Minimum expected driver acceptance probability required to count an offered ride as accepted in expected driver acceptance mode.")
    parser.add_argument("--driver_state_smoothing", type=float, default=0.35, help="EMA smoothing alpha for driver features fed into the RL state.")
    parser.add_argument("--enable_constrained_reward", dest="enable_constrained_reward", action="store_true", default=False, help="Deprecated compatibility option; constraints remain diagnostic only.")
    parser.add_argument("--disable_constrained_reward", dest="enable_constrained_reward", action="store_false", help="Keep operational constraints diagnostic only.")
    parser.add_argument("--constraint_lr", type=float, default=0.01, help="Learning rate for bounded operational-constraint multipliers. The lower default prevents feasibility duals from overwhelming the selected economic objective.")
    parser.add_argument("--constraint_penalty_scale", type=float, default=0.10, help="Risk-advantage coefficient at full constraint curriculum scale.")
    parser.add_argument("--constraint_curriculum_start_scale", type=float, default=0.25, help="Legacy constraint-diagnostic curriculum scale used during early exploration.")
    parser.add_argument("--constraint_curriculum_mid_scale", type=float, default=0.60, help="Legacy constraint-diagnostic curriculum scale reached by mid-training.")
    parser.add_argument("--constraint_curriculum_end_scale", type=float, default=1.00, help="Legacy constraint-diagnostic curriculum scale at the end of training.")
    parser.add_argument("--threshold_cache_path", type=str, default="", help="Optional JSONL path for cached GPT/fallback threshold-enriched profiles.")
    parser.add_argument("--threshold_profile_source", type=str, default="generated", choices=["generated", "cached"], help="Use completely newly generated threshold profiles, or load threshold profiles from --threshold_cache_path.")
    parser.add_argument(
        "--threshold_prompt_repeat_source",
        type=str,
        default="",
        help=(
            "Optional primary threshold JSONL whose exact profiles and cold-start rides "
            "are re-submitted to GPT. This isolates prompt/API repeatability from "
            "population and ride resampling."
        ),
    )
    parser.add_argument(
        "--training_generalization_fraction",
        type=float,
        default=0.0,
        help=(
            "Fraction of each training batch drawn from the same-market held-out "
            "partition. The case-study default is zero to keep that split unseen."
        ),
    )
    parser.add_argument(
        "--evaluation_generalization_fraction",
        type=float,
        default=1.0,
        help=(
            "Fraction of ordinary evaluation drawn from the same-market held-out "
            "partition. In cached mode it preserves the configured market's cached threshold distribution."
        ),
    )
    parser.add_argument("--no_save_threshold_cache", action="store_true", help="Do not write threshold profiles to --threshold_cache_path after bootstrap.")
    parser.add_argument("--gpt_threshold_include_rationales", action="store_true", help="Request GPT rationales for thresholds; slower but useful for debugging.")
    parser.add_argument("--gpt_threshold_coldstart_rides", type=int, default=5, help="Cold-start rides per profile sent/summarized for threshold bootstrapping.")
    parser.add_argument("--gpt_threshold_send_raw_rides", action="store_true", help="Send raw cold-start ride rows in addition to compact summaries; slower but more detailed.")
    parser.add_argument("--gpt_threshold_batch_size", type=int, default=20, help="Profiles per GPT price-threshold API request. Larger values improve API utilization but increase payload size.")
    parser.add_argument("--gpt_threshold_max_retries", type=int, default=2, help="Retries per GPT price-threshold batch before deterministic fallback is used.")
    parser.add_argument("--gpt_threshold_failure_pause", type=float, default=1.0, help="Seconds to pause after a failed GPT threshold batch before sending the next batch; helps avoid connection-refused cascades.")
    parser.add_argument("--calibration_csv", type=str, default="", help="Optional historical CSV used to calibrate priors and choice sensitivity.")
    parser.add_argument("--calibration_json", type=str, default="", help="City preprocessing calibration JSON produced by scripts/prepare_city_data.py.")
    parser.add_argument("--calibration_city", type=str, default="", help="Optional city filter for --calibration_csv; defaults to --market if omitted.")
    parser.add_argument("--calibration_preset", type=str, default="", choices=["", "nyc_public"], help="Optional built-in calibration preset. Use nyc_public explicitly for NYC TLC + ACS + weather priors.")
    parser.add_argument("--compare_with_dataset", action="store_true", help="After training/run, compare RL-implied prices against actual paid prices in the Kaggle rideshare dataset files.")
    parser.add_argument("--dataset_root", type=str, default="", help="Optional local dataset root. When omitted, --compare_with_dataset downloads/resolves the configured Kaggle dataset.")
    parser.add_argument("--dataset_city", type=str, default="", help="Dataset city; defaults to --market and must match it unless a separate transfer experiment is implemented.")
    parser.add_argument("--dataset_glob", type=str, default="auto", help="Glob for dataset discovery, or 'auto' to use the city catalog.")
    parser.add_argument("--weather_root", type=str, default="", help="Optional NOAA Global Hourly directory used for timezone-aware ride/weather joins.")
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
    parser.add_argument("--comparison_limit", type=int, default=5000, help="Max number of dataset rows to score during learned-policy validation; parquet row groups are sampled round-robin for temporal coverage.")
    parser.add_argument("--dataset_preview_rows", type=int, default=5, help="How many raw dataset rows to print once as format preview during RL-vs-actual comparison.")
    parser.add_argument("--comparison_policy_mode", type=str, default="argmax_raw", choices=["argmax", "argmax_raw", "sample_raw", "sample_low_temp"], help="One-step PPO actor action mode used independently for each dataset ride row.")
    parser.add_argument("--comparison_policy_temperature", type=float, default=0.25, help="Temperature for stochastic dataset-row policy scoring.")
    parser.add_argument("--comparison_policy_seed", type=int, default=99173, help="Sampling seed for reproducible dataset-row policy scoring.")
    parser.add_argument("--comparison_dataset_seed", type=int, default=104729, help="Seed controlling parquet row-group order and within-batch row sampling for independent dataset robustness runs.")
    parser.add_argument(
        "--comparison_duration_mode",
        type=str,
        default="actual_if_available",
        choices=["actual_if_available", "predicted_only"],
        help="Use observed trip duration when available, or force the model's estimated duration to stress-test price validation without actual-duration input.",
    )

    args = parser.parse_args()
    explicit_reward_flags = {
        "--long_term_profit_weight",
        "--profit_dominance_weight",
        "--market_share_competitiveness_weight",
        "--market_share_target_gap",
        "--market_share_level_weight",
        "--price_competitiveness_weight",
        "--target_price_gap",
        "--price_gap_reward_scale",
    }
    argv_flag_names = {
        str(token).split("=", 1)[0]
        for token in sys.argv[1:]
        if str(token).startswith("--")
    }
    ignored_reward_flags = explicit_reward_flags.intersection(argv_flag_names)
    if ignored_reward_flags:
        print(
            "[WARN] Shaped reward flags are ignored by the direct named "
            "objectives: " + ", ".join(sorted(ignored_reward_flags))
        )
    for key, value in policy_objective_defaults(args.policy_objective).items():
        setattr(args, key, value)
    if float(args.direct_actor_credit_weight) != 0.0:
        print(
            "[WARN] --direct_actor_credit_weight is ignored for named "
            "objectives; their held-action semi-MDP return is the sole actor "
            "credit signal."
        )
        args.direct_actor_credit_weight = 0.0
    if args.list_trained_models:
        records = list_registered_models(args.model_registry_dir)
        if not records:
            print(f"No trained policies registered in {os.path.abspath(os.path.expanduser(args.model_registry_dir))}")
        for record in records:
            print(json.dumps(record, sort_keys=True))
        return
    if args.eval_only and not args.trained_model_in:
        parser.error("--eval_only requires --trained_model_in")
    if args.dataset_only and (not args.trained_model_in or not args.compare_with_dataset):
        parser.error("--dataset_only requires --trained_model_in and --compare_with_dataset")
    if args.trained_model_in:
        args.firm1_mode = "RL"
        args.run_experiment = True
        try:
            artifact_path = resolve_model_reference(args.trained_model_in, args.model_registry_dir)
        except FileNotFoundError as exc:
            parser.error(str(exc))
        args.trained_model_in = artifact_path
        try:
            artifact_header = torch.load(artifact_path, map_location="cpu", weights_only=False)
        except TypeError:
            artifact_header = torch.load(artifact_path, map_location="cpu")
        if artifact_header.get("format") != "ride-response-platform-policy":
            parser.error(f"unsupported --trained_model_in artifact: {artifact_path}")
        if int(artifact_header.get("version", 0)) < 21:
            parser.error(
                "--trained_model_in predates the version-23 coherent-tariff, "
                "differential semi-MDP reward, and corrected critic contract; "
                "retrain it with the current staged PPO path"
            )
        artifact_opt_keys = list(artifact_header.get("opt_keys", []))
        artifact_manipulated_count = int(
            artifact_header.get(
                "manipulated_price_coefficient_count",
                len(artifact_opt_keys),
            )
        )
        if (
            not 1
            <= artifact_manipulated_count
            <= len(MANIPULATABLE_PRICE_COEFFICIENT_KEYS)
            or artifact_manipulated_count != len(artifact_opt_keys)
        ):
            parser.error(
                "--trained_model_in has an invalid manipulated "
                "price-coefficient contract"
            )
        if (
            "--manipulated_price_coefficients" in argv_flag_names
            or "--manipulated_coefficient_count" in argv_flag_names
        ) and int(args.manipulated_price_coefficients) != artifact_manipulated_count:
            parser.error(
                "--manipulated_price_coefficients conflicts with the trained "
                f"policy contract ({artifact_manipulated_count})"
            )
        args.manipulated_price_coefficients = artifact_manipulated_count
        artifact_market = str(
            artifact_header.get("market_name", args.market) or args.market
        )
        if "--market" not in argv_flag_names:
            args.market = artifact_market
        elif Core._normalized_market_name(args.market) != Core._normalized_market_name(
            artifact_market
        ):
            parser.error(
                "--market does not match the trained policy artifact: "
                f"requested={args.market!r}, artifact={artifact_market!r}"
            )
        args.state_frame_stack = int(artifact_header.get("state_frame_stack", args.state_frame_stack))
        saved_credit = dict(
            artifact_header.get("credit_assignment_config", {}) or {}
        )
        if not saved_credit:
            saved_credit = {
                "version": 23,
                "privileged_critic": False,
                "persistent_recurrence": False,
                "counterfactual_branches": 0,
                "counterfactual_q_coeff": 0.0,
                "counterfactual_actor_weight": 0.0,
                "potential_shaping_scale": 0.0,
                "hard_kl_stop": False,
                "reward_rate_beta": 0.0,
            }
        saved_version = int(saved_credit.get("version", 23))
        saved_v24 = saved_version >= 24
        saved_v25 = saved_version >= 25
        if "--credit_assignment_v24" in argv_flag_names and not saved_v24:
            parser.error(
                "--credit_assignment_v24 conflicts with the loaded v23 artifact"
            )
        if "--credit_assignment_v25" in argv_flag_names and not saved_v25:
            parser.error(
                "--credit_assignment_v25 conflicts with the loaded pre-v25 artifact"
            )
        args.credit_assignment_v24 = saved_v24
        args.credit_assignment_v25 = saved_v25
        credit_cli_contract = {
            "privileged_critic": (
                bool(saved_credit.get("privileged_critic", False)),
                {"--privileged_critic", "--no-privileged_critic"},
            ),
            "persistent_recurrence": (
                bool(saved_credit.get("persistent_recurrence", False)),
                {"--persistent_recurrence", "--no-persistent_recurrence"},
            ),
            "counterfactual_branches": (
                int(saved_credit.get("counterfactual_branches", 0)),
                {"--counterfactual_branches"},
            ),
            "counterfactual_q_coeff": (
                float(saved_credit.get("counterfactual_q_coeff", 0.0)),
                {"--counterfactual_q_coeff"},
            ),
            "counterfactual_actor_weight": (
                float(saved_credit.get("counterfactual_actor_weight", 0.0)),
                {"--counterfactual_actor_weight"},
            ),
            "potential_shaping_scale": (
                float(saved_credit.get("potential_shaping_scale", 0.0)),
                {"--potential_shaping_scale"},
            ),
            "hard_kl_stop": (
                bool(saved_credit.get("hard_kl_stop", False)),
                {"--hard_kl_stop", "--no-hard_kl_stop"},
            ),
            "reward_rate_beta": (
                float(saved_credit.get("reward_rate_beta", 0.0)),
                {"--reward_rate_beta"},
            ),
        }
        for argument_name, (saved_value, flags) in credit_cli_contract.items():
            current_value = getattr(args, argument_name)
            explicitly_set = bool(flags & argv_flag_names)
            if explicitly_set and current_value is not None:
                values_match = (
                    np.isclose(float(current_value), float(saved_value))
                    if isinstance(saved_value, float)
                    else current_value == saved_value
                )
                if not values_match:
                    parser.error(
                        f"--{argument_name} conflicts with the trained policy "
                        f"credit-assignment contract ({saved_value})"
                    )
            setattr(args, argument_name, saved_value)
        saved_mdp = artifact_header.get("mdp_config", {})
        saved_observation = saved_mdp.get("observation", {})
        saved_reward = saved_mdp.get("positive_reward", {})
        saved_active_reward = saved_mdp.get("active_reward", {})
        saved_constraints = saved_mdp.get("constraints", {})
        saved_curriculum = saved_mdp.get("training_curriculum", {})
        saved_policy_training = artifact_header.get("policy_training_config", {})
        saved_competition_protocol = dict(
            artifact_header.get("competition_protocol", {}) or {}
        )
        saved_steps_per_day = int(
            saved_competition_protocol.get("steps_per_day", 0)
        )
        if saved_steps_per_day <= 0:
            parser.error(
                "--trained_model_in does not record the market steps-per-day calendar"
            )
        if (
            "--train_steps_per_day" in argv_flag_names
            and int(args.train_steps_per_day) != saved_steps_per_day
        ):
            parser.error(
                "--train_steps_per_day conflicts with the trained policy "
                f"calendar ({saved_steps_per_day})"
            )
        args.train_steps_per_day = saved_steps_per_day
        saved_population = dict(
            artifact_header.get("case_study_population", {}) or {}
        )
        if saved_population:
            saved_profile_source = str(
                saved_population.get("threshold_profile_source", "generated")
            )
            if (
                "--threshold_profile_source" in argv_flag_names
                and args.threshold_profile_source != saved_profile_source
            ):
                parser.error(
                    "--threshold_profile_source conflicts with the trained "
                    f"policy population contract ({saved_profile_source})"
                )
            args.threshold_profile_source = saved_profile_source
            saved_pool_size = int(
                saved_population.get("profile_pool_size", args.pool)
            )
            if "--pool" in argv_flag_names and int(args.pool) != saved_pool_size:
                parser.error(
                    "--pool conflicts with the trained policy population "
                    f"contract ({saved_pool_size})"
                )
            args.pool = saved_pool_size
            args.training_generalization_fraction = float(
                saved_population.get(
                    "training_holdout_fraction",
                    args.training_generalization_fraction,
                )
            )
            args.evaluation_generalization_fraction = float(
                saved_population.get(
                    "evaluation_holdout_fraction",
                    args.evaluation_generalization_fraction,
                )
            )
            if saved_profile_source == "cached":
                if "--threshold_cache_path" not in argv_flag_names:
                    args.threshold_cache_path = str(
                        saved_population.get("threshold_cache_path", "") or ""
                    )
                resolved_cache_path = (
                    os.path.abspath(os.path.expanduser(args.threshold_cache_path))
                    if args.threshold_cache_path
                    else ""
                )
                if not resolved_cache_path or not os.path.isfile(
                    resolved_cache_path
                ):
                    parser.error(
                        "the trained policy requires its cached NYC threshold "
                        "population; pass the matching --threshold_cache_path"
                    )
                expected_cache_hash = str(
                    saved_population.get("threshold_cache_sha256", "") or ""
                )
                if (
                    expected_cache_hash
                    and _sha256_file(resolved_cache_path)
                    != expected_cache_hash
                ):
                    parser.error(
                        "--threshold_cache_path does not match the cache "
                        "fingerprint stored in the trained policy"
                    )
                args.threshold_cache_path = resolved_cache_path
        saved_direct_credit_weight = float(
            saved_policy_training.get("direct_actor_credit_weight", 0.0)
        )
        if saved_direct_credit_weight > 0.0:
            print(
                "[WARN] Loaded artifact used actor-only direct credit; the "
                "weight is forced to zero because held-action semi-MDP returns "
                "already provide causal credit."
            )
        args.direct_actor_credit_weight = 0.0
        args.policy_objective = str(
            saved_active_reward.get(
                "objective_mode",
                (
                    "competitiveness"
                    if float(
                        saved_active_reward.get(
                            "price_competitiveness_weight", 0.0
                        )
                    )
                    > (
                        float(
                            saved_active_reward.get(
                                "own_profit_weight", 1.0
                            )
                        )
                        + float(
                            saved_active_reward.get(
                                "profit_advantage_weight", 0.10
                            )
                        )
                    )
                    else "profit_maximization"
                ),
            )
        )
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
        args.market_share_competitiveness_weight = float(
            saved_active_reward.get(
                "market_share_competitiveness_weight",
                (
                    1.0
                    if args.policy_objective == "competitiveness"
                    else args.market_share_competitiveness_weight
                ),
            )
        )
        args.market_share_target_gap = float(
            saved_active_reward.get(
                "market_share_target_gap", args.market_share_target_gap
            )
        )
        args.market_share_gap_scale = float(
            saved_active_reward.get(
                "market_share_gap_scale", args.market_share_gap_scale
            )
        )
        args.market_share_level_weight = float(
            saved_active_reward.get(
                "market_share_level_weight", args.market_share_level_weight
            )
        )
        args.price_competitiveness_weight = float(
            saved_active_reward.get(
                "price_competitiveness_weight", args.price_competitiveness_weight
            )
        )
        args.price_gap_reward_scale = float(
            saved_active_reward.get("price_gap_scale", args.price_gap_reward_scale)
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
        args.target_price_gap = float(
            saved_active_reward.get(
                "price_gap_target",
                saved_constraints.get("target_gap", args.target_price_gap),
            )
        )
        args.price_gap_tolerance = float(saved_constraints.get("overall_tolerance", args.price_gap_tolerance))
        args.segment_gap_tolerances = list(saved_constraints.get("segment_tolerances", args.segment_gap_tolerances))
        args.min_profit_margin = float(saved_constraints.get("margin_floor", args.min_profit_margin))
        args.min_completion_coverage = float(
            saved_constraints.get(
                "completion_coverage_floor", args.min_completion_coverage
            )
        )
        if args.eval_only:
            args.training_curriculum = str(
                saved_curriculum.get("mode", args.training_curriculum)
            )
        if (
            args.eval_only
            and args.seed is None
            and args.experiment_seed is None
        ):
            artifact_training_seed = dict(
                artifact_header.get("metadata", {}) or {}
            ).get("training_seed")
            if artifact_training_seed is not None:
                args.experiment_seed = int(artifact_training_seed)
                args.deterministic_experiment_seed = True
                args.deterministic_torch = True
                print(
                    ">>> Evaluation seed inherited from trained-policy "
                    f"metadata: {args.experiment_seed}."
                )

    # v25's population training uses short, terminal opponent episodes and
    # complete-population on-policy batches by default. This prevents the PPO
    # optimizer from treating one opponent's payoff level as action quality or
    # overwriting the preceding opponent before seeing the others. Explicit
    # CLI choices remain authoritative for controlled ablations.
    if bool(args.credit_assignment_v25) and not bool(args.eval_only):
        if "--ppo_gae_lambda" not in argv_flag_names:
            # Complete terminal opponent episodes make lambda=1 an unbiased
            # Monte-Carlo actor return. The TD(0) critic remains separately
            # trained and diagnostic, but its residual error cannot truncate
            # or bias delayed policy credit.
            args.ppo_gae_lambda = 1.0
        if "--training_dynamic_episode_days" not in argv_flag_names:
            args.training_dynamic_episode_days = 15
        if "--ppo_min_rollout_transitions" not in argv_flag_names:
            resolved_interval = int(max(
                1,
                args.firm1_action_interval_steps
                if int(args.firm1_action_interval_steps) > 0
                else args.firm2_action_interval_days
                if int(args.firm2_action_interval_days) > 0
                else int(np.ceil(args.ppo_update_interval_days / 2.0)),
            ))
            population_modes = tuple(dict.fromkeys((
                str(args.firm2_mode),
                "adaptive_best_response",
                "heuristic",
                "heuristic_random",
                *BENCHMARK_OPPONENT_MODES,
            )))
            fast_episode_days = max(
                1, int(args.training_dynamic_episode_days)
            )
            slow_modes = {"pi_price_gap", "region_supply_demand"}
            expanded_population_modes = tuple(
                mode
                for mode in population_modes
                for _ in range(1 if mode in slow_modes else 3)
            )
            args.ppo_min_rollout_transitions = int(sum(
                int(np.ceil(
                    (
                        max(300, fast_episode_days)
                        if mode in slow_modes
                        else fast_episode_days
                    )
                    * max(1, int(args.train_steps_per_day))
                    / float(resolved_interval)
                ))
                for mode in expanded_population_modes
            ))
        print(
            ">>> v25 population batching: "
            f"episode_days={args.training_dynamic_episode_days}, "
            f"minimum_rollout={args.ppo_min_rollout_transitions}; "
            "terminal opponent regimes use lagged action-independent baselines."
        )
        
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
    args.state_action_mi_weight = 0.0
    args.enable_constrained_reward = False
    args.eval_guardrail_mode = "off"

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
        manipulated_price_coefficients=args.manipulated_price_coefficients,
        total_customers_pool=args.pool,
        deterministic_torch=args.deterministic_torch,
        positive_reward_profit_weight=args.positive_reward_profit_weight,
        positive_reward_revenue_weight=args.positive_reward_revenue_weight,
        positive_reward_completed_demand_weight=args.positive_reward_completed_demand_weight,
        positive_reward_service_weight=args.positive_reward_service_weight,
        long_term_profit_scale=args.long_term_profit_scale,
        long_term_profit_advantage_scale=args.long_term_profit_advantage_scale,
        policy_objective=args.policy_objective,
        long_term_profit_weight=args.long_term_profit_weight,
        profit_dominance_weight=args.profit_dominance_weight,
        dominance_quality_scale=args.dominance_quality_scale,
        market_share_competitiveness_weight=(
            args.market_share_competitiveness_weight
        ),
        market_share_target_gap=args.market_share_target_gap,
        market_share_gap_scale=args.market_share_gap_scale,
        market_share_level_weight=args.market_share_level_weight,
        price_competitiveness_weight=args.price_competitiveness_weight,
        price_gap_reward_scale=args.price_gap_reward_scale,
        intervention_cost_weight=args.intervention_cost_weight,
        reversal_cost_weight=args.reversal_cost_weight,
        ppo_gamma=args.ppo_gamma,
        ppo_gae_lambda=args.ppo_gae_lambda,
        reward_rev_scale=args.reward_rev_scale,
        reward_profit_scale=args.reward_profit_scale,
        min_profit_margin=args.min_profit_margin,
        min_completion_coverage=args.min_completion_coverage,
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
        constrained_reward=bool(args.enable_constrained_reward),
        constraint_lr=args.constraint_lr,
        constraint_penalty_scale=args.constraint_penalty_scale,
        constraint_curriculum_start_scale=args.constraint_curriculum_start_scale,
        constraint_curriculum_mid_scale=args.constraint_curriculum_mid_scale,
        constraint_curriculum_end_scale=args.constraint_curriculum_end_scale,
        target_price_gap=args.target_price_gap,
        price_gap_tolerance=args.price_gap_tolerance,
        segment_gap_tolerances=tuple(args.segment_gap_tolerances),
        telemetry_delay_steps=args.telemetry_delay_steps,
        quote_probe_interval_steps=args.quote_probe_interval_steps,
        quote_probe_delay_steps=args.quote_probe_delay_steps,
        quote_noise_dollars=args.quote_noise_dollars,
        quote_missing_probability=args.quote_missing_probability,
        training_curriculum=args.training_curriculum,
        training_stage_market_fraction=args.training_stage_market_fraction,
        training_stage_static_fraction=args.training_stage_static_fraction,
        training_stage_dynamic_fraction=args.training_stage_dynamic_fraction,
        training_dynamic_episode_days=args.training_dynamic_episode_days,
        training_opening_tariff_randomization=(
            args.training_opening_tariff_randomization
        ),
        ppo_batch_size=args.ppo_batch_size,
        ppo_update_epochs=args.ppo_update_epochs,
        state_action_mi_weight=args.state_action_mi_weight,
        direct_actor_credit_weight=args.direct_actor_credit_weight,
        state_frame_stack=args.state_frame_stack,
        credit_assignment_v24=args.credit_assignment_v24,
        credit_assignment_v25=args.credit_assignment_v25,
        privileged_critic=args.privileged_critic,
        persistent_recurrence=args.persistent_recurrence,
        counterfactual_branches=args.counterfactual_branches,
        counterfactual_q_coeff=args.counterfactual_q_coeff,
        counterfactual_actor_weight=args.counterfactual_actor_weight,
        potential_shaping_scale=args.potential_shaping_scale,
        hard_kl_stop=args.hard_kl_stop,
        reward_rate_beta=args.reward_rate_beta,
        threshold_cache_path=args.threshold_cache_path,
        threshold_profile_source=args.threshold_profile_source,
        threshold_prompt_repeat_source=args.threshold_prompt_repeat_source,
        training_generalization_fraction=args.training_generalization_fraction,
        evaluation_generalization_fraction=args.evaluation_generalization_fraction,
        save_threshold_cache=not args.no_save_threshold_cache,
        gpt_threshold_include_rationales=args.gpt_threshold_include_rationales,
        gpt_threshold_coldstart_rides=args.gpt_threshold_coldstart_rides,
        gpt_threshold_send_raw_rides=args.gpt_threshold_send_raw_rides,
        gpt_threshold_batch_size=args.gpt_threshold_batch_size,
        gpt_threshold_max_retries=args.gpt_threshold_max_retries,
        gpt_threshold_failure_pause=args.gpt_threshold_failure_pause,
    )
    
    if args.calibration_preset:
        if (
            str(args.calibration_preset).strip().lower() == "nyc_public"
            and Core._normalized_market_name(core.market_name)
            not in {"newyorkcity", "nyc"}
        ):
            raise ValueError(
                "the nyc_public calibration preset can only be applied to "
                "the New York City market"
            )
        calibration = load_calibration_preset(args.calibration_preset)
        core.apply_calibration(calibration)
        print(f"[Calibration] Applied preset={args.calibration_preset} for market={args.market}.")
        print(f"[Calibration] {json.dumps(calibration, indent=2)}")

    if args.calibration_json:
        calibration = load_calibration_json(
            args.calibration_json,
            expected_city=args.market,
        )
        core.apply_calibration(calibration)
        print(
            f"[Calibration] Applied preprocessing calibration from "
            f"{os.path.abspath(os.path.expanduser(args.calibration_json))}."
        )
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


    if args.dataset_only:
        print(">>> Dataset-only mode: skipping synthetic training/evaluation.")
        rows = []
    elif args.run_experiment:
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
            eval_policy_seed=args.eval_policy_seed,
            eval_guardrail_mode=args.eval_guardrail_mode,
            checkpoint_validation_horizon=args.checkpoint_validation_horizon,
            checkpoint_validation_interval_days=(
                args.checkpoint_validation_interval_days
            ),
            checkpoint_validation_customers_per_step=(
                args.checkpoint_validation_customers
            ),
            checkpoint_validation_openings=(
                args.checkpoint_validation_openings
            ),
            evaluate_benchmark_suite=bool(
                args.benchmark_suite or args.require_competitive_backtest
            ),
            training_enabled=not args.eval_only,
            trained_model_out=args.trained_model_out or None,
            model_registry_dir=args.model_registry_dir or None,
            trained_model_id=args.trained_model_id or default_model_id(args.report_prefix),
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
        benchmark_report=getattr(core, "benchmark_evaluation_report", None),
        diagnostic_plot_stride=args.diagnostic_plot_stride,
        fare_plot_stride_days=args.fare_plot_stride_days,
    )
    _write_run_config(args.report_prefix, args, core)
    
    if args.compare_with_dataset:
        summary = core.compare_trained_rl_to_dataset(
            dataset_root=(
                os.path.abspath(os.path.expanduser(args.dataset_root))
                if args.dataset_root
                else _resolve_dataset_path(args.dataset_city or args.market)
            ),
            dataset_glob=args.dataset_glob,
            out_csv=args.comparison_out,
            out_plot_prefix=args.comparison_plot_prefix,
            max_rows=args.comparison_limit,
            preview_rows=args.dataset_preview_rows,
            policy_mode=args.comparison_policy_mode,
            policy_temperature=args.comparison_policy_temperature,
            policy_seed=args.comparison_policy_seed,
            dataset_sample_seed=args.comparison_dataset_seed,
            duration_mode=args.comparison_duration_mode,
            dataset_city=args.dataset_city or args.market,
            weather_root=(
                os.path.abspath(os.path.expanduser(args.weather_root))
                if args.weather_root
                else ""
            ),
        )
        print(f"[RL vs Actual] {json.dumps(summary, indent=2)}")

    if args.require_competitive_backtest:
        audit_failures: List[str] = []
        validation = getattr(core, "benchmark_evaluation_report", None)
        if not validation:
            audit_failures.append("validation report unavailable")
        backtest = getattr(core, "competitive_backtest_report", None)
        if not backtest or not bool(backtest.get("passed", False)):
            failures = [] if not backtest else list(backtest.get("failures", []))
            audit_failures.append(
                "competitive durability backtest failed: "
                + (", ".join(failures) if failures else "report unavailable")
            )
        if audit_failures:
            print(
                "[DIAGNOSTIC AUDIT FAILED] "
                + " | ".join(audit_failures)
                + ". This diagnostic does not alter reward, checkpoints, "
                "model saving, registration, or deployed actions."
            )

if __name__ == "__main__":
    main()
