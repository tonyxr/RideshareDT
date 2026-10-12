"""Daily two-timescale continuous-control adapter for the v27/v28 market.

One macro action is selected per operating day:

``(base tariff multiplier, intraday surge gain, driver incentive multiplier)``.

The first and third controls are positive, uncapped physical values emitted by
the recurrent actor in log space.  During the six intraday blocks, an
interpretable supply-pressure rule turns the daily surge gain into a temporary
tariff response.  This preserves an operational fast loop while the RL policy
learns only the slower control problem.

The adapter deliberately uses public/noisy information only.  Latent shock
labels and simulator parameters are returned in evaluation ``info`` records,
but never enter the observation.  Conditional shock sampling is consumed
automatically when the installed Core exposes it; this module has no hard
dependency on that experimental implementation.
"""

from __future__ import annotations

import contextlib
import copy
import datetime as dt
import hashlib
import io
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from calibration_utils import load_calibration_json
from coeff_utils import set_coeff
from market_dynamics import MarketDynamicsConfig
from market_engine import EngineConfig
from market_shocks import PUBLIC_SHOCK_FEATURE_KEYS, ShockConfig, ShockProcess, ShockScenario


CITY_NAMES = {
    "nyc": "New York City",
    "boston": "Boston",
    "chicago": "Chicago",
}
WEATHER_CATEGORIES = ("clear", "cloudy", "rain", "snow", "storm")


def file_provenance(path: str | Path) -> Dict[str, Any]:
    source = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = source.stat()
    return {
        "path": str(source),
        "bytes": int(stat.st_size),
        "sha256": digest.hexdigest(),
    }


def _market_data_limitations(city: str, calibration: Mapping[str, Any]) -> List[str]:
    limitations: List[str] = []
    market = dict(calibration.get("market", {}) or {})
    trip_bank = dict(calibration.get("trip_bank", {}) or {})
    if not bool(trip_bank.get("airport_identified", market.get("airport_prob") is not None)):
        limitations.append("airport labels unavailable; conditional airport rows are synthetic fallback")
    if str(market.get("ride_semantics", "realized_trip")) != "realized_trip":
        limitations.append(
            f"source semantics are {market.get('ride_semantics')!r}, not realized TNC trips"
        )
    if city == "boston":
        limitations.append("Boston is partially pooled/OOD and not independently shock-calibrated")
    if not calibration.get("shock_calibration"):
        limitations.append("shock magnitudes use documented priors, not city event estimates")
    return limitations


def _raw_threshold_cache_contract(
    path: Path, *, loaded_count: int
) -> Dict[str, Any]:
    """Describe how the cached thresholds were originally generated.

    Loading a legacy cache in v28 marks the in-memory rows as unbounded and
    recovers ``ThresholdRawLLMValue`` when available.  That conversion must
    not be reported as if the source cache itself had been generated under an
    uncapped prompt/validation contract.
    """
    rows = 0
    native_unbounded = 0
    native_contract_rows = 0
    llm_numeric_rows = 0
    uncapped_prompt_rows = 0
    raw_matches_stored_rows = 0
    unclipped_rows = 0
    model_provenance_rows = 0
    raw_llm_values = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if rows >= int(loaded_count):
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows += 1
            native_unbounded += int(
                bool(row.get("ExperimentalUnboundedEconomics", False))
            )
            llm_numeric = row.get("PriceThresholdSource") == "llm_numeric"
            llm_numeric_rows += int(llm_numeric)
            uncapped_prompt = (
                row.get("ThresholdPromptProtocol")
                == "numeric_profile_threshold_v12_uncapped"
            )
            uncapped_prompt_rows += int(uncapped_prompt)
            unclipped = row.get("ThresholdLLMClipped") is False
            unclipped_rows += int(unclipped)
            prompt_model = str(row.get("ThresholdPromptModel", "")).strip()
            response_model = str(row.get("ThresholdResponseModel", "")).strip()
            model_provenance = bool(prompt_model and response_model)
            model_provenance_rows += int(model_provenance)
            raw = row.get("ThresholdRawLLMValue")
            stored = row.get("PriceThreshold")
            raw_valid = False
            raw_matches_stored = False
            try:
                raw_value = float(raw)
                stored_value = float(stored)
                raw_valid = bool(
                    math.isfinite(raw_value)
                    and raw_value > 0.0
                )
                raw_matches_stored = bool(
                    raw_valid
                    and math.isfinite(stored_value)
                    and stored_value > 0.0
                    and raw_value == stored_value
                )
            except (TypeError, ValueError):
                pass
            raw_llm_values += int(raw_valid)
            raw_matches_stored_rows += int(raw_matches_stored)
            native_contract_rows += int(
                bool(row.get("ExperimentalUnboundedEconomics", False))
                and row.get("ThresholdEconomicContract")
                == "v28_unbounded_finite_positive"
                and llm_numeric
                and uncapped_prompt
                and raw_matches_stored
                and unclipped
                and model_provenance
            )
    # A load-time v28 conversion, or even a direct Core run whose API calls all
    # fell back, may carry ExperimentalUnboundedEconomics=True.  Only the
    # dedicated fail-closed Batch artifact has every provenance invariant.
    native = bool(rows > 0 and native_contract_rows == rows)
    return {
        "source_rows_inspected": int(rows),
        "native_unbounded_rows": int(native_unbounded),
        "native_contract_rows": int(native_contract_rows),
        "llm_numeric_rows": int(llm_numeric_rows),
        "uncapped_prompt_rows": int(uncapped_prompt_rows),
        "raw_matches_stored_rows": int(raw_matches_stored_rows),
        "unclipped_rows": int(unclipped_rows),
        "model_provenance_rows": int(model_provenance_rows),
        "raw_llm_value_rows": int(raw_llm_values),
        "native_uncapped_generation": native,
        "source_contract": (
            "native_uncapped_v28"
            if native
            else "legacy_bounded_cache_with_v28_load_time_recovery"
        ),
    }


def _three_way_profile_split(
    profiles: Sequence[Mapping[str, Any]], *, seed: int
) -> Dict[str, List[Dict[str, Any]]]:
    """Deterministic 60/20/20 rider split for train/selection/final test."""
    if len(profiles) < 5:
        raise ValueError("v28 learner requires at least five cached rider profiles")

    def identity(row: Mapping[str, Any]) -> str:
        return json.dumps(dict(row), sort_keys=True, separators=(",", ":"), default=str)

    ranked = sorted(
        (dict(row) for row in profiles),
        key=lambda row: hashlib.sha256(
            f"{int(seed)}|{identity(row)}".encode("utf-8")
        ).hexdigest(),
    )
    count = len(ranked)
    test_count = max(1, int(round(0.20 * count)))
    validation_count = max(1, int(round(0.20 * count)))
    if test_count + validation_count >= count:
        test_count = 1
        validation_count = 1
    test = ranked[:test_count]
    validation = ranked[test_count:test_count + validation_count]
    train = ranked[test_count + validation_count:]
    return {"train": train, "validation": validation, "test": test}


def build_v28_core(
    *,
    city: str,
    calibration_path: str | Path,
    trip_bank_path: str | Path,
    threshold_cache_path: str | Path,
    seed: int,
    profile_pool_size: int,
    simulation_sample_cap: int,
    shock_config: ShockConfig,
    steps_per_day: int = 6,
    target_utilization: float = 0.60,
    rider_memory: bool = True,
    firm2_mode: str = "static",
) -> Tuple[Any, Dict[str, Any]]:
    """Construct a v28-ready Core and load the requested cached population."""
    import Core as core_module

    city_key = str(city).lower()
    if city_key not in CITY_NAMES:
        raise ValueError(f"unsupported city {city!r}")
    calibration_path = Path(calibration_path).expanduser().resolve()
    trip_bank_path = Path(trip_bank_path).expanduser().resolve()
    threshold_cache_path = Path(threshold_cache_path).expanduser().resolve()
    for required in (calibration_path, trip_bank_path, threshold_cache_path):
        if not required.exists():
            raise FileNotFoundError(required)
    calibration = load_calibration_json(
        str(calibration_path), expected_city=CITY_NAMES[city_key]
    )
    calibration["_source_dir"] = str(calibration_path.parent)
    # Core is verbose during calibration.  The runner prints an auditable
    # compact provenance record instead of interleaving thousands of lines.
    with contextlib.redirect_stdout(io.StringIO()):
        core = core_module.Core(
            market_name=CITY_NAMES[city_key],
            seed=int(seed),
            firm1_mode="static",
            firm2_mode=str(firm2_mode),
            total_customers_pool=int(profile_pool_size),
            simulation_sample_cap=int(simulation_sample_cap),
            openai_api_key="",
            threshold_cache_path=str(threshold_cache_path),
            threshold_profile_source="cached",
            shock_observations=True,
            experimental_unbounded_economics=True,
        )
        core.apply_calibration(calibration)
        core.configure_v26(
            market_dynamics=MarketDynamicsConfig(
                enabled=True,
                market_relative_bounds=False,
                capacity_target_fulfillment=0.0,
                rider_memory=bool(rider_memory),
            ),
            steps_per_day=int(steps_per_day),
            trip_bank_path=str(trip_bank_path),
            environment_reseed_per_episode=True,
        )
        core.configure_v27(
            engine_config=EngineConfig(
                enabled=True,
                target_utilization=float(target_utilization),
            ),
            shock_config=shock_config,
        )
        # This explicitly loads the requested GPT/profile cache now rather
        # than allowing a later training step to substitute generated riders.
        core._sample_profiles_from_pool(1)
    if not bool(getattr(core, "experimental_unbounded_economics", False)):
        raise RuntimeError("v28 Core was not constructed in unbounded-economics mode")
    unmarked = [
        index
        for index, profile in enumerate(core.synthetic_profile_pool)
        if not bool(profile.get("ExperimentalUnboundedEconomics", False))
    ]
    if unmarked:
        raise RuntimeError(
            "cached profile loader did not mark all rows as unbounded; "
            f"first unmarked indices={unmarked[:5]}"
        )
    thresholds = np.asarray(
        [float(profile["PriceThreshold"]) for profile in core.synthetic_profile_pool],
        dtype=np.float64,
    )
    profile_splits = _three_way_profile_split(
        core.synthetic_profile_pool, seed=590_028
    )
    core.v28_profile_splits = profile_splits
    source_threshold_contract = _raw_threshold_cache_contract(
        threshold_cache_path, loaded_count=len(core.synthetic_profile_pool)
    )
    limitations = _market_data_limitations(city_key, calibration)
    if not source_threshold_contract["native_uncapped_generation"]:
        limitations.append(
            "threshold cache was generated under the legacy bounded contract; "
            "v28 recovers raw LLM values where recorded, so this is not a "
            "native uncapped-threshold dataset"
        )
    provenance = {
        "city": city_key,
        "market_name": CITY_NAMES[city_key],
        "calibration": file_provenance(calibration_path),
        "trip_bank": file_provenance(trip_bank_path),
        "threshold_cache": file_provenance(threshold_cache_path),
        "loaded_profile_count": int(len(core.synthetic_profile_pool)),
        "profile_split_counts": {
            name: int(len(rows)) for name, rows in profile_splits.items()
        },
        "profile_split_contract": "deterministic_disjoint_60_20_20",
        "trip_bank_summary": core.trip_bank.summary() if core.trip_bank is not None else {},
        "shock_calibration_source": str(shock_config.calibration_source),
        "conditional_sampling_requested": bool(
            getattr(shock_config, "conditional_sampling", False)
        ),
        "experimental_unbounded_economics": True,
        "in_memory_unbounded_mode_rows": int(len(core.synthetic_profile_pool)),
        "threshold_recovered_from_raw_count": int(sum(
            bool(profile.get("ThresholdRecoveredFromRawLLMValue", False))
            for profile in core.synthetic_profile_pool
        )),
        "price_threshold_min": float(thresholds.min()),
        "price_threshold_max": float(thresholds.max()),
        "threshold_cache_generation": source_threshold_contract,
        "limitations": limitations,
        "firm2_mode": str(firm2_mode),
    }
    return core, provenance


@dataclass
class DailyEnvironmentConfig:
    episode_days: int = 60
    customers_per_block: int = 100
    steps_per_day: int = 6
    reward_profit_scale: float = 1.0
    surge_pressure_utilization: float = 0.60

    def __post_init__(self) -> None:
        if self.episode_days <= 0 or self.customers_per_block <= 0 or self.steps_per_day <= 0:
            raise ValueError("episode days, customers, and steps per day must be positive")
        if self.reward_profit_scale <= 0.0:
            raise ValueError("reward_profit_scale must be positive")


class ContinuousMarketEnv:
    """Stateful daily environment built on a calibrated v27/v28 Core."""

    def __init__(
        self,
        core_template: Any,
        config: Optional[DailyEnvironmentConfig] = None,
        scenario_factory: Optional[Callable[[int], ShockScenario]] = None,
        profile_split: str = "train",
        shock_sampling_mode: str = "natural",
    ) -> None:
        self._core_template = core_template
        self.config = config or DailyEnvironmentConfig()
        self.scenario_factory = scenario_factory
        self.profile_split = str(profile_split)
        self.shock_sampling_mode = str(shock_sampling_mode)
        if self.profile_split not in {"train", "validation", "test"}:
            raise ValueError("profile_split must be train, validation, or test")
        if self.shock_sampling_mode not in {"natural", "training_openings"}:
            raise ValueError(
                "shock_sampling_mode must be 'natural' or 'training_openings'"
            )
        available_splits = getattr(core_template, "v28_profile_splits", {})
        if self.profile_split not in available_splits:
            raise ValueError(
                f"Core has no explicit v28 {self.profile_split!r} profile split"
            )
        self.core: Any = None
        self.day = 0
        self.done = True
        self._day_context: Any = None
        self._day_hours: List[int] = []
        self._day_public: Dict[str, float] = {}
        self._day_weather = "clear"
        self._day_date = dt.date(2023, 1, 9)
        self._current_public_quote_gap = 0.0
        self._last_firm2_metrics: Any = None
        self._last_summary = self._neutral_summary()
        self._last_action = np.array([1.0, 0.0, 1.0], dtype=np.float64)
        self._build_observation_schema()

    def _build_observation_schema(self) -> None:
        names = [
            "calendar_sin",
            "calendar_cos",
            "weekday_sin",
            "weekday_cos",
            "time_to_go",
            "current_public_quote_gap",
        ]
        names.extend(f"weather_{weather}" for weather in WEATHER_CATEGORIES)
        names.extend(f"public_{name}" for name in PUBLIC_SHOCK_FEATURE_KEYS)
        names.extend(
            (
                "previous_profit_firm1",
                "previous_profit_firm2",
                "previous_completed_share_firm1",
                "previous_completed_share_firm2",
                "previous_fulfillment_firm1",
                "previous_fulfillment_firm2",
                "previous_wait_firm1",
                "previous_wait_firm2",
                "previous_price_gap",
                "previous_airport_share",
                "previous_mean_distance",
                "previous_request_ratio",
                "previous_utilization_firm1",
                "previous_utilization_firm2",
                "previous_served_fraction_firm1",
                "previous_served_fraction_firm2",
                "previous_online_drivers_firm1",
                "previous_online_drivers_firm2",
                "previous_earnings_firm1",
                "previous_earnings_firm2",
                "previous_multihoming_firm1_share",
                "posted_base_multiplier",
                "posted_surge_gain",
                "posted_driver_incentive_multiplier",
            )
        )
        self.observation_names = tuple(names)
        self.observation_index = {name: index for index, name in enumerate(names)}
        self.observation_dim = len(names)

    @staticmethod
    def _neutral_summary() -> Dict[str, float]:
        return {
            "profit_firm1": 0.0,
            "profit_firm2": 0.0,
            "completed_share_firm1": 0.0,
            "completed_share_firm2": 0.0,
            "fulfillment_firm1": 1.0,
            "fulfillment_firm2": 1.0,
            "wait_firm1": 0.0,
            "wait_firm2": 0.0,
            "price_gap": 0.0,
            "airport_share": 0.0,
            "mean_distance": 0.0,
            "request_ratio": 1.0,
            "utilization_firm1": 0.0,
            "utilization_firm2": 0.0,
            "served_fraction_firm1": 1.0,
            "served_fraction_firm2": 1.0,
            "online_drivers_firm1": 0.0,
            "online_drivers_firm2": 0.0,
            "earnings_firm1": 0.0,
            "earnings_firm2": 0.0,
            "multihoming_firm1_share": 0.5,
        }

    def _fresh_core(self) -> Any:
        # Core contains RNG and persistent fleet/reputation state, so a shallow
        # copy would violate CRN and episode independence.
        return copy.deepcopy(self._core_template)

    def reset(self, seed: int) -> np.ndarray:
        self.core = self._fresh_core()
        # The market resets its ShockProcess inside _reseed_environment, so
        # this mode must be installed first.  Setting it afterwards silently
        # made training-opening oversampling a no-op.
        self.core.market.shock_training_openings = bool(
            self.shock_sampling_mode == "training_openings"
        )
        self.core._reseed_environment(int(seed))
        if self.scenario_factory is not None:
            scenario = self.scenario_factory(int(seed))
            process = ShockProcess(
                self.core.shock_config,
                seed=int(seed) + 31,
                steps_per_day=int(self.config.steps_per_day),
            )
            process.scenario = scenario
            self.core.market.install_shocks(process)
        self.core.firm1.overrides = type(self.core.firm1.overrides)()
        self.core.firm2.overrides = type(self.core.firm2.overrides)()
        self.core.firm1.supply_incentive_multiplier = 1.0
        self.core.firm2.supply_incentive_multiplier = 1.0
        fleet = getattr(self.core.driver_supply, "v27_fleet", None)
        if fleet is not None:
            fleet.reset()
        reputation = getattr(self.core.choice_model, "reputation", None)
        if reputation is not None:
            reputation.reset()
        self.core.last_v27_step = {}
        self.day = 0
        self.done = False
        self._last_summary = self._neutral_summary()
        self._last_action = np.array([1.0, 0.0, 1.0], dtype=np.float64)
        self._last_firm2_metrics = None
        self._prepare_day()
        return self._observation()

    def _prepare_day(self) -> None:
        self._day_context = self.core.market.sample_day_context()
        self._day_weather = str(self._day_context.weather)
        self._day_hours = [
            int(self.core.market.sample_timestep_hour().hour)
            for _ in range(int(self.config.steps_per_day))
        ]
        shocks = self.core.market.active_shocks
        self._day_date = (
            shocks.current_date
            if shocks is not None
            else dt.date(2023, 1, 9) + dt.timedelta(days=int(self.day))
        )
        if str(getattr(self.core, "firm2_mode", "static")) != "static":
            self.core._firm2_leader_step(
                previous_metrics=self._last_firm2_metrics,
                day_of_week=int(self._day_context.day_of_week),
                hour=int(self._day_hours[0]),
                weather=str(self._day_weather),
                enabled=True,
            )
        quote_gaps = self.core._public_firm2_quote_gaps(
            day_of_week=int(self._day_context.day_of_week),
            hour=int(self._day_hours[0]),
            weather=str(self._day_weather),
        )
        self._current_public_quote_gap = float(np.mean(list(quote_gaps.values())))
        public: Dict[str, float] = {key: 0.0 for key in PUBLIC_SHOCK_FEATURE_KEYS}
        public["transit_availability"] = 1.0
        # The macro action is posted at the opening of the first block.  Only
        # information public at that instant is admissible.  Aggregating over
        # all six future blocks would reveal an unannounced afternoon outage
        # or event to the morning decision.
        opening_effect = self.core.market.shock_effect(self._day_hours[0])
        for key in PUBLIC_SHOCK_FEATURE_KEYS:
            public[key] = float(opening_effect.public.get(key, public[key]))
        self._day_public = public

    def _observation(self) -> np.ndarray:
        values: Dict[str, float] = {}
        day_of_year = int(self._day_date.timetuple().tm_yday) - 1
        day_angle = 2.0 * math.pi * float(day_of_year) / 365.0
        weekday = int(self._day_context.day_of_week)
        weekday_angle = 2.0 * math.pi * float(weekday) / 7.0
        values.update(
            calendar_sin=math.sin(day_angle),
            calendar_cos=math.cos(day_angle),
            weekday_sin=math.sin(weekday_angle),
            weekday_cos=math.cos(weekday_angle),
            time_to_go=max(
                0.0,
                float(self.config.episode_days - self.day)
                / float(self.config.episode_days),
            ),
            current_public_quote_gap=float(self._current_public_quote_gap),
        )
        for weather in WEATHER_CATEGORIES:
            values[f"weather_{weather}"] = float(self._day_weather.lower() == weather)
        for key in PUBLIC_SHOCK_FEATURE_KEYS:
            values[f"public_{key}"] = float(self._day_public.get(key, 0.0))
        for key, value in self._last_summary.items():
            values[f"previous_{key}"] = float(value)
        values["posted_base_multiplier"] = float(self._last_action[0])
        values["posted_surge_gain"] = float(self._last_action[1])
        values["posted_driver_incentive_multiplier"] = float(self._last_action[2])
        observation = np.asarray(
            [values.get(name, 0.0) for name in self.observation_names], dtype=np.float32
        )
        if not np.all(np.isfinite(observation)):
            raise FloatingPointError("market observation contains a non-finite value")
        return observation

    def public_event_indicator(self, observation: np.ndarray) -> bool:
        obs = np.asarray(observation, dtype=np.float64)
        warning = max(
            obs[self.observation_index.get("public_storm_warning", 0)],
            obs[self.observation_index.get("public_storm_active", 0)],
            obs[self.observation_index.get("public_holiday_active", 0)],
            obs[self.observation_index.get("public_event_announced", 0)],
        )
        transit_index = self.observation_index.get("public_transit_availability")
        disrupted = transit_index is not None and obs[transit_index] < 0.999
        return bool(warning > 0.0 or disrupted)

    def _supply_pressure(self) -> float:
        fleet = dict((self.core.last_v27_step or {}).get("fleet", {}).get("Firm1", {}) or {})
        if not fleet:
            return 0.0
        served = float(fleet.get("served_fraction", 1.0))
        utilization = float(fleet.get("utilization", self.config.surge_pressure_utilization))
        target = float(self.config.surge_pressure_utilization)
        return float((1.0 - served) + utilization - target)

    def _set_tariff_multiplier(self, multiplier: float) -> None:
        if not np.isfinite(multiplier) or multiplier <= 0.0:
            raise FloatingPointError("tariff multiplier must be finite and positive")
        base = self.core.market.curr_market
        for key in self.core.shared_edit_keys:
            value = float(multiplier) * float(getattr(base, key))
            if not np.isfinite(value):
                raise FloatingPointError(f"tariff coefficient {key!r} overflowed")
            set_coeff(base, self.core.firm1.overrides, key, value)

    def _sample_profiles(self, count: int) -> List[Dict[str, Any]]:
        pool = self.core.v28_profile_splits[self.profile_split]
        if not pool:
            raise RuntimeError(f"empty v28 profile split {self.profile_split!r}")
        indices = self.core.rng.integers(0, len(pool), size=int(count))
        return [pool[int(index)] for index in indices]

    def step(self, action: np.ndarray | Sequence[float]) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if self.done:
            raise RuntimeError("step called after episode termination")
        action_array = np.asarray(action, dtype=np.float64).reshape(-1)
        if action_array.shape != (3,):
            raise ValueError("daily action must be (base multiplier, surge gain, driver incentive)")
        if not np.all(np.isfinite(action_array)):
            raise FloatingPointError("daily action contains a non-finite value")
        base_multiplier, surge_gain, driver_incentive = (float(value) for value in action_array)
        if base_multiplier <= 0.0 or driver_incentive <= 0.0:
            raise ValueError("base multiplier and driver incentive must be positive")
        self.core.firm1.supply_incentive_multiplier = driver_incentive

        sums = self._neutral_summary()
        sums.update({key: 0.0 for key in sums})
        active_shocks: set[str] = set()
        shock_phases: set[str] = set()
        block_multipliers: List[float] = []
        conditional_sampling: List[Dict[str, Any]] = []
        daily_welfare: Dict[str, Any] = {}
        for hour in self._day_hours:
            pressure = self._supply_pressure()
            with np.errstate(over="ignore", invalid="ignore"):
                block_multiplier = base_multiplier * float(np.exp(surge_gain * pressure))
            self._set_tariff_multiplier(block_multiplier)
            block_multipliers.append(block_multiplier)
            effect = self.core.market.shock_effect(hour)
            active_shocks.update(str(name) for name in effect.active)
            phase = str(getattr(effect, "phase", "inactive"))
            if phase != "inactive":
                shock_phases.add(phase)
            requests = self.core._step_customers(
                int(self.config.customers_per_block),
                hour=int(hour),
                day_of_week=int(self._day_context.day_of_week),
                weather=str(self._day_context.weather),
            )
            profiles = self._sample_profiles(
                self.core._effective_simulation_sample_size(requests)
            )
            _, firm1, firm2, gap, airport_share, mean_distance = self.core.simulate_batch(
                int(self._day_context.day_of_week),
                str(self._day_context.weather),
                int(hour),
                int(requests),
                profiles,
                False,
                int(self.config.customers_per_block),
            )
            # Observational accounting only: reward remains Firm 1 profit.
            self.core._accumulate_welfare(
                daily_welfare,
                getattr(self.core, "last_welfare_stats", None),
            )
            fleet = dict((self.core.last_v27_step or {}).get("fleet", {}) or {})
            fleet1 = dict(fleet.get("Firm1", {}) or {})
            fleet2 = dict(fleet.get("Firm2", {}) or {})
            values = {
                "profit_firm1": float(firm1.profit_per_request),
                "profit_firm2": float(firm2.profit_per_request),
                "completed_share_firm1": float(firm1.completed_share),
                "completed_share_firm2": float(firm2.completed_share),
                "fulfillment_firm1": float(firm1.fulfillment_rate),
                "fulfillment_firm2": float(firm2.fulfillment_rate),
                "wait_firm1": float(firm1.avg_wait_minutes),
                "wait_firm2": float(firm2.avg_wait_minutes),
                "price_gap": float(gap),
                "airport_share": float(airport_share),
                "mean_distance": float(mean_distance),
                "request_ratio": float(requests) / float(self.config.customers_per_block),
                "utilization_firm1": float(fleet1.get("utilization", 0.0)),
                "utilization_firm2": float(fleet2.get("utilization", 0.0)),
                "served_fraction_firm1": float(fleet1.get("served_fraction", 1.0)),
                "served_fraction_firm2": float(fleet2.get("served_fraction", 1.0)),
                "online_drivers_firm1": float(fleet1.get("online_drivers", 0.0)),
                "online_drivers_firm2": float(fleet2.get("online_drivers", 0.0)),
                "earnings_firm1": float(fleet1.get("earnings_per_online_hour", 0.0)),
                "earnings_firm2": float(fleet2.get("earnings_per_online_hour", 0.0)),
                "multihoming_firm1_share": float(
                    (self.core.last_v27_step or {}).get("multihoming_firm1_time_share", 0.5)
                ),
            }
            for key, value in values.items():
                sums[key] += value
            self._last_firm2_metrics = firm2
            diagnostics = (self.core.last_v27_step or {}).get(
                "conditional_sampling", (self.core.last_v27_step or {}).get("shock_sampling")
            )
            if isinstance(diagnostics, Mapping):
                conditional_sampling.append(dict(diagnostics))

        blocks = float(self.config.steps_per_day)
        self._last_summary = {key: float(value) / blocks for key, value in sums.items()}
        reward = (
            float(sums["profit_firm1"]) / float(self.config.reward_profit_scale)
        )
        self._last_action = action_array.copy()
        self.day += 1
        self.done = self.day >= int(self.config.episode_days)
        info: Dict[str, Any] = {
            "day": int(self.day - 1),
            "reward": float(reward),
            "base_multiplier": base_multiplier,
            "surge_gain": surge_gain,
            "driver_incentive_multiplier": driver_incentive,
            "block_tariff_multipliers": block_multipliers,
            "maximum_block_tariff_multiplier": float(max(block_multipliers)),
            "minimum_block_tariff_multiplier": float(min(block_multipliers)),
            "active_shocks": sorted(active_shocks),
            "shock_phases": sorted(shock_phases),
            "public_shock": dict(self._day_public),
            "weather": self._day_weather,
            "date": self._day_date.isoformat(),
            "profile_split": self.profile_split,
            "shock_sampling_mode": self.shock_sampling_mode,
            "summary": dict(self._last_summary),
            "welfare_summary": self.core._summarize_welfare(daily_welfare),
            "conditional_sampling": conditional_sampling,
        }
        if not self.done:
            self._prepare_day()
        next_observation = self._observation()
        return next_observation, float(reward), bool(self.done), info

    def clairvoyant_event_days(self, seed: int) -> Tuple[int, ...]:
        """Return scripted physical-event/recovery days for oracle diagnostics.

        This method is deliberately absent from observations and normal policy
        evaluation.  Only the clearly labelled clairvoyant benchmark calls it.
        """
        if self.scenario_factory is None:
            return ()
        scenario = self.scenario_factory(int(seed))
        shock_config = getattr(self._core_template, "shock_config", None)
        days: Set[int] = set()
        for event in scenario.events:
            recovery = 0
            if event.kind == "driver_shortage" and shock_config is not None:
                recovery = int(max(0, shock_config.shortage_recovery_days))
            elif event.kind == "winter_storm" and shock_config is not None:
                recovery = int(max(0, shock_config.storm_supply_recovery_days))
            stop = min(
                int(self.config.episode_days) - 1,
                int(event.end_day) + recovery,
            )
            days.update(range(max(0, int(event.start_day)), stop + 1))
        return tuple(sorted(days))

    def planning_branch(self, seed: int, reveal_latent: bool = False) -> "ContinuousMarketEnv":
        """Clone exact simulator state for a perfect-state MPC upper bound.

        A correct belief particle must reconstruct latent current-day event
        state from public history rather than copy it.  That filter is not yet
        implemented, so the adapter refuses to label a deep copy as an
        information-matched planner.
        """
        del seed
        if not reveal_latent:
            raise NotImplementedError(
                "belief-state planning branches are not implemented; use "
                "reveal_latent=True only as a perfect-state upper bound"
            )
        return copy.deepcopy(self)

    def contract(self) -> Dict[str, Any]:
        return {
            "observation_names": list(self.observation_names),
            "observation_dim": int(self.observation_dim),
            "action_names": [
                "base_multiplier",
                "surge_gain",
                "driver_incentive_multiplier",
            ],
            "action_support": ["positive_uncapped_log", "unbounded", "positive_uncapped_log"],
            "config": asdict(self.config),
            "profile_split": self.profile_split,
            "shock_sampling_mode": self.shock_sampling_mode,
            "calendar_source": "ShockProcess.current_date when active",
            "finite_horizon_time_to_go": True,
            "latent_state_in_observation": False,
            "experimental_unbounded_economics": bool(
                getattr(self._core_template, "experimental_unbounded_economics", False)
            ),
            "intraday_rule": "base * exp(surge_gain * ((1-served_fraction)+utilization-target))",
        }


__all__ = [
    "CITY_NAMES",
    "ContinuousMarketEnv",
    "DailyEnvironmentConfig",
    "build_v28_core",
    "file_provenance",
]
