#!/usr/bin/env python3
"""Validate V17's single configuration and render its Core command."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


BASE = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = BASE / "configs/v17_production.yaml"
EXPECTED_STAGE_NAMES = (
    "market",
    "static_opponent",
    "price_gap_catchup",
    "profit_maximizers",
)
EXPECTED_STAGE_FRACTIONS = (0.05, 0.05, 0.30, 0.60)
EXPECTED_LEAD_WEIGHTS = (0.0, 0.50, 0.625, 0.75)
CITY_SLUGS = ("nyc", "boston", "chicago")
ACTION_LEVELS = {
    "two_lever_v1": (0.8, 0.9, 1.0, 1.2, 1.5),
    "two_lever_v2": (0.7, 0.8, 0.9, 1.0, 1.2, 1.5),
}


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("V17 configuration must be a YAML mapping")
    return value


def canonical_hash(config: Mapping[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _tuple(values: Any) -> tuple[Any, ...]:
    return tuple(values or ())


def resolve_input(path: str) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (BASE / value).resolve()


def validate_config(
    config: Mapping[str, Any], *, require_frozen: bool = False
) -> None:
    if config.get("schema_version") != "v17-production-v1":
        raise ValueError("unexpected V17 production schema")
    if _tuple(config.get("execution_order")) != (
        "T0", "T1", "T2", "T3_T4", "production", "evaluation", "results"
    ):
        raise ValueError("V17 execution order drifted from the manual")
    market = config["market"]
    expected_market = {
        "supply_model": "v3",
        "eta_v3": 0.6,
        "persistent_driver_stock": False,
        "shortage_willingness_share": 0.5,
        "price_sensitivity_units": "relative",
        "fare_model": "realistic",
        "driver_pay_basis": "fare",
    }
    for key, expected in expected_market.items():
        if market.get(key) != expected:
            raise ValueError(f"V17 market contract drifted for {key}")
    observation = config["observation"]
    if observation != {
        "exact_current_active_drivers": True,
        "scope": "own_platform_only",
        "opponent_active_drivers": False,
    }:
        raise ValueError("active-driver observation must be exact and own-only")
    action = config["action"]
    contract = str(action.get("contract"))
    levels = ACTION_LEVELS.get(contract)
    if (
        levels is None
        or _tuple(action.get("distance_levels")) != levels
        or _tuple(action.get("time_levels")) != levels
    ):
        raise ValueError("V17 two-lever action grid drifted")
    if config["rivals"].get("catchup_pool_size") != 4:
        raise ValueError("V17 catch-up pool must contain four rivals")
    if config["rivals"].get("profit_pool_size") != 8:
        raise ValueError("V17 profit pool must contain eight rivals")
    reward = config["reward"]
    if reward.get("mode") != "v17_progressive":
        raise ValueError("user-approved progressive reward is required")
    if _tuple(reward.get("stage_profit_lead_weights")) != EXPECTED_LEAD_WEIGHTS:
        raise ValueError("progressive profit-lead weights drifted")
    if any(float(value) != 0.0 for value in reward["market_share_shaping"]):
        raise ValueError("V17 progressive reward cannot shape market share")
    curriculum = config["curriculum"]
    if _tuple(curriculum.get("stages")) != EXPECTED_STAGE_NAMES:
        raise ValueError("V17 curriculum stage order drifted")
    if _tuple(curriculum.get("fractions")) != EXPECTED_STAGE_FRACTIONS:
        raise ValueError("V17 curriculum fractions drifted")
    if float(curriculum.get("expected_shortage_day_share")) != 0.35:
        raise ValueError("V17 curriculum must target 35% shortage days")
    learner = config["learner"]
    if (
        learner.get("value_head") != "concave2d"
        or learner.get("policy_mode") != "q"
        or int(learner.get("fit_steps_per_day")) != 50
        or int(learner.get("batch_size")) != 64
        or float(learner.get("polyak")) != 0.99
        or int(learner.get("store_rows")) != 20000
    ):
        raise ValueError("V17 learner contract drifted")
    if _tuple(config["training"].get("seeds")) != (101, 202, 303):
        raise ValueError("V17 training seeds drifted")
    if _tuple(config["training"].get("cities")) != CITY_SLUGS:
        raise ValueError("V17 city order drifted")
    city_inputs = config["city_inputs"]
    if tuple(city_inputs) != CITY_SLUGS:
        raise ValueError("V17 city inputs are incomplete or out of order")
    for slug in CITY_SLUGS:
        inputs = city_inputs[slug]
        fleet_scale = float(inputs.get("fleet_scale", 1.0))
        if not 0.5 <= fleet_scale <= 2.0:
            raise ValueError(f"{slug} fleet_scale must lie in [0.5, 2.0]")
        for key in ("calibration_json", "threshold_cache", "demand_calibration"):
            path = resolve_input(inputs[key])
            if not path.is_file():
                raise FileNotFoundError(path)
        demand = json.loads(
            resolve_input(inputs["demand_calibration"]).read_text(encoding="utf-8")
        )
        if demand.get("price_sensitivity_units") != "relative":
            raise ValueError(f"{slug} demand artifact is not in relative units")
        if float(demand["evidence"]["target_own_elasticity"]) != -1.15:
            raise ValueError(f"{slug} demand target drifted")
    unresolved = (
        config["training"].get("days") is None
        or config["credit"].get("horizon_days") is None
    )
    if require_frozen and (config.get("status") != "frozen" or unresolved):
        raise ValueError("T3 must freeze training days and branch horizon first")


def core_args(
    config: Mapping[str, Any],
    *,
    city: str,
    seed: int,
    output: Path,
    days: int | None = None,
    branch_horizon_days: int | None = None,
) -> list[str]:
    validate_config(config)
    if city not in CITY_SLUGS:
        raise ValueError(f"unknown V17 city: {city}")
    resolved_days = config["training"].get("days") if days is None else days
    horizon = (
        config["credit"].get("horizon_days")
        if branch_horizon_days is None
        else branch_horizon_days
    )
    if resolved_days is None or int(resolved_days) < 4:
        raise ValueError("training days are unresolved; run T3.6 first")
    if horizon not in config["credit"]["horizon_candidates_from_T3"]:
        raise ValueError("branch horizon must be the T3-selected 5 or 7 days")
    inputs = config["city_inputs"][city]
    out = output.resolve()
    return [
        "--market", inputs["market_name"],
        "--choice_mode", "cognitive",
        "--firm1_mode", "RL",
        "--firm2_mode", "adaptive_best_response_memory",
        "--run_experiment",
        "--credit_assignment_v25",
        "--v26",
        "--deterministic_experiment_seed",
        "--deterministic_torch",
        "--learning_corrections",
        "--learner_revision", "4",
        "--training_curriculum", "v17_progressive",
        "--current_active_driver_observation",
        "--seed", str(seed),
        "--experiment_seed", str(seed),
        "--policy_objective", "profit_maximization",
        "--pool", "2000",
        "--threshold_profile_source", "cached",
        "--threshold_cache_path", str(resolve_input(inputs["threshold_cache"])),
        "--calibration_json", str(resolve_input(inputs["calibration_json"])),
        "--demand_calibration_artifact", str(resolve_input(inputs["demand_calibration"])),
        "--gpt_threshold_mode", "llm_numeric",
        "--train_timesteps", str(int(resolved_days)),
        "--train_customers", "50",
        "--train_steps_per_day", "6",
        "--firm1_action_interval_steps", "6",
        "--firm2_action_interval_steps", "6",
        "--ppo_update_interval_days", "7",
        "--ppo_batch_size", "64",
        "--training_dynamic_episode_days", "30",
        "--training_opening_tariff_randomization", "0.15",
        "--profit_planner_sample_cap", "32",
        "--regime_market",
        "--regime_features",
        "--spillover_price_aware",
        "--supply_model", "v3",
        "--driver_pay_basis", "fare",
        "--loyalty_model", "v2",
        "--loyalty_pool_size", "180",
        "--persistence_features",
        "--fare_model", "realistic",
        "--shortage_willingness_share", "0.5",
        "--fleet_scale", str(float(inputs.get("fleet_scale", 1.0))),
        "--price_sensitivity_units", "relative",
        "--action_contract", str(config["action"]["contract"]),
        "--rivals_two_lever",
        "--v16_crn_tape",
        "--v16_branch_mode", "one_day_deviation",
        "--v16_branch_horizon_days", str(int(horizon)),
        "--v16_branch_design", "stencil11",
        "--v16_audit_every", "10",
        "--v16_store_cap", "20000",
        "--v16_value_head", "concave2d",
        "--v16_policy_mode", "q",
        "--v16_tau_start", "0.10",
        "--v16_tau_end", "0.03",
        "--v16_policy_epsilon", "0.05",
        "--report_prefix", str(out / city),
        "--trained_model_out", str(out / "policy.pt"),
        "--trained_model_id", f"v17-{city}-seed{seed}",
        "--model_registry_dir", str(out / "registry"),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--require-frozen", action="store_true")
    parser.add_argument("--print-command", action="store_true")
    parser.add_argument("--city", choices=CITY_SLUGS, default="nyc")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--days", type=int)
    parser.add_argument("--branch-horizon-days", type=int)
    parser.add_argument("--output", type=Path, default=BASE / "runs/v17")
    args = parser.parse_args()
    config = load_config(args.config)
    validate_config(config, require_frozen=args.require_frozen)
    result: dict[str, Any] = {
        "state": config["status"],
        "config_hash": canonical_hash(config),
        "config": str(args.config.resolve()),
    }
    if args.print_command:
        result["command"] = [
            str((BASE.parent / ".venv/bin/python").resolve()),
            "-u",
            str(BASE / "Core.py"),
            *core_args(
                config,
                city=args.city,
                seed=args.seed,
                output=args.output,
                days=args.days,
                branch_horizon_days=args.branch_horizon_days,
            ),
        ]
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
