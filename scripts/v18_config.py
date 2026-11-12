#!/usr/bin/env python3
"""Single source of truth for CRISP-Ride V18 Core commands."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import yaml

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from scripts import v17_config
from scripts import v17_results_eval

DEFAULT_CONFIG = BASE / "configs/v18_production.yaml"
CITY_SLUGS = ("nyc", "boston", "chicago")
HORIZON_CANDIDATES = (5, 7, 10)
V18_MODEL_FLAGS = (
    "v3_daily_stock", "supply_eta_within", "supply_eta_day",
    "stock_adjustment_rate", "stock_floor", "driver_pay_rule",
    "active_driver_feature_reference", "scenario_hazards",
    "v17_credit_source", "v17_branch_noise",
)


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("V18 configuration must be a YAML mapping")
    return value


def canonical_hash(config: Mapping[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def resolve_input(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (BASE / path).resolve()


def validate_config(
    config: Mapping[str, Any], *, require_frozen: bool = False
) -> None:
    if config.get("schema_version") != "v18-production-v1":
        raise ValueError("unexpected V18 production schema")
    if tuple(config.get("execution_order", ())) != (
        "P1_P7", "freeze", "wave_A", "wave_B", "E1_E10", "V1_V7", "results"
    ):
        raise ValueError("V18 execution order drifted")
    market = dict(config["market"])
    expected = {
        "supply_model": "v3", "eta_total": 0.6, "eta_within": 0.2,
        "eta_day": 0.4, "v3_daily_stock": True,
        "persistent_driver_stock": False, "stock_adjustment_rate": 0.05,
        "stock_floor": 0.4, "stock_ceiling": 2.0,
        "driver_pay_rule": "linear",
        "active_driver_feature_reference": "run_capacity",
        "acceptance_shock_wage": "declared", "rate_upper_multiple": 2.5,
    }
    for key, wanted in expected.items():
        if market.get(key) != wanted:
            raise ValueError(f"V18 market contract drifted for {key}")
    population = dict(config["population"])
    if int(population.get("pool", 0)) != 20_000:
        raise ValueError("V18 must load all 20,000 riders")
    if float(population.get("loyalty_ratio", 0.0)) != 3.6:
        raise ValueError("V18 loyalty ratio drifted")
    credit = dict(config["credit"])
    if tuple(credit.get("horizon_candidates", ())) != HORIZON_CANDIDATES:
        raise ValueError("V18 horizon candidates must be 5, 7 and 10")
    if tuple(config["training"]["seeds_by_city"]["nyc"]) != (
        101, 202, 303, 404, 505
    ):
        raise ValueError("V18 NYC seed set drifted")
    if tuple(config["city_inputs"]) != CITY_SLUGS:
        raise ValueError("V18 city inputs are incomplete or out of order")
    for city in CITY_SLUGS:
        inputs = config["city_inputs"][city]
        for key in ("calibration_json", "threshold_cache", "demand_calibration"):
            if not resolve_input(inputs[key]).is_file():
                raise FileNotFoundError(resolve_input(inputs[key]))
    if require_frozen:
        decisions = dict(config.get("decisions") or {})
        required = (
            "N_star", "H_star", "threshold_outcome", "branch_planner",
            "ppo_clock_days", "mc_clock_days", "value_head", "fitted_supply",
        )
        missing = [key for key in required if decisions.get(key) is None]
        if config.get("status") != "frozen" or missing:
            raise ValueError(
                "P1-P7 must freeze V18 decisions first: " + ", ".join(missing)
            )
        if int(decisions["N_star"]) not in (100, 150, 250):
            raise ValueError("N_star is outside the registered set")
        if int(decisions["H_star"]) not in HORIZON_CANDIDATES:
            raise ValueError("H_star is outside the registered set")
        for city in CITY_SLUGS:
            for key in ("pay_rule_json", "supply_json"):
                if not resolve_input(config["city_inputs"][city][key]).is_file():
                    raise FileNotFoundError(
                        resolve_input(config["city_inputs"][city][key])
                    )


def _set_option(values: Sequence[str], flag: str, value: Any) -> list[str]:
    result = list(map(str, values))
    while flag in result:
        index = result.index(flag)
        del result[index:index + 2]
    result.extend((flag, str(value)))
    return result


def _drop_options(values: Sequence[str], flags: Sequence[str]) -> list[str]:
    drop = set(flags)
    result: list[str] = []
    index = 0
    while index < len(values):
        value = str(values[index])
        if value in drop:
            index += 2
        else:
            result.append(value)
            index += 1
    return result


def _ensure_flag(values: Sequence[str], flag: str, enabled: bool = True) -> list[str]:
    result = [item for item in map(str, values) if item != flag]
    if enabled:
        result.append(flag)
    return result


def _supply_values(config: Mapping[str, Any], city: str) -> dict[str, Any]:
    decisions = dict(config.get("decisions") or {})
    fitted = dict(decisions.get("fitted_supply") or {})
    if city in fitted:
        return dict(fitted[city])
    # Preflight commands use the legacy starting point; the frozen production
    # config must replace these values with C3 results.
    return {"fleet_scale": {"nyc": 1.2, "boston": 1.0, "chicago": 1.1}[city],
            "base_pickup_minutes": 3.5}


def _apply_v18_common(
    values: Sequence[str], config: Mapping[str, Any], *, city: str,
    scenario_hazards: str,
) -> list[str]:
    market = config["market"]
    population = config["population"]
    decisions = dict(config.get("decisions") or {})
    n_star = int(decisions.get("N_star") or population.get("customers") or 50)
    supply = _supply_values(config, city)
    values = _ensure_flag(values, "--v3_daily_stock")
    for flag, value in (
        ("--supply_eta_v3", market["eta_total"]),
        ("--supply_eta_within", market["eta_within"]),
        ("--supply_eta_day", market["eta_day"]),
        ("--stock_adjustment_rate", market["stock_adjustment_rate"]),
        ("--stock_floor", market["stock_floor"]),
        ("--driver_pay_rule", market["driver_pay_rule"]),
        ("--pay_rule_json", resolve_input(config["city_inputs"][city]["pay_rule_json"])),
        ("--active_driver_feature_reference", market["active_driver_feature_reference"]),
        ("--scenario_hazards", scenario_hazards),
        ("--pool", population["pool"]),
        ("--train_customers", n_star),
        ("--eval_customers", n_star),
        ("--loyalty_pool_size", round(float(population["loyalty_ratio"]) * n_star)),
        ("--fleet_scale", supply["fleet_scale"]),
        ("--base_pickup_minutes", supply["base_pickup_minutes"]),
    ):
        values = _set_option(values, flag, value)
    return values


def train_args(
    config: Mapping[str, Any], *, city: str, seed: int, arm: str,
    output: Path,
) -> list[str]:
    validate_config(config, require_frozen=True)
    if city not in CITY_SLUGS:
        raise ValueError(f"unknown V18 city {city}")
    decisions = config["decisions"]
    days = int(config["training"]["days"])
    if arm == "ppo_clock":
        days = int(decisions["ppo_clock_days"])
    elif arm == "mc_clock":
        days = int(decisions["mc_clock_days"])
    base = v17_config.core_args(
        v17_config.load_config(), city=city, seed=int(seed), output=Path(output),
        days=days, branch_horizon_days=5,
    )
    values = _apply_v18_common(
        base, config, city=city, scenario_hazards="natural"
    )
    values = _set_option(values, "--v16_branch_horizon_days", decisions["H_star"])
    values = _set_option(values, "--v16_value_head", decisions["value_head"])
    planner = dict(decisions["branch_planner"])
    values = _set_option(values, "--branch_profit_planner_sample_cap", planner["sample_cap"])
    values = _set_option(values, "--branch_profit_planner_horizon_days", planner["horizon_days"])
    values = _set_option(values, "--policy_snapshot_every_days", config["learner"]["policy_snapshot_every_days"])
    values = _set_option(values, "--policy_snapshot_dir", Path(output) / "snapshots")
    values = _set_option(values, "--v17_credit_source", "mc_return" if arm == "mc_clock" else "branches")
    values = _set_option(values, "--v17_branch_noise", "independent" if arm == "nocrn" else "crn")
    if arm in {"ppo", "ppo_clock"}:
        values = _drop_options(values, (
            "--v16_branch_mode", "--v16_branch_horizon_days", "--v16_branch_design",
            "--v16_audit_every", "--v16_store_cap", "--v16_value_head",
            "--v17_move_radius", "--v17_temperature_mode",
            "--v17_tau_rel_start", "--v17_tau_rel_end",
        ))
        values = _set_option(values, "--v16_policy_mode", "ppo")
    if arm == "no_driver_obs":
        values = _ensure_flag(values, "--v17_ablate_driver_observation")
    if arm == "own_profit":
        values = _ensure_flag(values, "--v17_ablate_profit_lead")
    if arm == "halt":
        alternate = 10 if int(decisions["H_star"]) == 5 else 5
        values = _set_option(values, "--v16_branch_horizon_days", alternate)
    if arm == "rubric":
        values = _set_option(values, "--threshold_ablation", "rubric")
    return values


def eval_args(
    config: Mapping[str, Any], *, city: str, case: str,
    specs: Sequence[str], suite: str, output: Path,
    seed: int = 77_311, scenario: Path | None = None,
) -> list[str]:
    validate_config(config, require_frozen=True)
    raw = Path(output) / "raw.json"
    prefix = Path(output) / "diagnostics"
    base_case = case if case in v17_results_eval.CASES else "C0_calm"
    command = v17_results_eval._base_command(
        v17_config.load_config(), city=city, case=base_case, specs=specs,
        raw=raw, report_prefix=prefix,
    )[3:]
    natural = case == "natural_year_2023"
    values = _apply_v18_common(
        command, config, city=city,
        scenario_hazards="natural" if natural else "off",
    )
    values = _set_option(values, "--seed", seed)
    values = _set_option(values, "--experiment_seed", seed)
    values = _set_option(values, "--policy_sweep_seed", seed)
    values = _set_option(values, "--v16_branch_horizon_days", config["decisions"]["H_star"])
    values = _set_option(values, "--v16_value_head", config["decisions"]["value_head"])
    if scenario is not None:
        values = _set_option(values, "--shock_scenario", Path(scenario).resolve())
    if natural:
        values = _set_option(values, "--shock_scenario", BASE / "scenarios/v18/natural_year_2023.json")
        values = _set_option(values, "--policy_sweep_horizon", 364 * 6)
    if suite in {"E5", "E10"}:
        values = _set_option(values, "--policy_sweep_request_log", Path(output) / "requests")
    return values


def model_flags(config: Mapping[str, Any], *, city: str) -> dict[str, Any]:
    market = config["market"]
    return {
        "v3_daily_stock": True,
        "supply_eta_within": float(market["eta_within"]),
        "supply_eta_day": float(market["eta_day"]),
        "stock_adjustment_rate": float(market["stock_adjustment_rate"]),
        "stock_floor": float(market["stock_floor"]),
        "driver_pay_rule": str(market["driver_pay_rule"]),
        "active_driver_feature_reference": str(market["active_driver_feature_reference"]),
        "acceptance_shock_wage": str(market["acceptance_shock_wage"]),
        "rate_upper_multiple": float(market["rate_upper_multiple"]),
    }


def check_model(payload: Mapping[str, Any], config: Mapping[str, Any], *, city: str) -> None:
    got = dict(payload.get("model_switches") or {})
    for key, wanted in model_flags(config, city=city).items():
        actual = got.get(key)
        if isinstance(wanted, float):
            if actual is None or abs(float(actual) - wanted) > 1e-12:
                raise ValueError(f"V18 model mismatch for {key}: {actual} != {wanted}")
        elif actual != wanted:
            raise ValueError(f"V18 model mismatch for {key}: {actual} != {wanted}")


def check_command_model(command: Sequence[str], config: Mapping[str, Any], *, city: str) -> None:
    values = list(map(str, command))
    if values.count("--v3_daily_stock") != 1:
        raise ValueError("V18 command must declare v3_daily_stock exactly once")
    mapping = {
        "supply_eta_within": "--supply_eta_within",
        "supply_eta_day": "--supply_eta_day",
        "stock_adjustment_rate": "--stock_adjustment_rate",
        "stock_floor": "--stock_floor",
        "driver_pay_rule": "--driver_pay_rule",
        "active_driver_feature_reference": "--active_driver_feature_reference",
        "acceptance_shock_wage": "--acceptance_shock_wage",
        "rate_upper_multiple": "--rate_upper_multiple",
    }
    for key, flag in mapping.items():
        if values.count(flag) != 1:
            raise ValueError(f"V18 command must declare {flag} exactly once")
        actual = values[values.index(flag) + 1]
        wanted = model_flags(config, city=city)[key]
        if isinstance(wanted, float):
            if abs(float(actual) - wanted) > 1e-12:
                raise ValueError(f"V18 command mismatch for {flag}")
        elif actual != str(wanted):
            raise ValueError(f"V18 command mismatch for {flag}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--require-frozen", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    validate_config(config, require_frozen=args.require_frozen)
    print(json.dumps({"status": config["status"], "hash": canonical_hash(config)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
