"""Fast code-contract tests for the V18 threat-removal implementation."""

from datetime import date, timedelta
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import torch

from Core import Core, build_shock_config_from_args
from driver_supply import DriverPayPolicy, DriverSupplyConfig, DriverSupplyLayer, FirmDriverBatchState
from market_shocks import ShockConfig, ShockEvent, ShockProcess, ShockScenario
from scripts.v18_config import HORIZON_CANDIDATES, load_config, validate_config
from scripts.v18_fit_pay_rule import DEFAULT_INPUT, fit, load_rows, transfer
from scripts.v18_fleet_refit import FLEET_GRID, select_fit
from scripts.v18_p2 import Job, _histogram_median, _measurement, _run_parallel
from scripts.v18_select import card_grid, count_rule_candidates, reactive_candidates
from scripts.v18_stats import holm, two_level_bootstrap
from scripts.v18_value_of_timing import timing
from v17_branch_store import BranchReplayStore


def _v3(stock: bool) -> DriverSupplyLayer:
    layer = DriverSupplyLayer(seed=8, config=DriverSupplyConfig(
        supply_model="v3", reference_earnings_per_hour=20.0,
        eta_v3=0.6, eta_within=0.2, eta_day=0.4,
        v3_daily_stock=stock, stock_adjustment_rate=0.05,
        hour_profile={hour: 1.0 for hour in range(24)},
    ))
    state = FirmDriverBatchState(
        completed_rides=100, total_driver_pay=1000.0,
        total_online_minutes=3000.0,
    )
    layer.smoothed_states["Firm1"] = state
    layer.smoothed_states["Firm2"] = state
    return layer


def test_v18_config_is_prefreeze_and_registered():
    config = load_config()
    validate_config(config)
    assert config["status"] == "preflight"
    assert tuple(config["credit"]["horizon_candidates"]) == HORIZON_CANDIDATES
    with pytest.raises(ValueError, match="freeze"):
        validate_config(config, require_frozen=True)


def test_v3_flag_off_is_legacy_and_on_splits_elasticity():
    legacy, v18 = _v3(False), _v3(True)
    for layer in (legacy, v18):
        layer.set_pay_index("Firm1", 2.0)
    assert legacy.supply_v3_online_response("Firm1") == pytest.approx(2.0 ** 0.6)
    assert v18.supply_v3_online_response("Firm1") == pytest.approx(2.0 ** 0.2)
    assert v18._v3_eta_total() == pytest.approx(0.6)


def test_v3_daily_stock_uses_realized_pay_formula():
    layer = _v3(True)
    layer._daily_pay.update({"Firm1": 1200.0, "Firm2": 1000.0})
    layer._daily_online.update({"Firm1": 3000.0, "Firm2": 3000.0})
    value = layer.advance_day()["Firm1"]
    target = (24.0 / 20.0) ** 0.4
    assert value == pytest.approx(0.95 + 0.05 * target)


def test_linear_driver_pay_is_hand_computed():
    policy = DriverPayPolicy(
        rule="linear", intercept=-0.32, per_mile=0.567,
        per_minute=0.285, fare_coef=0.407, minimum_pay=3.5,
    )
    layer = DriverSupplyLayer(firm1_pay_policy=policy)
    expected = -0.32 + 0.567 * 3 + 0.285 * 15 + 0.407 * 20
    assert layer.quote_driver_pay("Firm1", 20, 3, 15, 4, False) == pytest.approx(expected)


def test_pay_fit_recovers_fixed_coefficients(tmp_path: Path):
    path = tmp_path / "trips.csv"
    beta = np.asarray([-0.32, 0.567, 0.285, 0.407])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "driver_pay", "distance_miles", "duration_minutes",
            "actual_price", "pickup_date",
        )); writer.writeheader()
        for index in range(14):
            miles, minutes, fare = 1 + index % 5, 8 + 2 * index, 10 + 3 * (index % 4)
            x = np.asarray([1, miles, minutes, fare], dtype=float)
            writer.writerow({"driver_pay": float(x @ beta), "distance_miles": miles,
                             "duration_minutes": minutes, "actual_price": fare,
                             "pickup_date": "2023-06-01" if index < 10 else "2023-07-15"})
    result = fit(load_rows(path))
    assert [result[key] for key in ("intercept", "per_mile", "per_minute", "fare_coef")] == pytest.approx(beta, abs=1e-6)
    assert result["test_r2"] == pytest.approx(1.0)


def test_pay_fit_default_uses_shared_registered_trip_bank():
    assert DEFAULT_INPUT == Path(__file__).resolve().parents[2] / "data/processed/nyc/trip_bank.csv"


def test_run_capacity_feature_is_scale_invariant():
    layer = DriverSupplyLayer(config=DriverSupplyConfig(
        base_active_drivers=50, count_feature_reference="run_capacity"
    ))
    layer._feature_capacity_reference = 5.0
    small = layer.current_active_driver_feature(5.0)
    layer._feature_capacity_reference = 50.0
    large = layer.current_active_driver_feature(50.0)
    assert small == pytest.approx(large, abs=1e-12)


def test_scripted_hazards_off_adds_no_random_events():
    script = ShockEvent("driver_shortage", 4, 6, intensity=0.65)
    for seed in range(10):
        process = ShockProcess(ShockConfig(scenario_hazards="off"), seed=seed)
        process.scenario = ShockScenario(date(2023, 1, 2), events=[script])
        process.reset()
        for _ in range(40):
            process.advance_day()
        assert {event.kind for event in process.events} <= {"driver_shortage"}


def _shock_args(*, scenario_hazards):
    return SimpleNamespace(
        shocks="calibrated",
        shock_intensity=1.0,
        actor_true_shortage_flag=False,
        regime_features=True,
        regime_feature_ablation=False,
        emergency_surge_cap=0.0,
        scenario_hazards=scenario_hazards,
        shock_types="",
        shock_scenario="scenario.json",
        shock_rates_json="",
    )


def test_implicit_scripted_scenario_is_exact_v17_compatibility():
    config = build_shock_config_from_args(
        _shock_args(scenario_hazards=None)
    )
    assert config.scenario_hazards == "natural"
    assert config.storm_daily_hazard == {}
    assert config.event_rate_per_day == 0.0
    assert config.shortage_rate_per_day == 0.0
    assert config.outage_rate_per_day == 0.0
    assert config.transit_disruption_rate_per_day == 0.0
    assert config.rival_promotion_rate_per_day == 0.0
    assert config.training_shock_opening_share == 0.0
    assert config.drift_sd == 0.0


def test_explicit_v18_scenario_modes_do_not_use_legacy_compatibility():
    natural = build_shock_config_from_args(
        _shock_args(scenario_hazards="natural")
    )
    assert natural.scenario_hazards == "natural"
    assert natural.event_rate_per_day > 0.0
    assert natural.drift_sd > 0.0

    scripted = build_shock_config_from_args(
        _shock_args(scenario_hazards="off")
    )
    assert scripted.scenario_hazards == "off"
    assert scripted.event_rate_per_day == 0.0
    assert scripted.drift_sd > 0.0


def test_reference_candidate_counts_are_registered():
    assert len(card_grid()) == 36
    assert len(count_rule_candidates(normal=(1, 1), event_cards=[(1, 1)] * 6, norm=[1] * 6)) == 84
    assert len(reactive_candidates((1, 1))) == 24


def test_mc_return_row_is_absolute_and_discounted():
    core = Core.__new__(Core)
    core.v17_credit_source = "mc_return"
    core.v16_branch_horizon_days = 3
    core._active_steps_per_day = 1
    core._v18_mc_pending = __import__("collections").deque()
    core.v18_mc_dropped_incomplete = 0
    core.v16_branch_store = BranchReplayStore(10, seed=1)
    core.v16_value_auxiliary_names = (
        "own_profit_per_request", "rival_profit_per_request",
        "own_completed_share", "rival_completed_share",
    )
    core.v17_move_radius = 1
    core.firm2_mode = "static"
    core.current_training_stage = SimpleNamespace(opponent_regime="market")
    core._regime_last_effect = SimpleNamespace(active=())
    core.firm1 = SimpleNamespace(agent=SimpleNamespace(gamma=0.9, action_dim=4))
    core._v18_mc_begin(observation_window=torch.tensor([1.0, 2.0]), action=2, day=4)
    for reward in (1.0, 2.0, 3.0):
        core._v18_mc_observe_day(reward=reward, components=[reward] * 4)
    row = core.v16_branch_store.rows()[0]
    assert row["label_kind"] == "absolute"
    assert row["action_credits"][2] == pytest.approx(1 + 0.9 * 2 + 0.9**2 * 3)
    assert sum(row["action_credit_mask"]) == 1


def test_analysis_helpers_are_deterministic():
    result = two_level_bootstrap({"101": [1, 2], "202": [2, 3]}, draws=200, seed=4)
    assert result["mean"] == pytest.approx(2.0)
    assert holm({"a": 0.01, "b": 0.04})["a"]["reject"] is True
    gain = timing([3, 3, 3], [1, 1, 1], [2, 2, 2], draws=200)
    assert gain["agent_share"] == pytest.approx(2.0)


def test_fleet_grid_and_selection():
    assert FLEET_GRID == tuple(round(0.6 + 0.1 * i, 1) for i in range(15))
    rows = [
        {"fleet_scale": 1.0, "base_pickup_minutes": 3.0,
         "trips_per_online_driver_hour": 1.5, "median_wait_minutes": 5.0,
         "fulfillment": 0.9},
        {"fleet_scale": 1.1, "base_pickup_minutes": 4.0,
         "trips_per_online_driver_hour": 1.78, "median_wait_minutes": 6.0,
         "fulfillment": 0.92},
    ]
    selected = select_fit(rows, target_trips_per_hour=1.78, target_median_wait=6.0)
    assert selected["fleet_scale"] == 1.1


def test_p2_pickup_grid_reports_infeasible_capacity_candidate(monkeypatch, tmp_path):
    jobs = [
        Job("ok", (), tmp_path / "ok.json", tmp_path / "ok.manifest",
            tmp_path / "ok.log", {}),
        Job("pickup_7.5", (), tmp_path / "bad.json", tmp_path / "bad.manifest",
            tmp_path / "bad.log", {}),
    ]

    def run(job, **_):
        if job.name == "pickup_7.5":
            raise RuntimeError("v3 capacity target unreachable: diagnostic candidate")
        return "completed"

    monkeypatch.setattr("scripts.v18_p2._run_job", run)
    monkeypatch.setattr("scripts.v18_p2._source_identity", lambda: {})
    monkeypatch.setattr("scripts.v18_p2.load_v18", lambda: {})
    monkeypatch.setattr("scripts.v18_p2.canonical_hash", lambda _: "config")
    monkeypatch.setattr(
        "scripts.v18_p2.subprocess.check_output", lambda *_, **__: "head\n",
    )
    completed, infeasible = _run_parallel(
        jobs, workers=2, output=tmp_path, phase="pickup_grid",
        allow_capacity_infeasible=True,
    )
    assert completed == 1
    assert [row["job"] for row in infeasible] == ["pickup_7.5"]
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["state"] == "running"
    assert status["processed_jobs"] == 2
    assert status["infeasible_jobs"][0]["job"] == "pickup_7.5"
    assert status["infeasible_jobs"][0]["reason"].startswith("capacity target unreachable")


def test_fleet_selection_separates_fleet_and_pickup_stages():
    rows = [
        {"stage": "fleet", "fleet_scale": 1.0, "base_pickup_minutes": 3.5,
         "trips_per_online_driver_hour": 1.79, "median_wait_minutes": 4.0,
         "fulfillment": 0.9},
        {"stage": "fleet", "fleet_scale": 1.1, "base_pickup_minutes": 3.5,
         "trips_per_online_driver_hour": 1.6, "median_wait_minutes": 4.0,
         "fulfillment": 0.9},
        {"stage": "pickup", "fleet_scale": 1.0, "base_pickup_minutes": 5.0,
         "trips_per_online_driver_hour": 1.0, "median_wait_minutes": 6.0,
         "fulfillment": 0.88},
    ]
    selected = select_fit(rows, target_trips_per_hour=1.78, target_median_wait=6.0)
    assert selected["fleet_scale"] == 1.0
    assert selected["base_pickup_minutes"] == 5.0


def test_wait_histogram_median_and_p2_measurement():
    counts = np.zeros(10, dtype=np.int64)
    counts[[2, 4, 8]] = [1, 2, 1]
    assert _histogram_median(counts) == 4.0
    payload = {
        "traces": {"card": [
            {"own_share": 0.4, "active_drivers_firm1": 25,
             "wait": 5.0, "fulfillment": 0.9},
            {"own_share": 0.5, "active_drivers_firm1": 25,
             "wait": 7.0, "fulfillment": 1.0},
        ]},
        "v26r": {"supply_reference_earnings": 20.0},
    }
    row = _measurement(
        payload, "card", stage="fleet", fleet_scale=1.2,
        base_pickup_minutes=3.5,
    )
    assert row["trips_per_online_driver_hour"] == pytest.approx(2.25)
    assert row["median_wait_minutes"] == pytest.approx(6.0)
