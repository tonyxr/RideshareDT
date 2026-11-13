"""Fast code-contract tests for the V18 threat-removal implementation."""

from datetime import date, timedelta
import csv
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import torch

from Core import (
    Core, _normalized_opening_tariff_randomization,
    build_shock_config_from_args,
)
from driver_supply import DriverPayPolicy, DriverSupplyConfig, DriverSupplyLayer, FirmDriverBatchState
from market_shocks import ShockConfig, ShockEvent, ShockProcess, ShockScenario
from scripts.v18_config import (
    HORIZON_CANDIDATES, demand_artifact, load_config, validate_config,
)
from scripts.v18_fit_pay_rule import DEFAULT_INPUT, fit, load_rows, transfer
from scripts.v18_revealed_beta import _interpolated_factor
from scripts.v18_p2 import (
    FLEET_GRID, PICKUP_GRID, PUBLISHED, TRANSFER_FLEET_GRID, Job, _block_waits,
    _histogram_median,
    _calm_published_stock_gate, _calm_published_stock_job,
    _measurement, _profile_from_factors, _run_parallel,
    run_reference_earnings,
    _select_joint_calibration, _select_kappa_candidate,
    _select_transfer_service_level,
    _updated_profile_factors,
)
from scripts.v17_config import load_config as load_v17_config
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
    assert config["market"]["demand_scale"] == "lyft_prior"
    assert demand_artifact(config, "nyc").name == "demand_1.15.json"
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


def test_v18_integral_stock_is_stationary_at_reference_and_monotone():
    layer = _v3(True)
    layer.config.stock_rule = "integral"
    layer.config.eta_within = 0.6
    for _ in range(3):
        layer._daily_pay.update({"Firm1": 1000.0, "Firm2": 1000.0})
        layer._daily_online.update({"Firm1": 3000.0, "Firm2": 3000.0})
        assert layer.advance_day()["Firm1"] == 1.0
    above = []
    for _ in range(10):
        layer._daily_pay.update({"Firm1": 1100.0, "Firm2": 1000.0})
        layer._daily_online.update({"Firm1": 3000.0, "Firm2": 3000.0})
        above.append(layer.advance_day()["Firm1"])
    assert all(right > left for left, right in zip([1.0, *above[:-1]], above))
    below = []
    for _ in range(10):
        layer._daily_pay.update({"Firm1": 900.0, "Firm2": 1000.0})
        layer._daily_online.update({"Firm1": 3000.0, "Firm2": 3000.0})
        below.append(layer.advance_day()["Firm1"])
    assert all(right < left for left, right in zip([above[-1], *below[:-1]], below))
    layer.online_stock["Firm1"] = 1.99
    layer._daily_pay["Firm1"] = 100_000.0
    layer._daily_online["Firm1"] = 1.0
    assert layer.advance_day()["Firm1"] == 2.0
    layer.online_stock["Firm1"] = 0.41
    layer._daily_pay["Firm1"] = 0.0
    layer._daily_online["Firm1"] = 1.0
    assert layer.advance_day()["Firm1"] == 0.4


def test_v18_integral_stock_ignores_shock_wage_and_uses_same_day_eta():
    layer = _v3(True)
    layer.config.stock_rule = "integral"
    layer.config.eta_within = 0.6
    layer.shock_wage_multiplier = 9.0
    layer._daily_pay.update({"Firm1": 1100.0, "Firm2": 1000.0})
    layer._daily_online.update({"Firm1": 3000.0, "Firm2": 3000.0})
    expected = float(np.exp(0.05 * np.log(1.1)))
    assert layer.advance_day()["Firm1"] == pytest.approx(expected)
    assert layer._v3_eta_total() == pytest.approx(0.6)


def test_v18_earnings_helper_is_ratio_of_sums():
    assert DriverSupplyLayer.earnings_per_online_hour(300.0, 900.0) == 20.0
    with pytest.raises(ValueError, match="positive online minutes"):
        DriverSupplyLayer.earnings_per_online_hour(1.0, 0.0)


def test_p2_calm_stock_gate_is_exact_both_published_cell(tmp_path: Path):
    assert _normalized_opening_tariff_randomization(0.0) == 0.0
    assert _normalized_opening_tariff_randomization(0.15) == 0.15
    assert _normalized_opening_tariff_randomization(1.0) == 0.40
    for city in ("nyc", "boston", "chicago"):
        job = _calm_published_stock_job(
            load_v17_config(), tmp_path, city=city
        )
        command = list(job.command)
        assert job.name == f"calm_published_stock_gate|{city}"
        assert job.raw.parent.name == city
        assert command[command.index("--policy_sweep") + 1] == PUBLISHED
        assert command[command.index("--policy_sweep_rivals") + 1] == "static"
        assert command[
            command.index("--policy_sweep_opening_tariff_randomization") + 1
        ] == "0.0"
        assert int(command[command.index("--policy_sweep_horizon") + 1]) == 240
        assert int(command[command.index("--policy_sweep_seed") + 1]) == 20231
        assert int(command[command.index("--train_customers") + 1]) == 500

    rows = []
    for opening in (1, 2):
        for step in range(240):
            rows.append({
                "rollout": f"static__opening_{opening}",
                "step": step,
                "firm1_level": 1.0,
                "firm2_level": 1.0,
                "stock_firm1": 1.02,
                "stock_firm2": 0.99,
            })
    gate = _calm_published_stock_gate(
        {"traces": {PUBLISHED: rows}}, city="nyc"
    )
    assert gate["pass"]
    assert gate["protocol_revision"] == 2
    assert gate["city"] == "nyc"
    assert gate["population"] == "pure_C0_both_platforms_published_exact"
    assert gate["days"] == [10, 39]
    assert gate["gated_stock"] == ["stock_firm1", "stock_firm2"]
    assert set(gate["platforms"]) == {"platform1", "platform2"}
    assert gate["rollouts"] == 2
    assert gate["trace_rows"] == 360

    rows[0]["firm2_level"] = 1.1
    with pytest.raises(ValueError, match="both platforms"):
        _calm_published_stock_gate(
            {"traces": {PUBLISHED: rows}}, city="nyc"
        )

    rows[0]["firm2_level"] = 1.0
    for row in rows:
        if 10 <= row["step"] // 6 < 40:
            row["stock_firm2"] = 1.031
    gate = _calm_published_stock_gate(
        {"traces": {PUBLISHED: rows}}, city="nyc"
    )
    assert not gate["pass"]
    assert gate["platforms"]["platform1"]["pass"]
    assert not gate["platforms"]["platform2"]["pass"]


def test_p2_revision2_reference_earnings_remeasures_only_registered_statistic(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    products = {}
    for city in ("nyc", "boston", "chicago"):
        supply = json.loads(
            (Path("calibration_artifacts/v18") / city / "supply.json").read_text()
        )
        rows = []
        for opening in (1, 2, 3):
            for step in range(168):
                rows.append({
                    "rollout": f"static__opening_{opening}",
                    "step": step,
                    "completed_rides_firm1": 1,
                    "completed_rides_firm2": 1,
                    "unfulfilled_requests_firm1": 0,
                    "unfulfilled_requests_firm2": 0,
                    "online_minutes_firm1": 60.0,
                    "online_minutes_firm2": 60.0,
                    "passenger_minutes_firm1": 30.0,
                    "passenger_minutes_firm2": 30.0,
                    "driver_pay_firm1": 20.0,
                    "driver_pay_firm2": 20.0,
                    "pickup_minutes_firm1": [4.0],
                    "pickup_minutes_firm2": [4.0],
                })
        products[city] = {"traces": {PUBLISHED: rows}}

    def fake_parallel(jobs, **kwargs):
        del kwargs
        for job in jobs:
            city = job.name.rsplit("/", 1)[-1]
            command = list(job.command)
            assert command[command.index("--seed") + 1] == "20231"
            assert command[command.index("--experiment_seed") + 1] == "20231"
            assert command[command.index("--policy_sweep_seed") + 1] == "20231"
            assert command[command.index("--policy_sweep") + 1] == PUBLISHED
            supplied_reference = float(
                command[command.index("--supply_reference_earnings") + 1]
            )
            expected_reference = json.loads(
                (supply_root / city / "supply.json").read_text()
            )["reference_earnings_per_hour"]
            assert supplied_reference == pytest.approx(expected_reference)
            job.raw.parent.mkdir(parents=True, exist_ok=True)
            job.raw.write_text(json.dumps(products[city]))
        return len(jobs), []

    monkeypatch.setattr("scripts.v18_p2._run_parallel", fake_parallel)
    monkeypatch.setattr(
        "scripts.v18_p2._validate_product",
        lambda raw, expected: json.loads(raw.read_text()),
    )
    monkeypatch.setattr(
        "scripts.v18_p2.PAY_DIR", Path("calibration_artifacts/v18").resolve()
    )
    # Work on temporary supply copies so the test proves the narrow mutation.
    supply_root = tmp_path / "pay"
    for city in ("nyc", "boston", "chicago"):
        target = supply_root / city
        target.mkdir(parents=True)
        source = Path("calibration_artifacts/v18") / city
        for name in ("supply.json", "pay_rule.json"):
            (target / name).write_bytes((source / name).read_bytes())
    monkeypatch.setattr("scripts.v18_p2.PAY_DIR", supply_root)
    report = run_reference_earnings(
        load_v17_config(), tmp_path / "run", workers=3
    )
    assert set(report["cities"]) == {"nyc", "boston", "chicago"}
    for city in report["cities"]:
        updated = json.loads((supply_root / city / "supply.json").read_text())
        assert updated["reference_earnings_per_hour"] == pytest.approx(20.0)
        assert updated["reference_earnings_revision2_validation"][
            "opening_tariff_randomization"
        ] == 0.0


def test_v18_batch_state_records_exact_calibration_quantities():
    layer = DriverSupplyLayer(config=DriverSupplyConfig(
        supply_model="v3", base_active_drivers=20,
        hour_profile={hour: 1.0 for hour in range(24)},
        expected_acceptance_cutoff=0.0,
    ), seed=3)
    layer.begin_batch(20, 12, "clear")
    result = layer.dispatch("Firm1", 20.0, 3.0, 15.0)
    assert result.fulfilled
    state = layer.end_batch()["Firm1"]
    assert state.total_passenger_minutes == pytest.approx(15.0)
    assert state.completed_pickup_minutes == pytest.approx([
        result.pickup_minutes
    ])


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


def test_frozen_capacity_metadata_preserves_volume_identity():
    core = Core.__new__(Core)
    core.v26_settings = {
        "driver_capacity_calibration": {
            "enabled": True,
            "method": "frozen_v26_capacity_anchor",
            "base_active_drivers": 415,
            "sample_size": 500,
        }
    }
    core.driver_supply = SimpleNamespace(supply_v2=False, supply_v3=True)
    core._check_supply_calibration_volume(500)
    with pytest.raises(ValueError, match="calibrated at 500"):
        core._check_supply_calibration_volume(499)


def test_calibration_totals_are_default_off_for_legacy_identity():
    assert inspect.signature(Core).parameters[
        "policy_sweep_supply_totals"
    ].default is False


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
    assert FLEET_GRID == (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    assert PICKUP_GRID == (1.0, 1.25, 1.5, 1.75, 2.0)


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


def test_joint_selection_interpolates_and_obeys_trip_ceiling():
    rows = []
    for fleet in (1.0, 1.2):
        for pickup in (2.0, 2.5):
            rows.append({
                "fleet_scale": fleet,
                "base_pickup_minutes": pickup,
                "utilization": 0.58 + (1.1 - fleet) * 0.2,
                "median_wait_minutes": 4.27 + (pickup - 2.25),
                "trips_per_online_driver_hour": 1.7,
            })
    selected = _select_joint_calibration(rows, target_wait=4.27)
    assert selected["fleet_scale"] == pytest.approx(1.1)
    assert selected["base_pickup_minutes"] == pytest.approx(2.25)
    assert selected["binding_targets_pass"] is True
    assert selected["utilization_fallback_used"] is False


def test_rev1_grid_and_utilization_fallback_are_registered():
    assert FLEET_GRID == (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    assert PICKUP_GRID == (1.0, 1.25, 1.5, 1.75, 2.0)
    rows = []
    for fleet, utilization in ((0.5, 0.49), (0.6, 0.47)):
        for pickup in (1.0, 1.25):
            rows.append({
                "fleet_scale": fleet,
                "base_pickup_minutes": pickup,
                "utilization": utilization,
                "median_wait_minutes": 4.27,
                "trips_per_online_driver_hour": 1.7,
            })
    selected = _select_joint_calibration(rows, target_wait=4.27)
    assert selected["fleet_scale"] == pytest.approx(0.5)
    assert selected["utilization_fallback_used"] is True
    assert selected["utilization_gap"] == pytest.approx(0.03)
    assert selected["binding_targets_pass"] is True


def test_rev1_city_transfer_uses_original_grid_and_only_block_wait_loss():
    assert TRANSFER_FLEET_GRID == (0.6, 0.8, 1.0, 1.2, 1.4, 1.6)
    selected = _select_transfer_service_level([
        {
            "fleet_scale": 0.6,
            "service_level_loss": 1.0,
            "median_wait_minutes": 2.2,
            "trips_per_online_driver_hour": 2.2,
            "utilization": 0.2,
        },
        {
            "fleet_scale": 0.8,
            "service_level_loss": 2.0,
            "median_wait_minutes": 4.27,
            "trips_per_online_driver_hour": 1.7,
            "utilization": 0.58,
        },
    ])
    assert selected["fleet_scale"] == pytest.approx(0.6)


def test_rev1_kappa_diagnostic_retains_registered_value_when_none_passes():
    rows = [
        {"kappa": 0.03, "pass": False},
        {"kappa": 0.05, "pass": False},
        {"kappa": 0.08, "pass": False},
    ]
    selected, retained = _select_kappa_candidate(
        rows, registered_kappa=0.05
    )
    assert retained is True
    assert selected["kappa"] == pytest.approx(0.05)


def test_rev1_kappa_diagnostic_still_selects_smallest_passing_candidate():
    rows = [
        {"kappa": 0.03, "pass": False},
        {"kappa": 0.05, "pass": True},
        {"kappa": 0.08, "pass": True},
    ]
    selected, retained = _select_kappa_candidate(
        rows, registered_kappa=0.05
    )
    assert retained is False
    assert selected["kappa"] == pytest.approx(0.05)


def test_rev1_selection_stops_only_without_binding_feasible_point():
    rows = [{
        "fleet_scale": fleet,
        "base_pickup_minutes": pickup,
        "utilization": 0.58,
        "median_wait_minutes": 6.0,
        "trips_per_online_driver_hour": 1.7,
    } for fleet in (0.5, 0.6) for pickup in (1.0, 1.25)]
    with pytest.raises(ValueError, match="wait and trips/hour"):
        _select_joint_calibration(rows, target_wait=4.27)


def test_rev1_hour_profile_update_preserves_mean_and_within_block_shape():
    factors = {block: 1.0 for block in range(6)}
    baseline = _profile_from_factors(factors)
    target = {block: 4.0 for block in range(6)}
    simulated = {block: (8.0 if block == 0 else 4.0) for block in range(6)}
    updated, profile = _updated_profile_factors(factors, simulated, target)
    assert np.mean(list(baseline.values())) == pytest.approx(1.0)
    assert np.mean(list(profile.values())) == pytest.approx(1.0)
    assert updated[0] > updated[1]
    assert profile[1] / profile[0] == pytest.approx(
        baseline[1] / baseline[0]
    )


def test_rev1_block_waits_use_completed_pickups_by_four_hour_block():
    traces = []
    for block in range(6):
        traces.append({
            "step": 24 + block,
            "hour": block * 4,
            "pickup_minutes_firm1": [float(block + 1)],
            "pickup_minutes_firm2": [float(block + 2)],
        })
    waits = _block_waits({"traces": {"card": traces}}, "card")
    assert waits == {
        block: pytest.approx(block + 1.5) for block in range(6)
    }


def test_revealed_beta_interpolates_unit_optimum():
    rows = [
        {"factor": 1.0, "quadratic_maximizer": 1.08},
        {"factor": 2.0, "quadratic_maximizer": 0.96},
        {"factor": 3.0, "quadratic_maximizer": 0.92},
    ]
    assert _interpolated_factor(rows) == pytest.approx(5.0 / 3.0)


def test_wait_histogram_median_and_p2_measurement():
    counts = np.zeros(10, dtype=np.int64)
    counts[[2, 4, 8]] = [1, 2, 1]
    assert _histogram_median(counts) == 4.0
    payload = {
        "traces": {"card": [
            {
                "step": 24,
                "completed_rides_firm1": 4, "completed_rides_firm2": 5,
                "unfulfilled_requests_firm1": 1,
                "unfulfilled_requests_firm2": 0,
                "online_minutes_firm1": 120, "online_minutes_firm2": 180,
                "passenger_minutes_firm1": 60,
                "passenger_minutes_firm2": 90,
                "driver_pay_firm1": 40, "driver_pay_firm2": 60,
                "pickup_minutes_firm1": [5, 5, 5, 5],
                "pickup_minutes_firm2": [7, 7, 7, 7, 7],
            },
            {
                "step": 25,
                "completed_rides_firm1": 5, "completed_rides_firm2": 4,
                "unfulfilled_requests_firm1": 0,
                "unfulfilled_requests_firm2": 1,
                "online_minutes_firm1": 180, "online_minutes_firm2": 120,
                "passenger_minutes_firm1": 90,
                "passenger_minutes_firm2": 60,
                "driver_pay_firm1": 60, "driver_pay_firm2": 40,
                "pickup_minutes_firm1": [5, 5, 5, 5, 5],
                "pickup_minutes_firm2": [7, 7, 7, 7],
            },
        ]},
    }
    row = _measurement(
        payload, "card", stage="fleet", fleet_scale=1.2,
        base_pickup_minutes=3.5,
    )
    assert row["trips_per_online_driver_hour"] == pytest.approx(1.8)
    # Fix A registers passenger minutes / online minutes and explicitly
    # excludes pickup time.
    assert row["utilization"] == pytest.approx(0.5)
    assert row["median_wait_minutes"] == pytest.approx(6.0)
    assert row["reference_earnings_per_hour"] == pytest.approx(20.0)
