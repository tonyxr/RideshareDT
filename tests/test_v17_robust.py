"""Unit and contract tests for the V17 robustness workflow."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from Core import Core
from Market_models import CoefficientOverrides, MarketCoefficients
from driver_supply import DriverSupplyConfig
from optim_config import default_specs_for
from pricing_models import FirmMemoryBestResponsePricer
from scripts import v17_robust_eval, v17_robust_select, v17_wp4_train
from scripts.v17_config import load_config
from scripts.v17_robust_common import check_overrides, set_option
from v17_surge_decay import surge_decay_spec, surge_decay_weight


KEYS = [
    "base_fare", "per_minute", "per_mile", "booking_fee", "airport_fee"
]


def _posting_core() -> Core:
    core = Core.__new__(Core)
    core.shared_steps_per_day = 6
    core.market = SimpleNamespace(curr_market=MarketCoefficients(
        base_fare=2.0,
        per_mile=3.0,
        per_minute=0.5,
        booking_fee=1.0,
        airport_fee=4.0,
        surge_hours=[],
        day_multiplier={},
        weather_multiplier={},
        service_multiplier={},
    ))
    core.firm1 = SimpleNamespace(
        two_lever_actions=True,
        opt_keys=KEYS,
        config=default_specs_for(KEYS),
        overrides=CoefficientOverrides(),
        supply_incentive_multiplier=1.0,
    )
    return core


def _ratios(core: Core) -> dict[str, float]:
    base = core.market.curr_market
    return {
        key: float(getattr(core.firm1.overrides, key)) / float(getattr(base, key))
        for key in KEYS
    }


def _small_core(**kwargs) -> Core:
    return Core(
        "Boston",
        seed=71,
        firm1_mode="RL",
        firm2_mode="adaptive_best_response_memory",
        total_customers_pool=8,
        simulation_sample_cap=8,
        enable_driver_supply=False,
        openai_api_key="",
        fare_model="realistic",
        action_contract="two_lever_v1",
        rivals_two_lever=True,
        **kwargs,
    )


def test_surge_decay_weight_table():
    windows = [[10, 15]]
    expected = {
        9: 0.0,
        10: 1.0,
        17: 1.0,
        18: 0.75,
        19: 0.5,
        20: 0.25,
        21: 0.0,
    }
    for day, weight in expected.items():
        assert surge_decay_weight(day, windows, 2, 3) == pytest.approx(weight)
    assert surge_decay_weight(18, [[10, 15], [18, 18]], 2, 3) == 1.0
    assert surge_decay_weight(10, [], 2, 3) == 0.0


def test_surge_decay_spec_roundtrip():
    spec = surge_decay_spec((0.8, 1.0), (1.5, 2.0), [[10, 15]], 2, 4)
    payload = Core._parse_policy_table(spec.split(":", 1)[1])
    assert payload == {
        "normal": [0.8, 1.0],
        "event": [1.5, 2.0],
        "windows": [[10, 15]],
        "hold_days": 2,
        "decay_days": 4,
    }


def test_surge_decay_posts_full_card():
    core = _posting_core()
    spec = surge_decay_spec((1.0, 1.0), (1.25, 1.25), [[1, 1]], 0, 0)
    observed = []
    for step in (0, 6, 12):
        action = Core._baseline_firm1_action(
            core,
            spec,
            step_index=step,
            action_mask=np.ones(25, dtype=bool),
            day_of_week=0,
            hour=12,
            weather="clear",
            presampled=None,
            customers_per_step=1,
        )
        assert action == -1
        observed.append(_ratios(core))
    assert observed[0] == pytest.approx({key: 1.0 for key in KEYS})
    assert observed[1] == pytest.approx({
        "base_fare": 1.0,
        "per_minute": 1.25,
        "per_mile": 1.25,
        "booking_fee": 1.0,
        "airport_fee": 1.0,
    })
    assert observed[2] == pytest.approx({key: 1.0 for key in KEYS})


def test_reactive_surge_uses_delayed_kpis_and_caps_full_card():
    core = _posting_core()
    core.firm1.config.bounds["per_mile"] = (0.8 * 3.0, 2.5 * 3.0)
    core.firm1.config.bounds["per_minute"] = (0.8 * 0.5, 2.5 * 0.5)
    kpis = {
        "fulfillment": 0.95,
        "wait": 5.0,
        "storm_warning": 0.0,
    }
    core._firm1_public_kpis = lambda _hour: dict(kpis)
    spec = (
        'reactive_surge:{"normal":[1.0,2.0],"fill_ref":0.95,'
        '"fill_gain":2.0,"wait_ref":5.0,"wait_gain":0.5,'
        '"storm_gain":0.0,"max_level":2.5}'
    )
    Core._baseline_firm1_action(
        core, spec, step_index=0, action_mask=np.ones(25, dtype=bool),
        day_of_week=0, hour=12, weather="clear", presampled=None,
        customers_per_step=1,
    )
    assert _ratios(core) == pytest.approx({
        "base_fare": 1.0, "per_minute": 2.0, "per_mile": 1.0,
        "booking_fee": 1.0, "airport_fee": 1.0,
    })
    kpis["fulfillment"] = 0.55
    Core._baseline_firm1_action(
        core, spec, step_index=6, action_mask=np.ones(25, dtype=bool),
        day_of_week=1, hour=12, weather="clear", presampled=None,
        customers_per_step=1,
    )
    assert _ratios(core) == pytest.approx({
        "base_fare": 1.0, "per_minute": 2.5, "per_mile": 1.8,
        "booking_fee": 1.0, "airport_fee": 1.0,
    })


def test_eta_v3_reaches_market():
    payload = {
        "v26r": {
            "effective_supply_config": {
                "eta_v3": 0.3,
                "shortage_willingness_share": 0.5,
            },
            "memory_rival_overrides": {
                "hold_mean_days": None,
                "retaliation": "on",
            },
        }
    }
    check_overrides(payload, {"supply_eta_v3": 0.3})
    assert DriverSupplyConfig().eta_v3 == pytest.approx(0.6)


def test_memory_rival_overrides():
    core = _small_core(
        memory_rival_hold_mean_days=10.0,
        memory_rival_retaliation="off",
    )
    core.shared_steps_per_day = 6
    pricer = core._make_pricer("adaptive_best_response_memory", 7)
    assert isinstance(pricer, FirmMemoryBestResponsePricer)
    assert pricer.hold_mean_steps == 60
    assert pricer.lag_steps == 10**9
    other = core._make_pricer("adaptive_best_response", 7)
    assert not isinstance(other, FirmMemoryBestResponsePricer)


def test_retaliation_off_never_retaliates():
    core = _small_core(memory_rival_retaliation="off")
    pricer = core._make_pricer("adaptive_best_response_memory", 7)
    assert pricer.lag_steps == 10**9
    pricer.undercut_steps = 1_000_000
    assert pricer.undercut_steps < pricer.lag_steps


def test_defaults_identity():
    implicit = _small_core()
    explicit = _small_core(
        memory_rival_retaliation="on",
        memory_rival_hold_mean_days=None,
    )
    implicit.shared_steps_per_day = explicit.shared_steps_per_day = 6
    left = implicit._make_pricer("adaptive_best_response_memory", 7)
    right = explicit._make_pricer("adaptive_best_response_memory", 7)
    assert left.hold_mean_steps == right.hold_mean_steps == 30
    assert left.lag_steps == right.lag_steps
    assert set_option(["python", "--x", "1"], "--x", 2) == [
        "python", "--x", "2"
    ]
    assert set_option(["python"], "--x", 2) == ["python", "--x", "2"]


def test_wp4_second_seed_ablations_keep_registered_commands(tmp_path):
    expected = {
        "nyc_s101_ppo_tuned": "plain_ppo",
        "nyc_s101_ppo_matched": "plain_ppo",
        "nyc_s202_no_driver_obs": "no_driver_observation",
        "nyc_s202_own_profit": "own_profit",
    }
    assert {name: spec.kind for name, spec in v17_wp4_train.RUNS.items()} == expected
    config = load_config()
    for name in ("nyc_s202_no_driver_obs", "nyc_s202_own_profit"):
        spec = v17_wp4_train.RUNS[name]
        prefix = tmp_path / name / "nyc"
        registered = v17_wp4_train.build_command(
            spec, days=1200, output_root=tmp_path, config=config,
            diagnostic_prefix=prefix,
        )
        observed = v17_wp4_train._tuned_command(
            spec, days=1200, output=tmp_path, config=config,
            diagnostic_prefix=prefix,
        )
        assert observed == registered


def test_selector_custom_status_name(monkeypatch, tmp_path):
    writes = []
    monkeypatch.setattr(v17_robust_select, "validate_config", lambda _config: None)
    monkeypatch.setattr(
        v17_robust_select, "_run_city",
        lambda args, city, config: {"city": city, "mode": args.mode},
    )
    monkeypatch.setattr(
        v17_robust_select, "atomic_json",
        lambda path, payload: writes.append((path, dict(payload))),
    )
    assert v17_robust_select.main([
        "--mode", "point", "--city", "nyc", "--workers", "1",
        "--from", str(tmp_path / "selected"), "--output", str(tmp_path),
        "--status-name", "status_eta0.3.json",
    ]) == 0
    assert len(writes) == 2
    assert {path.name for path, _payload in writes} == {"status_eta0.3.json"}
    assert writes[-1][1]["state"] == "completed"
    assert writes[-1][1]["pass"] is True


def test_abl_suite_has_second_seed_and_all_registered_cases(monkeypatch, tmp_path):
    monkeypatch.setattr(
        v17_robust_eval, "_require_run",
        lambda run, **_kwargs: (
            tmp_path / run.name / "status.json",
            tmp_path / run.name / "policy.pt",
            {},
        ),
    )
    monkeypatch.setattr(v17_robust_eval, "_sha256_file", lambda _path: "sha256")
    jobs, overrides, paths = v17_robust_eval._build_abl(
        load_config(), tmp_path / "abl",
    )
    assert len(jobs) == 16
    assert overrides == {}
    assert len(paths) == 8
    by_run = {}
    for job in jobs:
        by_run.setdefault(job.labels[0], set()).add(job.case)
    assert by_run["nyc_s101_no_driver_obs"] == {
        "A3_storm_transit_late", "S1_promotion_bait",
        "S6_shortage_during_promotion",
    }
    assert by_run["nyc_s101_own_profit"] == by_run["nyc_s101_no_driver_obs"]
    assert by_run["nyc_s202_no_driver_obs"] == set(v17_robust_eval.registered.CASES)
    assert by_run["nyc_s202_own_profit"] == set(v17_robust_eval.registered.CASES)


def test_rule_mode_plans_registered_candidate_grid(monkeypatch, tmp_path):
    selected = tmp_path / "selected.json"
    selected.write_text("{}", encoding="utf-8")
    wait_source = tmp_path / "calm.json"
    monkeypatch.setattr(v17_robust_select, "_normal_cell", lambda _city: (1.0, 1.0))
    monkeypatch.setattr(
        v17_robust_select, "_selected_comparators",
        lambda _city: (selected, {}),
    )
    monkeypatch.setattr(
        v17_robust_select, "_rule_wait_ref",
        lambda _city, _normal: (wait_source, 5.0),
    )
    monkeypatch.setattr(v17_robust_select, "_sha256_file", lambda _path: "sha256")
    args = SimpleNamespace(
        output=tmp_path, mode="rule", plan_only=True, workers=6,
        source=None, points="all",
    )
    plan = v17_robust_select._run_city(args, "nyc", load_config())
    assert plan["mode"] == "rule"
    assert plan["candidates"] == 24
    assert plan["jobs"] == 6
    assert plan["wait_ref"] == 5.0


def test_rule_suite_has_three_cities_and_five_cases(monkeypatch, tmp_path):
    selected_root = tmp_path / "select/rule"
    spec = (
        'reactive_surge:{"fill_gain":2.0,"fill_ref":0.95,'
        '"max_level":2.5,"normal":[1.0,1.0],"storm_gain":0.0,'
        '"wait_gain":0.5,"wait_ref":5.0}'
    )
    for city in ("nyc", "boston", "chicago"):
        path = selected_root / city / "selected_rule.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"reactive_surge": {"spec": spec}}), encoding="utf-8")
    monkeypatch.setattr(v17_robust_eval, "RULE_SELECTED", selected_root)
    jobs, overrides, paths = v17_robust_eval._build_rule(
        load_config(), tmp_path / "eval/rule",
    )
    assert len(jobs) == 15
    assert overrides == {}
    assert len(paths) == 3
    assert {job.city for job in jobs} == {"nyc", "boston", "chicago"}
    assert {job.case for job in jobs} == set(v17_robust_eval.registered.CASES)
    assert all(job.labels == ("reactive_surge",) for job in jobs)
    assert all(job.specs == (spec,) for job in jobs)


def test_leftout_overrides_render_once_and_validate(tmp_path):
    config = load_config()
    job = v17_robust_eval.make_job(
        config, suite="leftout_market", point="no_spillover", city="nyc",
        case="C0_calm", scenario=v17_robust_eval.registered.CASES["C0_calm"],
        labels=("published_card",), specs=("static_tariff:{}",),
        output_root=tmp_path,
        overrides={
            "loyalty_model": "v1", "threshold_ablation": "constant_mean",
            "disable_spillover": True,
        },
    )
    command = list(job.command)
    assert command.count("--loyalty_model") == 1
    assert command[command.index("--loyalty_model") + 1] == "v1"
    assert command.count("--threshold_ablation") == 1
    assert command[command.index("--threshold_ablation") + 1] == "constant_mean"
    assert command.count("--disable_spillover") == 1
    check_overrides({
        "model_switches": {
            "loyalty_model": "v1", "threshold_ablation": "constant_mean",
            "unfulfilled_spillover": False,
        }
    }, {
        "loyalty_model": "v1", "threshold_ablation": "constant_mean",
        "disable_spillover": True,
    })


def test_leftout_mode_plans_registered_variants(monkeypatch, tmp_path):
    selected = tmp_path / "selected.json"
    selected.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(v17_robust_select, "_normal_cell", lambda _city: (1.0, 1.0))
    monkeypatch.setattr(
        v17_robust_select, "_selected_comparators", lambda _city: (selected, {}),
    )
    monkeypatch.setattr(v17_robust_select, "_sha256_file", lambda _path: "sha256")
    args = SimpleNamespace(
        output=tmp_path, mode="leftout", plan_only=True, workers=6,
        source=None, points="all", days=28, openings=3,
        variant="drivers_fixed",
    )
    plan = v17_robust_select._run_city(args, "nyc", load_config())
    assert plan["candidates"] == 36
    assert plan["jobs"] == 6
    assert plan["overrides"] == {
        "supply_eta_v3": 0.01, "shortage_willingness_share": 0.0,
    }
    args.variant = "static_rival"
    plan = v17_robust_select._run_city(args, "nyc", load_config())
    assert plan["jobs"] == 2
    assert plan["rivals"] == ["static"]


def test_leftout_suite_has_fourteen_specs_per_city_case(monkeypatch, tmp_path):
    selected_root = tmp_path / "select/leftout"
    selection = {
        "best_fixed_cell": {
            "cell": [1.0, 1.0], "spec": v17_robust_eval.cell_spec(1.0, 1.0),
        },
        "best_shortage_cell": {
            "cell": [2.0, 2.0], "spec": v17_robust_eval.cell_spec(2.0, 2.0),
        },
    }
    for variant in ("full", *v17_robust_select.LEFTOUT_VARIANTS):
        path = selected_root / "nyc" / variant / "selected_leftout.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(selection), encoding="utf-8")
    monkeypatch.setattr(v17_robust_eval, "LEFTOUT_SELECTED", selected_root)
    jobs, overrides, paths = v17_robust_eval._build_leftout(
        load_config(), tmp_path / "eval/leftout", cities=("nyc",),
    )
    assert len(jobs) == 5
    assert overrides == {}
    assert len(paths) == 7
    assert all(len(job.labels) == 14 for job in jobs)
    assert all(len(job.specs) == 14 for job in jobs)
    assert all(job.labels[:2] == ("fixed@full", "dated@full") for job in jobs)


def test_leftout_market_plans_thirty_jobs_per_city(tmp_path):
    jobs, overrides, paths = v17_robust_eval._build_leftout_market(
        load_config(), tmp_path / "eval/leftout_market", cities=("nyc",),
    )
    assert len(jobs) == 30
    assert len(overrides) == 30
    assert paths == []
    static_jobs = [job for job in jobs if "/static_rival/" in f"/{job.name}/"]
    assert len(static_jobs) == 5
    for job in static_jobs:
        command = list(job.command)
        assert command[command.index("--firm2_mode") + 1] == "static"
        assert command[command.index("--policy_sweep_rivals") + 1] == "static"
