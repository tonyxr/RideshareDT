from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path
import sys

import numpy as np

from Core import (
    Core, _evaluation_supply_operating_point, _gated_rl_transition,
    _v18_card_clear_quote,
    _v18_request_step,
)
from MarketInteraction import RideContext
from scripts.v18_config import eval_args, load_config, train_args
from scripts.v18_market import market_flags
from scripts.v18_eval import SCENARIOS, build_jobs as build_evaluation_jobs
from scripts.v18_runtime import Job, run_job, run_jobs
from scripts.v18_select import (
    _grid_edge_selected, _selection_plan, _stage3_count_rules, card_grid,
    reactive_candidates,
)
from scripts.v18_validation.sim_events import (
    build_jobs as build_v3_jobs, validate_pairs as validate_v3_pairs,
)
from scripts import v18_gated


BASE = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
VALUE_FLAGS = {
    "--supply_eta_v3", "--supply_eta_within", "--stock_rule",
    "--stock_adjustment_rate", "--stock_floor", "--active_driver_feature_reference",
    "--pool", "--train_customers", "--eval_customers", "--loyalty_pool_size",
    "--demand_calibration_artifact", "--driver_pay_rule", "--pay_rule_json",
    "--fleet_scale", "--base_pickup_minutes", "--supply_base_active_drivers",
    "--supply_hour_profile_json", "--supply_reference_earnings",
}


def _frozen():
    config = copy.deepcopy(load_config())
    config["status"] = "frozen"
    config["decisions"].update({"N_star": 150, "ppo_clock_days": 1200,
        "mc_clock_days": 1200, "workers": 1, "projection": {"wave_a_hours": 1, "source": "test"}})
    return config


def _market(command):
    result = {}
    values = list(map(str, command))
    for flag in VALUE_FLAGS:
        if flag in values:
            result[flag] = values[values.index(flag) + 1]
    for flag in ("--v3_daily_stock", "--no-v3_daily_stock"):
        if flag in values:
            result[flag] = True
    return result


def _stored_command(path: Path):
    return json.loads(path.read_text())["identity"]["command"]


def test_production_flags_equal_p2_v17_model():
    path = BASE / "runs/v18/preflight/P2_fix/effects/v17_model/C0_calm/adaptive_best_response_memory/manifest.json"
    stored = _stored_command(path)
    rebuilt, _ = market_flags(stored, variant="production", city="nyc", customers=500)
    assert _market(rebuilt) == _market(stored)


def test_s3_flags_equal_registered_p2_calibrated_arm():
    path = BASE / "runs/v18/preflight/P2_fix/effects/calibrated/C0_calm/adaptive_best_response_memory/manifest.json"
    stored = _stored_command(path)
    rebuilt, _ = market_flags(stored, variant="s3", city="nyc", customers=500)
    assert _market(rebuilt) == _market(stored)


def test_training_arm_flags(tmp_path):
    config = _frozen()
    main = train_args(config, city="nyc", seed=101, arm="main", output=tmp_path / "main")
    assert main[main.index("--v16_policy_mode") + 1] == "q"
    for flag in ("--v16_branch_horizon_days", "--v16_value_head", "--v17_move_radius",
                 "--v17_temperature_mode", "--v17_tau_rel_start", "--v17_tau_rel_end"):
        assert flag in main
    assert train_args(config, city="nyc", seed=101, arm="mc_clock", output=tmp_path / "mc").count("mc_return") == 1
    assert train_args(config, city="nyc", seed=101, arm="nocrn", output=tmp_path / "nocrn").count("independent") == 1
    for arm in ("ppo", "ppo_clock"):
        values = train_args(config, city="nyc", seed=101, arm=arm, output=tmp_path / arm)
        assert values[values.index("--v16_policy_mode") + 1] == "ppo"
        assert values[values.index("--ppo_rollout_episodes") + 1] == "1"
        assert values[values.index("--ppo_min_rollout_transitions") + 1] == "30"
        assert values[values.index("--ppo_batch_size") + 1] == "32"
    no_driver = train_args(config, city="nyc", seed=101, arm="no_driver_obs", output=tmp_path / "no_driver")
    assert "--v17_ablate_driver_observation" in no_driver
    assert "--current_active_driver_observation" not in no_driver


def test_evaluation_policy_kinds_do_not_leak_value_flags(tmp_path):
    config = _frozen()
    scenario = BASE / "scenarios/calm.json"
    reference = eval_args(config, city="nyc", case="C0", specs=("opening_hold",),
                          suite="Z3", output=tmp_path / "ref", scenario=scenario)
    assert "--v16_value_head" not in reference
    main = eval_args(config, city="nyc", case="C0", specs=("rl_actor",), suite="Z3",
                     output=tmp_path / "main", scenario=scenario, kind="main",
                     policy=tmp_path / "main.pt")
    assert main[main.index("--v16_policy_mode") + 1] == "q"
    assert main[main.index("--v16_value_head") + 1] == "concave2d"
    assert main[main.index("--v16_store_cap") + 1] == "20000"
    ppo = eval_args(config, city="nyc", case="C0", specs=("rl_actor",), suite="Z3",
                    output=tmp_path / "ppo", scenario=scenario, kind="ppo",
                    policy=tmp_path / "ppo.pt")
    assert ppo[ppo.index("--v16_policy_mode") + 1] == "ppo"
    assert "--v16_value_head" not in ppo and "--v16_store_cap" not in ppo


def test_cards_are_static_tariffs():
    cards = card_grid()
    assert len(cards) == 36
    assert all(row["spec"].startswith("static_tariff:") for row in cards)


def test_grid_edge_selected_is_a_boolean_membership_check():
    assert _grid_edge_selected({"distance": 2.5, "time": 1.0}, (1.0, 1.0)) is True
    assert _grid_edge_selected({"distance": 1.0, "time": 1.0}, (1.25, 2.5)) is True
    assert _grid_edge_selected({"distance": 1.0, "time": 1.0}, (1.25, 1.5)) is False


def test_selection_plan_matches_registered_m4_counts():
    plan = _selection_plan()
    assert len(plan["cards"]) == 36
    assert plan["openings"] == 2
    assert plan["stage3_count_rules"] == 14
    assert plan["stage3_reactive_rules"] == 24


def test_registered_scenario_paths_exist():
    assert all(path.is_file() for path in SCENARIOS.values())


def test_e4_uses_recalibrated_e1_rows_and_has_no_credit_probe(tmp_path):
    jobs = build_evaluation_jobs(
        _frozen(), suite="E4", root=tmp_path / "eval",
    )
    assert len(jobs) == 40
    assert all("--v17_eval_credit_probe" not in job.command for job in jobs)
    assert all("--v17_eval_credit_horizons" not in job.command for job in jobs)
    assert all(
        "--v17_eval_credit_rival_planner" not in job.command for job in jobs
    )
    assert all(
        "--v17_eval_credit_probe_out" not in job.command for job in jobs
    )
    e1_rows = [
        job for job in jobs
        if job.name.rsplit("/", 1)[-1] in {"nyc_s101", "nyc_s202"}
    ]
    assert len(e1_rows) == 8
    assert all("--recalibrate_supply_on_eval" in job.command for job in e1_rows)
    assert all("E1_recalibrated_supply" in str(job.output) for job in e1_rows)
    wave_b = [job for job in jobs if job not in e1_rows]
    assert len(wave_b) == 32
    assert all("--recalibrate_supply_on_eval" not in job.command for job in wave_b)
    assert all("E4_no_probe" in str(job.output) for job in wave_b)
    assert all("/E4/" not in str(job.output) for job in jobs)


def test_t8_is_the_only_probe_suite_and_is_diagnostic(tmp_path):
    jobs = build_evaluation_jobs(
        _frozen(), suite="T8", root=tmp_path / "eval",
    )
    assert len(jobs) == 2
    assert all("--v17_eval_credit_probe" in job.command for job in jobs)
    assert all("--v17_eval_credit_probe_out" in job.command for job in jobs)


def test_recalibrate_supply_on_eval_ignores_only_the_saved_operating_point():
    saved = {"city": "New York City", "calibration": {"sample_size": 150}}
    artifact = {"supply_operating_point": saved}
    assert _evaluation_supply_operating_point(
        trained_model_in="policy.pt", artifact_header=artifact,
        city_artifact=None, eval_only=True,
        recalibrate_supply_on_eval=False,
    ) is saved
    assert _evaluation_supply_operating_point(
        trained_model_in="policy.pt", artifact_header=artifact,
        city_artifact=None, eval_only=True,
        recalibrate_supply_on_eval=True,
    ) is None
    with np.testing.assert_raises_regex(ValueError, "evaluation-only"):
        _evaluation_supply_operating_point(
            trained_model_in="policy.pt", artifact_header=artifact,
            city_artifact=None, eval_only=False,
            recalibrate_supply_on_eval=True,
        )


def test_gated_actor_hysteresis_scripted_driver_sequence():
    gate_open = False
    streak = 0
    observed = []
    for ratio in (0.85, 0.69, 0.75, 0.81, 0.82, 0.68, 0.85, 0.86):
        gate_open, streak = _gated_rl_transition(
            gate_open=gate_open, off_streak=streak, ratio=ratio,
            theta_on=0.7, theta_off=0.8, off_days=2,
        )
        observed.append((gate_open, streak))
    assert observed == [
        (False, 0), (True, 0), (True, 0), (True, 1),
        (False, 0), (True, 0), (True, 1), (False, 0),
    ]
    assert _gated_rl_transition(
        gate_open=False, off_streak=0, ratio=0.0,
        theta_on=-1.0, theta_off=-0.9, off_days=2,
    ) == (False, 0)
    assert _gated_rl_transition(
        gate_open=False, off_streak=0, ratio=1.0,
        theta_on=99.0, theta_off=99.1, off_days=2,
    ) == (True, 0)


def test_gated_selection_and_evaluation_plans(monkeypatch, tmp_path):
    config = _frozen()
    selection = v18_gated.selection_jobs(config)
    assert len(selection) == 18
    assert len({job.name for job in selection}) == 18
    assert all("--trained_model_in" in job.command for job in selection)
    assert all("--recalibrate_supply_on_eval" in job.command for job in selection)
    assert all("--run_experiment" not in job.command for job in selection)
    assert all(
        job.command[job.command.index("--policy_sweep_seed") + 1] == "20231"
        for job in selection
    )
    specs = {
        city: v18_gated.gated_spec(city, 0.8)
        for city in ("nyc", "boston", "chicago")
    }
    monkeypatch.setattr(v18_gated, "REPO", tmp_path)
    for city, spec in specs.items():
        path = tmp_path / f"runs/v18/select/{city}/gated.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"state": "complete", "spec": spec}))
    evaluations = v18_gated.build_jobs(config)
    assert len(evaluations) == 90
    assert sum(job.name.startswith("H1/") for job in evaluations) == 45
    assert sum(job.name.startswith("H2/") for job in evaluations) == 36
    assert sum(job.name.startswith("H5/") for job in evaluations) == 9
    assert all("--trained_model_in" in job.command for job in evaluations)
    assert all("--recalibrate_supply_on_eval" in job.command for job in evaluations)
    assert all("--run_experiment" not in job.command for job in evaluations)
    assert all("/runs/v18/eval/H" in str(job.output) for job in evaluations)


def test_gated_queue_precedes_sensitivity():
    source = (BASE / "scripts/v18_run_all.py").read_text(encoding="utf-8")
    assert source.index('"H", REPO / "scripts/v18_gated.py"') < source.index(
        '"S", REPO / "scripts/v18_sensitivity.py"'
    )
    sensitivity = (BASE / "scripts/v18_sensitivity.py").read_text(encoding="utf-8")
    assert "ensure_gated_evaluation(args.workers)" in sensitivity


def test_v3_event_twins_are_declared_paired_and_request_logged(tmp_path):
    validate_v3_pairs()
    jobs = build_v3_jobs(_frozen(), root=tmp_path / "v3")
    assert len(jobs) == 126
    assert len({job.name for job in jobs}) == 126
    assert all("--scenario_hazards" in job.command for job in jobs)
    assert all(
        job.command[job.command.index("--scenario_hazards") + 1] == "off"
        for job in jobs
    )
    assert all("--policy_sweep_request_log" in job.command for job in jobs)
    assert all(
        job.command[job.command.index("--policy_sweep_openings") + 1] == "2"
        for job in jobs
    )
    assert all(
        job.command[job.command.index("--policy_sweep_rivals") + 1]
        == "adaptive_best_response_memory,profit_best_response_myopic,"
           "profit_best_response_forward,profit_opt_mpc_grid,pi_price_gap"
        for job in jobs
    )


def test_count_rule_norm_shift_and_candidate_counts():
    original = [10, 20, 30, 40, 50, 60]
    shifted = [original[(block - 1) % 6] for block in range(6)]
    rules = _stage3_count_rules((0.8, 2.0), (1.5, 1.5), shifted)
    assert shifted == [60, 10, 20, 30, 40, 50]
    assert len(rules) == 14
    assert all(json.loads(rule.split(":", 1)[1])["norm"] == shifted for rule in rules)
    assert len(reactive_candidates((0.8, 2.0))) == 24


def test_retry_and_dependency_skip(tmp_path):
    counter = tmp_path / "count"
    output = tmp_path / "ok.json"
    code = (
        "from pathlib import Path; import sys; "
        f"p=Path({str(counter)!r}); n=int(p.read_text())+1 if p.exists() else 1; p.write_text(str(n)); "
        f"o=Path({str(output)!r}); o.write_text('{{}}') if n==3 else None; sys.exit(0 if n==3 else 9)"
    )
    job = Job("retry", (str(PYTHON), "-c", code), output, tmp_path / "run.log", tmp_path / "manifest.json")
    assert run_job(job, cwd=BASE, max_attempts=3) == "complete"
    assert counter.read_text() == "3"
    bad = Job("bad", (str(PYTHON), "-c", "raise SystemExit(4)"), tmp_path / "bad.out",
              tmp_path / "bad.log", tmp_path / "bad.manifest")
    child = Job("child", (str(PYTHON), "-c", "raise SystemExit(0)"), tmp_path / "child.out",
                tmp_path / "child.log", tmp_path / "child.manifest", requires=("bad",))
    states = run_jobs((bad, child), cwd=BASE, workers=1, progress=tmp_path / "progress.jsonl")
    assert states == {"bad": "failed", "child": "skipped"}
    assert run_job(bad, cwd=BASE, max_attempts=3) == "failed"
    exhausted = json.loads(bad.manifest.read_text(encoding="utf-8"))
    assert exhausted["state"] == "failed"
    assert exhausted["returncode"] == 4
    assert len(exhausted["attempts"]) == 3


def test_clear_quote_is_rng_free():
    class Market:
        def __init__(self):
            self.rng = np.random.default_rng(42)
        def quote_price(self, distance, duration, context, overrides=None):
            assert context.weather == "clear" and overrides is None
            return distance + duration
    market = Market()
    before = json.dumps(market.rng.bit_generator.state, sort_keys=True)
    context = RideContext(1, "rain", 8, False, "standard")
    assert _v18_card_clear_quote(market, 2.0, 7.0, context) == 9.0
    assert json.dumps(market.rng.bit_generator.state, sort_keys=True) == before


def test_request_step_is_explicit_and_has_a_deterministic_fallback():
    assert "request_step" in inspect.signature(Core.simulate_batch).parameters
    assert _v18_request_step(17, None) == 17
    assert _v18_request_step(None, (3, 4, 0)) == 4
    assert _v18_request_step(None, None) == 0
