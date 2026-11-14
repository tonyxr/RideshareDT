from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import numpy as np

from Core import _v18_card_clear_quote
from MarketInteraction import RideContext
from scripts.v18_config import eval_args, load_config, train_args
from scripts.v18_market import market_flags
from scripts.v18_runtime import Job, run_job, run_jobs
from scripts.v18_select import _stage3_count_rules, card_grid, reactive_candidates


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
