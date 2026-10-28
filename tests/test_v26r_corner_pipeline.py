import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from Core import Core, load_shock_scenario
from market_shocks import ShockConfig, ShockProcess, ShockEffect
from platform_mdp import TrainingStageScheduler
from scripts import v26r_case_eval as runner
from scripts.v26r_cases import CASES, static_specs
from scripts.v26r_case_matrix import values, paired, select
from scripts.run_v26r_staged import BASE, command


def process(case):
    cfg = ShockConfig(enabled=True, storm_daily_hazard={}, event_rate_per_day=0,
        shortage_rate_per_day=0, outage_rate_per_day=0, transit_disruption_rate_per_day=0,
        rival_promotion_rate_per_day=0, drift_sd=0)
    p = ShockProcess(cfg, seed=45, steps_per_day=6)
    p.scenario = load_shock_scenario(str(BASE / CASES[case]["scenario"]))
    p.reset()
    return p


@pytest.mark.parametrize("case", list(CASES))
def test_every_case_has_reproducible_finite_physical_path(case):
    a, b = process(case), process(case)
    for day in range(40):
        a.advance_day(); b.advance_day()
        for block in range(6):
            ea, eb = a.effect(block), b.effect(block)
            assert ea.as_dict() == eb.as_dict()
            assert np.isfinite([ea.fleet_multiplier, ea.volume_multiplier, ea.duration_multiplier,
                                ea.firm2_price_multiplier, ea.reservation_wage_multiplier]).all()
            assert ea.fleet_multiplier > 0 and ea.volume_multiplier > 0
    snapshot = a.snapshot()
    before = a.effect(3).as_dict()
    a.advance_day(); a.restore(snapshot)
    assert a.effect(3).as_dict() == before


def test_storm_warning_precedes_physical_storm_and_transit_loss():
    p = process("A3_storm_transit")
    daily = []
    for _ in range(30):
        p.advance_day()
        daily.append((p.effect(3), p.day_weather_override()))
    assert daily[15][0].public["storm_warning"] == 1
    assert daily[15][1] is None
    assert daily[16][1] == "snow"
    assert daily[16][0].transit_availability == 0
    assert daily[16][0].fleet_multiplier < 1
    assert daily[-1][0].fleet_multiplier == 1


@pytest.mark.parametrize("case,firm", [("A5_rival_outage", "Firm2"), ("S5_own_outage_loyalty", "Firm1")])
def test_outage_hits_the_named_firm_only(case, firm):
    p = process(case)
    effects = []
    for _ in range(20):
        p.advance_day(); effects.append(p.effect(3))
    other = "Firm1" if firm == "Firm2" else "Firm2"
    assert min(e.firm_capacity_multiplier[firm] for e in effects) < 1
    assert all(e.firm_capacity_multiplier[other] == 1 for e in effects)


def test_rival_switch_is_applied_exactly_once_at_onset():
    p = process("S4_rival_switch")
    core = Core.__new__(Core)
    core.regime_market = True
    core.enable_driver_supply = False
    core.market = SimpleNamespace(active_shocks=p)
    core.firm2 = SimpleNamespace()
    core._regime_effect = lambda hour: ShockEffect()
    core._regime_rival_switch_applied = {}
    switches = []
    core._set_firm2_mode_preserving_tariff = lambda mode, seed: switches.append((p.day_index, mode))
    for _ in range(40):
        p.advance_day(); core._regime_day_start([0], episode_seed=10)
    assert switches == [(16, "pi_price_gap")]


def test_case_cache_rejects_other_policy_or_unversioned_evidence(tmp_path):
    runner.ensure_manifest(tmp_path, {"policy_sha256": "one"})
    runner.ensure_manifest(tmp_path, {"policy_sha256": "one"})
    with pytest.raises(ValueError, match="Incompatible"):
        runner.ensure_manifest(tmp_path, {"policy_sha256": "two"})
    legacy = tmp_path / "legacy"; legacy.mkdir()
    (legacy / "old.json").write_text("{}")
    with pytest.raises(ValueError, match="Unversioned"):
        runner.ensure_manifest(legacy, {})


def test_single_invocation_runs_static_then_oracle_rl_and_myopic(tmp_path, monkeypatch):
    policy = tmp_path / "policy.pt"; policy.write_bytes(b"test identity only")
    calls = []
    def fake_run(city, slug, case, specs, seed_set, openings, channels, artifact, out_dir, **kw):
        calls.append((seed_set, list(specs), artifact))
        rival = 'adaptive_best_response_memory'
        for spec in specs:
            payload = dict(spec=spec, case=case, seed_set=seed_set, openings=openings,
                           horizon_steps=240, steps_per_day=6, rollouts=[[f'{rival}__opening_1', rival]],
                           results={}, traces=[dict(rollout=f'{rival}__opening_1', step=t,
                                                   regime='driver_shortage' if 14 <= t//6 <= 19 else 'calm',
                                                   reward=1, profit_per_request=1) for t in range(240)])
            runner.atomic_json(Path(out_dir) / (runner.spec_key(spec)+".json"), payload)
    monkeypatch.setattr(runner, "run_specs", fake_run)
    monkeypatch.setattr("sys.argv", ["case_eval", "Chicago", "chicago", "A1_shortage_severe",
        "--seed-set", "both", "--output-root", str(tmp_path/"cases"), "--policy", str(policy),
        "--rivals", "adaptive_best_response_memory", "--budget-seconds", "1000"])
    runner.main()
    for seed_set in ("val", "eval"):
        done = [s for seed, specs, _ in calls if seed == seed_set for s in specs]
        assert set(static_specs()).issubset(done)
        assert "rl_actor" in done and "myopic_greedy" in done
        assert any(s.startswith("oracle:") for s in done)
    assert all(artifact == str(policy) for _, _, artifact in calls)


def test_per_rival_selection_and_pairing_do_not_pool_rivals():
    def trace(a, b):
        return [dict(rollout=f"{r}__opening_1", step=step, profit_per_request=v)
                for r, v in (("A", a), ("B", b)) for step in range(60, 240)]
    sweep = {"traces": {"static:1": trace(3, 0), "static:2": trace(1, 5)}}
    assert select(sweep, "static:", "A", 10, 40) == "static:1"
    assert select(sweep, "static:", "B", 10, 40) == "static:2"
    assert values(trace(3, 5), "A", "profit_per_request", 10, 40) == {"A__opening_1": 3}
    incomplete = trace(3, 5)
    incomplete.pop(0)
    with pytest.raises(ValueError, match='Missing or duplicate'):
        values(incomplete, "A", "profit_per_request", 10, 40)
    with pytest.raises(ValueError, match="unmatched"):
        paired({"a": 1}, {"b": 1})


@pytest.mark.parametrize("city,slug", [("Chicago", "chicago"), ("Boston", "boston"), ("New York City", "nyc")])
def test_city_training_command_keeps_budget_and_save_contract(tmp_path, city, slug):
    cmd = command(tmp_path, days=20000, city=city, slug=slug)
    assert cmd[cmd.index("--market")+1] == city
    assert cmd[cmd.index("--train_timesteps")+1] == "20000"
    assert f"/{slug}/" in cmd[cmd.index("--threshold_cache_path")+1]
    assert "--trained_model_in" not in cmd
    assert "--trained_model_out" in cmd and "--benchmark_suite" in cmd
    assert cmd[cmd.index("--ppo_min_rollout_transitions")+1] == "180"


def test_twenty_thousand_day_allocation_has_complete_opponent_batches():
    from collections import Counter
    scheduler = TrainingStageScheduler(stage_fractions=(.1, .108, .792), dynamic_episode_days=60)
    counts = Counter(scheduler.stage_at(d/20000).opponent_regime for d in range(20000))
    assert counts == dict(market=2000, static=2160, dynamic=15840)
    assert counts["dynamic"] % 180 == 0
    assert counts["dynamic"] // 180 == 88
