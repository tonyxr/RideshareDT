"""Regime process: common random numbers, snapshot/restore, scripted scenarios."""
import datetime as dt
import numpy as np
from market_shocks import ShockConfig, ShockProcess, ShockScenario, ShockEvent, ShockEffect


def _run(process, days=40, steps=6):
    process.reset()
    rows = []
    for _ in range(days):
        process.advance_day()
        rows.append([process.effect(b).as_dict() for b in range(steps)])
    return rows


def test_same_seed_same_path_and_restore_roundtrip():
    cfg = ShockConfig(enabled=True, mode="stress", intensity_scale=1.0)
    a = ShockProcess(cfg, seed=11, steps_per_day=6)
    b = ShockProcess(cfg, seed=11, steps_per_day=6)
    c = ShockProcess(cfg, seed=12, steps_per_day=6)
    ra, rb, rc = _run(a), _run(b), _run(c)
    assert ra == rb
    assert ra != rc
    snap = a.snapshot()
    before = [a.effect(k).as_dict() for k in range(6)]
    a.advance_day(); a.advance_day()
    a.restore(snap)
    after = [a.effect(k).as_dict() for k in range(6)]
    assert before == after


def test_scripted_shortage_scenario_changes_fleet_only_in_window():
    cfg = ShockConfig(enabled=True, shortage_rate_per_day=0.0, storm_daily_hazard={}, event_rate_per_day=0.0,
                      outage_rate_per_day=0.0, transit_disruption_rate_per_day=0.0,
                      rival_promotion_rate_per_day=0.0, drift_sd=0.0)
    proc = ShockProcess(cfg, seed=1, steps_per_day=6)
    proc.scenario = ShockScenario(start_date=dt.date(2023, 3, 6),
                                  events=[ShockEvent("driver_shortage", 2, 4, intensity=0.65, announced=False)])
    proc.reset()
    fleet = []
    oracle_flags = []
    for _ in range(10):
        proc.advance_day()
        effect = proc.effect(3)
        fleet.append(effect.fleet_multiplier)
        oracle_flags.append(effect.public.get("driver_shortage_active", 0.0))
    assert fleet[0] == 1.0 and fleet[1] == 1.0
    assert min(fleet[2:5]) < 0.7
    assert oracle_flags[0:2] == [0.0, 0.0]
    assert oracle_flags[2:5] == [1.0, 1.0, 1.0]
    assert oracle_flags[5] == 0.0
    # recovery tail returns toward 1.0 but does not overshoot
    assert all(f <= 1.0 + 1e-9 for f in fleet)
    assert fleet[-1] > fleet[3]


def test_unknown_event_kind_is_carried_in_phases_without_physical_effect():
    cfg = ShockConfig(enabled=True, shortage_rate_per_day=0.0, storm_daily_hazard={}, event_rate_per_day=0.0,
                      outage_rate_per_day=0.0, transit_disruption_rate_per_day=0.0,
                      rival_promotion_rate_per_day=0.0, drift_sd=0.0)
    proc = ShockProcess(cfg, seed=1, steps_per_day=6)
    proc.scenario = ShockScenario(start_date=dt.date(2023, 3, 6),
                                  events=[ShockEvent("rival_switch", 3, 3, detail={"mode": "pi_price_gap"})])
    proc.reset()
    for _ in range(4):
        proc.advance_day()
    effect = proc.effect(0)
    assert effect.event_phases.get("rival_switch") in {"peak", "ramp"}
    assert effect.fleet_multiplier == 1.0 and effect.volume_multiplier == 1.0


def test_training_shortage_curriculum_has_requested_prevalence_depth_and_coverage():
    cfg = ShockConfig(
        enabled=True,
        types=("driver_shortage",),
        shortage_rate_per_day=1.0,  # must be suppressed during training
        training_shortage_episode_share=0.70,
        training_shortages_per_episode=2,
        training_shortage_duration_days=(6, 10),
        training_shortage_fleet=(0.40, 0.60),
    )
    selected = 0
    active_days = 0
    episodes = 400
    for seed in range(episodes):
        proc = ShockProcess(cfg, seed=seed, steps_per_day=6)
        proc.reset(
            episode_seed=seed,
            training_opening=True,
            training_episode_days=30,
        )
        events = [
            event for event in proc.events
            if event.kind == "driver_shortage"
        ]
        if proc.training_shortage_curriculum_selected:
            selected += 1
            assert len(events) == 2
            assert events[0].end_day < events[1].start_day
            for event in events:
                duration = event.end_day - event.start_day + 1
                assert 6 <= duration <= 10
                assert 0.40 <= event.intensity <= 0.60
                assert event.detail["training_curriculum"] is True
                active_days += duration
        else:
            assert events == []
        for _ in range(30):
            proc.advance_day()
        # Even a 100% natural hazard cannot add an uncontrolled third event.
        assert sum(e.kind == "driver_shortage" for e in proc.events) <= 2
    prevalence = selected / episodes
    assert 0.60 <= prevalence <= 0.80
    assert active_days / (episodes * 30) >= 0.15
