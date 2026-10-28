"""Revision 15d hybrid actions, dense PPO updates, and long credit wiring."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from Core import Core
from platform_mdp import SEGMENTS, PlatformObservationModel, TrainingStageScheduler
from pricing_models import (
    HYBRID_RELATIVE_ACTION_SEMANTICS,
    CoefficientOverrides,
    FirmRLPricer,
)
from scripts import run_revision15d_multicity_full as r15d
from scripts import summarize_revision15b_benchmarks as benchmark_summary
from scripts import v26r_case_eval


KEYS = ["base_fare", "per_minute", "per_mile", "booking_fee", "airport_fee"]


class Market:
    def __init__(self, anchors):
        self.curr_market = type("Anchors", (), dict(anchors))()


@pytest.fixture
def anchors():
    path = r15d.inherited.ROOT / "data/processed/nyc/calibration.json"
    tariff = json.loads(path.read_text())["calibration"]["tariff"]
    return {key: float(tariff[key]) for key in KEYS}


def make_pricer(keys=KEYS):
    return FirmRLPricer(
        seed=15,
        opt_keys=list(keys),
        independent_coefficient_actions=False,
        hybrid_relative_actions=True,
        relative_action_steps=(0.1, 0.2),
        learning_corrections=False,
    )


def find_action(pricer, target, step):
    return next(
        action
        for action, action_target in pricer.action_to_target_key.items()
        if action_target == target
        and pricer.action_to_relative_step[action] == pytest.approx(step)
    )


def test_hybrid_action_dimensions_labels_and_factorization():
    nyc = make_pricer(KEYS)
    boston = make_pricer(KEYS[:4])
    assert nyc.hybrid_relative_actions
    assert nyc.agent.action_dim == 25
    assert boston.agent.action_dim == 21
    assert nyc.action_label(0) == "hold"
    assert [nyc.action_label(action) for action in range(1, 5)] == [
        "whole_tariff_step:-0.20",
        "whole_tariff_step:-0.10",
        "whole_tariff_step:+0.10",
        "whole_tariff_step:+0.20",
    ]
    assert nyc.agent.net.pi_target_head is not None
    assert nyc.agent.net.pi_magnitude_head is not None
    assert nyc.action_to_multiplier == {}
    assert "airport_fee" not in boston.action_keys


def test_relative_steps_accumulate_instead_of_jumping_to_absolute_targets(anchors):
    pricer = make_pricer()
    market = Market(anchors)
    action = find_action(pricer, "per_mile", 0.2)
    start = anchors["per_mile"]
    pricer.apply_action(action, market)
    assert pricer.overrides.per_mile == pytest.approx(start * 1.2)
    pricer.apply_action(action, market)
    assert pricer.overrides.per_mile == pytest.approx(start * 1.4)
    assert pricer.last_action_descriptor.magnitude_level == pytest.approx(0.2)


def test_whole_tariff_and_individual_steps_have_distinct_scope(anchors):
    pricer = make_pricer()
    market = Market(anchors)
    whole = find_action(pricer, "whole_tariff", 0.1)
    pricer.apply_action(whole, market)
    for key in KEYS:
        lb, ub = pricer.config.bounds[key]
        assert getattr(pricer.overrides, key) == pytest.approx(
            np.clip(anchors[key] * 1.1, lb, ub)
        )

    pricer.overrides = CoefficientOverrides()
    single = find_action(pricer, "base_fare", -0.2)
    pricer.apply_action(single, market)
    assert pricer.overrides.base_fare == pytest.approx(anchors["base_fare"] * 0.8)
    assert all(getattr(pricer.overrides, key) is None for key in KEYS[1:])


def test_cap_aware_mask_removes_only_true_no_ops(anchors):
    pricer = make_pricer()
    market = Market(anchors)
    for key in KEYS:
        setattr(pricer.overrides, key, pricer.config.bounds[key][1])
    mask = pricer.feasible_action_mask(market)
    assert mask[0]
    for action, step in pricer.action_to_relative_step.items():
        assert bool(mask[action]) is (step < 0.0)
    for action in np.flatnonzero(mask):
        before = copy.deepcopy(pricer.overrides)
        pricer.apply_action(int(action), market)
        if action:
            assert not pricer.last_action_was_zero_effect
        pricer.overrides = before


def test_action_features_encode_signed_relative_magnitude(anchors):
    pricer = make_pricer()
    market = Market(anchors)
    standalone = pricer.build_action_feature_matrix(market, {})
    assert standalone.shape == (25, 20)
    assert np.isfinite(standalone).all()

    observer = PlatformObservationModel.__new__(PlatformObservationModel)
    observer.quote_probes = {
        segment: {"gap": 0.0, "uncertainty": 1.0} for segment in SEGMENTS
    }
    observer.config = type("Config", (), {"gap_scale_dollars": 2.0})()
    observer.action_feature_dim = PlatformObservationModel.action_feature_dim
    own = dict(anchors)
    matrix = observer.build_action_features(
        action_steps=pricer.action_to_steps,
        action_keys=pricer.action_keys,
        own_coefficients=own,
        anchor_coefficients=anchors,
        coefficient_steps=dict(pricer.config.step),
        coefficient_bounds=dict(pricer.config.bounds),
        step_scale=1.0,
        target_gap=0.0,
        action_target_keys=pricer.action_to_target_key,
        action_target_relative_steps=pricer.action_to_relative_step,
    )
    plus_one = find_action(pricer, "per_mile", 0.1)
    plus_two = find_action(pricer, "per_mile", 0.2)
    minus_two = find_action(pricer, "per_mile", -0.2)
    assert matrix.shape == (25, observer.action_feature_dim)
    assert matrix[plus_one, 1] > 0 and matrix[minus_two, 1] < 0
    # Four segment-fare impact channels start after the first ten columns.
    assert np.linalg.norm(matrix[plus_two] - matrix[0]) > np.linalg.norm(
        matrix[plus_one] - matrix[0]
    )


@pytest.mark.parametrize(
    ("city", "coefficients", "actions"),
    (("nyc", 5, 25), ("boston", 4, 21), ("chicago", 5, 25)),
)
def test_multicity_command_has_hybrid_dense_long_horizon_contract(
    city, coefficients, actions
):
    out = Path("/tmp") / f"revision15d-contract-{city}"
    command = r15d.build_training_command(city, out, 4000, 251015)
    assert "--hybrid_relative_actions" in command
    assert "--tariff_targets" not in command
    assert "--level_step_actions" not in command
    assert r15d.option(command, "--relative_action_steps") == "0.1,0.2"
    assert r15d.option(command, "--manipulated_price_coefficients") == str(coefficients)
    assert r15d.option(command, "--nmcc_branches") == str(actions)
    assert r15d.option(command, "--nmcc_horizon_days") == "12"
    assert r15d.option(command, "--nmcc_every_k_decisions") == "3"
    assert r15d.option(command, "--training_dynamic_episode_days") == "14"
    assert r15d.option(command, "--ppo_rollout_episodes") == "2"
    assert r15d.option(command, "--ppo_min_rollout_transitions") == "28"
    assert r15d.option(command, "--ppo_update_interval_days") == "2"
    assert r15d.option(command, "--ppo_batch_size") == "14"
    assert r15d.option(command, "--ppo_update_epochs") == "12"


def test_update_contract_retains_two_complete_long_trajectories():
    scheduler = TrainingStageScheduler(
        stage_fractions=(0.05, 0.05, 0.375, 0.525),
        dynamic_episode_days=r15d.TRAJECTORY_DAYS,
    )
    assert all(
        stage.episode_days in (0, r15d.TRAJECTORY_DAYS)
        for stage in scheduler.stages
    )
    assert r15d.MIN_ROLLOUT_TRANSITIONS == 28
    assert r15d.ROLLOUT_EPISODES == 2
    assert r15d.PPO_UPDATE_INTERVAL_DAYS < r15d.TRAJECTORY_DAYS
    effective_actions = 1.0 / (
        1.0 - r15d.PPO_GAMMA**6 * r15d.PPO_GAE_LAMBDA
    )
    assert effective_actions >= 7.0


def test_manifest_records_revision15d_contract(tmp_path):
    root = tmp_path / "manifest"
    r15d.prepare_root(root, 4000, 251015)
    payload = json.loads((root / "manifest.json").read_text())
    assert payload["revision"] == "15d"
    assert payload["action_contract"]["joint_actions_including_hold_by_city"] == {
        "nyc": 25,
        "boston": 21,
        "chicago": 25,
    }
    assert payload["action_contract"]["no_op_cap_actions_masked"] is True
    assert payload["learning_contract"]["trajectory_days"] == 14
    assert payload["learning_contract"]["nmcc_horizon_days"] == 12
    assert "heldout_evaluation_opponent_groups" not in payload
    assert "heldout_cases" not in payload
    assert payload["evaluation_contract"]["training_exposed_opponents"] == list(
        benchmark_summary.TRAINING_EXPOSED_RIVALS
    )
    assert payload["evaluation_contract"]["genuinely_held_out_opponents"] == list(
        benchmark_summary.GENUINELY_HELD_OUT_RIVALS
    )


def test_memory_rival_credit_horizon_covers_full_response_window():
    contract = r15d.learning_contract()
    memory = contract["memory_rival_response_window_days"]
    assert memory["total"] == pytest.approx(
        memory["reaction_lag"] + memory["retaliation_hold"] + memory["recovery"]
    )
    assert contract["nmcc_horizon_days"] > memory["total"]


def test_nmcc_cadence_includes_opening_and_every_third_decision():
    due = [
        index
        for index in range(1, 15)
        if Core._nmcc_branch_due(index, r15d.NMCC_EVERY_K_DECISIONS)
    ]
    assert due == [1, 4, 7, 10, 13]


def test_frozen_actor_queries_policy_exactly_once():
    assert Core._learned_actor_due_for_policy(None, 0)
    assert Core._learned_actor_due_for_policy(None, 9)
    assert Core._learned_actor_due_for_policy("rl_actor_frozen", 0)
    assert not Core._learned_actor_due_for_policy("rl_actor_frozen", 1)
    assert not Core._learned_actor_due_for_policy("opening_hold", 0)


def test_evaluation_separates_exposure_and_adds_perturbed_holdouts():
    exposed = set(benchmark_summary.TRAINING_EXPOSED_RIVALS)
    held_out = set(benchmark_summary.GENUINELY_HELD_OUT_RIVALS)
    perturbed = set(benchmark_summary.PERTURBED_HELD_OUT_RIVALS)
    assert len(exposed) == 12
    assert len(held_out) == 6
    assert len(perturbed) == 5
    assert not (exposed & held_out)
    assert not (exposed & perturbed)
    assert all(name.endswith("@perturbed0") for name in perturbed)
    assert all(
        benchmark_summary.RIVAL_EXPOSURE[name]
        == "genuinely_held_out_perturbed"
        for name in perturbed
    )


def test_full_evaluation_includes_adaptive_frozen_and_fixed_controls(tmp_path):
    command = r15d.evaluation_command("nyc", tmp_path, tmp_path / "policy.pt")
    specs = r15d.option(command, "--specs").split(";")
    rivals = r15d.option(command, "--rivals").split(",")
    assert specs == list(r15d.evaluation_specs("nyc"))
    assert {"rl_actor", "rl_actor_frozen", "opening_hold"} <= set(specs)
    assert {f"static:{target:g}" for target in r15d.STATIC_GRID} <= set(specs)
    assert len([spec for spec in specs if spec.startswith("static_tariff:@")]) == 1
    assert rivals == list(benchmark_summary.RIVALS)


def test_multispec_summary_preserves_policy_and_exposure_labels(tmp_path):
    specs = (
        "rl_actor", "rl_actor_frozen", "opening_hold", "static:1",
        "static_tariff:@revision15b.json",
    )
    traces = {}
    for spec_index, spec in enumerate(specs):
        rows = []
        for rival in benchmark_summary.RIVALS:
            rollout = f"{rival}__opening_1"
            for step in range(40):
                rows.append({
                    "rollout": rollout,
                    "step": step,
                    "reward": 1.0 - 0.1 * spec_index,
                    "own_share": 0.45,
                    "rival_share": 0.40,
                    "profit_per_request": 2.0,
                    "rival_profit_per_request": 1.5,
                    "profit_lead": 0.5,
                    "revenue_per_request": 8.0,
                    "rival_revenue_per_request": 7.0,
                    "fulfillment": 0.9,
                    "rival_fulfillment": 0.85,
                    "firm1_level": 1.0 + 0.01 * (step if spec == "rl_actor" else 0),
                    "firm2_level": 1.0,
                })
        traces[spec] = rows
    path = tmp_path / "sweep.json"
    path.write_text(json.dumps({
        "meta": {"steps_per_day": 1, "horizon_steps": 40, "openings": 1},
        "traces": traces,
    }))
    episodes, daily = benchmark_summary.summarize_case(path, "C0_calm")
    assert len(episodes) == len(specs) * len(benchmark_summary.RIVALS) * 3
    assert len(daily) == len(specs) * len(benchmark_summary.RIVALS) * 40
    assert {row["policy_spec"] for row in episodes} == set(specs)
    assert {row["evaluation_exposure"] for row in episodes} == {
        "training_exposed", "genuinely_held_out", "genuinely_held_out_perturbed"
    }
    comparisons = benchmark_summary.comparator_rows(episodes)
    assert len(comparisons) == len(benchmark_summary.RIVALS) * 3
    assert {row["best_static_spec"] for row in comparisons} == {"static:1"}


def test_heatmap_accepts_bounded_case_subset(tmp_path):
    rows = []
    for rival in benchmark_summary.RIVALS:
        rows.append({
            "case": "C0_calm",
            "rival": rival,
            "window": "post_burnin",
            "policy_spec": "rl_actor",
            "profit_lead_per_request": 0.1,
            "firm1_market_share": 0.5,
            "policy_switch_rate": 0.2,
        })
    output = tmp_path / "bounded_heatmap.png"
    benchmark_summary.plot_heatmaps(rows, output, "Test City")
    assert output.is_file()
    assert output.stat().st_size > 0


def test_recovery_requires_completed_primary_evaluation(tmp_path):
    from scripts import run_revision15d_extended_evaluation as extended

    out = tmp_path / "city"
    evaluation = out / "full_evaluation" / "nyc"
    for case in extended.bounded.EVALUATION_CASES:
        path = evaluation / case / "eval" / "sweep.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    (out / "policy.final_training.pt").write_bytes(b"policy")
    (out / "full_benchmark_evaluation_exit.json").write_text(
        json.dumps({"returncode": 0})
    )
    assert extended.recoverable_primary_policy(out, "nyc") == (
        out / "policy.final_training.pt"
    )
    (out / "full_benchmark_evaluation_exit.json").write_text(
        json.dumps({"returncode": 1})
    )
    assert extended.recoverable_primary_policy(out, "nyc") is None


def test_extended_status_clears_stale_error_on_recovery(tmp_path):
    from scripts import run_revision15d_extended_evaluation as extended

    out = tmp_path / "city"
    out.mkdir()
    status = out / "extended_evaluation_status.json"
    status.write_text(json.dumps({"state": "failed", "error": "old failure"}))
    extended.update(out, state="running", phase="additional_cases")
    recovered = json.loads(status.read_text())
    assert recovered["state"] == "running"
    assert "error" not in recovered


def test_wrapped_r15b_tariff_ignores_boston_zero_anchor_airport(tmp_path):
    boston_tariff = json.loads(
        (r15d.r15b_searched_tariff_path("boston")).read_text()
    )
    wrapped = tmp_path / "tariff_structure.json"
    wrapped.write_text(json.dumps(boston_tariff))
    calibration = json.loads(
        (r15d.inherited.ROOT / "data/processed/boston/calibration.json").read_text()
    )["calibration"]["tariff"]
    core = Core.__new__(Core)
    core.firm1 = make_pricer(KEYS[:4])
    core.market = Market({key: float(calibration[key]) for key in KEYS})
    action = Core._baseline_firm1_action(
        core,
        f"static_tariff:@{wrapped}",
        step_index=0,
        action_mask=np.ones(core.firm1.agent.action_dim, dtype=bool),
        day_of_week=1,
        hour=12,
        weather="clear",
        presampled=None,
        customers_per_step=1,
    )
    assert action == 0
    for key in KEYS[:4]:
        lb, ub = core.firm1.config.bounds[key]
        expected = np.clip(
            float(calibration[key]) * float(boston_tariff["structure"][key]),
            lb,
            ub,
        )
        assert getattr(core.firm1.overrides, key) == pytest.approx(expected)
    assert core.firm1.overrides.airport_fee is None


def test_evaluation_cache_identity_hashes_file_backed_comparators():
    specs = ";".join(r15d.evaluation_specs("nyc"))
    assert v26r_case_eval.spec_reference_paths(specs) == [
        r15d.r15b_searched_tariff_path("nyc").resolve()
    ]


def test_action_semantics_constant_is_stable():
    assert HYBRID_RELATIVE_ACTION_SEMANTICS == "hybrid_relative_tariff_steps_v1"


def test_core_artifact_round_trip_preserves_hybrid_contract(tmp_path):
    core = Core(
        "New York City",
        seed=15,
        total_customers_pool=20,
        simulation_sample_cap=20,
        enable_driver_supply=False,
        learning_corrections=True,
        learner_revision=4,
        credit_assignment_v25=True,
        stage_specific_rewards=True,
        hybrid_relative_actions=True,
        relative_action_steps=(0.1, 0.2),
        tariff_targets=(),
        counterfactual_day_credit=25,
        counterfactual_q_coeff=0.05,
        counterfactual_actor_weight=0.8,
        branch_imitation_coeff=0.05,
        threshold_profile_source="generated",
    )
    core.shared_steps_per_day = 6
    core.shared_action_interval_steps = 6
    core.ppo_min_rollout_transitions = 28
    path = tmp_path / "policy.pt"
    core.save_trained_model(str(path))
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    assert artifact["action_semantics"] == HYBRID_RELATIVE_ACTION_SEMANTICS
    assert artifact["action_factorization"] == {
        "target_choices": ["hold", "whole_tariff", *KEYS],
        "relative_steps": [-0.2, -0.1, 0.1, 0.2],
        "joint_action_count": 25,
        "no_op_actions_masked": True,
    }
    core.load_trained_model(str(path))
    original = dict(core.firm1.action_to_relative_step)
    core.firm1.action_to_relative_step[1] = -0.1
    with pytest.raises(ValueError, match="relative tariff steps"):
        core.load_trained_model(str(path))
    core.firm1.action_to_relative_step = original
