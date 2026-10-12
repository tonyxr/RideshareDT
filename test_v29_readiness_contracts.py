import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from Core import Core, _apply_v29_cli_contract
from platform_mdp import PlatformObservationModel, RobustObjectiveOptimizer
from pricing_models import FirmRLPricer


WORKSPACE = Path(__file__).resolve().parent


def test_v29_launchers_preserve_test_and_original_ppo_contracts():
    readiness = (WORKSPACE / "scripts/run_v29_ppo_readiness_chicago.sh").read_text()
    full = (WORKSPACE / "scripts/run_v29_ppo_full_chicago.sh").read_text()

    for script in (readiness, full):
        assert "--v29" in script
        assert "--rider_memory" in script
        assert "--minimum_tariff_multiplier 1.0" in script
        assert "--independent_coefficient_actions" in script
        assert "--compact_foresight_observation" in script
        assert "--no-persistent_recurrence" in script
        assert "--hard_kl_stop" not in script
        assert "--unrestricted_policy_experiment" in script
        assert "--require_competitive_backtest" not in script
        assert "--static_baseline_multipliers" in script
        assert "1.75" in script and "4.0" in script

    assert "--defer_final_test" in readiness
    assert '--baseline_suite "static_grid"' in readiness
    assert "--ood_perturbed_variants 0" in readiness
    assert '--model_registry_dir ""' in readiness
    assert "--defer_final_test" not in full
    assert '--model_registry_dir ""' in full


def test_v29_cli_contract_is_explicit_and_legacy_defaults_are_unchanged():
    v29 = SimpleNamespace(
        v29=True,
        v28=False,
        v27=False,
        v26=False,
        experimental_unbounded_economics=False,
        independent_coefficient_actions=False,
        compact_foresight_observation=False,
        rider_memory=False,
        choice_mode="parametric",
        training_curriculum="staged",
        policy_objective="profit_maximization",
        persistent_recurrence=None,
        manipulated_price_coefficients=2,
        minimum_tariff_multiplier=None,
    )
    _apply_v29_cli_contract(v29, set())
    assert v29.v26 and v29.v27 and v29.v28
    assert v29.experimental_unbounded_economics
    assert v29.independent_coefficient_actions
    assert v29.compact_foresight_observation
    assert v29.rider_memory
    assert v29.choice_mode == "cognitive"
    assert v29.training_curriculum == "environment_then_opponent"
    assert v29.policy_objective == "profit_with_dominance"
    assert v29.persistent_recurrence is False
    assert v29.manipulated_price_coefficients == 5
    assert v29.minimum_tariff_multiplier == pytest.approx(1.0)

    legacy = SimpleNamespace(v29=False, minimum_tariff_multiplier=None)
    _apply_v29_cli_contract(legacy, set())
    assert legacy.minimum_tariff_multiplier == pytest.approx(0.0)

    invalid = SimpleNamespace(**vars(v29))
    invalid.minimum_tariff_multiplier = 0.99
    with pytest.raises(ValueError, match="requires"):
        _apply_v29_cli_contract(invalid, {"--minimum_tariff_multiplier"})

    invalid_choice = SimpleNamespace(**vars(v29))
    invalid_choice.choice_mode = "parametric"
    with pytest.raises(ValueError, match="choice_mode cognitive"):
        _apply_v29_cli_contract(invalid_choice, {"--choice_mode"})


def test_identity_groups_never_cross_train_validation_or_test():
    core = Core.__new__(Core)
    core.seed = 91
    core.rng = np.random.default_rng(core.seed)
    core.market_name = "Chicago"
    core.threshold_profile_source = "cached"
    core.total_customers_pool = 60
    core.training_generalization_fraction = 0.0
    core.evaluation_generalization_fraction = 1.0
    core.synthetic_profile_pool = [
        {
            "ProfileId": f"rider-{identity}",
            "ProfileMarket": "Chicago",
            "PriceThreshold": 1.0 + 0.1 * variant,
            "PriceThresholdSource": "cached",
        }
        for identity in range(30)
        for variant in range(2)
    ]
    core.training_profile_pool = []
    core.validation_profile_pool = []
    core.test_profile_pool = []
    core.evaluation_profile_pool = []
    core.generalization_profile_pool = []
    core.generalization_profiles_by_city = {}
    core._threshold_ablation_applied = True

    core._build_generalization_profile_pool(pool_size=60)

    partitions = [
        {row["ProfileId"] for row in core.training_profile_pool},
        {row["ProfileId"] for row in core.validation_profile_pool},
        {row["ProfileId"] for row in core.test_profile_pool},
    ]
    assert not (partitions[0] & partitions[1])
    assert not (partitions[0] & partitions[2])
    assert not (partitions[1] & partitions[2])
    assert sum(len(partition) for partition in partitions) == 30
    assert core.evaluation_profile_pool is core.test_profile_pool

    validation = core._sample_profiles_from_pool(
        20, profile_partition="validation"
    )
    final_test = core._sample_profiles_from_pool(20, profile_partition="test")
    assert all(row in core.validation_profile_pool for row in validation)
    assert all(row in core.test_profile_pool for row in final_test)


def test_four_frame_stack_represents_four_policy_decisions_at_daily_cadence():
    core = Core.__new__(Core)
    core.persistent_recurrence = False
    core.firm1 = FirmRLPricer(
        seed=3,
        opt_keys=[
            "base_fare", "per_minute", "per_mile", "booking_fee",
            "airport_fee",
        ],
        state_frame_stack=4,
        single_state_dim=19,
        independent_coefficient_actions=True,
        experimental_unbounded_economics=True,
    )
    state = None
    for market_tick in range(31):
        state = core._stack_policy_decision_state(
            np.full(19, market_tick, dtype=np.float32),
            decision_due=(market_tick % 10 == 0),
        )
    frames = np.asarray(state).reshape(4, 19)
    np.testing.assert_array_equal(frames[:, 0], [0.0, 10.0, 20.0, 30.0])


def test_profit_with_dominance_checkpoint_uses_realized_reward_and_share_gate():
    core = Core.__new__(Core)
    core.long_term_profit_reward_config = SimpleNamespace(
        objective_mode="profit_with_dominance"
    )
    core.objective_optimizer = RobustObjectiveOptimizer(
        objective="profit_with_dominance",
        own_profit_scale=2.0,
        profit_advantage_target=0.0,
        market_share_advantage_target=0.0,
        fulfillment_floor=0.0,
        completion_coverage_floor=0.0,
        dominance_rate_floor=0.90,
        tail_weight=0.60,
    )
    summary = core._checkpoint_optimization_summary(
        own_profit=[50.0, 60.0],
        rival_profit=[1.0, 1.0],
        own_market_share=[0.60, 0.60],
        rival_market_share=[0.40, 0.40],
        fulfillment=[1.0, 1.0],
        completion_coverage=[1.0, 1.0],
        objective_dominance_rate=[0.95, 0.50],
        tail_own_profit=[40.0, 50.0],
        tail_completed_share_advantage=[0.10, 0.10],
        full_objective_reward=[-1.0, -2.0],
        tail_objective_reward=[-3.0, -4.0],
    )
    expected = 0.40 * np.mean([-1.0, -2.0]) + 0.60 * np.mean([-3.0, -4.0])
    assert np.isclose(summary["objective_score"], expected)
    assert summary["selection_key"] == (summary["objective_score"],)
    assert summary["score_source"] == "realized_profit_with_dominance_reward"
    assert summary["dominance_eligibility_source"] == (
        "positive_full_tail_own_profit_and_rolling_full_tail_"
        "completed_share_dominance"
    )
    assert summary["checkpoint_eligible"] is False
    assert summary["mode_feasible"] == [1.0, 0.0]

    negative_tail = core._checkpoint_optimization_summary(
        own_profit=[50.0, 60.0],
        rival_profit=[1.0, 1.0],
        own_market_share=[0.60, 0.60],
        rival_market_share=[0.40, 0.40],
        fulfillment=[1.0, 1.0],
        completion_coverage=[1.0, 1.0],
        objective_dominance_rate=[0.95, 0.95],
        tail_own_profit=[40.0, 50.0],
        tail_completed_share_advantage=[0.10, -0.01],
        full_objective_reward=[1.0, 1.0],
        tail_objective_reward=[1.0, 1.0],
    )
    assert negative_tail["checkpoint_eligible"] is False
    assert negative_tail["mode_feasible"] == [1.0, 0.0]

    negative_full_share = core._checkpoint_optimization_summary(
        own_profit=[50.0, 60.0],
        rival_profit=[1.0, 1.0],
        own_market_share=[0.60, 0.40],
        rival_market_share=[0.40, 0.60],
        fulfillment=[1.0, 1.0],
        completion_coverage=[1.0, 1.0],
        objective_dominance_rate=[0.95, 0.95],
        tail_own_profit=[40.0, 50.0],
        tail_completed_share_advantage=[0.10, 0.10],
        full_objective_reward=[1.0, 1.0],
        tail_objective_reward=[1.0, 1.0],
    )
    assert negative_full_share["checkpoint_eligible"] is False
    assert negative_full_share["mode_feasible"] == [1.0, 0.0]

    negative_profit = core._checkpoint_optimization_summary(
        own_profit=[50.0, -0.01],
        rival_profit=[1.0, 1.0],
        own_market_share=[0.60, 0.60],
        rival_market_share=[0.40, 0.40],
        fulfillment=[1.0, 1.0],
        completion_coverage=[1.0, 1.0],
        objective_dominance_rate=[0.95, 0.95],
        tail_own_profit=[40.0, 50.0],
        tail_completed_share_advantage=[0.10, 0.10],
        full_objective_reward=[1.0, 1.0],
        tail_objective_reward=[1.0, 1.0],
    )
    assert negative_profit["checkpoint_eligible"] is False
    assert negative_profit["mode_own_profit_positive"] == [1.0, 0.0]
    assert negative_profit["mode_feasible"] == [1.0, 0.0]

    negative_tail_profit = core._checkpoint_optimization_summary(
        own_profit=[50.0, 60.0],
        rival_profit=[1.0, 1.0],
        own_market_share=[0.60, 0.60],
        rival_market_share=[0.40, 0.40],
        fulfillment=[1.0, 1.0],
        completion_coverage=[1.0, 1.0],
        objective_dominance_rate=[0.95, 0.95],
        tail_own_profit=[40.0, -0.01],
        tail_completed_share_advantage=[0.10, 0.10],
        full_objective_reward=[1.0, 1.0],
        tail_objective_reward=[1.0, 1.0],
    )
    assert negative_tail_profit["checkpoint_eligible"] is False
    assert negative_tail_profit["mode_full_own_profit_positive"] == [1.0, 1.0]
    assert negative_tail_profit["mode_tail_own_profit_positive"] == [1.0, 0.0]
    assert negative_tail_profit["mode_feasible"] == [1.0, 0.0]


def test_deployment_promotion_gate_is_fail_closed():
    eligible = {"validation": {"checkpoint_eligible": True}}
    ineligible = {"validation": {"checkpoint_eligible": False}}
    passed_backtest = {"passed": True, "failures": []}
    failed_backtest = {"passed": False, "failures": ["late_loss"]}

    assert Core._deployment_promotion_gate(eligible, passed_backtest)["passed"]
    assert not Core._deployment_promotion_gate(None, passed_backtest)["passed"]
    assert not Core._deployment_promotion_gate(ineligible, passed_backtest)["passed"]
    failed = Core._deployment_promotion_gate(eligible, failed_backtest)
    assert not failed["passed"]
    assert any("late_loss" in reason for reason in failed["failures"])
    deferred = Core._apply_final_test_policy(
        Core._deployment_promotion_gate(eligible, passed_backtest),
        defer_final_test=True,
    )
    assert deferred["passed"] is False
    assert deferred["final_test_opened"] is False
    assert "final_test_deferred" in deferred["failures"]

    stale_presence_only = Core._apply_fresh_benchmark_policy(
        Core._deployment_promotion_gate(eligible, passed_backtest),
        {"checkpoint_eligible": False, "score": 123.0},
        required=True,
    )
    assert stale_presence_only["passed"] is False
    assert stale_presence_only["fresh_benchmark_report_present"] is True
    assert stale_presence_only["fresh_benchmark_checkpoint_eligible"] is False
    assert "fresh_benchmark_checkpoint_not_eligible" in stale_presence_only[
        "failures"
    ]
    fresh_pass = Core._apply_fresh_benchmark_policy(
        Core._deployment_promotion_gate(eligible, passed_backtest),
        {"checkpoint_eligible": True},
        required=True,
    )
    assert fresh_pass["passed"] is True

    static_lower_bound = {
        "in_distribution_and_benchmarks": {
            "comparison": {
                "static": {"rl_minus_best_uniform_static": 0.02},
                "adaptive": {"rl_minus_best_uniform_static": -0.011},
            }
        }
    }
    loses_to_static = Core._apply_uniform_static_lower_bound_policy(
        fresh_pass,
        static_lower_bound,
        required=True,
        tolerance=0.01,
    )
    assert loses_to_static["passed"] is False
    assert loses_to_static["uniform_static_lower_bound_min_rl_gap"] == pytest.approx(
        -0.011
    )
    assert loses_to_static["uniform_static_lower_bound_violations"] == {
        "adaptive": -0.011
    }
    assert "not_best_fixed_5d" in loses_to_static[
        "uniform_static_comparator_scope"
    ]
    within_tolerance = Core._apply_uniform_static_lower_bound_policy(
        fresh_pass,
        {
            "in_distribution_and_benchmarks": {
                "comparison": {
                    "static": {"rl_minus_best_uniform_static": -0.01}
                }
            }
        },
        required=True,
        tolerance=0.01,
    )
    assert within_tolerance["passed"] is True
    incomplete_static = Core._apply_uniform_static_lower_bound_policy(
        fresh_pass,
        {
            "in_distribution_and_benchmarks": {
                "comparison": {
                    "static": {"rl_minus_best_uniform_static": 0.02},
                    "adaptive": {"rl_objective": 1.0},
                }
            }
        },
        required=True,
    )
    assert incomplete_static["passed"] is False
    assert incomplete_static["uniform_static_lower_bound_missing_opponents"] == [
        "adaptive"
    ]


def test_v29_core_wiring_is_compact_independent_additive_ppo():
    core = Core(
        market_name="Chicago",
        seed=29,
        choice_mode="parametric",
        firm1_mode="RL",
        firm2_mode="adaptive_best_response",
        total_customers_pool=20,
        policy_objective="profit_with_dominance",
        independent_coefficient_actions=True,
        compact_foresight_observation=True,
        experimental_unbounded_economics=True,
        shock_observations=True,
        persistent_recurrence=False,
        minimum_tariff_multiplier=1.0,
    )
    agent = core.firm1.agent
    assert core.platform_observation_dim == 25  # 19 compact + 6 public shock fields
    assert agent.action_feature_dim == 0
    assert agent.action_dim == 11  # hold + decrease/increase for five coefficients
    assert agent.persistent_recurrence is False
    assert agent.scheduled_exploration is False
    assert agent.initial_exploration_rate == pytest.approx(0.0)
    assert agent.final_exploration_rate == pytest.approx(0.0)
    assert agent.initial_hold_probability == pytest.approx(1.0 / 11.0)
    assert core.firm1.minimum_tariff_multiplier == pytest.approx(1.0)

    assert set(core.firm1.action_to_steps) == set(range(11))
    assert all(value == 0 for value in core.firm1.action_to_steps[0].values())
    manipulated = set(core.shared_edit_keys)
    seen = set()
    for action in range(1, 11):
        steps = core.firm1.action_to_steps[action]
        changed = {key: value for key, value in steps.items() if value != 0}
        assert len(changed) == 1
        key, direction = next(iter(changed.items()))
        assert key in manipulated
        assert direction in {-1, 1}
        seen.add((key, direction))
    assert seen == {(key, direction) for key in manipulated for direction in (-1, 1)}

    core._reset_adaptive_episode(
        tariff_reset_fraction=0.15,
        opponent_mode="static",
        episode_seed=29001,
    )
    for key in core.shared_edit_keys:
        anchor = float(getattr(core.market.curr_market, key))
        value = float(getattr(core.firm1.overrides, key))
        assert value / anchor >= 1.0


def test_v29_tariff_floor_is_part_of_the_portable_policy_contract():
    kwargs = dict(
        market_name="Chicago",
        seed=29002,
        choice_mode="cognitive",
        firm1_mode="RL",
        firm2_mode="static",
        total_customers_pool=20,
        independent_coefficient_actions=True,
        compact_foresight_observation=True,
        experimental_unbounded_economics=True,
        minimum_tariff_multiplier=1.0,
    )
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "v29-floor.pt")
        source = Core(**kwargs)
        source.save_trained_model(path)
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        assert artifact["minimum_tariff_multiplier"] == pytest.approx(1.0)
        assert artifact["action_semantics"] == (
            "independent_upper_unbounded_additive_step_with_anchor_floor"
        )
        assert artifact["environment_contract"]["compatibility_payload"][
            "minimum_tariff_multiplier"
        ] == pytest.approx(1.0)
        matching = Core(**kwargs)
        matching.load_trained_model(path)
        mismatch = Core(**{**kwargs, "minimum_tariff_multiplier": 0.0})
        with pytest.raises(ValueError, match="minimum tariff multiplier"):
            mismatch.load_trained_model(path)


def test_artifact_binds_ordered_observation_and_environment_fingerprint():
    kwargs = dict(
        market_name="Chicago",
        seed=17,
        choice_mode="parametric",
        firm1_mode="RL",
        firm2_mode="static",
        policy_objective="profit_with_dominance",
        total_customers_pool=20,
        simulation_sample_cap=17,
        independent_coefficient_actions=True,
        compact_foresight_observation=True,
        experimental_unbounded_economics=True,
    )
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "policy.pt")
        trained = Core(**kwargs)
        trained.save_trained_model(path, model_id="schema-contract")
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        assert artifact["artifact_schema_version"] == 2
        assert artifact["observation_schema"]["ordered_feature_names"] == list(
            PlatformObservationModel.feature_names(
                shock_features=False, compact_foresight=True
            )
        )
        assert artifact["environment_contract"]["compatibility_sha256"]

        matching = Core(**kwargs)
        matching.load_trained_model(path)

        mismatched_kwargs = dict(kwargs)
        mismatched_kwargs["simulation_sample_cap"] = 18
        mismatched = Core(**mismatched_kwargs)
        with pytest.raises(ValueError, match="environment/data fingerprint"):
            mismatched.load_trained_model(path)


def test_validation_recovery_checkpoint_is_atomic_and_explicitly_partial():
    core = Core(
        market_name="Chicago",
        seed=19,
        choice_mode="parametric",
        firm1_mode="RL",
        firm2_mode="static",
        total_customers_pool=20,
    )
    candidate = {
        "day": 400,
        "training_stage": "opponent",
        "score": 0.25,
        "agent_state": {
            key: value.detach().cpu().clone()
            for key, value in core.firm1.agent.net.state_dict().items()
        },
        "optimizer_state": core.firm1.agent.optimizer_state_dict(),
        "normalizer_state": core.firm1.agent.normalizer_state_dict(),
        "validation": {"checkpoint_eligible": False},
    }
    with tempfile.TemporaryDirectory() as directory:
        path = core._persist_validation_recovery_checkpoint(
            candidate,
            recovery_root=directory,
            model_id="readiness",
            run_id="run-1",
        )
        payload = torch.load(path, map_location="cpu", weights_only=False)
        assert payload["format"] == "ride-response-validation-recovery"
        assert payload["checkpoint"]["day"] == 400
        assert payload["rng_metadata"]["numpy_generator_state"]
        assert payload["exact_environment_resume_supported"] is False
        assert "inflight_ppo_rollout_buffer_not_included" in payload[
            "resume_limitations"
        ]
        assert not os.path.exists(path + f".tmp-{os.getpid()}")
