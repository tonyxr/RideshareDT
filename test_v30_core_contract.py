from types import SimpleNamespace

import numpy as np
import pytest
import torch

from Core import Core, _apply_v29_cli_contract, _apply_v30_cli_contract
from digital_twin.observation import V30_CLOSED_LOOP_FEATURES
from pricing_models import FirmMetrics
from platform_mdp import LongTermProfitRewardConfig, RobustObjectiveOptimizer


def _core(**overrides):
    kwargs = dict(
        market_name="Chicago",
        seed=3001,
        choice_mode="parametric",
        firm1_mode="RL",
        firm2_mode="static",
        total_customers_pool=20,
        policy_objective="profit_dominance_v30",
        independent_coefficient_actions=True,
        compact_foresight_observation=True,
        experimental_unbounded_economics=True,
        minimum_tariff_multiplier=1.0,
        closed_loop_digital_twin=True,
        terminal_profit_dominance_weight=1.25,
        persistent_recurrence=False,
    )
    kwargs.update(overrides)
    return Core(**kwargs)


def test_v30_cli_adds_to_v29_without_mutating_legacy_contract():
    args = SimpleNamespace(
        v30=True, v29=True, v28=False, v27=False, v26=False,
        experimental_unbounded_economics=False,
        independent_coefficient_actions=False,
        compact_foresight_observation=False,
        closed_loop_digital_twin=False,
        rider_memory=False, choice_mode="parametric",
        training_curriculum="staged", policy_objective="profit_maximization",
        persistent_recurrence=None, manipulated_price_coefficients=2,
        minimum_tariff_multiplier=None,
    )
    _apply_v29_cli_contract(args, set())
    _apply_v30_cli_contract(args, set())
    assert args.v26 and args.v27 and args.v28 and args.v29
    assert args.closed_loop_digital_twin is True
    assert args.policy_objective == "profit_dominance_v30"
    assert args.compact_foresight_observation is True

    legacy = SimpleNamespace(v30=False, closed_loop_digital_twin=False)
    _apply_v30_cli_contract(legacy, set())
    assert legacy.closed_loop_digital_twin is False


def test_v30_core_appends_loop_features_and_public_episode_clock():
    core = _core(shock_observations=True)
    assert core.platform_observation_dim == 31  # v29 19 + six shocks + four loop + two clock
    contract = core._observation_schema_contract()
    assert contract["ordered_feature_names"][-6:-2] == list(V30_CLOSED_LOOP_FEATURES)
    state = core._build_rl_state(day_of_week=1, hour=8, weather="clear")
    assert state.shape == (31,)
    assert np.all(np.isfinite(state))
    assert core.firm1.single_state_dim == 31
    assert core.firm1.agent.action_dim == 11


def test_simulated_public_outcome_updates_only_closed_loop_estimator_state():
    core = _core(shock_observations=False)
    before = core.v30_state_estimator.snapshot()
    core._ingest_platform_observations(
        FirmMetrics(total=100, chosen=60, completed=50, profit=100),
        FirmMetrics(total=100, chosen=40, completed=30, profit=40),
    )
    after = core.v30_state_estimator.snapshot()
    assert after != before
    assert after["state"]["means"]["demand"] > 1
    assert after["state"]["means"]["supply"] > 1
    state = core._build_rl_state(day_of_week=1, hour=8, weather="clear")
    assert state.shape == (25,)  # concise v29 frame plus six v30 fields


def test_v30_artifact_binds_closed_loop_schema_and_reward(tmp_path):
    path = tmp_path / "v30.pt"
    source = _core()
    source.save_trained_model(str(path), model_id="v30-contract")
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    assert artifact["closed_loop_digital_twin"] is True
    assert artifact["policy_objective"] == "profit_dominance_v30"
    assert artifact["observation_schema"]["closed_loop_digital_twin"] is True
    assert artifact["environment_contract"]["compatibility_payload"][
        "closed_loop_digital_twin"
    ] is True
    matching = _core()
    matching.load_trained_model(str(path))
    mismatch = _core(closed_loop_digital_twin=False)
    with pytest.raises(ValueError, match="observation schema|environment/data"):
        mismatch.load_trained_model(str(path))


def test_v30_requires_concise_base_observation():
    with pytest.raises(ValueError, match="compact-foresight"):
        _core(compact_foresight_observation=False)


def test_v30_checkpoint_uses_profit_dominance_not_share():
    core = Core.__new__(Core)
    core.long_term_profit_reward_config = LongTermProfitRewardConfig(
        objective_mode="profit_dominance_v30"
    )
    core.objective_optimizer = RobustObjectiveOptimizer(
        objective="profit_dominance_v30",
        own_profit_scale=2.0,
        profit_advantage_target=0.0,
        market_share_advantage_target=0.0,
        fulfillment_floor=0.0,
        completion_coverage_floor=0.0,
        dominance_rate_floor=0.9,
    )
    kwargs = dict(
        own_profit=[3.0, 4.0], rival_profit=[2.0, 2.0],
        # Shares deliberately lose: they are not part of the v30 objective.
        own_market_share=[0.1, 0.1], rival_market_share=[0.9, 0.9],
        fulfillment=[1.0, 1.0], completion_coverage=[1.0, 1.0],
        objective_dominance_rate=[1.0, 1.0],
        tail_own_profit=[3.0, 4.0],
        tail_completed_share_advantage=[-0.8, -0.8],
        full_objective_reward=[1.0, 2.0],
        tail_objective_reward=[1.0, 2.0],
        tail_profit_advantage=[0.5, 1.0],
    )
    passing = core._checkpoint_optimization_summary(**kwargs)
    assert passing["checkpoint_eligible"] is True
    assert passing["dominance_eligibility_source"].endswith(
        "full_tail_profit_dominance"
    )
    failing = core._checkpoint_optimization_summary(
        **{**kwargs, "tail_profit_advantage": [0.5, -0.01]}
    )
    assert failing["checkpoint_eligible"] is False


def test_checkpoint_validation_restores_v30_training_episode_and_estimator():
    core = _core(firm2_mode="adaptive_best_response_aggressive")
    core._refresh_profile_pool(rides_per_timestep=2)
    core.v30_episode_reward_tracker.record(
        own_profit_per_request=3.0, rival_profit_per_request=1.0
    )
    tracker_before = core.v30_episode_reward_tracker.snapshot()
    estimator_before = core.v30_state_estimator.snapshot()
    measurement_before = core._v30_measurement_index

    report = core._adaptive_policy_validation(
        customers_per_step=2,
        profile_sample_size=2,
        horizon=2,
        aggressive_seed_count=1,
        opponent_interval_steps=1,
        firm1_interval_steps=1,
        steps_per_day=1,
        profile_partition="validation",
    )

    assert core.v30_episode_reward_tracker.snapshot() == tracker_before
    assert core.v30_state_estimator.snapshot() == estimator_before
    assert core._v30_measurement_index == measurement_before
    # Two decisions in a two-step episode: validation includes the final
    # terminal transition rather than dropping it from critic diagnostics.
    assert report["heldout_critic_td_samples"] == 2
    for row in report["rollout_summaries"].values():
        assert row["terminal_profit_dominance_reward"] == pytest.approx(
            1.25 * row["steps"] * row["late_objective_advantage"]
        )
        assert row["objective_reward_mean"] == pytest.approx(
            row["own_profit_per_request"] - row["rival_profit_per_request"]
            + row["terminal_profit_dominance_reward"] / row["steps"]
        )


def test_explicit_output_directory_is_not_rewritten_to_repository_root(tmp_path):
    from Core import _resolve_output_prefix
    prefix = str(tmp_path / "isolated" / "chicago")
    assert _resolve_output_prefix(prefix) == prefix
    assert _resolve_output_prefix("artifacts/check/chicago").endswith("/RideResponse/artifacts/check/chicago")
