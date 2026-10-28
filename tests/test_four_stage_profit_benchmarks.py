import numpy as np
import pytest

from Core import (
    BENCHMARK_OPPONENT_MODES,
    CATCHUP_OPPONENT_MODES,
    OPTIMIZED_BENCHMARK_OPPONENT_MODES,
    PROFIT_BEST_RESPONSE_OPPONENT_MODES,
    Core,
)
from PPOAgent import PPOAgent
from platform_mdp import LongTermProfitRewardConfig, TrainingStageScheduler
from pricing_models import (
    FirmFastPriceGapCatchUpPricer,
    FirmPIPriceGapPricer,
    FirmPriceGapCatchUpPricer,
    FirmProfitOptimizedPIPriceGapPricer,
    FirmTwinProfitOptimizerMixin,
    PROFIT_BEST_RESPONSE_GRID,
)
from strategic_benchmarks import cycle_diagnostics, solve_tariff_game


def test_canonical_benchmarks_are_unchanged_and_optimized_modes_are_additive():
    assert BENCHMARK_OPPONENT_MODES == (
        "pi_price_gap",
        "mpc_grid",
        "region_supply_demand",
        "surge_driver_incentive",
        "queue_service_threshold",
    )
    assert len(OPTIMIZED_BENCHMARK_OPPONENT_MODES) == 5
    assert not set(BENCHMARK_OPPONENT_MODES) & set(OPTIMIZED_BENCHMARK_OPPONENT_MODES)
    assert issubclass(FirmProfitOptimizedPIPriceGapPricer, FirmPIPriceGapPricer)
    assert FirmProfitOptimizedPIPriceGapPricer is not FirmPIPriceGapPricer
    assert len(PROFIT_BEST_RESPONSE_GRID) == 13


def test_four_stage_schedule_has_distinct_local_decays():
    scheduler = TrainingStageScheduler(
        stage_fractions=(0.10, 0.10, 0.30, 0.50),
        dynamic_episode_days=30,
    )
    assert [(stage.name, stage.opponent_regime) for stage in scheduler.stages] == [
        ("market", "market"),
        ("static_opponent", "static"),
        ("price_gap_catchup", "catchup"),
        ("profit_maximizers", "profit"),
    ]
    for stage in scheduler.stages:
        opening = scheduler.smooth_controls_at(stage.start)
        closing = scheduler.smooth_controls_at(min(1.0, stage.end - 1e-7))
        assert opening["exploration_rate"] > closing["exploration_rate"]
        assert opening["entropy_scale"] > closing["entropy_scale"]
        assert opening["learning_rate_scale"] > closing["learning_rate_scale"]
    starts = [scheduler.smooth_controls_at(stage.start) for stage in scheduler.stages]
    assert len({row["exploration_rate"] for row in starts}) == 4
    assert len({row["learning_rate_scale"] for row in starts}) == 4


def test_reward_curriculum_is_distinct_and_increases_forward_competitive_terms():
    config = LongTermProfitRewardConfig(
        stage_specific_rewards=True,
        four_stage_curriculum_rewards=True,
        learning_corrections=True,
        temporal_competitive_reward=True,
        opponent_profit_lead_weight=0.50,
        market_share_competitiveness_weight=0.05,
        durable_dominance_weight=0.20,
        adaptive_residual_weight=0.10,
    )
    weights = {
        regime: config.economic_weights(regime)
        for regime in ("market", "static", "catchup", "profit")
    }
    assert len({tuple(row.values()) for row in weights.values()}) == 4
    assert weights["market"]["profit_lead"] == 0.0
    assert weights["static"]["adaptive_profit_residual"] == 0.0
    assert weights["catchup"]["adaptive_profit_residual"] > 0.0
    assert (
        weights["profit"]["adaptive_profit_residual"]
        > weights["catchup"]["adaptive_profit_residual"]
    )
    assert weights["profit"]["durable_profit_lead"] > weights["static"]["durable_profit_lead"]
    for row in weights.values():
        assert sum(row.values()) == pytest.approx(1.0)


def test_production_reward_is_two_term_with_distinct_stage_weights():
    config = LongTermProfitRewardConfig(
        stage_specific_rewards=True,
        four_stage_curriculum_rewards=True,
        learning_corrections=True,
        temporal_competitive_reward=False,
        opponent_profit_lead_weight=0.50,
        market_share_competitiveness_weight=0.0,
        own_profit_scale=2.0,
    )
    weights = [
        config.economic_weights(regime)
        for regime in ("market", "static", "catchup", "profit")
    ]
    assert len({(row["own_profit"], row["profit_lead"]) for row in weights}) == 4
    for row in weights:
        assert row["own_completed_share"] == 0.0
        assert row["completed_share_lead"] == 0.0
        assert set(key for key, value in row.items() if value) <= {
            "own_profit", "profit_lead"
        }
        assert sum(row.values()) == pytest.approx(1.0)
    assert len(config.critic_component_weights("profit")) == 2


def test_gap_catchup_uses_gap_signal_and_changes_only_one_coefficient():
    pricer = FirmPriceGapCatchUpPricer(
        seed=3, managed_keys=["base_fare", "per_minute", "per_mile"]
    )
    pricer._ensure_initialized({
        "base_fare": 3.0,
        "per_minute": 0.4,
        "per_mile": 1.9,
        "booking_fee": 2.75,
        "airport_fee": 6.0,
    })
    before = {key: getattr(pricer.overrides, key) for key in pricer.managed_keys}
    pricer.ema_gap = 2.0
    pricer.ema_choice_share = pricer.target_share
    pricer.act(3.0, 0.4, 12, "clear", 1.9, 2.75, 6.0)
    after = {key: getattr(pricer.overrides, key) for key in pricer.managed_keys}
    changed = [key for key in before if before[key] != after[key]]
    assert len(changed) == 1
    assert after[changed[0]] < before[changed[0]]
    assert pricer.gap_response_weight == pytest.approx(0.90)
    assert issubclass(FirmFastPriceGapCatchUpPricer, FirmPriceGapCatchUpPricer)


def test_profit_optimizers_record_only_public_reactions_and_hysteresis_plan():
    pricer = FirmProfitOptimizedPIPriceGapPricer(
        seed=4, managed_keys=["base_fare", "per_minute"]
    )
    assert isinstance(pricer, FirmTwinProfitOptimizerMixin)
    assert tuple(pricer.profit_candidate_grid) == PROFIT_BEST_RESPONSE_GRID
    anchors = {"base_fare": 3.0, "per_minute": 0.4}
    pricer._last_own_grid_multiplier = 1.10
    pricer.observe_firm1_posted_tariff(
        firm1_tariff={"base_fare": 3.3, "per_minute": 0.44},
        anchors=anchors,
    )
    assert pricer.firm1_reaction_table == {"1.10": [pytest.approx(1.10)]}
    pricer.set_profit_plan(
        multiplier=None,
        improvement=0.01,
        standard_error=0.02,
        candidate_means={1.0: 2.0, 1.1: 2.01},
    )
    pricer.act(3.0, 0.4, 12, "clear", 1.9, 2.75, 6.0)
    assert pricer.last_reason == "profit_hysteresis_hold"


def test_stage_controls_drive_actual_ppo_exploration_entropy_and_lr():
    agent = PPOAgent(
        state_dim=3,
        action_dim=3,
        lr=3e-4,
        ent_coeff=0.01,
        min_ent_coeff=0.001,
        max_ent_coeff=0.08,
        scheduled_exploration=True,
        learning_corrections=True,
        learner_revision=4,
        persistent_recurrence=True,
        privileged_critic=True,
        privileged_value_only=True,
        critic_state_dim=3,
        single_state_dim=3,
    )
    agent.set_stage_controls(
        exploration_rate=0.12,
        entropy_scale=0.85,
        learning_rate_scale=0.60,
    )
    assert agent.exploration_rate == pytest.approx(0.12)
    assert agent.ent_coeff == pytest.approx(0.0085)
    assert agent.curr_lr == pytest.approx(1.8e-4)


def test_theory_backtest_reports_nash_stackelberg_and_cycles_separately():
    grid = (0.8, 1.0, 1.2)
    firm1 = np.asarray([[2, 1, 0], [3, 2, 1], [2, 4, 3]], dtype=float)
    firm2 = firm1.T + 0.2
    report = solve_tariff_game(grid, firm1, firm2)
    assert report["selected_static_nash"] is not None
    assert report["stackelberg_firm1_leader"]["firm1_profit_per_request"] == 3.0
    assert report["second_mover_advantage_at_stackelberg"] is True
    cycle = cycle_diagnostics([1.0, 1.1, 1.0, 1.1, 1.0])
    assert cycle["cycle_detected"] is True


def test_core_registers_separate_stage_three_and_stage_four_populations():
    core = Core(
        market_name="Chicago",
        seed=9,
        total_customers_pool=20,
        choice_mode="parametric",
        openai_api_key="",
        firm1_mode="RL",
        firm2_mode="static",
        manipulated_price_coefficients=2,
        credit_assignment_v25=True,
        stage_specific_rewards=True,
        learning_corrections=True,
        temporal_competitive_reward=True,
        state_frame_stack=2,
    )
    assert core.training_catchup_opponent_modes == CATCHUP_OPPONENT_MODES
    assert set(OPTIMIZED_BENCHMARK_OPPONENT_MODES) <= set(
        core.training_profit_opponent_modes
    )
    assert set(PROFIT_BEST_RESPONSE_OPPONENT_MODES) <= set(
        core.training_profit_opponent_modes
    )
    assert not set(BENCHMARK_OPPONENT_MODES) & set(
        core.training_profit_opponent_modes
    )
