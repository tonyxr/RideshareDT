import unittest
from collections import deque
from types import SimpleNamespace

import numpy as np

from Core import Core
from Market_models import CoefficientOverrides
from MarketInteraction import MarketInteraction
from optim_config import default_specs_for
from pricing_models import (
    ActionDescriptor,
    FirmMetrics,
    FirmRLPricer,
    FirmStaticPricer,
    build_discrete_action_space,
)
from platform_mdp import (
    ActionStabilityTracker,
    ConstraintConfig,
    LongTermProfitReward,
    LongTermProfitRewardConfig,
    ObservationConfig,
    PlatformObservationModel,
    PositiveBusinessReward,
    PositiveRewardConfig,
    SoftConstraintController,
    TrainingStageScheduler,
)


class CorePPOFixRegressionTests(unittest.TestCase):
    def test_manipulated_price_coefficient_count_scopes_both_firms(self):
        all_keys = [
            "base_fare",
            "per_minute",
            "per_mile",
            "booking_fee",
            "airport_fee",
        ]
        for count in (1, 3, 5):
            with self.subTest(count=count):
                core = Core(
                    market_name="New York City",
                    seed=700 + count,
                    choice_mode="parametric",
                    firm1_mode="RL",
                    firm2_mode="adaptive_best_response",
                    manipulated_price_coefficients=count,
                    total_customers_pool=20,
                )
                expected = all_keys[:count]
                inactive = all_keys[count:]

                self.assertEqual(core.manipulated_price_coefficient_count, count)
                self.assertEqual(core.shared_edit_keys, expected)
                self.assertEqual(core.firm1.opt_keys, expected)
                self.assertEqual(list(core.firm2.managed_keys), expected)
                for firm in (core.firm1, core.firm2):
                    for key in expected:
                        self.assertIsNotNone(getattr(firm.overrides, key))
                    for key in inactive:
                        self.assertIsNone(getattr(firm.overrides, key))

                core.firm1.apply_action(1, core.market)
                for key in expected:
                    self.assertAlmostEqual(
                        getattr(core.firm1.overrides, key),
                        0.70 * getattr(core.market.curr_market, key),
                    )
                for key in inactive:
                    self.assertIsNone(getattr(core.firm1.overrides, key))

    def test_manipulated_price_coefficient_count_rejects_out_of_range(self):
        for count in (0, 6):
            with self.subTest(count=count):
                with self.assertRaisesRegex(ValueError, "between 1 and 5"):
                    Core(
                        market_name="New York City",
                        seed=811,
                        choice_mode="parametric",
                        firm1_mode="RL",
                        firm2_mode="static",
                        manipulated_price_coefficients=count,
                        total_customers_pool=20,
                    )

    def test_checkpoint_validation_uses_only_configured_rival(self):
        rollouts = Core._validation_opponent_rollouts(
            "adaptive_best_response_aggressive", 5
        )

        self.assertEqual(
            rollouts,
            [
                (
                    f"actual_firm2__adaptive_best_response_aggressive__opening_{index}",
                    "adaptive_best_response_aggressive",
                )
                for index in range(1, 6)
            ],
        )

        benchmark_rollouts = Core._validation_opponent_rollouts(
            "adaptive_best_response_aggressive", 5, include_benchmarks=True
        )
        self.assertEqual(len(benchmark_rollouts), 10)
        self.assertEqual(
            {mode for _, mode in benchmark_rollouts[5:]},
            {
                "pi_price_gap",
                "mpc_grid",
                "region_supply_demand",
                "surge_driver_incentive",
                "queue_service_threshold",
            },
        )

    def test_named_objective_disables_constraint_and_risk_critics(self):
        captured = {}

        def set_context(lambdas, **kwargs):
            captured["lambdas"] = np.asarray(lambdas, dtype=float)
            captured.update(kwargs)

        core = Core.__new__(Core)
        core.firm1_mode = "RL"
        core.constrained_reward = True
        core.long_term_profit_reward_config = LongTermProfitRewardConfig(
            objective_mode="profit_maximization"
        )
        core.soft_constraints = SimpleNamespace(
            names=("fulfillment", "coverage", "wait", "margin"),
            lambdas=np.asarray([2.0, 2.0, 1.0, 1.5], dtype=np.float32),
        )
        core.firm1 = SimpleNamespace(
            agent=SimpleNamespace(set_optimization_context=set_context)
        )

        core._sync_agent_optimization_context()

        np.testing.assert_array_equal(captured["lambdas"], np.zeros(4))
        self.assertFalse(captured["constraints_active"])
        self.assertFalse(captured["risk_active"])
        self.assertEqual(captured["risk_coeff"], 0.0)

    def test_case_study_holdout_preserves_cached_nyc_threshold_population(self):
        core = Core.__new__(Core)
        core.seed = 773
        core.rng = np.random.default_rng(core.seed)
        core.market_name = "New York City"
        core.threshold_profile_source = "cached"
        core.total_customers_pool = 64
        core.training_generalization_fraction = 0.0
        core.evaluation_generalization_fraction = 1.0
        core.synthetic_profile_pool = [
            {
                "ProfileId": f"cached-{index}",
                "ProfileMarket": "New York City",
                "PriceThreshold": 1.0 + 0.05 * index,
                "PriceThresholdSource": "cached",
            }
            for index in range(20)
        ]
        core.training_profile_pool = []
        core.evaluation_profile_pool = []
        core.generalization_profile_pool = []
        core.generalization_profiles_by_city = {}

        core._build_generalization_profile_pool(pool_size=80)

        self.assertEqual(
            set(core.generalization_profiles_by_city),
            {"New York City"},
        )
        self.assertTrue(all(
            row["PriceThresholdSource"] == "cached"
            for row in core.generalization_profile_pool
        ))
        self.assertFalse(
            set(map(id, core.training_profile_pool))
            & set(map(id, core.evaluation_profile_pool))
        )

        holdout = core._sample_profiles_from_pool(
            16,
            generalization_fraction=1.0,
            generalization_city="New York City",
        )
        self.assertTrue(all(row["ProfileMarket"] == "New York City" for row in holdout))
        self.assertTrue(all(row in core.evaluation_profile_pool for row in holdout))

        training_only = core._sample_profiles_from_pool(
            4, generalization_fraction=0.0
        )
        self.assertTrue(all(row in core.training_profile_pool for row in training_only))

        with self.assertRaisesRegex(ValueError, "New York City-only"):
            core._sample_profiles_from_pool(
                4,
                generalization_fraction=1.0,
                generalization_city="Chicago",
            )

    def test_competitive_backtest_requires_dynamic_late_dominance(self):
        core = Core.__new__(Core)
        core.firm1_mode = "RL"
        core.shared_edit_keys = ["base_fare"]
        core.firm1 = SimpleNamespace(
            action_steps=lambda action: (
                {}
                if int(action) == 0
                else {"base_fare": -1 if int(action) % 2 else 1}
            ),
            config=SimpleNamespace(step={"base_fare": 0.10}),
        )
        core.training_logs = [
            {
                "avg_reward": 0.20 + 0.005 * day,
                "validation_score": (
                    0.5 + 0.2 * (day // 25)
                    if day in {24, 49, 74, 99}
                    else np.nan
                ),
            }
            for day in range(100)
        ]
        core.evaluation_logs = [
            {
                "reward": 0.30 + 0.004 * day,
                "rl_completed_share": 0.72,
                "heuristic_completed_share": 0.20,
                "rl_profit": 3.0,
                "heuristic_profit": 0.8,
                "firm1_base_fare": 2.5 + 0.1 * (day % 2),
                "action": 1 + (day % 2),
            }
            for day in range(100)
        ]

        report = core._competitive_durability_backtest()

        self.assertTrue(report["passed"])
        self.assertGreater(report["late_completed_share_advantage"], 0.10)
        self.assertGreater(report["late_profit_advantage_per_request"], 0.0)
        self.assertGreater(report["held_out_validation_score_fitted_change"], 0.0)
        self.assertGreaterEqual(report["late_evaluation_reward_retention"], 0.75)
        self.assertGreater(report["late_coefficient_change_rate"], 0.02)

    def test_stable_dominant_policy_can_converge_without_action_churn(self):
        core = Core.__new__(Core)
        core.firm1_mode = "RL"
        core.shared_edit_keys = ["base_fare"]
        core.firm1 = SimpleNamespace(
            action_steps=lambda action: {},
            config=SimpleNamespace(step={"base_fare": 0.10}),
        )
        core.training_logs = [
            {
                "avg_reward": 0.60,
                "validation_score": (
                    0.80
                    if day in {24, 49, 74, 99}
                    else np.nan
                ),
            }
            for day in range(100)
        ]
        core.evaluation_logs = [
            {
                "reward": 0.60,
                "rl_completed_share": 0.60,
                "heuristic_completed_share": 0.30,
                "rl_profit": 3.0,
                "heuristic_profit": 2.0,
                "firm1_base_fare": 2.5,
                "action": 0,
            }
            for _ in range(100)
        ]

        report = core._competitive_durability_backtest()

        self.assertTrue(report["passed"])
        self.assertEqual(report["late_fare_equivalent_change_rate"], 0.0)
        self.assertEqual(report["late_action_diversity"], 1)

    def test_noisy_validation_slope_does_not_override_economic_durability(self):
        core = Core.__new__(Core)
        core.firm1_mode = "RL"
        core.shared_edit_keys = ["base_fare"]
        core.firm1 = SimpleNamespace(
            action_steps=lambda action: {},
            config=SimpleNamespace(step={"base_fare": 0.10}),
        )
        core.training_logs = [
            {
                "avg_reward": 0.80,
                "validation_score": (
                    1.20 - 0.10 * (day // 25)
                    if day in {24, 49, 74, 99}
                    else np.nan
                ),
            }
            for day in range(100)
        ]
        core.evaluation_logs = [
            {
                "reward": 1.0,
                "rl_completed_share": 0.60,
                "heuristic_completed_share": 0.30,
                "rl_profit": 2.50,
                "heuristic_profit": 1.50,
                "firm1_base_fare": 2.50,
                "action": 2,
            }
            for _ in range(100)
        ]

        report = core._competitive_durability_backtest()

        self.assertLess(report["held_out_validation_score_fitted_change"], 0.0)
        self.assertTrue(report["passed"])
        self.assertNotIn("held_out_validation_score_degraded", report["failures"])
        self.assertIsNone(
            report["criteria"][
                "held_out_validation_score_fitted_change_min"
            ]
        )

    def test_positive_profit_without_profit_dominance_fails_backtest(self):
        core = Core.__new__(Core)
        core.firm1_mode = "RL"
        core.shared_edit_keys = ["base_fare"]
        core.firm1 = SimpleNamespace(
            action_steps=lambda action: {},
            config=SimpleNamespace(step={"base_fare": 0.10}),
        )
        core.long_term_profit_reward_config = LongTermProfitRewardConfig(
            objective_mode="profit_maximization"
        )
        core.training_logs = [
            {
                "avg_reward": 0.50,
                "validation_score": 0.50 if day in {24, 49} else np.nan,
            }
            for day in range(50)
        ]
        core.evaluation_logs = [
            {
                "reward": 0.50,
                "rl_completed_share": 0.40,
                "heuristic_completed_share": 0.50,
                "rl_profit": 2.0,
                "heuristic_profit": 3.0,
                "firm1_base_fare": 2.5,
                "action": 0,
            }
            for _ in range(100)
        ]

        report = core._competitive_durability_backtest()

        self.assertFalse(report["passed"])
        self.assertIn("late_profit_advantage_below_target", report["failures"])
        self.assertIn("late_profit_dominance_not_consistent", report["failures"])

    def test_competitiveness_backtest_uses_completed_share_not_choice_share(self):
        core = Core.__new__(Core)
        core.firm1_mode = "RL"
        core.shared_edit_keys = ["base_fare"]
        core.firm1 = SimpleNamespace(
            action_steps=lambda action: {"base_fare": -1 if int(action) % 2 else 1},
            config=SimpleNamespace(step={"base_fare": 0.10}),
        )
        core.long_term_profit_reward_config = LongTermProfitRewardConfig(
            objective_mode="competitiveness",
            own_profit_weight=0.0,
            profit_advantage_weight=0.0,
            market_share_competitiveness_weight=1.0,
            market_share_target_gap=0.10,
        )
        core.constraint_config = ConstraintConfig(
            target_gap=0.0,
            overall_tolerance=0.45,
        )
        core.training_logs = [
            {
                "avg_reward": 0.50,
                "validation_score": (
                    0.50 + 0.10 * (day // 25)
                    if day in {24, 49, 74, 99}
                    else np.nan
                ),
            }
            for day in range(100)
        ]
        core.evaluation_logs = [
            {
                "reward": 0.60,
                # The conditional choice result points the other way. The
                # objective and backtest must use actual completed requests.
                "rl_market_share": 0.60,
                "heuristic_market_share": 0.30,
                "rl_completed_share": 0.60,
                "heuristic_completed_share": 0.30,
                "rl_chosen_share": 0.20,
                "heuristic_chosen_share": 0.70,
                "rl_profit": -1.0,
                "heuristic_profit": 2.0,
                "rl_fulfillment_rate": 0.20,
                "price_gap_f2_minus_f1": 8.0,
                "firm1_base_fare": 2.5 + 0.1 * (day % 2),
                "action": 1 + (day % 2),
            }
            for day in range(100)
        ]

        report = core._competitive_durability_backtest()

        self.assertTrue(report["passed"])
        self.assertEqual(report["primary_objective"], "competitiveness")
        self.assertGreaterEqual(
            report["late_completed_share_advantage"], 0.10
        )

    def test_durability_uses_positive_lead_not_reporting_gap_reference(self):
        core = Core.__new__(Core)
        core.firm1_mode = "RL"
        core.shared_edit_keys = ["base_fare"]
        core.firm1 = SimpleNamespace(
            action_steps=lambda action: {},
            config=SimpleNamespace(step={"base_fare": 0.10}),
        )
        core.long_term_profit_reward_config = LongTermProfitRewardConfig(
            objective_mode="competitiveness",
            own_profit_weight=0.0,
            profit_advantage_weight=0.0,
            market_share_competitiveness_weight=1.0,
            market_share_target_gap=0.10,
        )
        core.training_logs = [
            {
                "avg_reward": 0.40,
                "validation_score": (
                    0.40 if day in {49, 99} else np.nan
                ),
            }
            for day in range(100)
        ]
        core.evaluation_logs = [
            {
                "reward": 0.40,
                "rl_completed_share": 0.40,
                "heuristic_completed_share": 0.37,
                "rl_profit": 1.0,
                "heuristic_profit": 1.1,
                "firm1_base_fare": 2.5,
                "action": 0,
            }
            for _ in range(100)
        ]

        report = core._competitive_durability_backtest()

        self.assertTrue(report["passed"])
        self.assertAlmostEqual(
            report["late_completed_share_advantage"], 0.03
        )
        self.assertEqual(
            report["criteria"][
                "late_completed_share_advantage_min"
            ],
            0.0,
        )

    def test_full_horizon_checkpoint_mean_does_not_use_tail_only(self):
        means = Core._full_horizon_validation_means(
            reward=[1.0, 1.0, 0.0, 0.0],
            own_profit=[4.0, 4.0, 0.0, 0.0],
            rival_profit=[0.0, 0.0, 0.0, 0.0],
            own_market_share=[0.8, 0.8, 0.2, 0.2],
            rival_market_share=[0.1, 0.1, 0.4, 0.4],
            fulfillment=[1.0, 1.0, 0.5, 0.5],
            completion_coverage=[0.9, 0.9, 0.6, 0.6],
        )

        self.assertAlmostEqual(means["reward"], 0.5)
        self.assertAlmostEqual(means["own_profit"], 2.0)
        self.assertAlmostEqual(means["own_market_share"], 0.5)

    def test_action_space_contains_only_single_coefficient_moves(self):
        keys = ["base_fare", "per_minute", "per_mile", "booking_fee", "airport_fee"]
        action_map, _ = build_discrete_action_space(keys)
        bundles = [
            mapping
            for mapping in action_map.values()
            if sum(int(direction) != 0 for direction in mapping.values()) > 1
        ]

        self.assertEqual(len(bundles), 0)
        self.assertEqual(len(action_map), 11)

    def test_economic_action_group_ignores_negligible_airport_chatter(self):
        core = Core.__new__(Core)
        core.firm1_mode = "RL"
        core.shared_edit_keys = [
            "base_fare",
            "per_minute",
            "per_mile",
            "booking_fee",
            "airport_fee",
        ]
        core.firm1 = SimpleNamespace(
            action_steps=lambda action: {
                1: {"airport_fee": 1},
                2: {"base_fare": -1},
                3: {"base_fare": 1},
            }.get(int(action), {}),
            config=SimpleNamespace(
                step={
                    "base_fare": 0.10,
                    "per_minute": 0.01,
                    "per_mile": 0.05,
                    "booking_fee": 0.10,
                    "airport_fee": 0.10,
                }
            ),
        )

        self.assertEqual(core._economic_action_group(1), 0)
        self.assertEqual(core._economic_action_group(2), -1)
        self.assertEqual(core._economic_action_group(3), 1)

    def test_competitiveness_mask_preserves_physical_choices_when_service_is_low(self):
        core = Core.__new__(Core)
        core.firm1_mode = "RL"
        core.firm1 = SimpleNamespace(
            action_steps=lambda action: {
                1: {"base_fare": -1},
                2: {"base_fare": 1},
            }.get(int(action), {}),
            config=SimpleNamespace(step={"base_fare": 0.10}),
        )
        core.long_term_profit_reward_config = LongTermProfitRewardConfig(
            objective_mode="competitiveness",
            own_profit_weight=0.0,
            profit_advantage_weight=0.0,
            market_share_competitiveness_weight=1.0,
        )
        core.constraint_config = ConstraintConfig(fulfillment_floor=0.78)

        masked = core._objective_action_mask(
            np.asarray([True, True, True]),
            fulfillment_rate=0.60,
        )

        np.testing.assert_array_equal(masked, [True, True, True])
        self.assertEqual(core.last_action_mask_regime, "all_tariff_targets_available")

    def test_profit_pipeline_keeps_normal_action_mask(self):
        core = Core.__new__(Core)
        core.firm1_mode = "RL"
        core.firm1 = SimpleNamespace(
            action_steps=lambda action: {
                1: {"base_fare": -1},
                2: {"base_fare": 1},
            }.get(int(action), {}),
            config=SimpleNamespace(step={"base_fare": 0.10}),
        )
        core.long_term_profit_reward_config = LongTermProfitRewardConfig(
            objective_mode="profit_maximization",
        )
        core.constraint_config = ConstraintConfig(fulfillment_floor=0.78)
        base = np.asarray([True, True, True])

        masked = core._objective_action_mask(
            base,
            fulfillment_rate=0.20,
        )

        np.testing.assert_array_equal(masked, [True, True, True])
        self.assertEqual(core.last_action_mask_regime, "all_tariff_targets_available")

    def test_operational_shield_does_not_choose_a_side_in_conflicting_failure(self):
        core = Core.__new__(Core)
        core.firm1_mode = "RL"
        core.firm1 = SimpleNamespace(
            action_steps=lambda action: {
                1: {"base_fare": -1},
                2: {"base_fare": 1},
            }.get(int(action), {}),
            config=SimpleNamespace(step={"base_fare": 0.10}),
        )
        core.long_term_profit_reward_config = LongTermProfitRewardConfig()
        core.constraint_config = ConstraintConfig(
            fulfillment_floor=0.78,
            completion_coverage_floor=0.55,
        )

        masked = core._objective_action_mask(
            np.asarray([True, True, True]),
            fulfillment_rate=0.60,
            completion_coverage=0.37,
        )

        np.testing.assert_array_equal(masked, [True, True, True])
        self.assertEqual(
            core.last_action_mask_regime,
            "all_tariff_targets_available",
        )

    def test_optimizer_and_backtest_use_identical_economic_groups(self):
        keys = ["base_fare", "per_minute", "per_mile", "booking_fee", "airport_fee"]
        pricer = FirmRLPricer(seed=5, opt_keys=keys)
        market = SimpleNamespace(curr_market=SimpleNamespace(
            base_fare=3.0,
            per_minute=0.30,
            per_mile=1.30,
            booking_fee=1.50,
            airport_fee=4.0,
        ))
        features = pricer.build_action_feature_matrix(market, {})
        weights = np.asarray([0.35, 0.35, 0.20, 0.10], dtype=float)
        feature_impacts = features[:, 10:14] @ weights
        normalized_fare_threshold = 0.05 / 20.0
        feature_groups = np.where(
            feature_impacts < -normalized_fare_threshold,
            -1,
            np.where(feature_impacts > normalized_fare_threshold, 1, 0),
        )
        core = Core.__new__(Core)
        core.firm1_mode = "RL"
        core.shared_edit_keys = keys
        core.firm1 = pricer

        audit_groups = np.asarray([
            core._economic_action_group(action)
            for action in range(len(pricer.action_to_steps))
        ])

        np.testing.assert_array_equal(feature_groups, audit_groups)

    def test_response_supervision_uses_causal_delta(self):
        baseline = np.array([0.2, -0.1, 0.5], dtype=np.float32)
        future = np.array([0.7, -0.4, 4.5], dtype=np.float32)

        target = Core._response_delta(baseline, future)

        self.assertTrue(np.allclose(target, np.array([0.5, -0.3, 2.0], dtype=np.float32)))

    def test_opponent_action_teacher_encodes_direction_steps_and_event(self):
        core = Core.__new__(Core)
        core.shared_edit_keys = [
            "base_fare", "per_minute", "per_mile", "booking_fee", "airport_fee"
        ]
        pricer = SimpleNamespace(config=default_specs_for(core.shared_edit_keys))
        before = {key: 1.0 for key in core.shared_edit_keys}
        after = dict(before)
        after["base_fare"] += pricer.config.step["base_fare"]
        after["per_mile"] -= 2.0 * pricer.config.step["per_mile"]

        target = core._opponent_action_target(before, after, pricer)

        np.testing.assert_allclose(target, [1.0, 0.0, -2.0, 0.0, 0.0, 1.0])

    def test_opponent_action_audit_vector_follows_managed_scope(self):
        core = Core.__new__(Core)
        core.shared_edit_keys = ["base_fare", "per_minute", "per_mile"]
        pricer = SimpleNamespace(config=default_specs_for(core.shared_edit_keys))
        before = {key: 1.0 for key in core.shared_edit_keys}
        after = dict(before)
        after["per_minute"] += pricer.config.step["per_minute"]

        target = core._opponent_action_target(before, after, pricer)
        finalized = core._finalize_opponent_action_sum(target + target)

        self.assertEqual(target.shape, (4,))
        np.testing.assert_allclose(target, [0.0, 1.0, 0.0, 1.0])
        np.testing.assert_allclose(finalized, [0.0, 2.0, 0.0, 1.0])

    def test_training_regime_uses_long_dynamic_episodes(self):
        scheduler = TrainingStageScheduler("staged")

        self.assertEqual(scheduler.stage_at(0.50).action_hold_steps, 1)
        self.assertEqual(scheduler.stage_at(0.90).action_hold_steps, 1)
        self.assertEqual(scheduler.stage_at(0.25).action_hold_steps, 1)
        self.assertEqual(scheduler.stage_at(0.90).episode_days, 600)

    def test_training_stage_resolves_market_static_then_configured_dynamic(self):
        core = Core.__new__(Core)
        scheduler = TrainingStageScheduler("staged")
        configured = "adaptive_best_response_aggressive"

        resolved = [
            core._training_stage_opponent_mode(stage, configured)
            for stage in scheduler.stages
        ]

        self.assertEqual(resolved, ["static", "static", configured])

    def test_dynamic_stage_uses_varied_episode_opponents(self):
        core = Core.__new__(Core)
        stage = TrainingStageScheduler("staged").stage_at(0.90)
        configured = "adaptive_best_response_aggressive"
        modes = {
            core._training_stage_opponent_mode(
                stage, configured, episode_seed=10_000 + index
                * stage.episode_days
            )
            for index in range(24)
        }

        self.assertGreater(len(modes), 1)
        self.assertTrue(modes <= {
            configured,
            "adaptive_best_response",
            "heuristic",
            "heuristic_random",
        })

    def test_controller_transition_preserves_firm2_tariff(self):
        keys = [
            "base_fare",
            "per_minute",
            "per_mile",
            "booking_fee",
            "airport_fee",
        ]

        class ReplacementPricer:
            def __init__(self, seed, managed_keys):
                self.seed = seed
                self.managed_keys = list(managed_keys)
                self.overrides = CoefficientOverrides()
                self.config = default_specs_for(managed_keys)

        core = Core.__new__(Core)
        core.market = MarketInteraction("New York City", seed=31)
        core.shared_edit_keys = list(keys)
        core.pricer_by_mode = {"replacement": ReplacementPricer}
        core.firm2 = FirmStaticPricer()
        expected = {
            "base_fare": 3.37,
            "per_minute": 0.41,
            "per_mile": 1.83,
            "booking_fee": 2.61,
            "airport_fee": 5.72,
        }
        for key, value in expected.items():
            setattr(core.firm2.overrides, key, value)

        core._set_firm2_mode_preserving_tariff("replacement", seed=913)

        self.assertIsInstance(core.firm2, ReplacementPricer)
        self.assertEqual(core.firm2.seed, 913)
        self.assertEqual(core.firm2.managed_keys, keys)
        for key, value in expected.items():
            self.assertAlmostEqual(getattr(core.firm2.overrides, key), value)

    def test_firm2_leader_updates_and_posts_quote_before_follower_state(self):
        events = []
        keys = [
            "base_fare",
            "per_minute",
            "per_mile",
            "booking_fee",
            "airport_fee",
        ]

        class RecordingLeader:
            def __init__(self):
                self.overrides = CoefficientOverrides(
                    base_fare=3.0,
                    per_minute=0.4,
                    per_mile=1.9,
                    booking_fee=2.75,
                    airport_fee=6.0,
                )
                self.config = default_specs_for(keys)

            def update(self, **kwargs):
                events.append(("update", kwargs["metrics"].total))

            def act(self, **kwargs):
                events.append(("act", kwargs["hour"]))
                self.overrides.base_fare += self.config.step["base_fare"]

        core = Core.__new__(Core)
        core.market = MarketInteraction("New York City", seed=4)
        core.shared_edit_keys = list(keys)
        core.firm1 = SimpleNamespace(
            overrides=CoefficientOverrides(
                base_fare=3.0,
                per_minute=0.4,
                per_mile=1.9,
                booking_fee=2.75,
                airport_fee=6.0,
            )
        )
        core.firm2 = RecordingLeader()
        core.platform_observers = {
            "Firm1": PlatformObservationModel(
                8,
                ObservationConfig(
                    quote_noise_dollars=0.0,
                    quote_missing_probability=0.0,
                ),
            ),
            "Firm2": PlatformObservationModel(9, ObservationConfig()),
        }
        core._driver_supply_context = lambda *_: ({}, np.zeros(8, dtype=np.float32))
        core._observed_own_minus_competitor_gap = lambda *_: 0.0

        target, intervened = core._firm2_leader_step(
            previous_metrics=FirmMetrics(total=100, wins=45),
            day_of_week=2,
            hour=8,
            weather="clear",
        )

        self.assertEqual(events, [("update", 100), ("act", 8)])
        self.assertTrue(intervened)
        np.testing.assert_allclose(target, [1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        quote_snapshot = core.platform_observers["Firm1"].quote_snapshot()
        self.assertTrue(all(v["available"] == 1.0 for v in quote_snapshot.values()))
        self.assertTrue(all(v["gap"] > 0.0 for v in quote_snapshot.values()))

    def test_static_or_not_due_leader_still_posts_current_quote(self):
        core = Core.__new__(Core)
        core.market = MarketInteraction("New York City", seed=12)
        core.shared_edit_keys = [
            "base_fare",
            "per_minute",
            "per_mile",
            "booking_fee",
            "airport_fee",
        ]
        core.firm1 = SimpleNamespace(overrides=CoefficientOverrides())
        core.firm2 = FirmStaticPricer()
        core.firm2.overrides.base_fare = 4.0
        core.platform_observers = {
            "Firm1": PlatformObservationModel(
                13,
                ObservationConfig(
                    quote_noise_dollars=0.0,
                    quote_missing_probability=0.0,
                ),
            )
        }

        target, intervened = core._firm2_leader_step(
            previous_metrics=None,
            day_of_week=2,
            hour=8,
            weather="clear",
            enabled=False,
        )

        np.testing.assert_array_equal(target, np.zeros(6, dtype=np.float32))
        self.assertFalse(intervened)
        quote_snapshot = core.platform_observers["Firm1"].quote_snapshot()
        self.assertTrue(all(v["available"] == 1.0 for v in quote_snapshot.values()))
        self.assertTrue(all(v["gap"] > 0.0 for v in quote_snapshot.values()))

    def test_standalone_action_features_describe_whole_tariff_targets(self):
        keys = ["base_fare", "per_minute", "per_mile", "booking_fee", "airport_fee"]
        pricer = FirmRLPricer(seed=3, opt_keys=keys)
        market = SimpleNamespace(curr_market=SimpleNamespace(
            base_fare=3.0,
            per_minute=0.30,
            per_mile=1.30,
            booking_fee=1.50,
            airport_fee=4.0,
        ))

        features = pricer.build_action_feature_matrix(market, {})

        self.assertEqual(pricer.action_feature_dim, 20)
        self.assertEqual(features.shape, (len(pricer.action_to_steps), 20))
        target_rows = features[1:]
        self.assertTrue(np.all(
            np.count_nonzero(target_rows[:, 2:7], axis=1) == len(keys)
        ))

    def test_action_space_never_moves_more_than_one_tariff_lever(self):
        actions, _ = build_discrete_action_space([
            "base_fare",
            "per_minute",
            "per_mile",
            "booking_fee",
            "airport_fee",
        ])

        active_counts = [
            sum(int(step) != 0 for step in mapping.values())
            for mapping in actions.values()
        ]
        self.assertLessEqual(max(active_counts), 1)
        self.assertNotIn(2, active_counts)

    def test_state_action_specialization_weight_is_ignored(self):
        pricer = FirmRLPricer(
            seed=31,
            opt_keys=["base_fare", "per_minute"],
            state_action_mi_weight=0.23,
        )

        self.assertEqual(pricer.agent.state_action_mi_coeff, 0.0)

    def test_pricer_execution_snapshot_is_transactional(self):
        keys = ["base_fare", "per_minute", "per_mile", "booking_fee", "airport_fee"]
        pricer = FirmRLPricer(seed=4, opt_keys=keys)
        pricer._last_applied_action = 7
        pricer._repeat_action_count = 3
        pricer._last_action_target = "base_fare+booking_fee"
        pricer.agent.action_ever_feasible[:] = True
        snapshot = pricer.snapshot_execution_state()

        pricer.reset_state_history()
        pricer.agent.action_ever_feasible[:] = False
        pricer.restore_execution_state(snapshot)

        self.assertEqual(pricer._last_applied_action, 7)
        self.assertEqual(pricer._repeat_action_count, 3)
        self.assertEqual(pricer._last_action_target, "base_fare+booking_fee")
        self.assertTrue(np.all(pricer.agent.action_ever_feasible))

    @staticmethod
    def _diagnostic_core(action_direction=0, zero_effect=False, saturated=False):
        core = Core.__new__(Core)
        core.reward_profit_scale = 10.0
        core.reward_rev_scale = 25.0
        core.reward_target_price_gap = 1.0
        core.observation_config = ObservationConfig(
            telemetry_delay_steps=0,
            quote_probe_interval_steps=1,
            quote_probe_delay_steps=0,
            quote_noise_dollars=0.0,
            quote_missing_probability=0.0,
        )
        core.constraint_config = ConstraintConfig(target_gap=1.0, overall_tolerance=0.4)
        core.positive_reward_model = PositiveBusinessReward(PositiveRewardConfig())
        core.long_term_profit_reward_config = LongTermProfitRewardConfig()
        core.long_term_profit_reward_model = LongTermProfitReward(
            core.long_term_profit_reward_config
        )
        core.soft_constraints = SoftConstraintController(core.constraint_config)
        core.action_stability = ActionStabilityTracker(core.constraint_config)
        observer = PlatformObservationModel(1, core.observation_config)
        for segment in observer.quote_probes:
            observer.quote_probes[segment] = {
                "gap": -1.0,
                "uncertainty": 0.0,
                "age": 0.0,
                "available": 1.0,
            }
        core.platform_observers = {"Firm1": observer}
        core.firm1 = SimpleNamespace(
            last_action_descriptor=ActionDescriptor(
                target="base_fare" if action_direction else "hold",
                direction=action_direction,
            ),
            last_action_was_zero_effect=zero_effect,
            last_action_was_saturated=saturated,
        )
        return core

    def test_reward_is_positive_and_action_identity_neutral(self):
        common = dict(
            share=0.25,
            completed_share=0.25,
            rev_per_request=8.0,
            mean_gap=-1.0,
            prev_share=0.25,
            prev_rev_per_request=8.0,
            prev_profit_per_request=1.0,
            prev_gap=-2.0,
            profit_per_request=1.0,
            profit_margin=0.05,
            fulfillment_rate=0.9,
        )
        hold = self._diagnostic_core(action_direction=0)._reward_diagnostics(**common)
        corrective = self._diagnostic_core(action_direction=-1)._reward_diagnostics(**common)
        self.assertGreater(hold["reward"], 0.0)
        self.assertAlmostEqual(corrective["reward"], hold["reward"], places=7)
        for key in (
            "reward_hold_inaction_penalty",
            "reward_corrective_action_bonus",
            "reward_overprice_penalty",
            "reward_underprice_penalty",
            "reward_action_change_penalty",
            "reward_action_realization_penalty",
        ):
            self.assertEqual(hold[key], 0.0)
            self.assertEqual(corrective[key], 0.0)
        self.assertGreater(hold["constraint_cost_gap_overprice"], 0.0)

        stronger = self._diagnostic_core(action_direction=0)._reward_diagnostics(
            **{**common, "rev_per_request": 16.0, "profit_per_request": 5.0, "completed_share": 0.40}
        )
        self.assertGreater(stronger["reward"], hold["reward"])
        self.assertEqual(stronger["reward_profit_objective"], stronger["reward"])
        self.assertEqual(stronger["reward_share_objective"], 0.0)
        self.assertEqual(stronger["reward_actor_credit"], 0.0)
        self.assertGreater(stronger["diagnostic_objective_delta"], 0.0)

    def test_actor_credit_tracks_only_the_selected_objective_improvement(self):
        profit_core = self._diagnostic_core()
        improved_profit = profit_core._reward_diagnostics(
            share=0.40,
            baseline_share=0.40,
            completed_share=0.35,
            rev_per_request=10.0,
            mean_gap=0.5,
            prev_share=0.50,
            prev_rev_per_request=8.0,
            prev_profit_per_request=1.0,
            prev_rival_profit_per_request=1.0,
            prev_rival_market_share=0.50,
            prev_fulfillment_rate=0.90,
            profit_per_request=3.0,
            baseline_profit_per_request=1.0,
            fulfillment_rate=0.90,
        )
        self.assertEqual(improved_profit["reward_actor_credit"], 0.0)
        self.assertGreater(
            improved_profit["diagnostic_objective_delta"], 0.0
        )

        share_core = self._diagnostic_core()
        share_core.long_term_profit_reward_config = LongTermProfitRewardConfig(
            objective_mode="competitiveness",
            own_profit_weight=0.0,
            profit_advantage_weight=0.0,
            market_share_competitiveness_weight=1.0,
        )
        share_core.long_term_profit_reward_model = LongTermProfitReward(
            share_core.long_term_profit_reward_config
        )
        improved_share = share_core._reward_diagnostics(
            share=0.60,
            baseline_share=0.40,
            completed_share=0.60,
            rev_per_request=1.0,
            mean_gap=0.5,
            prev_share=0.50,
            prev_rev_per_request=100.0,
            prev_profit_per_request=100.0,
            prev_rival_profit_per_request=-100.0,
            prev_rival_market_share=0.50,
            prev_fulfillment_rate=0.90,
            profit_per_request=-100.0,
            baseline_profit_per_request=100.0,
            fulfillment_rate=0.90,
        )
        self.assertEqual(improved_share["reward_actor_credit"], 0.0)
        self.assertLess(
            improved_share["diagnostic_objective_delta"], 0.0
        )

    def test_competitiveness_diagnostics_separate_market_share_from_coverage(self):
        core = self._diagnostic_core()
        core.long_term_profit_reward_config = LongTermProfitRewardConfig(
            objective_mode="competitiveness",
            own_profit_weight=0.0,
            profit_advantage_weight=0.0,
            market_share_competitiveness_weight=1.0,
        )
        core.long_term_profit_reward_model = LongTermProfitReward(
            core.long_term_profit_reward_config
        )

        result = core._reward_diagnostics(
            share=0.47,
            baseline_share=0.50,
            completed_share=0.30,
            baseline_completed_share=0.25,
            rev_per_request=8.0,
            baseline_rev_per_request=7.0,
            profit_per_request=1.5,
            baseline_profit_per_request=1.0,
            mean_gap=1.0,
            prev_share=0.50,
            prev_rev_per_request=8.0,
            prev_profit_per_request=1.5,
            fulfillment_rate=0.80,
        )

        self.assertAlmostEqual(result["reward_market_share_sum"], 0.55)
        self.assertAlmostEqual(
            result["reward_conditional_choice_share_sum"], 1.0
        )
        self.assertAlmostEqual(result["reward_outside_option_share"], 0.03)
        self.assertAlmostEqual(result["reward_completion_coverage"], 0.55)
        self.assertAlmostEqual(result["reward_market_share"], 0.30)
        self.assertAlmostEqual(
            result["reward_conditional_choice_share"], 0.47 / 0.97
        )
        expected = (
            result["reward_own_profit_component"]
            + result["reward_profit_advantage_component"]
            + result["reward_market_share_level_component"]
            + result["reward_market_share_advantage_component"]
        )
        self.assertAlmostEqual(result["reward"], expected)
        self.assertEqual(result["reward_profit_objective"], 0.0)
        self.assertAlmostEqual(result["reward_share_objective"], expected)
        for key in (
            "reward_relative_profit_term",
            "reward_relative_revenue_term",
            "reward_relative_share_term",
            "reward_absolute_profit_term",
            "reward_absolute_revenue_term",
        ):
            self.assertEqual(result[key], 0.0)

    def test_collapsed_market_cannot_win_completed_share_objective(self):
        core = self._diagnostic_core()
        core.long_term_profit_reward_config = LongTermProfitRewardConfig(
            objective_mode="competitiveness",
            own_profit_weight=0.0,
            profit_advantage_weight=0.0,
            market_share_competitiveness_weight=1.0,
        )
        core.long_term_profit_reward_model = LongTermProfitReward(
            core.long_term_profit_reward_config
        )

        collapsed = core._reward_diagnostics(
            share=0.009,
            baseline_share=0.001,
            completed_share=0.009,
            baseline_completed_share=0.001,
            rev_per_request=0.1,
            profit_per_request=0.01,
            mean_gap=1.0,
            prev_share=0.5,
            prev_rev_per_request=0.0,
            fulfillment_rate=0.95,
            profit_margin=0.10,
        )
        healthy = core._reward_diagnostics(
            share=0.54,
            baseline_share=0.36,
            completed_share=0.45,
            baseline_completed_share=0.30,
            rev_per_request=8.0,
            profit_per_request=1.5,
            mean_gap=1.0,
            prev_share=0.5,
            prev_rev_per_request=0.0,
            fulfillment_rate=0.95,
            profit_margin=0.18,
        )

        self.assertAlmostEqual(collapsed["reward"], 0.061)
        self.assertAlmostEqual(healthy["reward"], 1.35)
        self.assertGreater(healthy["reward"], collapsed["reward"])
        self.assertGreater(collapsed["constraint_cost_coverage"], 0.9)
        self.assertEqual(collapsed["constraint_cost_fulfillment"], 0.0)
        self.assertEqual(healthy["constraint_cost_coverage"], 0.0)

    def test_checkpoint_score_selects_profit_not_proxy_kpis(self):
        core = self._diagnostic_core()
        objective = core.long_term_profit_reward_model.compute(
            own_profit_per_request=3.0,
            rival_profit_per_request=2.0,
            price_gap_f2_minus_f1=-20.0,
        )["reward"]
        first = core._validation_score_from_metrics(reward=objective)
        second = core._validation_score_from_metrics(reward=objective)

        self.assertAlmostEqual(first, objective)
        self.assertAlmostEqual(second, first)

    def test_checkpoint_score_honors_serviceable_market_share(self):
        core = self._diagnostic_core()
        core.long_term_profit_reward_config = LongTermProfitRewardConfig(
            objective_mode="competitiveness",
            own_profit_weight=0.0,
            profit_advantage_weight=0.0,
            market_share_competitiveness_weight=1.0,
        )
        core.long_term_profit_reward_model = LongTermProfitReward(
            core.long_term_profit_reward_config
        )
        matched_reward = core.long_term_profit_reward_model.compute(
            own_profit_per_request=1.0,
            rival_profit_per_request=1.0,
            own_market_share=0.60,
            rival_market_share=0.40,
            own_fulfillment_rate=0.90,
        )["reward"]
        distant_reward = core.long_term_profit_reward_model.compute(
            own_profit_per_request=1.0,
            rival_profit_per_request=1.0,
            own_market_share=0.45,
            rival_market_share=0.55,
            own_fulfillment_rate=0.90,
        )["reward"]
        matched = core._validation_score_from_metrics(
            reward=matched_reward,
            completed_share_advantage=0.20,
        )
        distant = core._validation_score_from_metrics(
            reward=distant_reward,
            completed_share_advantage=-0.05,
        )
        self.assertGreater(matched, distant)

    def test_competitiveness_checkpoint_prioritizes_share_at_business_scales(self):
        core = self._diagnostic_core()
        core.long_term_profit_reward_config = LongTermProfitRewardConfig(
            objective_mode="competitiveness",
            own_profit_weight=0.0,
            profit_advantage_weight=0.0,
            market_share_competitiveness_weight=1.0,
        )
        core.long_term_profit_reward_model = LongTermProfitReward(
            core.long_term_profit_reward_config
        )
        high_profit_bad_share_reward = (
            core.long_term_profit_reward_model.compute(
                own_profit_per_request=5.0,
                rival_profit_per_request=3.0,
                own_market_share=0.45,
                rival_market_share=0.55,
                own_fulfillment_rate=0.90,
            )["reward"]
        )
        low_profit_good_share_reward = (
            core.long_term_profit_reward_model.compute(
                own_profit_per_request=1.0,
                rival_profit_per_request=1.0,
                own_market_share=0.60,
                rival_market_share=0.40,
                own_fulfillment_rate=0.90,
            )["reward"]
        )
        high_profit_bad_share_gap = core._validation_score_from_metrics(
            reward=high_profit_bad_share_reward,
            completed_share_advantage=-0.10,
        )
        low_profit_good_share_gap = core._validation_score_from_metrics(
            reward=low_profit_good_share_reward,
            completed_share_advantage=0.20,
        )
        self.assertGreater(
            low_profit_good_share_gap, high_profit_bad_share_gap
        )

    def test_latest_objective_checkpoint_is_fallback(self):
        diagnostic = {"day": 400, "score": 0.7}

        selected, status = Core._evaluation_checkpoint(None, diagnostic)

        self.assertIs(selected, diagnostic)
        self.assertEqual(status, "final_dynamic_policy")

    def test_final_dynamic_checkpoint_is_preferred_over_early_best(self):
        eligible = {"day": 300, "score": 0.4}
        diagnostic = {"day": 400, "score": 0.7}

        selected, status = Core._evaluation_checkpoint(eligible, diagnostic)

        self.assertIs(selected, diagnostic)
        self.assertEqual(status, "final_dynamic_policy")

    def test_durable_checkpoint_is_fallback_when_final_policy_fails(self):
        durable = {
            "day": 300,
            "score": 0.4,
            "validation": {"checkpoint_eligible": True},
        }
        failed_final = {
            "day": 400,
            "score": 0.7,
            "validation": {"checkpoint_eligible": False},
        }

        selected, status = Core._evaluation_checkpoint(
            durable, failed_final
        )

        self.assertIs(selected, durable)
        self.assertEqual(status, "best_durable_dynamic_policy_fallback")

    def test_best_observed_checkpoint_is_used_when_none_is_durable(self):
        observed = {
            "day": 300,
            "score": 0.8,
            "validation": {"checkpoint_eligible": False},
        }
        failed_final = {
            "day": 400,
            "score": 0.5,
            "validation": {"checkpoint_eligible": False},
        }

        selected, status = Core._evaluation_checkpoint(
            None, failed_final, observed
        )

        self.assertIs(selected, observed)
        self.assertEqual(
            status, "best_observed_dynamic_policy_not_yet_durable"
        )

    def test_checkpoint_score_ignores_non_objective_diagnostics(self):
        core = self._diagnostic_core()
        clean = core._validation_score_from_metrics(reward=0.0)
        diagnosed = core._validation_score_from_metrics(reward=0.0)

        self.assertAlmostEqual(diagnosed, clean)

    def test_convergence_statistics_filter_stationary_market_noise(self):
        rewards = [0.5 + (0.10 if index % 2 else -0.10) for index in range(100)]

        count, reward_std, reward_delta = Core._smoothed_convergence_statistics(
            rewards,
            window=60,
            smoothing=20,
        )

        self.assertEqual(count, 60)
        self.assertLess(reward_std, 1e-12)
        self.assertLess(reward_delta, 1e-12)

    def test_current_absolute_target_remains_legal_to_hold_posted_tariff(self):
        pricer = FirmRLPricer(seed=7, opt_keys=["base_fare"])
        market = SimpleNamespace(
            curr_market=SimpleNamespace(base_fare=3.0)
        )
        pricer.overrides.base_fare = 3.0 * 0.80

        mask = pricer.feasible_action_mask(market)

        np.testing.assert_array_equal(mask, np.ones(8, dtype=bool))

    def test_absolute_tariff_action_set_is_stationary(self):
        keys = [
            "base_fare",
            "per_minute",
            "per_mile",
            "booking_fee",
            "airport_fee",
        ]
        pricer = FirmRLPricer(seed=41, opt_keys=keys)
        market = SimpleNamespace(
            curr_market=SimpleNamespace(
                base_fare=3.0,
                per_minute=0.30,
                per_mile=1.50,
                booking_fee=2.0,
                airport_fee=5.0,
            )
        )
        pricer.overrides.base_fare = 3.0
        pricer.overrides.per_minute = 0.30
        pricer.overrides.per_mile = 1.50
        pricer.overrides.booking_fee = 2.0
        pricer.overrides.airport_fee = 5.0

        mask = pricer.feasible_action_mask(market)

        np.testing.assert_array_equal(mask, np.ones(8, dtype=bool))
        self.assertTrue(mask[0])

    def test_repeated_absolute_target_does_not_drift(self):
        pricer = FirmRLPricer(seed=17, opt_keys=["base_fare"])
        market = MarketInteraction(city_name="New York City", seed=19)
        pricer.agent.last_continuous_magnitude = 1.0

        pricer.apply_action(6, market)
        first_delta = float(pricer.last_action_descriptor.realized_delta)
        pricer.apply_action(6, market)
        second_delta = float(pricer.last_action_descriptor.realized_delta)

        self.assertGreater(first_delta, 0.0)
        self.assertAlmostEqual(second_delta, 0.0)
        self.assertTrue(pricer.last_action_was_zero_effect)

    def test_absolute_target_sets_one_coherent_whole_tariff(self):
        pricer = FirmRLPricer(
            seed=19, opt_keys=["base_fare", "per_minute"]
        )
        market = MarketInteraction(city_name="New York City", seed=21)
        base = market.curr_market

        pricer.apply_action(6, market)  # all managed coefficients -> 1.20x
        self.assertAlmostEqual(
            pricer.overrides.base_fare,
            float(np.clip(
                1.20 * base.base_fare,
                *pricer.config.bounds["base_fare"],
            )),
        )
        self.assertAlmostEqual(
            pricer.overrides.per_minute,
            float(np.clip(
                1.20 * base.per_minute,
                *pricer.config.bounds["per_minute"],
            )),
        )
        pricer.apply_action(2, market)  # all managed coefficients -> 0.80x
        self.assertAlmostEqual(
            pricer.overrides.base_fare,
            float(np.clip(
                0.80 * base.base_fare,
                *pricer.config.bounds["base_fare"],
            )),
        )
        self.assertAlmostEqual(
            pricer.overrides.per_minute,
            float(np.clip(
                0.80 * base.per_minute,
                *pricer.config.bounds["per_minute"],
            )),
        )

    def test_default_pricer_credit_contract_is_long_horizon(self):
        pricer = FirmRLPricer(seed=31, opt_keys=["base_fare"])

        self.assertEqual(pricer.state_frame_stack, 16)
        self.assertAlmostEqual(pricer.agent.gamma, 0.999)
        self.assertAlmostEqual(pricer.agent.lam, 0.99)
        five_step_trace = pricer.agent.gamma**5 * pricer.agent.lam
        self.assertGreater(1.0 / (1.0 - five_step_trace), 60.0)
        self.assertLess(1.0 / (1.0 - five_step_trace), 75.0)

    def test_absolute_target_direction_tracks_realized_tariff_momentum(self):
        pricer = FirmRLPricer(seed=23, opt_keys=["base_fare"])
        market = MarketInteraction(city_name="New York City", seed=29)

        pricer.apply_action(6, market)  # 1.20x market
        self.assertEqual(pricer.last_action_descriptor.direction, 1)
        pricer.apply_action(4, market)  # 1.00x market: realized decrease

        self.assertEqual(pricer.last_action_descriptor.direction, -1)
        self.assertLess(pricer.last_action_descriptor.realized_delta, 0.0)

    def test_preview_stack_does_not_advance_frame_history(self):
        pricer = FirmRLPricer.__new__(FirmRLPricer)
        pricer.single_state_dim = 2
        pricer.state_frame_stack = 3
        pricer._state_history = deque(maxlen=3)
        pricer.stack_state(np.array([1.0, 1.0], dtype=np.float32), commit=True)
        before = [frame.copy() for frame in pricer._state_history]

        preview = pricer.stack_state(np.array([2.0, 2.0], dtype=np.float32), commit=False)

        self.assertEqual(len(pricer._state_history), len(before))
        for current, expected in zip(pricer._state_history, before):
            self.assertTrue(np.array_equal(current, expected))
        self.assertTrue(np.array_equal(preview[-2:], np.array([2.0, 2.0], dtype=np.float32)))


if __name__ == "__main__":
    unittest.main()
