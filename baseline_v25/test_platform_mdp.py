import unittest

import numpy as np

from platform_mdp import (
    ActionStabilityTracker,
    ConstraintConfig,
    LongTermProfitReward,
    LongTermProfitRewardConfig,
    ObservationConfig,
    PlatformObservationModel,
    MarketShareCompetitivenessReward,
    MixedEconomicReward,
    ProfitMaximizationReward,
    RobustObjectiveOptimizer,
    PositiveBusinessReward,
    PositiveRewardConfig,
    SoftConstraintController,
    OperationalClock,
    TrainingStageScheduler,
    conditional_competitive_shares,
    objective_reward_weights,
    policy_objective_defaults,
)


class PlatformMDPTests(unittest.TestCase):
    def test_default_training_regime_is_three_environment_stages(self):
        scheduler = TrainingStageScheduler()

        self.assertEqual(scheduler.mode, "staged")
        self.assertEqual(
            [stage.name for stage in scheduler.stages],
            ["market", "static_opponent", "dynamic_opponent"],
        )
        self.assertEqual(
            [stage.opponent_regime for stage in scheduler.stages],
            ["market", "static", "dynamic"],
        )
        self.assertEqual(
            [stage.freeze_opponent for stage in scheduler.stages],
            [True, True, False],
        )
        self.assertEqual(
            [stage.episode_days for stage in scheduler.stages],
            [0, 0, 600],
        )
        self.assertTrue(all(stage.action_hold_steps == 1 for stage in scheduler.stages))
        self.assertEqual(
            {stage.tariff_reset_fraction for stage in scheduler.stages},
            {0.15},
        )

    def test_operational_clock_has_identical_cadence_across_workflows(self):
        train_clock = OperationalClock()
        eval_clock = OperationalClock()
        train_due = []
        eval_due = []
        for _ in range(13):
            train_due.append(train_clock.due(3))
            eval_due.append(eval_clock.due(3))
            train_clock.advance()
            eval_clock.advance()

        self.assertEqual(train_due, eval_due)
        self.assertEqual([i for i, due in enumerate(train_due) if due], [0, 3, 6, 9, 12])

    @staticmethod
    def _metrics():
        return {
            "chosen_share": 0.40,
            "completed_share": 0.36,
            "revenue_per_request": 11.0,
            "profit_per_request": 3.0,
            "fulfillment_rate": 0.90,
            "acceptance_rate": 0.92,
            "wait_minutes": 5.0,
            "driver_pay_per_request": 7.0,
        }

    def test_latent_rider_threshold_fields_do_not_enter_observation(self):
        config = ObservationConfig(
            telemetry_delay_steps=0,
            quote_probe_interval_steps=1,
            quote_probe_delay_steps=0,
            quote_noise_dollars=0.0,
            quote_missing_probability=0.0,
        )
        first = PlatformObservationModel(9, config)
        second = PlatformObservationModel(9, config)
        common = {
            "distance_mean": 5.0,
            "distance_std": 2.0,
            "distance_q25": 2.0,
            "distance_q75": 8.0,
            "duration_mean": 20.0,
            "duration_std": 8.0,
            "airport_rate": 0.2,
            "long_trip_share": 0.15,
            "distance_bin_0_2_price_gap_mean": 0.5,
            "distance_bin_2_5_price_gap_mean": 0.7,
            "distance_bin_5_10_price_gap_mean": 0.9,
            "distance_bin_10_plus_price_gap_mean": 1.1,
        }
        first.ingest(
            own_metrics=self._metrics(),
            supply_metrics={},
            crowd_stats={**common, "price_threshold_mean": 0.1, "low_income_share": 0.9},
        )
        second.ingest(
            own_metrics=self._metrics(),
            supply_metrics={},
            crowd_stats={**common, "price_threshold_mean": 99.0, "low_income_share": 0.0},
        )
        kwargs = dict(
            hour=8,
            day_of_week=2,
            weather="clear",
            own_coefficients={"base_fare": 3.0, "per_minute": 0.3, "per_mile": 1.8, "booking_fee": 2.0, "airport_fee": 5.0},
            anchor_coefficients={"base_fare": 3.0, "per_minute": 0.3, "per_mile": 1.8, "booking_fee": 2.0, "airport_fee": 5.0},
        )
        np.testing.assert_allclose(first.build_observation(**kwargs), second.build_observation(**kwargs))

    def test_action_features_expose_long_trip_fare_impact(self):
        observer = PlatformObservationModel(3, ObservationConfig())
        action_steps = {
            0: {"per_mile": 0},
            1: {"per_mile": -1},
            2: {"per_mile": 1},
        }
        features = observer.build_action_features(
            action_steps=action_steps,
            action_keys=["per_mile"],
            own_coefficients={"per_mile": 1.8},
            anchor_coefficients={"per_mile": 1.8},
            coefficient_steps={"per_mile": 0.1},
            coefficient_bounds={"per_mile": (0.5, 4.0)},
            step_scale=1.0,
            target_gap=0.75,
        )
        self.assertEqual(features.shape, (3, 20))
        self.assertGreater(abs(features[2, 13]), abs(features[2, 10]))

    def test_action_features_do_not_encode_a_target_price_gap(self):
        observer = PlatformObservationModel(3, ObservationConfig())
        kwargs = dict(
            action_steps={
                0: {},
                1: {"base_fare": -1},
                2: {"base_fare": 1},
            },
            action_keys=["base_fare"],
            own_coefficients={"base_fare": 3.0},
            anchor_coefficients={"base_fare": 3.0},
            coefficient_steps={"base_fare": 0.1},
            coefficient_bounds={"base_fare": (1.5, 5.0)},
            step_scale=1.0,
        )

        low_target = observer.build_action_features(target_gap=0.1, **kwargs)
        high_target = observer.build_action_features(target_gap=9.0, **kwargs)

        np.testing.assert_allclose(low_target, high_target)

    def test_reward_and_costs_are_separate(self):
        reward = PositiveBusinessReward().compute({
            "profit_per_request": 2.0,
            "revenue_per_request": 10.0,
            "completed_share": 0.30,
            "fulfillment_rate": 0.90,
            "acceptance_rate": 0.90,
            "wait_minutes": 5.0,
        })
        self.assertGreater(reward["reward"], 0.0)
        self.assertLessEqual(reward["reward"], 1.0)

        observer = PlatformObservationModel(4)
        for probe in observer.quote_probes.values():
            probe.update({"gap": -2.0, "uncertainty": 0.1, "available": 1.0})
        controller = SoftConstraintController(ConstraintConfig(target_gap=0.75))
        costs = controller.compute(
            observer=observer,
            fulfillment_rate=0.90,
            wait_minutes=5.0,
            profit_margin=0.20,
            oscillation_cost=0.0,
        )
        self.assertGreater(costs["gap_overprice"], 0.0)
        self.assertEqual(len(controller.vector(costs)), 4)
        diagnostics = controller.diagnostics(costs)
        self.assertEqual(diagnostics["constraint_active_gap_overprice"], 0.0)
        self.assertEqual(diagnostics["constraint_active_margin"], 1.0)
        self.assertEqual(diagnostics["constraint_active_oscillation"], 0.0)

    def test_long_term_profit_reward_has_exact_economic_decomposition(self):
        config = LongTermProfitRewardConfig()
        model = LongTermProfitReward(config)
        result = model.compute(
            own_profit_per_request=4.0,
            rival_profit_per_request=2.0,
            intervention_magnitude=0.4,
            reversal=1.0,
        )
        expected = 0.80 * (4.0 / 2.0) + 0.20 * (2.0 / 1.0)
        self.assertAlmostEqual(result["reward_raw"], expected)
        self.assertAlmostEqual(result["reward"], expected)
        self.assertAlmostEqual(
            result["reward_base"],
            result["reward_own_profit_component"]
            + result["reward_profit_advantage_component"]
            + result["reward_market_share_level_component"]
            + result["reward_market_share_advantage_component"],
        )
        self.assertAlmostEqual(result["reward_profit_advantage_component"], 0.4)
        self.assertEqual(result["reward_active_component_count"], 2.0)
        self.assertEqual(result["reward_applied_action_regularization"], 0.0)
        self.assertAlmostEqual(result["reward_profit_gain_utility"], 2.0)
        self.assertEqual(result["reward_profit_loss_utility"], 0.0)
        self.assertAlmostEqual(
            result["reward_profit_quality_gate"],
            1.0 - np.exp(-1.0),
        )

    def test_market_share_competitiveness_reward_tracks_target_lead(self):
        config = LongTermProfitRewardConfig(
            objective_mode="competitiveness",
            own_profit_weight=0.0,
            profit_advantage_weight=0.0,
            market_share_competitiveness_weight=1.0,
            market_share_target_gap=0.10,
            market_share_gap_scale=0.10,
            market_share_level_weight=0.0,
        )
        model = LongTermProfitReward(config)
        matched = model.compute(
            own_profit_per_request=1.0,
            rival_profit_per_request=1.0,
            own_completed_share=0.55,
            rival_completed_share=0.45,
        )
        distant = model.compute(
            own_profit_per_request=1.0,
            rival_profit_per_request=1.0,
            own_completed_share=0.45,
            rival_completed_share=0.55,
        )
        severe_loss = model.compute(
            own_profit_per_request=1.0,
            rival_profit_per_request=1.0,
            own_completed_share=0.05,
            rival_completed_share=0.95,
        )
        self.assertAlmostEqual(matched["reward"], 1.025)
        self.assertAlmostEqual(distant["reward"], -0.425)
        self.assertAlmostEqual(severe_loss["reward_raw"], -6.225)
        self.assertLess(severe_loss["reward"], distant["reward"])
        self.assertGreater(matched["reward"], distant["reward"])
        self.assertGreater(distant["reward_market_share_target_shortfall"], 0.0)
        self.assertAlmostEqual(
            matched["reward_market_share_advantage"], 0.10
        )

    def test_conditional_market_shares_sum_to_one_and_report_outside_option(self):
        own, rival, outside = conditional_competitive_shares(0.47, 0.50)

        self.assertAlmostEqual(own + rival, 1.0)
        self.assertAlmostEqual(outside, 0.03)
        self.assertAlmostEqual(own, 0.47 / 0.97)

    def test_conditional_market_shares_use_even_prior_when_market_is_empty(self):
        own, rival, outside = conditional_competitive_shares(0.0, 0.0)

        self.assertEqual((own, rival, outside), (0.5, 0.5, 1.0))

    def test_competitiveness_reward_is_dense_above_certification_target(self):
        model = LongTermProfitReward(LongTermProfitRewardConfig(
            objective_mode="competitiveness",
            own_profit_weight=0.0,
            profit_advantage_weight=0.0,
            market_share_competitiveness_weight=1.0,
        ))
        certified = model.compute(
            own_profit_per_request=1.0,
            rival_profit_per_request=1.0,
            own_market_share=0.56,
            rival_market_share=0.44,
            own_fulfillment_rate=0.90,
        )
        stronger = model.compute(
            own_profit_per_request=1.0,
            rival_profit_per_request=1.0,
            own_market_share=0.66,
            rival_market_share=0.34,
            own_fulfillment_rate=0.90,
        )

        self.assertAlmostEqual(certified["reward"], 1.17)
        self.assertAlmostEqual(stronger["reward"], 2.62)
        self.assertGreater(stronger["reward"], certified["reward"])

    def test_competitiveness_reward_rejects_unprofitable_race_to_bottom(self):
        model = LongTermProfitReward(LongTermProfitRewardConfig(
            objective_mode="competitiveness",
            own_profit_weight=0.0,
            profit_advantage_weight=0.0,
            market_share_competitiveness_weight=1.0,
        ))
        extreme_but_unserved = model.compute(
            own_profit_per_request=-10.0,
            rival_profit_per_request=20.0,
            own_market_share=0.60,
            rival_market_share=0.40,
            own_fulfillment_rate=0.42,
        )
        durable_lead = model.compute(
            own_profit_per_request=1.0,
            rival_profit_per_request=0.5,
            own_market_share=0.60,
            rival_market_share=0.40,
            own_fulfillment_rate=0.80,
        )

        self.assertLess(extreme_but_unserved["reward"], durable_lead["reward"])
        self.assertGreater(
            extreme_but_unserved["reward_market_share_service_penalty"], 0.0
        )

    def test_competitiveness_reward_does_not_double_count_service_constraint(self):
        model = LongTermProfitReward(LongTermProfitRewardConfig(
            objective_mode="competitiveness",
            own_profit_weight=0.0,
            profit_advantage_weight=0.0,
            market_share_competitiveness_weight=1.0,
            market_share_service_floor=0.78,
        ))
        extreme_but_below_floor = model.compute(
            own_profit_per_request=1.0,
            rival_profit_per_request=0.5,
            own_market_share=0.75,
            rival_market_share=0.25,
            own_fulfillment_rate=0.669,
        )
        durable_lead = model.compute(
            own_profit_per_request=1.0,
            rival_profit_per_request=0.5,
            own_market_share=0.75,
            rival_market_share=0.25,
            own_fulfillment_rate=0.80,
        )

        self.assertAlmostEqual(
            extreme_but_below_floor["reward"], durable_lead["reward"]
        )
        self.assertAlmostEqual(
            extreme_but_below_floor["reward_market_share_service_penalty"],
            5.0 * (0.78 - 0.669),
        )

    def test_profit_reward_is_uncapped_linear_and_zero_centered(self):
        model = LongTermProfitReward(LongTermProfitRewardConfig(
            profit_advantage_weight=0.0,
        ))
        gain = model.compute(
            own_profit_per_request=1.0,
            rival_profit_per_request=4.0,
        )
        loss = model.compute(
            own_profit_per_request=-1.0,
            rival_profit_per_request=4.0,
        )

        gain = model.compute(
            own_profit_per_request=1.0,
            rival_profit_per_request=1.0,
        )
        loss = model.compute(
            own_profit_per_request=-1.0,
            rival_profit_per_request=-1.0,
        )
        self.assertAlmostEqual(gain["reward"], 0.40)
        self.assertAlmostEqual(loss["reward"], -0.40)
        self.assertAlmostEqual(gain["reward"] + loss["reward"], 0.0)
        self.assertLess(loss["reward_raw"], 0.0)

    def test_profit_reward_preserves_ordering_beyond_old_clip_boundary(self):
        model = LongTermProfitReward(LongTermProfitRewardConfig(
            profit_advantage_weight=0.0,
        ))

        high = model.compute(
            own_profit_per_request=20.0,
            rival_profit_per_request=4.0,
        )["reward"]
        higher = model.compute(
            own_profit_per_request=36.0,
            rival_profit_per_request=4.0,
        )["reward"]
        loss = model.compute(
            own_profit_per_request=-12.0,
            rival_profit_per_request=4.0,
        )["reward"]
        worse_loss = model.compute(
            own_profit_per_request=-28.0,
            rival_profit_per_request=4.0,
        )["reward"]

        self.assertGreater(higher, high)
        self.assertLess(worse_loss, loss)

    def test_checkpoint_optimizer_uses_only_mean_scalar_objective(self):
        optimizer = RobustObjectiveOptimizer(
            objective="competitiveness",
            own_profit_scale=4.0,
            profit_advantage_target=0.25,
            market_share_advantage_target=0.10,
            fulfillment_floor=0.78,
            completion_coverage_floor=0.55,
        )
        feasible = optimizer.summarize(
            own_profit=[2.0, 2.0],
            rival_profit=[1.0, 1.0],
            own_market_share=[0.60, 0.61],
            rival_market_share=[0.40, 0.39],
            fulfillment=[0.90, 0.90],
            completion_coverage=[0.75, 0.75],
        )
        high_share_but_infeasible = optimizer.summarize(
            own_profit=[2.0, 2.0],
            rival_profit=[1.0, 1.0],
            own_market_share=[0.95, 0.95],
            rival_market_share=[0.05, 0.05],
            fulfillment=[0.50, 0.50],
            completion_coverage=[0.75, 0.75],
        )

        self.assertTrue(feasible["checkpoint_eligible"])
        self.assertTrue(high_share_but_infeasible["checkpoint_eligible"])
        self.assertTrue(optimizer.candidate_is_better(
            high_share_but_infeasible["selection_key"],
            feasible["selection_key"],
            require_eligible=False,
        ))
        self.assertGreater(
            high_share_but_infeasible["objective_score"],
            feasible["objective_score"],
        )

    def test_checkpoint_objective_ties_prefer_later_candidate(self):
        self.assertTrue(
            RobustObjectiveOptimizer.candidate_is_better(
                (1.25,), (1.25,), require_eligible=False
            )
        )

    def test_checkpoint_score_uses_same_reward_with_final_quarter_weight(self):
        optimizer = RobustObjectiveOptimizer(
            objective="profit_maximization",
            own_profit_scale=2.0,
            profit_advantage_target=0.0,
            market_share_advantage_target=0.0,
            fulfillment_floor=0.78,
            completion_coverage_floor=0.55,
            tail_weight=0.60,
        )
        summary = optimizer.summarize(
            own_profit=[2.0],
            rival_profit=[1.0],
            own_market_share=[0.50],
            rival_market_share=[0.30],
            fulfillment=[0.90],
            completion_coverage=[0.80],
            tail_objective_reward=[2.0],
        )

        full = summary["objective_mean"]
        self.assertAlmostEqual(
            summary["objective_score"], 0.40 * full + 0.60 * 2.0
        )
        self.assertEqual(len(summary["reward_weights"]), 4)

    def test_checkpoint_optimizer_rejects_invalid_rate_inputs(self):
        optimizer = RobustObjectiveOptimizer(
            objective="competitiveness",
            own_profit_scale=4.0,
            profit_advantage_target=0.25,
            market_share_advantage_target=0.10,
            fulfillment_floor=0.78,
            completion_coverage_floor=0.55,
        )
        common = dict(
            own_profit=[1.0],
            rival_profit=[1.0],
            rival_market_share=[0.20],
            fulfillment=[0.90],
            completion_coverage=[0.80],
        )

        with self.assertRaisesRegex(ValueError, "own_market_share"):
            optimizer.summarize(
                **common,
                own_market_share=[1.01],
            )

    def test_checkpoint_optimizer_requires_durable_dominance(self):
        optimizer = RobustObjectiveOptimizer(
            objective="profit_maximization",
            own_profit_scale=4.0,
            profit_advantage_target=0.25,
            market_share_advantage_target=0.10,
            fulfillment_floor=0.78,
            completion_coverage_floor=0.55,
            dominance_rate_floor=0.90,
        )
        summary = optimizer.summarize(
            own_profit=[3.0, 3.0],
            rival_profit=[2.0, 2.0],
            own_market_share=[0.50, 0.50],
            rival_market_share=[0.50, 0.50],
            fulfillment=[0.95, 0.95],
            completion_coverage=[0.80, 0.80],
            objective_dominance_rate=[0.95, 0.40],
        )

        self.assertFalse(summary["checkpoint_eligible"])
        self.assertAlmostEqual(
            summary["worst_objective_dominance_rate"], 0.40
        )
        self.assertEqual(summary["mode_feasible"], [1.0, 0.0])
        self.assertGreater(summary["maximum_normalized_shortfall"], 0.0)
        self.assertAlmostEqual(summary["objective_score"], 0.8)

    def test_market_share_competitiveness_ignores_public_quote_gap(self):
        model = LongTermProfitReward(LongTermProfitRewardConfig(
            objective_mode="competitiveness",
            own_profit_weight=0.0,
            profit_advantage_weight=0.0,
            market_share_competitiveness_weight=1.0,
        ))
        matched_quotes = model.compute(
            own_profit_per_request=1.0,
            rival_profit_per_request=1.0,
            own_completed_share=0.60,
            rival_completed_share=0.40,
            price_gap_f2_minus_f1=0.75,
            price_gap_abs_error=0.0,
        )
        distant_quotes = model.compute(
            own_profit_per_request=1.0,
            rival_profit_per_request=1.0,
            own_completed_share=0.60,
            rival_completed_share=0.40,
            price_gap_f2_minus_f1=8.0,
            price_gap_abs_error=7.25,
        )
        self.assertAlmostEqual(
            matched_quotes["reward"], distant_quotes["reward"]
        )
        self.assertEqual(
            matched_quotes["reward_price_competitiveness_component"], 0.0
        )

    def test_named_objective_profiles_select_profit_and_competitiveness(self):
        profit = policy_objective_defaults("profit_maximization")
        competitive = policy_objective_defaults("competitiveness")
        self.assertEqual(profit["long_term_profit_weight"], 0.80)
        self.assertEqual(profit["profit_dominance_weight"], 0.20)
        self.assertEqual(profit["market_share_level_weight"], 0.00)
        self.assertEqual(profit["market_share_competitiveness_weight"], 0.00)
        self.assertEqual(profit["price_competitiveness_weight"], 0.0)
        self.assertEqual(profit["intervention_cost_weight"], 0.0)
        self.assertEqual(profit["reversal_cost_weight"], 0.0)
        self.assertEqual(competitive["long_term_profit_weight"], 0.10)
        self.assertEqual(competitive["profit_dominance_weight"], 0.00)
        self.assertEqual(competitive["market_share_level_weight"], 0.20)
        self.assertEqual(
            competitive["market_share_competitiveness_weight"], 0.70
        )
        self.assertEqual(competitive["market_share_target_gap"], 0.10)
        self.assertEqual(competitive["market_share_gap_scale"], 0.10)
        self.assertEqual(competitive["price_competitiveness_weight"], 0.0)
        self.assertEqual(competitive["intervention_cost_weight"], 0.0)
        self.assertEqual(competitive["reversal_cost_weight"], 0.0)

    def test_preregistered_objective_mixtures_have_exact_convex_weights(self):
        expected = {
            "profit_10_competitiveness_90": (0.17, 0.02, 0.18, 0.63),
            "profit_30_competitiveness_70": (0.31, 0.06, 0.14, 0.49),
            "profit_50_competitiveness_50": (0.45, 0.10, 0.10, 0.35),
            "profit_70_competitiveness_30": (0.59, 0.14, 0.06, 0.21),
            "profit_90_competitiveness_10": (0.73, 0.18, 0.02, 0.07),
        }
        for objective, values in expected.items():
            with self.subTest(objective=objective):
                weights = objective_reward_weights(objective)
                actual = (
                    weights["own_profit"],
                    weights["profit_lead"],
                    weights["own_completed_share"],
                    weights["completed_share_lead"],
                )
                np.testing.assert_allclose(actual, values, atol=1e-12)
                self.assertAlmostEqual(sum(actual), 1.0)

    def test_mixed_reward_is_exact_convex_combination_of_endpoints(self):
        metrics = dict(
            own_profit_per_request=3.5,
            rival_profit_per_request=2.2,
            own_completed_share=0.47,
            rival_completed_share=0.31,
        )
        profit = LongTermProfitReward(
            LongTermProfitRewardConfig(objective_mode="profit_maximization")
        ).compute(**metrics)["reward"]
        competitive = LongTermProfitReward(
            LongTermProfitRewardConfig(objective_mode="competitiveness")
        ).compute(**metrics)["reward"]
        mixed = LongTermProfitReward(
            LongTermProfitRewardConfig(
                objective_mode="profit_30_competitiveness_70"
            )
        )
        self.assertIsInstance(mixed.implementation, MixedEconomicReward)
        self.assertAlmostEqual(
            mixed.compute(**metrics)["reward"],
            0.30 * profit + 0.70 * competitive,
        )

    def test_reward_weights_switch_policy_preference(self):
        profit_model = LongTermProfitReward(
            LongTermProfitRewardConfig(
                own_profit_weight=1.0,
                profit_advantage_weight=0.5,
                price_competitiveness_weight=0.0,
            )
        )
        competitive_model = LongTermProfitReward(
            LongTermProfitRewardConfig(
                objective_mode="competitiveness",
                own_profit_weight=0.0,
                profit_advantage_weight=0.0,
                market_share_competitiveness_weight=1.0,
            )
        )
        dominant = dict(
            own_profit_per_request=5.0,
            rival_profit_per_request=2.0,
            own_completed_share=0.40,
            rival_completed_share=0.50,
        )
        matched = dict(
            own_profit_per_request=3.0,
            rival_profit_per_request=2.0,
            own_completed_share=0.60,
            rival_completed_share=0.30,
        )
        self.assertGreater(
            profit_model.compute(**dominant)["reward_raw"],
            profit_model.compute(**matched)["reward_raw"],
        )
        self.assertGreater(
            competitive_model.compute(**matched)["reward_raw"],
            competitive_model.compute(**dominant)["reward_raw"],
        )

    def test_named_objectives_dispatch_to_distinct_reward_mechanisms(self):
        profit = LongTermProfitReward(LongTermProfitRewardConfig())
        competitive = LongTermProfitReward(LongTermProfitRewardConfig(
            objective_mode="competitiveness",
            own_profit_weight=0.0,
            profit_advantage_weight=0.0,
            market_share_competitiveness_weight=1.0,
        ))
        self.assertIsInstance(profit.implementation, ProfitMaximizationReward)
        self.assertIsInstance(
            competitive.implementation, MarketShareCompetitivenessReward
        )
        with self.assertRaises(ValueError):
            LongTermProfitRewardConfig(objective_mode="balanced")
        with self.assertRaises(ValueError):
            LongTermProfitRewardConfig(objective_mode="custom")

    def test_profit_reward_is_invariant_to_price_gap_at_fixed_economics(self):
        model = LongTermProfitReward(LongTermProfitRewardConfig())
        common = {
            "own_profit_per_request": 4.0,
            "rival_profit_per_request": 2.0,
        }
        matched = model.compute(
            **common, price_gap_f2_minus_f1=0.0, price_gap_abs_error=0.0
        )
        distant = model.compute(
            **common, price_gap_f2_minus_f1=8.0, price_gap_abs_error=8.0
        )
        self.assertAlmostEqual(matched["reward"], distant["reward"])
        self.assertEqual(matched["reward_price_competitiveness_component"], 0.0)

    def test_named_profit_reward_uses_profit_lead_but_ignores_action_costs(self):
        model = LongTermProfitReward(LongTermProfitRewardConfig(
            profit_advantage_weight=100.0,
            intervention_cost_weight=100.0,
            reversal_cost_weight=100.0,
        ))
        common = {
            "own_profit_per_request": 2.0,
            "intervention_magnitude": 1.0,
            "reversal": 1.0,
        }
        healthy_rival = model.compute(
            **common, rival_profit_per_request=4.0
        )
        harmed_rival = model.compute(
            **common, rival_profit_per_request=-40.0
        )

        self.assertGreater(harmed_rival["reward"], healthy_rival["reward"])
        self.assertAlmostEqual(healthy_rival["reward"], 0.40)
        self.assertAlmostEqual(
            healthy_rival["reward_profit_advantage_component"], -0.4
        )
        self.assertEqual(healthy_rival["reward_applied_action_regularization"], 0.0)

    def test_coverage_collapse_activates_service_constraint_without_reward_shaping(self):
        observer = PlatformObservationModel(9)
        controller = SoftConstraintController(ConstraintConfig(
            fulfillment_floor=0.78,
            completion_coverage_floor=0.55,
        ))
        costs = controller.compute(
            observer=observer,
            fulfillment_rate=0.95,
            completion_coverage=0.05,
            wait_minutes=2.0,
            profit_margin=0.25,
            oscillation_cost=0.0,
        )

        self.assertGreater(costs["coverage"], 0.9)
        self.assertEqual(costs["fulfillment"], 0.0)
        self.assertGreater(
            controller.diagnostics(costs)["constraint_cost_coverage"], 0.9
        )

    def test_competitiveness_reward_keeps_profit_credit_at_fixed_share_gap(self):
        model = LongTermProfitReward(LongTermProfitRewardConfig(
            objective_mode="competitiveness",
            own_profit_weight=0.0,
            profit_advantage_weight=0.0,
            market_share_competitiveness_weight=1.0,
        ))
        profitable = model.compute(
            own_profit_per_request=8.0,
            rival_profit_per_request=1.0,
            own_completed_share=0.60,
            rival_completed_share=0.40,
        )
        unprofitable = model.compute(
            own_profit_per_request=-8.0,
            rival_profit_per_request=10.0,
            own_completed_share=0.60,
            rival_completed_share=0.40,
        )
        self.assertGreater(profitable["reward"], unprofitable["reward"])
        self.assertGreater(profitable["reward_own_profit_component"], 0.0)
        self.assertEqual(
            profitable["reward_profit_advantage_component"], 0.0
        )

    def test_profit_reward_directly_prefers_higher_own_profit(self):
        model = LongTermProfitReward()
        destructive_win = model.compute(
            own_profit_per_request=0.05,
            rival_profit_per_request=-2.0,
        )
        profitable_parity = model.compute(
            own_profit_per_request=4.0,
            rival_profit_per_request=4.0,
        )

        self.assertGreater(
            profitable_parity["reward"],
            destructive_win["reward"],
        )
        self.assertAlmostEqual(profitable_parity["reward"], 1.6)
        self.assertAlmostEqual(destructive_win["reward"], 0.43)

    def test_observation_exposes_public_opponent_quote_trends(self):
        observer = PlatformObservationModel(
            3,
            ObservationConfig(
                telemetry_delay_steps=0,
                quote_probe_interval_steps=1,
                quote_probe_delay_steps=0,
                quote_noise_dollars=0.0,
                quote_missing_probability=0.0,
            ),
        )
        first_stats = {
            f"distance_bin_{segment}_price_gap_mean": 0.5
            for segment in ("0_2", "2_5", "5_10", "10_plus")
        }
        second_stats = {
            f"distance_bin_{segment}_price_gap_mean": 1.5
            for segment in ("0_2", "2_5", "5_10", "10_plus")
        }
        observer.ingest(
            own_metrics=self._metrics(),
            supply_metrics={},
            crowd_stats=first_stats,
        )
        observer.ingest(
            own_metrics=self._metrics(),
            supply_metrics={},
            crowd_stats=second_stats,
        )
        observation = observer.build_observation(
            hour=8,
            day_of_week=2,
            weather="clear",
            own_coefficients={
                "base_fare": 3.0,
                "per_minute": 0.3,
                "per_mile": 1.8,
                "booking_fee": 2.0,
                "airport_fee": 5.0,
            },
            anchor_coefficients={
                "base_fare": 3.0,
                "per_minute": 0.3,
                "per_mile": 1.8,
                "booking_fee": 2.0,
                "airport_fee": 5.0,
            },
        )

        self.assertEqual(observation.shape, (24,))
        np.testing.assert_allclose(
            observation[[18, 19, 20, 21]],
            np.full(4, 1.5 / observer.config.gap_scale_dollars),
        )

    def test_observation_exposes_each_own_tariff_coefficient(self):
        observer = PlatformObservationModel(11, ObservationConfig())
        anchors = {
            "base_fare": 3.0,
            "per_minute": 0.3,
            "per_mile": 1.8,
            "booking_fee": 2.0,
            "airport_fee": 5.0,
        }
        own = dict(anchors)
        multipliers = [0.8, 1.0, 1.2, 0.8, 1.2]
        for key, multiplier in zip(anchors, multipliers):
            own[key] *= multiplier

        observation = observer.build_observation(
            hour=8,
            day_of_week=2,
            weather="rain",
            own_coefficients=own,
            anchor_coefficients=anchors,
        )

        np.testing.assert_allclose(
            observation[4:9],
            np.asarray(multipliers) - 1.0,
            atol=1e-6,
        )

    def test_profit_reward_is_invariant_to_completed_share_at_fixed_profit(self):
        model = LongTermProfitReward()
        first = model.compute(
            own_profit_per_request=2.5,
            rival_profit_per_request=2.0,
            own_completed_share=0.80,
            rival_completed_share=0.10,
            price_gap_f2_minus_f1=8.0,
        )
        second = model.compute(
            own_profit_per_request=2.5,
            rival_profit_per_request=2.0,
            own_completed_share=0.10,
            rival_completed_share=0.80,
            price_gap_f2_minus_f1=-8.0,
        )
        self.assertAlmostEqual(first["reward"], second["reward"])
        self.assertEqual(
            first["reward_market_share_competitiveness_component"], 0.0
        )
        self.assertEqual(
            second["reward_market_share_competitiveness_component"], 0.0
        )

    def test_reversal_tracker_detects_coefficient_reversal_across_bundles(self):
        tracker = ActionStabilityTracker(ConstraintConfig(reversal_horizon=4))
        tracker.record(
            action_event=True,
            target="base_fare+booking_fee",
            direction=1,
            directions={"base_fare": 1, "booking_fee": 1},
        )
        tracker.record(
            action_event=True,
            target="base_fare+per_mile",
            direction=-1,
            directions={"base_fare": -1, "per_mile": 1},
        )
        self.assertEqual(tracker.last_reversal, 1.0)

    def test_positive_reward_weights_are_configurable_and_normalized(self):
        metrics = {
            "profit_per_request": 2.0,
            "revenue_per_request": 10.0,
            "completed_share": 0.30,
            "fulfillment_rate": 0.90,
            "acceptance_rate": 0.90,
            "wait_minutes": 5.0,
        }
        profit_only = PositiveBusinessReward(PositiveRewardConfig(
            profit_weight=7.0,
            revenue_weight=0.0,
            completed_demand_weight=0.0,
            service_weight=0.0,
        )).compute(metrics)
        self.assertAlmostEqual(profit_only["reward"], profit_only["reward_positive_profit"])
        self.assertEqual(profit_only["reward_weight_profit"], 1.0)
        self.assertEqual(profit_only["reward_weight_revenue"], 0.0)

        balanced = PositiveBusinessReward(PositiveRewardConfig(
            profit_weight=2.0,
            revenue_weight=2.0,
            completed_demand_weight=2.0,
            service_weight=2.0,
        )).compute(metrics)
        self.assertAlmostEqual(
            sum(balanced[key] for key in (
                "reward_weight_profit",
                "reward_weight_revenue",
                "reward_weight_completed_demand",
                "reward_weight_service",
            )),
            1.0,
        )

    def test_positive_reward_rejects_invalid_weight_sets(self):
        with self.assertRaises(ValueError):
            PositiveRewardConfig(
                profit_weight=0.0,
                revenue_weight=0.0,
                completed_demand_weight=0.0,
                service_weight=0.0,
            )
        with self.assertRaises(ValueError):
            PositiveRewardConfig(profit_weight=-0.1)
        with self.assertRaises(ValueError):
            PositiveRewardConfig(revenue_weight=float("nan"))

    def test_staged_curriculum_reserves_most_training_for_dynamic_opponent(self):
        scheduler = TrainingStageScheduler("staged")
        self.assertEqual(scheduler.stage_at(0.0).name, "market")
        self.assertEqual(scheduler.stage_at(0.20).name, "static_opponent")
        self.assertEqual(scheduler.stage_at(0.40).name, "dynamic_opponent")
        self.assertEqual(scheduler.stage_at(0.95).name, "dynamic_opponent")
        self.assertEqual(scheduler.stage_at(0.95).episode_days, 600)
        self.assertFalse(scheduler.stage_at(0.95).freeze_opponent)
        self.assertEqual(scheduler.stage_at(0.95).exploration_rate, 0.0)

    def test_staged_curriculum_accepts_explicit_schedule(self):
        scheduler = TrainingStageScheduler(
            "staged",
            stage_fractions=(0.10, 0.20, 0.70),
            dynamic_episode_days=750,
            tariff_reset_fraction=0.12,
        )
        self.assertEqual(scheduler.stage_at(0.10).name, "static_opponent")
        self.assertEqual(scheduler.stage_at(0.30).name, "dynamic_opponent")
        self.assertEqual(scheduler.stage_at(0.90).episode_days, 750)
        self.assertAlmostEqual(
            scheduler.stage_at(0.90).tariff_reset_fraction, 0.12
        )

    def test_direct_curriculum_remains_available_as_ablation(self):
        scheduler = TrainingStageScheduler("direct")
        self.assertEqual([stage.name for stage in scheduler.stages], ["direct"])
        self.assertEqual(scheduler.stages[0].opponent_regime, "dynamic")

    def test_training_controls_are_constant(self):
        scheduler = TrainingStageScheduler("staged")
        before = scheduler.smooth_controls_at(0.1999)
        boundary = scheduler.smooth_controls_at(0.2000)
        after = scheduler.smooth_controls_at(0.2010)
        self.assertAlmostEqual(
            before["exploration_rate"],
            boundary["exploration_rate"],
            places=3,
        )
        self.assertEqual(before, boundary)
        self.assertEqual(after, boundary)
        self.assertEqual(boundary["learning_rate_scale"], 1.0)

    def test_committed_leader_quotes_are_immediate_public_observations(self):
        observer = PlatformObservationModel(
            21,
            ObservationConfig(
                quote_probe_interval_steps=9,
                quote_probe_delay_steps=4,
                quote_noise_dollars=0.0,
                quote_missing_probability=0.95,
            ),
        )
        gaps = {
            "0_2": 0.25,
            "2_5": 0.50,
            "5_10": -0.75,
            "10_plus": 1.25,
        }

        observer.publish_public_quote_gaps(gaps)

        snapshot = observer.quote_snapshot()
        for segment, expected in gaps.items():
            self.assertEqual(snapshot[segment]["gap"], expected)
            self.assertEqual(snapshot[segment]["age"], 0.0)
            self.assertEqual(snapshot[segment]["available"], 1.0)


if __name__ == "__main__":
    unittest.main()
