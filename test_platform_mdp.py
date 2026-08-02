import unittest

import numpy as np

from platform_mdp import (
    ActionStabilityTracker,
    ConstraintConfig,
    BalancedPolicyReward,
    CustomPolicyReward,
    LongTermProfitReward,
    LongTermProfitRewardConfig,
    ObservationConfig,
    PlatformObservationModel,
    MarketShareCompetitivenessReward,
    ProfitMaximizationReward,
    PositiveBusinessReward,
    PositiveRewardConfig,
    SoftConstraintController,
    OperationalClock,
    TrainingStageScheduler,
    conditional_competitive_shares,
    policy_objective_defaults,
)


class PlatformMDPTests(unittest.TestCase):
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
        self.assertEqual(len(controller.vector(costs)), 3)
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
        expected = (
            np.arcsinh(1.0)
            + config.profit_advantage_weight
            * (1.0 - np.exp(-1.0))
            * np.tanh(2.0 / 2.5)
        )
        self.assertAlmostEqual(result["reward_raw"], expected)
        self.assertAlmostEqual(result["reward"], expected)
        self.assertAlmostEqual(
            result["reward_base"],
            result["reward_own_profit_component"]
            + result["reward_profit_advantage_component"]
            + result["reward_price_competitiveness_component"],
        )
        self.assertGreater(result["reward_profit_advantage_component"], 0.0)
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
            market_share_gap_scale=0.05,
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
        self.assertAlmostEqual(matched["reward"], 0.55)
        self.assertGreater(matched["reward"], distant["reward"])
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

        self.assertGreater(stronger["reward"], certified["reward"])
        self.assertAlmostEqual(stronger["reward"], 0.66 * 0.90)

    def test_competitiveness_reward_rejects_unserviceable_share_gaming(self):
        model = LongTermProfitReward(LongTermProfitRewardConfig(
            objective_mode="competitiveness",
            own_profit_weight=0.0,
            profit_advantage_weight=0.0,
            market_share_competitiveness_weight=1.0,
        ))
        extreme_but_unserved = model.compute(
            own_profit_per_request=1.0,
            rival_profit_per_request=0.0,
            own_market_share=0.97,
            rival_market_share=0.03,
            own_fulfillment_rate=0.42,
        )
        durable_lead = model.compute(
            own_profit_per_request=1.0,
            rival_profit_per_request=0.5,
            own_market_share=0.60,
            rival_market_share=0.40,
            own_fulfillment_rate=0.80,
        )

        self.assertGreater(durable_lead["reward"], extreme_but_unserved["reward"])

    def test_competitiveness_reward_prefers_serviceable_lead_near_floor(self):
        model = LongTermProfitReward(LongTermProfitRewardConfig(
            objective_mode="competitiveness",
            own_profit_weight=0.0,
            profit_advantage_weight=0.0,
            market_share_competitiveness_weight=1.0,
            market_share_service_floor=0.78,
        ))
        extreme_but_below_floor = model.compute(
            own_profit_per_request=2.0,
            rival_profit_per_request=0.0,
            own_market_share=0.996,
            rival_market_share=0.004,
            own_fulfillment_rate=0.669,
        )
        durable_lead = model.compute(
            own_profit_per_request=1.0,
            rival_profit_per_request=0.5,
            own_market_share=0.75,
            rival_market_share=0.25,
            own_fulfillment_rate=0.80,
        )

        self.assertGreater(durable_lead["reward"], extreme_but_below_floor["reward"])
        self.assertAlmostEqual(
            extreme_but_below_floor["reward_market_share_service_penalty"],
            2.0 * (0.78 - 0.669),
        )

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

    def test_named_objective_profiles_restore_v6_and_competitiveness(self):
        profit = policy_objective_defaults("profit_maximization")
        competitive = policy_objective_defaults("competitiveness")
        self.assertEqual(profit["long_term_profit_weight"], 1.0)
        self.assertEqual(profit["profit_dominance_weight"], 0.10)
        self.assertEqual(profit["price_competitiveness_weight"], 0.0)
        self.assertEqual(competitive["long_term_profit_weight"], 0.0)
        self.assertEqual(competitive["profit_dominance_weight"], 0.0)
        self.assertEqual(
            competitive["market_share_competitiveness_weight"], 1.0
        )
        self.assertEqual(competitive["market_share_target_gap"], 0.10)
        self.assertEqual(competitive["price_competitiveness_weight"], 0.0)

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
        balanced = LongTermProfitReward(LongTermProfitRewardConfig(
            objective_mode="balanced",
            market_share_competitiveness_weight=1.0,
        ))
        custom = LongTermProfitReward(LongTermProfitRewardConfig(
            objective_mode="custom",
            market_share_competitiveness_weight=1.0,
        ))

        self.assertIsInstance(profit.implementation, ProfitMaximizationReward)
        self.assertIsInstance(
            competitive.implementation, MarketShareCompetitivenessReward
        )
        self.assertIsInstance(balanced.implementation, BalancedPolicyReward)
        self.assertIsInstance(custom.implementation, CustomPolicyReward)

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

    def test_competitiveness_reward_is_invariant_to_profit_at_fixed_share_gap(self):
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
        self.assertAlmostEqual(profitable["reward"], unprofitable["reward"])
        self.assertEqual(profitable["reward_own_profit_component"], 0.0)
        self.assertEqual(
            profitable["reward_profit_advantage_component"], 0.0
        )

    def test_profitability_gate_prevents_destructive_dominance(self):
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

        self.assertEqual(observation.shape, (81,))
        np.testing.assert_allclose(
            observation[[51, 55, 59, 63]],
            np.full(4, 1.5 / observer.config.gap_scale_dollars),
        )

    def test_long_term_profit_reward_does_not_depend_on_share_or_revenue(self):
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
        self.assertEqual(first["reward"], second["reward"])
        self.assertEqual(
            first["reward_market_share_competitiveness_component"], 0.0
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

    def test_curriculum_progresses_to_consolidation(self):
        scheduler = TrainingStageScheduler("staged")
        self.assertEqual(scheduler.stage_at(0.05).name, "foundation")
        self.assertEqual(scheduler.stage_at(0.30).name, "robustness")
        self.assertEqual(scheduler.stage_at(0.65).name, "competition")
        self.assertEqual(scheduler.stage_at(0.95).name, "consolidation")
        self.assertGreaterEqual(scheduler.stage_at(0.95).episode_days, 512)
        self.assertLessEqual(scheduler.stage_at(0.95).exploration_rate, 0.01)
        self.assertTrue(scheduler.stage_at(0.05).freeze_opponent)
        self.assertFalse(scheduler.stage_at(0.95).freeze_opponent)

    def test_stage_optimizer_controls_transition_smoothly(self):
        scheduler = TrainingStageScheduler("staged")
        before = scheduler.smooth_controls_at(0.1799)
        boundary = scheduler.smooth_controls_at(0.1800)
        after = scheduler.smooth_controls_at(0.1810)
        self.assertAlmostEqual(
            before["exploration_rate"],
            boundary["exploration_rate"],
            places=3,
        )
        self.assertLess(
            abs(after["exploration_rate"] - boundary["exploration_rate"]),
            0.01,
        )


if __name__ == "__main__":
    unittest.main()
