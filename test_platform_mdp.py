import unittest

import numpy as np

from platform_mdp import (
    ConstraintConfig,
    ObservationConfig,
    PlatformObservationModel,
    PositiveBusinessReward,
    PositiveRewardConfig,
    SoftConstraintController,
    TrainingStageScheduler,
)


class PlatformMDPTests(unittest.TestCase):
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
        self.assertEqual(len(controller.vector(costs)), 10)

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
        self.assertTrue(scheduler.stage_at(0.05).freeze_opponent)
        self.assertFalse(scheduler.stage_at(0.95).freeze_opponent)


if __name__ == "__main__":
    unittest.main()
