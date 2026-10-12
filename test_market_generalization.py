import unittest

import numpy as np

from Core import Core
from GenerateAgent import GenerateAgent
from MarketInteraction import MarketInteraction
from platform_mdp import PlatformObservationModel
from pricing_models import build_absolute_tariff_action_space


class MarketGeneralizationContractTests(unittest.TestCase):
    def test_empirical_trip_context_calibration_reaches_market_generator(self):
        core = Core(
            market_name="Boston",
            seed=2602,
            choice_mode="parametric",
            firm1_mode="static",
            firm2_mode="static",
            total_customers_pool=32,
        )
        calibration = {
            "market": {
                "day_of_week_probs": {str(i): float(i == 2) for i in range(7)},
                "hour_probs": {str(i): float(i == 17) for i in range(24)},
                "distance_mean_miles": 2.2,
                "distance_std_miles": 0.4,
                "duration_mean_minutes": 12.0,
                "duration_std_minutes": 2.0,
            }
        }
        core.apply_calibration(calibration)
        days = {core.market.sample_day_context().day_of_week for _ in range(20)}
        hours = {core.market.sample_timestep_hour().hour for _ in range(20)}
        rides = [core.market.generate_ride() for _ in range(3000)]
        self.assertEqual(days, {2})
        self.assertEqual(hours, {17})
        self.assertAlmostEqual(
            np.mean([ride.distance_miles for ride in rides]), 2.2, delta=0.12
        )
        self.assertAlmostEqual(
            np.mean([ride.duration_minutes for ride in rides]), 12.0, delta=0.35
        )

    def test_market_names_are_canonical_and_never_silently_relabelled(self) -> None:
        self.assertEqual(
            MarketInteraction("NYC", seed=1).current_city,
            "New York City",
        )
        self.assertEqual(
            MarketInteraction("chicago", seed=1).current_city,
            "Chicago",
        )
        with self.assertRaisesRegex(ValueError, "unsupported market"):
            MarketInteraction("Imaginary City", seed=1)

    def test_core_and_population_use_the_same_non_nyc_market(self) -> None:
        for market in ("General", "Seattle", "Chicago"):
            with self.subTest(market=market):
                core = Core(
                    market_name=market,
                    seed=2601,
                    firm1_mode="RL",
                    firm2_mode="static",
                    total_customers_pool=64,
                    training_curriculum="direct",
                    credit_assignment_v25=True,
                )
                self.assertEqual(core.market_name, market)
                self.assertEqual(core.market.current_city, market)
                self.assertEqual(core.agent_gen.city_name, market)
                self.assertEqual(
                    core.firm1.agent.state_dim,
                    PlatformObservationModel.observation_dim
                    * core.firm1.state_frame_stack,
                )

    def test_relative_action_geometry_is_city_invariant(self) -> None:
        keys = [
            "base_fare",
            "per_minute",
            "per_mile",
            "booking_fee",
            "airport_fee",
        ]
        action_steps, multipliers, targets, action_keys = (
            build_absolute_tariff_action_space(keys)
        )
        projected_by_city = {}
        for market in ("General", "Seattle", "New York City", "Chicago"):
            interaction = MarketInteraction(market, seed=2602)
            base = interaction.curr_market
            anchors = {key: float(getattr(base, key)) for key in keys}
            observer = PlatformObservationModel(seed=2603)
            features = observer.build_action_features(
                action_steps=action_steps,
                action_keys=action_keys,
                own_coefficients=anchors,
                anchor_coefficients=anchors,
                coefficient_steps={key: 0.1 for key in keys},
                coefficient_bounds={key: (0.0, 20.0) for key in keys},
                step_scale=1.0,
                target_gap=0.0,
                action_target_keys=targets,
                action_target_multipliers=multipliers,
            )
            self.assertTrue(np.isfinite(features).all())
            projected_by_city[market] = features[:, 14:18]

        reference = projected_by_city["General"]
        for market, projected in projected_by_city.items():
            with self.subTest(market=market):
                self.assertTrue(np.allclose(projected, reference, atol=1e-6))
                self.assertTrue(np.allclose(projected[1], -0.30, atol=1e-6))
                self.assertTrue(np.allclose(projected[7], 0.30, atol=1e-6))

    def test_non_nyc_demographic_generators_are_finite_and_distinct(self) -> None:
        summaries = {}
        for market in ("General", "Seattle", "Chicago"):
            generator = GenerateAgent(
                seed=2604,
                total_customers=2048,
                city_name=market,
            )
            summaries[market] = (
                float(np.mean(generator._ages)),
                np.asarray(generator.income_probs, dtype=float),
            )
            self.assertTrue(np.isfinite(summaries[market][0]))
            self.assertAlmostEqual(
                float(np.sum(summaries[market][1])), 1.0, places=6
            )
        self.assertFalse(
            np.allclose(
                summaries["Seattle"][1], summaries["Chicago"][1]
            )
        )


if __name__ == "__main__":
    unittest.main()
