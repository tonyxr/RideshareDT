"""Focused contracts for v28's deliberately unbounded economic inputs."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from Core import Core
from MarketInteraction import MarketInteraction
from choice_models import CognitiveChoiceModel
from driver_supply import DriverPayPolicy, DriverSupplyLayer
from gpt_threshold_utils import build_threshold_profile, clip_price_threshold
from market_dynamics import MarketDynamicsConfig
from market_engine import driver_pay_vector, profile_features
from pricing_models import FirmRLPricer


class V28ThresholdTests(unittest.TestCase):
    def test_legacy_threshold_contract_is_unchanged(self):
        self.assertEqual(clip_price_threshold(0.1), 0.5)
        self.assertEqual(clip_price_threshold(25.0), 5.0)

    def test_unbounded_threshold_accepts_every_finite_positive_value(self):
        for value in (0.01, 0.1, 25.0, 1.0e100):
            with self.subTest(value=value):
                self.assertEqual(
                    clip_price_threshold(
                        value, experimental_unbounded_economics=True
                    ),
                    value,
                )
        for value in (0.0, -0.1, float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    clip_price_threshold(
                        value, experimental_unbounded_economics=True
                    )

    def test_generated_profile_retains_value_above_old_gpt_cap(self):
        enriched = build_threshold_profile(
            {"ProfileID": 7},
            [],
            {"price_threshold": 37.25, "source": "llm_numeric"},
            experimental_unbounded_economics=True,
        )
        self.assertEqual(enriched["PriceThreshold"], 37.25)
        self.assertTrue(enriched["ExperimentalUnboundedEconomics"])
        self.assertEqual(
            enriched["ThresholdEconomicContract"],
            "v28_unbounded_finite_positive",
        )

    def test_v28_defaults_to_numeric_generation_but_allows_labelled_rubric_ablation(self):
        core = Core(
            market_name="New York City",
            seed=23,
            choice_mode="cognitive",
            firm1_mode="static",
            firm2_mode="static",
            total_customers_pool=2,
            experimental_unbounded_economics=True,
        )
        self.assertEqual(core.gpt_threshold_mode, "llm_numeric")
        core.active_calibration = {}
        settings = core.configure_v26(
            market_dynamics=MarketDynamicsConfig(enabled=False),
            gpt_threshold_mode=None,
        )
        self.assertEqual(settings["gpt_threshold_mode"], "llm_numeric")
        self.assertEqual(
            settings["threshold_generation_contract"],
            "v28_unbounded_numeric",
        )

        ablation = core.configure_v26(
            market_dynamics=MarketDynamicsConfig(enabled=False),
            gpt_threshold_mode="rubric_context",
        )
        self.assertEqual(ablation["gpt_threshold_mode"], "rubric_context")
        self.assertEqual(
            ablation["threshold_generation_contract"],
            "v28_explicit_legacy_rubric_ablation",
        )

    def test_scalar_and_vectorized_consumers_preserve_out_of_range_values(self):
        model = CognitiveChoiceModel(seed=11)
        model.experimental_unbounded_economics = True
        profiles = [
            {
                "PriceThreshold": 0.10,
                "ExperimentalUnboundedEconomics": True,
            },
            {
                "PriceThreshold": 25.0,
                "ExperimentalUnboundedEconomics": True,
            },
        ]
        scalar = np.asarray(
            [model._profile_price_threshold(profile) for profile in profiles]
        )
        vectorized = profile_features(profiles)["threshold"]
        np.testing.assert_allclose(scalar, [0.10, 25.0])
        np.testing.assert_allclose(vectorized, scalar)

        legacy_model = CognitiveChoiceModel(seed=11)
        legacy_profiles = [
            {"PriceThreshold": 0.10},
            {"PriceThreshold": 25.0},
        ]
        legacy_scalar = np.asarray(
            [legacy_model._profile_price_threshold(profile) for profile in legacy_profiles]
        )
        legacy_vectorized = profile_features(legacy_profiles)["threshold"]
        np.testing.assert_allclose(legacy_scalar, [0.25, 8.0])
        np.testing.assert_allclose(legacy_vectorized, legacy_scalar)

    def test_explicit_nonpositive_profile_threshold_is_not_silently_defaulted(self):
        model = CognitiveChoiceModel(seed=11)
        model.experimental_unbounded_economics = True
        for value in (0.0, -0.25):
            profile = {
                "PriceThreshold": value,
                "ExperimentalUnboundedEconomics": True,
            }
            with self.subTest(value=value, consumer="scalar"):
                with self.assertRaises(ValueError):
                    model._profile_price_threshold(profile)
            with self.subTest(value=value, consumer="vectorized"):
                with self.assertRaises(ValueError):
                    profile_features([profile])

        # Historical false-y-as-missing semantics stay unchanged outside v28.
        self.assertEqual(
            CognitiveChoiceModel(seed=11)._profile_price_threshold(
                {"PriceThreshold": 0.0}
            ),
            1.50,
        )

    def test_v28_cache_recovers_previously_clipped_raw_llm_value(self):
        row = {
            "ProfileID": 1,
            "ProfileMarket": "New York City",
            "PriceThreshold": 5.0,
            "ThresholdRawLLMValue": 19.75,
            "PriceThresholdSource": "llm_numeric",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "thresholds.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            core = Core.__new__(Core)
            core.threshold_cache_path = str(path)
            core.threshold_profile_source = "cached"
            core.market_name = "New York City"
            core.experimental_unbounded_economics = True
            self.assertTrue(core._try_load_threshold_cache(1))

        loaded = core.synthetic_profile_pool[0]
        self.assertEqual(loaded["PriceThreshold"], 19.75)
        self.assertTrue(loaded["ThresholdRecoveredFromRawLLMValue"])
        self.assertTrue(loaded["ExperimentalUnboundedEconomics"])
        self.assertEqual(
            loaded["ThresholdCacheOriginEconomicContract"],
            "legacy_numeric_raw_value_recovered",
        )
        self.assertEqual(
            loaded["ThresholdConsumptionEconomicContract"],
            "v28_unbounded_finite_positive",
        )

    def test_numeric_gpt_path_has_no_range_prompt_or_post_response_cap(self):
        core = Core(
            market_name="New York City",
            seed=23,
            choice_mode="cognitive",
            openai_api_key="unit-test-key",
            firm1_mode="static",
            firm2_mode="static",
            total_customers_pool=2,
            experimental_unbounded_economics=True,
        )
        ride = {
            "Hour": 8,
            "Weather": "clear",
            "DistanceMiles": 4.0,
            "DurationMinutes": 18.0,
            "Service": "economy",
            "Airport": False,
            "DayOfWeek": 2,
        }
        response_body = json.dumps({
            "output_text": json.dumps({
                "thresholds": [{"profile_index": 0, "price_threshold": 17.75}]
            })
        }).encode("utf-8")
        captured_payload = {}

        class _Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return response_body

        def fake_urlopen(request, timeout):
            del timeout
            captured_payload.update(json.loads(request.data.decode("utf-8")))
            return _Response()

        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            result = core._gpt_batch_numeric_thresholds(
                [({"IncomeBracket": "50k-100k"}, [ride])]
            )[0]

        self.assertEqual(result["price_threshold"], 17.75)
        self.assertFalse(result["llm_threshold_clipped"])
        prompt = json.loads(captured_payload["input"])
        constraints = prompt["output_constraints"]
        self.assertEqual(constraints["economic_bounds"], "none")
        self.assertNotIn("price_threshold_range_usd", constraints)
        self.assertEqual(
            prompt["evidence_protocol"]["prompt_protocol"],
            "numeric_profile_threshold_v12_uncapped",
        )


class V28TariffActionTests(unittest.TestCase):
    def test_v28_actions_compound_past_legacy_coefficient_bounds(self):
        market = MarketInteraction(city_name="New York City", seed=17)
        pricer = FirmRLPricer(
            seed=19,
            opt_keys=["base_fare"],
            experimental_unbounded_economics=True,
        )
        anchor = float(market.curr_market.base_fare)
        for _ in range(4):
            pricer.apply_action(7, market)  # current tariff x 1.30

        expected = anchor * (1.30 ** 4)
        self.assertAlmostEqual(pricer.overrides.base_fare, expected)
        self.assertGreater(
            pricer.overrides.base_fare,
            pricer.config.bounds["base_fare"][1],
        )
        self.assertFalse(pricer.last_action_was_saturated)
        self.assertEqual(pricer.action_label(7), "tariff_scale:1.300x_current")

    def test_legacy_action_still_posts_one_bounded_anchor_target(self):
        market = MarketInteraction(city_name="New York City", seed=17)
        pricer = FirmRLPricer(seed=19, opt_keys=["base_fare"])
        pricer.apply_action(7, market)
        first = float(pricer.overrides.base_fare)
        pricer.apply_action(7, market)
        self.assertEqual(pricer.overrides.base_fare, first)
        self.assertLessEqual(first, pricer.config.bounds["base_fare"][1])
        self.assertEqual(pricer.action_label(7), "tariff_target:1.300x_market")

    def test_numerical_overflow_is_rejected_atomically_and_reported(self):
        market = MarketInteraction(city_name="New York City", seed=17)
        pricer = FirmRLPricer(
            seed=19,
            opt_keys=["base_fare", "per_minute"],
            experimental_unbounded_economics=True,
        )
        enormous = pricer.NUMERICAL_COEFFICIENT_LIMIT / 1.1
        pricer.overrides.base_fare = enormous
        pricer.overrides.per_minute = 0.25
        pricer.apply_action(7, market)

        self.assertEqual(pricer.overrides.base_fare, enormous)
        self.assertEqual(pricer.overrides.per_minute, 0.25)
        self.assertEqual(pricer.numerical_action_rejection_count, 1)
        self.assertIn("float32_headroom_exceeded", pricer.last_action_numerical_rejection)
        self.assertTrue(pricer.last_action_descriptor.numerical_rejection)
        self.assertTrue(math.isfinite(pricer.overrides.base_fare))

    def test_v28_static_benchmark_uses_same_uncapped_tariff_domain(self):
        core = Core.__new__(Core)
        core.market = MarketInteraction(city_name="New York City", seed=17)
        core.firm1 = FirmRLPricer(
            seed=19,
            opt_keys=["base_fare"],
            experimental_unbounded_economics=True,
        )
        core.experimental_unbounded_economics = True
        core._baseline_firm1_action(
            "static:4.0",
            step_index=0,
            action_mask=np.ones(len(core.firm1.action_to_steps), dtype=bool),
            day_of_week=2,
            hour=12,
            weather="clear",
            presampled=None,
            customers_per_step=1,
        )
        expected = 4.0 * core.market.curr_market.base_fare
        self.assertEqual(core.firm1.overrides.base_fare, expected)
        self.assertGreater(expected, core.firm1.config.bounds["base_fare"][1])

    def test_v28_policy_artifact_round_trip_keeps_action_semantics(self):
        kwargs = {
            "market_name": "New York City",
            "seed": 41,
            "firm1_mode": "RL",
            "firm2_mode": "static",
            "total_customers_pool": 2,
            "experimental_unbounded_economics": True,
        }
        source = Core(**kwargs)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "v28-policy.pt")
            source.save_trained_model(path=path)
            target = Core(**kwargs)
            artifact = target.load_trained_model(path)

        self.assertTrue(artifact["experimental_unbounded_economics"])
        self.assertEqual(
            artifact["action_semantics"],
            "unbounded_compounding_whole_tariff_scale",
        )
        population = artifact["case_study_population"]
        self.assertEqual(population["gpt_threshold_mode"], "llm_numeric")
        self.assertEqual(
            population["threshold_generation_contract"],
            "v28_unbounded_numeric",
        )


class V28DriverIncentiveTests(unittest.TestCase):
    @staticmethod
    def _core_stub(*, unbounded: bool, multiplier: float) -> Core:
        core = Core.__new__(Core)
        core.enable_driver_supply = True
        core.experimental_unbounded_economics = unbounded
        core.firm1 = SimpleNamespace(supply_incentive_multiplier=multiplier)
        core.firm2 = SimpleNamespace(supply_incentive_multiplier=1.0)
        core.driver_supply = DriverSupplyLayer(seed=31)
        return core

    def test_v28_incentive_above_legacy_cap_survives_sync_and_pay(self):
        core = self._core_stub(unbounded=True, multiplier=2.40)
        core._sync_driver_incentive_multipliers()
        policy = core.driver_supply.pay_policies["Firm1"]

        self.assertTrue(policy.experimental_unbounded_economics)
        self.assertEqual(policy.incentive_multiplier, 2.40)
        vectorized = driver_pay_vector(policy, np.asarray([10.0]))
        scalar = core.driver_supply.quote_driver_pay(
            "Firm1",
            rider_fare=10.0,
            distance_miles=3.0,
            duration_minutes=12.0,
            pickup_minutes=4.0,
            airport=False,
        )
        np.testing.assert_allclose(vectorized, [17.28])
        self.assertAlmostEqual(scalar, 17.28)

    def test_legacy_incentive_cap_is_unchanged(self):
        core = self._core_stub(unbounded=False, multiplier=2.40)
        core._sync_driver_incentive_multipliers()
        policy = core.driver_supply.pay_policies["Firm1"]

        self.assertFalse(policy.experimental_unbounded_economics)
        self.assertEqual(policy.incentive_multiplier, 1.25)
        np.testing.assert_allclose(
            driver_pay_vector(policy, np.asarray([10.0])),
            [9.0],
        )

    def test_v28_driver_pay_overflow_is_reported_not_clipped(self):
        policy = DriverPayPolicy(
            fare_share=1.0,
            minimum_pay=1.0,
            incentive_multiplier=np.finfo(np.float64).max,
            experimental_unbounded_economics=True,
        )
        with self.assertRaisesRegex(OverflowError, "driver pay overflowed"):
            driver_pay_vector(policy, np.asarray([2.0]))


if __name__ == "__main__":
    unittest.main()
