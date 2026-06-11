#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 08:28:35 2026

@author: Xiaoru Shi
"""

import importlib
import math
import sys
import types
import unittest

from gpt_threshold_utils import (
    build_threshold_profile,
    diagnose_gpt_threshold_usage,
    format_gpt_threshold_usage_summary,
    increment_gpt_threshold_usage,
    new_gpt_threshold_usage_counts,
    summarize_priced_coldstart_rides,
)


class _FakeRng:
    def random(self):
        return 0.5

    def normal(self, *_args, **_kwargs):
        return 0.0


class _FakeRandom:
    @staticmethod
    def default_rng(_seed=None):
        return _FakeRng()


class _FakeNumpy(types.SimpleNamespace):
    def __init__(self):
        super().__init__(random=_FakeRandom())

    @staticmethod
    def clip(value, lo, hi):
        return max(lo, min(hi, value))

    @staticmethod
    def sign(value):
        return 1.0 if value > 0 else (-1.0 if value < 0 else 0.0)

    @staticmethod
    def exp(value):
        return math.exp(value)


class PriceThresholdFlowTests(unittest.TestCase):
    def test_generated_threshold_is_attached_to_profile_for_simulation(self):
        profile = {"IncomeBracket": "100k-200k", "LoyaltyFirm": "Firm1"}
        coldstart_rides = [{"firm1_price": 12.0, "firm2_price": 14.5}]
        threshold = {"price_threshold": 3.25, "rationale": "unit-test", "source": "gpt"}

        enriched = build_threshold_profile(profile, coldstart_rides, threshold)

        self.assertEqual(enriched["IncomeBracket"], "100k-200k")
        self.assertEqual(enriched["PriceThreshold"], 3.25)
        self.assertEqual(enriched["PriceThresholdRationale"], "unit-test")
        self.assertEqual(enriched["PriceThresholdSource"], "gpt")
        self.assertEqual(enriched["ColdstartRides"], coldstart_rides)
        
    def test_coldstart_rides_are_summarized_for_one_profile_threshold(self):
        priced_rides = [
            {
                "Hour": 8,
                "Weather": "rain",
                "DistanceMiles": 2.0,
                "DurationMinutes": 12.0,
                "Service": "economy",
                "Airport": False,
                "firm1_price": 10.0,
                "firm2_price": 12.0,
            },
            {
                "Hour": 14,
                "Weather": "clear",
                "DistanceMiles": 6.0,
                "DurationMinutes": 24.0,
                "Service": "premium",
                "Airport": True,
                "firm1_price": 20.0,
                "firm2_price": 17.0,
            },
        ]

        summary = summarize_priced_coldstart_rides(priced_rides)

        self.assertEqual(summary["ride_count"], 2)
        self.assertEqual(summary["mean_firm1_price"], 15.0)
        self.assertEqual(summary["mean_firm2_price"], 14.5)
        self.assertEqual(summary["mean_absolute_price_gap"], 2.5)
        self.assertEqual(summary["max_absolute_price_gap"], 3.0)
        self.assertEqual(summary["airport_ride_count"], 1)
        self.assertEqual(summary["rush_hour_ride_count"], 1)
        self.assertEqual(summary["service_mix"], {"economy": 1, "premium": 1})
        self.assertEqual(summary["weather_mix"], {"clear": 1, "rain": 1})

    def test_usage_summary_explains_low_gpt_utilization(self):
        counts = new_gpt_threshold_usage_counts()
        increment_gpt_threshold_usage(counts, "batches_total", 25)
        increment_gpt_threshold_usage(counts, "batches_attempted", 25)
        increment_gpt_threshold_usage(counts, "batches_succeeded", 2)
        increment_gpt_threshold_usage(counts, "batches_failed", 23)
        increment_gpt_threshold_usage(counts, "profiles_requested", 500)
        increment_gpt_threshold_usage(counts, "profiles_gpt", 40)
        increment_gpt_threshold_usage(counts, "profiles_fallback", 460)

        summary = format_gpt_threshold_usage_summary(counts)
        notes = diagnose_gpt_threshold_usage(counts, {"batch_failed": 23}, max_retries=0)

        self.assertIn("batch_success_rate=8.0%", summary)
        self.assertIn("profile_gpt_rate=8.0%", summary)
        self.assertIn("profile_fallback_rate=92.0%", summary)
        self.assertTrue(any("most GPT threshold batches failed" in note for note in notes))
        self.assertTrue(any("max_retries=0" in note for note in notes))
        self.assertTrue(any("batch_failed" in note for note in notes))

    def test_cognitive_choice_model_consumes_profile_price_threshold(self):
        previous_numpy = sys.modules.get("numpy")
        previous_choice_models = sys.modules.get("choice_models")
        sys.modules["numpy"] = _FakeNumpy()
        try:
            sys.modules.pop("choice_models", None)
            choice_models = importlib.import_module("choice_models")
            model = choice_models.CognitiveChoiceModel(seed=11)
            scenario = {
                "Hour": 12,
                "Weather": "clear",
                "Airport": False,
                "Service": "economy",
                "DistanceMiles": 4.0,
                "DayOfWeek": 2,
            }
            base_profile = {
                "Age": 35,
                "IncomeBracket": "50k-100k",
                "HouseholdSize": 1,
                "EmploymentStatus": "Employed",
                "LoyaltyFirm": None,
                "LoyaltyStrength": 0.0,
            }

            low_threshold = model.choose({**base_profile, "PriceThreshold": 0.50}, scenario, 10.0, 12.0)
            high_threshold = model.choose({**base_profile, "PriceThreshold": 5.00}, scenario, 10.0, 12.0)
        finally:
            sys.modules.pop("choice_models", None)
            if previous_choice_models is not None:
                sys.modules["choice_models"] = previous_choice_models
            if previous_numpy is None:
                sys.modules.pop("numpy", None)
            else:
                sys.modules["numpy"] = previous_numpy

        self.assertIn("COG_PRICE", low_threshold.reason_codes)
        self.assertNotIn("COG_PRICE", high_threshold.reason_codes)

    def test_cognitive_choice_model_allows_no_ride_outside_option(self):
        previous_numpy = sys.modules.get("numpy")
        previous_choice_models = sys.modules.get("choice_models")
        sys.modules["numpy"] = _FakeNumpy()
        try:
            sys.modules.pop("choice_models", None)
            choice_models = importlib.import_module("choice_models")
            model = choice_models.CognitiveChoiceModel(seed=11)
            scenario = {
                "Hour": 12,
                "Weather": "clear",
                "Airport": False,
                "Service": "economy",
                "DistanceMiles": 4.0,
                "DurationMinutes": 16.0,
                "DayOfWeek": 2,
            }
            profile = {
                "Age": 35,
                "IncomeBracket": "50k-100k",
                "HouseholdSize": 1,
                "EmploymentStatus": "Employed",
                "LoyaltyFirm": None,
                "LoyaltyStrength": 0.0,
                "PriceThreshold": 1.50,
                "ReservationPrice": 15.00,
                "OutsideOptionCost": 3.00,
                "OutsideOptionInconvenience": 0.10,
            }

            result = model.choose(profile, scenario, 48.0, 50.0)
        finally:
            sys.modules.pop("choice_models", None)
            if previous_choice_models is not None:
                sys.modules["choice_models"] = previous_choice_models
            if previous_numpy is None:
                sys.modules.pop("numpy", None)
            else:
                sys.modules["numpy"] = previous_numpy

        self.assertEqual(result.choice, "NoRide")
        self.assertIn("COG_OUTSIDE_OPTION", result.reason_codes)
        self.assertIn("COG_TOO_EXPENSIVE", result.reason_codes)


if __name__ == "__main__":
    unittest.main()