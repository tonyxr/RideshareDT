"""v28 conditional shock sampling and persistent supply-shock tests."""

from __future__ import annotations

import csv
import datetime as dt
import tempfile
import unittest
from pathlib import Path

import numpy as np

from market_dynamics import TripBank
from market_shocks import ShockConfig, ShockEffect, ShockEvent, ShockProcess, ShockScenario
from shock_sampling import apply_conditional_shock_distribution


def _profiles(n: int):
    incomes = ("<50k", "50k-100k", "100k-200k", "200k+")
    jobs = ("Student", "Employed", "Unemployed", "Retired")
    return [
        {
            "Age": 20 + i % 50,
            "IncomeBracket": incomes[i % len(incomes)],
            "EmploymentStatus": jobs[i % len(jobs)],
            "HouseholdSize": 1 + i % 4,
            "PriceThreshold": 0.25 + 0.05 * i,
        }
        for i in range(n)
    ]


def _trips(n: int, airport_rows: int = 40):
    airport = np.zeros(n, dtype=bool)
    airport[:airport_rows] = True
    origin = np.zeros(n, dtype=bool)
    destination = np.zeros(n, dtype=bool)
    origin[: airport_rows // 2] = True
    destination[airport_rows // 2:airport_rows] = True
    return {
        "distance": np.linspace(1.0, 20.0, n),
        "duration": np.linspace(8.0, 55.0, n),
        "airport": airport,
        "airport_origin": origin,
        "airport_destination": destination,
        "service": np.array(["premium" if i % 7 == 0 else "economy" for i in range(n)], dtype=object),
        "source_weather": np.array(["snow"] * n, dtype=object),
    }


class ConditionalShockSamplingTest(unittest.TestCase):
    def test_joint_distribution_changes_and_is_deterministic(self):
        n = 400
        profiles = _profiles(n)
        trips = _trips(n)
        effect = ShockEffect(phase="peak", active=("holiday_wave",))
        effect.segment_volume_multiplier.update({
            "airport_origin": 6.0,
            "airport_destination": 0.35,
            "commute": 0.55,
        })
        effect.segment_duration_multiplier["airport_origin"] = 1.45
        effect.segment_price_sensitivity_multiplier["airport_origin"] = 0.70
        effect.segment_reservation_multiplier["airport_origin"] = 1.20
        effect.segment_outside_cost_multiplier["airport_origin"] = 1.15

        a = apply_conditional_shock_distribution(
            profiles, trips, effect, np.random.default_rng(123), hour=20, day_of_week=2,
            weather="snow", airport_identified=True, duration_source="observed",
        )
        b = apply_conditional_shock_distribution(
            profiles, trips, effect, np.random.default_rng(123), hour=20, day_of_week=2,
            weather="snow", airport_identified=True, duration_source="observed",
        )
        profiles_a, trips_a, diag_a = a
        profiles_b, trips_b, diag_b = b
        self.assertEqual(profiles_a, profiles_b)
        for key in trips_a:
            np.testing.assert_array_equal(trips_a[key], trips_b[key])
        self.assertEqual(diag_a.as_dict(), diag_b.as_dict())
        self.assertGreater(diag_a.airport_share_after, diag_a.airport_share_before)
        self.assertGreater(diag_a.segment_counts["airport_origin"], diag_a.segment_counts.get("airport_destination", 0))
        self.assertNotAlmostEqual(diag_a.mean_duration_after, diag_a.mean_duration_before, places=3)
        self.assertTrue(np.any(trips_a["price_sensitivity_multiplier"] < 1.0))
        self.assertTrue(all("ShockSegment" in profile for profile in profiles_a))
        self.assertEqual(diag_a.airport_direction_source, "observed_endpoints_with_prior_for_unknown")
        self.assertEqual(diag_a.weather_match_share_after, 1.0)

    def test_neutral_call_consumes_fixed_crn_budget_without_reordering(self):
        n = 30
        profiles = _profiles(n)
        trips = _trips(n, airport_rows=4)
        rng = np.random.default_rng(99)
        before = np.random.default_rng(99)
        sampled_profiles, sampled_trips, diagnostics = apply_conditional_shock_distribution(
            profiles, trips, ShockEffect(), rng, hour=13, day_of_week=1
        )
        before.random(3 * n)
        self.assertEqual(rng.random(), before.random())
        self.assertEqual(sampled_profiles, profiles)
        np.testing.assert_array_equal(sampled_trips["distance"], trips["distance"])
        self.assertFalse(diagnostics.changed)

    def test_uniform_weather_weights_do_not_bootstrap_resample(self):
        n = 100
        profiles = _profiles(n)
        trips = _trips(n, airport_rows=10)
        trips["source_weather"] = np.array(["clear"] * n, dtype=object)
        rng = np.random.default_rng(314)
        reference = np.random.default_rng(314)

        sampled_profiles, sampled_trips, diagnostics = apply_conditional_shock_distribution(
            profiles,
            trips,
            ShockEffect(),
            rng,
            hour=13,
            day_of_week=1,
            weather="clear",
        )

        reference.random(3 * n)
        self.assertEqual(rng.random(), reference.random())
        self.assertFalse(diagnostics.changed)
        self.assertEqual(sampled_profiles, profiles)
        np.testing.assert_array_equal(sampled_trips["distance"], trips["distance"])
        self.assertEqual(len(np.unique(sampled_trips["distance"])), n)
        self.assertEqual(diagnostics.weather_match_share_before, 1.0)
        self.assertEqual(diagnostics.weather_match_share_after, 1.0)

    def test_upstream_weather_conditioning_is_not_applied_twice(self):
        n = 80
        profiles = _profiles(n)
        trips = _trips(n, airport_rows=8)
        trips["source_weather"] = np.array(
            ["snow"] * 20 + ["rain"] * 20 + ["clear"] * 40,
            dtype=object,
        )

        sampled_profiles, sampled_trips, diagnostics = apply_conditional_shock_distribution(
            profiles,
            trips,
            ShockEffect(),
            np.random.default_rng(2718),
            hour=18,
            day_of_week=2,
            weather="snow",
            weather_conditioned_upstream=True,
        )

        self.assertFalse(diagnostics.changed)
        self.assertEqual(sampled_profiles, profiles)
        np.testing.assert_array_equal(sampled_trips["source_weather"], trips["source_weather"])
        self.assertEqual(diagnostics.weather_match_share_after, 0.25)

    def test_partial_pool_airport_fallback_is_explicit(self):
        n = 80
        profiles = _profiles(n)
        trips = _trips(n, airport_rows=0)
        effect = ShockEffect(phase="peak", active=("holiday_wave",))
        effect.segment_volume_multiplier["airport_origin"] = 4.0
        _, sampled, diagnostics = apply_conditional_shock_distribution(
            profiles, trips, effect, np.random.default_rng(4), hour=19, day_of_week=5,
            airport_identified=False, duration_source="assumed_hourly_speed_profile",
        )
        self.assertGreater(diagnostics.synthetic_airport_candidates, 0)
        self.assertGreater(np.mean(sampled["airport"]), 0.0)
        self.assertEqual(diagnostics.airport_direction_source, "partial_pool_synthetic_direction")
        self.assertEqual(diagnostics.duration_source, "assumed_hourly_speed_profile")

    def test_sparse_weather_fallback_resamples_actual_matching_rows(self):
        n = 200
        profiles = _profiles(n)
        trips = _trips(n, airport_rows=20)
        trips["source_weather"] = np.array(
            ["snow"] * 30 + ["rain"] * 30 + ["clear"] * 140, dtype=object
        )
        effect = ShockEffect(phase="peak", active=("winter_storm",))
        _, sampled, diagnostics = apply_conditional_shock_distribution(
            profiles, trips, effect, np.random.default_rng(44), hour=18, day_of_week=2,
            weather="snow", airport_identified=True, duration_source="observed",
        )
        self.assertGreater(
            diagnostics.weather_match_share_after,
            diagnostics.weather_match_share_before + 0.15,
        )
        self.assertAlmostEqual(
            diagnostics.weather_match_share_after,
            float(np.mean(sampled["source_weather"] == "snow")),
        )


class EmpiricalWeatherConditioningTest(unittest.TestCase):
    def test_trip_bank_conditions_on_weather_and_keeps_airport_direction(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bank.csv"
            fields = [
                "distance_miles", "duration_minutes", "hour", "day_of_week", "airport",
                "service", "actual_price", "weather", "pickup_location", "dropoff_location",
            ]
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for weather, duration, airport, pickup, dropoff in (
                    ("clear", 12.0, False, "10", "20"),
                    ("snow", 34.0, True, "138", "20"),
                ):
                    for i in range(80):
                        writer.writerow({
                            "distance_miles": 4.0 + i / 100.0,
                            "duration_minutes": duration,
                            "hour": 18,
                            "day_of_week": 2,
                            "airport": airport,
                            "service": "economy",
                            "actual_price": 20.0,
                            "weather": weather,
                            "pickup_location": pickup,
                            "dropoff_location": dropoff,
                        })
            bank = TripBank.from_csv(path, city="New York City")
            clear = bank.sample(np.random.default_rng(8), hour=18, day_of_week=2, size=200, jitter=0.0, weather="clear")
            snow = bank.sample(np.random.default_rng(8), hour=18, day_of_week=2, size=200, jitter=0.0, weather="snow")
            self.assertTrue(np.all(clear["source_weather"] == "clear"))
            self.assertTrue(np.all(snow["source_weather"] == "snow"))
            self.assertLess(clear["duration"].mean(), snow["duration"].mean())
            self.assertEqual(float(np.mean(clear["airport"])), 0.0)
            self.assertEqual(float(np.mean(snow["airport"])), 1.0)
            self.assertTrue(np.all(snow["airport_origin"]))


class PersistentDriverShockTest(unittest.TestCase):
    def test_shortage_has_recovery_and_snapshot_restore(self):
        config = ShockConfig(
            enabled=True,
            types=("driver_shortage",),
            shortage_rate_per_day=0.0,
            shortage_recovery_days=2,
        )
        process = ShockProcess(config, seed=77, steps_per_day=2)
        process.scenario = ShockScenario(
            start_date=dt.date(2023, 1, 2),
            events=[ShockEvent("driver_shortage", 1, 1, intensity=0.50, announced=False)],
        )
        process.reset()
        process.advance_day()
        snapshot = process.snapshot()
        path_a = []
        for _ in range(4):
            process.advance_day()
            path_a.append(process.effect(0).as_dict())
        process.restore(snapshot)
        path_b = []
        for _ in range(4):
            process.advance_day()
            path_b.append(process.effect(0).as_dict())
        self.assertEqual(path_a, path_b)
        self.assertIn("driver_shortage", path_a[0]["active"])
        recovery = [row for row in path_a if "driver_shortage_recovery" in row["active"]]
        self.assertTrue(recovery)
        self.assertTrue(all(0.5 < row["fleet_multiplier"] < 1.0 for row in recovery))
        self.assertEqual(path_a[-1]["fleet_multiplier"], 1.0)

    def test_calibration_provenance_never_silently_claims_calibrated(self):
        missing = ShockConfig.from_calibration({})
        present = ShockConfig.from_calibration({"shock_calibration": {"storm_fleet": 0.72}})
        self.assertEqual(missing.calibration_source, "documented_prior_missing_shock_calibration")
        self.assertEqual(present.calibration_source, "city_shock_calibration")
        self.assertEqual(present.storm_fleet, 0.72)


if __name__ == "__main__":
    unittest.main()
