"""Focused contracts for trip-bank-only v28 shock calibration readiness."""

from __future__ import annotations

import csv
import datetime as dt
import tempfile
import unittest
from pathlib import Path

from shock_calibration import (
    _holiday_windows,
    analyze_trip_bank,
    build_artifact,
    render_markdown,
)


FIELDS = (
    "city",
    "ride_semantics",
    "pickup_datetime",
    "hour",
    "day_of_week",
    "duration_minutes",
    "airport",
    "pickup_location",
    "dropoff_location",
    "weather",
    "weather_source",
)


def _write_rows(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _controlled_nyc_bank(path: Path) -> None:
    start = dt.date(2022, 1, 1)
    end = dt.date(2023, 12, 31)
    holiday_window = set(_holiday_windows((2021, 2022, 2023)))
    snow_days = {
        dt.date(year, 1, day)
        for year in (2022, 2023)
        for day in (7, 12, 17, 22, 27)
    }
    rows = []
    date = start
    while date <= end:
        if date in snow_days:
            count, airport_count, duration, weather = 5, 2, 20.0, "snow"
        elif date in holiday_window:
            count, airport_count, duration, weather = 20, 8, 10.0, "clear"
        else:
            count, airport_count, duration, weather = 10, 1, 10.0, "clear"
        for index in range(count):
            airport = index < airport_count
            rows.append({
                "city": "New York City",
                "ride_semantics": "realized_trip",
                "pickup_datetime": f"{date.isoformat()}T12:00:00-05:00",
                "hour": 12,
                "day_of_week": date.weekday(),
                "duration_minutes": duration,
                "airport": str(airport),
                "pickup_location": "132" if airport else "100",
                "dropoff_location": "100",
                "weather": weather,
                "weather_source": "noaa_global_hourly",
            })
        date += dt.timedelta(days=1)
    _write_rows(path, rows)


class TripBankShockCalibrationTests(unittest.TestCase):
    def test_matched_ratios_and_safe_field_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trip_bank.csv"
            _controlled_nyc_bank(path)
            report = analyze_trip_bank(path, bootstrap_replicates=100, seed=9)

        snow = report["weather"]["snow"]["metrics"]
        self.assertAlmostEqual(snow["request_volume_ratio"]["estimate"], 0.5)
        self.assertAlmostEqual(snow["airport_share_ratio"]["estimate"], 4.0)
        self.assertAlmostEqual(snow["duration_ratio"]["estimate"], 2.0)
        self.assertTrue(snow["duration_ratio"]["ready_for_training_override"])
        self.assertGreaterEqual(
            snow["duration_ratio"]["effective_n_calendar_days"], 9.0
        )

        holiday = report["holiday"]["groups"]["non_anchor_wave"]
        self.assertAlmostEqual(holiday["request_volume_ratio"]["estimate"], 2.0)
        self.assertAlmostEqual(holiday["airport_share_ratio"]["estimate"], 4.0)

        candidate = report["training_candidate"]["shock_calibration"]
        self.assertEqual(candidate["storm_duration_multiplier"], 2.0)
        self.assertEqual(candidate["storm_airport_share"], 4.0)
        self.assertEqual(candidate["holiday_airport_share_peak"], 4.0)
        self.assertEqual(candidate["storm_duration_days"], [1, 1])
        self.assertIn("storm_daily_hazard", candidate)
        provenance = report["field_decisions"]["storm_duration_multiplier"][
            "provenance"
        ]
        self.assertEqual(provenance["kind"], "empirical_trip_bank_estimate")
        self.assertEqual(provenance["sha256"], report["source"]["sha256"])
        self.assertIn("duration_minutes", provenance["source_fields"])

        # Observed total volume does not identify purpose-specific demand.
        self.assertNotIn("storm_discretionary_volume", candidate)
        self.assertNotIn("holiday_commute_volume", candidate)
        self.assertEqual(
            report["field_decisions"]["storm_discretionary_volume"]["status"],
            "retained_prior",
        )

    def test_quote_bank_never_fabricates_boston_airport_or_duration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trip_bank.csv"
            rows = []
            start = dt.date(2018, 11, 26)
            for offset in range(12):
                date = start + dt.timedelta(days=offset)
                for index in range(8):
                    rows.append({
                        "city": "Boston",
                        "ride_semantics": "price_quote",
                        "pickup_datetime": f"{date.isoformat()}T12:00:00-05:00",
                        "hour": 12,
                        "day_of_week": date.weekday(),
                        "duration_minutes": "",
                        "airport": "False",
                        "pickup_location": "Back Bay",
                        "dropoff_location": "North End",
                        "weather": "rain" if offset in {2, 6, 10} else "clear",
                        "weather_source": "noaa_global_hourly",
                    })
            _write_rows(path, rows)
            report = analyze_trip_bank(path, bootstrap_replicates=50, seed=2)

        self.assertFalse(report["identifiability"]["realized_request_volume"])
        self.assertFalse(report["identifiability"]["airport"])
        self.assertFalse(report["identifiability"]["airport_direction"])
        self.assertFalse(report["identifiability"]["duration"])
        candidate = report["training_candidate"]["shock_calibration"]
        self.assertNotIn("holiday_airport_share_peak", candidate)
        self.assertNotIn("storm_airport_share", candidate)
        self.assertNotIn("storm_duration_multiplier", candidate)
        self.assertIn(
            "must not be inferred",
            report["field_decisions"]["storm_duration_multiplier"]["reason"],
        )

    def test_non_identifiable_supply_elasticity_and_rival_fields_remain_priors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trip_bank.csv"
            _controlled_nyc_bank(path)
            report = analyze_trip_bank(path, bootstrap_replicates=20, seed=3)

        candidate = set(report["training_candidate"]["shock_calibration"])
        forbidden = {
            "shortage_rate_per_day",
            "shortage_fleet",
            "shortage_wage_multiplier",
            "storm_price_sensitivity",
            "storm_reservation_multiplier",
            "holiday_airport_price_sensitivity",
            "rival_promotion_rate_per_day",
            "rival_promotion_multiplier",
        }
        self.assertFalse(candidate & forbidden)
        for field_name in forbidden:
            with self.subTest(field=field_name):
                decision = report["field_decisions"][field_name]
                self.assertEqual(decision["status"], "retained_prior")
                self.assertEqual(
                    decision["provenance"]["source"],
                    "market_shocks.ShockConfig",
                )
                self.assertTrue(decision["reason"])

    def test_artifact_and_report_state_non_mutating_review_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trip_bank.csv"
            _controlled_nyc_bank(path)
            artifact = build_artifact([path], bootstrap_replicates=20, seed=5)
            markdown = render_markdown(artifact)

        self.assertFalse(artifact["automatic_calibration_mutation"])
        city = artifact["cities"]["New York City"]
        self.assertEqual(
            city["training_candidate"]["status"],
            "review_required_not_applied",
        )
        self.assertIn("Effective N days", markdown)
        self.assertIn("What cannot feed training", markdown)
        self.assertIn("No Boston airport or duration parameter is estimated", markdown)


if __name__ == "__main__":
    unittest.main()
