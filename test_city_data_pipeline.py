from __future__ import annotations

import csv
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from calibration_utils import load_calibration_json
from city_data_pipeline import (
    DataValidationError,
    NOAAWeatherIndex,
    WeatherObservation,
    default_source_paths,
    normalize_ride_record,
    puma_codes_for_city,
    resolve_city_spec,
    validate_city_bundle,
)


class CityRideAdapterTests(unittest.TestCase):
    def test_catalog_resolves_all_required_aliases(self):
        self.assertEqual(resolve_city_spec("NYC").market_name, "New York City")
        self.assertEqual(resolve_city_spec("Boston MA").market_name, "Boston")
        self.assertEqual(resolve_city_spec("Chicago Illinois").market_name, "Chicago")

    def test_nyc_realized_trip_units_and_airport_zone(self):
        record = normalize_ride_record(
            "nyc",
            {
                "business": "Uber",
                "pickup_location": 132,
                "dropoff_location": 161,
                "trip_length": 8.0,
                "total_ride_time": 1800,
                "date": "2022-01-01",
                "hour_of_day": 23,
                "passenger_fare": 42.0,
            },
        )
        self.assertEqual(record.ride_semantics, "realized_trip")
        self.assertEqual(record.price_semantics, "realized_passenger_fare")
        self.assertEqual(record.duration_minutes, 30.0)
        self.assertEqual(record.hour, 23)
        self.assertTrue(record.airport)

    def test_boston_is_quote_not_completed_trip(self):
        record = normalize_ride_record(
            "boston",
            {
                "id": "quote-1",
                "datetime": "2018-12-16 09:30:07",
                "timezone": "America/New_York",
                "source": "Haymarket Square",
                "destination": "Logan International Airport",
                "cab_type": "Lyft",
                "name": "Lux Black XL",
                "price": 31.5,
                "distance": 3.2,
                "short_summary": "Mostly Cloudy",
                "long_summary": "Rain throughout the day.",
                "temperature": 42.34,
                "precipIntensity": 0.01,
                "windSpeed": 8.66,
            },
        )
        self.assertEqual(record.ride_semantics, "price_quote")
        self.assertEqual(record.price_semantics, "posted_quote_excluding_tip")
        self.assertIsNone(record.duration_minutes)
        self.assertTrue(record.airport)
        self.assertEqual(record.service, "premium")
        self.assertEqual(record.weather, "rain")
        self.assertEqual(record.pickup_datetime.utcoffset().total_seconds(), -5 * 3600)

    def test_chicago_target_excludes_tip(self):
        record = normalize_ride_record(
            "chicago",
            {
                "Trip ID": "trip-1",
                "Trip Start Timestamp": "01/21/2019 10:30:00 AM",
                "Trip Seconds": 561,
                "Trip Miles": 1.4,
                "Pickup Community Area": 76,
                "Dropoff Community Area": 28,
                "Fare": 10.0,
                "Tip": 4.0,
                "Additional Charges": 2.55,
                "Trip Total": 16.55,
                "Shared Trip Authorized": "false",
                "Trips Pooled": 1,
            },
        )
        self.assertEqual(record.actual_price, 12.55)
        self.assertEqual(
            record.price_semantics,
            "realized_fare_plus_mandatory_charges_excluding_tip",
        )
        self.assertAlmostEqual(record.duration_minutes, 561.0 / 60.0)
        self.assertTrue(record.airport)

    def test_invalid_economic_rows_are_rejected_with_reason(self):
        with self.assertRaises(DataValidationError) as caught:
            normalize_ride_record(
                "boston",
                {"price": "", "distance": 2.0, "datetime": "2018-12-01 12:00:00"},
            )
        self.assertEqual(caught.exception.code, "missing_price")

    def test_explicit_city_mismatch_cannot_be_silently_relabelled(self):
        with self.assertRaises(DataValidationError) as caught:
            normalize_ride_record(
                "nyc",
                {
                    "city": "Chicago",
                    "passenger_fare": 20,
                    "trip_length": 2,
                    "date": "2022-01-01",
                },
            )
        self.assertEqual(caught.exception.code, "city_mismatch")


class WeatherPipelineTests(unittest.TestCase):
    def test_timezone_aware_nearest_weather_join(self):
        observation = WeatherObservation(
            timestamp_utc=datetime(2018, 12, 16, 15, 0, tzinfo=timezone.utc),
            weather="rain",
            temperature_c=5.0,
            precipitation_mm=1.2,
            wind_speed_mps=3.0,
            visibility_m=5000.0,
            station="72509014739",
        )
        index = NOAAWeatherIndex([observation], max_gap_minutes=60)
        record = normalize_ride_record(
            "boston",
            {
                "datetime": "2018-12-16 10:15:00",
                "price": 15,
                "distance": 2,
                "source": "Back Bay",
                "destination": "North End",
            },
            weather_index=index,
        )
        self.assertEqual(record.weather, "rain")
        self.assertEqual(record.weather_source, "noaa_global_hourly")
        self.assertEqual(record.temperature_c, 5.0)

    def test_noaa_loader_parses_scaled_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nyc_laguardia_2022.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["STATION", "DATE", "TMP", "WND", "VIS", "AA1", "REM"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "STATION": "72503014732",
                        "DATE": "2022-01-01T12:00:00",
                        "TMP": "+0050,1",
                        "WND": "180,1,N,0030,1",
                        "VIS": "016000,1,9,9",
                        "AA1": "01,0010,2,1",
                        "REM": "METAR KJFK 011200Z -RA",
                    }
                )
            index = NOAAWeatherIndex.from_directory("nyc", directory)
        self.assertEqual(len(index.observations), 1)
        self.assertEqual(index.observations[0].weather, "rain")
        self.assertEqual(index.observations[0].temperature_c, 5.0)
        self.assertEqual(index.observations[0].precipitation_mm, 1.0)
        self.assertEqual(index.observations[0].wind_speed_mps, 3.0)


class CalibrationContractTests(unittest.TestCase):
    def test_calibration_json_rejects_wrong_market(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            path.write_text(
                json.dumps({"city": "Boston", "calibration": {"market": {}}}),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_calibration_json(str(path), expected_city="Chicago")

    def test_downloaded_puma_filters_are_city_specific_when_sources_exist(self):
        data_root = Path(__file__).resolve().parent / "data" / "raw"
        for city, minimum in (("nyc", 40), ("boston", 3), ("chicago", 10)):
            geography = default_source_paths(city, data_root)["puma_geography"]
            if not geography.exists():
                self.skipTest("downloaded PUMA geography is not present")
            self.assertGreaterEqual(len(puma_codes_for_city(city, geography)), minimum)

    def test_downloaded_bundles_pass_small_backtest_when_sources_exist(self):
        data_root = Path(__file__).resolve().parent / "data" / "raw"
        if not data_root.exists():
            self.skipTest("downloaded public sources are not present")
        for city in ("nyc", "boston", "chicago"):
            report = validate_city_bundle(city, data_root, sample_rows=10)
            self.assertEqual(report["sample_rows_accepted"], 10)
            self.assertEqual(report["sample_weather_join_rate"], 1.0)


class CoreMultiCityDatasetTests(unittest.TestCase):
    def test_core_scores_boston_and_chicago_without_nyc_relabelling(self):
        from Core import Core

        cases = (
            (
                "Boston",
                {
                    "id": "b1",
                    "datetime": "2018-12-16 09:30:07",
                    "source": "Back Bay",
                    "destination": "Logan International Airport",
                    "cab_type": "Lyft",
                    "name": "Lyft",
                    "price": 18.0,
                    "distance": 3.0,
                },
                "price_quote",
                18.0,
            ),
            (
                "Chicago",
                {
                    "Trip ID": "c1",
                    "Trip Start Timestamp": "01/21/2019 10:30:00 AM",
                    "Trip Seconds": 600,
                    "Trip Miles": 2.0,
                    "Pickup Community Area": 8,
                    "Dropoff Community Area": 28,
                    "Fare": 10.0,
                    "Tip": 5.0,
                    "Additional Charges": 2.5,
                    "Trip Total": 17.5,
                    "Shared Trip Authorized": "false",
                    "Trips Pooled": 1,
                },
                "realized_trip",
                12.5,
            ),
        )
        for city, row, semantics, expected_price in cases:
            with self.subTest(city=city), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "rides.csv"
                output = root / "comparison.csv"
                model = root / "policy.pt"
                with source.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    writer.writeheader()
                    writer.writerow(row)
                core = Core(
                    market_name=city,
                    seed=23,
                    choice_mode="parametric",
                    firm1_mode="RL",
                    firm2_mode="static",
                    total_customers_pool=20,
                )
                core.save_trained_model(str(model), model_id=f"{city.lower()}-test")
                core.load_trained_model(str(model))
                summary = core.compare_trained_rl_to_dataset(
                    dataset_root=str(root),
                    dataset_glob="*.csv",
                    dataset_city=city,
                    out_csv=str(output),
                    max_rows=1,
                    preview_rows=0,
                    policy_mode="argmax",
                )
                with output.open("r", encoding="utf-8", newline="") as handle:
                    scored = next(csv.DictReader(handle))
                self.assertEqual(summary["dataset_city"], city)
                self.assertEqual(summary["ride_semantics"], semantics)
                self.assertAlmostEqual(float(scored["actual_paid"]), expected_price)

    def test_core_refuses_market_dataset_city_mismatch(self):
        from Core import Core

        core = Core(
            market_name="Boston",
            seed=29,
            choice_mode="parametric",
            firm1_mode="RL",
            firm2_mode="static",
            total_customers_pool=20,
        )
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "policy.pt"
            core.save_trained_model(str(model), model_id="mismatch-test")
            core.load_trained_model(str(model))
            with self.assertRaises(ValueError):
                core.compare_trained_rl_to_dataset(
                    dataset_root=directory,
                    dataset_city="Chicago",
                )


if __name__ == "__main__":
    unittest.main()
