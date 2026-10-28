"""Regression tests for the v26 environment/evaluation contract (2026-10 review)."""

import copy
import csv
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from city_market_calibration import (
    fit_tariff_anchors,
    market_dynamics_calibration,
    price_level_index,
    records_to_arrays,
    weather_demand_multipliers,
)
from choice_models import CognitiveChoiceModel
from driver_supply import DriverSupplyConfig, DriverSupplyLayer
from market_dynamics import (
    ControllerPerturbation,
    FirmReputationMemory,
    MarketCalendar,
    MarketDynamicsConfig,
    TripBank,
    apply_threshold_ablation,
    expected_volume_multiplier,
    market_relative_bounds,
    perturbed_mode_name,
    split_perturbed_mode,
)
from pricing_models import FirmMetrics, FirmPIPriceGapPricer


def _write_trip_bank(path: str, rows: int = 2000, seed: int = 3) -> None:
    rng = np.random.default_rng(seed)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "distance_miles", "duration_minutes", "hour", "day_of_week", "airport",
            "service", "actual_price", "shared_ride", "weather",
        ])
        writer.writeheader()
        for _ in range(rows):
            hour = int(rng.integers(0, 24))
            distance = float(rng.gamma(2.0, 1.5)) + (6.0 if hour < 5 else 0.0)
            duration = max(2.0, 4.0 + 2.6 * distance + float(rng.normal(0.0, 4.0)))
            airport = bool(rng.random() < 0.08)
            price = (5.0 + 1.8 * distance + 0.45 * duration) + (5.0 if airport else 0.0)
            writer.writerow({
                "distance_miles": round(distance, 3), "duration_minutes": round(duration, 3),
                "hour": hour, "day_of_week": int(rng.integers(0, 7)), "airport": airport,
                "service": "economy", "actual_price": round(price, 2), "shared_ride": False,
                "weather": "clear",
            })


class TripBankAndCalendarTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.bank_path = os.path.join(self.tmp.name, "bank.csv")
        _write_trip_bank(self.bank_path)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_trip_bank_is_hour_conditional_and_not_deep_copied(self) -> None:
        bank = TripBank.from_csv(self.bank_path, city="Test")
        rng = np.random.default_rng(1)
        night = bank.sample(rng, hour=2, day_of_week=2, size=2000, jitter=0.0)
        noon = bank.sample(rng, hour=12, day_of_week=2, size=2000, jitter=0.0)
        self.assertGreater(night["distance"].mean(), noon["distance"].mean() + 3.0)
        self.assertIs(copy.deepcopy(bank), bank)
        self.assertEqual(bank.duration_source, "observed")

    def test_missing_duration_uses_reported_speed_profile(self) -> None:
        bank = TripBank(
            city="Quotes", distance=np.array([2.0, 3.0]), duration=np.array([np.nan, np.nan]),
            hour=np.array([8, 13]), day_of_week=np.array([1, 1]), airport=np.array([False, False]),
            service=np.array(["economy", "economy"], dtype=object), price=np.array([10.0, 12.0]),
        )
        self.assertEqual(bank.duration_source, "assumed_hourly_speed_profile")
        self.assertAlmostEqual(bank.duration[0], 60.0 * 2.0 / 18.0)

    def test_calendar_is_sequential_with_persistent_weather(self) -> None:
        calendar = MarketCalendar()
        rng = np.random.default_rng(5)
        probs = {"clear": 0.8, "rain": 0.15, "snow": 0.05}
        calendar.reset(rng, day_of_week_probs=np.ones(7), weather_probs=probs)
        days, weather, changes = [], [], 0
        for _ in range(4000):
            calendar.advance_day(rng, steps_per_day=6, hour_probs=np.ones(24),
                                 weather_probs=probs, persistence=0.7)
            if weather and weather[-1] != calendar.weather:
                changes += 1
            days.append(calendar.day_of_week)
            weather.append(calendar.weather)
            hours = [calendar.next_block()[1] for _ in range(6)]
            self.assertEqual(hours, sorted(hours))
            self.assertTrue(all(lo * 4 <= h < lo * 4 + 4 for lo, h in enumerate(hours)))
        self.assertTrue(all((b - a) % 7 == 1 for a, b in zip(days, days[1:])))
        share_clear = weather.count("clear") / len(weather)
        self.assertAlmostEqual(share_clear, 0.8, delta=0.04)
        self.assertLess(changes / len(weather), 0.2)

    def test_hourly_volume_averages_to_one_over_a_week(self) -> None:
        hour_probs = np.array([0.5] * 6 + [1.5] * 12 + [1.0] * 6)
        hour_probs = hour_probs / hour_probs.sum()
        dow = np.ones(7) / 7.0
        total = 0.0
        for day in range(7):
            for block in range(6):
                share = float(hour_probs[block * 4 : block * 4 + 4].sum())
                total += expected_volume_multiplier(
                    block_share=share, steps_per_day=6, day_of_week=day,
                    day_of_week_probs=dow, weather="clear", weather_demand_multiplier={},
                )
        self.assertAlmostEqual(total / 42.0, 1.0, places=9)


class CalibrationTests(unittest.TestCase):
    def test_tariff_fit_recovers_synthetic_structure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bank.csv")
            _write_trip_bank(path, rows=4000)
            with open(path, encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        tariff = fit_tariff_anchors(records_to_arrays(rows), duration_identified=True,
                                    service_identified=False, airport_identified=True)
        self.assertAlmostEqual(tariff["base_fare"] + tariff["booking_fee"], 5.0, delta=0.3)
        self.assertAlmostEqual(tariff["per_mile"], 1.8, delta=0.15)
        self.assertAlmostEqual(tariff["per_minute"], 0.45, delta=0.08)
        self.assertAlmostEqual(tariff["airport_fee"], 5.0, delta=0.6)
        self.assertGreater(tariff["fit"]["r2"], 0.95)

    def test_price_level_and_weather_intensity(self) -> None:
        tariff = {"base_fare": 1.0, "booking_fee": 1.0, "per_mile": 1.0, "per_minute": 0.2}
        self.assertAlmostEqual(price_level_index(tariff, reference_fare=9.0), 1.0)
        arrays = {"weather": np.array(["rain"] * 300 + ["clear"] * 700, dtype=object)}
        mult = weather_demand_multipliers(arrays, {"clear": 0.8, "rain": 0.2})
        self.assertGreater(mult["rain"], mult["clear"])
        self.assertAlmostEqual(0.8 * mult["clear"] + 0.2 * mult["rain"], 1.0, places=9)

    def test_choice_model_scales_reservation_and_outside_option(self) -> None:
        model = CognitiveChoiceModel(seed=1)
        profile = {"IncomeBracket": "50k-100k", "HouseholdSize": 1}
        scenario = {"DistanceMiles": 4.0, "DurationMinutes": 15.0, "Hour": 12}
        base_reservation = model._reservation_price(profile, scenario)
        base_outside = model._outside_option_cost(profile, scenario)
        model.apply_calibration({"price_level_index": 0.5, "outside_option_base_fare": 2.5})
        self.assertAlmostEqual(model._reservation_price(profile, scenario), 0.5 * base_reservation)
        self.assertAlmostEqual(model._outside_option_cost(profile, scenario), 2.5 + 0.5 * (base_outside - 2.9))


class SupplyAndMetricsTests(unittest.TestCase):
    def test_linear_driver_scaling_keeps_request_driver_ratio(self) -> None:
        def active(customers: int, linear: bool) -> int:
            layer = DriverSupplyLayer(seed=1, config=DriverSupplyConfig(linear_demand_scaling=linear))
            layer.begin_batch(customers_per_step=customers, hour=12, weather="clear")
            return layer._batch_states["Firm1"].active_drivers
        self.assertNotAlmostEqual(100 / active(100, False), 200 / active(200, False), places=2)
        self.assertAlmostEqual(100 / active(100, True), 200 / active(200, True), delta=0.02)

    def test_money_metrics_use_reference_total(self) -> None:
        metrics = FirmMetrics(profit=150.0, revenue=300.0, total=150, completed=60)
        self.assertAlmostEqual(metrics.profit_per_request, 1.0)
        metrics.reference_total = 100
        self.assertAlmostEqual(metrics.profit_per_request, 1.5)
        self.assertAlmostEqual(metrics.completed_share, 60 / 150)


class OpponentAndAblationTests(unittest.TestCase):
    def test_perturbation_is_seeded_and_changes_parameters(self) -> None:
        a, b = FirmPIPriceGapPricer(seed=1), FirmPIPriceGapPricer(seed=1)
        ControllerPerturbation(scale=0.3).apply(a, np.random.default_rng(9))
        ControllerPerturbation(scale=0.3).apply(b, np.random.default_rng(9))
        self.assertEqual(a.kp_gap, b.kp_gap)
        self.assertNotEqual(a.kp_gap, FirmPIPriceGapPricer(seed=1).kp_gap)
        self.assertEqual(split_perturbed_mode(perturbed_mode_name("mpc_grid", 2)), ("mpc_grid", 2))
        self.assertEqual(split_perturbed_mode("mpc_grid"), ("mpc_grid", None))

    def test_relative_bounds_scale_with_city_anchor(self) -> None:
        bounds = market_relative_bounds({"base_fare": 2.0, "per_minute": 0.2, "per_mile": 1.0,
                                         "booking_fee": 1.0, "airport_fee": 3.0})
        self.assertAlmostEqual(bounds["per_mile"][0], 0.4)
        self.assertAlmostEqual(bounds["per_mile"][1], 2.0)

    def test_threshold_ablation_modes(self) -> None:
        profiles = [{"PriceThreshold": float(v), "IncomeBracket": "<50k"} for v in (1.0, 2.0, 3.0, 4.0)]
        apply_threshold_ablation(profiles, "constant_mean", seed=1)
        self.assertTrue(all(p["PriceThreshold"] == 2.5 for p in profiles))
        self.assertEqual([p["PriceThresholdOriginal"] for p in profiles], [1.0, 2.0, 3.0, 4.0])
        shuffled = [{"PriceThreshold": float(v)} for v in range(50)]
        apply_threshold_ablation(shuffled, "shuffled", seed=3)
        self.assertEqual(sorted(p["PriceThreshold"] for p in shuffled), [float(v) for v in range(50)])
        self.assertNotEqual([p["PriceThreshold"] for p in shuffled], [float(v) for v in range(50)])


class GenericCityAdapterTests(unittest.TestCase):
    def test_field_map_normalizes_units_and_airport(self) -> None:
        from city_data_pipeline import CityDataSpec, _LooseRow, _normalize_with_field_map
        spec = CityDataSpec(
            key="testcity", market_name="Test City", aliases=(), timezone="UTC",
            ride_semantics="realized_trip", price_semantics="fare", kaggle_slug="", ride_root="",
            ride_glob="*.csv", weather_glob="", weather_station="", pums_person="", pums_household="",
            puma_geography="", puma_name_pattern="", state_fips="", place_fips="", sources={},
            field_map={
                "price_components": {"add": ["fare", "fees"]},
                "distance": {"columns": ["km"], "unit": "km"},
                "duration": {"columns": ["secs"], "unit": "seconds"},
                "pickup_datetime": ["start"],
                "airport": {"lat_columns": ["lat"], "lon_columns": ["lon"],
                            "coordinates": [[47.45, -122.31, 2.0]]},
            },
            market={}, status="available",
        )
        mapped = _normalize_with_field_map(spec, _LooseRow({
            "fare": "10", "fees": "2.5", "km": "10", "secs": "600", "start": "2024-01-01 08:00:00",
            "lat": "47.451", "lon": "-122.309",
        }))
        price, _, distance, _, duration, *_ = mapped
        self.assertAlmostEqual(price, 12.5)
        self.assertAlmostEqual(distance, 6.21371, places=4)
        self.assertAlmostEqual(duration, 10.0)
        self.assertTrue(mapped[10])


class CoreV26IntegrationTests(unittest.TestCase):
    """Simulator-only integration (no learning)."""

    def _core(self, *, firm2_mode: str = "static"):
        from Core import Core
        tmp = tempfile.mkdtemp()
        bank = os.path.join(tmp, "trip_bank.csv")
        _write_trip_bank(bank, rows=3000)
        with open(bank, encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        v26 = market_dynamics_calibration(
            rows, weather_probs={"clear": 0.8, "rain": 0.15, "snow": 0.05},
            duration_identified=True, service_identified=False, airport_identified=True,
            outside_option_base_fare=2.5,
        )
        hour_probs = {str(h): (2.0 if 7 <= h < 10 or 16 <= h < 20 else 1.0) for h in range(24)}
        calibration = {
            "market": {"hour_probs": hour_probs, "weather_probs": {"clear": 0.8, "rain": 0.15, "snow": 0.05},
                       "weather_demand_multiplier": v26["weather_demand_multiplier"]},
            "tariff": v26["tariff"], "choice": v26["choice"], "supply": v26["supply"],
            "trip_bank": {"path": "trip_bank.csv", "airport_identified": True, "service_identified": False},
            "_source_dir": tmp,
        }
        core = Core(
            market_name="Chicago", seed=31, firm1_mode="static",
            firm2_mode=firm2_mode, total_customers_pool=300,
            openai_api_key="",
        )
        core.apply_calibration(calibration)
        core.configure_v26(market_dynamics=MarketDynamicsConfig(enabled=True), steps_per_day=6)
        return core, calibration

    def test_explicit_empty_api_key_disables_ambient_credential(self) -> None:
        from Core import Core

        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "ambient-test-key"}):
            core = Core(
                market_name="Chicago", seed=32,
                firm1_mode="static", firm2_mode="static",
                total_customers_pool=10, openai_api_key="",
            )

        self.assertEqual(core.openai_api_key, "")

    def test_calibrated_anchors_and_symmetric_firms(self) -> None:
        core, calibration = self._core()
        self.assertAlmostEqual(core.market.curr_market.per_mile, calibration["tariff"]["per_mile"])
        self.assertEqual(core.driver_supply.config.firm2_supply_bias, 1.0)
        self.assertEqual(core.driver_supply.pay_policies["Firm1"].fare_share,
                         core.driver_supply.pay_policies["Firm2"].fare_share)

    def test_v26_zeroes_firm_identity_utility_and_observes_volume(self) -> None:
        core, _ = self._core()
        self.assertEqual(core.choice_model.near_tie_habit_firm2, 0.0)
        self.assertEqual(core.choice_model.reliability_bias_firm1, 0.0)
        self.assertIsNone(core.choice_model.reputation)
        peak = core._build_city_context(hour=17, day_of_week=2, weather="clear")["demand_intensity"]
        night = core._build_city_context(hour=3, day_of_week=2, weather="clear")["demand_intensity"]
        self.assertGreater(peak, night)

    def test_rider_memory_updates_from_posted_tariffs(self) -> None:
        core, _ = self._core()
        core.configure_v26(market_dynamics=MarketDynamicsConfig(enabled=True, rider_memory=True,
                                                                supply_feedback=False), steps_per_day=6)
        self.assertEqual(core.driver_supply.config.supply_elasticity, 0.0)
        memory = core.choice_model.reputation
        self.assertIsNotNone(memory)
        base = core.market.curr_market
        for key in core.shared_edit_keys:
            setattr(core.firm1.overrides, key, 0.8 * float(getattr(base, key)))
            setattr(core.firm2.overrides, key, 1.0 * float(getattr(base, key)))
        profiles = core._sample_profiles_from_pool(120)
        core.simulate_batch(day_of_week=2, weather="clear", hour=12, customers_per_step=120,
                            sampled_profiles=profiles, collect_rows=False,
                            supply_reference_customers=120)
        self.assertLess(memory.price_index["Firm1"], memory.price_index["Firm2"])
        self.assertEqual(memory.updates, 1)
        acc = {}
        core._accumulate_welfare(acc, core.last_welfare_stats)
        summary = core._summarize_welfare(acc)
        self.assertGreater(summary["welfare_requests"], 0)
        snapshot = copy.deepcopy(core.choice_model)
        core._reset_rider_memory()
        self.assertEqual(memory.updates, 0)
        self.assertEqual(snapshot.reputation.updates, 1)

    def test_capacity_calibration_and_fleet_scale_are_recorded(self) -> None:
        core, calibration = self._core()
        cal = core.v26_settings["driver_capacity_calibration"]
        self.assertTrue(cal["enabled"])
        self.assertGreaterEqual(cal["anchor_fulfillment"], 0.95 - 0.02)
        expected_wage = 24.0 * calibration["choice"]["price_level_index"]
        self.assertAlmostEqual(core.driver_supply.config.reservation_wage_per_hour,
                               expected_wage)
        core.configure_v26(market_dynamics=MarketDynamicsConfig(enabled=True), steps_per_day=6)
        self.assertAlmostEqual(core.driver_supply.config.reservation_wage_per_hour,
                               expected_wage)
        fleet = core.driver_supply.config.base_active_drivers
        core.configure_v26(market_dynamics=MarketDynamicsConfig(enabled=True, fleet_scale=1.5,
                                                                capacity_target_fulfillment=0.0),
                           steps_per_day=6)
        self.assertEqual(core.driver_supply.config.base_active_drivers,
                         int(round(core._v26_legacy_fleet * 1.5)))
        self.assertGreater(fleet, 0)

    def test_anchor_pay_basis_decouples_driver_pay_from_rider_discount(self) -> None:
        layer = DriverSupplyLayer(seed=3)
        layer.begin_batch(customers_per_step=100, hour=12, weather="clear")
        cheap = layer.dispatch("Firm1", 10.0, 3.0, 12.0, pay_reference_fare=20.0)
        layer.begin_batch(customers_per_step=100, hour=12, weather="clear")
        legacy = layer.dispatch("Firm1", 10.0, 3.0, 12.0)
        if cheap.fulfilled and legacy.fulfilled:
            self.assertGreater(cheap.driver_pay, legacy.driver_pay)
        self.assertGreaterEqual(cheap.acceptance_probability, legacy.acceptance_probability)

    def test_rebuilt_validation_market_keeps_calibration(self) -> None:
        core, calibration = self._core()
        rebuilt = core._rebuild_market(core.market_name, 77)
        self.assertAlmostEqual(rebuilt.curr_market.per_mile, calibration["tariff"]["per_mile"])
        expected = np.array([calibration["market"]["hour_probs"][str(h)] for h in range(24)], dtype=float)
        np.testing.assert_allclose(rebuilt._base_hour_probs, expected / expected.sum())
        self.assertIsNotNone(rebuilt.trip_bank)

    def test_sequential_run_varies_volume_and_uses_bank(self) -> None:
        core, _ = self._core()
        rows = core.run(days=2, timesteps_per_day=6, customers_per_step=60)
        hours = [r["Hour"] for r in rows]
        self.assertTrue(all(0 <= h < 24 for h in hours))
        counts = {}
        for row in rows:
            counts[(row["DayOfWeek"], row["Hour"])] = counts.get((row["DayOfWeek"], row["Hour"]), 0) + 1
        self.assertGreater(max(counts.values()), min(counts.values()))
        self.assertEqual(len({r["DayOfWeek"] for r in rows}), 2)

    def test_benchmarks_excluded_and_perturbed_variants_build(self) -> None:
        core, _ = self._core()
        core.configure_v26(market_dynamics=core.market_dynamics, steps_per_day=6,
                           exclude_benchmarks_from_training=True, ood_perturbed_variants=2)
        from Core import BENCHMARK_OPPONENT_MODES
        self.assertFalse(set(core.training_dynamic_opponent_modes) & set(BENCHMARK_OPPONENT_MODES))
        names = [name for name, _ in core._ood_opponent_rollouts()]
        self.assertEqual(len(names), 10)
        a = core._make_pricer(names[0], seed=1)
        b = core._make_pricer(names[0], seed=999)
        self.assertEqual(getattr(a, "_perturbation_draw"), getattr(b, "_perturbation_draw"))

    def test_spillover_serves_unfulfilled_riders_on_the_other_platform(self) -> None:
        core, _ = self._core()
        profiles = core._sample_profiles_from_pool(400)
        base = core.market.curr_market
        for key in core.shared_edit_keys:
            setattr(core.firm1.overrides, key, 0.6 * float(getattr(base, key)))
            setattr(core.firm2.overrides, key, 1.0 * float(getattr(base, key)))
        rows, m1, m2, *_ = core.simulate_batch(
            day_of_week=2, weather="clear", hour=17,
            customers_per_step=400, sampled_profiles=profiles,
            collect_rows=True, supply_reference_customers=100,
        )
        self.assertGreater(m2.spillover_completed, 0)
        self.assertGreater(m2.spillover_attempts, 0)
        self.assertLessEqual(m1.chosen + m2.chosen, m1.total)
        spillovers = [
            row for row in rows
            if str(row["MatchStatus"]).startswith("CompletedAfterSpillover")
        ]
        self.assertTrue(spillovers)
        self.assertTrue(all(row["MatchedFirm"] == "Firm2" for row in spillovers))
        self.assertTrue(all(row["DriverAcceptanceProbability"] > 0.0 for row in spillovers))

    def test_fulfillment_rate_uses_all_dispatch_attempts(self) -> None:
        metrics = FirmMetrics(
            total=10,
            chosen=3,
            completed=4,
            dispatch_offers=5,
            spillover_attempts=2,
        )
        self.assertAlmostEqual(metrics.fulfillment_rate, 0.8)


class _FixedDraws:
    """RNG stand-in: no outside option, then a fixed platform-choice draw."""

    def __init__(self, choice_draw: float):
        self.draws = [0.999, float(choice_draw)]

    def random(self):
        return self.draws.pop(0) if self.draws else 0.5

    def normal(self, *args, **kwargs):
        return 0.0


class ChoiceSymmetryAndMemoryTests(unittest.TestCase):
    profile = {"IncomeBracket": "50k-100k", "Age": 35, "HouseholdSize": 1,
               "LoyaltyFirm": None, "LoyaltyStrength": 0.0, "PriceThreshold": 1.5}
    scenario = {"DistanceMiles": 3.0, "DurationMinutes": 12.0, "DayOfWeek": 2, "Hour": 12,
                "Weather": "clear", "Airport": False, "Service": "economy"}

    def _choose(self, model, draw, p1=12.0, p2=12.0):
        model.rng = _FixedDraws(draw)
        return model.choose(dict(self.profile), dict(self.scenario), p1, p2).choice

    def test_legacy_constants_favor_firm2_at_ties(self) -> None:
        model = CognitiveChoiceModel(seed=1)
        self.assertEqual(self._choose(model, 0.4999), "Firm2")

    def test_symmetric_model_is_neutral_at_ties(self) -> None:
        model = CognitiveChoiceModel(seed=1)
        model.make_symmetric()
        self.assertEqual(self._choose(model, 0.4999), "Firm1")
        self.assertEqual(self._choose(model, 0.5001), "Firm2")

    def test_price_reputation_shifts_future_choice(self) -> None:
        model = CognitiveChoiceModel(seed=1)
        model.make_symmetric()
        memory = FirmReputationMemory(alpha=1.0)
        memory.update("Firm1", price_index=0.8, fulfillment=1.0)
        memory.update("Firm2", price_index=1.0, fulfillment=1.0)
        model.reputation = memory
        self.assertGreater(memory.choice_utility(0.5), 0.0)
        self.assertEqual(self._choose(model, 0.52), "Firm1")
        memory.reset()
        self.assertEqual(memory.choice_utility(0.5), 0.0)
        self.assertEqual(memory.outside_option_factor(), 1.0)

    def test_reputation_smoothing_and_outside_option(self) -> None:
        memory = FirmReputationMemory(alpha=0.5)
        memory.update("Firm1", price_index=1.4, fulfillment=0.6)
        memory.update("Firm2", price_index=1.4, fulfillment=None)
        self.assertAlmostEqual(memory.price_index["Firm1"], 1.2)
        self.assertAlmostEqual(memory.reliability["Firm1"], 0.8)
        self.assertAlmostEqual(memory.reliability["Firm2"], 1.0)
        self.assertGreater(memory.outside_option_factor(), 1.0)
        self.assertLess(memory.choice_utility(0.5), 0.0)


class WelfareSummaryTests(unittest.TestCase):
    def test_summary_is_per_request_and_by_income(self) -> None:
        from Core import Core
        acc = {}
        stats = {"by_income": {"<50k": [10, 6, 60.0, 12.0, 3, 30.0],
                               "200k+": [10, 9, 99.0, 45.0, 0, 27.0]},
                 "firm1_profit": 20.0, "firm2_profit": 10.0, "driver_pay": 100.0}
        Core._accumulate_welfare(acc, stats)
        Core._accumulate_welfare(acc, stats)
        summary = Core._summarize_welfare(acc)
        self.assertEqual(summary["welfare_requests"], 40.0)
        self.assertAlmostEqual(summary["welfare_served_rate"], 30 / 40)
        self.assertAlmostEqual(summary["welfare_consumer_surplus_per_request"], 114.0 / 40)
        self.assertAlmostEqual(summary["equity_served_gap_high_minus_low"], 0.9 - 0.6)
        self.assertAlmostEqual(summary["equity_mean_fare_paid__<50k"], 10.0)
        self.assertAlmostEqual(summary["welfare_total_surplus_per_request"],
                               (114.0 + 60.0 + 200.0) / 40)
        self.assertEqual(Core._summarize_welfare({}), {})


if __name__ == "__main__":
    unittest.main()
