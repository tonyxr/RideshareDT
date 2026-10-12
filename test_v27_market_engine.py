"""v27 market engine and shock process: unit tests (no torch, no network)."""

from __future__ import annotations

import copy
import datetime as dt
import unittest

import numpy as np

from choice_models import CognitiveChoiceModel
from driver_supply import DriverPayPolicy, DriverSupplyLayer
from market_dynamics import FirmReputationMemory, RiderProfileMemory
from MarketInteraction import MarketInteraction, RideContext
from Market_models import CoefficientOverrides
from market_engine import (
    EngineConfig,
    FleetState,
    MarketEngine,
    apply_emergency_cap,
    cognitive_probabilities,
    driver_acceptance,
    driver_pay_vector,
    nested_logit_probabilities,
    pickup_minutes,
    profile_features,
    quote_vector,
    solve_capacity,
)
from market_shocks import (
    PUBLIC_SHOCK_FEATURE_KEYS,
    ShockConfig,
    ShockEffect,
    ShockProcess,
    holiday_dates,
)


class _FixedDraws:
    """Stand-in RNG whose ``random()`` returns a scripted sequence."""

    def __init__(self, values):
        self.values = list(values)

    def random(self):
        return float(self.values.pop(0))


def _scalar_probabilities(model, profile, scenario, p1, p2):
    """Recover P(no ride) and P(Firm1 | ride) from ``choose`` by bisection."""
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        model.rng = _FixedDraws([mid, 0.5])
        if model.choose(profile, scenario, p1, p2).choice == "NoRide":
            lo = mid
        else:
            hi = mid
    p0 = 0.5 * (lo + hi)
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        model.rng = _FixedDraws([0.999999, mid])
        if model.choose(profile, scenario, p1, p2).choice == "Firm1":
            lo = mid
        else:
            hi = mid
    return p0, 0.5 * (lo + hi)


def _random_profiles(rng, n):
    incomes = ["<50k", "50k-100k", "100k-200k", "200k+"]
    jobs = ["Student", "Employed", "Unemployed", "Retired"]
    profiles = []
    for _ in range(n):
        loyalty = rng.choice(["Firm1", "Firm2", None])
        profile = {
            "Age": int(rng.integers(18, 80)),
            "IncomeBracket": str(rng.choice(incomes)),
            "HouseholdSize": int(rng.integers(1, 6)),
            "EmploymentStatus": str(rng.choice(jobs)),
            "LoyaltyType": str(rng.choice(["New", "Returning"])),
            "LoyaltyFirm": loyalty,
            "LoyaltyStrength": float(rng.uniform(0.0, 1.0)) if loyalty else 0.0,
            "PriceThreshold": float(rng.uniform(0.3, 4.0)),
        }
        if rng.random() < 0.1:
            profile["ReservationPrice"] = float(rng.uniform(8.0, 60.0))
        if rng.random() < 0.1:
            profile["OutsideOptionCost"] = float(rng.uniform(1.0, 30.0))
        profiles.append(profile)
    return profiles


def _random_trips(rng, n):
    return {
        "distance": np.round(rng.uniform(0.3, 18.0, n), 2),
        "duration": rng.uniform(3.0, 55.0, n),
        "airport": rng.random(n) < 0.2,
        "service": rng.choice(["economy", "premium"], size=n, p=[0.8, 0.2]),
    }


class CognitiveParityTest(unittest.TestCase):
    """The vectorized choice probabilities equal ``CognitiveChoiceModel.choose``."""

    def _check(self, model, *, hour, day, weather, seed, memory=False):
        rng = np.random.default_rng(seed)
        n = 60
        profiles = _random_profiles(rng, n)
        if memory:
            profile_memory = RiderProfileMemory(
                alpha=1.0, choice_weight=0.8, outside_weight=0.6
            )
            profile_memory.update(
                profiles,
                choice_target=np.linspace(-0.8, 0.8, n),
                outside_target=np.linspace(0.6, -0.6, n),
            )
            model.profile_memory = profile_memory
        trips = _random_trips(rng, n)
        p1 = np.round(rng.uniform(6.0, 70.0, n), 2)
        p2 = np.round(p1 + rng.normal(0.0, 3.0, n), 2)
        w1 = rng.uniform(2.0, 12.0, n)
        w2 = rng.uniform(2.0, 12.0, n)
        c1 = np.clip((w1 - 6.0) / 18.0, 0.0, 0.95)
        c2 = np.clip((w2 - 6.0) / 18.0, 0.0, 0.95)
        feats = profile_features(
            profiles, profile_memory=getattr(model, "profile_memory", None)
        )
        p0_vec, pf1_vec, _ = cognitive_probabilities(
            model, feats, trips, hour=hour, day_of_week=day, weather=weather,
            price1=p1, price2=p2, wait1=w1, wait2=w2, cancel1=c1, cancel2=c2,
            shock=ShockEffect(),
        )
        for i in range(n):
            scenario = {
                "DistanceMiles": float(trips["distance"][i]),
                "DurationMinutes": float(trips["duration"][i]),
                "DayOfWeek": day, "Hour": hour, "Weather": weather,
                "Airport": bool(trips["airport"][i]), "Service": str(trips["service"][i]),
                "WaitEstimateFirm1": float(w1[i]), "WaitEstimateFirm2": float(w2[i]),
                "CancelRiskFirm1": float(c1[i]), "CancelRiskFirm2": float(c2[i]),
            }
            p0, pf1 = _scalar_probabilities(model, profiles[i], scenario, float(p1[i]), float(p2[i]))
            self.assertAlmostEqual(p0_vec[i], p0, places=8, msg=f"p0 rider {i}")
            if p0 < 0.98 - 1e-6:
                self.assertAlmostEqual(pf1_vec[i], pf1, places=8, msg=f"pf1 rider {i}")

    def test_legacy_constants(self):
        model = CognitiveChoiceModel(seed=1)
        model.apply_calibration({"price_sensitivity_scale": 1.3, "loyalty_scale": 0.8, "reliability_scale": 1.1})
        self._check(model, hour=8, day=6, weather="rain", seed=11)
        self._check(model, hour=13, day=2, weather="clear", seed=12)

    def test_symmetric_price_level_and_memory(self):
        model = CognitiveChoiceModel(seed=2)
        model.apply_calibration({"price_level_index": 0.85, "outside_option_base_fare": 2.4})
        model.activate_price_level(True)
        model.make_symmetric()
        memory = FirmReputationMemory(alpha=0.3)
        memory.update("Firm1", price_index=1.2, fulfillment=0.8)
        memory.update("Firm2", price_index=0.9, fulfillment=0.95)
        model.reputation = memory
        self._check(model, hour=17, day=5, weather="snow", seed=13)
        self._check(model, hour=22, day=1, weather="cloudy", seed=14)

    def test_profile_specific_memory_matches_scalar_choice_model(self):
        model = CognitiveChoiceModel(seed=3)
        model.experimental_unbounded_economics = True
        self._check(
            model,
            hour=9,
            day=2,
            weather="rain",
            seed=15,
            memory=True,
        )


class QuoteParityTest(unittest.TestCase):
    def test_quote_vector_matches_quote_price(self):
        market = MarketInteraction(city_name="Boston", seed=3)
        market.set_market("Boston")
        rng = np.random.default_rng(5)
        n = 200
        trips = _random_trips(rng, n)
        overrides = CoefficientOverrides(base_fare=3.1, per_mile=1.7, airport_fee=4.0)
        for hour, day, weather in ((8, 0, "clear"), (18, 5, "rain"), (2, 6, "snow")):
            for ov in (None, overrides):
                vec = quote_vector(
                    market.curr_market, ov, time_multiplier=market._time_multiplier(hour),
                    day_of_week=day, weather=weather, distance=trips["distance"],
                    duration=trips["duration"], airport=trips["airport"], service=trips["service"],
                )
                for i in range(n):
                    ctx = RideContext(day_of_week=day, weather=weather, hour=hour,
                                      airport=bool(trips["airport"][i]), service=str(trips["service"][i]))
                    scalar = market.quote_price(float(trips["distance"][i]), float(trips["duration"][i]), ctx, overrides=ov)
                    self.assertAlmostEqual(vec[i], scalar, delta=0.0100001)

    def test_emergency_cap(self):
        price = np.array([10.0, 30.0, 50.0])
        anchor = np.array([10.0, 10.0, 40.0])
        np.testing.assert_allclose(apply_emergency_cap(price, anchor, 1.5, True), [10.0, 15.0, 50.0])
        np.testing.assert_allclose(apply_emergency_cap(price, anchor, 1.5, False), price)
        np.testing.assert_allclose(apply_emergency_cap(price, anchor, 0.0, True), price)


class DriverParityTest(unittest.TestCase):
    def test_pay_and_acceptance_match_driver_layer(self):
        layer = DriverSupplyLayer(seed=1)
        layer.pay_policies["Firm1"] = DriverPayPolicy(fare_share=0.7, minimum_pay=4.0, incentive_multiplier=1.1)
        rng = np.random.default_rng(9)
        fares = rng.uniform(3.0, 60.0, 50)
        dist = rng.uniform(0.2, 15.0, 50)
        dur = rng.uniform(3.0, 50.0, 50)
        pay = driver_pay_vector(layer.pay_policies["Firm1"], fares)
        acc = driver_acceptance(
            pay, dist, dur, 5.0, reservation_wage=layer.config.reservation_wage_per_hour,
            operating_cost_per_mile=layer.config.operating_cost_per_mile, weather="snow",
            threshold=layer.config.acceptance_ratio_threshold,
        )
        for i in range(50):
            scalar_pay = layer.quote_driver_pay("Firm1", float(fares[i]), 0, 0, 0, False)
            self.assertAlmostEqual(pay[i], scalar_pay, places=10)
            scalar_acc = layer.driver_acceptance_probability(scalar_pay, float(dist[i]), float(dur[i]), 5.0, "snow")
            self.assertAlmostEqual(acc[i], scalar_acc, places=10)


class CapacityTest(unittest.TestCase):
    def setUp(self):
        self.cfg = EngineConfig(enabled=True)

    def test_ample_capacity_serves_all_at_fixed_point(self):
        demand = np.full(100, 1.0)
        trip = np.full(100, 15.0)
        phi, u, pick = solve_capacity(demand, trip, 100 * 30.0 * 3, self.cfg)
        self.assertEqual(phi, 1.0)
        self.assertAlmostEqual(u * 9000.0, 100 * 15.0 + 100 * pick, delta=1e-3)
        self.assertAlmostEqual(pick, pickup_minutes(u, self.cfg), places=10)

    def test_scarce_capacity_rations(self):
        demand = np.full(100, 1.0)
        trip = np.full(100, 15.0)
        phi, u, pick = solve_capacity(demand, trip, 600.0, self.cfg)
        self.assertLess(phi, 1.0)
        self.assertLessEqual(u, self.cfg.max_utilization + 1e-12)
        self.assertAlmostEqual(phi * (1500.0 + 100.0 * pick), u * 600.0, delta=1e-6)

    def test_monotone_in_fleet(self):
        demand = np.full(50, 1.0)
        trip = np.full(50, 20.0)
        served = [solve_capacity(demand, trip, T, self.cfg)[0] for T in (300, 600, 1200, 2400, 4800)]
        utils = [solve_capacity(demand, trip, T, self.cfg)[1] for T in (2400, 4800, 9600)]
        self.assertTrue(all(a <= b + 1e-12 for a, b in zip(served, served[1:])))
        self.assertTrue(all(a >= b - 1e-12 for a, b in zip(utils, utils[1:])))

    def test_lower_root_when_concave_peak(self):
        # Short trips: pickup dominates and h(u) peaks before u_max.  The
        # solver must pick the low-utilization equilibrium, not ration.
        demand = np.full(10, 1.0)
        trip = np.full(10, 1.0)
        T = 140.0
        phi, u, pick = solve_capacity(demand, trip, T, self.cfg)
        need_at_max = 10.0 + 10.0 * pickup_minutes(self.cfg.max_utilization, self.cfg)
        self.assertGreater(need_at_max, self.cfg.max_utilization * T)  # u_max itself is infeasible
        self.assertEqual(phi, 1.0)
        self.assertAlmostEqual(u * T, 10.0 + 10.0 * pick, delta=1e-4)

    def test_no_drivers_serves_nothing(self):
        phi, _, _ = solve_capacity(np.ones(3), np.ones(3) * 10, 0.0, self.cfg)
        self.assertEqual(phi, 0.0)


def _engine_inputs(seed=0, n=80):
    rng = np.random.default_rng(seed)
    model = CognitiveChoiceModel(seed=1)
    model.make_symmetric()
    profiles = _random_profiles(rng, n)
    trips = _random_trips(rng, n)
    anchor = np.round(4.0 + 1.8 * trips["distance"] + 0.3 * trips["duration"], 2)
    policies = {"Firm1": DriverPayPolicy(fare_share=0.72), "Firm2": DriverPayPolicy(fare_share=0.72)}
    return model, profiles, trips, anchor, policies


def _step(engine, fleet, model, profiles, trips, anchor, policies, *, p1_mult=1.0, p2_mult=1.0,
          shock=None, base=100.0, realized=100.0, spill=True, block_minutes=240.0):
    shock = shock or ShockEffect()
    online = engine.online_drivers(fleet, base_requests=base, supply_multiplier=1.0, weather="clear",
                                   shock=shock, reservation_wage=24.0)
    return engine.step(
        model=model, profiles=profiles, trips=trips, hour=13, day_of_week=2, weather="clear",
        price={"Firm1": np.round(anchor * p1_mult, 2), "Firm2": np.round(anchor * p2_mult, 2)},
        pay_basis={"Firm1": anchor, "Firm2": anchor}, pay_policies=policies, fleet=fleet,
        online=online, block_minutes=block_minutes, realized_requests=realized, shock=shock,
        reservation_wage=24.0, operating_cost_per_mile=0.35, acceptance_threshold=1.0,
        platform_variable_cost=0.45, spillover=spill,
    )


class EngineStepTest(unittest.TestCase):
    def test_engine_updates_identity_specific_rider_memory(self):
        engine = MarketEngine(EngineConfig(enabled=True, fleet_per_base_request=0.3))
        model, profiles, trips, anchor, policies = _engine_inputs(seed=21, n=32)
        for index, profile in enumerate(profiles):
            profile["ProfileID"] = f"memory-{index}"
        model.profile_memory = RiderProfileMemory(alpha=1.0)
        _step(
            engine,
            FleetState(),
            model,
            profiles,
            trips,
            anchor,
            policies,
            p1_mult=0.9,
        )
        self.assertEqual(sum(model.profile_memory.encounters.values()), 32)
        remembered = profile_features(
            profiles, profile_memory=model.profile_memory
        )
        self.assertGreater(
            float(np.max(np.abs(remembered["profile_memory_choice"]))), 0.0
        )
        self.assertGreater(
            float(np.max(np.abs(remembered["profile_memory_outside"]))), 0.0
        )

    def test_accounting_identities(self):
        engine = MarketEngine(EngineConfig(enabled=True, fleet_per_base_request=0.03))
        model, profiles, trips, anchor, policies = _engine_inputs()
        result = _step(engine, FleetState(), model, profiles, trips, anchor, policies, p1_mult=0.9)
        n = len(profiles)
        no_ride = float(np.sum(result.rider["p_no_ride"]))
        m1, m2 = result.metrics["Firm1"], result.metrics["Firm2"]
        self.assertAlmostEqual(no_ride + m1["chosen"] + m2["chosen"], n, places=8)
        for m in (m1, m2):
            self.assertLessEqual(m["completed"], m["dispatch_offers"] + 1e-9)
            self.assertAlmostEqual(m["unfulfilled"], m["dispatch_offers"] - m["completed"], places=9)
            self.assertAlmostEqual(m["acceptance_declines"] + m["capacity_shortfall"], m["unfulfilled"], places=9)
        # Each rider is served at most once across both platforms.
        served = result.rider["served_firm1"] + result.rider["served_firm2"]
        self.assertTrue(np.all(served <= 1.0 + 1e-9))
        self.assertTrue(np.all(served <= 1.0 - result.rider["p_no_ride"] + 1e-9))
        # Spillover into Firm2 equals Firm1 attempts' complement accounting.
        self.assertAlmostEqual(m1["spillover_attempts"] + m1["chosen"], m1["dispatch_offers"], places=9)
        welfare = result.welfare_by_income
        self.assertAlmostEqual(sum(v[0] for v in welfare.values()), n, places=9)
        self.assertAlmostEqual(sum(v[1] for v in welfare.values()), m1["completed"] + m2["completed"], places=8)
        self.assertAlmostEqual(sum(v[2] for v in welfare.values()), m1["revenue"] + m2["revenue"], places=6)

    def test_units_invariant_to_realized_scaling(self):
        """Doubling requests and the fleet together leaves per-rider outcomes unchanged."""
        model, profiles, trips, anchor, policies = _engine_inputs(seed=4)
        engine = MarketEngine(EngineConfig(enabled=True, fleet_per_base_request=0.05))
        a = _step(engine, FleetState(), model, profiles, trips, anchor, policies, base=100, realized=130)
        b = _step(engine, FleetState(), model, profiles, trips, anchor, policies, base=200, realized=260)
        for f in ("Firm1", "Firm2"):
            for key in ("completed", "revenue", "profit"):
                self.assertAlmostEqual(a.metrics[f][key], b.metrics[f][key], places=6)
            self.assertAlmostEqual(a.fleet[f]["utilization"], b.fleet[f]["utilization"], places=6)

    def test_rider_weights_equal_duplication(self):
        model, profiles, trips, anchor, policies = _engine_inputs(seed=10, n=40)
        engine = MarketEngine(EngineConfig(enabled=True, fleet_per_base_request=0.02))
        w = np.where(trips["airport"], 2.0, 1.0)
        a = _step(engine, FleetState(), model, profiles, trips, anchor, policies, realized=90.0)
        weighted = engine.step(
            model=model, profiles=profiles, trips=trips, hour=13, day_of_week=2, weather="clear",
            price={"Firm1": anchor, "Firm2": anchor}, pay_basis={"Firm1": anchor, "Firm2": anchor},
            pay_policies=policies, fleet=FleetState(),
            online=engine.online_drivers(FleetState(), base_requests=100.0, supply_multiplier=1.0, weather="clear",
                                         shock=ShockEffect(), reservation_wage=24.0),
            block_minutes=240.0, realized_requests=90.0, shock=ShockEffect(), reservation_wage=24.0,
            operating_cost_per_mile=0.35, acceptance_threshold=1.0, platform_variable_cost=0.45,
            spillover=True, rider_weights=w,
        )
        idx = np.concatenate([np.arange(40), np.flatnonzero(trips["airport"])])
        dup_profiles = [profiles[i] for i in idx]
        dup_trips = {k: np.asarray(v)[idx] for k, v in trips.items()}
        dup = engine.step(
            model=model, profiles=dup_profiles, trips=dup_trips, hour=13, day_of_week=2, weather="clear",
            price={"Firm1": anchor[idx], "Firm2": anchor[idx]}, pay_basis={"Firm1": anchor[idx], "Firm2": anchor[idx]},
            pay_policies=policies, fleet=FleetState(),
            online=engine.online_drivers(FleetState(), base_requests=100.0, supply_multiplier=1.0, weather="clear",
                                         shock=ShockEffect(), reservation_wage=24.0),
            block_minutes=240.0, realized_requests=90.0, shock=ShockEffect(), reservation_wage=24.0,
            operating_cost_per_mile=0.35, acceptance_threshold=1.0, platform_variable_cost=0.45,
            spillover=True,
        )
        scale = 40.0 / len(idx)
        for f in ("Firm1", "Firm2"):
            for key in ("chosen", "completed", "revenue", "profit", "spillover_attempts"):
                self.assertAlmostEqual(weighted.metrics[f][key], dup.metrics[f][key] * scale, places=8)
            self.assertAlmostEqual(weighted.fleet[f]["utilization"], dup.fleet[f]["utilization"], places=8)
        self.assertNotAlmostEqual(weighted.metrics["Firm1"]["revenue"], a.metrics["Firm1"]["revenue"], places=3)

    def test_scarcity_and_spillover(self):
        model, profiles, trips, anchor, policies = _engine_inputs(seed=5)
        tight = MarketEngine(EngineConfig(enabled=True, fleet_per_base_request=0.01))
        loose = MarketEngine(EngineConfig(enabled=True, fleet_per_base_request=0.5))
        rt = _step(tight, FleetState(), model, profiles, trips, anchor, policies)
        rl = _step(loose, FleetState(), model, profiles, trips, anchor, policies)
        self.assertLess(rt.metrics["Firm1"]["completed"], rl.metrics["Firm1"]["completed"])
        self.assertGreater(rt.fleet["Firm1"]["pickup_minutes"], rl.fleet["Firm1"]["pickup_minutes"])
        # An outage at Firm2 pushes spillover into Firm1.
        outage = ShockEffect(firm_capacity_multiplier={"Firm1": 1.0, "Firm2": 0.05}, active=("platform_outage",))
        ro = _step(tight, FleetState(), model, profiles, trips, anchor, policies, shock=outage)
        self.assertGreater(ro.metrics["Firm1"]["spillover_attempts"], rt.metrics["Firm1"]["spillover_attempts"])
        no_spill = _step(tight, FleetState(), model, profiles, trips, anchor, policies, shock=outage, spill=False)
        self.assertEqual(no_spill.metrics["Firm1"]["spillover_attempts"], 0.0)

    def test_price_response(self):
        model, profiles, trips, anchor, policies = _engine_inputs(seed=6)
        engine = MarketEngine(EngineConfig(enabled=True, fleet_per_base_request=0.3))
        base = _step(engine, FleetState(), model, profiles, trips, anchor, policies)
        cut = _step(engine, FleetState(), model, profiles, trips, anchor, policies, p1_mult=0.85)
        self.assertGreater(cut.metrics["Firm1"]["chosen"], base.metrics["Firm1"]["chosen"])
        self.assertLess(cut.metrics["Firm2"]["chosen"], base.metrics["Firm2"]["chosen"])

    def test_storm_shock_channels(self):
        model, profiles, trips, anchor, policies = _engine_inputs(seed=7)
        engine = MarketEngine(EngineConfig(enabled=True, fleet_per_base_request=0.3))
        base = _step(engine, FleetState(), model, profiles, trips, anchor, policies)
        no_transit = ShockEffect(transit_availability=0.0, active=("transit_disruption",))
        storm = _step(engine, FleetState(), model, profiles, trips, anchor, policies, shock=no_transit)
        self.assertLess(np.sum(storm.rider["p_no_ride"]), np.sum(base.rider["p_no_ride"]))
        fewer = ShockEffect(fleet_multiplier=0.5, active=("driver_shortage",))
        on_base = engine.online_drivers(FleetState(), base_requests=100, supply_multiplier=1.0,
                                        weather="clear", shock=ShockEffect(), reservation_wage=24.0)
        on_short = engine.online_drivers(FleetState(), base_requests=100, supply_multiplier=1.0,
                                         weather="clear", shock=fewer, reservation_wage=24.0)
        self.assertAlmostEqual(on_short.total, 0.5 * on_base.total, places=9)

    def test_fleet_dynamics_lagged_and_multihoming(self):
        model, profiles, trips, anchor, policies = _engine_inputs(seed=8)
        cfg = EngineConfig(enabled=True, fleet_per_base_request=0.05, reference_earnings_per_hour=20.0)
        engine = MarketEngine(cfg)
        fleet = FleetState()
        policies = {"Firm1": DriverPayPolicy(fare_share=0.85), "Firm2": DriverPayPolicy(fare_share=0.60)}
        dedicated = []
        for _ in range(10):
            online = engine.online_drivers(fleet, base_requests=100.0, supply_multiplier=1.0, weather="clear",
                                           shock=ShockEffect(), reservation_wage=24.0)
            dedicated.append(online.dedicated["Firm1"] / online.dedicated["Firm2"])
            _step(engine, fleet, model, profiles, trips, anchor, policies)
        # Multi-homers lean to the better-paying app at once ...
        self.assertGreater(fleet.shared_firm1_share, 0.5)
        # ... while single-homers respond to their platform's earnings with a lag.
        self.assertAlmostEqual(dedicated[0], 1.0, places=12)
        self.assertGreater(dedicated[-1], dedicated[1])
        self.assertGreater(fleet.expected_earnings_firm["Firm1"], fleet.expected_earnings_firm["Firm2"])
        frozen = FleetState(frozen=True)
        _step(engine, frozen, model, profiles, trips, anchor, policies)
        self.assertFalse(np.isfinite(frozen.expected_earnings))

    def test_pricing_out_cedes_shared_drivers(self):
        """A platform that prices itself out of demand loses multi-homer time."""
        model, profiles, trips, anchor, policies = _engine_inputs(seed=11)
        engine = MarketEngine(EngineConfig(enabled=True, fleet_per_base_request=0.03, allocation_sensitivity=0.0))
        even = _step(engine, FleetState(), model, profiles, trips, anchor, policies)
        high = _step(engine, FleetState(), model, profiles, trips, anchor, policies, p1_mult=1.6)
        self.assertAlmostEqual(even.fleet["Firm1"]["multihoming_time_share"], 0.5, delta=0.1)
        self.assertLess(high.fleet["Firm1"]["multihoming_time_share"], 0.3)
        self.assertGreater(high.fleet["Firm2"]["driver_minutes"], even.fleet["Firm2"]["driver_minutes"])
        no_mh = MarketEngine(EngineConfig(enabled=True, fleet_per_base_request=0.03, multihoming_share=0.0))
        rigid = _step(no_mh, FleetState(), model, profiles, trips, anchor, policies, p1_mult=1.6)
        # Without shared drivers the rival is short of capacity and Firm 1 harvests more overflow.
        self.assertGreater(rigid.metrics["Firm1"]["spillover_completed"], high.metrics["Firm1"]["spillover_completed"])

    def test_nested_logit_runs_and_responds(self):
        model, profiles, trips, anchor, policies = _engine_inputs(seed=9)
        engine = MarketEngine(EngineConfig(enabled=True, demand_model="nested_logit", fleet_per_base_request=0.3))
        base = _step(engine, FleetState(), model, profiles, trips, anchor, policies)
        cut = _step(engine, FleetState(), model, profiles, trips, anchor, policies, p1_mult=0.85)
        self.assertGreater(cut.metrics["Firm1"]["chosen"], base.metrics["Firm1"]["chosen"])
        feats = profile_features(profiles)
        p0, pf1, _ = nested_logit_probabilities(
            model, feats, trips, hour=12, day_of_week=2, weather="clear", price1=anchor, price2=anchor,
            wait1=np.full(len(profiles), 4.0), wait2=np.full(len(profiles), 4.0),
            cancel1=np.zeros(len(profiles)), cancel2=np.zeros(len(profiles)), shock=ShockEffect(),
        )
        self.assertTrue(np.all((p0 >= 0) & (p0 <= 0.98)))
        # Symmetric prices and no loyalty -> equal split.
        mask = feats["loyalty_sign"] == 0
        np.testing.assert_allclose(pf1[mask], 0.5, atol=1e-12)

    def test_deepcopy_engine_and_fleet(self):
        engine = MarketEngine(EngineConfig(enabled=True))
        fleet = FleetState()
        clone = copy.deepcopy((engine, fleet))
        self.assertIsNot(clone[1], fleet)


class ShockProcessTest(unittest.TestCase):
    def test_holiday_wave_and_public_features(self):
        cfg = ShockConfig(enabled=True, types=("holiday_wave",))
        proc = ShockProcess(cfg, seed=1, steps_per_day=6)
        thanksgiving = holiday_dates(2023)["thanksgiving"]
        proc.reset(start_date=thanksgiving - dt.timedelta(days=4))
        active_days = []
        for day in range(9):
            proc.advance_day()
            effect = proc.effect(2)
            self.assertEqual(set(effect.public), set(PUBLIC_SHOCK_FEATURE_KEYS))
            if "holiday_wave" in effect.active:
                active_days.append(day)
                self.assertGreater(effect.airport_share_multiplier, 1.0)
        # 2 days before through 3 days after the holiday: 6 consecutive days.
        self.assertEqual(active_days, [2, 3, 4, 5, 6, 7])

    def test_same_seed_same_path_and_crn(self):
        cfg = ShockConfig(enabled=True, mode="stress")
        a = ShockProcess(cfg, seed=5)
        b = ShockProcess(cfg, seed=5)
        a.reset(start_date=dt.date(2023, 1, 10))
        b.reset(start_date=dt.date(2023, 1, 10))
        for _ in range(60):
            a.advance_day()
            b.advance_day()
            for block in range(6):
                self.assertEqual(a.effect(block).as_dict(), b.effect(block).as_dict())
        c = copy.deepcopy(a)
        a.advance_day()
        c.advance_day()
        self.assertEqual(a.effect(0).as_dict(), c.effect(0).as_dict())

    def test_storm_season_and_backlog(self):
        cfg = ShockConfig(enabled=True, types=("winter_storm",), storm_daily_hazard={1: 0.5})
        proc = ShockProcess(cfg, seed=3)
        proc.reset(start_date=dt.date(2023, 1, 1))
        storms = 0
        for _ in range(31):
            proc.advance_day()
            effect = proc.effect(0)
            if "winter_storm" in effect.active:
                storms += 1
                self.assertLess(effect.transit_availability, 1.0)
                self.assertEqual(effect.weather_override, "snow")
                self.assertTrue(effect.emergency)
        self.assertGreater(storms, 3)
        proc.defer(120.0)
        self.assertEqual(proc.release_backlog(storm_active=True), 0.0)
        released = sum(proc.release_backlog(storm_active=False) for _ in range(40))
        self.assertAlmostEqual(released, 120.0 * cfg.storm_deferred_share, places=9)

    def test_scripted_scenario_survives_reset(self):
        from market_shocks import ShockEvent, ShockScenario
        cfg = ShockConfig(enabled=True, types=("winter_storm",), storm_daily_hazard={})
        proc = ShockProcess(cfg, seed=4)
        proc.scenario = ShockScenario(
            start_date=dt.date(2023, 2, 6),
            events=[ShockEvent("winter_storm", 2, 3, detail={"transit_suspended": True})],
            forecasts={2: True},
        )
        for _ in range(2):
            proc.reset(training_opening=True)
            self.assertEqual(proc.start_date, dt.date(2023, 2, 6))
            seen = []
            for day in range(5):
                proc.advance_day()
                effect = proc.effect(1)
                seen.append(("winter_storm" in effect.active, effect.public["storm_warning"],
                             proc.day_weather_override(), effect.transit_availability))
            self.assertEqual([s[0] for s in seen], [False, False, True, True, False])
            self.assertEqual(seen[1][1], 1.0)  # warned the day before onset
            self.assertEqual(seen[2][2], "snow")
            self.assertEqual(seen[2][3], 0.0)  # transit suspended during the storm
            self.assertEqual(seen[4][3], 1.0)

    def test_intensity_zero_is_neutral(self):
        cfg = ShockConfig(enabled=True, mode="stress", intensity_scale=0.0)
        proc = ShockProcess(cfg, seed=2)
        proc.reset(start_date=dt.date(2023, 11, 15))
        for _ in range(30):
            proc.advance_day()
            effect = proc.effect(3)
            self.assertAlmostEqual(effect.volume_multiplier, 1.0, places=12)
            self.assertAlmostEqual(effect.fleet_multiplier, 1.0, places=12)
            self.assertAlmostEqual(effect.transit_availability, 1.0, places=12)
            self.assertAlmostEqual(effect.firm2_price_multiplier, 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
