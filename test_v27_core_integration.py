"""v27 engine and shocks wired into Core (simulator only, no learning)."""

from __future__ import annotations

import copy
import csv
import datetime as dt
import os
import tempfile
import unittest

import numpy as np

from city_market_calibration import market_dynamics_calibration
from market_dynamics import MarketDynamicsConfig
from market_engine import EngineConfig, FleetBatchState
from market_shocks import PUBLIC_SHOCK_FEATURE_KEYS, ShockConfig
from Market_models import CoefficientOverrides
from test_v26_market_and_evaluation import _write_trip_bank


def _calibration(tmp: str) -> dict:
    bank = os.path.join(tmp, "trip_bank.csv")
    _write_trip_bank(bank, rows=3000)
    with open(bank, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    weather = {"clear": 0.8, "rain": 0.15, "snow": 0.05}
    v26 = market_dynamics_calibration(
        rows, weather_probs=weather, duration_identified=True, service_identified=False,
        airport_identified=True, outside_option_base_fare=2.5,
    )
    hour_probs = {str(h): (2.0 if 7 <= h < 10 or 16 <= h < 20 else 1.0) for h in range(24)}
    return {
        "market": {"hour_probs": hour_probs, "weather_probs": weather,
                   "weather_demand_multiplier": v26["weather_demand_multiplier"]},
        "tariff": v26["tariff"], "choice": v26["choice"], "supply": v26["supply"],
        "trip_bank": {"path": "trip_bank.csv", "airport_identified": True, "service_identified": False},
        "_source_dir": tmp,
    }


def _core(*, shocks: ShockConfig | None = None, observations: bool = False, seed: int = 41,
          engine: EngineConfig | None = None, privileged_critic: bool = False):
    from Core import Core
    tmp = tempfile.mkdtemp()
    core = Core(
        market_name="Chicago", seed=seed, firm1_mode="RL" if privileged_critic else "static",
        firm2_mode="static", total_customers_pool=300, openai_api_key="",
        shock_observations=observations, credit_assignment_v25=privileged_critic,
    )
    core.apply_calibration(_calibration(tmp))
    core.configure_v26(market_dynamics=MarketDynamicsConfig(enabled=True, capacity_target_fulfillment=0.0),
                       steps_per_day=6)
    core.configure_v27(engine_config=engine or EngineConfig(enabled=True),
                       shock_config=shocks or ShockConfig(enabled=False), calibration_sample_size=120)
    core.firm1.overrides = CoefficientOverrides()
    core.firm2.overrides = CoefficientOverrides()
    return core


def _run_day(core, base=80):
    ctx = core.market.sample_day_context()
    out = []
    for _ in range(6):
        hour = core.market.sample_timestep_hour().hour
        n = core._step_customers(base, hour=hour, day_of_week=ctx.day_of_week, weather=ctx.weather)
        profiles = core._sample_profiles_from_pool(min(n, 150))
        result = core.simulate_batch(ctx.day_of_week, ctx.weather, hour, n, profiles, False, base)
        out.append((ctx, hour, n, result, dict(core.last_crowd_response_stats)))
    return out


def _snapshot(core):
    return {
        "market": copy.deepcopy(core.market),
        "driver_supply": copy.deepcopy(core.driver_supply),
        "choice_model": copy.deepcopy(core.choice_model),
        "rng": copy.deepcopy(core.rng.bit_generator.state),
        "crn": (core._v26_crn_base, core._v26_crn_step),
        "firm1": copy.deepcopy(core.firm1.overrides),
    }


def _restore(core, snap):
    core.market = copy.deepcopy(snap["market"])
    core.driver_supply = copy.deepcopy(snap["driver_supply"])
    core.choice_model = copy.deepcopy(snap["choice_model"])
    core.rng.bit_generator.state = copy.deepcopy(snap["rng"])
    core._v26_crn_base, core._v26_crn_step = snap["crn"]
    core.firm1.overrides = copy.deepcopy(snap["firm1"])


class CoreV27Tests(unittest.TestCase):
    def test_calibration_hits_target_and_state_is_untouched(self):
        core = _core(engine=EngineConfig(enabled=True, target_utilization=0.55))
        cal = core.v27_settings["fleet_calibration"]
        self.assertAlmostEqual(cal["anchor_utilization"], 0.55, places=3)
        self.assertGreater(cal["fleet_per_base_request"], 0.0)
        self.assertGreater(core.v27_engine.config.reference_earnings_per_hour, 0.0)
        self.assertEqual(core.market.calendar.day_index, -1)
        self.assertFalse(np.isfinite(core.driver_supply.v27_fleet.expected_earnings))
        self.assertIn("v27", core.v26_settings)

    def test_simulate_batch_contract(self):
        core = _core()
        for ctx, hour, n, (rows, m1, m2, gap, air, dist), crowd in _run_day(core):
            self.assertEqual(rows, [])
            self.assertEqual(m1.total, min(n, 150))
            self.assertLessEqual(m1.completed, m1.dispatch_offers + 1e-9)
            self.assertLessEqual(m1.completed_share + m2.completed_share, 1.0 + 1e-9)
            no_ride = crowd["no_ride_rate"]
            self.assertAlmostEqual(no_ride + m1.chosen_share + m2.chosen_share, 1.0, places=8)
            self.assertGreater(m1.reference_total, 0)
            self.assertIsInstance(core.driver_supply.smoothed_states["Firm1"], FleetBatchState)
            features = core.driver_supply.state_features_for_firm("Firm1", opponent="Firm2")
            self.assertTrue(np.all(np.isfinite(features)))
            self.assertIn("by_income", core.last_welfare_stats)
        # Expected flows: identical inputs give identical outputs (no choice noise).
        profiles = core._sample_profiles_from_pool(100)
        snapshot = _snapshot(core)
        a = core.simulate_batch(2, "clear", 12, 100, profiles, False, 80)
        _restore(core, snapshot)
        b = core.simulate_batch(2, "clear", 12, 100, profiles, False, 80)
        self.assertEqual(a[1].profit, b[1].profit)
        self.assertEqual(a[2].completed, b[2].completed)

    def test_rows_and_price_response(self):
        core = _core()
        profiles = core._sample_profiles_from_pool(60)
        snapshot = _snapshot(core)
        rows, base1, _, _, _, _ = core.simulate_batch(2, "clear", 12, 60, profiles, True, 60)
        self.assertEqual(len(rows), 60)
        self.assertTrue({"P_NoRide", "P_Firm1", "ExpectedServedFirm1", "Choice"} <= set(rows[0]))
        _restore(core, snapshot)
        for key in core.shared_edit_keys:
            setattr(core.firm1.overrides, key, 0.85 * float(getattr(core.market.curr_market, key)))
        _, cut1, _, gap, _, _ = core.simulate_batch(2, "clear", 12, 60, profiles, False, 60)
        self.assertGreater(cut1.chosen_share, base1.chosen_share)
        self.assertGreater(gap, 0.0)

    def test_shock_paths_are_common_across_rebuilds(self):
        shocks = ShockConfig(enabled=True, mode="stress")
        core = _core(shocks=shocks)
        a = core._rebuild_market("Chicago", 99)
        b = core._rebuild_market("Chicago", 99)
        for _ in range(40):
            ca, cb = a.sample_day_context(), b.sample_day_context()
            self.assertEqual((ca.day_of_week, ca.weather), (cb.day_of_week, cb.weather))
            self.assertEqual(a.shocks.current_date, b.shocks.current_date)
            self.assertEqual(ca.day_of_week, a.shocks.current_date.weekday())
            for hour in (3, 9, 17):
                self.assertEqual(a.shock_effect(hour).as_dict(), b.shock_effect(hour).as_dict())
            for _ in range(6):
                a.sample_timestep_hour()
                b.sample_timestep_hour()
        c = core._rebuild_market("Chicago", 100)
        c.sample_day_context()
        self.assertNotEqual(c.shocks.start_date, None)

    def test_storm_defers_and_releases_requests(self):
        shocks = ShockConfig(enabled=True, types=("winter_storm",), storm_daily_hazard={m: 1.0 for m in range(1, 13)},
                             storm_duration_days=(1, 1), storm_deferred_share=1.0, storm_rebound_days=1)
        core = _core(shocks=shocks)
        core.market.shocks.reset(start_date=dt.date(2023, 1, 9))
        core.market.shocks.reset = lambda **kwargs: None  # keep the scripted opening
        core.market.calendar.day_index = -1
        ctx = core.market.sample_day_context()
        self.assertEqual(ctx.weather, "clear" if "winter_storm" not in core.market.shock_effect(0).active else "snow")
        # With hazard 1 every day after the first is a storm day.
        ctx = core.market.sample_day_context()
        self.assertEqual(ctx.weather, "snow")
        effect = core.market.shock_effect(12)
        self.assertIn("winter_storm", effect.active)
        before = core.market.shocks.backlog
        core._step_customers(100, hour=12, day_of_week=ctx.day_of_week, weather=ctx.weather)
        self.assertGreater(core.market.shocks.backlog, before)

    def test_public_shock_features_in_observation_and_critic(self):
        shocks = ShockConfig(enabled=True, types=("holiday_wave", "winter_storm"))
        core = _core(shocks=shocks, observations=True)
        self.assertEqual(core.platform_observation_dim, 24 + len(PUBLIC_SHOCK_FEATURE_KEYS))
        core.market.shocks.reset(start_date=dt.date(2023, 11, 21))
        core.market.shocks.reset = lambda **kwargs: None
        core.market.calendar.day_index = -1
        ctx = core.market.sample_day_context()
        state = core._build_rl_state(ctx.day_of_week, 12, ctx.weather)
        self.assertEqual(state.size, core.platform_observation_dim)
        block = dict(zip(PUBLIC_SHOCK_FEATURE_KEYS, state[24:]))
        self.assertEqual(block["holiday_active"], 1.0)  # Thanksgiving 2023 is Nov 23; wave starts Nov 21
        self.assertGreater(block["holiday_proximity"], 0.5)
        core.shock_config.public_features = False
        hidden = core._build_rl_state(ctx.day_of_week, 12, ctx.weather)
        np.testing.assert_allclose(hidden[24:], [0, 0, 0, 0, 1, 0])
        np.testing.assert_allclose(hidden[:24], state[:24])

    def test_privileged_critic_width_with_shock_observations(self):
        core = _core(shocks=ShockConfig(enabled=True), observations=True, privileged_critic=True)
        ctx = core.market.sample_day_context()
        public = core._build_rl_state(ctx.day_of_week, 12, ctx.weather)
        critic = core._build_privileged_critic_state(public)
        self.assertEqual(critic.size, int(core.firm1.agent.critic_state_dim))
        self.assertEqual(int(core.credit_assignment_config["privileged_critic_state_dim"]), critic.size)

    def test_reseed_keeps_v26_streams_and_reseeds_shocks(self):
        core = _core(shocks=ShockConfig(enabled=True, mode="stress"))
        core._reseed_environment(5)
        market_state = core.market.rng.bit_generator.state
        shock_draw = core.market.shocks.rng.random()
        core._reseed_environment(5)
        self.assertEqual(core.market.rng.bit_generator.state, market_state)
        self.assertEqual(core.market.shocks.rng.random(), shock_draw)
        sequence = np.random.SeedSequence([int(core.seed), 5, 26])
        legacy = np.random.default_rng(sequence.spawn(6)[0])
        self.assertEqual(np.random.default_rng(np.random.SeedSequence([int(core.seed), 5, 26]).spawn(7)[0]).random(),
                         legacy.random())

    def test_nested_logit_option(self):
        core = _core(engine=EngineConfig(enabled=True, demand_model="nested_logit"))
        results = _run_day(core)
        self.assertTrue(all(0.0 < r[3][1].completed_share < 1.0 for r in results))
        self.assertTrue(all(abs(r[4]["no_ride_rate"] + r[3][1].chosen_share + r[3][2].chosen_share - 1.0) < 1e-8
                            for r in results))

    def test_v27_requires_v26(self):
        from Core import Core
        core = Core(market_name="Chicago", seed=3, firm1_mode="static", firm2_mode="static",
                    total_customers_pool=50, openai_api_key="")
        with self.assertRaises(ValueError):
            core.configure_v27(engine_config=EngineConfig(enabled=True))


if __name__ == "__main__":
    unittest.main()
