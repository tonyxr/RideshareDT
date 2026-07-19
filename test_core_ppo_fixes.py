import unittest
from collections import deque
from types import SimpleNamespace

import numpy as np

from Core import Core
from Market_models import CoefficientOverrides
from optim_config import default_specs_for
from pricing_models import ActionDescriptor, FirmRLPricer, build_discrete_action_space
from platform_mdp import (
    ActionStabilityTracker,
    ConstraintConfig,
    LongTermProfitReward,
    LongTermProfitRewardConfig,
    ObservationConfig,
    PlatformObservationModel,
    PositiveBusinessReward,
    PositiveRewardConfig,
    SoftConstraintController,
)


class CorePPOFixRegressionTests(unittest.TestCase):
    def test_action_space_contains_coordinated_and_rebalancing_moves(self):
        keys = ["base_fare", "per_minute", "per_mile", "booking_fee", "airport_fee"]
        action_map, _ = build_discrete_action_space(keys)
        bundles = [mapping for mapping in action_map.values() if len(mapping) > 1]

        self.assertGreaterEqual(len(bundles), 8)
        self.assertTrue(any(
            mapping.get("base_fare") == -1
            and mapping.get("booking_fee") == -1
            and mapping.get("per_mile") == 1
            for mapping in bundles
        ))
        self.assertTrue(any(
            {key for key, direction in mapping.items() if direction}
            == {"base_fare", "per_minute", "per_mile", "booking_fee"}
            for mapping in bundles
        ))

    def test_response_supervision_uses_causal_delta(self):
        baseline = np.array([0.2, -0.1, 0.5], dtype=np.float32)
        future = np.array([0.7, -0.4, 4.5], dtype=np.float32)

        target = Core._response_delta(baseline, future)

        self.assertTrue(np.allclose(target, np.array([0.5, -0.3, 2.0], dtype=np.float32)))

    def test_standalone_action_features_preserve_bundle_identity(self):
        keys = ["base_fare", "per_minute", "per_mile", "booking_fee", "airport_fee"]
        pricer = FirmRLPricer(seed=3, opt_keys=keys)
        market = SimpleNamespace(curr_market=SimpleNamespace(
            base_fare=3.0,
            per_minute=0.30,
            per_mile=1.30,
            booking_fee=1.50,
            airport_fee=4.0,
        ))

        features = pricer.build_action_feature_matrix(market, {})

        self.assertEqual(pricer.action_feature_dim, 20)
        self.assertEqual(features.shape, (len(pricer.action_to_steps), 20))
        self.assertTrue(np.any(np.count_nonzero(features[:, 2:7], axis=1) > 1))

    def test_pricer_execution_snapshot_is_transactional(self):
        keys = ["base_fare", "per_minute", "per_mile", "booking_fee", "airport_fee"]
        pricer = FirmRLPricer(seed=4, opt_keys=keys)
        pricer._last_applied_action = 7
        pricer._repeat_action_count = 3
        pricer._last_action_target = "base_fare+booking_fee"
        pricer.agent.action_ever_feasible[:] = True
        snapshot = pricer.snapshot_execution_state()

        pricer.reset_state_history()
        pricer.agent.action_ever_feasible[:] = False
        pricer.restore_execution_state(snapshot)

        self.assertEqual(pricer._last_applied_action, 7)
        self.assertEqual(pricer._repeat_action_count, 3)
        self.assertEqual(pricer._last_action_target, "base_fare+booking_fee")
        self.assertTrue(np.all(pricer.agent.action_ever_feasible))

    @staticmethod
    def _diagnostic_core(action_direction=0, zero_effect=False, saturated=False):
        core = Core.__new__(Core)
        core.reward_profit_scale = 10.0
        core.reward_rev_scale = 25.0
        core.reward_target_price_gap = 1.0
        core.observation_config = ObservationConfig(
            telemetry_delay_steps=0,
            quote_probe_interval_steps=1,
            quote_probe_delay_steps=0,
            quote_noise_dollars=0.0,
            quote_missing_probability=0.0,
        )
        core.constraint_config = ConstraintConfig(target_gap=1.0, overall_tolerance=0.4)
        core.positive_reward_model = PositiveBusinessReward(PositiveRewardConfig())
        core.long_term_profit_reward_config = LongTermProfitRewardConfig()
        core.long_term_profit_reward_model = LongTermProfitReward(
            core.long_term_profit_reward_config
        )
        core.soft_constraints = SoftConstraintController(core.constraint_config)
        core.action_stability = ActionStabilityTracker(core.constraint_config)
        observer = PlatformObservationModel(1, core.observation_config)
        for segment in observer.quote_probes:
            observer.quote_probes[segment] = {
                "gap": -1.0,
                "uncertainty": 0.0,
                "age": 0.0,
                "available": 1.0,
            }
        core.platform_observers = {"Firm1": observer}
        core.firm1 = SimpleNamespace(
            last_action_descriptor=ActionDescriptor(
                target="base_fare" if action_direction else "hold",
                direction=action_direction,
            ),
            last_action_was_zero_effect=zero_effect,
            last_action_was_saturated=saturated,
        )
        return core

    def test_reward_is_positive_and_action_identity_neutral(self):
        common = dict(
            share=0.25,
            completed_share=0.25,
            rev_per_request=8.0,
            mean_gap=-1.0,
            prev_share=0.25,
            prev_rev_per_request=8.0,
            prev_profit_per_request=1.0,
            prev_gap=-2.0,
            profit_per_request=1.0,
            profit_margin=0.05,
            fulfillment_rate=0.9,
        )
        hold = self._diagnostic_core(action_direction=0)._reward_diagnostics(**common)
        corrective = self._diagnostic_core(action_direction=-1)._reward_diagnostics(**common)
        self.assertGreater(hold["reward"], 0.0)
        self.assertAlmostEqual(corrective["reward"], hold["reward"], places=7)
        for key in (
            "reward_hold_inaction_penalty",
            "reward_corrective_action_bonus",
            "reward_overprice_penalty",
            "reward_underprice_penalty",
            "reward_action_change_penalty",
            "reward_action_realization_penalty",
        ):
            self.assertEqual(hold[key], 0.0)
            self.assertEqual(corrective[key], 0.0)
        self.assertGreater(hold["constraint_cost_gap_overprice"], 0.0)

        stronger = self._diagnostic_core(action_direction=0)._reward_diagnostics(
            **{**common, "rev_per_request": 16.0, "profit_per_request": 5.0, "completed_share": 0.40}
        )
        self.assertGreater(stronger["reward"], hold["reward"])

    def test_checkpoint_score_selects_profit_not_proxy_kpis(self):
        core = self._diagnostic_core()
        feasible = {name: 0.0 for name in core.soft_constraints.names}
        first = core._validation_score_from_metrics(
            reward=-1.0,
            share=0.05,
            revpr=100.0,
            profitpr=3.0,
            fulfillment=0.2,
            gap=-20.0,
            rival_profitpr=2.0,
            constraint_costs=feasible,
        )
        second = core._validation_score_from_metrics(
            reward=1.0,
            share=0.95,
            revpr=1.0,
            profitpr=3.0,
            fulfillment=1.0,
            gap=20.0,
            rival_profitpr=2.0,
            constraint_costs=feasible,
        )

        self.assertAlmostEqual(first, 3.1)
        self.assertAlmostEqual(second, first)

    def test_convergence_statistics_filter_stationary_market_noise(self):
        rewards = [0.5 + (0.10 if index % 2 else -0.10) for index in range(100)]

        count, reward_std, reward_delta = Core._smoothed_convergence_statistics(
            rewards,
            window=60,
            smoothing=20,
        )

        self.assertEqual(count, 60)
        self.assertLess(reward_std, 1e-12)
        self.assertLess(reward_delta, 1e-12)

    def test_bound_saturated_actions_are_marked_infeasible(self):
        pricer = FirmRLPricer.__new__(FirmRLPricer)
        pricer.opt_keys = ["base_fare"]
        pricer.config = default_specs_for(pricer.opt_keys)
        pricer.action_to_steps, pricer.action_keys = build_discrete_action_space(pricer.opt_keys)
        pricer.max_relative_dev = 0.60
        pricer.overrides = CoefficientOverrides()
        market = SimpleNamespace(
            curr_market=SimpleNamespace(base_fare=3.0)
        )
        lower_bound = max(
            pricer.config.bounds["base_fare"][0],
            market.curr_market.base_fare * (1.0 - pricer.max_relative_dev),
        )
        pricer.overrides.base_fare = float(lower_bound)

        mask = pricer.feasible_action_mask(market)

        self.assertTrue(mask[0])
        self.assertFalse(mask[1])  # base_fare -1
        self.assertTrue(mask[2])   # base_fare +1

    def test_preview_stack_does_not_advance_frame_history(self):
        pricer = FirmRLPricer.__new__(FirmRLPricer)
        pricer.single_state_dim = 2
        pricer.state_frame_stack = 3
        pricer._state_history = deque(maxlen=3)
        pricer.stack_state(np.array([1.0, 1.0], dtype=np.float32), commit=True)
        before = [frame.copy() for frame in pricer._state_history]

        preview = pricer.stack_state(np.array([2.0, 2.0], dtype=np.float32), commit=False)

        self.assertEqual(len(pricer._state_history), len(before))
        for current, expected in zip(pricer._state_history, before):
            self.assertTrue(np.array_equal(current, expected))
        self.assertTrue(np.array_equal(preview[-2:], np.array([2.0, 2.0], dtype=np.float32)))


if __name__ == "__main__":
    unittest.main()
