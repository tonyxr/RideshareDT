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
    ObservationConfig,
    PlatformObservationModel,
    PositiveBusinessReward,
    PositiveRewardConfig,
    SoftConstraintController,
)


class CorePPOFixRegressionTests(unittest.TestCase):
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
