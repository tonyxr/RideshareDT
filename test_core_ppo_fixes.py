import unittest
from collections import deque
from types import SimpleNamespace

import numpy as np

from Core import Core
from GenerateAgent import GenerateAgent
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
    def test_generalization_profiles_bypass_cache_and_cover_all_markets(self):
        core = Core.__new__(Core)
        core.seed = 773
        core.rng = np.random.default_rng(core.seed)
        core.total_customers_pool = 64
        core.training_generalization_fraction = 0.25
        core.evaluation_generalization_fraction = 1.0
        cached_profile = {
            "ProfileId": "cached-only",
            "PriceThreshold": 1.25,
            "PriceThresholdSource": "cached",
        }
        core.synthetic_profile_pool = [cached_profile]
        core.generalization_profile_pool = []
        core.generalization_profiles_by_city = {}

        core._build_generalization_profile_pool(pool_size=80)

        self.assertEqual(
            set(core.generalization_profiles_by_city),
            set(GenerateAgent.CITY_DEMOGRAPHICS),
        )
        self.assertTrue(all(
            row["PriceThresholdSource"] == "generalization_fallback"
            for row in core.generalization_profile_pool
        ))
        thresholds = np.asarray([
            row["PriceThreshold"] for row in core.generalization_profile_pool
        ])
        self.assertGreater(float(np.std(thresholds)), 0.05)

        holdout = core._sample_profiles_from_pool(
            16,
            generalization_fraction=1.0,
            generalization_city="Chicago",
        )
        self.assertTrue(all(row["ProfileMarket"] == "Chicago" for row in holdout))
        self.assertTrue(all(row is not cached_profile for row in holdout))

        cached_only = core._sample_profiles_from_pool(
            4, generalization_fraction=0.0
        )
        self.assertTrue(all(row is cached_profile for row in cached_only))

    def test_competitive_backtest_requires_dynamic_late_dominance(self):
        core = Core.__new__(Core)
        core.firm1_mode = "RL"
        core.shared_edit_keys = ["base_fare"]
        core.firm1 = SimpleNamespace(
            action_steps=lambda action: (
                {}
                if int(action) == 0
                else {"base_fare": -1 if int(action) % 2 else 1}
            ),
            config=SimpleNamespace(step={"base_fare": 0.10}),
        )
        core.training_logs = [
            {
                "avg_reward": 0.20 + 0.005 * day,
                "validation_score": (
                    0.5 + 0.2 * (day // 25)
                    if day in {24, 49, 74, 99}
                    else np.nan
                ),
            }
            for day in range(100)
        ]
        core.evaluation_logs = [
            {
                "reward": 0.30 + 0.004 * day,
                "rl_completed_share": 0.72,
                "heuristic_completed_share": 0.20,
                "rl_profit": 3.0,
                "heuristic_profit": 0.8,
                "firm1_base_fare": 2.5 + 0.1 * (day % 2),
                "action": 1 + (day % 2),
            }
            for day in range(100)
        ]

        report = core._competitive_durability_backtest()

        self.assertTrue(report["passed"])
        self.assertGreater(report["late_completed_share_advantage"], 0.10)
        self.assertGreater(report["late_profit_advantage_per_request"], 0.0)
        self.assertGreater(report["held_out_validation_score_fitted_change"], 0.0)
        self.assertGreaterEqual(report["late_evaluation_reward_retention"], 0.75)
        self.assertGreater(report["late_coefficient_change_rate"], 0.02)

    def test_stable_dominant_policy_can_converge_without_action_churn(self):
        core = Core.__new__(Core)
        core.firm1_mode = "RL"
        core.shared_edit_keys = ["base_fare"]
        core.firm1 = SimpleNamespace(
            action_steps=lambda action: {},
            config=SimpleNamespace(step={"base_fare": 0.10}),
        )
        core.training_logs = [
            {
                "avg_reward": 0.60,
                "validation_score": (
                    0.80
                    if day in {24, 49, 74, 99}
                    else np.nan
                ),
            }
            for day in range(100)
        ]
        core.evaluation_logs = [
            {
                "reward": 0.60,
                "rl_completed_share": 0.60,
                "heuristic_completed_share": 0.30,
                "rl_profit": 3.0,
                "heuristic_profit": 2.0,
                "firm1_base_fare": 2.5,
                "action": 0,
            }
            for _ in range(100)
        ]

        report = core._competitive_durability_backtest()

        self.assertTrue(report["passed"])
        self.assertEqual(report["late_fare_equivalent_change_rate"], 0.0)
        self.assertEqual(report["late_action_diversity"], 1)

    def test_competitiveness_backtest_uses_share_gap_not_profit_or_quote_gap(self):
        core = Core.__new__(Core)
        core.firm1_mode = "RL"
        core.shared_edit_keys = ["base_fare"]
        core.firm1 = SimpleNamespace(
            action_steps=lambda action: {"base_fare": -1 if int(action) % 2 else 1},
            config=SimpleNamespace(step={"base_fare": 0.10}),
        )
        core.long_term_profit_reward_config = LongTermProfitRewardConfig(
            objective_mode="competitiveness",
            own_profit_weight=0.0,
            profit_advantage_weight=0.0,
            market_share_competitiveness_weight=1.0,
            market_share_target_gap=0.10,
        )
        core.constraint_config = ConstraintConfig(
            target_gap=0.0,
            overall_tolerance=0.45,
        )
        core.training_logs = [
            {
                "avg_reward": 0.50,
                "validation_score": (
                    0.50 + 0.10 * (day // 25)
                    if day in {24, 49, 74, 99}
                    else np.nan
                ),
            }
            for day in range(100)
        ]
        core.evaluation_logs = [
            {
                "reward": 0.75,
                # Firm 1 wins customer choice while Firm 2 completes more
                # requests because of supply. Competitiveness must use the
                # former; completion coverage is a separate service measure.
                "rl_market_share": 0.60,
                "heuristic_market_share": 0.40,
                "rl_completed_share": 0.20,
                "heuristic_completed_share": 0.50,
                "rl_profit": 1.0,
                "heuristic_profit": 2.0,
                "rl_fulfillment_rate": 0.90,
                "price_gap_f2_minus_f1": 8.0,
                "firm1_base_fare": 2.5 + 0.1 * (day % 2),
                "action": 1 + (day % 2),
            }
            for day in range(100)
        ]

        report = core._competitive_durability_backtest()

        self.assertTrue(report["passed"])
        self.assertEqual(report["primary_objective"], "competitiveness")
        self.assertGreaterEqual(
            report["late_completed_share_advantage"], 0.10
        )

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

    def test_economic_action_group_ignores_negligible_airport_chatter(self):
        core = Core.__new__(Core)
        core.firm1_mode = "RL"
        core.shared_edit_keys = [
            "base_fare",
            "per_minute",
            "per_mile",
            "booking_fee",
            "airport_fee",
        ]
        core.firm1 = SimpleNamespace(
            action_steps=lambda action: {
                1: {"airport_fee": 1},
                2: {"base_fare": -1},
                3: {"base_fare": 1},
            }.get(int(action), {}),
            config=SimpleNamespace(
                step={
                    "base_fare": 0.10,
                    "per_minute": 0.01,
                    "per_mile": 0.05,
                    "booking_fee": 0.10,
                    "airport_fee": 0.10,
                }
            ),
        )

        self.assertEqual(core._economic_action_group(1), 0)
        self.assertEqual(core._economic_action_group(2), -1)
        self.assertEqual(core._economic_action_group(3), 1)

    def test_competitiveness_mask_blocks_further_discount_when_service_is_low(self):
        core = Core.__new__(Core)
        core.firm1_mode = "RL"
        core.firm1 = SimpleNamespace(
            action_steps=lambda action: {
                1: {"base_fare": -1},
                2: {"base_fare": 1},
            }.get(int(action), {}),
            config=SimpleNamespace(step={"base_fare": 0.10}),
        )
        core.long_term_profit_reward_config = LongTermProfitRewardConfig(
            objective_mode="competitiveness",
            own_profit_weight=0.0,
            profit_advantage_weight=0.0,
            market_share_competitiveness_weight=1.0,
        )
        core.constraint_config = ConstraintConfig(fulfillment_floor=0.78)

        masked = core._objective_action_mask(
            np.asarray([True, True, True]),
            fulfillment_rate=0.60,
        )

        np.testing.assert_array_equal(masked, [False, False, True])

    def test_profit_pipeline_keeps_normal_action_mask(self):
        core = Core.__new__(Core)
        core.firm1_mode = "RL"
        core.firm1 = SimpleNamespace(
            action_steps=lambda action: {
                1: {"base_fare": -1},
                2: {"base_fare": 1},
            }.get(int(action), {}),
            config=SimpleNamespace(step={"base_fare": 0.10}),
        )
        core.long_term_profit_reward_config = LongTermProfitRewardConfig(
            objective_mode="profit_maximization",
        )
        core.constraint_config = ConstraintConfig(fulfillment_floor=0.78)
        base = np.asarray([True, True, True])

        masked = core._objective_action_mask(
            base,
            fulfillment_rate=0.20,
        )

        np.testing.assert_array_equal(masked, base)

    def test_optimizer_and_backtest_use_identical_economic_groups(self):
        keys = ["base_fare", "per_minute", "per_mile", "booking_fee", "airport_fee"]
        pricer = FirmRLPricer(seed=5, opt_keys=keys)
        market = SimpleNamespace(curr_market=SimpleNamespace(
            base_fare=3.0,
            per_minute=0.30,
            per_mile=1.30,
            booking_fee=1.50,
            airport_fee=4.0,
        ))
        features = pricer.build_action_feature_matrix(market, {})
        weights = np.asarray([0.35, 0.35, 0.20, 0.10], dtype=float)
        feature_impacts = features[:, 10:14] @ weights
        normalized_fare_threshold = 0.05 / 20.0
        feature_groups = np.where(
            feature_impacts < -normalized_fare_threshold,
            -1,
            np.where(feature_impacts > normalized_fare_threshold, 1, 0),
        )
        core = Core.__new__(Core)
        core.firm1_mode = "RL"
        core.shared_edit_keys = keys
        core.firm1 = pricer

        audit_groups = np.asarray([
            core._economic_action_group(action)
            for action in range(len(pricer.action_to_steps))
        ])

        np.testing.assert_array_equal(feature_groups, audit_groups)

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

    def test_state_action_specialization_weight_is_configurable(self):
        pricer = FirmRLPricer(
            seed=31,
            opt_keys=["base_fare", "per_minute"],
            state_action_mi_weight=0.23,
        )

        self.assertAlmostEqual(pricer.agent.state_action_mi_coeff, 0.23)

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

    def test_competitiveness_diagnostics_separate_market_share_from_coverage(self):
        core = self._diagnostic_core()
        core.long_term_profit_reward_config = LongTermProfitRewardConfig(
            objective_mode="competitiveness",
            own_profit_weight=0.0,
            profit_advantage_weight=0.0,
            market_share_competitiveness_weight=1.0,
        )
        core.long_term_profit_reward_model = LongTermProfitReward(
            core.long_term_profit_reward_config
        )

        result = core._reward_diagnostics(
            share=0.47,
            baseline_share=0.50,
            completed_share=0.30,
            baseline_completed_share=0.25,
            rev_per_request=8.0,
            baseline_rev_per_request=7.0,
            profit_per_request=1.5,
            baseline_profit_per_request=1.0,
            mean_gap=1.0,
            prev_share=0.50,
            prev_rev_per_request=8.0,
            prev_profit_per_request=1.5,
            fulfillment_rate=0.80,
        )

        self.assertAlmostEqual(result["reward_market_share_sum"], 1.0)
        self.assertAlmostEqual(result["reward_outside_option_share"], 0.03)
        self.assertAlmostEqual(result["reward_completion_coverage"], 0.55)
        self.assertAlmostEqual(result["reward_market_share"], 0.47 / 0.97)
        self.assertAlmostEqual(result["reward"], (0.47 / 0.97) * 0.80)

    def test_checkpoint_score_selects_profit_not_proxy_kpis(self):
        core = self._diagnostic_core()
        objective = core.long_term_profit_reward_model.compute(
            own_profit_per_request=3.0,
            rival_profit_per_request=2.0,
            price_gap_f2_minus_f1=-20.0,
        )["reward"]
        first = core._validation_score_from_metrics(reward=objective)
        second = core._validation_score_from_metrics(reward=objective)

        self.assertAlmostEqual(first, objective)
        self.assertAlmostEqual(second, first)

    def test_checkpoint_score_honors_serviceable_market_share(self):
        core = self._diagnostic_core()
        core.long_term_profit_reward_config = LongTermProfitRewardConfig(
            objective_mode="competitiveness",
            own_profit_weight=0.0,
            profit_advantage_weight=0.0,
            market_share_competitiveness_weight=1.0,
        )
        core.long_term_profit_reward_model = LongTermProfitReward(
            core.long_term_profit_reward_config
        )
        matched_reward = core.long_term_profit_reward_model.compute(
            own_profit_per_request=1.0,
            rival_profit_per_request=1.0,
            own_market_share=0.60,
            rival_market_share=0.40,
            own_fulfillment_rate=0.90,
        )["reward"]
        distant_reward = core.long_term_profit_reward_model.compute(
            own_profit_per_request=1.0,
            rival_profit_per_request=1.0,
            own_market_share=0.45,
            rival_market_share=0.55,
            own_fulfillment_rate=0.90,
        )["reward"]
        matched = core._validation_score_from_metrics(
            reward=matched_reward,
            completed_share_advantage=0.20,
        )
        distant = core._validation_score_from_metrics(
            reward=distant_reward,
            completed_share_advantage=-0.05,
        )
        self.assertGreater(matched, distant)

    def test_competitiveness_checkpoint_score_cannot_be_bought_with_profit(self):
        core = self._diagnostic_core()
        core.long_term_profit_reward_config = LongTermProfitRewardConfig(
            objective_mode="competitiveness",
            own_profit_weight=0.0,
            profit_advantage_weight=0.0,
            market_share_competitiveness_weight=1.0,
        )
        core.long_term_profit_reward_model = LongTermProfitReward(
            core.long_term_profit_reward_config
        )
        high_profit_bad_share_reward = (
            core.long_term_profit_reward_model.compute(
                own_profit_per_request=100.0,
                rival_profit_per_request=0.0,
                own_market_share=0.45,
                rival_market_share=0.55,
                own_fulfillment_rate=0.90,
            )["reward"]
        )
        low_profit_good_share_reward = (
            core.long_term_profit_reward_model.compute(
                own_profit_per_request=-100.0,
                rival_profit_per_request=100.0,
                own_market_share=0.60,
                rival_market_share=0.40,
                own_fulfillment_rate=0.90,
            )["reward"]
        )
        high_profit_bad_share_gap = core._validation_score_from_metrics(
            reward=high_profit_bad_share_reward,
            completed_share_advantage=-0.10,
        )
        low_profit_good_share_gap = core._validation_score_from_metrics(
            reward=low_profit_good_share_reward,
            completed_share_advantage=0.20,
        )
        self.assertGreater(
            low_profit_good_share_gap, high_profit_bad_share_gap
        )

    def test_ineligible_checkpoint_never_replaces_deployable_best(self):
        self.assertFalse(Core._checkpoint_candidate_is_better(
            score=100.0,
            eligible=False,
            best_eligible_score=0.1,
        ))
        self.assertTrue(Core._checkpoint_candidate_is_better(
            score=0.2,
            eligible=True,
            best_eligible_score=0.1,
        ))

    def test_checkpoint_score_ignores_non_objective_diagnostics(self):
        core = self._diagnostic_core()
        clean = core._validation_score_from_metrics(reward=0.0)
        diagnosed = core._validation_score_from_metrics(reward=0.0)

        self.assertAlmostEqual(diagnosed, clean)

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
