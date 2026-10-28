"""Executable contracts for revision-10 temporal competitive reward."""

import unittest

import numpy as np

from Core import Core
from platform_mdp import LongTermProfitReward, LongTermProfitRewardConfig
from PPOAgent import PPOAgent


def temporal_config() -> LongTermProfitRewardConfig:
    return LongTermProfitRewardConfig(
        objective_mode="profit_maximization",
        stage_specific_rewards=True,
        learning_corrections=True,
        opponent_profit_lead_weight=0.60,
        temporal_competitive_reward=True,
        durable_dominance_weight=0.20,
        adaptive_residual_weight=0.10,
        durable_dominance_horizon_days=5.0,
        own_profit_scale=2.0,
        market_share_competitiveness_weight=0.05,
        market_share_gap_scale=0.10,
    )


class Revision10TemporalRewardTests(unittest.TestCase):
    def test_dynamic_weights_reserve_dominance_and_residual_mass(self):
        weights = temporal_config().economic_weights("dynamic")
        self.assertAlmostEqual(weights["own_profit"], 0.26)
        self.assertAlmostEqual(weights["profit_lead"], 0.39)
        self.assertAlmostEqual(weights["completed_share_lead"], 0.05)
        self.assertAlmostEqual(weights["durable_profit_lead"], 0.20)
        self.assertAlmostEqual(weights["adaptive_profit_residual"], 0.10)
        self.assertAlmostEqual(sum(weights.values()), 1.0)

    def test_market_pretraining_keeps_temporal_terms_off(self):
        weights = temporal_config().economic_weights("market")
        self.assertEqual(weights["durable_profit_lead"], 0.0)
        self.assertEqual(weights["adaptive_profit_residual"], 0.0)
        self.assertAlmostEqual(weights["own_profit"], 0.95)
        self.assertAlmostEqual(weights["completed_share_lead"], 0.05)

    def test_six_raw_outcomes_exactly_reconstruct_reward(self):
        config = temporal_config()
        inputs = dict(
            own_profit_per_request=4.0,
            rival_profit_per_request=3.0,
            own_completed_share=0.50,
            rival_completed_share=0.40,
            durable_profit_advantage_per_request=0.50,
            adaptive_profit_residual_per_request=-0.25,
        )
        reward = LongTermProfitReward(config).compute(
            training_regime="dynamic", **inputs
        )["reward"]
        components = np.asarray([4.0, 3.0, 0.50, 0.40, 0.50, -0.25])
        reconstructed = components @ np.asarray(
            config.critic_component_weights("dynamic")
        )
        self.assertAlmostEqual(reward, 0.8025)
        self.assertAlmostEqual(reconstructed, reward, places=12)

    def test_counterfactual_preview_does_not_advance_temporal_state(self):
        core = Core.__new__(Core)
        core.long_term_profit_reward_config = temporal_config()
        core.shared_steps_per_day = 6
        core._durable_profit_lead_ema = 0.0
        core._temporal_reward_state_initialized = False
        preview = core._temporal_competitive_reward_terms(
            current_profit_lead=1.0,
            previous_profit_lead=0.0,
            update=False,
        )
        self.assertFalse(core._temporal_reward_state_initialized)
        actual = core._temporal_competitive_reward_terms(
            current_profit_lead=1.0,
            previous_profit_lead=0.0,
            update=True,
        )
        self.assertEqual(preview, actual)
        self.assertTrue(core._temporal_reward_state_initialized)
        durable, residual = core._temporal_competitive_reward_terms(
            current_profit_lead=1.0,
            previous_profit_lead=1.0,
            update=True,
        )
        self.assertGreater(durable, actual[0])
        self.assertLess(residual, actual[1])

    def test_ppo_accepts_exact_six_component_transition(self):
        config = temporal_config()
        model = PPOAgent(
            6, 3, hidden_dim=16, single_state_dim=2, frame_stack=3,
            persistent_recurrence=True, privileged_critic=True,
            privileged_value_only=True, critic_state_dim=2, gamma=0.99,
            learning_corrections=True, td_value_targets=True,
            learner_revision=4, value_component_dim=6, device="cpu",
        )
        model.set_value_objective_weights(
            config.critic_component_weights("dynamic")
        )
        components = np.asarray(
            [1.35, 0.85, 0.57, 0.31, 0.42, -0.08], dtype=np.float32
        )
        reward = float(components @ model.value_weights.cpu().numpy())
        action, state, logp, value, *_ = model.act(
            np.zeros(6, dtype=np.float32),
            critic_state=np.zeros(2, dtype=np.float32),
        )
        model.store(
            state, action, reward, True, None, logp, value,
            duration=1, reward_components=components,
        )
        metrics = model.update(epochs=1, batch_size=1)
        self.assertGreater(metrics["optimizer_steps"], 0)
        for name in ("durable_profit_lead", "adaptive_profit_residual"):
            self.assertTrue(np.isfinite(metrics[f"critic_{name}_rmse"]))


if __name__ == "__main__":
    unittest.main()
