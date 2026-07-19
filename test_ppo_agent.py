import unittest

import numpy as np
import torch

from PPOAgent import PPOAgent


class PPOAgentRegressionTests(unittest.TestCase):
    def test_recurrent_belief_and_action_conditioned_response_are_live(self):
        torch.manual_seed(17)
        agent = PPOAgent(
            state_dim=6,
            single_state_dim=3,
            frame_stack=2,
            action_dim=3,
            action_feature_dim=4,
            response_dim=2,
            device="cpu",
        )
        state = torch.tensor([[0.1, 0.2, 0.3, 0.8, -0.4, 0.6]])
        reversed_state = torch.tensor([[0.8, -0.4, 0.6, 0.1, 0.2, 0.3]])
        action_features = torch.tensor([[
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.5, 0.0],
            [-1.0, 1.0, 0.5, 1.0],
        ]])

        logits, *_, response, q_values = agent.net(state, action_features)
        reversed_logits, *_ = agent.net(reversed_state, action_features)

        self.assertEqual(tuple(response.shape), (1, 3, 2))
        self.assertEqual(tuple(q_values.shape), (1, 3))
        self.assertFalse(torch.allclose(response[:, 0], response[:, 1]))
        self.assertFalse(torch.allclose(logits, reversed_logits))

    def test_collapse_rescue_survives_later_stage_controls(self):
        agent = PPOAgent(
            state_dim=3,
            action_dim=3,
            initial_exploration_rate=0.20,
            final_exploration_rate=0.01,
            exploration_rescue_rate=0.09,
            ent_coeff=0.02,
            min_ent_coeff=0.001,
            device="cpu",
        )
        agent.trigger_collapse_rescue(updates=3)
        agent.set_stage_controls(
            exploration_rate=0.01,
            entropy_scale=0.05,
            learning_rate_scale=0.5,
        )

        self.assertTrue(agent.rescue_active)
        self.assertGreaterEqual(agent.exploration_rate, 0.09)
        self.assertGreaterEqual(agent.ent_coeff, 0.01)

    def test_structured_update_trains_causal_response_and_action_heads(self):
        rng = np.random.default_rng(5)
        agent = PPOAgent(
            state_dim=8,
            single_state_dim=4,
            frame_stack=2,
            action_dim=4,
            action_feature_dim=5,
            response_dim=3,
            constraint_dim=2,
            min_action_visits=0,
            device="cpu",
        )
        for index in range(16):
            state_values = rng.normal(size=8).astype(np.float32)
            features = rng.normal(size=(4, 5)).astype(np.float32)
            action, state, old_logp, old_value, action_features, action_mask = agent.act(
                state_values,
                action_features=features,
                action_mask=np.ones(4, dtype=bool),
            )
            agent.store(
                state,
                action,
                float(np.sin(index / 3.0)),
                index == 15,
                None,
                old_logp,
                old_value,
                constraint_costs=np.array([0.02, 0.04], dtype=np.float32),
                risk_cost=0.03,
                response_target=np.array([0.1, -0.1, 0.05], dtype=np.float32),
                action_features=action_features,
                action_mask=action_mask,
            )

        metrics = agent.update(epochs=1, batch_size=8, bootstrap_value=0.0)

        self.assertTrue(metrics["update_performed"])
        self.assertTrue(np.isfinite(metrics["response_loss"]))
        self.assertTrue(np.isfinite(metrics["state_action_mi"]))
        self.assertIn("raw_policy_argmax_concentration", metrics)

    def test_exploration_never_assigns_probability_to_infeasible_actions(self):
        logits = torch.tensor([[0.2, 0.5, 4.0, -0.3]], dtype=torch.float32)
        mask = torch.tensor([[True, True, False, True]])

        dist = PPOAgent._exploratory_distribution(logits, 0.75, mask)

        self.assertEqual(float(dist.probs[0, 2].item()), 0.0)
        self.assertAlmostEqual(float(dist.probs.sum().item()), 1.0, places=6)

    def test_gae_bootstraps_continuing_rollout_but_respects_terminal(self):
        agent = PPOAgent(state_dim=2, action_dim=2, gamma=0.9, lam=1.0, device="cpu")
        rewards = torch.tensor([1.0, 1.0])
        values = torch.zeros(2)

        _, continuing_returns = agent._gae_scalar(
            rewards,
            values,
            torch.tensor([0.0, 0.0]),
            bootstrap_value=2.0,
            normalize_advantage=False,
        )
        _, terminal_returns = agent._gae_scalar(
            rewards,
            values,
            torch.tensor([0.0, 1.0]),
            bootstrap_value=2.0,
            normalize_advantage=False,
        )

        self.assertTrue(torch.allclose(continuing_returns, torch.tensor([3.52, 2.80]), atol=1e-5))
        self.assertTrue(torch.allclose(terminal_returns, torch.tensor([1.90, 1.00]), atol=1e-5))

    def test_semi_mdp_gae_uses_duration_discount_and_bootstraps_truncation(self):
        agent = PPOAgent(state_dim=2, action_dim=2, gamma=0.9, lam=1.0, device="cpu")
        rewards = torch.tensor([1.0, 2.0])
        values = torch.zeros(2)
        # First decision lasts two operational steps. The second transition is a
        # time-limit truncation with its own final-observation value override.
        _, returns = agent._gae_scalar(
            rewards,
            values,
            torch.tensor([0.0, 0.0]),
            bootstrap_value=99.0,
            normalize_advantage=False,
            discounts=torch.tensor([0.9**2, 0.9]),
            truncations=torch.tensor([0.0, 1.0]),
            next_value_overrides=torch.tensor([float("nan"), 3.0]),
        )

        self.assertTrue(torch.allclose(returns, torch.tensor([4.807, 4.7]), atol=1e-5))

    def test_discrete_entropy_does_not_include_unused_hold_magnitude(self):
        torch.manual_seed(4)
        agent = PPOAgent(
            state_dim=3,
            action_dim=3,
            lr=1e-6,
            initial_exploration_rate=0.0,
            final_exploration_rate=0.0,
            min_action_visits=0,
            device="cpu",
        )
        final_policy_layer = agent.net.pi_head[-1]
        with torch.no_grad():
            final_policy_layer.weight.zero_()
            final_policy_layer.bias.copy_(torch.tensor([12.0, -12.0, -12.0]))

        for index in range(8):
            action, state, old_logp, old_value, action_features, action_mask = agent.act(
                [0.0, 0.0, 0.0],
                deterministic=True,
                action_mask=[True, True, True],
            )
            self.assertEqual(action, 0)
            agent.store(
                state,
                action,
                float(index % 2),
                False,
                None,
                old_logp,
                old_value,
                action_features=action_features,
                action_mask=action_mask,
            )

        metrics = agent.update(epochs=1, batch_size=8, bootstrap_value=0.0)

        self.assertLess(metrics["policy_entropy_fraction"], 1e-4)
        self.assertEqual(metrics["magnitude_entropy"], 0.0)

    def test_mask_is_preserved_through_optimizer_update(self):
        torch.manual_seed(9)
        agent = PPOAgent(
            state_dim=4,
            action_dim=4,
            initial_exploration_rate=0.2,
            min_action_visits=0,
            device="cpu",
        )
        before = {
            name: parameter.detach().clone()
            for name, parameter in agent.net.named_parameters()
            if name.startswith("pi_head") or name.startswith("trunk")
        }

        for index in range(32):
            state_values = [index / 32.0, (index % 3) / 3.0, 0.5, -0.25]
            action, state, old_logp, old_value, action_features, action_mask = agent.act(
                state_values,
                action_mask=[True, True, False, True],
            )
            self.assertNotEqual(action, 2)
            reward = 1.0 if action == 1 else -0.25
            agent.store(
                state,
                action,
                reward,
                False,
                None,
                old_logp,
                old_value,
                action_features=action_features,
                action_mask=action_mask,
            )

        metrics = agent.update(epochs=3, batch_size=128, bootstrap_value=0.0)
        max_change = max(
            float((parameter.detach() - before[name]).abs().max().item())
            for name, parameter in agent.net.named_parameters()
            if name in before
        )

        self.assertTrue(metrics["update_performed"])
        self.assertGreater(metrics["optimizer_steps"], 0)
        self.assertGreater(max_change, 0.0)
        self.assertLessEqual(metrics["effective_batch_size"], 8)


if __name__ == "__main__":
    unittest.main()
