import unittest

import torch

from PPOAgent import PPOAgent


class PPOAgentRegressionTests(unittest.TestCase):
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
