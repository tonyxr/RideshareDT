import unittest

import numpy as np
import torch

from PPOAgent import PPOAgent


class PPOAgentRegressionTests(unittest.TestCase):
    def test_actor_starts_uniform_with_hold_as_deterministic_tie_break(self):
        agent = PPOAgent(state_dim=6, action_dim=4, device="cpu")
        state = np.zeros(6, dtype=np.float32)
        with torch.no_grad():
            logits, _ = agent.net.actor_forward(
                torch.as_tensor(state).reshape(1, -1)
            )
        probabilities = torch.softmax(logits, dim=-1).squeeze(0)
        np.testing.assert_allclose(
            probabilities.cpu().numpy(),
            np.full(4, 0.25, dtype=np.float32),
            atol=1e-7,
        )
        action, *_ = agent.act(state, deterministic=True)
        self.assertEqual(action, 0)

    def test_raw_huber_critic_gradient_is_not_divided_by_target_scale(self):
        values = torch.zeros(2, requires_grad=True)
        targets = torch.tensor([30.0, 60.0])

        loss = PPOAgent._critic_value_loss(values, targets)
        loss.backward()

        self.assertGreater(float(values.grad.abs().mean().item()), 0.20)

    def test_stage_boundary_discard_clears_undersized_rollout(self):
        agent = PPOAgent(state_dim=3, action_dim=2, device="cpu")
        agent.buf.extend([object(), object()])

        discarded = agent.discard_rollout()

        self.assertEqual(discarded, 2)
        self.assertEqual(agent.buffer_size, 0)

    def test_mutual_information_regularizer_is_removed(self):
        self.assertFalse(hasattr(PPOAgent, "_mutual_information"))

    def test_removed_top2_mode_is_rejected(self):
        agent = PPOAgent(state_dim=3, action_dim=3, device="cpu")
        final_policy_layer = agent.net.pi_head
        with torch.no_grad():
            final_policy_layer.weight.zero_()
            final_policy_layer.bias.copy_(torch.tensor([2.0, 1.99, -10.0]))

        with self.assertRaisesRegex(ValueError, "policy_mode must be one of"):
            agent.act(
                [0.0, 0.0, 0.0],
                deterministic=True,
                action_mask=[True, True, True],
                policy_mode="top2_margin",
            )

    def test_argmax_and_argmax_raw_both_use_actor(self):
        agent = PPOAgent(state_dim=3, action_dim=3, device="cpu")
        final_policy_layer = agent.net.pi_head
        with torch.no_grad():
            final_policy_layer.weight.zero_()
            final_policy_layer.bias.copy_(torch.tensor([2.0, 1.0, 0.0]))

        optimized_action, *_ = agent.act(
            [0.0, 0.0, 0.0], deterministic=True, policy_mode="argmax"
        )
        raw_action, *_ = agent.act(
            [0.0, 0.0, 0.0], deterministic=True, policy_mode="argmax_raw"
        )

        self.assertEqual(optimized_action, 0)
        self.assertEqual(raw_action, 0)
        self.assertIsNone(agent.q_opt)

    def test_state_only_policy_ignores_action_features_and_has_no_auxiliaries(self):
        torch.manual_seed(17)
        agent = PPOAgent(
            state_dim=6,
            single_state_dim=3,
            frame_stack=2,
            action_dim=3,
            action_feature_dim=4,
            response_dim=2,
            action_q_coeff=0.1,
            device="cpu",
        )
        state = torch.tensor([[0.1, 0.2, 0.3, 0.8, -0.4, 0.6]])
        reversed_state = torch.tensor([[0.8, -0.4, 0.6, 0.1, 0.2, 0.3]])
        action_features = torch.tensor([[
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.5, 0.0],
            [-1.0, 1.0, 0.5, 1.0],
        ]])
        # Production initialization is deliberately uniform.  Give the output
        # head fixed non-zero weights here so this test isolates state ordering
        # from that separate initialization contract.
        with torch.no_grad():
            agent.net.pi_head.weight.copy_(
                torch.linspace(
                    -0.2,
                    0.2,
                    agent.net.pi_head.weight.numel(),
                ).reshape_as(agent.net.pi_head.weight)
            )

        logits, *_, response, q_values = agent.net(state, action_features)
        logits_without_features, *_ = agent.net(state, None)
        reversed_logits, *_ = agent.net(reversed_state, action_features)

        self.assertEqual(tuple(response.shape), (1, 2))
        self.assertEqual(q_values.numel(), 0)
        self.assertTrue(torch.all(response == 0))
        self.assertTrue(torch.allclose(logits, logits_without_features))
        self.assertFalse(torch.allclose(logits, reversed_logits))

    def test_actor_and_critic_optimizers_have_disjoint_parameters(self):
        agent = PPOAgent(
            state_dim=3,
            action_dim=3,
            initial_exploration_rate=0.20,
            final_exploration_rate=0.01,
            ent_coeff=0.02,
            min_ent_coeff=0.001,
            device="cpu",
        )
        actor_ids = {
            id(parameter)
            for group in agent.actor_opt.param_groups
            for parameter in group["params"]
        }
        critic_ids = {
            id(parameter)
            for group in agent.critic_opt.param_groups
            for parameter in group["params"]
        }

        self.assertTrue(actor_ids)
        self.assertTrue(critic_ids)
        self.assertFalse(actor_ids & critic_ids)
        self.assertIsInstance(agent.net.actor_recurrent, torch.nn.GRU)
        self.assertIsInstance(agent.net.critic_recurrent, torch.nn.GRU)

    def test_objective_q_optimizer_is_removed(self):
        agent = PPOAgent(
            state_dim=3,
            action_dim=3,
            action_feature_dim=2,
            action_q_coeff=1.0,
            objective_optimizer_min_updates=0,
            objective_optimizer_weight=1.0,
            device="cpu",
        )
        self.assertIsNone(agent.q_opt)
        self.assertEqual(agent.net.q_parameters(), [])

        optimized, diagnostics = agent._deployment_optimizer_logits(
            torch.tensor([[3.0, 2.9, 0.0]]),
            torch.tensor([[0.0, 4.0, -4.0]]),
            torch.tensor([[True, True, True]]),
        )
        self.assertEqual(int(torch.argmax(optimized, dim=-1).item()), 0)
        self.assertEqual(diagnostics["optimizer_action_changed"], 0.0)

        agent.deployment_optimizer_enabled = False
        bypassed, diagnostics = agent._deployment_optimizer_logits(
            torch.tensor([[3.0, 2.9, 0.0]]),
            torch.tensor([[0.0, 4.0, -4.0]]),
            torch.tensor([[True, True, True]]),
        )
        self.assertEqual(int(torch.argmax(bypassed, dim=-1).item()), 0)
        self.assertEqual(diagnostics["optimizer_ready"], 0.0)

    def test_opponent_forecast_and_response_parameters_are_removed(self):
        torch.manual_seed(23)
        agent = PPOAgent(
            state_dim=6,
            single_state_dim=3,
            frame_stack=2,
            action_dim=3,
            action_feature_dim=4,
            response_dim=8,
            opponent_forecast_dim=6,
            device="cpu",
        )
        state = torch.randn(2, 6)
        action_features = torch.randn(2, 3, 4)

        logits_before, *_, response, q_values = agent.net(state, action_features)
        logits_after, *_ = agent.net(state, torch.randn_like(action_features))
        self.assertTrue(torch.allclose(logits_before, logits_after, atol=1e-6))
        self.assertEqual(agent.net.response_parameters(), [])
        self.assertTrue(torch.all(response == 0))
        self.assertEqual(q_values.numel(), 0)

    def test_explicit_stage_controls_change_only_declared_optimizer_controls(self):
        agent = PPOAgent(
            state_dim=3,
            action_dim=3,
            initial_exploration_rate=0.20,
            final_exploration_rate=0.01,
            scheduled_exploration=True,
            ent_coeff=0.02,
            min_ent_coeff=0.001,
            device="cpu",
        )
        agent.set_stage_controls(
            exploration_rate=0.01,
            entropy_scale=0.05,
            learning_rate_scale=0.5,
        )

        self.assertAlmostEqual(agent.exploration_rate, 0.01)
        self.assertAlmostEqual(agent.ent_coeff, 0.02)
        self.assertAlmostEqual(agent.curr_lr, 0.5 * agent.base_lr)

    def test_behavior_exploration_mixture_respects_policy_and_mask(self):
        logits = torch.tensor([[4.0, -4.0, 2.0], [-1.0, 1.0, 3.0]])
        mask = torch.tensor([[True, True, False], [True, False, True]])

        uniform = PPOAgent._exploratory_distribution(
            logits, torch.tensor([1.0, 1.0]), mask
        ).probs
        actor = PPOAgent._exploratory_distribution(
            logits, torch.tensor([0.0, 0.0]), mask
        ).probs
        expected_actor = torch.softmax(
            PPOAgent._masked_logits(logits, mask), dim=-1
        )

        self.assertTrue(torch.allclose(
            uniform,
            torch.tensor([[0.5, 0.5, 0.0], [0.5, 0.0, 0.5]]),
        ))
        self.assertTrue(torch.allclose(actor, expected_actor))
        self.assertTrue(torch.all(uniform[~mask] == 0.0))

    def test_exploration_schedule_starts_high_and_decays_monotonically(self):
        agent = PPOAgent(
            state_dim=3,
            action_dim=3,
            initial_exploration_rate=0.30,
            final_exploration_rate=0.05,
            exploration_warmup_fraction=0.05,
            exploration_fraction=1.00,
            scheduled_exploration=True,
            device="cpu",
        )
        progress = np.linspace(0.0, 1.0, 101)
        rates = [agent.exploration_rate_at_progress(value) for value in progress]

        self.assertAlmostEqual(rates[0], 0.30)
        self.assertAlmostEqual(agent.exploration_rate_at_progress(0.05), 0.30)
        self.assertGreater(agent.exploration_rate_at_progress(0.85), 0.05)
        self.assertAlmostEqual(rates[-1], 0.05)
        self.assertTrue(all(left >= right for left, right in zip(rates, rates[1:])))

    def test_deterministic_actor_is_not_changed_by_exploration(self):
        agent = PPOAgent(
            state_dim=3,
            action_dim=3,
            initial_exploration_rate=1.0,
            final_exploration_rate=1.0,
            scheduled_exploration=True,
            device="cpu",
        )
        with torch.no_grad():
            agent.net.pi_head.weight.zero_()
            agent.net.pi_head.bias.copy_(torch.tensor([-2.0, 6.0, 1.0]))

        action, *_ = agent.act(
            [0.0, 0.0, 0.0],
            deterministic=True,
            action_mask=[True, True, True],
        )

        self.assertEqual(action, 1)
        self.assertEqual(agent.last_action_exploration_rate, 0.0)

    def test_rollout_retains_behavior_exploration_for_ppo_ratio(self):
        torch.manual_seed(123)
        agent = PPOAgent(
            state_dim=3,
            action_dim=3,
            initial_exploration_rate=0.30,
            final_exploration_rate=0.02,
            scheduled_exploration=True,
            device="cpu",
        )
        for index in range(12):
            action, state, old_logp, old_value, features, mask = agent.act(
                [index / 12.0, 0.25, -0.5],
                action_mask=[True, True, True],
            )
            agent.store(
                state,
                action,
                float(action == 1),
                False,
                None,
                old_logp,
                old_value,
                action_features=features,
                action_mask=mask,
            )

        self.assertTrue(all(
            np.isclose(transition.exploration_rate, 0.30)
            for transition in agent.buf
        ))
        metrics = agent.update(epochs=1, batch_size=12, bootstrap_value=0.0)
        self.assertTrue(metrics["update_performed"])
        self.assertAlmostEqual(metrics["exploration_rate"], 0.30, places=6)
        self.assertTrue(np.isfinite(metrics["approx_kl"]))

    def test_critic_checkpoint_round_trip_uses_raw_value(self):
        torch.manual_seed(12)
        source = PPOAgent(state_dim=3, action_dim=2, device="cpu")
        state = torch.tensor([[0.1, -0.2, 0.3]])
        with torch.no_grad():
            _, _, _, expected, *_ = source.net(state)

        target = PPOAgent(state_dim=3, action_dim=2, device="cpu")
        target.net.load_state_dict(source.net.state_dict())
        target.load_normalizer_state_dict(source.normalizer_state_dict())
        with torch.no_grad():
            _, _, _, actual, *_ = target.net(state)

        self.assertEqual(source.normalizer_state_dict(), {})
        self.assertTrue(torch.allclose(actual, expected))

    def test_minimal_update_ignores_auxiliary_targets(self):
        rng = np.random.default_rng(5)
        agent = PPOAgent(
            state_dim=8,
            single_state_dim=4,
            frame_stack=2,
            action_dim=4,
            action_feature_dim=5,
            response_dim=3,
            constraint_dim=2,
            action_q_coeff=1.0,
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
                actor_credit=float(index - 8),
                constraint_costs=np.array([0.02, 0.04], dtype=np.float32),
                risk_cost=0.03,
                response_target=np.array([0.1, -0.1, 0.05], dtype=np.float32),
                action_features=action_features,
                action_mask=action_mask,
            )

        metrics = agent.update(epochs=1, batch_size=8, bootstrap_value=0.0)

        self.assertTrue(metrics["update_performed"])
        self.assertEqual(agent.buffer_size, 0)
        self.assertAlmostEqual(metrics["target_kl"], agent.target_kl)
        self.assertTrue(np.isfinite(metrics["response_loss"]))
        self.assertTrue(np.isfinite(metrics["state_action_mi"]))
        self.assertEqual(metrics["direct_actor_credit_std"], 0.0)
        self.assertAlmostEqual(metrics["direct_actor_credit_weight"], 0.0)
        self.assertIn("raw_policy_argmax_concentration", metrics)
        self.assertEqual(metrics["objective_q_optimizer_steps"], 0)
        self.assertEqual(metrics["response_optimizer_steps"], 0)
        self.assertEqual(metrics["action_q_loss"], 0.0)

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

    def test_nonterminal_rollout_tail_cannot_silently_zero_bootstrap(self):
        agent = PPOAgent(state_dim=2, action_dim=2, gamma=0.9, lam=1.0, device="cpu")

        with self.assertRaisesRegex(ValueError, "finite-horizon end effects"):
            agent._gae_scalar(
                torch.tensor([1.0]),
                torch.tensor([2.0]),
                torch.tensor([0.0]),
                bootstrap_value=None,
                normalize_advantage=False,
            )

        _, returns = agent._gae_scalar(
            torch.tensor([1.0]),
            torch.tensor([2.0]),
            torch.tensor([0.0]),
            bootstrap_value=2.0,
            normalize_advantage=False,
        )
        self.assertTrue(torch.allclose(returns, torch.tensor([2.8]), atol=1e-5))

    def test_internal_truncation_requires_its_own_final_observation_value(self):
        agent = PPOAgent(state_dim=2, action_dim=2, gamma=0.9, lam=1.0, device="cpu")

        with self.assertRaisesRegex(ValueError, "internal truncated transitions"):
            agent._gae_scalar(
                torch.tensor([1.0, 1.0]),
                torch.tensor([2.0, 2.0]),
                torch.tensor([0.0, 0.0]),
                bootstrap_value=2.0,
                normalize_advantage=False,
                truncations=torch.tensor([1.0, 0.0]),
            )

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
        final_policy_layer = agent.net.pi_head
        with torch.no_grad():
            final_policy_layer.weight.zero_()
            final_policy_layer.bias.copy_(torch.tensor([12.0, -12.0, -12.0]))

        initial_entropy_coefficient = agent.ent_coeff
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
        self.assertGreater(metrics["ent_coeff"], initial_entropy_coefficient)
        self.assertGreater(metrics["entropy_fraction_error"], 0.0)

    def test_deterministic_action_uses_one_fixed_step(self):
        torch.manual_seed(14)
        agent = PPOAgent(
            state_dim=3,
            action_dim=2,
            initial_exploration_rate=0.0,
            final_exploration_rate=0.0,
            min_action_visits=0,
            device="cpu",
        )
        with torch.no_grad():
            final_policy_layer = agent.net.pi_head
            final_policy_layer.weight.zero_()
            final_policy_layer.bias.copy_(torch.tensor([-12.0, 12.0]))

        action, *_ = agent.act(
            [0.0, 0.0, 0.0],
            deterministic=True,
            action_mask=[True, True],
        )

        self.assertEqual(action, 1)
        self.assertEqual(agent.last_continuous_magnitude, 1.0)
        self.assertIsNone(agent.net.mag_mean_head)

    def test_mask_is_preserved_through_optimizer_update(self):
        torch.manual_seed(9)
        agent = PPOAgent(
            state_dim=4,
            action_dim=4,
            initial_exploration_rate=0.2,
            min_action_visits=0,
            target_kl=1e-4,
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
        self.assertEqual(metrics["effective_batch_size"], 32)
        self.assertEqual(metrics["actor_optimizer_steps"], 3)
        self.assertEqual(metrics["critic_optimizer_steps"], 3)
        self.assertFalse(metrics["stopped_early_kl"])

    def test_continuing_update_removes_constant_reward_level_from_critic_target(self):
        torch.manual_seed(23)
        agent = PPOAgent(
            state_dim=3,
            action_dim=2,
            gamma=0.999,
            lam=0.99,
            device="cpu",
        )
        with torch.no_grad():
            for parameter in agent.net.critic_parameters():
                parameter.zero_()
        for index in range(16):
            action, state, old_logp, old_value, action_features, action_mask = agent.act(
                [index / 16.0, 0.5, -0.25],
                action_mask=[True, True],
            )
            agent.store(
                state,
                action,
                3.0,
                False,
                None,
                old_logp,
                old_value,
                action_features=action_features,
                action_mask=action_mask,
            )

        metrics = agent.update(epochs=1, batch_size=16, bootstrap_value=0.0)

        self.assertAlmostEqual(metrics["critic_target_mean"], 0.0, places=6)
        self.assertAlmostEqual(metrics["critic_target_std"], 0.0, places=6)
        self.assertAlmostEqual(
            metrics["critic_lr"],
            3.0 * metrics["lr"],
        )


if __name__ == "__main__":
    unittest.main()
