import copy
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from Core import Core, PRIVILEGED_CRITIC_STATE_DIM
from PPOAgent import PPOAgent
from coeff_utils import get_coeff
from pricing_models import FirmMetrics


class V24CreditAssignmentTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(2401)

    def test_v23_defaults_keep_new_critic_paths_absent(self) -> None:
        agent = PPOAgent(state_dim=6, action_dim=3, device="cpu")
        self.assertFalse(agent.privileged_critic)
        self.assertFalse(agent.persistent_recurrence)
        self.assertFalse(agent.hard_kl_stop)
        self.assertEqual(agent.reward_rate_beta, 0.0)
        self.assertIsNone(agent.net.critic_privileged_encoder)
        self.assertIsNone(agent.net.critic_fusion)
        self.assertIsNone(agent.net.action_q_head)
        self.assertEqual(agent.normalizer_state_dict(), {})

    def test_privileged_state_cannot_change_actor_logits(self) -> None:
        agent = PPOAgent(
            state_dim=6,
            action_dim=3,
            single_state_dim=2,
            frame_stack=3,
            critic_state_dim=4,
            privileged_critic=True,
            device="cpu",
        )
        state = torch.tensor([[0.1, -0.2, 0.3, 0.4, -0.1, 0.2]])
        critic_a = torch.zeros((1, 4))
        critic_b = torch.full((1, 4), 5.0)
        with torch.no_grad():
            logits_a, _ = agent.net.actor_forward(state)
            value_a, _, _ = agent.net.critic_forward(
                state, critic_state=critic_a
            )
            logits_b, _ = agent.net.actor_forward(state)
            value_b, _, _ = agent.net.critic_forward(
                state, critic_state=critic_b
            )
        self.assertTrue(torch.equal(logits_a, logits_b))
        self.assertFalse(torch.equal(value_a, value_b))
        privileged_ids = {
            id(parameter)
            for parameter in agent.net.critic_privileged_encoder.parameters()
        }
        actor_ids = {id(parameter) for parameter in agent.net.actor_parameters()}
        self.assertFalse(privileged_ids & actor_ids)

    def test_persistent_recurrence_is_saved_for_exact_ppo_likelihood(self) -> None:
        agent = PPOAgent(
            state_dim=6,
            action_dim=3,
            single_state_dim=2,
            frame_stack=3,
            persistent_recurrence=True,
            device="cpu",
        )
        for index in range(3):
            state = np.full(6, 0.1 * (index + 1), dtype=np.float32)
            action, state_t, logp, value, _, mask = agent.act(state)
            agent.store(
                state_t,
                action,
                float(index),
                index == 2,
                None,
                logp,
                value,
                action_mask=mask,
            )
        states = torch.stack([transition.state for transition in agent.buf])
        logits = agent._actor_logits_batch(
            states, [transition.actor_hidden for transition in agent.buf]
        )
        distribution = torch.distributions.Categorical(logits=logits)
        actions = torch.stack([transition.action for transition in agent.buf])
        recomputed = distribution.log_prob(actions)
        stored = torch.stack([transition.old_logp for transition in agent.buf])
        self.assertTrue(torch.allclose(recomputed, stored, atol=1e-6))
        self.assertEqual(agent.buf[0].actor_hidden.numel(), 0)
        self.assertGreater(agent.buf[1].actor_hidden.numel(), 0)

    def test_counterfactual_q_head_receives_sparse_branch_supervision(self) -> None:
        agent = PPOAgent(
            state_dim=4,
            action_dim=3,
            enable_counterfactual_q=True,
            action_q_coeff=0.5,
            counterfactual_actor_weight=0.2,
            device="cpu",
        )
        before = [
            parameter.detach().clone()
            for parameter in agent.net.action_q_head.parameters()
        ]
        for index in range(8):
            state = np.asarray([index, 1.0, -1.0, 0.5], dtype=np.float32)
            action, state_t, logp, value, _, mask = agent.act(state)
            targets = np.asarray([-1.0, np.nan, 1.0], dtype=np.float32)
            agent.store(
                state_t,
                action,
                float(index % 3),
                index == 7,
                None,
                logp,
                value,
                action_mask=mask,
                counterfactual_action_values=targets,
            )
        metrics = agent.update(epochs=2, batch_size=4)
        after = list(agent.net.action_q_head.parameters())
        self.assertGreater(metrics["action_q_loss"], 0.0)
        self.assertAlmostEqual(metrics["counterfactual_target_coverage"], 2.0 / 3.0)
        self.assertTrue(metrics["objective_optimizer_ready"])
        self.assertTrue(
            any(not torch.equal(old, new) for old, new in zip(before, after))
        )

    def test_potential_shaping_telescopes_under_variable_discounts(self) -> None:
        rewards = [1.0, -0.5, 2.0]
        discounts = [0.9, 0.8, 0.7]
        potentials = [0.4, -0.2, 0.7, 1.3]
        scale = 0.25
        raw_return = 0.0
        shaped_return = 0.0
        weight = 1.0
        for index, reward in enumerate(rewards):
            shaped = PPOAgent.potential_based_reward(
                reward,
                discounts[index],
                potentials[index],
                potentials[index + 1],
                scale=scale,
            )
            raw_return += weight * reward
            shaped_return += weight * shaped
            weight *= discounts[index]
        expected_delta = scale * (
            weight * potentials[-1] - potentials[0]
        )
        self.assertAlmostEqual(
            shaped_return - raw_return, expected_delta, places=7
        )

    def test_hard_kl_gate_terminates_actor_epochs(self) -> None:
        agent = PPOAgent(
            state_dim=3,
            action_dim=2,
            hard_kl_stop=True,
            target_kl=1e-4,
            device="cpu",
        )
        for index in range(6):
            state = np.asarray([index, 1.0, -1.0], dtype=np.float32)
            action, state_t, logp, value, _, mask = agent.act(state)
            agent.store(
                state_t,
                action,
                float(index),
                index == 5,
                None,
                logp - 2.0,
                value,
                action_mask=mask,
            )
        metrics = agent.update(epochs=5, batch_size=6)
        self.assertTrue(metrics["stopped_early_kl"])
        self.assertLess(metrics["actor_optimizer_steps"], 5)

    def test_learned_reward_rate_is_persistent_not_rollout_local(self) -> None:
        agent = PPOAgent(
            state_dim=3,
            action_dim=2,
            reward_rate_beta=0.5,
            v_coeff=0.0,
            device="cpu",
        )

        def collect(reward: float) -> None:
            for index in range(3):
                state = np.asarray([index, 0.5, -0.5], dtype=np.float32)
                action, state_t, logp, value, _, mask = agent.act(state)
                agent.store(
                    state_t,
                    action,
                    reward,
                    index == 2,
                    None,
                    logp,
                    value,
                    action_mask=mask,
                )

        collect(2.0)
        first = agent.update(epochs=1, batch_size=3)
        self.assertAlmostEqual(first["average_reward_rate"], 2.0)
        collect(4.0)
        second = agent.update(epochs=1, batch_size=3)
        self.assertAlmostEqual(second["average_reward_rate"], 3.0)
        state = agent.normalizer_state_dict()
        self.assertTrue(state["average_reward_initialized"])
        self.assertAlmostEqual(state["average_reward_rate"], 3.0)

    def test_v24_core_schema_and_counterfactual_rollback(self) -> None:
        core = Core(
            market_name="New York City",
            seed=2402,
            choice_mode="parametric",
            firm1_mode="RL",
            firm2_mode="static",
            total_customers_pool=32,
            simulation_sample_cap=8,
            credit_assignment_v24=True,
            counterfactual_branches=2,
        )
        raw_state = core._build_rl_state(2, 12, "Clear")
        critic_state = core._build_privileged_critic_state(raw_state)
        self.assertEqual(critic_state.shape, (PRIVILEGED_CRITIC_STATE_DIM,))
        self.assertEqual(core.credit_assignment_config["version"], 24)
        self.assertFalse(core.constrained_reward)
        self.assertFalse(core.firm1.agent.deployment_optimizer_enabled)

        original_coefficients = core._coeff_snapshot()
        original_share = core.last_share
        rng_state = copy.deepcopy(core.rng.bit_generator.state)
        expected_rng = np.random.default_rng()
        expected_rng.bit_generator.state = copy.deepcopy(rng_state)
        expected_next_random = float(expected_rng.random())

        def fake_simulate_batch(**kwargs):
            del kwargs
            core.rng.random()
            core.last_share = 0.99
            return [], FirmMetrics(), FirmMetrics(), 0.0, 0.0, 0.0

        def fake_reward(**kwargs):
            del kwargs
            return {
                "reward": float(get_coeff(
                    core.market.curr_market,
                    core.firm1.overrides,
                    "base_fare",
                ))
            }

        with mock.patch.object(
            core, "simulate_batch", side_effect=fake_simulate_batch
        ), mock.patch.object(
            core, "_reward_diagnostics", side_effect=fake_reward
        ):
            values, branch_mask = core._counterfactual_action_values(
                selected_action=0,
                policy_top_action=1,
                policy_second_action=2,
                action_mask=np.ones(core.firm1.agent.action_dim, dtype=bool),
                day_of_week=2,
                hour=12,
                weather="Clear",
                customers_per_step=8,
                sampled_profiles=[],
            )
        self.assertEqual(int(branch_mask.sum()), 2)
        self.assertAlmostEqual(float(values[branch_mask].mean()), 0.0, places=6)
        self.assertEqual(core._coeff_snapshot(), original_coefficients)
        self.assertEqual(core.last_share, original_share)
        self.assertAlmostEqual(float(core.rng.random()), expected_next_random)

    def test_v24_artifact_round_trip_preserves_training_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/v24.pt"
            source = Core(
                market_name="New York City",
                seed=2403,
                choice_mode="parametric",
                firm1_mode="RL",
                firm2_mode="static",
                total_customers_pool=32,
                credit_assignment_v24=True,
            )
            source.save_trained_model(path, model_id="v24-test")
            artifact = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(artifact["version"], 24)
            self.assertEqual(
                artifact["credit_assignment_config"],
                source.credit_assignment_config,
            )
            target = Core(
                market_name="New York City",
                seed=2403,
                choice_mode="parametric",
                firm1_mode="RL",
                firm2_mode="static",
                total_customers_pool=32,
                credit_assignment_v24=True,
            )
            target.load_trained_model(path)
            for key, value in source.firm1.agent.net.state_dict().items():
                self.assertTrue(torch.equal(value, target.firm1.agent.net.state_dict()[key]))


if __name__ == "__main__":
    unittest.main()
