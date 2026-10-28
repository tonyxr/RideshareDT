import tempfile
import unittest

import numpy as np
import torch

from Core import Core, V25_CRITIC_OPPONENT_MODES
from PPOAgent import PPOAgent
from platform_mdp import PlatformObservationModel
from pricing_models import build_independent_coefficient_action_space


class V25CreditAssignmentContractTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(2501)

    def test_reward_rate_uses_primitive_discounted_exposure(self) -> None:
        agent = PPOAgent(
            state_dim=2,
            action_dim=2,
            gamma=0.9,
            lam=1.0,
            reward_rate_beta=1.0,
            device="cpu",
        )
        # Primitive rewards [1, 2] followed by [3] have cumulative discounted
        # rewards 2.8 and 3.0 and exposure weights 1.9 and 1.0.  The per-step
        # reward rate is therefore exactly 2.0, not mean([2.8, 3.0]) == 2.9.
        for index, (reward, duration, discount) in enumerate(
            ((2.8, 2, 0.9**2), (3.0, 1, 0.9))
        ):
            action, state, logp, value, action_features, mask = agent.act(
                np.asarray([float(index), 1.0], dtype=np.float32)
            )
            agent.store(
                state,
                action,
                reward,
                index == 1,
                None,
                logp,
                value,
                action_features=action_features,
                action_mask=mask,
                duration=duration,
                discount=discount,
            )
        metrics = agent.update(epochs=1, batch_size=2)
        self.assertAlmostEqual(metrics["average_reward_rate"], 2.0, places=6)
        self.assertAlmostEqual(metrics["rollout_reward_exposure"], 2.9, places=6)

    def test_reward_rate_resets_at_regime_boundary(self) -> None:
        agent = PPOAgent(
            state_dim=2,
            action_dim=2,
            reward_rate_beta=0.1,
            device="cpu",
        )
        agent.average_reward_rate = 7.0
        agent.average_reward_initialized = True
        agent.reset_learning_context()
        self.assertEqual(agent.average_reward_rate, 0.0)
        self.assertFalse(agent.average_reward_initialized)

    def test_reward_rate_is_centered_separately_for_batched_regimes(self) -> None:
        baselines, rates = PPOAgent._regime_local_reward_baselines(
            torch.tensor([1.0, 1.0, 3.0, 3.0]),
            torch.ones(4),
            torch.tensor([0.0, 1.0, 0.0, 1.0]),
        )
        self.assertTrue(torch.allclose(baselines, torch.tensor([1.0, 1.0, 3.0, 3.0])))
        self.assertTrue(torch.allclose(rates, torch.tensor([1.0, 3.0])))

    def test_causal_reward_baseline_never_uses_current_episode_itself(self) -> None:
        agent = PPOAgent(
            state_dim=2,
            action_dim=2,
            reward_rate_beta=1.0,
            td_value_targets=True,
            device="cpu",
        )
        baselines, rates, lagged = agent._causal_regime_reward_baselines(
            torch.tensor([1.0, 1.0, 3.0, 3.0]),
            torch.ones(4),
            torch.tensor([0.0, 1.0, 0.0, 1.0]),
            ["fast", "fast", "slow", "slow"],
        )

        # Cold-start baselines are leave-one-episode-out: fast uses slow's
        # rate and slow uses fast's, rather than erasing their own returns.
        self.assertTrue(torch.allclose(
            baselines, torch.tensor([3.0, 3.0, 1.0, 1.0])
        ))
        self.assertTrue(torch.allclose(rates, torch.tensor([1.0, 3.0])))
        self.assertEqual(lagged, 0.0)

        next_baselines, _, next_lagged = agent._causal_regime_reward_baselines(
            torch.tensor([2.0, 2.0, 4.0, 4.0]),
            torch.ones(4),
            torch.tensor([0.0, 1.0, 0.0, 1.0]),
            ["fast", "fast", "slow", "slow"],
        )
        # The next population uses only the prior population's regime rates.
        self.assertTrue(torch.allclose(
            next_baselines, torch.tensor([1.0, 1.0, 3.0, 3.0])
        ))
        self.assertEqual(next_lagged, 1.0)

    def test_actor_gae_accepts_multiple_true_terminal_regimes_in_one_batch(self) -> None:
        agent = PPOAgent(state_dim=2, action_dim=2, device="cpu")
        advantages, returns = agent._gae_scalar(
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
            torch.zeros(4),
            torch.tensor([0.0, 1.0, 0.0, 1.0]),
            discounts=torch.full((4,), 0.9),
            truncations=torch.tensor([0.0, 1.0, 0.0, 1.0]),
            next_value_overrides=torch.full((4,), float("nan")),
        )
        self.assertTrue(torch.isfinite(advantages).all())
        self.assertTrue(torch.isfinite(returns).all())

    def test_response_timescale_regimes_receive_equal_total_loss_weight(self) -> None:
        # Two 2-transition fast regimes and one 6-transition slow regime must
        # each contribute one third of the population objective.
        dones = torch.tensor(
            [0, 1, 0, 1, 0, 0, 0, 0, 0, 1], dtype=torch.float32
        )
        weights = PPOAgent._terminal_segment_weights(dones)
        expected_mass = 10.0 / 3.0
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)
        self.assertAlmostEqual(float(weights[:2].sum()), expected_mass, places=6)
        self.assertAlmostEqual(float(weights[2:4].sum()), expected_mass, places=6)
        self.assertAlmostEqual(float(weights[4:].sum()), expected_mass, places=6)

    def test_replicated_openings_preserve_equal_controller_class_weight(self) -> None:
        # Fast has two 2-transition openings while slow has one six-transition
        # opening. Controller classes, not raw episode counts, each receive
        # half of the total population loss mass.
        dones = torch.tensor(
            [0, 1, 0, 1, 0, 0, 0, 0, 0, 1], dtype=torch.float32
        )
        regimes = ["fast"] * 4 + ["slow"] * 6
        weights = PPOAgent._regime_balanced_weights(dones, regimes)

        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)
        self.assertAlmostEqual(float(weights[:4].sum()), 5.0, places=6)
        self.assertAlmostEqual(float(weights[4:].sum()), 5.0, places=6)
        self.assertAlmostEqual(float(weights[:2].sum()), 2.5, places=6)
        self.assertAlmostEqual(float(weights[2:4].sum()), 2.5, places=6)

    def test_v25_contract_preserves_static_policy_and_separates_critics(self) -> None:
        core = Core(
            market_name="New York City",
            seed=2503,
            firm1_mode="RL",
            firm2_mode="static",
            training_curriculum="direct",
            training_dynamic_episode_days=15,
            credit_assignment_v25=True,
        )
        config = core.credit_assignment_config
        self.assertEqual(config["version"], 25)
        self.assertEqual(config["reward_rate_beta"], 0.20)
        self.assertEqual(config["recurrent_ingestion"], "every_public_frame")
        self.assertEqual(
            config["critic_target_contract"], "actual_next_decision_td0"
        )
        self.assertEqual(
            config["privileged_value_contract"],
            "direct_centralized_state_mlp",
        )
        self.assertEqual(
            config["actor_credit_contract"],
            "ppo_complete_smdp_return_only",
        )
        self.assertEqual(
            config["opponent_loss_weighting"],
            "equal_terminal_regime_mass",
        )
        self.assertEqual(
            config["reward_baseline_contract"],
            "lagged_opponent_regime_action_independent",
        )
        self.assertEqual(
            config["behavior_policy_contract"],
            "cosine_actor_uniform_mixture_exact_ppo_ratio",
        )
        self.assertEqual(
            config["critic_opponent_context"],
            "training_only_controller_class_one_hot",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/v25.pt"
            core.save_trained_model(path, model_id="v25-contract-test")
            artifact = torch.load(path, map_location="cpu", weights_only=False)
        self.assertEqual(
            artifact["policy_training_config"]["held_action_reward"],
            "discounted_smdp_sum_over_operational_steps",
        )
        self.assertAlmostEqual(config["counterfactual_actor_weight"], 0.0)
        self.assertAlmostEqual(core.firm1.agent.target_entropy_fraction, 0.30)
        self.assertAlmostEqual(core.firm1.agent.max_ent_coeff, 0.040)
        self.assertTrue(core.firm1.agent.scheduled_exploration)
        self.assertAlmostEqual(core.firm1.agent.exploration_rate, 0.30)
        core.firm1.configure_training_controls(1.0, False, 1.0)
        self.assertAlmostEqual(core.firm1.agent.exploration_rate, 0.05)
        self.assertTrue({
            "pi_price_gap",
            "mpc_grid",
            "region_supply_demand",
            "surge_driver_incentive",
            "queue_service_threshold",
        } <= set(core.training_dynamic_opponent_modes))
        self.assertTrue(core.firm1.agent.td_value_targets)
        self.assertTrue(core.firm1.agent.privileged_value_only)
        feasible = np.ones(core.firm1.agent.action_dim, dtype=bool)
        # Hold remains feasible even after an arbitrarily long static history.
        core.firm1._repeat_action_count = 10_000
        self.assertTrue(core._objective_action_mask(feasible)[0])
        actor_ids = {id(parameter) for parameter in core.firm1.agent.net.actor_parameters()}
        critic_ids = {id(parameter) for parameter in core.firm1.agent.net.critic_parameters()}
        q_ids = {id(parameter) for parameter in core.firm1.agent.net.q_parameters()}
        self.assertFalse(actor_ids & critic_ids)
        self.assertFalse(actor_ids & q_ids)
        self.assertFalse(critic_ids & q_ids)
        core.firm1.agent.average_reward_rate = 3.25
        core.firm1.agent.average_reward_initialized = True
        snapshot = core.firm1.snapshot_execution_state()
        core.firm1.reset_state_history()
        core.firm1.restore_execution_state(snapshot)
        self.assertAlmostEqual(core.firm1.agent.average_reward_rate, 3.25)
        self.assertTrue(core.firm1.agent.average_reward_initialized)
        dynamic_stage = core.training_stage_scheduler.stage_at(0.95)
        self.assertEqual(
            core._training_opponent_episode_days(
                dynamic_stage, "region_supply_demand"
            ),
            300,
        )
        self.assertEqual(
            core._training_opponent_episode_days(dynamic_stage, "heuristic"),
            dynamic_stage.episode_days,
        )
        population = core._training_opponent_population()
        self.assertEqual(population.count("heuristic"), 3)
        self.assertEqual(population.count("surge_driver_incentive"), 3)
        self.assertEqual(population.count("pi_price_gap"), 1)
        self.assertEqual(population.count("region_supply_demand"), 1)

        public = np.zeros(
            PlatformObservationModel.observation_dim, dtype=np.float32
        )
        core.firm2._ride_response_mode = "surge_driver_incentive"
        critic_state = core._build_privileged_critic_state(public)
        regime_width = int(core.regime_critic_extra_dim)
        stop = -regime_width if regime_width else None
        start = -(regime_width + len(core.critic_opponent_modes))
        opponent_context = critic_state[start:stop]
        self.assertAlmostEqual(float(opponent_context.sum()), 1.0)
        self.assertEqual(
            int(np.argmax(opponent_context)),
            core.critic_opponent_modes.index("surge_driver_incentive"),
        )

    def test_td_critic_uses_actual_next_decision_without_shortening_actor_gae(self) -> None:
        targets = PPOAgent._one_step_value_targets(
            torch.tensor([1.0, 2.0]),
            torch.tensor([0.9, 0.9]),
            torch.tensor([0.0, 1.0]),
            torch.tensor([3.0, 99.0]),
        )
        self.assertTrue(torch.allclose(targets, torch.tensor([3.7, 2.0])))

    def test_every_intermediate_frame_advances_persistent_memory(self) -> None:
        agent = PPOAgent(
            state_dim=6,
            action_dim=2,
            single_state_dim=2,
            frame_stack=3,
            persistent_recurrence=True,
            device="cpu",
        )
        initial = np.asarray([0.0, 0.0] * 3, dtype=np.float32)
        agent.act(initial)
        before = agent.recurrent_state_dict()["actor"]
        intermediate = np.asarray([0.0, 0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        agent.observe_recurrent_state(intermediate)
        after = agent.recurrent_state_dict()["actor"]
        self.assertFalse(torch.equal(before, after))

    def test_potential_shaping_can_be_finalized_at_actual_next_decision(self) -> None:
        agent = PPOAgent(
            state_dim=2,
            action_dim=2,
            gamma=0.9,
            potential_shaping_scale=0.5,
            device="cpu",
        )
        action, state, logp, value, action_features, mask = agent.act(
            np.asarray([0.0, 1.0], dtype=np.float32)
        )
        agent.store(
            state,
            action,
            2.0,
            False,
            None,
            logp,
            value,
            action_features=action_features,
            action_mask=mask,
            discount=0.9,
            potential=0.4,
        )
        self.assertFalse(agent.buf[-1].next_state_finalized)
        agent.finalize_last_transition(next_value=1.25, next_potential=0.8)
        transition = agent.buf[-1]
        self.assertTrue(transition.next_state_finalized)
        self.assertAlmostEqual(transition.reward, 2.0 + 0.5 * (0.9 * 0.8 - 0.4))
        self.assertAlmostEqual(float(transition.next_value_override), 1.25)

    def test_counterfactual_actor_credit_ignores_unsupervised_actions(self) -> None:
        values = torch.tensor(
            [[1.0, 3.0, 1_000.0, -1_000.0], [2.0, 4.0, 9_000.0, -9_000.0]]
        )
        masks = torch.tensor(
            [[True, True, False, False], [True, True, False, False]]
        )
        actions = torch.tensor([1, 0])
        advantages, valid = PPOAgent._counterfactual_actor_advantages(
            values, masks, actions, normalize=False
        )
        self.assertTrue(torch.equal(valid, torch.tensor([True, True])))
        self.assertTrue(torch.allclose(advantages, torch.tensor([1.0, -1.0])))

    def test_independent_coefficient_features_encode_actual_target_delta(self) -> None:
        observer = PlatformObservationModel(seed=2502)
        keys = ["base_fare", "per_minute", "per_mile", "booking_fee", "airport_fee"]
        action_steps, multipliers, targets, action_keys = (
            build_independent_coefficient_action_space(keys)
        )
        anchors = {key: 10.0 for key in keys}
        own = dict(anchors)
        features = observer.build_action_features(
            action_steps=action_steps,
            action_keys=action_keys,
            own_coefficients=own,
            anchor_coefficients=anchors,
            coefficient_steps={key: 0.1 for key in keys},
            coefficient_bounds={key: (1.0, 20.0) for key in keys},
            step_scale=1.0,
            target_gap=0.0,
            action_target_keys=targets,
            action_target_multipliers=multipliers,
        )
        # The four projected-fare features occupy columns 14:18. Independent
        # base-fare 0.70x and 1.30x targets point in opposite directions.
        self.assertTrue(np.all(features[1, 14:18] < 0.0))
        self.assertTrue(np.all(features[7, 14:18] > 0.0))
        np.testing.assert_array_equal(features[1, 2:7], [1., 0., 0., 0., 0.])
        # A 1.00x base-fare target at the anchor remains distinct from hold.
        self.assertEqual(features[4, 0], 0.0)
        self.assertAlmostEqual(float(features[4, 19]), 0.2)

    def test_observed_counterfactual_policy_loss_prefers_better_replayed_action(self) -> None:
        logits = torch.zeros((1, 4), dtype=torch.float32, requires_grad=True)
        targets = torch.tensor([[1.0, -1.0, float("nan"), float("nan")]])
        masks = torch.tensor([[True, True, False, False]])
        loss = PPOAgent._observed_counterfactual_policy_loss(
            logits, targets, masks
        )
        loss.backward()
        self.assertLess(float(logits.grad[0, 0]), 0.0)
        self.assertGreater(float(logits.grad[0, 1]), 0.0)
        self.assertEqual(float(logits.grad[0, 2]), 0.0)
        self.assertEqual(float(logits.grad[0, 3]), 0.0)

    def test_observed_counterfactual_policy_loss_does_not_push_tied_actions(self) -> None:
        logits = torch.tensor(
            [[4.0, -2.0, 0.5]], dtype=torch.float32, requires_grad=True
        )
        targets = torch.zeros((1, 3), dtype=torch.float32)
        masks = torch.ones((1, 3), dtype=torch.bool)
        loss = PPOAgent._observed_counterfactual_policy_loss(
            logits, targets, masks
        )
        loss.backward()
        self.assertAlmostEqual(float(loss.item()), 0.0, places=8)
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))


if __name__ == "__main__":
    unittest.main()
