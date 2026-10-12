from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from recurrent_ppo import (
    ActionTransform,
    Episode,
    RecurrentPPOAgent,
    RecurrentPPOConfig,
    RecurrentPPOState,
)


def _episode(identifier: float, length: int = 7, observation_dim: int = 4) -> Episode:
    time = np.arange(length + 1, dtype=np.float32)
    observations = np.stack(
        (
            np.full_like(time, identifier),
            time / max(1, length),
            np.sin(time),
            np.cos(time),
        ),
        axis=1,
    )[:, :observation_dim]
    coordinates = np.stack(
        (
            0.08 * np.sin(time[:-1] + identifier),
            0.15 * np.cos(time[:-1]),
            0.05 * np.sin(0.5 * time[:-1]),
        ),
        axis=1,
    )
    actions = ActionTransform.market_controls().coordinates_to_physical(coordinates)
    rewards = (
        0.2 * observations[:-1, 1]
        - np.square(np.log(actions[:, 0] / 1.15))
        - 0.1 * np.square(actions[:, 1] - 0.1)
    ).astype(np.float32)
    dones = np.zeros(length, dtype=np.float32)
    dones[-1] = 1.0
    return Episode(observations, actions, rewards, dones)


def _small_config(**overrides) -> RecurrentPPOConfig:
    values = dict(
        hidden_dim=16,
        update_epochs=2,
        minibatch_episodes=2,
        rollout_transitions=8,
        target_kl=None,
        seed=17,
    )
    values.update(overrides)
    return RecurrentPPOConfig(**values)


def test_ppo_identity_normalization_matches_sac_and_survives_save(tmp_path) -> None:
    agent = RecurrentPPOAgent(
        4,
        config=_small_config(observation_identity_indices=(1, 2, 3)),
    )
    clear_only = np.zeros((32, 4), dtype=np.float32)
    clear_only[:, 1] = 1.0
    agent.observation_normalizer.update(clear_only)
    unseen = torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32)
    normalized = agent.observation_normalizer.normalize_tensor(unseen)
    torch.testing.assert_close(normalized[0, 1:], unseen[0, 1:])
    assert float(normalized.abs().max()) <= 1.0

    artifact = tmp_path / "identity_ppo.pt"
    agent.save(artifact)
    loaded = RecurrentPPOAgent.load(artifact)
    np.testing.assert_array_equal(
        loaded.observation_normalizer.identity_mask,
        agent.observation_normalizer.identity_mask,
    )
    torch.testing.assert_close(
        loaded.observation_normalizer.normalize_tensor(unseen), normalized
    )


def test_actor_and_value_have_independent_grus_and_parameters() -> None:
    agent = RecurrentPPOAgent(4, config=_small_config())
    assert agent.actor.encoder.gru is not agent.value.encoder.gru
    actor_parameters = {id(parameter) for parameter in agent.actor.parameters()}
    value_parameters = {id(parameter) for parameter in agent.value.parameters()}
    assert actor_parameters.isdisjoint(value_parameters)


def test_act_uses_uncapped_coordinates_and_separate_hidden_state() -> None:
    agent = RecurrentPPOAgent(4, config=_small_config())
    observation = np.array([1.0, 0.25, -0.5, 0.75], dtype=np.float32)
    action, hidden, info = agent.act(observation, deterministic=True)

    assert isinstance(hidden, RecurrentPPOState)
    assert hidden.actor.shape == (1, 1, 16)
    assert hidden.value.shape == (1, 1, 16)
    assert hidden.actor.data_ptr() != hidden.value.data_ptr()
    assert action.shape == (3,)
    assert action[0] > 0.0 and action[2] > 0.0
    assert np.isfinite(action).all()
    assert np.asarray(info["policy_coordinates"]).shape == (3,)
    assert np.isfinite(info["log_probability"])
    assert np.isfinite(info["value"])

    second, next_hidden, _ = agent.act(
        observation,
        previous_action=action,
        previous_reward=0.4,
        hidden=hidden,
        deterministic=True,
    )
    assert np.isfinite(second).all()
    assert not torch.equal(next_hidden.actor, hidden.actor)
    assert not torch.equal(next_hidden.value, hidden.value)


def test_terminal_aware_gae_does_not_cross_boundaries() -> None:
    rewards = torch.tensor([1.0, 1.0])
    values = torch.tensor([0.5, 0.25, 100.0])
    dones = torch.tensor([0.0, 1.0])
    advantages, returns = RecurrentPPOAgent.compute_gae(
        rewards, values, dones, gamma=0.9, gae_lambda=1.0
    )
    torch.testing.assert_close(advantages, torch.tensor([1.4, 0.75]))
    torch.testing.assert_close(returns, torch.tensor([1.9, 1.0]))

    split_advantages, _ = RecurrentPPOAgent.compute_gae(
        torch.tensor([1.0, 10.0]),
        torch.zeros(3),
        torch.tensor([1.0, 1.0]),
        gamma=0.99,
        gae_lambda=1.0,
    )
    assert split_advantages[0].item() == pytest.approx(1.0)


def test_clipped_surrogate_uses_pessimistic_objective() -> None:
    loss, ratio, clip_fraction = RecurrentPPOAgent.clipped_policy_loss(
        torch.tensor([np.log(2.0)], dtype=torch.float32),
        torch.tensor([0.0]),
        torch.tensor([1.0]),
        clip_ratio=0.2,
    )
    assert ratio.item() == pytest.approx(2.0)
    assert loss.item() == pytest.approx(-1.2)
    assert clip_fraction.item() == pytest.approx(1.0)


def test_episode_storage_is_boundary_safe_and_update_changes_both_networks() -> None:
    agent = RecurrentPPOAgent(4, config=_small_config())
    first = _episode(1.0, length=5)
    second = _episode(9.0, length=7)
    initial_normalizer_count = agent.observation_normalizer.count
    agent.add_episode(first, actions_are_physical=True)
    agent.add_episode(second, actions_are_physical=True)

    # Statistics stay frozen for the whole behavior batch, and each episode's
    # recurrent context begins from the neutral previous action/reward/done.
    assert agent.observation_normalizer.count == initial_normalizer_count
    assert agent.pending_transitions == 12
    assert agent.can_update()
    for stored in agent._episodes:
        context = agent._episode_context(stored.episode)
        previous_slice = context[0, 0, agent.observation_dim :].cpu().numpy()
        np.testing.assert_allclose(previous_slice, 0.0)

    actor_before = [parameter.detach().clone() for parameter in agent.actor.parameters()]
    value_before = [parameter.detach().clone() for parameter in agent.value.parameters()]
    metrics = agent.update()

    assert all(np.isfinite(value) for value in metrics.values())
    assert metrics["episodes"] == 2.0
    assert metrics["transitions"] == 12.0
    assert agent.pending_transitions == 0
    assert not agent.can_update(minimum_transitions=1)
    assert agent.observation_normalizer.count > initial_normalizer_count
    assert any(
        not torch.equal(before, after.detach())
        for before, after in zip(actor_before, agent.actor.parameters())
    )
    assert any(
        not torch.equal(before, after.detach())
        for before, after in zip(value_before, agent.value.parameters())
    )


def test_incomplete_or_crossed_episode_is_rejected() -> None:
    agent = RecurrentPPOAgent(4, config=_small_config())
    episode = _episode(2.0, length=5)
    incomplete = Episode(
        episode.observations,
        episode.actions,
        episode.rewards,
        np.zeros_like(episode.dones),
    )
    with pytest.raises(ValueError, match="complete episode"):
        agent.add_episode(incomplete)

    crossed_dones = episode.dones.copy()
    crossed_dones[1] = 1.0
    crossed = Episode(
        episode.observations,
        episode.actions,
        episode.rewards,
        crossed_dones,
    )
    with pytest.raises(ValueError, match="complete episode"):
        agent.add_episode(crossed)


def test_save_load_preserves_deterministic_action(tmp_path: Path) -> None:
    agent = RecurrentPPOAgent(4, config=_small_config())
    agent.add_episode(_episode(3.0, length=8), actions_are_physical=True)
    agent.update()
    observation = np.array([3.0, 0.5, -0.25, 0.75], dtype=np.float32)
    before, before_hidden, before_info = agent.act(observation, deterministic=True)

    path = tmp_path / "recurrent_ppo.pt"
    agent.save(path, metadata={"role": "benchmark"})
    restored = RecurrentPPOAgent.load(path)
    after, after_hidden, after_info = restored.act(observation, deterministic=True)

    np.testing.assert_allclose(after, before, rtol=1e-6, atol=1e-7)
    assert after_info["value"] == pytest.approx(before_info["value"], rel=1e-6)
    torch.testing.assert_close(after_hidden.actor, before_hidden.actor)
    torch.testing.assert_close(after_hidden.value, before_hidden.value)
