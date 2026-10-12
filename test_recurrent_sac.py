from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from continuous_benchmarks import CEMConfig, FixedPolicy, evaluate_policy, optimize_static_tariff
from recurrent_sac import (
    ActionTransform,
    Episode,
    RecurrentSACAgent,
    RecurrentSACConfig,
    SequenceReplayBuffer,
    NumericalPolicyActionError,
    effective_horizon_lengths,
)
from scripts.v28_learner_ladder import (
    GeometricActionCommitment,
    NUMERICAL_POLICY_PENALTY,
    _local_action_support,
    _observation_identity_indices,
    train_stage,
)


def test_seed81649_unseen_rain_indicator_stays_identity_and_persists(tmp_path) -> None:
    names = (
        "continuous",
        "weather_clear",
        "weather_rain",
        "public_storm_warning",
    )
    identity = _observation_identity_indices(names)
    assert identity == (1, 2, 3)
    agent = RecurrentSACAgent(
        len(names),
        config=RecurrentSACConfig(
            hidden_dim=8,
            batch_size=1,
            seed=81_649,
            observation_identity_indices=identity,
        ),
    )
    clear_only = np.zeros((64, len(names)), dtype=np.float32)
    clear_only[:, 1] = 1.0
    agent.observation_normalizer.update(clear_only)
    unseen_rain = torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32)
    normalized = agent.observation_normalizer.normalize_tensor(unseen_rain)
    torch.testing.assert_close(normalized[0, 1:], unseen_rain[0, 1:])
    assert float(normalized.abs().max()) <= 1.0

    snapshot = agent.full_training_state()
    agent.observation_normalizer.mean[0] = 99.0
    agent.load_full_training_state(snapshot)
    np.testing.assert_array_equal(
        agent.observation_normalizer.identity_mask,
        np.array([False, True, True, True]),
    )
    artifact = tmp_path / "identity_sac.pt"
    agent.save(artifact)
    loaded = RecurrentSACAgent.load(artifact)
    np.testing.assert_array_equal(
        loaded.observation_normalizer.identity_mask,
        agent.observation_normalizer.identity_mask,
    )
    torch.testing.assert_close(
        loaded.observation_normalizer.normalize_tensor(unseen_rain), normalized
    )


def _episode(identifier: float, length: int = 8, observation_dim: int = 4) -> Episode:
    time = np.arange(length + 1, dtype=np.float32)
    observations = np.stack(
        (
            np.full_like(time, identifier),
            time,
            np.sin(time),
            np.cos(time),
        ),
        axis=1,
    )[:, :observation_dim]
    coordinates = np.stack(
        (
            0.05 * np.sin(time[:-1]),
            0.10 * np.cos(time[:-1]),
            0.04 * np.sin(0.5 * time[:-1]),
        ),
        axis=1,
    )
    actions = ActionTransform.market_controls().coordinates_to_physical(coordinates)
    rewards = -np.square(actions[:, 0] - 1.1).astype(np.float32)
    dones = np.zeros(length, dtype=np.float32)
    dones[-1] = 1.0
    return Episode(observations, actions, rewards, dones)


def test_effective_horizon_respects_done_padding_and_gamma() -> None:
    dones = torch.tensor(
        [[0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    mask = torch.tensor(
        [[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    horizon = effective_horizon_lengths(dones, mask, gamma=0.5)
    torch.testing.assert_close(
        horizon,
        torch.tensor(
            [[1.75, 1.5, 1.0, 0.0], [1.5, 1.0, 0.0, 0.0]],
            dtype=torch.float32,
        ),
    )


def test_effective_horizon_normalization_equalizes_equal_relative_errors() -> None:
    dones = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
    mask = torch.ones_like(dones)
    horizon = effective_horizon_lengths(dones, mask, gamma=1.0)
    relative_error = 0.25
    raw_error = relative_error * horizon
    normalized_squared = (raw_error / horizon.clamp_min(1.0)).square()
    torch.testing.assert_close(
        normalized_squared,
        torch.full_like(normalized_squared, relative_error**2),
    )
    assert not torch.allclose(raw_error.square(), normalized_squared)


def _configure_critic_geometry(
    agent: RecurrentSACAgent,
    center: np.ndarray | None = None,
    scale: np.ndarray | None = None,
) -> None:
    agent.configure_critic_action_reference(
        np.zeros(agent.action_dim) if center is None else center,
        np.full(agent.action_dim, 0.2) if scale is None else scale,
    )


def _paired_train_fit_episodes(
    *,
    observation_dim: int = 4,
    length: int = 3,
) -> list[Episode]:
    """Small matched-CRN cache with one center and symmetric axis probes."""
    center = np.zeros(3, dtype=np.float32)
    offsets = [center.copy()]
    for axis in range(3):
        plus = center.copy()
        plus[axis] = 0.2
        offsets.extend((plus, -plus))
    episodes: list[Episode] = []
    for seed_index, seed in enumerate((101, 211)):
        deviation_day = seed_index % length
        time = np.arange(length + 1, dtype=np.float32)
        observations = np.stack(
            (
                np.full_like(time, float(seed_index)),
                time,
                np.sin(time),
                np.cos(time),
            ),
            axis=1,
        )[:, :observation_dim]
        for support_index, offset in enumerate(offsets):
            actions = np.zeros((length, 3), dtype=np.float32)
            actions[deviation_day] = offset
            rewards = np.full(length, 2.0 + seed_index, dtype=np.float32)
            rewards[deviation_day] += float(
                np.array([1.5, -0.75, 0.5], dtype=np.float32) @ offset
                - 4.0 * np.square(offset).sum()
            )
            dones = np.zeros(length, dtype=np.float32)
            dones[-1] = 1.0
            episodes.append(
                Episode(
                    observations=observations.copy(),
                    actions=actions,
                    rewards=rewards,
                    dones=dones,
                    metadata={
                        "group_id": "l0_train_local_support",
                        "profile_split": "train_fit",
                        "heldout_for_gate": False,
                        "seed": seed,
                        "support_index": support_index,
                        "deviation_day": deviation_day,
                    },
                )
            )
    return episodes


def _nested_assert_equal(actual, expected) -> None:
    if torch.is_tensor(expected):
        assert torch.equal(actual, expected)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            _nested_assert_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _nested_assert_equal(actual_item, expected_item)
    else:
        assert actual == expected


def test_absolute_prefit_excludes_replicated_local_support_rows() -> None:
    config = RecurrentSACConfig(hidden_dim=8, batch_size=1, seed=92)
    first = RecurrentSACAgent(4, config=config)
    second = RecurrentSACAgent(4, config=copy.deepcopy(config))
    for agent in (first, second):
        _configure_critic_geometry(agent, scale=np.full(3, 0.2))
    absolute = _episode(2.0, length=3)
    absolute.actions[:] = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    absolute.metadata = {
        "group_id": "l0_train_absolute_fit",
        "seed": 700,
        "support_index": 0,
        "deviation_day": 0,
    }
    first.add_episode(absolute, actions_are_physical=True)
    second.add_episode(absolute, actions_are_physical=True)
    for duplicate in range(15):
        support = _episode(50.0 + duplicate, length=3)
        support.rewards[:] = 1.0e4 + duplicate
        support.metadata = {
            "group_id": "l0_train_local_support",
            "seed": 900,
            "support_index": duplicate,
            "deviation_day": 0,
        }
        second.add_episode(
            support,
            actions_are_physical=True,
            update_normalizer=False,
        )
    torch.manual_seed(123)
    fit_first = first.fit_critics_monte_carlo(steps=2, paired_advantage_steps=0)
    torch.manual_seed(123)
    fit_second = second.fit_critics_monte_carlo(steps=2, paired_advantage_steps=0)
    assert fit_first["absolute_fit_contract"]["transition_count"] == 3
    assert fit_second["absolute_fit_contract"]["transition_count"] == 3
    assert fit_second["absolute_fit_contract"]["paired_support_rows_excluded"]
    scale_contract = fit_second["absolute_loss_scale_contract"]
    assert scale_contract["q_values_and_returns_reported_in_raw_reward_units"]
    assert not scale_contract["paired_advantage_objective_normalized"]
    assert not scale_contract["online_td_objective_normalized"]
    assert (
        fit_second["two_phase_contract"]["phase1"]["objective"]
        == "warm-reference_effective-horizon-normalized_mc_only"
    )
    for snapshot in fit_second["calibration_snapshots"].values():
        assert "absolute_mc_loss" in snapshot
        assert "effective_horizon_normalized_absolute_loss" in snapshot
    assert fit_second["final_absolute_mc_loss"] == pytest.approx(
        fit_first["final_absolute_mc_loss"], rel=1.0e-6, abs=1.0e-6
    )
    for left, right in zip(first.critic1.parameters(), second.critic1.parameters()):
        torch.testing.assert_close(left, right, rtol=1.0e-6, atol=1.0e-6)


def test_uncapped_log_positive_action_round_trip() -> None:
    transform = ActionTransform.market_controls()
    coordinates = np.array([8.0, -2.5, 6.0])
    physical = transform.coordinates_to_physical(coordinates)
    assert physical[0] > 2_000.0
    assert physical[2] > 400.0
    assert physical[1] == pytest.approx(-2.5)
    np.testing.assert_allclose(transform.physical_to_coordinates(physical), coordinates)
    with pytest.raises(FloatingPointError):
        transform.coordinates_to_physical(np.array([1_000.0, 0.0, 0.0]))
    with pytest.raises(NumericalPolicyActionError) as underflow:
        transform.coordinates_to_physical(np.array([-1_000.0, 0.0, 0.0]))
    assert underflow.value.coordinates[0] == pytest.approx(-1_000.0)


def test_sequence_replay_never_crosses_episode_boundary() -> None:
    replay = SequenceReplayBuffer(capacity_transitions=100, seed=7)
    transform = ActionTransform.market_controls()
    for identifier in (11.0, 22.0):
        episode = _episode(identifier, length=9)
        replay.add(
            Episode(
                episode.observations,
                transform.physical_to_coordinates(episode.actions),
                episode.rewards,
                episode.dones,
            )
        )
    batch = replay.sample(
        batch_size=32,
        burn_in=2,
        learn_length=4,
        device=torch.device("cpu"),
    )
    observations = batch["observations"].cpu().numpy()
    sequence_steps = batch["sequence_steps"].cpu().numpy()
    for row, steps in enumerate(sequence_steps):
        sequence = observations[row, : int(steps) + 1]
        assert np.all(sequence[:, 0] == sequence[0, 0])
        np.testing.assert_allclose(np.diff(sequence[:, 1]), 1.0)
        start = int(batch["learn_start"][row])
        actual_burn = int(batch["actual_burn"][row])
        assert actual_burn == start
        assert int(batch["context_start"][row]) == 0
        assert int(batch["configured_burn_in"][row]) == 2
        indices = batch["transition_indices"][row].cpu().numpy()
        loss = batch["loss_mask"][row].bool().cpu().numpy()
        assert int(indices[loss][0]) == start
    assert batch["observations"].shape[0] == 32
    assert batch["observations"].shape[2] == 4
    assert batch["actions"].shape[2] == 3


def test_replay_preserves_local_probe_metadata() -> None:
    agent = RecurrentSACAgent(
        4,
        config=RecurrentSACConfig(hidden_dim=8, burn_in=1, learn_length=2, seed=8),
    )
    episode = _episode(2.0, length=5)
    episode.metadata = {
        "group_id": "local",
        "support_index": 3,
        "deviation_day": 4,
    }
    agent.add_episode(episode, actions_are_physical=True)
    stored = agent.replay.episodes[0]
    assert stored.metadata == episode.metadata
    episode.metadata["support_index"] = 99
    assert stored.metadata["support_index"] == 3


def test_replay_windows_cover_every_transition_with_true_prefix_and_terminal() -> None:
    replay = SequenceReplayBuffer(capacity_transitions=100, seed=19)
    transform = ActionTransform.market_controls()
    episode = _episode(7.0, length=9)
    stored = Episode(
        episode.observations,
        transform.physical_to_coordinates(episode.actions),
        episode.rewards,
        episode.dones,
    )
    replay.add(stored)
    first_targets = []
    for learn_start in range(stored.length):
        batch = replay.transition_window(
            0,
            learn_start,
            burn_in=3,
            learn_length=4,
            device=torch.device("cpu"),
        )
        mask = batch["loss_mask"][0].bool()
        indices = batch["transition_indices"][0][mask].cpu().numpy()
        expected = np.arange(learn_start, min(stored.length, learn_start + 4))
        np.testing.assert_array_equal(indices, expected)
        first_targets.append(int(indices[0]))
        assert int(batch["actual_burn"][0]) == learn_start
        assert int(batch["context_start"][0]) == 0
        assert int(batch["configured_burn_in"][0]) == 3
        sequence_mask = batch["sequence_mask"][0].bool()
        sequence_indices = batch["transition_indices"][0][sequence_mask].cpu().numpy()
        np.testing.assert_allclose(
            batch["actions"][0][sequence_mask].cpu().numpy(),
            stored.actions[sequence_indices],
        )
        np.testing.assert_allclose(
            batch["rewards"][0][sequence_mask].cpu().numpy(),
            stored.rewards[sequence_indices],
        )
    assert first_targets == list(range(stored.length))
    terminal = replay.transition_window(
        0,
        stored.length - 1,
        burn_in=3,
        learn_length=4,
        device=torch.device("cpu"),
    )
    terminal_mask = terminal["loss_mask"][0].bool()
    assert terminal["dones"][0][terminal_mask].tolist() == [1.0]


def test_overlap_correction_gives_every_transition_equal_expected_weight() -> None:
    replay = SequenceReplayBuffer(capacity_transitions=100, seed=29)
    transform = ActionTransform.market_controls()
    episode = _episode(3.0, length=11)
    replay.add(
        Episode(
            episode.observations,
            transform.physical_to_coordinates(episode.actions),
            episode.rewards,
            episode.dones,
        )
    )
    inclusion_weight = np.zeros(episode.length, dtype=np.float64)
    for learn_start in range(episode.length):
        batch = replay.transition_window(
            0,
            learn_start,
            burn_in=3,
            learn_length=4,
            device=torch.device("cpu"),
        )
        mask = batch["loss_mask"][0].bool()
        indices = batch["transition_indices"][0][mask].cpu().numpy()
        weights = batch["loss_weights"][0][mask].cpu().numpy()
        inclusion_weight[indices] += weights
    np.testing.assert_allclose(inclusion_weight, np.ones(episode.length), atol=1e-7)

    # Exercise the actual SAC loss reducer over the complete distribution of
    # learn starts.  A unit basis loss at every global transition must receive
    # exactly the uniform episode coefficient 1 / T.
    stored = list(replay.episodes)[0]
    windows = [
        replay._training_window(
            stored,
            learn_start=learn_start,
            burn_in=3,
            learn_length=4,
        )
        for learn_start in range(stored.length)
    ]
    batch = replay._collate_windows(windows, device=torch.device("cpu"))
    coefficients = []
    for target in range(stored.length):
        basis_loss = (
            batch["transition_indices"] == target
        ).to(dtype=torch.float32)
        coefficients.append(
            float(
                RecurrentSACAgent._replay_ht_mean(
                    basis_loss, batch["loss_weights"]
                )
            )
        )
    np.testing.assert_allclose(
        coefficients,
        np.full(stored.length, 1.0 / stored.length),
        rtol=1.0e-6,
        atol=1.0e-7,
    )


def test_mid_episode_window_matches_exact_full_history_actor_and_critic_features() -> None:
    agent = RecurrentSACAgent(
        4,
        config=RecurrentSACConfig(
            hidden_dim=16,
            batch_size=1,
            burn_in=3,
            learn_length=2,
            seed=37,
        ),
    )
    _configure_critic_geometry(agent)
    agent.add_episode(_episode(6.0, length=10), actions_are_physical=True)
    learn_start = 7  # Deliberately beyond configured burn-in.
    window = agent.replay.transition_window(
        0,
        learn_start,
        burn_in=3,
        learn_length=2,
        device=agent.device,
    )
    full = agent.replay.full_episode_batch(agent.device)
    window_context = agent._context(
        window["observations"],
        window["previous_actions"],
        window["previous_rewards"],
        window["previous_dones"],
    )
    full_context = agent._context(
        full["observations"],
        full["previous_actions"],
        full["previous_rewards"],
        full["previous_dones"],
    )
    with torch.no_grad():
        window_actor, _ = agent.actor.features(window_context, burn_in=0)
        full_actor, _ = agent.actor.features(full_context, burn_in=0)
        window_critic, _ = agent.critic1.features(window_context, burn_in=0)
        full_critic, _ = agent.critic1.features(full_context, burn_in=0)
        window_target, _ = agent.target_critic1.features(window_context, burn_in=0)
        full_target, _ = agent.target_critic1.features(full_context, burn_in=0)
        window_mean, _ = agent.actor.distribution(window_actor)
        full_mean, _ = agent.actor.distribution(full_actor)
        stored_action = full["actions"][:, learn_start : learn_start + 1]
        window_q = agent.critic1.q_from_features(
            window_critic[:, learn_start : learn_start + 1], stored_action
        )
        full_q = agent.critic1.q_from_features(
            full_critic[:, learn_start : learn_start + 1], stored_action
        )
    torch.testing.assert_close(
        window_actor[:, learn_start], full_actor[:, learn_start]
    )
    torch.testing.assert_close(
        window_critic[:, learn_start], full_critic[:, learn_start]
    )
    torch.testing.assert_close(
        window_target[:, learn_start + 1], full_target[:, learn_start + 1]
    )
    torch.testing.assert_close(
        window_mean[:, learn_start], full_mean[:, learn_start]
    )
    torch.testing.assert_close(window_q, full_q)


def test_recurrent_sac_update_is_finite_and_changes_actor() -> None:
    config = RecurrentSACConfig(
        hidden_dim=16,
        replay_capacity=200,
        batch_size=4,
        burn_in=2,
        learn_length=4,
        gamma=0.95,
        seed=13,
    )
    agent = RecurrentSACAgent(4, config=config)
    _configure_critic_geometry(agent)
    for identifier in (1.0, 2.0, 3.0, 4.0):
        agent.add_episode(_episode(identifier), actions_are_physical=True)
    before = [parameter.detach().clone() for parameter in agent.actor.parameters()]
    metrics = agent.update()
    assert all(np.isfinite(value) for value in metrics.values())
    assert metrics["temperature"] > 0.0
    assert any(
        not torch.equal(previous, current.detach())
        for previous, current in zip(before, agent.actor.parameters())
    )


def test_online_td_update_does_not_use_offline_horizon_normalizer(monkeypatch) -> None:
    agent = RecurrentSACAgent(
        4,
        config=RecurrentSACConfig(
            hidden_dim=8,
            replay_capacity=100,
            batch_size=2,
            burn_in=1,
            learn_length=2,
            gamma=0.95,
            seed=131,
        ),
    )
    _configure_critic_geometry(agent)
    for identifier in (1.0, 2.0):
        agent.add_episode(_episode(identifier), actions_are_physical=True)

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("online TD called offline horizon normalization")

    monkeypatch.setattr("recurrent_sac.effective_horizon_lengths", forbidden)
    metrics = agent.update()
    assert np.isfinite(metrics["critic_loss"])


def test_critic_gate_freezes_actor_temperature_but_not_critics() -> None:
    agent = RecurrentSACAgent(
        4,
        config=RecurrentSACConfig(
            hidden_dim=16,
            replay_capacity=200,
            batch_size=4,
            burn_in=2,
            learn_length=4,
            gamma=1.0,
            seed=113,
        ),
    )
    _configure_critic_geometry(agent)
    for identifier in (1.0, 2.0, 3.0, 4.0):
        agent.add_episode(_episode(identifier), actions_are_physical=True)
    agent.configure_safe_policy_improvement(
        actor_updates_enabled=False,
        warm_anchor_coordinates=np.zeros(3),
        anchor_coefficient=2.0,
        anchor_decay_updates=4.0,
        anchor_scale=np.full(3, 0.1),
        actor_update_interval=2,
    )
    actor_before = [parameter.detach().clone() for parameter in agent.actor.parameters()]
    critic_before = [parameter.detach().clone() for parameter in agent.critic1.parameters()]
    temperature_before = agent.log_temperature.detach().clone()
    metrics = agent.update()
    assert metrics["actor_update_applied"] == 0.0
    assert all(
        torch.equal(previous, current.detach())
        for previous, current in zip(actor_before, agent.actor.parameters())
    )
    assert torch.equal(temperature_before, agent.log_temperature.detach())
    assert any(
        not torch.equal(previous, current.detach())
        for previous, current in zip(critic_before, agent.critic1.parameters())
    )


def test_train_fit_paired_cache_is_immutable_and_rejects_heldout_gate_data() -> None:
    agent = RecurrentSACAgent(
        4,
        config=RecurrentSACConfig(hidden_dim=12, batch_size=2, seed=331),
    )
    _configure_critic_geometry(agent)
    episodes = _paired_train_fit_episodes()
    original_first_action = episodes[0].actions.copy()
    contract = agent.configure_paired_crn_cache(
        episodes,
        source_group_id="l0_train_local_support",
        paired_advantage_weight=1.0,
    )
    assert contract["profile_split"] == "train_fit"
    assert not contract["heldout_gate_episodes_used"]
    assert contract["arrays_writeable"] is False
    assert all(
        not array.flags.writeable
        for episode in agent._paired_crn_cache
        for array in (
            episode.observations,
            episode.actions,
            episode.rewards,
            episode.dones,
        )
    )
    episodes[0].actions[0, 0] = 99.0
    np.testing.assert_array_equal(
        agent._paired_crn_cache[0].actions, original_first_action
    )
    with pytest.raises(ValueError):
        agent._paired_crn_cache[0].actions[0, 0] = 1.0

    heldout = _paired_train_fit_episodes()
    for episode in heldout:
        episode.metadata["group_id"] = "l0_heldout_train_critic_gate"
        episode.metadata["profile_split"] = "heldout_train_gate"
        episode.metadata["heldout_for_gate"] = True
    other = RecurrentSACAgent(4, config=RecurrentSACConfig(hidden_dim=12, seed=337))
    _configure_critic_geometry(other)
    with pytest.raises(ValueError, match="TRAIN-fit"):
        other.configure_paired_crn_cache(
            heldout,
            source_group_id="l0_heldout_train_critic_gate",
            paired_advantage_weight=1.0,
        )


def test_paired_cache_can_only_be_replaced_by_explicit_stage_transition() -> None:
    agent = RecurrentSACAgent(
        4, config=RecurrentSACConfig(hidden_dim=12, batch_size=2, seed=339)
    )
    _configure_critic_geometry(agent)
    first = _paired_train_fit_episodes()
    agent.configure_paired_crn_cache(
        first,
        source_group_id="l0_train_local_support",
        paired_advantage_weight=1.0,
    )
    second = _paired_train_fit_episodes()
    for episode in second:
        episode.metadata["group_id"] = "l1_train_local_support"
        episode.metadata["seed"] = int(episode.metadata["seed"]) + 10_000
    with pytest.raises(RuntimeError, match="immutable"):
        agent.configure_paired_crn_cache(
            second,
            source_group_id="l1_train_local_support",
            paired_advantage_weight=1.0,
        )
    contract = agent.configure_paired_crn_cache(
        second,
        source_group_id="l1_train_local_support",
        paired_advantage_weight=1.0,
        replace_for_new_stage=True,
    )
    assert contract["stage_transition_replacement"]
    assert contract["replaced_source_group_id"] == "l0_train_local_support"
    assert contract["source_group_id"] == "l1_train_local_support"
    assert set(contract["seeds"]) == {
        int(episode.metadata["seed"]) for episode in second
    }


def test_online_paired_refresh_updates_only_advantage_heads() -> None:
    agent = RecurrentSACAgent(
        4,
        config=RecurrentSACConfig(
            hidden_dim=16,
            critic_learning_rate=5.0e-3,
            gradient_clip=50.0,
            seed=347,
        ),
    )
    _configure_critic_geometry(agent)
    episodes = _paired_train_fit_episodes()
    for episode in episodes:
        agent.add_episode(episode, actions_are_physical=False)
    agent.configure_paired_crn_cache(
        episodes,
        source_group_id="l0_train_local_support",
        paired_advantage_weight=1.0,
    )
    frozen_before = {
        "critic1_encoder": [
            parameter.detach().clone()
            for parameter in agent.critic1.encoder_parameters()
        ],
        "critic1_value": [
            parameter.detach().clone()
            for parameter in agent.critic1.value_head_parameters()
        ],
        "critic2_encoder": [
            parameter.detach().clone()
            for parameter in agent.critic2.encoder_parameters()
        ],
        "critic2_value": [
            parameter.detach().clone()
            for parameter in agent.critic2.value_head_parameters()
        ],
    }
    advantage_before = [
        parameter.detach().clone()
        for critic in (agent.critic1, agent.critic2)
        for parameter in critic.advantage_parameters()
    ]
    td_optimizer_before = {
        "critic1": copy.deepcopy(agent.critic1_optimizer.state_dict()),
        "critic2": copy.deepcopy(agent.critic2_optimizer.state_dict()),
    }
    targets_before = {
        name: copy.deepcopy(critic.state_dict())
        for name, critic in (
            ("target1", agent.target_critic1),
            ("target2", agent.target_critic2),
        )
    }
    report = agent.paired_advantage_refresh()
    assert report["performed"]
    assert report["source_split"] == "train_fit"
    assert not report["heldout_gate_episodes_used"]
    assert report["features_recomputed_and_detached"]
    assert report["encoder_value_parameters_unchanged"]
    assert report["dedicated_advantage_optimizers"]
    assert report["optimizer_step_performed"]
    for name, parameters in (
        ("critic1_encoder", agent.critic1.encoder_parameters()),
        ("critic1_value", agent.critic1.value_head_parameters()),
        ("critic2_encoder", agent.critic2.encoder_parameters()),
        ("critic2_value", agent.critic2.value_head_parameters()),
    ):
        assert all(
            torch.equal(before, after.detach())
            for before, after in zip(frozen_before[name], parameters)
        )
    assert any(
        not torch.equal(before, after.detach())
        for before, after in zip(
            advantage_before,
            [
                parameter
                for critic in (agent.critic1, agent.critic2)
                for parameter in critic.advantage_parameters()
            ],
        )
    )
    _nested_assert_equal(agent.target_critic1.state_dict(), targets_before["target1"])
    _nested_assert_equal(agent.target_critic2.state_dict(), targets_before["target2"])
    _nested_assert_equal(
        agent.critic1_optimizer.state_dict(), td_optimizer_before["critic1"]
    )
    _nested_assert_equal(
        agent.critic2_optimizer.state_dict(), td_optimizer_before["critic2"]
    )
    assert agent.critic1_advantage_optimizer.state
    assert agent.critic2_advantage_optimizer.state


def test_zero_weight_paired_refresh_is_literal_no_op_after_td_adam_moments() -> None:
    agent = RecurrentSACAgent(
        4,
        config=RecurrentSACConfig(
            hidden_dim=16,
            critic_learning_rate=5.0e-3,
            gradient_clip=50.0,
            seed=348,
        ),
    )
    _configure_critic_geometry(agent)
    episodes = _paired_train_fit_episodes()
    for episode in episodes:
        agent.add_episode(episode, actions_are_physical=False)
    agent.configure_paired_crn_cache(
        episodes,
        source_group_id="l0_train_local_support",
        paired_advantage_weight=0.0,
    )

    # Populate the ordinary TD optimizers' moments on every critic parameter,
    # including the advantage heads.  The historical bug reused these Adam
    # instances, so Adam.step() moved the heads even when the auxiliary loss
    # was multiplied by zero.
    batch = agent.replay.full_episode_batch(agent.device)
    context = agent._context(
        batch["observations"],
        batch["previous_actions"],
        batch["previous_rewards"],
        batch["previous_dones"],
    )
    features1, _ = agent.critic1.features(context, burn_in=0)
    features2, _ = agent.critic2.features(context, burn_in=0)
    actions = batch["actions"]
    loss1 = agent.critic1.q_from_features(features1[:, :-1], actions).square().mean()
    loss2 = agent.critic2.q_from_features(features2[:, :-1], actions).square().mean()
    agent.critic1_optimizer.zero_grad(set_to_none=True)
    agent.critic2_optimizer.zero_grad(set_to_none=True)
    (loss1 + loss2).backward()
    agent.critic1_optimizer.step()
    agent.critic2_optimizer.step()
    assert any(
        parameter in agent.critic1_optimizer.state
        for parameter in agent.critic1.advantage_parameters()
    )
    assert any(
        parameter in agent.critic2_optimizer.state
        for parameter in agent.critic2.advantage_parameters()
    )

    parameters_before = [
        parameter.detach().clone()
        for critic in (agent.critic1, agent.critic2)
        for parameter in critic.advantage_parameters()
    ]
    td_optimizers_before = (
        copy.deepcopy(agent.critic1_optimizer.state_dict()),
        copy.deepcopy(agent.critic2_optimizer.state_dict()),
    )
    auxiliary_optimizers_before = (
        copy.deepcopy(agent.critic1_advantage_optimizer.state_dict()),
        copy.deepcopy(agent.critic2_advantage_optimizer.state_dict()),
    )
    report = agent.paired_advantage_refresh()
    assert report["zero_weight_literal_no_op"]
    assert not report["optimizer_step_performed"]
    assert report["gradient_norm_q1"] == 0.0
    assert report["gradient_norm_q2"] == 0.0
    assert all(
        torch.equal(before, after.detach())
        for before, after in zip(
            parameters_before,
            [
                parameter
                for critic in (agent.critic1, agent.critic2)
                for parameter in critic.advantage_parameters()
            ],
        )
    )
    _nested_assert_equal(
        agent.critic1_optimizer.state_dict(), td_optimizers_before[0]
    )
    _nested_assert_equal(
        agent.critic2_optimizer.state_dict(), td_optimizers_before[1]
    )
    _nested_assert_equal(
        agent.critic1_advantage_optimizer.state_dict(),
        auxiliary_optimizers_before[0],
    )
    _nested_assert_equal(
        agent.critic2_advantage_optimizer.state_dict(),
        auxiliary_optimizers_before[1],
    )


def test_safe_update_orders_refresh_target_gate_then_actor(monkeypatch) -> None:
    agent = RecurrentSACAgent(
        4,
        config=RecurrentSACConfig(
            hidden_dim=16,
            batch_size=4,
            burn_in=1,
            learn_length=2,
            seed=349,
        ),
    )
    _configure_critic_geometry(agent)
    episodes = _paired_train_fit_episodes()
    for episode in episodes:
        agent.add_episode(episode, actions_are_physical=False)
    agent.configure_paired_crn_cache(
        episodes,
        source_group_id="l0_train_local_support",
        paired_advantage_weight=1.0,
    )
    agent.capture_last_passing_state("warm")
    target_before = copy.deepcopy(agent.target_critic1.state_dict())
    events = []
    original_refresh = agent.paired_advantage_refresh
    original_polyak = agent._polyak_update_targets
    original_actor_step = agent.actor_optimizer.step

    def refresh():
        result = original_refresh()
        events.append("paired_refresh")
        return result

    def polyak():
        original_polyak()
        events.append("target_polyak")

    def gate(current_agent, update_index):
        assert current_agent is agent
        assert update_index == 1
        events.append("same_version_gate")
        assert any(
            not torch.equal(target_before[name], value)
            for name, value in current_agent.target_critic1.state_dict().items()
        )
        return {"passed": True, "heldout_gate_episodes_used": True}

    def actor_step(*args, **kwargs):
        events.append("actor_temperature_update")
        return original_actor_step(*args, **kwargs)

    monkeypatch.setattr(agent, "paired_advantage_refresh", refresh)
    monkeypatch.setattr(agent, "_polyak_update_targets", polyak)
    monkeypatch.setattr(agent.actor_optimizer, "step", actor_step)
    result = agent.update(critic_gate=gate)
    assert result["actor_update_applied"] == 1.0
    assert events == [
        "paired_refresh",
        "target_polyak",
        "same_version_gate",
        "actor_temperature_update",
    ]
    assert result["last_passing_checkpoint"]["capture_count"] == 2


def test_failed_same_version_gate_restores_complete_last_passing_state() -> None:
    agent = RecurrentSACAgent(
        4,
        config=RecurrentSACConfig(
            hidden_dim=16,
            batch_size=4,
            burn_in=1,
            learn_length=2,
            seed=353,
        ),
    )
    _configure_critic_geometry(agent)
    episodes = _paired_train_fit_episodes()
    for episode in episodes:
        agent.add_episode(episode, actions_are_physical=False)
    agent.configure_paired_crn_cache(
        episodes,
        source_group_id="l0_train_local_support",
        paired_advantage_weight=1.0,
    )
    agent.capture_last_passing_state("warm")
    first = agent.update(critic_gate=lambda *_: {"passed": True})
    assert first["actor_update_applied"] == 1.0
    checkpoint = copy.deepcopy(agent._last_passing_state)
    checkpoint_label = agent._last_passing_label

    failed = agent.update(critic_gate=lambda *_: {"passed": False})
    assert failed["actor_update_applied"] == 0.0
    assert failed["terminal_training_failure"] == 1.0
    assert failed["rollback_performed"] == 1.0
    assert failed["rollback"]["restored_label"] == checkpoint_label
    assert agent.training_terminal_failure
    assert not agent.actor_updates_enabled
    for key, module in (
        ("actor", agent.actor),
        ("critic1", agent.critic1),
        ("critic2", agent.critic2),
        ("target_critic1", agent.target_critic1),
        ("target_critic2", agent.target_critic2),
    ):
        _nested_assert_equal(module.state_dict(), checkpoint[key])
    for key, optimizer in (
        ("actor_optimizer", agent.actor_optimizer),
        ("critic1_optimizer", agent.critic1_optimizer),
        ("critic2_optimizer", agent.critic2_optimizer),
        (
            "critic1_advantage_optimizer",
            agent.critic1_advantage_optimizer,
        ),
        (
            "critic2_advantage_optimizer",
            agent.critic2_advantage_optimizer,
        ),
        ("temperature_optimizer", agent.temperature_optimizer),
    ):
        _nested_assert_equal(optimizer.state_dict(), checkpoint[key])
    torch.testing.assert_close(agent.log_temperature, checkpoint["log_temperature"])
    _nested_assert_equal(
        agent.observation_normalizer.state_dict(),
        checkpoint["observation_normalizer"],
    )
    assert agent.updates == checkpoint["updates"]
    assert agent.actor_updates_applied == checkpoint["actor_updates_applied"]
    assert len(agent.replay.episodes) == len(checkpoint["replay"].episodes)
    assert agent.replay.transitions == checkpoint["replay"].transitions
    for actual, expected in zip(
        agent.replay.episodes, checkpoint["replay"].episodes
    ):
        _nested_assert_equal(actual.observations, expected.observations)
        _nested_assert_equal(actual.actions, expected.actions)
        _nested_assert_equal(actual.rewards, expected.rewards)
        _nested_assert_equal(actual.dones, expected.dones)
        _nested_assert_equal(actual.metadata, expected.metadata)
    _nested_assert_equal(
        agent.replay.rng.bit_generator.state,
        checkpoint["replay"].rng.bit_generator.state,
    )
    _nested_assert_equal(agent.rng.bit_generator.state, checkpoint["agent_rng_state"])
    assert torch.equal(torch.get_rng_state(), checkpoint["torch_rng_state"])
    assert all(
        not array.flags.writeable
        for episode in agent._paired_crn_cache
        for array in (episode.observations, episode.actions, episode.rewards, episode.dones)
    )
    assert agent.paired_crn_cache_contract()["heldout_gate_episodes_used"] is False
    with pytest.raises(RuntimeError, match="terminally failed"):
        agent.update(critic_gate=lambda *_: {"passed": True})


def test_gate_miss_with_economic_controller_restores_verified_incumbent() -> None:
    agent = RecurrentSACAgent(
        4,
        config=RecurrentSACConfig(
            hidden_dim=16,
            batch_size=4,
            burn_in=1,
            learn_length=2,
            seed=357,
        ),
    )
    _configure_critic_geometry(agent)
    episodes = _paired_train_fit_episodes()
    for episode in episodes:
        agent.add_episode(episode, actions_are_physical=False)
    agent.configure_paired_crn_cache(
        episodes,
        source_group_id="l0_train_local_support",
        paired_advantage_weight=1.0,
    )
    agent.capture_last_passing_state("accepted_critic")
    agent.initialize_economic_incumbent("accepted_policy")
    incumbent = agent.economic_incumbent_transaction_state()

    passed = agent.update(critic_gate=lambda *_: {"passed": True})
    assert passed["actor_update_applied"] == 1.0
    assert agent.last_passing_state_contract()["label"] == "same_version_gate_update_1"

    failed = agent.update(
        critic_gate=lambda *_: {
            "passed": False,
            "gate": {"coverage_passed": True},
            "diagnostics": {"finite": True},
        },
        restore_economic_incumbent_on_gate_failure=True,
    )
    assert failed["terminal_training_failure"] == 0.0
    assert failed["critic_proposal_rejection"] == 1.0
    assert not failed["rollback"]["terminal"]
    assert failed["rollback"]["restored_label"] == "accepted_policy"
    assert not agent.training_terminal_failure
    restored = agent.training_transaction_state()
    for key in (
        "actor",
        "critic1",
        "critic2",
        "target_critic1",
        "target_critic2",
        "actor_optimizer",
        "critic1_optimizer",
        "critic2_optimizer",
        "critic1_advantage_optimizer",
        "critic2_advantage_optimizer",
        "temperature_optimizer",
        "observation_normalizer",
        "agent_rng_state",
        "torch_rng_state",
    ):
        _nested_assert_equal(restored["learner"][key], incumbent["learner"][key])
    assert restored["critic_safe_label"] == incumbent["critic_safe_label"]
    assert restored["learner"]["actor_updates_applied"] == 0


def test_save_load_persists_durable_economic_incumbent(tmp_path: Path) -> None:
    agent = RecurrentSACAgent(
        4,
        config=RecurrentSACConfig(hidden_dim=12, batch_size=2, seed=358),
    )
    _configure_critic_geometry(agent)
    agent.add_episode(_episode(1.0), actions_are_physical=True)
    agent.capture_last_passing_state("verified_critic")
    agent.initialize_economic_incumbent("verified_actor")
    expected = agent.economic_incumbent_transaction_state()
    artifact = tmp_path / "incumbent.pt"
    agent.save(artifact)

    loaded = RecurrentSACAgent.load(artifact)
    assert loaded.economic_incumbent_contract()["available"]
    assert loaded.economic_incumbent_contract()["label"] == "verified_actor"
    actual = loaded.economic_incumbent_transaction_state()
    _nested_assert_equal(actual["learner"]["actor"], expected["learner"]["actor"])
    _nested_assert_equal(
        actual["learner"]["observation_normalizer"],
        expected["learner"]["observation_normalizer"],
    )
    assert actual["critic_safe_label"] == "verified_critic"


def test_economic_incumbent_rollback_is_full_exact_nonterminal_and_distinct() -> None:
    agent = RecurrentSACAgent(
        4,
        config=RecurrentSACConfig(hidden_dim=12, batch_size=2, seed=359),
    )
    _configure_critic_geometry(agent)
    agent.add_episode(_episode(1.0), actions_are_physical=True)
    agent.capture_last_passing_state("critic_safe_incumbent")
    contract = agent.initialize_economic_incumbent("warm_incumbent")
    assert contract["distinct_from_critic_safe_pointer"]
    expected = agent.economic_incumbent_transaction_state()

    with torch.no_grad():
        next(agent.actor.parameters()).add_(3.0)
        next(agent.critic1.parameters()).sub_(2.0)
    agent.add_episode(_episode(9.0), actions_are_physical=True)
    agent.rng.random()
    agent.replay.rng.random()
    torch.rand(4)
    agent.actor_updates_applied += 7
    agent.capture_last_passing_state("candidate_critic_safe")
    restored = agent.restore_economic_incumbent("harmful actor proposal")
    assert restored["performed"]
    assert not restored["terminal"]
    assert restored["critic_safe_pointer_restored"]
    assert not agent.training_terminal_failure
    assert agent.last_passing_state_contract()["label"] == "critic_safe_incumbent"
    actual = agent.training_transaction_state()
    for key in (
        "actor",
        "critic1",
        "critic2",
        "target_critic1",
        "target_critic2",
        "actor_optimizer",
        "critic1_optimizer",
        "critic2_optimizer",
        "critic1_advantage_optimizer",
        "critic2_advantage_optimizer",
        "temperature_optimizer",
        "observation_normalizer",
        "agent_rng_state",
        "torch_rng_state",
    ):
        _nested_assert_equal(actual["learner"][key], expected["learner"][key])
    assert actual["learner"]["updates"] == expected["learner"]["updates"]
    assert actual["learner"]["actor_updates_applied"] == expected["learner"][
        "actor_updates_applied"
    ]
    assert len(agent.replay.episodes) == len(expected["learner"]["replay"].episodes)
    _nested_assert_equal(
        agent.replay.rng.bit_generator.state,
        expected["learner"]["replay"].rng.bit_generator.state,
    )
    assert agent.economic_incumbent_contract()["label"] == "warm_incumbent"


def test_policy_fallback_keeps_current_normalizer_and_uses_delayed_actor() -> None:
    agent = RecurrentSACAgent(
        4,
        config=RecurrentSACConfig(
            hidden_dim=16, batch_size=2, burn_in=1, learn_length=2, seed=127
        ),
    )
    _configure_critic_geometry(agent)
    for identifier in (1.0, 2.0):
        agent.add_episode(_episode(identifier), actions_are_physical=True)
    agent.configure_safe_policy_improvement(
        actor_updates_enabled=True,
        warm_anchor_coordinates=np.zeros(3),
        anchor_coefficient=1.0,
        anchor_decay_updates=1.0,
        anchor_scale=np.full(3, 0.2),
        actor_update_interval=2,
    )
    snapshot = agent.policy_training_state()
    agent.observation_normalizer.update(np.full((3, 4), 99.0))
    current_normalizer = agent.observation_normalizer.state_dict()
    first = agent.update()
    second = agent.update()
    third = agent.update()
    assert first["actor_update_applied"] == 0.0
    assert second["actor_update_applied"] == 1.0
    assert second["warm_anchor_coefficient"] == pytest.approx(1.0)
    assert third["warm_anchor_coefficient"] == pytest.approx(1.0)
    agent.accepted_actor_blocks = 1
    fourth = agent.update()
    assert fourth["warm_anchor_coefficient"] == 0.0
    agent.load_policy_training_state(snapshot)
    np.testing.assert_array_equal(
        agent.observation_normalizer.mean, current_normalizer["mean"]
    )
    assert agent.observation_normalizer.count == current_normalizer["count"]


def test_update_targets_opening_through_terminal_transitions(monkeypatch) -> None:
    config = RecurrentSACConfig(
        hidden_dim=16,
        replay_capacity=100,
        batch_size=1,
        burn_in=3,
        learn_length=9,
        gamma=1.0,
        seed=23,
    )
    agent = RecurrentSACAgent(4, config=config)
    _configure_critic_geometry(agent)
    agent.add_episode(_episode(4.0, length=9), actions_are_physical=True)
    full = agent.replay.transition_window(
        0,
        0,
        burn_in=3,
        learn_length=9,
        device=agent.device,
    )
    monkeypatch.setattr(agent.replay, "sample", lambda **kwargs: full)
    metrics = agent.update()
    assert metrics["loss_transition_count"] == 9.0
    assert metrics["minimum_loss_transition_index"] == 0.0
    assert metrics["maximum_loss_transition_index"] == 8.0
    assert metrics["episode_opening_targets"] == 1.0
    assert metrics["terminal_targets"] == 1.0


def test_behavior_clone_matches_fixed_rule_from_every_reset_state() -> None:
    agent = RecurrentSACAgent(
        4,
        config=RecurrentSACConfig(
            hidden_dim=16,
            replay_capacity=100,
            batch_size=2,
            burn_in=3,
            learn_length=4,
            seed=31,
        ),
    )
    warm_episodes = [_episode(1.0, length=8), _episode(9.0, length=8)]
    for episode in warm_episodes:
        agent.add_episode(episode, actions_are_physical=True)
    target_action = np.array([1.0187, 0.0253, 0.7715], dtype=np.float64)
    target_coordinates = agent.action_transform.physical_to_coordinates(
        target_action
    )
    report = agent.behavior_clone(target_coordinates, steps=2)
    assert report["pre_clone"]["reset_max_abs_coordinate_error"] > 0.1
    assert report["reset_action_parity_passed"]
    assert report["final"]["full_episode_coordinate_mse"] < 1.0e-12
    assert report["final"]["warm_transition_count"] == 16
    for episode in warm_episodes:
        action, _, _ = agent.act(
            episode.observations[0], hidden=None, deterministic=True
        )
        np.testing.assert_allclose(action, target_action, rtol=1.0e-6, atol=1.0e-7)


def test_paired_advantage_prefit_learns_action_deltas_and_absolute_scale() -> None:
    agent = RecurrentSACAgent(
        4,
        config=RecurrentSACConfig(
            hidden_dim=24,
            batch_size=8,
            burn_in=0,
            learn_length=1,
            gamma=1.0,
            critic_learning_rate=3.0e-3,
            gradient_clip=50.0,
            seed=211,
        ),
    )
    _configure_critic_geometry(agent)
    offsets = [np.zeros(3)]
    for axis in range(3):
        direction = np.zeros(3)
        direction[axis] = 0.2
        offsets.extend((direction.copy(), -direction.copy()))
    coefficients = np.array([8.0, -5.0, 6.0])
    for seed_index, seed in enumerate((401, 509)):
        for support_index, offset in enumerate(offsets):
            reward = 50.0 + 3.0 * seed_index + float(coefficients @ offset)
            episode = Episode(
                observations=np.array(
                    [
                        [float(seed_index), 0.0, 0.0, 0.0],
                        [float(seed_index), 1.0, 0.0, 0.0],
                    ],
                    dtype=np.float32,
                ),
                actions=np.asarray(offset, dtype=np.float32).reshape(1, 3),
                rewards=np.array([reward], dtype=np.float32),
                dones=np.array([1.0], dtype=np.float32),
                metadata={
                        "group_id": "l0_train_local_support",
                    "seed": seed,
                    "support_index": support_index,
                    "deviation_day": 0,
                },
            )
            agent.add_episode(episode, actions_are_physical=False)
    report = agent.fit_critics_monte_carlo(
        steps=200, paired_advantage_steps=200, paired_advantage_weight=1.0
    )
    assert report["absolute_loss_scale_contract"][
        "paired_advantage_objective_normalized"
    ] is False
    assert report["absolute_loss_scale_contract"][
        "online_td_objective_normalized"
    ] is False
    assert report["two_phase_contract"]["phase2"][
        "paired_advantage_targets_remain_raw"
    ]
    initial = report["paired_advantage"]["initial"]
    final = report["paired_advantage"]["final"]
    assert final["informative_pair_count"] == 12
    assert final["q1"]["rmse"] < initial["q1"]["rmse"]
    assert final["q2"]["rmse"] < initial["q2"]["rmse"]
    assert final["qmin"]["directional_accuracy"] >= 5.0 / 6.0
    assert report["final_absolute_mc_loss"] < report["initial_absolute_mc_loss"]
    assert report["two_phase_contract"]["optimizer_moment_reset"]["performed"]
    assert report["two_phase_contract"]["phase1"]["steps"] == 200
    assert report["two_phase_contract"]["phase2"]["steps"] == 200


def test_structured_critic_geometry_center_hessian_state_and_extreme_actions() -> None:
    agent = RecurrentSACAgent(
        4,
        config=RecurrentSACConfig(hidden_dim=16, seed=271),
    )
    center = np.array([0.2, -0.1, 0.3], dtype=np.float64)
    scale = np.array([0.1, 0.2, 0.15], dtype=np.float64)
    _configure_critic_geometry(agent, center=center, scale=scale)
    critic = agent.critic1
    curvature_output = critic.curvature_head[-1]
    assert isinstance(curvature_output, torch.nn.Linear)
    with torch.no_grad():
        # The (1,0) entry creates a genuine cross term.  A nonzero output
        # weight makes the curvature depend on recurrent state as required.
        off_diagonal = int(
            torch.nonzero(
                (critic._tril_rows == 1) & (critic._tril_columns == 0)
            )[0]
        )
        curvature_output.bias[off_diagonal] = 0.35
        curvature_output.weight[off_diagonal].fill_(0.05)
    features = torch.stack(
        (torch.zeros(16), torch.linspace(-1.0, 1.0, 16)), dim=0
    )
    centers = torch.as_tensor(center, dtype=torch.float32).repeat(2, 1)
    components = critic.structured_components(features, centers)
    torch.testing.assert_close(
        components["advantage"], torch.zeros_like(components["advantage"]),
        rtol=0.0, atol=0.0,
    )
    torch.testing.assert_close(
        components["q"], components["value"], rtol=0.0, atol=0.0
    )
    assert not torch.equal(
        components["lower_cholesky"][0],
        components["lower_cholesky"][1],
    )

    action = torch.tensor(center, dtype=torch.float32, requires_grad=True)

    def q_at(candidate: torch.Tensor) -> torch.Tensor:
        return critic.q_from_features(features[:1], candidate.reshape(1, 3)).squeeze()

    hessian = torch.autograd.functional.hessian(q_at, action)
    torch.testing.assert_close(hessian, hessian.transpose(0, 1), rtol=1e-5, atol=1e-5)
    assert float(torch.linalg.eigvalsh(hessian).max()) < 0.0
    assert abs(float(hessian[0, 1])) > 1.0e-4

    extreme = torch.as_tensor(
        np.stack((center + 10_000.0 * scale, center - 10_000.0 * scale)),
        dtype=torch.float32,
    )
    extreme_q = critic.q_from_features(features, extreme)
    assert torch.isfinite(extreme_q).all()
    assert torch.all(extreme_q < components["q"])

    # Stay uncapped across the float32 square-overflow boundary.  The critic
    # maps overflow only in value space to a finite negative tail; it never
    # substitutes or clips the economically valid coordinate.
    square_boundary = float(np.sqrt(np.finfo(np.float32).max))
    tail_coordinates = torch.tensor(
        [
            [square_boundary * 0.01, 0.0, 0.0],
            [square_boundary * 1.0e10, 0.0, 0.0],
            [-square_boundary * 1.0e10, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    tail_features = features[:1].expand(3, -1)
    tail_components = critic.structured_components(
        tail_features, tail_coordinates
    )
    assert torch.isfinite(tail_components["q"]).all()
    assert bool(tail_components["quadratic_value_saturated"][1:].all())
    assert torch.all(tail_components["q"][1:] < components["q"][0])
    assert torch.isfinite(tail_components["q"].square()).all()

    ordinary = torch.tensor([[10.0, -8.0, 6.0]], dtype=torch.float32)
    ordinary_components = critic.structured_components(features[:1], ordinary)
    expected = (
        ordinary_components["value"]
        + ordinary_components["linear_advantage"]
        + ordinary_components["quadratic_advantage"]
    )
    torch.testing.assert_close(ordinary_components["q"], expected)
    assert not bool(ordinary_components["quadratic_value_saturated"].any())


def test_structured_critic_gradient_isolation_and_common_offset_invariance() -> None:
    agent = RecurrentSACAgent(
        4,
        config=RecurrentSACConfig(hidden_dim=12, seed=277),
    )
    _configure_critic_geometry(agent)
    critic = agent.critic1
    context = torch.randn(2, 3, agent.context_dim)
    features, _ = critic.features(context)
    center = critic.action_reference.reshape(1, 1, -1).expand(2, 3, -1)

    critic.zero_grad(set_to_none=True)
    center_loss = critic.q_from_features(features, center).square().mean()
    center_loss.backward()
    advantage_gradients = [
        parameter.grad for parameter in critic.advantage_parameters()
    ]
    assert all(
        gradient is None or torch.count_nonzero(gradient) == 0
        for gradient in advantage_gradients
    )
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in critic.value_parameters()
    )

    critic.zero_grad(set_to_none=True)
    detached = features.detach()
    candidate = center.clone()
    candidate[..., 0] += 0.2
    delta = critic.q_from_features(detached, candidate) - critic.q_from_features(
        detached, center
    )
    (delta - 1.0).square().mean().backward()
    assert all(parameter.grad is None for parameter in critic.encoder_parameters())
    assert all(
        parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
        for parameter in critic.value_head_parameters()
    )
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in critic.advantage_parameters()
    )

    with torch.no_grad():
        before = critic.structured_components(detached, candidate)["advantage"]
        value_output = critic.value_head[-1]
        assert isinstance(value_output, torch.nn.Linear)
        value_output.bias.add_(10_000.0)
        after = critic.structured_components(detached, candidate)["advantage"]
    # Paired action effects are represented outside V, so arbitrary common
    # return offsets cannot contaminate the structured advantage itself.
    torch.testing.assert_close(after, before, rtol=0.0, atol=0.0)


def test_structured_geometry_is_transactional_immutable_and_anchor_independent() -> None:
    agent = RecurrentSACAgent(
        4,
        config=RecurrentSACConfig(hidden_dim=8, seed=281),
    )
    with pytest.raises(RuntimeError, match="not configured"):
        agent.critic1.q_from_features(torch.zeros(1, 8), torch.zeros(1, 3))

    # Prevalidation must stop a four-network transaction before critic1 is
    # mutated when a later twin already has incompatible geometry.
    agent.critic2.set_action_reference(np.zeros(3), np.ones(3))
    with pytest.raises(RuntimeError, match="immutable"):
        agent.configure_critic_action_reference(np.ones(3), np.ones(3))
    assert not bool(agent.critic1.action_reference_configured)

    clean = RecurrentSACAgent(
        4,
        config=RecurrentSACConfig(hidden_dim=8, seed=283),
    )
    center = np.array([0.1, 0.2, -0.3])
    scale = np.array([0.05, 0.1, 0.2])
    clean.configure_critic_action_reference(center, scale)
    reference_before = clean.critic1.action_reference.detach().clone()
    scale_before = clean.critic1.action_scale.detach().clone()
    clean.configure_safe_policy_improvement(
        actor_updates_enabled=False,
        warm_anchor_coordinates=center,
        anchor_coefficient=1.0,
        anchor_decay_updates=2.0,
        anchor_scale=scale,
    )
    clean.configure_safe_policy_improvement(
        actor_updates_enabled=False,
        warm_anchor_coordinates=None,
        anchor_coefficient=0.0,
        anchor_decay_updates=1.0,
    )
    torch.testing.assert_close(clean.critic1.action_reference, reference_before)
    torch.testing.assert_close(clean.critic1.action_scale, scale_before)
    for critic in (
        clean.critic2,
        clean.target_critic1,
        clean.target_critic2,
    ):
        torch.testing.assert_close(critic.action_reference, reference_before)
        torch.testing.assert_close(critic.action_scale, scale_before)
    assert clean.critic1_optimizer is not clean.critic2_optimizer
    assert all(
        first.data_ptr() != second.data_ptr()
        for first, second in zip(clean.critic1.parameters(), clean.critic2.parameters())
    )


def test_two_phase_prefit_recovers_centered_concave_quadratic_at_two_probe_days() -> None:
    agent = RecurrentSACAgent(
        4,
        config=RecurrentSACConfig(
            hidden_dim=32,
            batch_size=8,
            burn_in=0,
            learn_length=4,
            gamma=1.0,
            critic_learning_rate=3.0e-3,
            gradient_clip=50.0,
            seed=307,
        ),
    )
    radius = 0.2
    _configure_critic_geometry(agent, scale=np.full(3, radius))
    support = _local_action_support(
        agent.action_transform, np.zeros(3), radius
    )
    episodes = []
    for seed_index, (seed, deviation_day) in enumerate(((701, 0), (907, 3))):
        observations = np.stack(
            [
                np.array(
                    [float(seed_index), day / 4.0, np.sin(day), np.cos(day)],
                    dtype=np.float32,
                )
                for day in range(5)
            ]
        )
        for support_row in support["actions"]:
            offset = np.asarray(support_row["offset"], dtype=np.float32)
            actions = np.zeros((4, 3), dtype=np.float32)
            actions[deviation_day] = offset
            rewards = np.full(4, 5.0 + seed_index, dtype=np.float32)
            # Strictly concave around the center: every positive and negative
            # axis move is worse, while joint corners expose curvature.
            rewards[deviation_day] -= float(
                2.0 * np.sum(np.square(offset / radius))
            )
            episode = Episode(
                observations=observations.copy(),
                actions=actions,
                rewards=rewards,
                dones=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                metadata={
                    "group_id": "l0_train_local_support",
                    "seed": seed,
                    "support_index": int(support_row["index"]),
                    "deviation_day": deviation_day,
                },
            )
            episodes.append(episode)
            agent.add_episode(episode, actions_are_physical=False)

    report = agent.fit_critics_monte_carlo(
        steps=200,
        paired_advantage_steps=200,
        paired_advantage_weight=1.0,
    )
    phase1 = report["paired_advantage"]["after_phase1"]
    phase2 = report["paired_advantage"]["final"]
    for critic_name in ("qmin", "q1", "q2"):
        assert phase2[critic_name]["rmse"] < phase1[critic_name]["rmse"]
        assert phase2[critic_name]["directional_accuracy"] >= 0.90
        assert phase2[critic_name]["rmse_to_null_ratio"] < 0.50
    absolute_phase1 = report["calibration_snapshots"]["after_phase1"][
        "absolute_mc_loss"
    ]
    absolute_phase2 = report["calibration_snapshots"]["after_phase2"][
        "absolute_mc_loss"
    ]
    assert absolute_phase2 <= max(0.10, 1.25 * absolute_phase1)
    phase1_delta = report["two_phase_contract"]["phase1"]["parameter_delta_l2"]
    phase2_delta = report["two_phase_contract"]["phase2"]["parameter_delta_l2"]
    assert phase1_delta["critic1_advantage"] == 0.0
    assert phase1_delta["critic2_advantage"] == 0.0
    assert phase2_delta["critic1_encoder"] == 0.0
    assert phase2_delta["critic2_encoder"] == 0.0
    assert phase2_delta["critic1_advantage"] > 0.0
    assert phase2_delta["critic2_advantage"] > 0.0
    assert report["optimizer_steps_per_critic"] == 600

    validation = agent.critic_validation(episodes)
    by_group = {}
    for row in validation["episodes"]:
        metadata = row["metadata"]
        by_group.setdefault(metadata["seed"], {})[
            metadata["support_index"]
        ] = row
    for rows in by_group.values():
        center = rows[0]
        for support_index in range(1, 7):
            moved = rows[support_index]
            assert moved["q1_probe"] < center["q1_probe"]
            assert moved["q2_probe"] < center["q2_probe"]
            assert moved["q_probe"] < center["q_probe"]


def test_recurrent_sac_save_load_preserves_deterministic_action(tmp_path: Path) -> None:
    agent = RecurrentSACAgent(
        4,
        config=RecurrentSACConfig(hidden_dim=16, burn_in=1, learn_length=2, seed=5),
    )
    _configure_critic_geometry(
        agent,
        center=np.array([0.1, -0.2, 0.3]),
        scale=np.array([0.1, 0.2, 0.3]),
    )
    agent.add_episode(_episode(1.0), actions_are_physical=True)
    probe_features = torch.linspace(-1.0, 1.0, 16).reshape(1, 16)
    probe_coordinates = torch.tensor([[0.2, -0.1, 0.4]])
    critic_before = agent.critic1.q_from_features(
        probe_features, probe_coordinates
    )
    agent.critic1_optimizer.zero_grad(set_to_none=True)
    critic_before.square().mean().backward()
    agent.critic1_optimizer.step()
    center_coordinates = torch.as_tensor(
        agent.critic1.action_reference.detach(), dtype=torch.float32
    ).reshape(1, 3)
    for critic, optimizer, target_delta in (
        (agent.critic1, agent.critic1_advantage_optimizer, 1.0),
        (agent.critic2, agent.critic2_advantage_optimizer, -1.0),
    ):
        optimizer.zero_grad(set_to_none=True)
        predicted_delta = critic.q_from_features(
            probe_features, probe_coordinates
        ) - critic.q_from_features(probe_features, center_coordinates)
        (predicted_delta - target_delta).square().mean().backward()
        optimizer.step()
    critic_before = agent.critic1.q_from_features(
        probe_features, probe_coordinates
    ).detach()
    observation = np.array([1.0, 0.5, -0.25, 0.75], dtype=np.float32)
    agent.accepted_actor_blocks = 2
    before, _, _ = agent.act(observation, deterministic=True)
    path = tmp_path / "agent.pt"
    agent.save(path, metadata={"test": True})
    restored = RecurrentSACAgent.load(path)
    after, _, _ = restored.act(observation, deterministic=True)
    np.testing.assert_allclose(after, before, rtol=1e-6, atol=1e-7)
    assert restored.critic_action_reference_contract() == agent.critic_action_reference_contract()
    critic_after = restored.critic1.q_from_features(
        probe_features, probe_coordinates
    ).detach()
    torch.testing.assert_close(critic_after, critic_before)
    assert len(restored.critic1_optimizer.state) == len(agent.critic1_optimizer.state)
    assert len(restored.critic2_optimizer.state) == len(agent.critic2_optimizer.state)
    assert len(restored.critic1_advantage_optimizer.state) == len(
        agent.critic1_advantage_optimizer.state
    )
    assert len(restored.critic2_advantage_optimizer.state) == len(
        agent.critic2_advantage_optimizer.state
    )
    assert restored.accepted_actor_blocks == 2
    _nested_assert_equal(
        restored.critic1_advantage_optimizer.state_dict(),
        agent.critic1_advantage_optimizer.state_dict(),
    )
    _nested_assert_equal(
        restored.critic2_advantage_optimizer.state_dict(),
        agent.critic2_advantage_optimizer.state_dict(),
    )


class _QuadraticDailyEnv:
    observation_dim = 2

    def __init__(self, target: float = 1.7, days: int = 4) -> None:
        self.target = float(target)
        self.days = int(days)
        self.day = 0
        self.noise = 0.0

    def reset(self, seed: int) -> np.ndarray:
        self.day = 0
        self.noise = float(np.random.default_rng(seed).normal(0.0, 0.05))
        return np.array([0.0, self.noise], dtype=np.float32)

    def step(self, action: np.ndarray):
        self.day += 1
        reward = -float(np.log(action[0] / self.target) ** 2) + self.noise
        return (
            np.array([self.day / self.days, self.noise], dtype=np.float32),
            reward,
            self.day >= self.days,
            {},
        )


def test_static_cem_uses_crn_and_finds_continuous_tariff() -> None:
    seeds = [11, 29, 47]
    result = optimize_static_tariff(
        lambda: _QuadraticDailyEnv(),
        ActionTransform.market_controls(),
        seeds,
        config=CEMConfig(population=24, iterations=5, initial_std=0.6, seed=9),
    )
    assert result["action"][0] == pytest.approx(1.7, rel=0.15)
    assert result["bracket_verified"]
    coarse_scores = {
        round(row["multiplier"], 8): row["score"]
        for row in result["coarse_to_fine_evaluations"]
    }
    assert result["score"] >= max(coarse_scores.get(1.0, -np.inf), coarse_scores.get(2.0, -np.inf))
    evaluation_a = evaluate_policy(
        lambda: _QuadraticDailyEnv(), lambda: FixedPolicy(result["action"]), seeds
    )
    evaluation_b = evaluate_policy(
        lambda: _QuadraticDailyEnv(), lambda: FixedPolicy([1.0, 0.0, 1.0]), seeds
    )
    # The per-seed noise is identical across policies, so paired improvements
    # have zero numerical dispersion in this test environment.
    improvement = np.asarray(evaluation_a["values"]) - np.asarray(evaluation_b["values"])
    assert np.std(improvement) < 1e-12
    assert improvement.mean() > 0.5


class _FixedGeometricRng:
    def geometric(self, probability: float) -> int:
        assert probability == pytest.approx(1.0 / 3.0)
        return 3


def test_geometric_commitment_repeats_exact_sampled_duration() -> None:
    commitment = GeometricActionCommitment(3.0, _FixedGeometricRng())
    action = np.array([1.2, 0.4, 1.1])
    coordinates = ActionTransform.market_controls().physical_to_coordinates(action)
    assert commitment.begin(action, coordinates) == 3
    rows = [commitment.take() for _ in range(3)]
    for repeated_action, repeated_coordinates, _ in rows:
        np.testing.assert_array_equal(repeated_action, action)
        np.testing.assert_array_equal(repeated_coordinates, coordinates)
    assert [row[2]["repeated"] for row in rows] == [False, True, True]
    assert rows[-1][2]["remaining_after_step"] == 0
    with pytest.raises(RuntimeError):
        commitment.take()


def test_agent_reports_overflowing_unsquashed_coordinate() -> None:
    agent = RecurrentSACAgent(
        2,
        config=RecurrentSACConfig(hidden_dim=8, burn_in=0, learn_length=1, seed=4),
    )
    output = agent.actor.head[-1]
    with torch.no_grad():
        output.weight.zero_()
        output.bias.zero_()
        output.bias[0] = 1_000.0
    with pytest.raises(NumericalPolicyActionError) as caught:
        agent.act(np.zeros(2, dtype=np.float32), deterministic=True)
    assert caught.value.coordinates[0] == pytest.approx(1_000.0)
    assert agent.nonfinite_action_rejections == 1


class _OverflowDailyEnv:
    observation_dim = 2

    def reset(self, seed: int) -> np.ndarray:
        del seed
        return np.zeros(2, dtype=np.float32)

    def step(self, action: np.ndarray):
        del action
        raise FloatingPointError("synthetic tariff overflow")


def test_training_records_terminal_penalty_for_numerical_action_failure() -> None:
    agent = RecurrentSACAgent(
        2,
        config=RecurrentSACConfig(
            hidden_dim=8,
            batch_size=1,
            burn_in=0,
            learn_length=1,
            seed=5,
        ),
    )
    logs = train_stage(
        agent,
        lambda: _OverflowDailyEnv(),
        [17],
        warmup_episodes=1,
        updates_per_day=0.0,
        seed=9,
        exploration_commitment_mean_days=3.0,
    )
    assert logs[0]["numerical_invalid_actions"] == 1
    assert logs[0]["return"] == NUMERICAL_POLICY_PENALTY
    episode = agent.replay.episodes[0]
    assert episode.length == 1
    assert episode.dones[0] == 1.0
    assert episode.rewards[0] == NUMERICAL_POLICY_PENALTY
    assert np.all(np.isfinite(episode.actions))


def test_train_stage_rejection_restores_external_exploration_rng_and_continues() -> None:
    recorded_actions = []

    class OneDayRecordingEnv:
        observation_dim = 2

        def reset(self, seed: int) -> np.ndarray:
            return np.array([float(seed), 0.0], dtype=np.float32)

        def step(self, action: np.ndarray):
            recorded_actions.append(np.asarray(action, dtype=np.float64).copy())
            return np.zeros(2, dtype=np.float32), 0.0, True, {}

    agent = RecurrentSACAgent(
        2,
        config=RecurrentSACConfig(hidden_dim=8, batch_size=1, seed=991),
    )
    agent.initialize_economic_incumbent("warm")

    class AcceptanceStub:
        def evaluate_episode(self, episode_index, *, proposal_numerically_valid):
            assert proposal_numerically_valid
            if episode_index == 0:
                rollback = agent.restore_economic_incumbent("synthetic harm")
                return {
                    "actor_proposal": True,
                    "accepted": False,
                    "rollback": rollback,
                }
            agent.advance_economic_incumbent_without_actor_change("critic-only")
            return {
                "actor_proposal": False,
                "accepted": True,
                "rollback": None,
            }

    logs = train_stage(
        agent,
        OneDayRecordingEnv,
        [17, 29],
        warmup_episodes=2,
        updates_per_day=0.0,
        seed=101,
        exploration_commitment_mean_days=1.0,
        actor_acceptance=AcceptanceStub(),
    )
    assert len(logs) == 2
    assert logs[0]["nonterminal_economic_rollback"]
    assert logs[0]["training_rng_restored"]
    assert logs[0]["update_credit_restored"]
    np.testing.assert_array_equal(recorded_actions[0], recorded_actions[1])
    assert len(agent.replay.episodes) == 1
    assert not agent.training_terminal_failure


def test_train_stage_critic_rejection_rolls_back_episode_and_continues(
    monkeypatch,
) -> None:
    recorded_actions = []

    class OneDayRecordingEnv:
        observation_dim = 2

        def reset(self, seed: int) -> np.ndarray:
            del seed
            return np.zeros(2, dtype=np.float32)

        def step(self, action: np.ndarray):
            recorded_actions.append(np.asarray(action, dtype=np.float64).copy())
            return np.zeros(2, dtype=np.float32), 0.0, True, {}

    agent = RecurrentSACAgent(
        2,
        config=RecurrentSACConfig(hidden_dim=8, batch_size=1, seed=993),
    )
    agent.capture_last_passing_state("warm_critic")
    agent.initialize_economic_incumbent("warm_actor")
    calls = {"update": 0, "critic_rejections": 0, "acceptance": 0}

    def fake_update(*, critic_gate, restore_economic_incumbent_on_gate_failure):
        del critic_gate
        assert restore_economic_incumbent_on_gate_failure
        calls["update"] += 1
        if calls["update"] == 1:
            rollback = agent.restore_economic_incumbent("synthetic gate miss")
            return {
                "terminal_training_failure": 0.0,
                "critic_proposal_rejection": 1.0,
                "temperature": 1.0,
                "rejected_update": 1.0,
                "rollback": rollback,
                "safe_critic_gate": {"passed": False},
            }
        return {
            "terminal_training_failure": 0.0,
            "critic_proposal_rejection": 0.0,
            "temperature": 1.0,
        }

    monkeypatch.setattr(agent, "can_update", lambda: True)
    monkeypatch.setattr(agent, "update", fake_update)

    class AcceptanceStub:
        def record_critic_proposal_rejection(
            self, episode_index, *, rollback, rejected_update
        ):
            assert episode_index == 0
            assert not rollback["terminal"]
            assert rejected_update == 1
            calls["critic_rejections"] += 1
            return {
                "actor_proposal": False,
                "accepted": False,
                "critic_proposal_rejection": True,
                "seeds_consumed": False,
            }

        def evaluate_episode(self, episode_index, *, proposal_numerically_valid):
            assert episode_index == 1
            assert proposal_numerically_valid
            calls["acceptance"] += 1
            agent.advance_economic_incumbent_without_actor_change("continued")
            return {
                "actor_proposal": False,
                "accepted": True,
                "rollback": None,
            }

    logs = train_stage(
        agent,
        OneDayRecordingEnv,
        [17, 29],
        warmup_episodes=2,
        updates_per_day=1.0,
        seed=103,
        exploration_commitment_mean_days=1.0,
        actor_gate_recheck=lambda *_: {"passed": True},
        actor_acceptance=AcceptanceStub(),
    )
    assert len(logs) == 2
    assert logs[0]["critic_proposal_rejection"]
    assert logs[0]["nonterminal_economic_rollback"]
    assert logs[0]["training_rng_restored"]
    assert logs[0]["update_credit_restored"]
    assert not logs[0]["acceptance_block_consumed"]
    assert not logs[1]["critic_proposal_rejection"]
    assert calls == {"update": 2, "critic_rejections": 1, "acceptance": 1}
    np.testing.assert_array_equal(recorded_actions[0], recorded_actions[1])
    assert len(agent.replay.episodes) == 1
    assert agent.economic_incumbent_contract()["label"] == "continued"
    assert not agent.training_terminal_failure
