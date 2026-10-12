"""Sequence-aware recurrent Soft Actor-Critic for continuous market controls.

The legacy learner in :mod:`PPOAgent` is intentionally left untouched.  This
module is the v28 off-policy alternative used by the learner ladder.  It has a
few design choices that are important for this project:

* the actor and each of the two critics own separate GRUs;
* replay stores complete episodes and samples a learn start across the whole
  episode, including transition zero;
* each sampled loss window unrolls the exact episode history from time zero,
  while masks keep history/padding out of losses and inverse-inclusion weights
  give every transition equal expected training weight;
* positive economic controls use an *unsquashed Gaussian in log space*.
  Their physical values are ``exp(z)`` and therefore have no arbitrary upper
  action bound.  The only rejection is for non-finite floating-point values;
* bounded transforms remain available for ablations, but are not the v28
  default for price or driver-incentive multipliers.

The critics consume policy coordinates (log multiplier for a positive action)
rather than physical actions.  Entropy is consequently measured in those
coordinates.  That makes the entropy target invariant to currency/scale and
avoids rewarding the actor merely for changing physical units.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


_LOG_STD_MIN = -5.0
_LOG_STD_MAX = 1.0
_TANH_EPS = 1e-6


class NumericalPolicyActionError(FloatingPointError):
    """A finite policy coordinate could not be mapped to a finite action."""

    def __init__(self, coordinates: np.ndarray, message: str) -> None:
        super().__init__(message)
        self.coordinates = np.asarray(coordinates, dtype=np.float64).copy()


@dataclass(frozen=True)
class ActionTransform:
    """Mapping between policy coordinates and physical market controls.

    ``kinds`` entries are:

    ``"log_positive"``
        Coordinate ``z`` maps to ``exp(z)``.  It is positive and uncapped.
    ``"identity"``
        Coordinate and physical action are identical and unbounded.
    ``"bounded"``
        A legacy/ablation action mapped through tanh to finite ``low/high``.
    """

    names: Tuple[str, ...]
    kinds: Tuple[str, ...]
    low: Tuple[float, ...] = ()
    high: Tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not self.names or len(self.names) != len(self.kinds):
            raise ValueError("names and kinds must have the same non-zero length")
        allowed = {"log_positive", "identity", "bounded"}
        unknown = set(self.kinds) - allowed
        if unknown:
            raise ValueError(f"unknown action transform kinds: {sorted(unknown)}")
        if any(kind == "bounded" for kind in self.kinds):
            if len(self.low) != self.dim or len(self.high) != self.dim:
                raise ValueError("bounded transforms require low/high for every dimension")
            for index, kind in enumerate(self.kinds):
                if kind == "bounded" and not self.low[index] < self.high[index]:
                    raise ValueError(f"invalid bounds for action {self.names[index]!r}")

    @property
    def dim(self) -> int:
        return len(self.names)

    @classmethod
    def market_controls(cls) -> "ActionTransform":
        """Uncapped v28 controls: tariff level, surge response, incentive."""
        return cls(
            names=("base_multiplier", "surge_gain", "driver_incentive_multiplier"),
            kinds=("log_positive", "identity", "log_positive"),
        )

    def coordinates_to_physical_tensor(self, coordinates: torch.Tensor) -> torch.Tensor:
        if coordinates.shape[-1] != self.dim:
            raise ValueError(f"expected action width {self.dim}, got {coordinates.shape[-1]}")
        columns: List[torch.Tensor] = []
        for index, kind in enumerate(self.kinds):
            value = coordinates[..., index]
            if kind == "log_positive":
                value = torch.exp(value)
            elif kind == "bounded":
                lo = float(self.low[index])
                hi = float(self.high[index])
                value = lo + 0.5 * (value + 1.0) * (hi - lo)
            columns.append(value)
        return torch.stack(columns, dim=-1)

    def coordinates_to_physical(self, coordinates: np.ndarray | Sequence[float]) -> np.ndarray:
        array = np.asarray(coordinates, dtype=np.float64)
        if array.shape[-1] != self.dim:
            raise ValueError(f"expected action width {self.dim}, got {array.shape[-1]}")
        if not np.all(np.isfinite(array)):
            raise NumericalPolicyActionError(
                array, "policy emitted a non-finite native action coordinate"
            )
        result = array.copy()
        for index, kind in enumerate(self.kinds):
            if kind == "log_positive":
                # No economic cap.  Overflow is detected below instead of
                # silently replacing the learner's action with a guardrail.
                with np.errstate(over="ignore", invalid="ignore"):
                    result[..., index] = np.exp(array[..., index])
            elif kind == "bounded":
                lo = float(self.low[index])
                hi = float(self.high[index])
                result[..., index] = lo + 0.5 * (array[..., index] + 1.0) * (hi - lo)
        invalid_positive = any(
            kind == "log_positive" and np.any(result[..., index] <= 0.0)
            for index, kind in enumerate(self.kinds)
        )
        if not np.all(np.isfinite(result)) or invalid_positive:
            # exp(z) may underflow to the finite value 0.0 as well as overflow
            # to infinity.  Both are numerical failures of an otherwise
            # uncapped transform; neither is clipped or replaced.
            raise NumericalPolicyActionError(
                array,
                "policy coordinate overflowed or underflowed its physical action",
            )
        return result

    def physical_to_coordinates(self, physical: np.ndarray | Sequence[float]) -> np.ndarray:
        array = np.asarray(physical, dtype=np.float64)
        if array.shape[-1] != self.dim:
            raise ValueError(f"expected action width {self.dim}, got {array.shape[-1]}")
        if not np.all(np.isfinite(array)):
            raise FloatingPointError("cannot store a non-finite physical action")
        result = array.copy()
        for index, kind in enumerate(self.kinds):
            if kind == "log_positive":
                if np.any(array[..., index] <= 0.0):
                    raise ValueError(f"action {self.names[index]!r} must be positive")
                result[..., index] = np.log(array[..., index])
            elif kind == "bounded":
                lo = float(self.low[index])
                hi = float(self.high[index])
                unit = 2.0 * (array[..., index] - lo) / (hi - lo) - 1.0
                if np.any(np.abs(unit) >= 1.0):
                    raise ValueError(
                        f"action {self.names[index]!r} lies outside its bounded ablation support"
                    )
                result[..., index] = unit
        return result


@dataclass
class Episode:
    """One replay episode with ``T + 1`` observations and ``T`` transitions."""

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.observations = np.asarray(self.observations, dtype=np.float32)
        self.actions = np.asarray(self.actions, dtype=np.float32)
        self.rewards = np.asarray(self.rewards, dtype=np.float32).reshape(-1)
        self.dones = np.asarray(self.dones, dtype=np.float32).reshape(-1)
        self.metadata = copy.deepcopy(dict(self.metadata or {}))
        transitions = self.actions.shape[0]
        if self.observations.ndim != 2 or self.actions.ndim != 2:
            raise ValueError("observations and actions must be rank-2 arrays")
        if self.observations.shape[0] != transitions + 1:
            raise ValueError("an episode needs T+1 observations for T actions")
        if self.rewards.shape[0] != transitions or self.dones.shape[0] != transitions:
            raise ValueError("actions, rewards, and dones must have the same length")
        if transitions == 0:
            raise ValueError("empty replay episodes are not supported")
        for name, value in (
            ("observations", self.observations),
            ("actions", self.actions),
            ("rewards", self.rewards),
            ("dones", self.dones),
        ):
            if not np.all(np.isfinite(value)):
                raise FloatingPointError(f"episode {name} contains non-finite values")

    @property
    def length(self) -> int:
        return int(self.actions.shape[0])


class EpisodeBuilder:
    """Small helper that makes transition alignment explicit."""

    def __init__(self, initial_observation: np.ndarray | Sequence[float]) -> None:
        self._observations = [np.asarray(initial_observation, dtype=np.float32).copy()]
        self._actions: List[np.ndarray] = []
        self._rewards: List[float] = []
        self._dones: List[float] = []

    def append(
        self,
        action: np.ndarray | Sequence[float],
        reward: float,
        next_observation: np.ndarray | Sequence[float],
        done: bool,
    ) -> None:
        if self._dones and self._dones[-1] >= 1.0:
            raise RuntimeError("cannot append after a terminal transition")
        self._actions.append(np.asarray(action, dtype=np.float32).copy())
        self._rewards.append(float(reward))
        self._dones.append(float(bool(done)))
        self._observations.append(np.asarray(next_observation, dtype=np.float32).copy())

    def finish(self) -> Episode:
        return Episode(
            observations=np.stack(self._observations),
            actions=np.stack(self._actions),
            rewards=np.asarray(self._rewards, dtype=np.float32),
            dones=np.asarray(self._dones, dtype=np.float32),
        )


class SequenceReplayBuffer:
    """Episode replay with prefix-correct, whole-episode target coverage."""

    def __init__(self, capacity_transitions: int, seed: int = 0) -> None:
        self.capacity_transitions = int(max(1, capacity_transitions))
        self.episodes: Deque[Episode] = deque()
        self.transitions = 0
        self.rng = np.random.default_rng(int(seed))

    def __len__(self) -> int:
        return self.transitions

    def add(self, episode: Episode) -> None:
        self.episodes.append(episode)
        self.transitions += episode.length
        while self.transitions > self.capacity_transitions and len(self.episodes) > 1:
            removed = self.episodes.popleft()
            self.transitions -= removed.length

    def eligible_sequences(self, transition_length: int) -> int:
        length = int(transition_length)
        return sum(max(0, episode.length - length + 1) for episode in self.episodes)

    @staticmethod
    def _training_window(
        episode: Episode,
        *,
        learn_start: int,
        burn_in: int,
        learn_length: int,
    ) -> Dict[str, Any]:
        """Build one recurrent window whose loss begins at ``learn_start``.

        ``learn_start`` ranges over every transition in the episode.  Context
        always begins at episode time zero, so a mid-episode target sees the
        exact current-network recurrent history rather than a zero-initialized
        truncated approximation.  Since target windows overlap, target ``t``
        is weighted by ``1 / min(learn_length, t + 1)``, the reciprocal of its
        inclusion count under a uniformly sampled learn start.
        """
        start = int(learn_start)
        burn = int(max(0, burn_in))
        horizon = int(learn_length)
        if horizon <= 0:
            raise ValueError("learn_length must be positive")
        if start < 0 or start >= episode.length:
            raise ValueError(
                f"learn_start must lie in [0, {episode.length}), got {start}"
            )
        actual_burn = start
        context_start = 0
        learn_stop = min(episode.length, start + horizon)
        context_stop = learn_stop
        transition_indices = np.arange(
            context_start, context_stop, dtype=np.int64
        )
        steps = int(transition_indices.size)
        loss_mask = np.zeros(steps, dtype=np.float32)
        loss_mask[actual_burn:] = 1.0
        loss_weights = np.zeros(steps, dtype=np.float32)
        target_indices = transition_indices[actual_burn:]
        loss_weights[actual_burn:] = 1.0 / np.minimum(
            horizon, target_indices + 1
        ).astype(np.float32)

        previous_actions = np.zeros(
            (steps + 1, episode.actions.shape[1]), dtype=np.float32
        )
        previous_rewards = np.zeros((steps + 1, 1), dtype=np.float32)
        previous_dones = np.zeros((steps + 1, 1), dtype=np.float32)
        if context_start > 0:
            previous_actions[0] = episode.actions[context_start - 1]
            previous_rewards[0, 0] = episode.rewards[context_start - 1]
            previous_dones[0, 0] = episode.dones[context_start - 1]
        previous_actions[1:] = episode.actions[context_start:context_stop]
        previous_rewards[1:, 0] = episode.rewards[context_start:context_stop]
        previous_dones[1:, 0] = episode.dones[context_start:context_stop]
        return {
            "observations": episode.observations[context_start : context_stop + 1],
            "actions": episode.actions[context_start:context_stop],
            "rewards": episode.rewards[context_start:context_stop],
            "dones": episode.dones[context_start:context_stop],
            "previous_actions": previous_actions,
            "previous_rewards": previous_rewards,
            "previous_dones": previous_dones,
            "sequence_mask": np.ones(steps, dtype=np.float32),
            "loss_mask": loss_mask,
            "loss_weights": loss_weights,
            "transition_indices": transition_indices,
            "learn_start": start,
            "context_start": context_start,
            "actual_burn": actual_burn,
            "configured_burn_in": burn,
            "learn_count": int(learn_stop - start),
            "sequence_steps": steps,
        }

    @staticmethod
    def _collate_windows(
        windows: Sequence[Mapping[str, Any]],
        *,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        if not windows:
            raise ValueError("cannot collate an empty recurrent batch")
        batch_size = len(windows)
        max_steps = max(int(window["sequence_steps"]) for window in windows)
        observation_dim = int(np.asarray(windows[0]["observations"]).shape[1])
        action_dim = int(np.asarray(windows[0]["actions"]).shape[1])

        observations = np.zeros(
            (batch_size, max_steps + 1, observation_dim), dtype=np.float32
        )
        actions = np.zeros((batch_size, max_steps, action_dim), dtype=np.float32)
        rewards = np.zeros((batch_size, max_steps), dtype=np.float32)
        dones = np.zeros((batch_size, max_steps), dtype=np.float32)
        previous_actions = np.zeros(
            (batch_size, max_steps + 1, action_dim), dtype=np.float32
        )
        previous_rewards = np.zeros(
            (batch_size, max_steps + 1, 1), dtype=np.float32
        )
        previous_dones = np.zeros(
            (batch_size, max_steps + 1, 1), dtype=np.float32
        )
        sequence_mask = np.zeros((batch_size, max_steps), dtype=np.float32)
        loss_mask = np.zeros((batch_size, max_steps), dtype=np.float32)
        loss_weights = np.zeros((batch_size, max_steps), dtype=np.float32)
        transition_indices = np.full(
            (batch_size, max_steps), -1, dtype=np.int64
        )
        metadata = {
            name: np.zeros(batch_size, dtype=np.int64)
            for name in (
                "learn_start",
                "context_start",
                "actual_burn",
                "configured_burn_in",
                "learn_count",
                "sequence_steps",
            )
        }
        for row, window in enumerate(windows):
            steps = int(window["sequence_steps"])
            observations[row, : steps + 1] = window["observations"]
            actions[row, :steps] = window["actions"]
            rewards[row, :steps] = window["rewards"]
            dones[row, :steps] = window["dones"]
            previous_actions[row, : steps + 1] = window["previous_actions"]
            previous_rewards[row, : steps + 1] = window["previous_rewards"]
            previous_dones[row, : steps + 1] = window["previous_dones"]
            sequence_mask[row, :steps] = window["sequence_mask"]
            loss_mask[row, :steps] = window["loss_mask"]
            loss_weights[row, :steps] = window["loss_weights"]
            transition_indices[row, :steps] = window["transition_indices"]
            for name in metadata:
                metadata[name][row] = int(window[name])

        def float_tensor(value: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(value, dtype=torch.float32, device=device)

        batch: Dict[str, torch.Tensor] = {
            "observations": float_tensor(observations),
            "actions": float_tensor(actions),
            "rewards": float_tensor(rewards),
            "dones": float_tensor(dones),
            "previous_actions": float_tensor(previous_actions),
            "previous_rewards": float_tensor(previous_rewards),
            "previous_dones": float_tensor(previous_dones),
            "sequence_mask": float_tensor(sequence_mask),
            "loss_mask": float_tensor(loss_mask),
            "loss_weights": float_tensor(loss_weights),
            "transition_indices": torch.as_tensor(
                transition_indices, dtype=torch.long, device=device
            ),
        }
        batch.update(
            {
                name: torch.as_tensor(value, dtype=torch.long, device=device)
                for name, value in metadata.items()
            }
        )
        return batch

    def transition_window(
        self,
        episode_index: int,
        learn_start: int,
        *,
        burn_in: int,
        learn_length: int,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Materialize a deterministic window for diagnostics and tests."""
        episodes = list(self.episodes)
        index = int(episode_index)
        if index < 0 or index >= len(episodes):
            raise IndexError("episode_index is outside replay")
        window = self._training_window(
            episodes[index],
            learn_start=int(learn_start),
            burn_in=int(burn_in),
            learn_length=int(learn_length),
        )
        return self._collate_windows([window], device=device)

    def full_episode_batch(self, device: torch.device) -> Dict[str, torch.Tensor]:
        """Return every stored episode from its zero-hidden opening state."""
        episodes = list(self.episodes)
        if not episodes:
            raise RuntimeError("replay is empty")
        windows = [
            self._training_window(
                episode,
                learn_start=0,
                burn_in=0,
                learn_length=episode.length,
            )
            for episode in episodes
        ]
        # Full-episode cloning is not a randomly overlapping replay sample:
        # every warm action state receives equal supervised weight.
        for window in windows:
            window["loss_weights"] = np.asarray(
                window["loss_mask"], dtype=np.float32
            ).copy()
        return self._collate_windows(windows, device=device)

    def sample(
        self,
        batch_size: int,
        burn_in: int,
        learn_length: int,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        count = int(batch_size)
        if count <= 0:
            raise ValueError("batch_size must be positive")
        if int(burn_in) < 0 or int(learn_length) <= 0:
            raise ValueError("burn_in must be non-negative and learn_length positive")
        episodes = list(self.episodes)
        if not episodes:
            raise RuntimeError("replay is empty")
        # Weight episodes by length, then draw the learn start uniformly, so
        # every episode transition has equal probability of being the first
        # optimized target—including transition zero and the terminal one.
        weights = np.asarray([episode.length for episode in episodes], dtype=np.float64)
        weights /= weights.sum()
        windows = []
        for _ in range(count):
            episode_index = int(self.rng.choice(len(episodes), p=weights))
            episode = episodes[episode_index]
            learn_start = int(self.rng.integers(0, episode.length))
            window = self._training_window(
                episode,
                learn_start=learn_start,
                burn_in=int(burn_in),
                learn_length=int(learn_length),
            )
            window["episode_index"] = episode_index
            windows.append(window)
        batch = self._collate_windows(windows, device=device)
        batch["episode_indices"] = torch.as_tensor(
            [int(window["episode_index"]) for window in windows],
            dtype=torch.long,
            device=device,
        )
        return batch


class RunningMeanStd:
    """Streaming training-only observation normalization.

    Categorical/bounded public indicators are represented in their natural
    units.  Standardising an unobserved binary category with the RMS epsilon
    can otherwise turn ``1`` into a four-thousand-sigma input.  The identity
    mask is part of the serialized contract so a checkpoint cannot silently
    change observation coordinates when it is loaded or rolled back.
    """

    def __init__(
        self,
        shape: int,
        epsilon: float = 1e-4,
        identity_indices: Sequence[int] = (),
    ) -> None:
        self.mean = np.zeros(int(shape), dtype=np.float64)
        self.var = np.ones(int(shape), dtype=np.float64)
        self.count = float(epsilon)
        indices = tuple(int(index) for index in identity_indices)
        if len(indices) != len(set(indices)):
            raise ValueError("observation identity indices must be unique")
        if any(index < 0 or index >= int(shape) for index in indices):
            raise ValueError("observation identity index is out of bounds")
        self.identity_mask = np.zeros(int(shape), dtype=bool)
        self.identity_mask[list(indices)] = True

    def update(self, observations: np.ndarray) -> None:
        array = np.asarray(observations, dtype=np.float64).reshape(-1, self.mean.size)
        if array.size == 0:
            return
        batch_mean = array.mean(axis=0)
        batch_var = array.var(axis=0)
        batch_count = float(array.shape[0])
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta * delta * self.count * batch_count / total
        self.mean = new_mean
        self.var = np.maximum(m2 / total, 1e-8)
        # Identity features never inherit accidental training-distribution
        # moments.  Keeping their stored moments canonical also makes state
        # fingerprints and rollback checks unambiguous.
        self.mean[self.identity_mask] = 0.0
        self.var[self.identity_mask] = 1.0
        self.count = total

    def normalize_tensor(self, observations: torch.Tensor) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, dtype=observations.dtype, device=observations.device)
        std = torch.as_tensor(np.sqrt(self.var), dtype=observations.dtype, device=observations.device)
        normalized = (observations - mean) / std.clamp_min(1e-4)
        if bool(np.any(self.identity_mask)):
            identity = torch.as_tensor(
                self.identity_mask,
                dtype=torch.bool,
                device=observations.device,
            )
            normalized = torch.where(identity, observations, normalized)
        return normalized

    def state_dict(self) -> Dict[str, Any]:
        return {
            "mean": self.mean.copy(),
            "var": self.var.copy(),
            "count": self.count,
            "identity_mask": self.identity_mask.copy(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        mean = np.asarray(state["mean"], dtype=np.float64).copy()
        var = np.asarray(state["var"], dtype=np.float64).copy()
        if mean.shape != self.mean.shape or var.shape != self.var.shape:
            raise ValueError("observation normalizer shape does not match agent")
        stored_mask = np.asarray(
            state.get("identity_mask", np.zeros_like(mean, dtype=bool)),
            dtype=bool,
        )
        if stored_mask.shape != self.identity_mask.shape or not np.array_equal(
            stored_mask, self.identity_mask
        ):
            raise ValueError(
                "observation normalizer identity mask does not match checkpoint"
            )
        self.mean = mean
        self.var = var
        self.count = float(state["count"])


def effective_horizon_lengths(
    dones: torch.Tensor,
    sequence_mask: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Deterministic discounted remaining horizon for each valid transition.

    This is a scale for offline Monte-Carlo regression only; it does not alter
    raw returns, Q values, paired CRN advantages, or the online TD target.
    Padding has horizon zero and an in-sequence terminal has horizon one.
    """
    if dones.ndim != 2 or sequence_mask.ndim != 2 or dones.shape != sequence_mask.shape:
        raise ValueError("dones and sequence_mask must be matching [batch,time] tensors")
    discount = float(gamma)
    if not 0.0 < discount <= 1.0:
        raise ValueError("gamma must lie in (0,1]")
    mask = sequence_mask.to(dtype=dones.dtype)
    output = torch.zeros_like(dones)
    running = torch.zeros(dones.shape[0], dtype=dones.dtype, device=dones.device)
    for index in range(dones.shape[1] - 1, -1, -1):
        valid = mask[:, index]
        running = (
            1.0 + discount * (1.0 - dones[:, index]) * running
        ) * valid
        output[:, index] = running
    return output


class _RecurrentEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRU(int(input_dim), int(hidden_dim), batch_first=True)
        self.norm = nn.LayerNorm(int(hidden_dim))

    def forward(
        self,
        inputs: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        burn_in: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        burn = int(max(0, burn_in))
        if burn >= inputs.shape[1]:
            raise ValueError("burn-in must leave at least one recurrent state")
        if burn:
            with torch.no_grad():
                _, hidden = self.gru(inputs[:, :burn], hidden)
            hidden = hidden.detach()
        output, hidden = self.gru(inputs[:, burn:], hidden)
        return self.norm(output), hidden


class RecurrentGaussianActor(nn.Module):
    def __init__(self, context_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.encoder = _RecurrentEncoder(context_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * action_dim),
        )
        self.action_dim = int(action_dim)
        self._reset_output_layer()

    def _reset_output_layer(self) -> None:
        output = self.head[-1]
        assert isinstance(output, nn.Linear)
        nn.init.uniform_(output.weight, -1e-3, 1e-3)
        nn.init.zeros_(output.bias)
        # A modest initial standard deviation in log-multiplier space.
        with torch.no_grad():
            output.bias[self.action_dim :].fill_(-1.0)

    def features(
        self, context: torch.Tensor, hidden: Optional[torch.Tensor] = None, burn_in: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(context, hidden, burn_in)

    def distribution(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.head(features).chunk(2, dim=-1)
        return mean, log_std.clamp(_LOG_STD_MIN, _LOG_STD_MAX)

    def sample_from_features(
        self,
        features: torch.Tensor,
        transform: ActionTransform,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.distribution(features)
        normal = Normal(mean, log_std.exp())
        pre = mean if deterministic else normal.rsample()
        coordinates = pre.clone()
        log_probability = normal.log_prob(pre)
        for index, kind in enumerate(transform.kinds):
            if kind == "bounded":
                unit = torch.tanh(pre[..., index])
                coordinates[..., index] = unit
                log_probability[..., index] -= torch.log(1.0 - unit.square() + _TANH_EPS)
        return coordinates, log_probability.sum(dim=-1, keepdim=True), mean


class RecurrentQCritic(nn.Module):
    def __init__(self, context_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.encoder = _RecurrentEncoder(context_dim, hidden_dim)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.slope_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self._tril_size = int(action_dim * (action_dim + 1) // 2)
        self.curvature_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self._tril_size),
        )
        self.action_dim = int(action_dim)
        self.register_buffer(
            "action_reference", torch.zeros(self.action_dim, dtype=torch.float32)
        )
        self.register_buffer(
            "action_scale", torch.ones(self.action_dim, dtype=torch.float32)
        )
        self.register_buffer(
            "action_reference_configured", torch.tensor(False, dtype=torch.bool)
        )
        indices = torch.tril_indices(self.action_dim, self.action_dim)
        self.register_buffer("_tril_rows", indices[0])
        self.register_buffer("_tril_columns", indices[1])
        self._reset_advantage_heads()

    def _reset_advantage_heads(self) -> None:
        slope_output = self.slope_head[-1]
        curvature_output = self.curvature_head[-1]
        assert isinstance(slope_output, nn.Linear)
        assert isinstance(curvature_output, nn.Linear)
        with torch.no_grad():
            slope_output.weight.zero_()
            slope_output.bias.zero_()
            curvature_output.weight.zero_()
            curvature_output.bias.zero_()
            # A weak, finite negative-definite prior.  It is measured as a
            # frozen baseline and must be beaten; it is never treated as
            # evidence that the critic learned the action effect.
            desired_diagonal = 0.10
            raw_diagonal = math.log(math.expm1(desired_diagonal))
            diagonal = self._tril_rows == self._tril_columns
            curvature_output.bias[diagonal] = raw_diagonal

    def set_action_reference(
        self,
        center: np.ndarray | Sequence[float],
        scale: np.ndarray | Sequence[float],
    ) -> None:
        center_tensor, scale_tensor = self.validate_action_reference(center, scale)
        if bool(self.action_reference_configured):
            return
        with torch.no_grad():
            self.action_reference.copy_(center_tensor)
            self.action_scale.copy_(scale_tensor)
            self.action_reference_configured.fill_(True)

    def validate_action_reference(
        self,
        center: np.ndarray | Sequence[float],
        scale: np.ndarray | Sequence[float],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Validate immutable action geometry without mutating the critic."""
        center_array = np.asarray(center, dtype=np.float32).reshape(self.action_dim)
        scale_array = np.asarray(scale, dtype=np.float32).reshape(self.action_dim)
        if not np.all(np.isfinite(center_array)):
            raise ValueError("critic action reference must be finite")
        if not np.all(np.isfinite(scale_array)) or np.any(scale_array <= 0.0):
            raise ValueError("critic action scale must be finite and positive")
        center_tensor = torch.as_tensor(
            center_array, dtype=self.action_reference.dtype, device=self.action_reference.device
        )
        scale_tensor = torch.as_tensor(
            scale_array, dtype=self.action_scale.dtype, device=self.action_scale.device
        )
        if bool(self.action_reference_configured):
            if not torch.equal(self.action_reference, center_tensor) or not torch.equal(
                self.action_scale, scale_tensor
            ):
                raise RuntimeError("critic action reference/scale are immutable once configured")
        return center_tensor, scale_tensor

    def encoder_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.encoder.parameters()

    def value_head_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.value_head.parameters()

    def value_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.encoder_parameters()
        yield from self.value_head_parameters()

    def advantage_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.slope_head.parameters()
        yield from self.curvature_head.parameters()

    def features(
        self, context: torch.Tensor, hidden: Optional[torch.Tensor] = None, burn_in: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(context, hidden, burn_in)

    def structured_components(
        self, features: torch.Tensor, coordinates: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        if not bool(self.action_reference_configured):
            raise RuntimeError(
                "structured critic action reference is not configured; install the "
                "TRAIN-only warm-rule center and support scale before critic use"
            )
        if coordinates.shape[-1] != self.action_dim:
            raise ValueError("critic action width does not match structured head")
        value = self.value_head(features)
        slope = self.slope_head(features)
        raw = self.curvature_head(features)
        shape = (*raw.shape[:-1], self.action_dim, self.action_dim)
        lower = torch.zeros(shape, dtype=raw.dtype, device=raw.device)
        diagonal_mask = (
            self._tril_rows == self._tril_columns
        ).reshape(*([1] * (raw.ndim - 1)), self._tril_size)
        structured_values = torch.where(
            diagonal_mask,
            torch.nn.functional.softplus(raw) + 1.0e-4,
            raw,
        )
        lower[..., self._tril_rows, self._tril_columns] = structured_values
        reference = self.action_reference.to(dtype=coordinates.dtype)
        scale = self.action_scale.to(dtype=coordinates.dtype)
        normalized = (coordinates - reference) / scale
        linear = (slope * normalized).sum(dim=-1, keepdim=True)
        transformed = torch.matmul(
            lower.transpose(-1, -2), normalized.unsqueeze(-1)
        ).squeeze(-1)
        raw_quadratic = -0.5 * transformed.square().sum(dim=-1, keepdim=True)
        # The economics remain unbounded: no action is clipped or projected.
        # At coordinates beyond the floating-point square range, however, the
        # mathematically concave tail must not produce NaN/+inf and poison a
        # replay batch.  Saturate only the *value* to a very large finite
        # negative number.  Ordinary finite arithmetic is bit-for-bit
        # unchanged by nan_to_num.
        # A project-scale sentinel keeps both Q and its squared training loss
        # finite in float32 (using a fraction of finfo.max would make Q finite
        # but overflow the very next MSE operation).
        finite_negative_floor = -1.0e12
        quadratic_saturated = (
            ~torch.isfinite(raw_quadratic)
            | (raw_quadratic < finite_negative_floor)
        )
        quadratic = torch.nan_to_num(
            raw_quadratic,
            nan=float(finite_negative_floor),
            posinf=float(finite_negative_floor),
            neginf=float(finite_negative_floor),
        ).clamp(min=float(finite_negative_floor), max=0.0)
        safe_linear = torch.nan_to_num(
            linear,
            nan=0.0,
            posinf=-float(finite_negative_floor),
            neginf=float(finite_negative_floor),
        )
        advantage = torch.where(
            quadratic_saturated,
            torch.full_like(quadratic, finite_negative_floor),
            safe_linear + quadratic,
        )
        q = torch.nan_to_num(
            value + advantage,
            nan=float(finite_negative_floor),
            posinf=float(finite_negative_floor),
            neginf=float(finite_negative_floor),
        ).clamp(min=float(finite_negative_floor), max=-float(finite_negative_floor))
        return {
            "q": q,
            "value": value,
            "advantage": advantage,
            "slope": slope,
            "lower_cholesky": lower,
            "normalized_action": normalized,
            "linear_advantage": linear,
            "quadratic_advantage": quadratic,
            "quadratic_value_saturated": quadratic_saturated,
            "finite_negative_q_floor": torch.full_like(q, finite_negative_floor),
        }

    def q_from_features(self, features: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        return self.structured_components(features, coordinates)["q"]


@dataclass
class RecurrentSACConfig:
    hidden_dim: int = 128
    replay_capacity: int = 100_000
    batch_size: int = 64
    burn_in: int = 8
    learn_length: int = 24
    gamma: float = 0.97
    actor_learning_rate: float = 1e-4
    critic_learning_rate: float = 3e-4
    temperature_learning_rate: float = 3e-4
    polyak_tau: float = 0.005
    initial_temperature: float = 0.20
    target_entropy: Optional[float] = None
    reward_scale: float = 1.0
    gradient_clip: float = 10.0
    device: str = "cpu"
    seed: int = 0
    observation_identity_indices: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must lie in (0, 1]")
        if not 0.0 < self.polyak_tau <= 1.0:
            raise ValueError("polyak_tau must lie in (0, 1]")
        if self.burn_in < 0 or self.learn_length <= 0:
            raise ValueError("burn_in must be non-negative and learn_length positive")
        if self.initial_temperature <= 0.0:
            raise ValueError("initial_temperature must be positive")
        self.observation_identity_indices = tuple(
            int(index) for index in self.observation_identity_indices
        )
        if len(self.observation_identity_indices) != len(
            set(self.observation_identity_indices)
        ):
            raise ValueError("observation_identity_indices must be unique")
        if any(index < 0 for index in self.observation_identity_indices):
            raise ValueError("observation_identity_indices must be non-negative")


class RecurrentSACAgent:
    """Recurrent SAC with sequence replay and independent recurrent critics."""

    def __init__(
        self,
        observation_dim: int,
        action_transform: Optional[ActionTransform] = None,
        config: Optional[RecurrentSACConfig] = None,
    ) -> None:
        self.observation_dim = int(observation_dim)
        if self.observation_dim <= 0:
            raise ValueError("observation_dim must be positive")
        self.action_transform = action_transform or ActionTransform.market_controls()
        self.action_dim = self.action_transform.dim
        self.config = config or RecurrentSACConfig()
        self.device = torch.device(self.config.device)
        torch.manual_seed(int(self.config.seed))
        self.rng = np.random.default_rng(int(self.config.seed))
        self.context_dim = self.observation_dim + self.action_dim + 2

        self.actor = RecurrentGaussianActor(
            self.context_dim, self.action_dim, self.config.hidden_dim
        ).to(self.device)
        self.critic1 = RecurrentQCritic(
            self.context_dim, self.action_dim, self.config.hidden_dim
        ).to(self.device)
        self.critic2 = RecurrentQCritic(
            self.context_dim, self.action_dim, self.config.hidden_dim
        ).to(self.device)
        self.target_critic1 = copy.deepcopy(self.critic1).to(self.device).eval()
        self.target_critic2 = copy.deepcopy(self.critic2).to(self.device).eval()
        for parameter in (*self.target_critic1.parameters(), *self.target_critic2.parameters()):
            parameter.requires_grad_(False)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=float(self.config.actor_learning_rate)
        )
        self.critic1_optimizer = torch.optim.Adam(
            self.critic1.parameters(), lr=float(self.config.critic_learning_rate)
        )
        self.critic2_optimizer = torch.optim.Adam(
            self.critic2.parameters(), lr=float(self.config.critic_learning_rate)
        )
        # Online paired-CRN refreshes must not inherit Adam moments from the
        # TD objective.  In particular, a zero-weight auxiliary objective
        # must be a literal no-op rather than a momentum-driven parameter
        # update through the general critic optimizer.
        self.critic1_advantage_optimizer = torch.optim.Adam(
            self.critic1.advantage_parameters(),
            lr=float(self.config.critic_learning_rate),
        )
        self.critic2_advantage_optimizer = torch.optim.Adam(
            self.critic2.advantage_parameters(),
            lr=float(self.config.critic_learning_rate),
        )
        self.log_temperature = torch.tensor(
            math.log(float(self.config.initial_temperature)),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        self.temperature_optimizer = torch.optim.Adam(
            [self.log_temperature], lr=float(self.config.temperature_learning_rate)
        )
        self.target_entropy = float(
            -self.action_dim if self.config.target_entropy is None else self.config.target_entropy
        )
        self.replay = SequenceReplayBuffer(
            self.config.replay_capacity, seed=int(self.config.seed) + 17
        )
        self.observation_normalizer = RunningMeanStd(
            self.observation_dim,
            identity_indices=self.config.observation_identity_indices,
        )
        self.updates = 0
        self.extreme_action_count = 0
        self.nonfinite_action_rejections = 0
        self.max_abs_coordinate = np.zeros(self.action_dim, dtype=np.float64)
        self.max_physical_action = np.full(self.action_dim, -np.inf, dtype=np.float64)
        self.min_physical_action = np.full(self.action_dim, np.inf, dtype=np.float64)
        self.actor_updates_enabled = True
        self.warm_anchor_coordinates: Optional[np.ndarray] = None
        self.warm_anchor_initial_coefficient = 0.0
        self.warm_anchor_decay_updates = 1.0
        self.warm_anchor_scale = np.ones(self.action_dim, dtype=np.float64)
        self.actor_updates_applied = 0
        self.accepted_actor_blocks = 0
        self.actor_update_interval = 1
        self._paired_crn_cache: Tuple[Episode, ...] = ()
        self._paired_crn_cache_contract: Dict[str, Any] = {}
        self._paired_advantage_online_weight = 0.0
        self._same_version_gate_required = False
        self._last_passing_state: Optional[Dict[str, Any]] = None
        self._last_passing_label: Optional[str] = None
        self._last_passing_capture_count = 0
        self.training_terminal_failure = False
        self.training_terminal_failure_reason: Optional[str] = None
        self.safe_gate_failures = 0
        self._economic_incumbent_state: Optional[Dict[str, Any]] = None
        self._economic_incumbent_label: Optional[str] = None
        self._economic_incumbent_capture_count = 0

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp()

    @property
    def minimum_replay_transitions(self) -> int:
        # Prefix-correct windows can optimize transition zero without waiting
        # for a fixed burn+learn span to accumulate.
        return 1

    def add_episode(
        self,
        episode: Episode,
        actions_are_physical: bool = True,
        *,
        update_normalizer: bool = True,
    ) -> None:
        actions = episode.actions
        if actions_are_physical:
            actions = self.action_transform.physical_to_coordinates(actions).astype(np.float32)
        converted = Episode(
            observations=episode.observations,
            actions=actions,
            rewards=episode.rewards,
            dones=episode.dones,
            metadata=episode.metadata,
        )
        if update_normalizer:
            self.observation_normalizer.update(converted.observations[:-1])
        self.replay.add(converted)

    @staticmethod
    def _immutable_episode_copy(episode: Episode) -> Episode:
        copied = Episode(
            observations=episode.observations.copy(),
            actions=episode.actions.copy(),
            rewards=episode.rewards.copy(),
            dones=episode.dones.copy(),
            metadata=copy.deepcopy(episode.metadata),
        )
        for array in (
            copied.observations,
            copied.actions,
            copied.rewards,
            copied.dones,
        ):
            array.setflags(write=False)
        return copied

    @staticmethod
    def _episode_cache_fingerprint(episodes: Sequence[Episode]) -> str:
        digest = hashlib.sha256()
        for episode in episodes:
            for array in (
                episode.observations,
                episode.actions,
                episode.rewards,
                episode.dones,
            ):
                contiguous = np.ascontiguousarray(array)
                digest.update(str(contiguous.dtype).encode("utf-8"))
                digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
                digest.update(contiguous.tobytes())
            digest.update(
                json.dumps(
                    episode.metadata, sort_keys=True, separators=(",", ":"), default=str
                ).encode("utf-8")
            )
        return digest.hexdigest()

    def configure_paired_crn_cache(
        self,
        episodes: Sequence[Episode],
        *,
        source_group_id: str,
        paired_advantage_weight: float,
        replace_for_new_stage: bool = False,
    ) -> Dict[str, Any]:
        """Freeze the TRAIN-fit CRN probes used after every TD critic step.

        Held-out critic-gate episodes are structurally rejected.  The cache is
        immutable and independent from replay eviction, sampling, and later
        environment stages.
        """
        group_id = str(source_group_id)
        lowered = group_id.lower()
        if not group_id or "heldout" in lowered or "gate" in lowered:
            raise ValueError("paired CRN refresh cache must come from TRAIN-fit probes")
        weight = float(paired_advantage_weight)
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError("paired advantage weight must be finite and non-negative")
        copied: List[Episode] = []
        groups: Dict[Tuple[int, int], set[int]] = {}
        for episode in episodes:
            metadata = dict(episode.metadata)
            if str(metadata.get("group_id", "")) != group_id:
                raise ValueError("paired CRN cache contains a foreign replay group")
            if str(metadata.get("profile_split", "")) != "train_fit":
                raise ValueError("paired CRN cache may contain TRAIN-fit episodes only")
            if bool(metadata.get("heldout_for_gate", False)):
                raise ValueError("held-out critic-gate data cannot enter paired refresh")
            if "seed" not in metadata or "support_index" not in metadata:
                raise ValueError("paired CRN cache episode lacks seed/support provenance")
            key = (int(metadata["seed"]), int(metadata.get("deviation_day", 0)))
            groups.setdefault(key, set()).add(int(metadata["support_index"]))
            copied.append(self._immutable_episode_copy(episode))
        if not copied or any(0 not in members or len(members) < 2 for members in groups.values()):
            raise ValueError("paired CRN cache needs a center and deviations per TRAIN group")
        support_sets = {tuple(sorted(members)) for members in groups.values()}
        if len(support_sets) != 1:
            raise ValueError("paired CRN cache support differs across TRAIN seeds")
        fingerprint = self._episode_cache_fingerprint(copied)
        contract = {
            "source": "immutable_train_fit_paired_crn",
            "source_group_id": group_id,
            "profile_split": "train_fit",
            "heldout_gate_episodes_used": False,
            "episode_count": int(len(copied)),
            "group_count": int(len(groups)),
            "support_indices": list(next(iter(support_sets))),
            "seeds": sorted({key[0] for key in groups}),
            "paired_advantage_weight": weight,
            "sha256": fingerprint,
            "arrays_writeable": False,
        }
        if self._paired_crn_cache and not bool(replace_for_new_stage):
            if contract != self._paired_crn_cache_contract:
                raise RuntimeError("paired CRN refresh cache is immutable once configured")
            return copy.deepcopy(contract)
        if self._paired_crn_cache and bool(replace_for_new_stage):
            previous_contract = copy.deepcopy(self._paired_crn_cache_contract)
            contract["stage_transition_replacement"] = True
            contract["replaced_cache_sha256"] = previous_contract.get("sha256")
            contract["replaced_source_group_id"] = previous_contract.get(
                "source_group_id"
            )
        else:
            contract["stage_transition_replacement"] = False
        self._paired_crn_cache = tuple(copied)
        self._paired_crn_cache_contract = copy.deepcopy(contract)
        self._paired_advantage_online_weight = weight
        self._same_version_gate_required = True
        return copy.deepcopy(contract)

    def paired_crn_cache_contract(self) -> Dict[str, Any]:
        return copy.deepcopy(self._paired_crn_cache_contract)

    def configure_critic_action_reference(
        self,
        center_coordinates: np.ndarray | Sequence[float],
        scale_coordinates: np.ndarray | Sequence[float],
    ) -> None:
        """Install the immutable TRAIN-only local coordinates for both twins."""
        critics = (
            self.critic1,
            self.critic2,
            self.target_critic1,
            self.target_critic2,
        )
        # Validate the complete four-network transaction before mutating any
        # buffer.  This prevents a mismatched target from leaving a partial
        # geometry update behind.
        for critic in critics:
            critic.validate_action_reference(center_coordinates, scale_coordinates)
        for critic in critics:
            critic.set_action_reference(center_coordinates, scale_coordinates)

    def critic_action_reference_contract(self) -> Dict[str, Any]:
        rows = []
        for name, critic in (
            ("critic1", self.critic1),
            ("critic2", self.critic2),
            ("target_critic1", self.target_critic1),
            ("target_critic2", self.target_critic2),
        ):
            rows.append(
                {
                    "name": name,
                    "configured": bool(critic.action_reference_configured),
                    "center_coordinates": critic.action_reference.detach().cpu().tolist(),
                    "scale_coordinates": critic.action_scale.detach().cpu().tolist(),
                }
            )
        return {
            "immutable_after_configuration": True,
            "scale_is_units_not_action_bound": True,
            "critics": rows,
        }

    def _normalized(self, observations: torch.Tensor) -> torch.Tensor:
        return self.observation_normalizer.normalize_tensor(observations)

    def _context(
        self,
        observations: torch.Tensor,
        previous_actions: torch.Tensor,
        previous_rewards: torch.Tensor,
        previous_dones: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            (
                self._normalized(observations),
                previous_actions,
                previous_rewards,
                previous_dones,
            ),
            dim=-1,
        )

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Weighted mean over targets, with zero weight for prefix/padding."""
        weights = mask.to(dtype=values.dtype)
        while weights.ndim < values.ndim:
            weights = weights.unsqueeze(-1)
        if weights.shape != values.shape:
            weights = weights.expand_as(values)
        denominator = weights.sum().clamp_min(1.0)
        return (values * weights).sum() / denominator

    @staticmethod
    def _replay_ht_mean(
        values: torch.Tensor, inclusion_weights: torch.Tensor
    ) -> torch.Tensor:
        """Horvitz--Thompson replay mean with a fixed batch denominator.

        A learn start is uniform over episode transitions.  The weighted sum
        within one sampled window has expectation equal to the episode-wide
        transition mean, but its realized weight sum is random.  Dividing by
        that random sum would form a biased Hájek ratio (severely so for small
        batches), therefore the only denominator here is the number of sampled
        windows.
        """
        if values.ndim < 2 or inclusion_weights.ndim != 2:
            raise ValueError("replay HT tensors must have batch and time axes")
        if values.shape[:2] != inclusion_weights.shape:
            raise ValueError("replay values and inclusion weights are misaligned")
        weights = inclusion_weights.to(dtype=values.dtype)
        while weights.ndim < values.ndim:
            weights = weights.unsqueeze(-1)
        if weights.shape != values.shape:
            weights = weights.expand_as(values)
        per_window = (values * weights).reshape(values.shape[0], -1).sum(dim=1)
        return per_window.mean()

    @torch.no_grad()
    def _behavior_clone_diagnostics(
        self,
        batch: Mapping[str, torch.Tensor],
        target_coordinates: torch.Tensor,
    ) -> Dict[str, Any]:
        context = self._context(
            batch["observations"],
            batch["previous_actions"],
            batch["previous_rewards"],
            batch["previous_dones"],
        )
        features, _ = self.actor.features(
            context,
            hidden=self.initial_recurrent_state(context.shape[0]),
            burn_in=0,
        )
        means, _ = self.actor.distribution(features[:, :-1])
        target = target_coordinates.expand_as(means)
        squared = (means - target).square().mean(dim=-1)
        full_mse = self._masked_mean(squared, batch["loss_mask"])
        reset_coordinates = means[:, 0].detach().cpu().numpy().astype(np.float64)
        target_array = (
            target_coordinates[0, 0].detach().cpu().numpy().astype(np.float64)
        )
        reset_physical = self.action_transform.coordinates_to_physical(
            reset_coordinates
        )
        target_physical = self.action_transform.coordinates_to_physical(target_array)
        coordinate_error = reset_coordinates - target_array[None, :]
        relative_physical_error = (
            (reset_physical - target_physical[None, :])
            / np.maximum(np.abs(target_physical[None, :]), 1.0e-12)
        )
        max_coordinate_error = float(np.max(np.abs(coordinate_error)))
        max_relative_physical_error = float(
            np.max(np.abs(relative_physical_error))
        )
        tolerance = 1.0e-5
        return {
            "full_episode_coordinate_mse": float(full_mse.cpu()),
            "reset_coordinate_rmse": float(
                np.sqrt(np.mean(np.square(coordinate_error)))
            ),
            "reset_max_abs_coordinate_error": max_coordinate_error,
            "reset_max_abs_relative_physical_error": max_relative_physical_error,
            "reset_actions_coordinates": reset_coordinates.tolist(),
            "reset_actions_physical": reset_physical.tolist(),
            "target_coordinates": target_array.tolist(),
            "target_physical_action": target_physical.tolist(),
            "reset_action_parity_tolerance": tolerance,
            "reset_action_parity_passed": bool(
                max_coordinate_error <= tolerance
                and max_relative_physical_error <= tolerance
            ),
            "warm_episode_count": int(reset_coordinates.shape[0]),
            "warm_transition_count": int(batch["loss_mask"].sum().item()),
        }

    def initial_recurrent_state(self, batch_size: int = 1) -> torch.Tensor:
        return torch.zeros(
            1,
            int(batch_size),
            int(self.config.hidden_dim),
            dtype=torch.float32,
            device=self.device,
        )

    @torch.no_grad()
    def act(
        self,
        observation: np.ndarray | Sequence[float],
        *,
        previous_action: Optional[np.ndarray | Sequence[float]] = None,
        previous_reward: float = 0.0,
        previous_done: bool = False,
        hidden: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, torch.Tensor, Dict[str, Any]]:
        observation_array = np.asarray(observation, dtype=np.float32).reshape(1, 1, -1)
        if observation_array.shape[-1] != self.observation_dim:
            raise ValueError(
                f"expected observation width {self.observation_dim}, got {observation_array.shape[-1]}"
            )
        if previous_action is None:
            previous_coordinates = np.zeros(self.action_dim, dtype=np.float32)
        else:
            previous_coordinates = self.action_transform.physical_to_coordinates(
                np.asarray(previous_action, dtype=np.float64)
            ).astype(np.float32)
        observations = torch.as_tensor(observation_array, device=self.device)
        pa = torch.as_tensor(previous_coordinates.reshape(1, 1, -1), device=self.device)
        pr = torch.tensor([[[float(previous_reward)]]], dtype=torch.float32, device=self.device)
        pd = torch.tensor([[[float(previous_done)]]], dtype=torch.float32, device=self.device)
        context = self._context(observations, pa, pr, pd)
        features, next_hidden = self.actor.features(context, hidden=hidden)
        coordinates, log_probability, mean = self.actor.sample_from_features(
            features, self.action_transform, deterministic=deterministic
        )
        coordinate_array = coordinates[0, 0].cpu().numpy().astype(np.float64)
        absolute = np.abs(coordinate_array)
        self.max_abs_coordinate = np.maximum(self.max_abs_coordinate, absolute)
        # Diagnostic only: this threshold never modifies or rejects the action.
        extreme = bool(np.any(absolute > 4.0))
        self.extreme_action_count += int(extreme)
        try:
            physical = self.action_transform.coordinates_to_physical(coordinate_array)
        except FloatingPointError as exc:
            self.nonfinite_action_rejections += 1
            raise NumericalPolicyActionError(coordinate_array, str(exc)) from exc
        self.max_physical_action = np.maximum(self.max_physical_action, physical)
        self.min_physical_action = np.minimum(self.min_physical_action, physical)
        info = {
            "policy_coordinates": coordinate_array,
            "log_probability": float(log_probability.item()),
            "mean_coordinates": mean[0, 0].cpu().numpy().astype(np.float64),
            "extreme_action": extreme,
        }
        return physical.astype(np.float64), next_hidden.detach(), info

    @torch.no_grad()
    def advance_actor_hidden(
        self,
        observation: np.ndarray | Sequence[float],
        *,
        previous_action: Optional[np.ndarray | Sequence[float]] = None,
        previous_reward: float = 0.0,
        previous_done: bool = False,
        hidden: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Consume an observation while a committed behavior action repeats."""
        observation_array = np.asarray(observation, dtype=np.float32).reshape(1, 1, -1)
        if observation_array.shape[-1] != self.observation_dim:
            raise ValueError(
                f"expected observation width {self.observation_dim}, got "
                f"{observation_array.shape[-1]}"
            )
        if previous_action is None:
            previous_coordinates = np.zeros(self.action_dim, dtype=np.float32)
        else:
            previous_coordinates = self.action_transform.physical_to_coordinates(
                np.asarray(previous_action, dtype=np.float64)
            ).astype(np.float32)
        context = self._context(
            torch.as_tensor(observation_array, device=self.device),
            torch.as_tensor(
                previous_coordinates.reshape(1, 1, -1), device=self.device
            ),
            torch.tensor(
                [[[float(previous_reward)]]], dtype=torch.float32, device=self.device
            ),
            torch.tensor(
                [[[float(previous_done)]]], dtype=torch.float32, device=self.device
            ),
        )
        _, next_hidden = self.actor.features(context, hidden=hidden)
        return next_hidden.detach()

    def behavior_clone(
        self,
        target_coordinates: np.ndarray | Sequence[float],
        *,
        steps: int,
        log_std_coordinates: Optional[np.ndarray | Sequence[float]] = None,
    ) -> Dict[str, Any]:
        """Clone a fixed rule on every warm-episode state, including reset.

        Warm episodes are unrolled from a zero hidden state and every action
        state contributes to the supervised loss.  Because the target is a
        constant rule, projecting the actor's mean output head to that exact
        constant is the closed-form initialization; full-episode gradient
        passes then verify and preserve the solution.
        """
        target = torch.as_tensor(
            np.asarray(target_coordinates, dtype=np.float32).reshape(1, 1, -1),
            device=self.device,
        )
        if not self.replay.episodes:
            return {
                "steps": 0.0,
                "initial_loss": float("nan"),
                "final_loss": float("nan"),
                "reset_action_parity_passed": False,
                "reason": "empty replay",
            }
        batch = self.replay.full_episode_batch(self.device)
        before = self._behavior_clone_diagnostics(batch, target)

        output = self.actor.head[-1]
        assert isinstance(output, nn.Linear)
        with torch.no_grad():
            output.weight[: self.action_dim].zero_()
            output.bias[: self.action_dim].copy_(target[0, 0])
            if log_std_coordinates is not None:
                log_std = torch.as_tensor(
                    np.asarray(log_std_coordinates, dtype=np.float32).reshape(
                        self.action_dim
                    ),
                    device=self.device,
                ).clamp(_LOG_STD_MIN, _LOG_STD_MAX)
                output.weight[self.action_dim :].zero_()
                output.bias[self.action_dim :].copy_(log_std)
        # A stale Adam moment can immediately pull an exact warm start away
        # from its target if behavior_clone is called on an existing agent.
        self.actor_optimizer.state.clear()
        projected = self._behavior_clone_diagnostics(batch, target)

        losses: List[float] = []
        for _ in range(int(max(0, steps))):
            context = self._context(
                batch["observations"],
                batch["previous_actions"],
                batch["previous_rewards"],
                batch["previous_dones"],
            )
            features, _ = self.actor.features(
                context,
                hidden=self.initial_recurrent_state(context.shape[0]),
                burn_in=0,
            )
            mean, _ = self.actor.distribution(features[:, :-1])
            squared = (mean - target.expand_as(mean)).square().mean(dim=-1)
            loss = self._masked_mean(squared, batch["loss_mask"])
            self.actor_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.actor.parameters(), float(self.config.gradient_clip)
            )
            self.actor_optimizer.step()
            losses.append(float(loss.detach().cpu()))
        final = self._behavior_clone_diagnostics(batch, target)
        return {
            "steps": float(len(losses)),
            "initial_loss": float(losses[0]) if losses else float("nan"),
            "final_loss": float(losses[-1]) if losses else float("nan"),
            "pre_clone": before,
            "after_closed_form_projection": projected,
            "final": final,
            "reset_action_parity_passed": bool(
                final["reset_action_parity_passed"]
            ),
        }

    def fit_critics(self, *, steps: int) -> Dict[str, float]:
        """Fit twin critics on replay without modifying actor/temperature."""
        losses: List[float] = []
        for _ in range(int(max(0, steps))):
            if not self.can_update():
                break
            batch = self.replay.sample(
                batch_size=int(self.config.batch_size),
                burn_in=int(self.config.burn_in),
                learn_length=int(self.config.learn_length),
                device=self.device,
            )
            context = self._context(
                batch["observations"],
                batch["previous_actions"],
                batch["previous_rewards"],
                batch["previous_dones"],
            )
            actions = batch["actions"]
            rewards = batch["rewards"].unsqueeze(-1)
            dones = batch["dones"].unsqueeze(-1)
            weights = batch["loss_weights"]
            features1, _ = self.critic1.features(context, burn_in=0)
            features2, _ = self.critic2.features(context, burn_in=0)
            current_q1 = self.critic1.q_from_features(features1[:, :-1], actions)
            current_q2 = self.critic2.q_from_features(features2[:, :-1], actions)
            with torch.no_grad():
                actor_features, _ = self.actor.features(context, burn_in=0)
                next_actions, next_logp, _ = self.actor.sample_from_features(
                    actor_features[:, 1:], self.action_transform
                )
                target1_features, _ = self.target_critic1.features(
                    context, burn_in=0
                )
                target2_features, _ = self.target_critic2.features(
                    context, burn_in=0
                )
                target_q = torch.minimum(
                    self.target_critic1.q_from_features(
                        target1_features[:, 1:], next_actions
                    ),
                    self.target_critic2.q_from_features(
                        target2_features[:, 1:], next_actions
                    ),
                ) - self.temperature.detach() * next_logp
                target = (
                    float(self.config.reward_scale) * rewards
                    + float(self.config.gamma) * (1.0 - dones) * target_q
                )
            loss1 = self._replay_ht_mean((current_q1 - target).square(), weights)
            loss2 = self._replay_ht_mean((current_q2 - target).square(), weights)
            loss = loss1 + loss2
            self.critic1_optimizer.zero_grad(set_to_none=True)
            self.critic2_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.critic1.parameters(), float(self.config.gradient_clip))
            nn.utils.clip_grad_norm_(self.critic2.parameters(), float(self.config.gradient_clip))
            self.critic1_optimizer.step()
            self.critic2_optimizer.step()
            tau = float(self.config.polyak_tau)
            with torch.no_grad():
                for target_parameter, parameter in zip(
                    self.target_critic1.parameters(), self.critic1.parameters()
                ):
                    target_parameter.mul_(1.0 - tau).add_(parameter, alpha=tau)
                for target_parameter, parameter in zip(
                    self.target_critic2.parameters(), self.critic2.parameters()
                ):
                    target_parameter.mul_(1.0 - tau).add_(parameter, alpha=tau)
            losses.append(float(loss.detach().cpu()))
        return {
            "steps": float(len(losses)),
            "initial_loss": float(losses[0]) if losses else float("nan"),
            "final_loss": float(losses[-1]) if losses else float("nan"),
        }

    def fit_critics_monte_carlo(
        self,
        *,
        steps: int,
        paired_advantage_steps: int = 0,
        paired_advantage_weight: float = 1.0,
        absolute_fit_group_id: str = "l0_train_absolute_fit",
        paired_fit_group_id: str = "l0_train_local_support",
    ) -> Dict[str, Any]:
        """Calibrate critics to full fixed-policy returns before actor updates.

        The local-support warm replay contains complete CRN trajectories.
        Regressing both critics on their realized return-to-go establishes an
        absolute value scale and local action ranking before any policy
        gradient is trusted.  This is a project-specific safe-improvement
        device inspired by delayed actor updates and baseline-regularized
        offline RL; it does not inherit formal guarantees from those methods.
        """
        advantage_weight = float(paired_advantage_weight)
        phase1_steps = int(max(0, steps))
        phase2_steps = int(max(0, paired_advantage_steps))
        if not np.isfinite(advantage_weight) or advantage_weight < 0.0:
            raise ValueError("paired_advantage_weight must be finite and non-negative")
        if not self.replay.episodes:
            return {
                "steps": 0.0,
                "initial_loss": float("nan"),
                "final_loss": float("nan"),
            }
        batch = self.replay.full_episode_batch(self.device)
        context = self._context(
            batch["observations"],
            batch["previous_actions"],
            batch["previous_rewards"],
            batch["previous_dones"],
        )
        actions = batch["actions"]
        rewards = batch["rewards"]
        dones = batch["dones"]
        mask = batch["loss_mask"].clone()
        # A local-support episode follows the warm policy from its recorded
        # deviation onward.  Earlier center actions have a scheduled future
        # deviation and therefore are not Q^warm targets; exclude that prefix
        # from the Monte-Carlo regression.
        for row, episode in enumerate(self.replay.episodes):
            deviation_day = int(episode.metadata.get("deviation_day", 0))
            if deviation_day > 0:
                mask[row, :deviation_day] = 0.0
        returns = torch.zeros_like(rewards)
        running = torch.zeros(rewards.shape[0], device=self.device)
        for index in range(rewards.shape[1] - 1, -1, -1):
            running = (
                float(self.config.reward_scale) * rewards[:, index]
                + float(self.config.gamma)
                * (1.0 - dones[:, index])
                * running
            )
            running = running * batch["sequence_mask"][:, index]
            returns[:, index] = running
        targets = returns.unsqueeze(-1)
        effective_horizon = effective_horizon_lengths(
            dones,
            batch["sequence_mask"],
            float(self.config.gamma),
        ).unsqueeze(-1)
        if bool((effective_horizon[mask.unsqueeze(-1) > 0.0] <= 0.0).any()):
            raise RuntimeError("valid Monte-Carlo target has non-positive effective horizon")

        def normalized_squared_error(
            prediction: torch.Tensor, target: torch.Tensor
        ) -> torch.Tensor:
            return ((prediction - target) / effective_horizon.clamp_min(1.0)).square()

        episodes = list(self.replay.episodes)
        production_pair_group = str(paired_fit_group_id)
        has_production_pair_group = any(
            str(episode.metadata.get("group_id", "")) == production_pair_group
            for episode in episodes
        )
        groups: Dict[Tuple[str, int, int], Dict[int, int]] = {}
        for row, episode in enumerate(episodes):
            metadata = episode.metadata
            if (
                (
                    has_production_pair_group
                    and str(metadata.get("group_id", ""))
                    != production_pair_group
                )
                or "support_index" not in metadata
                or "seed" not in metadata
            ):
                continue
            key = (
                str(metadata.get("group_id", "local_support")),
                int(metadata["seed"]),
                int(metadata.get("deviation_day", 0)),
            )
            groups.setdefault(key, {})[int(metadata["support_index"])] = row
        pair_rows: List[int] = []
        pair_centers: List[int] = []
        pair_days: List[int] = []
        for (_, _, day), members in groups.items():
            if 0 not in members:
                continue
            for support_index, row in sorted(members.items()):
                if support_index == 0:
                    continue
                pair_rows.append(int(row))
                pair_centers.append(int(members[0]))
                pair_days.append(int(day))
        pair_row_tensor = torch.as_tensor(
            pair_rows, dtype=torch.long, device=self.device
        )
        pair_center_tensor = torch.as_tensor(
            pair_centers, dtype=torch.long, device=self.device
        )
        pair_day_tensor = torch.as_tensor(
            pair_days, dtype=torch.long, device=self.device
        )

        def paired_terms(
            q1_values: torch.Tensor, q2_values: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            if not pair_rows:
                zero = q1_values.sum() * 0.0
                empty = torch.empty(0, device=self.device)
                return zero, zero, empty, empty, empty
            target_delta = (
                targets[pair_row_tensor, pair_day_tensor, 0]
                - targets[pair_center_tensor, pair_day_tensor, 0]
            )
            q1_delta = (
                q1_values[pair_row_tensor, pair_day_tensor, 0]
                - q1_values[pair_center_tensor, pair_day_tensor, 0]
            )
            q2_delta = (
                q2_values[pair_row_tensor, pair_day_tensor, 0]
                - q2_values[pair_center_tensor, pair_day_tensor, 0]
            )
            informative = (
                torch.isfinite(target_delta)
                & torch.isfinite(q1_delta)
                & torch.isfinite(q2_delta)
                & (target_delta.abs() > 1.0e-8)
            )
            if not bool(informative.any()):
                zero = q1_values.sum() * 0.0
                empty = torch.empty(0, device=self.device)
                return zero, zero, empty, empty, empty
            target_delta = target_delta[informative]
            q1_delta = q1_delta[informative]
            q2_delta = q2_delta[informative]
            return (
                (q1_delta - target_delta).square().mean(),
                (q2_delta - target_delta).square().mean(),
                target_delta,
                q1_delta,
                q2_delta,
            )

        def delta_metrics(
            target_delta: torch.Tensor,
            q1_delta: torch.Tensor,
            q2_delta: torch.Tensor,
        ) -> Dict[str, Any]:
            actual = target_delta.detach().cpu().numpy().astype(np.float64)
            first = q1_delta.detach().cpu().numpy().astype(np.float64)
            second = q2_delta.detach().cpu().numpy().astype(np.float64)
            minimum = np.minimum(first, second)

            def ranks(values: np.ndarray) -> np.ndarray:
                order = np.argsort(values, kind="mergesort")
                output = np.empty(values.size, dtype=np.float64)
                cursor = 0
                while cursor < values.size:
                    stop = cursor + 1
                    while stop < values.size and values[order[stop]] == values[order[cursor]]:
                        stop += 1
                    output[order[cursor:stop]] = 0.5 * (cursor + stop - 1)
                    cursor = stop
                return output

            def one(predicted: np.ndarray) -> Dict[str, float]:
                if actual.size == 0:
                    return {
                        "rmse": float("nan"),
                        "null_zero_delta_rmse": float("nan"),
                        "rmse_to_null_ratio": float("nan"),
                        "spearman": float("nan"),
                        "directional_accuracy": float("nan"),
                    }
                actual_ranks = ranks(actual)
                predicted_ranks = ranks(predicted)
                spearman = (
                    float(np.corrcoef(actual_ranks, predicted_ranks)[0, 1])
                    if np.std(actual_ranks) > 0.0 and np.std(predicted_ranks) > 0.0
                    else 0.0
                )
                rmse = float(np.sqrt(np.mean(np.square(predicted - actual))))
                null_rmse = float(np.sqrt(np.mean(np.square(actual))))
                return {
                    "rmse": rmse,
                    "null_zero_delta_rmse": null_rmse,
                    "rmse_to_null_ratio": float(
                        rmse / max(null_rmse, 1.0e-12)
                    ),
                    "spearman": spearman,
                    "directional_accuracy": float(
                        np.mean(np.sign(predicted) == np.sign(actual))
                    ),
                }

            return {
                "informative_pair_count": int(actual.size),
                "target_delta_rms": (
                    float(np.sqrt(np.mean(np.square(actual))))
                    if actual.size
                    else float("nan")
                ),
                "qmin": one(minimum),
                "q1": one(first),
                "q2": one(second),
            }

        reference = self.critic1.action_reference.to(
            dtype=actions.dtype, device=actions.device
        ).reshape(1, 1, -1)
        if not torch.equal(
            self.critic1.action_reference, self.critic2.action_reference
        ) or not torch.equal(self.critic1.action_scale, self.critic2.action_scale):
            raise RuntimeError("twin critics do not share the immutable action units")
        absolute_fit_rows = torch.as_tensor(
            [
                str(episode.metadata.get("group_id", ""))
                == str(absolute_fit_group_id)
                for episode in episodes
            ],
            dtype=torch.bool,
            device=self.device,
        )
        if bool(absolute_fit_rows.any()):
            absolute_fit_mask = (
                mask
                * absolute_fit_rows.to(mask.dtype).unsqueeze(1)
            )
            support_rows_in_absolute_fit = int(
                sum(
                    str(episode.metadata.get("group_id", ""))
                    == production_pair_group
                    and bool(absolute_fit_rows[row].item())
                    for row, episode in enumerate(episodes)
                )
            )
            if support_rows_in_absolute_fit:
                raise RuntimeError("paired support leaked into absolute-fit mask")
            center_mask = absolute_fit_mask
            absolute_fit_contract = {
                "dedicated_group": str(absolute_fit_group_id),
                "episode_count": int(absolute_fit_rows.sum().item()),
                "transition_count": int(center_mask.sum().item()),
                "paired_support_rows_excluded": True,
                "normalizer_source_is_independent_center_paths": True,
                "center_policy_may_be_recurrent": True,
            }
        else:
            # Generic/synthetic callers predating the L0 runner retain the
            # centered-action behavior; production warm start always takes
            # the dedicated-group branch above.
            center_mask = mask * (
                (actions - reference).abs().amax(dim=-1) <= 1.0e-6
            ).to(mask.dtype)
            absolute_fit_contract = {
                "dedicated_group": None,
                "episode_count": 0,
                "transition_count": int(center_mask.sum().item()),
                "paired_support_rows_excluded": False,
                "legacy_generic_fallback": True,
            }
        if float(center_mask.sum().cpu()) <= 0.0:
            raise RuntimeError("phase-1 critic fit has no warm-reference actions")

        critic1_encoder_parameters = list(self.critic1.encoder_parameters())
        critic2_encoder_parameters = list(self.critic2.encoder_parameters())
        critic1_value_head_parameters = list(self.critic1.value_head_parameters())
        critic2_value_head_parameters = list(self.critic2.value_head_parameters())
        critic1_value_parameters = (
            critic1_encoder_parameters + critic1_value_head_parameters
        )
        critic2_value_parameters = (
            critic2_encoder_parameters + critic2_value_head_parameters
        )
        critic1_advantage_parameters = list(self.critic1.advantage_parameters())
        critic2_advantage_parameters = list(self.critic2.advantage_parameters())

        def training_mode(
            *, encoder: bool, value_head: bool, advantage: bool
        ) -> None:
            for parameter in critic1_encoder_parameters + critic2_encoder_parameters:
                parameter.requires_grad_(encoder)
            for parameter in critic1_value_head_parameters + critic2_value_head_parameters:
                parameter.requires_grad_(value_head)
            for parameter in critic1_advantage_parameters + critic2_advantage_parameters:
                parameter.requires_grad_(advantage)

        @torch.no_grad()
        def parameter_snapshot() -> Dict[str, torch.Tensor]:
            def flattened(parameters: Sequence[nn.Parameter]) -> torch.Tensor:
                if not parameters:
                    return torch.empty(0, device=self.device)
                return torch.cat([parameter.detach().reshape(-1) for parameter in parameters])

            return {
                "critic1_encoder": flattened(critic1_encoder_parameters),
                "critic1_value_head": flattened(critic1_value_head_parameters),
                "critic1_advantage": flattened(critic1_advantage_parameters),
                "critic2_encoder": flattened(critic2_encoder_parameters),
                "critic2_value_head": flattened(critic2_value_head_parameters),
                "critic2_advantage": flattened(critic2_advantage_parameters),
            }

        def parameter_delta(
            before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor]
        ) -> Dict[str, float]:
            return {
                name: float(torch.linalg.vector_norm(after[name] - value).cpu())
                for name, value in before.items()
            }

        def optimizer_step(
            loss1: torch.Tensor,
            loss2: torch.Tensor,
            parameters1: Sequence[nn.Parameter],
            parameters2: Sequence[nn.Parameter],
        ) -> Tuple[float, float]:
            self.critic1_optimizer.zero_grad(set_to_none=True)
            self.critic2_optimizer.zero_grad(set_to_none=True)
            (loss1 + loss2).backward()
            norm1 = nn.utils.clip_grad_norm_(
                parameters1, float(self.config.gradient_clip)
            )
            norm2 = nn.utils.clip_grad_norm_(
                parameters2, float(self.config.gradient_clip)
            )
            self.critic1_optimizer.step()
            self.critic2_optimizer.step()
            return float(norm1), float(norm2)

        @torch.no_grad()
        def snapshot() -> Dict[str, Any]:
            features1, _ = self.critic1.features(context, burn_in=0)
            features2, _ = self.critic2.features(context, burn_in=0)
            q1 = self.critic1.q_from_features(features1[:, :-1], actions)
            q2 = self.critic2.q_from_features(features2[:, :-1], actions)
            raw_absolute_loss = self._masked_mean(
                (q1 - targets).square(), center_mask
            ) + self._masked_mean((q2 - targets).square(), center_mask)
            normalized_absolute_loss = self._masked_mean(
                normalized_squared_error(q1, targets), center_mask
            ) + self._masked_mean(
                normalized_squared_error(q2, targets), center_mask
            )
            advantage_loss1, advantage_loss2, target_delta, q1_delta, q2_delta = (
                paired_terms(q1, q2)
            )
            advantage_loss = advantage_loss1 + advantage_loss2
            return {
                "absolute_mc_loss": float(raw_absolute_loss.cpu()),
                "effective_horizon_normalized_absolute_loss": float(
                    normalized_absolute_loss.cpu()
                ),
                "paired_advantage_loss": float(advantage_loss.cpu()),
                "joint_loss_at_weight": float(
                    (
                        normalized_absolute_loss
                        + advantage_weight * advantage_loss
                    ).cpu()
                ),
                "mean_q": float(
                    self._masked_mean(torch.minimum(q1, q2), center_mask).cpu()
                ),
                "paired_advantage": delta_metrics(
                    target_delta, q1_delta, q2_delta
                ),
            }

        def gradient_summary(values: Sequence[float]) -> Dict[str, Any]:
            threshold = float(self.config.gradient_clip)
            array = np.asarray(values, dtype=np.float64)
            return {
                "mean_before_clip": float(np.mean(array)) if array.size else float("nan"),
                "max_before_clip": float(np.max(array)) if array.size else float("nan"),
                "clip_threshold": threshold,
                "clipped_count": int(np.sum(array > threshold)),
                "clipping_fraction": (
                    float(np.mean(array > threshold)) if array.size else 0.0
                ),
            }

        def run_phase1(count: int) -> Dict[str, Any]:
            total_losses: List[float] = []
            gradient_norms: List[float] = []
            parameters_before = parameter_snapshot()
            training_mode(encoder=True, value_head=True, advantage=False)
            for _ in range(int(count)):
                features1, _ = self.critic1.features(context, burn_in=0)
                features2, _ = self.critic2.features(context, burn_in=0)
                q1 = self.critic1.q_from_features(features1[:, :-1], actions)
                q2 = self.critic2.q_from_features(features2[:, :-1], actions)
                loss1 = self._masked_mean(
                    normalized_squared_error(q1, targets), center_mask
                )
                loss2 = self._masked_mean(
                    normalized_squared_error(q2, targets), center_mask
                )
                norm1, norm2 = optimizer_step(
                    loss1,
                    loss2,
                    critic1_value_parameters,
                    critic2_value_parameters,
                )
                total_losses.append(float((loss1 + loss2).detach().cpu()))
                gradient_norms.extend((norm1, norm2))
            parameters_after = parameter_snapshot()
            return {
                "steps": int(count),
                "objective": "warm-reference_effective-horizon-normalized_mc_only",
                "q_and_return_targets_remain_raw": True,
                "advantage_parameters_frozen": True,
                "initial_total_loss": (
                    total_losses[0] if total_losses else float("nan")
                ),
                "final_total_loss": (
                    total_losses[-1] if total_losses else float("nan")
                ),
                "gradient_norms": gradient_summary(gradient_norms),
                "parameter_delta_l2": parameter_delta(
                    parameters_before, parameters_after
                ),
            }

        def run_phase2(count: int) -> Dict[str, Any]:
            paired_losses: List[float] = []
            refresh_losses: List[float] = []
            paired_gradient_norms: List[float] = []
            refresh_gradient_norms: List[float] = []
            parameters_before = parameter_snapshot()
            for _ in range(int(count)):
                # Action effects only: recurrent features and value heads are
                # frozen, so paired gradients cannot rewrite the baseline.
                training_mode(encoder=False, value_head=False, advantage=True)
                with torch.no_grad():
                    features1, _ = self.critic1.features(context, burn_in=0)
                    features2, _ = self.critic2.features(context, burn_in=0)
                if pair_rows:
                    target_delta = (
                        targets[pair_row_tensor, pair_day_tensor, 0]
                        - targets[pair_center_tensor, pair_day_tensor, 0]
                    )
                    center_features1 = features1[
                        pair_center_tensor, pair_day_tensor
                    ].detach()
                    center_features2 = features2[
                        pair_center_tensor, pair_day_tensor
                    ].detach()
                    candidate_actions = actions[
                        pair_row_tensor, pair_day_tensor
                    ]
                    center_actions = actions[
                        pair_center_tensor, pair_day_tensor
                    ]
                    q1_delta = (
                        self.critic1.q_from_features(
                            center_features1, candidate_actions
                        )
                        - self.critic1.q_from_features(
                            center_features1, center_actions
                        )
                    ).squeeze(-1)
                    q2_delta = (
                        self.critic2.q_from_features(
                            center_features2, candidate_actions
                        )
                        - self.critic2.q_from_features(
                            center_features2, center_actions
                        )
                    ).squeeze(-1)
                    informative = (
                        torch.isfinite(target_delta)
                        & torch.isfinite(q1_delta)
                        & torch.isfinite(q2_delta)
                        & (target_delta.abs() > 1.0e-8)
                    )
                    if not bool(informative.any()):
                        raise RuntimeError(
                            "phase-2 paired critic fit has no finite informative probes"
                        )
                    advantage_loss1 = (
                        q1_delta[informative] - target_delta[informative]
                    ).square().mean()
                    advantage_loss2 = (
                        q2_delta[informative] - target_delta[informative]
                    ).square().mean()
                else:
                    raise RuntimeError(
                        "phase-2 paired critic fit has no matched support groups"
                    )
                norm1, norm2 = optimizer_step(
                    advantage_weight * advantage_loss1,
                    advantage_weight * advantage_loss2,
                    critic1_advantage_parameters,
                    critic2_advantage_parameters,
                )
                paired_losses.append(
                    float((advantage_loss1 + advantage_loss2).detach().cpu())
                )
                paired_gradient_norms.extend((norm1, norm2))

                # Separate baseline refresh: advantage heads are frozen and
                # the exact centered action makes A(c)=0 by construction.
                training_mode(encoder=False, value_head=True, advantage=False)
                with torch.no_grad():
                    refresh_features1, _ = self.critic1.features(context, burn_in=0)
                    refresh_features2, _ = self.critic2.features(context, burn_in=0)
                refresh_q1 = self.critic1.value_head(
                    refresh_features1.detach()[:, :-1]
                )
                refresh_q2 = self.critic2.value_head(
                    refresh_features2.detach()[:, :-1]
                )
                refresh_loss1 = self._masked_mean(
                    normalized_squared_error(refresh_q1, targets), center_mask
                )
                refresh_loss2 = self._masked_mean(
                    normalized_squared_error(refresh_q2, targets), center_mask
                )
                norm1, norm2 = optimizer_step(
                    refresh_loss1,
                    refresh_loss2,
                    critic1_value_parameters,
                    critic2_value_parameters,
                )
                refresh_losses.append(
                    float((refresh_loss1 + refresh_loss2).detach().cpu())
                )
                refresh_gradient_norms.extend((norm1, norm2))
            training_mode(encoder=True, value_head=True, advantage=True)
            parameters_after = parameter_snapshot()
            return {
                "steps": int(count),
                "objective": (
                    "raw_paired_structured_advantage_plus_separate_"
                    "effective-horizon-normalized_absolute_refresh"
                ),
                "paired_advantage_targets_remain_raw": True,
                "recurrent_features_detached_for_paired_loss": True,
                "recurrent_encoder_frozen_for_entire_phase": True,
                "absolute_refresh_updates_value_head_only": True,
                "paired_initial_loss": (
                    paired_losses[0] if paired_losses else float("nan")
                ),
                "paired_final_loss": (
                    paired_losses[-1] if paired_losses else float("nan")
                ),
                "absolute_refresh_initial_loss": (
                    refresh_losses[0] if refresh_losses else float("nan")
                ),
                "absolute_refresh_final_loss": (
                    refresh_losses[-1] if refresh_losses else float("nan")
                ),
                "paired_gradient_norms": gradient_summary(paired_gradient_norms),
                "absolute_refresh_gradient_norms": gradient_summary(
                    refresh_gradient_norms
                ),
                "parameter_delta_l2": parameter_delta(
                    parameters_before, parameters_after
                ),
            }

        initialization = snapshot()
        phase1_training = run_phase1(phase1_steps)
        after_phase1 = snapshot()

        optimizer_state_entries_before_reset = {
            "critic1": len(self.critic1_optimizer.state),
            "critic2": len(self.critic2_optimizer.state),
        }
        self.critic1_optimizer.state.clear()
        self.critic2_optimizer.state.clear()
        optimizer_reset = {
            "performed": True,
            "boundary": "after_phase1_before_phase2",
            "state_entries_before": optimizer_state_entries_before_reset,
            "state_entries_after": {
                "critic1": len(self.critic1_optimizer.state),
                "critic2": len(self.critic2_optimizer.state),
            },
            "learning_rates_preserved": {
                "critic1": [
                    float(group["lr"])
                    for group in self.critic1_optimizer.param_groups
                ],
                "critic2": [
                    float(group["lr"])
                    for group in self.critic2_optimizer.param_groups
                ],
            },
        }
        phase2_training = run_phase2(phase2_steps)
        after_phase2 = snapshot()

        # The target networks inherit only the final two-phase calibration.
        if phase1_steps + phase2_steps > 0:
            self.target_critic1.load_state_dict(self.critic1.state_dict())
            self.target_critic2.load_state_dict(self.critic2.state_dict())
        return {
            "steps": float(phase1_steps + phase2_steps),
            "optimizer_steps_per_critic": int(phase1_steps + 2 * phase2_steps),
            "phase1_steps": int(phase1_steps),
            "phase2_steps": int(phase2_steps),
            "initial_loss": float(
                initialization["effective_horizon_normalized_absolute_loss"]
            ),
            "final_loss": float(after_phase2["joint_loss_at_weight"]),
            "initial_absolute_mc_loss": float(initialization["absolute_mc_loss"]),
            "phase1_absolute_mc_loss": float(after_phase1["absolute_mc_loss"]),
            "final_absolute_mc_loss": float(after_phase2["absolute_mc_loss"]),
            "initial_effective_horizon_normalized_absolute_loss": float(
                initialization["effective_horizon_normalized_absolute_loss"]
            ),
            "phase1_effective_horizon_normalized_absolute_loss": float(
                after_phase1["effective_horizon_normalized_absolute_loss"]
            ),
            "final_effective_horizon_normalized_absolute_loss": float(
                after_phase2["effective_horizon_normalized_absolute_loss"]
            ),
            "initial_paired_advantage_loss": float(
                initialization["paired_advantage_loss"]
            ),
            "final_paired_advantage_loss": float(
                after_phase2["paired_advantage_loss"]
            ),
            "paired_advantage_weight": advantage_weight,
            "two_phase_contract": {
                "fixed_training_budgets": True,
                "early_stopping": False,
                "heldout_gate_used_for_stopping_or_tuning": False,
                "phase2_optimizer_steps_per_critic": int(2 * phase2_steps),
                "phase2_substeps_per_nominal_step": [
                    "paired_advantage_head",
                    "absolute_value_head_refresh",
                ],
                "phase1": phase1_training,
                "optimizer_moment_reset": optimizer_reset,
                "phase2": phase2_training,
            },
            "calibration_snapshots": {
                "initialization": initialization,
                "after_phase1": after_phase1,
                "after_phase2": after_phase2,
            },
            "paired_advantage": {
                "initial": initialization["paired_advantage"],
                "after_phase1": after_phase1["paired_advantage"],
                "final": after_phase2["paired_advantage"],
                "group_count": int(len(groups)),
                "pair_count_before_informative_mask": int(len(pair_rows)),
                "loss": "independently averaged per-twin MSE of seed-centered probe deltas",
            },
            "absolute_fit_contract": absolute_fit_contract,
            "absolute_loss_scale_contract": {
                "training_loss": "mean(((Q-G)/effective_horizon)^2)",
                "effective_horizon_recurrence": (
                    "d_t=1+gamma*(1-done_t)*d_(t+1), padding=0"
                ),
                "q_values_and_returns_reported_in_raw_reward_units": True,
                "paired_advantage_objective_normalized": False,
                "online_td_objective_normalized": False,
                "minimum_effective_horizon": float(
                    effective_horizon[center_mask.unsqueeze(-1) > 0.0].min().cpu()
                ),
                "maximum_effective_horizon": float(
                    effective_horizon[center_mask.unsqueeze(-1) > 0.0].max().cpu()
                ),
            },
            "initial_mean_q": float(initialization["mean_q"]),
            "final_mean_q": float(after_phase2["mean_q"]),
            "mean_return_to_go": float(
                self._masked_mean(targets, center_mask).detach().cpu()
            ),
        }

    @torch.no_grad()
    def critic_validation(self, episodes: Sequence[Episode]) -> Dict[str, Any]:
        """Evaluate critics on held-out training episodes without mutation."""
        if not episodes:
            raise ValueError("critic validation needs at least one episode")
        replay = SequenceReplayBuffer(
            capacity_transitions=sum(episode.length for episode in episodes),
            seed=int(self.config.seed) + 91_337,
        )
        for episode in episodes:
            replay.add(episode)
        batch = replay.full_episode_batch(self.device)
        context = self._context(
            batch["observations"],
            batch["previous_actions"],
            batch["previous_rewards"],
            batch["previous_dones"],
        )
        actions = batch["actions"]
        rewards = batch["rewards"].unsqueeze(-1)
        dones = batch["dones"].unsqueeze(-1)
        mask = batch["loss_mask"]
        critic1_features, _ = self.critic1.features(context, burn_in=0)
        critic2_features, _ = self.critic2.features(context, burn_in=0)
        components1 = self.critic1.structured_components(
            critic1_features[:, :-1], actions
        )
        components2 = self.critic2.structured_components(
            critic2_features[:, :-1], actions
        )
        q1 = components1["q"]
        q2 = components2["q"]
        predicted_q = torch.minimum(q1, q2)

        actor_features, _ = self.actor.features(context, burn_in=0)
        next_actions, next_log_probability, _ = self.actor.sample_from_features(
            actor_features[:, 1:], self.action_transform, deterministic=True
        )
        target1_features, _ = self.target_critic1.features(context, burn_in=0)
        target2_features, _ = self.target_critic2.features(context, burn_in=0)
        target_q = torch.minimum(
            self.target_critic1.q_from_features(
                target1_features[:, 1:], next_actions
            ),
            self.target_critic2.q_from_features(
                target2_features[:, 1:], next_actions
            ),
        ) - self.temperature.detach() * next_log_probability
        td_target = (
            float(self.config.reward_scale) * rewards
            + float(self.config.gamma) * (1.0 - dones) * target_q
        )
        td_mse = self._masked_mean((predicted_q - td_target).square(), mask)
        disagreement = self._masked_mean((q1 - q2).abs(), mask)

        rows = []
        mc_squared_errors = []
        mc_return_values = []
        for row, episode in enumerate(episodes):
            running = 0.0
            returns = np.zeros(episode.length, dtype=np.float64)
            for index in range(episode.length - 1, -1, -1):
                running = (
                    float(self.config.reward_scale) * float(episode.rewards[index])
                    + float(self.config.gamma) * running
                )
                returns[index] = running
            predicted = (
                predicted_q[row, : episode.length, 0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            predicted1 = (
                q1[row, : episode.length, 0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            predicted2 = (
                q2[row, : episode.length, 0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            probe_index = int(episode.metadata.get("deviation_day", 0))
            if probe_index < 0 or probe_index >= episode.length:
                raise ValueError("critic probe transition lies outside episode")
            mc_squared_errors.extend(
                np.square(predicted[probe_index:] - returns[probe_index:]).tolist()
            )
            mc_return_values.extend(returns[probe_index:].tolist())
            rows.append(
                {
                    "episode_index": int(row),
                    "q0": float(predicted[0]),
                    "return0": float(returns[0]),
                    "mean_q": float(np.mean(predicted)),
                    "mean_return_to_go": float(np.mean(returns)),
                    "probe_transition": probe_index,
                    "q_probe": float(predicted[probe_index]),
                    "q1_probe": float(predicted1[probe_index]),
                    "q2_probe": float(predicted2[probe_index]),
                    "return_probe": float(returns[probe_index]),
                    "metadata": copy.deepcopy(episode.metadata),
                    "critic1_value_probe": float(
                        components1["value"][row, probe_index, 0].cpu()
                    ),
                    "critic2_value_probe": float(
                        components2["value"][row, probe_index, 0].cpu()
                    ),
                    "critic1_slope_probe": components1["slope"][
                        row, probe_index
                    ].cpu().tolist(),
                    "critic2_slope_probe": components2["slope"][
                        row, probe_index
                    ].cpu().tolist(),
                    "critic1_lower_cholesky_probe": components1[
                        "lower_cholesky"
                    ][row, probe_index].cpu().tolist(),
                    "critic2_lower_cholesky_probe": components2[
                        "lower_cholesky"
                    ][row, probe_index].cpu().tolist(),
                }
            )
        values = np.asarray(
            [float(td_mse.cpu()), float(disagreement.cpu()), *mc_squared_errors],
            dtype=np.float64,
        )
        q0 = np.asarray([row["q0"] for row in rows], dtype=np.float64)
        return0 = np.asarray([row["return0"] for row in rows], dtype=np.float64)
        q0_absolute_error = np.abs(q0 - return0)
        return_scale = max(float(np.mean(np.abs(return0))), 1.0e-8)
        mc_return_scale = max(
            float(np.mean(np.abs(np.asarray(mc_return_values, dtype=np.float64)))),
            1.0e-8,
        )
        q_probe = np.asarray([row["q_probe"] for row in rows], dtype=np.float64)
        return_probe = np.asarray(
            [row["return_probe"] for row in rows], dtype=np.float64
        )
        probe_scale = max(float(np.mean(np.abs(return_probe))), 1.0e-8)
        probe_absolute_error = np.abs(q_probe - return_probe)
        entropy_scale = max(
            float(self.temperature.detach().cpu()) * abs(float(self.target_entropy)),
            1.0e-8,
        )
        return {
            "finite": bool(np.all(np.isfinite(values))),
            "structured_quadratic_value_saturation_count": int(
                components1["quadratic_value_saturated"].sum().cpu()
                + components2["quadratic_value_saturated"].sum().cpu()
            ),
            "td_mse": float(td_mse.cpu()),
            "twin_q_mean_absolute_disagreement": float(disagreement.cpu()),
            "monte_carlo_mse": float(np.mean(mc_squared_errors)),
            "monte_carlo_normalized_rmse": float(
                np.sqrt(np.mean(mc_squared_errors)) / mc_return_scale
            ),
            "monte_carlo_valid_suffix_return_scale": mc_return_scale,
            "q0_mean_absolute_error": float(np.mean(q0_absolute_error)),
            "q0_relative_absolute_error": float(
                np.mean(q0_absolute_error) / return_scale
            ),
            "probe_mean_absolute_error": float(np.mean(probe_absolute_error)),
            "probe_relative_absolute_error": float(
                np.mean(probe_absolute_error) / probe_scale
            ),
            "q0_to_entropy_scale_ratio": float(
                np.median(np.abs(q0)) / entropy_scale
            ),
            "probe_q_to_entropy_scale_ratio": float(
                np.median(np.abs(q_probe)) / entropy_scale
            ),
            "q0_standard_deviation": float(
                np.std([row["q0"] for row in rows], ddof=0)
            ),
            "episodes": rows,
        }

    def can_update(self, batch_size: Optional[int] = None) -> bool:
        del batch_size
        return bool(self.replay.episodes and len(self.replay) >= self.minimum_replay_transitions)

    def reset_replay(self, *, seed: Optional[int] = None) -> None:
        """Discard data after an environment-distribution version change."""
        replay_seed = int(self.config.seed) + 17 if seed is None else int(seed)
        self.replay = SequenceReplayBuffer(self.config.replay_capacity, seed=replay_seed)

    def configure_safe_policy_improvement(
        self,
        *,
        actor_updates_enabled: bool,
        warm_anchor_coordinates: Optional[np.ndarray | Sequence[float]],
        anchor_coefficient: float,
        anchor_decay_updates: float,
        anchor_scale: Optional[np.ndarray | Sequence[float]] = None,
        actor_update_interval: int = 2,
    ) -> None:
        """Configure critic-gated actor learning around a frozen warm rule."""
        coefficient = float(anchor_coefficient)
        decay = float(anchor_decay_updates)
        if not np.isfinite(coefficient) or coefficient < 0.0:
            raise ValueError("anchor_coefficient must be finite and non-negative")
        if not np.isfinite(decay) or decay <= 0.0:
            raise ValueError("anchor_decay_updates must be finite and positive")
        if warm_anchor_coordinates is None:
            anchor = None
        else:
            anchor = np.asarray(warm_anchor_coordinates, dtype=np.float64).reshape(
                self.action_dim
            )
            if not np.all(np.isfinite(anchor)):
                raise ValueError("warm anchor coordinates must be finite")
        if anchor_scale is None:
            scale = np.ones(self.action_dim, dtype=np.float64)
        else:
            scale = np.asarray(anchor_scale, dtype=np.float64).reshape(
                self.action_dim
            )
            if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
                raise ValueError("anchor_scale must be finite and positive")
        interval = int(actor_update_interval)
        if interval <= 0:
            raise ValueError("actor_update_interval must be positive")
        self.actor_updates_enabled = bool(actor_updates_enabled)
        self.warm_anchor_coordinates = None if anchor is None else anchor.copy()
        self.warm_anchor_initial_coefficient = coefficient
        self.warm_anchor_decay_updates = decay
        self.warm_anchor_scale = scale.copy()
        self.actor_updates_applied = 0
        self.accepted_actor_blocks = 0
        self.actor_update_interval = interval

    def policy_training_state(self) -> Dict[str, Any]:
        """Snapshot every policy-side quantity needed for safe fallback."""
        return {
            "actor": copy.deepcopy(self.actor.state_dict()),
            "actor_optimizer": copy.deepcopy(self.actor_optimizer.state_dict()),
            "log_temperature": self.log_temperature.detach().cpu().clone(),
            "temperature_optimizer": copy.deepcopy(
                self.temperature_optimizer.state_dict()
            ),
            "actor_updates_enabled": bool(self.actor_updates_enabled),
            "warm_anchor_coordinates": (
                None
                if self.warm_anchor_coordinates is None
                else self.warm_anchor_coordinates.copy()
            ),
            "warm_anchor_initial_coefficient": float(
                self.warm_anchor_initial_coefficient
            ),
            "warm_anchor_decay_updates": float(self.warm_anchor_decay_updates),
            "warm_anchor_scale": self.warm_anchor_scale.copy(),
            "actor_updates_applied": int(self.actor_updates_applied),
            "accepted_actor_blocks": int(self.accepted_actor_blocks),
            "actor_update_interval": int(self.actor_update_interval),
        }

    def load_policy_training_state(self, state: Mapping[str, Any]) -> None:
        """Restore a frozen candidate/reference policy without touching critics."""
        self.actor.load_state_dict(state["actor"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        with torch.no_grad():
            self.log_temperature.copy_(state["log_temperature"].to(self.device))
        self.temperature_optimizer.load_state_dict(state["temperature_optimizer"])
        self.actor_updates_enabled = bool(state["actor_updates_enabled"])
        anchor = state.get("warm_anchor_coordinates")
        self.warm_anchor_coordinates = (
            None if anchor is None else np.asarray(anchor, dtype=np.float64).copy()
        )
        self.warm_anchor_initial_coefficient = float(
            state.get("warm_anchor_initial_coefficient", 0.0)
        )
        self.warm_anchor_decay_updates = float(
            state.get("warm_anchor_decay_updates", 1.0)
        )
        self.warm_anchor_scale = np.asarray(
            state.get("warm_anchor_scale", np.ones(self.action_dim)),
            dtype=np.float64,
        ).copy()
        self.actor_updates_applied = int(state.get("actor_updates_applied", 0))
        self.accepted_actor_blocks = int(state.get("accepted_actor_blocks", 0))
        self.actor_update_interval = int(state.get("actor_update_interval", 1))

    def full_training_state(self) -> Dict[str, Any]:
        """Snapshot all mutable learner state needed for an exact rollback."""
        cuda_rng = None
        if torch.cuda.is_available():
            cuda_rng = [value.cpu().clone() for value in torch.cuda.get_rng_state_all()]
        return {
            "actor": copy.deepcopy(self.actor.state_dict()),
            "critic1": copy.deepcopy(self.critic1.state_dict()),
            "critic2": copy.deepcopy(self.critic2.state_dict()),
            "target_critic1": copy.deepcopy(self.target_critic1.state_dict()),
            "target_critic2": copy.deepcopy(self.target_critic2.state_dict()),
            "module_training_modes": {
                "actor": bool(self.actor.training),
                "critic1": bool(self.critic1.training),
                "critic2": bool(self.critic2.training),
                "target_critic1": bool(self.target_critic1.training),
                "target_critic2": bool(self.target_critic2.training),
            },
            "actor_optimizer": copy.deepcopy(self.actor_optimizer.state_dict()),
            "critic1_optimizer": copy.deepcopy(self.critic1_optimizer.state_dict()),
            "critic2_optimizer": copy.deepcopy(self.critic2_optimizer.state_dict()),
            "critic1_advantage_optimizer": copy.deepcopy(
                self.critic1_advantage_optimizer.state_dict()
            ),
            "critic2_advantage_optimizer": copy.deepcopy(
                self.critic2_advantage_optimizer.state_dict()
            ),
            "temperature_optimizer": copy.deepcopy(
                self.temperature_optimizer.state_dict()
            ),
            "log_temperature": self.log_temperature.detach().cpu().clone(),
            "observation_normalizer": copy.deepcopy(
                self.observation_normalizer.state_dict()
            ),
            "replay": copy.deepcopy(self.replay),
            "paired_crn_cache": copy.deepcopy(self._paired_crn_cache),
            "paired_crn_cache_contract": copy.deepcopy(
                self._paired_crn_cache_contract
            ),
            "paired_advantage_online_weight": float(
                self._paired_advantage_online_weight
            ),
            "same_version_gate_required": bool(
                self._same_version_gate_required
            ),
            "agent_rng_state": copy.deepcopy(self.rng.bit_generator.state),
            "torch_rng_state": torch.get_rng_state().clone(),
            "torch_cuda_rng_state_all": cuda_rng,
            "updates": int(self.updates),
            "actor_updates_enabled": bool(self.actor_updates_enabled),
            "actor_updates_applied": int(self.actor_updates_applied),
            "accepted_actor_blocks": int(self.accepted_actor_blocks),
            "actor_update_interval": int(self.actor_update_interval),
            "warm_anchor_coordinates": (
                None
                if self.warm_anchor_coordinates is None
                else self.warm_anchor_coordinates.copy()
            ),
            "warm_anchor_initial_coefficient": float(
                self.warm_anchor_initial_coefficient
            ),
            "warm_anchor_decay_updates": float(self.warm_anchor_decay_updates),
            "warm_anchor_scale": self.warm_anchor_scale.copy(),
            "extreme_action_count": int(self.extreme_action_count),
            "nonfinite_action_rejections": int(self.nonfinite_action_rejections),
            "max_abs_coordinate": self.max_abs_coordinate.copy(),
            "max_physical_action": self.max_physical_action.copy(),
            "min_physical_action": self.min_physical_action.copy(),
            "training_terminal_failure": bool(self.training_terminal_failure),
            "training_terminal_failure_reason": self.training_terminal_failure_reason,
            "safe_gate_failures": int(self.safe_gate_failures),
            "last_passing_capture_count": int(self._last_passing_capture_count),
        }

    def load_full_training_state(self, state: Mapping[str, Any]) -> None:
        """Restore a full state captured by :meth:`full_training_state`."""
        self.actor.load_state_dict(state["actor"])
        self.critic1.load_state_dict(state["critic1"])
        self.critic2.load_state_dict(state["critic2"])
        self.target_critic1.load_state_dict(state["target_critic1"])
        self.target_critic2.load_state_dict(state["target_critic2"])
        modes = dict(state["module_training_modes"])
        self.actor.train(bool(modes["actor"]))
        self.critic1.train(bool(modes["critic1"]))
        self.critic2.train(bool(modes["critic2"]))
        self.target_critic1.train(bool(modes["target_critic1"]))
        self.target_critic2.train(bool(modes["target_critic2"]))
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic1_optimizer.load_state_dict(state["critic1_optimizer"])
        self.critic2_optimizer.load_state_dict(state["critic2_optimizer"])
        self.critic1_advantage_optimizer.load_state_dict(
            state["critic1_advantage_optimizer"]
        )
        self.critic2_advantage_optimizer.load_state_dict(
            state["critic2_advantage_optimizer"]
        )
        self.temperature_optimizer.load_state_dict(
            state["temperature_optimizer"]
        )
        with torch.no_grad():
            self.log_temperature.copy_(state["log_temperature"].to(self.device))
        self.observation_normalizer.load_state_dict(
            state["observation_normalizer"]
        )
        self.replay = copy.deepcopy(state["replay"])
        self._paired_crn_cache = tuple(
            self._immutable_episode_copy(episode)
            for episode in state["paired_crn_cache"]
        )
        self._paired_crn_cache_contract = copy.deepcopy(
            state["paired_crn_cache_contract"]
        )
        self._paired_advantage_online_weight = float(
            state["paired_advantage_online_weight"]
        )
        self._same_version_gate_required = bool(
            state["same_version_gate_required"]
        )
        self.rng.bit_generator.state = copy.deepcopy(state["agent_rng_state"])
        self.updates = int(state["updates"])
        self.actor_updates_enabled = bool(state["actor_updates_enabled"])
        self.actor_updates_applied = int(state["actor_updates_applied"])
        self.accepted_actor_blocks = int(state.get("accepted_actor_blocks", 0))
        self.actor_update_interval = int(state["actor_update_interval"])
        anchor = state["warm_anchor_coordinates"]
        self.warm_anchor_coordinates = (
            None if anchor is None else np.asarray(anchor, dtype=np.float64).copy()
        )
        self.warm_anchor_initial_coefficient = float(
            state["warm_anchor_initial_coefficient"]
        )
        self.warm_anchor_decay_updates = float(state["warm_anchor_decay_updates"])
        self.warm_anchor_scale = np.asarray(
            state["warm_anchor_scale"], dtype=np.float64
        ).copy()
        self.extreme_action_count = int(state["extreme_action_count"])
        self.nonfinite_action_rejections = int(
            state["nonfinite_action_rejections"]
        )
        self.max_abs_coordinate = np.asarray(
            state["max_abs_coordinate"], dtype=np.float64
        ).copy()
        self.max_physical_action = np.asarray(
            state["max_physical_action"], dtype=np.float64
        ).copy()
        self.min_physical_action = np.asarray(
            state["min_physical_action"], dtype=np.float64
        ).copy()
        self.training_terminal_failure = bool(state["training_terminal_failure"])
        self.training_terminal_failure_reason = state[
            "training_terminal_failure_reason"
        ]
        self.safe_gate_failures = int(state["safe_gate_failures"])
        self._last_passing_capture_count = int(
            state["last_passing_capture_count"]
        )
        torch.set_rng_state(state["torch_rng_state"].cpu())
        cuda_rng = state.get("torch_cuda_rng_state_all")
        if cuda_rng is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_rng)

    def capture_last_passing_state(self, label: str) -> Dict[str, Any]:
        if self.training_terminal_failure:
            raise RuntimeError("cannot checkpoint a terminally failed learner")
        self._last_passing_capture_count += 1
        self._last_passing_label = str(label)
        self._last_passing_state = self.full_training_state()
        return self.last_passing_state_contract()

    def last_passing_state_contract(self) -> Dict[str, Any]:
        return {
            "available": self._last_passing_state is not None,
            "label": self._last_passing_label,
            "capture_count": int(self._last_passing_capture_count),
            "full_state": True,
            "includes_replay_rng_torch_rng": True,
        }

    def restore_last_passing_state(self, reason: str) -> Dict[str, Any]:
        if self._last_passing_state is None:
            raise RuntimeError("no passing learner state is available for rollback")
        checkpoint = copy.deepcopy(self._last_passing_state)
        label = self._last_passing_label
        self.load_full_training_state(checkpoint)
        self.safe_gate_failures += 1
        self.training_terminal_failure = True
        self.training_terminal_failure_reason = str(reason)
        self.actor_updates_enabled = False
        return {
            "performed": True,
            "restored_label": label,
            "terminal": True,
            "reason": str(reason),
            "restored_capture_count": int(self._last_passing_capture_count),
        }

    def training_transaction_state(self) -> Dict[str, Any]:
        """Capture learner state plus the distinct critic-safe pointer."""
        return {
            "learner": self.full_training_state(),
            "critic_safe_state": copy.deepcopy(self._last_passing_state),
            "critic_safe_label": self._last_passing_label,
            "critic_safe_capture_count": int(self._last_passing_capture_count),
        }

    def load_training_transaction_state(self, state: Mapping[str, Any]) -> None:
        """Restore a complete nonterminal training transaction."""
        self.load_full_training_state(state["learner"])
        self._last_passing_state = copy.deepcopy(state["critic_safe_state"])
        self._last_passing_label = state["critic_safe_label"]
        self._last_passing_capture_count = int(
            state["critic_safe_capture_count"]
        )

    def initialize_economic_incumbent(self, label: str) -> Dict[str, Any]:
        """Freeze the first economically deployable state without a gain claim."""
        if self.training_terminal_failure:
            raise RuntimeError("cannot initialize an incumbent from a failed learner")
        if self._economic_incumbent_state is not None:
            raise RuntimeError("economic incumbent is already initialized")
        self._economic_incumbent_label = str(label)
        self._economic_incumbent_capture_count = 1
        self._economic_incumbent_state = self.training_transaction_state()
        return self.economic_incumbent_contract()

    def begin_economic_stage(self, label: str) -> Dict[str, Any]:
        """Start a new stage-local conservative-improvement transaction.

        The entering policy is the new stage's economic reference.  Replacing
        the prior stage pointer is deliberate: every later-stage proposal is
        compared with, and can roll back exactly to, the policy that entered
        that stage after its stage-specific TRAIN critic gate passed.
        """
        if self.training_terminal_failure:
            raise RuntimeError("cannot begin an economic stage from a failed learner")
        self._economic_incumbent_label = str(label)
        self._economic_incumbent_capture_count += 1
        self._economic_incumbent_state = self.training_transaction_state()
        return self.economic_incumbent_contract()

    def accept_economic_incumbent(self, label: str) -> Dict[str, Any]:
        """Accept a statistically safe actor block and advance both pointers."""
        if self.training_terminal_failure:
            raise RuntimeError("cannot accept an incumbent from a failed learner")
        if self._economic_incumbent_state is None:
            raise RuntimeError("economic incumbent is not initialized")
        self.accepted_actor_blocks += 1
        # The current critic version already passed its same-version gate.
        # Refresh its pointer so an eventual critic failure restores the new
        # accepted-block counter together with the accepted actor.
        self.capture_last_passing_state(
            f"critic_safe_at_economic_accept_{self.accepted_actor_blocks}"
        )
        self._economic_incumbent_label = str(label)
        self._economic_incumbent_capture_count += 1
        self._economic_incumbent_state = self.training_transaction_state()
        return self.economic_incumbent_contract()

    def advance_economic_incumbent_without_actor_change(
        self, label: str
    ) -> Dict[str, Any]:
        """Retain critic/replay progress when an episode produced no actor proposal."""
        if self.training_terminal_failure:
            raise RuntimeError("cannot advance an incumbent from a failed learner")
        if self._economic_incumbent_state is None:
            raise RuntimeError("economic incumbent is not initialized")
        self._economic_incumbent_label = str(label)
        self._economic_incumbent_capture_count += 1
        self._economic_incumbent_state = self.training_transaction_state()
        return self.economic_incumbent_contract()

    def restore_economic_incumbent(self, reason: str) -> Dict[str, Any]:
        """Rollback a harmful actor proposal without terminating training."""
        if self._economic_incumbent_state is None:
            raise RuntimeError("economic incumbent is not initialized")
        state = copy.deepcopy(self._economic_incumbent_state)
        label = self._economic_incumbent_label
        self.load_training_transaction_state(state)
        if self.training_terminal_failure:
            raise RuntimeError("economic incumbent unexpectedly contains terminal state")
        return {
            "performed": True,
            "terminal": False,
            "restored_label": label,
            "reason": str(reason),
            "critic_safe_pointer_restored": True,
            "full_state_rng_replay_restored": True,
        }

    def economic_incumbent_transaction_state(self) -> Dict[str, Any]:
        if self._economic_incumbent_state is None:
            raise RuntimeError("economic incumbent is not initialized")
        return copy.deepcopy(self._economic_incumbent_state)

    def economic_incumbent_contract(self) -> Dict[str, Any]:
        return {
            "available": self._economic_incumbent_state is not None,
            "label": self._economic_incumbent_label,
            "capture_count": int(self._economic_incumbent_capture_count),
            "accepted_actor_blocks": int(self.accepted_actor_blocks),
            "distinct_from_critic_safe_pointer": True,
            "full_state_rng_replay": True,
        }

    def paired_advantage_refresh(self) -> Dict[str, Any]:
        """One TRAIN-fit-only advantage-head refresh at current recurrent features."""
        if not self._paired_crn_cache:
            raise RuntimeError("paired CRN refresh cache is not configured")
        transient = SequenceReplayBuffer(
            capacity_transitions=sum(
                episode.length for episode in self._paired_crn_cache
            ),
            seed=0,
        )
        for episode in self._paired_crn_cache:
            transient.add(episode)
        batch = transient.full_episode_batch(self.device)
        context = self._context(
            batch["observations"],
            batch["previous_actions"],
            batch["previous_rewards"],
            batch["previous_dones"],
        )
        actions = batch["actions"]
        rewards = batch["rewards"]
        dones = batch["dones"]
        sequence_mask = batch["sequence_mask"]
        returns = torch.zeros_like(rewards)
        running = torch.zeros(rewards.shape[0], device=self.device)
        for index in range(rewards.shape[1] - 1, -1, -1):
            running = (
                float(self.config.reward_scale) * rewards[:, index]
                + float(self.config.gamma)
                * (1.0 - dones[:, index])
                * running
            )
            running = running * sequence_mask[:, index]
            returns[:, index] = running

        groups: Dict[Tuple[str, int, int], Dict[int, int]] = {}
        for row, episode in enumerate(self._paired_crn_cache):
            metadata = episode.metadata
            key = (
                str(metadata["group_id"]),
                int(metadata["seed"]),
                int(metadata.get("deviation_day", 0)),
            )
            groups.setdefault(key, {})[int(metadata["support_index"])] = row
        pair_rows: List[int] = []
        center_rows: List[int] = []
        pair_days: List[int] = []
        for (_, _, day), members in sorted(groups.items()):
            if 0 not in members:
                raise RuntimeError("paired CRN cache group has no center")
            for support_index, row in sorted(members.items()):
                if support_index == 0:
                    continue
                pair_rows.append(int(row))
                center_rows.append(int(members[0]))
                pair_days.append(int(day))
        if not pair_rows:
            raise RuntimeError("paired CRN cache contains no action deviations")
        pair_row_tensor = torch.as_tensor(
            pair_rows, dtype=torch.long, device=self.device
        )
        center_row_tensor = torch.as_tensor(
            center_rows, dtype=torch.long, device=self.device
        )
        pair_day_tensor = torch.as_tensor(
            pair_days, dtype=torch.long, device=self.device
        )
        target_delta = (
            returns[pair_row_tensor, pair_day_tensor]
            - returns[center_row_tensor, pair_day_tensor]
        )
        with torch.no_grad():
            features1, _ = self.critic1.features(context, burn_in=0)
            features2, _ = self.critic2.features(context, burn_in=0)
            center_features1 = features1[
                center_row_tensor, pair_day_tensor
            ].detach()
            center_features2 = features2[
                center_row_tensor, pair_day_tensor
            ].detach()
        candidate_actions = actions[pair_row_tensor, pair_day_tensor]
        center_actions = actions[center_row_tensor, pair_day_tensor]

        critic1_encoder = list(self.critic1.encoder_parameters())
        critic2_encoder = list(self.critic2.encoder_parameters())
        critic1_value = list(self.critic1.value_head_parameters())
        critic2_value = list(self.critic2.value_head_parameters())
        critic1_advantage = list(self.critic1.advantage_parameters())
        critic2_advantage = list(self.critic2.advantage_parameters())
        before_frozen = {
            "critic1_encoder": [value.detach().clone() for value in critic1_encoder],
            "critic1_value": [value.detach().clone() for value in critic1_value],
            "critic2_encoder": [value.detach().clone() for value in critic2_encoder],
            "critic2_value": [value.detach().clone() for value in critic2_value],
        }
        for parameter in (
            *critic1_encoder,
            *critic2_encoder,
            *critic1_value,
            *critic2_value,
        ):
            parameter.requires_grad_(False)
        for parameter in (*critic1_advantage, *critic2_advantage):
            parameter.requires_grad_(True)
        try:
            q1_delta = (
                self.critic1.q_from_features(
                    center_features1, candidate_actions
                )
                - self.critic1.q_from_features(
                    center_features1, center_actions
                )
            ).squeeze(-1)
            q2_delta = (
                self.critic2.q_from_features(
                    center_features2, candidate_actions
                )
                - self.critic2.q_from_features(
                    center_features2, center_actions
                )
            ).squeeze(-1)
            informative = (
                torch.isfinite(target_delta)
                & torch.isfinite(q1_delta)
                & torch.isfinite(q2_delta)
                & (target_delta.abs() > 1.0e-8)
            )
            if not bool(informative.any()):
                raise RuntimeError("paired CRN cache has no finite informative deltas")
            loss1 = (
                q1_delta[informative] - target_delta[informative]
            ).square().mean()
            loss2 = (
                q2_delta[informative] - target_delta[informative]
            ).square().mean()
            weight = float(self._paired_advantage_online_weight)
            if weight == 0.0:
                # Do not even call Adam.step(): a zero tensor gradient can
                # still move parameters through populated momentum buffers.
                grad1 = torch.zeros((), device=self.device)
                grad2 = torch.zeros((), device=self.device)
                optimizer_step_performed = False
            else:
                self.critic1_advantage_optimizer.zero_grad(set_to_none=True)
                self.critic2_advantage_optimizer.zero_grad(set_to_none=True)
                (weight * (loss1 + loss2)).backward()
                grad1 = nn.utils.clip_grad_norm_(
                    critic1_advantage, float(self.config.gradient_clip)
                )
                grad2 = nn.utils.clip_grad_norm_(
                    critic2_advantage, float(self.config.gradient_clip)
                )
                self.critic1_advantage_optimizer.step()
                self.critic2_advantage_optimizer.step()
                optimizer_step_performed = True
        finally:
            for parameter in (
                *critic1_encoder,
                *critic2_encoder,
                *critic1_value,
                *critic2_value,
                *critic1_advantage,
                *critic2_advantage,
            ):
                parameter.requires_grad_(True)

        frozen_unchanged = all(
            torch.equal(before, after.detach())
            for name, parameters in (
                ("critic1_encoder", critic1_encoder),
                ("critic1_value", critic1_value),
                ("critic2_encoder", critic2_encoder),
                ("critic2_value", critic2_value),
            )
            for before, after in zip(before_frozen[name], parameters)
        )
        if not frozen_unchanged:
            raise RuntimeError("paired refresh mutated encoder/value parameters")
        return {
            "performed": True,
            "source_split": "train_fit",
            "heldout_gate_episodes_used": False,
            "cache_sha256": self._paired_crn_cache_contract["sha256"],
            "pair_count": int(informative.sum().item()),
            "loss_q1": float(loss1.detach().cpu()),
            "loss_q2": float(loss2.detach().cpu()),
            "weight": float(self._paired_advantage_online_weight),
            "dedicated_advantage_optimizers": True,
            "optimizer_step_performed": bool(optimizer_step_performed),
            "zero_weight_literal_no_op": bool(weight == 0.0),
            "gradient_norm_q1": float(grad1),
            "gradient_norm_q2": float(grad2),
            "features_recomputed_and_detached": True,
            "encoder_value_parameters_unchanged": True,
        }

    def _polyak_update_targets(self) -> None:
        tau = float(self.config.polyak_tau)
        with torch.no_grad():
            for target_parameter, parameter in zip(
                self.target_critic1.parameters(), self.critic1.parameters()
            ):
                target_parameter.mul_(1.0 - tau).add_(parameter, alpha=tau)
            for target_parameter, parameter in zip(
                self.target_critic2.parameters(), self.critic2.parameters()
            ):
                target_parameter.mul_(1.0 - tau).add_(parameter, alpha=tau)

    def update(
        self,
        batch_size: Optional[int] = None,
        *,
        critic_gate: Optional[
            Callable[["RecurrentSACAgent", int], Mapping[str, Any]]
        ] = None,
        restore_economic_incumbent_on_gate_failure: bool = False,
    ) -> Dict[str, Any]:
        if self.training_terminal_failure:
            raise RuntimeError(
                "learner is terminally failed and cannot continue training"
            )
        if self._same_version_gate_required:
            if critic_gate is None:
                raise RuntimeError(
                    "same-version held-out TRAIN critic gate is required for every update"
                )
            if self._last_passing_state is None:
                raise RuntimeError(
                    "capture a full passing warm state before online critic updates"
                )
            if (
                restore_economic_incumbent_on_gate_failure
                and self._economic_incumbent_state is None
            ):
                raise RuntimeError(
                    "economic gate-failure rollback requires an initialized incumbent"
                )
        batch = self.replay.sample(
            batch_size=int(batch_size or self.config.batch_size),
            burn_in=int(self.config.burn_in),
            learn_length=int(self.config.learn_length),
            device=self.device,
        )
        context = self._context(
            batch["observations"],
            batch["previous_actions"],
            batch["previous_rewards"],
            batch["previous_dones"],
        )
        actions = batch["actions"]
        rewards = batch["rewards"].unsqueeze(-1)
        dones = batch["dones"].unsqueeze(-1)
        target_mask = batch["loss_mask"]
        weights = batch["loss_weights"]

        critic1_features, _ = self.critic1.features(context, burn_in=0)
        critic2_features, _ = self.critic2.features(context, burn_in=0)
        current_q1 = self.critic1.q_from_features(critic1_features[:, :-1], actions)
        current_q2 = self.critic2.q_from_features(critic2_features[:, :-1], actions)

        with torch.no_grad():
            actor_features, _ = self.actor.features(context, burn_in=0)
            next_actions, next_log_probability, _ = self.actor.sample_from_features(
                actor_features[:, 1:], self.action_transform
            )
            target1_features, _ = self.target_critic1.features(context, burn_in=0)
            target2_features, _ = self.target_critic2.features(context, burn_in=0)
            target_q1 = self.target_critic1.q_from_features(
                target1_features[:, 1:], next_actions
            )
            target_q2 = self.target_critic2.q_from_features(
                target2_features[:, 1:], next_actions
            )
            target_q = torch.minimum(target_q1, target_q2) - self.temperature.detach() * next_log_probability
            target = float(self.config.reward_scale) * rewards + float(self.config.gamma) * (1.0 - dones) * target_q

        critic_loss1 = self._replay_ht_mean((current_q1 - target).square(), weights)
        critic_loss2 = self._replay_ht_mean((current_q2 - target).square(), weights)
        critic_loss = critic_loss1 + critic_loss2
        self.critic1_optimizer.zero_grad(set_to_none=True)
        self.critic2_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad1 = nn.utils.clip_grad_norm_(
            self.critic1.parameters(), float(self.config.gradient_clip)
        )
        critic_grad2 = nn.utils.clip_grad_norm_(
            self.critic2.parameters(), float(self.config.gradient_clip)
        )
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()
        critic_grad = torch.maximum(
            torch.as_tensor(critic_grad1), torch.as_tensor(critic_grad2)
        )

        if self._paired_crn_cache:
            paired_refresh = self.paired_advantage_refresh()
        else:
            paired_refresh = {
                "performed": False,
                "source_split": None,
                "heldout_gate_episodes_used": False,
                "pair_count": 0,
                "loss_q1": 0.0,
                "loss_q2": 0.0,
                "weight": 0.0,
                "dedicated_advantage_optimizers": True,
                "optimizer_step_performed": False,
                "zero_weight_literal_no_op": True,
                "gradient_norm_q1": 0.0,
                "gradient_norm_q2": 0.0,
                "features_recomputed_and_detached": False,
                "encoder_value_parameters_unchanged": True,
            }

        # Targets must see both components of the critic update.  The
        # same-version gate below therefore evaluates the exact online/target
        # pair that would authorize the actor update.
        self._polyak_update_targets()
        gate_row: Optional[Mapping[str, Any]] = None
        if self._same_version_gate_required:
            assert critic_gate is not None
            gate_row = dict(critic_gate(self, int(self.updates + 1)))
            gate_passed = bool(gate_row.get("passed", False))
            self.actor_updates_enabled = gate_passed
        else:
            gate_passed = bool(self.actor_updates_enabled)

        selected_indices = batch["transition_indices"][target_mask.bool()]
        selected_dones = batch["dones"][target_mask.bool()]
        if self._same_version_gate_required and not gate_passed:
            rejected_update = int(self.updates + 1)
            reason = (
                "same-version held-out TRAIN critic gate failed after "
                "TD+paired refresh"
            )
            gate_details = dict(gate_row.get("gate", gate_row) or {})
            gate_diagnostics = dict(gate_row.get("diagnostics", {}) or {})
            structural_failure = bool(
                gate_details.get("coverage_passed") is False
                or gate_diagnostics.get("finite") is False
            )
            recoverable_critic_proposal = bool(
                restore_economic_incumbent_on_gate_failure
                and not structural_failure
            )
            if recoverable_critic_proposal:
                rollback = self.restore_economic_incumbent(reason)
            else:
                rollback = self.restore_last_passing_state(reason)
            terminal_failure = bool(rollback["terminal"])
            return {
                "critic_loss": float(critic_loss.detach().item()),
                "actor_loss": 0.0,
                "rl_actor_objective": 0.0,
                "warm_anchor_coefficient": float(
                    self.warm_anchor_initial_coefficient
                ),
                "warm_anchor_coordinate_mse": 0.0,
                "actor_update_applied": 0.0,
                "actor_updates_applied_total": float(
                    self.actor_updates_applied
                ),
                "accepted_actor_blocks_total": float(
                    self.accepted_actor_blocks
                ),
                "temperature_loss": 0.0,
                "temperature": float(self.temperature.detach().item()),
                "mean_log_probability": 0.0,
                "mean_q": float(
                    self._replay_ht_mean(
                        torch.minimum(current_q1, current_q2).detach(), weights
                    ).item()
                ),
                "target_q": float(
                    self._replay_ht_mean(target.detach(), weights).item()
                ),
                "critic_grad_norm": float(critic_grad),
                "actor_grad_norm": 0.0,
                "paired_refresh_loss": float(
                    paired_refresh["loss_q1"] + paired_refresh["loss_q2"]
                ),
                "same_version_gate_passed": 0.0,
                "terminal_training_failure": float(terminal_failure),
                "critic_proposal_rejection": float(
                    recoverable_critic_proposal
                ),
                "structural_gate_failure": float(structural_failure),
                "rollback_performed": 1.0,
                "rejected_update": float(rejected_update),
                "loss_transition_count": float(selected_indices.numel()),
                "minimum_loss_transition_index": float(
                    selected_indices.min().item()
                ),
                "maximum_loss_transition_index": float(
                    selected_indices.max().item()
                ),
                "episode_opening_targets": float(
                    (selected_indices == 0).sum().item()
                ),
                "terminal_targets": float(
                    (selected_dones >= 1.0).sum().item()
                ),
                "mean_actual_burn": float(
                    batch["actual_burn"].float().mean().item()
                ),
                "effective_loss_weight": float(weights.sum().item()),
                "updates": float(self.updates),
                "paired_advantage_refresh": paired_refresh,
                "safe_critic_gate": copy.deepcopy(dict(gate_row or {})),
                "rollback": rollback,
            }

        for parameter in (*self.critic1.parameters(), *self.critic2.parameters()):
            parameter.requires_grad_(False)
        actor_features, _ = self.actor.features(context, burn_in=0)
        sampled_actions, log_probability, actor_mean = self.actor.sample_from_features(
            actor_features[:, :-1], self.action_transform
        )
        with torch.no_grad():
            q1_features, _ = self.critic1.features(context, burn_in=0)
            q2_features, _ = self.critic2.features(context, burn_in=0)
        sampled_q1 = self.critic1.q_from_features(q1_features[:, :-1].detach(), sampled_actions)
        sampled_q2 = self.critic2.q_from_features(q2_features[:, :-1].detach(), sampled_actions)
        actor_objective = (
            self.temperature.detach() * log_probability - torch.minimum(sampled_q1, sampled_q2)
        )
        rl_actor_loss = self._replay_ht_mean(actor_objective, weights)
        anchor_coefficient = float(
            self.warm_anchor_initial_coefficient
            * max(
                0.0,
                1.0
                - float(self.accepted_actor_blocks)
                / float(self.warm_anchor_decay_updates),
            )
        )
        if self.warm_anchor_coordinates is None:
            anchor_divergence = torch.zeros((), device=self.device)
            anchor_coefficient = 0.0
        else:
            anchor = torch.as_tensor(
                self.warm_anchor_coordinates,
                dtype=actor_mean.dtype,
                device=self.device,
            ).reshape(1, 1, -1)
            anchor_scale = torch.as_tensor(
                self.warm_anchor_scale,
                dtype=actor_mean.dtype,
                device=self.device,
            ).reshape(1, 1, -1)
            coordinate_mse = ((actor_mean - anchor) / anchor_scale).square().mean(
                dim=-1, keepdim=True
            )
            anchor_divergence = self._replay_ht_mean(coordinate_mse, weights)
        actor_loss = rl_actor_loss + anchor_coefficient * anchor_divergence
        actor_update_due = bool(
            gate_passed
            and (self.updates + 1) % int(self.actor_update_interval) == 0
        )
        if actor_update_due:
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_grad = nn.utils.clip_grad_norm_(
                self.actor.parameters(), float(self.config.gradient_clip)
            )
            self.actor_optimizer.step()
            self.actor_updates_applied += 1
        else:
            actor_grad = torch.zeros((), device=self.device)
        for parameter in (*self.critic1.parameters(), *self.critic2.parameters()):
            parameter.requires_grad_(True)

        proposed_temperature_loss = -self._replay_ht_mean(
            self.log_temperature
            * (log_probability.detach() + self.target_entropy),
            weights,
        )
        if actor_update_due:
            self.temperature_optimizer.zero_grad(set_to_none=True)
            proposed_temperature_loss.backward()
            self.temperature_optimizer.step()
            temperature_loss = proposed_temperature_loss
        else:
            temperature_loss = torch.zeros((), device=self.device)

        self.updates += 1
        checkpoint_contract = None
        if self._same_version_gate_required:
            checkpoint_contract = self.capture_last_passing_state(
                f"same_version_gate_update_{self.updates}"
            )
        result: Dict[str, Any] = {
            "critic_loss": float(critic_loss.detach().item()),
            "actor_loss": float(actor_loss.detach().item()),
            "rl_actor_objective": float(rl_actor_loss.detach().item()),
            "warm_anchor_coefficient": float(anchor_coefficient),
            "warm_anchor_coordinate_mse": float(
                anchor_divergence.detach().item()
            ),
            "actor_update_applied": float(actor_update_due),
            "actor_updates_applied_total": float(self.actor_updates_applied),
            "accepted_actor_blocks_total": float(self.accepted_actor_blocks),
            "temperature_loss": float(temperature_loss.detach().item()),
            "temperature": float(self.temperature.detach().item()),
            "mean_log_probability": float(
                self._replay_ht_mean(log_probability.detach(), weights).item()
            ),
            "mean_q": float(
                self._replay_ht_mean(
                    torch.minimum(current_q1, current_q2).detach(), weights
                ).item()
            ),
            "target_q": float(self._replay_ht_mean(target.detach(), weights).item()),
            "critic_grad_norm": float(critic_grad),
            "actor_grad_norm": float(actor_grad),
            "paired_refresh_loss": float(
                paired_refresh["loss_q1"] + paired_refresh["loss_q2"]
            ),
            "same_version_gate_passed": float(gate_passed),
            "terminal_training_failure": 0.0,
            "rollback_performed": 0.0,
            "loss_transition_count": float(selected_indices.numel()),
            "minimum_loss_transition_index": float(selected_indices.min().item()),
            "maximum_loss_transition_index": float(selected_indices.max().item()),
            "episode_opening_targets": float((selected_indices == 0).sum().item()),
            "terminal_targets": float((selected_dones >= 1.0).sum().item()),
            "mean_actual_burn": float(batch["actual_burn"].float().mean().item()),
            "effective_loss_weight": float(weights.sum().item()),
            "updates": float(self.updates),
        }
        if gate_row is not None:
            result["paired_advantage_refresh"] = paired_refresh
            result["safe_critic_gate"] = copy.deepcopy(dict(gate_row))
            result["last_passing_checkpoint"] = checkpoint_contract
        return result

    def action_diagnostics(self) -> Dict[str, Any]:
        return {
            "names": list(self.action_transform.names),
            "kinds": list(self.action_transform.kinds),
            "extreme_action_count": int(self.extreme_action_count),
            "nonfinite_action_rejections": int(self.nonfinite_action_rejections),
            "max_abs_policy_coordinate": self.max_abs_coordinate.tolist(),
            "min_physical_action": self.min_physical_action.tolist(),
            "max_physical_action": self.max_physical_action.tolist(),
        }

    def save(self, path: str | Path, metadata: Optional[Mapping[str, Any]] = None) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "ride_response_recurrent_sac_v2_structured_critic",
            "observation_dim": self.observation_dim,
            "action_transform": asdict(self.action_transform),
            "config": asdict(self.config),
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "target_critic1": self.target_critic1.state_dict(),
            "target_critic2": self.target_critic2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic1_optimizer": self.critic1_optimizer.state_dict(),
            "critic2_optimizer": self.critic2_optimizer.state_dict(),
            "critic1_advantage_optimizer": (
                self.critic1_advantage_optimizer.state_dict()
            ),
            "critic2_advantage_optimizer": (
                self.critic2_advantage_optimizer.state_dict()
            ),
            "temperature_optimizer": self.temperature_optimizer.state_dict(),
            "log_temperature": self.log_temperature.detach().cpu(),
            "observation_normalizer": self.observation_normalizer.state_dict(),
            "updates": self.updates,
            "safe_policy_improvement": {
                "actor_updates_enabled": bool(self.actor_updates_enabled),
                "warm_anchor_coordinates": (
                    None
                    if self.warm_anchor_coordinates is None
                    else self.warm_anchor_coordinates.tolist()
                ),
                "warm_anchor_initial_coefficient": float(
                    self.warm_anchor_initial_coefficient
                ),
                "warm_anchor_decay_updates": float(
                    self.warm_anchor_decay_updates
                ),
                "warm_anchor_scale": self.warm_anchor_scale.tolist(),
                "actor_updates_applied": int(self.actor_updates_applied),
                "accepted_actor_blocks": int(self.accepted_actor_blocks),
                "actor_update_interval": int(self.actor_update_interval),
            },
            "economic_incumbent": {
                "state": copy.deepcopy(self._economic_incumbent_state),
                "label": self._economic_incumbent_label,
                "capture_count": int(self._economic_incumbent_capture_count),
            },
            "resume_contract": {
                "full_training_resume": False,
                "economic_incumbent_included": bool(
                    self._economic_incumbent_state is not None
                ),
                "purpose": "evaluation/deployment checkpoint with durable incumbent",
            },
            "action_diagnostics": self.action_diagnostics(),
            "metadata": dict(metadata or {}),
        }
        torch.save(payload, destination)

    @classmethod
    def load(cls, path: str | Path, device: Optional[str] = None) -> "RecurrentSACAgent":
        source = Path(path)
        try:
            payload = torch.load(source, map_location=device or "cpu", weights_only=False)
        except TypeError:
            payload = torch.load(source, map_location=device or "cpu")
        if payload.get("format") != "ride_response_recurrent_sac_v2_structured_critic":
            raise ValueError("unsupported recurrent SAC artifact")
        config_values = dict(payload["config"])
        if device is not None:
            config_values["device"] = str(device)
        transform = ActionTransform(**payload["action_transform"])
        agent = cls(
            observation_dim=int(payload["observation_dim"]),
            action_transform=transform,
            config=RecurrentSACConfig(**config_values),
        )
        agent.actor.load_state_dict(payload["actor"])
        agent.critic1.load_state_dict(payload["critic1"])
        agent.critic2.load_state_dict(payload["critic2"])
        agent.target_critic1.load_state_dict(payload["target_critic1"])
        agent.target_critic2.load_state_dict(payload["target_critic2"])
        agent.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        agent.critic1_optimizer.load_state_dict(payload["critic1_optimizer"])
        agent.critic2_optimizer.load_state_dict(payload["critic2_optimizer"])
        if "critic1_advantage_optimizer" in payload:
            agent.critic1_advantage_optimizer.load_state_dict(
                payload["critic1_advantage_optimizer"]
            )
        if "critic2_advantage_optimizer" in payload:
            agent.critic2_advantage_optimizer.load_state_dict(
                payload["critic2_advantage_optimizer"]
            )
        agent.temperature_optimizer.load_state_dict(payload["temperature_optimizer"])
        with torch.no_grad():
            agent.log_temperature.copy_(payload["log_temperature"].to(agent.device))
        agent.observation_normalizer.load_state_dict(payload["observation_normalizer"])
        agent.updates = int(payload.get("updates", 0))
        safe = dict(payload.get("safe_policy_improvement", {}) or {})
        agent.actor_updates_enabled = bool(
            safe.get("actor_updates_enabled", True)
        )
        anchor = safe.get("warm_anchor_coordinates")
        agent.warm_anchor_coordinates = (
            None if anchor is None else np.asarray(anchor, dtype=np.float64)
        )
        agent.warm_anchor_initial_coefficient = float(
            safe.get("warm_anchor_initial_coefficient", 0.0)
        )
        agent.warm_anchor_decay_updates = float(
            safe.get("warm_anchor_decay_updates", 1.0)
        )
        agent.warm_anchor_scale = np.asarray(
            safe.get("warm_anchor_scale", np.ones(agent.action_dim)),
            dtype=np.float64,
        )
        agent.actor_updates_applied = int(
            safe.get("actor_updates_applied", 0)
        )
        agent.accepted_actor_blocks = int(
            safe.get("accepted_actor_blocks", 0)
        )
        agent.actor_update_interval = int(safe.get("actor_update_interval", 1))
        incumbent = dict(payload.get("economic_incumbent", {}) or {})
        agent._economic_incumbent_state = copy.deepcopy(incumbent.get("state"))
        agent._economic_incumbent_label = incumbent.get("label")
        agent._economic_incumbent_capture_count = int(
            incumbent.get("capture_count", 0)
        )
        diagnostics = payload.get("action_diagnostics", {})
        agent.extreme_action_count = int(diagnostics.get("extreme_action_count", 0))
        agent.nonfinite_action_rejections = int(
            diagnostics.get("nonfinite_action_rejections", 0)
        )
        if "max_abs_policy_coordinate" in diagnostics:
            agent.max_abs_coordinate = np.asarray(
                diagnostics["max_abs_policy_coordinate"], dtype=np.float64
            )
        if "min_physical_action" in diagnostics:
            agent.min_physical_action = np.asarray(
                diagnostics["min_physical_action"], dtype=np.float64
            )
        if "max_physical_action" in diagnostics:
            agent.max_physical_action = np.asarray(
                diagnostics["max_physical_action"], dtype=np.float64
            )
        return agent


__all__ = [
    "ActionTransform",
    "Episode",
    "EpisodeBuilder",
    "NumericalPolicyActionError",
    "RecurrentSACAgent",
    "RecurrentSACConfig",
    "SequenceReplayBuffer",
]
