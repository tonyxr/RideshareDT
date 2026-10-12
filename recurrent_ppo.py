"""Compact recurrent PPO control for the continuous v28 market actions.

This module is an on-policy benchmark, not the primary learner.  It shares the
same action-coordinate contract as :mod:`recurrent_sac`: tariff and driver
incentive multipliers are Gaussian in uncapped log coordinates, while surge
gain is Gaussian in an unbounded identity coordinate.

The implementation keeps the policy and value recurrences fully separate.
Complete episodes are the unit of storage and recurrent minibatching, so a GRU
state is never carried across an episode boundary.  Observation statistics are
also frozen while a rollout batch is collected; they are updated only after
that batch has been consumed, preserving exact PPO behavior probabilities.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from recurrent_sac import (
    ActionTransform,
    Episode,
    EpisodeBuilder,
    NumericalPolicyActionError,
    RunningMeanStd,
)


class RecurrentPPONumericalActionError(NumericalPolicyActionError):
    """A sampled PPO coordinate could not be executed in physical space.

    The sampled coordinate and its *behavior-policy* log probability are kept
    together so a runner can terminate the episode with a numerical penalty
    without clipping/substituting the action or losing the exact on-policy
    likelihood needed by PPO.
    """

    def __init__(
        self,
        coordinates: np.ndarray,
        message: str,
        *,
        log_probability: float,
        recurrent_state: "RecurrentPPOState",
        action_info: Mapping[str, Any],
    ) -> None:
        super().__init__(coordinates, message)
        self.log_probability = float(log_probability)
        self.recurrent_state = recurrent_state
        self.action_info = dict(action_info)


@dataclass
class RecurrentPPOConfig:
    hidden_dim: int = 128
    gamma: float = 0.97
    gae_lambda: float = 0.95
    clip_ratio: float = 0.20
    value_clip: float = 0.20
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.50
    actor_learning_rate: float = 3e-4
    value_learning_rate: float = 3e-4
    update_epochs: int = 8
    minibatch_episodes: int = 4
    rollout_transitions: int = 1_024
    reward_scale: float = 1.0
    gradient_clip: float = 1.0
    normalize_advantages: bool = True
    target_kl: Optional[float] = 0.03
    log_std_min: float = -5.0
    log_std_max: float = 1.0
    behavior_log_probability_tolerance: float = 1e-4
    device: str = "cpu"
    seed: int = 0
    observation_identity_indices: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must lie in (0, 1]")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must lie in [0, 1]")
        if not 0.0 < self.clip_ratio < 1.0:
            raise ValueError("clip_ratio must lie in (0, 1)")
        if self.value_clip < 0.0:
            raise ValueError("value_clip must be non-negative")
        if self.entropy_coefficient < 0.0 or self.value_coefficient < 0.0:
            raise ValueError("loss coefficients must be non-negative")
        if self.actor_learning_rate <= 0.0 or self.value_learning_rate <= 0.0:
            raise ValueError("learning rates must be positive")
        if self.update_epochs <= 0 or self.minibatch_episodes <= 0:
            raise ValueError("update epochs and minibatch episodes must be positive")
        if self.rollout_transitions <= 0 or self.gradient_clip <= 0.0:
            raise ValueError("rollout_transitions and gradient_clip must be positive")
        if self.reward_scale <= 0.0:
            raise ValueError("reward_scale must be positive")
        if self.target_kl is not None and self.target_kl <= 0.0:
            raise ValueError("target_kl must be positive when specified")
        self.observation_identity_indices = tuple(
            int(index) for index in self.observation_identity_indices
        )
        if len(self.observation_identity_indices) != len(
            set(self.observation_identity_indices)
        ):
            raise ValueError("observation_identity_indices must be unique")
        if any(index < 0 for index in self.observation_identity_indices):
            raise ValueError("observation_identity_indices must be non-negative")
        if not self.log_std_min < self.log_std_max:
            raise ValueError("log_std_min must be smaller than log_std_max")
        if self.behavior_log_probability_tolerance <= 0.0:
            raise ValueError("behavior log-probability tolerance must be positive")


@dataclass
class RecurrentPPOState:
    """Separate online recurrent states for the actor and value network."""

    actor: torch.Tensor
    value: torch.Tensor

    def detach(self) -> "RecurrentPPOState":
        return RecurrentPPOState(self.actor.detach(), self.value.detach())


@dataclass
class _OnPolicyEpisode:
    episode: Episode
    behavior_log_probabilities: Optional[np.ndarray] = None


@dataclass
class _PreparedEpisode:
    context: torch.Tensor
    coordinates: torch.Tensor
    old_log_probabilities: torch.Tensor
    old_values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor

    @property
    def length(self) -> int:
        return int(self.coordinates.shape[0])


class _RecurrentEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gru = nn.GRU(int(input_dim), int(hidden_dim), batch_first=True)
        self.norm = nn.LayerNorm(int(hidden_dim))

    def forward(
        self,
        inputs: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output, next_hidden = self.gru(inputs, hidden)
        return self.norm(output), next_hidden


class RecurrentGaussianPolicy(nn.Module):
    def __init__(
        self,
        context_dim: int,
        action_dim: int,
        hidden_dim: int,
        log_std_min: float,
        log_std_max: float,
    ) -> None:
        super().__init__()
        self.encoder = _RecurrentEncoder(context_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * action_dim),
        )
        self.action_dim = int(action_dim)
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        output = self.head[-1]
        assert isinstance(output, nn.Linear)
        nn.init.uniform_(output.weight, -1e-3, 1e-3)
        nn.init.zeros_(output.bias)
        with torch.no_grad():
            output.bias[self.action_dim :].fill_(-1.0)

    def distribution(
        self,
        context: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[Normal, torch.Tensor, torch.Tensor]:
        features, next_hidden = self.encoder(context, hidden)
        mean, log_std = self.head(features).chunk(2, dim=-1)
        log_std = log_std.clamp(self.log_std_min, self.log_std_max)
        return Normal(mean, log_std.exp()), next_hidden, log_std


class RecurrentValueNetwork(nn.Module):
    def __init__(self, context_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = _RecurrentEncoder(context_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        context: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        features, next_hidden = self.encoder(context, hidden)
        return self.head(features), next_hidden


class RecurrentPPOAgent:
    """On-policy recurrent PPO with independent actor and value GRUs."""

    def __init__(
        self,
        observation_dim: int,
        action_transform: Optional[ActionTransform] = None,
        config: Optional[RecurrentPPOConfig] = None,
    ) -> None:
        self.observation_dim = int(observation_dim)
        if self.observation_dim <= 0:
            raise ValueError("observation_dim must be positive")
        self.action_transform = action_transform or ActionTransform.market_controls()
        if any(kind not in {"log_positive", "identity"} for kind in self.action_transform.kinds):
            raise ValueError(
                "recurrent PPO supports only uncapped log_positive/identity coordinates"
            )
        self.action_dim = int(self.action_transform.dim)
        self.config = config or RecurrentPPOConfig()
        self.device = torch.device(self.config.device)
        torch.manual_seed(int(self.config.seed))
        self.rng = np.random.default_rng(int(self.config.seed))
        self.context_dim = self.observation_dim + self.action_dim + 2

        self.actor = RecurrentGaussianPolicy(
            self.context_dim,
            self.action_dim,
            self.config.hidden_dim,
            self.config.log_std_min,
            self.config.log_std_max,
        ).to(self.device)
        self.value = RecurrentValueNetwork(
            self.context_dim, self.config.hidden_dim
        ).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=float(self.config.actor_learning_rate)
        )
        self.value_optimizer = torch.optim.Adam(
            self.value.parameters(), lr=float(self.config.value_learning_rate)
        )
        self.observation_normalizer = RunningMeanStd(
            self.observation_dim,
            identity_indices=self.config.observation_identity_indices,
        )
        self._episodes: List[_OnPolicyEpisode] = []
        self._pending_transitions = 0
        self.updates = 0
        self.extreme_action_count = 0
        self.nonfinite_action_rejections = 0
        self.max_abs_coordinate = np.zeros(self.action_dim, dtype=np.float64)
        self.max_physical_action = np.full(self.action_dim, -np.inf, dtype=np.float64)
        self.min_physical_action = np.full(self.action_dim, np.inf, dtype=np.float64)
        self.behavior_log_probability_checks = 0
        self.max_behavior_log_probability_error = 0.0

    @property
    def pending_transitions(self) -> int:
        return int(self._pending_transitions)

    def initial_recurrent_state(self, batch_size: int = 1) -> RecurrentPPOState:
        shape = (1, int(batch_size), int(self.config.hidden_dim))
        return RecurrentPPOState(
            actor=torch.zeros(shape, dtype=torch.float32, device=self.device),
            value=torch.zeros(shape, dtype=torch.float32, device=self.device),
        )

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

    def _episode_context(self, episode: Episode) -> torch.Tensor:
        length = int(episode.length)
        observations = torch.as_tensor(
            episode.observations[None], dtype=torch.float32, device=self.device
        )
        previous_actions = np.zeros((length + 1, self.action_dim), dtype=np.float32)
        previous_rewards = np.zeros((length + 1, 1), dtype=np.float32)
        previous_dones = np.zeros((length + 1, 1), dtype=np.float32)
        previous_actions[1:] = episode.actions
        previous_rewards[1:, 0] = episode.rewards
        previous_dones[1:, 0] = episode.dones
        return self._context(
            observations,
            torch.as_tensor(previous_actions[None], device=self.device),
            torch.as_tensor(previous_rewards[None], device=self.device),
            torch.as_tensor(previous_dones[None], device=self.device),
        )

    @torch.no_grad()
    def act(
        self,
        observation: np.ndarray | Sequence[float],
        *,
        previous_action: Optional[np.ndarray | Sequence[float]] = None,
        previous_reward: float = 0.0,
        previous_done: bool = False,
        hidden: Optional[RecurrentPPOState] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, RecurrentPPOState, Dict[str, Any]]:
        observation_array = np.asarray(observation, dtype=np.float32).reshape(1, 1, -1)
        if observation_array.shape[-1] != self.observation_dim:
            raise ValueError(
                f"expected observation width {self.observation_dim}, "
                f"got {observation_array.shape[-1]}"
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
        actor_hidden = None if hidden is None else hidden.actor
        value_hidden = None if hidden is None else hidden.value
        distribution, next_actor_hidden, log_std = self.actor.distribution(
            context, actor_hidden
        )
        coordinates = distribution.mean if deterministic else distribution.sample()
        log_probability = distribution.log_prob(coordinates).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        value, next_value_hidden = self.value(context, value_hidden)
        coordinate_array = coordinates[0, 0].cpu().numpy().astype(np.float64)
        state = RecurrentPPOState(next_actor_hidden, next_value_hidden).detach()
        info = {
            "policy_coordinates": coordinate_array,
            "log_probability": float(log_probability.item()),
            "mean_coordinates": distribution.mean[0, 0].cpu().numpy().astype(np.float64),
            "log_std": log_std[0, 0].cpu().numpy().astype(np.float64),
            "entropy": float(entropy.item()),
            "value": float(value.item()),
            "extreme_action": bool(np.any(np.abs(coordinate_array) > 4.0)),
        }
        try:
            physical = self.action_transform.coordinates_to_physical(coordinate_array)
        except NumericalPolicyActionError as exc:
            self.nonfinite_action_rejections += 1
            raise RecurrentPPONumericalActionError(
                coordinate_array,
                str(exc),
                log_probability=float(log_probability.item()),
                recurrent_state=state,
                action_info=info,
            ) from exc
        absolute = np.abs(coordinate_array)
        self.max_abs_coordinate = np.maximum(self.max_abs_coordinate, absolute)
        self.max_physical_action = np.maximum(self.max_physical_action, physical)
        self.min_physical_action = np.minimum(self.min_physical_action, physical)
        extreme = bool(info["extreme_action"])
        self.extreme_action_count += int(extreme)
        return physical.astype(np.float64), state, info

    def add_episode(
        self,
        episode: Episode,
        actions_are_physical: bool = True,
        *,
        behavior_log_probabilities: Optional[Sequence[float] | np.ndarray] = None,
    ) -> None:
        """Queue one on-policy episode without crossing or flattening boundaries.

        Observation normalization intentionally remains frozen until ``update``
        consumes the whole batch.  Optional log probabilities can be copied
        directly from ``act`` diagnostics; otherwise they are reconstructed
        exactly before the first optimizer step.
        """
        actions = episode.actions
        if actions_are_physical:
            actions = self.action_transform.physical_to_coordinates(actions).astype(np.float32)
        converted = Episode(
            observations=episode.observations,
            actions=actions,
            rewards=episode.rewards,
            dones=episode.dones,
        )
        if converted.actions.shape[1] != self.action_dim:
            raise ValueError(
                f"expected action width {self.action_dim}, got {converted.actions.shape[1]}"
            )
        if np.any(converted.dones[:-1] >= 0.5) or converted.dones[-1] < 0.5:
            raise ValueError(
                "on-policy recurrent PPO requires one complete episode with only its "
                "final transition terminal"
            )
        behavior = None
        if behavior_log_probabilities is not None:
            behavior = np.asarray(behavior_log_probabilities, dtype=np.float32).reshape(-1)
            if behavior.shape[0] != converted.length:
                raise ValueError("behavior log probabilities must align with episode actions")
            if not np.all(np.isfinite(behavior)):
                raise FloatingPointError("behavior log probabilities contain non-finite values")
            behavior = behavior.copy()
        self._episodes.append(_OnPolicyEpisode(converted, behavior))
        self._pending_transitions += converted.length

    def can_update(self, minimum_transitions: Optional[int] = None) -> bool:
        threshold = int(
            self.config.rollout_transitions
            if minimum_transitions is None
            else minimum_transitions
        )
        return bool(self._episodes and self._pending_transitions >= max(1, threshold))

    def reset_rollout(self) -> None:
        self._episodes.clear()
        self._pending_transitions = 0

    @staticmethod
    def compute_gae(
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        *,
        gamma: float,
        gae_lambda: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute terminal-aware GAE for one sequence.

        ``values`` contains T+1 states while rewards/dones contain T
        transitions.  A terminal at any position zeros both bootstrap and the
        recursive trace, preventing credit from flowing across a boundary.
        """
        rewards = rewards.reshape(-1)
        values = values.reshape(-1)
        dones = dones.reshape(-1)
        if values.numel() != rewards.numel() + 1 or dones.numel() != rewards.numel():
            raise ValueError("GAE requires T rewards/dones and T+1 values")
        advantages = torch.zeros_like(rewards)
        trace = torch.zeros((), dtype=rewards.dtype, device=rewards.device)
        for index in range(rewards.numel() - 1, -1, -1):
            continuation = 1.0 - dones[index]
            delta = rewards[index] + float(gamma) * continuation * values[index + 1] - values[index]
            trace = delta + float(gamma) * float(gae_lambda) * continuation * trace
            advantages[index] = trace
        return advantages, advantages + values[:-1]

    @staticmethod
    def clipped_policy_loss(
        new_log_probabilities: torch.Tensor,
        old_log_probabilities: torch.Tensor,
        advantages: torch.Tensor,
        clip_ratio: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_ratio = new_log_probabilities - old_log_probabilities
        ratio = torch.exp(log_ratio)
        if not torch.all(torch.isfinite(ratio)):
            raise FloatingPointError("non-finite PPO importance ratio")
        clipped = ratio.clamp(1.0 - float(clip_ratio), 1.0 + float(clip_ratio))
        objective = torch.minimum(ratio * advantages, clipped * advantages)
        clip_fraction = (torch.abs(ratio - 1.0) > float(clip_ratio)).float().mean()
        return -objective.mean(), ratio, clip_fraction

    @torch.no_grad()
    def _prepare_episodes(self) -> List[_PreparedEpisode]:
        prepared: List[_PreparedEpisode] = []
        for stored in self._episodes:
            episode = stored.episode
            context = self._episode_context(episode)
            distribution, _, _ = self.actor.distribution(context[:, :-1])
            coordinates = torch.as_tensor(
                episode.actions, dtype=torch.float32, device=self.device
            )
            reconstructed_log_probabilities = distribution.log_prob(
                coordinates.unsqueeze(0)
            ).sum(dim=-1)[0]
            if stored.behavior_log_probabilities is None:
                old_log_probabilities = reconstructed_log_probabilities
            else:
                old_log_probabilities = torch.as_tensor(
                    stored.behavior_log_probabilities,
                    dtype=torch.float32,
                    device=self.device,
                )
                error = float(
                    torch.max(
                        torch.abs(
                            old_log_probabilities - reconstructed_log_probabilities
                        )
                    ).item()
                )
                if error > float(self.config.behavior_log_probability_tolerance):
                    raise RuntimeError(
                        "stored behavior log probabilities do not reconstruct "
                        f"under the frozen rollout policy (max error={error:.6g})"
                    )
                self.behavior_log_probability_checks += int(episode.length)
                self.max_behavior_log_probability_error = max(
                    self.max_behavior_log_probability_error, error
                )
            all_values, _ = self.value(context)
            all_values = all_values[0, :, 0]
            rewards = torch.as_tensor(
                episode.rewards * float(self.config.reward_scale),
                dtype=torch.float32,
                device=self.device,
            )
            dones = torch.as_tensor(
                episode.dones, dtype=torch.float32, device=self.device
            )
            advantages, returns = self.compute_gae(
                rewards,
                all_values,
                dones,
                gamma=float(self.config.gamma),
                gae_lambda=float(self.config.gae_lambda),
            )
            prepared.append(
                _PreparedEpisode(
                    context=context.detach(),
                    coordinates=coordinates.detach(),
                    old_log_probabilities=old_log_probabilities.detach(),
                    old_values=all_values[:-1].detach(),
                    advantages=advantages.detach(),
                    returns=returns.detach(),
                )
            )
        if self.config.normalize_advantages:
            flat = torch.cat([row.advantages for row in prepared])
            mean = flat.mean()
            std = flat.std(unbiased=False).clamp_min(1e-8)
            for row in prepared:
                row.advantages = (row.advantages - mean) / std
        return prepared

    def update(self) -> Dict[str, float]:
        if not self._episodes:
            raise RuntimeError("recurrent PPO has no on-policy episodes to update from")
        prepared = self._prepare_episodes()
        episode_order = np.arange(len(prepared))
        metrics: Dict[str, List[float]] = {
            name: []
            for name in (
                "actor_loss",
                "value_loss",
                "entropy",
                "approx_kl",
                "clip_fraction",
                "actor_grad_norm",
                "value_grad_norm",
            )
        }
        epochs_completed = 0
        stopped_early = False
        for _ in range(int(self.config.update_epochs)):
            self.rng.shuffle(episode_order)
            epoch_kls: List[float] = []
            for start in range(0, len(episode_order), int(self.config.minibatch_episodes)):
                indices = episode_order[start : start + int(self.config.minibatch_episodes)]
                actor_log_probabilities: List[torch.Tensor] = []
                actor_entropies: List[torch.Tensor] = []
                old_log_probabilities: List[torch.Tensor] = []
                advantages: List[torch.Tensor] = []
                new_values: List[torch.Tensor] = []
                old_values: List[torch.Tensor] = []
                returns: List[torch.Tensor] = []
                for index in indices:
                    row = prepared[int(index)]
                    distribution, _, _ = self.actor.distribution(row.context[:, :-1])
                    actor_log_probabilities.append(
                        distribution.log_prob(row.coordinates.unsqueeze(0)).sum(dim=-1)[0]
                    )
                    actor_entropies.append(distribution.entropy().sum(dim=-1)[0])
                    old_log_probabilities.append(row.old_log_probabilities)
                    advantages.append(row.advantages)
                    predicted, _ = self.value(row.context[:, :-1])
                    new_values.append(predicted[0, :, 0])
                    old_values.append(row.old_values)
                    returns.append(row.returns)

                new_logp = torch.cat(actor_log_probabilities)
                old_logp = torch.cat(old_log_probabilities)
                advantage = torch.cat(advantages)
                entropy = torch.cat(actor_entropies).mean()
                policy_loss, _, clip_fraction = self.clipped_policy_loss(
                    new_logp,
                    old_logp,
                    advantage,
                    float(self.config.clip_ratio),
                )
                actor_loss = policy_loss - float(self.config.entropy_coefficient) * entropy
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                actor_grad = nn.utils.clip_grad_norm_(
                    self.actor.parameters(), float(self.config.gradient_clip)
                )
                self.actor_optimizer.step()

                predicted_values = torch.cat(new_values)
                previous_values = torch.cat(old_values)
                target_returns = torch.cat(returns)
                if self.config.value_clip > 0.0:
                    clipped_values = previous_values + (
                        predicted_values - previous_values
                    ).clamp(-float(self.config.value_clip), float(self.config.value_clip))
                    squared = torch.maximum(
                        (predicted_values - target_returns).square(),
                        (clipped_values - target_returns).square(),
                    )
                else:
                    squared = (predicted_values - target_returns).square()
                value_loss = 0.5 * squared.mean() * float(self.config.value_coefficient)
                self.value_optimizer.zero_grad(set_to_none=True)
                value_loss.backward()
                value_grad = nn.utils.clip_grad_norm_(
                    self.value.parameters(), float(self.config.gradient_clip)
                )
                self.value_optimizer.step()

                with torch.no_grad():
                    approximate_kl = (old_logp - new_logp).mean()
                values = {
                    "actor_loss": actor_loss,
                    "value_loss": value_loss,
                    "entropy": entropy,
                    "approx_kl": approximate_kl,
                    "clip_fraction": clip_fraction,
                    "actor_grad_norm": actor_grad,
                    "value_grad_norm": value_grad,
                }
                for name, value in values.items():
                    scalar = float(value.detach().item())
                    if not np.isfinite(scalar):
                        raise FloatingPointError(f"non-finite recurrent PPO metric {name}")
                    metrics[name].append(scalar)
                epoch_kls.append(float(approximate_kl.item()))
            epochs_completed += 1
            if (
                self.config.target_kl is not None
                and epoch_kls
                and float(np.mean(epoch_kls)) > float(self.config.target_kl)
            ):
                stopped_early = True
                break

        consumed_episodes = len(self._episodes)
        consumed_transitions = int(self._pending_transitions)
        observations = np.concatenate(
            [stored.episode.observations[:-1] for stored in self._episodes], axis=0
        )
        self.observation_normalizer.update(observations)
        self.reset_rollout()
        self.updates += 1
        result = {name: float(np.mean(rows)) for name, rows in metrics.items()}
        result.update(
            {
                "epochs_completed": float(epochs_completed),
                "stopped_early": float(stopped_early),
                "episodes": float(consumed_episodes),
                "transitions": float(consumed_transitions),
                "updates": float(self.updates),
            }
        )
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
            "behavior_log_probability_checks": int(
                self.behavior_log_probability_checks
            ),
            "max_behavior_log_probability_error": float(
                self.max_behavior_log_probability_error
            ),
        }

    def save(self, path: str | Path, metadata: Optional[Mapping[str, Any]] = None) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "ride_response_recurrent_ppo_v1",
            "observation_dim": self.observation_dim,
            "action_transform": asdict(self.action_transform),
            "config": asdict(self.config),
            "actor": self.actor.state_dict(),
            "value": self.value.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
            "observation_normalizer": self.observation_normalizer.state_dict(),
            "updates": self.updates,
            "action_diagnostics": self.action_diagnostics(),
            "metadata": dict(metadata or {}),
        }
        torch.save(payload, destination)

    @classmethod
    def load(cls, path: str | Path, device: Optional[str] = None) -> "RecurrentPPOAgent":
        source = Path(path)
        try:
            payload = torch.load(source, map_location=device or "cpu", weights_only=False)
        except TypeError:
            payload = torch.load(source, map_location=device or "cpu")
        if payload.get("format") != "ride_response_recurrent_ppo_v1":
            raise ValueError("unsupported recurrent PPO artifact")
        config_values = dict(payload["config"])
        if device is not None:
            config_values["device"] = str(device)
        agent = cls(
            observation_dim=int(payload["observation_dim"]),
            action_transform=ActionTransform(**payload["action_transform"]),
            config=RecurrentPPOConfig(**config_values),
        )
        agent.actor.load_state_dict(payload["actor"])
        agent.value.load_state_dict(payload["value"])
        agent.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        agent.value_optimizer.load_state_dict(payload["value_optimizer"])
        agent.observation_normalizer.load_state_dict(payload["observation_normalizer"])
        agent.updates = int(payload.get("updates", 0))
        diagnostics = dict(payload.get("action_diagnostics", {}) or {})
        agent.extreme_action_count = int(diagnostics.get("extreme_action_count", 0))
        agent.nonfinite_action_rejections = int(
            diagnostics.get("nonfinite_action_rejections", 0)
        )
        agent.behavior_log_probability_checks = int(
            diagnostics.get("behavior_log_probability_checks", 0)
        )
        agent.max_behavior_log_probability_error = float(
            diagnostics.get("max_behavior_log_probability_error", 0.0)
        )
        for attribute, key in (
            ("max_abs_coordinate", "max_abs_policy_coordinate"),
            ("min_physical_action", "min_physical_action"),
            ("max_physical_action", "max_physical_action"),
        ):
            if key in diagnostics:
                setattr(agent, attribute, np.asarray(diagnostics[key], dtype=np.float64))
        return agent


__all__ = [
    "ActionTransform",
    "Episode",
    "EpisodeBuilder",
    "RecurrentGaussianPolicy",
    "RecurrentPPOAgent",
    "RecurrentPPOConfig",
    "RecurrentPPONumericalActionError",
    "RecurrentPPOState",
    "RecurrentValueNetwork",
]
