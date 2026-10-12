from __future__ import annotations

"""Sequence-aware categorical PPO with fully separate actor and critic."""

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence, Tuple

import mkl_config  # noqa: F401 - configure oneMKL before NumPy/Torch imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ActorCritic(nn.Module):
    """Independent recurrent encoders for policy and state value.

    The v23 default keeps two public-state recurrent paths. Optional v24 modules
    add critic-only privileged-state fusion and discrete action values; actor
    parameters and inputs remain isolated.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden: int = 192,
        constraint_dim: int = 0,
        response_dim: int = 0,
        opponent_forecast_dim: int = 0,
        action_feature_dim: int = 0,
        enable_action_q: bool = False,
        critic_state_dim: Optional[int] = None,
        privileged_critic: bool = False,
        privileged_value_only: bool = False,
        persistent_recurrence: bool = False,
        single_state_dim: Optional[int] = None,
        frame_stack: int = 1,
        feature_groups: Optional[Dict[str, Sequence[int]]] = None,
        normalize_values: bool = False,
    ) -> None:
        super().__init__()
        del feature_groups

        self.state_dim = int(max(1, state_dim))
        self.action_dim = int(max(1, action_dim))
        self.constraint_dim = int(max(0, constraint_dim))
        self.response_dim = int(max(0, response_dim))
        self.opponent_forecast_dim = int(
            np.clip(opponent_forecast_dim, 0, self.response_dim)
        )
        self.action_feature_dim = int(max(0, action_feature_dim))
        self.enable_action_q = bool(enable_action_q)
        self.privileged_critic = bool(privileged_critic)
        self.privileged_value_only = bool(
            privileged_value_only and privileged_critic
        )
        self.persistent_recurrence = bool(persistent_recurrence)
        self.critic_state_dim = int(
            max(
                1,
                self.state_dim
                if critic_state_dim is None
                else critic_state_dim,
            )
        )
        self.frame_stack = int(max(1, frame_stack))
        self.single_state_dim = int(
            max(1, self.state_dim if single_state_dim is None else single_state_dim)
        )
        if self.single_state_dim * self.frame_stack != self.state_dim:
            raise ValueError(
                "state_dim must equal single_state_dim * frame_stack: "
                f"{self.state_dim} != {self.single_state_dim} * {self.frame_stack}"
            )

        self.actor_frame_encoder = nn.Sequential(
            nn.Linear(self.single_state_dim, hidden),
            nn.Tanh(),
        )
        self.actor_recurrent = nn.GRU(hidden, hidden, batch_first=True)
        self.actor_encoder = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.critic_frame_encoder = nn.Sequential(
            nn.Linear(self.single_state_dim, hidden),
            nn.Tanh(),
        )
        self.critic_recurrent = nn.GRU(hidden, hidden, batch_first=True)
        self.critic_encoder = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        if self.privileged_critic:
            self.critic_privileged_encoder: Optional[nn.Module] = nn.Sequential(
                nn.Linear(self.critic_state_dim, hidden),
                nn.Tanh(),
                *(
                    (nn.Linear(hidden, hidden), nn.Tanh())
                    if self.privileged_value_only
                    else ()
                ),
            )
            self.critic_fusion: Optional[nn.Module] = (
                None
                if self.privileged_value_only
                else nn.Sequential(
                    nn.Linear(hidden * 2, hidden),
                    nn.Tanh(),
                )
            )
        else:
            # ``None`` keeps the disabled network schema identical to v23.
            self.critic_privileged_encoder = None
            self.critic_fusion = None
        self.pi_head = nn.Linear(hidden, self.action_dim)
        self.v_head = nn.Linear(hidden, 1)
        self.normalize_values = bool(normalize_values)
        if self.normalize_values:
            self.register_buffer("value_scale", torch.ones(()))
            self.register_buffer("value_square_sum", torch.zeros((), dtype=torch.float64))
            self.register_buffer("value_count", torch.zeros((), dtype=torch.float64))

        # Explicit compatibility markers. These are not modules or parameters.
        self.mag_mean_head = None
        if self.enable_action_q:
            q_state_dim = (
                self.critic_state_dim
                if self.privileged_critic
                else self.state_dim
            )
            self.q_state_encoder: Optional[nn.Module] = nn.Sequential(
                nn.Linear(q_state_dim, hidden),
                nn.Tanh(),
                nn.Linear(hidden, hidden),
                nn.Tanh(),
            )
            self.q_action_encoder: Optional[nn.Module] = nn.Sequential(
                nn.Linear(
                    self.action_feature_dim
                    if self.action_feature_dim > 0
                    else self.action_dim,
                    hidden,
                ),
                nn.Tanh(),
            )
            self.action_q_head: Optional[nn.Module] = nn.Linear(
                hidden * 2, 1
            )
        else:
            self.q_state_encoder = None
            self.q_action_encoder = None
            self.action_q_head = None
        self.constraint_head = None
        self.risk_head = None
        self.response_head = None
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2.0))
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.GRU):
                for name, parameter in module.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(parameter)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(parameter)
                    elif "bias" in name:
                        nn.init.constant_(parameter, 0.0)
        # Start the categorical actor from an exactly neutral prior.  A tiny
        # random policy head makes deterministic checkpoint validation select
        # an arbitrary tariff before PPO has learned anything; with absolute
        # tariff targets, that lucky initial argmax can look like successful
        # learning for the entire run.  Zero logits retain a uniform stochastic
        # behavior policy (so PPO can break symmetry from reward evidence) and
        # make the deterministic pre-learning tie resolve to legal action 0
        # (hold).  No action is removed, masked, or forced after initialization.
        nn.init.zeros_(self.pi_head.weight)
        nn.init.zeros_(self.pi_head.bias)
        nn.init.orthogonal_(self.v_head.weight, gain=1.0)
        if self.action_q_head is not None:
            nn.init.orthogonal_(self.action_q_head.weight, gain=0.5)

    def actor_parameters(self) -> List[nn.Parameter]:
        return [
            *self.actor_frame_encoder.parameters(),
            *self.actor_recurrent.parameters(),
            *self.actor_encoder.parameters(),
            *self.pi_head.parameters(),
        ]

    def critic_parameters(self) -> List[nn.Parameter]:
        if self.privileged_value_only:
            assert self.critic_privileged_encoder is not None
            return [
                *self.critic_privileged_encoder.parameters(),
                *self.v_head.parameters(),
            ]
        parameters = [
            *self.critic_frame_encoder.parameters(),
            *self.critic_recurrent.parameters(),
            *self.critic_encoder.parameters(),
            *self.v_head.parameters(),
        ]
        if self.critic_privileged_encoder is not None:
            parameters.extend(self.critic_privileged_encoder.parameters())
        if self.critic_fusion is not None:
            parameters.extend(self.critic_fusion.parameters())
        return parameters

    def q_parameters(self) -> List[nn.Parameter]:
        if self.action_q_head is None:
            return []
        assert self.q_state_encoder is not None
        assert self.q_action_encoder is not None
        return [
            *self.q_state_encoder.parameters(),
            *self.q_action_encoder.parameters(),
            *self.action_q_head.parameters(),
        ]

    def response_parameters(self) -> List[nn.Parameter]:
        return []

    def _public_sequence(self, state: torch.Tensor) -> torch.Tensor:
        state = torch.nan_to_num(state, nan=0.0, posinf=1e3, neginf=-1e3)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if state.shape[-1] != self.state_dim:
            raise ValueError(
                f"State dim mismatch: got {state.shape[-1]}, expected {self.state_dim}"
            )
        return state.reshape(
            state.shape[0], self.frame_stack, self.single_state_dim
        )

    def actor_forward(
        self,
        state: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence = self._public_sequence(state)
        if self.persistent_recurrence and hidden is not None:
            # Frame stacks overlap heavily across pricing decisions. Once a
            # persistent hidden state exists, consume only the newest frame so
            # the same historical frames are not replayed into the GRU.
            sequence = sequence[:, -1:, :]
        frames = self.actor_frame_encoder(sequence)
        _, next_hidden = self.actor_recurrent(frames, hidden)
        latent = self.actor_encoder(next_hidden[-1])
        return self.pi_head(latent), next_hidden

    def critic_forward(
        self,
        state: torch.Tensor,
        critic_state: Optional[torch.Tensor] = None,
        hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sequence = self._public_sequence(state)
        batch = sequence.shape[0]
        if self.persistent_recurrence and hidden is not None:
            sequence = sequence[:, -1:, :]
        frames = self.critic_frame_encoder(sequence)
        _, next_hidden = self.critic_recurrent(frames, hidden)
        latent = self.critic_encoder(next_hidden[-1])
        if self.privileged_critic:
            if critic_state is None:
                raise ValueError(
                    "privileged critic requires a training-only critic_state"
                )
            critic_state = torch.nan_to_num(
                critic_state, nan=0.0, posinf=1e3, neginf=-1e3
            )
            if critic_state.ndim == 1:
                critic_state = critic_state.unsqueeze(0)
            if critic_state.shape != (batch, self.critic_state_dim):
                raise ValueError(
                    "Critic state shape mismatch: got "
                    f"{tuple(critic_state.shape)}, expected "
                    f"({batch}, {self.critic_state_dim})"
                )
            assert self.critic_privileged_encoder is not None
            privileged = self.critic_privileged_encoder(critic_state)
            if self.privileged_value_only:
                latent = privileged
            else:
                assert self.critic_fusion is not None
                latent = self.critic_fusion(
                    torch.cat((latent, privileged), dim=-1)
                )
        value = self.v_head(latent).squeeze(-1)
        if self.normalize_values:
            value = value * self.value_scale
        action_values = state.new_empty((batch, 0))
        return value, action_values, next_hidden

    @torch.no_grad()
    def update_value_scale(self, targets: torch.Tensor) -> float:
        """Running return-RMS coordinates; preserve every raw value prediction.

        Scaling both the head and its output avoids pushing bounded recurrent
        features toward raw economic return levels. No reward/value is clipped.
        Statistics are checkpoint buffers and update once per rollout, not epoch.
        """
        if not self.normalize_values:
            return 1.0
        targets = targets.detach().double()
        if not bool(torch.isfinite(targets).all()):
            raise ValueError("value normalization requires finite targets")
        self.value_square_sum.add_(targets.square().sum())
        self.value_count.add_(targets.numel())
        scale = (self.value_square_sum / self.value_count).sqrt().float().clamp_min(1e-6)
        ratio = self.value_scale / scale
        self.v_head.weight.mul_(ratio)
        self.v_head.bias.mul_(ratio)
        self.value_scale.copy_(scale)
        return float(ratio)

    def q_forward(
        self,
        state: torch.Tensor,
        *,
        action_features: Optional[torch.Tensor] = None,
        critic_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Independent action-value model; gradients never touch V or policy."""
        if self.action_q_head is None:
            batch = 1 if state.ndim == 1 else state.shape[0]
            return state.new_empty((batch, 0))
        assert self.q_state_encoder is not None
        assert self.q_action_encoder is not None
        if state.ndim == 1:
            state = state.unsqueeze(0)
        q_state = critic_state if self.privileged_critic else state
        if q_state is None:
            raise ValueError("privileged action Q requires critic_state")
        if q_state.ndim == 1:
            q_state = q_state.unsqueeze(0)
        q_state = torch.nan_to_num(
            q_state, nan=0.0, posinf=1e3, neginf=-1e3
        )
        batch = q_state.shape[0]
        state_latent = self.q_state_encoder(q_state)
        if self.action_feature_dim > 0:
            if action_features is None:
                raise ValueError("action Q requires action features")
            if action_features.ndim == 2:
                action_features = action_features.unsqueeze(0)
            expected = (batch, self.action_dim, self.action_feature_dim)
            if tuple(action_features.shape) != expected:
                raise ValueError(
                    f"action feature shape {tuple(action_features.shape)} != {expected}"
                )
            action_inputs = torch.nan_to_num(
                action_features, nan=0.0, posinf=2.0, neginf=-2.0
            )
        else:
            action_inputs = torch.eye(
                self.action_dim, dtype=state.dtype, device=state.device
            ).unsqueeze(0).expand(batch, -1, -1)
        action_latent = self.q_action_encoder(action_inputs)
        state_latent = state_latent.unsqueeze(1).expand(-1, self.action_dim, -1)
        return self.action_q_head(
            torch.cat((state_latent, action_latent), dim=-1)
        ).squeeze(-1)

    def forward(
        self,
        state: torch.Tensor,
        action_features: Optional[torch.Tensor] = None,
        critic_state: Optional[torch.Tensor] = None,
        actor_hidden: Optional[torch.Tensor] = None,
        critic_hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        logits, _ = self.actor_forward(state, actor_hidden)
        value, _, _ = self.critic_forward(
            state, critic_state=critic_state, hidden=critic_hidden
        )
        action_values = self.q_forward(
            state,
            action_features=action_features,
            critic_state=critic_state,
        )
        batch = logits.shape[0]
        zeros_like_logits = torch.zeros_like(logits)
        constraint_values = state.new_zeros((batch, self.constraint_dim))
        risk_values = state.new_zeros((batch,))
        response = state.new_zeros((batch, self.response_dim))
        return (
            logits,
            zeros_like_logits,
            zeros_like_logits,
            value,
            constraint_values,
            risk_values,
            response,
            action_values,
        )


@dataclass
class Transition:
    state: torch.Tensor
    critic_state: torch.Tensor
    action: torch.Tensor
    raw_reward: float
    reward: float
    done: bool
    truncated: bool
    discount: float
    duration: int
    old_logp: torch.Tensor
    old_value: torch.Tensor
    exploration_rate: float
    reward_regime: str
    action_features: torch.Tensor
    action_mask: torch.Tensor
    next_value_override: torch.Tensor
    actor_hidden: torch.Tensor
    critic_hidden: torch.Tensor
    counterfactual_values: torch.Tensor
    counterfactual_mask: torch.Tensor
    potential_delta: float
    potential: float
    next_potential_override: float
    next_state_finalized: bool


class PPOAgent:
    """Vanilla clipped PPO for the discrete pricing action space.

    V23 optimizes only the categorical policy and scalar reward value. Explicit
    v24 gates can add critic-only counterfactual Q supervision, persistent
    recurrence, a learned reward rate, KL stopping, and potential shaping.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.20,
        final_clip_eps: float = 0.10,
        v_coeff: float = 0.5,
        ent_coeff: float = 0.01,
        min_ent_coeff: float = 0.001,
        ent_decay: float = 0.995,
        max_ent_coeff: float = 0.08,
        target_entropy_fraction: float = 0.45,
        entropy_adaptation_rate: float = 0.20,
        max_grad_norm: float = 1.0,
        hidden_dim: int = 192,
        target_kl: float = 0.03,
        adv_clip: float = 4.0,
        lr_growth_on_stall: float = 1.08,
        min_lr: float = 5e-5,
        max_lr: Optional[float] = None,
        value_clip_eps: float = 0.20,
        initial_exploration_rate: float = 0.20,
        final_exploration_rate: float = 0.05,
        exploration_fraction: float = 1.00,
        exploration_warmup_fraction: float = 0.10,
        min_action_visits: int = 8,
        constraint_dim: int = 0,
        response_dim: int = 0,
        opponent_forecast_dim: int = 0,
        action_feature_dim: int = 0,
        action_q_coeff: float = 0.0,
        action_q_cql_coeff: float = 0.05,
        objective_optimizer_weight: float = 1.25,
        objective_optimizer_min_updates: int = 3,
        constraint_value_coeff: float = 0.25,
        risk_value_coeff: float = 0.15,
        response_coeff: float = 0.05,
        risk_coeff: float = 0.0,
        delayed_reward_horizon: int = 6,
        delayed_reward_blend: float = 0.0,
        single_state_dim: Optional[int] = None,
        frame_stack: int = 1,
        state_action_mi_coeff: float = 0.0,
        direct_actor_credit_weight: float = 0.0,
        feature_groups: Optional[Dict[str, Sequence[int]]] = None,
        hard_kl_stop: bool = False,
        reward_rate_beta: float = 0.0,
        center_rewards: bool = True,
        critic_state_dim: Optional[int] = None,
        privileged_critic: bool = False,
        privileged_value_only: bool = False,
        persistent_recurrence: bool = False,
        enable_counterfactual_q: bool = False,
        counterfactual_actor_weight: float = 0.0,
        potential_shaping_scale: float = 0.0,
        td_value_targets: bool = False,
        normalize_values: bool = False,
        scheduled_exploration: bool = False,
        initial_hold_probability: Optional[float] = None,
        device: Optional[str] = None,
    ) -> None:
        del (
            final_clip_eps,
            ent_decay,
            lr_growth_on_stall,
            max_lr,
            value_clip_eps,
            min_action_visits,
            action_q_cql_coeff,
            objective_optimizer_weight,
            objective_optimizer_min_updates,
            constraint_value_coeff,
            risk_value_coeff,
            response_coeff,
            risk_coeff,
            delayed_reward_horizon,
            delayed_reward_blend,
            state_action_mi_coeff,
            direct_actor_credit_weight,
        )
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.state_dim = int(max(1, state_dim))
        self.action_dim = int(max(1, action_dim))
        self.constraint_dim = int(max(0, constraint_dim))
        self.response_dim = int(max(0, response_dim))
        self.opponent_forecast_dim = int(
            np.clip(opponent_forecast_dim, 0, self.response_dim)
        )
        self.action_feature_dim = int(max(0, action_feature_dim))
        self.privileged_critic = bool(privileged_critic)
        self.privileged_value_only = bool(
            privileged_value_only and privileged_critic
        )
        self.persistent_recurrence = bool(persistent_recurrence)
        self.critic_state_dim = int(
            max(
                1,
                self.state_dim
                if critic_state_dim is None
                else critic_state_dim,
            )
        )
        self.enable_counterfactual_q = bool(enable_counterfactual_q)
        self.action_q_coeff = (
            float(max(0.0, action_q_coeff))
            if self.enable_counterfactual_q
            else 0.0
        )
        self.counterfactual_actor_weight = float(
            np.clip(counterfactual_actor_weight, 0.0, 1.0)
        ) if self.enable_counterfactual_q else 0.0
        self.potential_shaping_scale = float(max(0.0, potential_shaping_scale))
        self.td_value_targets = bool(td_value_targets)
        enable_action_q = bool(
            self.action_q_coeff > 0.0
            or self.counterfactual_actor_weight > 0.0
        )

        self.net = ActorCritic(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden=hidden_dim,
            constraint_dim=self.constraint_dim,
            response_dim=self.response_dim,
            opponent_forecast_dim=self.opponent_forecast_dim,
            action_feature_dim=self.action_feature_dim,
            enable_action_q=enable_action_q,
            critic_state_dim=self.critic_state_dim,
            privileged_critic=self.privileged_critic,
            privileged_value_only=self.privileged_value_only,
            persistent_recurrence=self.persistent_recurrence,
            single_state_dim=single_state_dim,
            frame_stack=frame_stack,
            feature_groups=feature_groups,
            normalize_values=normalize_values,
        ).to(self.device)
        if initial_hold_probability is None:
            self.initial_hold_probability = float(1.0 / self.action_dim)
        else:
            hold_probability = float(initial_hold_probability)
            if self.action_dim < 2 or not 0.0 < hold_probability < 1.0:
                raise ValueError(
                    "initial_hold_probability must be in (0, 1) with at "
                    "least two actions"
                )
            move_probability = float(
                (1.0 - hold_probability) / (self.action_dim - 1)
            )
            with torch.no_grad():
                self.net.pi_head.bias.fill_(math.log(move_probability))
                self.net.pi_head.bias[0] = math.log(hold_probability)
            self.initial_hold_probability = hold_probability

        self.base_lr = float(lr)
        self.curr_lr = float(lr)
        # Long-horizon value targets move much farther than clipped policy
        # logits. A fixed, separate critic rate lets the baseline catch the
        # return level without changing the actor's PPO trust-region schedule.
        self.critic_lr_multiplier = 3.0
        self.min_lr = float(max(1e-6, min_lr))
        self.actor_opt = optim.Adam(
            self.net.actor_parameters(), lr=self.curr_lr, eps=1e-5
        )
        self.critic_opt = optim.Adam(
            self.net.critic_parameters(),
            lr=self.curr_lr * self.critic_lr_multiplier,
            eps=1e-5,
        )
        self.response_opt = None
        q_parameters = self.net.q_parameters()
        self.q_opt = (
            optim.Adam(
                q_parameters,
                lr=self.curr_lr * self.critic_lr_multiplier,
                eps=1e-5,
            )
            if q_parameters
            else None
        )

        self.gamma = float(gamma)
        self.lam = float(lam)
        self.initial_clip_eps = float(max(0.01, clip_eps))
        self.clip_eps = self.initial_clip_eps
        self.v_coeff = float(max(0.0, v_coeff))
        self.ent_coeff = float(max(0.0, ent_coeff))
        self.base_ent_coeff = self.ent_coeff
        self.max_ent_coeff = float(max(self.ent_coeff, max_ent_coeff))
        self.min_ent_coeff = float(min(self.ent_coeff, max(0.0, min_ent_coeff)))
        self.target_entropy_fraction = float(
            np.clip(target_entropy_fraction, 0.05, 0.95)
        )
        self.entropy_adaptation_rate = float(
            np.clip(entropy_adaptation_rate, 0.0, 1.0)
        )
        self.max_grad_norm = float(max(1e-6, max_grad_norm))
        self.target_kl = float(max(1e-4, target_kl))
        self.adv_clip = float(max(1.0, adv_clip))
        self.hard_kl_stop = bool(hard_kl_stop)
        self.reward_rate_beta = float(np.clip(reward_rate_beta, 0.0, 1.0))
        self.center_rewards = bool(center_rewards)
        self.average_reward_rate = 0.0
        self.average_reward_initialized = False
        self.reward_rate_by_regime: Dict[str, float] = {}

        # The behavior policy is a scheduled mixture of the learned actor and
        # a uniform distribution over feasible actions.  Crucially, the same
        # mixture probability is retained in every transition and recomputed
        # by PPO, so this is on-policy coverage rather than an epsilon-greedy
        # action override with an incorrect likelihood ratio.  Hold remains in
        # the feasible set and is never suppressed or forced.
        self.scheduled_exploration = bool(scheduled_exploration)
        self.initial_exploration_rate = float(
            np.clip(initial_exploration_rate, 0.0, 1.0)
            if self.scheduled_exploration
            else 0.0
        )
        self.final_exploration_rate = float(
            np.clip(final_exploration_rate, 0.0, 1.0)
            if self.scheduled_exploration
            else 0.0
        )
        if self.final_exploration_rate > self.initial_exploration_rate:
            raise ValueError(
                "final_exploration_rate cannot exceed initial_exploration_rate"
            )
        self.exploration_warmup_fraction = float(
            np.clip(exploration_warmup_fraction, 0.0, 1.0)
            if self.scheduled_exploration
            else 0.0
        )
        self.exploration_fraction = float(
            np.clip(exploration_fraction, 0.0, 1.0)
            if self.scheduled_exploration
            else 0.0
        )
        if self.exploration_fraction < self.exploration_warmup_fraction:
            raise ValueError(
                "exploration_fraction must be at least the warmup fraction"
            )
        self.exploration_rate = self.initial_exploration_rate
        self.last_action_exploration_rate = 0.0
        # The Q head is critic-only and is inactive unless its explicit v24
        # supervision gate is non-zero.
        self.action_q_cql_coeff = 0.0
        self.objective_optimizer_weight = 0.0
        self.objective_optimizer_min_updates = 0
        self.deployment_optimizer_enabled = False
        self.constraint_value_coeff = 0.0
        self.risk_value_coeff = 0.0
        self.response_coeff = 0.0
        self.risk_coeff = 0.0
        self.delayed_reward_horizon = 1
        self.delayed_reward_blend = 0.0
        self.state_action_mi_coeff = 0.0
        self.direct_actor_credit_weight = 0.0
        self.constraints_active = False
        self.risk_active = False
        self.constraint_lambdas = torch.zeros(
            self.constraint_dim, dtype=torch.float32, device=self.device
        )

        self.buf: List[Transition] = []
        self.action_visits = np.zeros(self.action_dim, dtype=np.int64)
        self.action_ever_feasible = np.zeros(self.action_dim, dtype=bool)
        self.last_policy_entropy_fraction = 1.0
        self.last_continuous_magnitude = 0.0
        self.last_raw_argmax_concentration = 0.0
        self.last_state_action_sensitivity = 0.0
        self.last_state_action_mutual_information = 0.0
        self.last_optimizer_action_changed = 0.0
        self.last_optimizer_q_spread = 0.0
        self.last_constraint_values = torch.zeros(
            self.constraint_dim, dtype=torch.float32, device=self.device
        )
        self.last_risk_value = torch.zeros((), dtype=torch.float32, device=self.device)
        self.last_response_pred = torch.zeros(
            self.response_dim, dtype=torch.float32, device=self.device
        )
        self.update_calls = 0
        self.q_update_calls = 0
        self._actor_hidden: Optional[torch.Tensor] = None
        self._critic_hidden: Optional[torch.Tensor] = None
        self._pending_actor_hidden = torch.empty(
            0, dtype=torch.float32, device=self.device
        )
        self._pending_critic_hidden = torch.empty(
            0, dtype=torch.float32, device=self.device
        )
        self._pending_critic_state = torch.empty(
            0, dtype=torch.float32, device=self.device
        )
        self._pending_exploration_rate = 0.0

    @property
    def buffer_size(self) -> int:
        return len(self.buf)

    def discard_rollout(self) -> int:
        """Drop an undersized rollout at an environment-regime boundary."""
        count = len(self.buf)
        self.buf.clear()
        return int(count)

    def reset_recurrent_state(self) -> None:
        """Reset inference memory at a true episode/regime boundary."""
        self._actor_hidden = None
        self._critic_hidden = None
        self._pending_actor_hidden = torch.empty(
            0, dtype=torch.float32, device=self.device
        )
        self._pending_critic_hidden = torch.empty(
            0, dtype=torch.float32, device=self.device
        )
        self._pending_critic_state = torch.empty(
            0, dtype=torch.float32, device=self.device
        )

    def reset_learning_context(self) -> None:
        """Reset episode-local inference and aggregate reward diagnostics.

        Regime-keyed reward baselines deliberately persist: they were estimated
        from completed prior episodes and are therefore action-independent for
        the next episode. Policy and optimizer parameters remain untouched.
        """
        self.reset_recurrent_state()
        self.average_reward_rate = 0.0
        self.average_reward_initialized = False

    @torch.no_grad()
    def observe_recurrent_state(
        self,
        state: np.ndarray,
        critic_state: Optional[np.ndarray | torch.Tensor] = None,
    ) -> None:
        """Advance recurrent memory on a public frame without choosing an action."""
        if not self.persistent_recurrence:
            return
        state_t = self._tensor_from_value(
            state, dtype=torch.float32, device=self.device
        ).reshape(1, -1)
        critic_state_t = (
            None
            if not self.privileged_critic
            else self._tensor_from_value(
                critic_state, dtype=torch.float32, device=self.device
            ).reshape(1, -1)
        )
        _, actor_hidden = self.net.actor_forward(state_t, self._actor_hidden)
        _, _, critic_hidden = self.net.critic_forward(
            state_t,
            critic_state=critic_state_t,
            hidden=self._critic_hidden,
        )
        self._actor_hidden = actor_hidden.detach()
        self._critic_hidden = critic_hidden.detach()

    def recurrent_state_dict(self) -> Dict[str, torch.Tensor]:
        """Capture execution-only recurrent state for transactional evaluation."""
        return {
            "actor": (
                torch.empty(0, device=self.device)
                if self._actor_hidden is None
                else self._actor_hidden.detach().clone()
            ),
            "critic": (
                torch.empty(0, device=self.device)
                if self._critic_hidden is None
                else self._critic_hidden.detach().clone()
            ),
        }

    def load_recurrent_state_dict(
        self, state: Optional[Dict[str, torch.Tensor]]
    ) -> None:
        state = state or {}
        actor = state.get("actor")
        critic = state.get("critic")
        self._actor_hidden = (
            None
            if actor is None or actor.numel() == 0
            else actor.detach().clone().to(self.device)
        )
        self._critic_hidden = (
            None
            if critic is None or critic.numel() == 0
            else critic.detach().clone().to(self.device)
        )

    @staticmethod
    def _tensor_from_value(
        value: object,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.detach().to(device=device, dtype=dtype)
        # Torch builds compiled against NumPy 1.x cannot reliably consume NumPy
        # 2.x scalar/array objects directly (notably ``numpy.bool`` masks).
        # Converting through Python containers keeps the public API unchanged
        # and avoids depending on Torch's optional NumPy bridge.
        if isinstance(value, (np.ndarray, np.generic)):
            value = value.tolist()
        return torch.tensor(value, dtype=dtype, device=device)

    @classmethod
    def _coerce_action_mask(
        cls,
        action_mask: Optional[np.ndarray | List[bool] | torch.Tensor],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        if logits.ndim not in (1, 2):
            raise ValueError("categorical logits must have one or two dimensions")
        target_shape = logits.shape
        if action_mask is None:
            mask = torch.ones(target_shape, dtype=torch.bool, device=logits.device)
        else:
            mask = cls._tensor_from_value(
                action_mask, dtype=torch.bool, device=logits.device
            )
            if logits.ndim == 1:
                mask = mask.reshape(-1)
                if mask.shape != target_shape:
                    raise ValueError(
                        f"action mask shape {tuple(mask.shape)} != logits {tuple(target_shape)}"
                    )
            else:
                if mask.ndim == 1 and mask.numel() == target_shape[-1]:
                    mask = mask.reshape(1, -1).expand(target_shape[0], -1)
                elif tuple(mask.shape) != tuple(target_shape):
                    raise ValueError(
                        f"action mask shape {tuple(mask.shape)} != logits {tuple(target_shape)}"
                    )
        mask = mask.clone()
        if logits.ndim == 1:
            if not bool(mask.any()):
                mask[0] = True
        else:
            empty_rows = ~mask.any(dim=-1)
            if bool(empty_rows.any()):
                mask[empty_rows, 0] = True
        return mask

    @classmethod
    def _masked_logits(
        cls,
        logits: torch.Tensor,
        action_mask: Optional[np.ndarray | List[bool] | torch.Tensor],
    ) -> torch.Tensor:
        mask = cls._coerce_action_mask(action_mask, logits)
        return logits.masked_fill(~mask, -1e9)

    @classmethod
    def _exploratory_distribution(
        cls,
        logits: torch.Tensor,
        exploration_rate: torch.Tensor | float,
        action_mask: Optional[np.ndarray | List[bool] | torch.Tensor] = None,
    ) -> torch.distributions.Categorical:
        mask = cls._coerce_action_mask(action_mask, logits)
        masked_logits = cls._masked_logits(logits, mask)
        policy_probabilities = torch.softmax(masked_logits, dim=-1)
        uniform_probabilities = mask.to(dtype=logits.dtype)
        uniform_probabilities = uniform_probabilities / uniform_probabilities.sum(
            dim=-1, keepdim=True
        ).clamp(min=1.0)
        epsilon = torch.as_tensor(
            exploration_rate, dtype=logits.dtype, device=logits.device
        ).clamp(0.0, 1.0)
        if logits.ndim == 2:
            if epsilon.ndim == 0:
                epsilon = epsilon.expand(logits.shape[0])
            epsilon = epsilon.reshape(-1, 1)
            if epsilon.shape[0] != logits.shape[0]:
                raise ValueError(
                    "batched exploration rates must match the logits batch"
                )
        elif epsilon.numel() != 1:
            raise ValueError("scalar logits require one exploration rate")
        probabilities = (
            (1.0 - epsilon) * policy_probabilities
            + epsilon * uniform_probabilities
        )
        return torch.distributions.Categorical(probs=probabilities)

    def exploration_rate_at_progress(self, progress: float) -> float:
        """Cosine-decay behavior coverage after an initial warmup plateau."""
        progress = float(np.clip(progress, 0.0, 1.0))
        if progress <= self.exploration_warmup_fraction:
            return self.initial_exploration_rate
        if progress >= self.exploration_fraction:
            return self.final_exploration_rate
        span = max(
            1e-12,
            self.exploration_fraction - self.exploration_warmup_fraction,
        )
        phase = (progress - self.exploration_warmup_fraction) / span
        cosine_weight = 0.5 * (1.0 + math.cos(math.pi * phase))
        return float(
            self.final_exploration_rate
            + (self.initial_exploration_rate - self.final_exploration_rate)
            * cosine_weight
        )

    def set_training_progress(self, progress: float) -> None:
        self.exploration_rate = self.exploration_rate_at_progress(progress)

    def _set_learning_rates(self, actor_lr: float) -> None:
        self.curr_lr = float(max(self.min_lr, actor_lr))
        for group in self.actor_opt.param_groups:
            group["lr"] = self.curr_lr
        for group in self.critic_opt.param_groups:
            group["lr"] = self.curr_lr * self.critic_lr_multiplier
        if self.q_opt is not None:
            for group in self.q_opt.param_groups:
                group["lr"] = self.curr_lr * self.critic_lr_multiplier

    def set_stage_controls(
        self,
        *,
        exploration_rate: float,
        entropy_scale: float,
        learning_rate_scale: float,
    ) -> None:
        del entropy_scale
        self.exploration_rate = float(np.clip(exploration_rate, 0.0, 1.0))
        self._set_learning_rates(
            self.base_lr * float(max(1e-6, learning_rate_scale))
        )

    def set_optimization_context(
        self,
        constraint_lambdas: Optional[
            np.ndarray | List[float] | Tuple[float, ...]
        ] = None,
        risk_coeff: Optional[float] = None,
        constraints_active: Optional[bool] = None,
        risk_active: Optional[bool] = None,
    ) -> None:
        del constraint_lambdas, risk_coeff, constraints_active, risk_active
        self.constraint_lambdas.zero_()
        self.risk_coeff = 0.0
        self.constraints_active = False
        self.risk_active = False

    def _deployment_optimizer_logits(
        self,
        logits: torch.Tensor,
        q_values: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        del q_values, action_mask
        return logits, {
            "optimizer_ready": 0.0,
            "optimizer_action_changed": 0.0,
            "optimizer_q_spread": 0.0,
        }

    @torch.no_grad()
    def policy_diagnostics(
        self,
        state: np.ndarray,
        action_features: Optional[np.ndarray] = None,
        action_mask: Optional[np.ndarray] = None,
        temperature: float = 1.0,
    ) -> Dict[str, float]:
        del action_features
        state_t = self._tensor_from_value(
            state, dtype=torch.float32, device=self.device
        ).reshape(1, -1)
        logits, _ = self.net.actor_forward(
            state_t,
            self._actor_hidden if self.persistent_recurrence else None,
        )
        logits = torch.nan_to_num(
            logits, nan=0.0, posinf=20.0, neginf=-20.0
        ).clamp(-20.0, 20.0)
        mask = self._coerce_action_mask(action_mask, logits)
        logits = self._masked_logits(logits, mask)
        temperature = float(max(1e-6, temperature))
        probs = torch.softmax(logits / temperature, dim=-1).reshape(-1)
        order = torch.argsort(probs, descending=True)
        top = int(order[0].item())
        second = int(order[1].item()) if order.numel() > 1 else top
        top_prob = float(probs[top].item())
        second_prob = float(probs[second].item()) if second != top else 0.0
        entropy = float(
            (-(probs * torch.log(probs.clamp(min=1e-12))).sum()).item()
        )
        return {
            "policy_top_action": float(top),
            "policy_second_action": float(second),
            "policy_top_prob": top_prob,
            "policy_second_prob": second_prob,
            "policy_action_margin": top_prob - second_prob,
            "policy_hold_prob": float(probs[0].item()),
            "policy_entropy": entropy,
            "policy_temperature": temperature,
            "optimizer_top_action": float(top),
            "optimizer_second_action": float(second),
            "optimizer_top_prob": top_prob,
            "optimizer_action_margin": top_prob - second_prob,
            "optimizer_ready": 0.0,
            "optimizer_action_changed": 0.0,
            "optimizer_q_spread": 0.0,
        }

    @torch.no_grad()
    def act(
        self,
        state: np.ndarray,
        deterministic: bool = False,
        action_features: Optional[np.ndarray] = None,
        action_mask: Optional[np.ndarray] = None,
        policy_mode: str = "argmax_raw",
        policy_temperature: float = 0.50,
        critic_state: Optional[np.ndarray | torch.Tensor] = None,
    ) -> Tuple[
        int,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        state_t = self._tensor_from_value(
            state, dtype=torch.float32, device=self.device
        ).reshape(1, -1)
        if state_t.shape[-1] != self.state_dim:
            raise ValueError(
                f"State dim mismatch: got {state_t.shape[-1]}, expected {self.state_dim}"
            )
        critic_state_t = (
            state_t
            if not self.privileged_critic
            else self._tensor_from_value(
                critic_state,
                dtype=torch.float32,
                device=self.device,
            ).reshape(1, -1)
            if critic_state is not None
            else None
        )
        if self.action_feature_dim > 0:
            if action_features is None:
                raise ValueError(
                    "action features are required when action_feature_dim > 0"
                )
            action_features_t = self._tensor_from_value(
                action_features,
                dtype=torch.float32,
                device=self.device,
            ).reshape(self.action_dim, self.action_feature_dim)
        else:
            action_features_t = torch.empty(
                (self.action_dim, 0), dtype=torch.float32, device=self.device
            )
        actor_hidden_in = (
            self._actor_hidden.detach().clone()
            if self._actor_hidden is not None
            else None
        )
        critic_hidden_in = (
            self._critic_hidden.detach().clone()
            if self._critic_hidden is not None
            else None
        )
        logits, actor_hidden_out = self.net.actor_forward(
            state_t,
            actor_hidden_in if self.persistent_recurrence else None,
        )
        value, _, critic_hidden_out = self.net.critic_forward(
            state_t,
            critic_state=critic_state_t,
            hidden=critic_hidden_in if self.persistent_recurrence else None,
        )
        constraint_values = state_t.new_zeros((1, self.constraint_dim))
        risk_value = state_t.new_zeros((1,))
        response = state_t.new_zeros((1, self.response_dim))
        logits = torch.nan_to_num(
            logits, nan=0.0, posinf=20.0, neginf=-20.0
        ).clamp(-20.0, 20.0)
        mask = self._coerce_action_mask(action_mask, logits)
        logits = self._masked_logits(logits, mask)
        feasible = np.asarray(mask.squeeze(0).cpu().tolist(), dtype=bool)
        self.action_ever_feasible |= feasible

        raw_policy_dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            mode = str(policy_mode or "argmax_raw").strip().lower()
            if mode in {"argmax", "argmax_raw"}:
                action_t = torch.argmax(logits, dim=-1)
            elif mode == "sample_raw":
                action_t = raw_policy_dist.sample()
            elif mode == "sample_low_temp":
                action_t = torch.distributions.Categorical(
                    logits=logits / float(max(1e-6, policy_temperature))
                ).sample()
            else:
                raise ValueError(
                    "policy_mode must be one of: argmax, argmax_raw, "
                    "sample_raw, sample_low_temp"
                )
        else:
            behavior_dist = self._exploratory_distribution(
                logits, self.exploration_rate, mask
            )
            action_t = behavior_dist.sample()

        action = int(action_t.item())
        applied_exploration_rate = (
            0.0 if deterministic else float(self.exploration_rate)
        )
        self.last_action_exploration_rate = applied_exploration_rate
        self._pending_exploration_rate = applied_exploration_rate
        self.last_continuous_magnitude = 0.0 if action == 0 else 1.0
        self.last_constraint_values = constraint_values.squeeze(0).detach()
        self.last_risk_value = risk_value.squeeze(0).detach()
        self.last_response_pred = response.squeeze(0).detach()
        if not deterministic:
            self.action_visits[action] += 1
        if self.persistent_recurrence:
            self._pending_actor_hidden = (
                torch.empty(0, dtype=torch.float32, device=self.device)
                if actor_hidden_in is None
                else actor_hidden_in.squeeze(1).squeeze(0).detach().clone()
            )
            self._pending_critic_hidden = (
                torch.empty(0, dtype=torch.float32, device=self.device)
                if critic_hidden_in is None
                else critic_hidden_in.squeeze(1).squeeze(0).detach().clone()
            )
            self._actor_hidden = actor_hidden_out.detach()
            self._critic_hidden = critic_hidden_out.detach()
        self._pending_critic_state = (
            torch.empty(0, dtype=torch.float32, device=self.device)
            if critic_state_t is None
            else critic_state_t.squeeze(0).detach().clone()
        )
        return (
            action,
            state_t.squeeze(0).detach(),
            (
                raw_policy_dist if deterministic else behavior_dist
            ).log_prob(action_t).squeeze(0).detach(),
            value.squeeze(0).detach(),
            action_features_t.detach(),
            mask.squeeze(0).detach(),
        )

    @torch.no_grad()
    def value_estimates(
        self,
        state: np.ndarray,
        action_features: Optional[np.ndarray] = None,
        critic_state: Optional[np.ndarray | torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del action_features
        state_t = self._tensor_from_value(
            state, dtype=torch.float32, device=self.device
        ).reshape(1, -1)
        critic_state_t = (
            state_t
            if not self.privileged_critic
            else self._tensor_from_value(
                critic_state,
                dtype=torch.float32,
                device=self.device,
            ).reshape(1, -1)
            if critic_state is not None
            else None
        )
        value, _, _ = self.net.critic_forward(
            state_t,
            critic_state=critic_state_t,
            hidden=self._critic_hidden if self.persistent_recurrence else None,
        )
        constraint_values = state_t.new_zeros((1, self.constraint_dim))
        risk_value = state_t.new_zeros((1,))
        return (
            value.squeeze(0).detach(),
            constraint_values.squeeze(0).detach(),
            risk_value.squeeze(0).detach(),
        )

    def store(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        done: bool,
        next_state: Optional[torch.Tensor],
        old_logp: torch.Tensor,
        old_value: torch.Tensor,
        actor_credit: float = 0.0,
        constraint_costs: Optional[
            np.ndarray | List[float] | Tuple[float, ...] | torch.Tensor
        ] = None,
        risk_cost: float = 0.0,
        response_target: Optional[
            np.ndarray | List[float] | Tuple[float, ...] | torch.Tensor
        ] = None,
        action_features: Optional[
            np.ndarray | List[List[float]] | torch.Tensor
        ] = None,
        action_mask: Optional[np.ndarray | List[bool] | torch.Tensor] = None,
        discount: Optional[float] = None,
        duration: int = 1,
        truncated: bool = False,
        next_value: Optional[torch.Tensor | float] = None,
        next_constraint_values: Optional[torch.Tensor | np.ndarray] = None,
        next_risk_value: Optional[torch.Tensor | float] = None,
        critic_state: Optional[np.ndarray | List[float] | torch.Tensor] = None,
        counterfactual_action_values: Optional[
            np.ndarray | List[float] | torch.Tensor
        ] = None,
        counterfactual_action_mask: Optional[
            np.ndarray | List[bool] | torch.Tensor
        ] = None,
        potential: Optional[float] = None,
        next_potential: Optional[float] = None,
        reward_regime: str = "default",
    ) -> None:
        del (
            next_state,
            actor_credit,
            constraint_costs,
            risk_cost,
            response_target,
            next_constraint_values,
            next_risk_value,
        )
        duration = int(max(1, duration))
        resolved_discount = float(
            self.gamma**duration if discount is None else discount
        )
        if not np.isfinite(resolved_discount) or not 0.0 <= resolved_discount <= 1.0:
            raise ValueError(
                "transition discount must be finite and in [0, 1], "
                f"got {resolved_discount}"
            )
        mask = self._coerce_action_mask(
            action_mask,
            torch.zeros(self.action_dim, dtype=torch.float32, device=self.device),
        ).reshape(-1)
        if self.action_feature_dim > 0:
            if action_features is None:
                raise ValueError(
                    "action features are required when action_feature_dim > 0"
                )
            action_features_t = self._tensor_from_value(
                action_features,
                dtype=torch.float32,
                device=self.device,
            ).reshape(self.action_dim, self.action_feature_dim)
        else:
            action_features_t = torch.empty(
                (self.action_dim, 0), dtype=torch.float32, device=self.device
            )
        next_value_t = torch.as_tensor(
            float("nan") if next_value is None else next_value,
            dtype=torch.float32,
            device=self.device,
        ).reshape(())
        critic_state_t = (
            self._pending_critic_state.detach().clone()
            if critic_state is None
            else self._tensor_from_value(
                critic_state, dtype=torch.float32, device=self.device
            ).reshape(-1)
        )
        if self.privileged_critic and critic_state_t.numel() != self.critic_state_dim:
            raise ValueError(
                "privileged critic_state must have "
                f"{self.critic_state_dim} values, got {critic_state_t.numel()}"
            )
        if not self.privileged_critic:
            critic_state_t = torch.empty(
                0, dtype=torch.float32, device=self.device
            )

        counterfactual_values_t = torch.full(
            (self.action_dim,),
            float("nan"),
            dtype=torch.float32,
            device=self.device,
        )
        if counterfactual_action_values is not None:
            supplied = self._tensor_from_value(
                counterfactual_action_values,
                dtype=torch.float32,
                device=self.device,
            ).reshape(-1)
            if supplied.numel() != self.action_dim:
                raise ValueError(
                    "counterfactual_action_values must match action_dim: "
                    f"{supplied.numel()} != {self.action_dim}"
                )
            counterfactual_values_t = supplied
        counterfactual_mask_t = torch.isfinite(counterfactual_values_t)
        if counterfactual_action_mask is not None:
            supplied_mask = self._tensor_from_value(
                counterfactual_action_mask,
                dtype=torch.bool,
                device=self.device,
            ).reshape(-1)
            if supplied_mask.numel() != self.action_dim:
                raise ValueError(
                    "counterfactual_action_mask must match action_dim"
                )
            counterfactual_mask_t &= supplied_mask

        raw_reward = float(reward)
        potential_delta = 0.0
        if self.potential_shaping_scale > 0.0:
            if potential is None:
                raise ValueError(
                    "potential shaping requires the current-state potential"
                )
            if next_potential is None and not done:
                credited_reward = raw_reward
            else:
                credited_reward = self.potential_based_reward(
                    raw_reward,
                    resolved_discount,
                    float(potential),
                    0.0 if next_potential is None else float(next_potential),
                    terminal=bool(done),
                    scale=self.potential_shaping_scale,
                )
                potential_delta = credited_reward - raw_reward
        else:
            credited_reward = raw_reward
        self.buf.append(
            Transition(
                state=state.detach().to(self.device, dtype=torch.float32),
                critic_state=critic_state_t.detach(),
                action=torch.tensor(action, dtype=torch.long, device=self.device),
                raw_reward=raw_reward,
                reward=credited_reward,
                done=bool(done),
                truncated=bool(truncated),
                discount=resolved_discount,
                duration=duration,
                old_logp=old_logp.detach().to(self.device).reshape(()),
                old_value=old_value.detach().to(self.device).reshape(()),
                exploration_rate=float(self._pending_exploration_rate),
                reward_regime=str(reward_regime or "default"),
                action_features=action_features_t.detach(),
                action_mask=mask.detach(),
                next_value_override=next_value_t.detach(),
                actor_hidden=self._pending_actor_hidden.detach().clone(),
                critic_hidden=self._pending_critic_hidden.detach().clone(),
                counterfactual_values=counterfactual_values_t.detach(),
                counterfactual_mask=counterfactual_mask_t.detach(),
                potential_delta=float(potential_delta),
                potential=(
                    float("nan") if potential is None else float(potential)
                ),
                next_potential_override=(
                    float("nan")
                    if next_potential is None
                    else float(next_potential)
                ),
                next_state_finalized=bool(
                    done
                    or (
                        torch.isfinite(next_value_t).item()
                        and (
                            self.potential_shaping_scale <= 0.0
                            or next_potential is not None
                        )
                    )
                ),
            )
        )
        self._pending_actor_hidden = torch.empty(
            0, dtype=torch.float32, device=self.device
        )
        self._pending_critic_hidden = torch.empty(
            0, dtype=torch.float32, device=self.device
        )
        self._pending_critic_state = torch.empty(
            0, dtype=torch.float32, device=self.device
        )
        self._pending_exploration_rate = 0.0

    def finalize_last_transition(
        self,
        *,
        next_value: torch.Tensor | float,
        next_potential: Optional[float] = None,
    ) -> None:
        """Attach the value/potential of the actual next actor decision state."""
        if not self.buf:
            raise ValueError("cannot finalize an empty rollout")
        transition = self.buf[-1]
        transition.next_value_override = torch.as_tensor(
            next_value, dtype=torch.float32, device=self.device
        ).reshape(()).detach()
        if self.potential_shaping_scale > 0.0:
            if not np.isfinite(transition.potential):
                raise ValueError("stored transition is missing its potential")
            if next_potential is None or not np.isfinite(float(next_potential)):
                raise ValueError("a finite next potential is required")
            transition.next_potential_override = float(next_potential)
            transition.reward = self.potential_based_reward(
                transition.raw_reward,
                transition.discount,
                transition.potential,
                float(next_potential),
                terminal=transition.done,
                scale=self.potential_shaping_scale,
            )
            transition.potential_delta = (
                transition.reward - transition.raw_reward
            )
        transition.next_state_finalized = True

    @staticmethod
    def potential_based_reward(
        reward: float,
        discount: float,
        potential: float,
        next_potential: float,
        *,
        terminal: bool = False,
        scale: float = 1.0,
    ) -> float:
        """Return ``r + scale * (discount * Phi(s') - Phi(s))``.

        Keeping this transform explicit makes the telescoping/invariance
        contract independently testable and prevents reward-formula edits from
        being hidden inside the market simulator.
        """
        values = (reward, discount, potential, next_potential, scale)
        if not all(np.isfinite(float(value)) for value in values):
            raise ValueError("potential shaping inputs must be finite")
        if not 0.0 <= float(discount) <= 1.0:
            raise ValueError("potential shaping discount must be in [0, 1]")
        return float(
            reward
            + scale
            * (
                float(discount) * float(next_potential) * (not terminal)
                - float(potential)
            )
        )

    @staticmethod
    def center_smdp_reward_sums(
        reward_sums: torch.Tensor,
        exposures: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """Center discounted SMDP returns with a per-market-step rate.

        Both inputs retain their natural units: ``reward_sums`` is the
        discounted sum accrued while an action was held and ``exposures`` is
        the matching discounted count of primitive market steps.
        """
        reward_sums = torch.as_tensor(reward_sums, dtype=torch.float32)
        exposures = torch.as_tensor(
            exposures, dtype=torch.float32, device=reward_sums.device
        )
        if reward_sums.shape != exposures.shape:
            raise ValueError("SMDP reward sums and exposures must align")
        exposure = float(exposures.double().sum().item())
        if not np.isfinite(exposure) or exposure <= 0.0:
            raise ValueError("SMDP reward exposure must be finite and positive")
        rate = float(reward_sums.double().sum().item() / exposure)
        return reward_sums - rate * exposures, rate

    @torch.no_grad()
    def _gae_scalar(
        self,
        rewards: torch.Tensor,
        old_values: torch.Tensor,
        dones: torch.Tensor,
        bootstrap_value: Optional[torch.Tensor | float] = None,
        normalize_advantage: bool = True,
        discounts: Optional[torch.Tensor] = None,
        truncations: Optional[torch.Tensor] = None,
        next_value_overrides: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rewards = rewards.to(self.device, dtype=torch.float32).reshape(-1)
        old_values = old_values.to(self.device, dtype=torch.float32).reshape(-1)
        dones = dones.to(self.device, dtype=torch.float32).reshape(-1)
        size = rewards.numel()
        if not (old_values.numel() == dones.numel() == size):
            raise ValueError("GAE reward, value, and terminal vectors must align")
        discounts = (
            torch.full((size,), self.gamma, dtype=torch.float32, device=self.device)
            if discounts is None
            else torch.as_tensor(
                discounts, dtype=torch.float32, device=self.device
            ).reshape(-1)
        )
        truncations = (
            torch.zeros(size, dtype=torch.float32, device=self.device)
            if truncations is None
            else torch.as_tensor(
                truncations, dtype=torch.float32, device=self.device
            ).reshape(-1)
        )
        next_value_overrides = (
            torch.full(
                (size,), float("nan"), dtype=torch.float32, device=self.device
            )
            if next_value_overrides is None
            else torch.as_tensor(
                next_value_overrides, dtype=torch.float32, device=self.device
            ).reshape(-1)
        )
        if not (
            discounts.numel()
            == truncations.numel()
            == next_value_overrides.numel()
            == size
        ):
            raise ValueError("semi-MDP GAE metadata must match reward length")

        # This market is a continuing task. A rollout or training-horizon cut
        # is not an economic terminal state, so silently substituting V=0 at
        # the tail would teach the actor an artificial end-of-horizon policy.
        # Require either a true terminal, a value of the final observation, or
        # an explicit rollout-tail bootstrap. Internal time-limit boundaries
        # always need their own final-observation value because the following
        # transition may belong to a different episode or training stage.
        if size:
            internal_truncations_without_value = (
                truncations[:-1] > 0.5
            ) & (dones[:-1] <= 0.5) & ~torch.isfinite(
                next_value_overrides[:-1]
            )
            if bool(internal_truncations_without_value.any().item()):
                raise ValueError(
                    "internal truncated transitions require a finite "
                    "next_value override"
                )
            tail_is_terminal = bool(dones[-1].item() > 0.5)
            tail_has_override = bool(
                torch.isfinite(next_value_overrides[-1]).item()
            )
            if (
                not tail_is_terminal
                and not tail_has_override
                and bootstrap_value is None
            ):
                raise ValueError(
                    "non-terminal PPO rollout tails require bootstrap_value "
                    "or a finite next_value override; zero bootstrapping would "
                    "create finite-horizon end effects"
                )

        bootstrap = torch.as_tensor(
            0.0 if bootstrap_value is None else bootstrap_value,
            dtype=torch.float32,
            device=self.device,
        ).reshape(())
        advantages = torch.zeros(size, dtype=torch.float32, device=self.device)
        gae = torch.zeros((), dtype=torch.float32, device=self.device)
        for index in reversed(range(size)):
            if torch.isfinite(next_value_overrides[index]):
                next_value = next_value_overrides[index]
            elif index == size - 1:
                next_value = bootstrap
            else:
                next_value = old_values[index + 1]
            terminal = float(dones[index].item())
            truncated = float(truncations[index].item())
            delta = (
                rewards[index]
                + discounts[index] * next_value * (1.0 - terminal)
                - old_values[index]
            )
            trace_continuation = max(0.0, 1.0 - terminal - truncated)
            gae = (
                delta
                + discounts[index]
                * self.lam
                * trace_continuation
                * gae
            )
            advantages[index] = gae
        returns = advantages + old_values
        if normalize_advantage:
            mean = advantages.mean()
            std = advantages.std(unbiased=False)
            advantages = (
                (advantages - mean) / (std + 1e-8)
                if float(std.item()) > 1e-8
                else advantages - mean
            )
        return (
            torch.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0),
            torch.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0),
        )

    def _empty_update_metrics(self) -> Dict[str, object]:
        return {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "critic_loss": 0.0,
            "critic_grad_norm": 0.0,
            "critic_grad_clip_fraction": 0.0,
            "actor_grad_norm": 0.0,
            "actor_grad_clip_fraction": 0.0,
            "constraint_value_loss": 0.0,
            "risk_value_loss": 0.0,
            "action_q_loss": 0.0,
            "action_q_cql_loss": 0.0,
            "counterfactual_policy_loss": 0.0,
            "response_loss": 0.0,
            "opponent_forecast_loss": 0.0,
            "lagrangian_adv_mean": 0.0,
            "lagrangian_adv_std": 0.0,
            "direct_actor_credit_mean": 0.0,
            "direct_actor_credit_std": 0.0,
            "direct_actor_credit_weight": 0.0,
            "constraint_lambda_mean": 0.0,
            "risk_coeff": 0.0,
            "constraints_active": False,
            "risk_active": False,
            "entropy": 0.0,
            "policy_entropy": 0.0,
            "discrete_policy_entropy": 0.0,
            "magnitude_entropy": 0.0,
            "state_action_mi": 0.0,
            "state_action_mutual_information": 0.0,
            "raw_policy_entropy_fraction": 0.0,
            "raw_policy_hold_probability": 0.0,
            "raw_policy_argmax_concentration": 0.0,
            "state_action_sensitivity": 0.0,
            "approx_kl": 0.0,
            "target_kl": self.target_kl,
            "clipfrac": 0.0,
            "update_performed": False,
            "run_mode": "no_rollout_data",
            "explained_variance": 0.0,
            "ent_coeff": self.ent_coeff,
            "target_entropy_fraction": self.target_entropy_fraction,
            "entropy_fraction_error": 0.0,
            "clip_eps": self.clip_eps,
            "lr": self.curr_lr,
            "critic_lr": self.curr_lr * self.critic_lr_multiplier,
            "exploration_rate": float(self.exploration_rate),
            "action_coverage": self._action_coverage(),
            "rollout_action_diversity": 0.0,
            "rollout_reward_std": 0.0,
            "credited_reward_std": 0.0,
            "rollout_reward_exposure": 0.0,
            "rollout_reward_rate_residual": 0.0,
            "critic_target_mean": 0.0,
            "critic_target_std": 0.0,
            "gae_return_mean": 0.0,
            "gae_return_std": 0.0,
            "critic_target_contract": (
                "actual_next_decision_td0"
                if self.td_value_targets
                else "gae_lambda_return"
            ),
            "average_reward_rate": float(self.average_reward_rate),
            "reward_rate_beta": self.reward_rate_beta,
            "center_rewards": self.center_rewards,
            "reward_rate_segment_count": 0,
            "reward_rate_segment_std": 0.0,
            "reward_rate_lagged_baseline_fraction": 0.0,
            "opponent_segment_count": 0,
            "opponent_regime_count": 0,
            "opponent_sample_weight_min": 1.0,
            "opponent_sample_weight_max": 1.0,
            "delayed_reward_blend": 0.0,
            "continuous_magnitude_mean": 0.0,
            "continuous_magnitude_std": 0.0,
            "learning_signal_ok": False,
            "optimizer_steps": 0,
            "actor_optimizer_steps": 0,
            "critic_optimizer_steps": 0,
            "actor_epochs_requested": 0,
            "critic_epochs_requested": 0,
            "response_optimizer_steps": 0,
            "objective_q_optimizer_steps": 0,
            "objective_q_update_calls": 0,
            "objective_optimizer_ready": False,
            "counterfactual_target_coverage": 0.0,
            "counterfactual_actor_weight": self.counterfactual_actor_weight,
            "potential_shaping_delta_mean": 0.0,
            "stopped_early_kl": False,
            "rollout_size": 0,
            "effective_batch_size": 0,
            "policy_entropy_fraction": self.last_policy_entropy_fraction,
            "transition_duration_mean": 0.0,
            "transition_discount_mean": 0.0,
            "truncation_count": 0,
        }

    @staticmethod
    def _critic_value_loss(
        values: torch.Tensor,
        targets: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Robust raw-value regression without suppressing its gradient.

        Dividing both tensors by the rollout standard deviation and then
        backpropagating through the divided prediction attenuates the critic
        gradient by that standard deviation a second time.  Long-return
        rollouts therefore produced an effectively frozen value function.
        A raw Huber loss with ``beta`` set to the target scale has the same
        robust transition from quadratic to linear error, while preserving a
        useful gradient for the raw value head.
        """
        target_scale = targets.std(unbiased=False).detach().clamp(min=1.0)
        errors = nn.functional.smooth_l1_loss(
            values,
            targets,
            beta=float(target_scale.item()),
            reduction="none",
        )
        if sample_weights is None:
            return errors.mean()
        weights = torch.as_tensor(
            sample_weights, dtype=errors.dtype, device=errors.device
        ).reshape_as(errors).clamp(min=0.0)
        return (errors * weights).sum() / weights.sum().clamp(min=1e-8)

    @staticmethod
    def _one_step_value_targets(
        rewards: torch.Tensor,
        discounts: torch.Tensor,
        dones: torch.Tensor,
        next_values: torch.Tensor,
    ) -> torch.Tensor:
        """Exact decision-to-decision TD targets for the state-value critic."""
        vectors = tuple(
            torch.as_tensor(vector, dtype=torch.float32).reshape(-1)
            for vector in (rewards, discounts, dones, next_values)
        )
        if len({vector.numel() for vector in vectors}) != 1:
            raise ValueError("one-step value target vectors must align")
        reward_t, discount_t, done_t, next_value_t = vectors
        if not bool(torch.isfinite(next_value_t).all().item()):
            raise ValueError(
                "one-step value targets require every actual next-decision value"
            )
        return (
            reward_t
            + discount_t * next_value_t * (1.0 - done_t)
        ).detach()

    @staticmethod
    def _regime_local_reward_baselines(
        raw_rewards: torch.Tensor,
        reward_exposures: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a primitive reward rate for each terminal-delimited regime.

        A stratified on-policy batch can contain several independently reset
        opponent episodes. A batch-wide differential baseline would confound
        opponent difficulty with action quality, so terminal markers define
        the segments that are centered independently.
        """
        raw_rewards = raw_rewards.float().reshape(-1)
        reward_exposures = reward_exposures.float().reshape(-1)
        dones = dones.float().reshape(-1)
        if not (
            raw_rewards.numel()
            == reward_exposures.numel()
            == dones.numel()
        ):
            raise ValueError("reward baseline tensors must have equal length")
        baselines = torch.zeros_like(raw_rewards)
        rates: List[torch.Tensor] = []
        start = 0
        count = int(raw_rewards.numel())
        for index in range(count):
            if float(dones[index].item()) <= 0.0 and index + 1 < count:
                continue
            stop = index + 1
            exposure = reward_exposures[start:stop].sum().clamp(min=1e-8)
            rate = raw_rewards[start:stop].sum() / exposure
            baselines[start:stop] = rate
            rates.append(rate)
            start = stop
        return (
            baselines,
            torch.stack(rates) if rates else raw_rewards.new_zeros(0),
        )

    def _causal_regime_reward_baselines(
        self,
        raw_rewards: torch.Tensor,
        reward_exposures: torch.Tensor,
        dones: torch.Tensor,
        reward_regimes: Sequence[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Return action-independent, lagged baselines for complete episodes.

        A baseline estimated from the same trajectory depends on that
        trajectory's actions and can remove the causal signal of a uniformly
        better policy.  Each segment therefore uses its regime's rate from a
        *previous* completed episode.  At cold start, it uses a leave-one-
        segment-out population rate; when there is only one segment it uses
        zero. Current rates update the cache only after all baselines have been
        fixed.
        """
        raw_rewards = raw_rewards.float().reshape(-1)
        reward_exposures = reward_exposures.float().reshape(-1)
        dones = dones.float().reshape(-1)
        if not (
            raw_rewards.numel()
            == reward_exposures.numel()
            == dones.numel()
            == len(reward_regimes)
        ):
            raise ValueError("causal reward baseline inputs must align")
        segments: List[Tuple[int, int, str, torch.Tensor, torch.Tensor]] = []
        start = 0
        count = int(raw_rewards.numel())
        for index in range(count):
            if float(dones[index].item()) <= 0.0 and index + 1 < count:
                continue
            stop = index + 1
            labels = {str(value or "default") for value in reward_regimes[start:stop]}
            if len(labels) != 1:
                raise ValueError("one terminal segment cannot mix reward regimes")
            exposure = reward_exposures[start:stop].sum().clamp(min=1e-8)
            reward_sum = raw_rewards[start:stop].sum()
            segments.append((start, stop, labels.pop(), reward_sum, exposure))
            start = stop
        baselines = torch.zeros_like(raw_rewards)
        rates = torch.stack(
            [reward_sum / exposure for _, _, _, reward_sum, exposure in segments]
        ) if segments else raw_rewards.new_zeros(0)
        lagged_count = 0
        for segment_index, (start, stop, regime, _, _) in enumerate(segments):
            cached = self.reward_rate_by_regime.get(regime)
            if cached is not None and np.isfinite(cached):
                baseline = raw_rewards.new_tensor(float(cached))
                lagged_count += 1
            elif len(segments) > 1:
                other_reward = torch.stack([
                    candidate[3]
                    for index, candidate in enumerate(segments)
                    if index != segment_index
                ]).sum()
                other_exposure = torch.stack([
                    candidate[4]
                    for index, candidate in enumerate(segments)
                    if index != segment_index
                ]).sum().clamp(min=1e-8)
                baseline = other_reward / other_exposure
            else:
                baseline = raw_rewards.new_zeros(())
            baselines[start:stop] = baseline

        # Update only after every current baseline is fixed, preventing any
        # segment from leaking its own outcome into its actor advantages.
        for _, _, regime, reward_sum, exposure in segments:
            observed = float((reward_sum / exposure).item())
            previous = self.reward_rate_by_regime.get(regime)
            beta = float(np.clip(self.reward_rate_beta, 0.0, 1.0))
            self.reward_rate_by_regime[regime] = (
                observed
                if previous is None
                else float((1.0 - beta) * previous + beta * observed)
            )
        lagged_fraction = float(lagged_count / max(1, len(segments)))
        return baselines, rates, lagged_fraction

    @staticmethod
    def _terminal_segment_weights(dones: torch.Tensor) -> torch.Tensor:
        """Give every terminal-delimited opponent episode equal total mass.

        Response-timescale episodes intentionally have different lengths. A
        transition-uniform loss would therefore turn horizon length into an
        unintended opponent-priority weight. The returned weights have mean
        one, while every segment sums to the same value.
        """
        terminal = torch.as_tensor(dones, dtype=torch.float32).reshape(-1)
        count = int(terminal.numel())
        if count == 0:
            return terminal
        segments: List[Tuple[int, int]] = []
        start = 0
        for index in range(count):
            if float(terminal[index].item()) > 0.0 or index + 1 == count:
                segments.append((start, index + 1))
                start = index + 1
        segment_mass = float(count) / float(max(1, len(segments)))
        weights = torch.empty_like(terminal)
        for start, stop in segments:
            weights[start:stop] = segment_mass / float(max(1, stop - start))
        return weights

    @staticmethod
    def _regime_balanced_weights(
        dones: torch.Tensor,
        reward_regimes: Sequence[str],
    ) -> torch.Tensor:
        """Give every opponent class equal mass despite replicated openings."""
        terminal = torch.as_tensor(dones, dtype=torch.float32).reshape(-1)
        count = int(terminal.numel())
        if count != len(reward_regimes):
            raise ValueError("regime weights require one label per transition")
        if count == 0:
            return terminal
        segments: List[Tuple[int, int, str]] = []
        start = 0
        for index in range(count):
            if float(terminal[index].item()) > 0.0 or index + 1 == count:
                labels = {
                    str(value or "default")
                    for value in reward_regimes[start : index + 1]
                }
                if len(labels) != 1:
                    raise ValueError(
                        "one terminal segment cannot mix reward regimes"
                    )
                segments.append((start, index + 1, labels.pop()))
                start = index + 1
        regime_episode_counts: Dict[str, int] = {}
        for _, _, regime in segments:
            regime_episode_counts[regime] = (
                regime_episode_counts.get(regime, 0) + 1
            )
        regime_mass = float(count) / float(max(1, len(regime_episode_counts)))
        weights = torch.empty_like(terminal)
        for start, stop, regime in segments:
            episode_mass = regime_mass / float(regime_episode_counts[regime])
            weights[start:stop] = episode_mass / float(max(1, stop - start))
        return weights

    def _actor_logits_batch(
        self,
        states: torch.Tensor,
        hidden_states: Optional[Sequence[torch.Tensor]] = None,
    ) -> torch.Tensor:
        if not self.persistent_recurrence:
            logits, _ = self.net.actor_forward(states, None)
            return logits
        outputs: List[torch.Tensor] = []
        assert hidden_states is not None
        for state, hidden in zip(states, hidden_states):
            hidden_in = (
                None
                if hidden.numel() == 0
                else hidden.reshape(1, 1, -1)
            )
            logits, _ = self.net.actor_forward(state.unsqueeze(0), hidden_in)
            outputs.append(logits.squeeze(0))
        return torch.stack(outputs)

    def _critic_outputs_batch(
        self,
        states: torch.Tensor,
        critic_states: Optional[torch.Tensor] = None,
        hidden_states: Optional[Sequence[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.persistent_recurrence:
            values, action_values, _ = self.net.critic_forward(
                states,
                critic_state=critic_states if self.privileged_critic else None,
                hidden=None,
            )
            return values, action_values
        values_out: List[torch.Tensor] = []
        actions_out: List[torch.Tensor] = []
        assert hidden_states is not None
        for index, (state, hidden) in enumerate(zip(states, hidden_states)):
            hidden_in = (
                None
                if hidden.numel() == 0
                else hidden.reshape(1, 1, -1)
            )
            critic_state = (
                None
                if not self.privileged_critic
                else critic_states[index].unsqueeze(0)
            )
            value, action_values, _ = self.net.critic_forward(
                state.unsqueeze(0),
                critic_state=critic_state,
                hidden=hidden_in,
            )
            values_out.append(value.squeeze(0))
            actions_out.append(action_values.squeeze(0))
        return torch.stack(values_out), torch.stack(actions_out)

    @staticmethod
    def _counterfactual_actor_advantages(
        values: torch.Tensor,
        masks: torch.Tensor,
        actions: torch.Tensor,
        *,
        normalize: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return selected-action advantages from observed branches only.

        Unsupervised Q outputs must never enter the actor baseline. Rows need
        the selected action and at least one alternative branch; other rows
        retain their ordinary PPO advantage.
        """
        values = values.float()
        masks = masks.bool()
        actions = actions.long().reshape(-1)
        if values.shape != masks.shape or values.shape[0] != actions.numel():
            raise ValueError("counterfactual tensors must have aligned shapes")
        finite_masks = masks & torch.isfinite(values)
        counts = finite_masks.sum(dim=-1)
        selected_is_observed = finite_masks.gather(
            1, actions.unsqueeze(-1)
        ).squeeze(-1)
        valid = selected_is_observed & (counts >= 2)
        safe_values = torch.where(finite_masks, values, torch.zeros_like(values))
        baselines = safe_values.sum(dim=-1) / counts.clamp(min=1)
        selected = values.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        advantages = torch.where(
            valid, selected - baselines, torch.zeros_like(selected)
        )
        if normalize and bool(valid.any()):
            valid_values = advantages[valid]
            mean = valid_values.mean()
            std = valid_values.std(unbiased=False)
            normalized = (
                (valid_values - mean) / (std + 1e-8)
                if float(std.item()) > 1e-8
                else valid_values - mean
            )
            advantages = advantages.clone()
            advantages[valid] = normalized
        return advantages, valid

    @staticmethod
    def _counterfactual_pairwise_loss(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        masks: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fit within-state causal action differences, not noisy reward levels."""
        valid = masks.bool() & torch.isfinite(targets)
        pair_mask = valid.unsqueeze(2) & valid.unsqueeze(1)
        upper = torch.triu(
            torch.ones(
                predictions.shape[-1],
                predictions.shape[-1],
                dtype=torch.bool,
                device=predictions.device,
            ),
            diagonal=1,
        )
        pair_mask &= upper.unsqueeze(0)
        if not bool(pair_mask.any()):
            return predictions.sum() * 0.0
        prediction_differences = predictions.unsqueeze(2) - predictions.unsqueeze(1)
        target_differences = targets.unsqueeze(2) - targets.unsqueeze(1)
        pair_errors = nn.functional.smooth_l1_loss(
            prediction_differences[pair_mask],
            target_differences[pair_mask],
            reduction="none",
        )
        if sample_weights is None:
            return pair_errors.mean()
        pair_weights = torch.as_tensor(
            sample_weights,
            dtype=pair_errors.dtype,
            device=pair_errors.device,
        ).reshape(-1, 1, 1).expand_as(pair_mask)[pair_mask]
        return (
            (pair_errors * pair_weights).sum()
            / pair_weights.sum().clamp(min=1e-8)
        )

    @staticmethod
    def _observed_counterfactual_policy_loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
        masks: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Rank only action pairs replayed under common random numbers.

        The sampled-action PPO surrogate can reject a poor selected action but
        cannot identify which alternative should inherit its probability. The
        transactional branches provide measured within-state comparisons for
        a coverage-balanced action subset. This term uses only those observed
        pairs. Unobserved actions receive no direct gradient, while tied
        outcomes receive zero weight, so the update neither invents Q labels
        nor requires the deployed policy to move.
        """
        if logits.shape != targets.shape or logits.shape != masks.shape:
            raise ValueError(
                "counterfactual policy tensors must have aligned shapes"
            )
        valid = masks.bool() & torch.isfinite(targets)
        pair_mask = valid.unsqueeze(2) & valid.unsqueeze(1)
        upper = torch.triu(
            torch.ones(
                logits.shape[-1],
                logits.shape[-1],
                dtype=torch.bool,
                device=logits.device,
            ),
            diagonal=1,
        )
        pair_mask = pair_mask & upper.unsqueeze(0)
        target_diff = torch.nan_to_num(
            targets.unsqueeze(2) - targets.unsqueeze(1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        informative = pair_mask & (target_diff.abs() > 1e-6)
        if not bool(informative.any().item()):
            return logits.sum() * 0.0
        observed_abs_diff = target_diff.abs()[informative]
        robust_scale = observed_abs_diff.median().detach().clamp(min=1e-4)
        confidence = torch.tanh(target_diff.abs() / robust_scale)
        signed_logit_diff = (
            logits.unsqueeze(2) - logits.unsqueeze(1)
        ) * torch.sign(target_diff)
        pair_loss = nn.functional.softplus(-signed_logit_diff)
        weights = confidence * informative.float()
        if sample_weights is not None:
            weights = weights * torch.as_tensor(
                sample_weights,
                dtype=weights.dtype,
                device=weights.device,
            ).reshape(-1, 1, 1)
        return (pair_loss * weights).sum() / weights.sum().clamp(min=1e-8)

    def update(
        self,
        epochs: int = 4,
        batch_size: int = 64,
        bootstrap_value: Optional[torch.Tensor | float] = None,
        bootstrap_constraint_values: Optional[torch.Tensor] = None,
        bootstrap_risk_value: Optional[torch.Tensor | float] = None,
    ) -> Dict[str, object]:
        del bootstrap_constraint_values, bootstrap_risk_value
        if not self.buf:
            return self._empty_update_metrics()

        states = torch.stack([transition.state for transition in self.buf])
        states = torch.nan_to_num(states, nan=0.0, posinf=1e3, neginf=-1e3)
        critic_states = (
            torch.stack([transition.critic_state for transition in self.buf])
            if self.privileged_critic
            else None
        )
        actor_hiddens = [transition.actor_hidden for transition in self.buf]
        critic_hiddens = [transition.critic_hidden for transition in self.buf]
        actions = torch.stack([transition.action for transition in self.buf])
        action_features = torch.stack(
            [transition.action_features for transition in self.buf]
        )
        raw_rewards = torch.tensor(
            [transition.raw_reward for transition in self.buf],
            dtype=torch.float32,
            device=self.device,
        )
        rewards = torch.tensor(
            [transition.reward for transition in self.buf],
            dtype=torch.float32,
            device=self.device,
        )
        dones = torch.tensor(
            [float(transition.done) for transition in self.buf],
            dtype=torch.float32,
            device=self.device,
        )
        reward_regimes = [transition.reward_regime for transition in self.buf]
        truncations = torch.tensor(
            [float(transition.truncated) for transition in self.buf],
            dtype=torch.float32,
            device=self.device,
        )
        # Opponent-episode boundaries partition baselines and loss weights.  In
        # v25 they were stored as terminals; with v26 truncation bootstrapping
        # they are time-limit truncations (done=False) that still end a
        # segment, so segmentation uses done OR truncated.
        segment_ends = torch.maximum(dones, truncations)
        sample_weights = (
            self._regime_balanced_weights(segment_ends, reward_regimes).to(self.device)
            if self.td_value_targets
            else torch.ones_like(dones)
        )
        discounts = torch.tensor(
            [transition.discount for transition in self.buf],
            dtype=torch.float32,
            device=self.device,
        )
        durations = np.asarray(
            [transition.duration for transition in self.buf], dtype=float
        )
        if abs(1.0 - self.gamma) <= 1e-12:
            reward_exposure_values = durations
        else:
            # Compute in host float64 from the original transition discounts.
            # Using the float32 tensor here measurably perturbs even a one-step
            # exposure away from 1.0 when gamma is close to one, which then
            # biases the persistent average-reward estimate every rollout.
            reward_exposure_values = np.asarray(
                [
                    (1.0 - float(transition.discount)) / (1.0 - self.gamma)
                    for transition in self.buf
                ],
                dtype=float,
            )
        reward_exposures = torch.as_tensor(
            reward_exposure_values, dtype=torch.float32, device=self.device
        )
        old_logp = torch.stack(
            [transition.old_logp for transition in self.buf]
        ).detach()
        old_values = torch.stack(
            [transition.old_value for transition in self.buf]
        ).detach()
        exploration_rates = torch.tensor(
            [transition.exploration_rate for transition in self.buf],
            dtype=torch.float32,
            device=self.device,
        )
        masks = torch.stack(
            [transition.action_mask for transition in self.buf]
        ).detach().bool()
        next_values = torch.stack(
            [transition.next_value_override for transition in self.buf]
        ).detach()
        counterfactual_values = torch.stack(
            [transition.counterfactual_values for transition in self.buf]
        ).detach()
        counterfactual_masks = torch.stack(
            [transition.counterfactual_mask for transition in self.buf]
        ).detach().bool()
        potential_deltas = np.asarray(
            [transition.potential_delta for transition in self.buf], dtype=float
        )

        # Legacy continuing tasks use a differential reward rate. Episodic
        # tasks must retain their actual discounted reward and terminal bonus:
        # a mean estimated from this same rollout depends on its actions and
        # is not an action-independent policy-gradient baseline.
        rollout_reward_exposure = float(
            max(1e-8, np.sum(reward_exposure_values, dtype=np.float64))
        )
        _, rollout_reward_rate = self.center_smdp_reward_sums(
            raw_rewards,
            reward_exposures,
        )
        reward_rate_segments = rewards.new_tensor([rollout_reward_rate])
        reward_rate_lagged_baseline_fraction = 0.0
        if not self.center_rewards:
            reward_baselines = torch.zeros_like(rewards)
        elif self.reward_rate_beta > 0.0:
            if not self.average_reward_initialized:
                self.average_reward_rate = rollout_reward_rate
                self.average_reward_initialized = True
            else:
                beta = self.reward_rate_beta
                self.average_reward_rate = float(
                    (1.0 - beta) * self.average_reward_rate
                    + beta * rollout_reward_rate
                )
            if self.td_value_targets:
                (
                    reward_baselines,
                    reward_rate_segments,
                    reward_rate_lagged_baseline_fraction,
                ) = self._causal_regime_reward_baselines(
                    raw_rewards,
                    reward_exposures,
                    segment_ends,
                    reward_regimes,
                )
            else:
                reward_baselines = torch.full_like(
                    rewards, float(self.average_reward_rate)
                )
        else:
            # Rewards are discounted sums over possibly multi-step pricing
            # holds, so their action-independent baseline must be expressed as
            # a per-market-step rate and then multiplied by each transition's
            # exposure. Using mean transition return here mixed units and made
            # constant rewards acquire large non-zero advantages.
            reward_baselines = torch.full_like(
                rewards, float(rollout_reward_rate)
            )
        centered_rewards = rewards - reward_baselines * reward_exposures
        raw_advantages, returns = self._gae_scalar(
            centered_rewards,
            old_values,
            dones,
            bootstrap_value=bootstrap_value,
            normalize_advantage=False,
            discounts=discounts,
            truncations=truncations,
            next_value_overrides=next_values,
        )
        if self.td_value_targets:
            unresolved = (dones < 0.5) & ~torch.isfinite(next_values)
            if bool(unresolved.any().item()):
                raise ValueError(
                    "v25 TD critic received a non-terminal transition before "
                    "its actual next decision state was finalized"
                )
            resolved_next_values = torch.where(
                torch.isfinite(next_values),
                next_values,
                torch.zeros_like(next_values),
            )
            critic_targets = self._one_step_value_targets(
                centered_rewards,
                discounts,
                dones,
                resolved_next_values,
            ).to(self.device)
        else:
            critic_targets = returns
        if self.net.normalize_values:
            ratio = self.net.update_value_scale(critic_targets)
            # Adam moments follow the head's changed normalized-return units.
            for parameter in self.net.v_head.parameters():
                state = self.critic_opt.state.get(parameter, {})
                if "exp_avg" in state:
                    state["exp_avg"].mul_(ratio)
                    state["exp_avg_sq"].mul_(ratio * ratio)
        normalized_weight_sum = sample_weights.sum().clamp(min=1e-8)
        advantage_mean = (
            raw_advantages * sample_weights
        ).sum() / normalized_weight_sum
        advantage_variance = (
            sample_weights * (raw_advantages - advantage_mean).square()
        ).sum() / normalized_weight_sum
        advantage_std = torch.sqrt(advantage_variance.clamp(min=0.0))
        advantages = (
            (raw_advantages - advantage_mean) / (advantage_std + 1e-8)
            if float(advantage_std.item()) > 1e-8
            else raw_advantages - advantage_mean
        )
        advantages = advantages.clamp(-self.adv_clip, self.adv_clip).detach()

        count = states.shape[0]
        # Honor the caller's PPO batch size. The previous implementation
        # silently split every 32-transition rollout into four batches of 8,
        # so eight requested epochs became 32 actor updates on the same small,
        # noisy reaction-cycle sample. That mismatch caused dynamic-stage
        # overfitting and erased policies learned in the static stage.
        effective_batch_size = int(min(max(1, batch_size), count))

        with torch.no_grad():
            raw_logits = self._actor_logits_batch(states, actor_hiddens)
            raw_logits = self._masked_logits(
                torch.nan_to_num(
                    raw_logits, nan=0.0, posinf=20.0, neginf=-20.0
                ).clamp(-20.0, 20.0),
                masks,
            )
            raw_probs = torch.softmax(raw_logits, dim=-1)
            raw_top = torch.argmax(raw_probs, dim=-1)
            top_counts = torch.bincount(
                raw_top, minlength=self.action_dim
            ).float()
            raw_argmax_concentration = float(
                top_counts.max().item() / max(1, raw_top.numel())
            )
            raw_hold_probability = float(raw_probs[:, 0].mean().item())
            state_action_sensitivity = float(
                raw_probs.std(dim=0, unbiased=False).mean().item()
                if count > 1
                else 0.0
            )
            row_entropy = -(
                raw_probs * torch.log(raw_probs.clamp(min=1e-12))
            ).sum(dim=-1)
            feasible_entropy = torch.log(
                masks.sum(dim=-1).float().clamp(min=1.0)
            )
            raw_entropy_fraction = float(
                (row_entropy / feasible_entropy.clamp(min=1e-8))
                .masked_fill(feasible_entropy <= 1e-8, 0.0)
                .mean()
                .item()
            )

        # Maximum-entropy PPO prevents premature categorical collapse by
        # strengthening or relaxing its own entropy term according to the
        # observed policy distribution. It never replaces, masks, or forces an
        # action and therefore remains part of the learned objective.
        entropy_fraction_error = float(
            self.target_entropy_fraction - raw_entropy_fraction
        )
        self.ent_coeff = float(np.clip(
            self.ent_coeff
            * np.exp(self.entropy_adaptation_rate * entropy_fraction_error),
            self.min_ent_coeff,
            self.max_ent_coeff,
        ))

        critic_loss_sum = 0.0
        critic_grad_sum = 0.0
        critic_grad_clip_count = 0
        critic_steps = 0
        action_q_loss_sum = 0.0
        action_q_steps = 0
        counterfactual_policy_loss_sum = 0.0
        critic_parameters = self.net.critic_parameters()
        actor_epoch_count = (
            max(1, int(math.ceil(float(epochs) / 2.0)))
            if self.td_value_targets
            else max(1, int(epochs))
        )
        for _ in range(max(1, int(epochs))):
            order = torch.randperm(count, device=self.device)
            for start in range(0, count, effective_batch_size):
                batch = order[start : start + effective_batch_size]
                batch_indices = batch.detach().cpu().tolist()
                values, _ = self._critic_outputs_batch(
                    states[batch],
                    None if critic_states is None else critic_states[batch],
                    [critic_hiddens[index] for index in batch_indices],
                )
                value_loss = self._critic_value_loss(
                    values / self.net.value_scale if self.net.normalize_values else values,
                    critic_targets[batch] / self.net.value_scale if self.net.normalize_values else critic_targets[batch],
                    sample_weights[batch],
                )
                critic_loss = self.v_coeff * value_loss
                if not torch.isfinite(critic_loss):
                    continue
                self.critic_opt.zero_grad(set_to_none=True)
                critic_loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    critic_parameters, self.max_grad_norm
                )
                critic_grad_clip_count += int(
                    float(grad_norm.item()) > self.max_grad_norm
                )
                self.critic_opt.step()
                critic_steps += 1
                critic_loss_sum += float(value_loss.item())
                critic_grad_sum += float(grad_norm.item())

        if self.action_q_coeff > 0.0 and self.q_opt is not None:
            q_parameters = self.net.q_parameters()
            for _ in range(max(1, int(epochs))):
                order = torch.randperm(count, device=self.device)
                for start in range(0, count, effective_batch_size):
                    batch = order[start : start + effective_batch_size]
                    q_predictions = self.net.q_forward(
                        states[batch],
                        action_features=action_features[batch],
                        critic_state=(
                            None
                            if critic_states is None
                            else critic_states[batch]
                        ),
                    )
                    q_loss = self._counterfactual_pairwise_loss(
                        q_predictions,
                        counterfactual_values[batch],
                        counterfactual_masks[batch],
                        sample_weights[batch],
                    )
                    if not torch.isfinite(q_loss):
                        continue
                    self.q_opt.zero_grad(set_to_none=True)
                    (self.action_q_coeff * q_loss).backward()
                    nn.utils.clip_grad_norm_(q_parameters, self.max_grad_norm)
                    self.q_opt.step()
                    action_q_loss_sum += float(q_loss.item())
                    action_q_steps += 1

        if self.counterfactual_actor_weight > 0.0:
            with torch.no_grad():
                q_advantages, q_valid = self._counterfactual_actor_advantages(
                    counterfactual_values,
                    counterfactual_masks,
                    actions,
                )
                weight = self.counterfactual_actor_weight
                blended = (
                    (1.0 - weight) * advantages + weight * q_advantages
                )
                advantages = torch.where(
                    q_valid, blended, advantages
                ).clamp(-self.adv_clip, self.adv_clip).detach()

        actor_loss_sum = 0.0
        actor_grad_sum = 0.0
        actor_grad_clip_count = 0
        policy_loss_sum = 0.0
        entropy_sum = 0.0
        kl_sum = 0.0
        clipfrac_sum = 0.0
        actor_steps = 0
        stopped_early_kl = False
        actor_parameters = self.net.actor_parameters()
        for _ in range(actor_epoch_count):
            order = torch.randperm(count, device=self.device)
            for start in range(0, count, effective_batch_size):
                batch = order[start : start + effective_batch_size]
                batch_indices = batch.detach().cpu().tolist()
                logits = self._actor_logits_batch(
                    states[batch],
                    [actor_hiddens[index] for index in batch_indices],
                )
                logits = self._masked_logits(
                    torch.nan_to_num(
                        logits, nan=0.0, posinf=20.0, neginf=-20.0
                    ).clamp(-20.0, 20.0),
                    masks[batch],
                )
                # PPO ratios compare the exact behavior mixtures used to
                # collect these actions. Entropy regularization is applied to
                # the learned actor itself, not inflated by uniform coverage.
                distribution = self._exploratory_distribution(
                    logits, exploration_rates[batch], masks[batch]
                )
                policy_distribution = torch.distributions.Categorical(
                    logits=logits
                )
                new_logp = distribution.log_prob(actions[batch])
                log_ratio = new_logp - old_logp[batch]
                ratio = torch.exp(log_ratio)
                unclipped = ratio * advantages[batch]
                clipped = torch.clamp(
                    ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps
                ) * advantages[batch]
                batch_weights = sample_weights[batch]
                batch_weight_sum = batch_weights.sum().clamp(min=1e-8)
                policy_loss = -(
                    torch.min(unclipped, clipped) * batch_weights
                ).sum() / batch_weight_sum
                entropy = (
                    policy_distribution.entropy() * batch_weights
                ).sum() / batch_weight_sum
                counterfactual_policy_loss = (
                    self._observed_counterfactual_policy_loss(
                        logits,
                        counterfactual_values[batch],
                        counterfactual_masks[batch],
                        batch_weights,
                    )
                    if self.counterfactual_actor_weight > 0.0
                    else logits.sum() * 0.0
                )
                # PPO remains the primary long-horizon objective. The lower-
                # weight contextual term resolves only which measured branch
                # should receive probability, under the same pre-step KL gate.
                counterfactual_policy_coeff = (
                    0.10 * self.counterfactual_actor_weight
                )
                actor_loss = (
                    policy_loss
                    - self.ent_coeff * entropy
                    + counterfactual_policy_coeff
                    * counterfactual_policy_loss
                )
                if not torch.isfinite(actor_loss):
                    continue
                approx_kl = (
                    ((ratio - 1.0) - log_ratio) * batch_weights
                ).sum() / batch_weight_sum
                clipfrac = (
                    (ratio - 1.0).abs() > self.clip_eps
                ).float()
                clipfrac = (
                    clipfrac * batch_weights
                ).sum() / batch_weight_sum
                kl_value = float(approx_kl.item())
                # This is an early-stop gate, not a post-hoc diagnostic. Once
                # the current policy is already outside the KL trust region,
                # do not apply one additional optimizer step that pushes it
                # farther away from the behavior policy.
                if self.hard_kl_stop and kl_value > self.target_kl:
                    stopped_early_kl = True
                    break
                self.actor_opt.zero_grad(set_to_none=True)
                actor_loss.backward()
                actor_grad_norm = nn.utils.clip_grad_norm_(
                    actor_parameters, self.max_grad_norm
                )
                actor_grad_sum += float(actor_grad_norm.item())
                actor_grad_clip_count += int(
                    float(actor_grad_norm.item()) > self.max_grad_norm
                )
                self.actor_opt.step()
                actor_steps += 1
                actor_loss_sum += float(actor_loss.item())
                policy_loss_sum += float(policy_loss.item())
                entropy_sum += float(entropy.item())
                counterfactual_policy_loss_sum += float(
                    counterfactual_policy_loss.item()
                )
                kl_sum += kl_value
                clipfrac_sum += float(clipfrac.item())
            if stopped_early_kl:
                break

        with torch.no_grad():
            new_values, _ = self._critic_outputs_batch(
                states, critic_states, critic_hiddens
            )
            diagnostic_weight_sum = sample_weights.sum().clamp(min=1e-8)
            target_mean_weighted = (
                critic_targets * sample_weights
            ).sum() / diagnostic_weight_sum
            return_variance = (
                sample_weights
                * (critic_targets - target_mean_weighted).square()
            ).sum() / diagnostic_weight_sum
            explained_variance = 0.0
            if float(return_variance.item()) > 1e-8:
                residuals = critic_targets - new_values
                residual_mean = (
                    residuals * sample_weights
                ).sum() / diagnostic_weight_sum
                residual_variance = (
                    sample_weights * (residuals - residual_mean).square()
                ).sum() / diagnostic_weight_sum
                explained_variance = float(
                    torch.clamp(
                        1.0
                        - residual_variance / return_variance,
                        -1.0,
                        1.0,
                    ).item()
                )

        feasible_count = int(masks.any(dim=0).sum().item())
        action_diversity = float(
            torch.unique(actions).numel() / max(1, feasible_count)
        )
        magnitudes = (actions != 0).float()
        max_entropy = float(
            torch.log(masks.sum(dim=-1).float().clamp(min=1.0)).mean().item()
        )
        discrete_entropy = entropy_sum / max(1, actor_steps)
        self.last_policy_entropy_fraction = float(
            np.clip(discrete_entropy / max_entropy, 0.0, 1.0)
            if max_entropy > 1e-8
            else 0.0
        )
        self.last_raw_argmax_concentration = raw_argmax_concentration
        self.last_state_action_sensitivity = state_action_sensitivity
        self.last_state_action_mutual_information = 0.0

        metrics = self._empty_update_metrics()
        metrics.update(
            {
                "loss": actor_loss_sum / max(1, actor_steps),
                "policy_loss": policy_loss_sum / max(1, actor_steps),
                "value_loss": critic_loss_sum / max(1, critic_steps),
                "critic_loss": critic_loss_sum / max(1, critic_steps),
                "critic_grad_norm": critic_grad_sum / max(1, critic_steps),
                "critic_grad_clip_fraction": float(
                    critic_grad_clip_count / max(1, critic_steps)
                ),
                "actor_grad_norm": actor_grad_sum / max(1, actor_steps),
                "actor_grad_clip_fraction": float(
                    actor_grad_clip_count / max(1, actor_steps)
                ),
                "action_q_loss": action_q_loss_sum / max(1, action_q_steps),
                "counterfactual_policy_loss": (
                    counterfactual_policy_loss_sum / max(1, actor_steps)
                ),
                "lagrangian_adv_mean": float(advantage_mean.item()),
                "lagrangian_adv_std": float(advantage_std.item()),
                "entropy": discrete_entropy,
                "policy_entropy": discrete_entropy,
                "discrete_policy_entropy": discrete_entropy,
                "raw_policy_entropy_fraction": raw_entropy_fraction,
                "target_entropy_fraction": self.target_entropy_fraction,
                "entropy_fraction_error": entropy_fraction_error,
                "raw_policy_hold_probability": raw_hold_probability,
                "raw_policy_argmax_concentration": raw_argmax_concentration,
                "state_action_sensitivity": state_action_sensitivity,
                "approx_kl": kl_sum / max(1, actor_steps),
                "clipfrac": clipfrac_sum / max(1, actor_steps),
                "update_performed": bool(actor_steps and critic_steps),
                "run_mode": "training_update",
                "explained_variance": explained_variance,
                "ent_coeff": self.ent_coeff,
                "exploration_rate": float(
                    (
                        exploration_rates * sample_weights
                    ).sum().item()
                    / max(1e-8, float(sample_weights.sum().item()))
                ),
                "action_coverage": self._action_coverage(),
                "rollout_action_diversity": action_diversity,
                "rollout_reward_std": float(raw_rewards.std(unbiased=False).item()),
                "credited_reward_std": float(rewards.std(unbiased=False).item()),
                "rollout_reward_exposure": rollout_reward_exposure,
                "rollout_reward_rate_residual": float(
                    rollout_reward_rate
                    - float(
                        (
                            reward_baselines * reward_exposures
                        ).sum().item()
                        / max(1e-8, float(reward_exposures.sum().item()))
                    )
                ),
                "reward_rate_segment_count": int(
                    reward_rate_segments.numel()
                ),
                "reward_rate_segment_std": float(
                    reward_rate_segments.std(unbiased=False).item()
                    if reward_rate_segments.numel()
                    else 0.0
                ),
                "reward_rate_lagged_baseline_fraction": float(
                    reward_rate_lagged_baseline_fraction
                ),
                "opponent_segment_count": int(
                    dones.sum().item()
                    + (
                        1
                        if dones.numel()
                        and float(dones[-1].item()) <= 0.0
                        else 0
                    )
                ),
                "opponent_regime_count": int(len(set(reward_regimes))),
                "opponent_sample_weight_min": float(
                    sample_weights.min().item()
                ),
                "opponent_sample_weight_max": float(
                    sample_weights.max().item()
                ),
                "critic_target_mean": float(critic_targets.mean().item()),
                "critic_target_std": float(
                    critic_targets.std(unbiased=False).item()
                ),
                "gae_return_mean": float(returns.mean().item()),
                "critic_value_scale": float(self.net.value_scale) if self.net.normalize_values else 1.0,
                "gae_return_std": float(returns.std(unbiased=False).item()),
                "critic_target_contract": (
                    "actual_next_decision_td0"
                    if self.td_value_targets
                    else "gae_lambda_return"
                ),
                "average_reward_rate": float(
                    self.average_reward_rate
                    if self.reward_rate_beta > 0.0
                    else rollout_reward_rate
                ),
                "reward_rate_beta": self.reward_rate_beta,
                "center_rewards": self.center_rewards,
                "continuous_magnitude_mean": float(magnitudes.mean().item()),
                "continuous_magnitude_std": float(
                    magnitudes.std(unbiased=False).item()
                ),
                "learning_signal_ok": bool(actor_steps and critic_steps),
                "optimizer_steps": int(actor_steps + critic_steps),
                "actor_optimizer_steps": int(actor_steps),
                "actor_epochs_requested": int(actor_epoch_count),
                "critic_epochs_requested": int(max(1, int(epochs))),
                "critic_optimizer_steps": int(critic_steps),
                "objective_q_optimizer_steps": int(action_q_steps),
                "objective_q_update_calls": int(
                    self.q_update_calls + int(action_q_steps > 0)
                ),
                "objective_optimizer_ready": bool(action_q_steps > 0),
                "counterfactual_target_coverage": float(
                    counterfactual_masks.float().mean().item()
                ),
                "counterfactual_actor_weight": self.counterfactual_actor_weight,
                "potential_shaping_delta_mean": float(
                    potential_deltas.mean() if potential_deltas.size else 0.0
                ),
                "stopped_early_kl": stopped_early_kl,
                "rollout_size": int(count),
                "effective_batch_size": effective_batch_size,
                "policy_entropy_fraction": self.last_policy_entropy_fraction,
                "transition_duration_mean": float(durations.mean()),
                "transition_discount_mean": float(discounts.mean().item()),
                "truncation_count": int(truncations.sum().item()),
            }
        )
        self.buf.clear()
        self.update_calls += 1
        self.q_update_calls += int(action_q_steps > 0)
        # Hidden states were produced by the pre-update recurrent weights.
        # Reusing them with the updated network mixes incompatible latent
        # coordinate systems. The next action re-seeds memory from its full
        # public frame stack.
        self.reset_recurrent_state()
        return metrics

    def optimizer_state_dict(self) -> Dict[str, object]:
        state: Dict[str, object] = {
            "actor": self.actor_opt.state_dict(),
            "critic": self.critic_opt.state_dict(),
        }
        if self.q_opt is not None:
            state["q"] = self.q_opt.state_dict()
        return state

    def load_optimizer_state_dict(self, state: Optional[Dict[str, object]]) -> None:
        if not state:
            return
        if "actor" not in state or "critic" not in state:
            raise ValueError(
                "PPO optimizer state must contain separate actor and critic states"
            )
        self.actor_opt.load_state_dict(state["actor"])
        self.critic_opt.load_state_dict(state["critic"])
        if self.q_opt is not None and "q" in state:
            self.q_opt.load_state_dict(state["q"])
        self._set_learning_rates(self.curr_lr)

    def normalizer_state_dict(self) -> Dict[str, object]:
        """Persist the optional continuing-task reward-rate estimator."""
        if self.reward_rate_beta <= 0.0:
            return {}
        return {
            "average_reward_rate": float(self.average_reward_rate),
            "average_reward_initialized": bool(self.average_reward_initialized),
            "reward_rate_beta": float(self.reward_rate_beta),
            "reward_rate_by_regime": dict(self.reward_rate_by_regime),
        }

    def load_normalizer_state_dict(
        self, state: Optional[Dict[str, object]]
    ) -> None:
        if not state or self.reward_rate_beta <= 0.0:
            return
        saved_beta = float(state.get("reward_rate_beta", self.reward_rate_beta))
        if not np.isclose(saved_beta, self.reward_rate_beta):
            raise ValueError(
                "reward-rate beta does not match the configured PPO agent: "
                f"artifact={saved_beta}, runtime={self.reward_rate_beta}"
            )
        self.average_reward_rate = float(state.get("average_reward_rate", 0.0))
        self.average_reward_initialized = bool(
            state.get("average_reward_initialized", False)
        )
        saved_regimes = dict(state.get("reward_rate_by_regime", {}) or {})
        self.reward_rate_by_regime = {
            str(key): float(value)
            for key, value in saved_regimes.items()
            if np.isfinite(float(value))
        }

    def _action_coverage(self) -> float:
        if not bool(np.any(self.action_ever_feasible)):
            return 0.0
        eligible = self.action_ever_feasible
        return float(np.mean(self.action_visits[eligible] > 0))
