from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sun Feb  8 15:58:46 2026

@author: Xiaoru Shi

PPO actor-critic for hybrid discrete/continuous pricing policies.

This is designed for STABILITY:
- advantage normalization
- clipped policy and value objectives
- adaptive entropy and learning-rate controls
- gradient clipping

"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import mkl_config  # noqa: F401 - set oneMKL env before NumPy/Torch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class RunningNormalizer:
    """Numerically stable running moments for critic targets.

    PPO stores values in economic units for GAE, while the critic is trained in
    normalized units.  This keeps long-horizon profit and constraint returns
    from producing value losses that overwhelm the actor.
    """

    def __init__(self, shape: Tuple[int, ...] = (), device: Optional[torch.device] = None):
        self.device = device or torch.device("cpu")
        self.shape = tuple(shape)
        self.mean = torch.zeros(self.shape, dtype=torch.float32, device=self.device)
        self.var = torch.ones(self.shape, dtype=torch.float32, device=self.device)
        self.count = torch.tensor(1e-4, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        x = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        if x.numel() == 0:
            return
        if self.shape:
            x = x.reshape(-1, *self.shape)
        else:
            x = x.reshape(-1)
        batch_count = torch.tensor(float(x.shape[0]), dtype=torch.float32, device=self.device)
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.square() * self.count * batch_count / total
        self.mean = new_mean
        self.var = torch.clamp(m2 / total, min=1e-4)
        self.count = total

    def normalize(self, values: torch.Tensor, clip: float = 10.0) -> torch.Tensor:
        x = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        z = (x - self.mean) / torch.sqrt(self.var + 1e-6)
        return torch.clamp(z, -float(clip), float(clip))

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        return x * torch.sqrt(self.var + 1e-6) + self.mean

    def state_dict(self) -> Dict[str, object]:
        return {
            "mean": self.mean.detach().cpu(),
            "var": self.var.detach().cpu(),
            "count": float(self.count.item()),
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        if not state:
            return
        self.mean = torch.as_tensor(state["mean"], dtype=torch.float32, device=self.device).reshape(self.shape)
        self.var = torch.clamp(
            torch.as_tensor(state["var"], dtype=torch.float32, device=self.device).reshape(self.shape),
            min=1e-4,
        )
        self.count = torch.tensor(
            max(1e-4, float(state.get("count", 1e-4))),
            dtype=torch.float32,
            device=self.device,
        )


class ActorCritic(nn.Module):
    """Hierarchical recurrent actor-critic for a partially observed market game.

    The network deliberately keeps three kinds of information separate:

    * immediate public market/firm observations,
    * a recurrent belief inferred from *observable* opponent history, and
    * a continuous city/market context (never a city or opponent identity).

    Actor and critic use independent encoders so critic error cannot rewrite the
    policy representation.  The supervised response model is independent again:
    it is useful for diagnostics and representation validation, but its target
    cannot leak into PPO's policy gradient.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden: int = 192,
        constraint_dim: int = 5,
        response_dim: int = 4,
        action_feature_dim: int = 0,
        enable_action_q: bool = True,
        single_state_dim: Optional[int] = None,
        frame_stack: int = 1,
        feature_groups: Optional[Dict[str, Sequence[int]]] = None,
    ):
        super().__init__()
        self.constraint_dim = int(max(1, constraint_dim))
        self.response_dim = int(max(1, response_dim))
        self.action_dim = int(max(1, action_dim))
        self.action_feature_dim = int(max(0, action_feature_dim))
        self.enable_action_q = bool(enable_action_q and self.action_feature_dim > 0)
        self.frame_stack = int(max(1, frame_stack))
        self.single_state_dim = int(
            max(1, state_dim if single_state_dim is None else single_state_dim)
        )
        if self.single_state_dim * self.frame_stack != int(state_dim):
            raise ValueError(
                "state_dim must equal single_state_dim * frame_stack: "
                f"{state_dim} != {self.single_state_dim} * {self.frame_stack}"
            )
        groups = dict(feature_groups or {})

        def _group(name: str) -> Tuple[int, ...]:
            raw = groups.get(name, tuple(range(self.single_state_dim)))
            clean = tuple(dict.fromkeys(int(i) for i in raw))
            if not clean:
                raise ValueError(f"feature group {name!r} cannot be empty")
            if min(clean) < 0 or max(clean) >= self.single_state_dim:
                raise ValueError(
                    f"feature group {name!r} is outside [0, {self.single_state_dim})"
                )
            return clean

        immediate_idx = _group("immediate")
        opponent_idx = _group("opponent")
        city_idx = _group("city")
        self.register_buffer(
            "immediate_indices", torch.tensor(immediate_idx, dtype=torch.long)
        )
        self.register_buffer(
            "opponent_indices", torch.tensor(opponent_idx, dtype=torch.long)
        )
        self.register_buffer(
            "city_indices", torch.tensor(city_idx, dtype=torch.long)
        )

        immediate_hidden = max(64, hidden // 2)
        opponent_hidden = max(48, hidden // 3)
        city_hidden = max(32, hidden // 4)

        # Actor encoders.  The opponent GRU sees only public behavioral
        # consequences, not a strategy label or simulator-private parameters.
        self.immediate_frame_encoder = nn.Sequential(
            nn.Linear(len(immediate_idx), immediate_hidden),
            nn.LayerNorm(immediate_hidden),
            nn.SiLU(),
        )
        self.immediate_temporal_encoder = nn.GRU(
            input_size=immediate_hidden,
            hidden_size=immediate_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.opponent_frame_encoder = nn.Sequential(
            nn.Linear(len(opponent_idx), opponent_hidden),
            nn.LayerNorm(opponent_hidden),
            nn.SiLU(),
        )
        self.opponent_temporal_encoder = nn.GRU(
            input_size=opponent_hidden,
            hidden_size=opponent_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.city_encoder = nn.Sequential(
            nn.Linear(len(city_idx), city_hidden),
            nn.LayerNorm(city_hidden),
            nn.SiLU(),
            nn.Linear(city_hidden, city_hidden),
            nn.SiLU(),
        )
        fusion_dim = immediate_hidden + opponent_hidden + city_hidden
        self.actor_fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )

        # Fully separate critic encoders and fusion trunk.
        self.critic_immediate_frame_encoder = nn.Sequential(
            nn.Linear(len(immediate_idx), immediate_hidden),
            nn.LayerNorm(immediate_hidden),
            nn.SiLU(),
        )
        self.critic_immediate_temporal_encoder = nn.GRU(
            input_size=immediate_hidden,
            hidden_size=immediate_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.critic_opponent_frame_encoder = nn.Sequential(
            nn.Linear(len(opponent_idx), opponent_hidden),
            nn.LayerNorm(opponent_hidden),
            nn.SiLU(),
        )
        self.critic_opponent_temporal_encoder = nn.GRU(
            input_size=opponent_hidden,
            hidden_size=opponent_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.critic_city_encoder = nn.Sequential(
            nn.Linear(len(city_idx), city_hidden),
            nn.LayerNorm(city_hidden),
            nn.SiLU(),
            nn.Linear(city_hidden, city_hidden),
            nn.SiLU(),
        )
        self.critic_fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )

        # The response predictor has its own recurrent encoder and optimizer.
        # It can test whether public history is predictive without allowing a
        # supervised target to steer the actor toward reward-hacking actions.
        response_hidden = max(64, hidden // 2)
        self.response_frame_encoder = nn.Sequential(
            nn.Linear(self.single_state_dim, response_hidden),
            nn.LayerNorm(response_hidden),
            nn.SiLU(),
        )
        self.response_temporal_encoder = nn.GRU(
            input_size=response_hidden,
            hidden_size=response_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.response_state_encoder = nn.Sequential(
            nn.Linear(response_hidden, response_hidden),
            nn.LayerNorm(response_hidden),
            nn.SiLU(),
        )

        self.action_encoder = None
        self.critic_action_encoder = None
        self.response_action_encoder = None
        self.action_score_head = None
        self.action_q_head = None
        if self.action_feature_dim > 0:
            self.action_encoder = nn.Sequential(
                nn.Linear(self.action_feature_dim, hidden // 2),
                nn.SiLU(),
                nn.Linear(hidden // 2, hidden // 2),
                nn.SiLU(),
            )
            joint_dim = hidden + hidden // 2
            self.action_score_head = nn.Sequential(
                nn.Linear(joint_dim, hidden),
                nn.SiLU(),
                nn.Linear(hidden, 1),
            )
            if self.enable_action_q:
                self.critic_action_encoder = nn.Sequential(
                    nn.Linear(self.action_feature_dim, hidden // 2),
                    nn.SiLU(),
                    nn.Linear(hidden // 2, hidden // 2),
                    nn.SiLU(),
                )
                self.action_q_head = nn.Sequential(
                    nn.Linear(hidden + hidden // 2, hidden),
                    nn.SiLU(),
                    nn.Linear(hidden, 1),
                )
            self.response_action_encoder = nn.Sequential(
                nn.Linear(self.action_feature_dim, response_hidden // 2),
                nn.SiLU(),
                nn.Linear(response_hidden // 2, response_hidden // 2),
                nn.SiLU(),
            )
            self.response_head = nn.Sequential(
                nn.Linear(response_hidden + response_hidden // 2, response_hidden),
                nn.SiLU(),
                nn.Linear(response_hidden, self.response_dim),
            )
            self.pi_head = nn.Sequential(
                nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, action_dim)
            )
        else:
            self.pi_head = nn.Sequential(
                nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, action_dim)
            )
            self.response_head = nn.Sequential(
                nn.Linear(response_hidden, response_hidden),
                nn.SiLU(),
                nn.Linear(response_hidden, self.response_dim),
            )
        self.mag_mean_head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, action_dim))
        self.mag_logstd = nn.Parameter(torch.full((action_dim,), -0.35))
        self.v_head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.constraint_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.constraint_dim),
        )
        self.risk_head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self._init_weights()

    def actor_parameters(self) -> List[nn.Parameter]:
        modules = [
            self.immediate_frame_encoder,
            self.immediate_temporal_encoder,
            self.opponent_frame_encoder,
            self.opponent_temporal_encoder,
            self.city_encoder,
            self.actor_fusion,
            self.pi_head,
            self.mag_mean_head,
        ]
        for module in (self.action_encoder, self.action_score_head):
            if module is not None:
                modules.append(module)
        params: List[nn.Parameter] = [self.mag_logstd]
        for module in modules:
            params.extend(list(module.parameters()))
        return params

    def critic_parameters(self) -> List[nn.Parameter]:
        modules = [
            self.critic_immediate_frame_encoder,
            self.critic_immediate_temporal_encoder,
            self.critic_opponent_frame_encoder,
            self.critic_opponent_temporal_encoder,
            self.critic_city_encoder,
            self.critic_fusion,
            self.v_head,
            self.constraint_head,
            self.risk_head,
        ]
        for module in (self.critic_action_encoder, self.action_q_head):
            if module is not None:
                modules.append(module)
        params: List[nn.Parameter] = []
        for module in modules:
            params.extend(list(module.parameters()))
        return params

    def response_parameters(self) -> List[nn.Parameter]:
        modules = [
            self.response_frame_encoder,
            self.response_temporal_encoder,
            self.response_state_encoder,
            self.response_head,
        ]
        if self.response_action_encoder is not None:
            modules.append(self.response_action_encoder)
        params: List[nn.Parameter] = []
        for module in modules:
            params.extend(list(module.parameters()))
        return params

    def _init_weights(self) -> None:
        """Orthogonal init improves PPO stability and early optimization speed."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2.0))
                nn.init.constant_(module.bias, 0.0)

        if isinstance(self.pi_head[-1], nn.Linear):
            nn.init.orthogonal_(self.pi_head[-1].weight, gain=0.01)
            nn.init.constant_(self.pi_head[-1].bias, 0.0)
        if self.action_score_head is not None and isinstance(self.action_score_head[-1], nn.Linear):
            nn.init.orthogonal_(self.action_score_head[-1].weight, gain=0.01)
            nn.init.constant_(self.action_score_head[-1].bias, 0.0)
        if isinstance(self.mag_mean_head[-1], nn.Linear):
            nn.init.orthogonal_(self.mag_mean_head[-1].weight, gain=0.01)
            nn.init.constant_(self.mag_mean_head[-1].bias, 0.0)
        
        for head in (self.v_head, self.constraint_head, self.risk_head, self.response_head):
            if isinstance(head[-1], nn.Linear):
                nn.init.orthogonal_(head[-1].weight, gain=1.0)
                nn.init.constant_(head[-1].bias, 0.0)

    @staticmethod
    def _temporal_latent(
        frames: torch.Tensor,
        indices: torch.Tensor,
        frame_encoder: nn.Module,
        temporal_encoder: nn.GRU,
    ) -> torch.Tensor:
        selected = torch.index_select(frames, dim=-1, index=indices)
        encoded = frame_encoder(selected)
        temporal, _ = temporal_encoder(encoded)
        return temporal[:, -1, :]

    def _actor_latent(self, frames: torch.Tensor) -> torch.Tensor:
        immediate = self._temporal_latent(
            frames,
            self.immediate_indices,
            self.immediate_frame_encoder,
            self.immediate_temporal_encoder,
        )
        opponent = self._temporal_latent(
            frames,
            self.opponent_indices,
            self.opponent_frame_encoder,
            self.opponent_temporal_encoder,
        )
        city = self.city_encoder(
            torch.index_select(
                frames[:, -1, :], dim=-1, index=self.city_indices
            )
        )
        return self.actor_fusion(torch.cat([immediate, opponent, city], dim=-1))

    def _critic_latent(self, frames: torch.Tensor) -> torch.Tensor:
        immediate = self._temporal_latent(
            frames,
            self.immediate_indices,
            self.critic_immediate_frame_encoder,
            self.critic_immediate_temporal_encoder,
        )
        opponent = self._temporal_latent(
            frames,
            self.opponent_indices,
            self.critic_opponent_frame_encoder,
            self.critic_opponent_temporal_encoder,
        )
        city = self.critic_city_encoder(
            torch.index_select(
                frames[:, -1, :], dim=-1, index=self.city_indices
            )
        )
        return self.critic_fusion(torch.cat([immediate, opponent, city], dim=-1))

    def forward(
        self, s: torch.Tensor, action_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        frames = s.reshape(-1, self.frame_stack, self.single_state_dim)
        z = self._actor_latent(frames)
        critic_z = self._critic_latent(frames)
        response_frames = self.response_frame_encoder(frames)
        response_temporal, _ = self.response_temporal_encoder(response_frames)
        response_z = self.response_state_encoder(response_temporal[:, -1, :])
        q_values = torch.empty(0, dtype=z.dtype, device=z.device)
        if self.action_feature_dim > 0 and action_features is not None and self.action_encoder is not None:
            af = torch.nan_to_num(action_features, nan=0.0, posinf=1.0, neginf=-1.0).to(dtype=z.dtype, device=z.device)
            if af.ndim == 2:
                af = af.unsqueeze(0)
            a_emb = self.action_encoder(af)
            z_expanded = z.unsqueeze(1).expand(-1, a_emb.shape[1], -1)
            joint = torch.cat([z_expanded, a_emb], dim=-1)
            logits = self.action_score_head(joint).squeeze(-1)
            if self.critic_action_encoder is not None and self.action_q_head is not None:
                critic_a_emb = self.critic_action_encoder(af)
                critic_expanded = critic_z.unsqueeze(1).expand(
                    -1, critic_a_emb.shape[1], -1
                )
                q_values = self.action_q_head(
                    torch.cat([critic_expanded, critic_a_emb], dim=-1)
                ).squeeze(-1)
            response_a_emb = self.response_action_encoder(af)
            response_expanded = response_z.unsqueeze(1).expand(
                -1, response_a_emb.shape[1], -1
            )
            response = self.response_head(
                torch.cat([response_expanded, response_a_emb], dim=-1)
            )
        else:
            logits = self.pi_head(z)
            response = self.response_head(response_z)
        mag_mean = torch.tanh(self.mag_mean_head(z))
        mag_logstd = self.mag_logstd.clamp(-2.5, 0.75).expand_as(mag_mean)
        return (
            logits,
            mag_mean,
            mag_logstd,
            self.v_head(critic_z).squeeze(-1),
            self.constraint_head(critic_z),
            self.risk_head(critic_z).squeeze(-1),
            response,
            q_values,
        )
    
@dataclass
class Transition:
    s: torch.Tensor
    a: torch.Tensor
    r: float
    done: bool
    truncated: bool
    discount: float
    duration: int
    old_logp: torch.Tensor
    old_value: torch.Tensor
    magnitude: torch.Tensor
    exploration_rate: float
    constraint_costs: torch.Tensor
    risk_cost: float
    response_target: torch.Tensor
    action_features: torch.Tensor
    action_mask: torch.Tensor
    old_constraint_values: torch.Tensor
    old_risk_value: torch.Tensor
    next_value_override: torch.Tensor
    next_constraint_values_override: torch.Tensor
    next_risk_value_override: torch.Tensor

class PPOAgent:
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
        max_grad_norm: float = 1.0,
        hidden_dim: int = 192,
        target_kl: float = 0.03,
        adv_clip: float = 4.0,
        lr_growth_on_stall: float = 1.08,
        min_lr: float = 5e-5,
        max_lr: Optional[float] = None,
        value_clip_eps: float = 0.20,
        initial_exploration_rate: float = 0.20,
        final_exploration_rate: float = 0.02,
        exploration_fraction: float = 0.75,
        exploration_warmup_fraction: float = 0.10,
        min_action_visits: int = 8,
        constraint_dim: int = 5,
        response_dim: int = 4,
        action_feature_dim: int = 0,
        action_q_coeff: float = 0.0,
        constraint_value_coeff: float = 0.25,
        risk_value_coeff: float = 0.15,
        response_coeff: float = 0.05,
        risk_coeff: float = 0.0,
        delayed_reward_horizon: int = 6,
        delayed_reward_blend: float = 0.0,
        single_state_dim: Optional[int] = None,
        frame_stack: int = 1,
        state_action_mi_coeff: float = 0.0,
        feature_groups: Optional[Dict[str, Sequence[int]]] = None,
        device: Optional[str] = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.constraint_dim = int(max(1, constraint_dim))
        self.response_dim = int(max(1, response_dim))
        self.action_dim = int(max(1, action_dim))
        self.action_feature_dim = int(max(0, action_feature_dim))
        self.action_q_coeff = float(max(0.0, action_q_coeff))
        self.net = ActorCritic(
            state_dim,
            action_dim,
            hidden=hidden_dim,
            constraint_dim=self.constraint_dim,
            response_dim=self.response_dim,
            action_feature_dim=action_feature_dim,
            enable_action_q=self.action_q_coeff > 0.0,
            single_state_dim=single_state_dim,
            frame_stack=frame_stack,
            feature_groups=feature_groups,
        ).to(self.device)
        self.actor_opt = optim.Adam(
            self.net.actor_parameters(),
            lr=lr,
            eps=1e-5,
        )
        # A faster critic repeatedly outran the actor when the opponent changed
        # regimes, producing large value-target swings and poor late-stage
        # advantages. Keep actor and critic on the same learning-rate schedule.
        self.critic_lr_scale = 1.0
        self.critic_opt = optim.Adam(
            self.net.critic_parameters(),
            lr=float(lr) * self.critic_lr_scale,
            eps=1e-5,
        )
        self.response_opt = optim.Adam(
            self.net.response_parameters(),
            lr=float(lr),
            eps=1e-5,
        )
        self.base_lr = float(lr)
        self.curr_lr = float(lr)
        self.min_lr = float(max(1e-6, min_lr))
        self.initial_max_lr = float(max_lr if max_lr is not None else max(lr, 6e-4))
        self.max_lr = self.initial_max_lr

        self.gamma = gamma
        self.lam = lam
        self.initial_clip_eps = float(max(0.01, clip_eps))
        self.final_clip_eps = float(np.clip(final_clip_eps, 0.01, self.initial_clip_eps))
        self.clip_eps = self.initial_clip_eps
        self.v_coeff = v_coeff
        self.action_feature_dim = int(max(0, action_feature_dim))
        self.constraint_value_coeff = float(max(0.0, constraint_value_coeff))
        self.risk_value_coeff = float(max(0.0, risk_value_coeff))
        self.response_coeff = float(max(0.0, response_coeff))
        self.risk_coeff = float(max(0.0, risk_coeff))
        self.delayed_reward_horizon = int(max(1, delayed_reward_horizon))
        self.delayed_reward_blend = float(np.clip(delayed_reward_blend, 0.0, 1.0))
        self.state_action_mi_coeff = float(max(0.0, state_action_mi_coeff))
        self.constraint_lambdas = torch.zeros(self.constraint_dim, dtype=torch.float32, device=self.device)
        self.constraints_active = False
        self.risk_active = False
        self.ent_coeff = ent_coeff
        self.max_ent_coeff = float(max(ent_coeff, min_ent_coeff))
        self.min_ent_coeff = float(max(0.0, min_ent_coeff))
        self.ent_decay = float(np.clip(ent_decay, 0.90, 1.0))
        
        self.max_grad_norm = max_grad_norm
        self.update_calls = 0
        self.target_kl = float(max(1e-4, target_kl))
        self.adv_clip = float(max(1.0, adv_clip))
        self.lr_growth_on_stall = float(np.clip(lr_growth_on_stall, 1.0, 1.25))
        self.value_clip_eps = float(max(0.05, value_clip_eps))
        self.low_update_streak = 0
        
        self.initial_exploration_rate = float(np.clip(initial_exploration_rate, 0.0, 1.0))
        self.final_exploration_rate = float(
            np.clip(final_exploration_rate, 0.0, self.initial_exploration_rate)
        )
        self.exploration_fraction = float(np.clip(exploration_fraction, 1e-6, 1.0))
        self.exploration_warmup_fraction = float(
            np.clip(exploration_warmup_fraction, 0.0, self.exploration_fraction)
        )
        self.min_action_visits = int(max(0, min_action_visits))
        self.exploration_rate = self.initial_exploration_rate
        self.last_action_exploration_rate = self.exploration_rate
        self.action_visits = np.zeros(int(action_dim), dtype=np.int64)
        self.action_ever_feasible = np.zeros(int(action_dim), dtype=bool)
        self.last_policy_entropy_fraction = 1.0
        self.last_continuous_magnitude = 0.0
        self.last_raw_argmax_concentration = 0.0
        self.last_state_action_sensitivity = 0.0
        self.last_state_action_mutual_information = 0.0
        self.reward_normalizer = RunningNormalizer((), self.device)
        self.constraint_normalizer = RunningNormalizer((self.constraint_dim,), self.device)
        self.risk_normalizer = RunningNormalizer((), self.device)

        self.buf: List[Transition] = []
        self.last_constraint_values = torch.zeros(self.constraint_dim, dtype=torch.float32, device=self.device)
        self.last_risk_value = torch.zeros((), dtype=torch.float32, device=self.device)
        self.last_response_pred = torch.zeros(self.response_dim, dtype=torch.float32, device=self.device)

    @property
    def buffer_size(self) -> int:
        """Number of on-policy SMDP transitions awaiting one PPO update."""
        return len(self.buf)

    def _apply_control_floors(self) -> None:
        """Clamp scheduled controls to their declared numeric ranges."""
        self.exploration_rate = float(np.clip(
            self.exploration_rate,
            self.final_exploration_rate,
            self.initial_exploration_rate,
        ))
        self.ent_coeff = float(np.clip(
            self.ent_coeff,
            self.min_ent_coeff,
            self.max_ent_coeff,
        ))

    def _set_learning_rates(self, actor_lr: float) -> None:
        actor_lr = float(np.clip(actor_lr, self.min_lr, self.max_lr))
        self.curr_lr = actor_lr
        critic_lr = float(max(self.min_lr, actor_lr * self.critic_lr_scale))
        for group in self.actor_opt.param_groups:
            group["lr"] = actor_lr
        for group in self.critic_opt.param_groups:
            group["lr"] = critic_lr
        for group in self.response_opt.param_groups:
            group["lr"] = actor_lr
    
    def set_optimization_context(
        self,
        constraint_lambdas: Optional[np.ndarray | List[float] | Tuple[float, ...]] = None,
        risk_coeff: Optional[float] = None,
        constraints_active: Optional[bool] = None,
        risk_active: Optional[bool] = None,
    ) -> None:
        """Update Lagrangian pressure used by the structured PPO actor."""
        if constraint_lambdas is not None:
            arr = np.asarray(constraint_lambdas, dtype=np.float32).reshape(-1)
            padded = np.zeros(self.constraint_dim, dtype=np.float32)
            width = min(self.constraint_dim, arr.size)
            if width > 0:
                padded[:width] = np.maximum(arr[:width], 0.0)
            self.constraint_lambdas = self._tensor_from_value(
                padded, dtype=torch.float32, device=self.device
            )
        if risk_coeff is not None:
            self.risk_coeff = float(max(0.0, risk_coeff))
        if constraints_active is not None:
            self.constraints_active = bool(constraints_active)
        if risk_active is not None:
            self.risk_active = bool(risk_active)

    def set_stage_controls(
        self,
        *,
        exploration_rate: float,
        entropy_scale: float = 1.0,
        learning_rate_scale: float = 1.0,
    ) -> None:
        """Apply curriculum controls without changing the learned policy.

        This is deliberately separate from action preferences: a later stage may
        reduce random exploration and optimizer step size, but it never adds a
        prior for the hold action (or for any intervention action).
        """
        self.exploration_rate = float(
            np.clip(exploration_rate, self.final_exploration_rate, self.initial_exploration_rate)
        )
        entropy_scale_f = float(max(0.0, entropy_scale))
        self.ent_coeff = float(
            np.clip(self.max_ent_coeff * entropy_scale_f, self.min_ent_coeff, self.max_ent_coeff)
        )
        lr = float(np.clip(
            self.base_lr * max(0.0, learning_rate_scale),
            self.min_lr,
            self.max_lr,
        ))
        self._set_learning_rates(lr)
        self._apply_control_floors()
            
    def adapt_exploration(
        self,
        progress: float,
        reward_converged: bool,
        reward_std: Optional[float] = None,
    ) -> None:
        """Use a smooth, predetermined exploration schedule.

        The schedule is independent of short reward windows, so late reward
        noise cannot repeatedly reopen or close exploration.
        """
        del reward_std
        p = float(np.clip(progress, 0.0, 1.0))
        covered = bool(
            self.min_action_visits <= 0
            or self._action_coverage() >= 1.0
        )
        if p <= self.exploration_warmup_fraction or not covered:
            rate = self.initial_exploration_rate
        else:
            decay_width = max(1e-6, self.exploration_fraction - self.exploration_warmup_fraction)
            decay_progress = float(
                np.clip((p - self.exploration_warmup_fraction) / decay_width, 0.0, 1.0)
            )
            cosine = 0.5 * (1.0 + np.cos(np.pi * decay_progress))
            rate = self.final_exploration_rate + (
                self.initial_exploration_rate - self.final_exploration_rate
            ) * cosine
        if reward_converged:
            rate = self.final_exploration_rate
        self.exploration_rate = float(
            np.clip(rate, self.final_exploration_rate, self.initial_exploration_rate)
        )
        self._apply_control_floors()

    @staticmethod
    def _tensor_from_value(
        value,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Create a tensor without requiring Torch's NumPy C-API bridge.

        Some supported environments pair a Torch wheel compiled against NumPy
        1.x with NumPy 2.x. Converting ndarray inputs to native containers first
        keeps inference, Kaggle evaluation, and tests functional in that setup
        without changing numeric values.
        """
        if isinstance(value, np.ndarray):
            value = value.tolist()
        elif isinstance(value, np.generic):
            value = value.item()
        elif torch.is_tensor(value):
            return value.detach().to(device=device, dtype=dtype).clone()
        return torch.tensor(value, dtype=dtype, device=device)

    @staticmethod
    def _coerce_action_mask(
        action_mask: Optional[np.ndarray | List[bool] | torch.Tensor],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """Return a broadcastable mask with at least the hold action feasible."""
        if action_mask is None:
            mask = torch.ones_like(logits, dtype=torch.bool)
        else:
            mask = PPOAgent._tensor_from_value(
                action_mask,
                dtype=torch.bool,
                device=logits.device,
            )
            while mask.ndim < logits.ndim:
                mask = mask.unsqueeze(0)
            try:
                mask = torch.broadcast_to(mask, logits.shape).clone()
            except RuntimeError as exc:
                raise ValueError(
                    f"Action mask shape {tuple(mask.shape)} is incompatible with logits {tuple(logits.shape)}"
                ) from exc
        flat = mask.reshape(-1, mask.shape[-1])
        empty_rows = ~flat.any(dim=-1)
        if bool(empty_rows.any().item()):
            flat[empty_rows, 0] = True
        return flat.reshape(mask.shape)

    @classmethod
    def _masked_logits(
        cls,
        logits: torch.Tensor,
        action_mask: Optional[np.ndarray | List[bool] | torch.Tensor],
    ) -> torch.Tensor:
        mask = cls._coerce_action_mask(action_mask, logits)
        return logits.masked_fill(~mask, -1e9)

    @staticmethod
    def _economic_group_probabilities(
        logits: torch.Tensor,
        action_features: Optional[torch.Tensor],
        temperature: float = 0.20,
    ) -> Optional[torch.Tensor]:
        """Return sharpened lower/neutral/higher fare probabilities.

        Mutual information computed from the ordinary softmax can be increased
        through tiny changes in low-ranked action tails while deterministic
        argmax remains identical in every state. Sharpening only the
        specialization diagnostic/loss makes its gradient reflect economically
        different deterministic choices; it does not change the PPO sampling
        or deployment distribution.
        """
        if (
            action_features is None
            or action_features.ndim != 3
            or action_features.shape[-1] < 14
        ):
            return None
        specialization_probs = torch.softmax(
            logits / max(1e-3, float(temperature)),
            dim=-1,
        )
        # Match Core._economic_action_group exactly: the segment weights
        # represent the broad-market mix and imply about 14.65 minutes,
        # 3.95 miles, and 7.75% airport exposure. A simple unweighted segment
        # average classified mixed rebalancing bundles differently during
        # training and validation, allowing within-group switching to satisfy
        # the optimizer while deployment still looked economically constant.
        segment_weights = torch.as_tensor(
            [0.35, 0.35, 0.20, 0.10],
            dtype=action_features.dtype,
            device=action_features.device,
        )
        fare_impact = (
            action_features[:, :, 10:14] * segment_weights
        ).sum(dim=-1)
        # Action-feature fare impacts are stored in ``dollars / 20`` by the
        # observation model. Core's deployment audit uses a five-cent
        # broad-market threshold in dollars, so the matching threshold here is
        # 0.05 / 20.0. Applying 0.05 directly made virtually every single-
        # lever intervention look neutral to the optimizer even though the
        # exact same action was lower/higher fare in the deployment audit.
        normalized_fare_threshold = 0.05 / 20.0
        lower_mask = (
            fare_impact < -normalized_fare_threshold
        ).to(specialization_probs.dtype)
        higher_mask = (
            fare_impact > normalized_fare_threshold
        ).to(specialization_probs.dtype)
        neutral_mask = 1.0 - torch.clamp(
            lower_mask + higher_mask, 0.0, 1.0
        )
        economic_probs = torch.stack(
            [
                (specialization_probs * lower_mask).sum(dim=-1),
                (specialization_probs * neutral_mask).sum(dim=-1),
                (specialization_probs * higher_mask).sum(dim=-1),
            ],
            dim=-1,
        )
        return economic_probs / economic_probs.sum(
            dim=-1,
            keepdim=True,
        ).clamp(min=1e-8)

    @staticmethod
    def _mutual_information(
        probabilities: torch.Tensor,
        state_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Mutual information between sampled states and policy choices.

        ``state_weights`` is detached economic credit, not another policy
        target.  With positive-advantage weights, an unprofitable state cannot
        earn the actor a specialization bonus merely by selecting a different
        price direction.  If a minibatch has no positive advantages, the
        auxiliary is disabled for that minibatch and PPO remains the sole
        actor objective.
        """
        if probabilities.ndim != 2 or probabilities.shape[0] == 0:
            return probabilities.new_zeros(())
        if state_weights is None:
            weights = probabilities.new_full(
                (probabilities.shape[0],),
                1.0 / float(probabilities.shape[0]),
            )
        else:
            weights = torch.clamp(
                state_weights.detach().to(
                    dtype=probabilities.dtype,
                    device=probabilities.device,
                ),
                min=0.0,
            ).reshape(-1)
            if weights.shape[0] != probabilities.shape[0]:
                raise ValueError("state_weights must match probability rows")
            weight_sum = weights.sum()
            if float(weight_sum.item()) <= 1e-8:
                return probabilities.new_zeros(())
            weights = weights / weight_sum
        row_entropy = -(
            probabilities
            * torch.log(probabilities.clamp(min=1e-8))
        ).sum(dim=-1)
        marginal = (probabilities * weights.unsqueeze(-1)).sum(dim=0)
        marginal_entropy = -(
            marginal * torch.log(marginal.clamp(min=1e-8))
        ).sum()
        conditional_entropy = (row_entropy * weights).sum()
        return torch.clamp(
            marginal_entropy - conditional_entropy,
            min=0.0,
        )

    @classmethod
    def _exploratory_distribution(
        cls,
        logits: torch.Tensor,
        exploration_rate: torch.Tensor | float,
        action_mask: Optional[np.ndarray | List[bool] | torch.Tensor] = None,
    ) -> torch.distributions.Categorical:
        mask = cls._coerce_action_mask(action_mask, logits)
        logits = cls._masked_logits(logits, mask)
        policy_probs = torch.softmax(logits, dim=-1)
        eps = torch.as_tensor(
            exploration_rate, dtype=policy_probs.dtype, device=policy_probs.device
        )
        if eps.ndim == 0:
            eps = eps.expand(policy_probs.shape[:-1])
        eps = eps.unsqueeze(-1)
        feasible = mask.to(dtype=policy_probs.dtype)
        uniform_feasible = feasible / feasible.sum(dim=-1, keepdim=True).clamp(min=1.0)
        mixed_probs = (1.0 - eps) * policy_probs + eps * uniform_feasible
        return torch.distributions.Categorical(probs=mixed_probs)
    
    @staticmethod
    def _magnitude_dist(mean: torch.Tensor, logstd: torch.Tensor) -> torch.distributions.Normal:
        return torch.distributions.Normal(mean, torch.exp(logstd))

    @staticmethod
    def _squash_magnitude(raw: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(raw) * 2.0

    @staticmethod
    def _unsquash_magnitude(magnitude: torch.Tensor) -> torch.Tensor:
        y = torch.clamp(magnitude / 2.0, 1e-6, 1.0 - 1e-6)
        return torch.log(y) - torch.log1p(-y)

    @classmethod
    def _magnitude_log_prob(cls, dist: torch.distributions.Normal, magnitude: torch.Tensor) -> torch.Tensor:
        raw = cls._unsquash_magnitude(magnitude)
        y = torch.clamp(magnitude / 2.0, 1e-6, 1.0 - 1e-6)
        log_abs_det = torch.log(torch.clamp(2.0 * y * (1.0 - y), min=1e-6))
        return dist.log_prob(raw) - log_abs_det
    
    def adapt_entropy(self, progress: float, reward_converged: bool) -> None:
        """Smoothly anneal entropy without reward-triggered recovery floors."""
        p = float(np.clip(progress, 0.0, 1.0))
        target = self.min_ent_coeff + (
            self.max_ent_coeff - self.min_ent_coeff
        ) * max(0.0, 1.0 - p) ** 1.35
        if reward_converged:
            target = max(self.min_ent_coeff, 0.8 * target)
        self.ent_coeff = float(
            np.clip(target, self.min_ent_coeff, self.max_ent_coeff)
        )
        self._apply_control_floors()
    
    def adapt_optimization(self, progress: float, reward_converged: bool) -> None:
        """Apply an exploration-first schedule for PPO updates.

        Early training keeps the learning-rate cap and clip range wide so the
        policy can move out of weak initial logits.  Later training decays both
        controls for consolidation.  KL/clip-based adaptation in ``update`` can
        still lower the rate further, but cannot grow beyond the scheduled cap.
        """
        p = float(np.clip(progress, 0.0, 1.0))
        # Keep early optimization deliberately less conservative.  The cap stays
        # at the initial maximum through the first third of training, then uses a
        # cosine decay for consolidation.
        decay_p = float(np.clip((p - 0.70) / 0.30, 0.0, 1.0))
        cosine = 0.5 * (1.0 + np.cos(np.pi * decay_p))
        scheduled_lr_cap = self.min_lr + (self.initial_max_lr - self.min_lr) * cosine
        if reward_converged:
            scheduled_lr_cap = min(scheduled_lr_cap, self.min_lr * 1.5)
        self.max_lr = float(np.clip(scheduled_lr_cap, self.min_lr, self.initial_max_lr))
        if self.curr_lr > self.max_lr:
            self._set_learning_rates(self.max_lr)
        
        early_clip = self.initial_clip_eps
        mid_clip = self.final_clip_eps
        self.clip_eps = float(
            mid_clip + (early_clip - mid_clip) * cosine
        )
        if reward_converged:
            self.clip_eps = min(self.clip_eps, self.final_clip_eps)
    
    @torch.no_grad()
    def policy_diagnostics(
        self,
        s_np: np.ndarray,
        action_features: Optional[np.ndarray] = None,
        action_mask: Optional[np.ndarray] = None,
        temperature: float = 1.0,
    ) -> Dict[str, float]:
        """Return raw-policy action diagnostics without the training exploration mix."""
        s = self._tensor_from_value(
            s_np, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        s = torch.nan_to_num(s, nan=0.0, posinf=1e3, neginf=-1e3)
        af_tensor = None
        if self.action_feature_dim > 0 and action_features is not None:
            af_tensor = self._tensor_from_value(
                action_features, dtype=torch.float32, device=self.device
            )
            af_tensor = af_tensor.reshape(1, af_tensor.shape[-2], af_tensor.shape[-1])
        logits, mag_mean, mag_logstd, value, constraint_values, risk_value, response_pred, _ = self.net(s, af_tensor)
        del mag_mean, mag_logstd, value, constraint_values, risk_value, response_pred
        logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0).reshape(-1)
        mask_t = self._coerce_action_mask(action_mask, logits)
        logits = self._masked_logits(logits, mask_t)
        temp = float(max(1e-6, temperature))
        probs_t = torch.softmax(logits / temp, dim=-1)
        probs = np.asarray(probs_t.detach().cpu().tolist(), dtype=float)
        if probs.size == 0:
            return {
                "policy_top_action": -1.0,
                "policy_second_action": -1.0,
                "policy_top_prob": 0.0,
                "policy_second_prob": 0.0,
                "policy_action_margin": 0.0,
                "policy_hold_prob": 0.0,
                "policy_entropy": 0.0,
                "policy_temperature": temp,
            }
        order = np.argsort(-probs)
        top = int(order[0])
        second = int(order[1]) if probs.size > 1 else top
        safe_probs = np.clip(probs, 1e-12, 1.0)
        return {
            "policy_top_action": float(top),
            "policy_second_action": float(second),
            "policy_top_prob": float(probs[top]),
            "policy_second_prob": float(probs[second]) if second != top else 0.0,
            "policy_action_margin": float(probs[top] - (probs[second] if second != top else 0.0)),
            "policy_hold_prob": float(probs[0]) if probs.size > 0 else 0.0,
            "policy_entropy": float(-np.sum(safe_probs * np.log(safe_probs))),
            "policy_temperature": temp,
        }
    
    @torch.no_grad()
    def act(
        self,
        s_np: np.ndarray,
        deterministic: bool = False,
        action_features: Optional[np.ndarray] = None,
        action_mask: Optional[np.ndarray] = None,
        policy_mode: str = "argmax",
        policy_temperature: float = 0.50,
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s = self._tensor_from_value(
            s_np, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        s = torch.nan_to_num(s, nan=0.0, posinf=1e3, neginf=-1e3)
        expected_dim = self.net.single_state_dim * self.net.frame_stack
        if s.shape[-1] != expected_dim:
            raise ValueError(f"State dim mismatch: got {s.shape[-1]}, expected {expected_dim}")
        
        af_tensor = None
        if self.action_feature_dim > 0 and action_features is not None:
            af_tensor = self._tensor_from_value(
                action_features, dtype=torch.float32, device=self.device
            )
            af_tensor = af_tensor.reshape(1, af_tensor.shape[-2], af_tensor.shape[-1])
        logits, mag_mean, mag_logstd, value, constraint_values, risk_value, response_pred, _ = self.net(s, af_tensor)
        value = self.reward_normalizer.denormalize(value)
        constraint_values = self.constraint_normalizer.denormalize(constraint_values)
        risk_value = self.risk_normalizer.denormalize(risk_value)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
        if not torch.isfinite(logits).all():
            logits = torch.zeros_like(logits)
        mask_t = self._coerce_action_mask(action_mask, logits)
        feasible_now = np.asarray(mask_t.squeeze(0).detach().cpu().tolist(), dtype=bool)
        self.action_ever_feasible[: feasible_now.size] |= feasible_now[: self.action_ever_feasible.size]
        logits = self._masked_logits(logits, mask_t)
        exploration_rate = 0.0 if deterministic else self.exploration_rate
        self.last_action_exploration_rate = float(exploration_rate)
        dist = self._exploratory_distribution(logits, exploration_rate, mask_t)
        if deterministic:
            mode = str(policy_mode or "argmax").strip().lower()
            raw_dist = torch.distributions.Categorical(logits=logits)
            if mode == "sample_raw":
                a = raw_dist.sample()
            elif mode == "sample_low_temp":
                temp = float(max(1e-6, policy_temperature))
                a = torch.distributions.Categorical(logits=logits / temp).sample()
            elif mode == "argmax":
                a = torch.argmax(logits, dim=-1)
            else:
                raise ValueError(
                    "policy_mode must be one of: argmax, sample_raw, sample_low_temp"
                )
        else:
            a = dist.sample()
        chosen_mean = mag_mean.gather(1, a.reshape(-1, 1)).squeeze(1)
        chosen_logstd = mag_logstd.gather(1, a.reshape(-1, 1)).squeeze(1)
        mag_dist = self._magnitude_dist(chosen_mean, chosen_logstd)
        if deterministic:
            magnitude = self._squash_magnitude(chosen_mean)
        else:
            magnitude = self._squash_magnitude(mag_dist.rsample())
        magnitude = torch.where(a == 0, torch.zeros_like(magnitude), magnitude)
        logp = dist.log_prob(a) + torch.where(a == 0, torch.zeros_like(magnitude), self._magnitude_log_prob(mag_dist, magnitude))
        self.last_continuous_magnitude = float(magnitude.squeeze(0).detach().cpu().item())
        self.last_constraint_values = torch.nan_to_num(
            constraint_values.squeeze(0), nan=0.0, posinf=0.0, neginf=0.0
        ).detach()
        self.last_risk_value = torch.nan_to_num(
            risk_value.squeeze(0), nan=0.0, posinf=0.0, neginf=0.0
        ).detach()
        if response_pred.ndim == 3:
            chosen_response = response_pred.gather(
                1,
                a.reshape(-1, 1, 1).expand(-1, 1, self.response_dim),
            ).squeeze(1)
        else:
            chosen_response = response_pred
        self.last_response_pred = torch.nan_to_num(
            chosen_response.squeeze(0), nan=0.0, posinf=0.0, neginf=0.0
        ).detach()
        if not deterministic:
            self.action_visits[int(a.item())] += 1
        af_out = af_tensor.squeeze(0).detach() if af_tensor is not None else torch.empty(0, dtype=torch.float32, device=self.device)
        mask_out = mask_t.squeeze(0).detach()
        return int(a.item()), s.squeeze(0), logp.squeeze(0), value.squeeze(0), af_out, mask_out

    @torch.no_grad()
    def value_estimates(
        self,
        s_np: np.ndarray,
        action_features: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate all critics for bootstrapping a truncated rollout."""
        s = self._tensor_from_value(
            s_np, dtype=torch.float32, device=self.device
        ).reshape(1, -1)
        s = torch.nan_to_num(s, nan=0.0, posinf=1e3, neginf=-1e3)
        af_tensor = None
        if self.action_feature_dim > 0 and action_features is not None:
            af_tensor = self._tensor_from_value(
                action_features, dtype=torch.float32, device=self.device
            )
            af_tensor = af_tensor.reshape(1, af_tensor.shape[-2], af_tensor.shape[-1])
        _, _, _, value, constraint_values, risk_value, _, _ = self.net(s, af_tensor)
        value = self.reward_normalizer.denormalize(value)
        constraint_values = self.constraint_normalizer.denormalize(constraint_values)
        risk_value = self.risk_normalizer.denormalize(risk_value)
        return (
            torch.nan_to_num(value.squeeze(0), nan=0.0, posinf=0.0, neginf=0.0).detach(),
            torch.nan_to_num(constraint_values.squeeze(0), nan=0.0, posinf=0.0, neginf=0.0).detach(),
            torch.nan_to_num(risk_value.squeeze(0), nan=0.0, posinf=0.0, neginf=0.0).detach(),
        )
    
    def store(
        self,
        s: torch.Tensor,
        a: int,
        r: float,
        done: bool,
        s_next: Optional[torch.Tensor],
        old_logp: torch.Tensor,
        old_value: torch.Tensor,
        constraint_costs: Optional[np.ndarray | List[float] | Tuple[float, ...] | torch.Tensor] = None,
        risk_cost: float = 0.0,
        response_target: Optional[np.ndarray | List[float] | Tuple[float, ...] | torch.Tensor] = None,
        action_features: Optional[np.ndarray | List[List[float]] | torch.Tensor] = None,
        action_mask: Optional[np.ndarray | List[bool] | torch.Tensor] = None,
        discount: Optional[float] = None,
        duration: int = 1,
        truncated: bool = False,
        next_value: Optional[torch.Tensor | float] = None,
        next_constraint_values: Optional[torch.Tensor | np.ndarray] = None,
        next_risk_value: Optional[torch.Tensor | float] = None,
    ) -> None:
        del s_next
        if constraint_costs is None:
            constraint_tensor = torch.zeros(self.constraint_dim, dtype=torch.float32, device=self.device)
        else:
            constraint_tensor = self._tensor_from_value(
                constraint_costs, dtype=torch.float32, device=self.device
            ).reshape(-1)
            if constraint_tensor.numel() != self.constraint_dim:
                padded = torch.zeros(self.constraint_dim, dtype=torch.float32, device=self.device)
                width = min(self.constraint_dim, constraint_tensor.numel())
                if width > 0:
                    padded[:width] = constraint_tensor[:width]
                constraint_tensor = padded
        if response_target is None:
            response_tensor = self.last_response_pred.detach().clone()
        else:
            response_tensor = self._tensor_from_value(
                response_target, dtype=torch.float32, device=self.device
            ).reshape(-1)
            if response_tensor.numel() != self.response_dim:
                padded = torch.zeros(self.response_dim, dtype=torch.float32, device=self.device)
                width = min(self.response_dim, response_tensor.numel())
                if width > 0:
                    padded[:width] = response_tensor[:width]
                response_tensor = padded
        
        if action_features is None or self.action_feature_dim <= 0:
            action_feature_tensor = torch.empty(0, dtype=torch.float32, device=self.device)
        else:
            action_feature_tensor = self._tensor_from_value(
                action_features, dtype=torch.float32, device=self.device
            )
            action_feature_tensor = action_feature_tensor.reshape(-1, action_feature_tensor.shape[-1])
            if action_feature_tensor.shape[-1] != self.action_feature_dim:
                fixed = torch.zeros((self.action_visits.size, self.action_feature_dim), dtype=torch.float32, device=self.device)
                rows = min(fixed.shape[0], action_feature_tensor.shape[0])
                cols = min(fixed.shape[1], action_feature_tensor.shape[1])
                fixed[:rows, :cols] = action_feature_tensor[:rows, :cols]
                action_feature_tensor = fixed
    
        constraint_tensor = torch.nan_to_num(constraint_tensor, nan=0.0, posinf=1.0, neginf=0.0).clamp(min=0.0)
        response_tensor = torch.nan_to_num(response_tensor, nan=0.0, posinf=1.0, neginf=-1.0)
        
        action_feature_tensor = torch.nan_to_num(action_feature_tensor, nan=0.0, posinf=1.0, neginf=-1.0)
        action_mask_tensor = self._coerce_action_mask(action_mask, torch.zeros(self.action_dim, device=self.device)).reshape(-1)
        transition_duration = int(max(1, duration))
        transition_discount = float(
            self.gamma ** transition_duration if discount is None else discount
        )
        if not np.isfinite(transition_discount) or transition_discount < 0.0 or transition_discount > 1.0:
            raise ValueError(f"transition discount must be finite and in [0, 1], got {transition_discount}")
        next_value_tensor = torch.as_tensor(
            float("nan") if next_value is None else next_value,
            dtype=torch.float32,
            device=self.device,
        ).reshape(())
        if next_constraint_values is None:
            next_constraint_tensor = torch.full(
                (self.constraint_dim,), float("nan"), dtype=torch.float32, device=self.device
            )
        else:
            next_constraint_tensor = self._tensor_from_value(
                next_constraint_values,
                dtype=torch.float32,
                device=self.device,
            ).reshape(-1)
            if next_constraint_tensor.numel() != self.constraint_dim:
                raise ValueError(
                    "next constraint value override width "
                    f"{next_constraint_tensor.numel()} != {self.constraint_dim}"
                )
        next_risk_tensor = torch.as_tensor(
            float("nan") if next_risk_value is None else next_risk_value,
            dtype=torch.float32,
            device=self.device,
        ).reshape(())
        
        self.buf.append(
            Transition(
                s=s.detach(),
                a=torch.tensor(a, dtype=torch.long, device=self.device),
                r=float(r),
                done=bool(done),
                truncated=bool(truncated),
                discount=transition_discount,
                duration=transition_duration,
                old_logp=old_logp.detach(),
                old_value=old_value.detach(),
                magnitude=torch.tensor(float(getattr(self, "last_continuous_magnitude", 0.0)), dtype=torch.float32, device=self.device),
                exploration_rate=float(
                    np.clip(getattr(self, "last_action_exploration_rate", 0.0), 0.0, 1.0)
                ),
                constraint_costs=constraint_tensor.detach(),
                risk_cost=float(max(0.0, np.nan_to_num(risk_cost, nan=0.0, posinf=1.0, neginf=0.0))),
                response_target=response_tensor.detach(),
                action_features=action_feature_tensor.detach(),
                action_mask=action_mask_tensor.detach(),
                old_constraint_values=self.last_constraint_values.detach().clone(),
                old_risk_value=self.last_risk_value.detach().clone(),
                next_value_override=next_value_tensor.detach(),
                next_constraint_values_override=next_constraint_tensor.detach(),
                next_risk_value_override=next_risk_tensor.detach(),
            )
        )
    
    @torch.no_grad()
    def _delayed_reward_credit(self, rewards: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """Blend immediate rewards with bounded future-return credit.

        Price manipulations can affect share and revenue several decision points
        after the action is applied.  This light-weight reward redistribution
        keeps the original immediate signal but adds a normalized, finite-horizon
        future component before GAE, improving credit assignment without changing
        PPO's on-policy objective.
        """
        if self.delayed_reward_blend <= 0.0 or rewards.numel() <= 1:
            return rewards
        t_size = rewards.numel()
        future = torch.zeros_like(rewards)
        for t in range(t_size):
            acc = torch.zeros((), dtype=torch.float32, device=self.device)
            discount = 1.0
            for k in range(1, self.delayed_reward_horizon + 1):
                idx = t + k
                if idx >= t_size:
                    break
                if bool(dones[idx - 1].item()):
                    break
                discount *= float(self.gamma)
                acc = acc + discount * rewards[idx]
            future[t] = acc
        if rewards.numel() > 1:
            scale = torch.std(future, unbiased=False) + 1e-8
            centered_future = (future - torch.mean(future)) / scale
            reward_scale = torch.std(rewards, unbiased=False) + 1e-8
            future = centered_future * reward_scale
        blended = (1.0 - self.delayed_reward_blend) * rewards + self.delayed_reward_blend * future
        return torch.nan_to_num(blended, nan=0.0, posinf=0.0, neginf=0.0)

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
        """Semi-MDP generalized advantage estimation.

        A stored transition may represent several operational timesteps under
        one held tariff. ``discounts[t]`` is therefore :math:`gamma^tau`, not
        necessarily one-step gamma. Time-limit truncations bootstrap from the
        final observation but stop the GAE trace so it cannot leak into the
        next curriculum episode.
        """
        t_size = rewards.numel()
        adv = torch.zeros(t_size, dtype=torch.float32, device=self.device)
        ret = torch.zeros(t_size, dtype=torch.float32, device=self.device)

        gae = torch.zeros((), dtype=torch.float32, device=self.device)
        rollout_bootstrap = torch.as_tensor(
            0.0 if bootstrap_value is None else bootstrap_value,
            dtype=torch.float32,
            device=self.device,
        ).reshape(())
        if discounts is None:
            discounts = torch.full(
                (t_size,), float(self.gamma), dtype=torch.float32, device=self.device
            )
        else:
            discounts = torch.as_tensor(discounts, dtype=torch.float32, device=self.device).reshape(-1)
        if truncations is None:
            truncations = torch.zeros(t_size, dtype=torch.float32, device=self.device)
        else:
            truncations = torch.as_tensor(
                truncations, dtype=torch.float32, device=self.device
            ).reshape(-1)
        if next_value_overrides is None:
            next_value_overrides = torch.full(
                (t_size,), float("nan"), dtype=torch.float32, device=self.device
            )
        else:
            next_value_overrides = torch.as_tensor(
                next_value_overrides, dtype=torch.float32, device=self.device
            ).reshape(-1)
        if not (
            discounts.numel() == truncations.numel() == next_value_overrides.numel() == t_size
        ):
            raise ValueError("semi-MDP GAE metadata must match reward length")
        for t in reversed(range(t_size)):
            terminal = float(dones[t].item())
            truncated = float(truncations[t].item())
            if torch.isfinite(next_value_overrides[t]):
                next_v = next_value_overrides[t]
            elif t == t_size - 1:
                next_v = rollout_bootstrap
            else:
                next_v = old_values[t + 1]
            step_discount = discounts[t]
            bootstrap_continuation = 1.0 - terminal
            trace_continuation = max(0.0, 1.0 - terminal - truncated)
            delta = (
                rewards[t]
                + step_discount * next_v * bootstrap_continuation
                - old_values[t]
            )
            gae = (
                delta
                + step_discount * self.lam * trace_continuation * gae
            )
            adv[t] = gae
            ret[t] = adv[t] + old_values[t]

        if normalize_advantage:
            adv_mean = adv.mean()
            adv_std = adv.std(unbiased=False)
            if torch.isfinite(adv_std) and float(adv_std.item()) > 1e-8:
                adv = (adv - adv_mean) / (adv_std + 1e-8)
            else:
                adv = adv - adv_mean

        adv = torch.nan_to_num(adv, nan=0.0, posinf=0.0, neginf=0.0)
        ret = torch.nan_to_num(ret, nan=0.0, posinf=0.0, neginf=0.0)
        return adv, ret

    @torch.no_grad()
    def _update_popart(
        self,
        normalizer: RunningNormalizer,
        targets: torch.Tensor,
        heads: List[nn.Linear],
    ) -> None:
        """Update target moments without changing denormalized predictions."""
        old_mean = normalizer.mean.detach().clone()
        old_std = torch.sqrt(normalizer.var.detach().clone() + 1e-6)
        normalizer.update(targets)
        new_mean = normalizer.mean.detach()
        new_std = torch.sqrt(normalizer.var.detach() + 1e-6)
        scale = (old_std / new_std).reshape(-1)
        shift = ((old_mean - new_mean) / new_std).reshape(-1)
        for head in heads:
            if head.out_features != scale.numel():
                raise ValueError(
                    f"PopArt head width {head.out_features} != normalizer width {scale.numel()}"
                )
            head.weight.mul_(scale.reshape(-1, 1))
            head.bias.mul_(scale).add_(shift)

    def update(
        self,
        epochs: int = 5,
        batch_size: int = 256,
        bootstrap_value: Optional[torch.Tensor | float] = None,
        bootstrap_constraint_values: Optional[torch.Tensor | np.ndarray] = None,
        bootstrap_risk_value: Optional[torch.Tensor | float] = None,
    ) -> dict:
        if not self.buf:
            return {
                "loss": 0.0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "constraint_value_loss": 0.0,
                "risk_value_loss": 0.0,
                "response_loss": 0.0,
                "action_q_loss": 0.0,
                "lagrangian_adv_mean": 0.0,
                "lagrangian_adv_std": 0.0,
                "constraint_lambda_mean": 0.0,
                "risk_coeff": float(self.risk_coeff),
                "entropy": 0.0,
                "policy_entropy": 0.0,
                "policy_entropy_fraction": float(self.last_policy_entropy_fraction),
                "ent_coeff": float(self.ent_coeff),
                "clip_eps": float(self.clip_eps),
                "lr": float(self.curr_lr),
                "exploration_rate": float(self.exploration_rate),
                "action_coverage": self._action_coverage(),
                "target_kl": float(self.target_kl),
                "update_performed": False,
                "run_mode": "no_rollout_data",
                "optimizer_steps": 0,
                "actor_optimizer_steps": 0,
                "critic_optimizer_steps": 0,
                "response_optimizer_steps": 0,
            }

        reward_all = torch.tensor([tr.r for tr in self.buf], dtype=torch.float32, device=self.device)
        done_all = torch.tensor([float(tr.done) for tr in self.buf], dtype=torch.float32, device=self.device)
        truncated_all = torch.tensor(
            [float(tr.truncated) for tr in self.buf], dtype=torch.float32, device=self.device
        )
        discount_all = torch.tensor(
            [tr.discount for tr in self.buf], dtype=torch.float32, device=self.device
        )
        next_reward_values_all = torch.stack(
            [tr.next_value_override for tr in self.buf], dim=0
        ).detach()
        # Each transition already spans the full tariff hold interval and stores
        # its exact discounted economic reward. A second redistribution pass
        # would attribute the same future profit twice.
        credited_reward_all = reward_all
        old_reward_values_all = torch.stack([tr.old_value for tr in self.buf], dim=0).detach()
        old_reward_values_all = torch.nan_to_num(old_reward_values_all, nan=0.0, posinf=0.0, neginf=0.0)
        adv, ret = self._gae_scalar(
            credited_reward_all,
            old_reward_values_all,
            done_all,
            bootstrap_value=bootstrap_value,
            normalize_advantage=False,
            discounts=discount_all,
            truncations=truncated_all,
            next_value_overrides=next_reward_values_all,
        )
        constraint_costs_all = torch.stack([tr.constraint_costs for tr in self.buf], dim=0).detach()
        old_constraint_values_all = torch.stack([tr.old_constraint_values for tr in self.buf], dim=0).detach()
        next_constraint_values_all = torch.stack(
            [tr.next_constraint_values_override for tr in self.buf], dim=0
        ).detach()
        constraint_adv_cols = []
        constraint_ret_cols = []
        for k in range(self.constraint_dim):
            constraint_bootstrap_k = None
            if bootstrap_constraint_values is not None:
                constraint_bootstrap_k = torch.as_tensor(
                    bootstrap_constraint_values, dtype=torch.float32, device=self.device
                ).reshape(-1)[k]
            adv_k, ret_k = self._gae_scalar(
                constraint_costs_all[:, k],
                old_constraint_values_all[:, k],
                done_all,
                bootstrap_value=constraint_bootstrap_k,
                normalize_advantage=False,
                discounts=discount_all,
                truncations=truncated_all,
                next_value_overrides=next_constraint_values_all[:, k],
            )
            constraint_adv_cols.append(adv_k)
            constraint_ret_cols.append(ret_k)
        constraint_adv_all = torch.stack(constraint_adv_cols, dim=1)
        constraint_ret_all = torch.stack(constraint_ret_cols, dim=1)
        risk_cost_all = torch.tensor([tr.risk_cost for tr in self.buf], dtype=torch.float32, device=self.device)
        old_risk_value_all = torch.stack([tr.old_risk_value for tr in self.buf], dim=0).detach()
        next_risk_values_all = torch.stack(
            [tr.next_risk_value_override for tr in self.buf], dim=0
        ).detach()
        risk_adv_all, risk_ret_all = self._gae_scalar(
            risk_cost_all,
            old_risk_value_all,
            done_all,
            bootstrap_value=bootstrap_risk_value,
            normalize_advantage=False,
            discounts=discount_all,
            truncations=truncated_all,
            next_value_overrides=next_risk_values_all,
        )
        response_target_all = torch.stack([tr.response_target for tr in self.buf], dim=0).detach()
        action_feature_items = [tr.action_features for tr in self.buf]
        has_action_features = bool(action_feature_items and action_feature_items[0].numel() > 0)
        action_features_all = torch.stack(action_feature_items, dim=0).detach() if has_action_features else None
        action_masks_all = torch.stack([tr.action_mask for tr in self.buf], dim=0).detach().bool()
        lambda_vec = (
            self.constraint_lambdas.detach().to(self.device)
            if self.constraints_active
            else torch.zeros_like(self.constraint_lambdas)
        )
        effective_risk_coeff = self.risk_coeff if self.risk_active else 0.0
        lagrangian_adv = (
            adv
            - torch.sum(constraint_adv_all * lambda_vec.unsqueeze(0), dim=1)
            - effective_risk_coeff * risk_adv_all
        )
        lagrangian_adv = torch.nan_to_num(lagrangian_adv, nan=0.0, posinf=0.0, neginf=0.0)
        lag_mean = lagrangian_adv.mean()
        lag_std = lagrangian_adv.std(unbiased=False)
        if torch.isfinite(lag_std) and float(lag_std.item()) > 1e-8:
            lagrangian_adv = (lagrangian_adv - lag_mean) / (lag_std + 1e-8)
        else:
            lagrangian_adv = lagrangian_adv - lag_mean

        s_all = torch.stack([tr.s for tr in self.buf], dim=0)
        s_all = torch.nan_to_num(s_all, nan=0.0, posinf=1e3, neginf=-1e3)
        a_all = torch.stack([tr.a for tr in self.buf], dim=0)
        magnitude_all = torch.stack([tr.magnitude for tr in self.buf], dim=0).detach()
        feasible_action_count = int(action_masks_all.any(dim=0).sum().item())
        unique_action_fraction = float(
            torch.unique(a_all.detach()).numel() / max(1, feasible_action_count)
        )
        old_logp_all = torch.stack([tr.old_logp for tr in self.buf], dim=0)
        old_logp_all = torch.nan_to_num(old_logp_all, nan=0.0, posinf=20.0, neginf=-20.0)
        exploration_all = torch.tensor(
            [tr.exploration_rate for tr in self.buf],
            dtype=torch.float32,
            device=self.device,
        )
        n = s_all.size(0)

        with torch.no_grad():
            raw_logits_all, _, _, _, _, _, _, _ = self.net(s_all, action_features_all)
            raw_logits_all = self._masked_logits(
                torch.nan_to_num(raw_logits_all, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0),
                action_masks_all,
            )
            raw_probs_all = torch.softmax(raw_logits_all, dim=-1)
            raw_top_all = torch.argmax(raw_probs_all, dim=-1)
            top_counts = torch.bincount(raw_top_all, minlength=self.action_dim).to(torch.float32)
            raw_argmax_concentration = float(top_counts.max().item() / max(1, raw_top_all.numel()))
            raw_hold_probability = float(raw_probs_all[:, 0].mean().item())
            state_action_sensitivity = float(
                raw_probs_all.std(dim=0, unbiased=False).mean().item()
                if raw_probs_all.shape[0] > 1 else 0.0
            )
            row_entropy = -(raw_probs_all * torch.log(raw_probs_all.clamp(min=1e-8))).sum(dim=-1)
            feasible_entropy = torch.log(
                action_masks_all.sum(dim=-1).to(torch.float32).clamp(min=1.0)
            )
            raw_entropy_fraction = float(
                (row_entropy / feasible_entropy.clamp(min=1e-8))
                .masked_fill(feasible_entropy <= 1e-8, 0.0)
                .mean()
                .item()
            )
            economic_probs = self._economic_group_probabilities(
                raw_logits_all,
                action_features_all,
            )
            if economic_probs is not None:
                state_action_mi = float(
                    self._mutual_information(economic_probs).item()
                )
            else:
                state_action_mi = float(
                    self._mutual_information(raw_probs_all).item()
                )

        self.last_raw_argmax_concentration = raw_argmax_concentration
        self.last_state_action_sensitivity = state_action_sensitivity
        self.last_state_action_mutual_information = state_action_mi

        target_minibatches = 4 if n >= 32 else 1
        effective_batch_size = int(
            min(max(1, batch_size), max(1, int(np.ceil(n / target_minibatches))))
        )

        # PopArt keeps economic-unit value predictions invariant while running
        # target moments evolve. This is essential when profit regimes have very
        # different scales.
        reward_heads = [self.net.v_head[-1]]
        if self.net.action_q_head is not None:
            reward_heads.append(self.net.action_q_head[-1])
        self._update_popart(self.reward_normalizer, ret, reward_heads)
        if self.constraints_active:
            self._update_popart(
                self.constraint_normalizer,
                constraint_ret_all,
                [self.net.constraint_head[-1]],
            )
        if self.risk_active:
            self._update_popart(
                self.risk_normalizer,
                risk_ret_all,
                [self.net.risk_head[-1]],
            )
        ret_norm = self.reward_normalizer.normalize(ret).detach()
        old_reward_values_norm = self.reward_normalizer.normalize(
            old_reward_values_all
        ).detach()
        constraint_ret_norm = self.constraint_normalizer.normalize(constraint_ret_all).detach()
        risk_ret_norm = self.risk_normalizer.normalize(risk_ret_all).detach()

        critic_sums = {
            "value_loss": 0.0,
            "constraint_value_loss": 0.0,
            "risk_value_loss": 0.0,
            "action_q_loss": 0.0,
            "critic_loss": 0.0,
            "critic_grad_norm": 0.0,
        }
        critic_steps = 0
        # The state-value critic supports GAE.  Replaying each short,
        # non-stationary rollout through more critic epochs than the policy
        # made V fit the latest opponent regime while the actor still reflected
        # the broader population.  Keep the update budgets aligned.
        critic_epochs = max(1, min(4, int(epochs)))
        critic_params = self.net.critic_parameters()
        for _ in range(critic_epochs):
            idx = torch.randperm(n, device=self.device)
            for start in range(0, n, effective_batch_size):
                j = idx[start : start + effective_batch_size]
                s_b = s_all[j]
                a_b = a_all[j]
                action_features_b = action_features_all[j].detach() if action_features_all is not None else None
                _, _, _, v, constraint_v, risk_v, _, q_values = self.net(
                    s_b, action_features_b
                )
                old_v_b = old_reward_values_norm[j]
                value_clipped = old_v_b + torch.clamp(
                    v - old_v_b,
                    -self.value_clip_eps,
                    self.value_clip_eps,
                )
                value_loss_unclipped = nn.functional.smooth_l1_loss(
                    v, ret_norm[j], beta=1.0, reduction="none"
                )
                value_loss_clipped = nn.functional.smooth_l1_loss(
                    value_clipped, ret_norm[j], beta=1.0, reduction="none"
                )
                value_loss = torch.maximum(
                    value_loss_unclipped, value_loss_clipped
                ).mean()
                constraint_value_loss = nn.functional.smooth_l1_loss(
                    constraint_v, constraint_ret_norm[j], beta=1.0
                )
                risk_value_loss = nn.functional.smooth_l1_loss(
                    risk_v, risk_ret_norm[j], beta=1.0
                )
                action_q_loss = torch.zeros((), dtype=torch.float32, device=self.device)
                if q_values.numel() > 0:
                    chosen_q = q_values.gather(1, a_b.reshape(-1, 1)).squeeze(1)
                    action_q_loss = nn.functional.smooth_l1_loss(
                        chosen_q, ret_norm[j], beta=1.0
                    )
                critic_loss = (
                    value_loss
                    + (
                        self.constraint_value_coeff * constraint_value_loss
                        if self.constraints_active
                        else 0.0
                    )
                    + (
                        self.risk_value_coeff * risk_value_loss
                        if self.risk_active
                        else 0.0
                    )
                    + self.action_q_coeff * action_q_loss
                )
                if not torch.isfinite(critic_loss):
                    continue
                self.critic_opt.zero_grad(set_to_none=True)
                critic_loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(critic_params, 0.5)
                self.critic_opt.step()
                critic_steps += 1
                critic_sums["value_loss"] += float(value_loss.item())
                critic_sums["constraint_value_loss"] += float(constraint_value_loss.item())
                critic_sums["risk_value_loss"] += float(risk_value_loss.item())
                critic_sums["action_q_loss"] += float(action_q_loss.item())
                critic_sums["critic_loss"] += float(critic_loss.item())
                critic_sums["critic_grad_norm"] += float(grad_norm.item())

        # Opponent-response inference is an auxiliary supervised task.  It has
        # its own encoder and optimizer so prediction error cannot pull the
        # pricing actor or the return critic away from their PPO objectives.
        response_loss_sum = 0.0
        response_steps = 0
        if self.response_coeff > 0.0 and response_target_all.numel() > 0:
            response_params = self.net.response_parameters()
            response_epochs = max(1, min(3, int(epochs)))
            for _ in range(response_epochs):
                idx = torch.randperm(n, device=self.device)
                for start in range(0, n, effective_batch_size):
                    j = idx[start : start + effective_batch_size]
                    action_features_b = (
                        action_features_all[j].detach()
                        if action_features_all is not None
                        else None
                    )
                    with torch.enable_grad():
                        _, _, _, _, _, _, response_pred, _ = self.net(
                            s_all[j].detach(), action_features_b
                        )
                        if response_pred.ndim == 3:
                            chosen_response = response_pred.gather(
                                1,
                                a_all[j].reshape(-1, 1, 1).expand(
                                    -1, 1, self.response_dim
                                ),
                            ).squeeze(1)
                        else:
                            chosen_response = response_pred
                        response_loss = nn.functional.smooth_l1_loss(
                            chosen_response,
                            response_target_all[j],
                            beta=0.5,
                        )
                    if not torch.isfinite(response_loss):
                        continue
                    self.response_opt.zero_grad(set_to_none=True)
                    response_loss.backward()
                    nn.utils.clip_grad_norm_(response_params, self.max_grad_norm)
                    self.response_opt.step()
                    response_steps += 1
                    response_loss_sum += float(response_loss.item())

        actor_sums = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "state_action_mi": 0.0,
            "entropy": 0.0,
            "policy_entropy": 0.0,
            "discrete_policy_entropy": 0.0,
            "magnitude_entropy": 0.0,
            "approx_kl": 0.0,
            "clipfrac": 0.0,
        }
        actor_steps = 0
        stopped_early_kl = False
        actor_params = self.net.actor_parameters()
        for _ in range(max(1, int(epochs))):
            epoch_kl_sum = 0.0
            epoch_kl_steps = 0
            idx = torch.randperm(n, device=self.device)
            for start in range(0, n, effective_batch_size):
                j = idx[start : start + effective_batch_size]
                s_b = s_all[j]
                a_b = a_all[j]
                adv_b = lagrangian_adv[j].detach().clamp(-self.adv_clip, self.adv_clip)
                action_features_b = (
                    action_features_all[j].detach()
                    if action_features_all is not None
                    else None
                )
                action_mask_b = action_masks_all[j].detach()
                old_logp_b = old_logp_all[j].detach()
                exploration_b = exploration_all[j].detach()
                magnitude_b = magnitude_all[j].detach()
                logits, mag_mean, mag_logstd, _, _, _, _, _ = self.net(
                    s_b, action_features_b
                )
                logits = self._masked_logits(
                    torch.nan_to_num(
                        logits, nan=0.0, posinf=20.0, neginf=-20.0
                    ).clamp(-20.0, 20.0),
                    action_mask_b,
                )
                if not torch.isfinite(logits).all():
                    continue
                dist = self._exploratory_distribution(
                    logits, exploration_b, action_mask_b
                )
                policy_dist = torch.distributions.Categorical(logits=logits)
                chosen_mean = mag_mean.gather(
                    1, a_b.reshape(-1, 1)
                ).squeeze(1)
                chosen_logstd = mag_logstd.gather(
                    1, a_b.reshape(-1, 1)
                ).squeeze(1)
                mag_dist = self._magnitude_dist(chosen_mean, chosen_logstd)
                mag_logp = torch.where(
                    a_b == 0,
                    torch.zeros_like(magnitude_b),
                    self._magnitude_log_prob(mag_dist, magnitude_b),
                )
                logp = dist.log_prob(a_b) + mag_logp
                ratio = torch.exp(logp - old_logp_b)
                ratio = torch.nan_to_num(
                    ratio,
                    nan=1.0,
                    posinf=1.0 + self.clip_eps,
                    neginf=1.0 - self.clip_eps,
                )
                unclipped = ratio * adv_b
                clipped = torch.clamp(
                    ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps
                ) * adv_b
                policy_loss = -torch.min(unclipped, clipped).mean()
                non_hold = (a_b != 0).to(dtype=logits.dtype)
                mag_entropy_items = mag_dist.entropy()
                mag_entropy = (
                    (mag_entropy_items * non_hold).sum()
                    / non_hold.sum().clamp(min=1.0)
                )
                discrete_entropy = policy_dist.entropy().mean()
                exploratory_entropy = dist.entropy().mean()
                entropy = exploratory_entropy + mag_entropy
                raw_probs_b = torch.softmax(logits, dim=-1)
                economic_probs_b = self._economic_group_probabilities(
                    logits,
                    action_features_b,
                )
                if economic_probs_b is not None:
                    # Specialize economically meaningful direction, not raw
                    # action identity. Columns 10:14 are representative fare
                    # impacts for short through long trips. Grouping actions
                    # into lower / neutral / higher fare prevents a policy
                    # from earning this bonus through low-ranked probability
                    # tails, airport-only chatter, or nearly equivalent
                    # coefficient actions.
                    state_action_mi_b = self._mutual_information(
                        economic_probs_b,
                        torch.relu(adv_b),
                    )
                else:
                    state_action_mi_b = self._mutual_information(
                        raw_probs_b,
                        torch.relu(adv_b),
                    )
                logratio = logp - old_logp_b
                approx_kl = ((torch.exp(logratio) - 1.0) - logratio).mean()
                clipfrac = (
                    (ratio - 1.0).abs() > self.clip_eps
                ).float().mean()
                actor_loss = (
                    policy_loss
                    - self.ent_coeff * (discrete_entropy + mag_entropy)
                    - self.state_action_mi_coeff * state_action_mi_b
                )
                if not torch.isfinite(actor_loss):
                    continue
                self.actor_opt.zero_grad(set_to_none=True)
                actor_loss.backward()
                nn.utils.clip_grad_norm_(actor_params, self.max_grad_norm)
                self.actor_opt.step()
                actor_steps += 1
                actor_sums["loss"] += float(actor_loss.item())
                actor_sums["policy_loss"] += float(policy_loss.item())
                actor_sums["state_action_mi"] += float(state_action_mi_b.item())
                actor_sums["entropy"] += float(entropy.item())
                actor_sums["policy_entropy"] += float(discrete_entropy.item())
                actor_sums["discrete_policy_entropy"] += float(discrete_entropy.item())
                actor_sums["magnitude_entropy"] += float(mag_entropy.item())
                actor_sums["approx_kl"] += float(approx_kl.item())
                actor_sums["clipfrac"] += float(clipfrac.item())
                epoch_kl_sum += float(approx_kl.item())
                epoch_kl_steps += 1
            # A noisy minibatch must not terminate the entire actor update.
            # Trust-region stopping is based on the completed epoch average.
            if (
                epoch_kl_steps > 0
                and epoch_kl_sum / epoch_kl_steps > 1.5 * self.target_kl
            ):
                stopped_early_kl = True
            if stopped_early_kl:
                break

        last = {
            "loss": float(actor_sums["loss"] / max(1, actor_steps)),
            "policy_loss": float(actor_sums["policy_loss"] / max(1, actor_steps)),
            "value_loss": float(critic_sums["value_loss"] / max(1, critic_steps)),
            "constraint_value_loss": float(
                critic_sums["constraint_value_loss"] / max(1, critic_steps)
            ),
            "risk_value_loss": float(
                critic_sums["risk_value_loss"] / max(1, critic_steps)
            ),
            "response_loss": float(response_loss_sum / max(1, response_steps)),
            "action_q_loss": float(
                critic_sums["action_q_loss"] / max(1, critic_steps)
            ),
            "critic_loss": float(
                critic_sums["critic_loss"] / max(1, critic_steps)
            ),
            "critic_grad_norm": float(
                critic_sums["critic_grad_norm"] / max(1, critic_steps)
            ),
            "lagrangian_adv_mean": float(lag_mean.item()),
            "lagrangian_adv_std": float(lag_std.item()),
            "constraint_lambda_mean": float(lambda_vec.mean().item()),
            "risk_coeff": float(effective_risk_coeff),
            "constraints_active": bool(self.constraints_active),
            "risk_active": bool(self.risk_active),
            "entropy": float(actor_sums["entropy"] / max(1, actor_steps)),
            "policy_entropy": float(
                actor_sums["policy_entropy"] / max(1, actor_steps)
            ),
            "discrete_policy_entropy": float(
                actor_sums["discrete_policy_entropy"] / max(1, actor_steps)
            ),
            "magnitude_entropy": float(
                actor_sums["magnitude_entropy"] / max(1, actor_steps)
            ),
            "state_action_mi": float(
                actor_sums["state_action_mi"] / max(1, actor_steps)
            ),
            "state_action_mutual_information": float(state_action_mi),
            "raw_policy_entropy_fraction": float(raw_entropy_fraction),
            "raw_policy_hold_probability": float(raw_hold_probability),
            "raw_policy_argmax_concentration": float(raw_argmax_concentration),
            "state_action_sensitivity": float(state_action_sensitivity),
            "approx_kl": float(actor_sums["approx_kl"] / max(1, actor_steps)),
            "target_kl": float(self.target_kl),
            "clipfrac": float(actor_sums["clipfrac"] / max(1, actor_steps)),
            "update_performed": bool(actor_steps > 0 and critic_steps > 0),
            "run_mode": "training_update",
            "explained_variance": 0.0,
            "ent_coeff": float(self.ent_coeff),
            "clip_eps": float(self.clip_eps),
            "lr": float(self.curr_lr),
            "critic_lr": float(
                self.critic_opt.param_groups[0]["lr"]
            ),
            "exploration_rate": float(self.exploration_rate),
            "action_coverage": self._action_coverage(),
            "rollout_action_diversity": float(unique_action_fraction),
            "rollout_reward_std": float(
                torch.std(reward_all, unbiased=False).item()
            ) if reward_all.numel() > 1 else 0.0,
            "credited_reward_std": float(
                torch.std(credited_reward_all, unbiased=False).item()
            ) if credited_reward_all.numel() > 1 else 0.0,
            "critic_target_mean": float(ret.mean().item()),
            "critic_target_std": float(
                ret.std(unbiased=False).item()
            ) if ret.numel() > 1 else 0.0,
            "delayed_reward_blend": float(self.delayed_reward_blend),
            "continuous_magnitude_mean": float(
                torch.mean(magnitude_all).item()
            ) if magnitude_all.numel() > 0 else 0.0,
            "continuous_magnitude_std": float(
                torch.std(magnitude_all, unbiased=False).item()
            ) if magnitude_all.numel() > 1 else 0.0,
            "learning_signal_ok": bool(
                actor_steps > 0
                and critic_steps > 0
                and torch.isfinite(lag_std)
            ),
            "optimizer_steps": int(actor_steps + critic_steps + response_steps),
            "actor_optimizer_steps": int(actor_steps),
            "critic_optimizer_steps": int(critic_steps),
            "response_optimizer_steps": int(response_steps),
            "stopped_early_kl": bool(stopped_early_kl),
            "rollout_size": int(n),
            "effective_batch_size": int(effective_batch_size),
        }

        feasible_counts = action_masks_all.sum(dim=-1).to(dtype=torch.float32).clamp(min=1.0)
        max_entropy = float(torch.log(feasible_counts).mean().item())
        self.last_policy_entropy_fraction = float(
            np.clip(last["discrete_policy_entropy"] / max_entropy, 0.0, 1.0)
            if max_entropy > 1e-8
            else 0.0
        )
        last["policy_entropy_fraction"] = self.last_policy_entropy_fraction
        last["transition_duration_mean"] = float(
            np.mean([tr.duration for tr in self.buf])
        )
        last["transition_discount_mean"] = float(discount_all.mean().item())
        last["truncation_count"] = int(truncated_all.sum().item())

        with torch.no_grad():
            _, _, _, new_value_norm, _, _, _, _ = self.net(
                s_all, action_features_all
            )
            new_value = self.reward_normalizer.denormalize(new_value_norm)
            ret_var = torch.var(ret, unbiased=False)
            if torch.isfinite(ret_var) and float(ret_var.item()) > 1e-8:
                prediction_error = torch.var(ret - new_value, unbiased=False)
                last["explained_variance"] = float(
                    torch.clamp(
                        1.0 - prediction_error / ret_var, -1.0, 1.0
                    ).item()
                )

        self.buf.clear()
        self.update_calls += 1
        self._apply_control_floors()
        last["ent_coeff"] = float(self.ent_coeff)
        last["clip_eps"] = float(self.clip_eps)
        last["lr"] = float(self.curr_lr)
        last["exploration_rate"] = float(self.exploration_rate)
        return last

    def optimizer_state_dict(self) -> Dict[str, object]:
        return {
            "actor": self.actor_opt.state_dict(),
            "critic": self.critic_opt.state_dict(),
            "response": self.response_opt.state_dict(),
        }

    def load_optimizer_state_dict(self, state: Optional[Dict[str, object]]) -> None:
        if not state:
            return
        if "actor" not in state or "critic" not in state:
            raise ValueError(
                "PPO optimizer state must contain separate actor and critic states"
            )
        self.actor_opt.load_state_dict(state["actor"])
        self.critic_opt.load_state_dict(state["critic"])
        if "response" in state:
            self.response_opt.load_state_dict(state["response"])
        self._set_learning_rates(self.curr_lr)

    def normalizer_state_dict(self) -> Dict[str, object]:
        return {
            "reward": self.reward_normalizer.state_dict(),
            "constraints": self.constraint_normalizer.state_dict(),
            "risk": self.risk_normalizer.state_dict(),
        }

    def load_normalizer_state_dict(self, state: Optional[Dict[str, object]]) -> None:
        if not state:
            return
        self.reward_normalizer.load_state_dict(state.get("reward", {}))
        self.constraint_normalizer.load_state_dict(state.get("constraints", {}))
        self.risk_normalizer.load_state_dict(state.get("risk", {}))

    def _action_coverage(self) -> float:
        if self.action_visits.size == 0:
            return 1.0
        eligible = self.action_ever_feasible
        if not bool(np.any(eligible)):
            return 0.0
        if self.min_action_visits <= 0:
            return float(np.mean(self.action_visits[eligible] > 0))
        return float(np.mean(self.action_visits[eligible] >= self.min_action_visits))
