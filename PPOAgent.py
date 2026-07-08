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
from typing import Dict, List, Optional, Tuple

import mkl_config  # noqa: F401 - set oneMKL env before NumPy/Torch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class ActorCritic(nn.Module):
    """Actor-critic network with response, constraint, and risk heads.

    The policy head scores feasible centralized interventions.  The reward
    head estimates long-run operational benefit; constraint heads estimate
    future service/safety pressure; the risk head estimates tail-risk pressure;
    and the response head predicts next aggregate crowd/system response.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden: int = 192,
        constraint_dim: int = 5,
        response_dim: int = 4,
        action_feature_dim: int = 0,
    ):
        super().__init__()
        self.constraint_dim = int(max(1, constraint_dim))
        self.response_dim = int(max(1, response_dim))
        self.action_dim = int(max(1, action_dim))
        self.action_feature_dim = int(max(0, action_feature_dim))
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.action_encoder = None
        self.action_score_head = None
        self.action_q_head = None
        if self.action_feature_dim > 0:
            # Action-conditioned scoring lets PPO evaluate the causal footprint of
            # each option (target coefficient, direction, continuous magnitude,
            # and expected crowd segment exposure) instead of treating actions as
            # opaque integer IDs.
            self.action_encoder = nn.Sequential(
                nn.Linear(self.action_feature_dim, hidden // 2),
                nn.SiLU(),
                nn.Linear(hidden // 2, hidden // 2),
                nn.SiLU(),
            )
            self.action_score_head = nn.Sequential(nn.Linear(hidden + hidden // 2, hidden), nn.SiLU(), nn.Linear(hidden, 1))
            self.action_q_head = nn.Sequential(nn.Linear(hidden + hidden // 2, hidden), nn.SiLU(), nn.Linear(hidden, 1))
            self.pi_head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, action_dim))
        else:
            self.pi_head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, action_dim))
        self.mag_mean_head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, action_dim))
        self.mag_logstd = nn.Parameter(torch.full((action_dim,), -0.35))
        self.v_head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.constraint_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.constraint_dim),
        )
        self.risk_head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.response_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.response_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """Orthogonal init improves PPO stability and early optimization speed."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2.0))
                nn.init.constant_(module.bias, 0.0)

        if isinstance(self.pi_head[-1], nn.Linear):
            nn.init.orthogonal_(self.pi_head[-1].weight, gain=0.01)
            nn.init.constant_(self.pi_head[-1].bias, 0.0)
        if isinstance(self.mag_mean_head[-1], nn.Linear):
            nn.init.orthogonal_(self.mag_mean_head[-1].weight, gain=0.01)
            nn.init.constant_(self.mag_mean_head[-1].bias, 0.0)
        
        for head in (self.v_head, self.constraint_head, self.risk_head, self.response_head):
            if isinstance(head[-1], nn.Linear):
                nn.init.orthogonal_(head[-1].weight, gain=1.0)
                nn.init.constant_(head[-1].bias, 0.0)

    def forward(
        self, s: torch.Tensor, action_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.trunk(s)
        q_values = torch.empty(0, dtype=z.dtype, device=z.device)
        if self.action_feature_dim > 0 and action_features is not None and self.action_encoder is not None:
            af = torch.nan_to_num(action_features, nan=0.0, posinf=1.0, neginf=-1.0).to(dtype=z.dtype, device=z.device)
            if af.ndim == 2:
                af = af.unsqueeze(0)
            a_emb = self.action_encoder(af)
            z_expanded = z.unsqueeze(1).expand(-1, a_emb.shape[1], -1)
            joint = torch.cat([z_expanded, a_emb], dim=-1)
            logits = self.action_score_head(joint).squeeze(-1)
            q_values = self.action_q_head(joint).squeeze(-1)
        else:
            logits = self.pi_head(z)
        mag_mean = torch.tanh(self.mag_mean_head(z))
        mag_logstd = self.mag_logstd.clamp(-2.5, 0.75).expand_as(mag_mean)
        return (
            logits,
            mag_mean,
            mag_logstd,
            self.v_head(z).squeeze(-1),
            self.constraint_head(z),
            self.risk_head(z).squeeze(-1),
            self.response_head(z),
            q_values,
        )
    
@dataclass
class Transition:
    s: torch.Tensor
    a: torch.Tensor
    r: float
    done: bool
    old_logp: torch.Tensor
    old_value: torch.Tensor
    magnitude: torch.Tensor
    exploration_rate: float
    constraint_costs: torch.Tensor
    risk_cost: float
    response_target: torch.Tensor
    action_features: torch.Tensor
    action_trace: torch.Tensor
    old_constraint_values: torch.Tensor
    old_risk_value: torch.Tensor

class PPOAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.60,
        final_clip_eps: float = 0.35,
        v_coeff: float = 0.5,
        ent_coeff: float = 0.01,
        min_ent_coeff: float = 0.001,
        ent_decay: float = 0.995,
        max_grad_norm: float = 1.0,
        hidden_dim: int = 192,
        target_kl: float = 0.50,
        adv_clip: float = 4.0,
        lr_decay_on_spike: float = 0.85,
        lr_growth_on_stall: float = 1.08,
        min_lr: float = 5e-5,
        max_lr: Optional[float] = None,
        value_clip_eps: float = 0.20,
        initial_exploration_rate: float = 0.35,
        final_exploration_rate: float = 0.02,
        exploration_fraction: float = 0.75,
        exploration_warmup_fraction: float = 0.20,
        min_action_visits: int = 8,
        exploration_rescue_rate: float = 0.12,
        constraint_dim: int = 5,
        response_dim: int = 4,
        action_feature_dim: int = 0,
        action_trace_dim: int = 8,
        action_q_coeff: float = 0.08,
        constraint_value_coeff: float = 0.25,
        risk_value_coeff: float = 0.15,
        response_coeff: float = 0.05,
        risk_coeff: float = 0.10,
        delayed_reward_horizon: int = 6,
        delayed_reward_blend: float = 0.35,
        device: Optional[str] = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.constraint_dim = int(max(1, constraint_dim))
        self.response_dim = int(max(1, response_dim))
        self.action_dim = int(max(1, action_dim))
        self.action_feature_dim = int(max(0, action_feature_dim))
        self.net = ActorCritic(
            state_dim,
            action_dim,
            hidden=hidden_dim,
            constraint_dim=self.constraint_dim,
            response_dim=self.response_dim,
            action_feature_dim=action_feature_dim,
        ).to(self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)
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
        self.action_trace_dim = int(max(1, action_trace_dim))
        self.action_q_coeff = float(max(0.0, action_q_coeff))
        self.constraint_value_coeff = float(max(0.0, constraint_value_coeff))
        self.risk_value_coeff = float(max(0.0, risk_value_coeff))
        self.response_coeff = float(max(0.0, response_coeff))
        self.risk_coeff = float(max(0.0, risk_coeff))
        self.delayed_reward_horizon = int(max(1, delayed_reward_horizon))
        self.delayed_reward_blend = float(np.clip(delayed_reward_blend, 0.0, 1.0))
        self.constraint_lambdas = torch.zeros(self.constraint_dim, dtype=torch.float32, device=self.device)
        self.ent_coeff = ent_coeff
        self.max_ent_coeff = float(max(ent_coeff, min_ent_coeff))
        self.min_ent_coeff = float(max(0.0, min_ent_coeff))
        self.ent_decay = float(np.clip(ent_decay, 0.90, 1.0))
        
        self.max_grad_norm = max_grad_norm
        self.update_calls = 0
        self.target_kl = float(max(1e-4, target_kl))
        self.adv_clip = float(max(1.0, adv_clip))
        self.lr_decay_on_spike = float(np.clip(lr_decay_on_spike, 0.5, 0.99))
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
        self.exploration_rescue_rate = float(
            np.clip(exploration_rescue_rate, self.final_exploration_rate, self.initial_exploration_rate)
        )
        self.exploration_rate = self.initial_exploration_rate
        self.last_action_exploration_rate = self.exploration_rate
        self.action_visits = np.zeros(int(action_dim), dtype=np.int64)
        self.last_policy_entropy_fraction = 1.0
        self.last_continuous_magnitude = 0.0

        self.buf: List[Transition] = []
        self.last_constraint_values = torch.zeros(self.constraint_dim, dtype=torch.float32, device=self.device)
        self.last_risk_value = torch.zeros((), dtype=torch.float32, device=self.device)
        self.last_response_pred = torch.zeros(self.response_dim, dtype=torch.float32, device=self.device)
    
    def set_optimization_context(
        self,
        constraint_lambdas: Optional[np.ndarray | List[float] | Tuple[float, ...]] = None,
        risk_coeff: Optional[float] = None,
    ) -> None:
        """Update Lagrangian pressure used by the structured PPO actor."""
        if constraint_lambdas is not None:
            arr = np.asarray(constraint_lambdas, dtype=np.float32).reshape(-1)
            padded = np.zeros(self.constraint_dim, dtype=np.float32)
            width = min(self.constraint_dim, arr.size)
            if width > 0:
                padded[:width] = np.maximum(arr[:width], 0.0)
            self.constraint_lambdas = torch.tensor(padded, dtype=torch.float32, device=self.device)
        if risk_coeff is not None:
            self.risk_coeff = float(max(0.0, risk_coeff))
            
    def adapt_exploration(
        self,
        progress: float,
        reward_converged: bool,
        reward_std: Optional[float] = None,
    ) -> None:
        """Anneal exploration only after action coverage and a true warmup.

        Entropy bonuses only encourage a policy to remain diffuse; they do not
        guarantee that every action is sampled.  Mixing the learned categorical
        policy with a uniform distribution gives every action a known minimum
        probability in the large state space.  The same mixture is used when
        PPO recomputes log probabilities, so importance ratios remain valid.

        A cosine schedule avoids the abrupt loss of exploration caused by the
        old linear schedule.  Annealing is delayed until every action has been
        sampled enough times, and a rescue floor is restored when the learned
        policy collapses early or the reward becomes flat before convergence.
        """
        p = float(np.clip(progress, 0.0, 1.0))
        if reward_converged:
            self.exploration_rate = self.final_exploration_rate
            return

        covered = bool(
            self.min_action_visits <= 0
            or (self.action_visits.size > 0 and int(self.action_visits.min()) >= self.min_action_visits)
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

        flat_reward = reward_std is not None and np.isfinite(reward_std) and reward_std < 0.01
        premature_collapse = p < 0.85 and self.last_policy_entropy_fraction < 0.20
        if flat_reward or premature_collapse:
            rate = max(rate, self.exploration_rescue_rate)
        self.exploration_rate = float(
            np.clip(rate, self.final_exploration_rate, self.initial_exploration_rate)
        )

    @staticmethod
    def _exploratory_distribution(
        logits: torch.Tensor, exploration_rate: torch.Tensor | float
    ) -> torch.distributions.Categorical:
        policy_probs = torch.softmax(logits, dim=-1)
        eps = torch.as_tensor(
            exploration_rate, dtype=policy_probs.dtype, device=policy_probs.device
        )
        if eps.ndim == 0:
            eps = eps.expand(policy_probs.shape[:-1])
        eps = eps.unsqueeze(-1)
        action_count = max(1, policy_probs.shape[-1])
        mixed_probs = (1.0 - eps) * policy_probs + eps / float(action_count)
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
        """Keep exploration high early, then tighten as training converges."""
        p = float(np.clip(progress, 0.0, 1.0))
        if p <= 0.35 and not reward_converged:
            self.ent_coeff = float(max(self.ent_coeff, 0.65 * self.max_ent_coeff))
            return

        if p <= 0.55 and not reward_converged:
            self.ent_coeff = float(max(self.ent_coeff, 0.40 * self.max_ent_coeff))
            return

        # The driver-supply environment is non-stationary and high variance, so
        # keep enough exploration early but decay more decisively after the
        # policy has seen a representative warmup window.  Persistently high
        # entropy was preventing convergence in longer driver-enabled runs.
        target = self.min_ent_coeff + (self.max_ent_coeff - self.min_ent_coeff) * max(0.0, 1.0 - p) ** 1.5
        if reward_converged:
            target = max(self.min_ent_coeff, 0.75 * target)
        self.ent_coeff = float(np.clip(min(self.ent_coeff, target), self.min_ent_coeff, self.max_ent_coeff))
    
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
            self.curr_lr = self.max_lr
            for group in self.opt.param_groups:
                group["lr"] = self.curr_lr
        
        early_clip = max(self.initial_clip_eps, 0.60)
        mid_clip = max(self.final_clip_eps, 0.35)
        self.clip_eps = float(
            mid_clip + (early_clip - mid_clip) * cosine
        )
        if reward_converged:
            self.clip_eps = min(self.clip_eps, self.final_clip_eps)

    @torch.no_grad()
    def act(
        self,
        s_np: np.ndarray,
        deterministic: bool = False,
        action_features: Optional[np.ndarray] = None,
    ) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s = torch.tensor(s_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        s = torch.nan_to_num(s, nan=0.0, posinf=1e3, neginf=-1e3)
        expected_dim = self.net.trunk[0].in_features
        if s.shape[-1] != expected_dim:
            raise ValueError(f"State dim mismatch: got {s.shape[-1]}, expected {expected_dim}")
        
        af_tensor = None
        if self.action_feature_dim > 0 and action_features is not None:
            af_tensor = torch.as_tensor(action_features, dtype=torch.float32, device=self.device)
            af_tensor = af_tensor.reshape(1, af_tensor.shape[-2], af_tensor.shape[-1])
        logits, mag_mean, mag_logstd, value, constraint_values, risk_value, response_pred, _ = self.net(s, af_tensor)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
        if not torch.isfinite(logits).all():
            logits = torch.zeros_like(logits)
        exploration_rate = 0.0 if deterministic else self.exploration_rate
        self.last_action_exploration_rate = float(exploration_rate)
        dist = self._exploratory_distribution(logits, exploration_rate)
        a = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
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
        self.last_response_pred = torch.nan_to_num(
            response_pred.squeeze(0), nan=0.0, posinf=0.0, neginf=0.0
        ).detach()
        if not deterministic:
            self.action_visits[int(a.item())] += 1
        af_out = af_tensor.squeeze(0).detach() if af_tensor is not None else torch.empty(0, dtype=torch.float32, device=self.device)
        return int(a.item()), s.squeeze(0), logp.squeeze(0), value.squeeze(0), af_out
    
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
        action_trace: Optional[np.ndarray | List[float] | Tuple[float, ...] | torch.Tensor] = None,
    ) -> None:
        del s_next
        if constraint_costs is None:
            constraint_tensor = torch.zeros(self.constraint_dim, dtype=torch.float32, device=self.device)
        else:
            constraint_tensor = torch.as_tensor(constraint_costs, dtype=torch.float32, device=self.device).reshape(-1)
            if constraint_tensor.numel() != self.constraint_dim:
                padded = torch.zeros(self.constraint_dim, dtype=torch.float32, device=self.device)
                width = min(self.constraint_dim, constraint_tensor.numel())
                if width > 0:
                    padded[:width] = constraint_tensor[:width]
                constraint_tensor = padded
        if response_target is None:
            response_tensor = self.last_response_pred.detach().clone()
        else:
            response_tensor = torch.as_tensor(response_target, dtype=torch.float32, device=self.device).reshape(-1)
            if response_tensor.numel() != self.response_dim:
                padded = torch.zeros(self.response_dim, dtype=torch.float32, device=self.device)
                width = min(self.response_dim, response_tensor.numel())
                if width > 0:
                    padded[:width] = response_tensor[:width]
                response_tensor = padded
        
        if action_features is None or self.action_feature_dim <= 0:
            action_feature_tensor = torch.empty(0, dtype=torch.float32, device=self.device)
        else:
            action_feature_tensor = torch.as_tensor(action_features, dtype=torch.float32, device=self.device)
            action_feature_tensor = action_feature_tensor.reshape(-1, action_feature_tensor.shape[-1])
            if action_feature_tensor.shape[-1] != self.action_feature_dim:
                fixed = torch.zeros((self.action_visits.size, self.action_feature_dim), dtype=torch.float32, device=self.device)
                rows = min(fixed.shape[0], action_feature_tensor.shape[0])
                cols = min(fixed.shape[1], action_feature_tensor.shape[1])
                fixed[:rows, :cols] = action_feature_tensor[:rows, :cols]
                action_feature_tensor = fixed
        if action_trace is None:
            action_trace_tensor = torch.zeros(self.action_trace_dim, dtype=torch.float32, device=self.device)
        else:
            action_trace_tensor = torch.as_tensor(action_trace, dtype=torch.float32, device=self.device).reshape(-1)
            if action_trace_tensor.numel() != self.action_trace_dim:
                fixed_trace = torch.zeros(self.action_trace_dim, dtype=torch.float32, device=self.device)
                width = min(self.action_trace_dim, action_trace_tensor.numel())
                fixed_trace[:width] = action_trace_tensor[:width]
                action_trace_tensor = fixed_trace
                
        constraint_tensor = torch.nan_to_num(constraint_tensor, nan=0.0, posinf=1.0, neginf=0.0).clamp(min=0.0)
        response_tensor = torch.nan_to_num(response_tensor, nan=0.0, posinf=1.0, neginf=-1.0)
        
        action_feature_tensor = torch.nan_to_num(action_feature_tensor, nan=0.0, posinf=1.0, neginf=-1.0)
        action_trace_tensor = torch.nan_to_num(action_trace_tensor, nan=0.0, posinf=1.0, neginf=-1.0)
        
        self.buf.append(
            Transition(
                s=s.detach(),
                a=torch.tensor(a, dtype=torch.long, device=self.device),
                r=float(r),
                done=bool(done),
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
                action_trace=action_trace_tensor.detach(),
                old_constraint_values=self.last_constraint_values.detach().clone(),
                old_risk_value=self.last_risk_value.detach().clone(),
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generalized advantage estimation for reward, constraint, or risk streams."""
        t_size = rewards.numel()
        adv = torch.zeros(t_size, dtype=torch.float32, device=self.device)
        ret = torch.zeros(t_size, dtype=torch.float32, device=self.device)

        gae = torch.zeros((), dtype=torch.float32, device=self.device)
        next_v = torch.zeros((), dtype=torch.float32, device=self.device)
        for t in reversed(range(t_size)):
            continuation = 1.0 - float(dones[t].item())
            delta = rewards[t] + self.gamma * next_v * continuation - old_values[t]
            gae = delta + self.gamma * self.lam * continuation * gae
            adv[t] = gae
            ret[t] = adv[t] + old_values[t]
            next_v = old_values[t]

        adv_mean = adv.mean()
        adv_std = adv.std(unbiased=False)
        if torch.isfinite(adv_std) and float(adv_std.item()) > 1e-8:
            adv = (adv - adv_mean) / (adv_std + 1e-8)
        else:
            adv = adv - adv_mean

        adv = torch.nan_to_num(adv, nan=0.0, posinf=0.0, neginf=0.0)
        ret = torch.nan_to_num(ret, nan=0.0, posinf=0.0, neginf=0.0)
        return adv, ret

    def update(self, epochs: int = 5, batch_size: int = 256) -> dict:
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
            }

        reward_all = torch.tensor([tr.r for tr in self.buf], dtype=torch.float32, device=self.device)
        done_all = torch.tensor([float(tr.done) for tr in self.buf], dtype=torch.float32, device=self.device)
        credited_reward_all = self._delayed_reward_credit(reward_all, done_all)
        old_reward_values_all = torch.stack([tr.old_value for tr in self.buf], dim=0).detach()
        old_reward_values_all = torch.nan_to_num(old_reward_values_all, nan=0.0, posinf=0.0, neginf=0.0)
        adv, ret = self._gae_scalar(credited_reward_all, old_reward_values_all, done_all)
        constraint_costs_all = torch.stack([tr.constraint_costs for tr in self.buf], dim=0).detach()
        old_constraint_values_all = torch.stack([tr.old_constraint_values for tr in self.buf], dim=0).detach()
        constraint_adv_cols = []
        constraint_ret_cols = []
        for k in range(self.constraint_dim):
            adv_k, ret_k = self._gae_scalar(constraint_costs_all[:, k], old_constraint_values_all[:, k], done_all)
            constraint_adv_cols.append(adv_k)
            constraint_ret_cols.append(ret_k)
        constraint_adv_all = torch.stack(constraint_adv_cols, dim=1)
        constraint_ret_all = torch.stack(constraint_ret_cols, dim=1)
        risk_cost_all = torch.tensor([tr.risk_cost for tr in self.buf], dtype=torch.float32, device=self.device)
        old_risk_value_all = torch.stack([tr.old_risk_value for tr in self.buf], dim=0).detach()
        risk_adv_all, risk_ret_all = self._gae_scalar(risk_cost_all, old_risk_value_all, done_all)
        response_target_all = torch.stack([tr.response_target for tr in self.buf], dim=0).detach()
        action_feature_items = [tr.action_features for tr in self.buf]
        has_action_features = bool(action_feature_items and action_feature_items[0].numel() > 0)
        action_features_all = torch.stack(action_feature_items, dim=0).detach() if has_action_features else None
        lambda_vec = self.constraint_lambdas.detach().to(self.device)
        lagrangian_adv = adv - torch.sum(constraint_adv_all * lambda_vec.unsqueeze(0), dim=1) - self.risk_coeff * risk_adv_all
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
        # Rollout-local action diversity catches policy collapse that global
        # lifetime coverage cannot see.  A policy can have visited every action
        # early in training and still stop collecting informative on-policy
        # contrast later; PPO then has too little action-level signal to learn
        # meaningful state/action preferences.
        unique_action_fraction = float(
            torch.unique(a_all.detach()).numel() / max(1, self.action_dim)
        )
        old_logp_all = torch.stack([tr.old_logp for tr in self.buf], dim=0)
        old_logp_all = torch.nan_to_num(old_logp_all, nan=0.0, posinf=20.0, neginf=-20.0)
        old_value_all = old_reward_values_all
        exploration_all = torch.tensor(
            [tr.exploration_rate for tr in self.buf],
            dtype=torch.float32,
            device=self.device,
        )
        
        n = s_all.size(0)
        last = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "constraint_value_loss": 0.0,
            "risk_value_loss": 0.0,
            "response_loss": 0.0,
            "action_q_loss": 0.0,
            "lagrangian_adv_mean": float(lag_mean.item()),
            "lagrangian_adv_std": float(lag_std.item()),
            "constraint_lambda_mean": float(lambda_vec.mean().item()),
            "risk_coeff": float(self.risk_coeff),
            "entropy": 0.0,
            "policy_entropy": 0.0,
            "policy_entropy_fraction": float(self.last_policy_entropy_fraction),
            "approx_kl": 0.0,
            "clipfrac": 0.0,
            "explained_variance": 0.0,
            "ent_coeff": float(self.ent_coeff),
            "clip_eps": float(self.clip_eps),
            "lr": float(self.curr_lr),
            "stopped_early_kl": False,
            "exploration_rate": float(self.exploration_rate),
            "action_coverage": self._action_coverage(),
            "rollout_action_diversity": float(unique_action_fraction),
            "rollout_reward_std": float(torch.std(reward_all, unbiased=False).item()) if reward_all.numel() > 1 else 0.0,
            "credited_reward_std": float(torch.std(credited_reward_all, unbiased=False).item()) if credited_reward_all.numel() > 1 else 0.0,
            "delayed_reward_blend": float(self.delayed_reward_blend),
            "credited_reward_std": float(torch.std(credited_reward_all, unbiased=False).item()) if credited_reward_all.numel() > 1 else 0.0,
            "delayed_reward_blend": float(self.delayed_reward_blend),
            "continuous_magnitude_mean": float(torch.mean(magnitude_all).item()) if magnitude_all.numel() > 0 else 0.0,
            "continuous_magnitude_std": float(torch.std(magnitude_all, unbiased=False).item()) if magnitude_all.numel() > 1 else 0.0,
            "learning_signal_ok": True,
            "optimizer_steps": 0,
        }
        stop_for_kl = False
        metric_sums: Dict[str, float] = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "constraint_value_loss": 0.0,
            "risk_value_loss": 0.0,
            "response_loss": 0.0,
            "action_q_loss": 0.0,
            "entropy": 0.0,
            "policy_entropy": 0.0,
            "approx_kl": 0.0,
            "clipfrac": 0.0,
        }
        optimizer_steps = 0


        for _ in range(epochs):
            idx = torch.randperm(n, device=self.device)
            for start in range(0, n, batch_size):
                j = idx[start : start + batch_size]
                s_b = s_all[j]
                a_b = a_all[j]
                adv_b = lagrangian_adv[j].detach().clamp(-self.adv_clip, self.adv_clip)
                ret_b = ret[j].detach()
                constraint_ret_b = constraint_ret_all[j].detach()
                risk_ret_b = risk_ret_all[j].detach()
                response_target_b = response_target_all[j].detach()
                action_features_b = action_features_all[j].detach() if action_features_all is not None else None
                old_logp_b = old_logp_all[j].detach()
                old_value_b = old_value_all[j].detach()
                exploration_b = exploration_all[j].detach()

                magnitude_b = magnitude_all[j].detach()
                logits, mag_mean, mag_logstd, v, constraint_v, risk_v, response_pred, q_values = self.net(s_b, action_features_b)
                logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
                if not torch.isfinite(logits).all():
                    continue
                dist = self._exploratory_distribution(logits, exploration_b)
                policy_dist = torch.distributions.Categorical(logits=logits)
                chosen_mean = mag_mean.gather(1, a_b.reshape(-1, 1)).squeeze(1)
                chosen_logstd = mag_logstd.gather(1, a_b.reshape(-1, 1)).squeeze(1)
                mag_dist = self._magnitude_dist(chosen_mean, chosen_logstd)
                mag_logp = torch.where(a_b == 0, torch.zeros_like(magnitude_b), self._magnitude_log_prob(mag_dist, magnitude_b))
                logp = dist.log_prob(a_b) + mag_logp
                
                ratio = torch.exp(logp - old_logp_b)
                ratio = torch.nan_to_num(ratio, nan=1.0, posinf=1.0 + self.clip_eps, neginf=1.0 - self.clip_eps)
                unclipped = ratio * adv_b
                clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_b
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_unclipped = (v - ret_b) ** 2
                v_clipped = old_value_b + torch.clamp(v - old_value_b, -self.value_clip_eps, self.value_clip_eps)
                value_clipped = (v_clipped - ret_b) ** 2
                value_loss = torch.max(value_unclipped, value_clipped).mean()
                constraint_value_loss = torch.mean((constraint_v - constraint_ret_b) ** 2)
                risk_value_loss = torch.mean((risk_v - risk_ret_b) ** 2)
                response_loss = torch.mean((response_pred - response_target_b) ** 2)
                action_q_loss = torch.zeros((), dtype=torch.float32, device=self.device)
                if q_values.numel() > 0:
                    chosen_q = q_values.gather(1, a_b.reshape(-1, 1)).squeeze(1)
                    action_q_loss = torch.mean((chosen_q - ret_b) ** 2)
                mag_entropy = mag_dist.entropy().mean()
                entropy = dist.entropy().mean() + mag_entropy
                policy_entropy = policy_dist.entropy().mean() + mag_entropy
                logratio = logp - old_logp_b
                approx_kl = ((torch.exp(logratio) - 1.0) - logratio).mean()
                clipfrac = ((ratio - 1.0).abs() > self.clip_eps).float().mean()

                # KL is tracked as a diagnostic only.  Do not stop early here:
                # the pricing policy needs room for larger PPO updates,
                # especially in the early and middle phases of training.
                # Entropy regularizes the learned policy, not the externally
                # forced uniform mixture. Otherwise epsilon can hide collapse.
                loss = (
                    policy_loss
                    + self.v_coeff * value_loss
                    + self.constraint_value_coeff * constraint_value_loss
                    + self.risk_value_coeff * risk_value_loss
                    + self.response_coeff * response_loss
                    + self.action_q_coeff * action_q_loss
                    - self.ent_coeff * policy_entropy
                )
                if not torch.isfinite(loss):
                    continue

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.opt.step()
                optimizer_steps += 1
                metric_sums["loss"] += float(loss.item())
                metric_sums["policy_loss"] += float(policy_loss.item())
                metric_sums["value_loss"] += float(value_loss.item())
                metric_sums["constraint_value_loss"] += float(constraint_value_loss.item())
                metric_sums["risk_value_loss"] += float(risk_value_loss.item())
                metric_sums["response_loss"] += float(response_loss.item())
                metric_sums["action_q_loss"] += float(action_q_loss.item())
                metric_sums["entropy"] += float(entropy.item())
                metric_sums["policy_entropy"] += float(policy_entropy.item())
                metric_sums["approx_kl"] += float(approx_kl.item())
                metric_sums["clipfrac"] += float(clipfrac.item())
            if stop_for_kl:
                break
        
        if optimizer_steps > 0:
            for key, total in metric_sums.items():
                last[key] = float(total / optimizer_steps)
        max_entropy = float(np.log(max(2, self.action_visits.size)))
        self.last_policy_entropy_fraction = float(
            np.clip(last["policy_entropy"] / max_entropy, 0.0, 1.0)
        )
        last["policy_entropy_fraction"] = self.last_policy_entropy_fraction
        last["optimizer_steps"] = int(optimizer_steps)
        last["action_coverage"] = self._action_coverage()
        last["rollout_action_diversity"] = float(unique_action_fraction)
        last["rollout_reward_std"] = float(torch.std(reward_all, unbiased=False).item()) if reward_all.numel() > 1 else 0.0
        last["credited_reward_std"] = float(torch.std(credited_reward_all, unbiased=False).item()) if credited_reward_all.numel() > 1 else 0.0
        last["delayed_reward_blend"] = float(self.delayed_reward_blend)
        last["continuous_magnitude_mean"] = float(torch.mean(magnitude_all).item()) if magnitude_all.numel() > 0 else 0.0
        last["continuous_magnitude_std"] = float(torch.std(magnitude_all, unbiased=False).item()) if magnitude_all.numel() > 1 else 0.0
        # Preserve exploration when a rollout contains too little action contrast
        # or the optimizer made no usable update.  This avoids declaring success
        # after historical action coverage while the current on-policy data is
        # effectively one repeated action with weak credit assignment.
        weak_rollout_signal = bool(
            optimizer_steps <= 0
            or unique_action_fraction < min(0.35, 3.0 / max(1.0, float(self.action_dim)))
            or float(last.get("lagrangian_adv_std", 0.0)) < 1e-6
        )
        last["learning_signal_ok"] = not weak_rollout_signal
        if weak_rollout_signal:
            self.exploration_rate = float(max(self.exploration_rate, self.exploration_rescue_rate))
            self.ent_coeff = float(max(self.ent_coeff, 0.35 * self.max_ent_coeff))
            self.low_update_streak = 0
        with torch.no_grad():
            ret_var = torch.var(ret, unbiased=False)
            if torch.isfinite(ret_var) and float(ret_var.item()) > 1e-8:
                prediction_error = torch.var(ret - old_value_all, unbiased=False)
                last["explained_variance"] = float(
                    torch.clamp(1.0 - prediction_error / ret_var, -1.0, 1.0).item()
                )
                
        final_clipfrac = float(last.get("clipfrac", 0.0))
        final_kl = float(last.get("approx_kl", 0.0))
        # Treat KL/clip fraction as diagnostics rather than restrictive
        # guardrails.  Do not shrink the learning rate because KL or clipfrac
        # are high; PPO is allowed to take larger updates while the schedule
        # controls only the broad late-training consolidation.
        if final_clipfrac < 0.03 and final_kl < 0.35 * self.target_kl:
            self.low_update_streak += 1
            if self.low_update_streak >= 2 and self.curr_lr < self.max_lr:
                self.curr_lr = float(min(self.max_lr, self.curr_lr * self.lr_growth_on_stall))
                for g in self.opt.param_groups:
                    g["lr"] = self.curr_lr
                self.low_update_streak = 0
        else:
            self.low_update_streak = 0

        self.buf.clear()
        self.update_calls += 1
        if not bool(last.get("learning_signal_ok", True)):
            self.ent_coeff = float(max(self.ent_coeff, 0.35 * self.max_ent_coeff))
        else:
            self.ent_coeff = max(self.min_ent_coeff, self.ent_coeff * self.ent_decay)

        last["ent_coeff"] = float(self.ent_coeff)
        last["clip_eps"] = float(self.clip_eps)
        last["lr"] = float(self.curr_lr)
        last["stopped_early_kl"] = bool(stop_for_kl)
        last["exploration_rate"] = float(self.exploration_rate)
        return last

    def _action_coverage(self) -> float:
        if self.action_visits.size == 0:
            return 1.0
        if self.min_action_visits <= 0:
            return float(np.mean(self.action_visits > 0))
        return float(np.mean(self.action_visits >= self.min_action_visits))