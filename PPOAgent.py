from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sun Feb  8 15:58:46 2026

@author: Xiaoru Shi

PPO actor-critic for discrete pricing policies.

This is designed for STABILITY:
- advantage normalization
- clipped policy and value objectives
- adaptive entropy and learning-rate controls
- gradient clipping

"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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
    ):
        super().__init__()
        self.constraint_dim = int(max(1, constraint_dim))
        self.response_dim = int(max(1, response_dim))
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.pi_head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, action_dim))
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
        
        for head in (self.v_head, self.constraint_head, self.risk_head, self.response_head):
            if isinstance(head[-1], nn.Linear):
                nn.init.orthogonal_(head[-1].weight, gain=1.0)
                nn.init.constant_(head[-1].bias, 0.0)

    def forward(
        self, s: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.trunk(s)
        return (
            self.pi_head(z),
            self.v_head(z).squeeze(-1),
            self.constraint_head(z),
            self.risk_head(z).squeeze(-1),
            self.response_head(z),
        )
    
@dataclass
class Transition:
    s: torch.Tensor
    a: torch.Tensor
    r: float
    done: bool
    old_logp: torch.Tensor
    old_value: torch.Tensor
    exploration_rate: float
    constraint_costs: torch.Tensor
    risk_cost: float
    response_target: torch.Tensor
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
        clip_eps: float = 0.2,
        final_clip_eps: float = 0.10,
        v_coeff: float = 0.5,
        ent_coeff: float = 0.01,
        min_ent_coeff: float = 0.001,
        ent_decay: float = 0.995,
        max_grad_norm: float = 1.0,
        hidden_dim: int = 192,
        target_kl: float = 0.02,
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
        constraint_value_coeff: float = 0.25,
        risk_value_coeff: float = 0.15,
        response_coeff: float = 0.05,
        risk_coeff: float = 0.10,
        device: Optional[str] = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.constraint_dim = int(max(1, constraint_dim))
        self.response_dim = int(max(1, response_dim))
        self.net = ActorCritic(
            state_dim,
            action_dim,
            hidden=hidden_dim,
            constraint_dim=self.constraint_dim,
            response_dim=self.response_dim,
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
        self.constraint_value_coeff = float(max(0.0, constraint_value_coeff))
        self.risk_value_coeff = float(max(0.0, risk_value_coeff))
        self.response_coeff = float(max(0.0, response_coeff))
        self.risk_coeff = float(max(0.0, risk_coeff))
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
        decay_p = float(np.clip((p - 0.35) / 0.65, 0.0, 1.0))
        cosine = 0.5 * (1.0 + np.cos(np.pi * decay_p))
        scheduled_lr_cap = self.min_lr + (self.initial_max_lr - self.min_lr) * cosine
        if reward_converged:
            scheduled_lr_cap = min(scheduled_lr_cap, self.min_lr * 1.5)
        self.max_lr = float(np.clip(scheduled_lr_cap, self.min_lr, self.initial_max_lr))
        if self.curr_lr > self.max_lr:
            self.curr_lr = self.max_lr
            for group in self.opt.param_groups:
                group["lr"] = self.curr_lr
        
        early_clip = max(self.initial_clip_eps, 0.25)
        self.clip_eps = float(
            self.final_clip_eps + (early_clip - self.final_clip_eps) * cosine
        )
        if reward_converged:
            self.clip_eps = min(self.clip_eps, self.final_clip_eps)

    @torch.no_grad()
    def act(self, s_np: np.ndarray, deterministic: bool = False) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        s = torch.tensor(s_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        s = torch.nan_to_num(s, nan=0.0, posinf=1e3, neginf=-1e3)
        expected_dim = self.net.trunk[0].in_features
        if s.shape[-1] != expected_dim:
            raise ValueError(f"State dim mismatch: got {s.shape[-1]}, expected {expected_dim}")
        
        logits, value, constraint_values, risk_value, response_pred = self.net(s)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
        if not torch.isfinite(logits).all():
            logits = torch.zeros_like(logits)
        exploration_rate = 0.0 if deterministic else self.exploration_rate
        dist = self._exploratory_distribution(logits, exploration_rate)
        a = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        logp = dist.log_prob(a)
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
        return int(a.item()), s.squeeze(0), logp.squeeze(0), value.squeeze(0)
    
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
        constraint_tensor = torch.nan_to_num(constraint_tensor, nan=0.0, posinf=1.0, neginf=0.0).clamp(min=0.0)
        response_tensor = torch.nan_to_num(response_tensor, nan=0.0, posinf=1.0, neginf=-1.0)
        
        self.buf.append(
            Transition(
                s=s.detach(),
                a=torch.tensor(a, dtype=torch.long, device=self.device),
                r=float(r),
                done=bool(done),
                old_logp=old_logp.detach(),
                old_value=old_value.detach(),
                exploration_rate=float(
                    np.clip(getattr(self, "last_action_exploration_rate", 0.0), 0.0, 1.0)
                ),
                constraint_costs=constraint_tensor.detach(),
                risk_cost=float(max(0.0, np.nan_to_num(risk_cost, nan=0.0, posinf=1.0, neginf=0.0))),
                response_target=response_tensor.detach(),
                old_constraint_values=self.last_constraint_values.detach().clone(),
                old_risk_value=self.last_risk_value.detach().clone(),
            )
        )

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
        old_reward_values_all = torch.stack([tr.old_value for tr in self.buf], dim=0).detach()
        old_reward_values_all = torch.nan_to_num(old_reward_values_all, nan=0.0, posinf=0.0, neginf=0.0)
        adv, ret = self._gae_scalar(reward_all, old_reward_values_all, done_all)
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
                old_logp_b = old_logp_all[j].detach()
                old_value_b = old_value_all[j].detach()
                exploration_b = exploration_all[j].detach()

                logits, v, constraint_v, risk_v, response_pred = self.net(s_b)
                logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
                if not torch.isfinite(logits).all():
                    continue
                dist = self._exploratory_distribution(logits, exploration_b)
                policy_dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(a_b)
                
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
                entropy = dist.entropy().mean()
                policy_entropy = policy_dist.entropy().mean()
                logratio = logp - old_logp_b
                approx_kl = ((torch.exp(logratio) - 1.0) - logratio).mean()
                clipfrac = ((ratio - 1.0).abs() > self.clip_eps).float().mean()

                if float(approx_kl.item()) > 2.0 * self.target_kl:
                    stop_for_kl = True
                    break
                # Entropy regularizes the learned policy, not the externally
                # forced uniform mixture. Otherwise epsilon can hide collapse.
                loss = (
                    policy_loss
                    + self.v_coeff * value_loss
                    + self.constraint_value_coeff * constraint_value_loss
                    + self.risk_value_coeff * risk_value_loss
                    + self.response_coeff * response_loss
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
        with torch.no_grad():
            ret_var = torch.var(ret, unbiased=False)
            if torch.isfinite(ret_var) and float(ret_var.item()) > 1e-8:
                prediction_error = torch.var(ret - old_value_all, unbiased=False)
                last["explained_variance"] = float(
                    torch.clamp(1.0 - prediction_error / ret_var, -1.0, 1.0).item()
                )
                
        final_clipfrac = float(last.get("clipfrac", 0.0))
        final_kl = float(last.get("approx_kl", 0.0))
        if final_clipfrac > 0.40 or final_kl > 1.25 * self.target_kl:
            self.curr_lr = float(max(self.min_lr, self.curr_lr * self.lr_decay_on_spike))
            for g in self.opt.param_groups:
                g["lr"] = self.curr_lr
            self.low_update_streak = 0
        elif final_clipfrac < 0.03 and final_kl < 0.35 * self.target_kl:
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