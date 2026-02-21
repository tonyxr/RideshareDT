from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sun Feb  8 15:58:46 2026

@author: Xiaoru Shi

Actor-Critic with Wasserstein (Sinkhorn) trust region penalty for discrete policies.

This is designed for STABILITY:
- advantage normalization
- entropic Sinkhorn penalty to limit policy distribution shifts in Wasserstein geometry
- gradient clipping

"""

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.pi = nn.Linear(hidden, action_dim)
        self.v = nn.Linear(hidden, 1)

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.trunk(s)
        logits = self.pi(z)
        value = self.v(z).squeeze(-1)
        return logits, value
    
@dataclass
class Transition:
    s: torch.Tensor
    a: torch.Tensor
    r: float
    done: bool
    s_next: Optional[torch.Tensor]
    old_logits: torch.Tensor
    old_value: torch.Tensor


def sinkhorn_wasserstein(
    p: torch.Tensor,            # (B, A)
    q: torch.Tensor,            # (B, A)
    C: torch.Tensor,            # (A, A)
    epsilon: float = 0.1,
    n_iters: int = 30,
) -> torch.Tensor:
    """
    Differentiable entropic OT (Sinkhorn) distance between categorical distributions.
    """
    K = torch.exp(-C / epsilon).clamp_min(1e-9)  # (A,A)

    u = torch.ones_like(p) / p.size(-1)
    v = torch.ones_like(q) / q.size(-1)

    for _ in range(n_iters):
        Kv = torch.matmul(v, K.t())           # (B,A)
        u = p / (Kv + 1e-9)
        KTu = torch.matmul(u, K)              # (B,A)
        v = q / (KTu + 1e-9)

    P = u.unsqueeze(-1) * K.unsqueeze(0) * v.unsqueeze(1)   # (B,A,A)
    cost = (P * C.unsqueeze(0)).sum(dim=(1, 2))             # (B,)
    return cost.mean()


class WassersteinWPOAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        cost_matrix: np.ndarray,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        w_coeff: float = 0.5,     # Wasserstein penalty weight
        ent_coeff: float = 0.01,
        v_coeff: float = 0.5,
        epsilon: float = 0.1,
        max_grad_norm: float = 1.0,
        advantage_scale: float = 1.35,
        advantage_clip: float = 4.0,
        device: Optional[str] = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.net = ActorCritic(state_dim, action_dim).to(self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)

        self.gamma = gamma
        self.lam = lam
        self.w_coeff = w_coeff
        self.ent_coeff = ent_coeff
        self.v_coeff = v_coeff
        self.epsilon = epsilon
        self.max_grad_norm = max_grad_norm
        self.advantage_scale = advantage_scale
        self.advantage_clip = advantage_clip

        self.C = torch.tensor(cost_matrix, dtype=torch.float32, device=self.device)
        self.buf: List[Transition] = []

    @torch.no_grad()
    def act(self, s_np: np.ndarray) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        s = torch.tensor(s_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        expected_dim = self.net.trunk[0].in_features
        if s.shape[-1] != expected_dim:
            raise ValueError(f"State dim mismatch: got {s.shape[-1]}, expected {expected_dim}")
        logits, value = self.net(s)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        return int(a.item()), s.squeeze(0), logits.squeeze(0), value.squeeze(0)

    def store(self, s: torch.Tensor, a: int, r: float, done: bool, s_next: Optional[torch.Tensor],
              old_logits: torch.Tensor, old_value: torch.Tensor):
        self.buf.append(
            Transition(
                s=s.detach(),
                a=torch.tensor(a, dtype=torch.long, device=self.device),
                r=float(r),
                done=bool(done),
                s_next=None if s_next is None else s_next.detach(),
                old_logits=old_logits.detach(),
                old_value=old_value.detach(),
            )
        )

    @torch.no_grad()
    def _gae(self) -> Tuple[torch.Tensor, torch.Tensor]:
        T = len(self.buf)
        adv = torch.zeros(T, dtype=torch.float32, device=self.device)
        ret = torch.zeros(T, dtype=torch.float32, device=self.device)

        next_v = torch.tensor(0.0, device=self.device)
        if T > 0 and (self.buf[-1].s_next is not None) and (not self.buf[-1].done):
            _, next_v = self.net(self.buf[-1].s_next.unsqueeze(0))
            next_v = next_v.squeeze(0)

        gae = 0.0
        for t in reversed(range(T)):
            r_t = torch.tensor(self.buf[t].r, device=self.device)
            v_t = self.buf[t].old_value
            done = self.buf[t].done

            v_next = next_v if t == T - 1 else self.buf[t + 1].old_value
            if done:
                v_next = torch.tensor(0.0, device=self.device)

            delta = r_t + self.gamma * v_next - v_t
            gae = delta + self.gamma * self.lam * (0.0 if done else gae)
            adv[t] = gae
            ret[t] = adv[t] + v_t

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        adv = torch.clamp(adv * self.advantage_scale, -self.advantage_clip, self.advantage_clip)
        return adv, ret

    def update(self, epochs: int = 5, batch_size: int = 256) -> dict:
        if len(self.buf) == 0:
            return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "wdist": 0.0, "entropy": 0.0}

        adv, ret = self._gae()

        S = torch.stack([tr.s for tr in self.buf], dim=0)
        A = torch.stack([tr.a for tr in self.buf], dim=0)
        OLD_LOGITS = torch.stack([tr.old_logits for tr in self.buf], dim=0)
        p_old = torch.softmax(OLD_LOGITS, dim=-1).detach()

        N = S.size(0)
        last_loss = 0.0
        last_policy = 0.0
        last_value = 0.0
        last_wdist = 0.0
        last_entropy = 0.0

        for _ in range(epochs):
            idx = torch.randperm(N, device=self.device)
            for start in range(0, N, batch_size):
                j = idx[start:start + batch_size]
                s_b = S[j]
                a_b = A[j]
                adv_b = adv[j].detach()
                ret_b = ret[j].detach()
                p_old_b = p_old[j]

                logits, v = self.net(s_b)
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(a_b)
                entropy = dist.entropy().mean()

                # actor
                policy_loss = -(adv_b * logp).mean()

                # wasserstein trust region
                p_new = torch.softmax(logits, dim=-1)
                wdist = sinkhorn_wasserstein(p_old_b, p_new, self.C, epsilon=self.epsilon, n_iters=30)

                # critic
                value_loss = ((v - ret_b) ** 2).mean()

                loss = policy_loss + self.w_coeff * wdist + self.v_coeff * value_loss - self.ent_coeff * entropy

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.opt.step()
                
                last_loss = float(loss.item())
                last_policy = float(policy_loss.item())
                last_value = float(value_loss.item())
                last_wdist = float(wdist.item())
                last_entropy = float(entropy.item())

        self.buf.clear()
        return {
            "loss": last_loss,
            "policy_loss": last_policy,
            "value_loss": last_value,
            "wdist": last_wdist,
            "entropy": last_entropy,
        }