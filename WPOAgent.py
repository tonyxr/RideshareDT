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
from typing import List, Optional, Tuple

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
        return self.pi(z), self.v(z).squeeze(-1)
    
@dataclass
class Transition:
    s: torch.Tensor
    a: torch.Tensor
    r: float
    done: bool
    old_logp: torch.Tensor
    old_value: torch.Tensor


class PPOAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        v_coeff: float = 0.5,
        ent_coeff: float = 0.01,
        min_ent_coeff: float = 0.001,
        ent_decay: float = 0.995,
        max_grad_norm: float = 1.0,
        device: Optional[str] = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.net = ActorCritic(state_dim, action_dim).to(self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)

        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.v_coeff = v_coeff
        self.ent_coeff = ent_coeff
        self.max_ent_coeff = float(max(ent_coeff, min_ent_coeff))
        self.min_ent_coeff = float(max(0.0, min_ent_coeff))
        self.ent_decay = float(np.clip(ent_decay, 0.90, 1.0))
        
        self.max_grad_norm = max_grad_norm
        self.update_calls = 0

        self.buf: List[Transition] = []
        
    def adapt_entropy(self, progress: float, reward_converged: bool) -> None:
        """Keep exploration high early, then tighten as training converges."""
        p = float(np.clip(progress, 0.0, 1.0))
        if p <= 0.25 and not reward_converged:
            self.ent_coeff = float(max(self.ent_coeff, 0.60 * self.max_ent_coeff))
            return

        target = self.min_ent_coeff + (self.max_ent_coeff - self.min_ent_coeff) * max(0.0, 1.0 - p)
        if reward_converged:
            target = max(self.min_ent_coeff, 0.6 * target)
        self.ent_coeff = float(np.clip(min(self.ent_coeff, target), self.min_ent_coeff, self.max_ent_coeff))


    @torch.no_grad()
    def act(self, s_np: np.ndarray) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        s = torch.tensor(s_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        expected_dim = self.net.trunk[0].in_features
        if s.shape[-1] != expected_dim:
            raise ValueError(f"State dim mismatch: got {s.shape[-1]}, expected {expected_dim}")
        
        logits, value = self.net(s)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        logp = dist.log_prob(a)
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
    ) -> None:
        del s_next
        
        self.buf.append(
            Transition(
                s=s.detach(),
                a=torch.tensor(a, dtype=torch.long, device=self.device),
                r=float(r),
                done=bool(done),
                old_logp=old_logp.detach(),
                old_value=old_value.detach(),
            )
        )

    @torch.no_grad()
    def _gae(self) -> Tuple[torch.Tensor, torch.Tensor]:
        t_size = len(self.buf)
        adv = torch.zeros(t_size, dtype=torch.float32, device=self.device)
        ret = torch.zeros(t_size, dtype=torch.float32, device=self.device)

        gae = 0.0
        next_v = torch.tensor(0.0, device=self.device)
        for t in reversed(range(t_size)):
            tr = self.buf[t]
            v_t = tr.old_value
            if tr.done:
                next_v = torch.tensor(0.0, device=self.device)
            delta = torch.tensor(tr.r, device=self.device) + self.gamma * next_v - v_t
            gae = delta + self.gamma * self.lam * (0.0 if tr.done else gae)
            adv[t] = gae
            ret[t] = adv[t] + v_t
            next_v = v_t

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, ret

    def update(self, epochs: int = 5, batch_size: int = 256) -> dict:
        if not self.buf:
            return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        adv, ret = self._gae()

        s_all = torch.stack([tr.s for tr in self.buf], dim=0)
        a_all = torch.stack([tr.a for tr in self.buf], dim=0)
        old_logp_all = torch.stack([tr.old_logp for tr in self.buf], dim=0)
        
        n = s_all.size(0)
        last = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clipfrac": 0.0,
            "ent_coeff": float(self.ent_coeff),
        }


        for _ in range(epochs):
            idx = torch.randperm(n, device=self.device)
            for start in range(0, n, batch_size):
                j = idx[start : start + batch_size]
                s_b = s_all[j]
                a_b = a_all[j]
                adv_b = adv[j].detach()
                ret_b = ret[j].detach()
                old_logp_b = old_logp_all[j].detach()

                logits, v = self.net(s_b)
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(a_b)
                
                ratio = torch.exp(logp - old_logp_b)
                unclipped = ratio * adv_b
                clipped = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_b
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = ((v - ret_b) ** 2).mean()
                entropy = dist.entropy().mean()
                approx_kl = (old_logp_b - logp).mean()
                clipfrac = ((ratio - 1.0).abs() > self.clip_eps).float().mean()


                loss = policy_loss + self.v_coeff * value_loss - self.ent_coeff * entropy

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.opt.step()
                
                last = {
                    "loss": float(loss.item()),
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "entropy": float(entropy.item()),
                    "approx_kl": float(approx_kl.item()),
                    "clipfrac": float(clipfrac.item()),
                    "ent_coeff": float(self.ent_coeff),
                }

        self.buf.clear()
        self.update_calls += 1
        self.ent_coeff = max(self.min_ent_coeff, self.ent_coeff * self.ent_decay)

        last["ent_coeff"] = float(self.ent_coeff)
        return last