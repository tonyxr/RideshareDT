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
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 192):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.pi_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, action_dim),
        )
        self.v_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """Orthogonal init improves PPO stability and early optimization speed."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2.0))
                nn.init.constant_(module.bias, 0.0)

        # PPO convention: smaller final policy logits, unit-scale value head.
        if isinstance(self.pi_head[-1], nn.Linear):
            nn.init.orthogonal_(self.pi_head[-1].weight, gain=0.01)
            nn.init.constant_(self.pi_head[-1].bias, 0.0)
        if isinstance(self.v_head[-1], nn.Linear):
            nn.init.orthogonal_(self.v_head[-1].weight, gain=1.0)
            nn.init.constant_(self.v_head[-1].bias, 0.0)

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.trunk(s)
        return self.pi_head(z), self.v_head(z).squeeze(-1)
    
@dataclass
class Transition:
    s: torch.Tensor
    a: torch.Tensor
    r: float
    done: bool
    old_logp: torch.Tensor
    exploration_rate: float


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
        device: Optional[str] = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.net = ActorCritic(state_dim, action_dim, hidden=hidden_dim).to(self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)
        self.base_lr = float(lr)
        self.curr_lr = float(lr)
        self.min_lr = float(max(1e-6, min_lr))
        self.max_lr = float(max_lr if max_lr is not None else max(lr, 6e-4))

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
            self.ent_coeff = float(max(self.ent_coeff, 0.80 * self.max_ent_coeff))
            return

        if p <= 0.55 and not reward_converged:
            self.ent_coeff = float(max(self.ent_coeff, 0.55 * self.max_ent_coeff))
            return

        # The driver-supply environment is non-stationary and high variance, so
        # keep enough exploration early but decay more decisively after the
        # policy has seen a representative warmup window.  Persistently high
        # entropy was preventing convergence in longer driver-enabled runs.
        target = self.min_ent_coeff + (self.max_ent_coeff - self.min_ent_coeff) * max(0.0, 1.0 - p) ** 1.5
        if reward_converged:
            target = max(self.min_ent_coeff, 0.75 * target)
        self.ent_coeff = float(np.clip(min(self.ent_coeff, target), self.min_ent_coeff, self.max_ent_coeff))


    @torch.no_grad()
    def act(self, s_np: np.ndarray, deterministic: bool = False) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        s = torch.tensor(s_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        s = torch.nan_to_num(s, nan=0.0, posinf=1e3, neginf=-1e3)
        expected_dim = self.net.trunk[0].in_features
        if s.shape[-1] != expected_dim:
            raise ValueError(f"State dim mismatch: got {s.shape[-1]}, expected {expected_dim}")
        
        logits, value = self.net(s)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
        if not torch.isfinite(logits).all():
            logits = torch.zeros_like(logits)
        exploration_rate = 0.0 if deterministic else self.exploration_rate
        dist = self._exploratory_distribution(logits, exploration_rate)
        a = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        logp = dist.log_prob(a)
        self.last_action_exploration_rate = float(exploration_rate)
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
                exploration_rate=float(
                    np.clip(getattr(self, "last_action_exploration_rate", 0.0), 0.0, 1.0)
                ),
            )
        )

    @torch.no_grad()
    def _gae(self) -> Tuple[torch.Tensor, torch.Tensor]:
        t_size = len(self.buf)
        adv = torch.zeros(t_size, dtype=torch.float32, device=self.device)
        ret = torch.zeros(t_size, dtype=torch.float32, device=self.device)

        gae = torch.zeros((), dtype=torch.float32, device=self.device)
        next_v = torch.zeros((), dtype=torch.float32, device=self.device)
        for t in reversed(range(t_size)):
            tr = self.buf[t]
            v_t = tr.old_value
            continuation = 0.0 if tr.done else 1.0
            delta = tr.r + self.gamma * next_v * continuation - v_t
            gae = delta + self.gamma * self.lam * continuation * gae
            adv[t] = gae
            ret[t] = adv[t] + v_t
            next_v = v_t

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
                "entropy": 0.0,
                "policy_entropy": 0.0,
                "policy_entropy_fraction": float(self.last_policy_entropy_fraction),
                "exploration_rate": float(self.exploration_rate),
                "action_coverage": self._action_coverage(),
            }

        adv, ret = self._gae()

        s_all = torch.stack([tr.s for tr in self.buf], dim=0)
        s_all = torch.nan_to_num(s_all, nan=0.0, posinf=1e3, neginf=-1e3)
        a_all = torch.stack([tr.a for tr in self.buf], dim=0)
        old_logp_all = torch.stack([tr.old_logp for tr in self.buf], dim=0)
        old_logp_all = torch.nan_to_num(old_logp_all, nan=0.0, posinf=20.0, neginf=-20.0)
        old_value_all = torch.stack([tr.old_value for tr in self.buf], dim=0).detach()
        old_value_all = torch.nan_to_num(old_value_all, nan=0.0, posinf=0.0, neginf=0.0)
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
            "entropy": 0.0,
            "policy_entropy": 0.0,
            "policy_entropy_fraction": float(self.last_policy_entropy_fraction),
            "approx_kl": 0.0,
            "clipfrac": 0.0,
            "explained_variance": 0.0,
            "ent_coeff": float(self.ent_coeff),
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
                adv_b = adv[j].detach().clamp(-self.adv_clip, self.adv_clip)
                ret_b = ret[j].detach()
                old_logp_b = old_logp_all[j].detach()
                old_value_b = old_value_all[j].detach()
                exploration_b = exploration_all[j].detach()

                logits, v = self.net(s_b)
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
                entropy = dist.entropy().mean()
                policy_entropy = policy_dist.entropy().mean()
                logratio = logp - old_logp_b
                approx_kl = ((torch.exp(logratio) - 1.0) - logratio).mean()
                clipfrac = ((ratio - 1.0).abs() > self.clip_eps).float().mean()

                if float(approx_kl.item()) > 1.5 * self.target_kl:
                    stop_for_kl = True
                    break
                # Entropy regularizes the learned policy, not the externally
                # forced uniform mixture. Otherwise epsilon can hide collapse.
                loss = policy_loss + self.v_coeff * value_loss - self.ent_coeff * policy_entropy
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
        if final_clipfrac > 0.30 or final_kl > self.target_kl:
            self.curr_lr = float(max(self.min_lr, self.curr_lr * self.lr_decay_on_spike))
            for g in self.opt.param_groups:
                g["lr"] = self.curr_lr
            self.low_update_streak = 0
        elif final_clipfrac < 0.02 and final_kl < 0.25 * self.target_kl:
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