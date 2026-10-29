"""Public-observation concave two-lever value surface for CRISP-Ride V17."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
from torch import nn


class V17ConcaveValueModel(nn.Module):
    """Encode a public observation window and evaluate concave 2-D surfaces.

    Every surface uses six state-dependent numbers: an intercept, two linear
    terms, and a lower-triangular 2x2 factor.  ``C = L L^T + floor I`` is
    therefore positive definite for every observation, making Q concave in
    the distance/time levers while retaining their interaction.
    """

    def __init__(
        self,
        *,
        observation_dim: int,
        frame_stack: int,
        hidden_dim: int = 64,
        auxiliary_names: Sequence[str] = (),
        curvature_floor: float = 1e-3,
    ) -> None:
        super().__init__()
        self.observation_dim = int(observation_dim)
        self.frame_stack = int(frame_stack)
        self.hidden_dim = int(hidden_dim)
        self.auxiliary_names: Tuple[str, ...] = tuple(
            str(name) for name in auxiliary_names
        )
        self.curvature_floor = float(curvature_floor)
        if self.observation_dim <= 0 or self.frame_stack <= 0:
            raise ValueError("observation dimensions must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.curvature_floor <= 0.0:
            raise ValueError("curvature_floor must be positive")
        if len(set(self.auxiliary_names)) != len(self.auxiliary_names):
            raise ValueError("auxiliary names must be unique")

        self.frame_encoder = nn.Sequential(
            nn.Linear(self.observation_dim, self.hidden_dim),
            nn.Tanh(),
        )
        self.recurrent_encoder = nn.GRU(
            self.hidden_dim, self.hidden_dim, batch_first=True
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
        )
        self.surface_head = nn.Linear(
            self.hidden_dim, self.surface_count * 6
        )
        self._initialize()

    @property
    def surface_count(self) -> int:
        return 1 + len(self.auxiliary_names)

    def _initialize(self) -> None:
        for module in (self.frame_encoder[0], self.state_encoder[0]):
            nn.init.orthogonal_(module.weight, gain=2.0 ** 0.5)
            nn.init.zeros_(module.bias)
        for name, parameter in self.recurrent_encoder.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(parameter)
            elif "weight_hh" in name:
                nn.init.orthogonal_(parameter)
            else:
                nn.init.zeros_(parameter)
        # The registered warm start has b=0, so symmetric curvature puts the
        # initial maximum exactly at the published card (1,1).
        # Keep the diagonal Cholesky entries non-zero: C = L L^T has zero
        # derivative at L=0, which would otherwise make curvature impossible
        # to learn from the warm start.  The small symmetric diagonal retains
        # the unique card argmax while giving both curvature directions a
        # usable first gradient.
        nn.init.zeros_(self.surface_head.weight)
        nn.init.zeros_(self.surface_head.bias)
        with torch.no_grad():
            surface_bias = self.surface_head.bias.reshape(
                self.surface_count, 6
            )
            surface_bias[:, 3] = 0.10
            surface_bias[:, 5] = 0.10

    def _public_sequence(self, observations: torch.Tensor) -> torch.Tensor:
        values = torch.nan_to_num(
            observations, nan=0.0, posinf=1e3, neginf=-1e3
        )
        flat_dim = self.frame_stack * self.observation_dim
        if values.ndim == 1:
            if values.numel() != flat_dim:
                raise ValueError("flat public observation has the wrong size")
            return values.reshape(1, self.frame_stack, self.observation_dim)
        if values.ndim == 2:
            if tuple(values.shape) == (
                self.frame_stack,
                self.observation_dim,
            ):
                return values.unsqueeze(0)
            if values.shape[-1] == flat_dim:
                return values.reshape(
                    values.shape[0], self.frame_stack, self.observation_dim
                )
        if values.ndim == 3 and tuple(values.shape[-2:]) == (
            self.frame_stack,
            self.observation_dim,
        ):
            return values
        raise ValueError(
            "public observations must be flat or [batch, frames, features]"
        )

    def encode(self, observations: torch.Tensor) -> torch.Tensor:
        sequence = self._public_sequence(observations)
        frames = self.frame_encoder(sequence)
        _, hidden = self.recurrent_encoder(frames)
        return self.state_encoder(hidden[-1])

    def surface_parameters(
        self, observations: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = self.encode(observations)
        raw = self.surface_head(latent).reshape(
            latent.shape[0], self.surface_count, 6
        )
        intercept = raw[..., 0]
        linear = raw[..., 1:3]
        lower = raw.new_zeros((*raw.shape[:2], 2, 2))
        lower[..., 0, 0] = raw[..., 3]
        lower[..., 1, 0] = raw[..., 4]
        lower[..., 1, 1] = raw[..., 5]
        identity = torch.eye(2, dtype=raw.dtype, device=raw.device)
        curvature = lower @ lower.transpose(-1, -2)
        curvature = curvature + self.curvature_floor * identity
        return intercept, linear, curvature

    def forward(
        self,
        observations: torch.Tensor,
        cells: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if cells.ndim != 2 or tuple(cells.shape[-1:]) != (2,):
            raise ValueError("cells must have shape [cell, 2]")
        intercept, linear, curvature = self.surface_parameters(observations)
        coordinates = torch.nan_to_num(
            cells.to(dtype=intercept.dtype, device=intercept.device),
            nan=1.0,
            posinf=1.0,
            neginf=1.0,
        ) - 1.0
        linear_value = torch.einsum("bsi,ci->bsc", linear, coordinates)
        quadratic = torch.einsum(
            "ci,bsij,cj->bsc", coordinates, curvature, coordinates
        )
        surfaces = intercept.unsqueeze(-1) + linear_value - 0.5 * quadratic
        return {
            "q": surfaces[:, 0, :],
            "auxiliary": surfaces[:, 1:, :].transpose(1, 2),
            "intercept": intercept,
            "linear": linear,
            "curvature": curvature,
        }

