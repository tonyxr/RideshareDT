"""Addressable exogenous random-number tape for V17 counterfactual branches.

The tape has no mutable cursor. A stream is identified entirely by
``(root_seed, tape_id, day, step, purpose)`` so restoring simulator state and
requesting the same address reproduces the same draws exactly. Different
purposes never share a generator state.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


TAPE_PURPOSES: Tuple[str, ...] = (
    "calendar",
    "hours",
    "weather",
    "requests",
    "profiles",
    "trip",
    "shock",
    "endo",
)


@dataclass(frozen=True)
class ExogenousTape:
    root_seed: int
    tape_id: int = 0

    def __post_init__(self) -> None:
        if int(self.root_seed) < 0:
            raise ValueError("tape root_seed must be non-negative")
        if int(self.tape_id) < 0:
            raise ValueError("tape_id must be non-negative")

    @staticmethod
    def _purpose_code(purpose: str) -> int:
        name = str(purpose)
        if name not in TAPE_PURPOSES:
            raise ValueError(
                f"unknown tape purpose {name!r}; expected one of {TAPE_PURPOSES}"
            )
        digest = hashlib.sha256(name.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "little", signed=False)

    def gen(
        self,
        day: int,
        step: int,
        purpose: str,
        *,
        tape_id: int | None = None,
    ) -> np.random.Generator:
        day_i, step_i = int(day), int(step)
        tape_i = int(self.tape_id if tape_id is None else tape_id)
        if day_i < 0 or step_i < 0 or tape_i < 0:
            raise ValueError("tape day, step, and tape_id must be non-negative")
        sequence = np.random.SeedSequence(
            [
                int(self.root_seed),
                tape_i,
                day_i,
                step_i,
                self._purpose_code(purpose),
                17,
            ]
        )
        return np.random.default_rng(sequence)

    def state_dict(self) -> Dict[str, int]:
        return {"root_seed": int(self.root_seed), "tape_id": int(self.tape_id)}

    @classmethod
    def from_state_dict(cls, payload: Dict[str, int]) -> "ExogenousTape":
        return cls(
            root_seed=int(payload["root_seed"]),
            tape_id=int(payload.get("tape_id", 0)),
        )

    def fingerprint(self) -> str:
        material = f"v17-tape-v1|{int(self.root_seed)}|{int(self.tape_id)}"
        return hashlib.sha256(material.encode("utf-8")).hexdigest()
