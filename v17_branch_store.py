"""Persistent, deterministic replay store for V17 branch labels.

The store is intentionally independent of PPO rollout buffers: optimizer
updates may consume or clear those buffers, but branch rows remain available
until displaced by the fixed-capacity ring.  Its complete sampling state is
serializable so a training resume produces the same next fitted batch.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np


BRANCH_STORE_SCHEMA = "v17-branch-store-v1"


class BranchReplayStore:
    """Fixed-capacity ring with deterministic regime-balanced sampling."""

    def __init__(
        self,
        capacity: int,
        *,
        seed: int,
        model_version: str = "v17-value-v1",
    ) -> None:
        self.capacity = int(capacity)
        if self.capacity <= 0:
            raise ValueError("branch-store capacity must be positive")
        self.model_version = str(model_version)
        if not self.model_version:
            raise ValueError("branch-store model_version must be non-empty")
        self._rows: List[Dict[str, Any]] = []
        self._next_index = 0
        self.total_added = 0
        self._rng = np.random.default_rng(int(seed))

    def __len__(self) -> int:
        return len(self._rows)

    @staticmethod
    def _normalize_regime(row: Mapping[str, Any]) -> str:
        label = str(row.get("regime_label", "")).strip()
        return label or "calm"

    def add(self, row: Mapping[str, Any]) -> None:
        payload = copy.deepcopy(dict(row))
        payload["regime_label"] = self._normalize_regime(payload)
        payload["model_version"] = self.model_version
        if len(self._rows) < self.capacity:
            self._rows.append(payload)
        else:
            self._rows[self._next_index] = payload
        self._next_index = (self._next_index + 1) % self.capacity
        self.total_added += 1

    def rows(self) -> List[Dict[str, Any]]:
        """Return retained rows in chronological order, oldest first."""
        if len(self._rows) < self.capacity or self._next_index == 0:
            ordered = self._rows
        else:
            ordered = (
                self._rows[self._next_index :]
                + self._rows[: self._next_index]
            )
        return copy.deepcopy(ordered)

    def sample(
        self,
        batch_size: int,
        *,
        regime_balanced: bool = True,
        copy_rows: bool = True,
    ) -> List[Dict[str, Any]]:
        """Draw a batch; ``copy_rows=False`` returns the stored rows themselves.

        The random draws are identical either way. Use ``copy_rows=False`` only
        for read-only consumers (the daily value fit), which skips one deep copy
        per sampled row.
        """
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not self._rows:
            raise ValueError("cannot sample an empty branch store")
        if not regime_balanced:
            replace = batch_size > len(self._rows)
            indices = self._rng.choice(
                len(self._rows), size=batch_size, replace=replace
            )
        else:
            groups: Dict[str, List[int]] = defaultdict(list)
            for index, row in enumerate(self._rows):
                groups[self._normalize_regime(row)].append(index)
            labels = sorted(groups)
            base, extra = divmod(batch_size, len(labels))
            chosen: List[int] = []
            for position, label in enumerate(labels):
                count = base + int(position < extra)
                if count == 0:
                    continue
                pool = groups[label]
                sampled = self._rng.choice(
                    pool, size=count, replace=count > len(pool)
                )
                chosen.extend(int(index) for index in sampled)
            indices = np.asarray(chosen, dtype=np.int64)
            if len(indices) > 1:
                indices = indices[self._rng.permutation(len(indices))]
        if not copy_rows:
            return [self._rows[int(index)] for index in indices]
        return [copy.deepcopy(self._rows[int(index)]) for index in indices]

    def state_dict(self) -> Dict[str, Any]:
        return {
            "schema": BRANCH_STORE_SCHEMA,
            "capacity": int(self.capacity),
            "model_version": self.model_version,
            "rows": copy.deepcopy(self._rows),
            "next_index": int(self._next_index),
            "total_added": int(self.total_added),
            "rng_state": copy.deepcopy(self._rng.bit_generator.state),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        payload = dict(state)
        if payload.get("schema") != BRANCH_STORE_SCHEMA:
            raise ValueError("unsupported branch-store schema")
        if int(payload.get("capacity", -1)) != self.capacity:
            raise ValueError("branch-store capacity does not match runtime")
        if str(payload.get("model_version", "")) != self.model_version:
            raise ValueError("branch-store model version does not match runtime")
        rows = copy.deepcopy(list(payload.get("rows", [])))
        if len(rows) > self.capacity:
            raise ValueError("branch-store state exceeds runtime capacity")
        for row in rows:
            if str(row.get("model_version", "")) != self.model_version:
                raise ValueError("branch row model version does not match runtime")
        next_index = int(payload.get("next_index", 0))
        if not 0 <= next_index < self.capacity:
            raise ValueError("invalid branch-store ring position")
        if len(rows) < self.capacity and next_index != len(rows):
            raise ValueError("invalid branch-store partial-ring position")
        self._rows = rows
        self._next_index = next_index
        self.total_added = int(payload.get("total_added", len(rows)))
        self._rng.bit_generator.state = copy.deepcopy(payload["rng_state"])

