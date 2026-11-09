"""Timed two-lever surge with hold and linear decay (V17 robustness comparator).

Pure functions shared by :mod:`Core` (the ``surge_decay:`` policy-sweep
specification) and the robustness runners.  Event timing is privileged:
windows come directly from the scenario.
"""

from __future__ import annotations

import json
from typing import Iterable, Mapping, Sequence


SURGE_EVENT_KINDS = ("driver_shortage", "winter_storm")


def surge_decay_weight(
    day: int,
    windows: Iterable[Sequence[int]],
    hold_days: int,
    decay_days: int,
) -> float:
    """Return the event weight for ``day`` under hold and stepped decay."""
    hold, decay = max(0, int(hold_days)), max(0, int(decay_days))
    weight = 0.0
    for start, end in windows:
        start, end = int(start), int(end)
        if start <= int(day) <= end + hold:
            return 1.0
        k = int(day) - end - hold
        if 1 <= k <= decay:
            weight = max(weight, 1.0 - k / (decay + 1.0))
    return weight


def surge_decay_cell(
    day: int,
    normal: Sequence[float],
    event: Sequence[float],
    windows: Iterable[Sequence[int]],
    hold_days: int,
    decay_days: int,
) -> tuple[float, float]:
    """Interpolate the two-lever tariff cell at ``day``."""
    weight = surge_decay_weight(day, windows, hold_days, decay_days)
    return (
        float(normal[0]) + weight * (float(event[0]) - float(normal[0])),
        float(normal[1]) + weight * (float(event[1]) - float(normal[1])),
    )


def surge_decay_spec(
    normal: Sequence[float],
    event: Sequence[float],
    windows: Iterable[Sequence[int]],
    hold_days: int,
    decay_days: int,
) -> str:
    """Encode a deterministic, content-addressable policy-sweep spec."""
    payload = {
        "normal": [float(normal[0]), float(normal[1])],
        "event": [float(event[0]), float(event[1])],
        "windows": [[int(start), int(end)] for start, end in windows],
        "hold_days": int(hold_days),
        "decay_days": int(decay_days),
    }
    return "surge_decay:" + json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    )


def scenario_windows(
    scenario: Mapping[str, object],
    kinds: Sequence[str] = SURGE_EVENT_KINDS,
) -> list[list[int]]:
    """Extract sorted shortage/storm windows from a scenario mapping."""
    accepted = {str(kind) for kind in kinds}
    return sorted(
        [int(event["start_day"]), int(event["end_day"])]
        for event in scenario.get("events", ())
        if isinstance(event, Mapping) and str(event.get("kind")) in accepted
    )
