"""Shared helpers for the CRISP-Ride V17 robustness workflow."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts import v17_results_eval
from v17_objective import profit_report


BASE = Path(__file__).resolve().parents[1]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_option(command: Sequence[str], flag: str, value: Any) -> list[str]:
    """Replace one value option or append it without creating duplicates."""
    result = [str(item) for item in command]
    positions = [index for index, item in enumerate(result) if item == flag]
    if len(positions) > 1:
        raise ValueError(f"command contains duplicate option {flag}")
    if positions:
        index = positions[0]
        if index + 1 >= len(result) or result[index + 1].startswith("--"):
            raise ValueError(f"command option {flag} has no value")
        result[index + 1] = str(value)
    else:
        result.extend((str(flag), str(value)))
    return result


def check_overrides(
    payload: Mapping[str, Any], overrides: Mapping[str, Any]
) -> None:
    """Require every requested market override to be present in a product."""
    v26r = dict(payload.get("v26r", {}) or {})
    supply = dict(v26r.get("effective_supply_config", {}) or {})
    memory = dict(v26r.get("memory_rival_overrides", {}) or {})
    model = dict(payload.get("model_switches", {}) or {})
    locations = {
        "supply_eta_v3": (supply, "eta_v3"),
        "eta_v3": (supply, "eta_v3"),
        "shortage_willingness_share": (
            supply, "shortage_willingness_share"
        ),
        "memory_rival_hold_mean_days": (memory, "hold_mean_days"),
        "hold_mean_days": (memory, "hold_mean_days"),
        "memory_rival_retaliation": (memory, "retaliation"),
        "retaliation": (memory, "retaliation"),
        "acceptance_shock_wage": (model, "acceptance_shock_wage"),
        "rate_upper_multiple": (model, "rate_upper_multiple"),
        "loyalty_model": (model, "loyalty_model"),
        "unfulfilled_spillover": (model, "unfulfilled_spillover"),
        "threshold_ablation": (model, "threshold_ablation"),
    }
    for raw_key, expected in overrides.items():
        key = str(raw_key).lstrip("-").replace("-", "_")
        if key == "disable_spillover":
            key = "unfulfilled_spillover"
            expected = not bool(expected)
        if key not in locations:
            raise ValueError(f"unknown robustness override {raw_key!r}")
        container, field = locations[key]
        if field not in container:
            raise ValueError(
                f"robustness product omitted override field {key!r}"
            )
        actual = container[field]
        if isinstance(expected, (float, int)) and not isinstance(expected, bool):
            if abs(float(actual) - float(expected)) > 1e-12:
                raise ValueError(
                    f"robustness override mismatch for {key}: "
                    f"{actual!r} != {expected!r}"
                )
        elif str(actual) != str(expected):
            raise ValueError(
                f"robustness override mismatch for {key}: "
                f"{actual!r} != {expected!r}"
            )


def robust_sources(extra_paths: Sequence[Path | str] = ()) -> dict[str, str]:
    """Return the complete content identity for robustness products."""
    sources = dict(v17_results_eval.source_identity())
    required = [BASE / "v17_surge_decay.py"]
    required.extend(sorted((BASE / "scripts").glob("v17_robust_*.py")))
    required.extend(Path(path).resolve() for path in extra_paths)
    for path in required:
        resolved = Path(path).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        try:
            key = str(resolved.relative_to(BASE))
        except ValueError:
            key = str(resolved)
        sources[key] = _sha256_file(resolved)
    return dict(sorted(sources.items()))


def final_objective(own: float, rival: float) -> float:
    """Return the registered own-profit minus 0.75 rival-profit objective."""
    return float(profit_report(float(own), float(rival))["final_objective"])


def schedule_spec(
    event_level: float, scenario: Mapping[str, Any]
) -> str:
    """Build the registered full-card-compatible event-day surge schedule."""
    from v17_surge_decay import scenario_windows

    event_days: set[int] = set()
    for start, end in scenario_windows(scenario):
        event_days.update(range(int(start), int(end) + 1))
    payload = {
        "default": [1.0, 1.0],
        "days": {
            str(day): [float(event_level), 1.0]
            for day in sorted(event_days)
        },
    }
    return "schedule:" + json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    )
