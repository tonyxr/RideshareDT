#!/usr/bin/env python3
"""Summarise Revision 14 NMCC signal-probe records and apply the Phase-3 gate."""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ACTION_RE = re.compile(
    r"^(base_fare|booking_fee|per_minute|per_mile|airport_fee):"
    r"([0-9]+(?:\.[0-9]+)?)x_anchor$"
)


def _mean_se(values: Iterable[float]) -> dict[str, float | int | None]:
    xs = [float(x) for x in values if math.isfinite(float(x))]
    if not xs:
        return {"mean": None, "se": None, "n": 0}
    se = st.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else None
    return {"mean": st.mean(xs), "se": se, "n": len(xs)}


def _regime(record: dict[str, Any]) -> str:
    active = record.get("regime", [])
    if isinstance(active, str):
        active = [active] if active else []
    if "driver_shortage" in active:
        return "shortage"
    if not active:
        return "calm"
    return "other"


def _classify(label: str, ratios: dict[str, Any], tolerance: float) -> str:
    if label == "hold":
        return "same"
    match = ACTION_RE.match(label)
    if not match:
        raise ValueError(f"unrecognised independent-action label: {label!r}")
    coefficient, target_text = match.groups()
    target = float(target_text)
    current = float(ratios[coefficient])
    if target > current + tolerance:
        return "raise"
    if target < current - tolerance:
        return "cut"
    return "same"


def analyze(path: Path, tolerance: float = 0.01) -> dict[str, Any]:
    table: dict[tuple[str, str], list[float]] = defaultdict(list)
    per_decision_sd: dict[str, list[float]] = defaultdict(list)
    best_raise: dict[str, list[float]] = defaultdict(list)
    decision_raise: dict[str, list[float]] = defaultdict(list)
    records = 0
    hold_errors: list[dict[str, Any]] = []
    sampled_missing: list[dict[str, Any]] = []

    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            values = {str(k): float(v) for k, v in record["values"].items()}
            ratios = {str(k): float(v) for k, v in record["firm1_ratios"].items()}
            regime = _regime(record)
            records += 1
            if "hold" not in values or abs(values["hold"]) > 1e-7:
                hold_errors.append({"line": line_number, "value": values.get("hold")})
            selected = str(record.get("selected", ""))
            if selected not in values:
                sampled_missing.append({"line": line_number, "selected": selected})

            classified: list[tuple[str, str, float]] = []
            for label, advantage in values.items():
                action_class = _classify(label, ratios, tolerance)
                table[(action_class, regime)].append(advantage)
                classified.append((label, action_class, advantage))
            if len(classified) > 1:
                per_decision_sd[regime].append(
                    st.pstdev(item[2] for item in classified)
                )
            raises = [item[2] for item in classified if item[1] == "raise"]
            if raises:
                decision_raise[regime].append(st.mean(raises))
            best = max(classified, key=lambda item: (item[2], item[0]))
            best_raise[regime].append(float(best[1] == "raise"))

    table_report = {
        action_class: {
            regime: _mean_se(table[(action_class, regime)])
            for regime in ("shortage", "calm", "other")
        }
        for action_class in ("raise", "cut", "same")
    }
    shortage = decision_raise["shortage"]
    calm = decision_raise["calm"]
    difference = (
        st.mean(shortage) - st.mean(calm) if shortage and calm else None
    )
    if len(shortage) > 1 and len(calm) > 1:
        se = math.sqrt(
            st.variance(shortage) / len(shortage)
            + st.variance(calm) / len(calm)
        )
    else:
        se = None
    t_value = (
        difference / se
        if difference is not None and se is not None and se > 0
        else math.copysign(math.inf, difference)
        if difference is not None and se == 0 and difference != 0
        else None
    )
    gate = {
        "shortage_mean_raise_advantage": (
            st.mean(shortage) if shortage else None
        ),
        "calm_mean_raise_advantage": st.mean(calm) if calm else None,
        "difference": difference,
        "se": se,
        "t": t_value,
        "n_shortage": len(shortage),
        "n_calm": len(calm),
        "pass": bool(
            difference is not None
            and difference > 0.0
            and t_value is not None
            and t_value >= 3.0
        ),
    }
    return {
        "source": str(path.resolve()),
        "records": records,
        "candidate_tolerance": tolerance,
        "invariants": {
            "hold_exactly_zero": not hold_errors,
            "sampled_action_present": not sampled_missing,
            "hold_errors": hold_errors[:20],
            "sampled_missing": sampled_missing[:20],
        },
        "class_by_regime": table_report,
        "per_decision_advantage_sd": {
            regime: _mean_se(per_decision_sd[regime])
            for regime in ("shortage", "calm", "other")
        },
        "best_candidate_raise_share": {
            regime: _mean_se(best_raise[regime])
            for regime in ("shortage", "calm", "other")
        },
        "gate": gate,
        "pass": bool(
            not hold_errors and not sampled_missing and gate["pass"]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dump", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tolerance", type=float, default=0.01)
    args = parser.parse_args()
    report = analyze(args.dump, args.tolerance)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
