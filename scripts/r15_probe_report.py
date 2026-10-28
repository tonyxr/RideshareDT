#!/usr/bin/env python3
"""Summarise Revision 15 NMCC level-step probe records and apply the probe gate.

Each record holds hold-centred branch values for every feasible level step. A
step is a raise when its delta is positive and a cut when negative. Gate: the
mean raise advantage on shortage days exceeds the calm-day mean with t >= 3.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

LABEL_RE = re.compile(r"^level_step:([+-][0-9]+\.[0-9]+)$")


def mean_se(values: Iterable[float]) -> dict[str, Any]:
    xs = [float(x) for x in values if math.isfinite(float(x))]
    if not xs:
        return {"mean": None, "se": None, "n": 0}
    return {
        "mean": st.mean(xs),
        "se": st.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else None,
        "n": len(xs),
    }


def regime(record: dict[str, Any]) -> str:
    active = record.get("regime", [])
    if isinstance(active, str):
        active = [active] if active else []
    if "driver_shortage" in active:
        return "shortage"
    if not active:
        return "calm"
    return "other"


def classify(label: str) -> tuple[str, float]:
    if label == "hold":
        return "hold", 0.0
    match = LABEL_RE.match(label)
    if not match:
        raise ValueError(f"unrecognised level-step label: {label!r}")
    delta = float(match.group(1))
    return ("raise" if delta > 0 else "cut"), delta


def analyze(path: Path) -> dict[str, Any]:
    by_step: dict[tuple[str, str], list[float]] = defaultdict(list)
    decision_raise: dict[str, list[float]] = defaultdict(list)
    decision_cut: dict[str, list[float]] = defaultdict(list)
    best_class: dict[str, list[str]] = defaultdict(list)
    per_decision_sd: dict[str, list[float]] = defaultdict(list)
    levels: dict[str, list[float]] = defaultdict(list)
    hold_errors, sampled_missing, short_rows = [], [], []
    records = 0
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            records += 1
            values = {str(k): float(v) for k, v in record["values"].items()}
            label_regime = regime(record)
            if abs(values.get("hold", float("nan"))) > 1e-7 or "hold" not in values:
                hold_errors.append({"line": line_number, "hold": values.get("hold")})
            if str(record.get("selected", "")) not in values:
                sampled_missing.append({"line": line_number, "selected": record.get("selected")})
            if len(values) < 2:
                short_rows.append(line_number)
                continue
            if record.get("firm1_tariff_level") is not None:
                levels[label_regime].append(float(record["firm1_tariff_level"]))
            raises, cuts = [], []
            classified = []
            for label, advantage in values.items():
                action_class, delta = classify(label)
                classified.append((advantage, action_class, label))
                by_step[(label, label_regime)].append(advantage)
                if action_class == "raise":
                    raises.append(advantage)
                elif action_class == "cut":
                    cuts.append(advantage)
            if raises:
                decision_raise[label_regime].append(st.mean(raises))
            if cuts:
                decision_cut[label_regime].append(st.mean(cuts))
            per_decision_sd[label_regime].append(st.pstdev([item[0] for item in classified]))
            best_class[label_regime].append(max(classified)[1])
    shortage, calm = decision_raise["shortage"], decision_raise["calm"]
    difference = st.mean(shortage) - st.mean(calm) if shortage and calm else None
    se = (
        math.sqrt(st.variance(shortage) / len(shortage) + st.variance(calm) / len(calm))
        if len(shortage) > 1 and len(calm) > 1
        else None
    )
    t_value = difference / se if difference is not None and se else None
    gate = {
        "shortage_mean_raise_advantage": st.mean(shortage) if shortage else None,
        "calm_mean_raise_advantage": st.mean(calm) if calm else None,
        "difference": difference,
        "se": se,
        "t": t_value,
        "n_shortage": len(shortage),
        "n_calm": len(calm),
        "pass": bool(difference is not None and difference > 0.0 and t_value is not None and t_value >= 3.0),
    }
    labels = sorted({label for label, _ in by_step}, key=lambda s: (s != "hold", s))
    return {
        "source": str(path.resolve()),
        "records": records,
        "invariants": {
            "hold_exactly_zero": not hold_errors,
            "sampled_action_present": not sampled_missing,
            "at_least_two_branches": not short_rows,
            "hold_errors": hold_errors[:20],
            "sampled_missing": sampled_missing[:20],
            "short_rows": short_rows[:20],
        },
        "advantage_by_step_and_regime": {
            label: {r: mean_se(by_step[(label, r)]) for r in ("shortage", "calm", "other")}
            for label in labels
        },
        "mean_raise_advantage": {r: mean_se(decision_raise[r]) for r in ("shortage", "calm", "other")},
        "mean_cut_advantage": {r: mean_se(decision_cut[r]) for r in ("shortage", "calm", "other")},
        "best_step_share": {
            r: {c: (best_class[r].count(c) / len(best_class[r]) if best_class[r] else None)
                for c in ("raise", "hold", "cut")}
            for r in ("shortage", "calm", "other")
        },
        "per_decision_advantage_sd": {r: mean_se(per_decision_sd[r]) for r in ("shortage", "calm", "other")},
        "decision_level": {r: mean_se(levels[r]) for r in ("shortage", "calm", "other")},
        "gate": gate,
        "pass": bool(not hold_errors and not sampled_missing and not short_rows and gate["pass"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dump", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.dump)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
