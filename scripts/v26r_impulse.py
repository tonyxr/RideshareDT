#!/usr/bin/env python3
"""Paired impulse response from a --policy_sweep JSON with traces.

Usage: v26r_impulse.py <sweep.json> <treated_spec_prefix> <control_spec_prefix> <column> <day_from> <day_to> [steps_per_day]
Prints the paired mean difference (treated - control) of the daily mean of <column> over the window, with SE and t.
"""
import json, sys
import numpy as np
path, treated, control, col, d0, d1 = sys.argv[1:7]
spd = int(sys.argv[7]) if len(sys.argv) > 7 else 6
d = json.load(open(path))
def pick(prefix):
    for k in d["traces"]:
        if k.startswith(prefix):
            return d["traces"][k]
    raise SystemExit(f"no spec starting with {prefix}")
def daily(rows):
    out = {}
    for r in rows:
        out.setdefault(r["rollout"], {}).setdefault(r["step"] // spd, []).append(float(r[col]))
    return {ro: {day: float(np.mean(v)) for day, v in days.items()} for ro, days in out.items()}
T, C = daily(pick(treated)), daily(pick(control))
diffs = []
for ro in sorted(set(T) & set(C)):
    vals = [T[ro][day] - C[ro][day] for day in range(int(d0), int(d1) + 1) if day in T[ro] and day in C[ro]]
    if vals:
        diffs.append(float(np.mean(vals)))
diffs = np.array(diffs)
se = diffs.std(ddof=1) / np.sqrt(len(diffs)) if len(diffs) > 1 else float("nan")
print(f"{col} days {d0}-{d1}: treated-control = {diffs.mean():+.4f}  SE={se:.4f}  t={diffs.mean()/se if se>0 else 0:+.1f}  (n={len(diffs)} tapes)")
for day in range(int(d0) - 3, int(d1) + 1):
    t = np.mean([T[ro].get(day, np.nan) for ro in T]); c = np.mean([C[ro].get(day, np.nan) for ro in C])
    print(f"  day {day:2d}: treated={t:.4f} control={c:.4f} diff={t-c:+.4f}")
