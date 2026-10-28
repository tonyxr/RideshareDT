#!/usr/bin/env python3
"""Summarize Gate-1 sweeps: paired differences vs the best static on common tapes.

Usage: v26r_gate1_summary.py <slug> [scenario ...]
"""
import glob, json, os, sys
import numpy as np

slug = sys.argv[1]
root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "gate1", slug)
scenarios = sys.argv[2:] or sorted(os.listdir(root))
for scen in scenarios:
    specs = {}
    for f in glob.glob(os.path.join(root, scen, "*.json")):
        d = json.load(open(f))
        for spec, rows in d["results"].items():
            specs[spec] = {k: v["objective"] for k, v in rows.items()}
    if not specs:
        continue
    names = sorted(set.intersection(*[set(v) for v in specs.values()]))
    statics = {s: np.mean([specs[s][n] for n in names]) for s in specs if s.startswith("static:") and "@" not in s}
    best_static = max(statics, key=statics.get)
    base = np.array([specs[best_static][n] for n in names])
    print(f"\n=== {slug} / {scen}: {len(names)} paired rollouts; best static = {best_static} ({statics[best_static]:.4f})")
    rows = []
    for spec, vals in specs.items():
        v = np.array([vals[n] for n in names]); d = v - base
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
        rows.append((spec, v.mean(), d.mean(), se, d.mean() / se if se > 0 else 0.0))
    for spec, m, dm, se, t in sorted(rows, key=lambda r: -r[1]):
        flag = " <-- clears 3 SE" if t >= 3 else ""
        print(f"  {spec[:66]:66s} mean={m:.4f}  diff={dm:+.4f}  SE={se:.4f}  t={t:+.1f}{flag}")
