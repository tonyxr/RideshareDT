#!/usr/bin/env python3
"""Revision 15 flags-off bit identity against the Revision 14 tree."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
BASE15 = HERE.parent
ROOT = BASE15.parent
BASE14 = ROOT / "rideshare_v26r_revision14"
sys.path.insert(0, str(HERE))
import run_revision12_verify as r12  # noqa: E402
from r14_identity_check import _set_option  # noqa: E402

TRAINING_KEYS = {
    "batch", "avg_reward", "reward_raw", "last_action", "last_action_steps",
    "dominant_action", "dominant_action_label", "raw_policy_top_action",
    "raw_policy_top_action_label", "raw_policy_top_action_rate", "action_counts",
    "shock_block_counts", "training_stage", "training_stage_opponent_regime",
}


def rows(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def compare(left_path: Path, right_path: Path, *, kind: str) -> dict[str, Any]:
    """Exact comparison of the keys both trees log (Revision 15 adds new keys only)."""
    left, right = rows(left_path), rows(right_path)
    mismatches: list[dict[str, Any]] = []
    if len(left) != len(right):
        mismatches.append({"row_count": [len(left), len(right)]})
    for index, (a, b) in enumerate(zip(left, right)):
        common = set(a) & set(b)
        if kind == "training":
            keys = sorted(k for k in common if k in TRAINING_KEYS or k.startswith("reward_"))
        else:
            keys = sorted(k for k in common if k in {"batch", "training_stage"} or k.startswith("ppo_"))
        for key in keys:
            if type(a[key]) is not type(b[key]) or a[key] != b[key]:
                mismatches.append({"row": index, "key": key, "revision14": a[key], "revision15": b[key]})
                if len(mismatches) >= 50:
                    break
        if len(mismatches) >= 50:
            break
    return {"revision14_rows": len(left), "revision15_rows": len(right),
            "mismatch_count_capped": len(mismatches), "mismatches": mismatches, "pass": not mismatches}

FORBIDDEN = {
    "--level_step_actions",
    "--tariff_structure",
    "--tariff_level_bounds",
    "--ppo_entropy_start",
    "--branch_imitation_coeff",
    "--branch_imitation_temperature_scale",
    "--nmcc_branches",
    "--momentum_features",
    "--opponent_features",
}


def identity_command(base: Path, out: Path) -> list[str]:
    command = r12.train_cmd(out, 20, False, True)
    command[command.index(str(BASE15 / "Core.py"))] = str(base / "Core.py")
    for flag, value in (
        ("--training_stage_market_fraction", "0.05"),
        ("--training_stage_static_fraction", "0.05"),
        ("--training_stage_dynamic_fraction", "0.55"),
        ("--training_stage_profit_fraction", "0.35"),
        ("--trained_model_id", "r15-identity-20"),
        ("--report_prefix", os.path.relpath(out / "nyc", base)),
    ):
        _set_option(command, flag, value)
    if FORBIDDEN & set(command):
        raise AssertionError(f"identity command enables {sorted(FORBIDDEN & set(command))}")
    return command


def run(base: Path, out: Path) -> tuple[int, float]:
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("nyc_*.jsonl"):
        stale.unlink()
    command = identity_command(base, out)
    (out / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    started = time.perf_counter()
    with (out / "training.log").open("w") as log:
        completed = subprocess.run(command, cwd=base, env=r12.env(), stdout=log, stderr=subprocess.STDOUT)
    return completed.returncode, time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    codes: dict[str, int] = {}
    durations: dict[str, float] = {}
    for name, base in (("revision14", BASE14), ("revision15", BASE15)):
        codes[name], durations[name] = run(base, root / name)
        if codes[name] != 0:
            report = {"pass": False, "exit_codes": codes, "durations_seconds": durations,
                      "error": f"{name} training failed"}
            (root / "identity_report.json").write_text(json.dumps(report, indent=2) + "\n")
            return 1
    comparisons: dict[str, Any] = {}
    for filename, kind in (("nyc_training_live.jsonl", "training"), ("nyc_ppo_updates.jsonl", "ppo")):
        comparisons[filename] = compare(root / "revision14" / filename, root / "revision15" / filename, kind=kind)
    passed = all(item["pass"] for item in comparisons.values())
    report = {"pass": passed, "exit_codes": codes, "durations_seconds": durations, "comparisons": comparisons}
    (root / "identity_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
