#!/usr/bin/env python3
"""Revision 14 Phase-1 bit identity check against the Revision 12 tree."""
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
BASE14 = HERE.parent
ROOT = BASE14.parent
BASE12 = ROOT / "rideshare_v26r_revision12"
sys.path.insert(0, str(HERE))
import run_revision12_verify as r12  # noqa: E402


def _set_option(command: list[str], flag: str, value: str) -> None:
    if flag not in command:
        command.extend([flag, value])
    else:
        command[command.index(flag) + 1] = value


def _identity_command(base: Path, out: Path) -> list[str]:
    command = r12.train_cmd(out, 20, False, True)
    command[command.index(str(BASE14 / "Core.py"))] = str(base / "Core.py")
    for flag, value in (
        ("--training_stage_market_fraction", "0.05"),
        ("--training_stage_static_fraction", "0.05"),
        ("--training_stage_dynamic_fraction", "0.55"),
        ("--training_stage_profit_fraction", "0.35"),
        ("--trained_model_id", "r14-identity-20"),
        # Core pins report prefixes beneath its own code directory.  Pass the
        # run artifact path relative to the selected revision tree so both
        # executions write into their respective identity output folders.
        ("--report_prefix", os.path.relpath(out / "nyc", base)),
    ):
        _set_option(command, flag, value)
    # The absence of every Revision-14 opt-in flag is part of this test.
    forbidden = {
        "--nmcc_branches",
        "--momentum_features",
        "--opponent_features",
    }
    if forbidden & set(command):
        raise AssertionError(f"identity command enables {forbidden & set(command)}")
    return command


def _run(base: Path, out: Path) -> tuple[int, float]:
    out.mkdir(parents=True, exist_ok=True)
    # DiagnosticJournal intentionally creates JSONL streams exclusively.  A
    # failed/interrupted identity attempt must therefore clear only its own
    # run-local streams before retrying, or the next attempt fails before day 1.
    for stale in out.glob("nyc_*.jsonl"):
        stale.unlink()
    command = _identity_command(base, out)
    (out / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    started = time.perf_counter()
    with (out / "training.log").open("w") as log:
        completed = subprocess.run(
            command,
            cwd=base,
            env=r12.env(),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    return completed.returncode, time.perf_counter() - started


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _equal(a: Any, b: Any) -> bool:
    # JSON numbers are compared exactly: deterministic mode must be bit-identical.
    return type(a) is type(b) and a == b


def _compare(
    left_path: Path,
    right_path: Path,
    *,
    kind: str,
) -> dict[str, Any]:
    left, right = _rows(left_path), _rows(right_path)
    mismatches: list[dict[str, Any]] = []
    if len(left) != len(right):
        mismatches.append({"row_count": [len(left), len(right)]})
    for index, (a, b) in enumerate(zip(left, right)):
        common = set(a) & set(b)
        if kind == "training":
            keys = sorted(
                key
                for key in common
                if key in {
                    "batch",
                    "avg_reward",
                    "reward_raw",
                    "last_action",
                    "last_action_steps",
                    "dominant_action",
                    "dominant_action_label",
                    "raw_policy_top_action",
                    "raw_policy_top_action_label",
                    "raw_policy_top_action_rate",
                    "action_counts",
                    "shock_block_counts",
                    "training_stage",
                    "training_stage_opponent_regime",
                }
                or key.startswith("reward_")
            )
        else:
            keys = sorted(
                key
                for key in common
                if key == "batch"
                or key == "training_stage"
                or key.startswith("ppo_")
            )
        for key in keys:
            if not _equal(a[key], b[key]):
                mismatches.append(
                    {"row": index, "key": key, "revision12": a[key], "revision14": b[key]}
                )
                if len(mismatches) >= 50:
                    break
        if len(mismatches) >= 50:
            break
    return {
        "revision12_rows": len(left),
        "revision14_rows": len(right),
        "mismatch_count_capped": len(mismatches),
        "mismatches": mismatches,
        "pass": not mismatches,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=BASE14 / "runs" / "revision14_verify" / "identity",
    )
    args = parser.parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / "status.json"
    status_path.write_text(json.dumps({"phase": "running", "time": time.time()}, indent=2))
    codes: dict[str, int] = {}
    durations: dict[str, float] = {}
    for name, base in (("revision12", BASE12), ("revision14", BASE14)):
        codes[name], durations[name] = _run(base, root / name)
        if codes[name] != 0:
            report = {
                "pass": False,
                "exit_codes": codes,
                "durations_seconds": durations,
                "error": f"{name} training failed",
            }
            (root / "identity_report.json").write_text(json.dumps(report, indent=2) + "\n")
            status_path.write_text(json.dumps({"phase": "failed", **report, "time": time.time()}, indent=2))
            return 1

    comparisons = {}
    for filename, kind in (
        ("nyc_training_live.jsonl", "training"),
        ("nyc_ppo_updates.jsonl", "ppo"),
    ):
        comparisons[filename] = _compare(
            root / "revision12" / filename,
            root / "revision14" / filename,
            kind=kind,
        )
    passed = all(item["pass"] for item in comparisons.values())
    report = {
        "pass": passed,
        "exit_codes": codes,
        "durations_seconds": durations,
        "comparisons": comparisons,
    }
    (root / "identity_report.json").write_text(json.dumps(report, indent=2) + "\n")
    status_path.write_text(
        json.dumps(
            {"phase": "complete" if passed else "failed", **report, "time": time.time()},
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(report, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
