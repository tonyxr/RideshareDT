#!/usr/bin/env python3
"""Resume Revision 12 Stage 4 from the durable day-5,200 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from run_revision12_verify import BASE, ROOT, PY, assert_independent_action_contract, env, evaluate


SOURCE_RUN = BASE / "runs" / "revision12_long_oracle_balanced_8000_20261020_1416"
SOURCE_CHECKPOINT = SOURCE_RUN / "policy.stage_price_gap_catchup.pt"
SOURCE_COMMAND = SOURCE_RUN / "command.json"


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_option(command: list[str], option: str, value: str) -> None:
    if option not in command:
        command.extend((option, value))
        return
    index = command.index(option)
    if index + 1 >= len(command) or command[index + 1].startswith("--"):
        raise ValueError(f"{option} does not have a value")
    command[index + 1] = value


def continuation_command(run_dir: Path, days: int) -> list[str]:
    command = json.loads(SOURCE_COMMAND.read_text(encoding="utf-8"))
    if not isinstance(command, list) or not command:
        raise ValueError("source command is not a command list")
    command[0] = PY
    set_option(command, "--train_timesteps", str(days))
    set_option(command, "--training_stage_market_fraction", "0.000000000001")
    set_option(command, "--training_stage_static_fraction", "0.000000000001")
    set_option(command, "--training_stage_dynamic_fraction", "0.000000000001")
    set_option(command, "--training_stage_profit_fraction", "0.999999999997")
    try:
        report_prefix = (run_dir / "nyc").relative_to(BASE)
    except ValueError as exc:
        raise ValueError(f"run directory must be inside {BASE}") from exc
    set_option(command, "--report_prefix", str(report_prefix))
    set_option(command, "--trained_model_out", str(run_dir / "policy.pt"))
    set_option(command, "--trained_model_id", f"r12-nyc-oracle-stage4-resume-{days}")
    set_option(command, "--model_registry_dir", str(run_dir / "registry"))
    set_option(command, "--trained_model_in", str(SOURCE_CHECKPOINT))
    assert_independent_action_contract(command)
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--days", type=int, default=2800)
    parser.add_argument("--launchd-label", default="")
    args = parser.parse_args()
    if args.days < 4:
        parser.error("--days must be at least four")

    run_dir = args.run_dir.expanduser().resolve()
    if run_dir.exists():
        raise FileExistsError(f"refusing to reuse recovery directory: {run_dir}")
    if not SOURCE_CHECKPOINT.is_file():
        raise FileNotFoundError(SOURCE_CHECKPOINT)
    run_dir.mkdir(parents=True)
    command = continuation_command(run_dir, int(args.days))
    (run_dir / "command.json").write_text(
        json.dumps(command, indent=2) + "\n", encoding="utf-8"
    )
    atomic_json(
        run_dir / "resume_manifest.json",
        {
            "schema_version": 1,
            "kind": "revision12_stage4_optimizer_resume",
            "source_run": str(SOURCE_RUN),
            "source_checkpoint": str(SOURCE_CHECKPOINT),
            "source_checkpoint_sha256": sha256(SOURCE_CHECKPOINT),
            "source_checkpoint_training_day": 5200,
            "continuation_days": int(args.days),
            "optimizer_resume": True,
            "expected_stage_days": {
                "market": 1,
                "static_opponent": 0,
                "price_gap_catchup": 0,
                "profit_maximizers": int(args.days) - 1,
            },
            "numerical_fix": (
                "scale-aware component-return cancellation tolerance; "
                "material inconsistencies still fail closed"
            ),
            "created_at": time.time(),
        },
    )
    status_path = run_dir / "status.json"
    atomic_json(status_path, {
        "phase": "training",
        "runner_pid": os.getpid(),
        "source_checkpoint": str(SOURCE_CHECKPOINT),
        "continuation_days": int(args.days),
        "action_contract": "independent_coefficient_x_13_magnitudes",
        "action_count": 66,
        "updated_at": time.time(),
    })

    with (run_dir / "training.log").open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=BASE,
            env=env(),
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        (run_dir / "training_pid.txt").write_text(
            f"{process.pid}\n", encoding="utf-8"
        )
        atomic_json(status_path, {
            "phase": "training",
            "runner_pid": os.getpid(),
            "training_pid": process.pid,
            "source_checkpoint": str(SOURCE_CHECKPOINT),
            "continuation_days": int(args.days),
            "action_contract": "independent_coefficient_x_13_magnitudes",
            "action_count": 66,
            "updated_at": time.time(),
        })
        code = process.wait()

    (run_dir / "training_exit_status.txt").write_text(
        f"{code}\n", encoding="utf-8"
    )
    if code != 0:
        atomic_json(status_path, {
            "phase": "failed",
            "training_exit_code": code,
            "updated_at": time.time(),
        })
        return code

    atomic_json(status_path, {
        "phase": "evaluation",
        "training_exit_code": 0,
        "policy": str(run_dir / "policy.pt"),
        "updated_at": time.time(),
    })
    try:
        result = evaluate(run_dir / "policy.pt", run_dir / "eval", openings=10)
    except BaseException as exc:
        atomic_json(status_path, {
            "phase": "evaluation_failed",
            "training_exit_code": 0,
            "error": repr(exc),
            "updated_at": time.time(),
        })
        return 1

    atomic_json(status_path, {
        "phase": "complete",
        "training_exit_code": 0,
        "result": result,
        "policy": str(run_dir / "policy.pt"),
        "updated_at": time.time(),
    })
    return 0


if __name__ == "__main__":
    requested_label = ""
    if "--launchd-label" in sys.argv:
        label_index = sys.argv.index("--launchd-label")
        if label_index + 1 < len(sys.argv):
            requested_label = sys.argv[label_index + 1]
    try:
        exit_code = main()
    finally:
        # A submitted launchd job otherwise defaults to KeepAlive and would
        # restart after a successful training/evaluation run or any failure.
        if requested_label:
            subprocess.Popen(
                ["/bin/launchctl", "remove", requested_label],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
    raise SystemExit(exit_code)
