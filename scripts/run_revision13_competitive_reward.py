#!/usr/bin/env python3
"""Run the full Revision 13 competitive-reward curriculum and evaluation."""

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

from run_revision12_verify import (
    BASE,
    ROOT,
    assert_independent_action_contract,
    env,
    evaluate,
)


SOURCE_RUN = (
    BASE / "runs" / "revision12_long_oracle_balanced_8000_20261020_1416"
)
SOURCE_COMMAND = SOURCE_RUN / "command.json"
LEAD_SCHEDULE = "0,0.55,0.70,0.80"
PROFIT_OPPONENTS = ",".join((
    "profit_opt_pi_price_gap",
    "profit_opt_mpc_grid",
    "profit_opt_region_supply_demand",
    "profit_opt_surge_driver_incentive",
    "profit_opt_queue_service_threshold",
    "profit_best_response_myopic",
    "profit_best_response_forward",
    "profit_best_response_anticipating",
))


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
        command.insert(index + 1, value)
    else:
        command[index + 1] = value


def remove_option(command: list[str], option: str, *, takes_value: bool) -> None:
    while option in command:
        index = command.index(option)
        del command[index]
        if takes_value and index < len(command) and not command[index].startswith("--"):
            del command[index]


def training_command(run_dir: Path, days: int) -> list[str]:
    command = [str(value) for value in json.loads(SOURCE_COMMAND.read_text())]
    remove_option(command, "--trained_model_in", takes_value=True)
    remove_option(command, "--whole_tariff_actions", takes_value=False)
    set_option(command, "--train_timesteps", str(days))
    set_option(command, "--training_stage_market_fraction", "0.05")
    set_option(command, "--training_stage_static_fraction", "0.05")
    set_option(command, "--training_stage_dynamic_fraction", "0.55")
    set_option(command, "--training_stage_profit_fraction", "0.35")
    set_option(command, "--stage_profit_lead_weights", LEAD_SCHEDULE)
    set_option(
        command,
        "--training_catchup_opponent_modes",
        "adaptive_best_response_memory,adaptive_best_response_aggressive",
    )
    set_option(command, "--training_profit_opponent_modes", PROFIT_OPPONENTS)
    set_option(command, "--checkpoint_validation_interval_days", "600")
    set_option(command, "--checkpoint_validation_openings", "1")
    set_option(command, "--report_prefix", os.path.relpath(run_dir / "nyc", BASE))
    set_option(command, "--trained_model_out", str(run_dir / "policy.pt"))
    set_option(command, "--trained_model_id", f"r13-nyc-competitive-reward-{days}")
    set_option(command, "--model_registry_dir", str(run_dir / "registry"))
    if "--actor_true_shortage_flag" not in command:
        command.append("--actor_true_shortage_flag")
    assert_independent_action_contract(command)
    if command[command.index("--stage_profit_lead_weights") + 1] != LEAD_SCHEDULE:
        raise ValueError("Revision 13 competitive reward schedule is not wired")
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--days", type=int, default=8000)
    parser.add_argument("--launchd-label", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.days < 20:
        parser.error("--days must be at least 20")
    if not SOURCE_COMMAND.is_file():
        raise FileNotFoundError(SOURCE_COMMAND)

    run_dir = args.run_dir.expanduser().resolve()
    command = training_command(run_dir, int(args.days))
    if args.dry_run:
        print(json.dumps(command, indent=2))
        return 0
    if run_dir.exists():
        raise FileExistsError(f"refusing to reuse run directory: {run_dir}")
    run_dir.mkdir(parents=True)
    (run_dir / "command.json").write_text(
        json.dumps(command, indent=2) + "\n", encoding="utf-8"
    )
    atomic_json(run_dir / "run_manifest.json", {
        "schema_version": 1,
        "kind": "revision13_full_competitive_reward",
        "training_days": int(args.days),
        "stage_days": {
            "market": int(args.days * 0.05),
            "static_opponent": int(args.days * 0.05),
            "price_gap_catchup": int(args.days * 0.55),
            "profit_maximizers": int(args.days * 0.35),
        },
        "stage_profit_lead_weights": {
            "market": 0.0,
            "static": 0.55,
            "catchup": 0.70,
            "profit": 0.80,
        },
        "reward_definition": (
            "(1-w)*own contribution profit per incoming request + "
            "w*(own-rival contribution profit per incoming request)"
        ),
        "profit_opponents": PROFIT_OPPONENTS.split(","),
        "source_command": str(SOURCE_COMMAND),
        "source_command_sha256": sha256(SOURCE_COMMAND),
        "core_sha256": sha256(BASE / "Core.py"),
        "reward_code_sha256": sha256(BASE / "platform_mdp.py"),
        "created_at": time.time(),
    })
    status_path = run_dir / "status.json"
    atomic_json(status_path, {
        "phase": "training",
        "runner_pid": os.getpid(),
        "training_days": int(args.days),
        "stage_profit_lead_weights": [0.0, 0.55, 0.70, 0.80],
        "action_contract": "independent_coefficient_x_13_magnitudes",
        "action_count": 66,
        "updated_at": time.time(),
    })
    with (run_dir / "training.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=BASE,
            env=env(),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        atomic_json(status_path, {
            "phase": "training",
            "runner_pid": os.getpid(),
            "training_pid": process.pid,
            "training_days": int(args.days),
            "stage_profit_lead_weights": [0.0, 0.55, 0.70, 0.80],
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
        "policy": str(run_dir / "policy.pt"),
        "result": result,
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
        if requested_label:
            subprocess.Popen(
                ["/bin/launchctl", "remove", requested_label],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
    raise SystemExit(exit_code)
