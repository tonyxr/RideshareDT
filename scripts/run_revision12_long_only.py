#!/usr/bin/env python3
"""Run one full Revision 12 training followed by the registered evaluation."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from run_revision12_verify import BASE, evaluate, launch, status, train_cmd


BALANCED_STAGE_FRACTIONS = {
    "--training_stage_market_fraction": "0.05",
    "--training_stage_static_fraction": "0.05",
    "--training_stage_dynamic_fraction": "0.55",
    "--training_stage_profit_fraction": "0.35",
}


def balanced_train_cmd(root: Path, days: int) -> list[str]:
    """Return the registered Revision 12 command with the balanced schedule."""
    command = train_cmd(root, days, oracle=True, smoke=False)
    for flag, value in BALANCED_STAGE_FRACTIONS.items():
        command[command.index(flag) + 1] = value
    return command


def main() -> int:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    if len(sys.argv) > 2:
        root = Path(sys.argv[2]).resolve()
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        root = BASE / "runs" / f"revision12_long_oracle_{days}_{stamp}"

    root.mkdir(parents=True, exist_ok=False)
    status(
        root,
        phase="training",
        days=days,
        runner_pid=os.getpid(),
        oracle_shortage_flag=True,
        action_contract="independent_coefficient_x_13_magnitudes",
        action_count=66,
        evaluation_openings=10,
        stage_days={
            "market": int(round(days * 0.05)),
            "static_opponent": int(round(days * 0.05)),
            "price_gap_catchup": int(round(days * 0.55)),
            "profit_maximizers": days
            - int(round(days * 0.05))
            - int(round(days * 0.05))
            - int(round(days * 0.55)),
        },
    )
    process = launch(balanced_train_cmd(root, days), root)
    status(root, training_pid=process.pid)
    code = process.wait()
    status(root, training_exit_code=code)
    if code != 0:
        status(root, phase="failed", error="long training failed; see training.log")
        return code

    status(root, phase="evaluation")
    try:
        result = evaluate(root / "policy.pt", root / "eval", openings=10)
    except BaseException as exc:
        status(root, phase="failed", error=f"long evaluation failed: {exc!r}")
        return 1

    status(root, phase="complete", result=result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
