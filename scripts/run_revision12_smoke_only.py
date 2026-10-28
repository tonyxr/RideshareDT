#!/usr/bin/env python3
"""Run only the Revision 12 staged smoke training and shortage evaluation."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from run_revision12_verify import BASE, evaluate, launch, status, train_cmd


def main() -> int:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 192
    if len(sys.argv) > 2:
        root = Path(sys.argv[2]).resolve()
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        root = BASE / "runs" / f"revision12_smoke_{days}_{stamp}"

    root.mkdir(parents=True, exist_ok=False)
    status(
        root,
        phase="training",
        days=days,
        runner_pid=os.getpid(),
        oracle_shortage_flag=True,
        action_contract="independent_coefficient_x_13_magnitudes",
        action_count=66,
    )
    process = launch(train_cmd(root, days, oracle=True, smoke=True), root)
    status(root, training_pid=process.pid)
    code = process.wait()
    status(root, training_exit_code=code)
    if code != 0:
        status(root, phase="failed", error="smoke training failed; see training.log")
        return code

    status(root, phase="evaluation")
    try:
        result = evaluate(root / "policy.pt", root / "eval", openings=2)
    except BaseException as exc:
        status(root, phase="failed", error=f"smoke evaluation failed: {exc!r}")
        return 1

    status(root, phase="complete", smoke_eval=result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
