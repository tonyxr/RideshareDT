#!/usr/bin/env python3
"""Revision 15 background controller for launchctl submit.

Runs the requested phases of run_revision15_verify.py in order, skips phases whose
status.json already records pass=true, stops at the first failure, and always removes
its own launchd job on exit (launchctl submit jobs are restarted until removed).

usage: r15_controller.py --label L --run-root R [--contract-policy P] PHASE [PHASE ...]
phases: tests identity isolation structure probe smoke train evaluate replicate
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
PY = BASE.parent / ".venv/bin/python"
sys.path.insert(0, str(HERE))
PHASES = ("tests", "identity", "isolation", "structure", "probe", "smoke", "train", "evaluate", "replicate")


def log(path: Path, message: str) -> None:
    with path.open("a") as handle:
        handle.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {message}\n")


def passed(root: Path, phase: str) -> bool:
    path = root / phase / "status.json"
    return path.is_file() and bool(json.loads(path.read_text()).get("pass"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--contract-policy", type=Path, default=None)
    parser.add_argument("phases", nargs="+", choices=PHASES)
    args = parser.parse_args()
    root = args.run_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    controller_log = root / "controller.log"
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1")
    try:
        for phase in args.phases:
            if passed(root, phase):
                log(controller_log, f"skip {phase} (already passed)")
                continue
            command = [str(PY), "-u", str(HERE / "run_revision15_verify.py"), phase, "--run-root", str(root)]
            if phase == "structure":
                if args.contract_policy is None:
                    log(controller_log, "STOPPED: the structure phase needs --contract-policy")
                    return 2
                command += ["--contract-policy", str(args.contract_policy.resolve())]
            log(controller_log, f"start {phase}")
            with (root / f"{phase}.driver.log").open("a") as handle:
                code = subprocess.run(command, cwd=BASE, env=env, stdout=handle, stderr=subprocess.STDOUT).returncode
            log(controller_log, f"end {phase} exit={code}")
            if code != 0:
                log(controller_log, f"STOPPED at {phase}; see {root / phase / 'status.json'}")
                return code
        log(controller_log, "all requested phases passed")
        return 0
    finally:
        subprocess.Popen(["/bin/launchctl", "remove", args.label], stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)


if __name__ == "__main__":
    raise SystemExit(main())
