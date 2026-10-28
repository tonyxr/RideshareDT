#!/usr/bin/env python3
"""Queue revision-10 production work behind the active revision-8 baseline."""

import argparse
import json
import os
from pathlib import Path
import time


BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent
PYTHON = ROOT / ".venv/bin/python"


def atomic_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocker-status", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()
    blocker = args.blocker_status.resolve()
    output = args.output.resolve()
    sidecar = output.with_name(output.name + ".queue_status.json")
    if output.exists():
        raise FileExistsError(f"production output already exists: {output}")

    while True:
        payload = json.loads(blocker.read_text())
        phase = str(payload.get("phase", "unknown"))
        exit_codes = dict(payload.get("exit_codes", {}) or {})
        atomic_json(sidecar, {
            "phase": "waiting_for_revision8",
            "blocker": str(blocker),
            "blocker_phase": phase,
            "blocker_completed_jobs": len(exit_codes),
            "time": time.time(),
        })
        if phase == "complete":
            if len(exit_codes) != 9 or any(int(code) != 0 for code in exit_codes.values()):
                atomic_json(sidecar, {
                    "phase": "blocked_by_revision8_failure",
                    "blocker": str(blocker),
                    "exit_codes": exit_codes,
                    "time": time.time(),
                })
                raise RuntimeError(
                    "revision-8 baseline did not complete all nine jobs successfully"
                )
            break
        time.sleep(max(5.0, float(args.poll_seconds)))

    command = [
        str(PYTHON), "-u", str(BASE / "scripts/run_v26r_multiseed.py"),
        "--output", str(output),
        "--city-calibration-root",
        str(ROOT / "artifacts/revision7_direct_calibration_20261018"),
        "--training-scenario-root",
        str(BASE / "config/revision10_curriculum"),
        "--seeds", "251012,251013,251014",
        "--days", "20000",
        "--max-workers", "3",
        "--update-days", "5",
        "--rollout-transitions", "180",
        "--lead-weight", "0.6",
        "--figures",
    ]
    atomic_json(sidecar, {
        "phase": "launching_revision10",
        "command": command,
        "time": time.time(),
    })
    matplotlib_cache = ROOT / "artifacts/.matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.update(
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONDONTWRITEBYTECODE="1",
        MPLBACKEND="Agg",
        MPLCONFIGDIR=str(matplotlib_cache),
    )
    environment.pop("PYTHONPATH", None)
    os.chdir(BASE)
    os.execve(command[0], command, environment)


if __name__ == "__main__":
    main()
