#!/usr/bin/env python3
"""Wait for R15c continuations and R15d extension, then rebuild all graphs."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


CITIES = ("nyc", "boston", "chicago")
HERE = Path(__file__).resolve().parent
PYTHON = HERE.parents[1] / ".venv" / "bin" / "python"


def state(path: Path) -> str:
    if not path.is_file():
        return "missing"
    return str(json.loads(path.read_text()).get("state", "unknown"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v15c-4000-partial", required=True, type=Path)
    parser.add_argument("--v15c-8000-partial", required=True, type=Path)
    parser.add_argument("--v15c-4000-resume", required=True, type=Path)
    parser.add_argument("--v15c-8000-resume", required=True, type=Path)
    parser.add_argument("--v15d", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--launch-label", default="")
    args = parser.parse_args()
    try:
        while True:
            states = []
            for root in (args.v15c_4000_resume, args.v15c_8000_resume):
                states.extend(state(root / city / "status.json") for city in CITIES)
            states.extend(
                state(args.v15d / city / "extended_evaluation_status.json")
                for city in CITIES
            )
            if "failed" in states:
                raise RuntimeError(f"a required pipeline failed: {states}")
            if states and all(value == "complete" for value in states):
                break
            time.sleep(max(10, args.poll_seconds))

        command = [
            str(PYTHON),
            str(HERE / "generate_training_comparison_graphs.py"),
            "--run", f"v15c_4000_partial={args.v15c_4000_partial.resolve()}",
            "--run", f"v15c_8000_partial={args.v15c_8000_partial.resolve()}",
            "--run", f"v15c_4000_resumed={args.v15c_4000_resume.resolve()}",
            "--run", f"v15c_8000_resumed={args.v15c_8000_resume.resolve()}",
            "--run", f"v15d_6000={args.v15d.resolve()}",
            "--output", str(args.output.resolve()),
        ]
        return subprocess.run(command, cwd=HERE.parent).returncode
    finally:
        if args.launch_label:
            subprocess.Popen(
                ["/bin/launchctl", "remove", args.launch_label],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )


if __name__ == "__main__":
    raise SystemExit(main())
