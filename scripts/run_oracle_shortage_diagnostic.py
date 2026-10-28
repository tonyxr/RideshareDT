"""Run the opt-in NYC oracle-shortage staged-learning diagnostic.

This is deliberately a fresh-policy experiment.  The only information change
relative to the standard four-stage runner is ``--actor_true_shortage_flag``;
the canonical benchmark implementations and market mechanics are untouched.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent
sys.path.insert(0, str(BASE))

from scripts.run_v26r_staged import command, environment, source_hashes, summarize


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--days", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=251_012)
    parser.add_argument(
        "--city-calibration-artifact",
        type=Path,
        default=(
            ROOT
            / "artifacts/revision7_direct_calibration_20261018/nyc/city_calibration.json"
        ),
    )
    args = parser.parse_args()
    if args.days <= 0:
        parser.error("--days must be positive")

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    calibration = args.city_calibration_artifact.resolve()
    threshold_cache = (
        ROOT
        / "artifacts/v26_confirmatory/thresholds/nyc/llm_numeric_primary.jsonl"
    )
    city_data = ROOT / "data/processed/nyc/calibration.json"
    for required in (calibration, threshold_cache, city_data):
        if not required.is_file():
            raise FileNotFoundError(required)

    hashes = source_hashes()
    cmd = command(
        out,
        days=int(args.days),
        integration=False,
        city="New York City",
        slug="nyc",
        seed=int(args.seed),
        actor_true_shortage_flag=True,
    )
    cmd += ["--city_calibration_artifact", str(calibration)]
    manifest = {
        "diagnostic": "actor_true_shortage_flag",
        "oracle_information": True,
        "production_policy_claim": False,
        "city": "New York City",
        "seed": int(args.seed),
        "training_days": int(args.days),
        "stage_days": [
            int(round(args.days * 0.10)),
            int(round(args.days * 0.10)),
            int(round(args.days * 0.30)),
            int(args.days) - int(round(args.days * 0.50)),
        ],
        "source_hashes": hashes,
        "data_hashes": {
            str(path): _sha256(path)
            for path in (calibration, threshold_cache, city_data)
        },
        "command": cmd,
        "started_at": datetime.datetime.now().isoformat(),
        "pid": os.getpid(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "command.json").write_text(json.dumps(cmd, indent=2))

    def status(value: str) -> None:
        (out / "status.txt").write_text(value + "\n")
        print(f"{datetime.datetime.now().isoformat()} {value}", flush=True)

    try:
        status(
            f"training {args.days} NYC days with actor oracle shortage flag"
        )
        with (out / "run.log").open("x") as log:
            child = subprocess.Popen(
                cmd,
                cwd=BASE,
                env=environment(),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            (out / "training_pid.txt").write_text(f"{child.pid}\n")
            code = child.wait()
        (out / "training_exit_status.txt").write_text(f"{code}\n")
        if code:
            raise RuntimeError(f"training/evaluation exited {code}")
        if source_hashes() != hashes:
            raise RuntimeError("source changed during the run; results require review")
        status("summarizing final-policy evaluation")
        summarize(
            out,
            expected_days=int(args.days),
            slug="nyc",
            seed=int(args.seed),
        )
        status("completed")
        (out / "exit_status.txt").write_text("0\n")
    except BaseException:
        traceback.print_exc()
        status("failed; artifacts preserved")
        (out / "exit_status.txt").write_text("1\n")
        raise


if __name__ == "__main__":
    main()
