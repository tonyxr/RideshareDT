#!/usr/bin/env python3
"""Generate v28 shock-calibration evidence from existing trip banks.

Example:
    python scripts/v28_shock_readiness.py

The command writes a JSON evidence artifact and a Markdown readiness report.
It never edits ``calibration.json``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from shock_calibration import build_artifact, write_artifact, write_markdown  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trip-bank",
        action="append",
        default=[],
        help="Path to a trip_bank.csv; repeat for multiple cities. Defaults to data/processed/*/trip_bank.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "artifacts" / "v28_shock_readiness"),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2801)
    args = parser.parse_args()

    paths = [Path(value) for value in args.trip_bank]
    if not paths:
        paths = sorted((PROJECT_ROOT / "data" / "processed").glob("*/trip_bank.csv"))
    if not paths:
        parser.error("no trip_bank.csv files were found")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        parser.error(f"trip banks do not exist: {missing}")
    if args.bootstrap_replicates < 0:
        parser.error("--bootstrap-replicates must be non-negative")

    artifact = build_artifact(
        paths,
        bootstrap_replicates=int(args.bootstrap_replicates),
        seed=int(args.seed),
    )
    output_dir = Path(args.output_dir)
    json_path = output_dir / "trip_bank_shock_calibration.json"
    report_path = output_dir / "trip_bank_shock_readiness.md"
    write_artifact(artifact, json_path)
    write_markdown(artifact, report_path)
    print(f"Wrote evidence artifact: {json_path}")
    print(f"Wrote readiness report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
