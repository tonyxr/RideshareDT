#!/usr/bin/env python3
"""Re-score saved V18 outcomes at the four registered rival-profit weights."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


WEIGHTS = (0.0, 0.25, 0.5, 0.75)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/v18/results/rescored.csv"))
    args = parser.parse_args()
    with args.input.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    output = []
    for row in rows:
        own = float(row["profit_per_request"])
        rival = float(row["rival_profit_per_request"])
        for weight in WEIGHTS:
            output.append({**row, "rival_profit_weight": weight, "objective": own - weight * rival})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader(); writer.writerows(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
