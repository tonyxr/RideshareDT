#!/usr/bin/env python3
"""Verify and render a frozen v28 evaluation report.

This command never runs policies, changes a checkpoint, or reads a profile
split.  It only renders a content-hash-verified report that was frozen by
``v28_evaluation_reporting.freeze_report``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from v28_evaluation_reporting import render_frozen_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render JSON, Markdown, and PNGs from a verified frozen v28 report."
    )
    parser.add_argument("frozen_report", type=Path)
    parser.add_argument("output_directory", type=Path)
    arguments = parser.parse_args()
    manifest = render_frozen_report(
        arguments.frozen_report, arguments.output_directory
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
