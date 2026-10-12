#!/usr/bin/env python3
"""Create training-safe threshold caches while preserving raw GPT classifications."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List


PROMPT_PROTOCOL = "structured_context_rubric_v10"
AUDIT_PROTOCOL = "deterministic_context_validation_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audited_threshold(row: Dict[str, Any]) -> float:
    raw = (
        float(row["ThresholdIncomeAnchor"])
        + float(row["ThresholdHouseholdAdjustment"])
        + float(row["ThresholdLoyaltyAdjustment"])
        + float(row["ThresholdGapAdjustment"])
        + float(row["ThresholdExpectedContextAdjustment"])
    )
    return float(min(5.0, max(0.5, math.floor(raw * 4.0 + 0.5) / 4.0)))


def audit_cache(source: Path, target: Path) -> Dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    rows = gpt_rows = corrections = final_matches = 0
    with source.open("r", encoding="utf-8") as reader, temporary.open(
        "w", encoding="utf-8"
    ) as writer:
        for line in reader:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("ThresholdPromptProtocol") != PROMPT_PROTOCOL:
                raise ValueError(f"obsolete prompt protocol in {source}")
            raw_context = float(row["ThresholdContextAdjustment"])
            expected_context = float(row["ThresholdExpectedContextAdjustment"])
            corrected = not math.isclose(raw_context, expected_context)
            row["ThresholdRawGPTContextAdjustment"] = raw_context
            row["ThresholdContextAdjustment"] = expected_context
            row["ThresholdContextWasCorrected"] = corrected
            row["ThresholdAuditProtocol"] = AUDIT_PROTOCOL
            row["PriceThreshold"] = _audited_threshold(row)
            rows += 1
            gpt_rows += int(row.get("PriceThresholdSource") == "gpt")
            corrections += int(corrected)
            final_matches += int(math.isclose(
                float(row["ThresholdContextAdjustment"]),
                expected_context,
            ))
            writer.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temporary, target)
    return {
        "source": str(source),
        "source_sha256": _sha256(source),
        "target": str(target),
        "target_sha256": _sha256(target),
        "rows": rows,
        "gpt_source_rate": gpt_rows / max(1, rows),
        "raw_context_accuracy": 1.0 - corrections / max(1, rows),
        "correction_rate": corrections / max(1, rows),
        "audited_context_accuracy": final_matches / max(1, rows),
        "audit_protocol": AUDIT_PROTOCOL,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--target-root", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    source_root = Path(args.source_root).expanduser().resolve()
    target_root = Path(args.target_root).expanduser().resolve()
    reports: List[Dict[str, Any]] = []
    for city in ("nyc", "boston", "chicago"):
        for label in ("primary", "prompt_repeat_1", "prompt_repeat_2", "rationale_audit"):
            reports.append(audit_cache(
                source_root / city / f"{label}.jsonl",
                target_root / city / f"{label}.jsonl",
            ))
    report_path = Path(args.report).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps({"schema_version": 1, "caches": reports}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, report_path)


if __name__ == "__main__":
    main()
