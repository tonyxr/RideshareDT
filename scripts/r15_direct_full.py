#!/usr/bin/env python3
"""Run Revision 15 full-budget training and evaluation without prerequisite gates.

This entry point exists for an explicit operator override.  It still records the
override and the fixed tariff-structure provenance, but it does not fabricate
passing test, smoke, or structure-search statuses.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import run_revision14_verify as r14
import run_revision15_verify as r15
from pricing_models import snap_tariff_level, tariff_level_of, validate_tariff_structure


def prepare_structure(root: Path, checkpoint_tariff: Path) -> Path:
    anchors = r15.city_anchors()
    tariff = {
        key: float(value)
        for key, value in json.loads(checkpoint_tariff.read_text()).items()
    }
    if set(tariff) != set(r15.COEFFICIENTS):
        raise ValueError(
            "checkpoint tariff must contain exactly "
            f"{sorted(r15.COEFFICIENTS)}"
        )
    raw_level = tariff_level_of(tariff, anchors)
    reference = snap_tariff_level(round(raw_level / 0.1) * 0.1)
    structure = {key: tariff[key] / raw_level for key in r15.COEFFICIENTS}
    payload = validate_tariff_structure(
        {
            "format": "revision15-tariff-structure-v1",
            "structure": structure,
            "anchors": anchors,
            "reference_level": reference,
            "source": {
                "operator_override": True,
                "checkpoint_tariff": str(checkpoint_tariff.resolve()),
                "checkpoint_kind": "latest fully completed coordinate-search sweep",
                "raw_level": raw_level,
                "note": "Formal Revision 15 structure-search gate was bypassed by explicit user request.",
            },
        },
        list(r15.COEFFICIENTS),
        (0.6, 2.0),
    )
    structure_root = root / "structure"
    structure_root.mkdir(parents=True, exist_ok=True)
    structure_path = structure_root / "tariff_structure.json"
    r14.atomic_json(structure_path, payload)

    comparators = {}
    for index in range(15):
        level = round(0.6 + 0.1 * index, 1)
        if abs(level - reference) > 0.2 + 1e-9:
            continue
        path = structure_root / "level_tariffs" / f"level_{level:.1f}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        r14.atomic_json(
            path,
            {key: structure[key] * level for key in r15.COEFFICIENTS},
        )
        comparators[f"{level:.1f}"] = str(path.resolve())
    r14.atomic_json(
        structure_root / "search_summary.json",
        {
            "operator_override": True,
            "reference_level": reference,
            "best_static_tariff": tariff,
            "best_static_level": raw_level,
            "structure": structure,
            "comparators": comparators,
            "level_line_search": [],
        },
    )
    return structure_path


def phase_status(root: Path, phase: str, state: str, passed: bool, **extra: object) -> None:
    r14.atomic_json(
        root / phase / "status.json",
        {
            "phase": phase,
            "state": state,
            "pass": passed,
            **extra,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--checkpoint-tariff", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    r14.atomic_json(
        root / "operator_override.json",
        {
            "requested_action": "direct full-budget training followed by evaluation",
            "prerequisite_gates_bypassed": True,
            "failed_gate": "tests/test_v26r_staged_run.py timeout at 900 seconds",
            "recorded_at": time.time(),
        },
    )
    try:
        prepare_structure(root, args.checkpoint_tariff.resolve())
        phase_status(root, "train", "running", False, started_at=time.time())
        training = r15.phase_train(root)
        phase_status(
            root,
            "train",
            "complete" if training.get("pass") else "failed",
            bool(training.get("pass")),
            completed_at=time.time(),
            result=training,
        )
        if not training.get("pass"):
            return 1

        phase_status(root, "evaluate", "running", False, started_at=time.time())
        evaluation = r15.phase_evaluate(root)
        phase_status(
            root,
            "evaluate",
            "complete" if evaluation.get("pass") else "failed",
            bool(evaluation.get("pass")),
            completed_at=time.time(),
            result=evaluation,
        )
        return 0 if evaluation.get("pass") else 2
    finally:
        subprocess.Popen(
            ["/bin/launchctl", "remove", args.label],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )


if __name__ == "__main__":
    raise SystemExit(main())
