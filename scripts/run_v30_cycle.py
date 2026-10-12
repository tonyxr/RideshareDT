#!/usr/bin/env python3
"""Run or resume one crash-safe v30 optimization cycle from JSON config."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from digital_twin.orchestrator import (  # noqa: E402
    CYCLE_STAGES,
    ContinuousOptimizationOrchestrator,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stage_callback(
    stage: str,
    specification: Mapping[str, Any],
    cycle_dir: Path,
):
    def run(_manifest):
        command = specification.get("command")
        declared = dict(specification.get("metadata", {}) or {})
        if command is None:
            return {"mode": "record_only", "metadata": declared}
        if (
            not isinstance(command, list)
            or not command
            or any(not isinstance(value, str) or not value for value in command)
        ):
            raise ValueError(f"stage {stage!r} command must be a non-empty string list")
        log_path = cycle_dir / f"{stage}.log"
        with log_path.open("wb") as stream:
            completed = subprocess.run(
                command,
                cwd=str(ROOT),
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(
                f"stage {stage!r} exited {completed.returncode}; see {log_path}"
            )
        artifacts = []
        for raw_path in specification.get("artifacts", []):
            path = Path(str(raw_path))
            if not path.is_absolute():
                path = ROOT / path
            if not path.is_file():
                raise FileNotFoundError(f"stage {stage!r} artifact missing: {path}")
            artifacts.append({
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            })
        return {
            "mode": "command",
            "command": command,
            "returncode": completed.returncode,
            "log": str(log_path.resolve()),
            "log_sha256": _sha256(log_path),
            "artifacts": artifacts,
            "metadata": declared,
        }
    return run


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--cycle-id", required=True)
    parser.add_argument("--work-dir", default="artifacts/v30_cycles")
    parser.add_argument("--authorize", action="store_true")
    parser.add_argument("--authorizer", default="")
    parser.add_argument("--change-ticket", default="")
    parser.add_argument("--policy-sha256", default="")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    config = json.loads(config_path.read_text())
    stages = dict(config.get("stages", {}) or {})
    missing = [stage for stage in CYCLE_STAGES if stage not in stages]
    if missing:
        parser.error("configuration is missing stages: " + ",".join(missing))
    cycle_dir = Path(args.work_dir).expanduser().resolve() / args.cycle_id
    callbacks = {
        stage: _stage_callback(stage, dict(stages[stage]), cycle_dir)
        for stage in CYCLE_STAGES
    }
    runner = ContinuousOptimizationOrchestrator(
        root=Path(args.work_dir).expanduser().resolve(),
        cycle_id=args.cycle_id,
        configuration=config,
        callbacks=callbacks,
    )
    manifest = runner.run()
    if args.authorize:
        manifest = runner.authorize_deployment(
            authorizer=args.authorizer,
            change_ticket=args.change_ticket,
            policy_artifact_sha256=args.policy_sha256,
        )
    print(json.dumps(manifest.payload(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
