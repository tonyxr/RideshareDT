import json
from pathlib import Path
import subprocess
import sys

from digital_twin.orchestrator import CYCLE_STAGES


ROOT = Path(__file__).resolve().parent


def test_cycle_cli_executes_commands_and_hashes_artifacts(tmp_path):
    artifact = tmp_path / "ingested.json"
    config = {"market": "test", "stages": {}}
    for stage in CYCLE_STAGES:
        config["stages"][stage] = {"metadata": {"stage": stage}}
    config["stages"]["ingest"] = {
        "command": [
            sys.executable,
            "-c",
            f"from pathlib import Path; Path({str(artifact)!r}).write_text('ok')",
        ],
        "artifacts": [str(artifact)],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    result = subprocess.run(
        [
            sys.executable, str(ROOT / "scripts/run_v30_cycle.py"),
            "--config", str(config_path), "--cycle-id", "cycle-test",
            "--work-dir", str(tmp_path / "cycles"),
        ],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    manifest = json.loads(result.stdout.strip().splitlines()[-1])
    assert manifest["status"] == "shadow_complete"
    assert manifest["completed_stages"] == list(CYCLE_STAGES)
    ingested = manifest["stage_outputs"]["ingest"]
    assert ingested["artifacts"][0]["sha256"]
    assert ingested["returncode"] == 0
