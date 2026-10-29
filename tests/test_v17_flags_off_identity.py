"""T0.1: flags-off V17 must reproduce its untouched 15d source tree."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys


BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent
SOURCE_15D = ROOT / "rideshare_v26r_revision15d"
RUNTIME_PYTHON = (
    ROOT / ".venv/bin/python"
    if (ROOT / ".venv/bin/python").is_file()
    else Path(sys.executable)
)


def _run_flags_off(tree: Path, output: Path) -> tuple[int, bytes]:
    destination = output / tree.name
    destination.mkdir(parents=True, exist_ok=True)
    report_prefix = os.path.relpath(destination / "trace", tree)
    command = [
        str(RUNTIME_PYTHON),
        str(tree / "Core.py"),
        "--market",
        "Boston",
        "--choice_mode",
        "cognitive",
        "--firm1_mode",
        "static",
        "--firm2_mode",
        "adaptive_best_response_aggressive",
        "--run_experiment",
        "--credit_assignment_v25",
        "--v26",
        "--deterministic_experiment_seed",
        "--seed",
        "17001",
        "--experiment_seed",
        "17001",
        "--policy_objective",
        "profit_maximization",
        "--pool",
        "300",
        "--threshold_profile_source",
        "cached",
        "--threshold_cache_path",
        str(
            ROOT
            / "artifacts/v26_confirmatory/thresholds/boston/llm_numeric_primary.jsonl"
        ),
        "--calibration_json",
        str(ROOT / "data/processed/boston/calibration.json"),
        "--gpt_threshold_mode",
        "llm_numeric",
        "--train_timesteps",
        "0",
        "--train_customers",
        "20",
        "--train_steps_per_day",
        "6",
        # 120 market steps are exactly 20 configured simulation days.
        "--eval_timesteps",
        "120",
        "--eval_customers",
        "20",
        "--no-truncation_bootstrap",
        "--report_prefix",
        report_prefix,
    ]
    environment = dict(
        os.environ,
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        KMP_USE_SHM="0",
        MPLBACKEND="Agg",
        PYTHONDONTWRITEBYTECODE="1",
    )
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        command,
        cwd=tree,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=900,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"flags-off run failed in {tree} with {completed.returncode}:\n"
            + completed.stdout.decode("utf-8", errors="replace")[-8000:]
        )
    trace = destination / "trace_evaluation_diagnostics.csv"
    if not trace.is_file():
        raise AssertionError(f"missing flags-off trace: {trace}")
    return completed.returncode, trace.read_bytes()


def test_t0_1_flags_off_20_days_are_byte_identical_to_15d(tmp_path):
    assert SOURCE_15D.is_dir(), "the registered 15d source tree is unavailable"
    source_status, source_trace = _run_flags_off(SOURCE_15D, tmp_path)
    v17_status, v17_trace = _run_flags_off(BASE, tmp_path)

    assert source_status == v17_status == 0
    assert hashlib.sha256(v17_trace).digest() == hashlib.sha256(
        source_trace
    ).digest()
    assert v17_trace == source_trace
