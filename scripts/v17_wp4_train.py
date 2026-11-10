#!/usr/bin/env python3
"""Launch the two compute-matched V17 PPO robustness runs."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Sequence


BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from diagnostic_logging import atomic_json  # noqa: E402
from Core import Core  # noqa: E402,F401
from scripts.v17_config import load_config  # noqa: E402
from scripts.v17_results_train import (  # noqa: E402
    THREAD_ENV,
    RunSpec,
    _reserve_attempt_prefix,
    _status_payload,
    build_command,
)
from scripts.v17_robust_common import set_option  # noqa: E402


RUNS = {
    "nyc_s101_ppo_tuned": RunSpec("nyc_s101_ppo_tuned", "nyc", 101, "plain_ppo"),
    "nyc_s101_ppo_matched": RunSpec("nyc_s101_ppo_matched", "nyc", 101, "plain_ppo"),
}


def _attempt_prefix(run_dir: Path, city: str) -> tuple[str, Path]:
    """Reuse the checkpoint's journal on resume; otherwise reserve an attempt."""
    resume = run_dir / "resume/training_state.pt"
    if not resume.is_file():
        return _reserve_attempt_prefix(run_dir, city)

    import torch

    # Historical training states pickle Core as ``__main__.Core`` because
    # Core.py was the executable.  Provide that trusted local alias before
    # loading the preserved checkpoint.
    setattr(sys.modules["__main__"], "Core", Core)
    artifact = torch.load(resume, map_location="cpu", weights_only=False)
    contract = artifact.get("contract") or {}
    raw_prefix = str(contract.get("diagnostic_prefix") or "").strip()
    if not raw_prefix:
        raise ValueError(f"resume checkpoint lacks diagnostic_prefix: {resume}")
    prefix = Path(raw_prefix)
    if not prefix.is_absolute():
        prefix = (BASE / prefix).resolve()
    else:
        prefix = prefix.resolve()
    attempts_root = (run_dir / "attempts").resolve()
    try:
        prefix.relative_to(attempts_root)
    except ValueError as exc:
        raise ValueError(
            f"resume diagnostic_prefix is outside the run attempt store: {prefix}"
        ) from exc
    if prefix.name != city:
        raise ValueError(
            f"resume diagnostic_prefix city mismatch: {prefix.name} != {city}"
        )
    return prefix.parent.name, prefix


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _alive(pid: int) -> bool:
    if int(pid) <= 0:
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _selected(raw: str) -> tuple[RunSpec, ...]:
    names = tuple(token.strip() for token in str(raw).split(",") if token.strip())
    unknown = sorted(set(names) - set(RUNS))
    if unknown or not names or len(set(names)) != len(names):
        raise ValueError(f"invalid --runs: {raw}")
    return tuple(RUNS[name] for name in names)


def _days_for(spec: RunSpec, default: int, matched_file: Path | None) -> int:
    if spec.name != "nyc_s101_ppo_matched":
        return int(default)
    if matched_file is None:
        return int(default)
    payload = _load(matched_file.resolve())
    days = int(payload.get("matched_days", 0))
    if days < 1200 or days % 30:
        raise ValueError("matched day count must be >=1200 and divisible by 30")
    return days


def _tuned_command(
    spec: RunSpec, *, days: int, output: Path,
    config: dict[str, Any], diagnostic_prefix: Path,
) -> list[str]:
    command = build_command(
        spec, days=days, output_root=output, config=config,
        diagnostic_prefix=diagnostic_prefix,
    )
    command = set_option(command, "--ppo_rollout_episodes", 1)
    command = set_option(command, "--ppo_min_rollout_transitions", 30)
    command = set_option(command, "--ppo_batch_size", 32)
    return command


def _verify_smoke(output: Path) -> dict[str, Any]:
    run_dir = output / "nyc_s101_ppo_tuned"
    status = _load(run_dir / "status.json")
    if status.get("state") != "completed" or int(status.get("days_expected", -1)) != 120:
        raise ValueError("WP4 smoke status is not a completed 120-day run")
    prefix = Path(str(status["diagnostic_prefix"]))
    diagnostics = prefix.with_name(f"{prefix.name}_training_diagnostics.csv")
    with diagnostics.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    updates = sum(str(row.get("ppo_update_performed", "")).lower() == "true" for row in rows)
    transitions: list[int] = []
    previous = None
    for row in rows:
        stage = str(row.get("training_stage", ""))
        if previous is not None and stage != previous:
            transitions.append(int(float(row["batch"])))
        previous = stage
    expected = [6, 12, 48]
    result = {
        "schema_version": "v17-robust-wp4-smoke-v1",
        "state": "completed" if updates >= 3 and transitions[:3] == expected else "failed",
        "pass": updates >= 3 and transitions[:3] == expected,
        "ppo_updates": updates,
        "stage_transition_batches": transitions,
        "expected_stage_transition_batches": expected,
        "diagnostics": str(diagnostics),
    }
    atomic_json(output / "status.json", result)
    if not result["pass"]:
        raise ValueError(f"WP4 smoke contract failed: {result}")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--days", type=int, default=1200)
    parser.add_argument("--matched-days-file", type=Path)
    parser.add_argument("--output", type=Path, default=BASE / "runs/v17_results")
    args = parser.parse_args(argv)
    if args.days < 120:
        parser.error("--days must be at least 120")
    specs = _selected(args.runs)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = load_config()
    env = dict(os.environ)
    env.update(THREAD_ENV)
    children: dict[str, subprocess.Popen[Any]] = {}
    external: dict[str, int] = {}
    metadata: dict[str, tuple[RunSpec, int, list[str], str, Path]] = {}
    for spec in specs:
        days = _days_for(spec, args.days, args.matched_days_file)
        run_dir = output / spec.name
        status_path = run_dir / "status.json"
        status = _load(status_path)
        if (
            status.get("state") == "completed"
            and int(status.get("days_expected", -1)) == days
            and (run_dir / "policy.pt").is_file()
        ):
            print(f"[skip completed] {spec.name}", flush=True)
            continue
        prior_pid = int(status.get("pid") or 0)
        if status.get("state") == "running" and _alive(prior_pid):
            external[spec.name] = prior_pid
            print(f"[already running pid={prior_pid}] {spec.name}", flush=True)
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        attempt_id, prefix = _attempt_prefix(run_dir, spec.city)
        command = _tuned_command(
            spec, days=days, output=output, config=config,
            diagnostic_prefix=prefix,
        )
        log = (run_dir / "training.log").open("a", encoding="utf-8", buffering=1)
        process = subprocess.Popen(
            command, cwd=BASE, env=env, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
        log.close()
        children[spec.name] = process
        metadata[spec.name] = (spec, days, command, attempt_id, prefix)
        atomic_json(status_path, _status_payload(
            spec, state="running", days=days, run_dir=run_dir,
            command=command, pid=process.pid,
            message="WP4 PPO launched or resumed from daily state",
            attempt_id=attempt_id, diagnostic_prefix=prefix,
        ))
        print(f"[launched pid={process.pid}] {spec.name} days={days}", flush=True)

    interrupted = False

    def forward(signum: int, _frame: Any) -> None:
        nonlocal interrupted
        interrupted = True
        for process in children.values():
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signum)
                except ProcessLookupError:
                    pass

    signal.signal(signal.SIGINT, forward)
    signal.signal(signal.SIGTERM, forward)
    failed = False
    while children or external:
        for name, process in tuple(children.items()):
            code = process.poll()
            if code is None:
                continue
            spec, days, command, attempt_id, prefix = metadata[name]
            run_dir = output / name
            state = "completed" if code == 0 else "failed"
            failed |= code != 0
            atomic_json(run_dir / "status.json", _status_payload(
                spec, state=state, days=days, run_dir=run_dir,
                command=command, pid=process.pid, exit_code=code,
                message="WP4 PPO Core process exited",
                attempt_id=attempt_id, diagnostic_prefix=prefix,
            ))
            print(f"[{state} exit={code}] {name}", flush=True)
            children.pop(name)
        for name, pid in tuple(external.items()):
            if _alive(pid):
                continue
            external.pop(name)
            if _load(output / name / "status.json").get("state") != "completed":
                failed = True
        if children or external:
            time.sleep(2)
    if interrupted:
        return 130
    if not failed and len(specs) == 1 and specs[0].name == "nyc_s101_ppo_tuned" and args.days == 120:
        _verify_smoke(output)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
