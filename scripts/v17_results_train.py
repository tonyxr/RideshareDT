#!/usr/bin/env python3
"""Launch the registered V17 production and ablation training runs."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Iterable, Sequence

try:
    from scripts.v17_config import BASE, core_args, load_config
except ModuleNotFoundError:  # Direct execution sets scripts/ as sys.path[0].
    from v17_config import BASE, core_args, load_config


THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "MPLBACKEND": "Agg",
    "PYTHONDONTWRITEBYTECODE": "1",
}


@dataclass(frozen=True)
class RunSpec:
    name: str
    city: str
    seed: int
    kind: str = "main"


RUN_SPECS = (
    RunSpec("nyc_s101", "nyc", 101),
    RunSpec("nyc_s202", "nyc", 202),
    RunSpec("nyc_s303", "nyc", 303),
    RunSpec("boston_s101", "boston", 101),
    RunSpec("boston_s202", "boston", 202),
    RunSpec("chicago_s101", "chicago", 101),
    RunSpec("chicago_s202", "chicago", 202),
    RunSpec(
        "nyc_s101_no_driver_obs",
        "nyc",
        101,
        "no_driver_observation",
    ),
    RunSpec("nyc_s101_own_profit", "nyc", 101, "own_profit"),
    RunSpec("nyc_s101_plain_ppo", "nyc", 101, "plain_ppo"),
)
RUN_BY_NAME = {spec.name: spec for spec in RUN_SPECS}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_status(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _pid_alive(pid: int) -> bool:
    if int(pid) <= 0:
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _drop_value_options(
    command: Sequence[str], option_names: Iterable[str]
) -> list[str]:
    drop = set(option_names)
    result: list[str] = []
    index = 0
    while index < len(command):
        value = str(command[index])
        if value in drop:
            index += 2
            continue
        result.append(value)
        index += 1
    return result


def build_command(
    spec: RunSpec,
    *,
    days: int,
    output_root: Path,
    config: dict[str, Any] | None = None,
    diagnostic_prefix: Path | None = None,
) -> list[str]:
    if int(days) < 4:
        raise ValueError("--days must be at least 4")
    config = load_config() if config is None else config
    run_dir = (output_root / spec.name).resolve()
    values = core_args(
        config,
        city=spec.city,
        seed=spec.seed,
        output=run_dir,
        days=int(days),
        branch_horizon_days=5,
    )
    if diagnostic_prefix is not None:
        prefix_index = values.index("--report_prefix") + 1
        values[prefix_index] = os.path.relpath(
            diagnostic_prefix.resolve(), BASE
        )
    if spec.kind == "no_driver_observation":
        values.append("--v17_ablate_driver_observation")
    elif spec.kind == "own_profit":
        values.append("--v17_ablate_profit_lead")
    elif spec.kind == "plain_ppo":
        values = _drop_value_options(
            values,
            {
                "--v16_branch_mode",
                "--v16_branch_horizon_days",
                "--v16_branch_design",
                "--v16_audit_every",
                "--v16_store_cap",
                "--v16_value_head",
                "--v16_policy_mode",
                "--v16_tau_start",
                "--v16_tau_end",
                "--v16_policy_epsilon",
                "--v17_move_radius",
                "--v17_temperature_mode",
                "--v17_tau_rel_start",
                "--v17_tau_rel_end",
            },
        )
        values.extend(("--v16_policy_mode", "ppo"))
    values.extend(
        (
            "--training_resume_state",
            str(run_dir / "resume/training_state.pt"),
            "--training_resume_every_days",
            "1",
        )
    )
    return [
        sys.executable,
        "-u",
        str(BASE / "Core.py"),
        *values,
    ]


def _reserve_attempt_prefix(run_dir: Path, city: str) -> tuple[str, Path]:
    attempts_dir = run_dir / "attempts"
    attempts_dir.mkdir(parents=True, exist_ok=True)
    for number in range(1, 10_000):
        attempt_id = f"attempt-{number:04d}"
        attempt_dir = attempts_dir / attempt_id
        try:
            attempt_dir.mkdir()
        except FileExistsError:
            continue
        return attempt_id, attempt_dir / city
    raise RuntimeError(f"attempt namespace exhausted for {run_dir}")


def _selected_specs(raw: str) -> tuple[RunSpec, ...]:
    if not str(raw).strip():
        return RUN_SPECS
    names = tuple(name.strip() for name in str(raw).split(",") if name.strip())
    unknown = sorted(set(names) - set(RUN_BY_NAME))
    if unknown:
        raise ValueError(f"unknown run name(s): {', '.join(unknown)}")
    if len(set(names)) != len(names):
        raise ValueError("--runs contains a duplicate run name")
    return tuple(RUN_BY_NAME[name] for name in names)


def _status_payload(
    spec: RunSpec,
    *,
    state: str,
    days: int,
    run_dir: Path,
    command: Sequence[str],
    pid: int | None = None,
    exit_code: int | None = None,
    message: str = "",
    attempt_id: str = "",
    diagnostic_prefix: Path | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "v17-results-train-status-v1",
        "updated_at_utc": _utc_now(),
        "state": str(state),
        "run": asdict(spec),
        "days_expected": int(days),
        "pid": None if pid is None else int(pid),
        "exit_code": None if exit_code is None else int(exit_code),
        "cwd": str(BASE),
        "command": list(command),
        "training_log": str(run_dir / "training.log"),
        "resume_state": str(run_dir / "resume/training_state.pt"),
        "policy": str(run_dir / "policy.pt"),
        "attempt_id": str(attempt_id),
        "diagnostic_prefix": (
            "" if diagnostic_prefix is None else str(diagnostic_prefix)
        ),
        "message": str(message),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=1200)
    parser.add_argument("--output", type=Path, default=BASE / "runs/v17_results")
    parser.add_argument(
        "--runs",
        default="",
        help="Comma-separated registered run names; default launches all ten.",
    )
    args = parser.parse_args(argv)
    specs = _selected_specs(args.runs)
    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    config = load_config()
    env = dict(os.environ)
    env.update(THREAD_ENV)
    children: dict[str, subprocess.Popen[Any]] = {}
    commands: dict[str, list[str]] = {}
    external: dict[str, int] = {}
    attempts: dict[str, tuple[str, Path]] = {}

    for spec in specs:
        run_dir = output_root / spec.name
        status_path = run_dir / "status.json"
        status = _read_status(status_path)
        if (
            status.get("state") == "completed"
            and int(status.get("days_expected", -1)) == int(args.days)
            and (run_dir / "policy.pt").is_file()
        ):
            print(f"[skip completed] {spec.name}", flush=True)
            continue
        prior_pid = int(status.get("pid") or 0)
        if status.get("state") == "running" and _pid_alive(prior_pid):
            external[spec.name] = prior_pid
            print(f"[already running pid={prior_pid}] {spec.name}", flush=True)
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        attempt_id, diagnostic_prefix = _reserve_attempt_prefix(
            run_dir, spec.city
        )
        attempts[spec.name] = (attempt_id, diagnostic_prefix)
        command = build_command(
            spec,
            days=args.days,
            output_root=output_root,
            config=config,
            diagnostic_prefix=diagnostic_prefix,
        )
        commands[spec.name] = command
        log_path = run_dir / "training.log"
        log_stream = log_path.open("a", encoding="utf-8", buffering=1)
        process = subprocess.Popen(
            command,
            cwd=BASE,
            env=env,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log_stream.close()
        children[spec.name] = process
        _atomic_json(
            status_path,
            _status_payload(
                spec,
                state="running",
                days=args.days,
                run_dir=run_dir,
                command=command,
                pid=process.pid,
                message="launched or resumed from daily state",
                attempt_id=attempt_id,
                diagnostic_prefix=diagnostic_prefix,
            ),
        )
        print(f"[launched pid={process.pid}] {spec.name}", flush=True)

    interrupted = False

    def _forward(signum: int, _frame: Any) -> None:
        nonlocal interrupted
        interrupted = True
        for process in children.values():
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signum)
                except ProcessLookupError:
                    pass

    signal.signal(signal.SIGINT, _forward)
    signal.signal(signal.SIGTERM, _forward)

    failed = False
    while children or external:
        for name, process in tuple(children.items()):
            exit_code = process.poll()
            if exit_code is None:
                continue
            spec = RUN_BY_NAME[name]
            run_dir = output_root / name
            attempt_id, diagnostic_prefix = attempts[name]
            state = "completed" if exit_code == 0 else "failed"
            if exit_code != 0:
                failed = True
            _atomic_json(
                run_dir / "status.json",
                _status_payload(
                    spec,
                    state=state,
                    days=args.days,
                    run_dir=run_dir,
                    command=commands[name],
                    pid=process.pid,
                    exit_code=exit_code,
                    message="Core process exited",
                    attempt_id=attempt_id,
                    diagnostic_prefix=diagnostic_prefix,
                ),
            )
            print(f"[{state} exit={exit_code}] {name}", flush=True)
            children.pop(name)
        for name, pid in tuple(external.items()):
            if _pid_alive(pid):
                continue
            external.pop(name)
            status = _read_status(output_root / name / "status.json")
            if status.get("state") != "completed":
                failed = True
                print(
                    f"[external process ended; inspect status] {name}",
                    flush=True,
                )
        if children or external:
            time.sleep(2.0)

    if interrupted:
        return 130
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
