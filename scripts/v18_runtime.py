"""Small resumable process runner shared by V18 launchers."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping, Sequence


THREAD_ENV = {
    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1", "MPLBACKEND": "Agg",
    "PYTHONDONTWRITEBYTECODE": "1",
}


@dataclass(frozen=True)
class Job:
    name: str
    command: tuple[str, ...]
    output: Path
    log: Path
    manifest: Path
    requires: tuple[str, ...] = ()
    tier: int = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def command_hash(command: Sequence[str]) -> str:
    payload = json.dumps(list(command), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_progress(
    path: Path, *, job: str, state: str, attempt: int, seconds: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "time": utc_now(), "job": str(job), "state": str(state),
        "attempt": int(attempt), "seconds": round(float(seconds), 3),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def completed(job: Job) -> bool:
    try:
        manifest = json.loads(job.manifest.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return (
        manifest.get("state") == "complete"
        and manifest.get("command_hash") == command_hash(job.command)
        and job.output.is_file()
        and job.output.stat().st_size > 0
    )


def run_job(
    job: Job, *, cwd: Path, dry_run: bool = False,
    progress: Path | None = None, max_attempts: int = 3,
) -> str:
    if completed(job):
        return "cached"
    try:
        prior = json.loads(job.manifest.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        prior = {}
    attempts = (
        list(prior.get("attempts") or [])
        if prior.get("command_hash") == command_hash(job.command)
        else []
    )
    payload = {
        "name": job.name, "state": "planned" if dry_run else "running",
        "command": list(job.command), "command_hash": command_hash(job.command),
        "output": str(job.output), "log": str(job.log), "updated_utc": utc_now(),
        "requires": list(job.requires), "tier": int(job.tier),
        "attempts": attempts,
    }
    atomic_json(job.manifest, payload)
    if dry_run:
        return "planned"
    job.log.parent.mkdir(parents=True, exist_ok=True)
    job.output.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(THREAD_ENV)
    first_attempt = len(attempts) + 1
    if first_attempt > int(max_attempts):
        terminal = attempts[-1] if attempts else {}
        payload.update({
            "state": "failed",
            "returncode": prior.get(
                "returncode", terminal.get("returncode")
            ),
            "error": prior.get(
                "error", terminal.get("error", "attempt budget exhausted")
            ),
            "updated_utc": utc_now(),
        })
        atomic_json(job.manifest, payload)
        return "failed"
    for attempt in range(first_attempt, int(max_attempts) + 1):
        started = time.monotonic()
        payload.update({"state": "running", "attempt": attempt, "updated_utc": utc_now()})
        atomic_json(job.manifest, payload)
        if progress is not None:
            append_progress(progress, job=job.name, state="running", attempt=attempt, seconds=0)
        returncode = 127
        error = ""
        try:
            with job.log.open("ab") as log:
                process = subprocess.run(
                    list(job.command), cwd=cwd, env=environment,
                    stdout=log, stderr=subprocess.STDOUT, check=False,
                )
            returncode = int(process.returncode)
            if returncode == 0 and (not job.output.is_file() or job.output.stat().st_size == 0):
                returncode = 125
                error = f"job did not write {job.output}"
        except BaseException as exc:  # preserve the failed attempt, then retry
            error = repr(exc)
        elapsed = time.monotonic() - started
        attempts.append({
            "attempt": attempt, "returncode": returncode,
            "seconds": round(elapsed, 3), "error": error,
            "finished_utc": utc_now(),
        })
        success = returncode == 0
        payload.update({
            "state": "complete" if success else ("retrying" if attempt < max_attempts else "failed"),
            "returncode": returncode, "error": error,
            "attempts": attempts, "updated_utc": utc_now(),
        })
        atomic_json(job.manifest, payload)
        if progress is not None:
            append_progress(
                progress, job=job.name,
                state="complete" if success else payload["state"],
                attempt=attempt, seconds=elapsed,
            )
        if success:
            return "complete"
    return "failed"


def run_jobs(
    jobs: Sequence[Job], *, cwd: Path, workers: int, dry_run: bool = False,
    progress: Path | None = None,
) -> dict[str, str]:
    workers = int(workers)
    if not 1 <= workers <= 10:
        raise ValueError("V18 runners allow between 1 and 10 Core processes")
    if len({job.name for job in jobs}) != len(jobs):
        raise ValueError("V18 job names must be unique")
    progress = progress or (cwd / "runs/v18/controller/progress.jsonl")
    if dry_run:
        return {
            job.name: run_job(job, cwd=cwd, dry_run=True, progress=None)
            for job in jobs
        }
    results: dict[str, str] = {}
    pending = {job.name: job for job in jobs}
    known = set(pending)
    status_path = progress.parent / "status.json"

    def rewrite_status() -> None:
        counts = {
            state: sum(value == state for value in results.values())
            for state in ("complete", "cached", "failed", "skipped")
        }
        counts["pending"] = len(pending)
        atomic_json(status_path, {
            "state": "running" if pending else "complete",
            "updated_utc": utc_now(), "counts": counts,
            "workers": workers, "projected_hours_left": None,
        })
    while pending:
        for name, job in list(pending.items()):
            if any(req not in known for req in job.requires):
                results[name] = "skipped"
                append_progress(progress, job=name, state="skipped", attempt=0, seconds=0)
                atomic_json(job.manifest, {
                    "name": name, "state": "skipped",
                    "reason": "missing dependency", "requires": list(job.requires),
                })
                del pending[name]
                rewrite_status()
            elif any(results.get(req) in {"failed", "skipped"} for req in job.requires):
                results[name] = "skipped"
                append_progress(progress, job=name, state="skipped", attempt=0, seconds=0)
                atomic_json(job.manifest, {
                    "name": name, "state": "skipped",
                    "reason": "failed dependency", "requires": list(job.requires),
                })
                del pending[name]
                rewrite_status()
        ready = sorted([
            job for job in pending.values()
            if all(results.get(req) in {"complete", "cached"} for req in job.requires)
        ], key=lambda job: job.tier)
        if not ready and pending:
            for name, job in list(pending.items()):
                results[name] = "skipped"
                append_progress(progress, job=name, state="skipped", attempt=0, seconds=0)
                atomic_json(job.manifest, {
                    "name": name, "state": "skipped",
                    "reason": "dependency cycle", "requires": list(job.requires),
                })
                del pending[name]
                rewrite_status()
            break
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(run_job, job, cwd=cwd, progress=progress): job
                for job in ready
            }
            for future in as_completed(futures):
                job = futures[future]
                results[job.name] = future.result()
                del pending[job.name]
                rewrite_status()
    return results
