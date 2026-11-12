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


def run_job(job: Job, *, cwd: Path, dry_run: bool = False) -> str:
    if completed(job):
        return "cached"
    payload = {
        "name": job.name, "state": "planned" if dry_run else "running",
        "command": list(job.command), "command_hash": command_hash(job.command),
        "output": str(job.output), "log": str(job.log), "updated_utc": utc_now(),
    }
    atomic_json(job.manifest, payload)
    if dry_run:
        return "planned"
    job.log.parent.mkdir(parents=True, exist_ok=True)
    job.output.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(THREAD_ENV)
    with job.log.open("ab") as log:
        process = subprocess.run(
            list(job.command), cwd=cwd, env=environment,
            stdout=log, stderr=subprocess.STDOUT, check=False,
        )
    payload.update({
        "state": "complete" if process.returncode == 0 else "failed",
        "returncode": int(process.returncode), "updated_utc": utc_now(),
    })
    atomic_json(job.manifest, payload)
    if process.returncode:
        raise RuntimeError(f"{job.name} failed with exit code {process.returncode}")
    if not job.output.is_file() or job.output.stat().st_size == 0:
        raise RuntimeError(f"{job.name} did not write {job.output}")
    return "complete"


def run_jobs(
    jobs: Sequence[Job], *, cwd: Path, workers: int, dry_run: bool = False
) -> dict[str, str]:
    workers = int(workers)
    if not 1 <= workers <= 10:
        raise ValueError("V18 runners allow between 1 and 10 Core processes")
    if dry_run:
        return {job.name: run_job(job, cwd=cwd, dry_run=True) for job in jobs}
    results: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_job, job, cwd=cwd): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            results[job.name] = future.result()
    return results
