from __future__ import annotations

"""Immutable trained-policy archive and reference resolution helpers."""

import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


MANIFEST_NAME = "manifest.jsonl"


def _absolute(path: str) -> str:
    return os.path.abspath(os.path.expanduser(str(path)))


def safe_model_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "model").strip())
    return cleaned.strip("-._") or "model"


def utc_run_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def allocate_archive_path(registry_dir: str, model_id: str) -> str:
    directory = _absolute(registry_dir)
    os.makedirs(directory, exist_ok=True)
    base = f"{safe_model_id(model_id)}__{utc_run_stamp()}"
    candidate = os.path.join(directory, f"{base}.pt")
    suffix = 1
    while os.path.exists(candidate):
        candidate = os.path.join(directory, f"{base}__{suffix}.pt")
        suffix += 1
    return candidate


def append_manifest(registry_dir: str, record: Dict[str, Any]) -> str:
    directory = _absolute(registry_dir)
    os.makedirs(directory, exist_ok=True)
    manifest = os.path.join(directory, MANIFEST_NAME)
    payload = dict(record)
    payload.setdefault("registered_at", datetime.now(timezone.utc).isoformat())
    with open(manifest, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
    return manifest


def list_registered_models(registry_dir: str) -> List[Dict[str, Any]]:
    manifest = os.path.join(_absolute(registry_dir), MANIFEST_NAME)
    if not os.path.isfile(manifest):
        return []
    records: List[Dict[str, Any]] = []
    with open(manifest, "r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            archive_path = record.get("archive_path")
            if archive_path and os.path.isfile(str(archive_path)):
                records.append(record)
    return records


def resolve_model_reference(reference: str, registry_dir: str) -> str:
    """Resolve an explicit path, registry model id, or ``latest`` reference."""

    raw = str(reference or "").strip()
    if not raw:
        raise FileNotFoundError("empty trained-model reference")

    direct = _absolute(raw)
    if os.path.isfile(direct):
        return direct

    records = list_registered_models(registry_dir)
    if raw.lower() == "latest" and records:
        return _absolute(str(records[-1]["archive_path"]))

    matching = [
        record
        for record in records
        if raw in {
            str(record.get("model_id", "")),
            str(record.get("archive_id", "")),
            os.path.basename(str(record.get("archive_path", ""))),
        }
    ]
    if matching:
        return _absolute(str(matching[-1]["archive_path"]))

    candidate = os.path.join(_absolute(registry_dir), raw)
    if os.path.isfile(candidate):
        return candidate
    if not raw.endswith(".pt") and os.path.isfile(candidate + ".pt"):
        return candidate + ".pt"

    known = sorted({str(record.get("model_id", "")) for record in records if record.get("model_id")})
    suffix = f" Known model ids: {', '.join(known[-12:])}." if known else ""
    raise FileNotFoundError(f"trained model {reference!r} was not found.{suffix}")


def default_model_id(report_prefix: str) -> str:
    return safe_model_id(os.path.basename(str(report_prefix).rstrip(os.sep)) or "trained-policy")

