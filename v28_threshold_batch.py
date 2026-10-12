"""Resumable, fail-closed v28 rider-threshold Batch API pipeline.

The historical ``Core`` bootstrap sends synchronous requests serially and only
writes a cache after the whole population completes.  That is convenient for a
small experiment, but it is unsafe for a paid 60,000-profile regeneration: a
crash loses progress and API fallbacks can be persisted as if they were native
model judgments.

This module keeps prompt construction in :mod:`Core` while adding a durable
Batch API control plane:

* deterministic, content-addressed request shards;
* stable ``custom_id`` to source-row mappings;
* atomic manifests and downloaded outputs;
* prepare/submit/status/collect/retry-missing/validate stages;
* no fallback rows in a final cache; and
* exact preservation of the source rider and cold-start evidence.

No function performs network I/O unless explicitly called by the ``submit``,
``status``, or ``collect`` command paths.  API credentials are read from the
environment at call time and are never serialized.
"""

from __future__ import annotations

import contextlib
import copy
import datetime as dt
import fcntl
import hashlib
import io
import json
import math
import os
import re
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = 1
API_BASE = "https://api.openai.com/v1"
RESPONSES_ENDPOINT = "/v1/responses"
DEFAULT_MODEL = "gpt-4.1-2025-04-14"
DEFAULT_BATCH_SIZE = 20
DEFAULT_MAX_SHARD_BYTES = 190_000_000
DEFAULT_MAX_QUEUED_TOKENS = 1_200_000
DEFAULT_GLOBAL_MAX_QUEUED_TOKENS = 1_350_000
DEFAULT_CHARS_PER_TOKEN = 2.5
DEFAULT_OUTPUT_TOKENS_PER_REQUEST = 350
NATIVE_PROMPT_PROTOCOL = "numeric_profile_threshold_v12_uncapped"

# Pricing assumptions are explicit and overrideable in the CLI.  They are
# estimates only; the finalized manifest reports actual API usage tokens.
STANDARD_INPUT_USD_PER_MTOK = 2.0
STANDARD_OUTPUT_USD_PER_MTOK = 8.0
BATCH_INPUT_USD_PER_MTOK = 1.0
BATCH_OUTPUT_USD_PER_MTOK = 4.0

CITY_NAMES = {
    "nyc": "New York City",
    "boston": "Boston",
    "chicago": "Chicago",
}

ACTIVE_BATCH_STATUSES = {
    "submitting",
    "validating",
    "in_progress",
    "finalizing",
    "cancelling",
}
TERMINAL_BATCH_STATUSES = {"completed", "failed", "expired", "cancelled"}

SOURCE_EXCLUDED_PROFILE_KEYS = {
    "ColdstartRides",
    "PriceThreshold",
    "PriceThresholdRationale",
    "PriceThresholdSource",
    "ProfileMarket",
    "ThresholdEvidenceProtocol",
    "ThresholdEvidenceFirm1Multiplier",
    "ThresholdEvidenceFirm2Multiplier",
    "ThresholdPromptProtocol",
    "ExperimentalUnboundedEconomics",
}


class ThresholdBatchError(RuntimeError):
    """Raised when a resumability or provenance invariant is violated."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_digest(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _atomic_bytes(path: str | Path, payload: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def atomic_json(path: str | Path, value: Any) -> None:
    _atomic_bytes(path, (json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n").encode("utf-8"))


@contextlib.contextmanager
def work_lock(work_dir: str | Path) -> Iterator[None]:
    directory = Path(work_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / ".manifest.lock"
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _normalized_market(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def _source_base_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Remove every historical threshold annotation, preserving rider evidence."""
    return {
        str(key): copy.deepcopy(value)
        for key, value in row.items()
        if not str(key).startswith("Threshold")
        and str(key) not in {
            "PriceThreshold",
            "PriceThresholdRationale",
            "PriceThresholdSource",
            "ExperimentalUnboundedEconomics",
        }
    }


def source_row_hash(row: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json(_source_base_row(row)).encode("utf-8"))


def _source_prompt_pair(
    row: Mapping[str, Any], *, coldstart_rides: int
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    rides_value = row.get("ColdstartRides")
    if not isinstance(rides_value, list) or len(rides_value) < int(coldstart_rides):
        raise ThresholdBatchError(
            f"source profile has fewer than {int(coldstart_rides)} ColdstartRides"
        )
    profile = {
        str(key): copy.deepcopy(value)
        for key, value in row.items()
        if key not in SOURCE_EXCLUDED_PROFILE_KEYS
        and not str(key).startswith("Threshold")
    }
    rides = [dict(item) for item in rides_value[: int(coldstart_rides)]]
    return profile, rides


def prompt_identity_hash(row: Mapping[str, Any], *, coldstart_rides: int) -> str:
    profile, rides = _source_prompt_pair(row, coldstart_rides=coldstart_rides)
    protected = {"Gender", "MaritalStatus", "Race", "Ethnicity"}
    visible = {key: value for key, value in profile.items() if key not in protected}
    return sha256_bytes(canonical_json({
        "profile": visible,
        "coldstart_rides": rides,
    }).encode("utf-8"))


def load_source_rows(
    source_cache: str | Path,
    *,
    market_name: str,
    expected_rows: int,
    coldstart_rides: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    source = Path(source_cache).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    rows: List[Dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ThresholdBatchError(
                    f"invalid source JSON at line {line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ThresholdBatchError(
                    f"source line {line_number} is not a JSON object"
                )
            declared_market = row.get("ProfileMarket", row.get("City", row.get("Market")))
            if (
                declared_market
                and _normalized_market(declared_market) != _normalized_market(market_name)
            ):
                raise ThresholdBatchError(
                    f"source line {line_number} market {declared_market!r} does not "
                    f"match {market_name!r}"
                )
            _source_prompt_pair(row, coldstart_rides=coldstart_rides)
            rows.append(row)
    if len(rows) != int(expected_rows):
        raise ThresholdBatchError(
            f"expected exactly {int(expected_rows)} source rows, found {len(rows)}"
        )
    source_hashes = [source_row_hash(row) for row in rows]
    if len(set(source_hashes)) != len(source_hashes):
        raise ThresholdBatchError("source rider/cold-start rows are not unique")
    prompt_hashes = [
        prompt_identity_hash(row, coldstart_rides=coldstart_rides) for row in rows
    ]
    if len(set(prompt_hashes)) != len(prompt_hashes):
        raise ThresholdBatchError(
            "source rows are unique, but the evidence visible to the threshold "
            "prompt contains duplicates"
        )
    provenance = {
        "path": str(source),
        "bytes": int(source.stat().st_size),
        "sha256": sha256_file(source),
        "rows": int(len(rows)),
        "unique_source_rows": int(len(set(source_hashes))),
        "unique_prompt_identities": int(len(set(prompt_hashes))),
        "source_row_order_sha256": ordered_digest(source_hashes),
        "prompt_identity_order_sha256": ordered_digest(prompt_hashes),
    }
    return rows, provenance


def _build_prompt_core(
    *,
    city: str,
    calibration_path: str | Path,
    model: str,
    seed: int,
    coldstart_rides: int,
    include_rationales: bool,
) -> Any:
    """Build the same calibrated prompt owner used by synchronous Core runs."""
    from Core import Core
    from calibration_utils import load_calibration_json
    from market_dynamics import MarketDynamicsConfig

    city_key = str(city).lower()
    if city_key not in CITY_NAMES:
        raise ThresholdBatchError(f"unsupported city {city!r}")
    calibration_source = Path(calibration_path).expanduser().resolve()
    calibration = load_calibration_json(
        str(calibration_source), expected_city=CITY_NAMES[city_key]
    )
    calibration["_source_dir"] = str(calibration_source.parent)
    with contextlib.redirect_stdout(io.StringIO()):
        core = Core(
            market_name=CITY_NAMES[city_key],
            seed=int(seed),
            choice_mode="cognitive",
            model_name=str(model),
            openai_api_key="",
            firm1_mode="static",
            firm2_mode="static",
            total_customers_pool=1,
            experimental_unbounded_economics=True,
            gpt_threshold_include_rationales=bool(include_rationales),
            gpt_threshold_coldstart_rides=int(coldstart_rides),
        )
        core.apply_calibration(calibration)
        core.configure_v26(
            market_dynamics=MarketDynamicsConfig(enabled=True),
            steps_per_day=6,
            gpt_threshold_mode="llm_numeric",
        )
    return core


def _logical_prompt_chars(payload: Mapping[str, Any]) -> int:
    return int(
        len(str(payload.get("instructions", "")))
        + len(str(payload.get("input", "")))
        + len(canonical_json((payload.get("text") or {}).get("format", {})))
    )


def _request_record(
    *,
    core: Any,
    city: str,
    rows: Sequence[Mapping[str, Any]],
    global_indices: Sequence[int],
    coldstart_rides: int,
    attempt: int,
    chars_per_token: float,
) -> Tuple[bytes, Dict[str, Any]]:
    pairs = [
        _source_prompt_pair(rows[index], coldstart_rides=coldstart_rides)
        for index in global_indices
    ]
    payload, _ = core._build_gpt_numeric_threshold_request_payload(pairs)
    prompt = json.loads(str(payload["input"]))
    protocol = str((prompt.get("evidence_protocol") or {}).get("prompt_protocol", ""))
    constraints = dict(prompt.get("output_constraints") or {})
    if constraints.get("economic_bounds") != "none":
        raise ThresholdBatchError("Core did not build the uncapped v28 prompt contract")
    if "price_threshold_range_usd" in constraints:
        raise ThresholdBatchError("uncapped v28 request unexpectedly contains a threshold range")
    if protocol != NATIVE_PROMPT_PROTOCOL:
        raise ThresholdBatchError(
            f"uncapped prompt protocol {protocol!r} != {NATIVE_PROMPT_PROTOCOL!r}"
        )
    mapping = []
    for local_index, global_index in enumerate(global_indices):
        mapping.append({
            "local_index": int(local_index),
            "global_index": int(global_index),
            "source_row_sha256": source_row_hash(rows[global_index]),
            "prompt_identity_sha256": prompt_identity_hash(
                rows[global_index], coldstart_rides=coldstart_rides
            ),
        })
    identity_digest = ordered_digest(
        f"{entry['global_index']}:{entry['source_row_sha256']}" for entry in mapping
    )
    first = min(int(index) for index in global_indices)
    last = max(int(index) for index in global_indices)
    custom_id = (
        f"v28thr-{city}-a{int(attempt)}-{first:05d}-{last:05d}-"
        f"{identity_digest[:16]}"
    )
    record = {
        "custom_id": custom_id,
        "method": "POST",
        "url": RESPONSES_ENDPOINT,
        "body": payload,
    }
    encoded = (canonical_json(record) + "\n").encode("utf-8")
    logical_chars = _logical_prompt_chars(payload)
    estimated_prompt_tokens = int(math.ceil(logical_chars / float(chars_per_token)))
    metadata = {
        "custom_id": custom_id,
        "attempt": int(attempt),
        "global_indices": [int(index) for index in global_indices],
        "profiles": mapping,
        "profile_count": int(len(mapping)),
        "model": str(payload["model"]),
        "prompt_protocol": protocol,
        "prompt_sha256": sha256_bytes(str(payload["input"]).encode("utf-8")),
        "request_body_sha256": sha256_bytes(canonical_json(payload).encode("utf-8")),
        "request_line_sha256": sha256_bytes(encoded),
        "request_bytes": int(len(encoded)),
        "logical_prompt_chars": int(logical_chars),
        "estimated_prompt_tokens": int(estimated_prompt_tokens),
        "max_output_tokens": int(payload.get("max_output_tokens", 0)),
    }
    return encoded, metadata


def _write_request_shards(
    *,
    work_dir: Path,
    records: Sequence[Tuple[bytes, Dict[str, Any]]],
    attempt: int,
    max_shard_bytes: int,
    max_queued_tokens: int,
    write: bool,
) -> List[Dict[str, Any]]:
    if not (1 <= int(max_shard_bytes) < 200_000_000):
        raise ThresholdBatchError("max_shard_bytes must be in [1, 200,000,000)")
    if int(max_queued_tokens) <= 0:
        raise ThresholdBatchError("max_queued_tokens must be positive")
    shards: List[Dict[str, Any]] = []
    current_lines: List[bytes] = []
    current_map: List[Dict[str, Any]] = []
    current_bytes = 0
    current_tokens = 0

    def flush() -> None:
        nonlocal current_lines, current_map, current_bytes, current_tokens
        if not current_lines:
            return
        payload = b"".join(current_lines)
        digest = sha256_bytes(payload)
        sequence = len(shards)
        filename = f"requests_a{int(attempt)}_{sequence:03d}_{digest[:16]}.jsonl"
        map_filename = f"requests_a{int(attempt)}_{sequence:03d}_{digest[:16]}.map.json"
        request_path = work_dir / "requests" / filename
        mapping_payload = {
            "schema_version": SCHEMA_VERSION,
            "request_sha256": digest,
            "requests": current_map,
        }
        mapping_bytes = (json.dumps(
            mapping_payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n").encode("utf-8")
        map_path = work_dir / "requests" / map_filename
        if write:
            if request_path.exists() and sha256_file(request_path) != digest:
                raise ThresholdBatchError(f"content-address collision at {request_path}")
            if not request_path.exists():
                _atomic_bytes(request_path, payload)
            map_digest = sha256_bytes(mapping_bytes)
            if map_path.exists() and sha256_file(map_path) != map_digest:
                raise ThresholdBatchError(f"mapping collision at {map_path}")
            if not map_path.exists():
                _atomic_bytes(map_path, mapping_bytes)
        else:
            map_digest = sha256_bytes(mapping_bytes)
        shards.append({
            "shard_id": f"a{int(attempt)}-{sequence:03d}-{digest[:16]}",
            "attempt": int(attempt),
            "request_path": str(request_path),
            "request_sha256": digest,
            "request_bytes": int(len(payload)),
            "mapping_path": str(map_path),
            "mapping_sha256": map_digest,
            "request_count": int(len(current_lines)),
            "profile_count": int(sum(item["profile_count"] for item in current_map)),
            "estimated_prompt_tokens": int(current_tokens),
            "custom_ids": [item["custom_id"] for item in current_map],
            "state": "prepared",
            "submission_generation": 0,
            "submission_history": [],
            "input_file_id": None,
            "batch_id": None,
            "batch_status": None,
            "output_file_id": None,
            "error_file_id": None,
            "output_path": None,
            "error_path": None,
        })
        current_lines = []
        current_map = []
        current_bytes = 0
        current_tokens = 0

    for encoded, metadata in records:
        request_tokens = int(metadata["estimated_prompt_tokens"])
        if len(encoded) > int(max_shard_bytes):
            raise ThresholdBatchError(
                f"one request ({len(encoded)} bytes) exceeds max_shard_bytes"
            )
        if request_tokens > int(max_queued_tokens):
            raise ThresholdBatchError(
                f"one request ({request_tokens} estimated tokens) exceeds max_queued_tokens"
            )
        if current_lines and (
            current_bytes + len(encoded) > int(max_shard_bytes)
            or current_tokens + request_tokens > int(max_queued_tokens)
        ):
            flush()
        current_lines.append(encoded)
        current_map.append(metadata)
        current_bytes += len(encoded)
        current_tokens += request_tokens
    flush()
    return shards


def _estimate_summary(
    records: Sequence[Tuple[bytes, Mapping[str, Any]]],
    shards: Sequence[Mapping[str, Any]],
    *,
    profile_count: int,
    output_tokens_per_request: int,
    standard_input_rate: float,
    standard_output_rate: float,
    batch_input_rate: float,
    batch_output_rate: float,
) -> Dict[str, Any]:
    request_count = int(len(records))
    prompt_tokens = int(sum(item[1]["estimated_prompt_tokens"] for item in records))
    output_tokens = int(request_count * int(output_tokens_per_request))
    output_token_ceiling = int(sum(
        max(0, int(item[1].get("max_output_tokens", 0))) for item in records
    ))

    def cost(input_rate: float, output_rate: float, billed_output_tokens: int) -> float:
        return float(
            prompt_tokens * float(input_rate) / 1_000_000.0
            + int(billed_output_tokens) * float(output_rate) / 1_000_000.0
        )

    return {
        "profiles": int(profile_count),
        "requests": request_count,
        "shards": int(len(shards)),
        "request_bytes": int(sum(len(item[0]) for item in records)),
        "logical_prompt_chars": int(sum(item[1]["logical_prompt_chars"] for item in records)),
        "estimated_prompt_tokens": prompt_tokens,
        "estimated_output_tokens": output_tokens,
        "configured_output_token_ceiling": output_token_ceiling,
        "estimated_standard_cost_usd": cost(
            standard_input_rate, standard_output_rate, output_tokens
        ),
        "estimated_batch_cost_usd": cost(
            batch_input_rate, batch_output_rate, output_tokens
        ),
        "standard_cost_at_output_ceiling_usd": cost(
            standard_input_rate, standard_output_rate, output_token_ceiling
        ),
        "batch_cost_at_output_ceiling_usd": cost(
            batch_input_rate, batch_output_rate, output_token_ceiling
        ),
        "pricing_assumption": {
            "standard_input_usd_per_million": float(standard_input_rate),
            "standard_output_usd_per_million": float(standard_output_rate),
            "batch_input_usd_per_million": float(batch_input_rate),
            "batch_output_usd_per_million": float(batch_output_rate),
            "output_tokens_per_request": int(output_tokens_per_request),
            "note": (
                "input counts are conservative character-based estimates; expected "
                "output cost and the configured max_output_tokens ceiling are both "
                "reported until actual API usage is collected"
            ),
        },
    }


def prepare(
    *,
    city: str,
    source_cache: str | Path,
    calibration_path: str | Path,
    work_dir: str | Path,
    expected_rows: int = 20_000,
    batch_size: int = DEFAULT_BATCH_SIZE,
    model: str = DEFAULT_MODEL,
    seed: int = 4500,
    coldstart_rides: int = 5,
    include_rationales: bool = False,
    max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
    max_queued_tokens: int = DEFAULT_MAX_QUEUED_TOKENS,
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
    output_tokens_per_request: int = DEFAULT_OUTPUT_TOKENS_PER_REQUEST,
    standard_input_rate: float = STANDARD_INPUT_USD_PER_MTOK,
    standard_output_rate: float = STANDARD_OUTPUT_USD_PER_MTOK,
    batch_input_rate: float = BATCH_INPUT_USD_PER_MTOK,
    batch_output_rate: float = BATCH_OUTPUT_USD_PER_MTOK,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Prepare deterministic request shards; never contacts OpenAI."""
    city_key = str(city).lower()
    if city_key not in CITY_NAMES:
        raise ThresholdBatchError(f"unsupported city {city!r}")
    if int(batch_size) <= 0:
        raise ThresholdBatchError("batch_size must be positive")
    if float(chars_per_token) <= 0.0:
        raise ThresholdBatchError("chars_per_token must be positive")
    destination = Path(work_dir).expanduser().resolve()
    calibration = Path(calibration_path).expanduser().resolve()
    if not calibration.is_file():
        raise FileNotFoundError(calibration)
    rows, source_provenance = load_source_rows(
        source_cache,
        market_name=CITY_NAMES[city_key],
        expected_rows=int(expected_rows),
        coldstart_rides=int(coldstart_rides),
    )
    core = _build_prompt_core(
        city=city_key,
        calibration_path=calibration,
        model=str(model),
        seed=int(seed),
        coldstart_rides=int(coldstart_rides),
        include_rationales=bool(include_rationales),
    )
    records: List[Tuple[bytes, Dict[str, Any]]] = []
    for start in range(0, len(rows), int(batch_size)):
        indices = list(range(start, min(len(rows), start + int(batch_size))))
        records.append(_request_record(
            core=core,
            city=city_key,
            rows=rows,
            global_indices=indices,
            coldstart_rides=int(coldstart_rides),
            attempt=0,
            chars_per_token=float(chars_per_token),
        ))
    # First calculate content addresses without writing.  The manifest lock is
    # acquired before materializing them, so a conflicting prepare invocation
    # cannot leave unrelated orphan shards in an existing work directory.
    shards = _write_request_shards(
        work_dir=destination,
        records=records,
        attempt=0,
        max_shard_bytes=int(max_shard_bytes),
        max_queued_tokens=int(max_queued_tokens),
        write=False,
    )
    estimate = _estimate_summary(
        records,
        shards,
        profile_count=len(rows),
        output_tokens_per_request=int(output_tokens_per_request),
        standard_input_rate=float(standard_input_rate),
        standard_output_rate=float(standard_output_rate),
        batch_input_rate=float(batch_input_rate),
        batch_output_rate=float(batch_output_rate),
    )
    calibration_provenance = {
        "path": str(calibration),
        "bytes": int(calibration.stat().st_size),
        "sha256": sha256_file(calibration),
    }
    prompt_protocols = sorted({
        item[1]["prompt_protocol"] for item in records
    })
    if len(prompt_protocols) != 1:
        raise ThresholdBatchError(f"mixed prompt protocols: {prompt_protocols}")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": utc_now(),
        "updated_utc": utc_now(),
        "state": "prepared",
        "city": city_key,
        "market_name": CITY_NAMES[city_key],
        "endpoint": RESPONSES_ENDPOINT,
        "model": str(model),
        "prompt_protocol": prompt_protocols[0],
        "economic_contract": "v28_unbounded_finite_positive",
        "source": source_provenance,
        "calibration": calibration_provenance,
        "config": {
            "expected_rows": int(expected_rows),
            "batch_size": int(batch_size),
            "seed": int(seed),
            "coldstart_rides": int(coldstart_rides),
            "include_rationales": bool(include_rationales),
            "max_shard_bytes": int(max_shard_bytes),
            "max_queued_tokens": int(max_queued_tokens),
            "chars_per_token": float(chars_per_token),
        },
        "estimate": estimate,
        "shards": shards,
        "final_cache": None,
    }
    fingerprint_payload = {
        key: manifest[key]
        for key in (
            "schema_version", "city", "market_name", "endpoint", "model",
            "prompt_protocol", "economic_contract", "source", "calibration", "config",
        )
    }
    manifest["run_fingerprint"] = sha256_bytes(
        canonical_json(fingerprint_payload).encode("utf-8")
    )
    if dry_run:
        return {**estimate, "dry_run": True, "manifest_preview": manifest}

    with work_lock(destination):
        manifest_path = destination / "manifest.json"
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("run_fingerprint") != manifest["run_fingerprint"]:
                raise ThresholdBatchError(
                    "work directory already contains a different threshold run"
                )
            verify_manifest_files(existing)
            # Estimation logic and published prices can be refined without
            # changing a request.  Refresh only this non-operational section;
            # never replace shard/job state in a resumed run.
            if existing.get("estimate") != manifest["estimate"]:
                existing["estimate"] = manifest["estimate"]
                _save_manifest(manifest_path, existing)
            return existing
        written_shards = _write_request_shards(
            work_dir=destination,
            records=records,
            attempt=0,
            max_shard_bytes=int(max_shard_bytes),
            max_queued_tokens=int(max_queued_tokens),
            write=True,
        )
        if canonical_json(written_shards) != canonical_json(shards):
            raise ThresholdBatchError("request shard materialization was nondeterministic")
        manifest["shards"] = written_shards
        atomic_json(manifest_path, manifest)
    return manifest


def load_manifest(work_dir: str | Path) -> Tuple[Path, Dict[str, Any]]:
    directory = Path(work_dir).expanduser().resolve()
    path = directory / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ThresholdBatchError("unsupported threshold manifest schema")
    return path, manifest


def _verify_submission_history(shard: Mapping[str, Any]) -> None:
    history = shard.get("submission_history", [])
    if not isinstance(history, list):
        raise ThresholdBatchError(
            f"submission history is not a list for {shard.get('shard_id')}"
        )
    generations: List[int] = []
    for position, raw_entry in enumerate(history):
        if not isinstance(raw_entry, Mapping):
            raise ThresholdBatchError(
                f"invalid submission history entry {position} for {shard.get('shard_id')}"
            )
        entry = dict(raw_entry)
        claimed = str(entry.pop("entry_sha256", ""))
        actual = sha256_bytes(canonical_json(entry).encode("utf-8"))
        if not claimed or claimed != actual:
            raise ThresholdBatchError(
                f"submission history hash mismatch for {shard.get('shard_id')} "
                f"entry {position}"
            )
        generations.append(int(entry.get("submission_generation", -1)))
    if generations != sorted(set(generations)):
        raise ThresholdBatchError(
            f"submission history generations are not unique/increasing for "
            f"{shard.get('shard_id')}"
        )


def verify_manifest_files(manifest: Mapping[str, Any]) -> None:
    source = Path(str((manifest.get("source") or {}).get("path", "")))
    calibration = Path(str((manifest.get("calibration") or {}).get("path", "")))
    for label, path, expected in (
        ("source", source, (manifest.get("source") or {}).get("sha256")),
        ("calibration", calibration, (manifest.get("calibration") or {}).get("sha256")),
    ):
        if not path.is_file() or sha256_file(path) != expected:
            raise ThresholdBatchError(f"{label} provenance changed: {path}")
    for shard in manifest.get("shards", []):
        _verify_submission_history(shard)
        for label, path_key, hash_key in (
            ("request", "request_path", "request_sha256"),
            ("mapping", "mapping_path", "mapping_sha256"),
        ):
            path = Path(str(shard[path_key]))
            if not path.is_file() or sha256_file(path) != shard[hash_key]:
                raise ThresholdBatchError(
                    f"{label} shard provenance changed: {path}"
                )
        for path_key, hash_key in (
            ("output_path", "output_sha256"),
            ("error_path", "error_sha256"),
        ):
            if shard.get(path_key):
                path = Path(str(shard[path_key]))
                if not path.is_file() or sha256_file(path) != shard.get(hash_key):
                    raise ThresholdBatchError(f"downloaded artifact changed: {path}")


def _require_api_key() -> str:
    key = str(os.environ.get("OPENAI_API_KEY", ""))
    if not key:
        raise ThresholdBatchError("OPENAI_API_KEY is required for this network command")
    return key


def _api_json(
    method: str,
    path: str,
    *,
    api_key: str,
    body: Optional[Mapping[str, Any]] = None,
    extra_headers: Optional[Mapping[str, str]] = None,
    timeout: float = 120.0,
) -> Dict[str, Any]:
    data = None if body is None else canonical_json(body).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    headers.update(dict(extra_headers or {}))
    request = urllib.request.Request(
        API_BASE + str(path), data=data, headers=headers, method=str(method)
    )
    try:
        with urllib.request.urlopen(request, timeout=float(timeout)) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise ThresholdBatchError(f"OpenAI HTTP {exc.code}: {detail}") from exc
    if not isinstance(payload, dict):
        raise ThresholdBatchError("OpenAI API returned a non-object response")
    return payload


def _api_bytes(
    path: str,
    *,
    api_key: str,
    timeout: float = 120.0,
) -> bytes:
    request = urllib.request.Request(
        API_BASE + str(path),
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=float(timeout)) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise ThresholdBatchError(f"OpenAI HTTP {exc.code}: {detail}") from exc


def _upload_batch_file(
    path: Path,
    *,
    api_key: str,
    idempotency_key: str,
    timeout: float = 300.0,
) -> Dict[str, Any]:
    # The multipart boundary is derived from the stable idempotency key so a
    # retried upload has byte-identical content.
    boundary = f"codex-v28-{sha256_bytes(str(idempotency_key).encode())[:32]}"
    header = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="purpose"\r\n\r\n'
        "batch\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
        "Content-Type: application/jsonl\r\n\r\n"
    ).encode("utf-8")
    trailer = f"\r\n--{boundary}--\r\n".encode("utf-8")
    data = header + path.read_bytes() + trailer
    request = urllib.request.Request(
        API_BASE + "/files",
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Idempotency-Key": str(idempotency_key),
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=float(timeout)) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise ThresholdBatchError(f"OpenAI HTTP {exc.code}: {detail}") from exc
    if not isinstance(payload, dict) or not payload.get("id"):
        raise ThresholdBatchError("file upload response did not contain an id")
    return payload


def _save_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    manifest["updated_utc"] = utc_now()
    # A defensive invariant: credentials must never become persisted metadata.
    serialized = canonical_json(manifest)
    if "OPENAI_API_KEY" in serialized or re.search(r"sk-[A-Za-z0-9_-]{12,}", serialized):
        raise ThresholdBatchError("refusing to persist a possible API credential")
    atomic_json(path, manifest)


def _active_estimated_tokens(manifest: Mapping[str, Any]) -> int:
    return int(sum(
        int(shard.get("estimated_prompt_tokens", 0))
        for shard in manifest.get("shards", [])
        if str(shard.get("batch_status") or shard.get("state"))
        in ACTIVE_BATCH_STATUSES
    ))


def queue_ledger(
    work_dirs: Sequence[str | Path],
    *,
    global_max_active_estimated_tokens: int = DEFAULT_GLOBAL_MAX_QUEUED_TOKENS,
) -> Dict[str, Any]:
    """Read an advisory cross-manifest queue ledger without network I/O.

    Statuses must be refreshed before using this as a submission guard.  Batch
    quota is organization/model-wide, so per-city accounting alone is unsafe.
    """
    ceiling = int(global_max_active_estimated_tokens)
    if ceiling <= 0:
        raise ThresholdBatchError("global queue ceiling must be positive")
    seen: set[Path] = set()
    manifests: List[Tuple[Path, Dict[str, Any]]] = []
    for value in work_dirs:
        directory = Path(value).expanduser().resolve()
        if directory in seen:
            continue
        seen.add(directory)
        _, manifest = load_manifest(directory)
        verify_manifest_files(manifest)
        manifests.append((directory, manifest))
    if not manifests:
        raise ThresholdBatchError("queue ledger requires at least one work directory")
    active_jobs: List[Dict[str, Any]] = []
    prepared_jobs: List[Dict[str, Any]] = []
    for directory, manifest in manifests:
        for shard in manifest.get("shards", []):
            row = {
                "work_dir": str(directory),
                "city": str(manifest.get("city", "")),
                "shard_id": str(shard.get("shard_id", "")),
                "estimated_prompt_tokens": int(
                    shard.get("estimated_prompt_tokens", 0)
                ),
                "batch_status": shard.get("batch_status"),
                "submission_generation": int(
                    shard.get("submission_generation", 0)
                ),
            }
            status_value = str(shard.get("batch_status") or shard.get("state"))
            if status_value in ACTIVE_BATCH_STATUSES:
                active_jobs.append(row)
            elif not shard.get("batch_id") and str(shard.get("state", "prepared")) in {
                "prepared", "submitting"
            }:
                prepared_jobs.append(row)
    active = int(sum(item["estimated_prompt_tokens"] for item in active_jobs))
    available = max(0, ceiling - active)
    return {
        "global_max_active_estimated_tokens": ceiling,
        "active_estimated_tokens": active,
        "available_estimated_tokens": available,
        "active_jobs": active_jobs,
        "prepared_jobs": prepared_jobs,
        "prepared_jobs_fitting_now": [
            item for item in prepared_jobs
            if int(item["estimated_prompt_tokens"]) <= available
        ],
        "manifest_count": len(manifests),
        "advisory": (
            "Refresh every submitted manifest with status immediately before "
            "using this ledger; remote quota can include jobs outside these manifests."
        ),
    }


def submit(
    work_dir: str | Path,
    *,
    submit_all: bool = False,
    max_active_estimated_tokens: Optional[int] = None,
    peer_work_dirs: Sequence[str | Path] = (),
    global_max_active_estimated_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Upload and submit prepared shards, respecting a queued-token ceiling."""
    api_key = _require_api_key()
    with work_lock(work_dir):
        manifest_path, manifest = load_manifest(work_dir)
        verify_manifest_files(manifest)
        ceiling = int(
            max_active_estimated_tokens
            if max_active_estimated_tokens is not None
            else (manifest.get("config") or {}).get(
                "max_queued_tokens", DEFAULT_MAX_QUEUED_TOKENS
            )
        )
        active = _active_estimated_tokens(manifest)
        global_active = active
        global_ceiling = (
            None
            if global_max_active_estimated_tokens is None
            else int(global_max_active_estimated_tokens)
        )
        if global_ceiling is not None:
            if global_ceiling <= 0:
                raise ThresholdBatchError("global queue ceiling must be positive")
            current_dir = Path(work_dir).expanduser().resolve()
            seen_peers: set[Path] = {current_dir}
            for peer_value in peer_work_dirs:
                peer_dir = Path(peer_value).expanduser().resolve()
                if peer_dir in seen_peers:
                    continue
                seen_peers.add(peer_dir)
                _, peer = load_manifest(peer_dir)
                verify_manifest_files(peer)
                global_active += _active_estimated_tokens(peer)
        submitted = 0
        for shard in manifest.get("shards", []):
            if shard.get("batch_id"):
                continue
            state = str(shard.get("state", "prepared"))
            if state not in {"prepared", "submitting"}:
                continue
            tokens = int(shard.get("estimated_prompt_tokens", 0))
            if active + tokens > ceiling:
                if submitted == 0:
                    raise ThresholdBatchError(
                        f"queued-token ceiling {ceiling} leaves no room for shard "
                        f"{shard['shard_id']} ({tokens} estimated tokens)"
                    )
                break
            if global_ceiling is not None and global_active + tokens > global_ceiling:
                if submitted == 0:
                    raise ThresholdBatchError(
                        f"global queued-token ceiling {global_ceiling} leaves no room "
                        f"for shard {shard['shard_id']} ({tokens} estimated tokens); "
                        f"active across supplied manifests={global_active}"
                    )
                break
            shard["state"] = "submitting"
            _save_manifest(manifest_path, manifest)
            request_path = Path(str(shard["request_path"]))
            if not shard.get("input_file_id"):
                uploaded = _upload_batch_file(
                    request_path,
                    api_key=api_key,
                    idempotency_key=f"v28thr-file-{shard['request_sha256']}",
                )
                shard["input_file_id"] = str(uploaded["id"])
                _save_manifest(manifest_path, manifest)
            batch = _api_json(
                "POST",
                "/batches",
                api_key=api_key,
                body={
                    "input_file_id": shard["input_file_id"],
                    "endpoint": RESPONSES_ENDPOINT,
                    "completion_window": "24h",
                    "metadata": {
                        "pipeline": "v28_threshold_cache",
                        "shard_id": str(shard["shard_id"]),
                        "run": str(manifest["run_fingerprint"])[:32],
                        "submission_generation": str(
                            int(shard.get("submission_generation", 0))
                        ),
                    },
                },
                extra_headers={
                    "Idempotency-Key": (
                        f"v28thr-batch-{shard['request_sha256']}-g"
                        f"{int(shard.get('submission_generation', 0))}"
                    )
                },
            )
            if not batch.get("id"):
                raise ThresholdBatchError("batch submission response did not contain an id")
            shard["batch_id"] = str(batch["id"])
            shard["batch"] = _batch_snapshot(batch)
            shard["batch_status"] = str(batch.get("status", "validating"))
            shard["state"] = "submitted"
            shard["submitted_utc"] = utc_now()
            if shard["batch_status"] in ACTIVE_BATCH_STATUSES:
                active += tokens
                global_active += tokens
            submitted += 1
            _save_manifest(manifest_path, manifest)
            if not submit_all or shard["batch_status"] not in ACTIVE_BATCH_STATUSES:
                break
        return {
            "submitted_shards": int(submitted),
            "active_estimated_tokens": int(active),
            "global_active_estimated_tokens": int(global_active),
            "global_ceiling": global_ceiling,
            "ceiling": int(ceiling),
        }


def _redact_batch_error_text(value: Any, *, limit: int = 2_000) -> str:
    """Return bounded diagnostic text with credential-like material removed."""
    text = str(value)
    text = text.replace("OPENAI_API_KEY", "[REDACTED_ENV_NAME]")
    text = re.sub(
        r"(?i)\bBearer\s+[^\s,;\"']+",
        "Bearer [REDACTED]",
        text,
    )
    text = re.sub(
        r"\bsk-[A-Za-z0-9_-]{12,}",
        "[REDACTED_OPENAI_KEY]",
        text,
    )
    text = re.sub(
        r"(?i)\bapi[_ -]?key\s*[:=]\s*[^\s,;\"']+",
        "api_key=[REDACTED]",
        text,
    )
    return text[: int(limit)]


def _sanitize_batch_errors(raw_errors: Any) -> Dict[str, Any]:
    """Keep only documented Batch validation-error fields.

    The API response is treated as untrusted diagnostic data: arbitrary nested
    fields are dropped, strings are bounded, and credential-like material is
    redacted before the manifest's independent secret scan runs.
    """
    if isinstance(raw_errors, Mapping):
        raw_data = raw_errors.get("data", [])
        object_name = _redact_batch_error_text(
            raw_errors.get("object", "list"), limit=64
        )
    elif isinstance(raw_errors, list):
        raw_data = raw_errors
        object_name = "list"
    else:
        raw_data = []
        object_name = "list"
    if not isinstance(raw_data, list):
        raw_data = []
    sanitized: List[Dict[str, Any]] = []
    for item in raw_data[:100]:
        if not isinstance(item, Mapping):
            sanitized.append({
                "message": _redact_batch_error_text(item),
            })
            continue
        entry: Dict[str, Any] = {}
        for key in ("code", "message", "param"):
            value = item.get(key)
            if value is not None:
                entry[key] = _redact_batch_error_text(value)
        line = item.get("line")
        if line is not None:
            try:
                entry["line"] = int(line)
            except (TypeError, ValueError):
                entry["line"] = _redact_batch_error_text(line, limit=128)
        sanitized.append(entry)
    return {
        "object": object_name,
        "data": sanitized,
        "source_error_count": int(len(raw_data)),
        "truncated": bool(len(raw_data) > len(sanitized)),
    }


def _batch_snapshot(batch: Mapping[str, Any]) -> Dict[str, Any]:
    keep = (
        "id", "status", "input_file_id", "output_file_id", "error_file_id",
        "created_at", "in_progress_at", "finalizing_at", "completed_at",
        "failed_at", "expired_at", "cancelled_at", "expires_at",
        "request_counts", "usage",
    )
    snapshot = {
        key: copy.deepcopy(batch.get(key)) for key in keep if key in batch
    }
    if batch.get("errors") is not None:
        snapshot["errors"] = _sanitize_batch_errors(batch.get("errors"))
    return snapshot


def _zero_request_preprocessing_failure(
    shard: Mapping[str, Any],
) -> Tuple[bool, str]:
    """Return whether a failed job is provably safe to submit again."""
    if not shard.get("batch_id"):
        return False, "no batch_id"
    if str(shard.get("batch_status")) != "failed":
        return False, f"batch_status={shard.get('batch_status')!r}, not failed"
    if any(shard.get(key) for key in (
        "output_file_id", "error_file_id", "output_path", "error_path"
    )):
        return False, "batch has an output/error artifact"
    snapshot = shard.get("batch")
    if not isinstance(snapshot, Mapping):
        return False, "no persisted batch snapshot"
    if str(snapshot.get("status")) != "failed":
        return False, "persisted batch snapshot is not failed"
    counts = snapshot.get("request_counts")
    if not isinstance(counts, Mapping):
        return False, "no authoritative request_counts"
    try:
        total = int(counts.get("total"))
        completed = int(counts.get("completed"))
        failed = int(counts.get("failed"))
    except (TypeError, ValueError):
        return False, "invalid request_counts"
    if (total, completed, failed) != (0, 0, 0):
        return False, (
            "job may have processed requests: "
            f"total={total}, completed={completed}, failed={failed}"
        )
    return True, "failed before any request was processed"


def requeue_failed_validation(
    work_dir: str | Path,
    *,
    shard_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Explicitly requeue only failed jobs proven to have processed zero lines.

    The prior remote job is archived as a content-hashed history entry.  The
    request file and uploaded input file remain unchanged, while a monotonically
    increasing submission generation forces a fresh Batch idempotency key.
    """
    directory = Path(work_dir).expanduser().resolve()
    requested = None if shard_ids is None else {str(value) for value in shard_ids}
    if requested is not None and not requested:
        raise ThresholdBatchError("shard_ids must not be empty when supplied")
    with work_lock(directory):
        manifest_path, manifest = load_manifest(directory)
        verify_manifest_files(manifest)
        known = {str(shard.get("shard_id")) for shard in manifest.get("shards", [])}
        if requested is not None and not requested.issubset(known):
            raise ThresholdBatchError(
                f"unknown shard ids: {sorted(requested - known)}"
            )
        requeued: List[str] = []
        skipped: Dict[str, str] = {}
        transition_time = utc_now()
        for shard in manifest.get("shards", []):
            shard_id = str(shard.get("shard_id", ""))
            if requested is not None and shard_id not in requested:
                continue
            eligible, reason = _zero_request_preprocessing_failure(shard)
            if not eligible:
                skipped[shard_id] = reason
                continue
            generation = int(shard.get("submission_generation", 0))
            history_entry: Dict[str, Any] = {
                "submission_generation": generation,
                "request_sha256": str(shard.get("request_sha256", "")),
                "input_file_id": str(shard.get("input_file_id", "")),
                "batch_id": str(shard["batch_id"]),
                "batch_status": str(shard.get("batch_status", "")),
                "batch": copy.deepcopy(shard.get("batch")),
                "submitted_utc": shard.get("submitted_utc"),
                "archived_utc": transition_time,
                "transition": "requeue_zero_request_preprocessing_failure",
            }
            history_entry["entry_sha256"] = sha256_bytes(
                canonical_json(history_entry).encode("utf-8")
            )
            history = shard.setdefault("submission_history", [])
            if not isinstance(history, list):
                raise ThresholdBatchError(
                    f"submission history is not a list for {shard_id}"
                )
            history.append(history_entry)
            shard["submission_generation"] = generation + 1
            shard["state"] = "prepared"
            shard["requeued_utc"] = transition_time
            shard["requeue_reason"] = reason
            for key in (
                "batch_id", "batch_status", "batch", "output_file_id",
                "error_file_id", "output_path", "output_sha256", "error_path",
                "error_sha256", "submitted_utc", "collected_utc",
            ):
                if key in {"batch_id", "batch_status", "batch", "output_file_id", "error_file_id"}:
                    shard[key] = None
                else:
                    shard.pop(key, None)
            requeued.append(shard_id)
        if requested is not None and skipped:
            # The transition is all-or-nothing for an explicit selection.
            raise ThresholdBatchError(
                "selected shards are not safe to requeue: "
                + canonical_json(skipped)
            )
        if requeued:
            manifest["state"] = "failed_validation_requeued"
            _save_manifest(manifest_path, manifest)
        return {
            "requeued_shards": requeued,
            "requeued_count": len(requeued),
            "skipped": skipped,
            "network_calls": 0,
        }


def status(work_dir: str | Path) -> Dict[str, Any]:
    """Refresh batch state without downloading result files."""
    api_key = _require_api_key()
    with work_lock(work_dir):
        manifest_path, manifest = load_manifest(work_dir)
        verify_manifest_files(manifest)
        refreshed = 0
        counts: Dict[str, int] = {}
        for shard in manifest.get("shards", []):
            if not shard.get("batch_id"):
                continue
            batch = _api_json(
                "GET", f"/batches/{shard['batch_id']}", api_key=api_key
            )
            snapshot = _batch_snapshot(batch)
            shard["batch"] = snapshot
            shard["batch_status"] = str(snapshot.get("status", "unknown"))
            shard["output_file_id"] = snapshot.get("output_file_id")
            shard["error_file_id"] = snapshot.get("error_file_id")
            refreshed += 1
            counts[shard["batch_status"]] = counts.get(shard["batch_status"], 0) + 1
        _save_manifest(manifest_path, manifest)
        return {"refreshed_shards": refreshed, "statuses": counts}


def collect(work_dir: str | Path) -> Dict[str, Any]:
    """Refresh terminal batches and atomically download output/error JSONL."""
    api_key = _require_api_key()
    directory = Path(work_dir).expanduser().resolve()
    with work_lock(directory):
        manifest_path, manifest = load_manifest(directory)
        verify_manifest_files(manifest)
        downloaded = 0
        waiting = 0
        for shard in manifest.get("shards", []):
            if not shard.get("batch_id"):
                continue
            batch = _api_json(
                "GET", f"/batches/{shard['batch_id']}", api_key=api_key
            )
            snapshot = _batch_snapshot(batch)
            shard["batch"] = snapshot
            shard["batch_status"] = str(snapshot.get("status", "unknown"))
            shard["output_file_id"] = snapshot.get("output_file_id")
            shard["error_file_id"] = snapshot.get("error_file_id")
            if shard["batch_status"] not in TERMINAL_BATCH_STATUSES:
                waiting += 1
                continue
            if shard.get("output_file_id") and not shard.get("output_path"):
                payload = _api_bytes(
                    f"/files/{shard['output_file_id']}/content", api_key=api_key
                )
                path = directory / "responses" / f"{shard['shard_id']}.output.jsonl"
                _atomic_bytes(path, payload)
                shard["output_path"] = str(path)
                shard["output_sha256"] = sha256_bytes(payload)
                downloaded += 1
            if shard.get("error_file_id") and not shard.get("error_path"):
                payload = _api_bytes(
                    f"/files/{shard['error_file_id']}/content", api_key=api_key
                )
                path = directory / "responses" / f"{shard['shard_id']}.error.jsonl"
                _atomic_bytes(path, payload)
                shard["error_path"] = str(path)
                shard["error_sha256"] = sha256_bytes(payload)
                downloaded += 1
            shard["state"] = "collected"
            shard["collected_utc"] = utc_now()
            _save_manifest(manifest_path, manifest)
        _save_manifest(manifest_path, manifest)
        return {"downloaded_files": downloaded, "waiting_shards": waiting}


def _load_mapping(shard: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    path = Path(str(shard["mapping_path"]))
    if sha256_file(path) != shard["mapping_sha256"]:
        raise ThresholdBatchError(f"mapping hash mismatch: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    requests = payload.get("requests")
    if not isinstance(requests, list):
        raise ThresholdBatchError(f"mapping has no requests list: {path}")
    out: Dict[str, Dict[str, Any]] = {}
    for item in requests:
        custom_id = str(item.get("custom_id", ""))
        if not custom_id or custom_id in out:
            raise ThresholdBatchError(f"invalid/duplicate custom_id in {path}")
        out[custom_id] = dict(item)
    return out


def _parse_output_line(
    line: Mapping[str, Any],
    *,
    request_meta: Mapping[str, Any],
    expected_model: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, int], List[str]]:
    from Core import Core

    errors: List[str] = []
    response = line.get("response")
    if not isinstance(response, dict):
        return [], {}, [f"{request_meta['custom_id']}: missing response object"]
    if int(response.get("status_code", 0)) != 200:
        return [], {}, [
            f"{request_meta['custom_id']}: HTTP {response.get('status_code')}"
        ]
    body = response.get("body")
    if not isinstance(body, dict):
        return [], {}, [f"{request_meta['custom_id']}: missing response body"]
    response_model = str(body.get("model", ""))
    if response_model != str(expected_model):
        return [], {}, [
            f"{request_meta['custom_id']}: model {response_model!r} != {expected_model!r}"
        ]
    try:
        parsed = Core._parse_response_json(Core._extract_response_text(body))
    except Exception as exc:
        return [], {}, [f"{request_meta['custom_id']}: invalid output JSON ({exc})"]
    rows = parsed.get("thresholds")
    if not isinstance(rows, list):
        return [], {}, [f"{request_meta['custom_id']}: no thresholds array"]
    profiles = list(request_meta.get("profiles") or [])
    by_local = {int(item["local_index"]): item for item in profiles}
    seen: set[int] = set()
    accepted: List[Dict[str, Any]] = []
    for item in rows:
        if not isinstance(item, dict):
            errors.append(f"{request_meta['custom_id']}: non-object threshold row")
            continue
        try:
            local_index = int(item.get("profile_index", -1))
            value = float(item.get("price_threshold"))
        except (TypeError, ValueError):
            errors.append(f"{request_meta['custom_id']}: malformed threshold row")
            continue
        if local_index not in by_local or local_index in seen:
            errors.append(
                f"{request_meta['custom_id']}: duplicate/out-of-range local index {local_index}"
            )
            continue
        if not math.isfinite(value) or value <= 0.0:
            errors.append(
                f"{request_meta['custom_id']}: threshold must be finite positive"
            )
            continue
        seen.add(local_index)
        mapped = by_local[local_index]
        accepted.append({
            "global_index": int(mapped["global_index"]),
            "local_index": local_index,
            "price_threshold": value,
            "rationale": str(item.get("rationale", "")),
            "custom_id": str(request_meta["custom_id"]),
            "prompt_sha256": str(request_meta["prompt_sha256"]),
            "request_body_sha256": str(request_meta["request_body_sha256"]),
            "source_row_sha256": str(mapped["source_row_sha256"]),
            "prompt_identity_sha256": str(mapped["prompt_identity_sha256"]),
            "prompt_batch_size": int(request_meta["profile_count"]),
            "response_model": response_model,
            "response_id": str(body.get("id", "")),
        })
    missing_local = sorted(set(by_local) - seen)
    if missing_local:
        errors.append(
            f"{request_meta['custom_id']}: missing local indices {missing_local}"
        )
    usage_raw = body.get("usage") or {}
    usage: Dict[str, int] = {}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        try:
            usage[key] = int(usage_raw.get(key, 0))
        except (TypeError, ValueError):
            usage[key] = 0
    return accepted, usage, errors


def collect_results(
    manifest: Mapping[str, Any],
) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, int], List[str]]:
    results: Dict[int, Dict[str, Any]] = {}
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    errors: List[str] = []
    seen_custom_ids: set[str] = set()
    for shard in manifest.get("shards", []):
        mapping = _load_mapping(shard)
        output_path_value = shard.get("output_path")
        if not output_path_value:
            continue
        output_path = Path(str(output_path_value))
        if sha256_file(output_path) != shard.get("output_sha256"):
            raise ThresholdBatchError(f"output hash mismatch: {output_path}")
        with output_path.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, start=1):
                if not raw.strip():
                    continue
                try:
                    line = json.loads(raw)
                except json.JSONDecodeError as exc:
                    errors.append(f"{output_path}:{line_number}: invalid JSON ({exc})")
                    continue
                custom_id = str(line.get("custom_id", ""))
                if custom_id not in mapping:
                    errors.append(f"{output_path}:{line_number}: unknown custom_id {custom_id!r}")
                    continue
                if custom_id in seen_custom_ids:
                    errors.append(f"duplicate output custom_id {custom_id}")
                    continue
                seen_custom_ids.add(custom_id)
                accepted, request_usage, request_errors = _parse_output_line(
                    line,
                    request_meta=mapping[custom_id],
                    expected_model=str(manifest["model"]),
                )
                errors.extend(request_errors)
                for key in usage:
                    usage[key] += int(request_usage.get(key, 0))
                for result in accepted:
                    index = int(result["global_index"])
                    if index in results:
                        previous = results[index]
                        if (
                            not math.isclose(
                                float(previous["price_threshold"]),
                                float(result["price_threshold"]),
                                rel_tol=0.0,
                                abs_tol=0.0,
                            )
                            or str(previous.get("rationale", ""))
                            != str(result.get("rationale", ""))
                        ):
                            errors.append(
                                f"conflicting valid results for global index {index}"
                            )
                        continue
                    results[index] = result
        missing_custom = sorted(set(mapping) - seen_custom_ids)
        for custom_id in missing_custom:
            errors.append(f"{shard['shard_id']}: missing output for {custom_id}")
    return results, usage, errors


def retry_missing(work_dir: str | Path) -> Dict[str, Any]:
    """Create deterministic request shards containing only unresolved rows."""
    directory = Path(work_dir).expanduser().resolve()
    with work_lock(directory):
        manifest_path, manifest = load_manifest(directory)
        verify_manifest_files(manifest)
        outstanding = [
            shard["shard_id"]
            for shard in manifest.get("shards", [])
            if shard.get("batch_id")
            and str(shard.get("batch_status")) not in TERMINAL_BATCH_STATUSES
        ]
        if outstanding:
            raise ThresholdBatchError(
                f"cannot retry while batches are nonterminal: {outstanding[:5]}"
            )
        unsubmitted = [
            shard["shard_id"]
            for shard in manifest.get("shards", [])
            if int(shard.get("attempt", 0)) == 0 and not shard.get("batch_id")
        ]
        if unsubmitted:
            raise ThresholdBatchError(
                f"cannot retry before every original shard is submitted and "
                f"collected: {unsubmitted[:5]}"
            )
        results, _, _ = collect_results(manifest)
        expected = int((manifest.get("config") or {})["expected_rows"])
        missing = sorted(set(range(expected)) - set(results))
        if not missing:
            return {"missing_profiles": 0, "new_shards": 0}
        already_pending: set[int] = set()
        for shard in manifest.get("shards", []):
            if int(shard.get("attempt", 0)) <= 0 or shard.get("batch_id"):
                continue
            for request in _load_mapping(shard).values():
                already_pending.update(int(i) for i in request.get("global_indices", []))
        if set(missing).issubset(already_pending):
            return {
                "missing_profiles": len(missing),
                "new_shards": 0,
                "note": "matching retry shards are already prepared",
            }
        if already_pending.intersection(missing):
            raise ThresholdBatchError(
                "partially overlapping pending retry shards require manual audit"
            )
        rows, provenance = load_source_rows(
            (manifest.get("source") or {})["path"],
            market_name=str(manifest["market_name"]),
            expected_rows=expected,
            coldstart_rides=int((manifest.get("config") or {})["coldstart_rides"]),
        )
        if provenance["sha256"] != (manifest.get("source") or {}).get("sha256"):
            raise ThresholdBatchError("source changed before retry")
        core = _build_prompt_core(
            city=str(manifest["city"]),
            calibration_path=(manifest.get("calibration") or {})["path"],
            model=str(manifest["model"]),
            seed=int((manifest.get("config") or {})["seed"]),
            coldstart_rides=int((manifest.get("config") or {})["coldstart_rides"]),
            include_rationales=bool((manifest.get("config") or {})["include_rationales"]),
        )
        attempt = 1 + max(int(shard.get("attempt", 0)) for shard in manifest["shards"])
        batch_size = int((manifest.get("config") or {})["batch_size"])
        records: List[Tuple[bytes, Dict[str, Any]]] = []
        for start in range(0, len(missing), batch_size):
            records.append(_request_record(
                core=core,
                city=str(manifest["city"]),
                rows=rows,
                global_indices=missing[start:start + batch_size],
                coldstart_rides=int((manifest.get("config") or {})["coldstart_rides"]),
                attempt=attempt,
                chars_per_token=float((manifest.get("config") or {})["chars_per_token"]),
            ))
        shards = _write_request_shards(
            work_dir=directory,
            records=records,
            attempt=attempt,
            max_shard_bytes=int((manifest.get("config") or {})["max_shard_bytes"]),
            max_queued_tokens=int((manifest.get("config") or {})["max_queued_tokens"]),
            write=True,
        )
        manifest["shards"].extend(shards)
        manifest["state"] = "retry_prepared"
        _save_manifest(manifest_path, manifest)
        return {"missing_profiles": len(missing), "new_shards": len(shards)}


def _quantiles(values: Sequence[float]) -> Dict[str, float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {}

    def at(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lo = int(math.floor(position))
        hi = int(math.ceil(position))
        if lo == hi:
            return ordered[lo]
        weight = position - lo
        return ordered[lo] * (1.0 - weight) + ordered[hi] * weight

    return {
        "min": ordered[0],
        "p01": at(0.01),
        "p05": at(0.05),
        "median": at(0.50),
        "p95": at(0.95),
        "p99": at(0.99),
        "max": ordered[-1],
    }


def validate(
    work_dir: str | Path,
    *,
    output_cache: str | Path,
    actual_input_rate: float = BATCH_INPUT_USD_PER_MTOK,
    actual_output_rate: float = BATCH_OUTPUT_USD_PER_MTOK,
) -> Dict[str, Any]:
    """Build the final native cache only when every source row has a valid result."""
    directory = Path(work_dir).expanduser().resolve()
    with work_lock(directory):
        manifest_path, manifest = load_manifest(directory)
        verify_manifest_files(manifest)
        expected = int((manifest.get("config") or {})["expected_rows"])
        rows, source_provenance = load_source_rows(
            (manifest.get("source") or {})["path"],
            market_name=str(manifest["market_name"]),
            expected_rows=expected,
            coldstart_rides=int((manifest.get("config") or {})["coldstart_rides"]),
        )
        if source_provenance["sha256"] != (manifest.get("source") or {}).get("sha256"):
            raise ThresholdBatchError("source changed before final validation")
        results, usage, parse_errors = collect_results(manifest)
        missing = sorted(set(range(expected)) - set(results))
        if missing:
            raise ThresholdBatchError(
                f"final cache blocked: {len(missing)} profiles lack valid API results; "
                f"first indices={missing[:20]}"
            )
        conflicts = [error for error in parse_errors if error.startswith("conflicting")]
        if conflicts:
            raise ThresholdBatchError("final cache blocked: " + "; ".join(conflicts[:10]))
        # Nonfatal parse errors (for example an invalid partial row that was
        # successfully re-requested) remain in the audit report, not the cache.
        from Core import (
            CROWDLLM_REFERENCE_FIRM1_MULTIPLIER,
            CROWDLLM_REFERENCE_FIRM2_MULTIPLIER,
            CROWDLLM_THRESHOLD_EVIDENCE_PROTOCOL,
        )

        output_rows: List[Dict[str, Any]] = []
        values: List[float] = []
        for index, source in enumerate(rows):
            result = results[index]
            base = _source_base_row(source)
            if source_row_hash(source) != result["source_row_sha256"]:
                raise ThresholdBatchError(f"source-row mapping mismatch at index {index}")
            if prompt_identity_hash(
                source,
                coldstart_rides=int((manifest.get("config") or {})["coldstart_rides"]),
            ) != result["prompt_identity_sha256"]:
                raise ThresholdBatchError(
                    f"prompt-visible evidence mapping mismatch at index {index}"
                )
            value = float(result["price_threshold"])
            if not math.isfinite(value) or value <= 0.0:
                raise ThresholdBatchError(f"invalid threshold at index {index}")
            base.update({
                "PriceThreshold": value,
                "PriceThresholdRationale": str(result.get("rationale", "")),
                "PriceThresholdSource": "llm_numeric",
                "ThresholdEvidenceProtocol": CROWDLLM_THRESHOLD_EVIDENCE_PROTOCOL,
                "ThresholdEvidenceFirm1Multiplier": CROWDLLM_REFERENCE_FIRM1_MULTIPLIER,
                "ThresholdEvidenceFirm2Multiplier": CROWDLLM_REFERENCE_FIRM2_MULTIPLIER,
                "ThresholdPromptProtocol": str(manifest["prompt_protocol"]),
                "ThresholdGenerationMode": "llm_numeric",
                "ThresholdAuditProtocol": "none_numeric_llm_value_retained",
                "ThresholdPromptBatchSize": int(result["prompt_batch_size"]),
                "ThresholdPromptModel": str(manifest["model"]),
                "ThresholdResponseModel": str(result["response_model"]),
                "ThresholdResponseID": str(result.get("response_id", "")),
                "ThresholdRawLLMValue": value,
                "ThresholdLLMClipped": False,
                "ThresholdEconomicContract": "v28_unbounded_finite_positive",
                "ThresholdRequestCustomID": str(result["custom_id"]),
                "ThresholdPromptSHA256": str(result["prompt_sha256"]),
                "ThresholdRequestBodySHA256": str(result["request_body_sha256"]),
                "ThresholdSourceProfileSHA256": str(result["source_row_sha256"]),
                "ThresholdPromptIdentitySHA256": str(result["prompt_identity_sha256"]),
                "ExperimentalUnboundedEconomics": True,
            })
            output_rows.append(base)
            values.append(value)
        if len({source_row_hash(row) for row in output_rows}) != expected:
            raise ThresholdBatchError("final cache rider/cold-start rows are not unique")
        if ordered_digest(source_row_hash(row) for row in output_rows) != (
            manifest.get("source") or {}
        )["source_row_order_sha256"]:
            raise ThresholdBatchError("final cache source row order changed")
        destination = Path(output_cache).expanduser().resolve()
        if destination == Path(str((manifest.get("source") or {})["path"])).resolve():
            raise ThresholdBatchError("final cache path must not overwrite the source cache")
        cache_payload = b"".join(
            (canonical_json(row) + "\n").encode("utf-8") for row in output_rows
        )
        cache_sha = sha256_bytes(cache_payload)
        if destination.exists() and sha256_file(destination) != cache_sha:
            raise ThresholdBatchError(
                f"refusing to overwrite a different existing final cache: {destination}"
            )
        if not destination.exists():
            _atomic_bytes(destination, cache_payload)
        actual_cost = float(
            int(usage.get("input_tokens", 0)) * float(actual_input_rate) / 1_000_000.0
            + int(usage.get("output_tokens", 0)) * float(actual_output_rate) / 1_000_000.0
        )
        report = {
            "path": str(destination),
            "bytes": int(len(cache_payload)),
            "sha256": cache_sha,
            "rows": expected,
            "unique_source_rows": len({source_row_hash(row) for row in output_rows}),
            "unique_prompt_identities": len({
                prompt_identity_hash(
                    row,
                    coldstart_rides=int(
                        (manifest.get("config") or {})["coldstart_rides"]
                    ),
                )
                for row in output_rows
            }),
            "source_row_order_sha256": ordered_digest(
                source_row_hash(row) for row in output_rows
            ),
            "prompt_identity_order_sha256": ordered_digest(
                prompt_identity_hash(
                    row,
                    coldstart_rides=int(
                        (manifest.get("config") or {})["coldstart_rides"]
                    ),
                )
                for row in output_rows
            ),
            "threshold_distribution": _quantiles(values),
            "usage": usage,
            "actual_cost_usd_at_supplied_rates": actual_cost,
            "actual_rates": {
                "input_usd_per_million": float(actual_input_rate),
                "output_usd_per_million": float(actual_output_rate),
            },
            "historical_parse_errors_resolved_by_retry": parse_errors,
            "validated_utc": utc_now(),
        }
        manifest["state"] = "validated"
        manifest["final_cache"] = report
        _save_manifest(manifest_path, manifest)
        return report


def inject_collected_output_for_test(
    work_dir: str | Path,
    *,
    shard_id: str,
    payload: bytes,
    batch_status: str = "completed",
) -> None:
    """Test helper: persist a mocked downloaded output through normal invariants."""
    directory = Path(work_dir).expanduser().resolve()
    with work_lock(directory):
        manifest_path, manifest = load_manifest(directory)
        matches = [shard for shard in manifest["shards"] if shard["shard_id"] == shard_id]
        if len(matches) != 1:
            raise ThresholdBatchError(f"unknown shard {shard_id}")
        shard = matches[0]
        path = directory / "responses" / f"{shard_id}.output.jsonl"
        _atomic_bytes(path, payload)
        shard["output_path"] = str(path)
        shard["output_sha256"] = sha256_bytes(payload)
        shard["batch_id"] = shard.get("batch_id") or f"batch_test_{shard_id}"
        shard["batch_status"] = str(batch_status)
        shard["state"] = "collected"
        _save_manifest(manifest_path, manifest)
