#!/usr/bin/env python3
"""Prepare and operate resumable native-uncapped v28 threshold batches.

Examples (``prepare`` and ``estimate`` never contact OpenAI)::

    python scripts/v28_threshold_cache.py estimate \
      --city chicago \
      --source-cache artifacts/v26_confirmatory/thresholds/chicago/llm_numeric_primary.jsonl \
      --calibration-json data/processed/chicago/calibration.json \
      --work-dir artifacts/v28_thresholds/chicago

    python scripts/v28_threshold_cache.py prepare ...same arguments...
    python scripts/v28_threshold_cache.py submit --work-dir artifacts/v28_thresholds/chicago
    python scripts/v28_threshold_cache.py status --work-dir artifacts/v28_thresholds/chicago
    python scripts/v28_threshold_cache.py collect --work-dir artifacts/v28_thresholds/chicago
    python scripts/v28_threshold_cache.py requeue-failed-validation \
      --work-dir artifacts/v28_thresholds/chicago --shard-id SHARD_ID
    python scripts/v28_threshold_cache.py retry-missing --work-dir artifacts/v28_thresholds/chicago
    python scripts/v28_threshold_cache.py validate \
      --work-dir artifacts/v28_thresholds/chicago \
      --output-cache artifacts/v28_thresholds/chicago/native_uncapped_primary.jsonl

Only ``submit``, ``status``, and ``collect`` use ``OPENAI_API_KEY``.  The key is
never accepted as a CLI argument and is never written to an artifact.  The
current organization-wide GPT-4.1 Batch allowance requires one roughly 1.2M
token shard globally at a time, not one per city.  Refresh all city statuses,
inspect ``queue-ledger``, collect the terminal job, and only then submit the
next shard with the peer/global guard options.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v28_threshold_batch import (  # noqa: E402
    BATCH_INPUT_USD_PER_MTOK,
    BATCH_OUTPUT_USD_PER_MTOK,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHARS_PER_TOKEN,
    DEFAULT_GLOBAL_MAX_QUEUED_TOKENS,
    DEFAULT_MAX_QUEUED_TOKENS,
    DEFAULT_MAX_SHARD_BYTES,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_TOKENS_PER_REQUEST,
    STANDARD_INPUT_USD_PER_MTOK,
    STANDARD_OUTPUT_USD_PER_MTOK,
    collect,
    prepare,
    queue_ledger,
    requeue_failed_validation,
    retry_missing,
    status,
    submit,
    validate,
)


def _print(value: Dict[str, Any]) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def _add_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--city", required=True, choices=("nyc", "boston", "chicago"))
    parser.add_argument("--source-cache", required=True)
    parser.add_argument("--calibration-json", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--expected-rows", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Pinned Responses model; snapshot default makes regeneration reproducible.",
    )
    parser.add_argument("--seed", type=int, default=4500)
    parser.add_argument("--coldstart-rides", type=int, default=5)
    parser.add_argument("--include-rationales", action="store_true")
    parser.add_argument("--max-shard-bytes", type=int, default=DEFAULT_MAX_SHARD_BYTES)
    parser.add_argument(
        "--max-queued-tokens",
        type=int,
        default=DEFAULT_MAX_QUEUED_TOKENS,
        help="Conservative estimated prompt-token ceiling per content-addressed shard.",
    )
    parser.add_argument(
        "--chars-per-token",
        type=float,
        default=DEFAULT_CHARS_PER_TOKEN,
        help="Planning estimator only; lower is more conservative.",
    )
    parser.add_argument(
        "--estimated-output-tokens-per-request",
        type=int,
        default=DEFAULT_OUTPUT_TOKENS_PER_REQUEST,
    )
    parser.add_argument("--standard-input-rate", type=float, default=STANDARD_INPUT_USD_PER_MTOK)
    parser.add_argument("--standard-output-rate", type=float, default=STANDARD_OUTPUT_USD_PER_MTOK)
    parser.add_argument("--batch-input-rate", type=float, default=BATCH_INPUT_USD_PER_MTOK)
    parser.add_argument("--batch-output-rate", type=float, default=BATCH_OUTPUT_USD_PER_MTOK)


def _prepare_from_args(args: argparse.Namespace, *, dry_run: bool) -> Dict[str, Any]:
    return prepare(
        city=args.city,
        source_cache=args.source_cache,
        calibration_path=args.calibration_json,
        work_dir=args.work_dir,
        expected_rows=args.expected_rows,
        batch_size=args.batch_size,
        model=args.model,
        seed=args.seed,
        coldstart_rides=args.coldstart_rides,
        include_rationales=args.include_rationales,
        max_shard_bytes=args.max_shard_bytes,
        max_queued_tokens=args.max_queued_tokens,
        chars_per_token=args.chars_per_token,
        output_tokens_per_request=args.estimated_output_tokens_per_request,
        standard_input_rate=args.standard_input_rate,
        standard_output_rate=args.standard_output_rate,
        batch_input_rate=args.batch_input_rate,
        batch_output_rate=args.batch_output_rate,
        dry_run=dry_run,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    estimate_parser = subparsers.add_parser(
        "estimate", help="Build every payload in memory and report calls/tokens/cost; no writes or API calls."
    )
    _add_prepare_arguments(estimate_parser)

    prepare_parser = subparsers.add_parser(
        "prepare", help="Write deterministic request shards and an atomic manifest; no API calls."
    )
    _add_prepare_arguments(prepare_parser)

    submit_parser = subparsers.add_parser(
        "submit", help="Upload and submit prepared shards using OPENAI_API_KEY."
    )
    submit_parser.add_argument("--work-dir", required=True)
    submit_parser.add_argument(
        "--all",
        action="store_true",
        help="Fill the active estimated-token ceiling; default submits one shard.",
    )
    submit_parser.add_argument(
        "--max-active-estimated-tokens",
        type=int,
        default=None,
        help="Override the manifest's conservative queued-token ceiling.",
    )
    submit_parser.add_argument(
        "--peer-work-dir",
        action="append",
        default=[],
        help=(
            "Peer city manifest to include in the organization-wide queue guard; "
            "repeat for every other city after refreshing statuses."
        ),
    )
    submit_parser.add_argument(
        "--global-max-active-estimated-tokens",
        type=int,
        default=None,
        help=(
            "Enable a cross-manifest queue guard, normally 1350000 for the "
            "currently observed GPT-4.1 Batch organization limit."
        ),
    )

    for name, help_text in (
        ("status", "Refresh submitted Batch API job states."),
        ("collect", "Refresh states and atomically download terminal output/error files."),
        ("retry-missing", "Prepare new shards containing only unresolved profiles; no API calls."),
    ):
        command_parser = subparsers.add_parser(name, help=help_text)
        command_parser.add_argument("--work-dir", required=True)

    requeue_parser = subparsers.add_parser(
        "requeue-failed-validation",
        help=(
            "Locally reset only failed Batch jobs proven to have processed zero "
            "requests; archives immutable attempt history and performs no API call."
        ),
    )
    requeue_parser.add_argument("--work-dir", required=True)
    requeue_parser.add_argument(
        "--shard-id",
        action="append",
        default=None,
        help="Explicit failed shard to requeue; repeat as needed. Default audits all shards.",
    )

    ledger_parser = subparsers.add_parser(
        "queue-ledger",
        help="Report active/prepared tokens across city manifests; no API calls.",
    )
    ledger_parser.add_argument(
        "--work-dir", action="append", required=True,
        help="City work directory; repeat for every city sharing Batch quota.",
    )
    ledger_parser.add_argument(
        "--global-max-active-estimated-tokens",
        type=int,
        default=DEFAULT_GLOBAL_MAX_QUEUED_TOKENS,
    )

    validate_parser = subparsers.add_parser(
        "validate", help="Fail closed unless all profiles have native valid API values, then write cache."
    )
    validate_parser.add_argument("--work-dir", required=True)
    validate_parser.add_argument("--output-cache", required=True)
    validate_parser.add_argument(
        "--actual-input-rate", type=float, default=BATCH_INPUT_USD_PER_MTOK
    )
    validate_parser.add_argument(
        "--actual-output-rate", type=float, default=BATCH_OUTPUT_USD_PER_MTOK
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "estimate":
        result = _prepare_from_args(args, dry_run=True)
        # The full preview contains 20k custom IDs.  Keep CLI output compact
        # while the normal prepare command persists the complete manifest.
        result.pop("manifest_preview", None)
    elif args.command == "prepare":
        manifest = _prepare_from_args(args, dry_run=False)
        result = {
            "work_dir": str(Path(args.work_dir).expanduser().resolve()),
            "run_fingerprint": manifest["run_fingerprint"],
            "state": manifest["state"],
            **manifest["estimate"],
        }
    elif args.command == "submit":
        result = submit(
            args.work_dir,
            submit_all=bool(args.all),
            max_active_estimated_tokens=args.max_active_estimated_tokens,
            peer_work_dirs=args.peer_work_dir,
            global_max_active_estimated_tokens=(
                args.global_max_active_estimated_tokens
            ),
        )
    elif args.command == "status":
        result = status(args.work_dir)
    elif args.command == "collect":
        result = collect(args.work_dir)
    elif args.command == "retry-missing":
        result = retry_missing(args.work_dir)
    elif args.command == "requeue-failed-validation":
        result = requeue_failed_validation(
            args.work_dir, shard_ids=args.shard_id
        )
    elif args.command == "queue-ledger":
        result = queue_ledger(
            args.work_dir,
            global_max_active_estimated_tokens=(
                args.global_max_active_estimated_tokens
            ),
        )
    elif args.command == "validate":
        result = validate(
            args.work_dir,
            output_cache=args.output_cache,
            actual_input_rate=args.actual_input_rate,
            actual_output_rate=args.actual_output_rate,
        )
    else:  # pragma: no cover - argparse enforces the command set
        raise AssertionError(args.command)
    _print(result)


if __name__ == "__main__":
    main()
