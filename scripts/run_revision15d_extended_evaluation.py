#!/usr/bin/env python3
"""Extend the running eight-hour R15d screen to complete case coverage.

The primary pipeline keeps its bounded six-case evaluation.  This independent,
resumable follower waits for that pipeline to finish, evaluates the six omitted
cases with the identical policy/opponents/comparators/CRNs, and rebuilds the
summary over all twelve configured cases.  It never mutates policy weights.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BASE))

from diagnostic_logging import atomic_json  # noqa: E402
import run_revision15d_8h_multicity as bounded  # noqa: E402
import run_revision15d_multicity_full as full  # noqa: E402


ALL_CASES = tuple(full.CASES)
MISSING_CASES = tuple(
    name for name in ALL_CASES if name not in bounded.EVALUATION_CASES
)

TALK_J3_CASES = ("A1_shortage_severe", "C0_calm")
TALK_J3_OPENINGS = 10
TALK_J3_HORIZON_DAYS = 28
TALK_J3_RIVALS = ("adaptive_best_response_memory",)
TALK_J3_J1_STATUS = (
    BASE.parent
    / "rideshare_v26r_revision15"
    / "runs"
    / "j1_rule_vs_rl"
    / "status.json"
)
TALK_J3_RUN_ROOT = (
    BASE
    / "runs"
    / "revision15d_long_full_cf_588d_20261025_1410"
)


def update(out: Path, **values) -> None:
    path = out / "extended_evaluation_status.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    if values.get("state") != "failed":
        data.pop("error", None)
    data.update(values, updated_at=time.time())
    atomic_json(path, data)


def talk_j3_update(out: Path, **values) -> None:
    path = out / "talk_j3_status.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    if values.get("state") != "failed":
        data.pop("error", None)
    data.update(values, updated_at=time.time())
    atomic_json(path, data)


def talk_j3_specs(rule_spec: str) -> tuple[str, ...]:
    if not rule_spec.startswith("rule:"):
        raise ValueError("J1 winner is not a registered rule specification")
    return (
        "rl_actor",
        "rl_actor_frozen",
        "opening_hold",
        "static:1.0",
        "static:1.2",
        rule_spec,
    )


def talk_j3_contract(rule_spec: str = "rule:<J1 winner>") -> dict[str, object]:
    return {
        "mode": "talk_j3",
        "cases": list(TALK_J3_CASES),
        "openings": TALK_J3_OPENINGS,
        "horizon_days": TALK_J3_HORIZON_DAYS,
        "rivals": list(TALK_J3_RIVALS),
        "policy_specs": list(talk_j3_specs(rule_spec)),
        "cities": list(full.CITIES),
        "parallel_launch_required": True,
        "launch_not_before": "2026-10-27T07:00:00-07:00",
        "launch_cutoff": "2026-10-27T20:00:00-07:00",
        "requires_all_city_primary_status_complete": True,
        "output_subdirectory": "talk_j3_evaluation",
    }


def validate_talk_j3_launch_gate() -> None:
    now = dt.datetime.now(ZoneInfo("America/Los_Angeles"))
    start = dt.datetime(2026, 10, 27, 7, tzinfo=now.tzinfo)
    cutoff = dt.datetime(2026, 10, 27, 20, tzinfo=now.tzinfo)
    if not start <= now < cutoff:
        raise RuntimeError(
            f"talk J3 launch outside registered window: {now.isoformat()}"
        )
    incomplete = []
    for slug in full.CITIES:
        path = TALK_J3_RUN_ROOT / slug / "status.json"
        if not path.is_file():
            incomplete.append(f"{slug}:missing")
            continue
        status = json.loads(path.read_text())
        if status.get("state") != "complete" or status.get("phase") != "complete":
            incomplete.append(
                f"{slug}:{status.get('state')}/{status.get('phase')}"
            )
    if incomplete:
        raise RuntimeError(
            "talk J3 requires all three built-in pipelines complete: "
            + ", ".join(incomplete)
        )


def read_talk_j3_rule_spec(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    status = json.loads(path.read_text())
    if status.get("pass") is not True or status.get("stage") != "complete":
        raise RuntimeError(f"J1 status is not final and passing: {path}")
    return str(status.get("winner", ""))


def recoverable_primary_policy(out: Path, slug: str) -> Path | None:
    """Return the immutable policy when evaluation finished before reporting failed."""
    policy = out / "policy.final_training.pt"
    exit_path = out / "full_benchmark_evaluation_exit.json"
    if not policy.is_file() or not exit_path.is_file():
        return None
    try:
        evaluation_exit = json.loads(exit_path.read_text())
    except (OSError, ValueError):
        return None
    if int(evaluation_exit.get("returncode", 1)) != 0:
        return None
    expected = (
        out / "full_evaluation" / slug / case / "eval" / "sweep.json"
        for case in bounded.EVALUATION_CASES
    )
    return policy if all(path.is_file() for path in expected) else None


def wait_for_primary(out: Path, slug: str, poll_seconds: int) -> tuple[Path, bool]:
    while True:
        status_path = out / "status.json"
        if status_path.is_file():
            status = json.loads(status_path.read_text())
            state = str(status.get("state", ""))
            if state == "complete":
                policy = out / "policy.final_training.pt"
                if not policy.is_file():
                    raise FileNotFoundError(policy)
                return policy, False
            if state == "failed":
                policy = recoverable_primary_policy(out, slug)
                if policy is not None:
                    update(
                        out,
                        state="running",
                        phase="recovering_primary_summary",
                        recovered_primary_error=status.get("error"),
                    )
                    return policy, True
                raise RuntimeError(
                    f"primary pipeline failed before extension: {status.get('error')}"
                )
            update(
                out,
                state="waiting_for_primary",
                primary_phase=status.get("phase"),
            )
        time.sleep(max(5, poll_seconds))


def evaluation_command(slug: str, out: Path, policy: Path) -> list[str]:
    return [
        str(full.inherited.PYTHON),
        "-u",
        str(HERE / "v26r_case_eval.py"),
        full.CITIES[slug]["market"],
        slug,
        ",".join(MISSING_CASES),
        "--seed-set",
        "eval",
        "--openings",
        str(bounded.EVALUATION_OPENINGS),
        "--policy",
        str(policy.resolve()),
        "--output-root",
        str((out / "full_evaluation").resolve()),
        "--customers",
        "50",
        "--pool",
        "2000",
        "--rivals",
        ",".join(full.RIVALS),
        "--horizon-days",
        str(bounded.EVALUATION_HORIZON_DAYS),
        "--channels",
        "full",
        "--chunk",
        "1",
        "--workers",
        str(bounded.EVALUATION_WORKERS),
        "--budget-seconds",
        "1000000000",
        "--specs",
        ";".join(bounded.evaluation_specs(slug)),
    ]


def talk_j3_evaluation_command(
    slug: str, out: Path, policy: Path, rule_spec: str
) -> list[str]:
    return [
        str(full.inherited.PYTHON),
        "-u",
        str(HERE / "v26r_case_eval.py"),
        full.CITIES[slug]["market"],
        slug,
        ",".join(TALK_J3_CASES),
        "--seed-set",
        "eval",
        "--openings",
        str(TALK_J3_OPENINGS),
        "--policy",
        str(policy.resolve()),
        "--output-root",
        str((out / "talk_j3_evaluation").resolve()),
        "--customers",
        "50",
        "--pool",
        "2000",
        "--rivals",
        ",".join(TALK_J3_RIVALS),
        "--horizon-days",
        str(TALK_J3_HORIZON_DAYS),
        "--channels",
        "full",
        "--chunk",
        "1",
        "--workers",
        "1",
        "--budget-seconds",
        "1000000000",
        "--specs",
        ";".join(talk_j3_specs(rule_spec)),
    ]


def run_talk_j3(slug: str, out: Path, j1_status: Path) -> int:
    try:
        validate_talk_j3_launch_gate()
        rule_spec = read_talk_j3_rule_spec(j1_status)
        policy = out / "policy.final_training.pt"
        if not policy.is_file():
            raise FileNotFoundError(policy)
        digest = full.inherited.sha256(policy)
        contract = talk_j3_contract(rule_spec)
        talk_j3_update(
            out,
            state="running",
            phase="evaluation",
            contract=contract,
            policy_sha256=digest,
            j1_status=str(j1_status.resolve()),
        )
        full.inherited.run_process(
            talk_j3_evaluation_command(slug, out, policy, rule_spec),
            out,
            "talk_j3_benchmark_evaluation",
            "talk_j3_benchmark_evaluation.log",
        )
        if full.inherited.sha256(policy) != digest:
            raise ValueError("policy changed during talk J3 evaluation")
        talk_j3_update(
            out,
            state="complete",
            phase="complete",
            completed_at=time.time(),
            contract=contract,
            policy_sha256=digest,
        )
        return 0
    except BaseException as exc:
        traceback.print_exc()
        talk_j3_update(out, state="failed", phase="failed", error=repr(exc))
        return 1


def summarizer_command(slug: str, out: Path) -> list[str]:
    return [
        str(full.inherited.PYTHON),
        "-u",
        str(HERE / "summarize_revision15b_benchmarks.py"),
        "--evaluation-root",
        str((out / "full_evaluation").resolve()),
        "--slug",
        slug,
        "--city",
        full.CITIES[slug]["market"],
        "--output",
        str((out / "full_evaluation_summary_all_cases").resolve()),
        "--cases",
        "all",
        "--openings",
        str(bounded.EVALUATION_OPENINGS),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--city", choices=tuple(full.CITIES))
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--launch-label", default="")
    parser.add_argument("--talk-j3", action="store_true")
    parser.add_argument("--print-talk-j3-contract", action="store_true")
    parser.add_argument("--j1-status", type=Path, default=TALK_J3_J1_STATUS)
    args = parser.parse_args()
    if args.print_talk_j3_contract:
        print(json.dumps(talk_j3_contract(), indent=2))
        return 0
    if args.output is None or args.city is None:
        parser.error("--output and --city are required unless printing the contract")
    out = args.output.resolve()
    if args.talk_j3:
        return run_talk_j3(args.city, out, args.j1_status.resolve())
    try:
        update(
            out,
            state="waiting_for_primary",
            all_cases=list(ALL_CASES),
            added_cases=list(MISSING_CASES),
        )
        policy, repair_primary_summary = wait_for_primary(
            out, args.city, args.poll_seconds
        )
        digest = full.inherited.sha256(policy)
        if repair_primary_summary:
            update(out, state="running", phase="repairing_primary_summary")
            full.inherited.run_process(
                bounded.summarizer_command(args.city, out),
                out,
                "benchmark_summary_repair",
                "benchmark_summary_repair.log",
            )
        update(out, state="running", phase="additional_cases")
        full.inherited.run_process(
            evaluation_command(args.city, out, policy),
            out,
            "extended_benchmark_evaluation",
            "extended_benchmark_evaluation.log",
        )
        update(out, state="running", phase="all_case_summary")
        full.inherited.run_process(
            summarizer_command(args.city, out),
            out,
            "extended_benchmark_summary",
            "extended_benchmark_summary.log",
        )
        if full.inherited.sha256(policy) != digest:
            raise ValueError("policy changed during extended evaluation")
        full.inherited.v26r_figures.gallery(out)
        primary_status_path = out / "status.json"
        primary_status = json.loads(primary_status_path.read_text())
        primary_evaluation = bounded.validate_full_evaluation(args.city, out)
        primary_status.pop("error", None)
        primary_status.pop("failed_at", None)
        primary_status.update(
            state="complete",
            phase="complete",
            completed_at=time.time(),
            updated_at=time.time(),
            policy_sha256=digest,
            training=primary_status.get("training", {}),
            evaluation=primary_evaluation,
            extended_evaluation={
                "state": "complete",
                "total_cases": len(ALL_CASES),
                "added_cases": list(MISSING_CASES),
            },
        )
        atomic_json(primary_status_path, primary_status)
        atomic_json(
            out / "exit_status.json",
            {"returncode": 0, "completed_at": time.time()},
        )
        update(
            out,
            state="complete",
            phase="complete",
            completed_at=time.time(),
            policy_sha256=digest,
            total_cases=len(ALL_CASES),
        )
        return 0
    except BaseException as exc:
        traceback.print_exc()
        update(out, state="failed", phase="failed", error=repr(exc))
        return 1
    finally:
        if args.launch_label:
            subprocess.Popen(
                ["/bin/launchctl", "remove", args.launch_label],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )


if __name__ == "__main__":
    raise SystemExit(main())
