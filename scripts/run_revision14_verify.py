#!/usr/bin/env python3
"""Strict staged verification and acceptance runner for CRISP-Ride Revision 14."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics as st
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parent
PY = ROOT / ".venv/bin/python"
BASE12 = ROOT / "rideshare_v26r_revision12"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BASE))
import run_revision12_verify as r12  # noqa: E402
from best_static_tariff_search import COEFFICIENTS, frozen_rl_tariff  # noqa: E402
from v26r_case_eval import run_specs, spec_key  # noqa: E402
from v26r_cases import CASES  # noqa: E402


PHASE_ORDER = ("tests", "identity", "isolation", "signal", "smoke", "full")
FULL_VARIANTS = {
    "R14-A": {"oracle": True, "nmcc": True, "observations": True},
    "R14-B": {"oracle": False, "nmcc": True, "observations": True},
    "R14-C": {"oracle": False, "nmcc": True, "observations": False},
    "R14-D": {"oracle": False, "nmcc": False, "observations": True},
}


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def set_option(command: list[str], flag: str, value: str) -> None:
    if flag in command:
        command[command.index(flag) + 1] = str(value)
    else:
        command.extend([flag, str(value)])


def add_flag(command: list[str], flag: str) -> None:
    if flag not in command:
        command.append(flag)


def r14_train_command(
    out: Path,
    days: int,
    *,
    oracle: bool,
    smoke: bool,
    nmcc_branches: int = 0,
    nmcc_actor_weight: float = 0.8,
    nmcc_policy_coeff: float = 0.5,
    momentum: bool = False,
    opponent: bool = False,
    nmcc_dump: Path | None = None,
    seed: int = 251012,
    every_k: int = 1,
) -> list[str]:
    """Import Revision 12's command and change only the registered R14 fields."""
    command = r12.train_cmd(out, days, oracle, smoke)
    for flag, value in (
        ("--training_stage_market_fraction", "0.05"),
        ("--training_stage_static_fraction", "0.05"),
        ("--training_stage_dynamic_fraction", "0.55"),
        ("--training_stage_profit_fraction", "0.35"),
        ("--trained_model_id", f"r14-nyc-{'oracle' if oracle else 'public'}-{days}-s{seed}"),
        ("--seed", str(seed)),
        ("--experiment_seed", str(seed)),
    ):
        set_option(command, flag, value)
    if nmcc_branches:
        for flag, value in (
            ("--nmcc_branches", str(nmcc_branches)),
            ("--nmcc_horizon_days", "3"),
            ("--nmcc_actor_weight", str(nmcc_actor_weight)),
            ("--nmcc_policy_coeff", str(nmcc_policy_coeff)),
            ("--counterfactual_q_coeff", "0.05"),
            ("--nmcc_every_k_decisions", str(every_k)),
        ):
            set_option(command, flag, value)
        if nmcc_dump is not None:
            set_option(command, "--nmcc_dump", str(nmcc_dump.resolve()))
    if momentum:
        add_flag(command, "--momentum_features")
    if opponent:
        add_flag(command, "--opponent_features")
    r12.assert_independent_action_contract(command)
    return command


def run_process(
    command: list[str],
    out: Path,
    *,
    cwd: Path = BASE,
    log_name: str = "training.log",
) -> tuple[int, float]:
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "command.json", {"argv": command, "cwd": str(cwd)})
    started = time.perf_counter()
    with (out / log_name).open("w") as log:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=r12.env(),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    return completed.returncode, time.perf_counter() - started


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def require_previous(root: Path, phase: str) -> None:
    index = PHASE_ORDER.index(phase)
    if index == 0:
        return
    previous = PHASE_ORDER[index - 1]
    path = root / previous / "status.json"
    if not path.exists():
        raise RuntimeError(f"Phase {phase} requires completed Phase {previous}: {path}")
    status = json.loads(path.read_text())
    if not status.get("pass"):
        raise RuntimeError(f"Phase {phase} blocked by failed Phase {previous}")


def junit_failures(path: Path, exit_code: int) -> set[str]:
    if not path.exists():
        return {f"__pytest_process_exit_{exit_code}__"} if exit_code else set()
    root = ET.parse(path).getroot()
    failed = set()
    for case in root.iter("testcase"):
        if case.find("failure") is not None or case.find("error") is not None:
            failed.add(f"{case.get('classname')}::{case.get('name')}")
    return failed


def phase_tests(root: Path) -> dict[str, Any]:
    out = root / "tests"
    out.mkdir(parents=True, exist_ok=True)
    focused = [
        str(PY),
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests/test_revision14.py",
    ]
    focused_code, focused_seconds = run_process(
        focused, out / "focused", cwd=BASE, log_name="pytest.log"
    )
    suites: dict[str, Any] = {}
    for name, tree in (("revision12", BASE12), ("revision14", BASE)):
        suite_out = out / name
        suite_out.mkdir(parents=True, exist_ok=True)
        xml_path = suite_out / "junit.xml"
        if xml_path.exists():
            failures = sorted(junit_failures(xml_path, 0))
            suites[name] = {
                "exit_code": 1 if failures else 0,
                "seconds": None,
                "failures": failures,
                "reused_completed_junit": True,
            }
            continue
        command = [
            str(PY),
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            f"--junitxml={xml_path}",
        ]
        # These suites each exercise large PyArrow-backed datasets. Running them
        # concurrently can force both workers into swap and make the gate much
        # slower than running the same checks one after another.
        code, seconds = run_process(
            command, suite_out, cwd=tree, log_name="pytest.log"
        )
        suites[name] = {
            "exit_code": code,
            "seconds": seconds,
            "failures": sorted(junit_failures(xml_path, code)),
        }
    new_failures = sorted(
        set(suites["revision14"]["failures"])
        - set(suites["revision12"]["failures"])
    )
    missing_shared_failures = sorted(
        set(suites["revision12"]["failures"])
        - set(suites["revision14"]["failures"])
    )
    interrupted = any(
        item["exit_code"] in {-15, 130, 143} for item in suites.values()
    )
    passed = (
        focused_code == 0
        and not new_failures
        and not missing_shared_failures
        and not interrupted
    )
    return {
        "pass": passed,
        "focused": {"exit_code": focused_code, "seconds": focused_seconds},
        "full_suites": suites,
        "new_revision14_failures": new_failures,
        "revision12_failures_not_reproduced": missing_shared_failures,
        "interrupted": interrupted,
    }


def phase_identity(root: Path) -> dict[str, Any]:
    out = root / "identity"
    command = [
        str(PY),
        str(HERE / "r14_identity_check.py"),
        "--output-root",
        str(out),
    ]
    code, seconds = run_process(command, out / "driver", log_name="identity.log")
    report_path = out / "identity_report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    return {"pass": code == 0 and bool(report.get("pass")), "seconds": seconds, "report": report}


def compare_isolation_rows(left: Path, right: Path) -> dict[str, Any]:
    a, b = read_jsonl(left), read_jsonl(right)
    keys = (
        "avg_reward",
        "last_action",
        "raw_policy_top_action_label",
        "shock_block_counts",
    )
    mismatches = []
    if len(a) != len(b):
        mismatches.append({"row_count": [len(a), len(b)]})
    for index, (ra, rb) in enumerate(zip(a, b)):
        for key in keys:
            if ra.get(key) != rb.get(key):
                mismatches.append(
                    {"row": index, "key": key, "with_nmcc": ra.get(key), "without_nmcc": rb.get(key)}
                )
                if len(mismatches) >= 50:
                    break
        if len(mismatches) >= 50:
            break
    return {"pass": not mismatches, "rows": [len(a), len(b)], "mismatches": mismatches}


def phase_isolation(root: Path) -> dict[str, Any]:
    out = root / "isolation"
    run_a, run_b = out / "with_nmcc", out / "without_nmcc"
    command_a = r14_train_command(
        run_a,
        40,
        oracle=True,
        smoke=True,
        nmcc_branches=3,
        nmcc_actor_weight=0.0,
        nmcc_policy_coeff=0.0,
    )
    command_b = r14_train_command(run_b, 40, oracle=True, smoke=True)
    set_option(command_b, "--counterfactual_q_coeff", "0.05")
    code_b, seconds_b = run_process(command_b, run_b)
    if code_b != 0:
        return {"pass": False, "error": "control run failed", "exit_codes": {"control": code_b}}
    code_a, seconds_a = run_process(command_a, run_a)
    if code_a != 0:
        return {"pass": False, "error": "NMCC run failed", "exit_codes": {"nmcc": code_a, "control": code_b}}
    comparison = compare_isolation_rows(
        run_a / "nyc_training_live.jsonl", run_b / "nyc_training_live.jsonl"
    )
    rows = read_jsonl(run_a / "nyc_training_live.jsonl")
    calls = int(rows[-1].get("nmcc_calls", 0)) if rows else 0
    total_seconds = float(rows[-1].get("nmcc_seconds_total", 0.0)) if rows else 0.0
    return {
        "pass": comparison["pass"],
        "exit_codes": {"nmcc": code_a, "control": code_b},
        "elapsed_seconds": {"nmcc": seconds_a, "control": seconds_b},
        "seconds_per_nmcc_call": total_seconds / calls if calls else None,
        "nmcc_calls": calls,
        "comparison": comparison,
    }


def phase_signal(root: Path) -> dict[str, Any]:
    out = root / "signal"
    run = out / "training"
    dump = out / "probe.jsonl"
    command = r14_train_command(
        run,
        150,
        oracle=True,
        smoke=True,
        nmcc_branches=14,
        nmcc_actor_weight=0.0,
        nmcc_policy_coeff=0.0,
        nmcc_dump=dump,
    )
    code, seconds = run_process(command, run)
    if code != 0:
        return {"pass": False, "exit_code": code, "seconds": seconds, "error": "signal training failed"}
    report_path = out / "probe_report.json"
    report_command = [
        str(PY),
        str(HERE / "r14_probe_report.py"),
        str(dump),
        "--output",
        str(report_path),
    ]
    report_code, report_seconds = run_process(
        report_command, out / "report_driver", log_name="probe_report.log"
    )
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    return {
        "pass": report_code == 0 and bool(report.get("pass")),
        "exit_codes": {"training": code, "report": report_code},
        "seconds": {"training": seconds, "report": report_seconds},
        "report": report,
    }


def chi_square_action_regime(rows: list[dict[str, Any]]) -> dict[str, Any]:
    table: dict[str, Counter[str]] = {"shortage": Counter(), "calm": Counter()}
    for row in rows:
        raw = row.get("shock_block_counts", "{}")
        shocks = json.loads(raw) if isinstance(raw, str) else dict(raw or {})
        regime = "shortage" if int(shocks.get("driver_shortage", 0)) > 0 else "calm" if not shocks or set(shocks) <= {"calm"} else None
        if regime:
            table[regime][str(row.get("raw_policy_top_action_label", ""))] += 1
    actions = sorted(set(table["shortage"]) | set(table["calm"]))
    observed = [[table[regime][action] for action in actions] for regime in ("shortage", "calm")]
    row_totals = [sum(row) for row in observed]
    column_totals = [sum(observed[r][c] for r in range(2)) for c in range(len(actions))]
    total = sum(row_totals)
    chi2 = 0.0
    used_columns = 0
    if total and all(row_totals):
        for column, column_total in enumerate(column_totals):
            if not column_total:
                continue
            used_columns += 1
            for row in range(2):
                expected = row_totals[row] * column_total / total
                chi2 += (observed[row][column] - expected) ** 2 / expected
    df = max(0, used_columns - 1)
    if df:
        import mpmath

        p = float(mpmath.gammainc(df / 2.0, chi2 / 2.0, mpmath.inf, regularized=True))
    else:
        p = 1.0
    return {
        "actions": actions,
        "observed": observed,
        "regime_counts": dict(zip(("shortage", "calm"), row_totals)),
        "chi_square": chi2,
        "df": df,
        "p": p,
        "pass": p < 0.01,
    }


def phase_smoke(root: Path) -> dict[str, Any]:
    out = root / "smoke"
    run = out / "training"
    command = r14_train_command(
        run,
        600,
        oracle=True,
        smoke=True,
        nmcc_branches=3,
        momentum=True,
        opponent=True,
    )
    code, seconds = run_process(command, run)
    if code != 0:
        return {"pass": False, "exit_code": code, "seconds": seconds, "error": "smoke training failed"}
    rows = read_jsonl(run / "nyc_training_live.jsonl")
    ppo_rows = read_jsonl(run / "nyc_ppo_updates.jsonl")
    nmcc_stage_rows = [
        row for row in rows
        if row.get("training_stage_opponent_regime") in {"catchup", "profit"}
        and row.get("ppo_update_performed")
    ]
    valid_fractions = [float(row.get("ppo_nmcc_valid_fraction", 0.0)) for row in nmcc_stage_rows]
    valid_gate = bool(valid_fractions) and min(valid_fractions) > 0.5
    kls = [float(row["ppo_approx_kl"]) for row in ppo_rows if row.get("ppo_update_performed")]
    median_kl = st.median(kls) if kls else 0.0
    action_test = chi_square_action_regime(rows[-200:])
    days_per_hour = 600.0 / max(seconds / 3600.0, 1e-9)
    projected_hours = 4000.0 / days_per_hour
    recommended_k = 1 if projected_hours <= 36.0 else 2
    gates = {
        "nmcc_valid_fraction": {
            "minimum": min(valid_fractions) if valid_fractions else None,
            "median": st.median(valid_fractions) if valid_fractions else None,
            "n": len(valid_fractions),
            "pass": valid_gate,
        },
        "median_kl": {"value": median_kl, "pass": median_kl > 0.005},
        "greedy_action_chi_square": action_test,
    }
    return {
        "pass": all(item["pass"] for item in gates.values()),
        "exit_code": code,
        "seconds": seconds,
        "days_per_hour": days_per_hour,
        "projected_4000_day_hours": projected_hours,
        "recommended_nmcc_every_k_decisions": recommended_k,
        "gates": gates,
    }


def mean_se_t(values: Iterable[float]) -> dict[str, Any]:
    xs = [float(x) for x in values]
    if not xs:
        return {"mean": None, "se": None, "t": None, "lower_95": None, "n": 0}
    mean = st.mean(xs)
    se = st.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else None
    t_value = (
        mean / se
        if se is not None and se > 0
        else math.copysign(math.inf, mean)
        if se == 0 and mean != 0
        else None
    )
    return {
        "mean": mean,
        "se": se,
        "t": t_value,
        "lower_95": mean - 1.96 * se if se is not None else None,
        "n": len(xs),
    }


def payload(path: Path, spec: str) -> dict[str, Any]:
    return json.loads((path / f"{spec_key(spec)}.json").read_text())


def opening_rows(data: dict[str, Any], *, start_day: int = 5) -> dict[str, list[dict[str, Any]]]:
    spd = int(data["steps_per_day"])
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in data["traces"]:
        if int(row["step"]) // spd >= start_day:
            result[str(row["rollout"])].append(row)
    return dict(result)


def paired_reward_difference(
    left: dict[str, list[dict[str, Any]]],
    right: dict[str, list[dict[str, Any]]],
    predicate,
) -> dict[str, Any]:
    differences = []
    for opening in sorted(set(left) & set(right)):
        a = [float(row["reward"]) for row in left[opening] if predicate(row)]
        b = [float(row["reward"]) for row in right[opening] if predicate(row)]
        if a and b:
            differences.append(st.mean(a) - st.mean(b))
    return mean_se_t(differences)


def acceptance_for_policy(eval_root: Path, best_spec: str, frozen_spec: str) -> dict[str, Any]:
    case_data = {
        case: {
            spec: opening_rows(payload(eval_root / case, spec))
            for spec in ("rl_actor", best_spec, frozen_spec)
        }
        for case in ("A1_shortage_severe", "A2_shortage_moderate_null", "S1_promotion_bait", "S2_undercut_trap", "C0_calm")
    }
    a1_rl = case_data["A1_shortage_severe"]["rl_actor"]
    p1_diffs = []
    for rows in a1_rl.values():
        shortage = [float(r["firm1_reference_fare_index"]) for r in rows if "driver_shortage" in str(r.get("regime", "")) and "recovery" not in str(r.get("regime", ""))]
        calm = [float(r["firm1_reference_fare_index"]) for r in rows if str(r.get("regime", "")) == "calm"]
        if shortage and calm:
            p1_diffs.append(st.mean(shortage) - st.mean(calm))
    p1 = mean_se_t(p1_diffs)
    p1["pass"] = bool(p1["mean"] is not None and p1["mean"] >= 0.10 and p1["t"] is not None and p1["t"] >= 3.0)

    shortage_predicate = lambda row: "driver_shortage" in str(row.get("regime", "")) and "recovery" not in str(row.get("regime", ""))
    all_predicate = lambda row: True
    p2_shortage = paired_reward_difference(a1_rl, case_data["A1_shortage_severe"][best_spec], shortage_predicate)
    p2_all = paired_reward_difference(a1_rl, case_data["A1_shortage_severe"][best_spec], all_predicate)
    p2 = {
        "shortage_days": p2_shortage,
        "all_days": p2_all,
        "pass": bool(
            p2_shortage["mean"] is not None
            and p2_shortage["mean"] > 0.0
            and p2_shortage["t"] is not None
            and p2_shortage["t"] >= 2.0
            and p2_all["lower_95"] is not None
            and p2_all["lower_95"] > -0.01
        ),
    }
    p3_stats = paired_reward_difference(
        case_data["C0_calm"]["rl_actor"], case_data["C0_calm"][best_spec], all_predicate
    )
    p3_stats["pass"] = bool(p3_stats["mean"] is not None and p3_stats["se"] is not None and p3_stats["mean"] >= -2.0 * p3_stats["se"])
    p4_stats = paired_reward_difference(
        case_data["A2_shortage_moderate_null"]["rl_actor"], case_data["A2_shortage_moderate_null"][best_spec], all_predicate
    )
    p4_stats["pass"] = bool(p4_stats["mean"] is not None and p4_stats["se"] is not None and abs(p4_stats["mean"]) <= 2.0 * p4_stats["se"])

    p5: dict[str, Any] = {}
    for case in ("S1_promotion_bait", "S2_undercut_trap"):
        onset = int(CASES[case]["onset"])
        rl = case_data[case]["rl_actor"]
        frozen = case_data[case][frozen_spec]
        changes = []
        for rows in rl.values():
            spd = 6
            before = [float(r["firm1_reference_fare_index"]) for r in rows if onset - 2 <= int(r["step"]) // spd < onset]
            after = [float(r["firm1_reference_fare_index"]) for r in rows if onset <= int(r["step"]) // spd <= onset + 1]
            if before and after:
                changes.append(st.mean(after) - st.mean(before))
        change = mean_se_t(changes)
        move_predicate = lambda row, onset=onset: onset <= int(row["step"]) // 6 <= onset + 1
        reward = paired_reward_difference(rl, frozen, move_predicate)
        passed = bool(
            (change["mean"] is not None and abs(change["mean"]) >= 0.05)
            or (reward["mean"] is not None and reward["mean"] > 0 and reward["t"] is not None and reward["t"] >= 2.0)
        )
        p5[case] = {"reference_fare_change": change, "reward_vs_frozen": reward, "pass": passed}
    return {"P1": p1, "P2": p2, "P3": p3_stats, "P4": p4_stats, "P5": p5}


def write_frozen(path: Path, diagnostics: Path) -> None:
    path.write_text(json.dumps(frozen_rl_tariff(diagnostics), indent=2) + "\n")


def phase_full(root: Path) -> dict[str, Any]:
    out = root / "full"
    smoke = json.loads((root / "smoke" / "status.json").read_text())
    every_k = int(smoke.get("result", smoke).get("recommended_nmcc_every_k_decisions", 1))
    trainings: dict[str, Any] = {}
    jobs: dict[str, tuple[subprocess.Popen, Any, float, Path]] = {}
    for name, settings in FULL_VARIANTS.items():
        run = out / name / "training"
        command = r14_train_command(
            run,
            4000,
            oracle=settings["oracle"],
            smoke=False,
            nmcc_branches=3 if settings["nmcc"] else 0,
            momentum=settings["observations"],
            opponent=settings["observations"],
            every_k=every_k,
        )
        run.mkdir(parents=True, exist_ok=True)
        atomic_json(run / "command.json", {"argv": command, "cwd": str(BASE)})
        log = (run / "training.log").open("w")
        process = subprocess.Popen(
            command,
            cwd=BASE,
            env=r12.env(),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        jobs[name] = (process, log, time.perf_counter(), run)
        trainings[name] = {
            "state": "running",
            "pid": process.pid,
            "policy": str(run / "policy.pt"),
        }
    atomic_json(out / "training_status.json", trainings)
    for name, (process, log, started, run) in jobs.items():
        code = process.wait()
        log.close()
        seconds = time.perf_counter() - started
        trainings[name] = {
            "state": "complete" if code == 0 else "failed",
            "exit_code": code,
            "seconds": seconds,
            "policy": str(run / "policy.pt"),
        }
        atomic_json(out / "training_status.json", trainings)
    failed_trainings = [name for name, item in trainings.items() if item["exit_code"] != 0]
    if failed_trainings:
        return {
            "pass": False,
            "error": f"full training failed: {failed_trainings}",
            "trainings": trainings,
        }

    search_root = out / "static_tariff_search"
    b_run = out / "R14-B" / "training"
    search_command = [
        str(PY),
        str(HERE / "best_static_tariff_search.py"),
        "--policy",
        str(b_run / "policy.pt"),
        "--diagnostics",
        str(b_run / "nyc_evaluation_diagnostics.csv"),
        "--output-root",
        str(search_root),
    ]
    search_code, search_seconds = run_process(search_command, search_root / "driver", log_name="search.log")
    if search_code != 0:
        return {"pass": False, "error": "static tariff search failed", "trainings": trainings, "search_exit_code": search_code}
    best_path = (search_root / "best_static_tariff.json").resolve()
    best_spec = f"static_tariff:@{best_path}"
    static_specs = [f"static:{value:.1f}" for value in (1.1, 1.2, 1.3, 1.4, 1.5, 1.6)]
    acceptances: dict[str, Any] = {}
    for name in FULL_VARIANTS:
        run = out / name / "training"
        frozen_path = (out / name / "frozen_rl_tariff.json").resolve()
        write_frozen(frozen_path, run / "nyc_evaluation_diagnostics.csv")
        frozen_spec = f"static_tariff:@{frozen_path}"
        specs = ["rl_actor", *static_specs, best_spec, frozen_spec]
        eval_root = out / name / "evaluation"
        for case in ("A1_shortage_severe", "A2_shortage_moderate_null", "S1_promotion_bait", "S2_undercut_trap", "C0_calm"):
            case_out = eval_root / case
            case_out.mkdir(parents=True, exist_ok=True)
            run_specs(
                "New York City",
                "nyc",
                case,
                specs,
                "eval",
                10,
                "full",
                str(run / "policy.pt"),
                str(case_out),
                customers=50,
                horizon_days=40,
                rivals=["adaptive_best_response_memory"],
                pool=2000,
            )
        acceptances[name] = acceptance_for_policy(eval_root, best_spec, frozen_spec)
        atomic_json(out / "acceptance_partial.json", acceptances)
    b = acceptances["R14-B"]
    gated = bool(b["P1"]["pass"] and b["P2"]["pass"] and b["P3"]["pass"])
    attribution = {
        name: {"P1": acceptances[name]["P1"], "P2": acceptances[name]["P2"]}
        for name in ("R14-B", "R14-C", "R14-D")
    }
    return {
        "pass": gated,
        "trainings": trainings,
        "static_search_seconds": search_seconds,
        "nmcc_every_k_decisions": every_k,
        "acceptance": acceptances,
        "P6_attribution": attribution,
        "public_gated_P1_P2_P3_pass": gated,
        "replication_seeds_required_if_pass": [251013, 251014] if gated else [],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=PHASE_ORDER)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=BASE / "runs" / "revision14_verify",
    )
    parser.add_argument("--launchd-label", default="")
    parser.add_argument(
        "--override-failed-smoke",
        action="store_true",
        help=(
            "Allow the full phase to run after a completed but failed smoke gate. "
            "This does not bypass any earlier correctness phase."
        ),
    )
    args = parser.parse_args()
    root = args.run_root.resolve()
    phase_root = root / args.phase
    phase_root.mkdir(parents=True, exist_ok=True)
    atomic_json(phase_root / "status.json", {"phase": args.phase, "state": "running", "pass": False, "started_at": time.time()})
    try:
        if args.override_failed_smoke:
            if args.phase != "full":
                raise RuntimeError("--override-failed-smoke is valid only for the full phase")
            smoke_path = root / "smoke" / "status.json"
            if not smoke_path.exists():
                raise RuntimeError(f"Full-phase override requires a completed smoke result: {smoke_path}")
            smoke_status = json.loads(smoke_path.read_text())
            if smoke_status.get("state") not in {"complete", "failed"}:
                raise RuntimeError("Full-phase override requires the smoke phase to be finished")
            atomic_json(
                phase_root / "smoke_gate_override.json",
                {
                    "authorized_by": "user",
                    "authorized_action": "run the full budget, 4000 day staged training",
                    "created_at": time.time(),
                    "scope": "full phase only",
                    "smoke_status": smoke_status,
                },
            )
        else:
            require_previous(root, args.phase)
        function = globals()[f"phase_{args.phase}"]
        result = function(root)
        passed = bool(result.get("pass"))
        atomic_json(
            phase_root / "status.json",
            {
                "phase": args.phase,
                "state": "complete" if passed else "failed",
                "pass": passed,
                "completed_at": time.time(),
                "result": result,
            },
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if passed else 2
    except BaseException as exc:
        atomic_json(
            phase_root / "status.json",
            {
                "phase": args.phase,
                "state": "failed",
                "pass": False,
                "completed_at": time.time(),
                "error": repr(exc),
            },
        )
        raise


if __name__ == "__main__":
    requested_label = ""
    if "--launchd-label" in sys.argv:
        label_index = sys.argv.index("--launchd-label")
        if label_index + 1 < len(sys.argv):
            requested_label = sys.argv[label_index + 1]
    try:
        exit_code = main()
    finally:
        if requested_label:
            subprocess.Popen(
                ["/bin/launchctl", "remove", requested_label],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
    raise SystemExit(exit_code)
