#!/usr/bin/env python3
"""Strict staged verification, training and acceptance runner for CRISP-Ride Revision 15.

Phases run strictly in order; each phase requires the previous phase's status.json to
record pass=true. There is no override flag.

  tests      focused Revision 15 tests + full suite (zero failures required for 15b)
  identity   flags-off Revision 15 is bit-identical to Revision 14 (20-day run)
  isolation  NMCC with zero credit weights does not change the level-step trajectory
  structure  validation-only search for the structure S and reference level L_ref
  probe      150-day NMCC probe on the level contract: raise advantage, shortage > calm
  smoke      600-day R15-B run: NMCC rows valid, imitation reduces KL, policy moves
  train      4000-day R15-A (oracle), R15-B (public), R15-C (public, no imitation)
  evaluate   held-out case evaluation and acceptance P1-P5, robustness R1-R2
  replicate  R15-B settings on seeds 251013 and 251014; both must pass P1-P3
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parent
PY = ROOT / ".venv/bin/python"
BASE14 = ROOT / "rideshare_v26r_revision14"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BASE))
import run_revision12_verify as r12  # noqa: E402
import run_revision14_verify as r14  # noqa: E402
from v26r_case_eval import run_specs  # noqa: E402
from pricing_models import (  # noqa: E402
    snap_tariff_level,
    tariff_level_of,
    validate_tariff_structure,
)

PHASE_ORDER = (
    "tests", "identity", "isolation", "structure", "probe", "smoke", "train", "evaluate", "replicate",
)
REPLICATION_SEEDS = (251013, 251014)
LEVEL_BOUNDS = "0.6,2.0"
COEFFICIENTS = ("base_fare", "per_minute", "per_mile", "booking_fee", "airport_fee")
ACTION_COUNT = 5
TRAIN_DAYS = 4000
TRAIN_VARIANTS = {
    "R15-A": {"oracle": True, "imitation": 0.5},
    "R15-B": {"oracle": False, "imitation": 0.5},
    "R15-C": {"oracle": False, "imitation": 0.0},
}
ACCEPTANCE_CASES = (
    "A1_shortage_severe",
    "A2_shortage_moderate_null",
    "S1_promotion_bait",
    "S2_undercut_trap",
    "C0_calm",
)
ROBUSTNESS_CASES = ("A3_storm_transit", "A5_rival_outage")


# ------------------------------------------------------------------ helpers
def remove_option(command: list[str], flag: str) -> None:
    if flag in command:
        index = command.index(flag)
        del command[index:index + 2]


def option(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def city_anchors() -> dict[str, float]:
    calibration = json.loads((ROOT / "data/processed/nyc/calibration.json").read_text())
    tariff = calibration["calibration"]["tariff"]
    return {key: float(tariff[key]) for key in COEFFICIENTS}


def write_uniform_structure(path: Path, reference_level: float = 1.2) -> Path:
    """Correctness-phase structure: S = 1 for every coefficient (level 1 by definition)."""
    payload = {
        "format": "revision15-tariff-structure-v1",
        "structure": {key: 1.0 for key in COEFFICIENTS},
        "anchors": city_anchors(),
        "reference_level": reference_level,
        "source": {"purpose": "isolation phase only; not used for training"},
    }
    validated = validate_tariff_structure(payload, list(COEFFICIENTS), (0.6, 2.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(validated, indent=2, sort_keys=True) + "\n")
    return path.resolve()


def assert_level_contract(command: list[str]) -> None:
    """Fail before launch unless the command uses the Revision 15 level contract."""
    if "--level_step_actions" not in command:
        raise ValueError("Revision 15 requires --level_step_actions")
    for forbidden in ("--tariff_targets", "--whole_tariff_actions"):
        if forbidden in command:
            raise ValueError(f"Revision 15 must not pass {forbidden}")
    if "--learning_corrections" not in command:
        raise ValueError("Revision 15 requires --learning_corrections")
    if option(command, "--rl_warm_start_target") != "0":
        raise ValueError("Revision 15 has no warm start")
    if option(command, "--manipulated_price_coefficients") != "5":
        raise ValueError("Revision 15 manages all five coefficients")
    if not Path(option(command, "--tariff_structure")).is_file():
        raise ValueError("tariff structure file is missing")
    if option(command, "--tariff_level_bounds") != LEVEL_BOUNDS:
        raise ValueError("Revision 15 level bounds are 0.6,2.0")
    imitation = float(option(command, "--branch_imitation_coeff")) if "--branch_imitation_coeff" in command else 0.0
    if imitation > 0.0 and int(option(command, "--nmcc_branches")) != ACTION_COUNT:
        raise ValueError("branch imitation requires NMCC over all five actions")


def r15_train_command(
    out: Path,
    days: int,
    *,
    oracle: bool,
    structure: Path,
    smoke: bool,
    nmcc_branches: int = ACTION_COUNT,
    nmcc_actor_weight: float = 0.8,
    imitation: float = 0.5,
    nmcc_dump: Path | None = None,
    seed: int = 251012,
) -> list[str]:
    """Revision 14's command with exactly the registered Revision 15 changes."""
    if imitation > 0.0 and nmcc_branches != ACTION_COUNT:
        raise ValueError("branch imitation requires nmcc_branches == 5")
    command = r14.r14_train_command(
        out,
        days,
        oracle=oracle,
        smoke=smoke,
        nmcc_branches=nmcc_branches,
        nmcc_actor_weight=nmcc_actor_weight,
        nmcc_policy_coeff=0.0,
        momentum=True,
        opponent=True,
        nmcc_dump=nmcc_dump,
        seed=seed,
        every_k=1,
    )
    remove_option(command, "--tariff_targets")
    r14.set_option(command, "--rl_warm_start_target", "0")
    r14.add_flag(command, "--level_step_actions")
    r14.set_option(command, "--tariff_structure", str(Path(structure).resolve()))
    r14.set_option(command, "--tariff_level_bounds", LEVEL_BOUNDS)
    r14.set_option(command, "--ppo_entropy_start", "0.01")
    r14.set_option(command, "--ppo_min_rollout_transitions", "30")
    r14.set_option(command, "--stage_profit_lead_weights", "0.3,0.3,0.3,0.3")
    r14.set_option(
        command,
        "--trained_model_id",
        f"r15-nyc-{'oracle' if oracle else 'public'}-{days}-s{seed}",
    )
    if nmcc_branches:
        r14.set_option(command, "--nmcc_stages", "static,catchup,profit")
        r14.set_option(command, "--branch_imitation_coeff", str(imitation))
        r14.set_option(command, "--branch_imitation_temperature_scale", "0.25")
    assert_level_contract(command)
    return command


def require_previous(root: Path, phase: str) -> None:
    index = PHASE_ORDER.index(phase)
    if index == 0:
        return
    previous = PHASE_ORDER[index - 1]
    path = root / previous / "status.json"
    if not path.exists():
        raise RuntimeError(f"Phase {phase} requires completed Phase {previous}: {path}")
    if not json.loads(path.read_text()).get("pass"):
        raise RuntimeError(f"Phase {phase} blocked by failed Phase {previous}")


def structure_path(root: Path) -> Path:
    path = root / "structure" / "tariff_structure.json"
    if not path.is_file():
        raise RuntimeError(f"missing tariff structure: {path}")
    return path.resolve()


def revision14_run_root() -> Path:
    pointer = BASE14 / "runs" / "latest_revision14_verify_path.txt"
    path = Path(pointer.read_text().strip())
    if not path.is_dir():
        raise FileNotFoundError(f"Revision 14 verify root not found: {path}")
    return path


# ------------------------------------------------------------------ phases
def phase_tests(root: Path) -> dict[str, Any]:
    out = root / "tests"
    focused = [str(PY), "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_revision15.py"]
    focused_code, focused_seconds = r14.run_process(focused, out / "focused", cwd=BASE, log_name="pytest.log")
    suite_out = out / "revision15"
    suite_out.mkdir(parents=True, exist_ok=True)
    xml_path = suite_out / "junit.xml"
    command = [str(PY), "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={xml_path}"]
    code, seconds = r14.run_process(command, suite_out, cwd=BASE, log_name="pytest.log")
    failures = r14.junit_failures(xml_path, code)
    baseline_xml = revision14_run_root() / "tests" / "revision14" / "junit.xml"
    if not baseline_xml.is_file():
        raise FileNotFoundError(f"Revision 14 suite result not found: {baseline_xml}")
    baseline = r14.junit_failures(baseline_xml, 0)
    new_failures = sorted(failures - baseline)
    interrupted = code in {-15, 130, 143}
    # Revision 15b repairs the inherited Revision 14 failures. Keep the baseline
    # comparison as a diagnostic, but never let an inherited failure pass this
    # gate: the focused tests and the complete suite must both be clean.
    strict_zero_failures = code == 0 and not failures
    return {
        "pass": focused_code == 0 and strict_zero_failures and not interrupted,
        "focused": {"exit_code": focused_code, "seconds": focused_seconds},
        "suite": {"exit_code": code, "seconds": seconds, "failures": sorted(failures)},
        "strict_zero_failures": strict_zero_failures,
        "revision14_baseline_failures": sorted(baseline),
        "new_revision15_failures": new_failures,
        "revision14_failures_now_passing": sorted(baseline - failures),
        "interrupted": interrupted,
    }


def phase_identity(root: Path) -> dict[str, Any]:
    out = root / "identity"
    command = [str(PY), str(HERE / "r15_identity_check.py"), "--output-root", str(out)]
    code, seconds = r14.run_process(command, out / "driver", log_name="identity.log")
    report_path = out / "identity_report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    return {"pass": code == 0 and bool(report.get("pass")), "seconds": seconds, "report": report}


def phase_isolation(root: Path) -> dict[str, Any]:
    out = root / "isolation"
    structure = write_uniform_structure(out / "uniform_structure.json")
    run_a, run_b = out / "with_nmcc", out / "without_nmcc"
    command_a = r15_train_command(run_a, 40, oracle=True, structure=structure, smoke=True,
                                  nmcc_actor_weight=0.0, imitation=0.0)
    command_b = r15_train_command(run_b, 40, oracle=True, structure=structure, smoke=True,
                                  nmcc_branches=0, imitation=0.0)
    r14.set_option(command_b, "--counterfactual_q_coeff", "0.05")
    code_b, seconds_b = r14.run_process(command_b, run_b)
    if code_b != 0:
        return {"pass": False, "error": "control run failed", "exit_codes": {"control": code_b}}
    code_a, seconds_a = r14.run_process(command_a, run_a)
    if code_a != 0:
        return {"pass": False, "error": "NMCC run failed", "exit_codes": {"nmcc": code_a, "control": code_b}}
    comparison = r14.compare_isolation_rows(run_a / "nyc_training_live.jsonl", run_b / "nyc_training_live.jsonl")
    rows = r14.read_jsonl(run_a / "nyc_training_live.jsonl")
    calls = int(rows[-1].get("nmcc_calls", 0)) if rows else 0
    total = float(rows[-1].get("nmcc_seconds_total", 0.0)) if rows else 0.0
    posted = [row.get("firm1_tariff_level") for row in rows]
    level_ok = bool(posted) and all(
        level is not None and 0.6 - 1e-9 <= float(level) <= 2.0 + 1e-9 for level in posted
    )
    levels = sorted({float(level) for level in posted if level is not None})
    return {
        "pass": comparison["pass"] and calls > 0 and level_ok,
        "exit_codes": {"nmcc": code_a, "control": code_b},
        "elapsed_seconds": {"nmcc": seconds_a, "control": seconds_b},
        "nmcc_calls": calls,
        "seconds_per_nmcc_call": total / calls if calls else None,
        "observed_levels": levels,
        "comparison": comparison,
    }


def phase_structure(root: Path, contract_policy: Path) -> dict[str, Any]:
    out = root / "structure"
    command = [str(PY), str(HERE / "r15_structure_search.py"), "--contract-policy", str(contract_policy),
               "--output-root", str(out), "--level-bounds", LEVEL_BOUNDS]
    code, seconds = r14.run_process(command, out / "driver", log_name="search.log")
    if code != 0:
        return {"pass": False, "exit_code": code, "seconds": seconds, "error": "structure search failed"}
    payload = json.loads((out / "tariff_structure.json").read_text())
    validated = validate_tariff_structure(payload, list(COEFFICIENTS), (0.6, 2.0))
    if validated != payload:
        return {"pass": False, "error": "structure file is not in validated form"}
    if abs(tariff_level_of(payload["structure"], city_anchors()) - 1.0) > 1e-6:
        return {"pass": False, "error": "structure anchors differ from the city calibration"}
    summary = json.loads((out / "search_summary.json").read_text())
    return {
        "pass": True,
        "seconds": seconds,
        "reference_level": payload["reference_level"],
        "structure": payload["structure"],
        "best_static_tariff": summary["best_static_tariff"],
        "level_line_search": summary["level_line_search"],
        "comparators": summary["comparators"],
    }


def phase_probe(root: Path) -> dict[str, Any]:
    out = root / "probe"
    run = out / "training"
    dump = out / "probe.jsonl"
    if dump.exists():
        dump.unlink()
    command = r15_train_command(run, 150, oracle=True, structure=structure_path(root), smoke=True,
                                nmcc_actor_weight=0.0, imitation=0.0, nmcc_dump=dump)
    code, seconds = r14.run_process(command, run)
    if code != 0:
        return {"pass": False, "exit_code": code, "seconds": seconds, "error": "probe training failed"}
    report_path = out / "probe_report.json"
    report_command = [str(PY), str(HERE / "r15_probe_report.py"), str(dump), "--output", str(report_path)]
    report_code, report_seconds = r14.run_process(report_command, out / "report_driver", log_name="probe_report.log")
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    return {
        "pass": report_code == 0 and bool(report.get("pass")),
        "exit_codes": {"training": code, "report": report_code},
        "seconds": {"training": seconds, "report": report_seconds},
        "report": report,
    }


def phase_smoke(root: Path) -> dict[str, Any]:
    out = root / "smoke"
    run = out / "training"
    days = 600
    command = r15_train_command(run, days, oracle=False, structure=structure_path(root), smoke=True)
    code, seconds = r14.run_process(command, run)
    if code != 0:
        return {"pass": False, "exit_code": code, "seconds": seconds, "error": "smoke training failed"}
    rows = r14.read_jsonl(run / "nyc_training_live.jsonl")
    ppo_rows = r14.read_jsonl(run / "nyc_ppo_updates.jsonl")
    nmcc_rows = [
        row for row in rows
        if row.get("training_stage_opponent_regime") in {"static", "catchup", "profit"}
        and row.get("ppo_update_performed")
    ]
    valid = [float(row.get("ppo_nmcc_valid_fraction", 0.0)) for row in nmcc_rows]
    imitation_rows = [int(row.get("ppo_branch_imitation_rows", 0)) for row in nmcc_rows]
    ratios = [
        float(row["ppo_branch_imitation_kl_after"]) / float(row["ppo_branch_imitation_kl_before"])
        for row in nmcc_rows
        if float(row.get("ppo_branch_imitation_kl_before", 0.0)) > 1e-9
    ]
    kls = [float(row["ppo_approx_kl"]) for row in ppo_rows if row.get("ppo_update_performed")]
    median_kl = st.median(kls) if kls else 0.0
    levels = [float(row["firm1_tariff_level"]) for row in rows if row.get("firm1_tariff_level") is not None]
    days_per_hour = days / max(seconds / 3600.0, 1e-9)
    gates = {
        "nmcc_valid_fraction": {
            "minimum": min(valid) if valid else None,
            "n_updates": len(valid),
            "pass": bool(valid) and min(valid) > 0.5,
        },
        "imitation_rows_every_update": {
            "minimum": min(imitation_rows) if imitation_rows else None,
            "pass": bool(imitation_rows) and min(imitation_rows) > 0,
        },
        "imitation_reduces_kl": {
            "median_after_over_before": st.median(ratios) if ratios else None,
            "n": len(ratios),
            "pass": bool(ratios) and st.median(ratios) < 1.0,
        },
        "median_kl": {"value": median_kl, "pass": median_kl > 0.005},
    }
    return {
        "pass": all(item["pass"] for item in gates.values()),
        "exit_code": code,
        "seconds": seconds,
        "days_per_hour": days_per_hour,
        "projected_4000_day_hours": TRAIN_DAYS / days_per_hour,
        "gates": gates,
        "diagnostics": {
            "greedy_action_by_regime_last_200_days": r14.chi_square_action_regime(rows[-200:]),
            "argmax_probability_mean_last_200_days": (
                st.mean(float(row["raw_policy_top_prob"]) for row in rows[-200:]) if rows else None
            ),
            "sampled_not_argmax_rate_last_200_days": (
                st.mean(float(row["sampled_not_raw_top_rate"]) for row in rows[-200:]) if rows else None
            ),
            "level_range": [min(levels), max(levels)] if levels else None,
        },
    }


def train_parallel(commands: dict[str, tuple[list[str], Path]], status_path: Path) -> dict[str, Any]:
    """Launch every named training command at once and wait for all of them."""
    jobs: dict[str, tuple[subprocess.Popen, Any, float, Path]] = {}
    trainings: dict[str, Any] = {}
    for name, (command, run) in commands.items():
        run.mkdir(parents=True, exist_ok=True)
        r14.atomic_json(run / "command.json", {"argv": command, "cwd": str(BASE)})
        log = (run / "training.log").open("w")
        process = subprocess.Popen(command, cwd=BASE, env=r12.env(), stdout=log, stderr=subprocess.STDOUT)
        jobs[name] = (process, log, time.perf_counter(), run)
        trainings[name] = {"state": "running", "pid": process.pid, "policy": str(run / "policy.pt")}
    r14.atomic_json(status_path, trainings)
    for name, (process, log, started, run) in jobs.items():
        code = process.wait()
        log.close()
        trainings[name] = {
            "state": "complete" if code == 0 else "failed",
            "exit_code": code,
            "seconds": time.perf_counter() - started,
            "policy": str(run / "policy.pt"),
        }
        r14.atomic_json(status_path, trainings)
    return trainings


def phase_train(root: Path) -> dict[str, Any]:
    out = root / "train"
    structure = structure_path(root)
    commands = {}
    for name, settings in TRAIN_VARIANTS.items():
        run = out / name / "training"
        commands[name] = (
            r15_train_command(run, TRAIN_DAYS, oracle=settings["oracle"], structure=structure,
                              smoke=False, imitation=settings["imitation"]),
            run,
        )
    trainings = train_parallel(commands, out / "training_status.json")
    failed = [name for name, item in trainings.items() if item["exit_code"] != 0]
    return {"pass": not failed, "trainings": trainings, "failed": failed}


def robustness_for_policy(eval_root: Path, best_spec: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for case in ROBUSTNESS_CASES:
        rl = r14.opening_rows(r14.payload(eval_root / case, "rl_actor"))
        best = r14.opening_rows(r14.payload(eval_root / case, best_spec))
        stats = r14.paired_reward_difference(rl, best, lambda row: True)
        stats["pass"] = bool(stats["mean"] is not None and stats["se"] is not None
                             and stats["mean"] >= -2.0 * stats["se"])
        result[case] = stats
    return result


def level_response(eval_root: Path) -> dict[str, Any]:
    """RL posted level on shortage minus calm days, paired over openings (A1)."""
    rows = r14.opening_rows(r14.payload(eval_root / "A1_shortage_severe", "rl_actor"))
    differences = []
    for opening in rows.values():
        shortage = [float(r["firm1_tariff_level"]) for r in opening
                    if "driver_shortage" in str(r.get("regime", "")) and "recovery" not in str(r.get("regime", ""))
                    and r.get("firm1_tariff_level") is not None]
        calm = [float(r["firm1_tariff_level"]) for r in opening
                if str(r.get("regime", "")) == "calm" and r.get("firm1_tariff_level") is not None]
        if shortage and calm:
            differences.append(st.mean(shortage) - st.mean(calm))
    return r14.mean_se_t(differences)


def comparator_specs(root: Path) -> tuple[str, list[str], float]:
    """Best static spec S * L_ref and every S * L comparator from the structure phase."""
    summary = json.loads((root / "structure" / "search_summary.json").read_text())
    reference = snap_tariff_level(summary["reference_level"])
    comparators = {snap_tariff_level(float(level)): path for level, path in summary["comparators"].items()}
    if reference not in comparators:
        raise RuntimeError("reference level comparator missing from the structure search")
    best_spec = f"static_tariff:@{comparators[reference]}"
    level_specs = [f"static_tariff:@{comparators[level]}" for level in sorted(comparators)]
    return best_spec, level_specs, reference


def evaluate_policy(run: Path, out: Path, best_spec: str, level_specs: list[str]) -> dict[str, Any]:
    """Held-out case evaluation of one trained policy and its acceptance statistics."""
    frozen_path = (out / "frozen_rl_tariff.json").resolve()
    frozen_path.parent.mkdir(parents=True, exist_ok=True)
    r14.write_frozen(frozen_path, run / "nyc_evaluation_diagnostics.csv")
    frozen_spec = f"static_tariff:@{frozen_path}"
    specs = ["rl_actor", *level_specs, frozen_spec]
    eval_root = out / "evaluation"
    for case in (*ACCEPTANCE_CASES, *ROBUSTNESS_CASES):
        case_out = eval_root / case
        case_out.mkdir(parents=True, exist_ok=True)
        run_specs("New York City", "nyc", case, specs, "eval", 10, "full", str(run / "policy.pt"),
                  str(case_out), customers=50, horizon_days=40,
                  rivals=["adaptive_best_response_memory"], pool=2000)
    acceptance = r14.acceptance_for_policy(eval_root, best_spec, frozen_spec)
    acceptance["robustness"] = robustness_for_policy(eval_root, best_spec)
    acceptance["level_shortage_minus_calm"] = level_response(eval_root)
    return acceptance


def gated(acceptance: dict[str, Any]) -> bool:
    return bool(acceptance["P1"]["pass"] and acceptance["P2"]["pass"] and acceptance["P3"]["pass"])


def phase_evaluate(root: Path) -> dict[str, Any]:
    out = root / "evaluate"
    best_spec, level_specs, reference = comparator_specs(root)
    acceptances: dict[str, Any] = {}
    for name in TRAIN_VARIANTS:
        acceptances[name] = evaluate_policy(root / "train" / name / "training", out / name, best_spec, level_specs)
        r14.atomic_json(out / "acceptance_partial.json", acceptances)
    passed = gated(acceptances["R15-B"])
    return {
        "pass": passed,
        "best_static_spec": best_spec,
        "reference_level": reference,
        "acceptance": acceptances,
        "imitation_attribution": {
            name: {"P1": acceptances[name]["P1"], "P2": acceptances[name]["P2"]}
            for name in ("R15-B", "R15-C")
        },
        "public_gated_P1_P2_P3_pass": passed,
    }


def phase_replicate(root: Path) -> dict[str, Any]:
    """R15-B settings on two further seeds; each must pass P1, P2 and P3."""
    out = root / "replicate"
    structure = structure_path(root)
    commands = {}
    for seed in REPLICATION_SEEDS:
        run = out / f"R15-B-s{seed}" / "training"
        commands[f"R15-B-s{seed}"] = (
            r15_train_command(run, TRAIN_DAYS, oracle=False, structure=structure, smoke=False,
                              imitation=0.5, seed=seed),
            run,
        )
    trainings = train_parallel(commands, out / "training_status.json")
    failed = [name for name, item in trainings.items() if item["exit_code"] != 0]
    if failed:
        return {"pass": False, "trainings": trainings, "failed": failed}
    best_spec, level_specs, reference = comparator_specs(root)
    acceptances = {}
    for name in commands:
        acceptances[name] = evaluate_policy(out / name / "training", out / name, best_spec, level_specs)
        r14.atomic_json(out / "acceptance_partial.json", acceptances)
    return {
        "pass": all(gated(item) for item in acceptances.values()),
        "trainings": trainings,
        "reference_level": reference,
        "acceptance": acceptances,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=PHASE_ORDER)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--contract-policy", type=Path, default=None,
                        help="Structure phase only: a Revision 14 policy whose saved city, supply "
                             "and demand contract matches training. Its actor is never queried.")
    args = parser.parse_args()
    root = args.run_root.resolve()
    phase_root = root / args.phase
    phase_root.mkdir(parents=True, exist_ok=True)
    r14.atomic_json(phase_root / "status.json",
                    {"phase": args.phase, "state": "running", "pass": False, "started_at": time.time()})
    try:
        require_previous(root, args.phase)
        if args.phase == "structure":
            if args.contract_policy is None or not args.contract_policy.is_file():
                raise RuntimeError("the structure phase requires an existing --contract-policy")
            result = phase_structure(root, args.contract_policy.resolve())
        else:
            if args.contract_policy is not None:
                raise RuntimeError("--contract-policy is valid only for the structure phase")
            result = globals()[f"phase_{args.phase}"](root)
        passed = bool(result.get("pass"))
        r14.atomic_json(phase_root / "status.json", {
            "phase": args.phase,
            "state": "complete" if passed else "failed",
            "pass": passed,
            "completed_at": time.time(),
            "result": result,
        })
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0 if passed else 2
    except BaseException as exc:
        r14.atomic_json(phase_root / "status.json", {
            "phase": args.phase,
            "state": "failed",
            "pass": False,
            "completed_at": time.time(),
            "error": repr(exc),
        })
        raise


if __name__ == "__main__":
    raise SystemExit(main())
