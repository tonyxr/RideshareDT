#!/usr/bin/env python3
"""Select, evaluate, validate, and report the registered V18 gated actor."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.v18_config import eval_args, load_config, validate_config
from scripts.v18_eval import R3, SCENARIOS, SUITES
from scripts.v18_runtime import (
    Job, append_progress, atomic_json, run_job, run_jobs,
)
from scripts.v18_stats import two_level_bootstrap


PYTHON = (REPO.parent / ".venv/bin/python").absolute()
PROGRESS = REPO / "runs/v18/controller/progress.jsonl"
CITIES = ("nyc", "boston", "chicago")
SEEDS = (101, 202, 303)
THETAS = (0.7, 0.8, 0.9)
SELECTION_CASES = ("C0", "A1")
SUITE_CASES = {
    "H1": ("E1", ("C0", "A1", "A3", "S1", "S6")),
    "H2": ("E2", ("i080", "i065", "i050", "i035")),
    "H5": ("E5", ("natural_year_2023",)),
}


def _set(values: Sequence[str], flag: str, value: Any) -> list[str]:
    result = list(map(str, values))
    while flag in result:
        index = result.index(flag)
        del result[index:index + 2]
    result.extend((str(flag), str(value)))
    return result


def _selected(city: str) -> dict[str, Any]:
    path = REPO / f"runs/v18/select/{city}/selected.json"
    return json.loads(path.read_text(encoding="utf-8"))


def gated_payload(city: str, theta_on: float) -> dict[str, Any]:
    selected = _selected(city)
    best = dict(selected["best_fixed"])
    count = str(selected["count_rule_1step"]["spec"])
    count_table = json.loads(count.split(":", 1)[1])
    return {
        "calm": [float(best["distance"]), float(best["time"])],
        "norm": [float(value) for value in count_table["norm"]],
        "theta_on": float(theta_on),
        "theta_off": float(min(1.0, round(float(theta_on) + 0.10, 12))),
        "off_days": 2,
    }


def gated_spec(city: str, theta_on: float) -> str:
    payload = json.dumps(
        gated_payload(city, theta_on), sort_keys=True, separators=(",", ":"),
    )
    return f"gated_rl_actor:{payload}"


def _policy_path(city: str, seed: int) -> Path:
    return REPO / f"runs/v18/train/{city}_s{seed}/policy.pt"


def equivalence_job(config: Mapping[str, Any]) -> tuple[Job, dict[str, str]]:
    selected = _selected("nyc")
    calm = gated_payload("nyc", 0.7)
    never = dict(calm, theta_on=-1.0, theta_off=-0.9)
    always = dict(calm, theta_on=99.0, theta_off=99.1)
    never_spec = "gated_rl_actor:" + json.dumps(
        never, sort_keys=True, separators=(",", ":"),
    )
    always_spec = "gated_rl_actor:" + json.dumps(
        always, sort_keys=True, separators=(",", ":"),
    )
    fixed_spec = str(selected["best_fixed"]["spec"])
    specs = (fixed_spec, never_spec, "rl_actor", always_spec)
    leaf = REPO / "runs/v18/controller/gated_code_check"
    values = eval_args(
        config, city="nyc", case="A1", specs=specs, suite="E1",
        output=leaf, seed=77311, scenario=SCENARIOS["A1"], kind="main",
        policy=_policy_path("nyc", 101), customers=150,
    )
    values = _set(values, "--policy_sweep_horizon", 40 * 6)
    values = _set(values, "--policy_sweep_openings", 1)
    values = _set(values, "--policy_sweep_rivals", R3[0])
    if "--recalibrate_supply_on_eval" not in values:
        values.append("--recalibrate_supply_on_eval")
    job = Job(
        "H_code_check/nyc/A1", (str(PYTHON), "-u", str(REPO / "Core.py"), *values),
        leaf / "raw.json", leaf / "run.log", leaf / "manifest.json", tier=0,
    )
    return job, {
        "fixed": fixed_spec, "never": never_spec,
        "actor": "rl_actor", "always": always_spec,
    }


def validate_equivalence(path: Path, specs: Mapping[str, str]) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = dict(payload["results"])
    traces = dict(payload["traces"])
    never_results_equal = results[specs["never"]] == results[specs["fixed"]]
    always_results_equal = results[specs["always"]] == results[specs["actor"]]
    never_traces_equal = traces[specs["never"]] == traces[specs["fixed"]]
    always_traces_equal = traces[specs["always"]] == traces[specs["actor"]]
    report = {
        "cell": "NYC/A1/adaptive_best_response_memory/opening_1",
        "days": 40,
        "never_open_matches_best_fixed": bool(
            never_results_equal and never_traces_equal
        ),
        "always_open_matches_rl_actor": bool(
            always_results_equal and always_traces_equal
        ),
        "results": {
            "never_open": never_results_equal,
            "always_open": always_results_equal,
        },
        "traces": {
            "never_open": never_traces_equal,
            "always_open": always_traces_equal,
        },
    }
    report["passed"] = bool(
        report["never_open_matches_best_fixed"]
        and report["always_open_matches_rl_actor"]
    )
    atomic_json(path.with_name("equivalence.json"), report)
    return report


def selection_jobs(config: Mapping[str, Any]) -> list[Job]:
    jobs: list[Job] = []
    for city in CITIES:
        for theta in THETAS:
            for case in SELECTION_CASES:
                leaf = REPO / (
                    f"runs/v18/select/{city}/gated_candidates/"
                    f"theta_{theta:.1f}/{case}"
                )
                values = eval_args(
                    config, city=city, case=case,
                    specs=(gated_spec(city, theta),), suite="E1",
                    output=leaf, seed=20231, scenario=SCENARIOS[case],
                    kind="main", policy=_policy_path(city, 101), customers=150,
                )
                values = _set(values, "--policy_sweep_horizon", 28 * 6)
                values = _set(values, "--policy_sweep_openings", 2)
                values = _set(values, "--policy_sweep_rivals", ",".join(R3))
                if "--recalibrate_supply_on_eval" not in values:
                    values.append("--recalibrate_supply_on_eval")
                jobs.append(Job(
                    f"H_select/{city}/theta_{theta:.1f}/{case}",
                    (str(PYTHON), "-u", str(REPO / "Core.py"), *values),
                    leaf / "raw.json", leaf / "run.log", leaf / "manifest.json",
                    tier=0,
                ))
    return jobs


def _only_spec(payload: Mapping[str, Any]) -> str:
    specs = tuple(dict(payload.get("results") or {}))
    if len(specs) != 1:
        raise ValueError(f"expected exactly one policy spec, found {specs}")
    return str(specs[0])


def select_gates(states: Mapping[str, str]) -> dict[str, str]:
    results: dict[str, str] = {}
    for city in CITIES:
        candidates: dict[str, Any] = {}
        for theta in THETAS:
            objectives: list[float] = []
            complete = True
            for case in SELECTION_CASES:
                name = f"H_select/{city}/theta_{theta:.1f}/{case}"
                path = REPO / (
                    f"runs/v18/select/{city}/gated_candidates/"
                    f"theta_{theta:.1f}/{case}/raw.json"
                )
                if states.get(name) not in {"complete", "cached"} or not path.is_file():
                    complete = False
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
                spec = _only_spec(payload)
                objectives.extend(
                    float(row["objective"])
                    for row in dict(payload["results"][spec]).values()
                )
            candidates[f"{theta:.1f}"] = {
                "theta_on": theta,
                "theta_off": min(1.0, round(theta + 0.10, 12)),
                "complete": complete,
                "mean_objective": (
                    float(np.mean(objectives)) if complete and objectives else None
                ),
                "rollouts": len(objectives),
            }
        eligible = [
            row for row in candidates.values()
            if row["complete"] and row["mean_objective"] is not None
        ]
        path = REPO / f"runs/v18/select/{city}/gated.json"
        if not eligible:
            atomic_json(path, {
                "state": "failed", "city": city, "seed": 20231,
                "candidates": candidates,
            })
            append_progress(
                PROGRESS, job=f"H_select/{city}", state="failed",
                attempt=3, seconds=0,
            )
            results[city] = "failed"
            continue
        chosen = sorted(
            eligible, key=lambda row: (-float(row["mean_objective"]), float(row["theta_on"]))
        )[0]
        theta = float(chosen["theta_on"])
        atomic_json(path, {
            "state": "complete", "city": city, "seed": 20231,
            "cases": list(SELECTION_CASES), "rivals": list(R3),
            "openings": 2, "days": 28, "agent": f"{city}_s101",
            "tie_break": "lower_theta_on", "candidates": candidates,
            "calm": gated_payload(city, theta)["calm"],
            "norm": gated_payload(city, theta)["norm"],
            "theta_on": theta, "theta_off": float(chosen["theta_off"]),
            "off_days": 2, "spec": gated_spec(city, theta),
        })
        append_progress(
            PROGRESS, job=f"H_select/{city}", state="complete",
            attempt=1, seconds=0,
        )
        results[city] = "complete"
    return results


def build_jobs(config: Mapping[str, Any]) -> list[Job]:
    jobs: list[Job] = []
    for h_suite, (e_suite, cases) in SUITE_CASES.items():
        definition = SUITES[e_suite]
        rivals = tuple(config["evaluation"]["rivals"][: int(definition["rivals"])])
        if int(definition["rivals"]) == 3:
            rivals = R3
        for city in CITIES:
            gated_path = REPO / f"runs/v18/select/{city}/gated.json"
            if not gated_path.is_file():
                continue
            gated = json.loads(gated_path.read_text(encoding="utf-8"))
            if gated.get("state") != "complete":
                continue
            spec = str(gated["spec"])
            for case in cases:
                for seed in SEEDS:
                    run_name = f"{city}_s{seed}"
                    leaf = REPO / f"runs/v18/eval/{h_suite}/{city}/{case}/{run_name}"
                    values = eval_args(
                        config, city=city, case=case, specs=(spec,), suite=e_suite,
                        output=leaf, seed=77311, scenario=SCENARIOS[case],
                        kind="main", policy=_policy_path(city, seed), customers=150,
                    )
                    values = _set(
                        values, "--policy_sweep_openings", definition["openings"]
                    )
                    values = _set(
                        values, "--policy_sweep_rivals", ",".join(rivals)
                    )
                    if "--recalibrate_supply_on_eval" not in values:
                        values.append("--recalibrate_supply_on_eval")
                    jobs.append(Job(
                        f"{h_suite}/{city}/{case}/{run_name}",
                        (str(PYTHON), "-u", str(REPO / "Core.py"), *values),
                        leaf / "raw.json", leaf / "run.log", leaf / "manifest.json",
                        tier=1,
                    ))
    return jobs


def _finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_finite(item) for item in value.values())
    if isinstance(value, Sequence):
        return all(_finite(item) for item in value)
    return True


def _supply_identity(payload: Mapping[str, Any]) -> dict[str, float | int]:
    v26r = dict(payload["v26r"])
    effective = dict(v26r["effective_supply_config"])
    return {
        "base_active_drivers": int(effective["base_active_drivers"]),
        "reference_earnings": float(v26r["supply_reference_earnings"]),
    }


def _reference_path(h_suite: str, city: str, case: str, run_name: str) -> Path:
    e_suite = SUITE_CASES[h_suite][0]
    return REPO / f"runs/v18/eval/{e_suite}/{city}/{case}/{run_name}/raw.json"


def validate_products(jobs: Iterable[Job], states: Mapping[str, str]) -> dict[str, Any]:
    validations: dict[str, Any] = {}
    for job in jobs:
        if states.get(job.name) not in {"complete", "cached"}:
            validations[job.name] = {"state": states.get(job.name, "missing")}
            continue
        h_suite, city, case, run_name = job.name.split("/", 3)
        payload = json.loads(job.output.read_text(encoding="utf-8"))
        reference_path = _reference_path(h_suite, city, case, run_name)
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        actual_supply = _supply_identity(payload)
        reference_supply = _supply_identity(reference)
        spec = _only_spec(payload)
        expected_rollouts = int(SUITES[SUITE_CASES[h_suite][0]]["rivals"]) * int(
            SUITES[SUITE_CASES[h_suite][0]]["openings"]
        )
        observed_rollouts = len(payload["results"][spec])
        trace_rows = len(payload["traces"][spec])
        expected_trace_rows = int(payload["horizon"]) * expected_rollouts
        result = {
            "state": "validated",
            "finite": _finite(payload),
            "expected_rollouts": expected_rollouts,
            "observed_rollouts": observed_rollouts,
            "expected_trace_rows": expected_trace_rows,
            "observed_trace_rows": trace_rows,
            "base_active_drivers": actual_supply["base_active_drivers"],
            "reference_earnings": actual_supply["reference_earnings"],
            "reference_product": str(reference_path),
            "reference_identity": reference_supply,
        }
        result["identity_match"] = bool(actual_supply == reference_supply)
        result["passed"] = bool(
            result["finite"] and observed_rollouts == expected_rollouts
            and trace_rows == expected_trace_rows and result["identity_match"]
        )
        atomic_json(job.output.with_name("validation.json"), result)
        validations[job.name] = result
    return validations


def _payload(path: Path) -> tuple[dict[str, Any], str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value, _only_spec(value)


def _result_objectives(payload: Mapping[str, Any], spec: str) -> dict[str, float]:
    return {
        str(name): float(row["objective"])
        for name, row in dict(payload["results"][spec]).items()
    }


def _trace_window_means(
    payload: Mapping[str, Any], spec: str, window: str,
) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in payload["traces"][spec]:
        event = bool(row.get("shock_active"))
        if window == "event_days" and not event:
            continue
        if window == "calm_days" and event:
            continue
        grouped.setdefault(str(row["rollout"]), []).append(float(row["reward"]))
    return {
        rollout: float(np.mean(values))
        for rollout, values in grouped.items() if values
    }


def _gate_share(payload: Mapping[str, Any], spec: str, event: bool) -> float | None:
    days: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    steps_per_day = int(payload["steps_per_day"])
    for row in payload["traces"][spec]:
        is_event = bool(row.get("shock_active"))
        if is_event != bool(event):
            continue
        key = (str(row["rollout"]), int(row["step"]) // steps_per_day)
        days.setdefault(key, []).append(row)
    if not days:
        return None
    return float(np.mean([
        float(rows[0]["gate_open"]) for rows in days.values()
    ]))


def _bootstrap_or_missing(rows: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    usable = {key: list(values) for key, values in rows.items() if values}
    if len(usable) != len(SEEDS):
        return {"available": False, "reason": "window has no paired cells for every seed"}
    return {"available": True, **two_level_bootstrap(usable)}


def report_results(
    jobs: Iterable[Job], states: Mapping[str, str], validations: Mapping[str, Any],
) -> dict[str, Any]:
    job_map = {job.name: job for job in jobs}
    report: dict[str, Any] = {
        "state": "complete", "seed": 77311,
        "bootstrap": "seed-level outer, controller/opening inner",
        "suites": {}, "validation": dict(validations),
        "job_states": dict(states),
    }
    for h_suite, (_, cases) in SUITE_CASES.items():
        suite_report: dict[str, Any] = {}
        for city in CITIES:
            city_report: dict[str, Any] = {}
            for case in cases:
                gated_rows: dict[int, tuple[dict[str, Any], str]] = {}
                for seed in SEEDS:
                    name = f"{h_suite}/{city}/{case}/{city}_s{seed}"
                    job = job_map.get(name)
                    if job is not None and states.get(name) in {"complete", "cached"}:
                        gated_rows[seed] = _payload(job.output)
                if len(gated_rows) != len(SEEDS):
                    city_report[case] = {
                        "state": "incomplete", "available_seeds": sorted(gated_rows),
                    }
                    continue
                e_suite = SUITE_CASES[h_suite][0]
                fixed = _payload(REPO / f"runs/v18/eval/{e_suite}/{city}/{case}/best_fixed/raw.json")
                count = _payload(REPO / f"runs/v18/eval/{e_suite}/{city}/{case}/count_rule_1step/raw.json")
                comparisons: dict[str, Any] = {}
                for window in ("whole_run", "event_days", "calm_days"):
                    paired: dict[str, dict[str, list[float]]] = {
                        "gated_minus_best_fixed": {},
                        "gated_minus_agent": {},
                        "gated_minus_count_rule_1step": {},
                    }
                    fixed_values = (
                        _result_objectives(*fixed)
                        if window == "whole_run" else _trace_window_means(*fixed, window)
                    )
                    count_values = (
                        _result_objectives(*count)
                        if window == "whole_run" else _trace_window_means(*count, window)
                    )
                    for seed, (gated_payload_value, gated_policy) in gated_rows.items():
                        run_name = f"{city}_s{seed}"
                        agent = _payload(
                            REPO / f"runs/v18/eval/{e_suite}/{city}/{case}/{run_name}/raw.json"
                        )
                        gated_values = (
                            _result_objectives(gated_payload_value, gated_policy)
                            if window == "whole_run" else _trace_window_means(
                                gated_payload_value, gated_policy, window
                            )
                        )
                        agent_values = (
                            _result_objectives(*agent)
                            if window == "whole_run" else _trace_window_means(*agent, window)
                        )
                        common = sorted(
                            set(gated_values) & set(fixed_values)
                            & set(agent_values) & set(count_values)
                        )
                        key = str(seed)
                        paired["gated_minus_best_fixed"][key] = [
                            gated_values[name] - fixed_values[name] for name in common
                        ]
                        paired["gated_minus_agent"][key] = [
                            gated_values[name] - agent_values[name] for name in common
                        ]
                        paired["gated_minus_count_rule_1step"][key] = [
                            gated_values[name] - count_values[name] for name in common
                        ]
                    comparisons[window] = {
                        name: _bootstrap_or_missing(values)
                        for name, values in paired.items()
                    }
                calm_shares = [
                    _gate_share(payload, spec, False)
                    for payload, spec in gated_rows.values()
                ]
                event_shares = [
                    _gate_share(payload, spec, True)
                    for payload, spec in gated_rows.values()
                ]
                cell = {
                    "state": "complete", "comparisons": comparisons,
                    "gate_open_share": {
                        "calm_days": (
                            float(np.mean([value for value in calm_shares if value is not None]))
                            if any(value is not None for value in calm_shares) else None
                        ),
                        "event_days": (
                            float(np.mean([value for value in event_shares if value is not None]))
                            if any(value is not None for value in event_shares) else None
                        ),
                    },
                }
                if h_suite == "H5":
                    cell["annual_mean"] = comparisons["whole_run"]
                city_report[case] = cell
            suite_report[city] = city_report
        report["suites"][h_suite] = suite_report
    return report


def plan(config: Mapping[str, Any]) -> dict[str, Any]:
    selected_jobs = selection_jobs(config)
    # Build counts without requiring gated.json products.
    counts = {"H1": 45, "H2": 36, "H5": 9}
    return {
        "name": "gated_agent", "evaluation_only": True, "training": False,
        "code_check": {
            "cell": "NYC A1", "rival": R3[0], "openings": 1, "days": 40,
            "checks": ["never_open_equals_best_fixed", "always_open_equals_rl_actor"],
        },
        "selection": {
            "jobs": len(selected_jobs), "cities": list(CITIES),
            "theta_on": list(THETAS), "cases": list(SELECTION_CASES),
            "seed": 20231, "rivals": list(R3), "openings": 2,
            "days": 28, "customers": 150,
        },
        "evaluation": {
            "counts": counts, "total_jobs": sum(counts.values()),
            "seed": 77311, "recalibrate_supply_on_eval": True,
        },
        "queue": "after fixes 1-7 reruns and before S",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config = load_config()
    validate_config(config, require_frozen=True)
    plan_payload = plan(config)
    atomic_json(REPO / "runs/v18/eval/H_plan.json", plan_payload)
    if not args.execute:
        print(json.dumps(plan_payload, indent=2, sort_keys=True))
        return 0

    code_job, specs = equivalence_job(config)
    code_state = run_job(
        code_job, cwd=REPO, progress=PROGRESS, max_attempts=3,
    )
    if code_state not in {"complete", "cached"}:
        raise RuntimeError("gated actor equivalence code check did not complete")
    equivalence = validate_equivalence(code_job.output, specs)
    if not equivalence["passed"]:
        raise RuntimeError(f"gated actor equivalence failed: {equivalence}")

    select_states = run_jobs(
        selection_jobs(config), cwd=REPO, workers=args.workers, progress=PROGRESS,
    )
    city_selection = select_gates(select_states)
    jobs = build_jobs(config)
    states = run_jobs(
        jobs, cwd=REPO, workers=args.workers, progress=PROGRESS,
    )
    for city, state in city_selection.items():
        if state != "complete":
            for h_suite, (_, cases) in SUITE_CASES.items():
                for case in cases:
                    for seed in SEEDS:
                        name = f"{h_suite}/{city}/{case}/{city}_s{seed}"
                        states[name] = "skipped"
                        append_progress(
                            PROGRESS, job=name, state="skipped",
                            attempt=0, seconds=0,
                        )
    validations = validate_products(jobs, states)
    report = report_results(jobs, states, validations)
    report["equivalence"] = equivalence
    report["selection"] = city_selection
    report["counts"] = {
        state: sum(value == state for value in states.values())
        for state in ("complete", "cached", "failed", "skipped")
    }
    atomic_json(REPO / "runs/v18/eval/H_report.json", report)
    append_progress(
        PROGRESS, job="H_report", state="complete", attempt=1, seconds=0,
    )
    print(json.dumps(report["counts"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
