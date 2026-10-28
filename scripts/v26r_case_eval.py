#!/usr/bin/env python3
"""Run the corner-case evaluation sweeps (resumable, cached per spec).

Usage:
  scripts/v26r_case_eval.py <City> <slug> <case|all> [--seed-set eval|val|both] [--openings 8]
                            [--policy runs/.../policy.pt] [--budget-seconds 150] [--channels full|regime]
                            [--chunk 3] [--stage static|oracle|all]

City is the market name ("Boston", "Chicago", "New York City"); slug the data folder (boston, chicago, nyc).
Each spec's results and traces are cached in runs/cases/<slug>/<case>/<seed-set>/<md5>.json, so the
script can be called repeatedly inside a time budget (VM calls are limited to 180 s) and resumes where
it stopped.  Stage "oracle" needs the static stage of the same seed set to pick the calm target.
Merged output per case and seed set: runs/cases/<slug>/<case>/<seed-set>/sweep.json
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, BASE)
from diagnostic_logging import atomic_json
from v26r_cases import CASES, STATIC_GRID, BURNIN_DAYS, HORIZON_DAYS, static_specs, oracle_specs, myopic_specs, required_regime_labels  # noqa: E402
from evaluation_selection import select_constant

SEEDS = {"eval": 99173, "val": 20231}
CHANNEL_FLAGS = {
    "full": ["--spillover_price_aware", "--supply_model", "v2", "--driver_pay_basis", "fare",
             "--loyalty_model", "v2", "--loyalty_pool_size", "360", "--persistence_features"],
    "regime": ["--spillover_price_aware"],
}
RIVAL = {"full": "adaptive_best_response_memory", "regime": "adaptive_best_response_aggressive"}


def repo_root() -> str:
    for candidate in ("/Users/huali/Desktop/RideResponse", os.path.dirname(BASE)):
        if os.path.isdir(os.path.join(candidate, "data", "processed")):
            return candidate
    return os.path.dirname(BASE)


def python_cmd():
    override = os.environ.get("V26R_PY")
    if override:
        return [override, "-u"], {}
    venv = os.path.join(repo_root(), ".venv", "bin", "python")
    if os.path.exists(venv):
        return [venv, "-u"], {}
    raise FileNotFoundError('Real PyTorch runtime required: install the repository .venv or set V26R_PY. Stub evaluation is unsupported.')


def spec_key(spec: str) -> str:
    return hashlib.md5(spec.encode("utf-8")).hexdigest()[:12]


def spec_reference_paths(specs: str) -> list[Path]:
    """Files embedded as ``policy_kind:@path`` in an explicit sweep."""
    paths: list[Path] = []
    for spec in str(specs or "").split(";"):
        payload = spec.partition(":")[2].strip()
        if payload.startswith("@"):
            paths.append(Path(os.path.expanduser(payload[1:])).resolve())
    return paths


def run_specs(city: str, slug: str, case_name: str, specs, seed_set: str, openings: int, channels: str,
              policy: str | None, out_dir: str, *, customers=100, horizon_days=HORIZON_DAYS, rivals=None, pool=2000) -> None:
    case = CASES[case_name]
    root = repo_root()
    py, env = python_cmd()
    tmp_dir = tempfile.TemporaryDirectory(prefix="v26r-case-")
    tmp_out = os.path.join(tmp_dir.name, "sweep.json")
    channel_flags = list(CHANNEL_FLAGS[channels])
    if "--loyalty_pool_size" in channel_flags:
        channel_flags[channel_flags.index("--loyalty_pool_size") + 1] = str(customers * 6 * 6 // 10)
    cmd = py + [os.path.join(BASE, "Core.py"), "--market", city, "--choice_mode", "cognitive",
                "--firm1_mode", "RL", "--firm2_mode", RIVAL[channels], "--run_experiment",
                "--credit_assignment_v25", "--v26", "--deterministic_experiment_seed", "--seed", "4500",
                "--experiment_seed", "4500", "--policy_objective", "profit_maximization", "--pool", str(pool),
                "--threshold_profile_source", "cached",
                "--threshold_cache_path", os.path.join(root, "artifacts", "v26_confirmatory", "thresholds", slug, "llm_numeric_primary.jsonl"),
                "--calibration_json", os.path.join(root, "data", "processed", slug, "calibration.json"),
                "--gpt_threshold_mode", "llm_numeric", "--train_timesteps", "0", "--train_customers", str(customers),
                "--train_steps_per_day", "6", "--eval_customers", str(customers), "--firm1_action_interval_steps", "6",
                "--firm2_action_interval_steps", "6", "--no-live_diagnostics", "--regime_market", "--regime_features",
                "--shock_scenario", os.path.join(BASE, case["scenario"]),
                "--tariff_targets", ",".join(map(str, STATIC_GRID)),
                "--policy_sweep", ";".join(specs), "--policy_sweep_out", tmp_out,
                "--policy_sweep_openings", str(openings), "--eval_timesteps", str(horizon_days * 6),
                "--policy_sweep_seed", str(SEEDS[seed_set]), "--policy_sweep_rivals", ",".join(rivals or [RIVAL[channels]]),
                "--report_prefix", os.path.relpath(os.path.join(out_dir, "logs", str(time.time_ns())), BASE)] + channel_flags
    if policy:
        # Core restores learner, objective and action contracts from the policy.
        # A comparison harness must not silently impose its current defaults.
        grid_index = cmd.index("--tariff_targets")
        del cmd[grid_index:grid_index + 2]
        cmd += ["--trained_model_in", policy, "--eval_only"]
    else:
        cmd += ["--learning_corrections", "--learner_revision", "4", "--stage_specific_rewards"]
    full_env = dict(os.environ); full_env.update(env)
    full_env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", MPLBACKEND="Agg")
    proc = subprocess.run(cmd, cwd=BASE, env=full_env, capture_output=True, text=True)
    Path(out_dir, "chunk_" + spec_key(";".join(specs)) + ".log").write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0 or not os.path.exists(tmp_out):
        sys.stderr.write(proc.stdout[-3000:] + proc.stderr[-3000:])
        raise SystemExit(f"sweep failed for {case_name} {specs}")
    data = json.load(open(tmp_out))
    tmp_dir.cleanup()
    for spec in specs:
        payload = {"spec": spec, "case": case_name, "seed_set": seed_set, "openings": openings,
                   "horizon_steps": data["horizon"], "steps_per_day": data["steps_per_day"],
                   "rollouts": data["rollouts"], "v26r": data.get("v26r", {}),
                   "results": data["results"][spec], "traces": data["traces"][spec]}
        atomic_json(os.path.join(out_dir, spec_key(spec) + ".json"), payload)


def ensure_manifest(out_dir, identity):
    """Cache protection, never policy selection: reject incompatible evidence."""
    path = Path(out_dir, "manifest.json")
    if path.exists():
        if json.loads(path.read_text()) != identity:
            raise ValueError(f"Incompatible case cache: {path}; use a fresh output root")
    elif any(Path(out_dir).glob("*.json")):
        raise ValueError(f"Unversioned case cache: {out_dir}; use a fresh output root")
    else:
        atomic_json(path, identity)


def merge(out_dir: str) -> dict:
    merged = {"results": {}, "traces": {}, "specs": {}}
    for name in sorted(os.listdir(out_dir)):
        if not name.endswith(".json") or len(name) != 17:
            continue
        d = json.load(open(os.path.join(out_dir, name)))
        merged["results"][d["spec"]] = d["results"]
        merged["traces"][d["spec"]] = d["traces"]
        merged["specs"][d["spec"]] = {k: d[k] for k in ("case", "seed_set", "openings", "horizon_steps", "steps_per_day")}
        merged.setdefault("meta", {k: d[k] for k in ("case", "seed_set", "openings", "horizon_steps", "steps_per_day", "rollouts")})
    atomic_json(os.path.join(out_dir, "sweep.json"), merged)
    return merged


def best_static_from(out_dir: str, burnin: int, *, rival=None, end=None, column="reward"):
    """Validation-only, rival-specific, explicit window and objective selection."""
    traces = {}
    spd = None
    for name in sorted(os.listdir(out_dir)):
        if not name.endswith(".json") or len(name) != 17:
            continue
        d = json.load(open(os.path.join(out_dir, name)))
        if not d["spec"].startswith("static:"):
            continue
        if d['seed_set'] != 'val':
            raise ValueError('Static anchors must be selected on validation data')
        if spd is not None and spd != int(d['steps_per_day']):
            raise ValueError('Inconsistent simulator cadence')
        spd = int(d['steps_per_day'])
        horizon = int(d['horizon_steps']) / spd
        traces[d['spec']] = d['traces']
    if not traces:
        raise ValueError('Missing static validation evidence')
    spec = select_constant(traces, start=burnin, end=horizon if end is None else end,
                           spd=spd, column=column, rival=rival)
    return float(spec.split(':')[1])


def best_regime_targets_from(out_dir, *, rival, burnin):
    """Independent per-label initial targets, selected only from validation."""
    from collections import defaultdict
    from statistics import mean
    scores = defaultdict(dict)
    for spec in static_specs():
        data = json.loads(Path(out_dir, spec_key(spec)+'.json').read_text())
        if data['seed_set'] != 'val':
            raise ValueError('regime targets require validation evidence')
        grouped = defaultdict(lambda: defaultdict(list))
        for row in data['traces']:
            # These are candidate initializers, NOT the final oracle ranking.
            # Include pre-burn-in labels: they influence the scored future.
            if row['rollout'].rsplit('__opening_', 1)[0] == rival:
                grouped[row['regime']][row['rollout']].append(float(row['reward']))
        for label, openings in grouped.items():
            scores[label][float(spec.split(':')[1])] = mean(mean(v) for v in openings.values())
    if not scores or any(set(values) != set(STATIC_GRID) for values in scores.values()):
        raise ValueError('incomplete regime-label static validation evidence')
    return {label: min(values, key=lambda target: (-values[target], target))
            for label, values in scores.items() if label != 'calm'}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("city"); ap.add_argument("slug"); ap.add_argument("case")
    ap.add_argument("--seed-set", default="eval", choices=["eval", "val", "both"])
    ap.add_argument("--openings", type=int, default=8)
    ap.add_argument("--policy", default=None)
    ap.add_argument("--budget-seconds", type=float, default=150.0)
    ap.add_argument("--channels", default="full", choices=["full", "regime"])
    ap.add_argument("--chunk", type=int, default=3)
    ap.add_argument("--stage", default="all", choices=["static", "oracle", "all"])
    ap.add_argument("--output-root", default=os.path.join(BASE, "runs", "cases"))
    ap.add_argument("--customers", type=int, default=100)
    ap.add_argument("--pool", type=int, default=2000)
    ap.add_argument("--horizon-days", type=int, default=HORIZON_DAYS)
    ap.add_argument("--rivals", default="")
    ap.add_argument("--specs", default="", help="Explicit subset for integration tests; recorded in manifest")
    ap.add_argument(
        "--workers",
        type=int,
        default=3,
        help=(
            "Independent case workers used when case=all. Each worker owns a "
            "separate case cache, so parallelism does not alter rollout seeds "
            "or policy state (default: 3)."
        ),
    )
    args = ap.parse_args()

    cases = (
        list(CASES)
        if args.case == "all"
        else [name.strip() for name in args.case.split(",") if name.strip()]
    )
    unknown_cases = [name for name in cases if name not in CASES]
    if not cases or unknown_cases:
        ap.error(
            "case must be 'all', one configured case, or a comma-separated "
            f"configured subset; unknown={unknown_cases}"
        )
    seed_sets = ["val", "eval"] if args.seed_set == "both" else [args.seed_set]
    start = time.time()
    pending = 0
    if args.openings < 1 or args.customers < 1 or args.chunk < 1 or args.workers < 1:
        ap.error("openings, customers, chunk and workers must be positive")

    # Cases have disjoint cache directories and deterministic seed contracts.
    # Fan out only at this outer boundary: a single case remains sequential, and
    # every Core subprocess still uses one BLAS/OpenMP thread. This consumes
    # spare cores without changing action ordering, CRNs, or trained policies.
    if len(cases) > 1 and args.workers > 1:
        tail = list(sys.argv[4:])
        while "--workers" in tail:
            index = tail.index("--workers")
            del tail[index:index + 2]

        def run_case(case_name: str) -> tuple[str, int]:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                args.city,
                args.slug,
                case_name,
                *tail,
                "--workers",
                "1",
            ]
            return case_name, subprocess.run(command).returncode

        failures = []
        worker_count = min(int(args.workers), len(cases))
        print(
            f"[case-eval] parallel case execution: {worker_count} workers "
            f"for {len(cases)} cases",
            flush=True,
        )
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=worker_count
        ) as executor:
            futures = [executor.submit(run_case, case_name) for case_name in cases]
            for future in concurrent.futures.as_completed(futures):
                case_name, returncode = future.result()
                print(
                    f"[case-eval] case={case_name} exit={returncode}",
                    flush=True,
                )
                if returncode:
                    failures.append((case_name, returncode))
        if failures:
            raise SystemExit(
                "parallel case evaluation failed: "
                + ", ".join(
                    f"{case_name}={returncode}"
                    for case_name, returncode in failures
                )
            )
        print("[case-eval] complete", flush=True)
        return

    rivals = [r for r in args.rivals.split(",") if r] or [RIVAL[args.channels]]
    files = list(Path(BASE).glob("*.py")) + [Path(__file__), Path(HERE, "v26r_cases.py")]
    root = Path(repo_root())
    files += [root/f"artifacts/v26_confirmatory/thresholds/{args.slug}/llm_numeric_primary.jsonl",
              root/f"data/processed/{args.slug}/calibration.json", root/f"data/processed/{args.slug}/trip_bank.csv"]
    if args.policy:
        args.policy = str(Path(args.policy).resolve())
        files.append(Path(args.policy))
    files += spec_reference_paths(args.specs)
    missing_files = [str(path) for path in files if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(
            f"evaluation identity inputs are missing: {missing_files}"
        )
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    for case_name in cases:
        case = CASES[case_name]
        labels = required_regime_labels(case, args.horizon_days,
            json.loads((root/f'data/processed/{args.slug}/calibration.json').read_text()))
        burnin = int(case.get("burnin", BURNIN_DAYS))
        for seed_set in seed_sets:
            if args.horizon_days <= burnin:
                raise ValueError("Case horizon must extend beyond burn-in")
            out_dir = os.path.join(args.output_root, args.slug, case_name, seed_set)
            os.makedirs(out_dir, exist_ok=True)
            identity = dict(version=2, city=args.city, slug=args.slug, case=case_name, seed_set=seed_set,
                            seed=SEEDS[seed_set], openings=args.openings, customers=args.customers,
                            horizon_days=args.horizon_days, rivals=rivals, channels=args.channels, pool=args.pool,
                            policy=args.policy, specs_subset=args.specs, hashes=hashes, oracle_label_support=labels,
                            scenario_sha256=hashlib.sha256(Path(BASE, case["scenario"]).read_bytes()).hexdigest())
            ensure_manifest(out_dir, identity)
            phases = ["subset"] if args.specs else (["static", "oracle"] if args.stage == "all" else [args.stage])
            for phase in phases:
                if phase == "subset":
                    specs = [s for s in args.specs.split(";") if s]
                elif phase == "static":
                    specs = static_specs()
                else:
                    selection_dir = os.path.join(args.output_root, args.slug, case_name, "val")
                    if not all(Path(selection_dir, spec_key(s)+".json").exists() for s in static_specs()):
                        raise ValueError("Oracle/schedule anchors require complete validation static sweeps; run --seed-set both")
                    targets = {r: best_static_from(selection_dir, burnin, rival=r, end=case['onset'])
                               for r in rivals}
                    specs = list(dict.fromkeys(s for rival, target in targets.items()
                        for s in oracle_specs(case, target,
                            best_regime_targets_from(selection_dir, rival=rival, burnin=burnin), required_labels=labels)
                            + list(myopic_specs(case, target).values())))
                    specs += ["myopic_greedy"]
                    if args.policy:
                        specs += ["rl_actor"]
                todo = [s for s in specs if not Path(out_dir, spec_key(s)+".json").exists()]
                while todo:
                    if time.time() - start > args.budget_seconds:
                        merge(out_dir)
                        print("[case-eval] INCOMPLETE: time budget reached; cached work preserved", flush=True)
                        raise SystemExit(3)
                    chunk, todo = todo[:args.chunk], todo[args.chunk:]
                    print(f"[case-eval] {case_name}/{seed_set}/{phase}: {len(chunk)} spec(s), {len(todo)} left", flush=True)
                    # Load the SAME artifact for every spec, not only RL: this
                    # preserves its population, normalization and reward contract.
                    run_specs(args.city, args.slug, case_name, chunk, seed_set, args.openings, args.channels,
                              args.policy, out_dir, customers=args.customers, horizon_days=args.horizon_days, rivals=rivals, pool=args.pool)
            merge(out_dir)
    print("[case-eval] complete")


if __name__ == "__main__":
    main()
