"""Queue independent city/seed pipelines with bounded parallelism and no retries."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "scripts"))
from diagnostic_logging import atomic_json
from run_v26r_multicity import CITIES, PYTHON


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--city-calibration-root", required=True, type=Path)
    ap.add_argument("--training-scenario-root", required=True, type=Path)
    ap.add_argument("--seeds", default="251012,251013,251014")
    ap.add_argument("--days", type=int, default=20000)
    ap.add_argument("--max-workers", type=int, default=3)
    ap.add_argument("--update-days", type=int, default=5)
    ap.add_argument("--rollout-transitions", type=int, default=180)
    ap.add_argument("--lead-weight", type=float, default=.6)
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()
    seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())
    if len(seeds) != 3 or len(set(seeds)) != 3:
        ap.error("exactly three distinct seeds are required")
    if args.days < 1 or args.max_workers < 1:
        ap.error("days and max-workers must be positive")
    for slug in CITIES:
        for path in (args.city_calibration_root / slug / "city_calibration.json",
                     args.training_scenario_root / f"{slug}.json"):
            if not path.is_file():
                ap.error(f"missing required input: {path}")

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    jobs = [(seed, slug) for seed in seeds for slug in CITIES]
    atomic_json(out / "manifest.json", {
        "format": "ride-response-three-seed-multicity-v1",
        "seeds": seeds,
        "cities": CITIES,
        "days_per_policy": args.days,
        "jobs": len(jobs),
        "max_workers": args.max_workers,
        "policy": "final training policy only; no checkpoint fallback or automatic retry",
        "curriculum_root": str(args.training_scenario_root.resolve()),
        "calibration_root": str(args.city_calibration_root.resolve()),
    })
    active, results = {}, {}
    stopping = False

    def stop(signum, _frame):
        nonlocal stopping
        stopping = True
        atomic_json(out / "termination_signal.json", {"signal": signum, "time": time.time()})
        for process, _log in active.values():
            process.send_signal(signum)

    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        pending = list(jobs)
        while (pending or active) and not stopping:
            while pending and len(active) < args.max_workers:
                seed, slug = pending.pop(0)
                job_out = out / f"seed_{seed}" / slug
                job_out.parent.mkdir(parents=True, exist_ok=True)
                cmd = [str(PYTHON), "-u", str(BASE / "scripts/run_v26r_multicity.py"),
                       "--output", str(job_out), "--city", slug, "--seed", str(seed),
                       "--days", str(args.days), "--update-days", str(args.update_days),
                       "--rollout-transitions", str(args.rollout_transitions),
                       "--lead-weight", str(args.lead_weight), "--learner-revision", "4",
                       "--ppo-rollout-episodes", "2", "--city-calibration-root",
                       str(args.city_calibration_root.resolve()), "--training-scenario-root",
                       str(args.training_scenario_root.resolve())]
                if args.figures:
                    cmd.append("--figures")
                job_out.mkdir()
                atomic_json(job_out / "launcher_command.json", cmd)
                log = (job_out / "pipeline.log").open("x")
                env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                           MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg")
                env.pop("PYTHONPATH", None)
                proc = subprocess.Popen(cmd, cwd=BASE, stdout=log, stderr=subprocess.STDOUT,
                                        start_new_session=True, env=env)
                active[(seed, slug)] = (proc, log)
            for key, (proc, log) in list(active.items()):
                code = proc.poll()
                if code is None:
                    continue
                log.close()
                results[f"{key[0]}:{key[1]}"] = code
                del active[key]
            atomic_json(out / "status.json", {"phase": "running" if pending or active else "complete",
                        "time": time.time(), "pending": [f"{s}:{c}" for s, c in pending],
                        "active": {f"{s}:{c}": p.pid for (s, c), (p, _l) in active.items()},
                        "exit_codes": results})
            if pending or active:
                time.sleep(30)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        for proc, log in active.values():
            if proc.poll() is None:
                proc.wait()
            log.close()
    if stopping:
        raise SystemExit(143)
    if len(results) != len(jobs) or any(code != 0 for code in results.values()):
        raise RuntimeError(f"one or more independent jobs failed: {results}; no retry was attempted")


if __name__ == "__main__":
    main()
