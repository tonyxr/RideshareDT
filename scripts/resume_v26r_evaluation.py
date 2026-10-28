"""Evaluate the preserved final policy only. No training, retries or policy selection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from diagnostic_logging import atomic_json
from scripts.run_v26r_staged import environment, source_hashes, summarize


def evaluation_command(source_run, output):
    command = json.loads((source_run / "command.json").read_text())
    replacements = {"--train_timesteps": "0", "--rl_warm_start_target": "0",
                    "--report_prefix": os.path.relpath(output / "chicago", BASE),
                    "--model_registry_dir": ""}
    remove = {"--trained_model_out", "--trained_model_id"}
    result = []
    index = 0
    while index < len(command):
        token = command[index]
        if token in remove:
            index += 2
            continue
        if token in replacements:
            result.extend([token, replacements[token]])
            index += 2
            continue
        result.append(token)
        index += 1
    result.extend(["--eval_only", "--trained_model_in", str(source_run / "policy.final_training.pt")])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source, output = args.source_run.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "started.json").open("x") as f:
        json.dump({"pid": os.getpid(), "started_at": time.time()}, f)
    policy = source / "policy.final_training.pt"
    digest = hashlib.sha256(policy.read_bytes()).hexdigest()
    command = evaluation_command(source, output)
    atomic_json(output / "command.json", command)
    atomic_json(output / "manifest.json", {"source_run": str(source), "policy_sha256": digest,
                "source_hashes": source_hashes(), "training_days_this_process": 0,
                "protocol": "Original horizons, profiles, seed, five benchmarks, 15 perturbed rivals and static/myopic baselines",
                "source_exit_cause": "Unresolved: both PIDs disappeared without traceback or exit status"})
    child = None
    def status(phase, **details):
        atomic_json(output / "status.json", {"time": time.time(), "phase": phase, **details})
        print(phase, details, flush=True)
    def terminated(signum, frame):
        atomic_json(output / "termination_signal.json", {"signal": signum, "time": time.time()})
        if child is not None and child.poll() is None:
            child.send_signal(signum)
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminated)
    signal.signal(signal.SIGINT, terminated)
    try:
        status("evaluating_saved_day10000_policy")
        env = dict(environment(), PYTHONFAULTHANDLER="1", MPLCONFIGDIR=str(output / ".matplotlib"))
        with (output / "run.log").open("x") as log:
            child = subprocess.Popen(command, cwd=BASE, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            atomic_json(output / "worker.json", {"pid": child.pid})
            while True:
                try:
                    code = child.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    atomic_json(output / "heartbeat.json", {"time": time.time(), "pid": child.pid,
                                "log_bytes": (output / "run.log").stat().st_size})
        atomic_json(output / "worker_exit.json", {"returncode": code, "signal": -code if code < 0 else None})
        if code:
            raise RuntimeError(f"Evaluation exited {code}; no retry")
        if hashlib.sha256(policy.read_bytes()).hexdigest() != digest:
            raise RuntimeError("Source checkpoint changed")
        status("summarizing")
        report = summarize(output, training_required=False)
        atomic_json(output / "policy_integrity.json", {"unchanged": True, "sha256": digest})
        status("completed", all_six_late_leads_positive=report["all_six_late_leads_positive"])
        atomic_json(output / "exit_status.json", {"returncode": 0})
    except BaseException as exc:
        traceback.print_exc()
        status("failed", error=str(exc))
        atomic_json(output / "exit_status.json", {"returncode": 1, "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
