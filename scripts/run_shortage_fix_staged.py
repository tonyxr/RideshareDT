"""Run the production four-stage learner with the shortage-signal fixes."""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parent
sys.path.insert(0, str(BASE))

from scripts.run_v26r_staged import command, environment, source_hashes, summarize


def _smoke_summary(out: Path, days: int) -> dict:
    with (out / "nyc_training_diagnostics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    with (out / "nyc_evaluation_diagnostics.csv").open() as handle:
        evaluation = list(csv.DictReader(handle))
    config = json.loads((out / "nyc_run_config.json").read_text())
    stages = {
        row["training_stage_opponent_regime"]
        for row in rows
    }
    required_stages = {"market", "static", "catchup", "profit"}
    if len(rows) != days or len(evaluation) != 12 or stages != required_stages:
        raise RuntimeError(
            f"incomplete staged smoke: train={len(rows)}, eval={len(evaluation)}, "
            f"stages={sorted(stages)}"
        )
    reward = config["platform_mdp"]["active_reward"]
    components = config["credit_assignment"]["value_components"]
    if (
        reward["temporal_competitive_reward"]
        or float(reward["market_share_competitiveness_weight"]) != 0.0
        or components != ["own_profit_per_request", "rival_profit_per_request"]
    ):
        raise RuntimeError("smoke did not use the production two-term reward")
    observation_dim = int(config["v26"]["v26r"]["observation_dim"])
    if observation_dim != 34:
        raise RuntimeError("same-day supply proxy is absent from the actor frame")
    updates = [row for row in rows if row["ppo_update_performed"] == "True"]
    shortage_days = sum(
        "driver_shortage" in json.loads(row.get("shock_block_counts") or "{}")
        for row in rows
    )
    if not updates or shortage_days == 0:
        raise RuntimeError("smoke did not exercise PPO updates and shortage days")
    result = {
        "status": "passed",
        "training_days": len(rows),
        "evaluation_steps": len(evaluation),
        "stages": sorted(stages),
        "ppo_updates": len(updates),
        "shortage_days": shortage_days,
        "actor_trace_lambda": float(updates[-1]["ppo_actor_trace_lambda"]),
        "critic_trace_lambda": float(updates[-1]["ppo_critic_trace_lambda"]),
        "final_entropy_coefficient": float(rows[-1]["ppo_ent_coeff"]),
        "observation_dim": observation_dim,
        "value_components": components,
    }
    (out / "smoke_result.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--days", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=251_012)
    parser.add_argument("--integration", action="store_true")
    parser.add_argument(
        "--city-calibration-artifact",
        type=Path,
        default=(
            ROOT
            / "artifacts/revision7_direct_calibration_20261018/nyc/city_calibration.json"
        ),
    )
    args = parser.parse_args()
    if args.days <= 0:
        parser.error("--days must be positive")

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    hashes = source_hashes()
    calibration = args.city_calibration_artifact.resolve()
    cmd = command(
        out,
        days=int(args.days),
        integration=bool(args.integration),
        city="New York City",
        slug="nyc",
        seed=int(args.seed),
        actor_true_shortage_flag=False,
    )
    # The saved operating point is calibrated at the production volume of 50
    # requests/block.  Fast smoke mode uses 5 requests/block and therefore
    # lets Core calibrate that small fixture locally instead of misapplying the
    # production supply point.
    if not args.integration:
        cmd += ["--city_calibration_artifact", str(calibration)]
    (out / "command.json").write_text(json.dumps(cmd, indent=2))
    (out / "manifest.json").write_text(json.dumps({
        "production_shortage_fixes": True,
        "oracle_information": False,
        "city": "New York City",
        "training_days": int(args.days),
        "integration": bool(args.integration),
        "seed": int(args.seed),
        "source_hashes": hashes,
        "started_at": datetime.datetime.now().isoformat(),
        "pid": os.getpid(),
    }, indent=2))

    def status(message: str) -> None:
        (out / "status.txt").write_text(message + "\n")
        print(f"{datetime.datetime.now().isoformat()} {message}", flush=True)

    try:
        status(f"training {args.days} NYC days with all production shortage fixes")
        with (out / "run.log").open("x") as log:
            child = subprocess.Popen(
                cmd,
                cwd=BASE,
                env=environment(),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            (out / "training_pid.txt").write_text(f"{child.pid}\n")
            returncode = child.wait()
        (out / "training_exit_status.txt").write_text(f"{returncode}\n")
        if returncode:
            raise RuntimeError(f"training/evaluation exited {returncode}")
        if source_hashes() != hashes:
            raise RuntimeError("source changed during run; results require review")
        status("checking staged training and evaluation artifacts")
        if args.integration:
            _smoke_summary(out, int(args.days))
        else:
            summarize(
                out,
                expected_days=int(args.days),
                slug="nyc",
                seed=int(args.seed),
            )
        status("completed")
        (out / "exit_status.txt").write_text("0\n")
    except BaseException:
        traceback.print_exc()
        status("failed; artifacts preserved")
        (out / "exit_status.txt").write_text("1\n")
        raise


if __name__ == "__main__":
    main()
