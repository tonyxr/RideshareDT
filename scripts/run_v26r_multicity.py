"""Independent city jobs: staged PPO, final-policy benchmarks, then corner cases.

No performance gating, automatic retries, early stopping or policy substitution.
Use --smoke for isolated 192-day integration fixtures, never production evidence.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time
import traceback

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "scripts"))
from diagnostic_logging import atomic_json
from platform_mdp import TrainingStageScheduler
from run_v26r_staged import command, environment, source_hashes, summarize, PYTHON, ROOT
from v26r_cases import CASES
from v26r_case_matrix import summarize as summarize_corners
import v26r_figures

CITIES = {"nyc": "New York City", "boston": "Boston", "chicago": "Chicago"}
RIVALS = ("adaptive_best_response_memory", "pi_price_gap", "mpc_grid", "region_supply_demand",
          "surge_driver_incentive", "queue_service_threshold")


def stage_days(days, smoke=False, four_stage=False):
    # Preserve the historical helper result for callers that did not request
    # the revision-10 pipeline. The production runner opts into four stages
    # explicitly when constructing its manifest below.
    fractions = (
        ((.0625, .0625, .25, .625) if smoke else (.1, .1, .3, .5))
        if four_stage
        else (.1, .108, .792)
    )
    scheduler = TrainingStageScheduler(stage_fractions=fractions)
    counts = Counter(scheduler.stage_at(d / days).opponent_regime for d in range(days))
    names = (
        ('market', 'static', 'catchup', 'profit')
        if four_stage
        else ('market', 'static', 'dynamic')
    )
    return [counts[name] for name in names]


def watch(cmd, out, label):
    atomic_json(out / f"{label}_command.json", cmd)
    with (out / f"{label}.log").open("x") as log:
        child = subprocess.Popen(cmd, cwd=BASE, env=environment(), stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        atomic_json(out / f"{label}_pid.json", {"pid": child.pid, "started": time.time()})
        def terminate(signum, frame):
            atomic_json(out / "termination_signal.json", {"signal": signum, "time": time.time()})
            child.send_signal(signum)
            raise SystemExit(128 + signum)
        previous = {sig: signal.signal(sig, terminate) for sig in (signal.SIGTERM, signal.SIGINT)}
        try:
            while True:
                try:
                    code = child.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    atomic_json(out / "heartbeat.json", {"phase": label, "pid": child.pid, "time": time.time(),
                                "log_bytes": (out / f"{label}.log").stat().st_size})
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    atomic_json(out / f"{label}_exit.json", {"returncode": code, "signal": -code if code < 0 else None})
    if code != 0:
        raise RuntimeError(f"{label} exited {code}; artifacts retained, no automatic retry")


def training_evidence(out, slug, days):
    rows = list(csv.DictReader((out / f"{slug}_training_diagnostics.csv").open()))
    if len(rows) != days:
        raise ValueError("Incomplete training history")
    counts, phases, updates, scenarios = defaultdict(Counter), defaultdict(Counter), Counter(), Counter()
    for row in rows:
        stage = row["training_stage"]
        counts[stage].update(json.loads(row["shock_block_counts"]))
        phases[stage].update(json.loads(row["shock_phase_counts"]))
        scenarios[row.get("training_scenario", "unreported")] += 1
        if row["ppo_update_performed"].lower() == "true":
            updates[stage] += 1
    report = dict(days=days, stage_days=dict(Counter(r["training_stage"] for r in rows)),
                  updates_by_stage=dict(updates), shock_blocks_by_stage=dict(counts), shock_phases_by_stage=dict(phases),
                  scenario_days=dict(scenarios),
                  interpretation="Exposure is not competence. Foresight is tested against myopic and commitment baselines.")
    report["trajectory_endpoints"] = {key: {"first": float(rows[0][key]), "last": float(rows[-1][key])}
                                    for key in ("avg_reward", "raw_policy_entropy", "ppo_lr", "behavior_exploration_rate")}
    atomic_json(out / "training_coverage.json", report)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(3, 2, figsize=(13, 11))
    panels = [("avg_reward", "Training reward (stage-specific objective)"),
              ("raw_policy_entropy", "Raw actor entropy / nats"),
              ("ppo_lr", "Actor learning rate"), ("ppo_ent_coeff", "Entropy regularization coefficient (not probability)"),
              ("ppo_explained_variance", "Critic explained variance at PPO updates"),
              ("avg_profit_advantage_per_request", "Own minus rival contribution profit / request")]
    for ax, (key, title) in zip(axes.flat, panels):
        for stage in report["stage_days"]:
            sample = [r for r in rows if r["training_stage"] == stage and
                      (key != "ppo_explained_variance" or r["ppo_update_performed"].lower() == "true")]
            x = np.array([int(r["batch"])+1 for r in sample])
            y = np.array([float(r[key]) for r in sample])
            ax.plot(x, y, ".-", ms=2, lw=.6, alpha=.5, label=stage)
            if len(y) >= 60 and key in ("avg_reward", "raw_policy_entropy", "avg_profit_advantage_per_request"):
                ax.plot(x[59:], np.convolve(y, np.ones(60)/60, mode="valid"), lw=1.3)
        ax.set(title=title, xlabel="Training day")
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"{slug}: {days:,} training days | thin: daily; thick: within-stage 60-day mean")
    fig.tight_layout()
    fig.savefig(out / "training_trajectory.png", dpi=150)
    plt.close(fig)
    return report


def validate_case_outputs(root, slug, specs, openings):
    """Runnability/completeness checks, never a profit or adaptation threshold."""
    report = {}
    for name in CASES:
        data = json.loads((root / slug / name / "eval/sweep.json").read_text())
        if len(data["meta"]["rollouts"]) != len(RIVALS)*openings:
            raise ValueError(f"Missing rival openings: {name}")
        for spec in specs:
            trace = data["traces"][spec]
            counts = Counter(r["rollout"] for r in trace)
            if len(counts) != len(RIVALS)*openings or set(counts.values()) != {240}:
                raise ValueError(f"Incomplete 40-day case: {name}/{spec}")
        report[name] = {"rollouts_per_spec": len(data["meta"]["rollouts"]),
                        "specs": len(data["traces"]), "steps_per_rollout": 240,
                        "regimes": sorted({r["regime"] for r in data["traces"]["rl_actor"]})}
    atomic_json(root / slug / "coverage.json", report)
    return report


def city_job(args, out):
    slug = args.city
    days = 192 if args.smoke else args.days
    cmd = command(out, days=days, integration=args.smoke, city=CITIES[slug], slug=slug,
                  update_days=args.update_days, rollout_transitions=args.rollout_transitions,
                  lead_weight=args.lead_weight, seed=getattr(args, "seed", 251012),
                  training_scenario_manifest=(Path(getattr(args, "training_scenario_root", "")) / f"{slug}.json"
                                              if getattr(args, "training_scenario_root", None) else None))
    cmd.append('--learning_corrections')
    if args.smoke and getattr(args, 'training_scenario_root', None):
        # The shortened smoke must cross at least one complete six-opponent
        # population cycle so it actually executes a scripted case.
        cmd[cmd.index('--training_dynamic_episode_days') + 1] = '20'
    if getattr(args, 'city_calibration_root', None):
        cmd += ['--city_calibration_artifact', str((Path(args.city_calibration_root).resolve()/slug/'city_calibration.json'))]
        cmd[cmd.index('--rl_warm_start_target') + 1] = 'auto'
        if args.smoke:
            # A saved supply operating point is valid only at its audited
            # request volume. Keep the smoke horizon small, not the market.
            for flag in ('--train_customers', '--checkpoint_validation_customers', '--eval_customers'):
                cmd[cmd.index(flag) + 1] = '50'
            cmd[cmd.index('--loyalty_pool_size') + 1] = '180'
    cmd += ['--learner_revision', str(getattr(args, 'learner_revision', 4)),
            '--ppo_rollout_episodes', str(getattr(args, 'ppo_rollout_episodes', 2))]
    cmd[cmd.index('--exploration_commit_mean') + 1] = '0'
    atomic_json(out / "status.json", {"phase": "staged_training_and_standard_evaluation", "days": days})
    watch(cmd, out, "training")
    evidence = training_evidence(out, slug, days)
    policy = out / "policy.final_training.pt"
    digest = hashlib.sha256(policy.read_bytes()).hexdigest()
    if not args.smoke:
        summarize(out, expected_days=days, slug=slug, seed=getattr(args, "seed", 251012))
    if getattr(args, 'figures', False):
        v26r_figures.training(out, slug)
        if not args.smoke:
            v26r_figures.standard(out, slug)
    if args.skip_corners:
        if hashlib.sha256(policy.read_bytes()).hexdigest() != digest:
            raise ValueError("Final checkpoint changed during reporting")
        atomic_json(out / "policy_integrity.json", {"unchanged": True, "sha256": digest})
        atomic_json(out / "status.json", {"phase": "completed", "training": evidence,
                                          "corner_evaluation": "not requested"})
        return
    corner_root = out / "corners"
    atomic_json(out / "status.json", {"phase": "corner_case_evaluation", "policy_sha256": digest})
    case_cmd = [str(PYTHON), "-u", str(BASE / "scripts/v26r_case_eval.py"), CITIES[slug], slug, "all",
                "--seed-set", "eval" if args.smoke else "both", "--policy", str(policy),
                "--output-root", str(corner_root), "--openings", "1" if args.smoke else "10",
                "--customers", "5" if args.smoke else "50", "--pool", "300" if args.smoke else "2000",
                "--rivals", ",".join(RIVALS), "--horizon-days", "40", "--channels", "full",
                "--chunk", "1", "--budget-seconds", "1000000000"]
    if args.smoke:
        case_cmd += ["--specs", "rl_actor;static:1.1"]
    watch(case_cmd, out, "corners")
    validate_case_outputs(corner_root, slug, ["rl_actor", "static:1.1"], 1 if args.smoke else 10)
    if not args.smoke:
        summarize_corners(corner_root, slug, corner_root / slug / "matrix")
        if getattr(args, 'figures', False):
            v26r_figures.corners(out, corner_root / slug / 'matrix/matrix.csv', slug)
    if hashlib.sha256(policy.read_bytes()).hexdigest() != digest:
        raise ValueError("Final checkpoint changed during corner evaluation")
    atomic_json(out / "policy_integrity.json", {"unchanged": True, "sha256": digest})
    atomic_json(out / "status.json", {"phase": "completed", "smoke": args.smoke, "training": evidence})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--days", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=251012)
    ap.add_argument("--city", choices=CITIES)
    ap.add_argument('--city-calibration-root', type=Path,
                    help='Completed repaired audit containing each city/city_calibration.json')
    ap.add_argument('--training-scenario-root', type=Path,
                    help='Directory containing nyc.json, boston.json and chicago.json curricula.')
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--update-days", type=int, default=5)
    ap.add_argument("--rollout-transitions", type=int, default=180)
    from model_contract import OPPONENT_PROFIT_LEAD_WEIGHT
    ap.add_argument("--lead-weight", type=float, default=OPPONENT_PROFIT_LEAD_WEIGHT)
    ap.add_argument("--learner-revision", type=int, choices=(3, 4), default=4)
    ap.add_argument("--ppo-rollout-episodes", type=int, default=2)
    ap.add_argument("--figures", action="store_true", help="PNG/PDF trajectories and all corner/rival figures.")
    ap.add_argument("--skip-corners", action="store_true", help="Finish after standard final-policy evaluation; do not launch the large corner suite.")
    args = ap.parse_args()
    if args.days < 1 or args.ppo_rollout_episodes < 1:
        ap.error('days and rollout episodes must be positive')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / "started.json").open("x") as f:
        json.dump({"pid": os.getpid(), "started": time.time()}, f)
    hashes = source_hashes()
    data_paths = [ROOT / f"data/processed/{s}/{n}" for s in CITIES for n in ("calibration.json", "trip_bank.csv")]
    data_paths += [ROOT / f"artifacts/v26_confirmatory/thresholds/{s}/llm_numeric_primary.jsonl" for s in CITIES]
    manifest = dict(days_per_city=192 if args.smoke else args.days, cities=CITIES,
                    seed=args.seed, source_hashes=hashes, smoke=args.smoke,
                    ppo_update_interval_days=args.update_days, ppo_min_rollout_transitions=args.rollout_transitions,
                    learner_revision=args.learner_revision, ppo_rollout_episodes=args.ppo_rollout_episodes,
                    schedules=dict(clock='configured_stage_simulation_days', actor_lr=[.0003, .00005],
                                   critic_lr_multiplier=3, entropy=[.04, .01] if args.learner_revision==4 else [.01,.0005],
                                   external_random_exploration=0, reset='each stage, not each episode'),
                    opponent_reward_weights={
                        "own_profit": 0.65 * (1.0 - args.lead_weight),
                        "instant_profit_lead": 0.65 * args.lead_weight,
                        "durable_profit_lead": 0.20,
                        "adaptive_profit_residual": 0.10,
                        "completed_share_lead": 0.05,
                    },
                    reward_scale="common dollar scale for profit, lead, durable lead, and innovation; 10-point scale for completed-share lead",
                    temporal_reward_contract={
                        "durable_horizon_days": 5.0,
                        "residual": "realized profit lead minus pre-step EMA forecast; not critic TD error",
                        "critic_components": 6,
                    },
                    skip_corners=args.skip_corners,
                    training_scenario_root=(str(args.training_scenario_root.resolve()) if args.training_scenario_root else None),
                    data_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in data_paths},
                    stage_days=stage_days(
                        192 if args.smoke else args.days,
                        args.smoke,
                        four_stage=True,
                    ),
                    corner_cases=[] if args.skip_corners else list(CASES), rivals=RIVALS, corner_openings=0 if args.skip_corners else (1 if args.smoke else 10),
                    corner_horizon_days=40, statistical_protocol="Paired opening-level summaries; one training seed per city",
                    timeout_policy="No automatic time/performance termination; heartbeat and exit evidence only")
    atomic_json(out / "manifest.json", manifest)
    if not args.city:
        with tarfile.open(out / "source_snapshot.tar.gz", "w:gz") as archive:
            for name in hashes:
                archive.add(BASE / name, arcname=name)
            archive.add(BASE / "scenarios", arcname="scenarios")
    try:
        if args.city:
            city_job(args, out)
        else:
            children = []
            for slug in CITIES:
                city_out = out / slug
                city_out.mkdir()
                cmd = [str(PYTHON), "-u", str(Path(__file__).resolve()), "--output", str(city_out),
                       "--days", str(args.days), "--city", slug,
                       "--seed", str(args.seed),
                       "--update-days", str(args.update_days), "--rollout-transitions", str(args.rollout_transitions),
                       "--lead-weight", str(args.lead_weight), '--learner-revision', str(args.learner_revision),
                       '--ppo-rollout-episodes', str(args.ppo_rollout_episodes)] + (["--smoke"] if args.smoke else [])
                if args.figures:
                    cmd.append('--figures')
                if args.skip_corners:
                    cmd.append("--skip-corners")
                if args.city_calibration_root:
                    cmd += ['--city-calibration-root', str(args.city_calibration_root.resolve())]
                if args.training_scenario_root:
                    cmd += ['--training-scenario-root', str(args.training_scenario_root.resolve())]
                with (city_out / "pipeline.log").open("x") as log:
                    proc = subprocess.Popen(cmd, cwd=BASE, env=environment(), stdout=log,
                                            stderr=subprocess.STDOUT, start_new_session=True)
                children.append((slug, proc))
            atomic_json(out / "workers.json", {slug: p.pid for slug, p in children})
            while any(p.poll() is None for _, p in children):
                atomic_json(out / "status.json", {"phase": "running", "time": time.time(),
                            "exit_codes": {slug: p.poll() for slug, p in children}})
                time.sleep(30)
            codes = {slug: p.returncode for slug, p in children}
            atomic_json(out / "status.json", {"phase": "completed" if all(c == 0 for c in codes.values()) else "failed",
                                              "exit_codes": codes})
            if any(codes.values()):
                raise RuntimeError(f"City failures: {codes}; no retry, other artifacts preserved")
            if not args.smoke:
                aggregate = {}
                for slug in CITIES:
                    standard = json.loads((out/slug/"result.json").read_text())
                    aggregate[slug] = dict(standard_profit_leads=standard["profit_leads"],
                        adaptation=standard["adaptation"], heldout_critic_ev=standard["evaluated_critic_ev"])
                    if not args.skip_corners:
                        corners = json.loads((out/slug/"corners"/slug/"matrix/summary.json").read_text())
                        aggregate[slug].update(
                            corner_full_profit_losses=len(corners["full_profit_losing_cases"]),
                            corner_late_profit_losses=len(corners["late_profit_losing_cases"]),
                            corner_late_marketshare_losses=len(corners["late_marketshare_losing_cases"]))
                atomic_json(out/"summary.json", aggregate)
                if args.figures:
                    v26r_figures.gallery(out)
        if source_hashes() != hashes:
            raise RuntimeError("Sources changed during experiment; evidence requires review")
        atomic_json(out / "exit_status.json", {"returncode": 0})
    except BaseException as exc:
        traceback.print_exc()
        atomic_json(out / "status.json", {"phase": "failed", "error": str(exc)})
        atomic_json(out / "exit_status.json", {"returncode": 1, "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
