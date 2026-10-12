"""Report economic outcomes, never promotion-gate or positive-reward labels."""
import argparse
import json
import math
from pathlib import Path


def summarize_suite(report):
    rows = {}
    for name, result in report.get("rollout_summaries", {}).items():
        own = float(result["own_profit_per_request"])
        rival = float(result["rival_profit_per_request"])
        tail = float(result["late_objective_advantage"])
        if not all(math.isfinite(x) for x in (own, rival, tail)):
            raise ValueError(f"Nonfinite profit outcome for {name}")
        rows[name] = {
            "own_profit_per_request": own,
            "rival_profit_per_request": rival,
            "full_horizon_profit_lead": own - rival,
            "last_quarter_profit_lead": tail,
            "tail_rolling_profit_win_rate": result.get("objective_dominance_rate"),
            "positive_full_and_tail_lead": own > rival and tail > 0,
            "own_profit_positive": own > 0,
            "market_steps": int(result["steps"]),
        }
    return {
        "rollouts": rows,
        "positive_full_and_tail_lead_on_every_observed_path": (
            all(row["positive_full_and_tail_lead"] for row in rows.values())
            if rows else None
        ),
        "losing_or_tied_paths": [
            name for name, row in rows.items()
            if not row["positive_full_and_tail_lead"]
        ],
    }


def build_report(config):
    if config["args"]["policy_objective"] != "profit_dominance_v30":
        raise ValueError("This report requires monetary v30 profit dominance")
    primary = summarize_suite(config.get("benchmark_evaluation", {}))
    if not primary["rollouts"]:
        raise ValueError("No final benchmark outcomes found")
    return {
        "effective_seed": config["effective_seed"],
        "training_days": config["args"]["train_timesteps"],
        "policy_mode": config["eval_policy_mode"],
        "profile_partition": config["benchmark_evaluation"].get("profile_partition"),
        "definition": "Own minus rival contribution profit per request; full-horizon and final-quarter means must both exceed zero on every reported path. This is reporting only, not a learning gate or a claim of winning every time step.",
        "actual_rival_and_benchmarks": primary,
        "perturbed_rivals": summarize_suite(config.get("v26_ood_evaluation", {})),
        "baseline_comparisons": {
            name: section.get("comparison", {})
            for name, section in config.get("v26_baseline_evaluation", {}).items()
        },
        "limitations": [
            "One training seed; evaluation openings are not independent training seeds.",
            "The standard suite has three actual-rival openings and one path per benchmark; perturbed rivals are separate stress cases.",
            "Positive reward or a promotion-gate pass does not establish profit dominance.",
            "The uniform static grid is not the optimal five-dimensional fixed tariff.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(json.loads(args.config.read_text()))
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(report["definition"])
    print(f"Seed: {report['effective_seed']}; mode: {report['policy_mode']}; partition: {report['profile_partition']}")
    for suite in ("actual_rival_and_benchmarks", "perturbed_rivals"):
        summary = report[suite]
        print(f"\n{suite}: {summary['positive_full_and_tail_lead_on_every_observed_path']}")
        for name, row in summary["rollouts"].items():
            print(f"{name}: full={row['full_horizon_profit_lead']:+.6f}, tail={row['last_quarter_profit_lead']:+.6f}, maintained={row['positive_full_and_tail_lead']}")
    print("\nLimitations: " + " ".join(report["limitations"]))


if __name__ == "__main__":
    main()
