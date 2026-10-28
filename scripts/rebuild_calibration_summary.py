"""Rebuild a missing calibration summary from complete saved policy sweeps."""
import argparse
import json
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from calibration_artifact import supply_point
from diagnostic_logging import atomic_json
from evaluation_selection import selected_outcomes, three_comparators
from scripts.audit_city_calibration import CITIES, load_core
from scripts.v26r_cases import STATIC_GRID


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", required=True, choices=CITIES)
    ap.add_argument("--saved-directory", required=True, type=Path)
    ap.add_argument("--demand-artifact", required=True, type=Path)
    ap.add_argument("--burnin-days", type=int, default=15)
    ap.add_argument("--horizon-days", type=int, default=40)
    args = ap.parse_args()
    saved = args.saved_directory.resolve()
    targets = (saved / "comparators.json", saved / "city_calibration.json")
    if any(path.exists() for path in targets):
        ap.error("refusing to replace an existing summary")
    validation = json.loads((saved / "validation_statics.json").read_text())
    evaluation = json.loads((saved / "evaluation_statics.json").read_text())
    expected = {f"static:{value:g}" for value in STATIC_GRID}
    if set(validation["traces"]) != expected or set(evaluation["traces"]) != expected:
        raise ValueError("saved sweeps do not contain the complete declared grid")
    core = load_core(args.city, saved / "reconstruction_setup",
                     demand_artifact=args.demand_artifact.resolve())
    report = three_comparators(validation["traces"], evaluation["traces"],
                               start=args.burnin_days, end=args.horizon_days)
    report["window"] = {"burnin_days": args.burnin_days,
                        "horizon_days": args.horizon_days,
                        "openings_per_rival": len(validation["rollouts"]) // 2}
    report["warm_start_target"] = float(report["global_constant"].split(":")[1])
    report["outcomes"] = selected_outcomes(report, evaluation["traces"],
                                            start=args.burnin_days, end=args.horizon_days)
    report["losing_outcomes"] = [row for row in report["outcomes"]
        if row["full_profit_lead"] < 0 or row["late_profit_lead"] < 0]
    report["reconstructed_from_complete_saved_sweeps"] = True
    point = supply_point(core)
    atomic_json(targets[0], report)
    atomic_json(targets[1], {"format": "revision6-city-calibration-v1",
        "city": core.market_name, "grid": list(STATIC_GRID),
        "demand_calibration": getattr(core.choice_model, "demand_spec", None),
        "opponent_profit_lead_weight": core.long_term_profit_reward_config.opponent_profit_lead_weight,
        "warm_start_target": report["warm_start_target"], "warm_start_selection": report,
        "supply_operating_point": point})
    atomic_json(saved / "reconstruction.json", {
        "selection_and_outcomes": "recomputed from saved validation/evaluation tapes",
        "supply_operating_point": "deterministically rebuilt under the same demand contract",
        "no_rollouts_repeated": True, "undefined_market_share_values_imputed": False})


if __name__ == "__main__":
    main()
