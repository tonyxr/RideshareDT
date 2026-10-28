"""Per-rival corner-case reports; selection uses validation tapes only."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "scripts"))
from diagnostic_logging import atomic_json
from v26r_cases import CASES, BURNIN_DAYS
from evaluation_selection import three_comparators, trajectory_scores


def values(traces, rival, column, start, end, spd=6):
    return trajectory_scores(traces, rival=rival, column=column, start=start, end=end, spd=spd)


def paired(treated, control):
    if not treated or set(treated) != set(control):
        raise ValueError("Missing or unmatched evaluation tapes")
    diffs = np.array([treated[k] - control[k] for k in sorted(treated)])
    return dict(mean=float(diffs.mean()), n=len(diffs),
                se=float(diffs.std(ddof=1)/np.sqrt(len(diffs))) if len(diffs) > 1 else None)


def select(sweep, prefix, rival, burnin, horizon, column="profit_per_request"):
    candidates = {spec: values(trace, rival, column, burnin, horizon)
                  for spec, trace in sweep["traces"].items() if spec.startswith(prefix)}
    candidates = {k: v for k, v in candidates.items() if v}
    return min(candidates, key=lambda k: (-np.mean(list(candidates[k].values())), k)) if candidates else None


def summarize(root, slug, out):
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    comparisons = {}
    for case_name, case in CASES.items():
        case_dir = root / slug / case_name
        val = json.loads((case_dir / "val/sweep.json").read_text())
        ev = json.loads((case_dir / "eval/sweep.json").read_text())
        horizon = ev["meta"]["horizon_steps"] / ev["meta"]["steps_per_day"]
        burnin = case.get("burnin", BURNIN_DAYS)
        comparisons[case_name] = {objective: three_comparators(val['traces'], ev['traces'],
            start=burnin, end=horizon, spd=ev['meta']['steps_per_day'], column=objective)
            for objective in ('reward', 'profit_per_request')}
        for rival in sorted({r[1] for r in ev["meta"]["rollouts"]}):
            best = select(val, "static:", rival, burnin, horizon)
            oracle = select(val, "oracle:", rival, burnin, horizon, column='reward')
            objective_static = select(val, 'static:', rival, burnin, horizon, column='reward')
            if best is None or best not in ev["traces"]:
                raise ValueError(f"Missing validation-selected static: {case_name}/{rival}")
            if objective_static is None or objective_static not in ev['traces']:
                raise ValueError(f"Missing validation objective-static evidence: {case_name}/{rival}")
            specs = [best, "rl_actor", "myopic_greedy"] + ([oracle] if oracle else []) + ([objective_static] if objective_static else [])
            specs += [s for s in ev["traces"] if s.startswith("schedule:")]
            for spec in dict.fromkeys(specs):
                trace = ev["traces"][spec]
                row = dict(case=case_name, role=case["role"], rival=rival, policy=spec,
                           selected_static=best, selection="validation per rival: profit static and reward oracle; reward static reported separately")
                row['selected_objective_static'] = objective_static
                for window, start in (("full", burnin), ("late", horizon*.75)):
                    for col in ("profit_per_request", "rival_profit_per_request", "profit_lead",
                                "own_share", "rival_share", "own_market_share", "share_lead", "reward"):
                        v = values(trace, rival, col, start, horizon)
                        row[f"{window}_{col}"] = float(np.mean(list(v.values()))) if v else None
                    for col in ("profit_per_request", "own_market_share"):
                        diff = paired(values(trace, rival, col, start, horizon),
                                      values(ev["traces"][best], rival, col, start, horizon))
                        row[f"{window}_{col}_minus_static"] = diff["mean"]
                        row[f"{window}_{col}_paired_se"] = diff["se"]
                        row["openings"] = diff["n"]
                    if objective_static:
                        for col in ('reward', 'profit_per_request', 'profit_lead', 'own_market_share'):
                            diff = paired(values(trace, rival, col, start, horizon),
                                values(ev['traces'][objective_static], rival, col, start, horizon))
                            row[f'{window}_{col}_minus_objective_static'] = diff['mean']
                rows.append(row)
            events = []
            for spec in dict.fromkeys(specs):
                for day in range(max(0, case["onset"]-3), int(horizon)):
                    item = dict(day=day, relative_day=day-case["onset"], policy=spec)
                    for col in ("firm1_level", "firm2_level_effective", "profit_lead", "own_market_share",
                                "stock_firm1", "loyalty_mean_firm1", "profit_per_request", "rival_profit_per_request", "reward"):
                        v = values(ev["traces"][spec], rival, col, day, day+1)
                        item[col] = float(np.mean(list(v.values()))) if v else None
                    events.append(item)
            event_dir = out / "events"
            event_dir.mkdir(exist_ok=True)
            with (event_dir / f"{case_name}_{rival}.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(events[0]))
                writer.writeheader(); writer.writerows(events)
            if events:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(3, 2, figsize=(12, 11))
                for ax, col in zip(axes.flat, ("firm1_level", "profit_lead", "own_market_share", "stock_firm1", "profit_per_request", "reward")):
                    for spec in dict.fromkeys(specs):
                        series = [e for e in events if e["policy"] == spec]
                        label = "regime oracle" if spec.startswith("oracle:") else "scripted schedule" if spec.startswith("schedule:") else spec
                        ax.plot([e["relative_day"] for e in series], [e[col] for e in series], label=label, lw=1)
                    ax.axvspan(0, case["end"]-case["onset"]+1, color="grey", alpha=.12)
                    ax.set(xlabel="Days relative to event onset", ylabel=col)
                    ax.grid(alpha=.15)
                axes[0, 0].legend(fontsize=7)
                fig.suptitle(f"{slug}: {case_name}, initial {rival} | mean over evaluation openings")
                fig.tight_layout()
                for extension in ('png', 'pdf'):
                    fig.savefig(event_dir / f"{case_name}_{rival}.{extension}", dpi=170)
                plt.close(fig)
    atomic_json(out / "matrix.json", rows)
    atomic_json(out / "three_comparators.json", comparisons)
    with (out / "matrix.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    rl = [r for r in rows if r["policy"] == "rl_actor"]
    summary = dict(city=slug, cases=len(CASES), rl_case_rival_cells=len(rl),
        full_profit_losing_cases=[r for r in rl if r["full_profit_lead"] < 0],
        late_profit_losing_cases=[r for r in rl if r["late_profit_lead"] < 0],
        late_marketshare_losing_cases=[r for r in rl if r["late_own_market_share"] is not None and r["late_own_market_share"] < .5],
        limitations=["One trained seed per city; openings are evaluation replications, not training replications.",
                     "Full means exclude the declared burn-in; late is the final quarter of the 40-day tape.",
                     "Market share is own completed rides / both firms' completed rides; own_share uses all reference requests.",
                     "SEs are descriptive paired opening-level errors; no multiple-comparison significance claim.",
                     "Named null/strategic effects are hypotheses, not required simulator outcomes.",
                     "S2 and S3 share the same promotion tape; different scripted controllers do not make independent shock cases.",
                     "Hindsight is the best evaluated independent complete trajectory, not a certified dynamic upper bound.",
                     "S4 labels the initial rival; all policies face its scripted switch to PI price gap."])
    atomic_json(out / "summary.json", summary)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rivals = sorted({r["rival"] for r in rl})
    fig, axes = plt.subplots(1, 2, figsize=(14, 9))
    for ax, key, title in zip(axes, ("late_profit_lead", "late_share_lead"),
                             ("Late contribution-profit lead ($/request)", "Late completed-share lead (fraction)")):
        grid = np.array([[next(r[key] for r in rl if r["case"] == c and r["rival"] == rival)
                          for rival in rivals] for c in CASES])
        lim = max(.01, float(np.nanmax(np.abs(grid))))
        im = ax.imshow(grid, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
        ax.set_xticks(range(len(rivals)), rivals, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(len(CASES)), list(CASES), fontsize=8)
        ax.set_title(title)
        for i in range(len(CASES)):
            for j in range(len(rivals)):
                ax.text(j, i, f"{grid[i,j]:+.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(grid[i,j]) > .6*lim else "black")
        fig.colorbar(im, ax=ax, shrink=.6)
    fig.suptitle(f"{slug}: final saved policy on corner cases; red = losing")
    fig.tight_layout()
    fig.savefig(out / "corner_dominance.png", dpi=150)
    plt.close(fig)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("slug")
    ap.add_argument("--root", type=Path, default=BASE/"runs/cases")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    print(json.dumps(summarize(args.root, args.slug, args.out or args.root/args.slug/"matrix"), indent=2))
