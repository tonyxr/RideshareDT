"""Durable, observational diagnostics. No model imports or RNG operations."""
import csv
import json
import math
import os
from pathlib import Path
from datetime import datetime, timezone


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if hasattr(value, "tolist"):
        return clean(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as f:
        json.dump(clean(value), f, allow_nan=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


class DiagnosticJournal:
    def __init__(self, prefix, *, plots=True):
        self.prefix = str(prefix)
        Path(self.prefix).parent.mkdir(parents=True, exist_ok=True)
        self.streams = set()
        self.pending = []
        self.episodes = []
        self.plots = plots
        self.event("started", {})

    def append(self, stream, record):
        path = self.prefix + "_" + stream + ".jsonl"
        # Never append a new experiment to an old experiment's evidence.
        mode = "a" if stream in self.streams else "x"
        with open(path, mode) as f:
            f.write(json.dumps(clean(record), allow_nan=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.streams.add(stream)

    def event(self, phase, details):
        record = {"time_utc": datetime.now(timezone.utc).isoformat(),
                  "phase": phase, "details": details}
        self.append("events", record)
        atomic_json(self.prefix + "_progress.json", record)

    def snapshot(self, name, report):
        atomic_json(self.prefix + "_" + name + ".json", report)
        self.event(name + "_saved", {})

    def training_day(self, row, *, episode_end=False):
        self.append("training_live", row)
        self.pending.append(clean(row))
        if row.get("ppo_update_performed", False):
            self.append("ppo_updates", {k: v for k, v in row.items()
                if k.startswith("ppo_") or k in ("batch", "training_stage", "training_episode", "training_opponent")})
        if episode_end:
            self.finish_episode()

    def finish_episode(self):
        if not self.pending:
            return
        rows, self.pending = self.pending, []
        def mean(key, source=rows):
            values = [r.get(key) for r in source if isinstance(r.get(key), (float, int))]
            return sum(values) / len(values) if values else None
        updates = [r for r in rows if r.get("ppo_update_performed")]
        summary = {"episode": len(self.episodes) + 1,
            "stage": rows[0]["training_stage"], "opponent": rows[0]["training_opponent"],
            "start_day": rows[0]["batch"] + 1, "end_day": rows[-1]["batch"] + 1,
            "days": len(rows), "reward_mean": mean("avg_reward"),
            "entropy_mean": mean("raw_policy_entropy"),
            "actor_lr_mean": mean("ppo_lr"),
            "critic_lr_mean": mean("ppo_critic_lr"),
            "exploration_mean": mean("behavior_exploration_rate"),
            "entropy_coefficient_mean": mean("ppo_ent_coeff"),
            "sampled_non_greedy_mean": mean("sampled_not_raw_top_rate"),
            "profit_lead_mean": mean("avg_profit_advantage_per_request"),
            "ppo_updates": len(updates), "ppo_kl_mean": mean("ppo_approx_kl", updates),
            "critic_ev_mean": mean("ppo_explained_variance", updates),
            "critic_loss_mean": mean("ppo_value_loss", updates)}
        self.append("episodes", summary)
        self.episodes.append(summary)
        path = Path(self.prefix + "_episodes.csv")
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary))
            writer.writeheader()
            writer.writerows(self.episodes)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
        self.event("episode_completed", summary)
        if self.plots:
            self.plot_episodes()

    def plot_episodes(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 2, figsize=(12, 11))
        panels = [("reward_mean", "Sampled training reward", "Reward / block"),
                  ("entropy_mean", "Raw actor entropy (not action diversity)", "Entropy / nats"),
                  ("actor_lr_mean", "Actor learning rate", "Learning rate"),
                  ("entropy_coefficient_mean", "Scheduled entropy regularization", "Coefficient (not exploration probability)"),
                  ("profit_lead_mean", "Own minus rival contribution profit", "Profit / reference request"),
                  ("critic_ev_mean", "Critic fit at actual PPO updates", "Value-target explained variance")]
        groups = list(dict.fromkeys((r["stage"], r["opponent"]) for r in self.episodes))
        for ax, (key, title, unit) in zip(axes.flat, panels):
            for stage, opponent in groups:
                data = [r for r in self.episodes if (r["stage"], r["opponent"]) == (stage, opponent)]
                ax.plot([r["end_day"] for r in data],
                        [r[key] if r[key] is not None else float("nan") for r in data],
                        "o-", ms=3, lw=1, label=f"{stage}: {opponent}")
            ax.set(title=title, ylabel=unit, xlabel="Episode-end training day")
            ax.grid(alpha=.2)
        axes[0, 0].legend(fontsize=6)
        fig.suptitle("Live episode diagnostics | completed episodes only")
        fig.text(.04, .015, "Reward objectives may differ by stage. Episodes differ in shocks/openings; these are descriptive means, not confidence intervals.", fontsize=8)
        fig.tight_layout(rect=(0, .04, 1, .94))
        path = Path(self.prefix + "_episode_diagnostics.png")
        temporary = path.with_name(path.stem + ".tmp.png")
        fig.savefig(temporary, dpi=140)
        plt.close(fig)
        os.replace(temporary, path)
