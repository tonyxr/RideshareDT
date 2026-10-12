"""Frozen-checkpoint, paired validation of categorical sampling versus argmax.

Uses the run's recorded CLI/config and cached validation profiles. Never trains,
selects a checkpoint, or opens the final-test partition.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import Core as core_module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    config = json.loads((args.run / "chicago_run_config.json").read_text())
    args.output.mkdir(parents=True, exist_ok=False)

    class Finished(Exception):
        pass

    def diagnose(core, **kwargs):
        core.shared_steps_per_day = kwargs["train_steps_per_day"]
        core.shared_action_interval_steps = kwargs["firm1_action_interval_steps"]
        agent = core.firm1.agent
        original_act = agent.act
        results = {}

        def digest():
            return hashlib.sha256(b"".join(
                value.detach().cpu().numpy().tobytes()
                for value in agent.net.state_dict().values()
            )).hexdigest()

        before = digest()
        for mode in ("argmax_raw", "sample_raw"):
            observations, actions = [], []

            def traced_act(state, *pos, **kw):
                observations.append(np.asarray(state).reshape(-1)[-4:].tolist())
                result = original_act(state, *pos, **kw)
                actions.append(int(result[0]))
                return result

            agent.act = traced_act
            result = core._adaptive_policy_validation(
                customers_per_step=kwargs["eval_customers_per_step"],
                profile_sample_size=kwargs["eval_customers_per_step"],
                horizon=kwargs["eval_timesteps"],
                policy_mode=mode,
                policy_seed=core.seed + 810001,
                opponent_interval_steps=kwargs["firm2_action_interval_days"],
                firm1_interval_steps=core.shared_action_interval_steps,
                steps_per_day=core.shared_steps_per_day,
                aggressive_seed_count=3,
                profile_partition="validation",
            )
            result["diagnostic_action_counts"] = dict(Counter(actions))
            result["diagnostic_twin_features"] = {
                "min": np.min(observations, axis=0).tolist(),
                "max": np.max(observations, axis=0).tolist(),
                "mean": np.mean(observations, axis=0).tolist(),
            }
            result["policy_unchanged"] = before == digest()
            (args.output / (mode + ".json")).write_text(json.dumps(result, indent=2))
            results[mode] = result["rollout_summaries"]
            print(mode, json.dumps(results[mode]), flush=True)
        agent.act = original_act
        (args.output / "summary.json").write_text(json.dumps(results, indent=2))
        raise Finished()

    core_module.Core.run_experiment = diagnose
    sys.argv = config["argv"] + [
        "--trained_model_in", str(args.run / "policy.pt"), "--eval_only",
        "--report_prefix", str(args.output / "diagnostic"),
        "--trained_model_out", str(args.output / "unused.pt"),
        "--profiles_out", str(args.output / "unused.csv"),
    ]
    try:
        core_module.main()
    except Finished:
        pass


if __name__ == "__main__":
    main()
