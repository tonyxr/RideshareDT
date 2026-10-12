#!/usr/bin/env python3
"""Train-only diagnostic for the centered structured recurrent critic.

This command deliberately does *not* train or evaluate a policy.  It reuses
the warm rule, 15-point local support, fit seeds, and held-out TRAIN critic
gate seeds recorded by a v12 diagnostic.  Validation and test environments
are never constructed.  The resulting report compares the structured critic
against its own frozen prior, the zero-delta baseline, and the recorded v12
generic-MLP critic on identical CRN probes.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from calibration_utils import load_calibration_json  # noqa: E402
from continuous_market_env import (  # noqa: E402
    ContinuousMarketEnv,
    DailyEnvironmentConfig,
    build_v28_core,
)
from recurrent_sac import ActionTransform, Episode, RecurrentSACAgent, RecurrentSACConfig  # noqa: E402
from scripts.v28_learner_ladder import (  # noqa: E402
    _critic_ranking_gate,
    _dataset_paths,
    _local_action_support,
    _local_deviation_episode,
    _observation_identity_indices,
    _shock_config,
    _stratified_deviation_days,
    _verify_local_replay_coverage,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _v12_warm_start(report: Mapping[str, Any]) -> Mapping[str, Any]:
    stages = report.get("stage_results", {})
    if "l0" not in stages or "warm_start" not in stages["l0"]:
        raise ValueError("reference report does not contain an L0 warm start")
    return stages["l0"]["warm_start"]


def _paired_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    groups: Dict[Tuple[int, int], Dict[int, Mapping[str, Any]]] = {}
    for row in rows:
        metadata = row["metadata"]
        key = (int(metadata["seed"]), int(metadata["deviation_day"]))
        groups.setdefault(key, {})[int(metadata["support_index"])] = row
    actual: List[float] = []
    predicted: Dict[str, List[float]] = {"qmin": [], "q1": [], "q2": []}
    for members in groups.values():
        center = members[0]
        for support_index in sorted(index for index in members if index != 0):
            row = members[support_index]
            actual.append(float(row["return_probe"]) - float(center["return_probe"]))
            predicted["qmin"].append(float(row["q_probe"]) - float(center["q_probe"]))
            predicted["q1"].append(float(row["q1_probe"]) - float(center["q1_probe"]))
            predicted["q2"].append(float(row["q2_probe"]) - float(center["q2_probe"]))
    actual_array = np.asarray(actual, dtype=np.float64)
    zero_rmse = float(np.sqrt(np.mean(np.square(actual_array))))
    output: Dict[str, Any] = {
        "pair_count": int(actual_array.size),
        "zero_delta_rmse": zero_rmse,
    }
    for name, values in predicted.items():
        array = np.asarray(values, dtype=np.float64)
        rmse = float(np.sqrt(np.mean(np.square(array - actual_array))))
        output[name] = {
            "rmse": rmse,
            "rmse_to_zero_delta_ratio": float(rmse / max(zero_rmse, 1.0e-12)),
            "directional_accuracy": float(
                np.mean(np.sign(array) == np.sign(actual_array))
            ),
        }
    return output


def _quadratic_design(normalized_offsets: np.ndarray) -> np.ndarray:
    z = np.asarray(normalized_offsets, dtype=np.float64)
    if z.ndim != 2 or z.shape[1] != 3:
        raise ValueError("quadratic probe offsets must have shape [N,3]")
    return np.column_stack(
        (
            z,
            -0.5 * np.square(z[:, 0]),
            -0.5 * np.square(z[:, 1]),
            -0.5 * np.square(z[:, 2]),
            -z[:, 0] * z[:, 1],
            -z[:, 0] * z[:, 2],
            -z[:, 1] * z[:, 2],
        )
    )


def _surface_from_deltas(
    normalized_offsets: np.ndarray, deltas: np.ndarray
) -> Dict[str, Any]:
    design = _quadratic_design(normalized_offsets)
    target = np.asarray(deltas, dtype=np.float64)
    coefficients, _, rank, singular_values = np.linalg.lstsq(
        design, target, rcond=None
    )
    hessian = np.array(
        [
            [coefficients[3], coefficients[6], coefficients[7]],
            [coefficients[6], coefficients[4], coefficients[8]],
            [coefficients[7], coefficients[8], coefficients[5]],
        ],
        dtype=np.float64,
    )
    fitted = design @ coefficients
    return {
        "slope": coefficients[:3].tolist(),
        "positive_curvature_matrix": hessian.tolist(),
        "negative_hessian": (-hessian).tolist(),
        "positive_curvature_eigenvalues": np.linalg.eigvalsh(hessian).tolist(),
        "fit_rmse": float(np.sqrt(np.mean(np.square(fitted - target)))),
        "design_rank": int(rank),
        "design_singular_values": singular_values.tolist(),
    }


def _curvature_recovery(
    support: Mapping[str, Any], diagnostics: Mapping[str, Any]
) -> Dict[str, Any]:
    offsets = {
        int(row["index"]): np.asarray(row["offset"], dtype=np.float64)
        / np.asarray(support["radius_coordinates"], dtype=np.float64)
        for row in support["actions"]
    }
    groups: Dict[Tuple[int, int], Dict[int, Mapping[str, Any]]] = {}
    for row in diagnostics["episodes"]:
        metadata = row["metadata"]
        key = (int(metadata["seed"]), int(metadata["deviation_day"]))
        groups.setdefault(key, {})[int(metadata["support_index"])] = row
    rows: List[Dict[str, Any]] = []
    for (seed, deviation_day), members in sorted(groups.items()):
        center = members[0]
        support_indices = sorted(index for index in members if index != 0)
        normalized = np.stack([offsets[index] for index in support_indices])
        actual_delta = np.asarray(
            [
                float(members[index]["return_probe"])
                - float(center["return_probe"])
                for index in support_indices
            ],
            dtype=np.float64,
        )
        actual_surface = _surface_from_deltas(normalized, actual_delta)
        twins: Dict[str, Any] = {}
        for prefix in ("critic1", "critic2"):
            lower = np.asarray(
                center[f"{prefix}_lower_cholesky_probe"], dtype=np.float64
            )
            learned_curvature = lower @ lower.T
            learned_slope = np.asarray(
                center[f"{prefix}_slope_probe"], dtype=np.float64
            )
            target_curvature = np.asarray(
                actual_surface["positive_curvature_matrix"], dtype=np.float64
            )
            twins[prefix] = {
                "slope": learned_slope.tolist(),
                "positive_curvature_matrix": learned_curvature.tolist(),
                "negative_hessian": (-learned_curvature).tolist(),
                "positive_curvature_eigenvalues": np.linalg.eigvalsh(
                    learned_curvature
                ).tolist(),
                "curvature_matrix_rmse_vs_local_crn_fit": float(
                    np.sqrt(np.mean(np.square(learned_curvature - target_curvature)))
                ),
                "has_cross_terms": bool(
                    np.max(
                        np.abs(learned_curvature - np.diag(np.diag(learned_curvature)))
                    )
                    > 1.0e-8
                ),
            }
        rows.append(
            {
                "seed": seed,
                "deviation_day": deviation_day,
                "actual_crn_surface": actual_surface,
                "twins": twins,
            }
        )
    return {
        "rows": rows,
        "all_actual_surfaces_concave": bool(
            rows
            and all(
                min(row["actual_crn_surface"]["positive_curvature_eigenvalues"])
                > 0.0
                for row in rows
            )
        ),
        "all_learned_surfaces_strictly_concave": bool(
            rows
            and all(
                min(twin["positive_curvature_eigenvalues"]) > 0.0
                for row in rows
                for twin in row["twins"].values()
            )
        ),
        "cross_terms_recovered": bool(
            rows
            and all(
                twin["has_cross_terms"]
                for row in rows
                for twin in row["twins"].values()
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--v12-report",
        default="artifacts/v28_rsac/chicago_l0_v12_20261007/report.json",
    )
    parser.add_argument(
        "--profile-cache",
        required=True,
        help="explicit legacy cache used by the v12 comparator",
    )
    parser.add_argument(
        "--out",
        default="artifacts/v28_rsac/chicago_structured_critic_probe_v13_20261007",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--phase1-steps", type=int, default=200)
    parser.add_argument("--phase2-steps", type=int, default=200)
    args = parser.parse_args()
    if args.phase1_steps != 200 or args.phase2_steps != 200:
        raise ValueError(
            "this preregistered diagnostic fixes both critic phases at 200 steps"
        )

    started = time.time()
    reference_path = Path(args.v12_report).expanduser().resolve()
    profile_path = Path(args.profile_cache).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    if not reference_path.is_file() or not profile_path.is_file():
        raise FileNotFoundError("reference report and explicit profile cache must exist")
    reference_report = json.loads(reference_path.read_text(encoding="utf-8"))
    if reference_report.get("city") != "chicago" or reference_report.get(
        "budget_name"
    ) != "smoke":
        raise ValueError("structured comparator requires the Chicago v12 smoke report")
    warm = _v12_warm_start(reference_report)
    expected_profile_sha = str(
        reference_report["provenance"]["threshold_cache"]["sha256"]
    )
    actual_profile_sha = _sha256(profile_path)
    if actual_profile_sha != expected_profile_sha:
        raise ValueError(
            "explicit legacy cache does not match the v12 comparator provenance"
        )
    budget = dict(reference_report["budget"])
    target_action = np.asarray(warm["target_action"], dtype=np.float64)
    target_coordinates = np.asarray(warm["target_coordinates"], dtype=np.float64)
    np.testing.assert_allclose(
        ActionTransform.market_controls().physical_to_coordinates(target_action),
        target_coordinates,
        rtol=0.0,
        atol=1.0e-12,
    )
    fit_seeds = [int(seed) for seed in warm["rollout_seeds"]]
    gate_seeds = [int(seed) for seed in warm["critic_gate_seeds"]]
    if not set(fit_seeds).isdisjoint(gate_seeds):
        raise RuntimeError("fit and held-out TRAIN critic-gate seeds overlap")

    calibration_path, trip_bank_path, _ = _dataset_paths("chicago")
    calibration = load_calibration_json(
        str(calibration_path), expected_city="Chicago"
    )
    core, provenance = build_v28_core(
        city="chicago",
        calibration_path=calibration_path,
        trip_bank_path=trip_bank_path,
        threshold_cache_path=profile_path,
        seed=int(reference_report.get("learner_config", {}).get("seed", 28_028)),
        profile_pool_size=int(budget["profile_pool_size"]),
        simulation_sample_cap=int(budget["customers_per_block"]),
        shock_config=_shock_config("l0", calibration),
        steps_per_day=6,
        rider_memory=True,
        firm2_mode="static",
    )
    env_config = DailyEnvironmentConfig(
        episode_days=int(budget["episode_days"]),
        customers_per_block=int(budget["customers_per_block"]),
        steps_per_day=6,
    )

    def training_environment_factory():
        return ContinuousMarketEnv(
            core,
            config=env_config,
            scenario_factory=None,
            profile_split="train",
            shock_sampling_mode="training_openings",
        )

    probe = training_environment_factory()
    action_transform = ActionTransform.market_controls()
    agent = RecurrentSACAgent(
        observation_dim=probe.observation_dim,
        action_transform=action_transform,
        config=RecurrentSACConfig(
            hidden_dim=128,
            replay_capacity=100_000,
            batch_size=int(budget["batch_size"]),
            burn_in=int(budget["burn_in"]),
            learn_length=int(budget["learn_length"]),
            gamma=1.0,
            actor_learning_rate=1.0e-4,
            critic_learning_rate=3.0e-4,
            temperature_learning_rate=3.0e-4,
            polyak_tau=0.005,
            device=str(args.device),
            seed=int(reference_report.get("learner_config", {}).get("seed", 28_028)),
            observation_identity_indices=_observation_identity_indices(
                probe.observation_names
            ),
        ),
    )
    radius = float(warm["local_action_support"]["radius_coordinates"][0])
    support = _local_action_support(action_transform, target_coordinates, radius)
    if support["actions"] != warm["local_action_support"]["actions"]:
        raise RuntimeError("reconstructed support differs from the v12 support")
    agent.configure_critic_action_reference(
        target_coordinates,
        np.asarray(support["radius_coordinates"], dtype=np.float64),
    )
    actor_before = copy.deepcopy(agent.actor.state_dict())
    temperature_before = agent.log_temperature.detach().clone()

    fit_days = _stratified_deviation_days(fit_seeds, int(budget["episode_days"]))
    gate_days = _stratified_deviation_days(gate_seeds, int(budget["episode_days"]))
    group_id = "structured_critic_train_fit"
    for support_row in support["actions"]:
        for seed in fit_seeds:
            episode, _, _ = _local_deviation_episode(
                training_environment_factory,
                target_action,
                np.asarray(support_row["action"], dtype=np.float64),
                seed,
                fit_days[seed],
                metadata={
                    "group_id": group_id,
                    "support_index": int(support_row["index"]),
                    "support_label": str(support_row["label"]),
                    "offset_coordinates": list(support_row["offset"]),
                },
            )
            agent.add_episode(episode, actions_are_physical=True)
    coverage = _verify_local_replay_coverage(
        agent.replay,
        group_id=group_id,
        support=support,
        seeds=fit_seeds,
        center_coordinates=target_coordinates,
    )
    if not coverage["passed"]:
        raise RuntimeError(f"local replay coverage failed: {coverage['failures']}")

    gate_episodes: List[Episode] = []
    gate_support_indices: List[int] = []
    gate_episode_seeds: List[int] = []
    for support_row in support["actions"]:
        for seed in gate_seeds:
            episode, _, _ = _local_deviation_episode(
                training_environment_factory,
                target_action,
                np.asarray(support_row["action"], dtype=np.float64),
                seed,
                gate_days[seed],
                metadata={
                    "group_id": "structured_critic_heldout_train_gate",
                    "support_index": int(support_row["index"]),
                    "support_label": str(support_row["label"]),
                    "offset_coordinates": list(support_row["offset"]),
                },
            )
            gate_episodes.append(
                Episode(
                    episode.observations,
                    action_transform.physical_to_coordinates(episode.actions).astype(
                        np.float32
                    ),
                    episode.rewards,
                    episode.dones,
                    metadata=episode.metadata,
                )
            )
            gate_support_indices.append(int(support_row["index"]))
            gate_episode_seeds.append(seed)

    before = agent.critic_validation(gate_episodes)
    fit_report = agent.fit_critics_monte_carlo(
        steps=200,
        paired_advantage_steps=200,
        paired_advantage_weight=1.0,
    )
    after = agent.critic_validation(gate_episodes)
    gate = _critic_ranking_gate(
        support,
        coverage,
        before,
        after,
        gate_support_indices,
        gate_episode_seeds,
    )
    actor_unchanged = all(
        torch.equal(actor_before[name], value)
        for name, value in agent.actor.state_dict().items()
    )
    temperature_unchanged = bool(
        torch.equal(temperature_before, agent.log_temperature.detach())
    )
    v12_after = warm["critic_validation_after"]
    report = {
        "schema": "ride_response_v28_structured_critic_probe_v1",
        "scope": "critic_only_train_split_diagnostic",
        "city": "chicago",
        "reference_v12": {
            "path": str(reference_path),
            "sha256": _sha256(reference_path),
            "paired_metrics_recomputed": _paired_metrics(v12_after["episodes"]),
            "recorded_gate": warm["critic_coverage_and_ranking_gate"],
        },
        "profile_cache": {
            "path": str(profile_path),
            "sha256": actual_profile_sha,
            "label": "legacy_bounded_cache_explicit_v12_comparator_only",
            "native_v28_cache_used": False,
        },
        "provenance": provenance,
        "split_contract": {
            "fit_profile_split": "train",
            "heldout_gate_profile_split": "train",
            "fit_seeds": fit_seeds,
            "heldout_train_gate_seeds": gate_seeds,
            "seed_streams_disjoint": True,
            "validation_environment_constructed": False,
            "validation_opened": False,
            "test_environment_constructed": False,
            "test_opened": False,
        },
        "execution_contract": {
            "actor_training": False,
            "actor_parameter_mutation": not actor_unchanged,
            "sac_updates": 0,
            "temperature_updates": 0,
            "temperature_mutation": not temperature_unchanged,
            "economic_policy_validation": False,
            "promotion_eligible": False,
            "purpose": "critic identifiability only; not an L0 learner gate",
        },
        "target_action": target_action.tolist(),
        "target_coordinates": target_coordinates.tolist(),
        "support": support,
        "critic_action_reference_contract": agent.critic_action_reference_contract(),
        "replay_coverage": coverage,
        "critic_validation_before": before,
        "critic_fit": fit_report,
        "critic_validation_after": after,
        "heldout_train_gate": gate,
        "structured_paired_metrics": _paired_metrics(after["episodes"]),
        "curvature_recovery": _curvature_recovery(support, after),
        "elapsed_seconds": float(time.time() - started),
    }
    out_dir.mkdir(parents=True, exist_ok=False)
    destination = out_dir / "report.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(destination),
                "gate_passed": bool(gate["passed"]),
                "anti_prior_passed": bool(gate["anti_prior_baseline_passed"]),
                "structured_paired": report["structured_paired_metrics"],
                "elapsed_seconds": report["elapsed_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
