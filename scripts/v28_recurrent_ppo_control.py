#!/usr/bin/env python3
"""Validation-only recurrent-PPO control for the v28 learner ladder.

This runner is deliberately separate from ``v28_learner_ladder.py``.  It
uses the same calibrated market, rider splits, physical action transform,
stage scenarios, and per-stage episode/day/customer budgets, but trains a
strictly on-policy recurrent PPO control:

* every behavior draw is made once per day (no option persistence);
* only complete episodes enter an update;
* sampled behavior log probabilities are stored and reconstructed before the
  first optimizer step;
* ``gamma=1`` is paired with the environment's finite-horizon time-to-go;
* invalid uncapped actions terminate with a numerical penalty and are never
  clipped or replaced;
* train profiles are used for learning, two disjoint validation seed sets are
  used for comparator fitting and gates, and this program has no test-split
  evaluation path.

The control retains one policy across L0--L3 and audits the final frozen
policy on every completed rung.  A pilot is the intended use; this module
does not start work when imported.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from calibration_utils import load_calibration_json  # noqa: E402
from continuous_benchmarks import (  # noqa: E402
    CEMConfig,
    CEMOptimizer,
    FixedPolicy,
    PublicSchedulePolicy,
    evaluate_clairvoyant_event_schedule,
    evaluate_policy,
    evaluate_uncapped_price_tail,
    optimize_clairvoyant_event_action,
    optimize_static_policy,
    optimize_static_tariff,
    paired_difference,
)
from continuous_market_env import (  # noqa: E402
    ContinuousMarketEnv,
    DailyEnvironmentConfig,
    build_v28_core,
)
from recurrent_ppo import (  # noqa: E402
    ActionTransform,
    Episode,
    EpisodeBuilder,
    RecurrentPPOAgent,
    RecurrentPPOConfig,
    RecurrentPPONumericalActionError,
)
from scripts.v28_learner_ladder import (  # noqa: E402
    BUDGETS,
    STAGE_OFFSETS,
    Budget,
    _dataset_paths,
    _observation_identity_indices,
    _scenario_factory,
    _shock_config,
)


CONTROL_STAGES = ("l0", "l1", "l2", "l3")
NUMERICAL_POLICY_PENALTY = -1.0e6
NUMERICAL_CANDIDATE_PENALTY = -1.0e12
VALIDATION_GATE_OFFSET = 15_000_003


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _stage_seeds(
    seed: int,
    stage: str,
    count: int,
    purpose: str,
) -> List[int]:
    """Return disjoint train/validation-fit/validation-gate streams."""
    offsets = {
        "train": 0,
        "validation_fit": 10_000_000,
        "validation_gate": VALIDATION_GATE_OFFSET,
        "tail": 30_000_000,
    }
    if purpose not in offsets:
        raise ValueError(
            "PPO control exposes only train, validation_fit, validation_gate, and tail seeds"
        )
    base = int(seed) + int(STAGE_OFFSETS[stage]) + offsets[purpose]
    return [base + 104_729 * index for index in range(int(count))]


def _make_environment_factory(
    core: Any,
    config: DailyEnvironmentConfig,
    scenario_factory: Optional[Callable[[int], Any]],
    *,
    profile_split: str,
    training: bool,
) -> Callable[[], ContinuousMarketEnv]:
    """Construct only the two split roles authorized for this control.

    There is intentionally no test factory.  Keeping the rejection here makes
    accidental held-out access a runtime error even if a caller bypasses the
    command-line interface.
    """
    if profile_split not in {"train", "validation"}:
        raise ValueError(
            "recurrent-PPO control may construct only train or validation environments"
        )
    expected = "train" if training else "validation"
    if profile_split != expected:
        raise ValueError(
            f"environment role mismatch: training={training} requires {expected!r} split"
        )

    def factory() -> ContinuousMarketEnv:
        return ContinuousMarketEnv(
            core,
            config=config,
            scenario_factory=scenario_factory,
            profile_split=profile_split,
            shock_sampling_mode="training_openings" if training else "natural",
        )

    return factory


def collect_on_policy_episode(
    agent: RecurrentPPOAgent,
    environment: Any,
    *,
    seed: int,
    deterministic: bool = False,
    numerical_penalty: float = NUMERICAL_POLICY_PENALTY,
) -> Tuple[Episode, np.ndarray, Dict[str, Any]]:
    """Collect one complete episode and its exact behavior log probabilities."""
    observation = environment.reset(int(seed))
    builder = EpisodeBuilder(observation)
    behavior_log_probabilities: List[float] = []
    hidden = None
    previous_action: Optional[np.ndarray] = None
    previous_reward = 0.0
    episode_return = 0.0
    shock_days = 0
    extreme_action_days = 0
    numerical_invalid_actions = 0
    daily_actions: List[List[float]] = []
    done = False
    days = 0
    started = time.time()

    while not done:
        try:
            action, hidden, action_info = agent.act(
                observation,
                previous_action=previous_action,
                previous_reward=previous_reward,
                hidden=hidden,
                deterministic=bool(deterministic),
            )
            coordinates = np.asarray(
                action_info["policy_coordinates"], dtype=np.float64
            )
            behavior_log_probability = float(action_info["log_probability"])
        except RecurrentPPONumericalActionError as exc:
            # The failed sample is still an exact behavior-policy draw.  It is
            # stored in native coordinates and terminates rather than being
            # projected onto an arbitrary feasible boundary.
            coordinates = np.asarray(exc.coordinates, dtype=np.float64)
            behavior_log_probability = float(exc.log_probability)
            next_observation = np.asarray(observation, dtype=np.float32).copy()
            reward = float(numerical_penalty)
            done = True
            info: Mapping[str, Any] = {
                "active_shocks": (),
                "numerical_invalid_action": True,
                "numerical_error": str(exc),
            }
            action_info = dict(exc.action_info)
            daily_actions.append([float("nan")] * agent.action_dim)
            numerical_invalid_actions += 1
        else:
            try:
                next_observation, reward, done, info = environment.step(action)
            except (FloatingPointError, OverflowError) as exc:
                # Environment-level numerical rejection follows the same
                # no-clipping terminal contract.  The actor draw and log
                # probability above remain exactly the executed proposal.
                agent.nonfinite_action_rejections += 1
                numerical_invalid_actions += 1
                next_observation = np.asarray(observation, dtype=np.float32).copy()
                reward = float(numerical_penalty)
                done = True
                info = {
                    "active_shocks": (),
                    "numerical_invalid_action": True,
                    "numerical_error": f"{type(exc).__name__}: {exc}",
                }
            daily_actions.append(np.asarray(action, dtype=np.float64).tolist())

        builder.append(coordinates, reward, next_observation, done)
        behavior_log_probabilities.append(behavior_log_probability)
        observation = np.asarray(next_observation, dtype=np.float32)
        if not bool(info.get("numerical_invalid_action", False)):
            previous_action = np.asarray(action, dtype=np.float64).copy()
        previous_reward = float(reward)
        episode_return += float(reward)
        shock_days += int(bool(info.get("active_shocks")))
        extreme_action_days += int(bool(action_info.get("extreme_action", False)))
        days += 1

    episode = builder.finish()
    log_probabilities = np.asarray(behavior_log_probabilities, dtype=np.float32)
    if episode.length != log_probabilities.shape[0] or episode.dones[-1] < 0.5:
        raise RuntimeError("on-policy collector did not produce one complete episode")
    return episode, log_probabilities, {
        "seed": int(seed),
        "return": float(episode_return),
        "days": int(days),
        "shock_days": int(shock_days),
        "extreme_action_days": int(extreme_action_days),
        "numerical_invalid_actions": int(numerical_invalid_actions),
        "behavior_log_probabilities_stored": int(log_probabilities.size),
        "daily_action_draws": int(days),
        "option_persistence": False,
        "action_persistence_days": 1,
        "daily_actions": daily_actions,
        "elapsed_seconds": float(time.time() - started),
    }


def train_ppo_stage(
    agent: RecurrentPPOAgent,
    environment_factory: Callable[[], Any],
    seeds: Sequence[int],
) -> List[Dict[str, Any]]:
    """Train from complete on-policy episodes, flushing at the stage boundary."""
    if agent.pending_transitions:
        raise RuntimeError("PPO stage began with stale on-policy transitions")
    logs: List[Dict[str, Any]] = []
    for episode_index, episode_seed in enumerate(seeds):
        episode, log_probabilities, row = collect_on_policy_episode(
            agent,
            environment_factory(),
            seed=int(episode_seed),
            deterministic=False,
        )
        agent.add_episode(
            episode,
            actions_are_physical=False,
            behavior_log_probabilities=log_probabilities,
        )
        update: Dict[str, float] = {}
        if agent.can_update():
            update = agent.update()
        row.update(
            {
                "episode": int(episode_index),
                "update_after_episode": bool(update),
                **update,
            }
        )
        logs.append(row)
        print(
            f"  PPO episode {episode_index + 1:4d}/{len(seeds)} "
            f"return={row['return']:+.3f} shocks={row['shock_days']:2d} "
            f"update={'yes' if update else 'no':>3s} "
            f"[{row['elapsed_seconds']:.1f}s]",
            flush=True,
        )
    # Leftover episodes were collected under one unchanged behavior policy;
    # consume them now so no on-policy data crosses a stage distribution.
    if agent.pending_transitions:
        final_update = agent.update()
        logs[-1]["stage_boundary_update"] = final_update
    return logs


def _rollout_ppo_agent(
    environment: Any,
    agent: RecurrentPPOAgent,
    *,
    seed: int,
) -> Dict[str, Any]:
    observation = environment.reset(int(seed))
    hidden = None
    previous_action = None
    previous_reward = 0.0
    total_reward = 0.0
    daily_rewards: List[float] = []
    actions: List[List[float]] = []
    infos: List[Dict[str, Any]] = []
    done = False
    day = 0
    while not done:
        try:
            action, hidden, actor_info = agent.act(
                observation,
                previous_action=previous_action,
                previous_reward=previous_reward,
                hidden=hidden,
                deterministic=True,
            )
            next_observation, reward, done, info = environment.step(action)
        except RecurrentPPONumericalActionError as exc:
            action = np.full(agent.action_dim, np.nan, dtype=np.float64)
            actor_info = dict(exc.action_info)
            next_observation = np.asarray(observation, dtype=np.float32).copy()
            reward = NUMERICAL_POLICY_PENALTY
            done = True
            info = {
                "active_shocks": (),
                "numerical_invalid_action": True,
                "numerical_error": str(exc),
            }
        except (FloatingPointError, OverflowError) as exc:
            next_observation = np.asarray(observation, dtype=np.float32).copy()
            reward = NUMERICAL_POLICY_PENALTY
            done = True
            info = {
                "active_shocks": (),
                "numerical_invalid_action": True,
                "numerical_error": f"{type(exc).__name__}: {exc}",
            }
        total_reward += float(reward)
        daily_rewards.append(float(reward))
        actions.append(np.asarray(action, dtype=np.float64).tolist())
        infos.append({**dict(info), "actor": actor_info})
        observation = np.asarray(next_observation, dtype=np.float32)
        if np.all(np.isfinite(action)):
            previous_action = np.asarray(action, dtype=np.float64).copy()
        previous_reward = float(reward)
        day += 1
    return {
        "seed": int(seed),
        "return": float(total_reward),
        "days": int(day),
        "daily_rewards": daily_rewards,
        "actions": actions,
        "infos": infos,
    }


def evaluate_ppo_agent(
    environment_factory: Callable[[], Any],
    agent: RecurrentPPOAgent,
    seeds: Sequence[int],
) -> Dict[str, Any]:
    episodes = [
        _rollout_ppo_agent(environment_factory(), agent, seed=int(seed))
        for seed in seeds
    ]
    values = np.asarray([row["return"] for row in episodes], dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "standard_error": (
            float(values.std(ddof=1) / np.sqrt(values.size))
            if values.size > 1
            else float("nan")
        ),
        "median": float(np.median(values)),
        "values": values.tolist(),
        "episodes": episodes,
    }


def _optimize_public_event_action(
    environment_factory: Callable[[], Any],
    action_transform: ActionTransform,
    seeds: Sequence[int],
    public_indicator: Callable[[np.ndarray], bool],
    normal_action: Sequence[float],
    *,
    config: CEMConfig,
) -> Dict[str, Any]:
    """Optimize only the public-event action around the best fixed 3-D rule.

    Allowing the schedule optimizer to refit its normal action makes its gain
    partly a second fixed-policy search and invalidates the adaptivity
    denominator.  The normal regime is therefore held exactly at the fitted
    full-control rule while CEM searches the uncapped event action.
    """
    normal = np.asarray(normal_action, dtype=np.float64).reshape(
        action_transform.dim
    )
    initial_coordinates = action_transform.physical_to_coordinates(normal)
    optimizer = CEMOptimizer(action_transform.dim, config=copy.copy(config))

    def objective(coordinates: np.ndarray) -> float:
        try:
            event = action_transform.coordinates_to_physical(coordinates)
            return float(
                evaluate_policy(
                    environment_factory,
                    lambda: PublicSchedulePolicy(
                        normal, event, public_indicator
                    ),
                    seeds,
                )["mean"]
            )
        except (FloatingPointError, OverflowError):
            return float(NUMERICAL_CANDIDATE_PENALTY)

    result = optimizer.optimize(objective, initial_mean=initial_coordinates)
    event_action = action_transform.coordinates_to_physical(result["coordinates"])
    result["normal_action"] = normal.tolist()
    result["event_action"] = event_action.tolist()
    result["seeds"] = [int(seed) for seed in seeds]
    result["normal_regime_fixed_to"] = "best_fixed_full_3d_rule"
    result["adaptivity_denominator_contract"] = (
        "public schedule minus best fixed full-3D rule"
    )
    return result


def _adaptivity_statistics(
    stage: str, paired: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Any]:
    """Compute adaptivity only relative to the best fixed full-control rule."""
    opportunity_name = "oracle_minus_rule" if stage == "l2" else "schedule_minus_rule"
    opportunity = paired[opportunity_name]
    learner_gain = paired["ppo_minus_rule"]
    denominator = float(opportunity["mean"])
    standard_error = float(opportunity["standard_error"])
    identified = bool(
        denominator > 0.0
        and np.isfinite(standard_error)
        and denominator > 3.0 * standard_error
    )
    return {
        "adaptive_opportunity_baseline": "best_fixed_full_3d_rule",
        "adaptive_opportunity_pair": opportunity_name,
        "adaptive_opportunity_identified": identified,
        "adaptivity_index": (
            float(learner_gain["mean"]) / denominator if identified else None
        ),
    }


def _compact_evaluation(evaluation: Mapping[str, Any]) -> Dict[str, Any]:
    episodes = []
    for episode in evaluation.get("episodes", []):
        actions = np.asarray(episode.get("actions", []), dtype=np.float64)
        infos = list(episode.get("infos", []))
        finite_rows = (
            actions[np.all(np.isfinite(actions), axis=1)]
            if actions.ndim == 2 and actions.size
            else np.empty((0, 3), dtype=np.float64)
        )
        episodes.append(
            {
                "seed": int(episode["seed"]),
                "return": float(episode["return"]),
                "days": int(episode["days"]),
                "oracle_event_days": [
                    int(day) for day in episode.get("oracle_event_days", [])
                ],
                "daily_actions": actions.tolist() if actions.size else [],
                "action_mean": (
                    finite_rows.mean(axis=0).tolist() if finite_rows.size else []
                ),
                "active_shock_path": [
                    bool(info.get("active_shocks")) for info in infos
                ],
                "dates": [info.get("date") for info in infos],
                "numerical_invalid_days": int(
                    sum(bool(info.get("numerical_invalid_action")) for info in infos)
                ),
            }
        )
    return {
        "mean": float(evaluation["mean"]),
        "standard_error": float(evaluation["standard_error"]),
        "median": float(evaluation["median"]),
        "values": [float(value) for value in evaluation["values"]],
        "episodes": episodes,
    }


def _evaluate_fitted_policies(
    stage: str,
    agent: RecurrentPPOAgent,
    environment_factory: Callable[[], Any],
    gate_seeds: Sequence[int],
    fits: Mapping[str, Any],
) -> Dict[str, Any]:
    probe = environment_factory()
    static_action = np.asarray(fits["best_static_tariff"]["action"], dtype=np.float64)
    rule_action = np.asarray(fits["two_timescale_rule"]["action"], dtype=np.float64)
    normal_action = np.asarray(fits["public_schedule"]["normal_action"], dtype=np.float64)
    event_action = np.asarray(fits["public_schedule"]["event_action"], dtype=np.float64)
    if not np.allclose(normal_action, rule_action, rtol=0.0, atol=1e-12):
        raise RuntimeError(
            "public schedule normal action must equal the fitted full-3D rule"
        )
    static = evaluate_policy(
        environment_factory, lambda: FixedPolicy(static_action), gate_seeds
    )
    rule = evaluate_policy(
        environment_factory, lambda: FixedPolicy(rule_action), gate_seeds
    )
    schedule = evaluate_policy(
        environment_factory,
        lambda: PublicSchedulePolicy(
            normal_action, event_action, probe.public_event_indicator
        ),
        gate_seeds,
    )
    learned = evaluate_ppo_agent(environment_factory, agent, gate_seeds)
    policies: Dict[str, Any] = {
        "best_static_tariff": _compact_evaluation(static),
        "two_timescale_rule": _compact_evaluation(rule),
        "public_schedule": _compact_evaluation(schedule),
        "recurrent_ppo": _compact_evaluation(learned),
    }
    paired: Dict[str, Any] = {
        "ppo_minus_static": paired_difference(learned, static),
        "ppo_minus_rule": paired_difference(learned, rule),
        "ppo_minus_schedule": paired_difference(learned, schedule),
        "schedule_minus_static": paired_difference(schedule, static),
        "schedule_minus_rule": paired_difference(schedule, rule),
    }
    if stage == "l2":
        oracle_fit = fits["clairvoyant_event_window_oracle"]
        oracle_normal = np.asarray(
            oracle_fit["normal_action"], dtype=np.float64
        )
        if not np.allclose(oracle_normal, rule_action, rtol=0.0, atol=1e-12):
            raise RuntimeError(
                "clairvoyant oracle normal action must equal the fitted full-3D rule"
            )
        oracle = evaluate_clairvoyant_event_schedule(
            environment_factory,
            oracle_fit["normal_action"],
            oracle_fit["event_action"],
            probe.clairvoyant_event_days,
            gate_seeds,
        )
        policies["clairvoyant_event_window_oracle"] = _compact_evaluation(oracle)
        policies["clairvoyant_event_window_oracle"].update(
            {
                "fair_comparison": False,
                "information": "latent shortage event/recovery window known by seed",
            }
        )
        paired["oracle_minus_static"] = paired_difference(oracle, static)
        paired["oracle_minus_rule"] = paired_difference(oracle, rule)
        paired["ppo_minus_oracle"] = paired_difference(learned, oracle)
    adaptivity = _adaptivity_statistics(stage, paired)
    return {
        "split": "validation_gate",
        "validation_gate_seeds": [int(seed) for seed in gate_seeds],
        "profile_split": "validation",
        "fits": copy.deepcopy(dict(fits)),
        "policies": policies,
        "paired": paired,
        **adaptivity,
    }


def evaluate_stage(
    stage: str,
    agent: RecurrentPPOAgent,
    environment_factory: Callable[[], Any],
    fit_seeds: Sequence[int],
    gate_seeds: Sequence[int],
    *,
    cem_config: CEMConfig,
) -> Dict[str, Any]:
    """Fit comparators on one validation stream and gate on another."""
    probe = environment_factory()
    transform = agent.action_transform
    static_fit = optimize_static_tariff(
        environment_factory, transform, fit_seeds, config=copy.copy(cem_config)
    )
    static_action = np.asarray(static_fit["action"], dtype=np.float64)
    rule_fit = optimize_static_policy(
        environment_factory,
        transform,
        fit_seeds,
        config=copy.copy(cem_config),
        initial_action=static_action,
    )
    rule_action = np.asarray(rule_fit["action"], dtype=np.float64)
    schedule_fit = _optimize_public_event_action(
        environment_factory,
        transform,
        fit_seeds,
        probe.public_event_indicator,
        rule_action,
        config=copy.copy(cem_config),
    )
    fits: Dict[str, Any] = {
        "best_static_tariff": static_fit,
        "two_timescale_rule": rule_fit,
        "public_schedule": schedule_fit,
    }
    if stage == "l2":
        oracle_fit = optimize_clairvoyant_event_action(
            environment_factory,
            transform,
            fit_seeds,
            probe.clairvoyant_event_days,
            rule_action,
            config=copy.copy(cem_config),
        )
        oracle_fit["normal_regime_fixed_to"] = "best_fixed_full_3d_rule"
        oracle_fit["adaptivity_denominator_contract"] = (
            "clairvoyant oracle minus best fixed full-3D rule"
        )
        fits["clairvoyant_event_window_oracle"] = oracle_fit
    result = _evaluate_fitted_policies(
        stage, agent, environment_factory, gate_seeds, fits
    )
    result.update(
        {
            "validation_fit_seeds": [int(seed) for seed in fit_seeds],
            "split_contract": {
                "comparator_fit": "validation profiles + validation_fit seeds",
                "rung_gate": "validation profiles + disjoint validation_gate seeds",
                "test_opened": False,
            },
        }
    )
    return result


def gate_stage(stage: str, evaluation: Mapping[str, Any]) -> Dict[str, Any]:
    policies = evaluation["policies"]
    paired = evaluation["paired"]
    if stage == "l0":
        baseline = float(policies["two_timescale_rule"]["mean"])
        tolerance = 0.01 * max(1e-9, abs(baseline))
        delta = float(paired["ppo_minus_rule"]["mean"])
        return {
            "name": "L0 recurrent-PPO optimizer sanity",
            "passed": bool(delta >= -tolerance),
            "criterion": "PPO no more than 1% below the best fixed full-3D rule",
            "baseline": "best_fixed_full_3d_rule",
            "delta": delta,
            "tolerance": tolerance,
            "tariff_static_diagnostic": {
                "mean": float(policies["best_static_tariff"]["mean"]),
                "ppo_minus_static": paired["ppo_minus_static"],
            },
        }
    if stage == "l1":
        index = evaluation.get("adaptivity_index")
        identified = bool(evaluation["adaptive_opportunity_identified"])
        return {
            "name": "L1 recurrent-PPO public shock adaptation",
            "passed": bool(identified and index is not None and float(index) >= 0.50),
            "criterion": (
                "recover >=50% of significant public-schedule gain over the "
                "best fixed full-3D rule"
            ),
            "opportunity_identified": identified,
            "adaptivity_index": index,
        }
    if stage == "l2":
        index = evaluation.get("adaptivity_index")
        identified = bool(evaluation["adaptive_opportunity_identified"])
        return {
            "name": "L2 recurrent-PPO latent driver-supply shock",
            "passed": bool(identified and index is not None and float(index) >= 0.30),
            "criterion": (
                "recover >=30% of significant clairvoyant event-window gain over "
                "the best fixed full-3D rule"
            ),
            "opportunity_identified": identified,
            "adaptivity_index": index,
            "oracle_is_fair_policy": False,
        }
    if stage == "l3":
        static = paired["ppo_minus_static"]
        rule = paired["ppo_minus_rule"]
        return {
            "name": "L3 recurrent-PPO strategic-rival response",
            "passed": bool(float(static["mean"]) > 0.0 and float(rule["mean"]) > 0.0),
            "criterion": "beat continuous best static and tuned two-timescale rule",
            "ppo_minus_static": static,
            "ppo_minus_rule": rule,
        }
    raise ValueError(f"unknown recurrent-PPO control stage {stage!r}")


def _plot_control_report(report: Mapping[str, Any], destination: Path) -> Optional[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    stages = [
        stage
        for stage in report.get("completed_stages", [])
        if report.get("stage_results", {}).get(stage, {}).get(
            "validation_evaluation"
        )
    ]
    if not stages:
        return None
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for stage in stages:
        returns = np.asarray(
            [row["return"] for row in report["stage_results"][stage]["training"]],
            dtype=float,
        )
        axes[0].plot(np.arange(1, returns.size + 1), returns, marker="o", label=stage.upper())
    axes[0].set_title("Recurrent-PPO on-policy training")
    axes[0].set_xlabel("Episode within stage")
    axes[0].set_ylabel("Episode return")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    labels: List[str] = []
    means: List[float] = []
    errors: List[float] = []
    for stage in stages:
        policies = report["stage_results"][stage]["validation_evaluation"]["policies"]
        for name in ("best_static_tariff", "two_timescale_rule", "public_schedule", "recurrent_ppo"):
            labels.append(f"{stage.upper()}\n{name.replace('_', ' ')}")
            means.append(float(policies[name]["mean"]))
            error = float(policies[name]["standard_error"])
            errors.append(error if np.isfinite(error) else 0.0)
    x = np.arange(len(labels))
    axes[1].bar(x, means, yerr=errors, capsize=2)
    axes[1].set_xticks(x, labels, rotation=65, ha="right", fontsize=8)
    axes[1].set_title("Disjoint validation-gate comparison")
    axes[1].set_ylabel("Episode return")
    axes[1].grid(axis="y", alpha=0.25)
    figure.suptitle("v28 recurrent-PPO pilot control — no held-out test access")
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return str(destination)


def _plot_shock_actions(report: Mapping[str, Any], destination: Path) -> Optional[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    before, after = 3, 5
    aligned: Dict[str, List[np.ndarray]] = {}
    for stage in report.get("completed_stages", []):
        evaluation = report.get("stage_results", {}).get(stage, {}).get(
            "validation_evaluation"
        )
        if not evaluation:
            continue
        episodes = evaluation["policies"][
            "recurrent_ppo"
        ]["episodes"]
        rows = []
        for episode in episodes:
            action = np.asarray(episode.get("daily_actions", []), dtype=float)
            shock = np.asarray(episode.get("active_shock_path", []), dtype=bool)
            onset = np.flatnonzero(shock)
            if action.ndim != 2 or action.shape[1] != 3 or onset.size == 0:
                continue
            center = int(onset[0])
            window = np.full((before + after + 1, 3), np.nan)
            start, stop = max(0, center - before), min(action.shape[0], center + after + 1)
            left = before - (center - start)
            window[left : left + stop - start] = action[start:stop]
            rows.append(window)
        if rows:
            aligned[stage] = rows
    if not aligned:
        return None
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharex=True)
    x = np.arange(-before, after + 1)
    titles = ("Base tariff", "Surge gain", "Driver incentive")
    for stage, rows in aligned.items():
        values = np.stack(rows)
        mean = np.nanmean(values, axis=0)
        for index, axis in enumerate(axes):
            axis.plot(x, mean[:, index], marker="o", label=stage.upper())
    for index, axis in enumerate(axes):
        axis.axvline(0, color="black", linestyle="--", linewidth=1)
        axis.set_title(titles[index])
        axis.set_xlabel("Days from shock onset")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Deterministic physical action")
    axes[-1].legend()
    figure.suptitle("Validation-only recurrent-PPO shock response")
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return str(destination)


def _resolve(value: Optional[int], default: int) -> int:
    return int(default if value is None else value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--city", choices=("nyc", "boston", "chicago"), default="chicago")
    parser.add_argument("--budget", choices=tuple(BUDGETS), default="pilot")
    parser.add_argument("--stages", default="l0,l1,l2,l3")
    parser.add_argument("--seed", type=int, default=38_028)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default="artifacts/v28_rppo_control/pilot")
    parser.add_argument("--calibration", default="")
    parser.add_argument("--trip-bank", default="")
    parser.add_argument("--profile-cache", default="")
    parser.add_argument("--train-episodes", type=int)
    parser.add_argument("--episode-days", type=int)
    parser.add_argument("--evaluation-seeds", type=int)
    parser.add_argument("--benchmark-train-seeds", type=int)
    parser.add_argument("--customers", type=int)
    parser.add_argument("--profile-pool-size", type=int)
    parser.add_argument("--ppo-rollout-episodes", type=int)
    parser.add_argument("--ppo-update-epochs", type=int, default=8)
    parser.add_argument("--cem-population", type=int)
    parser.add_argument("--cem-iterations", type=int)
    parser.add_argument(
        "--diagnostic-continue",
        action="store_true",
        help="continue after a failed structural/rung gate; never makes the run promotable",
    )
    args = parser.parse_args()

    selected_stages = tuple(
        item.strip().lower() for item in args.stages.split(",") if item.strip()
    )
    canonical = tuple(stage for stage in CONTROL_STAGES if stage in selected_stages)
    if not selected_stages or selected_stages != canonical or len(set(selected_stages)) != len(selected_stages):
        raise ValueError(f"stages must be a unique ordered subset of {CONTROL_STAGES}")

    preset: Budget = BUDGETS[args.budget]
    budget = Budget(
        train_episodes=_resolve(args.train_episodes, preset.train_episodes),
        episode_days=_resolve(args.episode_days, preset.episode_days),
        warmup_episodes=0,
        evaluation_seeds=_resolve(args.evaluation_seeds, preset.evaluation_seeds),
        benchmark_train_seeds=_resolve(
            args.benchmark_train_seeds, preset.benchmark_train_seeds
        ),
        customers_per_block=_resolve(args.customers, preset.customers_per_block),
        profile_pool_size=_resolve(args.profile_pool_size, preset.profile_pool_size),
        batch_size=preset.batch_size,
        burn_in=preset.burn_in,
        learn_length=preset.learn_length,
        updates_per_day=preset.updates_per_day,
        cem_population=_resolve(args.cem_population, preset.cem_population),
        cem_iterations=_resolve(args.cem_iterations, preset.cem_iterations),
    )
    default_rollout_episodes = {"smoke": 1, "pilot": 4, "full": 8}[args.budget]
    rollout_episodes = _resolve(args.ppo_rollout_episodes, default_rollout_episodes)
    if rollout_episodes <= 0:
        raise ValueError("PPO rollout episodes must be positive")

    default_calibration, default_trip_bank, default_profiles = _dataset_paths(args.city)
    calibration_path = Path(args.calibration or default_calibration)
    trip_bank_path = Path(args.trip_bank or default_trip_bank)
    profile_cache_path = Path(args.profile_cache or default_profiles)
    calibration = load_calibration_json(
        str(calibration_path),
        expected_city={
            "nyc": "New York City",
            "boston": "Boston",
            "chicago": "Chicago",
        }[args.city],
    )
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    estimate = {
        "training_episodes": int(len(selected_stages) * budget.train_episodes),
        "training_environment_days": int(
            len(selected_stages) * budget.train_episodes * budget.episode_days
        ),
        "training_market_blocks": int(
            len(selected_stages)
            * budget.train_episodes
            * budget.episode_days
            * 6
        ),
        "training_profile_row_evaluations": int(
            len(selected_stages)
            * budget.train_episodes
            * budget.episode_days
            * 6
            * budget.customers_per_block
        ),
        "behavior_action_draws": int(
            len(selected_stages) * budget.train_episodes * budget.episode_days
        ),
        "comparison_to_primary_runner": (
            "identical per-stage train episodes, days, customers/block, profile-pool size, "
            "and six market blocks/day; PPO optimizer epochs are algorithm-specific"
        ),
        "test_profile_evaluations": 0,
    }
    print("[PPO control budget] " + json.dumps(estimate), flush=True)

    action_transform = ActionTransform.market_controls()
    agent: Optional[RecurrentPPOAgent] = None
    stage_results: Dict[str, Any] = {}
    retention_specs: Dict[str, Any] = {}
    provenance: Optional[Dict[str, Any]] = None
    environment_contract: Optional[Dict[str, Any]] = None
    started = time.time()

    for stage in selected_stages:
        print(f"\n=== PPO CONTROL {stage.upper()} ===", flush=True)
        stage_started = time.time()
        shock_config = _shock_config(stage, calibration)
        core, stage_provenance = build_v28_core(
            city=args.city,
            calibration_path=calibration_path,
            trip_bank_path=trip_bank_path,
            threshold_cache_path=profile_cache_path,
            seed=int(args.seed) + STAGE_OFFSETS[stage],
            profile_pool_size=int(budget.profile_pool_size),
            simulation_sample_cap=int(budget.customers_per_block),
            shock_config=shock_config,
            steps_per_day=6,
            rider_memory=True,
            firm2_mode=(
                "adaptive_best_response_aggressive" if stage == "l3" else "static"
            ),
        )
        env_config = DailyEnvironmentConfig(
            episode_days=int(budget.episode_days),
            customers_per_block=int(budget.customers_per_block),
            steps_per_day=6,
        )
        scenario = _scenario_factory(stage, budget.episode_days)
        training_factory = _make_environment_factory(
            core, env_config, scenario, profile_split="train", training=True
        )
        validation_factory = _make_environment_factory(
            core, env_config, scenario, profile_split="validation", training=False
        )
        probe = training_factory()
        if agent is None:
            agent = RecurrentPPOAgent(
                probe.observation_dim,
                action_transform=action_transform,
                config=RecurrentPPOConfig(
                    hidden_dim=128,
                    gamma=1.0,
                    gae_lambda=0.90,
                    clip_ratio=0.20,
                    entropy_coefficient=0.01,
                    actor_learning_rate=3e-4,
                    value_learning_rate=3e-4,
                    update_epochs=int(args.ppo_update_epochs),
                    minibatch_episodes=min(4, rollout_episodes),
                    rollout_transitions=int(rollout_episodes * budget.episode_days),
                    target_kl=0.03,
                    device=str(args.device),
                    seed=int(args.seed),
                    observation_identity_indices=_observation_identity_indices(
                        probe.observation_names
                    ),
                ),
            )
            provenance = stage_provenance
            environment_contract = probe.contract()

        train_seeds = _stage_seeds(
            int(args.seed), stage, budget.train_episodes, "train"
        )
        fit_seeds = _stage_seeds(
            int(args.seed), stage, budget.benchmark_train_seeds, "validation_fit"
        )
        gate_seeds = _stage_seeds(
            int(args.seed), stage, budget.evaluation_seeds, "validation_gate"
        )
        tail_seeds = _stage_seeds(
            int(args.seed), stage, max(2, budget.benchmark_train_seeds), "tail"
        )
        cem_config = CEMConfig(
            population=int(budget.cem_population),
            iterations=int(budget.cem_iterations),
            elite_fraction=0.20,
            initial_std=0.50,
            minimum_std=0.025,
            seed=int(args.seed) + STAGE_OFFSETS[stage] + 1_991,
        )
        price_tail = evaluate_uncapped_price_tail(validation_factory, tail_seeds)
        if not price_tail["passed"] and not args.diagnostic_continue:
            stage_results[stage] = {
                "status": "preflight_failed",
                "price_tail_gate": price_tail,
                "training": [],
                "gate": {
                    "name": "uncapped multidimensional action-tail preflight",
                    "passed": False,
                    "criterion": price_tail["criterion"],
                },
                "provenance": stage_provenance,
                "elapsed_seconds": float(time.time() - stage_started),
            }
            break

        training = train_ppo_stage(agent, training_factory, train_seeds)
        evaluation = evaluate_stage(
            stage,
            agent,
            validation_factory,
            fit_seeds,
            gate_seeds,
            cem_config=cem_config,
        )
        gate = gate_stage(stage, evaluation)
        gate["promotion_eligible"] = bool(
            args.budget in {"pilot", "full"} and not args.diagnostic_continue
        )
        if args.budget == "smoke":
            gate["promotion_eligible"] = False
            gate["statistical_status"] = "underpowered diagnostic only"

        checkpoint = out_dir / f"policy_after_{stage}.pt"
        agent.save(
            checkpoint,
            metadata={
                "stage": stage,
                "city": args.city,
                "budget": args.budget,
                "split_contract": "train learning; validation fit/gate; test never opened",
                "gate": gate,
            },
        )
        stage_results[stage] = {
            "training_seeds": train_seeds,
            "validation_fit_seeds": fit_seeds,
            "validation_gate_seeds": gate_seeds,
            "training": training,
            "price_tail_gate": price_tail,
            "validation_evaluation": evaluation,
            "gate": gate,
            "checkpoint": str(checkpoint),
            "shock_config": shock_config.as_dict(),
            "provenance": stage_provenance,
            "elapsed_seconds": float(time.time() - stage_started),
        }
        retention_specs[stage] = {
            "environment_factory": validation_factory,
            "gate_seeds": tuple(gate_seeds),
            "fits": copy.deepcopy(evaluation["fits"]),
        }
        partial = {
            "schema": "ride_response_v28_recurrent_ppo_control_v1",
            "completed_stages": list(stage_results),
            "stage_results": stage_results,
            "test_split_contract": {"opened": False, "implemented": False},
        }
        (out_dir / "report.partial.json").write_text(
            json.dumps(_jsonable(partial), indent=2, allow_nan=False)
        )
        print(
            f"[{stage}] gate={'PASS' if gate['passed'] else 'FAIL'} "
            f"PPO={evaluation['policies']['recurrent_ppo']['mean']:+.3f} "
            f"static={evaluation['policies']['best_static_tariff']['mean']:+.3f}",
            flush=True,
        )
        if not gate["passed"] and not args.diagnostic_continue:
            break

    if agent is None:
        raise RuntimeError("no recurrent-PPO agent was constructed")

    final_validation_retention: Dict[str, Any] = {}
    for stage, specification in retention_specs.items():
        retained = _evaluate_fitted_policies(
            stage,
            agent,
            specification["environment_factory"],
            specification["gate_seeds"],
            specification["fits"],
        )
        retained["gate"] = gate_stage(stage, retained)
        retained["frozen_policy"] = "final recurrent-PPO checkpoint"
        retained["used_for_training"] = False
        final_validation_retention[stage] = retained

    all_requested_completed = list(stage_results) == list(selected_stages)
    all_stage_gates = bool(
        all_requested_completed
        and all(stage_results[stage]["gate"]["passed"] for stage in selected_stages)
    )
    retention_passed = bool(
        final_validation_retention
        and all(row["gate"]["passed"] for row in final_validation_retention.values())
    )
    report: Dict[str, Any] = {
        "schema": "ride_response_v28_recurrent_ppo_control_v1",
        "role": "mandatory independent on-policy pilot control",
        "city": args.city,
        "budget_name": args.budget,
        "budget": asdict(budget),
        "requested_stages": list(selected_stages),
        "completed_stages": list(stage_results),
        "stage_results": stage_results,
        "final_validation_retention": final_validation_retention,
        "provenance": provenance,
        "environment_contract": environment_contract,
        "learner_config": asdict(agent.config),
        "action_transform": asdict(action_transform),
        "algorithm_contract": {
            "gamma": 1.0,
            "finite_horizon_time_to_go": True,
            "storage_unit": "complete episode",
            "behavior_log_probabilities": "stored at draw and reconstructed before update",
            "decision_cadence": "one fresh action draw per day",
            "option_persistence": False,
            "option_persistence_reason": (
                "not applied because a repeated option requires a joint continue/stop "
                "behavior likelihood; this benchmark keeps exact ordinary PPO ratios"
            ),
            "invalid_action_handling": "terminal numerical penalty; no clipping/substitution",
            "policy_retention": "one recurrent-PPO instance retained across L0-L3",
        },
        "split_contract": {
            "learning": "train profiles and train seeds only",
            "comparator_fit": "validation profiles, validation_fit seeds",
            "gates": "validation profiles, disjoint validation_gate seeds",
            "test": "not implemented in this pilot-control runner",
        },
        "test_split_contract": {
            "opened": False,
            "implemented": False,
            "profile_rows_evaluated": 0,
            "reason": "independent pilot control is validation-only by construction",
        },
        "compute_estimate": estimate,
        "fail_fast_enabled": not bool(args.diagnostic_continue),
        "all_requested_stage_gates_passed": all_stage_gates,
        "final_validation_retention_passed": retention_passed,
        "control_passed": bool(
            args.budget in {"pilot", "full"}
            and not args.diagnostic_continue
            and selected_stages == CONTROL_STAGES
            and all_stage_gates
            and retention_passed
        ),
        "action_diagnostics": agent.action_diagnostics(),
        "elapsed_seconds": float(time.time() - started),
    }
    report["summary_figure"] = _plot_control_report(
        report, out_dir / "recurrent_ppo_control.png"
    )
    report["shock_action_figure"] = _plot_shock_actions(
        report, out_dir / "recurrent_ppo_shock_actions.png"
    )
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(_jsonable(report), indent=2, allow_nan=False))
    agent.save(
        out_dir / "policy_final.pt",
        metadata={
            "city": args.city,
            "budget": args.budget,
            "completed_stages": list(stage_results),
            "control_passed": report["control_passed"],
            "report": str(report_path),
        },
    )
    print(f"\nwrote {report_path}")
    print(f"control passed: {report['control_passed']}")


if __name__ == "__main__":
    main()
