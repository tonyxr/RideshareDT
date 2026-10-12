"""Focused contracts for the v28 environment and staged runner."""

from __future__ import annotations

import datetime as dt
import copy
import json
import os
from pathlib import Path
from statistics import NormalDist
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from continuous_benchmarks import (
    CEMConfig,
    CEMOptimizer,
    evaluate_uncapped_price_tail,
    optimize_public_schedule,
)
from continuous_market_env import (
    ContinuousMarketEnv,
    DailyEnvironmentConfig,
    _raw_threshold_cache_contract,
    _three_way_profile_split,
)
from recurrent_sac import (
    ActionTransform,
    Episode,
    RecurrentSACAgent,
    RecurrentSACConfig,
)
from scripts.v28_learner_ladder import (
    ACTOR_ACCEPTANCE_SEEDS_PER_EPISODE,
    BUDGETS,
    EpisodeActorAcceptance,
    _compact_evaluation,
    _critic_ranking_gate,
    _dataset_paths,
    _episode_seeds,
    _expected_value_calibration,
    _local_action_support,
    _local_deviation_episode,
    _l0_learner_readiness,
    _adaptive_stage_learner_readiness,
    _observation_identity_indices,
    _recurrent_policy_deviation_episode,
    _verify_recurrent_policy_panel,
    _scenario_factory,
    _shock_config,
    gate_stage,
    load_committed_economic_incumbent,
    make_train_critic_gate_rechecker,
    prepare_stage_critic_safety,
    run_critic_only_diagnostic,
    select_train_safe_policy,
    _validate_critic_only_request,
    warm_start_l0,
)


def test_three_way_profile_split_is_deterministic_disjoint_and_roughly_60_20_20() -> None:
    rows = [{"ProfileID": index, "PriceThreshold": 1.0 + index / 100.0} for index in range(100)]
    first = _three_way_profile_split(rows, seed=7)
    second = _three_way_profile_split(rows, seed=7)
    assert {name: [row["ProfileID"] for row in split] for name, split in first.items()} == {
        name: [row["ProfileID"] for row in split] for name, split in second.items()
    }
    ids = {name: {row["ProfileID"] for row in split} for name, split in first.items()}
    assert [len(first[name]) for name in ("train", "validation", "test")] == [60, 20, 20]
    assert ids["train"].isdisjoint(ids["validation"])
    assert ids["train"].isdisjoint(ids["test"])
    assert ids["validation"].isdisjoint(ids["test"])


def test_validation_fit_gate_and_final_test_seed_streams_are_disjoint() -> None:
    train = set(_episode_seeds(17, "l2", 12, "train"))
    warm_fit = set(_episode_seeds(17, "l2", 12, "train_warm_fit"))
    warm_rollout = set(_episode_seeds(17, "l2", 12, "warm"))
    fit = set(_episode_seeds(17, "l2", 12, "validation_fit"))
    gate = set(_episode_seeds(17, "l2", 12, "validation_gate"))
    test = set(_episode_seeds(17, "l2", 12, "test"))
    critic_gate = set(_episode_seeds(17, "l2", 12, "train_critic_gate"))
    selection = set(_episode_seeds(17, "l2", 12, "train_selection"))
    actor_accept = set(_episode_seeds(17, "l2", 12, "train_actor_accept"))
    absolute_fit = set(_episode_seeds(17, "l2", 64, "train_absolute_fit"))
    critic_rank = set(_episode_seeds(17, "l2", 8, "train_critic_rank"))
    absolute_gate = set(
        _episode_seeds(17, "l2", 32, "train_critic_absolute_gate")
    )
    assert train.isdisjoint(warm_fit)
    assert train.isdisjoint(warm_rollout)
    assert warm_fit.isdisjoint(warm_rollout)
    assert fit.isdisjoint(gate)
    assert fit.isdisjoint(test)
    assert gate.isdisjoint(test)
    all_streams = [
        train,
        warm_fit,
        warm_rollout,
        critic_gate,
        selection,
        actor_accept,
        absolute_fit,
        critic_rank,
        absolute_gate,
        fit,
        gate,
        test,
    ]
    assert all(
        left.isdisjoint(right)
        for index, left in enumerate(all_streams)
        for right in all_streams[index + 1 :]
    )


def test_expected_value_gate_uses_independent_equal_day_seed_clusters_and_t_ucb() -> None:
    days = (0, 10, 19, 29)
    rows = []
    seed = 1_000
    for day in days:
        for offset in range(8):
            realized = 100.0 + float(offset - 3)
            rows.append(
                {
                    "q_probe": realized + (0.5 if offset % 2 else -0.5),
                    "return_probe": realized,
                    "probe_transition": day,
                    "metadata": {
                        "seed": seed,
                        "calibration_day": day,
                        "support_index": 0,
                        "heldout_for_gate": True,
                    },
                }
            )
            seed += 1
    diagnostics = {"monte_carlo_mse": 1.0e12, "episodes": rows}
    result = _expected_value_calibration(
        diagnostics, deviation_days=days, seeds_per_stratum=8
    )
    assert result["passed"]
    assert result["unique_seed_count"] == 32
    assert not result["duplicate_seed_clusters"]
    assert result["raw_pooled_monte_carlo_rmse_descriptive"] == pytest.approx(1.0e6)
    assert result["per_stratum_alpha"] == pytest.approx(0.05 / len(days))
    assert result["per_stratum_tail_alpha"] == pytest.approx(
        0.05 / (2 * len(days))
    )
    normal_critical = NormalDist().inv_cdf(1.0 - 0.05 / (2 * len(days)))
    assert all(
        row["two_sided_student_t_critical_value"] > normal_critical
        for row in result["strata"].values()
    )
    assert list({
        row["two_sided_student_t_critical_value"]
        for row in result["strata"].values()
    }) == pytest.approx([3.3352949398170852])

    duplicated = copy.deepcopy(diagnostics)
    duplicated["episodes"][8]["metadata"]["seed"] = duplicated["episodes"][0][
        "metadata"
    ]["seed"]
    assert not _expected_value_calibration(
        duplicated, deviation_days=days, seeds_per_stratum=8
    )["passed"]


def test_actor_acceptance_budget_uses_eight_pilot_seeds_and_student_t() -> None:
    assert ACTOR_ACCEPTANCE_SEEDS_PER_EPISODE["smoke"] >= 4
    assert ACTOR_ACCEPTANCE_SEEDS_PER_EPISODE["pilot"] == 8
    assert ACTOR_ACCEPTANCE_SEEDS_PER_EPISODE["full"] >= 8


def test_critic_only_rejects_any_stage_other_than_l0() -> None:
    _validate_critic_only_request(True, ("l0",))
    with pytest.raises(ValueError, match="exactly --stages l0"):
        _validate_critic_only_request(True, ("l0", "l1"))
    with pytest.raises(ValueError, match="exactly --stages l0"):
        _validate_critic_only_request(True, ("l1",))


def test_critic_only_never_constructs_cpi_trains_or_opens_validation_test(
    monkeypatch, tmp_path
) -> None:
    environment_splits = []

    class FakeEnvironment:
        def __init__(
            self,
            core,
            *,
            config,
            scenario_factory,
            profile_split,
            shock_sampling_mode,
        ):
            del core, scenario_factory, shock_sampling_mode
            environment_splits.append(profile_split)
            assert profile_split == "train"
            self.config = config
            self.observation_names = (
                "continuous",
                "weather_clear",
                "weather_rain",
                "public_storm_warning",
            )
            self.observation_dim = len(self.observation_names)

        def contract(self):
            return {
                "profile_split": "train",
                "observation_names": list(self.observation_names),
            }

    monkeypatch.setattr(
        "scripts.v28_learner_ladder.build_v28_core",
        lambda **kwargs: (object(), {"native_uncapped": True}),
    )
    monkeypatch.setattr(
        "scripts.v28_learner_ladder.ContinuousMarketEnv", FakeEnvironment
    )
    monkeypatch.setattr(
        "scripts.v28_learner_ladder.evaluate_uncapped_price_tail",
        lambda factory, seeds: {
            "passed": True,
            "criterion": "tail",
            "seeds": list(seeds),
            "profile_split": factory().contract()["profile_split"],
        },
    )

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("critic-only entered forbidden actor-training path")

    monkeypatch.setattr("scripts.v28_learner_ladder.train_stage", forbidden)
    monkeypatch.setattr(
        "scripts.v28_learner_ladder.EpisodeActorAcceptance", forbidden
    )

    def fake_warm(
        agent,
        factory,
        training_fit_seeds,
        rollout_seeds,
        critic_gate_seeds,
        training_episode_seeds,
        **kwargs,
    ):
        assert factory().contract()["profile_split"] == "train"
        assert len(training_fit_seeds) >= 2
        assert len(rollout_seeds) == 4
        assert len(critic_gate_seeds) == 8
        assert training_episode_seeds
        assert len(kwargs["absolute_fit_seeds"]) == 64
        assert len(kwargs["absolute_gate_seeds"]) == 32
        assert tuple(kwargs["critic_probe_days"]) == (0, 10, 19, 29)
        assert agent.actor_updates_applied == 0
        return {
            "critic_coverage_and_ranking_gate": {"passed": True},
            "profile_split_contract": {
                "validation_profiles_used": False,
                "final_test_used": False,
            },
            "_critic_gate_runtime": {"must_not_be_serialized": object()},
        }

    monkeypatch.setattr(
        "scripts.v28_learner_ladder.warm_start_l0", fake_warm
    )
    args = SimpleNamespace(
        city="chicago",
        seed=91,
        device="cpu",
        budget="pilot",
        warm_support_radius=0.15,
        warm_paired_advantage_weight=1.0,
        warm_anchor_coefficient=10.0,
        warm_anchor_decay_updates=100.0,
    )
    report = run_critic_only_diagnostic(
        args=args,
        budget=copy.deepcopy(BUDGETS["pilot"]),
        calibration={},
        calibration_path=tmp_path / "calibration.json",
        trip_bank_path=tmp_path / "trip_bank.csv",
        profile_cache_path=_dataset_paths("chicago")[2],
        out_dir=tmp_path,
        warm_start_rollouts=4,
        warm_start_clone_steps=1,
        warm_start_critic_steps=1,
        warm_paired_advantage_steps=1,
    )
    assert report["critic_gate_passed"]
    assert not report["promotion_ready"]
    assert report["execution_contract"] == {
        "episode_actor_acceptance_constructed": False,
        "train_stage_called": False,
        "sac_actor_updates_called": False,
        "temperature_updates_called": False,
        "behavior_clone_only_policy_optimization": True,
        "promotion_eligible": False,
        "quarantined_diagnostic": True,
    }
    assert report["profile_split_contract"]["validation_split_opened"] is False
    assert report["profile_split_contract"]["test_split_opened"] is False
    assert set(environment_splits) == {"train"}
    assert (tmp_path / "critic_only_report.json").is_file()
    assert (tmp_path / "critic_only_diagnostic_unpromoted.pt").is_file()


def test_local_support_is_uncapped_symmetric_full_rank_and_branch_is_one_day() -> None:
    transform = ActionTransform.market_controls()
    center = np.array([np.log(1.2), 0.3, np.log(0.8)])
    support = _local_action_support(transform, center, 0.2)
    assert len(support["actions"]) == 15
    assert support["support_rank"] == 3
    assert support["symmetric"]
    plus = next(row for row in support["actions"] if row["label"] == "axis:base_multiplier:+")
    minus = next(row for row in support["actions"] if row["label"] == "axis:base_multiplier:-")
    assert plus["action"][0] * minus["action"][0] == pytest.approx(1.2**2)

    class ThreeDayEnvironment:
        def reset(self, seed):
            self.day = 0
            return np.array([float(seed)], dtype=np.float32)

        def step(self, action):
            self.day += 1
            return (
                np.array([float(self.day)], dtype=np.float32),
                float(action[0]),
                self.day == 3,
                {},
            )

    center_action = transform.coordinates_to_physical(center)
    branch_action = np.asarray(plus["action"])
    episode, _, _ = _local_deviation_episode(
        ThreeDayEnvironment, center_action, branch_action, 9, 1
    )
    np.testing.assert_allclose(episode.actions[1], branch_action)
    np.testing.assert_allclose(episode.actions[[0, 2]], np.tile(center_action, (2, 1)))


def test_recurrent_stage_panel_has_exact_prefix_and_relative_one_day_branches() -> None:
    class ThreeDayEnvironment:
        def reset(self, seed):
            self.day = 0
            self.state = float(seed % 5)
            return np.array([self.state, 0.0], dtype=np.float32)

        def step(self, action):
            self.day += 1
            self.state += float(action[0]) * 0.01
            return (
                np.array([self.state, float(self.day)], dtype=np.float32),
                float(action[0] - action[2]),
                self.day == 3,
                {},
            )

    agent = RecurrentSACAgent(
        2, config=RecurrentSACConfig(hidden_dim=8, batch_size=1, seed=371)
    )
    support = _local_action_support(
        agent.action_transform, np.zeros(3), 0.1
    )
    episodes = []
    for row in support["actions"]:
        episode, _, _ = _recurrent_policy_deviation_episode(
            ThreeDayEnvironment,
            agent,
            73,
            1,
            row["offset"],
            metadata={"support_index": row["index"]},
        )
        episodes.append(episode)
    contract = _verify_recurrent_policy_panel(
        episodes,
        support=support,
        seeds=(73,),
        deviation_days=(1,),
        transform=agent.action_transform,
    )
    assert contract["passed"]
    assert contract["matched_prefix_verified"]
    assert contract["episode_count"] == 15
    center = episodes[0]
    plus = next(
        episode
        for episode in episodes
        if episode.metadata["support_index"] == 1
    )
    center_coordinates = agent.action_transform.physical_to_coordinates(
        center.actions
    )
    plus_coordinates = agent.action_transform.physical_to_coordinates(plus.actions)
    np.testing.assert_allclose(plus_coordinates[0], center_coordinates[0])
    np.testing.assert_allclose(
        plus_coordinates[1] - center_coordinates[1],
        np.array([0.1, 0.0, 0.0]),
        atol=1.0e-6,
    )


def test_later_stage_safety_uses_stage_specific_train_only_disjoint_panels(
    monkeypatch,
) -> None:
    calls = []

    class OneDayEnvironment:
        observation_names = ("weather_clear", "weather_rain")
        config = SimpleNamespace(episode_days=1)

        def reset(self, seed):
            calls.append(int(seed))
            self.seed = int(seed)
            rain = float(self.seed % 2)
            return np.array([1.0 - rain, rain], dtype=np.float32)

        def step(self, action):
            coordinates = ActionTransform.market_controls().physical_to_coordinates(
                np.asarray(action, dtype=np.float64)
            )
            reward = float(10.0 - np.square(coordinates).sum())
            return np.array([1.0, 0.0], dtype=np.float32), reward, True, {}

    class FakeReplay:
        def __init__(self):
            self.episodes = []

        def __len__(self):
            return sum(episode.length for episode in self.episodes)

    class FakeAgent:
        action_transform = ActionTransform.market_controls()
        action_dim = 3
        config = SimpleNamespace(observation_identity_indices=(0, 1))
        updates = 9

        def __init__(self):
            self.replay = FakeReplay()
            self.extreme_action_count = 0
            self.nonfinite_action_rejections = 0
            self.max_abs_coordinate = np.zeros(3)
            self.max_physical_action = np.full(3, -np.inf)
            self.min_physical_action = np.full(3, np.inf)
            self.cache_call = None
            self.safe_config = None

        def act(self, observation, **kwargs):
            del observation, kwargs
            coordinates = np.zeros(3, dtype=np.float64)
            return (
                self.action_transform.coordinates_to_physical(coordinates),
                None,
                {"policy_coordinates": coordinates},
            )

        def add_episode(self, episode, actions_are_physical, update_normalizer):
            assert actions_are_physical and not update_normalizer
            self.replay.episodes.append(
                Episode(
                    episode.observations,
                    self.action_transform.physical_to_coordinates(
                        episode.actions
                    ).astype(np.float32),
                    episode.rewards,
                    episode.dones,
                    metadata=copy.deepcopy(episode.metadata),
                )
            )

        def configure_paired_crn_cache(self, episodes, **kwargs):
            self.cache_call = (episodes, kwargs)
            assert kwargs["replace_for_new_stage"] is True
            assert kwargs["source_group_id"] == "l1_train_local_support"
            assert all(
                episode.metadata["profile_split"] == "train_fit"
                and not episode.metadata["heldout_for_gate"]
                for episode in episodes
            )
            return {
                "source_group_id": kwargs["source_group_id"],
                "heldout_gate_episodes_used": False,
            }

        def critic_validation(self, episodes):
            rows = []
            for episode in episodes:
                total = float(np.sum(episode.rewards))
                day = int(episode.metadata["deviation_day"])
                rows.append(
                    {
                        "q0": total,
                        "return0": total,
                        "q_probe": total,
                        "q1_probe": total,
                        "q2_probe": total,
                        "return_probe": total,
                        "probe_transition": day,
                        "metadata": copy.deepcopy(episode.metadata),
                    }
                )
            return {
                "finite": True,
                "td_mse": 0.0,
                "monte_carlo_mse": 0.0,
                "monte_carlo_normalized_rmse": 0.0,
                "q0_relative_absolute_error": 0.0,
                "q0_to_entropy_scale_ratio": 10.0,
                "episodes": rows,
            }

        def fit_critics_monte_carlo(self, **kwargs):
            assert kwargs["absolute_fit_group_id"] == "l1_train_absolute_fit"
            assert kwargs["paired_fit_group_id"] == "l1_train_local_support"
            return {"steps": kwargs["steps"], "final_loss": 0.0}

        def configure_safe_policy_improvement(self, **kwargs):
            self.safe_config = kwargs

        def capture_last_passing_state(self, label):
            assert label == "l1_entering_policy_critic_gate_pass"
            return {"label": label, "available": True}

    def fake_gate(*args, **kwargs):
        del args, kwargs
        return {"passed": True, "spearman": 1.0}

    monkeypatch.setattr(
        "scripts.v28_learner_ladder._critic_ranking_gate", fake_gate
    )
    agent = FakeAgent()
    result = prepare_stage_critic_safety(
        agent,
        OneDayEnvironment,
        stage="l1",
        rollout_seeds=(101, 102),
        critic_gate_seeds=tuple(range(201, 209)),
        absolute_fit_seeds=tuple(range(301, 365)),
        absolute_gate_seeds=tuple(range(401, 409)),
        training_episode_seeds=(501, 502),
        critic_probe_days=(0,),
        critic_steps=2,
        paired_advantage_steps=2,
        support_radius=0.1,
        paired_advantage_weight=1.0,
    )
    runtime = result.pop("_critic_gate_runtime")
    assert result["critic_coverage_and_ranking_gate"]["passed"]
    assert len(result["absolute_fit_rollouts"]) == 64
    assert result["absolute_fit_weather_coverage"]["passed"]
    assert result["replay_support_coverage"]["passed"]
    assert result["critic_gate_crossed_panel_coverage"]["passed"]
    assert result["profile_split_contract"]["streams_pairwise_disjoint"]
    assert result["profile_split_contract"]["validation_used"] is False
    assert result["profile_split_contract"]["test_used"] is False
    assert len(agent.replay.episodes) == 64 + 2 * 15
    assert all(not episode.metadata["heldout_for_gate"] for episode in agent.replay.episodes)
    assert runtime["stage"] == "l1"
    assert len(runtime["episodes"]) == 8 * 1 * 15
    assert len(runtime["absolute_episodes"]) == 8
    assert agent.safe_config["actor_updates_enabled"] is False


def test_later_stage_readiness_requires_positive_accepted_actor_gain() -> None:
    gate = {
        "passed": True,
        "actor_temperature_authorized_this_update": True,
    }
    report = {
        "events": [{"actor_proposal": True, "numerical_reject": False}],
        "proposed_actor_updates": 1,
        "accepted_actor_proposals": 0,
        "positive_actor_gain_reported": False,
    }
    assert _l0_learner_readiness(gate, report)
    assert not _adaptive_stage_learner_readiness(gate, report)
    report["accepted_actor_proposals"] = 1
    report["positive_actor_gain_reported"] = True
    assert _adaptive_stage_learner_readiness(gate, report)


def test_critic_gate_ranks_paired_crn_deltas_not_seed_value_offsets() -> None:
    support = _local_action_support(
        ActionTransform.market_controls(), np.zeros(3), 0.1
    )
    support_indices = []
    seeds = []
    rows = []
    for support_row in support["actions"]:
        index = int(support_row["index"])
        for seed, offset in ((11, 1_000.0), (29, -1_000.0)):
            support_indices.append(index)
            seeds.append(seed)
            delta = 0.0 if index == 0 else float(index)
            rows.append(
                {
                    "return0": offset + delta,
                    "q0": -3.0 * offset + delta,
                }
            )
    prior_rows = []
    for index, seed, row in zip(support_indices, seeds, rows):
        center_position = seeds.index(seed)
        prior_rows.append({"return0": row["return0"], "q0": rows[center_position]["q0"]})
    before = {
        "td_mse": 100.0,
        "monte_carlo_mse": 100.0,
        "episodes": prior_rows,
    }
    after = {
        "finite": True,
        "td_mse": 2.0,
        "monte_carlo_mse": 1.0,
        "monte_carlo_normalized_rmse": 0.1,
        "q0_relative_absolute_error": 0.1,
        "q0_to_entropy_scale_ratio": 20.0,
        "episodes": rows,
    }
    coverage = {"passed": True}
    gate = _critic_ranking_gate(
        support, coverage, before, after, support_indices, seeds
    )
    assert gate["passed"]
    assert gate["spearman"] == pytest.approx(1.0)
    reversed_rows = [dict(row) for row in rows]
    for position, (index, seed) in enumerate(zip(support_indices, seeds)):
        center_position = seeds.index(seed)
        center_q = rows[center_position]["q0"]
        reversed_rows[position]["q0"] = center_q - (
            rows[position]["q0"] - center_q
        )
    after["episodes"] = reversed_rows
    assert not _critic_ranking_gate(
        support, coverage, before, after, support_indices, seeds
    )["passed"]


def test_post_training_critic_gate_needs_two_passes_and_refreezes(monkeypatch) -> None:
    outcomes = iter((True, True, False))

    def fake_gate(*args, **kwargs):
        del args, kwargs
        return {"passed": next(outcomes)}

    monkeypatch.setattr(
        "scripts.v28_learner_ladder._critic_ranking_gate", fake_gate
    )

    class FakeAgent:
        actor_updates_enabled = False

        def critic_validation(self, episodes):
            if episodes == ["absolute"]:
                return {
                    "monte_carlo_mse": 1.0,
                    "episodes": [
                        {
                            "q_probe": 10.0,
                            "return_probe": 10.0,
                            "probe_transition": 0,
                            "metadata": {
                                "seed": seed,
                                "calibration_day": 0,
                                "support_index": 0,
                                "heldout_for_gate": True,
                            },
                        }
                        for seed in (901, 902)
                    ],
                }
            assert episodes == ["heldout"]
            return {"finite": True}

    runtime = {
        "episodes": ["heldout"],
        "support": {},
        "replay_coverage": {},
        "diagnostics_before": {},
        "support_indices": [],
        "episode_seeds": [],
        "probe_days": [],
        "gate_panel_coverage": {"passed": True},
        "absolute_fit_coverage": {"passed": True},
        "absolute_episodes": ["absolute"],
        "absolute_deviation_days": [0],
        "absolute_seeds_per_stratum": 2,
        "absolute_diagnostics_before": FakeAgent().critic_validation(["absolute"]),
        "pass_streak": 0,
        "events": [],
    }
    agent = FakeAgent()
    recheck = make_train_critic_gate_rechecker(runtime)
    first = recheck(agent, 0)
    assert first["consecutive_passes"] == 1
    assert not agent.actor_updates_enabled
    second = recheck(agent, 1)
    assert second["consecutive_passes"] == 2
    assert agent.actor_updates_enabled
    third = recheck(agent, 2)
    assert third["consecutive_passes"] == 0
    assert not agent.actor_updates_enabled


def test_episode_actor_acceptance_rejects_harm_accepts_gain_and_never_learns_seeds(
    monkeypatch,
    tmp_path,
) -> None:
    agent = RecurrentSACAgent(
        2,
        config=RecurrentSACConfig(
            hidden_dim=8,
            batch_size=1,
            seed=811,
            observation_identity_indices=(1,),
        ),
    )
    agent.configure_critic_action_reference(np.zeros(3), np.full(3, 0.2))
    base_episode = Episode(
        observations=np.zeros((2, 2), dtype=np.float32),
        actions=np.zeros((1, 3), dtype=np.float32),
        rewards=np.array([1.0], dtype=np.float32),
        dones=np.array([1.0], dtype=np.float32),
        metadata={"group_id": "ordinary_training", "seed": 1},
    )
    agent.add_episode(base_episode, actions_are_physical=False)
    agent.capture_last_passing_state("warm_critic_safe")
    initial_gate = {
        "passed": True,
        "gate_passed_current_version": True,
        "actor_temperature_authorized_this_update": True,
        "consecutive_passes": 3,
        "marker": "preexisting_gate_event",
    }
    runtime = {"pass_streak": 3, "events": [initial_gate]}
    mode = {"sign": -1.0, "calls": 0}

    def fake_evaluate(factory, current_agent, seeds, *, deterministic):
        del factory
        assert deterministic
        mode["calls"] += 1
        actor_output = current_agent.actor.head[-1]
        assert isinstance(actor_output, torch.nn.Linear)
        policy_coordinate = float(actor_output.bias[0].detach())
        values = [
            mode["sign"] * policy_coordinate + (int(seed) % 13) / 100.0
            for seed in seeds
        ]
        return {
            "mean": float(np.mean(values)),
            "standard_error": 0.0,
            "median": float(np.median(values)),
            "values": values,
            "episodes": [],
        }

    monkeypatch.setattr(
        "scripts.v28_learner_ladder.evaluate_recurrent_agent", fake_evaluate
    )
    blocks = ((9001, 9002, 9003), (9011, 9012, 9013), (9021, 9022, 9023))
    controller = EpisodeActorAcceptance(
        agent,
        lambda: object(),
        blocks,
        runtime,
        forbidden_seeds=(1, 2, 3, 4),
        familywise_alpha=0.05,
        incumbent_checkpoint_directory=tmp_path / "incumbents",
    )
    incumbent = agent.economic_incumbent_transaction_state()

    # First proposal is harmful on the full later return path.  Its replay,
    # RNG, policy, critic-safe pointer, and pass streak are all rolled back,
    # while the gate and rejection audit events remain external.
    output = agent.actor.head[-1]
    assert isinstance(output, torch.nn.Linear)
    with torch.no_grad():
        output.bias[0] = 1.0
    agent.actor_updates_applied = 1
    agent.add_episode(
        Episode(
            observations=np.ones((2, 2), dtype=np.float32),
            actions=np.ones((1, 3), dtype=np.float32) * 0.1,
            rewards=np.array([2.0], dtype=np.float32),
            dones=np.array([1.0], dtype=np.float32),
            metadata={"group_id": "candidate_training", "seed": 2},
        ),
        actions_are_physical=False,
    )
    agent.rng.random()
    agent.replay.rng.random()
    torch.rand(2)
    runtime["pass_streak"] = 5
    harmful_gate = {
        "passed": True,
        "gate_passed_current_version": True,
        "actor_temperature_authorized_this_update": True,
        "consecutive_passes": 5,
        "marker": "candidate_gate_event",
    }
    runtime["events"].append(harmful_gate)
    agent.capture_last_passing_state("harmful_candidate_critic")
    rejected = controller.evaluate_episode(0, proposal_numerically_valid=True)
    assert not rejected["accepted"]
    assert rejected["paired_lcb"] < 0.0
    assert rejected["rollback"]["full_state_rng_replay_restored"]
    assert not agent.training_terminal_failure
    assert runtime["pass_streak"] == 3
    assert runtime["events"] == [initial_gate, harmful_gate]
    restored = agent.training_transaction_state()
    assert torch.equal(
        restored["learner"]["torch_rng_state"],
        incumbent["learner"]["torch_rng_state"],
    )
    assert (
        restored["learner"]["replay"].rng.bit_generator.state
        == incumbent["learner"]["replay"].rng.bit_generator.state
    )
    assert len(agent.replay.episodes) == len(incumbent["learner"]["replay"].episodes)
    assert agent.last_passing_state_contract()["label"] == "warm_critic_safe"

    # A deterministic positive paired gain has SE=0, hence its corrected LCB
    # equals the gain and is accepted exactly.
    mode["sign"] = 1.0
    output = agent.actor.head[-1]
    with torch.no_grad():
        output.bias[0] = 1.0
    agent.actor_updates_applied = 1
    runtime["pass_streak"] = 6
    runtime["events"].append(
        {
            "passed": True,
            "gate_passed_current_version": True,
            "actor_temperature_authorized_this_update": True,
            "consecutive_passes": 6,
            "marker": "beneficial_candidate_gate",
        }
    )
    agent.capture_last_passing_state("beneficial_candidate_critic")
    accepted = controller.evaluate_episode(1, proposal_numerically_valid=True)
    assert accepted["accepted"]
    assert accepted["paired_lcb"] == pytest.approx(1.0)
    assert accepted["per_look_bonferroni_alpha"] == pytest.approx(0.05 / 3.0)
    normal_critical = NormalDist().inv_cdf(1.0 - 0.05 / 3.0)
    assert accepted["one_sided_student_t_critical_value"] > normal_critical
    assert accepted["student_t_degrees_of_freedom"] == 2
    assert agent.accepted_actor_blocks == 1
    assert runtime["pass_streak"] == 6
    assert agent.economic_incumbent_contract()["label"] == "episode_1_positive_lcb"
    assert accepted["incumbent_checkpoint"] is not None
    assert Path(accepted["incumbent_checkpoint"]).is_file()

    # Numerical invalidity is an unconditional rejection and restores the
    # accepted incumbent without consuming either validation or test data.
    output = agent.actor.head[-1]
    with torch.no_grad():
        output.bias[0] = 2.0
    agent.actor_updates_applied = 2
    runtime["pass_streak"] = 7
    agent.capture_last_passing_state("numerical_candidate_critic")
    calls_before = mode["calls"]
    numerical = controller.evaluate_episode(2, proposal_numerically_valid=False)
    assert not numerical["accepted"]
    assert numerical["numerical_reject"]
    assert mode["calls"] == calls_before
    assert agent.accepted_actor_blocks == 1
    output = agent.actor.head[-1]
    assert float(output.bias[0].detach()) == pytest.approx(1.0)

    summary = controller.summary()
    assert summary["accepted_actor_proposals"] == 1
    assert summary["rejected_actor_proposals"] == 2
    assert summary["positive_actor_gain_reported"]
    assert len(summary["incumbent_checkpoints"]) == 2
    assert all(Path(path).is_file() for path in summary["incumbent_checkpoints"])
    assert len(summary["incumbent_checkpoint_history"]) == 2
    assert (tmp_path / "incumbents" / "manifest.json").is_file()
    assert not summary["seed_reuse"]
    assert not summary["acceptance_episodes_added_to_replay"]
    assert set(seed for block in blocks for seed in block) == controller.consumed_seeds
    assert all(
        episode.metadata.get("group_id") != "train_actor_accept"
        for episode in agent.replay.episodes
    )


def test_policy_change_detection_includes_normalizer_not_temperature() -> None:
    agent = RecurrentSACAgent(
        2,
        config=RecurrentSACConfig(hidden_dim=8, batch_size=1, seed=812),
    )
    agent.configure_critic_action_reference(np.zeros(3), np.full(3, 0.2))
    candidate = agent.training_transaction_state()
    incumbent = copy.deepcopy(candidate)
    candidate["learner"]["log_temperature"] = (
        candidate["learner"]["log_temperature"] + 1.0
    )
    assert not EpisodeActorAcceptance._policy_changed(candidate, incumbent)
    candidate = copy.deepcopy(incumbent)
    candidate["learner"]["observation_normalizer"]["mean"][0] += 0.01
    assert EpisodeActorAcceptance._policy_changed(candidate, incumbent)


def test_critic_rejection_spends_no_acceptance_block_and_next_episode_continues() -> None:
    agent = RecurrentSACAgent(
        2,
        config=RecurrentSACConfig(hidden_dim=8, batch_size=1, seed=813),
    )
    agent.configure_critic_action_reference(np.zeros(3), np.full(3, 0.2))
    agent.capture_last_passing_state("warm_critic")
    runtime = {
        "pass_streak": 4,
        "events": [{"passed": False, "terminal_on_failure": True}],
    }
    blocks = ((8101, 8102, 8103), (8111, 8112, 8113))
    controller = EpisodeActorAcceptance(
        agent,
        lambda: object(),
        blocks,
        runtime,
        familywise_alpha=0.05,
    )
    with torch.no_grad():
        next(agent.actor.parameters()).add_(1.0)
    agent.actor_updates_applied = 1
    agent.capture_last_passing_state("unverified_actor_critic")
    rollback = agent.restore_economic_incumbent("synthetic critic gate miss")
    rejected = controller.record_critic_proposal_rejection(
        0,
        rollback=rollback,
        rejected_update=7,
    )
    assert rejected["critic_proposal_rejection"]
    assert not rejected["seeds_consumed"]
    assert rejected["acceptance_seeds"] == []
    assert controller.consumed_seeds == set()
    assert runtime["pass_streak"] == 4
    assert not runtime["events"][-1]["terminal_on_failure"]
    assert agent.last_passing_state_contract()["label"] == "warm_critic"

    runtime["events"].append(
        {
            "passed": True,
            "gate_passed_current_version": True,
            "actor_temperature_authorized_this_update": True,
            "consecutive_passes": 5,
        }
    )
    runtime["pass_streak"] = 5
    continued = controller.evaluate_episode(1, proposal_numerically_valid=True)
    assert continued["decision"] == "critic_only_advance"
    assert len(controller.events) == 2
    assert controller.summary()["acceptance_blocks_consumed"] == 0
    assert not agent.training_terminal_failure


def test_final_gate_miss_reports_attempt_but_preserves_incumbent_readiness(
    monkeypatch,
) -> None:
    agent = RecurrentSACAgent(
        2,
        config=RecurrentSACConfig(hidden_dim=8, batch_size=1, seed=814),
    )
    agent.configure_critic_action_reference(np.zeros(3), np.full(3, 0.2))
    agent.capture_last_passing_state("warm_critic")
    passing_gate = {
        "update": 1,
        "passed": True,
        "gate_passed_current_version": True,
        "consecutive_passes": 2,
        "actor_temperature_authorized_this_update": True,
    }
    runtime = {
        "pass_streak": 1,
        "warm_gate_authorization": {
            "update": 0,
            "passed": False,
            "gate_passed_current_version": True,
            "consecutive_passes": 1,
            "actor_temperature_authorized_this_update": False,
        },
        "events": [],
    }

    def fake_evaluate(factory, current_agent, seeds, *, deterministic):
        del factory
        assert deterministic
        output = current_agent.actor.head[-1]
        assert isinstance(output, torch.nn.Linear)
        value = float(output.bias[0].detach())
        values = [value + (int(seed) % 3) * 0.01 for seed in seeds]
        return {
            "mean": float(np.mean(values)),
            "standard_error": 0.0,
            "median": float(np.median(values)),
            "values": values,
            "episodes": [],
        }

    monkeypatch.setattr(
        "scripts.v28_learner_ladder.evaluate_recurrent_agent", fake_evaluate
    )
    controller = EpisodeActorAcceptance(
        agent,
        lambda: object(),
        ((8201, 8202, 8203), (8211, 8212, 8213)),
        runtime,
    )
    output = agent.actor.head[-1]
    assert isinstance(output, torch.nn.Linear)
    with torch.no_grad():
        output.bias[0] += 1.0
    agent.actor_updates_applied = 1
    runtime["events"].append(copy.deepcopy(passing_gate))
    runtime["pass_streak"] = 2
    agent.capture_last_passing_state("accepted_candidate_critic")
    accepted = controller.evaluate_episode(0, proposal_numerically_valid=True)
    assert accepted["accepted"]
    incumbent = agent.economic_incumbent_transaction_state()

    with torch.no_grad():
        output.bias[0] += 0.5
    agent.observation_normalizer.mean[0] += 2.0
    agent.actor_updates_applied += 1
    agent.capture_last_passing_state("unverified_later_actor")
    failed_gate = {
        "update": 2,
        "passed": False,
        "gate_passed_current_version": False,
        "consecutive_passes": 0,
        "actor_temperature_authorized_this_update": False,
    }
    runtime["events"].append(failed_gate)
    runtime["pass_streak"] = 0
    rollback = agent.restore_economic_incumbent("borderline gate miss")
    controller.record_critic_proposal_rejection(
        1,
        rollback=rollback,
        rejected_update=2,
    )

    assert runtime["events"][-1]["passed"] is False
    current = controller.current_incumbent_gate()
    assert current["passed"] is True
    assert current["gate_passed_current_version"] is True
    assert current["actor_temperature_authorized_this_update"] is True
    assert current["restored_after_rejection"] is True
    assert current["incumbent_label"] == "episode_0_positive_lcb"
    assert current["incumbent_capture_count"] == 2
    restored = agent.economic_incumbent_transaction_state()
    for name, tensor in incumbent["learner"]["actor"].items():
        assert torch.equal(restored["learner"]["actor"][name], tensor)
    np.testing.assert_array_equal(
        restored["learner"]["observation_normalizer"]["mean"],
        incumbent["learner"]["observation_normalizer"]["mean"],
    )
    summary = controller.summary()
    assert summary["current_incumbent_gate"]["restored_after_rejection"]
    assert _l0_learner_readiness(current, summary)


def _incumbent_store_controller(root: Path, seed: int = 815):
    agent = RecurrentSACAgent(
        2,
        config=RecurrentSACConfig(
            hidden_dim=8,
            batch_size=1,
            seed=seed,
            observation_identity_indices=(1,),
        ),
    )
    agent.configure_critic_action_reference(np.zeros(3), np.full(3, 0.2))
    agent.capture_last_passing_state("warm_critic")
    runtime = {"pass_streak": 1, "events": []}
    controller = EpisodeActorAcceptance(
        agent,
        lambda: object(),
        ((8301, 8302),),
        runtime,
        incumbent_checkpoint_directory=root,
    )
    return agent, controller


def test_atomic_incumbent_store_loads_and_retains_only_two_generations(tmp_path) -> None:
    root = tmp_path / "store"
    agent, controller = _incumbent_store_controller(root)
    for capture in (2, 3):
        with torch.no_grad():
            output = agent.actor.head[-1]
            assert isinstance(output, torch.nn.Linear)
            output.bias[0] += 0.1
        agent.accept_economic_incumbent(f"accepted_{capture}")
        controller._persist_incumbent(f"accepted_{capture}")

    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["current"]["capture_count"] == 3
    assert manifest["previous"]["capture_count"] == 2
    assert len(manifest["history"]) == 3
    assert all(
        row["filename"].startswith(f"incumbent_{row['capture_count']:04d}_")
        and row["filename"].endswith(".pt")
        for row in manifest["history"]
    )
    assert len(list(root.glob("incumbent_*.pt"))) == 2
    loaded_current, current_contract = load_committed_economic_incumbent(root)
    loaded_previous, previous_contract = load_committed_economic_incumbent(
        root, generation="previous"
    )
    assert current_contract["capture_count"] == 3
    assert previous_contract["capture_count"] == 2
    assert loaded_current.economic_incumbent_contract()["label"] == "accepted_3"
    assert loaded_previous.economic_incumbent_contract()["label"] == "accepted_2"


def test_incumbent_manifest_is_commit_point_and_ignores_newer_orphan(
    monkeypatch, tmp_path
) -> None:
    root = tmp_path / "store"
    agent, controller = _incumbent_store_controller(root, seed=816)
    original_manifest = json.loads((root / "manifest.json").read_text())
    with torch.no_grad():
        output = agent.actor.head[-1]
        assert isinstance(output, torch.nn.Linear)
        output.bias[0] += 0.2
    agent.accept_economic_incumbent("orphaned_accept")
    real_replace = os.replace

    def crash_before_manifest_commit(source, destination):
        if Path(destination).name == "manifest.json":
            raise OSError("synthetic manifest commit crash")
        return real_replace(source, destination)

    monkeypatch.setattr(
        "scripts.v28_learner_ladder.os.replace", crash_before_manifest_commit
    )
    with pytest.raises(OSError, match="synthetic manifest"):
        controller._persist_incumbent("orphaned_accept")
    assert json.loads((root / "manifest.json").read_text()) == original_manifest
    assert len(list(root.glob("incumbent_*.pt"))) == 2
    loaded, verified = load_committed_economic_incumbent(root)
    assert verified["capture_count"] == 1
    assert loaded.economic_incumbent_contract()["label"] == "l0_warm_economic_incumbent"


def test_incumbent_reader_rejects_manifest_mismatch_and_file_corruption(tmp_path) -> None:
    root = tmp_path / "store"
    _incumbent_store_controller(root, seed=817)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    mismatched = copy.deepcopy(manifest)
    mismatched["current"]["label"] = "forged_label"
    manifest_path.write_text(json.dumps(mismatched))
    with pytest.raises(ValueError, match="embedded contract mismatch"):
        load_committed_economic_incumbent(root)

    manifest_path.write_text(json.dumps(manifest))
    artifact = root / manifest["current"]["filename"]
    with artifact.open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_committed_economic_incumbent(root)


def test_l0_warm_start_selects_and_prefills_on_training_only(monkeypatch) -> None:
    calls = []

    class OneDayTrainingEnvironment:
        def reset(self, seed: int) -> np.ndarray:
            self.seed = seed
            return np.zeros(2, dtype=np.float32)

        def step(self, action: np.ndarray):
            return np.ones(2, dtype=np.float32), float(np.sum(action)), True, {}

    def training_factory():
        return OneDayTrainingEnvironment()

    def fake_static(factory, transform, seeds, *, config):
        del transform, config
        calls.append(("tariff", factory, tuple(seeds)))
        return {"action": [1.1, 0.0, 1.0]}

    def fake_rule(factory, transform, seeds, *, config, initial_action):
        del transform, config
        calls.append(("rule", factory, tuple(seeds), tuple(initial_action)))
        return {"action": [1.2, 0.3, 0.8]}

    monkeypatch.setattr(
        "scripts.v28_learner_ladder.optimize_static_tariff", fake_static
    )
    monkeypatch.setattr(
        "scripts.v28_learner_ladder.optimize_static_policy", fake_rule
    )

    class FakeAgent:
        action_transform = ActionTransform.market_controls()

        def __init__(self) -> None:
            class FakeReplay:
                def __init__(self):
                    self.episodes = []

                def __len__(self):
                    return len(self.episodes)

            self.replay = FakeReplay()
            self.validation_calls = 0
            self.critic_geometry = None

        def configure_critic_action_reference(self, center, scale):
            self.critic_geometry = (np.asarray(center), np.asarray(scale))

        def critic_action_reference_contract(self):
            assert self.critic_geometry is not None
            center, scale = self.critic_geometry
            return {
                "immutable_after_configuration": True,
                "scale_is_units_not_action_bound": True,
                "critics": [
                    {
                        "name": name,
                        "configured": True,
                        "center_coordinates": center.tolist(),
                        "scale_coordinates": scale.tolist(),
                    }
                    for name in ("critic1", "critic2", "target_critic1", "target_critic2")
                ],
            }

        def add_episode(
            self,
            episode,
            *,
            actions_are_physical: bool,
            update_normalizer: bool = True,
        ) -> None:
            assert actions_are_physical
            coordinates = self.action_transform.physical_to_coordinates(
                episode.actions
            ).astype(np.float32)
            self.replay.episodes.append(
                SimpleNamespace(
                    actions=coordinates,
                    observations=episode.observations,
                    rewards=episode.rewards,
                    metadata=episode.metadata,
                    length=episode.length,
                )
            )

        def behavior_clone(self, target, *, steps: int, log_std_coordinates=None):
            assert log_std_coordinates is not None
            self.clone_target = np.asarray(target)
            self.clone_action = self.action_transform.coordinates_to_physical(target)
            return {
                "final_loss": 0.0,
                "steps": steps,
                "reset_action_parity_passed": True,
            }

        def act(self, observation, **kwargs):
            del observation, kwargs
            coordinates = self.action_transform.physical_to_coordinates(
                self.clone_action
            )
            return (
                self.clone_action.copy(),
                None,
                {"policy_coordinates": coordinates},
            )

        def critic_validation(self, episodes):
            self.validation_calls += 1
            error = 100.0 if self.validation_calls == 1 else 1.0
            rows = []
            for episode in episodes:
                total = float(np.sum(episode.rewards))
                probe = int(episode.metadata.get("deviation_day", 0))
                rows.append(
                    {
                        "q0": total,
                        "return0": total,
                        "q_probe": total,
                        "q1_probe": total,
                        "q2_probe": total,
                        "return_probe": total,
                        "probe_transition": probe,
                        "metadata": copy.deepcopy(episode.metadata),
                    }
                )
            return {
                "finite": True,
                "td_mse": error,
                "monte_carlo_mse": error,
                "monte_carlo_normalized_rmse": 0.1,
                "q0_relative_absolute_error": 0.1,
                "q0_to_entropy_scale_ratio": 10.0,
                "episodes": rows,
            }

        def fit_critics_monte_carlo(
            self,
            *,
            steps: int,
            paired_advantage_steps: int,
            paired_advantage_weight: float,
        ):
            assert paired_advantage_weight == 1.0
            assert paired_advantage_steps == 2
            return {"final_loss": 0.0, "steps": steps}

        def configure_safe_policy_improvement(self, **kwargs):
            self.safe_config = kwargs

        def configure_paired_crn_cache(
            self, episodes, *, source_group_id, paired_advantage_weight
        ):
            assert source_group_id == "l0_train_local_support"
            assert paired_advantage_weight == 1.0
            assert episodes
            assert all(
                episode.metadata["profile_split"] == "train_fit"
                and not episode.metadata["heldout_for_gate"]
                for episode in episodes
            )
            return {
                "source": "immutable_train_fit_paired_crn",
                "episode_count": len(episodes),
                "heldout_gate_episodes_used": False,
            }

        def capture_last_passing_state(self, label):
            assert label == "l0_warm_prefit_gate_pass"
            return {"available": True, "label": label, "full_state": True}

    agent = FakeAgent()
    result = warm_start_l0(
        agent,
        training_factory,
        (101, 102),
        (201, 202),
        (301, 302),
        (401, 402),
        absolute_fit_seeds=tuple(range(501, 565)),
        absolute_gate_seeds=(601, 602),
        critic_probe_days=(0,),
        cem_config=CEMConfig(population=2, iterations=1),
        clone_steps=2,
        critic_steps=2,
        paired_advantage_steps=2,
        support_radius=0.1,
        paired_advantage_weight=1.0,
        anchor_coefficient=2.0,
        anchor_decay_updates=5.0,
    )
    assert calls[0][:3] == ("tariff", training_factory, (101, 102))
    assert calls[1][:3] == ("rule", training_factory, (101, 102))
    assert result["target_action"] == [1.2, 0.3, 0.8]
    assert result["profile_split_contract"]["validation_profiles_used"] is False
    assert result["profile_split_contract"]["fit_rollout_seeds_disjoint"]
    assert result["profile_split_contract"]["fit_learning_seeds_disjoint"]
    assert result["profile_split_contract"]["prefill_learning_seeds_disjoint"]
    assert result["deterministic_reset_action_parity"]["passed"]
    assert result["replay_support_coverage"]["passed"]
    assert result["profile_split_contract"]["critic_gate_learning_seeds_disjoint"]
    assert len(result["absolute_fit_rollouts"]) == 64
    assert result["absolute_fit_weather_coverage"]["passed"]
    assert result["critic_gate_crossed_panel_coverage"]["passed"]
    assert result["expected_value_calibration_after"]["passed"]
    assert result["profile_split_contract"]["absolute_gate_disjoint_from_all_learning"]
    assert result["profile_split_contract"][
        "absolute_fit_disjoint_from_all_other_streams"
    ]


def test_train_only_safe_selector_falls_back_without_touching_validation(monkeypatch) -> None:
    factory_calls = []

    def training_factory():
        factory_calls.append("train")
        return object()

    class FakeAgent:
        state = 1

        def policy_training_state(self):
            return {"state": self.state}

        def load_policy_training_state(self, state):
            self.state = int(state["state"])

    def fake_evaluate(factory, agent, seeds, *, deterministic):
        assert factory is training_factory
        assert deterministic
        value = 8.0 if agent.state == 1 else 10.0
        return {
            "mean": value,
            "standard_error": 0.0,
            "median": value,
            "values": [value for _ in seeds],
            "episodes": [
                {
                    "seed": int(seed),
                    "return": value,
                    "days": 1,
                    "actions": [[1.0, 0.0, 1.0]],
                    "infos": [{}],
                }
                for seed in seeds
            ],
        }

    monkeypatch.setattr(
        "scripts.v28_learner_ladder.evaluate_recurrent_agent", fake_evaluate
    )
    agent = FakeAgent()
    result = select_train_safe_policy(
        agent, {"state": 0}, training_factory, (701, 702)
    )
    assert result["selected"] == "warm_reference"
    assert not result["retained_candidate"]
    assert agent.state == 0
    assert result["validation_used"] is False
    assert result["test_used"] is False
    assert not factory_calls  # evaluator spy never instantiated another factory


def test_threshold_cache_provenance_does_not_relabel_legacy_rows(tmp_path) -> None:
    legacy = tmp_path / "legacy.jsonl"
    legacy.write_text(
        "\n".join(json.dumps({"ProfileID": index, "PriceThreshold": 5.0}) for index in range(4))
        + "\n",
        encoding="utf-8",
    )
    contract = _raw_threshold_cache_contract(legacy, loaded_count=4)
    assert not contract["native_uncapped_generation"]
    assert contract["native_unbounded_rows"] == 0
    assert contract["source_contract"].startswith("legacy_bounded")


def test_default_dataset_paths_require_native_uncapped_v28_cache() -> None:
    for city in ("nyc", "boston", "chicago"):
        _, _, profiles = _dataset_paths(city)
        assert profiles == (
            profiles.parents[1] / city / "native_uncapped_primary.jsonl"
        )
        assert profiles.is_file()
        with profiles.open("r", encoding="utf-8") as handle:
            first = json.loads(next(handle))
        assert first["ExperimentalUnboundedEconomics"] is True
        assert first["ThresholdEconomicContract"] == "v28_unbounded_finite_positive"
        assert first["ThresholdPromptProtocol"] == "numeric_profile_threshold_v12_uncapped"
        assert first["ThresholdLLMClipped"] is False
        assert first["ThresholdRawLLMValue"] == first["PriceThreshold"]


def test_threshold_cache_provenance_requires_fail_closed_native_markers(tmp_path) -> None:
    cache = tmp_path / "thresholds.jsonl"
    base = {
        "ProfileID": 1,
        "PriceThreshold": 7.25,
        "ThresholdRawLLMValue": 7.25,
        "ExperimentalUnboundedEconomics": True,
    }
    cache.write_text(json.dumps(base) + "\n", encoding="utf-8")
    weak = _raw_threshold_cache_contract(cache, loaded_count=1)
    assert weak["native_unbounded_rows"] == 1
    assert weak["native_contract_rows"] == 0
    assert not weak["native_uncapped_generation"]

    base.update(
        {
            "PriceThresholdSource": "llm_numeric",
            "ThresholdEconomicContract": "v28_unbounded_finite_positive",
            "ThresholdPromptProtocol": "numeric_profile_threshold_v12_uncapped",
            "ThresholdPromptModel": "gpt-4.1-2025-04-14",
            "ThresholdResponseModel": "gpt-4.1-2025-04-14",
            "ThresholdLLMClipped": False,
        }
    )
    cache.write_text(json.dumps(base) + "\n", encoding="utf-8")
    native = _raw_threshold_cache_contract(cache, loaded_count=1)
    assert native["native_contract_rows"] == 1
    assert native["native_uncapped_generation"]


def test_isolated_rungs_have_no_background_hazards_and_mixed_keeps_all_types() -> None:
    calibration = {}
    l1 = _shock_config("l1", calibration)
    l2 = _shock_config("l2", calibration)
    mixed = _shock_config("mixed", calibration)
    assert l1.types == ("winter_storm",)
    assert l1.storm_daily_hazard == {}
    assert l1.training_shock_opening_share == 0.0
    assert l2.types == ("driver_shortage",)
    assert l2.shortage_rate_per_day == 0.0
    assert l2.training_shock_opening_share == 0.0
    assert mixed.enabled
    assert "driver_shortage" in mixed.types
    assert "winter_storm" in mixed.types
    assert mixed.training_shock_opening_share > 0.0


class _OpeningMarket:
    def __init__(self) -> None:
        self._hours = iter((7, 11, 15, 19, 21, 23))
        self.active_shocks = SimpleNamespace(current_date=dt.date(2024, 7, 4))

    def sample_day_context(self):
        return SimpleNamespace(weather="clear", day_of_week=3)

    def sample_timestep_hour(self):
        return SimpleNamespace(hour=next(self._hours))

    def shock_effect(self, hour: int):
        # An unannounced event becomes public after the opening block.  The
        # daily action must not see that later feature in its initial state.
        return SimpleNamespace(
            public={
                "event_announced": float(hour > 7),
                "transit_availability": 1.0,
            }
        )


def test_daily_public_state_uses_opening_block_and_real_shock_date_only() -> None:
    template = SimpleNamespace(v28_profile_splits={"train": [{}], "validation": [{}], "test": [{}]})
    environment = ContinuousMarketEnv(
        template,
        config=DailyEnvironmentConfig(episode_days=10, customers_per_block=1),
        profile_split="train",
    )
    environment.core = SimpleNamespace(
        market=_OpeningMarket(),
        firm2_mode="static",
        _public_firm2_quote_gaps=lambda **kwargs: {"short": 0.0, "long": 0.0},
    )
    environment.day = 0
    environment._prepare_day()
    assert environment._day_date == dt.date(2024, 7, 4)
    assert environment._day_public["event_announced"] == 0.0
    observation = environment._observation()
    angle = 2.0 * np.pi * (dt.date(2024, 7, 4).timetuple().tm_yday - 1) / 365.0
    assert observation[environment.observation_index["calendar_sin"]] == pytest.approx(np.sin(angle))
    assert observation[environment.observation_index["time_to_go"]] == 1.0


def test_environment_refuses_to_call_exact_copy_information_matched_mpc() -> None:
    environment = ContinuousMarketEnv.__new__(ContinuousMarketEnv)
    with pytest.raises(NotImplementedError, match="belief-state"):
        environment.planning_branch(7, reveal_latent=False)


def test_l2_latent_shortage_timing_and_severity_vary_by_episode_seed() -> None:
    factory = _scenario_factory("l2", 30)
    assert factory is not None
    first = [factory(seed).events[0] for seed in range(24)]
    second = [factory(seed).events[0] for seed in range(24)]
    assert [event.start_day for event in first] == [
        event.start_day for event in second
    ]
    assert len({event.start_day for event in first}) >= 6
    assert len({round(event.intensity, 6) for event in first}) >= 12
    assert all(not event.announced for event in first)
    assert all(event.start_day >= 2 for event in first)
    assert all(event.end_day <= 27 for event in first)


def test_l0_gate_uses_full_three_control_rule_not_tariff_only_static() -> None:
    evaluation = {
        "policies": {
            "best_static_tariff": {"mean": 90.0},
            "two_timescale_rule": {"mean": 100.0},
        },
        "paired": {
            # The old tariff-only gate would pass by a wide margin.
            "rl_minus_static": {"mean": 9.5},
            "rl_minus_rule": {"mean": -0.5},
        },
    }
    gate = gate_stage("l0", evaluation)
    assert gate["passed"]
    assert gate["baseline"] == "best_fixed_full_3d_rule"
    evaluation["paired"]["rl_minus_rule"]["mean"] = -1.01
    assert not gate_stage("l0", evaluation)["passed"]


def test_l2_gate_uses_oracle_gain_over_full_three_control_rule() -> None:
    evaluation = {
        "policies": {},
        "paired": {
            "oracle_minus_rule": {
                "mean": 10.0,
                "standard_error": 1.0,
                "median": 10.0,
                "values": [9.0, 11.0],
                "t": 10.0,
            },
            "rl_minus_rule": {
                "mean": 3.1,
                "standard_error": 0.5,
                "median": 3.1,
                "values": [2.6, 3.6],
                "t": 6.2,
            },
            # Tariff-only deltas are retained only as diagnostics and must not
            # determine the adaptivity gate.
            "oracle_minus_static": {"mean": 20.0},
            "rl_minus_static": {"mean": 13.1},
        },
        "adaptive_opportunity_identified": True,
        "adaptivity_index": 0.31,
    }
    gate = gate_stage("l2", evaluation)
    assert gate["passed"]
    assert gate["adaptivity_index"] == pytest.approx(0.31)
    assert not gate["oracle_is_fair_policy"]


def test_public_schedule_locks_normal_regime_to_fitted_full_rule() -> None:
    class TwoDayEnvironment:
        observation_dim = 1

        def reset(self, seed: int) -> np.ndarray:
            del seed
            self.day = 0
            return np.array([0.0], dtype=np.float32)

        def step(self, action: np.ndarray):
            target = np.array([1.1, 0.2, 0.9]) if self.day == 0 else np.array([1.4, 0.5, 1.2])
            reward = -float(np.square(np.asarray(action) - target).sum())
            self.day += 1
            return (
                np.array([float(self.day)], dtype=np.float32),
                reward,
                self.day >= 2,
                {},
            )

    normal = np.array([1.1, 0.2, 0.9])
    fitted = optimize_public_schedule(
        TwoDayEnvironment,
        ActionTransform.market_controls(),
        (11, 12),
        lambda observation: bool(observation[0] >= 1.0),
        normal,
        config=CEMConfig(population=4, iterations=1, seed=3),
    )
    np.testing.assert_array_equal(fitted["normal_action"], normal)
    assert fitted["normal_regime_fixed_to"] == "best_fixed_full_3d_rule"


def test_small_budget_cem_keeps_at_least_two_elites_and_nonzero_dispersion() -> None:
    optimizer = CEMOptimizer(
        1,
        CEMConfig(
            population=6,
            elite_fraction=0.20,
            iterations=1,
            initial_std=0.5,
            minimum_std=1.0e-9,
            smoothing=0.0,
            seed=7,
        ),
    )
    fitted = optimizer.optimize(
        lambda coordinate: -float((coordinate[0] - 0.8) ** 2)
    )
    assert fitted["elite_count"] == 2
    assert fitted["history"][0]["elite_count"] == 2
    assert fitted["final_std"][0] > 1.0e-9


def test_tail_preflight_covers_open_lower_incentive_and_joint_corner(monkeypatch) -> None:
    actions = []

    def fake_evaluate_policy(environment_factory, policy_factory, seeds):
        del environment_factory
        action = np.asarray(
            policy_factory()(np.zeros(1, dtype=np.float32), 0), dtype=float
        )
        actions.append(tuple(float(value) for value in action))
        # Every open direction has a finite interior maximum at [1, 0, 1].
        score = -float(
            np.log(action[0]) ** 2
            + (action[1] / 4.0) ** 2
            + np.log(action[2]) ** 2
        )
        values = [score for _ in seeds]
        share = float(np.exp(-action[0]))
        return {
            "mean": score,
            "standard_error": 0.0,
            "median": score,
            "values": values,
            "episodes": [
                {"infos": [{"summary": {"completed_share_firm1": share}}]}
                for _ in seeds
            ],
        }

    monkeypatch.setattr(
        "continuous_benchmarks.evaluate_policy", fake_evaluate_policy
    )
    result = evaluate_uncapped_price_tail(lambda: None, (3, 5))
    assert result["passed"]
    assert result["stress_axes"]["driver_incentive_low"]["passed"]
    assert result["stress_axes"]["joint_low_incentive"]["passed"]
    assert (1.0, 0.0, 0.0625) in actions
    assert (16.0, 16.0, 0.0625) in actions
    assert len(actions) == 38


def test_l1_preregistered_gate_is_fifty_percent_of_public_oracle_gain() -> None:
    evaluation = {
        "policies": {},
        "paired": {},
        "adaptive_opportunity_identified": True,
        "adaptivity_index": 0.50,
    }
    assert gate_stage("l1", evaluation)["passed"]
    evaluation["adaptivity_index"] = 0.499
    assert not gate_stage("l1", evaluation)["passed"]


def test_clairvoyant_days_include_shortage_recovery_but_not_observation() -> None:
    factory = _scenario_factory("l2", 30)
    template = SimpleNamespace(
        v28_profile_splits={"train": [{}], "validation": [{}], "test": [{}]},
        shock_config=SimpleNamespace(
            shortage_recovery_days=4,
            storm_supply_recovery_days=2,
        ),
    )
    environment = ContinuousMarketEnv(
        template,
        config=DailyEnvironmentConfig(episode_days=30, customers_per_block=1),
        scenario_factory=factory,
        profile_split="validation",
        shock_sampling_mode="natural",
    )
    event = factory(91).events[0]
    oracle_days = environment.clairvoyant_event_days(91)
    assert oracle_days[0] == event.start_day
    assert oracle_days[-1] == min(29, event.end_day + 4)
    assert not any("latent" in name or "event_day" in name for name in environment.observation_names)


def test_shock_opening_sampling_mode_is_explicit_and_split_independent() -> None:
    template = SimpleNamespace(
        v28_profile_splits={"train": [{}], "validation": [{}], "test": [{}]},
        experimental_unbounded_economics=True,
    )
    train = ContinuousMarketEnv(
        template,
        profile_split="train",
        shock_sampling_mode="training_openings",
    )
    natural_train_profiles = ContinuousMarketEnv(
        template,
        profile_split="train",
        shock_sampling_mode="natural",
    )
    assert train.contract()["shock_sampling_mode"] == "training_openings"
    assert natural_train_profiles.contract()["shock_sampling_mode"] == "natural"
    with pytest.raises(ValueError, match="shock_sampling_mode"):
        ContinuousMarketEnv(
            template,
            profile_split="validation",
            shock_sampling_mode="biased_test",
        )


def test_sampling_mode_is_installed_before_shock_process_reseed() -> None:
    class ProbeCore:
        v28_profile_splits = {"train": [{}], "validation": [{}], "test": [{}]}
        experimental_unbounded_economics = True

        def __init__(self) -> None:
            self.market = SimpleNamespace(shock_training_openings=False)
            self.firm1 = SimpleNamespace(overrides=SimpleNamespace())
            self.firm2 = SimpleNamespace(overrides=SimpleNamespace())
            self.driver_supply = SimpleNamespace(v27_fleet=None)
            self.choice_model = SimpleNamespace(reputation=None)
            self.last_v27_step = {}
            self.reseed_saw_training_opening = None

        def _reseed_environment(self, seed: int) -> None:
            del seed
            self.reseed_saw_training_opening = bool(
                self.market.shock_training_openings
            )

    class ProbeEnvironment(ContinuousMarketEnv):
        def _prepare_day(self) -> None:
            return None

        def _observation(self) -> np.ndarray:
            return np.zeros(self.observation_dim, dtype=np.float32)

    training = ProbeEnvironment(
        ProbeCore(), profile_split="train", shock_sampling_mode="training_openings"
    )
    natural = ProbeEnvironment(
        ProbeCore(), profile_split="train", shock_sampling_mode="natural"
    )
    training.reset(1)
    natural.reset(1)
    assert training.core.reseed_saw_training_opening is True
    assert natural.core.reseed_saw_training_opening is False


def test_frozen_evaluator_retains_observational_welfare_and_equity() -> None:
    evaluation = {
        "mean": 3.0,
        "standard_error": float("nan"),
        "median": 3.0,
        "values": [3.0],
        "episodes": [
            {
                "seed": 7,
                "return": 3.0,
                "days": 2,
                "actions": [[1.0, 0.0, 1.0], [1.1, 0.2, 1.0]],
                "infos": [
                    {
                        "welfare_summary": {
                            "welfare_consumer_surplus_per_request": 4.0,
                            "equity_served_gap_high_minus_low": 0.10,
                        }
                    },
                    {
                        "welfare_summary": {
                            "welfare_consumer_surplus_per_request": 6.0,
                            "equity_served_gap_high_minus_low": 0.20,
                        }
                    },
                ],
            }
        ],
    }
    compact = _compact_evaluation(evaluation)
    welfare = compact["welfare_and_equity"]
    assert welfare["welfare_consumer_surplus_per_request"]["mean"] == 5.0
    assert welfare["equity_served_gap_high_minus_low"]["mean"] == pytest.approx(0.15)
    assert len(compact["episodes"][0]["welfare_path"]) == 2
