from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import scripts.v28_recurrent_ppo_control as control
from continuous_benchmarks import CEMConfig
from recurrent_sac import ActionTransform
from recurrent_ppo import RecurrentPPOAgent, RecurrentPPOConfig
from scripts.v28_recurrent_ppo_control import (
    NUMERICAL_POLICY_PENALTY,
    _adaptivity_statistics,
    _evaluate_fitted_policies,
    _make_environment_factory,
    _optimize_public_event_action,
    _stage_seeds,
    collect_on_policy_episode,
    gate_stage,
    train_ppo_stage,
)


class _TinyDailyEnv:
    observation_dim = 4

    def __init__(self, days: int = 4) -> None:
        self.days = int(days)
        self.day = 0

    def reset(self, seed: int) -> np.ndarray:
        self.day = 0
        return np.asarray([seed % 7, 0.0, 0.0, 1.0], dtype=np.float32)

    def step(self, action: np.ndarray):
        self.day += 1
        done = self.day >= self.days
        observation = np.asarray(
            [0.0, self.day / self.days, float(action[1]), 1.0 - self.day / self.days],
            dtype=np.float32,
        )
        reward = 1.0 - float(np.square(np.log(action[0])))
        return observation, reward, done, {"active_shocks": []}


class _MustNotStepEnv:
    def reset(self, seed: int) -> np.ndarray:
        del seed
        return np.zeros(4, dtype=np.float32)

    def step(self, action: np.ndarray):
        raise AssertionError(f"invalid physical action must not be substituted: {action}")


def _agent(*, transitions: int = 4, seed: int = 7) -> RecurrentPPOAgent:
    return RecurrentPPOAgent(
        4,
        config=RecurrentPPOConfig(
            hidden_dim=8,
            gamma=1.0,
            gae_lambda=0.9,
            rollout_transitions=transitions,
            minibatch_episodes=1,
            update_epochs=1,
            target_kl=None,
            seed=seed,
        ),
    )


def test_collector_stores_complete_episode_and_exact_behavior_probabilities() -> None:
    agent = _agent()
    episode, log_probabilities, row = collect_on_policy_episode(
        agent, _TinyDailyEnv(days=4), seed=101
    )

    assert episode.length == 4
    assert episode.dones.tolist() == [0.0, 0.0, 0.0, 1.0]
    assert log_probabilities.shape == (4,)
    assert np.isfinite(log_probabilities).all()
    assert row["behavior_log_probabilities_stored"] == 4
    assert row["daily_action_draws"] == 4
    assert row["option_persistence"] is False

    agent.add_episode(
        episode,
        actions_are_physical=False,
        behavior_log_probabilities=log_probabilities,
    )
    metrics = agent.update()
    assert metrics["episodes"] == 1.0
    assert metrics["transitions"] == 4.0
    diagnostics = agent.action_diagnostics()
    assert diagnostics["behavior_log_probability_checks"] == 4
    assert diagnostics["max_behavior_log_probability_error"] < 1e-4


def test_corrupted_behavior_probability_fails_before_optimizer_step() -> None:
    agent = _agent()
    episode, log_probabilities, _ = collect_on_policy_episode(
        agent, _TinyDailyEnv(days=4), seed=102
    )
    log_probabilities[2] += 0.25
    agent.add_episode(
        episode,
        actions_are_physical=False,
        behavior_log_probabilities=log_probabilities,
    )
    actor_before = [parameter.detach().clone() for parameter in agent.actor.parameters()]
    with pytest.raises(RuntimeError, match="do not reconstruct"):
        agent.update()
    assert all(
        torch.equal(before, after.detach())
        for before, after in zip(actor_before, agent.actor.parameters())
    )


def test_invalid_uncapped_draw_is_terminal_and_not_clipped_or_substituted() -> None:
    agent = _agent(transitions=1)
    with torch.no_grad():
        for parameter in agent.actor.parameters():
            parameter.zero_()
        output = agent.actor.head[-1]
        assert isinstance(output, torch.nn.Linear)
        output.bias[0] = 1_000.0
        output.bias[agent.action_dim :] = -5.0

    episode, log_probabilities, row = collect_on_policy_episode(
        agent,
        _MustNotStepEnv(),
        seed=103,
        deterministic=True,
    )
    assert episode.length == 1
    assert episode.actions[0, 0] == pytest.approx(1_000.0)
    assert episode.rewards[0] == pytest.approx(NUMERICAL_POLICY_PENALTY)
    assert episode.dones[0] == 1.0
    assert row["numerical_invalid_actions"] == 1
    assert np.isfinite(log_probabilities).all()
    assert agent.nonfinite_action_rejections == 1


def test_stage_training_flushes_only_complete_on_policy_episodes() -> None:
    agent = _agent(transitions=8)
    logs = train_ppo_stage(agent, lambda: _TinyDailyEnv(days=4), [11, 12, 13])
    assert len(logs) == 3
    assert agent.pending_transitions == 0
    assert agent.updates == 2
    assert sum(int(row["days"]) for row in logs) == 12
    assert agent.action_diagnostics()["behavior_log_probability_checks"] == 12


def test_control_rejects_any_test_split_factory() -> None:
    with pytest.raises(ValueError, match="only train or validation"):
        _make_environment_factory(
            object(),
            object(),  # type: ignore[arg-type]
            None,
            profile_split="test",
            training=False,
        )


def test_seed_streams_are_disjoint_and_have_no_test_role() -> None:
    train = set(_stage_seeds(9, "l2", 12, "train"))
    fit = set(_stage_seeds(9, "l2", 12, "validation_fit"))
    gate = set(_stage_seeds(9, "l2", 12, "validation_gate"))
    tail = set(_stage_seeds(9, "l2", 12, "tail"))
    assert not (train & fit or train & gate or fit & gate or tail & train or tail & gate)
    with pytest.raises(ValueError, match="exposes only"):
        _stage_seeds(9, "l2", 1, "test")


def test_pilot_budget_parity_is_documented_in_runner_source() -> None:
    source = (
        Path(__file__).parent / "scripts" / "v28_recurrent_ppo_control.py"
    ).read_text()
    assert "preset.train_episodes" in source
    assert "preset.episode_days" in source
    assert "preset.customers_per_block" in source
    assert '"opened": False' in source
    assert "gamma=1" in source


def _estimate(mean: float, standard_error: float = 0.1):
    return {
        "mean": float(mean),
        "standard_error": float(standard_error),
        "median": float(mean),
        "values": [float(mean - standard_error), float(mean + standard_error)],
    }


def test_adaptivity_uses_full_3d_rule_for_public_and_latent_shocks() -> None:
    paired = {
        # These tariff-static values are deliberately much larger; neither is
        # allowed into the adaptivity numerator or denominator.
        "ppo_minus_static": _estimate(100.0),
        "schedule_minus_static": _estimate(110.0),
        "oracle_minus_static": _estimate(120.0),
        "ppo_minus_rule": _estimate(2.0),
        "schedule_minus_rule": _estimate(4.0),
        "oracle_minus_rule": _estimate(8.0),
    }
    public = _adaptivity_statistics("l1", paired)
    latent = _adaptivity_statistics("l2", paired)

    assert public["adaptive_opportunity_baseline"] == "best_fixed_full_3d_rule"
    assert public["adaptive_opportunity_pair"] == "schedule_minus_rule"
    assert public["adaptivity_index"] == pytest.approx(0.5)
    assert latent["adaptive_opportunity_pair"] == "oracle_minus_rule"
    assert latent["adaptivity_index"] == pytest.approx(0.25)


def test_public_event_optimizer_holds_normal_regime_at_full_rule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    normal = np.asarray([1.17, -0.4, 1.09], dtype=np.float64)
    observed = []

    def fake_evaluate(environment_factory, policy_factory, seeds):
        del environment_factory, seeds
        policy = policy_factory()
        observed.append(policy.normal_action.copy())
        return _estimate(3.0)

    def fake_optimize(self, objective, *, initial_mean=None, initial_std=None):
        del self, initial_std
        coordinates = np.asarray(initial_mean, dtype=np.float64)
        score = objective(coordinates)
        return {
            "coordinates": coordinates,
            "score": score,
            "final_mean": coordinates,
            "final_std": np.ones(3),
            "history": [],
        }

    monkeypatch.setattr(control, "evaluate_policy", fake_evaluate)
    monkeypatch.setattr(control.CEMOptimizer, "optimize", fake_optimize)
    result = _optimize_public_event_action(
        lambda: object(),
        ActionTransform.market_controls(),
        [1, 2],
        lambda observation: bool(observation[0]),
        normal,
        config=CEMConfig(population=2, iterations=1),
    )

    np.testing.assert_allclose(result["normal_action"], normal, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(observed[0], normal, rtol=0.0, atol=0.0)
    assert result["normal_regime_fixed_to"] == "best_fixed_full_3d_rule"


def test_fitted_policy_evaluation_rejects_schedule_normal_not_equal_to_rule() -> None:
    class Probe:
        @staticmethod
        def public_event_indicator(observation):
            del observation
            return False

    fits = {
        "best_static_tariff": {"action": [1.0, 0.0, 1.0]},
        "two_timescale_rule": {"action": [1.1, 0.2, 1.05]},
        "public_schedule": {
            "normal_action": [1.0, 0.0, 1.0],
            "event_action": [1.2, 0.4, 1.1],
        },
    }
    with pytest.raises(RuntimeError, match="normal action must equal"):
        _evaluate_fitted_policies(
            "l1", object(), lambda: Probe(), [1, 2], fits  # type: ignore[arg-type]
        )


def test_l0_gate_uses_best_fixed_full_3d_rule_not_tariff_static() -> None:
    evaluation = {
        "policies": {
            "best_static_tariff": {"mean": 100.0},
            "two_timescale_rule": {"mean": 120.0},
        },
        "paired": {
            # PPO is well above tariff-static but more than 1% below the
            # stronger full-control rule, so the corrected gate must fail.
            "ppo_minus_static": _estimate(18.0),
            "ppo_minus_rule": _estimate(-2.0),
        },
    }
    gate = gate_stage("l0", evaluation)
    assert gate["baseline"] == "best_fixed_full_3d_rule"
    assert gate["tolerance"] == pytest.approx(1.2)
    assert gate["delta"] == pytest.approx(-2.0)
    assert gate["passed"] is False
    assert gate["tariff_static_diagnostic"]["mean"] == pytest.approx(100.0)
