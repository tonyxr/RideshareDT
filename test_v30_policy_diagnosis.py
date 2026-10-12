"""Regression tests for the v30 reward/clock/deployment mismatch."""
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from Core import _apply_v30_cli_contract
from PPOAgent import PPOAgent
from pricing_models import FirmMetrics
from test_v30_core_contract import _core


def test_v30_deploys_learned_distribution_unless_argmax_explicit():
    args = SimpleNamespace(v30=True, policy_objective="profit_dominance_v30",
                           eval_policy_mode="argmax_raw")
    _apply_v30_cli_contract(args, [])
    assert args.eval_policy_mode == "sample_raw"
    args.eval_policy_mode = "argmax_raw"
    _apply_v30_cli_contract(args, ["--eval_policy_mode"])
    assert args.eval_policy_mode == "argmax_raw"


@pytest.mark.parametrize("steps_per_day", [4, 10, 24])
def test_estimator_clock_uses_market_time_not_block_count(steps_per_day):
    core = _core(shock_observations=False)
    core.shared_steps_per_day = steps_per_day
    metric = FirmMetrics(total=100, chosen=50, completed=40, profit=100)
    for _ in range(steps_per_day):
        core._ingest_platform_observations(metric, metric)
    assert core.v30_state_estimator.last_time == datetime(2000, 1, 2, tzinfo=timezone.utc)


def test_v30_episodic_returns_not_differential_reward():
    core = _core()
    assert core.firm1.agent.center_rewards is False
    assert core.credit_assignment_config["center_rewards"] is False


@pytest.mark.parametrize("center_rewards", [False, True])
def test_actual_ppo_update_preserves_discounted_episode_rewards(center_rewards):
    torch.manual_seed(17)
    agent = PPOAgent(state_dim=3, action_dim=2, gamma=0.9, lam=1.0,
                     center_rewards=center_rewards, device="cpu")
    with torch.no_grad():
        for parameter in agent.net.critic_parameters():
            parameter.zero_()
    rewards = [2.0] * 7 + [5.0]  # constant gains plus a terminal bonus
    for i, reward in enumerate(rewards):
        action, state, logp, value, features, mask = agent.act([i / 8, 0, 0])
        agent.store(state, action, reward, i == 7, None, logp, value,
                    action_features=features, action_mask=mask)
    result = agent.update(epochs=1, batch_size=8, bootstrap_value=0.0)
    expected, running = [], 0.0
    baseline = np.mean(rewards) if center_rewards else 0.0
    for reward in reversed(rewards):
        running = reward - baseline + 0.9 * running
        expected.append(running)
    assert result["critic_target_mean"] == pytest.approx(np.mean(expected), abs=1e-5)
    assert result["center_rewards"] is center_rewards


def test_argmax_is_not_equivalent_to_the_incremental_stochastic_policy():
    torch.manual_seed(91)
    agent = PPOAgent(state_dim=3, action_dim=3, device="cpu")
    with torch.no_grad():
        for parameter in agent.net.actor_parameters():
            parameter.zero_()
        agent.net.pi_head.bias.copy_(torch.log(torch.tensor([0.1, 0.45, 0.45])))
    samples = [agent.act([0, 0, 0], deterministic=True, policy_mode="sample_raw")[0]
               for _ in range(600)]
    assert 200 < samples.count(1) < 340
    assert 200 < samples.count(2) < 340
    assert len({agent.act([0, 0, 0], deterministic=True, policy_mode="argmax_raw")[0]
                for _ in range(20)}) == 1


def test_legacy_checkpoint_can_be_evaluated_but_not_silently_resumed(tmp_path):
    core = _core()
    path = tmp_path / "old.pt"
    core.save_trained_model(str(path))
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    del artifact["credit_assignment_config"]["center_rewards"]
    torch.save(artifact, path)
    core.load_trained_model(str(path), load_optimizer=False)
    with pytest.raises(ValueError, match="center_rewards"):
        core.load_trained_model(str(path), load_optimizer=True)
