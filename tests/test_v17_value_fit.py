"""Registered V17 daily value-fit, target, and resume contracts."""

import copy

import numpy as np
import pytest
import torch

from Core import Core
from test_learning_corrections import collect_episode, recurrent_agent


def _core(*, seed: int = 1761, scale: float = 2.0) -> Core:
    return Core(
        "Chicago",
        seed=seed,
        firm1_mode="RL",
        firm2_mode="static",
        total_customers_pool=8,
        simulation_sample_cap=3,
        enable_driver_supply=False,
        openai_api_key="",
        training_curriculum="v17_progressive",
        action_contract="two_lever_v1",
        v16_crn_tape=True,
        v16_branch_mode="one_day_deviation",
        v16_branch_design="stencil11",
        v16_store_cap=200,
        v16_value_head="concave2d",
        v16_policy_mode="q",
        v16_action_credit_scale=scale,
    )


def _branch_row(core: Core, index: int, regime: str) -> dict:
    observation_dim = (
        core.firm1.state_frame_stack * core.platform_observation_dim
    )
    observation = np.linspace(
        -0.2, 0.2, observation_dim, dtype=np.float32
    )
    observation = observation + np.float32(index * 0.001)
    cells = np.asarray(core.v16_value_cells, dtype=np.float32)
    coordinates = cells - 1.0
    values = (
        0.35 * coordinates[:, 0]
        - 0.15 * coordinates[:, 1]
        - 0.4 * coordinates[:, 0] ** 2
        - 0.25 * coordinates[:, 1] ** 2
    ).astype(np.float32)
    incumbent = 12
    values = values - values[incumbent]
    components = np.stack(
        (values, -0.5 * values, 0.1 * values, -0.1 * values), axis=1
    ).astype(np.float32)
    return {
        "observation_window": observation.tolist(),
        "observation_shape": [
            core.firm1.state_frame_stack,
            core.platform_observation_dim,
        ],
        "incumbent_action": incumbent,
        "selected_action": incumbent,
        "candidate_actions": list(range(25)),
        "action_credits": values.tolist(),
        "action_credit_mask": [True] * 25,
        "component_credit_names": list(core.v16_value_auxiliary_names),
        "component_credit_matrix": components.tolist(),
        "regime_label": regime,
        "regime_events": [] if regime == "calm" else [regime],
        "stage": "profit",
        "rival": "static",
        "day": index,
        "tape_seed": 1761,
        "tape_id": index,
        "audit": True,
        "horizon_days": 7,
    }


def _populate(core: Core, rows: int = 12) -> None:
    for index in range(rows):
        core.v16_branch_store.add(
            _branch_row(
                core,
                index,
                "shortage" if index % 3 == 0 else "calm",
            )
        )


def _parameter_state(module: torch.nn.Module) -> dict:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def test_value_learning_rate_is_registered_absolute_cosine():
    core = _core()
    assert core._v16_value_learning_rate(0.0) == pytest.approx(3e-4)
    assert core._v16_value_learning_rate(0.5) == pytest.approx(1.75e-4)
    assert core._v16_value_learning_rate(1.0) == pytest.approx(5e-5)


def test_t0_12_replay_labels_use_the_current_stage_objective():
    core = _core(seed=1766)
    row = _branch_row(core, 0, "calm")
    components = np.asarray(row["component_credit_matrix"], dtype=np.float32)

    market = core.training_stage_scheduler.stage_at(0.0)
    core.current_training_stage = market
    market_weights = core._v17_current_component_weights()
    stored = components @ market_weights
    row["action_credits"] = stored.tolist()
    original_audit_values = list(row["action_credits"])
    market_batch = core._v16_value_fit_batch([row])[2][0].cpu().numpy()
    assert market_batch == pytest.approx(stored, abs=1e-6)

    profit = core.training_stage_scheduler.stage_at(0.999999)
    assert profit.opponent_regime == "profit"
    core.current_training_stage = profit
    profit_weights = core._v17_current_component_weights()
    relabeled = core._v16_value_fit_batch([row])[2][0].cpu().numpy()
    expected = components @ profit_weights
    assert relabeled == pytest.approx(expected, abs=1e-6)
    assert not np.allclose(relabeled, stored, rtol=0.0, atol=1e-6)
    assert row["action_credits"] == original_audit_values


def test_daily_fit_runs_exact_steps_losses_and_polyak_update():
    core = _core(seed=1762)
    _populate(core)
    target_before = _parameter_state(core.v16_value_target_model)
    metrics = core._v16_fit_value_model(training_progress=0.25)
    assert metrics["performed"] is True
    assert metrics["optimizer_steps"] == 50
    assert metrics["batch_size"] == 64
    assert metrics["huber_delta"] == pytest.approx(1.0)
    assert metrics["auxiliary_weight"] == pytest.approx(0.25)
    assert metrics["polyak"] == pytest.approx(0.99)
    assert metrics["regime_balanced_sampling"] is True
    assert np.isfinite(metrics["total_loss"])
    assert core.v16_value_fit_calls == 1
    assert core.v16_value_optimizer_steps == 50
    for name, target_after in core.v16_value_target_model.state_dict().items():
        expected = (
            0.99 * target_before[name]
            + 0.01 * core.v16_value_model.state_dict()[name].detach().cpu()
        )
        assert torch.allclose(target_after.detach().cpu(), expected)


def test_value_fit_resume_reproduces_exact_next_update(tmp_path):
    core = _core(seed=1763)
    _populate(core)
    core._v16_fit_value_model(training_progress=0.2)
    artifact = tmp_path / "value-fit.pt"
    core.save_trained_model(path=str(artifact), model_id="value-fit")

    expected_metrics = core._v16_fit_value_model(training_progress=0.4)
    expected_online = _parameter_state(core.v16_value_model)
    expected_target = _parameter_state(core.v16_value_target_model)
    expected_optimizer = copy.deepcopy(core.v16_value_optimizer.state_dict())

    resumed = _core(seed=1763)
    resumed.load_trained_model(str(artifact), load_optimizer=True)
    actual_metrics = resumed._v16_fit_value_model(training_progress=0.4)
    assert actual_metrics == expected_metrics
    for name, value in resumed.v16_value_model.state_dict().items():
        assert torch.equal(value.detach().cpu(), expected_online[name])
    for name, value in resumed.v16_value_target_model.state_dict().items():
        assert torch.equal(value.detach().cpu(), expected_target[name])
    actual_optimizer = resumed.v16_value_optimizer.state_dict()
    assert actual_optimizer["param_groups"] == expected_optimizer["param_groups"]
    for parameter, state in expected_optimizer["state"].items():
        for key, expected in state.items():
            actual = actual_optimizer["state"][parameter][key]
            if torch.is_tensor(expected):
                assert torch.equal(actual, expected)
            else:
                assert actual == expected


def test_q_mode_disables_ppo_actor_and_uses_weekly_critic_contract():
    core = _core(seed=1764)
    assert core.firm1.agent.ent_coeff == 0.0
    assert core.firm1.agent.action_q_coeff == 0.0
    assert core.firm1.agent.counterfactual_actor_weight == 0.0
    assert core.firm1.agent.branch_imitation_coeff == 0.0


def test_zero_actor_weight_updates_only_the_weekly_critic():
    agent = recurrent_agent()
    collect_episode(agent)
    actor_before = {
        name: value.detach().clone()
        for name, value in agent.net.named_parameters()
        if id(value) in {id(parameter) for parameter in agent.net.actor_parameters()}
    }
    critic_before = {
        name: value.detach().clone()
        for name, value in agent.net.named_parameters()
        if id(value) in {id(parameter) for parameter in agent.net.critic_parameters()}
    }
    metrics = agent.update(
        epochs=2, batch_size=4, actor_loss_weight=0.0
    )
    assert metrics["update_performed"] is True
    assert metrics["actor_optimizer_steps"] == 0
    assert metrics["actor_epochs_requested"] == 0
    assert metrics["critic_optimizer_steps"] > 0
    for name, before in actor_before.items():
        assert torch.equal(agent.net.state_dict()[name], before)
    assert any(
        not torch.equal(agent.net.state_dict()[name], before)
        for name, before in critic_before.items()
    )


def test_action_credit_scale_is_a_resume_contract(tmp_path):
    core = _core(seed=1765, scale=2.0)
    artifact = tmp_path / "scale.pt"
    core.save_trained_model(path=str(artifact), model_id="scale")
    other = _core(seed=1765, scale=3.0)
    with pytest.raises(ValueError, match="value-model configuration"):
        other.load_trained_model(str(artifact), load_optimizer=True)
