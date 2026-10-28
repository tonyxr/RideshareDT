"""Registered V17 Q-policy distribution and target-network contracts."""

import torch
import pytest

from Core import Core


def _core(*, seed: int = 1741, mode: str = "q") -> Core:
    return Core(
        "Chicago",
        seed=seed,
        firm1_mode="RL",
        firm2_mode="static",
        total_customers_pool=8,
        simulation_sample_cap=3,
        enable_driver_supply=False,
        openai_api_key="",
        action_contract="two_lever_v1",
        v16_crn_tape=True,
        v16_branch_mode="one_day_deviation",
        v16_branch_design="stencil11",
        v16_store_cap=20,
        v16_value_head="concave2d",
        v16_policy_mode=mode,
    )


def _state(core: Core) -> torch.Tensor:
    return torch.zeros(
        core.firm1.state_frame_stack * core.platform_observation_dim,
        dtype=torch.float32,
    )


def test_q_policy_requires_registered_value_head():
    with pytest.raises(ValueError, match="requires the concave2d"):
        Core(
            "Chicago",
            seed=1742,
            firm1_mode="RL",
            firm2_mode="static",
            total_customers_pool=8,
            simulation_sample_cap=3,
            enable_driver_supply=False,
            openai_api_key="",
            v16_policy_mode="q",
        )


def test_target_network_starts_equal_frozen_and_independent():
    core = _core()
    online = core.v16_value_model
    target = core.v16_value_target_model
    assert online is not None and target is not None
    assert not any(parameter.requires_grad for parameter in target.parameters())
    for online_value, target_value in zip(
        online.state_dict().values(), target.state_dict().values()
    ):
        assert torch.equal(online_value, target_value)
    with torch.no_grad():
        online.surface_head.bias[1] = 1.0
    assert target.surface_head.bias[1].item() == 0.0


def test_absolute_temperature_uses_registered_cosine_schedule():
    core = _core()
    assert core._v16_q_temperature(0.0) == pytest.approx(0.10)
    assert core._v16_q_temperature(0.5) == pytest.approx(0.065)
    assert core._v16_q_temperature(1.0) == pytest.approx(0.03)


def test_warm_start_evaluation_argmax_is_published_card():
    core = _core()
    action, _, diagnostics = core._v16_q_policy_decision(
        _state(core),
        torch.ones(25, dtype=torch.bool),
        deterministic=True,
        training_progress=1.0,
    )
    assert core.firm1.action_to_two_lever_cell[action] == (1.0, 1.0)
    assert diagnostics["v16_q_epsilon"] == 0.0
    assert diagnostics["v16_q_posted_distance"] == 1.0
    assert diagnostics["v16_q_posted_time"] == 1.0


def test_training_probability_is_softmax_plus_uniform_exploration():
    core = _core(seed=1743)
    with torch.no_grad():
        core.v16_value_target_model.surface_head.bias[1] = 0.4
        core.v16_value_target_model.surface_head.bias[2] = -0.2
    state = _state(core)
    mask = torch.ones(25, dtype=torch.bool)
    action, log_probability, diagnostics = core._v16_q_policy_decision(
        state,
        mask,
        deterministic=False,
        training_progress=0.0,
    )
    with torch.no_grad():
        cells = torch.as_tensor(core.v16_value_cells)
        q_values = core.v16_value_target_model(state.reshape(1, -1), cells)[
            "q"
        ][0]
        target = torch.softmax(q_values / 0.10, dim=-1)
        expected = 0.95 * target + 0.05 / 25.0
    assert log_probability.item() == pytest.approx(
        torch.log(expected[action]).item(), abs=5e-7
    )
    assert diagnostics["v16_q_temperature"] == pytest.approx(0.10)
    assert diagnostics["v16_q_epsilon"] == pytest.approx(0.05)
    assert diagnostics["v16_q_policy_entropy"] >= diagnostics[
        "v16_q_target_entropy"
    ]


def test_core_action_path_uses_q_target_not_ppo_actor():
    core = _core(seed=1744)
    with torch.no_grad():
        core.v16_value_target_model.surface_head.bias[1] = 2.0
        core.v16_value_target_model.surface_head.bias[2] = -0.5
    state = _state(core)
    features = torch.zeros(
        core.firm1.agent.action_dim,
        core.firm1.agent.action_feature_dim,
    )
    result, diagnostics = core._firm1_policy_action(
        state,
        deterministic=True,
        action_features=features,
        action_mask=torch.ones(25, dtype=torch.bool),
        critic_state=None,
        training_progress=1.0,
    )
    action = result[0]
    assert action == int(diagnostics["policy_top_action"])
    assert core.firm1.action_to_two_lever_cell[action][0] == 1.5
    assert diagnostics["optimizer_ready"] == 1.0


def test_q_target_and_policy_config_round_trip(tmp_path):
    core = _core(seed=1745)
    with torch.no_grad():
        core.v16_value_target_model.surface_head.bias[1] = 0.75
    artifact = tmp_path / "q-policy.pt"
    core.save_trained_model(path=str(artifact), model_id="q-roundtrip")
    resumed = _core(seed=1745)
    resumed.load_trained_model(str(artifact), load_optimizer=True)
    assert resumed.v16_policy_mode == "q"
    assert torch.equal(
        resumed.v16_value_target_model.surface_head.bias,
        core.v16_value_target_model.surface_head.bias,
    )


def test_policy_defaults_remain_ppo():
    core = Core(
        "Chicago",
        seed=1746,
        firm1_mode="RL",
        firm2_mode="static",
        total_customers_pool=8,
        simulation_sample_cap=3,
        enable_driver_supply=False,
        openai_api_key="",
    )
    assert core.v16_policy_mode == "ppo"
    assert core.v16_value_target_model is None
