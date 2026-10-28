"""Registered V17 concave two-dimensional value-head contracts."""

import itertools

import pytest
import torch

from Core import Core
from pricing_models import TWO_LEVER_LEVELS
from v17_value_model import V17ConcaveValueModel


AUXILIARIES = (
    "own_profit_per_request",
    "rival_profit_per_request",
    "own_completed_share",
    "rival_completed_share",
)


def _cells() -> torch.Tensor:
    return torch.tensor(
        list(itertools.product(TWO_LEVER_LEVELS, repeat=2)),
        dtype=torch.float32,
    )


def _model() -> V17ConcaveValueModel:
    return V17ConcaveValueModel(
        observation_dim=6,
        frame_stack=4,
        hidden_dim=16,
        auxiliary_names=AUXILIARIES,
    )


def test_head_returns_main_and_component_surfaces_for_all_cells():
    model = _model()
    result = model(torch.zeros(3, 4, 6), _cells())
    assert result["q"].shape == (3, 25)
    assert result["auxiliary"].shape == (3, 25, len(AUXILIARIES))
    assert result["linear"].shape == (3, 1 + len(AUXILIARIES), 2)
    assert result["curvature"].shape == (
        3,
        1 + len(AUXILIARIES),
        2,
        2,
    )


def test_curvature_is_positive_definite_and_allows_interaction():
    model = _model()
    with torch.no_grad():
        model.surface_head.bias[3:6] = torch.tensor([0.5, 0.7, 0.4])
    curvature = model.surface_parameters(torch.zeros(4, 6))[2]
    eigenvalues = torch.linalg.eigvalsh(curvature)
    assert torch.all(eigenvalues >= model.curvature_floor - 1e-8)
    assert curvature[0, 0, 0, 1].abs() > 0.0


def test_surface_is_concave_in_both_levers():
    model = _model()
    with torch.no_grad():
        model.surface_head.bias[3:6] = torch.tensor([0.5, 0.2, 0.4])
    observation = torch.randn(4, 6)
    endpoints = torch.tensor([[0.8, 1.5], [1.5, 0.8]])
    midpoint = torch.tensor([[1.15, 1.15]])
    q_end = model(observation, endpoints)["q"][0]
    q_mid = model(observation, midpoint)["q"][0, 0]
    assert q_mid >= 0.5 * (q_end[0] + q_end[1]) - 1e-7


def test_warm_start_has_unique_card_argmax():
    model = _model()
    q = model(torch.zeros(4, 6), _cells())["q"][0]
    best = int(torch.argmax(q).item())
    assert tuple(_cells()[best].tolist()) == pytest.approx((1.0, 1.0))
    assert int((q == q.max()).sum().item()) == 1


def test_public_encoder_receives_value_gradients():
    model = _model()
    with torch.no_grad():
        torch.nn.init.normal_(model.surface_head.weight, std=0.05)
    observations = torch.randn(2, 4, 6, requires_grad=True)
    loss = model(observations, _cells())["q"].square().mean()
    loss.backward()
    assert observations.grad is not None
    assert observations.grad.abs().sum() > 0.0
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0.0
        for parameter in model.frame_encoder.parameters()
    )


def _core(*, seed: int) -> Core:
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
    )


def test_core_value_model_artifact_round_trip(tmp_path):
    core = _core(seed=1731)
    assert core.v16_value_model is not None
    assert core.v16_value_auxiliary_names == AUXILIARIES
    with torch.no_grad():
        core.v16_value_model.surface_head.bias[1] = 0.125
    observation = torch.zeros(
        core.firm1.state_frame_stack,
        core.platform_observation_dim,
        device=core.firm1.agent.device,
    )
    cells = _cells().to(core.firm1.agent.device)
    expected = core.v16_value_model(observation, cells)["q"].detach().cpu()
    artifact = tmp_path / "value-policy.pt"
    core.save_trained_model(path=str(artifact), model_id="value-roundtrip")

    resumed = _core(seed=1731)
    resumed.load_trained_model(str(artifact), load_optimizer=True)
    actual = resumed.v16_value_model(observation, cells)["q"].detach().cpu()
    assert torch.equal(actual, expected)


def test_value_head_defaults_off_and_rejects_missing_store():
    legacy = Core(
        "Chicago",
        seed=1732,
        firm1_mode="RL",
        firm2_mode="static",
        total_customers_pool=8,
        simulation_sample_cap=3,
        enable_driver_supply=False,
        openai_api_key="",
    )
    assert legacy.v16_value_head == "none"
    assert legacy.v16_value_model is None
    with pytest.raises(ValueError, match="requires v16_store_cap"):
        Core(
            "Chicago",
            seed=1733,
            firm1_mode="RL",
            firm2_mode="static",
            total_customers_pool=8,
            simulation_sample_cap=3,
            enable_driver_supply=False,
            openai_api_key="",
            v16_value_head="concave2d",
        )

