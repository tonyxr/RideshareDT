"""Fast-path learner contracts required by the v17 results plan."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from Core import Core


def _core(
    *,
    seed: int = 17101,
    move_radius: int = 1,
    temperature_mode: str = "relative",
) -> Core:
    return Core(
        "Chicago",
        seed=seed,
        firm1_mode="RL",
        firm2_mode="static",
        total_customers_pool=8,
        simulation_sample_cap=3,
        enable_driver_supply=False,
        openai_api_key="",
        action_contract="two_lever_v3",
        v16_crn_tape=True,
        v16_branch_mode="one_day_deviation",
        v16_branch_design="stencil11",
        v16_store_cap=40,
        v16_value_head="concave2d",
        v16_policy_mode="q",
        v17_move_radius=move_radius,
        v17_temperature_mode=temperature_mode,
        v17_tau_rel_start=0.50,
        v17_tau_rel_end=0.15,
    )


def _state(core: Core) -> torch.Tensor:
    return torch.zeros(
        core.firm1.state_frame_stack * core.platform_observation_dim,
        dtype=torch.float32,
    )


class _FixedQ(torch.nn.Module):
    def __init__(self, values: torch.Tensor):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.register_buffer("values", values.clone().float())

    def forward(self, observation, cells):
        del cells
        q = self.values.reshape(1, -1).expand(observation.shape[0], -1)
        return {"q": q + 0.0 * self.anchor}


def _incumbent_action(core: Core, cell=(1.25, 1.25)) -> int:
    for action, value in core.firm1.action_to_two_lever_cell.items():
        if tuple(value) == tuple(cell):
            return int(action)
    raise AssertionError(cell)


def test_flags_off_identity():
    core = _core(move_radius=0, temperature_mode="absolute")
    assert core.v17_move_radius == 0
    assert core.v17_temperature_mode == "absolute"
    assert core._v17_move_mask(0).all()
    assert core._v17_tau_rel(0.0) == pytest.approx(0.50)
    assert core._v17_tau_rel(1.0) == pytest.approx(0.15)


def test_move_mask_shapes():
    core = _core()
    interior = core._v17_move_mask(_incumbent_action(core))
    edge = core._v17_move_mask(_incumbent_action(core, (0.8, 1.25)))
    corner = core._v17_move_mask(_incumbent_action(core, (0.8, 0.8)))
    assert int(interior.sum()) == 9
    assert int(edge.sum()) == 6
    assert int(corner.sum()) == 4
    assert interior[_incumbent_action(core)]
    core.v17_move_radius = 0
    assert int(core._v17_move_mask(0).sum()) == 36


def test_q_policy_respects_move_radius(monkeypatch):
    core = _core(seed=17102)
    incumbent = _incumbent_action(core)
    monkeypatch.setattr(core, "_v16_incumbent_two_lever_action", lambda: incumbent)
    values = torch.linspace(-1.0, 1.0, 36)
    core.v16_value_target_model = _FixedQ(values)
    allowed = core._v17_move_mask(incumbent)
    mask = torch.ones(36, dtype=torch.bool)
    for _ in range(2000):
        action, _, _ = core._v16_q_policy_decision(
            _state(core), mask, deterministic=False, training_progress=0.5
        )
        assert allowed[action]
    action, _, _ = core._v16_q_policy_decision(
        _state(core), mask, deterministic=True, training_progress=0.5
    )
    assert allowed[action]


def test_relative_temperature(monkeypatch):
    core = _core(seed=17103)
    incumbent = _incumbent_action(core)
    monkeypatch.setattr(core, "_v16_incumbent_two_lever_action", lambda: incumbent)
    allowed = core._v17_move_mask(incumbent)
    mask = torch.ones(36, dtype=torch.bool)
    values = torch.linspace(-2.0, 3.0, 36)

    def probabilities(scale: float) -> np.ndarray:
        core.v16_value_target_model = _FixedQ(values * scale)
        state = _state(core)
        with torch.no_grad():
            q = core.v16_value_target_model(
                state.reshape(1, -1), torch.as_tensor(core.v16_value_cells)
            )["q"][0]
            aq = q[torch.as_tensor(allowed)]
            spread = aq.max() - aq.min()
            logits = (q - aq.max()) / (0.50 * spread)
            logits[~torch.as_tensor(allowed)] = float("-inf")
            return torch.softmax(logits, dim=-1).numpy()

    assert np.allclose(probabilities(1.0), probabilities(100.0), atol=1e-7)
    core.v16_value_target_model = _FixedQ(torch.ones(36))
    action, logp, diagnostics = core._v16_q_policy_decision(
        _state(core), mask, deterministic=False, training_progress=0.0
    )
    assert allowed[action]
    assert np.exp(logp.item()) == pytest.approx(1.0 / allowed.sum())
    assert diagnostics["v17_q_spread_allowed"] == pytest.approx(0.0)
    probs = probabilities(1.0)
    ratio = probs[allowed].max() / probs[allowed].min()
    assert ratio == pytest.approx(np.exp(1.0 / 0.50), rel=1e-6)


def test_prequential_recorded_before_fit(monkeypatch):
    core = _core(seed=17105)
    incumbent = _incumbent_action(core)
    monkeypatch.setattr(core, "_v16_incumbent_two_lever_action", lambda: incumbent)
    values = torch.linspace(-0.5, 0.5, 36)
    core.v16_value_target_model = _FixedQ(values)
    mask = core._v17_move_mask(incumbent)
    credits = np.zeros(36, dtype=np.float32)
    actions = np.flatnonzero(mask).tolist()
    core._nmcc_last = {
        "incumbent_action": incumbent,
        "selected_action": incumbent,
        "candidate_action_indices": actions,
        "fit_candidate_action_indices": actions,
        "component_credit_names": ["own_profit"],
        "component_credit_matrix": [[0.0] for _ in range(36)],
        "regime": [],
        "stage": "profit",
        "rival": "static",
        "day": 1,
        "audit": False,
        "horizon_days": 5,
    }
    core._append_v16_branch_store_row(
        observation_window=_state(core),
        counterfactual_values=credits,
        counterfactual_mask=mask,
    )
    row = core.v16_branch_store.state_dict()["rows"][-1]
    stored = copy.deepcopy(row["prequential_prediction"])
    expected = values.numpy() - values[incumbent].item()
    assert np.allclose(np.asarray(stored)[mask], expected[mask], atol=0.0)
    with torch.no_grad():
        core.v16_value_target_model.values.add_(10.0)
    assert row["prequential_prediction"] == stored
    assert mask[int(row["greedy_action"])]
