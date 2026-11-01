"""V17 one-day deviation, stencil, and audit branch contracts."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from Core import Core
from MarketInteraction import DayContext
from PPOAgent import PPOAgent
from pricing_models import TWO_LEVER_LEVELS


def _core(*, horizon: int = 3, audit_every: int = 4) -> Core:
    core = Core(
        "Chicago",
        seed=1709,
        firm1_mode="RL",
        firm2_mode="static",
        total_customers_pool=16,
        simulation_sample_cap=3,
        enable_driver_supply=False,
        openai_api_key="",
        action_contract="two_lever_v1",
        rivals_two_lever=True,
        v16_crn_tape=True,
        v16_branch_mode="one_day_deviation",
        v16_branch_horizon_days=horizon,
        v16_branch_design="stencil11",
        v16_audit_every=audit_every,
    )
    card_action = next(
        action
        for action, cell in core.firm1.action_to_two_lever_cell.items()
        if cell == (1.0, 1.0)
    )
    core.firm1.apply_action(card_action, core.market)
    core._refresh_profile_pool(rides_per_timestep=3)
    core.current_training_stage = SimpleNamespace(opponent_regime="profit")
    return core


def _action(core: Core, distance: float, time: float) -> int:
    return next(
        action
        for action, cell in core.firm1.action_to_two_lever_cell.items()
        if cell == (distance, time)
    )


def test_stencil_has_11_cells_and_audit_has_all_25():
    core = _core()
    assert core.nmcc_stages == ("market", "static", "catchup", "profit")
    assert core.nmcc_every_k_decisions == 1
    assert core.nmcc_advantage_scale == "none"
    core._set_v16_tape_position(0, 0)
    mask = core.firm1.feasible_action_mask(core.market)
    selected = _action(core, 1.5, 1.5)
    candidates, incumbent, audit = core._v16_branch_candidates(
        selected_action=selected,
        action_mask=mask,
        decision_count=1,
    )
    expected_stencil = {
        _action(core, distance, time)
        for distance in (0.9, 1.0, 1.2)
        for time in (0.9, 1.0, 1.2)
    }
    assert not audit
    assert incumbent == _action(core, 1.0, 1.0)
    assert len(candidates) == 11
    assert len(candidates) == len(set(candidates))
    assert expected_stencil <= set(candidates)
    assert selected in candidates

    candidates, incumbent, audit = core._v16_branch_candidates(
        selected_action=selected,
        action_mask=mask,
        decision_count=4,
    )
    assert audit
    assert incumbent == _action(core, 1.0, 1.0)
    assert candidates == list(range(25))


def test_edge_stencil_is_replenished_to_registered_11_cells():
    core = _core()
    core.firm1.apply_action(_action(core, 0.8, 0.8), core.market)
    core._set_v16_tape_position(0, 0)
    candidates, incumbent, audit = core._v16_branch_candidates(
        selected_action=_action(core, 1.5, 1.5),
        action_mask=core.firm1.feasible_action_mask(core.market),
        decision_count=1,
    )
    assert not audit
    assert incumbent == _action(core, 0.8, 0.8)
    assert len(candidates) == 11
    assert len(set(candidates)) == 11


def test_every_branch_restores_incumbent_from_day_one():
    core = _core(horizon=3)
    incumbent = _action(core, 1.0, 1.0)
    selected = _action(core, 1.5, 1.5)
    core.firm1.apply_action(incumbent, core.market)
    core._set_v16_tape_position(0, 0)
    profiles = core._sample_profiles_from_pool(
        3, generalization_fraction=0.0
    )
    values, mask = core._counterfactual_day_values(
        selected_action=selected,
        action_mask=core.firm1.feasible_action_mask(core.market),
        day_index=0,
        step_in_day=0,
        hours=[12],
        day_ctx=DayContext(day_of_week=2, weather="clear"),
        presampled=(3, profiles),
        customers_per_step=3,
        firm2_interval_steps=1,
        days_left=None,
    )
    incumbent_label = core.firm1.action_label(incumbent)
    assert int(mask.sum()) == 11
    assert values[incumbent] == pytest.approx(0.0, abs=1e-7)
    assert np.isfinite(values[selected])
    assert not core._nmcc_last["audit"]
    assert core._nmcc_last["branch_mode"] == "one_day_deviation"
    for candidate_label, posted in core._nmcc_last[
        "posted_actions_by_day"
    ].items():
        assert len(posted) == 3
        assert posted[0] == candidate_label
        assert posted[1:] == [incumbent_label, incumbent_label]
    assert core._v16_incumbent_two_lever_action() == incumbent


def test_t3_audit_uses_heldout_stencil_and_independent_second_tape():
    core = _core(horizon=1, audit_every=4)
    core.v17_probe_mode = True
    core.v17_probe_second_tape_fraction = 0.25
    core._nmcc_decision_counter = 4
    incumbent = _action(core, 1.0, 1.0)
    core.firm1.apply_action(incumbent, core.market)
    core._set_v16_tape_position(3, 0)
    profiles = core._sample_profiles_from_pool(3, generalization_fraction=0.0)
    values, mask = core._counterfactual_day_values(
        selected_action=incumbent,
        action_mask=core.firm1.feasible_action_mask(core.market),
        day_index=3,
        step_in_day=0,
        hours=[12],
        day_ctx=DayContext(day_of_week=2, weather="clear"),
        presampled=(3, profiles),
        customers_per_step=3,
        firm2_interval_steps=1,
        days_left=None,
    )
    assert int(mask.sum()) == 25
    assert np.isfinite(values).sum() == 25
    assert core._nmcc_last["audit"] is True
    assert core._nmcc_last["second_tape"] is True
    assert core._nmcc_last["second_tape_id"] == 1
    assert len(core._nmcc_last["fit_candidate_action_indices"]) == 11
    assert len(core._nmcc_last["second_tape_action_advantages"]) == 25
    assert np.isfinite(core._nmcc_last["label_noise_sd"])


def test_unscaled_counterfactual_advantage_keeps_reward_units():
    values = torch.tensor([[0.0, 2.0, -1.0]], dtype=torch.float32)
    masks = torch.ones_like(values, dtype=torch.bool)
    actions = torch.tensor([1])
    advantage, valid = PPOAgent._counterfactual_actor_advantages(
        values,
        masks,
        actions,
        baseline="provided",
        scale="none",
    )
    assert valid.tolist() == [True]
    assert advantage.tolist() == pytest.approx([2.0])


def test_flags_default_to_legacy_hold_h_path():
    core = Core(
        "Chicago",
        seed=1710,
        firm1_mode="RL",
        firm2_mode="static",
        total_customers_pool=8,
        simulation_sample_cap=3,
        enable_driver_supply=False,
        openai_api_key="",
    )
    assert core.v16_branch_mode == "hold_h"
    assert core.v16_branch_design == "all"
    assert core.v16_audit_every == 0
    assert core.counterfactual_day_credit == 0


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (
            {"v16_crn_tape": False},
            "requires v16_crn_tape",
        ),
        (
            {"action_contract": "legacy"},
            "requires action_contract",
        ),
        (
            {"v16_branch_design": "all"},
            "requires v16_branch_design",
        ),
    ],
)
def test_one_day_deviation_rejects_incomplete_contract(kwargs, match):
    config = dict(
        market_name="Chicago",
        seed=1711,
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
    )
    config.update(kwargs)
    with pytest.raises(ValueError, match=match):
        Core(**config)


def test_registered_levels_remain_the_fixed_five_cell_axes():
    assert TWO_LEVER_LEVELS == (0.8, 0.9, 1.0, 1.2, 1.5)


def test_nested_branch_rival_planner_is_local_but_live_planner_stays_full(
    monkeypatch,
):
    core = Core(
        "Chicago",
        seed=1712,
        firm1_mode="RL",
        firm2_mode="profit_opt_mpc_grid",
        total_customers_pool=16,
        simulation_sample_cap=3,
        enable_driver_supply=False,
        openai_api_key="",
        action_contract="two_lever_v1",
        rivals_two_lever=True,
        v16_crn_tape=True,
        v16_branch_mode="one_day_deviation",
        v16_branch_design="stencil11",
        branch_profit_planner_sample_cap=2,
        branch_profit_planner_horizon_days=1,
        branch_profit_planner_candidate_radius=1,
    )
    core._refresh_profile_pool(rides_per_timestep=3)
    core.shared_steps_per_day = 2
    core._profit_planner_reference_customers = 10
    calls = []

    def fake_simulate_batch(**kwargs):
        calls.append(core.simulation_sample_cap)
        metrics = SimpleNamespace(
            profit_per_request=core._tariff_grid_multiplier(core.firm2)
        )
        return None, None, metrics, None, None, None

    monkeypatch.setattr(core, "simulate_batch", fake_simulate_batch)
    full_grid_size = len(core.firm2.profit_candidate_grid)
    full_horizon = core.firm2.profit_horizon_days * core.shared_steps_per_day
    core._firm2_profit_best_response_plan(
        day_of_week=2, hour=12, weather="clear"
    )
    assert len(calls) == (full_grid_size + 1) * full_horizon
    assert set(calls) == {10}

    calls.clear()
    core._v16_branch_rollout_depth = 1
    core._firm2_profit_best_response_plan(
        day_of_week=2, hour=12, weather="clear"
    )
    assert len(calls) <= 4 * core.shared_steps_per_day
    assert set(calls) == {2}
    assert core.branch_profit_planner_calls == 1
    assert core.profit_planner_calls == 2
