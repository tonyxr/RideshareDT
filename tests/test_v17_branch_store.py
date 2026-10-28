"""V17 persistent branch-store and deterministic resume contracts."""

from collections import Counter

import numpy as np
import pytest

from Core import Core
from v17_branch_store import BranchReplayStore


def _row(index: int, regime: str) -> dict:
    return {
        "row": index,
        "regime_label": regime,
        "observation_window": [float(index)],
    }


def test_ring_retains_last_rows_in_chronological_order():
    store = BranchReplayStore(3, seed=17)
    for index in range(5):
        store.add(_row(index, "calm"))
    assert len(store) == 3
    assert store.total_added == 5
    assert [row["row"] for row in store.rows()] == [2, 3, 4]


def test_regime_balanced_sampling_uses_equal_regime_mass():
    store = BranchReplayStore(20, seed=18)
    for index in range(18):
        store.add(_row(index, "calm"))
    for index in range(18, 20):
        store.add(_row(index, "shortage"))
    batch = store.sample(10, regime_balanced=True)
    counts = Counter(row["regime_label"] for row in batch)
    assert counts == {"calm": 5, "shortage": 5}


def test_saved_sampling_state_resumes_bit_identical_next_fit():
    store = BranchReplayStore(20, seed=19)
    for index in range(12):
        store.add(_row(index, "calm" if index % 3 else "shortage"))
    state = store.state_dict()
    expected = store.sample(8)

    resumed = BranchReplayStore(20, seed=999)
    resumed.load_state_dict(state)
    assert resumed.sample(8) == expected


def test_store_rejects_different_model_version():
    store = BranchReplayStore(4, seed=20, model_version="v17-value-v1")
    store.add(_row(1, "calm"))
    other = BranchReplayStore(4, seed=20, model_version="other")
    with pytest.raises(ValueError, match="model version"):
        other.load_state_dict(store.state_dict())


def _core(*, seed: int = 1721) -> Core:
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
    )


def test_core_appends_complete_branch_row():
    core = _core()
    action_dim = core.firm1.agent.action_dim
    core._nmcc_last = {
        "incumbent_action": 12,
        "selected_action": 13,
        "candidate_action_indices": [12, 13],
        "component_credit_names": ["own", "rival", "share", "rival_share"],
        "component_credit_matrix": [None] * 12
        + [[0.0, 0.0, 0.0, 0.0], [1.0, -1.0, 0.1, -0.1]]
        + [None] * (action_dim - 14),
        "regime": ["shortage"],
        "stage": "profit",
        "rival": "static",
        "day": 3,
        "tape_seed": 1721,
        "tape_id": 99,
        "audit": False,
        "horizon_days": 7,
    }
    values = np.full(action_dim, np.nan, dtype=np.float32)
    mask = np.zeros(action_dim, dtype=bool)
    values[[12, 13]] = [0.0, 1.0]
    mask[[12, 13]] = True
    core._append_v16_branch_store_row(
        observation_window=np.arange(6, dtype=np.float32).reshape(2, 3),
        counterfactual_values=values,
        counterfactual_mask=mask,
    )
    row = core.v16_branch_store.rows()[0]
    assert row["observation_shape"] == [2, 3]
    assert row["observation_window"] == pytest.approx(range(6))
    assert row["candidate_actions"] == [12, 13]
    assert row["regime_label"] == "shortage"
    assert row["model_version"] == "v17-value-v1"


def test_policy_artifact_restores_next_branch_batch(tmp_path):
    core = _core(seed=1722)
    for index in range(8):
        core.v16_branch_store.add(
            _row(index, "calm" if index % 2 else "shortage")
        )
    artifact_path = tmp_path / "policy.pt"
    core.save_trained_model(path=str(artifact_path), model_id="store-resume")
    expected = core.v16_branch_store.sample(6)

    resumed = _core(seed=1722)
    resumed.load_trained_model(str(artifact_path), load_optimizer=True)
    assert resumed.v16_branch_store.sample(6) == expected


def test_store_defaults_off_and_requires_one_day_credit():
    legacy = Core(
        "Chicago",
        seed=1723,
        firm1_mode="RL",
        firm2_mode="static",
        total_customers_pool=8,
        simulation_sample_cap=3,
        enable_driver_supply=False,
        openai_api_key="",
    )
    assert legacy.v16_store_cap == 0
    assert legacy.v16_branch_store is None
    with pytest.raises(ValueError, match="requires v16_branch_mode"):
        Core(
            "Chicago",
            seed=1724,
            firm1_mode="RL",
            firm2_mode="static",
            total_customers_pool=8,
            simulation_sample_cap=3,
            enable_driver_supply=False,
            openai_api_key="",
            v16_store_cap=20,
        )

